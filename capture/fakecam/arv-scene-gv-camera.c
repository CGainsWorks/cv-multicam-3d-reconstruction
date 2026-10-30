/*
 * arv-scene-gv-camera: Aravis's fake GigE Vision camera, showing a rendered scene.
 *
 * SIMULATED.  This is not a camera and nothing it produces is a measurement.
 *
 * Aravis ships a GigE Vision camera simulator, ArvGvFakeCamera, wrapped by its
 * `arv-fake-gv-camera` tool.  It speaks real GVCP and GVSP on a UDP socket,
 * exposes a GenICam XML description, honours AcquisitionStart, TriggerMode and
 * TriggerSoftware and stamps every frame with a timestamp.  What it puts in
 * the frame is a diagonal ramp.
 *
 * This program is the same ArvGvFakeCamera object with one thing changed: the
 * fill-pattern callback, which Aravis exposes for exactly this purpose through
 * arv_fake_camera_set_fill_pattern().  Instead of the ramp it serves a frame
 * rendered by the repository's simulator, so that what arrives over GigE
 * Vision can be fed into the reconstruction pipeline and checked against the
 * ground truth it was rendered from.  Everything on the wire and every
 * GenICam feature is Aravis's and is untouched.
 *
 * Which frame is served is chosen by the fake device's `TestRegister`, a
 * spare 32-bit register in Aravis's fake-camera GenICam description.  The
 * capture code writes the station index into it over GVCP before each
 * trigger, standing in for a turntable moving the part in front of the
 * cameras.  The frame for index N of a camera is `<dir>/NNNNNN.pgm`, a binary
 * 8-bit PGM of exactly the camera's Width x Height.  An index with no file
 * produces an empty frame and a warning, never a silent substitute.
 *
 * One private header, arvbufferprivate.h, is included because the size of a
 * filled buffer is private in Aravis 0.8 and Aravis's own fill callback sets
 * it directly.  setup_aravis.sh compiles this file against the source tree of
 * the pinned release only, so the structure layout always matches the library.
 *
 * Usage: arv-scene-gv-camera -i 127.0.0.2 -s SIM1 -d frames/cam1
 */

#include <arv.h>
#include <arvbufferprivate.h>

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	ArvFakeCamera *camera;
	char *directory;
	guint8 *pixels;
	gsize capacity;
} SceneSource;

static volatile sig_atomic_t cancel = 0;

static void
set_cancel (int signal_number)
{
	(void) signal_number;
	cancel = 1;
}

/* Parse a binary PGM (P5, maxval 255).  Returns a pointer to the pixels inside
 * contents, or NULL.  Comment lines in the header are allowed. */
static const guint8 *
parse_pgm (const char *contents, gsize length, guint *width, guint *height)
{
	const char *p = contents;
	const char *end = contents + length;
	unsigned long values[3];
	int n = 0;

	if (length < 2 || p[0] != 'P' || p[1] != '5')
		return NULL;
	p += 2;

	while (n < 3 && p < end) {
		char *next;

		while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n'))
			p++;
		if (p < end && *p == '#') {
			while (p < end && *p != '\n')
				p++;
			continue;
		}
		if (p >= end)
			return NULL;
		values[n] = strtoul (p, &next, 10);
		if (next == p)
			return NULL;
		p = next;
		n++;
	}
	/* Exactly one whitespace byte separates the header from the raster. */
	if (n != 3 || p >= end || values[2] != 255)
		return NULL;
	p++;

	*width = (guint) values[0];
	*height = (guint) values[1];
	if ((gsize) (end - p) < (gsize) values[0] * values[1])
		return NULL;
	return (const guint8 *) p;
}

static void
fill_from_scene (ArvBuffer *buffer, void *data, guint32 exposure_time_us,
		 guint32 gain, ArvPixelFormat pixel_format)
{
	SceneSource *source = data;
	guint32 index = 0;
	guint width, height, file_width = 0, file_height = 0;
	gchar *path, *contents = NULL;
	gsize length = 0;
	const guint8 *raster;

	(void) exposure_time_us;
	(void) gain;

	buffer->priv->received_size = 0;

	if (pixel_format != ARV_PIXEL_FORMAT_MONO_8) {
		g_warning ("scene camera serves Mono8 only");
		return;
	}

	arv_fake_camera_read_register (source->camera, ARV_FAKE_CAMERA_REGISTER_TEST, &index);
	width = buffer->priv->parts[0].width;
	height = buffer->priv->parts[0].height;

	path = g_strdup_printf ("%s/%06u.pgm", source->directory, index);
	if (!g_file_get_contents (path, &contents, &length, NULL)) {
		g_warning ("no frame for index %u (%s)", index, path);
		g_free (path);
		return;
	}

	raster = parse_pgm (contents, length, &file_width, &file_height);
	if (raster == NULL || file_width != width || file_height != height ||
	    (gsize) width * height > buffer->priv->allocated_size) {
		g_warning ("%s is not an 8-bit PGM of %ux%u", path, width, height);
	} else {
		memcpy (buffer->priv->data, raster, (gsize) width * height);
		buffer->priv->received_size = (gsize) width * height;
	}

	g_free (contents);
	g_free (path);
}

int
main (int argc, char **argv)
{
	static char *interface_name = NULL;
	static char *serial_number = NULL;
	static char *directory = NULL;
	static const GOptionEntry entries[] = {
		{ "interface", 'i', 0, G_OPTION_ARG_STRING, &interface_name,
		  "Listening interface name or address", "interface" },
		{ "serial", 's', 0, G_OPTION_ARG_STRING, &serial_number,
		  "Fake camera serial number", "serial" },
		{ "frames", 'd', 0, G_OPTION_ARG_STRING, &directory,
		  "Directory of NNNNNN.pgm frames, indexed by TestRegister", "dir" },
		{ NULL }
	};
	GOptionContext *context;
	GError *error = NULL;
	ArvGvFakeCamera *gv_camera;
	SceneSource source = { 0 };

	context = g_option_context_new (NULL);
	g_option_context_set_summary (context,
		"Aravis fake GigE Vision camera serving rendered frames.  Simulated.");
	g_option_context_add_main_entries (context, entries, NULL);
	if (!g_option_context_parse (context, &argc, &argv, &error)) {
		fprintf (stderr, "option parsing failed: %s\n", error->message);
		g_error_free (error);
		g_option_context_free (context);
		return EXIT_FAILURE;
	}
	g_option_context_free (context);

	if (directory == NULL) {
		fprintf (stderr, "--frames is required\n");
		return EXIT_FAILURE;
	}

	gv_camera = arv_gv_fake_camera_new (interface_name, serial_number);
	if (!arv_gv_fake_camera_is_running (gv_camera)) {
		fprintf (stderr, "failed to start the fake camera on %s\n",
			 interface_name != NULL ? interface_name : "the default interface");
		g_object_unref (gv_camera);
		return EXIT_FAILURE;
	}

	source.camera = arv_gv_fake_camera_get_fake_camera (gv_camera);
	source.directory = directory;
	arv_fake_camera_set_fill_pattern (source.camera, fill_from_scene, &source);

	signal (SIGINT, set_cancel);
	signal (SIGTERM, set_cancel);

	printf ("scene camera %s listening on %s\n",
		serial_number != NULL ? serial_number : "GV01",
		interface_name != NULL ? interface_name : "the default interface");
	fflush (stdout);

	while (!cancel)
		g_usleep (100000);

	g_object_unref (gv_camera);
	return EXIT_SUCCESS;
}
