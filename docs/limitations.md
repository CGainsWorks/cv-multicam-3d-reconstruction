# Limitations

What this repository does not show, could not show, and gets wrong. Read this
before quoting any number from it.

---

## 1. There is no hardware. All of it is synthetic.

This is the limitation that contains most of the others.

Every image comes from the software rasteriser in
[`src/multicam/sim`](../src/multicam/sim). No camera, no lens, no turntable, no
part. The renderer's ground truth is exact, which is the only reason errors can
be quoted to a hundredth of a millimetre; it is also why those errors are
smaller than a real cell would produce.

**What that buys.** The calibration can be scored against the rig it was
trying to recover, not just against its own residual — which matters, because a
calibration with a beautiful residual and a wrong answer is the common failure
and the residual never shows it. Every point can be labelled part or fixture.
Depth can be checked per pixel. None of that exists on real hardware.

**What it costs**, in roughly descending order of how much it flatters the
result:

| | Here | A real cell |
| --- | --- | --- |
| **Surface appearance** | Broadband procedural texture with energy down to the pixel, on every face. | Machined aluminium is often shiny and nearly featureless, and its specular highlights move with the viewpoint rather than staying on the surface. Feature density would be far lower and matching much harder. **This is the single most flattering assumption here.** |
| **Illumination** | One directional light, flat-shaded per triangle, constant across the sweep. | Shadows, interreflections, mains flicker, and a part whose brightness changes as it turns. |
| **Lens model** | Exactly the five-parameter radial-tangential model the calibration fits. | Approximately that model. The residual between a real lens and any parametric model is what a real calibration's error floor is made of, and here that residual is zero by construction. |
| **Synchronisation** | Exact. The four exposures are the same instant. | The hardest thing to get right in a multi-camera cell, and the thing that silently destroys a reconstruction when it is wrong. See below. |
| **Geometry** | A procedurally generated mesh, closed, clean and exactly the thing being measured. | A scanned or exported mesh with holes, self-intersections, and a tessellation that is not the part. |
| **Sensor** | Gaussian read noise, nothing else. | Fixed-pattern noise, hot pixels, rolling shutter, compression. |

A reconstruction accurate to a third of a millimetre on this data would not be
on a shiny part under real lighting. When judging whether any of this transfers,
look at the **feature count per square millimetre of surface**, not at the final
error.

## 2. The capture code has been run against simulated cameras only

The GenICam / GigE Vision capture path
([`multicam.capture`](../src/multicam/capture), command line in
[`capture/capture_sync.py`](../capture/capture_sync.py)) is implemented on
Aravis 0.8.36 and runs in CI against three of Aravis's simulated GigE Vision
cameras on the loopback interface: feature access over GVCP,
`AcquisitionStart`, a software trigger per frame set, GVSP streaming, device
timestamps plus the timestamp check that drops and re-triggers a frame set
whose exposures are too far apart. `scripts/genicam_demo.py` serves a rendered
capture from those cameras and runs the calibration, reconstruction and pose
pipeline on what arrives.

That is a test of the software path, not of a rig. **It has never touched a
real camera.** What the simulation cannot show:

- **The hardware trigger.** The default `TriggerSource` is `Line1`, a wired
  trigger input; the simulator has no trigger lines and every simulated frame
  set is software triggered. Whether a real trigger wiring delivers
  simultaneous exposures is untested.
- **Clock agreement.** The simulated cameras all stamp frames from one host
  clock. Real cameras need PTP (IEEE 1588) for their timestamps to be
  comparable; the backend switches it on where the camera offers it and
  records whether it could; no PTP-capable camera has been tried.
- **The spreads.** The synchronisation spreads measured in CI are a property of
  the simulator (it exposes on a 25 Hz grid of the host clock) and say nothing
  about real cameras, which is why the simulated runs use a 5 ms tolerance and
  not the 0.2 ms one meant for a hardware trigger.
- **Real networks.** Loopback has no packet loss to speak of, no switch and no
  bandwidth limit shared by four cameras. Packet sizes, receive buffers and
  inter-packet delay would all need tuning on a real network.
- **The turntable.** There is no motion control; the simulated turntable only
  tells the simulated cameras which rendered frame to serve.

The OpenCV backend can be run and warns, every time, that it is software
triggered and therefore not synchronised.

**Everything in this repository assumes the four exposures are simultaneous.**
On a part moving at 100 mm/s, a 10 ms spread between the first camera and the
last is a millimetre of motion — three times the reconstruction error reported
here, and it would appear as a plausible-looking reconstruction that is simply
wrong.

## 3. The part is only weakly asymmetric, and that matters for the pose numbers

The bracket is 187 mm across. A 180-degree rotation about any of its principal
axes moves its surface by 13 to 15 mm, which is 7 to 8 per cent of its
diameter. That is far above the errors reported here, so ADD and ADD-S do
measure different things — but it is not a comfortable margin, and
**both metrics are reported everywhere** for that reason. A part with a real
symmetry would make ADD meaningless and ADD-S the only honest metric, and the
whole pose evaluation would have to be rewritten around that.

There is a test that measures this
([`tests/test_render.py`](../tests/test_render.py)) so the claim cannot quietly
stop being true.

### 3a. One near-symmetry was an artefact, and it cost six frames

Worth recording because it is the kind of bug this sort of project produces.

The triplanar texture projection originally gave every face the same function
of its world coordinates, so a face with normal along `z` at `(x, y)` and a
face with normal along `y` at `(x, z = y)` sampled the same texel. That gave the
L-bracket an exact **appearance** symmetry under a 90-degree rotation about `x`
— one its geometry does not have at all.

The pose estimator landed on the wrong limb on six of thirty-nine held-out
frames. Every one of those failures was within half a degree of exactly 90, and
every one had a healthy inlier count and a 0.4 px reprojection residual.
Nothing in the output looked wrong; the median error looked excellent and the
90th percentile was 90 degrees.

It is fixed — each of the six face orientations now maps into its own region of
the texture — and the invariant is asserted by a test. The reason it is written
down here rather than quietly corrected is that the failure mode is general:
**a synthetic dataset can give an object symmetries that the real object does
not have**, and the symptom is a bimodal error distribution with a flawless
median.

## 4. The CAD model is procedurally generated

"Evaluation against CAD ground truth" here means evaluation against a mesh this
repository generated from boxes, cylinders and prisms. It is a fair test of the
*geometry* of the pipeline. It is not a test of how the pipeline copes with a
real CAD export: no non-manifold edges, no degenerate triangles, no units
confusion, no mesh that disagrees with the part it describes.

## 5. The reconstruction only sees the part from one side

The turntable sweep rotates the part about a single axis, and the cameras sit
on a 76-degree arc above it. The underside and the steeply undercut faces are
never observed by anything. Consequences:

- **Completeness is capped** by what the sweep can see, and the completeness
  figures in [results.md](results.md) should be read as "of the whole CAD
  surface", including the parts no camera could ever reach.
- **The Poisson mesh invents the rest.** Screened Poisson reconstruction always
  returns a closed watertight surface, so the underside of the part in the
  meshed output is fiction. The low-density vertices are trimmed for exactly
  this reason, and the trim fraction is a parameter rather than a secret.
- **The appearance model inherits the gap.** The pose estimator matches against
  points reconstructed from that sweep, so a test pose presenting an unscanned
  face has fewer correspondences to work with. A real workflow would flip the
  part and scan it again; this one does not.

## 6. The dense stage is coarser than the sparse stage

The plane sweep runs at half resolution with a bounded number of depth
hypotheses, and its depth resolution is set by how finely the sweep volume is
sampled rather than by the image noise. The dense cloud is therefore *less*
accurate than the sparse points, which is the opposite of what people usually
expect from a dense stage, and it is a consequence of the sampling, not of the
matching. More hypotheses and full resolution would narrow it, at a cost
proportional to both.

## 7. Covariances are first order and optimistic

Every covariance reported — the pose covariance, the calibration covariance —
is the inverse of the Gauss-Newton information matrix at the solution. That is
the Cramer-Rao bound *given that the model is right and the residuals are
Gaussian*. Neither is exactly true, the residuals have a tail, and the numbers
should be read as a lower bound on the uncertainty rather than an estimate of
it. They are labelled as first order wherever they appear.

## 8. The evaluation uses ground-truth segmentation, deliberately

A turntable scan reconstructs the part **and the fixture it is bolted to**, and
there is no signal in the images that separates them — both are rigid and both
turn. Scoring the whole cloud against a CAD model of the part alone would
report a median error of forty millimetres and blame the reconstruction for the
turntable.

So the reconstructed points are labelled using the renderer's own segmentation
masks and the accuracy is reported per class. This is a use of ground truth. It
is confined to `multicam.evaluation`, it is labelled at every point where a
number depends on it, and a test parses every pipeline module and fails if one
so much as names a ground-truth accessor.

## 9. One seed, one rig, one part

Every number comes from a single capture with a single seed, a single rig
geometry and a single object. There is no spread over seeds, no second part, no
sensitivity analysis over the rig's geometry. The reconstruction has no
stochastic component once the data is fixed — RANSAC is seeded — so re-running
gives identical numbers, which means the quoted figures have no error bars at
all rather than tight ones.

## 10. What was not attempted

- **Online or real-time operation.** Nothing here is optimised for latency. The
  timings in [results.md](results.md) are wall clock on one machine and are
  reported because they are honest, not because they are good.
- **Multiple objects, or clutter.** One part, in isolation, on a turntable.
- **A learned front end.** The features are SIFT. A learned detector and
  descriptor would very likely do better on a low-texture part, which is
  precisely the case this dataset does not contain.
- **Rolling shutter, motion blur, or a moving part.** Everything is static at
  the moment of exposure.
- **Automatic detection of the part in the frame.** The pose estimator is given
  a model and matches against the whole image; there is no detector in front of
  it, and on a scene containing two of the same part it would have nothing to
  say about which one it found.
