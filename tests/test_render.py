"""The rasteriser is a measurement device, so it is tested like one.

The renderer's job is to produce images whose geometry is *exactly* what the
camera model says it should be.  Everything downstream -- calibration,
triangulation, pose -- is measured against ground truth that only means
anything if that holds.  These tests check it against the projection equations
directly rather than against a saved image.
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam.geometry import PinholeCamera, se3
from multicam.sim import mesh as M
from multicam.sim.render import NEAR, Rasteriser, _clip_near


@pytest.fixture(scope="module")
def flat_texture():
    return [np.full((16, 16), 0.7, dtype=np.float32)]


def _ideal_camera(width=160, height=120, f=150.0):
    return PinholeCamera(f, f, (width - 1) / 2.0, (height - 1) / 2.0, width, height,
                         np.zeros(5))


def test_a_box_lands_where_the_projection_says(flat_texture):
    """The silhouette of a box must match its own projected convex hull."""
    camera = _ideal_camera()
    box = M.box((0.2, 0.2, 0.2), (0.0, 0.0, 0.0), uv_scale=3.0)
    T = se3.look_at([0.0, 0.0, -1.2], [0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])

    ras = Rasteriser(flat_texture, ambient=1.0, diffuse=0.0)
    result = ras.render(camera, T, [box], supersample=1)

    uv = camera.project(se3.transform(T, box.vertices))
    v, u = np.nonzero(result.mask >= 0)
    assert len(u) > 400, "the box should cover a substantial patch"
    # Every rendered pixel is inside the projected vertices' bounding box, and
    # the extent matches to within a pixel of rasterisation rounding.
    assert u.min() >= np.floor(uv[:, 0].min()) - 1
    assert u.max() <= np.ceil(uv[:, 0].max()) + 1
    assert v.min() >= np.floor(uv[:, 1].min()) - 1
    assert v.max() <= np.ceil(uv[:, 1].max()) + 1
    assert abs(u.min() - uv[:, 0].min()) < 1.5
    assert abs(u.max() - uv[:, 0].max()) < 1.5


def test_depth_matches_the_analytic_plane(flat_texture):
    """A plane at a known distance must render at exactly that depth."""
    camera = _ideal_camera()
    plane = M.plane((2.0, 2.0), (0.0, 0.0, 0.0), normal_axis=2, uv_scale=2.0, subdivisions=2)
    # Camera on the -z axis looking at the plane, which is the z = 0 plane.
    T = se3.look_at([0.0, 0.0, -0.8], [0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    ras = Rasteriser(flat_texture, ambient=1.0, diffuse=0.0)
    result = ras.render(camera, T, [plane], supersample=1)

    assert result.valid.any()
    v, u = np.nonzero(result.valid)
    # Every visible point is on z = 0, so its camera-frame depth is determined.
    xy = camera.undistort(np.stack([u, v], axis=1).astype(float))
    rays = np.concatenate([xy, np.ones((len(xy), 1))], axis=1)
    origin = se3.camera_centre(T)
    R = se3.inverse(T)[:3, :3]
    dirs = rays @ R.T
    t = -origin[2] / dirs[:, 2]
    expected_depth = t  # because the ray direction has unit z in the camera frame
    assert np.allclose(result.depth[v, u], expected_depth, atol=2e-4)


def test_occlusion_uses_the_z_buffer(flat_texture):
    """A near box must hide a far one, and the depth must be the near one's."""
    camera = _ideal_camera()
    near = M.box((0.3, 0.3, 0.02), (0.0, 0.0, -0.3), uv_scale=3.0)
    far = M.box((0.6, 0.6, 0.02), (0.0, 0.0, 0.3), uv_scale=3.0)
    T = se3.look_at([0.0, 0.0, -1.5], [0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    ras = Rasteriser(flat_texture, ambient=1.0, diffuse=0.0)
    result = ras.render(camera, T, [near, far], supersample=1)

    centre = (result.mask.shape[0] // 2, result.mask.shape[1] // 2)
    assert result.mask[centre] == 0, "the near box owns the centre pixel"
    assert result.depth[centre] < 1.3
    assert (result.mask == 1).any(), "the far box must still be visible around it"


def test_distortion_is_applied_in_the_right_direction():
    """Render a known 3D point and find it where the distorted model puts it.

    This is the test that would have caught the inverted distortion map.  A
    render warped the wrong way still produces a plausible image, and only a
    comparison against the projection equations shows it up.
    """
    camera = PinholeCamera(180.0, 180.0, 99.5, 79.5, 200, 160,
                           np.array([-0.25, 0.08, 0.0, 0.0, 0.0]))
    # A small bright square, near the corner of the frame, where the radial
    # term is large enough that getting its sign wrong is unmissable.
    marker = M.box((0.02, 0.02, 0.005), (0.28, 0.20, 0.0), uv_scale=1.0)
    T = se3.look_at([0.0, 0.0, -0.6], [0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])

    ras = Rasteriser([np.ones((8, 8), dtype=np.float32)], ambient=1.0, diffuse=0.0)
    result = ras.render(camera, T, [marker], supersample=2)

    v, u = np.nonzero(result.mask == 0)
    assert len(u) > 8, "the marker must be visible"
    observed = np.array([u.mean(), v.mean()])
    expected = camera.project(se3.transform(T, marker.centre))

    # An ideal-pinhole render, or one distorted the wrong way, lands several
    # pixels away here; the tolerance is well inside that gap.
    assert np.linalg.norm(observed - expected) < 1.2, (observed, expected)

    pinhole = camera.with_params(np.concatenate([camera.params[:4], np.zeros(5)]))
    undistorted_expected = pinhole.project(se3.transform(T, marker.centre))
    # The test is only meaningful if distortion actually moves the marker.  An
    # ideal-pinhole render, or one warped the wrong way, misses by at least
    # this much, which is several times the tolerance above.
    assert np.linalg.norm(undistorted_expected - expected) > 4.0, (
        "distortion barely moves this marker, so the test would pass on a "
        "renderer that ignored distortion; move the marker further off axis"
    )


def test_near_plane_clipping_splits_a_straddling_triangle():
    verts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pieces = list(_clip_near(verts, uv))
    assert len(pieces) == 2, "one vertex behind the near plane gives a quad, so two triangles"
    for v3, _ in pieces:
        assert np.all(v3[:, 2] >= NEAR - 1e-9)


def test_fully_behind_triangle_is_dropped():
    verts = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -2.0], [0.0, 1.0, -1.5]])
    uv = np.zeros((3, 2))
    assert list(_clip_near(verts, uv)) == []


def test_supersampling_changes_the_image_but_not_the_geometry(flat_texture):
    camera = _ideal_camera()
    box = M.box((0.2, 0.2, 0.2), uv_scale=3.0)
    T = se3.look_at([0.4, 0.3, -1.0], [0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    ras = Rasteriser(flat_texture, ambient=1.0, diffuse=0.0)
    a = ras.render(camera, T, [box], supersample=1)
    b = ras.render(camera, T, [box], supersample=3)

    # Same silhouette to within the antialiased boundary.
    overlap = (a.mask >= 0) & (b.mask >= 0)
    union = (a.mask >= 0) | (b.mask >= 0)
    assert overlap.sum() / max(union.sum(), 1) > 0.95


def test_render_is_deterministic(flat_texture):
    camera = _ideal_camera()
    box = M.box((0.2, 0.2, 0.2), uv_scale=3.0)
    T = se3.look_at([0.4, 0.3, -1.0], [0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    ras = Rasteriser(flat_texture)
    a = ras.render(camera, T, [box])
    b = ras.render(camera, T, [box])
    assert np.array_equal(a.image, b.image)
    assert np.array_equal(a.mask, b.mask)


def test_noise_needs_a_generator(flat_texture):
    camera = _ideal_camera()
    box = M.box((0.2, 0.2, 0.2), uv_scale=3.0)
    T = se3.look_at([0.0, 0.0, -1.0], [0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    ras = Rasteriser(flat_texture)
    with pytest.raises(ValueError):
        ras.render(camera, T, [box], noise_sigma=0.01)


def test_mesh_surface_sampling_is_area_weighted(rng):
    """A box's faces should be sampled in proportion to their areas."""
    box = M.box((0.4, 0.2, 0.1), uv_scale=1.0)
    pts, _ = box.sample_surface(40000, rng)
    lo, hi = box.bounds
    assert np.all(pts >= lo - 1e-9) and np.all(pts <= hi + 1e-9)
    # The +-z faces are 0.4 x 0.2 = 0.08 each; total area is
    # 2*(0.08 + 0.04 + 0.02) = 0.28, so they should take 8/28 of the samples.
    on_z = np.isclose(np.abs(pts[:, 2]), 0.05, atol=1e-6)
    assert abs(on_z.mean() - 2 * 0.08 / 0.28) < 0.02


def test_ply_round_trip(tmp_path):
    box = M.box((0.1, 0.2, 0.3), uv_scale=1.0)
    path = tmp_path / "box.ply"
    box.save_ply(path)
    back = M.TriMesh.load_ply(path)
    assert np.allclose(back.vertices, box.vertices, atol=1e-6)
    assert np.array_equal(back.faces, box.faces)


def test_bracket_is_asymmetric_enough_for_add_to_mean_something():
    """ADD is only meaningful on a part that is not symmetric.

    A 180-degree rotation about each principal axis is applied and the mean
    distance to the *nearest* surface point measured -- which is exactly what
    ADD-S would forgive.  Measured on this part it is 13 to 15 mm, which is 7
    to 8 per cent of the 187 mm diameter and four orders of magnitude above
    the sub-millimetre errors the pose evaluation reports, so ADD and ADD-S
    genuinely measure different things here.

    The part is *not* wildly asymmetric, and 5 per cent of the diameter is
    therefore the honest bar rather than a comfortable one.  Both metrics are
    reported everywhere for that reason.
    """
    from scipy.spatial import cKDTree

    from multicam.geometry import so3

    part = M.bracket()
    pts, _ = part.sample_surface(6000, np.random.default_rng(0))
    tree = cKDTree(pts)
    for axis in np.eye(3):
        R = so3.exp(axis * np.pi)
        d, _ = tree.query(pts @ R.T, k=1)
        assert np.mean(d) > 0.05 * part.diameter, (
            "a 180-degree rotation about %s moves the part by only %.1f mm"
            % (axis, np.mean(d) * 1000)
        )
