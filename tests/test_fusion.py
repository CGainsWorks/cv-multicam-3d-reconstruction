"""Dense reconstruction: the ray-box sweep bounds, ZNCC, and a real sweep.

The plane sweep is tested against a *rendered* scene with known depth rather
than against a synthetic cost volume, because the failure modes that matter --
a warp that is off by a pixel, a cost that is computed on the wrong window, a
sub-pixel refinement that moves the answer the wrong way -- all survive a test
on fabricated costs and none of them survive a comparison with the renderer's
own depth buffer.
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam.fusion.stereo import _ray_box_range, _zncc, cross_check, sweep
from multicam.geometry import PinholeCamera, se3
from multicam.sim import mesh as M
from multicam.sim.render import Rasteriser


def _camera(width=120, height=96, f=130.0):
    return PinholeCamera(f, f, (width - 1) / 2.0, (height - 1) / 2.0, width, height,
                         np.zeros(5))


# --------------------------------------------------------------------------
# Sweep bounds
# --------------------------------------------------------------------------


def test_ray_box_range_brackets_the_box():
    camera = _camera()
    # Camera a metre back on -z, box of half-size 0.1 at the origin.
    T = se3.look_at([0.0, 0.0, -1.0], [0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    lo, hi = np.full(3, -0.1), np.full(3, 0.1)
    near, far, hit = _ray_box_range(camera, T, lo, hi, (camera.height, camera.width))

    centre = (camera.height // 2, camera.width // 2)
    assert hit[centre]
    # The central ray enters at z = -0.1 and leaves at z = +0.1 in world, which
    # is 0.9 m and 1.1 m from the camera.
    assert np.isclose(near[centre], 0.9, atol=0.02)
    assert np.isclose(far[centre], 1.1, atol=0.02)
    # A corner ray misses the box entirely.
    assert not hit[0, 0]
    assert (~hit).any() and hit.any(), "the box should cover part of the frame"


def test_ray_box_range_handles_a_ray_parallel_to_a_slab():
    camera = _camera()
    T = se3.look_at([0.0, 0.0, -1.0], [0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    # A slab that is infinite in x and y, so every ray is "parallel" to those.
    lo = np.array([-1e6, -1e6, -0.05])
    hi = np.array([1e6, 1e6, 0.05])
    near, far, hit = _ray_box_range(camera, T, lo, hi, (camera.height, camera.width))
    assert hit.all()
    assert np.all(far[hit] > near[hit])


def test_ray_box_range_misses_a_box_behind_the_camera():
    camera = _camera()
    T = se3.look_at([0.0, 0.0, -1.0], [0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    lo, hi = np.array([-0.1, -0.1, -3.1]), np.array([0.1, 0.1, -2.9])
    _, _, hit = _ray_box_range(camera, T, lo, hi, (camera.height, camera.width))
    assert not hit.any()


# --------------------------------------------------------------------------
# ZNCC
# --------------------------------------------------------------------------


def test_zncc_is_one_on_identical_textured_images(rng):
    a = rng.random((40, 40)).astype(np.float32)
    assert np.allclose(_zncc(a, a, 7)[5:-5, 5:-5], 1.0, atol=1e-4)


def test_zncc_is_invariant_to_gain_and_offset(rng):
    a = rng.random((40, 40)).astype(np.float32)
    b = (a * 0.6 + 0.2).astype(np.float32)
    assert np.allclose(_zncc(a, b, 7)[5:-5, 5:-5], 1.0, atol=1e-4)


def test_zncc_is_minus_one_on_an_inverted_image(rng):
    a = rng.random((40, 40)).astype(np.float32)
    assert np.allclose(_zncc(a, (1.0 - a).astype(np.float32), 7)[5:-5, 5:-5], -1.0,
                       atol=1e-4)


def test_zncc_is_zero_on_a_flat_window():
    flat = np.full((30, 30), 0.5, dtype=np.float32)
    rng = np.random.default_rng(0)
    other = rng.random((30, 30)).astype(np.float32)
    # No variance in one image means no correlation to measure, and the value
    # must be zero rather than a division by zero.
    out = _zncc(flat, other, 5)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)


def test_zncc_is_near_zero_on_independent_noise(rng):
    a = rng.random((80, 80)).astype(np.float32)
    b = rng.random((80, 80)).astype(np.float32)
    assert abs(float(_zncc(a, b, 9)[10:-10, 10:-10].mean())) < 0.15


# --------------------------------------------------------------------------
# A real sweep
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def swept():
    """Render a textured box from three views and sweep the middle one."""
    from multicam.sim.textures import surface_texture

    rng = np.random.default_rng(3)
    texture = surface_texture(rng, 512, contrast=0.9, brightness=0.5, grain=0.03, beta=1.2)
    box = M.box((0.12, 0.12, 0.12), (0.0, 0.0, 0.0), uv_scale=6.0)
    ras = Rasteriser([texture], ambient=0.75, diffuse=0.25)

    camera = _camera(160, 128, 180.0)
    eyes = [[-0.22, -0.10, -0.55], [0.0, -0.10, -0.58], [0.22, -0.10, -0.55]]
    views = []
    for eye in eyes:
        T = se3.look_at(eye, [0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
        r = ras.render(camera, T, [box], supersample=2)
        views.append((r.image, camera, T, r.depth))

    lo, hi = box.bounds
    reference = (views[1][0], camera, views[1][2], 0, 1)
    sources = [(views[0][0], camera, views[0][2]), (views[2][0], camera, views[2][2])]
    depth_map = sweep(reference, sources, (lo - 0.01, hi + 0.01),
                      n_depths=128, window=7, min_score=0.5, scale=1.0)
    return depth_map, views[1][3], box


def test_sweep_recovers_the_rendered_depth(swept):
    depth_map, truth_depth, _ = swept
    assert depth_map.valid.sum() > 500, depth_map.stats

    m = depth_map.valid & np.isfinite(truth_depth)
    assert m.sum() > 400
    error = np.abs(depth_map.depth[m] - truth_depth[m])
    assert np.median(error) < 1.5e-3, (np.median(error), depth_map.stats)
    assert np.mean(error < 3e-3) > 0.85


def test_sweep_only_claims_depth_inside_the_bounds(swept):
    depth_map, truth_depth, box = swept
    points, _ = depth_map.points()
    assert len(points)
    lo, hi = box.bounds
    inside = np.all((points >= lo - 0.02) & (points <= hi + 0.02), axis=1)
    assert inside.mean() > 0.98


def test_sweep_rejects_everything_when_the_box_is_elsewhere():
    camera = _camera()
    T = se3.look_at([0.0, 0.0, -0.6], [0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])
    image = np.zeros((camera.height, camera.width), dtype=np.uint8)
    far = (np.array([5.0, 5.0, 5.0]), np.array([5.1, 5.1, 5.1]))
    out = sweep((image, camera, T, 0, 0), [(image, camera, T)], far, n_depths=8)
    assert out.valid.sum() == 0
    assert "reason" in out.stats


def test_cross_check_removes_a_view_nobody_agrees_with(swept):
    depth_map, _, _ = swept
    # A deliberately wrong copy: every depth shifted by two centimetres.
    bogus = type(depth_map)(
        depth_map.station, 99, (depth_map.depth + 0.02).astype(np.float32),
        depth_map.score, depth_map.valid.copy(), depth_map.camera_model,
        depth_map.T_cam_object, {},
    )
    checked = cross_check([depth_map, bogus], tolerance_m=0.0015, min_agree=1)
    kept_good = checked[0].valid.sum()
    kept_bad = checked[1].valid.sum()
    assert kept_bad < 0.2 * depth_map.valid.sum(), (kept_bad, depth_map.valid.sum())
    assert kept_good < depth_map.valid.sum() or kept_good == 0


@pytest.mark.slow
def test_fusion_and_meshing_produce_a_surface(swept):
    """Open3D fusion end to end: cloud, normals, Poisson mesh."""
    pytest.importorskip("open3d")
    from multicam.fusion import fuse_depth_maps, poisson_mesh

    depth_map, _, box = swept
    fused = fuse_depth_maps([depth_map], voxel_m=0.0012)
    assert len(fused.points) > 100
    assert fused.normals is not None
    assert np.allclose(np.linalg.norm(fused.normals, axis=1), 1.0, atol=1e-6)
    assert fused.stats["after_voxel"] <= fused.stats["raw_points"]

    meshed = poisson_mesh(fused, depth=7, density_quantile=0.2)
    # A single view of a box cannot close a surface, so a mesh is welcome but
    # not required; what must hold is that nothing invented geometry outside
    # the data.
    if meshed.has_mesh:
        lo, hi = box.bounds
        assert np.all(meshed.mesh_vertices >= lo - 0.03)
        assert np.all(meshed.mesh_vertices <= hi + 0.03)
