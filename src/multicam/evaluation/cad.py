"""Scoring a reconstruction against the CAD model.

The reconstruction comes out in an arbitrary frame -- the rig frame at the
reference station -- so before anything can be measured it has to be put into
the CAD model's frame.  That registration is done explicitly, by ICP, and the
amount it had to move the initial guess is **reported**, not absorbed.  A
registration step that is allowed to move freely will make almost any point
cloud look accurate; one that reports it moved 8 mm to get there has told you
the reconstruction was 8 mm out.

Distances are exact point-to-surface distances, computed with Open3D's
raycasting scene rather than a nearest-vertex lookup.  On a mesh whose
triangles are several millimetres across, nearest-vertex overstates the error
by roughly a third of the triangle size, which is the same order as the error
being measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import se3


def to_open3d_mesh(mesh):
    """Convert a :class:`multicam.sim.mesh.TriMesh` to an Open3D mesh."""
    import open3d as o3d

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    m.compute_vertex_normals()
    return m


class SurfaceDistance:
    """Exact unsigned distance from arbitrary points to a triangle mesh."""

    def __init__(self, mesh):
        import open3d as o3d

        self._o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(to_open3d_mesh(mesh))
        )

    def __call__(self, points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if not len(points):
            return np.zeros(0)
        q = self._o3d.core.Tensor(points, dtype=self._o3d.core.Dtype.Float32)
        return self.scene.compute_distance(q).numpy().astype(np.float64)


@dataclass
class Registration:
    T_cad_recon: np.ndarray
    points_in_cad: np.ndarray
    correction_mm: float
    correction_deg: float
    fitness: float
    inlier_rmse_mm: float
    iterations: int
    inlier_fraction_used: float = float("nan")
    metadata: dict = field(default_factory=dict)

    def summary(self):
        return {
            "correction_mm": self.correction_mm,
            "correction_deg": self.correction_deg,
            "fitness": self.fitness,
            "inlier_rmse_mm": self.inlier_rmse_mm,
            "iterations": self.iterations,
            "inlier_fraction_used": self.inlier_fraction_used,
            **self.metadata,
        }


def register_to_cad(points, cad, T_init_cad_from_recon=None, max_correspondence_mm=6.0,
                    iterations=80, trim_fraction=0.75, refine_passes=3):
    """Register a reconstruction into the CAD frame by trimmed ICP.

    ``T_init_cad_from_recon`` is the starting guess.  On synthetic data that is
    the ground-truth pose, and using it is legitimate *as an initialisation*
    only -- ICP then moves from there, and :attr:`Registration.correction_mm`
    says how far.  Without a starting guess the identity is used, which will
    only converge if the two frames were already close.

    Trimmed ICP -- fitting on the closest ``trim_fraction`` of points each pass
    -- rather than plain ICP, because a sparse reconstruction always retains
    some points that are nowhere near the surface, and those would otherwise
    pull the whole registration towards themselves.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    T = np.eye(4) if T_init_cad_from_recon is None else np.array(
        T_init_cad_from_recon, dtype=float
    ).reshape(4, 4)
    T0 = T.copy()

    if not len(points):
        return Registration(T, points, 0.0, 0.0, 0.0, float("nan"), 0)

    import open3d as o3d

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(points)

    # ICP against a dense surface sampling of the mesh.  Sampling rather than
    # using the vertices: the bracket's flat faces are two triangles each, so
    # its vertices are a dozen corners and ICP against them is meaningless.
    mesh = to_open3d_mesh(cad)
    target = mesh.sample_points_poisson_disk(
        number_of_points=min(30000, max(4000, 40 * len(cad.faces))), init_factor=4
    )
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )

    distance = SurfaceDistance(cad)
    threshold = max_correspondence_mm * 1e-3
    total_iterations, fitness, rmse = 0, 0.0, float("nan")
    used = float("nan")

    for _ in range(int(refine_passes)):
        d = distance(se3.transform(T, points))
        if not len(d):
            break
        cutoff = np.quantile(d, trim_fraction)
        keep = d <= max(cutoff, 1e-4)
        used = float(keep.mean())
        if keep.sum() < 4:
            break
        trimmed = o3d.geometry.PointCloud()
        trimmed.points = o3d.utility.Vector3dVector(points[keep])

        result = o3d.pipelines.registration.registration_icp(
            trimmed, target, threshold, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iterations)),
        )
        T = np.array(result.transformation, dtype=float)
        fitness = float(result.fitness)
        rmse = float(result.inlier_rmse) * 1000.0
        total_iterations += int(iterations)

    correction = se3.pose_error(T, T0)
    return Registration(
        T_cad_recon=T,
        points_in_cad=se3.transform(T, points),
        correction_mm=correction[0] * 1000.0,
        correction_deg=correction[1],
        fitness=fitness,
        inlier_rmse_mm=rmse,
        iterations=total_iterations,
        inlier_fraction_used=used,
        metadata={"initialised_from": "identity" if T_init_cad_from_recon is None
                  else "ground truth (evaluation only)"},
    )


def sparse_accuracy(points_in_cad, cad, inlier_mm=5.0, distance=None):
    """Point-to-surface error statistics for a cloud already in the CAD frame.

    The median is quoted alongside the mean because the distribution has a
    long tail: a handful of points that survived every filter and are still
    wrong will move the mean and not the median, and the difference between
    the two is itself informative.
    """
    points_in_cad = np.asarray(points_in_cad, dtype=float).reshape(-1, 3)
    if not len(points_in_cad):
        return {"points": 0, "median_mm": float("nan"), "mean_mm": float("nan"),
                "p90_mm": float("nan"), "rms_mm": float("nan"), "inlier_fraction": 0.0}
    distance = distance or SurfaceDistance(cad)
    d = distance(points_in_cad) * 1000.0
    return {
        "points": int(len(d)),
        "median_mm": float(np.median(d)),
        "mean_mm": float(np.mean(d)),
        "p90_mm": float(np.quantile(d, 0.90)),
        "p99_mm": float(np.quantile(d, 0.99)),
        "max_mm": float(np.max(d)),
        "rms_mm": float(np.sqrt(np.mean(d ** 2))),
        "inlier_fraction": float(np.mean(d <= inlier_mm)),
        "inlier_threshold_mm": float(inlier_mm),
    }


def completeness(cad, points_in_cad, threshold_mm=3.0, n_samples=20000, seed=0):
    """How much of the model's surface has a reconstructed point near it.

    Accuracy and completeness are different failures and both matter: a
    reconstruction of one well-textured face can be accurate to a tenth of a
    millimetre and useless.  This samples the CAD surface uniformly by area and
    asks what fraction of those samples has a reconstructed point within
    ``threshold_mm``.
    """
    points_in_cad = np.asarray(points_in_cad, dtype=float).reshape(-1, 3)
    if not len(points_in_cad):
        return {"completeness": 0.0, "threshold_mm": float(threshold_mm),
                "samples": 0, "median_mm": float("nan")}

    rng = np.random.default_rng(seed)
    samples, _ = cad.sample_surface(int(n_samples), rng)

    from scipy.spatial import cKDTree

    tree = cKDTree(points_in_cad)
    d, _ = tree.query(samples, k=1)
    d = d * 1000.0
    return {
        "completeness": float(np.mean(d <= threshold_mm)),
        "threshold_mm": float(threshold_mm),
        "samples": int(len(samples)),
        "median_mm": float(np.median(d)),
        "p90_mm": float(np.quantile(d, 0.90)),
    }


def chamfer(points_in_cad, cad, n_samples=20000, seed=0, distance=None):
    """Symmetric chamfer distance between a cloud and the CAD surface, in mm.

    Reported as its two halves as well as the sum, because they mean different
    things: accuracy is cloud-to-surface and completeness is surface-to-cloud,
    and a single number hides which one failed.
    """
    points_in_cad = np.asarray(points_in_cad, dtype=float).reshape(-1, 3)
    distance = distance or SurfaceDistance(cad)
    a = distance(points_in_cad) * 1000.0

    rng = np.random.default_rng(seed)
    samples, _ = cad.sample_surface(int(n_samples), rng)
    from scipy.spatial import cKDTree

    b = cKDTree(points_in_cad).query(samples, k=1)[0] * 1000.0
    return {
        "accuracy_mm": float(np.mean(a)),
        "completeness_mm": float(np.mean(b)),
        "chamfer_mm": float(np.mean(a) + np.mean(b)),
        "accuracy_median_mm": float(np.median(a)),
        "completeness_median_mm": float(np.median(b)),
    }
