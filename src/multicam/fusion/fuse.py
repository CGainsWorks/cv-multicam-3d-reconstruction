"""Point-cloud fusion and meshing, with Open3D.

The depth maps out of the plane sweep are per-view, overlapping and noisy.
This turns them into one surface:

1. Concatenate every unprojected point into the object frame.
2. Voxel downsample.  This is the fusion step: a voxel containing points from
   four views returns their centroid, which averages away the independent noise
   in each view, and it also bounds the size of everything downstream.
3. Statistical outlier removal, on the points that survive.
4. Estimate and orient normals.  Poisson reconstruction needs consistently
   oriented normals and will produce a bag of disconnected blobs without them;
   they are oriented towards the cameras that saw each point, which is the only
   information available that actually determines the sign.
5. Screened Poisson reconstruction, then trim the low-density vertices.
   Poisson always returns a closed watertight surface, so the parts of it that
   no data supports are invented.  Trimming by density is what removes them,
   and not trimming is the single easiest way to produce a mesh that looks
   beautiful and is fiction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import se3


@dataclass
class FusionResult:
    points: np.ndarray                  #: (N, 3) fused cloud, object frame
    normals: np.ndarray = None
    mesh_vertices: np.ndarray = None
    mesh_faces: np.ndarray = None
    stats: dict = field(default_factory=dict)

    @property
    def has_mesh(self):
        return self.mesh_faces is not None and len(self.mesh_faces) > 0


def fuse_depth_maps(depth_maps, voxel_m=0.0008, outlier_neighbours=20, outlier_std=2.0,
                    normal_radius_m=0.004, normal_neighbours=40, subsample=1):
    """Fuse per-view depth maps into a single oriented point cloud."""
    import open3d as o3d

    clouds, viewpoints = [], []
    for dm in depth_maps:
        pts, _ = dm.points(subsample=subsample)
        if len(pts):
            clouds.append(pts)
            viewpoints.append(np.repeat(
                se3.camera_centre(dm.T_cam_object)[None, :], len(pts), axis=0
            ))
    if not clouds:
        return FusionResult(np.zeros((0, 3)), stats={"reason": "no valid depths"})

    points = np.vstack(clouds)
    viewpoints = np.vstack(viewpoints)
    raw = len(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Carry the viewpoint through the voxel filter as a colour, so that after
    # downsampling each surviving point still knows roughly where it was seen
    # from and the normals can be oriented.  Colours are averaged by the same
    # voxel filter, which is exactly the wanted behaviour.
    span = max(float(np.abs(viewpoints).max()), 1e-6)
    pcd.colors = o3d.utility.Vector3dVector((viewpoints / span + 1.0) * 0.5)

    down = pcd.voxel_down_sample(voxel_size=float(voxel_m)) if voxel_m > 0 else pcd
    after_voxel = len(down.points)

    if outlier_neighbours and len(down.points) > outlier_neighbours:
        down, _ = down.remove_statistical_outlier(
            nb_neighbors=int(outlier_neighbours), std_ratio=float(outlier_std)
        )
    after_outlier = len(down.points)
    if after_outlier < 16:
        return FusionResult(np.asarray(down.points), stats={
            "raw_points": raw, "after_voxel": after_voxel, "after_outlier": after_outlier,
            "reason": "too few points to estimate normals",
        })

    down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal_radius_m), max_nn=int(normal_neighbours)
        )
    )
    P = np.asarray(down.points)
    N = np.asarray(down.normals)
    V = (np.asarray(down.colors) * 2.0 - 1.0) * span
    # Flip any normal pointing away from the camera that saw it.
    towards = V - P
    flip = np.einsum("ij,ij->i", N, towards) < 0
    N[flip] *= -1.0
    down.normals = o3d.utility.Vector3dVector(N)

    return FusionResult(
        points=P, normals=N,
        stats={
            "raw_points": int(raw),
            "after_voxel": int(after_voxel),
            "after_outlier": int(after_outlier),
            "voxel_m": float(voxel_m),
            "views": int(len(clouds)),
        },
    )


def poisson_mesh(fusion, depth=9, density_quantile=0.12, crop_to_points=True,
                 margin_m=0.003):
    """Screened Poisson reconstruction with density trimming.

    ``density_quantile`` is the fraction of the lowest-density vertices to
    discard.  Those are the parts of the watertight surface that no measurement
    supports -- the underside of the part, which no camera on a 76-degree arc
    can ever see -- and keeping them would mean scoring invented geometry
    against the CAD model.
    """
    import open3d as o3d

    if fusion.normals is None or len(fusion.points) < 64:
        return fusion

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fusion.points)
    pcd.normals = o3d.utility.Vector3dVector(fusion.normals)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth), linear_fit=False
    )
    densities = np.asarray(densities)
    stats = dict(fusion.stats)
    stats["poisson_vertices"] = int(len(mesh.vertices))

    if density_quantile > 0 and len(densities):
        keep = densities > np.quantile(densities, float(density_quantile))
        mesh.remove_vertices_by_mask(~keep)
        stats["after_density_trim"] = int(len(mesh.vertices))

    if crop_to_points and len(mesh.vertices):
        lo = fusion.points.min(axis=0) - margin_m
        hi = fusion.points.max(axis=0) + margin_m
        box = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
        mesh = mesh.crop(box)
        stats["after_crop"] = int(len(mesh.vertices))

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    stats["mesh_vertices"] = int(len(mesh.vertices))
    stats["mesh_faces"] = int(len(mesh.triangles))
    return FusionResult(
        points=fusion.points, normals=fusion.normals,
        mesh_vertices=np.asarray(mesh.vertices),
        mesh_faces=np.asarray(mesh.triangles),
        stats=stats,
    )


def save_point_cloud(path, points, normals=None, colours=None):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float).reshape(-1, 3))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float))
    if colours is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colours, dtype=float))
    o3d.io.write_point_cloud(str(path), pcd, compressed=True)
    return path


def save_mesh(path, vertices, faces):
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(path), mesh)
    return path
