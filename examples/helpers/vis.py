import numpy as np
import open3d as o3d


def o3d_visualize(xyz: np.ndarray):
    """Visualizes a pointcloud in Open3D.

    Args:
        xyz (np.ndarray): (N, 3) array -> (N, [x, y, z])

    Raises:
        ValueError: Array is not (N, 3)
    """

    if np.shape(xyz)[1] != 3:
        raise ValueError(
            f"Pointcloud needs to be (N, 3) for Open3d not {np.shape(xyz)}."
        )

    vis = o3d.geometry.PointCloud()
    vis.points = o3d.utility.Vector3dVector(np.copy(xyz))
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=3.0, origin=[0, 0, 0]
    )

    o3d.visualization.draw_geometries([vis, coordinate_frame])
