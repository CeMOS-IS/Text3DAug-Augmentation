import math
from typing import List, Type

import numpy as np
import open3d as o3d


def o3d_visualize_pc(xyz: np.ndarray, geometries: List = None):
    """Visualizes a pointcloud in Open3D.

    Args:
        xyz (np.ndarray): (N, 3) array -> (N, [x, y, z])
        geometries (List): Other Open3D geometries to add to visualization.
            Defaults to None.

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

    vis_elements = [vis]
    if geometries:
        vis_elements.extend(geometries)

    o3d.visualization.draw_geometries(vis_elements)


def o3d_visualize_mesh(mesh):
    """Visualizes a mesh in Open3D."""

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=3.0, origin=[0, 0, 0]
    )

    o3d.visualization.draw_geometries([mesh])


def bbox3d_to_o3d(
    bbox: np.ndarray, rgb: List[float] = [1, 0, 0]
) -> Type[o3d.utility.Vector3dVector]:
    """Converts a bounding box dataclass to Open3D visualization format.

    Args:
        bbox (np.ndarray): BoundingBox vector [x, y, z, dx, dy, dz, heading].
        rgb (List[float]): Normalized RGB values for visualization. Default red.
    """

    # Get min max of all box corners from lwh, center
    center = bbox[:3]
    x, y, z = center
    l, w, h = bbox[3:6]
    rotation = bbox[-1]

    # Get corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2] + x
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2] + y
    z_corners = [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2] + z

    corners = np.stack([x_corners, y_corners, z_corners], axis=-1)  # (3, 8)

    # Rotate corner coordinates according to bbox heading
    heading_matrix = np.array(
        [
            [math.cos(rotation), -math.sin(rotation), 0],
            [math.sin(rotation), math.cos(rotation), 0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,  # Numpy deprecation stuff
    )
    corners = corners - center
    corners = np.dot(heading_matrix, corners.T)
    corners = corners.T + center

    line_ids = np.asarray(
        [
            [0, 1],
            [0, 3],
            [0, 4],
            [1, 2],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [5, 4],
            [5, 6],
            [7, 4],
            [7, 6],
        ]
    )

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(line_ids),
    )

    line_set.colors = o3d.utility.Vector3dVector([rgb for _ in range(len(line_ids))])
    return line_set
