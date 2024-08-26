from typing import Tuple

import numpy as np

from . import RayTracerCython as rtc


def raytracing(
    verts: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    remission: np.ndarray,
    rays: np.ndarray,
    H: int,
    W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs Raytracing in LiDAR characteristics on a mesh.

    Args:
        verts (np.ndarray): Mesh vertices (N, 3)
        faces (np.ndarray): Mesh faces (N, 3)
        colors (np.ndarray): Mesh colors (N, 3)
        remission (np.ndarray): Mesh remission (N, 1)
        rays (np.ndarray): (H*W, [x, y, z]) Ray array from f(x) create_rays
        H (int): Nr. of vertical scanlines of LiDAR.
        W (int): Nr. of horizontal scanlines of LiDAR.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Data returned by Raytracing.

            Returned points (N, [x, y, z])

            Returned remission (N, 1)
    """

    # Arrays must be contiguous and 1D
    verts_c = np.ascontiguousarray(verts.reshape(-1).astype(np.float32))
    faces_c = np.ascontiguousarray(faces.reshape(-1).astype(np.int32))
    colors_c = np.ascontiguousarray(colors.reshape(-1).astype(np.int32))
    rem_c = remission.reshape(-1).astype(np.float32)
    rays_c = rays.reshape(-1).astype(np.float32)

    origin = np.array([0, 0, 0]).astype(np.float32)

    ray_endpoints = np.ascontiguousarray(
        np.zeros((H, W, 3)).reshape(-1).astype(np.float32)
    )
    ray_colors = np.ascontiguousarray(np.zeros((H, W, 3)).reshape(-1).astype(np.int32))
    range_image = np.ascontiguousarray(np.zeros((H, W)).reshape(-1).astype(np.float32))
    rem_image = np.ascontiguousarray(np.zeros((H, W)).reshape(-1).astype(np.float32))

    rtc.C_Trace(
        rays_c,
        origin,
        verts_c,
        faces_c,
        colors_c,
        rem_c,
        ray_endpoints,
        ray_colors,
        range_image,
        rem_image,
        H,
        W,
    )

    return ray_endpoints.reshape(-1, 3), rem_image.reshape(-1, 1)


def create_rays(fov_up, fov_down, H, W):
    """Creates a set of Rays for Raytracing based on rotating LiDAR characteristics.

    Args:
        fov_up (_type_): LiDAR vertical FOV up.
        fov_down (_type_): LiDAR vertical FOV down.
        H (_type_): Number of horizontal scanlines. Assumes 360° horizontal FOV.
        W (_type_): Number of vertical scanlines.

    Returns:
        np.ndarray: Rays (H*W, [x, y, z])
    """

    beams = []

    # correct initial rotation of sensor
    initial = 180
    yaw_angles = np.linspace(0, 360, W) + initial
    larger = yaw_angles > 360
    yaw_angles[larger] -= 360
    yaw_angles = yaw_angles / 180.0 * np.pi
    pitch = np.linspace(fov_up, fov_down, H) / 180.0 * np.pi
    pitch = np.pi / 2 - pitch
    yaw = yaw_angles

    point_x = np.expand_dims(np.sin(pitch), axis=-1) * np.expand_dims(
        np.cos(-yaw), axis=0
    )
    point_y = np.expand_dims(np.sin(pitch), axis=-1) * np.expand_dims(
        np.sin(-yaw), axis=0
    )
    point_z = np.expand_dims(np.cos(pitch), axis=-1) * np.expand_dims(
        np.ones(yaw.shape), axis=0
    )
    beams = np.stack([point_x, point_y, point_z], axis=-1)

    beams = np.array(beams).reshape(W * H, -1)
    return np.ascontiguousarray(beams.astype(np.float32))
