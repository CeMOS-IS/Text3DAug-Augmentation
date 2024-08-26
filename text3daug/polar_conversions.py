import numpy as np


def cart2polar(input_xyz: np.ndarray) -> np.ndarray:
    """Converts a cartesian [x, y, z] pointcloud into polar
    coordinates [range, phi-azimuth, z-height].

    Args:
        input_xyz (np.ndarray): Cartesian coordinates (N, [x, y, z])

    Returns:
        np.ndarray: Polar coordinates (N, [range, phi, z])
    """
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cart(input_polar: np.ndarray) -> np.ndarray:
    """Converts a polar pointcloud [range, phi, z] into cartesian
    coordinates [x, y, z]

    Args:
        input_polar (np.ndarray): Polar coordinates (N, [range, phi, z])

    Returns:
        np.ndarray: Cartesian coordinates (N, [x, y, z])
    """
    x = input_polar[:, 0] * np.cos(input_polar[:, 1])
    y = input_polar[:, 0] * np.sin(input_polar[:, 1])
    return np.stack((x, y, input_polar[:, 2]), axis=1)
