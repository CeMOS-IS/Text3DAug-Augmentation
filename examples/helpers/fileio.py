import glob
import os
from typing import List, Optional, Tuple

import numpy as np


def open_kitti_points(
    file_path: str, data_columns: int = 4, file_dtype: np.dtype = np.float32
) -> np.ndarray:
    """Opens a KITTI formatted pointcloud file.

    Args:
        file_path (str): Path to .bin file.
        data_columns (int, optional): Columns in file [x, y, z, remission]. Defaults to 4.
        file_dtype (np.dtype, optional): Datatype for file contents. Defaults to np.float32.

    Returns:
        np.ndarray: (N, data_columns) pointcloud array.
    """

    # Open .bin files as np.array -> format x, y, z, remission
    xyzr = np.fromfile(file_path, dtype=file_dtype)
    return xyzr.reshape((-1, data_columns))


def open_kitti_label(
    file_path: str, kitti_remap: dict, file_dtype: np.dtype = np.uint32
) -> Tuple[np.ndarray, np.ndarray]:
    """Opens a KITTI formated label file.

    Args:
        file_path (str): Path to .label file.
        kitti_remap (dict): Dictionary for remapping semantic classes. Can be None.
        file_dtype (np.dtype, optional): Datatype for file contents. Defaults to np.uint32.

    Returns:
        Tuple[np.ndarray, np.ndarray]:

            Array of pointwise semantic labels
            Array of pointwise instance labels
    """

    label = np.fromfile(file_path, dtype=file_dtype)
    label = label.reshape((-1))

    # only semantic labels, not instances
    label_ints = label & 0xFFFF
    inst_data = label

    # Mapping to training classes
    if kitti_remap:
        maxkey = max(kitti_remap.keys())
        remap_lut = np.zeros((maxkey + 100), dtype=file_dtype)
        remap_lut[list(kitti_remap.keys())] = list(kitti_remap.values())

        label_ints = remap_lut[label_ints]

    if label_ints.ndim < 2:
        label_ints = np.expand_dims(label_ints, axis=-1)
    if inst_data.ndim < 2:
        inst_data = np.expand_dims(inst_data, axis=-1)
    return label_ints, inst_data


def search_directory(
    folder_path: str, extension: Optional[str] = None, recursive: bool = False
) -> List[str]:
    """
    Search a folder for files of a given extension. Sorts them by filename.

    Args:
        folder_path (str): The path to the folder to search.
        extension (Optional[str], optional): The file extension to search for. If None, returns all files. Defaults to None.
        recursive (bool, optional): If True, search all subfolders recursively. Defaults to False.

    Returns:
        List[str]: A sorted list of paths to the files found.
    """

    # Check if directory exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided path '{folder_path}' is not a valid directory.")

    # Build the search pattern
    if extension is None:
        search_pattern = "**/*" if recursive else "*"
    else:
        extension = extension.replace(".", "")  # Invariant to "."
        search_pattern = f"**/*.{extension}" if recursive else f"*.{extension}"

    # Get the full search path
    search_path = os.path.join(folder_path, search_pattern)

    # Use glob to find the files
    matched_files = glob.glob(search_path, recursive=recursive)

    # Filter out directories if any
    matched_files = [file for file in matched_files if os.path.isfile(file)]

    return sorted(matched_files)
