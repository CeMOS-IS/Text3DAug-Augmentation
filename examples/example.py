import os
import pickle

import numpy as np
import yaml

import text3daug

try:
    from helpers.kitti_miniset import SemanticKITTIMiniset
except ImportError:
    from .helpers.kitti_miniset import SemanticKITTIMiniset

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
VIS = True  # Visualize steps in Open3D

if __name__ == "__main__":
    # All paths
    path_scanner_config = os.path.join(THIS_DIR, "configs", "kitti_scanner.yaml")
    path_class_mesh_mapping = os.path.join(THIS_DIR, "kitti_all_instance_path.pkl")
    path_remission_file = os.path.join(THIS_DIR, "configs", "remission_kitti.txt")

    # Get the LiDAR scanner configuration
    with open(path_scanner_config, "r") as f:
        scanner_config_dict = yaml.safe_load(f)

    # Open the class integer to mesh path mapping
    with open(path_class_mesh_mapping, "rb") as f:
        # {
        # class (int): List[.obj path (str), ...]
        # }
        class_int_mesh_path_mapping = pickle.load(f)

    # Define what classes will be augmented. In this case, we keep all classes
    # that are defined in the pickle file. If you want to reduce the classes
    # to use, after pickling, change this list.
    class_list = list(class_int_mesh_path_mapping.keys())

    # Create dummy oversampling weights. If you want to oversample an instance
    # Change this dict accordingly. Here we sample with the same probability p
    # for all classes, so we set weights for each class to 1.0
    oversampling_weights = [1.0] * len(class_list)
    oversampling_weights = [x / len(class_list) for x in oversampling_weights]

    # Initialize Text3-Augmentation

    # NOTE: Uncomment for Segmentation
    # aug = text3daug.SegmentationInstancesInstances(
    #     scanner_config=scanner_config_dict,
    #     instance_dict=class_int_mesh_path_mapping,
    #     instance_list=class_list,
    #     instance_weights=oversampling_weights,
    #     rem_path=path_remission_file,
    #     visualize=VIS,
    # )

    aug = text3daug.DetectionInstances(
        scanner_config=scanner_config_dict,
        instance_dict=class_int_mesh_path_mapping,
        instance_list=class_list,
        instance_weights=oversampling_weights,
        rem_path=path_remission_file,
        visualize=VIS,
    )

    # Iterate over the Pseudo-Miniset
    # Lidar (N, 4), Semantic Label (N, 1), Instance Label (N, 1)
    while True:
        for lidar, semantic_label, instance_label in SemanticKITTIMiniset(THIS_DIR):

            # NOTE: Uncomment for Segmentation
            # # Actually apply the augmentation
            # lidar_xyz, semantic_label, instance_label, lidar_remission = (
            #     aug.instance_aug(
            #         lidar[:, :3], semantic_label, instance_label, lidar[:, 3:]
            #     )
            # )

            lidar_xyz, new_bboxes, new_class_ints, lidar_remission = aug.instance_aug(
                lidar[:, :3], lidar[:, 3:]
            )
