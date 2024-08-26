#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Union

import numpy as np
import open3d as o3d

from .helpers import expand_array_2, suppress_print
from .polar_conversions import cart2polar, polar2cart
from .raytrace_operations import create_rays, raytracing
from .visualize import o3d_visualize_mesh, o3d_visualize_pc


class SegmentationInstances(object):
    def __init__(
        self,
        scanner_config: Dict,
        instance_dict: Dict,
        instance_list: List[int],
        instance_weights: List[float],
        rem_path: str,
        weight_noise: float = 0.01,
        p_noise: float = 0.6,
        p_drop: float = 0.1,
        min_range: float = 3,
        max_range: float = 40,
        min_height: float = -0.5,
        add_num: int = 5,
        visualize: bool = False,
    ):
        """Text3DAugmentation for a pointcloud.

        Args:
            scanner_config (Dict): LiDAR configuration dictionary to setup Raytracing.
            instance_dict (Dict): Dictonary mapping class integers to a list of .obj mesh filepaths.
            instance_list (List[int]): List of class integers that is used for augmentation. Modify this list
                to include / exclude classes.
            instance_weights (List[float]): Weights to sample instances by. This over-/undersamples a class. If
                you want to deactivate this, set all weights to 1.0.
            rem_path (str): Path to the remission .txt file. Contains a remission value in each new line. Values
                are sampled randomly from this file.
            weight_noise (float, optional): The magnitude of noise, when noise is added to an instance point. Defaults to 0.01.
            p_noise (float, optional): The probability 0..1 to add noise to an instance point. Defaults to 0.6.
            p_drop (float, optional): The probability 0..1 to remove an instance point. Defaults to 0.1.
            min_range: (float): The minimum range in meters to search for space to place an instance. Defaults to 3
            max_range: (float): The maximum range in meters to search for space to place an instance. Defaults to 40
            min_height: (float): The minimum height of points to ignore as road points while searchinf for space. Defaults to -0.5
            add_num (int, optional): The number of instances to add. Less will be added if no space is found.. Defaults to 5.
            visualize (bool, optional): Visualize the pointcloud before and after augmentation. Defaults to False.
        """
        # Instance information
        self.instance_list = instance_list  # List of "Thing" classes in a dataset which are placed as instances
        self.instance_weights = instance_weights  # Weights of a class
        self.instance_dict = instance_dict  # Dictionary of class names and matching List of .obj mesh files

        # Local Augmentations
        self.add_num = add_num
        self.weight_noise = weight_noise
        self.p_noise = p_noise
        self.min_range = min_range
        self.max_range = max_range
        self.min_height = min_height
        self.p_drop = p_drop

        # Scanner params
        self.name = scanner_config["name"]
        self.fov_up = scanner_config["fov_up"]
        self.fov_down = scanner_config["fov_down"]
        self.beams = scanner_config["beams"]
        self.angle_res_hor = scanner_config["angle_res_hor"]
        self.fov_hor = scanner_config["fov_hor"]
        self.W = int(self.fov_hor / self.angle_res_hor)

        # Path to remission file
        self.remission_data = np.loadtxt(rem_path)

        # Show steps in Open3D
        self.visualize = visualize

    def instance_aug(self, point_xyz, point_label, point_inst, point_feat=None):
        """random rotate and flip each instance independently.

        Args:
            point_xyz: [N, 3], point location
            point_label: [N, 1], semantic class label
            point_inst: [N, 1], instance label
            point_feat: [N, 1], Remission
        """
        if self.visualize:
            o3d_visualize_pc(point_xyz)

        # Keep track of instances that were successfully placed
        # in a LiDAR pointcloud.
        self.added_instances = []

        # Randomly choose a class to place.
        #   len(list):  Number of classes to choose from
        #   add_num:    Number of classes to choose
        #   replace:    Allow for choosing the same class multiple times
        #   p:          Weights for oversampling classes
        class_choice = np.random.choice(
            len(self.instance_list),
            self.add_num,
            replace=True,
            p=self.instance_weights,
        )
        # class_inst:       Class to place
        # class_inst_count: How often to place a class
        class_inst, class_inst_count = np.unique(class_choice, return_counts=True)

        # Iterate over each class, randomly select instances and place them
        add_idx = np.max(point_inst)  # For instance-seg labels

        for n, count in zip(class_inst, class_inst_count):

            # Get the list of .obj files in self.instance_dict corresponding to
            # class n.
            # Randomly choose count .obj files from this list.
            instance_choices = np.random.choice(
                len(self.instance_dict[self.instance_list[n]]), count
            )

            # Iterate over each instance and place it
            for idx in instance_choices:

                # Open an .obj mesh as an Open3D geometry (silently)
                with suppress_print():
                    mesh = o3d.io.read_triangle_mesh(
                        self.instance_dict[self.instance_list[n]][idx],
                        enable_post_processing=True,
                    )

                # if self.visualize:
                #     o3d_visualize_mesh(mesh)

                # Add a remission value to the mesh vertices
                instance_points = self.add_remission(mesh.vertices)
                instance_xyz = instance_points[:, :3]  # x, y, z
                instance_remission = instance_points[:, 3]  # remission

                # Place instance randomly in scan for 5 attempts.
                # If no space can be found, do not place instance
                fail_flag = True

                for _ in range(5):
                    # Random rotation and translation of instance as initial placement
                    instance_xyz_init = self.random_z_rotate_and_xy_translate(
                        instance_xyz
                    )

                    # Transform pointcloud and instance into polar coordinates.
                    # This serverd as an initial starting point for the instance.
                    # Based on the translation, the range of the instance is determined.
                    point_polar = cart2polar(point_xyz)
                    instance_polar_init = cart2polar(instance_xyz_init)

                    # Use polar coordinates of instance and pointcloud to search for
                    # an empty space.
                    # The placement of the instance is modified to fit into a free space
                    # of the original pointcloud.
                    instance_polar_placed = self.search_empty_space(
                        point_polar, instance_polar_init
                    )

                    # Shift Instance in its polar z-coordinate so it is placed
                    # on ground level
                    if instance_polar_placed is not None:
                        instance_polar_placed = self.adjust_to_ground(
                            point_polar, instance_polar_placed
                        )

                        fail_flag = False
                        break  # Exit loop as instance was sucessfully placed

                if fail_flag:
                    continue  # Try again with a new instance

                # Convert the placed instance coordinates back into cartesian
                # coordinates.
                instance_xyz_placed = polar2cart(instance_polar_placed)

                # Prepare for Raytracing. Use the instance cartesian coordinates
                # as the new mesh vertices.
                verts = instance_xyz_placed
                faces = np.asarray(mesh.triangles)
                colors = np.zeros_like(verts)

                # Create Rays for Raytracing based on the LiDAR properties
                rays = create_rays(self.fov_up, self.fov_down, self.beams, self.W)

                # Perform Raytracing on the placed instance and get back the results
                raytraced_points, raytraced_remissions = raytracing(
                    verts, faces, colors, instance_remission, rays, self.beams, self.W
                )

                # Filter out rays that did not return a result based on distance=0.
                # Then update the Instance points to only include returns of raytracing.
                empty_rays = np.nonzero(np.linalg.norm(raytraced_points, axis=1))
                instance_lidar = raytraced_points[empty_rays]
                instance_lidar_feat = raytraced_remissions[empty_rays]

                # Add random noise to the Raytraced instance
                points_to_noise = np.random.choice(
                    [True, False],
                    size=instance_lidar.shape[0],
                    p=[self.p_noise, 1 - self.p_noise],
                )
                random_noise = (
                    np.random.normal(
                        scale=self.weight_noise, size=(instance_lidar.shape[0], 3)
                    )
                    * points_to_noise[:, None]
                )
                instance_lidar = instance_lidar + random_noise

                # Randomly remove points from Raytraced instance
                self.p_drop = 0.1  # 1.0 = 100% der Punkte wurden entfernt
                random_remove = np.random.choice(
                    [True, False],
                    size=instance_lidar.shape[0],
                    p=[1 - self.p_drop, self.p_drop],
                )
                instance_lidar = instance_lidar[random_remove]
                instance_lidar_feat = instance_lidar_feat[random_remove]

                # if self.visualize:
                #     o3d_visualize_pc(instance_lidar[:, :3])

                # Remove points behind a instance.
                if instance_lidar.size == 0:
                    continue

                else:
                    instance_lidar_polar = cart2polar(instance_lidar)
                    (
                        filtered_point_xyz,
                        filtered_label,
                        filtered_inst,
                        filtered_feat,
                    ) = self.remove_occluded_points(
                        point_xyz,
                        point_label,
                        point_inst,
                        point_feat,
                        point_polar,
                        instance_lidar_polar,
                    )

                # Create an instance label for the instance.
                instance_label = np.ones((instance_lidar.shape[0],), dtype=np.uint8) * (
                    self.instance_list[n]
                )
                instance_inst = np.ones((instance_lidar.shape[0],), dtype=np.uint32) * (
                    add_idx << 16
                )
                add_idx += 1

                # Add the instance to the LiDAR pointcloud
                point_xyz = np.concatenate((filtered_point_xyz, instance_lidar), axis=0)

                filtered_label = expand_array_2(filtered_label)
                instance_label = expand_array_2(instance_label)
                filtered_inst = expand_array_2(filtered_inst)
                instance_inst = expand_array_2(instance_inst)

                point_label = np.concatenate((filtered_label, instance_label), axis=0)
                point_inst = np.concatenate((filtered_inst, instance_inst), axis=0)

                # Add instance remission to LiDAR remission
                if point_feat is not None:
                    instance_fea = expand_array_2(instance_remission)

                    if len(instance_fea.shape) == 1:
                        instance_fea = instance_fea[..., np.newaxis]

                    point_feat = np.concatenate(
                        (filtered_feat, instance_lidar_feat), axis=0
                    )

        if self.visualize:
            o3d_visualize_pc(point_xyz[:, :3])

        # Return data
        if len(point_label.shape) == 1:
            point_label = point_label[..., np.newaxis]
        if len(point_inst.shape) == 1:
            point_inst = point_inst[..., np.newaxis]
        if point_feat is not None:
            return point_xyz, point_label, point_inst, point_feat
        else:
            return point_xyz, point_label, point_inst

    def add_remission(self, vertices: np.ndarray) -> np.ndarray:
        """Add remission to cartesian vertices.
        (N, [x, y, z]) -> (N, [x, y, z, remission])

        Args:
            vertices (np.ndarray): Vertices of a mesh (N, 3).

        Returns:
            np.ndarray: Vertices with added remission (N, 4).
        """
        instance_xyz = np.asarray(vertices)  # (N, 3)
        N_pts = instance_xyz.shape[0]

        remission_data = self.remission_data

        # Randomly choose vertices where remission is assigned.
        # Half the vertices are not asigned a value (reason for N*0.5)
        random_points = np.random.choice(N_pts, size=int(N_pts * 0.5), replace=False)

        # Add zero remission values (N, 3) -> (N, 4)
        instance_points = np.hstack(
            (instance_xyz, np.zeros((instance_xyz.shape[0], 1)))
        )

        # Asign a randomly chosen remission value
        instance_points[random_points, 3] = remission_data[
            np.random.choice(remission_data.shape[0], size=random_points.shape[0])
        ]
        return instance_points  # (N, 4)

    def random_z_rotate_and_xy_translate(self, xyz: np.ndarray) -> np.ndarray:
        """Randomly rotate a pointcloud in its z-Axis, then translate the
        pointcloud in x,y.

        Args:
            xyz (np.ndarray): Pointcloud (N, [x, y, z, ... features])

        Returns:
            np.ndarray: Transformed pointcloud (N, [x, y, z, ... features]).
        """
        new_xyz = xyz.copy()

        # Rotate z-axis of pointcloud by a random angle (-180..180 deg in radians)
        rot_angle = np.random.uniform(low=-np.pi, high=np.pi)
        R = np.array(
            [
                [np.cos(rot_angle), -np.sin(rot_angle)],
                [np.sin(rot_angle), np.cos(rot_angle)],
            ]
        )
        new_xyz[:, :2] = np.dot(xyz[:, :2], R.T)

        # Randomly translate a pointcloud in its x,y plane
        # Translation value is randomly chosen to be within a quadrant
        # of the pointcloud
        quadrant = np.random.randint(1, 5)  # Random{Q1, Q2, Q3, Q4}

        # Q1: positive x, positive y
        if quadrant == 1:
            random = np.random.uniform(low=10, high=35, size=(1, 2))

        # Q2: negative x, positive y
        elif quadrant == 2:
            random = np.array(
                [
                    np.random.uniform(low=-35, high=-10),
                    np.random.uniform(low=10, high=35),
                ]
            ).reshape(1, 2)

        # Q3: negative x, negative y
        elif quadrant == 3:
            random = np.random.uniform(low=-35, high=-10, size=(1, 2))

        # Q4: positive x, negative y
        else:
            random = np.array(
                [
                    np.random.uniform(low=10, high=35),
                    np.random.uniform(low=-35, high=-10),
                ]
            ).reshape(1, 2)

        # Actual Translation of pointcloud
        new_xyz[:, :2] += random
        return new_xyz

    def search_empty_space(
        self,
        points_polar: np.ndarray,
        instance_points_polar: np.ndarray,
        min_range: float = 3,
        max_range: float = 40,
        min_height: float = -0.5,
    ) -> Union[None, np.ndarray]:
        """Uses polar coordinates to find an empty space for a instance
        in the pointcloud where it will be placed.

        Ignores ground points when placing the instance, so it can be placed on the street.

        Args:
            points_polar (np.ndarray): Pointcloud in polar coordinates.
            instance_points_polar (np.ndarray): Instance in polar coordinates.
            min_range (float, optional): Min. range in meters to place instance at. Defaults to 3.
            max_range (float, optional): Max. range in meters to place instance at. Defaults to 40.
            min_height (float, optional): Min. height to place instance in scan. Defined as ground level.
                                          Depends on LiDAR sensor setup. Defaults to -0.5.

        Returns:
            Union[None, np.ndarray]: None if no space is found. Array if successfull.
        """

        inst_polar = instance_points_polar.copy()

        # Filter out points in the pointcloud which are not within
        # min. / max. range and min. height.
        # Min. height means that ground points are ignored as a condition
        # when search for a free space. As a result, the instance can be placed
        # on the ground.
        points_polar_filtered = points_polar[
            (points_polar[:, 2] > min_height)
            & (points_polar[:, 0] < max_range)
            & (points_polar[:, 0] > min_range)
        ]

        # Go through all instances which have already been added and add them
        # to the filtered pointcloud. This assures, that no two instances can
        # take the same place in the pointcloud.
        for instance in self.added_instances:
            points_polar_filtered = np.vstack([points_polar_filtered, instance])

        # Calculate the angular width (azimuth, Phi) of the instance to search
        # for a free space in the
        instance_azimuth_width = np.max(instance_points_polar[:, 1]) - np.min(
            instance_points_polar[:, 1]
        )

        # Sort points by increasing Azimuth
        points_azimuth_sort = points_polar_filtered[
            np.argsort(points_polar_filtered[:, 1])
        ]

        # Go from point to point and check if the distance between azimuths
        # is large enough to fit the instance. Use this to propose free areas.
        free_areas = []
        for i in range(len(points_azimuth_sort) - 1):

            area = abs(points_azimuth_sort[i + 1, 1] - points_azimuth_sort[i, 1])
            if area > instance_azimuth_width:

                start_index = i
                end_index = i + 1
                free_areas.append((start_index, end_index))

        # Randomly choose a free area, then rotate the instance into
        # this area, by adding the Azimuth.
        if free_areas:

            random_area = np.random.choice(len(free_areas))
            start_index, end_index = free_areas[random_area]
            random_points = points_azimuth_sort[start_index, 1]

            offset = random_points - np.min(inst_polar[:, 1])
            inst_polar[:, 1] += offset

            # Keep track of the placed instance in the list
            self.added_instances.append(inst_polar)

        else:
            # Failed to find a free area for the instance
            inst_polar = None

        return inst_polar

    def adjust_to_ground(
        self, points_polar: np.ndarray, instance_points_polar: np.ndarray
    ) -> np.ndarray:
        """Checks there are ground points below the instance. Shifts the instance
        up or down to make sure it aligns with ground level.

        Args:
            points_polar (np.ndarray): Original pointcloud in polar coordinates.
            instance_points_polar (np.ndarray): Instance in polar coordinates.

        Returns:
            np.ndarray: Shifted instance in polar coordinates.
        """
        new_polar = instance_points_polar.copy()

        # Get the size of the instance as a bounding box in polar coordinates
        min_rho = np.min(instance_points_polar[:, 0])
        max_rho = np.max(instance_points_polar[:, 0])
        min_phi = np.min(instance_points_polar[:, 1])
        max_phi = np.max(instance_points_polar[:, 1])

        # Search in the original pointcloud, if any points are within the bounding
        # box of the instance.
        occluded_points = []
        rho = points_polar[:, 0]
        phi = points_polar[:, 1]
        z = points_polar[:, 2]

        cond = np.where(
            (min_rho <= rho) & (rho <= max_rho) & (min_phi <= phi) & (phi <= max_phi)
        )

        # If points are found, assume that they are the street level.
        # Shift the instance up in polar coordinate z to raise it above the street level.
        # This only works in tandem with f(x) search_empty_space, as empty space
        # ignores points at the ground level when choosing a free space.
        if np.count_nonzero(cond) > 0:
            occluded_points = z[cond]
            offset = np.min(occluded_points) - np.min(new_polar[:, 2])
            new_polar[:, 2] += offset

        # Return the ground aligned instance
        return new_polar

    def remove_occluded_points(
        self,
        point_xyz: np.ndarray,
        point_label: np.ndarray,
        point_inst: np.ndarray,
        point_feat: np.ndarray,
        point_polar: np.ndarray,
        instance_lidar_polar: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compares the original pointcloud with the instance in polar coordinates.
        Original points of the pointcloud are removed, only if they are within the
        azimuth bounds of the instance and behind the instance.

        Args:
            point_xyz (np.ndarray): Original pointcloud coordinates (N, 3)
            point_label (np.ndarray): Pointcloud semantic label. (N, 1)
            point_inst (np.ndarray): Pointcloud instance seg. label. (N, 1)
            point_feat (np.ndarray): Pointcloud remission. (N, 1)
            point_polar (np.ndarray): Pointcloud polar coordinates. (N, 3)
            instance_lidar_polar (np.ndarray): Instance in polar coordinates. (N, 3)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

                Pointcloud with occluded points removed. (N, 3)

                Semantic label with occluded points removed. (N, 1)

                Instance label with occluded points removed. (N, 1)

                Remission with occluded points removed. (N, 1)
        """
        # Get the bounds of the instances azimuth, as well as its maximum range
        min_phi_instance = np.min(instance_lidar_polar[:, 1])
        max_phi_instance = np.max(instance_lidar_polar[:, 1])
        max_rho_instance = np.max(instance_lidar_polar[:, 0])

        # Remove all points which are inside the azimuth window and behind the
        # maximum range
        mask_phi = (point_polar[:, 1] >= min_phi_instance) & (
            point_polar[:, 1] <= max_phi_instance
        )
        mask_rho = point_polar[:, 0] > max_rho_instance
        mask_to_remove = mask_phi & mask_rho
        mask_to_keep = ~mask_to_remove

        filtered_point_xyz = point_xyz[mask_to_keep]
        filtered_label = point_label[mask_to_keep]
        filtered_inst = point_inst[mask_to_keep]
        if point_feat is not None:
            filtered_feat = point_feat[mask_to_keep]

        return filtered_point_xyz, filtered_label, filtered_inst, filtered_feat
