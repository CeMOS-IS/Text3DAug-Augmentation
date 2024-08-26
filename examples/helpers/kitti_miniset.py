import os

import yaml

from .fileio import open_kitti_label, open_kitti_points, search_directory


class SemanticKITTIMiniset:
    """A small iterator to return data from the example miniset."""

    def __init__(self, root_dir: str) -> None:

        miniset_path = os.path.join(root_dir, "minimal", "sequences", "00")
        config_path = os.path.join(root_dir, "configs", "semantic-kitti.yaml")

        self.bin_files = search_directory(
            miniset_path, extension=".bin", recursive=True
        )
        self.label_files = search_directory(
            miniset_path, extension=".label", recursive=True
        )
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.index = 0

    def __len__(self):
        """
        Return shortest length
        """
        return min(len(self.bin_files), len(self.label_files))

    def __iter__(self):

        return self

    def __next__(self):
        if self.index < len(self):
            bin = self.bin_files[self.index]
            label = self.label_files[self.index]

            lidar = open_kitti_points(bin)
            semantic_label, instance_label = open_kitti_label(
                label, self.config["learning_map"]
            )
            self.index += 1

            return lidar, semantic_label, instance_label

        else:
            raise StopIteration
