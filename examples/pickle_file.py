import argparse
import glob
import os
import pickle
from collections import defaultdict

import yaml


def get_child_folders(folder: str) -> dict:
    """Gets the child folders of a directory.

    Args:
        folder (str): Path to directory.

    Returns:
        dict: Matching between folder name and
        folder path.

        In this mapping file, the folder name
        corresponds to the class name.
    """

    cf = {}
    for f in os.scandir(folder):
        if f.is_dir():
            cf[f.name] = f.path
    return cf


def get_obj_meshes(folder: str) -> list:
    """Searches a folder for .obj files. Non-recursive.
    Child folders will not be searched.

    Args:
        folder (str): Path to folder.

    Returns:
        list: List of .obj files in the folder.
    """
    search_path = os.path.join(folder, "*.obj")
    mesh_files = glob.glob(search_path)
    mesh_files = [os.path.abspath(m) for m in mesh_files]
    return mesh_files


def read_clip_eval(clip_eval: str) -> dict:
    """Read CLIP score .txt file.

    Args:
        clip_eval (str): Path to .txt file.

    Returns:
        dict: Matching between mesh filepath and corresponding clip score.

        {
            class name:
                file_name: clip_score,

                .
                .
                .

                file_name_n : ...,

            .
            .
            .

            class_name_n:
                file_name: clip_score,

                .
                .
                .
        }
    """

    with open(clip_eval, "r") as txt:
        content = txt.readlines()

    clip_mapping = defaultdict(dict)
    for mesh_score in content:
        class_name, _, file_name, clip_score = mesh_score.split()

        clip_mapping[class_name][file_name] = clip_score

    return clip_mapping


def parse_mesh_files(folder_path: str, label_mapping: dict, clip_mapping: dict) -> dict:
    """Walks through a mesh output folder and matched class with .obj filepaths.
    - Get child folder of mesh output folder and use it as the class name
    - Converts class name to integer label based on dataset mapping
    - Search each child folder for .obj files
    - Read CLIP score per mesh
    - Match class name and list of .obj filepaths with their CLIP score.

    Assumes the following folder structure:


    ├── folder_path/
        ├── car/
                ├── 0.obj
                ├── 1.obj
                ├── ...
        ├── motorcycle/
                ├── ...
        ├── .../
        .
        .
        .
        ├── clip.txt


    Args:
        folder_path (str): Path to mesh output folder.
        label_mapping (dict): Class name -> dataset integer mapping.
        clip_mapping (dict): Mesh path to CLIP score mapping.

    Returns:
        dict: Mapping between class and clipscore
            {
                class integer label: [sorted mesh file paths],
                class clip scores: [CLIP scores matching sorted mesh paths]
            }
    """
    # Walk through the directory to find classes
    class_folder_matching = get_child_folders(folder_path)

    # Walk through each class folder to obtain all meshes
    # and match them to the class
    class_mesh_mapping = defaultdict(list)
    for class_name, class_folder in class_folder_matching.items():
        mesh_files = get_obj_meshes(class_folder)

        # Match the clip score to each file
        matched_clip = []
        for m in mesh_files:
            m = os.path.basename(m)
            matched_clip.append(clip_mapping[class_name][m])

        # Match each class to the dataset integer label
        # Skip classes not defined above
        if class_name not in label_mapping.keys():
            continue
        else:
            int_label = label_mapping[class_name]
            class_mesh_mapping[int(int_label)].extend(mesh_files)
            class_mesh_mapping[f"{int(int_label)}_clip"].extend(matched_clip)

    return class_mesh_mapping


def sort_by_clip_and_subsample(class_mesh_mapping: dict, samples: int) -> dict:
    """Sorts found meshes by their associated CLIP score and samples
    the top-k meshes.

    Args:
        class_mesh_mapping (dict): Class / mesh path mapping.
        samples (int): Number of top-k samples to choose.

    Returns:
        dict: Subsampled class / mesh mapping.
    """

    sorted_mapping = defaultdict(list)
    for class_name in class_mesh_mapping.keys():
        if isinstance(class_name, str):
            continue

        obj_files = class_mesh_mapping[class_name]
        clip_scores = class_mesh_mapping[f"{class_name}_clip"]

        obj_files = [
            x
            for x, _ in sorted(
                zip(obj_files, clip_scores), key=lambda pair: pair[1], reverse=True
            )
        ]

        if samples:
            obj_files = obj_files[:samples]  # Only keep top n meshes

        sorted_mapping[class_name].extend(obj_files)
    return sorted_mapping


def save_to_pickle(out_path: str, class_mesh_mapping: dict):
    """Saves a dictionary into a pickle file.

    Args:
        out_path (str): Output path of pickle file.
        class_mesh_mapping (dict): Mapping of class name to .obj file paths
    """

    if len(class_mesh_mapping.items()) == 0:
        raise FileNotFoundError("No meshes found, check CLI args.")

    with open(out_path, mode="wb") as f:
        pickle.dump(class_mesh_mapping, f)

    # Print out stats of created pickle
    with open(out_path, mode="rb") as f:
        x = pickle.load(f)

    for cn, mf in x.items():
        reverse_dict = {v: k for k, v in label_mapping.items()}
        print(
            f"class {reverse_dict[cn]}",
            f"int label {cn}",
            f"nr. meshes {len(mf)}",
            "\t\t",
            class_mesh_mapping[cn][-1],
        )


def cli_parse():
    parser = argparse.ArgumentParser(
        description="Maps a mesh class (prompted string) to the corresponding dataset "
        "integer labels. Associates the corresponding .obj mesh files to this label. "
        "Returns a .pkl file containing a dict of Dict = class (int): List[paths to .obj meshes]"
    )
    parser.add_argument(
        "--mesh_path", type=str, help="Path to instance mesh folder.", required=True
    )

    _ = (
        "Mapping prompt string class to dataset integer"
        " - - - "
        "NOTE: The standard KITTI training map assumes that unlabled "
        "index is zero. This map may need to be adjusted depending on "
        "the network. "
        "If the network uses 255 or some other value for unlabeled, the "
        "standard traing map does not apply. "
    )
    parser.add_argument(
        "--mapping",
        type=str,
        help=_,
        required=True,
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name which is used during saving of .pkl file.",
        required=True,
        default=None,
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of top-k samples per class, filtered by CLIP score. Optional.",
        required=False,
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = cli_parse()

    folder_path = args.mesh_path
    if not os.path.isdir(folder_path):
        raise NotADirectoryError("CLI arg mesh_path is not a directory path.")

    clip_mapping = read_clip_eval(os.path.join(folder_path, "clip.txt"))

    with open(args.mapping) as f:
        label_mapping = yaml.safe_load(f)

    class_mesh_mapping = parse_mesh_files(args.mesh_path, label_mapping, clip_mapping)
    class_mesh_mapping = sort_by_clip_and_subsample(class_mesh_mapping, args.samples)

    # Dump the mapping into a pickle file
    nr_meshes = "all"
    if args.samples:
        nr_meshes = str(args.samples)

    out_path = f"./{args.dataset}_{nr_meshes}_instance_path.pkl"
    save_to_pickle(out_path, class_mesh_mapping)
