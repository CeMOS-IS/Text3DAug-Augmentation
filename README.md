# Text3DAug – Prompted Instance Augmentation for LiDAR Perception

:fire: Accepted at IROS 2024 (oral) :fire:

[Laurenz Reichardt](https://scholar.google.com/citations?user=cBhzz5kAAAAJ&hl=en), Luca Uhr, [Oliver Wasenmüller](https://scholar.google.de/citations?user=GkHxKY8AAAAJ&hl=de) \
**[CeMOS - Research and Transfer Center](https://www.cemos.hs-mannheim.de/ "CeMOS - Research and Transfer Center"), [University of Applied Sciences Mannheim](https://www.english.hs-mannheim.de/the-university.html "University of Applied Sciences Mannheim")**

[![arxiv.org](https://img.shields.io/badge/cs.CV-arXiv%3A0000.0000-B31B1B.svg)](https://arxiv.org/)
[![cite-bibtex](https://img.shields.io/badge/Cite-BibTeX-1f425f.svg)](#citing)
[![download meshes](https://img.shields.io/badge/Download-Meshes-b3a017.svg)](https://clousi.hs-mannheim.de/index.php/s/4qknpPB6PjPWEg9)

## About - Instance Augmentation

This repository contains the Text3DAugmentation which inserts object meshes into pointclouds as instances. This repository is meant to be a
plug-and-play augmentation, which can be adapted depending on the specifics of your training pipeline.
The evaluation was performed using [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) for detection.
[Cylinder3D](https://github.com/L-Reichardt/Cylinder3D-updated-CUDA) and [SPVCNN](https://github.com/yanx27/2DPASS) were used for segmentation.

For the generation of meshes, please refer to the following [repository](https://github.com/CeMOS-IS/Text3DAug-Generation).

## Installation

Install Text3DAug locally through

```
pip install -e .
```

If you use visualization components e.g. for debugging purposes or running the example, please install open3d:

```
pip install open3d==0.13.0
```

## How to Use

Please refer to *./examples/example.py*.
Download a KITTI mini dataset from [here](https://github.com/PRBonn/lidar_transfer/blob/main/minimal.zip) and place it into the *./examples* folder.

```
wget -P examples/ https://github.com/PRBonn/lidar_transfer/raw/main/minimal.zip
```

```
unzip examples/minimal.zip -d examples/ && rm examples/minimal.zip 
```

Example meshes can be downloaded from [here](https://clousi.hs-mannheim.de/index.php/s/4qknpPB6PjPWEg9).
Generate a pickle file for the meshes using *pickle_file.py*.

## Adapting to your Data

You will have to modify either the augmentation or your pipeline. For example, OpenPCDet uses a dictionary to move data and uses class strings instead of integers
in the ground truth bounding boxes.

:warning: The augmentation for detection returns new bounding boxes only. These need to be added to your ground truth.

The augmentation operates on the following components:

1. A dictionary mapping a List of *.obj*-mesh file paths to a class integer.
In the example, a pickle file is created based on the folders from the [generation pipeline](https://github.com/CeMOS-IS/Text3DAug-Generation).
However, a pickle file isnt necessary. You can use any dictionary for the augmentation.
If you do create a pickle file using *pickle_file.py* the folder structure should be like follows:

```
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
```

2. A dataset mapping in *./examples/configs*. The *.yaml* file maps the class string to a class integer. This has to match your dataset and task.
Adjust as necessary.
3. The hardware specifications of the LiDAR scanner in *./examples/configs* used for raytracing. Adjust to your sensor and setup height as necessary.
4. A remission or intensity file e.g. *./examples/remission_kitti.txt*. These are just the remission / intensity values of your data and will be randomly
sampled during augmentation. These values depend on your sensor and you will have to create your own *.txt* file if you use different datasets.

