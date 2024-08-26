#!/usr/bin/env python
# encoding: utf-8
from sys import platform

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

extra_link_args = []
if platform != "darwin":
    extra_link_args.append("-lgomp")

ext = Extension(
    "text3daug.RayTracerCython",
    sources=["./text3daug/raytracer/RayTracerCython.pyx"],
    language="c++",
    include_dirs=[np.get_include()],
    extra_compile_args=[
        "-std=c++14",
        "-O3",
        # "-fopenmp",
        "-msse3",
        "-fPIC",
        "-march=native",
    ],
)


setup(
    name="text3daug",
    version="0.0",
    description="Text3DAug for LiDAR pointclouds.",
    author="Laurenz Reichardt and Luca Uhr",
    url="https://github.com/CeMOS-IS",
    packages=find_packages(),
    install_requires=["numpy"],
    ext_modules=cythonize(ext, language_level=3),
)
