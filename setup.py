#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "detectron2", "__init__.py")
    init_py = open(init_py_path).read()
    version_line = [line.strip() for line in init_py.split("\n") if line.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip('"\'')
    return version


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "detectron2", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    from torch.utils.cpp_extension import ROCM_HOME

    is_rocm_pytorch = (
        True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    )
    if is_rocm_pytorch:
        assert torch_ver >= [1, 8], "ROCM support requires PyTorch >= 1.8!"

    # common code between cuda and rocm platforms,
    # for hipify version hipify_python.py is used.
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources
    extension = CppExtension

    # Using -O0 instead of -O2/-O3 to minimize compile time during local development.
    # Note: -O0 disables all optimizations; fine for iterating locally, not for benchmarking.
    # Personal note: switch cxx to -O2 when running actual benchmarks or profiling.
    extra_compile_args = {"cxx": ["-O0", "-std=c++17"]}
    define_macros = []

    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-O0",  # minimized for fast local builds; bump to -O2 before any perf testing
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        if not is_rocm_pytorch:
            extra_compile_args["nvcc"] += ["-std=c++17"]
            # Targeting Ampere GPU (RTX 3090) on my local machine — speeds up nvcc compilation
            # by avoiding generating code for older architectures.
            extra_compile_args["nvcc"] += ["-gencode", "arch=compute_86,code=sm_86"]
            # Also emit PTX for forward compatibility with future GPU architectures.
            extra_compile_args["nvcc"] += ["-gencode", "arch=compute_86,code=compute_86"]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "detectron2._C",
            sources,
            include_dirs=include_dirs,
