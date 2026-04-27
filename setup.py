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

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "detectron2._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


def get_model_zoo_configs() -> List[str]:
    """Return a list of configs to include in package.

    Note: only YAML configs are included; any .py-based configs are excluded
    since they are not needed for model zoo usage and add unnecessary weight
    to the installed package.
    """
    source_configs_dir = path.join(path.dirname(path.realpath(__file__)), "configs")
    configs = glob.glob(path.join(source_configs_dir, "**", "*.yaml"), recursive=True)
    configs = [path.relpath(c, path.dirname(path.realpath(__file__))) for c in configs]
    return configs


setup(
    name="detectron2",
    version=get_version(),
    author="FAIR",
    url="https://github.com/facebookresearch/detectron2",
    description="Detectron2 is Facebook AI Research's next generation library"
    " that provides state-of-the-art detection and segmentation algorithms.",
    packages=find_packages(exclude=("configs", "tests", "*.tests", "*.tests.*", "tests.*")),
    python_requires=">=3.7",
    install_requires=[
        # Do not add opencv here. Just like pytorch, user should install
        # temporary dependencies themselves.
        "termcolor>=1.1",
        "Pillow>=7.1",  # or use Pillow-SIMD for faster performance
        "yacs>=0.1.8",
        "tabulate",
        "cloudpickle",
        "matplotlib",
        "mock",
        "tqdm>4.29.0",
        "tensorboard",
        "fvcore>=0.1.5,<0.1.6",
        "iopath>=0.1.7,<0.1.10",
        "pycocotools>=2.0.2",
        "omegaconf>=2.1,<2.4",
        "hydra-core>=1.1",
        "black",
        "packaging",
    ],
    extras_require={
        "all": [
            "fairscale",
            "timm",
            "scipy>1.5.1",
            "shapely",
            "pygments>=2.2",
            "psutil",
            "panopticapi @ https://github.com/cocodataset/panopticapi/archive/master.zip",
        ],
        "dev": [
            "flake8==3.8.1",
            "isort==4.3.21",
            "flake8-bugbear",
            "flake8-comprehensions",
        ],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    package_data={"detectron2.model_zoo": get_model_zoo_configs()},
)
