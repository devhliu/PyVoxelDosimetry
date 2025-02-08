from setuptools import setup, find_packages

setup(
    name="pyvoxeldosimetry",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "nibabel>=3.2.0",
        "cupy-cuda11x>=10.0.0",  # Adjust CUDA version as needed
        "torch>=1.9.0",
        "SimpleITK>=2.1.0",
        "nnunetv2>=2.1.1",  # Added for segmentation
    ],
    package_data={
        'pyvoxeldosimetry': [
            'data/dose_kernels/*/*.dat',  # Include dose kernel data
            'data/pretrained_models/*/*.model',  # Include pretrained models
        ]
    },
    author="devhliu, claude 3.5 sonnet",
    author_email="huiliu.liu@gmail.com",
    description="A package for voxel-level internal dosimetry calculations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/devhliu/pyvoxeldosimetry",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
)