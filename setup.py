import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fairmot",
    version="0.1.0",
    author="Yifu Zhang",
    description="A simple baseline for one-shot multi-object tracking.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ifzhang/FairMOT",
    packages=["fairmot"],
    package_dir={"fairmot": "src/lib"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "cython",
        "cython-bbox",
        "matplotlib",
        "motmetrics",
        "numba",
        "opencv-python",
        "openpyxl",
        "Pillow",
        "lap",
        "progress",
        "scipy",
        "torch>=1.2.0",
        "torchvision>=0.4.0",
        "yacs",
    ],
    extras_require={"tensorboard": ["tensorboardX"]},
)
