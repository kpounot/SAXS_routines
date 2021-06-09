import os, sys

import subprocess

from setuptools import setup, Extension
from distutils.dist import Distribution

try:
    import versioneer
except ImportError:
    os.system("python3 -m pip install versioneer")
    import versioneer

filePath = os.path.abspath(__file__)
dirPath = os.path.dirname(filePath)


with open(dirPath + "/README.rst") as f:
    description = f.read()


packagesList = [
    "saxs_routines",
]

setup(
    name="SAXS_routines",
    version=versioneer.get_version(),
    cmdclass={**versioneer.get_cmdclass()},
    description="Python package for analysis of small-angle scattering data",
    long_description=description,
    long_description_content_type="text/x-rst",
    platforms=["Windows", "Linux", "Mac OS X"],
    author="Kevin Pounot",
    author_email="kpounot@hotmail.fr",
    url="https://github.com/kpounot/SAXS_routines",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU General Public "
        "License v3 or later (GPLv3+)",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    packages=packagesList,
    package_dir={"SAXS_routines": dirPath + "/SAXS_routines"},
    install_requires=[
        "scipy",
        "numpy",
        "matplotlib",
        "PyQt5==5.14",
        "h5py",
    ],
)
