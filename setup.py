#  Copyright (c) 2024 Andrew McDonald, University of Cambridge. All Rights Reserved.

from setuptools import setup, find_packages

# See e.g. https://hynek.me/articles/testing-packaging/
# for why a src/ directory is best practice

setup(
    name="pcg_analysis",
    version="0.0.1",
    url="",
    license="",
    author="am2234",
    author_email="am2234@cam.ac.uk",
    description="Acoustic PCG",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
