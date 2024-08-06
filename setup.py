import os
from setuptools import setup


setup(
    name="qcore",
    version="1.2",
    packages=["qcore"],
    url="https://github.com/ucgmsim/qcore",
    description="QuakeCoRE Library",
    package_data={
        "qcore": [os.path.join("configs", "*.json"), "data/*", "data/*/*", "data/*/*/*"]
    },
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy>=0.16",
        "dataclasses",
        "alphashape",
        "descartes",
        "pandas",
        "pyproj",
        "shapely",
        "numba",
    ],
)
