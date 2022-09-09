from setuptools import setup, find_packages

VERSION = "2.0.2"
DESCRIPTION = "DeepTile"
LONG_DESCRIPTION = "Large image tiling and stitching algorithm for deep learning libraries."

setup(
    name="deeptile",
    version=VERSION,
    author="William Niu",
    author_email="<wniu721@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['dask', 'nd2', 'numpy', 'scikit-image', 'tifffile'],
    keywords=["segmentation", "stitching"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
