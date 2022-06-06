import numpy as np
from deeptile import deeptile
from pathlib import Path


def load(image):

    dt = None

    if isinstance(image, np.ndarray):
        dt = from_array(image)
    elif Path(image).is_file():
        if image.endswith(('.tif', '.tiff')):
            dt = from_tiff(image)
        elif image.endswith('.nd2'):
            dt = from_nd2(image)
    else:
        raise ValueError("Invalid image.")

    return dt


def from_array(image):

    from deeptile.sources import array
    image = array.read(image)
    dt = deeptile.DeepTileArray(image)

    return dt


def from_large_image(image):

    dt = deeptile.DeepTileLargeImage(image)

    return dt


def from_nd2(image):

    from deeptile.sources import nd2
    image = nd2.read(image)
    dt = deeptile.DeepTileND2(image)

    return dt


def from_tiff(image):

    from deeptile.sources import tiff
    image = tiff.read(image)
    dt = deeptile.DeepTileArray(image)

    return dt
