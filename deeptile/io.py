import numpy as np
from deeptile import deeptile
from pathlib import Path


def load(image, link_data=True):

    """ Load image into a DeepTile object.

    Parameters
    ----------
        image
            An object or path of an image.
        link_data : bool, optional, default True
            Whether to link input and output data to the profile and job objects. Set to ``False`` to reduce memory
            usage.

    Returns
    -------
        dt : DeepTile
            DeepTile object.
    """

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

    dt.link_data = link_data

    return dt


def from_array(image):

    """ Create a DeepTileArray object from an array.

    Parameters
    ----------
        image : array_like
            An array-like object of an image.

    Returns
    -------
        dt : DeepTileArray
            DeepTileArray object.
    """

    from deeptile.sources import array
    image = array.read(image)
    dt = deeptile.DeepTileArray(image)

    return dt


def from_large_image(image):

    """ Create a DeepTileLargeImage object from a large_image tile source.

    Parameters
    ----------
        image : large_image tile source
            A large_image tile source.

    Returns
    -------
        dt : DeepTileLargeImage
            DeepTileLargeImage object.
    """

    dt = deeptile.DeepTileLargeImage(image)

    return dt


def from_nd2(image):

    """ Create a DeepTileND2 object from an ND2 file.

    Parameters
    ----------
        image : str
            Path to an ND2 file.

    Returns
    -------
        dt : DeepTileND2
            DeepTileND2 object.
    """

    from deeptile.sources import nd2
    image, image_sizes, axes_order = nd2.read(image)
    dt = deeptile.DeepTileND2(image)
    dt.image_sizes = image_sizes
    dt.axes_order = axes_order

    return dt


def from_tiff(image):

    """ Create a DeepTileArray object from a TIFF file.

    Parameters
    ----------
        image : str
            Path to a TIFF file.

    Returns
    -------
        dt : DeepTileArray
            DeepTileArray object.
    """

    from deeptile.sources import tiff
    image = tiff.read(image)
    dt = deeptile.DeepTileArray(image)

    return dt
