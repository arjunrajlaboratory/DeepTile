from dask.array import Array
from deeptile import deeptile
from numpy import ndarray
from pathlib import Path


def load(image, dask=True, link_data=True):

    """ Load image into a DeepTile object.

    Parameters
    ----------
        image
            An object or path of an image.
        dask : bool, optional, default True
            Whether to use Dask for lazy loading and processing.
        link_data : bool, optional, default True
            Whether to link input and output data to the profile and job objects. Set to ``False`` to reduce memory
            usage.

    Returns
    -------
        dt : DeepTile
            DeepTile object.

    Raises
    ------
        ValueError
            If ``image`` has an unsupported file type.
        ValueError
            If ``image`` is invalid.
    """

    if isinstance(image, Array):
        dask = True
        dt = from_array(image, dask)
    elif isinstance(image, ndarray):
        dt = from_array(image, dask)
    elif Path(image).is_file():
        if image.endswith(('.tif', '.tiff')):
            dt = from_tiff(image, dask)
        elif image.endswith('.nd2'):
            dt = from_nd2(image)
        else:
            raise ValueError('unsupported file type.')
    else:
        raise ValueError("invalid image.")

    dt.dask = dask
    dt.link_data = link_data

    return dt


def from_array(image, dask):

    """ Create a DeepTileArray object from an array.

    Parameters
    ----------
        image : array_like
            An array-like object of an image.
        dask : bool, optional, default True
            Whether to use Dask for lazy loading and processing.

    Returns
    -------
        dt : DeepTileArray
            DeepTileArray object.
    """

    from deeptile.sources import array
    image = array.read(image, dask)
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
    image_sizes, axes_order = nd2.read(image)
    dt = deeptile.DeepTileND2(image)
    dt.axis_sizes = image_sizes
    dt.axis_order = axes_order

    return dt


def from_tiff(image, dask):

    """ Create a DeepTileArray object from a TIFF file.

    Parameters
    ----------
        image : str
            Path to a TIFF file.
        dask : bool, optional, default True
            Whether to use Dask for lazy loading and processing.

    Returns
    -------
        dt : DeepTileArray
            DeepTileArray object.
    """

    from deeptile.sources import tiff
    image = tiff.read(image, dask)
    dt = deeptile.DeepTileArray(image)

    return dt
