import dask.array as da
from dask import delayed
from tifffile import imread


def read(image):

    lazy_imread = delayed(imread)
    delayed_reader = lazy_imread(image)

    array = imread(image)
    image = da.from_delayed(delayed_reader, shape=array.shape, dtype=array.dtype)

    return image
