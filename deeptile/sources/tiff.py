import dask.array as da
import rasterio
from dask import delayed
from tifffile import imread


def read(image):

    with rasterio.open(image) as i:
        shape = i.shape
        dtype = i.dtypes[0]

    lazy_imread = delayed(imread)
    delayed_reader = lazy_imread(image)
    image = da.from_delayed(delayed_reader, shape=shape, dtype=dtype)

    return image
