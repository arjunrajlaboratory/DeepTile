import dask.array as da
import tifffile
from dask import delayed


def read(image, dask):

    if dask:
        delayed_reader = delayed(tifffile.memmap)(image)
        image = tifffile.memmap(image)
        image = da.from_delayed(delayed_reader, shape=image.shape, dtype=image.dtype)
    else:
        image = tifffile.imread(image)

    return image
