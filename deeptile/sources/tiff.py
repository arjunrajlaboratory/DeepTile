import dask.array as da
from dask import delayed
from tifffile import memmap


def read(image):

    delayed_reader = delayed(memmap)(image)
    image = memmap(image)
    image = da.from_delayed(delayed_reader, shape=image.shape, dtype=image.dtype)

    return image
