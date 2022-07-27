import dask.array as da


def read(image, dask):

    if dask:
        image = da.from_array(image)

    return image
