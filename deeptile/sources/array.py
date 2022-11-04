import dask.array as da


def read(image, dask):

    if dask and (not isinstance(image, da.Array)):
        image = da.from_array(image)

    return image
