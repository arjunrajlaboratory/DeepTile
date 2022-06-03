import dask.array as da


def read(image):

    image = da.from_array(image)

    return image
