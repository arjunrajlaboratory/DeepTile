from dask_image.imread import imread


def read(image):

    image = imread(image)

    return image
