import dask.array as da
import numpy as np
from dask import delayed
from deeptile.core import utils


def parse(image, image_shape, tiling, tile_indices, slices):

    tiles = np.empty(tiling, dtype=object)
    sliced_image_shape = utils.calculate_sliced_shape(image_shape, slices)

    lazy_imread = delayed(imread)
    for i in range(tiling[0]):
        for j in range(tiling[1]):
            tile_index = np.stack((tile_indices[0][i], tile_indices[1][j]))
            tile_size = (tile_index[0, 1] - tile_index[0, 0], tile_index[1, 1] - tile_index[1, 0])
            tile_shape = (*sliced_image_shape[:-2], *tile_size)
            delayed_reader = lazy_imread(image, tile_index, slices)
            tiles[i, j] = da.from_delayed(delayed_reader, shape=tile_shape, dtype=object)

    return tiles


def imread(image, tile_index, slices):

    tile = image(tile_index, slices)

    return tile
