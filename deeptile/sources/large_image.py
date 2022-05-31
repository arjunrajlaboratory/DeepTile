import dask.array as da
import numpy as np
from dask import delayed
from deeptile import utils


def parse(image, image_shape, tile_size, overlap, slices):

    overlap_size = utils.calculate_overlap_size(np.array(tile_size), np.array(overlap))
    tiling = utils.calculate_tiling(np.array(image_shape), np.array(tile_size), overlap_size)
    tiling = tuple(tiling)
    tile_size = np.ceil(tile_size)
    overlap_size = np.floor(overlap_size)

    tiles = np.empty(shape=tiling, dtype=object)
    gys = []
    gxs = []
    heights = []
    widths = []

    tile_iterator = image.tileIterator(frame=slices,
                                       tile_size=dict(height=tile_size[0], width=tile_size[1]),
                                       tile_overlap=dict(y=overlap_size[0], x=overlap_size[1]))
    lazy_imread = delayed(imread)
    for tile_dict in tile_iterator:
        delayed_reader = lazy_imread(tile_dict)
        shape = (tile_dict['height'], tile_dict['width'])
        tiles[tile_dict['level_y'], tile_dict['level_x']] = da.from_delayed(delayed_reader, shape=shape, dtype=object)
        gys.append(tile_dict['gy'])
        gxs.append(tile_dict['gx'])
        heights.append(tile_dict['height'])
        widths.append(tile_dict['width'])

    gys = gys[::tiling[1]]
    gxs = gxs[:tiling[1]]
    heights = heights[::tiling[1]]
    widths = widths[:tiling[1]]

    v_tile_indices = np.cumsum((gys, heights), axis=0).T.astype(int)
    h_tile_indices = np.cumsum((gxs, widths), axis=0).T.astype(int)
    tile_indices = (v_tile_indices, h_tile_indices)

    v_border_indices = np.mean(v_tile_indices.ravel()[1:-1].reshape(-1, 2), axis=1).astype(int)
    v_border_indices = np.concatenate(([0], v_border_indices, [image_shape[0]]))
    h_border_indices = np.mean(h_tile_indices.ravel()[1:-1].reshape(-1, 2), axis=1).astype(int)
    h_border_indices = np.concatenate(([0], h_border_indices, [image_shape[1]]))
    border_indices = (v_border_indices, h_border_indices)

    return tiles, tiling, tile_indices, border_indices


def imread(tile_dict):

    tile = tile_dict['tile'][:, :, 0]
    tile_dict.release()

    return tile
