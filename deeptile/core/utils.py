import numpy as np
from collections.abc import Sequence
from dask.array import Array


def to_tuple(obj):

    if isinstance(obj, str) or (not isinstance(obj, Sequence)):
        obj = (obj, )

    return obj


def compute_dask(obj):

    if isinstance(obj, Array):
        obj = obj.compute()

    return obj


def calculate_tiling(axis_size, tile_size, overlap_size):

    tiling = (axis_size - overlap_size) / (tile_size - overlap_size)
    tiling = np.ceil(tiling).astype(int)

    return tiling


def calculate_overlap_size(tile_size, overlap):

    overlap_size = tile_size * overlap
    overlap_size = np.rint(overlap_size).astype(int)

    return overlap_size


def calculate_indices_1d(axis_size, tile_size, overlap):

    overlap_size = calculate_overlap_size(tile_size, overlap)
    tiling = calculate_tiling(axis_size, tile_size, overlap_size)

    li = np.arange(tiling) * (tile_size - overlap_size)
    ri = np.append(li[:-1] + tile_size, axis_size)
    tile_indices = np.stack((li, ri), axis=1)

    border_indices = np.arange(1, tiling) * (tile_size - overlap_size) + round(0.5 * overlap_size)
    border_indices = np.hstack([0, border_indices, axis_size])

    return tiling, tile_indices, border_indices


def calculate_indices(image_shape, tile_size, overlap):

    v_tiling, v_tile_indices, v_border_indices = calculate_indices_1d(image_shape[-2], tile_size[0], overlap[0])
    h_tiling, h_tile_indices, h_border_indices = calculate_indices_1d(image_shape[-1], tile_size[1], overlap[1])

    tiling = (v_tiling, h_tiling)
    tile_indices = (v_tile_indices, h_tile_indices)
    border_indices = (v_border_indices, h_border_indices)

    return tiling, tile_indices, border_indices


def get_nonempty_indices(tiles):

    nonempty_indices = []

    for index, tile in np.ndenumerate(tiles):
        if tile is not None:
            nonempty_indices.append(index)

    nonempty_indices = tuple(nonempty_indices)

    return nonempty_indices


def axis_take(ary, axis, index):

    return ary[(slice(None), ) * (axis % ary.ndim) + (index, )]


def axis_slice(ary, axis, start, end, step=1):

    return ary[(slice(None), ) * (axis % ary.ndim) + (slice(start, end, step), )]


def array_split(ary, indices, axis):

    sub_arys = [axis_slice(ary, axis, *i) for i in indices]

    return sub_arys


def array_split_2d(ary, indices):

    sub_arys = array_split(ary, indices[0], -2)
    sub_arys = [array_split(sub_ary, indices[1], -1) for sub_ary in sub_arys]

    return sub_arys


def array_pad(ary, padding, axis=0):

    pad_width = [(0, 0)] * ary.ndim
    pad_width[axis] = (0, padding)

    return np.pad(ary, pad_width)


def cast_list_to_array(lst):

    ary = np.empty((len(lst), len(lst[0])), dtype=object)

    for i, sublst in enumerate(lst):
        for j, subary in enumerate(sublst):
            ary[i, j] = subary

    return ary


def pad_tiles(tiles, tile_size, tile_indices):

    tile_padding = (tile_size[0] - (tile_indices[0][-1, 1] - tile_indices[0][-1, 0]),
                    tile_size[1] - (tile_indices[1][-1, 1] - tile_indices[1][-1, 0]))

    if tile_padding[0] > 0:
        for i, tile in enumerate(tiles[-1]):
            tiles[-1, i] = array_pad(tile, tile_padding[0], -2)

    if tile_padding[1] > 0:
        for i, tile in enumerate(tiles[:, -1]):
            tiles[i, -1] = array_pad(tile, tile_padding[1], -1)

    return tiles


def unpad_tiles(tiles):

    tile_size = tiles.nonempty_tiles[0].shape[-2:]
    tile_indices = tiles.tile_indices
    tile_padding = (tile_size[0] - (tile_indices[0][-1, 1] - tile_indices[0][-1, 0]),
                    tile_size[1] - (tile_indices[1][-1, 1] - tile_indices[1][-1, 0]))

    tiles = tiles.copy()

    if tile_padding[0] > 0:
        for i, tile in enumerate(tiles[-1]):
            tiles[-1, i] = tile[..., :-tile_padding[0], :]
    if tile_padding[1] > 0:
        for i, tile in enumerate(tiles[:, -1]):
            tiles[i, -1] = tile[..., :-tile_padding[1]]

    return tiles


def update_tiles(tiles, index, tile, batch_axis, output_type):

    for i_output in range(len(tiles)):

        ts = tiles[i_output]
        t = tile[i_output]
        otype = output_type[i_output]

        if batch_axis:

            current_tile = ts[index]
            new_tile = None

            if otype == 'tiled_image':

                new_tile = t[None]

            elif otype == 'tiled_coords':

                new_tile = np.empty(1, dtype=object)
                new_tile[0] = t

            if isinstance(current_tile, np.ndarray):
                ts[index] = np.concatenate((current_tile, new_tile), 0)
            else:
                ts[index] = new_tile

        else:

            ts[index] = t

    return tiles


def tile_coords(coords, tile):

    tile_index = tile

    s = (tile_index[0, 0] < coords[:, 0]) & (coords[:, 0] < tile_index[0, 1]) & \
        (tile_index[1, 0] < coords[:, 1]) & (coords[:, 1] < tile_index[1, 1])
    coords = coords[s] - tile_index[:, 0]

    return coords
