import numpy as np


def axis_slice(ary, axis, start, end, step=1):

    return ary[(slice(None),) * (axis % ary.ndim) + (slice(start, end, step),)]


def array_pad(ary, tile_size):

    padding = (ary.ndim - 2) * ((0, 0), ) + ((0, tile_size[0] - ary.shape[-2]), (0, tile_size[1] - ary.shape[-1]))
    ary = np.pad(ary, padding)

    return ary


def array_split(ary, indices, axis):

    sub_arys = [axis_slice(ary, axis, *i) for i in indices]

    return sub_arys


def array_split_2d(ary, indices):

    sub_arys = array_split(ary, indices[0], -2)
    sub_arys = [array_split(sub_ary, indices[1], -1) for sub_ary in sub_arys]

    return sub_arys


def cast_list_to_array(lst):

    ary = np.empty(shape=(len(lst), len(lst[0])), dtype=object)

    for i, sublst in enumerate(lst):
        for j, subary in enumerate(sublst):
            ary[i, j] = subary

    return ary


def pad_tiles(tiles, tile_size):

    if tiles[-1, 0].shape[-2] < tile_size[0]:
        for i, tile in enumerate(tiles[-1]):
            tiles[-1, i] = array_pad(tile, tile_size)
    if tiles[0, -1].shape[-1] < tile_size[1]:
        for i, tile in enumerate(tiles[:-1, -1]):
            tiles[i, -1] = array_pad(tile, tile_size)

    return tiles


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


def calculate_stitch_indices(tiles, tile_indices, border_indices):

    stitch_indices = dict()

    for (n_i, n_j), tile in np.ndenumerate(tiles):

        if tile is not None:
            i_image = border_indices[0][n_i:n_i + 2]
            j_image = border_indices[1][n_j:n_j + 2]
            i = i_image - tile_indices[0][n_i, 0]
            j = j_image - tile_indices[1][n_j, 0]
            stitch_indices[(n_i, n_j)] = (i_image, j_image, i, j)

    return stitch_indices
