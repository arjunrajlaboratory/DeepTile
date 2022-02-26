import numpy as np


def axis_slice(ary, axis, start, end, step=1):

    return ary[(slice(None),) * (axis % ary.ndim) + (slice(start, end, step),)]


def array_split(ary, indices, axis):

    sub_arys = [axis_slice(ary, axis, *i) for i in indices]

    return sub_arys


def calculate_indices_1d(axis_size, n_blocks, overlap):

    block_size = axis_size / (n_blocks - (n_blocks - 1) * overlap)

    li = np.arange(n_blocks) * block_size * (1 - overlap)
    ri = np.append(li[:-1] + block_size, axis_size)
    tile_indices = np.stack((li, ri), axis=1)

    border_indices = (np.arange(1, n_blocks) * (1 - overlap) + 0.5 * overlap) * block_size
    border_indices = np.hstack([0, border_indices, axis_size])

    indices = np.rint(tile_indices).astype(int), np.rint(border_indices).astype(int)

    return indices


def calculate_indices(shape, n_blocks, overlap):

    v_tile_indices, v_border_indices = calculate_indices_1d(shape[-2], n_blocks[0], overlap[0])
    h_tile_indices, h_border_indices = calculate_indices_1d(shape[-1], n_blocks[1], overlap[1])

    tile_indices = (v_tile_indices, h_tile_indices)
    border_indices = (v_border_indices, h_border_indices)

    return tile_indices, border_indices


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


def array_split_2d(ary, indices):

    sub_arys = array_split(ary, indices[0], -2)
    sub_arys = [array_split(sub_ary, indices[1], -1) for sub_ary in sub_arys]

    return sub_arys


def remove_object(mask, objects):

    objects = np.unique(objects)
    objects = objects[objects > 0]
    mask[np.isin(mask, objects)] = 0

    return mask


def clear_border(mask, i, j):

    for row in i:
        mask = remove_object(mask, mask[row])

    for col in j:
        mask = remove_object(mask, mask[:, col])

    return mask
