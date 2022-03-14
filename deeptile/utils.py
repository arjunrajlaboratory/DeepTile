import numpy as np


def calculate_tiling(axis_size, max_tile_size, overlap):

    tiling = (axis_size / max_tile_size - overlap) / (1 - overlap)
    tiling = np.ceil(tiling).astype(int)

    return tiling


def check_tiling(tiling, image_shape, max_tile_size, overlap):

    new_tiling = None

    if max_tile_size is not None:
        new_tiling = calculate_tiling(np.array(image_shape[-2]), np.array(max_tile_size), np.array(overlap))
        if tiling is not None:
            new_tiling = np.max(np.stack((tiling, new_tiling)), axis=0)

    new_tiling = tuple(new_tiling)

    return new_tiling


def calculate_tile_size(axis_size, tiling, overlap):

    tile_size = axis_size / (tiling - (tiling - 1) * overlap)

    return tile_size


def calculate_overlap_size(tile_size, overlap):

    overlap_size = tile_size * overlap

    return overlap_size


def calculate_indices_1d(axis_size, tiling, overlap):

    tile_size = calculate_tile_size(axis_size, tiling, overlap)

    li = np.arange(tiling) * tile_size * (1 - overlap)
    ri = np.append(li[:-1] + tile_size, axis_size)
    tile_indices = np.stack((li, ri), axis=1)
    tile_indices = np.rint(tile_indices).astype(int)

    border_indices = (np.arange(1, tiling) * (1 - overlap) + 0.5 * overlap) * tile_size
    border_indices = np.hstack([0, border_indices, axis_size])
    border_indices = np.rint(border_indices).astype(int)

    return tile_indices, border_indices


def calculate_indices(image_shape, tiling, overlap):

    v_tile_indices, v_border_indices = calculate_indices_1d(image_shape[-2], tiling[0], overlap[0])
    h_tile_indices, h_border_indices = calculate_indices_1d(image_shape[-1], tiling[1], overlap[1])

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


def cast_list_to_array(lst):

    ary = np.empty(shape=(len(lst), len(lst[0])), dtype=object)

    for i, sublst in enumerate(lst):
        for j, subary in enumerate(sublst):
            ary[i, j] = subary

    return ary
