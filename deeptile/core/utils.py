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


def calculate_sliced_shape(shape, slices):

    sliced_shape = list(shape)

    for i, s in enumerate(slices):

        if isinstance(s, int):
            s = slice(s, s + 1)
        elif s is None:
            s = slice(None)

        start, stop, step = s.indices(sliced_shape[i])
        sliced_shape[i] = len(range(start, stop, step))

    sliced_shape = tuple(sliced_shape)

    return sliced_shape


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


def array_pad(ary, padding, axis=0, **kwargs):

    pad_width = [(0, 0)] * ary.ndim
    pad_width[axis] = (0, padding)

    return np.pad(ary, pad_width, **kwargs)


def cast_list_to_array(lst):

    ary = np.empty(len(lst), dtype=object)

    for i, subary in enumerate(lst):
        ary[i] = subary

    return ary


def cast_list_to_array_2d(lst):

    ary = np.empty((len(lst), len(lst[0])), dtype=object)

    for i, sublst in enumerate(lst):
        for j, subary in enumerate(sublst):
            ary[i, j] = subary

    return ary


def tile_image(tile, image):

    tile_index = tile

    image_slice = np.s_[..., tile_index[0, 0]:tile_index[0, 1], tile_index[1, 0]:tile_index[1, 1]]
    tiled_image = image[image_slice]

    return tiled_image


def tile_coords(tile, coords):

    tile_index = tile

    if isinstance(coords, Sequence) or (coords.dtype is np.dtype('O')):
        batch_axis = True
        n_batches = len(coords)
    else:
        coords = (coords,)
        batch_axis = False
        n_batches = 1

    tiled_coords = np.empty(n_batches, dtype=object)

    for n in range(n_batches):

        coord = coords[n]

        s = (tile_index[0, 0] < coord[:, 0]) & (coord[:, 0] < tile_index[0, 1]) & \
            (tile_index[1, 0] < coord[:, 1]) & (coord[:, 1] < tile_index[1, 1])
        tiled_coords[n] = coord[s] - tile_index[:, 0]

    if not batch_axis:
        tiled_coords = tiled_coords[0]

    return tiled_coords
