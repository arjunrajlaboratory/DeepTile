import numpy as np


def axis_slice(ary, axis, start, end, step=1):

    return ary[(slice(None),) * (axis % ary.ndim) + (slice(start, end, step),)]


def array_split(ary, indices, axis):

    sub_arys = [axis_slice(ary, axis, *i) for i in indices]

    return sub_arys


def calculate_array_split_indices(axis_size, n_blocks, overlap):

    block_size = round(axis_size / (n_blocks - (n_blocks - 1) * overlap))
    step_size = round(block_size * (1 - overlap))

    li = np.arange(n_blocks) * step_size
    ri = np.append(li[:-1] + block_size, axis_size - 1)
    indices = np.stack((li, ri), axis=1)

    return indices


def array_split_2d(ary, n_blocks, overlap):

    hindices = calculate_array_split_indices(ary.shape[-2], n_blocks[0], overlap[0])
    vindices = calculate_array_split_indices(ary.shape[-1], n_blocks[1], overlap[1])
    indices = np.array([(*i, *j) for i in hindices for j in vindices]).reshape(*n_blocks, -1)

    sub_arys = array_split(ary, hindices, -2)
    sub_arys = [array_split(sub_ary, vindices, -1) for sub_ary in sub_arys]

    return sub_arys, indices


def get_tiles(image, n_blocks, overlap):

    tiles = np.empty(shape=n_blocks, dtype=object)
    tiles[:], indices = array_split_2d(image, n_blocks, overlap)

    return tiles, indices


def segment_tile(tile, algorithm, parameters):

    if algorithm == 'Cellori':
        from cellori import Cellori
        return Cellori(tile).segment(**parameters)


def segment_image(image, n_blocks, overlap, algorithm, parameters):

    tiles, indices = get_tiles(image, n_blocks, overlap)
    masks = np.zeros_like(tiles)

    for index, tile in np.ndenumerate(tiles):
        mask = segment_tile(tile, algorithm, parameters)[0]
        masks[index] = mask

    return masks, indices


def stitch_masks(masks, indices, shape):

    stitched_mask = np.zeros(shape)

    total_count = 0
    for index, mask in np.ndenumerate(masks):
        il, ir, jl, jr = indices[index]
        count = mask.max()
        mask[mask > 0] += total_count
        total_count += count
        stitched_mask[il:ir, jl:jr] += mask

    return stitched_mask


def deeptile(image, n_blocks=(2, 2), overlap=(0.1, 0.1), algorithm='Cellori', parameters=dict()):

    masks, indices = segment_image(image, n_blocks, overlap, algorithm, parameters)
    mask = stitch_masks(masks, indices, image.shape)

    return mask
