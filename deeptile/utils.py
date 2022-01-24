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

    stitch_indices = (np.arange(1, n_blocks) * (1 - overlap) + 0.5 * overlap) * block_size
    stitch_indices = np.hstack([0, stitch_indices, axis_size])

    indices = np.rint(tile_indices).astype(int), np.rint(stitch_indices).astype(int)

    return indices


def calculate_indices_2d(shape, n_blocks, overlap):

    v_tile_indices, v_stitch_indices = calculate_indices_1d(shape[-2], n_blocks[0], overlap[0])
    h_tile_indices, h_stitch_indices = calculate_indices_1d(shape[-1], n_blocks[1], overlap[1])

    tile_indices = (v_tile_indices, h_tile_indices)
    stitch_indices = (v_stitch_indices, h_stitch_indices)

    return tile_indices, stitch_indices


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


def read_nd2(path):

    from nd2reader import ND2Reader

    file = open(path, 'rb')
    image = ND2Reader(file)
    axes = dict(image.sizes)
    if 'v' in axes.keys():
        image.iter_axes = 'v'
        axes.pop('v')
    if axes['t'] == 1:
        axes.pop('t')
    axes = ''.join(reversed(list(axes.keys())))
    image.bundle_axes = axes

    metadata = image.parser._raw_metadata

    return image, metadata


def parse_nd2(image, metadata, overlap, slices):

    if 'v' not in image.iter_axes:
        shape = image.frame_shape
        tile = np.array(image[0])
        tiles = np.empty((1, 1), dtype=object)
        tiles[0, 0] = tile
        return tiles, shape

    positions = metadata.image_metadata[b'SLxExperiment'][b'uLoopPars'][b'Points'][b'']
    coords = np.array([(position[b'dPosX'], position[b'dPosY']) for position in positions]).T

    rotation = np.array([[
        metadata.image_metadata_sequence[b'SLxPictureMetadata'][b'dStgLgCT11'],
        metadata.image_metadata_sequence[b'SLxPictureMetadata'][b'dStgLgCT12']
    ], [
        metadata.image_metadata_sequence[b'SLxPictureMetadata'][b'dStgLgCT21'],
        metadata.image_metadata_sequence[b'SLxPictureMetadata'][b'dStgLgCT22']
    ]])

    x, y = rotation @ coords

    if coords[0, 1] - coords[0, 0] > 0:
        x = -x
    if coords[1, -1] - coords[1, 0] > 0:
        y = -y

    x_ndim = round(np.ptp(x) / abs(x[0] - x[1])) + 1
    y_ndim = round(np.ptp(y) / abs(x[0] - x[1])) + 1

    j = np.rint((x - min(x)) / (np.ptp(x) / (x_ndim - 1))).astype(int)
    i = np.rint((y - min(y)) / (np.ptp(y) / (y_ndim - 1))).astype(int)

    if overlap is None:
        x_overlaps = np.empty(0)
        y_overlaps = np.empty(0)
        for col in np.unique(j):
            x_overlaps = np.append(x_overlaps, np.mean(x[np.where(j == col)]))
        for row in np.unique(i):
            y_overlaps = np.append(y_overlaps, np.mean(y[np.where(i == row)]))
        x_overlap = round(1 - (np.mean(np.diff(x_overlaps)) /
                               image.metadata['pixel_microns']) / image.metadata['width'], 2)
        y_overlap = round(1 - (np.mean(np.diff(y_overlaps)) /
                               image.metadata['pixel_microns']) / image.metadata['height'], 2)

        if not ((0 < x_overlap < 1) & (0 < y_overlap < 1)):
            raise RuntimeError("Failed to determine overlap percentage from metadata.")

        overlap = (y_overlap, x_overlap)

    width = round(image.metadata['width'] * (x_ndim - (x_ndim - 1) * overlap[1]))
    height = round(image.metadata['height'] * (y_ndim - (y_ndim - 1) * overlap[0]))

    shape = None
    tiles = np.empty(shape=(y_ndim, x_ndim), dtype=object)

    for n in image.metadata['fields_of_view']:

        tile = np.array(image[n])
        if tile.ndim > 2:
            tile = tile[slices]
        if shape is None:
            shape = (*tile.shape[:-2], height, width)
        tile = tile.reshape(-1, *tile.shape[-2:])
        tiles[i[n], j[n]] = tile

    return tiles, overlap, shape
