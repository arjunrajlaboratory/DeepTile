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

    theta = metadata.image_metadata_sequence[b'SLxPictureMetadata'][b'dAngle']
    cos, sin = np.cos(theta), np.sin(theta)
    r = np.array(((cos, -sin), (sin, cos)))
    x, y = np.rint(np.dot(r, coords))

    x_dim = round(np.ptp(x) / abs(x[0] - x[1])) + 1
    y_dim = round(np.ptp(y) / abs(x[0] - x[1])) + 1

    x_scaled = np.rint((x - min(x)) / (np.ptp(x) / (x_dim - 1))).astype(int)
    y_scaled = np.rint((y - min(y)) / (np.ptp(y) / (y_dim - 1))).astype(int)
    y_scaled = y_scaled.max() - y_scaled

    width = round(image.metadata['width'] * (x_dim - (x_dim - 1) * overlap[1]))
    height = round(image.metadata['height'] * (y_dim - (y_dim - 1) * overlap[0]))

    shape = None
    tiles = np.empty(shape=(y_dim, x_dim), dtype=object)

    for i in image.metadata['fields_of_view']:

        tile = np.array(image[i])
        if tile.ndim > 2:
            tile = tile[slices]
        if shape is None:
            shape = (*tile.shape[:-2], height, width)
        tile = tile.reshape(-1, *tile.shape[-2:])
        tiles[y_scaled[i], x_scaled[i]] = tile

    return tiles, shape
