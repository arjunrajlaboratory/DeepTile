import nd2
import numpy as np
from deeptile import utils
from nd2reader import ND2Reader


def read(image):

    return nd2.ND2File(image)


def parse(image, overlap, slices):

    if 'P' not in image.sizes.keys():

        tile = image.asarray()
        tiles = np.empty((1, 1), dtype=object)
        tiles[0, 0] = tile
        n_blocks = (1, 1)
        overlap = (0, 0)
        image_shape = tile.shape

    else:

        positions = None

        for loop in image.experiment:
            if isinstance(loop, nd2.structures.XYPosLoop):
                positions = np.array([point.stagePositionUm[:2] for point in loop.parameters.points]).T
                break

        camera_transformation = np.array(image.metadata.channels[0].volume.cameraTransformationMatrix).reshape(2, 2)

        x, y = np.linalg.inv(camera_transformation) @ positions

        if positions[0, 1] - positions[0, 0] > 0:
            x = -x
        if positions[1, -1] - positions[1, 0] > 0:
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
            x_overlap = round(1 - (np.mean(np.diff(x_overlaps)) / image.metadata.channels[0].volume.axesCalibration[
                0]) / image.attributes.widthPx, 2)
            y_overlap = round(1 - (np.mean(np.diff(y_overlaps)) / image.metadata.channels[0].volume.axesCalibration[
                1]) / image.attributes.heightPx, 2)

            if not ((0 < x_overlap < 1) & (0 < y_overlap < 1)):
                raise RuntimeError("Failed to determine overlap percentage from metadata.")

            overlap = (y_overlap, x_overlap)

        width = round(image.attributes.widthPx * (x_ndim - (x_ndim - 1) * overlap[1]))
        height = round(image.attributes.heightPx * (y_ndim - (y_ndim - 1) * overlap[0]))

        image_shape = None
        image_array = image.asarray()
        tiles = np.empty(shape=(y_ndim, x_ndim), dtype=object)

        for n in range(image.sizes['P']):

            tile = image_array[n]
            if tile.ndim > 2:
                tile = tile[slices]
            if image_shape is None:
                image_shape = (*tile.shape[:-2], height, width)
            tiles[i[n], j[n]] = tile

        n_blocks = tiles.shape

    tile_indices, border_indices = utils.calculate_indices(image_shape, n_blocks, overlap)

    return tiles, n_blocks, overlap, image_shape, tile_indices, border_indices


def stitch(image, overlap, slices):

    tiles, n_blocks, overlap, image_shape, tile_indices, border_indices = parse(image, overlap, slices)
    stitch_indices = utils.calculate_stitch_indices(tiles, border_indices, tile_indices)
    stitched_image = np.zeros(image_shape)

    for (n_i, n_j), (i_image, j_image, i, j) in stitch_indices.items():

        tile = tiles[n_i, n_j]
        tile_crop = tile[..., i[0]:i[1], j[0]:j[1]]
        stitched_image[..., i_image[0]:i_image[1], j_image[0]:j_image[1]] = tile_crop

    return stitched_image, tiles, n_blocks, overlap, image_shape, tile_indices, border_indices, stitch_indices


def legacy_read(image):

    image = ND2Reader(image)
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


def legacy_parse(image, metadata, overlap, slices):

    if 'v' not in image.iter_axes:

        tile = np.array(image[0])
        tiles = np.empty((1, 1), dtype=object)
        tiles[0, 0] = tile
        n_blocks = (1, 1)
        overlap = (0, 0)
        image_shape = image.frame_shape

    else:

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

        image_shape = None
        tiles = np.empty(shape=(y_ndim, x_ndim), dtype=object)

        for n in image.metadata['fields_of_view']:

            tile = np.array(image[n])
            if tile.ndim > 2:
                tile = tile[slices]
            if image_shape is None:
                image_shape = (*tile.shape[:-2], height, width)
            tiles[i[n], j[n]] = tile

        n_blocks = tiles.shape

    tile_indices, border_indices = utils.calculate_indices(image_shape, n_blocks, overlap)

    return tiles, n_blocks, overlap, image_shape, tile_indices, border_indices
