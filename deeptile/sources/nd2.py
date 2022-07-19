import nd2
import numpy as np
from collections import OrderedDict

AXES_ORDER = ('P', 'T', 'Z', 'C', 'S', 'Y', 'X')


def read(image):

    image = nd2.ND2File(image)
    raw_image_sizes = image.sizes
    axes_order = tuple(axis for axis in AXES_ORDER if axis in raw_image_sizes.keys())
    image_sizes = OrderedDict((axis, raw_image_sizes[axis]) for axis in axes_order)

    return image, image_sizes, axes_order


def parse(image, image_sizes, axes_order, overlap, slices):

    image_array = image.to_dask()
    raw_axes_order = image.sizes.keys()
    image_array = np.moveaxis(image_array,
                              range(len(raw_axes_order)), (axes_order.index(axis) for axis in raw_axes_order))

    if 'P' not in axes_order:

        tiles = np.empty((1, 1), dtype=object)
        tiles[0, 0] = image_array[slices]
        tiling = (1, 1)
        overlap = (0, 0)
        image_shape = image_array.shape

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

        width = x_ndim * image.attributes.widthPx - (x_ndim - 1) * round(image.attributes.widthPx * overlap[1])
        height = y_ndim * image.attributes.heightPx - (y_ndim - 1) * round(image.attributes.heightPx * overlap[0])

        image_shape = None
        tiles = np.empty(shape=(y_ndim, x_ndim), dtype=object)

        for n in range(image_sizes['P']):

            tile = image_array[n]
            if tile.ndim > 2:
                tile = tile[slices]
            if image_shape is None:
                image_shape = (*tile.shape[:-2], height, width)
            tiles[i[n], j[n]] = tile

        tiling = tiles.shape

    tile_size = (image.attributes.heightPx, image.attributes.widthPx)

    return tiles, tiling, tile_size, overlap, image_shape
