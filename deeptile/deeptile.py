import numpy as np
from dask.array import Array
from deeptile import utils
from deeptile.algorithms import AlgorithmBase


class DeepTile:

    def __init__(self, image):

        self.image = image
        self.image_type = None
        self.pad = None

        self.tiling = None
        self.tile_size = None
        self.overlap = None
        self.slices = None
        self.configured = False

        self.image_shape = None
        self.tile_indices = None
        self.border_indices = None
        self.stitch_indices = None
        self.job_summary = None

    def process(self, tiles, func_process, batch_axis=None, batch_size=None, pad_final_batch=False):

        self._check_configuration()

        if not isinstance(func_process, AlgorithmBase):

            raise TypeError("func_process must be transformed to an instance of the Algorithm class.")

        nonempty_indices = np.array(tuple(self.stitch_indices.keys()))
        nonempty_tiles = tiles[tuple(zip(*nonempty_indices))]

        if batch_axis is not None:
            batch_axis_len = nonempty_tiles[0].shape[batch_axis]
            nonempty_indices = np.repeat(nonempty_indices, batch_axis_len, 0)
            nonempty_tiles = [subtile for tile in nonempty_tiles for subtile in list(np.moveaxis(tile, batch_axis, 0))]

        processed_tiles = np.empty_like(tiles)

        if func_process.vectorized:

            if batch_size is None:
                batch_size = func_process.default_batch_size

            n_batches = np.ceil(len(nonempty_tiles) / batch_size).astype(int)

            for n in range(n_batches):

                batch_indices = nonempty_indices[n * batch_size:(n + 1) * batch_size]
                batch_tiles = np.stack(nonempty_tiles[n * batch_size:(n + 1) * batch_size])
                if pad_final_batch & batch_tiles.shape[0] < batch_size:
                    padding = ((0, batch_size - batch_tiles.shape[0]), ) + (batch_tiles.ndim - 1) * ((0, 0), )
                    batch_tiles = np.pad(batch_tiles, padding)
                    print(batch_tiles.shape)
                if isinstance(batch_tiles, Array):
                    batch_tiles = batch_tiles.compute()

                processed_batch_tiles = func_process(batch_tiles)

                for index, processed_tile in zip(batch_indices, processed_batch_tiles):

                    processed_tiles = utils.update_tiles(processed_tiles, tuple(index), processed_tile, batch_axis)

        else:

            for index, tile in zip(nonempty_indices, nonempty_tiles):

                if isinstance(tile, Array):
                    tile = tile.compute()

                processed_tile = func_process(tile)
                processed_tiles = utils.update_tiles(processed_tiles, tuple(index), processed_tile, batch_axis)

        self._update_job_summary('process')

        return processed_tiles

    def stitch(self, tiles, func_stitch):

        self._check_configuration()

        if self.pad:
            tiles = utils.unpad_tiles(tiles, self.tile_indices)
        stitch = func_stitch(self, tiles)

        self._update_job_summary('stitch')

        return stitch

    def _reset(self):

        self.image_shape = None
        self.tile_indices = None
        self.border_indices = None
        self.stitch_indices = None
        self.job_summary = None

    def _check_configuration(self):

        if not self.configured:
            raise RuntimeError("DeepTile object not configured.")

    def _update_job_summary(self, job_type):

        self.job_summary = {
            'job_type': job_type,
            'image_type': self.image_type,
            'tiling': self.tiling,
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'slices': self.slices,
            'image_shape': self.image_shape,
            'tile_indices': self.tile_indices,
            'border_indices': self.border_indices,
            'stitch_indices': self.stitch_indices
        }


class DeepTileArray(DeepTile):

    def __init__(self, image):

        super().__init__(image)
        self.image_type = 'array'
        self.pad = True

    def configure(self, tile_size, overlap=(0.1, 0.1), slices=(slice(None))):

        self._reset()

        self.tile_size = tile_size
        self.overlap = overlap
        self.slices = slices

        self.configured = True

    def get_tiles(self):

        self._check_configuration()

        image = self.image[self.slices]
        self.image_shape = image.shape

        self.tiling, self.tile_indices, self.border_indices = utils.calculate_indices(self.image_shape, self.tile_size,
                                                                                      self.overlap)
        tiles = utils.array_split_2d(image, self.tile_indices)
        tiles = utils.cast_list_to_array(tiles)
        tiles = utils.pad_tiles(tiles, self.tile_size)

        self.stitch_indices = utils.calculate_stitch_indices(tiles, self.tile_indices, self.border_indices)

        self._update_job_summary('get_tiles')

        return tiles


class DeepTileLargeImage(DeepTile):

    def __init__(self, image):

        super().__init__(image)
        self.image_type = 'large_image'
        self.pad = True

    def configure(self, tile_size, overlap=(0.1, 0.1), slices=0):

        self._reset()

        self.tile_size = tile_size
        self.overlap = overlap
        self.slices = slices

        self.configured = True

    def get_tiles(self):

        self._check_configuration()

        from deeptile.sources import large_image

        self.image_shape = (self.image.getMetadata()['sizeY'], self.image.getMetadata()['sizeX'])
        tiles, self.tiling, self.tile_indices, self.border_indices = \
            large_image.parse(self.image, self.image_shape, self.tile_size, self.overlap, self.slices)
        tiles = utils.pad_tiles(tiles, self.tile_size)

        self.stitch_indices = utils.calculate_stitch_indices(tiles, self.tile_indices, self.border_indices)

        self._update_job_summary('get_tiles')

        return tiles


class DeepTileND2(DeepTile):

    def __init__(self, image):

        super().__init__(image)
        self.image_type = 'nd2'
        self.pad = False

    def configure(self, overlap=(0.1, 0.1), slices=(slice(None))):

        self._reset()

        self.overlap = overlap
        self.slices = slices

        self.configured = True

    def get_tiles(self):

        self._check_configuration()

        from deeptile.sources import nd2

        tiles, self.tiling, self.tile_size, self.overlap, self.image_shape = nd2.parse(self.image, self.overlap,
                                                                                       self.slices)

        _, self.tile_indices, self.border_indices = utils.calculate_indices(self.image_shape, self.tile_size,
                                                                            self.overlap)
        self.stitch_indices = utils.calculate_stitch_indices(tiles, self.tile_indices, self.border_indices)

        self._update_job_summary('get_tiles')

        return tiles
