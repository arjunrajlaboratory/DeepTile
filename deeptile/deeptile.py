import numpy as np
from deeptile import algorithms, utils


class DeepTile:

    def __init__(self, image):

        self.image = image
        self.image_type = None

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

    def process(self, tiles, func_process, batch_size=None):

        self._check_configuration()

        if not isinstance(func_process, algorithms.AlgorithmBase):

            func_process = algorithms.transform(func_process)

        tiles = utils.pad_tiles(tiles, self.tile_size)
        nonempty_indices = tuple(self.stitch_indices.keys())
        nonempty_tiles = tiles[tuple(zip(*nonempty_indices))]

        processed_tiles = np.zeros_like(tiles)

        if func_process.batch:

            if batch_size is None:
                batch_size = func_process.default_batch_size

            n_batches = np.ceil(len(nonempty_tiles) / batch_size).astype(int)

            for n in range(n_batches):

                batch_indices = nonempty_indices[n * batch_size:(n + 1) * batch_size]
                batch_tiles = np.stack(nonempty_tiles[n * batch_size:(n + 1) * batch_size])

                processed_batch_tiles = func_process(batch_tiles, batch_size=len(batch_tiles))
                processed_tiles[tuple(zip(*batch_indices))] = tuple(processed_batch_tiles)

        else:

            for index, tile in zip(nonempty_indices, nonempty_tiles):

                processed_tile = func_process(tile)
                processed_tiles[index] = processed_tile

        processed_tiles = utils.unpad_tiles(processed_tiles, self.tile_indices)

        self._update_job_summary('process')

        return processed_tiles

    def stitch(self, tiles, func_stitch):

        self._check_configuration()

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

    def configure(self, tile_size=(1024, 1024), overlap=(0.1, 0.1), slices=(slice(None))):

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

        self.stitch_indices = utils.calculate_stitch_indices(tiles, self.tile_indices, self.border_indices)

        self._update_job_summary('get_tiles')

        return tiles


class DeepTileLargeImage(DeepTile):

    def __init__(self, image):

        super().__init__(image)
        self.image_type = 'large_image'

    def configure(self, tile_size=(1024, 1024), overlap=(0.1, 0.1), slices=0):

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

        self.stitch_indices = utils.calculate_stitch_indices(tiles, self.tile_indices, self.border_indices)

        self._update_job_summary('get_tiles')

        return tiles


class DeepTileND2(DeepTile):

    def __init__(self, image):

        super().__init__(image)
        self.image_type = 'nd2'

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
