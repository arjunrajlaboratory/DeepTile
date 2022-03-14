import numpy as np
from deeptile import sources, utils


class DeepTile:

    def __init__(self, image):

        self.image = image
        self.image_type = None

        self.tiling = None
        self.max_tile_size = None
        self.overlap = None
        self.slices = None
        self.configured = False

        self.image_shape = None
        self.tile_indices = None
        self.border_indices = None
        self.stitch_indices = None
        self.job_summary = None

    def process(self, tiles, func_process):

        self._check_configuration()

        processed_tiles = np.zeros_like(tiles)
        processed_tile_dims = None

        for index, tile in np.ndenumerate(tiles):

            if tile is None:
                processed_tile = None
            else:
                tile = tile.reshape(-1, *tile.shape[-2:])
                processed_tile = func_process(tile)
                processed_tile = np.squeeze(processed_tile)
                if processed_tile_dims is None:
                    processed_tile_dims = processed_tile.shape[:-2]
                if processed_tile.shape[:-2] != processed_tile_dims:
                    raise ValueError("Processed tile dimension mismatch.")

            processed_tiles[index] = processed_tile

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
            'max_tile_size': self.max_tile_size,
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

    def configure(self, tiling=None, max_tile_size=(1024, 1024), overlap=(0.1, 0.1), slices=(slice(None))):

        self._reset()

        if (tiling is None) and (max_tile_size is None):
            raise ValueError("At least one of tiling and max_tile_size must be specified.")
        self.tiling = tiling
        self.max_tile_size = max_tile_size
        self.overlap = overlap
        self.slices = slices

        self.configured = True

    def get_tiles(self):

        self._check_configuration()

        image = self.image[self.slices]
        self.image_shape = image.shape

        self.tiling = utils.check_tiling(self.tiling, self.image_shape, self.max_tile_size, self.overlap)
        self.tile_indices, self.border_indices = utils.calculate_indices(self.image_shape, self.tiling,
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

    def configure(self, tiling=None, max_tile_size=(1024, 1024), overlap=(0.1, 0.1), slices=0):

        self._reset()

        if (tiling is None) and (max_tile_size is None):
            raise ValueError("At least one of tiling and max_tile_size must be specified.")
        self.tiling = tiling
        self.max_tile_size = max_tile_size
        self.overlap = overlap
        self.slices = slices

        self.configured = True

    def get_tiles(self):

        self._check_configuration()

        self.image_shape = (self.image.getMetadata()['sizeY'], self.image.getMetadata()['sizeX'])
        self.tiling = utils.check_tiling(self.tiling, self.image_shape, self.max_tile_size, self.overlap)
        tiles, self.tile_indices, self.border_indices = \
            sources.large_image.parse(self.image, self.image_shape, self.tiling, self.overlap, self.slices)

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

        tiles, self.tiling, self.overlap, self.image_shape = sources.nd2.parse(self.image, self.overlap, self.slices)

        self.tile_indices, self.border_indices = utils.calculate_indices(self.image_shape, self.tiling, self.overlap)
        self.stitch_indices = utils.calculate_stitch_indices(tiles, self.tile_indices, self.border_indices)

        self._update_job_summary('get_tiles')

        return tiles
