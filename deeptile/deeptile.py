import numpy as np
from dask.array import Array
from deeptile import utils
from deeptile.algorithms import AlgorithmBase


class DeepTile:

    """ Base DeepTile class that handles tile processing and stitching.

    Parameters
    ----------
        image
            An object containing an image.
    """

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
        self.tile_padding = None
        self.job_log = {}

    def process(self, tiles, func_process, batch_axis=None, batch_size=None, pad_final_batch=False):

        """ Process tiles using a transformed function.

        Parameters
        ----------
            tiles : numpy.array
                Array of tiles.
            func_process : Algorithm
                Processing function transformed into an Algorithm object.
            batch_axis : int or None, optional, default None
                Image axis used to create batches during processing. If ``None``, no batch axis will be used.
            batch_size : int or None, optional, default None
                Number of tiles in each batch. If ``None``, the default batching configuration will be determined by
                ``func_process``. If ``func_process`` is not vectorized, this value is ignored.
            pad_final_batch : bool, optional, default False
                Whether to pad the final batch to the specified ``batch_size``. If ``func_process`` is not vectorized,
                this value is ignored.

        Returns
        -------
            processed_tiles : numpy.ndarray
                Array of tiles after processing with ``func_process``.

        Raises
        ------
            TypeError
                If ``func_process`` has not been transformed to an instance of the Algorithm class.
        """

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
                if pad_final_batch & (batch_tiles.shape[0] < batch_size):
                    batch_tiles = utils.array_pad(batch_tiles, batch_size - batch_tiles.shape[0], 0)
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

        self._update_job_log('process')

        return processed_tiles

    def stitch(self, tiles, func_stitch):

        """ Stitch tiles using a transformed function.

        Parameters
        ----------
            tiles : numpy.array
                Array of tiles.
            func_stitch : Algorithm
                Stitching function transformed into an Algorithm object.

        Returns
        -------
            stitched : numpy.ndarray
                Stitched array.

        Raises
        ------
            TypeError
                If ``func_stitch`` has not been transformed to an instance of the Algorithm class.
        """

        self._check_configuration()

        if not isinstance(func_stitch, AlgorithmBase):
            raise TypeError("func_stitch must be transformed to an instance of the Algorithm class.")

        tiles = utils.unpad_tiles(tiles, self.tile_padding)
        stitched = func_stitch(self, tiles)

        self._update_job_log('stitch')

        return stitched

    def _reset(self):

        """ (For internal use) Reset DeepTile object.
        """

        self.image_shape = None
        self.tile_indices = None
        self.border_indices = None
        self.stitch_indices = None

    def _check_configuration(self):

        """ (For internal use) Check if DeepTile object is configured.

        Raises
        ------
            RuntimeError
                If ``DeepTile`` object has not been configured.
        """

        if not self.configured:
            raise RuntimeError("DeepTile object not configured.")

    def _update_job_log(self, job_type):

        """ (For internal use) Update job log.

        Parameters
        ----------
            job_type : str
                Type of job.
        """

        n = len(self.job_log) + 1
        self.job_log[n] = {
            'job_type': job_type,
            'image_type': self.image_type,
            'tiling': self.tiling,
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'slices': self.slices,
            'image_shape': self.image_shape,
            'tile_indices': self.tile_indices,
            'border_indices': self.border_indices,
            'stitch_indices': self.stitch_indices,
            'tile_padding': self.tile_padding
        }


class DeepTileArray(DeepTile):

    """ DeepTile subclass for arrays.

    Parameters
    ----------
        image : array_like
            An array-like object of an image.
    """

    def __init__(self, image):

        super().__init__(image)
        self.image_type = 'array'

    def configure(self, tile_size, overlap=(0.1, 0.1), slices=(slice(None))):

        """ Configure DeepTileArray object.

        Parameters
        ----------
            tile_size : tuple
                Size of each tile.
            overlap : tuple, optional, default (0.1, 0.1)
                Fractions of ``tile_size`` to use for overlap.
            slices : tuple, optional, default (slice(None))
                Tuple of slice objects designating slices to be extracted.
        """

        self._reset()

        self.tile_size = tile_size
        self.overlap = overlap
        self.slices = slices

        self.configured = True

    def get_tiles(self):

        """ Split array into tiles.

        Returns
        -------
            tiles : numpy.ndarray
                Array of tiles.
        """

        self._check_configuration()

        image = self.image[self.slices]
        self.image_shape = image.shape

        self.tiling, self.tile_indices, self.border_indices = utils.calculate_indices(self.image_shape, self.tile_size,
                                                                                      self.overlap)
        tiles = utils.array_split_2d(image, self.tile_indices)
        tiles = utils.cast_list_to_array(tiles)
        tiles, self.tile_padding = utils.pad_tiles(tiles, self.tile_size, self.tile_indices)

        self.stitch_indices = utils.calculate_stitch_indices(tiles, self.tile_indices, self.border_indices)

        self._update_job_log('get_tiles')

        return tiles


class DeepTileLargeImage(DeepTile):

    """ DeepTile subclass for large_image tile sources.

    Parameters
    ----------
        image : large_image tile source
            A large_image tile source.
    """

    def __init__(self, image):

        super().__init__(image)
        self.image_type = 'large_image'

    def configure(self, tile_size, overlap=(0.1, 0.1), slices=0):

        """ Configure DeepTileLargeImage object.

        Parameters
        ----------
            tile_size : tuple
                Size of each tile.
            overlap : tuple, optional, default (0.1, 0.1)
                Fractions of ``tile_size`` to use for overlap.
            slices : int, optional, default 0
                Frame index designating frame to be extracted.
        """

        self._reset()

        self.tile_size = tile_size
        self.overlap = overlap
        self.slices = slices

        self.configured = True

    def get_tiles(self):

        """ Obtain tiles from large_image tile source.

        Returns
        -------
            tiles : numpy.ndarray
                Array of tiles.
        """

        self._check_configuration()

        from deeptile.sources import large_image

        self.image_shape = (self.image.getMetadata()['sizeY'], self.image.getMetadata()['sizeX'])
        tiles, self.tiling, self.tile_indices, self.border_indices = \
            large_image.parse(self.image, self.image_shape, self.tile_size, self.overlap, self.slices)
        tiles, self.tile_padding = utils.pad_tiles(tiles, self.tile_size, self.tile_indices)

        self.stitch_indices = utils.calculate_stitch_indices(tiles, self.tile_indices, self.border_indices)

        self._update_job_log('get_tiles')

        return tiles


class DeepTileND2(DeepTile):

    """ DeepTile subclass for ND2 files.

    Parameters
    ----------
        image : ND2 file
            An ND2 file.
    """

    def __init__(self, image):

        super().__init__(image)
        self.image_type = 'nd2'

    def configure(self, overlap=(0.1, 0.1), slices=(slice(None))):

        """ Configure DeepTileND2 object.

        Parameters
        ----------
            overlap : tuple or None, optional, default (0.1, 0.1)
                Fractions to use for overlap. If ``None``, overlap is automatically determined from the ND2 metadata.
            slices : tuple, optional, default (slice(None))
                Tuple of slice objects designating slices to be extracted.
        """

        self._reset()

        self.overlap = overlap
        self.slices = slices

        self.configured = True

    def get_tiles(self):

        """ Obtain tiles from ND2 file.

        Returns
        -------
            tiles : numpy.ndarray
                Array of tiles.
        """

        self._check_configuration()

        from deeptile.sources import nd2

        tiles, self.tiling, self.tile_size, self.overlap, self.image_shape = nd2.parse(self.image, self.overlap,
                                                                                       self.slices)

        _, self.tile_indices, self.border_indices = utils.calculate_indices(self.image_shape, self.tile_size,
                                                                            self.overlap)
        self.stitch_indices = utils.calculate_stitch_indices(tiles, self.tile_indices, self.border_indices)
        self.tile_padding = (0, 0)

        self._update_job_log('get_tiles')

        return tiles
