import numpy as np
from deeptile import utils
from deeptile.algorithms import AlgorithmBase
from deeptile.data import Tiled, Stitched
from deeptile.jobs import Job
from deeptile.profiles import Profile


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
        self.image_shape = None
        self.link_data = True
        self.profiles = []

    def process(self, tiles, func_process, batch_axis=None, batch_size=None, pad_final_batch=False):

        """ Process tiles using a transformed function.

        Parameters
        ----------
            tiles : Tiled or tuple of Tiled
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
            processed_tiles : Tiled or tuple of Tiled
                Array of tiles after processing with ``func_process``.
        """

        tiles = utils.to_tuple(tiles)
        self._check_compatibility(tiles, func_process, 'process')

        job = Job(tiles, 'process', locals())

        profile = job.profile
        nonempty_indices = profile.nonempty_indices
        nonempty_tiles = [ts.nonempty_tiles for ts in tiles]

        output_type = utils.to_tuple(func_process.output_type)
        processed_tiles = [np.empty(profile.tiling, dtype=object) for _ in range(len(output_type))]

        if batch_axis is not None:
            batch_axis_len = nonempty_tiles[0][0].shape[batch_axis]
            nonempty_indices = np.repeat(np.array(nonempty_indices), batch_axis_len, 0)
            nonempty_tiles = [[subt for t in ts for subt in list(np.moveaxis(t, batch_axis, 0))]
                              for ts in nonempty_tiles]

        if func_process.vectorized:

            if batch_size is None:
                batch_size = func_process.default_batch_size

            n_batches = np.ceil(len(nonempty_tiles[0]) / batch_size).astype(int)

            for n in range(n_batches):

                batch_indices = nonempty_indices[n * batch_size:(n + 1) * batch_size]
                batch_tiles = [np.stack(ts[n * batch_size:(n + 1) * batch_size]) for ts in nonempty_tiles]
                if pad_final_batch & (batch_tiles[0].shape[0] < batch_size):
                    batch_tiles = [utils.array_pad(ts, batch_size - ts.shape[0], 0) for ts in batch_tiles]
                batch_tiles = utils.compute_dask(batch_tiles)

                processed_batch_tiles = func_process(*batch_tiles)
                processed_batch_tiles = utils.to_tuple(processed_batch_tiles)

                for i_batch, index in enumerate(batch_indices):

                    processed_tile = tuple(ts[i_batch] for ts in processed_batch_tiles)
                    processed_tiles = utils.update_tiles(processed_tiles, tuple(index), processed_tile, batch_axis,
                                                         output_type)

        else:

            for i_nonempty, index in enumerate(nonempty_indices):

                tile = [ts[i_nonempty] for ts in nonempty_tiles]
                tile = utils.compute_dask(tile)

                processed_tile = func_process(*tile)
                processed_tile = utils.to_tuple(processed_tile)
                processed_tiles = utils.update_tiles(processed_tiles, tuple(index), processed_tile, batch_axis,
                                                     output_type)

        processed_tiles = [Tiled(ts, job, otype) for ts, otype in zip(processed_tiles, output_type)]

        if len(processed_tiles) == 1:
            processed_tiles = processed_tiles[0]
        else:
            processed_tiles = tuple(processed_tiles)

        return processed_tiles

    def stitch(self, tiles, func_stitch):

        """ Stitch tiles using a transformed function.

        Parameters
        ----------
            tiles : Tiled or tuple of Tiled
                Array of tiles.
            func_stitch : Algorithm
                Stitching function transformed into an Algorithm object.

        Returns
        -------
            stitched : Stitched
                Stitched array.
        """

        tiles = utils.to_tuple(tiles)
        self._check_compatibility(tiles, func_stitch, 'stitch')

        job = Job(tiles, 'stitch', locals())

        stitched = func_stitch(*tiles)
        stitched = Stitched(stitched, job, func_stitch.output_type)

        return stitched

    def _check_compatibility(self, tiles, func, job_type):

        """ (For internal use) Check if the given tiles and func are compatible.

        Parameters
        ----------
            tiles : tuple of Tiled
                Array of tiles.
            func : Algorithm
                Function transformed into an Algorithm object.
            job_type : str
                Type of job.

        Raises
        ------
            ValueError
                If ``tiles`` and ``func`` are not compatible.
            ValueError
                If ``tiles`` were not created by this DeepTile object.
            TypeError
                If ``func`` has not been transformed to an instance of the Algorithm class.
            TypeError
                If ``func`` has the incorrect algorithm type.
        """

        input_type = utils.to_tuple(func.input_type)
        num_expected = len(input_type)
        num_got = len(tiles)

        if num_expected != num_got:
            raise ValueError(f'Expected input count {num_expected}, got {num_got}.')

        for i, ts in enumerate(tiles):
            if self is not ts.dt:
                raise ValueError("Tiles were not created by this DeepTile object.")
            if ts.otype != input_type[i]:
                raise ValueError(f"Tile object type does not match the expected function input object type.")

        if not issubclass(type(func), AlgorithmBase):
            raise TypeError(f"func_{job_type} must be transformed to an instance of the Algorithm class.")

        if func.algorithm_type != job_type:
            raise TypeError(f"func_{job_type} has the incorrect algorithm type of {func.algorithm_type}.")


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

    def get_tiles(self, tile_size, overlap=(0.1, 0.1), slices=(slice(None))):

        """ Split array into tiles.

        Parameters
        ----------
            tile_size : tuple
                Size of each tile.
            overlap : tuple, optional, default (0.1, 0.1)
                Fractions of ``tile_size`` to use for overlap.
            slices : tuple, optional, default (slice(None))
                Tuple of slice objects designating slices to be extracted.

        Returns
        -------
            tiles : Tiled
                Array of tiles.
        """

        image = self.image[slices]
        self.image_shape = image.shape

        tiling, tile_indices, border_indices = utils.calculate_indices(self.image_shape, tile_size, overlap)
        tiles = utils.array_split_2d(image, tile_indices)
        tiles = utils.cast_list_to_array(tiles)
        tiles = utils.pad_tiles(tiles, tile_size, tile_indices)
        nonempty_indices = utils.get_nonempty_indices(tiles)

        profile = Profile(self, tiling, tile_size, overlap, slices, nonempty_indices, tile_indices, border_indices)
        job = Job(self.image, 'get_tiles', locals(), profile)
        tiles = Tiled(tiles, job, 'tiled_image')

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

    def get_tiles(self, tile_size, overlap=(0.1, 0.1), slices=0):

        """ Obtain tiles from large_image tile source.

        Parameters
        ----------
            tile_size : tuple
                Size of each tile.
            overlap : tuple, optional, default (0.1, 0.1)
                Fractions of ``tile_size`` to use for overlap.
            slices : int, optional, default 0
                Frame index designating frame to be extracted.

        Returns
        -------
            tiles : Tiled
                Array of tiles.
        """

        from deeptile.sources import large_image

        self.image_shape = (self.image.getMetadata()['sizeY'], self.image.getMetadata()['sizeX'])
        tiles, tiling, tile_indices, border_indices = large_image.parse(self.image, self.image_shape,
                                                                        tile_size, overlap, slices)
        tiles = utils.pad_tiles(tiles, tile_size, tile_indices)
        nonempty_indices = utils.get_nonempty_indices(tiles)

        profile = Profile(self, tiling, tile_size, overlap, slices, nonempty_indices, tile_indices, border_indices)
        job = Job(self.image, 'get_tiles', locals(), profile)
        tiles = Tiled(tiles, job, 'tiled_image')

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
        self.image_sizes = None
        self.axes_order = None

    def get_tiles(self, overlap=(0.1, 0.1), slices=(slice(None))):

        """ Obtain tiles from ND2 file.

        Parameters
        ----------
            overlap : tuple or None, optional, default (0.1, 0.1)
                Fractions to use for overlap. If ``None``, overlap is automatically determined from the ND2 metadata.
            slices : tuple, optional, default (slice(None))
                Tuple of slice objects designating slices to be extracted.

        Returns
        -------
            tiles : Tiled
                Array of tiles.
        """

        from deeptile.sources import nd2

        tiles, tiling, tile_size, overlap, self.image_shape = nd2.parse(self.image, self.image_sizes, self.axes_order,
                                                                        overlap, slices)
        _, tile_indices, border_indices = utils.calculate_indices(self.image_shape, tile_size, overlap)
        nonempty_indices = utils.get_nonempty_indices(tiles)

        profile = Profile(self, tiling, tile_size, overlap, slices, nonempty_indices, tile_indices, border_indices)
        job = Job(self.image, 'get_tiles', locals(), profile)
        tiles = Tiled(tiles, job, 'tiled_image')

        return tiles
