import numpy as np
from dask.array import Array
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
            tiles : Tiled
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
            processed_tiles : Tiled
                Array of tiles after processing with ``func_process``.
        """

        self._check_compatibility(tiles, func_process, 'process')

        job = Job(tiles, 'process', locals())

        nonempty_indices = tiles.profile.nonempty_indices
        nonempty_tiles = tiles[tuple(zip(*nonempty_indices))]

        if batch_axis is not None:
            batch_axis_len = nonempty_tiles[0].shape[batch_axis]
            nonempty_indices = np.repeat(np.array(nonempty_indices), batch_axis_len, 0)
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

                    processed_tiles = utils.update_tiles(processed_tiles, tuple(index), processed_tile,
                                                         batch_axis, func_process.output_type)

        else:

            for index, tile in zip(nonempty_indices, nonempty_tiles):

                if isinstance(tile, Array):
                    tile = tile.compute()

                processed_tile = func_process(tile)
                processed_tiles = utils.update_tiles(processed_tiles, tuple(index), processed_tile,
                                                     batch_axis, func_process.output_type)

        processed_tiles = Tiled(processed_tiles, job, func_process.output_type)

        return processed_tiles

    def stitch(self, tiles, func_stitch):

        """ Stitch tiles using a transformed function.

        Parameters
        ----------
            tiles : Tiled
                Array of tiles.
            func_stitch : Algorithm
                Stitching function transformed into an Algorithm object.

        Returns
        -------
            stitched : Stitched
                Stitched array.
        """

        self._check_compatibility(tiles, func_stitch, 'stitch')

        job = Job(tiles, 'stitch', locals())

        stitched = func_stitch(tiles)
        stitched = Stitched(stitched, job, func_stitch.output_type)

        return stitched

    def _check_compatibility(self, tiles, func, job_type):

        """ (For internal use) Check if the given tiles and func are compatible.

        Parameters
        ----------
            tiles : Tiled
                Array of tiles.
            func : Algorithm
                Function transformed into an Algorithm object.
            job_type : str
                Type of job.

        Raises
        ------
            ValueError
                If ``tiles`` were not created by this DeepTile object.
            TypeError
                If ``func`` has not been transformed to an instance of the Algorithm class.
            TypeError
                If ``func`` has the incorrect algorithm type.
            ValueError
                If ``tiles`` and ``func`` are not compatible.
        """

        if self is not tiles.dt:
            raise ValueError("Tiles were not created by this DeepTile object.")

        if not issubclass(type(func), AlgorithmBase):
            raise TypeError(f"func_{job_type} must be transformed to an instance of the Algorithm class.")

        if func.algorithm_type != job_type:
            raise TypeError(f"func_{job_type} has the incorrect algorithm type of {func.algorithm_type}.")

        if tiles.otype != func.input_type:
            raise ValueError(f"Tile object type {tiles.otype} does not match the expected "
                             f"function input object type {func.input_type}.")

    @staticmethod
    def _create_profile_kwargs(tiling, tile_size, overlap, slices,
                               nonempty_indices, tile_indices, border_indices):

        """ (For internal use) Create profile keyword arguments.

        Parameters
        ----------
            tiling : tuple
                Number of tiles in each dimension.
            tile_size : tuple
                Size of each tile.
            overlap : tuple
                Fractions of ``tile_size`` to use for overlap.
            slices : tuple or int
                Slices to be extracted.
            nonempty_indices : tuple
                Indices of nonempty tiles.
            tile_indices : tuple
                Indices of tiles.
            border_indices : tuple
                Indices of borders at the middle of tile overlaps.

        Returns
        -------
            profile_kwargs : dict
                Profile keyword arguments.

        """

        profile_kwargs = {
            'tiling': tiling,
            'tile_size': tile_size,
            'overlap': overlap,
            'slices': slices,
            'nonempty_indices': nonempty_indices,
            'tile_indices': tile_indices,
            'border_indices': border_indices,
        }

        return profile_kwargs


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

        profile_kwargs = self._create_profile_kwargs(tiling, tile_size, overlap, slices,
                                                     nonempty_indices, tile_indices, border_indices)
        profile = Profile(self, **profile_kwargs)
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

        profile_kwargs = self._create_profile_kwargs(tiling, tile_size, overlap, slices,
                                                     nonempty_indices, tile_indices, border_indices)
        profile = Profile(self, **profile_kwargs)
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

        tiles, tiling, tile_size, overlap, self.image_shape = nd2.parse(self.image, overlap, slices)
        _, tile_indices, border_indices = utils.calculate_indices(self.image_shape, tile_size, overlap)
        nonempty_indices = utils.get_nonempty_indices(tiles)

        profile_kwargs = self._create_profile_kwargs(tiling, tile_size, overlap, slices,
                                                     nonempty_indices, tile_indices, border_indices)
        profile = Profile(self, **profile_kwargs)
        job = Job(self.image, 'get_tiles', locals(), profile)
        tiles = Tiled(tiles, job, 'tiled_image')

        return tiles
