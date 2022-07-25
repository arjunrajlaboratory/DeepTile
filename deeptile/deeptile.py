import numpy as np
from deeptile.core import utils
from deeptile.core.algorithms import AlgorithmBase
from deeptile.core.data import Tiled, Stitched
from deeptile.core.jobs import Job
from deeptile.core.profiles import Profile


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
        self.link_data = None
        self.profiles = []

    def process(self, tiles, func_process, batch_axis=False, batch_size=None, pad_final_batch=False):

        """ Process tiles using a transformed function.

        Parameters
        ----------
            tiles : Tiled or tuple of Tiled
                Array of tiles.
            func_process : Algorithm
                Processing function transformed into an Algorithm object.
            batch_axis : bool, optional, default False
                Whether to use the first axis to create batches. If ``None``, no batch axis will be used.
            batch_size : int or None, optional, default None
                Number of tiles in each batch. If ``None``, the default batching configuration will be determined by
                ``func_process``. If ``func_process`` does not support batching, this value is ignored.
            pad_final_batch : bool, optional, default False
                Whether to pad the final batch to the specified ``batch_size``. If ``func_process`` does not support
                batching, this value is ignored.

        Returns
        -------
            processed_tiles : Tiled or tuple of Tiled
                Array of tiles after processing with ``func_process``.
        """

        job_kwargs = locals()
        job_kwargs.pop('self')
        job_kwargs.pop('tiles')

        tiles = utils.to_tuple(tiles)
        self._check_compatibility(tiles, func_process, 'process')

        job = Job(tiles, 'process', job_kwargs)
        profile = job.profile

        unpack_input_singleton = isinstance(func_process.input_type, str)
        unpack_output_singleton = isinstance(func_process.output_type, str)
        output_type = utils.to_tuple(func_process.output_type)

        nonempty_indices = profile.nonempty_indices
        nonempty_tiles = [ts.nonempty_tiles for ts in tiles]
        processed_tiles = [np.empty(profile.tiling, dtype=object) for _ in range(len(output_type))]

        if batch_axis:
            batch_axis_len = nonempty_tiles[0][0].shape[0]
            nonempty_indices = np.repeat(np.array(nonempty_indices), batch_axis_len, 0)
            nonempty_tiles = [[subt for t in ts for subt in t] for ts in nonempty_tiles]

        if func_process.batching:

            if batch_size is None:
                batch_size = func_process.default_batch_size

            n_batches = np.ceil(len(nonempty_tiles[0]) / batch_size).astype(int)

            for n in range(n_batches):

                batch_indices = nonempty_indices[n * batch_size:(n + 1) * batch_size]
                batch_tiles = [np.stack(ts[n * batch_size:(n + 1) * batch_size]) for ts in nonempty_tiles]
                if pad_final_batch and (batch_tiles[0].shape[0] < batch_size):
                    batch_tiles = [utils.array_pad(ts, batch_size - ts.shape[0], 0) for ts in batch_tiles]
                batch_tiles = utils.compute_dask(batch_tiles)
                if unpack_input_singleton:
                    batch_tiles = batch_tiles[0]

                processed_batch_tiles = func_process(tiles=batch_tiles)
                processed_batch_tiles = utils.to_tuple(processed_batch_tiles)

                for i_batch, index in enumerate(batch_indices):

                    processed_tile = tuple(ts[i_batch] for ts in processed_batch_tiles)
                    utils.check_data_count(processed_tile, output_type=output_type)
                    processed_tiles = utils.update_tiles(processed_tiles, tuple(index), processed_tile, batch_axis,
                                                         output_type)

        else:

            for i_nonempty, index in enumerate(nonempty_indices):

                tile = [ts[i_nonempty] for ts in nonempty_tiles]
                tile = utils.compute_dask(tile)
                if unpack_input_singleton:
                    tile = tile[0]

                processed_tile = func_process(tile=tile)
                processed_tile = utils.to_tuple(processed_tile)
                utils.check_data_count(processed_tile, output_type=output_type)
                processed_tiles = utils.update_tiles(processed_tiles, tuple(index), processed_tile, batch_axis,
                                                     output_type)

        processed_tiles = [Tiled(ts, job, otype) for ts, otype in zip(processed_tiles, output_type)]

        if unpack_output_singleton:
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

        job_kwargs = locals()
        job_kwargs.pop('self')
        job_kwargs.pop('tiles')

        tiles = utils.to_tuple(tiles)
        self._check_compatibility(tiles, func_stitch, 'stitch')

        job = Job(tiles, 'stitch', job_kwargs)

        unpack_input_singleton = isinstance(func_stitch.input_type, str)
        unpack_output_singleton = isinstance(func_stitch.output_type, str)
        output_type = utils.to_tuple(func_stitch.output_type)

        if unpack_input_singleton:
            tiles = tiles[0]

        stitched = func_stitch(tiles=tiles)
        stitched = utils.to_tuple(stitched)
        utils.check_data_count(stitched, output_type=output_type)
        stitched = [Stitched(s, job, otype) for s, otype in zip(stitched, output_type)]

        if unpack_output_singleton:
            stitched = stitched[0]
        else:
            stitched = tuple(stitched)

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
            TypeError
                If ``func`` has not been transformed to an instance of the Algorithm class.
            TypeError
                If ``func`` has the incorrect algorithm type.
            ValueError
                If ``tiles`` has an object type that does not match the expected ``func`` input object type.
            ValueError
                If ``tiles`` are not associated with this DeepTile object.
            ValueError
                If ``tiles`` do not all share a common profile.
        """

        if not issubclass(type(func), AlgorithmBase):
            raise TypeError(f"func_{job_type} must be transformed to an instance of the Algorithm class.")

        if func.algorithm_type != job_type:
            raise TypeError(f"func_{job_type} has the incorrect algorithm type of {func.algorithm_type}.")

        input_type = utils.to_tuple(func.input_type)
        utils.check_data_count(tiles, input_type=input_type)

        profile = None
        for i, ts in enumerate(tiles):
            if ts.otype != input_type[i]:
                raise ValueError(f"tile object type does not match the expected function input object type.")
            if profile is None:
                if ts.dt is self:
                    profile = ts.profile
                else:
                    raise ValueError("tiles are not associated with this DeepTile object.")
            else:
                if ts.profile is not profile:
                    raise ValueError(f'tiles must all share a common profile.')


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

        job_kwargs = locals()
        job_kwargs.pop('self')

        image = self.image[slices]
        self.image_shape = image.shape

        tiling, tile_indices, border_indices = utils.calculate_indices(self.image_shape, tile_size, overlap)
        tiles = utils.array_split_2d(image, tile_indices)
        tiles = utils.cast_list_to_array(tiles)
        tiles = utils.pad_tiles(tiles, tile_size, tile_indices)
        nonempty_indices = utils.get_nonempty_indices(tiles)

        profile = Profile(self, tiling, tile_size, overlap, slices, nonempty_indices, tile_indices, border_indices)
        job = Job(self.image, 'get_tiles', job_kwargs, profile)
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

        job_kwargs = locals()
        job_kwargs.pop('self')

        from deeptile.sources import large_image

        self.image_shape = (self.image.getMetadata()['sizeY'], self.image.getMetadata()['sizeX'])
        tiles, tiling, tile_indices, border_indices = large_image.parse(self.image, self.image_shape,
                                                                        tile_size, overlap, slices)
        tiles = utils.pad_tiles(tiles, tile_size, tile_indices)
        nonempty_indices = utils.get_nonempty_indices(tiles)

        profile = Profile(self, tiling, tile_size, overlap, slices, nonempty_indices, tile_indices, border_indices)
        job = Job(self.image, 'get_tiles', job_kwargs, profile)
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

        job_kwargs = locals()
        job_kwargs.pop('self')

        from deeptile.sources import nd2

        tiles, tiling, tile_size, overlap, self.image_shape = nd2.parse(self.image, self.image_sizes, self.axes_order,
                                                                        overlap, slices)
        _, tile_indices, border_indices = utils.calculate_indices(self.image_shape, tile_size, overlap)
        nonempty_indices = utils.get_nonempty_indices(tiles)

        profile = Profile(self, tiling, tile_size, overlap, slices, nonempty_indices, tile_indices, border_indices)
        job = Job(self.image, 'get_tiles', job_kwargs, profile)
        tiles = Tiled(tiles, job, 'tiled_image')

        return tiles
