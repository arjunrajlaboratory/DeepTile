import numpy as np
from dask.array import Array
from deeptile.core import utils
from deeptile.core.algorithms import partial, transform
from deeptile.core.iterators import IndicesIterator, TileIndicesIterator, BorderIndicesIterator, StitchIndicesIterator
from deeptile.core.jobs import Job
from deeptile.core.types import ALLOWED_TILED_TYPES, ALLOWED_STITCHED_TYPES
from functools import cached_property


class Data(np.ndarray):

    """ numpy.ndarray subclass for storing DeepTile data.
    """

    def __new__(cls, data, job, otype, allowed_otypes):

        """ Create new Data object.

        Parameters
        ----------
            data : numpy.ndarray or Data
                Data array.
            job : Job
                Job that generated this data object.
            otype : str
                Data object type.
            allowed_otypes : tuple
                List of allowed data object type.

        Returns
        -------
            data : Data
                Data array.

        Raises
        ------
            ValueError
                If ``otype`` is invalid.
        """

        data = np.asarray(data).view(cls)

        if otype not in allowed_otypes:
            raise ValueError("invalid data object type.")

        data.dt = job.dt
        data.profile = job.profile
        data.job = job
        data.otype = otype

        if data.dt.link_data:
            data.job.output = data

        return data


class Tiled(Data):

    """ numpy.ndarray subclass for storing DeepTile tiled data.
    """

    def __new__(cls, tiles, job, otype, mask=None):

        """ Create new Tiled object.

        Parameters
        ----------
            tiles : numpy.ndarray or Tiled
                Array of tiles.
            job : Job
                Job that generated this tiled object.
            otype : str
                Tiled object type.
            mask : numpy.ndarray or None, optional, default None
                Boolean mask.

        Returns
        -------
            tiles : Tiled
                Array of tiles.
        """

        tiles = super().__new__(cls, tiles, job, otype, ALLOWED_TILED_TYPES)
        tiles.parent = tiles
        tiles.slices = []
        if mask is None:
            tiles.mask = np.ones(tiles.profile.tiling, dtype=bool)
        else:
            tiles.mask = mask

        return tiles

    def __array_finalize__(self, tiles):

        """ Finalize Tiled object.

        Parameters
        ----------
            tiles : numpy.ndarray or Tiled
                Array of tiles.
        """

        if tiles is None:
            return
        self.dt = getattr(tiles, 'dt', None)
        self.profile = getattr(tiles, 'profile', None)
        self.job = getattr(tiles, 'job', None)
        self.otype = getattr(tiles, 'otype', None)
        self.parent = getattr(tiles, 'parent', None)
        self.slices = getattr(tiles, 'slices', None)
        self.mask = getattr(tiles, 'mask', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **_kwargs):

        """ Process tiles using a universal function.

        Parameters
        ----------
            ufunc : numpy.ufunc
                The ufunc object that was called.
            methods : str
                String indicating how the ``ufunc`` was called, either ``"__call__"`` to indicate it was called
                directly, or one of its methods: ``"reduce"``, ``"accumulate"``, ``"reduceat"``, ``"outer"``, or
                ``"at"``.
            inputs : tuple
                Tuple of the input arguments to the ``ufunc``.
            **kwargs : dict
                Contains any optional or keyword arguments passed to the function. This includes any ``out`` arguments,
                which are always contained in a tuple.

        Returns
        -------
            processed_tiles : Tiled
                Array of tiles after processing with ``ufunc``.
        """

        other_inputs = [i for i in inputs if i is not self]
        input_type = self.otype
        output_type = input_type

        transformed_func = transform(lambda tiles, *args, **kwargs:
                                     tiles.__array_ufunc__(ufunc, method, *(other_inputs + [tiles]), **_kwargs),
                                     input_type=input_type, output_type=output_type)

        dt = self.dt
        processed_tiles = dt.process(self, transformed_func, batch_size=len(self.nonempty_tiles))
        processed_tiles.job.type = 'array_ufunc'
        processed_tiles.job.kwargs = {
            'ufunc': ufunc,
            'method': method,
            'inputs': inputs,
            'kwargs': _kwargs
        }

        return processed_tiles

    def compute(self, batch_axis=False, batch_size=None, pad_final_batch=False, **kwargs):

        """ Compute Dask arrays.

        Parameters
        ----------
            batch_axis : bool, optional, default False
                Whether to use the first axis to create batches.
            batch_size : int or None, optional, default None
                Number of tiles in each batch.
            pad_final_batch : bool, optional, default False
                Whether to pad the final batch to the specified ``batch_size``.

        Returns
        -------
            tiles : Tiled
                Array of tiles.
        """

        if isinstance(self[self.nonempty_indices[0]], Array):

            tiles = self.dt.process(self, transform(lambda tile: tile.compute(**kwargs),
                                                    input_type=self.otype, output_type=self.otype),
                                    batch_axis=batch_axis, batch_size=batch_size, pad_final_batch=pad_final_batch)
            tiles.job.type = 'compute_dask'
            tiles.job.kwargs = kwargs

        else:

            tiles = self

        return tiles

    def persist(self, batch_axis=False, batch_size=None, pad_final_batch=False, **kwargs):

        """ Persist Dask arrays into memory.

        Parameters
        ----------
            batch_axis : bool, optional, default False
                Whether to use the first axis to create batches.
            batch_size : int or None, optional, default None
                Number of tiles in each batch.
            pad_final_batch : bool, optional, default False
                Whether to pad the final batch to the specified ``batch_size``.

        Returns
        -------
            tiles : Tiled
                Array of tiles.
        """

        if isinstance(self[self.nonempty_indices[0]], Array):

            tiles = self.dt.process(self, transform(lambda tile: tile.persist(**kwargs),
                                                    input_type=self.otype, output_type=self.otype),
                                    batch_axis=batch_axis, batch_size=batch_size, pad_final_batch=pad_final_batch)
            tiles.job.type = 'persist_dask'
            tiles.job.kwargs = kwargs

        else:

            tiles = self

        return tiles

    def import_data(self, data, data_type):

        """ Import external data and tile using the same tiling profile.

        Parameters
        ----------
            data
                Data to be imported.
            data_type : str
                Data object type.

        Returns
        -------
            tiles : Tiled
                Array of tiles.

        Raises
        ------
            ValueError
                If ``data_type`` is invalid.
        """

        job_kwargs = locals()
        job_kwargs.pop('self')
        job_kwargs.pop('data')

        if data_type == 'image':
            func_tile = transform(partial(utils.tile_image, image=data),
                                  input_type='tile_index_iterator', output_type='tiled_image')
            tiles = self.dt.process(self.tile_indices_iterator, func_tile)
            tile_size = self[self.nonempty_indices[0]].shape[-2:]
            tiles = utils.pad_tiles(tiles, tile_size, self.tile_indices)
        elif data_type == 'coords':
            func_tile = transform(partial(utils.tile_coords, coords=data),
                                  input_type='tile_index_iterator', output_type='tiled_coords')
            tiles = self.dt.process(self.tile_indices_iterator, func_tile)
        else:
            raise ValueError("invalid data object type.")

        tiles.job.type = 'import_data'
        tiles.job.kwargs = job_kwargs
        if self.dt.link_data:
            tiles.job.input = data
        else:
            tiles.job.input = None

        return tiles

    @cached_property
    def image_shape(self):

        """ Calculate scaled image shape.

        Returns
        -------
            image_shape : tuple of int
                Scaled image shape.
        """

        image_shape = None

        if self.otype == 'tiled_image':

            profile = self.profile
            tile_size = self[self.nonempty_indices[0]].shape[-2:]
            profile_tile_size = profile.tile_size
            profile_image_shape = profile.dt.image_shape
            scales = (tile_size[0] / profile_tile_size[0], tile_size[1] / profile_tile_size[1])
            image_shape = (round(profile_image_shape[-2] * scales[0]), round(profile_image_shape[-1] * scales[1]))

        elif self.otype == 'tiled_coords':

            image_shape = self.dt.image_shape

        return image_shape

    @cached_property
    def scales(self):

        """ Calculate tile scales relative to profile tile sizes.

        Returns
        -------
            scales : tuple of float
                Tile scales relative to profile tile sizes.
        """

        scales = None

        if self.otype == 'tiled_image':

            profile_image_shape = self.dt.image_shape
            image_shape = self.image_shape
            scales = (image_shape[0] / profile_image_shape[-2], image_shape[1] / profile_image_shape[-1])

        elif self.otype == 'tiled_coords':

            scales = (1.0, 1.0)

        return scales

    @cached_property
    def nonempty_mask(self):

        """ Get a mask for nonempty tiles to be processed.

        Returns
        -------
            nonempty_mask : numpy.ndarray
                Mask for nonempty tiles to be processed.

        Raises
        ------
            ValueError
                If there are no nonempty tiles to process.
        """

        nonempty_mask = self.mask * self.profile.nonempty_mask
        if not np.any(nonempty_mask):
            raise ValueError("No nonempty tiles to process.")

        return nonempty_mask

    @cached_property
    def nonempty_indices(self):

        """ Get a list of indices for nonempty tiles to be processed.

        Returns
        -------
            nonempty_indices : numpy.ndarray
                Indices for nonempty tiles to be processed
        """

        nonempty_indices = tuple(zip(*(tuple(indices) for indices in np.where(self.nonempty_mask))))

        return nonempty_indices

    @cached_property
    def nonempty_tiles(self):

        """ Get a list of nonempty tiles to be processed.

        Returns
        -------
            nonempty_tiles : list
                Nonempty tiles to be processed.
        """

        nonempty_tiles = self[self.nonempty_mask].tolist()

        return nonempty_tiles

    @cached_property
    def tile_indices(self):

        """ Calculate scaled tile indices.

        Returns
        -------
            tile_indices : tuple of numpy.ndarray
                Scaled tile indices.
        """

        scales = self.scales
        profile_tile_indices = self.profile.tile_indices
        tile_indices = (np.rint(profile_tile_indices[0] * scales[0]).astype(int),
                        np.rint(profile_tile_indices[1] * scales[1]).astype(int))

        return tile_indices

    @cached_property
    def border_indices(self):

        """ Calculate scaled border indices.

        Returns
        -------
            border_indices : tuple of numpy.ndarray
                Scaled border indices.
        """

        scales = self.scales
        profile_border_indices = self.profile.border_indices
        border_indices = (np.rint(profile_border_indices[0] * scales[0]).astype(int),
                          np.rint(profile_border_indices[1] * scales[1]).astype(int))

        return border_indices

    @cached_property
    def indices_iterator(self):

        """ Get a Tiled iterator for array indices.

        Returns
        -------
            indices_iterator : IndicesIterator
                Tiled iterator for array indices.
        """

        indices_iterator = IndicesIterator(self)

        return indices_iterator

    @cached_property
    def tile_indices_iterator(self):

        """ Get a Tiled iterator for tile indices.

        Returns
        -------
            tile_indices_iterator : TileIndicesIterator
                Tiled iterator for tile indices.
        """

        tile_indices_iterator = TileIndicesIterator(self)

        return tile_indices_iterator

    @cached_property
    def border_indices_iterator(self):

        """ Get a Tiled iterator for border indices.

        Returns
        -------
            border_indices_iterator : BorderIndicesIterator
                Tiled iterator for border indices.
        """

        border_indices_iterator = BorderIndicesIterator(self)

        return border_indices_iterator

    @cached_property
    def stitch_indices_iterator(self):

        """ Get a Tiled iterator for stitch indices.

        Returns
        -------
            stitch_indices_iterator : StitchIndicesIterator
                Tiled iterator for stitch indices.
        """

        stitch_indices_iterator = StitchIndicesIterator(self)

        return stitch_indices_iterator

    @cached_property
    def s(self):

        """ Get the Slice object for tile-wise slicing.

        Returns
        -------
            s : Slice
                Slice object for tile-wise slicing.
        """

        s = Slice(self)

        return s

    @cached_property
    def m(self):

        """ Get the Mask object for masking.

        Returns
        -------
            m : Mask
                Mask object for masking.
        """

        m = Mask(self)

        return m


class Stitched(Data):

    """ numpy.ndarray subclass for storing DeepTile stitched data.
    """

    def __new__(cls, stitched, job, otype):

        """ Create new Stitched object.

        Parameters
        ----------
            stitched : numpy.ndarray
                Stitched object.
            job : Job
                Job that generated this stitched object.
            otype : str
                Stitched object type.

        Returns
        -------
            stitched : Stitched
                Stitched object.
        """

        stitched = super().__new__(cls, stitched, job, otype, ALLOWED_STITCHED_TYPES)

        return stitched

    def __array_finalize__(self, stitched):

        """ Finalize Stitched object.

        Parameters
        ----------
            stitched : numpy.ndarray
                Stitched object.
        """

        if stitched is None:
            return
        self.dt = getattr(stitched, 'dt', None)
        self.profile = getattr(stitched, 'profile', None)
        self.job = getattr(stitched, 'job', None)
        self.otype = getattr(stitched, 'otype', None)


class Slice:

    """ Slice class for tile-wise slicing.

    Parameters
    ----------
        tiles : Tiled
            Array of tiles.
    """

    def __init__(self, tiles):

        self.tiles = tiles

    def __getitem__(self, slices):

        """ Apply slices to each tile.

        Parameters
        ----------
            slices : tuple
                Tuple of slice objects designating slices to be extracted.

        Returns
        -------
            sliced_tiles : Tiled
                Sliced array of tiles.
        """

        sliced_tiles = np.empty_like(self.tiles)
        sliced_tiles.slices = self.tiles.slices + [slices]
        nonempty_indices = self.tiles.nonempty_indices
        nonempty_tiles = self.tiles.nonempty_tiles

        for index, tile in zip(nonempty_indices, nonempty_tiles):
            sliced_tiles[index] = tile[slices]

        return sliced_tiles


class Mask:

    """ Mask class for masking Tiled objects.

    Parameters
    ----------
        tiles : Tiled
            Array of tiles.
    """

    def __init__(self, tiles):

        self.tiles = tiles

    def __getitem__(self, mask):

        """ Mask Tiled object.

        Parameters
        ----------
            mask : numpy.ndarray
                Boolean mask.

        Returns
        -------
            masked_tiles : Tiled
                Masked array of tiles.
        """

        masked_tiles = np.empty_like(self.tiles)
        masked_tiles[mask] = self.tiles[mask]
        masked_tiles.mask = mask

        return masked_tiles
