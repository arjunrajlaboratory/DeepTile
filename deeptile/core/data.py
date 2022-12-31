import numpy as np
from dask.array import Array
from deeptile.core import trees, utils
from deeptile.core.iterators import IndexIterator, TileIndicesIterator, BorderIndicesIterator, StitchIndicesIterator
from deeptile.core.jobs import Job
from functools import cached_property, partial


class Output:

    """ Output class for attaching metadata to function outputs.

    Parameters
    ----------
        output : numpy.ndarray
            Output array.
        **kwargs : dict
            Metadata associated with this function output.

    Returns
    -------
        output : Output
            Output array.
    """

    def __init__(self, output, **metadata):

        self.output = output
        self.metadata = metadata


class Data(np.ndarray):

    """ numpy.ndarray subclass for storing DeepTile data.
    """

    def __new__(cls, data, job):

        """ Create new Data object.

        Parameters
        ----------
            data : numpy.ndarray or Data
                Data array.
            job : Job
                Job that generated this data object.

        Returns
        -------
            data : Data
                Data array.
        """

        data = np.asarray(data).view(cls)

        data.dt = job.dt
        data.profile = job.profile
        data.job = job

        if data.dt.link_data:
            data.job.output = data

        return data


class Tiled(Data):

    """ numpy.ndarray subclass for storing DeepTile tiled data.
    """

    def __new__(cls, tiles, job, mask=None, isimage=True, stackable=False, tile_scales=None):

        """ Create new Tiled object.

        Parameters
        ----------
            tiles : numpy.ndarray or Tiled
                Array of tiles.
            job : Job
                Job that generated this tiled object.
            mask : numpy.ndarray or None, optional, default None
                Boolean mask. If ``None``, a boolean array with all True values will be used.
            isimage : bool, optional, default True
                Whether each tile is an image.
            stackable : bool, optional, default True
                Whether tiles can be stacked.
            tile_scales : tuple of float or None, optional, default None
                Tile scales relative to profile tile sizes. If ``None``, the values will be inferred.

        Returns
        -------
            tiles : Tiled
                Array of tiles.
        """

        tiles = super().__new__(cls, tiles, job)
        tiles.parent = tiles
        tiles.slices = []
        if mask is None:
            mask = np.ones(tiles.profile.tiling, dtype=bool)
        tiles.mask = mask
        tiles.metadata = {
            'isimage': isimage,
            'stackable': stackable,
            'tile_scales': tile_scales
        }

        return tiles

    def __array_finalize__(self, tiles, **kwargs):

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
        self.parent = getattr(tiles, 'parent', None)
        self.slices = getattr(tiles, 'slices', None)
        self.metadata = getattr(tiles, 'metadata', None)
        self.mask = getattr(tiles, 'mask', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        """ Process tiles using a NumPy universal function.

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

        from deeptile.core import process

        args = (ufunc, method, *inputs)
        arg_indices = [arg_index for arg_index in trees.tree_scan(args)[2]
                       if isinstance(trees.tree_index(args, arg_index), Tiled)]

        job_locals = {
            'args': trees.tree_apply(args, arg_indices, lambda ts: Tiled),
            'kwargs': kwargs
        }
        job = Job(inputs, 'lifted_array_ufunc', job_locals)

        tiles = [inp for inp in inputs if isinstance(inp, Tiled)]
        process.check_compatability(tiles)

        reference = tiles[0]
        nonempty_indices = reference.nonempty_indices
        processed_istree = None
        processed_indices = None
        processed_tiles = None

        for index in zip(*nonempty_indices):

            processed_istree, processed_indices, processed_tiles = \
                process.process_single(reference[index].__array_ufunc__, False,
                                       args, kwargs, arg_indices, [],
                                       job, reference, processed_istree, processed_indices, processed_tiles,
                                       index)

        return processed_tiles

    def __array_function__(self, func, types, args, kwargs):

        """ Process tiles using an arbitrary NumPy function.

        Parameters
        ----------
            func : Callable
                Arbitrary callable exposed by NumPy’s public API, which was called in the form func(*args, **kwargs).
            types : tuple
                types is a collection collections.abc.Collection of unique argument types from the original NumPy
                function call that implement __array_function__.
            args : tuple
                Tuple of arguments directly passed on from the original call.
            kwargs : dict
                Dictionary of keyword arguments directly passed on from the original call.

        Returns
        -------
            processed_tiles : Tiled
                Array of tiles after processing with ``func``.
        """

        from deeptile.core import process

        args = (func, types, args, kwargs)
        arg_indices = [arg_index for arg_index in trees.tree_scan(args)[2]
                       if isinstance(trees.tree_index(args, arg_index), Tiled)]
        inputs = [trees.tree_index(args, arg_index) for arg_index in arg_indices]

        job_locals = {
            'args': trees.tree_apply(args, arg_indices, lambda ts: Tiled),
            'kwargs': {}
        }
        job = Job(inputs, 'lifted_array_function', job_locals)

        tiles = [inp for inp in inputs if isinstance(inp, Tiled)]
        process.check_compatability(tiles)

        reference = tiles[0]
        nonempty_indices = reference.nonempty_indices
        processed_istree = None
        processed_indices = None
        processed_tiles = None

        for index in zip(*nonempty_indices):

            processed_istree, processed_indices, processed_tiles = \
                process.process_single(lambda _func, _types, _args, _kwargs:
                                       reference[index].__array_function__(_func, _types, tuple(_args), _kwargs), False,
                                       args, {}, arg_indices, [],
                                       job, reference, processed_istree, processed_indices, processed_tiles,
                                       index)

        return processed_tiles

    def compute(self, batch_axis=False, pad_final_batch=False, batch_size=None, **kwargs):

        """ Compute Dask arrays.

        Parameters
        ----------
            batch_axis : bool, optional, default False
                Whether to use the first axis to create batches.
            pad_final_batch : bool, optional, default False
                Whether to pad the final batch to the specified ``batch_size``.
            batch_size : int or None, optional, default None
                Number of tiles in each batch.

        Returns
        -------
            tiles : Tiled
                Array of tiles.
        """

        from deeptile.core.lift import lift

        if isinstance(self[self.nonempty_indices_tuples[0]], Array):

            tiles = lift(lambda tile: Output(tile.compute(**kwargs), **self.metadata),
                         batch_axis=batch_axis, pad_final_batch=pad_final_batch, batch_size=batch_size)(self)
            tiles.job.type = 'compute_dask'
            tiles.job.locals = kwargs

        else:

            tiles = self

        return tiles

    def persist(self, batch_axis=False, pad_final_batch=False, batch_size=None, **kwargs):

        """ Persist Dask arrays into memory.

        Parameters
        ----------
            batch_axis : bool, optional, default False
                Whether to use the first axis to create batches.
            pad_final_batch : bool, optional, default False
                Whether to pad the final batch to the specified ``batch_size``.
            batch_size : int or None, optional, default None
                Number of tiles in each batch.

        Returns
        -------
            tiles : Tiled
                Array of tiles.
        """

        from deeptile.core.lift import lift

        if isinstance(self[self.nonempty_indices_tuples[0]], Array):

            tiles = lift(lambda tile: Output(tile.persist(**kwargs), **self.metadata),
                         batch_axis=batch_axis, pad_final_batch=pad_final_batch, batch_size=batch_size)(self)
            tiles.job.type = 'persist_dask'
            tiles.job.locals = kwargs

        else:

            tiles = self

        return tiles

    def pad(self, **kwargs):

        """ Pad tiles to the same size.

        Returns
        -------
            tiles : Tiled
                Array of tiles.

        Raises
        ------
        NotImplementedError
            If padding ``mode`` is not supported.
    """

        mode = kwargs.get('mode', 'constant')

        if mode not in ['constant', 'edge', 'linear_ramp', 'reflect', 'symmetric']:

            raise NotImplementedError('Padding mode is not supported.')

        if self.metadata['isimage'] and not self.metadata['stackable']:

            tiles = np.array(self)

            tile_size = self.tile_size
            tile_indices = self.tile_indices
            tile_padding = (tile_size[0] - (tile_indices[0][-1, 1] - tile_indices[0][-1, 0]),
                            tile_size[1] - (tile_indices[1][-1, 1] - tile_indices[1][-1, 0]))

            if tile_padding[0] > 0:
                if (mode in ['reflect', 'symmetric']) and (tiles.shape[0] > 1):
                    for i, (inner_tile, edge_tile) in enumerate(np.moveaxis(tiles[-2:], 1, 0)):
                        if edge_tile is not None:
                            if inner_tile is not None:
                                extension = self.profile.tile_indices[0][-1, 0] - self.profile.tile_indices[0][-2, 0]
                                extended_tile = np.concatenate((inner_tile[..., :extension, :], edge_tile), axis=-2)
                                padded_tile = utils.array_pad(extended_tile, tile_padding[0], -2, **kwargs)
                                padded_tile = padded_tile[..., -tile_size[0]:, :]
                            else:
                                padded_tile = utils.array_pad(edge_tile, tile_padding[0], -2, **kwargs)
                            tiles[-1, i] = padded_tile
                else:
                    for i, tile in enumerate(tiles[-1]):
                        if tile is not None:
                            tiles[-1, i] = utils.array_pad(tile, tile_padding[0], -2, **kwargs)

            if tile_padding[1] > 0:
                if (mode in ['reflect', 'symmetric']) and (tiles.shape[1] > 1):
                    for i, (inner_tile, edge_tile) in enumerate(tiles[:, -2:]):
                        if edge_tile is not None:
                            if inner_tile is not None:
                                extension = self.profile.tile_indices[1][-1, 0] - self.profile.tile_indices[1][-2, 0]
                                extended_tile = np.concatenate((inner_tile[..., :extension], edge_tile), axis=-1)
                                padded_tile = utils.array_pad(extended_tile, tile_padding[1], -1, **kwargs)
                                padded_tile = padded_tile[..., -tile_size[1]:]
                            else:
                                padded_tile = utils.array_pad(edge_tile, tile_padding[0], -1, **kwargs)
                            tiles[i, -1] = padded_tile
                else:
                    for i, tile in enumerate(tiles[:, -1]):
                        if tile is not None:
                            tiles[i, -1] = utils.array_pad(tile, tile_padding[1], -1, **kwargs)

            job = Job(self, 'pad_tiles', kwargs)

            metadata = self.metadata.copy()
            metadata['stackable'] = True
            tiles = Tiled(tiles, job, self.mask, **metadata)

        else:

            tiles = self

        return tiles

    def unpad(self):

        """ Unpad tiles.

        Returns
        -------
            tiles : Tiled
                Array of tiles.
        """

        if self.metadata['isimage'] and self.metadata['stackable']:

            tiles = np.array(self)

            tile_indices = self.tile_indices

            for i, tile in enumerate(tiles[-1]):
                tiles[-1, i] = tile[..., :tile_indices[0][-1, 1] - tile_indices[0][-1, 0], :]
            for i, tile in enumerate(tiles[:, -1]):
                tiles[i, -1] = tile[..., :tile_indices[1][-1, 1] - tile_indices[1][-1, 0]]

            job = Job(self, 'unpad_tiles', {})

            metadata = self.metadata.copy()
            metadata['stackable'] = False
            tiles = Tiled(tiles, job, self.mask, **metadata)

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

        job_locals = locals()
        job_locals.pop('self')
        job_locals.pop('data')

        from deeptile.core.lift import lift

        if data_type == 'image':
            tiles = lift(partial(utils.tile_image, image=data))(self.tile_indices_iterator)
        elif data_type == 'coords':
            tiles = lift(partial(utils.tile_coords, coords=data))(self.tile_indices_iterator)
        else:
            raise ValueError("invalid data object type.")

        tiles.job.type = 'import_data'
        tiles.job.locals = job_locals
        if self.dt.link_data:
            tiles.job.input = data
        else:
            tiles.job.input = None

        return tiles

    @cached_property
    def tile_size(self):

        """ Calculate scaled tile size.

        Returns
        -------
            tile_size : tuple of int
                Scaled tile size.
        """

        profile_tile_size = self.profile.tile_size

        if self.metadata['isimage']:
            if self.metadata['stackable']:
                tile_size = self[self.nonempty_indices_tuples[0]].shape[-2:]
            else:
                tile_size = profile_tile_size
        else:
            tile_scales = self.metadata['tile_scales']
            if tile_scales is None:
                tile_size = profile_tile_size
            else:
                tile_size = (round(profile_tile_size[0] * tile_scales[0]), round(profile_tile_size[1] * tile_scales[1]))

        return tile_size

    @cached_property
    def tile_scales(self):

        """ Calculate tile scales relative to profile tile sizes.

        Returns
        -------
            image_scales : tuple of float
                Tile scales relative to profile tile sizes.
        """

        profile_tile_size = self.profile.tile_size
        tile_size = self.tile_size
        tile_scales = (tile_size[0] / profile_tile_size[0], tile_size[1] / profile_tile_size[1])

        return tile_scales

    @cached_property
    def image_size(self):

        """ Calculate scaled image size.

        Returns
        -------
            image_size : tuple of int
                Scaled image size.
        """

        profile_image_shape = self.dt.image_shape
        tile_scales = self.tile_scales
        image_size = (round(profile_image_shape[-2] * tile_scales[0]), round(profile_image_shape[-1] * tile_scales[1]))

        return image_size

    @cached_property
    def image_scales(self):

        """ Calculate image scales relative to profile image size.

        Returns
        -------
            image_scales : tuple of float
                Image scales relative to profile image size.
        """

        profile_image_shape = self.dt.image_shape
        image_size = self.image_size
        image_scales = (image_size[0] / profile_image_shape[-2], image_size[1] / profile_image_shape[-1])

        return image_scales

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
            raise ValueError("no nonempty tiles to process.")

        return nonempty_mask

    @cached_property
    def nonempty_indices(self):

        """ Get arrays of indices for nonempty tiles to be processed.

        Returns
        -------
            nonempty_indices : tuple of numpy.ndarray
                Arrays of indices for nonempty tiles to be processed.
        """

        nonempty_indices = np.where(self.nonempty_mask)

        return nonempty_indices

    @cached_property
    def nonempty_indices_tuples(self):

        """ Get tuples of indices for nonempty tiles to be processed.

        Returns
        -------
            nonempty_indices_tuples : tuple of tuple
                Tuples of indices for nonempty tiles to be processed.
        """

        nonempty_indices_tuples = tuple(zip(*(tuple(indices) for indices in self.nonempty_indices)))

        return nonempty_indices_tuples

    @cached_property
    def tile_indices(self):

        """ Calculate scaled tile indices.

        Returns
        -------
            tile_indices : tuple of numpy.ndarray
                Scaled tile indices.
        """

        image_scales = self.image_scales
        profile_tile_indices = self.profile.tile_indices
        tile_indices = (np.rint(profile_tile_indices[0] * image_scales[0]).astype(int),
                        np.rint(profile_tile_indices[1] * image_scales[1]).astype(int))

        return tile_indices

    @cached_property
    def border_indices(self):

        """ Calculate scaled border indices.

        Returns
        -------
            border_indices : tuple of numpy.ndarray
                Scaled border indices.
        """

        image_scales = self.image_scales
        profile_border_indices = self.profile.border_indices
        border_indices = (np.rint(profile_border_indices[0] * image_scales[0]).astype(int),
                          np.rint(profile_border_indices[1] * image_scales[1]).astype(int))

        return border_indices

    @cached_property
    def index_iterator(self):

        """ Get a Tiled iterator for array indices.

        Returns
        -------
            index_iterator : IndexIterator
                Tiled iterator for array indices.
        """

        index_iterator = IndexIterator(self)

        return index_iterator

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

    def __new__(cls, stitched, job):

        """ Create new Stitched object.

        Parameters
        ----------
            stitched : numpy.ndarray
                Stitched object.
            job : Job
                Job that generated this stitched object.

        Returns
        -------
            stitched : Stitched
                Stitched object.
        """

        stitched = super().__new__(cls, stitched, job)

        return stitched

    def __array_finalize__(self, stitched, **kwargs):

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

        sliced_tiles = self.tiles.copy()
        sliced_tiles[:] = None
        sliced_tiles.slices = self.tiles.slices + [slices]
        nonempty_indices = self.tiles.nonempty_indices_tuples

        for index in nonempty_indices:
            sliced_tiles[index] = self.tiles[index][slices]

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

        mask = np.broadcast_to(mask, self.tiles.shape).astype(bool)

        masked_tiles = self.tiles.copy()
        masked_tiles[:] = None
        masked_tiles[mask] = self.tiles[mask]
        masked_tiles.mask = mask

        return masked_tiles
