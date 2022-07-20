import numpy as np
from deeptile.iterators import IndicesIterator, TileIndicesIterator, BorderIndicesIterator, StitchIndicesIterator
from deeptile.jobs import Job
from functools import cached_property

ALLOWED_TILED_TYPES = ('tiled_image', 'tiled_coords')
ALLOWED_STITCHED_TYPES = ('stitched_image', 'stitched_coords')


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
        """

        data = np.asarray(data).view(cls)

        if otype not in allowed_otypes:
            raise ValueError("Invalid tiled object type.")

        data.dt = job.dt
        data.profile = job.profile
        data.job = job
        data.id = None
        data.otype = otype

        if data.dt.link_data:
            data.job.output = data

        return data

    def __array_finalize__(self, data):

        if data is None:
            return
        self.dt = getattr(data, 'dt', None)
        self.profile = getattr(data, 'profile', None)
        self.job = getattr(data, 'job', None)
        self.id = getattr(data, 'id', None)
        self.otype = getattr(data, 'otype', None)


class Tiled(Data):

    """ numpy.ndarray subclass for storing DeepTile tiled data.
    """

    def __new__(cls, tiles, job, otype):

        """ Create new Tiled object.

        Parameters
        ----------
            tiles : numpy.ndarray or Tiled
                Array of tiles.
            job : Job
                Job that generated this tiled object.
            otype : str
                Tiled object type.

        Returns
        -------
            tiles : Tiled
                Array of tiles.
        """

        tiles = super().__new__(cls, tiles, job, otype, ALLOWED_TILED_TYPES)
        tiles.parent = tiles
        tiles.slices = []

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
            tile_size = self[profile.nonempty_indices[0]].shape[-2:]
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
    def nonempty_tiles(self):

        """ Get a list of nonempty tiles.

        Returns
        -------
            nonempty_tiles : list
                List of nonempty tiles.
        """

        nonempty_indices = self.profile.nonempty_indices
        nonempty_tiles = self[tuple(zip(*nonempty_indices))]

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

        job = Job(self, 'get_iterator', {}, self.profile)
        indices_iterator = IndicesIterator(self, job)

        return indices_iterator

    @cached_property
    def tile_indices_iterator(self):

        """ Get a Tiled iterator for tile indices.

        Returns
        -------
            tile_indices_iterator : TileIndicesIterator
                Tiled iterator for tile indices.
        """

        job = Job(self, 'get_iterator', {}, self.profile)
        tile_indices_iterator = TileIndicesIterator(self, job)

        return tile_indices_iterator

    @cached_property
    def border_indices_iterator(self):

        """ Get a Tiled iterator for border indices.

        Returns
        -------
            border_indices_iterator : BorderIndicesIterator
                Tiled iterator for border indices.
        """

        job = Job(self, 'get_iterator', {}, self.profile)
        border_indices_iterator = BorderIndicesIterator(self, job)

        return border_indices_iterator

    @cached_property
    def stitch_indices_iterator(self):

        """ Get a Tiled iterator for stitch indices.

        Returns
        -------
            stitch_indices_iterator :
                Tiled iterator for stitch indices.
        """

        job = Job(self, 'get_iterator', {}, self.profile)
        stitch_indices_iterator = StitchIndicesIterator(self, job)

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
                Stitched object.
        """

        sliced_tiles = self.tiles.copy()
        sliced_tiles.parent = self.tiles.parent
        sliced_tiles.slices = self.tiles.slices + [slices]
        nonempty_indices = sliced_tiles.profile.nonempty_indices
        nonempty_tiles = sliced_tiles.nonempty_tiles

        for index, tile in zip(nonempty_indices, nonempty_tiles):
            sliced_tiles[index] = tile[slices]

        return sliced_tiles
