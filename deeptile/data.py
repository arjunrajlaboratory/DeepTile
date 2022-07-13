import numpy as np

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
            data : Tiled
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

    @property
    def s(self):

        """ Get the Slice object for tile-wise slicing.
        """

        return Slice(self)


class Stitched(Data):

    """ numpy.ndarray subclass for storing DeepTile stitched data.
    """

    def __new__(cls, stitched, job, otype):

        """ Create new Stitched object.

        Parameters
        ----------
            stitched : numpy.ndarray or Stitched
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
        tiles : numpy.ndarray or Tiled
            Array of tiles.
    """

    def __init__(self, tiles):

        self.tiles = tiles

    def __getitem__(self, slices):

        """ Apply slices to each tile.

        Parameters
        ----------
            slices : tuple, optional, default (slice(None))
                Tuple of slice objects designating slices to be extracted.

        Returns
        -------
            sliced_tiles : Tiled
                Stitched object.
        """

        sliced_tiles = self.tiles.copy()
        sliced_tiles.parent = self.tiles.parent
        sliced_tiles.slices = self.tiles.slices + [slices]
        nonempty_indices = self.tiles.profile.nonempty_indices
        nonempty_tiles = sliced_tiles[tuple(zip(*nonempty_indices))]

        for index, tile in zip(nonempty_indices, nonempty_tiles):
            sliced_tiles[index] = tile[slices]

        return sliced_tiles
