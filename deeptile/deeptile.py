from deeptile.core import utils
from deeptile.core.data import Tiled
from deeptile.core.jobs import Job
from deeptile.core.profiles import Profile


class DeepTile:

    """ Base DeepTile class.

    Parameters
    ----------
        image
            An object containing an image.
    """

    def __init__(self, image):

        self.image = image
        self.image_type = None
        self.image_shape = None
        self.dask = None
        self.link_data = None
        self.profiles = []


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

    def get_tiles(self, tile_size, overlap=(0.1, 0.1), slices=(slice(None), )):

        """ Split array into tiles.

        Parameters
        ----------
            tile_size : tuple
                Size of each tile.
            overlap : tuple, optional, default (0.1, 0.1)
                Fractions of ``tile_size`` to use for overlap.
            slices : optional, default (slice(None))
                Tuple of slice objects designating slices to be extracted.

        Returns
        -------
            tiles : Tiled
                Array of tiles.
        """

        job_locals = locals()
        job_locals.pop('self')

        image = self.image[slices]
        self.image_shape = image.shape

        tiling, tile_indices, border_indices = utils.calculate_indices(self.image_shape, tile_size, overlap)
        tiles = utils.array_split_2d(image, tile_indices)
        tiles = utils.cast_list_to_array_2d(tiles)
        nonempty_indices = utils.get_nonempty_indices(tiles)

        profile = Profile(self, tiling, tile_size, overlap, slices, nonempty_indices, tile_indices, border_indices)
        job = Job(self.image, 'get_tiles', job_locals, profile)
        tiles = Tiled(tiles, job, stackable=False, tile_scales=(1.0, 1.0))

        return tiles


class DeepTileFunction(DeepTile):

    """ DeepTile subclass for functions.

    Parameters
    ----------
        image : Callable
            A function for obtaining image regions.
        image_shape : tuple
            Shape of the image.
    """

    def __init__(self, image, image_shape):

        super().__init__(image)
        self.image_type = 'function'
        self.image_shape = image_shape

    def get_tiles(self, tile_size, overlap=(0.1, 0.1), slices=(slice(None), )):

        """ Obtain tiles from function calls.

        Parameters
        ----------
            tile_size : tuple
                Size of each tile.
            overlap : tuple, optional, default (0.1, 0.1)
                Fractions of ``tile_size`` to use for overlap.
            slices : optional, default (slice(None))
                Tuple of slice objects designating slices to be extracted.

        Returns
        -------
            tiles : Tiled
                Array of tiles.
        """

        job_locals = locals()
        job_locals.pop('self')

        from deeptile.sources import function

        tiling, tile_indices, border_indices = utils.calculate_indices(self.image_shape, tile_size, overlap)
        tiles = function.parse(self.image, self.image_shape, tiling, tile_indices, slices)
        nonempty_indices = utils.get_nonempty_indices(tiles)

        profile = Profile(self, tiling, tile_size, overlap, slices, nonempty_indices, tile_indices, border_indices)
        job = Job(self.image, 'get_tiles', job_locals, profile)
        tiles = Tiled(tiles, job, stackable=False, tile_scales=(1.0, 1.0))

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

        job_locals = locals()
        job_locals.pop('self')

        from deeptile.sources import large_image

        self.image_shape = (self.image.getMetadata()['sizeY'], self.image.getMetadata()['sizeX'])
        tiles, tiling, tile_indices, border_indices = large_image.parse(self.image, self.image_shape,
                                                                        tile_size, overlap, slices)
        nonempty_indices = utils.get_nonempty_indices(tiles)

        profile = Profile(self, tiling, tile_size, overlap, slices, nonempty_indices, tile_indices, border_indices)
        job = Job(self.image, 'get_tiles', job_locals, profile)
        tiles = Tiled(tiles, job, stackable=False, tile_scales=(1.0, 1.0))

        return tiles


class DeepTileND2(DeepTile):

    """ DeepTile subclass for ND2 files.

    Parameters
    ----------
        image : str
            An ND2 file.
    """

    def __init__(self, image):

        super().__init__(image)
        self.image_type = 'nd2'
        self.axis_sizes = None
        self.axis_order = None

    def get_tiles(self, overlap=(0.1, 0.1), slices=(slice(None))):

        """ Obtain tiles from ND2 file.

        Parameters
        ----------
            overlap : tuple or None, optional, default (0.1, 0.1)
                Fractions to use for overlap. If ``None``, overlap is automatically determined from the ND2 metadata.
            slices : optional, default (slice(None))
                Tuple of slice objects designating slices to be extracted.

        Returns
        -------
            tiles : Tiled
                Array of tiles.
        """

        job_locals = locals()
        job_locals.pop('self')

        from deeptile.sources import nd2

        tiles, tiling, tile_size, overlap, self.image_shape = nd2.parse(self.image, self.dask, self.axis_sizes,
                                                                        self.axis_order, overlap, slices)
        _, tile_indices, border_indices = utils.calculate_indices(self.image_shape, tile_size, overlap)
        nonempty_indices = utils.get_nonempty_indices(tiles)

        profile = Profile(self, tiling, tile_size, overlap, slices, nonempty_indices, tile_indices, border_indices)
        job = Job(self.image, 'get_tiles', job_locals, profile)
        tiles = Tiled(tiles, job, tile_scales=(1.0, 1.0))

        return tiles
