import numpy as np

ALLOWED_TILED_TYPES = ('tiled_image', 'tiled_coords')
ALLOWED_STITCHED_TYPES = ('stitched_image', 'stitched_coords')


class Tiled(np.ndarray):

    """ numpy.ndarray subclass for storing tiled data.
    """

    def __new__(cls, tiles, dt, otype):

        """ Create new Tiled object.

        Parameters
        ----------
            tiles : numpy.ndarray or Tiled
                Array of tiles.
            dt : DeepTile
                DeepTile instance used to generate this object.
            otype : str
                Tiled object type.

        Returns
        -------
            tiles : Tiled
                Array of tiles.
        """

        tiles = np.asarray(tiles).view(cls)
        if otype not in ALLOWED_TILED_TYPES:
            raise ValueError("Invalid tiled object type.")
        else:
            tiles.dt = dt
            tiles.otype = otype

        return tiles

    def __array_finalize__(self, tiles):

        if tiles is None:
            return
        self.dt = getattr(tiles, 'dt', None)
        self.otype = getattr(tiles, 'otype', None)


class Stitched(np.ndarray):

    """ numpy.ndarray subclass for storing stitched data.
    """

    def __new__(cls, stitched, dt, otype):

        """ Create new Stitched object.

        Parameters
        ----------
            stitched : numpy.ndarray or Stitched
                Stitched array.
            dt : DeepTile
                DeepTile instance used to generate this object.
            otype : str
                Stitched object type.

        Returns
        -------
            stitched : Stitched
                Stitched array.
        """

        stitched = np.asarray(stitched).view(cls)
        if otype not in ALLOWED_STITCHED_TYPES:
            raise ValueError("Invalid stitched object type.")
        else:
            stitched.dt = dt
            stitched.otype = otype

        return stitched

    def __array_finalize__(self, stitched):

        if stitched is None:
            return
        self.dt = getattr(stitched, 'dt', None)
        self.otype = getattr(stitched, 'otype', None)
