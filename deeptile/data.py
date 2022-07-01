import numpy as np

ALLOWED_TILED_TYPES = ('tiled_image', 'tiled_coords')
ALLOWED_STITCHED_TYPES = ('stitched_image', 'stitched_coords')


class Tiled(np.ndarray):

    """ numpy.ndarray subclass for storing tiled data.
    """

    def __new__(cls, job, tiles, otype):

        """ Create new Tiled object.

        Parameters
        ----------
            job : Job
                Job that generated this tiled object.
            tiles : numpy.ndarray or Tiled
                Array of tiles.
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

        tiles.dt = job.dt
        tiles.profile = job.profile
        tiles.job = job
        tiles.id = None
        tiles.otype = otype

        if tiles.dt.link_data:
            tiles.id = len(job.profile.data)
            tiles.profile.data.append(tiles)
            tiles.job.output = tiles

        return tiles

    def __array_finalize__(self, tiles):

        if tiles is None:
            return
        self.dt = getattr(tiles, 'dt', None)
        self.profile = getattr(tiles, 'profile', None)
        self.job = getattr(tiles, 'job', None)
        self.id = getattr(tiles, 'id', None)
        self.otype = getattr(tiles, 'otype', None)


class Stitched(np.ndarray):

    """ numpy.ndarray subclass for storing stitched data.
    """

    def __new__(cls, job, stitched, otype):

        """ Create new Stitched object.

        Parameters
        ----------
            job : Job
                Job that generated this stitched object.
            stitched : numpy.ndarray or Stitched
                Stitched array.
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

        stitched.dt = job.dt
        stitched.profile = job.profile
        stitched.job = job
        stitched.id = None
        stitched.otype = otype

        if stitched.dt.link_data:
            stitched.id = len(job.profile.data)
            stitched.profile.data.append(stitched)
            stitched.job.output = stitched

        return stitched

    def __array_finalize__(self, stitched):

        if stitched is None:
            return
        self.dt = getattr(stitched, 'dt', None)
        self.job = getattr(stitched, 'job', None)
        self.profile = getattr(stitched, 'profile', None)
        self.id = getattr(stitched, 'id', None)
        self.otype = getattr(stitched, 'otype', None)
