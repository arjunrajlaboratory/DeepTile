import numpy as np
from functools import cached_property


class Profile:

    """ Profile class that stores tiling profiles created by a DeepTile object.

    Parameters
    ----------
        dt : DeepTile
            DeepTile object.
        tiling : tuple of int
            Number of tiles in each dimension.
        tile_size : tuple of int
            Size of each tile.
        overlap : tuple of float
            Fractions of ``tile_size`` to use for overlap.
        slices
            Slices to be extracted.
        nonempty_indices : tuple of tuple
            Indices of nonempty tiles.
        tile_indices : tuple of numpy.ndarray
            Indices of tiles.
        border_indices : tuple of numpy.ndarray
            Indices of borders at the middle of tile overlaps.
    """

    def __init__(self, dt, tiling, tile_size, overlap, slices, nonempty_indices, tile_indices, border_indices):

        self.dt = dt
        self.id = len(dt.profiles)
        self.tiling = tiling
        self.tile_size = tile_size
        self.overlap = overlap
        self.slices = slices
        self.nonempty_indices = nonempty_indices
        self.tile_indices = tile_indices
        self.border_indices = border_indices
        self.jobs = []

        dt.profiles.append(self)

    @cached_property
    def nonempty_mask(self):

        """ Get a mask for nonempty tiles.

        Returns
        -------
            nonempty_mask : numpy.ndarray
                Mask for nonempty tiles.
        """

        nonempty_mask = np.zeros(self.tiling, dtype=bool)
        nonempty_mask[tuple(zip(*self.nonempty_indices))] = True

        return nonempty_mask
