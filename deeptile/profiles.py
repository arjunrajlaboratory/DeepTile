class Profile:

    """ Profile class that stores tiling profiles created by a DeepTile object.

    Parameters
    ----------
        dt : DeepTile
            DeepTile object.
    """

    def __init__(self, dt, **kwargs):

        self.dt = dt
        self.id = len(dt.profiles)
        self.__dict__.update((key, value) for key, value in kwargs.items())
        self.jobs = []
        self.data_arrays = []

        dt.profiles.append(self)
