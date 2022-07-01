class Profile:

    """ Profile class that stores tiling profiles created by a DeepTile object.

    Parameters
    ----------
        dt : DeepTile
            DeepTile object.
        **profile_kwargs : dict
            Profile keyword arguments.
    """

    def __init__(self, dt, **profile_kwargs):

        self.dt = dt
        self.id = len(dt.profiles)
        self.__dict__.update((k, v) for k, v in profile_kwargs.items())
        self.jobs = []

        dt.profiles.append(self)
