class AlgorithmBase:

    """ AlgorithmBase class for use in tile processing and stitching.

    Parameters
    ----------
        vectorized : bool
            Whether the algorithm is vectorized to support batching.
        default_batch_size : int or None
            Default number of tiles in each batch.
    """

    def __init__(self, vectorized, default_batch_size):

        self.vectorized = vectorized
        self.default_batch_size = default_batch_size

    def __call__(self, tile):

        """ Placeholder callable.

        Raises
        ------
            NotImplementedError
                If no callable has been set.
        """

        raise NotImplementedError("No callable has been set.")

    @classmethod
    def set_callable(cls, func):

        """ Set callable.

        Parameters
        ----------
            func : callable
                Callable for use in tile processing and stitching.
        """

        class Algorithm(cls):

            __call__ = staticmethod(func)

        return Algorithm


def transform(func, vectorized, default_batch_size=8):

    """ Transform callable into an instance of the Algorithm class.

    Parameters
    ----------
        func : callable
            Callable for use in tile processing and stitching.
        vectorized : bool
            Whether the algorithm is vectorized to support batching.
        default_batch_size : int or None, optional, default 8
            Default number of tiles in each batch. If ``vectorized`` is ``False``, this value is set to ``None``.
    """

    if not vectorized:
        default_batch_size = None

    return AlgorithmBase.set_callable(func)(vectorized, default_batch_size)
