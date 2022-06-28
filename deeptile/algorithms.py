from deeptile.data import ALLOWED_TILED_TYPES, ALLOWED_STITCHED_TYPES

ALLOWED_INPUT_TYPES = ALLOWED_TILED_TYPES
ALLOWED_OUTPUT_TYPES = ALLOWED_TILED_TYPES + ALLOWED_STITCHED_TYPES


class AlgorithmBase:

    """ AlgorithmBase class for use in tile processing and stitching.

    Parameters
    ----------
        vectorized : bool
            Whether the algorithm is vectorized to support batching.
        default_batch_size : int or None
            Default number of tiles in each batch.
        algorithm_type : str
            Type of algorithm.
        input_type : str
            Object type of algorithm input.
        output_type : str
            Object type of algorithm output.
    """

    def __init__(self, vectorized, default_batch_size, algorithm_type, input_type, output_type):

        self.vectorized = vectorized
        self.default_batch_size = default_batch_size
        self.algorithm_type = algorithm_type
        self.input_type = input_type
        self.output_type = output_type

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

        Returns
        -------
            Algorithm : class
                Algorithm class with ``func`` as the callable.
        """

        class Algorithm(cls):

            __call__ = staticmethod(func)

        return Algorithm


def transform(func, vectorized=False, default_batch_size=8, input_type='tiled_image', output_type='tiled_image'):

    """ Transform callable into an instance of the Algorithm class.

    Parameters
    ----------
        func : callable
            Callable for use in tile processing and stitching.
        vectorized : bool
            Whether the algorithm is vectorized to support batching.
        default_batch_size : int or None, optional, default None
            Default number of tiles in each batch. If ``vectorized`` is ``False``, this value is set to ``None``.
        input_type : str
            Object type of algorithm input.
        output_type : str
            Object type of algorithm output.

    Returns
    -------
        transformed_func : Algorithm
            Function transformed into an Algorithm object.
    """

    if not vectorized:
        default_batch_size = None

    if input_type not in ALLOWED_INPUT_TYPES:
        raise ValueError("Invalid input object type.")

    if output_type not in ALLOWED_OUTPUT_TYPES:
        raise ValueError("Invalid output object type.")

    algorithm_type = None
    if output_type in ALLOWED_TILED_TYPES:
        algorithm_type = 'process'
    elif output_type in ALLOWED_STITCHED_TYPES:
        algorithm_type = 'stitch'

    transformed_func = AlgorithmBase.set_callable(func)(vectorized, default_batch_size,
                                                        algorithm_type, input_type, output_type)

    return transformed_func
