from deeptile.data import ALLOWED_TILED_TYPES, ALLOWED_STITCHED_TYPES
from deeptile.iterators import ALLOWED_ITERATOR_TYPES
from deeptile.utils import to_tuple
from functools import partial as _partial
from types import FunctionType

ALLOWED_INPUT_TYPES = ALLOWED_TILED_TYPES + ALLOWED_ITERATOR_TYPES


class AlgorithmBase:

    """ AlgorithmBase class for use in tile processing and stitching.

    Parameters
    ----------
        **algorithm_kwargs : dict
            Algorithm keyword arguments.
    """

    def __init__(self, **algorithm_kwargs):

        self.__dict__.update((k, v) for k, v in algorithm_kwargs.items())

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
        input_type : str or tuple of str
            Object type of algorithm input.
        output_type : str or tuple of str
            Object type of algorithm output.

    Returns
    -------
        transformed_func : Algorithm
            Function transformed into an Algorithm object.

    Raises
    ------
        ValueError
            If ``input_type`` is invalid.
        ValueError
            If ``output_type`` is invalid.
    """

    if not vectorized:
        default_batch_size = None

    for otype in to_tuple(input_type):
        if otype not in ALLOWED_INPUT_TYPES:
            raise ValueError("Invalid input object type.")

    algorithm_type = None
    allowed_output_types = ()
    if isinstance(output_type, tuple):
        first_otype = output_type[0]
    else:
        first_otype = output_type
    if first_otype in ALLOWED_TILED_TYPES:
        algorithm_type = 'process'
        allowed_output_types = ALLOWED_TILED_TYPES
    elif first_otype in ALLOWED_STITCHED_TYPES:
        algorithm_type = 'stitch'
        allowed_output_types = ALLOWED_STITCHED_TYPES
    for otype in to_tuple(output_type):
        if otype not in allowed_output_types:
            raise ValueError("Invalid output object type.")

    algorithm_kwargs = {
        'vectorized': vectorized,
        'default_batch_size': default_batch_size,
        'algorithm_type': algorithm_type,
        'input_type': input_type,
        'output_type': output_type
    }

    transformed_func = AlgorithmBase.set_callable(func)(**algorithm_kwargs)

    return transformed_func


def partial(func, *args, **kwargs):

    """ Generate new function with partial application of given arguments and keywords.

    Parameters
    ----------
        func : callable
            Callable function.

    Returns
    -------
        func : callable
            Callable partial function.
    """

    if isinstance(func, FunctionType):
        func = _partial(func, *args, **kwargs)
    elif isinstance(func, AlgorithmBase):
        func = func.set_callable(_partial(func.__call__, *args, **kwargs))(**vars(func))
    else:
        raise ValueError("Invalid function type.")

    return func
