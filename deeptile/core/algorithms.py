from deeptile.core.types import ALLOWED_INPUT_TYPES, ALLOWED_TILED_TYPES, ALLOWED_STITCHED_TYPES
from deeptile.core.utils import to_tuple
from functools import partial as _partial
from inspect import signature
from types import FunctionType


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

        Parameters
        ----------
            tile
                Tile to be processed.

        Raises
        ------
            NotImplementedError
                If no callable has been set.
        """

        raise NotImplementedError("no callable has been set.")

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


def transform(func, input_type='tiled_image', output_type='tiled_image', default_batch_size=8):

    """ Transform callable into an instance of the Algorithm class.

    Parameters
    ----------
        func : callable
            Callable for use in tile processing and stitching.
        input_type : str or tuple of str
            Object type of algorithm input.
        output_type : str or tuple of str
            Object type of algorithm output.
        default_batch_size : int or None, optional, default None
            Default number of tiles in each batch. If ``func`` does not support batching, this value is set to ``None``.

    Returns
    -------
        transformed_func : Algorithm
            Function transformed into an Algorithm object.

    Raises
    ------
        ValueError
            If ``func`` has both tile and tiles arguments for the Tiled input.
        ValueError
            If ``func`` has no argument for the Tiled input.
        ValueError
            If ``input_type`` is invalid.
        ValueError
            If ``output_type`` is invalid.
        ValueError
            If ``default_batch_size`` is invalid.
    """

    arg_names = signature(func).parameters.keys()

    if ('tile' in arg_names) and ('tiles' in arg_names):
        raise ValueError('func has both tile and tiles arguments for the Tiled input.')

    algorithm_type = None
    allowed_output_types = ()
    batching = None

    if isinstance(output_type, str):
        first_otype = output_type
    else:
        first_otype = output_type[0]

    if first_otype in ALLOWED_TILED_TYPES:

        algorithm_type = 'process'
        allowed_output_types = ALLOWED_TILED_TYPES

        if 'tile' in arg_names:
            batching = False
        elif 'tiles' in arg_names:
            batching = True
        else:
            raise ValueError('func has no argument for the Tiled input.')

    elif first_otype in ALLOWED_STITCHED_TYPES:

        algorithm_type = 'stitch'
        allowed_output_types = ALLOWED_STITCHED_TYPES
        batching = False

        if 'tiles' not in arg_names:
            raise ValueError('func has no argument for the Tiled input.')

    for otype in to_tuple(input_type):
        if otype not in ALLOWED_INPUT_TYPES:
            raise ValueError("invalid input object type.")

    for otype in to_tuple(output_type):
        if otype not in allowed_output_types:
            raise ValueError("invalid output object type.")

    if not batching:
        default_batch_size = None
    elif not isinstance(default_batch_size, int):
        raise ValueError("invalid default batch size.")

    algorithm_kwargs = {
        'algorithm_type': algorithm_type,
        'input_type': input_type,
        'output_type': output_type,
        'batching': batching,
        'default_batch_size': default_batch_size
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
        raise ValueError("invalid function type.")

    return func
