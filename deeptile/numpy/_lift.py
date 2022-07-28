from collections.abc import Sequence
from deeptile.core.algorithms import transform
from functools import wraps
from inspect import signature


def _lift(func):

    """ (For internal use) Lift NumPy function to be applied on all tiles.

    Parameters
    ----------
        func : Callable
            NumPy function.

    Returns
    ------
        lifted_func : Callable
            Lifted NumPy function.
    """

    closure = getattr(func, '__closure__', None)
    if closure is None:
        parameters = signature(func).parameters
    else:
        parameters = signature(func.__closure__[0].cell_contents).parameters
    arg_names = list(parameters.keys())

    if parameters[arg_names.pop(0)].kind.name == 'VAR_POSITIONAL':

        @wraps(func)
        def lifted_func(*tiles, **kwargs):

            input_type = tuple(t.otype for t in tiles)
            if len(set(input_type)) > 1:
                raise ValueError("tile contains multiple object types.")

            transformed_func = transform(lambda tile: func(*tile, **kwargs),
                                         input_type=input_type, output_type=input_type)

            dt = tiles[0].dt
            processed_tiles = dt.process(tiles, transformed_func)
            job = processed_tiles[0].job
            job.type = 'array_func'
            job.kwargs = {
                'func': func,
                'tiles': tiles,
                'kwargs': kwargs
            }

            return processed_tiles

    else:

        @wraps(func)
        def lifted_func(tiles, *args, **kwargs):

            if isinstance(tiles, Sequence):
                input_type = tuple(t.otype for t in tiles)
                if len(set(input_type)) > 1:
                    raise ValueError("tile contains multiple object types.")
                output_type = input_type[0]
            else:
                input_type = tiles.otype
                output_type = input_type

            transformed_func = transform(lambda tile: func(tile, *args, **kwargs),
                                         input_type=input_type, output_type=output_type)

            dt = tiles[0].dt
            processed_tiles = dt.process(tiles, transformed_func)
            processed_tiles.job.type = 'array_func'
            processed_tiles.job.kwargs = {
                'func': func,
                'tiles': tiles,
                'args': args,
                'kwargs': kwargs
            }

            return processed_tiles

    return lifted_func
