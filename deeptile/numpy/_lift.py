from collections.abc import Sequence
from deeptile.core.algorithms import transform, partial
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

    parameters = signature(func.__closure__[0].cell_contents).parameters
    arg_names = list(parameters.keys())

    if parameters[arg_names.pop(0)].kind.name == 'VAR_POSITIONAL':

        @wraps(func)
        def lifted_func(*_tile, **_kwargs):

            input_type = tuple(t.otype for t in _tile)
            if len(set(input_type)) > 1:
                raise ValueError("tile contains multiple object types.")

            transformed_func = transform(lambda tile, **kwargs: func(*tile, **kwargs),
                                         input_type=input_type, output_type=input_type)

            dt = _tile[0].dt
            _tile = dt.process(_tile, partial(transformed_func, **_kwargs))

            return _tile

    else:

        @wraps(func)
        def lifted_func(_tile, *_args, **_kwargs):

            if isinstance(_tile, Sequence):
                input_type = tuple(t.otype for t in _tile)
                if len(set(input_type)) > 1:
                    raise ValueError("tile contains multiple object types.")
                output_type = input_type[0]
            else:
                input_type = _tile.otype
                output_type = input_type

            transformed_func = transform(lambda tile, *args, **kwargs: func(tile, *args, **kwargs),
                                         input_type=input_type, output_type=output_type)
            _kwargs.update({arg_name: _arg for _arg, arg_name in zip(_args, arg_names)})

            dt = _tile[0].dt
            _tile = dt.process(_tile, partial(transformed_func, **_kwargs))

            return _tile

    return lifted_func
