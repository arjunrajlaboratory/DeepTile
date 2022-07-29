from deeptile.core.algorithms import partial, transform
from deeptile.core.data import _scan_arguments, _get_new_arguments, _remove_tiles, Tiled
from functools import wraps
import numpy as np


def lift(func):

    """ Lift function to be applied on all tiles.

    Parameters
    ----------
        func : Callable
            Callable for use in tile processing.

    Returns
    ------
        lifted_func : Callable
            Lifted function.
    """

    @wraps(func)
    def lifted_func(*args, **kwargs):

        new_args, new_kwargs, arg_indices, kwarg_indices, inputs = _scan_arguments(args, kwargs)

        tiles = inputs[0]
        input_type = tiles.otype
        if func is np.broadcast_arrays:
            output_type = (input_type, ) * len([i for i in arg_indices if isinstance(i, int)])
        else:
            output_type = input_type

        @partial(transform, input_type=(input_type, 'index_iterator'), output_type=output_type)
        def transformed_func(tile):

            tile, tile_index = tile

            _new_args, _new_kwargs = _get_new_arguments(args, kwargs, new_args, new_kwargs,
                                                        arg_indices, kwarg_indices, tile_index)

            processed_tile = func(*new_args, **new_kwargs)

            return processed_tile

        processed_tiles = tiles.dt.process((tiles, tiles.index_iterator), transformed_func)

        new_args, new_kwargs = _remove_tiles(new_args, new_kwargs, arg_indices, kwarg_indices)

        if isinstance(processed_tiles, Tiled):
            job = processed_tiles.job
        else:
            job = processed_tiles[0].job
        job.type = 'lifted_function'
        job.kwargs = {
            'func': func,
            'args': new_args,
            'kwargs': new_kwargs
        }
        if tiles.dt.link_data:
            job.input = inputs

        return processed_tiles

    return lifted_func
