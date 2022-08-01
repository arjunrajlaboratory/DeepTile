from deeptile.core import trees
from deeptile.core.algorithms import partial, transform
from deeptile.core.data import Tiled
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

        arg_indices = [arg_index for arg_index in trees.tree_scan(args)[1]
                       if isinstance(trees.tree_index(args, arg_index), Tiled)]
        kwarg_indices = [kwarg_index for kwarg_index in trees.tree_scan(kwargs)[1]
                         if isinstance(trees.tree_index(kwargs, kwarg_index), Tiled)]
        inputs = [trees.tree_index(args, arg_index) for arg_index in arg_indices] + \
                 [trees.tree_index(kwargs, kwarg_index) for kwarg_index in kwarg_indices]

        tiles = inputs[0]
        input_type = tiles.otype
        if func is np.broadcast_arrays:
            output_type = (input_type,) * len([arg_index for arg_index in arg_indices if len(arg_index) == 1])
        else:
            output_type = input_type

        @partial(transform, input_type=(input_type, 'index_iterator'), output_type=output_type)
        def transformed_func(tile):

            tile, tile_index = tile

            new_args = trees.tree_apply(args, arg_indices, lambda tiles: tiles[tile_index])
            new_kwargs = trees.tree_apply(kwargs, kwarg_indices, lambda tiles: tiles[tile_index])

            processed_tile = func(*new_args, **new_kwargs)

            return processed_tile

        processed_tiles = tiles.dt.process((tiles, tiles.index_iterator), transformed_func)

        lite_args = trees.tree_apply(args, arg_indices, lambda tiles: Tiled)
        lite_kwargs = trees.tree_apply(kwargs, kwarg_indices, lambda tiles: Tiled)

        if isinstance(processed_tiles, Tiled):
            job = processed_tiles.job
        else:
            job = processed_tiles[0].job
        job.type = 'lifted_function'
        job.kwargs = {
            'func': func,
            'args': lite_args,
            'kwargs': lite_kwargs
        }
        if tiles.dt.link_data:
            job.input = inputs

        return processed_tiles

    return lifted_func
