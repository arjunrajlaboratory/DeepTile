from deeptile.core import process, trees
from deeptile.core.data import Tiled
from deeptile.core.iterators import Iterator
from deeptile.core.jobs import Job
from functools import wraps
import numpy as np


def lift(func, vectorized=False, batch_axis=False, pad_final_batch=False, batch_size=4):

    """ Lift function to be applied on all tiles.

    Parameters
    ----------
        func : Callable
            Callable for use in tile processing.
        vectorized : bool, optional, default False
            Whether the algorithm is vectorized to support batching.
        batch_axis : bool, optional, default False
            Whether to use the first axis to create batches.
        pad_final_batch : bool, optional, default False
            Whether to pad the final batch to the specified ``batch_size``. If ``func_process`` does not support
            batching, this value is ignored.
        batch_size : int, optional, default 4
            Number of tiles in each batch. If ``func`` is not vectorized, this value is ignored.

    Returns
    -------
        lifted_func : Callable
            Lifted function.
    """

    @wraps(func)
    def lifted_func(*args, **kwargs):

        job_locals = locals()

        arg_indices = [arg_index for arg_index in trees.tree_scan(args)[2]
                       if isinstance(trees.tree_index(args, arg_index), (Iterator, Tiled))]
        kwarg_indices = [kwarg_index for kwarg_index in trees.tree_scan(kwargs)[2]
                         if isinstance(trees.tree_index(kwargs, kwarg_index), (Iterator, Tiled))]
        inputs = [trees.tree_index(args, arg_index) for arg_index in arg_indices] + \
                 [trees.tree_index(kwargs, kwarg_index) for kwarg_index in kwarg_indices]
        tiles = [inp if isinstance(inp, Tiled) else inp.tiles for inp in inputs]
        process.check_compatability(tiles)

        job_locals['args'] = trees.tree_apply(args, arg_indices, lambda ts: Tiled)
        job_locals['kwargs'] = trees.tree_apply(kwargs, kwarg_indices, lambda ts: Tiled)
        job = Job(inputs, 'lifted_func', job_locals)

        reference = tiles[0]
        profile = reference.profile
        nonempty_indices = reference.nonempty_indices
        processed_istree = None
        processed_indices = None
        processed_tiles = None

        batch_axis_len = None
        if batch_axis:
            batch_axis_len = reference[nonempty_indices[0][0], nonempty_indices[0][1]].shape[0]
            batch_axis_indices = np.tile(np.arange(batch_axis_len), len(nonempty_indices[0]))
            nonempty_indices = [np.repeat(np.array(indices), batch_axis_len, 0) for indices in nonempty_indices]
            nonempty_indices.append(batch_axis_indices)
            nonempty_indices = tuple(nonempty_indices)

        if vectorized:

            n_batches = np.ceil(len(nonempty_indices[0]) / batch_size).astype(int)

            for batch in range(n_batches):

                batch_offset = batch * batch_size
                batch_indices = tuple(indices[batch_offset:batch_offset + batch_size] for indices in nonempty_indices)

                processed_istree, processed_indices, processed_tiles = \
                    process.process_vectorized(func, batch_axis, pad_final_batch, batch_size,
                                               args, kwargs, arg_indices, kwarg_indices,
                                               job, profile, processed_istree, processed_indices, processed_tiles,
                                               batch_indices)

        else:

            for index in zip(*nonempty_indices):

                processed_istree, processed_indices, processed_tiles = \
                    process.process_single(func, batch_axis,
                                           args, kwargs, arg_indices, kwarg_indices,
                                           job, profile, processed_istree, processed_indices, processed_tiles,
                                           index)

        return processed_tiles

    return lifted_func
