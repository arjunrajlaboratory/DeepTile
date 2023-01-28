import numpy as np
from deeptile.core import process, trees
from deeptile.core.data import Tiled
from deeptile.core.iterators import Iterator
from deeptile.core.jobs import Job
from functools import wraps


class Lifted:

    """ Lifted class for functions lifted to be applied on Tiled objects.
    """

    def __new__(cls, func, vectorized, batch_axis, pad_final_batch, batch_size):

        """ Lift function.

        Parameters
        ----------
            func : Callable
                Callable for use in tile processing.
            vectorized : bool
                Whether the algorithm is vectorized to support batching.
            batch_axis : bool
                Whether to use the first axis to create batches.
            pad_final_batch : bool
                Whether to pad the final batch to the specified ``batch_size``. If ``func_process`` does not support
                batching, this value is ignored.
            batch_size : int
                Number of tiles in each batch. If ``func`` is not vectorized, this value is ignored.

        Returns
        -------
            lifted_func : Callable
                Lifted function.
        """

        lifted_func = super().__new__(cls)
        lifted_func.__call__ = wraps(func)(lifted_func)
        lifted_func.func = func
        lifted_func.vectorized = vectorized
        lifted_func.batch_axis = batch_axis
        lifted_func.pad_final_batch = pad_final_batch
        lifted_func.batch_size = batch_size

        return lifted_func

    def __call__(self, *args, **kwargs):

        """ Apply lifted function on Tiled objects.

        Returns
        -------
            processed_tiles
                Tiles processed by lifted function.
        """

        processed_tiles, variables = self.init(*args, **kwargs)

        n_steps = variables['n_steps']
        for _ in range(n_steps):
            processed_tiles, variables = self.apply(processed_tiles, variables)

        return processed_tiles

    def init(self, *args, **kwargs):

        """ Initialize a lifted job.

        Returns
        -------
            processed_tiles
                Tiles processed by lifted function.
            variables : dict
                Dictionary of variables used in the lifted job.
        """

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
        nonempty_indices = reference.nonempty_indices
        processed_istree = None
        processed_indices = None
        processed_tiles = None

        if self.batch_axis:
            batch_axis_len = reference[nonempty_indices[0][0], nonempty_indices[1][0]].shape[0]
            batch_axis_indices = np.tile(np.arange(batch_axis_len), len(nonempty_indices[0]))
            nonempty_indices = [np.repeat(np.array(indices), batch_axis_len, 0) for indices in nonempty_indices]
            nonempty_indices.append(batch_axis_indices)
            nonempty_indices = tuple(nonempty_indices)

        if self.vectorized:
            n_steps = np.ceil(len(nonempty_indices[0]) / self.batch_size).astype(int)
        else:
            n_steps = len(nonempty_indices[0])

        step = 0

        variables = {
            'args': args,
            'kwargs': kwargs,
            'arg_indices': arg_indices,
            'kwarg_indices': kwarg_indices,
            'job': job,
            'reference': reference,
            'nonempty_indices': nonempty_indices,
            'processed_istree': processed_istree,
            'processed_indices': processed_indices,
            'n_steps': n_steps,
            'step': step
        }

        return processed_tiles, variables

    def apply(self, processed_tiles, variables):

        """ Take one step in a lifted job.

        Parameters
        ----------
            processed_tiles
                Tiles processed by lifted function.
            variables : dict
                Dictionary of variables used in the lifted job.

        Returns
        -------
            processed_tiles
                Tiles processed by lifted function.
            variables : dict
                Dictionary of variables used in the lifted job.
        """

        n_steps = variables['n_steps']
        step = variables['step']

        if step < n_steps:

            args = variables['args']
            kwargs = variables['kwargs']
            arg_indices = variables['arg_indices']
            kwarg_indices = variables['kwarg_indices']
            job = variables['job']
            reference = variables['reference']
            nonempty_indices = variables['nonempty_indices']
            processed_istree = variables['processed_istree']
            processed_indices = variables['processed_indices']

            if self.vectorized:

                batch_offset = step * self.batch_size
                batch_indices = tuple(i[batch_offset:batch_offset + self.batch_size] for i in nonempty_indices)

                processed_istree, processed_indices, processed_tiles = \
                    process.process_vectorized(self.func, self.batch_axis, self.pad_final_batch, self.batch_size,
                                               args, kwargs, arg_indices, kwarg_indices,
                                               job, reference, processed_istree, processed_indices, processed_tiles,
                                               batch_indices)

            else:

                index = tuple(i[step] for i in nonempty_indices)
                processed_istree, processed_indices, processed_tiles = \
                    process.process_single(self.func, self.batch_axis,
                                           args, kwargs, arg_indices, kwarg_indices,
                                           job, reference, processed_istree, processed_indices, processed_tiles,
                                           index)

            variables['processed_istree'] = processed_istree
            variables['processed_indices'] = processed_indices
            variables['step'] = step + 1

        return processed_tiles, variables


def lift(func, vectorized=False, batch_axis=False, pad_final_batch=False, batch_size=4):

    """ Lift function to be applied on Tiled objects.

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

    lifted_func = Lifted(func, vectorized, batch_axis, pad_final_batch, batch_size)

    return lifted_func
