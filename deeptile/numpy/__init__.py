import numpy as _np
from deeptile.numpy._lift import _lift

_funcs = (_np.reshape, _np.ravel,
          _np.moveaxis, _np.rollaxis, _np.swapaxes, _np.transpose,
          _np.broadcast_to, _np.broadcast_arrays, _np.expand_dims, _np.squeeze,
          _np.concatenate, _np.stack, _np.block, _np.vstack, _np.hstack, _np.dstack, _np.column_stack, _np.row_stack,
          _np.tile, _np.repeat,
          _np.delete, _np.insert, _np.append, _np.resize,
          _np.flip, _np.fliplr, _np.flipud, _np.roll, _np.rot90)

for _func in _funcs:
    globals()[_func.__name__] = _lift(_func)
