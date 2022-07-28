import numpy as _np
from deeptile.numpy._lift import _lift

_ufuncs = _np.core.umath.__all__
_funcs = ('empty_like', 'ones_like', 'zeros_like', 'full_like',
          'packbits', 'unpackbits',
          'reshape', 'ravel',
          'moveaxis', 'rollaxis', 'swapaxes', 'transpose',
          'broadcast_to', 'broadcast_arrays', 'expand_dims', 'squeeze',
          'concatenate', 'stack', 'block', 'vstack', 'hstack', 'dstack', 'column_stack', 'row_stack',
          'tile', 'repeat',
          'delete', 'insert', 'append', 'resize',
          'flip', 'fliplr', 'flipud', 'roll', 'rot90')

for _ufunc in _ufuncs:
    globals()[_ufunc] = getattr(_np, _ufunc)

for _func in _funcs:
    globals()[_func] = _lift(getattr(_np, _func))
