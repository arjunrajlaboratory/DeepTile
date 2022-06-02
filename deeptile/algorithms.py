from inspect import signature


class AlgorithmBase:

    def __init__(self, batch, default_batch_size):

        self.batch = batch
        self.default_batch_size = default_batch_size

    @classmethod
    def set_func(cls, func):

        class Algorithm(cls):

            __call__ = staticmethod(func)

        return Algorithm


def transform(func, default_batch_size=None):

    if 'batch_size' in signature(func).parameters.keys():
        batch = True
        if default_batch_size is None:
            default_batch_size = 8
    else:
        batch = False

    return AlgorithmBase.set_func(func)(batch, default_batch_size)
