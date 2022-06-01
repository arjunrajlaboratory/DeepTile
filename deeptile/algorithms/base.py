class AlgorithmBase:

    def __init__(self, batch, default_batch_size):

        self.batch = batch
        self.default_batch_size = default_batch_size

    @classmethod
    def set_func(cls, func):

        class Algorithm(cls):

            __call__ = staticmethod(func)

        return Algorithm


def transform(func, batch=False, default_batch_size=None):

    return AlgorithmBase.set_func(func)(batch, default_batch_size)
