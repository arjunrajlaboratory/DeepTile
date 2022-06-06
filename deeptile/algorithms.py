class AlgorithmBase:

    def __init__(self, vectorized, default_batch_size):

        self.vectorized = vectorized
        self.default_batch_size = default_batch_size

    def __call__(self, tile):

        raise NotImplementedError("No callable function has been set.")

    @classmethod
    def set_func(cls, func):

        class Algorithm(cls):

            __call__ = staticmethod(func)

        return Algorithm


def transform(func, vectorized, default_batch_size=8):

    if not vectorized:
        default_batch_size = None

    return AlgorithmBase.set_func(func)(vectorized, default_batch_size)
