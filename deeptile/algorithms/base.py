class Algorithm:

    def __init__(self, func, batch=False, default_batch_size=None):

        self.func = func
        self.batch = batch
        self.default_batch_size = default_batch_size

    def __call__(self, *args, **kwargs):

        return self.func(*args, **kwargs)
