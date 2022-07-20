from deeptile.algorithms import partial, transform


def tile_coords(coords):

    def func(tile_index, _coords):

        s = (tile_index[0, 0] < _coords[:, 0]) & (_coords[:, 0] < tile_index[0, 1]) & \
            (tile_index[1, 0] < _coords[:, 1]) & (_coords[:, 1] < tile_index[1, 1])
        _coords = _coords[s] - tile_index[:, 0]

        return _coords

    func = transform(partial(func, _coords=coords), vectorized=False,
                     input_type='tile_index_iterator', output_type='tiled_coords')

    return func
