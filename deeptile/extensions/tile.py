from deeptile.algorithms import partial, transform


def tile_coords(coords):

    def func_tile(tile_index, _coords):

        s = (tile_index[0, 0] < _coords[:, 0]) & (_coords[:, 0] < tile_index[0, 1]) & \
            (tile_index[1, 0] < _coords[:, 1]) & (_coords[:, 1] < tile_index[1, 1])
        tiles = _coords[s] - tile_index[:, 0]

        return tiles

    func = transform(partial(func_tile, _coords=coords), vectorized=False,
                     input_type='tile_index_iterator', output_type='tiled_coords')

    return func
