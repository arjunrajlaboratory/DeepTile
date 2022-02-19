import numpy as np


def cellori_backend(model_parameters, eval_parameters):

    def f(tile):

        from cellori import Cellori

        def algorithm(tile_frame): return Cellori(tile_frame, **model_parameters).segment(**eval_parameters)[0]
        mask = _process_tile_by_frame(algorithm, tile)

        return mask

    return f


def cellpose_backend(model_parameters, eval_parameters):

    def f(tile):

        from cellpose.models import Cellpose
        from cellpose.io import logger_setup
        logger_setup()

        model = Cellpose(**model_parameters)

        def algorithm(tile_frame): return model.eval(tile_frame, channels=[1, 1], tile=False, **eval_parameters)[0]
        mask = _process_tile_by_frame(algorithm, tile)

        return mask

    return f


def deepcell_mesmer_backend(model_parameters, eval_parameters):

    def f(tile):

        from deepcell.applications import Mesmer

        model = Mesmer(**model_parameters)
        tile = tile.reshape(-1, 2, *tile.shape[-2:])
        tile = np.moveaxis(tile, 1, -1)
        tile = np.expand_dims(tile, axis=1)

        def algorithm(tile_frame): return model.predict(tile_frame, **eval_parameters)[0]
        mask = _process_tile_by_frame(algorithm, tile)
        mask = np.moveaxis(mask, -1, 0)

        return mask

    return f


def deepcell_nuclear_backend(model_parameters, eval_parameters):

    def f(tile):

        from deepcell.applications import NuclearSegmentation

        model = NuclearSegmentation(**model_parameters)
        tile = np.expand_dims(tile, axis=-1)
        tile = np.expand_dims(tile, axis=1)

        def algorithm(tile_frame): return model.predict(tile_frame, **eval_parameters)[0, :, :, 0]
        mask = _process_tile_by_frame(algorithm, tile)

        return mask

    return f


def deepcell_cytoplasm_backend(model_parameters, eval_parameters):

    def f(tile):

        from deepcell.applications import CytoplasmSegmentation

        model = CytoplasmSegmentation(**model_parameters)
        tile = np.expand_dims(tile, axis=-1)
        tile = np.expand_dims(tile, axis=1)

        def algorithm(tile_frame): return model.predict(tile_frame, **eval_parameters)[0, :, :, 0]
        mask = _process_tile_by_frame(algorithm, tile)

        return mask

    return f


def _process_tile_by_frame(algorithm, tile):

    mask_list = list()
    for tile_frame in tile:
        mask_list.append(algorithm(tile_frame))
    mask = np.stack(mask_list)

    return mask
