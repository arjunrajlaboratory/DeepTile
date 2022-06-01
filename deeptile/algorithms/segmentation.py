import numpy as np
from deeptile.algorithms.base import transform


def cellori_segmentation(model_parameters, eval_parameters):

    from cellori import Cellori

    def func_segment(tile):

        def algorithm(tile_frame): return Cellori(tile_frame, **model_parameters).segment(**eval_parameters)[0]
        mask = _process_tile_by_frame(algorithm, tile)

        return mask

    func_segment = transform(func_segment)

    return func_segment


def cellpose_segmentation(model_parameters, eval_parameters):

    from cellpose.models import Cellpose
    from cellpose.io import logger_setup
    logger_setup()

    def func_segment(tile):

        model = Cellpose(**model_parameters)

        def algorithm(tile_frame): return model.eval(tile_frame, channels=[1, 1], tile=False, **eval_parameters)[0]
        mask = _process_tile_by_frame(algorithm, tile)

        return mask

    func_segment = transform(func_segment)

    return func_segment


def deepcell_mesmer_segmentation(model_parameters, eval_parameters):

    from deepcell.applications import Mesmer

    def func_segment(tile):

        model = Mesmer(**model_parameters)
        tile = tile.reshape(-1, 2, *tile.shape[-2:])
        tile = np.moveaxis(tile, 1, -1)
        tile = np.expand_dims(tile, axis=1)

        def algorithm(tile_frame): return model.predict(tile_frame, **eval_parameters)[0]
        mask = _process_tile_by_frame(algorithm, tile)
        mask = np.moveaxis(mask, -1, 0)

        return mask

    func_segment = transform(func_segment, default_batch_size=8)

    return func_segment


def deepcell_nuclear_segmentation(model_parameters, eval_parameters):

    from deepcell.applications import NuclearSegmentation

    def func_segment(tile):

        model = NuclearSegmentation(**model_parameters)
        tile = np.expand_dims(tile, axis=-1)
        tile = np.expand_dims(tile, axis=1)

        def algorithm(tile_frame): return model.predict(tile_frame, **eval_parameters)[0, :, :, 0]
        mask = _process_tile_by_frame(algorithm, tile)

        return mask

    func_segment = transform(func_segment, default_batch_size=8)

    return func_segment


def deepcell_cytoplasm_segmentation(model_parameters, eval_parameters):

    from deepcell.applications import CytoplasmSegmentation

    def func_segment(tile):

        model = CytoplasmSegmentation(**model_parameters)
        tile = np.expand_dims(tile, axis=-1)
        tile = np.expand_dims(tile, axis=1)

        def algorithm(tile_frame): return model.predict(tile_frame, **eval_parameters)[0, :, :, 0]
        mask = _process_tile_by_frame(algorithm, tile)

        return mask

    func_segment = transform(func_segment, default_batch_size=8)

    return func_segment


def _process_tile_by_frame(algorithm, tile):

    mask_list = list()
    for tile_frame in tile:
        mask_list.append(algorithm(tile_frame))
    mask = np.stack(mask_list)

    return mask
