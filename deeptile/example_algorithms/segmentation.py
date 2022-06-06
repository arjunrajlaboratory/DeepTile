import numpy as np
from deeptile.algorithms import transform


def cellori_segmentation(model_parameters, eval_parameters):

    from cellori import Cellori

    def func_segment(tile):

        mask = Cellori(tile, **model_parameters).segment(**eval_parameters)[0]

        return mask

    func_segment = transform(func_segment, batch=False)

    return func_segment


def cellpose_segmentation(model_parameters, eval_parameters):

    from cellpose.models import Cellpose
    from cellpose.io import logger_setup
    logger_setup()

    model = Cellpose(**model_parameters)

    def func_segment(tile):

        mask = model.eval(tile, tile=False, **eval_parameters)[0]

        return mask

    func_segment = transform(func_segment, batch=False)

    return func_segment


def deepcell_mesmer_segmentation(model_parameters, eval_parameters):

    from deepcell.applications import Mesmer

    model = Mesmer(**model_parameters)

    def func_segment(tile):

        tile = np.moveaxis(tile, 1, -1)
        mask = model.predict(tile, batch_size=tile.shape[0], **eval_parameters)
        mask = np.moveaxis(mask, -1, 1)

        return mask

    func_segment = transform(func_segment, batch=True, default_batch_size=8)

    return func_segment


def deepcell_nuclear_segmentation(model_parameters, eval_parameters):

    from deepcell.applications import NuclearSegmentation

    model = NuclearSegmentation(**model_parameters)

    def func_segment(tile):

        tile = np.expand_dims(tile, axis=-1)
        mask = model.predict(tile, batch_size=tile.shape[0], **eval_parameters)[:, :, :, 0]

        return mask

    func_segment = transform(func_segment, batch=True, default_batch_size=8)

    return func_segment


def deepcell_cytoplasm_segmentation(model_parameters, eval_parameters):

    from deepcell.applications import CytoplasmSegmentation

    model = CytoplasmSegmentation(**model_parameters)

    def func_segment(tile):

        tile = np.expand_dims(tile, axis=-1)
        mask = model.predict(tile, batch_size=tile.shape[0], **eval_parameters)[:, :, :, 0]

        return mask

    func_segment = transform(func_segment, batch=True, default_batch_size=8)

    return func_segment
