import numpy as np
from deeptile.core.lift import lift
from deeptile.core.utils import compute_dask
from functools import partial


def cellpose_segmentation(model_parameters, eval_parameters):

    """ Generate lifted function for the Cellpose segmentation algorithm.

    Parameters
    ----------
        model_parameters : dict
            Dictionary of model parameters.
        eval_parameters : dict
            Dictionary of evaluation parameters.

    Returns
    -------
        func_segment : Callable
            Lifted function for the Cellpose segmentation algorithm.
    """

    from cellpose.models import Cellpose
    from cellpose.io import logger_setup
    logger_setup()

    model = Cellpose(**model_parameters)

    @lift
    def func_segment(tile):

        tile = compute_dask(tile)
        mask = model.eval(tile, tile=False, **eval_parameters)[0]

        return mask

    return func_segment


def deepcell_mesmer_segmentation(model_parameters, eval_parameters):

    """ Generate lifted function for the DeepCell Mesmer segmentation algorithm.

    Parameters
    ----------
        model_parameters : dict
            Dictionary of model parameters.
        eval_parameters : dict
            Dictionary of evaluation parameters.

    Returns
    -------
        func_segment : Callable
            Lifted function for the DeepCell Mesmer segmentation algorithm.
    """

    from deepcell.applications import Mesmer

    model = Mesmer(**model_parameters)

    @partial(lift, vectorized=True, batch_axis=True)
    def func_segment(tiles):

        tiles = compute_dask(tiles)
        tiles = np.moveaxis(tiles, 1, -1)
        masks = model.predict(tiles, **eval_parameters)
        masks = np.moveaxis(masks, -1, 1)

        return masks

    return func_segment


def deepcell_nuclear_segmentation(model_parameters, eval_parameters):

    """ Generate lifted function for the DeepCell Nuclear segmentation algorithm.

    Parameters
    ----------
        model_parameters : dict
            Dictionary of model parameters.
        eval_parameters : dict
            Dictionary of evaluation parameters.

    Returns
    -------
        func_segment : Callable
            Lifted function for the DeepCell Nuclear segmentation algorithm.
    """

    from deepcell.applications import NuclearSegmentation

    model = NuclearSegmentation(**model_parameters)

    @partial(lift, vectorized=True, batch_axis=True)
    def func_segment(tiles):

        tiles = compute_dask(tiles)
        tiles = np.expand_dims(tiles, axis=-1)
        masks = model.predict(tiles, **eval_parameters)[:, :, :, 0]

        return masks

    return func_segment


def deepcell_cytoplasm_segmentation(model_parameters, eval_parameters):

    """ Generate lifted function for the DeepCell Cytoplasm segmentation algorithm.

    Parameters
    ----------
        model_parameters : dict
            Dictionary of model parameters.
        eval_parameters : dict
            Dictionary of evaluation parameters.

    Returns
    -------
        func_segment : Callable
            Lifted function for the DeepCell Cytoplasm segmentation algorithm.
    """

    from deepcell.applications import CytoplasmSegmentation

    model = CytoplasmSegmentation(**model_parameters)

    @partial(lift, vectorized=True, batch_axis=True)
    def func_segment(tiles):

        tiles = compute_dask(tiles)
        tiles = np.expand_dims(tiles, axis=-1)
        masks = model.predict(tiles, **eval_parameters)[:, :, :, 0]

        return masks

    return func_segment
