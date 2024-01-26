import numpy as np
from deeptile.core.data import Output
from deeptile.core.lift import lift
from deeptile.core.utils import compute_dask
from functools import partial
from rasterio import features


def cellpose_segmentation(model_parameters, eval_parameters, output_format='masks'):

    """Generate lifted function for the Cellpose segmentation algorithm.

    Parameters
    ----------
    model_parameters : dict
        Dictionary of model parameters.
    eval_parameters : dict
        Dictionary of evaluation parameters.
    output_format : str, optional
        Format of the output. Supported formats are 'masks' and 'polygons'. Default is 'masks'.

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
    def _func_segment(tile, index, tile_index, stitch_index, tiling):

        tile = compute_dask(tile)
        mask = model.eval(tile, tile=False, **eval_parameters)[0]

        if output_format == 'masks':
            return mask
        elif output_format == 'polygons':
            polygons = mask_to_polygons(mask, index, tile_index, stitch_index, tiling)
            return polygons

    def func_segment(tiles):

        return _func_segment(tiles, tiles.index_iterator, tiles.tile_indices_iterator, tiles.stitch_indices_iterator,
                             tiles.profile.tiling)

    return func_segment


def deepcell_mesmer_segmentation(model_parameters, eval_parameters):

    """Generate lifted function for the DeepCell Mesmer segmentation algorithm.

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

    """Generate lifted function for the DeepCell Nuclear segmentation algorithm.

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

    """Generate lifted function for the DeepCell Cytoplasm segmentation algorithm.

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


def mask_to_polygons(mask, index, tile_index, stitch_index, tiling):

    i_tile, j_tile = tile_index
    _, _, i_stitch, j_stitch = stitch_index

    offset = np.array((j_tile[0], i_tile[0]))

    valid = np.unique(mask[i_stitch[0]:i_stitch[1], j_stitch[0]:j_stitch[1]])
    valid = set(valid[valid > 0])

    raw_polygons = {int(polygon[1]): np.array(polygon[0]['coordinates'][0]).astype(int) + offset for polygon in
                    features.shapes(mask, mask > 0)}

    border_polygons = ({}, {})
    border = set()
    delta = 5
    if index[0] > 0:
        new_border = np.unique(mask[i_stitch[0]:i_stitch[0] + delta, j_stitch[0]:j_stitch[1]])
        new_border = set(new_border[new_border > 0])
        new_border = new_border - border
        border = border | new_border
        border_polygons[0][index[0] - 1] = [raw_polygons[i] for i in new_border]
    if index[0] < tiling[0] - 1:
        new_border = np.unique(mask[i_stitch[1] - delta:i_stitch[1], j_stitch[0]:j_stitch[1]])
        new_border = set(new_border[new_border > 0])
        new_border = new_border - border
        border = border | new_border
        border_polygons[0][index[0]] = [raw_polygons[i] for i in new_border]
    if index[1] > 0:
        new_border = np.unique(mask[i_stitch[0]:i_stitch[1], j_stitch[0]:j_stitch[0] + delta])
        new_border = set(new_border[new_border > 0])
        new_border = new_border - border
        border = border | new_border
        border_polygons[1][index[1] - 1] = [raw_polygons[i] for i in new_border]
    if index[1] < tiling[1] - 1:
        new_border = np.unique(mask[i_stitch[0]:i_stitch[1], j_stitch[1] - delta:j_stitch[1]])
        new_border = set(new_border[new_border > 0])
        new_border = new_border - border
        border = border | new_border
        border_polygons[1][index[1]] = [raw_polygons[i] for i in new_border]

    valid = valid - border
    valid_polygons = [raw_polygons[i] for i in valid]
    polygons = (valid_polygons, border_polygons)
    polygons = Output(polygons, isimage=False, stackable=False)

    return polygons
