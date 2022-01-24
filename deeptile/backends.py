import numpy as np


def create_app(algorithm):

    if algorithm == 'cellori':
        from cellori import Cellori
        app = Cellori

    if algorithm == 'cellpose':
        from cellpose.models import Cellpose
        from cellpose.io import logger_setup
        logger_setup()
        app = Cellpose

    if algorithm in ['deepcell_mesmer', 'deepcell_cytoplasm', 'deepcell_nuclear']:
        from deepcell import applications
        if algorithm == 'deepcell_mesmer':
            app = applications.Mesmer
        if algorithm == 'deepcell_cytoplasm':
            app = applications.CytoplasmSegmentation
        if algorithm == 'deepcell_nuclear':
            app = applications.NuclearSegmentation

    return app


def segment_tile(tile, app, model_parameters, eval_parameters, image_shape, mask_shape):

    mask = None

    if app.__name__ == 'Cellori':
        mask_list = list()
        for tile_frame in tile:
            mask_list.append(app(tile_frame, **model_parameters).segment(**eval_parameters)[0])
        mask = np.stack(mask_list)

    if app.__name__ == 'Cellpose':
        mask_list = list()
        for tile_frame in tile:
            mask_list.append(app(**model_parameters).eval(tile_frame, channels=[1, 1], tile=False,
                                                          **eval_parameters)[0])
        mask = np.stack(mask_list)

    if app.__name__ in ['Mesmer']:
        model = app(**model_parameters)
        tile = tile.reshape(-1, 2, *tile.shape[-2:])
        tile = np.moveaxis(tile, 1, -1)
        tile = np.expand_dims(tile, axis=1)
        mask_list = list()
        for tile_frame in tile:
            mask_list.append(model.predict(tile_frame, **eval_parameters)[0])
        mask = np.stack(mask_list)
        mask = np.moveaxis(mask, -1, 0)
        if mask.shape[0] == 1:
            mask_shape = (*image_shape[:-3], *image_shape[-2:])
        elif mask.shape[0] == 2:
            mask_shape = (2, *image_shape[:-3], *image_shape[-2:])

    if app.__name__ in ['NuclearSegmentation', 'CytoplasmSegmentation']:
        model = app(**model_parameters)
        mask_list = list()
        for tile_frame in tile:
            tile_frame = np.expand_dims(tile_frame, axis=-1)
            tile_frame = np.expand_dims(tile_frame, axis=0)
            mask_list.append(model.predict(tile_frame, **eval_parameters)[0, :, :, 0])
        mask = np.stack(mask_list)

    mask = np.squeeze(mask)

    if mask_shape is None:
        mask_shape = image_shape

    return mask, mask_shape
