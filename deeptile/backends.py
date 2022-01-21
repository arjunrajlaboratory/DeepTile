import numpy as np


def create_app(dt):

    if dt.algorithm == 'cellori':
        from cellori import Cellori
        dt.app = Cellori

    if dt.algorithm in ['cellpose_cytoplasm', 'cellpose_nuclear']:
        from cellpose.models import Cellpose
        from cellpose.io import logger_setup
        logger_setup()
        dt.app = Cellpose

    if dt.algorithm in ['deepcell_mesmer', 'deepcell_cytoplasm', 'deepcell_nuclear']:
        from deepcell import applications
        if dt.algorithm == 'deepcell_mesmer':
            dt.app = applications.Mesmer
        if dt.algorithm == 'deepcell_cytoplasm':
            dt.app = applications.CytoplasmSegmentation
        if dt.algorithm == 'deepcell_nuclear':
            dt.app = applications.NuclearSegmentation


def segment_tile(dt, tile):

    mask = None

    if dt.app.__name__ == 'Cellori':
        mask_list = list()
        for tile_frame in tile:
            mask_list.append(dt.app(tile_frame).segment(**dt.parameters)[0])
        mask = np.stack(mask_list)

    if dt.app.__name__ == 'Cellpose':
        model_type = None
        if dt.algorithm == 'cellpose_cytoplasm':
            model_type = 'cyto'
        elif dt.algorithm == 'cellpose_nuclear':
            model_type = 'nuclei'
        app = dt.app(model_type=model_type)
        mask_list = list()
        for tile_frame in tile:
            mask_list.append(app.eval(tile_frame, channels=[1, 1], diameter=None, tile=False, **dt.parameters)[0])
        mask = np.stack(mask_list)

    if dt.app.__name__ in ['Mesmer']:
        app = dt.app()
        tile = tile.reshape(-1, 2, *tile.shape[-2:])
        tile = np.moveaxis(tile, 1, -1)
        tile = np.expand_dims(tile, axis=1)
        mask_list = list()
        for tile_frame in tile:
            mask_list.append(app.predict(tile_frame, **dt.parameters)[0])
        mask = np.stack(mask_list)
        mask = np.moveaxis(mask, -1, 0)
        dt.mask_shape = (mask.shape[0], *dt.image_shape[:-3], *dt.image_shape[-2:])
        mask = mask.reshape(-1, *mask.shape[-2:])

    if dt.app.__name__ in ['NuclearSegmentation', 'CytoplasmSegmentation']:
        app = dt.app()
        mask_list = list()
        for tile_frame in tile:
            tile_frame = np.expand_dims(tile_frame, axis=-1)
            tile_frame = np.expand_dims(tile_frame, axis=0)
            mask_list.append(app.predict(tile_frame, **dt.parameters)[0, :, :, 0])
        mask = np.stack(mask_list)

    if dt.mask_shape is None:
        dt.mask_shape = dt.image_shape

    return mask
