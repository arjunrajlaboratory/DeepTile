import numpy as np
from deeptile import segment_image
from tifffile import imread


def test_conflicting_segmentations():
    image = imread('test_data/wm989.tif')
    masks, indices = segment_image(image, (1, 2), (0.2, 0.2), 'Cellori', dict())
    tile1_roi = masks[0, 0][903:927, 1164:1185]
    tile2_roi = masks[0, 1][903:927, 30:51]

    return len(np.unique(tile1_roi)) == len(np.unique(tile2_roi))
