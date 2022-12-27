def test_stitch_image():

    """ Test that DeepTile can stitch an image. """

    import numpy as np
    import tifffile
    from deeptile import load
    from deeptile.extensions import stitch
    from pathlib import Path

    image = str(Path(__file__).parents[1] / 'data' / 'sample.nd2')
    ground_truth = tifffile.imread(str(Path(__file__).parents[1] / 'data' / 'sample_nd2_stitched_sigma_10.tif'))
    ground_truth_no_blend = \
        tifffile.imread(str(Path(__file__).parents[1] / 'data' / 'sample_nd2_stitched_no_blend.tif'))

    dt = load(image)
    tiles = dt.get_tiles()
    stitched = stitch.stitch_image(tiles, blend=True, sigma=10)

    assert stitched.dtype is ground_truth.dtype
    assert np.all(stitched == ground_truth)

    stitched_no_blend = stitch.stitch_image(tiles, blend=False)

    assert stitched_no_blend.dtype is ground_truth_no_blend.dtype
    assert np.all(stitched_no_blend == ground_truth_no_blend)


def test_stitch_masks():

    """ Test that DeepTile can stitch masks. """

    import numpy as np
    import pickle
    import tifffile
    from deeptile import load
    from deeptile.extensions import stitch
    from pathlib import Path

    with open(str(Path(__file__).parents[1] / 'data' / 'sample_tif_masks'), 'rb') as f:
        masks = pickle.load(f)

    ground_truth = \
        tifffile.imread(str(Path(__file__).parents[1] / 'data' / 'sample_tif_stitched_mask_iou_threshold_10.tif'))

    dt = load(np.zeros((1022, 1024)))
    tiles = dt.get_tiles(tile_size=(300, 300), overlap=(0.1, 0.1))
    tiles[:] = masks
    stitched = stitch.stitch_masks(tiles, iou_threshold=0.1)

    assert np.all(stitched == ground_truth)
