def test_load_dask_array():

    """ Test that DeepTile can load a Dask array. """

    import dask.array as da
    import numpy as np
    from deeptile import load

    image = np.random.rand(10, 10)
    image = da.from_array(image)
    dt = load(image, dask=True)

    assert dt.dask is True
    assert dt.image is image

    dt_no_dask = load(image, dask=False)

    assert dt_no_dask.dask is True
    assert dt_no_dask.image is image


def test_load_numpy_array():

    """ Test that DeepTile can load a NumPy array. """

    import dask.array as da
    import numpy as np
    from deeptile import load

    image = np.random.rand(10, 10)
    dt = load(image, dask=True)

    assert dt.dask is True
    assert isinstance(dt.image, da.Array)
    assert np.all(dt.image == image)

    dt_no_dask = load(image, dask=False)

    assert dt_no_dask.dask is False
    assert dt_no_dask.image is image


def test_from_large_image():

    """ Test that DeepTile can load a large_image tile source. """

    import pytest

    try:
        import large_image
    except ImportError:
        pytest.skip('large_image not installed.')

    import dask.array as da
    import large_image
    import numpy as np
    import tifffile
    from deeptile.io import from_large_image
    from pathlib import Path

    image = str(Path(__file__).parents[1] / 'data' / 'sample.tif')
    ground_truth = tifffile.imread(image)

    ts = large_image.getTileSource(image)
    dt = from_large_image(ts)
    tiles = dt.get_tiles(ground_truth.shape[-2:])

    assert dt.dask is None
    assert dt.image is ts
    assert isinstance(tiles[0, 0], da.Array)
    assert np.all(tiles[0, 0] == ground_truth)


def test_load_nd2():

    """ Test that DeepTile can load an ND2 file. """

    import dask.array as da
    import nd2
    import numpy as np
    from deeptile import load
    from pathlib import Path

    image = str(Path(__file__).parents[1] / 'data' / 'sample.nd2')
    ground_truth = nd2.ND2File(image)

    dt = load(image, dask=True)
    tiles = dt.get_tiles(ground_truth.shape[-2:])

    assert dt.dask is True
    assert dt.image.path == ground_truth.path
    assert isinstance(tiles[tiles.nonempty_indices_tuples[0]], da.Array)

    dt_no_dask = load(image, dask=False)
    tiles_no_dask = dt_no_dask.get_tiles(ground_truth.shape[-2:])

    assert dt_no_dask.dask is False
    assert dt_no_dask.image.path == ground_truth.path
    assert isinstance(tiles_no_dask[tiles_no_dask.nonempty_indices_tuples[0]], np.ndarray)


def test_load_tiff():

    """ Test that DeepTile can load a TIFF file. """

    import dask.array as da
    import numpy as np
    import tifffile
    from deeptile import load
    from pathlib import Path

    image = str(Path(__file__).parents[1] / 'data' / 'sample.tif')
    ground_truth = tifffile.imread(image)

    dt = load(image, dask=True)

    assert dt.dask is True
    assert dt.image.dtype is ground_truth.dtype
    assert isinstance(dt.image, da.Array)
    assert np.all(dt.image == ground_truth)

    dt_no_dask = load(image, dask=False)

    assert dt_no_dask.dask is False
    assert dt_no_dask.image.dtype is ground_truth.dtype
    assert isinstance(dt_no_dask.image, np.ndarray)
    assert np.all(dt_no_dask.image == ground_truth)
