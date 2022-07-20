import numpy as np

from deeptile.algorithms import transform
from skimage import measure


def stitch_tiles(blend=True, sigma=5):

    """ Generate Algorithm object for a tile stitching algorithm.

    Parameters
    ----------
        blend : bool, optional, default True
            Whether to blend tile overlaps.
        sigma : int, optional, default 5
            Sigma bandwidth parameter used to generate sigmoid taper for blending. If ``blend`` is ``False``, this value
            is ignored.

    Returns
    -------
        func_stitch : Algorithm
            Algorithm object with a tile stitching algorithm as the callable.
    """

    def func_stitch(tiles):

        profile = tiles.profile
        nonempty_indices = profile.nonempty_indices
        first_tile = tiles[nonempty_indices[0]]
        tile_size = first_tile.shape[-2:]
        dtype = first_tile.dtype

        image_shape = tiles.image_shape
        tile_indices_iterator = tiles.tile_indices_iterator
        stitch_shape = (*first_tile.shape[:-2], *image_shape[-2:])
        stitched = np.zeros(stitch_shape)

        if blend:

            avg = np.zeros(stitch_shape[-2:])
            taper = _generate_taper(tile_size, profile.overlap, sigma)

            for index in nonempty_indices:

                tile_index = tile_indices_iterator[index]

                tile = tiles[index]
                stitch_slice = np.s_[..., tile_index[0, 0]:tile_index[0, 1],
                                     tile_index[1, 0]:tile_index[1, 1]]
                tile_slice = np.s_[..., :tile_index[0, 1] - tile_index[0, 0],
                                        :tile_index[1, 1] - tile_index[1, 0]]
                stitched[stitch_slice] = stitched[stitch_slice] + tile[tile_slice] * taper[tile_slice]
                avg[stitch_slice[1:]] = avg[stitch_slice[1:]] + taper[tile_slice]

            stitched = stitched / avg
            stitched = stitched.astype(dtype)

        else:

            stitch_indices_iterator = tiles.stitch_indices_iterator

            for index in nonempty_indices:

                i_image, j_image, i_tile, j_tile = stitch_indices_iterator[index]

                tile = tiles[index]
                tile_crop = tile[..., i_tile[0]:i_tile[1], j_tile[0]:j_tile[1]]
                stitched[..., i_image[0]:i_image[1], j_image[0]:j_image[1]] = tile_crop

        return stitched

    func_stitch = transform(func_stitch, vectorized=False, output_type='stitched_image')

    return func_stitch


def stitch_masks(iou_threshold=0.1):

    """ Generate Algorithm object for a mask stitching algorithm.

    Parameters
    ----------
        iou_threshold : float, optional, default 0.1
            IOU score threshold used for mask stitching at tile borders.

    Returns
    -------
        func_stitch : Algorithm
            Algorithm object with a mask stitching algorithm as the callable.
    """

    def func_stitch(masks):

        profile = masks.profile
        nonempty_indices = profile.nonempty_indices
        first_mask = masks[nonempty_indices[0]]

        image_shape = masks.image_shape
        tile_indices = masks.tile_indices
        border_indices = masks.border_indices
        tile_indices_iterator = masks.tile_indices_iterator
        stitch_indices_iterator = masks.stitch_indices_iterator
        mask_shape = (*first_mask.shape[:-2], *image_shape[-2:])
        mask_flat_shape = (np.prod(mask_shape[:-2], dtype=int), *mask_shape[-2:])
        stitched_mask = np.zeros(mask_flat_shape, dtype=int)

        for z in range(mask_flat_shape[0]):

            total_count = 0

            for index in nonempty_indices:

                i_image, j_image, i_tile, j_tile = stitch_indices_iterator[index]

                i_clear = i_tile[(0 < i_image) & (i_image < mask_shape[-2])]
                j_clear = j_tile[(0 < j_image) & (j_image < mask_shape[-1])]

                mask = masks[index]
                mask = mask.reshape(-1, *mask.shape[-2:])
                mask = mask[z].copy()
                mask = _clear_border(mask, i_clear, j_clear)

                mask_crop = mask[i_tile[0]:i_tile[1], j_tile[0]:j_tile[1]]
                mask_crop = measure.label(mask_crop)
                count = mask_crop.max()
                mask_crop[mask_crop > 0] += total_count
                total_count += count
                stitched_mask[z, i_image[0]:i_image[1], j_image[0]:j_image[1]] = mask_crop

            border_blobs = _find_border_blobs(masks, tile_indices, border_indices, z)

            for index, blobs in border_blobs.items():

                tile_index = tile_indices_iterator[index]

                mask = masks[index]
                mask = mask.reshape(-1, *mask.shape[-2:])
                mask = mask[z]
                regions = measure.regionprops(mask)

                for blob in blobs:

                    mask_crop = regions[blob - 1].image
                    s = regions[blob - 1].slice
                    s_image = (slice(s[0].start + tile_index[0, 0],
                                     s[0].stop + tile_index[0, 0]),
                               slice(s[1].start + tile_index[1, 0],
                                     s[1].stop + tile_index[1, 0]))
                    image_crop = stitched_mask[z][s_image]

                    if _calculate_iou_score(mask_crop, image_crop > 0) < iou_threshold:
                        image_crop[mask_crop] = total_count + 1
                        total_count += 1

            stitched_mask[z] = measure.label(stitched_mask[z])

        stitched_mask = stitched_mask.reshape(mask_shape)

        return stitched_mask

    func_stitch = transform(func_stitch, vectorized=False, output_type='stitched_image')

    return func_stitch


def stitch_coords():

    """ Generate Algorithm object for a coordinate stitching algorithm.

    Returns
    -------
        func_stitch : Algorithm
            Algorithm object with a coordinate stitching algorithm as the callable.
    """

    def func_stitch(coords):

        profile = coords.profile
        nonempty_indices = profile.nonempty_indices
        tile_indices_iterator = coords.tile_indices_iterator
        border_indices_iterator = coords.border_indices_iterator
        first_coord = coords[nonempty_indices[0]]
        n_batches = first_coord.shape[0]
        stitched_coords = np.empty(n_batches, dtype=object)

        for n in range(n_batches):

            batch_coords = []

            for index in nonempty_indices:

                tile_index = tile_indices_iterator[index]
                border_index = border_indices_iterator[index]

                coord = coords[index][n]
                coord = coord + np.array([tile_index[0, 0], tile_index[1, 0]])
                s = (border_index[0, 0] < coord[:, 0]) & (coord[:, 0] < border_index[0, 1]) & \
                    (border_index[1, 0] < coord[:, 1]) & (coord[:, 1] < border_index[1, 1])
                batch_coords.append(coord[s])

            stitched_coords[n] = np.concatenate(batch_coords, axis=0)

        return stitched_coords

    func_stitch = transform(func_stitch, vectorized=False, input_type='tiled_coords', output_type='stitched_coords')

    return func_stitch


def _generate_taper(tile_size, overlap, sigma):

    """ (For internal use) Generate taper used for blending tile overlaps.

    Parameters
    ----------
        tile_size : tuple
            Size of each tile.
        overlap : tuple
            Fractions of ``tile_size`` to use for overlap.
        sigma : int
            Sigma bandwidth parameter.

    Returns
    -------
        taper : numpy.array
            Taper used for blending tile overlaps.
    """

    x = np.arange(tile_size[1])
    x = np.abs(x - np.median(x))
    y = np.arange(tile_size[0])
    y = np.abs(y - np.median(y))
    taperx = 1 / (1 + np.exp((x - tile_size[1] * (1 - overlap[1]) / 2) / sigma))
    tapery = 1 / (1 + np.exp((y - tile_size[0] * (1 - overlap[0]) / 2) / sigma))
    taper = tapery[:, None] * taperx

    return taper


def _calculate_iou_score(a, b):

    """ (For internal use) Calculate IOU score.

    Parameters
    ----------
        a : numpy.array
            Boolean array containing object A.
        b : numpy.array
            Boolean array containing object B.

    Returns
    -------
        iou_score : float
            IOU score for objects A and B.
    """

    iou_score = np.sum(a & b) / np.sum(a | b)

    return iou_score


def _find_border_blobs(masks, tile_indices, border_indices, z):

    tile_indices_flat = (tile_indices[0].ravel(), tile_indices[1].ravel())
    border_blobs = dict()

    for axis in range(2):

        if axis == 1:
            masks = masks.T

        for i in range(masks.shape[0]):

            i_image = border_indices[axis][i:i + 2]
            i_tile = i_image - tile_indices[axis][i, 0]

            for j in range(masks.shape[1] - 1):

                j_image = np.flip(tile_indices_flat[1 - axis][2 * j + 1:2 * j + 3])
                offset = tile_indices[1 - axis][j:j + 2, 0]
                j_tile = j_image - offset.reshape(2, 1)
                border_index = border_indices[1 - axis][j + 1] - j_image[0]

                position_l, position_r = None, None

                mask_l_all = masks[i, j]
                if mask_l_all is not None:
                    mask_l_all = mask_l_all.reshape(-1, *mask_l_all.shape[-2:])
                    mask_l_all = mask_l_all[z]
                    if axis == 1:
                        mask_l_all = mask_l_all.T
                        position_l = (j, i)
                    elif axis == 0:
                        position_l = (i, j)
                    border_blobs = _scan_border(border_blobs, mask_l_all, (i_tile, j_tile[0]), position_l, border_index)

                mask_r_all = masks[i, j + 1]
                if mask_r_all is not None:
                    mask_r_all = mask_r_all.reshape(-1, *mask_r_all.shape[-2:])
                    mask_r_all = mask_r_all[z]
                    if axis == 1:
                        mask_r_all = mask_r_all.T
                        position_r = (j + 1, i)
                    elif axis == 0:
                        position_r = (i, j + 1)
                    border_blobs = _scan_border(border_blobs, mask_r_all, (i_tile, j_tile[1]), position_r, border_index)

    return border_blobs


def _scan_border(all_border_blobs, mask_all, position, position_adjacent, border_index):

    if mask_all is not None:

        i, j = position
        mask = mask_all[i[0]:i[1], j[0]:j[1]]
        border = mask[:, border_index]
        blobs = np.unique(border)
        blobs = blobs[blobs > 0]

        if len(blobs) > 0:
            border_blobs = all_border_blobs.get(position_adjacent, set())
            border_blobs.update(blobs)
            all_border_blobs[position_adjacent] = border_blobs

    return all_border_blobs


def _clear_border(mask, i, j):

    for row in i:
        mask = _remove_blob(mask, mask[row])

    for col in j:
        mask = _remove_blob(mask, mask[:, col])

    return mask


def _remove_blob(mask, blobs):

    blobs = np.unique(blobs)
    blobs = blobs[blobs > 0]
    mask[np.isin(mask, blobs)] = 0

    return mask
