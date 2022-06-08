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

    def func_stitch(dt, tiles):

        first_tile = tiles[list(dt.stitch_indices.keys())[0]]
        dtype = first_tile.dtype
        stitch_shape = (*first_tile.shape[:-2], *dt.image_shape[-2:])
        stitched = np.zeros(stitch_shape)

        if blend:

            avg = np.zeros(stitch_shape[-2:])
            taper = _generate_taper(dt.tile_size, dt.overlap, sigma)

            for n_i, n_j in dt.stitch_indices.keys():
                tile = tiles[n_i, n_j]
                stitch_slice = np.s_[..., dt.tile_indices[0][n_i, 0]:dt.tile_indices[0][n_i, 1],
                                     dt.tile_indices[1][n_j, 0]:dt.tile_indices[1][n_j, 1]]
                taper_slice = np.s_[:tile.shape[-2], :tile.shape[-1]]
                stitched[stitch_slice] = stitched[stitch_slice] + tile * taper[taper_slice]
                avg[stitch_slice[1:]] = avg[stitch_slice[1:]] + taper[taper_slice]

            stitched = stitched / avg
            stitched = stitched.astype(dtype)

        else:

            for (n_i, n_j), (i_image, j_image, i, j) in dt.stitch_indices.items():
                tile = tiles[n_i, n_j]
                tile_crop = tile[..., i[0]:i[1], j[0]:j[1]]
                stitched[..., i_image[0]:i_image[1], j_image[0]:j_image[1]] = tile_crop

        return stitched

    func_stitch = transform(func_stitch, vectorized=False)

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

    def func_stitch(dt, masks):

        first_mask = masks[list(dt.stitch_indices.keys())[0]]
        mask_shape = (*first_mask.shape[:-2], *dt.image_shape[-2:])
        mask_flat_shape = (np.prod(mask_shape[:-2], dtype=int), *mask_shape[-2:])
        stitched_mask = np.zeros(mask_flat_shape, dtype=int)

        for z in range(mask_flat_shape[0]):

            total_count = 0

            for (n_i, n_j), (i_image, j_image, i, j) in dt.stitch_indices.items():
                i_clear = i[(0 < i_image) & (i_image < mask_shape[-2])]
                j_clear = j[(0 < j_image) & (j_image < mask_shape[-1])]

                mask = masks[n_i, n_j]
                mask = mask.reshape(-1, *mask.shape[-2:])
                mask = mask[z].copy()
                mask = _clear_border(mask, i_clear, j_clear)

                mask_crop = mask[i[0]:i[1], j[0]:j[1]]
                mask_crop = measure.label(mask_crop)
                count = mask_crop.max()
                mask_crop[mask_crop > 0] += total_count
                total_count += count
                stitched_mask[z, i_image[0]:i_image[1], j_image[0]:j_image[1]] = mask_crop

            border_blobs = _find_border_blobs(masks, dt.tile_indices, dt.border_indices, z)

            for (n_i, n_j), blobs in border_blobs.items():

                mask = masks[n_i, n_j]
                mask = mask.reshape(-1, *mask.shape[-2:])
                mask = mask[z]
                regions = measure.regionprops(mask)

                for blob in blobs:

                    mask_crop = regions[blob - 1].image
                    s = regions[blob - 1].slice
                    s_image = (slice(s[0].start + dt.tile_indices[0][n_i, 0],
                                     s[0].stop + dt.tile_indices[0][n_i, 0]),
                               slice(s[1].start + dt.tile_indices[1][n_j, 0],
                                     s[1].stop + dt.tile_indices[1][n_j, 0]))
                    image_crop = stitched_mask[z][s_image]

                    if _calculate_iou_score(mask_crop, image_crop > 0) < iou_threshold:
                        image_crop[mask_crop] = total_count + 1
                        total_count += 1

            stitched_mask[z] = measure.label(stitched_mask[z])

        stitched_mask = stitched_mask.reshape(mask_shape)

        return stitched_mask

    func_stitch = transform(func_stitch, vectorized=False)

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

        for n_i in range(masks.shape[0]):

            i_image = border_indices[axis][n_i:n_i + 2]
            i = i_image - tile_indices[axis][n_i, 0]

            for n_j in range(masks.shape[1] - 1):

                j_image = np.flip(tile_indices_flat[1 - axis][2 * n_j + 1:2 * n_j + 3])
                offset = tile_indices[1 - axis][n_j:n_j + 2, 0]
                j = j_image - offset.reshape(2, 1)
                border_index = border_indices[1 - axis][n_j + 1] - j_image[0]

                position_l, position_r = None, None

                mask_l_all = masks[n_i, n_j]
                if mask_l_all is not None:
                    mask_l_all = mask_l_all.reshape(-1, *mask_l_all.shape[-2:])
                    mask_l_all = mask_l_all[z]
                    if axis == 1:
                        mask_l_all = mask_l_all.T
                        position_l = (n_j, n_i)
                    elif axis == 0:
                        position_l = (n_i, n_j)
                    border_blobs = _scan_border(border_blobs, mask_l_all, (i, j[0]), position_l, border_index)

                mask_r_all = masks[n_i, n_j + 1]
                if mask_r_all is not None:
                    mask_r_all = mask_r_all.reshape(-1, *mask_r_all.shape[-2:])
                    mask_r_all = mask_r_all[z]
                    if axis == 1:
                        mask_r_all = mask_r_all.T
                        position_r = (n_j + 1, n_i)
                    elif axis == 0:
                        position_r = (n_i, n_j + 1)
                    border_blobs = _scan_border(border_blobs, mask_r_all, (i, j[1]), position_r, border_index)

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
