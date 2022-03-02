import numpy as np
from skimage import measure


def stitch_tiles():

    def func_stitch(tiles, image_shape, tile_indices, border_indices, stitch_indices):

        dtype = tiles[list(stitch_indices.keys())[0]].dtype
        stitch = np.zeros(image_shape, dtype=dtype)

        for (n_i, n_j), (i_image, j_image, i, j) in stitch_indices.items():
            tile = tiles[n_i, n_j]
            tile_crop = tile[..., i[0]:i[1], j[0]:j[1]]
            stitch[..., i_image[0]:i_image[1], j_image[0]:j_image[1]] = tile_crop

        return stitch

    return func_stitch


def stitch_masks():

    def func_stitch(masks, image_shape, tile_indices, border_indices, stitch_indices):

        mask_dims = None

        for mask in masks:
            if mask is None:
                continue
            else:
                mask_dims = mask.shape[:-2]
                break

        mask_shape = (*mask_dims, *image_shape[-2:])
        mask_flat_shape = (np.prod(mask_dims, dtype=int), *image_shape[-2:])
        stitched_mask = np.zeros(mask_flat_shape, dtype=int)

        for z in range(mask_flat_shape[0]):

            total_count = 0

            for (n_i, n_j), (i_image, j_image, i, j) in stitch_indices.items():
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

            border_objects = _find_border_objects(masks, tile_indices, border_indices, z)

            for (n_i, n_j), objects in border_objects.items():

                mask = masks[n_i, n_j]
                mask = mask.reshape(-1, *mask.shape[-2:])
                mask = mask[z]
                regions = measure.regionprops(mask)

                for object in objects:

                    mask_crop = regions[object - 1].image
                    s = regions[object - 1].slice
                    s_image = (slice(s[0].start + tile_indices[0][n_i, 0],
                                     s[0].stop + tile_indices[0][n_i, 0]),
                               slice(s[1].start + tile_indices[1][n_j, 0],
                                     s[1].stop + tile_indices[1][n_j, 0]))
                    image_crop = stitched_mask[z][s_image]

                    if not np.any(mask_crop & (image_crop > 0)):
                        image_crop[mask_crop] = total_count + 1
                        total_count += 1

            stitched_mask[z] = measure.label(stitched_mask[z])

        stitched_mask = stitched_mask.reshape(mask_shape)

        return stitched_mask

    return func_stitch


def _find_border_objects(masks, tile_indices, border_indices, z):

    tile_indices_flat = (tile_indices[0].ravel(), tile_indices[1].ravel())
    border_objects = dict()

    for axis in range(2):

        if axis == 1:
            masks = masks.T

        for n_i in range(masks.shape[axis]):

            i_image = border_indices[axis][n_i:n_i + 2]
            i = i_image - tile_indices[axis][n_i, 0]

            for n_j in range(masks.shape[1 - axis] - 1):

                j_image = np.flip(tile_indices_flat[1 - axis][2 * n_j + 1:2 * n_j + 3])
                offset = tile_indices[1 - axis][n_j:n_j + 2, 0]
                j = j_image - offset.reshape(2, 1)
                border_index = border_indices[1 - axis][n_j + 1] - j_image[0]

                position_l, position_r = None, None

                mask_l_all = masks[n_i, n_j]
                mask_l_all = mask_l_all.reshape(-1, *mask_l_all.shape[-2:])
                if mask_l_all is not None:
                    mask_l_all = mask_l_all[z]
                    if axis == 1:
                        mask_l_all = mask_l_all.T
                        position_l = (n_j, n_i)
                    elif axis == 0:
                        position_l = (n_i, n_j)
                    border_objects = _scan_border(border_objects, mask_l_all, (i, j[0]), position_l, border_index)

                mask_r_all = masks[n_i, n_j + 1]
                mask_r_all = mask_r_all.reshape(-1, *mask_r_all.shape[-2:])
                if mask_r_all is not None:
                    mask_r_all = mask_r_all[z]
                    if axis == 1:
                        mask_r_all = mask_r_all.T
                        position_r = (n_j + 1, n_i)
                    elif axis == 0:
                        position_r = (n_i, n_j + 1)
                    border_objects = _scan_border(border_objects, mask_r_all, (i, j[1]), position_r, border_index)

    return border_objects


def _scan_border(all_border_objects, mask_all, position, position_adjacent, border_index):

    if mask_all is not None:

        i, j = position
        mask = mask_all[i[0]:i[1], j[0]:j[1]]
        border = mask[:, border_index]
        objects = np.unique(border)
        objects = objects[objects > 0]

        if len(objects) > 0:
            border_objects = all_border_objects.get(position_adjacent, set())
            border_objects.update(objects)
            all_border_objects[position_adjacent] = border_objects

    return all_border_objects


def _clear_border(mask, i, j):

    for row in i:
        mask = _remove_object(mask, mask[row])

    for col in j:
        mask = _remove_object(mask, mask[:, col])

    return mask


def _remove_object(mask, objects):

    objects = np.unique(objects)
    objects = objects[objects > 0]
    mask[np.isin(mask, objects)] = 0

    return mask
