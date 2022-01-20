import numpy as np
from deeptile import utils
from skimage import measure


class DeepTile:

    def __init__(self, image):

        self.image = image
        self.n_blocks = None
        self.overlap = None
        self.algorithm = None
        self.parameters = None
        self.app = None
        self.tile_indices = None
        self.stitch_indices = None

    def create_job(self, n_blocks=(2, 2), overlap=(0.1, 0.1), algorithm='cellori', parameters=None):

        self.n_blocks = n_blocks
        self.overlap = overlap
        self.algorithm = algorithm

        if parameters is None:
            self.parameters = dict()

        if self.algorithm == 'cellori':
            from cellori import Cellori
            self.app = Cellori

    def run_job(self):

        self.tile_indices, self.stitch_indices = utils.calculate_indices_2d(self.image, self.n_blocks, self.overlap)

        masks = self.segment_image()
        mask = self._stitch_masks(masks)
        mask = measure.label(mask)

        return mask

    def get_tiles(self):

        tiles = np.empty(shape=self.n_blocks, dtype=object)
        tiles[:] = utils.array_split_2d(self.image, self.tile_indices)

        return tiles

    def segment_image(self):

        tiles = self.get_tiles()
        masks = np.zeros_like(tiles)

        for index, tile in np.ndenumerate(tiles):

            mask = self._segment_tile(tile)[0]
            masks[index] = mask

        return masks

    def _segment_tile(self, tile):

        if self.app.__name__ == 'Cellori':
            return self.app(tile).segment(**self.parameters)

    def _find_border_cells(self, masks):

        tile_indices_flat = (self.tile_indices[0].flatten(), self.tile_indices[1].flatten())
        border_cells = dict()

        for axis in range(2):

            if axis == 1:
                masks = masks.T

            for n_i in range(self.n_blocks[axis]):

                i_image = self.stitch_indices[axis][n_i:n_i + 2]
                i = i_image - self.tile_indices[axis][n_i, 0]

                for n_j in range(self.n_blocks[1 - axis] - 1):

                    j_image = np.flip(tile_indices_flat[1 - axis][2 * n_j + 1:2 * n_j + 3])
                    offset = self.tile_indices[1 - axis][n_j:n_j + 2, 0]
                    j = j_image - offset.reshape(2, 1)
                    stitch_index = self.stitch_indices[1 - axis][n_j + 1] - j_image[0]

                    mask_l_all = masks[n_i, n_j]
                    mask_r_all = masks[n_i, n_j + 1]
                    position_l, position_r = None, None

                    if axis == 1:
                        mask_l_all = mask_l_all.T
                        mask_r_all = mask_r_all.T
                        position_l, position_r = (n_j, n_i), (n_j + 1, n_i)
                    elif axis == 0:
                        position_l, position_r = (n_i, n_j), (n_i, n_j + 1)

                    mask_l = mask_l_all[i[0]:i[1], j[0, 0]:j[0, 1]]
                    mask_r = mask_r_all[i[0]:i[1], j[1, 0]:j[1, 1]]
                    border_l = mask_l[:, stitch_index]
                    border_r = mask_r[:, stitch_index]
                    cells_l, cells_r = np.unique(border_l), np.unique(border_r)
                    cells_l, cells_r = cells_l[cells_l > 0], cells_r[cells_r > 0]

                    if len(cells_l) > 0:
                        border_cells_l = border_cells.get(position_l, set())
                        border_cells_l.update(cells_l)
                        border_cells[position_l] = border_cells_l

                    if len(cells_r) > 0:
                        border_cells_r = border_cells.get(position_r, set())
                        border_cells_r.update(cells_r)
                        border_cells[position_r] = border_cells_r

        return border_cells

    def _stitch_masks(self, masks):

        stitched_mask = np.zeros_like(self.image, dtype=int)

        total_count = 0

        for (n_i, n_j), mask in np.ndenumerate(masks):

            i_image = self.stitch_indices[0][n_i:n_i + 2] + np.array([0, 1])
            j_image = self.stitch_indices[1][n_j:n_j + 2] + np.array([0, 1])
            i = i_image - self.tile_indices[0][n_i, 0]
            j = j_image - self.tile_indices[1][n_j, 0]

            i_clear = (i - np.array([0, 1]))[(0 < i_image) & (i_image < self.image.shape[-2])]
            j_clear = (j - np.array([0, 1]))[(0 < j_image) & (j_image < self.image.shape[-1])]
            mask = utils.clear_border(mask.copy(), i_clear, j_clear)

            mask_crop = mask[i[0]:i[1], j[0]:j[1]]
            mask_crop = measure.label(mask_crop)
            count = mask_crop.max()
            mask_crop[mask_crop > 0] += total_count
            total_count += count
            stitched_mask[i_image[0]:i_image[1], j_image[0]:j_image[1]] += mask_crop

        border_cells = self._find_border_cells(masks)

        for (n_i, n_j), cells in border_cells.items():

            mask = masks[n_i, n_j]
            regions = measure.regionprops(mask)

            for cell in cells:

                mask_crop = regions[cell - 1].image
                s = regions[cell - 1].slice
                s_image = (slice(s[0].start + self.tile_indices[0][n_i, 0], s[0].stop + self.tile_indices[0][n_i, 0]),
                           slice(s[1].start + self.tile_indices[1][n_j, 0], s[1].stop + self.tile_indices[1][n_j, 0]))
                image_crop = stitched_mask[s_image]

                if not np.any(mask_crop & (image_crop > 0)):
                    image_crop[mask_crop] = total_count + 1
                    total_count += 1

        return stitched_mask
