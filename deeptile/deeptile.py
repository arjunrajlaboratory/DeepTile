import numpy as np
from deeptile import utils
from pathlib import Path
from skimage import measure


class DeepTile:

    def __init__(self, image, nd2_overlap=(0.1, 0.1)):

        if isinstance(image, np.ndarray):
            self.image = image
            self.image_type = 'array'
        elif Path(image).is_file():
            if image.endswith('.nd2'):
                self.image, self.nd2_metadata, self.axes = utils.read_nd2(image)
                self.image_type = 'nd2'
                self.nd2_overlap = nd2_overlap
            elif image.endswith(('.tif', '.tiff')):
                from tifffile import imread
                self.image = imread(image)
                self.image_type = 'array'
        else:
            raise ValueError("Invalid image.")

        if isinstance(self.image, np.ndarray):
            self.shape = self.image.shape
        else:
            self.shape = None

        self.n_blocks = None
        self.overlap = None
        self.algorithm = None
        self.parameters = None
        self.app = None
        self.tile_indices = None
        self.border_indices = None
        self.stitch_indices = None

    def create_job(self, n_blocks=(2, 2), overlap=(0.1, 0.1), algorithm='cellori', parameters=None):

        self.n_blocks = n_blocks
        self.overlap = overlap
        self.algorithm = algorithm

        if parameters is None:
            self.parameters = dict()
        else:
            self.parameters = parameters

        if self.algorithm == 'cellori':
            from cellori import Cellori
            self.app = Cellori

        if self.algorithm in ['cellpose_cytoplasm', 'cellpose_nuclear']:
            from cellpose.models import Cellpose
            from cellpose.io import logger_setup
            logger_setup()
            self.app = Cellpose

        if self.algorithm in ['deepcell_mesmer', 'deepcell_cytoplasm', 'deepcell_nuclear']:
            from deepcell import applications
            if self.algorithm == 'deepcell_mesmer':
                self.app = applications.Mesmer
            if self.algorithm == 'deepcell_cytoplasm':
                self.app = applications.CytoplasmSegmentation
            if self.algorithm == 'deepcell_nuclear':
                self.app = applications.NuclearSegmentation

    def run_job(self, nd2_indices=None, stitch_nd2=True):

        masks, tiles = self.segment_image(nd2_indices)

        self._calculate_stitch_indices(masks)
        mask = self._stitch_masks(masks)

        if (self.image_type == 'nd2') & stitch_nd2:
            image = self._stitch_nd2(tiles)
            return image
        else:
            return mask

    def get_tiles(self):

        tiles = np.empty(shape=self.n_blocks, dtype=object)
        tiles[:] = utils.array_split_2d(self.image, self.tile_indices)

        return tiles

    def segment_image(self, nd2_indices=None):

        tiles = None
        if self.image_type == 'array':
            self.tile_indices, self.border_indices = \
                utils.calculate_indices_2d(self.shape, self.n_blocks, self.overlap)
            tiles = self.get_tiles()
        elif self.image_type == 'nd2':
            tiles, self.shape = utils.parse_nd2(self.image, self.nd2_metadata, self.axes, self.nd2_overlap, nd2_indices)
            self.tile_indices, self.border_indices = \
                utils.calculate_indices_2d(self.shape, tiles.shape, self.nd2_overlap)

        masks = np.zeros_like(tiles)

        for index, tile in np.ndenumerate(tiles):

            if tile is None:
                mask = None
            else:
                mask = self._segment_tile(tile)

            masks[index] = mask

        return masks, tiles

    def _segment_tile(self, tile):

        if self.app.__name__ == 'Cellori':
            mask_list = list()
            for tile_channel in tile:
                mask_list.append(self.app(tile_channel).segment(**self.parameters)[0])
            mask = np.stack(mask_list)
            return mask

        if self.app.__name__ == 'Cellpose':
            model_type = None
            if self.algorithm == 'cellpose_cytoplasm':
                model_type = 'cyto'
            elif self.algorithm == 'cellpose_nuclear':
                model_type = 'nuclei'
            app = self.app(model_type=model_type)
            mask_list = list()
            for tile_channel in tile:
                mask_list.append(app.eval(tile_channel, channels=[1, 1], diameter=None, tile=False,
                                          **self.parameters)[0])
            mask = np.stack(mask_list)
            return mask

        if self.app.__name__ in ['Mesmer']:
            app = self.app()
            tile = np.moveaxis(tile, 0, -1)
            tile = np.expand_dims(tile, axis=0)
            mask = app.predict(tile, **self.parameters)[0]
            mask = np.moveaxis(mask, -1, 0)
            return mask

        if self.app.__name__ in ['NuclearSegmentation', 'CytoplasmSegmentation']:
            app = self.app()
            mask_list = list()
            for tile_channel in tile:
                tile_channel = np.expand_dims(tile_channel, axis=-1)
                tile_channel = np.expand_dims(tile_channel, axis=0)
                mask_list.append(app.predict(tile_channel, **self.parameters)[0, :, :, 0])
            mask = np.stack(mask_list)
            return mask

    def _stitch_masks(self, masks):

        stitched_mask = np.zeros(self.shape, dtype=int)

        for channel in range(self.shape[0]):

            total_count = 0

            for (n_i, n_j), (i_image, j_image, i, j) in self.stitch_indices.items():
                i_clear = i[(0 < i_image) & (i_image < self.shape[-2])]
                j_clear = j[(0 < j_image) & (j_image < self.shape[-1])]
                mask = utils.clear_border(masks[n_i, n_j][channel].copy(), i_clear, j_clear)

                mask_crop = mask[i[0]:i[1], j[0]:j[1]]
                mask_crop = measure.label(mask_crop)
                count = mask_crop.max()
                mask_crop[mask_crop > 0] += total_count
                total_count += count
                stitched_mask[channel, i_image[0]:i_image[1], j_image[0]:j_image[1]] += mask_crop

            border_cells = self._find_border_cells(masks, channel)

            for (n_i, n_j), cells in border_cells.items():

                mask = masks[n_i, n_j][channel]
                regions = measure.regionprops(mask)

                for cell in cells:

                    mask_crop = regions[cell - 1].image
                    s = regions[cell - 1].slice
                    s_image = (slice(s[0].start + self.tile_indices[0][n_i, 0],
                                     s[0].stop + self.tile_indices[0][n_i, 0]),
                               slice(s[1].start + self.tile_indices[1][n_j, 0],
                                     s[1].stop + self.tile_indices[1][n_j, 0]))
                    image_crop = stitched_mask[channel][s_image]

                    if not np.any(mask_crop & (image_crop > 0)):
                        image_crop[mask_crop] = total_count + 1
                        total_count += 1

            stitched_mask[channel] = measure.label(stitched_mask[channel])

        return stitched_mask

    def _stitch_nd2(self, tiles):

        stitched_image = np.zeros(self.shape, dtype=int)

        for (n_i, n_j), (i_image, j_image, i, j) in self.stitch_indices.items():
            tile = tiles[n_i, n_j]
            tile_crop = tile[:, i[0]:i[1], j[0]:j[1]]
            stitched_image[:, i_image[0]:i_image[1], j_image[0]:j_image[1]] += tile_crop

        return stitched_image

    def _find_border_cells(self, masks, channel):

        tile_indices_flat = (self.tile_indices[0].flatten(), self.tile_indices[1].flatten())
        border_cells = dict()

        for axis in range(2):

            if axis == 1:
                masks = masks.T

            for n_i in range(self.n_blocks[axis]):

                i_image = self.border_indices[axis][n_i:n_i + 2]
                i = i_image - self.tile_indices[axis][n_i, 0]

                for n_j in range(self.n_blocks[1 - axis] - 1):

                    j_image = np.flip(tile_indices_flat[1 - axis][2 * n_j + 1:2 * n_j + 3])
                    offset = self.tile_indices[1 - axis][n_j:n_j + 2, 0]
                    j = j_image - offset.reshape(2, 1)
                    border_index = self.border_indices[1 - axis][n_j + 1] - j_image[0]

                    mask_l_all = masks[n_i, n_j][channel]
                    mask_r_all = masks[n_i, n_j + 1][channel]
                    position_l, position_r = None, None
                    if axis == 1:
                        mask_l_all = mask_l_all.T
                        mask_r_all = mask_r_all.T
                        position_l = (n_j, n_i)
                        position_r = (n_j + 1, n_i)
                    elif axis == 0:
                        position_l = (n_i, n_j)
                        position_r = (n_i, n_j + 1)

                    border_cells = self._scan_border(border_cells, mask_l_all, (i, j[0]), position_l, border_index)
                    border_cells = self._scan_border(border_cells, mask_r_all, (i, j[1]), position_r, border_index)

        return border_cells

    def _calculate_stitch_indices(self, masks):

        self.stitch_indices = dict()

        for (n_i, n_j), mask in np.ndenumerate(masks):

            if mask is not None:
                i_image = self.border_indices[0][n_i:n_i + 2]
                j_image = self.border_indices[1][n_j:n_j + 2]
                i = i_image - self.tile_indices[0][n_i, 0]
                j = j_image - self.tile_indices[1][n_j, 0]
                self.stitch_indices[(n_i, n_j)] = (i_image, j_image, i, j)

    @staticmethod
    def _scan_border(all_border_cells, mask_all, position, position_adjacent, border_index):

        if mask_all is not None:

            i, j = position
            mask = mask_all[i[0]:i[1], j[0]:j[1]]
            border = mask[:, border_index]
            cells = np.unique(border)
            cells = cells[cells > 0]

            if len(cells) > 0:
                border_cells = all_border_cells.get(position_adjacent, set())
                border_cells.update(cells)
                all_border_cells[position_adjacent] = border_cells

        return all_border_cells
