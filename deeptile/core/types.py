ALLOWED_TILED_TYPES = ('tiled_image', 'tiled_coords')
ALLOWED_STITCHED_TYPES = ('stitched_image', 'stitched_coords')
ALLOWED_ITERATOR_TYPES = ('index_iterator', 'tile_indices_iterator', 'border_indices_iterator',
                          'stitch_indices_iterator')
ALLOWED_INPUT_TYPES = ALLOWED_TILED_TYPES + ALLOWED_ITERATOR_TYPES
