from PIL import Image

import histoslice.functional as F
from histoslice import Slide, export_tiles
from histoslice.tiles import tile_regions

from ._utils import SLIDE_PATH_JPEG, TMP_DIRECTORY, clean_temporary_directory


def test_image_collage() -> None:
    clean_temporary_directory()
    slide = Slide(SLIDE_PATH_JPEG)
    regions = tile_regions(slide.dimensions, 128, out_of_bounds=False)
    result = export_tiles(
        slide, regions, TMP_DIRECTORY / slide.name, tile_size=128, save_thumbnails=False
    )
    collage = F.get_random_image_collage(
        result.metadata["path"], num_rows=4, num_cols=8, shape=(32, 32)
    )
    assert isinstance(collage, Image.Image)
    assert collage.size == (8 * 32, 4 * 32)
    # Not enough images for all rows.
    collage = F.get_random_image_collage(
        result.metadata["path"][:6], num_rows=4, num_cols=8, shape=(32, 32)
    )
    assert isinstance(collage, Image.Image)
    assert collage.size == (8 * 32, 1 * 32)
    clean_temporary_directory()
