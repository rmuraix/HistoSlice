import histoslice.functional as F
from histoslice import Slide, export_tiles
from histoslice.tiles import tile_regions

from ._utils import SLIDE_PATH_JPEG, TMP_DIRECTORY, clean_temporary_directory


def test_mean_and_std_from_images() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    regions = tile_regions(slide.dimensions, 512, out_of_bounds=True)
    images = (slide.read_region(r) for r in regions)
    mean, std = F.get_mean_and_std_from_images(images)
    assert [round(x, 2) for x in mean] == [0.84, 0.7, 0.78]
    assert [round(x, 2) for x in std] == [0.14, 0.19, 0.14]


def test_mean_and_std_from_paths() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    regions = tile_regions(slide.dimensions, 512, out_of_bounds=True)
    clean_temporary_directory()
    export_tiles(
        slide, regions, TMP_DIRECTORY / slide.name, tile_size=512, save_thumbnails=False
    )
    paths = list((TMP_DIRECTORY / "slide" / "tiles").iterdir())
    mean, std = F.get_mean_and_std_from_paths(paths)
    clean_temporary_directory()
    assert [round(x, 2) for x in mean] == [0.84, 0.7, 0.78]
    assert [round(x, 2) for x in std] == [0.14, 0.19, 0.14]


def test_mean_and_std_from_paths_multiprocessing() -> None:
    """Test get_mean_and_std_from_paths with multiple workers."""
    slide = Slide(SLIDE_PATH_JPEG)
    regions = tile_regions(slide.dimensions, 512, out_of_bounds=True)
    clean_temporary_directory()
    export_tiles(
        slide, regions, TMP_DIRECTORY / slide.name, tile_size=512, save_thumbnails=False
    )
    paths = list((TMP_DIRECTORY / "slide" / "tiles").iterdir())
    mean, std = F.get_mean_and_std_from_paths(paths, num_workers=2)
    clean_temporary_directory()
    assert [round(x, 2) for x in mean] == [0.84, 0.7, 0.78]
    assert [round(x, 2) for x in std] == [0.14, 0.19, 0.14]


def test_mean_and_std_grayscale() -> None:
    """Test _get_mean_and_std with grayscale images."""
    import numpy as np
    from histoslice.functional._mean_std import _get_mean_and_std

    grayscale_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    mean, std = _get_mean_and_std(grayscale_image)
    assert len(mean) == 1
    assert len(std) == 1
    assert 0 <= mean[0] <= 1
    assert 0 <= std[0] <= 1
