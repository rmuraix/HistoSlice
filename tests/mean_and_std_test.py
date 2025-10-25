import pytest

import histoslice.functional as F
from histoslice import SlideReader

from ._utils import SLIDE_PATH_JPEG, TMP_DIRECTORY, clean_temporary_directory


def test_mean_and_std_from_images() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(None, 512)
    images = (tile for tile, __ in reader.yield_regions(tile_coords))
    mean, std = F.get_mean_and_std_from_images(images)
    assert [round(x, 2) for x in mean] == [0.84, 0.70, 0.78]
    assert [round(x, 2) for x in std] == [0.14, 0.19, 0.14]


def test_mean_and_std_from_paths() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(None, 512)
    clean_temporary_directory()
    reader.save_regions(TMP_DIRECTORY, tile_coords)
    paths = list((TMP_DIRECTORY / "slide" / "tiles").iterdir())
    mean, std = F.get_mean_and_std_from_paths(paths)
    clean_temporary_directory()
    assert [round(x, 2) for x in mean] == [0.84, 0.70, 0.78]
    assert [round(x, 2) for x in std] == [0.14, 0.19, 0.14]


@pytest.mark.skip(reason="PyVips backend has issues with multiprocessing in CI")
def test_mean_and_std_from_paths_multiprocessing() -> None:
    """Test get_mean_and_std_from_paths with multiple workers."""
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(None, 512)
    clean_temporary_directory()
    reader.save_regions(TMP_DIRECTORY, tile_coords)
    paths = list((TMP_DIRECTORY / "slide" / "tiles").iterdir())
    mean, std = F.get_mean_and_std_from_paths(paths, num_workers=2)
    clean_temporary_directory()
    assert [round(x, 2) for x in mean] == [0.84, 0.70, 0.78]
    assert [round(x, 2) for x in std] == [0.14, 0.19, 0.14]


def test_mean_and_std_grayscale() -> None:
    """Test _get_mean_and_std with grayscale images."""
    import numpy as np
    from histoslice.functional._mean_std import _get_mean_and_std

    # Create a grayscale image
    grayscale_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    mean, std = _get_mean_and_std(grayscale_image)

    # Should return a single value for grayscale
    assert len(mean) == 1
    assert len(std) == 1
    assert 0 <= mean[0] <= 1
    assert 0 <= std[0] <= 1
