from unittest.mock import PropertyMock, patch

import pytest
from PIL import Image

from histoslice import Slide, slice_slide
from histoslice.api import mean_and_std
from histoslice.export import ExportResult
from histoslice.tiles import tile_regions

from ._utils import SLIDE_PATH_JPEG, TMP_DIRECTORY, clean_temporary_directory


def test_slice_slide_native_resolution() -> None:
    clean_temporary_directory()
    result = slice_slide(
        SLIDE_PATH_JPEG, TMP_DIRECTORY, tile_size=512, max_background=0.5
    )
    assert isinstance(result, ExportResult)
    assert len(result.metadata) > 0
    for path in result.metadata["path"]:
        assert Image.open(path).size == (512, 512)
    clean_temporary_directory()


def test_slice_slide_target_mpp_output_dimensions() -> None:
    """Output tiles are exactly `tile_size` regardless of the level-0 crop size."""
    clean_temporary_directory()
    result = slice_slide(
        SLIDE_PATH_JPEG,
        TMP_DIRECTORY,
        tile_size=256,
        target_mpp=0.5,
        mpp=(0.25, 0.25),
        max_background=0.9,
        save_thumbnails=False,
    )
    assert len(result.metadata) > 0
    for path in result.metadata["path"]:
        assert Image.open(path).size == (256, 256)
    clean_temporary_directory()


def test_slice_slide_target_mpp_downscale() -> None:
    """target_mpp < slide mpp means a smaller level-0 crop, upsampled to tile_size."""
    clean_temporary_directory()
    result = slice_slide(
        SLIDE_PATH_JPEG,
        TMP_DIRECTORY,
        tile_size=256,
        target_mpp=0.1,
        mpp=(0.25, 0.25),
        max_background=0.9,
        save_thumbnails=False,
    )
    for path in result.metadata["path"]:
        assert Image.open(path).size == (256, 256)
    clean_temporary_directory()


def test_slice_slide_anisotropic_target_mpp() -> None:
    clean_temporary_directory()
    result = slice_slide(
        SLIDE_PATH_JPEG,
        TMP_DIRECTORY,
        tile_size=(256, 128),
        target_mpp=(0.5, 0.5),
        mpp=(0.25, 0.5),
        max_background=0.9,
        save_thumbnails=False,
    )
    for path in result.metadata["path"]:
        assert Image.open(path).size == (256, 128)
    clean_temporary_directory()


def test_slice_slide_target_mpp_without_mpp_raises() -> None:
    """slide.jpeg carries (non-biological) DPI metadata, so mpp is normally
    available; force it to None to exercise the "no mpp" error path."""
    clean_temporary_directory()
    with patch.object(Slide, "mpp", new_callable=PropertyMock, return_value=None):
        with pytest.raises(ValueError, match="target mpp was requested"):
            slice_slide(SLIDE_PATH_JPEG, TMP_DIRECTORY, target_mpp=0.5)


def test_slice_slide_creates_named_output_dir() -> None:
    clean_temporary_directory()
    result = slice_slide(SLIDE_PATH_JPEG, TMP_DIRECTORY, save_thumbnails=False)
    assert result.output_dir == TMP_DIRECTORY / Slide(SLIDE_PATH_JPEG).name
    clean_temporary_directory()


def test_mean_and_std() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    regions = tile_regions(slide.dimensions, 512, out_of_bounds=True)
    mean, std = mean_and_std(slide, regions)
    assert [round(x, 2) for x in mean] == [0.84, 0.7, 0.78]
    assert [round(x, 2) for x in std] == [0.14, 0.19, 0.14]


def test_mean_and_std_subsamples_when_over_max_samples() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    regions = tile_regions(slide.dimensions, 512, out_of_bounds=True)
    mean, std = mean_and_std(slide, regions, max_samples=2)
    assert len(mean) == 3
    assert len(std) == 3
