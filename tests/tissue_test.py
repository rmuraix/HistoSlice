import numpy as np
import pytest

from histoslice import Slide
from histoslice.tissue import clean_tissue_mask, tissue_mask

from ._utils import IMAGE, SLIDE_PATH_TMA


def test_tissue_mask_otsu_default() -> None:
    """Default multiplier is 1.05 (matches the old `SlideReader` default)."""
    thresh, mask = tissue_mask(IMAGE)
    assert mask.shape == IMAGE.shape[:2]
    assert thresh == 210
    assert mask.sum() == 192803


def test_tissue_mask_otsu_multiplier() -> None:
    thresh, mask = tissue_mask(IMAGE, multiplier=1.0)
    assert thresh == 200
    assert mask.sum() == 184158


def test_tissue_mask_threshold() -> None:
    thresh, mask = tissue_mask(IMAGE, threshold=210)
    assert thresh == 210
    assert mask.sum() == 192803
    # Boundary condition: THRESH_BINARY_INV semantics are inclusive (<= threshold)
    boundary = np.array([[209, 210, 211]], dtype=np.uint8)
    __, bmask = tissue_mask(boundary, threshold=210, sigma=0.0)
    assert bmask.tolist() == [[1, 1, 0]]


def test_tissue_mask_bad_threshold() -> None:
    with pytest.raises(ValueError, match="Threshold should be in range"):
        tissue_mask(IMAGE, threshold=500)


def test_clean_tissue_mask() -> None:
    image = Slide(SLIDE_PATH_TMA).read_level(-1)
    __, mask = tissue_mask(image, sigma=0.0)
    assert clean_tissue_mask(mask).sum() > mask.sum()


def test_clean_empty_mask() -> None:
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    assert clean_tissue_mask(empty_mask).sum() == 0


def test_clean_tissue_mask_max_area_pixel() -> None:
    image = Slide(SLIDE_PATH_TMA).read_level(-1)
    __, mask = tissue_mask(image, sigma=0.0)
    # A max_area_pixel smaller than every contour drops everything.
    assert clean_tissue_mask(mask, max_area_pixel=1).sum() == 0


def test_clean_tissue_mask_all_contours_below_min_area() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5, 5] = 1  # a single-pixel contour
    assert clean_tissue_mask(mask, min_area_pixel=1000).sum() == 0


def test_tissue_mask_empty_image() -> None:
    thresh, mask = tissue_mask(np.zeros((0, 0), dtype=np.uint8))
    assert mask.shape == (0, 0)


def test_tissue_mask_edge_cases() -> None:
    """Edge cases that could otherwise cause OpenCV threshold errors."""
    black_white_image = np.array([[0, 255], [0, 255]], dtype=np.uint8)
    thresh, mask = tissue_mask(black_white_image)
    assert thresh >= 0
    assert mask.shape == black_white_image.shape

    single_pixel_image = np.array([[128]], dtype=np.uint8)
    thresh, mask = tissue_mask(single_pixel_image)
    assert thresh >= 0
    assert mask.shape == single_pixel_image.shape

    uniform_gray_image = np.array([[100, 100], [100, 100]], dtype=np.uint8)
    thresh, mask = tissue_mask(uniform_gray_image)
    assert thresh >= 0
    assert mask.shape == uniform_gray_image.shape

    # Without blurring, `_otsu_threshold` should fall back to a default value (127)
    # when every pixel is pure black/white (nothing left after filtering for Otsu).
    # multiplier=1.0 isolates that fallback constant from the default multiplier.
    thresh_no_blur, mask_no_blur = tissue_mask(
        black_white_image, sigma=0.0, multiplier=1.0
    )
    assert thresh_no_blur == 127
    assert mask_no_blur.shape == black_white_image.shape
