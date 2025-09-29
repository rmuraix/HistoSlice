import numpy as np
import pytest

import histoslice.functional as F
from histoslice import SlideReader
from histoslice.functional._tissue import _downscale_for_threshold
from tests._utils import IMAGE, SLIDE_PATH_TMA


def test_tissue_mask_otsu() -> None:
    thresh, mask = F.get_tissue_mask(IMAGE)
    assert mask.shape == IMAGE.shape[:2]
    assert thresh == 200
    assert mask.sum() == 184158


def test_tissue_mask_otsu_multiplier() -> None:
    thresh, mask = F.get_tissue_mask(IMAGE, multiplier=1.05)
    assert thresh == 210
    assert mask.sum() == 192803


def test_tissue_mask_threshold() -> None:
    thresh, mask = F.get_tissue_mask(IMAGE, threshold=210)
    assert thresh == 210
    assert mask.sum() == 192803
    # Boundary condition: THRESH_BINARY_INV semantics are inclusive (<= threshold)
    boundary = np.array([[209, 210, 211]], dtype=np.uint8)
    __, bmask = F.get_tissue_mask(boundary, threshold=210, sigma=0.0)
    assert bmask.tolist() == [[1, 1, 0]]


def test_tissue_mask_bad_threshold() -> None:
    with pytest.raises(ValueError, match="Threshold should be in range"):
        F.get_tissue_mask(IMAGE, threshold=500)


def test_clean_tissue_mask() -> None:
    image = SlideReader(SLIDE_PATH_TMA).read_level(-1)
    __, tissue_mask = F.get_tissue_mask(image, sigma=0.0)
    # We fill the areas.
    assert F.clean_tissue_mask(tissue_mask).sum() > tissue_mask.sum()


def test_clean_empty_mask() -> None:
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    assert F.clean_tissue_mask(empty_mask).sum() == 0


def test_tissue_mask_edge_cases() -> None:
    """Test edge cases that could cause OpenCV threshold errors."""
    # Test with all black and white pixels (empty array after filtering in original image)
    # Note: Gaussian blur will create intermediate values from pure black/white
    black_white_image = np.array([[0, 255], [0, 255]], dtype=np.uint8)
    thresh, mask = F.get_tissue_mask(black_white_image)
    assert thresh >= 0  # Should not error and return valid threshold
    assert mask.shape == black_white_image.shape

    # Test with single pixel
    single_pixel_image = np.array([[128]], dtype=np.uint8)
    thresh, mask = F.get_tissue_mask(single_pixel_image)
    assert thresh >= 0  # Should not error
    assert mask.shape == single_pixel_image.shape

    # Test with all same gray value
    uniform_gray_image = np.array([[100, 100], [100, 100]], dtype=np.uint8)
    thresh, mask = F.get_tissue_mask(uniform_gray_image)
    assert thresh >= 0  # Should not error
    assert mask.shape == uniform_gray_image.shape

    # Test edge case without blur to ensure _otsu_threshold handles empty arrays
    black_white_image_no_blur = np.array([[0, 255], [0, 255]], dtype=np.uint8)
    thresh_no_blur, mask_no_blur = F.get_tissue_mask(
        black_white_image_no_blur, sigma=0.0
    )
    assert thresh_no_blur == 127  # Should return default fallback threshold
    assert mask_no_blur.shape == black_white_image_no_blur.shape


def test_downscale_for_threshold_small_noop() -> None:
    img = (np.random.rand(100, 100) * 255).astype(np.uint8)
    out = _downscale_for_threshold(img, max_pixels=4_000_000)
    # Should return the original array (no resize)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out, img)


def test_downscale_for_threshold_large_scales_down() -> None:
    h, w = 3000, 3000  # 9,000,000 px > 4,000,000
    img = (np.random.rand(h, w) * 255).astype(np.uint8)
    out = _downscale_for_threshold(img, max_pixels=4_000_000)
    # Expected uniform scale factor
    scale = np.sqrt(4_000_000 / float(h * w))
    exp_w = max(1, int(w * scale))
    exp_h = max(1, int(h * scale))
    assert out.shape == (exp_h, exp_w)
    assert out.size <= 4_000_000
    assert out.dtype == np.uint8


def test_downscale_for_threshold_dtype_cast_and_empty() -> None:
    # Non-uint8 image is cast before resize
    img_f = np.linspace(0, 1, 200 * 200, dtype=np.float32).reshape(200, 200)
    out_f = _downscale_for_threshold(img_f, max_pixels=10_000)
    # 200*200 = 40,000 > 10,000 -> expect downscale to ~100x100 (depending on rounding)
    assert out_f.size <= 10_000
    assert out_f.dtype == np.uint8

    # Empty input stays empty
    empty = np.zeros((0, 0), dtype=np.uint8)
    out_e = _downscale_for_threshold(empty, max_pixels=1)
    assert out_e.shape == (0, 0)
