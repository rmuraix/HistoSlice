import numpy as np
import pytest

import histoslice.functional as F
from histoslice import SlideReader
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
    thresh_no_blur, mask_no_blur = F.get_tissue_mask(black_white_image_no_blur, sigma=0.0)
    assert thresh_no_blur == 127  # Should return default fallback threshold
    assert mask_no_blur.shape == black_white_image_no_blur.shape
