"""Tests for thumbnail downscaling functionality."""

import numpy as np

from histoslice.functional import downscale_for_thumbnail


def test_downscale_for_thumbnail_small_noop() -> None:
    """Test that small images are not modified."""
    img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    out = downscale_for_thumbnail(img, max_pixels=1_000_000)
    # Should return the original array (no resize)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out, img)


def test_downscale_for_thumbnail_large_scales_down() -> None:
    """Test that large images are downscaled properly."""
    h, w = 2000, 2000  # 4,000,000 px > 1,000,000
    img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    out = downscale_for_thumbnail(img, max_pixels=1_000_000)
    # Expected uniform scale factor
    scale = np.sqrt(1_000_000 / float(h * w))
    exp_w = max(1, int(w * scale))
    exp_h = max(1, int(h * scale))
    assert out.shape == (exp_h, exp_w, 3)
    assert out.size <= 1_000_000 * 3  # 3 channels
    assert out.dtype == np.uint8


def test_downscale_for_thumbnail_grayscale() -> None:
    """Test downscaling with grayscale images."""
    h, w = 1500, 1500  # 2,250,000 px > 1,000,000
    img = (np.random.rand(h, w) * 255).astype(np.uint8)
    out = downscale_for_thumbnail(img, max_pixels=1_000_000)
    # Should be downscaled
    assert out.size <= 1_000_000
    assert out.dtype == np.uint8
    assert out.ndim == 2


def test_downscale_for_thumbnail_default_max_pixels() -> None:
    """Test default max_pixels parameter."""
    h, w = 1500, 1500  # 2,250,000 px > 1,000,000 (default)
    img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    out = downscale_for_thumbnail(img)
    # Should be downscaled to ~1M pixels
    assert out.size <= 1_000_000 * 3
    assert out.dtype == np.uint8


def test_downscale_for_thumbnail_empty() -> None:
    """Test with empty image."""
    empty = np.zeros((0, 0), dtype=np.uint8)
    out = downscale_for_thumbnail(empty, max_pixels=1_000_000)
    assert out.shape == (0, 0)
    assert out.dtype == np.uint8


def test_downscale_for_thumbnail_dtype_cast() -> None:
    """Test that non-uint8 images are cast properly."""
    img_f = np.linspace(0, 1, 200 * 200 * 3, dtype=np.float32).reshape(200, 200, 3)
    out_f = downscale_for_thumbnail(img_f, max_pixels=10_000)
    # 200*200 = 40,000 > 10,000 -> expect downscale
    assert out_f.size <= 10_000 * 3
    assert out_f.dtype == np.uint8