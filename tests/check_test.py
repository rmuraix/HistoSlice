"""Tests for histoslice.functional._check module."""

import numpy as np
import pytest
from PIL import Image

from histoslice.functional._check import check_image


def test_check_image_with_numpy_array() -> None:
    """Test check_image with a valid numpy array."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = check_image(image)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == (100, 100, 3)


def test_check_image_with_pil_image() -> None:
    """Test check_image with a PIL Image."""
    pil_image = Image.new("RGB", (100, 100), color=(255, 0, 0))
    result = check_image(pil_image)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == (100, 100, 3)


def test_check_image_with_grayscale() -> None:
    """Test check_image with a grayscale image."""
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = check_image(image)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == (100, 100)


def test_check_image_invalid_type() -> None:
    """Test check_image raises TypeError with invalid type."""
    with pytest.raises(TypeError, match="Expected an array image"):
        check_image("not an image")  # type: ignore


def test_check_image_invalid_dimensions() -> None:
    """Test check_image raises TypeError with invalid dimensions."""
    # 1D array
    with pytest.raises(TypeError, match="Image should have 2 or 3 dimensions"):
        check_image(np.array([1, 2, 3], dtype=np.uint8))

    # 4D array
    with pytest.raises(TypeError, match="Image should have 2 or 3 dimensions"):
        check_image(np.random.randint(0, 255, (10, 10, 10, 3), dtype=np.uint8))


def test_check_image_invalid_channels() -> None:
    """Test check_image raises TypeError with invalid number of channels."""
    # 1 channel in 3D array (should be 3)
    with pytest.raises(TypeError, match="Image should have 3 colour channels"):
        check_image(np.random.randint(0, 255, (100, 100, 1), dtype=np.uint8))

    # 4 channels
    with pytest.raises(TypeError, match="Image should have 3 colour channels"):
        check_image(np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8))


def test_check_image_invalid_dtype() -> None:
    """Test check_image raises TypeError with invalid dtype."""
    # float32 instead of uint8
    with pytest.raises(TypeError, match="Expected image dtype to be uint8"):
        check_image(np.random.rand(100, 100, 3).astype(np.float32))

    # int32 instead of uint8
    with pytest.raises(TypeError, match="Expected image dtype to be uint8"):
        check_image(np.random.randint(0, 255, (100, 100, 3), dtype=np.int32))
