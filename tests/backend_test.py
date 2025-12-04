from __future__ import annotations

import pytest

from histoslice._backend import PyVipsBackend
from tests._utils import (
    SLIDE_PATH_JPEG,
    SLIDE_PATH_TIFF,
    HAS_PYVIPS_ASSET,
)


def read_zero_sized_region(backend: PyVipsBackend) -> None:
    assert backend.read_region((0, 0, 0, 0), 0).shape == (0, 0, 3)
    assert backend.read_region((0, 0, 1, 0), 0).shape == (0, 1, 3)
    if len(backend.level_dimensions) > 1:
        assert backend.read_region((0, 0, 1, 1), level=1).shape == (0, 0, 3)


def read_region_from_all_levels(
    backend: PyVipsBackend,
    tile_width: int = 256,
) -> None:
    for level in backend.level_dimensions:
        tile_dims = backend.read_region(
            (0, 0, tile_width, tile_width), level=level
        ).shape
        h_d, w_d = backend.level_downsamples[level]
        expected_dims = (round(tile_width / h_d), round(tile_width / w_d), 3)
        assert tile_dims == expected_dims


def read_invalid_level(backend: PyVipsBackend) -> None:
    with pytest.raises(ValueError, match="Level 100 could not be found"):
        backend.read_region((0, 0, 10, 10), level=100)
    with pytest.raises(ValueError, match="Level 100 could not be found"):
        backend.read_region((0, 0, 10, 10), level=100)


def test_pyvips_backend_properties() -> None:
    """Test PyVipsBackend properties."""
    if not HAS_PYVIPS_ASSET:
        pytest.skip("PyVips test data or dependency missing")
    backend = PyVipsBackend(SLIDE_PATH_TIFF)
    assert backend.path == str(SLIDE_PATH_TIFF.resolve())
    assert backend.name == "slide"
    assert backend.suffix == ".tiff"
    assert backend.reader is not None


def test_pyvips_backend_invalid_path() -> None:
    """Test PyVipsBackend with invalid path."""
    from pathlib import Path

    with pytest.raises(FileNotFoundError):
        PyVipsBackend(Path("/nonexistent/path.tiff"))


def test_pyvips_init() -> None:
    if not HAS_PYVIPS_ASSET:
        pytest.skip("PyVips test data or dependency missing")
    __ = PyVipsBackend(SLIDE_PATH_TIFF)
    # PyVips should be able to handle simple JPEG files now
    __ = PyVipsBackend(SLIDE_PATH_JPEG)


def test_zero_region_pyvips() -> None:
    if not HAS_PYVIPS_ASSET:
        pytest.skip("PyVips test data or dependency missing")
    backend = PyVipsBackend(SLIDE_PATH_TIFF)
    read_zero_sized_region(backend)


def test_invalid_level_pyvips() -> None:
    if not HAS_PYVIPS_ASSET:
        pytest.skip("PyVips test data or dependency missing")
    backend = PyVipsBackend(SLIDE_PATH_TIFF)
    read_invalid_level(backend)


def test_read_region_pyvips() -> None:
    if not HAS_PYVIPS_ASSET:
        pytest.skip("PyVips test data or dependency missing")
    backend = PyVipsBackend(SLIDE_PATH_TIFF)
    read_region_from_all_levels(backend)


def test_read_level_pyvips() -> None:
    if not HAS_PYVIPS_ASSET:
        pytest.skip("PyVips test data or dependency missing")
    backend = PyVipsBackend(SLIDE_PATH_TIFF)
    assert backend.read_level(-1).shape == (78, 78, 3)


def test_properties_pyvips() -> None:
    """Test properties for PyVips backend that reads the pyramidal TIFF asset."""
    if not HAS_PYVIPS_ASSET:
        pytest.skip("PyVips test data or dependency missing")

    backend = PyVipsBackend(SLIDE_PATH_TIFF)
    assert backend.path == str(SLIDE_PATH_TIFF)
    assert backend.name == "slide"
    assert backend.suffix == ".tiff"
    assert backend.BACKEND_NAME == "PYVIPS"
    assert backend.level_count == 6
    assert backend.dimensions == (2500, 2500)
    assert backend.level_dimensions == {
        0: (2500, 2500),
        1: (1250, 1250),
        2: (625, 625),
        3: (312, 312),
        4: (156, 156),
        5: (78, 78),
    }
    assert backend.level_downsamples == {
        0: (1.0, 1.0),
        1: (2.0, 2.0),
        2: (4.0, 4.0),
        3: (8.012820512820513, 8.012820512820513),
        4: (16.025641025641026, 16.025641025641026),
        5: (32.05128205128205, 32.05128205128205),
    }
    assert backend.data_bounds == (0, 0, 2500, 2500)

    # Check reader type
    import pyvips

    assert isinstance(backend.reader, pyvips.Image)


def test_pyvips_jpeg_support() -> None:
    """Test that PyVips backend can handle non-pyramidal JPEG files."""
    if not HAS_PYVIPS_ASSET:
        pytest.skip("PyVips test data or dependency missing")

    backend = PyVipsBackend(SLIDE_PATH_JPEG)
    assert backend.path == str(SLIDE_PATH_JPEG.resolve())
    assert backend.name == "slide"
    assert backend.suffix == ".jpeg"
    assert backend.BACKEND_NAME == "PYVIPS"
    assert backend.level_count == 1  # JPEG files have only one level
    assert backend.dimensions == (2500, 2500)
    assert backend.level_dimensions == {0: (2500, 2500)}
    assert backend.level_downsamples == {0: (1.0, 1.0)}

    # Test reading
    level_data = backend.read_level(0)
    assert level_data.shape == (2500, 2500, 3)

    # Test region reading
    region_data = backend.read_region((0, 0, 100, 100), 0)
    assert region_data.shape == (100, 100, 3)
