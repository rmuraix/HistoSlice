from __future__ import annotations

import pytest
from histoslice._backend import PyVipsBackend
from tests._utils import (
    SLIDE_PATH_CZI,
    SLIDE_PATH_JPEG,
    SLIDE_PATH_TIFF,
    HAS_PYVIPS_ASSET,
    HAS_PYVIPS_CZI_ASSET,
    HAS_PYVIPS_JPEG_ASSET,
)


def read_zero_sized_region(backend: PyVipsBackend) -> None:
    assert backend.read_region((0, 0, 0, 0), 0).shape == (0, 0, 3)
    assert backend.read_region((0, 0, 1, 0), 0).shape == (0, 1, 3)
    if len(backend.level_dimensions) > 1:
        assert backend.read_region((0, 0, 1, 1), level=1).shape == (0, 0, 3)


def read_region_from_all_levels(backend: PyVipsBackend, tile_width: int = 256) -> None:
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


def test_pyvips_backend_properties_jpeg() -> None:
    """Test PyVipsBackend properties for flat JPEG asset."""
    if not HAS_PYVIPS_JPEG_ASSET:
        pytest.skip("PyVips or JPEG test data missing")
    backend = PyVipsBackend(SLIDE_PATH_JPEG)
    assert backend.path == str(SLIDE_PATH_JPEG.resolve())
    assert backend.name == "slide"
    assert backend.suffix == ".jpeg"
    assert backend.reader is not None
    assert backend.BACKEND_NAME == "PYVIPS"
    assert backend.level_count == 3
    assert backend.dimensions == (2500, 2500)
    assert backend.level_dimensions == {0: (2500, 2500), 1: (1250, 1250), 2: (625, 625)}
    assert backend.level_downsamples == {0: (1.0, 1.0), 1: (2.0, 2.0), 2: (4.0, 4.0)}
    assert backend.data_bounds == (0, 0, 2500, 2500)


def test_pyvips_backend_invalid_path() -> None:
    """Test PyVipsBackend with invalid path."""
    from pathlib import Path

    with pytest.raises(FileNotFoundError):
        PyVipsBackend(Path("/nonexistent/path.jpeg"))


def test_pyvips_init_tiff() -> None:
    if not HAS_PYVIPS_ASSET:
        pytest.skip("PyVips test data or dependency missing")
    __ = PyVipsBackend(SLIDE_PATH_TIFF)


def test_pyvips_init_czi() -> None:
    if not HAS_PYVIPS_CZI_ASSET:
        pytest.skip("PyVips or CZI test data missing")
    try:
        __ = PyVipsBackend(SLIDE_PATH_CZI)
    except Exception:
        pytest.skip("PyVips cannot read CZI in this environment")


def test_zero_region_pyvips_jpeg() -> None:
    if not HAS_PYVIPS_JPEG_ASSET:
        pytest.skip("PyVips or JPEG test data missing")
    backend = PyVipsBackend(SLIDE_PATH_JPEG)
    read_zero_sized_region(backend)


def test_invalid_level_pyvips_jpeg() -> None:
    if not HAS_PYVIPS_JPEG_ASSET:
        pytest.skip("PyVips or JPEG test data missing")
    backend = PyVipsBackend(SLIDE_PATH_JPEG)
    read_invalid_level(backend)


def test_read_region_pyvips_jpeg() -> None:
    if not HAS_PYVIPS_JPEG_ASSET:
        pytest.skip("PyVips or JPEG test data missing")
    backend = PyVipsBackend(SLIDE_PATH_JPEG)
    read_region_from_all_levels(backend)


def test_read_level_pyvips_jpeg() -> None:
    if not HAS_PYVIPS_JPEG_ASSET:
        pytest.skip("PyVips or JPEG test data missing")
    backend = PyVipsBackend(SLIDE_PATH_JPEG)
    assert backend.read_level(-1).shape == (625, 625, 3)


def test_tiff_backend_properties() -> None:
    """Test properties for the pyramidal TIFF asset."""
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

    import pyvips

    assert isinstance(backend.reader, pyvips.Image)


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
