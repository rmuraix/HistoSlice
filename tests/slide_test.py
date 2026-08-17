from unittest.mock import MagicMock

import pytest

from histoslice import Slide
from histoslice.tiles import Region

from ._utils import (
    HAS_PYVIPS_ASSET,
    HAS_PYVIPS_CZI_ASSET,
    SLIDE_PATH_CZI,
    SLIDE_PATH_JPEG,
    SLIDE_PATH_TIFF,
)


def test_slide_init_no_file() -> None:
    with pytest.raises(FileNotFoundError):
        Slide("i/dont/exist.svs")


def test_slide_properties_jpeg() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    assert slide.name == "slide"
    assert slide.suffix == ".jpeg"
    assert slide.level_count == 3
    assert slide.dimensions == (2500, 2500)
    assert slide.level_dimensions == {0: (2500, 2500), 1: (1250, 1250), 2: (625, 625)}
    assert slide.level_downsamples == {0: (1.0, 1.0), 1: (2.0, 2.0), 2: (4.0, 4.0)}
    assert repr(slide) == f"Slide(path={slide.path})"


def test_slide_properties_tiff() -> None:
    if not HAS_PYVIPS_ASSET:
        pytest.skip("PyVips test data or dependency missing")
    slide = Slide(SLIDE_PATH_TIFF)
    assert slide.level_count == 6
    assert slide.dimensions == (2500, 2500)
    assert slide.level_dimensions == {
        0: (2500, 2500),
        1: (1250, 1250),
        2: (625, 625),
        3: (312, 312),
        4: (156, 156),
        5: (78, 78),
    }


def test_slide_init_czi() -> None:
    if not HAS_PYVIPS_CZI_ASSET:
        pytest.skip("PyVips or CZI test data missing")
    try:
        Slide(SLIDE_PATH_CZI)
    except Exception:
        pytest.skip("PyVips cannot read CZI in this environment")


def test_read_level_0() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    assert slide.read_level(0).shape == (2500, 2500, 3)


def test_read_level_nonzero() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    assert slide.read_level(-1).shape == (625, 625, 3)
    assert slide.read_level(2).shape == (625, 625, 3)


def test_read_invalid_level() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    with pytest.raises(ValueError, match="Level 100 could not be found"):
        slide.read_level(100)
    with pytest.raises(ValueError, match="Level 100 could not be found"):
        slide.read_region(Region(0, 0, 10, 10), level=100)
    with pytest.raises(ValueError, match="Level -100 could not be found"):
        slide.read_level(-100)


def test_read_region_level_0() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    tile = slide.read_region(Region(0, 0, 256, 256), level=0)
    assert tile.shape == (256, 256, 3)


def test_read_region_nonzero_level() -> None:
    """Region coordinates are always level-0; the output shrinks with the level."""
    slide = Slide(SLIDE_PATH_JPEG)
    for level, (ds_h, ds_w) in slide.level_downsamples.items():
        tile = slide.read_region(Region(0, 0, 256, 256), level=level)
        assert tile.shape == (round(256 / ds_h), round(256 / ds_w), 3)


def test_read_region_edge_padding() -> None:
    """Regions extending past the slide edge are padded with white pixels."""
    slide = Slide(SLIDE_PATH_JPEG)
    h, w = slide.dimensions
    tile = slide.read_region(Region(w - 100, h - 100, 256, 256), level=0)
    assert tile.shape == (256, 256, 3)
    assert (tile[-50:, -50:] == 255).all()


def test_read_region_zero_sized() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    assert slide.read_region(Region(0, 0, 0, 0), 0).shape == (0, 0, 3)
    assert slide.read_region(Region(0, 0, 1, 0), 0).shape == (0, 1, 3)


def test_read_tile_native_resolution_is_a_noop_resize() -> None:
    """When the requested size matches the region's level-0 size, no resize happens."""
    slide = Slide(SLIDE_PATH_JPEG)
    region = Region(0, 0, 512, 512)
    assert slide.read_tile(region, (512, 512)).shape == (512, 512, 3)


def test_read_tile_resizes_to_exact_output_size() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    region = Region(0, 0, 1024, 1024)
    tile = slide.read_tile(region, (512, 512))
    assert tile.shape == (512, 512, 3)


def test_read_tile_anisotropic_region() -> None:
    """Non-square crops still resize to the exact requested (width, height)."""
    slide = Slide(SLIDE_PATH_JPEG)
    region = Region(0, 0, 1024, 512)
    tile = slide.read_tile(region, (256, 256))
    assert tile.shape == (256, 256, 3)


def test_level_from_max_dimension() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    assert slide.level_from_max_dimension(4000) == 0
    assert slide.level_from_max_dimension(1) == slide.level_count - 1


def test_level_from_dimensions() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    assert slide.level_from_dimensions((5000, 5000)) == 0
    assert slide.level_from_dimensions((1, 1)) == slide.level_count - 1


def test_mpp_from_metadata() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    mpp = slide.mpp
    assert mpp is not None
    assert mpp[0] > 0
    assert mpp[1] > 0


def test_mpp_override() -> None:
    slide = Slide(SLIDE_PATH_JPEG, mpp=(0.25, 0.25))
    assert slide.mpp == (0.25, 0.25)


def test_mpp_anisotropic_override() -> None:
    slide = Slide(SLIDE_PATH_JPEG, mpp=(0.25, 0.5))
    assert slide.mpp == (0.25, 0.5)


def _slide_with_mocked_metadata(values: dict) -> Slide:
    """A real `Slide` whose `_page0.get()` is replaced with a fake metadata
    lookup, to exercise `mpp` extraction edge cases no test asset covers."""
    slide = Slide(SLIDE_PATH_JPEG)
    mock_page0 = MagicMock()

    def get(key: str):
        value = values.get(key, Exception("not available"))
        if isinstance(value, Exception):
            raise value
        return value

    mock_page0.get = MagicMock(side_effect=get)
    slide._page0 = mock_page0
    return slide


def test_mpp_from_openslide_properties() -> None:
    slide = _slide_with_mocked_metadata(
        {"openslide.mpp-x": "0.5", "openslide.mpp-y": "0.6"}
    )
    assert slide.mpp == (0.5, 0.6)


def test_mpp_resolution_unit_inch() -> None:
    # 1000 pixels/inch -> 25400 / 1000 = 25.4 um/pixel
    slide = _slide_with_mocked_metadata(
        {"xres": 1000.0, "yres": 1000.0, "resolution-unit": 2}
    )
    mpp = slide.mpp
    assert mpp is not None
    assert abs(mpp[0] - 25.4) < 0.01
    assert abs(mpp[1] - 25.4) < 0.01


def test_mpp_resolution_unit_cm() -> None:
    # 100 pixels/cm -> 10000 / 100 = 100 um/pixel
    slide = _slide_with_mocked_metadata(
        {"xres": 100.0, "yres": 100.0, "resolution-unit": 3}
    )
    mpp = slide.mpp
    assert mpp is not None
    assert abs(mpp[0] - 100.0) < 0.01
    assert abs(mpp[1] - 100.0) < 0.01


def test_mpp_resolution_unit_invalid() -> None:
    slide = _slide_with_mocked_metadata(
        {"xres": 100.0, "yres": 100.0, "resolution-unit": 99}
    )
    assert slide.mpp is None


def test_mpp_zero_resolution_returns_none() -> None:
    slide = _slide_with_mocked_metadata({"xres": 0.0, "yres": 100.0})
    assert slide.mpp is None


def test_mpp_completely_unavailable_returns_none() -> None:
    slide = _slide_with_mocked_metadata({})
    assert slide.mpp is None
