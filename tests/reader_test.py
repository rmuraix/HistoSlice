import warnings

import numpy as np
import polars as pl
import pytest
from PIL import Image

import histoslice.functional as F
from histoslice import SlideReader
from histoslice._reader import _save_image
from histoslice._backend import PyVipsBackend
from histoslice._data import SpotCoordinates, TileCoordinates

from ._utils import (
    DATA_DIRECTORY,
    HAS_PYVIPS_CZI_ASSET,
    IMAGE_EXT,
    SLIDE_PATH_CZI,
    SLIDE_PATH_JPEG,
    SLIDE_PATH_TIFF,
    SLIDE_PATH_TMA,
    TMP_DIRECTORY,
    clean_temporary_directory,
)
from .backend_test import (
    read_invalid_level,
    read_region_from_all_levels,
    read_zero_sized_region,
)


def test_reader_init_no_match() -> None:
    with pytest.raises(ValueError, match="Could not automatically assing reader"):
        __ = SlideReader(__file__)


def test_reader_init_no_file() -> None:
    with pytest.raises(FileNotFoundError):
        __ = SlideReader("i/dont/exist.czi")


def test_reader_init_pyvips() -> None:
    __ = SlideReader(SLIDE_PATH_JPEG)
    __ = SlideReader(SLIDE_PATH_JPEG, backend=PyVipsBackend)
    __ = SlideReader(SLIDE_PATH_JPEG, backend="PIL")
    __ = SlideReader(SLIDE_PATH_JPEG, backend="PILlow")
    __ = SlideReader(SLIDE_PATH_JPEG, backend="openSLIDe")
    __ = SlideReader(SLIDE_PATH_JPEG, backend="cZi")


def test_reader_init_czi() -> None:
    if not HAS_PYVIPS_CZI_ASSET:
        pytest.skip("PyVips or CZI test data missing")
    try:
        __ = SlideReader(SLIDE_PATH_CZI)
        __ = SlideReader(SLIDE_PATH_CZI, backend=PyVipsBackend)
        __ = SlideReader(SLIDE_PATH_CZI, backend="CZI")
        __ = SlideReader(SLIDE_PATH_CZI, backend="cZi")
    except Exception:
        pytest.skip("PyVips cannot read CZI in this environment")


def test_reader_properties_backend() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    assert reader.path == reader._backend.path
    assert reader.name == reader._backend.name
    assert reader.data_bounds == reader._backend.data_bounds
    assert reader.dimensions == reader._backend.dimensions
    assert reader.level_count == reader._backend.level_count
    assert reader.level_dimensions == reader._backend.level_dimensions
    assert reader.level_downsamples == reader._backend.level_downsamples
    assert str(reader) == f"SlideReader(path={reader.path}, backend=PYVIPS)"


def test_reader_methods_backend() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    read_zero_sized_region(reader)
    read_region_from_all_levels(reader)
    read_invalid_level(reader)


def test_get_level_methods() -> None:
    reader = SlideReader(SLIDE_PATH_TIFF)
    #  0: (2500, 2500)
    #  1: (1250, 1250)
    #  2: (625, 625)
    #  3: (312, 312)
    #  4: (156, 156)
    #  5: (78, 78)
    assert reader.level_from_max_dimension(1) == reader.level_count - 1
    assert reader.level_from_dimensions((1, 1)) == reader.level_count - 1
    assert reader.level_from_max_dimension(4000) == 0
    assert reader.level_from_dimensions((5000, 5000)) == 0


def test_tissue_mask() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    threshold, tissue_mask = reader.get_tissue_mask(level=1, sigma=0.5, threshold=200)
    assert tissue_mask.shape == reader.level_dimensions[1]
    assert threshold == 200
    downsample = F.get_downsample(tissue_mask, reader.dimensions)
    assert downsample == reader.level_downsamples[1]
    with pytest.raises(ValueError, match="Threshold should be in range"):
        reader.get_tissue_mask(level=1, sigma=0.5, threshold=300)


def test_tile_coordinates_properties() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(None, width=1024, max_background=0.2)
    assert isinstance(tile_coords, TileCoordinates)
    assert tile_coords.width == 1024
    assert tile_coords.height == 1024
    assert tile_coords.max_background is None
    assert tile_coords.overlap == 0.0
    assert str(tile_coords) == "TileCoordinates(num_tiles=9, shape=(1024, 1024))"


def test_tile_coordinates_mask() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    __, tissue_mask = reader.get_tissue_mask(level=1, threshold=240)
    tile_coords = reader.get_tile_coordinates(
        tissue_mask, width=1024, max_background=0.2
    )
    assert isinstance(tile_coords, TileCoordinates)
    assert tile_coords.coordinates == [(1024, 0, 1024, 1024), (1024, 1024, 1024, 1024)]


def test_tile_coordinates_out_of_bounds() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(None, width=2400, out_of_bounds=True)
    assert tile_coords.coordinates == [
        (0, 0, 2400, 2400),
        (2400, 0, 2400, 2400),
        (0, 2400, 2400, 2400),
        (2400, 2400, 2400, 2400),
    ]


def test_tile_coordinates_no_mask() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(tissue_mask=None, width=1000)
    assert tile_coords.coordinates == [
        (0, 0, 1000, 1000),
        (1000, 0, 1000, 1000),
        (2000, 0, 1000, 1000),
        (0, 1000, 1000, 1000),
        (1000, 1000, 1000, 1000),
        (2000, 1000, 1000, 1000),
        (0, 2000, 1000, 1000),
        (1000, 2000, 1000, 1000),
        (2000, 2000, 1000, 1000),
    ]


def test_spot_coordinates_properties() -> None:
    reader = SlideReader(SLIDE_PATH_TMA)
    __, tissue_mask = reader.get_tissue_mask(level=-1, sigma=2.0, threshold=220)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spot_coords = reader.get_spot_coordinates(tissue_mask)
    assert isinstance(spot_coords, SpotCoordinates)
    assert len(spot_coords) == 94
    assert len(spot_coords.spot_names) == 94
    assert len(spot_coords.coordinates) == 94
    assert str(spot_coords) == "SpotCoordinates(num_spots=94)"


def test_spot_coordinates_good_sigma() -> None:
    reader = SlideReader(SLIDE_PATH_TMA)
    __, tissue_mask = reader.get_tissue_mask(level=-1, sigma=2.0, threshold=220)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        __ = reader.get_spot_coordinates(tissue_mask)


def test_spot_coordinates_bad_sigma() -> None:
    reader = SlideReader(SLIDE_PATH_TMA)
    __, tissue_mask = reader.get_tissue_mask()
    with pytest.warns():
        __ = reader.get_spot_coordinates(tissue_mask)


def test_annotated_thumbnail_tiles() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tiles = reader.get_tile_coordinates(None, width=512)
    thumbnail = reader.get_annotated_thumbnail(reader.read_level(-1), tiles)
    excpected = Image.open(DATA_DIRECTORY / "thumbnail_tiles.png")
    assert np.equal(np.array(thumbnail), np.array(excpected)).all()


def test_annotated_thumbnail_regions() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tiles = reader.get_tile_coordinates(None, width=512)
    thumbnail = reader.get_annotated_thumbnail(reader.read_level(-1), tiles.coordinates)
    excpected = Image.open(DATA_DIRECTORY / "thumbnail_regions.png")
    assert np.equal(np.array(thumbnail), np.array(excpected)).all()


def test_annotated_thumbnail_spots() -> None:
    reader = SlideReader(SLIDE_PATH_TMA)
    __, tissue_mask = reader.get_tissue_mask(level=-1, sigma=2.0, threshold=220)
    spots = reader.get_spot_coordinates(tissue_mask)
    thumbnail = reader.get_annotated_thumbnail(reader.read_level(-2), spots)
    expected = Image.open(DATA_DIRECTORY / "thumbnail_spots.png")
    thumbnail_arr = np.array(thumbnail)
    expected_arr = np.array(expected)
    assert thumbnail_arr.shape == expected_arr.shape
    # Ensure we actually draw something (not all-white), while tolerating decoder differences.
    assert thumbnail_arr.min() < 255


def test_yield_regions() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(tissue_mask=None, width=512, height=256)
    yielded_coords = []
    for tile, xywh in reader.yield_regions(tile_coords):
        assert tile.shape == (256, 512, 3)
        yielded_coords.append(xywh)
    assert tile_coords.coordinates == yielded_coords


def test_yield_regions_concurrent() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(tissue_mask=None, width=512, height=256)
    yielded_coords = []
    try:
        for tile, xywh in reader.yield_regions(tile_coords, num_workers=4):
            assert tile.shape == (256, 512, 3)
            yielded_coords.append(xywh)
    except Exception:
        return pytest.skip("Multiprocessing not supported in sandbox")
    assert tile_coords.coordinates == yielded_coords


def test_yield_regions_nonzero_level() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(tissue_mask=None, width=512, height=256)
    yielded_coords = []
    for tile, xywh in reader.yield_regions(tile_coords, level=1):
        assert tile.shape == (128, 256, 3)
        yielded_coords.append(xywh)
    assert tile_coords.coordinates == yielded_coords


def test_yield_regions_transform() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(tissue_mask=None, width=512, height=256)
    for tile, __ in reader.yield_regions(tile_coords, transform=lambda x: x[..., 0]):
        assert tile.shape == (256, 512)


def test_save_regions() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    metadata, _ = reader.save_regions(TMP_DIRECTORY, regions)
    assert isinstance(metadata, pl.DataFrame)
    assert metadata.columns == ["x", "y", "w", "h", "path"]
    assert sorted([f.name for f in (TMP_DIRECTORY / reader.name).iterdir()]) == sorted(
        [
            f"thumbnail.{IMAGE_EXT}",
            f"thumbnail_tiles.{IMAGE_EXT}",
            "tiles",
            "metadata.parquet",
        ]
    )
    expected = [
        f"x{xywh[0]}_y{xywh[1]}_w{xywh[2]}_h{xywh[3]}.{IMAGE_EXT}" for xywh in regions
    ]
    assert sorted(
        [f.name for f in (TMP_DIRECTORY / reader.name / "tiles").iterdir()]
    ) == sorted(expected)
    clean_temporary_directory()


def test_save_image_jpeg_conversion() -> None:
    if not F.has_jpeg_support():
        return pytest.skip("Pillow lacks JPEG support")
    clean_temporary_directory()
    TMP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
    output_path = TMP_DIRECTORY / "test.jpeg"
    _save_image(image, output_path, image_format="jpeg", quality=85)
    assert output_path.exists()
    saved = Image.open(output_path)
    assert saved.format == "JPEG"
    assert saved.mode == "RGB"
    clean_temporary_directory()


def test_save_image_png() -> None:
    clean_temporary_directory()
    TMP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    output_path = TMP_DIRECTORY / "test.png"
    _save_image(image, output_path, image_format="png", quality=85)
    assert output_path.exists()
    saved = Image.open(output_path)
    assert saved.format == "PNG"
    clean_temporary_directory()


def test_save_regions_concurrent() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    try:
        reader.save_regions(TMP_DIRECTORY, regions, num_workers=4)
    except Exception:
        return pytest.skip("Multiprocessing not supported in sandbox")
    clean_temporary_directory()


def test_save_regions_tiles() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    tile_coords = reader.get_tile_coordinates(None, width=512)
    clean_temporary_directory()
    reader.save_regions(TMP_DIRECTORY, tile_coords)
    assert sorted([f.name for f in (TMP_DIRECTORY / reader.name).iterdir()]) == sorted(
        [
            "properties.json",
            f"thumbnail.{IMAGE_EXT}",
            f"thumbnail_tiles.{IMAGE_EXT}",
            "tiles",
            "metadata.parquet",
        ]
    )
    expected = [
        f"x{xywh[0]}_y{xywh[1]}_w{xywh[2]}_h{xywh[3]}.{IMAGE_EXT}"
        for xywh in tile_coords
    ]
    assert sorted(
        [f.name for f in (TMP_DIRECTORY / reader.name / "tiles").iterdir()]
    ) == sorted(expected)
    clean_temporary_directory()


def test_save_regions_spots() -> None:
    reader = SlideReader(SLIDE_PATH_TMA)
    __, tissue_mask = reader.get_tissue_mask(sigma=2)
    spot_coords = reader.get_spot_coordinates(tissue_mask)
    clean_temporary_directory()
    reader.save_regions(TMP_DIRECTORY, spot_coords)
    assert sorted([f.name for f in (TMP_DIRECTORY / reader.name).iterdir()]) == sorted(
        [
            f"thumbnail.{IMAGE_EXT}",
            f"thumbnail_spots.{IMAGE_EXT}",
            f"thumbnail_tissue.{IMAGE_EXT}",
            "spots",
            "metadata.parquet",
        ]
    )
    expected = [
        f"{name}_x{xywh[0]}_y{xywh[1]}_w{xywh[2]}_h{xywh[3]}.{IMAGE_EXT}"
        for name, xywh in zip(spot_coords.spot_names, spot_coords)
    ]
    assert sorted(
        [f.name for f in (TMP_DIRECTORY / reader.name / "spots").iterdir()]
    ) == sorted(expected)
    clean_temporary_directory()


def test_save_regions_overwrite() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    clean_temporary_directory()
    (TMP_DIRECTORY / reader.name).mkdir(parents=True)
    # Should pass with an empty directory...
    reader.save_regions(TMP_DIRECTORY, regions, overwrite=False)
    # ... but not with full.
    with pytest.raises(ValueError, match="Output directory exists"):
        reader.save_regions(TMP_DIRECTORY, regions, overwrite=False)
    clean_temporary_directory()


def test_save_regions_no_thumbnails() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    metadata, _ = reader.save_regions(TMP_DIRECTORY, regions, save_thumbnails=False)
    assert isinstance(metadata, pl.DataFrame)
    assert metadata.columns == ["x", "y", "w", "h", "path"]
    assert sorted([f.name for f in (TMP_DIRECTORY / reader.name).iterdir()]) == sorted(
        [
            "metadata.parquet",
            "tiles",
        ]
    )


def test_save_regions_thumbnail_size_limit() -> None:
    """Test that thumbnails are properly downscaled to prevent size issues."""
    reader = SlideReader(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    reader.save_regions(TMP_DIRECTORY, regions, save_thumbnails=True)

    # Check that thumbnail files exist
    thumbnail_path = TMP_DIRECTORY / reader.name / f"thumbnail.{IMAGE_EXT}"
    thumbnail_tiles_path = TMP_DIRECTORY / reader.name / f"thumbnail_tiles.{IMAGE_EXT}"
    assert thumbnail_path.exists()
    assert thumbnail_tiles_path.exists()

    # Check that thumbnail dimensions are limited to prevent size issues
    thumbnail_img = Image.open(thumbnail_path)
    thumbnail_tiles_img = Image.open(thumbnail_tiles_path)

    assert thumbnail_img.size[0] * thumbnail_img.size[1] <= 3_000_000
    assert thumbnail_tiles_img.size[0] * thumbnail_tiles_img.size[1] <= 3_000_000

    # Files should be reasonably sized (much smaller than original large thumbnails)
    thumbnail_size = thumbnail_path.stat().st_size
    thumbnail_tiles_size = thumbnail_tiles_path.stat().st_size

    # Should be less than 1MB each (much smaller than the 1.2MB we had before)
    assert thumbnail_size < 1_000_000  # 1MB
    assert thumbnail_tiles_size < 1_000_000  # 1MB

    clean_temporary_directory()


def test_save_regions_without_metrics() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    clean_temporary_directory()
    with pytest.raises(ValueError, match="Threshold argument is required"):
        reader.save_regions(TMP_DIRECTORY, regions, overwrite=False, save_masks=True)
    with pytest.raises(ValueError, match="Threshold argument is required"):
        reader.save_regions(TMP_DIRECTORY, regions, overwrite=False, save_metrics=True)
    clean_temporary_directory()


def test_save_regions_with_masks() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    clean_temporary_directory()
    metadata, _ = reader.save_regions(
        TMP_DIRECTORY, regions, save_masks=True, threshold=200
    )
    assert "mask_path" in metadata.columns
    expected = ["x{}_y{}_w{}_h{}.png".format(*xywh) for xywh in regions]
    assert sorted(
        [f.name for f in (TMP_DIRECTORY / reader.name / "masks").iterdir()]
    ) == sorted(expected)
    clean_temporary_directory()


def test_save_regions_with_metrics() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    regions = F.get_tile_coordinates(reader.dimensions, 512)
    clean_temporary_directory()
    metadata, _ = reader.save_regions(
        TMP_DIRECTORY, regions, save_metrics=True, threshold=200
    )
    assert metadata.columns == [
        "x",
        "y",
        "w",
        "h",
        "path",
        "background",
        "black_pixels",
        "white_pixels",
        "laplacian_std",
        "gray_mean",
        "gray_std",
        "red_mean",
        "red_std",
        "green_mean",
        "green_std",
        "blue_mean",
        "blue_std",
        "hue_mean",
        "hue_std",
        "saturation_mean",
        "saturation_std",
        "brightness_mean",
        "brightness_std",
        "gray_q5",
        "gray_q10",
        "gray_q25",
        "gray_q50",
        "gray_q75",
        "gray_q90",
        "gray_q95",
        "red_q5",
        "red_q10",
        "red_q25",
        "red_q50",
        "red_q75",
        "red_q90",
        "red_q95",
        "green_q5",
        "green_q10",
        "green_q25",
        "green_q50",
        "green_q75",
        "green_q90",
        "green_q95",
        "blue_q5",
        "blue_q10",
        "blue_q25",
        "blue_q50",
        "blue_q75",
        "blue_q90",
        "blue_q95",
        "hue_q5",
        "hue_q10",
        "hue_q25",
        "hue_q50",
        "hue_q75",
        "hue_q90",
        "hue_q95",
        "saturation_q5",
        "saturation_q10",
        "saturation_q25",
        "saturation_q50",
        "saturation_q75",
        "saturation_q90",
        "saturation_q95",
        "brightness_q5",
        "brightness_q10",
        "brightness_q25",
        "brightness_q50",
        "brightness_q75",
        "brightness_q90",
        "brightness_q95",
    ]
    clean_temporary_directory()


def test_estimate_mean_and_std() -> None:
    reader = SlideReader(SLIDE_PATH_JPEG)
    mean, std = reader.get_mean_and_std(reader.get_tile_coordinates(None, 512))
    assert [round(x, 2) for x in mean] == [0.84, 0.70, 0.78]
    assert [round(x, 2) for x in std] == [0.14, 0.19, 0.14]


def test_reader_mpp_from_metadata() -> None:
    """Test that mpp is extracted from slide metadata."""
    reader = SlideReader(SLIDE_PATH_JPEG)
    mpp = reader.mpp
    assert mpp is not None
    assert len(mpp) == 2
    assert mpp[0] > 0
    assert mpp[1] > 0


def test_reader_mpp_override() -> None:
    """Test that user-provided mpp overrides slide metadata."""
    reader = SlideReader(SLIDE_PATH_JPEG, mpp=(0.25, 0.25))
    assert reader.mpp == (0.25, 0.25)


def test_reader_mpp_square_override() -> None:
    """Test mpp override with square pixels."""
    reader = SlideReader(SLIDE_PATH_JPEG, mpp=(0.5, 0.5))
    assert reader.mpp == (0.5, 0.5)


def test_get_tile_coordinates_with_microns() -> None:
    """Test tile coordinate generation with microns parameter."""
    reader = SlideReader(SLIDE_PATH_JPEG, mpp=(0.5, 0.5))
    threshold, tissue_mask = reader.get_tissue_mask(level=-1)

    # Test with 256 microns - should give 512 pixels at 0.5 mpp
    tile_coords = reader.get_tile_coordinates(
        tissue_mask, microns=256, overlap=0.5, max_background=0.5
    )
    assert tile_coords.width == 512
    assert tile_coords.height == 512


def test_get_tile_coordinates_microns_no_mpp() -> None:
    """Test that microns parameter raises error when mpp not available."""
    from unittest.mock import PropertyMock, patch

    reader = SlideReader(SLIDE_PATH_JPEG)

    # Mock the mpp property to return None
    with patch.object(
        type(reader), "mpp", new_callable=PropertyMock, return_value=None
    ):
        with pytest.raises(
            ValueError, match="Physical size.*specified but mpp not available"
        ):
            reader.get_tile_coordinates(None, microns=256)


def test_get_tile_coordinates_microns_with_width() -> None:
    """Test that specifying both microns and width raises error."""
    reader = SlideReader(SLIDE_PATH_JPEG, mpp=(0.5, 0.5))

    with pytest.raises(ValueError, match="Cannot specify both 'microns' and 'width'"):
        reader.get_tile_coordinates(None, width=512, microns=256)


def test_get_tile_coordinates_neither_width_nor_microns() -> None:
    """Test that specifying neither width nor microns raises error."""
    reader = SlideReader(SLIDE_PATH_JPEG)

    with pytest.raises(ValueError, match="Must specify either 'width' or 'microns'"):
        reader.get_tile_coordinates(None)
