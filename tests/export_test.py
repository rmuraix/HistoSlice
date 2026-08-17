import numpy as np
import polars as pl
import pytest
import pyvips

from histoslice import Slide
from histoslice.export import (
    ExportResult,
    _resolve_image_format,
    _save_image,
    export_tiles,
)
from histoslice.functional import has_jpeg_support
from histoslice.functional._imageio import read_image
from histoslice.tiles import Region, tile_regions

from ._utils import IMAGE_EXT, SLIDE_PATH_JPEG, TMP_DIRECTORY, clean_temporary_directory


def _regions(slide: Slide, size: int = 512) -> list[Region]:
    return tile_regions(slide.dimensions, size, out_of_bounds=False)


def test_export_tiles_basic() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    regions = _regions(slide)
    result = export_tiles(slide, regions, TMP_DIRECTORY / slide.name, tile_size=512)
    assert isinstance(result, ExportResult)
    assert isinstance(result.metadata, pl.DataFrame)
    assert result.metadata.columns == ["x", "y", "w", "h", "path"]
    assert len(result.metadata) == len(regions)
    assert result.failures == []
    assert sorted(p.name for p in result.output_dir.iterdir()) == sorted(
        [
            f"thumbnail.{IMAGE_EXT}",
            f"thumbnail_tiles.{IMAGE_EXT}",
            "tiles",
            "metadata.parquet",
        ]
    )
    expected = [f"x{r.x}_y{r.y}_w{r.width}_h{r.height}.{IMAGE_EXT}" for r in regions]
    assert sorted(p.name for p in (result.output_dir / "tiles").iterdir()) == sorted(
        expected
    )
    clean_temporary_directory()


def test_export_tiles_output_dimensions() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    regions = _regions(slide, size=300)
    result = export_tiles(
        slide,
        regions,
        TMP_DIRECTORY / slide.name,
        tile_size=(300, 300),
        save_thumbnails=False,
    )
    for path in result.metadata["path"]:
        assert read_image(path).shape[:2] == (300, 300)
    clean_temporary_directory()


def test_export_tiles_no_thumbnails() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    regions = _regions(slide)
    result = export_tiles(
        slide, regions, TMP_DIRECTORY / slide.name, tile_size=512, save_thumbnails=False
    )
    assert sorted(p.name for p in result.output_dir.iterdir()) == sorted(
        ["tiles", "metadata.parquet"]
    )
    clean_temporary_directory()


def test_export_tiles_overwrite() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    output_dir = TMP_DIRECTORY / slide.name
    output_dir.mkdir(parents=True)
    regions = _regions(slide)
    # Empty directory is fine even without overwrite...
    export_tiles(
        slide,
        regions,
        output_dir,
        tile_size=512,
        overwrite=False,
        save_thumbnails=False,
    )
    # ...but a populated one requires overwrite=True.
    with pytest.raises(ValueError, match="Output directory exists"):
        export_tiles(slide, regions, output_dir, tile_size=512, overwrite=False)
    export_tiles(
        slide, regions, output_dir, tile_size=512, overwrite=True, save_thumbnails=False
    )
    clean_temporary_directory()


def test_export_tiles_requires_threshold_for_masks_and_metrics() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    regions = _regions(slide)
    with pytest.raises(ValueError, match="Threshold argument is required"):
        export_tiles(
            slide, regions, TMP_DIRECTORY / slide.name, tile_size=512, save_masks=True
        )
    with pytest.raises(ValueError, match="Threshold argument is required"):
        export_tiles(
            slide, regions, TMP_DIRECTORY / slide.name, tile_size=512, save_metrics=True
        )


def test_export_tiles_with_masks() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    regions = _regions(slide)
    result = export_tiles(
        slide,
        regions,
        TMP_DIRECTORY / slide.name,
        tile_size=512,
        save_masks=True,
        threshold=200,
        save_thumbnails=False,
    )
    assert "mask_path" in result.metadata.columns
    expected = [f"x{r.x}_y{r.y}_w{r.width}_h{r.height}.png" for r in regions]
    assert sorted(p.name for p in (result.output_dir / "masks").iterdir()) == sorted(
        expected
    )
    clean_temporary_directory()


def test_export_tiles_with_metrics() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    regions = _regions(slide)
    result = export_tiles(
        slide,
        regions,
        TMP_DIRECTORY / slide.name,
        tile_size=512,
        save_metrics=True,
        threshold=200,
        save_thumbnails=False,
    )
    assert "background" in result.metadata.columns
    assert "laplacian_std" in result.metadata.columns
    assert "red_mean" in result.metadata.columns
    clean_temporary_directory()


def test_export_tiles_names_and_region_dir() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    regions = _regions(slide)[:3]
    names = [f"spot_{i}" for i in range(len(regions))]
    result = export_tiles(
        slide,
        regions,
        TMP_DIRECTORY / slide.name,
        tile_size=512,
        names=names,
        region_dir="spots",
    )
    tile_files = sorted(p.name for p in (result.output_dir / "spots").iterdir())
    expected = [
        f"{name}_x{r.x}_y{r.y}_w{r.width}_h{r.height}.{IMAGE_EXT}"
        for name, r in zip(names, regions)
    ]
    assert tile_files == sorted(expected)
    assert f"thumbnail_spots.{IMAGE_EXT}" in [
        p.name for p in result.output_dir.iterdir()
    ]
    clean_temporary_directory()


def test_export_tiles_per_tile_failure_reporting() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    regions = _regions(slide)

    calls = {"n": 0}
    original_read_tile = slide.read_tile

    def flaky_read_tile(region, size):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return original_read_tile(region, size)

    slide.read_tile = flaky_read_tile
    result = export_tiles(
        slide, regions, TMP_DIRECTORY / slide.name, tile_size=512, save_thumbnails=False
    )
    assert len(result.failures) == 1
    assert len(result.metadata) == len(regions) - 1
    assert (result.output_dir / "failures.json").exists()
    clean_temporary_directory()


def test_export_tiles_output_dir_is_file_raises() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    TMP_DIRECTORY.mkdir(parents=True)
    output_dir = TMP_DIRECTORY / slide.name
    output_dir.touch()  # a file, not a directory
    with pytest.raises(NotADirectoryError, match="Output directory exists"):
        export_tiles(slide, _regions(slide), output_dir, tile_size=512)
    clean_temporary_directory()


def test_export_tiles_thumbnail_resizes_mismatched_tissue_mask() -> None:
    """A tissue mask read at a different level than the thumbnail is resized
    to match before being drawn as the tissue overlay."""
    from histoslice.tissue import tissue_mask as detect_tissue_mask

    slide = Slide(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    __, mask = detect_tissue_mask(slide.read_level(-1))
    assert mask.shape[:2] != slide.level_dimensions[0]
    result = export_tiles(
        slide,
        _regions(slide),
        TMP_DIRECTORY / slide.name,
        tile_size=512,
        thumbnail_level=0,
        tissue_mask=mask,
    )
    assert f"thumbnail_tissue.{IMAGE_EXT}" in [
        p.name for p in result.output_dir.iterdir()
    ]
    clean_temporary_directory()


def test_export_tiles_png_thumbnails_are_downscaled() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    clean_temporary_directory()
    result = export_tiles(
        slide,
        _regions(slide),
        TMP_DIRECTORY / slide.name,
        tile_size=512,
        image_format="png",
        thumbnail_level=0,
    )
    thumbnail = read_image(result.output_dir / "thumbnail.png")
    assert thumbnail.shape[0] * thumbnail.shape[1] <= 300_000
    clean_temporary_directory()


def test_resolve_image_format_falls_back_to_png_without_jpeg_support(
    monkeypatch,
) -> None:
    monkeypatch.setattr("histoslice.export.has_jpeg_support", lambda: False)
    assert _resolve_image_format("jpeg") == "png"
    assert _resolve_image_format("tiff") == "tiff"


def test_save_image_jpeg_conversion() -> None:
    if not has_jpeg_support():
        pytest.skip("libvips lacks JPEG support")
    clean_temporary_directory()
    TMP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    image = np.zeros((10, 10), dtype=np.uint8)
    output_path = TMP_DIRECTORY / "test.jpeg"
    _save_image(image, output_path, image_format="jpeg", quality=85)
    saved = pyvips.Image.new_from_file(str(output_path))
    assert saved.get("vips-loader") == "jpegload"
    assert saved.width == 10 and saved.height == 10
    clean_temporary_directory()


def test_save_image_png() -> None:
    clean_temporary_directory()
    TMP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    output_path = TMP_DIRECTORY / "test.png"
    _save_image(image, output_path, image_format="png", quality=85)
    saved = pyvips.Image.new_from_file(str(output_path))
    assert saved.get("vips-loader") == "pngload"
    clean_temporary_directory()
