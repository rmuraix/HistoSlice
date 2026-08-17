import shutil
import time
from pathlib import Path

import polars as pl

from histoslice import Slide
from histoslice.functional import has_jpeg_support

DATA_DIRECTORY = Path(__file__).parent / "data"
TMP_DIRECTORY = DATA_DIRECTORY.parent / "tmp"
SLIDE_PATH_JPEG = DATA_DIRECTORY / "slide.jpeg"
SLIDE_PATH_TIFF = DATA_DIRECTORY / "slide.tiff"
SLIDE_PATH_SVS = DATA_DIRECTORY / "slide.svs"
SLIDE_PATH_CZI = DATA_DIRECTORY / "slide.czi"
SLIDE_PATH_TMA = DATA_DIRECTORY / "tma_spots.jpeg"

IMAGE = Slide(SLIDE_PATH_JPEG).read_level(-1)[:500, :500, :]

IMAGE_EXT = "jpeg" if has_jpeg_support() else "png"


def clean_temporary_directory() -> None:
    # Retry: some filesystems (notably overlay/networked ones under containers)
    # occasionally race shutil.rmtree's directory scan against still-flushing
    # writes from the previous test, raising a spurious "not empty" error.
    for attempt in range(5):
        if not TMP_DIRECTORY.exists():
            return
        try:
            shutil.rmtree(TMP_DIRECTORY)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.1)


def create_tiles_with_metrics() -> Path:
    """Export real tiles (with metrics) into `TMP_DIRECTORY`, e.g. as input for
    the `clean` command / `OutlierDetector`. Returns the slide's output directory.
    """
    from histoslice import export_tiles
    from histoslice.tiles import tile_regions

    slide = Slide(SLIDE_PATH_JPEG)
    regions = tile_regions(slide.dimensions, 256, overlap=0.0, out_of_bounds=False)
    output_dir = TMP_DIRECTORY / slide.name
    export_tiles(
        slide,
        regions,
        output_dir,
        tile_size=256,
        save_metrics=True,
        threshold=200,
        save_thumbnails=False,
    )
    return output_dir


def make_bad_slide_dir(name: str) -> Path:
    """A slide directory with metadata.parquet but no metric columns, so
    `OutlierDetector` raises when `clean` processes it. Returns the directory.
    """
    bad_dir = TMP_DIRECTORY / name
    bad_dir.mkdir(parents=True)
    pl.DataFrame(
        {"x": [0], "y": [0], "w": [1], "h": [1], "path": ["x.jpeg"]}
    ).write_parquet(bad_dir / "metadata.parquet")
    return bad_dir


# Optional dependency flags and asset availability
try:
    import pyvips  # noqa: F401

    HAS_PYVIPS = True
except Exception:
    HAS_PYVIPS = False

HAS_PYVIPS_ASSET = HAS_PYVIPS and SLIDE_PATH_TIFF.exists()
HAS_PYVIPS_CZI_ASSET = HAS_PYVIPS and SLIDE_PATH_CZI.exists()
HAS_PYVIPS_JPEG_ASSET = HAS_PYVIPS and SLIDE_PATH_JPEG.exists()
