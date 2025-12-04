import shutil
from pathlib import Path

import importlib.util

from histoslice import SlideReader

DATA_DIRECTORY = Path(__file__).parent / "data"
TMP_DIRECTORY = DATA_DIRECTORY.parent / "tmp"
SLIDE_PATH_JPEG = DATA_DIRECTORY / "slide.jpeg"
SLIDE_PATH_TIFF = DATA_DIRECTORY / "slide.tiff"
SLIDE_PATH_SVS = DATA_DIRECTORY / "slide.svs"
SLIDE_PATH_TMA = DATA_DIRECTORY / "tma_spots.jpeg"

IMAGE = SlideReader(SLIDE_PATH_JPEG).read_level(-1)[:500, :500, :]


def clean_temporary_directory() -> None:
    if TMP_DIRECTORY.exists():
        shutil.rmtree(TMP_DIRECTORY)


# Optional dependency flags and asset availability
HAS_PYVIPS = importlib.util.find_spec("pyvips") is not None

HAS_PYVIPS_ASSET = HAS_PYVIPS and SLIDE_PATH_TIFF.exists()
