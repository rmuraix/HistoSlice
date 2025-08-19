import shutil
from pathlib import Path

import importlib.util

from histoslice import SlideReader

DATA_DIRECTORY = Path(__file__).parent / "data"
TMP_DIRECTORY = DATA_DIRECTORY.parent / "tmp"
SLIDE_PATH_JPEG = DATA_DIRECTORY / "slide.jpeg"
SLIDE_PATH_SVS = DATA_DIRECTORY / "slide.svs"
SLIDE_PATH_CZI = DATA_DIRECTORY / "slide.czi"
SLIDE_PATH_TMA = DATA_DIRECTORY / "tma_spots.jpeg"

IMAGE = SlideReader(SLIDE_PATH_JPEG).read_level(-1)[:500, :500, :]


def clean_temporary_directory() -> None:
    if TMP_DIRECTORY.exists():
        shutil.rmtree(TMP_DIRECTORY)


# Optional dependency flags and asset availability
HAS_CZI = importlib.util.find_spec("aicspylibczi") is not None
HAS_OPENSLIDE = importlib.util.find_spec("openslide") is not None

HAS_CZI_ASSET = HAS_CZI and SLIDE_PATH_CZI.exists()
HAS_OPENSLIDE_ASSET = HAS_OPENSLIDE and SLIDE_PATH_SVS.exists()
