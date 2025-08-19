from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Dict,
    Tuple,
    Union,
)

import cv2
import numpy as np

try:
    from aicspylibczi import CziFile

    HAS_CZI = True
    CZI_ERROR = None
except Exception as e:  # pragma: no cover - import guard
    CziFile = None  # type: ignore
    HAS_CZI = False
    CZI_ERROR = e
from PIL import Image

from histoslice.functional._level import format_level
from histoslice.functional._tiles import (
    _divide_xywh,
    _get_allowed_dimensions,
    _pad_tile,
)

try:
    import openslide

    OPENSLIDE_ERROR = None
    HAS_OPENSLIDE = True
except ImportError as error:
    openslide = None
    OPENSLIDE_ERROR = error
    HAS_OPENSLIDE = False

try:
    import pyvips

    HAS_PYVIPS = True
    PYVIPS_ERROR = None
except Exception as e:  # pragma: no cover - import guard
    pyvips = None  # type: ignore
    HAS_PYVIPS = False
    PYVIPS_ERROR = e

ERROR_PYVIPS_IMPORT = (
    "pyvips (libvips) could not be imported. Install with `pip install pyvips` "
    "and ensure libvips is available on your system."
)


Image.MAX_IMAGE_PIXELS = 20_000 * 20_000
ERROR_OPENSLIDE_IMPORT = (
    "Could not import `openslide-python`, make sure `OpenSlide` is installed "
    "(https://openslide.org/api/python/)."
)
ERROR_CZI_IMPORT = (
    "Could not import `aicspylibczi`. Install with `pip install aicspylibczi` "
    "and ensure libCZI prerequisites are available."
)
ERROR_NON_MOSAIC = "HistoPrep does not support reading non-mosaic czi-files."
BACKGROUND_COLOR = (1.0, 1.0, 1.0)


class SlideReaderBackend(ABC):
    """Base class for all backends."""

    def __init__(self, path: Union[str, Path]) -> None:
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path.resolve()))
        self.__path = path if isinstance(path, Path) else Path(path)
        self.__name = self.__path.name.removesuffix(self.__path.suffix)

    @property
    def path(self) -> str:
        """Full slide filepath."""
        return str(self.__path.resolve())

    @property
    def name(self) -> str:
        """Slide filename without an extension."""
        return self.__name

    @property
    def suffix(self) -> str:
        """Slide file-extension."""
        return self.__path.suffix

    @property
    @abstractmethod
    def reader(self):  # noqa
        pass

    @property
    @abstractmethod
    def data_bounds(self) -> tuple[int, int, int, int]:
        """Data bounds defined by `xywh`-coordinates at `level=0`."""

    @property
    @abstractmethod
    def dimensions(self) -> tuple[int, int]:
        """Image dimensions (height, width) at `level=0`."""

    @property
    @abstractmethod
    def level_count(self) -> int:
        """Number of slide pyramid levels."""

    @property
    @abstractmethod
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        """Image dimensions (height, width) for each pyramid level."""

    @property
    @abstractmethod
    def level_downsamples(self) -> dict[int, tuple[float, float]]:
        """Image downsample factors (height, width) for each pyramid level."""

    @abstractmethod
    def read_level(self, level: int) -> np.ndarray:
        """Read full pyramid level data.

        Args:
            level: Slide pyramid level.

        Raises:
            ValueError: Invalid `level` argument.

        Returns:
            Array containing image data from `level`.
        """

    @abstractmethod
    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        """Read region based on `xywh`-coordinates.

        Args:
            xywh: Coordinates for the region.
            level: Slide pyramid level to read from. Defaults to 0.

        Raises:
            ValueError: Invalid `level` argument.

        Returns:
            Array containing image data from `xywh`-region.
        """


class CziBackend(SlideReaderBackend):
    """Slide reader using `aicspylibczi.CziFile` as a backend (by Allen Institute
    for Cell Science).
    """

    MIN_LEVEL_DIMENSION = 1024
    BACKEND_NAME = "CZI"

    def __init__(self, path: str) -> None:
        """Initialize CziBackend class instance.

        Args:
            path: Path to slide image.

        Raises:
            NotImplementedError: Image is a non-mosaic czi-file.
        """
        if not HAS_CZI:
            raise ImportError(ERROR_CZI_IMPORT) from CZI_ERROR
        super().__init__(path)
        self.__reader = CziFile(path)  # type: ignore[name-defined]
        if not self.__reader.is_mosaic():
            raise NotImplementedError(ERROR_NON_MOSAIC)
        # Get plane constraints.
        bbox = self.__reader.get_mosaic_bounding_box()
        self.__origo = (bbox.x, bbox.y)
        # Define dimensions and downsamples.
        slide_h, slide_w = (bbox.h, bbox.w)
        lvl = 0
        self.__level_dimensions, self.__level_downsamples = {}, {}
        while lvl == 0 or max(slide_w, slide_h) // 2**lvl >= self.MIN_LEVEL_DIMENSION:
            level_h, level_w = (slide_h // 2**lvl, slide_w // 2**lvl)
            self.__level_dimensions[lvl] = (
                round(slide_h / 2**lvl),
                round(slide_w / 2**lvl),
            )
            self.__level_downsamples[lvl] = slide_h / level_h, slide_w / level_w
            lvl += 1

    @property
    def reader(self) -> "CziFile":  # type: ignore[name-defined]
        """CziFile instance."""
        return self.__reader

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        h, w = self.dimensions
        return (0, 0, w, h)

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.__level_dimensions[0]

    @property
    def level_count(self) -> int:
        return len(self.level_dimensions)

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        return self.__level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[int, int]]:
        return self.__level_downsamples

    def read_level(self, level: int) -> np.ndarray:
        level = format_level(level, available=list(self.level_dimensions))
        return self.read_region(xywh=self.data_bounds, level=level)

    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        level = format_level(level, available=list(self.level_dimensions))
        x, y, w, h = xywh
        # Define allowed dims, output dims and expected dims.
        allowed_h, allowed_w = _get_allowed_dimensions(xywh, dimensions=self.dimensions)
        output_h, output_w = round(h / 2**level), round(w / 2**level)
        # Read allowed reagion.
        scale_factor = 1 / 2**level
        if allowed_h * scale_factor < 1 or allowed_w * scale_factor < 1:
            # LibCzi crashes with zero size.
            return np.zeros((output_h, output_w, 3), dtype=np.uint8) + 255
        tile = self.__reader.read_mosaic(
            region=(self.__origo[0] + x, self.__origo[1] + y, allowed_w, allowed_h),
            scale_factor=scale_factor,
            C=0,
            background_color=BACKGROUND_COLOR,
        )[0]
        # Resize to match expected size (Zeiss's libCZI is buggy).
        excepted_h, excepted_w = (
            round(allowed_h / 2**level),
            round(allowed_w / 2**level),
        )
        tile_h, tile_w = tile.shape[:2]
        if tile_h != excepted_h or tile_w != excepted_w:
            tile = cv2.resize(
                tile, dsize=(excepted_w, excepted_h), interpolation=cv2.INTER_NEAREST
            )
        # Convert to RGB and pad.
        return _pad_tile(
            cv2.cvtColor(tile, cv2.COLOR_BGR2RGB), shape=(excepted_h, excepted_w)
        )


class OpenSlideBackend(SlideReaderBackend):
    """Slide reader using `OpenSlide` as a backend."""

    BACKEND_NAME = "OPENSLIDE"

    def __init__(self, path: str) -> None:
        """Initialize OpenSlideBackend class instance.

        Args:
            path: Path to the slide image.

        Raises:
            ImportError: OpenSlide could not be imported.
        """
        if not HAS_OPENSLIDE:
            raise ImportError(ERROR_OPENSLIDE_IMPORT) from OPENSLIDE_ERROR
        super().__init__(path)
        self.__reader = openslide.OpenSlide(path)
        # Openslide has (width, height) dimensions.
        self.__level_dimensions = {
            lvl: (h, w) for lvl, (w, h) in enumerate(self.__reader.level_dimensions)
        }
        # Calculate actual downsamples.
        slide_h, slide_w = self.dimensions
        self.__level_downsamples = {}
        for lvl, (level_h, level_w) in self.__level_dimensions.items():
            self.__level_downsamples[lvl] = (slide_h / level_h, slide_w / level_w)

    @property
    def reader(self) -> "openslide.OpenSlide":
        """OpenSlide instance."""
        return self.__reader

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        properties = dict(self.__reader.properties)
        x_bound = int(properties.get("openslide.bounds-x", 0))
        y_bound = int(properties.get("openslide.bounds-y", 0))
        w_bound = int(properties.get("openslide.bounds-width", self.dimensions[1]))
        h_bound = int(properties.get("openslide.bounds-heigh", self.dimensions[0]))
        return (x_bound, y_bound, w_bound, h_bound)

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.level_dimensions[0]

    @property
    def level_count(self) -> int:
        return len(self.level_dimensions)

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        return self.__level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[int, int]]:
        return self.__level_downsamples

    def read_level(self, level: int) -> np.ndarray:
        level = format_level(level, available=list(self.level_dimensions))
        level_h, level_w = self.level_dimensions[level]
        return np.array(self.__reader.get_thumbnail(size=(level_w, level_h)))

    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        level = format_level(level, available=list(self.level_dimensions))
        # Only width and height have to be adjusted for the level.
        x, y, *__ = xywh
        *__, w, h = _divide_xywh(xywh, self.level_downsamples[level])
        # Read allowed region.
        allowed_h, allowed_w = _get_allowed_dimensions((x, y, w, h), self.dimensions)
        tile = self.__reader.read_region(
            location=(x, y), level=level, size=(allowed_w, allowed_h)
        )
        tile = np.array(tile)[..., :3]  # only rgb channels
        # Pad tile.
        return _pad_tile(tile, shape=(h, w))


class PyVipsBackend(SlideReaderBackend):
    """Slide reader using `pyvips` (libvips) as a backend.

    This backend loads pyramidal Whole Slide Images (WSI) through libvips via
    pyvips. libvips exposes WSI pyramid *levels* as TIFF *pages*. We map those
    pages to `level` indices (0 == full resolution), compute level dimensions
    and level-wise downsample factors, and provide `read_level`/`read_region`
    compatible with the OpenSlide-based backend.
    """

    BACKEND_NAME = "PYVIPS"

    def __init__(self, path: str) -> None:
        """Initialize PyVipsBackend.

        Args:
            path: Path to the slide image.

        Raises:
            ImportError: pyvips could not be imported.
        """
        if not HAS_PYVIPS:
            raise ImportError(ERROR_PYVIPS_IMPORT) from PYVIPS_ERROR

        super().__init__(path)
        # Open the level-0 page lazily. pyvips is demand-driven, so this does
        # not decode full pixels until needed.
        self.__img0 = pyvips.Image.new_from_file(path, access="random", page=0)
        self.__path = path

        # libvips exposes the pyramid levels as pages.
        try:
            n_pages = int(self.__img0.get("n-pages"))
        except Exception:
            # Fallback: if metadata missing, assume single page.
            n_pages = 1

        # Build (height, width) per level. pyvips uses width/height props.
        level_dims: Dict[int, Tuple[int, int]] = {}
        for lvl in range(n_pages):
            page = pyvips.Image.new_from_file(path, access="random", page=lvl)
            # Ensure dimensions are (H, W) to match user's class contract.
            level_dims[lvl] = (int(page.height), int(page.width))
        self.__level_dimensions = level_dims

        # Calculate actual downsamples relative to level 0 (H, W).
        slide_h, slide_w = self.dimensions
        self.__level_downsamples: Dict[int, Tuple[float, float]] = {}
        for lvl, (level_h, level_w) in self.__level_dimensions.items():
            self.__level_downsamples[lvl] = (
                slide_h / float(level_h),
                slide_w / float(level_w),
            )

    # -------------------- properties --------------------
    @property
    def reader(self) -> "pyvips.Image":
        """Underlying pyvips Image corresponding to level 0."""
        return self.__img0

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        # libvips propagates OpenSlide properties if the openslide loader is used.
        # Use safe defaults when not present.
        def _get_int(meta: str, default: int) -> int:
            try:
                return int(self.__img0.get(meta))
            except Exception:
                return default

        x_bound = _get_int("openslide.bounds-x", 0)
        y_bound = _get_int("openslide.bounds-y", 0)
        # width/height defaults are whole-slide dimensions (W, H) -> convert to ints
        w_default = self.dimensions[1]
        h_default = self.dimensions[0]
        w_bound = _get_int("openslide.bounds-width", w_default)
        h_bound = _get_int("openslide.bounds-height", h_default)
        return (x_bound, y_bound, w_bound, h_bound)

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.level_dimensions[0]

    @property
    def level_count(self) -> int:
        return len(self.level_dimensions)

    @property
    def level_dimensions(self) -> Dict[int, Tuple[int, int]]:
        return self.__level_dimensions

    @property
    def level_downsamples(self) -> Dict[int, Tuple[float, float]]:
        return self.__level_downsamples

    # -------------------- reading APIs --------------------
    def _page(self, level: int) -> "pyvips.Image":
        # Helper: fetch a given level/page lazily.
        return pyvips.Image.new_from_file(self.__path, access="random", page=level)

    def read_level(self, level: int) -> np.ndarray:
        level = format_level(level, available=list(self.level_dimensions))
        page = self._page(level)
        arr = np.ndarray(
            buffer=page.write_to_memory(),
            dtype=np.uint8,
            shape=[page.height, page.width, page.bands],
        )
        # Ensure RGB
        if arr.shape[2] > 3:
            arr = arr[..., :3]
        return arr

    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        level = format_level(level, available=list(self.level_dimensions))
        # Coordinates are given in level-0 space. For libvips page-crop, we
        # convert x, y to the target-level coordinate system, while width/height
        # must also be expressed at the target level.
        x, y, *_ = xywh
        _, _, w_l, h_l = _divide_xywh(xywh, self.level_downsamples[level])

        # Convert top-left to level space.
        ds_h, ds_w = self.level_downsamples[level]
        x_l = int(x / ds_w)
        y_l = int(y / ds_h)

        # Bound the requested region against the slide bounds at level 0, but
        # expressed in level coordinates for cropping.
        allowed_h0, allowed_w0 = _get_allowed_dimensions(
            (x, y, w_l, h_l), self.dimensions
        )
        # allowed_* are sizes at the *requested level* per the upstream helper,
        # but to be safe, clamp again to the level's own dimensions.
        level_h, level_w = self.level_dimensions[level]
        allowed_w = min(int(allowed_w0), level_w - x_l)
        allowed_h = min(int(allowed_h0), level_h - y_l)
        if allowed_w < 0:
            allowed_w = 0
        if allowed_h < 0:
            allowed_h = 0

        page = self._page(level)
        # extract_area expects level-space coordinates and sizes.
        tile_img = page.extract_area(x_l, y_l, allowed_w, allowed_h)

        # Materialize to numpy
        tile = np.ndarray(
            buffer=tile_img.write_to_memory(),
            dtype=np.uint8,
            shape=[tile_img.height, tile_img.width, tile_img.bands],
        )
        if tile.shape[2] > 3:
            tile = tile[..., :3]

        # Pad to the requested (h_l, w_l) shape if we hit the edge.
        tile = _pad_tile(tile, shape=(int(h_l), int(w_l)))
        return tile


class PillowBackend(SlideReaderBackend):
    """Slide reader using `Pillow` as a backend.

    NOTE: `Pillow` reads the the whole slide into memory and thus isn't suitable for
    large images.
    """

    MIN_LEVEL_DIMENSION = 512
    BACKEND_NAME = "PILLOW"

    def __init__(self, path: str) -> None:
        """Initialize PillowBackend class instance.

        Args:
            path: Path to the slide image.
        """
        super().__init__(path)
        # Read full image.
        self.__pyramid = {0: Image.open(path)}
        # Generate downsamples.
        slide_h, slide_w = self.dimensions
        lvl = 0
        self.__level_dimensions, self.__level_downsamples = {}, {}
        while lvl == 0 or max(slide_w, slide_h) // 2**lvl >= self.MIN_LEVEL_DIMENSION:
            level_h, level_w = (slide_h // 2**lvl, slide_w // 2**lvl)
            self.__level_dimensions[lvl] = (slide_h // 2**lvl, slide_w // 2**lvl)
            self.__level_downsamples[lvl] = (slide_h / level_h, slide_w / level_w)
            lvl += 1

    @property
    def reader(self) -> None:
        """PIL image at level=0."""
        return self.__pyramid[0]

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        h, w = self.dimensions
        return (0, 0, w, h)

    @property
    def dimensions(self) -> tuple[int, int]:
        # PIL has (width, height) size.
        return self.__pyramid[0].size[::-1]

    @property
    def level_count(self) -> int:
        return len(self.level_dimensions)

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        return self.__level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[float, float]]:
        return self.__level_downsamples

    def read_level(self, level: int) -> np.ndarray:
        level = format_level(level, available=list(self.level_dimensions))
        self.__lazy_load(level)
        return np.array(self.__pyramid[level])

    def read_region(self, xywh: tuple[int, int, int, int], level: int) -> np.ndarray:
        level = format_level(level, available=list(self.level_dimensions))
        self.__lazy_load(level)
        # Read allowed region.
        x, y, output_w, output_h = _divide_xywh(xywh, self.level_downsamples[level])
        allowed_h, allowed_w = _get_allowed_dimensions(
            xywh=(x, y, output_w, output_h), dimensions=self.level_dimensions[level]
        )
        tile = np.array(
            self.__pyramid[level].crop((x, y, x + allowed_w, y + allowed_h))
        )
        # Pad tile.
        return _pad_tile(tile, shape=(output_h, output_w))

    def __lazy_load(self, level: int) -> None:
        if level not in self.__pyramid:
            height, width = self.level_dimensions[level]
            self.__pyramid[level] = self.__pyramid[0].resize(
                (width, height), resample=Image.Resampling.NEAREST
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path})"
