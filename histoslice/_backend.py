from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Dict,
    Tuple,
    Union,
)

import cv2
import numpy as np

from histoslice.functional._level import format_level
from histoslice.functional._tiles import (
    _divide_xywh,
    _get_allowed_dimensions,
    _pad_tile,
)

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
        self.__path = path
        
        # Try to detect if this is a multi-page/pyramidal image
        try:
            self.__img0 = pyvips.Image.new_from_file(path, access="random", page=0)
            self.__is_pyramidal = True
        except Exception:
            # Fallback for single-page formats (JPEG, PNG, etc.)
            self.__img0 = pyvips.Image.new_from_file(path, access="random")
            self.__is_pyramidal = False

        # libvips exposes the pyramid levels as pages.
        if self.__is_pyramidal:
            try:
                n_pages = int(self.__img0.get("n-pages"))
            except Exception:
                # Fallback: if metadata missing, assume single page.
                n_pages = 1
        else:
            # Single-page images have only one level
            n_pages = 1

        # Build (height, width) per level. pyvips uses width/height props.
        level_dims: Dict[int, Tuple[int, int]] = {}
        for lvl in range(n_pages):
            if self.__is_pyramidal:
                page = pyvips.Image.new_from_file(path, access="random", page=lvl)
            else:
                # For single-page images, we only have level 0
                page = self.__img0
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
        if self.__is_pyramidal:
            return pyvips.Image.new_from_file(self.__path, access="random", page=level)
        else:
            # For single-page images, always return the same image
            return self.__img0

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

        # Handle zero-sized regions: pyvips extract_area doesn't support zero width/height
        if allowed_w == 0 or allowed_h == 0:
            # Return an empty array with the requested shape
            tile = np.zeros((int(h_l), int(w_l), 3), dtype=np.uint8) + 255
            return tile

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


