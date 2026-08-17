"""Whole slide image I/O.

`Slide` is responsible for exactly one thing: reading pixel data from a
whole slide image file via `pyvips`/`libvips`. It knows nothing about
tissue detection, tile grids, or saving files to disk - see
`histoslice.tissue`, `histoslice.tiles` and `histoslice.export`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import pyvips

from histoslice.tiles import Region, pad_to_shape

ERROR_LEVEL = "Level {} could not be found, select from {}."

MIN_LEVEL_DIMENSION = 512


class Slide:
    """Read pixel data from a whole slide image."""

    def __init__(
        self, path: Union[str, Path], *, mpp: Optional[tuple[float, float]] = None
    ) -> None:
        """Open a whole slide image.

        Args:
            path: Path to the slide image.
            mpp: Override microns per pixel as `(mpp_x, mpp_y)`. If None, the value is
                extracted from slide metadata. Defaults to None.

        Raises:
            FileNotFoundError: `path` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path.resolve()))
        self._path = path
        self._mpp_override = mpp
        try:
            self._page0 = pyvips.Image.new_from_file(str(path), access="random", page=0)
        except Exception:
            self._page0 = pyvips.Image.new_from_file(str(path), access="random")

        try:
            n_pages = int(self._page0.get("n-pages"))
        except Exception:
            n_pages = 1

        level_dims: dict[int, tuple[int, int]] = {}
        if n_pages > 1:
            for level in range(n_pages):
                page = pyvips.Image.new_from_file(
                    str(path), access="random", page=level
                )
                level_dims[level] = (int(page.height), int(page.width))
            self._synthetic_levels = False
        else:
            slide_h, slide_w = int(self._page0.height), int(self._page0.width)
            level = 0
            while (
                level == 0 or max(slide_w, slide_h) // 2**level >= MIN_LEVEL_DIMENSION
            ):
                level_dims[level] = (slide_h // 2**level, slide_w // 2**level)
                level += 1
            self._synthetic_levels = True

        self._level_dimensions = level_dims
        slide_h, slide_w = self._level_dimensions[0]
        self._level_downsamples = {
            level: (slide_h / h, slide_w / w) for level, (h, w) in level_dims.items()
        }

    @property
    def path(self) -> str:
        """Full slide filepath."""
        return str(self._path.resolve())

    @property
    def name(self) -> str:
        """Slide filename without an extension."""
        return self._path.name.removesuffix(self._path.suffix)

    @property
    def suffix(self) -> str:
        """Slide file extension."""
        return self._path.suffix

    @property
    def dimensions(self) -> tuple[int, int]:
        """Image dimensions `(height, width)` at level 0."""
        return self._level_dimensions[0]

    @property
    def level_count(self) -> int:
        """Number of pyramid levels."""
        return len(self._level_dimensions)

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        """Image dimensions `(height, width)` for each pyramid level."""
        return self._level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[float, float]]:
        """Downsample factor `(height, width)` for each pyramid level."""
        return self._level_downsamples

    @property
    def mpp(self) -> Optional[tuple[float, float]]:
        """Microns per pixel `(mpp_x, mpp_y)` at level 0, or None if unavailable.

        Returns the constructor override if one was given, otherwise reads slide
        metadata.
        """
        if self._mpp_override is not None:
            return self._mpp_override
        return self._read_mpp_from_metadata()

    def level_from_max_dimension(self, max_dimension: int = 4096) -> int:
        """Find the pyramid level with both dimensions <= `max_dimension`.

        If no such level exists, the last (smallest) level is returned.
        """
        for level, (h, w) in self.level_dimensions.items():
            if h <= max_dimension and w <= max_dimension:
                return level
        return list(self.level_dimensions)[-1]

    def level_from_dimensions(self, dimensions: tuple[int, int]) -> int:
        """Find the pyramid level closest to `dimensions` (`height, width`)."""
        height, width = dimensions
        levels = list(self.level_dimensions)
        distances = [
            abs(h - height) + abs(w - width) for h, w in self.level_dimensions.values()
        ]
        return levels[distances.index(min(distances))]

    def read_level(self, level: int) -> np.ndarray:
        """Read a full pyramid level.

        Args:
            level: Pyramid level to read.

        Raises:
            ValueError: Invalid `level`.

        Returns:
            RGB image array for `level`.
        """
        level = self._format_level(level)
        page = self._page(level)
        return _to_array(page)

    def read_region(self, region: Region, level: int = 0) -> np.ndarray:
        """Read `region` (level-0 coordinates), downsampled to `level`.

        The output array is sized `(round(region.height / downsample_h),
        round(region.width / downsample_w), 3)`; areas outside the slide are padded
        with white pixels.

        Args:
            region: Crop area in level-0 coordinates.
            level: Pyramid level to read from. Defaults to 0.

        Raises:
            ValueError: Invalid `level`.

        Returns:
            RGB image array for `region`.
        """
        level = self._format_level(level)
        ds_h, ds_w = self.level_downsamples[level]
        x_l, y_l = int(region.x / ds_w), int(region.y / ds_h)
        w_l, h_l = round(region.width / ds_w), round(region.height / ds_h)

        level_h, level_w = self.level_dimensions[level]
        allowed_w = max(min(w_l, level_w - x_l), 0)
        allowed_h = max(min(h_l, level_h - y_l), 0)
        if allowed_w == 0 or allowed_h == 0:
            return np.zeros((h_l, w_l, 3), dtype=np.uint8) + 255

        page = self._page(level)
        cropped = page.extract_area(x_l, y_l, allowed_w, allowed_h)
        tile = _to_array(cropped)
        return pad_to_shape(tile, shape=(h_l, w_l), fill=255)

    def read_tile(self, region: Region, size: tuple[int, int]) -> np.ndarray:
        """Read `region` and resize to exactly `size` (`width, height`) pixels.

        Automatically selects the most efficient pyramid level whose resolution is
        sufficient to produce `size` without upsampling, then resizes precisely to
        `size`. This is the building block for resolution-normalized (`target_mpp`)
        tile extraction.

        Args:
            region: Crop area in level-0 coordinates.
            size: Desired output size as `(width, height)`.

        Returns:
            RGB image array of shape `(size[1], size[0], 3)`.
        """
        out_w, out_h = size
        scale_w = region.width / out_w if out_w else 1.0
        scale_h = region.height / out_h if out_h else 1.0
        level = self._level_for_scale(max(scale_w, scale_h))
        tile = self.read_region(region, level=level)
        if tile.shape[:2] == (out_h, out_w):
            return tile
        interpolation = (
            cv2.INTER_AREA if max(scale_w, scale_h) >= 1 else cv2.INTER_CUBIC
        )
        return cv2.resize(tile, (out_w, out_h), interpolation=interpolation)

    def _level_for_scale(self, scale: float) -> int:
        """Largest level whose downsample does not exceed `scale`."""
        best = 0
        for level, (ds_h, ds_w) in self.level_downsamples.items():
            if max(ds_h, ds_w) <= scale + 1e-6:
                best = level
        return best

    def _format_level(self, level: int) -> int:
        available = list(self.level_dimensions)
        if level < 0:
            if abs(level) > len(available):
                raise ValueError(ERROR_LEVEL.format(level, available))
            return available[level]
        if level in available:
            return level
        raise ValueError(ERROR_LEVEL.format(level, available))

    def _page(self, level: int) -> "pyvips.Image":
        if not self._synthetic_levels:
            return pyvips.Image.new_from_file(
                str(self._path), access="random", page=level
            )
        if level == 0:
            return self._page0
        target_h, target_w = self.level_dimensions[level]
        scale_w = target_w / self._page0.width
        scale_h = target_h / self._page0.height
        page = self._page0.resize(scale_w, vscale=scale_h, kernel="nearest")
        if page.width == target_w and page.height == target_h:
            return page
        if page.width >= target_w and page.height >= target_h:
            return page.extract_area(0, 0, target_w, target_h)
        return page.embed(
            0, 0, target_w, target_h, extend="white", background=[255, 255, 255]
        )

    def _read_mpp_from_metadata(self) -> Optional[tuple[float, float]]:
        try:
            mpp_x = float(self._page0.get("openslide.mpp-x"))
            mpp_y = float(self._page0.get("openslide.mpp-y"))
            return (mpp_x, mpp_y)
        except Exception:
            pass
        try:
            xres = float(self._page0.get("xres"))
            yres = float(self._page0.get("yres"))
            try:
                unit_val = self._page0.get("resolution-unit")
                unit = int(unit_val) if unit_val is not None else None
            except Exception:
                unit = None
            if xres <= 0 or yres <= 0:
                return None
            # xres/yres are in pixels per unit; convert to microns per pixel.
            um_per_unit = {1: 1000.0, 2: 25400.0, 3: 10000.0}.get(
                1 if unit is None else unit
            )
            if um_per_unit is None:
                return None
            return (um_per_unit / xres, um_per_unit / yres)
        except Exception:
            return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path})"


def _to_array(image: "pyvips.Image") -> np.ndarray:
    arr = np.ndarray(
        buffer=image.write_to_memory(),
        dtype=np.uint8,
        shape=[image.height, image.width, image.bands],
    )
    if arr.shape[2] > 3:  # noqa
        arr = arr[..., :3]
    return arr
