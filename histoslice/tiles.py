"""Tile region generation and filtering.

Convention used throughout this module (and the rest of `histoslice`):
    * `dimensions` (whole slide / pyramid level shapes) are `(height, width)`,
      matching `numpy.ndarray.shape`.
    * `size` (a single tile/region size) is `(width, height)`, matching the
      `PIL`/`OpenCV` convention. `Region` fields follow the same `x, y, width,
      height` order.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

ERROR_NONZERO = "Tile width and height should be non-zero positive integers, got {}."
ERROR_DIMENSION = "Tile size {} should be smaller than image dimensions {}."
ERROR_OVERLAP = "Overlap should be in range [0, 1), got {}."

OVERLAP_LIMIT = 1.0


@dataclass(frozen=True)
class Region:
    """A rectangular crop area, defined in the level-0 coordinate system."""

    x: int
    y: int
    width: int
    height: int

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


def _as_size(size: Union[int, tuple[int, int]]) -> tuple[int, int]:
    """Normalize a size argument to `(width, height)`."""
    if isinstance(size, int):
        return (size, size)
    return (int(size[0]), int(size[1]))


def tile_regions(
    dimensions: tuple[int, int],
    size: Union[int, tuple[int, int]],
    *,
    overlap: float = 0.0,
    out_of_bounds: bool = True,
) -> list[Region]:
    """Generate a grid of tile regions covering `dimensions`.

    Args:
        dimensions: Slide (or level) dimensions as `(height, width)`.
        size: Tile size as `(width, height)`, or a single int for square tiles.
        overlap: Overlap between neighbouring tiles, in range [0, 1). Defaults to 0.0.
        out_of_bounds: Keep tiles which extend beyond `dimensions`. Defaults to True.

    Raises:
        ValueError: Tile size is non-positive, larger than `dimensions`, or overlap
            is out of range.

    Returns:
        List of `Region` instances covering the slide.

    Example:
        >>> tile_regions((16, 8), size=8, overlap=0.5, out_of_bounds=False)
        [Region(x=0, y=0, width=8, height=8), Region(x=0, y=4, width=8, height=8), Region(x=0, y=8, width=8, height=8)]
    """
    width, height = _as_size(size)
    if height <= 0 or width <= 0:
        raise ValueError(ERROR_NONZERO.format((width, height)))
    slide_height, slide_width = dimensions
    if height > slide_height or width > slide_width:
        raise ValueError(ERROR_DIMENSION.format((width, height), dimensions))
    if not 0 <= overlap < OVERLAP_LIMIT:
        raise ValueError(ERROR_OVERLAP.format(overlap))
    width_step = max(width - round(width * overlap), 1)
    height_step = max(height - round(height * overlap), 1)
    x_coords = list(range(0, slide_width, width_step))
    y_coords = list(range(0, slide_height, height_step))
    if not out_of_bounds and x_coords and x_coords[-1] + width > slide_width:
        x_coords = x_coords[:-1]
    if not out_of_bounds and y_coords and y_coords[-1] + height > slide_height:
        y_coords = y_coords[:-1]
    return [
        Region(x, y, width, height) for y, x in itertools.product(y_coords, x_coords)
    ]


def get_downsample(
    mask: np.ndarray, dimensions: tuple[int, int]
) -> tuple[float, float]:
    """Height/width downsample factor between `dimensions` and `mask`.

    Example:
        >>> get_downsample(np.zeros((8, 8)), dimensions=(128, 128))
        (16.0, 16.0)
    """
    mask_h, mask_w = mask.shape[:2]
    slide_h, slide_w = dimensions
    return (slide_h / mask_h, slide_w / mask_w)


def region_from_array(
    image: np.ndarray,
    region: Region,
    *,
    downsample: tuple[float, float] = (1.0, 1.0),
    fill: int = 0,
) -> np.ndarray:
    """Read `region` from an array, downsampling coordinates and padding out-of-bounds."""
    ds_h, ds_w = downsample
    x = round(region.x / ds_w)
    y = round(region.y / ds_h)
    out_w = round(region.width / ds_w)
    out_h = round(region.height / ds_h)
    image_h, image_w = image.shape[:2]
    allowed_w = max(min(out_w, image_w - x), 0)
    allowed_h = max(min(out_h, image_h - y), 0)
    cropped = image[y : y + allowed_h, x : x + allowed_w]
    return _pad(cropped, shape=(out_h, out_w), fill=fill)


def background_percentages(
    regions: list[Region], tissue_mask: np.ndarray, downsample: tuple[float, float]
) -> list[float]:
    """Fraction of background (non-tissue) pixels for each region."""
    output = []
    for region in regions:
        tile_mask = region_from_array(
            tissue_mask, region, downsample=downsample, fill=0
        )
        output.append((tile_mask == 0).sum() / tile_mask.size)
    return output


def filter_by_tissue(
    regions: list[Region],
    tissue_mask: np.ndarray,
    *,
    slide_dimensions: tuple[int, int],
    max_background: float = 0.5,
) -> list[Region]:
    """Keep only regions whose background fraction is at most `max_background`.

    Args:
        regions: Regions to filter.
        tissue_mask: Tissue mask (0=background, 1=tissue), typically read from a
            downsampled pyramid level.
        slide_dimensions: Full slide dimensions `(height, width)` at level 0,
            used to compute the mask's downsample factor.
        max_background: Maximum allowed background fraction. Defaults to 0.5.

    Returns:
        Filtered list of regions.
    """
    downsample = get_downsample(tissue_mask, slide_dimensions)
    backgrounds = background_percentages(regions, tissue_mask, downsample)
    return [r for r, bg in zip(regions, backgrounds) if bg <= max_background]


def level0_tile_size(
    size: tuple[int, int],
    *,
    slide_mpp: tuple[float, float],
    target_mpp: tuple[float, float],
) -> tuple[int, int]:
    """Level-0 crop size needed to produce `size` output pixels at `target_mpp`.

    Handles anisotropic pixel sizes (`mpp_x != mpp_y`) by scaling each axis
    independently.

    Example:
        >>> level0_tile_size((512, 512), slide_mpp=(0.25, 0.25), target_mpp=(0.5, 0.5))
        (1024, 1024)
    """
    out_w, out_h = size
    slide_mpp_x, slide_mpp_y = slide_mpp
    target_mpp_x, target_mpp_y = target_mpp
    crop_w = round(out_w * target_mpp_x / slide_mpp_x)
    crop_h = round(out_h * target_mpp_y / slide_mpp_y)
    return (crop_w, crop_h)


ERROR_NO_MPP = (
    "A target mpp was requested but the slide's mpp is not available. "
    "Pass mpp=(mpp_x, mpp_y) explicitly, or omit the target mpp."
)


@dataclass(frozen=True)
class TileSpec:
    """Contract for a resolution-normalized output tile.

    `size` is the *final output* pixel size (`width, height`). `mpp` is the
    target physical resolution (`mpp_x, mpp_y`); use None to read tiles at the
    slide's native resolution (`size` level-0 pixels, no resampling).
    """

    size: tuple[int, int]
    mpp: Optional[tuple[float, float]] = None

    def level0_size(self, slide_mpp: Optional[tuple[float, float]]) -> tuple[int, int]:
        """Level-0 crop size needed to satisfy this spec, given the slide's mpp.

        Raises:
            ValueError: `self.mpp` is set but `slide_mpp` is None.
        """
        if self.mpp is None:
            return self.size
        if slide_mpp is None:
            raise ValueError(ERROR_NO_MPP)
        return level0_tile_size(self.size, slide_mpp=slide_mpp, target_mpp=self.mpp)


def spot_regions(
    tissue_mask: np.ndarray,
    slide_dimensions: tuple[int, int],
    *,
    min_area_pixel: int = 10,
    max_area_pixel: Optional[int] = None,
    min_area_relative: float = 0.2,
    max_area_relative: Optional[float] = 2.0,
) -> tuple[list[Region], list[str]]:
    """Detect tissue microarray (TMA) spot regions from a tissue mask.

    Args:
        tissue_mask: Tissue mask of the (TMA) slide. It's recommended to increase
            `sigma` when detecting tissue, to remove non-TMA artifacts from the mask.
        slide_dimensions: Full slide dimensions `(height, width)` at level 0.
        min_area_pixel: Minimum pixel area for contours. Defaults to 10.
        max_area_pixel: Maximum pixel area for contours. Defaults to None.
        min_area_relative: Relative minimum contour area, calculated from the median
            contour area after filtering contours with `[min,max]_pixel` arguments.
            Defaults to 0.2.
        max_area_relative: Relative maximum contour area, calculated from the median
            contour area after filtering contours with `[min,max]_pixel` arguments.
            Defaults to 2.0.

    Returns:
        Tuple of (spot regions, spot names), upsampled to level-0 coordinates.
    """
    from histoslice.functional._dearray import get_spot_coordinates
    from histoslice.tissue import clean_tissue_mask

    cleaned = clean_tissue_mask(
        tissue_mask,
        min_area_pixel=min_area_pixel,
        max_area_pixel=max_area_pixel,
        min_area_relative=min_area_relative,
        max_area_relative=max_area_relative,
    )
    spot_info = get_spot_coordinates(cleaned)
    downsample = get_downsample(tissue_mask, slide_dimensions)
    ds_h, ds_w = downsample
    regions, names = [], []
    for name, (x, y, w, h) in spot_info.items():
        regions.append(
            Region(round(x * ds_w), round(y * ds_h), round(w * ds_w), round(h * ds_h))
        )
        names.append(name)
    return regions, names


def _pad(tile: np.ndarray, *, shape: tuple[int, int], fill: int) -> np.ndarray:
    """Pad or crop `tile` to exactly `shape`, filling new pixels with `fill`."""
    tile_h, tile_w = tile.shape[:2]
    out_h, out_w = shape
    if tile_h == out_h and tile_w == out_w:
        return tile
    if tile_h > out_h or tile_w > out_w:
        return tile[:out_h, :out_w]
    out_shape = (out_h, out_w, tile.shape[-1]) if tile.ndim > 2 else (out_h, out_w)  # noqa
    output = np.zeros(out_shape, dtype=np.uint8) + fill
    output[:tile_h, :tile_w] = tile
    return output
