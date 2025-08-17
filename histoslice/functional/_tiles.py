import itertools
from typing import Optional, Union

import numpy as np

ERROR_TYPE = "Tile width and height should be integers, got {} and {}."
ERROR_NONZERO = (
    "Tile width and height should non-zero positive integers, got {} and {}."
)
ERROR_DIMENSION = (
    "Tile width ({}) and height ({}) should be smaller than image dimensions {}."
)
ERROR_OVERLAP = "Overlap should be in range [0, 1), got {}."

OVERLAP_LIMIT = 1.0
XYWH = tuple[int, int, int, int]


def get_tile_coordinates(
    dimensions: tuple[int, int],
    width: int,
    *,
    height: Optional[int] = None,
    overlap: float = 0.0,
    out_of_bounds: bool = False,
) -> list[XYWH]:
    """Create tile coordinates (xywh).

    Args:
        dimensions: Image dimensions (height, width).
        width: Tile width.
        height: Tile height. If None, will be set to `width`. Defaults to None.
        overlap: Overlap between neighbouring tiles. Defaults to 0.0.
        out_of_bounds: Allow tiles to go out of image bounds. Defaults to False.

    Raises:
        TypeError: Height and/or width are not integers.
        ValueError: Height and/or width are zero or less.
        ValueError: Height and/or width are larger than dimensions.
        ValueError: Overlap is not in range [0, 1).

    Returns:
        List of xywh-coordinates.

    Example:
        >>> get_tile_coordinates((16, 8), width=8, overlap=0.5)
        [(0, 0, 8, 8), (0, 4, 8, 8), (0, 8, 8, 8)]
    """
    # Check arguments.
    if height is None:
        height = width
    if not isinstance(height, int) or not isinstance(width, int):
        raise TypeError(ERROR_TYPE.format(width, height))
    if height <= 0 or width <= 0:
        raise ValueError(ERROR_NONZERO.format(width, height))
    if height > dimensions[0] or width > dimensions[1]:
        raise ValueError(ERROR_DIMENSION.format(width, height, dimensions))
    if not 0 <= overlap < OVERLAP_LIMIT:
        raise ValueError(ERROR_OVERLAP.format(overlap))
    # Collect xy-coordinates.
    level_height, level_width = dimensions
    width_step = max(width - round(width * overlap), 1)
    height_step = max(height - round(height * overlap), 1)
    x_coords = range(0, level_width, width_step)
    y_coords = range(0, level_height, height_step)
    # Filter out of bounds coordinates.
    if not out_of_bounds and max(x_coords) + width > level_width:
        x_coords = x_coords[:-1]
    if not out_of_bounds and max(y_coords) + height > level_height:
        y_coords = y_coords[:-1]
    # Take product and add width and height.
    return [(x, y, width, height) for y, x in itertools.product(y_coords, x_coords)]


def get_region_from_array(
    image: np.ndarray,
    xywh: XYWH,
    downsample: Union[float, tuple[float, float]] = 1,
    fill: int = 0,
) -> np.ndarray:
    """Read region from an image array.

    Args:
        image: Input image
        xywh: Region coordinates.
        downsample: Downsample for coordinates. Defaults to 1.
        fill: Fill value for out of bounds areas. Defaults to 0.

    Returns:
        Image data from xywh-region.
    """
    # Downsample xywh.
    xywh_d = _divide_xywh(xywh, downsample)
    # Read allowed region and pad.
    x, y, output_w, output_h = xywh_d
    allowed_h, allowed_w = _get_allowed_dimensions(xywh_d, dimensions=image.shape[:2])
    return _pad_tile(
        tile=image[y : y + allowed_h, x : x + allowed_w],
        shape=(output_h, output_w),
        fill=fill,
    )


def get_background_percentages(
    tile_coordinates: list[XYWH],
    tissue_mask: np.ndarray,
    downsample: Union[float, tuple[float, float]],
) -> list[float]:
    """Calculate background percentages for tile coordinates.

    Args:
        tile_coordinates: List of xywh-coordinates.
        tissue_mask: Tissue mask.
        downsample: Downsample of the tissue mask.

    Returns:
        List of background percentages for each tile.
    """
    output = []
    for xywh in tile_coordinates:
        tile_mask = get_region_from_array(
            tissue_mask, xywh=xywh, downsample=downsample, fill=0
        )
        output.append((tile_mask == 0).sum() / tile_mask.size)
    return output


def get_overlap_index(xywh: XYWH, coordinates: list[XYWH]) -> np.ndarray:
    """Indices of tiles in `coordinates` which overlap with `xywh`.

    Args:
        xywh: Coordinates.
        coordinates: List of tile coordinates.

    Returns:
        Indices of tiles which overlap with xywh.
    """
    x, y, w, h = xywh
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates, dtype=int)
    return np.argwhere(
        (coordinates[:, 0] < x + w)
        & (coordinates[:, 0] + coordinates[:, 2] > x)
        & (coordinates[:, 1] < y + h)
        & (coordinates[:, 1] + coordinates[:, 3] > y)
    ).flatten()


def get_overlap_area(
    xywh: tuple[int, int, int, int],
    coordinates: list[XYWH],
) -> np.ndarray:
    """Calculate how much each coordinate overlaps with `xywh`.

    Args:
        xywh: Coordinates.
        coordinates: List of coordinates.

    Returns:
        Overlapping area for each tile in `coordinates`
    """
    x, y, w, h = xywh
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    x_other, y_other, w_other, h_other = (coordinates[:, i] for i in range(4))
    # Exclude non overlapping.
    exclude = np.array([True] * len(coordinates), dtype=bool)
    exclude[get_overlap_index(xywh, coordinates)] = False
    # Caluculate overlap.
    x_overlap = np.maximum(np.minimum(x_other + w_other - x, w), 0)
    y_overlap = np.maximum(np.minimum(y_other + h_other - y, h), 0)
    areas = x_overlap * y_overlap
    areas[exclude] = 0
    return areas


def get_downsample(
    image: np.ndarray, dimensions: tuple[int, int]
) -> tuple[float, float]:
    """Calculate height and width dowmsaple between image and dimensions.

    Args:
        image: Input image.
        dimensions: Original dimensions.

    Returns:
        Height and width dowmsample.

    Example:
        >>> image = np.zeros((8, 8, 3))
        >>> get_downsample(image, dimensions=(128, 128))
        (16.0, 16.0)
    """
    return tuple([a / b for a, b in zip(dimensions, image.shape[:2])])


def _get_allowed_dimensions(
    xywh: tuple[int, int, int, int], dimensions: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Get height and width for `xywh` which are inside `dimensions`."""
    x, y, w, h = xywh
    height, width = dimensions
    if y > height or x > width:
        # x or y is outside of dimensions.
        return (0, 0)
    if y + h > height:
        h = height - y
    if x + w > width:
        w = width - x
    return h, w


def _divide_xywh(
    xywh: tuple[int, int, int, int], divisor: Union[float, tuple[float, float]]
) -> tuple[int, int, int, int]:
    """Divide xywh-coordinates with a divisor."""
    if not isinstance(divisor, (tuple, list)):
        divisor = (divisor, divisor)
    w_div, h_div = divisor
    x, y, w, h = xywh
    return round(x / w_div), round(y / h_div), round(w / w_div), round(h / h_div)


def _multiply_xywh(
    xywh: tuple[int, int, int, int], multiplier: Union[float, tuple[float, float]]
) -> tuple[int, int, int, int]:
    """Divide xywh-coordinates with divisor(s)."""
    if not isinstance(multiplier, (tuple, list)):
        multiplier = (multiplier, multiplier)
    w_mult, h_mult = multiplier
    x, y, w, h = xywh
    return round(x * w_mult), round(y * h_mult), round(w * w_mult), round(h * h_mult)


def _pad_tile(
    tile: np.ndarray, *, shape: tuple[int, int], fill: int = 255
) -> np.ndarray:
    """Pad tile image into shape with `fill` values.

    Args:
        tile: Tile image.
        shape: Output shape.
        fill: Fill value. Defaults to 255.

    Returns:
        Tile image padded into shape.
    """
    tile_h, tile_w = tile.shape[:2]
    out_h, out_w = shape
    if tile_h == out_h and tile_w == out_w:
        return tile
    if tile_h > out_h or tile_w > out_w:
        return tile[:out_h, :out_w]
    if tile.ndim > 2:  # noqa
        shape = (out_h, out_w, tile.shape[-1])
    output = np.zeros(shape, dtype=np.uint8) + fill
    output[:tile_h, :tile_w] = tile
    return output
