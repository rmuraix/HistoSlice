"""Reading/writing image files, backed by `pyvips`/`libvips`."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pyvips

ERROR_EMPTY_ARRAY = "Cannot write an empty image array."

_SAVE_METHODS = {
    "jpg": "jpegsave",
    "jpeg": "jpegsave",
    "png": "pngsave",
    "tif": "tiffsave",
    "tiff": "tiffsave",
}


def read_image(path: Union[str, Path]) -> np.ndarray:
    """Read an image file into a `(H, W)` or `(H, W, C)` `uint8` array.

    Grayscale images are kept single-channel, everything else (RGBA, CMYK,
    palette, ...) is flattened down to RGB, mirroring `PIL`'s
    `convert("L"/"RGB")` behaviour that this replaces.
    """
    image = pyvips.Image.new_from_file(str(path), access="sequential")
    return _vips_to_array(_normalise_bands(image))


def write_image(
    image: np.ndarray, path: Union[str, Path], *, image_format: str, quality: int = 80
) -> None:
    """Write a `(H, W)` or `(H, W, C)` `uint8` array to `path`.

    Args:
        image: Image array to write.
        path: Output file path.
        image_format: One of "jpg"/"jpeg", "png", "tif"/"tiff", or any other
            format `libvips` recognises from the file extension.
        quality: JPEG compression quality, ignored for other formats.
    """
    if image.size == 0:
        raise ValueError(ERROR_EMPTY_ARRAY)
    vips_image = _array_to_vips(image)
    method = _SAVE_METHODS.get(image_format.strip().lower())
    if method == "jpegsave":
        vips_image.jpegsave(str(path), Q=int(quality))
    elif method == "pngsave":
        vips_image.pngsave(str(path))
    elif method == "tiffsave":
        vips_image.tiffsave(str(path))
    else:
        vips_image.write_to_file(str(path))


def has_jpeg_support() -> bool:
    """Return True if the underlying `libvips` build can save JPEG images."""
    try:
        pyvips.Image.black(1, 1).jpegsave_buffer()
    except Exception:
        return False
    return True


def has_svg_support() -> bool:
    """Return True if the underlying `libvips` build can rasterise SVG (needs
    `librsvg`)."""
    try:
        pyvips.Image.black(1, 1).pngsave_buffer()
        pyvips.Image.svgload_buffer(
            b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>'
        )
    except Exception:
        return False
    return True


def _normalise_bands(image: "pyvips.Image") -> "pyvips.Image":
    if image.format != "uchar":
        image = image.cast("uchar")
    if image.bands == 1:
        return image
    if image.bands != 3 or image.interpretation not in ("srgb", "rgb"):
        image = image.flatten() if image.hasalpha() else image
        image = image.colourspace("srgb")
    if image.bands > 3:
        image = image.extract_band(0, n=3)
    return image


def _vips_to_array(image: "pyvips.Image") -> np.ndarray:
    buffer = image.write_to_memory()
    array = np.frombuffer(buffer, dtype=np.uint8)
    array = array.reshape(image.height, image.width, image.bands)
    return array[:, :, 0] if image.bands == 1 else array


def _array_to_vips(image: np.ndarray) -> "pyvips.Image":
    if image.ndim == 2:
        image = image[:, :, None]
    return pyvips.Image.new_from_array(image)
