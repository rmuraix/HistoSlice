import multiprocessing as mp
import random
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

from ._concurrent import DEFAULT_START_METHOD


def get_random_image_collage(
    paths: Iterable[Union[str, Path]],
    num_rows: int = 4,
    num_cols: int = 16,
    shape: tuple[int, int] = (64, 64),
    num_workers: int = 1,
) -> Image.Image:
    """Create image collage from randomly sampled images from paths.

    Args:
        paths: Image paths.
        num_rows: Number of rows in the collage. Ignored if there isn't enough images to
            fill enough rows. Defaults to 4.
        num_cols: Number of columns in the collage. Defaults to 16.
        shape: Shape of each image in the collage. Defaults to (64, 64).
        num_workers: Number of image loading workers. Defaults to 1.

    Returns:
        Image collage.
    """
    if len(paths) > num_cols * num_rows:
        paths = random.choices(paths, k=num_cols * num_rows)  # noqa
    images = read_images_from_paths(paths=paths, num_workers=num_workers)
    return create_image_collage(images=images, num_cols=num_cols, shape=shape)


def read_images_from_paths(
    paths: Iterable[Union[str, Path, None]], num_workers: int
) -> list[np.ndarray]:
    """Read images from paths.

    Args:
        paths: Image paths.
        num_workers: Number of image loading workers.

    Returns:
        List of numpy array images.
    """
    if num_workers <= 1:
        return [_read_image(x) for x in paths]

    ctx = mp.get_context(DEFAULT_START_METHOD)
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
        output = list(pool.map(_read_image, paths))
    return output  # noqa


def create_image_collage(
    images: list[np.ndarray], num_cols: int, shape: tuple[int, int]
) -> Image.Image:
    """Collect images into a collage.

    Args:
        images: List of array images.
        num_cols: Number of columns. Number of rows is determined by
            `ceil(images/num_cols)`.
        shape: Shape for each image.

    Returns:
        Image collage.
    """
    if len(images) == 0:
        return None
    output, row = [], []
    for img in images:
        resized = cv2.resize(img, dsize=shape[::-1])
        row.append(resized)
        if len(row) == num_cols:
            output.append(np.hstack(row))
            row = []
    if len(row) > 0:
        row.extend([np.zeros_like(resized)] * (num_cols - len(row)))
        output.append(np.hstack(row))
    return Image.fromarray(np.vstack(output))


def has_jpeg_support() -> bool:
    """Return True if Pillow has JPEG support enabled."""
    try:
        exts = Image.registered_extensions()
    except Exception:
        return False
    return exts.get(".jpg") == "JPEG" or exts.get(".jpeg") == "JPEG"


def _read_image(path: Optional[str]) -> np.ndarray:
    """Parallisable."""
    if path is None:
        return None
    return np.array(Image.open(path))


def downscale_to_max_pixels(
    image: np.ndarray,
    *,
    max_pixels: int,
    ensure_uint8: bool = True,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Downscale an image uniformly so that total pixels <= max_pixels.

    This is a shared utility function used by specific downscaling functions
    (e.g., for thresholding or thumbnail generation).

    Args:
        image: Input image array (H, W[, C]).
        max_pixels: Maximum allowed total number of pixels.
        ensure_uint8: Convert to uint8 before resizing (recommended for OpenCV).
        interpolation: Interpolation method (default: cv2.INTER_AREA, good for downscaling).

    Returns:
        Downscaled image if needed, otherwise the original image.
    """
    # Return immediately if the image is empty
    if image.size == 0:
        return image

    h, w = image.shape[:2]
    total = h * w

    # Skip resizing if the total number of pixels is already below the limit
    if total <= max_pixels:
        return image

    # Compute uniform scale factor so that total pixels <= max_pixels
    scale = float(np.sqrt(max_pixels / float(total)))

    # Ensure uint8 type if required (some OpenCV operations assume this)
    src = image
    if ensure_uint8 and src.dtype != np.uint8:
        src = src.astype(np.uint8, copy=False)

    # Ensure the array is C-contiguous for cv2.resize
    if not src.flags.c_contiguous:
        src = np.ascontiguousarray(src)

    # Compute new dimensions (at least 1 pixel in each dimension)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return cv2.resize(src, (new_w, new_h), interpolation=interpolation)
