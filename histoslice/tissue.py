"""Tissue detection: turning a slide image into a tissue/background mask."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from histoslice.functional._check import check_image
from histoslice.functional._images import downscale_to_max_pixels

ERROR_THRESHOLD = "Threshold should be in range [0, 255], got {}."

MAX_THRESHOLD = 255
WHITE_PIXEL = 255
BLACK_PIXEL = 0
SIGMA_NO_OP = 0.0
GRAY_NDIM = 2


def tissue_mask(
    image: np.ndarray,
    *,
    threshold: Optional[int] = None,
    multiplier: float = 1.05,
    sigma: float = 1.0,
) -> tuple[int, np.ndarray]:
    """Detect tissue from an image.

    Args:
        image: Input image.
        threshold: Threshold for tissue detection. If set, tissue is detected by
            global thresholding, otherwise Otsu's method is used to find a
            threshold. Defaults to None.
        multiplier: Otsu's method finds an optimal threshold by minimizing the
            weighted within-class variance. This threshold is then multiplied with
            `multiplier`. Ignored if `threshold` is not None. Defaults to 1.05.
        sigma: Sigma for gaussian blurring. Defaults to 1.0.

    Raises:
        ValueError: Threshold not between 0 and 255.

    Returns:
        Tuple of `threshold` and `tissue_mask` (0=background, 1=tissue).
    """
    image = check_image(image)
    if threshold is not None and not 0 <= threshold <= MAX_THRESHOLD:
        raise ValueError(ERROR_THRESHOLD.format(threshold))
    gray = image if image.ndim == GRAY_NDIM else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = _gaussian_blur(image=gray, sigma=sigma, truncate=3.5)
    if threshold is None:
        small_for_thr = downscale_to_max_pixels(blur, max_pixels=4_000_000)
        threshold = _otsu_threshold(gray=small_for_thr)
        threshold = max(min(255, int(threshold * max(0.0, multiplier) + 0.5)), 0)
    if blur.size == 0:
        return int(threshold), np.zeros_like(blur, dtype=np.uint8)
    if blur.dtype != np.uint8:
        blur = blur.astype(np.uint8)
    # THRESH_BINARY_INV semantics: 1 for values <= threshold.
    mask = (blur <= int(threshold)).astype(np.uint8)
    return int(threshold), mask


def clean_tissue_mask(
    tissue_mask: np.ndarray,
    min_area_pixel: int = 10,
    max_area_pixel: Optional[int] = None,
    min_area_relative: float = 0.2,
    max_area_relative: Optional[float] = 2.0,
) -> np.ndarray:
    """Remove too small/large contours from a tissue mask.

    Args:
        tissue_mask: Tissue mask to be cleaned.
        min_area_pixel: Minimum pixel area for contours. Defaults to 10.
        max_area_pixel: Maximum pixel area for contours. Defaults to None.
        min_area_relative: Relative minimum contour area, calculated from the median
            contour area after filtering contours with `[min,max]_pixel` arguments
            (`min_area_relative * median(contour_areas)`). Defaults to 0.2.
        max_area_relative: Relative maximum contour area, calculated from the median
            contour area after filtering contours with `[min,max]_pixel` arguments
            (`max_area_relative * median(contour_areas)`). Defaults to 2.0.

    Returns:
        Tissue mask with too small/large contours removed.
    """
    contours, __ = cv2.findContours(
        tissue_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return tissue_mask
    contour_areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    selection = contour_areas >= min_area_pixel
    if max_area_pixel is not None:
        selection = selection & (contour_areas <= max_area_pixel)
    if selection.sum() == 0:
        return np.zeros_like(tissue_mask)
    area_median = np.median(contour_areas[selection])
    area_min = area_median * min_area_relative
    area_max = None if max_area_relative is None else area_median * max_area_relative
    new_mask = np.zeros_like(tissue_mask)
    for select, area, cnt in zip(selection, contour_areas, contours):
        if select and area >= area_min and (area_max is None or area <= area_max):
            cv2.drawContours(new_mask, [cnt], -1, 1, -1)
    return new_mask


def downscale_for_thumbnail(
    image: np.ndarray, *, max_pixels: int = 3_000_000
) -> np.ndarray:
    """Downscale an image for thumbnail generation, if larger than `max_pixels`."""
    return downscale_to_max_pixels(image, max_pixels=max_pixels)


def _otsu_threshold(*, gray: np.ndarray) -> int:
    values = gray.flatten()
    values = values[(values != WHITE_PIXEL) & (values != BLACK_PIXEL)]
    if len(values) == 0:
        return 127  # Mid-range default when no valid pixels remain for Otsu.
    if len(values) == 1:
        return int(values[0])
    threshold, __ = cv2.threshold(
        values, None, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return threshold


def _gaussian_blur(
    *, image: np.ndarray, sigma: float, truncate: float = 3.5
) -> np.ndarray:
    if sigma <= SIGMA_NO_OP or image.size == 0:
        return image
    ksize = int(truncate * sigma + 0.5)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)
