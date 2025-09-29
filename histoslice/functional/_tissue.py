from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

from ._check import check_image

ERROR_THRESHOLD = "Threshold should be in range [0, 255], got {}."

MAX_THRESHOLD = 255
WHITE_PIXEL = 255
BLACK_PIXEL = 0
SIGMA_NO_OP = 0.0
GRAY_NDIM = 2


def get_tissue_mask(
    image: Union[Image.Image, np.ndarray],
    *,
    threshold: Optional[int] = None,
    multiplier: float = 1.0,
    sigma: float = 1.0,
) -> tuple[int, np.ndarray]:
    """Detect tissue from image.

    Args:
        image: Input image.
        threshold: Threshold for tissue detection. If set, will detect tissue by
            global thresholding, and otherwise Otsu's method is used to find
            a threshold. Defaults to None.
        multiplier: Otsu's method is used to find an optimal threshold by
            minimizing the weighted within-class variance. This threshold is
            then multiplied with `multiplier`. Ignored if `threshold` is not None.
            Defaults to 1.0.
        sigma: Sigma for gaussian blurring. Defaults to 1.0.

    Raises:
        ValueError: Threshold not between 0 and 255.

    Returns:
        Tuple with `threshold` and `tissue_mask` (0=background and 1=tissue).
    """
    # Check image and convert to array.
    image = check_image(image)
    # Check arguments.
    if threshold is not None and not 0 <= threshold <= MAX_THRESHOLD:
        raise ValueError(ERROR_THRESHOLD.format(threshold))
    # Convert to grayscale.
    gray = image if image.ndim == GRAY_NDIM else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Gaussian blurring.
    blur = _gaussian_blur(image=gray, sigma=sigma, truncate=3.5)
    # Get threshold.
    if threshold is None:
        # Compute Otsu threshold on a downscaled image to avoid OpenCV limits
        small_for_thr = _downscale_for_threshold(blur)
        threshold = _otsu_threshold(gray=small_for_thr)
        threshold = max(min(255, int(threshold * max(0.0, multiplier) + 0.5)), 0)
    # Global thresholding: avoid cv2.threshold on gigantic arrays; use NumPy instead
    if blur.size == 0:
        mask = np.zeros_like(blur, dtype=np.uint8)
        return int(threshold), mask
    if blur.dtype != np.uint8:
        blur = blur.astype(np.uint8)
    # OpenCV's THRESH_BINARY_INV sets 1 for values <= threshold
    mask = (blur <= int(threshold)).astype(np.uint8)
    return int(threshold), mask


def clean_tissue_mask(
    tissue_mask: np.ndarray,
    min_area_pixel: int = 10,
    max_area_pixel: Optional[int] = None,
    min_area_relative: float = 0.2,
    max_area_relative: Optional[float] = 2.0,
) -> np.ndarray:
    """Remove too small/large contours from tissue mask.

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
    # Filter based on pixel values.
    selection = contour_areas >= min_area_pixel
    if max_area_pixel is not None:
        selection = selection & (contour_areas <= max_area_pixel)
    if selection.sum() == 0:
        # Nothing to draw
        return np.zeros_like(tissue_mask)
    # Define relative min/max values.
    area_median = np.median(contour_areas[selection])
    area_min = area_median * min_area_relative
    area_max = None if max_area_relative is None else area_median * max_area_relative
    # Draw new mask.
    new_mask = np.zeros_like(tissue_mask)
    for select, area, cnt in zip(selection, contour_areas, contours):
        if select and area >= area_min and (area_max is None or area <= area_max):
            cv2.drawContours(new_mask, [cnt], -1, 1, -1)
    return new_mask


def _otsu_threshold(*, gray: np.ndarray) -> int:
    """Helper function to calculate Otsu's thresold from a grayscale image."""
    values = gray.flatten()
    values = values[(values != WHITE_PIXEL) & (values != BLACK_PIXEL)]

    # Handle case where all pixels are black or white (empty array after filtering)
    if len(values) == 0:
        # Return a default threshold value when no valid pixels for Otsu calculation
        return 127  # Mid-range default threshold

    # Ensure values array is properly formatted for cv2.threshold
    # Some OpenCV versions require proper array structure for OTSU method
    if len(values) == 1:
        # Single pixel case - return the pixel value as threshold
        return int(values[0])

    threshold, __ = cv2.threshold(
        values, None, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return threshold


def _gaussian_blur(
    *, image: np.ndarray, sigma: float, truncate: float = 3.5
) -> np.ndarray:
    """Apply gaussian blurring."""
    if sigma <= SIGMA_NO_OP:
        return image

    # Handle empty arrays that would cause GaussianBlur to fail
    if image.size == 0:
        return image

    ksize = int(truncate * sigma + 0.5)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def _downscale_for_threshold(
    image: np.ndarray, *, max_pixels: int = 4_000_000
) -> np.ndarray:
    """Downscale image for threshold computation if too large.

    Ensures the number of pixels does not exceed `max_pixels`.
    Uses area interpolation for downscaling; leaves image unchanged if small enough.
    """
    # Guard against empty input
    if image.size == 0:
        return image
    h, w = image.shape[:2]
    total = int(h) * int(w)
    if total <= max_pixels:
        return image
    # Compute uniform scale factor
    scale = float(np.sqrt(max_pixels / float(total)))
    # OpenCV expects contiguous uint8 for resize here
    src = image if image.dtype == np.uint8 else image.astype(np.uint8)
    if not src.flags.c_contiguous:
        src = np.ascontiguousarray(src)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)
