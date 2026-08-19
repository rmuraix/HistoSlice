import cv2
import numpy as np

from ._check import check_image

ERROR_QUANTILES = "Quantiles should be between (0, 1)."

DEFAULT_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
DEFAULT_SHAPE = (64, 64)
GRAYSCALE_NDIM = 2
MIN_TISSUE_PIXELS = 10
MIN_QUANTILE = 0.0
MAX_QUANTILE = 1.0
BLACK_PIXEL = 0
WHITE_PIXEL = 255

DARK_PIXEL_THRESHOLD = 8
BRIGHT_PIXEL_THRESHOLD = 247
CORE_EROSION_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
CORE_EROSION_ITERATIONS = 2
MIN_CORE_PIXELS = 50


def get_image_metrics(
    image: np.ndarray,
    tissue_mask: np.ndarray,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    shape: tuple[int, int] = DEFAULT_SHAPE,
) -> dict[str, float]:
    """Calculate image metrics for preprocessing.

    The following metrics are computed:
        - Background percentage.
        - Percentage of black & white pixels (0 or 255).
        - Laplacian standard deviation.
        - Channel mean and std values for RGB/HSV/grayscale (if image is RGB).
        - Channel quantile values for RGB/HSV/grayscale (if image is RGB).

    Args:
        image: Input image
        tissue_mask: Tissue mask.
        quantiles: Possible quantile values to use. Defaults to
            (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95).
        shape: Resize shape for faster channel metric calculation. Defaults to (64, 64).

    Raises:
        ValueError: Quantiles are not between (0, 1).

    Returns:
        Dictionary of image metrics.
    """
    image = check_image(image)
    if not all(MIN_QUANTILE < x < MAX_QUANTILE for x in quantiles):
        raise ValueError(ERROR_QUANTILES)
    metrics = {}
    # Generate images for metric calculation
    if image.ndim > GRAYSCALE_NDIM:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    else:
        gray = image
    # Calculate metrics.
    metrics["background"] = ((tissue_mask == 0).sum() / tissue_mask.size).round(3)
    metrics.update(get_data_loss(gray))
    metrics.update(get_laplacian_std(gray))
    # Resize images for quicker channel metrics.
    tissue_mask = cv2.resize(tissue_mask, shape, interpolation=cv2.INTER_NEAREST)
    gray = cv2.resize(gray, shape, interpolation=cv2.INTER_NEAREST)
    if image.ndim > GRAYSCALE_NDIM:
        image = cv2.resize(image, shape, interpolation=cv2.INTER_NEAREST)
        hsv = cv2.resize(hsv, shape, interpolation=cv2.INTER_NEAREST)  # type: ignore
    # Check that there is tissue...
    if tissue_mask.sum() < MIN_TISSUE_PIXELS:
        tissue_mask[...] = 1
    # Channel mean and std.
    metrics.update(get_mean_and_std(gray, ["gray"]))
    if image.ndim > GRAYSCALE_NDIM:
        metrics.update(get_mean_and_std(image, ["red", "green", "blue"]))
        metrics.update(get_mean_and_std(hsv, ["hue", "saturation", "brightness"]))
    # Channel quantiles.
    metrics.update(get_quantiles(gray, tissue_mask, quantiles, ["gray"]))
    if image.ndim > GRAYSCALE_NDIM:
        metrics.update(
            get_quantiles(image, tissue_mask, quantiles, ["red", "green", "blue"])
        )
        metrics.update(
            get_quantiles(
                hsv, tissue_mask, quantiles, ["hue", "saturation", "brightness"]
            )
        )
    return metrics


def get_qc_metrics(image: np.ndarray, tissue_mask: np.ndarray) -> dict[str, float]:
    """Calculate minimal technical quality-control (QC) metrics for a tile.

    Unlike `get_image_metrics`, which computes a large set of exploratory
    channel statistics, this returns only the small set of metrics used by
    `histoslice.qc.quality_control` to flag technical failures (corruption,
    blown-out exposure, out-of-focus tiles) - see that module for how these
    are turned into pass/warn/fail decisions. Cheap enough to compute for
    every tile, regardless of whether full metrics are requested.

    The following metrics are computed:
        - `background`: fraction of non-tissue pixels.
        - `dark_fraction` / `bright_fraction`: fraction of near-black /
          near-white grayscale pixels.
        - `gray_std`: whole-tile grayscale standard deviation (full
          resolution, unlike the resized value in `get_image_metrics`).
        - `focus_score`: Laplacian variance restricted to an eroded
          "tissue core" mask, to avoid the tissue/background boundary
          dominating the sharpness estimate.
        - `tissue_brightness` / `tissue_saturation`: tissue-only median HSV
          value/saturation.
        - `tissue_contrast`: tissue-only grayscale q90 - q10.

    Args:
        image: Input image.
        tissue_mask: Tissue mask (0=background, 1=tissue).

    Returns:
        Dictionary of QC metrics.
    """
    image = check_image(image)
    if image.ndim > GRAYSCALE_NDIM:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    else:
        gray = image
        hsv = None
    tissue_selected = tissue_mask == 1
    has_tissue = int(tissue_selected.sum()) >= MIN_TISSUE_PIXELS
    tissue_gray = gray[tissue_selected] if has_tissue else gray.reshape(-1)

    metrics = {
        "background": round(float((tissue_mask == 0).sum() / tissue_mask.size), 3),
        "dark_fraction": round(float(np.mean(gray <= DARK_PIXEL_THRESHOLD)), 3),
        "bright_fraction": round(float(np.mean(gray >= BRIGHT_PIXEL_THRESHOLD)), 3),
        "gray_std": round(float(gray.std()), 3),
        "focus_score": get_focus_score(gray, tissue_mask),
        "tissue_contrast": round(
            float(np.quantile(tissue_gray, 0.9) - np.quantile(tissue_gray, 0.1)), 3
        ),
    }
    if hsv is not None:
        tissue_hsv = hsv[tissue_selected] if has_tissue else hsv.reshape(-1, 3)
        metrics["tissue_brightness"] = round(float(np.median(tissue_hsv[:, 2])), 3)
        metrics["tissue_saturation"] = round(float(np.median(tissue_hsv[:, 1])), 3)
    else:
        metrics["tissue_brightness"] = round(float(np.median(tissue_gray)), 3)
        metrics["tissue_saturation"] = 0.0
    return metrics


def get_focus_score(gray: np.ndarray, tissue_mask: np.ndarray) -> float:
    """Laplacian variance over an eroded tissue-core mask.

    Restricting the Laplacian to a "core" mask (tissue mask eroded by a few
    pixels) avoids the sharp tissue/background boundary itself dominating the
    variance, which would make background-heavy tiles look artificially
    sharp regardless of actual focus. Falls back to the full tissue mask (or
    `0.0` if there isn't enough tissue at all) when erosion leaves too small
    a core to measure.
    """
    core = cv2.erode(
        tissue_mask.astype(np.uint8),
        CORE_EROSION_KERNEL,
        iterations=CORE_EROSION_ITERATIONS,
    )
    core_selected = core == 1
    if int(core_selected.sum()) < MIN_CORE_PIXELS:
        core_selected = tissue_mask == 1
    if int(core_selected.sum()) < MIN_TISSUE_PIXELS:
        return 0.0
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    return round(float(laplacian[core_selected].var()), 3)


def get_mean_and_std(image: np.ndarray, names: list[str]) -> dict[str, float]:
    """Collect mean and standard deviation for image."""
    if image.ndim == GRAYSCALE_NDIM:
        return {
            **_get_channel_mean(image, names[0]),
            **_get_channel_std(image, names[0]),
        }
    output = {}
    for channel_idx, name in enumerate(names):
        output.update(_get_channel_mean(image[..., channel_idx], name))
        output.update(_get_channel_std(image[..., channel_idx], name))
    return output


def get_quantiles(
    image: np.ndarray, tissue_mask: np.ndarray, quantiles: list[float], names: list[str]
) -> dict[str, float]:
    if image.ndim == GRAYSCALE_NDIM:
        return _get_channel_quantiles(image, tissue_mask, quantiles, names[0])
    output = {}
    for channel_idx, name in enumerate(names):
        output.update(
            _get_channel_quantiles(
                image[..., channel_idx], tissue_mask, quantiles, name
            )
        )
    return output


def get_data_loss(gray: np.ndarray) -> dict[str, float]:
    """Calculate percentage of black and white pixels."""
    return {
        "black_pixels": (gray == BLACK_PIXEL).sum() / gray.size,
        "white_pixels": (gray == WHITE_PIXEL).sum() / gray.size,
    }


def get_laplacian_std(gray: np.ndarray) -> dict[str, float]:
    """Calculate laplacian standard deviation for sharpness evaluation."""
    return {"laplacian_std": cv2.Laplacian(gray, cv2.CV_32F).std()}


def _get_channel_mean(channel: np.ndarray, name: str) -> dict[str, float]:
    """Calculate mean value for the channel."""
    return {f"{name}_mean": channel.mean().round(3).tolist()}


def _get_channel_std(channel: np.ndarray, name: str) -> dict[str, float]:
    """Calculate std value for the channel."""
    return {f"{name}_std": channel.std().round(3).tolist()}


def _get_channel_quantiles(
    channel: np.ndarray,
    tissue_mask: np.ndarray,
    quantiles: tuple[float, ...],
    name: str,
) -> dict[str, int]:
    """Calculate quantile values for the channel."""
    output = {}
    bins = np.cumsum(np.bincount(channel[tissue_mask == 1].flatten(), minlength=256))
    n_pixels = (tissue_mask == 1).sum()

    for q in quantiles:
        threshold = int(q * n_pixels)
        idx = np.flatnonzero(bins > threshold)[0]
        output[f"{name}_q{int(100 * q)}"] = int(idx)

    return output
