import cv2
import numpy as np

from histoslice import functional as F


def test_grayscale_metrics() -> None:
    # Create image.
    image = np.zeros((100, 100), dtype=np.uint8) + 255
    image[:20, :20] = 60
    image[50:60, 20:50] = 100
    image[80:, 80:] = 120
    # Detect tissue.
    __, tissue_mask = F.get_tissue_mask(image, threshold=200)
    metrics = F.get_image_metrics(image, tissue_mask)
    assert int(metrics.pop("laplacian_std")) == 29
    assert metrics == {
        "background": 0.886,
        "black_pixels": 0.0,
        "white_pixels": 0.89,
        "gray_mean": 237.175,
        "gray_std": 51.692,
        "gray_q5": 60,
        "gray_q10": 60,
        "gray_q25": 60,
        "gray_q50": 100,
        "gray_q75": 120,
        "gray_q90": 120,
        "gray_q95": 255,
    }


def test_rgb_metrics() -> None:
    # Create image.
    image = np.zeros((100, 100, 3), dtype=np.uint8) + 255
    image[:20, :20, 1] = 0
    image[:20, :20, 2] = 0
    image[50:60, 20:50, 0] = 0
    image[50:60, 20:50, 1] = 128
    image[50:60, 20:50, 2] = 128
    image[80:, 80:, 0] = 0
    image[80:, 80:, 1] = 0
    image[80:, 80:, 2] = 255
    # Detect tissue.
    __, tissue_mask = F.get_tissue_mask(image, threshold=200)
    # Check metrics.
    metrics = F.get_image_metrics(image, tissue_mask, quantiles=[0.9])
    assert int(metrics.pop("laplacian_std")) == 33
    assert metrics == {
        "background": 0.886,
        "black_pixels": 0.0,
        "white_pixels": 0.89,
        "gray_mean": 234.312,
        "gray_std": 59.78,
        "red_mean": 237.755,
        "red_std": 64.032,
        "green_mean": 231.39,
        "green_std": 70.251,
        "blue_mean": 240.355,
        "blue_std": 54.701,
        "hue_mean": 7.141,
        "hue_std": 26.801,
        "saturation_mean": 27.766,
        "saturation_std": 79.432,
        "brightness_mean": 250.876,
        "brightness_std": 22.51,
        "gray_q90": 90,
        "red_q90": 255,
        "green_q90": 128,
        "blue_q90": 255,
        "hue_q90": 120,
        "saturation_q90": 255,
        "brightness_q90": 255,
    }


def _textured_image(
    shape: tuple[int, int] = (128, 128), *, seed: int = 0
) -> np.ndarray:
    """A grayscale image with real texture, so Laplacian variance is meaningful."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def test_qc_metrics_keys_and_background() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8) + 200
    tissue_mask = np.zeros((64, 64), dtype=np.uint8)
    tissue_mask[16:48, 16:48] = 1  # 25% tissue -> 75% background
    metrics = F.get_qc_metrics(image, tissue_mask)
    assert set(metrics) == {
        "background",
        "dark_fraction",
        "bright_fraction",
        "gray_std",
        "focus_score",
        "tissue_brightness",
        "tissue_saturation",
        "tissue_contrast",
    }
    assert metrics["background"] == 0.75


def test_qc_metrics_dark_and_bright_fraction() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:32, :, :] = 0  # near-black half
    image[32:, :, :] = 255  # near-white half
    tissue_mask = np.ones((64, 64), dtype=np.uint8)
    metrics = F.get_qc_metrics(image, tissue_mask)
    assert metrics["dark_fraction"] == 0.5
    assert metrics["bright_fraction"] == 0.5


def test_qc_metrics_grayscale_image_has_no_saturation() -> None:
    image = _textured_image()
    tissue_mask = np.ones(image.shape, dtype=np.uint8)
    metrics = F.get_qc_metrics(image, tissue_mask)
    assert metrics["tissue_saturation"] == 0.0
    assert metrics["focus_score"] > 0


def test_focus_score_no_tissue_is_safe() -> None:
    image = _textured_image()
    tissue_mask = np.zeros(image.shape, dtype=np.uint8)
    assert F.get_qc_metrics(image, tissue_mask)["focus_score"] == 0.0


def test_focus_score_blur_progression() -> None:
    """Sharper images should have a strictly higher focus score than blurred
    ones - the key regression test for using focus (not clustering) for QC."""
    gray = _textured_image()
    tissue_mask = np.ones(gray.shape, dtype=np.uint8)

    original = F.get_qc_metrics(gray, tissue_mask)["focus_score"]
    moderate_blur = cv2.GaussianBlur(gray, (9, 9), sigmaX=3.0)
    severe_blur = cv2.GaussianBlur(gray, (25, 25), sigmaX=10.0)

    moderate_score = F.get_qc_metrics(moderate_blur, tissue_mask)["focus_score"]
    severe_score = F.get_qc_metrics(severe_blur, tissue_mask)["focus_score"]

    assert original > moderate_score > severe_score
