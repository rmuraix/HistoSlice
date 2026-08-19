"""Technical quality control (QC) for extracted tiles.

`quality_control` flags tiles for clear technical failures (corruption,
blown-out exposure, near-constant images) or for showing a slide-relative
anomaly strong enough to warrant a manual look. It deliberately does **not**
try to detect biologically unusual tissue - a rare tissue type is not a
technical failure. See `histoslice.utils.OutlierDetector.cluster_kmeans` if
you want exploratory clustering of tile metrics instead.

`qc_status` is one of:
    - `"pass"`: no issues detected.
    - `"warn"`: a possible technical artifact, not certain enough to exclude
      automatically. Kept by default - filter on `needs_review` yourself.
    - `"fail"`: a clear technical failure, safe to exclude from downstream
      analysis (`is_outlier`).

`qc_score` is a relative severity score, not a probability. It is only
meaningful for ranking/reviewing tiles within the same slide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

__all__ = ["QCConfig", "quality_control"]

QC_METHOD = "technical_qc_v1"

QC_METRIC_COLUMNS = (
    "background",
    "dark_fraction",
    "bright_fraction",
    "gray_std",
    "focus_score",
    "tissue_brightness",
    "tissue_saturation",
    "tissue_contrast",
)

ERROR_NO_QC_METRICS = (
    "Metadata does not contain QC metric columns ({}). Re-run `slice` with "
    "the current HistoSlice version to regenerate `metadata.parquet`."
)

EPS = 1e-9


@dataclass(frozen=True)
class QCConfig:
    """Thresholds used by `quality_control`.

    Attributes:
        min_reference_tiles: Minimum number of reference tiles required to
            compute slide-relative (soft) QC. Below this, only the absolute
            hard-fail rules are applied.
        min_reference_tissue_fraction: Minimum tissue fraction (`1 -
            background`) for a tile to be eligible as a reference tile.
        dark_fraction_fail: `dark_fraction >=` this fails as `"near_black"`.
        bright_fraction_fail: `bright_fraction >=` this fails as
            `"near_white"`.
        gray_std_fail: `gray_std <=` this fails as `"low_dynamic_range"`.
        focus_z_warn: `z_focus <` this warns as `"possible_blur"`.
        appearance_z_warn: absolute z-score threshold for brightness/
            saturation votes towards `"appearance_shift"`.
        contrast_z_warn: `z_contrast <` this counts as a vote towards
            `"appearance_shift"`.
        min_appearance_votes: number of brightness/saturation/contrast votes
            required before warning `"appearance_shift"` - a single unusual
            appearance metric is not enough (avoids flagging biologically
            valid but differently-colored tissue).
    """

    min_reference_tiles: int = 32
    min_reference_tissue_fraction: float = 0.5

    dark_fraction_fail: float = 0.90
    bright_fraction_fail: float = 0.995
    gray_std_fail: float = 2.0

    focus_z_warn: float = -4.0
    appearance_z_warn: float = 5.0
    contrast_z_warn: float = -5.0
    min_appearance_votes: int = 2


def quality_control(
    dataframe: pl.DataFrame, *, config: QCConfig = QCConfig()
) -> pl.DataFrame:
    """Run technical QC over a slide's tile metadata.

    Args:
        dataframe: Tile metadata containing `QC_METRIC_COLUMNS` (saved by
            `slice`/`export_tiles` regardless of `save_metrics`).
        config: Thresholds. Defaults to `QCConfig()`.

    Raises:
        ValueError: `dataframe` is missing required QC metric columns.

    Returns:
        `dataframe` with QC columns added: `qc_status`, `qc_score`,
        `qc_reasons`, `qc_focus_z`, `qc_brightness_z`, `qc_saturation_z`,
        `qc_contrast_z`, `qc_method`, `is_outlier` (`qc_status == "fail"`),
        `needs_review` (`qc_status == "warn"`).
    """
    missing = [c for c in QC_METRIC_COLUMNS if c not in dataframe.columns]
    if missing:
        raise ValueError(ERROR_NO_QC_METRICS.format(", ".join(missing)))

    num_tiles = len(dataframe)
    background = dataframe["background"].to_numpy()
    dark_fraction = dataframe["dark_fraction"].to_numpy()
    bright_fraction = dataframe["bright_fraction"].to_numpy()
    gray_std = dataframe["gray_std"].to_numpy()
    focus_score = dataframe["focus_score"].to_numpy()
    tissue_brightness = dataframe["tissue_brightness"].to_numpy()
    tissue_saturation = dataframe["tissue_saturation"].to_numpy()
    tissue_contrast = dataframe["tissue_contrast"].to_numpy()

    hard_fail = np.zeros(num_tiles, dtype=bool)
    reasons: list[list[str]] = [[] for _ in range(num_tiles)]

    _flag(hard_fail, reasons, dark_fraction >= config.dark_fraction_fail, "near_black")
    _flag(
        hard_fail, reasons, bright_fraction >= config.bright_fraction_fail, "near_white"
    )
    _flag(hard_fail, reasons, gray_std <= config.gray_std_fail, "low_dynamic_range")

    reference = (~hard_fail) & (
        background <= (1.0 - config.min_reference_tissue_fraction)
    )
    num_reference = int(reference.sum())

    z_focus = np.zeros(num_tiles)
    z_brightness = np.zeros(num_tiles)
    z_saturation = np.zeros(num_tiles)
    z_contrast = np.zeros(num_tiles)
    warn = np.zeros(num_tiles, dtype=bool)

    if num_reference >= config.min_reference_tiles:
        z_focus = _robust_z(np.log1p(np.clip(focus_score, 0.0, None)), reference)
        z_brightness = _robust_z(tissue_brightness, reference)
        z_saturation = _robust_z(tissue_saturation, reference)
        z_contrast = _robust_z(tissue_contrast, reference)

        possible_blur = (~hard_fail) & (z_focus < config.focus_z_warn)
        _flag(warn, reasons, possible_blur, "possible_blur")

        appearance_votes = (
            (np.abs(z_brightness) > config.appearance_z_warn).astype(int)
            + (np.abs(z_saturation) > config.appearance_z_warn).astype(int)
            + (z_contrast < config.contrast_z_warn).astype(int)
        )
        appearance_shift = (~hard_fail) & (
            appearance_votes >= config.min_appearance_votes
        )
        _flag(warn, reasons, appearance_shift, "appearance_shift")

    qc_status = np.full(num_tiles, "pass", dtype=object)
    qc_status[warn] = "warn"
    qc_status[hard_fail] = "fail"

    focus_severity = np.maximum(0.0, -z_focus / 4.0)
    brightness_severity = np.abs(z_brightness) / 5.0
    saturation_severity = np.abs(z_saturation) / 5.0
    contrast_severity = np.maximum(0.0, -z_contrast / 5.0)
    qc_score = np.maximum.reduce(
        [focus_severity, brightness_severity, saturation_severity, contrast_severity]
    )
    qc_score = np.where(hard_fail, np.maximum(qc_score, 2.0), qc_score)

    return dataframe.with_columns(
        [
            pl.Series("qc_status", qc_status.tolist(), dtype=pl.Utf8),
            pl.Series("qc_score", np.round(qc_score, 3)),
            pl.Series("qc_reasons", reasons, dtype=pl.List(pl.Utf8)),
            pl.Series("qc_focus_z", np.round(z_focus, 3)),
            pl.Series("qc_brightness_z", np.round(z_brightness, 3)),
            pl.Series("qc_saturation_z", np.round(z_saturation, 3)),
            pl.Series("qc_contrast_z", np.round(z_contrast, 3)),
            pl.lit(QC_METHOD).alias("qc_method"),
            pl.Series("is_outlier", hard_fail),
            pl.Series("needs_review", warn),
        ]
    )


def _flag(
    target: np.ndarray, reasons: list[list[str]], mask: np.ndarray, reason: str
) -> None:
    """Set `target[mask] = True` in place and append `reason` for those rows."""
    for i in np.flatnonzero(mask):
        reasons[i].append(reason)
    target |= mask


def _robust_z(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Robust z-score of `x`, scaled from the MAD (falling back to the IQR,
    then to an all-zero score) of `x[reference]`. Never produces NaN/inf."""
    ref_values = x[reference]
    median = np.median(ref_values)
    mad = np.median(np.abs(ref_values - median))
    if mad > EPS:
        scale = 1.4826 * mad
    else:
        q25, q75 = np.quantile(ref_values, [0.25, 0.75])
        iqr = q75 - q25
        if iqr > EPS:
            scale = iqr / 1.349
        else:
            return np.zeros_like(x, dtype=float)
    z = (x - median) / scale
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
