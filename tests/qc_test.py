"""Tests for `histoslice.qc` - technical QC for extracted tiles.

Uses synthetic metadata dataframes (rather than real tile images) since
`quality_control` operates purely on the QC metric columns already saved in
`metadata.parquet` - see `tests/metrics_test.py` for `get_qc_metrics`/
`get_focus_score` tests against real images.
"""

import numpy as np
import polars as pl
import pytest

from histoslice.qc import QC_METRIC_COLUMNS, QCConfig, quality_control

NUM_REFERENCE_TILES = 50


def _normal_frame(n: int = NUM_REFERENCE_TILES, *, seed: int = 0) -> pl.DataFrame:
    """Synthetic "normal" tiles: plausible, mildly varying QC metrics with no
    hard failures and (by construction) no slide-relative anomalies."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "x": np.arange(n),
            "y": np.arange(n),
            "w": np.full(n, 256),
            "h": np.full(n, 256),
            "path": [f"tile_{i}.jpeg" for i in range(n)],
            "background": rng.uniform(0.1, 0.4, n),
            "dark_fraction": rng.uniform(0.0, 0.02, n),
            "bright_fraction": rng.uniform(0.0, 0.01, n),
            "gray_std": rng.uniform(40.0, 60.0, n),
            "focus_score": rng.normal(300.0, 10.0, n).clip(min=1.0),
            "tissue_brightness": rng.normal(160.0, 5.0, n),
            "tissue_saturation": rng.normal(100.0, 5.0, n),
            "tissue_contrast": rng.normal(80.0, 5.0, n),
        }
    )


def _set_row(df: pl.DataFrame, index: int, **values: float) -> pl.DataFrame:
    """Overwrite metric values for a single row (by position)."""
    for column, value in values.items():
        df = df.with_columns(
            pl.when(pl.int_range(pl.len()) == index)
            .then(pl.lit(value))
            .otherwise(pl.col(column))
            .alias(column)
        )
    return df


def _reasons(df: pl.DataFrame, index: int) -> list[str]:
    return df["qc_reasons"][index]


# --- A. Normal tiles do not require an outlier -----------------------------


def test_normal_tiles_have_no_hard_failures() -> None:
    result = quality_control(_normal_frame())
    assert int(result["is_outlier"].sum()) == 0
    assert (result["qc_status"] != "fail").all()


# --- B/C/D. Hard-fail rules -------------------------------------------------


def test_near_white_tile_fails() -> None:
    df = _set_row(_normal_frame(), 0, bright_fraction=0.999)
    result = quality_control(df)
    assert result["qc_status"][0] == "fail"
    assert "near_white" in _reasons(result, 0)
    assert result["is_outlier"][0]


def test_near_black_tile_fails() -> None:
    df = _set_row(_normal_frame(), 0, dark_fraction=0.95)
    result = quality_control(df)
    assert result["qc_status"][0] == "fail"
    assert "near_black" in _reasons(result, 0)
    assert result["is_outlier"][0]


def test_constant_image_fails_low_dynamic_range() -> None:
    df = _set_row(_normal_frame(), 0, gray_std=1.0)
    result = quality_control(df)
    assert result["qc_status"][0] == "fail"
    assert "low_dynamic_range" in _reasons(result, 0)
    assert result["is_outlier"][0]


def test_hard_fail_score_is_at_least_two() -> None:
    df = _set_row(_normal_frame(), 0, gray_std=1.0)
    result = quality_control(df)
    assert result["qc_score"][0] >= 2.0


# --- E. Blur is a warning, never a hard failure -----------------------------


def test_severe_relative_blur_warns_but_does_not_fail() -> None:
    df = _set_row(_normal_frame(), 0, focus_score=0.5)
    result = quality_control(df)
    assert result["qc_focus_z"][0] < -4
    assert result["qc_status"][0] == "warn"
    assert "possible_blur" in _reasons(result, 0)
    assert not result["is_outlier"][0]


def test_blur_alone_never_produces_fail_status() -> None:
    """Even an essentially-zero focus score must not by itself hard-fail -
    Laplacian-based focus can be legitimately low for real tissue (adipose,
    mucin, loose stroma, necrosis)."""
    df = _set_row(_normal_frame(), 0, focus_score=0.0)
    result = quality_control(df)
    assert result["qc_status"][0] in ("pass", "warn")


# --- F. Appearance rarity alone is not automatic failure --------------------


def test_single_appearance_anomaly_does_not_warn_or_fail() -> None:
    df = _set_row(_normal_frame(), 0, tissue_brightness=400.0)
    result = quality_control(df)
    assert abs(result["qc_brightness_z"][0]) > 5
    assert result["qc_status"][0] == "pass"
    assert not result["is_outlier"][0]
    assert not result["needs_review"][0]


# --- G. Multiple appearance anomalies warn ----------------------------------


def test_two_appearance_anomalies_warn_appearance_shift() -> None:
    df = _set_row(_normal_frame(), 0, tissue_brightness=400.0, tissue_saturation=400.0)
    result = quality_control(df)
    assert result["qc_status"][0] == "warn"
    assert "appearance_shift" in _reasons(result, 0)
    assert result["needs_review"][0]
    assert not result["is_outlier"][0]


# --- H. Small reference set --------------------------------------------------


def test_small_reference_set_disables_soft_qc() -> None:
    small = _normal_frame(n=10)
    small = _set_row(small, 0, tissue_brightness=1000.0, tissue_saturation=1000.0)
    result = quality_control(small)  # must not raise
    assert (result["qc_focus_z"] == 0).all()
    assert (result["qc_brightness_z"] == 0).all()
    assert (result["qc_status"] != "warn").all()
    assert (result["qc_status"] != "fail").all()


# --- I. Zero MAD -------------------------------------------------------------


def test_zero_mad_produces_no_nan_inf_or_false_anomalies() -> None:
    n = NUM_REFERENCE_TILES
    df = pl.DataFrame(
        {
            "x": np.arange(n),
            "y": np.arange(n),
            "w": np.full(n, 256),
            "h": np.full(n, 256),
            "path": [f"tile_{i}.jpeg" for i in range(n)],
            "background": np.full(n, 0.2),
            "dark_fraction": np.zeros(n),
            "bright_fraction": np.zeros(n),
            "gray_std": np.full(n, 50.0),
            "focus_score": np.full(n, 300.0),
            "tissue_brightness": np.full(n, 160.0),
            "tissue_saturation": np.full(n, 100.0),
            "tissue_contrast": np.full(n, 80.0),
        }
    )
    result = quality_control(df)
    for column in (
        "qc_focus_z",
        "qc_brightness_z",
        "qc_saturation_z",
        "qc_contrast_z",
        "qc_score",
    ):
        values = result[column].to_numpy()
        assert np.isfinite(values).all()
    assert int(result["is_outlier"].sum()) == 0
    assert int(result["needs_review"].sum()) == 0


# --- J. Schema invariants ----------------------------------------------------


def test_is_outlier_and_needs_review_match_qc_status() -> None:
    df = _normal_frame()
    df = _set_row(df, 0, bright_fraction=0.999)  # fail
    df = _set_row(df, 1, tissue_brightness=400.0, tissue_saturation=400.0)  # warn
    result = quality_control(df)
    assert (result["is_outlier"] == (result["qc_status"] == "fail")).all()
    assert (result["needs_review"] == (result["qc_status"] == "warn")).all()
    assert result["qc_method"].unique().to_list() == ["technical_qc_v1"]


def test_quality_control_missing_columns_raises() -> None:
    df = pl.DataFrame({"x": [0], "y": [0], "w": [1], "h": [1], "path": ["x.jpeg"]})
    with pytest.raises(ValueError, match="QC metric columns"):
        quality_control(df)


def test_quality_control_preserves_existing_columns() -> None:
    df = _normal_frame()
    result = quality_control(df)
    for column in df.columns:
        assert column in result.columns
    for column in QC_METRIC_COLUMNS:
        assert column in result.columns


def test_custom_config_thresholds() -> None:
    df = _set_row(_normal_frame(), 0, dark_fraction=0.5)
    # Default threshold (0.9) does not fail this tile...
    assert quality_control(df)["qc_status"][0] != "fail"
    # ...but a stricter custom config does.
    strict = quality_control(df, config=QCConfig(dark_fraction_fail=0.4))
    assert strict["qc_status"][0] == "fail"
    assert "near_black" in strict["qc_reasons"][0]
