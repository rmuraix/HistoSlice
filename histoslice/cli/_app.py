"""CLI interface for cutting slides into small tile images."""

from __future__ import annotations

import functools
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Dict, NoReturn, Optional

import typer
from tqdm import tqdm
from typer_di import Depends, TyperDI
from typing_extensions import Annotated

from histoslice import SlideReader
from histoslice.cli._models import Settings
from histoslice.cli._options import (
    CutSlideKwargs,
    ReaderKwargs,
    SaveKwargs,
    TileKwargs,
    TissueKwargs,
    clean_opts,
    io_opts,
    save_opts,
    tile_opts,
    tissue_opts,
)
from histoslice.functional._concurrent import DEFAULT_START_METHOD

if TYPE_CHECKING:
    import numpy as np
    import polars as pl

app = TyperDI(
    name="histoslice", help="Tools for preprocessing histological slide images."
)


@app.command("slice")
def cut_slides(
    io: Annotated[Dict, Depends(io_opts)],
    tile: Annotated[Dict, Depends(tile_opts)],
    tissue: Annotated[Dict, Depends(tissue_opts)],
    save: Annotated[Dict, Depends(save_opts)],
) -> None:
    """Extract tile images from histological slides."""
    cfg = Settings(**io, **tile, **tissue, **save)

    paths = filter_slide_paths(
        all_paths=[p if isinstance(p, Path) else Path(p) for p in cfg.paths],
        parent_dir=cfg.parent_dir,
        overwrite=cfg.overwrite,
        overwrite_unfinished=cfg.overwrite_unfinished,
    )

    kwargs: CutSlideKwargs = {
        "reader_kwargs": {
            "mpp": (cfg.mpp, cfg.mpp) if cfg.mpp is not None else None,
        },
        "max_dimension": cfg.max_dimension,
        "tissue_kwargs": {
            "level": cfg.tissue_level,
            "threshold": cfg.threshold,
            "multiplier": cfg.multiplier,
            "sigma": cfg.sigma,
        },
        "tile_kwargs": {
            "width": cfg.width,
            "height": cfg.height,
            "target_mpp": cfg.target_mpp,
            "overlap": cfg.overlap,
            "out_of_bounds": not cfg.in_bounds,
            "max_background": cfg.max_background,
        },
        "save_kwargs": {
            "parent_dir": cfg.parent_dir,
            "level": cfg.level,
            "save_metrics": cfg.save_metrics,
            "save_masks": cfg.save_masks,
            "save_thumbnails": cfg.save_thumbnails,
            "image_format": cfg.image_format,
            "quality": cfg.quality,
            "raise_exception": False,  # handled here
            "num_workers": 0,  # slide-per-process
            "overwrite": True,  # filtered earlier
            "verbose": False,  # common progressbar
        },
    }

    # resolve num_workers
    effective_workers = (
        (os.cpu_count() or 1) if cfg.num_workers is None else cfg.num_workers
    )

    if effective_workers == 0:
        for path in paths:
            _, exception, failures = cut_slide(path, **kwargs)
            if isinstance(exception, Exception):
                warning(f"Could not process {path} due to exception: {exception!r}")
            elif failures:
                warning(
                    f"Slide {path} completed with {failures} failed tile(s). "
                    "See failures.json for details."
                )
    else:
        ctx = mp.get_context(DEFAULT_START_METHOD)
        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as pool:
            func = functools.partial(cut_slide, **kwargs)
            # Submit all tasks and track with futures dict
            futures = {pool.submit(func, path): path for path in paths}
            # Use as_completed for responsive progress bar
            for future in tqdm(
                as_completed(futures), desc="Cutting slides", total=len(paths)
            ):
                path, exception, failures = future.result()
                if isinstance(exception, Exception):
                    warning(f"Could not process {path} due to exception: {exception!r}")
                elif failures:
                    warning(
                        f"Slide {path} completed with {failures} failed tile(s). "
                        "See failures.json for details."
                    )


@app.command("clean")
def clean_tiles(
    clean: Annotated[Dict, Depends(clean_opts)],
) -> None:
    """Detect outlier tile images and save metadata_clean.parquet."""
    import glob
    from pathlib import Path

    # Validate mode
    if clean["mode"] not in ("calibrate", "clustering"):
        error(
            f"Unknown mode '{clean['mode']}'. Supported modes: 'calibrate', 'clustering'."
        )

    # Find slide directories (each containing tiles and metadata)
    slide_dirs = []
    for path in glob.glob(clean["input_pattern"], recursive=True):
        path = Path(path)
        if path.is_dir():
            # Check if directory contains metadata
            metadata_files = list(path.glob("metadata.parquet"))
            if metadata_files:
                slide_dirs.append(path)

    if not slide_dirs:
        error(
            f"Found no slide directories with metadata matching pattern '{clean['input_pattern']}'."
        )

    info(f"Found {len(slide_dirs)} slide(s) to process.")

    # Prepare kwargs for processing
    clean_kwargs = {
        "mode": clean["mode"],
        "outlier_frac": clean["outlier_frac"],
        "num_clusters": clean["num_clusters"],
    }

    # Resolve num_workers
    effective_workers = (
        (os.cpu_count() or 1) if clean["num_workers"] is None else clean["num_workers"]
    )

    if effective_workers == 0:
        # Sequential processing
        for slide_dir in slide_dirs:
            _, exception = process_slide_outliers(slide_dir, **clean_kwargs)
            if isinstance(exception, Exception):
                warning(
                    f"Could not process {slide_dir} due to exception: {exception!r}"
                )
    else:
        # Parallel processing
        ctx = mp.get_context(DEFAULT_START_METHOD)
        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as pool:
            func = functools.partial(process_slide_outliers, **clean_kwargs)
            # Submit all tasks and track with futures dict
            futures = {
                pool.submit(func, slide_dir): slide_dir for slide_dir in slide_dirs
            }
            # Use as_completed for responsive progress bar
            for future in tqdm(
                as_completed(futures), desc="Cleaning slides", total=len(slide_dirs)
            ):
                slide_dir, exception = future.result()
                if isinstance(exception, Exception):
                    warning(
                        f"Could not process {slide_dir} due to exception: {exception!r}"
                    )


def process_slide_outliers(
    slide_dir: Path,
    *,
    mode: str,
    outlier_frac: float = 0.01,
    num_clusters: int,
) -> tuple[Path, Optional[Exception]]:
    """Process a single slide directory for outlier detection.

    Detects outlier tiles and saves a ``metadata_clean.parquet`` file alongside
    the existing ``metadata.parquet``. The new file contains all original metric
    columns plus extra columns:

    * ``is_outlier`` – boolean flag indicating whether the tile is an outlier.
    * ``method`` – the outlier detection method used.
    * ``outlier_score`` – continuous anomaly score (calibrate mode only).

    Args:
        slide_dir: Path to slide directory containing metadata and tiles
        mode: Outlier detection mode ('calibrate' or 'clustering')
        outlier_frac: Fraction of tiles to label as outliers (calibrate mode only)
        num_clusters: Number of clusters for k-means

    Returns:
        Tuple of (slide_dir, exception). Exception is None if successful.
    """
    import polars as pl

    from histoslice.utils import OutlierDetector

    try:
        # Find metadata file
        if not (slide_dir / "metadata.parquet").exists():
            return slide_dir, ValueError(f"No metadata file found in {slide_dir}")

        # Load metadata
        detector = OutlierDetector.from_parquet(slide_dir / "metadata.parquet")

        if mode == "calibrate":
            scores, is_outlier = _calibrate_slide_outliers(
                detector.dataframe, outlier_frac=outlier_frac
            )
            import numpy as np

            df = detector.dataframe.with_columns(
                [
                    pl.Series("is_outlier", is_outlier.astype(bool)),
                    pl.lit(mode).alias("method"),
                    pl.Series("outlier_score", scores.astype(np.float64)),
                ]
            )
        else:
            # clustering mode
            clusters = detector.cluster_kmeans(num_clusters=num_clusters)

            # Cluster 0 contains outliers after reordering by distance from mean center.
            # The cluster_kmeans method orders clusters by distance from the mean cluster
            # center, so cluster 0 is the most distant (likely outliers).
            outlier_mask = clusters == 0

            # Write metadata_clean.parquet with is_outlier and method columns.
            df = detector.dataframe.with_columns(
                [
                    pl.Series("is_outlier", outlier_mask),
                    pl.lit(mode).alias("method"),
                ]
            )

        df.write_parquet(slide_dir / "metadata_clean.parquet")

        return slide_dir, None

    except Exception as e:
        return slide_dir, e


def _build_calibrate_features(df: "pl.DataFrame") -> "np.ndarray":
    """Build a feature matrix for calibrate outlier detection from tile metadata.

    For each channel (gray, red, green, blue, saturation, brightness) derives
    five quantile-based shape features: range, high tail, low tail, skew, and
    median.  Also appends ``laplacian_std`` when present.

    Args:
        df: Polars dataframe with tile metadata quantile columns.

    Returns:
        2-D float64 array of shape ``(n_tiles, n_features)``.

    Raises:
        ValueError: No usable quantile columns found in the dataframe.
    """
    import numpy as np

    channels = ["gray", "red", "green", "blue", "saturation", "brightness"]
    cols = []
    for c in channels:
        required = [f"{c}_q5", f"{c}_q10", f"{c}_q50", f"{c}_q90", f"{c}_q95"]
        if not all(col in df.columns for col in required):
            continue
        v5 = df[f"{c}_q5"].to_numpy().astype(np.float64)
        v10 = df[f"{c}_q10"].to_numpy().astype(np.float64)
        v50 = df[f"{c}_q50"].to_numpy().astype(np.float64)
        v90 = df[f"{c}_q90"].to_numpy().astype(np.float64)
        v95 = df[f"{c}_q95"].to_numpy().astype(np.float64)
        cols.extend(
            [
                v95 - v5,  # range
                v95 - v90,  # tail_hi
                v10 - v5,  # tail_lo
                (v90 - v50) - (v50 - v10),  # skew
                v50,  # median
            ]
        )
    if "laplacian_std" in df.columns:
        cols.append(df["laplacian_std"].to_numpy().astype(np.float64))
    if not cols:
        raise ValueError("No usable feature columns found for calibrate mode.")
    return np.column_stack(cols)


def _calibrate_slide_outliers(
    df: "pl.DataFrame",
    *,
    outlier_frac: float,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Compute MCD-based outlier scores and binary labels for a slide.

    Uses Minimum Covariance Determinant (MCD) robust covariance fitting to
    compute Mahalanobis distances as anomaly scores.  Heavy outliers are
    optionally trimmed before fitting to prevent distorting the model.

    Args:
        df: Polars dataframe with tile metadata.
        outlier_frac: Fraction of tiles to label as outliers.

    Returns:
        Tuple ``(scores, is_outlier)`` where *scores* are float64 Mahalanobis
        distances and *is_outlier* is a boolean array.
    """
    import numpy as np
    from sklearn.covariance import MinCovDet

    X = _build_calibrate_features(df)
    n_samples, n_features = X.shape

    # Deterministic light trimming to exclude extreme artifacts before fitting.
    fit_mask = np.ones(n_samples, dtype=bool)
    if "saturation_q95" in df.columns:
        sat = df["saturation_q95"].to_numpy().astype(np.float64)
        fit_mask &= sat <= np.quantile(sat, 0.98)
    if "brightness_q5" in df.columns:
        brt = df["brightness_q5"].to_numpy().astype(np.float64)
        fit_mask &= brt >= np.quantile(brt, 0.01)

    # Fall back to all tiles if trimming removes too many.
    if fit_mask.sum() < 0.3 * n_samples:
        fit_mask = np.ones(n_samples, dtype=bool)

    X_fit = X[fit_mask]

    # Subsample for very large slides (keep deterministic via fixed seed).
    _MAX_FIT = 5000
    if len(X_fit) > _MAX_FIT:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_fit), size=_MAX_FIT, replace=False)
        X_fit = X_fit[idx]

    scores: np.ndarray
    min_samples = max(n_features + 1, 10)
    if len(X_fit) >= min_samples:
        try:
            mcd = MinCovDet(random_state=42)
            mcd.fit(X_fit)
            scores = mcd.mahalanobis(X)
        except Exception:
            scores = _fallback_scores(X)
    else:
        scores = _fallback_scores(X)

    threshold = np.quantile(scores, 1.0 - outlier_frac)
    is_outlier = scores > threshold
    return scores, is_outlier


def _fallback_scores(X: "np.ndarray") -> "np.ndarray":
    """Return absolute z-scores on the first feature as a fallback."""
    import numpy as np

    col = X[:, 0]
    std = col.std()
    if std == 0 or len(col) < 2:
        return np.zeros(len(col))
    return np.abs((col - col.mean()) / std)


def filter_slide_paths(  # noqa
    *,
    all_paths: list[Path],
    parent_dir: Path,
    overwrite: bool,
    overwrite_unfinished: bool,
) -> list[Path]:
    # Get processed and unprocessed slides.
    output, processed, interrupted = ([], [], [])
    for path in all_paths:
        if not path.is_file():
            continue
        output_dir = parent_dir / path.name.removesuffix(path.suffix)
        if output_dir.exists():
            if (output_dir / "metadata.parquet").exists():
                processed.append(path)
            else:
                interrupted.append(path)
        else:
            output.append(path)
    # Add processed/unfinished to output.
    if overwrite:
        output += processed + interrupted
        if len(processed + interrupted) > 0:
            warning(f"Overwriting {len(processed + interrupted)} slide outputs.")
    elif overwrite_unfinished:
        output += interrupted
        if len(interrupted) > 0:
            warning(f"Overwriting {len(interrupted)} unfinished slide outputs.")
    elif len(processed) > 0:
        info(f"Skipping {len(processed)} processed slides.")
    # Verbose.
    if len(output) == 0:
        error("No slides to process.")
    info(f"Processing {len(output)} slides.")
    return output


def cut_slide(
    path: Path,
    *,
    reader_kwargs: ReaderKwargs,
    max_dimension: int,
    tissue_kwargs: TissueKwargs,
    tile_kwargs: TileKwargs,
    save_kwargs: SaveKwargs,
) -> tuple[Path, Optional[Exception], int]:
    try:
        reader = SlideReader(path, **reader_kwargs)
        if tissue_kwargs["level"] is None:
            tissue_kwargs["level"] = reader.level_from_max_dimension(max_dimension)
        threshold, tissue_mask = reader.get_tissue_mask(**tissue_kwargs)
        coords = reader.get_tile_coordinates(tissue_mask=tissue_mask, **tile_kwargs)
        _, failures = reader.save_regions(
            coordinates=coords,
            threshold=threshold,
            tissue_mask=tissue_mask,
            **save_kwargs,
        )
        failure_count = len(failures)
    except Exception as e:  # noqa
        return path, e, 0
    return path, None, failure_count


def warning(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.YELLOW, bold=True)


def info(msg: str) -> None:
    typer.secho(msg, fg=typer.colors.CYAN, bold=True)


def error(msg: str, exit_integer: int = 1) -> NoReturn:
    """Display error message and exit."""
    typer.secho(msg, fg=typer.colors.RED, bold=True, err=True)
    sys.exit(exit_integer)


def main() -> None:
    app()
