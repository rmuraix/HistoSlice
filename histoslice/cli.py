"""Command line interface - a thin wrapper around the `histoslice` library API."""

from __future__ import annotations

import functools
import glob
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NoReturn, Optional

import typer
from tqdm import tqdm
from typing_extensions import Annotated

from histoslice.api import slice_slide

DEFAULT_START_METHOD = "spawn"

app = typer.Typer(
    name="histoslice", help="Tools for preprocessing histological slide images."
)


@app.command("slice")
def slice_command(
    input_pattern: Annotated[
        str, typer.Option("--input", "-i", help="File pattern to glob.")
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output", "-o", help="Parent directory for all outputs.", file_okay=False
        ),
    ],
    tile_size: Annotated[
        int, typer.Option("--width", "-w", min=1, help="Tile size in pixels.")
    ] = 512,
    overlap: Annotated[
        float,
        typer.Option("--overlap", "-n", min=0.0, max=1.0, help="Tile overlap."),
    ] = 0.0,
    max_background: Annotated[
        float,
        typer.Option(
            "--max-background",
            "-b",
            min=0.0,
            max=1.0,
            help="Maximum background per tile.",
        ),
    ] = 0.75,
    target_mpp: Annotated[
        Optional[float],
        typer.Option(
            "--target-mpp",
            min=0.0,
            help="Target microns per pixel for the output tiles.",
        ),
    ] = None,
    mpp: Annotated[
        Optional[float],
        typer.Option(
            "--mpp", help="Microns per pixel override (assumes square pixels)."
        ),
    ] = None,
    threshold: Annotated[
        Optional[int],
        typer.Option(
            "--threshold",
            "-t",
            min=0,
            max=255,
            show_default="Otsu",
            help="Tissue threshold.",
        ),
    ] = None,
    multiplier: Annotated[
        float,
        typer.Option("--multiplier", "-x", min=0.0, help="Otsu threshold multiplier."),
    ] = 1.05,
    sigma: Annotated[
        float, typer.Option("--sigma", min=0.0, help="Gaussian blur sigma.")
    ] = 1.0,
    tissue_level: Annotated[
        Optional[int],
        typer.Option(
            "--tissue-level",
            min=0,
            show_default="auto",
            help="Pyramid level for tissue detection.",
        ),
    ] = None,
    save_metrics: Annotated[
        bool, typer.Option("--metrics", help="Save image metrics.")
    ] = False,
    save_masks: Annotated[
        bool, typer.Option("--masks", help="Save tissue masks.")
    ] = False,
    save_thumbnails: Annotated[
        bool, typer.Option("--thumbnails", help="Save slide thumbnails.")
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", "-z", help="Overwrite existing slide outputs."),
    ] = False,
    overwrite_unfinished: Annotated[
        bool,
        typer.Option(
            "--unfinished", "-u", help="Overwrite only if metadata is missing."
        ),
    ] = False,
    image_format: Annotated[
        str, typer.Option("--image-format", help="Tile image file format.")
    ] = "jpeg",
    quality: Annotated[
        int, typer.Option("--quality", min=0, max=100, help="JPEG compression quality.")
    ] = 80,
    num_workers: Annotated[
        Optional[int],
        typer.Option(
            "--num-workers",
            "-j",
            min=0,
            show_default="CPU-count",
            help="Parallel slides.",
        ),
    ] = None,
) -> None:
    """Extract tile images from histological slides."""
    all_paths = [
        Path(p) for p in glob.glob(input_pattern, recursive=True) if Path(p).is_file()
    ]
    if not all_paths:
        error(f"Found no files matching pattern '{input_pattern}'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    info(f"Found {len(all_paths)} files matching pattern '{input_pattern}'.")

    paths = filter_slide_paths(
        all_paths=all_paths,
        output_dir=output_dir,
        overwrite=overwrite,
        overwrite_unfinished=overwrite_unfinished,
    )

    kwargs = {
        "tile_size": tile_size,
        "overlap": overlap,
        "max_background": max_background,
        "target_mpp": target_mpp,
        "mpp": None if mpp is None else (mpp, mpp),
        "threshold": threshold,
        "multiplier": multiplier,
        "sigma": sigma,
        "tissue_level": tissue_level,
        "image_format": image_format,
        "quality": quality,
        "save_masks": save_masks,
        "save_metrics": save_metrics,
        "save_thumbnails": save_thumbnails,
        "overwrite": True,  # already filtered above
        "verbose": False,  # common progress bar below
    }

    effective_workers = (os.cpu_count() or 1) if num_workers is None else num_workers
    if effective_workers == 0:
        for path in paths:
            _report(*slice_one(path, output_dir, kwargs))
    else:
        ctx = mp.get_context(DEFAULT_START_METHOD)
        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as pool:
            func = functools.partial(slice_one, output_dir=output_dir, kwargs=kwargs)
            futures = {pool.submit(func, path): path for path in paths}
            for future in tqdm(
                as_completed(futures), desc="Cutting slides", total=len(paths)
            ):
                _report(*future.result())


def slice_one(
    path: Path, output_dir: Path, kwargs: dict
) -> tuple[Path, Optional[Exception], int]:
    """Slice a single slide; runs in a worker process when `num_workers > 0`."""
    try:
        result = slice_slide(path, output_dir, **kwargs)
    except Exception as e:  # noqa
        return path, e, 0
    return path, None, len(result.failures)


def _report(path: Path, exception: Optional[Exception], num_failed: int) -> None:
    if exception is not None:
        warning(f"Could not process {path} due to exception: {exception!r}")
    elif num_failed:
        warning(
            f"Slide {path} completed with {num_failed} failed tile(s). "
            "See failures.json for details."
        )


def filter_slide_paths(
    *,
    all_paths: list[Path],
    output_dir: Path,
    overwrite: bool,
    overwrite_unfinished: bool,
) -> list[Path]:
    """Split slide paths into unprocessed / already-processed / interrupted, and pick
    which ones to (re)process based on `overwrite`/`overwrite_unfinished`."""
    output, processed, interrupted = [], [], []
    for path in all_paths:
        slide_dir = output_dir / path.name.removesuffix(path.suffix)
        if not slide_dir.exists():
            output.append(path)
        elif (slide_dir / "metadata.parquet").exists():
            processed.append(path)
        else:
            interrupted.append(path)
    if overwrite:
        output += processed + interrupted
        if processed or interrupted:
            warning(f"Overwriting {len(processed) + len(interrupted)} slide outputs.")
    elif overwrite_unfinished:
        output += interrupted
        if interrupted:
            warning(f"Overwriting {len(interrupted)} unfinished slide outputs.")
    elif processed:
        info(f"Skipping {len(processed)} processed slides.")
    if not output:
        error("No slides to process.")
    info(f"Processing {len(output)} slides.")
    return output


@app.command("clean")
def clean_command(
    input_pattern: Annotated[
        str,
        typer.Option(
            "--input", "-i", help="Directory pattern to glob for slide outputs."
        ),
    ],
    mode: Annotated[
        str, typer.Option("--mode", "-m", help="Outlier detection mode.")
    ] = "clustering",
    num_clusters: Annotated[
        int,
        typer.Option("--num-clusters", "-k", min=2, help="Number of k-means clusters."),
    ] = 4,
    num_workers: Annotated[
        Optional[int],
        typer.Option(
            "--num-workers",
            "-j",
            min=0,
            show_default="CPU-count",
            help="Parallel slides.",
        ),
    ] = None,
) -> None:
    """Detect outlier tile images using clustering and save metadata_clean.parquet."""
    if mode != "clustering":
        error(f"Unknown mode '{mode}'. Currently only 'clustering' is supported.")

    slide_dirs = [
        Path(p)
        for p in glob.glob(input_pattern, recursive=True)
        if Path(p).is_dir() and (Path(p) / "metadata.parquet").exists()
    ]
    if not slide_dirs:
        error(
            f"Found no slide directories with metadata matching pattern '{input_pattern}'."
        )
    info(f"Found {len(slide_dirs)} slide(s) to process.")

    effective_workers = (os.cpu_count() or 1) if num_workers is None else num_workers
    if effective_workers == 0:
        for slide_dir in slide_dirs:
            _, exception = process_slide_outliers(
                slide_dir, mode=mode, num_clusters=num_clusters
            )
            if exception is not None:
                warning(
                    f"Could not process {slide_dir} due to exception: {exception!r}"
                )
    else:
        ctx = mp.get_context(DEFAULT_START_METHOD)
        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as pool:
            func = functools.partial(
                process_slide_outliers, mode=mode, num_clusters=num_clusters
            )
            futures = {
                pool.submit(func, slide_dir): slide_dir for slide_dir in slide_dirs
            }
            for future in tqdm(
                as_completed(futures), desc="Cleaning slides", total=len(slide_dirs)
            ):
                slide_dir, exception = future.result()
                if exception is not None:
                    warning(
                        f"Could not process {slide_dir} due to exception: {exception!r}"
                    )


def process_slide_outliers(
    slide_dir: Path, *, mode: str, num_clusters: int
) -> tuple[Path, Optional[Exception]]:
    """Detect outlier tiles for one slide directory and write metadata_clean.parquet.

    Adds two columns to the existing `metadata.parquet`: `is_outlier` (bool) and
    `method` (the detection mode used), saved as `metadata_clean.parquet`.
    """
    import polars as pl

    from histoslice.utils import OutlierDetector

    try:
        detector = OutlierDetector.from_parquet(slide_dir / "metadata.parquet")
        clusters = detector.cluster_kmeans(num_clusters=num_clusters)
        # cluster_kmeans orders clusters by distance from the mean center, so
        # cluster 0 is the most distant (likely outliers).
        outlier_mask = clusters == 0
        df = detector.dataframe.with_columns(
            [pl.Series("is_outlier", outlier_mask), pl.lit(mode).alias("method")]
        )
        df.write_parquet(slide_dir / "metadata_clean.parquet")
        return slide_dir, None
    except Exception as e:  # noqa
        return slide_dir, e


def warning(msg: str) -> None:
    """Print `msg` to stdout in bold yellow."""
    typer.secho(msg, fg=typer.colors.YELLOW, bold=True)


def info(msg: str) -> None:
    """Print `msg` to stdout in bold cyan."""
    typer.secho(msg, fg=typer.colors.CYAN, bold=True)


def error(msg: str, exit_code: int = 1) -> NoReturn:
    """Print `msg` to stderr in bold red and exit with `exit_code`."""
    typer.secho(msg, fg=typer.colors.RED, bold=True, err=True)
    sys.exit(exit_code)


def main() -> None:
    """CLI entry point (`histoslice` console script)."""
    app()
