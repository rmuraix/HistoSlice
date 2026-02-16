"""CLI interface for cutting slides into small tile images."""

import functools
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, NoReturn, Optional

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
            "backend": cfg.backend,
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
            "microns": cfg.microns,
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
    """Detect and remove outlier tile images using clustering."""
    import glob
    from pathlib import Path

    # Validate mode
    if clean["mode"] != "clustering":
        error(
            f"Unknown mode '{clean['mode']}'. Currently only 'clustering' is supported."
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
        "num_clusters": clean["num_clusters"],
        "delete": clean["delete"],
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
    num_clusters: int,
    delete: bool,
) -> tuple[Path, Optional[Exception]]:
    """Process a single slide directory for outlier detection and removal.

    Args:
        slide_dir: Path to slide directory containing metadata and tiles
        mode: Outlier detection mode (currently only 'clustering')
        num_clusters: Number of clusters for k-means
        delete: If True, delete outliers; if False, move to 'outliers' subdirectory

    Returns:
        Tuple of (slide_dir, exception). Exception is None if successful.
    """
    import shutil
    from pathlib import Path

    from histoslice.utils import OutlierDetector

    try:
        # Find metadata file
        metadata_path = None
        if (slide_dir / "metadata.parquet").exists():
            metadata_path = slide_dir / "metadata.parquet"
        else:
            return slide_dir, ValueError(f"No metadata file found in {slide_dir}")

        # Load metadata
        detector = OutlierDetector.from_parquet(metadata_path)

        # Perform clustering
        clusters = detector.cluster_kmeans(num_clusters=num_clusters)

        # Cluster 0 contains outliers after reordering by distance from mean center.
        # The cluster_kmeans method orders clusters by distance from the mean cluster
        # center, so cluster 0 is the most distant (likely outliers).
        outlier_mask = clusters == 0
        num_outliers = outlier_mask.sum()

        if num_outliers == 0:
            return slide_dir, None

        # Get paths of outlier tiles
        outlier_paths = detector.paths[outlier_mask]

        # Process each outlier tile
        for tile_path in outlier_paths:
            tile_path = Path(tile_path)
            if not tile_path.exists():
                continue

            if delete:
                # Delete the file
                tile_path.unlink()
            else:
                # Move to outliers subdirectory
                outliers_dir = slide_dir / "outliers"
                outliers_dir.mkdir(exist_ok=True)
                destination = outliers_dir / tile_path.name
                shutil.move(str(tile_path), str(destination))

        return slide_dir, None

    except Exception as e:
        return slide_dir, e


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
