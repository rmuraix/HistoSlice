"""CLI interface for cutting slides into small tile images."""

import functools
import os
import sys
from pathlib import Path
from typing import Dict, NoReturn, Optional

import mpire
import typer
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
        "reader_kwargs": {"backend": cfg.backend},
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
            "use_csv": cfg.use_csv,
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
            _, exception = cut_slide(path, **kwargs)
            if isinstance(exception, Exception):
                warning(f"Could not process {path} due to exception: {exception!r}")
    else:
        with mpire.WorkerPool(n_jobs=effective_workers) as pool:
            for path, exception in pool.imap(
                func=functools.partial(cut_slide, **kwargs),
                iterable_of_args=paths,
                progress_bar=True,
                progress_bar_options={"desc": "Cutting slides"},
            ):
                if isinstance(exception, Exception):
                    warning(f"Could not process {path} due to exception: {exception!r}")


@app.command("clean")
def clean_tiles(
    clean: Annotated[Dict, Depends(clean_opts)],
) -> None:
    """Detect and remove outlier tile images using clustering."""
    import glob
    import shutil
    from pathlib import Path

    from histoslice.utils import OutlierDetector

    # Validate mode
    if clean["mode"] != "clustering":
        error(
            f"Unknown mode '{clean['mode']}'. Currently only 'clustering' is supported."
        )

    # Find metadata files
    metadata_paths = glob.glob(clean["input_pattern"], recursive=True)
    if not metadata_paths:
        error(f"Found no metadata files matching pattern '{clean['input_pattern']}'.")

    info(f"Found {len(metadata_paths)} metadata file(s) to process.")

    total_outliers = 0
    total_deleted = 0
    total_moved = 0

    for metadata_path in metadata_paths:
        metadata_path = Path(metadata_path)

        try:
            # Load metadata
            if metadata_path.suffix == ".parquet":
                detector = OutlierDetector.from_parquet(metadata_path)
            elif metadata_path.suffix == ".csv":
                detector = OutlierDetector.from_csv(metadata_path)
            else:
                warning(f"Skipping unsupported file format: {metadata_path}")
                continue

            info(f"Processing {metadata_path.name} with {len(detector)} tiles...")

            # Perform clustering
            clusters = detector.cluster_kmeans(num_clusters=clean["num_clusters"])

            # Assume cluster 0 contains outliers (smallest cluster by distance)
            outlier_mask = clusters == 0
            num_outliers = outlier_mask.sum()

            if num_outliers == 0:
                info(f"No outliers detected in {metadata_path.name}")
                continue

            info(f"Detected {num_outliers} outlier tiles in cluster 0")
            total_outliers += num_outliers

            # Get paths of outlier tiles
            outlier_paths = detector.paths[outlier_mask]

            # Process each outlier tile
            for tile_path in outlier_paths:
                tile_path = Path(tile_path)
                if not tile_path.exists():
                    warning(f"Tile file not found: {tile_path}")
                    continue

                if clean["delete"]:
                    # Delete the file
                    tile_path.unlink()
                    total_deleted += 1
                else:
                    # Move to outliers subdirectory
                    outliers_dir = tile_path.parent.parent / "outliers"
                    outliers_dir.mkdir(exist_ok=True)
                    destination = outliers_dir / tile_path.name
                    shutil.move(str(tile_path), str(destination))
                    total_moved += 1

        except Exception as e:
            warning(f"Could not process {metadata_path} due to exception: {e!r}")
            continue

    # Summary
    if clean["delete"]:
        info(
            f"Detected {total_outliers} outliers across all slides, deleted {total_deleted} files."
        )
    else:
        info(
            f"Detected {total_outliers} outliers across all slides, moved {total_moved} files to 'outliers' subdirectories."
        )


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
            if (output_dir / "metadata.parquet").exists() or (
                output_dir / "metadata.csv"
            ).exists():
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
) -> tuple[Path, Optional[Exception]]:
    try:
        reader = SlideReader(path, **reader_kwargs)
        if tissue_kwargs["level"] is None:
            tissue_kwargs["level"] = reader.level_from_max_dimension(max_dimension)
        threshold, tissue_mask = reader.get_tissue_mask(**tissue_kwargs)
        coords = reader.get_tile_coordinates(tissue_mask=tissue_mask, **tile_kwargs)
        reader.save_regions(coordinates=coords, threshold=threshold, **save_kwargs)
    except Exception as e:  # noqa
        return path, e
    return path, None


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
