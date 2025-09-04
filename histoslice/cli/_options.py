# histoslice/cli/options.py
from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import typer
from typing_extensions import Annotated


# ---- Input / Output ----
def io_opts(
    input_pattern: Annotated[
        str,
        typer.Option(
            "--input",
            "-i",
            help="File pattern to glob.",
            rich_help_panel="Input/output",
        ),
    ],
    parent_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Parent directory for all outputs.",
            dir_okay=True,
            file_okay=False,
            rich_help_panel="Input/output",
        ),
    ],
    backend: Annotated[
        Optional[str],
        typer.Option(
            "--backend",
            help="Backend for reading slides.",
            case_sensitive=False,
            show_default="automatic",
            rich_help_panel="Input/output",
        ),
    ] = None,
) -> Dict:
    paths: List[Path] = [
        Path(p) for p in glob.glob(input_pattern, recursive=True) if Path(p).is_file()
    ]
    if not paths:
        typer.secho(
            f"Found no files matching pattern '{input_pattern}'.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=2)
    parent_dir.mkdir(parents=True, exist_ok=True)
    typer.secho(
        f"Found {len(paths)} files matching pattern '{input_pattern}'.",
        fg=typer.colors.CYAN,
    )
    return {"paths": paths, "parent_dir": parent_dir, "backend": backend}


# ---- Tile extraction ----
def tile_opts(
    level: Annotated[
        int,
        typer.Option(
            "--level",
            "-l",
            min=0,
            help="Pyramid level for tile extraction.",
            rich_help_panel="Tile extraction",
        ),
    ] = 0,
    width: Annotated[
        int,
        typer.Option(
            "--width",
            "-w",
            min=0,
            help="Tile width.",
            rich_help_panel="Tile extraction",
        ),
    ] = 640,
    height: Annotated[
        Optional[int],
        typer.Option(
            "--height",
            "-h",
            min=0,
            show_default="width",
            help="Tile height.",
            rich_help_panel="Tile extraction",
        ),
    ] = None,
    overlap: Annotated[
        float,
        typer.Option(
            "--overlap",
            "-n",
            min=0.0,
            max=1.0,
            help="Overlap between neighbouring tiles.",
            rich_help_panel="Tile extraction",
        ),
    ] = 0.0,
    max_background: Annotated[
        float,
        typer.Option(
            "--max-background",
            "-b",
            min=0.0,
            max=1.0,
            help="Maximum background for tiles.",
            rich_help_panel="Tile extraction",
        ),
    ] = 0.75,
    in_bounds: Annotated[
        bool,
        typer.Option(
            "--in-bounds",
            show_default="False",
            help="Do not allow tiles to go out-of-bounds.",
            rich_help_panel="Tile extraction",
        ),
    ] = False,
) -> Dict:
    return {
        "level": level,
        "width": width,
        "height": height,
        "overlap": overlap,
        "max_background": max_background,
        "in_bounds": in_bounds,
    }


# ---- Tissue detection ----
def tissue_opts(
    threshold: Annotated[
        Optional[int],
        typer.Option(
            "--threshold",
            "-t",
            min=0,
            max=255,
            show_default="Otsu",
            help="Global thresholding value.",
            rich_help_panel="Tissue detection",
        ),
    ] = None,
    multiplier: Annotated[
        float,
        typer.Option(
            "--multiplier",
            "-x",
            min=0.0,
            help="Multiplier for Otsu's threshold.",
            rich_help_panel="Tissue detection",
        ),
    ] = 1.05,
    tissue_level: Annotated[
        Optional[int],
        typer.Option(
            "--tissue-level",
            min=0,
            show_default="max_dimension",
            help="Pyramid level for tissue detection.",
            rich_help_panel="Tissue detection",
        ),
    ] = None,
    max_dimension: Annotated[
        int,
        typer.Option(
            "--max-dimension",
            min=0,
            help="Maximum dimension for tissue detection.",
            rich_help_panel="Tissue detection",
        ),
    ] = 8192,
    sigma: Annotated[
        float,
        typer.Option(
            "--sigma",
            min=0.0,
            help="Sigma for gaussian blurring.",
            rich_help_panel="Tissue detection",
        ),
    ] = 1.0,
) -> Dict:
    return {
        "threshold": threshold,
        "multiplier": multiplier,
        "tissue_level": tissue_level,
        "max_dimension": max_dimension,
        "sigma": sigma,
    }


# ---- Tile saving ----
def save_opts(
    save_metrics: Annotated[
        bool,
        typer.Option(
            "--metrics",
            show_default="False",
            help="Save image metrics.",
            rich_help_panel="Tile saving",
        ),
    ] = False,
    save_masks: Annotated[
        bool,
        typer.Option(
            "--masks",
            show_default="False",
            help="Save tissue masks.",
            rich_help_panel="Tile saving",
        ),
    ] = False,
    save_thumbnails: Annotated[
        bool,
        typer.Option(
            "--thumbnails",
            show_default="False",
            help="Save thumbnails of tiles.",
            rich_help_panel="Tile saving",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            "-z",
            show_default="False",
            help="Overwrite any existing slide outputs.",
            rich_help_panel="Tile saving",
        ),
    ] = False,
    overwrite_unfinished: Annotated[
        bool,
        typer.Option(
            "--unfinished",
            "-u",
            show_default="False",
            help="Overwrite only if metadata is missing.",
            rich_help_panel="Tile saving",
        ),
    ] = False,
    image_format: Annotated[
        str,
        typer.Option(
            "--image-format",
            help="File format for tile images.",
            rich_help_panel="Tile saving",
        ),
    ] = "jpeg",
    quality: Annotated[
        int,
        typer.Option(
            "--quality",
            min=0,
            max=100,
            show_default=True,
            help="Quality for jpeg-compression.",
            rich_help_panel="Tile saving",
        ),
    ] = 80,
    num_workers: Annotated[
        Optional[int],
        typer.Option(
            "--num-workers",
            "-j",
            min=0,
            show_default="CPU-count",
            help="Number of data saving workers.",
            rich_help_panel="Tile saving",
        ),
    ] = None,
) -> Dict:
    return {
        "save_metrics": save_metrics,
        "save_masks": save_masks,
        "save_thumbnails": save_thumbnails,
        "overwrite": overwrite,
        "overwrite_unfinished": overwrite_unfinished,
        "image_format": image_format,
        "quality": quality,
        "num_workers": num_workers,
    }


class ReaderKwargs(TypedDict):
    backend: Optional[str]


class TissueKwargs(TypedDict, total=False):
    level: Optional[int]
    threshold: Optional[int]
    multiplier: float
    sigma: float


class TileKwargs(TypedDict):
    width: int
    height: Optional[int]
    overlap: float
    out_of_bounds: bool
    max_background: float


class SaveKwargs(TypedDict):
    parent_dir: Path
    level: int
    save_metrics: bool
    save_masks: bool
    save_thumbnails: bool
    image_format: str
    quality: int
    use_csv: bool
    raise_exception: bool
    num_workers: int
    overwrite: bool
    verbose: bool


class CutSlideKwargs(TypedDict):
    reader_kwargs: ReaderKwargs
    max_dimension: int
    tissue_kwargs: TissueKwargs
    tile_kwargs: TileKwargs
    save_kwargs: SaveKwargs
