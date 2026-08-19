"""Saving tile regions (and thumbnails/masks/metrics) to disk."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import polars as pl
import tqdm

from histoslice.functional._draw import get_annotated_image
from histoslice.functional._imageio import write_image
from histoslice.functional._images import downscale_to_max_pixels, has_jpeg_support
from histoslice.functional._metrics import get_image_metrics, get_qc_metrics
from histoslice.slide import Slide
from histoslice.tiles import Region, get_downsample
from histoslice.tissue import downscale_for_thumbnail, tissue_mask as detect_tissue_mask

ERROR_NO_THRESHOLD = "Threshold argument is required to save masks/metrics."
ERROR_OUTPUT_DIR_IS_FILE = "Output directory exists but it is a file."
ERROR_CANNOT_OVERWRITE = "Output directory exists, but `overwrite=False`."


@dataclass
class ExportResult:
    """Result of `export_tiles`."""

    metadata: pl.DataFrame
    failures: list[dict[str, object]]
    output_dir: Path


def export_tiles(
    slide: Slide,
    regions: list[Region],
    output_dir: Union[str, Path],
    *,
    tile_size: Union[int, tuple[int, int]],
    names: Optional[list[str]] = None,
    region_dir: str = "tiles",
    threshold: Optional[int] = None,
    tissue_mask: Optional[np.ndarray] = None,
    save_masks: bool = False,
    save_metrics: bool = False,
    save_thumbnails: bool = True,
    thumbnail_level: Optional[int] = None,
    image_format: str = "jpeg",
    quality: int = 80,
    overwrite: bool = False,
    verbose: bool = True,
) -> ExportResult:
    """Read `regions` from `slide` and save them to `output_dir`.

    Args:
        slide: Slide to read tiles from.
        regions: Regions (level-0 coordinates) to extract.
        output_dir: Directory tiles (and thumbnails/masks/metadata) are saved to.
        tile_size: Output tile size as `(width, height)`, or an int for square
            tiles. `Slide.read_tile` automatically picks an efficient pyramid level
            and resizes to this exact size.
        names: Optional per-region name, used as a filename prefix (and as the
            `region_dir` thumbnail overlay text). Defaults to None.
        region_dir: Subdirectory tile images are saved to. Defaults to "tiles".
        threshold: Tissue detection threshold, required when `save_masks` or
            `save_metrics` is True. When set (even if both are False), minimal
            technical QC metrics (see `histoslice.functional.get_qc_metrics`)
            are also computed and saved - these are what `histoslice clean`
            needs, and don't require `save_metrics`. Defaults to None.
        tissue_mask: Tissue mask used for thumbnail visualisation. Defaults to None.
        save_masks: Save a tissue mask (`png`) alongside each tile. Defaults to False.
        save_metrics: Save the full set of exploratory per-tile image metrics
            into the metadata (see `histoslice.functional.get_image_metrics`).
            Defaults to False.
        save_thumbnails: Save slide thumbnails (plain, annotated, and tissue mask
            overlay). Defaults to True.
        thumbnail_level: Pyramid level for thumbnails. If None, picked automatically.
            Defaults to None.
        image_format: Output image file format. Defaults to "jpeg".
        quality: JPEG compression quality. Defaults to 80.
        overwrite: Overwrite `output_dir` if it already has content. Defaults to False.
        verbose: Show a progress bar. Defaults to True.

    Raises:
        ValueError: `save_masks`/`save_metrics` requested without `threshold`.
        NotADirectoryError: `output_dir` exists and is a file.
        ValueError: `output_dir` exists, has content, and `overwrite` is False.

    Returns:
        `ExportResult` with the saved metadata, per-tile failures, and output
        directory.
    """
    if (save_masks or save_metrics) and threshold is None:
        raise ValueError(ERROR_NO_THRESHOLD)
    size = (tile_size, tile_size) if isinstance(tile_size, int) else tile_size
    output_dir = _prepare_output_dir(Path(output_dir), overwrite=overwrite)
    image_format = _resolve_image_format(image_format)

    if save_thumbnails:
        _save_thumbnails(
            slide,
            regions,
            names,
            tissue_mask,
            output_dir=output_dir,
            region_dir=region_dir,
            level=thumbnail_level,
            image_format=image_format,
            quality=quality,
        )

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    progress = tqdm.tqdm(regions, desc=slide.name, disable=not verbose)
    for index, region in enumerate(progress):
        try:
            tile = slide.read_tile(region, size)
        except KeyboardInterrupt:
            raise
        except Exception as error:  # noqa
            failures.append({"xywh": region.xywh, "error": repr(error)})
            progress.set_postfix({"failed": len(failures)}, refresh=False)
            continue
        name = None if names is None else names[index]
        rows.append(
            _save_tile(
                tile,
                region,
                name=name,
                output_dir=output_dir,
                region_dir=region_dir,
                image_format=image_format,
                quality=quality,
                threshold=threshold,
                save_masks=save_masks,
                save_metrics=save_metrics,
            )
        )

    metadata = pl.DataFrame(rows)
    metadata.write_parquet(output_dir / "metadata.parquet")
    if failures:
        (output_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    return ExportResult(metadata=metadata, failures=failures, output_dir=output_dir)


def annotated_thumbnail(
    image: np.ndarray,
    regions: list[Region],
    *,
    slide_dimensions: tuple[int, int],
    names: Optional[list[str]] = None,
    highlight_first: bool = False,
    linewidth: int = 1,
) -> np.ndarray:
    """Draw `regions` on top of a (thumbnail) `image`.

    Args:
        image: Thumbnail image, typically read from a low-resolution pyramid level.
        regions: Regions to draw, in level-0 coordinates.
        slide_dimensions: Full slide dimensions `(height, width)` at level 0, used to
            downscale region coordinates onto `image`.
        names: Optional per-region text label. Defaults to None.
        highlight_first: Draw the first region with a different outline colour.
            Defaults to False.
        linewidth: Rectangle outline width. Defaults to 1.

    Returns:
        Annotated thumbnail image.
    """
    downsample = get_downsample(image, slide_dimensions)
    return get_annotated_image(
        image=image,
        coordinates=[r.xywh for r in regions],
        downsample=downsample,
        rectangle_width=linewidth,
        highlight_first=highlight_first,
        text_items=names,
    )


def _save_tile(
    tile: np.ndarray,
    region: Region,
    *,
    name: Optional[str],
    output_dir: Path,
    region_dir: str,
    image_format: str,
    quality: int,
    threshold: Optional[int],
    save_masks: bool,
    save_metrics: bool,
) -> dict[str, object]:
    filename = f"x{region.x}_y{region.y}_w{region.width}_h{region.height}"
    if name is not None:
        filename = f"{name}_{filename}"
    row: dict[str, object] = dict(zip("xywh", region.xywh))

    image_dir = output_dir / region_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{filename}.{image_format}"
    _save_image(tile, image_path, image_format=image_format, quality=quality)
    row["path"] = str(image_path.resolve())

    if save_masks or save_metrics or threshold is not None:
        __, mask = detect_tissue_mask(tile, threshold=threshold)
        if save_masks:
            mask_dir = output_dir / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / f"{filename}.png"
            write_image(mask, mask_path, image_format="png")
            row["mask_path"] = str(mask_path.resolve())
        if save_metrics:
            row.update(get_image_metrics(image=tile, tissue_mask=mask))
        # Minimal technical QC metrics are always computed (cheap, and
        # required by `histoslice clean`), independent of `save_metrics`.
        # Applied after `get_image_metrics` so its full-resolution `gray_std`
        # (used by QC's low-dynamic-range check) wins over the resized value
        # `get_image_metrics` computes for its own exploratory `gray_std`.
        row.update(get_qc_metrics(image=tile, tissue_mask=mask))
    return row


def _save_thumbnails(
    slide: Slide,
    regions: list[Region],
    names: Optional[list[str]],
    mask: Optional[np.ndarray],
    *,
    output_dir: Path,
    region_dir: str,
    level: Optional[int],
    image_format: str,
    quality: int,
) -> None:
    if level is None:
        level = slide.level_from_max_dimension()
    thumbnail = slide.read_level(level)
    thumbnail_small = downscale_for_thumbnail(thumbnail)
    if image_format == "png":
        thumbnail_small = downscale_to_max_pixels(thumbnail_small, max_pixels=300_000)

    _save_image(
        thumbnail_small,
        output_dir / f"thumbnail.{image_format}",
        image_format=image_format,
        quality=quality,
    )
    _save_image(
        annotated_thumbnail(
            thumbnail_small,
            regions,
            slide_dimensions=slide.dimensions,
            names=names,
            highlight_first=names is None,
        ),
        output_dir / f"thumbnail_{region_dir}.{image_format}",
        image_format=image_format,
        quality=quality,
    )
    if mask is not None:
        if mask.shape[:2] != thumbnail.shape[:2]:
            scale_h = thumbnail_small.shape[0] / thumbnail.shape[0]
            scale_w = thumbnail_small.shape[1] / thumbnail.shape[1]
            mask = cv2.resize(
                mask.astype(np.uint8),
                (
                    max(1, int(mask.shape[1] * scale_w)),
                    max(1, int(mask.shape[0] * scale_h)),
                ),
                interpolation=cv2.INTER_AREA,
            )
        _save_image(
            (255 - 255 * mask).astype(np.uint8),
            output_dir / f"thumbnail_tissue.{image_format}",
            image_format=image_format,
            quality=quality,
        )


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> Path:
    if output_dir.exists():
        if output_dir.is_file():
            raise NotADirectoryError(ERROR_OUTPUT_DIR_IS_FILE)
        if len(list(output_dir.iterdir())) > 0 and not overwrite:
            raise ValueError(ERROR_CANNOT_OVERWRITE)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_image_format(image_format: str) -> str:
    fmt = image_format.strip().lower()
    if fmt in ("jpg", "jpeg") and not has_jpeg_support():
        return "png"
    return fmt


def _save_image(
    image: np.ndarray, path: Path, *, image_format: str, quality: int
) -> None:
    write_image(image, path, image_format=image_format, quality=quality)
