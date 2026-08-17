"""High-level, one-call convenience API."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from histoslice.export import ExportResult, export_tiles
from histoslice.slide import Slide
from histoslice.tiles import Region, TileSpec, filter_by_tissue, tile_regions
from histoslice.tissue import tissue_mask as detect_tissue_mask


def slice_slide(
    path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    tile_size: Union[int, tuple[int, int]] = 512,
    overlap: float = 0.0,
    max_background: float = 0.5,
    target_mpp: Optional[Union[float, tuple[float, float]]] = None,
    mpp: Optional[tuple[float, float]] = None,
    threshold: Optional[int] = None,
    multiplier: float = 1.05,
    sigma: float = 1.0,
    tissue_level: Optional[int] = None,
    image_format: str = "jpeg",
    quality: int = 80,
    save_masks: bool = False,
    save_metrics: bool = False,
    save_thumbnails: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> ExportResult:
    """Slice a whole slide image into tile images.

    This is the convenience entry point combining the low-level building blocks:
    read a low-resolution level, detect tissue, generate a tile grid, filter by
    tissue content, and export tiles - see `histoslice.slide.Slide`,
    `histoslice.tissue.tissue_mask`, `histoslice.tiles.tile_regions`,
    `histoslice.tiles.filter_by_tissue` and `histoslice.export.export_tiles` to
    compose a custom pipeline instead.

    Args:
        path: Path to the slide image.
        output_dir: Parent output directory; tiles are written to
            `output_dir/{slide_name}/`.
        tile_size: Output tile size as `(width, height)`, or an int for square
            tiles. Defaults to 512.
        overlap: Overlap between neighbouring tiles, in range [0, 1). Defaults to 0.0.
        max_background: Maximum background fraction allowed per tile. Defaults to 0.5.
        target_mpp: Target microns per pixel for the output tiles, as a single value
            (isotropic) or `(mpp_x, mpp_y)`. If None, tiles are extracted at the
            slide's native resolution. Defaults to None.
        mpp: Override slide microns per pixel as `(mpp_x, mpp_y)`. Required if
            `target_mpp` is set and the slide has no mpp metadata. Defaults to None.
        threshold: Tissue detection threshold. If None, Otsu's method is used.
            Defaults to None.
        multiplier: Multiplier applied to the Otsu threshold. Ignored if `threshold`
            is set. Defaults to 1.05.
        sigma: Gaussian blur sigma for tissue detection. Defaults to 1.0.
        tissue_level: Pyramid level used for tissue detection. If None, picked
            automatically. Defaults to None.
        image_format: Output image file format. Defaults to "jpeg".
        quality: JPEG compression quality. Defaults to 80.
        save_masks: Save a tissue mask alongside each tile. Defaults to False.
        save_metrics: Save per-tile image metrics into the metadata. Defaults to False.
        save_thumbnails: Save slide thumbnails. Defaults to True.
        overwrite: Overwrite existing output for this slide. Defaults to False.
        verbose: Show a progress bar. Defaults to True.

    Raises:
        ValueError: `target_mpp` was given but slide mpp is not available.

    Returns:
        `ExportResult` with the saved metadata, per-tile failures, and output
        directory.
    """
    slide = Slide(path, mpp=mpp)

    tissue_level = (
        slide.level_from_max_dimension() if tissue_level is None else tissue_level
    )
    threshold, mask = detect_tissue_mask(
        slide.read_level(tissue_level),
        threshold=threshold,
        multiplier=multiplier,
        sigma=sigma,
    )

    size = (tile_size, tile_size) if isinstance(tile_size, int) else tile_size
    target_mpp_xy = (
        None
        if target_mpp is None
        else (target_mpp, target_mpp)
        if isinstance(target_mpp, (int, float))
        else target_mpp
    )
    spec = TileSpec(size=size, mpp=target_mpp_xy)
    regions = tile_regions(
        slide.dimensions, spec.level0_size(slide.mpp), overlap=overlap
    )
    regions = filter_by_tissue(
        regions, mask, slide_dimensions=slide.dimensions, max_background=max_background
    )

    return export_tiles(
        slide,
        regions,
        Path(output_dir) / slide.name,
        tile_size=size,
        threshold=threshold,
        tissue_mask=mask,
        image_format=image_format,
        quality=quality,
        save_masks=save_masks,
        save_metrics=save_metrics,
        save_thumbnails=save_thumbnails,
        overwrite=overwrite,
        verbose=verbose,
    )


def mean_and_std(
    slide: Slide,
    regions: list[Region],
    *,
    level: int = 0,
    max_samples: int = 1000,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Estimate per-channel mean and standard deviation from a sample of tiles."""
    from histoslice.functional._mean_std import get_mean_and_std_from_images

    if len(regions) > max_samples:
        rng = np.random.default_rng()
        regions = [
            regions[i] for i in rng.choice(len(regions), max_samples, replace=False)
        ]
    return get_mean_and_std_from_images(
        slide.read_region(r, level=level) for r in regions
    )
