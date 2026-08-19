# HistoSlice Documentation

[![PyPI - Version](https://img.shields.io/pypi/v/histoslice)](https://pypi.org/project/histoslice/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/histoslice)](https://pypi.org/project/histoslice/)

## Overview

`HistoSlice` makes is easy to prepare your histological slide images for deep learning models. You can easily cut large slide images into smaller tiles and then preprocess those tiles (remove tiles with shitty tissue, finger marks etc).

This project was forked from [HistoPrep](https://github.com/jopo666/HistoPrep), and further modified for additional features and improvements.

## Installation

Install install HistoSlice with `uv` or `pip`!

```bash
uv add histoslice
# or
pip install histoslice
```

Or install latest development version from GitHub:

```bash
uv add git+https://github.com/rmuraix/HistoSlice
# or
pip install git+https://github.com/rmuraix/HistoSlice
```

Alternatively, you can install the package from source:

```bash
git clone https://github.com/rmuraix/HistoSlice.git
cd HistoSlice
uv sync
```

## Usage

### Cutting Slide Images

Cut each slide image into smaller tile images.

`histoslice slice --help` will show you all available options. For Python API usage, see the [API documentation](api/public/slice_slide/).

=== "CLI"
    ```bash
    histoslice slice \
        --input './images/*.tiff' \
        --output ./tiles \
        --width 512 \
        --overlap 0.5 \
        --max-background 0.5 \
        --metrics \
        --thumbnails
    ```
=== "Python API"
    ```python
    from histoslice import slice_slide

    # Detect tissue, tile, filter, and save in one call.
    result = slice_slide(
        "./path/to/slide_image.tiff",
        "./tiles/",
        tile_size=512,
        overlap=0.5,
        max_background=0.5,
        save_metrics=True,
        save_thumbnails=True,
    )
    if result.failures:
        print(f"Some tiles failed: {len(result.failures)}")
    ```
=== "Python API (low-level)"
    ```python
    from histoslice import Slide, tissue_mask, tile_regions, filter_by_tissue, export_tiles

    # Read slide image.
    slide = Slide("./path/to/slide_image.tiff")
    # Detect tissue.
    threshold, mask = tissue_mask(slide.read_level(-1))
    # Extract overlapping tile regions with less than 50% background.
    regions = tile_regions(slide.dimensions, size=512, overlap=0.5)
    regions = filter_by_tissue(
        regions, mask, slide_dimensions=slide.dimensions, max_background=0.5
    )
    # Save tile images with image metrics for preprocessing.
    result = export_tiles(
        slide,
        regions,
        "./tiles/",
        tile_size=512,
        threshold=threshold,
        save_metrics=True,
        save_thumbnails=True,
    )
    if result.failures:
        print(f"Some tiles failed: {len(result.failures)}")
    ```

### Physical Scale Normalization

HistoSlice supports normalizing slides to a consistent physical resolution using `target_mpp`. This ensures both consistent physical scale AND consistent tensor dimensions for deep learning pipelines.

=== "CLI"
    ```bash
    # Normalize to 0.5 mpp with 512x512 pixel tiles
    # All slides will produce 512x512 tiles representing the same physical area
    histoslice slice \
        --input './images/*.tiff' \
        --output ./tiles \
        --width 512 \
        --target-mpp 0.5 \
        --overlap 0.5 \
        --max-background 0.5

    # Override slide mpp if metadata is missing or incorrect
    histoslice slice \
        --input './images/*.tiff' \
        --output ./tiles \
        --mpp 0.5 \
        --width 512 \
        --target-mpp 0.25
    ```
=== "Python API"
    ```python
    from histoslice import slice_slide

    # Normalize to target resolution - always get 512x512 pixel tiles.
    # mpp is read from slide metadata unless overridden below.
    result = slice_slide(
        "./path/to/slide.tiff",
        "./tiles/",
        tile_size=512,      # Output tile size in pixels
        target_mpp=0.5,     # Target resolution (512px * 0.5mpp = 256µm physical size)
        overlap=0.5,
        max_background=0.5,
    )
    # Result: 512x512 pixel tiles representing 256x256 µm physical area

    # Override mpp if slide metadata is missing or incorrect.
    result = slice_slide(
        "./path/to/slide.tiff",
        "./tiles/",
        tile_size=512,
        target_mpp=0.25,
        mpp=(0.5, 0.5),
    )
    ```
=== "Python API (low-level)"
    ```python
    from histoslice import Slide
    from histoslice.tiles import TileSpec, tile_regions

    # Read slide image - mpp extracted from metadata.
    slide = Slide("./path/to/slide.tiff")
    print(f"Slide mpp: {slide.mpp}")  # e.g., (0.25, 0.25)

    # TileSpec makes the contract explicit: `size` is the final output size,
    # `mpp` is the target physical resolution, independent of slide mpp.
    spec = TileSpec(size=(512, 512), mpp=(0.5, 0.5))
    # Region is always in level-0 coordinates - this is the level-0 crop size
    # needed to reach `spec.mpp` after resizing down to `spec.size`.
    crop_size = spec.level0_size(slide.mpp)
    regions = tile_regions(slide.dimensions, crop_size, overlap=0.5)
    # slide.read_tile(region, spec.size) picks an efficient pyramid level and
    # resizes to exactly 512x512 - export_tiles does this for every region.
    ```

!!! info "Resolution Normalization"
    When `target_mpp` is specified:

    - Tiles are extracted at the appropriate resolution to achieve the target mpp
    - Output tiles are always `tile_size` pixels (consistent tensor dimensions)
    - Anisotropic pixel sizes (`mpp_x != mpp_y`) are handled correctly, per axis
    - Example: 512px tiles at 0.5 mpp = 256µm x 256µm physical area

    This is ideal for deep learning where you need:
    - **Consistent physical scale** across slides (same biological structures)
    - **Consistent tensor shape** for neural networks (e.g., always 512x512)

!!! info "MPP Extraction"
    HistoSlice automatically extracts microns-per-pixel (mpp) from slide metadata when available. It supports:
    
    - OpenSlide properties (`openslide.mpp-x`, `openslide.mpp-y`)
    - TIFF resolution tags with unit conversion
    - Generic resolution metadata (xres, yres)
    
    If your slides don't have mpp metadata, you can provide it manually using the `--mpp` CLI option or `mpp` parameter in the Python API.

Output directory structure will look like this:

```bash
tiles
└── slide_id
    ├── metadata.parquet       # tile metadata
    ├── failures.json          # per-tile failures (only written if any failures occur)
    ├── thumbnail.jpeg         # thumbnail image (or .png if JPEG unsupported)
    ├── thumbnail_tiles.jpeg   # thumbnail with tiles (or .png if JPEG unsupported)
    ├── thumbnail_tissue.jpeg  # thumbnail of the tissue mask (or .png if JPEG unsupported)
    ├── masks                  # per-tile tissue masks (only if save_masks=True / --masks)
    └── tiles
```

!!! note
    If the underlying `libvips` build lacks JPEG support in your environment, HistoSlice
    will write `.png` files and update filenames accordingly. Developers can check
    support via `histoslice.functional.has_jpeg_support()`.

!!! note
    If any tiles fail during extraction, the CLI prints a warning and writes
    `failures.json` with per-tile error details.

![Prostate biopsy sample](https://github.com/rmuraix/HistoSlice/blob/main/images/thumbnail.jpeg?raw=true)
![Tissue mask](https://github.com/rmuraix/HistoSlice/blob/main/images/thumbnail_tissue.jpeg?raw=true)
![Thumbnail with tiles](https://github.com/rmuraix/HistoSlice/blob/main/images/thumbnail_tiles.jpeg?raw=true)

### Remove Bad Tiles

Histological slide images often contain tiles with technical problems - corrupted reads, blown-out exposure, near-blank scans. `clean` flags those, without treating biologically unusual (but valid) tissue as a problem.

=== "CLI"
    ```bash
    # First, extract tiles (technical QC metrics are always saved, --metrics is optional)
    histoslice slice \
        --input './images/*.tiff' \
        --output ./tiles \
        --width 512

    # Then, run technical QC and save metadata_clean.parquet
    # Specify the parent directory containing slide outputs
    histoslice clean \
        --input './tiles/*'

    # For parallel processing of multiple slides
    histoslice clean \
        --input './tiles/*' \
        --num-workers 4
    ```

=== "Python API"
    ```python
    from histoslice.qc import quality_control

    # Run technical QC directly on the metadata.
    checked = quality_control(result.metadata)
    # "fail": clear technical failures, safe to drop.
    outliers = checked.filter(checked["is_outlier"])
    # "warn": possible technical artifacts, kept by default - review before dropping.
    for_review = checked.filter(checked["needs_review"])
    ```

`clean` writes all original columns plus `qc_status` (`"pass"`/`"warn"`/`"fail"`), `qc_score`, `qc_reasons`, per-metric z-scores, `qc_method`, `is_outlier` (`qc_status == "fail"`) and `needs_review` (`qc_status == "warn"`) to `metadata_clean.parquet` in each slide directory. `is_outlier` only covers clear technical failures (corrupted/near-black/near-white/near-constant tiles) - it is **not** a biological or statistical rarity detector, and a technically clean slide can (and often will) have zero outliers. `needs_review` tiles are kept by default; the command supports parallel processing of multiple slides using the `--num-workers` option. See the [CLI documentation](cli.md#clean-technical-quality-control) for the full set of rules and thresholds.

For exploratory clustering of tile metrics (e.g. to visually browse a slide's tissue diversity, not for QC), see [`OutlierDetector.cluster_kmeans`](api/public/outlierdetector.md).
