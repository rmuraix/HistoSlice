# Metadata Fields

This page documents all metadata fields that HistoSlice collects when processing histological slide images. These fields are saved in the `metadata.parquet` file and can be used for quality control, filtering, and analysis of extracted tiles. If any tile fails during saving, HistoSlice also writes a `failures.json` file with per-tile error details.

## Overview

`metadata.parquet` always contains coordinates, file paths, and a small set
of **technical QC metrics** (see below) - these are cheap to compute and are
what `histoslice clean` uses, no extra flag required. The much larger set of
**exploratory image metrics** (RGB/HSV/grayscale statistics and quantiles)
is only collected when you pass `save_metrics=True` (`--metrics` on the
CLI):

=== "CLI"
    ```bash
    # Technical QC metrics are always saved.
    histoslice slice --input './images/*.tiff' --output ./tiles

    # Add --metrics for the full exploratory metric set too.
    histoslice slice --input './images/*.tiff' --output ./tiles --metrics
    ```

=== "Python API"
    ```python
    from histoslice import slice_slide

    result = slice_slide(
        "./path/to/slide.tiff",
        "./tiles/",
        tile_size=512,
        overlap=0.5,
        max_background=0.5,
        save_metrics=True,  # Also collect the full exploratory metric set
    )
    metadata = result.metadata
    if result.failures:
        print(f"Some tiles failed: {len(result.failures)}")
    ```

## Metadata Fields

### Coordinate Information

These fields define the location and dimensions of each tile in the original slide image.

| Field | Type | Description |
|-------|------|-------------|
| `x` | `int64` | X-coordinate of the tile's top-left corner (in pixels) |
| `y` | `int64` | Y-coordinate of the tile's top-left corner (in pixels) |
| `w` | `int64` | Width of the tile (in pixels) |
| `h` | `int64` | Height of the tile (in pixels) |

### File Path

| Field | Type | Description |
|-------|------|-------------|
| `path` | `str` | Absolute file path to the saved tile image |
| `mask_path` | `str` | Absolute file path to the tissue mask image (only present if `save_masks=True`) |

### Technical QC Metrics (always saved)

These fields are always computed during `slice`, regardless of `save_metrics`/`--metrics` - they're what `histoslice clean` (`histoslice.qc.quality_control`) uses to flag technical failures. See [Quality Control](api/public/qc.md) and the [CLI reference](cli.md#clean-technical-quality-control) for how they're turned into `pass`/`warn`/`fail` decisions.

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `background` | `float64` | 0.0 - 1.0 | Proportion of background (non-tissue) pixels in the tile. |
| `dark_fraction` | `float64` | 0.0 - 1.0 | Proportion of near-black grayscale pixels (value ≤ 8). High values suggest a corrupted or blank-scan tile. |
| `bright_fraction` | `float64` | 0.0 - 1.0 | Proportion of near-white grayscale pixels (value ≥ 247). High values suggest a blown-out/overexposed tile. |
| `gray_std` | `float64` | 0.0+ | Whole-tile (full-resolution) grayscale standard deviation. Near-zero indicates a near-constant/corrupted image. |
| `focus_score` | `float64` | 0.0+ | Laplacian variance restricted to an eroded "tissue core" mask, so the tissue/background boundary doesn't dominate the sharpness estimate. Higher is sharper. |
| `tissue_brightness` | `float64` | 0.0 - 255.0 | Tissue-only median HSV value (brightness). |
| `tissue_saturation` | `float64` | 0.0 - 255.0 | Tissue-only median HSV saturation. |
| `tissue_contrast` | `float64` | 0.0+ | Tissue-only grayscale q90 - q10. |

!!! note "Not a probability, not a biological signal"
    None of these metrics - nor `qc_score` (added by `clean`) - are probabilities. A tile with unusual `tissue_brightness`/`tissue_saturation`/`tissue_contrast` is not automatically wrong; it may simply be a different, biologically valid tissue type (tumor, stroma, adipose, necrosis, mucin, ...). Only the absolute hard-fail rules (near-black/near-white/near-constant) mark a tile `is_outlier` by themselves - see [Quality Control](api/public/qc.md).

### Image Quality Metrics (requires `--metrics`)

These metrics are part of the full exploratory metric set (`save_metrics=True`) and, unlike the technical QC metrics above, help with open-ended exploration rather than automated QC decisions.

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `black_pixels` | `float64` | 0.0 - 1.0 | Proportion of pure black pixels (value = 0). High values may indicate artifacts or scanning issues. |
| `white_pixels` | `float64` | 0.0 - 1.0 | Proportion of pure white pixels (value = 255). High values may indicate overexposed areas or background. |
| `laplacian_std` | `float64` | 0.0+ | Whole-tile (not tissue-restricted) standard deviation of the Laplacian operator. Higher values indicate sharper images; unlike `focus_score`, this is not restricted to a tissue core, so it's more sensitive to the tissue/background boundary. |

!!! tip "Quality Filtering"
    Common filtering criteria:

    - Filter tiles with `background > 0.5` (more than 50% background)
    - Filter tiles with `laplacian_std < 5.0` (out-of-focus or blurry)
    - Filter tiles with high `white_pixels` or `black_pixels` (artifacts)
    - Prefer `histoslice clean` for automated, slide-relative QC instead of picking fixed thresholds yourself.

### Color Channel Statistics

Mean and standard deviation values for each color channel across multiple color spaces.

#### RGB Color Space

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `red_mean` | `float64` | 0.0 - 255.0 | Mean value of the red channel |
| `red_std` | `float64` | 0.0+ | Standard deviation of the red channel |
| `green_mean` | `float64` | 0.0 - 255.0 | Mean value of the green channel |
| `green_std` | `float64` | 0.0+ | Standard deviation of the green channel |
| `blue_mean` | `float64` | 0.0 - 255.0 | Mean value of the blue channel |
| `blue_std` | `float64` | 0.0+ | Standard deviation of the blue channel |

#### HSV Color Space

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `hue_mean` | `float64` | 0.0 - 179.0 | Mean value of the hue channel (OpenCV range) |
| `hue_std` | `float64` | 0.0+ | Standard deviation of the hue channel |
| `saturation_mean` | `float64` | 0.0 - 255.0 | Mean value of the saturation channel |
| `saturation_std` | `float64` | 0.0+ | Standard deviation of the saturation channel |
| `brightness_mean` | `float64` | 0.0 - 255.0 | Mean value of the brightness (value) channel |
| `brightness_std` | `float64` | 0.0+ | Standard deviation of the brightness channel |

#### Grayscale

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `gray_mean` | `float64` | 0.0 - 255.0 | Mean value of the grayscale conversion, computed on a resized (64x64) image for speed |
| `gray_std` | `float64` | 0.0+ | Same field as in [Technical QC Metrics](#technical-qc-metrics-always-saved) above (full-resolution, not resized) - listed here for completeness |

### Color Channel Quantiles

Quantile values (percentiles) for tissue pixels in each color channel. These are computed at the following quantiles: 5%, 10%, 25%, 50% (median), 75%, 90%, and 95%.

!!! info "Quantile Calculation"
    Quantiles are calculated only for tissue pixels (non-background). The image is first resized to 64x64 for efficient computation.

#### RGB Quantiles

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `red_q5` | `int64` | 0 - 255 | 5th percentile of red channel values in tissue |
| `red_q10` | `int64` | 0 - 255 | 10th percentile of red channel values in tissue |
| `red_q25` | `int64` | 0 - 255 | 25th percentile (Q1) of red channel values in tissue |
| `red_q50` | `int64` | 0 - 255 | 50th percentile (median) of red channel values in tissue |
| `red_q75` | `int64` | 0 - 255 | 75th percentile (Q3) of red channel values in tissue |
| `red_q90` | `int64` | 0 - 255 | 90th percentile of red channel values in tissue |
| `red_q95` | `int64` | 0 - 255 | 95th percentile of red channel values in tissue |
| `green_q5` | `int64` | 0 - 255 | 5th percentile of green channel values in tissue |
| `green_q10` | `int64` | 0 - 255 | 10th percentile of green channel values in tissue |
| `green_q25` | `int64` | 0 - 255 | 25th percentile (Q1) of green channel values in tissue |
| `green_q50` | `int64` | 0 - 255 | 50th percentile (median) of green channel values in tissue |
| `green_q75` | `int64` | 0 - 255 | 75th percentile (Q3) of green channel values in tissue |
| `green_q90` | `int64` | 0 - 255 | 90th percentile of green channel values in tissue |
| `green_q95` | `int64` | 0 - 255 | 95th percentile of green channel values in tissue |
| `blue_q5` | `int64` | 0 - 255 | 5th percentile of blue channel values in tissue |
| `blue_q10` | `int64` | 0 - 255 | 10th percentile of blue channel values in tissue |
| `blue_q25` | `int64` | 0 - 255 | 25th percentile (Q1) of blue channel values in tissue |
| `blue_q50` | `int64` | 0 - 255 | 50th percentile (median) of blue channel values in tissue |
| `blue_q75` | `int64` | 0 - 255 | 75th percentile (Q3) of blue channel values in tissue |
| `blue_q90` | `int64` | 0 - 255 | 90th percentile of blue channel values in tissue |
| `blue_q95` | `int64` | 0 - 255 | 95th percentile of blue channel values in tissue |

#### HSV Quantiles

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `hue_q5` | `int64` | 0 - 179 | 5th percentile of hue channel values in tissue |
| `hue_q10` | `int64` | 0 - 179 | 10th percentile of hue channel values in tissue |
| `hue_q25` | `int64` | 0 - 179 | 25th percentile (Q1) of hue channel values in tissue |
| `hue_q50` | `int64` | 0 - 179 | 50th percentile (median) of hue channel values in tissue |
| `hue_q75` | `int64` | 0 - 179 | 75th percentile (Q3) of hue channel values in tissue |
| `hue_q90` | `int64` | 0 - 179 | 90th percentile of hue channel values in tissue |
| `hue_q95` | `int64` | 0 - 179 | 95th percentile of hue channel values in tissue |
| `saturation_q5` | `int64` | 0 - 255 | 5th percentile of saturation channel values in tissue |
| `saturation_q10` | `int64` | 0 - 255 | 10th percentile of saturation channel values in tissue |
| `saturation_q25` | `int64` | 0 - 255 | 25th percentile (Q1) of saturation channel values in tissue |
| `saturation_q50` | `int64` | 0 - 255 | 50th percentile (median) of saturation channel values in tissue |
| `saturation_q75` | `int64` | 0 - 255 | 75th percentile (Q3) of saturation channel values in tissue |
| `saturation_q90` | `int64` | 0 - 255 | 90th percentile of saturation channel values in tissue |
| `saturation_q95` | `int64` | 0 - 255 | 95th percentile of saturation channel values in tissue |
| `brightness_q5` | `int64` | 0 - 255 | 5th percentile of brightness channel values in tissue |
| `brightness_q10` | `int64` | 0 - 255 | 10th percentile of brightness channel values in tissue |
| `brightness_q25` | `int64` | 0 - 255 | 25th percentile (Q1) of brightness channel values in tissue |
| `brightness_q50` | `int64` | 0 - 255 | 50th percentile (median) of brightness channel values in tissue |
| `brightness_q75` | `int64` | 0 - 255 | 75th percentile (Q3) of brightness channel values in tissue |
| `brightness_q90` | `int64` | 0 - 255 | 90th percentile of brightness channel values in tissue |
| `brightness_q95` | `int64` | 0 - 255 | 95th percentile of brightness channel values in tissue |

#### Grayscale Quantiles

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `gray_q5` | `int64` | 0 - 255 | 5th percentile of grayscale values in tissue |
| `gray_q10` | `int64` | 0 - 255 | 10th percentile of grayscale values in tissue |
| `gray_q25` | `int64` | 0 - 255 | 25th percentile (Q1) of grayscale values in tissue |
| `gray_q50` | `int64` | 0 - 255 | 50th percentile (median) of grayscale values in tissue |
| `gray_q75` | `int64` | 0 - 255 | 75th percentile (Q3) of grayscale values in tissue |
| `gray_q90` | `int64` | 0 - 255 | 90th percentile of grayscale values in tissue |
| `gray_q95` | `int64` | 0 - 255 | 95th percentile of grayscale values in tissue |

## Total Metadata Fields

By default (no `--metrics`), a total of **13 fields** are collected:

- 4 coordinate fields (x, y, w, h)
- 1-2 file path fields (path, and optionally mask_path)
- 8 technical QC metrics

When `save_metrics=True`/`--metrics` is also enabled, the full exploratory metric set is added on top - 4 image quality metrics, 14 color channel statistics, and 49 quantile values (67 fields), of which `background` and `gray_std` are shared with (and, for `gray_std`, superseded by - see the note above) the technical QC metrics rather than being new columns. That's 65 new fields, for **78 fields** total.

After `histoslice clean`, `metadata_clean.parquet` adds 10 more QC result columns (`qc_status`, `qc_score`, `qc_reasons`, `qc_focus_z`, `qc_brightness_z`, `qc_saturation_z`, `qc_contrast_z`, `qc_method`, `is_outlier`, `needs_review`) on top of whatever `metadata.parquet` already had.

## Usage Examples

### Loading and Filtering Metadata

```python
import polars as pl

# Load metadata
metadata = pl.read_parquet("./tiles/slide_id/metadata.parquet")

# Filter tiles with high background
good_tiles = metadata.filter(pl.col("background") < 0.5)

# Filter tiles with good sharpness
sharp_tiles = metadata.filter(pl.col("laplacian_std") > 10.0)

# Combine multiple filters
quality_tiles = metadata.filter(
    (pl.col("background") < 0.5) &
    (pl.col("laplacian_std") > 10.0) &
    (pl.col("white_pixels") < 0.1)
)
```

### Running Technical QC

```python
import polars as pl
from histoslice.qc import quality_control

metadata = pl.read_parquet("./tiles/slide_id/metadata.parquet")
checked = quality_control(metadata)  # same as `histoslice clean`

good_tiles = checked.filter(~pl.col("is_outlier"))       # drop clear technical failures
reviewed = checked.filter(~pl.col("needs_review"))        # also drop tiles flagged for review
print(checked.group_by("qc_status").len())
```

`is_outlier` only ever means a technical QC failure (corrupted, near-black,
near-white, near-constant) - it is not set for tiles that are simply
biologically or statistically unusual. See [Quality Control](api/public/qc.md).

### Exploring Tile Metrics with OutlierDetector

`OutlierDetector` (including `cluster_kmeans`) is for interactive
exploration/visualisation of the full metric set (`save_metrics=True`), not
for technical QC - use `quality_control` above for that.

```python
from histoslice.utils import OutlierDetector

# Load metadata with OutlierDetector
detector = OutlierDetector.from_parquet("./tiles/slide_id/metadata.parquet")

# Add custom selection criteria for exploration
detector.add_outliers(detector["background"] > 0.5, desc="high background")
detector.add_outliers(detector["laplacian_std"] < 5.0, desc="blurry")

# Visualize
detector.plot_histogram("laplacian_std", num_images=20)
collage = detector.random_image_collage(~detector.outliers, num_rows=4)
collage.show()
```

### Statistical Analysis

```python
# Get summary statistics
print(metadata.describe())

# Check correlations between metrics
correlations = metadata.select([
    "background", "laplacian_std", "red_mean", "green_mean", "blue_mean"
]).corr()
print(correlations)

# Find tiles with extreme values
darkest_tiles = metadata.sort("gray_mean").head(10)
brightest_tiles = metadata.sort("gray_mean", descending=True).head(10)
```

## Related Documentation

- [API Reference](api/public/slice_slide.md) - `slice_slide` API documentation
- [Quality Control](api/public/qc.md) - `quality_control`/`QCConfig` (used by `histoslice clean`)
- [Outlier Detection](api/public/outlierdetector.md) - `OutlierDetector` for exploring tile metrics
