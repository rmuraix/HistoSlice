# Metadata Fields

This page documents all metadata fields that HistoSlice collects when processing histological slide images. These fields are saved in the `metadata.parquet` file and can be used for quality control, filtering, and analysis of extracted tiles.

## Overview

Metadata is collected when you use the `save_metrics=True` option in the CLI or API:

=== "CLI"
    ```bash
    histoslice slice --input './images/*.tiff' --output ./tiles --metrics
    ```

=== "Python API"
    ```python
    from histoslice import SlideReader

    reader = SlideReader("./path/to/slide.tiff")
    threshold, tissue_mask = reader.get_tissue_mask(level=-1)
    tile_coordinates = reader.get_tile_coordinates(
        tissue_mask, width=512, overlap=0.5, max_background=0.5
    )
    
    # Save with metrics
    metadata = reader.save_regions(
        "./tiles/",
        tile_coordinates,
        threshold=threshold,
        save_metrics=True,  # Enable metadata collection
    )
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

### Image Quality Metrics

These metrics help identify problematic tiles that may need to be filtered out.

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `background` | `float64` | 0.0 - 1.0 | Proportion of background (non-tissue) pixels in the tile. Higher values indicate more background. |
| `black_pixels` | `float64` | 0.0 - 1.0 | Proportion of pure black pixels (value = 0). High values may indicate artifacts or scanning issues. |
| `white_pixels` | `float64` | 0.0 - 1.0 | Proportion of pure white pixels (value = 255). High values may indicate overexposed areas or background. |
| `laplacian_std` | `float64` | 0.0+ | Standard deviation of the Laplacian operator, measuring image sharpness. Higher values indicate sharper images. |

!!! tip "Quality Filtering"
    Common filtering criteria:
    
    - Filter tiles with `background > 0.5` (more than 50% background)
    - Filter tiles with `laplacian_std < 5.0` (out-of-focus or blurry)
    - Filter tiles with high `white_pixels` or `black_pixels` (artifacts)

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
| `gray_mean` | `float64` | 0.0 - 255.0 | Mean value of the grayscale conversion |
| `gray_std` | `float64` | 0.0+ | Standard deviation of the grayscale conversion |

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

When `save_metrics=True` is enabled, a total of **72 fields** are collected:

- 4 coordinate fields (x, y, w, h)
- 1-2 file path fields (path, and optionally mask_path)
- 4 image quality metrics
- 14 color channel statistics (mean/std for RGB, HSV, and grayscale)
- 49 quantile values (7 quantiles × 7 channels)

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

### Using with OutlierDetector

```python
from histoslice.utils import OutlierDetector

# Load metadata with OutlierDetector
detector = OutlierDetector.from_parquet("./tiles/slide_id/metadata.parquet")

# Add outlier criteria
detector.add_outliers(detector["background"] > 0.5, desc="high background")
detector.add_outliers(detector["laplacian_std"] < 5.0, desc="blurry")

# Visualize outliers
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

- [API Reference](api/public/slidereader.md) - SlideReader API documentation
- [Outlier Detection](api/public/outlierdetector.md) - OutlierDetector for filtering tiles
