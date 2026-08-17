# Command Line Interface

HistoSlice provides a command-line interface (CLI) for preprocessing histological slide images. The CLI is a thin wrapper around the Python API - `histoslice slice` calls `histoslice.slice_slide()` for each matched file, and `histoslice clean` calls `OutlierDetector` on each matched output directory. It includes two commands:

- **`slice`**: Extract tile images from histological slides
- **`clean`**: Detect outlier tile images using clustering

## Installation

Before using the CLI, ensure HistoSlice is installed:

```bash
pip install histoslice
# or
uv add histoslice
```

## General Usage

```bash
histoslice [OPTIONS] COMMAND [ARGS]...
```

To see available commands:

```bash
histoslice --help
```

## Commands

### `slice` - Extract Tile Images

Extract tile images from histological slides with tissue detection and configurable tiling parameters. Slides matching `--input` are processed in parallel (one process per slide, see `--num-workers`); see the [API documentation](api/public/slice_slide.md) if you need control over any individual step (tissue detection, tile grid, filtering, saving) instead.

#### Usage

```bash
histoslice slice [OPTIONS]
```

#### Options

##### Input/Output

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--input` | `-i` | TEXT | *required* | File pattern to glob (e.g., `'./slides/*.tiff'`). Supports wildcards for batch processing. |
| `--output` | `-o` | DIRECTORY | *required* | Parent directory for all outputs. Will be created if it doesn't exist. |

##### Tile Extraction

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--width` | `-w` | INTEGER | 512 | Tile size in pixels (square tiles). Must be ≥ 1. |
| `--overlap` | `-n` | FLOAT | 0.0 | Overlap between neighbouring tiles as a fraction (0.0-1.0). E.g., 0.5 = 50% overlap. |
| `--max-background` | `-b` | FLOAT | 0.75 | Maximum background ratio allowed in tiles (0.0-1.0). Tiles with more background are excluded. |
| `--target-mpp` | | FLOAT | None | Target microns per pixel for the output tiles. If set, tiles are resampled to this physical resolution while staying `--width` pixels. |
| `--mpp` | | FLOAT | from metadata | Microns per pixel override (assumes square pixels). Overrides slide metadata; required together with `--target-mpp` if the slide has no mpp metadata. |

##### Tissue Detection

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--threshold` | `-t` | INTEGER | Otsu | Global thresholding value for tissue detection (0-255). If not specified, Otsu's method is used. |
| `--multiplier` | `-x` | FLOAT | 1.05 | Multiplier for Otsu's threshold. Must be ≥ 0.0. Values > 1.0 increase sensitivity. |
| `--sigma` | | FLOAT | 1.0 | Sigma for Gaussian blurring before tissue detection. Must be ≥ 0.0. |
| `--tissue-level` | | INTEGER | auto | Pyramid level for tissue detection. If not specified, the lowest-resolution level with both dimensions ≤ 4096px is picked automatically. |

##### Tile Saving

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--metrics` | | FLAG | False | Save image metrics (contrast, brightness, etc.) to metadata. Required for the `clean` command. |
| `--masks` | | FLAG | False | Save per-tile tissue masks under a `masks/` subdirectory. |
| `--thumbnails` | | FLAG | False | Save slide thumbnails: plain, with the tile grid overlay, and with the tissue mask overlay. |
| `--overwrite` | `-z` | FLAG | False | Overwrite any existing slide outputs. |
| `--unfinished` | `-u` | FLAG | False | Overwrite only slides whose previous run didn't finish (no `metadata.parquet`). |
| `--image-format` | | TEXT | jpeg | File format for tile images (e.g., `jpeg`, `png`, `tiff`). If JPEG support is unavailable, output falls back to `png` regardless of this setting. |
| `--quality` | | INTEGER | 80 | Quality for JPEG compression (0-100). Higher values = better quality but larger files. |
| `--num-workers` | `-j` | INTEGER | CPU-count | Number of slides processed in parallel (one process per slide). `0` = sequential processing. |

#### Examples

**Basic usage - Extract 512x512 tiles:**

```bash
histoslice slice \
    --input './slides/*.tiff' \
    --output ./tiles \
    --width 512
```

**Advanced usage - Extract overlapping tiles with metrics:**

```bash
histoslice slice \
    --input './slides/*.tiff' \
    --output ./tiles \
    --width 512 \
    --overlap 0.5 \
    --max-background 0.5 \
    --metrics \
    --thumbnails
```

**Resolution normalization - Consistent physical scale and tensor dimensions:**

```bash
# Normalize to 0.5 mpp with 512x512 pixel tiles
# All slides produce 512x512 tiles representing 256µm x 256µm physical area
histoslice slice \
    --input './slides/*.tiff' \
    --output ./tiles \
    --width 512 \
    --target-mpp 0.5 \
    --overlap 0.5 \
    --max-background 0.5
```

**Resolution normalization with mpp override:**

```bash
# Override slide mpp if metadata is missing or incorrect
histoslice slice \
    --input './slides/*.tiff' \
    --output ./tiles \
    --mpp 0.5 \
    --width 512 \
    --target-mpp 0.25 \
    --overlap 0.5
```

**Custom tissue detection:**

```bash
histoslice slice \
    --input './slides/*.svs' \
    --output ./tiles \
    --width 256 \
    --multiplier 1.1 \
    --tissue-level 3 \
    --sigma 2.0
```

**Parallel processing with a specific worker count:**

```bash
histoslice slice \
    --input './slides/**/*.tiff' \
    --output ./output \
    --width 512 \
    --overlap 0.25 \
    --num-workers 8
```

#### Output Structure

The `slice` command creates the following directory structure:

```
output/
└── slide_name/
    ├── metadata.parquet          # Tile metadata (coordinates, metrics, etc.)
    ├── failures.json             # Per-tile failures (only written if any failures occur)
    ├── thumbnail.jpeg            # Original slide thumbnail (if --thumbnails; .png if JPEG unsupported)
    ├── thumbnail_tiles.jpeg      # Thumbnail with tile grid (if --thumbnails; .png if JPEG unsupported)
    ├── thumbnail_tissue.jpeg     # Tissue mask thumbnail (if --thumbnails; .png if JPEG unsupported)
    ├── masks/                    # Per-tile tissue masks (if --masks)
    │   ├── x0_y0_w512_h512.png
    │   └── ...
    └── tiles/                    # Directory containing tile images
        ├── x0_y0_w512_h512.jpeg  # Uses chosen image format (.png if JPEG unsupported)
        ├── x512_y0_w512_h512.jpeg
        └── ...
```

!!! note
    If Pillow lacks JPEG support in your environment, HistoSlice will write `.png` files
    and update filenames accordingly. Developers can check support via
    `histoslice.functional.has_jpeg_support()`.

!!! note
    If any tiles fail during extraction, the CLI prints a warning and writes
    `failures.json` with per-tile error details.

---

### `clean` - Detect Outlier Tiles

Detect outlier tile images using k-means clustering on image metrics. This writes a `metadata_clean.parquet` file next to `metadata.parquet` in each matched slide directory - the `slice`/`clean` commands never move or delete tile files themselves; use the extra `is_outlier` column to filter tiles downstream (e.g., when building your training dataset).

!!! note "Prerequisite"
    The `clean` command requires that tiles were extracted with `--metrics`, as it uses image metrics for clustering.

#### Usage

```bash
histoslice clean [OPTIONS]
```

#### Options

##### Input

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--input` | `-i` | TEXT | *required* | Directory pattern to glob for slide outputs (e.g., `'./tiles/*'` or `'./tiles/slide_*'`). Looks for directories containing `metadata.parquet`. |

##### Outlier Detection

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--mode` | `-m` | TEXT | clustering | Outlier detection mode. Currently only `clustering` is supported. |
| `--num-clusters` | `-k` | INTEGER | 4 | Number of clusters for k-means clustering. Must be ≥ 2. Cluster 0 contains detected outliers. |

##### Output

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--num-workers` | `-j` | INTEGER | CPU-count | Number of slides processed in parallel. `0` = sequential processing. |

#### How It Works

1. **Clustering**: The command performs k-means clustering on tile image metrics (contrast, brightness, sharpness, etc.) for each slide directory.
2. **Outlier Identification**: Clusters are ordered by distance from the mean cluster center. Cluster 0 (most distant) is marked as outliers.
3. **Output**: `metadata_clean.parquet` is written with all original columns plus `is_outlier` (bool) and `method` (the detection mode, e.g. `"clustering"`). Tile files on disk are left untouched.

#### Examples

**Basic usage - Detect outliers with default settings:**

```bash
# First extract tiles with metrics
histoslice slice \
    --input './slides/*.tiff' \
    --output ./tiles \
    --width 512 \
    --metrics

# Then detect outliers (writes metadata_clean.parquet per slide)
histoslice clean \
    --input './tiles/*'
```

**Process specific slides:**

```bash
histoslice clean \
    --input './tiles/slide_0*' \
    --num-clusters 5
```

**Parallel processing of multiple slides:**

```bash
histoslice clean \
    --input './tiles/*' \
    --num-clusters 4 \
    --num-workers 8
```

#### Output Structure

After running the `clean` command:

```
output/
└── slide_name/
    ├── metadata.parquet
    ├── metadata_clean.parquet    # original columns + is_outlier, method
    └── tiles/                    # untouched
        ├── x0_y0_w512_h512.jpeg
        └── ...
```

To act on the outliers, filter `metadata_clean.parquet` yourself, e.g.:

```python
import polars as pl

df = pl.read_parquet("./output/slide_name/metadata_clean.parquet")
good_tiles = df.filter(~pl.col("is_outlier"))
```

---

## Complete Workflow Example

Here's a complete example workflow for processing histological slides:

```bash
# Step 1: Extract tiles with metrics and thumbnails
histoslice slice \
    --input './raw_slides/*.tiff' \
    --output ./processed \
    --width 512 \
    --overlap 0.5 \
    --max-background 0.5 \
    --metrics \
    --thumbnails \
    --num-workers 4

# Step 2: Review thumbnails (check thumbnail_tiles.jpeg files)
# Adjust parameters if needed and re-run with --overwrite

# Step 3: Detect outliers in the processed tiles
histoslice clean \
    --input './processed/*' \
    --num-clusters 4 \
    --num-workers 4

# Step 4: Filter out is_outlier==True rows when you build your training dataset
```

## Tips and Best Practices

### Tile Extraction

- **Start with defaults**: Use default parameters first, then adjust based on your needs.
- **Use `--metrics`**: Always include `--metrics` if you plan to use the `clean` command later.
- **Check thumbnails**: Use `--thumbnails` to visually verify tile placement and tissue detection.
- **Optimize overlap**: Use `--overlap 0.5` for better coverage, but note this increases tile count.
- **Adjust background threshold**: Lower `--max-background` (e.g., 0.5) for stricter tissue selection.
- **Consider memory**: Large slides with small tiles and high overlap can generate many tiles. Monitor memory usage.

### Tissue Detection

- **Automatic thresholding**: Omit `--threshold` to use Otsu's method (works well for most slides).
- **Fine-tune with multiplier**: Adjust `--multiplier` (e.g., 1.1 or 0.95) to increase/decrease sensitivity.
- **Speed vs. accuracy**: Set `--tissue-level` to a coarser pyramid level for faster processing, a finer one for more precise tissue detection.
- **Blurring**: Increase `--sigma` for slides with noise or fine details that interfere with tissue detection.

### Outlier Detection

- **Cluster count**: Start with `--num-clusters 4`, increase for more granular separation.
- **Review before filtering**: Inspect `metadata_clean.parquet` (e.g., with `OutlierDetector`) before excluding tiles from training.
- **Iterate**: You can run `clean` multiple times with different `--num-clusters` values; each run overwrites `metadata_clean.parquet`.

### Performance

- **Parallel processing**: Use `--num-workers` to match your CPU count for faster processing. Parallelism is per-slide, not per-tile.
- **Sequential for debugging**: Use `--num-workers 0` when debugging or for small datasets.
- **JPEG quality**: Lower `--quality` (e.g., 70) reduces file size with minimal quality loss.
- **Batch processing**: Use glob patterns to process multiple slides at once.

## Troubleshooting

### No files found

```
Found no files matching pattern './slides/*.tiff'.
```

**Solution**: Check your input pattern and ensure files exist. Use absolute paths or verify your current directory.

### Memory issues

For very large slides or many tiles:

- Reduce `--num-workers`
- Increase `--max-background` to extract fewer tiles
- Process slides individually instead of in batch

### No outliers detected

If `clean` reports no outliers:

- Ensure you used `--metrics` when extracting tiles
- Try increasing `--num-clusters`
- Verify `metadata.parquet` exists in slide directories

## See Also

- [API Documentation](api/public/slice_slide.md) - Python API for programmatic access
- [Main Documentation](index.md) - Overview and Python examples
