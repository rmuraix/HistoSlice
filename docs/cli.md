# Command Line Interface

HistoSlice provides a powerful command-line interface (CLI) for preprocessing histological slide images. The CLI includes two main commands:

- **`slice`**: Extract tile images from histological slides
- **`clean`**: Detect and remove outlier tile images using clustering

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

Extract tile images from histological slides with tissue detection and configurable tiling parameters.

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
| `--mpp` | | FLOAT | from metadata | Microns per pixel (assumes square pixels). Overrides slide metadata. Used with `--target-mpp` for normalization. |

##### Tile Extraction

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--level` | `-l` | INTEGER | 0 | Pyramid level for tile extraction (0 = highest resolution). Must be ≥ 0. |
| `--width` | `-w` | INTEGER | 640 | Tile width in pixels at target resolution. Must be ≥ 0. |
| `--height` | `-h` | INTEGER | width | Tile height in pixels at target resolution. Defaults to same as width for square tiles. Must be ≥ 0. |
| `--target-mpp` | | FLOAT | None | Target microns per pixel for normalization. Tiles will be scaled to achieve this resolution. Guarantees consistent physical scale and tensor dimensions. |
| `--overlap` | `-n` | FLOAT | 0.0 | Overlap between neighbouring tiles as a fraction (0.0-1.0). E.g., 0.5 = 50% overlap. |
| `--max-background` | `-b` | FLOAT | 0.75 | Maximum background ratio allowed in tiles (0.0-1.0). Tiles with more background are excluded. |
| `--in-bounds` | | FLAG | False | If set, prevents tiles from going out-of-bounds of the slide. |

##### Tissue Detection

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--threshold` | `-t` | INTEGER | Otsu | Global thresholding value for tissue detection (0-255). If not specified, Otsu's method is used. |
| `--multiplier` | `-x` | FLOAT | 1.05 | Multiplier for Otsu's threshold. Must be ≥ 0.0. Values > 1.0 increase sensitivity. |
| `--tissue-level` | | INTEGER | max_dimension | Pyramid level for tissue detection. If not specified, determined by `--max-dimension`. |
| `--max-dimension` | | INTEGER | 8192 | Maximum dimension for tissue detection. Lower values are faster but less precise. |
| `--sigma` | | FLOAT | 1.0 | Sigma for Gaussian blurring before tissue detection. Must be ≥ 0.0. |

##### Tile Saving

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--metrics` | | FLAG | False | Save image metrics (contrast, brightness, etc.) to metadata. Required for outlier detection. |
| `--masks` | | FLAG | False | Save tissue masks as separate images. |
| `--thumbnails` | | FLAG | False | Save thumbnail images of the slide with tissue overlay and tile grid. |
| `--overwrite` | `-z` | FLAG | False | Overwrite any existing slide outputs. |
| `--unfinished` | `-u` | FLAG | False | Overwrite only if metadata is missing (incomplete previous run). |
| `--image-format` | | TEXT | jpeg | File format for tile images (e.g., `jpeg`, `png`, `tiff`). If JPEG support is unavailable, output will use `png` regardless of this setting. |
| `--quality` | | INTEGER | 80 | Quality for JPEG compression (0-100). Higher values = better quality but larger files. |
| `--num-workers` | `-j` | INTEGER | CPU-count | Number of parallel workers for saving tiles. 0 = sequential processing. |

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
    --height 256 \
    --multiplier 1.1 \
    --max-dimension 4096 \
    --sigma 2.0
```

**Parallel processing with custom tile extraction:**

```bash
histoslice slice \
    --input './slides/**/*.tiff' \
    --output ./output \
    --level 1 \
    --width 512 \
    --overlap 0.25 \
    --in-bounds \
    --num-workers 8
```

#### Output Structure

The `slice` command creates the following directory structure:

```
output/
└── slide_name/
    ├── metadata.parquet          # Tile metadata (coordinates, metrics, etc.)
    ├── failures.json             # Per-tile failures (only written if any failures occur)
    ├── properties.json           # Slide properties
    ├── thumbnail.jpeg            # Original slide thumbnail (if --thumbnails; .png if JPEG unsupported)
    ├── thumbnail_tiles.jpeg      # Thumbnail with tile grid (if --thumbnails; .png if JPEG unsupported)
    ├── thumbnail_tissue.jpeg     # Tissue mask thumbnail (if --thumbnails; .png if JPEG unsupported)
    ├── mask.png                  # Tissue mask (if --masks)
    └── tiles/                    # Directory containing tile images
        ├── tile_0000.jpeg        # Uses chosen image format (.png if JPEG unsupported)
        ├── tile_0001.jpeg
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

### `clean` - Remove Outlier Tiles

Detect and remove outlier tile images using k-means clustering on image metrics. This helps eliminate tiles with artifacts, poor tissue quality, or other anomalies.

!!! note "Prerequisite"
    The `clean` command requires that tiles were extracted with the `--metrics` flag, as it uses image metrics for clustering.

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
| `--delete` | `-d` | FLAG | False | Delete detected outlier images permanently. If not set, moves to `outliers/` subdirectory. |
| `--num-workers` | `-j` | INTEGER | CPU-count | Number of parallel workers for processing multiple slides. 0 = sequential processing. |

#### How It Works

1. **Clustering**: The command performs k-means clustering on tile image metrics (contrast, brightness, sharpness, etc.)
2. **Outlier Identification**: Clusters are ordered by distance from the mean cluster center. Cluster 0 (most distant) is identified as outliers.
3. **Action**: Outliers are either moved to an `outliers/` subdirectory (default) or deleted (with `--delete`).

#### Examples

**Basic usage - Detect outliers with default settings:**

```bash
# First extract tiles with metrics
histoslice slice \
    --input './slides/*.tiff' \
    --output ./tiles \
    --width 512 \
    --metrics

# Then clean outliers (moves to outliers/ subdirectory)
histoslice clean \
    --input './tiles/*'
```

**Delete outliers instead of moving:**

```bash
histoslice clean \
    --input './tiles/*' \
    --num-clusters 4 \
    --delete
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

**Fine-tuned clustering:**

```bash
histoslice clean \
    --input './output/**/*' \
    --mode clustering \
    --num-clusters 6 \
    --num-workers 4
```

#### Output Structure

After running the `clean` command with default settings (without `--delete`):

```
output/
└── slide_name/
    ├── metadata.parquet
    ├── properties.json
    ├── tiles/
    │   ├── tile_0002.jpeg        # Good tiles remain (extension matches output format)
    │   ├── tile_0005.jpeg
    │   └── ...
    └── outliers/                 # Outlier tiles moved here
        ├── tile_0000.jpeg
        ├── tile_0003.jpeg
        └── ...
```

With `--delete` flag, outlier tiles are permanently deleted instead of moved.

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

# Step 3: Clean outliers from processed tiles
histoslice clean \
    --input './processed/*' \
    --num-clusters 4 \
    --num-workers 4

# Step 4: Review outliers in outliers/ subdirectories
# If satisfied, delete outliers:
find ./processed -type d -name "outliers" -exec rm -rf {} +

# Or re-run with --delete to skip manual review:
# histoslice clean --input './processed/*' --num-clusters 4 --delete
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
- **Speed vs. accuracy**: Lower `--max-dimension` for faster processing, higher for more precise tissue detection.
- **Blurring**: Increase `--sigma` for slides with noise or fine details that interfere with tissue detection.

### Outlier Detection

- **Cluster count**: Start with `--num-clusters 4`, increase for more granular separation.
- **Review before deleting**: Don't use `--delete` until you've verified outliers in the `outliers/` directory.
- **Iterate**: You can run `clean` multiple times with different `--num-clusters` values.
- **Manual curation**: For critical applications, manually review outliers before final deletion.

### Performance

- **Parallel processing**: Use `--num-workers` to match your CPU count for faster processing.
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
- Use a higher `--level` to extract from a lower resolution

### No outliers detected

If `clean` reports no outliers:

- Ensure you used `--metrics` when extracting tiles
- Try increasing `--num-clusters`
- Verify `metadata.parquet` exists in slide directories

## See Also

- [API Documentation](api/public/slidereader/) - Python API for programmatic access
- [Main Documentation](index.md) - Overview and Python examples
