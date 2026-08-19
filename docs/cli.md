# Command Line Interface

HistoSlice provides a command-line interface (CLI) for preprocessing histological slide images. The CLI is a thin wrapper around the Python API - `histoslice slice` calls `histoslice.slice_slide()` for each matched file, and `histoslice clean` calls `histoslice.qc.quality_control()` on each matched output directory. It includes two commands:

- **`slice`**: Extract tile images from histological slides
- **`clean`**: Run technical quality control (QC) on extracted tiles

!!! note "`is_outlier` is a technical QC flag, not a biological one"
    `clean` never flags tiles for being biologically unusual - tumor, stroma,
    adipose, necrosis, and other genuinely different tissue types are all
    valid. It only flags clear technical failures (corrupted/blank/blown-out
    tiles) as `is_outlier`, plus slide-relative anomalies worth a manual look
    as `needs_review`. **`needs_review` tiles are kept by default** - `clean`
    never deletes or moves anything; filter `metadata_clean.parquet` yourself.

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
| `--metrics` | | FLAG | False | Save the full set of exploratory image metrics (contrast, brightness, quantiles, etc.) to metadata. Not required for `clean` - a small set of technical QC metrics is always saved regardless of this flag. |
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
    If the underlying `libvips` build lacks JPEG support in your environment, HistoSlice
    will write `.png` files and update filenames accordingly. Developers can check
    support via `histoslice.functional.has_jpeg_support()`.

!!! note
    If any tiles fail during extraction, the CLI prints a warning and writes
    `failures.json` with per-tile error details.

---

### `clean` - Technical Quality Control

Run technical QC on already-extracted tiles: flag tiles with clear technical failures (corrupted, blank, blown-out) and, when enough tiles exist to compare against, tiles that deviate strongly from the rest of the slide. This writes a `metadata_clean.parquet` file next to `metadata.parquet` in each matched slide directory - the `slice`/`clean` commands never move or delete tile files themselves; use the extra `qc_status`/`is_outlier`/`needs_review` columns to filter tiles downstream (e.g., when building your training dataset).

`clean` works on any slide extracted with `slice` - it does not require `--metrics`, since a small set of technical QC metrics is always saved during extraction (see [Metadata Fields](metadata.md)).

#### Usage

```bash
histoslice clean [OPTIONS]
```

#### Options

##### Input

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--input` | `-i` | TEXT | *required* | Directory pattern to glob for slide outputs (e.g., `'./tiles/*'` or `'./tiles/slide_*'`). Looks for directories containing `metadata.parquet`. |

##### Output

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--num-workers` | `-j` | INTEGER | CPU-count | Number of slides processed in parallel. `0` = sequential processing. |

Threshold customization is not exposed on the CLI - pass a custom `histoslice.qc.QCConfig` to `histoslice.qc.quality_control()` directly if you need to tune it.

#### How It Works

Each tile gets a `qc_status` of `"pass"`, `"warn"`, or `"fail"`:

1. **Hard rules** (`"fail"`, absolute thresholds, never slide-relative): a near-black tile (`dark_fraction >= 0.90`), a near-white/blown-out tile (`bright_fraction >= 0.995`), or a near-constant/corrupted tile (`gray_std <= 2.0`).
2. **Soft, slide-relative rules** (`"warn"`), computed only when the slide has at least 32 reference tiles (non-failed tiles with ≥50% tissue) to compare against: a tile whose focus score is a robust z-score below -4 relative to the slide (`"possible_blur"`), or a tile where at least 2 of brightness/saturation/contrast are far from the slide's typical range (`"appearance_shift"`). A single unusual appearance metric does not warn - biologically distinct tissue (tumor, stroma, adipose, necrosis, mucin, ...) is expected to vary and is not by itself a technical problem.
3. **Output**: `metadata_clean.parquet` is written with all original columns plus `qc_status`, `qc_score` (a relative severity score, not a probability), `qc_reasons` (list of triggered reasons), `qc_focus_z`/`qc_brightness_z`/`qc_saturation_z`/`qc_contrast_z`, `qc_method` (`"technical_qc_v1"`), `is_outlier` (`qc_status == "fail"`), and `needs_review` (`qc_status == "warn"`). Tile files on disk are left untouched.

On a slide with only technically normal tiles, `is_outlier` is `False` for every tile - unlike the old k-means-based clustering, `clean` never forces a "worst" cluster to be treated as outliers.

#### Examples

**Basic usage:**

```bash
# Extract tiles (technical QC metrics are always saved)
histoslice slice \
    --input './slides/*.tiff' \
    --output ./tiles \
    --width 512

# Run technical QC (writes metadata_clean.parquet per slide)
histoslice clean \
    --input './tiles/*'
```

**Process specific slides:**

```bash
histoslice clean \
    --input './tiles/slide_0*'
```

**Parallel processing of multiple slides:**

```bash
histoslice clean \
    --input './tiles/*' \
    --num-workers 8
```

#### Output Structure

After running the `clean` command:

```
output/
└── slide_name/
    ├── metadata.parquet
    ├── metadata_clean.parquet    # original columns + qc_status, qc_score, qc_reasons, is_outlier, needs_review, ...
    └── tiles/                    # untouched
        ├── x0_y0_w512_h512.jpeg
        └── ...
```

To act on the results, filter `metadata_clean.parquet` yourself, e.g.:

```python
import polars as pl

df = pl.read_parquet("./output/slide_name/metadata_clean.parquet")
good_tiles = df.filter(~pl.col("is_outlier"))          # drop clear technical failures
reviewed_tiles = df.filter(~pl.col("needs_review"))    # also drop tiles flagged for manual review
```

---

## Complete Workflow Example

Here's a complete example workflow for processing histological slides:

```bash
# Step 1: Extract tiles with thumbnails (technical QC metrics are always saved)
histoslice slice \
    --input './raw_slides/*.tiff' \
    --output ./processed \
    --width 512 \
    --overlap 0.5 \
    --max-background 0.5 \
    --thumbnails \
    --num-workers 4

# Step 2: Review thumbnails (check thumbnail_tiles.jpeg files)
# Adjust parameters if needed and re-run with --overwrite

# Step 3: Run technical QC on the processed tiles
histoslice clean \
    --input './processed/*' \
    --num-workers 4

# Step 4: Filter out is_outlier==True (and, optionally, needs_review==True)
# rows when you build your training dataset
```

## Tips and Best Practices

### Tile Extraction

- **Start with defaults**: Use default parameters first, then adjust based on your needs.
- **`--metrics` is optional**: Only needed for the full exploratory metric set; `clean` works without it.
- **Check thumbnails**: Use `--thumbnails` to visually verify tile placement and tissue detection.
- **Optimize overlap**: Use `--overlap 0.5` for better coverage, but note this increases tile count.
- **Adjust background threshold**: Lower `--max-background` (e.g., 0.5) for stricter tissue selection.
- **Consider memory**: Large slides with small tiles and high overlap can generate many tiles. Monitor memory usage.

### Tissue Detection

- **Automatic thresholding**: Omit `--threshold` to use Otsu's method (works well for most slides).
- **Fine-tune with multiplier**: Adjust `--multiplier` (e.g., 1.1 or 0.95) to increase/decrease sensitivity.
- **Speed vs. accuracy**: Set `--tissue-level` to a coarser pyramid level for faster processing, a finer one for more precise tissue detection.
- **Blurring**: Increase `--sigma` for slides with noise or fine details that interfere with tissue detection.

### Quality Control

- **Review before filtering**: Inspect `metadata_clean.parquet` and `qc_reasons` before excluding tiles from training.
- **`needs_review` tiles are kept**: `clean` never deletes/moves tiles; decide yourself whether to drop `needs_review` rows.
- **Few reference tiles**: Slides with fewer than 32 reference tiles only get the absolute hard-fail rules - `needs_review` will never be set for them.
- **Custom thresholds**: Use `histoslice.qc.QCConfig` via the Python API if the defaults don't fit your data; the CLI intentionally exposes no threshold flags.

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

### `clean` reports no failures/warnings

This is expected and normal for a technically clean slide - unlike the old
clustering-based approach, `clean` does not force any tile to be flagged.
If you expected `needs_review` tiles and got none, check that the slide has
at least 32 reference tiles (non-failed tiles with ≥50% tissue); below that,
only the absolute hard-fail rules run.

## See Also

- [API Documentation](api/public/slice_slide.md) - Python API for programmatic access
- [Main Documentation](index.md) - Overview and Python examples
