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

`histoslice --help` will show you all available options. For Python API usage, see the [API documentation](api/public/slidereader/). 

=== "CLI"
    ```bash
    histoslice \
        --input './images/*.tiff' \
        --output ./tiles \
        --width 512 \
        --overlap 0.5 \
        --max-background 0.5 \
        --metrics \
        --thumbnail
    ```
=== "Python API"
    ```python
    from histoslice import SlideReader

    # Read slide image.
    reader = SlideReader("./parh/to/slide_image.tiff")
    # Detect tissue.
    threshold, tissue_mask = reader.get_tissue_mask(level=-1)
    # Extract overlapping tile coordinates with less than 50% background.
    tile_coordinates = reader.get_tile_coordinates(
        tissue_mask, width=512, overlap=0.5, max_background=0.5
    )
    # Save tile images with image metrics for preprocessing.
    tile_metadata, failures = reader.save_regions(
        "./tiles/",
        tile_coordinates,
        threshold=threshold,
        save_metrics=True,
        save_thumbnail=True
    )
    if failures:
        print(f"Some tiles failed: {len(failures)}")
    ```

### Physical Scale Normalization

HistoSlice supports normalizing slides to a consistent physical resolution using `target_mpp`. This ensures both consistent physical scale AND consistent tensor dimensions for deep learning pipelines.

=== "CLI"
    ```bash
    # Normalize to 0.5 mpp with 512x512 pixel tiles
    # All slides will produce 512x512 tiles representing the same physical area
    histoslice \
        --input './images/*.tiff' \
        --output ./tiles \
        --width 512 \
        --target-mpp 0.5 \
        --overlap 0.5 \
        --max-background 0.5
    
    # Override slide mpp if metadata is missing or incorrect
    histoslice \
        --input './images/*.tiff' \
        --output ./tiles \
        --mpp 0.5 \
        --width 512 \
        --target-mpp 0.25
    ```
=== "Python API"
    ```python
    from histoslice import SlideReader
    
    # Read slide image - mpp extracted from metadata
    reader = SlideReader("./path/to/slide.tiff")
    print(f"Slide mpp: {reader.mpp}")  # e.g., (0.25, 0.25)
    
    # Override mpp if needed
    reader = SlideReader("./path/to/slide.tiff", mpp=(0.5, 0.5))
    
    # Normalize to target resolution - always get 512x512 pixel tiles
    threshold, tissue_mask = reader.get_tissue_mask(level=-1)
    tile_coordinates = reader.get_tile_coordinates(
        tissue_mask, 
        width=512,       # Output tile size in pixels
        target_mpp=0.5,  # Target resolution (512px * 0.5mpp = 256µm physical size)
        overlap=0.5, 
        max_background=0.5
    )
    # Result: 512x512 pixel tiles representing 256x256 µm physical area
    ```

!!! info "Resolution Normalization"
    When `target_mpp` is specified:
    
    - Tiles are extracted at the appropriate resolution to achieve the target mpp
    - Output tiles are always `width` x `height` pixels (consistent tensor dimensions)
    - Each tile represents `width * target_mpp` x `height * target_mpp` microns
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
    ├── properties.json        # tile properties
    ├── thumbnail.jpeg         # thumbnail image (or .png if JPEG unsupported)
    ├── thumbnail_tiles.jpeg   # thumbnail with tiles (or .png if JPEG unsupported)
    ├── thumbnail_tissue.jpeg  # thumbnail of the tissue mask (or .png if JPEG unsupported)
    └── tiles
```

!!! note
    If Pillow lacks JPEG support in your environment, HistoSlice will write `.png` files
    and update filenames accordingly. Developers can check support via
    `histoslice.functional.has_jpeg_support()`.

!!! note
    If any tiles fail during extraction, the CLI prints a warning and writes
    `failures.json` with per-tile error details.

![Prostate biopsy sample](https://github.com/rmuraix/HistoSlice/blob/main/images/thumbnail.jpeg?raw=true)
![Tissue mask](https://github.com/rmuraix/HistoSlice/blob/main/images/thumbnail_tissue.jpeg?raw=true)
![Thumbnail with tiles](https://github.com/rmuraix/HistoSlice/blob/main/images/thumbnail_tiles.jpeg?raw=true)

### Remove Bad Tiles

Histological slide images often contain areas that we would not like to include into our training data. Might seem like a daunting task but let's try it out!

=== "CLI"
    ```bash
    # First, extract tiles with metrics
    histoslice slice \
        --input './images/*.tiff' \
        --output ./tiles \
        --width 512 \
        --metrics
    
    # Then, detect and remove outliers using clustering
    # Specify the parent directory containing slide outputs
    histoslice clean \
        --input './tiles/*' \
        --num-clusters 4
    
    # Or delete outliers instead of moving them
    histoslice clean \
        --input './tiles/*' \
        --num-clusters 4 \
        --delete
    
    # For parallel processing of multiple slides
    histoslice clean \
        --input './tiles/*' \
        --num-clusters 4 \
        --num-workers 4
    ```

=== "Python API"
    ```python
    from histoslice.utils import OutlierDetector

    # Let's wrap the tile metadata with a helper class.
    detector = OutlierDetector(tile_metadata)
    # Cluster tiles based on image metrics.
    clusters = detector.cluster_kmeans(num_clusters=4, random_state=666)
    # Visualise first cluster.
    reader.get_annotated_thumbnail(
        image=reader.read_level(-1), coordinates=detector.coordinates[clusters == 0]
    )
    ```

Now we can mark tiles in cluster `0` as outliers!

![Tiles in cluster 0](https://github.com/rmuraix/HistoSlice/blob/main/images/thumbnail_blue.jpeg?raw=true)

The `clean` command automatically detects outliers in cluster 0 (the cluster most distant from the mean cluster center after k-means clustering orders them by distance) and either moves them to an `outliers` subdirectory (default) or deletes them (with `--delete` flag). The command supports parallel processing of multiple slides using the `--num-workers` option.

For more information on how to use the `OutlierDetector`, see the [API documentation](api/public/outlierdetector/).
