<div align="center">

# HistoSlice

[![PyPI - Version](https://img.shields.io/pypi/v/histoslice)](https://pypi.org/project/histoslice/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/histoslice)](https://pypi.org/project/histoslice/)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue?logo=docker)](https://github.com/rmuraix/HistoSlice/pkgs/container/histoslice)
[![GitHub License](https://img.shields.io/github/license/rmuraix/HistoSlice)](./LICENSE)
[![Check](https://github.com/rmuraix/HistoSlice/actions/workflows/check.yaml/badge.svg)](https://github.com/rmuraix/HistoSlice/actions/workflows/check.yaml)
[![Docs](https://github.com/rmuraix/HistoSlice/actions/workflows/docs.yaml/badge.svg)](https://github.com/rmuraix/HistoSlice/actions/workflows/docs.yaml)
[![codecov](https://codecov.io/github/rmuraix/HistoSlice/graph/badge.svg?token=NDSf4tDhzF)](https://codecov.io/github/rmuraix/HistoSlice)

Preprocessing large medical images for machine learning made easy!

<p align="center">
  <a href="https://lab.rmurai.com/HistoSlice/">Documentation</a> •
  <a href="https://pypi.org/project/histoslice/">PyPI</a>
</p>

</div>

## Description

`HistoSlice` makes is easy to prepare your histological slide images for deep
learning models. You can easily cut large slide images into smaller tiles and then
preprocess those tiles (remove tiles with shitty tissue, finger marks etc).

> [!NOTE]
> This project was forked from [HistoPrep](https://github.com/jopo666/HistoPrep), and further modified for additional features and improvements.

## Installation

```bash
uv add histoslice
# or
pip install histoslice
```

### Docker

You can also use HistoSlice via Docker, which comes with all dependencies pre-installed:

```bash
# Pull the latest image
docker pull ghcr.io/rmuraix/histoslice:latest

# Run CLI (mount your data directory)
docker run --rm -v $(pwd)/slides:/data/input -v $(pwd)/output:/data/output \
  ghcr.io/rmuraix/histoslice:latest \
  --input '/data/input/*.tiff' --output /data/output --width 512 --overlap 0.5

# Use Python API inside container
docker run --rm -it -v $(pwd):/workspace -w /workspace \
  ghcr.io/rmuraix/histoslice:latest \
  python your_script.py
```

## Usage

> [!NOTE]
> HistoSlice uses **pyvips** as the only slide backend. The `backend` argument is still accepted for compatibility, but it always resolves to pyvips.
>
> If Pillow is built without JPEG support, HistoSlice will automatically save tiles/thumbnails as `.png`
> and update filenames accordingly. Developers can check availability via `histoslice.functional.has_jpeg_support()`.

Typical workflow for training deep learning models with histological images is the
following:

1. Cut each slide image into smaller tile images.
2. Preprocess smaller tile images by removing tiles with bad tissue, staining artifacts.

```bash
histoslice --input './train_images/*.tiff' --output ./tiles --width 512 --overlap 0.5 --max-background 0.5 --backend pyvips --metrics --thumbnail
```

Or you can use the `HistoSlice` python API to do the same thing!

```python
from histoslice import SlideReader

# Read slide image.
reader = SlideReader("./slides/slide_with_ink.jpeg")
# Detect tissue.
threshold, tissue_mask = reader.get_tissue_mask(level=-1)
# Extract overlapping tile coordinates with less than 50% background.
tile_coordinates = reader.get_tile_coordinates(
    tissue_mask, width=512, overlap=0.5, max_background=0.5
)
# Save tile images with image metrics for preprocessing.
tile_metadata, failures = reader.save_regions(
    "./train_tiles/",
    tile_coordinates,
    threshold=threshold,
    save_metrics=True,
    save_thumbnail=True
)
if failures:
    print(f"Some tiles failed: {len(failures)}")
```

Let's take a look at the output and visualise the thumbnails.

```bash
train_tiles
└── slide_with_ink
    ├── metadata.parquet       # tile metadata
    ├── failures.json          # per-tile failures (only written if any failures occur)
    ├── properties.json        # tile properties
    ├── thumbnail.jpeg         # thumbnail image (or .png if JPEG support is unavailable)
    ├── thumbnail_tiles.jpeg   # thumbnail with tiles (or .png if JPEG support is unavailable)
    ├── thumbnail_tissue.jpeg  # thumbnail of the tissue mask (or .png if JPEG support is unavailable)
    └── tiles [390 entries exceeds filelimit, not opening dir]
```

![Prostate biopsy sample](https://github.com/rmuraix/HistoSlice/raw/main/images/thumbnail.jpeg)
![Tissue mask](https://github.com/rmuraix/HistoSlice/raw/main/images/thumbnail_tissue.jpeg)
![Thumbnail with tiles](https://github.com/rmuraix/HistoSlice/raw/main/images/thumbnail_tiles.jpeg)

As we can see from the above images, histological slide images often contain areas that
we would not like to include into our training data. Might seem like a daunting task but
let's try it out!

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

![Tiles in cluster 0](https://github.com/rmuraix/HistoSlice/raw/main/images/thumbnail_blue.jpeg)

Now we can mark tiles in cluster `0` as outliers!
