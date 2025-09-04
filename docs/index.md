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
    tile_metadata = reader.save_regions(
        "./tiles/",
        tile_coordinates,
        threshold=threshold,
        save_metrics=True,
        save_thumbnail=True
    )
    ```

Output directory structure will look like this:

```bash
tiles
└── slide_id
    ├── metadata.parquet       # tile metadata
    ├── properties.json        # tile properties
    ├── thumbnail.jpeg         # thumbnail image
    ├── thumbnail_tiles.jpeg   # thumbnail with tiles
    ├── thumbnail_tissue.jpeg  # thumbnail of the tissue mask
    └── tiles
```

![Prostate biopsy sample](https://github.com/rmuraix/HistoSlice/blob/main/images/thumbnail.jpeg?raw=true)
![Tissue mask](https://github.com/rmuraix/HistoSlice/blob/main/images/thumbnail_tissue.jpeg?raw=true)
![Thumbnail with tiles](https://github.com/rmuraix/HistoSlice/blob/main/images/thumbnail_tiles.jpeg?raw=true)

### Remove Bad Tiles

Histological slide images often contain areas that we would not like to include into our training data. Might seem like a daunting task but let's try it out!

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

For more information on how to use the `OutlierDetector`, see the [API documentation](api/public/outlierdetector/).
