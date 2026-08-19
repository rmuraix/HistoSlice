<div align="center">

# HistoSlice

[![PyPI - Version](https://img.shields.io/pypi/v/histoslice)](https://pypi.org/project/histoslice/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/histoslice)](https://pypi.org/project/histoslice/)
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

## Usage

> [!NOTE]
> HistoSlice reads slides through **pyvips**/libvips only - there is no backend to choose.
>
> If the underlying `libvips` build lacks JPEG support, HistoSlice will automatically save
> tiles/thumbnails as `.png` and update filenames accordingly. Developers can check
> availability via `histoslice.functional.has_jpeg_support()`.

Typical workflow for training deep learning models with histological images is the
following:

1. Cut each slide image into smaller tile images.
2. Preprocess smaller tile images by removing tiles with bad tissue, staining artifacts.

```bash
histoslice slice --input './train_images/*.tiff' --output ./tiles --width 512 --overlap 0.5 --max-background 0.5 --metrics --thumbnails
```

Or use the `histoslice` Python API to do the same thing, one call for the common case:

```python
from histoslice import slice_slide

result = slice_slide(
    "./slides/slide_with_ink.jpeg",
    "./train_tiles/",
    tile_size=512,
    overlap=0.5,
    max_background=0.5,
    save_metrics=True,
)
if result.failures:
    print(f"Some tiles failed: {len(result.failures)}")
```

...or compose the same pipeline yourself from its building blocks, when you need more
control over any single step:

```python
from histoslice import Slide, tissue_mask, tile_regions, filter_by_tissue, export_tiles

slide = Slide("./slides/slide_with_ink.jpeg")
threshold, mask = tissue_mask(slide.read_level(-1))
regions = tile_regions(slide.dimensions, size=512, overlap=0.5)
regions = filter_by_tissue(regions, mask, slide_dimensions=slide.dimensions, max_background=0.5)
result = export_tiles(slide, regions, "./train_tiles/", tile_size=512, threshold=threshold, save_metrics=True)
```

Let's take a look at the output and visualise the thumbnails.

```bash
train_tiles
└── slide_with_ink
    ├── metadata.parquet       # tile metadata
    ├── failures.json          # per-tile failures (only written if any failures occur)
    ├── thumbnail.jpeg         # thumbnail image (or .png if JPEG support is unavailable)
    ├── thumbnail_tiles.jpeg   # thumbnail with tiles (or .png if JPEG support is unavailable)
    ├── thumbnail_tissue.jpeg  # thumbnail of the tissue mask (or .png if JPEG support is unavailable)
    └── tiles [390 entries exceeds filelimit, not opening dir]
```

![Prostate biopsy sample](https://github.com/rmuraix/HistoSlice/raw/main/images/thumbnail.jpeg)
![Tissue mask](https://github.com/rmuraix/HistoSlice/raw/main/images/thumbnail_tissue.jpeg)
![Thumbnail with tiles](https://github.com/rmuraix/HistoSlice/raw/main/images/thumbnail_tiles.jpeg)

As we can see from the above images, histological slide images often contain tiles with
technical problems - corrupted reads, blown-out exposure, near-blank scans. Let's run
technical quality control (QC) to flag those, without treating biologically unusual (but
valid) tissue as a problem:

```bash
histoslice clean --input './train_tiles/*'
```

This writes a `metadata_clean.parquet` next to `metadata.parquet`, with the original
columns plus `qc_status` (`"pass"`/`"warn"`/`"fail"`), `qc_score`, `qc_reasons`,
`is_outlier` (clear technical failures, e.g. corrupted/near-black/near-white tiles), and
`needs_review` (possible artifacts, kept by default - review before dropping).
`is_outlier` is never set just because a tile looks biologically unusual (tumor, stroma,
adipose, necrosis, mucin, ...) - see the [metadata reference](https://lab.rmurai.com/HistoSlice/metadata/)
for details.

For interactive exploration/visualisation of the full metric set (not technical QC), use
`OutlierDetector`:

```python
from histoslice.utils import OutlierDetector

# Let's wrap the tile metadata with a helper class.
detector = OutlierDetector(result.metadata)
# Cluster tiles based on image metrics.
clusters = detector.cluster_kmeans(num_clusters=4, random_state=666)
# Visualise the first cluster.
detector.random_image_collage(clusters == 0)
```

![Tiles in cluster 0](https://github.com/rmuraix/HistoSlice/raw/main/images/thumbnail_blue.jpeg)
