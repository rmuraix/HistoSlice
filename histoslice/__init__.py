"""HistoSlice: read and process histological whole slide images."""

__all__ = [
    "ExportResult",
    "Region",
    "Slide",
    "SlideReader",
    "TileSpec",
    "clean_tissue_mask",
    "export_tiles",
    "filter_by_tissue",
    "functional",
    "slice_slide",
    "tile_regions",
    "tissue_mask",
    "utils",
]

from histoslice import functional, utils
from histoslice.api import slice_slide
from histoslice.export import ExportResult, export_tiles
from histoslice.slide import Slide
from histoslice.tiles import Region, TileSpec, filter_by_tissue, tile_regions
from histoslice.tissue import clean_tissue_mask, tissue_mask

# Deprecated alias: `Slide` replaces the old `SlideReader`/`SlideReaderBackend`
# split. Tissue detection, tile generation and saving are no longer methods on
# this class - see `tissue_mask`, `tile_regions`, `filter_by_tissue` and
# `export_tiles` (or the `slice_slide` convenience function).
SlideReader = Slide
