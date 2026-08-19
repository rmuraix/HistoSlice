"""Lower-level functional utilities.

Tissue detection and tile-region utilities now live in `histoslice.tissue` and
`histoslice.tiles` - kept here as deprecated aliases. Everything else in this
package (drawing, image metrics, TMA dearraying, mean/std estimation) is
unchanged.
"""

__all__ = [
    "clean_tissue_mask",
    "downscale_for_thumbnail",
    "get_annotated_image",
    "get_image_metrics",
    "get_mean_and_std_from_images",
    "get_mean_and_std_from_paths",
    "get_qc_metrics",
    "get_random_image_collage",
    "get_spot_coordinates",
    "get_tissue_mask",
    "has_jpeg_support",
]

from histoslice.tissue import clean_tissue_mask, downscale_for_thumbnail
from histoslice.tissue import tissue_mask as get_tissue_mask

from ._dearray import get_spot_coordinates
from ._draw import get_annotated_image
from ._images import get_random_image_collage, has_jpeg_support
from ._mean_std import get_mean_and_std_from_images, get_mean_and_std_from_paths
from ._metrics import get_image_metrics, get_qc_metrics
