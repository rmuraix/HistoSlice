import numpy as np
import pytest

from histoslice import Slide
from histoslice import functional as F
from histoslice.tiles import (
    Region,
    TileSpec,
    background_percentages,
    filter_by_tissue,
    get_downsample,
    level0_tile_size,
    pad_to_shape,
    region_from_array,
    spot_regions,
    tile_regions,
)

from ._utils import SLIDE_PATH_JPEG, SLIDE_PATH_TMA

KWARGS = {"dimensions": (100, 80), "size": (40, 30)}


def test_tile_regions() -> None:
    assert tile_regions(**KWARGS, out_of_bounds=False) == [
        Region(0, 0, 40, 30),
        Region(40, 0, 40, 30),
        Region(0, 30, 40, 30),
        Region(40, 30, 40, 30),
        Region(0, 60, 40, 30),
        Region(40, 60, 40, 30),
    ]


def test_tile_regions_out_of_bounds() -> None:
    assert tile_regions(**KWARGS, out_of_bounds=True) == [
        Region(0, 0, 40, 30),
        Region(40, 0, 40, 30),
        Region(0, 30, 40, 30),
        Region(40, 30, 40, 30),
        Region(0, 60, 40, 30),
        Region(40, 60, 40, 30),
        Region(0, 90, 40, 30),
        Region(40, 90, 40, 30),
    ]


def test_tile_regions_square_size() -> None:
    assert tile_regions((100, 100), size=50, out_of_bounds=False) == [
        Region(0, 0, 50, 50),
        Region(50, 0, 50, 50),
        Region(0, 50, 50, 50),
        Region(50, 50, 50, 50),
    ]


def test_tile_regions_overlap() -> None:
    assert tile_regions(**KWARGS, overlap=0.25, out_of_bounds=False) == [
        Region(0, 0, 40, 30),
        Region(30, 0, 40, 30),
        Region(0, 22, 40, 30),
        Region(30, 22, 40, 30),
        Region(0, 44, 40, 30),
        Region(30, 44, 40, 30),
        Region(0, 66, 40, 30),
        Region(30, 66, 40, 30),
    ]


def test_tile_regions_overlap_always_at_least_one_pixel_step() -> None:
    regions = tile_regions(
        (4, 4), size=2, overlap=0.9999999999999999, out_of_bounds=False
    )
    assert len(regions) == 9


def test_tile_regions_bad_inputs() -> None:
    with pytest.raises(ValueError, match="Overlap should be in range"):
        tile_regions(**KWARGS, overlap=1.0)
    with pytest.raises(ValueError, match="Overlap should be in range"):
        tile_regions(**KWARGS, overlap=-1)
    with pytest.raises(ValueError, match="non-zero positive integers"):
        tile_regions((10, 10), size=0)
    with pytest.raises(ValueError, match="should be smaller than image dimensions"):
        tile_regions((10, 10), size=11)


def test_filter_by_tissue() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    __, mask = F.get_tissue_mask(slide.read_level(-1))
    regions = tile_regions(slide.dimensions, 1000, out_of_bounds=False)
    filtered = filter_by_tissue(
        regions, mask, slide_dimensions=slide.dimensions, max_background=0.3
    )
    assert len(filtered) == 3
    assert len(filtered) < len(regions)
    assert all(isinstance(r, Region) for r in filtered)


def test_background_percentages() -> None:
    slide = Slide(SLIDE_PATH_JPEG)
    __, mask = F.get_tissue_mask(slide.read_level(-1), multiplier=1.0)
    regions = tile_regions(slide.dimensions, 1000, out_of_bounds=False)
    percentages = background_percentages(
        regions, mask, get_downsample(mask, slide.dimensions)
    )
    assert [round(p, 6) for p in percentages] == [0.27648, 0.188448, 0.378544, 0.192192]


def test_level0_tile_size_isotropic() -> None:
    assert level0_tile_size(
        (512, 512), slide_mpp=(0.5, 0.5), target_mpp=(0.25, 0.25)
    ) == (
        256,
        256,
    )
    assert level0_tile_size(
        (512, 512), slide_mpp=(0.25, 0.25), target_mpp=(0.5, 0.5)
    ) == (
        1024,
        1024,
    )


def test_level0_tile_size_anisotropic() -> None:
    assert level0_tile_size(
        (512, 256), slide_mpp=(0.25, 0.5), target_mpp=(0.5, 0.5)
    ) == (1024, 256)


def test_tile_spec_native_resolution() -> None:
    spec = TileSpec(size=(512, 512))
    assert spec.level0_size(slide_mpp=None) == (512, 512)
    assert spec.level0_size(slide_mpp=(0.25, 0.25)) == (512, 512)


def test_tile_spec_target_mpp_requires_slide_mpp() -> None:
    spec = TileSpec(size=(512, 512), mpp=(0.5, 0.5))
    with pytest.raises(ValueError, match="target mpp was requested"):
        spec.level0_size(slide_mpp=None)


def test_region_from_array() -> None:
    image = np.arange(9).reshape(3, 3)
    result = region_from_array(image, Region(0, 0, 2, 2))
    assert (result == np.array([[0, 1], [3, 4]])).all()


def test_pad_to_shape_exact_fit_is_a_noop() -> None:
    tile = np.zeros((4, 4, 3), dtype=np.uint8)
    assert pad_to_shape(tile, shape=(4, 4), fill=0) is tile


def test_pad_to_shape_crops_larger_tiles() -> None:
    tile = np.arange(16, dtype=np.uint8).reshape(4, 4)
    cropped = pad_to_shape(tile, shape=(2, 3), fill=0)
    assert cropped.shape == (2, 3)
    assert (cropped == tile[:2, :3]).all()


def test_pad_to_shape_pads_smaller_tiles() -> None:
    tile = np.ones((2, 2), dtype=np.uint8)
    padded = pad_to_shape(tile, shape=(4, 4), fill=9)
    assert padded.shape == (4, 4)
    assert (padded[:2, :2] == 1).all()
    assert (padded[2:, :] == 9).all()
    assert (padded[:, 2:] == 9).all()


def test_spot_regions() -> None:
    slide = Slide(SLIDE_PATH_TMA)
    __, mask = F.get_tissue_mask(slide.read_level(-1), sigma=2.0, threshold=220)
    regions, names = spot_regions(mask, slide.dimensions)
    assert len(regions) == 94
    assert len(names) == 94
    assert all(isinstance(r, Region) for r in regions)


def test_draw_tiles() -> None:
    image = np.zeros((200, 200), dtype=np.uint8)
    coords = [
        r.xywh for r in tile_regions(image.shape[:2], size=40, out_of_bounds=False)
    ]
    img = F.get_annotated_image(
        image=image,
        coordinates=coords,
        downsample=1.0,
        rectangle_fill=None,
        rectangle_outline="red",
        rectangle_width=2,
        highlight_first=True,
        text_items=list(range(len(coords))),
        text_color="white",
        text_proportion=0.8,
        text_font="monospace",
    )
    assert isinstance(img, np.ndarray)
    assert img.shape == (200, 200, 3)
    assert img.dtype == np.uint8

    def is_red(pixels: np.ndarray) -> np.ndarray:
        return (pixels[..., 0] > 150) & (pixels[..., 1] < 100) & (pixels[..., 2] < 100)

    def is_blue(pixels: np.ndarray) -> np.ndarray:
        return (pixels[..., 2] > 150) & (pixels[..., 0] < 100) & (pixels[..., 1] < 100)

    # First tile is drawn with `highlight_outline` ("blue" by default), overwriting
    # its `rectangle_outline` ("red").
    x, y, w, h = coords[0]
    top_edge = img[y : y + 2, x : x + w]
    assert is_blue(top_edge).any()
    assert not is_red(top_edge).any()
    # A later, non-highlighted tile keeps its red outline.
    x, y, w, h = coords[1]
    top_edge = img[y : y + 2, x : x + w]
    assert is_red(top_edge).any()
    # White text was drawn somewhere inside the tiles.
    assert (img.reshape(-1, 3) == 255).all(axis=1).any()
    # Untouched background pixels are still black.
    assert (image == 0).all()
