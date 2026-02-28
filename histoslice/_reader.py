import functools
import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import polars as pl
import tqdm
from PIL import Image

import histoslice.functional as F
from histoslice.functional._images import downscale_to_max_pixels, has_jpeg_support
from histoslice._backend import PyVipsBackend
from histoslice._data import SpotCoordinates, TileCoordinates
from histoslice.functional._concurrent import close_pool, prepare_worker_pool
from histoslice.functional._level import format_level
from histoslice.functional._tiles import _multiply_xywh

ERROR_WRONG_TYPE = "Expected '{}' to be of type {}, not {}."
ERROR_NO_THRESHOLD = "Threshold argument is required to save masks/metrics."
ERROR_OUTPUT_DIR_IS_FILE = "Output directory exists but it is a file."
ERROR_CANNOT_OVERWRITE = "Output directory exists, but `overwrite=False`."


class SlideReader:
    """Reader class for histological slide images."""

    def __init__(
        self,
        path: Union[str, Path],
        mpp: Optional[tuple[float, float]] = None,
    ) -> None:
        """Initialize `SlideReader` instance.

        Args:
            path: Path to slide image.
            mpp: Override microns per pixel as (mpp_x, mpp_y). If None, attempts to
                extract from slide metadata. Defaults to None.

        Raises:
            FileNotFoundError: Path does not exist.
        """
        super().__init__()
        self._backend = _read_slide(path=path)
        self._mpp_override = mpp

    @property
    def path(self) -> str:
        """Full slide filepath."""
        return self._backend.path

    @property
    def name(self) -> str:
        """Slide filename without an extension."""
        return self._backend.name

    @property
    def suffix(self) -> str:
        """Slide file-extension."""
        return self._backend.suffix

    @property
    def backend_name(self) -> str:
        """Name of the slide reader backend."""
        return self._backend.BACKEND_NAME

    @property
    def data_bounds(self) -> tuple[int, int, int, int]:
        """Data bounds defined by `xywh`-coordinates at `level=0`.

        Some image formats (eg. `.mrxs`) define a bounding box where image data resides,
        which may differ from the actual image dimensions. `HistoPrep` always uses the
        full image dimensions, but other software (such as `QuPath`) uses the image
        dimensions defined by this data bound.
        """
        return self._backend.data_bounds

    @property
    def dimensions(self) -> tuple[int, int]:
        """Image dimensions (height, width) at `level=0`."""
        return self._backend.dimensions

    @property
    def level_count(self) -> int:
        """Number of slide pyramid levels."""
        return self._backend.level_count

    @property
    def level_dimensions(self) -> dict[int, tuple[int, int]]:
        """Image dimensions (height, width) for each pyramid level."""
        return self._backend.level_dimensions

    @property
    def level_downsamples(self) -> dict[int, tuple[float, float]]:
        """Image downsample factors (height, width) for each pyramid level."""
        return self._backend.level_downsamples

    @property
    def mpp(self) -> tuple[float, float] | None:
        """Microns per pixel (mpp_x, mpp_y) at level 0.

        Returns user-provided override if available, otherwise extracts from
        slide metadata. Returns None if not available.
        """
        if self._mpp_override is not None:
            return self._mpp_override
        return self._backend.mpp

    def read_level(self, level: int) -> np.ndarray:
        """Read full pyramid level data.

        Args:
            level: Slide pyramid level to read.

        Raises:
            ValueError: Invalid level argument.

        Returns:
            Array containing image data from `level`.
        """
        return self._backend.read_level(level=level)

    def read_region(
        self, xywh: tuple[int, int, int, int], level: int = 0
    ) -> np.ndarray:
        """Read region based on `xywh`-coordinates.

        Args:
            xywh: Coordinates for the region.
            level: Slide pyramid level to read from. Defaults to 0.

        Raises:
            ValueError: Invalid `level` argument.

        Returns:
            Array containing image data from `xywh`-region.
        """
        return self._backend.read_region(xywh=xywh, level=level)

    def level_from_max_dimension(self, max_dimension: int = 4096) -> int:
        """Find pyramid level with *both* dimensions less or equal to `max_dimension`.
        If one isn't found, return the last pyramid level.

        Args:
            max_dimension: Maximum dimension for the level. Defaults to 4096.

        Returns:
            Slide pyramid level.
        """
        for level, (level_h, level_w) in self.level_dimensions.items():
            if level_h <= max_dimension and level_w <= max_dimension:
                return level
        return list(self.level_dimensions.keys())[-1]

    def level_from_dimensions(self, dimensions: tuple[int, int]) -> int:
        """Find pyramid level which is closest to `dimensions`.

        Args:
            dimensions: Height and width.

        Returns:
            Slide pyramid level.
        """
        height, width = dimensions
        available = []
        distances = []
        for level, (level_h, level_w) in self.level_dimensions.items():
            available.append(level)
            distances.append(abs(level_h - height) + abs(level_w - width))
        return available[distances.index(min(distances))]

    def get_tissue_mask(
        self,
        *,
        level: Optional[int] = None,
        threshold: Optional[int] = None,
        multiplier: float = 1.05,
        sigma: float = 0.0,
    ) -> tuple[int, np.ndarray]:
        """Detect tissue from slide pyramid level image.

        Args:
            level: Slide pyramid level to use for tissue detection. If None, uses the
                `level_from_max_dimension` method. Defaults to None.
            threshold: Threshold for tissue detection. If set, will detect tissue by
                global thresholding. Otherwise Otsu's method is used to find a
                threshold. Defaults to None.
            multiplier: Otsu's method finds an optimal threshold by minimizing the
                weighted within-class variance. This threshold is then multiplied with
                `multiplier`. Ignored if `threshold` is not None. Defaults to 1.0.
            sigma: Sigma for gaussian blurring. Defaults to 0.0.

        Raises:
            ValueError: Threshold not between 0 and 255.

        Returns:
            Threshold and tissue mask.
        """
        level = (
            self.level_from_max_dimension()
            if level is None
            else format_level(level, available=list(self.level_dimensions))
        )
        return F.get_tissue_mask(
            image=self.read_level(level),
            threshold=threshold,
            multiplier=multiplier,
            sigma=sigma,
        )

    def get_tile_coordinates(
        self,
        tissue_mask: Optional[np.ndarray],
        width: int,
        *,
        height: Optional[int] = None,
        target_mpp: Optional[float] = None,
        overlap: float = 0.0,
        max_background: float = 0.95,
        out_of_bounds: bool = True,
    ) -> TileCoordinates:
        """Generate tile coordinates.

        Args:
            tissue_mask: Tissue mask for filtering tiles with too much background. If
                None, the filtering is disabled.
            width: Width of a tile in pixels at target resolution.
            height: Height of a tile in pixels at target resolution. If None, will be
                set to `width`. Defaults to None.
            target_mpp: Target microns per pixel for normalization. If specified, tiles
                are extracted at the appropriate level to achieve this resolution. The
                output tiles will be `width` x `height` pixels representing a physical
                area of `width * target_mpp` x `height * target_mpp` microns.
                Defaults to None (use native slide resolution).
            overlap: Overlap between neighbouring tiles. Defaults to 0.0.
            max_background: Maximum proportion of background in tiles. Ignored if
                `tissue_mask` is None. Defaults to 0.95.
            out_of_bounds: Keep tiles which contain regions outside of the image.
                Defaults to True.

        Raises:
            ValueError: `target_mpp` specified but slide mpp not available.
            ValueError: Height and/or width are smaller than 1.
            ValueError: Height and/or width is larger than dimensions.
            ValueError: Overlap is not in range [0, 1).

        Returns:
            `TileCoordinates` dataclass.
        """
        # Handle target_mpp parameter for resolution normalization
        if target_mpp is not None:
            slide_mpp = self.mpp
            if slide_mpp is None:
                raise ValueError(
                    "Target mpp specified but slide mpp not available. "
                    "Provide mpp to SlideReader constructor or omit target_mpp."
                )
            # Calculate scaling factor: target_mpp / slide_mpp
            # Physical size = width * target_mpp (e.g., 512px * 0.25mpp = 128µm)
            # At slide resolution: need (width * target_mpp) / slide_mpp pixels
            # Example: 512px at 0.25mpp target, slide at 0.5mpp → 256px needed
            avg_slide_mpp = (slide_mpp[0] + slide_mpp[1]) / 2.0
            scale = target_mpp / avg_slide_mpp

            # Scale width/height to extract at native resolution
            # These will represent the desired physical size
            width = int(round(width * scale))
            if height is not None:
                height = int(round(height * scale))

        tile_coordinates = F.get_tile_coordinates(
            dimensions=self.dimensions,
            width=width,
            height=height,
            overlap=overlap,
            out_of_bounds=out_of_bounds,
        )
        if tissue_mask is not None:
            all_backgrounds = F.get_background_percentages(
                tile_coordinates=tile_coordinates,
                tissue_mask=tissue_mask,
                downsample=F.get_downsample(tissue_mask, self.dimensions),
            )
            filtered_coordinates = []
            for xywh, background in zip(tile_coordinates, all_backgrounds):
                if background <= max_background:
                    filtered_coordinates.append(xywh)
            tile_coordinates = filtered_coordinates
        return TileCoordinates(
            coordinates=tile_coordinates,
            width=width,
            height=width if height is None else height,
            overlap=overlap,
            max_background=None if tissue_mask is None else max_background,
            tissue_mask=tissue_mask,
        )

    def get_spot_coordinates(
        self,
        tissue_mask: np.ndarray,
        *,
        min_area_pixel: int = 10,
        max_area_pixel: Optional[int] = None,
        min_area_relative: float = 0.2,
        max_area_relative: Optional[float] = 2.0,
    ) -> SpotCoordinates:
        """Generate tissue microarray spot coordinates.

        Args:
            tissue_mask: Tissue mask of the slide. It's recommended to increase `sigma`
                value when detecting tissue to remove non-TMA spots from the mask. Rest
                of the areas can be handled with the following arguments.
            min_area_pixel: Minimum pixel area for contours. Defaults to 10.
            max_area_pixel: Maximum pixel area for contours. Defaults to None.
            min_area_relative: Relative minimum contour area, calculated from the median
                contour area after filtering contours with `[min,max]_pixel` arguments
                (`min_area_relative * median(contour_areas)`). Defaults to 0.2.
            max_area_relative: Relative maximum contour area, calculated from the median
                contour area after filtering contours with `[min,max]_pixel` arguments
                (`max_area_relative * median(contour_areas)`). Defaults to 2.0.

        Returns:
            `TMASpotCoordinates` instance.
        """
        spot_mask = F.clean_tissue_mask(
            tissue_mask=tissue_mask,
            min_area_pixel=min_area_pixel,
            max_area_pixel=max_area_pixel,
            min_area_relative=min_area_relative,
            max_area_relative=max_area_relative,
        )
        # Dearray spots.
        spot_info = F.get_spot_coordinates(spot_mask)
        spot_coordinates = [  # upsample to level zero.
            _multiply_xywh(x, F.get_downsample(tissue_mask, self.dimensions))
            for x in spot_info.values()
        ]
        return SpotCoordinates(
            coordinates=spot_coordinates,
            spot_names=list(spot_info.keys()),
            tissue_mask=spot_mask,
        )

    def get_annotated_thumbnail(
        self,
        image: np.ndarray,
        coordinates: Iterator[tuple[int, int, int, int]],
        linewidth: int = 1,
    ) -> Image.Image:
        """Generate annotated thumbnail from coordinates.

        Args:
            image: Input image.
            coordinates: Coordinates to annotate.
            linewidth: Width of rectangle lines.

        Returns:
            Annotated thumbnail.
        """
        kwargs = {
            "image": image,
            "downsample": F.get_downsample(image, self.dimensions),
            "rectangle_width": linewidth,
        }
        if isinstance(coordinates, SpotCoordinates):
            text_items = [x.lstrip("spot_") for x in coordinates.spot_names]
            kwargs.update(
                {"coordinates": coordinates.coordinates, "text_items": text_items}
            )
        elif isinstance(coordinates, TileCoordinates):
            kwargs.update(
                {"coordinates": coordinates.coordinates, "highlight_first": True}
            )
        else:
            kwargs.update({"coordinates": coordinates})
        return F.get_annotated_image(**kwargs)

    def yield_regions(
        self,
        coordinates: Iterator[tuple[int, int, int, int]],
        *,
        level: int = 0,
        transform: Optional[Callable[[np.ndarray], Any]] = None,
        num_workers: int = 1,
        return_exception: bool = False,
    ) -> Iterator[tuple[Union[np.ndarray, Exception, Any], tuple[int, int, int, int]]]:
        """Yield tile images and corresponding xywh coordinates.

        Args:
            coordinates: List of xywh-coordinates.
            level: Slide pyramid level for reading tile images. Defaults to 0.
            transform: Transform function for tile image. Defaults to None.
            num_workers: Number of worker processes. Defaults to 1.
            return_exception: Whether to return exception in case there is a failure to
                read region, instead of raising the exception. Defaults to False.

        Yields:
            Tuple of (possibly transformed) tile image and corresponding
            xywh-coordinate.
        """
        pool, iterable = prepare_worker_pool(
            worker_fn=functools.partial(
                _read_tile,
                level=level,
                transform=transform,
                return_exception=return_exception,
            ),
            reader=self,
            iterable_of_args=((x,) for x in coordinates),
            iterable_length=len(coordinates),
            num_workers=num_workers,
        )
        yield from zip(iterable, coordinates)
        close_pool(pool)

    def get_mean_and_std(
        self,
        coordinates: Iterator[tuple[int, int, int, int]],
        *,
        level: int = 0,
        max_samples: int = 1000,
        num_workers: int = 1,
        raise_exception: bool = True,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Calculate mean and std for each image channel.

        Args:
            coordinates: `TileCoordinates` instance or a list of xywh-coordinates.
            level: Slide pyramid level for reading tile images. Defaults to 0.
            max_samples: Maximum tiles to load. Defaults to 1000.
            num_workers: Number of worker processes for yielding tiles. Defaults to 1.
            raise_exception: Whether to raise an exception if there are problems with
                reading tile regions. Defaults to True.

        Returns:
            Tuples of mean and std values for each image channel.
        """
        if isinstance(coordinates, TileCoordinates):
            coordinates = coordinates.coordinates
        if len(coordinates) > max_samples:
            rng = np.random.default_rng()
            coordinates = rng.choice(
                coordinates, size=max_samples, replace=False
            ).tolist()
        iterable = self.yield_regions(
            coordinates=coordinates,
            level=level,
            num_workers=num_workers,
            return_exception=not raise_exception,
        )
        return F.get_mean_and_std_from_images(
            images=(tile for tile, __ in iterable if not isinstance(tile, Exception))
        )

    def save_regions(
        self,
        parent_dir: Union[str, Path],
        coordinates: Iterator[tuple[int, int, int, int]],
        *,
        level: int = 0,
        threshold: Optional[int] = None,
        tissue_mask: Optional[np.ndarray] = None,
        overwrite: bool = False,
        save_metrics: bool = False,
        save_masks: bool = False,
        save_thumbnails: bool = True,
        thumbnail_level: Optional[int] = None,
        image_format: str = "jpeg",
        quality: int = 80,
        num_workers: int = 1,
        raise_exception: bool = True,
        verbose: bool = True,
    ) -> tuple[pl.DataFrame, list[dict[str, object]]]:
        """Save regions from an iterable of xywh-coordinates.

        Args:
            parent_dir: Parent directory for output. All output is saved to
                `parent_dir/{self.name}/`.
            coordinates: Iterator of xywh-coordinates.
            level: Slide pyramid level for extracting xywh-regions. Defaults to 0.
            threshold: Tissue detection threshold. Required when either `save_masks` or
                `save_metrics` is True. Defaults to None.
            overwrite: Overwrite everything in `parent_dir/{slide_name}/` if it exists.
                Defaults to False.
            save_metrics: Save image metrics to metadata, requires that threshold is
                set. Defaults to False.
            save_masks: Save tissue masks as `png` images, requires that threshold is
                set. Defaults to False.
            save_thumbnails: Save slide thumbnail with and without region annotations.
                Defaults to True.
            tissue_mask: Optional tissue mask to use for thumbnail visualization.
                If None and coordinates contains a tissue mask, that mask is used.
            thumbnail_level: Slide pyramid level for thumbnail images. If None, uses the
                `level_from_max_dimension` method. Ignored when `save_thumbnails=False`.
                Defaults to None.
            image_format: File format for `Pillow` image writer. Defaults to "jpeg".
            quality: JPEG compression quality if `format="jpeg"`. Defaults to 80.
            num_workers: Number of data saving workers. Defaults to 1.
            raise_exception: Whether to raise an exception if there are problems with
                reading tile regions. Defaults to True.
            verbose: Enables `tqdm` progress bar. Defaults to True.

        Raises:
            ValueError: Invalid `level` argument.
            ValueError: Threshold is not between 0 and 255.

        Returns:
            Tuple of (metadata dataframe, failure reports).
        """
        if (save_metrics or save_masks) and threshold is None:
            raise ValueError(ERROR_NO_THRESHOLD)
        level = format_level(level, available=list(self.level_dimensions))
        parent_dir = parent_dir if isinstance(parent_dir, Path) else Path(parent_dir)
        output_dir = _prepare_output_dir(parent_dir / self.name, overwrite=overwrite)
        image_dir = "spots" if isinstance(coordinates, SpotCoordinates) else "tiles"
        # Save properties.
        if isinstance(coordinates, TileCoordinates):
            with (output_dir / "properties.json").open("w") as f:
                json.dump(
                    coordinates.get_properties(
                        level=level, level_downsample=self.level_downsamples[level]
                    ),
                    f,
                )
        actual_image_format = _resolve_image_format(image_format)
        # Save thumbnails.
        if save_thumbnails:
            if thumbnail_level is None:
                thumbnail_level = self.level_from_max_dimension()
            thumbnail = self.read_level(thumbnail_level)

            # Downscale thumbnail if too large to prevent JPEG size limits and reduce disk space
            thumbnail_small = F.downscale_for_thumbnail(thumbnail)
            if actual_image_format == "png":
                thumbnail_small = downscale_to_max_pixels(
                    thumbnail_small, max_pixels=300_000
                )

            _save_image(
                Image.fromarray(thumbnail_small),
                output_dir / f"thumbnail.{actual_image_format}",
                image_format=actual_image_format,
                quality=quality,
            )
            thumbnail_regions = self.get_annotated_thumbnail(
                thumbnail_small, coordinates
            )
            _save_image(
                thumbnail_regions,
                output_dir / f"thumbnail_{image_dir}.{actual_image_format}",
                image_format=actual_image_format,
                quality=quality,
            )
            coords_mask = None
            if isinstance(coordinates, (TileCoordinates, SpotCoordinates)):
                coords_mask = coordinates.tissue_mask
            mask_for_thumbnail = tissue_mask if tissue_mask is not None else coords_mask
            if mask_for_thumbnail is not None:
                # For tissue mask, scale it to match the thumbnail dimensions if needed
                original_tissue_mask = mask_for_thumbnail
                if thumbnail_small.shape[:2] != thumbnail.shape[:2]:
                    # If thumbnail was downscaled, apply the same downscaling to tissue mask
                    scale_h = thumbnail_small.shape[0] / thumbnail.shape[0]
                    scale_w = thumbnail_small.shape[1] / thumbnail.shape[1]
                    new_h = max(1, int(original_tissue_mask.shape[0] * scale_h))
                    new_w = max(1, int(original_tissue_mask.shape[1] * scale_w))
                    tissue_mask_resized = cv2.resize(
                        original_tissue_mask.astype(np.uint8),
                        (new_w, new_h),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    tissue_mask_resized = original_tissue_mask

                _save_image(
                    Image.fromarray(255 - 255 * tissue_mask_resized),
                    output_dir / f"thumbnail_tissue.{actual_image_format}",
                    image_format=actual_image_format,
                    quality=quality,
                )
        metadata, failures = _save_regions(
            output_dir=output_dir,
            iterable=self.yield_regions(
                coordinates=coordinates,
                level=level,
                transform=functools.partial(
                    _load_region_data,
                    save_masks=save_masks,
                    save_metrics=save_metrics,
                    threshold=threshold,
                ),
                num_workers=num_workers,
                return_exception=not raise_exception,
            ),
            desc=self.name,
            total=len(coordinates),
            quality=quality,
            image_format=actual_image_format,
            image_dir=image_dir,
            file_prefixes=coordinates.spot_names
            if isinstance(coordinates, SpotCoordinates)
            else None,
            verbose=verbose,
        )
        metadata.write_parquet(output_dir / "metadata.parquet")
        if failures:
            (output_dir / "failures.json").write_text(json.dumps(failures, indent=2))
        return metadata, failures

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(path={self.path}, "
            f"backend={self._backend.BACKEND_NAME})"
        )


@dataclass
class RegionData:
    """Dataclass representing data for a slide region."""

    image: np.ndarray
    mask: Optional[np.ndarray]
    metrics: dict[str, float]

    def save_data(
        self,
        *,
        image_dir: Path,
        mask_dir: Path,
        xywh: tuple[int, int, int, int],
        quality: int,
        image_format: str,
        prefix: Optional[str],
    ) -> dict[str, float]:
        """Save image (and mask) and return region metadata."""
        metadata = dict(zip("xywh", xywh))
        filename = "x{}_y{}_w{}_h{}".format(*xywh)
        if prefix is not None:
            filename = f"{prefix}_{filename}"
        # Save image.
        image_path = image_dir / f"{filename}.{image_format}"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        _save_image(
            Image.fromarray(self.image),
            image_path,
            image_format=image_format,
            quality=quality,
        )
        metadata["path"] = str(image_path.resolve())
        # Save mask.
        if self.mask is not None:
            mask_path = mask_dir / f"{filename}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(self.mask).save(mask_path, format="PNG")
            metadata["mask_path"] = str(mask_path.resolve())
        return {**metadata, **self.metrics}


def _get_pil_format(image_format: str) -> str:
    fmt = image_format.strip().lower()
    if fmt in ("jpg", "jpeg"):
        return "JPEG"
    if fmt == "png":
        return "PNG"
    if fmt in ("tif", "tiff"):
        return "TIFF"
    return fmt.upper()


def _save_image(
    image: Image.Image,
    path: Path,
    *,
    image_format: str,
    quality: int,
) -> None:
    fmt = image_format.strip().lower()
    pil_format = _get_pil_format(image_format)
    if fmt in ("jpg", "jpeg"):
        img = image.convert("RGB")
        img.save(path, format=pil_format, quality=int(quality))
        return
    image.save(path, format=pil_format)


def _resolve_image_format(image_format: str) -> str:
    fmt = image_format.strip().lower()
    if fmt in ("jpg", "jpeg") and not has_jpeg_support():
        return "png"
    return fmt


def _read_slide(path: Union[str, Path]) -> PyVipsBackend:
    """Read slide using PyVipsBackend.

    Args:
        path: Path to the slide image.

    Raises:
        FileNotFoundError: Path does not exist.

    Returns:
        Slide reader backend.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path.resolve()))
    return PyVipsBackend(path)


def _read_tile(
    worker_state: dict,
    xywh: tuple[int, int, int, int],
    *,
    level: int,
    transform: Optional[Callable[[np.ndarray], Any]],
    return_exception: bool,
) -> Union[np.ndarray, Exception, Any]:
    """Parallisable tile reading function."""
    reader = worker_state["reader"]
    try:
        tile = reader.read_region(xywh=xywh, level=level)
    except KeyboardInterrupt:
        raise KeyboardInterrupt from None
    except Exception as catched_exception:  # noqa
        if not return_exception:
            raise catched_exception  # noqa
        return catched_exception
    if transform is not None:
        return transform(tile)
    return tile


def _prepare_output_dir(output_dir: Union[str, Path], *, overwrite: bool) -> Path:
    """Prepare output directory for saving regions."""
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if output_dir.exists():
        if output_dir.is_file():
            raise NotADirectoryError(ERROR_OUTPUT_DIR_IS_FILE)
        if len(list(output_dir.iterdir())) > 0 and not overwrite:
            raise ValueError(ERROR_CANNOT_OVERWRITE)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_region_data(
    image: np.ndarray,
    *,
    save_masks: bool,
    save_metrics: bool,
    threshold: Optional[int],
) -> RegionData:
    """Helper transform to add tissue mask and image metrics during tile loading."""
    tissue_mask = None
    metrics = {}
    if save_masks or save_metrics:
        __, tissue_mask = F.get_tissue_mask(image=image, threshold=threshold)
    if save_metrics:
        metrics = F.get_image_metrics(image=image, tissue_mask=tissue_mask)
    return RegionData(
        image=image, mask=tissue_mask if save_masks else None, metrics=metrics
    )


def _save_regions(
    output_dir: Path,
    iterable: Iterator[RegionData, tuple[int, int, int, int]],
    *,
    desc: str,
    total: int,
    quality: int,
    image_format: str,
    image_dir: str,
    file_prefixes: list[str],
    verbose: bool,
) -> tuple[pl.DataFrame, list[dict[str, object]]]:
    """Save region data to output directory.

    Args:
        output_dir: Output directory.
        iterable: Iterable yieldin RegionData and xywh-coordinates.
        desc: For the progress bar.
        total: For the progress bar.
        quality: Quality of jpeg-compression.
        image_format: Image extension.
        image_dir: Image directory name
        file_prefixes: List of file prefixes.
        verbose: Enable progress bar.

    Returns:
        Tuple of (metadata dataframe, failure reports).
    """
    progress_bar = tqdm.tqdm(
        iterable=iterable,
        desc=desc,
        disable=not verbose,
        total=total,
    )
    rows = []
    failures: list[dict[str, object]] = []
    num_failed = 0
    for i, (region_data, xywh) in enumerate(progress_bar):
        if isinstance(region_data, Exception):
            num_failed += 1
            progress_bar.set_postfix({"failed": num_failed}, refresh=False)
            failures.append(
                {
                    "xywh": xywh,
                    "error": repr(region_data),
                }
            )
            continue
        rows.append(
            region_data.save_data(
                image_dir=output_dir / image_dir,
                mask_dir=output_dir / "masks",
                xywh=xywh,
                quality=quality,
                image_format=image_format,
                prefix=None if file_prefixes is None else file_prefixes[i],
            )
        )
    return pl.DataFrame(rows), failures
