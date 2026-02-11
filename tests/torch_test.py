from pathlib import Path

import numpy as np
import pytest

try:  # optional dependency for tests
    import torch
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except Exception:  # pragma: no cover - import guard
    HAS_TORCH = False
    torch = None  # type: ignore

from histoslice import SlideReader
from histoslice.utils import (
    SlideReaderDataset,
    TileImageDataset,
)

from ._utils import (
    HAS_PYVIPS_CZI_ASSET,
    SLIDE_PATH_CZI,
    SLIDE_PATH_JPEG,
    TMP_DIRECTORY,
    clean_temporary_directory,
)


def test_posix_paths() -> None:
    if not HAS_TORCH:
        return pytest.skip("PyTorch is not installed")
    clean_temporary_directory()
    reader = SlideReader(SLIDE_PATH_JPEG)
    metadata, _ = reader.save_regions(TMP_DIRECTORY, reader.get_tile_coordinates(None, 96))
    dataset = TileImageDataset(
        paths=[Path(x) for x in metadata["path"].to_list()],
        labels=metadata[list("xywh")].to_numpy(),
        transform=lambda x: x[..., 0],
    )
    next(iter(DataLoader(dataset, batch_size=32)))


def test_reader_dataset_loader_pyvips() -> None:
    if not HAS_TORCH:
        return pytest.skip("PyTorch is not installed")
    reader = SlideReader(SLIDE_PATH_JPEG)
    __, tissue_mask = reader.get_tissue_mask()
    coords = reader.get_tile_coordinates(tissue_mask, 512, max_background=0.01)
    dataset = SlideReaderDataset(reader, coords, level=1, transform=lambda z: z)
    assert isinstance(dataset, Dataset)
    loader = DataLoader(dataset, batch_size=4, num_workers=0, drop_last=True)
    for i, (batch_images, batch_coords) in enumerate(loader):
        assert batch_images.shape == (4, 256, 256, 3)
        assert isinstance(batch_images, torch.Tensor)
        assert batch_coords.shape == (4, 4)
        assert isinstance(batch_coords, torch.Tensor)
        if i > 20:
            break


def test_reader_dataset_loader_czi() -> None:
    if not HAS_TORCH:
        return pytest.skip("PyTorch is not installed")
    if not HAS_PYVIPS_CZI_ASSET:
        return pytest.skip("PyVips or CZI test data missing")
    try:
        reader = SlideReader(SLIDE_PATH_CZI)
    except Exception:
        return pytest.skip("PyVips cannot read CZI in this environment")
    __, tissue_mask = reader.get_tissue_mask()
    coords = reader.get_tile_coordinates(tissue_mask, 512, max_background=0.01)
    dataset = SlideReaderDataset(reader, coords, level=1, transform=lambda z: z)
    assert isinstance(dataset, Dataset)
    loader = DataLoader(dataset, batch_size=4, num_workers=0, drop_last=True)
    for i, (batch_images, batch_coords) in enumerate(loader):
        assert batch_images.shape == (4, 256, 256, 3)
        assert isinstance(batch_images, torch.Tensor)
        assert batch_coords.shape == (4, 4)
        assert isinstance(batch_coords, torch.Tensor)
        if i > 20:
            break


def test_tile_dataset_loader() -> None:
    if not HAS_TORCH:
        return pytest.skip("PyTorch is not installed")
    clean_temporary_directory()
    reader = SlideReader(SLIDE_PATH_JPEG)
    metadata, _ = reader.save_regions(TMP_DIRECTORY, reader.get_tile_coordinates(None, 96))
    dataset = TileImageDataset(
        metadata["path"].to_numpy(),
        labels=metadata[list("xywh")].to_numpy(),
        transform=lambda x: x[..., 0],
    )
    batch_images, batch_paths, batch_coords = next(
        iter(DataLoader(dataset, batch_size=32))
    )
    clean_temporary_directory()
    assert batch_images.shape == (32, 96, 96)
    assert len(batch_paths) == 32
    assert batch_coords.shape == (32, 4)


def test_tile_dataset_cache() -> None:
    if not HAS_TORCH:
        return pytest.skip("PyTorch is not installed")
    clean_temporary_directory()
    reader = SlideReader(SLIDE_PATH_JPEG)
    metadata, _ = reader.save_regions(TMP_DIRECTORY, reader.get_tile_coordinates(None, 96))
    dataset = TileImageDataset(
        metadata["path"].to_numpy(),
        labels=metadata[list("xywh")].to_numpy(),
        transform=lambda x: x[..., 0],
        use_cache=True,
        tile_shape=(96, 96, 3),
    )
    batch_images, batch_paths, batch_coords = next(
        iter(DataLoader(dataset, batch_size=32))
    )
    clean_temporary_directory()
    assert batch_images.shape == (32, 96, 96)
    assert len(batch_paths) == 32
    assert batch_coords.shape == (32, 4)
    assert dataset._cached_indices == set(range(32))
    assert np.equal(dataset._cache_array[0][..., 0], batch_images[0].numpy()).all()


def test_tile_dataset_no_labels() -> None:
    """Test TileImageDataset without labels."""
    if not HAS_TORCH:
        return pytest.skip("PyTorch is not installed")
    clean_temporary_directory()
    reader = SlideReader(SLIDE_PATH_JPEG)
    metadata, _ = reader.save_regions(TMP_DIRECTORY, reader.get_tile_coordinates(None, 96))
    dataset = TileImageDataset(
        metadata["path"].to_numpy(),
        labels=None,
    )
    batch_images, batch_paths = next(iter(DataLoader(dataset, batch_size=4)))
    clean_temporary_directory()
    assert batch_images.shape == (4, 96, 96, 3)
    assert len(batch_paths) == 4


def test_tile_dataset_label_length_mismatch() -> None:
    """Test TileImageDataset raises error when labels length doesn't match paths."""
    if not HAS_TORCH:
        return pytest.skip("PyTorch is not installed")
    clean_temporary_directory()
    reader = SlideReader(SLIDE_PATH_JPEG)
    metadata, _ = reader.save_regions(TMP_DIRECTORY, reader.get_tile_coordinates(None, 96))
    paths = metadata["path"].to_numpy()

    with pytest.raises(ValueError, match="Path length .* does not match label length"):
        TileImageDataset(paths=paths, labels=["label1", "label2"])
    clean_temporary_directory()


def test_tile_dataset_cache_without_shape() -> None:
    """Test TileImageDataset raises error when use_cache=True but tile_shape=None."""
    if not HAS_TORCH:
        return pytest.skip("PyTorch is not installed")
    clean_temporary_directory()
    reader = SlideReader(SLIDE_PATH_JPEG)
    metadata, _ = reader.save_regions(TMP_DIRECTORY, reader.get_tile_coordinates(None, 96))

    with pytest.raises(ValueError, match="Tile shape must be defined"):
        TileImageDataset(
            paths=metadata["path"].to_numpy(),
            use_cache=True,
            tile_shape=None,
        )
    clean_temporary_directory()


def test_slide_reader_dataset_no_pytorch() -> None:
    """Test SlideReaderDataset raises ImportError when PyTorch is not available."""
    import histoslice.utils._torch

    # Save the original value
    original_has_pytorch = histoslice.utils._torch.HAS_PYTORCH

    # Temporarily set HAS_PYTORCH to False
    histoslice.utils._torch.HAS_PYTORCH = False

    try:
        reader = SlideReader(SLIDE_PATH_JPEG)
        __, tissue_mask = reader.get_tissue_mask()
        coords = reader.get_tile_coordinates(tissue_mask, 512, max_background=0.01)

        with pytest.raises(ImportError, match="Could not import torch"):
            SlideReaderDataset(reader, coords, level=1)
    finally:
        # Restore the original value
        histoslice.utils._torch.HAS_PYTORCH = original_has_pytorch


def test_tile_image_dataset_no_pytorch() -> None:
    """Test TileImageDataset raises ImportError when PyTorch is not available."""
    import histoslice.utils._torch

    # Save the original value
    original_has_pytorch = histoslice.utils._torch.HAS_PYTORCH

    # Temporarily set HAS_PYTORCH to False
    histoslice.utils._torch.HAS_PYTORCH = False

    try:
        with pytest.raises(ImportError, match="Could not import torch"):
            TileImageDataset(paths=["path1.jpg", "path2.jpg"])
    finally:
        # Restore the original value
        histoslice.utils._torch.HAS_PYTORCH = original_has_pytorch
