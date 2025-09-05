from pathlib import Path
from unittest.mock import MagicMock

import pytest

from histoslice.cli._app import cut_slide, filter_slide_paths
from tests._utils import TMP_DIRECTORY, clean_temporary_directory


@pytest.fixture
def mock_typer(monkeypatch):
    """Mock typer.secho and sys.exit."""
    mock_secho = MagicMock()
    mock_exit = MagicMock()
    monkeypatch.setattr("typer.secho", mock_secho)
    monkeypatch.setattr("sys.exit", mock_exit)
    return mock_secho, mock_exit


def setup_test_files(files_to_create):
    """Create dummy files and directories for testing."""
    clean_temporary_directory()
    for file_info in files_to_create:
        path = TMP_DIRECTORY / file_info["name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        if file_info.get("processed"):
            meta_path = (
                TMP_DIRECTORY / path.name.removesuffix(path.suffix) / "metadata.parquet"
            )
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.touch()
        elif file_info.get("interrupted"):
            (TMP_DIRECTORY / path.name.removesuffix(path.suffix)).mkdir(
                parents=True, exist_ok=True
            )


def test_filter_slides_no_slides(mock_typer):
    """Test case where no slides are found."""
    mock_secho, mock_exit = mock_typer
    setup_test_files([])

    paths = filter_slide_paths(
        all_paths=[],
        parent_dir=TMP_DIRECTORY,
        overwrite=False,
        overwrite_unfinished=False,
    )

    assert len(paths) == 0
    mock_exit.assert_called_once_with(1)
    assert "No slides to process" in mock_secho.call_args_list[-2][0][0]


def test_filter_slides_skip_processed(mock_typer):
    """Test skipping already processed slides."""
    mock_secho, _ = mock_typer
    files = [
        {"name": "slide1.jpeg", "processed": True},
        {"name": "slide2.jpeg"},
    ]
    setup_test_files(files)

    all_paths = [TMP_DIRECTORY / f["name"] for f in files]

    result = filter_slide_paths(
        all_paths=all_paths,
        parent_dir=TMP_DIRECTORY,
        overwrite=False,
        overwrite_unfinished=False,
    )

    assert len(result) == 1
    assert result[0].name == "slide2.jpeg"
    mock_secho.assert_any_call("Skipping 1 processed slides.", fg="cyan", bold=True)


def test_filter_slides_overwrite_processed(mock_typer):
    """Test overwriting processed slides."""
    mock_secho, _ = mock_typer
    files = [
        {"name": "slide1.jpeg", "processed": True},
        {"name": "slide2.jpeg"},
    ]
    setup_test_files(files)

    all_paths = [TMP_DIRECTORY / f["name"] for f in files]

    result = filter_slide_paths(
        all_paths=all_paths,
        parent_dir=TMP_DIRECTORY,
        overwrite=True,
        overwrite_unfinished=False,
    )

    assert len(result) == 2
    mock_secho.assert_any_call("Overwriting 1 slide outputs.", fg="yellow", bold=True)


def test_filter_slides_overwrite_unfinished(mock_typer):
    """Test overwriting unfinished slides."""
    mock_secho, _ = mock_typer
    files = [
        {"name": "slide1.jpeg", "interrupted": True},
        {"name": "slide2.jpeg", "processed": True},
        {"name": "slide3.jpeg"},
    ]
    setup_test_files(files)

    all_paths = [TMP_DIRECTORY / f["name"] for f in files]

    result = filter_slide_paths(
        all_paths=all_paths,
        parent_dir=TMP_DIRECTORY,
        overwrite=False,
        overwrite_unfinished=True,
    )

    assert len(result) == 2
    assert {p.name for p in result} == {"slide1.jpeg", "slide3.jpeg"}
    mock_secho.assert_any_call(
        "Overwriting 1 unfinished slide outputs.", fg="yellow", bold=True
    )


def test_filter_slides_ignore_non_files(mock_typer):
    """Test that non-file paths are ignored."""
    setup_test_files([])

    all_paths = [TMP_DIRECTORY / "not_a_file"]  # Directory, not a file
    (TMP_DIRECTORY / "not_a_file").mkdir(parents=True, exist_ok=True)

    result = filter_slide_paths(
        all_paths=all_paths,
        parent_dir=TMP_DIRECTORY,
        overwrite=False,
        overwrite_unfinished=False,
    )
    # The function will exit because no slides are found to process
    assert len(result) == 0


@pytest.fixture
def mock_slide_reader(monkeypatch):
    """Mock the SlideReader class."""
    mock = MagicMock()
    mock.return_value.get_tissue_mask.return_value = (128, MagicMock())
    monkeypatch.setattr("histoslice.cli._app.SlideReader", mock)
    return mock


def test_cut_slide_success(mock_slide_reader):
    """Test a successful run of cut_slide."""
    path = Path("slide.svs")
    reader_kwargs = {"backend": "openslide"}
    tissue_kwargs = {"level": 0}
    tile_kwargs = {"width": 256, "height": 256}
    save_kwargs = {"parent_dir": Path("/tmp")}

    result_path, exception = cut_slide(
        path,
        reader_kwargs=reader_kwargs,
        max_dimension=512,
        tissue_kwargs=tissue_kwargs,
        tile_kwargs=tile_kwargs,
        save_kwargs=save_kwargs,
    )

    assert result_path == path
    assert exception is None
    mock_slide_reader.assert_called_once_with(path, **reader_kwargs)
    reader_instance = mock_slide_reader.return_value
    reader_instance.get_tissue_mask.assert_called_once_with(**tissue_kwargs)
    reader_instance.get_tile_coordinates.assert_called_once()
    reader_instance.save_regions.assert_called_once()


def test_cut_slide_exception(mock_slide_reader):
    """Test that cut_slide catches and returns exceptions."""
    mock_slide_reader.side_effect = ValueError("Test error")

    path = Path("slide.svs")

    result_path, exception = cut_slide(
        path,
        reader_kwargs={},
        max_dimension=512,
        tissue_kwargs={},
        tile_kwargs={},
        save_kwargs={},
    )

    assert result_path == path
    assert isinstance(exception, ValueError)
    assert str(exception) == "Test error"


def test_cut_slide_auto_level(mock_slide_reader):
    """Test automatic level detection in cut_slide."""
    path = Path("slide.svs")
    tissue_kwargs = {"level": None}  # Level is None to trigger auto-detection

    reader_instance = mock_slide_reader.return_value
    reader_instance.level_from_max_dimension.return_value = 5

    _, exception = cut_slide(
        path,
        reader_kwargs={},
        max_dimension=512,
        tissue_kwargs=tissue_kwargs,
        tile_kwargs={},
        save_kwargs={},
    )

    assert exception is None
    reader_instance.level_from_max_dimension.assert_called_once_with(512)
    # The modified tissue_kwargs is passed to get_tissue_mask
    assert reader_instance.get_tissue_mask.call_args[1]["level"] == 5
