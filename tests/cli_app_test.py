from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from histoslice.cli._app import cut_slide, filter_slide_paths, process_slide_outliers
from tests._utils import TMP_DIRECTORY, clean_temporary_directory

_OUTLIER_DETECTOR_PATH = "histoslice.utils.OutlierDetector"


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
    mock.return_value.save_regions.return_value = (MagicMock(), [])
    monkeypatch.setattr("histoslice.cli._app.SlideReader", mock)
    return mock


def test_cut_slide_success(mock_slide_reader):
    """Test a successful run of cut_slide."""
    path = Path("slide.svs")
    reader_kwargs = {"backend": "openslide"}
    tissue_kwargs = {"level": 0}
    tile_kwargs = {"width": 256, "height": 256}
    save_kwargs = {"parent_dir": Path("/tmp")}

    result_path, exception, failures = cut_slide(
        path,
        reader_kwargs=reader_kwargs,
        max_dimension=512,
        tissue_kwargs=tissue_kwargs,
        tile_kwargs=tile_kwargs,
        save_kwargs=save_kwargs,
    )

    assert result_path == path
    assert exception is None
    assert failures == 0
    mock_slide_reader.assert_called_once_with(path, **reader_kwargs)
    reader_instance = mock_slide_reader.return_value
    reader_instance.get_tissue_mask.assert_called_once_with(**tissue_kwargs)
    reader_instance.get_tile_coordinates.assert_called_once()
    reader_instance.save_regions.assert_called_once()


def test_cut_slide_exception(mock_slide_reader):
    """Test that cut_slide catches and returns exceptions."""
    mock_slide_reader.side_effect = ValueError("Test error")

    path = Path("slide.svs")

    result_path, exception, failures = cut_slide(
        path,
        reader_kwargs={},
        max_dimension=512,
        tissue_kwargs={},
        tile_kwargs={},
        save_kwargs={},
    )

    assert result_path == path
    assert isinstance(exception, ValueError)
    assert failures == 0
    assert str(exception) == "Test error"


def test_cut_slide_auto_level(mock_slide_reader):
    """Test automatic level detection in cut_slide."""
    path = Path("slide.svs")
    tissue_kwargs = {"level": None}  # Level is None to trigger auto-detection

    reader_instance = mock_slide_reader.return_value
    reader_instance.level_from_max_dimension.return_value = 5

    _, exception, failures = cut_slide(
        path,
        reader_kwargs={},
        max_dimension=512,
        tissue_kwargs=tissue_kwargs,
        tile_kwargs={},
        save_kwargs={},
    )

    assert exception is None
    assert failures == 0
    reader_instance.level_from_max_dimension.assert_called_once_with(512)
    # The modified tissue_kwargs is passed to get_tissue_mask
    assert reader_instance.get_tissue_mask.call_args[1]["level"] == 5


def test_warning_function(monkeypatch):
    """Test the warning() function."""
    from histoslice.cli._app import warning

    mock_secho = MagicMock()
    monkeypatch.setattr("typer.secho", mock_secho)

    warning("Test warning message")

    mock_secho.assert_called_once()
    call_args = mock_secho.call_args
    assert call_args[0][0] == "Test warning message"
    assert call_args[1]["fg"] == "yellow"
    assert call_args[1]["bold"] is True


def test_info_function(monkeypatch):
    """Test the info() function."""
    from histoslice.cli._app import info

    mock_secho = MagicMock()
    monkeypatch.setattr("typer.secho", mock_secho)

    info("Test info message")

    mock_secho.assert_called_once()
    call_args = mock_secho.call_args
    assert call_args[0][0] == "Test info message"
    assert call_args[1]["fg"] == "cyan"
    assert call_args[1]["bold"] is True


def test_error_function(monkeypatch):
    """Test the error() function."""
    from histoslice.cli._app import error

    mock_secho = MagicMock()
    mock_exit = MagicMock()
    monkeypatch.setattr("typer.secho", mock_secho)
    monkeypatch.setattr("sys.exit", mock_exit)

    error("Test error message")

    mock_secho.assert_called_once()
    call_args = mock_secho.call_args
    assert call_args[0][0] == "Test error message"
    assert call_args[1]["fg"] == "red"
    assert call_args[1]["bold"] is True
    assert call_args[1]["err"] is True
    mock_exit.assert_called_once_with(1)


def test_error_function_custom_code(monkeypatch):
    """Test the error() function with custom exit code."""
    from histoslice.cli._app import error

    mock_secho = MagicMock()
    mock_exit = MagicMock()
    monkeypatch.setattr("typer.secho", mock_secho)
    monkeypatch.setattr("sys.exit", mock_exit)

    error("Test error message", exit_integer=42)

    mock_exit.assert_called_once_with(42)


def test_cut_slide_with_failures(mock_slide_reader):
    """Test that cut_slide returns the correct failure count."""
    path = Path("slide.svs")
    mock_slide_reader.return_value.save_regions.return_value = (
        MagicMock(),
        ["failure1", "failure2"],
    )

    result_path, exception, failures = cut_slide(
        path,
        reader_kwargs={},
        max_dimension=512,
        tissue_kwargs={"level": 0},
        tile_kwargs={},
        save_kwargs={},
    )

    assert result_path == path
    assert exception is None
    assert failures == 2


def _make_outlier_detector_mock(monkeypatch, clusters, paths):
    """Helper that patches OutlierDetector and returns the mock instance."""
    mock_instance = MagicMock()
    mock_instance.cluster_kmeans.return_value = np.array(clusters)
    mock_instance.paths = np.array(paths)

    mock_class = MagicMock()
    mock_class.from_parquet.return_value = mock_instance

    monkeypatch.setattr(_OUTLIER_DETECTOR_PATH, mock_class)
    return mock_class, mock_instance


def test_process_slide_outliers_no_metadata(tmp_path):
    """Test process_slide_outliers when no metadata file is present."""
    slide_dir = tmp_path / "slide"
    slide_dir.mkdir()

    result_dir, exception = process_slide_outliers(
        slide_dir, mode="clustering", num_clusters=3, delete=False
    )

    assert result_dir == slide_dir
    assert isinstance(exception, ValueError)
    assert "No metadata file found" in str(exception)


def test_process_slide_outliers_no_outliers(tmp_path, monkeypatch):
    """Test process_slide_outliers when cluster 0 contains no tiles."""
    slide_dir = tmp_path / "slide"
    slide_dir.mkdir()
    (slide_dir / "metadata.parquet").touch()

    # Cluster labels: none are 0, so outlier_mask is all False
    _make_outlier_detector_mock(monkeypatch, clusters=[1, 1, 2, 2], paths=[])

    result_dir, exception = process_slide_outliers(
        slide_dir, mode="clustering", num_clusters=3, delete=False
    )

    assert result_dir == slide_dir
    assert exception is None


def test_process_slide_outliers_move(tmp_path, monkeypatch):
    """Test process_slide_outliers moves outlier tiles to an 'outliers' subdirectory."""
    slide_dir = tmp_path / "slide"
    slide_dir.mkdir()
    (slide_dir / "metadata.parquet").touch()

    # Create two tile files that will be identified as outliers (cluster 0)
    tile1 = slide_dir / "tile_0001.jpeg"
    tile2 = slide_dir / "tile_0002.jpeg"
    tile1.touch()
    tile2.touch()

    _make_outlier_detector_mock(
        monkeypatch,
        clusters=[0, 0, 1, 1],
        paths=[str(tile1), str(tile2), "tile_0003.jpeg", "tile_0004.jpeg"],
    )

    result_dir, exception = process_slide_outliers(
        slide_dir, mode="clustering", num_clusters=2, delete=False
    )

    assert result_dir == slide_dir
    assert exception is None
    outliers_dir = slide_dir / "outliers"
    assert outliers_dir.exists()
    assert (outliers_dir / tile1.name).exists()
    assert (outliers_dir / tile2.name).exists()
    assert not tile1.exists()
    assert not tile2.exists()


def test_process_slide_outliers_delete(tmp_path, monkeypatch):
    """Test process_slide_outliers deletes outlier tiles when delete=True."""
    slide_dir = tmp_path / "slide"
    slide_dir.mkdir()
    (slide_dir / "metadata.parquet").touch()

    tile1 = slide_dir / "tile_0001.jpeg"
    tile1.touch()

    _make_outlier_detector_mock(
        monkeypatch,
        clusters=[0, 1],
        paths=[str(tile1), "tile_0002.jpeg"],
    )

    result_dir, exception = process_slide_outliers(
        slide_dir, mode="clustering", num_clusters=2, delete=True
    )

    assert result_dir == slide_dir
    assert exception is None
    assert not tile1.exists()
    assert not (slide_dir / "outliers").exists()


def test_process_slide_outliers_missing_tile_skipped(tmp_path, monkeypatch):
    """Test that process_slide_outliers skips tile paths that no longer exist."""
    slide_dir = tmp_path / "slide"
    slide_dir.mkdir()
    (slide_dir / "metadata.parquet").touch()

    nonexistent = str(slide_dir / "missing_tile.jpeg")
    _make_outlier_detector_mock(
        monkeypatch,
        clusters=[0],
        paths=[nonexistent],
    )

    result_dir, exception = process_slide_outliers(
        slide_dir, mode="clustering", num_clusters=2, delete=False
    )

    assert result_dir == slide_dir
    assert exception is None
    # No outliers dir created because the tile file didn't exist
    assert not (slide_dir / "outliers").exists()


def test_process_slide_outliers_exception(tmp_path, monkeypatch):
    """Test process_slide_outliers catches and returns unexpected exceptions."""
    slide_dir = tmp_path / "slide"
    slide_dir.mkdir()
    (slide_dir / "metadata.parquet").touch()

    mock_class = MagicMock()
    mock_class.from_parquet.side_effect = RuntimeError("disk read error")
    monkeypatch.setattr(_OUTLIER_DETECTOR_PATH, mock_class)

    result_dir, exception = process_slide_outliers(
        slide_dir, mode="clustering", num_clusters=3, delete=False
    )

    assert result_dir == slide_dir
    assert isinstance(exception, RuntimeError)
    assert "disk read error" in str(exception)


def test_filter_slides_overwrite_with_no_prior_output(mock_typer):
    """Test overwrite=True when there are no processed or interrupted slides."""
    mock_secho, _ = mock_typer
    files = [{"name": "slide1.jpeg"}]
    setup_test_files(files)

    all_paths = [TMP_DIRECTORY / f["name"] for f in files]

    result = filter_slide_paths(
        all_paths=all_paths,
        parent_dir=TMP_DIRECTORY,
        overwrite=True,
        overwrite_unfinished=False,
    )

    assert len(result) == 1
    # No "Overwriting" warning when there is nothing previously processed
    warning_calls = [
        c
        for c in mock_secho.call_args_list
        if "Overwriting" in (c.args[0] if c.args else "")
    ]
    assert len(warning_calls) == 0
