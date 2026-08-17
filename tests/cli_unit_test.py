"""In-process CLI tests via Typer's `CliRunner`.

`tests/cli_test.py` exercises the packaged `histoslice` console script as a
real subprocess (`uv run histoslice ...`) - good for verifying the installed
entry point, but coverage tooling can't see inside that separate process.
These tests invoke the same `app` in-process to cover argument parsing,
orchestration, and error handling directly.

Multi-process (`-j` > 0) dispatch is intentionally NOT exercised here: it
spawns real worker processes via `ProcessPoolExecutor`, which need to
reconstruct pytest's own `__main__` entry point when run in-process. That is
unreliable across Python versions/environments, so parallel dispatch is
covered instead via real subprocess invocations in `tests/cli_test.py`.
"""

import sys
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from histoslice.cli import (
    app,
    filter_slide_paths,
    main,
    process_slide_outliers,
    slice_one,
)
from histoslice.slide import Slide

from ._utils import (
    SLIDE_PATH_JPEG,
    TMP_DIRECTORY,
    clean_temporary_directory,
    create_tiles_with_metrics,
    make_bad_slide_dir,
)

runner = CliRunner()


def test_slice_command_sequential() -> None:
    clean_temporary_directory()
    result = runner.invoke(
        app,
        ["slice", "-i", str(SLIDE_PATH_JPEG), "-o", str(TMP_DIRECTORY), "-j", "0"],
    )
    assert result.exit_code == 0
    assert (TMP_DIRECTORY / "slide" / "metadata.parquet").exists()
    clean_temporary_directory()


def test_slice_command_reports_per_slide_exception() -> None:
    """A slide that fails to open is reported, without failing the whole run."""
    clean_temporary_directory()
    TMP_DIRECTORY.mkdir(parents=True)
    (TMP_DIRECTORY / "broken.jpeg").touch()  # not a real image -> Slide() raises
    result = runner.invoke(
        app,
        [
            "slice",
            "-i",
            str(TMP_DIRECTORY / "*.jpeg"),
            "-o",
            str(TMP_DIRECTORY),
            "-j",
            "0",
        ],
    )
    assert result.exit_code == 0
    assert "Could not process" in result.output
    clean_temporary_directory()


def test_slice_command_reports_per_tile_failures(monkeypatch) -> None:
    """A slide that completes with some failed tiles is reported by count."""
    clean_temporary_directory()
    original_read_tile = Slide.read_tile
    calls = {"n": 0}

    def flaky_read_tile(self, region, size):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return original_read_tile(self, region, size)

    monkeypatch.setattr(Slide, "read_tile", flaky_read_tile)
    result = runner.invoke(
        app, ["slice", "-i", str(SLIDE_PATH_JPEG), "-o", str(TMP_DIRECTORY), "-j", "0"]
    )
    assert result.exit_code == 0
    assert "failed tile(s)" in result.output
    clean_temporary_directory()


def test_slice_command_no_files_found() -> None:
    clean_temporary_directory()
    result = runner.invoke(
        app, ["slice", "-i", str(TMP_DIRECTORY / "*.jpeg"), "-o", str(TMP_DIRECTORY)]
    )
    assert result.exit_code == 1
    assert "Found no files matching pattern" in result.output


def test_slice_one_success() -> None:
    clean_temporary_directory()
    path, exception, num_failed = slice_one(
        SLIDE_PATH_JPEG, TMP_DIRECTORY, {"save_thumbnails": False, "verbose": False}
    )
    assert path == SLIDE_PATH_JPEG
    assert exception is None
    assert num_failed == 0
    clean_temporary_directory()


def test_slice_one_exception() -> None:
    path, exception, num_failed = slice_one(
        Path("/does/not/exist.jpeg"), TMP_DIRECTORY, {}
    )
    assert isinstance(exception, FileNotFoundError)
    assert num_failed == 0


def test_filter_slide_paths_default_skips_processed_and_interrupted() -> None:
    clean_temporary_directory()
    TMP_DIRECTORY.mkdir(parents=True)
    processed, interrupted, fresh = (
        TMP_DIRECTORY / "processed.jpeg",
        TMP_DIRECTORY / "interrupted.jpeg",
        TMP_DIRECTORY / "fresh.jpeg",
    )
    for path in (processed, interrupted, fresh):
        path.touch()
    (TMP_DIRECTORY / "processed").mkdir()
    (TMP_DIRECTORY / "processed" / "metadata.parquet").touch()
    (TMP_DIRECTORY / "interrupted").mkdir()

    paths = filter_slide_paths(
        all_paths=[processed, interrupted, fresh],
        output_dir=TMP_DIRECTORY,
        overwrite=False,
        overwrite_unfinished=False,
    )
    assert paths == [fresh]

    paths = filter_slide_paths(
        all_paths=[processed, interrupted, fresh],
        output_dir=TMP_DIRECTORY,
        overwrite=False,
        overwrite_unfinished=True,
    )
    assert sorted(paths) == sorted([fresh, interrupted])

    paths = filter_slide_paths(
        all_paths=[processed, interrupted, fresh],
        output_dir=TMP_DIRECTORY,
        overwrite=True,
        overwrite_unfinished=False,
    )
    assert sorted(paths) == sorted([fresh, processed, interrupted])
    clean_temporary_directory()


def test_filter_slide_paths_nothing_to_overwrite_skips_warnings() -> None:
    """overwrite/overwrite_unfinished with no processed or interrupted slides
    take the "nothing to warn about" branch."""
    clean_temporary_directory()
    TMP_DIRECTORY.mkdir(parents=True)
    fresh = TMP_DIRECTORY / "fresh.jpeg"
    fresh.touch()

    for overwrite, overwrite_unfinished in [(True, False), (False, True)]:
        paths = filter_slide_paths(
            all_paths=[fresh],
            output_dir=TMP_DIRECTORY,
            overwrite=overwrite,
            overwrite_unfinished=overwrite_unfinished,
        )
        assert paths == [fresh]
    clean_temporary_directory()


def test_filter_slide_paths_none_found_exits() -> None:
    with pytest.raises(SystemExit):
        filter_slide_paths(
            all_paths=[],
            output_dir=TMP_DIRECTORY,
            overwrite=False,
            overwrite_unfinished=False,
        )


def test_clean_command_unknown_mode() -> None:
    result = runner.invoke(app, ["clean", "-i", str(TMP_DIRECTORY), "--mode", "bogus"])
    assert result.exit_code == 1
    assert "Unknown mode" in result.output


def test_clean_command_no_slide_dirs() -> None:
    clean_temporary_directory()
    result = runner.invoke(app, ["clean", "-i", str(TMP_DIRECTORY / "nothing")])
    assert result.exit_code == 1


def test_clean_command_sequential() -> None:
    clean_temporary_directory()
    slide_dir = create_tiles_with_metrics()

    result = runner.invoke(app, ["clean", "-i", str(slide_dir), "-k", "2", "-j", "0"])
    assert result.exit_code == 0
    assert (slide_dir / "metadata_clean.parquet").exists()
    clean_temporary_directory()


def test_clean_command_reports_per_slide_exception_sequential() -> None:
    clean_temporary_directory()
    create_tiles_with_metrics()
    make_bad_slide_dir("bad")
    result = runner.invoke(app, ["clean", "-i", str(TMP_DIRECTORY / "*"), "-j", "0"])
    assert result.exit_code == 0
    assert "Could not process" in result.output
    clean_temporary_directory()


def test_process_slide_outliers_success() -> None:
    clean_temporary_directory()
    slide_dir = create_tiles_with_metrics()
    result_dir, exception = process_slide_outliers(
        slide_dir, mode="clustering", num_clusters=2
    )
    assert result_dir == slide_dir
    assert exception is None
    df = pl.read_parquet(slide_dir / "metadata_clean.parquet")
    assert "is_outlier" in df.columns
    clean_temporary_directory()


def test_process_slide_outliers_exception() -> None:
    clean_temporary_directory()
    TMP_DIRECTORY.mkdir(parents=True)
    slide_dir, exception = process_slide_outliers(
        TMP_DIRECTORY, mode="clustering", num_clusters=2
    )
    assert isinstance(exception, Exception)
    clean_temporary_directory()


def test_main_entrypoint(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["histoslice", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
