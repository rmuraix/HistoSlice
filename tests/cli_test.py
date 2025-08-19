import shutil
from pathlib import Path
import pytest

from ._utils import (
    SLIDE_PATH_JPEG,
    TMP_DIRECTORY,
    clean_temporary_directory,
)


def create_metadata(unfinished: bool = False) -> None:  # noqa
    meta_path = TMP_DIRECTORY / "slide" / "metadata.parquet"
    meta_path.parent.mkdir(parents=True)
    if not unfinished:
        meta_path.touch()


def test_run(script_runner) -> None:  # noqa
    # Use project console script path to avoid relying on poetry in PATH
    cli = shutil.which("HistoSlice") or str(
        (TMP_DIRECTORY.parent.parent / ".venv" / "bin" / "HistoSlice").resolve()
    )
    if not cli or not Path(cli).exists():
        return pytest.skip("HistoSlice CLI not available in test environment")
    clean_temporary_directory()
    ret = script_runner.run(
        [
            cli,
            "-i",
            str(SLIDE_PATH_JPEG),
            "-o",
            str(TMP_DIRECTORY),
            "--thumbnails",
            "-j",
            "0",
        ]
    )
    assert ret.success
    assert sorted([x.name for x in (TMP_DIRECTORY / "slide").iterdir()]) == sorted(
        [
            "properties.json",
            "thumbnail.jpeg",
            "thumbnail_tiles.jpeg",
            "thumbnail_tissue.jpeg",
            "tiles",
            "metadata.parquet",
        ]
    )
    clean_temporary_directory()


def test_skip_processed(script_runner) -> None:  # noqa
    cli = shutil.which("HistoSlice") or str(
        (TMP_DIRECTORY.parent.parent / ".venv" / "bin" / "HistoSlice").resolve()
    )
    if not cli or not Path(cli).exists():
        return pytest.skip("HistoSlice CLI not available in test environment")
    clean_temporary_directory()
    create_metadata(unfinished=False)
    ret = script_runner.run(
        [
            cli,
            "-i",
            str(SLIDE_PATH_JPEG),
            "-o",
            str(TMP_DIRECTORY),
            "--thumbnails",
            "-j",
            "0",
        ]
    )
    # Expect failure exit due to no work
    assert not ret.success
    clean_temporary_directory()


def test_overwrite(script_runner) -> None:  # noqa
    cli = shutil.which("HistoSlice") or str(
        (TMP_DIRECTORY.parent.parent / ".venv" / "bin" / "HistoSlice").resolve()
    )
    if not cli or not Path(cli).exists():
        return pytest.skip("HistoSlice CLI not available in test environment")
    clean_temporary_directory()
    create_metadata(unfinished=False)
    ret = script_runner.run(
        [
            cli,
            "-i",
            str(SLIDE_PATH_JPEG),
            "-o",
            str(TMP_DIRECTORY),
            "--thumbnails",
            "-z",
            "-j",
            "0",
        ]
    )
    assert ret.success
    assert sorted([x.name for x in (TMP_DIRECTORY / "slide").iterdir()]) == sorted(
        [
            "properties.json",
            "thumbnail.jpeg",
            "thumbnail_tiles.jpeg",
            "thumbnail_tissue.jpeg",
            "tiles",
            "metadata.parquet",
        ]
    )
    clean_temporary_directory()


def test_unfinished(script_runner) -> None:  # noqa
    cli = shutil.which("HistoSlice") or str(
        (TMP_DIRECTORY.parent.parent / ".venv" / "bin" / "HistoSlice").resolve()
    )
    if not cli or not Path(cli).exists():
        return pytest.skip("HistoSlice CLI not available in test environment")
    clean_temporary_directory()
    create_metadata(unfinished=True)
    ret = script_runner.run(
        [
            cli,
            "-i",
            str(SLIDE_PATH_JPEG),
            "-o",
            str(TMP_DIRECTORY),
            "--thumbnails",
            "-u",
            "-j",
            "0",
        ]
    )
    assert ret.success
    assert sorted([x.name for x in (TMP_DIRECTORY / "slide").iterdir()]) == sorted(
        [
            "properties.json",
            "thumbnail.jpeg",
            "thumbnail_tiles.jpeg",
            "thumbnail_tissue.jpeg",
            "tiles",
            "metadata.parquet",
        ]
    )
    clean_temporary_directory()
