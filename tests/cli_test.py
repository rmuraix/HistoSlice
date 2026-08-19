import polars as pl

from ._utils import (
    IMAGE_EXT,
    SLIDE_PATH_JPEG,
    TMP_DIRECTORY,
    clean_temporary_directory,
    create_tiles_with_metrics,
    make_bad_slide_dir,
)


def create_metadata(unfinished: bool = False) -> None:  # noqa
    meta_path = TMP_DIRECTORY / "slide" / "metadata.parquet"
    meta_path.parent.mkdir(parents=True)
    if not unfinished:
        meta_path.touch()


def test_run(script_runner) -> None:  # noqa
    # Use uv run histoslice to ensure proper environment
    clean_temporary_directory()
    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "slice",
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
            f"thumbnail.{IMAGE_EXT}",
            f"thumbnail_tiles.{IMAGE_EXT}",
            f"thumbnail_tissue.{IMAGE_EXT}",
            "tiles",
            "metadata.parquet",
        ]
    )
    clean_temporary_directory()


def test_slice_command_parallel(script_runner) -> None:  # noqa
    """`-j 2` dispatches through `ProcessPoolExecutor` in a real subprocess."""
    clean_temporary_directory()
    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "slice",
            "-i",
            str(SLIDE_PATH_JPEG),
            "-o",
            str(TMP_DIRECTORY),
            "-j",
            "2",
        ]
    )
    assert ret.success
    assert (TMP_DIRECTORY / "slide" / "metadata.parquet").exists()
    clean_temporary_directory()


def test_skip_processed(script_runner) -> None:  # noqa
    clean_temporary_directory()
    create_metadata(unfinished=False)
    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "slice",
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
    clean_temporary_directory()
    create_metadata(unfinished=False)
    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "slice",
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
            f"thumbnail.{IMAGE_EXT}",
            f"thumbnail_tiles.{IMAGE_EXT}",
            f"thumbnail_tissue.{IMAGE_EXT}",
            "tiles",
            "metadata.parquet",
        ]
    )
    clean_temporary_directory()


def test_unfinished(script_runner) -> None:  # noqa
    clean_temporary_directory()
    create_metadata(unfinished=True)
    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "slice",
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
            f"thumbnail.{IMAGE_EXT}",
            f"thumbnail_tiles.{IMAGE_EXT}",
            f"thumbnail_tissue.{IMAGE_EXT}",
            "tiles",
            "metadata.parquet",
        ]
    )
    clean_temporary_directory()


def test_run_with_error_multi_process(script_runner, monkeypatch) -> None:  # noqa
    def mock_slice_one(path, output_dir, kwargs):
        if "error_slide" in str(path):
            return path, ValueError("Processing error"), 0
        from histoslice.cli import slice_one as original_slice_one

        return original_slice_one(path, output_dir, kwargs)

    monkeypatch.setattr("histoslice.cli.slice_one", mock_slice_one)

    clean_temporary_directory()
    TMP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (TMP_DIRECTORY / "error_slide.jpeg").touch()

    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "slice",
            "-i",
            str(TMP_DIRECTORY / "*.jpeg"),
            "-o",
            str(TMP_DIRECTORY),
            "-j",
            "2",
        ]
    )

    assert ret.success
    assert "Could not process" in ret.stdout
    clean_temporary_directory()


def test_run_with_error_single_process(script_runner, monkeypatch) -> None:  # noqa
    def mock_slice_one(path, output_dir, kwargs):
        if "error_slide" in str(path):
            return path, ValueError("Processing error"), 0
        from histoslice.cli import slice_one as original_slice_one

        return original_slice_one(path, output_dir, kwargs)

    monkeypatch.setattr("histoslice.cli.slice_one", mock_slice_one)

    clean_temporary_directory()
    TMP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (TMP_DIRECTORY / "error_slide.jpeg").touch()

    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "slice",
            "-i",
            str(TMP_DIRECTORY / "*.jpeg"),
            "-o",
            str(TMP_DIRECTORY),
            "-j",
            "0",
        ]
    )

    assert ret.success
    assert "Could not process" in ret.stdout
    clean_temporary_directory()


def test_clean_command_move(script_runner) -> None:  # noqa
    """Test clean command creates metadata_clean.parquet with QC columns."""
    clean_temporary_directory()
    create_tiles_with_metrics()

    # Count initial tiles (should be unchanged after clean)
    tiles_dir = TMP_DIRECTORY / "slide" / "tiles"
    initial_tile_count = len(list(tiles_dir.glob(f"*.{IMAGE_EXT}")))
    assert initial_tile_count > 0

    # Run clean command with directory pattern
    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "slide"),
            "-j",
            "0",
        ]
    )

    assert ret.success

    # Check that metadata_clean.parquet was created
    clean_parquet = TMP_DIRECTORY / "slide" / "metadata_clean.parquet"
    assert clean_parquet.exists()

    # Verify it contains the QC columns
    df = pl.read_parquet(clean_parquet)
    assert "is_outlier" in df.columns
    assert "needs_review" in df.columns
    assert "qc_status" in df.columns
    assert df["qc_method"].unique().to_list() == ["technical_qc_v1"]
    assert (df["is_outlier"] == (df["qc_status"] == "fail")).all()
    assert (df["needs_review"] == (df["qc_status"] == "warn")).all()

    # Tile files should be untouched
    assert len(list(tiles_dir.glob(f"*.{IMAGE_EXT}"))) == initial_tile_count
    assert not (TMP_DIRECTORY / "slide" / "outliers").exists()

    clean_temporary_directory()


def test_clean_command_parallel(script_runner) -> None:  # noqa
    """`-j 2` dispatches through `ProcessPoolExecutor` in a real subprocess."""
    clean_temporary_directory()
    create_tiles_with_metrics()

    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "slide"),
            "-j",
            "2",
        ]
    )

    assert ret.success
    clean_parquet = TMP_DIRECTORY / "slide" / "metadata_clean.parquet"
    assert clean_parquet.exists()
    df = pl.read_parquet(clean_parquet)
    assert "is_outlier" in df.columns

    clean_temporary_directory()


def test_clean_command_delete(script_runner) -> None:  # noqa
    """Test that the --delete flag is no longer accepted (removed from CLI)."""
    clean_temporary_directory()
    create_tiles_with_metrics()

    # Run clean command with --delete (should fail – option no longer exists)
    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "slide"),
            "--delete",
            "-j",
            "0",
        ]
    )

    # --delete is no longer a valid option
    assert not ret.success
    assert "No such option" in ret.stderr or "no such option" in ret.stderr.lower()

    clean_temporary_directory()


def test_clean_command_no_metadata(script_runner) -> None:  # noqa
    """Test clean command with non-existent directory."""
    clean_temporary_directory()

    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "nonexistent"),
        ]
    )

    # Should fail because no slide directories found
    assert not ret.success
    clean_temporary_directory()


def test_clean_command_mode_option_removed(script_runner) -> None:  # noqa
    """Test that the --mode/--num-clusters flags are no longer accepted (removed
    together with k-means-based outlier detection)."""
    clean_temporary_directory()
    create_tiles_with_metrics()

    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "slide"),
            "--mode",
            "clustering",
        ]
    )

    assert not ret.success
    assert "No such option" in ret.stderr or "no such option" in ret.stderr.lower()

    clean_temporary_directory()


def test_clean_command_unsupported_format(script_runner) -> None:  # noqa
    """Test clean command with directory without metadata."""
    clean_temporary_directory()

    # Create a directory without metadata
    TMP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    test_dir = TMP_DIRECTORY / "slide"
    test_dir.mkdir(exist_ok=True)
    # Create a dummy file but no metadata
    (test_dir / "dummy.txt").write_text("dummy content")

    # Run clean command
    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "clean",
            "-i",
            str(test_dir),
        ]
    )

    # Should fail because no directories with metadata found
    assert not ret.success

    clean_temporary_directory()


def test_clean_command_missing_tile_files(script_runner) -> None:  # noqa
    """Test clean command succeeds even when some tile files are missing."""
    clean_temporary_directory()
    create_tiles_with_metrics()

    # Delete multiple tile files to simulate a partially missing dataset
    tiles_dir = TMP_DIRECTORY / "slide" / "tiles"
    tile_files = list(tiles_dir.glob(f"*.{IMAGE_EXT}"))
    if len(tile_files) > 10:
        for i in range(10):
            tile_files[i].unlink()

    # Run clean command – missing files do not affect the parquet output
    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "slide"),
            "-j",
            "0",
        ]
    )

    # Should succeed and create metadata_clean.parquet
    assert ret.success
    assert (TMP_DIRECTORY / "slide" / "metadata_clean.parquet").exists()

    clean_temporary_directory()


def test_clean_command_exception_handling(script_runner, monkeypatch) -> None:  # noqa
    """Test clean command exception handling."""
    clean_temporary_directory()
    create_tiles_with_metrics()

    # Make the metadata file unreadable to trigger an exception
    metadata_file = TMP_DIRECTORY / "slide" / "metadata.parquet"
    metadata_file.chmod(0o000)

    # Run clean command
    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "slide"),
            "-j",
            "0",
        ]
    )

    # Restore permissions
    metadata_file.chmod(0o644)

    # Should succeed but warn about the exception
    assert ret.success
    assert "Could not process" in ret.stdout

    clean_temporary_directory()


def test_clean_command_reports_per_slide_exception_parallel(script_runner) -> None:  # noqa
    """`-j 2` reports per-slide exceptions the same way as sequential mode."""
    clean_temporary_directory()
    create_tiles_with_metrics()
    make_bad_slide_dir("bad")

    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "*"),
            "-j",
            "2",
        ]
    )

    assert ret.success
    assert "Could not process" in ret.stdout

    clean_temporary_directory()


def test_clean_command_basic_usage(script_runner) -> None:  # noqa
    """`clean -i ...` with no other options is the whole basic usage surface."""
    clean_temporary_directory()
    create_tiles_with_metrics()

    ret = script_runner.run(
        [
            "uv",
            "run",
            "--no-sync",
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "slide"),
        ]
    )

    assert ret.success
    df = pl.read_parquet(TMP_DIRECTORY / "slide" / "metadata_clean.parquet")
    assert set(df["qc_status"].unique().to_list()) <= {"pass", "warn", "fail"}

    clean_temporary_directory()
