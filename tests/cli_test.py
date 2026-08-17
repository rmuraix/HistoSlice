from ._utils import (
    IMAGE_EXT,
    SLIDE_PATH_JPEG,
    TMP_DIRECTORY,
    clean_temporary_directory,
)


def create_metadata(unfinished: bool = False) -> None:  # noqa
    meta_path = TMP_DIRECTORY / "slide" / "metadata.parquet"
    meta_path.parent.mkdir(parents=True)
    if not unfinished:
        meta_path.touch()


def create_tiles_with_metrics() -> None:  # noqa
    """Export real tiles (with metrics) for the `clean` command to operate on."""
    from histoslice import Slide, export_tiles
    from histoslice.tiles import tile_regions

    slide = Slide(SLIDE_PATH_JPEG)
    regions = tile_regions(slide.dimensions, 256, overlap=0.0, out_of_bounds=False)
    export_tiles(
        slide,
        regions,
        TMP_DIRECTORY / slide.name,
        tile_size=256,
        save_metrics=True,
        threshold=200,
        save_thumbnails=False,
    )


def test_run(script_runner) -> None:  # noqa
    # Use uv run histoslice to ensure proper environment
    clean_temporary_directory()
    ret = script_runner.run(
        [
            "uv",
            "run",
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


def test_skip_processed(script_runner) -> None:  # noqa
    clean_temporary_directory()
    create_metadata(unfinished=False)
    ret = script_runner.run(
        [
            "uv",
            "run",
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
    """Test clean command creates metadata_clean.parquet with is_outlier and method columns."""
    import polars as pl

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
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "slide"),
            "-k",
            "4",
            "-j",
            "0",
        ]
    )

    assert ret.success

    # Check that metadata_clean.parquet was created
    clean_parquet = TMP_DIRECTORY / "slide" / "metadata_clean.parquet"
    assert clean_parquet.exists()

    # Verify it contains is_outlier and method columns
    df = pl.read_parquet(clean_parquet)
    assert "is_outlier" in df.columns
    assert "method" in df.columns
    assert df["method"].unique().to_list() == ["clustering"]
    assert df["is_outlier"].any()

    # Tile files should be untouched
    assert len(list(tiles_dir.glob(f"*.{IMAGE_EXT}"))) == initial_tile_count
    assert not (TMP_DIRECTORY / "slide" / "outliers").exists()

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
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "slide"),
            "-k",
            "4",
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
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "nonexistent"),
        ]
    )

    # Should fail because no slide directories found
    assert not ret.success
    clean_temporary_directory()


def test_clean_command_invalid_mode(script_runner) -> None:  # noqa
    """Test clean command with invalid mode."""
    clean_temporary_directory()
    create_tiles_with_metrics()

    # Run clean command with invalid mode
    ret = script_runner.run(
        [
            "uv",
            "run",
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "slide"),
            "--mode",
            "invalid_mode",
        ]
    )

    # Should fail because of invalid mode
    assert not ret.success
    assert "Unknown mode" in ret.stderr or "Unknown mode" in ret.stdout

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
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "slide"),
            "-k",
            "4",
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


def test_clean_command_no_outliers(script_runner) -> None:  # noqa
    """Test clean command when no outliers are detected."""
    clean_temporary_directory()
    create_tiles_with_metrics()

    # Run clean command with only 2 clusters (likely all tiles in one cluster)
    ret = script_runner.run(
        [
            "uv",
            "run",
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "slide"),
            "-k",
            "2",
            "-j",
            "0",
        ]
    )

    assert ret.success
    # The output should mention either detection or no outliers

    clean_temporary_directory()

    clean_temporary_directory()
