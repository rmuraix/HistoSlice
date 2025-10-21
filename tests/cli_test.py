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
            "properties.json",
            "thumbnail.jpeg",
            "thumbnail_tiles.jpeg",
            "thumbnail_tissue.jpeg",
            "tiles",
            "metadata.parquet",
        ]
    )
    clean_temporary_directory()


def test_run_with_error_multi_process(script_runner, monkeypatch) -> None:  # noqa
    def mock_cut_slide(path, **kwargs):
        if "error_slide" in str(path):
            return path, ValueError("Processing error")
        from histoslice.cli._app import cut_slide as original_cut_slide

        return original_cut_slide(path, **kwargs)

    monkeypatch.setattr("histoslice.cli._app.cut_slide", mock_cut_slide)

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
    def mock_cut_slide(path, **kwargs):
        if "error_slide" in str(path):
            return path, ValueError("Processing error")
        from histoslice.cli._app import cut_slide as original_cut_slide

        return original_cut_slide(path, **kwargs)

    monkeypatch.setattr("histoslice.cli._app.cut_slide", mock_cut_slide)

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
    """Test clean command with move (default) behavior."""
    from histoslice import SlideReader

    clean_temporary_directory()
    # First, create tiles with metrics
    reader = SlideReader(SLIDE_PATH_JPEG)
    reader.save_regions(
        TMP_DIRECTORY,
        reader.get_tile_coordinates(None, 256, overlap=0.0),
        save_metrics=True,
        threshold=200,
    )

    # Count initial tiles
    tiles_dir = TMP_DIRECTORY / "slide" / "tiles"
    initial_tile_count = len(list(tiles_dir.glob("*.jpeg")))
    assert initial_tile_count > 0

    # Run clean command (move mode)
    ret = script_runner.run(
        [
            "uv",
            "run",
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "slide" / "metadata.parquet"),
            "-k",
            "4",
        ]
    )

    assert ret.success

    # Check that outliers directory was created
    outliers_dir = TMP_DIRECTORY / "slide" / "outliers"
    assert outliers_dir.exists()

    # Check that some files were moved
    moved_count = len(list(outliers_dir.glob("*.jpeg")))
    remaining_count = len(list(tiles_dir.glob("*.jpeg")))
    assert moved_count > 0
    assert remaining_count + moved_count == initial_tile_count

    clean_temporary_directory()


def test_clean_command_delete(script_runner) -> None:  # noqa
    """Test clean command with delete behavior."""
    from histoslice import SlideReader

    clean_temporary_directory()
    # First, create tiles with metrics
    reader = SlideReader(SLIDE_PATH_JPEG)
    reader.save_regions(
        TMP_DIRECTORY,
        reader.get_tile_coordinates(None, 256, overlap=0.0),
        save_metrics=True,
        threshold=200,
    )

    # Count initial tiles
    tiles_dir = TMP_DIRECTORY / "slide" / "tiles"
    initial_tile_count = len(list(tiles_dir.glob("*.jpeg")))
    assert initial_tile_count > 0

    # Run clean command (delete mode)
    ret = script_runner.run(
        [
            "uv",
            "run",
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "slide" / "metadata.parquet"),
            "-k",
            "4",
            "--delete",
        ]
    )

    assert ret.success

    # Check that some files were deleted
    remaining_count = len(list(tiles_dir.glob("*.jpeg")))
    assert remaining_count < initial_tile_count

    # Check that outliers directory was NOT created
    outliers_dir = TMP_DIRECTORY / "slide" / "outliers"
    assert not outliers_dir.exists()

    clean_temporary_directory()


def test_clean_command_no_metadata(script_runner) -> None:  # noqa
    """Test clean command with non-existent metadata file."""
    clean_temporary_directory()

    ret = script_runner.run(
        [
            "uv",
            "run",
            "histoslice",
            "clean",
            "-i",
            str(TMP_DIRECTORY / "nonexistent" / "metadata.parquet"),
        ]
    )

    # Should fail because no metadata files found
    assert not ret.success
    clean_temporary_directory()
