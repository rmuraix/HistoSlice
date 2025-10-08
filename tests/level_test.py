"""Tests for histoslice.functional._level module."""

import pytest

from histoslice.functional._level import format_level


def test_format_level_positive_valid() -> None:
    """Test format_level with valid positive level."""
    available = [0, 1, 2, 3]
    assert format_level(0, available) == 0
    assert format_level(1, available) == 1
    assert format_level(2, available) == 2
    assert format_level(3, available) == 3


def test_format_level_negative_valid() -> None:
    """Test format_level with valid negative level."""
    available = [0, 1, 2, 3]
    assert format_level(-1, available) == 3
    assert format_level(-2, available) == 2
    assert format_level(-3, available) == 1
    assert format_level(-4, available) == 0


def test_format_level_positive_invalid() -> None:
    """Test format_level raises ValueError with invalid positive level."""
    available = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="Level .* could not be found"):
        format_level(5, available)


def test_format_level_negative_invalid() -> None:
    """Test format_level raises ValueError with invalid negative level."""
    available = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="Level .* could not be found"):
        format_level(-5, available)

    with pytest.raises(ValueError, match="Level .* could not be found"):
        format_level(-10, available)


def test_format_level_empty_available() -> None:
    """Test format_level with empty available list."""
    available = []
    with pytest.raises(ValueError, match="Level .* could not be found"):
        format_level(0, available)

    with pytest.raises(ValueError, match="Level .* could not be found"):
        format_level(-1, available)
