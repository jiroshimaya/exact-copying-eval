"""Pytest configuration and fixtures for manual tests."""

import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset environment variables for each test."""
    # Remove any environment variables that might affect tests
    env_vars_to_remove = [
        "LOG_LEVEL",
        "TEST_LOG_LEVEL",
    ]
    for var in env_vars_to_remove:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def set_test_log_level(monkeypatch: pytest.MonkeyPatch):
    """Fixture to set log level during tests."""

    def _set_log_level(level: str) -> None:
        monkeypatch.setenv("TEST_LOG_LEVEL", level)
        # Also set the log level directly
        logging.getLogger().setLevel(getattr(logging, level.upper()))

    return _set_log_level


@pytest.fixture
def sample_questions() -> list[str]:
    """Sample questions for testing."""
    return [
        "日本の首都はどこですか？",
        "地球の周りを回る天体は何ですか？",
        "一番大きな惑星はどれですか？",
    ]


@pytest.fixture
def sample_contexts() -> list[str]:
    """Sample contexts for testing."""
    return [
        "日本は東アジアに位置する島国です。首都は東京で、人口は約1400万人です。東京は政治、経済、文化の中心地として機能しています。",
        "月は地球の唯一の天然衛星です。地球から約38万4400キロメートル離れており、約27.3日で地球の周りを一周します。潮の満ち引きにも影響を与えています。",
        "太陽系には8つの惑星があります。その中で最も大きいのは木星です。木星は地球の約11倍の直径を持ち、ガス惑星として知られています。",
    ]
