"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    """Return a temporary output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output


@pytest.fixture
def sample_markdown() -> str:
    """Return sample markdown content for testing."""
    return """# Test Document

This is a test document with some content.

## Section 1

Some text in section 1.

- Item 1
- Item 2
- Item 3

## Section 2

A table:

| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
"""


@pytest.fixture
def sample_config_dict() -> dict:
    """Return sample configuration dictionary."""
    return {
        "output": {
            "dir": "./custom_output",
            "on_conflict": "overwrite",
        },
        "llm": {
            "enabled": True,
            "concurrency": 5,
        },
        "image": {
            "compress": True,
            "quality": 90,
        },
    }
