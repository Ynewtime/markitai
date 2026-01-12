"""Pytest configuration and fixtures."""

import shutil
import tempfile
from pathlib import Path

import pytest

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def cleanup_test_outputs():
    """Auto-cleanup output directories in test fixtures after each test."""
    yield

    # Clean up any output directories created in fixtures folder
    fixtures_output = PROJECT_ROOT / "tests" / "fixtures" / "documents" / "output"
    if fixtures_output.exists():
        shutil.rmtree(fixtures_output, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_all_tests():
    """Clean up after all tests complete."""
    yield

    # Final cleanup

    # Remove output in fixtures if exists
    fixtures_output = PROJECT_ROOT / "tests" / "fixtures" / "documents" / "output"
    if fixtures_output.exists():
        shutil.rmtree(fixtures_output, ignore_errors=True)

    # Remove any .markit-state.json files in project root
    state_file = PROJECT_ROOT / ".markit-state.json"
    if state_file.exists():
        state_file.unlink(missing_ok=True)

    # Clean up converted legacy files in fixtures/documents directory
    # (LibreOffice creates .docx/.pptx/.xlsx from .doc/.ppt/.xls)
    documents_dir = PROJECT_ROOT / "tests" / "fixtures" / "documents"
    if documents_dir.exists():
        for f in documents_dir.iterdir():
            # Check for pairs: if we have both .doc and .docx with same stem,
            # the .docx might be a converted file
            if f.suffix.lower() in {".docx", ".pptx", ".xlsx"}:
                # Only delete if it seems to be a converted file
                # (we can't be 100% sure, so be conservative)
                pass  # Skip auto-deletion to avoid deleting user files


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """Create a sample text file."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("# Sample Document\n\nThis is a test document.", encoding="utf-8")
    return file_path


@pytest.fixture
def sample_markdown_file(temp_dir: Path) -> Path:
    """Create a sample markdown file."""
    file_path = temp_dir / "sample.md"
    content = """# Test Document

## Section 1

This is the first section with some text.

## Section 2

This section has a list:
- Item 1
- Item 2
- Item 3

## Section 3

This section has a table:

| Column A | Column B |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |
"""
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def output_dir(temp_dir: Path) -> Path:
    """Create an output directory."""
    output = temp_dir / "output"
    output.mkdir()
    return output


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def documents_dir(fixtures_dir: Path) -> Path:
    """Return the path to document fixtures."""
    return fixtures_dir / "documents"


@pytest.fixture
def images_dir(fixtures_dir: Path) -> Path:
    """Return the path to image fixtures."""
    return fixtures_dir / "images"
