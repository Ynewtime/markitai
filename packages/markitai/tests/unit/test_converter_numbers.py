"""Tests for the local Apple Numbers converter."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from markitai.converter.base import FileFormat, get_converter
from markitai.utils.errors import ConversionError


def test_numbers_fails_instead_of_converting_package_metadata(tmp_path: Path) -> None:
    """No local reader understands Numbers tables: markitdown's generic zip
    handler returned the package's metadata members as a "successful"
    document. The item must fail with a reason instead."""
    path = tmp_path / "budget.numbers"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Index/Document.iwa", b"\x00\x8a\x02\x00\x08\x01Rent 1200")
        archive.writestr("Metadata/DocumentIdentifier", "7C1F6D2A-1234-4F0B-9E7A")

    converter = get_converter(path)
    assert converter is not None
    assert FileFormat.NUMBERS in converter.supported_formats

    with pytest.raises(ConversionError, match=r"-b cloudflare"):
        converter.convert(path)


def test_numbers_stays_a_known_format_for_cloudflare() -> None:
    from markitai.converter.base import detect_format
    from markitai.converter.cloudflare import CF_SUPPORTED_FORMATS

    assert detect_format("budget.numbers") == FileFormat.NUMBERS
    assert FileFormat.NUMBERS in CF_SUPPORTED_FORMATS
