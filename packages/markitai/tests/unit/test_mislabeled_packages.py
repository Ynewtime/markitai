"""A file named like a ZIP package that holds something else says so."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from markitai.converter.base import not_a_package
from markitai.converter.markitdown_ext import EpubConverter
from markitai.converter.opendocument import OdtConverter
from markitai.utils.errors import ConversionError

POSTSCRIPT = b"%!PS-Adobe-3.0\n72 720 moveto (Desk lamp manual) show\nshowpage\n"


def test_gzip_named_epub(tmp_path: Path):
    path = tmp_path / "lamp-manual.epub"
    path.write_bytes(gzip.compress(POSTSCRIPT))
    with pytest.raises(ConversionError) as caught:
        EpubConverter().convert(path)
    message = str(caught.value)
    assert "not an EPUB file" in message
    assert "gzip-compressed data" in message


def test_html_named_odt(tmp_path: Path):
    path = tmp_path / "notes.odt"
    path.write_text("<!DOCTYPE html><html><body>notes</body></html>", encoding="utf-8")
    with pytest.raises(ConversionError, match="HTML or XML text"):
        OdtConverter().convert(path)


@pytest.mark.parametrize(
    ("head", "name"),
    [
        (POSTSCRIPT, "a PostScript document"),
        (b"%PDF-1.7\n", "a PDF document"),
        (b"Rar!\x1a\x07\x01\x00", "a RAR archive"),
    ],
)
def test_recognized_contents(tmp_path: Path, head: bytes, name: str):
    path = tmp_path / "book.epub"
    path.write_bytes(head)
    reason = not_a_package(path, "EPUB")
    assert reason is not None and name in reason


def test_a_damaged_zip_is_not_explained_away(tmp_path: Path):
    path = tmp_path / "book.epub"
    path.write_bytes(b"PK\x03\x04truncated")
    assert not_a_package(path, "EPUB") is None
    with pytest.raises(ConversionError, match="Malformed EPUB"):
        EpubConverter().convert(path)
