"""An e-book's own pictures become the item's assets, like a DOCX's.

markitdown keeps each chapter's reference (``../Images/fig.png``, relative to
a chapter file inside the archive), which resolves nowhere beside the
output, so the serve preview showed a placeholder for every picture.
"""

from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path

import pytest

from markitai.config import MarkitaiConfig
from markitai.converter import markitdown_ext
from markitai.converter.markitdown_ext import EpubConverter


def _png(color: tuple[int, int, int] = (37, 99, 235)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (400, 300), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _epub(path: Path, body: str, files: dict[str, bytes]) -> Path:
    container = (
        '<?xml version="1.0"?><container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
        '<rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )
    opf = (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" '
        'version="3.0" unique-identifier="id"><metadata '
        'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Charts</dc:title>'
        '<dc:identifier id="id">charts</dc:identifier></metadata><manifest>'
        '<item id="ch1" href="Text/ch1.xhtml" media-type="application/xhtml+xml"/>'
        '</manifest><spine><itemref idref="ch1"/></spine></package>'
    )
    chapter = (
        '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><head>'
        f"<title>Chapter</title></head><body>{body}</body></html>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/Text/ch1.xhtml", chapter)
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def test_a_chapter_picture_becomes_a_data_uri(tmp_path: Path) -> None:
    picture = _png()
    book = _epub(
        tmp_path / "charts.epub",
        '<h1>Bars</h1><p><img src="../Images/fig%201.png" alt="Bar chart"/></p>',
        {"OEBPS/Images/fig 1.png": picture},
    )
    markdown = EpubConverter().convert(book).markdown
    encoded = base64.b64encode(picture).decode("ascii")
    assert f"![Bar chart](data:image/png;base64,{encoded})" in markdown
    assert "../Images" not in markdown


@pytest.mark.parametrize(
    ("src", "files"),
    [
        # two pictures of that name: which one is unknown
        ("fig.png", {"OEBPS/a/fig.png": b"1", "OEBPS/b/fig.png": b"2"}),
        # not in the archive
        ("../Images/missing.png", {"OEBPS/Images/fig.png": b"1"}),
        # a remote picture is never fetched
        ("https://example.com/fig.png", {"OEBPS/Images/fig.png": b"1"}),
        # SVG stays a reference
        ("../Images/fig.svg", {"OEBPS/Images/fig.svg": b"<svg/>"}),
    ],
)
def test_other_references_are_left_as_they_are(
    tmp_path: Path, src: str, files: dict[str, bytes]
) -> None:
    book = _epub(tmp_path / "book.epub", f'<p><img src="{src}" alt="x"/></p>', files)
    markdown = EpubConverter().convert(book).markdown
    assert "data:image" not in markdown
    assert f"]({src})" in markdown


def test_an_oversized_picture_keeps_its_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(markitdown_ext, "_EPUB_IMAGE_MAX_BYTES", 16)
    book = _epub(
        tmp_path / "book.epub",
        '<p><img src="../Images/fig.png" alt="x"/></p>',
        {"OEBPS/Images/fig.png": _png()},
    )
    assert "data:image" not in EpubConverter().convert(book).markdown


async def test_the_pictures_are_saved_as_the_items_assets(tmp_path: Path) -> None:
    from markitai.workflow.core import ConversionContext, convert_document_core

    book = _epub(
        tmp_path / "charts.epub",
        '<h1>Bars</h1><p><img src="../Images/fig.png" alt="Bar chart"/></p>'
        '<p><img src="../Images/fig.png" alt="Again"/></p>',
        {"OEBPS/Images/fig.png": _png()},
    )
    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    out = tmp_path / "out"
    ctx = ConversionContext(input_path=book, output_dir=out, config=cfg)
    result = await convert_document_core(ctx, 100 * 1024 * 1024)
    assert result.success, result.error
    assets = sorted((out / ".markitai" / "assets").iterdir())
    assert [a.name.startswith("charts.epub.0001.") for a in assets] == [True]
    markdown = (out / "charts.epub.md").read_text(encoding="utf-8")
    assert f"](.markitai/assets/{assets[0].name})" in markdown
    assert "data:image" not in markdown
