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


def _epub(
    path: Path,
    body: str,
    files: dict[str, bytes],
    *,
    chapters: list[tuple[str, str]] | None = None,
    metadata: str = "",
    manifest: str = "",
) -> Path:
    """A book whose spine is *chapters* (OPF href, body), else one chapter.

    A chapter is stored under ``OEBPS/`` at its href, percent-decoded.
    """
    from urllib.parse import unquote

    chapters = chapters or [("Text/ch1.xhtml", body)]
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
        f'<dc:identifier id="id">charts</dc:identifier>{metadata}</metadata>'
        "<manifest>"
        + "".join(
            f'<item id="ch{n}" href="{href}" media-type="application/xhtml+xml"/>'
            for n, (href, _) in enumerate(chapters, 1)
        )
        + f"{manifest}</manifest><spine>"
        + "".join(f'<itemref idref="ch{n}"/>' for n in range(1, len(chapters) + 1))
        + "</spine></package>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/content.opf", opf)
        for href, chapter_body in chapters:
            archive.writestr(
                f"OEBPS/{unquote(href)}",
                '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml">'
                f"<head><title>Chapter</title></head><body>{chapter_body}</body></html>",
            )
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def _data_uri(data: bytes, kind: str = "png") -> str:
    return f"data:image/{kind};base64,{base64.b64encode(data).decode('ascii')}"


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


# How Calibre, Sigil and ebookmaker (Project Gutenberg) write a cover page
_SVG_COVER = (
    '<div><svg xmlns="http://www.w3.org/2000/svg" '
    'xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" '
    'viewBox="0 0 600 800" preserveAspectRatio="xMidYMid meet">'
    '<image width="600" height="800" {attr}="{href}"/></svg></div>'
)


@pytest.mark.parametrize("attr", ["xlink:href", "href"])
def test_a_cover_picture_wrapped_in_svg_is_kept(tmp_path: Path, attr: str) -> None:
    cover = _png((180, 40, 40))
    book = _epub(
        tmp_path / "book.epub",
        "",
        {"OEBPS/Images/cover.jpg": cover},
        chapters=[
            (
                "Text/cover.xhtml",
                _SVG_COVER.format(attr=attr, href="../Images/cover.jpg"),
            ),
            ("Text/ch1.xhtml", "<h1>One</h1><p>Text.</p>"),
        ],
    )
    markdown = EpubConverter().convert(book).markdown
    assert f"![]({_data_uri(cover, 'jpeg')})" in markdown
    assert markdown.index("data:image") < markdown.index("# One")
    assert "<svg" not in markdown and "../Images" not in markdown


def test_a_chapter_the_opf_percent_encodes_is_converted(tmp_path: Path) -> None:
    # Sigil writes a non-ASCII file name percent-encoded in the OPF;
    # markitdown looked it up as written and skipped the whole chapter
    picture = _png()
    book = _epub(
        tmp_path / "book.epub",
        "",
        {"OEBPS/Images/插图 1.png": picture},
        chapters=[
            (
                "Text/%E7%AC%AC%E4%B8%80%E7%AB%A0.xhtml",
                "<h1>第一章</h1><p>正文。</p>"
                '<p><img src="../Images/%E6%8F%92%E5%9B%BE%201.png" alt="插图"/></p>',
            )
        ],
    )
    markdown = EpubConverter().convert(book).markdown
    assert "# 第一章" in markdown and "正文。" in markdown
    assert f"![插图]({_data_uri(picture)})" in markdown


@pytest.mark.parametrize(
    ("metadata", "manifest"),
    [
        # EPUB 2: <meta name="cover"> names the manifest item
        (
            '<meta name="cover" content="cover-image"/>',
            '<item id="cover-image" href="Images/cover.png" media-type="image/png"/>',
        ),
        # EPUB 3: the manifest item says it is the cover
        (
            "",
            '<item id="c" href="Images/cover.png" media-type="image/png" '
            'properties="cover-image"/>',
        ),
    ],
)
def test_a_cover_only_the_opf_names_comes_first(
    tmp_path: Path, metadata: str, manifest: str
) -> None:
    cover = _png((220, 120, 0))
    book = _epub(
        tmp_path / "book.epub",
        "<h1>One</h1><p>Text.</p>",
        {"OEBPS/Images/cover.png": cover},
        metadata=metadata,
        manifest=manifest,
    )
    markdown = EpubConverter().convert(book).markdown
    assert f"![Cover]({_data_uri(cover)})\n\n# One" in markdown


def test_a_cover_a_page_shows_is_not_added_again(tmp_path: Path) -> None:
    cover = _png((220, 120, 0))
    book = _epub(
        tmp_path / "book.epub",
        "",
        {"OEBPS/Images/cover.png": cover},
        chapters=[
            ("Text/cover.xhtml", '<p><img src="../Images/cover.png" alt=""/></p>'),
        ],
        metadata='<meta name="cover" content="cover-image"/>',
        manifest='<item id="cover-image" href="Images/cover.png" '
        'media-type="image/png"/>',
    )
    markdown = EpubConverter().convert(book).markdown
    assert markdown.count("data:image") == 1
    assert "![Cover]" not in markdown


def test_a_reference_resolves_against_its_chapters_folder(tmp_path: Path) -> None:
    # Two pictures of that name: the one beside the chapter's ../Images
    wanted, other = _png((0, 150, 0)), _png((150, 0, 0))
    book = _epub(
        tmp_path / "book.epub",
        '<p><img src="../Images/fig.png" alt="fig"/></p>',
        {"OEBPS/Images/fig.png": wanted, "OEBPS/Extra/Images/fig.png": other},
    )
    assert f"![fig]({_data_uri(wanted)})" in EpubConverter().convert(book).markdown


def test_a_picture_with_a_title_stays_a_plain_link(tmp_path: Path) -> None:
    # The image pipeline replaces ![alt](data:…) only; a title, or a bracket
    # in the alt text, would leave the whole data: URI in the output
    picture = _png()
    book = _epub(
        tmp_path / "book.epub",
        '<p><img src="../Images/a.png" title="Figure [1]"/></p>'
        '<p><img src="../Images/a.png" alt="Bars [2]" title="t"/></p>',
        {"OEBPS/Images/a.png": picture},
    )
    markdown = EpubConverter().convert(book).markdown
    assert f"![Figure 1]({_data_uri(picture)})" in markdown
    assert f"![Bars 2]({_data_uri(picture)})" in markdown


def test_a_book_over_its_picture_budget_keeps_the_rest_as_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _png((1, 2, 3)), _png((4, 5, 6))
    monkeypatch.setattr(markitdown_ext, "_EPUB_IMAGES_MAX_BYTES", len(first))
    book = _epub(
        tmp_path / "book.epub",
        '<p><img src="../Images/a.png" alt="a"/></p>'
        '<p><img src="../Images/b.png" alt="b"/></p>',
        {"OEBPS/Images/a.png": first, "OEBPS/Images/b.png": second},
    )
    markdown = EpubConverter().convert(book).markdown
    assert f"![a]({_data_uri(first)})" in markdown
    assert "![b](../Images/b.png)" in markdown


def test_the_text_is_markitdowns(tmp_path: Path) -> None:
    # Only the pictures differ from markitdown's own EPUB conversion, and
    # markitdown's bold metadata block, which goes to the front matter
    import re

    from markitai.converter.markitdown_ext import _convert

    book = _epub(
        tmp_path / "book.epub",
        "<h1>Poem</h1><p>One line<br/>\n\u00a0\u00a0another line</p>"
        "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"
        '<p><a href="ch1.xhtml#x">link</a> <img src="../Images/a.png" alt="a"/></p>'
        "<script>var x;</script><ul><li>item</li></ul>",
        {"OEBPS/Images/a.png": _png()},
        metadata="<dc:creator>Ada</dc:creator><dc:creator>Bob</dc:creator>"
        "<dc:language>en</dc:language>",
    )

    def masked(markdown: str) -> str:
        return re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"![\1]()", markdown)

    ours = EpubConverter().convert(book)
    theirs = _convert(book)
    block = "**Title:** Charts\n**Authors:** Ada, Bob\n**Language:** en\n"
    block += "**Identifier:** charts\n\n"
    assert theirs.markdown.startswith(block)
    assert masked(ours.markdown) == masked(theirs.markdown.removeprefix(block))
    assert ours.metadata["title"] == theirs.metadata["title"] == "Charts"
    assert ours.metadata["document_meta"] == {
        "authors": ["Ada", "Bob"],
        "language": "en",
    }


async def test_a_calibre_books_cover_and_figures_are_the_items_assets(
    tmp_path: Path,
) -> None:
    from markitai.workflow.core import ConversionContext, convert_document_core

    # Calibre: files at the archive root beside titlepage.xhtml
    cover, figure = _png((180, 40, 40)), _png((40, 120, 40))
    book = tmp_path / "novel.epub"
    _epub(
        book,
        "",
        {"OEBPS/cover.jpeg": cover, "OEBPS/images/00001.jpg": figure},
        chapters=[
            (
                "titlepage.xhtml",
                _SVG_COVER.format(attr="xlink:href", href="cover.jpeg"),
            ),
            (
                "index_split_000.html",
                '<h1>Chapter 1</h1><p><img src="images/00001.jpg" class="c1"/></p>',
            ),
        ],
    )
    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    out = tmp_path / "out"
    ctx = ConversionContext(input_path=book, output_dir=out, config=cfg)
    result = await convert_document_core(ctx, 100 * 1024 * 1024)
    assert result.success, result.error
    assets = sorted(p.name for p in (out / ".markitai" / "assets").iterdir())
    assert assets == ["novel.epub.0001.jpg", "novel.epub.0002.jpg"]
    markdown = (out / "novel.epub.md").read_text(encoding="utf-8")
    for name in assets:
        assert f"](.markitai/assets/{name})" in markdown
    assert "data:image" not in markdown and "cover.jpeg" not in markdown


def test_a_cover_meta_naming_the_cover_page_falls_back_to_the_cover_image(
    tmp_path: Path,
) -> None:
    cover = _png((220, 120, 0))
    book = _epub(
        tmp_path / "book.epub",
        "<h1>One</h1>",
        {"OEBPS/Images/cover.png": cover},
        metadata='<meta name="cover" content="ch1"/>',
        manifest='<item id="c" href="Images/cover.png" media-type="image/png" '
        'properties="cover-image"/>',
    )
    assert f"![Cover]({_data_uri(cover)})" in EpubConverter().convert(book).markdown


@pytest.mark.parametrize(
    "files",
    [
        {"mimetype": b"application/epub+zip"},  # no META-INF/container.xml
        {"META-INF/container.xml": b"<container><rootfiles>"},  # not XML
    ],
)
def test_a_malformed_book_fails(tmp_path: Path, files: dict[str, bytes]) -> None:
    from markitai.utils.errors import ConversionError

    book = tmp_path / "book.epub"
    with zipfile.ZipFile(book, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with pytest.raises(ConversionError, match="Malformed EPUB"):
        EpubConverter().convert(book)
