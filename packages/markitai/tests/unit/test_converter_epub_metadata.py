"""An e-book's metadata goes to the front matter, not into the body.

Owner report (2026-10-12): the base Markdown of an EPUB started with
markitdown's ``**Title:** … **Authors:** … **Language:** … **Publisher:** …
**Date:** … **Description:** …`` block. The book's facts are front matter:
``authors`` (a list), ``language``, ``publisher``, ``date``,
``description``, the keys of Rust's EPUB metadata (``office_meta``), after
``title`` and ``source``. The LLM output keeps them too; its ``description``
is the model's summary, as for every format, and the book's own when the
model gives none.

Fixtures: ``sample.epub`` (markitdown's test book) and
``zh-gutenberg-23825.epub``, Project Gutenberg's 施公案 (public domain) with
its OPF metadata as published and only the start of its first chapter.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import yaml

from markitai.config import MarkitaiConfig
from markitai.converter.markitdown_ext import EpubConverter
from markitai.utils.frontmatter import build_frontmatter_dict, frontmatter_to_yaml

FIXTURES = Path(__file__).parent.parent / "fixtures"


async def _base_output(book: Path, out: Path) -> str:
    from markitai.workflow.core import ConversionContext, convert_document_core

    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    ctx = ConversionContext(input_path=book, output_dir=out, config=cfg)
    result = await convert_document_core(ctx, 100 * 1024 * 1024)
    assert result.success, result.error
    return (out / f"{book.name}.md").read_text(encoding="utf-8")


def _split(text: str) -> tuple[dict, str]:
    _, frontmatter, body = text.split("---\n", 2)
    return yaml.safe_load(frontmatter), body


def _opf_book(path: Path, metadata: str, *, version: str = "2.0") -> Path:
    """A book with *metadata* in its OPF (an epubBuilder or Calibre shape)."""
    container = (
        '<?xml version="1.0"?><container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
        '<rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )
    opf = (
        f'<?xml version="1.0" encoding="UTF-8"?><package version="{version}" '
        'unique-identifier="PrimaryID" xmlns="http://www.idpf.org/2007/opf">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/" '
        f'xmlns:opf="http://www.idpf.org/2007/opf">{metadata}</metadata>'
        '<manifest><item id="c1" href="chapter1.html" '
        'media-type="application/xhtml+xml"/></manifest>'
        '<spine><itemref idref="c1"/></spine></package>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr(
            "OEBPS/chapter1.html",
            '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
            "<h1>第一章</h1><p>正文。</p></body></html>",
        )
    return path


class TestRealBooks:
    async def test_markitdowns_test_book(self, tmp_path: Path) -> None:
        data, body = _split(
            await _base_output(FIXTURES / "sample.epub", tmp_path / "out")
        )
        assert data["title"] == "Test EPUB Document"
        assert data["source"] == "sample.epub"
        assert data["authors"] == ["Test Author"]
        assert data["language"] == "en"
        assert data["description"] == "A test EPUB document for MarkItDown testing"
        assert list(data) == [
            "title",
            "source",
            "authors",
            "language",
            "description",
            "markitai_processed",
        ]
        assert "**Title:**" not in body and "**Authors:**" not in body
        assert body.lstrip().startswith("#")

    async def test_a_chinese_gutenberg_book(self, tmp_path: Path) -> None:
        data, body = _split(
            await _base_output(FIXTURES / "zh-gutenberg-23825.epub", tmp_path / "out")
        )
        assert data["title"] == "施公案"
        assert data["authors"] == ["Anonymous"]
        assert data["language"] == "zh"
        # A date stays a string, not a YAML date
        assert data["date"] == "2007-12-11"
        assert "**Title:** 施公案" not in body
        assert "胡秀才告狀鳴冤" in body


class TestOpfShapes:
    def test_an_epubbuilder_book(self, tmp_path: Path) -> None:
        # The owner's book's shape: EPUB 2, an empty ISBN, a description
        # indented with ideographic spaces, empty elements
        book = _opf_book(
            tmp_path / "book.epub",
            "<dc:title>人月神话(二十周年纪念版)</dc:title>"
            '<dc:identifier opf:scheme="ISBN"></dc:identifier>'
            "<dc:language>zh_CN</dc:language>"
            "<dc:creator>[美] 弗雷德里克·布鲁克斯</dc:creator>"
            "<dc:publisher>epub掌上书苑</dc:publisher>"
            "<dc:description>　　本书讲软件工程。\n  第二句。</dc:description>"
            "<dc:coverage></dc:coverage><dc:date>2011-03-03</dc:date>"
            "<dc:contributor></dc:contributor>",
        )
        result = EpubConverter().convert(book)
        assert result.metadata["title"] == "人月神话(二十周年纪念版)"
        assert result.metadata["document_meta"] == {
            "authors": ["[美] 弗雷德里克·布鲁克斯"],
            "language": "zh_CN",
            "publisher": "epub掌上书苑",
            "date": "2011-03-03",
            "description": "本书讲软件工程。 第二句。",
        }
        assert result.markdown.startswith("# 第一章")

    def test_a_calibre_book(self, tmp_path: Path) -> None:
        # Calibre: several authors, HTML in the description, a timestamp
        # at midnight, and "0101-01-01" for no date
        book = _opf_book(
            tmp_path / "book.epub",
            "<dc:title>Book</dc:title><dc:creator>Ada</dc:creator>"
            "<dc:creator>Bob</dc:creator>"
            "<dc:description>&lt;p&gt;One &amp;amp; &lt;b&gt;two&lt;/b&gt;"
            "&lt;/p&gt;</dc:description>"
            "<dc:date>2019-05-01T00:00:00+00:00</dc:date>",
        )
        meta = EpubConverter().convert(book).metadata["document_meta"]
        assert meta == {
            "authors": ["Ada", "Bob"],
            "date": "2019-05-01",
            "description": "One & two",
        }
        undated = _opf_book(
            tmp_path / "undated.epub",
            "<dc:title>B</dc:title><dc:date>0101-01-01T00:00:00+00:00</dc:date>",
        )
        assert "document_meta" not in EpubConverter().convert(undated).metadata

    @pytest.mark.parametrize("version", ["2.0", "3.0"])
    def test_a_book_without_metadata_has_none(
        self, tmp_path: Path, version: str
    ) -> None:
        book = _opf_book(tmp_path / "b.epub", "", version=version)
        result = EpubConverter().convert(book)
        assert "document_meta" not in result.metadata
        assert "title" not in result.metadata


class TestLlmOutput:
    _META = {
        "authors": ["[美] 弗雷德里克·布鲁克斯"],
        "language": "zh_CN",
        "publisher": "epub掌上书苑",
        "date": "2011-03-03",
        "description": "出版社的长简介。",
    }

    def test_the_books_fields_follow_the_source(self) -> None:
        data = yaml.safe_load(
            frontmatter_to_yaml(
                build_frontmatter_dict(
                    source="book.epub",
                    title="人月神话(二十周年纪念版)",
                    description="模型写的摘要。",
                    tags=["软件工程"],
                    document_meta=self._META,
                )
            )
        )
        assert list(data)[:7] == [
            "title",
            "source",
            "authors",
            "language",
            "publisher",
            "date",
            "description",
        ]
        assert data["description"] == "模型写的摘要。"
        assert data["date"] == "2011-03-03"

    def test_the_books_description_when_the_model_gives_none(self) -> None:
        data = build_frontmatter_dict(
            source="book.epub", title="T", document_meta=self._META
        )
        assert data["description"] == "出版社的长简介。"

    def test_the_batch_path_reads_them_back_from_the_base(self, tmp_path: Path) -> None:
        from markitai.cli.processors.batch_llm import _base_facts
        from markitai.workflow.helpers import add_basic_frontmatter

        base = tmp_path / "book.epub.md"
        base.write_text(
            add_basic_frontmatter(
                "# 第一章", "book.epub", title="人月神话", document_meta=self._META
            ),
            encoding="utf-8",
        )
        assert _base_facts(base) == ("人月神话", self._META)
