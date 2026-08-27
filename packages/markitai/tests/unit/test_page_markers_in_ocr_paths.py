"""A PDF converted with --ocr still arrives as pages.

The standard PDF path splits into `page_chunks` and writes a marker before
each page. Both OCR paths joined their pages with a bare blank line, so an
OCR'd document reached every later stage as one undivided run of text:

* `llm/content.py` splits a document into batches by page marker; with none
  it falls back to character counting, so a batch can cut mid-page.
* `llm/document.py` skips its structural-drift protection entirely for a
  document it reads as having no pages.
* `output_profiles` rewrites markers to `<!-- page: N -->` for publication;
  it had nothing to rewrite.

None of this failed loudly, because a document with no pages at all — a
`.txt`, a web page — is a legitimate case that looks exactly the same.

These run the real converter over the real fixture; a mocked pymupdf4llm
would only confirm that the mock returns what it was told to.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from markitai.config import MarkitaiConfig
from markitai.constants import PAGE_MARKER_RE, page_marker
from markitai.converter.pdf import PdfConverter

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.pdf"
_FIXTURE_PAGES = 5


def _marked_pages(markdown: str) -> list[int]:
    return [int(m.group(1)) for m in PAGE_MARKER_RE.finditer(markdown)]


@pytest.mark.skipif(not _FIXTURE.is_file(), reason="sample.pdf fixture missing")
class TestVlmOcrPath:
    """`--ocr --llm`: text extracted for the model, pages rendered as images."""

    def test_extracted_text_carries_one_marker_per_page(self, tmp_path: Path) -> None:
        config = MarkitaiConfig()
        config.ocr.enabled = True
        config.llm.enabled = True
        converter = PdfConverter(config=config)

        result = converter._render_pages_for_llm(_FIXTURE, tmp_path)

        assert _marked_pages(result.markdown) == list(range(1, _FIXTURE_PAGES + 1)), (
            "the vision prompt asks the model to keep each page's content "
            "under its own marker and aligned with that page's image; it "
            "cannot do that for text with no page boundaries in it"
        )

    def test_each_marker_introduces_that_page(self, tmp_path: Path) -> None:
        """Markers in the wrong order would misalign every page image."""
        config = MarkitaiConfig()
        config.ocr.enabled = True
        config.llm.enabled = True

        markdown = (
            PdfConverter(config=config)
            ._render_pages_for_llm(_FIXTURE, tmp_path)
            .markdown
        )

        positions = [
            markdown.index(page_marker(n)) for n in range(1, _FIXTURE_PAGES + 1)
        ]
        assert positions == sorted(positions)


@pytest.mark.skipif(not _FIXTURE.is_file(), reason="sample.pdf fixture missing")
class TestRapidOcrPath:
    """`--ocr` alone: RapidOCR reads each page, and the pages are joined."""

    def test_ocr_output_carries_one_marker_per_page(self, tmp_path: Path) -> None:
        config = MarkitaiConfig()
        config.ocr.enabled = True
        config.llm.enabled = False
        config.screenshot.enabled = False
        # Per-page routing keeps pages that already have a text layer, which
        # is all of them here; stub the recognizer so the test does not
        # depend on a model download, and turn routing off so it is used.
        config.ocr.per_page_routing = False

        recognized = Mock()
        recognized.text = "Recognized page text"
        with patch("markitai.ocr.OCRProcessor") as ocr_class:
            ocr_class.return_value.recognize_pdf_page.return_value = recognized
            result = PdfConverter(config=config)._convert_with_ocr(_FIXTURE, tmp_path)

        assert _marked_pages(result.markdown) == list(range(1, _FIXTURE_PAGES + 1))
        assert result.metadata["pages"] == _FIXTURE_PAGES

    def test_a_page_that_failed_ocr_still_gets_its_marker(self, tmp_path: Path) -> None:
        """Otherwise the failure silently merges into the previous page."""
        config = MarkitaiConfig()
        config.ocr.enabled = True
        config.llm.enabled = False
        config.screenshot.enabled = False
        config.ocr.per_page_routing = False

        with patch("markitai.ocr.OCRProcessor") as ocr_class:
            ocr_class.return_value.recognize_pdf_page.side_effect = RuntimeError("boom")
            result = PdfConverter(config=config)._convert_with_ocr(_FIXTURE, tmp_path)

        assert _marked_pages(result.markdown) == list(range(1, _FIXTURE_PAGES + 1))


@pytest.mark.skipif(not _FIXTURE.is_file(), reason="sample.pdf fixture missing")
def test_the_standard_path_and_the_ocr_paths_agree(tmp_path: Path) -> None:
    """One document, two routes, the same page boundaries.

    A reader downstream cannot ask which flags produced a file, so the two
    have to be indistinguishable in how they mark pages.
    """
    standard = PdfConverter(config=MarkitaiConfig())
    plain = standard.convert(_FIXTURE, tmp_path / "plain")

    vlm_config = MarkitaiConfig()
    vlm_config.ocr.enabled = True
    vlm_config.llm.enabled = True
    ocr = PdfConverter(config=vlm_config)._render_pages_for_llm(
        _FIXTURE, tmp_path / "ocr"
    )

    assert _marked_pages(plain.markdown) == _marked_pages(ocr.markdown)
