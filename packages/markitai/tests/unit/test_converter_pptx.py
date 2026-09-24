"""Unit tests for PPTX converter — OCR/Vision slide rendering paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from markitai.config import (
    LLMConfig,
    MarkitaiConfig,
    OCRConfig,
    ScreenshotConfig,
)
from markitai.converter.base import ConvertResult, ExtractedImage
from markitai.converter.office import PptxConverter


def _make_pptx_converter(
    *,
    ocr: bool = False,
    llm: bool = False,
    screenshot: bool = False,
) -> PptxConverter:
    """Create a PptxConverter with the given flag combination."""
    config = MarkitaiConfig(
        ocr=OCRConfig(enabled=ocr),
        llm=LLMConfig(enabled=llm),
        screenshot=ScreenshotConfig(enabled=screenshot),
    )
    return PptxConverter(config)


def _stub_render_slides(
    converter: PptxConverter,
    slide_count: int = 3,
) -> Mock:
    """Patch _render_slides_to_images to return fake slide data.

    Returns the mock so the caller can assert it was called.
    """
    images = [
        ExtractedImage(
            path=Path(f"/tmp/slide{i}.jpg"),
            index=i,
            original_name=f"slide{i}.jpg",
            mime_type="image/jpeg",
            width=800,
            height=600,
        )
        for i in range(1, slide_count + 1)
    ]
    slide_images = [
        {"page": i, "path": f"/tmp/slide{i}.jpg", "name": f"slide{i}.jpg"}
        for i in range(1, slide_count + 1)
    ]
    mock = Mock(return_value=(images, slide_images))
    converter._render_slides_to_images = mock  # type: ignore[method-assign]
    return mock


class TestPptxOcrOnlySlideRendering:
    """OCR-only mode (--ocr, no --llm) must render slides regardless of screenshot flag."""

    def test_ocr_only_renders_slides_when_screenshot_disabled(
        self, tmp_path: Path
    ) -> None:
        """Slide images must be generated even when screenshot.enabled=False."""
        converter = _make_pptx_converter(ocr=True, llm=False, screenshot=False)

        # Stub MarkItDown extraction
        mock_markitdown_result = MagicMock()
        mock_markitdown_result.markdown = "# Slide text"
        mock_markitdown_result.title = None
        converter._markitdown = MagicMock()  # type: ignore[reportAttributeAccessIssue]
        converter._markitdown.convert.return_value = mock_markitdown_result  # type: ignore[reportAttributeAccessIssue]

        render_mock = _stub_render_slides(converter, slide_count=2)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(Path("test.pptx"), output_dir)

        # The render function MUST have been called
        render_mock.assert_called_once()
        # Result must contain the slide images
        # Slides are screenshots, not embedded images
        assert result.metadata["slides"] == 2
        assert result.images == []
        # No picture on the slides, so nothing was OCR'd: say so
        assert result.metadata.get("ocr_used") is False

    def test_ocr_only_renders_slides_when_screenshot_enabled(
        self, tmp_path: Path
    ) -> None:
        """Sanity: slide images are also generated when screenshot.enabled=True."""
        converter = _make_pptx_converter(ocr=True, llm=False, screenshot=True)

        mock_markitdown_result = MagicMock()
        mock_markitdown_result.markdown = "# Slide text"
        mock_markitdown_result.title = None
        converter._markitdown = MagicMock()  # type: ignore[reportAttributeAccessIssue]
        converter._markitdown.convert.return_value = mock_markitdown_result  # type: ignore[reportAttributeAccessIssue]

        render_mock = _stub_render_slides(converter, slide_count=2)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(Path("test.pptx"), output_dir)

        render_mock.assert_called_once()
        # Slides are screenshots, not embedded images
        assert result.metadata["slides"] == 2
        assert result.images == []

    def test_ocr_only_includes_commented_image_refs_in_markdown(
        self, tmp_path: Path
    ) -> None:
        """OCR-only markdown should contain commented slide image references."""
        converter = _make_pptx_converter(ocr=True, llm=False, screenshot=False)

        mock_markitdown_result = MagicMock()
        mock_markitdown_result.markdown = "# Slide text"
        mock_markitdown_result.title = None
        converter._markitdown = MagicMock()  # type: ignore[reportAttributeAccessIssue]
        converter._markitdown.convert.return_value = mock_markitdown_result  # type: ignore[reportAttributeAccessIssue]

        _stub_render_slides(converter, slide_count=2)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(Path("test.pptx"), output_dir)

        # Commented image references should be present
        assert "<!-- " in result.markdown
        assert "Slide" in result.markdown

    def test_ocr_only_without_output_dir_uses_temp(self) -> None:
        """When no output_dir, a temp directory should be used for rendering."""
        converter = _make_pptx_converter(ocr=True, llm=False, screenshot=False)

        mock_markitdown_result = MagicMock()
        mock_markitdown_result.markdown = "# Slide text"
        mock_markitdown_result.title = None
        converter._markitdown = MagicMock()  # type: ignore[reportAttributeAccessIssue]
        converter._markitdown.convert.return_value = mock_markitdown_result  # type: ignore[reportAttributeAccessIssue]

        render_mock = _stub_render_slides(converter, slide_count=1)

        result = converter.convert(Path("test.pptx"), None)

        # Even without output_dir, rendering must happen
        render_mock.assert_called_once()
        # Slides are screenshots, not embedded images
        assert result.metadata["slides"] == 1
        assert result.images == []


class TestPptxOcrLlmSlideRendering:
    """OCR+LLM mode (--ocr --llm) must render slides regardless of screenshot flag."""

    def test_ocr_llm_renders_slides_when_screenshot_disabled(
        self, tmp_path: Path
    ) -> None:
        """Slide images for LLM Vision must be generated even when screenshot.enabled=False."""
        converter = _make_pptx_converter(ocr=True, llm=True, screenshot=False)

        mock_markitdown_result = MagicMock()
        mock_markitdown_result.markdown = "# Extracted text"
        mock_markitdown_result.title = None
        converter._markitdown = MagicMock()  # type: ignore[reportAttributeAccessIssue]
        converter._markitdown.convert.return_value = mock_markitdown_result  # type: ignore[reportAttributeAccessIssue]

        render_mock = _stub_render_slides(converter, slide_count=3)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(Path("test.pptx"), output_dir)

        # The render function MUST have been called
        render_mock.assert_called_once()
        # Result must contain the slide images
        # Slides are screenshots, not embedded images
        assert result.metadata["slides"] == 3
        assert result.images == []
        # page_images metadata must be populated for LLM processing
        assert len(result.metadata.get("page_images", [])) == 3

    def test_ocr_llm_renders_slides_when_screenshot_enabled(
        self, tmp_path: Path
    ) -> None:
        """Sanity: slide images are also generated when screenshot.enabled=True."""
        converter = _make_pptx_converter(ocr=True, llm=True, screenshot=True)

        mock_markitdown_result = MagicMock()
        mock_markitdown_result.markdown = "# Extracted text"
        mock_markitdown_result.title = None
        converter._markitdown = MagicMock()  # type: ignore[reportAttributeAccessIssue]
        converter._markitdown.convert.return_value = mock_markitdown_result  # type: ignore[reportAttributeAccessIssue]

        render_mock = _stub_render_slides(converter, slide_count=3)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(Path("test.pptx"), output_dir)

        render_mock.assert_called_once()
        # Slides are screenshots, not embedded images
        assert result.metadata["slides"] == 3
        assert result.images == []
        assert len(result.metadata.get("page_images", [])) == 3

    def test_ocr_llm_metadata_contains_extracted_text(self, tmp_path: Path) -> None:
        """Metadata should contain extracted_text for LLM to combine with images."""
        converter = _make_pptx_converter(ocr=True, llm=True, screenshot=False)

        mock_markitdown_result = MagicMock()
        mock_markitdown_result.markdown = "# Extracted text from slides"
        mock_markitdown_result.title = None
        converter._markitdown = MagicMock()  # type: ignore[reportAttributeAccessIssue]
        converter._markitdown.convert.return_value = mock_markitdown_result  # type: ignore[reportAttributeAccessIssue]

        _stub_render_slides(converter, slide_count=1)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(Path("test.pptx"), output_dir)

        assert "extracted_text" in result.metadata
        assert "Extracted text" in result.metadata["extracted_text"]

    def test_ocr_llm_without_output_dir_uses_temp(self) -> None:
        """When no output_dir, a temp directory should be used for rendering."""
        converter = _make_pptx_converter(ocr=True, llm=True, screenshot=False)

        mock_markitdown_result = MagicMock()
        mock_markitdown_result.markdown = "# Slide text"
        mock_markitdown_result.title = None
        converter._markitdown = MagicMock()  # type: ignore[reportAttributeAccessIssue]
        converter._markitdown.convert.return_value = mock_markitdown_result  # type: ignore[reportAttributeAccessIssue]

        render_mock = _stub_render_slides(converter, slide_count=1)

        result = converter.convert(Path("test.pptx"), None)

        render_mock.assert_called_once()
        # Slides are screenshots, not embedded images
        assert result.metadata["slides"] == 1
        assert result.images == []
        assert len(result.metadata.get("page_images", [])) == 1


class TestPptxDefaultModeScreenshotIndependence:
    """Default mode (no --ocr) should NOT render slides unless --screenshot is on."""

    def test_default_mode_no_screenshots_when_disabled(self, tmp_path: Path) -> None:
        """Without --ocr and --screenshot, no slide rendering should happen."""
        converter = _make_pptx_converter(ocr=False, llm=False, screenshot=False)

        mock_markitdown_result = MagicMock()
        mock_markitdown_result.markdown = "# Just text"
        mock_markitdown_result.title = None
        converter._markitdown = MagicMock()  # type: ignore[reportAttributeAccessIssue]
        converter._markitdown.convert.return_value = mock_markitdown_result  # type: ignore[reportAttributeAccessIssue]

        render_mock = _stub_render_slides(converter, slide_count=2)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(Path("test.pptx"), output_dir)

        # Render should NOT be called in default mode without --screenshot
        render_mock.assert_not_called()
        assert len(result.images) == 0


class TestPptxScreenshotExtensionConsistency:
    """save_screenshot may change .png to .jpg in the extreme fallback.

    PPTX converter must use the actual path returned by save_screenshot
    so that ExtractedImage metadata points to the real file on disk.
    """

    def test_pptx_render_uses_actual_path_from_save_screenshot(
        self, tmp_path: Path
    ) -> None:
        """When save_screenshot returns a different path (e.g. .jpg instead of .png),
        the PPTX converter _render_slides_via_pdf must use that path in
        ExtractedImage and slide_images metadata.
        """
        import sys
        from unittest.mock import Mock, patch

        # Setup pymupdf mock
        mock_pymupdf = Mock()
        mock_page = Mock()
        mock_pix = Mock()
        mock_pix.samples = b"fake_pixel_data"
        mock_pix.width = 800
        mock_pix.height = 600
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_doc.close = Mock()
        mock_pymupdf.open.return_value = mock_doc
        mock_pymupdf.Matrix.return_value = Mock()

        screenshots_dir = tmp_path / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Simulate save_screenshot returning a different path (.jpg instead of .png)
        actual_jpg_path = screenshots_dir / "test.pptx.slide0001.jpg"
        actual_jpg_path.write_bytes(b"\xff\xd8fake_jpeg")

        mock_img_processor = Mock()
        mock_img_processor.save_screenshot.return_value = (
            (800, 600),
            actual_jpg_path,
        )

        converter = _make_pptx_converter(ocr=True, llm=True, screenshot=False)

        # Use a known temp directory and create the fake PDF there
        lo_temp_dir = tmp_path / "lo_temp"
        lo_temp_dir.mkdir()
        fake_pdf = lo_temp_dir / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        # Mock subprocess to succeed
        mock_subprocess_result = Mock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stderr = ""

        def fake_subprocess_run(*args, **kwargs):
            """Simulate LibreOffice creating the PDF file."""
            # The PDF was already created above
            return mock_subprocess_result

        # Create a mock TemporaryDirectory context manager
        mock_temp_dir = MagicMock()
        mock_temp_dir.__enter__ = Mock(return_value=str(lo_temp_dir))
        mock_temp_dir.__exit__ = Mock(return_value=False)

        with (
            patch.dict(sys.modules, {"pymupdf": mock_pymupdf}),
            patch(
                "markitai.image.ImageProcessor",
                return_value=mock_img_processor,
            ),
            patch(
                "markitai.converter.office.find_libreoffice",
                return_value="/usr/bin/soffice",
            ),
            patch("markitai.converter.office.has_ms_office", return_value=False),
            patch("subprocess.run", side_effect=fake_subprocess_run),
            patch("tempfile.TemporaryDirectory", return_value=mock_temp_dir),
        ):
            images, slide_images = converter._render_slides_via_pdf(  # type: ignore[reportAttributeAccessIssue]
                Path("test.pptx"), screenshots_dir, "png"
            )

        # The ExtractedImage path must match the actual file on disk
        assert len(images) == 1
        assert images[0].path == actual_jpg_path
        assert images[0].mime_type == "image/jpeg"
        assert images[0].original_name == "test.pptx.slide0001.jpg"

        # slide_images metadata must also use the actual path
        assert len(slide_images) == 1
        assert slide_images[0]["path"] == str(actual_jpg_path)
        assert slide_images[0]["name"] == "test.pptx.slide0001.jpg"


def _stub_text(converter: PptxConverter, markdown: str) -> None:
    result = MagicMock()
    result.markdown = markdown
    result.title = None
    converter._markitdown = MagicMock()  # type: ignore[reportAttributeAccessIssue]
    converter._markitdown.convert.return_value = result  # type: ignore[reportAttributeAccessIssue]


def _picture_data_uri(width: int, height: int) -> str:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, "PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"![chart](data:image/png;base64,{encoded})"


class TestPptxVlmOcrPrivacyGate:
    """--ocr --llm on PPTX honors MARKITAI_NO_VLM_OCR like the PDF path."""

    def test_no_vlm_ocr_never_hands_slides_to_the_vision_model(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("MARKITAI_NO_VLM_OCR", "1")
        converter = _make_pptx_converter(ocr=True, llm=True)
        _stub_text(converter, "# Slide text")
        _stub_render_slides(converter, slide_count=2)

        with (
            patch("markitai.converter.office.is_ocr_available", return_value=True),
            patch("markitai.converter.office.ensure_vlm_ocr_disclosed") as disclose,
        ):
            result = converter.convert(Path("test.pptx"), tmp_path)

        assert "page_images" not in result.metadata
        assert result.metadata.get("ocr_path") != "vlm"
        disclose.assert_not_called()

    def test_no_vlm_ocr_without_rapidocr_fails_clearly(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from markitai.ocr import OCRBackendMissing

        monkeypatch.setenv("MARKITAI_NO_VLM_OCR", "1")
        converter = _make_pptx_converter(ocr=True, llm=True)
        with (
            patch("markitai.converter.office.is_ocr_available", return_value=False),
            pytest.raises(OCRBackendMissing, match="MARKITAI_NO_VLM_OCR"),
        ):
            converter.convert(Path("test.pptx"), tmp_path)

    def test_vlm_path_discloses_when_slides_are_sent(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.delenv("MARKITAI_NO_VLM_OCR", raising=False)
        converter = _make_pptx_converter(ocr=True, llm=True)
        _stub_text(converter, "# Slide text")
        _stub_render_slides(converter, slide_count=2)

        with patch("markitai.converter.office.ensure_vlm_ocr_disclosed") as disclose:
            result = converter.convert(Path("test.pptx"), tmp_path)

        disclose.assert_called_once()
        assert disclose.call_args.kwargs["page_count"] == 2
        assert result.metadata["ocr_path"] == "vlm"

    def test_nothing_rendered_means_nothing_disclosed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.delenv("MARKITAI_NO_VLM_OCR", raising=False)
        converter = _make_pptx_converter(ocr=True, llm=True)
        _stub_text(converter, "# Slide text")
        converter._render_slides_to_images = Mock(return_value=([], []))  # type: ignore[method-assign]

        with patch("markitai.converter.office.ensure_vlm_ocr_disclosed") as disclose:
            converter.convert(Path("test.pptx"), tmp_path)

        disclose.assert_not_called()


class TestPptxLocalOcr:
    """--ocr alone reads the pictures on the slides; ocr_used says whether."""

    def test_pictures_are_ocrd_under_their_reference(self, tmp_path: Path) -> None:
        converter = _make_pptx_converter(ocr=True)
        big = _picture_data_uri(400, 300)
        icon = _picture_data_uri(32, 32)
        _stub_text(converter, f"# Slide\n\n{big}\n\n{icon}\n")
        _stub_render_slides(converter, slide_count=1)

        with patch("markitai.ocr.OCRProcessor") as ocr_cls:
            ocr_cls.return_value.recognize_bytes.return_value = Mock(
                text="Revenue grew 42%"
            )
            result = converter.convert(Path("test.pptx"), tmp_path)

        ocr_cls.return_value.recognize_bytes.assert_called_once()  # icon skipped
        assert f"{big}\n\nRevenue grew 42%" in result.markdown
        assert result.metadata["ocr_used"] is True

    def test_text_only_deck_reports_no_ocr(self, tmp_path: Path) -> None:
        converter = _make_pptx_converter(ocr=True)
        _stub_text(converter, "# Slide\n\nJust text")
        _stub_render_slides(converter, slide_count=1)

        with patch("markitai.ocr.OCRProcessor") as ocr_cls:
            result = converter.convert(Path("test.pptx"), tmp_path)

        ocr_cls.assert_not_called()
        assert result.metadata["ocr_used"] is False


class TestPptxRenderFailuresAreUserNotices:
    """A deck whose slides cannot be rendered says so on the default console."""

    def _records(self):
        from loguru import logger

        records: list[Any] = []
        sink_id = logger.add(lambda m: records.append(m.record), level="WARNING")
        return records, sink_id

    def test_missing_renderer_is_a_user_notice_naming_the_deck(
        self, tmp_path: Path
    ) -> None:
        from loguru import logger

        from markitai.notices import is_user_notice

        converter = _make_pptx_converter(screenshot=True)
        records, sink_id = self._records()
        try:
            with (
                patch("markitai.converter.office.find_libreoffice", return_value=None),
                patch("markitai.converter.office.has_ms_office", return_value=False),
                patch("platform.system", return_value="Linux"),
            ):
                images, slides = converter._render_slides_to_images(
                    Path("deck.pptx"), tmp_path, "jpg"
                )
        finally:
            logger.remove(sink_id)

        assert (images, slides) == ([], [])
        notices = [r for r in records if is_user_notice(r)]
        assert len(notices) == 1
        assert "Cannot render slides of deck.pptx" in notices[0]["message"]

    def test_powerpoint_staging_permission_error_degrades(self, tmp_path: Path) -> None:
        """macOS App Data protection (EPERM) must not fail the whole deck."""
        from loguru import logger

        from markitai.notices import is_user_notice

        converter = _make_pptx_converter(screenshot=True)
        records, sink_id = self._records()
        try:
            with (
                patch("markitai.converter.office.find_libreoffice", return_value=None),
                patch("platform.system", return_value="Darwin"),
                patch(
                    "markitai.converter.office.office_mac.powerpoint_available",
                    return_value=True,
                ),
                patch(
                    "markitai.converter.office.office_mac.pptx_to_pdf",
                    side_effect=PermissionError(1, "Operation not permitted"),
                ),
            ):
                images, slides = converter._render_slides_via_pdf(
                    Path("deck.pptx"), tmp_path, "jpg"
                )
        finally:
            logger.remove(sink_id)

        assert (images, slides) == ([], [])
        assert any(
            is_user_notice(r) and "PowerPoint PDF export failed" in r["message"]
            for r in records
        )

    def test_blocked_office_container_names_the_way_out(self, tmp_path: Path) -> None:
        """EPERM on the Office container gets an actionable hint, not errno."""
        from loguru import logger

        from markitai.notices import is_user_notice
        from markitai.utils import office_mac

        blocked = office_mac._STAGING_ROOT / "0123abcd"
        converter = _make_pptx_converter(screenshot=True)
        records, sink_id = self._records()
        try:
            with (
                patch("markitai.converter.office.find_libreoffice", return_value=None),
                patch("platform.system", return_value="Darwin"),
                patch(
                    "markitai.converter.office.office_mac.powerpoint_available",
                    return_value=True,
                ),
                patch(
                    "markitai.converter.office.office_mac.pptx_to_pdf",
                    side_effect=PermissionError(
                        1, "Operation not permitted", str(blocked)
                    ),
                ),
            ):
                images, slides = converter._render_slides_via_pdf(
                    Path("deck.pptx"), tmp_path, "jpg"
                )
        finally:
            logger.remove(sink_id)

        assert (images, slides) == ([], [])
        notices = [r["message"] for r in records if is_user_notice(r)]
        assert len(notices) == 1
        assert "Cannot render slides of deck.pptx" in notices[0]
        assert "Full Disk Access" in notices[0]
        assert "brew install --cask libreoffice" in notices[0]
        assert "Errno" not in notices[0]


class TestOutputDerivedSlideNames:
    """Slide screenshots take the resolved output's prefix (asset_prefix)."""

    def test_slides_via_pdf_use_the_asset_prefix(self, tmp_path: Path) -> None:
        import sys

        mock_pymupdf = Mock()
        mock_pix = Mock(samples=b"px", width=8, height=6)
        mock_page = Mock()
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=2)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_pymupdf.open.return_value = mock_doc

        screenshots_dir = tmp_path / "screenshots"
        screenshots_dir.mkdir()
        mock_img_processor = Mock()

        def save(_samples, width, height, path):
            path.write_bytes(b"img")
            return (width, height), path

        mock_img_processor.save_screenshot.side_effect = save

        lo_temp_dir = tmp_path / "lo_temp"
        lo_temp_dir.mkdir()
        (lo_temp_dir / "deck.pdf").write_bytes(b"%PDF-1.4 fake")
        mock_temp_dir = MagicMock()
        mock_temp_dir.__enter__ = Mock(return_value=str(lo_temp_dir))
        mock_temp_dir.__exit__ = Mock(return_value=False)

        converter = _make_pptx_converter(ocr=True, llm=True, screenshot=False)
        converter.asset_prefix = "deck.pptx.v2"

        with (
            patch.dict(sys.modules, {"pymupdf": mock_pymupdf}),
            patch("markitai.image.ImageProcessor", return_value=mock_img_processor),
            patch(
                "markitai.converter.office.find_libreoffice",
                return_value="/usr/bin/soffice",
            ),
            patch("markitai.converter.office.has_ms_office", return_value=False),
            patch("subprocess.run", return_value=Mock(returncode=0, stderr="")),
            patch("tempfile.TemporaryDirectory", return_value=mock_temp_dir),
        ):
            _images, slide_images = converter._render_slides_via_pdf(  # type: ignore[reportAttributeAccessIssue]
                Path("deck.pptx"), screenshots_dir, "png"
            )

        assert [s["name"] for s in slide_images] == [
            "deck.pptx.v2.slide0001.png",
            "deck.pptx.v2.slide0002.png",
        ]


class TestDefaultScreenshotMode:
    """--screenshot without --ocr: slides are screenshots, referenced in comments."""

    TEXT = (
        "<!-- Slide number: 1 -->\n# Intro\n\nHello\n\n"
        "<!-- Slide number: 2 -->\n# Picture\n\n![logo](data:image/png;base64,AAAA)\n"
    )

    def _convert(self, tmp_path: Path, slide_count: int = 2) -> ConvertResult:
        converter = _make_pptx_converter(screenshot=True)
        converter._convert_with_markitdown = Mock(  # type: ignore[method-assign]
            return_value=ConvertResult(markdown=self.TEXT, metadata={})
        )
        slide_images = [
            {
                "page": i,
                "path": f"/x/deck.pptx.v3.slide{i:04d}.jpg",
                "name": f"deck.pptx.v3.slide{i:04d}.jpg",
            }
            for i in range(1, slide_count + 1)
        ]
        slides = [
            ExtractedImage(
                path=Path(info["path"]),
                index=info["page"],
                original_name=info["name"],
                mime_type="image/jpeg",
                width=8,
                height=6,
            )
            for info in slide_images
        ]
        converter._render_slides_to_images = Mock(  # type: ignore[method-assign]
            return_value=(slides, slide_images)
        )
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        return converter.convert(Path("deck.pptx"), out)

    def test_slide_screenshots_are_not_counted_as_images(self, tmp_path: Path) -> None:
        result = self._convert(tmp_path)

        assert result.images == []
        assert len(result.metadata["page_images"]) == 2

    def test_each_slide_references_its_screenshot_after_its_content(
        self, tmp_path: Path
    ) -> None:
        result = self._convert(tmp_path)
        md = result.markdown

        first = "<!-- ![Slide 1](.markitai/screenshots/deck.pptx.v3.slide0001.jpg) -->"
        second = "<!-- ![Slide 2](.markitai/screenshots/deck.pptx.v3.slide0002.jpg) -->"
        assert md.index("Hello") < md.index(first) < md.index("<!-- Slide number: 2")
        assert md.index("![logo]") < md.index(second)
        assert md.rstrip().endswith(second)

    def test_llm_text_stays_free_of_screenshot_comments(self, tmp_path: Path) -> None:
        result = self._convert(tmp_path)

        assert result.metadata["extracted_text"] == self.TEXT
        assert "screenshots/" not in result.metadata["extracted_text"]


class TestAppendSlideScreenshotComments:
    def test_slide_without_a_marker_is_referenced_at_the_end(self) -> None:
        from markitai.converter.base import append_screenshot_comments
        from markitai.converter.office import _SLIDE_MARKER_RE

        md = "<!-- Slide number: 1 -->\nOne"
        shots = [{"page": 1, "name": "a.slide0001.jpg"}]
        shots.append({"page": 2, "name": "a.slide0002.jpg"})

        result = append_screenshot_comments(md, shots, _SLIDE_MARKER_RE, "Slide")

        assert result == (
            "<!-- Slide number: 1 -->\nOne\n\n"
            "<!-- ![Slide 1](.markitai/screenshots/a.slide0001.jpg) -->\n\n"
            "<!-- ![Slide 2](.markitai/screenshots/a.slide0002.jpg) -->"
        )

    def test_no_screenshots_leaves_markdown_unchanged(self) -> None:
        from markitai.converter.base import append_screenshot_comments
        from markitai.converter.office import _SLIDE_MARKER_RE

        assert (
            append_screenshot_comments("text\n", [], _SLIDE_MARKER_RE, "Slide")
            == "text\n"
        )


class TestOcrOnlyScreenshotCount:
    def test_local_ocr_slides_are_counted_as_screenshots(self, tmp_path: Path) -> None:
        """The OCR-only render keeps page_images out (NO_VLM_OCR privacy) but
        its slides must still be counted, as screenshots."""
        from markitai.workflow.core import ConversionContext, process_embedded_images

        converter = _make_pptx_converter(ocr=True, llm=False)
        converter._markitdown = MagicMock()  # type: ignore[reportAttributeAccessIssue]
        converter._markitdown.convert.return_value = MagicMock(
            markdown="# Slide text", title=None
        )
        _stub_render_slides(converter, slide_count=3)
        out = tmp_path / "out"
        out.mkdir()
        result = converter.convert(Path("deck.pptx"), out)
        assert "page_images" not in result.metadata

        ctx = ConversionContext(
            input_path=Path("deck.pptx"),
            output_dir=out,
            config=converter.config,  # type: ignore[arg-type]
        )
        ctx.conversion_result = result
        import asyncio

        asyncio.run(process_embedded_images(ctx))

        assert ctx.screenshots_count == 3
        assert ctx.embedded_images_count == 0


class TestScreenshotCommentsOnlyWithoutLlm:
    def test_llm_mode_leaves_the_text_for_the_vision_step(self, tmp_path: Path) -> None:
        converter = _make_pptx_converter(llm=True, screenshot=True)
        text = "<!-- Slide number: 1 -->\nOne"
        converter._convert_with_markitdown = Mock(  # type: ignore[method-assign]
            return_value=ConvertResult(markdown=text, metadata={})
        )
        _stub_render_slides(converter, slide_count=1)
        out = tmp_path / "out"
        out.mkdir()

        result = converter.convert(Path("deck.pptx"), out)

        # enhance_with_vision reads result.markdown and appends its own
        # page comments to .llm.md; a second copy must not ride along.
        assert result.markdown == text
        assert result.images == []
        assert len(result.metadata["page_images"]) == 1
