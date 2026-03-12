"""Unit tests for PPTX converter — OCR/Vision slide rendering paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock

from markitai.config import (
    LLMConfig,
    MarkitaiConfig,
    OCRConfig,
    ScreenshotConfig,
)
from markitai.converter.base import ExtractedImage
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
        converter._markitdown = MagicMock()
        converter._markitdown.convert.return_value = mock_markitdown_result

        render_mock = _stub_render_slides(converter, slide_count=2)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(Path("test.pptx"), output_dir)

        # The render function MUST have been called
        render_mock.assert_called_once()
        # Result must contain the slide images
        assert len(result.images) == 2
        assert result.metadata.get("ocr_used") is True

    def test_ocr_only_renders_slides_when_screenshot_enabled(
        self, tmp_path: Path
    ) -> None:
        """Sanity: slide images are also generated when screenshot.enabled=True."""
        converter = _make_pptx_converter(ocr=True, llm=False, screenshot=True)

        mock_markitdown_result = MagicMock()
        mock_markitdown_result.markdown = "# Slide text"
        mock_markitdown_result.title = None
        converter._markitdown = MagicMock()
        converter._markitdown.convert.return_value = mock_markitdown_result

        render_mock = _stub_render_slides(converter, slide_count=2)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(Path("test.pptx"), output_dir)

        render_mock.assert_called_once()
        assert len(result.images) == 2

    def test_ocr_only_includes_commented_image_refs_in_markdown(
        self, tmp_path: Path
    ) -> None:
        """OCR-only markdown should contain commented slide image references."""
        converter = _make_pptx_converter(ocr=True, llm=False, screenshot=False)

        mock_markitdown_result = MagicMock()
        mock_markitdown_result.markdown = "# Slide text"
        mock_markitdown_result.title = None
        converter._markitdown = MagicMock()
        converter._markitdown.convert.return_value = mock_markitdown_result

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
        converter._markitdown = MagicMock()
        converter._markitdown.convert.return_value = mock_markitdown_result

        render_mock = _stub_render_slides(converter, slide_count=1)

        result = converter.convert(Path("test.pptx"), None)

        # Even without output_dir, rendering must happen
        render_mock.assert_called_once()
        assert len(result.images) == 1


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
        converter._markitdown = MagicMock()
        converter._markitdown.convert.return_value = mock_markitdown_result

        render_mock = _stub_render_slides(converter, slide_count=3)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(Path("test.pptx"), output_dir)

        # The render function MUST have been called
        render_mock.assert_called_once()
        # Result must contain the slide images
        assert len(result.images) == 3
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
        converter._markitdown = MagicMock()
        converter._markitdown.convert.return_value = mock_markitdown_result

        render_mock = _stub_render_slides(converter, slide_count=3)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = converter.convert(Path("test.pptx"), output_dir)

        render_mock.assert_called_once()
        assert len(result.images) == 3
        assert len(result.metadata.get("page_images", [])) == 3

    def test_ocr_llm_metadata_contains_extracted_text(self, tmp_path: Path) -> None:
        """Metadata should contain extracted_text for LLM to combine with images."""
        converter = _make_pptx_converter(ocr=True, llm=True, screenshot=False)

        mock_markitdown_result = MagicMock()
        mock_markitdown_result.markdown = "# Extracted text from slides"
        mock_markitdown_result.title = None
        converter._markitdown = MagicMock()
        converter._markitdown.convert.return_value = mock_markitdown_result

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
        converter._markitdown = MagicMock()
        converter._markitdown.convert.return_value = mock_markitdown_result

        render_mock = _stub_render_slides(converter, slide_count=1)

        result = converter.convert(Path("test.pptx"), None)

        render_mock.assert_called_once()
        assert len(result.images) == 1
        assert len(result.metadata.get("page_images", [])) == 1


class TestPptxDefaultModeScreenshotIndependence:
    """Default mode (no --ocr) should NOT render slides unless --screenshot is on."""

    def test_default_mode_no_screenshots_when_disabled(self, tmp_path: Path) -> None:
        """Without --ocr and --screenshot, no slide rendering should happen."""
        converter = _make_pptx_converter(ocr=False, llm=False, screenshot=False)

        mock_markitdown_result = MagicMock()
        mock_markitdown_result.markdown = "# Just text"
        mock_markitdown_result.title = None
        converter._markitdown = MagicMock()
        converter._markitdown.convert.return_value = mock_markitdown_result

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
                "markitai.converter.office.ImageProcessor",
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
            images, slide_images = converter._render_slides_via_pdf(
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
