"""Tests for converters module."""

from pathlib import Path

import pytest


class TestBaseConverter:
    """Tests for BaseConverter."""

    def test_conversion_result_dataclass(self):
        """Test ConversionResult dataclass."""
        from markit.converters.base import ConversionResult

        result = ConversionResult(
            markdown="# Test\n\nContent",
            success=True,
        )

        assert result.markdown == "# Test\n\nContent"
        assert result.success is True
        assert result.images == []
        assert result.images_count == 0
        assert result.error is None

    def test_conversion_result_with_images(self):
        """Test ConversionResult with images."""
        from markit.converters.base import ConversionResult, ExtractedImage

        image = ExtractedImage(
            data=b"fake image data",
            format="png",
            filename="image.png",
            source_document=Path("test.docx"),
            position=0,
        )

        result = ConversionResult(
            markdown="# Test",
            images=[image],
            success=True,
        )

        assert result.images_count == 1
        assert result.images[0].format == "png"

    def test_extracted_image_dataclass(self):
        """Test ExtractedImage dataclass."""
        from markit.converters.base import ExtractedImage

        image = ExtractedImage(
            data=b"test data",
            format="jpeg",
            filename="photo.jpg",
            source_document=Path("doc.docx"),
            position=0,
            width=800,
            height=600,
        )

        assert image.format == "jpeg"
        assert image.filename == "photo.jpg"
        assert image.width == 800
        assert image.height == 600


class TestMarkItDownConverter:
    """Tests for MarkItDownConverter."""

    def test_supported_extensions(self):
        """Test supported extensions."""
        from markit.converters.markitdown import MarkItDownConverter

        converter = MarkItDownConverter()

        assert converter.supports(".docx")
        assert converter.supports(".pptx")
        assert converter.supports(".xlsx")
        assert converter.supports(".pdf")
        assert converter.supports(".txt")
        assert converter.supports(".html")
        assert not converter.supports(".doc")  # Legacy format
        assert not converter.supports(".unknown")

    def test_converter_name(self):
        """Test converter name."""
        from markit.converters.markitdown import MarkItDownConverter

        converter = MarkItDownConverter()

        assert converter.name == "markitdown"

    @pytest.mark.asyncio
    async def test_validate_nonexistent_file(self, temp_dir):
        """Test validation of non-existent file."""
        from markit.converters.markitdown import MarkItDownConverter

        converter = MarkItDownConverter()
        nonexistent = temp_dir / "nonexistent.docx"

        assert await converter.validate(nonexistent) is False

    @pytest.mark.asyncio
    async def test_validate_unsupported_format(self, temp_dir):
        """Test validation of unsupported format."""
        from markit.converters.markitdown import MarkItDownConverter

        converter = MarkItDownConverter()
        unsupported = temp_dir / "file.xyz"
        unsupported.touch()

        assert await converter.validate(unsupported) is False

    @pytest.mark.asyncio
    async def test_validate_valid_file(self, sample_text_file):
        """Test validation of valid file."""
        from markit.converters.markitdown import MarkItDownConverter

        converter = MarkItDownConverter()

        assert await converter.validate(sample_text_file) is True

    @pytest.mark.asyncio
    async def test_convert_text_file(self, sample_text_file):
        """Test converting a text file."""
        from markit.converters.markitdown import MarkItDownConverter

        converter = MarkItDownConverter()
        result = await converter.convert(sample_text_file)

        assert result.success is True
        assert "Sample Document" in result.markdown
        assert result.metadata["converter"] == "markitdown"


class TestFormatRouter:
    """Tests for FormatRouter."""

    def test_route_docx(self, temp_dir):
        """Test routing docx file."""
        from markit.core.router import FormatRouter

        router = FormatRouter()
        docx_file = temp_dir / "test.docx"
        docx_file.touch()

        plan = router.route(docx_file)

        assert plan.primary_converter.name == "markitdown"

    def test_route_pdf(self, temp_dir):
        """Test routing PDF file."""
        from markit.core.router import FormatRouter

        router = FormatRouter(pdf_engine="markitdown")
        pdf_file = temp_dir / "test.pdf"
        pdf_file.touch()

        plan = router.route(pdf_file)

        assert plan.primary_converter.name == "markitdown"

    def test_unsupported_extension(self, temp_dir):
        """Test routing unsupported file."""
        from markit.core.router import FormatRouter
        from markit.exceptions import ConverterNotFoundError

        router = FormatRouter()
        unknown_file = temp_dir / "test.xyz"
        unknown_file.touch()

        with pytest.raises(ConverterNotFoundError):
            router.route(unknown_file)

    def test_is_supported(self):
        """Test is_supported method."""
        from markit.core.router import FormatRouter

        router = FormatRouter()

        assert router.is_supported(".docx")
        assert router.is_supported(".DOCX")  # case insensitive
        assert router.is_supported(".pdf")
        assert not router.is_supported(".xyz")

    def test_get_supported_extensions(self):
        """Test get_supported_extensions method."""
        from markit.core.router import FormatRouter

        router = FormatRouter()
        extensions = router.get_supported_extensions()

        assert ".docx" in extensions
        assert ".pptx" in extensions
        assert ".pdf" in extensions
