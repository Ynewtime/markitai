"""Tests for pipeline functionality (conflict resolution, image processing, etc.)."""

from pathlib import Path

import pytest


class TestOutputConflictResolution:
    """Tests for output file conflict resolution."""

    def test_default_conflict_strategy_is_rename(self):
        """Test that default conflict strategy is rename."""
        from markit.config.settings import MarkitSettings

        settings = MarkitSettings()
        assert settings.output.on_conflict == "rename"

    def test_resolve_conflict_no_existing_file(self, temp_dir):
        """Test resolution when no conflict exists."""
        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        settings = MarkitSettings()
        pipeline = ConversionPipeline(settings)

        output_path = temp_dir / "test.md"
        resolved = pipeline._resolve_conflict(output_path)

        assert resolved == output_path

    def test_resolve_conflict_rename_strategy(self, temp_dir):
        """Test rename strategy adds sequence number."""
        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        settings = MarkitSettings()
        settings.output.on_conflict = "rename"
        pipeline = ConversionPipeline(settings)

        # Create existing file
        output_path = temp_dir / "test.md"
        output_path.write_text("existing content")

        resolved = pipeline._resolve_conflict(output_path)

        assert resolved == temp_dir / "test_1.md"
        assert resolved != output_path

    def test_resolve_conflict_rename_multiple(self, temp_dir):
        """Test rename strategy with multiple conflicts."""
        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        settings = MarkitSettings()
        settings.output.on_conflict = "rename"
        pipeline = ConversionPipeline(settings)

        # Create existing files
        output_path = temp_dir / "test.md"
        output_path.write_text("existing")
        (temp_dir / "test_1.md").write_text("existing 1")
        (temp_dir / "test_2.md").write_text("existing 2")

        resolved = pipeline._resolve_conflict(output_path)

        assert resolved == temp_dir / "test_3.md"

    def test_resolve_conflict_overwrite_strategy(self, temp_dir):
        """Test overwrite strategy returns same path."""
        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        settings = MarkitSettings()
        settings.output.on_conflict = "overwrite"
        pipeline = ConversionPipeline(settings)

        output_path = temp_dir / "test.md"
        output_path.write_text("existing content")

        resolved = pipeline._resolve_conflict(output_path)

        assert resolved == output_path

    def test_resolve_conflict_skip_strategy_raises(self, temp_dir):
        """Test skip strategy raises ConversionError."""
        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline
        from markit.exceptions import ConversionError

        settings = MarkitSettings()
        settings.output.on_conflict = "skip"
        pipeline = ConversionPipeline(settings)

        output_path = temp_dir / "test.md"
        output_path.write_text("existing content")

        with pytest.raises(ConversionError) as exc_info:
            pipeline._resolve_conflict(output_path)

        assert "already exists" in str(exc_info.value)


class TestImageFiltering:
    """Tests for image filtering in PDF converter."""

    def test_filter_by_dimension(self):
        """Test that images below minimum dimension are filtered."""
        from markit.converters.pdf.pymupdf import PyMuPDFConverter

        converter = PyMuPDFConverter(
            min_image_dimension=50,
            min_image_area=100,
            min_image_size=100,
        )

        # Small dimension should be filtered
        assert converter._should_keep_image(b"x" * 1000, 30, 100) is False
        assert converter._should_keep_image(b"x" * 1000, 100, 30) is False

        # Large dimension should be kept
        assert converter._should_keep_image(b"x" * 1000, 100, 100) is True

    def test_filter_by_area(self):
        """Test that images below minimum area are filtered."""
        from markit.converters.pdf.pymupdf import PyMuPDFConverter

        converter = PyMuPDFConverter(
            min_image_dimension=10,
            min_image_area=2500,
            min_image_size=100,
        )

        # Small area (40x40=1600) should be filtered
        assert converter._should_keep_image(b"x" * 1000, 40, 40) is False

        # Large area (60x60=3600) should be kept
        assert converter._should_keep_image(b"x" * 1000, 60, 60) is True

    def test_filter_by_file_size(self):
        """Test that images below minimum file size are filtered."""
        from markit.converters.pdf.pymupdf import PyMuPDFConverter

        converter = PyMuPDFConverter(
            min_image_dimension=10,
            min_image_area=100,
            min_image_size=3072,  # 3KB
        )

        # Small file size should be filtered
        assert converter._should_keep_image(b"x" * 1000, 100, 100) is False

        # Large file size should be kept
        assert converter._should_keep_image(b"x" * 5000, 100, 100) is True

    def test_filter_disabled(self):
        """Test that filtering can be disabled."""
        from markit.converters.pdf.pymupdf import PyMuPDFConverter

        converter = PyMuPDFConverter(filter_small_images=False)

        # Even small images should be kept when filtering disabled
        # Note: _should_keep_image is still callable but won't be used
        assert converter.filter_small_images is False

    def test_default_filter_settings(self):
        """Test default filter settings from config."""
        from markit.config.settings import ImageConfig

        config = ImageConfig()

        assert config.filter_small_images is True
        assert config.min_dimension == 100
        assert config.min_area == 40000
        assert config.min_file_size == 10240


class TestImageExtraction:
    """Tests for image extraction from documents."""

    def test_data_uri_pattern(self):
        """Test data URI regex pattern."""
        from markit.converters.markitdown import DATA_URI_PATTERN

        # Valid data URI
        match = DATA_URI_PATTERN.search("![alt](data:image/png;base64,iVBORw0KGgo=)")
        assert match is not None
        assert match.group(1) == "alt"
        assert match.group(3) == "png"
        assert match.group(4) == "iVBORw0KGgo="

        # JPEG format
        match = DATA_URI_PATTERN.search("![](data:image/jpeg;base64,/9j/4AAQ=)")
        assert match is not None
        assert match.group(3) == "jpeg"

    def test_image_ref_pattern(self):
        """Test image reference regex pattern."""
        from markit.converters.markitdown import IMAGE_REF_PATTERN

        # Standard image reference
        match = IMAGE_REF_PATTERN.search("![alt text](path/to/image.png)")
        assert match is not None
        assert match.group(1) == "alt text"
        assert match.group(2) == "path/to/image.png"

        # Empty alt text
        match = IMAGE_REF_PATTERN.search("![](image.jpg)")
        assert match is not None
        assert match.group(1) == ""
        assert match.group(2) == "image.jpg"

    def test_extracted_image_filename_format(self):
        """Test that extracted images follow naming convention."""
        from markit.converters.base import ExtractedImage

        image = ExtractedImage(
            data=b"test",
            format="png",
            filename="document.docx.001.png",
            source_document=Path("document.docx"),
            position=1,
        )

        # Filename should follow pattern: {name}.{ext}.{seq:03d}.{image_ext}
        assert image.filename == "document.docx.001.png"
        assert ".001." in image.filename

    def test_generate_image_filename(self):
        """Test standardized image filename generation."""
        from markit.core.pipeline import _generate_image_filename

        # Test basic case
        filename = _generate_image_filename(Path("document.docx"), 1, "png")
        assert filename == "document.docx.001.png"

        # Test with different index
        filename = _generate_image_filename(Path("document.docx"), 42, "jpeg")
        assert filename == "document.docx.042.jpeg"

        # Test with .doc file (legacy format)
        filename = _generate_image_filename(Path("file-sample_100kB.doc"), 1, "jpeg")
        assert filename == "file-sample_100kB.doc.001.jpeg"

        # Test with PDF
        filename = _generate_image_filename(Path("report.pdf"), 3, "png")
        assert filename == "report.pdf.003.png"

        # Test filename sanitization (special characters)
        filename = _generate_image_filename(Path("file<with>special:chars.pdf"), 1, "png")
        assert "<" not in filename
        assert ">" not in filename
        assert ":" not in filename


class TestPipelineSettings:
    """Tests for pipeline settings and configuration."""

    def test_pipeline_uses_image_filter_settings(self):
        """Test that pipeline passes filter settings to router."""
        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        settings = MarkitSettings()
        settings.image.filter_small_images = True
        settings.image.min_dimension = 100
        settings.image.min_area = 5000
        settings.image.min_file_size = 4096

        pipeline = ConversionPipeline(settings)

        # Router should have these settings
        assert pipeline.router.filter_small_images is True
        assert pipeline.router.min_image_dimension == 100
        assert pipeline.router.min_image_area == 5000
        assert pipeline.router.min_image_size == 4096

    def test_pipeline_result_dataclass(self, temp_dir):
        """Test PipelineResult dataclass."""
        from markit.core.pipeline import PipelineResult

        result = PipelineResult(
            output_path=temp_dir / "output.md",
            markdown_content="# Test",
            images_count=5,
            metadata={"key": "value"},
            success=True,
        )

        assert result.output_path == temp_dir / "output.md"
        assert result.markdown_content == "# Test"
        assert result.images_count == 5
        assert result.metadata == {"key": "value"}
        assert result.success is True
        assert result.error is None

    def test_pipeline_result_with_error(self, temp_dir):
        """Test PipelineResult with error."""
        from markit.core.pipeline import PipelineResult

        result = PipelineResult(
            output_path=temp_dir / "output.md",
            markdown_content="",
            success=False,
            error="Conversion failed",
        )

        assert result.success is False
        assert result.error == "Conversion failed"
