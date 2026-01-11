"""Tests for pipeline functionality (conflict resolution, image processing, etc.)."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

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


class TestPipelineServiceIntegration:
    """Tests for pipeline integration with extracted services."""

    @pytest.fixture
    def mock_image_processor(self):
        """Create mock ImageProcessingService."""
        processor = AsyncMock()
        processor.optimize_images_parallel = AsyncMock(return_value=([], {}))
        processor.update_markdown_references = Mock(return_value="# Test")
        processor.prepare_for_analysis = Mock(return_value=[])
        processor.shutdown = Mock()
        return processor

    @pytest.fixture
    def mock_llm_orchestrator(self):
        """Create mock LLMOrchestrator."""
        orchestrator = AsyncMock()
        orchestrator.warmup = AsyncMock()
        orchestrator.create_llm_tasks = AsyncMock(return_value=[])
        orchestrator.has_capability = Mock(return_value=False)
        return orchestrator

    @pytest.fixture
    def mock_output_manager(self):
        """Create mock OutputManager."""
        manager = AsyncMock()
        manager.write_output = AsyncMock(return_value=Path("output.md"))
        manager.resolve_conflict = Mock(side_effect=lambda p: p)
        return manager

    def test_pipeline_creates_default_services(self):
        """Pipeline creates default service instances when not injected."""
        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        settings = MarkitSettings()
        pipeline = ConversionPipeline(settings)

        # Services should be created
        assert pipeline._image_processor is not None
        assert pipeline._llm_orchestrator is not None
        assert pipeline._output_manager is not None

    def test_pipeline_accepts_injected_services(
        self,
        mock_image_processor,
        mock_llm_orchestrator,
        mock_output_manager,
    ):
        """Pipeline accepts injected service instances for testing."""
        from markit.config.settings import MarkitSettings
        from markit.core.pipeline import ConversionPipeline

        settings = MarkitSettings()
        pipeline = ConversionPipeline(
            settings=settings,
            image_processor=mock_image_processor,
            llm_orchestrator=mock_llm_orchestrator,
            output_manager=mock_output_manager,
        )

        assert pipeline._image_processor is mock_image_processor
        assert pipeline._llm_orchestrator is mock_llm_orchestrator
        assert pipeline._output_manager is mock_output_manager


class TestDocumentConversionResult:
    """Tests for DocumentConversionResult dataclass."""

    def test_success_when_no_error(self, temp_dir):
        """Success is True when no error and conversion succeeded."""
        from markit.converters.base import ConversionResult
        from markit.core.pipeline import DocumentConversionResult

        result = DocumentConversionResult(
            input_file=Path("test.pdf"),
            output_dir=temp_dir,
            conversion_result=ConversionResult(
                markdown="# Test",
                success=True,
            ),
            plan=Mock(),
        )

        assert result.success is True

    def test_failure_on_conversion_error(self, temp_dir):
        """Success is False when conversion failed."""
        from markit.converters.base import ConversionResult
        from markit.core.pipeline import DocumentConversionResult

        result = DocumentConversionResult(
            input_file=Path("test.pdf"),
            output_dir=temp_dir,
            conversion_result=ConversionResult(
                markdown="",
                success=False,
                error="Conversion failed",
            ),
            plan=Mock(),
        )

        assert result.success is False

    def test_failure_on_explicit_error(self, temp_dir):
        """Success is False when error is explicitly set."""
        from markit.converters.base import ConversionResult
        from markit.core.pipeline import DocumentConversionResult

        result = DocumentConversionResult(
            input_file=Path("test.pdf"),
            output_dir=temp_dir,
            conversion_result=ConversionResult(
                markdown="# Test",
                success=True,
            ),
            plan=Mock(),
            error="Post-processing failed",
        )

        assert result.success is False


class TestPipelineReExports:
    """Tests for backward compatibility re-exports."""

    def test_generate_image_filename_reexported(self):
        """_generate_image_filename is re-exported from pipeline."""
        from markit.core.pipeline import _generate_image_filename

        # Should be callable
        filename = _generate_image_filename(Path("test.pdf"), 1, "png")
        assert filename == "test.pdf.001.png"

    def test_sanitize_filename_reexported(self):
        """_sanitize_filename is re-exported from pipeline."""
        from markit.core.pipeline import _sanitize_filename

        # Should be callable
        sanitized = _sanitize_filename("file<>:name.png")
        assert "<" not in sanitized

    def test_processed_image_info_reexported(self):
        """ProcessedImageInfo is re-exported from pipeline."""
        from markit.core.pipeline import ProcessedImageInfo

        # Should be importable and usable
        info = ProcessedImageInfo(filename="test.png")
        assert info.filename == "test.png"
        assert info.analysis is None
