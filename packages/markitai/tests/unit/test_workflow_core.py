"""Tests for workflow/core.py module."""

from __future__ import annotations

from pathlib import Path

import pytest

from markitai.config import MarkitaiConfig
from markitai.converter.base import ConvertResult
from markitai.workflow.core import (
    ConversionContext,
    ConversionStepResult,
    DocumentConversionError,
    FileSizeError,
    UnsupportedFormatError,
    prepare_output_directory,
    resolve_output_file,
    validate_and_detect_format,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> MarkitaiConfig:
    """Return a default MarkitaiConfig for testing."""
    return MarkitaiConfig()


@pytest.fixture
def sample_txt_path(tmp_path: Path) -> Path:
    """Create a sample text file and return its path."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("# Test Content\n\nSome text here.", encoding="utf-8")
    return txt_file


@pytest.fixture
def sample_context(
    sample_txt_path: Path, tmp_path: Path, default_config: MarkitaiConfig
) -> ConversionContext:
    """Return a sample ConversionContext for testing."""
    output_dir = tmp_path / "output"
    return ConversionContext(
        input_path=sample_txt_path,
        output_dir=output_dir,
        config=default_config,
    )


# =============================================================================
# ConversionContext Tests
# =============================================================================


class TestConversionContext:
    """Tests for ConversionContext dataclass."""

    def test_effective_input_default(self, sample_context: ConversionContext) -> None:
        """Test effective_input returns input_path when no actual_file."""
        assert sample_context.effective_input == sample_context.input_path

    def test_effective_input_with_actual_file(
        self, sample_context: ConversionContext, tmp_path: Path
    ) -> None:
        """Test effective_input returns actual_file when set."""
        actual = tmp_path / "converted.md"
        actual.touch()
        sample_context.actual_file = actual
        assert sample_context.effective_input == actual

    def test_is_preconverted_false(self, sample_context: ConversionContext) -> None:
        """Test is_preconverted returns False when no actual_file."""
        assert sample_context.is_preconverted is False

    def test_is_preconverted_true(
        self, sample_context: ConversionContext, tmp_path: Path
    ) -> None:
        """Test is_preconverted returns True when actual_file differs."""
        actual = tmp_path / "converted.md"
        actual.touch()
        sample_context.actual_file = actual
        assert sample_context.is_preconverted is True

    def test_is_preconverted_same_file(self, sample_context: ConversionContext) -> None:
        """Test is_preconverted returns False when actual_file same as input."""
        sample_context.actual_file = sample_context.input_path
        assert sample_context.is_preconverted is False

    def test_default_values(self, sample_context: ConversionContext) -> None:
        """Test default values are set correctly."""
        assert sample_context.use_multiprocess_images is False
        assert sample_context.converter is None
        assert sample_context.conversion_result is None
        assert sample_context.output_file is None
        assert sample_context.embedded_images_count == 0
        assert sample_context.screenshots_count == 0
        assert sample_context.llm_cost == 0.0
        assert sample_context.llm_usage == {}
        assert sample_context.image_analysis is None
        # New fields
        assert sample_context.duration == 0.0
        assert sample_context.cache_hit is False
        assert sample_context.input_base_dir is None
        assert sample_context.on_stage_complete is None

    def test_on_stage_complete_callback(
        self, sample_txt_path: Path, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """Test on_stage_complete callback can be set and called."""
        callback_calls: list[tuple[str, float]] = []

        def callback(stage: str, duration: float) -> None:
            callback_calls.append((stage, duration))

        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=tmp_path / "output",
            config=default_config,
            on_stage_complete=callback,
        )

        # Verify callback is set
        assert ctx.on_stage_complete is not None

        # Invoke callback
        ctx.on_stage_complete("test_stage", 1.5)
        assert callback_calls == [("test_stage", 1.5)]


class TestConversionStepResult:
    """Tests for ConversionStepResult dataclass."""

    def test_success_result(self) -> None:
        """Test creating a success result."""
        result = ConversionStepResult(success=True)
        assert result.success is True
        assert result.error is None
        assert result.skip_reason is None

    def test_error_result(self) -> None:
        """Test creating an error result."""
        result = ConversionStepResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.skip_reason is None

    def test_skip_result(self) -> None:
        """Test creating a skip result."""
        result = ConversionStepResult(success=True, skip_reason="exists")
        assert result.success is True
        assert result.error is None
        assert result.skip_reason == "exists"


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for custom exceptions."""

    def test_document_conversion_error(self) -> None:
        """Test DocumentConversionError."""
        error = DocumentConversionError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_unsupported_format_error(self) -> None:
        """Test UnsupportedFormatError."""
        error = UnsupportedFormatError("Unknown format")
        assert str(error) == "Unknown format"
        assert isinstance(error, DocumentConversionError)

    def test_file_size_error(self) -> None:
        """Test FileSizeError."""
        error = FileSizeError("File too large")
        assert str(error) == "File too large"
        assert isinstance(error, DocumentConversionError)


# =============================================================================
# validate_and_detect_format Tests
# =============================================================================


class TestValidateAndDetectFormat:
    """Tests for validate_and_detect_format function."""

    def test_success_txt_file(self, sample_context: ConversionContext) -> None:
        """Test successful validation of a TXT file."""
        max_size = 100 * 1024 * 1024  # 100MB
        result = validate_and_detect_format(sample_context, max_size)

        assert result.success is True
        assert result.error is None
        assert sample_context.converter is not None

    def test_file_too_large(
        self, sample_context: ConversionContext, tmp_path: Path
    ) -> None:
        """Test validation fails when file exceeds max size."""
        # Create a file larger than max_size
        large_file = tmp_path / "large.txt"
        large_file.write_bytes(b"x" * 1000)
        sample_context.input_path = large_file

        max_size = 100  # 100 bytes
        result = validate_and_detect_format(sample_context, max_size)

        assert result.success is False
        assert result.error is not None
        assert "too large" in result.error.lower()

    def test_unknown_format(
        self, sample_context: ConversionContext, tmp_path: Path
    ) -> None:
        """Test validation fails for unknown file format."""
        unknown_file = tmp_path / "file.xyz"
        unknown_file.write_text("content")
        sample_context.input_path = unknown_file

        max_size = 100 * 1024 * 1024
        result = validate_and_detect_format(sample_context, max_size)

        assert result.success is False
        assert result.error is not None
        assert "unsupported" in result.error.lower()

    def test_uses_effective_input_for_format_detection(
        self, sample_context: ConversionContext, tmp_path: Path
    ) -> None:
        """Test that format detection uses effective_input (actual_file if set)."""
        # Create a markdown file as the actual (pre-converted) file
        md_file = tmp_path / "converted.md"
        md_file.write_text("# Converted content")
        sample_context.actual_file = md_file

        max_size = 100 * 1024 * 1024
        result = validate_and_detect_format(sample_context, max_size)

        assert result.success is True
        # Converter should be for markdown, not txt
        from markitai.converter.text import MarkdownConverter

        assert isinstance(sample_context.converter, MarkdownConverter)


# =============================================================================
# prepare_output_directory Tests
# =============================================================================


class TestPrepareOutputDirectory:
    """Tests for prepare_output_directory function."""

    def test_creates_output_dir(self, sample_context: ConversionContext) -> None:
        """Test that output directory is created."""
        assert not sample_context.output_dir.exists()

        result = prepare_output_directory(sample_context)

        assert result.success is True
        assert sample_context.output_dir.exists()

    def test_existing_dir_ok(self, sample_context: ConversionContext) -> None:
        """Test that existing directory is handled."""
        sample_context.output_dir.mkdir(parents=True)

        result = prepare_output_directory(sample_context)

        assert result.success is True
        assert sample_context.output_dir.exists()

    def test_symlink_disallowed(
        self, sample_context: ConversionContext, tmp_path: Path
    ) -> None:
        """Test that symlink is rejected when disallowed."""
        # Create a real directory
        real_dir = tmp_path / "real_output"
        real_dir.mkdir()

        # Create a symlink to it
        symlink_dir = tmp_path / "symlink_output"
        symlink_dir.symlink_to(real_dir)

        # Configure to disallow symlinks
        sample_context.output_dir = symlink_dir
        sample_context.config.output.allow_symlinks = False

        result = prepare_output_directory(sample_context)

        assert result.success is False
        assert result.error is not None

    def test_symlink_allowed(
        self, sample_context: ConversionContext, tmp_path: Path
    ) -> None:
        """Test that symlink is accepted when allowed."""
        # Create a real directory
        real_dir = tmp_path / "real_output"
        real_dir.mkdir()

        # Create a symlink to it
        symlink_dir = tmp_path / "symlink_output"
        symlink_dir.symlink_to(real_dir)

        # Configure to allow symlinks
        sample_context.output_dir = symlink_dir
        sample_context.config.output.allow_symlinks = True

        result = prepare_output_directory(sample_context)

        assert result.success is True


# =============================================================================
# resolve_output_file Tests
# =============================================================================


class TestResolveOutputFile:
    """Tests for resolve_output_file function."""

    def test_new_file_path(self, sample_context: ConversionContext) -> None:
        """Test resolving output path for new file."""
        sample_context.output_dir.mkdir(parents=True)

        result = resolve_output_file(sample_context)

        assert result.success is True
        assert result.skip_reason is None
        assert sample_context.output_file is not None
        assert sample_context.output_file.suffix == ".md"

    def test_skip_when_exists(self, sample_context: ConversionContext) -> None:
        """Test skip when output file exists and on_conflict=skip."""
        sample_context.output_dir.mkdir(parents=True)
        sample_context.config.output.on_conflict = "skip"

        # Create existing output file
        expected_output = (
            sample_context.output_dir / f"{sample_context.input_path.name}.md"
        )
        expected_output.touch()

        result = resolve_output_file(sample_context)

        assert result.success is True
        assert result.skip_reason == "exists"
        assert sample_context.output_file is None

    def test_overwrite_when_exists(self, sample_context: ConversionContext) -> None:
        """Test overwrite when output file exists and on_conflict=overwrite."""
        sample_context.output_dir.mkdir(parents=True)
        sample_context.config.output.on_conflict = "overwrite"

        # Create existing output file
        expected_output = (
            sample_context.output_dir / f"{sample_context.input_path.name}.md"
        )
        expected_output.touch()

        result = resolve_output_file(sample_context)

        assert result.success is True
        assert result.skip_reason is None
        assert sample_context.output_file is not None
        assert sample_context.output_file == expected_output

    def test_rename_when_exists(self, sample_context: ConversionContext) -> None:
        """Test rename when output file exists and on_conflict=rename."""
        sample_context.output_dir.mkdir(parents=True)
        sample_context.config.output.on_conflict = "rename"

        # Create existing output file
        expected_output = (
            sample_context.output_dir / f"{sample_context.input_path.name}.md"
        )
        expected_output.touch()

        result = resolve_output_file(sample_context)

        assert result.success is True
        assert result.skip_reason is None
        assert sample_context.output_file is not None
        # Should have a different name
        assert sample_context.output_file != expected_output
        assert sample_context.output_file.name.endswith(".md")


# =============================================================================
# Integration-style Tests (without actual async)
# =============================================================================


class TestGetSavedImages:
    """Tests for get_saved_images function."""

    def test_no_assets_dir(self, sample_context: ConversionContext) -> None:
        """Test returns empty list when assets dir doesn't exist."""
        from markitai.workflow.core import get_saved_images

        sample_context.output_dir.mkdir(parents=True)
        # No assets subdirectory

        images = get_saved_images(sample_context)
        assert images == []

    def test_finds_images(self, sample_context: ConversionContext) -> None:
        """Test finds images matching input file name."""
        from markitai.workflow.core import get_saved_images

        sample_context.output_dir.mkdir(parents=True)
        assets_dir = sample_context.output_dir / "assets"
        assets_dir.mkdir()

        # Create test images
        (assets_dir / f"{sample_context.input_path.name}.image1.png").touch()
        (assets_dir / f"{sample_context.input_path.name}.image2.jpg").touch()
        (assets_dir / "other_file.png").touch()  # Should not match

        images = get_saved_images(sample_context)

        assert len(images) == 2
        assert all(sample_context.input_path.name in p.name for p in images)

    def test_filters_non_image_files(self, sample_context: ConversionContext) -> None:
        """Test filters out non-image files."""
        from markitai.workflow.core import get_saved_images

        sample_context.output_dir.mkdir(parents=True)
        assets_dir = sample_context.output_dir / "assets"
        assets_dir.mkdir()

        # Create mixed files
        (assets_dir / f"{sample_context.input_path.name}.image1.png").touch()
        (
            assets_dir / f"{sample_context.input_path.name}.data.json"
        ).touch()  # Not image

        images = get_saved_images(sample_context)

        assert len(images) == 1
        assert images[0].suffix == ".png"


class TestWriteBaseMarkdown:
    """Tests for write_base_markdown function."""

    def test_writes_markdown_with_frontmatter(
        self, sample_context: ConversionContext
    ) -> None:
        """Test writes markdown file with basic frontmatter."""
        from markitai.workflow.core import write_base_markdown

        sample_context.output_dir.mkdir(parents=True)
        sample_context.output_file = sample_context.output_dir / "output.md"
        sample_context.conversion_result = ConvertResult(
            markdown="# Test Document\n\nContent here.",
            metadata={"format": "TXT"},
        )

        result = write_base_markdown(sample_context)

        assert result.success is True
        assert sample_context.output_file.exists()

        content = sample_context.output_file.read_text()
        assert "---" in content  # Frontmatter delimiters
        assert "# Test Document" in content

    def test_fails_without_conversion_result(
        self, sample_context: ConversionContext
    ) -> None:
        """Test fails when no conversion result."""
        from markitai.workflow.core import write_base_markdown

        sample_context.output_dir.mkdir(parents=True)
        sample_context.output_file = sample_context.output_dir / "output.md"
        sample_context.conversion_result = None

        result = write_base_markdown(sample_context)

        assert result.success is False
        assert result.error is not None

    def test_fails_without_output_file(self, sample_context: ConversionContext) -> None:
        """Test fails when no output file path."""
        from markitai.workflow.core import write_base_markdown

        sample_context.output_dir.mkdir(parents=True)
        sample_context.conversion_result = ConvertResult(markdown="content")
        sample_context.output_file = None

        result = write_base_markdown(sample_context)

        assert result.success is False
        assert result.error is not None


# =============================================================================
# Async Function Tests
# =============================================================================


class TestConvertDocument:
    """Tests for convert_document async function."""

    @pytest.mark.asyncio
    async def test_success(self, sample_context: ConversionContext) -> None:
        """Test successful document conversion."""
        from markitai.converter.text import TxtConverter
        from markitai.workflow.core import convert_document

        sample_context.converter = TxtConverter()

        result = await convert_document(sample_context)

        assert result.success is True
        assert sample_context.conversion_result is not None
        assert sample_context.conversion_result.markdown is not None

    @pytest.mark.asyncio
    async def test_failure_no_converter(
        self, sample_context: ConversionContext
    ) -> None:
        """Test failure when converter is not set."""
        from markitai.workflow.core import convert_document

        sample_context.converter = None

        result = await convert_document(sample_context)

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_converter_exception(self, sample_context: ConversionContext) -> None:
        """Test handling of converter exception."""
        from unittest.mock import MagicMock

        from markitai.workflow.core import convert_document

        mock_converter = MagicMock()
        mock_converter.convert.side_effect = RuntimeError("Conversion failed")
        sample_context.converter = mock_converter

        result = await convert_document(sample_context)

        assert result.success is False
        assert result.error is not None
        assert "failed" in result.error.lower()


class TestProcessEmbeddedImages:
    """Tests for process_embedded_images async function."""

    @pytest.mark.asyncio
    async def test_no_images(self, sample_context: ConversionContext) -> None:
        """Test processing with no embedded images."""
        from markitai.workflow.core import process_embedded_images

        sample_context.conversion_result = ConvertResult(
            markdown="# Test\n\nNo images here.",
            metadata={},
        )

        result = await process_embedded_images(sample_context)

        assert result.success is True
        assert sample_context.embedded_images_count == 0

    @pytest.mark.asyncio
    async def test_no_conversion_result(
        self, sample_context: ConversionContext
    ) -> None:
        """Test failure when no conversion result."""
        from markitai.workflow.core import process_embedded_images

        sample_context.conversion_result = None

        result = await process_embedded_images(sample_context)

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_with_page_images_metadata(
        self, sample_context: ConversionContext
    ) -> None:
        """Test counting page images from metadata."""
        from markitai.workflow.core import process_embedded_images

        sample_context.conversion_result = ConvertResult(
            markdown="# Test",
            metadata={
                "page_images": [
                    {"page": 1, "name": "page1.png"},
                    {"page": 2, "name": "page2.png"},
                ]
            },
        )

        result = await process_embedded_images(sample_context)

        assert result.success is True
        assert sample_context.screenshots_count == 2


class TestConvertDocumentCore:
    """Tests for convert_document_core async function."""

    @pytest.mark.asyncio
    async def test_full_pipeline_success(
        self, sample_txt_path: Path, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """Test full conversion pipeline."""
        from markitai.workflow.core import ConversionContext, convert_document_core

        output_dir = tmp_path / "output"
        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=output_dir,
            config=default_config,
        )

        max_size = 100 * 1024 * 1024  # 100MB
        result = await convert_document_core(ctx, max_size)

        assert result.success is True
        assert ctx.conversion_result is not None
        assert ctx.output_file is not None
        assert ctx.output_file.exists()

    @pytest.mark.asyncio
    async def test_skips_existing_file(
        self, sample_txt_path: Path, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """Test pipeline skips existing output file when on_conflict=skip."""
        from markitai.workflow.core import ConversionContext, convert_document_core

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        default_config.output.on_conflict = "skip"

        # Create existing output file
        existing_file = output_dir / f"{sample_txt_path.name}.md"
        existing_file.write_text("existing content")

        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=output_dir,
            config=default_config,
        )

        max_size = 100 * 1024 * 1024
        result = await convert_document_core(ctx, max_size)

        assert result.success is True
        assert result.skip_reason == "exists"
        # Original content should be unchanged
        assert existing_file.read_text() == "existing content"

    @pytest.mark.asyncio
    async def test_fails_for_unsupported_format(
        self, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """Test pipeline fails for unsupported format."""
        from markitai.workflow.core import ConversionContext, convert_document_core

        # Create file with unsupported extension
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("content")

        output_dir = tmp_path / "output"
        ctx = ConversionContext(
            input_path=unsupported_file,
            output_dir=output_dir,
            config=default_config,
        )

        max_size = 100 * 1024 * 1024
        result = await convert_document_core(ctx, max_size)

        assert result.success is False
        assert result.error is not None
        assert "unsupported" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fails_for_large_file(
        self, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """Test pipeline fails for file exceeding size limit."""
        from markitai.workflow.core import ConversionContext, convert_document_core

        # Create large file
        large_file = tmp_path / "large.txt"
        large_file.write_bytes(b"x" * 2000)

        output_dir = tmp_path / "output"
        ctx = ConversionContext(
            input_path=large_file,
            output_dir=output_dir,
            config=default_config,
        )

        max_size = 1000  # 1KB limit
        result = await convert_document_core(ctx, max_size)

        assert result.success is False
        assert result.error is not None
        assert "too large" in result.error.lower()

    @pytest.mark.asyncio
    async def test_uses_preconverted_file(
        self, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """Test pipeline uses preconverted file when provided."""
        from markitai.workflow.core import ConversionContext, convert_document_core

        # Original file (docx - would need real converter)
        original_file = tmp_path / "test.docx"
        original_file.touch()

        # Pre-converted markdown file
        preconverted = tmp_path / "test_converted.md"
        preconverted.write_text("# Pre-converted Content\n\nAlready converted.")

        output_dir = tmp_path / "output"
        ctx = ConversionContext(
            input_path=original_file,
            output_dir=output_dir,
            config=default_config,
            actual_file=preconverted,
        )

        max_size = 100 * 1024 * 1024
        result = await convert_document_core(ctx, max_size)

        assert result.success is True
        assert ctx.output_file is not None
        assert ctx.output_file.exists()

        # Content should come from pre-converted file
        output_content = ctx.output_file.read_text()
        assert "Pre-converted Content" in output_content

    @pytest.mark.asyncio
    async def test_llm_disabled_by_default(
        self, sample_txt_path: Path, tmp_path: Path
    ) -> None:
        """Test LLM processing is skipped when disabled (default)."""
        from markitai.config import MarkitaiConfig
        from markitai.workflow.core import ConversionContext, convert_document_core

        config = MarkitaiConfig()
        assert config.llm.enabled is False  # Verify default

        output_dir = tmp_path / "output"
        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=output_dir,
            config=config,
        )

        max_size = 100 * 1024 * 1024
        result = await convert_document_core(ctx, max_size)

        assert result.success is True
        assert ctx.llm_cost == 0.0
        assert ctx.llm_usage == {}

        # Should only have base .md file, no .llm.md
        output_files = list(output_dir.glob("*.md"))
        assert len(output_files) == 1
        assert not any(f.name.endswith(".llm.md") for f in output_files)
