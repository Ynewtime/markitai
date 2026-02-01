"""Tests for workflow/core.py module."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from markitai.config import MarkitaiConfig


def _can_create_symlink() -> bool:
    """Check if the current process can create symlinks."""
    if sys.platform != "win32":
        return True
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target = tmp_path / "target"
            target.mkdir()
            link = tmp_path / "link"
            link.symlink_to(target)
            return True
    except OSError:
        return False


requires_symlink = pytest.mark.skipif(
    not _can_create_symlink(),
    reason="Symlink creation requires elevated privileges on Windows",
)
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

    @requires_symlink
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

    @requires_symlink
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


class TestApplyAltTextUpdates:
    """Tests for apply_alt_text_updates function."""

    def test_returns_false_for_nonexistent_file(self, tmp_path: Path) -> None:
        """Test returns False when LLM file doesn't exist."""
        from markitai.workflow.core import apply_alt_text_updates

        nonexistent = tmp_path / "nonexistent.llm.md"
        result = apply_alt_text_updates(nonexistent, None)
        assert result is False

    def test_returns_false_for_none_analysis(self, tmp_path: Path) -> None:
        """Test returns False when image_analysis is None."""
        from markitai.workflow.core import apply_alt_text_updates

        llm_file = tmp_path / "test.llm.md"
        llm_file.write_text("# Test\n\n![](assets/image.png)")

        result = apply_alt_text_updates(llm_file, None)
        assert result is False

    def test_updates_alt_text(self, tmp_path: Path) -> None:
        """Test successfully updates alt text in file."""
        from dataclasses import dataclass

        from markitai.workflow.core import apply_alt_text_updates

        @dataclass
        class MockImageAnalysisResult:
            assets: list[dict]

        llm_file = tmp_path / "test.llm.md"
        original_content = "# Test\n\n![](assets/test_image.png)\n\nSome text."
        llm_file.write_text(original_content)

        analysis = MockImageAnalysisResult(
            assets=[
                {
                    "asset": str(tmp_path / "assets" / "test_image.png"),
                    "alt": "A test image",
                }
            ]
        )

        result = apply_alt_text_updates(llm_file, analysis)

        assert result is True
        updated_content = llm_file.read_text()
        assert "![A test image](assets/test_image.png)" in updated_content

    def test_no_update_when_no_match(self, tmp_path: Path) -> None:
        """Test returns False when no images match."""
        from dataclasses import dataclass

        from markitai.workflow.core import apply_alt_text_updates

        @dataclass
        class MockImageAnalysisResult:
            assets: list[dict]

        llm_file = tmp_path / "test.llm.md"
        original_content = "# Test\n\nNo images here."
        llm_file.write_text(original_content)

        analysis = MockImageAnalysisResult(
            assets=[
                {"asset": str(tmp_path / "assets" / "other.png"), "alt": "Other image"}
            ]
        )

        result = apply_alt_text_updates(llm_file, analysis)

        assert result is False
        # Content unchanged
        assert llm_file.read_text() == original_content

    def test_handles_empty_alt_text(self, tmp_path: Path) -> None:
        """Test skips assets with empty alt text."""
        from dataclasses import dataclass

        from markitai.workflow.core import apply_alt_text_updates

        @dataclass
        class MockImageAnalysisResult:
            assets: list[dict]

        llm_file = tmp_path / "test.llm.md"
        original_content = "# Test\n\n![](assets/image.png)"
        llm_file.write_text(original_content)

        analysis = MockImageAnalysisResult(
            assets=[{"asset": str(tmp_path / "assets" / "image.png"), "alt": ""}]
        )

        result = apply_alt_text_updates(llm_file, analysis)

        assert result is False

    def test_handles_multiple_images(self, tmp_path: Path) -> None:
        """Test updates multiple images correctly."""
        from dataclasses import dataclass

        from markitai.workflow.core import apply_alt_text_updates

        @dataclass
        class MockImageAnalysisResult:
            assets: list[dict]

        llm_file = tmp_path / "test.llm.md"
        original_content = (
            "# Test\n\n"
            "![](assets/image1.png)\n\n"
            "Some text\n\n"
            "![old alt](assets/image2.jpg)"
        )
        llm_file.write_text(original_content)

        analysis = MockImageAnalysisResult(
            assets=[
                {
                    "asset": str(tmp_path / "assets" / "image1.png"),
                    "alt": "First image",
                },
                {
                    "asset": str(tmp_path / "assets" / "image2.jpg"),
                    "alt": "Second image",
                },
            ]
        )

        result = apply_alt_text_updates(llm_file, analysis)

        assert result is True
        updated_content = llm_file.read_text()
        assert "![First image](assets/image1.png)" in updated_content
        assert "![Second image](assets/image2.jpg)" in updated_content


class TestRunInConverterThread:
    """Tests for run_in_converter_thread function."""

    @pytest.mark.asyncio
    async def test_runs_sync_function_in_thread(self) -> None:
        """Test that sync function is executed in thread pool."""
        from markitai.workflow.core import run_in_converter_thread

        def sync_func(x: int, y: int) -> int:
            return x + y

        result = await run_in_converter_thread(sync_func, 3, 5)
        assert result == 8

    @pytest.mark.asyncio
    async def test_handles_exception(self) -> None:
        """Test that exceptions are propagated."""
        from markitai.workflow.core import run_in_converter_thread

        def failing_func() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await run_in_converter_thread(failing_func)

    @pytest.mark.asyncio
    async def test_supports_kwargs(self) -> None:
        """Test that kwargs are passed correctly."""
        from markitai.workflow.core import run_in_converter_thread

        def func_with_kwargs(a: int, b: int = 10) -> int:
            return a * b

        result = await run_in_converter_thread(func_with_kwargs, 5, b=3)
        assert result == 15


# =============================================================================
# LLM Processing Tests (with mocks)
# =============================================================================


class TestProcessWithVisionLLM:
    """Tests for process_with_vision_llm async function."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock LLM processor."""
        from unittest.mock import AsyncMock, MagicMock

        processor = MagicMock()
        processor.process_document = AsyncMock(return_value=("cleaned", "frontmatter"))
        processor.format_llm_output = MagicMock(return_value="# LLM Content")
        processor.get_context_cost = MagicMock(return_value=0.05)
        processor.get_context_usage = MagicMock(return_value={"gpt-4": {"requests": 1}})
        return processor

    @pytest.mark.asyncio
    async def test_no_page_images_returns_early(
        self, sample_txt_path: Path, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """Test that function returns early when no page images."""
        from markitai.workflow.core import (
            ConversionContext,
            process_with_vision_llm,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=output_dir,
            config=default_config,
        )
        ctx.output_file = output_dir / "test.md"
        ctx.conversion_result = ConvertResult(markdown="# Test", metadata={})

        result = await process_with_vision_llm(ctx)

        assert result.success is True
        # No LLM processing should have occurred
        assert ctx.llm_cost == 0.0

    @pytest.mark.asyncio
    async def test_missing_conversion_result_fails(
        self, sample_txt_path: Path, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """Test that function fails when conversion_result is None."""
        from markitai.workflow.core import (
            ConversionContext,
            process_with_vision_llm,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=output_dir,
            config=default_config,
        )
        ctx.conversion_result = None

        result = await process_with_vision_llm(ctx)

        assert result.success is False
        assert "Missing conversion result" in result.error

    @pytest.mark.asyncio
    async def test_with_page_images_calls_workflow(
        self,
        sample_txt_path: Path,
        tmp_path: Path,
        default_config: MarkitaiConfig,
        mock_processor,
    ) -> None:
        """Test processing with page images uses SingleFileWorkflow."""
        from unittest.mock import AsyncMock, patch

        from markitai.workflow.core import (
            ConversionContext,
            process_with_vision_llm,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # Create test page images
        screenshots_dir = output_dir / "screenshots"
        screenshots_dir.mkdir()
        page1 = screenshots_dir / "page1.png"
        page1.write_bytes(b"fake image")

        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=output_dir,
            config=default_config,
            shared_processor=mock_processor,
        )
        ctx.output_file = output_dir / "test.md"
        ctx.output_file.touch()
        ctx.conversion_result = ConvertResult(
            markdown="# Test",
            metadata={
                "page_images": [{"page": 1, "name": "page1.png", "path": str(page1)}]
            },
        )

        # Mock the SingleFileWorkflow - patch at the module where it's imported
        with patch("markitai.workflow.single.SingleFileWorkflow") as MockWorkflow:
            mock_workflow_instance = MockWorkflow.return_value
            mock_workflow_instance.enhance_with_vision = AsyncMock(
                return_value=(
                    "# Enhanced",
                    "title: Test",
                    0.05,
                    {"gpt-4": {"requests": 1}},
                )
            )

            result = await process_with_vision_llm(ctx)

            assert result.success is True
            mock_workflow_instance.enhance_with_vision.assert_called_once()


class TestProcessWithStandardLLM:
    """Tests for process_with_standard_llm async function."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock LLM processor."""
        from unittest.mock import AsyncMock, MagicMock

        processor = MagicMock()
        processor.process_document = AsyncMock(return_value=("cleaned", "frontmatter"))
        processor.format_llm_output = MagicMock(return_value="# LLM Content")
        processor.get_context_cost = MagicMock(return_value=0.05)
        processor.get_context_usage = MagicMock(return_value={"gpt-4": {"requests": 1}})
        return processor

    @pytest.mark.asyncio
    async def test_missing_conversion_result_fails(
        self, sample_txt_path: Path, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """Test that function fails when conversion_result is None."""
        from markitai.workflow.core import (
            ConversionContext,
            process_with_standard_llm,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=output_dir,
            config=default_config,
        )
        ctx.conversion_result = None

        result = await process_with_standard_llm(ctx)

        assert result.success is False
        assert "Missing conversion result" in result.error

    @pytest.mark.asyncio
    async def test_standalone_image_processing(
        self,
        tmp_path: Path,
        default_config: MarkitaiConfig,
        mock_processor,
    ) -> None:
        """Test processing a standalone image file."""
        from unittest.mock import AsyncMock, patch

        from markitai.workflow.core import (
            ConversionContext,
            process_with_standard_llm,
        )

        # Create a fake image file
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"fake image")

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        assets_dir = output_dir / "assets"
        assets_dir.mkdir()

        # Create saved image in assets
        saved_image = assets_dir / "test.png.image1.png"
        saved_image.write_bytes(b"fake saved")

        ctx = ConversionContext(
            input_path=image_file,
            output_dir=output_dir,
            config=default_config,
            shared_processor=mock_processor,
        )
        ctx.output_file = output_dir / "test.png.md"
        ctx.output_file.touch()
        ctx.conversion_result = ConvertResult(
            markdown="![](assets/test.png.image1.png)",
            metadata={},
        )

        # Mock the SingleFileWorkflow - patch at the module where it's imported
        with patch("markitai.workflow.single.SingleFileWorkflow") as MockWorkflow:
            mock_workflow_instance = MockWorkflow.return_value
            mock_workflow_instance.analyze_images = AsyncMock(
                return_value=("![Updated](assets/test.png.image1.png)", 0.02, {}, None)
            )

            result = await process_with_standard_llm(ctx)

            assert result.success is True
            mock_workflow_instance.analyze_images.assert_called_once()


class TestAnalyzeEmbeddedImages:
    """Tests for analyze_embedded_images async function."""

    @pytest.mark.asyncio
    async def test_returns_success_when_disabled(
        self, sample_txt_path: Path, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """Test returns success when image analysis is disabled."""
        from markitai.workflow.core import (
            ConversionContext,
            analyze_embedded_images,
        )

        # Disable image analysis
        default_config.image.alt_enabled = False
        default_config.image.desc_enabled = False

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=output_dir,
            config=default_config,
        )
        ctx.output_file = output_dir / "test.md"
        ctx.conversion_result = ConvertResult(markdown="# Test", metadata={})

        result = await analyze_embedded_images(ctx)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_returns_success_when_no_images(
        self, sample_txt_path: Path, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """Test returns success when no images to analyze."""
        from markitai.workflow.core import (
            ConversionContext,
            analyze_embedded_images,
        )

        # Enable image analysis
        default_config.image.alt_enabled = True

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=output_dir,
            config=default_config,
        )
        ctx.output_file = output_dir / "test.md"
        ctx.conversion_result = ConvertResult(markdown="# Test", metadata={})

        result = await analyze_embedded_images(ctx)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_filters_page_screenshots(
        self, sample_txt_path: Path, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """Test that page screenshots are filtered out."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from markitai.workflow.core import (
            ConversionContext,
            analyze_embedded_images,
        )

        # Enable image analysis
        default_config.image.alt_enabled = True

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        assets_dir = output_dir / "assets"
        assets_dir.mkdir()

        # Create images - one page screenshot, one embedded
        page_screenshot = assets_dir / f"{sample_txt_path.name}.page1.png"
        page_screenshot.write_bytes(b"fake")
        embedded_image = assets_dir / f"{sample_txt_path.name}.figure1.png"
        embedded_image.write_bytes(b"fake")

        mock_processor = MagicMock()
        mock_processor.get_context_cost = MagicMock(return_value=0.01)
        mock_processor.get_context_usage = MagicMock(return_value={})

        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=output_dir,
            config=default_config,
            shared_processor=mock_processor,
        )
        ctx.output_file = output_dir / "test.md"
        ctx.conversion_result = ConvertResult(markdown="# Test", metadata={})

        # Mock the SingleFileWorkflow - patch at the module where it's imported
        with patch("markitai.workflow.single.SingleFileWorkflow") as MockWorkflow:
            mock_workflow_instance = MockWorkflow.return_value
            mock_workflow_instance.analyze_images = AsyncMock(
                return_value=("# Test", 0.01, {}, None)
            )

            result = await analyze_embedded_images(ctx)

            assert result.success is True
            # Should only analyze embedded image, not page screenshot
            if mock_workflow_instance.analyze_images.called:
                call_args = mock_workflow_instance.analyze_images.call_args
                analyzed_images = call_args[0][0]  # First positional arg
                # Should only contain non-page images
                assert all("page" not in p.name.lower() for p in analyzed_images)


class TestConversionContextEdgeCases:
    """Additional edge case tests for ConversionContext."""

    def test_llm_usage_is_mutable_dict(
        self, sample_txt_path: Path, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """Test that llm_usage can be modified."""
        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=tmp_path / "output",
            config=default_config,
        )

        # Should be able to modify the dict
        ctx.llm_usage["gpt-4"] = {"requests": 1, "cost_usd": 0.01}
        assert ctx.llm_usage["gpt-4"]["requests"] == 1

    def test_multiple_contexts_have_independent_dicts(
        self, sample_txt_path: Path, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """Test that different contexts don't share mutable state."""
        ctx1 = ConversionContext(
            input_path=sample_txt_path,
            output_dir=tmp_path / "output1",
            config=default_config,
        )
        ctx2 = ConversionContext(
            input_path=sample_txt_path,
            output_dir=tmp_path / "output2",
            config=default_config,
        )

        ctx1.llm_usage["model"] = {"requests": 5}
        assert "model" not in ctx2.llm_usage


class TestGetSavedImagesEdgeCases:
    """Additional tests for get_saved_images function."""

    def test_handles_special_characters_in_filename(
        self, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """Test handling of files with special glob characters in name."""
        from markitai.workflow.core import ConversionContext, get_saved_images

        # Create file with special characters that need escaping
        input_file = tmp_path / "test[1].txt"
        input_file.write_text("content")

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        assets_dir = output_dir / "assets"
        assets_dir.mkdir()

        # Create matching image
        (assets_dir / "test[1].txt.image1.png").touch()

        ctx = ConversionContext(
            input_path=input_file,
            output_dir=output_dir,
            config=default_config,
        )

        images = get_saved_images(ctx)
        assert len(images) == 1

    def test_case_insensitive_extension_matching(
        self, sample_context: ConversionContext
    ) -> None:
        """Test that image extension matching is case-insensitive."""
        from markitai.workflow.core import get_saved_images

        sample_context.output_dir.mkdir(parents=True)
        assets_dir = sample_context.output_dir / "assets"
        assets_dir.mkdir()

        # Create images with various case extensions
        (assets_dir / f"{sample_context.input_path.name}.image1.PNG").touch()
        (assets_dir / f"{sample_context.input_path.name}.image2.JpG").touch()
        (assets_dir / f"{sample_context.input_path.name}.image3.jpeg").touch()

        images = get_saved_images(sample_context)
        assert len(images) == 3


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

    @pytest.mark.asyncio
    async def test_with_llm_enabled_no_page_images(
        self, sample_txt_path: Path, tmp_path: Path
    ) -> None:
        """Test LLM processing when enabled but no page images (standard mode)."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from markitai.config import MarkitaiConfig
        from markitai.workflow.core import ConversionContext, convert_document_core

        config = MarkitaiConfig()
        config.llm.enabled = True

        output_dir = tmp_path / "output"
        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=output_dir,
            config=config,
        )

        # Mock the LLM processor
        mock_processor = MagicMock()
        mock_processor.process_document = AsyncMock(
            return_value=("cleaned", "title: Test")
        )
        mock_processor.format_llm_output = MagicMock(
            return_value="---\ntitle: Test\n---\n\n# Cleaned"
        )
        mock_processor.get_context_cost = MagicMock(return_value=0.01)
        mock_processor.get_context_usage = MagicMock(return_value={})

        # Patch at the modules where they're imported
        with (
            patch(
                "markitai.workflow.helpers.create_llm_processor",
                return_value=mock_processor,
            ),
            patch("markitai.workflow.single.SingleFileWorkflow") as MockWorkflow,
        ):
            mock_workflow = MockWorkflow.return_value
            mock_workflow.process_document_with_llm = AsyncMock(
                return_value=("# Cleaned", 0.01, {})
            )
            mock_workflow.analyze_images = AsyncMock(
                return_value=("# Cleaned", 0.0, {}, None)
            )

            max_size = 100 * 1024 * 1024
            result = await convert_document_core(ctx, max_size)

            assert result.success is True

    @pytest.mark.asyncio
    async def test_image_conversion(self, tmp_path: Path) -> None:
        """Test converting an image file (PNG)."""
        from markitai.config import MarkitaiConfig
        from markitai.workflow.core import ConversionContext, convert_document_core

        # Create a simple PNG image (1x1 white pixel)
        png_file = tmp_path / "test.png"
        # Minimal valid PNG file
        png_data = bytes(
            [
                0x89,
                0x50,
                0x4E,
                0x47,
                0x0D,
                0x0A,
                0x1A,
                0x0A,  # PNG signature
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x48,
                0x44,
                0x52,  # IHDR chunk
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,  # Width=1, Height=1
                0x08,
                0x02,
                0x00,
                0x00,
                0x00,
                0x90,
                0x77,
                0x53,  # Bit depth, color type
                0xDE,
                0x00,
                0x00,
                0x00,
                0x0C,
                0x49,
                0x44,
                0x41,  # IDAT chunk
                0x54,
                0x08,
                0xD7,
                0x63,
                0xF8,
                0xFF,
                0xFF,
                0xFF,  # Image data
                0x00,
                0x05,
                0xFE,
                0x02,
                0xFE,
                0xA3,
                0xAC,
                0xB4,
                0x1D,
                0x00,
                0x00,
                0x00,
                0x00,
                0x49,
                0x45,
                0x4E,  # IEND chunk
                0x44,
                0xAE,
                0x42,
                0x60,
                0x82,
            ]
        )
        png_file.write_bytes(png_data)

        config = MarkitaiConfig()
        output_dir = tmp_path / "output"

        ctx = ConversionContext(
            input_path=png_file,
            output_dir=output_dir,
            config=config,
        )

        max_size = 100 * 1024 * 1024
        result = await convert_document_core(ctx, max_size)

        assert result.success is True
        assert ctx.output_file is not None
        assert ctx.output_file.exists()
        # Output should contain image reference
        content = ctx.output_file.read_text()
        assert "test.png" in content
        assert "assets/test.png" in content  # Image copied to assets
