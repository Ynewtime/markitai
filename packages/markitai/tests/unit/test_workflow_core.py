"""Tests for workflow/core.py module."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

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
        assets_dir = sample_context.output_dir / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)

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
        assets_dir = sample_context.output_dir / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)

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

    def test_normalizes_markdown_before_frontmatter(
        self, sample_context: ConversionContext
    ) -> None:
        """Base markdown output should run shared cleanup before writing."""
        from markitai.workflow.core import write_base_markdown

        sample_context.output_dir.mkdir(parents=True)
        sample_context.output_file = sample_context.output_dir / "output.md"
        sample_context.conversion_result = ConvertResult(
            markdown="[Title\n\nDescription](/docs)\n\n__MARKITAI_FILE_ASSET__",
            metadata={"format": "TXT"},
        )

        result = write_base_markdown(sample_context)

        assert result.success is True
        content = sample_context.output_file.read_text()
        assert "[Title](/docs)" in content
        assert "__MARKITAI" not in content


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
        original_content = (
            "# Test\n\n![](.markitai/assets/test_image.png)\n\nSome text."
        )
        llm_file.write_text(original_content)

        analysis = MockImageAnalysisResult(
            assets=[
                {
                    "asset": str(tmp_path / ".markitai" / "assets" / "test_image.png"),
                    "alt": "A test image",
                }
            ]
        )

        result = apply_alt_text_updates(llm_file, analysis)

        assert result is True
        updated_content = llm_file.read_text()
        assert "![A test image](.markitai/assets/test_image.png)" in updated_content

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
                {
                    "asset": str(tmp_path / ".markitai" / "assets" / "other.png"),
                    "alt": "Other image",
                }
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
        original_content = "# Test\n\n![](.markitai/assets/image.png)"
        llm_file.write_text(original_content)

        analysis = MockImageAnalysisResult(
            assets=[
                {
                    "asset": str(tmp_path / ".markitai" / "assets" / "image.png"),
                    "alt": "",
                }
            ]
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
            "![](.markitai/assets/image1.png)\n\n"
            "Some text\n\n"
            "![old alt](.markitai/assets/image2.jpg)"
        )
        llm_file.write_text(original_content)

        analysis = MockImageAnalysisResult(
            assets=[
                {
                    "asset": str(tmp_path / ".markitai" / "assets" / "image1.png"),
                    "alt": "First image",
                },
                {
                    "asset": str(tmp_path / ".markitai" / "assets" / "image2.jpg"),
                    "alt": "Second image",
                },
            ]
        )

        result = apply_alt_text_updates(llm_file, analysis)

        assert result is True
        updated_content = llm_file.read_text()
        assert "![First image](.markitai/assets/image1.png)" in updated_content
        assert "![Second image](.markitai/assets/image2.jpg)" in updated_content


class TestAltTextSurvivesStabilization:
    """Test that alt text updates survive markdown stabilization.

    Regression test for a bug where stabilize_written_llm_output() was called
    BEFORE apply_alt_text_updates(), causing alt text to be lost when the
    stabilizer rewrote the .llm.md file from a baseline without alt text.
    """

    def test_alt_text_survives_stabilization(self, tmp_path: Path) -> None:
        """Alt text should be present after both stabilize and alt text update."""
        from dataclasses import dataclass
        from unittest.mock import MagicMock

        from markitai.workflow.core import (
            apply_alt_text_updates,
            stabilize_written_llm_output,
        )

        # Set up baseline .md file (no alt text — simulates converter output)
        base_md = tmp_path / "doc.md"
        base_md.write_text(
            "# Report\n\n![](.markitai/assets/chart.png)\n\nSome text.\n"
        )

        # Set up .llm.md file (LLM-generated, also without alt text initially)
        llm_md = tmp_path / "doc.llm.md"
        llm_md.write_text(
            "---\ntitle: Report\n---\n\n# Report\n\n"
            "![](.markitai/assets/chart.png)\n\nSome text.\n"
        )

        # Mock context for stabilize
        ctx = MagicMock()
        ctx.output_file = base_md
        ctx.input_path.name = "doc.pdf"
        ctx.conversion_result.markdown = base_md.read_text()

        # Mock processor without _stabilize_paged_markdown (no-op stabilize)
        processor = MagicMock(spec=[])  # empty spec prevents auto-attributes

        # Run stabilize first (should be no-op since content matches baseline)
        stabilize_written_llm_output(ctx, processor)

        # Now apply alt text updates
        @dataclass
        class MockAnalysis:
            assets: list[dict]

        analysis = MockAnalysis(
            assets=[
                {
                    "asset": str(tmp_path / ".markitai" / "assets" / "chart.png"),
                    "alt": "Bar chart showing quarterly revenue",
                }
            ]
        )
        result = apply_alt_text_updates(llm_md, analysis)

        assert result is True
        final_content = llm_md.read_text()
        assert (
            "![Bar chart showing quarterly revenue](.markitai/assets/chart.png)"
            in final_content
        )
        # Empty alt text should NOT remain
        assert "![](.markitai/assets/chart.png)" not in final_content


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
        processor._stabilize_paged_markdown = MagicMock(
            side_effect=lambda _original, cleaned, _source: cleaned
        )
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
        screenshots_dir = output_dir / ".markitai" / "screenshots"
        screenshots_dir.mkdir(parents=True)
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

    @pytest.mark.asyncio
    async def test_passes_converter_title_to_vision_workflow(
        self,
        sample_txt_path: Path,
        tmp_path: Path,
        default_config: MarkitaiConfig,
        mock_processor,
    ) -> None:
        """Vision workflow should receive converter-provided title."""
        from unittest.mock import AsyncMock, patch

        from markitai.workflow.core import (
            ConversionContext,
            process_with_vision_llm,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        screenshots_dir = output_dir / ".markitai" / "screenshots"
        screenshots_dir.mkdir(parents=True)
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
            markdown="# Chapter 1",
            metadata={
                "title": "Canonical Document Title",
                "page_images": [{"page": 1, "name": "page1.png", "path": str(page1)}],
            },
        )

        with patch("markitai.workflow.single.SingleFileWorkflow") as MockWorkflow:
            mock_workflow_instance = MockWorkflow.return_value
            mock_workflow_instance.enhance_with_vision = AsyncMock(
                return_value=(
                    "# Enhanced",
                    "title: Canonical Document Title",
                    0.05,
                    {"gpt-4": {"requests": 1}},
                )
            )

            result = await process_with_vision_llm(ctx)

        assert result.success is True
        mock_workflow_instance.enhance_with_vision.assert_awaited_once_with(
            "# Chapter 1",
            [{"page": 1, "name": "page1.png", "path": str(page1)}],
            source=sample_txt_path.name,
            original_title="Canonical Document Title",
        )

    @pytest.mark.asyncio
    async def test_no_redundant_stabilization_in_core_vision_path(
        self,
        sample_txt_path: Path,
        tmp_path: Path,
        default_config: MarkitaiConfig,
        mock_processor,
    ) -> None:
        """Core vision path should NOT call _stabilize_paged_markdown directly.

        Stabilization is the responsibility of enhance_with_vision() in single.py;
        core.py must not duplicate those calls.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from markitai.workflow.core import (
            ConversionContext,
            process_with_vision_llm,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        screenshots_dir = output_dir / ".markitai" / "screenshots"
        screenshots_dir.mkdir(parents=True)
        page1 = screenshots_dir / "page1.png"
        page1.write_bytes(b"fake image")

        raw_markdown = """<!-- Page number: 1 -->

# Page 1

Alpha

Repeated baseline drift
"""

        def format_output(markdown: str, frontmatter: str, **_: object) -> str:
            return f"---\n{frontmatter}\n---\n\n{markdown}"

        mock_processor.format_llm_output = MagicMock(side_effect=format_output)

        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=output_dir,
            config=default_config,
            shared_processor=mock_processor,
        )
        ctx.output_file = output_dir / "test.md"
        ctx.output_file.write_text(
            f"---\ntitle: Test\n---\n\n{raw_markdown}",
            encoding="utf-8",
        )
        ctx.conversion_result = ConvertResult(
            markdown=raw_markdown,
            metadata={
                "page_images": [{"page": 1, "name": "page1.png", "path": str(page1)}]
            },
        )

        enhanced_content = "<!-- Page number: 1 -->\n\n# Page 1\n\nAlpha"

        with patch("markitai.workflow.single.SingleFileWorkflow") as MockWorkflow:
            mock_workflow_instance = MockWorkflow.return_value
            mock_workflow_instance.enhance_with_vision = AsyncMock(
                return_value=(
                    enhanced_content,
                    "title: Test",
                    0.05,
                    {"gpt-4": {"requests": 1}},
                )
            )
            mock_processor._stabilize_paged_markdown = MagicMock()

            result = await process_with_vision_llm(ctx)

        assert result.success is True
        # core.py must NOT call _stabilize_paged_markdown — that is single.py's job
        mock_processor._stabilize_paged_markdown.assert_not_called()
        llm_output = ctx.output_file.with_suffix(".llm.md").read_text()
        assert "Alpha" in llm_output

    @pytest.mark.asyncio
    async def test_screenshot_only_passes_converter_title_to_extraction_workflow(
        self,
        sample_txt_path: Path,
        tmp_path: Path,
        default_config: MarkitaiConfig,
        mock_processor,
    ) -> None:
        """Screenshot-only vision mode should also receive converter title."""
        from unittest.mock import AsyncMock, patch

        from markitai.workflow.core import (
            ConversionContext,
            process_with_vision_llm,
        )

        default_config.screenshot.screenshot_only = True

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        screenshots_dir = output_dir / ".markitai" / "screenshots"
        screenshots_dir.mkdir(parents=True)
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
            markdown="# Chapter 1",
            metadata={
                "title": "Canonical Document Title",
                "page_images": [{"page": 1, "name": "page1.png", "path": str(page1)}],
            },
        )

        with patch("markitai.workflow.single.SingleFileWorkflow") as MockWorkflow:
            mock_workflow_instance = MockWorkflow.return_value
            mock_workflow_instance.extract_from_screenshots = AsyncMock(
                return_value=(
                    "# Enhanced",
                    "title: Canonical Document Title",
                    0.05,
                    {"gpt-4": {"requests": 1}},
                )
            )

            result = await process_with_vision_llm(ctx)

        assert result.success is True
        mock_workflow_instance.extract_from_screenshots.assert_awaited_once_with(
            [{"page": 1, "name": "page1.png", "path": str(page1)}],
            source=sample_txt_path.name,
            original_title="Canonical Document Title",
        )

    @pytest.mark.asyncio
    async def test_screenshot_only_does_not_reference_extracted_text(
        self,
        sample_txt_path: Path,
        tmp_path: Path,
        default_config: MarkitaiConfig,
        mock_processor,
    ) -> None:
        """Screenshot-only mode must not reference extracted_text (unbound in that branch)."""
        from unittest.mock import AsyncMock, patch

        from markitai.workflow.core import (
            ConversionContext,
            process_with_vision_llm,
        )

        default_config.screenshot.screenshot_only = True

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        screenshots_dir = output_dir / ".markitai" / "screenshots"
        screenshots_dir.mkdir(parents=True)
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
            markdown="# Original from converter",
            metadata={
                "page_images": [{"page": 1, "name": "page1.png", "path": str(page1)}],
            },
        )

        with patch("markitai.workflow.single.SingleFileWorkflow") as MockWorkflow:
            mock_workflow_instance = MockWorkflow.return_value
            mock_workflow_instance.extract_from_screenshots = AsyncMock(
                return_value=(
                    "# Enhanced from screenshots",
                    "title: Enhanced",
                    0.05,
                    {"gpt-4": {"requests": 1}},
                )
            )

            # Must not raise UnboundLocalError for extracted_text
            result = await process_with_vision_llm(ctx)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_preserves_base_timestamp_before_vision_llm_output(
        self,
        sample_txt_path: Path,
        tmp_path: Path,
        default_config: MarkitaiConfig,
        mock_processor,
    ) -> None:
        """Vision path should keep base .md older than the generated .llm.md."""
        from markitai.workflow.core import (
            ConversionContext,
            process_with_vision_llm,
            write_base_markdown,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        screenshots_dir = output_dir / ".markitai" / "screenshots"
        screenshots_dir.mkdir(parents=True)
        page1 = screenshots_dir / "page1.png"
        page1.write_bytes(b"fake image")

        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=output_dir,
            config=default_config,
            shared_processor=mock_processor,
        )
        ctx.output_file = output_dir / "test.md"
        ctx.conversion_result = ConvertResult(
            markdown="# Test",
            metadata={
                "page_images": [{"page": 1, "name": "page1.png", "path": str(page1)}]
            },
        )

        with patch(
            "markitai.utils.frontmatter.frontmatter_timestamp",
            side_effect=["2026-03-07T11:50:42.843+08:00"],
        ):
            base_result = write_base_markdown(ctx)

        assert base_result.success is True

        with patch("markitai.workflow.single.SingleFileWorkflow") as MockWorkflow:
            mock_workflow_instance = MockWorkflow.return_value
            mock_workflow_instance.enhance_with_vision = AsyncMock(
                return_value=(
                    "# Enhanced",
                    "title: Test\nmarkitai_processed: '2026-03-07T11:51:58.318+08:00'",
                    0.05,
                    {"gpt-4": {"requests": 1}},
                )
            )
            mock_processor.format_llm_output = MagicMock(
                side_effect=lambda markdown, frontmatter, **_: (
                    f"---\n{frontmatter}\n---\n\n{markdown}"
                )
            )

            result = await process_with_vision_llm(ctx)

        assert result.success is True

        base_frontmatter = yaml.safe_load(ctx.output_file.read_text().split("---")[1])
        llm_frontmatter = yaml.safe_load(
            ctx.output_file.with_suffix(".llm.md").read_text().split("---")[1]
        )
        assert base_frontmatter["markitai_processed"] == "2026-03-07T11:50:42.843+08:00"
        assert llm_frontmatter["markitai_processed"] == "2026-03-07T11:51:58.318+08:00"


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
        assets_dir = output_dir / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)

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
            markdown="![](.markitai/assets/test.png.image1.png)",
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

    @pytest.mark.asyncio
    async def test_passes_converter_title_to_document_processing(
        self,
        tmp_path: Path,
        default_config: MarkitaiConfig,
        mock_processor,
    ) -> None:
        """Converter-provided titles should flow into standard LLM processing."""
        from unittest.mock import AsyncMock, patch

        from markitai.workflow.core import (
            ConversionContext,
            process_with_standard_llm,
        )

        epub_file = tmp_path / "sample.epub"
        epub_file.write_text("fake epub payload")

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        ctx = ConversionContext(
            input_path=epub_file,
            output_dir=output_dir,
            config=default_config,
            shared_processor=mock_processor,
        )
        ctx.output_file = output_dir / "sample.epub.md"
        ctx.output_file.touch()
        ctx.conversion_result = ConvertResult(
            markdown=(
                "**Title:** Test EPUB Document\n\n"
                "## Test EPUB Document\n\n"
                "# Chapter 1: Test Content"
            ),
            metadata={"title": "Test EPUB Document"},
        )

        with patch("markitai.workflow.single.SingleFileWorkflow") as MockWorkflow:
            mock_workflow_instance = MockWorkflow.return_value
            mock_workflow_instance.process_document_with_llm = AsyncMock(
                return_value=(ctx.conversion_result.markdown, 0.01, {})
            )
            mock_workflow_instance.analyze_images = AsyncMock(
                return_value=(ctx.conversion_result.markdown, 0.0, {}, None)
            )

            result = await process_with_standard_llm(ctx)

        assert result.success is True
        mock_workflow_instance.process_document_with_llm.assert_awaited_once_with(
            ctx.conversion_result.markdown,
            "sample.epub",
            ctx.output_file,
            title="Test EPUB Document",
        )

    @pytest.mark.asyncio
    async def test_standard_llm_does_not_regenerate_base_timestamp(
        self,
        sample_txt_path: Path,
        tmp_path: Path,
        default_config: MarkitaiConfig,
        mock_processor,
    ) -> None:
        """Standard LLM path should not rewrite base .md with a newer timestamp."""
        from markitai.workflow.core import (
            ConversionContext,
            process_with_standard_llm,
            write_base_markdown,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        mock_processor.process_document.return_value = (
            "# Cleaned",
            "title: Test\nmarkitai_processed: '2026-03-07T11:50:43.800+08:00'",
        )
        mock_processor.format_llm_output = MagicMock(
            side_effect=lambda markdown, frontmatter, **_: (
                f"---\n{frontmatter}\n---\n\n{markdown}"
            )
        )

        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=output_dir,
            config=default_config,
            shared_processor=mock_processor,
        )
        ctx.output_file = output_dir / "test.md"
        ctx.conversion_result = ConvertResult(
            markdown="# Test\n\nContent",
            metadata={},
        )

        with patch(
            "markitai.utils.frontmatter.frontmatter_timestamp",
            side_effect=[
                "2026-03-07T11:50:43.799+08:00",
                "2026-03-07T11:50:43.801+08:00",
            ],
        ):
            base_result = write_base_markdown(ctx)
            assert base_result.success is True
            result = await process_with_standard_llm(ctx)

        assert result.success is True

        base_frontmatter = yaml.safe_load(ctx.output_file.read_text().split("---")[1])
        llm_frontmatter = yaml.safe_load(
            ctx.output_file.with_suffix(".llm.md").read_text().split("---")[1]
        )
        assert base_frontmatter["markitai_processed"] == "2026-03-07T11:50:43.799+08:00"
        assert llm_frontmatter["markitai_processed"] == "2026-03-07T11:50:43.800+08:00"

    @pytest.mark.asyncio
    async def test_standard_llm_restabilizes_written_llm_output(
        self,
        sample_txt_path: Path,
        tmp_path: Path,
        default_config: MarkitaiConfig,
        mock_processor,
    ) -> None:
        """Standard LLM should stabilize the final .llm.md body against base markdown."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from markitai.workflow.core import (
            ConversionContext,
            process_with_standard_llm,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        original_markdown = """<!-- Page number: 1 -->

# Page 1

Alpha
"""
        drifted_markdown = """<!-- Page number: 1 -->

# Page 1

Alpha

Extra drift
"""
        mock_processor._stabilize_paged_markdown = MagicMock(
            return_value=original_markdown
        )

        ctx = ConversionContext(
            input_path=sample_txt_path,
            output_dir=output_dir,
            config=default_config,
            shared_processor=mock_processor,
        )
        ctx.output_file = output_dir / "test.md"
        ctx.output_file.write_text(
            f"---\ntitle: Test\n---\n\n{original_markdown}",
            encoding="utf-8",
        )
        ctx.conversion_result = ConvertResult(
            markdown=original_markdown,
            metadata={},
        )

        async def write_drifted_llm(
            *args: object, **kwargs: object
        ) -> tuple[str, float, dict]:
            del args, kwargs
            llm_output = ctx.output_file.with_suffix(".llm.md")
            llm_output.write_text(
                f"---\ntitle: Test\n---\n\n{drifted_markdown}",
                encoding="utf-8",
            )
            return original_markdown, 0.01, {}

        with patch("markitai.workflow.single.SingleFileWorkflow") as MockWorkflow:
            mock_workflow_instance = MockWorkflow.return_value
            mock_workflow_instance.process_document_with_llm = AsyncMock(
                side_effect=write_drifted_llm
            )

            result = await process_with_standard_llm(ctx)

        assert result.success is True
        llm_output = ctx.output_file.with_suffix(".llm.md").read_text(encoding="utf-8")
        assert "Extra drift" not in llm_output
        assert "Alpha" in llm_output


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
        assets_dir = output_dir / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)

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
        assets_dir = output_dir / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)

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
        assets_dir = sample_context.output_dir / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)

        # Create images with various case extensions
        (assets_dir / f"{sample_context.input_path.name}.image1.PNG").touch()
        (assets_dir / f"{sample_context.input_path.name}.image2.JpG").touch()
        (assets_dir / f"{sample_context.input_path.name}.image3.jpeg").touch()

        images = get_saved_images(sample_context)
        assert len(images) == 3


class TestGetSavedImagesTranscoded:
    """Tests for get_saved_images with transcoded image formats (BMP/TIFF→PNG)."""

    def test_finds_transcoded_bmp_via_metadata(
        self, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """BMP files are transcoded to PNG with a hash suffix; get_saved_images must find them."""
        from markitai.workflow.core import ConversionContext, get_saved_images

        input_file = tmp_path / "sample.bmp"
        input_file.write_bytes(b"fake bmp")

        output_dir = tmp_path / "output"
        assets_dir = output_dir / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)

        # Simulate the transcoded PNG that ImageConverter creates
        transcoded = assets_dir / "sample-56b050ec0595.png"
        transcoded.touch()

        ctx = ConversionContext(
            input_path=input_file,
            output_dir=output_dir,
            config=default_config,
        )
        # Simulate conversion_result with asset_path metadata
        ctx.conversion_result = ConvertResult(
            markdown="# sample\n\n![sample](.markitai/assets/sample-56b050ec0595.png)\n",
            metadata={"asset_path": ".markitai/assets/sample-56b050ec0595.png"},
        )

        images = get_saved_images(ctx)
        assert len(images) == 1
        assert images[0].name == "sample-56b050ec0595.png"

    def test_finds_transcoded_tiff_via_metadata(
        self, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """TIFF files are also transcoded; get_saved_images must find them."""
        from markitai.workflow.core import ConversionContext, get_saved_images

        input_file = tmp_path / "photo.tiff"
        input_file.write_bytes(b"fake tiff")

        output_dir = tmp_path / "output"
        assets_dir = output_dir / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)

        transcoded = assets_dir / "photo-aabbccdd1122.png"
        transcoded.touch()

        ctx = ConversionContext(
            input_path=input_file,
            output_dir=output_dir,
            config=default_config,
        )
        ctx.conversion_result = ConvertResult(
            markdown="# photo\n\n![photo](.markitai/assets/photo-aabbccdd1122.png)\n",
            metadata={"asset_path": ".markitai/assets/photo-aabbccdd1122.png"},
        )

        images = get_saved_images(ctx)
        assert len(images) == 1
        assert images[0].name == "photo-aabbccdd1122.png"

    def test_non_transcoded_jpg_still_works(
        self, tmp_path: Path, default_config: MarkitaiConfig
    ) -> None:
        """JPG files are NOT transcoded; existing glob logic must still work."""
        from markitai.workflow.core import ConversionContext, get_saved_images

        input_file = tmp_path / "sample.jpg"
        input_file.write_bytes(b"fake jpg")

        output_dir = tmp_path / "output"
        assets_dir = output_dir / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "sample.jpg").touch()

        ctx = ConversionContext(
            input_path=input_file,
            output_dir=output_dir,
            config=default_config,
        )
        ctx.conversion_result = ConvertResult(
            markdown="# sample\n\n![sample](.markitai/assets/sample.jpg)\n",
            metadata={"asset_path": ".markitai/assets/sample.jpg"},
        )

        images = get_saved_images(ctx)
        assert len(images) == 1
        assert images[0].name == "sample.jpg"


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
        """Image files without LLM or OCR are skipped (Rule A: image_only)."""
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

        # Rule A: image-only format without LLM or OCR → skip cleanly
        assert result.success is True
        assert result.skip_reason == "image_only"
        assert ctx.output_file is None


class TestHeavyTaskIdentification:
    """Tests for heavy task identification in convert_document.

    Medium-1: OCR+LLM paths for PDF and PPTX should be classified as heavy tasks.
    """

    @pytest.mark.asyncio
    async def test_pdf_with_ocr_and_llm_is_heavy(self, tmp_path: Path) -> None:
        """PDF with --ocr --llm renders page images (CPU heavy) and should use heavy semaphore."""
        from markitai.config import MarkitaiConfig
        from markitai.workflow.core import ConversionContext, convert_document

        config = MarkitaiConfig()
        config.ocr.enabled = True
        config.llm.enabled = True
        # screenshot is NOT enabled — the point is OCR+LLM alone is heavy

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        ctx = ConversionContext(
            input_path=pdf_file,
            output_dir=tmp_path / "output",
            config=config,
        )
        mock_converter = MagicMock()
        mock_converter.convert.return_value = ConvertResult(markdown="# Test")
        ctx.converter = mock_converter

        with patch(
            "markitai.utils.executor.get_heavy_task_semaphore"
        ) as mock_semaphore_fn:
            mock_sem = AsyncMock()
            mock_sem.__aenter__ = AsyncMock(return_value=None)
            mock_sem.__aexit__ = AsyncMock(return_value=False)
            mock_semaphore_fn.return_value = mock_sem

            await convert_document(ctx)

            # Should have acquired the heavy semaphore
            mock_semaphore_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_pptx_with_ocr_is_heavy(self, tmp_path: Path) -> None:
        """PPTX with --ocr renders slide images (CPU heavy) and should use heavy semaphore."""
        from markitai.config import MarkitaiConfig
        from markitai.workflow.core import ConversionContext, convert_document

        config = MarkitaiConfig()
        config.ocr.enabled = True
        # llm and screenshot are NOT enabled

        pptx_file = tmp_path / "test.pptx"
        pptx_file.write_bytes(b"PK fake")

        ctx = ConversionContext(
            input_path=pptx_file,
            output_dir=tmp_path / "output",
            config=config,
        )
        mock_converter = MagicMock()
        mock_converter.convert.return_value = ConvertResult(markdown="# Slides")
        ctx.converter = mock_converter

        with patch(
            "markitai.utils.executor.get_heavy_task_semaphore"
        ) as mock_semaphore_fn:
            mock_sem = AsyncMock()
            mock_sem.__aenter__ = AsyncMock(return_value=None)
            mock_sem.__aexit__ = AsyncMock(return_value=False)
            mock_semaphore_fn.return_value = mock_sem

            await convert_document(ctx)

            mock_semaphore_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_pdf_without_ocr_llm_or_screenshot_is_not_heavy(
        self, tmp_path: Path
    ) -> None:
        """PDF without OCR, LLM, or screenshot should NOT be classified as heavy."""
        from markitai.config import MarkitaiConfig
        from markitai.workflow.core import ConversionContext, convert_document

        config = MarkitaiConfig()
        # All disabled by default

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        ctx = ConversionContext(
            input_path=pdf_file,
            output_dir=tmp_path / "output",
            config=config,
        )
        mock_converter = MagicMock()
        mock_converter.convert.return_value = ConvertResult(markdown="# Test")
        ctx.converter = mock_converter

        with patch(
            "markitai.utils.executor.get_heavy_task_semaphore"
        ) as mock_semaphore_fn:
            mock_sem = AsyncMock()
            mock_sem.__aenter__ = AsyncMock(return_value=None)
            mock_sem.__aexit__ = AsyncMock(return_value=False)
            mock_semaphore_fn.return_value = mock_sem

            await convert_document(ctx)

            # Should NOT have acquired the heavy semaphore
            mock_semaphore_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_pptx_with_ocr_and_llm_is_heavy(self, tmp_path: Path) -> None:
        """PPTX with --ocr --llm should also be heavy."""
        from markitai.config import MarkitaiConfig
        from markitai.workflow.core import ConversionContext, convert_document

        config = MarkitaiConfig()
        config.ocr.enabled = True
        config.llm.enabled = True

        pptx_file = tmp_path / "test.pptx"
        pptx_file.write_bytes(b"PK fake")

        ctx = ConversionContext(
            input_path=pptx_file,
            output_dir=tmp_path / "output",
            config=config,
        )
        mock_converter = MagicMock()
        mock_converter.convert.return_value = ConvertResult(markdown="# Slides")
        ctx.converter = mock_converter

        with patch(
            "markitai.utils.executor.get_heavy_task_semaphore"
        ) as mock_semaphore_fn:
            mock_sem = AsyncMock()
            mock_sem.__aenter__ = AsyncMock(return_value=None)
            mock_sem.__aexit__ = AsyncMock(return_value=False)
            mock_semaphore_fn.return_value = mock_sem

            await convert_document(ctx)

            mock_semaphore_fn.assert_called_once()


class TestOnConflictSkipBeforeConversion:
    """Tests for on_conflict=skip check happening BEFORE heavy conversion work.

    Medium-3: When on_conflict=skip and output already exists, the pipeline
    should skip before running the expensive conversion step.
    """

    @pytest.mark.asyncio
    async def test_skip_avoids_conversion(
        self,
        tmp_path: Path,
    ) -> None:
        """When output exists and on_conflict=skip, conversion should NOT run."""
        from markitai.config import MarkitaiConfig
        from markitai.workflow.core import ConversionContext, convert_document_core

        config = MarkitaiConfig()
        config.output.on_conflict = "skip"

        # Create input file
        input_file = tmp_path / "test.txt"
        input_file.write_text("hello world")

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # Create existing output file (the one that should cause a skip)
        existing_output = output_dir / "test.txt.md"
        existing_output.write_text("already exists")

        ctx = ConversionContext(
            input_path=input_file,
            output_dir=output_dir,
            config=config,
        )

        # Patch convert_document to track if it gets called
        with patch(
            "markitai.workflow.core.convert_document", new_callable=AsyncMock
        ) as mock_convert:
            mock_convert.return_value = ConversionStepResult(success=True)

            result = await convert_document_core(
                ctx, max_document_size=100 * 1024 * 1024
            )

            # Should skip with "exists" reason
            assert result.success is True
            assert result.skip_reason == "exists"

            # The expensive conversion step should NOT have been called
            mock_convert.assert_not_called()

    @pytest.mark.asyncio
    async def test_overwrite_still_runs_conversion(
        self,
        tmp_path: Path,
    ) -> None:
        """When on_conflict=overwrite, conversion should still run even if output exists."""
        from markitai.config import MarkitaiConfig
        from markitai.workflow.core import ConversionContext, convert_document_core

        config = MarkitaiConfig()
        config.output.on_conflict = "overwrite"

        input_file = tmp_path / "test.txt"
        input_file.write_text("hello world")

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # Existing output file
        existing_output = output_dir / "test.txt.md"
        existing_output.write_text("old content")

        ctx = ConversionContext(
            input_path=input_file,
            output_dir=output_dir,
            config=config,
        )

        result = await convert_document_core(ctx, max_document_size=100 * 1024 * 1024)

        # Should succeed and actually convert (overwrite)
        assert result.success is True
        assert result.skip_reason is None
        assert ctx.output_file is not None
        # Content should be from actual conversion, not old
        new_content = ctx.output_file.read_text()
        assert "old content" not in new_content


class TestVisionEmbedSequentialExecution:
    """Verify vision and embed tasks run sequentially, not in parallel."""

    async def test_embed_sees_vision_modified_markdown(self, tmp_path: Path) -> None:
        """analyze_embedded_images must read the markdown that
        process_with_vision_llm wrote, not a stale snapshot."""
        import asyncio

        from markitai.config import MarkitaiConfig
        from markitai.workflow.core import convert_document_core

        config = MarkitaiConfig()
        config.llm.enabled = True

        input_file = tmp_path / "test.txt"
        input_file.write_text("# Test Content\n\nSome text here.")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        ctx = ConversionContext(
            input_path=input_file,
            output_dir=output_dir,
            config=config,
        )

        call_order: list[str] = []
        markdown_seen_by_embed: list[str] = []
        vision_md = "# Vision Enhanced\n\nCleaned by LLM"

        async def mock_vision(
            c: ConversionContext,
        ) -> ConversionStepResult:
            call_order.append("vision_start")
            await asyncio.sleep(0.01)
            c.conversion_result.markdown = vision_md  # type: ignore[union-attr]
            call_order.append("vision_end")
            return ConversionStepResult(success=True)

        async def mock_embed(
            c: ConversionContext,
        ) -> ConversionStepResult:
            call_order.append("embed_start")
            markdown_seen_by_embed.append(
                c.conversion_result.markdown  # type: ignore[union-attr]
            )
            call_order.append("embed_end")
            return ConversionStepResult(success=True)

        mock_processor = MagicMock()

        # We need to patch the early pipeline steps to pass through,
        # then set up ctx as if steps 1-6 completed, with page_images
        # triggering the parallel vision+embed path.
        def fake_write_base(c: ConversionContext) -> ConversionStepResult:
            """After base markdown is written, inject page_images metadata."""
            if c.conversion_result is not None:
                c.conversion_result.metadata["page_images"] = [
                    {"page": 1, "name": "p1.png", "path": "/tmp/p1.png"}
                ]
            return ConversionStepResult(success=True)

        with (
            patch(
                "markitai.workflow.core.process_with_vision_llm",
                side_effect=mock_vision,
            ),
            patch(
                "markitai.workflow.core.analyze_embedded_images",
                side_effect=mock_embed,
            ),
            patch(
                "markitai.workflow.helpers.create_llm_processor",
                return_value=mock_processor,
            ),
            patch(
                "markitai.workflow.core.write_base_markdown",
                side_effect=fake_write_base,
            ),
            patch("markitai.workflow.core.stabilize_written_llm_output"),
        ):
            await convert_document_core(ctx, max_document_size=500 * 1024 * 1024)

        # embed must start AFTER vision ends (sequential, not parallel)
        assert "vision_end" in call_order, f"vision never completed: {call_order}"
        assert "embed_start" in call_order, f"embed never started: {call_order}"
        assert call_order.index("vision_end") < call_order.index("embed_start"), (
            f"embed started before vision finished: {call_order}"
        )

        # embed should see the markdown that vision wrote
        assert len(markdown_seen_by_embed) == 1
        assert markdown_seen_by_embed[0] == vision_md


# =============================================================================
# TestPagedStabilizedFlag Tests
# =============================================================================


class TestPagedStabilizedFlag:
    """Test that paged_stabilized flag skips stabilize_written_llm_output."""

    def test_context_has_paged_stabilized_field(self):
        """ConversionContext should have paged_stabilized field, default False."""
        ctx = ConversionContext(
            input_path=Path("test.pptx"),
            output_dir=Path("/tmp/out"),
            config=MarkitaiConfig(),
        )
        assert ctx.paged_stabilized is False

    def test_stabilize_skipped_when_flag_set(self, tmp_path: Path):
        """stabilize_written_llm_output should be skipped when paged_stabilized=True."""
        from markitai.workflow.core import (
            ConversionContext,
            stabilize_written_llm_output,
        )

        output_file = tmp_path / "test.md"
        output_file.write_text("# baseline", encoding="utf-8")
        llm_file = tmp_path / "test.llm.md"
        llm_file.write_text("---\ntitle: t\n---\n\n# changed", encoding="utf-8")

        ctx = ConversionContext(
            input_path=Path("test.pptx"),
            output_dir=tmp_path,
            config=MarkitaiConfig(),
        )
        ctx.output_file = output_file
        ctx.conversion_result = ConvertResult(
            markdown="# baseline", images=[], metadata={}
        )
        ctx.paged_stabilized = True

        result = stabilize_written_llm_output(ctx, MagicMock())
        assert result is False

        assert llm_file.read_text(encoding="utf-8") == "---\ntitle: t\n---\n\n# changed"


class TestProcessWithPureLLM:
    """Test pure mode pipeline integration."""

    async def test_pure_mode_calls_process_document_pure(self, tmp_path: Path):
        """process_with_pure_llm should call process_document_pure."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from markitai.config import (
            LiteLLMParams,
            MarkitaiConfig,
            ModelConfig,
        )
        from markitai.converter.base import ConvertResult
        from markitai.workflow.core import ConversionContext, process_with_pure_llm

        config = MarkitaiConfig()
        config.llm.enabled = True
        config.llm.pure = True
        config.llm.model_list = [
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model="openai/gpt-4o-mini", api_key="t"),
            )
        ]

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("# Hello", encoding="utf-8")
        output_file = tmp_path / "test.md"
        output_file.write_text("# Hello", encoding="utf-8")

        ctx = ConversionContext(
            input_path=txt_file,
            output_dir=tmp_path,
            config=config,
        )
        ctx.output_file = output_file
        ctx.conversion_result = ConvertResult(
            markdown="# Hello", images=[], metadata={}
        )

        mock_workflow = MagicMock()
        mock_workflow.process_document_pure = AsyncMock(
            return_value=("# Hello", 0.001, {})
        )

        with (
            patch(
                "markitai.workflow.single.SingleFileWorkflow",
                return_value=mock_workflow,
            ),
            patch("markitai.workflow.helpers.create_llm_processor"),
        ):
            result = await process_with_pure_llm(ctx)

        assert result.success
        mock_workflow.process_document_pure.assert_called_once()
        assert ctx.llm_cost == 0.001


# =============================================================================
# Task 4: detected_format stored on ConversionContext
# =============================================================================


class TestDetectedFormat:
    def test_detected_format_set_after_validation(self, tmp_path, fixtures_dir):
        from markitai.constants import MAX_DOCUMENT_SIZE
        from markitai.converter.base import FileFormat

        input_path = fixtures_dir / "sample.csv"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        result = validate_and_detect_format(ctx, MAX_DOCUMENT_SIZE)
        assert result.success
        assert ctx.detected_format == FileFormat.CSV

    def test_detected_format_for_image(self, tmp_path, fixtures_dir):
        from markitai.constants import MAX_DOCUMENT_SIZE
        from markitai.converter.base import FileFormat

        input_path = fixtures_dir / "sample.bmp"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        result = validate_and_detect_format(ctx, MAX_DOCUMENT_SIZE)
        assert result.success
        assert ctx.detected_format == FileFormat.BMP


# =============================================================================
# Task 5: image-only skip logic in convert_document_core()
# =============================================================================


class TestImageOnlySkip:
    async def test_image_skipped_without_llm_or_ocr(self, tmp_path, fixtures_dir):
        from markitai.constants import MAX_DOCUMENT_SIZE
        from markitai.workflow.core import convert_document_core

        input_path = fixtures_dir / "sample.bmp"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)
        assert result.success is True
        assert result.skip_reason == "image_only"
        output_files = list(output_dir.glob("*.md"))
        assert len(output_files) == 0

    async def test_image_not_skipped_with_llm(self, tmp_path, fixtures_dir):
        from markitai.constants import MAX_DOCUMENT_SIZE
        from markitai.workflow.core import convert_document_core

        input_path = fixtures_dir / "sample.bmp"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)
        assert result.skip_reason != "image_only"

    async def test_image_not_skipped_with_ocr(self, tmp_path, fixtures_dir):
        from markitai.constants import MAX_DOCUMENT_SIZE
        from markitai.workflow.core import convert_document_core

        input_path = fixtures_dir / "sample.bmp"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        cfg.ocr.enabled = True
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)
        assert result.skip_reason != "image_only"

    async def test_document_not_skipped_without_llm(self, tmp_path, fixtures_dir):
        from markitai.constants import MAX_DOCUMENT_SIZE
        from markitai.workflow.core import convert_document_core

        input_path = fixtures_dir / "sample.csv"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)
        assert result.skip_reason != "image_only"


class TestLLMOnlyOutput:
    """Tests for LLM mode outputting only .llm.md."""

    async def test_llm_mode_skips_base_md_file(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """In LLM mode, base .md file should not exist on disk after write_base_markdown."""
        from markitai.constants import MAX_DOCUMENT_SIZE
        from markitai.workflow.core import (
            convert_document,
            prepare_output_directory,
            process_embedded_images,
            resolve_output_file,
            validate_and_detect_format,
            write_base_markdown,
        )

        input_path = fixtures_dir / "sample.csv"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )

        validate_and_detect_format(ctx, MAX_DOCUMENT_SIZE)
        prepare_output_directory(ctx)
        resolve_output_file(ctx)
        await convert_document(ctx)
        await process_embedded_images(ctx)
        result = write_base_markdown(ctx)
        assert result.success is True
        # Base .md should NOT exist on disk (LLM mode, no --keep-base)
        assert not ctx.output_file.exists()
        # But in-memory markdown is still available
        assert ctx.conversion_result is not None
        assert len(ctx.conversion_result.markdown) > 0

    async def test_keep_base_writes_md_file(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """With --keep-base, base .md SHOULD be written even in LLM mode."""
        from markitai.constants import MAX_DOCUMENT_SIZE
        from markitai.workflow.core import (
            convert_document,
            prepare_output_directory,
            process_embedded_images,
            resolve_output_file,
            validate_and_detect_format,
            write_base_markdown,
        )

        input_path = fixtures_dir / "sample.csv"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.llm.keep_base = True
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )

        validate_and_detect_format(ctx, MAX_DOCUMENT_SIZE)
        prepare_output_directory(ctx)
        resolve_output_file(ctx)
        await convert_document(ctx)
        await process_embedded_images(ctx)
        result = write_base_markdown(ctx)
        assert result.success is True
        assert ctx.output_file.exists()

    async def test_non_llm_always_writes_md_file(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """Without LLM, base .md should always be written (unchanged behavior)."""
        from markitai.constants import MAX_DOCUMENT_SIZE
        from markitai.workflow.core import convert_document_core

        input_path = fixtures_dir / "sample.csv"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)
        assert result.success is True
        assert ctx.output_file is not None
        assert ctx.output_file.exists()


class TestReadMarkdownBodyFallback:
    """Tests that _read_markdown_body falls back to in-memory content."""

    def test_fallback_when_file_missing(self, tmp_path: Path) -> None:
        """When output .md file doesn't exist, should return fallback string."""
        from markitai.workflow.single import _read_markdown_body

        nonexistent = tmp_path / "nonexistent.md"
        fallback = "# Hello\n\nSome content"
        result = _read_markdown_body(nonexistent, fallback)
        assert result == fallback

    def test_reads_file_when_exists(self, tmp_path: Path) -> None:
        """When output .md file exists, should read from it."""
        from markitai.workflow.single import _read_markdown_body

        md_file = tmp_path / "test.md"
        md_file.write_text("---\ntitle: test\n---\n\n# From File\n\nContent")
        result = _read_markdown_body(md_file, "fallback")
        assert "From File" in result
        assert "fallback" not in result


class TestLLMFailureFallback:
    """Tests for writing .md as fallback when LLM processing fails."""

    async def test_md_written_on_llm_failure(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """When LLM fails, .md should be written as fallback."""
        import markitai.workflow.core as core_module
        from markitai.constants import MAX_DOCUMENT_SIZE
        from markitai.workflow.core import convert_document_core

        input_path = fixtures_dir / "sample.csv"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.llm.pure = True  # Use pure mode path (failure point 1)
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )

        # Mock process_with_pure_llm to simulate LLM failure
        async def fake_pure_llm_failure(
            ctx: ConversionContext,
        ) -> ConversionStepResult:
            return ConversionStepResult(success=False, error="LLM unavailable")

        with patch.object(core_module, "process_with_pure_llm", fake_pure_llm_failure):
            result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)

        # LLM should fail
        assert not result.success
        # But .md should be written as fallback
        assert ctx.output_file is not None
        assert ctx.output_file.exists()
        # Verify it has content (not empty)
        content = ctx.output_file.read_text(encoding="utf-8")
        assert len(content) > 0
