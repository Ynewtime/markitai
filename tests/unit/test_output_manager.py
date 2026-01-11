"""Tests for OutputManager service."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from markit.services.output_manager import OutputManager


class TestOutputManagerInit:
    """Tests for OutputManager initialization."""

    def test_default_values(self):
        """Default configuration values are correct."""
        manager = OutputManager()
        assert manager.on_conflict == "rename"
        assert manager.create_assets_subdir is True
        assert manager.generate_image_descriptions is False

    def test_custom_values(self):
        """Custom values are set correctly."""
        manager = OutputManager(
            on_conflict="overwrite",
            create_assets_subdir=False,
            generate_image_descriptions=True,
        )
        assert manager.on_conflict == "overwrite"
        assert manager.create_assets_subdir is False
        assert manager.generate_image_descriptions is True


class TestResolveConflict:
    """Tests for conflict resolution strategies."""

    @pytest.fixture
    def rename_manager(self):
        """Create manager with rename strategy."""
        return OutputManager(on_conflict="rename")

    @pytest.fixture
    def overwrite_manager(self):
        """Create manager with overwrite strategy."""
        return OutputManager(on_conflict="overwrite")

    @pytest.fixture
    def skip_manager(self):
        """Create manager with skip strategy."""
        return OutputManager(on_conflict="skip")

    def test_no_conflict_returns_same_path(self, rename_manager, temp_dir):
        """Non-existing file returns the same path."""
        output_path = temp_dir / "test.md"
        result = rename_manager.resolve_conflict(output_path)
        assert result == output_path

    def test_rename_adds_suffix(self, rename_manager, temp_dir):
        """Rename strategy adds _1 suffix."""
        output_path = temp_dir / "test.md"
        output_path.touch()

        result = rename_manager.resolve_conflict(output_path)

        assert result == temp_dir / "test_1.md"
        assert result != output_path

    def test_rename_increments_suffix(self, rename_manager, temp_dir):
        """Rename strategy increments suffix for multiple conflicts."""
        output_path = temp_dir / "test.md"
        output_path.touch()
        (temp_dir / "test_1.md").touch()
        (temp_dir / "test_2.md").touch()

        result = rename_manager.resolve_conflict(output_path)

        assert result == temp_dir / "test_3.md"

    def test_overwrite_returns_same_path(self, overwrite_manager, temp_dir):
        """Overwrite strategy returns the same path."""
        output_path = temp_dir / "test.md"
        output_path.write_text("existing content")

        result = overwrite_manager.resolve_conflict(output_path)

        assert result == output_path

    def test_skip_raises_error(self, skip_manager, temp_dir):
        """Skip strategy raises ConversionError."""
        from markit.exceptions import ConversionError

        output_path = temp_dir / "test.md"
        output_path.write_text("existing content")

        with pytest.raises(ConversionError) as exc_info:
            skip_manager.resolve_conflict(output_path)

        assert "already exists" in str(exc_info.value)

    def test_rename_preserves_extension(self, rename_manager, temp_dir):
        """Rename preserves the file extension."""
        output_path = temp_dir / "document.pdf.md"
        output_path.touch()

        result = rename_manager.resolve_conflict(output_path)

        assert result.suffix == ".md"
        assert result.stem == "document.pdf_1"


class TestWriteOutput:
    """Tests for write_output method."""

    @pytest.fixture
    def manager(self):
        """Create manager with default settings."""
        return OutputManager()

    @pytest.fixture
    def mock_result(self):
        """Create a mock conversion result."""
        result = Mock()
        result.markdown = "# Test Document\n\nContent here."
        result.images = []
        return result

    async def test_creates_output_directory(self, manager, mock_result, temp_dir):
        """Output directory is created if it doesn't exist."""
        output_dir = temp_dir / "new_output_dir"
        assert not output_dir.exists()

        await manager.write_output(
            input_file=Path("test.pdf"),
            output_dir=output_dir,
            result=mock_result,
        )

        assert output_dir.exists()

    async def test_writes_markdown_content(self, manager, mock_result, temp_dir):
        """Markdown content is written to file."""
        await manager.write_output(
            input_file=Path("test.pdf"),
            output_dir=temp_dir,
            result=mock_result,
        )

        output_file = temp_dir / "test.pdf.md"
        assert output_file.exists()
        content = output_file.read_text()
        assert content == mock_result.markdown

    async def test_writes_images_to_assets(self, manager, temp_dir):
        """Images are written to assets subdirectory."""
        result = Mock()
        result.markdown = "# Test"
        result.images = [
            Mock(filename="image1.png", data=b"png_data"),
            Mock(filename="image2.jpg", data=b"jpg_data"),
        ]

        await manager.write_output(
            input_file=Path("test.pdf"),
            output_dir=temp_dir,
            result=result,
        )

        assets_dir = temp_dir / "assets"
        assert assets_dir.exists()
        assert (assets_dir / "image1.png").exists()
        assert (assets_dir / "image2.jpg").exists()
        assert (assets_dir / "image1.png").read_bytes() == b"png_data"

    async def test_skips_assets_when_disabled(self, temp_dir):
        """Assets subdirectory is not created when disabled."""
        manager = OutputManager(create_assets_subdir=False)
        result = Mock()
        result.markdown = "# Test"
        result.images = [Mock(filename="image.png", data=b"data")]

        await manager.write_output(
            input_file=Path("test.pdf"),
            output_dir=temp_dir,
            result=result,
        )

        # Assets dir should not be created
        assert not (temp_dir / "assets").exists()

    async def test_returns_output_path(self, manager, mock_result, temp_dir):
        """Returns the path to the output file."""
        result = await manager.write_output(
            input_file=Path("document.docx"),
            output_dir=temp_dir,
            result=mock_result,
        )

        assert result == temp_dir / "document.docx.md"

    async def test_handles_conflict(self, manager, mock_result, temp_dir):
        """Handles file conflicts according to strategy."""
        # Create existing file
        (temp_dir / "test.pdf.md").write_text("existing")

        result = await manager.write_output(
            input_file=Path("test.pdf"),
            output_dir=temp_dir,
            result=mock_result,
        )

        # Should be renamed
        assert result == temp_dir / "test.pdf_1.md"


class TestWriteMarkdownOnly:
    """Tests for write_markdown_only method."""

    @pytest.fixture
    def manager(self):
        """Create manager with default settings."""
        return OutputManager()

    async def test_writes_content(self, manager, temp_dir):
        """Markdown content is written."""
        output_file = temp_dir / "output.md"

        await manager.write_markdown_only(
            output_file=output_file,
            markdown_content="# Simple Content",
        )

        assert output_file.read_text() == "# Simple Content"

    async def test_creates_parent_directory(self, manager, temp_dir):
        """Parent directory is created if needed."""
        output_file = temp_dir / "subdir" / "output.md"
        assert not output_file.parent.exists()

        await manager.write_markdown_only(
            output_file=output_file,
            markdown_content="# Content",
        )

        assert output_file.exists()

    async def test_handles_conflict(self, manager, temp_dir):
        """Handles file conflicts."""
        output_file = temp_dir / "output.md"
        output_file.write_text("existing")

        result = await manager.write_markdown_only(
            output_file=output_file,
            markdown_content="# New Content",
        )

        # Should be renamed
        assert result == temp_dir / "output_1.md"


class TestGenerateImageDescriptionMd:
    """Tests for image description markdown generation."""

    @pytest.fixture
    def manager(self):
        """Create manager with image descriptions enabled."""
        return OutputManager(generate_image_descriptions=True)

    @pytest.fixture
    def basic_analysis(self):
        """Create a basic image analysis mock."""
        analysis = Mock()
        analysis.alt_text = "A test image"
        analysis.detailed_description = "This is a detailed description of the test image."
        analysis.detected_text = None
        analysis.image_type = "photo"
        analysis.knowledge_meta = None
        return analysis

    @pytest.fixture
    def full_analysis(self):
        """Create analysis with all fields including knowledge meta."""
        analysis = Mock()
        analysis.alt_text = "Architecture diagram"
        analysis.detailed_description = "A system architecture diagram showing microservices."
        analysis.detected_text = "API Gateway, Service A, Service B"
        analysis.image_type = "diagram"
        analysis.knowledge_meta = Mock(
            entities=["API Gateway", "Service A", "Service B"],
            topics=["microservices", "architecture"],
            relationships=["Service A -> API Gateway", "Service B -> API Gateway"],
            domain="technology",
        )
        return analysis

    def test_basic_format(self, manager, basic_analysis):
        """Basic image description format is correct."""
        generated_at = datetime(2024, 1, 15, 10, 30, tzinfo=UTC)

        result = manager.generate_image_description_md(
            filename="test.png",
            analysis=basic_analysis,
            generated_at=generated_at,
        )

        assert "---" in result  # YAML frontmatter
        assert "source_image: test.png" in result
        assert "image_type: photo" in result
        assert "# Image Description" in result
        assert "## Alt Text" in result
        assert "A test image" in result
        assert "## Detailed Description" in result
        assert "detailed description of the test image" in result

    def test_includes_detected_text(self, manager, full_analysis):
        """Detected text section is included when available."""
        result = manager.generate_image_description_md(
            filename="diagram.png",
            analysis=full_analysis,
            generated_at=datetime.now(UTC),
        )

        assert "## Detected Text" in result
        assert "API Gateway, Service A, Service B" in result

    def test_includes_knowledge_meta(self, manager, full_analysis):
        """Knowledge graph metadata is included in frontmatter."""
        result = manager.generate_image_description_md(
            filename="diagram.png",
            analysis=full_analysis,
            generated_at=datetime.now(UTC),
        )

        assert "entities:" in result
        assert "API Gateway" in result
        assert "topics:" in result
        assert "microservices" in result
        assert "domain: technology" in result

    def test_includes_relationships(self, manager, full_analysis):
        """Relationships section is included when available."""
        result = manager.generate_image_description_md(
            filename="diagram.png",
            analysis=full_analysis,
            generated_at=datetime.now(UTC),
        )

        assert "## Relationships" in result
        assert "Service A -> API Gateway" in result

    def test_no_detected_text_when_none(self, manager, basic_analysis):
        """Detected Text section is omitted when None."""
        result = manager.generate_image_description_md(
            filename="test.png",
            analysis=basic_analysis,
            generated_at=datetime.now(UTC),
        )

        assert "## Detected Text" not in result


class TestWriteOutputWithDescriptions:
    """Tests for writing output with image descriptions."""

    async def test_generates_description_files(self, temp_dir):
        """Image description .md files are generated when enabled."""
        manager = OutputManager(generate_image_descriptions=True)

        # Create mock analysis
        mock_analysis = Mock()
        mock_analysis.alt_text = "Test image"
        mock_analysis.detailed_description = "Description"
        mock_analysis.detected_text = None
        mock_analysis.image_type = "photo"
        mock_analysis.knowledge_meta = None

        # Create mock image info with analysis
        image_info = Mock()
        image_info.filename = "image1.png"
        image_info.analysis = mock_analysis

        # Create mock result
        result = Mock()
        result.markdown = "# Test"
        result.images = [Mock(filename="image1.png", data=b"png_data")]

        await manager.write_output(
            input_file=Path("test.pdf"),
            output_dir=temp_dir,
            result=result,
            image_info_list=[image_info],
        )

        # Check description file was created
        desc_file = temp_dir / "assets" / "image1.png.md"
        assert desc_file.exists()
        content = desc_file.read_text()
        assert "Test image" in content
