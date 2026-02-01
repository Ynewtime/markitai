"""Tests for CLI LLM processor module (cli/processors/llm.py)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.config import (
    ImageConfig,
    LiteLLMParams,
    LLMConfig,
    MarkitaiConfig,
    ModelConfig,
    PromptsConfig,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def markitai_config() -> MarkitaiConfig:
    """Return a test Markitai configuration."""
    return MarkitaiConfig(
        llm=LLMConfig(
            enabled=True,
            model_list=[
                ModelConfig(
                    model_name="default",
                    litellm_params=LiteLLMParams(
                        model="openai/gpt-4o-mini",
                        api_key="test-key",
                    ),
                ),
            ],
            concurrency=2,
        ),
        prompts=PromptsConfig(),
        image=ImageConfig(alt_enabled=True, desc_enabled=True),
    )


@pytest.fixture
def mock_llm_processor():
    """Create a mock LLM processor."""
    processor = MagicMock()
    processor.process_document = AsyncMock(
        return_value=("# Cleaned\n\nContent", "title: Test\nsource: test.md")
    )
    processor.format_llm_output = MagicMock(
        return_value="---\ntitle: Test\n---\n\n# Cleaned\n\nContent"
    )
    processor.get_context_cost = MagicMock(return_value=0.05)
    processor.get_context_usage = MagicMock(
        return_value={
            "openai/gpt-4o-mini": {
                "requests": 1,
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.05,
            }
        }
    )
    processor.analyze_images_batch = AsyncMock(return_value=[])
    processor.enhance_document_complete = AsyncMock(
        return_value=("# Enhanced\n\nContent", "title: Enhanced\nsource: doc.pdf")
    )
    return processor


@pytest.fixture
def sample_image_analysis():
    """Create a sample ImageAnalysis object."""
    analysis = MagicMock()
    analysis.caption = "A test image"
    analysis.description = "This is a detailed description of the test image."
    analysis.extracted_text = "OCR extracted text"
    analysis.llm_usage = {
        "openai/gpt-4o-mini": {
            "requests": 1,
            "input_tokens": 500,
            "output_tokens": 100,
            "cost_usd": 0.01,
        }
    }
    return analysis


# =============================================================================
# Tests for process_with_llm
# =============================================================================


class TestProcessWithLLM:
    """Tests for process_with_llm function."""

    async def test_process_with_llm_basic(
        self, tmp_path: Path, markitai_config: MarkitaiConfig, mock_llm_processor
    ):
        """Test basic LLM processing."""
        from markitai.cli.processors.llm import process_with_llm

        output_file = tmp_path / "output.md"
        markdown = "# Test\n\nContent here."

        with (
            patch(
                "markitai.cli.processors.llm.create_llm_processor",
                return_value=mock_llm_processor,
            ),
            patch(
                "markitai.cli.processors.llm.ImageProcessor.remove_hallucinated_images",
                return_value="# Cleaned\n\nContent",
            ),
        ):
            result_md, cost, usage = await process_with_llm(
                markdown=markdown,
                source="test.md",
                cfg=markitai_config,
                output_file=output_file,
                processor=mock_llm_processor,
            )

        # Should return original markdown unchanged
        assert result_md == markdown
        assert cost == 0.05
        assert "openai/gpt-4o-mini" in usage

        # Should write .llm.md file
        llm_output = output_file.with_suffix(".llm.md")
        assert llm_output.exists()

    async def test_process_with_llm_with_page_images(
        self, tmp_path: Path, markitai_config: MarkitaiConfig, mock_llm_processor
    ):
        """Test LLM processing with page images."""
        from markitai.cli.processors.llm import process_with_llm

        output_file = tmp_path / "output.md"
        markdown = "# Test\n\nContent here."
        page_images = [
            {"page": 1, "name": "page1.png"},
            {"page": 2, "name": "page2.png"},
        ]

        with (
            patch(
                "markitai.cli.processors.llm.create_llm_processor",
                return_value=mock_llm_processor,
            ),
            patch(
                "markitai.cli.processors.llm.ImageProcessor.remove_hallucinated_images",
                return_value="# Cleaned\n\nContent",
            ),
        ):
            result_md, cost, usage = await process_with_llm(
                markdown=markdown,
                source="test.md",
                cfg=markitai_config,
                output_file=output_file,
                page_images=page_images,
                processor=mock_llm_processor,
            )

        # Check that page images are added to llm output
        llm_output = output_file.with_suffix(".llm.md")
        content = llm_output.read_text()
        assert "<!-- Page images for reference -->" in content
        assert "<!-- ![Page 1](screenshots/page1.png) -->" in content
        assert "<!-- ![Page 2](screenshots/page2.png) -->" in content

    async def test_process_with_llm_creates_processor_if_none(
        self, tmp_path: Path, markitai_config: MarkitaiConfig, mock_llm_processor
    ):
        """Test that processor is created if not provided."""
        from markitai.cli.processors.llm import process_with_llm

        output_file = tmp_path / "output.md"
        markdown = "# Test\n\nContent here."

        with (
            patch(
                "markitai.cli.processors.llm.create_llm_processor",
                return_value=mock_llm_processor,
            ) as mock_create,
            patch(
                "markitai.cli.processors.llm.ImageProcessor.remove_hallucinated_images",
                return_value="# Cleaned\n\nContent",
            ),
        ):
            await process_with_llm(
                markdown=markdown,
                source="test.md",
                cfg=markitai_config,
                output_file=output_file,
                processor=None,  # No processor provided
            )

        mock_create.assert_called_once_with(markitai_config)

    async def test_process_with_llm_error_handling(
        self, tmp_path: Path, markitai_config: MarkitaiConfig
    ):
        """Test error handling in process_with_llm."""
        from markitai.cli.processors.llm import process_with_llm

        output_file = tmp_path / "output.md"
        markdown = "# Test\n\nContent here."

        # Create a processor that raises an exception
        failing_processor = MagicMock()
        failing_processor.process_document = AsyncMock(
            side_effect=Exception("LLM API Error")
        )

        result_md, cost, usage = await process_with_llm(
            markdown=markdown,
            source="test.md",
            cfg=markitai_config,
            output_file=output_file,
            processor=failing_processor,
        )

        # Should return original markdown on error
        assert result_md == markdown
        assert cost == 0.0
        assert usage == {}

    async def test_process_with_llm_removes_nonexistent_images(
        self, tmp_path: Path, markitai_config: MarkitaiConfig, mock_llm_processor
    ):
        """Test that nonexistent images are removed when assets dir exists."""
        from markitai.cli.processors.llm import process_with_llm

        output_file = tmp_path / "output.md"
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        markdown = "# Test\n\n![Image](assets/missing.png)"

        with (
            patch(
                "markitai.cli.processors.llm.create_llm_processor",
                return_value=mock_llm_processor,
            ),
            patch(
                "markitai.cli.processors.llm.ImageProcessor.remove_hallucinated_images",
                return_value="# Cleaned\n\n![Image](assets/missing.png)",
            ),
            patch(
                "markitai.cli.processors.llm.ImageProcessor.remove_nonexistent_images",
                return_value="# Cleaned",
            ) as mock_remove,
        ):
            await process_with_llm(
                markdown=markdown,
                source="test.md",
                cfg=markitai_config,
                output_file=output_file,
                processor=mock_llm_processor,
            )

        mock_remove.assert_called_once()

    async def test_process_with_llm_preserves_existing_page_images(
        self, tmp_path: Path, markitai_config: MarkitaiConfig
    ):
        """Test that existing page images are not duplicated."""
        from markitai.cli.processors.llm import process_with_llm

        output_file = tmp_path / "output.md"
        markdown = "# Test\n\nContent here."
        page_images = [
            {"page": 1, "name": "page1.png"},
        ]

        # Processor that returns content already containing page images header
        processor = MagicMock()
        processor.process_document = AsyncMock(
            return_value=("# Cleaned\n\nContent", "title: Test\nsource: test.md")
        )
        processor.format_llm_output = MagicMock(
            return_value="---\ntitle: Test\n---\n\n# Cleaned\n\nContent\n\n<!-- Page images for reference -->\n<!-- ![Page 1](screenshots/page1.png) -->"
        )
        processor.get_context_cost = MagicMock(return_value=0.05)
        processor.get_context_usage = MagicMock(return_value={})

        with patch(
            "markitai.cli.processors.llm.ImageProcessor.remove_hallucinated_images",
            return_value="# Cleaned\n\nContent",
        ):
            await process_with_llm(
                markdown=markdown,
                source="test.md",
                cfg=markitai_config,
                output_file=output_file,
                page_images=page_images,
                processor=processor,
            )

        # Check that page images header appears exactly once
        llm_output = output_file.with_suffix(".llm.md")
        content = llm_output.read_text()
        assert content.count("<!-- Page images for reference -->") == 1


# =============================================================================
# Tests for format_standalone_image_markdown
# =============================================================================


class TestFormatStandaloneImageMarkdown:
    """Tests for format_standalone_image_markdown function."""

    def test_basic_format(self, sample_image_analysis):
        """Test basic image markdown formatting."""
        from markitai.cli.processors.llm import format_standalone_image_markdown

        result = format_standalone_image_markdown(
            input_path=Path("test.jpg"),
            analysis=sample_image_analysis,
            image_ref_path="assets/test.jpg",
            include_frontmatter=False,
        )

        assert "# test" in result
        assert "![A test image](assets/test.jpg)" in result
        assert "This is a detailed description" in result

    def test_with_frontmatter(self, sample_image_analysis):
        """Test formatting with frontmatter included."""
        from markitai.cli.processors.llm import format_standalone_image_markdown

        result = format_standalone_image_markdown(
            input_path=Path("photo.png"),
            analysis=sample_image_analysis,
            image_ref_path="assets/photo.png",
            include_frontmatter=True,
        )

        assert "---" in result
        assert "title: photo" in result
        assert "description: A test image" in result
        assert "source: photo.png" in result
        assert "- image" in result

    def test_with_extracted_text(self, sample_image_analysis):
        """Test formatting with extracted text."""
        from markitai.cli.processors.llm import format_standalone_image_markdown

        result = format_standalone_image_markdown(
            input_path=Path("document.jpg"),
            analysis=sample_image_analysis,
            image_ref_path="assets/document.jpg",
            include_frontmatter=False,
        )

        assert "## Extracted Text" in result
        assert "OCR extracted text" in result
        assert "```" in result

    def test_without_extracted_text(self):
        """Test formatting when no text is extracted."""
        from markitai.cli.processors.llm import format_standalone_image_markdown

        analysis = MagicMock()
        analysis.caption = "A photo"
        analysis.description = "A beautiful photo"
        analysis.extracted_text = ""

        result = format_standalone_image_markdown(
            input_path=Path("photo.jpg"),
            analysis=analysis,
            image_ref_path="assets/photo.jpg",
            include_frontmatter=False,
        )

        # Should not have extracted text section
        assert "## Extracted Text" not in result

    def test_delegates_to_workflow_helpers(self, sample_image_analysis):
        """Test that function delegates to workflow helpers."""
        from markitai.cli.processors.llm import format_standalone_image_markdown

        # The function internally imports from workflow.helpers,
        # so we patch at that location
        with patch(
            "markitai.workflow.helpers.format_standalone_image_markdown"
        ) as mock_func:
            mock_func.return_value = "mocked result"

            result = format_standalone_image_markdown(
                input_path=Path("test.jpg"),
                analysis=sample_image_analysis,
                image_ref_path="assets/test.jpg",
                include_frontmatter=True,
            )

            mock_func.assert_called_once_with(
                Path("test.jpg"),
                sample_image_analysis,
                "assets/test.jpg",
                True,
            )
            assert result == "mocked result"


# =============================================================================
# Tests for analyze_images_with_llm
# =============================================================================


class TestAnalyzeImagesWithLLM:
    """Tests for analyze_images_with_llm function."""

    async def test_analyze_images_basic(
        self,
        tmp_path: Path,
        markitai_config: MarkitaiConfig,
        mock_llm_processor,
        sample_image_analysis,
    ):
        """Test basic image analysis."""
        from markitai.cli.processors.llm import analyze_images_with_llm

        # Create test image
        image_path = tmp_path / "assets" / "test.jpg"
        image_path.parent.mkdir()
        image_path.write_bytes(b"fake image data")

        # Create .llm.md file that the function expects
        output_file = tmp_path / "output.md"
        llm_output = output_file.with_suffix(".llm.md")
        llm_output.write_text(
            "---\ntitle: Test\n---\n\n# Content\n\n![](assets/test.jpg)"
        )

        mock_llm_processor.analyze_images_batch = AsyncMock(
            return_value=[sample_image_analysis]
        )

        markdown, cost, usage, analysis_result = await analyze_images_with_llm(
            image_paths=[image_path],
            markdown="# Test\n\n![](assets/test.jpg)",
            output_file=output_file,
            cfg=markitai_config,
            input_path=tmp_path / "source.pdf",
            processor=mock_llm_processor,
        )

        # Should update alt text in markdown
        assert "![A test image]" in markdown

        # Should have analysis result
        assert analysis_result is not None
        assert len(analysis_result.assets) == 1
        assert analysis_result.assets[0]["alt"] == "A test image"

    async def test_analyze_images_standalone_image(
        self,
        tmp_path: Path,
        markitai_config: MarkitaiConfig,
        mock_llm_processor,
        sample_image_analysis,
    ):
        """Test analyzing a standalone image file."""
        from markitai.cli.processors.llm import analyze_images_with_llm

        # Create test image
        image_path = tmp_path / "assets" / "photo.jpg"
        image_path.parent.mkdir()
        image_path.write_bytes(b"fake image data")

        output_file = tmp_path / "photo.md"

        mock_llm_processor.analyze_images_batch = AsyncMock(
            return_value=[sample_image_analysis]
        )

        with patch(
            "markitai.cli.processors.llm.format_standalone_image_markdown",
            return_value="# Formatted standalone image",
        ):
            markdown, cost, usage, analysis_result = await analyze_images_with_llm(
                image_paths=[image_path],
                markdown="![](assets/photo.jpg)",
                output_file=output_file,
                cfg=markitai_config,
                input_path=tmp_path / "photo.jpg",  # Same as image - standalone
                processor=mock_llm_processor,
            )

        # Should write .llm.md file
        llm_output = output_file.with_suffix(".llm.md")
        assert llm_output.exists()

    async def test_analyze_images_empty_list(
        self, tmp_path: Path, markitai_config: MarkitaiConfig, mock_llm_processor
    ):
        """Test analyzing empty image list."""
        from markitai.cli.processors.llm import analyze_images_with_llm

        output_file = tmp_path / "output.md"
        # Create the .llm.md file that the function waits for
        llm_output = output_file.with_suffix(".llm.md")
        llm_output.write_text("# No images")

        mock_llm_processor.analyze_images_batch = AsyncMock(return_value=[])

        markdown, cost, usage, analysis_result = await analyze_images_with_llm(
            image_paths=[],
            markdown="# No images",
            output_file=output_file,
            cfg=markitai_config,
            processor=mock_llm_processor,
        )

        assert markdown == "# No images"
        # No analysis result when no images and desc_enabled
        # (but empty list means no results)
        assert analysis_result is None

    async def test_analyze_images_error_handling(
        self, tmp_path: Path, markitai_config: MarkitaiConfig
    ):
        """Test error handling in image analysis."""
        from markitai.cli.processors.llm import analyze_images_with_llm

        image_path = tmp_path / "assets" / "test.jpg"
        image_path.parent.mkdir()
        image_path.write_bytes(b"fake image data")

        output_file = tmp_path / "output.md"

        # Create processor that fails
        failing_processor = MagicMock()
        failing_processor.analyze_images_batch = AsyncMock(
            side_effect=Exception("Vision API Error")
        )

        markdown, cost, usage, analysis_result = await analyze_images_with_llm(
            image_paths=[image_path],
            markdown="# Test\n\n![](assets/test.jpg)",
            output_file=output_file,
            cfg=markitai_config,
            processor=failing_processor,
        )

        # Should return original markdown on error
        assert markdown == "# Test\n\n![](assets/test.jpg)"
        assert cost == 0.0
        assert analysis_result is None

    async def test_analyze_images_alt_disabled(
        self,
        tmp_path: Path,
        mock_llm_processor,
        sample_image_analysis,
    ):
        """Test image analysis with alt text disabled."""
        from markitai.cli.processors.llm import analyze_images_with_llm

        # Config with alt_enabled=False
        config = MarkitaiConfig(
            llm=LLMConfig(
                enabled=True,
                model_list=[
                    ModelConfig(
                        model_name="default",
                        litellm_params=LiteLLMParams(
                            model="openai/gpt-4o-mini",
                            api_key="test-key",
                        ),
                    ),
                ],
            ),
            image=ImageConfig(alt_enabled=False, desc_enabled=True),
        )

        image_path = tmp_path / "assets" / "test.jpg"
        image_path.parent.mkdir()
        image_path.write_bytes(b"fake image data")

        output_file = tmp_path / "output.md"

        mock_llm_processor.analyze_images_batch = AsyncMock(
            return_value=[sample_image_analysis]
        )

        markdown, cost, usage, analysis_result = await analyze_images_with_llm(
            image_paths=[image_path],
            markdown="# Test\n\n![](assets/test.jpg)",
            output_file=output_file,
            cfg=config,
            processor=mock_llm_processor,
        )

        # Alt text should NOT be updated
        assert "![A test image]" not in markdown
        assert "![](assets/test.jpg)" in markdown

    async def test_analyze_images_desc_disabled(
        self,
        tmp_path: Path,
        mock_llm_processor,
        sample_image_analysis,
    ):
        """Test image analysis with descriptions disabled."""
        from markitai.cli.processors.llm import analyze_images_with_llm

        # Config with desc_enabled=False
        config = MarkitaiConfig(
            llm=LLMConfig(
                enabled=True,
                model_list=[
                    ModelConfig(
                        model_name="default",
                        litellm_params=LiteLLMParams(
                            model="openai/gpt-4o-mini",
                            api_key="test-key",
                        ),
                    ),
                ],
            ),
            image=ImageConfig(alt_enabled=True, desc_enabled=False),
        )

        image_path = tmp_path / "assets" / "test.jpg"
        image_path.parent.mkdir()
        image_path.write_bytes(b"fake image data")

        output_file = tmp_path / "output.md"
        llm_output = output_file.with_suffix(".llm.md")
        llm_output.write_text("---\ntitle: Test\n---\n\n![](assets/test.jpg)")

        mock_llm_processor.analyze_images_batch = AsyncMock(
            return_value=[sample_image_analysis]
        )

        markdown, cost, usage, analysis_result = await analyze_images_with_llm(
            image_paths=[image_path],
            markdown="# Test\n\n![](assets/test.jpg)",
            output_file=output_file,
            cfg=config,
            processor=mock_llm_processor,
        )

        # Analysis result should be None when desc_enabled=False
        assert analysis_result is None

    async def test_analyze_images_creates_processor_if_none(
        self,
        tmp_path: Path,
        markitai_config: MarkitaiConfig,
        mock_llm_processor,
        sample_image_analysis,
    ):
        """Test that processor is created if not provided."""
        from markitai.cli.processors.llm import analyze_images_with_llm

        image_path = tmp_path / "assets" / "test.jpg"
        image_path.parent.mkdir()
        image_path.write_bytes(b"fake image data")

        output_file = tmp_path / "output.md"
        # Create the .llm.md file that the function waits for
        llm_output = output_file.with_suffix(".llm.md")
        llm_output.write_text("![](assets/test.jpg)")

        mock_llm_processor.analyze_images_batch = AsyncMock(
            return_value=[sample_image_analysis]
        )

        with patch(
            "markitai.cli.processors.llm.create_llm_processor",
            return_value=mock_llm_processor,
        ) as mock_create:
            await analyze_images_with_llm(
                image_paths=[image_path],
                markdown="# Test\n\n![](assets/test.jpg)",
                output_file=output_file,
                cfg=markitai_config,
                processor=None,
            )

        mock_create.assert_called_once()

    async def test_analyze_images_waits_for_llm_file(
        self,
        tmp_path: Path,
        markitai_config: MarkitaiConfig,
        mock_llm_processor,
        sample_image_analysis,
    ):
        """Test that analysis waits for .llm.md file to exist."""
        from markitai.cli.processors.llm import analyze_images_with_llm

        image_path = tmp_path / "assets" / "test.jpg"
        image_path.parent.mkdir()
        image_path.write_bytes(b"fake image data")

        output_file = tmp_path / "output.md"
        llm_output = output_file.with_suffix(".llm.md")

        mock_llm_processor.analyze_images_batch = AsyncMock(
            return_value=[sample_image_analysis]
        )

        # Create .llm.md file after a short delay
        async def create_llm_file():
            await asyncio.sleep(0.1)
            llm_output.write_text("---\ntitle: Test\n---\n\n![](assets/test.jpg)")

        # Run both concurrently
        task1 = asyncio.create_task(create_llm_file())
        task2 = asyncio.create_task(
            analyze_images_with_llm(
                image_paths=[image_path],
                markdown="# Test\n\n![](assets/test.jpg)",
                output_file=output_file,
                cfg=markitai_config,
                processor=mock_llm_processor,
            )
        )
        await asyncio.gather(task1, task2)

        # Alt text should be updated in .llm.md
        content = llm_output.read_text()
        assert "![A test image]" in content


# =============================================================================
# Tests for enhance_document_with_vision
# =============================================================================


class TestEnhanceDocumentWithVision:
    """Tests for enhance_document_with_vision function."""

    async def test_enhance_basic(
        self, tmp_path: Path, markitai_config: MarkitaiConfig, mock_llm_processor
    ):
        """Test basic document enhancement with vision."""
        from markitai.cli.processors.llm import enhance_document_with_vision

        # Create page images
        screenshots_dir = tmp_path / "screenshots"
        screenshots_dir.mkdir()
        page1 = screenshots_dir / "page1.png"
        page1.write_bytes(b"fake image")

        page_images = [{"page": 1, "path": str(page1)}]
        extracted_text = "# Document\n\nSome extracted text."

        cleaned, frontmatter, cost, usage = await enhance_document_with_vision(
            extracted_text=extracted_text,
            page_images=page_images,
            cfg=markitai_config,
            source="document.pdf",
            processor=mock_llm_processor,
        )

        assert "Enhanced" in cleaned
        assert cost == 0.05

    async def test_enhance_multiple_pages(
        self, tmp_path: Path, markitai_config: MarkitaiConfig, mock_llm_processor
    ):
        """Test enhancement with multiple page images."""
        from markitai.cli.processors.llm import enhance_document_with_vision

        screenshots_dir = tmp_path / "screenshots"
        screenshots_dir.mkdir()

        page_images = []
        for i in range(3):
            page = screenshots_dir / f"page{i + 1}.png"
            page.write_bytes(b"fake image")
            page_images.append({"page": i + 1, "path": str(page)})

        cleaned, frontmatter, cost, usage = await enhance_document_with_vision(
            extracted_text="# Multi-page document",
            page_images=page_images,
            cfg=markitai_config,
            source="doc.pdf",
            processor=mock_llm_processor,
        )

        # Should call enhance with sorted images
        mock_llm_processor.enhance_document_complete.assert_called_once()
        call_args = mock_llm_processor.enhance_document_complete.call_args
        assert len(call_args[0][1]) == 3  # 3 image paths

    async def test_enhance_creates_processor_if_none(
        self, tmp_path: Path, markitai_config: MarkitaiConfig, mock_llm_processor
    ):
        """Test that processor is created if not provided."""
        from markitai.cli.processors.llm import enhance_document_with_vision

        screenshots_dir = tmp_path / "screenshots"
        screenshots_dir.mkdir()
        page1 = screenshots_dir / "page1.png"
        page1.write_bytes(b"fake image")

        page_images = [{"page": 1, "path": str(page1)}]

        with patch(
            "markitai.cli.processors.llm.create_llm_processor",
            return_value=mock_llm_processor,
        ) as mock_create:
            await enhance_document_with_vision(
                extracted_text="# Document",
                page_images=page_images,
                cfg=markitai_config,
                source="doc.pdf",
                processor=None,
            )

        mock_create.assert_called_once()

    async def test_enhance_error_handling(
        self, tmp_path: Path, markitai_config: MarkitaiConfig
    ):
        """Test error handling in document enhancement."""
        from markitai.cli.processors.llm import enhance_document_with_vision

        screenshots_dir = tmp_path / "screenshots"
        screenshots_dir.mkdir()
        page1 = screenshots_dir / "page1.png"
        page1.write_bytes(b"fake image")

        page_images = [{"page": 1, "path": str(page1)}]
        extracted_text = "# Original Document"

        # Create processor that fails
        failing_processor = MagicMock()
        failing_processor.enhance_document_complete = AsyncMock(
            side_effect=Exception("Vision API Error")
        )

        cleaned, frontmatter, cost, usage = await enhance_document_with_vision(
            extracted_text=extracted_text,
            page_images=page_images,
            cfg=markitai_config,
            source="document.pdf",
            processor=failing_processor,
        )

        # Should return original text on error
        assert cleaned == extracted_text
        assert "title: document.pdf" in frontmatter
        assert cost == 0.0
        assert usage == {}

    async def test_enhance_sorts_pages_by_number(
        self, tmp_path: Path, markitai_config: MarkitaiConfig, mock_llm_processor
    ):
        """Test that pages are sorted by page number."""
        from markitai.cli.processors.llm import enhance_document_with_vision

        screenshots_dir = tmp_path / "screenshots"
        screenshots_dir.mkdir()

        # Create pages in non-sequential order
        page_images = []
        for page_num in [3, 1, 2]:
            page = screenshots_dir / f"page{page_num}.png"
            page.write_bytes(b"fake image")
            page_images.append({"page": page_num, "path": str(page)})

        await enhance_document_with_vision(
            extracted_text="# Document",
            page_images=page_images,
            cfg=markitai_config,
            source="doc.pdf",
            processor=mock_llm_processor,
        )

        # Check that images were passed in sorted order
        call_args = mock_llm_processor.enhance_document_complete.call_args
        image_paths = call_args[0][1]
        assert image_paths[0].name == "page1.png"
        assert image_paths[1].name == "page2.png"
        assert image_paths[2].name == "page3.png"


# =============================================================================
# Tests for edge cases and integration
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_process_with_llm_original_markdown_comparison(
        self, tmp_path: Path, markitai_config: MarkitaiConfig, mock_llm_processor
    ):
        """Test that original_markdown is used for hallucination detection."""
        from markitai.cli.processors.llm import process_with_llm

        output_file = tmp_path / "output.md"
        markdown = "# Modified\n\nNew content"
        original_markdown = "# Original\n\nOriginal content with ![Image](url)"

        with patch(
            "markitai.cli.processors.llm.ImageProcessor.remove_hallucinated_images"
        ) as mock_remove:
            mock_remove.return_value = "# Cleaned"
            await process_with_llm(
                markdown=markdown,
                source="test.md",
                cfg=markitai_config,
                output_file=output_file,
                processor=mock_llm_processor,
                original_markdown=original_markdown,
            )

        # Should use original_markdown for comparison
        mock_remove.assert_called_once()
        call_args = mock_remove.call_args
        assert call_args[0][1] == original_markdown

    async def test_analyze_images_updates_llm_file_alt_text(
        self,
        tmp_path: Path,
        markitai_config: MarkitaiConfig,
        mock_llm_processor,
        sample_image_analysis,
    ):
        """Test that .llm.md file alt text is updated."""
        from markitai.cli.processors.llm import analyze_images_with_llm

        image_path = tmp_path / "assets" / "test.jpg"
        image_path.parent.mkdir()
        image_path.write_bytes(b"fake image data")

        output_file = tmp_path / "output.md"
        llm_output = output_file.with_suffix(".llm.md")
        llm_output.write_text("---\ntitle: Test\n---\n\n![old alt](assets/test.jpg)")

        mock_llm_processor.analyze_images_batch = AsyncMock(
            return_value=[sample_image_analysis]
        )

        await analyze_images_with_llm(
            image_paths=[image_path],
            markdown="![old alt](assets/test.jpg)",
            output_file=output_file,
            cfg=markitai_config,
            processor=mock_llm_processor,
        )

        # .llm.md should have updated alt text
        content = llm_output.read_text()
        assert "![A test image](assets/test.jpg)" in content

    def test_backward_compatibility_alias(self):
        """Test that backward compatibility alias exists."""
        from markitai.cli.processors.llm import (
            _format_standalone_image_markdown,
            format_standalone_image_markdown,
        )

        # Both should reference the same function
        assert _format_standalone_image_markdown is format_standalone_image_markdown

    async def test_process_with_llm_missing_page_appended(
        self, tmp_path: Path, markitai_config: MarkitaiConfig
    ):
        """Test that missing page comments are appended."""
        from markitai.cli.processors.llm import process_with_llm

        output_file = tmp_path / "output.md"
        markdown = "# Test\n\nContent here."
        page_images = [
            {"page": 1, "name": "page1.png"},
            {"page": 2, "name": "page2.png"},
        ]

        # Processor that returns content with only page 1
        processor = MagicMock()
        processor.process_document = AsyncMock(
            return_value=("# Cleaned\n\nContent", "title: Test\nsource: test.md")
        )
        processor.format_llm_output = MagicMock(
            return_value="---\ntitle: Test\n---\n\n# Cleaned\n\n<!-- Page images for reference -->\n<!-- ![Page 1](screenshots/page1.png) -->"
        )
        processor.get_context_cost = MagicMock(return_value=0.05)
        processor.get_context_usage = MagicMock(return_value={})

        with patch(
            "markitai.cli.processors.llm.ImageProcessor.remove_hallucinated_images",
            return_value="# Cleaned\n\nContent",
        ):
            await process_with_llm(
                markdown=markdown,
                source="test.md",
                cfg=markitai_config,
                output_file=output_file,
                page_images=page_images,
                processor=processor,
            )

        # Check that missing page 2 is appended
        llm_output = output_file.with_suffix(".llm.md")
        content = llm_output.read_text()
        assert "![Page 1]" in content
        assert "![Page 2]" in content


class TestFunctionSignatures:
    """Tests for function signatures and type annotations."""

    def test_process_with_llm_signature(self):
        """Test process_with_llm function signature."""
        from inspect import signature

        from markitai.cli.processors.llm import process_with_llm

        sig = signature(process_with_llm)
        params = list(sig.parameters.keys())

        assert "markdown" in params
        assert "source" in params
        assert "cfg" in params
        assert "output_file" in params
        assert "page_images" in params
        assert "processor" in params
        assert "original_markdown" in params

    def test_format_standalone_image_markdown_signature(self):
        """Test format_standalone_image_markdown function signature."""
        from inspect import signature

        from markitai.cli.processors.llm import format_standalone_image_markdown

        sig = signature(format_standalone_image_markdown)
        params = list(sig.parameters.keys())

        assert "input_path" in params
        assert "analysis" in params
        assert "image_ref_path" in params
        assert "include_frontmatter" in params

    def test_analyze_images_with_llm_signature(self):
        """Test analyze_images_with_llm function signature."""
        from inspect import signature

        from markitai.cli.processors.llm import analyze_images_with_llm

        sig = signature(analyze_images_with_llm)
        params = list(sig.parameters.keys())

        assert "image_paths" in params
        assert "markdown" in params
        assert "output_file" in params
        assert "cfg" in params
        assert "input_path" in params
        assert "concurrency_limit" in params
        assert "processor" in params

    def test_enhance_document_with_vision_signature(self):
        """Test enhance_document_with_vision function signature."""
        from inspect import signature

        from markitai.cli.processors.llm import enhance_document_with_vision

        sig = signature(enhance_document_with_vision)
        params = list(sig.parameters.keys())

        assert "extracted_text" in params
        assert "page_images" in params
        assert "cfg" in params
        assert "source" in params
        assert "processor" in params


class TestReturnTypes:
    """Tests for return type validation."""

    async def test_process_with_llm_returns_tuple(
        self, tmp_path: Path, markitai_config: MarkitaiConfig, mock_llm_processor
    ):
        """Test that process_with_llm returns correct tuple."""
        from markitai.cli.processors.llm import process_with_llm

        output_file = tmp_path / "output.md"

        with patch(
            "markitai.cli.processors.llm.ImageProcessor.remove_hallucinated_images",
            return_value="# Cleaned",
        ):
            result = await process_with_llm(
                markdown="# Test",
                source="test.md",
                cfg=markitai_config,
                output_file=output_file,
                processor=mock_llm_processor,
            )

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], str)  # markdown
        assert isinstance(result[1], float)  # cost
        assert isinstance(result[2], dict)  # usage

    async def test_analyze_images_with_llm_returns_tuple(
        self,
        tmp_path: Path,
        markitai_config: MarkitaiConfig,
        mock_llm_processor,
        sample_image_analysis,
    ):
        """Test that analyze_images_with_llm returns correct tuple."""
        from markitai.cli.processors.llm import analyze_images_with_llm

        image_path = tmp_path / "assets" / "test.jpg"
        image_path.parent.mkdir()
        image_path.write_bytes(b"fake")

        output_file = tmp_path / "output.md"
        llm_output = output_file.with_suffix(".llm.md")
        llm_output.write_text("![](assets/test.jpg)")

        mock_llm_processor.analyze_images_batch = AsyncMock(
            return_value=[sample_image_analysis]
        )

        result = await analyze_images_with_llm(
            image_paths=[image_path],
            markdown="![](assets/test.jpg)",
            output_file=output_file,
            cfg=markitai_config,
            processor=mock_llm_processor,
        )

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert isinstance(result[0], str)  # markdown
        assert isinstance(result[1], float)  # cost
        assert isinstance(result[2], dict)  # usage
        # result[3] is ImageAnalysisResult or None

    async def test_enhance_document_with_vision_returns_tuple(
        self, tmp_path: Path, markitai_config: MarkitaiConfig, mock_llm_processor
    ):
        """Test that enhance_document_with_vision returns correct tuple."""
        from markitai.cli.processors.llm import enhance_document_with_vision

        screenshots_dir = tmp_path / "screenshots"
        screenshots_dir.mkdir()
        page1 = screenshots_dir / "page1.png"
        page1.write_bytes(b"fake")

        result = await enhance_document_with_vision(
            extracted_text="# Doc",
            page_images=[{"page": 1, "path": str(page1)}],
            cfg=markitai_config,
            source="doc.pdf",
            processor=mock_llm_processor,
        )

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert isinstance(result[0], str)  # cleaned_markdown
        assert isinstance(result[1], str)  # frontmatter
        assert isinstance(result[2], float)  # cost
        assert isinstance(result[3], dict)  # usage
