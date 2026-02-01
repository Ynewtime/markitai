"""Tests for workflow/single.py module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.workflow.single import ImageAnalysisResult, SingleFileWorkflow


class TestImageAnalysisResult:
    """Tests for ImageAnalysisResult dataclass."""

    def test_basic_creation(self):
        """Test basic creation with required fields."""
        result = ImageAnalysisResult(
            source_file="/path/to/file.pdf",
            assets=[{"asset": "/path/to/image.png", "alt": "Test"}],
        )
        assert result.source_file == "/path/to/file.pdf"
        assert len(result.assets) == 1

    def test_empty_assets(self):
        """Test creation with empty assets."""
        result = ImageAnalysisResult(source_file="file.pdf", assets=[])
        assert result.assets == []

    def test_multiple_assets(self):
        """Test creation with multiple assets."""
        assets = [
            {"asset": "img1.png", "alt": "Image 1"},
            {"asset": "img2.png", "alt": "Image 2"},
        ]
        result = ImageAnalysisResult(source_file="doc.pdf", assets=assets)
        assert len(result.assets) == 2


class TestSingleFileWorkflow:
    """Tests for SingleFileWorkflow class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock()
        config.llm.concurrency = 5
        config.image.alt_enabled = True
        config.image.desc_enabled = True
        config.cache.no_cache = False
        config.cache.no_cache_patterns = []
        return config

    @pytest.fixture
    def mock_processor(self):
        """Create a mock LLM processor."""
        processor = MagicMock()
        processor.get_context_cost = MagicMock(return_value=0.05)
        processor.get_context_usage = MagicMock(
            return_value={"gpt-4": {"requests": 1, "input_tokens": 100}}
        )
        return processor

    def test_init_with_processor(self, mock_config, mock_processor):
        """Test initialization with existing processor."""
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)
        assert workflow._processor is mock_processor

    def test_init_without_processor(self, mock_config):
        """Test initialization without processor (lazy creation)."""
        workflow = SingleFileWorkflow(mock_config)
        assert workflow._processor is None

    def test_llm_cost_initial(self, mock_config, mock_processor):
        """Test initial LLM cost is zero."""
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)
        assert workflow._llm_cost == 0.0

    def test_llm_usage_initial(self, mock_config, mock_processor):
        """Test initial LLM usage is empty."""
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)
        assert workflow._llm_usage == {}

    def test_processor_property_creates_on_demand(self, mock_config):
        """Test that processor property creates processor on demand."""
        workflow = SingleFileWorkflow(mock_config)

        with patch("markitai.workflow.helpers.create_llm_processor") as mock_create:
            mock_processor = MagicMock()
            mock_create.return_value = mock_processor

            # Access processor property
            processor = workflow.processor

            # Should have created the processor
            mock_create.assert_called_once()
            assert processor is mock_processor

    def test_merge_usage(self, mock_config, mock_processor):
        """Test _merge_usage method."""
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)

        # Add some usage
        workflow._merge_usage(
            {
                "gpt-4": {
                    "requests": 1,
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.01,
                }
            }
        )

        assert "gpt-4" in workflow._llm_usage
        assert workflow._llm_usage["gpt-4"]["requests"] == 1

        # Add more usage
        workflow._merge_usage(
            {
                "gpt-4": {
                    "requests": 2,
                    "input_tokens": 200,
                    "output_tokens": 100,
                    "cost_usd": 0.02,
                }
            }
        )

        assert workflow._llm_usage["gpt-4"]["requests"] == 3


class TestSingleFileWorkflowProcessDocument:
    """Tests for SingleFileWorkflow.process_document_with_llm method."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock()
        config.llm.concurrency = 5
        config.image.alt_enabled = True
        config.image.desc_enabled = True
        config.cache.no_cache = False
        config.cache.no_cache_patterns = []
        config.llm.document_mode = "process"
        return config

    @pytest.fixture
    def mock_processor(self):
        """Create a mock LLM processor."""
        processor = MagicMock()
        processor.process_document = AsyncMock()
        processor.process_document.return_value = MagicMock(
            frontmatter="title: Test\nsource: doc.pdf",
            markdown="# Processed Content",
            llm_usage={"gpt-4": {"requests": 1}},
        )
        processor.get_context_cost = MagicMock(return_value=0.05)
        processor.get_context_usage = MagicMock(return_value={})
        return processor

    @pytest.mark.asyncio
    async def test_process_document_basic(
        self, mock_config, mock_processor, tmp_path: Path
    ):
        """Test basic document processing."""
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)

        output_file = tmp_path / "output.md"
        output_file.write_text("# Original Content")

        result, cost, usage = await workflow.process_document_with_llm(
            markdown="# Original Content",
            source="doc.pdf",
            output_file=output_file,
        )

        assert mock_processor.process_document.called
        assert cost >= 0


class TestSingleFileWorkflowAnalyzeImages:
    """Tests for SingleFileWorkflow.analyze_images method."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock()
        config.llm.concurrency = 2
        config.image.alt_enabled = True
        config.image.desc_enabled = True
        config.cache.no_cache = False
        config.cache.no_cache_patterns = []
        return config

    @pytest.fixture
    def mock_processor(self):
        """Create a mock LLM processor."""
        processor = MagicMock()

        # Mock image analysis
        @dataclass
        class MockImageAnalysis:
            caption: str = "Test caption"
            description: str = "Test description"
            extracted_text: str = ""
            llm_usage: dict[str, Any] | None = None

        processor.analyze_image = AsyncMock(return_value=MockImageAnalysis())
        processor.get_context_cost = MagicMock(return_value=0.02)
        processor.get_context_usage = MagicMock(return_value={"gpt-4": {"requests": 1}})
        return processor

    @pytest.mark.asyncio
    async def test_analyze_images_empty_list(self, mock_config, mock_processor):
        """Test analyzing empty image list returns original markdown."""
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)

        markdown, cost, usage, result = await workflow.analyze_images(
            markdown="# Content",
            image_paths=[],
            output_file=Path("/tmp/test.md"),
        )

        # With empty image list, markdown should be unchanged
        assert markdown == "# Content"
        # Result should be None (no images to analyze)
        assert result is None
        # Cost and usage come from processor's context tracking
        # (may be non-zero if processor tracks any overhead)

    @pytest.mark.asyncio
    async def test_analyze_images_with_images(
        self, mock_config, mock_processor, tmp_path: Path
    ):
        """Test analyzing images."""
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)

        # Create test image file
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        image_file = assets_dir / "test.png"
        image_file.write_bytes(b"fake image data")

        output_file = tmp_path / "output.md"
        output_file.write_text("# Content\n\n![](assets/test.png)")

        markdown, cost, usage, result = await workflow.analyze_images(
            markdown="# Content\n\n![](assets/test.png)",
            image_paths=[image_file],
            output_file=output_file,
        )

        # Should have called analyze_image
        assert mock_processor.analyze_image.called
        assert cost > 0

    @pytest.mark.asyncio
    async def test_analyze_images_standalone_image(
        self, mock_config, mock_processor, tmp_path: Path
    ):
        """Test analyzing a standalone image file."""
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)

        # Create test image file
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        image_file = assets_dir / "standalone.png"
        image_file.write_bytes(b"fake image data")

        input_path = tmp_path / "standalone.png"
        input_path.write_bytes(b"fake image data")

        output_file = tmp_path / "standalone.png.md"
        output_file.write_text("# standalone\n\n![](assets/standalone.png)")

        markdown, cost, usage, result = await workflow.analyze_images(
            markdown="# standalone\n\n![](assets/standalone.png)",
            image_paths=[image_file],
            output_file=output_file,
            input_path=input_path,  # Pass original image as input
        )

        # Should have called analyze_image
        assert mock_processor.analyze_image.called

        # For standalone images with desc_enabled, result should have data
        if mock_config.image.desc_enabled:
            # Check that .llm.md was created with rich content
            llm_file = output_file.with_suffix(".llm.md")
            if llm_file.exists():
                content = llm_file.read_text()
                assert "---" in content  # Has frontmatter


class TestSingleFileWorkflowEnhanceWithVision:
    """Tests for SingleFileWorkflow.enhance_with_vision method."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock()
        config.llm.concurrency = 5
        config.image.alt_enabled = True
        config.image.desc_enabled = True
        config.cache.no_cache = False
        config.cache.no_cache_patterns = []
        return config

    @pytest.fixture
    def mock_processor(self):
        """Create a mock LLM processor."""
        processor = MagicMock()
        # enhance_with_vision calls enhance_document_complete, not enhance_document_with_vision
        processor.enhance_document_complete = AsyncMock(
            return_value=(
                "# Enhanced Content",  # cleaned_content
                "title: Enhanced\nmarkitai_processed: 2026-01-01",  # frontmatter
            )
        )
        processor.get_context_cost = MagicMock(return_value=0.10)
        processor.get_context_usage = MagicMock(return_value={"gpt-4": {"requests": 1}})
        return processor

    @pytest.mark.asyncio
    async def test_enhance_with_vision(
        self, mock_config, mock_processor, tmp_path: Path
    ):
        """Test document enhancement with vision."""
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)

        # Create temp image files
        img1 = tmp_path / "page1.png"
        img1.write_bytes(b"fake image 1")
        img2 = tmp_path / "page2.png"
        img2.write_bytes(b"fake image 2")

        page_images = [
            {"path": str(img1), "page": 1},
            {"path": str(img2), "page": 2},
        ]

        markdown, frontmatter, cost, usage = await workflow.enhance_with_vision(
            extracted_text="Original text",
            page_images=page_images,
            source="doc.pdf",
        )

        assert mock_processor.enhance_document_complete.called
        assert "Enhanced" in frontmatter
        assert "Enhanced Content" in markdown

    @pytest.mark.asyncio
    async def test_enhance_with_vision_sorts_by_page(
        self, mock_config, mock_processor, tmp_path: Path
    ):
        """Test that page images are sorted by page number."""
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)

        # Create temp image files
        img1 = tmp_path / "page1.png"
        img1.write_bytes(b"fake image 1")
        img2 = tmp_path / "page2.png"
        img2.write_bytes(b"fake image 2")
        img3 = tmp_path / "page3.png"
        img3.write_bytes(b"fake image 3")

        # Provide in unsorted order
        page_images = [
            {"path": str(img3), "page": 3},
            {"path": str(img1), "page": 1},
            {"path": str(img2), "page": 2},
        ]

        await workflow.enhance_with_vision(
            extracted_text="Original text",
            page_images=page_images,
            source="doc.pdf",
        )

        # Verify enhance_document_complete was called with sorted paths
        call_args = mock_processor.enhance_document_complete.call_args
        image_paths = call_args[0][1]  # Second positional arg
        assert len(image_paths) == 3
        assert image_paths[0].name == "page1.png"
        assert image_paths[1].name == "page2.png"
        assert image_paths[2].name == "page3.png"

    @pytest.mark.asyncio
    async def test_enhance_with_vision_handles_error(
        self, mock_config, mock_processor, tmp_path: Path
    ):
        """Test error handling in enhance_with_vision."""
        # Make processor raise an error
        mock_processor.enhance_document_complete = AsyncMock(
            side_effect=RuntimeError("LLM API error")
        )
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)

        img1 = tmp_path / "page1.png"
        img1.write_bytes(b"fake image")

        page_images = [{"path": str(img1), "page": 1}]

        markdown, frontmatter, cost, usage = await workflow.enhance_with_vision(
            extracted_text="Original text",
            page_images=page_images,
            source="doc.pdf",
        )

        # Should return original text on error
        assert markdown == "Original text"
        assert "doc.pdf" in frontmatter
        assert cost == 0.0
        assert usage == {}


class TestSingleFileWorkflowExtractFromScreenshots:
    """Tests for SingleFileWorkflow.extract_from_screenshots method."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock()
        config.llm.concurrency = 5
        config.image.alt_enabled = True
        config.image.desc_enabled = True
        config.cache.no_cache = False
        config.cache.no_cache_patterns = []
        return config

    @pytest.fixture
    def mock_processor(self):
        """Create a mock LLM processor."""
        processor = MagicMock()
        processor.extract_from_screenshot = AsyncMock(
            return_value=("# Page content", "title: Test")
        )
        processor.get_context_cost = MagicMock(return_value=0.05)
        processor.get_context_usage = MagicMock(return_value={"gpt-4": {"requests": 1}})
        return processor

    @pytest.mark.asyncio
    async def test_extract_from_screenshots_empty_list(
        self, mock_config, mock_processor
    ):
        """Test extraction with empty page images list."""
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)

        markdown, frontmatter, cost, usage = await workflow.extract_from_screenshots(
            page_images=[],
            source="doc.pdf",
        )

        assert markdown == ""
        assert "doc.pdf" in frontmatter
        assert cost == 0.0
        assert usage == {}

    @pytest.mark.asyncio
    async def test_extract_from_screenshots_single_page(
        self, mock_config, mock_processor, tmp_path: Path
    ):
        """Test extraction from a single page."""
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)

        img1 = tmp_path / "page1.png"
        img1.write_bytes(b"fake image")

        page_images = [{"path": str(img1), "page": 1}]

        markdown, frontmatter, cost, usage = await workflow.extract_from_screenshots(
            page_images=page_images,
            source="doc.pdf",
        )

        assert mock_processor.extract_from_screenshot.called
        assert "Page 1" in markdown
        assert "Page content" in markdown

    @pytest.mark.asyncio
    async def test_extract_from_screenshots_multiple_pages(
        self, mock_config, mock_processor, tmp_path: Path
    ):
        """Test extraction from multiple pages."""
        # Make each call return different content
        call_count = [0]

        async def mock_extract(image_path, context=None):
            call_count[0] += 1
            return (f"Content from page {call_count[0]}", "")

        mock_processor.extract_from_screenshot = mock_extract
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)

        img1 = tmp_path / "page1.png"
        img1.write_bytes(b"fake image 1")
        img2 = tmp_path / "page2.png"
        img2.write_bytes(b"fake image 2")
        img3 = tmp_path / "page3.png"
        img3.write_bytes(b"fake image 3")

        page_images = [
            {"path": str(img2), "page": 2},
            {"path": str(img1), "page": 1},
            {"path": str(img3), "page": 3},
        ]

        markdown, frontmatter, cost, usage = await workflow.extract_from_screenshots(
            page_images=page_images,
            source="doc.pdf",
        )

        # Should contain all page markers (sorted order)
        assert "Page 1" in markdown
        assert "Page 2" in markdown
        assert "Page 3" in markdown

    @pytest.mark.asyncio
    async def test_extract_from_screenshots_handles_error(
        self, mock_config, mock_processor, tmp_path: Path
    ):
        """Test error handling in extract_from_screenshots."""
        mock_processor.extract_from_screenshot = AsyncMock(
            side_effect=RuntimeError("Vision API error")
        )
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)

        img1 = tmp_path / "page1.png"
        img1.write_bytes(b"fake image")

        page_images = [{"path": str(img1), "page": 1}]

        markdown, frontmatter, cost, usage = await workflow.extract_from_screenshots(
            page_images=page_images,
            source="doc.pdf",
        )

        # Should return empty on error
        assert markdown == ""
        assert "doc.pdf" in frontmatter
        assert cost == 0.0


class TestSingleFileWorkflowNoCacheOptions:
    """Tests for SingleFileWorkflow cache options."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock()
        config.llm.concurrency = 5
        config.image.alt_enabled = False
        config.image.desc_enabled = False
        config.cache.no_cache = False
        config.cache.no_cache_patterns = []
        config.model_copy = MagicMock(return_value=config)
        return config

    def test_init_with_no_cache(self, mock_config):
        """Test initialization with no_cache option."""
        workflow = SingleFileWorkflow(mock_config, no_cache=True)
        assert workflow._no_cache is True

    def test_init_with_no_cache_patterns(self, mock_config):
        """Test initialization with no_cache_patterns."""
        patterns = ["*.pdf", "docs/*"]
        workflow = SingleFileWorkflow(mock_config, no_cache_patterns=patterns)
        assert workflow._no_cache_patterns == patterns

    def test_processor_applies_no_cache_to_config(self, mock_config):
        """Test that processor creation applies no_cache settings."""
        with patch("markitai.workflow.helpers.create_llm_processor") as mock_create:
            mock_processor = MagicMock()
            mock_create.return_value = mock_processor

            workflow = SingleFileWorkflow(
                mock_config,
                no_cache=True,
                no_cache_patterns=["*.pdf"],
            )

            # Access processor to trigger creation
            _ = workflow.processor

            # Verify create_llm_processor was called
            mock_create.assert_called_once()
            # The temp_config should have no_cache settings applied
            call_config = mock_create.call_args[0][0]
            assert call_config.cache.no_cache is True
            assert call_config.cache.no_cache_patterns == ["*.pdf"]


class TestSingleFileWorkflowAnalyzeImagesAltTextUpdate:
    """Tests for analyze_images alt text update behavior."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock()
        config.llm.concurrency = 2
        config.image.alt_enabled = True
        config.image.desc_enabled = False
        config.cache.no_cache = False
        config.cache.no_cache_patterns = []
        return config

    @pytest.fixture
    def mock_processor(self):
        """Create a mock LLM processor."""
        processor = MagicMock()

        @dataclass
        class MockImageAnalysis:
            caption: str = "A beautiful sunset"
            description: str = "The image shows a sunset"
            extracted_text: str = ""
            llm_usage: dict[str, Any] | None = None

        processor.analyze_image = AsyncMock(return_value=MockImageAnalysis())
        processor.get_context_cost = MagicMock(return_value=0.02)
        processor.get_context_usage = MagicMock(return_value={"gpt-4": {"requests": 1}})
        return processor

    @pytest.mark.asyncio
    async def test_updates_alt_text_in_markdown(
        self, mock_config, mock_processor, tmp_path: Path
    ):
        """Test that alt text is updated in returned markdown."""
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)

        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        image_file = assets_dir / "sunset.png"
        image_file.write_bytes(b"fake image")

        output_file = tmp_path / "output.md"
        output_file.write_text("# Content\n\n![](assets/sunset.png)")

        markdown, cost, usage, result = await workflow.analyze_images(
            markdown="# Content\n\n![](assets/sunset.png)",
            image_paths=[image_file],
            output_file=output_file,
        )

        # Alt text should be updated in the returned markdown
        assert "![A beautiful sunset](assets/sunset.png)" in markdown

    @pytest.mark.asyncio
    async def test_handles_failed_analysis(
        self, mock_config, mock_processor, tmp_path: Path
    ):
        """Test handling when image analysis fails."""
        # Make analysis return None to simulate failure
        mock_processor.analyze_image = AsyncMock(return_value=None)
        workflow = SingleFileWorkflow(mock_config, processor=mock_processor)

        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        image_file = assets_dir / "broken.png"
        image_file.write_bytes(b"fake")

        output_file = tmp_path / "output.md"
        output_file.write_text("# Content\n\n![](assets/broken.png)")

        markdown, cost, usage, result = await workflow.analyze_images(
            markdown="# Content\n\n![](assets/broken.png)",
            image_paths=[image_file],
            output_file=output_file,
        )

        # Should still return result with default values
        assert result is not None
        assert len(result.assets) == 1
        # Default caption should be used
        assert result.assets[0]["alt"] == "Image"


class TestWorkflowResult:
    """Tests for WorkflowResult dataclass."""

    def test_basic_creation(self):
        """Test basic WorkflowResult creation."""
        from markitai.workflow.single import WorkflowResult

        result = WorkflowResult(markdown="# Test")
        assert result.markdown == "# Test"
        assert result.llm_cost == 0.0
        assert result.llm_usage == {}
        assert result.image_analysis is None

    def test_with_all_fields(self):
        """Test WorkflowResult with all fields."""
        from markitai.workflow.single import ImageAnalysisResult, WorkflowResult

        analysis = ImageAnalysisResult(
            source_file="test.pdf",
            assets=[{"asset": "img.png", "alt": "Test"}],
        )
        result = WorkflowResult(
            markdown="# Test",
            llm_cost=0.05,
            llm_usage={"gpt-4": {"requests": 1}},
            image_analysis=analysis,
        )

        assert result.llm_cost == 0.05
        assert result.llm_usage["gpt-4"]["requests"] == 1
        assert result.image_analysis.source_file == "test.pdf"
