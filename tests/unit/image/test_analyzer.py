"""Tests for image analyzer module."""

import asyncio
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from markit.image.analyzer import (
    IMAGE_ANALYSIS_PROMPT,
    IMAGE_ANALYSIS_PROMPT_EN,
    IMAGE_ANALYSIS_PROMPT_ZH,
    ImageAnalysis,
    ImageAnalyzer,
    KnowledgeGraphMeta,
    get_image_analysis_prompt,
)
from markit.image.compressor import CompressedImage
from markit.llm.base import LLMResponse


class TestKnowledgeGraphMeta:
    """Tests for KnowledgeGraphMeta dataclass."""

    def test_creation(self):
        """Test creating KnowledgeGraphMeta."""
        meta = KnowledgeGraphMeta(
            entities=["Entity1", "Entity2"],
            relationships=["Entity1 -> uses -> Entity2"],
            topics=["topic1", "topic2"],
            domain="technology",
        )

        assert meta.entities == ["Entity1", "Entity2"]
        assert meta.relationships == ["Entity1 -> uses -> Entity2"]
        assert meta.topics == ["topic1", "topic2"]
        assert meta.domain == "technology"

    def test_creation_empty_lists(self):
        """Test creating KnowledgeGraphMeta with empty lists."""
        meta = KnowledgeGraphMeta(
            entities=[],
            relationships=[],
            topics=[],
            domain=None,
        )

        assert meta.entities == []
        assert meta.relationships == []
        assert meta.topics == []
        assert meta.domain is None


class TestImageAnalysis:
    """Tests for ImageAnalysis dataclass."""

    def test_creation_basic(self):
        """Test creating basic ImageAnalysis."""
        analysis = ImageAnalysis(
            alt_text="A test image",
            detailed_description="This is a detailed description.",
            detected_text="Some text",
            image_type="diagram",
        )

        assert analysis.alt_text == "A test image"
        assert analysis.detailed_description == "This is a detailed description."
        assert analysis.detected_text == "Some text"
        assert analysis.image_type == "diagram"
        assert analysis.knowledge_meta is None

    def test_creation_with_knowledge_meta(self):
        """Test creating ImageAnalysis with knowledge metadata."""
        meta = KnowledgeGraphMeta(
            entities=["API Gateway"],
            relationships=["Service -> calls -> API Gateway"],
            topics=["microservices"],
            domain="technology",
        )

        analysis = ImageAnalysis(
            alt_text="Architecture diagram",
            detailed_description="Shows microservices architecture.",
            detected_text=None,
            image_type="diagram",
            knowledge_meta=meta,
        )

        assert analysis.knowledge_meta is not None
        assert analysis.knowledge_meta.entities == ["API Gateway"]


class TestGetImageAnalysisPrompt:
    """Tests for get_image_analysis_prompt function."""

    def test_chinese_prompt(self):
        """Test getting Chinese prompt."""
        prompt = get_image_analysis_prompt("zh")
        assert prompt == IMAGE_ANALYSIS_PROMPT_ZH
        assert "中文" in prompt

    def test_english_prompt(self):
        """Test getting English prompt."""
        prompt = get_image_analysis_prompt("en")
        assert prompt == IMAGE_ANALYSIS_PROMPT_EN
        assert "English" not in prompt or "Analyze this image" in prompt

    def test_default_is_chinese(self):
        """Test default prompt is Chinese."""
        assert IMAGE_ANALYSIS_PROMPT == IMAGE_ANALYSIS_PROMPT_ZH

    def test_unknown_language_defaults_to_english(self):
        """Test unknown language defaults to English."""
        prompt = get_image_analysis_prompt("fr")
        assert prompt == IMAGE_ANALYSIS_PROMPT_EN


class TestImageAnalyzerInit:
    """Tests for ImageAnalyzer initialization."""

    def test_init(self):
        """Test initialization with provider manager."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        assert analyzer.provider_manager is mock_manager


class TestImageAnalyzerConvertToSupportedFormat:
    """Tests for _convert_to_supported_format method."""

    def test_supported_format_unchanged(self):
        """Test that supported formats are not converted."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        # Create a PNG image
        img = Image.new("RGB", (50, 50), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        image = CompressedImage(
            data=buffer.getvalue(),
            format="png",
            filename="test.png",
            original_size=5000,
            compressed_size=4000,
            width=50,
            height=50,
        )

        data, fmt = analyzer._convert_to_supported_format(image)

        assert fmt == "png"
        assert data == buffer.getvalue()

    def test_unsupported_format_converted_to_png(self):
        """Test that unsupported formats are converted to PNG."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        # Create a GIF image
        img = Image.new("P", (50, 50))
        buffer = io.BytesIO()
        img.save(buffer, format="GIF")

        image = CompressedImage(
            data=buffer.getvalue(),
            format="gif",
            filename="test.gif",
            original_size=5000,
            compressed_size=4000,
            width=50,
            height=50,
        )

        data, fmt = analyzer._convert_to_supported_format(image)

        assert fmt == "png"
        assert data != buffer.getvalue()  # Should be converted

    def test_conversion_error_returns_original(self):
        """Test that conversion error returns original data."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        image = CompressedImage(
            data=b"invalid image data",
            format="gif",
            filename="invalid.gif",
            original_size=100,
            compressed_size=100,
            width=50,
            height=50,
        )

        data, fmt = analyzer._convert_to_supported_format(image)

        assert fmt == "gif"
        assert data == b"invalid image data"

    def test_animated_gif_uses_first_frame(self):
        """Test that animated GIF uses first frame."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        # Create an animated GIF
        frames = [Image.new("P", (50, 50), color=i) for i in range(3)]
        buffer = io.BytesIO()
        frames[0].save(
            buffer,
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=100,
        )

        image = CompressedImage(
            data=buffer.getvalue(),
            format="gif",
            filename="animated.gif",
            original_size=5000,
            compressed_size=4000,
            width=50,
            height=50,
        )

        data, fmt = analyzer._convert_to_supported_format(image)

        assert fmt == "png"
        # Verify it's a valid PNG
        img = Image.open(io.BytesIO(data))
        assert img.format == "PNG"


class TestImageAnalyzerParseResponse:
    """Tests for _parse_response method."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        json_content = json.dumps(
            {
                "alt_text": "A test image",
                "detailed_description": "This is a detailed description.",
                "detected_text": "Some text",
                "image_type": "diagram",
            }
        )

        response = MagicMock(spec=LLMResponse)
        response.content = json_content

        result = analyzer._parse_response(response)

        assert result.alt_text == "A test image"
        assert result.detailed_description == "This is a detailed description."
        assert result.detected_text == "Some text"
        assert result.image_type == "diagram"

    def test_parse_json_with_code_block(self):
        """Test parsing JSON wrapped in code block."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        json_content = """```json
        {
            "alt_text": "Test image",
            "detailed_description": "Description.",
            "detected_text": null,
            "image_type": "photo"
        }
        ```"""

        response = MagicMock(spec=LLMResponse)
        response.content = json_content

        result = analyzer._parse_response(response)

        assert result.alt_text == "Test image"
        assert result.image_type == "photo"

    def test_parse_json_with_extra_text(self):
        """Test parsing JSON with extra text around it."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        json_content = """Here's the analysis:
        {"alt_text": "Image", "detailed_description": "Desc", "detected_text": null, "image_type": "other"}
        Hope this helps!"""

        response = MagicMock(spec=LLMResponse)
        response.content = json_content

        result = analyzer._parse_response(response)

        assert result.alt_text == "Image"

    def test_parse_json_with_knowledge_meta(self):
        """Test parsing JSON with knowledge metadata."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        json_content = json.dumps(
            {
                "alt_text": "Architecture diagram",
                "detailed_description": "Shows system architecture.",
                "detected_text": "API Gateway",
                "image_type": "diagram",
                "knowledge_meta": {
                    "entities": ["API Gateway", "User Service"],
                    "relationships": ["User -> calls -> API Gateway"],
                    "topics": ["microservices", "architecture"],
                    "domain": "technology",
                },
            }
        )

        response = MagicMock(spec=LLMResponse)
        response.content = json_content

        result = analyzer._parse_response(response)

        assert result.knowledge_meta is not None
        assert result.knowledge_meta.entities == ["API Gateway", "User Service"]
        assert result.knowledge_meta.domain == "technology"

    def test_parse_invalid_json_fallback(self):
        """Test fallback for invalid JSON."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        response = MagicMock(spec=LLMResponse)
        response.content = "This is not valid JSON at all"

        result = analyzer._parse_response(response)

        assert result.alt_text == "Image"
        assert result.image_type == "other"


class TestImageAnalyzerFixJsonString:
    """Tests for _fix_json_string method."""

    def test_fix_trailing_comma(self):
        """Test fixing trailing comma."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        json_str = '{"key": "value",}'
        fixed = analyzer._fix_json_string(json_str)

        assert fixed == '{"key": "value"}'

    def test_fix_trailing_comma_in_array(self):
        """Test fixing trailing comma in array."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        json_str = '["a", "b",]'
        fixed = analyzer._fix_json_string(json_str)

        assert fixed == '["a", "b"]'

    def test_fix_missing_closing_brace(self):
        """Test fixing missing closing brace."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        json_str = '{"key": "value"'
        fixed = analyzer._fix_json_string(json_str)

        assert fixed == '{"key": "value"}'

    def test_fix_missing_closing_bracket(self):
        """Test fixing missing closing bracket."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        json_str = '{"arr": ["a", "b"'
        fixed = analyzer._fix_json_string(json_str)

        assert fixed == '{"arr": ["a", "b"]}'

    def test_empty_string_unchanged(self):
        """Test empty string is unchanged."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        assert analyzer._fix_json_string("") == ""

    def test_valid_json_unchanged(self):
        """Test valid JSON is unchanged."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        json_str = '{"key": "value"}'
        fixed = analyzer._fix_json_string(json_str)

        assert fixed == json_str


class TestImageAnalyzerEnsureString:
    """Tests for _ensure_string method."""

    def test_string_unchanged(self):
        """Test string is returned unchanged."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        assert analyzer._ensure_string("hello") == "hello"

    def test_none_returns_none(self):
        """Test None returns None."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        assert analyzer._ensure_string(None) is None

    def test_list_joined(self):
        """Test list is joined with newlines."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        result = analyzer._ensure_string(["line1", "line2", "line3"])
        assert result == "line1\nline2\nline3"

    def test_number_converted(self):
        """Test number is converted to string."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        assert analyzer._ensure_string(42) == "42"


class TestImageAnalyzerEnsureStringList:
    """Tests for _ensure_string_list method."""

    def test_list_returned(self):
        """Test list is returned with strings."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        result = analyzer._ensure_string_list(["a", "b", 3])
        assert result == ["a", "b", "3"]

    def test_none_returns_empty_list(self):
        """Test None returns empty list."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        assert analyzer._ensure_string_list(None) == []

    def test_string_wrapped_in_list(self):
        """Test string is wrapped in list."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        assert analyzer._ensure_string_list("single") == ["single"]

    def test_number_wrapped_in_list(self):
        """Test number is wrapped in list."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        assert analyzer._ensure_string_list(42) == ["42"]


class TestImageAnalyzerGenerateDescriptionMd:
    """Tests for _generate_description_md method."""

    def test_basic_description(self):
        """Test generating basic description markdown."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        analysis = ImageAnalysis(
            alt_text="Test image",
            detailed_description="This is a test.",
            detected_text=None,
            image_type="photo",
        )

        result = analyzer._generate_description_md("test.png", analysis)

        assert "source_image: test.png" in result
        assert "image_type: photo" in result
        assert "# Image Description" in result
        assert "## Alt Text" in result
        assert "Test image" in result
        assert "## Detailed Description" in result
        assert "This is a test." in result

    def test_description_with_detected_text(self):
        """Test description includes detected text."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        analysis = ImageAnalysis(
            alt_text="Screenshot",
            detailed_description="A screenshot.",
            detected_text="Hello World",
            image_type="screenshot",
        )

        result = analyzer._generate_description_md("screen.png", analysis)

        assert "## Detected Text" in result
        assert "Hello World" in result

    def test_description_with_knowledge_meta(self):
        """Test description includes knowledge metadata."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        meta = KnowledgeGraphMeta(
            entities=["Entity1", "Entity2"],
            relationships=["Entity1 -> uses -> Entity2"],
            topics=["topic1", "topic2"],
            domain="technology",
        )

        analysis = ImageAnalysis(
            alt_text="Diagram",
            detailed_description="Architecture diagram.",
            detected_text=None,
            image_type="diagram",
            knowledge_meta=meta,
        )

        result = analyzer._generate_description_md("diagram.png", analysis)

        assert "entities: [Entity1, Entity2]" in result
        assert "topics: [topic1, topic2]" in result
        assert "domain: technology" in result
        assert "## Relationships" in result
        assert "- Entity1 -> uses -> Entity2" in result


class TestImageAnalyzerAnalyze:
    """Tests for analyze method."""

    @pytest.mark.asyncio
    async def test_analyze_success(self):
        """Test successful image analysis."""
        mock_manager = MagicMock()
        mock_response = MagicMock(spec=LLMResponse)
        mock_response.content = json.dumps(
            {
                "alt_text": "Test image",
                "detailed_description": "A detailed description.",
                "detected_text": "Some text",
                "image_type": "photo",
            }
        )

        mock_manager.analyze_image_with_fallback = AsyncMock(return_value=mock_response)

        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        # Create test image
        img = Image.new("RGB", (50, 50), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        image = CompressedImage(
            data=buffer.getvalue(),
            format="png",
            filename="test.png",
            original_size=5000,
            compressed_size=4000,
            width=50,
            height=50,
        )

        result = await analyzer.analyze(image)

        assert isinstance(result, ImageAnalysis)
        assert result.alt_text == "Test image"
        assert result.image_type == "photo"

    @pytest.mark.asyncio
    async def test_analyze_with_context(self):
        """Test analysis with context."""
        mock_manager = MagicMock()
        mock_response = MagicMock(spec=LLMResponse)
        mock_response.content = json.dumps(
            {
                "alt_text": "Test",
                "detailed_description": "Description",
                "detected_text": None,
                "image_type": "other",
            }
        )

        mock_manager.analyze_image_with_fallback = AsyncMock(return_value=mock_response)

        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        img = Image.new("RGB", (50, 50), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        image = CompressedImage(
            data=buffer.getvalue(),
            format="png",
            filename="test.png",
            original_size=5000,
            compressed_size=4000,
            width=50,
            height=50,
        )

        await analyzer.analyze(image, context="This is from a document about APIs")

        # Verify context was included in prompt
        call_args = mock_manager.analyze_image_with_fallback.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
        assert "APIs" in prompt

    @pytest.mark.asyncio
    async def test_analyze_failure_returns_fallback(self):
        """Test that analysis failure returns fallback."""
        mock_manager = MagicMock()
        mock_manager.analyze_image_with_fallback = AsyncMock(side_effect=Exception("API error"))

        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        img = Image.new("RGB", (50, 50), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        image = CompressedImage(
            data=buffer.getvalue(),
            format="png",
            filename="test.png",
            original_size=5000,
            compressed_size=4000,
            width=50,
            height=50,
        )

        result = await analyzer.analyze(image)

        assert isinstance(result, ImageAnalysis)
        assert "test.png" in result.alt_text
        assert "failed" in result.detailed_description.lower()

    @pytest.mark.asyncio
    async def test_analyze_with_return_stats(self):
        """Test analysis with return_stats=True."""
        mock_manager = MagicMock()
        mock_response = MagicMock(spec=LLMResponse)
        mock_response.content = json.dumps(
            {
                "alt_text": "Test",
                "detailed_description": "Description",
                "detected_text": None,
                "image_type": "other",
            }
        )
        mock_response.model = "gpt-4"
        mock_response.usage = None

        mock_manager.analyze_image_with_fallback = AsyncMock(return_value=mock_response)

        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        img = Image.new("RGB", (50, 50), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        image = CompressedImage(
            data=buffer.getvalue(),
            format="png",
            filename="test.png",
            original_size=5000,
            compressed_size=4000,
            width=50,
            height=50,
        )

        result = await analyzer.analyze(image, return_stats=True)

        # Should return LLMTaskResultWithStats
        assert hasattr(result, "result")


class TestImageAnalyzerGenerateAltText:
    """Tests for generate_alt_text method."""

    @pytest.mark.asyncio
    async def test_generate_alt_text(self):
        """Test generating alt text."""
        mock_manager = MagicMock()
        mock_response = MagicMock(spec=LLMResponse)
        mock_response.content = json.dumps(
            {
                "alt_text": "A beautiful sunset",
                "detailed_description": "Sunset over the ocean.",
                "detected_text": None,
                "image_type": "photo",
            }
        )

        mock_manager.analyze_image_with_fallback = AsyncMock(return_value=mock_response)

        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        img = Image.new("RGB", (50, 50), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        image = CompressedImage(
            data=buffer.getvalue(),
            format="png",
            filename="sunset.png",
            original_size=5000,
            compressed_size=4000,
            width=50,
            height=50,
        )

        alt_text = await analyzer.generate_alt_text(image)

        assert alt_text == "A beautiful sunset"


class TestImageAnalyzerBatchAnalyze:
    """Tests for batch_analyze method."""

    @pytest.mark.asyncio
    async def test_batch_analyze_success(self):
        """Test batch analyzing multiple images."""
        mock_manager = MagicMock()

        call_count = 0

        async def mock_analyze(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            response = MagicMock(spec=LLMResponse)
            response.content = json.dumps(
                {
                    "alt_text": f"Image {call_count}",
                    "detailed_description": f"Description {call_count}",
                    "detected_text": None,
                    "image_type": "photo",
                }
            )
            return response

        mock_manager.analyze_image_with_fallback = mock_analyze

        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        # Create test images
        images = []
        for i in range(3):
            img = Image.new("RGB", (50, 50), color="red")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")

            images.append(
                CompressedImage(
                    data=buffer.getvalue(),
                    format="png",
                    filename=f"image{i}.png",
                    original_size=5000,
                    compressed_size=4000,
                    width=50,
                    height=50,
                )
            )

        results = await analyzer.batch_analyze(images)

        assert len(results) == 3
        assert all(isinstance(r, ImageAnalysis) for r in results)

    @pytest.mark.asyncio
    async def test_batch_analyze_with_semaphore(self):
        """Test batch analysis with rate limiting semaphore."""
        mock_manager = MagicMock()

        async def mock_analyze(*_args, **_kwargs):
            await asyncio.sleep(0.01)
            response = MagicMock(spec=LLMResponse)
            response.content = json.dumps(
                {
                    "alt_text": "Image",
                    "detailed_description": "Description",
                    "detected_text": None,
                    "image_type": "photo",
                }
            )
            return response

        mock_manager.analyze_image_with_fallback = mock_analyze

        analyzer = ImageAnalyzer(provider_manager=mock_manager)
        semaphore = asyncio.Semaphore(2)

        # Create test images
        images = []
        for i in range(4):
            img = Image.new("RGB", (50, 50), color="red")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")

            images.append(
                CompressedImage(
                    data=buffer.getvalue(),
                    format="png",
                    filename=f"image{i}.png",
                    original_size=5000,
                    compressed_size=4000,
                    width=50,
                    height=50,
                )
            )

        results = await analyzer.batch_analyze(images, semaphore=semaphore)

        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_batch_analyze_handles_exceptions(self):
        """Test batch analysis handles individual failures."""
        mock_manager = MagicMock()

        call_count = 0

        async def mock_analyze(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Analysis failed")
            response = MagicMock(spec=LLMResponse)
            response.content = json.dumps(
                {
                    "alt_text": f"Image {call_count}",
                    "detailed_description": "Description",
                    "detected_text": None,
                    "image_type": "photo",
                }
            )
            return response

        mock_manager.analyze_image_with_fallback = mock_analyze

        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        # Create test images
        images = []
        for i in range(3):
            img = Image.new("RGB", (50, 50), color="red")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")

            images.append(
                CompressedImage(
                    data=buffer.getvalue(),
                    format="png",
                    filename=f"image{i}.png",
                    original_size=5000,
                    compressed_size=4000,
                    width=50,
                    height=50,
                )
            )

        results = await analyzer.batch_analyze(images)

        assert len(results) == 3
        # The second one should be a fallback
        assert "failed" in results[1].detailed_description.lower()


class TestImageAnalyzerWriteAnalysisImmediately:
    """Tests for _write_analysis_immediately method."""

    @pytest.mark.asyncio
    async def test_write_analysis_creates_files(self, tmp_path):
        """Test that write analysis creates image and md files."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        # Create test image data
        img = Image.new("RGB", (50, 50), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = buffer.getvalue()

        analysis = ImageAnalysis(
            alt_text="Test image",
            detailed_description="A test image description.",
            detected_text=None,
            image_type="photo",
        )

        await analyzer._write_analysis_immediately(
            filename="test.png",
            image_data=image_data,
            analysis=analysis,
            output_dir=tmp_path,
        )

        # Check files were created
        assets_dir = tmp_path / "assets"
        assert assets_dir.exists()
        assert (assets_dir / "test.png").exists()
        assert (assets_dir / "test.png.md").exists()

        # Check md content
        md_content = (assets_dir / "test.png.md").read_text()
        assert "Test image" in md_content
        assert "A test image description." in md_content

    @pytest.mark.asyncio
    async def test_write_analysis_handles_error(self, tmp_path):
        """Test that write errors don't fail the analysis."""
        mock_manager = MagicMock()
        analyzer = ImageAnalyzer(provider_manager=mock_manager)

        analysis = ImageAnalysis(
            alt_text="Test",
            detailed_description="Description",
            detected_text=None,
            image_type="photo",
        )

        # Use a path that doesn't exist and can't be created
        with patch("anyio.open_file", side_effect=OSError("Cannot write")):
            # Should not raise
            await analyzer._write_analysis_immediately(
                filename="test.png",
                image_data=b"data",
                analysis=analysis,
                output_dir=tmp_path,
            )
