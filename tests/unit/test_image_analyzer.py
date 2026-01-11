"""Tests for ImageAnalyzer JSON parsing."""

import pytest

from markit.image.analyzer import ImageAnalysis, ImageAnalyzer


class TestFixJsonString:
    """Tests for _fix_json_string method."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer without provider manager (not needed for parsing)."""
        # Create analyzer with mock provider manager
        from unittest.mock import Mock

        return ImageAnalyzer(Mock())

    def test_removes_trailing_comma_before_brace(self, analyzer):
        """Trailing comma before closing brace is removed."""
        json_str = '{"key": "value",}'
        result = analyzer._fix_json_string(json_str)
        assert result == '{"key": "value"}'

    def test_removes_trailing_comma_before_bracket(self, analyzer):
        """Trailing comma before closing bracket is removed."""
        json_str = '["a", "b", "c",]'
        result = analyzer._fix_json_string(json_str)
        assert result == '["a", "b", "c"]'

    def test_removes_nested_trailing_commas(self, analyzer):
        """Trailing commas in nested structures are removed."""
        json_str = '{"list": ["a", "b",], "obj": {"x": 1,},}'
        result = analyzer._fix_json_string(json_str)
        assert result == '{"list": ["a", "b"], "obj": {"x": 1}}'

    def test_fixes_missing_closing_brace(self, analyzer):
        """Missing closing brace is added."""
        json_str = '{"key": "value"'
        result = analyzer._fix_json_string(json_str)
        assert result == '{"key": "value"}'

    def test_fixes_missing_closing_bracket(self, analyzer):
        """Missing closing bracket is added."""
        json_str = '["a", "b"'
        result = analyzer._fix_json_string(json_str)
        assert result == '["a", "b"]'

    def test_fixes_multiple_missing_closures(self, analyzer):
        """Multiple missing closures are added."""
        json_str = '{"list": ["a", "b"'
        result = analyzer._fix_json_string(json_str)
        # Should add ] then }
        assert result == '{"list": ["a", "b"]}'

    def test_fixes_real_world_truncated_json(self, analyzer):
        """Fixes truncated JSON like in the actual bug."""
        # This is similar to the actual truncated JSON from the bug
        json_str = """{
    "alt_text": "FreeTestData品牌logo设计",
    "detailed_description": "该图为FreeTestData公司的品牌logo设计。",
    "detected_text": "FreeTestData, YOUR SLOGAN HERE",
    "image_type": "logo",
    "knowledge_meta": {
        "entities": ["FreeTestData", "品牌logo"],
        "relationships": ["FreeTestData -> 属于 -> 数据测试服务"],
        "topics": ["品牌设计", "企业logo", "视觉识别"],
    """
        result = analyzer._fix_json_string(json_str)

        # Should be parseable now
        import json

        data = json.loads(result)
        assert data["alt_text"] == "FreeTestData品牌logo设计"
        assert data["image_type"] == "logo"
        assert "knowledge_meta" in data

    def test_fixes_trailing_comma_with_incomplete_content(self, analyzer):
        """Handles trailing comma followed by incomplete key."""
        json_str = '{"a": 1, "b": 2, "'
        result = analyzer._fix_json_string(json_str)
        # Should remove incomplete key and add closing brace
        assert result == '{"a": 1, "b": 2}'

    def test_handles_empty_string(self, analyzer):
        """Empty string returns empty string."""
        assert analyzer._fix_json_string("") == ""
        assert analyzer._fix_json_string(None) is None

    def test_valid_json_unchanged(self, analyzer):
        """Valid JSON is not modified."""
        json_str = '{"key": "value", "list": [1, 2, 3]}'
        result = analyzer._fix_json_string(json_str)
        assert result == json_str


class TestParseResponse:
    """Tests for _parse_response method handling various LLM outputs."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer without provider manager."""
        from unittest.mock import Mock

        return ImageAnalyzer(Mock())

    @pytest.fixture
    def mock_response(self):
        """Create a mock LLM response."""
        from unittest.mock import Mock

        def create_response(content: str):
            response = Mock()
            response.content = content
            return response

        return create_response

    def test_parses_valid_json(self, analyzer, mock_response):
        """Valid JSON is parsed correctly."""
        content = """{
            "alt_text": "Test image",
            "detailed_description": "A detailed description.",
            "detected_text": "Some text",
            "image_type": "photo"
        }"""
        result = analyzer._parse_response(mock_response(content))

        assert isinstance(result, ImageAnalysis)
        assert result.alt_text == "Test image"
        assert result.detailed_description == "A detailed description."
        assert result.detected_text == "Some text"
        assert result.image_type == "photo"

    def test_parses_json_with_trailing_comma(self, analyzer, mock_response):
        """JSON with trailing comma is parsed correctly."""
        content = """{
            "alt_text": "Test image",
            "detailed_description": "Description",
            "detected_text": null,
            "image_type": "diagram",
        }"""
        result = analyzer._parse_response(mock_response(content))

        assert result.alt_text == "Test image"
        assert result.image_type == "diagram"

    def test_parses_truncated_json(self, analyzer, mock_response):
        """Truncated JSON is fixed and parsed."""
        content = """{
            "alt_text": "Logo image",
            "detailed_description": "A company logo",
            "detected_text": "Company Name",
            "image_type": "logo",
            "knowledge_meta": {
                "entities": ["Company"],
                "relationships": [],
                "topics": ["branding"],
        """
        result = analyzer._parse_response(mock_response(content))

        assert result.alt_text == "Logo image"
        assert result.image_type == "logo"
        assert result.knowledge_meta is not None
        assert "Company" in result.knowledge_meta.entities

    def test_parses_json_in_code_block(self, analyzer, mock_response):
        """JSON wrapped in code block is extracted and parsed."""
        content = """Here's the analysis:
```json
{
    "alt_text": "Code block JSON",
    "detailed_description": "Extracted from code block",
    "detected_text": null,
    "image_type": "screenshot"
}
```
"""
        result = analyzer._parse_response(mock_response(content))

        assert result.alt_text == "Code block JSON"
        assert result.image_type == "screenshot"

    def test_parses_complex_json_in_code_block(self, analyzer, mock_response):
        """Complex JSON with knowledge_meta wrapped in code block is parsed."""
        # This is the actual format returned by Claude claude-sonnet-4-5-20250929
        content = """```json
{
    "alt_text": "黑色背景上的FreeTestData品牌标志设计",
    "detailed_description": "这是一个品牌标志图像，采用黑色背景。主标题FreeTestData使用了红色和白色的配色方案。",
    "detected_text": "FreeTestData\\nYOUR SLOGAN HERE",
    "image_type": "logo",
    "knowledge_meta": {
        "entities": ["FreeTestData"],
        "relationships": [],
        "topics": ["品牌标识", "测试数据", "Logo设计"],
        "domain": "商业"
    }
}
```"""
        result = analyzer._parse_response(mock_response(content))

        assert result.alt_text == "黑色背景上的FreeTestData品牌标志设计"
        assert result.image_type == "logo"
        assert result.knowledge_meta is not None
        assert "FreeTestData" in result.knowledge_meta.entities
        assert result.knowledge_meta.domain == "商业"

    def test_parses_json_with_surrounding_text(self, analyzer, mock_response):
        """JSON with surrounding text is extracted."""
        content = """I've analyzed the image. Here's my response:

{
    "alt_text": "Surrounded JSON",
    "detailed_description": "Found in text",
    "detected_text": "OCR text",
    "image_type": "chart"
}

Hope this helps!"""
        result = analyzer._parse_response(mock_response(content))

        assert result.alt_text == "Surrounded JSON"
        assert result.image_type == "chart"

    def test_parses_knowledge_meta(self, analyzer, mock_response):
        """Knowledge meta is correctly parsed."""
        content = """{
            "alt_text": "Architecture diagram",
            "detailed_description": "System architecture",
            "detected_text": "API Gateway",
            "image_type": "diagram",
            "knowledge_meta": {
                "entities": ["API Gateway", "Database"],
                "relationships": ["Gateway -> connects -> Database"],
                "topics": ["architecture", "microservices"],
                "domain": "technology"
            }
        }"""
        result = analyzer._parse_response(mock_response(content))

        assert result.knowledge_meta is not None
        assert "API Gateway" in result.knowledge_meta.entities
        assert "Database" in result.knowledge_meta.entities
        assert len(result.knowledge_meta.relationships) == 1
        assert result.knowledge_meta.domain == "technology"

    def test_fallback_on_completely_invalid_content(self, analyzer, mock_response):
        """Completely invalid content falls back gracefully."""
        content = "This is not JSON at all, just plain text."
        result = analyzer._parse_response(mock_response(content))

        assert result.alt_text == "Image"
        assert result.image_type == "other"
        assert "not JSON" in result.detailed_description

    def test_handles_null_detected_text(self, analyzer, mock_response):
        """null detected_text is handled correctly."""
        content = """{
            "alt_text": "Image",
            "detailed_description": "Description",
            "detected_text": null,
            "image_type": "photo"
        }"""
        result = analyzer._parse_response(mock_response(content))

        assert result.detected_text is None

    def test_handles_missing_optional_fields(self, analyzer, mock_response):
        """Missing optional fields use defaults."""
        content = """{
            "alt_text": "Minimal",
            "detailed_description": "Just the basics"
        }"""
        result = analyzer._parse_response(mock_response(content))

        assert result.alt_text == "Minimal"
        assert result.image_type == "other"  # Default
        assert result.knowledge_meta is None
