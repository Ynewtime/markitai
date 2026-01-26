"""Tests for workflow/helpers.py module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from markitai.workflow.helpers import (
    LLMUsageAccumulator,
    add_basic_frontmatter,
    detect_language,
    format_standalone_image_markdown,
    get_language_name,
    merge_llm_usage,
    normalize_frontmatter,
    write_images_json,
)


class TestNormalizeFrontmatter:
    """Tests for normalize_frontmatter function."""

    def test_dict_input(self):
        """Test normalizing a dict input."""
        data = {
            "title": "My Title",
            "source": "file.pdf",
            "markitai_processed": "2026-01-01T00:00:00",
        }
        result = normalize_frontmatter(data)
        assert "title: My Title" in result
        assert "source: file.pdf" in result
        assert "markitai_processed:" in result

    def test_string_input_with_markers(self):
        """Test normalizing a string with --- markers."""
        input_str = "---\ntitle: Test\nsource: file.txt\n---"
        result = normalize_frontmatter(input_str)
        assert "---" not in result
        assert "title: Test" in result
        assert "source: file.txt" in result

    def test_string_input_with_code_block(self):
        """Test normalizing a string wrapped in code block."""
        input_str = "```yaml\ntitle: Test\nsource: file.txt\n```"
        result = normalize_frontmatter(input_str)
        assert "```" not in result
        assert "title: Test" in result

    def test_field_ordering(self):
        """Test that fields are ordered according to FRONTMATTER_FIELD_ORDER."""
        data = {
            "markitai_processed": "2026-01-01T00:00:00",
            "custom_field": "value",
            "title": "My Title",
            "source": "file.pdf",
        }
        result = normalize_frontmatter(data)
        lines = result.strip().split("\n")
        # title should come before source, source before markitai_processed
        title_idx = next(i for i, line in enumerate(lines) if line.startswith("title:"))
        source_idx = next(
            i for i, line in enumerate(lines) if line.startswith("source:")
        )
        markitai_idx = next(
            i for i, line in enumerate(lines) if line.startswith("markitai_processed:")
        )
        custom_idx = next(
            i for i, line in enumerate(lines) if line.startswith("custom_field:")
        )

        assert title_idx < source_idx < markitai_idx
        assert markitai_idx < custom_idx  # custom fields at the end

    def test_list_values(self):
        """Test normalizing list values like tags."""
        data = {"title": "Test", "tags": ["tag1", "tag2"]}
        result = normalize_frontmatter(data)
        assert "tags:" in result
        # YAML flow style or multi-line list
        assert "tag1" in result
        assert "tag2" in result

    def test_invalid_yaml_returns_as_is(self):
        """Test that invalid YAML is returned as-is."""
        input_str = "not: valid: yaml: ][}"
        result = normalize_frontmatter(input_str)
        # Should not raise, returns cleaned input
        assert "not:" in result

    def test_non_dict_data(self):
        """Test normalizing non-dict data."""
        input_str = "just a string"
        result = normalize_frontmatter(input_str)
        assert result == "just a string"


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_empty_content(self):
        """Test with empty content returns English."""
        assert detect_language("") == "en"

    def test_english_content(self):
        """Test English content."""
        assert detect_language("This is a test document with English content.") == "en"

    def test_chinese_content(self):
        """Test Chinese content."""
        assert detect_language("这是一个测试文档，包含中文内容。") == "zh"

    def test_mixed_content_mostly_chinese(self):
        """Test mixed content with majority Chinese."""
        # More than 10% CJK should return zh
        assert detect_language("Hello 你好世界测试 World") == "zh"

    def test_mixed_content_mostly_english(self):
        """Test mixed content with majority English."""
        assert (
            detect_language(
                "This is a very long English sentence with just one 字 character."
            )
            == "en"
        )

    def test_only_symbols(self):
        """Test content with only symbols returns English."""
        assert detect_language("123 !@#$%^&*()") == "en"


class TestGetLanguageName:
    """Tests for get_language_name function."""

    def test_chinese_code(self):
        """Test Chinese language code."""
        assert get_language_name("zh") == "Chinese"

    def test_english_code(self):
        """Test English language code."""
        assert get_language_name("en") == "English"

    def test_unknown_code(self):
        """Test unknown language code defaults to English."""
        assert get_language_name("fr") == "English"
        assert get_language_name("unknown") == "English"


class TestAddBasicFrontmatter:
    """Tests for add_basic_frontmatter function."""

    def test_adds_frontmatter(self):
        """Test that frontmatter is added."""
        content = "# My Document\n\nSome content here."
        result = add_basic_frontmatter(content, "document.pdf")

        assert result.startswith("---\n")
        assert "title: My Document" in result
        assert "source: document.pdf" in result
        assert "markitai_processed:" in result
        assert content in result

    def test_extracts_title_from_heading(self):
        """Test that title is extracted from first heading."""
        content = "# **Bold Title**\n\nContent"
        result = add_basic_frontmatter(content, "file.txt")

        # Should extract "Bold Title" without # and **
        assert "title: Bold Title" in result

    def test_uses_source_as_title_fallback(self):
        """Test that source is used as title when no heading found."""
        content = "No heading here, just content."
        result = add_basic_frontmatter(content, "document.pdf")

        assert "title: document.pdf" in result


class TestMergeLlmUsage:
    """Tests for merge_llm_usage function."""

    def test_merge_into_empty_target(self):
        """Test merging into an empty target."""
        target: dict = {}
        source = {
            "gpt-4": {
                "requests": 5,
                "input_tokens": 1000,
                "output_tokens": 500,
                "cost_usd": 0.05,
            }
        }
        merge_llm_usage(target, source)

        assert target["gpt-4"]["requests"] == 5
        assert target["gpt-4"]["input_tokens"] == 1000
        assert target["gpt-4"]["output_tokens"] == 500
        assert target["gpt-4"]["cost_usd"] == 0.05

    def test_merge_with_existing_model(self):
        """Test merging when model already exists."""
        target = {
            "gpt-4": {
                "requests": 3,
                "input_tokens": 500,
                "output_tokens": 200,
                "cost_usd": 0.02,
            }
        }
        source = {
            "gpt-4": {
                "requests": 2,
                "input_tokens": 300,
                "output_tokens": 100,
                "cost_usd": 0.01,
            }
        }
        merge_llm_usage(target, source)

        assert target["gpt-4"]["requests"] == 5
        assert target["gpt-4"]["input_tokens"] == 800
        assert target["gpt-4"]["output_tokens"] == 300
        assert target["gpt-4"]["cost_usd"] == 0.03

    def test_merge_multiple_models(self):
        """Test merging with multiple models."""
        target = {"gpt-4": {"requests": 1, "input_tokens": 100}}
        source = {"claude-3": {"requests": 2, "input_tokens": 200}}
        merge_llm_usage(target, source)

        assert "gpt-4" in target
        assert "claude-3" in target


class TestLLMUsageAccumulator:
    """Tests for LLMUsageAccumulator class."""

    def test_initial_state(self):
        """Test initial state is zero."""
        acc = LLMUsageAccumulator()
        assert acc.total_cost == 0.0
        assert acc.usage == {}

    def test_add_cost_and_usage(self):
        """Test adding cost and usage."""
        acc = LLMUsageAccumulator()
        acc.add(
            cost=0.05,
            usage={
                "gpt-4": {
                    "requests": 1,
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.05,
                }
            },
        )

        assert acc.total_cost == 0.05
        assert acc.usage["gpt-4"]["requests"] == 1

    def test_add_multiple_times(self):
        """Test adding multiple times accumulates correctly."""
        acc = LLMUsageAccumulator()
        acc.add(cost=0.05, usage={"gpt-4": {"requests": 1}})
        acc.add(cost=0.03, usage={"gpt-4": {"requests": 2}})

        assert acc.total_cost == 0.08
        assert acc.usage["gpt-4"]["requests"] == 3

    def test_add_cost_only(self):
        """Test adding cost without usage."""
        acc = LLMUsageAccumulator()
        acc.add(cost=0.10)
        assert acc.total_cost == 0.10
        assert acc.usage == {}

    def test_reset(self):
        """Test reset clears all state."""
        acc = LLMUsageAccumulator()
        acc.add(cost=0.05, usage={"gpt-4": {"requests": 1}})
        acc.reset()

        assert acc.total_cost == 0.0
        assert acc.usage == {}


class TestWriteImagesJson:
    """Tests for write_images_json function."""

    def test_empty_results(self, tmp_path: Path):
        """Test with empty results returns empty list."""
        result = write_images_json(tmp_path, [])
        assert result == []

    def test_writes_images_json(self, tmp_path: Path):
        """Test writing images JSON file."""
        from markitai.workflow.single import ImageAnalysisResult

        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        results = [
            ImageAnalysisResult(
                source_file="/path/to/file.pdf",
                assets=[
                    {
                        "asset": str(assets_dir / "image1.png"),
                        "alt": "Test image",
                        "desc": "A test image",
                        "text": "OCR text",
                    }
                ],
            )
        ]

        created_files = write_images_json(tmp_path, results)

        assert len(created_files) == 1
        assert created_files[0] == assets_dir / "images.json"
        assert created_files[0].exists()

    def test_merges_with_existing(self, tmp_path: Path):
        """Test merging with existing images.json."""
        import json

        from markitai.workflow.single import ImageAnalysisResult

        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()

        # Create existing images.json (using new field names)
        existing = {
            "version": "1.0",
            "created": "2026-01-01T00:00:00",
            "images": [
                {
                    "path": str(assets_dir / "existing.png"),
                    "alt": "Existing",
                    "desc": "Existing image",
                }
            ],
        }
        (assets_dir / "images.json").write_text(json.dumps(existing))

        # Add new image
        results = [
            ImageAnalysisResult(
                source_file="/path/to/file.pdf",
                assets=[
                    {
                        "asset": str(assets_dir / "new.png"),
                        "alt": "New image",
                        "desc": "New description",
                    }
                ],
            )
        ]

        write_images_json(tmp_path, results)

        # Check merged result
        merged = json.loads((assets_dir / "images.json").read_text())
        assert len(merged["images"]) == 2
        assert merged["created"] == "2026-01-01T00:00:00"  # Original creation time


class TestFormatStandaloneImageMarkdown:
    """Tests for format_standalone_image_markdown function."""

    def test_basic_format(self):
        """Test basic formatting without frontmatter."""
        analysis = MagicMock()
        analysis.caption = "A beautiful sunset"
        analysis.description = "A sunset over the ocean"
        analysis.extracted_text = ""

        result = format_standalone_image_markdown(
            Path("sunset.jpg"), analysis, "assets/sunset.jpg", include_frontmatter=False
        )

        assert "# sunset" in result
        assert "![A beautiful sunset](assets/sunset.jpg)" in result
        assert "A sunset over the ocean" in result
        assert "---" not in result  # No frontmatter

    def test_with_frontmatter(self):
        """Test formatting with frontmatter."""
        analysis = MagicMock()
        analysis.caption = "A beautiful sunset"
        analysis.description = "A sunset over the ocean"
        analysis.extracted_text = ""

        result = format_standalone_image_markdown(
            Path("sunset.jpg"), analysis, "assets/sunset.jpg", include_frontmatter=True
        )

        assert "---" in result
        assert "title: sunset" in result
        assert "description: A beautiful sunset" in result
        assert "source: sunset.jpg" in result
        assert "tags:" in result
        assert "- image" in result
        assert "markitai_processed:" in result

    def test_with_extracted_text(self):
        """Test formatting with extracted text."""
        analysis = MagicMock()
        analysis.caption = "Document scan"
        analysis.description = "A scanned document"
        analysis.extracted_text = "Some extracted text from OCR"

        result = format_standalone_image_markdown(
            Path("scan.png"), analysis, "assets/scan.png", include_frontmatter=False
        )

        assert "## Extracted Text" in result
        assert "Some extracted text from OCR" in result
        assert "```" in result  # Code block

    def test_description_with_header(self):
        """Test that description with existing header doesn't add extra header."""
        analysis = MagicMock()
        analysis.caption = "Test"
        analysis.description = "## Custom Section\n\nContent here"
        analysis.extracted_text = ""

        result = format_standalone_image_markdown(
            Path("test.jpg"), analysis, "assets/test.jpg", include_frontmatter=False
        )

        # Should not have "## Image Description" since description starts with #
        assert "## Custom Section" in result
        assert "## Image Description" not in result
