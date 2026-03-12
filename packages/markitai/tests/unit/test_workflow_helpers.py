"""Tests for workflow/helpers.py module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from markitai.workflow.helpers import (
    add_basic_frontmatter,
    extract_document_context,
    format_standalone_image_markdown,
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

    def test_uses_stem_as_title_fallback_for_document_sources(self):
        """Document-like sources should fall back to a stable stem title."""
        content = "No heading here, just content."
        result = add_basic_frontmatter(content, "document.pdf")

        assert "title: document" in result

    def test_structured_sources_keep_filename_title(self):
        """Structured sources should not derive title from markdown headings."""
        content = "# Reminder\n\nStructured content."
        result = add_basic_frontmatter(content, "sample.xml")

        assert "title: sample.xml" in result


class TestAddBasicFrontmatterDedupeDefault:
    """Tests for dedupe default behavior — off by default."""

    def test_duplicate_paragraphs_preserved_by_default(self):
        """Pre-LLM .md files should faithfully preserve original content.

        Deduplication is off by default — .md files keep everything as extracted.
        LLM cleanup handles duplicates in the .llm.md output.
        """
        paragraph = (
            "This is a long enough paragraph that appears twice in the "
            "original extracted content and should be preserved as-is."
        )
        content = f"# Title\n\n{paragraph}\n\n{paragraph}"
        result = add_basic_frontmatter(content, "page.html")

        assert result.count(paragraph) == 2

    def test_dedupe_opt_in_works(self):
        """Callers can explicitly opt-in to deduplication."""
        paragraph = (
            "This is a long enough paragraph that appears twice and should be "
            "deduplicated when the caller explicitly requests it."
        )
        content = f"# Title\n\n{paragraph}\n\n{paragraph}"
        result = add_basic_frontmatter(content, "page.html", dedupe=True)

        assert result.count(paragraph) == 1


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


class TestWriteImagesJson:
    """Tests for write_images_json function."""

    def test_empty_results(self, tmp_path: Path):
        """Test with empty results returns empty list."""
        result = write_images_json(tmp_path, [])
        assert result == []

    def test_writes_images_json(self, tmp_path: Path):
        """Test writing images JSON file."""
        from markitai.workflow.single import ImageAnalysisResult

        assets_dir = tmp_path / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)

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

        assets_dir = tmp_path / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)

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
            Path("sunset.jpg"),
            analysis,
            ".markitai/assets/sunset.jpg",
            include_frontmatter=False,
        )

        assert "# sunset" in result
        assert "![A beautiful sunset](.markitai/assets/sunset.jpg)" in result
        assert "A sunset over the ocean" in result
        assert "---" not in result  # No frontmatter

    def test_with_frontmatter(self):
        """Frontmatter should carry metadata without expanding body content."""
        analysis = MagicMock()
        analysis.caption = "A beautiful sunset"
        analysis.description = "A sunset over the ocean"
        analysis.extracted_text = ""

        result = format_standalone_image_markdown(
            Path("sunset.jpg"),
            analysis,
            ".markitai/assets/sunset.jpg",
            include_frontmatter=True,
        )

        assert "---" in result
        assert "title: sunset" in result
        assert "description: A beautiful sunset" in result
        assert "source: sunset.jpg" in result
        assert "tags:" in result
        assert "- image" in result
        assert "markitai_processed:" in result

    def test_with_frontmatter_preserves_analysis_sections(self):
        """Standalone .llm.md output should retain detailed analysis content."""
        analysis = MagicMock()
        analysis.caption = "Document scan"
        analysis.description = "A scanned document"
        analysis.extracted_text = "Some extracted text from OCR"

        result = format_standalone_image_markdown(
            Path("scan.png"),
            analysis,
            ".markitai/assets/scan.png",
            include_frontmatter=True,
        )

        assert "## Image Description" in result
        assert "## Extracted Text" in result
        assert "Some extracted text from OCR" in result

    def test_with_extracted_text(self):
        """Test formatting with extracted text."""
        analysis = MagicMock()
        analysis.caption = "Document scan"
        analysis.description = "A scanned document"
        analysis.extracted_text = "Some extracted text from OCR"

        result = format_standalone_image_markdown(
            Path("scan.png"),
            analysis,
            ".markitai/assets/scan.png",
            include_frontmatter=False,
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
            Path("test.jpg"),
            analysis,
            ".markitai/assets/test.jpg",
            include_frontmatter=False,
        )

        assert "## Custom Section" in result
        assert "## Image Description" not in result


class TestFormatStandaloneImageYamlSafety:
    """Test that format_standalone_image_markdown produces valid YAML."""

    def test_special_chars_in_caption_produce_valid_yaml(self):
        """Captions with colons/quotes should produce parseable YAML frontmatter."""
        import yaml

        analysis = MagicMock()
        analysis.caption = "Photo: \"A sunset\" at O'Brien's pier"
        analysis.description = "A sunset over the ocean"
        analysis.extracted_text = ""

        result = format_standalone_image_markdown(
            Path("sunset.jpg"),
            analysis,
            ".markitai/assets/sunset.jpg",
            include_frontmatter=True,
        )

        # Extract frontmatter
        parts = result.split("---")
        assert len(parts) >= 3, "Should have frontmatter delimiters"
        fm_text = parts[1]
        # Must be parseable YAML
        parsed = yaml.safe_load(fm_text)
        assert "sunset" in parsed.get(
            "description", ""
        ).lower() or "Photo" in parsed.get("description", "")


class TestNormalizeFrontmatterPromptLeakage:
    """Tests for prompt leakage filtering in normalize_frontmatter."""

    def test_filters_chinese_prompt_leakage(self):
        """Test that Chinese prompt leakage keys are filtered."""
        data = {
            "title": "Valid Title",
            "根据文档内容生成元数据": "Invalid key",
            "source": "file.pdf",
        }
        result = normalize_frontmatter(data)

        assert "title: Valid Title" in result
        assert "source: file.pdf" in result
        assert "根据文档内容生成" not in result

    def test_filters_task_number_leakage(self):
        """Test that task number patterns are filtered."""
        data = {
            "title": "Valid Title",
            "任务 1": "Do something",
            "Task 2": "Do something else",
            "source": "file.pdf",
        }
        result = normalize_frontmatter(data)

        assert "title: Valid Title" in result
        assert "任务" not in result
        assert "Task 2" not in result

    def test_filters_yaml_frontmatter_leakage(self):
        """Test that YAML frontmatter pattern is filtered."""
        data = {
            "title": "Test",
            "YAML frontmatter 格式": "invalid",
        }
        result = normalize_frontmatter(data)

        assert "title: Test" in result
        assert "YAML frontmatter" not in result

    def test_preserves_valid_custom_fields(self):
        """Test that valid custom fields are preserved."""
        data = {
            "title": "Test",
            "author": "John Doe",
            "version": "1.0",
        }
        result = normalize_frontmatter(data)

        assert "title: Test" in result
        assert "author: John Doe" in result
        assert "version: '1.0'" in result or "version: 1.0" in result


class TestAddBasicFrontmatterAdvanced:
    """Advanced tests for add_basic_frontmatter function."""

    def test_explicit_title_takes_precedence(self):
        """Test that explicit title parameter overrides heading extraction."""
        content = "# Heading Title\n\nContent"
        result = add_basic_frontmatter(content, "file.txt", title="Explicit Title")

        assert "title: Explicit Title" in result
        assert "Heading Title" not in result.split("---")[1]  # Not in frontmatter

    def test_fetch_strategy_added(self):
        """Test that fetch_strategy is added when provided."""
        content = "# Test\n\nContent"
        result = add_basic_frontmatter(content, "file.txt", fetch_strategy="browser")

        assert "fetch_strategy: browser" in result

    def test_fetch_strategy_not_added_when_none(self):
        """Test that fetch_strategy is omitted when None."""
        content = "# Test\n\nContent"
        result = add_basic_frontmatter(content, "file.txt", fetch_strategy=None)

        assert "fetch_strategy" not in result

    def test_screenshot_path_added(self, tmp_path: Path):
        """Test that screenshot path is added as comment."""
        content = "# Test\n\nContent"
        screenshot = tmp_path / "screenshot.png"
        screenshot.write_text("fake image")

        result = add_basic_frontmatter(
            content,
            "file.txt",
            screenshot_path=screenshot,
            output_dir=tmp_path,
        )

        assert "<!-- Screenshot for reference -->" in result
        assert "screenshot.png" in result

    def test_screenshot_path_skipped_if_not_exists(self, tmp_path: Path):
        """Test that non-existent screenshot path is ignored."""
        content = "# Test\n\nContent"
        screenshot = tmp_path / "nonexistent.png"

        result = add_basic_frontmatter(
            content,
            "file.txt",
            screenshot_path=screenshot,
            output_dir=tmp_path,
        )

        assert "Screenshot" not in result

    def test_dedupe_false_preserves_duplicates(self):
        """Test that dedupe=False preserves duplicate paragraphs."""
        content = "Paragraph one.\n\nParagraph one.\n\nParagraph two."
        result = add_basic_frontmatter(content, "file.txt", dedupe=False)

        # Count occurrences of "Paragraph one"
        count = result.count("Paragraph one.")
        assert count == 2  # Both preserved

    def test_title_with_newlines_normalized(self):
        """Test that title with newlines is normalized."""
        # Explicitly pass title with newlines
        result = add_basic_frontmatter(
            "Content",
            "file.txt",
            title="Title with\nnewline",
        )

        # Title should have newline replaced with space
        assert "title: Title with newline" in result
        assert "\n" not in result.split("---")[1].split("title:")[1].split("\n")[0]

    def test_h2_heading_extracted(self):
        """Test that H2 heading is extracted when no H1."""
        content = "## Section Title\n\nContent"
        result = add_basic_frontmatter(content, "file.txt")

        assert "title: Section Title" in result


class TestBasicFrontmatterConsistency:
    """Tests for frontmatter consistency in add_basic_frontmatter."""

    def test_timestamp_format_has_timezone(self):
        """markitai_processed in .md should have timezone and milliseconds."""
        import re

        content = "# Test\n\nContent"
        result = add_basic_frontmatter(content, "file.txt")
        # Extract timestamp from frontmatter
        for line in result.split("\n"):
            if "markitai_processed:" in line:
                ts = line.split("markitai_processed:")[1].strip().strip("'\"")
                assert re.match(
                    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2}$",
                    ts,
                ), f"Timestamp format wrong: {ts}"
                break
        else:
            raise AssertionError("markitai_processed not found in frontmatter")

    def test_base_and_llm_frontmatter_generate_independent_timestamps(self):
        """Base and LLM frontmatter generation should not reuse the same timestamp."""
        from unittest.mock import patch

        from markitai.utils.frontmatter import build_frontmatter_dict

        with patch(
            "markitai.utils.frontmatter.frontmatter_timestamp",
            side_effect=[
                "2026-03-07T00:46:22.123+08:00",
                "2026-03-07T00:46:22.456+08:00",
            ],
        ):
            base = add_basic_frontmatter("# Test\n\nContent", "file.txt")
            llm = build_frontmatter_dict(
                source="file.txt",
                description="Desc",
                content="# Test\n\nContent",
            )

        assert "markitai_processed: '2026-03-07T00:46:22.123+08:00'" in base
        assert llm["markitai_processed"] == "2026-03-07T00:46:22.456+08:00"

    def test_extra_meta_merged_into_frontmatter(self):
        """Extra metadata should appear in .md frontmatter."""
        import yaml

        content = "# Test\n\nContent"
        extra = {"author": "John Doe", "published": "2024-01-15"}
        result = add_basic_frontmatter(content, "https://example.com", extra_meta=extra)
        # Parse frontmatter
        fm_text = result.split("---")[1]
        parsed = yaml.safe_load(fm_text)
        assert parsed["author"] == "John Doe"
        assert parsed["published"] == "2024-01-15"

    def test_extra_meta_filters_unreliable_language(self):
        """Language from HTML meta tags is unreliable and must be excluded."""
        import yaml

        content = "# 测试\n\n中文内容"
        extra = {"author": "张三", "language": "en-us", "domain": "example.com"}
        result = add_basic_frontmatter(content, "https://example.com", extra_meta=extra)
        fm_text = result.split("---")[1]
        parsed = yaml.safe_load(fm_text)
        assert "language" not in parsed
        assert parsed["author"] == "张三"

    def test_extra_meta_does_not_override_core_fields(self):
        """Extra meta must not override title/source/markitai_processed."""
        import yaml

        content = "# Test\n\nContent"
        extra = {"title": "WRONG", "source": "WRONG"}
        result = add_basic_frontmatter(content, "https://example.com", extra_meta=extra)
        fm_text = result.split("---")[1]
        parsed = yaml.safe_load(fm_text)
        assert parsed["source"] == "https://example.com"
        assert parsed["title"] != "WRONG"

    def test_native_source_metadata_uses_stable_field_order(self):
        """Native source metadata should serialize in a stable, readable order."""
        result = add_basic_frontmatter(
            "# Test\n\nContent",
            "https://example.com",
            fetch_strategy="static",
            extra_meta={
                "author": "Jane Doe",
                "site": "Example",
                "published": "2026-02-01",
                "canonical_url": "https://example.com/canonical",
            },
        )

        title_pos = result.find("title:")
        source_pos = result.find("source:")
        author_pos = result.find("author:")
        site_pos = result.find("site:")
        published_pos = result.find("published:")
        canonical_pos = result.find("canonical_url:")
        processed_pos = result.find("markitai_processed:")
        fetch_pos = result.find("fetch_strategy:")

        assert title_pos < source_pos < author_pos < site_pos < published_pos
        assert published_pos < canonical_pos < processed_pos < fetch_pos


class TestMergeLlmUsageEdgeCases:
    """Edge case tests for merge_llm_usage function."""

    def test_merge_with_incomplete_target_fields(self):
        """Test merging when target has incomplete fields."""
        target = {
            "gpt-4": {
                "requests": 1,
                # Missing other fields
            }
        }
        source = {
            "gpt-4": {
                "requests": 2,
                "input_tokens": 500,
                "output_tokens": 200,
                "cost_usd": 0.05,
            }
        }
        merge_llm_usage(target, source)

        assert target["gpt-4"]["requests"] == 3
        assert target["gpt-4"]["input_tokens"] == 500
        assert target["gpt-4"]["output_tokens"] == 200
        assert target["gpt-4"]["cost_usd"] == 0.05

    def test_merge_with_incomplete_source_fields(self):
        """Test merging when source has incomplete fields."""
        target = {
            "gpt-4": {
                "requests": 1,
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.01,
            }
        }
        source = {
            "gpt-4": {
                "requests": 2,
                # Missing other fields
            }
        }
        merge_llm_usage(target, source)

        assert target["gpt-4"]["requests"] == 3
        assert target["gpt-4"]["input_tokens"] == 100  # Unchanged
        assert target["gpt-4"]["output_tokens"] == 50  # Unchanged
        assert target["gpt-4"]["cost_usd"] == 0.01  # Unchanged

    def test_merge_empty_source(self):
        """Test merging with empty source doesn't change target."""
        target = {
            "gpt-4": {
                "requests": 5,
                "input_tokens": 1000,
            }
        }
        source: dict = {}
        merge_llm_usage(target, source)

        assert target["gpt-4"]["requests"] == 5
        assert target["gpt-4"]["input_tokens"] == 1000


class TestExtractDocumentContext:
    """Tests for extract_document_context function."""

    def test_strips_frontmatter_before_extracting(self):
        """Document context should come from body text, not YAML frontmatter."""
        markdown = (
            "---\n"
            "title: 人是什么单位？ | Y\n"
            "source: https://ynewtime.com/jekyll-ynewtime/人是什么单位\n"
            "author: 熊培云\n"
            "published: 2018-04-10\n"
            "markitai_processed: '2026-03-12T14:38:01.760+08:00'\n"
            "fetch_strategy: defuddle\n"
            "domain: ynewtime.com\n"
            "word_count: 1897\n"
            "---\n\n"
            "![](https://example.com/image.jpg)\n\n"
            "我曾说，一人即一国，每个人都有属于自己的疆土。\n\n"
            "同样，区别于\u201c你属于某个单位\u201d。\n"
        )
        result = extract_document_context(markdown)
        # Body text should be present, not frontmatter metadata
        assert "我曾说" in result
        # YAML keys should NOT be in the context
        assert "markitai_processed" not in result
        assert "fetch_strategy" not in result

    def test_skips_image_lines(self):
        """Image reference lines should be excluded from context."""
        markdown = "![alt text](image.png)\n\nThis is the body text.\n"
        result = extract_document_context(markdown)
        assert "body text" in result
        assert "![alt text]" not in result

    def test_truncates_to_200_chars(self):
        """Context should be at most 200 characters."""
        markdown = "A" * 500 + "\n"
        result = extract_document_context(markdown)
        assert len(result) <= 200

    def test_no_frontmatter(self):
        """Works correctly when no frontmatter is present."""
        markdown = "# Title\n\nSome body text here.\n"
        result = extract_document_context(markdown)
        assert "Title" in result
        assert "body text" in result

    def test_empty_input(self):
        """Returns empty string for empty input."""
        assert extract_document_context("") == ""
        assert extract_document_context("\n\n\n") == ""
