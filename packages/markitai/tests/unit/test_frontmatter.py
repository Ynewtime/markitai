"""Unit tests for frontmatter utilities."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from markitai.utils.frontmatter import (
    build_frontmatter_dict,
    extract_title_from_content,
    frontmatter_to_yaml,
)


class TestExtractTitleFromContent:
    """Tests for extract_title_from_content function."""

    def test_extract_h1_heading(self) -> None:
        """Should extract title from H1 heading."""
        content = "# My Document Title\n\nSome content here."
        assert extract_title_from_content(content) == "My Document Title"

    def test_extract_h1_heading_with_leading_whitespace(self) -> None:
        """Should handle H1 with leading whitespace in content."""
        content = "\n\n# My Document Title\n\nSome content here."
        assert extract_title_from_content(content) == "My Document Title"

    def test_extract_h2_heading_when_no_h1(self) -> None:
        """Should extract title from H2 when no H1 present."""
        content = "## Section Title\n\nSome content here."
        assert extract_title_from_content(content) == "Section Title"

    def test_h1_takes_priority_over_h2(self) -> None:
        """H1 should take priority over H2."""
        content = "## Section\n\n# Main Title\n\nContent"
        # Should find H1 even though H2 comes first
        assert extract_title_from_content(content) == "Main Title"

    def test_extract_first_line_when_no_heading(self) -> None:
        """Should extract first non-empty line when no heading present."""
        content = "This is the first line of content.\n\nMore content here."
        assert (
            extract_title_from_content(content) == "This is the first line of content."
        )

    def test_skip_frontmatter_block(self) -> None:
        """Should skip YAML frontmatter block."""
        content = """---
title: Old Title
description: Some description
---

# Actual Title

Content here."""
        assert extract_title_from_content(content) == "Actual Title"

    def test_skip_html_comments(self) -> None:
        """Should skip HTML comments."""
        content = """<!-- This is a comment -->
# Real Title

Content."""
        assert extract_title_from_content(content) == "Real Title"

    def test_skip_empty_lines(self) -> None:
        """Should skip empty lines when looking for first line."""
        content = """


First actual line of content.

More content."""
        assert extract_title_from_content(content) == "First actual line of content."

    def test_fallback_when_empty_content(self) -> None:
        """Should return fallback for empty content."""
        assert extract_title_from_content("", "Untitled") == "Untitled"
        assert extract_title_from_content("   \n\n   ", "Fallback") == "Fallback"

    def test_fallback_default_is_empty_string(self) -> None:
        """Default fallback should be empty string."""
        assert extract_title_from_content("") == ""

    def test_max_length_100_chars(self) -> None:
        """Title should be truncated to 100 characters max."""
        long_title = "A" * 150
        content = f"# {long_title}\n\nContent"
        result = extract_title_from_content(content)
        assert len(result) <= 100
        assert result == "A" * 100

    def test_strip_whitespace_from_title(self) -> None:
        """Should strip leading/trailing whitespace from title."""
        content = "#   Title with spaces   \n\nContent"
        assert extract_title_from_content(content) == "Title with spaces"

    def test_strip_markdown_formatting_from_heading_title(self) -> None:
        """Markdown emphasis should not leak into extracted titles."""
        content = "# **Bold Title**\n\nContent"
        assert extract_title_from_content(content) == "Bold Title"

    def test_h1_with_multiple_hashes(self) -> None:
        """Should only match single # for H1."""
        content = "## Not H1\n### Also not H1\n# This is H1\n\nContent"
        assert extract_title_from_content(content) == "This is H1"

    def test_code_block_with_hash_not_heading(self) -> None:
        """Hash inside code block should not be treated as heading."""
        content = """```python
# This is a comment, not a heading
print("hello")
```

# Actual Title

Content."""
        assert extract_title_from_content(content) == "Actual Title"

    def test_first_line_after_frontmatter(self) -> None:
        """Should use first line after frontmatter if no heading."""
        content = """---
description: test
---

This is the first content line without heading."""
        assert (
            extract_title_from_content(content)
            == "This is the first content line without heading."
        )


class TestStripFrontmatter:
    """Tests for _strip_frontmatter and FRONTMATTER_PATTERN."""

    def test_basic_strip(self) -> None:
        """Should strip basic frontmatter."""
        from markitai.utils.frontmatter import _strip_frontmatter

        content = "---\ntitle: Test\n---\n\nBody text."
        result = _strip_frontmatter(content)
        assert "Body text." in result
        assert "title: Test" not in result

    def test_yaml_value_with_triple_dashes(self) -> None:
        """Closing --- inside YAML value should not end frontmatter early."""
        from markitai.utils.frontmatter import _strip_frontmatter

        content = "---\ntitle: Some --- value\nsource: file.pdf\n---\n\nBody text.\n"
        result = _strip_frontmatter(content)
        assert "Body text." in result
        assert "title:" not in result
        assert "source:" not in result

    def test_no_frontmatter(self) -> None:
        """Content without frontmatter should be returned as-is."""
        from markitai.utils.frontmatter import _strip_frontmatter

        content = "Just some text."
        result = _strip_frontmatter(content)
        assert result == content

    def test_frontmatter_pattern_is_shared_constant(self) -> None:
        """FRONTMATTER_PATTERN should be a compiled regex constant."""
        from markitai.utils.frontmatter import FRONTMATTER_PATTERN

        assert hasattr(FRONTMATTER_PATTERN, "sub")
        assert hasattr(FRONTMATTER_PATTERN, "match")


class TestBuildFrontmatterDict:
    """Tests for build_frontmatter_dict function."""

    def test_basic_frontmatter_build(self) -> None:
        """Should build basic frontmatter with required fields."""
        result = build_frontmatter_dict(
            source="document.pdf",
            description="A test document",
            tags=["test", "document"],
        )

        assert result["source"] == "document.pdf"
        assert result["description"] == "A test document"
        assert result["tags"] == ["test", "document"]
        assert "markitai_processed" in result
        # Check ISO 8601 format
        assert re.match(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", result["markitai_processed"]
        )

    def test_explicit_title(self) -> None:
        """Should use explicit title when provided."""
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="Description",
            title="Explicit Title",
        )

        assert result["title"] == "Explicit Title"

    def test_explicit_title_strips_markdown_formatting(self) -> None:
        """Explicit titles should be normalized for frontmatter."""
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="Description",
            title="**Explicit Title**",
        )

        assert result["title"] == "Explicit Title"

    def test_title_from_content_extraction(self) -> None:
        """Should extract title from content when not provided explicitly."""
        content = "# Extracted Title\n\nSome content."
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="Description",
            content=content,
        )

        assert result["title"] == "Extracted Title"

    def test_structured_sources_do_not_derive_title_from_content(self) -> None:
        """Structured files should keep a stable source-based title."""
        result = build_frontmatter_dict(
            source="sample.tsv",
            description="Dataset",
            content="# Employee Roster\n\nTabular content.",
        )

        assert result["title"] == "sample.tsv"

    def test_title_fallback_to_source_filename(self) -> None:
        """Should fallback to source filename when no title extractable."""
        result = build_frontmatter_dict(
            source="my-document.pdf",
            description="Description",
            content="",  # Empty content, no title extractable
        )

        # Should use source filename (without extension) as fallback
        assert result["title"] == "my-document"

    def test_title_fallback_removes_extension(self) -> None:
        """Fallback title should remove file extension."""
        result = build_frontmatter_dict(
            source="report_2024.docx",
            description="A report",
        )

        assert result["title"] == "report_2024"

    def test_tags_optional(self) -> None:
        """Tags should be optional."""
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="Description",
        )

        # Tags should not be present or be empty list
        assert result.get("tags") is None or result.get("tags") == []

    def test_empty_tags_excluded(self) -> None:
        """Empty tags list should be excluded from output."""
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="Description",
            tags=[],
        )

        assert "tags" not in result or result.get("tags") is None

    def test_timestamp_is_current(self) -> None:
        """Timestamp should be close to current time."""
        before = datetime.now().astimezone().replace(microsecond=0)
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="Description",
        )
        after = datetime.now().astimezone().replace(microsecond=0)

        timestamp = datetime.fromisoformat(result["markitai_processed"])
        # Allow 1 second tolerance for test execution time
        assert before <= timestamp <= after + timedelta(seconds=1)

    def test_field_order(self) -> None:
        """Fields should be in expected order."""
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="Description",
            tags=["tag1"],
            title="Title",
        )

        keys = list(result.keys())
        # Expected order: title, source, description, tags, markitai_processed
        assert keys.index("title") < keys.index("source")
        assert keys.index("source") < keys.index("description")
        assert keys.index("description") < keys.index("markitai_processed")

    def test_description_can_be_empty(self) -> None:
        """Empty description is valid."""
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="",
        )

        assert result["description"] == ""

    def test_title_newlines_normalized(self) -> None:
        """Newlines in title should be replaced with spaces."""
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="Description",
            title="Title with\nnewline\ncharacters",
        )

        assert "\n" not in result["title"]
        assert result["title"] == "Title with newline characters"

    def test_title_truncated_at_200_chars(self) -> None:
        """Title longer than 200 chars should be truncated."""
        long_title = "A" * 250
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="Description",
            title=long_title,
        )

        assert len(result["title"]) <= 200
        assert result["title"].endswith("...")

    def test_description_newlines_normalized(self) -> None:
        """Newlines in description should be replaced with spaces."""
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="Line 1\nLine 2\nLine 3",
        )

        assert "\n" not in result["description"]
        assert result["description"] == "Line 1 Line 2 Line 3"

    def test_description_truncated_at_150_chars(self) -> None:
        """Description longer than 150 chars should be truncated."""
        long_desc = "B" * 200
        result = build_frontmatter_dict(
            source="doc.pdf",
            description=long_desc,
        )

        assert len(result["description"]) <= 150
        assert result["description"].endswith("...")

    def test_tags_spaces_replaced_with_hyphens(self) -> None:
        """Spaces in tags should be replaced with hyphens."""
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="Description",
            tags=["machine learning", "web development", "AI"],
        )

        assert "machine learning" not in result["tags"]
        assert "machine-learning" in result["tags"]
        assert "web-development" in result["tags"]
        assert "AI" in result["tags"]

    def test_tags_truncated_at_30_chars(self) -> None:
        """Tags longer than 30 chars should be truncated."""
        long_tag = "C" * 50
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="Description",
            tags=[long_tag, "short"],
        )

        for tag in result["tags"]:
            assert len(tag) <= 30

    def test_tags_special_chars_removed(self) -> None:
        """Special characters in tags should be handled."""
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="Description",
            tags=["tag:with:colon", "tag'quote", 'tag"double'],
        )

        for tag in result["tags"]:
            assert ":" not in tag or tag.count("-") >= tag.count(":")
            assert "'" not in tag
            assert '"' not in tag


class TestFrontmatterToYaml:
    """Tests for frontmatter_to_yaml function."""

    def test_basic_yaml_output(self) -> None:
        """Should convert frontmatter dict to YAML string."""
        frontmatter = {
            "title": "Test Title",
            "source": "doc.pdf",
            "description": "A test document",
            "markitai_processed": "2024-01-15T10:30:00",
        }

        yaml_str = frontmatter_to_yaml(frontmatter)

        assert "title: Test Title" in yaml_str
        assert "source: doc.pdf" in yaml_str
        assert "description: A test document" in yaml_str
        assert (
            "markitai_processed: '2024-01-15T10:30:00'" in yaml_str
            or "markitai_processed: 2024-01-15T10:30:00" in yaml_str
        )

    def test_yaml_no_markers(self) -> None:
        """YAML output should not include --- markers."""
        frontmatter = {"title": "Test"}
        yaml_str = frontmatter_to_yaml(frontmatter)

        assert not yaml_str.startswith("---")
        assert "---" not in yaml_str

    def test_list_formatting(self) -> None:
        """Tags list should be formatted correctly."""
        frontmatter = {
            "title": "Test",
            "tags": ["tag1", "tag2", "tag3"],
        }

        yaml_str = frontmatter_to_yaml(frontmatter)

        # Should contain tags as YAML list
        assert "tags:" in yaml_str
        # Either flow style [tag1, tag2, tag3] or block style with - tag1
        assert "tag1" in yaml_str and "tag2" in yaml_str and "tag3" in yaml_str

    def test_multiline_description(self) -> None:
        """Should handle multiline descriptions properly."""
        frontmatter = {
            "title": "Test",
            "description": "Line 1\nLine 2\nLine 3",
        }

        yaml_str = frontmatter_to_yaml(frontmatter)

        # Should contain description (may be quoted or use YAML multiline syntax)
        assert "description:" in yaml_str

    def test_special_characters_escaped(self) -> None:
        """Special characters should be properly escaped."""
        frontmatter = {
            "title": "Test: With Colon",
            "description": "Has 'quotes' and \"double quotes\"",
        }

        yaml_str = frontmatter_to_yaml(frontmatter)

        # Should be valid YAML (properly escaped)
        assert "title:" in yaml_str
        assert "description:" in yaml_str

    def test_empty_dict(self) -> None:
        """Should handle empty frontmatter dict."""
        yaml_str = frontmatter_to_yaml({})
        assert yaml_str == "" or yaml_str == "{}\n"

    def test_preserves_field_order(self) -> None:
        """Should preserve field order from dict."""
        from collections import OrderedDict

        frontmatter = OrderedDict(
            [
                ("title", "First"),
                ("source", "Second"),
                ("description", "Third"),
            ]
        )

        yaml_str = frontmatter_to_yaml(frontmatter)

        # Check order in output
        title_pos = yaml_str.find("title:")
        source_pos = yaml_str.find("source:")
        desc_pos = yaml_str.find("description:")

        assert title_pos < source_pos < desc_pos


class TestBuildFrontmatterConsistency:
    """Tests for frontmatter consistency fixes.

    Issue 1: External strategy metadata (defuddle author, etc.) should be preserved.
    Issue 2: markitai_processed format must be consistent; fetch_strategy in .llm.md.
    Issue 3: YAML special characters must be properly escaped.
    """

    def test_fetch_strategy_accepted_by_build_frontmatter_dict(self) -> None:
        """build_frontmatter_dict should accept and include fetch_strategy."""
        result = build_frontmatter_dict(
            source="https://example.com",
            description="A page",
            fetch_strategy="defuddle",
        )
        assert result["fetch_strategy"] == "defuddle"

    def test_fetch_strategy_in_field_order(self) -> None:
        """fetch_strategy should appear after markitai_processed in output."""
        result = build_frontmatter_dict(
            source="https://example.com",
            description="A page",
            tags=["test"],
            fetch_strategy="static",
        )
        keys = list(result.keys())
        assert keys.index("markitai_processed") < keys.index("fetch_strategy")

    def test_fetch_strategy_omitted_when_none(self) -> None:
        """fetch_strategy should not appear when not provided."""
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="A doc",
        )
        assert "fetch_strategy" not in result

    def test_extra_meta_preserved(self) -> None:
        """Extra metadata from external strategies should be preserved."""
        extra = {"author": "John", "published": "2024-01-15", "domain": "example.com"}
        result = build_frontmatter_dict(
            source="https://example.com",
            description="A page",
            extra_meta=extra,
        )
        assert result["author"] == "John"
        assert result["published"] == "2024-01-15"
        assert result["domain"] == "example.com"

    def test_extra_meta_does_not_override_canonical_fields(self) -> None:
        """Extra metadata must not override canonical fields like title/source."""
        extra = {"title": "WRONG", "source": "WRONG", "markitai_processed": "WRONG"}
        result = build_frontmatter_dict(
            source="https://example.com",
            description="A page",
            title="Correct Title",
            extra_meta=extra,
        )
        assert result["title"] == "Correct Title"
        assert result["source"] == "https://example.com"
        assert result["markitai_processed"] != "WRONG"

    def test_extra_meta_filters_unreliable_language(self) -> None:
        """Language from HTML meta tags is unreliable and must be excluded."""
        extra = {"author": "John", "language": "en-us", "domain": "example.com"}
        result = build_frontmatter_dict(
            source="https://example.com",
            description="A page",
            extra_meta=extra,
        )
        assert "language" not in result
        assert result["author"] == "John"

    def test_extra_meta_appears_after_canonical_fields(self) -> None:
        """Extra metadata fields should come after canonical fields in order."""
        extra = {"author": "John"}
        result = build_frontmatter_dict(
            source="https://example.com",
            description="A page",
            extra_meta=extra,
        )
        keys = list(result.keys())
        assert keys.index("markitai_processed") < keys.index("author")

    def test_timestamp_format_consistent(self) -> None:
        """markitai_processed should use ISO 8601 with milliseconds + timezone."""
        import re

        result = build_frontmatter_dict(
            source="doc.pdf",
            description="A doc",
        )
        ts = result["markitai_processed"]
        # Should match: 2026-03-06T14:20:21.123+08:00
        assert re.match(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2}$", ts
        ), f"Timestamp format wrong: {ts}"

    def test_yaml_special_chars_in_title(self) -> None:
        """Titles with YAML special characters should produce valid YAML."""
        import yaml

        result = build_frontmatter_dict(
            source="doc.pdf",
            description="A doc",
            title='Title: with "quotes" and [brackets]',
        )
        yaml_str = frontmatter_to_yaml(result)
        # Must be parseable
        parsed = yaml.safe_load(yaml_str)
        assert parsed["title"] == 'Title: with "quotes" and [brackets]'

    def test_yaml_special_chars_in_extra_meta(self) -> None:
        """Extra meta values with special chars should produce valid YAML."""
        import yaml

        extra = {"author": "O'Brien & Sons: Ltd."}
        result = build_frontmatter_dict(
            source="doc.pdf",
            description="A doc",
            extra_meta=extra,
        )
        yaml_str = frontmatter_to_yaml(result)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["author"] == "O'Brien & Sons: Ltd."
