"""Tests for markdown formatter module."""

from markit.markdown.formatter import (
    FormatterConfig,
    MarkdownCleaner,
    MarkdownFormatter,
    clean_markdown,
    format_markdown,
)


class TestFormatterConfig:
    """Tests for FormatterConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FormatterConfig()
        assert config.heading_style == "atx"
        assert config.max_heading_level == 6
        assert config.ensure_h2_start is True
        assert config.unordered_list_marker == "-"
        assert config.ordered_list_delimiter == "."
        assert config.line_width == 0
        assert config.trailing_whitespace is False
        assert config.blank_lines_after_heading == 1
        assert config.blank_lines_between_paragraphs == 1
        assert config.max_consecutive_blank_lines == 2
        assert config.code_block_style == "fenced"
        assert config.code_fence_char == "`"
        assert config.link_style == "inline"
        assert config.table_pipe_padding is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FormatterConfig(
            heading_style="setext",
            unordered_list_marker="*",
            code_fence_char="~",
            table_pipe_padding=False,
        )
        assert config.heading_style == "setext"
        assert config.unordered_list_marker == "*"
        assert config.code_fence_char == "~"
        assert config.table_pipe_padding is False


class TestMarkdownFormatter:
    """Tests for MarkdownFormatter class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        formatter = MarkdownFormatter()
        assert formatter.config is not None
        assert isinstance(formatter.config, FormatterConfig)

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = FormatterConfig(max_consecutive_blank_lines=1)
        formatter = MarkdownFormatter(config)
        assert formatter.config.max_consecutive_blank_lines == 1

    def test_format_basic(self):
        """Test basic formatting."""
        formatter = MarkdownFormatter()
        result = formatter.format("# Title\n\nSome text")
        assert result.endswith("\n")

    def test_normalize_line_endings_crlf(self):
        """Test CRLF line ending normalization."""
        formatter = MarkdownFormatter()
        result = formatter._normalize_line_endings("Line1\r\nLine2\r\n")
        assert result == "Line1\nLine2\n"

    def test_normalize_line_endings_cr(self):
        """Test CR line ending normalization."""
        formatter = MarkdownFormatter()
        result = formatter._normalize_line_endings("Line1\rLine2\r")
        assert result == "Line1\nLine2\n"

    def test_normalize_blank_lines(self):
        """Test blank line normalization."""
        formatter = MarkdownFormatter()
        text = "Line1\n\n\n\n\nLine2"  # 4 blank lines
        result = formatter._normalize_blank_lines(text)
        # Should reduce to max_consecutive_blank_lines (2)
        assert result.count("\n") <= 4  # max 2 blank lines + 1 for separation

    def test_fix_heading_levels_h1_to_h2(self):
        """Test h1 to h2 conversion."""
        formatter = MarkdownFormatter()
        result = formatter._fix_heading_levels("# Title")
        assert result == "## Title"

    def test_fix_heading_levels_multiple_h1(self):
        """Test multiple h1 conversion."""
        formatter = MarkdownFormatter()
        result = formatter._fix_heading_levels("# Title1\n# Title2")
        assert result == "## Title1\n## Title2"

    def test_fix_heading_levels_preserve_h2_plus(self):
        """Test h2+ headings are preserved when not starting with h1."""
        formatter = MarkdownFormatter()
        # When document doesn't start with h1, headings are preserved
        result = formatter._fix_heading_levels("## Title\n### Subtitle")
        assert result == "## Title\n### Subtitle"

    def test_fix_heading_levels_shift_all_when_h1_start(self):
        """Test all headings shift down when starting with h1."""
        formatter = MarkdownFormatter()
        # When document starts with h1, ALL headings shift down one level
        result = formatter._fix_heading_levels("# Main\n## Section\n### Sub")
        assert result == "## Main\n### Section\n#### Sub"

    def test_fix_heading_levels_h6_capped(self):
        """Test h6 remains h6 when shifting (no h7)."""
        formatter = MarkdownFormatter()
        result = formatter._fix_heading_levels("# Main\n###### Deep")
        assert result == "## Main\n###### Deep"  # h6 stays h6

    def test_fix_heading_levels_disabled(self):
        """Test heading level fix can be disabled."""
        config = FormatterConfig(ensure_h2_start=False)
        formatter = MarkdownFormatter(config)
        result = formatter._fix_heading_levels("# Title")
        assert result == "# Title"

    def test_normalize_headings_trailing_hashes(self):
        """Test removing trailing hashes from headings."""
        formatter = MarkdownFormatter()
        result = formatter._normalize_headings("## Title ##")
        assert "##" not in result.split(" ", 1)[1] if " " in result else True

    def test_normalize_headings_spacing(self):
        """Test heading spacing normalization."""
        formatter = MarkdownFormatter()
        result = formatter._normalize_headings("##  Title  ")
        assert "##" in result and "Title" in result

    def test_normalize_lists_dash_marker(self):
        """Test list marker normalization to dash."""
        config = FormatterConfig(unordered_list_marker="-")
        formatter = MarkdownFormatter(config)
        result = formatter._normalize_lists("* Item1\n+ Item2")
        assert "- Item1" in result
        assert "- Item2" in result

    def test_normalize_lists_asterisk_marker(self):
        """Test list marker normalization to asterisk."""
        config = FormatterConfig(unordered_list_marker="*")
        formatter = MarkdownFormatter(config)
        result = formatter._normalize_lists("- Item1\n+ Item2")
        assert "* Item1" in result
        assert "* Item2" in result

    def test_normalize_lists_plus_marker(self):
        """Test list marker normalization to plus."""
        config = FormatterConfig(unordered_list_marker="+")
        formatter = MarkdownFormatter(config)
        result = formatter._normalize_lists("- Item1\n* Item2")
        assert "+ Item1" in result
        assert "+ Item2" in result

    def test_normalize_lists_indented(self):
        """Test indented list normalization."""
        config = FormatterConfig(unordered_list_marker="-")
        formatter = MarkdownFormatter(config)
        result = formatter._normalize_lists("  * Nested item")
        assert "  - Nested item" in result

    def test_normalize_code_blocks_backticks(self):
        """Test code block normalization to backticks."""
        config = FormatterConfig(code_fence_char="`")
        formatter = MarkdownFormatter(config)
        result = formatter._normalize_code_blocks("~~~python\ncode\n~~~")
        assert "```" in result
        assert "~~~" not in result

    def test_normalize_code_blocks_tildes(self):
        """Test code block normalization to tildes."""
        config = FormatterConfig(code_fence_char="~")
        formatter = MarkdownFormatter(config)
        result = formatter._normalize_code_blocks("```python\ncode\n```")
        assert "~~~" in result
        assert "```" not in result

    def test_normalize_tables_basic(self):
        """Test basic table normalization."""
        formatter = MarkdownFormatter()
        table = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = formatter._normalize_tables(table)
        assert "|" in result
        assert "A" in result
        assert "B" in result

    def test_normalize_tables_at_end(self):
        """Test table at end of document."""
        formatter = MarkdownFormatter()
        text = "Text\n\n| A |\n|---|\n| 1 |"
        result = formatter._normalize_tables(text)
        assert "Text" in result
        assert "|" in result

    def test_format_table_empty(self):
        """Test formatting empty table."""
        formatter = MarkdownFormatter()
        result = formatter._format_table([])
        assert result == []

    def test_format_table_with_padding(self):
        """Test table formatting with padding."""
        formatter = MarkdownFormatter()
        table_lines = ["| A | B |", "|---|---|", "| 1 | 2 |"]
        result = formatter._format_table(table_lines)
        assert len(result) == 3
        for line in result:
            assert line.startswith("|")
            assert line.endswith("|")

    def test_format_table_no_padding(self):
        """Test table formatting without padding."""
        config = FormatterConfig(table_pipe_padding=False)
        formatter = MarkdownFormatter(config)
        table_lines = ["| A | B |", "|---|---|", "| 1 | 2 |"]
        result = formatter._format_table(table_lines)
        assert len(result) == 3

    def test_format_table_uneven_columns(self):
        """Test formatting table with uneven columns."""
        formatter = MarkdownFormatter()
        table_lines = ["| A | B | C |", "|---|---|---|", "| 1 |"]
        result = formatter._format_table(table_lines)
        assert len(result) == 3

    def test_clean_trailing_whitespace(self):
        """Test trailing whitespace removal."""
        formatter = MarkdownFormatter()
        result = formatter._clean_trailing_whitespace("Line1   \nLine2  ")
        assert result == "Line1\nLine2"

    def test_clean_trailing_whitespace_preserved(self):
        """Test trailing whitespace preservation when configured."""
        config = FormatterConfig(trailing_whitespace=True)
        formatter = MarkdownFormatter(config)
        result = formatter._clean_trailing_whitespace("Line1   \nLine2  ")
        assert result == "Line1   \nLine2  "

    def test_ensure_final_newline(self):
        """Test final newline is ensured."""
        formatter = MarkdownFormatter()
        result = formatter._ensure_final_newline("Text")
        assert result == "Text\n"

    def test_ensure_final_newline_no_extra(self):
        """Test no extra newlines added."""
        formatter = MarkdownFormatter()
        result = formatter._ensure_final_newline("Text\n\n\n")
        assert result == "Text\n"


class TestMarkdownCleaner:
    """Tests for MarkdownCleaner class.

    Note: MarkdownCleaner now only handles format-level cleaning.
    Content cleaning (page numbers, separators, chart residue) is delegated to LLM.
    """

    def test_init(self):
        """Test cleaner initialization."""
        cleaner = MarkdownCleaner()
        assert cleaner._patterns is not None
        assert "zero_width" in cleaner._patterns
        assert "empty_links" in cleaner._patterns
        assert "html_comments" in cleaner._patterns

    def test_compile_patterns(self):
        """Test pattern compilation."""
        cleaner = MarkdownCleaner()
        patterns = cleaner._compile_patterns()
        # Format-level patterns only
        assert "zero_width" in patterns
        assert "empty_links" in patterns
        assert "html_comments" in patterns
        # Content-level patterns removed (delegated to LLM)
        assert "page_numbers" not in patterns
        assert "separator_lines" not in patterns
        assert "repeated_chars" not in patterns

    def test_clean_zero_width_chars(self):
        """Test zero-width character removal."""
        cleaner = MarkdownCleaner()
        text = "Hello\u200bWorld\u200c\u200d\ufeff"
        result = cleaner.clean(text)
        assert "\u200b" not in result
        assert "\u200c" not in result
        assert "\u200d" not in result
        assert "\ufeff" not in result

    def test_clean_empty_links(self):
        """Test empty link removal."""
        cleaner = MarkdownCleaner()
        result = cleaner.clean("Text []()")
        assert "[]()" not in result

    def test_clean_html_comments(self):
        """Test HTML comment removal."""
        cleaner = MarkdownCleaner()
        result = cleaner.clean("Text <!-- comment --> more")
        assert "<!-- comment -->" not in result

    def test_clean_html_comments_multiline(self):
        """Test multiline HTML comment removal."""
        cleaner = MarkdownCleaner()
        result = cleaner.clean("Text <!-- \nmultiline\n --> more")
        assert "multiline" not in result

    def test_clean_html_comments_disabled(self):
        """Test HTML comment removal can be disabled."""
        cleaner = MarkdownCleaner()
        result = cleaner.clean("<!-- comment -->", remove_html_comments=False)
        assert "<!-- comment -->" in result

    def test_clean_multiple_blank_lines(self):
        """Test multiple blank lines collapsed."""
        cleaner = MarkdownCleaner()
        result = cleaner.clean("A\n\n\n\n\nB")  # 4 blank lines
        assert result.count("\n\n\n") == 0  # Max 2 consecutive

    def test_clean_returns_stripped(self):
        """Test result is stripped."""
        cleaner = MarkdownCleaner()
        result = cleaner.clean("  Content  ")
        assert result == "Content"

    def test_clean_preserves_page_numbers(self):
        """Test page numbers are preserved (LLM responsibility)."""
        cleaner = MarkdownCleaner()
        result = cleaner.clean("Content\nPage 1\nMore content")
        # Page numbers are now preserved - LLM handles content cleaning
        assert "Page 1" in result

    def test_clean_preserves_separators(self):
        """Test separator lines are preserved."""
        cleaner = MarkdownCleaner()
        result = cleaner.clean("Content\n----------\nMore")
        # Separators are preserved
        assert "----------" in result


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_format_markdown_default(self):
        """Test format_markdown with default config."""
        result = format_markdown("# Title")
        assert "##" in result  # h1 -> h2
        assert result.endswith("\n")

    def test_format_markdown_custom_config(self):
        """Test format_markdown with custom config."""
        config = FormatterConfig(ensure_h2_start=False)
        result = format_markdown("# Title", config)
        assert result.startswith("# ")

    def test_clean_markdown(self):
        """Test clean_markdown function."""
        result = clean_markdown("Text []()")
        assert "[]()" not in result


class TestFormatterIntegration:
    """Integration tests for formatter."""

    def test_full_document_format(self):
        """Test formatting a full document."""
        document = """# Main Title

* Item 1
* Item 2

~~~python
code
~~~

| A | B |
|---|---|
| 1 | 2 |

Text
"""
        formatter = MarkdownFormatter()
        result = formatter.format(document)

        # Check h1 -> h2
        assert "## Main Title" in result
        # Check list markers
        assert "- Item" in result
        # Check code blocks (default is backticks)
        assert "```" in result
        # Check final newline
        assert result.endswith("\n")
        # Check trailing whitespace removed
        assert "Text   " not in result

    def test_preserves_content(self):
        """Test that content is preserved during formatting."""
        content = "Important text with **bold** and *italic*"
        formatter = MarkdownFormatter()
        result = formatter.format(content)
        assert "Important text" in result
        assert "**bold**" in result
        assert "*italic*" in result
