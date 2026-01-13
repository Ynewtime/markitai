"""Markdown formatting utilities.

Provides tools for cleaning, normalizing, and formatting Markdown content
according to GFM (GitHub Flavored Markdown) specifications.
"""

import re
from dataclasses import dataclass


@dataclass
class FormatterConfig:
    """Configuration for Markdown formatter."""

    # Heading settings
    heading_style: str = "atx"  # atx (#) or setext (underline)
    max_heading_level: int = 6
    ensure_h2_start: bool = True  # Start content with h2, not h1

    # List settings
    unordered_list_marker: str = "-"  # - or * or +
    ordered_list_delimiter: str = "."  # . or )

    # Line settings
    line_width: int = 0  # 0 = no wrapping
    trailing_whitespace: bool = False

    # Blank line settings
    blank_lines_after_heading: int = 1
    blank_lines_between_paragraphs: int = 1
    max_consecutive_blank_lines: int = 2

    # Code block settings
    code_block_style: str = "fenced"  # fenced or indented
    code_fence_char: str = "`"  # ` or ~

    # Link settings
    link_style: str = "inline"  # inline or reference

    # Table settings
    table_pipe_padding: bool = True


class MarkdownFormatter:
    """Format Markdown content according to GFM specifications."""

    def __init__(self, config: FormatterConfig | None = None):
        """Initialize the formatter.

        Args:
            config: Formatting configuration
        """
        self.config = config or FormatterConfig()

    def format(self, markdown: str) -> str:
        """Format Markdown content.

        Args:
            markdown: Raw Markdown content

        Returns:
            Formatted Markdown content
        """
        result = markdown

        # Apply formatting rules in order
        result = self._normalize_line_endings(result)
        result = self._normalize_blank_lines(result)
        result = self._fix_heading_levels(result)
        result = self._normalize_headings(result)
        result = self._normalize_lists(result)
        result = self._normalize_code_blocks(result)
        result = self._normalize_tables(result)
        result = self._clean_trailing_whitespace(result)
        result = self._ensure_final_newline(result)

        return result

    def _normalize_line_endings(self, text: str) -> str:
        """Normalize line endings to Unix style."""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _normalize_blank_lines(self, text: str) -> str:
        """Normalize consecutive blank lines."""
        max_blanks = self.config.max_consecutive_blank_lines

        # Replace more than max consecutive blank lines
        pattern = r"\n{" + str(max_blanks + 2) + r",}"
        replacement = "\n" * (max_blanks + 1)

        return re.sub(pattern, replacement, text)

    def _fix_heading_levels(self, text: str) -> str:
        """Fix heading levels to start from h2.

        If the document starts with h1, shift ALL headings down one level
        to maintain relative hierarchy:
        - h1 → h2
        - h2 → h3
        - h3 → h4
        - h4 → h5
        - h5 → h6
        - h6 → h6 (cannot go lower)
        """
        if not self.config.ensure_h2_start:
            return text

        lines = text.split("\n")

        # First pass: check if document starts with h1
        starts_with_h1 = False
        for line in lines:
            match = re.match(r"^(#{1,6})\s+.+$", line)
            if match:
                if len(match.group(1)) == 1:
                    starts_with_h1 = True
                break  # Only check first heading

        if not starts_with_h1:
            return text  # No adjustment needed

        # Second pass: shift all headings down one level
        result_lines = []
        for line in lines:
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                level = len(match.group(1))
                content = match.group(2)
                # Shift down one level, max 6
                new_level = min(level + 1, 6)
                result_lines.append(f"{'#' * new_level} {content}")
            else:
                result_lines.append(line)

        return "\n".join(result_lines)

    def _normalize_headings(self, text: str) -> str:
        """Normalize heading format and spacing."""
        lines = text.split("\n")
        result_lines = []

        for _i, line in enumerate(lines):
            # Match ATX headings
            match = re.match(r"^(#{1,6})\s*(.+?)\s*#*$", line)
            if match:
                hashes = match.group(1)
                content = match.group(2).strip()

                # Normalize: # Heading (no trailing #)
                normalized = f"{hashes} {content}"
                result_lines.append(normalized)
            else:
                result_lines.append(line)

        # Add blank lines around headings
        final_lines = []
        for i, line in enumerate(result_lines):
            is_heading = re.match(r"^#{1,6}\s+", line)
            prev_is_heading = i > 0 and re.match(r"^#{1,6}\s+", result_lines[i - 1])

            if is_heading:
                # Add blank line before heading (if not first line and prev line has content)
                if i > 0 and final_lines and final_lines[-1].strip():
                    final_lines.append("")

                final_lines.append(line)
            else:
                # Add blank line after heading (if prev was heading and current has content)
                if prev_is_heading and line.strip() and final_lines and final_lines[-1].strip():
                    final_lines.append("")

                final_lines.append(line)

        return "\n".join(final_lines)

    def _normalize_lists(self, text: str) -> str:
        """Normalize list markers."""
        marker = self.config.unordered_list_marker

        # Replace other unordered list markers
        if marker == "-":
            text = re.sub(r"^(\s*)[*+]\s+", r"\1- ", text, flags=re.MULTILINE)
        elif marker == "*":
            text = re.sub(r"^(\s*)[-+]\s+", r"\1* ", text, flags=re.MULTILINE)
        elif marker == "+":
            text = re.sub(r"^(\s*)[-*]\s+", r"\1+ ", text, flags=re.MULTILINE)

        return text

    def _normalize_code_blocks(self, text: str) -> str:
        """Normalize code block format."""
        fence_char = self.config.code_fence_char

        # Normalize fence characters
        if fence_char == "`":
            text = re.sub(r"^~~~", "```", text, flags=re.MULTILINE)
        elif fence_char == "~":
            text = re.sub(r"^```", "~~~", text, flags=re.MULTILINE)

        return text

    def _normalize_tables(self, text: str) -> str:
        """Normalize table formatting."""
        lines = text.split("\n")
        result_lines = []
        in_table = False
        table_lines = []

        for line in lines:
            # Detect table row
            if re.match(r"^\s*\|.*\|\s*$", line):
                in_table = True
                table_lines.append(line)
            else:
                if in_table and table_lines:
                    # Process collected table
                    formatted_table = self._format_table(table_lines)
                    result_lines.extend(formatted_table)
                    table_lines = []
                    in_table = False

                result_lines.append(line)

        # Handle table at end of document
        if table_lines:
            formatted_table = self._format_table(table_lines)
            result_lines.extend(formatted_table)

        return "\n".join(result_lines)

    def _format_table(self, table_lines: list[str]) -> list[str]:
        """Format a Markdown table."""
        if not table_lines:
            return []

        # Parse table into cells
        rows = []
        for line in table_lines:
            # Remove leading/trailing pipes and split
            cells = line.strip().strip("|").split("|")
            cells = [cell.strip() for cell in cells]
            rows.append(cells)

        if not rows:
            return table_lines

        # Calculate column widths
        col_count = max(len(row) for row in rows)
        col_widths = [0] * col_count

        for row in rows:
            for i, cell in enumerate(row):
                if i < col_count:
                    # Skip separator row for width calculation
                    if not re.match(r"^[-:]+$", cell):
                        col_widths[i] = max(col_widths[i], len(cell))

        # Minimum width of 3 for separator
        col_widths = [max(w, 3) for w in col_widths]

        # Format rows
        formatted = []
        for _row_idx, row in enumerate(rows):
            # Pad row to correct column count
            while len(row) < col_count:
                row.append("")

            if self.config.table_pipe_padding:
                cells = []
                for i, cell in enumerate(row):
                    if re.match(r"^[-:]+$", cell):
                        # Separator row
                        cells.append("-" * col_widths[i])
                    else:
                        cells.append(cell.ljust(col_widths[i]))
                formatted.append("| " + " | ".join(cells) + " |")
            else:
                formatted.append("|" + "|".join(row) + "|")

        return formatted

    def _clean_trailing_whitespace(self, text: str) -> str:
        """Remove trailing whitespace from lines."""
        if self.config.trailing_whitespace:
            return text

        lines = text.split("\n")
        return "\n".join(line.rstrip() for line in lines)

    def _ensure_final_newline(self, text: str) -> str:
        """Ensure file ends with exactly one newline."""
        return text.rstrip() + "\n"


class MarkdownCleaner:
    """Clean formatting issues from Markdown.

    This cleaner only handles FORMAT issues (invisible characters, broken syntax).
    Content cleaning (page numbers, separators, repeated text) is delegated to LLM.

    Principle: "程序提取，LLM 清理" - Programs extract, LLM cleans content.
    """

    def __init__(self):
        """Initialize the cleaner."""
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> dict[str, re.Pattern]:
        """Compile regex patterns for cleaning.

        Only format-related patterns are included:
        - zero_width: Invisible Unicode characters that break rendering
        - empty_links: Broken markdown link syntax []()
        - html_comments: Used internally for PARTIAL_METADATA markers
        """
        return {
            # Zero-width characters (invisible, break rendering)
            "zero_width": re.compile(r"[\u200b\u200c\u200d\ufeff]"),
            # Empty links (broken syntax)
            "empty_links": re.compile(r"\[]\(\)"),
            # HTML comments (used for internal markers like PARTIAL_METADATA)
            "html_comments": re.compile(r"<!--.*?-->", re.DOTALL),
        }

    def clean(
        self,
        markdown: str,
        remove_html_comments: bool = True,
    ) -> str:
        """Clean Markdown formatting issues.

        Args:
            markdown: Markdown content to clean
            remove_html_comments: Remove HTML comments (default True)

        Returns:
            Cleaned Markdown content
        """
        result = markdown

        # Remove zero-width characters (always)
        result = self._patterns["zero_width"].sub("", result)

        # Remove empty links (always)
        result = self._patterns["empty_links"].sub("", result)

        # Remove HTML comments (optional, used for PARTIAL_METADATA cleanup)
        if remove_html_comments:
            result = self._patterns["html_comments"].sub("", result)

        # Clean up multiple blank lines that may have been created
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result.strip()


def format_markdown(
    content: str,
    config: FormatterConfig | None = None,
) -> str:
    """Convenience function to format Markdown.

    Args:
        content: Markdown content
        config: Optional formatter configuration

    Returns:
        Formatted Markdown
    """
    formatter = MarkdownFormatter(config)
    return formatter.format(content)


def clean_markdown(content: str) -> str:
    """Convenience function to clean Markdown.

    Args:
        content: Markdown content

    Returns:
        Cleaned Markdown
    """
    cleaner = MarkdownCleaner()
    return cleaner.clean(content)
