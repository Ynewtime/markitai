"""Tests for vision utility functions in markitai.llm.vision module."""

from __future__ import annotations

from markitai.llm.vision import _context_display_name


class TestContextDisplayName:
    """Tests for _context_display_name function."""

    def test_empty_context_returns_batch(self):
        """Test that empty string returns 'batch' as default."""
        result = _context_display_name("")
        assert result == "batch"

    def test_simple_filename_returns_as_is(self):
        """Test that simple filename without path is returned as-is."""
        result = _context_display_name("document.pdf")
        assert result == "document.pdf"

    def test_path_with_slashes_returns_last_component(self):
        """Test that path with forward slashes returns last component."""
        result = _context_display_name("path/to/file.pdf")
        assert result == "file.pdf"

    def test_absolute_path_returns_filename(self):
        """Test that absolute path returns just the filename."""
        result = _context_display_name("/home/user/documents/report.pdf")
        assert result == "report.pdf"

    def test_windows_style_path_not_handled(self):
        """Test that Windows-style path with backslashes is NOT handled.

        The function uses forward slash split only, so Windows paths
        are returned as-is (backslashes are not treated as separators).
        """
        result = _context_display_name(r"C:\Users\docs\file.pdf")
        # Backslashes are not split, so full string is returned
        assert result == r"C:\Users\docs\file.pdf"

    def test_trailing_slash_returns_empty_string(self):
        """Test that trailing slash results in empty string from split."""
        result = _context_display_name("path/to/directory/")
        assert result == ""

    def test_multiple_slashes_returns_last_component(self):
        """Test that multiple consecutive slashes are handled."""
        result = _context_display_name("path//to///file.pdf")
        assert result == "file.pdf"

    def test_single_slash_returns_second_part(self):
        """Test that single slash path returns component after slash."""
        result = _context_display_name("/filename.pdf")
        assert result == "filename.pdf"
