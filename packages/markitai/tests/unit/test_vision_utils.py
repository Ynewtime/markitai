"""Tests for context_display_name in markitai.llm.models module.

Previously tested _context_display_name from vision.py, which was a simplified
duplicate. Now tests the canonical context_display_name from models.py.
"""

from __future__ import annotations

from markitai.llm.models import context_display_name


class TestContextDisplayName:
    """Tests for context_display_name function."""

    def test_empty_context_returns_empty(self):
        """Test that empty string returns empty string."""
        result = context_display_name("")
        assert result == ""

    def test_simple_filename_returns_as_is(self):
        """Test that simple filename without path is returned as-is."""
        result = context_display_name("document.pdf")
        assert result == "document.pdf"

    def test_path_with_slashes_returns_last_component(self):
        """Test that path with forward slashes returns last component."""
        result = context_display_name("path/to/file.pdf")
        assert result == "file.pdf"

    def test_absolute_path_returns_filename(self):
        """Test that absolute path returns just the filename."""
        result = context_display_name("/home/user/documents/report.pdf")
        assert result == "report.pdf"

    def test_windows_style_path(self):
        """Test that Windows-style paths are handled via Path().name."""
        result = context_display_name(r"C:\Users\docs\file.pdf")
        assert result == "file.pdf"

    def test_trailing_slash_returns_empty_string(self):
        """Test that trailing slash results in empty string from Path().name."""
        result = context_display_name("path/to/directory/")
        assert result == "directory"

    def test_multiple_slashes_returns_last_component(self):
        """Test that multiple consecutive slashes are handled."""
        result = context_display_name("path//to///file.pdf")
        assert result == "file.pdf"

    def test_single_slash_returns_second_part(self):
        """Test that single slash path returns component after slash."""
        result = context_display_name("/filename.pdf")
        assert result == "filename.pdf"

    def test_context_with_suffix(self):
        """Test that colon-suffixed context preserves the suffix."""
        result = context_display_name("/path/to/file.pdf:images")
        assert result == "file.pdf:images"
