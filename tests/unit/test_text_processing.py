"""Tests for text processing functions (filename sanitization, image spacing, etc.)."""


class TestFilenameSanitization:
    """Tests for filename sanitization function."""

    def test_sanitize_spaces(self):
        """Test that spaces are replaced with underscores."""
        from markit.converters.markitdown import _sanitize_filename

        assert _sanitize_filename("hello world") == "hello_world"
        assert _sanitize_filename("file name here") == "file_name_here"

    def test_sanitize_colons(self):
        """Test that colons are replaced with underscores."""
        from markit.converters.markitdown import _sanitize_filename

        # ASCII colon
        assert _sanitize_filename("file:name") == "file_name"
        # Full-width colon (Chinese)
        assert _sanitize_filename("file：name") == "file_name"

    def test_sanitize_special_chars(self):
        """Test that various special characters are replaced."""
        from markit.converters.markitdown import _sanitize_filename

        assert _sanitize_filename("file/name") == "file_name"
        assert _sanitize_filename("file\\name") == "file_name"
        assert _sanitize_filename("file?name") == "file_name"
        assert _sanitize_filename("file*name") == "file_name"
        assert _sanitize_filename('file"name') == "file_name"
        assert _sanitize_filename("file<name") == "file_name"
        assert _sanitize_filename("file>name") == "file_name"
        assert _sanitize_filename("file|name") == "file_name"

    def test_sanitize_chinese_filename(self):
        """Test that Chinese characters are preserved."""
        from markit.converters.markitdown import _sanitize_filename

        result = _sanitize_filename("实例化手工配置指导：Cash in业务实例化配置")
        assert result == "实例化手工配置指导_Cash_in业务实例化配置"
        # Chinese characters preserved
        assert "实例化" in result
        assert "业务" in result

    def test_sanitize_collapses_multiple_underscores(self):
        """Test that multiple consecutive underscores are collapsed."""
        from markit.converters.markitdown import _sanitize_filename

        assert _sanitize_filename("file  name") == "file_name"
        assert _sanitize_filename("file: :name") == "file_name"
        assert _sanitize_filename("a   b   c") == "a_b_c"

    def test_sanitize_empty_string(self):
        """Test sanitization of empty string."""
        from markit.converters.markitdown import _sanitize_filename

        assert _sanitize_filename("") == ""

    def test_sanitize_already_clean(self):
        """Test that already clean filenames are unchanged."""
        from markit.converters.markitdown import _sanitize_filename

        assert _sanitize_filename("clean_filename") == "clean_filename"
        assert _sanitize_filename("file123") == "file123"

    def test_sanitize_pdf_converter(self):
        """Test that PDF converter has same sanitization."""
        from markit.converters.pdf.pymupdf import _sanitize_filename

        result = _sanitize_filename("CBS 5.5 R2X自升级需求调研")
        assert result == "CBS_5.5_R2X自升级需求调研"
        assert " " not in result


class TestImageSpacingNormalization:
    """Tests for image spacing normalization function."""

    def test_add_blank_before_image(self):
        """Test that blank line is added before image when needed."""
        from markit.converters.markitdown import _normalize_image_spacing

        markdown = "Some text\n![](image.png)"
        result = _normalize_image_spacing(markdown)
        assert result == "Some text\n\n![](image.png)"

    def test_no_extra_blank_before_image(self):
        """Test that no extra blank is added if already present."""
        from markit.converters.markitdown import _normalize_image_spacing

        markdown = "Some text\n\n![](image.png)"
        result = _normalize_image_spacing(markdown)
        assert result == "Some text\n\n![](image.png)"

    def test_add_blank_after_image(self):
        """Test that blank line is added after image when needed."""
        from markit.converters.markitdown import _normalize_image_spacing

        markdown = "![](image.png)\nSome text"
        result = _normalize_image_spacing(markdown)
        assert result == "![](image.png)\n\nSome text"

    def test_no_extra_blank_after_image(self):
        """Test that no extra blank is added if already present."""
        from markit.converters.markitdown import _normalize_image_spacing

        markdown = "![](image.png)\n\nSome text"
        result = _normalize_image_spacing(markdown)
        assert result == "![](image.png)\n\nSome text"

    def test_consecutive_images_no_blank(self):
        """Test that consecutive images have no blank lines between them."""
        from markit.converters.markitdown import _normalize_image_spacing

        markdown = "![](img1.png)\n![](img2.png)\n![](img3.png)"
        result = _normalize_image_spacing(markdown)
        assert result == "![](img1.png)\n![](img2.png)\n![](img3.png)"

    def test_consecutive_images_limits_blanks(self):
        """Test that multiple blanks between consecutive images are limited to one."""
        from markit.converters.markitdown import _normalize_image_spacing

        markdown = "![](img1.png)\n\n\n![](img2.png)\n\n\n\n![](img3.png)"
        result = _normalize_image_spacing(markdown)
        # At most one blank line between consecutive images
        assert result == "![](img1.png)\n\n![](img2.png)\n\n![](img3.png)"

    def test_image_with_alt_text(self):
        """Test that images with alt text are handled correctly."""
        from markit.converters.markitdown import _normalize_image_spacing

        markdown = "Text\n![alt text](image.png)\nMore text"
        result = _normalize_image_spacing(markdown)
        assert "![alt text](image.png)" in result
        assert result.count("\n\n") == 2  # One before, one after

    def test_image_in_assets_folder(self):
        """Test that images in assets folder are handled."""
        from markit.converters.markitdown import _normalize_image_spacing

        markdown = "Text\n![](assets/image_001.png)\nMore text"
        result = _normalize_image_spacing(markdown)
        assert "![](assets/image_001.png)" in result

    def test_empty_markdown(self):
        """Test normalization of empty markdown."""
        from markit.converters.markitdown import _normalize_image_spacing

        assert _normalize_image_spacing("") == ""

    def test_no_images(self):
        """Test markdown without images is unchanged."""
        from markit.converters.markitdown import _normalize_image_spacing

        markdown = "# Title\n\nSome text\n\nMore text"
        result = _normalize_image_spacing(markdown)
        assert result == markdown

    def test_pdf_converter_normalization(self):
        """Test that PDF converter has same normalization."""
        from markit.converters.pdf.pymupdf import _normalize_image_spacing

        markdown = "Text\n![](img1.png)\n![](img2.png)\nMore"
        result = _normalize_image_spacing(markdown)
        # Consecutive images should not have blank between
        assert "![](img1.png)\n![](img2.png)" in result


class TestExcelNaNCleanup:
    """Tests for Excel NaN cleanup function."""

    def test_clean_nan_in_table(self):
        """Test that NaN values in table cells are cleaned."""
        from markit.converters.markitdown import MarkItDownConverter

        converter = MarkItDownConverter()
        markdown = "| Name | Value |\n|------|-------|\n| Test | NaN |"
        result = converter._clean_excel_markdown(markdown)
        assert "NaN" not in result
        assert "| Test |" in result

    def test_clean_multiple_nan(self):
        """Test cleaning multiple NaN values."""
        from markit.converters.markitdown import MarkItDownConverter

        converter = MarkItDownConverter()
        markdown = "| A | B | C |\n|---|---|---|\n| NaN | NaN | NaN |"
        result = converter._clean_excel_markdown(markdown)
        assert result.count("NaN") == 0

    def test_preserve_nan_in_content(self):
        """Test that NaN within content is preserved."""
        from markit.converters.markitdown import MarkItDownConverter

        converter = MarkItDownConverter()
        markdown = "| Description |\n|-------------|\n| NaNometer reading |"
        result = converter._clean_excel_markdown(markdown)
        # NaN as part of word should be preserved
        assert "NaNometer" in result

    def test_clean_nan_with_whitespace(self):
        """Test cleaning NaN with surrounding whitespace."""
        from markit.converters.markitdown import MarkItDownConverter

        converter = MarkItDownConverter()
        markdown = "| A |\n|---|\n|  NaN  |"
        result = converter._clean_excel_markdown(markdown)
        assert "NaN" not in result

    def test_non_table_content_unchanged(self):
        """Test that non-table content is unchanged."""
        from markit.converters.markitdown import MarkItDownConverter

        converter = MarkItDownConverter()
        markdown = "# Title\n\nSome paragraph text."
        result = converter._clean_excel_markdown(markdown)
        assert result == markdown
