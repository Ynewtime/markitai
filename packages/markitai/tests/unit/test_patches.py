"""Tests for converter/_patches.py compatibility patches."""

from __future__ import annotations

from unittest.mock import patch


class TestApplyOpenpyxlPatches:
    """Tests for openpyxl FileVersion 'bg' kwarg patch."""

    def test_removes_bg_kwarg(self):
        """Patched FileVersion.__init__ removes 'bg' kwarg silently."""
        from openpyxl.workbook.properties import FileVersion

        from markitai.converter._patches import apply_openpyxl_patches

        apply_openpyxl_patches()

        # Should not raise even with 'bg' kwarg
        fv = FileVersion(appName="test", bg="some_value")
        assert fv.appName == "test"

    def test_preserves_normal_kwargs(self):
        """Patched FileVersion.__init__ passes non-bg kwargs through."""
        from openpyxl.workbook.properties import FileVersion

        from markitai.converter._patches import apply_openpyxl_patches

        apply_openpyxl_patches()

        fv = FileVersion(appName="Excel", lastEdited="7", lowestEdited="4")
        assert fv.appName == "Excel"

    def test_patch_is_idempotent(self):
        """Calling apply_openpyxl_patches multiple times only patches once."""
        from openpyxl.workbook.properties import FileVersion

        from markitai.converter._patches import apply_openpyxl_patches

        apply_openpyxl_patches()
        first_init = FileVersion.__init__

        apply_openpyxl_patches()
        second_init = FileVersion.__init__

        # Should be the same function (not double-wrapped)
        assert first_init is second_init

    def test_markitai_patched_flag_set(self):
        """Patch sets _markitai_patched flag on FileVersion.__init__."""
        from openpyxl.workbook.properties import FileVersion

        from markitai.converter._patches import apply_openpyxl_patches

        apply_openpyxl_patches()
        assert getattr(FileVersion.__init__, "_markitai_patched", False) is True

    def test_skips_if_openpyxl_not_installed(self):
        """Gracefully handles missing openpyxl."""
        from markitai.converter._patches import apply_openpyxl_patches

        with patch.dict("sys.modules", {"openpyxl.workbook.properties": None}):
            # Should not raise
            apply_openpyxl_patches()


class TestApplyPptxPatches:
    """Tests for pptx XML lenient parser patch."""

    def test_valid_xml_uses_original_parser(self):
        """Valid XML should use original parser, not lenient fallback."""
        import pptx.oxml

        from markitai.converter._patches import apply_pptx_patches

        apply_pptx_patches()

        # Valid XML should parse without issues
        valid_xml = b'<a:t xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">Hello</a:t>'
        result = pptx.oxml.parse_xml(valid_xml)
        assert result is not None
        assert result.text == "Hello"

    def test_malformed_xml_falls_back_to_lenient(self):
        """Malformed XML should fallback to lenient parser instead of raising."""
        import pptx.oxml

        from markitai.converter._patches import apply_pptx_patches

        apply_pptx_patches()

        # Malformed XML with mismatched tags (simulates .ppt→.pptx conversion)
        malformed_xml = b"<root><unclosed>text</root>"
        # Should not raise XMLSyntaxError
        result = pptx.oxml.parse_xml(malformed_xml)
        assert result is not None

    def test_string_xml_input_converted_to_bytes(self):
        """String XML input should be converted to bytes for lenient parser."""
        import pptx.oxml

        from markitai.converter._patches import apply_pptx_patches

        apply_pptx_patches()

        # String input with malformed XML → triggers lenient path which
        # must handle str→bytes conversion
        malformed_str = "<root><unclosed>text</root>"
        result = pptx.oxml.parse_xml(malformed_str)
        assert result is not None

    def test_patch_is_idempotent(self):
        """Calling apply_pptx_patches multiple times only patches once."""
        import pptx.oxml

        from markitai.converter._patches import apply_pptx_patches

        apply_pptx_patches()
        first_parse = pptx.oxml.parse_xml

        apply_pptx_patches()
        second_parse = pptx.oxml.parse_xml

        assert first_parse is second_parse

    def test_markitai_patched_flag_set(self):
        """Patch sets _markitai_patched flag on parse_xml."""
        import pptx.oxml

        from markitai.converter._patches import apply_pptx_patches

        apply_pptx_patches()
        assert getattr(pptx.oxml.parse_xml, "_markitai_patched", False) is True

    def test_skips_if_lxml_not_installed(self):
        """Gracefully handles missing lxml."""
        from markitai.converter._patches import apply_pptx_patches

        with patch.dict("sys.modules", {"lxml": None, "lxml.etree": None}):
            apply_pptx_patches()

    def test_skips_if_pptx_not_installed(self):
        """Gracefully handles missing python-pptx."""
        from markitai.converter._patches import apply_pptx_patches

        with patch.dict("sys.modules", {"pptx": None, "pptx.oxml": None}):
            apply_pptx_patches()


class TestApplyAllPatches:
    """Tests for the top-level apply_all_patches function."""

    def test_idempotent(self):
        """apply_all_patches is idempotent via _patches_applied flag."""
        import markitai.converter._patches as patches_mod

        # Reset state
        original_flag = patches_mod._patches_applied
        try:
            patches_mod._patches_applied = False

            with (
                patch.object(patches_mod, "apply_openpyxl_patches") as mock_openpyxl,
                patch.object(patches_mod, "apply_pptx_patches") as mock_pptx,
            ):
                patches_mod.apply_all_patches()
                assert mock_openpyxl.call_count == 1
                assert mock_pptx.call_count == 1

                # Second call should be a no-op
                patches_mod.apply_all_patches()
                assert mock_openpyxl.call_count == 1
                assert mock_pptx.call_count == 1
        finally:
            patches_mod._patches_applied = original_flag

    def test_calls_both_patches(self):
        """apply_all_patches calls both openpyxl and pptx patches."""
        import markitai.converter._patches as patches_mod

        original_flag = patches_mod._patches_applied
        try:
            patches_mod._patches_applied = False

            with (
                patch.object(patches_mod, "apply_openpyxl_patches") as mock_openpyxl,
                patch.object(patches_mod, "apply_pptx_patches") as mock_pptx,
            ):
                patches_mod.apply_all_patches()

                mock_openpyxl.assert_called_once()
                mock_pptx.assert_called_once()
        finally:
            patches_mod._patches_applied = original_flag
