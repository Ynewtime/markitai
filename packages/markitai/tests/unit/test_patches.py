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
        fv = FileVersion(appName="test", bg="some_value")  # type: ignore[reportCallIssue]
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
            ):
                patches_mod.apply_all_patches()
                assert mock_openpyxl.call_count == 1

                # Second call should be a no-op
                patches_mod.apply_all_patches()
                assert mock_openpyxl.call_count == 1
        finally:
            patches_mod._patches_applied = original_flag

    def test_calls_openpyxl_patch(self):
        """apply_all_patches calls the openpyxl patch."""
        import markitai.converter._patches as patches_mod

        original_flag = patches_mod._patches_applied
        try:
            patches_mod._patches_applied = False

            with (
                patch.object(patches_mod, "apply_openpyxl_patches") as mock_openpyxl,
            ):
                patches_mod.apply_all_patches()

                mock_openpyxl.assert_called_once()
        finally:
            patches_mod._patches_applied = original_flag
