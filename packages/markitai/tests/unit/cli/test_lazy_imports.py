"""Tests for lazy __getattr__ imports in package __init__ modules."""

from __future__ import annotations

import pytest


class TestCliLazyImports:
    """Tests for markitai.cli lazy __getattr__."""

    def test_import_ui(self) -> None:
        """Should be able to import ui from markitai.cli."""
        from markitai.cli import ui

        assert hasattr(ui, "build_feature_str")

    def test_import_nonexistent_raises_error(self) -> None:
        """Should raise ImportError for nonexistent attribute."""
        import markitai.cli as cli_mod

        with pytest.raises(AttributeError, match="nonexistent"):
            cli_mod.__getattr__("nonexistent")


class TestWorkflowLazyImports:
    """Tests for markitai.workflow lazy __getattr__."""

    def test_import_convert_document_core(self) -> None:
        """Should lazily import convert_document_core."""
        from markitai.workflow import convert_document_core

        assert callable(convert_document_core)

    def test_import_write_assets_json_alias(self) -> None:
        """Should lazily import write_assets_json as alias for write_images_json."""
        from markitai.workflow import write_assets_json

        assert callable(write_assets_json)

    def test_import_nonexistent_raises_error(self) -> None:
        """Should raise AttributeError for nonexistent attribute."""
        import markitai.workflow as wf_mod

        with pytest.raises(AttributeError, match="nonexistent"):
            wf_mod.__getattr__("nonexistent")


class TestProcessorsLazyImports:
    """Tests for markitai.cli.processors lazy __getattr__."""

    def test_import_process_batch(self) -> None:
        """Should lazily import process_batch."""
        from markitai.cli.processors import process_batch

        assert callable(process_batch)

    def test_import_nonexistent_raises_error(self) -> None:
        """Should raise AttributeError for nonexistent attribute."""
        import markitai.cli.processors as proc_mod

        with pytest.raises(AttributeError, match="nonexistent"):
            proc_mod.__getattr__("nonexistent")
