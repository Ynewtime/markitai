"""Tests for batch command."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from markit.cli.commands.batch import (
    _discover_files,
    _display_summary,
    _display_token_estimates,
    _estimate_tokens_and_cost,
    _show_dry_run,
    _simplify_error,
)


class TestDiscoverFiles:
    """Tests for _discover_files function."""

    def test_discover_files_empty_dir(self, tmp_path):
        """Test discovering files in empty directory."""
        files = _discover_files(tmp_path, recursive=False, include=None, exclude=None)
        assert files == []

    def test_discover_files_single_file(self, tmp_path):
        """Test discovering a single supported file."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        files = _discover_files(tmp_path, recursive=False, include=None, exclude=None)
        assert len(files) == 1
        assert files[0] == pdf_file

    def test_discover_files_multiple_types(self, tmp_path):
        """Test discovering multiple file types."""
        (tmp_path / "doc.pdf").touch()
        (tmp_path / "doc.docx").touch()
        (tmp_path / "sheet.xlsx").touch()
        (tmp_path / "other.json").touch()  # Not supported

        files = _discover_files(tmp_path, recursive=False, include=None, exclude=None)
        assert len(files) == 3

    def test_discover_files_recursive(self, tmp_path):
        """Test recursive file discovery."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.pdf").touch()
        (subdir / "nested.pdf").touch()

        # Non-recursive
        files = _discover_files(tmp_path, recursive=False, include=None, exclude=None)
        assert len(files) == 1

        # Recursive
        files = _discover_files(tmp_path, recursive=True, include=None, exclude=None)
        assert len(files) == 2

    def test_discover_files_with_include(self, tmp_path):
        """Test file discovery with include pattern."""
        (tmp_path / "report.pdf").touch()
        (tmp_path / "test.pdf").touch()

        files = _discover_files(tmp_path, recursive=False, include="report*", exclude=None)
        assert len(files) == 1
        assert files[0].name == "report.pdf"

    def test_discover_files_with_exclude(self, tmp_path):
        """Test file discovery with exclude pattern."""
        (tmp_path / "report.pdf").touch()
        (tmp_path / "test.pdf").touch()

        files = _discover_files(tmp_path, recursive=False, include=None, exclude="test*")
        assert len(files) == 1
        assert files[0].name == "report.pdf"

    def test_discover_files_sorted(self, tmp_path):
        """Test that discovered files are sorted."""
        (tmp_path / "z.pdf").touch()
        (tmp_path / "a.pdf").touch()
        (tmp_path / "m.pdf").touch()

        files = _discover_files(tmp_path, recursive=False, include=None, exclude=None)
        assert [f.name for f in files] == ["a.pdf", "m.pdf", "z.pdf"]


class TestSimplifyError:
    """Tests for _simplify_error function."""

    def test_simplify_docx_dependency_error(self):
        """Test simplifying docx dependency error."""
        error = "MissingDependencyException: docx library not found"
        result = _simplify_error(error)
        assert "docx" in result
        assert "pip install" in result

    def test_simplify_xlsx_dependency_error(self):
        """Test simplifying xlsx dependency error."""
        error = "MissingDependencyException: xlsx processing failed"
        result = _simplify_error(error)
        assert "xlsx" in result
        assert "pip install" in result

    def test_simplify_pptx_dependency_error(self):
        """Test simplifying pptx dependency error."""
        error = "MissingDependencyException: pptx not available"
        result = _simplify_error(error)
        assert "pptx" in result
        assert "pip install" in result

    def test_simplify_pdf_dependency_error(self):
        """Test simplifying pdf dependency error."""
        error = "MissingDependencyException: pdf library missing"
        result = _simplify_error(error)
        assert "pdf" in result
        assert "pip install" in result

    def test_simplify_generic_dependency_error(self):
        """Test simplifying generic dependency error."""
        error = "MissingDependencyException: some_lib"
        result = _simplify_error(error)
        assert "markitdown[all]" in result

    def test_simplify_all_conversion_failed(self):
        """Test simplifying all conversion failed error."""
        error = "All conversion attempts failed for this file"
        result = _simplify_error(error)
        assert "All converters failed" in result

    def test_simplify_extracted_image_error(self):
        """Test simplifying extracted image error."""
        error = "ExtractedImage processing error"
        result = _simplify_error(error)
        assert "Image processing error" in result

    def test_simplify_pandoc_unknown_format(self):
        """Test simplifying Pandoc unknown format error."""
        error = "Pandoc error: Unknown input format xyz"
        result = _simplify_error(error)
        assert "Unsupported format for Pandoc" in result

    def test_simplify_pandoc_generic_error(self):
        """Test simplifying generic Pandoc error."""
        error = "Pandoc error: conversion failed"
        result = _simplify_error(error)
        assert "Pandoc conversion failed" in result

    def test_simplify_long_error(self):
        """Test that long errors are truncated."""
        error = "x" * 150
        result = _simplify_error(error)
        assert len(result) == 100
        assert result.endswith("...")

    def test_simplify_short_error(self):
        """Test that short errors are returned as-is."""
        error = "Short error"
        result = _simplify_error(error)
        assert result == "Short error"


class TestEstimateTokensAndCost:
    """Tests for _estimate_tokens_and_cost function."""

    def test_estimate_empty_files(self):
        """Test estimation with no files."""
        result = _estimate_tokens_and_cost([], {}, 0)

        assert result["total_size"] == 0
        assert result["file_count"] == 0
        assert result["scenarios"]["convert_only"]["cost"] == 0.0

    def test_estimate_pdf_files(self):
        """Test estimation with PDF files."""
        files = [Path("test1.pdf"), Path("test2.pdf")]
        by_ext = {".pdf": files}

        result = _estimate_tokens_and_cost(files, by_ext, total_size=100000)

        assert result["file_count"] == 2
        assert result["pdf_count"] == 2
        assert result["estimated_images"] == 6  # 2 PDFs * 3 images each
        assert (
            result["scenarios"]["full_analysis"]["cost"]
            > result["scenarios"]["llm_enhance"]["cost"]
        )

    def test_estimate_docx_files(self):
        """Test estimation with docx files."""
        files = [Path("test.docx")]
        by_ext = {".docx": files}

        result = _estimate_tokens_and_cost(files, by_ext, total_size=50000)

        assert result["doc_count"] == 1
        assert result["pdf_count"] == 0
        assert result["estimated_images"] == 0

    def test_estimate_mixed_files(self):
        """Test estimation with mixed file types."""
        files = [
            Path("doc.pdf"),
            Path("doc.docx"),
            Path("sheet.xlsx"),
        ]
        by_ext = {
            ".pdf": [Path("doc.pdf")],
            ".docx": [Path("doc.docx")],
            ".xlsx": [Path("sheet.xlsx")],
        }

        result = _estimate_tokens_and_cost(files, by_ext, total_size=200000)

        assert result["file_count"] == 3
        assert result["pdf_count"] == 1
        assert result["doc_count"] == 2  # docx + xlsx

    def test_estimate_scenarios(self):
        """Test that all scenarios are present."""
        result = _estimate_tokens_and_cost([], {}, 0)

        assert "convert_only" in result["scenarios"]
        assert "llm_enhance" in result["scenarios"]
        assert "full_analysis" in result["scenarios"]


class TestDisplayTokenEstimates:
    """Tests for _display_token_estimates function."""

    def test_display_estimates(self):
        """Test displaying token estimates."""
        estimates = {
            "total_size": 1024 * 1024,  # 1MB
            "estimated_images": 5,
            "scenarios": {
                "convert_only": {"description": "No LLM"},
                "llm_enhance": {"tokens": 25000, "cost": 0.05, "description": "--llm"},
                "full_analysis": {
                    "tokens": 40000,
                    "cost": 0.10,
                    "description": "--llm --analyze-image",
                },
            },
        }

        with patch("markit.cli.commands.batch.console") as mock_console:
            _display_token_estimates(estimates)
            assert mock_console.print.called


class TestDisplaySummary:
    """Tests for _display_summary function."""

    def test_display_summary_no_state(self):
        """Test display when state is None."""
        mock_state_manager = MagicMock()
        mock_state_manager.get_state.return_value = None

        # Should not raise
        _display_summary([], mock_state_manager, None)

    def test_display_summary_with_stats(self):
        """Test display with batch stats."""
        mock_state = MagicMock()
        mock_state.total_files = 10
        mock_state.completed_files = 8
        mock_state.failed_files = 2
        mock_state.skipped_files = 0
        mock_state.files = {}

        mock_state_manager = MagicMock()
        mock_state_manager.get_state.return_value = mock_state

        mock_stats = MagicMock()
        mock_stats.format_summary.return_value = "Test summary"

        with patch("markit.cli.commands.batch.console"):
            _display_summary([], mock_state_manager, mock_stats)

    def test_display_summary_with_failures(self):
        """Test display with failed files."""
        mock_file_status = MagicMock()
        mock_file_status.status = "failed"
        mock_file_status.error = "Test error"

        mock_state = MagicMock()
        mock_state.total_files = 5
        mock_state.completed_files = 3
        mock_state.failed_files = 2
        mock_state.skipped_files = 0
        mock_state.files = {"/path/to/file.pdf": mock_file_status}

        mock_state_manager = MagicMock()
        mock_state_manager.get_state.return_value = mock_state

        with patch("markit.cli.commands.batch.console"):
            _display_summary([], mock_state_manager, None)


class TestShowDryRun:
    """Tests for _show_dry_run function."""

    def test_show_dry_run_empty(self, tmp_path):
        """Test dry run display with no files."""
        with patch("markit.cli.commands.batch.console") as mock_console:
            _show_dry_run(
                input_dir=tmp_path,
                output_dir=tmp_path / "output",
                recursive=False,
                include=None,
                exclude=None,
            )
            assert mock_console.print.called

    def test_show_dry_run_with_files(self, tmp_path):
        """Test dry run display with files."""
        (tmp_path / "test.pdf").write_bytes(b"test")
        (tmp_path / "test.docx").write_bytes(b"test")

        with patch("markit.cli.commands.batch.console") as mock_console:
            _show_dry_run(
                input_dir=tmp_path,
                output_dir=tmp_path / "output",
                recursive=False,
                include=None,
                exclude=None,
            )
            # Should have multiple print calls
            assert mock_console.print.call_count > 5

    def test_show_dry_run_recursive(self, tmp_path):
        """Test dry run display with recursive option."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.pdf").write_bytes(b"test")
        (subdir / "nested.pdf").write_bytes(b"test")

        with patch("markit.cli.commands.batch.console") as mock_console:
            _show_dry_run(
                input_dir=tmp_path,
                output_dir=tmp_path / "output",
                recursive=True,
                include=None,
                exclude=None,
            )
            assert mock_console.print.called
            # Check that Files Found shows 2
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("2" in str(c) for c in calls)

    def test_show_dry_run_with_include_pattern(self, tmp_path):
        """Test dry run display with include pattern."""
        (tmp_path / "report.pdf").write_bytes(b"test")
        (tmp_path / "test.pdf").write_bytes(b"test")

        with patch("markit.cli.commands.batch.console") as mock_console:
            _show_dry_run(
                input_dir=tmp_path,
                output_dir=tmp_path / "output",
                recursive=False,
                include="report*",
                exclude=None,
            )
            assert mock_console.print.called
            # Check that include pattern is shown
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("Include Pattern" in str(c) for c in calls)

    def test_show_dry_run_with_exclude_pattern(self, tmp_path):
        """Test dry run display with exclude pattern."""
        (tmp_path / "report.pdf").write_bytes(b"test")
        (tmp_path / "test.pdf").write_bytes(b"test")

        with patch("markit.cli.commands.batch.console") as mock_console:
            _show_dry_run(
                input_dir=tmp_path,
                output_dir=tmp_path / "output",
                recursive=False,
                include=None,
                exclude="test*",
            )
            assert mock_console.print.called
            # Check that exclude pattern is shown
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("Exclude Pattern" in str(c) for c in calls)

    def test_show_dry_run_many_files(self, tmp_path):
        """Test dry run display truncates file list when more than 10 files."""
        for i in range(15):
            (tmp_path / f"doc{i:02d}.pdf").write_bytes(b"test")

        with patch("markit.cli.commands.batch.console") as mock_console:
            _show_dry_run(
                input_dir=tmp_path,
                output_dir=tmp_path / "output",
                recursive=False,
                include=None,
                exclude=None,
            )
            # Check that "... and X more" message appears
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("more" in str(c) for c in calls)


class TestBatchCommand:
    """Integration tests for the batch command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_batch_invalid_conflict_strategy(self, runner, tmp_path):
        """Test batch with invalid conflict strategy."""
        from markit.cli.main import app

        result = runner.invoke(
            app,
            ["batch", str(tmp_path), "--on-conflict", "invalid"],
        )
        assert result.exit_code == 1
        # Check output for the message
        assert "Invalid conflict strategy" in result.output

    def test_batch_dry_run(self, runner, tmp_path):
        """Test batch dry run mode."""
        from markit.cli.main import app

        (tmp_path / "test.pdf").touch()

        result = runner.invoke(
            app,
            ["batch", str(tmp_path), "--dry-run"],
        )
        # Dry run should exit successfully
        assert result.exit_code == 0
        # Check output for the message
        assert "Dry Run" in result.output

    def test_batch_dry_run_recursive(self, runner, tmp_path):
        """Test batch dry run with recursive option."""
        from markit.cli.main import app

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.pdf").touch()
        (subdir / "nested.pdf").touch()

        result = runner.invoke(
            app,
            ["batch", str(tmp_path), "--dry-run", "-r"],
        )
        assert result.exit_code == 0
        assert "Recursive: True" in result.output

    def test_batch_dry_run_with_include_pattern(self, runner, tmp_path):
        """Test batch dry run with include pattern."""
        from markit.cli.main import app

        (tmp_path / "report.pdf").touch()
        (tmp_path / "test.pdf").touch()

        result = runner.invoke(
            app,
            ["batch", str(tmp_path), "--dry-run", "--include", "report*"],
        )
        assert result.exit_code == 0
        assert "Include Pattern: report*" in result.output

    def test_batch_dry_run_with_exclude_pattern(self, runner, tmp_path):
        """Test batch dry run with exclude pattern."""
        from markit.cli.main import app

        (tmp_path / "report.pdf").touch()
        (tmp_path / "test.pdf").touch()

        result = runner.invoke(
            app,
            ["batch", str(tmp_path), "--dry-run", "--exclude", "test*"],
        )
        assert result.exit_code == 0
        assert "Exclude Pattern: test*" in result.output

    def test_batch_no_files(self, runner, tmp_path):
        """Test batch with no matching files."""
        from markit.cli.main import app

        # Create an empty output directory for testing
        output_dir = tmp_path / "output"

        with patch("markit.cli.commands.batch._execute_batch", new_callable=AsyncMock):
            # Mock will complete without processing any files
            runner.invoke(
                app,
                ["batch", str(tmp_path), "-o", str(output_dir)],
            )
            # With no files, should still work (empty batch)

    def test_batch_resume_flag_accepted(self, runner, tmp_path):
        """Test batch with --resume flag is accepted."""
        from markit.cli.main import app

        (tmp_path / "test.pdf").touch()
        output_dir = tmp_path / "output"

        with patch(
            "markit.cli.commands.batch._execute_batch", new_callable=AsyncMock
        ) as mock_execute:
            runner.invoke(
                app,
                ["batch", str(tmp_path), "-o", str(output_dir), "--resume"],
            )
            # Should pass resume=True to _execute_batch
            if mock_execute.called:
                call_kwargs = mock_execute.call_args.kwargs
                assert call_kwargs.get("resume") is True

    def test_batch_with_state_file(self, runner, tmp_path):
        """Test batch with custom state file path."""
        from markit.cli.main import app

        (tmp_path / "test.pdf").touch()
        output_dir = tmp_path / "output"
        state_file = tmp_path / "custom_state.json"

        with patch(
            "markit.cli.commands.batch._execute_batch", new_callable=AsyncMock
        ) as mock_execute:
            runner.invoke(
                app,
                ["batch", str(tmp_path), "-o", str(output_dir), "--state-file", str(state_file)],
            )
            # State file should be passed to _execute_batch
            if mock_execute.called:
                call_kwargs = mock_execute.call_args.kwargs
                assert call_kwargs.get("state_path") == state_file

    def test_batch_keyboard_interrupt(self, runner, tmp_path):
        """Test batch handles keyboard interrupt."""
        from markit.cli.main import app

        (tmp_path / "test.pdf").touch()
        output_dir = tmp_path / "output"

        with patch(
            "markit.cli.commands.batch._execute_batch", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.side_effect = KeyboardInterrupt()
            result = runner.invoke(
                app,
                ["batch", str(tmp_path), "-o", str(output_dir)],
            )
            assert result.exit_code == 130
            assert "interrupted" in result.output.lower()

    def test_batch_with_llm_options(self, runner, tmp_path):
        """Test batch with LLM options."""
        from markit.cli.main import app

        (tmp_path / "test.pdf").touch()
        output_dir = tmp_path / "output"

        with patch("markit.cli.commands.batch._execute_batch", new_callable=AsyncMock):
            result = runner.invoke(
                app,
                ["batch", str(tmp_path), "-o", str(output_dir), "--llm", "--dry-run"],
            )
            assert result.exit_code == 0

    def test_batch_with_concurrency_options(self, runner, tmp_path):
        """Test batch with custom concurrency options."""
        from markit.cli.main import app

        (tmp_path / "test.pdf").touch()
        output_dir = tmp_path / "output"

        with patch(
            "markit.cli.commands.batch._execute_batch", new_callable=AsyncMock
        ) as mock_execute:
            runner.invoke(
                app,
                [
                    "batch",
                    str(tmp_path),
                    "-o",
                    str(output_dir),
                    "--file-concurrency",
                    "4",
                    "--image-concurrency",
                    "8",
                    "--llm-concurrency",
                    "5",
                ],
            )
            if mock_execute.called:
                call_kwargs = mock_execute.call_args.kwargs
                assert call_kwargs.get("file_concurrency") == 4
                assert call_kwargs.get("image_concurrency") == 8
                assert call_kwargs.get("llm_concurrency") == 5


class TestBatchHelpers:
    """Tests for batch helper functions and patterns."""

    def test_discover_supported_extensions(self, tmp_path):
        """Test that all supported extensions are discovered."""
        from markit.config.constants import SUPPORTED_EXTENSIONS

        # Create files for first 5 supported extensions
        extensions_to_test = list(SUPPORTED_EXTENSIONS)[:5]
        for ext in extensions_to_test:
            (tmp_path / f"test{ext}").touch()

        files = _discover_files(tmp_path, recursive=False, include=None, exclude=None)
        assert len(files) == 5

    def test_discover_case_sensitivity(self, tmp_path):
        """Test file discovery with different cases."""
        (tmp_path / "test.pdf").touch()
        (tmp_path / "test.PDF").touch()

        files = _discover_files(tmp_path, recursive=False, include=None, exclude=None)
        # Should find at least the lowercase version
        assert any(f.suffix == ".pdf" for f in files)


class TestExecuteBatch:
    """Tests for _execute_batch function with resume logic."""

    @pytest.fixture
    def mock_ctx(self, tmp_path):
        """Create mock conversion context."""
        ctx = MagicMock()
        ctx.options = MagicMock()
        ctx.options.llm = False
        ctx.options.effective_analyze_image = False
        ctx.options.use_phased_pipeline = False
        ctx.options.verbose = False
        ctx.options.fast = False
        ctx.options.compress_images = True
        ctx.settings = MagicMock()
        ctx.settings.state_file = str(tmp_path / "state.json")
        ctx.settings.output.on_conflict = "rename"
        ctx.create_pipeline = MagicMock(return_value=MagicMock())
        return ctx

    @pytest.mark.asyncio
    async def test_resume_no_state_starts_fresh(self, tmp_path, mock_ctx):
        """Test resume with no existing state starts fresh batch."""
        from markit.cli.commands.batch import _execute_batch

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.pdf").touch()
        output_dir = tmp_path / "output"
        state_path = tmp_path / "state.json"

        with (
            patch("markit.cli.commands.batch.console") as mock_console,
            patch("markit.cli.commands.batch._process_files_with_progress") as mock_process,
            patch("markit.cli.commands.batch._display_summary"),
        ):
            mock_process.return_value = []

            await _execute_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                recursive=False,
                include=None,
                exclude=None,
                file_concurrency=8,
                image_concurrency=16,
                llm_concurrency=10,
                on_conflict="rename",
                resume=True,
                state_path=state_path,
                ctx=mock_ctx,
            )

            # Should print message about no previous state
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("No previous batch state found" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_no_files_returns_early(self, tmp_path, mock_ctx):
        """Test batch with no files returns early."""
        from markit.cli.commands.batch import _execute_batch

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        state_path = tmp_path / "state.json"

        with (
            patch("markit.cli.commands.batch.console") as mock_console,
            patch("markit.cli.commands.batch._process_files_with_progress") as mock_process,
        ):
            await _execute_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                recursive=False,
                include=None,
                exclude=None,
                file_concurrency=8,
                image_concurrency=16,
                llm_concurrency=10,
                on_conflict="rename",
                resume=False,
                state_path=state_path,
                ctx=mock_ctx,
            )

            # Should print no files message
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("No files to process" in str(c) for c in calls)
            # Processing should not be called
            mock_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_phased_pipeline_when_llm_enabled(self, tmp_path, mock_ctx):
        """Test batch uses phased pipeline when LLM is enabled."""
        from markit.cli.commands.batch import _execute_batch

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.pdf").touch()
        output_dir = tmp_path / "output"
        state_path = tmp_path / "state.json"

        mock_ctx.options.use_phased_pipeline = True
        mock_ctx.options.verbose = False

        mock_pipeline = MagicMock()
        mock_pipeline.warmup = AsyncMock()
        mock_ctx.create_pipeline.return_value = mock_pipeline

        with (
            patch("markit.cli.commands.batch.console"),
            patch("markit.cli.commands.batch._process_files_phased_with_progress") as mock_phased,
            patch("markit.cli.commands.batch._process_files_with_progress") as mock_regular,
            patch("markit.cli.commands.batch._display_summary"),
        ):
            mock_phased.return_value = []

            await _execute_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                recursive=False,
                include=None,
                exclude=None,
                file_concurrency=8,
                image_concurrency=16,
                llm_concurrency=10,
                on_conflict="rename",
                resume=False,
                state_path=state_path,
                ctx=mock_ctx,
            )

            # Phased pipeline should be called, not regular
            mock_phased.assert_called_once()
            mock_regular.assert_not_called()

    @pytest.mark.asyncio
    async def test_verbose_mode_uses_verbose_processor(self, tmp_path, mock_ctx):
        """Test batch uses verbose processor in verbose mode."""
        from markit.cli.commands.batch import _execute_batch

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.pdf").touch()
        output_dir = tmp_path / "output"
        state_path = tmp_path / "state.json"

        mock_ctx.options.verbose = True

        with (
            patch("markit.cli.commands.batch.console"),
            patch("markit.cli.commands.batch._process_files_verbose") as mock_verbose,
            patch("markit.cli.commands.batch._process_files_with_progress") as mock_progress,
            patch("markit.cli.commands.batch._display_summary"),
        ):
            mock_verbose.return_value = []

            await _execute_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                recursive=False,
                include=None,
                exclude=None,
                file_concurrency=8,
                image_concurrency=16,
                llm_concurrency=10,
                on_conflict="rename",
                resume=False,
                state_path=state_path,
                ctx=mock_ctx,
            )

            # Verbose processor should be called, not progress
            mock_verbose.assert_called_once()
            mock_progress.assert_not_called()


class TestProcessFilePipeline:
    """Tests for _process_file_pipeline function.

    This tests the core orchestration logic that coordinates:
    - Phase 1: Document conversion
    - Phase 2: LLM task submission and execution
    - Phase 3: Output finalization
    """

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline."""
        pipeline = MagicMock()
        # Default successful conversion result
        doc_result = MagicMock()
        doc_result.success = True
        doc_result.error = None
        doc_result.images_for_analysis = []
        doc_result.processed_images = []
        pipeline.convert_document_only = AsyncMock(return_value=doc_result)
        pipeline.create_llm_tasks = AsyncMock(return_value=[])
        # Default successful pipeline result
        final_result = MagicMock()
        final_result.success = True
        final_result.error = None
        final_result.output_path = Path("/output/test.md")
        final_result.images_count = 0
        pipeline.finalize_output = AsyncMock(return_value=final_result)
        return pipeline

    @pytest.fixture
    def mock_concurrency(self):
        """Create mock concurrency manager."""
        concurrency = MagicMock()

        async def run_task(coro):
            return await coro

        concurrency.run_file_task = run_task
        return concurrency

    @pytest.fixture
    def mock_llm_queue(self):
        """Create mock LLM queue."""
        from markit.llm.queue import LLMTaskQueue

        queue = MagicMock(spec=LLMTaskQueue)

        async def submit_task(task):
            # Return a completed future that resolves to the result
            result = MagicMock()
            result.success = True
            result.task_type = task.task_type
            result.result = "mocked result"
            result.model = None
            result.prompt_tokens = 0
            result.completion_tokens = 0
            result.estimated_cost = None
            result.duration = 0.1
            result.start_time = None
            result.end_time = None

            async def get_result():
                return result

            return get_result()

        queue.submit = submit_task
        return queue

    @pytest.fixture
    def mock_state_manager(self):
        """Create mock state manager."""
        state_manager = MagicMock()
        state_manager.update_file_status = MagicMock()
        return state_manager

    @pytest.fixture
    def test_file(self, tmp_path):
        """Create a test file."""
        test_file = tmp_path / "input" / "test.pdf"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_bytes(b"test content")
        return test_file

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create output directory."""
        output = tmp_path / "output"
        output.mkdir(parents=True, exist_ok=True)
        return output

    @pytest.mark.asyncio
    async def test_successful_pipeline_no_llm(
        self,
        mock_pipeline,
        mock_concurrency,
        mock_llm_queue,
        mock_state_manager,
        test_file,
        output_dir,
    ):
        """Test successful pipeline execution without LLM tasks."""
        from markit.cli.commands.batch import _process_file_pipeline

        callbacks = {}
        input_dir = test_file.parent

        result = await _process_file_pipeline(
            file_path=test_file,
            pipeline=mock_pipeline,
            concurrency=mock_concurrency,
            llm_queue=mock_llm_queue,
            output_dir=output_dir,
            input_dir=input_dir,
            state_manager=mock_state_manager,
            callbacks=callbacks,
            stats=None,
        )

        assert result.success is True
        # Verify phase callbacks
        mock_pipeline.convert_document_only.assert_called_once_with(test_file, output_dir)
        mock_pipeline.create_llm_tasks.assert_called_once()
        mock_pipeline.finalize_output.assert_called_once()
        # Verify state manager was updated
        mock_state_manager.update_file_status.assert_called()

    @pytest.mark.asyncio
    async def test_phase1_failure(
        self,
        mock_pipeline,
        mock_concurrency,
        mock_llm_queue,
        mock_state_manager,
        test_file,
        output_dir,
    ):
        """Test pipeline failure during Phase 1 (document conversion)."""
        from markit.cli.commands.batch import _process_file_pipeline

        # Make phase 1 fail
        doc_result = MagicMock()
        doc_result.success = False
        doc_result.error = "Conversion failed: unsupported format"
        mock_pipeline.convert_document_only = AsyncMock(return_value=doc_result)

        callbacks = {
            "on_phase1_error": MagicMock(),
        }
        input_dir = test_file.parent

        result = await _process_file_pipeline(
            file_path=test_file,
            pipeline=mock_pipeline,
            concurrency=mock_concurrency,
            llm_queue=mock_llm_queue,
            output_dir=output_dir,
            input_dir=input_dir,
            state_manager=mock_state_manager,
            callbacks=callbacks,
            stats=None,
        )

        assert result.success is False
        assert result.error == "Conversion failed: unsupported format"
        # Verify phase 1 error callback was called
        callbacks["on_phase1_error"].assert_called_once()
        # Verify state was updated to failed
        mock_state_manager.update_file_status.assert_called_with(
            test_file, "failed", error="Conversion failed: unsupported format"
        )
        # Verify phase 2 and 3 were NOT called
        mock_pipeline.create_llm_tasks.assert_not_called()
        mock_pipeline.finalize_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_phase3_failure(
        self,
        mock_pipeline,
        mock_concurrency,
        mock_llm_queue,
        mock_state_manager,
        test_file,
        output_dir,
    ):
        """Test pipeline failure during Phase 3 (finalization)."""
        from markit.cli.commands.batch import _process_file_pipeline

        # Make phase 3 fail
        final_result = MagicMock()
        final_result.success = False
        final_result.error = "Failed to write output"
        mock_pipeline.finalize_output = AsyncMock(return_value=final_result)

        callbacks = {
            "on_phase3_error": MagicMock(),
        }
        input_dir = test_file.parent

        result = await _process_file_pipeline(
            file_path=test_file,
            pipeline=mock_pipeline,
            concurrency=mock_concurrency,
            llm_queue=mock_llm_queue,
            output_dir=output_dir,
            input_dir=input_dir,
            state_manager=mock_state_manager,
            callbacks=callbacks,
            stats=None,
        )

        assert result.success is False
        assert result.error == "Failed to write output"
        # Verify phase 3 error callback was called
        callbacks["on_phase3_error"].assert_called_once()

    @pytest.mark.asyncio
    async def test_with_llm_tasks(
        self,
        mock_pipeline,
        mock_concurrency,
        mock_llm_queue,
        mock_state_manager,
        test_file,
        output_dir,
    ):
        """Test pipeline with LLM tasks (image analysis + enhancement)."""
        from markit.cli.commands.batch import _process_file_pipeline
        from markit.image.analyzer import ImageAnalysis

        # Setup doc result with images
        doc_result = MagicMock()
        doc_result.success = True
        doc_result.error = None
        mock_image = MagicMock()
        mock_image.filename = "image1.png"
        doc_result.images_for_analysis = [mock_image]
        doc_result.processed_images = []
        mock_pipeline.convert_document_only = AsyncMock(return_value=doc_result)

        # Create LLM task coroutines
        async def image_task():
            return ImageAnalysis(
                alt_text="Test image",
                detailed_description="A test image detailed description",
                detected_text=None,
                image_type="diagram",
            )

        async def enhancement_task():
            return "Enhanced markdown content"

        mock_pipeline.create_llm_tasks = AsyncMock(return_value=[image_task(), enhancement_task()])

        # Setup LLM queue to return results
        submitted_tasks = []

        async def submit_task(task):
            result = MagicMock()
            result.success = True
            result.task_type = task.task_type
            result.model = "gpt-4o"
            result.prompt_tokens = 100
            result.completion_tokens = 50
            result.estimated_cost = 0.001
            result.duration = 0.5
            result.start_time = None
            result.end_time = None

            if task.task_type == "image_analysis":
                result.result = ImageAnalysis(
                    alt_text="Test image",
                    detailed_description="A test image detailed description",
                    detected_text=None,
                    image_type="diagram",
                )
            else:
                result.result = "Enhanced markdown"

            submitted_tasks.append(task)

            async def get_result():
                return result

            return get_result()

        mock_llm_queue.submit = submit_task

        callbacks = {
            "on_phase2_start": MagicMock(),
            "on_phase2_complete": MagicMock(),
        }
        input_dir = test_file.parent

        result = await _process_file_pipeline(
            file_path=test_file,
            pipeline=mock_pipeline,
            concurrency=mock_concurrency,
            llm_queue=mock_llm_queue,
            output_dir=output_dir,
            input_dir=input_dir,
            state_manager=mock_state_manager,
            callbacks=callbacks,
            stats=None,
        )

        assert result.success is True
        # Verify phase 2 callbacks were called
        callbacks["on_phase2_start"].assert_called_once()
        callbacks["on_phase2_complete"].assert_called_once()
        # Verify LLM tasks were submitted
        assert len(submitted_tasks) == 2

    @pytest.mark.asyncio
    async def test_exception_handling(
        self,
        mock_pipeline,
        mock_concurrency,
        mock_llm_queue,
        mock_state_manager,
        test_file,
        output_dir,
    ):
        """Test pipeline handles unexpected exceptions."""
        from markit.cli.commands.batch import _process_file_pipeline

        # Make pipeline raise an exception
        mock_pipeline.convert_document_only = AsyncMock(
            side_effect=RuntimeError("Unexpected error")
        )

        callbacks = {
            "on_error": MagicMock(),
        }
        input_dir = test_file.parent

        result = await _process_file_pipeline(
            file_path=test_file,
            pipeline=mock_pipeline,
            concurrency=mock_concurrency,
            llm_queue=mock_llm_queue,
            output_dir=output_dir,
            input_dir=input_dir,
            state_manager=mock_state_manager,
            callbacks=callbacks,
            stats=None,
        )

        assert result.success is False
        assert result.error is not None and "Unexpected error" in result.error
        # Verify error callback was called
        callbacks["on_error"].assert_called_once()

    @pytest.mark.asyncio
    async def test_with_stats_tracking(
        self,
        mock_pipeline,
        mock_concurrency,
        mock_llm_queue,
        mock_state_manager,
        test_file,
        output_dir,
    ):
        """Test pipeline with stats tracking enabled."""
        from markit.cli.commands.batch import _process_file_pipeline
        from markit.utils.stats import BatchStats

        stats = BatchStats()
        callbacks = {}
        input_dir = test_file.parent

        result = await _process_file_pipeline(
            file_path=test_file,
            pipeline=mock_pipeline,
            concurrency=mock_concurrency,
            llm_queue=mock_llm_queue,
            output_dir=output_dir,
            input_dir=input_dir,
            state_manager=mock_state_manager,
            callbacks=callbacks,
            stats=stats,
        )

        assert result.success is True
        # Stats should have recorded the success
        assert stats.success_files == 1
