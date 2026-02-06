"""Tests for UI components module."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from markitai.cli import ui


class TestSymbolConstants:
    """Tests for symbol constants."""

    def test_mark_success_defined(self) -> None:
        """Test MARK_SUCCESS is defined correctly."""
        assert ui.MARK_SUCCESS == "\u2713"

    def test_mark_error_defined(self) -> None:
        """Test MARK_ERROR is defined correctly."""
        assert ui.MARK_ERROR == "\u2717"

    def test_mark_warning_defined(self) -> None:
        """Test MARK_WARNING is defined correctly."""
        assert ui.MARK_WARNING == "!"

    def test_mark_info_defined(self) -> None:
        """Test MARK_INFO is defined correctly."""
        assert ui.MARK_INFO == "\u2022"

    def test_mark_title_defined(self) -> None:
        """Test MARK_TITLE is defined correctly."""
        assert ui.MARK_TITLE == "\u25c6"

    def test_mark_line_defined(self) -> None:
        """Test MARK_LINE is defined correctly."""
        assert ui.MARK_LINE == "\u2502"


class TestTitleFunction:
    """Tests for title function."""

    def test_title_outputs_text_with_symbol(self) -> None:
        """Test title outputs text with diamond symbol."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.title("Test Title", console=console)

        output = buffer.getvalue()
        assert ui.MARK_TITLE in output
        assert "Test Title" in output

    def test_title_adds_empty_line(self) -> None:
        """Test title adds an empty line after output."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.title("Test Title", console=console)

        output = buffer.getvalue()
        # Should end with newline from print + empty line
        assert output.endswith("\n\n")


class TestSuccessFunction:
    """Tests for success function."""

    def test_success_outputs_text_with_checkmark(self) -> None:
        """Test success outputs text with checkmark symbol."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.success("Operation completed", console=console)

        output = buffer.getvalue()
        assert ui.MARK_SUCCESS in output
        assert "Operation completed" in output

    def test_success_has_indentation(self) -> None:
        """Test success output has leading indentation."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.success("Test", console=console)

        output = buffer.getvalue()
        # Should start with 2 spaces
        assert output.startswith("  ")


class TestErrorFunction:
    """Tests for error function."""

    def test_error_outputs_text_with_cross(self) -> None:
        """Test error outputs text with cross symbol."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.error("Operation failed", console=console)

        output = buffer.getvalue()
        assert ui.MARK_ERROR in output
        assert "Operation failed" in output

    def test_error_with_detail(self) -> None:
        """Test error with detail shows additional line."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.error("Operation failed", detail="Connection timeout", console=console)

        output = buffer.getvalue()
        assert ui.MARK_ERROR in output
        assert "Operation failed" in output
        assert "Connection timeout" in output
        assert ui.MARK_LINE in output

    def test_error_without_detail(self) -> None:
        """Test error without detail does not show line symbol."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.error("Operation failed", console=console)

        output = buffer.getvalue()
        lines = output.strip().split("\n")
        assert len(lines) == 1


class TestWarningFunction:
    """Tests for warning function."""

    def test_warning_outputs_text_with_exclamation(self) -> None:
        """Test warning outputs text with exclamation symbol."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.warning("Caution required", console=console)

        output = buffer.getvalue()
        assert ui.MARK_WARNING in output
        assert "Caution required" in output

    def test_warning_with_detail(self) -> None:
        """Test warning with detail shows additional line."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.warning("Caution required", detail="File may be large", console=console)

        output = buffer.getvalue()
        assert ui.MARK_WARNING in output
        assert "Caution required" in output
        assert "File may be large" in output
        assert ui.MARK_LINE in output

    def test_warning_without_detail(self) -> None:
        """Test warning without detail does not show line symbol."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.warning("Caution required", console=console)

        output = buffer.getvalue()
        lines = output.strip().split("\n")
        assert len(lines) == 1


class TestInfoFunction:
    """Tests for info function."""

    def test_info_outputs_text_with_bullet(self) -> None:
        """Test info outputs text with bullet symbol."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.info("Processing file", console=console)

        output = buffer.getvalue()
        assert ui.MARK_INFO in output
        assert "Processing file" in output

    def test_info_has_indentation(self) -> None:
        """Test info output has leading indentation."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.info("Test", console=console)

        output = buffer.getvalue()
        # Should start with 2 spaces
        assert output.startswith("  ")


class TestStepFunction:
    """Tests for step function."""

    def test_step_outputs_text_with_line(self) -> None:
        """Test step outputs text with vertical line symbol."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.step("Downloading data", console=console)

        output = buffer.getvalue()
        assert ui.MARK_LINE in output
        assert "Downloading data" in output

    def test_step_has_indentation(self) -> None:
        """Test step output has leading indentation."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.step("Test", console=console)

        output = buffer.getvalue()
        # Should start with 2 spaces
        assert output.startswith("  ")


class TestSectionFunction:
    """Tests for section function."""

    def test_section_outputs_bold_text(self) -> None:
        """Test section outputs text (bold styling)."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.section("Configuration", console=console)

        output = buffer.getvalue()
        assert "Configuration" in output


class TestSummaryFunction:
    """Tests for summary function."""

    def test_summary_outputs_text_with_checkmark(self) -> None:
        """Test summary outputs text with checkmark symbol."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.summary("All tasks completed", console=console)

        output = buffer.getvalue()
        assert ui.MARK_SUCCESS in output
        assert "All tasks completed" in output

    def test_summary_has_leading_empty_line(self) -> None:
        """Test summary has an empty line before output."""
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)

        ui.summary("Done", console=console)

        output = buffer.getvalue()
        # Should start with empty line (newline character)
        assert output.startswith("\n")


class TestDefaultConsole:
    """Tests for default console behavior."""

    def test_functions_work_without_console_parameter(self) -> None:
        """Test all functions work when console parameter is not provided."""
        # These should not raise exceptions
        # We just verify they can be called; output goes to default console
        # Since we cannot easily capture stdout in this test, we just check no exception
        try:
            # We don't actually call these without console in tests
            # as it would pollute test output, but we can verify the signature
            import inspect

            for func in [
                ui.title,
                ui.success,
                ui.error,
                ui.warning,
                ui.info,
                ui.step,
                ui.section,
                ui.summary,
            ]:
                sig = inspect.signature(func)
                assert "console" in sig.parameters
                assert sig.parameters["console"].default is None
        except Exception as e:
            pytest.fail(f"Function signature check failed: {e}")
