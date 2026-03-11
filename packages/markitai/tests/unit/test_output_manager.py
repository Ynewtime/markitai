from __future__ import annotations

import io

from rich.console import Console

from markitai.cli.output_manager import OutputManager


class TestOutputManagerLineCounting:
    """OutputManager accurately counts rendered terminal lines."""

    def test_print_single_line(self) -> None:
        """Single short print counts as 1 line."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)
        om.print("hello")
        assert om.line_count == 1

    def test_print_multiple_calls(self) -> None:
        """Multiple print calls accumulate line count."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)
        om.print("line 1")
        om.print("line 2")
        om.print("line 3")
        assert om.line_count == 3

    def test_print_multiline_text(self) -> None:
        """Text with embedded newlines counts each line."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)
        om.print("line 1\nline 2\nline 3")
        assert om.line_count == 3

    def test_print_wrapped_line(self) -> None:
        """Long text wrapping on narrow console counts wrapped lines."""
        buf = io.StringIO()
        # Width=20 forces 40-char text to wrap to ~2 lines
        console = Console(file=buf, width=20, force_terminal=True)
        om = OutputManager(console=console)
        om.print("a" * 40)  # Should wrap to 2 lines at width 20
        assert om.line_count == 2

    def test_print_rich_markup(self) -> None:
        """Rich markup doesn't inflate line count."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)
        om.print("[bold cyan]styled text[/bold cyan]")
        assert om.line_count == 1

    def test_print_empty_string(self) -> None:
        """Empty string print counts as 1 line (Rich adds newline)."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)
        om.print("")
        assert om.line_count == 1

    def test_print_with_style(self) -> None:
        """Style parameter doesn't affect line count."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)
        om.print("dimmed text", style="dim")
        assert om.line_count == 1


class TestOutputManagerErasure:
    """OutputManager erases tracked lines with ANSI sequences."""

    def test_erase_sends_correct_ansi(self) -> None:
        """erase_all sends N cursor-up + clear-line sequences."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)
        om.print("line 1")
        om.print("line 2")
        om.print("line 3")
        buf.truncate(0)
        buf.seek(0)
        om.erase_all()
        output = buf.getvalue()
        assert output.count("\033[A") == 3
        assert output.count("\033[2K") == 3

    def test_erase_resets_count(self) -> None:
        """line_count is 0 after erase_all."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)
        om.print("line 1")
        om.print("line 2")
        om.erase_all()
        assert om.line_count == 0

    def test_erase_noop_when_disabled(self) -> None:
        """No ANSI output when enabled=False."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console, enabled=False)
        om.print("line 1")
        buf.truncate(0)
        buf.seek(0)
        om.erase_all()
        assert buf.getvalue() == ""

    def test_erase_noop_empty(self) -> None:
        """No ANSI output when no lines tracked."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)
        buf.truncate(0)
        buf.seek(0)
        om.erase_all()
        assert buf.getvalue() == ""


class TestOutputManagerExternalTracking:
    """OutputManager tracks externally written lines."""

    def test_track_external_lines(self) -> None:
        """track_external_lines adds to count."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)
        om.track_external_lines(3)
        assert om.line_count == 3

    def test_track_mixed(self) -> None:
        """External lines combine with print lines."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)
        om.print("line 1")
        om.track_external_lines(2)
        om.print("line 2")
        assert om.line_count == 4


class TestOutputManagerSpinner:
    """Spinner lifecycle works correctly."""

    def test_start_stop_spinner(self) -> None:
        """Spinner starts and stops without error."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)
        om.start_spinner("Processing...")
        om.stop_spinner()

    def test_stop_without_start(self) -> None:
        """Stopping without starting is a no-op."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)
        om.stop_spinner()  # Should not raise

    def test_context_manager_stops_spinner(self) -> None:
        """Exiting context manager stops active spinner."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        with OutputManager(console=console) as om:
            om.start_spinner("Processing...")
        # Should not raise or leave spinner running


class TestOutputManagerDisabled:
    """Disabled OutputManager suppresses all output."""

    def test_disabled_suppresses_print(self) -> None:
        """enabled=False suppresses print output."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console, enabled=False)
        om.print("should not appear")
        assert om.line_count == 0
        assert buf.getvalue() == ""

    def test_disabled_suppresses_spinner(self) -> None:
        """enabled=False suppresses spinner."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console, enabled=False)
        om.start_spinner("Processing...")
        om.stop_spinner()

    def test_disabled_suppresses_print_persistent(self) -> None:
        """enabled=False suppresses print_persistent."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console, enabled=False)
        om.print_persistent("should not appear")
        assert buf.getvalue() == ""


class TestOutputManagerPersistentPrint:
    """print_persistent writes to stderr without tracking for erasure."""

    def test_persistent_does_not_increment_line_count(self) -> None:
        """print_persistent should NOT increase line_count."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)
        om.print_persistent("auth result")
        assert om.line_count == 0

    def test_persistent_survives_erase_all(self) -> None:
        """Content from print_persistent is not erased by erase_all."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)

        # Phase 1: tracked messages
        om.print("ephemeral warning")
        om.print("ephemeral prompt")
        assert om.line_count == 2

        # Erase all tracked
        om.erase_all()
        assert om.line_count == 0

        # Phase 2: persistent result (should survive)
        om.print_persistent("  ✓ authenticated")

        # Phase 3: more tracked messages
        om.print("progress spinner")
        assert om.line_count == 1

        # Erase only the progress
        buf.truncate(0)
        buf.seek(0)
        om.erase_all()
        output = buf.getvalue()
        # Only 1 line should be erased (the progress), not the persistent one
        assert output.count("\033[A") == 1
        assert output.count("\033[2K") == 1

    def test_persistent_writes_to_output(self) -> None:
        """print_persistent actually writes text to the console."""
        buf = io.StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        om = OutputManager(console=console)
        om.print_persistent("hello world")
        assert "hello world" in buf.getvalue()
