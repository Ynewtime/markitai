"""Tests for suspending the active StageList Live around interactive prompts.

rich's Live proxies sys.stdout/sys.stderr while started. An interactive
prompt written through the proxy loses text without a trailing newline
(FileProxy.flush prints with markup enabled, eating e.g. "[y/N]"), and the
user's Enter echo desyncs Live's cursor tracking so later refreshes stack
stale spinner frames. suspend_active_live() must stop the Live for the
duration of the prompt and restart it afterwards.
"""

from __future__ import annotations

import io

from rich.console import Console

from markitai.cli import ui


def _tty_console() -> Console:
    """A console that reports as a terminal so StageList starts a Live."""
    return Console(file=io.StringIO(), force_terminal=True, width=80)


class TestSuspendActiveLive:
    def test_suspends_and_resumes_active_live(self) -> None:
        stages = ui.StageList(enabled=True, transient=False, console=_tty_console())
        stages.start()
        try:
            stages.advance("render", "Rendering...")
            assert stages._live is not None and stages._live.is_started

            with ui.suspend_active_live():
                assert not stages._live.is_started, (
                    "Live must be stopped during the prompt"
                )

            assert stages._live.is_started, "Live must resume after the prompt"
        finally:
            stages.stop()

    def test_noop_without_active_stagelist(self) -> None:
        with ui.suspend_active_live():
            pass  # must not raise

    def test_noop_after_stagelist_stopped(self) -> None:
        stages = ui.StageList(enabled=True, transient=False, console=_tty_console())
        stages.start()
        stages.stop()
        with ui.suspend_active_live():
            pass  # must not raise

    def test_restores_real_stderr_during_suspend(self) -> None:
        """During suspend the rich FileProxy must be off sys.stderr, so a
        click prompt writes to the real stream and keeps its [y/N] suffix."""
        import sys

        from rich.file_proxy import FileProxy

        stages = ui.StageList(enabled=True, transient=False, console=_tty_console())
        stages.start()
        try:
            with ui.suspend_active_live():
                assert not isinstance(sys.stderr, FileProxy)
                assert not isinstance(sys.stdout, FileProxy)
        finally:
            stages.stop()
