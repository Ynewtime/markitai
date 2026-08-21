"""Guard: bracketed text in a message must survive to the terminal.

``ui.error("… markitai[ocr] …")`` interpolated the caller's text into a
markup-enabled ``Console.print``, so rich parsed ``[ocr]`` as a style tag and
dropped it. The remedy printed for a missing OCR backend therefore read
``uv tool install "markitai" --force`` — a command that reinstalls what the
user already has. Extras are the most common bracketed text markitai prints,
which is exactly why this must be escaped at the shared helper rather than at
each call site.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from markitai.cli import ui

EXTRAS_MESSAGE = 'Install with: uv tool install "markitai[ocr]" --force'


@pytest.fixture
def capturing_console() -> Console:
    # width kept wide so wrapping never splits the token under assertion
    return Console(file=None, record=True, width=200, force_terminal=False)


@pytest.mark.parametrize("helper", ["success", "error", "warning", "info", "title"])
def test_helper_keeps_bracketed_text(capturing_console: Console, helper: str) -> None:
    getattr(ui, helper)(EXTRAS_MESSAGE, console=capturing_console)
    assert "markitai[ocr]" in capturing_console.export_text()


@pytest.mark.parametrize("helper", ["error", "warning"])
def test_detail_keeps_bracketed_text(capturing_console: Console, helper: str) -> None:
    getattr(ui, helper)(
        "Conversion failed", detail=EXTRAS_MESSAGE, console=capturing_console
    )
    assert "markitai[ocr]" in capturing_console.export_text()


def test_a_stray_bracket_does_not_crash_rendering(capturing_console: Console) -> None:
    """Unbalanced brackets are data too; rich must not raise on them."""
    ui.error("unclosed [tag and a lone ] bracket", console=capturing_console)
    assert "unclosed [tag" in capturing_console.export_text()
