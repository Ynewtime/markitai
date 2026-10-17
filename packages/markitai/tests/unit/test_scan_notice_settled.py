"""The "pages look scanned, re-run with --ocr" notice and a model that read them."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from markitai.converter.base import ConvertResult
from markitai.notices import capture_task_notices
from markitai.workflow.core import _has_prose, settle_scan_notice

NOTICE = [
    "[PDF] {}: {} of {} pages look scanned/garbled ({}); {}",
    "a.pdf",
    2,
    3,
    "pages 1–2",
    "consider re-running with --ocr",
]


def _ctx(**metadata: object) -> SimpleNamespace:
    return SimpleNamespace(
        conversion_result=ConvertResult(markdown="", metadata=dict(metadata))
    )


@pytest.mark.parametrize(
    ("text", "prose"),
    [
        ("# Title\n\nBody", True),
        ("<!-- Page images for reference -->\n<!-- ![Page 1](a.jpg) -->", False),
        ("![Page 1](screenshots/a.jpg)\n\n   \n", False),
        ("", False),
        ("![x](a.png)\n\n日本語", True),
    ],
)
def test_prose_is_text_beyond_pictures_and_comments(text: str, prose: bool) -> None:
    assert _has_prose(text) is prose


def test_the_notice_is_raised_when_the_model_returned_no_text() -> None:
    ctx = _ctx(scan_notice=NOTICE)
    with capture_task_notices() as notices:
        settle_scan_notice(ctx)  # type: ignore[arg-type]
    assert len(notices) == 1
    assert (
        "look scanned/garbled (pages 1–2); consider re-running with --ocr" in notices[0]
    )
    assert "scan_notice" not in ctx.conversion_result.metadata


def test_the_notice_is_dropped_when_the_model_read_the_pages() -> None:
    ctx = _ctx(scan_notice=NOTICE, scan_read=True)
    with capture_task_notices() as notices:
        settle_scan_notice(ctx)  # type: ignore[arg-type]
    assert notices == []
    assert ctx.conversion_result.metadata == {}


def test_nothing_to_settle_is_quiet() -> None:
    with capture_task_notices() as notices:
        settle_scan_notice(_ctx())  # type: ignore[arg-type]
        settle_scan_notice(SimpleNamespace(conversion_result=None))  # type: ignore[arg-type]
    assert notices == []


def test_the_converter_hands_the_notice_over_only_when_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import markitai.converter.pdf as pdf_module
    from markitai.converter.pdf import PdfConverter

    monkeypatch.setattr(
        pdf_module, "_scan_advisories", lambda *_args, **_kwargs: ([1, 2], [])
    )
    converter = PdfConverter()
    with capture_task_notices() as notices:
        handed = converter._warn_scanned_or_garbled(
            Path("a.pdf"), total_pages=3, defer=True
        )
    assert notices == []
    assert handed is not None
    assert handed[1:3] == ("a.pdf", 2)
    with capture_task_notices() as notices:
        assert converter._warn_scanned_or_garbled(Path("a.pdf"), total_pages=3) is None
    assert len(notices) == 1
