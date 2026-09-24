"""The user-notice channel: actionable warnings the default console must show.

The single-file console only let ERROR through unless ``-v`` was given, and
the batch progress bar detaches the console log handler entirely, so the
lines a user can act on -- "these pages look scanned, re-run with --ocr",
"slides could not be rendered", "OCR found no text" -- were never seen.
Notices get through both; every other warning stays where it was, and
``--quiet`` still silences them.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from loguru import logger

from markitai.batch import BatchProcessor, BatchState, FileState, FileStatus
from markitai.config import BatchConfig
from markitai.notices import collect_user_notices, is_user_notice, user_notice


class TestUserNotice:
    def test_notice_is_a_tagged_warning(self) -> None:
        records: list[Any] = []
        sink_id = logger.add(lambda m: records.append(m.record), level="WARNING")
        try:
            user_notice("[PDF] {}: {} page(s) look scanned", "a.pdf", 2)
            logger.warning("plain warning")
        finally:
            logger.remove(sink_id)

        assert [r["message"] for r in records] == [
            "[PDF] a.pdf: 2 page(s) look scanned",
            "plain warning",
        ]
        assert is_user_notice(records[0])
        assert not is_user_notice(records[1])
        # Attributed to the caller, not to the helper
        assert records[0]["function"] == "test_notice_is_a_tagged_warning"

    def test_collector_dedupes_notices_from_many_threads(self) -> None:
        with collect_user_notices() as notices:
            threads = [
                threading.Thread(target=user_notice, args=(f"notice {i % 3}",))
                for i in range(12)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            logger.warning("not a notice")
        user_notice("after the block")

        assert sorted(notices) == ["notice 0", "notice 1", "notice 2"]


class TestCaptureTaskNotices:
    """Per-task capture for hosts with no console (serve, API, MCP)."""

    async def test_concurrent_tasks_never_see_each_other(self) -> None:
        import asyncio

        from markitai.notices import capture_task_notices
        from markitai.utils.executor import run_in_converter_thread

        both_started = asyncio.Barrier(2)

        async def convert(name: str) -> list[str]:
            with capture_task_notices() as notices:
                await both_started.wait()
                # Raised in the shared converter pool, as converters do
                await run_in_converter_thread(
                    user_notice, "[OCR] No text found in {}", name
                )
                await both_started.wait()
            return notices

        first, second = await asyncio.gather(convert("a.png"), convert("b.png"))
        assert first == ["[OCR] No text found in a.png"]
        assert second == ["[OCR] No text found in b.png"]

    def test_capture_is_scoped_and_nests(self) -> None:
        from markitai.notices import capture_task_notices

        user_notice("before")
        with capture_task_notices() as outer:
            user_notice("outer {}", 1)
            with capture_task_notices() as inner:
                user_notice("inner")
                user_notice("inner")
                logger.warning("plain warning")
            user_notice("outer {}", 2)
        user_notice("after")

        assert inner == ["inner"]
        assert outer == ["outer 1", "inner", "outer 2"]

    def test_the_global_collector_still_sees_captured_notices(self) -> None:
        from markitai.notices import capture_task_notices

        with collect_user_notices() as everything, capture_task_notices() as mine:
            user_notice("scanned pages in {}", "a.pdf")
        assert everything == mine == ["scanned pages in a.pdf"]


class TestQuietConsoleShowsNotices:
    """setup_logging(quiet=True): notices pass unless --quiet was explicit."""

    def test_default_quiet_console_shows_notices_only(self, capsys) -> None:
        from markitai.cli.logging_config import setup_logging

        handler_id, _ = setup_logging(
            verbose=False, quiet=True, log_dir=None, show_notices=True
        )
        try:
            user_notice("[PPTX] Cannot render slides of deck.pptx")
            logger.warning("internal detail")
            logger.error("real error")
            captured = capsys.readouterr()
        finally:
            logger.remove(handler_id)

        assert "Cannot render slides of deck.pptx" in captured.err
        assert "internal detail" not in captured.err
        assert "real error" in captured.err

    def test_explicit_quiet_hides_notices(self, capsys) -> None:
        from markitai.cli.logging_config import setup_logging

        handler_id, _ = setup_logging(
            verbose=False, quiet=True, log_dir=None, show_notices=False
        )
        try:
            user_notice("[OCR] No text found in blank.png")
            captured = capsys.readouterr()
        finally:
            logger.remove(handler_id)

        assert "No text found" not in captured.err


class TestBatchSummaryListsNotices:
    def test_notices_logged_under_the_progress_bar_reach_the_summary(
        self, tmp_path: Path, capsys
    ) -> None:
        processor = BatchProcessor(BatchConfig(), tmp_path)
        processor.state = BatchState(
            started_at="2026-01-15T10:00:00Z",
            input_dir=str(tmp_path),
            output_dir=str(tmp_path),
        )
        processor.state.files = {
            "/in/scan.pdf": FileState(path="/in/scan.pdf", status=FileStatus.COMPLETED)
        }

        processor.start_live_display(total_files=1)
        try:
            user_notice("[PDF] scan.pdf: 3 page(s) look scanned/garbled (pages 1-3)")
            logger.warning("routine warning")
        finally:
            processor.stop_live_display()
        user_notice("logged after the run")
        capsys.readouterr()

        processor.print_summary()
        err = capsys.readouterr().err

        assert "scan.pdf: 3 page(s) look scanned/garbled" in err
        assert "routine warning" not in err
        assert "logged after the run" not in err
