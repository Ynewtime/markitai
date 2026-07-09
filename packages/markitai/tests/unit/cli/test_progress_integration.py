"""Integration tests: StageList wiring inside the URL/file processors.

The URL tests live here from Task 2; Task 3 adds the file-processor tests.
Replaces the processor-level classes of test_conversion_status.py.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from loguru import logger
from rich.console import Console

from markitai.cli.processors.url import process_url
from markitai.config import MarkitaiConfig

HIDE_CURSOR = "\x1b[?25l"


def make_tty_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, width=80)


def make_fetch_result(content: str = "# Test Page\n\nSome content.") -> MagicMock:
    result = MagicMock()
    result.content = content
    result.cache_hit = False
    result.strategy_used = "static"
    result.screenshot_path = None
    result.title = "Test Page"
    result.static_content = None
    result.browser_content = None
    result.metadata = {}
    return result


class TestUrlProcessorGating:
    """process_url must enable StageList in BOTH file and stdout modes."""

    async def _run(self, output_dir: Path | None, **kwargs) -> MagicMock:
        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False
        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                return_value=make_fetch_result(),
            ),
            patch("markitai.cli.ui.StageList") as stage_cls,
        ):
            await process_url(
                url="https://example.com/test",
                output_dir=output_dir,
                cfg=cfg,
                dry_run=False,
                verbose=kwargs.get("verbose", False),
                quiet=kwargs.get("quiet", False),
            )
        return stage_cls

    async def test_file_mode_enables_persistent_stagelist(self, tmp_path: Path) -> None:
        stage_cls = await self._run(tmp_path)
        kwargs = stage_cls.call_args.kwargs
        assert kwargs["enabled"] is True
        assert kwargs["transient"] is False

    async def test_stdout_mode_enables_transient_stagelist(self, capsys) -> None:
        stage_cls = await self._run(None)
        capsys.readouterr()  # discard markdown on stdout
        kwargs = stage_cls.call_args.kwargs
        assert kwargs["enabled"] is True  # THE bug fix: was False before
        assert kwargs["transient"] is True

    async def test_verbose_disables_stagelist(self, tmp_path: Path) -> None:
        stage_cls = await self._run(tmp_path, verbose=True)
        assert stage_cls.call_args.kwargs["enabled"] is False

    async def test_quiet_disables_stagelist(self, tmp_path: Path) -> None:
        stage_cls = await self._run(tmp_path, quiet=True)
        assert stage_cls.call_args.kwargs["enabled"] is False


class TestUrlStdoutStaysPure:
    """Stage frames go to stderr only; stdout carries only markdown."""

    async def test_stdout_mode_output_contains_no_frames(
        self, capsys, monkeypatch
    ) -> None:
        fake_stderr = make_tty_console()
        monkeypatch.setattr("markitai.cli.ui.get_stderr_console", lambda: fake_stderr)

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        async def fetch_with_log(*args, **kwargs):
            logger.debug("Fetching URL with static strategy: https://example.com")
            return make_fetch_result()

        with patch("markitai.fetch.fetch_url", side_effect=fetch_with_log):
            await process_url(
                url="https://example.com/test",
                output_dir=None,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        captured = capsys.readouterr()
        # Live rendered on (fake) stderr
        assert HIDE_CURSOR in fake_stderr.file.getvalue()  # type: ignore[union-attr]
        # stdout carries clean markdown only
        assert HIDE_CURSOR not in captured.out
        assert "Fetching (static)" not in captured.out
        assert "Test Page" in captured.out

    async def test_file_mode_persists_stage_lines_on_stderr(
        self, tmp_path: Path, capsys, monkeypatch
    ) -> None:
        fake_stderr = make_tty_console()
        monkeypatch.setattr("markitai.cli.ui.get_stderr_console", lambda: fake_stderr)

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        with patch(
            "markitai.fetch.fetch_url",
            new_callable=AsyncMock,
            return_value=make_fetch_result(),
        ):
            await process_url(
                url="https://example.com/test",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        stderr_text = fake_stderr.file.getvalue()  # type: ignore[union-attr]
        assert "Fetched via static" in stderr_text
        assert (tmp_path / "test.md").exists()
        captured = capsys.readouterr()
        assert HIDE_CURSOR not in captured.out


class TestUrlErrorPath:
    """Fetch errors must finalize the active stage as a failure."""

    async def test_fetch_error_marks_stage_failed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import pytest

        from markitai.fetch import FetchError

        fake_stderr = make_tty_console()
        monkeypatch.setattr("markitai.cli.ui.get_stderr_console", lambda: fake_stderr)

        cfg = MarkitaiConfig()
        cfg.llm.enabled = False
        cfg.cache.enabled = False

        with (
            patch(
                "markitai.fetch.fetch_url",
                new_callable=AsyncMock,
                side_effect=FetchError("boom"),
            ),
            pytest.raises(SystemExit),
        ):
            await process_url(
                url="https://example.com/test",
                output_dir=tmp_path,
                cfg=cfg,
                dry_run=False,
                verbose=False,
            )

        stderr_text = fake_stderr.file.getvalue()  # type: ignore[union-attr]
        # The failed fetch stage line persisted (file mode persists on stop)
        assert "Fetching" in stderr_text
