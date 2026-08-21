"""Runtime guidance must not point at options the CLI no longer accepts.

Six deprecated aliases (``--playwright``, ``--static``, ...) were replaced by
``-s``/``-b`` and now fail as a usage error. Error messages and hints that
still spell the old names send the user straight into that error, so they are
worse than no hint at all — the parser's migration table is the single source
of truth for what may no longer appear in user-facing text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from markitai.cli.framework import _REMOVED_OPTIONS

SRC_ROOT = Path(__file__).parent.parent.parent / "src" / "markitai"

# The fetch/URL guidance surface: every module here can put a strategy or
# backend hint in front of a user (exception text, log line, console step)
# or document one in a comment.
SCANNED_FILES: tuple[str, ...] = (
    "fetch_support.py",
    "fetch_types.py",
    "config.py",
    "converter/cloudflare.py",
    "fetch_strategies/cloudflare.py",
    "fetch_strategies/defuddle.py",
    "fetch_strategies/jina.py",
    "fetch_strategies/playwright.py",
    "fetch_strategies/static.py",
    "cli/processors/batch.py",
    "cli/processors/file.py",
    "cli/processors/llm.py",
    "cli/processors/url.py",
    "cli/processors/validators.py",
    "llm/document.py",
    "serve/jobs.py",
    "workflow/helpers.py",
    "workflow/single.py",
)


class TestRemovedOptionsAreNotAdvertised:
    """No scanned module may name a removed option anywhere in its text."""

    def test_migration_table_is_not_empty(self) -> None:
        """Positive control: an empty table would make every check vacuous."""
        assert _REMOVED_OPTIONS
        assert "--playwright" in _REMOVED_OPTIONS

    @pytest.mark.parametrize("relative_path", SCANNED_FILES)
    def test_module_text_never_names_a_removed_option(self, relative_path: str) -> None:
        path = SRC_ROOT / relative_path
        text = path.read_text(encoding="utf-8")

        found = sorted(name for name in _REMOVED_OPTIONS if name in text)

        assert not found, (
            f"{relative_path} still names removed option(s) {found}; passing "
            "them now fails as a usage error. Use the replacement spelling "
            f"({', '.join(f'{k} -> {v}' for k, v in _REMOVED_OPTIONS.items())})."
        )


class TestRateLimitHintsUseTheCurrentSpelling:
    """The two rate-limit paths are the hints users actually hit."""

    def test_jina_rate_limit_error_points_at_strategy_flag(self) -> None:
        from markitai.fetch_types import JinaRateLimitError

        message = str(JinaRateLimitError())

        assert "-s playwright" in message
        assert "--playwright" not in message

    @pytest.mark.asyncio
    async def test_defuddle_rate_limit_error_points_at_strategy_flag(self) -> None:
        from markitai.fetch_strategies import defuddle as defuddle_module
        from markitai.fetch_types import FetchError

        class _RateLimited:
            status_code = 429
            text = "too many requests"

        class _Client:
            async def get(self, *args: Any, **kwargs: Any) -> _RateLimited:
                return _RateLimited()

        with (
            patch.object(defuddle_module, "_get_defuddle_client", lambda _t: _Client()),
            pytest.raises(FetchError) as excinfo,
        ):
            await defuddle_module.fetch_with_defuddle("https://example.com/page")

        message = str(excinfo.value)
        assert "-s playwright" in message
        assert "--playwright" not in message
