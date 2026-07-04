"""Regression guard for scripts/setup.sh set -e safety.

Root cause of the bug this guards against: setup.sh runs under `set -eu`,
and its optional-component installers return non-zero to signal a
declined/failed OPTIONAL step (skip → `return 2`, fail → `return 1`).
When such a call is unguarded, `set -e` treats the non-zero return as
fatal and exits the whole script — so declining LibreOffice would abort
setup before the summary/next-steps/outro ever printed.

The fix guards every non-fatal orchestration call with `|| true`. This
test fails if any of them regresses back to a bare call.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SETUP_SH = Path(__file__).resolve().parents[4] / "scripts" / "setup.sh"

# Non-fatal orchestration calls: declining/failing these must NOT abort setup.
# (install_uv / install_markitai are intentionally fatal and use their own
# `|| { ...; exit 1; }` handler, so they are not in this list.)
_NON_FATAL_CALLS = (
    "install_optional_playwright",
    "install_optional_libreoffice",
    "install_optional_ffmpeg",
    "install_optional_claude_cli",
    "install_optional_copilot_cli",
    "detect_gemini_cli",
    "finalize_markitai_extras",
    "install_precommit",
)


@pytest.mark.skipif(not _SETUP_SH.exists(), reason="scripts/setup.sh not present")
def test_setup_sh_runs_under_set_e() -> None:
    """The guards only matter because the script is `set -e`."""
    assert re.search(
        r"^set -eu?\b", _SETUP_SH.read_text(encoding="utf-8"), re.MULTILINE
    ), (
        "setup.sh is expected to run under set -e; if that changed, revisit "
        "whether the || true guards are still needed."
    )


@pytest.mark.skipif(not _SETUP_SH.exists(), reason="scripts/setup.sh not present")
@pytest.mark.parametrize("func", _NON_FATAL_CALLS)
def test_non_fatal_calls_are_guarded(func: str) -> None:
    """Every orchestration call to a non-fatal installer must be guarded.

    A bare `    install_optional_libreoffice` line under set -e aborts the
    script when the function returns non-zero (declined/failed).
    """
    text = _SETUP_SH.read_text(encoding="utf-8")
    # A call site is the function name as a whole indented statement. The
    # definition line ends with `()`, so `\b(?!\()` excludes it.
    bare = re.compile(rf"^[ \t]+{re.escape(func)}[ \t]*$", re.MULTILINE)
    guarded = re.compile(rf"^[ \t]+{re.escape(func)}[ \t]*\|\|", re.MULTILINE)

    bare_calls = bare.findall(text)
    # There must be at least one call site (guarded), and zero bare ones.
    assert guarded.search(text), f"expected a guarded call to {func}()"
    assert not bare_calls, (
        f"unguarded call to {func}() found: under set -e this aborts setup "
        f"when the user declines or it fails. Append ` || true`."
    )
