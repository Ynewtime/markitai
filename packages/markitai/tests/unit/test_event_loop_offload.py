"""Blocking work that must not run on the event loop (markitai serve).

A server answers every request on one loop: an auth check that runs a CLI,
or a converter module importing its backend, would stall them all.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_provider_auth_checks_run_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from markitai.providers.auth import AuthManager, AuthStatus

    loop_thread = threading.get_ident()
    seen: list[int] = []

    def fake_check(self: AuthManager) -> AuthStatus:
        seen.append(threading.get_ident())
        return AuthStatus(
            provider="claude-agent",
            authenticated=True,
            user=None,
            expires_at=None,
            error=None,
        )

    monkeypatch.setattr(AuthManager, "_check_claude", fake_check)
    status = await AuthManager().check_auth("claude-agent", force_refresh=True)

    assert status.authenticated
    assert seen and seen[0] != loop_thread


@pytest.mark.asyncio
async def test_a_converter_backend_is_imported_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from markitai.converter import base

    loop_thread = threading.get_ident()
    seen: list[int] = []

    def fake_load(fmt: base.FileFormat) -> None:
        seen.append(threading.get_ident())

    # A format whose converter is not registered yet
    fmt = base.FileFormat.PDF
    monkeypatch.delitem(base._converter_registry, fmt, raising=False)
    monkeypatch.setattr(base, "load_converter_class", fake_load)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    await base.preload_converter_class(pdf)

    assert seen and seen[0] != loop_thread


@pytest.mark.asyncio
async def test_a_loaded_converter_is_not_loaded_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from markitai.converter import base

    calls: list[object] = []
    monkeypatch.setattr(base, "load_converter_class", calls.append)
    monkeypatch.setitem(base._converter_registry, base.FileFormat.CSV, object)
    csv = tmp_path / "table.csv"
    csv.write_text("a,b\n1,2\n")

    await base.preload_converter_class(csv)

    assert calls == []


def test_the_serve_shutdown_does_not_import_litellm_to_clean_it_up() -> None:
    """Only a LiteLLM that was loaded has clients to close."""
    source = (
        Path(sys.modules["markitai"].__file__ or "").parent / "serve" / "app.py"
    ).read_text(encoding="utf-8")
    cleanup = source.index("close_litellm_async_clients")
    guard = source.rfind('if "litellm" in sys.modules', 0, cleanup)
    assert guard != -1 and cleanup - guard < 400
