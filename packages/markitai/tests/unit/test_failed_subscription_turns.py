"""A subscription runtime's failed turn is a failed LLM call, never an answer.

The Claude Code CLI reports a failed turn (expired login, usage limit, an
API error) as an assistant message holding the error text, followed by a
result with ``is_error`` set. markitai read only the text, so the error
became the model's answer: a standalone image got a ``.llm.md`` whose
description and alt text were "Claude AI usage limit reached", with no
warning, and documents failed as "Invalid JSON" instead of naming the cause.

Every entry point (single file, directory batch, URL, serve) must keep only
the base output and say why in the item's notices.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sdk = pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import types as sdk_types  # noqa: E402

from markitai.config import LiteLLMParams, MarkitaiConfig, ModelConfig  # noqa: E402

LIMIT = "Claude AI usage limit reached|1791720000"


def _failing_client(
    text: str = LIMIT,
    *,
    error: str | None = "rate_limit",
    status: int | None = 429,
    subtype: str = "success",
) -> type:
    """An SDK client whose every turn fails the way the CLI reports it."""

    class Client:
        calls = 0

        def __init__(self, options: object) -> None:
            pass

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def query(self, prompt: object) -> None:
            Client.calls += 1

        async def receive_response(self) -> Any:
            if text:
                yield sdk_types.AssistantMessage(
                    content=[sdk_types.TextBlock(text=text)],
                    model="<synthetic>",
                    error=error,  # type: ignore[arg-type]
                )
            yield sdk_types.ResultMessage(
                subtype=subtype,
                duration_ms=5,
                duration_api_ms=0,
                is_error=True,
                num_turns=1,
                session_id="session",
                total_cost_usd=0.0,
                usage={"input_tokens": 0, "output_tokens": 0},
                result=text or None,
                api_error_status=status,
                errors=None if text else ["Claude Code process exited"],
            )

    return Client


@pytest.fixture
def failing_claude(monkeypatch: pytest.MonkeyPatch) -> type:
    client = _failing_client()
    monkeypatch.setattr(sdk, "ClaudeSDKClient", client)
    return client


def _config() -> MarkitaiConfig:
    cfg = MarkitaiConfig()
    cfg.llm.enabled = True
    cfg.llm.model_list = [
        ModelConfig(
            model_name="default",
            litellm_params=LiteLLMParams(model="claude-agent/haiku"),
        )
    ]
    # Transport retries would only repeat the same failed turn
    cfg.llm.router_settings.num_retries = 0
    cfg.cache.enabled = False
    return cfg


def _png(path: Path) -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 200), "white")
    ImageDraw.Draw(image).rectangle((40, 60, 120, 180), fill=(37, 99, 235))
    image.save(path)
    return path


# ---------------------------------------------------------------------------
# The providers
# ---------------------------------------------------------------------------


class TestClaudeAgentProvider:
    async def _complete(self, monkeypatch: pytest.MonkeyPatch, client: type) -> Any:
        from markitai.providers.claude_agent import ClaudeAgentProvider

        monkeypatch.setattr(sdk, "ClaudeSDKClient", client)
        return await ClaudeAgentProvider().acompletion(
            "claude-agent/haiku", [{"role": "user", "content": "Describe it."}]
        )

    async def test_a_usage_limit_is_a_rate_limit_not_an_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from litellm.exceptions import RateLimitError

        with pytest.raises(RateLimitError, match="usage limit reached") as raised:
            await self._complete(monkeypatch, _failing_client())
        assert "HTTP 429" in str(raised.value)

    @pytest.mark.parametrize(
        ("error", "status", "text"),
        [
            ("authentication_failed", 401, "Invalid API key · Please run /login"),
            ("billing_error", 402, "Credit balance is too low"),
        ],
    )
    async def test_a_refused_login_or_balance_needs_the_user(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error: str,
        status: int,
        text: str,
    ) -> None:
        from litellm.exceptions import AuthenticationError

        with pytest.raises(AuthenticationError, match=text.split(" ·")[0]):
            await self._complete(
                monkeypatch, _failing_client(text, error=error, status=status)
            )

    async def test_any_other_failed_turn_raises_with_its_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(RuntimeError, match="server error, HTTP 529.*Overloaded"):
            await self._complete(
                monkeypatch,
                _failing_client(
                    "API Error: Overloaded", error="server_error", status=529
                ),
            )

    async def test_a_failed_result_without_text_still_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(RuntimeError, match="process exited"):
            await self._complete(
                monkeypatch,
                _failing_client(
                    "", error=None, status=None, subtype="error_during_execution"
                ),
            )

    async def test_a_successful_turn_still_answers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Answering(_failing_client()):  # type: ignore[misc]
            async def receive_response(self) -> Any:
                yield sdk_types.AssistantMessage(
                    content=[sdk_types.TextBlock(text="A blue bar.")], model="haiku"
                )
                yield sdk_types.ResultMessage(
                    subtype="success",
                    duration_ms=5,
                    duration_api_ms=5,
                    is_error=False,
                    num_turns=1,
                    session_id="session",
                    usage={"input_tokens": 80, "output_tokens": 4},
                    result="A blue bar.",
                )

        response = await self._complete(monkeypatch, Answering)
        assert response.choices[0].message.content == "A blue bar."


async def test_a_copilot_turn_without_an_answer_is_not_the_text_none() -> None:
    import sys
    import types

    from markitai.providers.copilot import CopilotProvider

    client = AsyncMock()
    client.get_auth_status.return_value = MagicMock(isAuthenticated=True)
    client.create_session.return_value.send_and_wait.return_value = None
    copilot = types.ModuleType("copilot")
    copilot.CopilotClient = MagicMock(return_value=client)  # type: ignore[attr-defined]
    copilot.PermissionHandler = MagicMock()  # type: ignore[attr-defined]
    with (
        patch(
            "markitai.providers.copilot._is_copilot_sdk_available", return_value=True
        ),
        patch.dict(sys.modules, {"copilot": copilot}),
        pytest.raises(RuntimeError, match="without an answer"),
    ):
        await CopilotProvider().acompletion(
            "copilot/gpt-5-mini", [{"role": "user", "content": "Describe it."}]
        )


# ---------------------------------------------------------------------------
# The entry points
# ---------------------------------------------------------------------------


def _assert_kept_base_and_said_why(
    out: Path, base_name: str, warnings: list[str]
) -> None:
    assert (out / base_name).is_file()
    assert not list(out.rglob("*.llm.md"))
    assert any(
        "LLM enhancement failed" in w and "usage limit reached" in w for w in warnings
    ), warnings
    assert LIMIT not in (out / base_name).read_text(encoding="utf-8")


@pytest.mark.usefixtures("failing_claude")
@pytest.mark.parametrize("name", ["chart.png", "notes.md"])
async def test_a_single_file_keeps_its_base_output(tmp_path: Path, name: str) -> None:
    from markitai.cli.processors.file import process_single_file
    from markitai.runs import Outcome

    source = tmp_path / name
    if name.endswith(".png"):
        _png(source)
    else:
        source.write_text("# Notes\n\n" + "Plain text. " * 40, encoding="utf-8")
    cfg = _config()
    cfg.image.alt_enabled = True
    history: list[Outcome] = []
    await process_single_file(
        source, tmp_path / "out", cfg, dry_run=False, quiet=True, history=history
    )
    (outcome,) = history
    assert outcome.status == "completed"
    assert outcome.output_path == tmp_path / "out" / f"{name}.md"
    _assert_kept_base_and_said_why(tmp_path / "out", f"{name}.md", outcome.warnings)


@pytest.mark.usefixtures("failing_claude")
async def test_a_directory_batch_keeps_each_base_output(tmp_path: Path) -> None:
    from markitai.cli.processors.batch import create_process_file
    from markitai.workflow.helpers import create_llm_processor

    inputs = tmp_path / "in"
    inputs.mkdir()
    _png(inputs / "chart.png")
    (inputs / "notes.md").write_text("# Notes\n\n" + "Text. " * 60, encoding="utf-8")
    cfg = _config()
    cfg.image.alt_enabled = True
    cfg.image.desc_enabled = True
    process = create_process_file(
        cfg, inputs, tmp_path / "out", create_llm_processor(cfg)
    )
    for name in ("chart.png", "notes.md"):
        result = await process(inputs / name)
        assert result.success, result.error
        assert result.llm_enhanced is False
        assert result.output_path == str(tmp_path / "out" / f"{name}.md")
        _assert_kept_base_and_said_why(tmp_path / "out", f"{name}.md", result.warnings)


@pytest.mark.usefixtures("failing_claude")
async def test_a_url_keeps_its_base_output(tmp_path: Path) -> None:
    from markitai.fetch_types import FetchResult
    from markitai.workflow.helpers import create_llm_processor
    from markitai.workflow.url import convert_url_cascade

    url = "https://example.com/report"
    cfg = _config()
    result = await convert_url_cascade(
        url,
        cfg,
        tmp_path / "out",
        processor=create_llm_processor(cfg),
        fetch_result=FetchResult(
            content="# Report\n\n" + "Revenue grew. " * 40,
            strategy_used="static",
            url=url,
        ),
    )
    assert result.llm_output_path is None
    assert result.output_path is not None
    _assert_kept_base_and_said_why(
        tmp_path / "out", result.output_path.name, result.warnings
    )


@pytest.mark.usefixtures("failing_claude")
async def test_a_serve_job_keeps_the_base_output_and_says_why(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from tests.unit.serve.test_serve_api import (
        _make_app,
        _multipart,
        _serve_client,
        _wait_job_done,
    )

    png = _png(tmp_path / "chart.png")
    async with _serve_client(_make_app(tmp_path, _config())) as client:
        response = await client.post(
            "/api/jobs",
            files=_multipart(
                files=[("chart.png", png.read_bytes())],
                options={"llm": True, "alt": True},
            ),
        )
        assert response.status_code == 201, response.text
        job = await asyncio.wait_for(
            _wait_job_done(client, response.json()["job_id"]), timeout=90
        )
    (item,) = job["items"]
    assert item["status"] == "done", item
    assert item["output"].endswith("chart.png.md")
    assert item["llm_enhanced"] is False
    out = next((tmp_path / "jobs").glob("*/out"))
    _assert_kept_base_and_said_why(out, "chart.png.md", item["warnings"])
