"""The configured ``max_tokens`` is what goes out on the wire.

Regression: ``litellm_params.max_tokens: 8192`` was dropped when the router
was built, and every request carried the model's catalog maximum instead
(``max_tokens: 128000`` for DeepSeek V4, ``max_output_tokens: 65535`` for
Gemini), raising the worst-case cost of each call. The wire tests point a
real router at an in-process capture server on 127.0.0.1, so they inspect
the outgoing request body without touching the network.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import MagicMock, patch

import instructor
import pytest
from pydantic import BaseModel

from markitai.cli.processors.batch_llm import (
    _anthropic_max_tokens,
    _configured_max_tokens,
)
from markitai.config import (
    LiteLLMParams,
    LLMConfig,
    ModelConfig,
    ModelInfo,
    PromptsConfig,
    RouterSettings,
)
from markitai.constants import DEFAULT_MAX_OUTPUT_TOKENS_HARD_CAP
from markitai.llm.batch_api import build_openai_batch_request
from markitai.llm.engine import LLMCall
from markitai.llm.processor import LLMProcessor

# DeepSeek V4 Flash as litellm's catalog describes it: an output ceiling far
# above the hard cap, so an ignored config shows up as the 128k cap.
_CATALOG = {"max_input_tokens": 1_000_000, "max_output_tokens": 393_216}


class _Title(BaseModel):
    title: str


class _CaptureHandler(BaseHTTPRequestHandler):
    """Records each chat-completions body and answers with a fixed JSON."""

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        length = int(self.headers.get("Content-Length", 0))
        self.server.bodies.append(json.loads(self.rfile.read(length)))  # type: ignore[attr-defined]
        reply = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"title": "Hello"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def capture_server() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    server.bodies = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", server.bodies  # type: ignore[attr-defined]
    finally:
        server.shutdown()
        server.server_close()


def _processor(api_base: str, max_tokens: int | None) -> LLMProcessor:
    config = LLMConfig(
        enabled=True,
        model_list=[
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(
                    model="deepseek/deepseek-v4-flash",
                    api_key="test-key",
                    api_base=api_base,
                    max_tokens=max_tokens,
                ),
            )
        ],
        router_settings=RouterSettings(num_retries=0),
    )
    return LLMProcessor(config, PromptsConfig(), cache_enabled=False)


async def _send_text_and_structured(processor: LLMProcessor) -> None:
    messages = [{"role": "user", "content": "Give the title as JSON."}]
    await processor.engine.complete_text(
        model="default", messages=messages, call_id="text"
    )
    await processor.engine.complete_structured(
        LLMCall(
            purpose="test",
            messages=messages,
            response_model=_Title,
            context="doc.md",
        )
    )


class TestWireMaxTokens:
    async def test_configured_max_tokens_is_sent(
        self, capture_server: tuple[str, list[dict[str, Any]]]
    ) -> None:
        api_base, bodies = capture_server
        with patch(
            "markitai.llm.processor.get_model_info_cached", return_value=_CATALOG
        ):
            await _send_text_and_structured(_processor(api_base, max_tokens=8192))

        assert len(bodies) == 2
        assert [b["max_tokens"] for b in bodies] == [8192, 8192]

    async def test_unset_max_tokens_keeps_the_capped_model_limit(
        self, capture_server: tuple[str, list[dict[str, Any]]]
    ) -> None:
        api_base, bodies = capture_server
        with patch(
            "markitai.llm.processor.get_model_info_cached", return_value=_CATALOG
        ):
            await _send_text_and_structured(_processor(api_base, max_tokens=None))

        assert [b["max_tokens"] for b in bodies] == [
            DEFAULT_MAX_OUTPUT_TOKENS_HARD_CAP,
            DEFAULT_MAX_OUTPUT_TOKENS_HARD_CAP,
        ]


class TestDynamicMaxTokensOverrides:
    """``_calculate_dynamic_max_tokens`` reads the router entries' config."""

    def _calculate(self, entries: list[dict[str, Any]], content: str = "hi") -> Any:
        processor = LLMProcessor(LLMConfig(enabled=True), PromptsConfig())
        router = MagicMock()
        router.model_list = entries
        with (
            patch(
                "markitai.llm.processor.get_model_info_cached", return_value=_CATALOG
            ),
            patch("litellm.token_counter", return_value=100),
        ):
            return processor._calculate_dynamic_max_tokens(
                [{"role": "user", "content": content}], router=router
            )

    def test_smallest_configured_cap_in_the_pool_wins(self) -> None:
        entries = [
            {"litellm_params": {"model": "deepseek/a", "max_tokens": 8192}},
            {"litellm_params": {"model": "deepseek/b", "max_tokens": 4096}},
        ]
        assert self._calculate(entries) == 4096

    def test_model_info_limit_overrides_the_catalog(self) -> None:
        entries = [
            {
                "litellm_params": {"model": "deepseek/a"},
                "model_info": {"max_tokens": 16000, "max_input_tokens": None},
            }
        ]
        assert self._calculate(entries) == 16000

    def test_floor_does_not_exceed_a_small_configured_cap(self) -> None:
        entries = [{"litellm_params": {"model": "deepseek/a", "max_tokens": 512}}]
        assert self._calculate(entries, content="|a|b|\n" * 30) == 512


class TestBatchMaxTokens:
    def _cfg(self, *entries: ModelConfig) -> Any:
        cfg = MagicMock()
        cfg.llm.model_list = list(entries)
        return cfg

    def test_configured_cap_for_the_batch_model(self) -> None:
        cfg = self._cfg(
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(
                    model="anthropic/claude-haiku-4-5", max_tokens=8192
                ),
            ),
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model="openai/gpt-4o", max_tokens=100),
            ),
        )
        assert _configured_max_tokens(cfg, "anthropic/claude-haiku-4-5") == 8192
        assert _anthropic_max_tokens("claude-haiku-4-5", 8192) == 8192

    def test_cap_above_the_model_limit_is_clamped(self) -> None:
        # The live path's litellm.modify_params clamps this; a batch body
        # with 100000 would fail every request against a 64000 limit.
        cfg = self._cfg(
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(
                    model="anthropic/claude-haiku-4-5", max_tokens=100_000
                ),
            )
        )
        with patch("litellm.get_model_info", return_value={"max_output_tokens": 64000}):
            assert _configured_max_tokens(cfg, "anthropic/claude-haiku-4-5") == 64000
        with patch("litellm.get_model_info", side_effect=Exception("unknown")):
            assert _configured_max_tokens(cfg, "anthropic/claude-haiku-4-5") == 100_000

    def test_model_info_limit_replaces_the_catalog_limit(self) -> None:
        cfg = self._cfg(
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model="openai/gpt-4o", max_tokens=50_000),
                model_info=ModelInfo(max_tokens=32_000),
            )
        )
        with patch("litellm.get_model_info", return_value={"max_output_tokens": 16384}):
            assert _configured_max_tokens(cfg, "openai/gpt-4o") == 32_000

    def test_model_info_limit_is_the_fallback(self) -> None:
        cfg = self._cfg(
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model="openai/gpt-4o"),
                model_info=ModelInfo(max_tokens=2048),
            )
        )
        assert _configured_max_tokens(cfg, "openai/gpt-4o") == 2048

    def test_unset_keeps_the_previous_defaults(self) -> None:
        cfg = self._cfg(
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(model="anthropic/claude-haiku-4-5"),
            )
        )
        configured = _configured_max_tokens(cfg, "anthropic/claude-haiku-4-5")
        assert configured is None
        assert _anthropic_max_tokens("claude-haiku-4-5", configured) > 8192

    def test_openai_batch_uses_max_completion_tokens(self) -> None:
        req = build_openai_batch_request(
            "d",
            messages=[{"role": "user", "content": "x"}],
            response_model=_Title,
            model="gpt-5.4-mini",
            mode=instructor.Mode.TOOLS,
            max_tokens=8192,
        )
        assert req["body"]["max_completion_tokens"] == 8192
        assert "max_tokens" not in req["body"]
