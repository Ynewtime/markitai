"""A missing API key is said once and nothing is sent; a refused one is said once.

Regression: with ``DEEPSEEK_API_KEY`` unset, ``--llm`` sent the request
anyway (LiteLLM sends it without a key), DeepSeek answered HTTP 401
"Authentication Fails (governor)", and the console printed that four times
(the engine's error with a generic hint, "LLM enhancement failed; using
fallback frontmatter", "LLM processing failed: ...", the item's warning)
without ever naming the variable to set; the item still ended "completed".

Every request here goes to a loopback stub (``/ok/...`` answers a chat
completion, anything else HTTP 401 the way DeepSeek does); a test that
expects nothing to be sent asserts the stub saw nothing.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from typing import Any

import litellm
import pytest
from litellm.exceptions import AuthenticationError

from markitai.config import (
    LiteLLMParams,
    LLMConfig,
    MarkitaiConfig,
    ModelConfig,
    ModelInfo,
    PromptsConfig,
    RouterSettings,
)
from markitai.constants import (
    PROVIDER_API_KEY_ENV,
    PROVIDER_API_KEY_ENV_NAMES,
    PROVIDER_DEFAULT_MODELS,
)
from markitai.llm.engine import LLMCall
from markitai.llm.processor import LLMProcessor
from markitai.llm.router import MarkitaiRouter
from markitai.notices import capture_task_notices
from markitai.providers import api_keys
from markitai.providers.api_keys import (
    APIKeyRejectedError,
    MissingAPIKeyError,
    MissingKey,
    api_key_rejection,
    env_set_command,
    is_api_key_problem,
    left_out_message,
    missing_api_key,
    no_usable_model_message,
    split_by_api_key,
)
from markitai.providers.errors import ProviderError
from markitai.utils.text import format_error_message
from tests.local_http import LocalHTTPServer

FAKE_KEY = "kv-" + "unit-test-not-a-key-" * 2
AUTH_FAILS = "Authentication Fails (governor)"


class _Stub(BaseHTTPRequestHandler):
    """Chat endpoint: ``/ok/...`` answers, every other path is a 401."""

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.seen.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
            }
        )
        if self.path.startswith("/ok"):
            status = 200
            payload: dict[str, Any] = {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "stub"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Cleaned."},
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            }
        else:
            status = 401
            payload = {
                "error": {
                    "message": AUTH_FAILS,
                    "type": "authentication_error",
                    "param": None,
                    "code": None,
                }
            }
        out = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *_args: Any) -> None:
        pass


class StubServer:
    def __init__(self) -> None:
        self.server = LocalHTTPServer(("127.0.0.1", 0), _Stub)
        self.server.seen = []  # type: ignore[attr-defined]
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    @property
    def seen(self) -> list[dict[str, Any]]:
        return self.server.seen  # type: ignore[attr-defined]


@pytest.fixture
def stub() -> Iterator[StubServer]:
    server = StubServer()
    thread = threading.Thread(target=server.server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.server.shutdown()
        server.server.server_close()


@pytest.fixture(autouse=True)
def _fresh_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with no left-out model already reported."""
    monkeypatch.setattr(api_keys, "_reported", set())


def _model(model: str, **params: Any) -> ModelConfig:
    info = params.pop("model_info", None)
    return ModelConfig(
        model_name="default",
        litellm_params=LiteLLMParams(model=model, **params),
        model_info=info,
    )


def _processor(*models: ModelConfig) -> LLMProcessor:
    config = LLMConfig(
        enabled=True,
        model_list=list(models),
        router_settings=RouterSettings(num_retries=0),
    )
    return LLMProcessor(config, PromptsConfig(), cache_enabled=False)


def _messages() -> list[dict[str, Any]]:
    return [{"role": "user", "content": "Say hi."}]


# =============================================================================
# Which models lack a key
# =============================================================================


class TestMissingApiKey:
    def test_deepseek_without_its_key(self) -> None:
        assert missing_api_key(LiteLLMParams(model="deepseek/deepseek-flash")) == (
            MissingKey("deepseek/deepseek-flash", "DEEPSEEK_API_KEY")
        )

    @pytest.mark.parametrize(
        ("provider", "env_name"),
        [
            (provider, name)
            for provider, names in PROVIDER_API_KEY_ENV_NAMES.items()
            for name in names
        ],
    )
    def test_every_name_litellm_reads_counts(
        self, provider: str, env_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        params = LiteLLMParams(model=PROVIDER_DEFAULT_MODELS[provider])
        assert missing_api_key(params) == MissingKey(
            params.model, PROVIDER_API_KEY_ENV[provider]
        )
        monkeypatch.setenv(env_name, FAKE_KEY)
        assert missing_api_key(params) is None

    def test_an_empty_variable_is_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "")
        assert missing_api_key(LiteLLMParams(model="deepseek/x")) is not None

    def test_configured_key_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert (
            missing_api_key(LiteLLMParams(model="openai/x", api_key=FAKE_KEY)) is None
        )
        params = LiteLLMParams(model="openai/x", api_key="env:MY_TEAM_KEY")
        assert missing_api_key(params) == MissingKey("openai/x", "MY_TEAM_KEY")
        monkeypatch.setenv("MY_TEAM_KEY", FAKE_KEY)
        assert missing_api_key(params) is None

    def test_unset_env_reference_in_api_base(self) -> None:
        params = LiteLLMParams(model="ollama/llama3", api_base="env:MY_OLLAMA")
        assert missing_api_key(params) == MissingKey("ollama/llama3", "MY_OLLAMA")

    @pytest.mark.parametrize("provider", ["deepseek", "openrouter"])
    def test_custom_api_base_may_need_no_key(self, provider: str) -> None:
        """A local server or a key-injecting gateway is never held back."""
        params = LiteLLMParams(
            model=PROVIDER_DEFAULT_MODELS[provider], api_base="http://127.0.0.1:8000/v1"
        )
        assert missing_api_key(params) is None

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
    def test_api_base_does_not_spare_a_key_litellm_requires(
        self, provider: str
    ) -> None:
        params = LiteLLMParams(
            model=PROVIDER_DEFAULT_MODELS[provider], api_base="http://127.0.0.1:8000/v1"
        )
        missing = missing_api_key(params)
        assert missing is not None
        assert missing == MissingKey(
            params.model, PROVIDER_API_KEY_ENV[provider], api_base=True
        )
        assert "any value works" in left_out_message(missing)

    @pytest.mark.parametrize(
        "model",
        [
            "claude-agent/sonnet",
            "copilot/auto",
            "chatgpt/gpt-6-luna",
            "ollama/llama3",
            "groq/llama-3.3-70b",
            "gpt-4o-mini",  # unprefixed: LiteLLM's guess, not checked
        ],
    )
    def test_other_providers_are_left_alone(self, model: str) -> None:
        assert missing_api_key(LiteLLMParams(model=model)) is None

    def test_split_skips_disabled_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
        usable, missing = split_by_api_key(
            [
                _model("deepseek/deepseek-flash"),
                _model("openrouter/x/y"),
                _model("gemini/z", weight=0),
            ]
        )
        assert [m.litellm_params.model for m in usable] == ["openrouter/x/y"]
        assert missing == [MissingKey("deepseek/deepseek-flash", "DEEPSEEK_API_KEY")]


class TestMessages:
    def test_one_variable(self) -> None:
        message = no_usable_model_message(
            [MissingKey("deepseek/deepseek-flash", "DEEPSEEK_API_KEY")], dotenv=True
        )
        assert message == (
            "DEEPSEEK_API_KEY is not set, and deepseek/deepseek-flash needs it; "
            f"set it with: {env_set_command('DEEPSEEK_API_KEY')} "
            "(or add DEEPSEEK_API_KEY=your_key to ~/.markitai/.env)"
        )
        assert "\n" not in message

    def test_several_variables(self) -> None:
        message = no_usable_model_message(
            [
                MissingKey("deepseek/deepseek-flash", "DEEPSEEK_API_KEY"),
                MissingKey("gemini/gemini-flash-lite-latest", "GEMINI_API_KEY"),
            ]
        )
        assert message.startswith(
            "No LLM model can be used: DEEPSEEK_API_KEY is not set "
            "(deepseek/deepseek-flash); GEMINI_API_KEY is not set "
            "(gemini/gemini-flash-lite-latest)."
        )
        assert env_set_command("DEEPSEEK_API_KEY") in message
        assert ".env" not in message  # the Python API reads no .env file

    def test_left_out(self) -> None:
        assert left_out_message(
            MissingKey("deepseek/deepseek-flash", "DEEPSEEK_API_KEY")
        ) == (
            "deepseek/deepseek-flash is left out of the model pool: "
            "DEEPSEEK_API_KEY is not set; set it with: "
            f"{env_set_command('DEEPSEEK_API_KEY')}"
        )


# =============================================================================
# The model pool: nothing is sent without a key
# =============================================================================


class TestModelPool:
    def test_no_usable_model_fails_before_anything_is_sent(
        self, stub: StubServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Where the request would go if the check let it through
        monkeypatch.setenv("DEEPSEEK_API_BASE", stub.url)
        with pytest.raises(MissingAPIKeyError) as caught:
            _processor(_model("deepseek/deepseek-flash"))
        assert isinstance(caught.value, ValueError)
        message = format_error_message(caught.value)
        assert message.startswith("DEEPSEEK_API_KEY is not set")
        assert stub.seen == []

    def test_unset_env_reference_keeps_its_variable_name(self) -> None:
        with pytest.raises(MissingAPIKeyError, match="MY_TEAM_KEY is not set"):
            _processor(_model("deepseek/deepseek-flash", api_key="env:MY_TEAM_KEY"))

    async def test_a_model_without_its_key_is_left_out_once(
        self, stub: StubServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Where deepseek's requests would go if the check let them through
        monkeypatch.setenv("DEEPSEEK_API_BASE", stub.url)
        monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
        models = (
            _model("deepseek/deepseek-flash"),
            _model("openrouter/stub/model", api_base=f"{stub.url}/ok"),
        )
        with capture_task_notices() as notices:
            processor = _processor(*models)
            for _ in range(3):
                await processor.engine.complete_text(
                    model="default", messages=_messages(), call_id="t", max_retries=0
                )
            # A second processor (a serve job, an API call) says nothing new
            _ = _processor(*models).router

        assert [r["path"] for r in stub.seen] == ["/ok/chat/completions"] * 3
        assert notices == [
            left_out_message(MissingKey("deepseek/deepseek-flash", "DEEPSEEK_API_KEY"))
        ]

    def test_reported_by_the_cli_is_not_repeated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
        missing = MissingKey("deepseek/deepseek-flash", "DEEPSEEK_API_KEY")
        api_keys.mark_reported([missing])
        with capture_task_notices() as notices:
            _ = _processor(
                _model("deepseek/deepseek-flash"), _model("openrouter/x/y")
            ).router
        assert notices == []

    def test_vision_pool_without_keys_falls_back_to_the_main_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A vision subset made only of keyless models must not fail the
        processor while the main pool can still serve."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
        processor = _processor(
            _model(
                "deepseek/deepseek-flash", model_info=ModelInfo(supports_vision=False)
            ),
            _model("gemini/vision-model", model_info=ModelInfo(supports_vision=True)),
        )
        assert processor.vision_router is processor.router
        assert [e["litellm_params"]["model"] for e in processor.router.model_list] == [
            "deepseek/deepseek-flash"
        ]


# =============================================================================
# A key the provider refuses
# =============================================================================


def _router(stub: StubServer, **params: Any) -> MarkitaiRouter:
    return MarkitaiRouter(
        model_list=[
            {
                "model_name": "default",
                "litellm_params": {
                    "model": "deepseek/deepseek-flash",
                    "api_base": stub.url,
                    **params,
                },
            }
        ],
        router_settings={"num_retries": 0},
    )


class TestRefusedKey:
    async def test_names_the_variable_once_and_is_not_resent(
        self, stub: StubServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
        router = _router(stub)
        with pytest.raises(APIKeyRejectedError) as caught:
            await router.acompletion("default", _messages())
        error = caught.value
        assert str(error) == (
            "deepseek/deepseek-flash rejected the API key in DEEPSEEK_API_KEY with "
            f"HTTP 401; set a valid one with: {env_set_command('DEEPSEEK_API_KEY')}. "
            f"Provider message: {AUTH_FAILS}"
        )
        assert error.status_code == 401
        assert isinstance(error, ProviderError) and not error.retryable
        # Shown bare (no class-name prefix) and never cut before the fix
        assert format_error_message(error) == str(error)
        wrapped = format_error_message(RuntimeError(f"Vision LLM failed: {error}"))
        assert env_set_command("DEEPSEEK_API_KEY") in wrapped
        assert is_api_key_problem(str(error))
        assert FAKE_KEY not in str(error)

        # The key does not change during a run: no second request
        with pytest.raises(APIKeyRejectedError):
            await router.acompletion("default", _messages())
        assert len(stub.seen) == 1

    async def test_configured_key(self, stub: StubServer) -> None:
        with pytest.raises(APIKeyRejectedError) as caught:
            await _router(stub, api_key=FAKE_KEY).acompletion("default", _messages())
        assert str(caught.value).startswith(
            "deepseek/deepseek-flash rejected the api_key configured for it in "
            "llm.model_list with HTTP 401; check that key. Provider message: "
        )

    async def test_structured_call_stops_at_the_first_refusal(
        self, stub: StubServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The structured-output staircase does not resend a refused key."""
        from pydantic import BaseModel

        class Title(BaseModel):
            title: str

        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        processor = _processor(_model("openai/gpt-4o-mini", api_base=stub.url))
        call = LLMCall(
            purpose="test", messages=_messages(), response_model=Title, context="a.md"
        )
        with pytest.raises(APIKeyRejectedError, match="OPENAI_API_KEY"):
            await processor.engine.complete_structured(call)
        assert len(stub.seen) == 1

    def test_the_variable_litellm_read_is_named(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LiteLLM prefers GOOGLE_API_KEY to GEMINI_API_KEY: name that one."""
        monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
        monkeypatch.setenv("GOOGLE_API_KEY", FAKE_KEY + "-stale")
        error = AuthenticationError(
            message="API key not valid", llm_provider="gemini", model="gemini/x"
        )
        rejected = api_key_rejection(error, [])
        assert rejected is not None
        assert "rejected the API key in GOOGLE_API_KEY" in str(rejected)

    def test_local_providers_keep_their_own_errors(self) -> None:
        error = AuthenticationError(
            message="claude-agent authentication error: not logged in",
            llm_provider="claude-agent",
            model="claude-agent/sonnet",
        )
        assert api_key_rejection(error, []) is None


# =============================================================================
# The pinned LiteLLM behaviour the check relies on
# =============================================================================


def _credential_headers(request: dict[str, Any]) -> str:
    headers = request["headers"]
    return " ".join(
        headers.get(name, "")
        for name in ("authorization", "x-api-key", "x-goog-api-key")
    )


class TestLiteLLMParity:
    @pytest.mark.parametrize(
        ("provider", "env_name"),
        [
            (provider, name)
            for provider, names in PROVIDER_API_KEY_ENV_NAMES.items()
            for name in names
        ],
    )
    async def test_litellm_sends_the_key_of_every_listed_variable(
        self,
        provider: str,
        env_name: str,
        stub: StubServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(env_name, FAKE_KEY)
        with pytest.raises(Exception):  # noqa: B017 - the stub answers 401
            await litellm.acompletion(
                model=PROVIDER_DEFAULT_MODELS[provider],
                messages=_messages(),
                api_base=stub.url,
                num_retries=0,
                max_tokens=8,
            )
        [request] = stub.seen
        assert FAKE_KEY in _credential_headers(request)

    @pytest.mark.parametrize("provider", sorted(PROVIDER_API_KEY_ENV_NAMES))
    async def test_custom_api_base_without_a_key(
        self, provider: str, stub: StubServer
    ) -> None:
        """markitai lets a keyless model with api_base through exactly when
        LiteLLM would send its request there."""
        model = PROVIDER_DEFAULT_MODELS[provider]
        with pytest.raises(Exception):  # noqa: B017 - no key: refused or 401
            await litellm.acompletion(
                model=model,
                messages=_messages(),
                api_base=stub.url,
                num_retries=0,
                max_tokens=8,
            )
        sent = bool(stub.seen)
        held_back = missing_api_key(LiteLLMParams(model=model, api_base=stub.url))
        assert sent == (held_back is None)


# =============================================================================
# Entry points
# =============================================================================


class TestEntryPoints:
    def test_python_api_fails_before_converting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.api import NoModelConfiguredError, _resolve_config

        monkeypatch.setenv("MODEL", "deepseek/deepseek-flash")
        with pytest.raises(
            NoModelConfiguredError, match="^DEEPSEEK_API_KEY is not set"
        ):
            _resolve_config(
                MarkitaiConfig(),
                llm=True,
                ocr=None,
                screenshot=None,
                alt=None,
                desc=None,
            )

    def test_python_api_runs_with_the_models_that_have_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.api import _resolve_config

        monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
        config = MarkitaiConfig()
        config.llm.model_list = [
            _model("deepseek/deepseek-flash"),
            _model("openrouter/x/y"),
        ]
        cfg = _resolve_config(
            config, llm=True, ocr=None, screenshot=None, alt=None, desc=None
        )
        assert len(cfg.llm.model_list) == 2  # the router leaves deepseek out

    async def test_mcp_tool_error(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("mcp")
        from mcp.server.mcpserver.exceptions import ToolError

        from markitai.mcp.server import convert_document

        source = tmp_path / "a.md"
        source.write_text("# A\n\nBody.\n", encoding="utf-8")
        monkeypatch.setenv("MODEL", "deepseek/deepseek-flash")
        with pytest.raises(ToolError) as caught:
            await convert_document(
                str(source), output_dir=str(tmp_path / "o"), llm=True
            )
        message = str(caught.value)
        assert message.startswith("DEEPSEEK_API_KEY is not set")
        assert f"{env_set_command('DEEPSEEK_API_KEY')}. For this MCP server" in message


class TestFallbackHint:
    def test_policy_hint_waits_for_a_failure_it_helps_with(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai.workflow import llm_failure

        monkeypatch.setattr(llm_failure, "_hinted", False)
        refused = (
            "LLM processing failed: deepseek/deepseek-flash rejected the API key "
            "in DEEPSEEK_API_KEY with HTTP 401; set a valid one with: export "
            "DEEPSEEK_API_KEY=your_key"
        )
        with capture_task_notices() as notices:
            llm_failure.llm_fallback_warning("a.md", refused)
            llm_failure.llm_fallback_warning("b.md", "LLM processing failed: boom")
        assert len(notices) == 3
        assert notices[0].startswith("a.md: LLM enhancement failed (deepseek/")
        assert notices[1] == (
            "b.md: LLM enhancement failed (boom); kept the unenhanced output"
        )
        assert "llm.on_failure" in notices[2]


class TestRedaction:
    def test_set_key_hints_are_not_masked(self) -> None:
        from markitai.utils.secret_redaction import redact_secrets

        for hint in (
            "export DEEPSEEK_API_KEY=your_key",
            '$env:DEEPSEEK_API_KEY="your_key"',
            "set DEEPSEEK_API_KEY=your_key",
        ):
            assert redact_secrets(hint) == hint
        assert "yourself-is-a-real-key" not in redact_secrets(
            "api_key=yourself-is-a-real-key"
        )
