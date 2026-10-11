"""A failed LLM request must not print the key or the document it sent.

A loopback provider answers like a real one that echoes its input: an HTTP
401 naming the key the request carried, and HTTP 400s quoting (or simply
repeating) the prompt. Every surface that shows the provider's error (the
raised exception, logs, ``--json``, stderr, log files, batch reports and the
Python API that MCP calls) must keep the error type, status and message
while dropping the key and the document text.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest
from loguru import logger
from pydantic import BaseModel

from markitai.config import (
    LiteLLMParams,
    LLMConfig,
    MarkitaiConfig,
    ModelConfig,
    PromptsConfig,
    RouterSettings,
)
from markitai.llm.batch_api import read_openai_batch_output
from markitai.llm.engine import LLMCall
from markitai.llm.error_redaction import (
    WITHHELD,
    redact_provider_exception,
    redact_provider_message,
)
from markitai.llm.processor import LLMProcessor
from markitai.utils.secret_redaction import REDACTED, redact_secrets, register_secret
from markitai.utils.text import format_error_message
from markitai.utils.url_redaction import redact_log_record
from tests.local_http import LocalHTTPServer

# Built at run time so no key-shaped literal sits in the repository. The
# configured key has no known prefix: only its registered value can mask it.
KEY = "markitai-dummy-" + "Q7x9Lm2P" * 4
PREFIXED_KEY = "sk-" + "proj-" + "a1B2c3D4" * 4
DOC_TEXT = "CONFIDENTIAL-DOC-TEXT the quarterly widget count rose from 12 to 15"
DOCUMENT = f"# Synthetic note\n\n{DOC_TEXT} in the synthetic test region.\n"


class _Title(BaseModel):
    title: str


class _EchoingProvider(BaseHTTPRequestHandler):
    """OpenAI-compatible endpoint whose errors echo the request.

    ``/key/v1``: 401 naming the bearer key; ``/quote/v1``: 400 quoting the
    user prompt; ``/inline/v1``: 400 repeating it without quotes.
    """

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length))
        content = request["messages"][-1]["content"]
        user = content if isinstance(content, str) else content[0]["text"]
        mode = self.path.strip("/").split("/")[0]
        if mode == "key":
            key = self.headers.get("Authorization", "").removeprefix("Bearer ")
            status, message, code = (
                401,
                f"Incorrect API key provided: {key}. You can find your API key "
                "at https://platform.example.test/account/api-keys.",
                "invalid_api_key",
            )
        elif mode == "quote":
            status, message, code = (
                400,
                f"Could not parse the request content starting with: {user[:400]!r}",
                "invalid_request",
            )
        else:
            status, message, code = 400, f"Invalid content: {user[:300]}", None
        body = json.dumps(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": code,
                }
            }
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:
        pass


@pytest.fixture
def provider() -> Iterator[str]:
    """Base URL of the echoing provider; append ``/<mode>/v1``."""
    server = LocalHTTPServer(("127.0.0.1", 0), _EchoingProvider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def captured_logs() -> Iterator[list[str]]:
    lines: list[str] = []
    handler_id = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
    try:
        yield lines
    finally:
        logger.remove(handler_id)


def _model(api_base: str) -> ModelConfig:
    return ModelConfig(
        model_name="default",
        litellm_params=LiteLLMParams(
            model="openai/gpt-4o-mini", api_key=KEY, api_base=api_base
        ),
    )


def _processor(api_base: str) -> LLMProcessor:
    config = LLMConfig(
        enabled=True,
        model_list=[_model(api_base)],
        router_settings=RouterSettings(num_retries=0),
    )
    return LLMProcessor(config, PromptsConfig(), cache_enabled=False)


def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "Return the document title."},
        {"role": "user", "content": f"Please process the document:\n\n{DOCUMENT}"},
    ]


def _assert_clean(text: str) -> None:
    assert KEY not in text
    assert KEY[15:31] not in text  # no long fragment either
    assert "Synthetic note" not in text  # the document's first line
    assert "CONFIDENTIAL-DOC-TEXT" not in text
    assert "quarterly widget" not in text


def _chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


class TestEngineRedaction:
    """The real LiteLLM path: exceptions are cleaned where the request fails."""

    async def test_401_echoing_the_key(
        self, provider: str, captured_logs: list[str]
    ) -> None:
        processor = _processor(f"{provider}/key/v1")
        with pytest.raises(Exception) as caught:
            await processor.engine.complete_text(
                model="default", messages=_messages(), call_id="t", max_retries=0
            )
        error = caught.value
        assert getattr(error, "status_code", None) == 401
        # The router turns the provider's 401 into one actionable error
        assert type(error).__name__ == "APIKeyRejectedError"
        assert "rejected the api_key configured for it" in str(error)
        assert f"Incorrect API key provided: {REDACTED}" in str(error)
        _assert_clean(str(error))
        _assert_clean(format_error_message(error))
        logs = "\n".join(captured_logs)
        assert "Authentication failed" in logs
        _assert_clean(logs)

    async def test_400_quoting_the_document(
        self, provider: str, captured_logs: list[str]
    ) -> None:
        processor = _processor(f"{provider}/quote/v1")
        call = LLMCall(
            purpose="test",
            messages=_messages(),
            response_model=_Title,
            context="note.md",
        )
        with pytest.raises(Exception) as caught:
            await processor.engine.complete_structured(call)
        error = caught.value
        assert any(getattr(e, "status_code", None) == 400 for e in _chain(error))
        assert "Could not parse the request content starting with: '…'" in str(error)
        _assert_clean(str(error))
        _assert_clean(format_error_message(error))
        _assert_clean("\n".join(captured_logs))

    async def test_400_repeating_the_document_unquoted(
        self, provider: str, captured_logs: list[str]
    ) -> None:
        processor = _processor(f"{provider}/inline/v1")
        with pytest.raises(Exception) as caught:
            await processor.engine.complete_text(
                model="default", messages=_messages(), call_id="t", max_retries=0
            )
        error = caught.value
        assert getattr(error, "status_code", None) == 400
        assert str(error).startswith("litellm.BadRequestError: OpenAIException - ")
        assert WITHHELD in str(error)
        _assert_clean(str(error))
        _assert_clean("\n".join(captured_logs))

    async def test_guarded_vision_path(self) -> None:
        """Vision hands the router to instructor directly, past the retry loop."""
        processor = _processor("http://127.0.0.1:9/v1")
        register_secret(KEY)

        async def failing(**_kwargs: Any) -> Any:
            raise RuntimeError(
                f"rejected {KEY}: content 'Please process the document: {DOCUMENT}'"
            )

        guarded = processor.engine.guard_acompletion(failing, "note.md")
        with pytest.raises(RuntimeError) as caught:
            await guarded(model="default", messages=_messages())
        assert str(caught.value) == f"rejected {REDACTED}: content '…'"


class TestApiRedaction:
    async def test_aconvert_error_carries_no_document(
        self, provider: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MCP tools report what ``aconvert`` raises."""
        from markitai.api import aconvert

        monkeypatch.setattr("markitai.api._configure_host_logging", lambda: None)
        source = tmp_path / "note.md"
        source.write_text(DOCUMENT, encoding="utf-8")
        config = MarkitaiConfig()
        config.llm.enabled = True
        config.llm.model_list = [_model(f"{provider}/quote/v1")]
        config.llm.router_settings = RouterSettings(num_retries=0)
        config.cache.enabled = False
        # Default on_failure: the error rides along as a warning
        out = await aconvert(source, output_dir=tmp_path / "out", config=config)
        warnings = " ".join(out.warnings)
        assert "BadRequestError" in warnings
        _assert_clean(warnings)
        # on_failure "fail": the error is raised
        config.llm.on_failure = "fail"
        with pytest.raises(Exception) as caught:
            await aconvert(source, output_dir=tmp_path / "out2", config=config)
        assert "BadRequestError" in str(caught.value)
        _assert_clean(str(caught.value))


class TestCliRedaction:
    """``--json``, stderr, the log file and batch reports, end to end."""

    @pytest.mark.parametrize("mode", ["key", "quote"])
    def test_cli_outputs(self, provider: str, tmp_path: Path, mode: str) -> None:
        inputs = tmp_path / "in"
        inputs.mkdir()
        (inputs / "note.md").write_text(DOCUMENT, encoding="utf-8")
        logs = tmp_path / "logs"
        config = {
            "llm": {
                "enabled": True,
                "on_failure": "fail",
                "model_list": [
                    {
                        "model_name": "default",
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini",
                            "api_key": KEY,
                            "api_base": f"{provider}/{mode}/v1",
                        },
                    }
                ],
                "router_settings": {"num_retries": 0},
            },
            "cache": {"enabled": False},
            "log": {"dir": str(logs), "level": "DEBUG"},
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        env = {**os.environ, "NO_COLOR": "1", "PYTHON_DOTENV_DISABLED": "1"}

        def run(*args: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sys.executable, "-m", "markitai", *args, "-c", str(config_path)],
                capture_output=True,
                text=True,
                env=env,
                cwd=tmp_path,
                timeout=180,
            )

        single = run(str(inputs / "note.md"), "--json", "-o", str(tmp_path / "o1"))
        batch = run(str(inputs), "-o", str(tmp_path / "o2"))

        item = json.loads(single.stdout)["items"][0]
        assert item["status"] == "failed"
        expected = (
            "rejected the api_key configured for it in llm.model_list with HTTP 401"
            if mode == "key"
            else "BadRequestError"
        )
        assert expected in single.stderr
        written = [
            path.read_text(encoding="utf-8", errors="replace")
            for root in (logs, tmp_path / "o1", tmp_path / "o2")
            for path in root.rglob("*")
            if path.is_file() and path.suffix in {".log", ".json"}
        ]
        assert any(path.suffix == ".log" for path in logs.rglob("*"))
        assert any(".markitai" in str(p) for p in (tmp_path / "o2").rglob("*.json"))
        for text in (
            single.stdout,
            single.stderr,
            batch.stdout,
            batch.stderr,
            *written,
        ):
            _assert_clean(text)


class TestHelpers:
    def test_registered_value_and_key_patterns(self) -> None:
        register_secret(KEY)
        text = (
            f"key {KEY}; {PREFIXED_KEY}; Authorization: Bearer abcdefgh12345678; "
            "x-api-key=hunter2hunter2; starred sk-abc*****************wxyz"
        )
        redacted = redact_secrets(text)
        assert KEY not in redacted
        assert PREFIXED_KEY not in redacted
        assert "abcdefgh12345678" not in redacted
        assert "hunter2hunter2" not in redacted
        assert "sk-abc" not in redacted
        assert "Bearer [REDACTED]" in redacted

    def test_env_reference_and_short_values_stay(self) -> None:
        register_secret("short")
        assert redact_secrets("api_key: env:OPENAI_API_KEY short") == (
            "api_key: env:OPENAI_API_KEY short"
        )

    def test_provider_env_key_is_masked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A key LiteLLM reads from the environment by itself."""
        env_key = "unprefixed" + "Zz9" * 8
        monkeypatch.setenv("DEEPSEEK_API_KEY", env_key)
        assert redact_secrets(f"bad key {env_key}") == f"bad key {REDACTED}"

    def test_log_patcher_masks_keys(self) -> None:
        register_secret(KEY)
        record: dict[str, Any] = {"message": f"LiteLLM said {KEY}"}
        redact_log_record(record)
        assert record["message"] == f"LiteLLM said {REDACTED}"

    def test_format_error_message_masks_keys(self) -> None:
        register_secret(KEY)
        assert KEY not in format_error_message(RuntimeError(f"bad key {KEY}"))

    def test_quotations_and_apostrophes(self) -> None:
        long_quote = "x" * 41
        assert redact_provider_message(f"near '{long_quote}' here") == ("near '…' here")
        # Python's repr escapes an inner quote; it does not end the quotation
        escaped = repr('it\'s "quoted" ' + "z" * 40)
        assert redact_provider_message(f"near {escaped}") == "near '…'"
        kept = "Invalid 'max_tokens': can't exceed \"4096\" for 'gpt-4o-mini'."
        assert redact_provider_message(kept) == kept

    def test_embedded_json_body_keeps_its_message(self) -> None:
        """LiteLLM often embeds the raw body; its strings are not quotations."""
        body = json.dumps(
            {
                "error": {
                    "message": "Thinking mode does not support this tool_choice "
                    f"(near {DOC_TEXT!r})",
                    "type": "invalid_request_error",
                }
            }
        )
        assert redact_provider_message(f"DeepseekException - {body}") == (
            "DeepseekException - invalid_request_error: Thinking mode does not "
            "support this tool_choice (near '…')"
        )

    def test_message_cap_and_url_secrets(self) -> None:
        message = redact_provider_message(
            "see https://user:pw@api.example.test/v1?key=abc " + "y " * 400
        )
        assert "user:pw" not in message and "key=abc" not in message
        assert len(message) <= 501

    def test_exception_cleaned_in_place_with_its_cause(self) -> None:
        cause = ValueError(f"upstream quoted {DOC_TEXT!r}")
        error = RuntimeError(f"wrapper: {DOCUMENT}")
        error.__cause__ = cause
        redact_provider_exception(
            error, [{"role": "user", "content": [{"type": "text", "text": DOCUMENT}]}]
        )
        assert str(error) == WITHHELD
        assert str(cause) == "upstream quoted '…'"

    def test_batch_error_lines(self, tmp_path: Path) -> None:
        register_secret(KEY)
        output = tmp_path / "out.jsonl"
        lines = [
            {
                "custom_id": "a",
                "error": {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"messages.0: {DOCUMENT!r} is invalid",
                    },
                },
            },
            {"custom_id": "b", "error": {"code": "401", "message": f"bad {KEY}"}},
        ]
        output.write_text("\n".join(json.dumps(line) for line in lines))
        errors = [line.error for line in read_openai_batch_output(output)]
        assert errors == [
            "invalid_request_error: messages.0: '…' is invalid",
            f"401: bad {REDACTED}",
        ]
