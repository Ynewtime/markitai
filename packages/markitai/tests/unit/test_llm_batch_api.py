"""Tests for llm/batch_api.py — offline Batch API support (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import instructor
from pydantic import BaseModel

from markitai.llm.batch_api import (
    build_anthropic_batch_request,
    build_openai_batch_request,
    parse_batch_result,
    read_openai_batch_output,
    write_batch_jsonl,
)
from markitai.llm.structured import instructor_mode_for_model


class _Doc(BaseModel):
    cleaned_markdown: str
    summary: str


MESSAGES = [
    {"role": "system", "content": "You process documents."},
    {"role": "user", "content": "Process this: hello"},
]


class TestBuildRequest:
    def test_tools_mode_body(self) -> None:
        req = build_openai_batch_request(
            "doc1::main",
            messages=MESSAGES,
            response_model=_Doc,
            model="gpt-5.6-luna",
            mode=instructor.Mode.TOOLS,
        )
        assert req["custom_id"] == "doc1::main"
        assert req["method"] == "POST"
        assert req["url"] == "/v1/chat/completions"
        body = req["body"]
        assert body["model"] == "gpt-5.6-luna"
        tools = body["tools"]
        assert tools[0]["type"] == "function"
        assert "cleaned_markdown" in tools[0]["function"]["parameters"]["properties"]
        assert body["tool_choice"] is not None
        # Instructor must not mutate the caller's messages (MD_JSON appends)
        assert MESSAGES[0]["content"] == "You process documents."

    def test_json_schema_mode_body(self) -> None:
        req = build_openai_batch_request(
            "doc1",
            messages=MESSAGES,
            response_model=_Doc,
            model="gpt-5.6-luna",
            mode=instructor.Mode.JSON_SCHEMA,
        )
        rf = req["body"]["response_format"]
        assert rf["type"] == "json_schema"
        assert "cleaned_markdown" in rf["json_schema"]["schema"]["properties"]

    def test_md_json_mode_appends_schema_to_system(self) -> None:
        req = build_openai_batch_request(
            "doc1",
            messages=MESSAGES,
            response_model=_Doc,
            model="gpt-5.6-luna",
            mode=instructor.Mode.MD_JSON,
        )
        system = req["body"]["messages"][0]["content"]
        assert "cleaned_markdown" in system  # schema was appended
        assert "tools" not in req["body"]

    def test_tools_mode_turns_reasoning_off(self) -> None:
        """Batch deployments reject function tools unless reasoning is off."""
        req = build_openai_batch_request(
            "doc1",
            messages=MESSAGES,
            response_model=_Doc,
            model="gpt-5.6-luna",
            mode=instructor.Mode.TOOLS,
        )
        assert req["body"]["reasoning_effort"] == "none"

    def test_reasoning_untouched_without_tools(self) -> None:
        req = build_openai_batch_request(
            "doc1",
            messages=MESSAGES,
            response_model=_Doc,
            model="gpt-5.6-luna",
            mode=instructor.Mode.JSON_SCHEMA,
        )
        assert "reasoning_effort" not in req["body"]

    def test_reasoning_untouched_for_non_reasoning_model(self) -> None:
        req = build_openai_batch_request(
            "doc1",
            messages=MESSAGES,
            response_model=_Doc,
            model="gpt-4o",
            mode=instructor.Mode.TOOLS,
        )
        assert "reasoning_effort" not in req["body"]

    def test_max_tokens_forwarded(self) -> None:
        req = build_openai_batch_request(
            "d",
            messages=MESSAGES,
            response_model=_Doc,
            model="m",
            mode=instructor.Mode.TOOLS,
            max_tokens=4096,
        )
        assert req["body"]["max_tokens"] == 4096


ANTHROPIC_BODY = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "_Doc",
            "input": {"cleaned_markdown": "clean", "summary": "short"},
        }
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 1979, "output_tokens": 96},
}


class TestParseAnthropicResult:
    def test_a_message_body_parses_through_the_anthropic_mode(self) -> None:
        """The OpenAI envelope cannot hold an Anthropic Message."""
        result = parse_batch_result(
            ANTHROPIC_BODY,
            response_model=_Doc,
            mode=instructor.Mode.TOOLS,  # what the live ladder picked
            provider="anthropic",
        )

        assert result.cleaned_markdown == "clean"
        assert result.summary == "short"


class TestBuildAnthropicRequest:
    """Anthropic's Messages API is not the OpenAI shape."""

    def _request(self) -> dict:
        return build_anthropic_batch_request(
            "doc_0_note_md",
            messages=MESSAGES,
            response_model=_Doc,
            model="claude-haiku-4-5",
            max_tokens=8192,
        )

    def test_system_prompt_leaves_the_message_list(self) -> None:
        params = self._request()["params"]

        assert params["system"] == [{"type": "text", "text": MESSAGES[0]["content"]}]
        assert [m["role"] for m in params["messages"]] == ["user"]
        # Instructor must not mutate the caller's messages
        assert MESSAGES[0]["role"] == "system"

    def test_tools_use_anthropic_spelling(self) -> None:
        params = self._request()["params"]

        (tool,) = params["tools"]
        assert "input_schema" in tool  # not OpenAI's function.parameters
        assert "cleaned_markdown" in tool["input_schema"]["properties"]
        # instructor >=1.17 also sets disable_parallel_tool_use; the forced
        # single-tool shape is the only part that is our contract
        assert params["tool_choice"]["type"] == "tool"
        assert params["tool_choice"]["name"] == "_Doc"

    def test_max_tokens_is_always_sent(self) -> None:
        """The Messages API has no server-side default to fall back on."""
        assert self._request()["params"]["max_tokens"] == 8192

    def test_custom_id_is_carried_beside_the_params(self) -> None:
        request = self._request()

        assert request["custom_id"] == "doc_0_note_md"
        assert set(request) == {"custom_id", "params"}


class TestParseOutput:
    def _write_output(self, tmp_path: Path, lines: list[dict]) -> Path:
        path = tmp_path / "out.jsonl"
        path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
        return path

    def _completion_body(self, text: str) -> dict:
        return {
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5.6-luna",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "_Doc", "arguments": text},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def test_roundtrip_tools(self, tmp_path: Path) -> None:
        payload = json.dumps({"cleaned_markdown": "# Clean", "summary": "s"})
        path = self._write_output(
            tmp_path,
            [
                {
                    "custom_id": "doc1::main",
                    "response": {
                        "status_code": 200,
                        "body": self._completion_body(payload),
                    },
                    "error": None,
                }
            ],
        )
        (line,) = list(read_openai_batch_output(path))
        assert line.custom_id == "doc1::main"
        assert line.error is None
        assert line.body is not None
        result = parse_batch_result(
            line.body, response_model=_Doc, mode=instructor.Mode.TOOLS
        )
        assert result.cleaned_markdown == "# Clean"
        assert result.summary == "s"

    def test_error_line(self, tmp_path: Path) -> None:
        path = self._write_output(
            tmp_path,
            [
                {
                    "custom_id": "doc2",
                    "response": {"status_code": 429, "body": None},
                    "error": {"message": "rate limited"},
                }
            ],
        )
        (line,) = list(read_openai_batch_output(path))
        assert line.body is None
        assert line.error is not None and "rate limited" in line.error

    def test_write_jsonl(self, tmp_path: Path) -> None:
        reqs = [
            build_openai_batch_request(
                f"d{i}",
                messages=MESSAGES,
                response_model=_Doc,
                model="m",
                mode=instructor.Mode.TOOLS,
            )
            for i in range(3)
        ]
        path = tmp_path / "in.jsonl"
        write_batch_jsonl(reqs, path)
        loaded = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["custom_id"] for r in loaded] == ["d0", "d1", "d2"]


class TestModeSelection:
    def test_preselected_modes(self) -> None:
        assert instructor_mode_for_model("openai/gpt-5.6-luna") == instructor.Mode.TOOLS
        assert (
            instructor_mode_for_model("claude-agent/sonnet")
            == instructor.Mode.JSON_SCHEMA
        )
        # Unknown models land on the bottom rung rather than assuming capability
        assert (
            instructor_mode_for_model("some-unknown/model-xyz")
            == instructor.Mode.MD_JSON
        )
