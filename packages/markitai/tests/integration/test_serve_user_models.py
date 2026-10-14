"""Contract v1.15 end to end: users' conversions route over their own pool.

Two user tokens and the admin on one server; real conversions with LLM
enhancement reach local OpenAI-compatible model servers that record the
``Authorization`` each request carried. The admin's model (key
``admin-fixture-key``) is on loopback; user A's own model (key
``user-a-fixture-key``) on this machine's non-loopback address (a user may
not aim a model at the server's own machine). The server's environment holds
``OPENAI_API_KEY`` and an OpenAI base pointing at a sentinel server: neither
may ever serve a user. Everything is invented and nothing leaves the machine.
"""

from __future__ import annotations

import asyncio
import json
import re
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("fastapi")

from markitai.config import MarkitaiConfig
from markitai.serve import create_app
from tests.local_http import LocalHTTPServer

ADMIN_TOKEN = "admin-token"  # nosec B105 - a test token
ADMIN = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
ADMIN_KEY = "admin-fixture-key"
USER_A_KEY = "user-a-fixture-key"
USER_B_KEY = "user-b-fixture-key"
SENTINEL_KEY = "env-sentinel-not-for-users"
NOTES = (
    b"# Garden notes\n\nThe tomatoes along the south fence ripened a week early "
    b"this year.\n\n- Water before nine\n- Mulch the beans\n"
)

_DOC = re.compile(r"<document>\n?(.*?)\n?</document>", re.DOTALL)


def _texts(messages: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out.extend(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return out


def _synth(name: str, schema: dict[str, Any], defs: dict[str, Any], doc: str) -> Any:
    """A value of *schema*: the document for Markdown fields."""
    if "$ref" in schema:
        return _synth(name, defs.get(schema["$ref"].split("/")[-1], {}), defs, doc)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            options = [s for s in schema[key] if s.get("type") != "null"]
            return _synth(name, (options or schema[key])[0], defs, doc)
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    if kind == "object" or "properties" in schema:
        return {
            key: _synth(key, value, defs, doc)
            for key, value in schema.get("properties", {}).items()
        }
    if kind == "array":
        return ["garden", "notes", "summer"] if name.lower() == "tags" else []
    if kind in ("integer", "number"):
        return 1
    if kind == "boolean":
        return False
    lowered = name.lower()
    if "markdown" in lowered or lowered in ("content", "text", "cleaned"):
        return doc
    return "Garden notes" if "title" in lowered else "A note about a garden."


def _answer(body: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]] | None]:
    texts = _texts(body.get("messages") or [])
    doc = next((m.group(1) for t in texts if (m := _DOC.search(t))), texts[-1])
    tools = body.get("tools") or []
    if tools:
        function = tools[0].get("function", {})
        params = function.get("parameters", {})
        arguments = _synth("result", params, params.get("$defs", {}), doc)
        return None, [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": function.get("name", "result"),
                    "arguments": json.dumps(arguments),
                },
            }
        ]
    fmt = body.get("response_format") or {}
    if fmt.get("type") == "json_schema":
        schema = fmt.get("json_schema", {}).get("schema", {})
        return json.dumps(_synth("result", schema, schema.get("$defs", {}), doc)), None
    for text in texts:
        at = text.find("json_schema:")
        if at >= 0:
            schema, _ = json.JSONDecoder().raw_decode(text[text.find("{", at) :])
            value = json.dumps(_synth("result", schema, schema.get("$defs", {}), doc))
            return f"```json\n{value}\n```", None
    return doc, None


class _Model(BaseHTTPRequestHandler):
    """An OpenAI-compatible chat model that records each request's key."""

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        raw = self.rfile.read(int(self.headers.get("content-length") or 0))
        self.server.keys.append(self.headers.get("authorization") or "")  # type: ignore[attr-defined]
        body = json.loads(raw or b"{}")
        content, calls = _answer(body)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        data = json.dumps(
            {
                "id": "c",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if calls else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 20,
                    "total_tokens": 60,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _Endpoint:
    def __init__(self, host: str) -> None:
        self.server = LocalHTTPServer((host, 0), _Model)
        self.server.keys = []  # type: ignore[attr-defined]
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def received(self) -> list[str]:
        """The keys of the requests since the last call."""
        keys: list[str] = self.server.keys  # type: ignore[attr-defined]
        taken = list(keys)
        keys.clear()
        return taken

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _lan_address() -> str | None:
    """This machine's address other than loopback, if it has one."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 80))
            address = probe.getsockname()[0]
    except OSError:
        return None
    return None if address.startswith(("127.", "0.")) else address


@pytest.fixture
def endpoints(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    sentinel = _Endpoint("127.0.0.1")
    admin = _Endpoint("127.0.0.1")
    lan = _lan_address()
    own = _Endpoint("0.0.0.0") if lan else None  # noqa: S104 - a test server
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(name, raising=False)
    # The server's own key and default address: never a user's
    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL_KEY)
    for name in ("OPENAI_API_BASE", "OPENAI_BASE_URL"):
        monkeypatch.setenv(name, f"http://127.0.0.1:{sentinel.port}/v1")
    try:
        yield {"sentinel": sentinel, "admin": admin, "own": own, "lan": lan}
    finally:
        for endpoint in (sentinel, admin, own):
            if endpoint is not None:
                endpoint.stop()


def _write_config(tmp_path: Path, admin_port: int) -> MarkitaiConfig:
    base = f"http://127.0.0.1:{admin_port}/v1"
    document = {
        "llm": {
            "model_list": [
                {
                    "model_name": "default",
                    "litellm_params": {
                        "model": "openai/fake-model",
                        "api_key": ADMIN_KEY,
                        "api_base": base,
                    },
                    "model_info": {
                        "id": "shared-model-1",
                        "provider_id": "admin-connection-1",
                    },
                }
            ],
            "providers": [
                {
                    "id": "admin-connection-1",
                    "provider": "custom",
                    "api_key": ADMIN_KEY,
                    "api_base": base,
                }
            ],
            "router_settings": {"num_retries": 0, "timeout": 20},
        },
        "cache": {"enabled": False, "global_dir": str(tmp_path / "cache")},
    }
    (tmp_path / "config.json").write_text(json.dumps(document), encoding="utf-8")
    return MarkitaiConfig.model_validate(document)


@asynccontextmanager
async def _server(tmp_path: Path, admin_port: int) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        config=_write_config(tmp_path, admin_port)
        if not (tmp_path / "config.json").exists()
        else MarkitaiConfig.model_validate(
            json.loads((tmp_path / "config.json").read_text())
        ),
        configure_logging=False,
        config_path=tmp_path / "config.json",
        token=ADMIN_TOKEN,
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:3600", timeout=60
        ) as client:
            yield client


async def _user(client: httpx.AsyncClient, label: str) -> tuple[str, dict[str, str]]:
    created = (
        await client.post("/api/access-tokens", json={"label": label}, headers=ADMIN)
    ).json()
    return created["id"], {"Authorization": f"Bearer {created['token']}"}


async def _submit(
    client: httpx.AsyncClient, headers: dict[str, str], *, llm: bool
) -> httpx.Response:
    return await client.post(
        "/api/jobs",
        files=[
            ("files", ("garden-notes.md", NOTES, "text/markdown")),
            ("options", (None, json.dumps({"llm": True} if llm else {}))),
        ],
        headers=headers,
    )


async def _done(
    client: httpx.AsyncClient, headers: dict[str, str], job_id: str
) -> dict[str, Any]:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        job = (await client.get(f"/api/jobs/{job_id}", headers=headers)).json()
        if job["status"] == "done":
            return job
        await asyncio.sleep(0.1)
    pytest.fail(f"job {job_id} did not finish")


async def _converted(
    client: httpx.AsyncClient, headers: dict[str, str]
) -> dict[str, Any]:
    created = await _submit(client, headers, llm=True)
    assert created.status_code == 201, created.text
    job = await _done(client, headers, created.json()["job_id"])
    item = job["items"][0]
    assert item["status"] == "done", job
    assert item["llm_enhanced"] is True, job
    return job


async def _revision(
    client: httpx.AsyncClient, path: str, headers: dict[str, str]
) -> str:
    return (await client.get(path, headers=headers)).json()["revision"]


async def _share(client: httpx.AsyncClient, value: str) -> dict[str, Any]:
    response = await client.patch(
        "/api/settings/llm/deployments/shared-model-1",
        json={
            "expected_revision": await _revision(client, "/api/settings/llm", ADMIN),
            "share": value,
        },
        headers=ADMIN,
    )
    assert response.status_code == 200, response.text
    return response.json()


async def test_shared_and_own_models_serve_users_and_never_the_environment(
    tmp_path: Path, endpoints: dict[str, Any]
) -> None:
    sentinel, admin_model = endpoints["sentinel"], endpoints["admin"]
    own_model, lan = endpoints["own"], endpoints["lan"]
    base = f"http://127.0.0.1:{admin_model.port}/v1"
    async with _server(tmp_path, admin_model.port) as client:
        a_id, a = await _user(client, "Reading group laptop")
        _b_id, b = await _user(client, "Studio tablet")
        admin_job = await _converted(client, ADMIN)
        assert set(admin_model.received()) == {f"Bearer {ADMIN_KEY}"}

        # B has nothing: no LLM job; a plain conversion sends no model request
        capabilities = (await client.get("/api/capabilities", headers=b)).json()
        assert capabilities["llm"] == {
            "configured": False,
            "routable": False,
            "effective": False,
            "models": [],
        }
        refused = await _submit(client, b, llm=True)
        assert (refused.status_code, refused.json()["reason"]) == (
            422,
            "llm_unavailable",
        )
        plain = await _submit(client, b, llm=False)
        plain_job = await _done(client, b, plain.json()["job_id"])
        assert plain_job["items"][0]["status"] == "done"
        assert plain_job["items"][0]["llm_enhanced"] is False
        assert admin_model.received() == []

        # Shared: B's next conversion uses it and sees only what it is
        await _share(client, "shared")
        view = (await client.get("/api/me/models", headers=b)).json()
        assert view["shared"] == [
            {
                "id": "shared-model-1",
                "model": "openai/fake-model",
                "provider": "custom",
                "weight": 1,
            }
        ]
        user_job = await _converted(client, b)
        keys = admin_model.received()
        assert keys and set(keys) == {f"Bearer {ADMIN_KEY}"}
        mine, theirs = user_job["items"][0], admin_job["items"][0]
        assert mine["models"] == theirs["models"] == ["fake-model"]
        assert mine["pricing"] == theirs["pricing"]
        for answer in (json.dumps(view), json.dumps(user_job)):
            for hidden in (ADMIN_KEY, base, f":{admin_model.port}", "routing_group"):
                assert hidden not in answer, hidden

        # Private again: at once nothing for B
        await _share(client, "private")
        refused = await _submit(client, b, llm=True)
        assert refused.json()["reason"] == "llm_unavailable"

        # A's own model with the shared one: A's conversions reach only those
        if own_model is not None:
            added = await client.post(
                "/api/me/models/deployments/batch",
                json={
                    "expected_revision": await _revision(client, "/api/me/models", a),
                    "deployments": [
                        {
                            "model_name": "default",
                            "provider": "custom",
                            "model": "openai/fake-model",
                            "api_key": USER_A_KEY,
                            "api_base": f"http://{lan}:{own_model.port}/v1",
                        }
                    ],
                },
                headers=a,
            )
            assert added.status_code == 200, added.text
            own_id = added.json()["deployments"][0]["deployment_id"]
            tested = await client.post(
                "/api/me/models/test", json={"deployment_id": own_id}, headers=a
            )
            assert tested.json()["ok"] is True, tested.text
            await _converted(client, a)
            assert set(own_model.received()) == {f"Bearer {USER_A_KEY}"}
            assert admin_model.received() == []
            # Both on: each request goes to one of the two, by weight
            await _share(client, "shared")
            for _ in range(3):
                await _converted(client, a)
            assert set(own_model.received()) | set(admin_model.received()) <= {
                f"Bearer {USER_A_KEY}",
                f"Bearer {ADMIN_KEY}",
            }
            # Own model off: the shared one only
            await client.patch(
                f"/api/me/models/deployments/{own_id}",
                json={
                    "expected_revision": await _revision(client, "/api/me/models", a),
                    "weight": 0,
                },
                headers=a,
            )
            a_job = await _converted(client, a)
            assert own_model.received() == []
            assert set(admin_model.received()) == {f"Bearer {ADMIN_KEY}"}
        else:
            await _share(client, "shared")
            added = await client.post(
                "/api/me/models/deployments/batch",
                json={
                    "expected_revision": await _revision(client, "/api/me/models", a),
                    "deployments": [
                        {
                            "model_name": "default",
                            "model": "openai/own-model",
                            "api_key": USER_A_KEY,
                            "weight": 0,
                        }
                    ],
                },
                headers=a,
            )
            assert added.status_code == 200, added.text
            a_job = await _converted(client, a)
            admin_model.received()

        # Revoking A deletes its models; its job reruns over the shared ones
        tokens = (await client.get("/api/access-tokens", headers=ADMIN)).json()
        assert {t["label"]: t["model_count"] for t in tokens["tokens"]} == {
            "Reading group laptop": 1,
            "Studio tablet": 0,
        }
        revoked = await client.delete(f"/api/access-tokens/{a_id}", headers=ADMIN)
        assert revoked.status_code == 204
        assert not (tmp_path / "user-models" / f"{a_id}.json").exists()
        job_id, item_id = a_job["job_id"], a_job["items"][0]["item_id"]
        retried = await client.post(
            f"/api/jobs/{job_id}/items/{item_id}/retry",
            json={"options": {"llm": True}},
            headers=ADMIN,
        )
        assert retried.status_code == 202, retried.text
        job = await _done(client, ADMIN, job_id)
        assert job["items"][0]["llm_enhanced"] is True, job
        assert set(admin_model.received()) == {f"Bearer {ADMIN_KEY}"}
        if own_model is not None:
            assert own_model.received() == []

        # A shared model that fails: the user's job never names its host
        admin_model.stop()
        failed = await _submit(client, b, llm=True)
        failed_job = await _done(client, b, failed.json()["job_id"])
        text = json.dumps(failed_job)
        assert failed_job["items"][0]["llm_enhanced"] is False
        assert f"127.0.0.1:{admin_model.port}" not in text, text
    assert sentinel.received() == []
