"""Workbench API contract v1.13: what Python 1.2.1 adds for the 1bc3d25 bundle.

- Several conversions deleted at once (§6.8, ``features.batch_delete``):
  ``POST /api/history/delete``, each entry exactly as its single route.
- The framed web page shares the workbench's origin (§7) and still runs
  nothing: ``sandbox allow-same-origin allow-popups
  allow-popups-to-escape-sandbox`` and no script source.
- Access levels (§2, ``features.user_tokens``): user tokens created by the
  admin, who see and touch only their own conversions and never the
  settings; only SHA-256 digests are kept, owner-only, in the serve store.
- Edit config (§8, ``features.config_edit``): the file's own text with its
  secrets masked, saved exactly as sent with kept masks restored in place.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("fastapi")

from markitai.serve import config_editor
from markitai.serve.access import MAX_TOKENS, AccessTokens, clean_label
from tests.unit.serve.test_serve_item_files import _pdf_like_converter
from tests.unit.serve.test_workbench_contract import check_shape
from tests.unit.serve.test_workbench_v111 import (
    URL,
    _allow_every_target,
    _client,
    _FakeUrlConversion,
    _form,
    _job,
    _make_app,
    _settled,
)

if TYPE_CHECKING:
    import httpx
    from fastapi import FastAPI

ADMIN = {"Authorization": "Bearer admin-token"}
REMOTE = ("192.168.1.30", 40000)


@pytest.fixture(autouse=True)
def _converter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("markitai.serve.jobs.process_file_item", _pdf_like_converter())


@pytest.fixture
def web_page(monkeypatch: pytest.MonkeyPatch) -> _FakeUrlConversion:
    """A URL conversion that fetches nothing and keeps the page it is given."""
    fake = _FakeUrlConversion()
    monkeypatch.setattr("markitai.serve.jobs.process_url_item", fake)
    monkeypatch.setattr("markitai.serve.app._guard_fetch_targets", _allow_every_target)
    return fake


def _reason(response: httpx.Response) -> tuple[int, str]:
    return response.status_code, response.json()["reason"]


# ---------------------------------------------------------------- 6.8 delete


async def _jobs_with_items(client: httpx.AsyncClient) -> tuple[str, str]:
    """A one-item job and a two-item job."""
    single = await _job(client, files=[("one.pdf", b"%")])
    double = await _job(client, files=[("a.pdf", b"%"), ("b.pdf", b"%")])
    return single, double


class TestBatchDelete:
    async def test_entries_run_in_order_as_their_single_routes(
        self, tmp_path: Path
    ) -> None:
        async with _client(_make_app(tmp_path)) as client:
            single, double = await _jobs_with_items(client)
            answer = await client.post(
                "/api/history/delete",
                json={
                    "jobs": [single, "000000000000"],
                    "items": [
                        {"job_id": double, "item_id": "i1"},
                        {"job_id": double, "item_id": "i9"},
                        {"job_id": double, "item_id": "i2"},
                        {"job_id": single, "item_id": "i1"},
                    ],
                },
            )
            history = (await client.get("/api/history")).json()
        assert answer.status_code == 200
        assert answer.headers["cache-control"] == "no-store"
        body = answer.json()
        check_shape("BatchDeleteResult", body)
        assert (body["deleted"], body["failed"]) == (3, 3)
        assert [
            (r["job_id"], r["item_id"], r["ok"], r.get("reason"))
            for r in body["results"]
        ] == [
            (single, None, True, None),
            ("000000000000", None, False, "job_not_found"),
            (double, "i1", True, None),
            (double, "i9", False, "item_not_found"),
            # The job's last item deletes the job
            (double, "i2", True, None),
            (single, "i1", False, "job_not_found"),
        ]
        assert body["results"][1]["detail"] == "job not found"
        assert history == []
        assert not (tmp_path / "jobs" / single).exists()
        assert not (tmp_path / "jobs" / double).exists()

    async def test_a_running_job_is_refused_as_its_route_would(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        release = asyncio.Event()

        async def slow(file_path: Path, cfg: Any, out_dir: Path, shared: Any) -> Any:
            await release.wait()
            return await _pdf_like_converter()(file_path, cfg, out_dir, shared)

        async with _client(_make_app(tmp_path)) as client:
            done = await _job(client, files=[("one.pdf", b"%")])
            monkeypatch.setattr("markitai.serve.jobs.process_file_item", slow)
            created = await client.post(
                "/api/jobs", files=_form(files=[("b.pdf", b"%")])
            )
            running = created.json()["job_id"]
            single = await client.delete(f"/api/history/{running}")
            batch = await client.post(
                "/api/history/delete",
                json={
                    "jobs": [running, done],
                    "items": [{"job_id": running, "item_id": "i1"}],
                },
            )
            release.set()
            await _settled(client, running)
        assert _reason(single) == (409, "job_running")
        results = batch.json()["results"]
        assert [(r["ok"], r.get("reason")) for r in results] == [
            (False, "job_running"),
            (True, None),
            (False, "job_running"),
        ]
        assert results[0]["detail"] == single.json()["detail"]

    @pytest.mark.parametrize(
        "body",
        [
            [],
            {},
            {"jobs": []},
            {"jobs": [], "items": []},
            {"jobs": "abc"},
            {"jobs": [1]},
            {"items": [{"job_id": "a"}]},
            {"items": [{"job_id": "a", "item_id": "i1", "x": 1}]},
            {"items": [{"job_id": "a", "item_id": 1}]},
            {"items": ["a"]},
            {"jobs": ["a"], "other": 1},
            {"jobs": ["a"] * 600, "items": [{"job_id": "a", "item_id": "b"}] * 401},
        ],
        ids=[
            "array",
            "empty",
            "no-entries",
            "two-empty",
            "jobs-not-array",
            "job-not-string",
            "item-without-id",
            "item-extra-key",
            "item-id-not-string",
            "item-not-object",
            "unknown-key",
            "over-1000",
        ],
    )
    async def test_other_bodies_are_refused(self, tmp_path: Path, body: Any) -> None:
        async with _client(_make_app(tmp_path)) as client:
            refused = await client.post("/api/history/delete", json=body)
        assert _reason(refused) == (422, "invalid_delete_request")

    async def test_a_thousand_entries_are_accepted(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            answer = await client.post(
                "/api/history/delete", json={"jobs": ["000000000000"] * 1000}
            )
        assert answer.status_code == 200
        assert answer.json()["failed"] == 1000

    async def test_not_json_and_too_large(self, tmp_path: Path) -> None:
        async with _client(_make_app(tmp_path)) as client:
            form = await client.post("/api/history/delete", data={"jobs": "a"})
            broken = await client.post(
                "/api/history/delete",
                content=b"{",
                headers={"Content-Type": "application/json"},
            )
            large = await client.post(
                "/api/history/delete",
                content=b'{"jobs": ["' + b"a" * (256 * 1024) + b'"]}',
                headers={"Content-Type": "application/json"},
            )
        assert _reason(form) == (422, "json_required")
        assert _reason(broken) == (422, "invalid_json")
        assert _reason(large) == (413, "request_too_large")


# --------------------------------------------------------------- 7 the frame

#: Everything a hostile page could try (contract v1.13 §7): none may run or
#: request anything under the view policy.
HOSTILE_PAGE = (
    "<!doctype html><html><head><title>Hostile</title>"
    '<meta http-equiv="refresh" content="0;url=https://attacker.test/refresh">'
    "<script>parent.document.title='script ran'</script>"
    '<script src="https://attacker.test/a.js"></script>'
    "</head>"
    "<body onload=\"parent.document.title='onload ran'\">"
    "<h1>Hostile page</h1>"
    '<img src="x" onerror="parent.document.title=\'onerror ran\'">'
    '<a id="js" href="javascript:parent.document.title=\'link ran\'">link</a>'
    '<a id="jsb" target="_blank" href="javascript:alert(1)">new tab</a>'
    '<svg><a href="javascript:alert(2)"><text y="20">svg</text></a></svg>'
    '<form action="https://attacker.test/post" method="post">'
    '<input name="q" value="1"><button>send</button></form>'
    '<form action="https://attacker.test/blank" target="_blank"><button>'
    "send</button></form>"
    '<iframe src="https://attacker.test/frame"></iframe>'
    '<object data="https://attacker.test/object"></object>'
    "</body></html>"
)


class TestFramedPagePolicy:
    def test_the_policy_is_the_contracts_and_stays_script_less(self) -> None:
        from markitai.serve.originals import VIEW_POLICY

        assert VIEW_POLICY == (
            "sandbox allow-same-origin allow-popups allow-popups-to-escape-sandbox; "
            "default-src 'none'; img-src https: data:; style-src https: "
            "'unsafe-inline'; font-src https: data:; media-src https:; "
            "form-action 'none'; frame-ancestors 'self'"
        )
        directives = {
            name: values
            for name, *values in (part.split() for part in VIEW_POLICY.split(";"))
        }
        assert not {"allow-scripts", "allow-forms", "allow-top-navigation"} & set(
            directives["sandbox"]
        )
        assert not any(name.startswith("script-src") for name in directives)

    async def test_a_hostile_kept_page_is_served_verbatim_under_the_policy(
        self, tmp_path: Path, web_page: _FakeUrlConversion
    ) -> None:
        from markitai.serve.originals import VIEW_POLICY

        web_page.html = HOSTILE_PAGE
        async with _client(_make_app(tmp_path)) as client:
            job_id = await _job(client, urls=[URL])
            view = await client.get(f"/api/jobs/{job_id}/items/i1/original/view")
            file = await client.get(f"/api/jobs/{job_id}/items/i1/original/file")
        assert view.status_code == 200
        assert view.headers["content-security-policy"] == VIEW_POLICY
        assert view.headers["content-type"] == "text/html; charset=utf-8"
        assert "x-frame-options" not in view.headers
        # One insertion after <head>, the rest exactly as fetched
        head = HOSTILE_PAGE.index("<head>") + len("<head>")
        inserted = (
            '<meta http-equiv="x-dns-prefetch-control" content="off">'
            f'<base href="{URL.replace("&", "&amp;")}" target="_blank">'
        )
        assert view.text == HOSTILE_PAGE[:head] + inserted + HOSTILE_PAGE[head:]
        # The download is never rendered: plain text, its own empty sandbox
        assert file.headers["content-type"] == "text/plain; charset=utf-8"
        assert file.headers["content-security-policy"] == "sandbox; default-src 'none'"

    def test_no_other_route_grants_the_frame_sandbox(self) -> None:
        source = Path(__file__).parents[3] / "src" / "markitai" / "serve"
        holders = [
            path.name
            for path in source.glob("*.py")
            if "allow-same-origin" in path.read_text(encoding="utf-8")
        ]
        assert holders == ["originals.py"]


# ------------------------------------------------------------ 2 access levels


async def _user(
    client: httpx.AsyncClient, label: str = "Team laptop"
) -> tuple[dict[str, Any], dict[str, str]]:
    created = await client.post(
        "/api/access-tokens", json={"label": label}, headers=ADMIN
    )
    assert created.status_code == 201, created.text
    body = created.json()
    return body, {"Authorization": f"Bearer {body['token']}"}


class TestAccessTokens:
    async def test_create_list_and_revoke(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from loguru import logger

        handler = logger.add(caplog.handler, level="INFO", format="{message}")
        app = _make_app(tmp_path, token="admin-token")
        try:
            async with _client(app, peer=REMOTE) as client:
                created, user = await _user(client, "  Team laptop ")
                listed = await client.get("/api/access-tokens", headers=ADMIN)
                capabilities = (
                    await client.get("/api/capabilities", headers=user)
                ).json()
                used = (await client.get("/api/access-tokens", headers=ADMIN)).json()
                revoked = await client.delete(
                    f"/api/access-tokens/{created['id']}", headers=ADMIN
                )
                after = await client.get("/api/capabilities", headers=user)
                again = await client.delete(
                    f"/api/access-tokens/{created['id']}", headers=ADMIN
                )
                empty = (await client.get("/api/access-tokens", headers=ADMIN)).json()
        finally:
            logger.remove(handler)
        check_shape("CreatedAccessToken", created)
        token = created["token"]
        assert re.fullmatch(r"mku_[0-9a-f]{64}", token)
        assert re.fullmatch(r"u_[0-9a-f]{12}", created["id"])
        assert created["label"] == "Team laptop"
        assert created["hint"] == token[-4:]
        assert created["last_used_at"] is None
        assert listed.status_code == 200
        assert listed.headers["cache-control"] == "no-store"
        check_shape("AccessTokenList", listed.json())
        assert listed.json()["tokens"] == [
            {key: value for key, value in created.items() if key != "token"}
            | {"model_count": 0}
        ]
        assert capabilities["access"] == {"role": "user", "label": "Team laptop"}
        assert capabilities["features"]["user_tokens"] is True
        assert used["tokens"][0]["last_used_at"] is not None
        assert revoked.status_code == 204
        assert revoked.headers["cache-control"] == "no-store"
        assert _reason(after) == (401, "token_required")
        assert _reason(again) == (404, "token_not_found")
        assert empty == {"tokens": []}
        logged = [record.getMessage() for record in caplog.records]
        assert '[Serve] Access token "Team laptop" created from the web workbench.' in (
            logged
        )
        assert '[Serve] Access token "Team laptop" revoked from the web workbench.' in (
            logged
        )
        assert not any(token in line or token[4:] in line for line in logged)

    async def test_only_digests_are_kept_owner_only_and_survive_a_restart(
        self, tmp_path: Path
    ) -> None:
        app = _make_app(tmp_path, token="admin-token")
        async with _client(app) as client:
            created, _ = await _user(client)
        stored = tmp_path / "access.json"
        text = stored.read_text(encoding="utf-8")
        token = created["token"]
        assert token not in text and token[4:] not in text
        saved = json.loads(text)
        assert saved == {
            "version": 1,
            "tokens": [
                {
                    "id": created["id"],
                    "label": "Team laptop",
                    "sha256": hashlib.sha256(token.encode()).hexdigest(),
                    "hint": token[-4:],
                    "created_at": created["created_at"],
                }
            ],
        }
        if sys.platform != "win32":
            assert stored.stat().st_mode & 0o777 == 0o600
        # Not in the configuration file, which the Edit config page shows
        assert not (tmp_path / "config.json").exists()
        # A restart (and a new admin token) keeps the user token working
        restarted = _make_app(tmp_path, token="another-admin-token")
        async with _client(restarted, peer=REMOTE) as client:
            answer = await client.get(
                "/api/capabilities", headers={"Authorization": f"Bearer {token}"}
            )
            listed = await client.get(
                "/api/access-tokens",
                headers={"Authorization": "Bearer another-admin-token"},
            )
        assert answer.json()["access"] == {"role": "user", "label": "Team laptop"}
        # Last use is kept in memory only
        assert listed.json()["tokens"][0]["last_used_at"] is not None

    async def test_an_unreadable_store_is_left_alone(self, tmp_path: Path) -> None:
        (tmp_path / "access.json").write_text("{broken", encoding="utf-8")
        tokens = AccessTokens(tmp_path / "access.json")
        assert tokens.list() == {"tokens": []}
        assert (tmp_path / "access.json").read_text(encoding="utf-8") == "{broken"

    @pytest.mark.parametrize(
        ("content", "status", "reason"),
        [
            (b'{"label": ""}', 422, "invalid_label"),
            (b'{"label": "   "}', 422, "invalid_label"),
            (b'{"label": "a\\nb"}', 422, "invalid_label"),
            (b'{"label": "' + b"a" * 61 + b'"}', 422, "invalid_label"),
            (b'{"label": 1}', 422, "invalid_json"),
            (b'{"label": "a", "x": 1}', 422, "invalid_json"),
            (b"[]", 422, "invalid_json"),
            (b"nope", 422, "invalid_json"),
            (b'{"label": "' + b"a" * 1100 + b'"}', 413, "request_too_large"),
        ],
        ids=[
            "empty",
            "blank",
            "control",
            "61-characters",
            "not-a-string",
            "extra-key",
            "array",
            "not-json",
            "over-1-kib",
        ],
    )
    async def test_refused_bodies(
        self, tmp_path: Path, content: bytes, status: int, reason: str
    ) -> None:
        app = _make_app(tmp_path, token="admin-token")
        async with _client(app, headers=ADMIN) as client:
            refused = await client.post(
                "/api/access-tokens",
                content=content,
                headers={"Content-Type": "application/json"},
            )
        assert _reason(refused) == (status, reason)

    def test_labels(self) -> None:
        assert clean_label("  Team laptop ") == "Team laptop"
        assert clean_label("界" * 60) == "界" * 60
        for refused in ["", "   ", "a\nb", "a\x07b", "a" * 61, 5]:
            assert clean_label(refused) is None

    async def test_at_most_fifty(self, tmp_path: Path) -> None:
        app = _make_app(tmp_path, token="admin-token")
        async with _client(app, headers=ADMIN) as client:
            for index in range(MAX_TOKENS):
                await _user(client, f"Token {index}")
            refused = await client.post(
                "/api/access-tokens", json={"label": "One more"}
            )
        assert _reason(refused) == (409, "token_limit")

    async def test_admin_only(self, tmp_path: Path) -> None:
        tokened = _make_app(tmp_path / "a", token="admin-token")
        async with _client(tokened) as local:
            created, user = await _user(local)
            without = await local.get("/api/access-tokens")
            as_user = [
                await local.get("/api/access-tokens", headers=user),
                await local.post(
                    "/api/access-tokens", json={"label": "x"}, headers=user
                ),
                await local.delete(f"/api/access-tokens/{created['id']}", headers=user),
                await local.post("/api/access-token/regenerate", headers=user),
                await local.get("/api/settings/llm", headers=user),
                await local.get("/api/settings/config", headers=user),
                await local.put(
                    "/api/settings/config",
                    json={"expected_revision": "absent", "text": "{}"},
                    headers=user,
                ),
            ]
        no_auth = _make_app(tmp_path / "b")
        async with _client(no_auth) as local:
            disabled = [
                await local.get("/api/access-tokens"),
                await local.post("/api/access-tokens", json={"label": "x"}),
                await local.delete("/api/access-tokens/u_000000000000"),
            ]
            capabilities = (await local.get("/api/capabilities")).json()
        async with _client(no_auth, peer=REMOTE) as remote:
            untrusted = await remote.get("/api/access-tokens")
            anonymous = (await remote.get("/api/capabilities")).json()
        assert _reason(without) == (401, "token_required")
        for refused in as_user:
            assert _reason(refused) == (403, "settings_forbidden"), refused.url
            assert refused.headers["cache-control"] == "no-store"
        for refused in disabled:
            assert _reason(refused) == (409, "token_disabled")
        assert capabilities["features"]["user_tokens"] is False
        assert capabilities["access"]["role"] == "admin"
        assert _reason(untrusted) == (403, "settings_forbidden")
        # An anonymous caller cannot use the settings either
        assert anonymous["access"]["role"] == "user"


class TestUserSessions:
    async def test_a_user_sees_and_touches_only_its_own_jobs(
        self, tmp_path: Path
    ) -> None:
        app = _make_app(tmp_path, token="admin-token")
        async with _client(app, peer=REMOTE) as client:
            _, user = await _user(client)
            _, other = await _user(client, "Other")
            admins = await _job_as(client, ADMIN)
            mine = await _job_as(client, user)
            others = await _job_as(client, other)
            history = (await client.get("/api/history", headers=user)).json()
            with_items = (
                await client.get("/api/history?include=items", headers=user)
            ).json()
            admin_history = (await client.get("/api/history", headers=ADMIN)).json()
            snapshot = await client.get(f"/api/jobs/{mine}", headers=user)
            archive = await client.get("/api/history/archive", headers=user)
        assert [entry["job_id"] for entry in history] == [mine]
        assert [entry["job_id"] for entry in with_items] == [mine]
        assert {entry["job_id"] for entry in admin_history} == {admins, mine, others}
        assert snapshot.status_code == 200
        assert "owner" not in snapshot.text
        assert all("owner" not in json.dumps(entry) for entry in with_items)
        names = zipfile.ZipFile(io.BytesIO(archive.content)).namelist()
        assert names and all("one.pdf" in name for name in names), names

    async def test_every_job_route_hides_anothers_job(self, tmp_path: Path) -> None:
        app = _make_app(tmp_path, token="admin-token")
        async with _client(app, peer=REMOTE) as client:
            _, user = await _user(client)
            job_id = await _job_as(client, ADMIN)
            answers = {}
            for route in app.routes:
                path = getattr(route, "path", "")
                if "{job_id}" not in path:
                    continue
                for method in sorted(getattr(route, "methods", set()) - {"HEAD"}):
                    url = (
                        path.replace("{job_id}", job_id)
                        .replace("{item_id}", "i1")
                        .replace("{relpath:path}", "x.md")
                        .replace("{page}", "1")
                    )
                    answers[(method, path)] = await client.request(
                        method, url, headers=user
                    )
            own = await client.get(f"/api/jobs/{job_id}", headers=ADMIN)
        # Every one of them: the cancel, retry, delete, snapshot, events,
        # result, assets, files, archives and original routes (v1.14: blocks, v1.17: outline)
        assert len(answers) == 18, sorted(answers)
        for (method, path), answer in answers.items():
            assert _reason(answer) == (404, "job_not_found"), (method, path)
        assert own.status_code == 200

    async def test_batch_delete_and_tickets_are_limited_to_its_own_jobs(
        self, tmp_path: Path
    ) -> None:
        app = _make_app(tmp_path, token="admin-token")
        async with _client(app, peer=REMOTE) as client:
            _, user = await _user(client)
            admins = await _job_as(client, ADMIN)
            mine = await _job_as(client, user)
            refused_ticket = await client.post(
                "/api/download-tickets",
                json={"path": f"/api/jobs/{admins}/archive"},
                headers=user,
            )
            own_ticket = await client.post(
                "/api/download-tickets",
                json={"path": f"/api/jobs/{mine}/archive"},
                headers=user,
            )
            history_ticket = await client.post(
                "/api/download-tickets",
                json={"path": "/api/history/archive"},
                headers=user,
            )
            own_archive = await client.get(own_ticket.json()["url"])
            history_archive = await client.get(history_ticket.json()["url"])
            deleted = await client.post(
                "/api/history/delete",
                json={
                    "jobs": [admins],
                    "items": [
                        {"job_id": admins, "item_id": "i1"},
                        {"job_id": mine, "item_id": "i1"},
                    ],
                },
                headers=user,
            )
            still = await client.get(f"/api/jobs/{admins}", headers=ADMIN)
        assert _reason(refused_ticket) == (404, "job_not_found")
        assert own_archive.status_code == 200
        assert history_archive.status_code == 200
        # The ticket acts as the user that asked: only its own job inside
        assert len(zipfile.ZipFile(io.BytesIO(history_archive.content)).namelist()) == (
            len(zipfile.ZipFile(io.BytesIO(own_archive.content)).namelist())
        )
        assert [(r["ok"], r.get("reason")) for r in deleted.json()["results"]] == [
            (False, "job_not_found"),
            (False, "job_not_found"),
            (True, None),
        ]
        assert still.status_code == 200

    async def test_revoking_ends_requests_tickets_and_streams_at_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        release = asyncio.Event()

        async def slow(file_path: Path, cfg: Any, out_dir: Path, shared: Any) -> Any:
            await release.wait()
            return await _pdf_like_converter()(file_path, cfg, out_dir, shared)

        app = _make_app(tmp_path, token="admin-token")
        # From loopback: a user token is that user even there, and a revoked
        # one is refused there too
        async with _client(app) as client:
            created, user = await _user(client)
            done = await _job_as(client, user)
            ticket = (
                await client.post(
                    "/api/download-tickets",
                    json={"path": f"/api/jobs/{done}/archive"},
                    headers=user,
                )
            ).json()
            monkeypatch.setattr("markitai.serve.jobs.process_file_item", slow)
            running = (
                await client.post(
                    "/api/jobs", files=_form(files=[("b.pdf", b"%")]), headers=user
                )
            ).json()["job_id"]
            stream = asyncio.create_task(
                client.get(f"/api/jobs/{running}/events", headers=user)
            )
            await asyncio.sleep(0.2)
            assert not stream.done()
            capabilities = (await client.get("/api/capabilities", headers=user)).json()
            settings = await client.get("/api/settings/llm", headers=user)
            await client.delete(f"/api/access-tokens/{created['id']}", headers=ADMIN)
            ended = await asyncio.wait_for(stream, timeout=5)
            reconnect = await client.get(f"/api/jobs/{running}/events", headers=user)
            spent = await client.get(ticket["url"])
            refused = await client.get("/api/history", headers=user)
            release.set()
            admin = await _settled_as(client, running, ADMIN)
        assert capabilities["access"]["role"] == "user"
        assert _reason(settings) == (403, "settings_forbidden")
        assert "event: snapshot" in ended.text
        assert _reason(reconnect) == (401, "token_required")
        assert _reason(spent) == (401, "ticket_invalid")
        assert _reason(refused) == (401, "token_required")
        # The admin still sees a revoked user's jobs
        assert admin["status"] == "done"

    async def test_an_unknown_user_token_is_refused_even_from_loopback(
        self, tmp_path: Path
    ) -> None:
        app = _make_app(tmp_path, token="admin-token")
        async with _client(app) as client:
            forged = await client.get(
                "/api/history", headers={"Authorization": "Bearer mku_" + "0" * 64}
            )
            anonymous = (await client.get("/api/capabilities")).json()
            stale = await client.get(
                "/api/history", headers={"Authorization": "Bearer an-old-token"}
            )
        assert _reason(forged) == (401, "token_required")
        # Loopback without a token (or with a stale admin token) is the admin
        assert anonymous["access"] == {"role": "admin", "label": None}
        assert stale.status_code == 200

    async def test_the_owner_is_saved_and_read_back_but_never_sent(
        self, tmp_path: Path
    ) -> None:
        app = _make_app(tmp_path, token="admin-token")
        async with _client(app, peer=REMOTE) as client:
            created, user = await _user(client)
            mine = await _job_as(client, user)
            admins = await _job_as(client, ADMIN)
        meta = json.loads((tmp_path / "jobs" / mine / "meta.json").read_text())
        assert meta["owner"] == created["id"]
        assert "owner" not in json.loads(
            (tmp_path / "jobs" / admins / "meta.json").read_text()
        )
        restarted = _make_app(tmp_path, token="admin-token")
        async with _client(restarted, peer=REMOTE) as client:
            history = (await client.get("/api/history", headers=user)).json()
            hidden = await client.get(f"/api/jobs/{admins}", headers=user)
        assert [entry["job_id"] for entry in history] == [mine]
        assert _reason(hidden) == (404, "job_not_found")

    async def test_a_user_may_convert_urls(
        self, tmp_path: Path, web_page: _FakeUrlConversion
    ) -> None:
        app = _make_app(tmp_path, token="admin-token")
        async with _client(app, peer=REMOTE) as client:
            _, user = await _user(client)
            created = await client.post(
                "/api/jobs", files=_form(urls=[URL]), headers=user
            )
        assert created.status_code == 201, created.text


async def _job_as(client: httpx.AsyncClient, headers: dict[str, str]) -> str:
    created = await client.post(
        "/api/jobs", files=_form(files=[("one.pdf", b"%")]), headers=headers
    )
    assert created.status_code == 201, created.text
    job_id = created.json()["job_id"]
    await _settled_as(client, job_id, headers)
    return job_id


async def _settled_as(
    client: httpx.AsyncClient, job_id: str, headers: dict[str, str]
) -> dict[str, Any]:
    for _ in range(1500):
        data = (await client.get(f"/api/jobs/{job_id}", headers=headers)).json()
        if data["status"] != "running" and all(
            item["status"] in ("done", "error") for item in data["items"]
        ):
            return data
        await asyncio.sleep(0.02)
    pytest.fail(f"job {job_id} did not settle")


# ------------------------------------------------------------ 8 edit config

#: Unusual layout, keys out of order, escapes in keys and values; invented
#: values throughout.
CONFIG = (
    '{\n\t"notes" : { "password":"abc","empty_secret":"", "title":"a \\"quoted\\" '
    '\\u00e9 value" },\n   "llm":{"providers":[ {"id":"p1", "provider":"openai",'
    '"api_key":"env:OPENAI_API_KEY","api_base":"https://api.example/v1"} ],\n'
    '  "model_list" :[{"model_name":"default","litellm_params":{\n'
    '      "model":"openai/gpt-4o-mini",  "api_key":"sk-test-abcdef1234567890",\n'
    '      "api_base":"https://user:pass@gateway.example/v1?key=s3cret&x=1" }}]},\n'
    '"fetch":{"playwright":{"extra_http_headers":{"X-Custom":"header-value-123",'
    '"Accept":"text/html"}},\n "cloudflare":{"api_token":"cf-token-abcdefghijkl",'
    '"account_id":"acct-1"}}, "n": [1, -2.5e3, true, false, null, {}, []]\n}'
)
SECRETS = (
    '"abc"',
    "abcdef1234567890",
    "pass@",
    "s3cret",
    "header-value-123",
    "text/html",
    "cf-token",
)


class TestConfigEditor:
    def test_the_scanner_finds_every_string_value_with_its_pointer(self) -> None:
        found = config_editor.literals(CONFIG)
        assert found is not None
        listed = [(item.pointer, CONFIG[item.start : item.end]) for item in found]
        assert listed[0] == ("/notes/password", '"abc"')
        assert listed[2] == ("/notes/title", '"a \\"quoted\\" \\u00e9 value"')
        assert ("/llm/providers/0/id", '"p1"') in listed
        assert len(listed) == 15
        value = json.loads(CONFIG)
        for item in found:
            node: Any = value
            for step in item.path:
                node = node[step]
            assert node == json.loads(CONFIG[item.start : item.end])
        assert config_editor.literals('{"a":1} x') is None
        escaped = config_editor.literals('{"a/b~c": ["\\\\", "x"]}')
        assert escaped is not None
        assert [item.pointer for item in escaped] == ["/a~1b~0c/0", "/a~1b~0c/1"]

    def test_secrets_headers_and_addresses_are_masked_in_the_files_own_text(
        self,
    ) -> None:
        view = config_editor.view("~/c.json", "user", CONFIG.encode(), read_only=False)
        text = view["text"]
        assert view["masked"] == 6
        for secret in SECRETS:
            assert secret not in text, secret
        expected = (
            CONFIG.replace('"abc"', '"…"')
            .replace('"sk-test-abcdef1234567890"', '"sk…7890"')
            .replace('"header-value-123"', '"he…-123"')
            .replace('"text/html"', '"…ml"')
            .replace('"cf-token-abcdefghijkl"', '"cf…ijkl"')
            .replace(
                "https://user:pass@gateway.example/v1?key=s3cret&x=1",
                "https://…@gateway.example/v1?key=…&x=1",
            )
        )
        assert text == expected
        assert "env:OPENAI_API_KEY" in text and '"acct-1"' in text

    def test_a_header_value_is_masked_even_as_an_env_reference(self) -> None:
        # Playwright sends header values as written: env: is no reference there
        text = '{"fetch": {"playwright": {"extra_http_headers": {"X": "env:HEADER_VALUE"}}}}'
        view = config_editor.view("p", "user", text.encode(), read_only=False)
        assert view["masked"] == 1
        assert "HEADER_VALUE" not in view["text"]

    def test_kept_masks_come_back_and_moved_ones_are_refused(self) -> None:
        masked, _ = config_editor.mask(CONFIG)
        sent = (
            masked.replace('"acct-1"', '"acct-2"')
            .replace('"…"', '"typed-new-password"')
            .replace("\n", "\r\n")
        )
        assert config_editor.unmask(sent, CONFIG) == (
            CONFIG.replace('"acct-1"', '"acct-2"')
            .replace('"abc"', '"typed-new-password"')
            .replace("\n", "\r\n")
        )
        assert config_editor.unmask(masked, CONFIG) == CONFIG
        moved = masked.replace('"account_id":"acct-1"', '"secret_copy":"cf…ijkl"')
        with pytest.raises(ValueError) as refused:
            config_editor.unmask(moved, CONFIG)
        assert str(refused.value) == (
            "/fetch/cloudflare/secret_copy: a masked value can only be kept where "
            "it was; type the full value or an env: reference"
        )
        swapped = masked.replace('"X-Custom":', '"X-Other":')
        with pytest.raises(
            ValueError, match="^/fetch/playwright/extra_http_headers/X-Other: "
        ):
            config_editor.unmask(swapped, CONFIG)
        # A plain value may hold the character; without a file nothing is kept
        plain = masked.replace('a \\"quoted\\" \\u00e9 value', "To be continued…")
        config_editor.unmask(plain, CONFIG)
        with pytest.raises(ValueError):
            config_editor.unmask(masked, None)

    def test_revisions_problems_and_restart_needs(self) -> None:
        assert config_editor.revision(None) == "absent"
        assert config_editor.revision(b"{}") == (
            "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        )
        broken = config_editor.view(
            "p", "explicit", b'{\n  "a": 1\n  "b": 2\n}', read_only=False
        )
        assert broken["text"] is None
        assert broken["problem"] == "line 3, column 3: Expecting ',' delimiter"
        binary = config_editor.view("p", "explicit", b'{"a":"\xff"}', read_only=False)
        assert binary["problem"] == "the configuration is not UTF-8 text"
        array = config_editor.view("p", "explicit", b"[]", read_only=False)
        assert array["problem"] == "the configuration must be a JSON object"
        nan = config_editor.view("p", "explicit", b'{"a": NaN}', read_only=False)
        assert nan["problem"] == "line 1, column 7: NaN is not allowed in JSON"
        assert config_editor.view("p", "default", None, read_only=True) == {
            "path": "p",
            "origin": "default",
            "exists": False,
            "revision": "absent",
            "text": "{}\n",
            "masked": 0,
            "problem": None,
            "read_only": True,
        }
        old = {"llm": {"enabled": True, "model_list": []}, "batch": {"concurrency": 2}}
        models = {
            "llm": {
                "enabled": True,
                "model_list": [{"model_name": "m"}],
                "providers": [],
            },
            "batch": {"concurrency": 2},
        }
        assert not config_editor.restart_required(old, models)
        assert not config_editor.restart_required({}, {"llm": {"model_list": []}})
        other = {
            "llm": {"enabled": True, "model_list": []},
            "batch": {"concurrency": 3},
        }
        assert config_editor.restart_required(old, other)
        assert config_editor.restart_required(None, old)


#: A valid configuration with one model (invented key) and one other setting.
SERVED_CONFIG = (
    "{\n"
    '  "llm": {\n'
    '    "model_list": [\n'
    '      {"model_name": "default", "litellm_params": {"model": "openai/gpt-4o-mini",'
    ' "api_key": "sk-invented-0000aaaa1111"}}\n'
    "    ]\n"
    "  },\n"
    '  "batch": {"concurrency": 2}\n'
    "}\n"
)


def _config_app(tmp_path: Path, text: str | None = SERVED_CONFIG) -> FastAPI:
    if text is not None:
        (tmp_path / "config.json").write_text(text, encoding="utf-8", newline="")
    return _make_app(tmp_path)


class TestConfigRoutes:
    async def test_get_shows_the_file_masked(self, tmp_path: Path) -> None:
        async with _client(_config_app(tmp_path)) as client:
            answer = await client.get("/api/settings/config")
        assert answer.status_code == 200
        assert answer.headers["cache-control"] == "no-store"
        body = answer.json()
        check_shape("ConfigDocument", body)
        assert body["exists"] is True
        assert body["revision"] == config_editor.revision(SERVED_CONFIG.encode())
        assert body["text"] == SERVED_CONFIG.replace(
            "sk-invented-0000aaaa1111", "sk…1111"
        )
        assert body["masked"] == 1
        assert body["problem"] is None
        assert body["read_only"] is False
        assert body["path"].endswith("config.json")
        assert body["origin"] == "explicit"

    async def test_save_keeps_the_masked_key_and_applies_the_models(
        self, tmp_path: Path
    ) -> None:
        async with _client(_config_app(tmp_path)) as client:
            shown = (await client.get("/api/settings/config")).json()
            before = (await client.get("/api/settings/llm")).json()
            # A second model, typed with its own key; layout kept as typed
            edited = shown["text"].replace(
                "\n    ]",
                ',\n      {"model_name": "default", "litellm_params": '
                '{"model": "openai/gpt-4o", "api_key": "env:OPENAI_API_KEY"}}\n    ]',
            )
            saved = await client.put(
                "/api/settings/config",
                json={"expected_revision": shown["revision"], "text": edited},
            )
            after = (await client.get("/api/settings/llm")).json()
            capabilities = (await client.get("/api/capabilities")).json()
        assert saved.status_code == 200, saved.text
        body = saved.json()
        check_shape("SavedConfigDocument", body)
        written = (tmp_path / "config.json").read_bytes()
        assert written == edited.replace("sk…1111", "sk-invented-0000aaaa1111").encode()
        assert body["revision"] == config_editor.revision(written)
        assert body["text"] == edited
        assert body["restart_required"] is False
        assert len(after["deployments"]) == 2
        assert after["revision"] != before["revision"]
        assert "openai/gpt-4o" in capabilities["llm"]["models"]
        if sys.platform != "win32":
            assert (tmp_path / "config.json").stat().st_mode & 0o777 == 0o600

    @pytest.mark.parametrize(
        "final", ["\r\n", ""], ids=["final-break", "no-final-break"]
    )
    async def test_a_crlf_file_round_trips_byte_for_byte(
        self, tmp_path: Path, final: str
    ) -> None:
        # Written on Windows: CRLF line breaks (contract v1.13 clarification)
        text = SERVED_CONFIG.rstrip("\n").replace("\n", "\r\n") + final
        stored = text.encode()
        async with _client(_config_app(tmp_path, text)) as client:
            shown = (await client.get("/api/settings/config")).json()
            unchanged = await client.put(
                "/api/settings/config",
                json={"expected_revision": shown["revision"], "text": shown["text"]},
            )
            after_unchanged = (tmp_path / "config.json").read_bytes()
            edited = shown["text"].replace('"concurrency": 2', '"concurrency": 3')
            changed = await client.put(
                "/api/settings/config",
                json={
                    "expected_revision": unchanged.json()["revision"],
                    "text": edited,
                },
            )
            after_changed = (tmp_path / "config.json").read_bytes()
        # GET sends the file's own breaks, the key masked
        assert shown["revision"] == config_editor.revision(stored)
        assert shown["text"] == text.replace("sk-invented-0000aaaa1111", "sk…1111")
        assert "\r\n" in shown["text"]
        # An unchanged PUT writes identical bytes and keeps the revision
        assert unchanged.status_code == 200, unchanged.text
        assert after_unchanged == stored
        assert unchanged.json()["revision"] == shown["revision"]
        assert unchanged.json()["restart_required"] is False
        # One changed value changes only its line; the breaks and the
        # (missing) final break stay as they were
        assert changed.status_code == 200, changed.text
        before_lines = stored.split(b"\r\n")
        after_lines = after_changed.split(b"\r\n")
        assert len(after_lines) == len(before_lines)
        differing = [
            (old, new)
            for old, new in zip(before_lines, after_lines, strict=True)
            if old != new
        ]
        assert differing == [
            (b'  "batch": {"concurrency": 2}', b'  "batch": {"concurrency": 3}')
        ]
        assert b"\n" not in after_changed.replace(b"\r\n", b"")
        assert after_changed.endswith(b"}" + final.encode())
        assert b"sk-invented-0000aaaa1111" in after_changed

    async def test_saved_exactly_as_sent(self, tmp_path: Path) -> None:
        async with _client(_config_app(tmp_path)) as client:
            shown = (await client.get("/api/settings/config")).json()
            edited = (
                shown["text"]
                .replace('"concurrency": 2', '"concurrency" :3')
                .replace("\n", "\r\n")
            )
            saved = await client.put(
                "/api/settings/config",
                json={"expected_revision": shown["revision"], "text": edited},
            )
        assert saved.json()["restart_required"] is True
        assert (tmp_path / "config.json").read_bytes() == edited.replace(
            "sk…1111", "sk-invented-0000aaaa1111"
        ).encode()

    async def test_a_new_file(self, tmp_path: Path) -> None:
        async with _client(_config_app(tmp_path, None)) as client:
            shown = (await client.get("/api/settings/config")).json()
            saved = await client.put(
                "/api/settings/config",
                json={"expected_revision": "absent", "text": '{"batch": {}}'},
            )
        assert shown["exists"] is False
        assert shown["revision"] == "absent"
        assert shown["text"] == "{}\n"
        assert saved.status_code == 200
        assert saved.json()["exists"] is True
        assert (tmp_path / "config.json").read_text() == '{"batch": {}}'

    async def test_a_stale_revision_names_the_current_one(self, tmp_path: Path) -> None:
        async with _client(_config_app(tmp_path)) as client:
            shown = (await client.get("/api/settings/config")).json()
            (tmp_path / "config.json").write_text('{"batch": {}}\n', encoding="utf-8")
            stale = await client.put(
                "/api/settings/config",
                json={"expected_revision": shown["revision"], "text": shown["text"]},
            )
        assert _reason(stale) == (409, "stale_revision")
        assert stale.json()["detail"] == {
            "code": "stale_revision",
            "current_revision": config_editor.revision(
                (tmp_path / "config.json").read_bytes()
            ),
        }
        assert stale.headers["cache-control"] == "no-store"

    async def test_a_file_changed_while_saving_is_not_overwritten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from markitai import security

        real = security.atomic_write_bytes

        def edited_meanwhile(path: Path, data: bytes, **kwargs: Any) -> None:
            path.write_text('{"edited": "by hand"}', encoding="utf-8")
            real(path, data, **kwargs)

        monkeypatch.setattr(security, "atomic_write_bytes", edited_meanwhile)
        async with _client(_config_app(tmp_path)) as client:
            shown = (await client.get("/api/settings/config")).json()
            changed = await client.put(
                "/api/settings/config",
                json={"expected_revision": shown["revision"], "text": "{}"},
            )
        assert _reason(changed) == (409, "config_changed")
        assert changed.json()["detail"]["code"] == "config_changed"
        assert (tmp_path / "config.json").read_text() == '{"edited": "by hand"}'

    @pytest.mark.parametrize(
        ("text", "detail"),
        [
            ('{\n  "a": 1\n  "b": 2\n}', "line 3, column 3: Expecting ',' delimiter"),
            ("[]", "the configuration must be a JSON object"),
            ('{"batch": {"concurrency": "many"}}', "batch.concurrency: "),
            ('{"llm": {"model_list": [{"model_name": "x"}]}}', "llm.model_list.0."),
            (
                '{"llm": {"model_list": [{"model_name": "default", "litellm_params": '
                '{"model": "openai/y"}}, {"model_name": "default", "litellm_params": '
                '{"model": "openai/x", "api_key": "sk…1111"}}]}}',
                "/llm/model_list/1/litellm_params/api_key: a masked value can only be",
            ),
        ],
        ids=["not-json", "not-an-object", "not-config", "not-a-model", "moved-mask"],
    )
    async def test_refused_texts(self, tmp_path: Path, text: str, detail: str) -> None:
        async with _client(_config_app(tmp_path)) as client:
            shown = (await client.get("/api/settings/config")).json()
            refused = await client.put(
                "/api/settings/config",
                json={"expected_revision": shown["revision"], "text": text},
            )
        assert _reason(refused) == (422, "invalid_config")
        assert refused.json()["detail"].startswith(detail), refused.json()["detail"]
        assert (tmp_path / "config.json").read_text() == SERVED_CONFIG
        assert "sk-invented" not in refused.text

    @pytest.mark.parametrize(
        "body",
        [
            {"text": "{}"},
            {"expected_revision": "absent"},
            {"expected_revision": "absent", "text": 1},
            {"expected_revision": "absent", "text": "{}", "x": 1},
            [],
        ],
    )
    async def test_other_bodies_are_refused(self, tmp_path: Path, body: Any) -> None:
        async with _client(_config_app(tmp_path)) as client:
            refused = await client.put("/api/settings/config", json=body)
            form = await client.put("/api/settings/config", data={"text": "{}"})
        assert _reason(refused) == (422, "invalid_settings_request")
        assert _reason(form) == (422, "invalid_settings_request")

    async def test_over_1_mib_is_refused(self, tmp_path: Path) -> None:
        text = '{"a": "' + "x" * (1024 * 1024) + '"}'
        async with _client(_config_app(tmp_path)) as client:
            refused = await client.put(
                "/api/settings/config",
                json={"expected_revision": "absent", "text": text},
            )
        assert _reason(refused) == (413, "config_too_large")

    async def test_trust_as_for_the_settings(self, tmp_path: Path) -> None:
        app = _make_app(tmp_path / "a", token="admin-token")
        async with _client(app) as local:
            without = await local.get("/api/settings/config")
            admin = await local.get("/api/settings/config", headers=ADMIN)
        no_auth = _make_app(tmp_path / "b")
        async with _client(no_auth, peer=REMOTE) as remote:
            untrusted = await remote.get("/api/settings/config")
        assert _reason(without) == (401, "token_required")
        assert admin.status_code == 200
        assert _reason(untrusted) == (403, "settings_forbidden")

    async def test_the_page_is_served(self) -> None:
        from markitai.serve.workbench import APP_PATHS

        assert "/settings/config" in APP_PATHS


def test_access_store_path_is_beside_the_jobs_root(tmp_path: Path) -> None:
    # ~/.markitai/serve/access.json for the default ~/.markitai/serve/jobs
    app = _make_app(tmp_path)
    assert app.state.access_tokens.path == tmp_path / "access.json"


def test_a_duplicated_key_is_refused_so_no_secret_is_shown_unmasked() -> None:
    """JSON keeps the last of two equal keys, so the first secret was shown
    in plain text; the file is refused with the position instead."""
    from markitai.serve import config_editor

    text = (
        '{"llm":{"model_list":[{"litellm_params":{"model":"openai/x",'
        '"api_key":"sk-first-secret-1234","api_key":"sk-second-secret-5678"}}]}}'
    )
    view = config_editor.view("c.json", "user", text.encode(), read_only=False)
    assert view["text"] is None
    assert "appears twice" in view["problem"]
    assert "sk-first-secret-1234" not in str(view)
