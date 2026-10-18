"""A ``./markitai.json`` found where ``markitai serve`` starts stays sealed.

Loaded, its models at their own address use no key from the environment
(``markitai.config.seal_project_models``). Two settings paths ran without
that: Edit config stored the sealed copy's revision, so the next settings
write was refused as stale; and any model write ran the file's models
unsealed. Everything is invented; nothing leaves the machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from markitai.config import KEYLESS_PLACEHOLDER
from tests.unit.serve.test_user_models import ADMIN, ADMIN_TOKEN
from tests.unit.serve.test_workbench_v111 import _client

CAPTURE_BASE = "https://capture.example/v1"


def _folder_config() -> dict[str, Any]:
    return {
        "llm": {
            "model_list": [
                {
                    "model_name": "default",
                    "litellm_params": {
                        "model": "openai/gpt-folder",
                        "api_base": CAPTURE_BASE,
                    },
                    "model_info": {"id": "folder-model"},
                }
            ]
        }
    }


def _runtime_keys(app: Any) -> dict[str, str | None]:
    return {
        model.litellm_params.model: model.litellm_params.api_key
        for model in app.state.markitai.config.llm.model_list
    }


async def test_a_folder_config_stays_sealed_through_settings_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "download"
    folder.mkdir()
    config = folder / "markitai.json"
    config.write_text(json.dumps(_folder_config()), encoding="utf-8")
    monkeypatch.chdir(folder)
    monkeypatch.delenv("MARKITAI_CONFIG", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-environment-0000")
    from markitai.serve import create_app

    app = create_app(
        static_dir=tmp_path / "no-static",
        jobs_root=tmp_path / "jobs",
        configure_logging=False,
        token=ADMIN_TOKEN,
    )
    async with _client(app) as client:
        assert app.state.markitai.config_origin == "project"
        assert _runtime_keys(app) == {"openai/gpt-folder": KEYLESS_PLACEHOLDER}

        # Edit config: saved as sent, run sealed, and the revision it leaves
        # is the file's, so the next settings write is not stale
        document = (await client.get("/api/settings/config", headers=ADMIN)).json()
        edited = json.loads(document["text"])
        edited["llm"]["model_list"][0]["litellm_params"]["weight"] = 3
        saved = await client.put(
            "/api/settings/config",
            json={
                "expected_revision": document["revision"],
                "text": json.dumps(edited),
            },
            headers=ADMIN,
        )
        assert saved.status_code == 200, saved.text
        assert KEYLESS_PLACEHOLDER not in config.read_text(encoding="utf-8")
        assert _runtime_keys(app) == {"openai/gpt-folder": KEYLESS_PLACEHOLDER}

        view = (await client.get("/api/settings/llm", headers=ADMIN)).json()
        added = await client.post(
            "/api/settings/llm/models",
            json={
                "expected_revision": view["revision"],
                "model_name": "default",
                "model": "openai/gpt-added",
                "api_key": "sk-added-1234567890",
            },
            headers=ADMIN,
        )
        assert added.status_code == 200, added.text

    # A model write keeps the folder's model sealed too
    assert _runtime_keys(app) == {
        "openai/gpt-folder": KEYLESS_PLACEHOLDER,
        "openai/gpt-added": "sk-added-1234567890",
    }
    on_disk = json.loads(config.read_text(encoding="utf-8"))
    assert "api_key" not in on_disk["llm"]["model_list"][0]["litellm_params"]
