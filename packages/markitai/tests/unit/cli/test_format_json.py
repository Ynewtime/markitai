"""``-f json``: the document on stdout, beside the Markdown, under an exact
name and in a batch with its index (markitai-rust's tests/document_json.rs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from markitai.cli.main import app

ORDER = [
    "schema_version",
    "tool",
    "source",
    "status",
    "skip_reason",
    "title",
    "properties",
    "markdown",
    "text",
    "pages",
    "images",
    "tables",
    "notices",
    "usage",
    "timings",
    "outputs",
]


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "markitai.json"
    config.write_text(
        json.dumps({"cache": {"enabled": False}, "llm": {"enabled": False}}),
        encoding="utf-8",
    )

    def invoke(*args: str) -> Any:
        return CliRunner().invoke(app, [*args, "-c", str(config)])

    return invoke


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_stdout_carries_the_document_with_every_key(tmp_path: Path, run: Any) -> None:
    (tmp_path / "note.txt").write_bytes(b"# Note\n\nHello world.\n")
    # The option before the input: its value is not read as the input
    result = run("--format", "JSON", "note.txt")
    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert list(document) == ORDER
    assert document["schema_version"] == "1.0"
    assert document["source"]["path"] == "note.txt"
    assert document["source"]["size"] == 21
    assert len(document["source"]["sha256"]) == 64
    assert document["title"] == "Note"
    assert document["markdown"]["llm"] is None
    assert "Hello world." in document["markdown"]["base"]
    assert document["text"] == "Note\n\nHello world."
    assert document["outputs"]["markdown"] is None
    # Markdown stays the default
    plain = run("note.txt", "-f", "md")
    assert plain.stdout.startswith("---\n")


def test_beside_the_markdown_the_json_follows_its_renames(
    tmp_path: Path, run: Any
) -> None:
    (tmp_path / "note.txt").write_text("Hello.\n")
    result = run("note.txt", "-f", "json", "-o", "out")
    assert result.exit_code == 0, result.output
    assert "note.txt.json" in result.output
    assert _json(tmp_path / "out/note.txt.json")["outputs"]["markdown"] == "note.txt.md"
    assert (tmp_path / "out/note.txt.md").is_file()
    assert run("note.txt", "-f", "json", "-o", "out").exit_code == 0
    assert _json(tmp_path / "out/note.txt.v2.json")["outputs"]["markdown"] == (
        "note.txt.v2.md"
    )
    # A JSON an earlier run left takes its name under `rename`
    (tmp_path / "out/note.txt.v2.md").unlink()
    assert run("note.txt", "-f", "json", "-o", "out").exit_code == 0
    assert (tmp_path / "out/note.txt.v3.json").is_file()
    # Without -f json an earlier JSON takes nothing
    assert run("note.txt", "-o", "out").exit_code == 0
    assert (tmp_path / "out/note.txt.v2.md").is_file()


def test_an_exact_json_name_names_the_markdown_beside_it(
    tmp_path: Path, run: Any
) -> None:
    (tmp_path / "note.txt").write_text("Hello.\n")
    result = run("note.txt", "-f", "json", "-o", "out/result.json")
    assert result.exit_code == 0, result.output
    assert _json(tmp_path / "out/result.json")["outputs"]["markdown"] == "result.md"
    assert (tmp_path / "out/result.md").is_file()


def test_a_batch_writes_one_document_per_file_and_an_index(
    tmp_path: Path, run: Any
) -> None:
    (tmp_path / "in/sub").mkdir(parents=True)
    (tmp_path / "in/a.txt").write_text("A.\n")
    (tmp_path / "in/sub/b.md").write_text("# B\n")
    result = run("in", "-f", "json", "-o", "out")
    assert result.exit_code == 0, result.output
    index = _json(tmp_path / "out/markitai.index.json")
    assert index["schema_version"] == "1.0"
    assert [(i["source"], i["document"], i["markdown"]) for i in index["items"]] == [
        ("a.txt", "a.txt.json", "a.txt.md"),
        ("sub/b.md", "sub/b.md.json", "sub/b.md.md"),
    ]
    assert index["items"][0]["status"] == "completed"
    assert "# B" in _json(tmp_path / "out/sub/b.md.json")["markdown"]["base"]
    # A repeated run keeps the items it did not touch
    (tmp_path / "in/a.txt").unlink()
    assert run("in", "-f", "json", "-o", "out").exit_code == 0
    sources = [
        i["source"] for i in _json(tmp_path / "out/markitai.index.json")["items"]
    ]
    assert sources == ["a.txt", "sub/b.md"]


def test_a_batch_refuses_a_json_file_and_the_llm_batch(
    tmp_path: Path, run: Any
) -> None:
    (tmp_path / "in").mkdir()
    (tmp_path / "in/a.txt").write_text("A.\n")
    result = run("in", "-f", "json", "-o", "out.json")
    assert result.exit_code == 2
    assert "A batch requires an output directory, not a .json file" in result.output
    refused = run("in", "-f", "json", "--llm-batch", "--llm", "-o", "out")
    assert refused.exit_code == 1
    assert "-f json is not supported with --llm-batch" in refused.output
