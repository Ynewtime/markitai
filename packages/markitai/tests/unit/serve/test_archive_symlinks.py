"""A job's download archive never follows a symlink out of the job tree."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_job_archive_skips_symlinks_and_out_of_tree_targets(tmp_path: Path) -> None:
    from markitai.serve.app import _add_job_outputs_to_zip

    out_dir = tmp_path / "job" / "out"
    out_dir.mkdir(parents=True)
    (out_dir / "result.md").write_text("real output\n", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("private contents outside the job\n", encoding="utf-8")
    # A symlink to a file outside the job tree, and one to a directory.
    try:
        os.symlink(secret, out_dir / "leak.txt")
    except OSError as exc:
        # Windows without administrator rights or Developer Mode (WinError
        # 1314): this user cannot make the symlinks the archive must skip
        pytest.skip(f"symlinks unavailable in this environment: {exc}")
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    (outside_dir / "inside.txt").write_text("also outside\n", encoding="utf-8")
    os.symlink(outside_dir, out_dir / "linked-dir")

    job = SimpleNamespace(out_dir=out_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        _add_job_outputs_to_zip(archive, job)
    names = set(zipfile.ZipFile(io.BytesIO(buffer.getvalue())).namelist())

    assert names == {"result.md"}
    assert not any("secret" in name or "inside" in name for name in names)
