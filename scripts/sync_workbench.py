#!/usr/bin/env python3
"""Vendor the shared web workbench into the package, or check the vendored copy.

The workbench UI is built in the markitai-rust repository (one frontend for
both backends) and published as a bundle: a reproducible ``.tar.gz`` holding,
at its root, ``manifest.json``, ``index.html``, ``ui/<file>`` (everything served
under ``/ui/``), ``capture/<file>`` (scripts the server runs in a browser it
drives, contract v1.14) and ``LICENSES/``, plus a copy of the manifest beside it. The
manifest lists every file with its size and SHA-256, the source commit and the
build command. This script copies a bundle, unchanged, to
``packages/markitai/src/markitai/serve/workbench/``. The wheel ships that copy,
so installing or building markitai never needs Node or Rust.

Usage::

    python scripts/sync_workbench.py --from ~/reviews/web-workbench/dist   # latest.tar.gz
    python scripts/sync_workbench.py --from markitai-workbench-x.tar.gz
    python scripts/sync_workbench.py --check                 # vendored copy vs its manifest
    python scripts/sync_workbench.py --check --from DIST     # ... and vs a published bundle

A publish directory's ``latest.manifest.json`` must match the manifest inside
``latest.tar.gz``, and its ``history.json``, when present, must list the
archive's digest. Every file is verified against the manifest before anything
is written, and ``--check`` fails on any missing, extra or altered file.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "packages" / "markitai" / "src" / "markitai" / "serve" / "workbench"
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_BUNDLE_BYTES = 128 * 1024 * 1024
#: Top-level folders a bundle may carry, and how deep their files sit.
FOLDERS = {"ui": 1, "capture": 1, "LICENSES": 1}


class BundleError(Exception):
    """The bundle is incomplete, altered or malformed."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _usable_path(name: str) -> bool:
    """``index.html``, ``ui/<file>``, ``capture/<file>`` or ``LICENSES/<file>``."""
    parts = PurePosixPath(name).parts
    if not parts or any(part in ("", ".", "..") for part in parts) or "\\" in name:
        return False
    if len(parts) == 1:
        return parts[0] == "index.html"
    return parts[0] in FOLDERS and len(parts) == FOLDERS[parts[0]] + 1


def _verify(files: dict[str, bytes], manifest: object) -> dict[str, object]:
    if not isinstance(manifest, dict):
        raise BundleError("manifest is not a JSON object")
    listed = manifest.get("files")
    if not isinstance(listed, dict) or not listed:
        raise BundleError("manifest has no 'files' map")
    unusable = sorted(name for name in listed if not _usable_path(str(name)))
    if unusable:
        raise BundleError(f"manifest lists unusable paths: {unusable}")
    missing = sorted(set(listed) - set(files))
    extra = sorted(set(files) - set(listed))
    if missing or extra:
        raise BundleError(
            f"files differ from the manifest: missing {missing}, extra {extra}"
        )
    for name, data in sorted(files.items()):
        entry = listed[name]
        if (
            not isinstance(entry, dict)
            or entry.get("bytes") != len(data)
            or entry.get("sha256") != _sha256(data)
        ):
            raise BundleError(f"{name} does not match its manifest digest")
    if "index.html" not in files and "ui/index.html" not in files:
        raise BundleError("bundle has no index.html")
    if not isinstance(manifest.get("source_commit"), str):
        raise BundleError("manifest names no source_commit")
    return manifest


def _read_tarball(data: bytes) -> tuple[dict[str, object], dict[str, bytes]]:
    """The manifest and files of a bundle archive; anything unexpected fails."""
    manifest: object = None
    files: dict[str, bytes] = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name).as_posix().removeprefix("./")
            if member.isdir():
                continue
            if not member.isfile():
                raise BundleError(f"non-file tar entry: {member.name!r}")
            if member.size > MAX_FILE_BYTES:
                raise BundleError(f"tar entry too large: {member.name!r}")
            total += member.size
            if total > MAX_BUNDLE_BYTES:
                raise BundleError("archive too large for a workbench bundle")
            handle = archive.extractfile(member)
            assert handle is not None
            content = handle.read()
            if name == "manifest.json":
                manifest = json.loads(content.decode("utf-8"))
                continue
            if not _usable_path(name):
                raise BundleError(f"unexpected tar entry: {member.name!r}")
            files[name] = content
    if manifest is None:
        raise BundleError("archive carries no manifest.json")
    return _verify(files, manifest), files


def _read_tree(directory: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    files: dict[str, bytes] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            continue
        name = path.relative_to(directory).as_posix()
        if name == "manifest.json":
            continue
        if not path.is_file() or not _usable_path(name):
            raise BundleError(f"unexpected entry in {directory}: {name}")
        files[name] = path.read_bytes()
    return _verify(files, manifest), files


def _check_history(directory: Path, archive_name: str, data: bytes) -> None:
    history_path = directory / "history.json"
    if not history_path.is_file():
        return
    history = json.loads(history_path.read_text(encoding="utf-8"))
    digest = _sha256(data)
    if not any(
        isinstance(entry, dict) and entry.get("sha256") == digest for entry in history
    ):
        raise BundleError(
            f"{archive_name} ({digest[:12]}) is not listed in history.json"
        )


def load_bundle(source: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    """Read and verify a published bundle: a publish dir, a tarball or an unpacked tree."""
    source = source.expanduser()
    if source.is_file():
        return _read_tarball(source.read_bytes())
    if not source.is_dir():
        raise BundleError(f"{source} does not exist")
    if (source / "manifest.json").is_file() and (
        (source / "index.html").is_file() or (source / "ui").is_dir()
    ):
        return _read_tree(source)
    tarball = source / "latest.tar.gz"
    beside = source / "latest.manifest.json"
    if not tarball.is_file():
        candidates = sorted(
            source.glob("*.tar.gz"), key=lambda path: path.stat().st_mtime
        )
        if not candidates:
            raise BundleError(f"{source} has no bundle archive")
        tarball = candidates[-1]
        beside = tarball.with_name(
            tarball.name.removesuffix(".tar.gz") + ".manifest.json"
        )
    data = tarball.read_bytes()
    manifest, files = _read_tarball(data)
    if beside.is_file() and json.loads(beside.read_text(encoding="utf-8")) != manifest:
        raise BundleError(
            f"{beside.name} differs from the manifest inside {tarball.name}"
        )
    _check_history(source, tarball.name, data)
    return manifest, files


def vendored() -> tuple[dict[str, object], dict[str, bytes]]:
    if not (TARGET / "manifest.json").is_file():
        raise BundleError(f"no vendored workbench in {TARGET.relative_to(ROOT)}")
    return _read_tree(TARGET)


def write(manifest: dict[str, object], files: dict[str, bytes]) -> None:
    """Replace the vendored copy with *files* and their manifest."""
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TARGET.parent) as tmp:
        staged = Path(tmp) / "workbench"
        for name, data in files.items():
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if TARGET.exists():
            shutil.rmtree(TARGET)
        staged.rename(TARGET)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--from",
        dest="source",
        type=Path,
        help="published bundle: a publish dir (latest.tar.gz), a .tar.gz, or an unpacked bundle",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the vendored copy (and, with --from, that it equals that bundle)",
    )
    args = parser.parse_args(argv)
    try:
        if args.check:
            manifest, files = vendored()
            if args.source is not None:
                published_manifest, published = load_bundle(args.source)
                if published != files:
                    raise BundleError(
                        f"vendored workbench ({manifest.get('source_commit')}) differs from "
                        f"{args.source} ({published_manifest.get('source_commit')}); "
                        "run scripts/sync_workbench.py --from <bundle>"
                    )
            print(
                f"workbench OK: {len(files)} files, source {manifest.get('source_commit')}"
            )
            return 0
        if args.source is None:
            parser.error("--from is required unless --check is given")
        manifest, files = load_bundle(args.source)
        write(manifest, files)
        total = sum(len(data) for data in files.values())
        print(
            f"vendored {len(files)} files ({total} bytes) from source "
            f"{manifest.get('source_commit')} into {TARGET.relative_to(ROOT)}"
        )
        return 0
    except (BundleError, OSError, ValueError, tarfile.TarError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
