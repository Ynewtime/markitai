#!/usr/bin/env python3
"""Vendor the shared web workbench into the package, or check the vendored copy.

The workbench UI is built in the markitai-rust repository (one frontend for
both backends) and published as a bundle: a ``manifest.json`` listing every
file of its ``ui/`` folder with size and SHA-256, the source commit and the
build command, plus the files themselves (in a ``.tar.gz`` next to the
manifest, or a ``ui/`` folder). This script copies a bundle, unchanged, to
``packages/markitai/src/markitai/serve/workbench/``. The wheel ships that copy,
so installing or building markitai never needs Node or Rust.

Usage::

    python scripts/sync_workbench.py --from ../workbench-builds/dist
    python scripts/sync_workbench.py --check                 # vendored copy vs its manifest
    python scripts/sync_workbench.py --check --from DIST     # ... and vs a published bundle

Every file is verified against the manifest before anything is written, and
``--check`` fails on any missing, extra or altered file.
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


class BundleError(Exception):
    """The bundle is incomplete, altered or malformed."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _listed_files(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    """Manifest entries keyed by name inside ``ui/``."""
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise BundleError("manifest has no 'files' map")
    listed: dict[str, dict[str, object]] = {}
    for raw_name, entry in files.items():
        name = str(raw_name).removeprefix("ui/")
        if not _plain_name(name) or not isinstance(entry, dict):
            raise BundleError(f"manifest lists an unusable entry: {raw_name!r}")
        listed[name] = entry
    return listed


def _plain_name(name: str) -> bool:
    return bool(name) and "/" not in name and "\\" not in name and name not in (".", "..")


def _verify(files: dict[str, bytes], manifest: dict[str, object]) -> None:
    listed = _listed_files(manifest)
    missing = sorted(set(listed) - set(files))
    extra = sorted(set(files) - set(listed))
    if missing or extra:
        raise BundleError(f"files differ from the manifest: missing {missing}, extra {extra}")
    for name, data in sorted(files.items()):
        entry = listed[name]
        if entry.get("bytes") != len(data) or entry.get("sha256") != _sha256(data):
            raise BundleError(f"ui/{name} does not match its manifest digest")
    if "index.html" not in files:
        raise BundleError("bundle has no ui/index.html")


def _read_tarball(data: bytes) -> tuple[dict[str, object] | None, dict[str, bytes]]:
    """Regular files of a bundle tarball: (inner manifest, ui files)."""
    manifest: dict[str, object] | None = None
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            parts = [part for part in path.parts if part not in ("", ".")]
            if member.isdir():
                continue
            if not member.isfile() or path.is_absolute() or ".." in parts:
                raise BundleError(f"unsafe or non-file tar entry: {member.name!r}")
            if member.size > MAX_FILE_BYTES:
                raise BundleError(f"tar entry too large: {member.name!r}")
            # A bundle may wrap its contents in one top-level folder.
            if parts and parts[-1] == "manifest.json" and "ui" not in parts:
                handle = archive.extractfile(member)
                assert handle is not None
                manifest = json.loads(handle.read().decode("utf-8"))
                continue
            if "ui" not in parts:
                continue  # licences and notes beside the bundle stay out of the package
            inner = parts[parts.index("ui") + 1 :]
            if len(inner) != 1 or not _plain_name(inner[0]):
                raise BundleError(f"nested or unusable ui entry: {member.name!r}")
            handle = archive.extractfile(member)
            assert handle is not None
            files[inner[0]] = handle.read()
    return manifest, files


def load_bundle(source: Path) -> tuple[dict[str, object], dict[str, bytes]]:
    """Read and verify a published bundle: a dist dir, a tarball, or a ui/ dir."""
    source = source.expanduser()
    if source.is_file():
        tarball, outer = source, None
    else:
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            raise BundleError(f"{source} has no manifest.json")
        outer = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (source / "ui").is_dir():
            files = {
                path.name: path.read_bytes()
                for path in sorted((source / "ui").iterdir())
                if path.is_file()
            }
            _verify(files, outer)
            return outer, files
        bundle = outer.get("bundle") if isinstance(outer, dict) else None
        named = bundle.get("file") if isinstance(bundle, dict) else None
        candidates = (
            [source / str(named)]
            if named
            else sorted(source.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime)
        )
        if not candidates or not candidates[-1].is_file():
            raise BundleError(f"{source} has no bundle tarball")
        tarball = candidates[-1]
    data = tarball.read_bytes()
    if outer is not None and isinstance(outer.get("bundle"), dict):
        expected = outer["bundle"].get("sha256")  # type: ignore[union-attr]
        if expected and expected != _sha256(data):
            raise BundleError(f"{tarball.name} does not match the manifest's bundle digest")
    inner, files = _read_tarball(data)
    manifest = inner if inner is not None else outer
    if manifest is None:
        raise BundleError(f"{tarball.name} carries no manifest.json")
    _verify(files, manifest)
    if outer is not None and inner is not None and _listed_files(outer) != _listed_files(inner):
        raise BundleError("the manifest beside the tarball differs from the one inside it")
    return manifest, files


def vendored() -> tuple[dict[str, object], dict[str, bytes]]:
    manifest = json.loads((TARGET / "manifest.json").read_text(encoding="utf-8"))
    files = {
        path.name: path.read_bytes()
        for path in sorted((TARGET / "ui").iterdir())
        if path.is_file()
    }
    entries = [path for path in (TARGET / "ui").iterdir() if not path.is_file()]
    if entries:
        raise BundleError(f"unexpected entries in the vendored ui/: {entries}")
    return manifest, files


def write(manifest: dict[str, object], files: dict[str, bytes]) -> None:
    """Replace the vendored copy atomically enough for a checkout."""
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TARGET.parent) as tmp:
        staged = Path(tmp) / "workbench"
        (staged / "ui").mkdir(parents=True)
        for name, data in files.items():
            (staged / "ui" / name).write_bytes(data)
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if TARGET.exists():
            shutil.rmtree(TARGET)
        staged.rename(TARGET)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", type=Path, help="published bundle: dist dir, tarball, or dir with manifest.json + ui/")
    parser.add_argument("--check", action="store_true", help="verify the vendored copy (and, with --from, that it equals that bundle)")
    args = parser.parse_args(argv)
    try:
        if args.check:
            manifest, files = vendored()
            _verify(files, manifest)
            if args.source is not None:
                published_manifest, published = load_bundle(args.source)
                if published != files:
                    raise BundleError(
                        f"vendored workbench ({manifest.get('source_commit')}) differs from "
                        f"{args.source} ({published_manifest.get('source_commit')}); "
                        "run scripts/sync_workbench.py --from <bundle>"
                    )
            print(f"workbench OK: {len(files)} files, source {manifest.get('source_commit')}")
            return 0
        if args.source is None:
            parser.error("--from is required unless --check is given")
        manifest, files = load_bundle(args.source)
        write(manifest, files)
        total = sum(len(data) for data in files.values())
        print(f"vendored {len(files)} files ({total} bytes) from source {manifest.get('source_commit')} into {TARGET.relative_to(ROOT)}")
        return 0
    except (BundleError, OSError, ValueError, tarfile.TarError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
