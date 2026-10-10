"""The shared browser workbench, served from the vendored bundle.

The UI is built in the markitai-rust repository (one frontend for both
backends) and vendored here unchanged: ``serve/workbench/manifest.json`` names
its source commit, build command and every file with its size and SHA-256;
``serve/workbench/ui/`` holds the files. ``scripts/sync_workbench.py``
refreshes the copy and checks the digests, so neither Node nor Rust is needed
to install or build this package.

Only fixed names are served: the HTML shell at the app's own paths and each
file of ``ui/`` by its exact name under ``/ui/``; never a directory listing or
any other path. A ``.gz`` twin is sent as-is to clients that accept gzip and
is never served under its own name; a file shipped only compressed is
decompressed once for the others. Every response revalidates (``no-cache``)
against a content ETag, so reloading an unchanged page costs only 304s.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import Request
from loguru import logger

if TYPE_CHECKING:
    from fastapi import FastAPI, Response

#: The vendored bundle inside the package.
BUNDLED_DIR = Path(__file__).with_name("workbench")

#: Addresses of the app's views; each serves the same HTML shell.
APP_PATHS = (
    "/",
    "/jobs",
    "/settings",
    "/settings/models",
    "/settings/appearance",
    "/settings/access",
)

CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "img-src 'self' blob:; font-src 'self'; frame-src 'none'; object-src 'none'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)

_CONTENT_TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
    ".json": "application/json",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
    ".webmanifest": "application/manifest+json",
}


@dataclass(frozen=True)
class Asset:
    """One servable file: plain bytes, its gzip twin when shipped, a validator."""

    plain: bytes
    compressed: bytes | None
    content_type: str
    etag: str


def _etag(data: bytes) -> str:
    # Weak: both encodings of one file share it.
    return f'W/"{hashlib.sha256(data).hexdigest()[:16]}"'


class Workbench:
    """The loaded bundle: name -> asset, plus provenance from the manifest."""

    def __init__(self, assets: dict[str, Asset], manifest: dict[str, object]) -> None:
        self.assets = assets
        self.manifest = manifest

    @property
    def index(self) -> Asset:
        return self.assets["index.html"]

    @property
    def source_commit(self) -> str | None:
        value = self.manifest.get("source_commit")
        return value if isinstance(value, str) else None

    @classmethod
    def load(cls, directory: Path) -> Workbench | None:
        """Load and verify a bundle directory; None when absent or damaged.

        Every file must match the manifest's size and digest and the manifest
        must list every file: a partial or edited copy is not served.
        """
        manifest_path = directory / "manifest.json"
        ui_dir = directory / "ui"
        if not manifest_path.is_file() or not (ui_dir / "index.html").is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            listed = manifest["files"]
            if not isinstance(listed, dict):
                raise ValueError("manifest 'files' is not an object")
            files: dict[str, bytes] = {}
            for path in sorted(ui_dir.iterdir()):
                if not path.is_file():
                    raise ValueError(f"unexpected entry {path.name}")
                files[path.name] = path.read_bytes()
            expected = {
                str(name).removeprefix("ui/"): entry for name, entry in listed.items()
            }
            if set(expected) != set(files):
                raise ValueError("bundle files differ from the manifest")
            for name, data in files.items():
                entry = expected[name]
                if (
                    not isinstance(entry, dict)
                    or entry.get("bytes") != len(data)
                    or entry.get("sha256") != hashlib.sha256(data).hexdigest()
                ):
                    raise ValueError(f"{name} does not match the manifest")
            assets: dict[str, Asset] = {}
            for name, data in files.items():
                if name.endswith(".gz"):
                    base = name.removesuffix(".gz")
                    if base in files:
                        continue  # a twin; picked up with its plain file
                    plain, compressed = gzip.decompress(data), data
                else:
                    base, plain = name, data
                    compressed = files.get(f"{name}.gz")
                    if compressed is not None and gzip.decompress(compressed) != plain:
                        raise ValueError(f"{name}.gz is not a gzip copy of {name}")
                content_type = _CONTENT_TYPES.get(
                    Path(base).suffix.lower(), "application/octet-stream"
                )
                assets[base] = Asset(plain, compressed, content_type, _etag(plain))
        except (OSError, ValueError, KeyError, TypeError, EOFError) as e:
            logger.warning("[Serve] Web UI bundle in {} not served: {}", directory, e)
            return None
        return cls(assets, manifest)


def _accepts_gzip(header: str | None) -> bool:
    """``gzip`` (or ``*``) with a nonzero weight in Accept-Encoding."""
    for part in (header or "").split(","):
        coding, *params = (field.strip() for field in part.split(";"))
        weight = 1.0
        for param in params:
            if param.startswith("q="):
                try:
                    weight = float(param[2:])
                except ValueError:
                    weight = 0.0
        if coding.lower() in ("gzip", "*") and weight > 0:
            return True
    return False


def _not_modified(header: str | None, etag: str) -> bool:
    ours = etag.removeprefix("W/")
    for tag in (header or "").split(","):
        tag = tag.strip()
        if tag == "*" or tag.removeprefix("W/") == ours:
            return True
    return False


def static_headers() -> dict[str, str]:
    """Headers every UI response carries."""
    return {
        "Content-Security-Policy": CSP,
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-cache",
    }


def serve_asset(asset: Asset, request: Request) -> Response:
    """Answer one asset request (200 or 304, gzip when accepted)."""
    from fastapi.responses import Response

    headers = static_headers()
    headers["ETag"] = asset.etag
    if asset.compressed is not None:
        headers["Vary"] = "Accept-Encoding"
    if _not_modified(request.headers.get("if-none-match"), asset.etag):
        return Response(status_code=304, headers=headers)
    body = asset.plain
    if asset.compressed is not None and _accepts_gzip(
        request.headers.get("accept-encoding")
    ):
        body = asset.compressed
        headers["Content-Encoding"] = "gzip"
    return Response(content=body, media_type=asset.content_type, headers=headers)


def mount_workbench(app: FastAPI, workbench: Workbench) -> None:
    """Register the shell and ``/ui/<file>`` routes on *app*."""
    from markitai.serve.errors import ApiError

    async def shell(request: Request) -> Response:
        return serve_asset(workbench.index, request)

    for path in APP_PATHS:
        app.add_api_route(path, shell, methods=["GET", "HEAD"], include_in_schema=False)

    async def ui_file(request: Request, name: str) -> Response:
        asset = workbench.assets.get(name)
        if asset is None:
            raise ApiError(404, "route_not_found", "route not found")
        return serve_asset(asset, request)

    app.add_api_route(
        "/ui/{name}", ui_file, methods=["GET", "HEAD"], include_in_schema=False
    )
