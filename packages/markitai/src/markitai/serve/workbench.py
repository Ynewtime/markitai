"""The shared browser workbench, served from the vendored bundle.

The UI is built in the markitai-rust repository (one frontend for both
backends) and vendored here unchanged, as published: ``manifest.json`` names
its source commit, build command and every file with its size and SHA-256;
``index.html`` is the page shell, ``ui/`` holds the files it loads and
``LICENSES/`` the licences of the libraries and font inside. ``scripts/sync_workbench.py``
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
    "img-src 'self' blob:; font-src 'self'; frame-src 'self'; object-src 'none'; "
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
    """The loaded bundle: the shell, ``/ui/`` assets and the manifest."""

    def __init__(
        self,
        index: Asset,
        assets: dict[str, Asset],
        manifest: dict[str, object],
        page_paths: tuple[str, ...],
    ) -> None:
        self.index = index
        self.assets = assets
        self.manifest = manifest
        self.page_paths = page_paths

    @property
    def source_commit(self) -> str | None:
        value = self.manifest.get("source_commit")
        return value if isinstance(value, str) else None

    @classmethod
    def load(cls, directory: Path) -> Workbench | None:
        """Load and verify a bundle directory; None when absent or damaged.

        The directory holds the bundle as published: ``manifest.json``,
        ``index.html``, ``ui/<file>`` and ``LICENSES/``. Every file the manifest
        lists must be there with its size and digest: a partial or edited copy
        is not served. A file it does not list (a ``.DS_Store`` left by a file
        manager) is ignored, never served.
        """
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = _verified_files(directory, manifest)
            # The shell sits at the bundle root (an older layout kept it in ui/)
            shell = files.get("index.html", files.get("ui/index.html"))
            if shell is None:
                raise ValueError("the bundle has no index.html")
            ui = {
                name.removeprefix("ui/"): data
                for name, data in files.items()
                if name.startswith("ui/") and "/" not in name.removeprefix("ui/")
            }
            assets: dict[str, Asset] = {}
            for name, data in ui.items():
                if name.endswith(".gz"):
                    base = name.removesuffix(".gz")
                    if base in ui:
                        continue  # a twin; picked up with its plain file
                    plain, compressed = gzip.decompress(data), data
                else:
                    base, plain = name, data
                    compressed = ui.get(f"{name}.gz")
                    if compressed is not None and gzip.decompress(compressed) != plain:
                        raise ValueError(f"{name}.gz is not a gzip copy of {name}")
                content_type = _CONTENT_TYPES.get(
                    Path(base).suffix.lower(), "application/octet-stream"
                )
                assets[base] = Asset(plain, compressed, content_type, _etag(plain))
            index = Asset(shell, None, _CONTENT_TYPES[".html"], _etag(shell))
        except (OSError, ValueError, KeyError, TypeError, EOFError) as e:
            logger.warning("[Serve] Web UI bundle in {} not served: {}", directory, e)
            return None
        return cls(index, assets, manifest, _page_paths(manifest))


def _verified_files(directory: Path, manifest: object) -> dict[str, bytes]:
    """Every file of the bundle by archive path, checked against the manifest."""
    listed = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(listed, dict) or not listed:
        raise ValueError("manifest 'files' is missing")
    files: dict[str, bytes] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            continue
        name = path.relative_to(directory).as_posix()
        if name == "manifest.json":
            continue
        if not path.is_file():
            raise ValueError(f"unexpected entry {name}")
        files[name] = path.read_bytes()
    missing = sorted(set(listed) - set(files))
    if missing:
        raise ValueError(f"bundle files missing: {missing}")
    for extra in sorted(set(files) - set(listed)):
        logger.debug("[Serve] Ignoring {} (not in the bundle manifest)", extra)
        del files[extra]
    for name, data in files.items():
        entry = listed[name]
        if (
            not isinstance(entry, dict)
            or entry.get("bytes") != len(data)
            or entry.get("sha256") != hashlib.sha256(data).hexdigest()
        ):
            raise ValueError(f"{name} does not match the manifest")
    return files


def _page_paths(manifest: dict[str, object]) -> tuple[str, ...]:
    """The app's addresses: the contract's, plus any the bundle adds.

    Only plain paths outside ``/api/`` and ``/ui/`` qualify.
    """
    paths: list[str] = list(APP_PATHS)
    serve = manifest.get("serve")
    listed = serve.get("page_paths") if isinstance(serve, dict) else None
    for path in listed if isinstance(listed, list) else []:
        if (
            isinstance(path, str)
            and path.startswith("/")
            and not path.startswith(("/api/", "/ui/", "//"))
            and all(char.isalnum() or char in "/-_" for char in path)
            and path not in paths
        ):
            paths.append(path)
    return tuple(paths)


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


def serve_asset(
    asset: Asset, request: Request, *, conditional: bool = True
) -> Response:
    """Answer one asset request (200 or 304, gzip when accepted)."""
    from fastapi.responses import Response

    headers = static_headers()
    headers["ETag"] = asset.etag
    if asset.compressed is not None:
        headers["Vary"] = "Accept-Encoding"
    if conditional and _not_modified(request.headers.get("if-none-match"), asset.etag):
        return Response(status_code=304, headers=headers)
    body = asset.plain
    if asset.compressed is not None and _accepts_gzip(
        request.headers.get("accept-encoding")
    ):
        body = asset.compressed
        headers["Content-Encoding"] = "gzip"
    return Response(content=body, media_type=asset.content_type, headers=headers)


def missing_page(workbench: Workbench, request: Request) -> Response | None:
    """The workbench with status 404 for a browser asking for an unknown page.

    The UI then says the page does not exist and offers the way back
    (contract §1, v1.4); an API client, a ``/ui/`` file or another method
    keeps the JSON 404.
    """
    path = request.url.path
    page = (
        request.method in ("GET", "HEAD")
        and path != "/api"
        and not path.startswith(("/api/", "/ui/"))
        and any("text/html" in value for value in request.headers.getlist("accept"))
    )
    if not page:
        return None
    # Always the page itself, never "not modified"
    response = serve_asset(workbench.index, request, conditional=False)
    response.status_code = 404
    return response


def mount_workbench(app: FastAPI, workbench: Workbench) -> None:
    """Register the shell and ``/ui/<file>`` routes on *app*."""
    from markitai.serve.errors import ApiError

    async def shell(request: Request) -> Response:
        return serve_asset(workbench.index, request)

    for path in workbench.page_paths:
        app.add_api_route(path, shell, methods=["GET", "HEAD"], include_in_schema=False)

    async def ui_file(request: Request, name: str) -> Response:
        asset = workbench.assets.get(name)
        if asset is None:
            raise ApiError(404, "route_not_found", "route not found")
        return serve_asset(asset, request)

    app.add_api_route(
        "/ui/{name}", ui_file, methods=["GET", "HEAD"], include_in_schema=False
    )
