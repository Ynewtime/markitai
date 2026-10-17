"""The ``.env`` files markitai reads, and what a folder's ``.env`` may not set.

A working directory may come with a ``.env`` its user did not write (a
download, a cloned repository). Keys and options still come from it; names
that choose programs, search paths, endpoints, proxies, home or configuration
folders, credential files, loader or interpreter hooks and certificate
bundles do not. The same rule as the Rust implementation.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

_EXACT = frozenset(
    {
        "PATH",
        "PATHEXT",
        "HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "TMPDIR",
        "TMP",
        "TEMP",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "CLOUDSDK_CONFIG",
        "MARKITAI_CONFIG",
        "MARKITAI_HOME",
        "MARKITAI_LOG_DIR",
        "NODE_OPTIONS",
        "BASH_ENV",
        "ENV",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "AWS_CA_BUNDLE",
    }
)
_PREFIXES = ("LD_", "DYLD_", "XDG_", "PYTHON")
_SUFFIXES = (
    "_PATH",
    "_EXECUTABLE",
    "_API_BASE",
    "_BASE_URL",
    "_ENDPOINT",
    "_HOME",
    "_CONFIG_DIR",
)

_warned: set[str] = set()


def project_may_not_set(name: str) -> bool:
    """Whether a working directory's ``.env`` may not set *name*."""
    upper = name.strip().upper()
    return (
        upper in _EXACT
        or upper.startswith(_PREFIXES)
        or upper.endswith(_SUFFIXES)
        or (upper.startswith("AWS_") and upper.endswith("_FILE"))
    )


def merge_env_files(
    environ: Mapping[str, str],
    project: Iterable[tuple[str, str | None]],
    *,
    trusted_here: bool,
    own: Iterable[tuple[str, str | None]],
) -> tuple[dict[str, str], list[str]]:
    """The values to add to *environ*: the folder's *project* pairs (without
    the refused names unless *trusted_here*), then the user's *own*; values
    already present win. Also the refused names, sorted."""
    added: dict[str, str] = {}
    ignored: set[str] = set()
    for key, value in project:
        if value is None:
            continue
        if not trusted_here and project_may_not_set(key):
            ignored.add(key)
            continue
        if key not in environ and key not in added:
            added[key] = value
    for key, value in own:
        if value is not None and key not in environ and key not in added:
            added[key] = value
    return added, sorted(ignored)


def load_env_files(cwd: Path | None = None, home: Path | None = None) -> list[str]:
    """Load ``<cwd>/.env`` (filtered) and ``~/.markitai/.env`` into
    ``os.environ`` without replacing a value already set; warn once on
    stderr about the names the folder's file may not set. Returns them."""
    from dotenv import dotenv_values

    project_file = (cwd or Path.cwd()) / ".env"
    own_file = (home or Path.home()) / ".markitai" / ".env"
    try:
        trusted_here = project_file.resolve() == own_file.resolve()
    except OSError:
        trusted_here = False

    def pairs(path: Path) -> list[tuple[str, str | None]]:
        return list(dotenv_values(path).items()) if path.is_file() else []

    added, ignored = merge_env_files(
        os.environ,
        pairs(project_file),
        trusted_here=trusted_here,
        own=pairs(own_file),
    )
    os.environ.update(added)
    if ignored:
        message = (
            f"Warning: Ignored {', '.join(ignored)} from ./.env: a folder's .env "
            "may not set programs, paths, endpoints, proxies, home folders or "
            f"credential files; set them in your environment or in {own_file}"
        )
        if message not in _warned:
            _warned.add(message)
            print(message, file=sys.stderr)
    return ignored
