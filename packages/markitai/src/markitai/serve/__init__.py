"""Optional web server exposing the conversion core over REST + SSE.

The heavy dependencies (fastapi, uvicorn, python-multipart) are an optional
extra (``markitai[serve]``); nothing in this package imports them at module
level so that importing :mod:`markitai.serve` stays safe on base installs.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from fastapi import FastAPI

    from markitai.config import MarkitaiConfig

SERVE_INSTALL_HINT = (
    "fastapi, uvicorn and python-multipart are required for the serve "
    "command but are not installed. Install them with: "
    'uv tool install "markitai[serve]" (or: pip install "markitai[serve]")'
)


def is_serve_available() -> bool:
    """Return whether the optional serve dependencies are installed."""
    if find_spec("fastapi") is None or find_spec("uvicorn") is None:
        return False
    # python-multipart>=0.0.13 installs as python_multipart; older as multipart
    return (
        find_spec("python_multipart") is not None or find_spec("multipart") is not None
    )


def create_app(
    static_dir: Path | None = None,
    jobs_root: Path | None = None,
    config: MarkitaiConfig | None = None,
    configure_logging: bool = True,
    config_path: Path | None = None,
    allowed_hosts: Sequence[str] | None = None,
) -> FastAPI:
    """Create the FastAPI app (guarded re-export of :func:`serve.app.create_app`).

    Args:
        static_dir: Directory with a built web UI. Auto-detected when None.
        jobs_root: Directory holding per-job workspaces. Defaults to
            ``~/.markitai/serve/jobs``.
        config: Base configuration. Loaded via ConfigManager when None.
        configure_logging: Whether the app lifespan takes over loguru.
        config_path: Config file the ``/api/settings/llm/models`` endpoints
            write to. Defaults to ``~/.markitai/config.json``.
        allowed_hosts: Extra hostnames accepted in the Host and Origin
            headers besides localhost and IP literals.

    Raises:
        ImportError: When the serve extra is not installed.
    """
    try:
        from markitai.serve.app import create_app as _create_app
    except ImportError as e:
        raise ImportError(SERVE_INSTALL_HINT) from e
    return _create_app(
        static_dir=static_dir,
        jobs_root=jobs_root,
        config=config,
        configure_logging=configure_logging,
        config_path=config_path,
        allowed_hosts=allowed_hosts,
    )
