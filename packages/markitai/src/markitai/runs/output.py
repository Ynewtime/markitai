"""Shared output-target skeleton for single-input runs.

Both single-input processors (single file and single URL in
``markitai.cli.processors``) used to carry near-identical inline copies of
the output scaffolding: interpreting an ``-o`` value that may name a file,
switching to a temp directory in stdout mode, preparing the output
directory, moving the final content onto an explicit ``-o`` file target,
and resolving ``.markitai/`` asset references for stdout output. This
module single-sources that skeleton; the path-specific pipelines (document
conversion, URL fetch/LLM chains) stay in the CLI processors.

Layering: ``markitai.runs`` sits below ``markitai.cli`` and must never
import it (enforced by the import-linter contracts in the root
pyproject.toml).
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from loguru import logger

from markitai.security import check_symlink_safety
from markitai.utils.paths import ensure_dir


def split_output_file_target(output: Path) -> tuple[Path, str | None]:
    """Interpret an ``-o`` value that looks like a markdown file path.

    For single-file/single-URL conversions, ``-o out.md`` means "write the
    output exactly to this file" rather than "create a directory named
    out.md". Returns ``(output_dir, output_name)``:

    - ``-o out.md`` (not an existing directory) -> ``(Path("."), "out.md")``
    - ``-o results/out.md`` -> ``(Path("results"), "out.md")``
    - ``-o out.md`` where a *directory* named ``out.md`` exists -> treated
      as a directory: ``(Path("out.md"), None)``
    - any other value -> ``(output, None)``
    """
    if output.suffix == ".md" and not output.is_dir():
        return output.parent, output.name
    return output, None


@dataclass
class OutputTarget:
    """Resolved output destination for a single-input run.

    Attributes:
        output_dir: Post-split output directory (``None`` means stdout mode).
        explicit_file_name: File name from a file-like ``-o`` value
            (e.g. ``-o out.md``), else ``None``.
        stdout_mode: True when no output location was given: the payload
            goes to stdout and intermediate files live in a temp directory.
        temp_dir: stdout-mode temp working directory; created by
            :meth:`create_workdir`, removed by :meth:`cleanup`.
    """

    output_dir: Path | None
    explicit_file_name: str | None
    stdout_mode: bool
    temp_dir: Path | None = None

    def create_workdir(
        self, *, ensure: bool = False, allow_symlinks: bool = False
    ) -> Path:
        """Create and return the effective working directory for the run.

        In stdout mode this creates (and remembers) a fresh temp directory;
        otherwise it returns ``output_dir`` unchanged. With ``ensure=True``
        the directory is symlink-checked and created eagerly — the URL path
        needs this, while the file path defers directory creation to
        ``convert_document_core``.

        Call this after the dry-run early exit so dry runs never create a
        temp directory.
        """
        if self.stdout_mode:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="markitai_"))
            effective = self.temp_dir
        else:
            assert self.output_dir is not None  # not stdout mode
            effective = self.output_dir
        if ensure:
            check_symlink_safety(effective, allow_symlinks=allow_symlinks)
            ensure_dir(effective)
        return effective

    def cleanup(self) -> None:
        """Remove the stdout-mode temp directory (no-op in file mode)."""
        if self.temp_dir is not None:
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def prepare_output_target(output: Path | None) -> OutputTarget:
    """Resolve an ``-o`` CLI value into an :class:`OutputTarget`.

    Splits a file-like ``-o`` value into directory + explicit file name and
    determines stdout mode (no output location at all). Filesystem work is
    deferred to :meth:`OutputTarget.create_workdir` so the dry-run path can
    exit before anything is created.
    """
    output_dir: Path | None = output
    explicit_file_name: str | None = None
    if output_dir is not None:
        output_dir, explicit_file_name = split_output_file_target(output_dir)
    return OutputTarget(
        output_dir=output_dir,
        explicit_file_name=explicit_file_name,
        stdout_mode=output_dir is None,
    )


def finalize_explicit_output(result_file: Path, explicit_target: Path | None) -> Path:
    """Move the final content onto an explicit ``-o`` file target.

    With an explicit ``-o`` file target the final content must land exactly
    at the requested path. In LLM mode (without --keep-base) the final file
    is ``<name>.llm.md``; move it onto the requested ``.md`` path. Pass
    ``explicit_target=None`` when no explicit file target was requested
    (the call is then a no-op).

    Returns:
        The path where the final content now lives.
    """
    if (
        explicit_target is not None
        and result_file != explicit_target
        and result_file.exists()
        and not explicit_target.exists()
    ):
        result_file.replace(explicit_target)
        return explicit_target
    return result_file


# Pattern matches markdown image references to .markitai/assets/ or
# .markitai/screenshots/. Supports both forward slash and backslash for
# Windows compatibility.
# Groups: 1=alt text, 2=subdir (assets|screenshots), 3=filename
ASSET_REF_PATTERN = re.compile(
    r"!\[([^\]]*)\]\(\.markitai[/\\](assets|screenshots)[/\\]([^)]+)\)"
)


def warn_ephemeral_links() -> None:
    """Warn on stderr that stdout image links will not outlive the process.

    Used when ``image.stdout_persist`` is explicitly disabled. Writes to
    stderr directly because the stdout-mode console handler only shows
    ERROR+, and stdout itself must stay clean for the markdown content.
    """
    message = (
        "Warning: image.stdout_persist is disabled — extracted images are "
        "deleted at exit, so image links in the output are ephemeral"
    )
    print(message, file=sys.stderr)
    logger.warning(message)


def normalize_temp_asset_refs(markdown: str, temp_dir: Path) -> str:
    """Rewrite absolute temp-dir asset refs to relative ``.markitai/`` refs.

    Some converters emit image refs with absolute paths into the stdout-mode
    temp directory (e.g. pymupdf4llm canonicalizes macOS ``/var/...`` temp
    paths to ``/private/var/...``, defeating the converter's own relative
    rewrite). Normalizing here lets ``resolve_asset_references`` handle them
    instead of leaking dead temp links to stdout.

    Args:
        markdown: Markdown content possibly containing absolute asset refs.
        temp_dir: The stdout-mode temp directory used for conversion.

    Returns:
        Markdown with absolute temp-dir asset refs rewritten to relative form.
    """
    for base in {temp_dir.as_posix(), temp_dir.resolve().as_posix()}:
        markdown = markdown.replace(f"]({base}/.markitai/", "](.markitai/")
    return markdown


def resolve_asset_references(
    markdown: str,
    temp_dir: Path,
    protocol: Any = None,
    asset_store: Any = None,
    source_name: str = "unknown",
) -> str:
    """Resolve .markitai/assets/ and .markitai/screenshots/ image references.

    Priority cascade:
    1. If protocol is set: replace with terminal inline image escape sequence.
    2. If asset_store is set: persist image, replace with absolute-path URI.
    3. Fallback: replace with ``![image: filename]()`` placeholder.

    Args:
        markdown: Markdown content with asset references.
        temp_dir: Path to the temp directory containing .markitai/ assets.
        protocol: Detected terminal image protocol, or None.
        asset_store: Configured asset store, or None.
        source_name: Source document name for asset store grouping.

    Returns:
        Markdown with asset references resolved.
    """

    def _resolve_image_path(subdir: str, filename: str) -> Path:
        """Resolve the actual image file path from captured regex groups."""
        filename_normalized = filename.replace("\\", "/")
        return temp_dir / ".markitai" / subdir / filename_normalized

    def _replace(match: re.Match[str]) -> str:
        subdir = match.group(2)  # "assets" or "screenshots" — captured group
        filename = unquote(match.group(3))

        if protocol is not None:
            # Tier 1: terminal inline image
            image_path = _resolve_image_path(subdir, filename)
            if image_path.exists():
                from markitai.utils.terminal_image import render_inline_image

                try:
                    return render_inline_image(image_path, protocol)
                except Exception:
                    pass  # fall through

        if asset_store is not None:
            # Tier 2: persistent asset store
            image_path = _resolve_image_path(subdir, filename)
            if image_path.exists():
                try:
                    ref_path = asset_store.save(image_path, source_name)
                    uri = asset_store.ref_path_to_markdown_uri(ref_path)
                    # Keep LLM-generated alt text; fall back to the filename
                    alt_text = match.group(1) or filename
                    return f"![{alt_text}]({uri})"
                except Exception as e:
                    logger.warning(f"Asset store save failed for {filename}: {e}")
                    # fall through to placeholder

        # Tier 3: placeholder fallback
        return f"![image: {filename}]()"

    return ASSET_REF_PATTERN.sub(_replace, markdown)
