"""Output profiles: reshape written outputs for downstream consumers.

Profiles are orthogonal to presets: presets decide which features run
(LLM, OCR, screenshots, ...), a profile decides what the written output
looks like. No profile means the pipeline output is byte-identical to the
default behavior — every transform in this module runs only when
``config.output.profile`` is set.

Supported profiles:

- ``rag``: relocate images from the hidden ``.markitai/assets/`` directory
  to a visible ``assets/`` directory (hidden paths are skipped by common
  ingestors such as LlamaIndex ``SimpleDirectoryReader``), rewrite PDF page
  markers to ``<!-- page: N -->``, and warn on pipe tables with
  inconsistent column counts.
- ``obsidian``: relocate assets like ``rag``; optionally rewrite local
  image references to wikilinks (``![[assets/x.png]]``) when
  ``output.wikilinks`` is enabled.
- ``okf``: rename frontmatter fields to align with the Open Knowledge
  Format spec (https://github.com/GoogleCloudPlatform/knowledge-catalog):
  ``type`` is injected, ``source`` becomes ``resource``,
  ``markitai_processed`` becomes ``generated: {by, at}``. Fields without
  an OKF equivalent keep their current names — the spec explicitly allows
  unknown keys.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger

from markitai.constants import (
    ASSETS_REL_PATH,
    MARKITAI_META_DIR,
    VISIBLE_ASSETS_REL_PATH,
)
from markitai.security import atomic_write_text

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig

# Package version for the OKF generated.by actor. Read from installed
# metadata (hatch single-sources it from markitai.__init__) instead of the
# markitai package root: importing the root would pull the lazy api exports
# into this domain module and break the import-linter layering contract.
try:
    __version__ = metadata.version("markitai")
except metadata.PackageNotFoundError:  # pragma: no cover — not installed
    __version__ = "0.0.0"

# Profiles that move assets out of the hidden metadata directory
ASSET_VISIBLE_PROFILES: tuple[str, ...] = ("rag", "obsidian")

# Extra cleaning rules injected into the text-cleaning prompts under the
# rag profile. Appending an empty string keeps prompts (and cache keys)
# unchanged, so the default path is untouched.
RAG_TABLE_PROMPT_RULES = (
    "\n\n## Table Column Consistency — CRITICAL\n"
    "- Every row of a pipe table (including the delimiter row) MUST have "
    "exactly the same number of columns as its header row\n"
    "- Pad missing cells with empty cells; never merge or drop cells to "
    "make a row fit"
)

# <!-- Page number: N --> markers written by the PDF converter at pymupdf
# page boundaries (converter/pdf.py joins page_chunks with these markers)
_PAGE_NUMBER_MARKER_RE = re.compile(r"<!--\s*Page number:\s*(\d+)\s*-->")

# ![alt](assets/name) references after asset relocation
_VISIBLE_IMAGE_REF_RE = re.compile(
    rf"!\[((?:[^\]\\]|\\.)*)\]\({re.escape(VISIBLE_ASSETS_REL_PATH)}/([^)]+)\)"
)

_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")

_DELIMITER_CELL_RE = re.compile(r"^:?-+:?$")


def _split_frontmatter(content: str) -> tuple[str | None, str]:
    """Split markdown into a frontmatter block (without fences) and body."""
    if content.startswith("---\n"):
        closing = content.find("\n---\n", 4)
        if closing != -1:
            return content[4:closing], content[closing + 5 :].lstrip("\n")
    return None, content


def _dump_frontmatter(data: dict[str, Any]) -> str:
    """Serialize a frontmatter dict to YAML without fences."""
    return yaml.safe_dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=1000,
    ).strip()


def _to_utc_timestamp(value: Any) -> str | None:
    """Convert an ISO 8601 timestamp to the OKF UTC ``...Z`` form."""
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def okf_frontmatter(frontmatter: dict[str, Any]) -> dict[str, Any]:
    """Map markitai frontmatter fields onto OKF field names.

    Alignment verified against the OKF spec (v0.2): ``type`` is the only
    required field, ``title``/``description``/``tags`` match by name,
    ``source`` maps to ``resource`` (canonical identifier of the
    underlying asset), and ``markitai_processed`` maps to
    ``generated: {by, at}`` using the spec's ``<producer>/<version>``
    actor convention. All other markitai fields are kept under their
    current names — consumers "MUST NOT reject documents with
    unrecognized fields".

    Args:
        frontmatter: Parsed markitai frontmatter (may be empty).

    Returns:
        A new dict in OKF field order.
    """
    mapped_keys = {"title", "description", "source", "tags", "markitai_processed"}

    result: dict[str, Any] = {"type": "Document"}
    for key in ("title", "description"):
        if key in frontmatter:
            result[key] = frontmatter[key]
    if "source" in frontmatter:
        result["resource"] = frontmatter["source"]
    if "tags" in frontmatter:
        result["tags"] = frontmatter["tags"]

    generated: dict[str, Any] = {"by": f"markitai/{__version__}"}
    generated_at = _to_utc_timestamp(frontmatter.get("markitai_processed"))
    if generated_at is not None:
        generated["at"] = generated_at
    result["generated"] = generated

    for key, value in frontmatter.items():
        if key not in mapped_keys:
            result[key] = value
    return result


def _count_table_cells(line: str) -> int:
    """Count cells in one pipe-table row (escaped pipes excluded)."""
    stripped = line.strip().replace("\\|", "\x00")
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return len(stripped.split("|"))


def _is_delimiter_row(line: str) -> bool:
    """Check whether a table row is a header delimiter (``|---|---|``)."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    return all(_DELIMITER_CELL_RE.fullmatch(cell) for cell in cells if cell != "") and (
        any(cell for cell in cells)
    )


def table_column_warnings(markdown: str) -> list[str]:
    """Find pipe tables whose rows disagree with the header column count.

    Detection only — the content is never rewritten. Tables are runs of
    consecutive lines starting with ``|`` outside fenced code blocks; the
    first row of a run is treated as the header.

    Args:
        markdown: Markdown body to scan.

    Returns:
        One human-readable warning per inconsistent table.
    """
    warnings: list[str] = []
    lines = markdown.splitlines()
    in_fence = False
    table_start: int | None = None
    header_cells = 0
    mismatches: list[tuple[int, int]] = []  # (line number, cell count)

    def flush() -> None:
        nonlocal table_start, mismatches
        if table_start is not None and mismatches:
            detail = ", ".join(f"line {n}: {c}" for n, c in mismatches[:3])
            warnings.append(
                f"pipe table at line {table_start} has {header_cells} header "
                f"column(s) but inconsistent row(s) ({detail})"
            )
        table_start = None
        mismatches = []

    for lineno, line in enumerate(lines, start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            flush()
            continue
        if not in_fence and line.lstrip().startswith("|"):
            cells = _count_table_cells(line)
            if table_start is None:
                table_start = lineno
                header_cells = cells
            elif cells != header_cells and not _is_delimiter_row(line):
                mismatches.append((lineno, cells))
        else:
            flush()
    flush()
    return warnings


def _relocate_referenced_assets(body: str, output_dir: Path) -> str:
    """Move referenced hidden assets to ``assets/`` and rewrite references.

    Files are moved from ``<output_dir>/.markitai/assets/`` to
    ``<output_dir>/assets/``; a file another output of the same run already
    moved is only re-referenced. Emptied hidden directories are removed.

    Args:
        body: Markdown body containing ``.markitai/assets/`` references.
        output_dir: Directory the markdown file lives in.

    Returns:
        Body with references rewritten to the visible directory.
    """
    from markitai.utils.text import extract_asset_image_names

    hidden_dir = output_dir / ASSETS_REL_PATH
    visible_dir = output_dir / VISIBLE_ASSETS_REL_PATH

    for name in extract_asset_image_names(body):
        src = hidden_dir / name
        dst = visible_dir / name
        if not src.is_file():
            continue
        visible_dir.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            # Same-run sibling output already moved this file
            src.unlink()
        else:
            src.rename(dst)

    _prune_empty_meta_dirs(output_dir)

    body = body.replace(f"]({ASSETS_REL_PATH}/", f"]({VISIBLE_ASSETS_REL_PATH}/")
    return body.replace(f"]({ASSETS_REL_PATH}\\", f"]({VISIBLE_ASSETS_REL_PATH}\\")


def _prune_empty_meta_dirs(output_dir: Path) -> None:
    """Remove the hidden assets dir (and ``.markitai``) once emptied."""
    for directory in (output_dir / ASSETS_REL_PATH, output_dir / MARKITAI_META_DIR):
        try:
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        except OSError as e:
            logger.debug("[Profile] Could not prune {}: {}", directory, e)


def _unescape_image_alt(alt: str) -> str:
    """Reverse the escaping applied by ``markdown_image_reference``."""
    return alt.replace("\\[", "[").replace("\\]", "]").replace("\\\\", "\\")


def _to_wikilinks(body: str) -> str:
    """Rewrite visible-asset image references to Obsidian wikilinks."""
    from urllib.parse import unquote

    def replace(match: re.Match[str]) -> str:
        alt = _unescape_image_alt(match.group(1)).strip()
        target = f"{VISIBLE_ASSETS_REL_PATH}/{unquote(match.group(2))}"
        return f"![[{target}|{alt}]]" if alt else f"![[{target}]]"

    return _VISIBLE_IMAGE_REF_RE.sub(replace, body)


def _rewrite_page_markers(body: str) -> str:
    """Rewrite converter page markers to ``<!-- page: N -->``.

    The source markers are written by the PDF converter at pymupdf page
    boundaries, so the rewritten markers stay exactly aligned with the
    original pagination — no positions are guessed.
    """
    return _PAGE_NUMBER_MARKER_RE.sub(lambda m: f"<!-- page: {m.group(1)} -->", body)


def visible_asset_names(markdown: str) -> list[str]:
    """Extract basenames of ``assets/`` image refs from profiled markdown.

    Counterpart of ``utils.text.extract_asset_image_names`` for outputs
    written under an asset-visible profile.

    Args:
        markdown: Markdown content with image references.

    Returns:
        Referenced asset basenames in document order, deduplicated.
    """
    from urllib.parse import unquote

    names: list[str] = []
    for match in _VISIBLE_IMAGE_REF_RE.finditer(markdown):
        name = unquote(match.group(2)).replace("\\", "/").split("/")[-1]
        if name and name not in names:
            names.append(name)
    return names


def relocate_analysis_asset_path(path_str: str) -> str:
    """Map an image-analysis asset path from the hidden dir to the visible one.

    Used by ``write_images_json`` under asset-visible profiles: analysis
    entries record absolute paths captured before relocation.

    Args:
        path_str: Absolute asset path as recorded by image analysis.

    Returns:
        The corresponding visible path, or the input unchanged when it is
        not under a ``.markitai/assets`` directory.
    """
    path = Path(path_str)
    parent = path.parent
    if parent.name == "assets" and parent.parent.name == MARKITAI_META_DIR:
        return str(parent.parent.parent / VISIBLE_ASSETS_REL_PATH / path.name)
    return path_str


def assets_visible(config: MarkitaiConfig) -> bool:
    """Return whether the active profile relocates assets to ``assets/``."""
    return config.output.profile in ASSET_VISIBLE_PROFILES


def extra_cleaning_rules(config: MarkitaiConfig) -> str:
    """Return profile-specific rules appended to text-cleaning prompts.

    Empty for every profile except ``rag``, so default prompts (and their
    cache keys) stay byte-identical.
    """
    if config.output.profile == "rag":
        return RAG_TABLE_PROMPT_RULES
    return ""


def apply_profile_to_file(
    md_file: Path,
    output_dir: Path,
    config: MarkitaiConfig,
) -> None:
    """Post-process one written markdown file for the active output profile.

    No-op when ``config.output.profile`` is unset or the file is missing.
    The file is rewritten atomically only when its content changed.

    Args:
        md_file: Written ``.md`` or ``.llm.md`` file.
        output_dir: Directory the file was written into (asset dir anchor).
        config: Effective configuration carrying ``output.profile``.
    """
    profile = config.output.profile
    if profile is None or not md_file.is_file():
        return

    content = md_file.read_text(encoding="utf-8")
    frontmatter_text, body = _split_frontmatter(content)

    if profile in ASSET_VISIBLE_PROFILES:
        body = _relocate_referenced_assets(body, output_dir)

    if profile == "rag":
        body = _rewrite_page_markers(body)
        for warning in table_column_warnings(body):
            logger.warning("[Profile:rag] {}: {}", md_file.name, warning)
    elif profile == "obsidian":
        if config.output.wikilinks:
            body = _to_wikilinks(body)
    elif profile == "okf":
        if frontmatter_text is None:
            # OKF conformance requires frontmatter with a type on every file
            frontmatter_text = _dump_frontmatter(okf_frontmatter({}))
        else:
            try:
                loaded = yaml.safe_load(frontmatter_text)
            except yaml.YAMLError as e:
                logger.warning(
                    "[Profile:okf] {}: frontmatter is not valid YAML, "
                    "leaving it unchanged: {}",
                    md_file.name,
                    e,
                )
                loaded = None
            if isinstance(loaded, dict):
                frontmatter_text = _dump_frontmatter(okf_frontmatter(loaded))

    if frontmatter_text is None:
        new_content = body
    else:
        new_content = f"---\n{frontmatter_text}\n---\n\n{body}"

    if new_content != content:
        atomic_write_text(md_file, new_content)
        logger.debug("[Profile:{}] Rewrote {}", profile, md_file.name)
