"""``-f json``: the document JSON of a published result, beside its Markdown
(markitai-rust's ``docs/json-output.md``).

The Markdown files stay the result that resume, reports, history and the
web server own; the JSON is derived from them once they are written, under
the same name (``report.pdf.md`` / ``report.pdf.llm.md`` ->
``report.pdf.json``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit

import yaml

from markitai.constants import (
    ASSETS_REL_PATH,
    SCREENSHOTS_REL_PATH,
    VISIBLE_ASSETS_REL_PATH,
)
from markitai.document_json import (
    INDEX_NAME,
    SCHEMA_VERSION,
    DocumentParts,
    build_document,
    render,
)

if TYPE_CHECKING:
    from markitai.config import MarkitaiConfig
    from markitai.workflow.core import ConversionContext

__all__ = [
    "INDEX_NAME",
    "IndexItem",
    "context_parts",
    "json_path_beside",
    "publish",
    "published_parts",
    "skipped_parts",
    "url_parts",
    "wanted",
    "write_beside",
    "write_single",
    "write_index",
]


def wanted(config: MarkitaiConfig) -> bool:
    """Whether the run asked for the document JSON (``output.format``)."""
    return config.output.format == "json"


def json_path_beside(
    output_path: Path | None, llm_output_path: Path | None = None
) -> Path | None:
    """``<stem>.json`` beside the Markdown a result published, or None when
    it published none (a skip)."""
    if output_path is not None and output_path.name.endswith(".md"):
        name = output_path.name.removesuffix(".llm.md").removesuffix(".md")
        return output_path.with_name(f"{name}.json")
    if llm_output_path is not None:
        name = llm_output_path.name
        for suffix in (".llm.md", ".md"):
            if name.endswith(suffix):
                return llm_output_path.with_name(f"{name.removesuffix(suffix)}.json")
    return None


class _StringDates(yaml.SafeLoader):
    """Front matter as written: a date stays the text the file has."""


_StringDates.yaml_implicit_resolvers = {
    key: [
        (tag, regex) for tag, regex in resolvers if tag != "tag:yaml.org,2002:timestamp"
    ]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _read_published(path: Path) -> tuple[dict[str, Any], str]:
    """``(front matter, body)`` of a written Markdown file."""
    from markitai.utils.frontmatter import split_frontmatter

    text = path.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter(text)
    data: Any = None
    if frontmatter is not None:
        try:
            data = yaml.load(frontmatter, Loader=_StringDates)  # noqa: S506 - SafeLoader subclass
        except yaml.YAMLError:
            data = None
    return (dict(data) if isinstance(data, dict) else {}), body


# Pictures a body links, visibly or in a comment (a reference picture a
# converter demoted)
_LINKED = re.compile(r"!\[(?:[^\]\\]|\\.)*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_WIKILINK = re.compile(r"!\[\[([^|\]]+)(?:\|[^\]]*)?\]\]")


def _linked_files(
    bodies: Iterable[str], output_dir: Path
) -> tuple[list[Path], list[Path]]:
    """The asset and screenshot files the bodies link that exist on disk."""
    assets: list[Path] = []
    shots: list[Path] = []
    for body in bodies:
        targets = [m.group(1) for m in _LINKED.finditer(body)]
        targets += [m.group(1) for m in _WIKILINK.finditer(body)]
        for target in targets:
            if target.startswith("file://"):
                from urllib.request import url2pathname

                path = Path(url2pathname(urlsplit(target).path))
            elif "://" in target or target.startswith(("data:", "#", "/")):
                continue
            else:
                path = output_dir / unquote(target)
            relative = target.replace("\\", "/")
            is_shot = f"{SCREENSHOTS_REL_PATH}/" in relative
            is_asset = relative.startswith(
                (f"{ASSETS_REL_PATH}/", f"{VISIBLE_ASSETS_REL_PATH}/")
            ) or (target.startswith("file://") and not is_shot)
            if not (is_shot or is_asset) or not path.is_file():
                continue
            found = shots if is_shot else assets
            if path not in found:
                found.append(path)
    return assets, shots


def own_screenshots(output_dir: Path, prefix: str) -> list[Path]:
    """The page and slide screenshots a result named *prefix* wrote."""
    folder = output_dir / SCREENSHOTS_REL_PATH
    pattern = re.compile(rf"^{re.escape(prefix)}\.(?:page|slide)\d+\.[A-Za-z0-9]+$")
    try:
        return sorted(
            p for p in folder.iterdir() if pattern.match(p.name) and p.is_file()
        )
    except OSError:
        return []


def published_parts(
    source: str,
    *,
    output_dir: Path,
    base_file: Path | None,
    llm_file: Path | None,
    base_markdown: str | None = None,
    warnings: Sequence[str] = (),
    llm_usage: Mapping[str, Mapping[str, Any]] | None = None,
    analysis: Sequence[Mapping[str, Any]] = (),
    duration_s: float = 0.0,
    screenshots: Sequence[Path] = (),
    source_label: str | None = None,
    source_name: str | None = None,
    config: MarkitaiConfig | None = None,
) -> DocumentParts:
    """The parts of a document whose Markdown was published.

    *base_file* is the base ``.md`` this run wrote (None when the LLM output
    alone was kept); *base_markdown* the converted body in that case, with
    or without front matter. The front matter is the final file's.
    """
    from markitai.utils.frontmatter import split_frontmatter

    frontmatter: dict[str, Any] = {}
    base = ""
    if base_file is not None and base_file.is_file():
        frontmatter, base = _read_published(base_file)
    elif base_markdown is not None:
        base = split_frontmatter(base_markdown)[1]
        if config is not None:
            from markitai.output_profiles import profile_body

            base = profile_body(base, config)
    llm = None
    if llm_file is not None and llm_file.is_file():
        frontmatter, llm = _read_published(llm_file)
    assets, shots = _linked_files([b for b in (base, llm) if b], output_dir)
    for path in screenshots:
        if path not in shots and path.is_file():
            shots.append(path)
    prefix = None
    for published in (base_file, llm_file):
        if published is not None:
            from markitai.utils.output import split_markdown_name

            prefix = split_markdown_name(published.name)[0]
            break
    if prefix:
        for path in own_screenshots(output_dir, prefix):
            if path not in shots:
                shots.append(path)
    return DocumentParts(
        source=source,
        base=base,
        llm=llm,
        frontmatter=frontmatter,
        source_label=source_label,
        source_name=source_name or prefix,
        warnings=list(warnings),
        llm_usage=dict(llm_usage or {}),
        analysis=list(analysis),
        assets=assets,
        screenshots=shots,
        output_path=base_file
        if base_file is not None and base_file.is_file()
        else None,
        llm_output_path=llm_file
        if llm_file is not None and llm_file.is_file()
        else None,
        duration_s=duration_s,
    )


def context_parts(
    ctx: ConversionContext,
    *,
    warnings: Sequence[str] | None = None,
    duration_s: float = 0.0,
    source_label: str | None = None,
) -> DocumentParts:
    """The parts of a file conversion that finished (not skipped)."""
    from markitai.workflow.core import _base_markdown_content

    enhanced = ctx.llm_output_file if not ctx.llm_fell_back else None
    base_file = ctx.output_file if ctx.base_written else None
    base_markdown = None
    if base_file is None and ctx.conversion_result is not None:
        base_markdown = _base_markdown_content(ctx)
    analysis = ctx.image_analysis.assets if ctx.image_analysis is not None else []
    return published_parts(
        str(ctx.input_path),
        output_dir=ctx.output_dir,
        base_file=base_file,
        llm_file=enhanced,
        base_markdown=base_markdown,
        warnings=list(ctx.warnings) if warnings is None else warnings,
        llm_usage=ctx.llm_usage,
        analysis=analysis,
        duration_s=duration_s,
        source_label=source_label,
        config=ctx.config,
    )


def url_parts(
    url: str,
    config: MarkitaiConfig,
    output_file: Path,
    *,
    llm_fell_back: bool = False,
    llm_file: Path | None = None,
    base_markdown: str | None = None,
    analysis: Sequence[Mapping[str, Any]] = (),
    screenshots: Sequence[Path | None] = (),
    warnings: Sequence[str] = (),
    llm_usage: Mapping[str, Mapping[str, Any]] | None = None,
    duration_s: float = 0.0,
) -> DocumentParts:
    """The parts of a URL conversion published as *output_file* (its base
    ``.md`` name, written or not) and its ``.llm.md``."""
    from markitai.utils.output import split_markdown_name

    base_written = output_file.is_file() and (
        not config.llm.enabled or config.llm.keep_base or llm_fell_back
    )
    if llm_file is None and config.llm.enabled and not llm_fell_back:
        candidate = output_file.with_suffix(".llm.md")
        llm_file = candidate if candidate.is_file() else None
    return published_parts(
        url,
        output_dir=output_file.parent,
        base_file=output_file if base_written else None,
        llm_file=None if llm_fell_back else llm_file,
        base_markdown=None if base_written else base_markdown,
        warnings=warnings,
        llm_usage=llm_usage,
        analysis=analysis,
        duration_s=duration_s,
        screenshots=[path for path in screenshots if path is not None],
        source_name=split_markdown_name(output_file.name)[0],
        config=config,
    )


def skipped_parts(source: str, reason: str, duration_s: float = 0.0) -> DocumentParts:
    """A skipped input is still a document, with ``status: "skipped"``."""
    return DocumentParts(source=source, skip_reason=reason, duration_s=duration_s)


def write_beside(
    parts: DocumentParts, json_path: Path, *, allow_symlinks: bool = False
) -> Path:
    """Write the document of *parts* to *json_path* (paths relative to its
    folder). Raises OSError when it cannot be written."""
    from markitai.security import atomic_write_text

    document = build_document(parts, json_path.parent)
    atomic_write_text(json_path, render(document), follow_symlinks=allow_symlinks)
    return json_path


def publish(
    parts: DocumentParts,
    *,
    warnings: Sequence[str],
    duration_s: float,
    llm_usage: Mapping[str, Mapping[str, Any]] | None = None,
    allow_symlinks: bool = False,
) -> Path | None:
    """Write a batch item's document beside its Markdown, with the item's
    final warnings and duration. Raises OSError with the sentence a failed
    item shows when it cannot be written."""
    from dataclasses import replace

    target = json_path_beside(parts.output_path, parts.llm_output_path)
    if target is None:
        return None
    try:
        return write_beside(
            replace(
                parts,
                warnings=list(warnings),
                duration_s=duration_s,
                llm_usage=dict(parts.llm_usage if llm_usage is None else llm_usage),
            ),
            target,
            allow_symlinks=allow_symlinks,
        )
    except OSError as e:
        raise OSError(f"Cannot write the JSON document {target}: {e}") from e


def write_single(
    parts: DocumentParts | None,
    produced: Path,
    final: Path,
    *,
    allow_symlinks: bool = False,
) -> Path | None:
    """Write a single input's document beside its published Markdown and
    return its path (None without ``-f json`` or Markdown). *final* is where
    *produced* ended up (an explicit ``-o report.md`` moves the LLM output
    there). Raises ConversionError when it cannot be written."""
    from dataclasses import replace

    from markitai.utils.errors import ConversionError

    if parts is None:
        return None
    if final != produced:
        parts = replace(parts, llm_output_path=final)
    target = json_path_beside(parts.output_path, parts.llm_output_path)
    if target is None:
        return None
    try:
        return write_beside(parts, target, allow_symlinks=allow_symlinks)
    except OSError as e:
        raise ConversionError(f"Cannot write the JSON document {target}: {e}") from e


@dataclass
class IndexItem:
    """One item of ``markitai.index.json``."""

    source: str
    status: str
    document: str | None
    markdown: str | None
    error: str | None
    skip_reason: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status,
            "document": self.document,
            "markdown": self.markdown,
            "error": self.error,
            "skip_reason": self.skip_reason,
        }


def _relative(path: Path, directory: Path) -> str:
    try:
        return "/".join(path.resolve().relative_to(directory.resolve()).parts)
    except (ValueError, OSError):
        return str(path)


def index_item(
    source: str,
    status: str,
    directory: Path,
    *,
    output: Path | None = None,
    error: str | None = None,
    skip_reason: str | None = None,
) -> IndexItem:
    """The index entry of one batch item; *output* is its Markdown."""
    markdown = output if output is not None and output.name.endswith(".md") else None
    document = None
    if status == "completed" and markdown is not None:
        beside = json_path_beside(
            None if markdown.name.endswith(".llm.md") else markdown,
            markdown if markdown.name.endswith(".llm.md") else None,
        )
        if beside is not None and beside.is_file():
            document = _relative(beside, directory)
    return IndexItem(
        source=source,
        status=status,
        document=document,
        markdown=_relative(markdown, directory) if markdown is not None else None,
        error=error,
        skip_reason=skip_reason,
    )


def write_index(
    directory: Path, items: Iterable[IndexItem], *, allow_symlinks: bool = False
) -> Path:
    """Write ``markitai.index.json`` for this run's items. Items an earlier
    run into the same folder listed (a resumed batch) that this run did not
    touch stay; every item is ordered by source."""
    from markitai.document_json import tool_identity
    from markitai.security import atomic_write_text

    path = directory / INDEX_NAME
    listed: dict[str, dict[str, Any]] = {}
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        existing = None
    if isinstance(existing, dict) and isinstance(existing.get("items"), list):
        for item in existing["items"]:
            if (
                isinstance(item, dict)
                and isinstance(item.get("source"), str)
                and isinstance(item.get("status"), str)
            ):
                listed[item["source"]] = {
                    key: item.get(key)
                    for key in (
                        "source",
                        "status",
                        "document",
                        "markdown",
                        "error",
                        "skip_reason",
                    )
                }
    for item in items:
        listed[item.source] = item.to_json()
    index = {
        "schema_version": SCHEMA_VERSION,
        "tool": tool_identity(),
        "items": [listed[key] for key in sorted(listed)],
    }
    atomic_write_text(
        path,
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        follow_symlinks=allow_symlinks,
    )
    return path
