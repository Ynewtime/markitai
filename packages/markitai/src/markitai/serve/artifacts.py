"""Which files in a job's out dir belong to which item.

Every item of a job writes into one shared out dir: its ``<base>.md`` /
``<base>.llm.md`` pair next to each other, and its extracted images and
screenshots into the shared ``.markitai/assets``, ``.markitai/screenshots``
(or, with a visible-assets output profile, ``assets/``) directories. The
result endpoint lists an item's files and item deletion removes them, so
both need the same answer to "is this file this item's?".

An item owns a file in an asset directory when either

* its name is ``<base>`` followed by a suffix the converters write:
  ``.0001.png`` (numbered assets), ``.page0001.jpg`` / ``.slide0001.jpg``
  (page renders), ``.full.jpg`` / ``.full--1.jpg`` (URL screenshots and
  their tiles), or ``-0001-10.jpg`` (PDF-extracted images); ``<base>`` is
  the output name without its markdown suffix (``report.pdf.v2.md`` ->
  ``report.pdf.v2``), which is also the asset prefix the pipeline uses, or
* one of the item's markdown files references it (URL screenshots are
  named after the URL, not the output, and only a reference ties them to
  their item).

The suffix must match in full, so item ``a`` never claims ``a.txt.0001.png``
belonging to sibling item ``a.txt``. Matching is a plain string prefix, not
a glob: metacharacters in upload names (``report[2024].pdf``) stay literal.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import unquote

from markitai.constants import (
    ASSETS_REL_PATH,
    SCREENSHOTS_REL_PATH,
    VISIBLE_ASSETS_REL_PATH,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

#: Directories (relative to the out dir) holding per-item asset files.
ASSET_DIRS = (ASSETS_REL_PATH, SCREENSHOTS_REL_PATH, VISIBLE_ASSETS_REL_PATH)

_IMAGE_EXT = r"\.(?:png|jpe?g|gif|webp|bmp|tiff?|svg|ico|avif|heic|heif)$"
# After "<base>.": numbered assets, page/slide renders, full-page URL shots
# and their tiles ("full--1.jpg").
_DOT_SUFFIX_RE = re.compile(
    rf"^(?:\d{{1,6}}|page\d{{1,6}}|slide\d{{1,6}}|full(?:--\d{{1,4}})?){_IMAGE_EXT}",
    re.IGNORECASE,
)
# After "<base>-": PDF-extracted images ("0001-10.jpg", page-index) and the
# single-number fallback the PDF converter uses for an unrecognized tail.
_DASH_SUFFIX_RE = re.compile(rf"^\d{{1,6}}(?:-\d{{1,6}})?{_IMAGE_EXT}", re.IGNORECASE)
# References to an asset directory, as markdown links (percent-encoded,
# possibly inside an HTML comment) or Obsidian wikilinks (raw names).
_DIR_ALTERNATION = "|".join(re.escape(d) for d in ASSET_DIRS)
_LINK_REF_RE = re.compile(rf"\]\(<?({_DIR_ALTERNATION})[/\\]([^)>\s]+)")
_WIKILINK_REF_RE = re.compile(rf"\[\[({_DIR_ALTERNATION})/([^\]|#]+)")


def split_output_name(name: str) -> str:
    """Strip the markitai markdown suffix: 'a.pdf.llm.md' -> 'a.pdf'."""
    if name.endswith(".llm.md"):
        return name[: -len(".llm.md")]
    if name.endswith(".md"):
        return name[: -len(".md")]
    return name


def claims_asset_name(base_name: str, file_name: str) -> bool:
    """Whether *file_name* is one of the asset names *base_name* writes."""
    for separator, suffix_re in ((".", _DOT_SUFFIX_RE), ("-", _DASH_SUFFIX_RE)):
        prefix = f"{base_name}{separator}"
        if file_name.startswith(prefix) and suffix_re.match(file_name[len(prefix) :]):
            return True
    return False


def referenced_assets(markdown: str) -> set[tuple[str, str]]:
    """``(asset dir, file name)`` pairs a markdown text references."""
    refs: set[tuple[str, str]] = set()
    for match in _LINK_REF_RE.finditer(markdown):
        refs.add((match.group(1), unquote(match.group(2))))
    for match in _WIKILINK_REF_RE.finditer(markdown):
        refs.add((match.group(1), match.group(2).strip()))
    # Only flat names: a reference never reaches outside its asset dir.
    return {
        (rel, name)
        for rel, name in refs
        if name and "/" not in name and "\\" not in name and name not in (".", "..")
    }


def markdown_pair(out_dir: Path, base_name: str) -> tuple[Path, Path]:
    """The item's ``<base>.md`` and ``<base>.llm.md`` paths."""
    return out_dir / f"{base_name}.md", out_dir / f"{base_name}.llm.md"


def item_asset_files(
    out_dir: Path, base_name: str, markdown_files: Iterable[Path]
) -> list[Path]:
    """Asset files owned by the item with output base *base_name*.

    Args:
        out_dir: The job's (resolved) out dir.
        base_name: The item's output name without its markdown suffix.
        markdown_files: The item's markdown outputs; missing files are fine.

    Returns:
        Existing files, asset dir by asset dir, each dir in name order.
    """
    refs: set[tuple[str, str]] = set()
    for md_file in markdown_files:
        try:
            refs |= referenced_assets(md_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    owned: list[Path] = []
    for rel in ASSET_DIRS:
        asset_dir = out_dir / rel
        if not asset_dir.is_dir():
            continue
        for path in sorted(asset_dir.iterdir()):
            if not path.is_file():
                continue
            if (rel, path.name) in refs or claims_asset_name(base_name, path.name):
                owned.append(path)
    return owned
