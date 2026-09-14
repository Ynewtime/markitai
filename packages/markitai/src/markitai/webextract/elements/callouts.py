"""Callout standardization: Obsidian, GitHub alerts, Bootstrap alerts."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag


def normalize_callouts(root: Tag) -> None:
    """Standardize callout/alert elements to blockquote with data-callout.

    Detects:
    - GitHub markdown alerts (.markdown-alert)
    - Bootstrap alerts (.alert.alert-*)
    - Callout asides (aside[class*="callout"])

    Converts to: <blockquote data-callout="type"><p>content</p></blockquote>

    Args:
        root: Content root element.
    """
    _normalize_obsidian_callouts(root)
    _normalize_github_alerts(root)
    _normalize_bootstrap_alerts(root)
    _normalize_callout_asides(root)
    _normalize_admonitions(root)


def _normalize_obsidian_callouts(root: Tag) -> None:
    """Preserve collapsed callout bodies before hidden-element removal.

    Normalize inner callouts first, then move the content wrapper's children
    into a blockquote. Its display:none represents the fold state, not absent
    content. Hidden elements *inside* the body still follow normal removal.
    """
    for callout in reversed(root.select("div.callout[data-callout]")):
        content = callout.find("div", class_="callout-content", recursive=False)
        if not isinstance(content, Tag):
            continue
        callout_type = str(callout.get("data-callout") or "note")
        if not re.fullmatch(r"[\w-]+", callout_type):
            callout_type = "note"
        title_el = callout.select_one(".callout-title-inner")
        title = title_el.get_text(" ", strip=True) if title_el else ""
        classes = callout.get("class") or []
        fold = str(callout.get("data-callout-fold") or "")
        if fold not in {"-", "+"}:
            fold = (
                "-"
                if "is-collapsed" in classes
                else ("+" if "is-collapsible" in classes else "")
            )
        _convert_to_blockquote(
            callout, callout_type, title=title, content=content, fold=fold
        )


def _normalize_github_alerts(root: Tag) -> None:
    """Convert GitHub .markdown-alert to blockquote."""
    for alert in root.select(".markdown-alert"):
        # Extract type from class: markdown-alert-note → note
        classes = alert.get("class")
        callout_type = "note"
        for cls in classes if isinstance(classes, list) else []:
            match = re.match(r"markdown-alert-(\w+)", cls)
            if match:
                callout_type = match.group(1).lower()
                break

        # Remove title element
        title_el = alert.select_one(".markdown-alert-title")
        if title_el:
            title_el.decompose()

        # Convert to blockquote
        _convert_to_blockquote(alert, callout_type)


def _normalize_bootstrap_alerts(root: Tag) -> None:
    """Convert Bootstrap .alert.alert-* to blockquote."""
    for alert in root.select(".alert"):
        classes = alert.get("class")
        callout_type = "note"
        for cls in classes if isinstance(classes, list) else []:
            match = re.match(r"alert-(\w+)", cls)
            if match and match.group(1) != "dismissible":
                callout_type = match.group(1).lower()
                break

        # Extract title if present
        title_el = alert.select_one(".alert-heading, .alert-title")
        if title_el:
            title_el.decompose()

        _convert_to_blockquote(alert, callout_type)


def _normalize_callout_asides(root: Tag) -> None:
    """Convert aside[class*="callout"] to blockquote."""
    for aside in root.find_all("aside"):
        classes = aside.get("class") or []
        classes_str = " ".join(classes) if isinstance(classes, list) else ""
        if "callout" not in classes_str.lower():
            continue

        callout_type = "note"
        for cls in classes if isinstance(classes, list) else []:
            match = re.match(r"callout-(\w+)", cls, re.IGNORECASE)
            if match:
                callout_type = match.group(1).lower()
                break

        _convert_to_blockquote(aside, callout_type)


# Admonition types recognized by Hugo/Docsy themes (mirrors defuddle).
_ADMONITION_TYPES = frozenset(
    {
        "info",
        "warning",
        "note",
        "tip",
        "danger",
        "caution",
        "important",
        "abstract",
        "success",
        "question",
        "failure",
        "bug",
        "example",
        "quote",
    }
)


def _normalize_admonitions(root: Tag) -> None:
    """Convert Hugo/Docsy .admonition blocks to blockquote callouts."""
    for el in root.select(".admonition"):
        if el.get("data-callout"):
            continue
        classes = el.get("class")
        class_list = classes if isinstance(classes, list) else []
        callout_type = next((c for c in class_list if c in _ADMONITION_TYPES), "note")

        # Title text from .admonition-title (icon SVGs contribute none);
        # remove it so it doesn't duplicate inside the content.
        title_el = el.select_one(".admonition-title")
        title = title_el.get_text(strip=True) if title_el is not None else ""
        if title_el is not None:
            title_el.decompose()

        content = (
            el.select_one(".admonition-content")
            or el.select_one(".details-content")
            or el
        )
        _convert_to_blockquote(el, callout_type, title=title, content=content)


def _convert_to_blockquote(
    el: Tag,
    callout_type: str,
    *,
    title: str = "",
    content: Tag | None = None,
    fold: str = "",
) -> None:
    """Replace element with a blockquote containing [!type] marker.

    Injects a ``[!type]`` paragraph at the start of the blockquote so
    that downstream HTML-to-Markdown converters produce Obsidian-style
    callout syntax: ``> [!type]``.

    Args:
        el: Element to replace.
        callout_type: Callout type token (e.g. ``"warning"``).
        title: Callout title; defaults to the capitalized type.
        content: Element whose children hold the callout body; defaults
            to ``el`` itself.
        fold: Obsidian fold marker ("-", "+", or empty).
    """
    doc = BeautifulSoup("", "html.parser")
    bq = doc.new_tag("blockquote")
    bq["data-callout"] = callout_type

    # Inject [!type] marker as first paragraph
    marker = doc.new_tag("p")
    marker.string = f"[!{callout_type}]{fold} {title or callout_type.capitalize()}"
    bq.append(marker)

    # Move children to blockquote
    for child in list((content if content is not None else el).children):
        bq.append(child.extract())

    el.replace_with(bq)
