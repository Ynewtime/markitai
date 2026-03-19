"""Callout standardization: GitHub alerts, Bootstrap alerts."""

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
    _normalize_github_alerts(root)
    _normalize_bootstrap_alerts(root)
    _normalize_callout_asides(root)


def _normalize_github_alerts(root: Tag) -> None:
    """Convert GitHub .markdown-alert to blockquote."""
    for alert in root.select(".markdown-alert"):
        # Extract type from class: markdown-alert-note → note
        classes = alert.get("class", [])
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
        classes = alert.get("class", [])
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
        classes = aside.get("class", [])
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


def _convert_to_blockquote(el: Tag, callout_type: str) -> None:
    """Replace element with a blockquote carrying data-callout attribute."""
    bq = BeautifulSoup("", "html.parser").new_tag("blockquote")
    bq["data-callout"] = callout_type

    # Move children to blockquote
    for child in list(el.children):
        bq.append(child.extract())

    el.replace_with(bq)
