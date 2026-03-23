"""CSS @media mobile style analysis and pruning.

Analyzes <style> tags for @media (max-width: <=768px) rules that hide
elements (display: none). Removes matching elements before content scoring.
"""

from __future__ import annotations

import re

import tinycss2
from bs4 import BeautifulSoup, Tag
from loguru import logger

_MAX_WIDTH_RE = re.compile(
    r"max-width\s*:\s*(\d+(?:\.\d+)?)\s*(px|em|rem)", re.IGNORECASE
)
_DISPLAY_NONE_RE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_MOBILE_MAX_WIDTH_PX = 768


def apply_mobile_style_pruning(soup: BeautifulSoup) -> int:
    """Remove elements hidden via CSS @media mobile breakpoints."""
    hidden_selectors: list[str] = []
    for style_tag in soup.find_all("style"):
        css_text = style_tag.string
        if not css_text:
            continue
        hidden_selectors.extend(_extract_mobile_hidden_selectors(css_text))

    if not hidden_selectors:
        return 0

    removed = 0
    for selector in hidden_selectors:
        try:
            for el in soup.select(selector):
                if isinstance(el, Tag):
                    el.decompose()
                    removed += 1
        except Exception:  # noqa: BLE001
            continue

    if removed > 0:
        logger.debug(
            "[MobileStyles] Removed {} elements via {} selectors",
            removed,
            len(hidden_selectors),
        )
    return removed


def _extract_mobile_hidden_selectors(css_text: str) -> list[str]:
    """Extract selectors hidden at mobile breakpoints from CSS text."""
    selectors: list[str] = []
    rules = tinycss2.parse_stylesheet(css_text, skip_comments=True)

    for rule in rules:
        if rule.type != "at-rule" or rule.lower_at_keyword != "media":
            continue
        prelude_str = tinycss2.serialize(rule.prelude)
        match = _MAX_WIDTH_RE.search(prelude_str)
        if not match:
            continue
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit in ("em", "rem"):
            value *= 16
        if value > _MOBILE_MAX_WIDTH_PX:
            continue
        if not rule.content:
            continue
        content_rules = tinycss2.parse_rule_list(rule.content, skip_comments=True)
        for content_rule in content_rules:
            if content_rule.type != "qualified-rule":
                continue
            declarations_str = tinycss2.serialize(content_rule.content)
            if not _DISPLAY_NONE_RE.search(declarations_str):
                continue
            selector_str = tinycss2.serialize(content_rule.prelude).strip()
            if selector_str:
                selectors.append(selector_str)
    return selectors
