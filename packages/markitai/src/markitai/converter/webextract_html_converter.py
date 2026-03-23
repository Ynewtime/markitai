"""Custom Markdown converter with enhanced code-block language detection.

Extends MarkItDown's ``_CustomMarkdownify`` with rules ported from
defuddle's ``elements/code.ts``.
"""

from __future__ import annotations

import re
from typing import Any

from markitdown._base_converter import DocumentConverterResult
from markitdown.converters._html_converter import HtmlConverter
from markitdown.converters._markdownify import _CustomMarkdownify

# ---------------------------------------------------------------------------
# Language detection patterns (ported from defuddle elements/code.ts)
# ---------------------------------------------------------------------------

_LANG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^language-(\w+)$"),
    re.compile(r"^lang-(\w+)$"),
    re.compile(r"^highlight-(\w+)$"),
    re.compile(r"^(\w+)-code$"),
    re.compile(r"^code-(\w+)$"),
    re.compile(r"^syntax-(\w+)$"),
    re.compile(r"^code-snippet__(\w+)$"),
    re.compile(r"^(\w+)-snippet$"),
]

_LANG_FALLBACK_RE = re.compile(
    r"(?:^|\s)(?:language|lang|brush|syntax)[:\-]\s*(\w+)(?:\s|$)", re.IGNORECASE
)


def _detect_language(el: Any) -> str | None:
    """Detect programming language from element attributes.

    Checks data-lang attribute, then class names on the element and
    its parent <pre> (Prism.js style).
    """
    data_lang = el.get("data-lang")
    if data_lang:
        return str(data_lang).strip()

    candidates = [el]
    parent = el.parent
    if parent and parent.name == "pre":
        candidates.append(parent)
    elif el.name == "pre":
        code = el.find("code")
        if code:
            candidates.insert(0, code)

    for candidate in candidates:
        classes = candidate.get("class", [])
        if isinstance(classes, str):
            classes = classes.split()

        for cls in classes:
            for pattern in _LANG_PATTERNS:
                m = pattern.match(cls)
                if m:
                    return m.group(1).lower()

        class_str = " ".join(classes) if isinstance(classes, list) else str(classes)
        m = _LANG_FALLBACK_RE.search(class_str)
        if m:
            return m.group(1).lower()

    return None


class WebExtractMarkdownConverter(_CustomMarkdownify):
    """Markdownify converter with enhanced rules for web content."""

    def convert_pre(self, el: Any, text: str, parent_tags: set) -> str:
        """Convert <pre> to fenced code block with language detection."""
        code_el = el.find("code") if el.name == "pre" else el
        if code_el is None:
            code_el = el

        language = _detect_language(code_el) or ""

        code_text = code_el.get_text() if code_el else el.get_text()
        if code_text.startswith("\n"):
            code_text = code_text[1:]
        if code_text.endswith("\n"):
            code_text = code_text[:-1]

        return f"\n\n```{language}\n{code_text}\n```\n\n"


class WebExtractHtmlConverter(HtmlConverter):
    """HtmlConverter that uses our custom markdownify converter."""

    def convert(
        self, file_stream: Any, stream_info: Any, **kwargs: Any
    ) -> DocumentConverterResult:
        from bs4 import BeautifulSoup

        encoding = "utf-8" if stream_info.charset is None else stream_info.charset
        soup = BeautifulSoup(file_stream, "html.parser", from_encoding=encoding)

        for script in soup(["script", "style"]):
            script.extract()

        body = soup.find("body")
        if body:
            webpage_text = WebExtractMarkdownConverter(**kwargs).convert_soup(body)
        else:
            webpage_text = WebExtractMarkdownConverter(**kwargs).convert_soup(soup)

        return DocumentConverterResult(
            markdown=webpage_text.strip(),
            title=None if soup.title is None else soup.title.string,
        )
