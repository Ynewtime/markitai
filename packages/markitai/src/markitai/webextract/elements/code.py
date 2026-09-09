from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from markitai.webextract.constants import CODE_LANGUAGES

# Language extraction patterns (ported from defuddle elements/code.ts)
_LANG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"language-(\w+)", re.IGNORECASE),
    re.compile(r"lang-(\w+)", re.IGNORECASE),
    re.compile(r"highlight-(\w+)", re.IGNORECASE),
    re.compile(r"syntax-(\w+)", re.IGNORECASE),
    re.compile(r"(\w+)-code\b", re.IGNORECASE),
    re.compile(r"code-(\w+)", re.IGNORECASE),
    re.compile(r"code-snippet__(\w+)", re.IGNORECASE),
    re.compile(r"(\w+)-snippet\b", re.IGNORECASE),
]


# Line-number gutter elements emitted by syntax highlighters (Hugo/Chroma
# ``span.lnt`` and gutter ``td``s, Pygments ``span.lineno``, Rouge/Jekyll
# ``.rouge-gutter``, react-syntax-highlighter). Skipped during upstream's
# structured text extraction; removed from the DOM here so plain
# ``get_text`` conversion never sees them.
_LINE_NUMBER_GUTTER_SELECTOR = ", ".join(
    f"{scope} {sel}"
    for scope in ("pre", "code")
    for sel in (
        "span.lnt",
        "span.lineno",
        ".rouge-gutter",
        ".react-syntax-highlighter-line-number",
        "td.linenos",
    )
)


def normalize_code_blocks(root: Tag) -> None:
    """Normalize code blocks: wrap in pre, detect and normalize language class.

    Args:
        root: Content root.
    """
    for gutter in root.select(_LINE_NUMBER_GUTTER_SELECTOR):
        gutter.decompose()
    _strip_inline_numeric_gutters(root)
    _collapse_code_layout_tables(root)
    _rebuild_codemirror_blocks(root)

    for code in list(root.find_all("code")):
        raw_classes = code.get("class")
        classes: list[str] = list(raw_classes) if isinstance(raw_classes, list) else []
        style = str(code.get("style", ""))

        # Wrap in <pre> if inline code with pre-style
        if "white-space: pre" in style and code.parent and code.parent.name != "pre":
            pre = BeautifulSoup("", "html.parser").new_tag("pre")
            code.wrap(pre)

        # Detect language from code or parent element
        lang = _detect_language(code) or _detect_language_from_parent(code)
        if lang and not any(c.startswith("language-") for c in classes):
            classes = [f"language-{lang}"] + [
                c for c in classes if not _is_lang_class(c)
            ]

        if classes:
            code["class"] = classes  # type: ignore[assignment]


# Syntax-highlighter wrapper containers (subset of upstream's
# codeBlockRules selector list that uses table-based line-number layouts).
_HIGHLIGHTER_WRAPPER_SELECTOR = (
    '.highlight, .highlight-source, .chroma, div[class*="prismjs"], '
    ".syntaxhighlighter, .wp-block-syntaxhighlighter-code, .wp-block-code, "
    'div[class*="language-"]'
)


def _strip_inline_numeric_gutters(root: Tag) -> None:
    """Drop numeric gutters from two-child line wrappers inside code.

    Some viewers render each code line as a row whose first child is the
    line number and second the code (e.g. Chroma inline line numbers:
    ``<span style="display:flex"><span>1</span><span>code</span></span>``).
    Mirrors the two-child all-digits rule in upstream's
    ``extractStructuredText``; without it the text concatenates as
    ``"1p = 61"``.
    """
    for el in root.select("pre span, pre div, code span, code div"):
        children = el.find_all(True, recursive=False)
        if len(children) != 2:
            continue
        if not children[0].get_text(strip=True).isdigit():
            continue
        # A plain numeric literal also sits first in a two-token span
        # (``<span class="mi">42</span><span> + x</span>``); only rows that
        # announce themselves as line rows lose their first child.
        if _looks_like_line_row(el, children[0]):
            children[0].decompose()


_LINE_ROW_CLASS_RE = re.compile(
    r"(?:^|[-_:])(?:ln|line|lineno|linenumber|number|gutter)(?:$|[-_:])"
)


def _looks_like_line_row(row: Tag, gutter: Tag) -> bool:
    style = str(row.get("style") or "")
    if "flex" in style or "table-row" in style:
        return True
    for el in (row, gutter):
        classes = " ".join(str(c) for c in (el.get("class") or []))
        if _LINE_ROW_CLASS_RE.search(classes):
            return True
    return False


def _collapse_code_layout_tables(root: Tag) -> None:
    """Replace line-number layout tables with their code ``<pre>``.

    Hugo/Chroma render code as a two-column table (line-number gutter +
    code). Upstream rebuilds the whole highlighter wrapper via structured
    extraction that picks the code ``<pre>``; here the table is replaced
    with that ``<pre>`` so it never reaches Markdown table conversion.
    """
    for wrapper in root.select(_HIGHLIGHTER_WRAPPER_SELECTOR):
        for table in list(wrapper.find_all("table")):
            if table.parent is None:
                continue
            pres = table.find_all("pre")
            if not pres:
                continue
            # The code <pre> has a language-annotated <code> or line spans
            # (the gutter <pre>, if still present, has neither).
            code_pre = next(
                (
                    p
                    for p in pres
                    if p.select_one(
                        'code[data-lang], code[class*="language-"], .line, [data-line]'
                    )
                ),
                None,
            ) or next((p for p in pres if p.find("span", class_=True)), None)
            if code_pre is None:
                continue
            table.replace_with(code_pre.extract())


# Intentional subset of constants.BLOCK_LEVEL_ELEMENTS: block types whose
# text is rescued into rebuilt <pre> blocks (headings/list items excluded
# from rescue are handled by their own normalizers).
_RESCUE_BLOCK_TAGS = ("p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "blockquote")


def _rebuild_codemirror_blocks(root: Tag) -> None:
    """Rebuild CodeMirror editor UIs as plain ``<pre><code>`` blocks.

    ChatGPT-style runnable code blocks wrap a full CodeMirror editor in
    ``<pre>``: the code lives in ``.cm-content`` spans separated by
    ``<br>``, and the language is a bare text label in the header (not a
    class or data attribute). Ported from the CodeMirror branches of
    defuddle ``elements/code.ts`` ``codeBlockRules``.

    Deviation from upstream: real-world editor markup leaves ``<div>``s
    unbalanced inside ``<pre>``, and lxml recovers by swallowing the
    following siblings into the editor subtree. Block-level content that
    sits inside the ``<pre>`` but outside the editor is re-emitted after
    the rebuilt block instead of being dropped with the editor chrome.

    Args:
        root: Content root.
    """
    for cm_content in list(root.select(".cm-content")):
        pre = cm_content.find_parent("pre")
        if pre is None or pre.parent is None:
            continue

        # Language label: a div outside the code area whose text is a
        # bare language name (mirrors upstream's header-text scan).
        language = ""
        for div in pre.find_all("div"):
            if div is cm_content or cm_content in div.descendants:
                continue
            text = div.get_text(strip=True).lower()
            if text and text in CODE_LANGUAGES:
                language = text
                break

        code_text = _cleanup_code_text(_structured_code_text(cm_content))

        # Rescue block content lxml swallowed into the editor wrappers.
        rescued = [
            el
            for el in pre.find_all(_RESCUE_BLOCK_TAGS)
            if cm_content not in el.parents
        ]

        builder = BeautifulSoup("", "html.parser")
        new_pre = builder.new_tag("pre")
        new_code = builder.new_tag("code")
        if language:
            new_code["data-lang"] = language
            new_code["class"] = f"language-{language}"
        new_code.string = code_text
        new_pre.append(new_code)

        anchor: Tag = new_pre
        pre.replace_with(new_pre)
        for el in rescued:
            anchor.insert_after(el)
            anchor = el


def _structured_code_text(el: Tag) -> str:
    """Extract text from a code container, turning ``<br>`` into newlines."""
    parts: list[str] = []
    for node in el.descendants:
        if isinstance(node, NavigableString):
            parts.append(str(node))
        elif isinstance(node, Tag) and node.name == "br":
            parts.append("\n")
    return "".join(parts)


def _cleanup_code_text(text: str) -> str:
    """Normalize extracted code text (mirrors upstream's cleanup pass)."""
    text = text.replace("\t", "    ").replace(" ", " ")
    # Dedent: strip the common leading indent
    lines = text.split("\n")
    indents = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    min_indent = min(indents, default=0)
    if min_indent:
        lines = [ln[min_indent:] for ln in lines]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def _detect_language(el: Tag) -> str | None:
    """Extract language from element's class or data attributes.

    Fuzzy pattern hits (e.g. ``CodeBlock-code``) only count when the
    detected token is a known language (defuddle's CODE_LANGUAGES check).
    """
    # Check data attributes first
    for attr in ("data-lang", "data-language", "language"):
        val = el.get(attr)
        if val and isinstance(val, str):
            return val.strip().lower()

    # Check class patterns
    raw_classes = el.get("class")
    if isinstance(raw_classes, list):
        for cls in raw_classes:
            for pattern in _LANG_PATTERNS:
                match = pattern.search(cls)
                if match and match.group(1).lower() in CODE_LANGUAGES:
                    return match.group(1).lower()
    return None


def _detect_language_from_parent(code: Tag) -> str | None:
    """Check parent <pre> or wrapper for language hints."""
    parent = code.parent
    if parent is None:
        return None
    # Check parent <pre> or <div> with highlight/syntax classes
    for ancestor in [parent, parent.parent]:
        if ancestor is None or not isinstance(ancestor, Tag):
            continue
        lang = _detect_language(ancestor)
        if lang:
            return lang
    return None


def _is_lang_class(cls: str) -> bool:
    """Check if a class name is a language indicator."""
    return any(p.search(cls) for p in _LANG_PATTERNS)
