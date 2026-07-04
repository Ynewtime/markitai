"""Custom Markdown converter with enhanced code-block language detection.

Extends MarkItDown's ``_CustomMarkdownify`` with rules ported from
defuddle's ``elements/code.ts``.
"""

from __future__ import annotations

import re
from typing import Any, cast

from bs4 import Tag
from bs4.element import NavigableString
from markitdown._base_converter import DocumentConverterResult
from markitdown.converters._html_converter import HtmlConverter
from markitdown.converters._markdownify import _CustomMarkdownify

from markitai.webextract.constants import CODE_LANGUAGES

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
                if m and m.group(1).lower() in CODE_LANGUAGES:
                    return m.group(1).lower()

        class_str = " ".join(classes) if isinstance(classes, list) else str(classes)
        m = _LANG_FALLBACK_RE.search(class_str)
        if m:
            return m.group(1).lower()

    return None


# ---------------------------------------------------------------------------
# MathML → LaTeX conversion
# ---------------------------------------------------------------------------

# Operator replacements for common MathML entities
_MO_REPLACEMENTS: dict[str, str] = {
    "≠": r"\neq",
    "≤": r"\leq",
    "≥": r"\geq",
    "±": r"\pm",
    "∓": r"\mp",
    "×": r"\times",
    "÷": r"\div",
    "∞": r"\infty",
    "∑": r"\sum",
    "∏": r"\prod",
    "∫": r"\int",
    "∂": r"\partial",
    "∇": r"\nabla",
    "→": r"\rightarrow",
    "←": r"\leftarrow",
    "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow",
    "↦": r"\mapsto",
    "∈": r"\in",
    "∉": r"\notin",
    "⊂": r"\subset",
    "⊃": r"\supset",
    "⊆": r"\subseteq",
    "⊇": r"\supseteq",
    "∪": r"\cup",
    "∩": r"\cap",
    "∧": r"\wedge",
    "∨": r"\vee",
    "¬": r"\neg",
    "⟨": r"\langle",
    "⟩": r"\rangle",
    "⌊": r"\lfloor",
    "⌋": r"\rfloor",
    "⌈": r"\lceil",
    "⌉": r"\rceil",
    "∅": r"\emptyset",
    "∀": r"\forall",
    "∃": r"\exists",
    "⟹": r"\implies",
    "⟸": r"\impliedby",
    "⟺": r"\iff",
    "⋯": r"\cdots",
    "…": r"\ldots",
    "⋅": r"\cdot",
    "·": r"\cdot",
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "τ": r"\tau",
    "υ": r"\upsilon",
    "ϕ": r"\phi",
    "φ": r"\varphi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
}


def _mathml_to_latex(el: Any) -> str:
    """Convert a MathML element tree to LaTeX notation.

    Handles the common MathML elements: mi, mn, mo, mrow, msup, msub,
    msubsup, mfrac, msqrt, mover, munder, munderover, mtable, mtr, mtd,
    mtext, mspace, semantics.
    """
    if isinstance(el, NavigableString):
        return str(el).strip()

    if not isinstance(el, Tag):
        return ""

    tag = el.name

    # Skip semantics wrapper — process children
    if tag == "semantics":
        # Prefer annotation if present (already checked by caller)
        children = [c for c in el.children if isinstance(c, Tag)]
        if children:
            return _mathml_to_latex(children[0])
        return ""

    if tag in ("math", "mrow", "mpadded", "mstyle", "merror", "mphantom"):
        return _convert_children(el)

    if tag == "mi":
        text = el.get_text(strip=True)
        if len(text) > 1 and text.isalpha():
            return rf"\mathrm{{{text}}}"
        return _mo_replace(text)

    if tag == "mn":
        return el.get_text(strip=True)

    if tag == "mo":
        text = el.get_text(strip=True)
        return _mo_replace(text)

    if tag == "mtext":
        text = el.get_text(strip=True)
        if text:
            return rf"\text{{{text}}}"
        return ""

    if tag == "mspace":
        return r"\;"

    if tag == "msup":
        children = _child_tags(el)
        if len(children) >= 2:
            base = _mathml_to_latex(children[0])
            sup = _mathml_to_latex(children[1])
            return f"{base}^{{{sup}}}"
        return _convert_children(el)

    if tag == "msub":
        children = _child_tags(el)
        if len(children) >= 2:
            base = _mathml_to_latex(children[0])
            sub = _mathml_to_latex(children[1])
            return f"{base}_{{{sub}}}"
        return _convert_children(el)

    if tag == "msubsup":
        children = _child_tags(el)
        if len(children) >= 3:
            base = _mathml_to_latex(children[0])
            sub = _mathml_to_latex(children[1])
            sup = _mathml_to_latex(children[2])
            return f"{base}_{{{sub}}}^{{{sup}}}"
        return _convert_children(el)

    if tag == "mfrac":
        children = _child_tags(el)
        if len(children) >= 2:
            num = _mathml_to_latex(children[0])
            den = _mathml_to_latex(children[1])
            return rf"\frac{{{num}}}{{{den}}}"
        return _convert_children(el)

    if tag == "msqrt":
        inner = _convert_children(el)
        return rf"\sqrt{{{inner}}}"

    if tag == "mroot":
        children = _child_tags(el)
        if len(children) >= 2:
            base = _mathml_to_latex(children[0])
            index = _mathml_to_latex(children[1])
            return rf"\sqrt[{index}]{{{base}}}"
        return _convert_children(el)

    if tag == "mover":
        children = _child_tags(el)
        if len(children) >= 2:
            base = _mathml_to_latex(children[0])
            # Check the raw text of the overscript *before* recursive conversion
            # so that e.g. <mo>→</mo> is matched as "→" not "\rightarrow".
            over_raw = children[1].get_text(strip=True)
            if over_raw in ("˙", "̇"):
                return rf"\dot{{{base}}}"
            if over_raw in ("¯", "‾"):
                return rf"\overline{{{base}}}"
            if over_raw in ("^", "̂"):
                return rf"\hat{{{base}}}"
            if over_raw in ("~", "̃", "˜"):
                return rf"\tilde{{{base}}}"
            if over_raw in ("→", "⃗"):
                return rf"\vec{{{base}}}"
            over = _mathml_to_latex(children[1])
            return rf"\overset{{{over}}}{{{base}}}"
        return _convert_children(el)

    if tag == "munder":
        children = _child_tags(el)
        if len(children) >= 2:
            base = _mathml_to_latex(children[0])
            under = _mathml_to_latex(children[1])
            return rf"\underset{{{under}}}{{{base}}}"
        return _convert_children(el)

    if tag == "munderover":
        children = _child_tags(el)
        if len(children) >= 3:
            base = _mathml_to_latex(children[0])
            under = _mathml_to_latex(children[1])
            over = _mathml_to_latex(children[2])
            return f"{base}_{{{under}}}^{{{over}}}"
        return _convert_children(el)

    if tag == "mtable":
        rows: list[str] = []
        for tr in el.find_all("mtr", recursive=False):
            cells = [_mathml_to_latex(td) for td in tr.find_all("mtd", recursive=False)]
            rows.append(" & ".join(cells))
        return r"\begin{aligned}" + " \\\\ ".join(rows) + r"\end{aligned}"

    if tag == "mfenced":
        open_d = el.get("open", "(")
        close_d = el.get("close", ")")
        sep = el.get("separators", ",")
        children = _child_tags(el)
        inner = f" {sep} ".join(_mathml_to_latex(c) for c in children)
        return rf"\left{open_d}{inner}\right{close_d}"

    # Fallback: just convert children
    return _convert_children(el)


def _child_tags(el: Any) -> list[Any]:
    """Return direct child Tag elements."""
    return [c for c in el.children if isinstance(c, Tag)]


def _convert_children(el: Any) -> str:
    """Convert all children and join."""
    parts: list[str] = []
    for child in el.children:
        if isinstance(child, Tag):
            parts.append(_mathml_to_latex(child))
        elif isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                parts.append(text)
    return " ".join(parts) if parts else ""


def _mo_replace(text: str) -> str:
    """Replace Unicode math operators with LaTeX commands."""
    return _MO_REPLACEMENTS.get(text, text)


def _is_footnote_list(el: Any) -> bool:
    """Check whether an <ol> is the standardized footnotes list.

    Keyed on the ``li.footnote`` items with ``id="fn:N"`` (robust even if
    the ``div#footnotes`` wrapper was flattened away).
    """
    items = [c for c in el.children if isinstance(c, Tag) and c.name == "li"]
    if not items:
        return False
    return all(str(li.get("id") or "").startswith("fn:") for li in items)


class WebExtractMarkdownConverter(_CustomMarkdownify):
    """Markdownify converter with enhanced rules for web content."""

    def convert_math(self, el: Any, text: str, parent_tags: set) -> str:
        """Convert <math> to LaTeX.

        Prefers data-latex (set by webextract math standardization), then
        alttext, annotation, and finally structural MathML conversion.
        """
        is_block = el.get("display") == "block"
        latex = str(el.get("data-latex") or "").strip()
        if not latex:
            latex = str(el.get("alttext") or "").strip()
        if not latex:
            annotation = el.find("annotation", attrs={"encoding": "application/x-tex"})
            if annotation and annotation.string:
                latex = annotation.string.strip()
        if not latex:
            # Try converting MathML structure to LaTeX
            latex = _mathml_to_latex(el)
        if not latex:
            latex = el.get_text(strip=True)
        if not latex:
            return ""
        return f"\n\n$${latex}$$\n\n" if is_block else f"${latex}$"

    def convert_script(self, el: Any, text: str, parent_tags: set) -> str:
        """Convert math/tex script elements to LaTeX."""
        script_type = str(el.get("type", ""))
        if "math/tex" not in script_type:
            return ""
        latex = (el.string or "").strip()
        if not latex:
            return ""
        is_display = "mode=display" in script_type
        return f"\n\n$${latex}$$\n\n" if is_display else f"${latex}$"

    def convert_span(self, el: Any, text: str, parent_tags: set) -> str:
        """Handle KaTeX wrapper spans."""
        classes = el.get("class", [])
        if isinstance(classes, str):
            classes = classes.split()
        if "katex" in classes:
            annotation = el.find("annotation", attrs={"encoding": "application/x-tex"})
            if annotation and annotation.string:
                latex = annotation.string.strip()
                if "katex-display" in classes:
                    return f"\n\n$${latex}$$\n\n"
                return f"${latex}$"
        return text

    def convert_mark(self, el: Any, text: str, parent_tags: set) -> str:
        return f"=={text}==" if text.strip() else ""

    def convert_del(self, el: Any, text: str, parent_tags: set) -> str:
        return f"~~{text}~~" if text.strip() else ""

    def convert_s(self, el: Any, text: str, parent_tags: set) -> str:
        return f"~~{text}~~" if text.strip() else ""

    def convert_sup(self, el: Any, text: str, parent_tags: set) -> str:
        # Standardized footnote reference: <sup id="fnref:N"><a href="#fn:N">
        # (duplicate refs use id="fnref:N-2" — emit the primary number).
        sup_id = str(el.get("id") or "")
        if sup_id.startswith("fnref:"):
            primary = sup_id[len("fnref:") :].split("-")[0]
            if primary:
                return f"[^{primary}]"
        link = el.find("a")
        if link:
            href = str(link.get("href", ""))
            if (
                href.startswith("#fn")
                or href.startswith("#note")
                or href.startswith("#cite")
            ):
                ref_text = el.get_text(strip=True)
                if ref_text.isdigit():
                    return f"[^{ref_text}]"
        return text

    def convert_a(self, el: Any, text: str, parent_tags: set) -> str:
        """Drop footnote back-reference links; defer to the base rule."""
        classes = el.get("class") or []
        if isinstance(classes, str):
            classes = classes.split()
        href = str(el.get("href") or "")
        if "footnote-backref" in classes or href.startswith("#fnref"):
            return ""
        return super().convert_a(el, text, parent_tags=parent_tags)

    def convert_ol(self, el: Any, text: str, parent_tags: set) -> str:
        """Emit the standardized footnotes list as [^N]: definition blocks."""
        if _is_footnote_list(el):
            body = text.strip()
            return f"\n\n{body}\n\n" if body else ""
        # markdownify's list rules are untyped (convert_ol = convert_list)
        base = cast(Any, super())
        return base.convert_ol(el, text, parent_tags)

    def convert_li(self, el: Any, text: str, parent_tags: set) -> str:
        """Emit footnote definition items as ``[^N]: content``.

        Multi-paragraph definitions keep paragraphs separated by blank
        lines (unindented continuation, matching defuddle's output).
        """
        li_id = str(el.get("id") or "")
        if li_id.startswith("fn:"):
            number = li_id[len("fn:") :]
            content = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
            return f"[^{number}]: {content}\n\n"
        base = cast(Any, super())
        return base.convert_li(el, text, parent_tags)

    def convert_pre(self, el: Any, text: str, parent_tags: set) -> str:
        """Convert <pre> to fenced code block with language detection."""
        code_el = el.find("code") if el.name == "pre" else el
        if code_el is None:
            code_el = el

        language = _detect_language(code_el) or ""

        code_text = code_el.get_text()
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
