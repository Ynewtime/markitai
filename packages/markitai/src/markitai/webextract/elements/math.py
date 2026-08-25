"""Math standardization: MathJax tex scripts, MediaWiki math, MathJax v3.

Ported from a focused subset of defuddle's ``elements/math.base.ts`` and
``elements/math.core.ts``. Runs during standardization — before
sanitization strips ``<script>`` elements — and rewrites math markup into
clean ``<math>`` elements carrying a ``data-latex`` attribute so the
Markdown converter can emit ``$...$`` / ``$$...$$``.
"""

from __future__ import annotations

import re
from urllib.parse import unquote

from bs4 import BeautifulSoup, Tag

# MathJax v2 render/preview elements that duplicate a math/tex script's
# content (defuddle math.core.ts cleans these up per parent).
_MATHJAX_RENDER_CLASSES = frozenset(
    {
        "MathJax_Preview",
        "MathJax",
        "MathJax_Display",
        "MathJax_SVG",
        "MathJax_MathML",
    }
)


def normalize_math(root: Tag) -> None:
    """Standardize math markup into clean ``<math>`` elements.

    Handles four sources (in order):
    1. MathJax v2 ``<script type="math/tex">`` sources (plus removal of
       the sibling rendered/preview elements that duplicate them).
    2. MediaWiki ``.mwe-math-element`` wrappers (hidden MathML + image
       fallback) — collapsed to the inner ``<math>``.
    3. MathJax v3 ``<mjx-container>`` with assistive MathML — hoisted.
    4. Images rendered by LaTeX services (CodeCogs, Google Charts, …) —
       replaced by the LaTeX encoded in their URL or alt text.

    Args:
        root: Content root element (mutated in place).
    """
    _convert_tex_scripts(root)
    _collapse_mediawiki_math(root)
    _hoist_mjx_containers(root)
    _convert_latex_images(root)


def _new_math_tag(latex: str, *, is_block: bool) -> Tag:
    """Create a clean ``<math>`` element carrying LaTeX source."""
    math = BeautifulSoup("", "html.parser").new_tag("math")
    math["display"] = "block" if is_block else "inline"
    math["data-latex"] = latex
    math.string = latex
    return math


def _convert_tex_scripts(root: Tag) -> None:
    """Replace ``<script type="math/tex">`` elements with ``<math>``."""
    converted_parents: list[Tag] = []
    for script in list(root.find_all("script")):
        script_type = str(script.get("type") or "")
        if not script_type.startswith("math/tex"):
            continue
        latex = script.get_text().strip()
        if not latex:
            script.decompose()
            continue
        is_block = "mode=display" in script_type
        parent = script.parent
        script.replace_with(_new_math_tag(latex, is_block=is_block))
        if isinstance(parent, Tag):
            converted_parents.append(parent)

    # Remove MathJax v2 rendered/preview siblings that duplicate the
    # LaTeX we just extracted (scoped per parent, like defuddle).
    for parent in converted_parents:
        for el in list(parent.find_all(True, recursive=False)):
            classes = el.get("class")
            if isinstance(classes, list) and _MATHJAX_RENDER_CLASSES & set(classes):
                el.decompose()


def _collapse_mediawiki_math(root: Tag) -> None:
    """Collapse MediaWiki ``.mwe-math-element`` wrappers to one ``<math>``.

    MediaWiki ships MathML inside a ``display: none`` span plus a visible
    image fallback. Keeping both duplicates the equation; keep the MathML
    (with LaTeX from its annotation/alttext) and drop the fallback.
    """
    for wrapper in list(root.find_all(class_="mwe-math-element")):
        inner = wrapper.find("math")
        if isinstance(inner, Tag):
            latex = _latex_from_math(inner)
            if latex and not inner.get("data-latex"):
                inner["data-latex"] = latex
            wrapper.replace_with(inner.extract())
            continue
        # No MathML survived — fall back to the image's alt text (LaTeX).
        img = wrapper.find("img", alt=True)
        if isinstance(img, Tag):
            alt = str(img.get("alt") or "").strip()
            if alt:
                img_classes = img.get("class")
                classes_str = (
                    " ".join(img_classes) if isinstance(img_classes, list) else ""
                )
                is_block = "display" in classes_str
                wrapper.replace_with(_new_math_tag(alt, is_block=is_block))


def _hoist_mjx_containers(root: Tag) -> None:
    """Hoist assistive MathML out of MathJax v3 ``<mjx-container>``."""
    for container in list(root.find_all("mjx-container")):
        math = container.find("math")
        if not isinstance(math, Tag):
            continue
        if str(container.get("display") or "") == "true":
            math["display"] = "block"
        latex = _latex_from_math(math)
        if latex and not math.get("data-latex"):
            math["data-latex"] = latex
        container.replace_with(math.extract())


# Named query parameters used by LaTeX rendering services
# (latex/tex/eq/math = generic; chl = Google Charts).
_LATEX_PARAM_RES = [
    re.compile(rf"[?&]{param}=([^&#]+)", re.IGNORECASE)
    for param in ("latex", "chl", "tex", "eq", "math")
]
_LOOKS_LIKE_LATEX_RE = re.compile(r"\\[a-zA-Z]{2,}")


def _convert_latex_images(root: Tag) -> None:
    """Replace images rendered by LaTeX services with ``<math>`` elements.

    Uses URL-based heuristics (not domain allowlists) to detect encoded
    LaTeX, falling back to alt text containing LaTeX commands. Ported
    from defuddle ``standardize.ts`` ``convertLatexImages``.
    """
    for img in list(root.find_all("img", src=True)):
        src = str(img.get("src") or "")
        latex = _latex_from_image_src(src)
        if not latex:
            alt = str(img.get("alt") or "")
            if _LOOKS_LIKE_LATEX_RE.search(alt):
                latex = alt
        if not latex:
            continue

        parent = img.parent
        is_block = "\\begin{" in latex or (
            isinstance(parent, Tag) and parent.name == "p" and len(parent.contents) == 1
        )
        img.replace_with(_new_math_tag(latex, is_block=is_block))


def _latex_from_image_src(src: str) -> str | None:
    """Extract LaTeX from an image URL with URL-encoded LaTeX commands."""
    # Named query parameters
    for param_re in _LATEX_PARAM_RES:
        match = param_re.search(src)
        if match:
            latex = _decode_latex(match.group(1))
            if latex:
                return latex

    # The full query string as bare LaTeX (e.g. CodeCogs, mimeTeX)
    query_match = re.search(r"\?([^#]+)", src)
    if query_match:
        latex = _decode_latex(query_match.group(1))
        if latex:
            return latex

    # URL path segments containing encoded LaTeX (%5C = backslash)
    for segment in reversed(src.split("?")[0].split("/")):
        if re.search(r"%5[Cc]", segment):
            latex = _decode_latex(segment)
            if latex:
                return latex
    return None


def _decode_latex(raw: str) -> str | None:
    """Decode a URL-encoded string if it contains a LaTeX command."""
    decoded = unquote(raw.replace("+", " "))
    return decoded if _LOOKS_LIKE_LATEX_RE.search(decoded) else None


def _latex_from_math(math: Tag) -> str:
    """Extract LaTeX source from a ``<math>`` element, if present."""
    alttext = str(math.get("alttext") or "").strip()
    if alttext:
        return alttext
    annotation = math.find("annotation", attrs={"encoding": "application/x-tex"})
    if isinstance(annotation, Tag):
        return annotation.get_text().strip()
    return ""
