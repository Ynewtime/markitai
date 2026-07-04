"""Math standardization: MathJax tex scripts, MediaWiki math, MathJax v3.

Ported from a focused subset of defuddle's ``elements/math.base.ts`` and
``elements/math.core.ts``. Runs during standardization — before
sanitization strips ``<script>`` elements — and rewrites math markup into
clean ``<math>`` elements carrying a ``data-latex`` attribute so the
Markdown converter can emit ``$...$`` / ``$$...$$``.
"""

from __future__ import annotations

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

    Handles three sources (in order):
    1. MathJax v2 ``<script type="math/tex">`` sources (plus removal of
       the sibling rendered/preview elements that duplicate them).
    2. MediaWiki ``.mwe-math-element`` wrappers (hidden MathML + image
       fallback) — collapsed to the inner ``<math>``.
    3. MathJax v3 ``<mjx-container>`` with assistive MathML — hoisted.

    Args:
        root: Content root element (mutated in place).
    """
    _convert_tex_scripts(root)
    _collapse_mediawiki_math(root)
    _hoist_mjx_containers(root)


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


def _latex_from_math(math: Tag) -> str:
    """Extract LaTeX source from a ``<math>`` element, if present."""
    alttext = str(math.get("alttext") or "").strip()
    if alttext:
        return alttext
    annotation = math.find("annotation", attrs={"encoding": "application/x-tex"})
    if isinstance(annotation, Tag):
        return annotation.get_text().strip()
    return ""
