from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Comment, Tag
from bs4.element import NavigableString

from markitai.webextract.constants import (
    ALLOWED_EMPTY_ELEMENTS,
    BLOCK_LEVEL_ELEMENTS,
    TAILWIND_COLORS,
    TAILWIND_SPECIAL,
)
from markitai.webextract.elements.callouts import normalize_callouts
from markitai.webextract.elements.code import normalize_code_blocks
from markitai.webextract.elements.headings import normalize_headings
from markitai.webextract.elements.images import normalize_images
from markitai.webextract.elements.math import normalize_math

_PRESERVE_ELEMENTS = frozenset(
    {
        "pre",
        "code",
        "table",
        "thead",
        "tbody",
        "tr",
        "td",
        "th",
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        "figure",
        "figcaption",
        "picture",
        "details",
        "summary",
        "blockquote",
        "form",
        "fieldset",
    }
)

# standardize's view of block elements: BLOCK_LEVEL_ELEMENTS minus the three
# members defuddle's standardize.ts also excludes here (content/form/fieldset).
_BLOCK_LEVEL = BLOCK_LEVEL_ELEMENTS - {"content", "form", "fieldset"}


def standardize_content(root: Tag, title: str | None, base_url: str) -> None:
    """Normalize extracted content in place.

    Args:
        root: Content root.
        title: Extracted title.
        base_url: Base URL for relative link resolution.
    """

    _remove_comments(root)
    _convert_h1_to_h2(root)
    _dedupe_title_headings(root, title)
    _resolve_relative_urls(root, base_url)
    _remove_javascript_links(root)
    _unwrap_special_links(root)
    normalize_math(root)
    normalize_code_blocks(root)
    normalize_images(root, base_url)
    _standardize_inline_svgs(root)
    normalize_headings(root)
    normalize_callouts(root)
    _unwrap_layout_tables(root)
    _flatten_wrapper_divs(root)
    _unwrap_bare_spans(root)
    _remove_empty_elements(root)
    _remove_trailing_content(root)


_SVG_FILLED_TAGS = frozenset({"path", "rect", "circle", "ellipse", "polygon"})
_SVG_STROKE_TAGS = frozenset({"line", "polyline"})
_SVG_TEXT_TAGS = frozenset({"text", "tspan"})
_SVG_NON_RENDERED_ANCESTORS = frozenset(
    {"defs", "clippath", "mask", "pattern", "marker"}
)
_GRIDLINE_STROKE_OPACITY = "0.2"
_CLOSED_PATH_RE = re.compile(r"Z\s*$", re.IGNORECASE)


_LIGHT_DARK_RE = re.compile(r"light-dark\(\s*([^,]+?)\s*,\s*[^)]+?\)")
_CSS_VAR_RE = re.compile(r"var\(--([^,)]+)(?:,\s*([^)]+))?\)")
_CSS_VAR_GLOBAL_RE = re.compile(r"var\(--[^,)]+(?:,\s*[^)]+)?\)")
_SVG_COLOR_ATTRS = (
    "fill",
    "stroke",
    "color",
    "stop-color",
    "flood-color",
    "lighting-color",
)
_TW_COLOR_CLASS_RE = re.compile(r"^(fill|stroke)-([a-z]+)-(\d{2,3})(?:/(\d+))?$")
_TW_SPECIAL_CLASS_RE = re.compile(r"^(fill|stroke)-(black|white|transparent|current)$")
_TW_ARBITRARY_RE = re.compile(r"^text-\[(.+)\]$")
_TW_FONT_STYLES = {
    "font-semibold": "font-weight:600",
    "font-bold": "font-weight:700",
    "font-medium": "font-weight:500",
    "font-mono": "font-family:monospace",
}


def _standardize_inline_svgs(root: Tag) -> None:
    """Normalize inline SVGs whose class-based styling was lost.

    Pages that style chart SVGs from stylesheets render as invisible
    shapes once the CSS is gone. Resolve CSS variables / ``light-dark()``
    / Tailwind color classes to concrete colors, apply fallback
    fill/stroke, and strip class attributes (no CSS remains for them to
    reference). Ported from defuddle ``standardize.ts`` SVG
    normalization (``resolveVar``, ``resolveTailwindClasses``,
    ``applySvgFallbackStyles``, and the SVG branch of
    ``stripUnwantedAttributes``); the browser-only getComputedStyle
    resolution path is not applicable.
    """
    for svg in root.find_all("svg"):
        for el in [svg, *svg.find_all(True)]:
            if not isinstance(el, Tag):
                continue
            for attr in _SVG_COLOR_ATTRS:
                val = el.get(attr)
                if isinstance(val, str) and ("var(" in val or "light-dark(" in val):
                    el[attr] = _resolve_css_color(val)
            style = el.get("style")
            if isinstance(style, str) and ("var(" in style or "light-dark(" in style):
                resolved = _LIGHT_DARK_RE.sub(lambda m: m.group(1).strip(), style)
                resolved = _CSS_VAR_GLOBAL_RE.sub(
                    lambda m: _resolve_css_color(m.group(0)), resolved
                )
                el["style"] = resolved
            _resolve_tailwind_classes(el)

        _apply_svg_fallback_styles(svg)
        for el in [svg, *svg.find_all(True)]:
            if isinstance(el, Tag) and el.has_attr("class"):
                del el["class"]


def _resolve_css_color(value: str) -> str:
    """Resolve ``var()`` / ``light-dark()`` color values without a CSSOM.

    Uses the CSS fallback value when present, the Tailwind palette for
    ``--color-name-shade`` variables, then semantic guesses by variable
    name; ``currentColor`` otherwise.
    """
    value = _LIGHT_DARK_RE.sub(lambda m: m.group(1).strip(), value)
    if "var(" not in value:
        return value

    var_match = _CSS_VAR_RE.search(value)
    if var_match:
        fallback = (var_match.group(2) or "").strip()
        if fallback and "var(" not in fallback:
            return fallback

        name = var_match.group(1).lower()
        tw_match = re.search(r"(?:^|-)([a-z]+)-(\d{2,3})$", name)
        if tw_match:
            hex_color = TAILWIND_COLORS.get(tw_match.group(1), {}).get(
                tw_match.group(2)
            )
            if hex_color:
                return hex_color
        if name.endswith("-black"):
            return "#000"
        if name.endswith("-white"):
            return "#fff"

        # Semantic fallbacks
        if any(t in name for t in ("background", "card", "surface", "bg")):
            return "Canvas"
        if any(t in name for t in ("border", "divider", "separator")):
            return "#ccc"
        if any(t in name for t in ("muted", "subtle", "secondary", "placeholder")):
            return "#888"
    return "currentColor"


def _resolve_tailwind_classes(el: Tag) -> None:
    """Convert Tailwind fill/stroke/text utility classes to attributes."""
    classes = el.get("class")
    if not isinstance(classes, list) or not classes:
        return

    keep: list[str] = []
    styles: list[str] = []
    for token in classes:
        match = _TW_COLOR_CLASS_RE.match(token)
        if match:
            prop, color, shade, opacity = match.groups()
            hex_color = TAILWIND_COLORS.get(color, {}).get(shade)
            if hex_color:
                if opacity:
                    alpha = int(opacity) / 100
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                    el[prop] = f"rgba({r},{g},{b},{alpha})"
                else:
                    el[prop] = hex_color
                continue

        match = _TW_SPECIAL_CLASS_RE.match(token)
        if match:
            el[match.group(1)] = TAILWIND_SPECIAL[match.group(2)]
            continue

        match = _TW_ARBITRARY_RE.match(token)
        if match and not match.group(1).startswith(("#", "rgb", "hsl")):
            styles.append(f"font-size:{match.group(1)}")
            continue

        font_style = _TW_FONT_STYLES.get(token)
        if font_style is not None:
            styles.append(font_style)
            continue

        keep.append(token)

    if len(keep) == len(classes):
        return  # nothing changed
    if keep:
        el["class"] = " ".join(keep)
    else:
        del el["class"]
    if styles:
        existing = str(el.get("style") or "")
        sep = ";" if existing and not existing.endswith(";") else ""
        el["style"] = existing + sep + ";".join(styles)


def _apply_svg_fallback_styles(svg: Tag) -> None:
    """Apply fallback fill/stroke to class-styled shapes missing paint."""
    if svg.find("style") is not None:
        return

    all_els = [el for el in svg.find_all(True) if isinstance(el, Tag)]

    # Only apply fallbacks when at least one filled shape has a class but
    # no fill — indicating CSS-based styling was lost.
    if not any(
        el.name in _SVG_FILLED_TAGS
        and el.get("class")
        and not _in_non_rendered_svg_context(el)
        and not el.has_attr("fill")
        and not _has_style_prop(el, "fill")
        for el in all_els
    ):
        return

    for el in all_els:
        tag = el.name or ""
        is_filled = tag in _SVG_FILLED_TAGS
        is_stroke = tag in _SVG_STROKE_TAGS
        is_text = tag in _SVG_TEXT_TAGS
        if not (is_filled or is_stroke or is_text):
            continue
        if not el.get("class") or _in_non_rendered_svg_context(el):
            continue

        if is_text:
            if not el.has_attr("fill") and not _has_style_prop(el, "fill"):
                el["fill"] = "currentColor"
            continue

        has_fill = el.has_attr("fill") and el.get("fill") != "none"
        has_stroke = el.has_attr("stroke") or _has_style_prop(el, "stroke")

        if is_filled and not el.has_attr("fill") and not _has_style_prop(el, "fill"):
            el["fill"] = "none"

        if not has_stroke:
            if is_stroke:
                el["stroke"] = "currentColor"
                if not el.has_attr("stroke-opacity"):
                    el["stroke-opacity"] = _GRIDLINE_STROKE_OPACITY
            elif is_filled and not has_fill:
                d = str(el.get("d") or "")
                if not _CLOSED_PATH_RE.search(d.strip()):
                    el["stroke"] = "currentColor"


def _in_non_rendered_svg_context(el: Tag) -> bool:
    """Check if the element sits inside defs/clipPath/mask/pattern/marker."""
    return any(
        isinstance(p, Tag) and (p.name or "").lower() in _SVG_NON_RENDERED_ANCESTORS
        for p in el.parents
    )


def _has_style_prop(el: Tag, prop: str) -> bool:
    """Check if an inline style attribute sets a specific CSS property."""
    style = el.get("style")
    if not isinstance(style, str):
        return False
    return re.search(rf"(?:^|;)\s*{prop}\s*:", style) is not None


def _remove_comments(root: Tag) -> None:
    for comment in root.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()


def _dedupe_title_headings(root: Tag, title: str | None) -> None:
    if not title:
        return

    seen = False
    normalized_title = " ".join(title.split())
    for heading in root.find_all(["h1", "h2", "h3"]):
        text = " ".join(heading.get_text(" ", strip=True).split())
        if text != normalized_title:
            continue
        if seen:
            heading.decompose()
        else:
            seen = True


def _resolve_relative_urls(root: Tag, base_url: str) -> None:
    for tag in root.find_all(["a", "img"]):
        attr = "href" if tag.name == "a" else "src"
        value = tag.get(attr)
        if (
            not isinstance(value, str)
            or not value
            or value.startswith(("#", "http", "data:"))
        ):
            continue
        tag[attr] = urljoin(base_url, value)


def _remove_javascript_links(root: Tag) -> None:
    for link in root.find_all("a", href=True):
        href = str(link["href"]).strip().lower()
        if href.startswith("javascript:"):
            del link["href"]


_HEADING_NAMES = ("h1", "h2", "h3", "h4", "h5", "h6")


def _unwrap_special_links(root: Tag) -> None:
    """Unwrap links that Markdown cannot represent well.

    Ported from defuddle's ``unwrapSpecialLinks`` step (standardize.ts):

    - Links inside inline code — Markdown can't render links in backticks.
    - Card links wrapping block content with a direct-child heading:
      ``<a href="/x"><h2>T</h2><p>d</p></a>`` becomes
      ``<h2><a href="/x">T</a></h2><p>d</p>``.
    - Same-page anchor links (``href="#..."``) wrapping a heading (e.g.
      clickable section headers) — unwrapped entirely.
    """
    for link in list(root.find_all("a")):
        if link.find_parent("code") is not None:
            link.unwrap()

    for link in list(root.find_all("a")):
        if link.parent is None:
            continue
        href = str(link.get("href") or "")
        if not href or href.startswith("#"):
            continue
        heading = next(
            (
                c
                for c in link.children
                if isinstance(c, Tag) and c.name in _HEADING_NAMES
            ),
            None,
        )
        if heading is None:
            continue
        # Move the href into the heading by wrapping its children
        inner = BeautifulSoup("", "html.parser").new_tag("a", href=href)
        for child in list(heading.children):
            inner.append(child.extract())
        heading.append(inner)
        link.unwrap()

    for link in list(root.find_all("a", href=True)):
        if link.parent is None:
            continue
        if str(link["href"]).startswith("#") and link.find(_HEADING_NAMES):
            link.unwrap()


def _convert_h1_to_h2(root: Tag) -> None:
    """Convert H1 tags to H2 when there are multiple H1s."""
    h1s = root.find_all("h1")
    if len(h1s) <= 1:
        return
    for h1 in h1s:
        h1.name = "h2"


def _unwrap_bare_spans(root: Tag) -> None:
    """Remove <span> elements with no attributes, keeping their content."""
    for span in root.find_all("span"):
        if not span.attrs:
            span.unwrap()


def _remove_empty_elements(root: Tag) -> None:
    """Remove elements with no text content or children.

    Preserves void elements (img, br, hr, etc.), whitespace-significant
    content inside ``<pre>``/``<code>`` (syntax highlighters emit
    whitespace-only token spans), and the root itself.
    """
    for el in root.find_all(True):
        if el is root:
            continue
        if el.name in ALLOWED_EMPTY_ELEMENTS:
            continue
        # <pre>/<code> whitespace tokens and SVG shapes (line, path, …)
        # are meaningful despite having no text.
        if el.find_parent(("pre", "code", "svg")) is not None:
            continue
        if not el.get_text(strip=True) and not el.find(list(ALLOWED_EMPTY_ELEMENTS)):
            el.decompose()


def _remove_trailing_content(root: Tag) -> None:
    """Remove trailing <hr> elements and leading <hr> elements."""
    # Remove trailing hr
    children = [c for c in root.children if isinstance(c, Tag)]
    while children and children[-1].name == "hr":
        children[-1].decompose()
        children = [c for c in root.children if isinstance(c, Tag)]
    # Remove leading hr
    children = [c for c in root.children if isinstance(c, Tag)]
    while children and children[0].name == "hr":
        children[0].decompose()
        children = [c for c in root.children if isinstance(c, Tag)]


def _flatten_wrapper_divs(root: Tag) -> None:
    """Unwrap wrapper divs that add no semantic value.

    Targets divs that are purely structural wrappers:
    - Empty divs (no text, no children)
    - Divs with a single block-level child
    """
    changed = True
    while changed:
        changed = False
        for div in root.find_all("div"):
            if div is root:
                continue
            if div.name in _PRESERVE_ELEMENTS:
                continue

            # Skip divs with semantic roles or classes
            if not div.attrs:
                continue
            if div.get("role"):
                continue
            # Preserve the standardized footnotes container
            if div.get("id") == "footnotes":
                continue
            classes = div.get("class")
            if isinstance(classes, list) and any(
                c.lower() in ("article", "main", "content", "footnote", "reference")
                for c in classes
            ):
                continue

            # Empty div (only whitespace text)
            if not div.get_text(strip=True) and not div.find(
                list(ALLOWED_EMPTY_ELEMENTS)
            ):
                div.decompose()
                changed = True
                continue

            # Single block-level child — unwrap the wrapper
            tag_children = [c for c in div.children if isinstance(c, Tag)]
            text_children = [
                c
                for c in div.children
                if isinstance(c, NavigableString)
                and not isinstance(c, Comment)
                and c.strip()
            ]
            if (
                len(tag_children) == 1
                and not text_children
                and tag_children[0].name in _BLOCK_LEVEL
            ):
                div.unwrap()
                changed = True


def _unwrap_layout_tables(root: Tag) -> None:
    """Unwrap layout tables, preserving data tables.

    Layout tables are detected by:
    - Single-column tables (every row has ≤1 cell) without <th> headers
    - Tables containing nested tables

    Data tables (with <th> or multi-column structure) are preserved.
    Ported from defuddle's standardize.ts table unwrapping logic.
    """
    for table in list(root.find_all("table")):
        if table.parent is None:
            continue  # already detached

        # Get direct cells and rows (not from nested tables)
        direct_cells = [
            cell
            for cell in table.find_all(["td", "th"])
            if _is_direct_table_child(cell, table)
        ]
        direct_rows = [
            row for row in table.find_all("tr") if _is_direct_table_child(row, table)
        ]

        # Has nested tables → layout table, unwrap
        if table.find("table"):
            _unwrap_table_cells(table, direct_cells)
            continue

        # Skip data tables with header cells
        if any(cell.name == "th" for cell in direct_cells):
            continue

        # Skip if no rows
        if not direct_rows:
            continue

        # Check single-column: every row has at most 1 direct cell
        is_single_column = all(
            sum(1 for cell in direct_cells if cell.parent is row) <= 1
            for row in direct_rows
        )
        if is_single_column:
            _unwrap_table_cells(table, direct_cells)


def _is_direct_table_child(el: Tag, table: Tag) -> bool:
    """Check if element belongs directly to this table (not a nested one)."""
    parent = el.parent
    while parent is not None and parent is not table:
        if isinstance(parent, Tag) and parent.name == "table":
            return False  # belongs to a nested table
        parent = parent.parent
    return parent is table


def _unwrap_table_cells(table: Tag, cells: list[Tag]) -> None:
    """Replace a layout table with its cell contents."""
    # Collect all cell content
    fragments: list[Tag | NavigableString] = []
    for cell in cells:
        for child in list(cell.children):
            extracted = child.extract()
            if isinstance(extracted, (Tag, NavigableString)):
                fragments.append(extracted)

    # Replace table with cell contents
    for fragment in reversed(fragments):
        table.insert_after(fragment)
    table.decompose()
