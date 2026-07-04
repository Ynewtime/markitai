"""Footnote standardization ported from defuddle's ``elements/footnotes.ts``.

Collects footnote/citation definitions across common publisher formats
(Wikipedia/MediaWiki, Substack, Wikidot, arXiv, loose numbered paragraphs,
inline sidenotes, Word/Google Docs exports, ...) and rewrites the DOM to a
canonical structure:

- inline references become ``<sup id="fnref:N"><a href="#fn:N">N</a></sup>``
- definitions are collected into ``<div id="footnotes"><ol>`` with
  ``<li class="footnote" id="fn:N">`` items ending in back-reference links
  (``<a class="footnote-backref" href="#fnref:N">↩</a>``).

The markdown converter recognizes this structure and emits ``[^N]`` /
``[^N]: ...`` Markdown footnote syntax.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString, PageElement

from markitai.webextract.constants import (
    FOOTNOTE_INLINE_REFERENCES,
    FOOTNOTE_LIST_SELECTORS,
)

# Matches heading text for loose footnote section delimiters
FOOTNOTE_SECTION_RE = re.compile(
    r"^(foot\s*notes?|end\s*notes?|notes?|references?)$", re.IGNORECASE
)

# Return/backref symbols used as backlink text (Unicode arrows + ASCII caret)
_BACKREF_SYMBOLS_RE = re.compile(r"^[\^↩↥↑↵⤴⤵⏎]+$")

# MediaWiki cite_ref backref href pattern
_CITE_REF_RE = re.compile(r"^#cite_ref-")

# Numeric footnote marker with optional wrapping brackets/parens (e.g. "1",
# "[1]", "(23)", "([1])"). Group 1 is the digits.
_FOOTNOTE_MARKER_RE = re.compile(r"^\[?\(?(\d{1,4})\)?\]?$")

_HEADING_NAMES = ("h1", "h2", "h3", "h4", "h5", "h6")

# All block-level HTML elements (defuddle constants.ts BLOCK_LEVEL_ELEMENTS)
_BLOCK_LEVEL_ELEMENTS = frozenset(
    {
        "div", "section", "article", "main", "aside", "header", "footer",
        "nav", "content",
        "p", "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li", "dl", "dt", "dd",
        "pre", "blockquote", "figure", "figcaption",
        "table", "thead", "tbody", "tfoot", "tr", "td", "th",
        "details", "summary", "address", "hr",
        "form", "fieldset",
    }
)  # fmt: skip


# ---------------------------------------------------------------------------
# DOM helpers (BeautifulSoup equivalents of defuddle's utils/dom)
# ---------------------------------------------------------------------------


def _matches(el: Tag, selector: str) -> bool:
    try:
        return el.css.match(selector)
    except Exception:  # noqa: BLE001
        return False


def _closest(el: Tag, selector: str) -> Tag | None:
    try:
        return el.css.closest(selector)
    except Exception:  # noqa: BLE001
        return None


def _clone(el: Tag) -> Tag:
    return copy.copy(el)


def _transfer_content(source: Tag, target: Tag) -> None:
    """Move all child nodes from source to target (clearing target first)."""
    target.clear()
    for child in list(source.contents):
        target.append(child.extract())


def _tag_children(el: Tag) -> list[Tag]:
    return [c for c in el.contents if isinstance(c, Tag)]


def _first_element_child(el: Tag) -> Tag | None:
    for c in el.contents:
        if isinstance(c, Tag):
            return c
    return None


def _next_element_sibling(el: Tag) -> Tag | None:
    return el.find_next_sibling(True)


def _previous_element_sibling(el: Tag) -> Tag | None:
    return el.find_previous_sibling(True)


def _parent_element(el: Tag) -> Tag | None:
    parent = el.parent
    if isinstance(parent, Tag) and parent.name != "[document]":
        return parent
    return None


def _get_attr(el: Tag, name: str) -> str:
    value = el.get(name)
    if isinstance(value, list):
        return " ".join(value)
    return str(value) if value is not None else ""


def _get_id(el: Tag) -> str:
    return _get_attr(el, "id")


def _class_list(el: Tag) -> list[str]:
    value = el.get("class")
    if isinstance(value, list):
        return [str(c) for c in value]
    if isinstance(value, str):
        return value.split()
    return []


def _class_name(el: Tag) -> str:
    return " ".join(_class_list(el))


def _text_content(el: Tag | NavigableString | None) -> str:
    if el is None:
        return ""
    if isinstance(el, NavigableString):
        return str(el)
    return el.get_text()


def _contains(ancestor: Tag, node: PageElement) -> bool:
    """DOM ``contains``: node is ancestor itself or a descendant of it."""
    if node is ancestor:
        return True
    return any(parent is ancestor for parent in node.parents)


def _get_href_fragment(anchor: Tag | None) -> str:
    """Lowercase fragment id from an anchor's href (after the last ``#``)."""
    if anchor is None:
        return ""
    href = _get_attr(anchor, "href")
    return href.split("#")[-1].lower() if href else ""


def _is_attached(el: Tag) -> bool:
    return el.parent is not None


# ---------------------------------------------------------------------------
# Inline reference extraction rules
# ---------------------------------------------------------------------------


def _extract_footnoteref(el: Tag) -> str:
    link = el.select_one('a[id^="footnoteref-"]')
    if link is not None:
        m = re.match(r"^footnoteref-(\d+)$", _get_id(link))
        if m:
            return m.group(1)
    return ""


def _extract_science_org(el: Tag) -> str:
    xml_rid = _get_attr(el, "data-xml-rid")
    if xml_rid:
        return xml_rid
    href = _get_attr(el, "href")
    if href.startswith("#core-R"):
        return href.replace("#core-", "", 1)
    return ""


def _extract_mediawiki(el: Tag) -> str:
    ref_id = ""
    for link in el.select("a"):
        href = _get_attr(link, "href")
        segment = href.split("/")[-1] if href else ""
        m = re.search(r"(?:cite_note|cite_ref)-(.+)", segment)
        if m:
            ref_id = m.group(1).lower()
    return ref_id


def _extract_lesswrong_span(el: Tag) -> str:
    attr_id = _get_attr(el, "data-footnote-id")
    if attr_id:
        return attr_id
    el_id = _get_id(el)
    if el_id.startswith("fnref"):
        return el_id.replace("fnref", "", 1).lower()
    return ""


# Per-selector rules for extracting the footnote id from an inline reference
# element. First matching selector wins. Arxiv's multi-citation
# ``cite.ltx_cite`` is handled as a special case in standardize_footnotes.
_INLINE_REF_EXTRACTORS: list[tuple[str, Callable[[Tag], str]]] = [
    ("sup.footnoteref", _extract_footnoteref),
    # Nature.com
    ('a[id^="ref-link"]', lambda el: _text_content(el).strip()),
    # Science.org
    ('a[role="doc-biblioref"]', _extract_science_org),
    # Substack
    (
        "a.footnote-anchor, span.footnote-hovercard-target a",
        lambda el: _get_id(el).replace("footnote-anchor-", "", 1).lower(),
    ),
    # MediaWiki / Wikipedia: take the id from the last matching child anchor.
    ("sup.reference", _extract_mediawiki),
    (
        'sup[id^="fnref:"], span[id^="fnref:"]',
        lambda el: _get_id(el).replace("fnref:", "", 1).lower(),
    ),
    ('sup[id^="fnr"]', lambda el: _get_id(el).replace("fnr", "", 1).lower()),
    (
        "sup.footnote-reference",
        lambda el: _get_href_fragment(el.select_one('a[href^="#"]')),
    ),
    # LessWrong uses id="fnrefXXX" on the span when data-footnote-id is missing.
    ("span.footnote-reference", _extract_lesswrong_span),
    ("span.footnote-link", lambda el: _get_attr(el, "data-footnote-id")),
    ("a.citation", lambda el: _text_content(el).strip()),
    ('a[id^="fnref"]', lambda el: _get_id(el).replace("fnref", "", 1).lower()),
    # O'Reilly/HTMLBook: <a data-type="noteref" href="chNN.html#chMMfnK">
    ('a[data-type="noteref"]', _get_href_fragment),
]


@dataclass
class _FootnoteData:
    content: Tag
    original_id: str
    refs: list[str] = field(default_factory=list)


@dataclass
class _CollectState:
    footnotes: dict[int, _FootnoteData] = field(default_factory=dict)
    processed_ids: set[str] = field(default_factory=set)
    count: int = 1


class _FootnoteHandler:
    def __init__(self) -> None:
        self._factory = BeautifulSoup("", "html.parser")
        self._pending_removals: list[Tag] = []

    # -- small helpers ------------------------------------------------------

    def _new_tag(self, name: str, **attrs: str) -> Tag:
        return self._factory.new_tag(name, attrs=attrs)

    def _parse_fragment(self, html: str) -> list[PageElement]:
        if not html:
            return []
        return list(BeautifulSoup(html, "html.parser").contents)

    @staticmethod
    def _make_ref_id(footnote_number: str, refs_length: int) -> str:
        if refs_length > 0:
            return f"fnref:{footnote_number}-{refs_length + 1}"
        return f"fnref:{footnote_number}"

    @staticmethod
    def _merge_footnotes(
        target: dict[int, _FootnoteData], source: dict[int, _FootnoteData]
    ) -> None:
        for num, data in source.items():
            if num not in target:
                target[num] = data

    def _add_footnote(
        self,
        state: _CollectState,
        footnote_id: str,
        content: Tag,
        explicit_num: int | None = None,
    ) -> bool:
        """Record a footnote if id is non-empty and unseen.

        Without ``explicit_num``, assigns the next sequential slot. With
        ``explicit_num``, uses that key and advances ``state.count`` past it
        so later sequential adds don't collide.
        """
        if not footnote_id or footnote_id in state.processed_ids:
            return False
        key = explicit_num if explicit_num is not None else state.count
        state.footnotes[key] = _FootnoteData(content=content, original_id=footnote_id)
        state.processed_ids.add(footnote_id)
        if explicit_num is None:
            state.count += 1
        elif explicit_num >= state.count:
            state.count = explicit_num + 1
        return True

    # -- footnote item construction -----------------------------------------

    def create_footnote_item(
        self, footnote_number: int, content: Tag | str, refs: list[str]
    ) -> Tag:
        new_item = self._new_tag("li")
        new_item["class"] = "footnote"
        new_item["id"] = f"fn:{footnote_number}"

        if isinstance(content, str):
            paragraph = self._new_tag("p")
            for node in self._parse_fragment(content):
                paragraph.append(node.extract())
            new_item.append(paragraph)
        else:
            children = _tag_children(content)
            has_paragraphs = any(c.name == "p" for c in children)
            has_block_children = any(c.name in _BLOCK_LEVEL_ELEMENTS for c in children)
            if not has_paragraphs and not has_block_children:
                # Wrap inline content in a paragraph
                paragraph = self._new_tag("p")
                _transfer_content(content, paragraph)
                self.remove_backrefs(paragraph)
                new_item.append(paragraph)
            elif not has_paragraphs and has_block_children:
                # Append block children directly (avoid invalid <p><div>)
                for child in children:
                    if self.is_backref_link(child):
                        continue
                    clone = _clone(child)
                    self.remove_backrefs(clone)
                    new_item.append(clone)
            else:
                for child in children:
                    if self.is_backref_link(child):
                        continue
                    if child.name == "p":
                        if not _text_content(child).strip() and not child.select_one(
                            "img, br"
                        ):
                            continue
                        new_p = self._new_tag("p")
                        _transfer_content(child, new_p)
                        self.remove_backrefs(new_p)
                        new_item.append(new_p)
                    else:
                        clone = _clone(child)
                        self.remove_backrefs(clone)
                        new_item.append(clone)

        last_paragraph = new_item.select_one("p:last-of-type") or new_item
        for index, ref_id in enumerate(refs):
            backlink = self._new_tag("a")
            backlink["href"] = f"#{ref_id}"
            backlink["title"] = "return to article"
            backlink["class"] = "footnote-backref"
            text = "↩"
            if index < len(refs) - 1:
                text += " "
            backlink.string = text
            last_paragraph.append(backlink)

        return new_item

    # -- collection ---------------------------------------------------------

    def collect_footnotes(self, element: Tag) -> dict[int, _FootnoteData]:
        state = _CollectState()
        for note_list in element.select(FOOTNOTE_LIST_SELECTORS):
            if self._collect_from_list(element, note_list, state):
                continue

        fallbacks = [
            self._try_data_type_footnotes,
            self._try_generic_id_detection,
            self._try_word_export,
            self._try_google_docs,
            self._try_labeled_section,
            self._try_loose_footnotes,
            self._try_class_footnote,
        ]
        for fallback in fallbacks:
            if state.count > 1:
                break
            fallback(element, state)

        return state.footnotes

    def _collect_from_list(
        self, element: Tag, note_list: Tag, state: _CollectState
    ) -> bool:
        # Wikidot uses div.footnotes-footer containing div.footnote-footer
        if _matches(note_list, "div.footnotes-footer"):
            for div in note_list.select("div.footnote-footer"):
                m = re.match(r"^footnote-(\d+)$", _get_id(div))
                if not m:
                    continue
                footnote_id = m.group(1)
                if footnote_id in state.processed_ids:
                    continue
                # Clone to avoid modifying the original DOM
                clone = _clone(div)
                back_link = clone.select_one("a")
                if back_link is not None:
                    back_link.decompose()
                text = re.sub(r"^\s*\.\s*", "", clone.decode_contents())
                content_div = self._new_tag("div")
                for node in self._parse_fragment(text.strip()):
                    content_div.append(node.extract())
                self._add_footnote(state, footnote_id, content_div)
            return True

        # pulldown-cmark / mdBook / zola: standalone div.footnote-definition.
        # Skip if wrapped in div.footnote-definitions (next branch handles it).
        if _matches(note_list, "div.footnote-definition"):
            parent = _parent_element(note_list)
            if parent is not None and _matches(parent, "div.footnote-definitions"):
                return True
            footnote_id = _get_id(note_list).lower()
            clone = _clone(note_list)
            label = clone.select_one("sup.footnote-definition-label")
            if label is not None:
                label.decompose()
            self._add_footnote(state, footnote_id, clone)
            return True

        # Hugo/org-mode: div.footnote-definitions with div.footnote-definition
        # children: <sup id="footnote-N"><a href="#footnote-reference-N">N</a>
        # </sup><div class="footnote-body"><p>content</p></div>
        if _matches(note_list, "div.footnote-definitions"):
            for definition in note_list.select("div.footnote-definition"):
                sup_el = definition.select_one("sup[id]")
                body = definition.select_one(".footnote-body")
                if sup_el is None or body is None:
                    continue
                self._add_footnote(state, _get_id(sup_el).lower(), _clone(body))
            parent = _parent_element(note_list)
            if (
                parent is not None
                and parent is not element
                and "footnotes" in _class_list(parent)
            ):
                self._pending_removals.append(parent)
            return True

        # Easy Footnotes WP plugin: li items have no id, id is on a child span
        if _matches(note_list, "ol.easy-footnotes-wrapper"):
            for li in note_list.select("li.easy-footnote-single"):
                id_span = li.select_one('span[id^="easy-footnote-bottom-"]')
                if id_span is None:
                    continue
                clone = _clone(li)
                for sel in (
                    'span[id^="easy-footnote-bottom-"]',
                    "a.easy-footnote-to-top",
                ):
                    found = clone.select_one(sel)
                    if found is not None:
                        found.decompose()
                self._add_footnote(state, _get_id(id_span).lower(), clone)
            # Track empty anchor spans left in the body by the plugin
            self._pending_removals.extend(
                element.select("span.easy-footnote-margin-adjust")
            )
            return True

        # GNU Texinfo / makeinfo: div.footnotes-segment with a heading per
        # footnote followed by the body <p>. Inline markers reference the
        # heading anchor id, so register under that id.
        if _matches(note_list, "div.footnotes-segment"):
            for heading in note_list.select("h5.footnote-body-heading"):
                anchor = heading.select_one("a[id]")
                heading_id = _get_id(anchor).lower() if anchor is not None else ""
                if not heading_id:
                    continue
                content_div = self._new_tag("div")
                sibling = _next_element_sibling(heading)
                while sibling is not None and not (
                    sibling.name == "h5"
                    and "footnote-body-heading" in _class_list(sibling)
                ):
                    if _text_content(sibling).strip() or sibling.select_one("img, br"):
                        content_div.append(_clone(sibling))
                    sibling = _next_element_sibling(sibling)
                self._add_footnote(state, heading_id, content_div)
            self._pending_removals.append(note_list)
            return True

        # Substack has individual footnote divs with no parent
        if _matches(note_list, 'div.footnote[data-component-name="FootnoteToDOM"]'):
            anchor = note_list.select_one("a.footnote-number")
            content = note_list.select_one(".footnote-content")
            if anchor is not None and content is not None:
                self._add_footnote(
                    state,
                    _get_id(anchor).replace("footnote-", "", 1).lower(),
                    content,
                )
            return True

        for li in note_list.select('li, div[role="listitem"]'):
            footnote_id, content = self._extract_list_item_id_and_content(li)
            self._add_footnote(
                state, footnote_id, content if content is not None else li
            )
        return True

    # -- fallback collectors --------------------------------------------------

    def _try_data_type_footnotes(self, element: Tag, state: _CollectState) -> None:
        """O'Reilly / HTMLBook: <p data-type="footnote" id="chNNfnK"> definitions.

        Each opens with a <sup><a href="#…-marker">K</a></sup> backlink. Handle
        this explicit markup before the generic id fallback, which would
        mis-collect surrounding paragraphs as footnote content.
        """
        for definition in element.select('p[data-type="footnote"][id]'):
            def_id = _get_id(definition).lower()
            if not def_id:
                continue
            content_div = self._new_tag("div")
            clone = _clone(definition)
            # Strip the leading backlink marker (<sup><a href="#…">K</a></sup>).
            marker = _first_element_child(clone)
            if (
                marker is not None
                and marker.name == "sup"
                and marker.select_one('a[href*="#"]') is not None
            ):
                marker.decompose()
                self._trim_leading_whitespace(clone)
            content_div.append(clone)
            self._add_footnote(state, def_id, content_div)
            self._pending_removals.append(definition)

    def _try_generic_id_detection(self, element: Tag, state: _CollectState) -> None:
        """Detect footnotes by numeric anchor text referencing in-container ids."""
        candidate_refs: dict[str, list[Tag]] = {}
        for a in element.select('a[href*="#"]'):
            fragment = _get_href_fragment(a)
            if not fragment:
                continue
            text = _text_content(a).strip()
            if not _FOOTNOTE_MARKER_RE.match(text):
                continue
            candidate_refs.setdefault(fragment, []).append(a)

        if len(candidate_refs) < 2:
            return

        fragment_set = set(candidate_refs.keys())
        best_container: Tag | None = None
        best_match_count = 0
        for container in element.select("div, section, aside, footer, ol, ul"):
            if container is element:
                continue
            match_count = len(
                self._find_matching_footnote_elements(container, fragment_set)
            )
            if match_count >= 2 and match_count >= best_match_count:
                best_match_count = match_count
                best_container = container

        if best_container is None:
            return

        # Validate: require >=75% of external candidate refs (anchors outside
        # the container, i.e. not back-links) to point to footnote elements in
        # this container. Prevents equation/theorem cross-references from being
        # mis-classified as footnotes.
        ordered_elements = self._find_matching_footnote_elements(
            best_container, fragment_set
        )
        footnote_fragments = {frag for _, frag in ordered_elements}
        external_total = 0
        external_match = 0
        for frag, anchors in candidate_refs.items():
            if any(_contains(best_container, a) for a in anchors):
                continue  # back-link
            external_total += 1
            if frag in footnote_fragments:
                external_match += 1
        threshold = max(2, -(-external_total * 3 // 4))  # ceil(total * 0.75)
        if external_match < threshold:
            return

        for el, frag_id in ordered_elements:
            if frag_id in state.processed_ids:
                continue

            content_div = self._new_tag("div")
            clone = _clone(el)

            # Remove empty/numeric ID anchors (e.g. <a id="r1"></a>)
            id_anchor = clone.select_one(f'a[id="{frag_id}"]')
            if id_anchor is not None:
                anchor_text = _text_content(id_anchor).strip()
                if not anchor_text or re.match(r"^\d+[.)]*\s*$", anchor_text):
                    id_anchor.decompose()

            # Remove named anchor footnote marker (e.g. Gutenberg)
            named_anchor = clone.select_one("a[name]")
            if (
                named_anchor is not None
                and _get_attr(named_anchor, "name").lower() == frag_id
            ):
                named_anchor.decompose()

            first_node = clone.contents[0] if clone.contents else None
            if isinstance(first_node, NavigableString):
                new_text = re.sub(r"^\d+\.\s*", "", str(first_node))
                new_text = re.sub(r"^\s+", "", new_text)
                first_node.replace_with(NavigableString(new_text))

            if clone.name == "li":
                _transfer_content(clone, content_div)
            else:
                content_div.append(clone)

            sibling = _next_element_sibling(el)
            while sibling is not None and not _get_id(sibling):
                sib_anchor_id = self._get_child_anchor_id(sibling)
                if sib_anchor_id and sib_anchor_id in fragment_set:
                    break
                content_div.append(_clone(sibling))
                sibling = _next_element_sibling(sibling)

            self._add_footnote(state, frag_id, content_div)

        self._pending_removals.append(best_container)

    def _try_word_export(self, element: Tag, state: _CollectState) -> None:
        """Microsoft Word HTML export: refs #_ftn[N], back-links #_ftnref[N]."""
        word_backrefs = element.select('a[href*="#_ftnref"]')
        if len(word_backrefs) < 2:
            return

        pairs: list[tuple[int, Tag]] = []
        for anchor in word_backrefs:
            m = re.match(r"^_ftnref(\d+)$", _get_href_fragment(anchor))
            if m:
                pairs.append((int(m.group(1)), anchor))
        pairs.sort(key=lambda pair: pair[0])

        for num, anchor in pairs:
            original_id = f"_ftn{num}"
            if original_id in state.processed_ids:
                continue

            container = _parent_element(anchor)
            while container is not None and container is not element:
                if container.name in ("p", "div", "li"):
                    break
                container = _parent_element(container)
            if container is None or container is element:
                continue

            clone = _clone(container)
            backref_anchor = clone.select_one('a[href*="_ftnref"]')
            if backref_anchor is not None:
                wrap_sup = backref_anchor.find_parent("sup")
                if wrap_sup is not None:
                    wrap_sup.decompose()
                else:
                    backref_anchor.decompose()

            content_div = self._new_tag("div")
            content_div.append(clone)

            self._add_footnote(state, original_id, content_div, num)
            self._pending_removals.append(container)

    def _try_google_docs(self, element: Tag, state: _CollectState) -> None:
        """Google Docs/Sites: p[id^="ftnt"] with back-link a[href*="#ftnt_ref"]."""
        gdoc_pairs: list[tuple[int, Tag]] = []
        for p in element.select('p[id^="ftnt"]'):
            m = re.match(r"^ftnt(\d+)$", _get_id(p))
            if m:
                gdoc_pairs.append((int(m.group(1)), p))

        if len(gdoc_pairs) < 2:
            return

        gdoc_pairs.sort(key=lambda pair: pair[0])
        for num, el in gdoc_pairs:
            original_id = f"ftnt{num}"
            if original_id in state.processed_ids:
                continue

            clone = _clone(el)
            backref = clone.select_one('a[href*="#ftnt_ref"]')
            if backref is not None:
                backref.decompose()

            content_div = self._new_tag("div")
            content_div.append(clone)

            self._add_footnote(state, original_id, content_div, num)

            # Remove the paragraph and its wrapper div
            self._pending_removals.append(el)
            parent = _parent_element(el)
            if (
                parent is not None
                and parent is not element
                and parent.name == "div"
                and len(_tag_children(parent)) == 1
            ):
                self._pending_removals.append(parent)

        # Remove "Footnotes" heading preceding the first footnote
        first_el = gdoc_pairs[0][1]
        first_parent = _parent_element(first_el)
        scan_from = (
            first_parent
            if first_parent is not None
            and first_parent is not element
            and first_parent.name == "div"
            else first_el
        )
        prev = _previous_element_sibling(scan_from)
        if (
            prev is not None
            and prev.name in _HEADING_NAMES
            and FOOTNOTE_SECTION_RE.match(_text_content(prev).strip())
        ):
            self._pending_removals.append(prev)

    def _try_loose_footnotes(self, element: Tag, state: _CollectState) -> None:
        """Trailing numbered paragraphs cross-referenced by inline <sup>N</sup>."""
        numbered = self._find_loose_footnote_paragraphs(element)
        if numbered is None:
            return

        paragraphs, to_remove = numbered
        to_remove_ids = {id(el) for el in to_remove}
        for i, (num, def_para) in enumerate(paragraphs):
            next_def = paragraphs[i + 1][1] if i + 1 < len(paragraphs) else None

            content_div = self._strip_marker_and_wrap(def_para)
            sibling = _next_element_sibling(def_para)
            while (
                sibling is not None
                and sibling is not next_def
                and id(sibling) in to_remove_ids
            ):
                content_div.append(_clone(sibling))
                sibling = _next_element_sibling(sibling)

            self._add_footnote(state, str(num), content_div)

        self._pending_removals.extend(to_remove)

    def _try_class_footnote(self, element: Tag, state: _CollectState) -> None:
        """Class-based footnote paragraphs: <p class="footnote"><sup>N</sup>….

        The "footnote" class is a strong enough signal that no cross-validation
        or minimum count is required — a single footnote is detected.
        """
        footnote_paragraphs: list[tuple[int, Tag]] = []
        for p in element.select("p.footnote"):
            num = self._parse_footnote_num(p)
            if num is not None:
                footnote_paragraphs.append((num, p))

        for num, def_para in footnote_paragraphs:
            self._add_footnote(state, str(num), self._strip_marker_and_wrap(def_para))
        self._pending_removals.extend(el for _, el in footnote_paragraphs)

    def _try_labeled_section(self, element: Tag, state: _CollectState) -> None:
        """Containers whose class/id contains "footnote" with a matching heading.

        The heading (e.g. "Footnotes") is a strong enough signal that even a
        single footnote is detected. Handles CSS module hashes like
        ``PostDetail__UQuRMa__footnotes``.
        """
        for container in element.select("div, section, aside"):
            class_name = _class_name(container)
            container_id = _get_id(container)
            if not re.search(r"footnote", class_name, re.IGNORECASE) and not re.search(
                r"footnote", container_id, re.IGNORECASE
            ):
                continue

            heading = container.select_one("h1, h2, h3, h4, h5, h6")
            if heading is None or not FOOTNOTE_SECTION_RE.match(
                _text_content(heading).strip()
            ):
                continue

            paragraphs: list[tuple[int, Tag]] = []
            for p in container.select("p"):
                num = self._parse_footnote_num(p)
                if num is not None:
                    paragraphs.append((num, p))

            if not paragraphs:
                continue

            numbered_ids = {id(el) for _, el in paragraphs}
            for num, def_para in paragraphs:
                content_div = self._strip_marker_and_wrap(def_para)

                # Collect subsequent siblings until the next numbered paragraph
                sibling = _next_element_sibling(def_para)
                while sibling is not None and id(sibling) not in numbered_ids:
                    if _text_content(sibling).strip():
                        content_div.append(_clone(sibling))
                    self._pending_removals.append(sibling)
                    sibling = _next_element_sibling(sibling)

                self._add_footnote(state, str(num), content_div)
                self._pending_removals.append(def_para)

            self._pending_removals.append(container)
            break

    # -- marker parsing helpers ---------------------------------------------

    @staticmethod
    def _trim_leading_whitespace(parent: Tag) -> None:
        first = parent.contents[0] if parent.contents else None
        if isinstance(first, NavigableString):
            first.replace_with(NavigableString(re.sub(r"^\s+", "", str(first))))

    @staticmethod
    def _is_bold_wrapped_sup(el: Tag) -> bool:
        if el.name not in ("b", "strong"):
            return False
        first_child = el.contents[0] if el.contents else None
        first_element = _first_element_child(el)
        return (
            first_element is not None
            and first_child is first_element
            and first_element.name == "sup"
        )

    def _strip_marker_and_wrap(self, el: Tag) -> Tag:
        content_div = self._new_tag("div")
        clone = _clone(el)
        marker = _first_element_child(clone)
        if marker is not None:
            if self._is_bold_wrapped_sup(marker):
                inner = _first_element_child(marker)
                if inner is not None:
                    inner.decompose()
                self._trim_leading_whitespace(marker)
            else:
                marker.decompose()
                self._trim_leading_whitespace(clone)
        content_div.append(clone)
        return content_div

    def _parse_footnote_num(self, el: Tag) -> int | None:
        # Marker must be the very first child (no preceding text node)
        if not el.contents:
            return None
        first = _first_element_child(el)
        if first is None or first is not el.contents[0]:
            return None
        tag = first.name
        if self._is_bold_wrapped_sup(first):
            first = _first_element_child(first)
            if first is None:
                return None
            tag = "sup"
        if tag not in ("sup", "strong"):
            return None
        num_text = _text_content(first).strip()
        if not num_text.isdigit():
            return None
        num = int(num_text)
        return num if num >= 1 and str(num) == num_text else None

    def _cross_validate(self, element: Tag, paragraphs: list[tuple[int, Tag]]) -> bool:
        numbered_nums = {num for num, _ in paragraphs}
        matched_nums: set[int] = set()
        for sup in element.select("sup"):
            if any(_contains(el, sup) for _, el in paragraphs):
                continue
            if sup.select_one("a") is not None:
                continue  # already standardized or linked
            text = _text_content(sup).strip()
            if text.isdigit():
                n = int(text)
                if n >= 1 and str(n) == text and n in numbered_nums:
                    matched_nums.add(n)
        return len(matched_nums) >= 2

    def _find_loose_footnote_paragraphs(
        self, element: Tag
    ) -> tuple[list[tuple[int, Tag]], list[Tag]] | None:
        # Use parent of last <p> as scan container for nested layouts
        all_ps = element.select("p")
        container: Tag = element
        if all_ps:
            last_parent = _parent_element(all_ps[-1])
            container = last_parent if last_parent is not None else element
        children = _tag_children(container)

        # Strategy 1: forward-scan after the last <hr>
        for i in range(len(children) - 1, -1, -1):
            if children[i].name != "hr":
                continue

            paragraphs: list[tuple[int, Tag]] = []
            for j in range(i + 1, len(children)):
                num = self._parse_footnote_num(children[j])
                if num is not None:
                    paragraphs.append((num, children[j]))

            if len(paragraphs) >= 2 and self._cross_validate(element, paragraphs):
                return paragraphs, children[i:]
            break  # only check the last <hr>

        # Strategy 2: backward-scan for trailing numbered paragraphs
        trailing_numbered: list[tuple[int, Tag]] = []
        first_footnote_idx = -1
        for i in range(len(children) - 1, -1, -1):
            child = children[i]
            tag = child.name

            if tag == "p":
                num = self._parse_footnote_num(child)
                if num is not None:
                    trailing_numbered.insert(0, (num, child))
                    first_footnote_idx = i
                    continue
                break  # non-numbered paragraph — stop

            if tag in ("ul", "ol", "blockquote"):
                continue
            break  # any other element (heading, div, ...) — stop

        if len(trailing_numbered) >= 2 and self._cross_validate(
            element, trailing_numbered
        ):
            to_remove: list[Tag] = children[first_footnote_idx:]

            prev = _previous_element_sibling(trailing_numbered[0][1])
            if (
                prev is not None
                and prev.name in _HEADING_NAMES
                and FOOTNOTE_SECTION_RE.match(_text_content(prev).strip())
            ):
                to_remove.insert(0, prev)

            return trailing_numbered, to_remove

        # Strategy 3: footnotes followed by non-footnote trailing content, or
        # footnotes in a different container than the last <p>
        half_para_idx = len(all_ps) // 2
        scattered: list[tuple[int, Tag]] = []
        for p in all_ps[half_para_idx:]:
            num = self._parse_footnote_num(p)
            if num is not None:
                scattered.append((num, p))

        if len(scattered) >= 2 and self._cross_validate(element, scattered):
            return scattered, [el for _, el in scattered]

        return None

    # -- backref handling -----------------------------------------------------

    def is_backref_link(self, el: Tag) -> bool:
        if el.name != "a":
            return False
        text = _text_content(el).strip().replace("︎", "").replace("️", "")
        if _BACKREF_SYMBOLS_RE.match(text) or "footnote-backref" in _class_list(el):
            return True
        # MediaWiki multi-ref backrefs: <a href="#cite_ref-...">3.0</a>
        return bool(_CITE_REF_RE.match(_get_attr(el, "href")))

    def remove_backrefs(self, el: Tag) -> None:
        for a in el.select("a"):
            if self.is_backref_link(a):
                # Remove the wrapping <sup> if it only contained this link
                parent = _parent_element(a)
                if (
                    parent is not None
                    and parent.name == "sup"
                    and len(_tag_children(parent)) == 1
                ):
                    parent.decompose()
                else:
                    a.decompose()
        # Trim leading backref text nodes (bare "^" before multi-ref links)
        while el.contents and isinstance(el.contents[0], NavigableString):
            text = str(el.contents[0])
            if text and re.match(r"^[\s\^,.;]*$", text) and "^" in text:
                el.contents[0].extract()
            else:
                break
        while el.contents and isinstance(el.contents[-1], NavigableString):
            text = str(el.contents[-1])
            if re.match(r"^[\s,.;]*$", text):
                el.contents[-1].extract()
            else:
                break

    # -- id/content extraction helpers ---------------------------------------

    @staticmethod
    def _get_child_anchor_id(el: Tag) -> str:
        anchor = el.select_one("a[id], a[name]")
        if anchor is None:
            return ""
        return (_get_id(anchor) or _get_attr(anchor, "name")).lower()

    @staticmethod
    def _extract_list_item_id_and_content(li: Tag) -> tuple[str, Tag | None]:
        """Extract footnote id and content from a generic list item.

        Handles Science ``.citations``, Arxiv ``bib.bib*``, ``fn:*``, ``fn*``,
        Nature ``data-counter``, MediaWiki ``cite_note``.
        """
        citations_div = li.select_one(".citations")
        if citations_div is not None and _get_id(citations_div).lower().startswith("r"):
            return (
                _get_id(citations_div).lower(),
                citations_div.select_one(".citation-content"),
            )

        raw_id = _get_id(li).lower()
        # Order matters: `fn:` must precede `fn`, else `fn:3` strips to `:3`.
        for prefix in ("bib.bib", "fn:", "fn"):
            if raw_id.startswith(prefix):
                return raw_id[len(prefix) :], li
        if li.has_attr("data-counter"):
            counter = re.sub(r"\.$", "", _get_attr(li, "data-counter")).lower()
            return counter, li
        segment = raw_id.split("/")[-1]
        m = re.search(r"cite_note-(.+)", segment)
        return (m.group(1) if m else raw_id), li

    def _find_matching_footnote_elements(
        self, container: Tag, fragment_set: set[str]
    ) -> list[tuple[Tag, str]]:
        results: list[tuple[Tag, str]] = []
        seen: set[str] = set()
        for el in container.select("li, p, div"):
            matched_id = ""
            el_id = _get_id(el)
            if el_id and el_id.lower() in fragment_set:
                matched_id = el_id.lower()
            elif not el_id:
                anchor_id = self._get_child_anchor_id(el)
                if anchor_id and anchor_id in fragment_set:
                    matched_id = anchor_id
            if matched_id and matched_id not in seen:
                results.append((el, matched_id))
                seen.add(matched_id)
        return results

    # -- reference replacement -------------------------------------------------

    @staticmethod
    def _replace_container_preserving_text(container: Tag, footnote_ref: Tag) -> None:
        direct_text = ""
        has_child_elements = False
        for node in container.contents:
            if isinstance(node, NavigableString):
                direct_text += str(node)
            elif isinstance(node, Tag):
                has_child_elements = True
        direct_text = direct_text.strip()

        if direct_text and has_child_elements:
            container.replace_with(NavigableString(direct_text), footnote_ref)
        else:
            container.replace_with(footnote_ref)

    @staticmethod
    def _find_outer_footnote_container(el: Tag) -> Tag:
        current = el
        parent = _parent_element(el)

        while parent is not None:
            tag = parent.name
            if tag not in ("span", "sup"):
                break

            # Don't walk into spans with substantial non-footnote content
            if tag == "span":
                has_non_footnote_content = False
                for child in parent.contents:
                    if child is current:
                        continue
                    if isinstance(child, NavigableString) and str(child).strip():
                        has_non_footnote_content = True
                        break
                    if isinstance(child, Tag) and child.name != "sup":
                        has_non_footnote_content = True
                        break
                if has_non_footnote_content:
                    break
            current = parent
            parent = _parent_element(parent)

        return current

    def _create_footnote_reference(self, footnote_number: str, ref_id: str) -> Tag:
        sup = self._new_tag("sup")
        sup["id"] = ref_id
        link = self._new_tag("a")
        link["href"] = f"#fn:{footnote_number}"
        link.string = footnote_number
        sup.append(link)
        return sup

    # -- sidenote collectors ----------------------------------------------------

    def collect_inline_sidenotes(self, element: Tag) -> dict[int, _FootnoteData]:
        """Tufte-style and inline sidenotes embedded in text."""
        footnotes: dict[int, _FootnoteData] = {}
        containers = element.select(
            "span.footnote-container, span.sidenote-container, span.inline-footnote"
        )

        if not containers:
            # Org Mode CSS sidenotes: label.footref + input.footref-toggle +
            # span.sidenote
            footrefs = element.select("label.footref")
            if footrefs:
                footnote_count = 1
                for label in footrefs:
                    # Find the sidenote following this label (input may sit
                    # between them)
                    sibling = _next_element_sibling(label)
                    if (
                        sibling is not None
                        and sibling.name == "input"
                        and "footref-toggle" in _class_list(sibling)
                    ):
                        sibling = _next_element_sibling(sibling)
                    if (
                        sibling is None
                        or sibling.name != "span"
                        or "sidenote" not in _class_list(sibling)
                    ):
                        continue

                    content = _clone(sibling)
                    # Remove the leading sup number from the sidenote content
                    leading_sup = content.select_one("sup")
                    if leading_sup is not None and (
                        content.contents and content.contents[0] is leading_sup
                    ):
                        leading_sup.decompose()

                    footnotes[footnote_count] = _FootnoteData(
                        content=content,
                        original_id=str(footnote_count),
                        refs=[f"fnref:{footnote_count}"],
                    )

                    # Replace label + input + sidenote with a footnote ref
                    ref = self._create_footnote_reference(
                        str(footnote_count), f"fnref:{footnote_count}"
                    )
                    input_el = _next_element_sibling(label)
                    if (
                        input_el is not None
                        and input_el.name == "input"
                        and "footref-toggle" in _class_list(input_el)
                    ):
                        input_el.decompose()
                    sibling.decompose()
                    label.replace_with(ref)

                    footnote_count += 1

                # Remove the footer that duplicates these sidenotes
                for footer in element.select("footer"):
                    if footer.select_one(".footdef") is not None:
                        footer.decompose()

                return footnotes

            # Remove standalone sidenotes that duplicate the footnote list
            for sidenote in element.select("span.sidenote"):
                sidenote.decompose()
            return footnotes

        footnote_count = 1
        for container in containers:
            content = container.select_one(
                "span.footnote, span.sidenote, span.footnoteContent"
            )
            if content is None:
                continue

            footnotes[footnote_count] = _FootnoteData(
                content=_clone(content),
                original_id=str(footnote_count),
                refs=[f"fnref:{footnote_count}"],
            )

            ref = self._create_footnote_reference(
                str(footnote_count), f"fnref:{footnote_count}"
            )
            container.replace_with(ref)

            footnote_count += 1

        return footnotes

    def collect_sidenotes_column(self, element: Tag) -> dict[int, _FootnoteData]:
        """Sidenotes rendered in a separate column/container."""
        footnotes: dict[int, _FootnoteData] = {}

        columns = element.select(".sidenotes-column")

        # Sidenote columns are often siblings of an ancestor
        if not columns:
            ancestor = _parent_element(element)
            for _ in range(3):
                if ancestor is None or columns:
                    break
                columns = ancestor.select(":scope > .sidenotes-column")
                ancestor = _parent_element(ancestor)
        if not columns:
            return footnotes

        footnote_count = 1
        for column in columns:
            for sidenote in column.select(".sidenote[id]"):
                sidenote_id = _get_id(sidenote)
                if not sidenote_id:
                    continue

                id_span = sidenote.select_one(".sidenote__id")
                num_text = re.sub(r"\D", "", _text_content(id_span)) if id_span else ""
                footnote_number = int(num_text) if num_text else footnote_count

                content_div = self._new_tag("div")
                for node in list(sidenote.contents):
                    if isinstance(node, Tag):
                        classes = _class_list(node)
                        if (
                            "sidenote__id" in classes
                            or "sidenote__label" in classes
                            or "sn-backref" in classes
                        ):
                            continue
                    content_div.append(copy.copy(node))

                self.remove_backrefs(content_div)

                footnotes[footnote_number] = _FootnoteData(
                    content=content_div,
                    original_id=sidenote_id.lower(),
                )
                footnote_count += 1

            column.decompose()

        return footnotes

    def collect_aside_footnotes(self, element: Tag) -> dict[int, _FootnoteData]:
        """Footnotes in asides with numbered ordered lists."""
        footnotes: dict[int, _FootnoteData] = {}

        ols = element.select("aside > ol[start]")
        if not ols:
            return footnotes

        for ol in ols:
            aside = _parent_element(ol)
            start = _get_attr(ol, "start")
            if not start.isdigit():
                continue
            footnote_number = int(start)
            if footnote_number < 1:
                continue

            items = ol.select("li")
            if not items:
                continue

            content_div = self._new_tag("div")
            if len(items) == 1:
                _transfer_content(_clone(items[0]), content_div)
            else:
                for li in items:
                    p = self._new_tag("p")
                    _transfer_content(_clone(li), p)
                    content_div.append(p)

            footnotes[footnote_number] = _FootnoteData(
                content=content_div,
                original_id=str(footnote_number),
            )

            if aside is not None:
                aside.decompose()

        return footnotes

    def collect_hidden_aside_footnotes(self, element: Tag) -> dict[int, _FootnoteData]:
        """Hidden aside footnotes linked by data-definition attribute.

        Pattern: ``<span data-definition="id"><a href="#">*</a></span>`` +
        ``<aside style="display:none" id="id">content</aside>``.
        """
        footnotes: dict[int, _FootnoteData] = {}

        refs = element.select("span[data-definition]")
        if not refs:
            return footnotes

        aside_map: dict[str, Tag] = {}
        for aside in element.select("aside[id]"):
            aside_map[_get_id(aside)] = aside

        footnote_count = 1
        for ref in refs:
            def_id = _get_attr(ref, "data-definition")
            if not def_id:
                continue

            aside = aside_map.get(def_id)
            if aside is None:
                continue

            content_div = self._new_tag("div")
            _transfer_content(aside, content_div)
            aside.decompose()

            footnote_number = str(footnote_count)
            ref_id = f"fnref:{footnote_number}"
            footnotes[footnote_count] = _FootnoteData(
                content=content_div,
                original_id=def_id.lower(),
                refs=[ref_id],
            )

            ref.replace_with(self._create_footnote_reference(footnote_number, ref_id))
            footnote_count += 1

        return footnotes

    # -- main entry point --------------------------------------------------------

    def standardize_footnotes(self, element: Tag) -> None:
        sidenotes = self.collect_inline_sidenotes(element)
        footnotes = self.collect_hidden_aside_footnotes(element)
        self._merge_footnotes(footnotes, self.collect_footnotes(element))
        self._merge_footnotes(footnotes, self.collect_sidenotes_column(element))
        self._merge_footnotes(footnotes, self.collect_aside_footnotes(element))

        inline_references = element.select(FOOTNOTE_INLINE_REFERENCES)
        # Grouped sup containers: container id -> (container, refs)
        sup_groups: dict[int, tuple[Tag, list[tuple[str, str]]]] = {}

        footnotes_by_original_id: dict[str, tuple[str, _FootnoteData]] = {}
        for num in sorted(footnotes):
            data = footnotes[num]
            footnotes_by_original_id[data.original_id.lower()] = (str(num), data)

        for el in inline_references:
            if el.parent is None:
                continue
            if not _text_content(el).strip():
                continue

            # Arxiv multi-citation groups (e.g. [35, 2, 5]) expand into
            # several refs.
            if _matches(el, "cite.ltx_cite"):
                refs: list[Tag] = []
                for link in el.select("a"):
                    href = _get_attr(link, "href")
                    if not href:
                        continue
                    m = re.search(r"bib\.bib(\d+)", href.split("/")[-1])
                    if not m:
                        continue
                    entry = footnotes_by_original_id.get(m.group(1).lower())
                    if entry is None:
                        continue
                    fn_num, fn_data = entry
                    ref_id = self._make_ref_id(fn_num, len(fn_data.refs))
                    fn_data.refs.append(ref_id)
                    refs.append(self._create_footnote_reference(fn_num, ref_id))
                if refs:
                    container = self._find_outer_footnote_container(el)
                    nodes: list[PageElement] = []
                    for i, ref in enumerate(refs):
                        if i > 0:
                            nodes.append(NavigableString(" "))
                        nodes.append(ref)
                    container.replace_with(*nodes)
                continue

            footnote_id = ""
            for selector, extract in _INLINE_REF_EXTRACTORS:
                if _matches(el, selector):
                    footnote_id = extract(el)
                    break
            if not footnote_id:
                # Fallback: use the href fragment when no selector matched.
                href = el.get("href")
                if href is not None:
                    footnote_id = re.sub(r"^#", "", str(href)).lower()

            if not footnote_id:
                continue
            footnote_entry = footnotes_by_original_id.get(footnote_id.lower())
            if footnote_entry is None:
                continue

            footnote_number, footnote_data = footnote_entry
            container = self._find_outer_footnote_container(el)
            is_sup = container.name == "sup"

            # Dedupe: when an outer sup and its inner anchor both match the
            # same footnote, keep only the first reference.
            if is_sup:
                group = sup_groups.get(id(container))
                if group is not None and any(
                    num == footnote_number for num, _ in group[1]
                ):
                    continue

            ref_id = self._make_ref_id(footnote_number, len(footnote_data.refs))
            footnote_data.refs.append(ref_id)

            if is_sup:
                sup_groups.setdefault(id(container), (container, []))[1].append(
                    (footnote_number, ref_id)
                )
            else:
                self._replace_container_preserving_text(
                    container,
                    self._create_footnote_reference(footnote_number, ref_id),
                )

        # Fallback: match remaining unmatched footnotes
        unmatched = [
            (str(num), footnotes[num])
            for num in sorted(footnotes)
            if not footnotes[num].refs
        ]

        if unmatched:
            footnote_id_map: dict[str, tuple[str, _FootnoteData]] = {}
            footnote_num_map: dict[str, tuple[str, _FootnoteData]] = {}
            for num, data in unmatched:
                footnote_id_map[data.original_id] = (num, data)
                footnote_num_map[num] = (num, data)

            def is_inside_footnotes(el: Tag) -> bool:
                if _closest(el, '[id^="fnref:"]') is not None:
                    return True
                if _closest(el, "#footnotes") is not None:
                    return True
                return any(
                    _contains(g, el) for g in self._pending_removals if _is_attached(g)
                )

            def assign_ref(el: Tag, entry: tuple[str, _FootnoteData]) -> None:
                footnote_number, footnote_data = entry
                ref_id = self._make_ref_id(footnote_number, len(footnote_data.refs))
                footnote_data.refs.append(ref_id)
                container = self._find_outer_footnote_container(el)
                self._replace_container_preserving_text(
                    container,
                    self._create_footnote_reference(footnote_number, ref_id),
                )

            # Pass 1: Match by fragment link
            for link in element.select('a[href*="#"]'):
                if link.parent is None or is_inside_footnotes(link):
                    continue
                fragment = _get_href_fragment(link)
                if not fragment:
                    continue
                entry = footnote_id_map.get(fragment)
                if entry is None:
                    continue
                text = _text_content(link).strip()
                if not _FOOTNOTE_MARKER_RE.match(text):
                    continue
                assign_ref(link, entry)

            # Pass 2: Match sup/span elements with numeric text
            has_unmatched = any(not data.refs for data in footnotes.values())
            if has_unmatched:
                for el in element.select("sup, span.footnote-ref"):
                    if el.parent is None or _get_id(el).startswith("fnref:"):
                        continue
                    if _closest(el, "#footnotes") is not None:
                        continue
                    m = _FOOTNOTE_MARKER_RE.match(_text_content(el).strip())
                    if not m:
                        continue
                    entry = footnote_num_map.get(m.group(1)) or footnote_id_map.get(
                        m.group(1)
                    )
                    if entry is None or entry[1].refs:
                        continue
                    assign_ref(el, entry)

        for container, group_refs in sup_groups.values():
            replacements = [
                self._create_footnote_reference(num, ref_id)
                for num, ref_id in group_refs
            ]
            container.replace_with(*replacements)

        new_list = self._new_tag("div")
        new_list["id"] = "footnotes"
        ordered_list = self._new_tag("ol")
        all_footnotes = {**sidenotes, **footnotes}

        for number in sorted(all_footnotes):
            data = all_footnotes[number]
            ordered_list.append(
                self.create_footnote_item(number, data.content, data.refs)
            )

        for note_list in element.select(FOOTNOTE_LIST_SELECTORS):
            note_list.decompose()
        for el in self._pending_removals:
            if el.parent is not None:
                el.decompose()

        _remove_orphaned_dividers(element)

        if _tag_children(ordered_list):
            new_list.append(ordered_list)
            element.append(new_list)


def _remove_orphaned_dividers(element: Tag) -> None:
    """Remove leading/trailing <hr> and collapse consecutive <hr> elements."""
    # Remove leading <hr> elements (skipping whitespace text nodes)
    while True:
        node: PageElement | None = None
        for child in element.contents:
            if isinstance(child, NavigableString) and not str(child).strip():
                continue
            node = child
            break
        if isinstance(node, Tag) and node.name == "hr":
            node.decompose()
        else:
            break

    # Remove trailing <hr> elements (skipping whitespace text nodes)
    while True:
        node = None
        for child in reversed(element.contents):
            if isinstance(child, NavigableString) and not str(child).strip():
                continue
            node = child
            break
        if isinstance(node, Tag) and node.name == "hr":
            node.decompose()
        else:
            break

    # Collapse consecutive <hr> elements (skipping whitespace between them)
    for hr in element.select("hr"):
        if hr.parent is None:
            continue
        node = hr.next_sibling
        while node is not None:
            if isinstance(node, NavigableString) and not str(node).strip():
                node = node.next_sibling
                continue
            if isinstance(node, Tag) and node.name == "hr":
                nxt = node.next_sibling
                node.decompose()
                node = nxt
                continue
            break


def adopt_external_footnotes(main_content: Tag) -> None:
    """Pull footnote sections that live outside the main content element.

    Ported from defuddle's ``adoptExternalFootnotes``: scans the document body
    for footnote-labeled containers (class/id contains "footnote" with a
    matching section heading) that are disjoint from the main content and
    appends them to it.
    """
    body: Tag | None = None
    for ancestor in main_content.parents:
        if isinstance(ancestor, Tag) and ancestor.name == "body":
            body = ancestor
            break
    if body is None or main_content is body:
        return

    for el in body.select("div, section, aside"):
        class_name = _class_name(el)
        el_id = _get_id(el)
        if not re.search(r"footnote", class_name, re.IGNORECASE) and not re.search(
            r"footnote", el_id, re.IGNORECASE
        ):
            continue

        if _contains(main_content, el) or _contains(el, main_content):
            continue

        heading = el.select_one("h1, h2, h3, h4, h5, h6")
        if heading is None or not FOOTNOTE_SECTION_RE.match(
            _text_content(heading).strip()
        ):
            continue

        main_content.append(el.extract())


def standardize_footnotes(element: Tag) -> None:
    """Standardize footnote references and definitions in ``element``.

    Args:
        element: Content root (mutated in place).
    """
    handler = _FootnoteHandler()
    handler.standardize_footnotes(element)
