"""Project plain Word paragraphs and tables without constructing a second AST.

The supported subset follows Mammoth 1.11 default style and HTML collapsing
semantics (BSD-2-Clause; see NOTICE). Unsupported packages return None so the
caller retains Mammoth and its customizations, rich features, and error handling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape
from xml.etree.ElementTree import Element
from zipfile import ZipFile

from defusedxml import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class Unsupported(ValueError):
    pass


@dataclass
class Node:
    tag: str
    children: list[Node | str] = field(default_factory=list)
    merge: bool = False
    force: bool = False


def append(nodes: list[Node | str], item: Node | str | None) -> None:
    if item is None or item == "":
        return
    if isinstance(item, str):
        if nodes and isinstance(nodes[-1], str):
            nodes[-1] += item
        else:
            nodes.append(item)
    elif item.children or item.force or item.tag == "br":
        if (
            item.merge
            and nodes
            and isinstance(nodes[-1], Node)
            and nodes[-1].tag == item.tag
        ):
            for child in item.children:
                append(nodes[-1].children, child)
        else:
            nodes.append(item)


def serialize(node: Node | str) -> str:
    if isinstance(node, str):
        return escape(node, quote=False).replace('"', "&quot;")
    if node.tag == "br":
        return "<br />"
    return (
        "<"
        + node.tag
        + ">"
        + "".join(serialize(c) for c in node.children)
        + "</"
        + node.tag
        + ">"
    )


def children(node: Element, allowed: set[str]) -> None:
    if any(child.tag not in {W + name for name in allowed} for child in node):
        raise Unsupported(node.tag)


def first(node: Element | None, name: str) -> Element | None:
    return None if node is None else node.find(W + name)


def val(node: Element | None, name: str) -> str | None:
    child = first(node, name)
    return None if child is None else child.get(W + "val")


class Reader:
    def __init__(self, styles: Element, numbering: Element) -> None:
        self.styles: dict[tuple[str, str | None], str] = {}
        for style in styles:
            if style.tag != W + "style":
                continue
            key = (style.attrib[W + "type"], style.attrib[W + "styleId"])
            self.styles.setdefault(key, (val(style, "name") or "").upper())
        abstracts = {}
        for abstract in numbering.findall(W + "abstractNum"):
            if first(abstract, "numStyleLink") is not None:
                raise Unsupported("numbering style links")
            levels = {}
            unindexed = None
            for level in abstract.findall(W + "lvl"):
                index = level.get(W + "ilvl")
                item = (
                    index if index is not None else "0",
                    "ul" if val(level, "numFmt") == "bullet" else "ol",
                    val(level, "pStyle"),
                )
                if index is None:
                    unindexed = item
                else:
                    levels[index] = item
            if unindexed is not None:
                levels.setdefault("0", unindexed)
            abstracts[abstract.get(W + "abstractNumId")] = levels
        self.style_numbering = {
            item[2]: item
            for levels in abstracts.values()
            for item in levels.values()
            if item[2] is not None
        }
        self.numbering = {}
        for num in numbering.findall(W + "num"):
            if first(num, "lvlOverride") is not None:
                raise Unsupported("numbering overrides")
            if val(num, "abstractNumId") is None:
                raise Unsupported("missing abstract numbering reference")
            self.numbering[num.get(W + "numId")] = abstracts.get(
                val(num, "abstractNumId"), {}
            )

    def run(self, node: Element) -> list[Node | str]:
        children(node, {"rPr", "t", "tab", "br"})
        props = first(node, "rPr")
        if props is not None:
            children(
                props,
                {
                    "b",
                    "i",
                    "rStyle",
                    "rFonts",
                    "sz",
                    "szCs",
                    "color",
                    "lang",
                    "noProof",
                },
            )
        inner: list[Node | str] = []
        for child in node:
            if child.tag == W + "t":
                if len(child):
                    raise Unsupported("nested text")
                append(inner, child.text or "")
            elif child.tag == W + "tab":
                append(inner, "\t")
            elif child.tag == W + "br":
                if child.get(W + "type", "textWrapping") == "textWrapping":
                    append(inner, Node("br"))
        style = self.styles.get(("character", val(props, "rStyle")))
        wraps = []
        if props is not None:
            for property_name, tag in [("i", "em"), ("b", "strong")]:
                prop = first(props, property_name)
                if prop is not None and prop.get(W + "val") not in ("0", "false"):
                    wraps.append(tag)
        if style == "STRONG":
            wraps.append("strong")
        for tag in wraps:
            inner = [Node(tag, inner, merge=True)] if inner else []
        return inner

    def paragraph(self, node: Element) -> Node:
        children(node, {"pPr", "r"})
        props = first(node, "pPr")
        if props is not None:
            children(
                props,
                {
                    "pStyle",
                    "numPr",
                    "spacing",
                    "jc",
                    "ind",
                    "keepNext",
                    "keepLines",
                    "widowControl",
                    "outlineLvl",
                },
            )
        content: list[Node | str] = []
        for run in node.findall(W + "r"):
            for child in self.run(run):
                append(content, child)
        style_id = val(props, "pStyle")
        style_name = self.styles.get(("paragraph", style_id))
        tag = "p"
        for index in range(1, 7):
            if style_id == f"Heading{index}":
                tag = f"h{index}"
                break
        if tag == "p":
            for index in range(1, 7):
                if style_name == f"HEADING {index}":
                    tag = f"h{index}"
                    break
        if tag == "p" and (style_id == "Heading" or style_name == "HEADING"):
            tag = "h1"
        numbering = first(props, "numPr")
        if tag == "p" and style_name not in {
            "FOOTNOTE TEXT",
            "ENDNOTE TEXT",
            "ANNOTATION TEXT",
            "FOOTNOTE",
            "ENDNOTE",
        }:
            if numbering is not None:
                children(numbering, {"numId", "ilvl"})
            num_id = val(numbering, "numId")
            index = val(numbering, "ilvl")
            if num_id is not None and index is not None:
                level = self.numbering.get(num_id, {}).get(index)
            else:
                level = self.style_numbering.get(style_id)
                if level is None and num_id is not None:
                    level = self.numbering.get(num_id, {}).get("0")
            if level:
                if level[0] != "0":
                    raise Unsupported("nested lists")
                return Node(
                    level[1], [Node("li", content)] if content else [], merge=True
                )
        return Node(tag, content)

    def blocks(self, body: Element) -> list[Node | str]:
        result: list[Node | str] = []
        for block in body:
            if block.tag == W + "p":
                append(result, self.paragraph(block))
            elif block.tag == W + "tbl":
                append(result, self.table(block))
            elif block.tag != W + "sectPr":
                raise Unsupported(block.tag)
        return result

    def table(self, table: Element) -> Node:
        children(table, {"tblPr", "tblGrid", "tr"})
        rows: list[Node | str] = []
        for row in table.findall(W + "tr"):
            children(row, {"tc"})
            cells: list[Node | str] = []
            for cell in row:
                children(cell, {"tcPr", "p", "tbl"})
                props = first(cell, "tcPr")
                if props is not None:
                    children(props, {"tcW", "tcMar", "vAlign"})
                body = Element(W + "body")
                body.extend(child for child in cell if child.tag != W + "tcPr")
                cells.append(Node("td", self.blocks(body), force=True))
            rows.append(Node("tr", cells, force=True))
        return Node("table", rows, force=True)


def read_xml(data: bytes) -> Element:
    if b"\x00" in data:
        raise Unsupported("wide XML encoding")
    text = data.decode("utf-8-sig")
    if "<![CDATA[" in text or "<!DOCTYPE" in text:
        raise Unsupported("CDATA or DTD requires reference XML semantics")
    if text.startswith("<?xml"):
        header = text.partition("?>")[0]
        encoding = re.search(r"encoding\s*=\s*(['\"])(.*?)\1", header)
        if encoding and encoding.group(2).lower() != "utf-8":
            raise Unsupported("XML encoding")
    return ET.fromstring(data)


def validate_document(root: Element) -> None:
    # Inspect structure before opening and parsing the much larger styles part.
    # Property subtrees that the reference reader ignores need no traversal.
    allowed = {
        W + "document": {"body", "background"},
        W + "body": {"p", "tbl", "sectPr"},
        W + "p": {"pPr", "r"},
        W + "pPr": {
            "pStyle",
            "numPr",
            "spacing",
            "jc",
            "ind",
            "keepNext",
            "keepLines",
            "widowControl",
            "outlineLvl",
        },
        W + "r": {"rPr", "t", "tab", "br"},
        W + "rPr": {
            "b",
            "i",
            "rStyle",
            "rFonts",
            "sz",
            "szCs",
            "color",
            "lang",
            "noProof",
        },
        W + "numPr": {"numId", "ilvl"},
        W + "tbl": {"tblPr", "tblGrid", "tr"},
        W + "tr": {"tc"},
        W + "tc": {"tcPr", "p", "tbl"},
        W + "tcPr": {"tcW", "tcMar", "vAlign"},
        W + "t": set(),
    }
    plan = {tag: {W + child for child in values} for tag, values in allowed.items()}
    stack = [root]
    while stack:
        node = stack.pop()
        wanted = plan.get(node.tag)
        if wanted is not None:
            for child in node:
                if child.tag not in wanted:
                    raise Unsupported(child.tag)
                stack.append(child)


def _project_archive(
    archive: ZipFile, root: Element, document_bytes: bytes
) -> str | None:
    names = archive.namelist()
    if any(
        name in names
        for name in (
            "word/footnotes.xml",
            "word/endnotes.xml",
            "word/comments.xml",
            "mammoth/style-map",
            "word/_rels/footnotes.xml.rels",
            "word/_rels/endnotes.xml.rels",
            "word/_rels/comments.xml.rels",
        )
    ):
        return None
    if any(name.startswith("word/media/") for name in names):
        return None
    # Reuse the XML tree required by the existing equation scan. Only lexical
    # constructs with different reference-parser semantics cause a decline.
    if (
        b"\x00" in document_bytes
        or b"<![CDATA[" in document_bytes
        or b"<!DOCTYPE" in document_bytes
    ):
        return None
    raw = document_bytes.decode("utf-8-sig")
    if raw.startswith("<?xml"):
        match = re.search(r"encoding\s*=\s*(['\"])(.*?)\1", raw.partition("?>")[0])
        if match and match.group(2).lower() != "utf-8":
            return None
    if root.tag != W + "document":
        return None
    validate_document(root)
    # This experiment only accepts the conventional root part paths.
    for name in ("_rels/.rels", "word/_rels/document.xml.rels"):
        if name not in names:
            continue
        rels = read_xml(archive.read(name))
        if (
            rels.tag
            != "{http://schemas.openxmlformats.org/package/2006/relationships}Relationships"
        ):
            return None
        for rel in rels:
            if (
                rel.tag
                != "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"
            ):
                return None
            if not {"Id", "Target", "Type"} <= rel.attrib.keys():
                return None
            kind = rel.get("Type", "").rsplit("/", 1)[-1]
            if kind in {"footnotes", "endnotes", "comments"}:
                return None
            if kind in {"officeDocument", "styles", "numbering"}:
                expected = {
                    "officeDocument": "word/document.xml",
                    "styles": "styles.xml",
                    "numbering": "numbering.xml",
                }[kind]
                if rel.get("Target") != expected:
                    return None
    # The reference reads content types even for plain text; retain its errors.
    if "[Content_Types].xml" in names:
        types = read_xml(archive.read("[Content_Types].xml"))
        ns = "{http://schemas.openxmlformats.org/package/2006/content-types}"
        if types.tag != ns + "Types":
            return None
        for item in types:
            required = {
                ns + "Default": {"Extension", "ContentType"},
                ns + "Override": {"PartName", "ContentType"},
            }.get(item.tag)
            if required is None or not required <= item.attrib.keys():
                return None
    numbering = (
        read_xml(archive.read("word/numbering.xml"))
        if "word/numbering.xml" in names
        else Element(W + "numbering")
    )
    if numbering.tag != W + "numbering" or any(
        node.tag in {W + "numStyleLink", W + "lvlOverride"} for node in numbering.iter()
    ):
        return None
    styles = (
        read_xml(archive.read("word/styles.xml"))
        if "word/styles.xml" in names
        else Element(W + "styles")
    )
    if styles.tag != W + "styles":
        return None
    if any(
        "markup-compatibility" in node.tag
        for tree in (root, styles, numbering)
        for node in tree.iter()
    ):
        return None
    body = first(root, "body")
    if body is None:
        return None
    return "".join(serialize(node) for node in Reader(styles, numbering).blocks(body))


def plain_docx_html(
    archive: ZipFile, root: Element, document_bytes: bytes
) -> str | None:
    """Return reference-compatible HTML or decline to the established reader.

    `root` and `document_bytes` come from the caller's existing equation scan;
    this helper never reopens the DOCX or parses the document XML a second time.
    """
    try:
        return _project_archive(archive, root, document_bytes)
    except (Unsupported, KeyError, ValueError, ET.ParseError):
        return None
