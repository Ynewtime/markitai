//! Bounded OOXML presentation reader. Slide identity comes from package relationships.

use crate::opc::{self, Relationship};
use crate::{Asset, Document, Error, Result};
use quick_xml::{events::Event, name::ResolveResult};
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

const MAX_PART: u64 = 16 * 1024 * 1024;
const MAX_ASSET: u64 = 64 * 1024 * 1024;
const MAX_TOTAL: u64 = 256 * 1024 * 1024;
const MAX_SLIDES: usize = 10_000;
const MAX_INHERITED_XML: usize = 8 * 1024 * 1024;
const MAX_INHERITED_NODES: usize = 50_000;

fn error(message: impl std::fmt::Display) -> Error {
    Error::Conversion(format!("Presentation conversion failed: {message}"))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Ns {
    Presentation,
    Drawing,
    Chart,
    Diagram,
    Relationships,
    Compatibility,
    #[default]
    Other,
}

fn namespace(uri: &[u8]) -> Ns {
    match uri {
        b"http://schemas.openxmlformats.org/presentationml/2006/main"
        | b"http://purl.oclc.org/ooxml/presentationml/main" => Ns::Presentation,
        b"http://schemas.openxmlformats.org/drawingml/2006/main"
        | b"http://purl.oclc.org/ooxml/drawingml/main" => Ns::Drawing,
        b"http://schemas.openxmlformats.org/drawingml/2006/chart"
        | b"http://purl.oclc.org/ooxml/drawingml/chart" => Ns::Chart,
        b"http://schemas.openxmlformats.org/drawingml/2006/diagram"
        | b"http://purl.oclc.org/ooxml/drawingml/diagram" => Ns::Diagram,
        b"http://schemas.openxmlformats.org/package/2006/relationships" => Ns::Relationships,
        b"http://schemas.openxmlformats.org/markup-compatibility/2006" => Ns::Compatibility,
        _ => Ns::Other,
    }
}

#[derive(Clone, Debug, Default)]
struct Node {
    ns: Ns,
    name: String,
    attributes: BTreeMap<String, String>,
    relations: BTreeMap<String, String>,
    text: String,
    children: Vec<Node>,
}

impl Node {
    fn is(&self, ns: Ns, name: &str) -> bool {
        self.ns == ns && self.name == name
    }

    fn attr(&self, name: &str) -> Option<&str> {
        self.attributes.get(name).map(String::as_str)
    }

    fn relation(&self, name: &str) -> Option<&str> {
        self.relations.get(name).map(String::as_str)
    }

    fn child(&self, ns: Ns, name: &str) -> Option<&Self> {
        self.children.iter().find(|node| node.is(ns, name))
    }

    fn descendant(&self, ns: Ns, name: &str) -> Option<&Self> {
        if self.is(ns, name) {
            return Some(self);
        }
        self.children
            .iter()
            .find_map(|node| node.descendant(ns, name))
    }

    fn node_count(&self) -> usize {
        1 + self.children.iter().map(Self::node_count).sum::<usize>()
    }

    fn collect<'a>(&'a self, ns: Ns, name: &str, out: &mut Vec<&'a Self>) {
        if self.is(ns, name) {
            out.push(self);
        } else {
            for child in &self.children {
                child.collect(ns, name, out);
            }
        }
    }
}

fn xml(bytes: &[u8]) -> Result<Node> {
    let mut reader = quick_xml::NsReader::from_reader(bytes);
    let mut stack = vec![Node::default()];
    let mut nodes = 0usize;
    loop {
        let event = reader.read_event().map_err(error)?;
        let empty = matches!(event, Event::Empty(_));
        match event {
            Event::Start(event) | Event::Empty(event) => {
                nodes += 1;
                if nodes > 200_000 || stack.len() >= 128 {
                    return Err(error("XML structure exceeds node/depth limits"));
                }
                let ns = match reader.resolver().resolve_element(event.name()).0 {
                    ResolveResult::Bound(uri) => namespace(uri.into_inner().as_bytes()),
                    _ => Ns::Other,
                };
                let mut node = Node {
                    ns,
                    name: event.local_name().into_inner().to_owned(),
                    ..Node::default()
                };
                for attribute in event.attributes() {
                    let attribute = attribute.map_err(error)?;
                    let name = attribute.key.local_name().into_inner().to_owned();
                    let value = attribute
                        .normalized_value(quick_xml::XmlVersion::Implicit1_0)
                        .map_err(error)?
                        .into_owned();
                    match reader.resolver().resolve_attribute(attribute.key).0 {
                        ResolveResult::Unbound => {
                            node.attributes.insert(name, value);
                        }
                        ResolveResult::Bound(uri)
                            if matches!(
                                uri.into_inner(),
                                "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                                    | "http://purl.oclc.org/ooxml/officeDocument/relationships"
                            ) =>
                        {
                            node.relations.insert(name, value);
                        }
                        _ => {}
                    }
                }
                if empty {
                    stack.last_mut().unwrap().children.push(node);
                } else {
                    stack.push(node);
                }
            }
            Event::Text(value) => {
                stack
                    .last_mut()
                    .unwrap()
                    .text
                    .push_str(&value.xml_content(quick_xml::XmlVersion::Implicit1_0));
            }
            Event::CData(value) => {
                stack.last_mut().unwrap().text.push_str(&value);
            }
            Event::GeneralRef(value) => {
                let entity = format!("&{};", &*value);
                stack
                    .last_mut()
                    .unwrap()
                    .text
                    .push_str(&quick_xml::escape::unescape(&entity).map_err(error)?);
            }
            Event::End(_) => {
                if stack.len() < 2 {
                    return Err(error("unbalanced XML"));
                }
                let node = stack.pop().unwrap();
                stack.last_mut().unwrap().children.push(node);
            }
            Event::DocType(_) => return Err(error("document types are not accepted")),
            Event::Eof => break,
            _ => {}
        }
    }
    if stack.len() != 1 || stack[0].children.len() != 1 {
        return Err(error("XML must contain one complete root element"));
    }
    Ok(stack.pop().unwrap().children.remove(0))
}

struct Package<'a> {
    zip: opc::Zip<'a>,
    remaining: u64,
}

impl<'a> Package<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self> {
        Ok(Self {
            zip: opc::Zip::open(bytes, opc::MAX_ENTRIES).map_err(error)?,
            remaining: MAX_TOTAL,
        })
    }

    /// A part within `limit` and the package's remaining decompression budget.
    fn read(&mut self, path: &str, limit: u64) -> Result<Option<Vec<u8>>> {
        let bytes = self
            .zip
            .read(path, limit.min(self.remaining))
            .map_err(error)?;
        self.remaining -= bytes.as_ref().map_or(0, |bytes| bytes.len() as u64);
        Ok(bytes)
    }

    fn tree(&mut self, path: &str) -> Result<Node> {
        xml(&self
            .read(path, MAX_PART)?
            .ok_or_else(|| error(format!("missing part {path}")))?)
    }

    fn relationships(&mut self, part: &str) -> Result<BTreeMap<String, Relationship>> {
        let path = opc::relationships_part(part);
        let Some(bytes) = self.read(&path, MAX_PART)? else {
            return Ok(BTreeMap::new());
        };
        let root = xml(&bytes)?;
        let mut relationships = BTreeMap::new();
        for node in root
            .children
            .iter()
            .filter(|node| node.is(Ns::Relationships, "Relationship"))
        {
            let id = node
                .attr("Id")
                .ok_or_else(|| error("relationship has no ID"))?;
            let relationship = Relationship::new(
                node.attr("Type").unwrap_or(""),
                node.attr("Target").unwrap_or(""),
                node.attr("TargetMode"),
            );
            if relationships.insert(id.to_owned(), relationship).is_some() {
                return Err(error(format!("duplicate relationship {id} in {path}")));
            }
        }
        Ok(relationships)
    }
}

fn resolve(base: &str, target: &str) -> Result<String> {
    opc::resolve(base, target).map_err(error)
}

fn related(
    rels: &BTreeMap<String, Relationship>,
    base: &str,
    kind: &str,
) -> Result<Option<String>> {
    rels.values()
        .find(|rel| rel.kind.ends_with(kind) && !rel.external)
        .map(|rel| resolve(base, &rel.target))
        .transpose()
}

fn placeholder(shape: &Node) -> Option<&Node> {
    shape
        .children
        .iter()
        .filter(|node| {
            node.ns == Ns::Presentation
                && matches!(
                    node.name.as_str(),
                    "nvSpPr" | "nvPicPr" | "nvGraphicFramePr" | "nvCxnSpPr"
                )
        })
        .find_map(|node| {
            node.child(Ns::Presentation, "nvPr")?
                .child(Ns::Presentation, "ph")
        })
}

fn own_position(shape: &Node) -> (Option<i64>, Option<i64>) {
    let transform = shape
        .child(Ns::Presentation, "spPr")
        .or_else(|| shape.child(Ns::Presentation, "grpSpPr"))
        .and_then(|properties| properties.child(Ns::Drawing, "xfrm"))
        .or_else(|| shape.child(Ns::Presentation, "xfrm"));
    let offset = transform.and_then(|transform| transform.child(Ns::Drawing, "off"));
    (
        offset
            .and_then(|node| node.attr("y"))
            .and_then(|v| v.parse().ok()),
        offset
            .and_then(|node| node.attr("x"))
            .and_then(|v| v.parse().ok()),
    )
}

fn base_placeholder<'a>(shape: &Node, tree: Option<&'a Node>, by_index: bool) -> Option<&'a Node> {
    let ph = placeholder(shape)?;
    let tree = tree?.descendant(Ns::Presentation, "spTree")?;
    tree.children.iter().find(|candidate| {
        placeholder(candidate).is_some_and(|other| {
            if by_index {
                placeholder_index(ph)
                    .zip(placeholder_index(other))
                    .is_some_and(|(left, right)| left == right)
            } else {
                placeholder_family(ph.attr("type").unwrap_or("obj"))
                    == other.attr("type").unwrap_or("obj")
            }
        })
    })
}

fn placeholder_index(placeholder: &Node) -> Option<u32> {
    placeholder
        .attr("idx")
        .map_or(Some(0), |value| value.parse().ok())
}

fn placeholder_family(value: &str) -> &str {
    match value {
        "ctrTitle" => "title",
        "chart" | "clipArt" | "dgm" | "media" | "obj" | "pic" | "subTitle" | "tbl" => "body",
        value => value,
    }
}

/// The reference presentation writer's final pass: no trailing whitespace on
/// a line, and runs of blank lines (from text-free shapes) collapsed to one.
fn tidy(markdown: &str) -> String {
    let mut output = String::with_capacity(markdown.len());
    let mut blank = false;
    for (index, line) in markdown.split('\n').enumerate() {
        let line = line.trim_end();
        if line.is_empty() && blank {
            continue;
        }
        blank = line.is_empty();
        if index > 0 {
            output.push('\n');
        }
        output.push_str(line);
    }
    output
}

fn position(shape: &Node, layout: Option<&Node>, master: Option<&Node>) -> (i64, i64) {
    let (mut top, mut left) = own_position(shape);
    if let Some(base) = base_placeholder(shape, layout, true) {
        let (y, x) = own_position(base);
        top = top.or(y);
        left = left.or(x);
        if let Some(base) = base_placeholder(base, master, false) {
            let (y, x) = own_position(base);
            top = top.or(y);
            left = left.or(x);
        }
    }
    // The reference orders missing and zero coordinates before positive ones.
    (
        top.filter(|v| *v != 0).unwrap_or(i64::MIN),
        left.filter(|v| *v != 0).unwrap_or(i64::MIN),
    )
}

fn paragraph(node: &Node, output: &mut String) {
    if node.is(Ns::Drawing, "t") {
        output.push_str(&node.text);
    } else if node.is(Ns::Drawing, "br") {
        output.push('\n');
    } else {
        for child in &node.children {
            paragraph(child, output);
        }
    }
}

/// The nodes below `node` with one of `names` as their local name, whatever
/// their namespace (a Microsoft 365 part uses its own), not searching inside
/// a match.
fn collect_named<'a>(node: &'a Node, names: &[&str], out: &mut Vec<&'a Node>) {
    for child in &node.children {
        if names.contains(&child.name.as_str()) {
            out.push(child);
        } else {
            collect_named(child, names, out);
        }
    }
}

/// Text on one line, its whitespace runs collapsed.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn text_body(node: &Node) -> String {
    node.children
        .iter()
        .filter(|node| node.is(Ns::Drawing, "p"))
        .map(|node| {
            let mut text = String::new();
            paragraph(node, &mut text);
            text
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A hyperlink target the output may carry: web and mail links only. Script,
/// file and application-action targets stay plain text.
fn safe_link(target: &str) -> Option<&str> {
    let url = url::Url::parse(target).ok()?;
    matches!(url.scheme(), "http" | "https" | "mailto").then_some(target)
}

/// A run's hyperlink (`a:rPr/a:hlinkClick` naming an external relationship).
fn run_link<'a>(run: &Node, rels: &'a BTreeMap<String, Relationship>) -> Option<&'a str> {
    let id = run
        .child(Ns::Drawing, "rPr")?
        .child(Ns::Drawing, "hlinkClick")?
        .relation("id")?;
    let relationship = rels.get(id).filter(|relationship| relationship.external)?;
    safe_link(&relationship.target)
}

/// A paragraph's text: runs, with a break as a newline, and runs of one
/// hyperlink as one `[text](url)`.
fn paragraph_text(paragraph: &Node, rels: &BTreeMap<String, Relationship>) -> String {
    enum Piece<'a> {
        Text(String, Option<&'a str>),
        Break,
    }
    fn collect<'a>(
        node: &Node,
        rels: &'a BTreeMap<String, Relationship>,
        pieces: &mut Vec<Piece<'a>>,
    ) {
        if node.is(Ns::Drawing, "r") || node.is(Ns::Drawing, "fld") {
            let mut text = String::new();
            for child in node.children.iter().filter(|c| c.is(Ns::Drawing, "t")) {
                text.push_str(&child.text);
            }
            pieces.push(Piece::Text(text, run_link(node, rels)));
        } else if node.is(Ns::Drawing, "t") {
            pieces.push(Piece::Text(node.text.clone(), None));
        } else if node.is(Ns::Drawing, "br") {
            pieces.push(Piece::Break);
        } else {
            for child in &node.children {
                collect(child, rels, pieces);
            }
        }
    }
    let mut pieces = Vec::new();
    collect(paragraph, rels, &mut pieces);
    let mut output = String::new();
    let mut at = 0;
    while at < pieces.len() {
        match &pieces[at] {
            Piece::Break => {
                output.push('\n');
                at += 1;
            }
            Piece::Text(text, None) => {
                output.push_str(text);
                at += 1;
            }
            Piece::Text(_, Some(target)) => {
                let mut label = String::new();
                while let Some(Piece::Text(text, Some(next))) = pieces.get(at)
                    && next == target
                {
                    label.push_str(text);
                    at += 1;
                }
                let words = label.trim();
                if words.is_empty() {
                    output.push_str(&label);
                    continue;
                }
                let start = label.len() - label.trim_start().len();
                output.push_str(&label[..start]);
                output.push_str(&format!(
                    "[{}]({})",
                    words.replace('[', "\\[").replace(']', "\\]"),
                    super::destination(target)
                ));
                output.push_str(&label[start + words.len()..]);
            }
        }
    }
    output
}

/// A text body's paragraphs, one to a line, with hyperlinks.
fn linked_text_body(node: &Node, rels: &BTreeMap<String, Relationship>) -> String {
    node.children
        .iter()
        .filter(|node| node.is(Ns::Drawing, "p"))
        .map(|node| paragraph_text(node, rels))
        .collect::<Vec<_>>()
        .join("\n")
}

/// What a paragraph's bullet says at one layer of its text style.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Bullet {
    /// Nothing here: the next layer decides.
    Inherit,
    None,
    Mark,
    /// Auto-numbered, from this number.
    Number(u32),
}

fn bullet_of(properties: &Node) -> Bullet {
    for child in properties.children.iter().filter(|c| c.ns == Ns::Drawing) {
        match child.name.as_str() {
            "buNone" => return Bullet::None,
            "buAutoNum" => {
                let start = child.attr("startAt").and_then(|v| v.parse().ok());
                return Bullet::Number(start.unwrap_or(1).clamp(1, 32767));
            }
            "buChar" | "buBlip" => return Bullet::Mark,
            _ => {}
        }
    }
    Bullet::Inherit
}

/// The bullet a list style (`a:lstStyle`, `p:bodyStyle`, ...) gives a level.
fn level_bullet(styles: Option<&Node>, level: usize) -> Bullet {
    styles
        .and_then(|styles| styles.child(Ns::Drawing, &format!("lvl{}pPr", level + 1)))
        .map_or(Bullet::Inherit, bullet_of)
}

fn shape_list_style(shape: &Node) -> Option<&Node> {
    shape
        .child(Ns::Presentation, "txBody")?
        .child(Ns::Drawing, "lstStyle")
}

/// The master's text style for a shape: titles, other placeholders (body
/// text, subtitles, objects) and everything that is not a placeholder.
fn master_text_style<'a>(master: Option<&'a Node>, shape: &Node) -> Option<&'a Node> {
    let name = match placeholder(shape).map(|ph| ph.attr("type").unwrap_or("obj")) {
        Some("title" | "ctrTitle") => "titleStyle",
        Some("dt" | "ftr" | "sldNum" | "hdr") | None => "otherStyle",
        Some(_) => "bodyStyle",
    };
    master?
        .descendant(Ns::Presentation, "txStyles")?
        .child(Ns::Presentation, name)
}

/// The list styles a shape's paragraphs inherit bullets from, found once for
/// the shape: its own, the layout's placeholder, the master's placeholder and
/// the master's text styles.
struct BulletLayers<'a> {
    shape: Option<&'a Node>,
    layout: Option<&'a Node>,
    master_placeholder: Option<&'a Node>,
    master_text: Option<&'a Node>,
}

impl<'a> BulletLayers<'a> {
    fn new(shape: &'a Node, context: &SlideContext<'a>) -> Self {
        let layout = base_placeholder(shape, context.layout, true);
        let master = base_placeholder(layout.unwrap_or(shape), context.master, false);
        Self {
            shape: shape_list_style(shape),
            layout: layout.and_then(shape_list_style),
            master_placeholder: master.and_then(shape_list_style),
            master_text: master_text_style(context.master, shape),
        }
    }

    /// A paragraph's bullet: its own properties, then each layer's style for
    /// its level, the closest one that says anything deciding; none says
    /// nothing is no bullet.
    fn bullet(&self, paragraph: &Node, level: usize) -> Bullet {
        let own = paragraph
            .child(Ns::Drawing, "pPr")
            .map_or(Bullet::Inherit, bullet_of);
        [
            own,
            level_bullet(self.shape, level),
            level_bullet(self.layout, level),
            level_bullet(self.master_placeholder, level),
            level_bullet(self.master_text, level),
        ]
        .into_iter()
        .find(|bullet| *bullet != Bullet::Inherit)
        .unwrap_or(Bullet::None)
    }
}

/// A shape's text with its bulleted paragraphs as a Markdown list, a level
/// deeper for each level of the text, and its hyperlinks. A list stands apart
/// from the text before and after it, which would otherwise continue an item.
fn listed_text_body<'a>(
    body: &Node,
    shape: &'a Node,
    context: &SlideContext<'a>,
) -> (String, bool) {
    #[derive(PartialEq)]
    enum Last {
        Blank,
        Text,
        Item,
    }
    let mut lines: Vec<String> = Vec::new();
    let mut last = Last::Blank;
    // The enclosing items: their PowerPoint level and the column their text
    // starts at, which a deeper item is indented to.
    let mut open: Vec<(usize, usize)> = Vec::new();
    let mut counters = [None::<u32>; 9];
    let mut listed = false;
    let layers = BulletLayers::new(shape, context);
    for paragraph in body
        .children
        .iter()
        .filter(|node| node.is(Ns::Drawing, "p"))
    {
        let text = paragraph_text(paragraph, context.relationships);
        if text.trim().is_empty() {
            lines.push(String::new());
            last = Last::Blank;
            open.clear();
            counters = [None; 9];
            continue;
        }
        let level = paragraph
            .child(Ns::Drawing, "pPr")
            .and_then(|properties| properties.attr("lvl"))
            .and_then(|value| value.parse::<usize>().ok())
            .map_or(0, |level| level.min(8));
        let marker = match layers.bullet(paragraph, level) {
            Bullet::Mark => {
                counters[level..].fill(None);
                "*".to_owned()
            }
            Bullet::Number(start) => {
                let number = counters[level].map_or(start, |previous| previous.saturating_add(1));
                counters[level] = Some(number);
                counters[level + 1..].fill(None);
                format!("{number}.")
            }
            Bullet::None | Bullet::Inherit => {
                if last == Last::Item {
                    lines.push(String::new());
                }
                lines.push(text);
                last = Last::Text;
                open.clear();
                counters = [None; 9];
                continue;
            }
        };
        if last == Last::Text {
            lines.push(String::new());
        }
        while open
            .last()
            .is_some_and(|&(open_level, _)| open_level >= level)
        {
            open.pop();
        }
        let indent = open.last().map_or(0, |&(_, column)| column);
        let column = indent + marker.chars().count() + 1;
        open.push((level, column));
        let mut text_lines = text.split('\n');
        lines.push(format!(
            "{}{marker} {}",
            " ".repeat(indent),
            text_lines.next().unwrap_or("")
        ));
        for line in text_lines {
            lines.push(format!("{}{line}", " ".repeat(column)));
        }
        last = Last::Item;
        listed = true;
    }
    (lines.join("\n"), listed)
}

struct Reader<'a> {
    package: Package<'a>,
    document: Document,
    asset_parts: HashMap<String, String>,
    inherited: HashMap<String, Rc<Node>>,
    inherited_bytes: usize,
    inherited_nodes: usize,
}

#[derive(Clone, Copy)]
struct SlideContext<'a> {
    part: &'a str,
    relationships: &'a BTreeMap<String, Relationship>,
    layout: Option<&'a Node>,
    master: Option<&'a Node>,
    number: usize,
    title: Option<&'a Node>,
}

#[derive(Default)]
struct InheritedTrees {
    layout: Option<Rc<Node>>,
    master: Option<Rc<Node>>,
}

impl Reader<'_> {
    fn warn(&mut self, slide: usize, message: impl std::fmt::Display) {
        self.document
            .warnings
            .push(format!("Presentation slide {slide}: {message}"));
    }

    fn inherited_tree(&mut self, path: &str) -> Result<Rc<Node>> {
        if let Some(tree) = self.inherited.get(path) {
            return Ok(tree.clone());
        }
        let bytes = self
            .package
            .read(path, MAX_PART)?
            .ok_or_else(|| error(format!("missing part {path}")))?;
        let tree = Rc::new(xml(&bytes)?);
        let nodes = tree.node_count();
        if bytes.len() <= MAX_INHERITED_XML && nodes <= MAX_INHERITED_NODES {
            if self.inherited_bytes + bytes.len() > MAX_INHERITED_XML
                || self.inherited_nodes + nodes > MAX_INHERITED_NODES
            {
                self.inherited.clear();
                self.inherited_bytes = 0;
                self.inherited_nodes = 0;
            }
            self.inherited_bytes += bytes.len();
            self.inherited_nodes += nodes;
            self.inherited.insert(path.to_owned(), tree.clone());
        }
        Ok(tree)
    }

    fn picture(
        &mut self,
        shape: &Node,
        part: &str,
        rels: &BTreeMap<String, Relationship>,
    ) -> Result<String> {
        let label = shape
            .descendant(Ns::Presentation, "cNvPr")
            .and_then(|n| n.attr("descr"))
            .unwrap_or("")
            .replace(['[', ']'], " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let blip = shape.descendant(Ns::Drawing, "blip");
        let rid = blip
            .and_then(|b| b.relation("embed").or_else(|| b.relation("link")))
            .or_else(|| {
                let mut nodes = Vec::new();
                shape.collect(Ns::Other, "svgBlip", &mut nodes);
                nodes.first().and_then(|n| n.relation("embed"))
            });
        let rel = rid
            .and_then(|rid| rels.get(rid))
            .ok_or_else(|| error(format!("image relationship missing ({label})")))?;
        if rel.external {
            let url = url::Url::parse(&rel.target).map_err(error)?;
            if !matches!(url.scheme(), "http" | "https") {
                return Err(error("external image uses unsupported URL scheme"));
            }
            return Ok(format!(
                "\n![{}]({})\n",
                super::escape(&label),
                super::destination(&rel.target)
            ));
        }
        let path = resolve(part, &rel.target)?;
        let name = if let Some(name) = self.asset_parts.get(&path) {
            name.clone()
        } else {
            let bytes = self
                .package
                .read(&path, MAX_ASSET)?
                .ok_or_else(|| error(format!("image part missing: {path}")))?;
            let name = format!(
                "asset-{}.{}",
                self.document.assets.len() + 1,
                super::image_extension("", &path)
            );
            self.document.assets.push(Asset {
                name: name.clone(),
                bytes,
            });
            self.asset_parts.insert(path, name.clone());
            name
        };
        Ok(format!(
            "\n![{}](.markitai/assets/{name})\n",
            super::escape(&label)
        ))
    }

    /// The data an embedded OLE object holds (a worksheet, an Excel or MS
    /// Graph chart, an OpenDocument chart, a workbook package), read as a
    /// legacy deck's embedded objects are; `None` for a linked object or one
    /// holding nothing that reader reads (an equation, a document).
    fn embedded_object(
        &mut self,
        object: &Node,
        part: &str,
        rels: &BTreeMap<String, Relationship>,
    ) -> Result<Option<String>> {
        let Some(rel) = object.relation("id").and_then(|rid| rels.get(rid)) else {
            return Ok(None);
        };
        if rel.external {
            return Ok(None);
        }
        let path = resolve(part, &rel.target)?;
        let Some(bytes) = self.package.read(&path, MAX_ASSET)? else {
            return Ok(None);
        };
        let blocks = anydoc::embedded_object(&bytes).map_err(super::conversion_error)?;
        Ok((!blocks.is_empty()).then(|| super::object_markdown(&blocks)))
    }

    /// A SmartArt diagram's text points as a Markdown list, read from its
    /// data part (`dgm:relIds/@r:dm`) as the Word reader reads one; `None`
    /// for a diagram whose data part is missing or holds no text, which then
    /// keeps the frame's own DrawingML text.
    fn diagram(
        &mut self,
        ids: &Node,
        part: &str,
        rels: &BTreeMap<String, Relationship>,
        slide: usize,
    ) -> Option<String> {
        let rel = ids.relation("dm").and_then(|rid| rels.get(rid))?;
        if rel.external {
            return None;
        }
        let read = resolve(part, &rel.target).and_then(|path| self.package.read(&path, MAX_PART));
        let bytes = match read {
            Ok(bytes) => bytes?,
            Err(e) => {
                self.warn(slide, format!("diagram data not read: {e}"));
                return None;
            }
        };
        match anydoc::diagram_data(&bytes) {
            Ok(blocks) if !blocks.is_empty() => Some(super::object_markdown(&blocks)),
            Ok(_) => None,
            Err(e) => {
                self.warn(slide, format!("diagram data not read: {e}"));
                None
            }
        }
    }

    /// The names of the deck's comment authors by id, from the legacy
    /// `commentAuthors` part and the modern `authors` part.
    fn comment_authors(
        &mut self,
        presentation: &str,
        rels: &BTreeMap<String, Relationship>,
    ) -> HashMap<String, String> {
        let mut authors = HashMap::new();
        for rel in rels.values().filter(|rel| {
            !rel.external
                && (rel.kind.ends_with("/commentAuthors") || rel.kind.ends_with("/authors"))
        }) {
            let Ok(tree) =
                resolve(presentation, &rel.target).and_then(|path| self.package.tree(&path))
            else {
                continue;
            };
            let mut nodes = Vec::new();
            collect_named(&tree, &["cmAuthor", "author"], &mut nodes);
            for node in nodes {
                if let (Some(id), Some(name)) = (node.attr("id"), node.attr("name")) {
                    authors.insert(id.to_owned(), one_line(name));
                }
            }
        }
        authors
    }

    /// A slide's review comments, legacy (`p:cmLst`) and modern (Microsoft
    /// 365 threads with their replies), each as `Author: text` on one line,
    /// in the order their parts list them.
    fn slide_comments(
        &mut self,
        part: &str,
        rels: &BTreeMap<String, Relationship>,
        authors: &HashMap<String, String>,
        slide: usize,
    ) -> Vec<String> {
        let mut paths = rels
            .values()
            .filter(|rel| !rel.external && rel.kind.ends_with("/comments"))
            .filter_map(|rel| resolve(part, &rel.target).ok())
            .collect::<Vec<_>>();
        paths.sort();
        paths.dedup();
        let mut comments = Vec::new();
        for path in paths {
            let tree = match self.package.tree(&path) {
                Ok(tree) => tree,
                Err(e) => {
                    self.warn(slide, format!("comments unavailable: {e}"));
                    continue;
                }
            };
            let mut nodes = Vec::new();
            collect_named(&tree, &["cm"], &mut nodes);
            for comment in nodes {
                let mut replies = Vec::new();
                if let Some(list) = comment.children.iter().find(|n| n.name == "replyLst") {
                    collect_named(list, &["reply"], &mut replies);
                }
                for entry in std::iter::once(comment).chain(replies) {
                    let text = entry
                        .children
                        .iter()
                        .find_map(|child| match child.name.as_str() {
                            "text" => Some(child.text.clone()),
                            "txBody" => Some(text_body(child)),
                            _ => None,
                        })
                        .map(|text| one_line(&text))
                        .unwrap_or_default();
                    if text.is_empty() {
                        continue;
                    }
                    let text = super::escape(&text);
                    comments.push(
                        match entry.attr("authorId").and_then(|id| authors.get(id)) {
                            Some(name) if !name.is_empty() => {
                                format!("{}: {text}", super::escape(name))
                            }
                            _ => text,
                        },
                    );
                }
            }
        }
        comments
    }

    fn table(
        &mut self,
        table: &Node,
        rels: &BTreeMap<String, Relationship>,
        slide: usize,
    ) -> String {
        let rows = table.children.iter().filter(|node| node.is(Ns::Drawing, "tr")).map(|row| {
            row.children.iter().filter(|node| node.is(Ns::Drawing, "tc")).map(|cell| {
                if cell.attr("gridSpan").is_some_and(|v| v != "1") || cell.attr("rowSpan").is_some_and(|v| v != "1") {
                    self.warn(slide, "merged table cells retain their origin text and covered cells; Markdown has no merged-cell geometry");
                }
                super::super::text::cell(&cell.child(Ns::Drawing, "txBody").map(|body| linked_text_body(body, rels)).unwrap_or_default())
            }).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        format!("\n{}\n", super::super::text::table(&rows, true).trim_end())
    }

    fn chart(
        &mut self,
        frame: &Node,
        part: &str,
        rels: &BTreeMap<String, Relationship>,
        slide: usize,
    ) -> Result<String> {
        let rid = frame
            .descendant(Ns::Chart, "chart")
            .and_then(|node| node.relation("id"))
            .ok_or_else(|| error("chart relationship missing"))?;
        let rel = rels
            .get(rid)
            .filter(|rel| !rel.external)
            .ok_or_else(|| error("chart part is external or missing"))?;
        let path = resolve(part, &rel.target)?;
        let tree = self.package.tree(&path)?;
        let mut series = Vec::new();
        tree.collect(Ns::Chart, "ser", &mut series);
        let date1904 = tree
            .descendant(Ns::Chart, "date1904")
            .is_some_and(|node| matches!(node.attr("val"), Some("1" | "true") | None));
        let mut rows = Vec::new();
        for (index, series) in series.iter().enumerate() {
            let name = series
                .child(Ns::Chart, "tx")
                .and_then(|n| n.descendant(Ns::Chart, "v"))
                .map(|n| n.text.clone())
                .unwrap_or_else(|| format!("Series {}", index + 1));
            let values = cached_points(
                series
                    .child(Ns::Chart, "val")
                    .or_else(|| series.child(Ns::Chart, "yVal")),
                date1904,
            )?;
            let categories = cached_points(
                series
                    .child(Ns::Chart, "cat")
                    .or_else(|| series.child(Ns::Chart, "xVal")),
                date1904,
            )?;
            if rows.is_empty() {
                rows.push(vec!["Category".into()]);
            }
            rows[0].push(name);
            let count = values
                .keys()
                .chain(categories.keys())
                .max()
                .map_or(0, |i| i + 1);
            if count.max(rows.len()).saturating_mul(index + 2) > 1_000_000 {
                return Err(error("chart exceeds cell budget"));
            }
            for row in 0..count {
                if rows.len() <= row + 1 {
                    rows.push(vec![
                        categories
                            .get(&row)
                            .cloned()
                            .unwrap_or_else(|| (row + 1).to_string()),
                    ]);
                }
                rows[row + 1].resize(index + 1, String::new());
                rows[row + 1].push(values.get(&row).cloned().unwrap_or_default());
            }
        }
        if rows.len() < 2 {
            return Err(error(
                "chart has no cached category/value data; linked workbooks are not loaded",
            ));
        }
        if tree.descendant(Ns::Chart, "externalData").is_some() {
            self.warn(
                slide,
                "chart uses cached data; its linked workbook is not evaluated",
            );
        }
        let title = tree
            .descendant(Ns::Chart, "title")
            .map(|n| {
                let mut text = String::new();
                paragraph(n, &mut text);
                text
            })
            .unwrap_or_default();
        let title = if title.is_empty() {
            String::new()
        } else {
            format!(": {title}")
        };
        for row in &mut rows {
            for cell in row {
                *cell = super::super::text::cell(cell);
            }
        }
        Ok(format!(
            "\n\n### Chart{title}\n\n{}",
            super::super::text::table(&rows, true)
        ))
    }

    fn shapes(&mut self, parent: &Node, context: &SlideContext<'_>) -> String {
        let SlideContext {
            part,
            relationships: rels,
            layout,
            master,
            number: slide,
            title,
        } = *context;
        // Each shape's position is read once and the stable sort compares
        // the stored keys: the order of sorting by `position` itself, without
        // its lookups compiled into every step of the sort.
        let mut children = parent
            .children
            .iter()
            .map(|shape| (position(shape, layout, master), shape))
            .collect::<Vec<_>>();
        crate::sort::by_key(&mut children, |&(at, _)| at);
        let mut output = String::new();
        for (_, shape) in children {
            if shape.is(Ns::Compatibility, "AlternateContent") {
                if let Some(branch) = shape
                    .child(Ns::Compatibility, "Fallback")
                    .or_else(|| shape.child(Ns::Compatibility, "Choice"))
                {
                    self.warn(
                        slide,
                        "alternate-content branch rendered; advanced drawing appearance may differ",
                    );
                    output.push_str(&self.shapes(branch, context));
                }
                continue;
            }
            if shape.ns != Ns::Presentation {
                continue;
            }
            match shape.name.as_str() {
                "sp" | "cxnSp" => {
                    if let Some(body) = shape.child(Ns::Presentation, "txBody") {
                        let is_title = title.is_some_and(|title| std::ptr::eq(title, shape));
                        let (text, listed) = if is_title {
                            (linked_text_body(body, rels), false)
                        } else {
                            listed_text_body(body, shape, context)
                        };
                        // A list stands apart from the shapes around it, whose
                        // lines would continue an item or be taken into it.
                        if listed && !output.is_empty() && !output.ends_with("\n\n") {
                            output.push('\n');
                        }
                        if is_title && !text.trim().is_empty() {
                            output.push_str("# ");
                            output.push_str(text.trim_start());
                        } else {
                            output.push_str(&text);
                        }
                        output.push_str(if listed { "\n\n" } else { "\n" });
                    }
                }
                "grpSp" => output.push_str(&self.shapes(
                    shape,
                    &SlideContext {
                        layout: None,
                        master: None,
                        ..*context
                    },
                )),
                "pic" => match self.picture(shape, part, rels) {
                    Ok(image) => output.push_str(&image),
                    Err(e) => {
                        self.warn(slide, &e);
                        if let Some(label) = shape
                            .descendant(Ns::Presentation, "cNvPr")
                            .and_then(|n| n.attr("descr"))
                        {
                            output.push_str(label);
                            output.push('\n');
                        }
                    }
                },
                "graphicFrame" => {
                    if let Some(table) = shape.descendant(Ns::Drawing, "tbl") {
                        output.push_str(&self.table(table, rels, slide));
                    } else if shape.descendant(Ns::Chart, "chart").is_some() {
                        match self.chart(shape, part, rels, slide) {
                            Ok(chart) => output.push_str(&chart),
                            Err(e) => {
                                self.warn(slide, e);
                                output.push_str("\n\n[unsupported chart]\n\n");
                            }
                        }
                    } else if let Some(object) = shape.descendant(Ns::Presentation, "oleObj") {
                        match self.embedded_object(object, part, rels) {
                            Ok(Some(markdown)) => {
                                output.push_str("\n\n");
                                output.push_str(markdown.trim());
                                output.push_str("\n\n");
                            }
                            read => {
                                let program = object.attr("progId").unwrap_or("unnamed");
                                match read {
                                    Err(e) => self.warn(slide, format!("embedded object ({program}) not read: {e}; available DrawingML text retained")),
                                    _ => self.warn(slide, format!("embedded object ({program}) holds no data this reader reads; available DrawingML text retained")),
                                }
                                let mut text = String::new();
                                paragraph(shape, &mut text);
                                output.push_str(&text);
                                output.push('\n');
                            }
                        }
                    } else if let Some(Some(list)) = shape
                        .descendant(Ns::Diagram, "relIds")
                        .map(|ids| self.diagram(ids, part, rels, slide))
                    {
                        // A SmartArt diagram's text points, as a list.
                        output.push_str("\n\n");
                        output.push_str(list.trim());
                        output.push_str("\n\n");
                    } else {
                        let mut text = String::new();
                        paragraph(shape, &mut text);
                        self.warn(slide, "unsupported graphic frame; available DrawingML text retained without layout");
                        output.push_str(&text);
                        output.push('\n');
                    }
                }
                "nvGrpSpPr" | "grpSpPr" | "extLst" => {}
                _ => {
                    let mut text = String::new();
                    paragraph(shape, &mut text);
                    self.warn(
                        slide,
                        format!("unsupported shape {}; available text retained", shape.name),
                    );
                    output.push_str(&text);
                    output.push('\n');
                }
            }
        }
        output
    }
}

/// A series' cached points by index. Numbers read as the chart shows them,
/// through their cache's format code (or a point's own): a date category is
/// a date rather than its serial number, and `0%` a percentage. Text points
/// and numbers without a code stay as cached.
fn cached_points(node: Option<&Node>, date1904: bool) -> Result<BTreeMap<usize, String>> {
    let mut points = Vec::new();
    let mut numbers = None;
    if let Some(node) = node {
        numbers = node
            .descendant(Ns::Chart, "numCache")
            .or_else(|| node.descendant(Ns::Chart, "numLit"));
        node.collect(Ns::Chart, "pt", &mut points);
    }
    let format = numbers
        .and_then(|cache| cache.child(Ns::Chart, "formatCode"))
        .map(|code| code.text.trim())
        .filter(|code| !code.is_empty());
    let mut result = BTreeMap::new();
    for point in points {
        let index: usize = point.attr("idx").unwrap_or("0").parse().map_err(error)?;
        if index >= 100_000 {
            return Err(error("chart point index exceeds limit"));
        }
        if let Some(value) = point.child(Ns::Chart, "v") {
            let code = numbers.and(point.attr("formatCode").or(format));
            let text = match (code, value.text.trim().parse::<f64>()) {
                (Some(code), Ok(number)) if number.is_finite() => {
                    anydoc::format_number(code, number, date1904)
                }
                _ => value.text.clone(),
            };
            result.insert(index, text);
        }
    }
    Ok(result)
}

/// Count the ordered, referenced slides, including hidden and empty slides.
pub(crate) fn extract_presentation_count(bytes: &[u8]) -> Result<usize> {
    let mut package = Package::new(bytes)?;
    let root = package.relationships("")?;
    let presentation =
        related(&root, "", "/officeDocument")?.unwrap_or_else(|| "ppt/presentation.xml".into());
    let tree = package.tree(&presentation)?;
    let rels = package.relationships(&presentation)?;
    let list = tree
        .descendant(Ns::Presentation, "sldIdLst")
        .ok_or_else(|| error("presentation has no slide list"))?;
    let mut paths = std::collections::HashSet::new();
    for node in list
        .children
        .iter()
        .filter(|node| node.is(Ns::Presentation, "sldId"))
    {
        let rel = node
            .relation("id")
            .and_then(|id| rels.get(id))
            .filter(|rel| !rel.external && rel.kind.ends_with("/slide"))
            .ok_or_else(|| error("slide relationship missing or external"))?;
        let path = resolve(&presentation, &rel.target)?;
        if !paths.insert(path.clone()) || paths.len() > MAX_SLIDES {
            return Err(error("duplicate slide reference or slide limit exceeded"));
        }
        let slide = package.tree(&path)?;
        if !slide.is(Ns::Presentation, "sld") {
            return Err(error("slide relationship does not identify a slide"));
        }
    }
    if paths.is_empty() {
        return Err(error("presentation has no slides"));
    }
    Ok(paths.len())
}

pub(super) fn extract_presentation(bytes: &[u8]) -> Result<Document> {
    let mut reader = Reader {
        package: Package::new(bytes)?,
        document: Document::default(),
        asset_parts: HashMap::new(),
        inherited: HashMap::new(),
        inherited_bytes: 0,
        inherited_nodes: 0,
    };
    let root = reader.package.relationships("")?;
    let presentation =
        related(&root, "", "/officeDocument")?.unwrap_or_else(|| "ppt/presentation.xml".into());
    let tree = reader.package.tree(&presentation)?;
    let rels = reader.package.relationships(&presentation)?;
    let list = tree
        .descendant(Ns::Presentation, "sldIdLst")
        .ok_or_else(|| error("presentation has no slide list"))?;
    let slides = list
        .children
        .iter()
        .filter(|node| node.is(Ns::Presentation, "sldId"))
        .collect::<Vec<_>>();
    if slides.is_empty() || slides.len() > MAX_SLIDES {
        return Err(error("slide count is empty or exceeds 10,000"));
    }
    let authors = reader.comment_authors(&presentation, &rels);
    let mut pages = Vec::new();
    let mut readable = 0usize;
    for (index, node) in slides.iter().enumerate() {
        let number = index + 1;
        let mut page = format!("<!-- Slide number: {number} -->\n");
        let result = (|| -> Result<String> {
            let rel = node
                .relation("id")
                .and_then(|id| rels.get(id))
                .filter(|rel| !rel.external && rel.kind.ends_with("/slide"))
                .ok_or_else(|| error("slide relationship missing or external"))?;
            let path = resolve(&presentation, &rel.target)?;
            let tree = reader.package.tree(&path)?;
            let shapes = tree
                .descendant(Ns::Presentation, "spTree")
                .ok_or_else(|| error("slide has no shape tree"))?;
            // A slide hidden from the slide show keeps its content, marked.
            let hidden = matches!(tree.attr("show"), Some("0" | "false"));
            let relationships = match reader.package.relationships(&path) {
                Ok(rels) => rels,
                Err(e) => {
                    reader.warn(number, e);
                    BTreeMap::new()
                }
            };
            let inheritance = (|| -> Result<InheritedTrees> {
                let Some(path) = related(&relationships, &path, "/slideLayout")? else {
                    return Ok(InheritedTrees::default());
                };
                let layout = reader.inherited_tree(&path)?;
                let rels = reader.package.relationships(&path)?;
                let master = related(&rels, &path, "/slideMaster")?
                    .map(|path| reader.inherited_tree(&path))
                    .transpose()?;
                Ok(InheritedTrees {
                    layout: Some(layout),
                    master,
                })
            })();
            let InheritedTrees { layout, master } = match inheritance {
                Ok(trees) => trees,
                Err(e) => {
                    reader.warn(number, format!("placeholder geometry unavailable: {e}"));
                    InheritedTrees::default()
                }
            };
            let title = shapes.children.iter().find(|shape| {
                placeholder(shape).is_some_and(|ph| placeholder_index(ph) == Some(0))
            });
            let context = SlideContext {
                part: &path,
                relationships: &relationships,
                layout: layout.as_deref(),
                master: master.as_deref(),
                number,
                title,
            };
            let mut content = reader.shapes(shapes, &context);
            if hidden {
                content.insert_str(0, "<!-- Hidden slide -->\n");
            }
            let notes_path = match related(&relationships, &path, "/notesSlide") {
                Ok(path) => path,
                Err(e) => {
                    reader.warn(number, format!("notes relationship unavailable: {e}"));
                    None
                }
            };
            if let Some(notes_path) = notes_path {
                content.push_str("\n\n### Notes:\n");
                match reader.package.tree(&notes_path) {
                    Ok(notes) => {
                        let mut shapes = Vec::new();
                        notes.collect(Ns::Presentation, "sp", &mut shapes);
                        let bodies = shapes
                            .iter()
                            .filter(|shape| {
                                placeholder(shape).is_some_and(|ph| ph.attr("type") == Some("body"))
                            })
                            .collect::<Vec<_>>();
                        if bodies.is_empty() && !shapes.is_empty() {
                            reader.warn(number, "notes have no body placeholder; only ordinary text boxes were retained");
                        }
                        for shape in shapes {
                            if (placeholder(shape).is_none()
                                || placeholder(shape)
                                    .is_some_and(|ph| ph.attr("type") == Some("body")))
                                && let Some(body) = shape.child(Ns::Presentation, "txBody")
                            {
                                content.push_str(&text_body(body));
                                content.push('\n');
                            }
                        }
                    }
                    Err(e) => reader.warn(number, format!("notes unavailable: {e}")),
                }
            }
            // Review comments close the slide, after its speaker notes.
            let comments = reader.slide_comments(&path, &relationships, &authors, number);
            if !comments.is_empty() {
                content.push_str("\n\n### Comments:\n");
                for comment in comments {
                    content.push_str("* ");
                    content.push_str(&comment);
                    content.push('\n');
                }
            }
            Ok(content)
        })();
        match result {
            Ok(content) => {
                readable += 1;
                page.push_str(&content);
            }
            Err(e) => reader.warn(
                number,
                format!("slide content unavailable; numbered marker retained: {e}"),
            ),
        }
        pages.push(page.trim().to_owned());
    }
    if readable == 0 {
        return Err(error("no slide could be read"));
    }
    reader.document.markdown = tidy(&pages.join("\n\n"));
    reader
        .document
        .metadata
        .insert("converter".into(), "native-presentation".into());
    Ok(reader.document)
}

#[cfg(test)]
mod tests {
    #[test]
    fn presentation_output_drops_trailing_spaces_and_collapses_blank_runs_like_the_reference() {
        assert_eq!(
            super::tidy("<!-- Slide number: 2 -->\n\n\n\n# Title  \n\nBody\n\n\n\nFooter"),
            "<!-- Slide number: 2 -->\n\n# Title\n\nBody\n\nFooter"
        );
    }

    use super::*;
    use std::io::{Cursor, Write};

    const P: &str = "http://schemas.openxmlformats.org/presentationml/2006/main";
    const A: &str = "http://schemas.openxmlformats.org/drawingml/2006/main";
    const R: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
    const C: &str = "http://schemas.openxmlformats.org/drawingml/2006/chart";

    fn slide(shapes: &str) -> String {
        format!(
            r#"<p:sld xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}" xmlns:c="{C}"><p:cSld><p:spTree>{shapes}</p:spTree></p:cSld></p:sld>"#
        )
    }

    fn shape(text: &str, kind: &str, y: i64) -> String {
        let ph = if kind.is_empty() {
            String::new()
        } else {
            format!(r#"<p:ph type="{kind}"/>"#)
        };
        format!(
            r#"<p:sp><p:nvSpPr><p:cNvPr id="1" name="shape"/><p:nvPr>{ph}</p:nvPr></p:nvSpPr><p:spPr><a:xfrm><a:off x="1" y="{y}"/></a:xfrm></p:spPr><p:txBody><a:lstStyle/><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp>"#
        )
    }

    fn relationships(entries: &[(&str, &str, &str)]) -> String {
        let entries = entries
            .iter()
            .map(|(id, kind, target)| {
                format!(r#"<Relationship Id="{id}" Type="{R}/{kind}" Target="{target}"/>"#)
            })
            .collect::<String>();
        format!(
            r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{entries}</Relationships>"#
        )
    }

    fn package(order: &[(&str, &str)], parts: Vec<(&str, Vec<u8>)>) -> Vec<u8> {
        let ids = order
            .iter()
            .enumerate()
            .map(|(index, (id, _))| format!(r#"<p:sldId id="{}" r:id="{id}"/>"#, index + 256))
            .collect::<String>();
        let presentation = format!(
            r#"<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst>{ids}</p:sldIdLst></p:presentation>"#
        );
        let rels = relationships(
            &order
                .iter()
                .map(|(id, target)| (*id, "slide", *target))
                .collect::<Vec<_>>(),
        );
        let mut all = vec![
            ("ppt/presentation.xml", presentation.into_bytes()),
            ("ppt/_rels/presentation.xml.rels", rels.into_bytes()),
        ];
        all.extend(parts);
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, bytes) in all {
            writer
                .start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(&bytes).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    #[test]
    fn slide_order_empty_pages_and_aliases_do_not_depend_on_headings() {
        let last = slide(&format!(
            "{}{}",
            shape("Second paragraph", "", 20),
            shape("Slide title", "title", 10)
        ));
        let first = slide(&shape("Untitled paragraph", "", 10));
        let bytes = package(
            &[
                ("rB", "slides/z.xml"),
                ("rA", "slides/a.xml"),
                ("rE", "slides/empty.xml"),
            ],
            vec![
                ("ppt/slides/a.xml", first.into_bytes()),
                ("ppt/slides/empty.xml", slide("").into_bytes()),
                ("ppt/slides/z.xml", last.into_bytes()),
            ],
        );
        for extension in ["pptx", "pptm", "ppsx", "ppsm"] {
            let document = super::super::extract(&bytes, extension).unwrap();
            assert_eq!(
                document.markdown,
                "<!-- Slide number: 1 -->\n# Slide title\nSecond paragraph\n\n<!-- Slide number: 2 -->\nUntitled paragraph\n\n<!-- Slide number: 3 -->"
            );
            assert!(document.warnings.is_empty(), "{:?}", document.warnings);
        }
    }

    #[test]
    fn inherited_placeholder_geometry_and_group_children_determine_order() {
        let title = r#"<p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Title</a:t></a:r></a:p></p:txBody></p:sp>"#;
        let group = format!(
            r#"<p:grpSp><p:grpSpPr><a:xfrm><a:off x="1" y="20"/></a:xfrm></p:grpSpPr>{}{}</p:grpSp>"#,
            shape("Group second", "", 2),
            shape("Group first", "", 1)
        );
        let layout = format!(
            r#"<p:sldLayout xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree>{}</p:spTree></p:cSld></p:sldLayout>"#,
            shape("Layout text is not slide content", "title", 30)
        );
        let bytes = package(
            &[("rS", "slides/one.xml")],
            vec![
                (
                    "ppt/slides/one.xml",
                    slide(&format!("{}{group}{title}", shape("Bottom", "", 100))).into_bytes(),
                ),
                (
                    "ppt/slides/_rels/one.xml.rels",
                    relationships(&[("rL", "slideLayout", "../slideLayouts/one.xml")]).into_bytes(),
                ),
                ("ppt/slideLayouts/one.xml", layout.into_bytes()),
            ],
        );
        let document = extract_presentation(&bytes).unwrap();
        assert_eq!(
            document.markdown,
            "<!-- Slide number: 1 -->\nGroup first\nGroup second\n# Title\nBottom"
        );
    }

    #[test]
    fn first_top_level_zero_index_placeholder_is_the_only_slide_title() {
        let group = format!(
            "<p:grpSp>{}</p:grpSp>",
            shape("Grouped placeholder", "title", 1)
        );
        let body = shape("Body-based title", "body", 10)
            .replace("<p:ph type=\"body\"/>", "<p:ph type=\"body\" idx=\"00\"/>")
            .replace(
                "</a:p>",
                "</a:p><a:p><a:r><a:t>Continuation</a:t></a:r></a:p>",
            );
        let later = shape("Later zero-index placeholder", "title", 20);
        let nonzero = shape("Nonzero title-shaped placeholder", "title", 30)
            .replace("<p:ph type=\"title\"/>", "<p:ph type=\"title\" idx=\"3\"/>");
        let bytes = package(
            &[("rS", "slides/one.xml")],
            vec![(
                "ppt/slides/one.xml",
                slide(&format!(
                    "{group}{body}{later}{nonzero}{}",
                    shape("Ordinary body", "", 40)
                ))
                .into_bytes(),
            )],
        );
        let document = extract_presentation(&bytes).unwrap();
        assert!(
            document
                .markdown
                .contains("# Body-based title\nContinuation")
        );
        assert_eq!(
            document
                .markdown
                .lines()
                .filter(|line| line.starts_with("# "))
                .count(),
            1
        );
        assert!(document.markdown.contains("Grouped placeholder"));
        assert!(document.markdown.contains("Later zero-index placeholder"));
        assert!(
            document
                .markdown
                .contains("Nonzero title-shaped placeholder")
        );
        assert!(document.markdown.ends_with("Ordinary body"));
    }

    #[test]
    fn oversized_inheritance_is_parsed_without_unbounded_cache_retention() {
        let large = format!("<layout>{}</layout>", " ".repeat(MAX_INHERITED_XML));
        let bytes = package(
            &[("rS", "slides/one.xml")],
            vec![
                ("large.xml", large.into_bytes()),
                ("small.xml", b"<layout/>".to_vec()),
            ],
        );
        let mut reader = Reader {
            package: Package::new(&bytes).unwrap(),
            document: Document::default(),
            asset_parts: HashMap::new(),
            inherited: HashMap::new(),
            inherited_bytes: 0,
            inherited_nodes: 0,
        };
        let large = reader.inherited_tree("large.xml").unwrap();
        assert_eq!(large.name, "layout");
        assert!(reader.inherited.is_empty());
        let small = reader.inherited_tree("small.xml").unwrap();
        let remaining = reader.package.remaining;
        assert!(Rc::ptr_eq(
            &small,
            &reader.inherited_tree("small.xml").unwrap()
        ));
        assert_eq!(reader.package.remaining, remaining);
        assert!(reader.inherited_bytes <= MAX_INHERITED_XML);
        assert!(reader.inherited_nodes <= MAX_INHERITED_NODES);
    }

    #[test]
    fn master_body_geometry_applies_to_all_reference_placeholder_families() {
        for kind in [
            "body", "chart", "clipArt", "dgm", "media", "obj", "pic", "subTitle", "tbl",
            "ctrTitle", "title", "dt", "ftr", "sldNum",
        ] {
            let mut inherited = shape("Inherited geometry", kind, 1);
            let start = inherited.find("<p:spPr>").unwrap();
            let end = inherited.find("</p:spPr>").unwrap() + "</p:spPr>".len();
            inherited.replace_range(start..end, "<p:spPr/>");
            inherited = inherited.replace(
                &format!("<p:ph type=\"{kind}\"/>"),
                &format!("<p:ph type=\"{kind}\" idx=\"1\"/>"),
            );
            let layout = format!(
                r#"<p:sldLayout xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree>{inherited}</p:spTree></p:cSld></p:sldLayout>"#
            );
            let master = format!(
                r#"<p:sldMaster xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree>{}</p:spTree></p:cSld></p:sldMaster>"#,
                shape(
                    "Master content is not slide content",
                    placeholder_family(kind),
                    30
                )
            );
            let bytes = package(
                &[("rS", "slides/one.xml")],
                vec![
                    (
                        "ppt/slides/one.xml",
                        slide(&format!(
                            "{}{inherited}",
                            shape("Explicit geometry", "", 20)
                        ))
                        .into_bytes(),
                    ),
                    (
                        "ppt/slides/_rels/one.xml.rels",
                        relationships(&[("rL", "slideLayout", "../slideLayouts/one.xml")])
                            .into_bytes(),
                    ),
                    ("ppt/slideLayouts/one.xml", layout.into_bytes()),
                    (
                        "ppt/slideLayouts/_rels/one.xml.rels",
                        relationships(&[("rM", "slideMaster", "../slideMasters/one.xml")])
                            .into_bytes(),
                    ),
                    ("ppt/slideMasters/one.xml", master.into_bytes()),
                ],
            );
            let document = extract_presentation(&bytes).unwrap();
            assert_eq!(
                document.markdown,
                "<!-- Slide number: 1 -->\nExplicit geometry\nInherited geometry",
                "{kind}"
            );
            assert!(
                document.warnings.is_empty(),
                "{kind}: {:?}",
                document.warnings
            );
        }
    }

    #[test]
    fn tables_assets_and_notes_remain_on_their_slide() {
        let table = r#"<p:graphicFrame><p:xfrm><a:off x="1" y="10"/></p:xfrm><a:graphic><a:graphicData><a:tbl><a:tr><a:tc><a:txBody><a:p><a:r><a:t>A | B</a:t></a:r></a:p></a:txBody></a:tc></a:tr><a:tr><a:tc><a:txBody><a:p><a:r><a:t>Data</a:t></a:r></a:p></a:txBody></a:tc></a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>"#;
        let picture = r#"<p:pic><p:nvPicPr><p:cNvPr id="5" descr="Logo&#10;&#10;description"/></p:nvPicPr><p:spPr><a:xfrm><a:off x="1" y="20"/></a:xfrm></p:spPr><p:blipFill><a:blip r:embed="rImage"/></p:blipFill></p:pic>"#;
        let notes = format!(
            r#"<p:notes xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree>{}{}</p:spTree></p:cSld></p:notes>"#,
            shape("Speaker notes", "body", 1),
            shape("Unwanted note date", "dt", 2)
        );
        let bytes = package(
            &[("rS", "slides/one.xml")],
            vec![
                (
                    "ppt/slides/one.xml",
                    slide(&format!("{table}{picture}{picture}")).into_bytes(),
                ),
                (
                    "ppt/slides/_rels/one.xml.rels",
                    relationships(&[
                        ("rImage", "image", "../media/logo.png"),
                        ("rNotes", "notesSlide", "../notesSlides/one.xml"),
                    ])
                    .into_bytes(),
                ),
                ("ppt/media/logo.png", vec![0, 1, 255]),
                ("ppt/media/unused.png", vec![99]),
                ("ppt/notesSlides/one.xml", notes.into_bytes()),
            ],
        );
        let document = extract_presentation(&bytes).unwrap();
        assert!(document.markdown.contains("| A \\| B |\n| --- |\n| Data |"));
        assert_eq!(
            document
                .markdown
                .matches("![Logo description](.markitai/assets/asset-1.png)")
                .count(),
            2
        );
        assert!(document.markdown.ends_with("### Notes:\nSpeaker notes"));
        assert!(!document.markdown.contains("Unwanted note date"));
        assert_eq!(document.assets.len(), 1);
        assert_eq!(document.assets[0].bytes, [0, 1, 255]);
    }

    #[test]
    fn unreadable_slide_keeps_marker_and_unknown_shape_keeps_text() {
        let content = format!(
            "{}<p:futureShape><a:r><a:t>Recoverable unknown text</a:t></a:r></p:futureShape>",
            shape("Still readable", "", 1)
        );
        let bytes = package(
            &[("r1", "slides/one.xml"), ("r2", "slides/missing.xml")],
            vec![
                ("ppt/slides/one.xml", slide(&content).into_bytes()),
                (
                    "ppt/slides/_rels/one.xml.rels",
                    relationships(&[("rNotes", "notesSlide", "../../../outside.xml")]).into_bytes(),
                ),
            ],
        );
        let document = extract_presentation(&bytes).unwrap();
        assert!(document.markdown.contains("Still readable"));
        assert!(document.markdown.contains("Recoverable unknown text"));
        assert!(document.markdown.ends_with("<!-- Slide number: 2 -->"));
        assert!(
            document
                .warnings
                .iter()
                .any(|warning| warning.contains("notes relationship unavailable"))
        );
        assert!(
            document
                .warnings
                .iter()
                .any(|warning| warning.contains("slide 2"))
        );
        assert!(extract_presentation(&package(&[("r", "missing.xml")], vec![])).is_err());
    }

    #[test]
    fn cached_chart_data_is_retained_without_loading_linked_workbooks() {
        let frame = r#"<p:graphicFrame><a:graphic><a:graphicData><c:chart r:id="rC"/></a:graphicData></a:graphic></p:graphicFrame>"#;
        let chart = format!(
            r#"<c:chartSpace xmlns:c="{C}" xmlns:a="{A}" xmlns:r="{R}"><c:chart><c:title><a:p><a:r><a:t>Sales</a:t></a:r></a:p></c:title><c:plotArea><c:barChart><c:ser><c:tx><c:v>North</c:v></c:tx><c:cat><c:strLit><c:pt idx="0"><c:v>Q1</c:v></c:pt></c:strLit></c:cat><c:val><c:numLit><c:pt idx="0"><c:v>3.5</c:v></c:pt></c:numLit></c:val></c:ser></c:barChart></c:plotArea></c:chart><c:externalData r:id="workbook"/></c:chartSpace>"#
        );
        let bytes = package(
            &[("rS", "slides/one.xml")],
            vec![
                ("ppt/slides/one.xml", slide(frame).into_bytes()),
                (
                    "ppt/slides/_rels/one.xml.rels",
                    relationships(&[("rC", "chart", "../charts/one.xml")]).into_bytes(),
                ),
                ("ppt/charts/one.xml", chart.into_bytes()),
            ],
        );
        let document = extract_presentation(&bytes).unwrap();
        assert!(document.markdown.contains("### Chart: Sales"));
        assert!(
            document
                .markdown
                .contains("| Category | North |\n| --- | --- |\n| Q1 | 3.5 |")
        );
        assert!(
            document
                .warnings
                .iter()
                .any(|warning| warning.contains("linked workbook"))
        );
    }

    #[test]
    fn embedded_objects_read_as_their_data_and_others_keep_their_text() {
        // An Excel 2007 worksheet object (a zipped workbook part), an object
        // of no data this reader reads, and a linked object.
        let frame = |y: i64, object: &str| {
            format!(
                r#"<p:graphicFrame><p:xfrm><a:off x="1" y="{y}"/></p:xfrm><a:graphic><a:graphicData>{object}<a:p><a:r><a:t>Preview {y}</a:t></a:r></a:p></a:graphicData></a:graphic></p:graphicFrame>"#
            )
        };
        let shapes = [
            frame(
                1,
                r#"<p:oleObj progId="Excel.Sheet.12" r:id="rW"><p:embed/></p:oleObj>"#,
            ),
            frame(
                2,
                r#"<p:oleObj progId="Equation.3" r:id="rE"><p:embed/></p:oleObj>"#,
            ),
            frame(
                3,
                r#"<p:oleObj progId="Excel.Sheet.8" r:id="rL"><p:link/></p:oleObj>"#,
            ),
        ]
        .concat();
        const SML: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
        let workbook = [
            (
                "xl/workbook.xml",
                format!(
                    r#"<workbook xmlns="{SML}" xmlns:r="{R}"><sheets><sheet name="A" sheetId="1" r:id="rId1"/></sheets></workbook>"#
                ),
            ),
            (
                "xl/_rels/workbook.xml.rels",
                relationships(&[("rId1", "worksheet", "worksheets/sheet1.xml")]),
            ),
            (
                "xl/worksheets/sheet1.xml",
                format!(
                    r#"<worksheet xmlns="{SML}"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Region</t></is></c><c r="B1" t="inlineStr"><is><t>Q1</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>North</t></is></c><c r="B2"><v>10</v></c></row></sheetData></worksheet>"#
                ),
            ),
        ];
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, body) in &workbook {
            writer
                .start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(body.as_bytes()).unwrap();
        }
        let xlsx = writer.finish().unwrap().into_inner();
        let rels = relationships(&[
            ("rW", "package", "../embeddings/sheet.xlsx"),
            ("rE", "oleObject", "../embeddings/oleObject1.bin"),
        ])
        .replace(
            "</Relationships>",
            &format!(
                r#"<Relationship Id="rL" Type="{R}/oleObject" Target="file:///C:/data.xls" TargetMode="External"/></Relationships>"#
            ),
        );
        let bytes = package(
            &[("rS", "slides/one.xml")],
            vec![
                ("ppt/slides/one.xml", slide(&shapes).into_bytes()),
                ("ppt/slides/_rels/one.xml.rels", rels.into_bytes()),
                ("ppt/embeddings/sheet.xlsx", xlsx),
                ("ppt/embeddings/oleObject1.bin", b"Equation Native".to_vec()),
            ],
        );
        let document = extract_presentation(&bytes).unwrap();
        assert!(
            document
                .markdown
                .contains("| Region | Q1 |\n| --- | --- |\n| North | 10 |"),
            "{}",
            document.markdown
        );
        assert!(!document.markdown.contains("Preview 1"));
        assert!(document.markdown.contains("Preview 2"));
        assert!(document.markdown.contains("Preview 3"));
        let warned = |program: &str| {
            document.warnings.iter().any(|warning| {
                warning.contains(&format!("embedded object ({program}) holds no data"))
            })
        };
        assert!(warned("Equation.3") && warned("Excel.Sheet.8") && !warned("Excel.Sheet.12"));
    }

    #[test]
    fn cached_chart_numbers_read_in_their_format_codes() {
        // Date categories are cached as serial numbers and a share as a
        // fraction; the chart shows a date and a percentage. Text points and
        // General numbers are unchanged, and a point's own code wins.
        let frame = r#"<p:graphicFrame><a:graphic><a:graphicData><c:chart r:id="rC"/></a:graphicData></a:graphic></p:graphicFrame>"#;
        let series = |name: &str, cat: &str, val: &str| {
            format!(
                r#"<c:ser><c:tx><c:v>{name}</c:v></c:tx><c:cat>{cat}</c:cat><c:val>{val}</c:val></c:ser>"#
            )
        };
        let dates = r#"<c:numRef><c:numCache><c:formatCode>m/d/yyyy</c:formatCode><c:pt idx="0"><c:v>45658.0</c:v></c:pt><c:pt idx="1"><c:v>45689</c:v></c:pt></c:numCache></c:numRef>"#;
        let users = r#"<c:numRef><c:numCache><c:formatCode>General</c:formatCode><c:pt idx="0"><c:v>100.0</c:v></c:pt><c:pt idx="1"><c:v>0.30000000000000004</c:v></c:pt></c:numCache></c:numRef>"#;
        let names = r#"<c:strRef><c:strCache><c:pt idx="0"><c:v>0.5</c:v></c:pt><c:pt idx="1"><c:v>Pears</c:v></c:pt></c:strCache></c:strRef>"#;
        let share = r#"<c:numLit><c:formatCode>0%</c:formatCode><c:pt idx="0"><c:v>0.65</c:v></c:pt><c:pt idx="1" formatCode="0.0%"><c:v>0.35</c:v></c:pt></c:numLit>"#;
        let chart = |date1904: &str, body: String| {
            format!(
                r#"<c:chartSpace xmlns:c="{C}" xmlns:a="{A}" xmlns:r="{R}">{date1904}<c:chart><c:plotArea><c:lineChart>{body}</c:lineChart></c:plotArea></c:chart></c:chartSpace>"#
            )
        };
        let convert = |chart: String| {
            let bytes = package(
                &[("rS", "slides/one.xml")],
                vec![
                    ("ppt/slides/one.xml", slide(frame).into_bytes()),
                    (
                        "ppt/slides/_rels/one.xml.rels",
                        relationships(&[("rC", "chart", "../charts/one.xml")]).into_bytes(),
                    ),
                    ("ppt/charts/one.xml", chart.into_bytes()),
                ],
            );
            extract_presentation(&bytes).unwrap().markdown
        };
        let markdown = convert(chart(
            r#"<c:date1904 val="0"/>"#,
            series("Users", dates, users) + &series("Share", names, share),
        ));
        assert!(
            markdown.contains(
                "| Category | Users | Share |\n| --- | --- | --- |\n| 2025-01-01 | 100 | 65% |\n| 2025-02-01 | 0.3 | 35.0% |"
            ),
            "{markdown}"
        );
        // The 1904 date system counts from 1904-01-01, 1,462 days later.
        let markdown = convert(chart(r#"<c:date1904/>"#, series("Users", dates, users)));
        assert!(markdown.contains("| 2029-01-02 | 100 |"), "{markdown}");
    }

    /// A text shape whose paragraphs are `(level, pPr children, runs)`.
    fn paragraphs(
        placeholder: &str,
        list_style: &str,
        paragraphs: &[(u8, &str, &str)],
        y: i64,
    ) -> String {
        let paragraphs: String = paragraphs
            .iter()
            .map(|(level, properties, runs)| {
                format!(r#"<a:p><a:pPr lvl="{level}">{properties}</a:pPr>{runs}</a:p>"#)
            })
            .collect();
        format!(
            r#"<p:sp><p:nvSpPr><p:cNvPr id="2" name="text"/><p:nvPr>{placeholder}</p:nvPr></p:nvSpPr><p:spPr><a:xfrm><a:off x="1" y="{y}"/></a:xfrm></p:spPr><p:txBody><a:lstStyle>{list_style}</a:lstStyle>{paragraphs}</p:txBody></p:sp>"#
        )
    }

    fn run(text: &str) -> String {
        format!("<a:r><a:t>{text}</a:t></a:r>")
    }

    fn link_run(text: &str, id: &str) -> String {
        format!(r#"<a:r><a:rPr><a:hlinkClick r:id="{id}"/></a:rPr><a:t>{text}</a:t></a:r>"#)
    }

    /// One slide on a layout and a master carrying the given text styles.
    fn deck(
        slide_shapes: &str,
        layout_shapes: &str,
        master_styles: &str,
        slide_rels: &str,
    ) -> Vec<u8> {
        let layout = format!(
            r#"<p:sldLayout xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree>{layout_shapes}</p:spTree></p:cSld></p:sldLayout>"#
        );
        let master = format!(
            r#"<p:sldMaster xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree/></p:cSld><p:txStyles>{master_styles}</p:txStyles></p:sldMaster>"#
        );
        let rels = relationships(&[("rL", "slideLayout", "../slideLayouts/one.xml")])
            .replace("</Relationships>", &format!("{slide_rels}</Relationships>"));
        package(
            &[("rS", "slides/one.xml")],
            vec![
                ("ppt/slides/one.xml", slide(slide_shapes).into_bytes()),
                ("ppt/slides/_rels/one.xml.rels", rels.into_bytes()),
                ("ppt/slideLayouts/one.xml", layout.into_bytes()),
                (
                    "ppt/slideLayouts/_rels/one.xml.rels",
                    relationships(&[("rM", "slideMaster", "../slideMasters/one.xml")]).into_bytes(),
                ),
                ("ppt/slideMasters/one.xml", master.into_bytes()),
            ],
        )
    }

    fn external(id: &str, target: &str) -> String {
        format!(
            r#"<Relationship Id="{id}" Type="{R}/hyperlink" Target="{target}" TargetMode="External"/>"#
        )
    }

    const BULLETS: &str = r#"<p:bodyStyle><a:lvl1pPr><a:buChar char="•"/></a:lvl1pPr><a:lvl2pPr><a:buChar char="–"/></a:lvl2pPr><a:lvl3pPr><a:buChar char="»"/></a:lvl3pPr></p:bodyStyle><p:otherStyle><a:lvl1pPr/></p:otherStyle>"#;
    const BODY: &str = r#"<p:ph type="body" idx="1"/>"#;

    #[test]
    fn bullets_become_a_nested_markdown_list_with_the_text_around_it_set_apart() {
        let bytes = deck(
            &paragraphs(
                BODY,
                "",
                &[
                    (0, "<a:buNone/>", &run("Intro")),
                    (0, "", &run("First")),
                    (1, "", &run("Nested")),
                    (2, "", &run("Deep")),
                    (1, "", &run("Back one")),
                    (0, "", &run("Again")),
                    (0, "<a:buNone/>", &run("Plain closing")),
                    (2, "", &run("Deep with no parent")),
                    (
                        0,
                        "",
                        &format!("<a:r><a:t>Two</a:t></a:r><a:br/>{}", run("lines")),
                    ),
                    (0, "", ""),
                    (0, "", &run("After a blank")),
                ],
                10,
            ),
            "",
            BULLETS,
            "",
        );
        let document = extract_presentation(&bytes).unwrap();
        assert_eq!(
            document.markdown,
            "<!-- Slide number: 1 -->\nIntro\n\n* First\n  * Nested\n    * Deep\n  * Back one\n* Again\n\n\
             Plain closing\n\n* Deep with no parent\n* Two\n  lines\n\n* After a blank"
        );
        assert!(document.warnings.is_empty(), "{:?}", document.warnings);
    }

    #[test]
    fn auto_numbered_paragraphs_count_per_level_and_honor_their_start() {
        let styles = r#"<p:bodyStyle><a:lvl1pPr><a:buAutoNum type="arabicPeriod"/></a:lvl1pPr><a:lvl2pPr><a:buAutoNum type="arabicPeriod" startAt="3"/></a:lvl2pPr></p:bodyStyle>"#;
        let bytes = deck(
            &paragraphs(
                BODY,
                "",
                &[
                    (0, "", &run("One")),
                    (1, "", &run("Sub a")),
                    (1, "", &run("Sub b")),
                    (0, "", &run("Two")),
                    (1, "", &run("Sub again")),
                    (0, r#"<a:buChar char="-"/>"#, &run("A bullet")),
                    (
                        0,
                        r#"<a:buAutoNum type="alphaLcPeriod" startAt="9"/>"#,
                        &run("Ninth"),
                    ),
                ],
                10,
            ),
            "",
            styles,
            "",
        );
        assert_eq!(
            extract_presentation(&bytes).unwrap().markdown,
            "<!-- Slide number: 1 -->\n1. One\n   3. Sub a\n   4. Sub b\n2. Two\n   3. Sub again\n* A bullet\n9. Ninth"
        );
    }

    #[test]
    fn a_bullet_is_decided_by_the_closest_layer_that_says_something() {
        let layout = r#"<p:sp><p:nvSpPr><p:nvPr><p:ph type="subTitle" idx="1"/></p:nvPr></p:nvSpPr><p:txBody><a:lstStyle><a:lvl1pPr><a:buNone/></a:lvl1pPr></a:lstStyle><a:p><a:r><a:t>Layout text is not slide content</a:t></a:r></a:p></p:txBody></p:sp><p:sp><p:nvSpPr><p:nvPr><p:ph type="body" idx="2"/></p:nvPr></p:nvSpPr><p:txBody><a:lstStyle><a:lvl1pPr><a:buAutoNum type="arabicPeriod"/></a:lvl1pPr></a:lstStyle></p:txBody></p:sp>"#;
        let shapes = [
            // The layout turns bullets off for this subtitle.
            paragraphs(
                r#"<p:ph type="subTitle" idx="1"/>"#,
                "",
                &[(0, "", &run("Subtitle"))],
                10,
            ),
            // The layout numbers this body.
            paragraphs(
                r#"<p:ph type="body" idx="2"/>"#,
                "",
                &[(0, "", &run("Numbered by layout"))],
                20,
            ),
            // The master's body style bullets everything else; the shape's own
            // list style, the closest layer, switches it off.
            paragraphs(
                r#"<p:ph type="body" idx="3"/>"#,
                "<a:lvl1pPr><a:buNone/></a:lvl1pPr>",
                &[(0, "", &run("Shape opts out"))],
                30,
            ),
            paragraphs(
                r#"<p:ph type="obj" idx="4"/>"#,
                "",
                &[(0, "", &run("Master bullet"))],
                40,
            ),
            // A text box has no placeholder style: none, unless it sets one.
            paragraphs("", "", &[(0, "", &run("Text box"))], 50),
            paragraphs(
                "",
                r#"<a:lvl1pPr><a:buChar char="-"/></a:lvl1pPr>"#,
                &[(0, "", &run("Text box with its own bullet"))],
                60,
            ),
        ]
        .concat();
        let bytes = deck(&shapes, layout, BULLETS, "");
        assert_eq!(
            extract_presentation(&bytes).unwrap().markdown,
            "<!-- Slide number: 1 -->\nSubtitle\n\n1. Numbered by layout\n\nShape opts out\n\n* Master bullet\n\n\
             Text box\n\n* Text box with its own bullet"
        );
    }

    #[test]
    fn hyperlinks_with_web_and_mail_targets_stay_links() {
        let rels = [
            external("rA", "https://example.com/a(b)"),
            external("rM", "mailto:team@example.com"),
            external("rJ", "javascript:alert(1)"),
            external("rF", "file:///etc/passwd"),
        ]
        .concat()
            + r#"<Relationship Id="rI" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slide2.xml"/>"#;
        let split = format!(
            "{}{}{}",
            link_run("Deep ", "rA"),
            link_run("link", "rA"),
            run(" and more")
        );
        let table = format!(
            r#"<p:graphicFrame><p:xfrm><a:off x="1" y="40"/></p:xfrm><a:graphic><a:graphicData><a:tbl><a:tr><a:tc><a:txBody><a:p>{}</a:p></a:txBody></a:tc></a:tr><a:tr><a:tc><a:txBody><a:p><a:r><a:t>Data</a:t></a:r></a:p></a:txBody></a:tc></a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>"#,
            link_run("in a cell", "rA")
        );
        let shapes = [
            paragraphs(
                r#"<p:ph type="title"/>"#,
                "",
                &[(0, "", &link_run("Site", "rA"))],
                1,
            ),
            paragraphs(
                "",
                "",
                &[
                    (0, "", &split),
                    (0, "", &link_run("write us", "rM")),
                    (
                        0,
                        "",
                        &format!("{}{}", run("plain "), link_run(" padded [x] ", "rA")),
                    ),
                    (0, "", &link_run("script", "rJ")),
                    (0, "", &link_run("local file", "rF")),
                    (0, "", &link_run("slide jump", "rI")),
                    (0, "", &link_run("unknown", "rMissing")),
                    (0, "", &link_run("   ", "rA")),
                    (
                        0,
                        "",
                        &format!("{}<a:br/>{}", link_run("one", "rA"), link_run("two", "rA")),
                    ),
                ],
                20,
            ),
            table,
        ]
        .concat();
        let bytes = deck(&shapes, "", BULLETS, &rels);
        let document = extract_presentation(&bytes).unwrap();
        assert_eq!(
            document.markdown,
            "<!-- Slide number: 1 -->\n# [Site](https://example.com/a%28b%29)\n\
             [Deep link](https://example.com/a%28b%29) and more\n\
             [write us](mailto:team@example.com)\n\
             plain  [padded \\[x\\]](https://example.com/a%28b%29)\n\
             script\nlocal file\nslide jump\nunknown\n\n\
             [one](https://example.com/a%28b%29)\n[two](https://example.com/a%28b%29)\n\n\
             | [in a cell](https://example.com/a%28b%29) |\n| --- |\n| Data |"
        );
    }

    #[test]
    fn malformed_xml_traversal_and_resource_bounds_are_explicit() {
        assert!(xml(b"<!DOCTYPE a [<!ENTITY x SYSTEM 'file:///secret'>]><a/>").is_err());
        assert!(xml(b"<a><b></a>").is_err());
        assert!(xml(format!("{}{}", "<a>".repeat(130), "</a>".repeat(130)).as_bytes()).is_err());
        for target in [
            "../../../secret",
            "%2e%2e/%2e%2e/%2e%2e/secret",
            "file:///secret",
            "//server/path",
            "..\\secret",
            "a%00b",
        ] {
            assert!(resolve("ppt/slides/one.xml", target).is_err(), "{target}");
        }
        assert_eq!(
            resolve("ppt/slides/one.xml", "../media/a%20b.png").unwrap(),
            "ppt/media/a b.png"
        );
        let bytes = package(&[("r", "slides/one.xml")], vec![]);
        let mut package = Package::new(&bytes).unwrap();
        assert!(package.read("ppt/presentation.xml", 2).is_err());
        assert_eq!(package.remaining, MAX_TOTAL);
        let mut package = Package::new(&bytes).unwrap();
        package.remaining = 2;
        assert!(package.read("ppt/presentation.xml", MAX_PART).is_err());
    }

    #[test]
    fn smartart_hidden_slides_and_review_comments_are_kept() {
        let diagram = r#"<p:graphicFrame><p:xfrm><a:off x="1" y="50"/></p:xfrm><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/diagram"><dgm:relIds xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram" r:dm="rD" r:lo="rLo"/></a:graphicData></a:graphic></p:graphicFrame>"#;
        let data = r#"<dgm:dataModel xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><dgm:ptLst><dgm:pt modelId="0" type="doc"><dgm:t><a:p/></dgm:t></dgm:pt><dgm:pt modelId="1"><dgm:t><a:p><a:r><a:t>Discover</a:t></a:r></a:p></dgm:t></dgm:pt><dgm:pt modelId="2"><dgm:t><a:p><a:r><a:t>Deliver</a:t></a:r></a:p></dgm:t></dgm:pt></dgm:ptLst></dgm:dataModel>"#;
        let hidden =
            slide(&shape("Backup slide", "", 10)).replace("<p:sld ", r#"<p:sld show="0" "#);
        let legacy = format!(
            r#"<p:cmLst xmlns:p="{P}"><p:cm authorId="0" idx="1"><p:text>Please verify   this number.</p:text></p:cm></p:cmLst>"#
        );
        let modern = format!(
            r#"<p188:cmLst xmlns:p188="http://schemas.microsoft.com/office/powerpoint/2018/8/main" xmlns:a="{A}"><p188:cm id="{{C1}}" authorId="{{A1}}"><p188:txBody><a:bodyPr/><a:p><a:r><a:t>Is *this* final?</a:t></a:r></a:p></p188:txBody><p188:replyLst><p188:reply id="{{R1}}" authorId="{{A2}}"><p188:txBody><a:p><a:r><a:t>Yes.</a:t></a:r></a:p></p188:txBody></p188:reply></p188:replyLst></p188:cm></p188:cmLst>"#
        );
        let parts: Vec<(&str, Vec<u8>)> = vec![
            (
                "ppt/presentation.xml",
                format!(r#"<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst><p:sldId id="256" r:id="r1"/><p:sldId id="257" r:id="r2"/></p:sldIdLst></p:presentation>"#).into_bytes(),
            ),
            (
                "ppt/_rels/presentation.xml.rels",
                relationships(&[
                    ("r1", "slide", "slides/one.xml"),
                    ("r2", "slide", "slides/two.xml"),
                    ("rA", "commentAuthors", "commentAuthors.xml"),
                ])
                .replace(
                    "</Relationships>",
                    r#"<Relationship Id="rB" Type="http://schemas.microsoft.com/office/2018/10/relationships/authors" Target="authors.xml"/></Relationships>"#,
                )
                .into_bytes(),
            ),
            (
                "ppt/commentAuthors.xml",
                format!(r#"<p:cmAuthorLst xmlns:p="{P}"><p:cmAuthor id="0" name="Reviewer One"/></p:cmAuthorLst>"#).into_bytes(),
            ),
            (
                "ppt/authors.xml",
                r#"<p188:authorLst xmlns:p188="http://schemas.microsoft.com/office/powerpoint/2018/8/main"><p188:author id="{A1}" name="Ann"/><p188:author id="{A2}" name="Bo"/></p188:authorLst>"#.as_bytes().to_vec(),
            ),
            (
                "ppt/slides/one.xml",
                slide(&format!("{}{diagram}", shape("Process", "title", 10))).into_bytes(),
            ),
            (
                "ppt/slides/_rels/one.xml.rels",
                relationships(&[
                    ("rD", "diagramData", "../diagrams/data1.xml"),
                    ("rC", "comments", "../comments/comment1.xml"),
                ])
                .into_bytes(),
            ),
            ("ppt/diagrams/data1.xml", data.as_bytes().to_vec()),
            ("ppt/comments/comment1.xml", legacy.into_bytes()),
            ("ppt/slides/two.xml", hidden.into_bytes()),
            (
                "ppt/slides/_rels/two.xml.rels",
                relationships(&[])
                    .replace(
                        "</Relationships>",
                        r#"<Relationship Id="rM" Type="http://schemas.microsoft.com/office/2018/10/relationships/comments" Target="../comments/modernComment_1.xml"/></Relationships>"#,
                    )
                    .into_bytes(),
            ),
            ("ppt/comments/modernComment_1.xml", modern.into_bytes()),
        ];
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, bytes) in parts {
            writer
                .start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(&bytes).unwrap();
        }
        let document = extract_presentation(&writer.finish().unwrap().into_inner()).unwrap();
        assert_eq!(
            document.markdown,
            "<!-- Slide number: 1 -->\n# Process\n\n* Discover\n* Deliver\n\n\
             ### Comments:\n* Reviewer One: Please verify this number.\n\n\
             <!-- Slide number: 2 -->\n<!-- Hidden slide -->\nBackup slide\n\n\
             ### Comments:\n* Ann: Is \\*this\\* final?\n* Bo: Yes."
        );
        assert!(document.warnings.is_empty(), "{:?}", document.warnings);
    }
}
