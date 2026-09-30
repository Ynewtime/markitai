//! Bounded OOXML presentation reader. Slide identity comes from package relationships.

use crate::{Asset, Document, Error, Result};
use quick_xml::{events::Event, name::ResolveResult};
use std::collections::{BTreeMap, HashMap};
use std::io::{Cursor, Read};
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
                    ResolveResult::Bound(uri) => namespace(uri.as_ref()),
                    _ => Ns::Other,
                };
                let mut node = Node {
                    ns,
                    name: String::from_utf8_lossy(event.local_name().as_ref()).into_owned(),
                    ..Node::default()
                };
                for attribute in event.attributes() {
                    let attribute = attribute.map_err(error)?;
                    let name =
                        String::from_utf8_lossy(attribute.key.local_name().as_ref()).into_owned();
                    let value = attribute
                        .decoded_and_normalized_value(
                            quick_xml::XmlVersion::Implicit1_0,
                            reader.decoder(),
                        )
                        .map_err(error)?
                        .into_owned();
                    match reader.resolver().resolve_attribute(attribute.key).0 {
                        ResolveResult::Unbound => {
                            node.attributes.insert(name, value);
                        }
                        ResolveResult::Bound(uri)
                            if matches!(uri.as_ref(),
                            b"http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                            | b"http://purl.oclc.org/ooxml/officeDocument/relationships") =>
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
                stack.last_mut().unwrap().text.push_str(
                    &value
                        .xml_content(quick_xml::XmlVersion::Implicit1_0)
                        .map_err(error)?,
                );
            }
            Event::CData(value) => {
                stack
                    .last_mut()
                    .unwrap()
                    .text
                    .push_str(&value.decode().map_err(error)?);
            }
            Event::GeneralRef(value) => {
                let entity = format!("&{};", value.decode().map_err(error)?);
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
    archive: zip::ZipArchive<Cursor<&'a [u8]>>,
    remaining: u64,
}

impl<'a> Package<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self> {
        let archive = zip::ZipArchive::new(Cursor::new(bytes)).map_err(error)?;
        if archive.len() > 16_384 {
            return Err(error("package has more than 16,384 entries"));
        }
        Ok(Self {
            archive,
            remaining: MAX_TOTAL,
        })
    }

    fn read(&mut self, path: &str, limit: u64) -> Result<Option<Vec<u8>>> {
        let file = match self.archive.by_name(path) {
            Ok(file) => file,
            Err(zip::result::ZipError::FileNotFound) => return Ok(None),
            Err(e) => return Err(error(e)),
        };
        if file.size() > limit || file.size() > self.remaining {
            return Err(error(format!(
                "part {path} exceeds its decompression budget"
            )));
        }
        let mut bytes = Vec::new();
        file.take(limit.min(self.remaining) + 1)
            .read_to_end(&mut bytes)
            .map_err(error)?;
        if bytes.len() as u64 > limit.min(self.remaining) {
            return Err(error(format!("part {path} exceeds its read budget")));
        }
        self.remaining -= bytes.len() as u64;
        Ok(Some(bytes))
    }

    fn tree(&mut self, path: &str) -> Result<Node> {
        xml(&self
            .read(path, MAX_PART)?
            .ok_or_else(|| error(format!("missing part {path}")))?)
    }

    fn relationships(&mut self, part: &str) -> Result<BTreeMap<String, Relationship>> {
        let path = if part.is_empty() {
            "_rels/.rels".into()
        } else {
            let (directory, filename) = part.rsplit_once('/').unwrap_or(("", part));
            format!("{directory}/_rels/{filename}.rels")
                .trim_start_matches('/')
                .to_owned()
        };
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
            let relationship = Relationship {
                kind: node.attr("Type").unwrap_or("").to_owned(),
                target: node.attr("Target").unwrap_or("").to_owned(),
                external: node.attr("TargetMode") == Some("External"),
            };
            if relationships.insert(id.to_owned(), relationship).is_some() {
                return Err(error(format!("duplicate relationship {id} in {path}")));
            }
        }
        Ok(relationships)
    }
}

#[derive(Clone)]
struct Relationship {
    kind: String,
    target: String,
    external: bool,
}

fn resolve(base: &str, target: &str) -> Result<String> {
    if target.is_empty() || target.starts_with("//") || target.contains(['\\', '\0', ':', '?', '#'])
    {
        return Err(error("invalid internal relationship target"));
    }
    let mut decoded = Vec::new();
    let bytes = target.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let pair = bytes
                .get(i + 1..i + 3)
                .ok_or_else(|| error("invalid escaped part name"))?;
            let pair = std::str::from_utf8(pair).map_err(error)?;
            decoded.push(u8::from_str_radix(pair, 16).map_err(error)?);
            i += 3;
        } else {
            decoded.push(bytes[i]);
            i += 1;
        }
    }
    let decoded = String::from_utf8(decoded).map_err(error)?;
    if decoded.contains(['\\', '\0', ':', '?', '#']) {
        return Err(error("unsafe escaped part name"));
    }
    let mut components = if decoded.starts_with('/') {
        Vec::new()
    } else {
        base.rsplit_once('/')
            .map_or(Vec::new(), |(parent, _)| parent.split('/').collect())
    };
    for component in decoded.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                if components.pop().is_none() {
                    return Err(error("relationship escapes package root"));
                }
            }
            value => components.push(value),
        }
    }
    if components.is_empty() {
        return Err(error("relationship has no part name"));
    }
    Ok(components.join("/"))
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

    fn table(&mut self, table: &Node, slide: usize) -> String {
        let rows = table.children.iter().filter(|node| node.is(Ns::Drawing, "tr")).map(|row| {
            row.children.iter().filter(|node| node.is(Ns::Drawing, "tc")).map(|cell| {
                if cell.attr("gridSpan").is_some_and(|v| v != "1") || cell.attr("rowSpan").is_some_and(|v| v != "1") {
                    self.warn(slide, "merged table cells retain their origin text and covered cells; Markdown has no merged-cell geometry");
                }
                super::super::text::cell(&cell.child(Ns::Drawing, "txBody").map(text_body).unwrap_or_default())
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
            )?;
            let categories = cached_points(
                series
                    .child(Ns::Chart, "cat")
                    .or_else(|| series.child(Ns::Chart, "xVal")),
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
        let mut children = parent.children.iter().collect::<Vec<_>>();
        children.sort_by_key(|shape| position(shape, layout, master));
        let mut output = String::new();
        for shape in children {
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
                        let text = text_body(body);
                        if title.is_some_and(|title| std::ptr::eq(title, shape))
                            && !text.trim().is_empty()
                        {
                            output.push_str("# ");
                            output.push_str(text.trim_start());
                        } else {
                            output.push_str(&text);
                        }
                        output.push('\n');
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
                        output.push_str(&self.table(table, slide));
                    } else if shape.descendant(Ns::Chart, "chart").is_some() {
                        match self.chart(shape, part, rels, slide) {
                            Ok(chart) => output.push_str(&chart),
                            Err(e) => {
                                self.warn(slide, e);
                                output.push_str("\n\n[unsupported chart]\n\n");
                            }
                        }
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

fn cached_points(node: Option<&Node>) -> Result<BTreeMap<usize, String>> {
    let mut points = Vec::new();
    if let Some(node) = node {
        node.collect(Ns::Chart, "pt", &mut points);
    }
    let mut result = BTreeMap::new();
    for point in points {
        let index: usize = point.attr("idx").unwrap_or("0").parse().map_err(error)?;
        if index >= 100_000 {
            return Err(error("chart point index exceeds limit"));
        }
        if let Some(value) = point.child(Ns::Chart, "v") {
            result.insert(index, value.text.clone());
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
    use std::io::Write;

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
            r#"<p:sp><p:nvSpPr><p:cNvPr id="1" name="shape"/><p:nvPr>{ph}</p:nvPr></p:nvSpPr><p:spPr><a:xfrm><a:off x="1" y="{y}"/></a:xfrm></p:spPr><p:txBody><a:lstStyle><a:lvl1pPr><a:buChar char="•"/></a:lvl1pPr></a:lstStyle><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp>"#
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
}
