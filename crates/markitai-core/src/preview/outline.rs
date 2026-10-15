//! A PDF's own outline (its bookmarks) as page numbers, for the side-by-side
//! view's outline panel: titles decoded from PDFDocEncoding, UTF-16 or UTF-8,
//! one line each and bounded, nested by level, in the outline's order.
//!
//! The walk is bounded: entries listed, nodes visited, nesting depth and a
//! name tree's size all have limits, and a node seen before (an outline whose
//! `Next` or `First` links loop) ends that branch.
use lopdf::{Dictionary, Document, Object, ObjectId, StringFormat};
use std::collections::{HashMap, HashSet};

/// Entries an outline answer lists at most; `truncated` says more exist.
pub const MAX_OUTLINE_ENTRIES: usize = 2000;
/// Characters a title keeps at most.
pub const MAX_OUTLINE_TITLE: usize = 300;
/// Outline nodes visited at most (entries without a page count too).
const MAX_NODES: usize = 20_000;
/// Nesting levels followed; deeper entries are not listed.
const MAX_DEPTH: usize = 64;
/// Name-tree nodes read at most while resolving named destinations.
const MAX_NAME_NODES: usize = 20_000;

/// One bookmark: its title, the 1-based page it opens and its 1-based level.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutlineEntry {
    pub title: String,
    pub page: usize,
    pub level: usize,
}

/// A PDF's outline; `entries` is empty when it has none.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PdfOutline {
    pub entries: Vec<OutlineEntry>,
    /// More entries exist than [`MAX_OUTLINE_ENTRIES`] (or than the walk's bounds reached).
    pub truncated: bool,
}

/// The outline of a PDF. A document that cannot be read, or that has no
/// outline, has an empty one: the outline is optional and never fails a view.
///
/// An entry whose destination is not a page of this document (a link to
/// another file, a broken reference, a container without a destination)
/// opens the page of the next entry that has one; when none follows, it is
/// left out.
pub fn pdf_outline(bytes: &[u8]) -> PdfOutline {
    match Document::load_mem(bytes) {
        Ok(document) => outline_of(&document),
        Err(_) => PdfOutline::default(),
    }
}

fn outline_of(document: &Document) -> PdfOutline {
    let Ok(catalog) = document.catalog() else {
        return PdfOutline::default();
    };
    let Some(root) = catalog
        .get(b"Outlines")
        .ok()
        .and_then(|value| dictionary(document, value))
    else {
        return PdfOutline::default();
    };
    let pages: HashMap<ObjectId, usize> = document
        .get_pages()
        .into_iter()
        .map(|(number, id)| (id, number as usize))
        .collect();
    let count = pages.len();
    let mut names = Names::new(document, catalog);
    // Depth-first, in order: a node, its children, then its next sibling.
    let mut stack: Vec<(&Object, usize)> = Vec::new();
    if let Ok(first) = root.get(b"First") {
        stack.push((first, 1));
    }
    let mut seen: HashSet<ObjectId> = HashSet::new();
    let mut found: Vec<(String, Option<usize>, usize)> = Vec::new();
    let mut truncated = false;
    let mut visited = 0usize;
    while let Some((link, level)) = stack.pop() {
        if let Ok(id) = link.as_reference()
            && !seen.insert(id)
        {
            continue;
        }
        let Some(node) = dictionary(document, link) else {
            continue;
        };
        visited += 1;
        if visited > MAX_NODES {
            truncated = true;
            break;
        }
        if let Ok(next) = node.get(b"Next") {
            stack.push((next, level));
        }
        if let Ok(first) = node.get(b"First") {
            if level < MAX_DEPTH {
                stack.push((first, level + 1));
            } else {
                truncated = true;
            }
        }
        let page = destination(document, node, &mut names)
            .and_then(|target| page_of(document, target, &pages, count));
        found.push((title(document, node), page, level));
    }
    // Entries without a page take the next entry's; trailing ones are left out.
    let mut next_page = None;
    let mut entries: Vec<OutlineEntry> = Vec::with_capacity(found.len().min(MAX_OUTLINE_ENTRIES));
    for (title, page, level) in found.into_iter().rev() {
        if page.is_some() {
            next_page = page;
        }
        if let Some(page) = page.or(next_page) {
            entries.push(OutlineEntry { title, page, level });
        }
    }
    entries.reverse();
    if entries.len() > MAX_OUTLINE_ENTRIES {
        entries.truncate(MAX_OUTLINE_ENTRIES);
        truncated = true;
    }
    PdfOutline { entries, truncated }
}

fn dictionary<'a>(document: &'a Document, value: &'a Object) -> Option<&'a Dictionary> {
    document
        .dereference(value)
        .ok()
        .and_then(|(_, object)| object.as_dict().ok())
}

fn resolved<'a>(document: &'a Document, value: &'a Object) -> Option<&'a Object> {
    document.dereference(value).ok().map(|(_, object)| object)
}

/// A bookmark's title on one line: control characters and runs of white
/// space as one space, trimmed, at most [`MAX_OUTLINE_TITLE`] characters.
fn title(document: &Document, node: &Dictionary) -> String {
    let raw = node
        .get(b"Title")
        .ok()
        .and_then(|value| resolved(document, value))
        .and_then(|value| value.as_str().ok())
        .map(decode_text)
        .unwrap_or_default();
    let mut text = String::with_capacity(raw.len().min(MAX_OUTLINE_TITLE * 4));
    let mut kept = 0usize;
    let mut space = false;
    for character in raw.chars() {
        if character.is_whitespace() || character.is_control() {
            space = !text.is_empty();
            continue;
        }
        if space {
            if kept + 1 >= MAX_OUTLINE_TITLE {
                break;
            }
            text.push(' ');
            kept += 1;
            space = false;
        }
        if kept >= MAX_OUTLINE_TITLE {
            break;
        }
        text.push(character);
        kept += 1;
    }
    text
}

/// A PDF text string: UTF-16BE (or, written by some tools, UTF-16LE) after
/// its byte-order mark, UTF-8 after its mark, else PDFDocEncoding.
pub(crate) fn decode_text(bytes: &[u8]) -> String {
    let utf16 = |rest: &[u8], big: bool| {
        let units: Vec<u16> = rest
            .chunks(2)
            .map(|pair| {
                let pair = [pair[0], *pair.get(1).unwrap_or(&0)];
                if big {
                    u16::from_be_bytes(pair)
                } else {
                    u16::from_le_bytes(pair)
                }
            })
            .collect();
        String::from_utf16_lossy(&units)
    };
    if let Some(rest) = bytes.strip_prefix(b"\xFE\xFF") {
        utf16(rest, true)
    } else if let Some(rest) = bytes.strip_prefix(b"\xFF\xFE") {
        utf16(rest, false)
    } else if let Some(rest) = bytes.strip_prefix(b"\xEF\xBB\xBF") {
        String::from_utf8_lossy(rest).into_owned()
    } else {
        lopdf::decode_text_string(&Object::String(bytes.to_vec(), StringFormat::Literal))
            .unwrap_or_else(|_| String::from_utf8_lossy(bytes).into_owned())
    }
}

/// Where a bookmark goes: its `Dest`, or the `D` of its `GoTo` action
/// (other actions, `GoToR` to another file among them, go to no page here).
fn destination<'a>(
    document: &'a Document,
    node: &'a Dictionary,
    names: &mut Names<'a>,
) -> Option<&'a Object> {
    let target = match node.get(b"Dest") {
        Ok(dest) => dest,
        Err(_) => {
            let action = node
                .get(b"A")
                .ok()
                .and_then(|value| dictionary(document, value))?;
            let kind = action
                .get(b"S")
                .ok()
                .and_then(|value| resolved(document, value))?;
            if kind.as_name().ok()? != b"GoTo" {
                return None;
            }
            action.get(b"D").ok()?
        }
    };
    explicit(document, resolved(document, target)?, names, 0)
}

/// An explicit destination (an array whose first element is the page), from
/// a name or string through the document's named destinations, or from a
/// dictionary's `D`.
fn explicit<'a>(
    document: &'a Document,
    target: &'a Object,
    names: &mut Names<'a>,
    depth: usize,
) -> Option<&'a Object> {
    if depth > 4 {
        return None;
    }
    match target {
        Object::Array(items) => items.first(),
        Object::Name(name) | Object::String(name, _) => {
            let found = names.lookup(name)?;
            explicit(document, resolved(document, found)?, names, depth + 1)
        }
        Object::Dictionary(dict) => {
            let inner = resolved(document, dict.get(b"D").ok()?)?;
            explicit(document, inner, names, depth + 1)
        }
        _ => None,
    }
}

/// The 1-based number of a destination's page: a page object of this
/// document, or (written by some tools for local links) a 0-based index.
fn page_of(
    document: &Document,
    page: &Object,
    pages: &HashMap<ObjectId, usize>,
    count: usize,
) -> Option<usize> {
    match page {
        Object::Reference(id) => pages.get(id).copied().or_else(|| {
            // A reference to a reference of the page.
            match document.dereference(page).ok()? {
                (Some(last), _) => pages.get(&last).copied(),
                _ => None,
            }
        }),
        Object::Integer(index) => usize::try_from(*index)
            .ok()
            .filter(|index| *index < count)
            .map(|index| index + 1),
        _ => None,
    }
}

/// The document's named destinations: the catalog's `Dests` dictionary (PDF
/// 1.1) and the `Dests` name tree of its `Names`, read once when first needed.
struct Names<'a> {
    document: &'a Document,
    catalog: &'a Dictionary,
    table: Option<HashMap<Vec<u8>, &'a Object>>,
}

impl<'a> Names<'a> {
    fn new(document: &'a Document, catalog: &'a Dictionary) -> Self {
        Self {
            document,
            catalog,
            table: None,
        }
    }

    fn lookup(&mut self, name: &[u8]) -> Option<&'a Object> {
        if self.table.is_none() {
            self.table = Some(self.read());
        }
        self.table.as_ref()?.get(name).copied()
    }

    fn read(&self) -> HashMap<Vec<u8>, &'a Object> {
        let document = self.document;
        let mut table = HashMap::new();
        if let Some(dests) = self
            .catalog
            .get(b"Dests")
            .ok()
            .and_then(|value| dictionary(document, value))
        {
            for (key, value) in dests.iter() {
                table.insert(key.clone(), value);
            }
        }
        let tree = self
            .catalog
            .get(b"Names")
            .ok()
            .and_then(|value| dictionary(document, value))
            .and_then(|names| names.get(b"Dests").ok());
        let mut stack: Vec<&Object> = tree.into_iter().collect();
        let mut seen: HashSet<ObjectId> = HashSet::new();
        let mut read = 0usize;
        while let Some(link) = stack.pop() {
            if let Ok(id) = link.as_reference()
                && !seen.insert(id)
            {
                continue;
            }
            read += 1;
            if read > MAX_NAME_NODES {
                break;
            }
            let Some(node) = dictionary(document, link) else {
                continue;
            };
            if let Some(pairs) = node
                .get(b"Names")
                .ok()
                .and_then(|value| resolved(document, value))
                .and_then(|value| value.as_array().ok())
            {
                for pair in pairs.chunks(2) {
                    if let [key, value] = pair
                        && let Some(key) = resolved(document, key).and_then(|key| key.as_str().ok())
                    {
                        table.entry(key.to_vec()).or_insert(value);
                    }
                }
            }
            if let Some(kids) = node
                .get(b"Kids")
                .ok()
                .and_then(|value| resolved(document, value))
                .and_then(|value| value.as_array().ok())
            {
                stack.extend(kids.iter().rev());
            }
        }
        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::{Stream, dictionary};

    /// A document of `pages` empty pages, and its page ids.
    fn blank(pages: usize) -> (Document, Vec<ObjectId>) {
        let mut document = Document::with_version("1.7");
        let tree = document.new_object_id();
        let ids: Vec<ObjectId> = (0..pages)
            .map(|_| {
                let content = document.add_object(Stream::new(dictionary! {}, Vec::new()));
                document.add_object(dictionary! {
                    "Type" => "Page",
                    "Parent" => tree,
                    "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
                    "Contents" => content,
                })
            })
            .collect();
        document.objects.insert(
            tree,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => ids.iter().map(|id| Object::Reference(*id)).collect::<Vec<_>>(),
                "Count" => pages as i64,
            }),
        );
        let catalog = document.add_object(dictionary! { "Type" => "Catalog", "Pages" => tree });
        document.trailer.set("Root", catalog);
        (document, ids)
    }

    /// Items `nodes` linked as siblings (`Next`), in order; returns their ids.
    fn siblings(document: &mut Document, nodes: Vec<Dictionary>) -> Vec<ObjectId> {
        let ids: Vec<ObjectId> = nodes.iter().map(|_| document.new_object_id()).collect();
        for (index, mut node) in nodes.into_iter().enumerate() {
            if let Some(next) = ids.get(index + 1) {
                node.set("Next", *next);
            }
            document
                .objects
                .insert(ids[index], Object::Dictionary(node));
        }
        ids
    }

    fn item(title: Object, dest: Option<Object>) -> Dictionary {
        let mut node = dictionary! { "Title" => title };
        if let Some(dest) = dest {
            node.set("Dest", dest);
        }
        node
    }

    fn text(value: &str) -> Object {
        Object::string_literal(value)
    }

    fn at(page: ObjectId) -> Option<Object> {
        Some(Object::Array(vec![page.into(), "Fit".into()]))
    }

    /// The document saved with `first` as its outline's first item.
    fn saved(mut document: Document, first: ObjectId) -> Vec<u8> {
        let outlines = document.add_object(dictionary! { "Type" => "Outlines", "First" => first });
        let catalog = document
            .trailer
            .get(b"Root")
            .unwrap()
            .as_reference()
            .unwrap();
        document
            .get_dictionary_mut(catalog)
            .unwrap()
            .set("Outlines", outlines);
        let mut bytes = Vec::new();
        document.save_to(&mut bytes).unwrap();
        bytes
    }

    fn listed(outline: &PdfOutline) -> Vec<(&str, usize, usize)> {
        outline
            .entries
            .iter()
            .map(|entry| (entry.title.as_str(), entry.page, entry.level))
            .collect()
    }

    #[test]
    fn a_nested_outline_lists_titles_pages_and_levels_in_order() {
        let (mut document, pages) = blank(5);
        // A UTF-16BE title with a character outside the BMP; a PDFDocEncoding one
        // (0x8B is "‰", 0x84 is "—"); white space and control characters folded.
        let utf16: Vec<u8> = [0xFE, 0xFF]
            .into_iter()
            .chain(
                "Évaluation 𝄞 notes"
                    .encode_utf16()
                    .flat_map(u16::to_be_bytes),
            )
            .collect();
        let children = siblings(
            &mut document,
            vec![
                item(text("1.1 Sowing"), at(pages[1])),
                item(
                    Object::String(utf16, StringFormat::Hexadecimal),
                    at(pages[2]),
                ),
            ],
        );
        let mut first = item(text("1. Seed  trays\r\n"), at(pages[0]));
        first.set("First", children[0]);
        let pdfdoc = Object::String(b"2. Yield \x8B \x84 notes".to_vec(), StringFormat::Literal);
        let top = siblings(&mut document, vec![first, item(pdfdoc, at(pages[4]))]);
        let outline = pdf_outline(&saved(document, top[0]));
        assert_eq!(
            listed(&outline),
            [
                ("1. Seed trays", 1, 1),
                ("1.1 Sowing", 2, 2),
                ("Évaluation 𝄞 notes", 3, 2),
                ("2. Yield ‰ — notes", 5, 1),
            ]
        );
        assert!(!outline.truncated);
    }

    #[test]
    fn a_pdf_without_an_outline_or_that_cannot_be_read_has_none() {
        let (mut document, _) = blank(2);
        let mut bytes = Vec::new();
        document.save_to(&mut bytes).unwrap();
        assert_eq!(pdf_outline(&bytes), PdfOutline::default());
        assert_eq!(pdf_outline(b"not a pdf"), PdfOutline::default());
        assert_eq!(pdf_outline(b""), PdfOutline::default());
    }

    #[test]
    fn go_to_actions_named_destinations_and_page_indexes_resolve_to_pages() {
        let (mut document, pages) = blank(4);
        // A name tree with one kid, and the older catalog `Dests` dictionary.
        let leaf = document.add_object(dictionary! {
            "Names" => vec![text("chapter-two"), Object::Array(vec![pages[1].into(), "Fit".into()])],
        });
        let tree = document.add_object(dictionary! { "Kids" => vec![leaf.into()] });
        let names = document.add_object(dictionary! { "Dests" => tree });
        let dests = document.add_object(dictionary! {
            "appendix" => dictionary! { "D" => vec![pages[3].into(), "XYZ".into()] },
        });
        let catalog = document
            .trailer
            .get(b"Root")
            .unwrap()
            .as_reference()
            .unwrap();
        let dict = document.get_dictionary_mut(catalog).unwrap();
        dict.set("Names", names);
        dict.set("Dests", dests);
        let mut action = item(text("By action"), None);
        action.set(
            "A",
            dictionary! { "S" => "GoTo", "D" => vec![pages[2].into(), "Fit".into()] },
        );
        let mut uri = item(text("Elsewhere"), None);
        uri.set(
            "A",
            dictionary! { "S" => "URI", "URI" => text("https://example.invalid/") },
        );
        let mut remote = item(text("Another file"), None);
        remote.set("A", dictionary! { "S" => "GoToR", "F" => text("other.pdf"), "D" => vec![0.into(), "Fit".into()] });
        let top = siblings(
            &mut document,
            vec![
                action,
                item(text("By string name"), Some(text("chapter-two"))),
                item(
                    text("By name object"),
                    Some(Object::Name(b"appendix".to_vec())),
                ),
                item(
                    text("By index"),
                    Some(Object::Array(vec![0.into(), "Fit".into()])),
                ),
                uri,
                remote,
            ],
        );
        // The URI and other-file bookmarks go to no page here, and no entry with a page follows.
        assert_eq!(
            listed(&pdf_outline(&saved(document, top[0]))),
            [
                ("By action", 3, 1),
                ("By string name", 2, 1),
                ("By name object", 4, 1),
                ("By index", 1, 1)
            ]
        );
    }

    #[test]
    fn an_entry_without_a_page_opens_the_next_entrys_page() {
        let (mut document, pages) = blank(3);
        let chapter = siblings(&mut document, vec![item(text("Chapter"), at(pages[1]))]);
        let mut part = item(text("Part one"), None);
        part.set("First", chapter[0]);
        let broken = Some(Object::Array(vec![
            Object::Reference((999, 0)),
            "Fit".into(),
        ]));
        let top = siblings(
            &mut document,
            vec![
                part,
                item(text("Broken"), broken),
                item(text("Last"), at(pages[2])),
            ],
        );
        assert_eq!(
            listed(&pdf_outline(&saved(document, top[0]))),
            [
                ("Part one", 2, 1),
                ("Chapter", 2, 2),
                ("Broken", 3, 1),
                ("Last", 3, 1)
            ]
        );
    }

    #[test]
    fn looping_links_and_long_titles_are_bounded() {
        let (mut document, pages) = blank(2);
        // Two items whose `Next` links point at each other; the first is also its own child.
        let a = document.new_object_id();
        let b = document.new_object_id();
        let long = "Bed ".repeat(200);
        let mut first = item(text(&long), at(pages[0]));
        first.set("Next", b);
        first.set("First", a);
        let mut second = item(text("B"), at(pages[1]));
        second.set("Next", a);
        document.objects.insert(a, Object::Dictionary(first));
        document.objects.insert(b, Object::Dictionary(second));
        let outline = pdf_outline(&saved(document, a));
        assert_eq!(outline.entries.len(), 2);
        let title = &outline.entries[0].title;
        assert_eq!(title.chars().count(), MAX_OUTLINE_TITLE - 1);
        assert!(title.starts_with("Bed Bed") && title.ends_with("Bed"));
        assert_eq!(outline.entries[1].title, "B");
    }

    #[test]
    fn an_outline_longer_than_the_limit_is_cut_and_says_so() {
        let (mut document, pages) = blank(2);
        let count = MAX_OUTLINE_ENTRIES + 5;
        let nodes = (0..count)
            .map(|index| item(text(&format!("Row {index}")), at(pages[index % 2])))
            .collect();
        let top = siblings(&mut document, nodes);
        let outline = pdf_outline(&saved(document, top[0]));
        assert_eq!(outline.entries.len(), MAX_OUTLINE_ENTRIES);
        assert!(outline.truncated);
        let last = &outline.entries[MAX_OUTLINE_ENTRIES - 1];
        assert_eq!((last.title.as_str(), last.page), ("Row 1999", 2));
    }

    #[test]
    fn text_strings_decode_from_each_encoding() {
        assert_eq!(decode_text(b"\xFE\xFF\x00A\x4E\x2D"), "A中");
        assert_eq!(decode_text(b"\xFF\xFEA\x00\x2D\x4E"), "A中");
        assert_eq!(decode_text("\u{FEFF}café".as_bytes()), "café");
        assert_eq!(decode_text(b"caf\xE9"), "café");
        // A lone surrogate and an odd last byte do not fail the title.
        assert_eq!(decode_text(b"\xFE\xFF\xD8\x00"), "\u{FFFD}");
        assert_eq!(decode_text(b"\xFE\xFF\x00\x41\x00"), "A\u{0}");
    }
}
