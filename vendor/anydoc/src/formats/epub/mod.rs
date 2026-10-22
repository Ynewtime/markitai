//! EPUB: XHTML chapters in spine order, concatenated into one document with
//! chapter-scoped anchors so intra-book navigation survives.

use crate::error::ConvertError;
use crate::model::{
    AnchorId, Block, CellSlot, Document, ImageSource, Inline, LinkTarget, Note,
    inlines_to_plain_text,
};
use crate::package::xml::Element;
use crate::package::{Package, path};
use crate::shared::assets::{AssetSink, media_type_for};
use crate::shared::html::{HtmlCtx, Stylesheet};
use crate::shared::uri::is_absolute_uri;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

pub fn parse(bytes: &[u8]) -> Result<Document, ConvertError> {
    let pkg = RefCell::new(Package::open(bytes)?);

    let container = pkg.borrow_mut().required_xml_part("META-INF/container.xml")?;
    let opf_path = container
        .descendants_any("rootfile")
        .next()
        .and_then(|r| r.attr_any("full-path"))
        .ok_or_else(|| ConvertError::malformed_part("META-INF/container.xml", "no rootfile entry"))?
        .to_string();

    let opf = pkg.borrow_mut().required_xml_part(&opf_path)?;

    let mut doc = Document::default();
    if let Some(title) = opf.descendants_any("title").next().map(|t| t.text()) {
        let title = title.trim().to_string();
        if !title.is_empty() {
            doc.blocks.push(Block::heading(1, vec![Inline::plain(title)]));
        }
    }

    let mut manifest: HashMap<String, (String, String)> = HashMap::new();
    for item in opf.descendants_any("item") {
        if let (Some(id), Some(href)) = (item.attr_any("id"), item.attr_any("href")) {
            let media = item.attr_any("media-type").unwrap_or("").to_string();
            manifest.insert(id.to_string(), (href.to_string(), media));
        }
    }

    // Every spine part in spine order: non-linear items are auxiliary but
    // still publication content, and unusable parts degrade at parse time.
    // Intra-book links target these; links to any other resource stay
    // Relative.
    // A part holds one position in a reading order, and each repeat would
    // cost another parse of it and another copy of its anchor.
    let mut spine_entries = 0usize;
    let mut spine_paths: Vec<String> = Vec::new();
    let mut spine_parts: HashSet<String> = HashSet::new();
    for href in opf
        .descendants_any("itemref")
        .filter_map(|ir| ir.attr_any("idref"))
        .filter_map(|idref| manifest.get(idref))
        .map(|(href, _)| href.as_str())
    {
        spine_entries += 1;
        let target = match path::resolve(&opf_path, href) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("skipping chapter with unresolvable href {href:?}: {e}");
                continue;
            }
        };
        if spine_parts.insert(target.path.clone()) {
            spine_paths.push(target.path);
        }
    }

    let assets = RefCell::new(AssetSink::new());
    let notes = RefCell::new(Vec::new());
    let cover_at = doc.blocks.len();
    let mut css_cache: HashMap<String, Option<String>> = HashMap::new();
    let mut converted = 0usize;
    for chapter_path in &spine_paths {
        let Some(tree) = pkg.borrow_mut().optional_xml_part(chapter_path)? else {
            log::warn!("skipping unusable chapter {chapter_path}");
            continue;
        };
        let Some(body) = tree
            .child_elems()
            .find(|e| e.local == "html")
            .and_then(|h| h.child_elems().find(|e| e.local == "body"))
        else {
            log::warn!("skipping chapter {chapter_path}: no body element");
            continue;
        };
        let css = chapter_stylesheet(&tree, chapter_path, &pkg, &mut css_cache)?;
        let ctx = ChapterCtx {
            pkg: &pkg,
            assets: &assets,
            chapter_path: chapter_path.clone(),
            spine_parts: &spine_parts,
            notes: &notes,
        };
        // Chapter-start anchor: renders only when a link targets this chapter.
        doc.blocks.push(Block::Paragraph(vec![Inline::Anchor(chapter_path.clone())]));
        doc.blocks.extend(crate::shared::html::to_blocks(body, &css, &ctx)?);
        converted += 1;
    }
    if spine_entries > 0 && converted == 0 {
        return Err(ConvertError::malformed("no chapter in the book could be read"));
    }
    // markitai: a cover the package names (EPUB 3's `cover-image` item, EPUB
    // 2's `<meta name="cover">`) that no page shows comes first, as a
    // reader's library shows it.
    if let Some(cover) = opf_cover(&opf, &manifest)
        && let Ok(target) = path::resolve(&opf_path, cover)
        && !assets.borrow().contains_part(&target.path)
        && let Some(bytes) = pkg.borrow_mut().optional_part(&target.path)?
    {
        let media = media_type_for(&target.path);
        let id = assets.borrow_mut().add(media, target.path, &bytes)?;
        doc.blocks.insert(
            cover_at,
            Block::Paragraph(vec![Inline::Image {
                alt: "Cover".into(),
                source: ImageSource::Asset(id),
            }]),
        );
    }

    doc.assets = std::mem::take(&mut assets.borrow_mut().assets);
    doc.notes = notes.into_inner();
    // markitai: a short link to a note (`<a href="#fn1">1</a>`) is its
    // reference mark, whatever the markup calls it. Notes are numbered
    // through the book in reading order: their chapter-scoped anchors
    // (`EPUB/text/ch001.xhtml#fn1`) are no label for a reader.
    let ids: HashMap<String, String> = doc
        .notes
        .iter()
        .enumerate()
        .map(|(n, note)| (note.id.clone(), (n + 1).to_string()))
        .collect();
    if !ids.is_empty() {
        for note in &mut doc.notes {
            note.id = ids[&note.id].clone();
        }
        mark_note_refs(&mut doc.blocks, &ids);
        for note in &mut doc.notes {
            mark_note_refs(&mut note.blocks, &ids);
        }
    }
    Ok(doc)
}

/// markitai: the href of the picture the package names as the book's cover:
/// the manifest item whose `properties` hold `cover-image` (EPUB 3), else the
/// item `<meta name="cover" content="…">` names by id (EPUB 2). Only an
/// image item counts.
fn opf_cover<'m>(
    opf: &Element,
    manifest: &'m HashMap<String, (String, String)>,
) -> Option<&'m str> {
    let image = |id: &str| {
        manifest
            .get(id)
            .filter(|(_, media)| media.starts_with("image/"))
            .map(|(href, _)| href.as_str())
    };
    opf.descendants_any("item")
        .find(|item| {
            item.attr_any("properties")
                .is_some_and(|value| value.split_whitespace().any(|word| word == "cover-image"))
        })
        .and_then(|item| item.attr_any("id"))
        .and_then(image)
        .or_else(|| {
            opf.descendants_any("meta")
                .find(|meta| meta.attr_any("name") == Some("cover"))
                .and_then(|meta| meta.attr_any("content"))
                .and_then(image)
        })
}

/// markitai: the longest link text read as a note's mark (`[12]`, `iv`, `*`).
const MAX_NOTE_MARK: usize = 8;

fn mark_note_refs(blocks: &mut [Block], ids: &HashMap<String, String>) {
    // markitai: longer prose linking to a note keeps its words, followed by
    // the note's reference: a link to the note's chapter-scoped anchor
    // (`#EPUB/c2.xhtml#fn1`) leads nowhere once the note is a definition.
    fn walk(inlines: &mut Vec<Inline>, ids: &HashMap<String, String>) {
        for inline in std::mem::take(inlines) {
            match inline {
                Inline::Link { content, target: LinkTarget::Anchor(id) }
                    if ids.contains_key(&id)
                        && !content.iter().any(|c| matches!(c, Inline::Image { .. }))
                        && inlines_to_plain_text(&content).trim().chars().count()
                            <= MAX_NOTE_MARK =>
                {
                    inlines.push(Inline::NoteRef(ids[&id].clone()));
                }
                Inline::Link { mut content, target: LinkTarget::Anchor(id) }
                    if ids.contains_key(&id) =>
                {
                    walk(&mut content, ids);
                    inlines.extend(content);
                    inlines.push(Inline::NoteRef(ids[&id].clone()));
                }
                Inline::Link { mut content, target } => {
                    walk(&mut content, ids);
                    inlines.push(Inline::Link { content, target });
                }
                other => inlines.push(other),
            }
        }
    }
    for block in blocks {
        match block {
            Block::Heading { content, .. } | Block::Paragraph(content) => walk(content, ids),
            Block::List(list) => {
                for item in &mut list.items {
                    mark_note_refs(&mut item.blocks, ids);
                }
            }
            Block::Table(table) => {
                for row in &mut table.grid {
                    for slot in row {
                        if let CellSlot::Origin(cell) = slot {
                            mark_note_refs(&mut cell.blocks, ids);
                        }
                    }
                }
            }
            Block::BlockQuote(inner) => mark_note_refs(inner, ids),
            Block::CodeBlock { .. } | Block::Rule | Block::Math(_) => {}
        }
    }
}

/// A chapter's CSS cascade: its linked stylesheets and inline `<style>`
/// blocks, in document order. Stylesheet parts are cached across chapters.
fn chapter_stylesheet(
    tree: &Element,
    chapter_path: &str,
    pkg: &RefCell<Package>,
    cache: &mut HashMap<String, Option<String>>,
) -> Result<Stylesheet, ConvertError> {
    let mut css = Stylesheet::default();
    let mut stack: Vec<&Element> = tree.child_elems().collect();
    stack.reverse();
    while let Some(elem) = stack.pop() {
        match elem.local.as_str() {
            "link" => {
                let rel = elem.attr_any("rel").unwrap_or("");
                let is_sheet = rel.split_whitespace().any(|r| r.eq_ignore_ascii_case("stylesheet"));
                if is_sheet && let Some(href) = elem.attr_any("href") {
                    let Ok(target) = path::resolve(chapter_path, href) else {
                        continue;
                    };
                    if !cache.contains_key(&target.path) {
                        let text = pkg
                            .borrow_mut()
                            .optional_part(&target.path)?
                            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned());
                        cache.insert(target.path.clone(), text);
                    }
                    if let Some(Some(text)) = cache.get(&target.path) {
                        css.add(text);
                    }
                }
            }
            "style" => css.add(&elem.text()),
            _ => {
                let start = stack.len();
                stack.extend(elem.child_elems());
                stack[start..].reverse();
            }
        }
    }
    Ok(css)
}

struct ChapterCtx<'a, 'b> {
    pkg: &'b RefCell<Package<'a>>,
    assets: &'b RefCell<AssetSink>,
    chapter_path: String,
    spine_parts: &'b HashSet<String>,
    /// markitai: the book's footnotes and endnotes, in reading order.
    notes: &'b RefCell<Vec<Note>>,
}

impl HtmlCtx for ChapterCtx<'_, '_> {
    fn link_target(&self, href: &str) -> Option<LinkTarget> {
        if href.is_empty() {
            return None;
        }
        if let Some(fragment) = href.strip_prefix('#') {
            let fragment = path::decode_fragment(fragment);
            return Some(LinkTarget::Anchor(scoped(&self.chapter_path, Some(&fragment))));
        }
        if is_absolute_uri(href) {
            return Some(LinkTarget::External(href.to_string()));
        }
        // Anchors only for converted spine documents; links to any other
        // package resource (images, downloads, non-linear content) keep
        // their relative form.
        match path::resolve(&self.chapter_path, href) {
            Ok(target) if self.spine_parts.contains(&target.path) => {
                Some(LinkTarget::Anchor(scoped(&target.path, target.fragment.as_deref())))
            }
            _ => Some(LinkTarget::Relative(href.to_string())),
        }
    }

    fn image_source(&self, src: &str) -> Result<Option<ImageSource>, ConvertError> {
        if src.is_empty() {
            return Ok(None);
        }
        if is_absolute_uri(src) {
            return Ok(Some(ImageSource::External(src.to_string())));
        }
        let Ok(target) = path::resolve(&self.chapter_path, src) else {
            return Ok(None);
        };
        match self.pkg.borrow_mut().optional_part(&target.path)? {
            Some(bytes) => {
                let media = media_type_for(&target.path);
                let id = self.assets.borrow_mut().add(media, target.path, &bytes)?;
                Ok(Some(ImageSource::Asset(id)))
            }
            None => Ok(None),
        }
    }

    fn anchor_id(&self, raw: &str) -> AnchorId {
        scoped(&self.chapter_path, Some(raw))
    }

    fn collects_notes(&self) -> bool {
        true
    }

    fn note(&self, note: Note) {
        self.notes.borrow_mut().push(note);
    }
}

/// Chapter-scoped anchor id: the chapter path itself targets the chapter
/// start; `path#fragment` targets an element inside it.
fn scoped(chapter_path: &str, fragment: Option<&str>) -> AnchorId {
    match fragment {
        Some(f) if !f.is_empty() => format!("{chapter_path}#{f}"),
        _ => chapter_path.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    /// markitai: a book of the given chapters, in spine order.
    fn book(chapters: &[&str]) -> Vec<u8> {
        let items: String = (0..chapters.len())
            .map(|i| {
                format!(r#"<item id="c{i}" href="c{i}.xhtml" media-type="application/xhtml+xml"/>"#)
            })
            .collect();
        let refs: String =
            (0..chapters.len()).map(|i| format!(r#"<itemref idref="c{i}"/>"#)).collect();
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default();
        w.start_file("META-INF/container.xml", options).unwrap();
        w.write_all(
            br#"<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="c.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"#,
        )
        .unwrap();
        w.start_file("c.opf", options).unwrap();
        w.write_all(
            format!(
                r#"<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata/><manifest>{items}</manifest><spine>{refs}</spine></package>"#
            )
            .as_bytes(),
        )
        .unwrap();
        for (i, body) in chapters.iter().enumerate() {
            w.start_file(format!("c{i}.xhtml"), options).unwrap();
            w.write_all(
                format!(
                    r#"<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><body>{body}</body></html>"#
                )
                .as_bytes(),
            )
            .unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    #[test]
    fn footnotes_are_markdown_footnotes_not_anchors_and_links() {
        // Pandoc's EPUB 3 (epub:type) and EPUB 2 (class) notes, each
        // chapter numbering its own from 1.
        let epub3 = r##"<section><h1>One</h1><p>A note.<a href="#fn1" class="footnote-ref" id="fnref1" epub:type="noteref">1</a> And <a href="#fn2" epub:type="noteref">2</a>.</p></section><section id="footnotes" class="footnotes footnotes-end-of-document" epub:type="footnotes"><hr /><aside epub:type="footnote" id="fn1"><p><a href="#fnref1" class="footnote-back" role="doc-backlink">1</a>. The first note, with <em>emphasis</em>.</p></aside><aside epub:type="footnote" id="fn2"><p><a href="#fnref2" class="footnote-back" role="doc-backlink">2</a>. The second note.</p></aside></section>"##;
        let epub2 = r##"<div><h1>Two</h1><p>Third.<a href="#fn1" class="footnote-ref" id="fnref1">1</a> See <a href="#fn1">the note on sources</a>.</p></div><div class="footnotes"><hr /><ol><li id="fn1"><p>The third note.<a href="#fnref1" class="footnote-back">↩︎</a></p></li></ol></div>"##;
        let doc = parse(&book(&[epub3, epub2])).unwrap();
        // Numbered through the book, not by their chapter-scoped anchors.
        let ids: Vec<&str> = doc.notes.iter().map(|note| note.id.as_str()).collect();
        assert_eq!(ids, ["1", "2", "3"]);
        let markdown = crate::render::markdown::document_to_markdown(&doc);
        assert!(markdown.contains("A note.[^1] And [^2]."), "{markdown}");
        assert!(markdown.contains("Third.[^3]"), "{markdown}");
        assert!(markdown.contains("[^1]: The first note, with *emphasis*."), "{markdown}");
        assert!(markdown.contains("[^2]: The second note."), "{markdown}");
        assert!(markdown.contains("[^3]: The third note."), "{markdown}");
        for stray in ["<a id", "c0.xhtml", "c1.xhtml", "↩", "---", "](#"] {
            assert!(!markdown.contains(stray), "{stray}: {markdown}");
        }
        // Prose linking to a note keeps its words, and points at the note.
        assert!(markdown.contains("See the note on sources[^3]."), "{markdown}");
    }

    /// markitai: a book of one chapter and a cover picture the package names
    /// as `manifest` and `metadata` give.
    fn covered_book(metadata: &str, cover_item: &str, chapter: &str) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default();
        let parts = [
            (
                "META-INF/container.xml",
                r#"<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OPS/c.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"#.to_string(),
            ),
            (
                "OPS/c.opf",
                format!(
                    r#"<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Invented</dc:title>{metadata}</metadata><manifest><item id="c0" href="Text/c0.xhtml" media-type="application/xhtml+xml"/>{cover_item}</manifest><spine><itemref idref="c0"/></spine></package>"#
                ),
            ),
            (
                "OPS/Text/c0.xhtml",
                format!(
                    r#"<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><body>{chapter}</body></html>"#
                ),
            ),
            ("OPS/Images/cover.png", "not decoded here".to_string()),
        ];
        for (name, body) in &parts {
            w.start_file(*name, options).unwrap();
            w.write_all(body.as_bytes()).unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    /// markitai: the alt text of each picture the book's blocks show, in order.
    fn pictures(doc: &Document) -> Vec<String> {
        doc.blocks
            .iter()
            .filter_map(|block| match block {
                Block::Paragraph(inlines) => Some(inlines),
                _ => None,
            })
            .flatten()
            .filter_map(|inline| match inline {
                Inline::Image { alt, source: ImageSource::Asset(_) } => Some(alt.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_cover_only_the_package_names_comes_first() {
        let epub3 = r#"<item id="cv" properties="cover-image" href="Images/cover.png" media-type="image/png"/>"#;
        let epub2 = r#"<item id="cv" href="Images/cover.png" media-type="image/png"/>"#;
        for (metadata, item) in [("", epub3), (r#"<meta name="cover" content="cv"/>"#, epub2)] {
            let doc = parse(&covered_book(metadata, item, "<h1>One</h1><p>Text.</p>")).unwrap();
            assert_eq!(doc.assets.len(), 1, "{item}");
            assert_eq!(doc.assets[0].origin_part, "OPS/Images/cover.png");
            assert_eq!(pictures(&doc), ["Cover"]);
            // After the title, before the first chapter.
            assert!(matches!(&doc.blocks[1], Block::Paragraph(inlines)
                if matches!(inlines[..], [Inline::Image { .. }])));
        }
        // A page that shows it already: once, where the page has it.
        let doc = parse(&covered_book(
            "",
            epub3,
            r#"<p><img src="../Images/cover.png" alt="Front"/></p><h1>One</h1>"#,
        ))
        .unwrap();
        assert_eq!(doc.assets.len(), 1);
        assert_eq!(pictures(&doc), ["Front"]);
        // A `cover` meta naming no image item names no cover.
        let doc = parse(&covered_book(
            r#"<meta name="cover" content="c0"/>"#,
            "",
            "<p>Text.</p>",
        ))
        .unwrap();
        assert!(doc.assets.is_empty());
    }

    #[test]
    fn a_part_repeated_across_the_spine_is_read_once() {
        let items: String = (0..64)
            .map(|i| {
                format!(r#"<item id="i{i}" href="ch.xhtml" media-type="application/xhtml+xml"/>"#)
            })
            .collect();
        let refs: String = (0..64).map(|i| format!(r#"<itemref idref="i{i}"/>"#)).collect();
        let parts = [
            (
                "META-INF/container.xml",
                r#"<?xml version="1.0"?>
                <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
                <rootfiles><rootfile full-path="c.opf"
                media-type="application/oebps-package+xml"/></rootfiles></container>"#
                    .to_string(),
            ),
            (
                "c.opf",
                format!(
                    r#"<?xml version="1.0"?>
                    <package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata/>
                    <manifest>{items}</manifest><spine>{refs}</spine></package>"#
                ),
            ),
            (
                "ch.xhtml",
                r#"<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml">
                <body><p>chapter text</p></body></html>"#
                    .to_string(),
            ),
        ];
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, body) in &parts {
            w.start_file(*name, zip::write::SimpleFileOptions::default()).unwrap();
            w.write_all(body.as_bytes()).unwrap();
        }

        let doc = parse(&w.finish().unwrap().into_inner()).unwrap();
        let text = doc
            .blocks
            .iter()
            .filter_map(|b| match b {
                Block::Paragraph(inlines) => Some(crate::model::inlines_to_plain_text(inlines)),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(text, "chapter text", "sixty-four itemrefs naming one part");
    }
}
