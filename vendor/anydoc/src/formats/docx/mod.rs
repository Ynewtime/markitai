//! OOXML WordprocessingML (.docx / .docm).
//!
//! Resolution pipeline: package parts -> style/numbering models ->
//! spec-order property resolution -> document model.

// markitai: embedded parts (`w:altChunk`).
mod altchunk;
mod code;
mod content;
mod numbering;
mod numerals;
// markitai: shared with the ODF and RTF readers.
pub(crate) mod scripts;
mod styles;
// markitai: shared with the RTF reader.
pub(crate) mod symbols;

use crate::error::ConvertError;
use crate::model::{Document, Note, NoteKind};
use crate::package::Package;
use crate::package::relationships::{Relationships, read_rels, rel_type, rels_part_for};
use crate::package::xml::ns;
use crate::shared::assets::AssetSink;
use content::Ctx;
use numbering::Counters;
use std::cell::RefCell;

pub fn parse(bytes: &[u8]) -> Result<Document, ConvertError> {
    parse_at(bytes, 0, 0).map(|(document, _)| document)
}

/// markitai: a Word package stored `depth` `altChunk`s deep inside another
/// (0 for the document itself), whose decompressed bytes count on from the
/// `spent` bytes the outer packages have read; with the total reached.
fn parse_at(bytes: &[u8], depth: u32, spent: u64) -> Result<(Document, u64), ConvertError> {
    let pkg = match Package::open_spent(bytes, spent) {
        Ok(p) => p,
        Err(e) => return Err(crate::package::archive::probe_ole(bytes).unwrap_or(e)),
    };
    let pkg = RefCell::new(pkg);

    // OPC part discovery: the main part comes from the package-level
    // officeDocument relationship; its own parts (styles, numbering, notes)
    // from the main part's typed relationships. Conventional paths are the
    // fallback for packages with missing or unusable rels.
    let root_rels = read_rels(&mut pkg.borrow_mut(), "_rels/.rels")?;
    let main_part = root_rels
        .first_of_type(rel_type::OFFICE_DOCUMENT)
        .and_then(|rel| crate::package::path::resolve("", &rel.target).ok())
        .map(|t| t.path)
        .unwrap_or_else(|| "word/document.xml".to_string());
    let doc_rels = read_rels(&mut pkg.borrow_mut(), &rels_part_for(&main_part))?;

    let styles_part = typed_part_path(&doc_rels, &main_part, rel_type::STYLES, "styles.xml");
    let styles_tree = pkg.borrow_mut().optional_xml_part(&styles_part)?;
    let styles =
        styles::Styles::parse_opt(styles_tree.as_ref().and_then(|t| t.find(ns::W, "styles")));

    let numbering_part =
        typed_part_path(&doc_rels, &main_part, rel_type::NUMBERING, "numbering.xml");
    let numbering_tree = pkg.borrow_mut().optional_xml_part(&numbering_part)?;
    let numbering = match numbering_tree.as_ref().and_then(|t| t.find(ns::W, "numbering")) {
        Some(root) => numbering::parse(root, &|style_id| styles.direct_num_id(style_id))?,
        None => Default::default(),
    };

    let doc_tree = pkg.borrow_mut().required_xml_part(&main_part)?;
    let body = doc_tree
        .find(ns::W, "document")
        .and_then(|d| d.find(ns::W, "body"))
        .ok_or_else(|| ConvertError::malformed_part(main_part.clone(), "no document body"))?;

    let counters = RefCell::new(Counters::default());
    let assets = RefCell::new(AssetSink::new());
    // markitai: how the body's text looks, for headings set by hand, and
    // the paragraphs that may be rows of tables set with tab stops.
    let looks = RefCell::new(crate::shared::visual::Looks::default());
    let tab_rows = RefCell::new(crate::shared::tabs::TabRows::default());
    let typed_lists = RefCell::new(crate::shared::typed_lists::TypedLists::default());
    // markitai: what the document's embedded parts (`w:altChunk`) add.
    let embedded = RefCell::new(altchunk::Embedded::default());

    let footnotes_part =
        typed_part_path(&doc_rels, &main_part, rel_type::FOOTNOTES, "footnotes.xml");
    let endnotes_part = typed_part_path(&doc_rels, &main_part, rel_type::ENDNOTES, "endnotes.xml");

    let ctx = Ctx {
        pkg: &pkg,
        rels: doc_rels,
        base_part: main_part,
        styles: &styles,
        numbering: &numbering,
        counters: &counters,
        assets: &assets,
        code_fonts: code::sets_code_apart(body, &styles)?,
        cell_depth: Default::default(),
        looks: Some(&looks),
        block_depth: Default::default(),
        tabs: Some(&tab_rows),
        lists: Some(&typed_lists),
        chunk_depth: depth,
        embedded: &embedded,
    };
    let mut blocks = content::parse_blocks(body, &ctx)?;
    // markitai: tables set with tab stops, headings set by hand and lists
    // typed by hand (see `crate::shared::tabs`, `crate::shared::visual` and
    // `crate::shared::typed_lists`), and a table laying out a code listing
    // is its code (`crate::shared::code`).
    crate::shared::tabs::finish(
        tab_rows.take(),
        typed_lists.take(),
        Some(looks.take()),
        &mut blocks,
        &mut [],
    );
    crate::shared::code::listing_tables(&mut blocks);

    let mut notes = Vec::new();
    for (part, root_name, elem_name, prefix, kind) in [
        (footnotes_part, "footnotes", "footnote", "fn", NoteKind::Footnote),
        (endnotes_part, "endnotes", "endnote", "en", NoteKind::Endnote),
    ] {
        let Some(tree) = pkg.borrow_mut().optional_xml_part(&part)? else {
            continue;
        };
        let Some(root) = tree.find(ns::W, root_name) else {
            continue;
        };
        let note_rels = read_rels(&mut pkg.borrow_mut(), &rels_part_for(&part))?;
        let note_ctx = ctx.for_part(note_rels, part);
        for note in root.find_all(ns::W, elem_name) {
            if matches!(
                note.attr(ns::W, "type"),
                Some("separator") | Some("continuationSeparator") | Some("continuationNotice")
            ) {
                continue;
            }
            let Some(id) = note.attr(ns::W, "id") else { continue };
            notes.push(Note {
                id: format!("{prefix}{id}"),
                kind,
                blocks: content::parse_blocks(note, &note_ctx)?,
            });
        }
    }
    // markitai: a note's table laying out a code listing is its code too.
    for note in &mut notes {
        crate::shared::code::listing_tables(&mut note.blocks);
    }

    // markitai: the notes of embedded documents follow the document's own.
    let warnings = embedded.into_inner().finish(&mut notes);
    let assets = std::mem::take(&mut assets.borrow_mut().assets);
    let total = pkg.borrow().total_read();
    Ok((Document { blocks, notes, assets, slide_starts: Vec::new(), warnings }, total))
}

/// Path of a typed related part, resolved against the main part; falls back
/// to the conventional sibling name when the relationship is absent.
fn typed_part_path(rels: &Relationships, base: &str, rel_type: &str, fallback: &str) -> String {
    let reference = rels.first_of_type(rel_type).map(|rel| rel.target.as_str()).unwrap_or(fallback);
    match crate::package::path::resolve(base, reference) {
        Ok(target) => target.path,
        Err(e) => {
            log::warn!("skipping unresolvable related-part target {reference:?}: {e}");
            crate::package::path::resolve(base, fallback)
                .map(|t| t.path)
                .unwrap_or_else(|_| fallback.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Block, ImageSource, Inline};
    use std::io::{Cursor, Write};

    fn docx_parts(parts: &[(&str, &str)]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default();
        for (name, body) in parts {
            w.start_file(*name, opts).unwrap();
            w.write_all(body.as_bytes()).unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    fn docx(document: &str, rels: &str) -> Vec<u8> {
        docx_parts(&[("word/document.xml", document), ("word/_rels/document.xml.rels", rels)])
    }

    const W: &str = r#"xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main""#;

    fn find_image(blocks: &[Block]) -> Option<&Inline> {
        blocks.iter().find_map(|b| match b {
            Block::Paragraph(inlines) => inlines.iter().find(|i| matches!(i, Inline::Image { .. })),
            _ => None,
        })
    }

    #[test]
    fn linked_image_relationship_becomes_an_external_source() {
        // M9: `r:link` with an external-mode relationship must carry the URL
        // instead of failing the internal-part loader.
        let document = r#"<w:document
            xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
            <w:body><w:p><w:r><w:drawing>
                <a:blip r:link="rId9"/>
            </w:drawing></w:r></w:p></w:body></w:document>"#;
        let rels = r#"<Relationships
            xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
            <Relationship Id="rId9"
                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
                Target="https://e.com/pic.png" TargetMode="External"/>
            </Relationships>"#;
        let doc = parse(&docx(document, rels)).unwrap();
        let image = find_image(&doc.blocks).expect("image inline");
        let Inline::Image { source, .. } = image else { unreachable!() };
        assert_eq!(*source, ImageSource::External("https://e.com/pic.png".into()));
        assert!(doc.assets.is_empty(), "external images are not retained as assets");
    }

    fn numbering_with_start(start: &str) -> String {
        format!(
            r#"<w:numbering {W}>
            <w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0">
                <w:numFmt w:val="decimal"/><w:start w:val="{start}"/>
            </w:lvl></w:abstractNum>
            <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
            </w:numbering>"#
        )
    }

    fn numbered_paragraphs() -> String {
        let para = r#"<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>
            <w:r><w:t>item</w:t></w:r></w:p>"#;
        format!(r#"<w:document {W}><w:body>{para}{para}</w:body></w:document>"#)
    }

    #[test]
    fn huge_numbering_start_values_cannot_overflow() {
        // H2: w:start is ST_DecimalNumber (xsd:int); out-of-range values are
        // clamped so document-order increments can never overflow.
        for start in ["18446744073709551615", "-5", "2147483647"] {
            let bytes = docx_parts(&[
                ("word/document.xml", &numbered_paragraphs()),
                ("word/numbering.xml", &numbering_with_start(start)),
            ]);
            let doc = parse(&bytes).expect(start);
            assert!(!doc.blocks.is_empty());
        }
    }

    #[test]
    fn partial_num_pr_inherits_property_by_property() {
        // M1: a direct numPr carrying only ilvl merges with the style's
        // numId instead of suppressing numbering.
        let document = format!(
            r#"<w:document {W}><w:body>
            <w:p><w:pPr><w:pStyle w:val="Listy"/>
                <w:numPr><w:ilvl w:val="1"/></w:numPr></w:pPr>
                <w:r><w:t>second level</w:t></w:r></w:p>
            </w:body></w:document>"#
        );
        let styles = format!(
            r#"<w:styles {W}>
            <w:style w:type="paragraph" w:styleId="Listy">
                <w:pPr><w:numPr><w:numId w:val="1"/></w:numPr></w:pPr>
            </w:style></w:styles>"#
        );
        let numbering = format!(
            r#"<w:numbering {W}>
            <w:abstractNum w:abstractNumId="0">
                <w:lvl w:ilvl="0"><w:numFmt w:val="decimal"/><w:start w:val="1"/></w:lvl>
                <w:lvl w:ilvl="1"><w:numFmt w:val="lowerLetter"/><w:start w:val="1"/></w:lvl>
            </w:abstractNum>
            <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
            </w:numbering>"#
        );
        let bytes = docx_parts(&[
            ("word/document.xml", &document),
            ("word/styles.xml", &styles),
            ("word/numbering.xml", &numbering),
        ]);
        let doc = parse(&bytes).unwrap();
        let Some(Block::List(list)) = doc.blocks.first() else {
            panic!("expected a list, got {:?}", doc.blocks.first());
        };
        assert_eq!(
            list.marker,
            crate::model::MarkerKind::LowerAlpha,
            "level 1 of the style's numbering"
        );
    }

    #[test]
    fn unmarked_run_edge_whitespace_is_kept() {
        // Converters that never write xml:space carry inter-word spacing on
        // run edges; dropping it glues the words together.
        let document = format!(
            r#"<w:document {W}><w:body><w:p>
            <w:r><w:t>This</w:t></w:r>
            <w:r><w:t> by-law</w:t></w:r>
            <w:r><w:t> grants</w:t></w:r>
            </w:p></w:body></w:document>"#
        );
        let doc = parse(&docx_parts(&[("word/document.xml", &document)])).unwrap();
        let Some(Block::Paragraph(inlines)) = doc.blocks.first() else { panic!() };
        assert_eq!(crate::model::inlines_to_plain_text(inlines), "This by-law grants");
    }

    #[test]
    fn page_break_between_runs_keeps_the_word_boundary() {
        // A w:br is unrepresentable in Markdown whatever its type, but the
        // runs it separates must not merge into a word the document never
        // had. A break ending a paragraph still leaves no stray marker.
        let cases = [
            (
                r#"<w:p><w:r><w:t>Alfa</w:t><w:br w:type="page"/><w:t>Beta</w:t></w:r></w:p>"#,
                "Alfa\\\nBeta\n",
            ),
            (
                r#"<w:p><w:r><w:t>Alfa</w:t><w:br w:type="page"/></w:r></w:p>
                <w:p><w:r><w:t>Beta</w:t></w:r></w:p>"#,
                "Alfa\n\nBeta\n",
            ),
        ];
        for (body, expected) in cases {
            let document = format!(r#"<w:document {W}><w:body>{body}</w:body></w:document>"#);
            let bytes = docx_parts(&[("word/document.xml", &document)]);
            let markdown = crate::to_markdown_bytes(&bytes, crate::Format::Docx).unwrap();
            assert_eq!(markdown, expected, "body: {body}");
        }
    }

    #[test]
    fn numbered_heading_keeps_its_number() {
        // H1: a heading style with numbering shows its label and advances
        // the sequence.
        let document = format!(
            r#"<w:document {W}><w:body>
            <w:p><w:pPr><w:pStyle w:val="H1"/></w:pPr><w:r><w:t>Intro</w:t></w:r></w:p>
            <w:p><w:pPr><w:pStyle w:val="H1"/></w:pPr><w:r><w:t>Details</w:t></w:r></w:p>
            </w:body></w:document>"#
        );
        let styles = format!(
            r#"<w:styles {W}>
            <w:style w:type="paragraph" w:styleId="H1"><w:name w:val="heading 1"/>
                <w:pPr><w:numPr><w:numId w:val="1"/></w:numPr></w:pPr>
            </w:style></w:styles>"#
        );
        let numbering = format!(
            r#"<w:numbering {W}>
            <w:abstractNum w:abstractNumId="0">
                <w:lvl w:ilvl="0"><w:numFmt w:val="decimal"/><w:start w:val="1"/>
                    <w:lvlText w:val="%1."/><w:pStyle w:val="H1"/></w:lvl>
            </w:abstractNum>
            <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
            </w:numbering>"#
        );
        let bytes = docx_parts(&[
            ("word/document.xml", &document),
            ("word/styles.xml", &styles),
            ("word/numbering.xml", &numbering),
        ]);
        let doc = parse(&bytes).unwrap();
        let headings: Vec<String> = doc
            .blocks
            .iter()
            .filter_map(|b| match b {
                Block::Heading { content, .. } => {
                    Some(crate::model::inlines_to_plain_text(content))
                }
                _ => None,
            })
            .collect();
        assert_eq!(headings, vec!["1. Intro", "2. Details"]);
    }

    /// The plain text of each paragraph block, in order.
    fn plain_paragraphs(doc: &Document) -> Vec<String> {
        doc.blocks
            .iter()
            .filter_map(|b| match b {
                Block::Paragraph(inlines) => Some(crate::model::inlines_to_plain_text(inlines)),
                _ => None,
            })
            .collect()
    }

    fn body_doc(body: &str) -> Document {
        let document = format!(r#"<w:document {W}><w:body>{body}</w:body></w:document>"#);
        parse(&docx_parts(&[("word/document.xml", &document)])).unwrap()
    }

    #[test]
    fn a_phonetic_guide_keeps_its_base_text() {
        // markitai: the base text of a ruby is the sentence's own words; only
        // the reading above it is annotation.
        let doc = body_doc(
            r#"<w:p><w:r><w:t>日本</w:t></w:r>
            <w:r><w:ruby><w:rt><w:r><w:t>かんじ</w:t></w:r></w:rt>
                <w:rubyBase><w:r><w:t>漢字</w:t></w:r></w:rubyBase></w:ruby></w:r>
            <w:r><w:t>を読む</w:t></w:r></w:p>"#,
        );
        assert_eq!(plain_paragraphs(&doc), ["日本漢字を読む"]);
    }

    #[test]
    fn wordart_text_is_read_from_its_text_path() {
        // markitai: the words of VML WordArt are in the `string` attribute.
        let doc = body_doc(
            r##"<w:p><w:r><w:pict>
                <v:shape xmlns:v="urn:schemas-microsoft-com:vml" id="wordart" type="#_x0000_t136">
                    <v:textpath style="font-family:Impact" string="Grand Opening"/>
                </v:shape></w:pict></w:r></w:p>
            <w:p><w:r><w:t>Body.</w:t></w:r></w:p>"##,
        );
        assert_eq!(plain_paragraphs(&doc), ["Grand Opening", "Body."]);
    }

    #[test]
    fn a_non_breaking_hyphen_is_a_hyphen_and_a_soft_hyphen_is_nothing() {
        let doc = body_doc(
            r#"<w:p><w:r><w:t>e</w:t><w:noBreakHyphen/><w:t>mail and soft</w:t><w:softHyphen/>
            <w:t>ware</w:t></w:r></w:p>"#,
        );
        assert_eq!(plain_paragraphs(&doc), ["e-mail and software"]);
    }

    #[test]
    fn symbol_characters_read_as_their_unicode_mark() {
        let doc = body_doc(
            r#"<w:p><w:r><w:t>Done </w:t><w:sym w:font="Wingdings" w:char="F0FC"/>
            <w:t> alpha </w:t><w:sym w:font="Symbol" w:char="F061"/>
            <w:t> copyright </w:t><w:sym w:font="Arial" w:char="00A9"/>
            <w:t> unknown </w:t><w:sym w:font="Webdings" w:char="F0FC"/><w:t>.</w:t></w:r></w:p>"#,
        );
        assert_eq!(plain_paragraphs(&doc), ["Done ✓ alpha α copyright © unknown ."]);
    }

    // markitai: a VML picture's alt text.
    #[test]
    fn a_vml_pictures_alt_text_is_read() {
        let doc = body_doc(
            r##"<w:p><w:r><w:t xml:space="preserve">Logo: </w:t></w:r><w:r><w:pict
            xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
            <v:shape id="i1" type="#_x0000_t75" alt="Company logo" style="width:40pt;height:20pt">
            <v:imagedata r:id="rId9" o:title=""/></v:shape></w:pict></w:r></w:p>"##,
        );
        let [Block::Paragraph(inlines)] = &doc.blocks[..] else { panic!("{:?}", doc.blocks) };
        assert!(
            inlines.iter().any(|i| matches!(i, Inline::Image { alt, .. } if alt == "Company logo")),
            "{inlines:?}"
        );
    }

    // markitai: text set in a symbol font.
    #[test]
    fn text_in_a_symbol_font_reads_as_its_glyphs() {
        let styles = format!(
            r#"<w:styles {W}><w:style w:type="character" w:styleId="Greek">
            <w:rPr><w:rFonts w:ascii="Symbol" w:hAnsi="Symbol"/></w:rPr></w:style></w:styles>"#
        );
        let symbol = r#"<w:rPr><w:rFonts w:ascii="Symbol" w:hAnsi="Symbol"/></w:rPr>"#;
        let wingdings = r#"<w:rPr><w:rFonts w:ascii="Wingdings" w:hAnsi="Wingdings"/></w:rPr>"#;
        let document = format!(
            r#"<w:document {W}><w:body><w:p>
            <w:r><w:t xml:space="preserve">Typed </w:t></w:r><w:r>{symbol}<w:t>abg</w:t></w:r>
            <w:r><w:t xml:space="preserve">, stored </w:t></w:r><w:r>{symbol}<w:t>&#xF06D;</w:t></w:r>
            <w:r><w:t xml:space="preserve">m, styled </w:t></w:r><w:r><w:rPr><w:rStyle w:val="Greek"/></w:rPr><w:t>p</w:t></w:r>
            <w:r><w:t xml:space="preserve">, ticks </w:t></w:r><w:r>{wingdings}<w:t>&#xFC;&#xF0FC;</w:t></w:r>
            <w:r><w:t xml:space="preserve">, real </w:t></w:r><w:r>{symbol}<w:t>&#x3B1;</w:t></w:r>
            <w:r><w:t xml:space="preserve">, plain abg.</w:t></w:r></w:p></w:body></w:document>"#
        );
        let doc =
            parse(&docx_parts(&[("word/document.xml", &document), ("word/styles.xml", &styles)]))
                .unwrap();
        assert_eq!(
            plain_paragraphs(&doc),
            ["Typed αβγ, stored μm, styled π, ticks ✓✓, real α, plain abg."]
        );
    }

    // markitai: a legacy form field's check box and drop-down list.
    #[test]
    fn legacy_form_fields_show_their_state() {
        fn field(data: &str, instr: &str, result: Option<&str>) -> String {
            let result = result.map_or(String::new(), |text| {
                format!(r#"<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>{text}</w:t></w:r>"#)
            });
            format!(
                r#"<w:r><w:fldChar w:fldCharType="begin"><w:ffData>{data}</w:ffData></w:fldChar></w:r>
                <w:r><w:instrText xml:space="preserve"> {instr} </w:instrText></w:r>{result}
                <w:r><w:fldChar w:fldCharType="end"/></w:r>"#
            )
        }
        let paragraphs = [
            ("Typed:", field("<w:textInput/>", "FORMTEXT", Some("Jane"))),
            (
                "Checked:",
                field(
                    r#"<w:checkBox><w:default w:val="0"/><w:checked/></w:checkBox>"#,
                    "FORMCHECKBOX",
                    None,
                ),
            ),
            (
                "By default:",
                field(r#"<w:checkBox><w:default w:val="1"/></w:checkBox>"#, "FORMCHECKBOX", None),
            ),
            (
                "Cleared:",
                field(
                    r#"<w:checkBox><w:default w:val="1"/><w:checked w:val="0"/></w:checkBox>"#,
                    "FORMCHECKBOX",
                    None,
                ),
            ),
            ("Never set:", field("<w:checkBox><w:sizeAuto/></w:checkBox>", "FORMCHECKBOX", None)),
            (
                "Chosen:",
                field(
                    r#"<w:ddList><w:result w:val="1"/><w:listEntry w:val="Red"/><w:listEntry w:val="Green"/></w:ddList>"#,
                    "FORMDROPDOWN",
                    None,
                ),
            ),
            (
                "First:",
                field(
                    r#"<w:ddList><w:listEntry w:val="Small"/><w:listEntry w:val="Large"/></w:ddList>"#,
                    "FORMDROPDOWN",
                    None,
                ),
            ),
            (
                "Out of range:",
                field(
                    r#"<w:ddList><w:result w:val="9"/><w:default w:val="1"/><w:listEntry w:val="A"/><w:listEntry w:val="B"/></w:ddList>"#,
                    "FORMDROPDOWN",
                    None,
                ),
            ),
            (
                "Written result:",
                field(
                    r#"<w:ddList><w:result w:val="0"/><w:listEntry w:val="One"/><w:listEntry w:val="Two"/></w:ddList>"#,
                    "FORMDROPDOWN",
                    Some("Two"),
                ),
            ),
            ("Not a form field:", field("<w:checkBox><w:checked/></w:checkBox>", "PAGE", None)),
        ];
        let body: String = paragraphs
            .iter()
            .map(|(label, field)| format!(r#"<w:p><w:r><w:t xml:space="preserve">{label} </w:t></w:r>{field}<w:r><w:t>.</w:t></w:r></w:p>"#))
            .collect();
        let hidden = field("<w:checkBox><w:checked/></w:checkBox>", "FORMCHECKBOX", None).replacen(
            "<w:r><w:fldChar",
            "<w:r><w:rPr><w:vanish/></w:rPr><w:fldChar",
            1,
        );
        let body = format!(
            r#"{body}<w:p><w:r><w:t xml:space="preserve">Hidden: </w:t></w:r>{hidden}</w:p>"#
        );
        assert_eq!(
            plain_paragraphs(&body_doc(&body)),
            [
                "Typed: Jane.",
                "Checked: ☒.",
                "By default: ☒.",
                "Cleared: ☐.",
                "Never set: ☐.",
                "Chosen: Green.",
                "First: Small.",
                "Out of range: B.",
                "Written result: Two.",
                "Not a form field: .",
                "Hidden: ",
            ]
        );
    }

    #[test]
    fn hidden_text_is_left_out_whatever_hides_it() {
        // Direct formatting, a character style and a paragraph style hide a
        // run; an explicit off value shows it again; a hidden run's field
        // marks still balance the fields around it.
        let styles = format!(
            r#"<w:styles {W}>
            <w:style w:type="character" w:styleId="Secret"><w:rPr><w:vanish/></w:rPr></w:style>
            <w:style w:type="paragraph" w:styleId="Aside"><w:rPr><w:vanish/></w:rPr></w:style>
            </w:styles>"#
        );
        let document = format!(
            r#"<w:document {W}><w:body>
            <w:p><w:r><w:t xml:space="preserve">Shown one. </w:t></w:r>
                <w:r><w:rPr><w:vanish/></w:rPr><w:t>Direct hidden. </w:t></w:r>
                <w:r><w:rPr><w:rStyle w:val="Secret"/></w:rPr><w:t>Styled hidden. </w:t></w:r>
                <w:r><w:rPr><w:rStyle w:val="Secret"/><w:vanish w:val="0"/></w:rPr><w:t xml:space="preserve">Shown again. </w:t></w:r>
                <w:r><w:rPr><w:vanish/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>
                <w:r><w:instrText> PAGE </w:instrText></w:r>
                <w:r><w:fldChar w:fldCharType="separate"/></w:r>
                <w:r><w:t>7</w:t></w:r>
                <w:r><w:fldChar w:fldCharType="end"/></w:r>
                <w:r><w:t xml:space="preserve"> Tail.</w:t></w:r></w:p>
            <w:p><w:pPr><w:pStyle w:val="Aside"/></w:pPr><w:r><w:t>Paragraph hidden.</w:t></w:r></w:p>
            <w:p><w:pPr><w:pStyle w:val="Aside"/></w:pPr>
                <w:r><w:rPr><w:vanish w:val="0"/></w:rPr><w:t>Overridden paragraph.</w:t></w:r></w:p>
            </w:body></w:document>"#
        );
        let doc =
            parse(&docx_parts(&[("word/document.xml", &document), ("word/styles.xml", &styles)]))
                .unwrap();
        assert_eq!(
            plain_paragraphs(&doc),
            ["Shown one. Shown again. 7 Tail.", "Overridden paragraph."]
        );
    }

    #[test]
    fn raised_and_lowered_runs_are_written_in_unicode_script_forms() {
        // markitai: the exponent of "10⁻³" and the "2" of "H₂O" are content;
        // a run with no complete form ("1st") keeps its baseline text.
        let styles = format!(
            r#"<w:styles {W}>
            <w:style w:type="character" w:styleId="Exp"><w:rPr><w:vertAlign w:val="superscript"/></w:rPr></w:style>
            </w:styles>"#
        );
        let document = format!(
            r#"<w:document {W}><w:body>
            <w:p><w:r><w:t>H</w:t></w:r><w:r><w:rPr><w:vertAlign w:val="subscript"/></w:rPr><w:t>2</w:t></w:r>
                <w:r><w:t xml:space="preserve">O and 10</w:t></w:r>
                <w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:t>-3</w:t></w:r>
                <w:r><w:t xml:space="preserve"> and 5</w:t></w:r>
                <w:r><w:rPr><w:rStyle w:val="Exp"/></w:rPr><w:t>2</w:t></w:r>
                <w:r><w:rPr><w:rStyle w:val="Exp"/><w:vertAlign w:val="baseline"/></w:rPr><w:t xml:space="preserve"> flat</w:t></w:r>
                <w:r><w:t xml:space="preserve"> and 1</w:t></w:r>
                <w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:t>st</w:t></w:r></w:p>
            </w:body></w:document>"#
        );
        let doc =
            parse(&docx_parts(&[("word/document.xml", &document), ("word/styles.xml", &styles)]))
                .unwrap();
        assert_eq!(plain_paragraphs(&doc), ["H₂O and 10⁻³ and 5² flat and 1st"]);
    }

    #[test]
    fn a_row_whose_deletion_is_tracked_leaves_the_table() {
        let doc = body_doc(
            r#"<w:tbl><w:tblGrid><w:gridCol w:w="1000"/></w:tblGrid>
            <w:tr><w:tc><w:p><w:r><w:t>Kept</w:t></w:r></w:p></w:tc></w:tr>
            <w:tr><w:trPr><w:del w:id="1" w:author="A"/></w:trPr>
                <w:tc><w:p><w:del w:id="2" w:author="A"><w:r><w:delText>Removed</w:delText></w:r></w:del></w:p></w:tc></w:tr>
            <w:tr><w:trPr><w:ins w:id="3" w:author="A"/></w:trPr>
                <w:tc><w:p><w:ins w:id="4" w:author="A"><w:r><w:t>Added</w:t></w:r></w:ins></w:p></w:tc></w:tr>
            </w:tbl>"#,
        );
        let [Block::Table(table)] = &doc.blocks[..] else { panic!("{:?}", doc.blocks) };
        assert_eq!(table.grid.len(), 2, "{:?}", table.grid);
    }

    #[test]
    fn only_declared_header_rows_make_a_header() {
        // A table marks the rows repeated at the top of each page; the types
        // of the columns below say nothing about a row being a header.
        let cell = |text: &str| format!("<w:tc><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:tc>");
        let row = |header: bool, a: &str, b: &str| {
            let props = if header { "<w:trPr><w:tblHeader/></w:trPr>" } else { "" };
            format!("<w:tr>{props}{}{}</w:tr>", cell(a), cell(b))
        };
        let table = |first_is_header: bool| {
            format!(
                "<w:tbl><w:tblGrid><w:gridCol w:w=\"1000\"/><w:gridCol w:w=\"1000\"/></w:tblGrid>{}{}{}</w:tbl>",
                row(first_is_header, "Name", "Count"),
                row(false, "Alpha", "1"),
                row(false, "Beta", "2"),
            )
        };
        let header_rows = |body: String| match &body_doc(&body).blocks[..] {
            [Block::Table(table)] => table.header_rows,
            other => panic!("{other:?}"),
        };
        assert_eq!(header_rows(table(false)), 0);
        assert_eq!(header_rows(table(true)), 1);
    }

    #[test]
    fn cjk_and_enclosed_numbering_formats_count_in_their_own_characters() {
        let numbering = format!(
            r#"<w:numbering {W}>
            <w:abstractNum w:abstractNumId="0">
                <w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="chineseCounting"/><w:lvlText w:val="%1、"/></w:lvl>
                <w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="chineseCounting"/><w:lvlText w:val="（%2）"/></w:lvl>
                <w:lvl w:ilvl="2"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1.%2.%3"/></w:lvl>
            </w:abstractNum>
            <w:abstractNum w:abstractNumId="1">
                <w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimalEnclosedCircle"/><w:lvlText w:val="%1"/></w:lvl>
            </w:abstractNum>
            <w:abstractNum w:abstractNumId="2">
                <w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="chineseCounting"/><w:isLgl/><w:lvlText w:val="第%1章"/></w:lvl>
            </w:abstractNum>
            <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
            <w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>
            <w:num w:numId="3"><w:abstractNumId w:val="2"/></w:num>
            </w:numbering>"#
        );
        let item = |num_id: u32, level: u32, text: &str| {
            format!(
                r#"<w:p><w:pPr><w:numPr><w:ilvl w:val="{level}"/><w:numId w:val="{num_id}"/></w:numPr></w:pPr>
                <w:r><w:t>{text}</w:t></w:r></w:p>"#
            )
        };
        let document = format!(
            r#"<w:document {W}><w:body>{}{}{}{}{}{}{}{}</w:body></w:document>"#,
            item(1, 0, "甲"),
            item(1, 1, "乙"),
            item(1, 1, "丙"),
            item(1, 2, "丁"),
            item(1, 0, "戊"),
            item(2, 0, "圈"),
            item(2, 0, "圈二"),
            item(3, 0, "法律"),
        );
        let doc = parse(&docx_parts(&[
            ("word/document.xml", &document),
            ("word/numbering.xml", &numbering),
        ]))
        .unwrap();
        fn labels(blocks: &[Block], out: &mut Vec<String>) {
            for block in blocks {
                if let Block::List(list) = block {
                    for item in &list.items {
                        out.push(item.marker_label.clone().unwrap_or_else(|| "-".into()));
                        labels(&item.blocks, out);
                    }
                }
            }
        }
        let mut found = Vec::new();
        labels(&doc.blocks, &mut found);
        assert_eq!(
            found,
            ["一、", "（一）", "（二）", "一.二.1", "二、", "①", "②", "第1章"],
            "{:?}",
            doc.blocks
        );
    }

    /// markitai: what each top-level block is: `h<level>`, `p`, `list` or
    /// `table`.
    fn shapes(blocks: &[Block]) -> Vec<String> {
        blocks
            .iter()
            .map(|block| match block {
                Block::Heading { level, .. } => format!("h{level}"),
                Block::Paragraph(_) => "p".into(),
                Block::List(_) => "list".into(),
                Block::Table(_) => "table".into(),
                other => format!("{other:?}"),
            })
            .collect()
    }

    /// markitai: a paragraph of one run, bold or not, at a size in
    /// half-points (none written for `0`).
    fn sized(text: &str, bold: bool, size: u32) -> String {
        let bold = if bold { "<w:b/>" } else { "" };
        let size = if size > 0 { format!(r#"<w:sz w:val="{size}"/>"#) } else { String::new() };
        format!(r#"<w:p><w:r><w:rPr>{bold}{size}</w:rPr><w:t>{text}</w:t></w:r></w:p>"#)
    }

    const PROSE: &str = "Body text long enough to outweigh every heading together, \
        as the text of a document does: a few sentences set at one ordinary size.";

    // markitai: headings set by hand (see `crate::shared::visual`).
    #[test]
    fn bold_paragraphs_set_above_the_body_size_are_headings_ranked_by_size() {
        let numbered = r#"<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>
            <w:r><w:rPr><w:b/><w:sz w:val="36"/></w:rPr><w:t>Listed</w:t></w:r></w:p>"#;
        let table = format!("<w:tbl><w:tr><w:tc>{}</w:tc></w:tr></w:tbl>", sized("Cell", true, 36));
        // A short line first: a cell's or a list's paragraph taken for one
        // of the body's would land on it.
        let body = [
            sized("By the authors", false, 24),
            sized("The Title", true, 48),
            sized(PROSE, false, 24),
            sized("A Section", true, 36),
            sized(PROSE, false, 24),
            numbered.to_string(),
            table,
            sized("let x = 1", false, 26),
            sized("A Subsection", true, 28),
            sized(PROSE, false, 24),
            sized("Bold at the body size", true, 24),
        ]
        .concat();
        let document = format!(r#"<w:document {W}><w:body>{body}</w:body></w:document>"#);
        let doc = parse(&docx_parts(&[
            ("word/document.xml", &document),
            ("word/numbering.xml", &numbering_with_start("1")),
        ]))
        .unwrap();
        assert_eq!(
            shapes(&doc.blocks),
            ["p", "h1", "p", "h2", "p", "list", "table", "p", "h3", "p", "p"],
            "{:?}",
            doc.blocks
        );
        let Block::Heading { content, .. } = &doc.blocks[1] else { unreachable!() };
        assert!(
            matches!(&content[..], [Inline::Text { text, style }] if text == "The Title" && !style.bold),
            "{content:?}"
        );
    }

    // markitai: sizes come from styles and docDefaults as well as runs.
    #[test]
    fn style_sizes_count_and_a_styled_heading_turns_the_guess_off() {
        let styles = format!(
            r#"<w:styles {W}>
            <w:docDefaults><w:rPrDefault><w:rPr><w:sz w:val="22"/></w:rPr></w:rPrDefault></w:docDefaults>
            <w:style w:type="paragraph" w:styleId="Big"><w:name w:val="Big Bold"/>
                <w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
            <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>
            </w:styles>"#
        );
        let big =
            r#"<w:p><w:pPr><w:pStyle w:val="Big"/></w:pPr><w:r><w:t>Styled Big</w:t></w:r></w:p>"#;
        let body = [big.to_string(), sized(PROSE, false, 0), sized(PROSE, false, 0)].concat();
        let read = |body: &str| {
            let document = format!(r#"<w:document {W}><w:body>{body}</w:body></w:document>"#);
            parse(&docx_parts(&[("word/document.xml", &document), ("word/styles.xml", &styles)]))
                .unwrap()
                .blocks
        };
        assert_eq!(shapes(&read(&body)), ["h1", "p", "p"]);
        let heading =
            r#"<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Real</w:t></w:r></w:p>"#;
        assert_eq!(shapes(&read(&format!("{body}{heading}"))), ["p", "p", "p", "h1"]);
    }

    #[test]
    fn columns_set_with_tab_stops_are_a_table_and_a_code_box_is_code() {
        // markitai: the stops of a paragraph style, inherited through
        // `basedOn`, and a stop a paragraph clears; a table of contents
        // style; a one-cell table of code; a tab in a line of code.
        let styles = format!(
            r#"<w:styles {W}>
            <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
            <w:style w:type="paragraph" w:styleId="Stops"><w:name w:val="Stops"/>
              <w:pPr><w:tabs><w:tab w:val="left" w:pos="2880"/></w:tabs></w:pPr></w:style>
            <w:style w:type="paragraph" w:styleId="Cols"><w:name w:val="Columns"/>
              <w:basedOn w:val="Stops"/>
              <w:pPr><w:tabs><w:tab w:val="end" w:pos="5760"/></w:tabs></w:pPr></w:style>
            <w:style w:type="paragraph" w:styleId="TOC1"><w:name w:val="toc 1"/></w:style>
            </w:styles>"#
        );
        let run = |text: &str| format!(r#"<w:r><w:t xml:space="preserve">{text}</w:t></w:r>"#);
        let row = |ppr: &str, cells: &[&str]| {
            let runs: Vec<String> = cells.iter().map(|cell| run(cell)).collect();
            format!("<w:p><w:pPr>{ppr}</w:pPr>{}</w:p>", runs.join("<w:r><w:tab/></w:r>"))
        };
        let menlo = r#"<w:rPr><w:rFonts w:ascii="Menlo"/></w:rPr>"#;
        let code = |text: &str| format!(r#"<w:p><w:r>{menlo}<w:t>{text}</w:t></w:r></w:p>"#);
        let cols = r#"<w:pStyle w:val="Cols"/>"#;
        let cleared =
            r#"<w:pStyle w:val="Cols"/><w:tabs><w:tab w:val="clear" w:pos="5760"/></w:tabs>"#;
        let inherited =
            r#"<w:pStyle w:val="Cols"/><w:tabs><w:tab w:val="clear" w:pos="2880"/></w:tabs>"#;
        let toc = r#"<w:pStyle w:val="TOC1"/>"#;
        let body = [
            row(cols, &["Item", "Qty", "Price"]),
            row(cols, &["Apple", "3", "1.20"]),
            row(cols, &["Pear", "12", "0.50"]),
            row(inherited, &["Fig", "1", "0.10"]),
            row(cleared, &["Plum", "7", "0.20"]),
            row(toc, &["1", "Introduction", "Page one"]),
            row(toc, &["2", "Methods", "Page two"]),
            row(toc, &["3", "Results", "Page three"]),
            format!(
                "<w:tbl><w:tr><w:tc>{}{}</w:tc></w:tr></w:tbl>",
                code("make"),
                code("make install")
            ),
            format!(r#"<w:p><w:r>{menlo}<w:t>a</w:t><w:tab/><w:t>b</w:t></w:r></w:p>"#),
            row("", &["And the prose of the document goes on in its own face for a while."]),
        ]
        .concat();
        let document = format!(r#"<w:document {W}><w:body>{body}</w:body></w:document>"#);
        let doc =
            parse(&docx_parts(&[("word/document.xml", &document), ("word/styles.xml", &styles)]))
                .unwrap();
        assert_eq!(
            crate::shared::code::describe(&doc.blocks),
            [
                "table:p:Item|p:Qty|p:Price/p:Apple|p:3|p:1.20/p:Pear|p:12|p:0.50",
                "p:Fig 1 0.10",
                "p:Plum 7 0.20",
                "p:1 Introduction Page one",
                "p:2 Methods Page two",
                "p:3 Results Page three",
                "code:make\nmake install",
                "code:a b",
                "p:And the prose of the document goes on in its own face for a while.",
            ]
        );
    }

    #[test]
    fn indents_set_the_levels_of_a_list_typed_by_hand() {
        // markitai: textutil's `w:first-line`, `w:hanging`, character units
        // and a style's indent through `basedOn` each place the bullet; a
        // contents style lists pages, and a bullet line in Menlo is an item.
        let styles = format!(
            r#"<w:styles {W}>
            <w:style w:type="paragraph" w:styleId="Deep"><w:name w:val="Deep"/>
              <w:pPr><w:ind w:left="1440" w:hanging="360"/></w:pPr></w:style>
            <w:style w:type="paragraph" w:styleId="Deeper"><w:name w:val="Deeper"/>
              <w:basedOn w:val="Deep"/></w:style>
            <w:style w:type="paragraph" w:styleId="TOC1"><w:name w:val="toc 1"/></w:style>
            </w:styles>"#
        );
        let item = |ppr: &str, lead: &str, text: &str| {
            format!(
                r#"<w:p><w:pPr>{ppr}</w:pPr><w:r>{lead}<w:t>•</w:t><w:tab/><w:t>{text}</w:t></w:r></w:p>"#
            )
        };
        let read = |body: &[String]| {
            let body = body.concat();
            let document = format!(r#"<w:document {W}><w:body>{body}</w:body></w:document>"#);
            let parts = [("word/document.xml", document.as_str()), ("word/styles.xml", &styles)];
            crate::shared::code::describe(&parse(&docx_parts(&parts)).unwrap().blocks)
        };
        let textutil = r#"<w:ind w:left="720" w:first-line="-720"/>"#;
        assert_eq!(
            read(&[item("", "<w:tab/>", "a"), item(textutil, "<w:tab/>", "b")]),
            ["list:p:a|p:b"]
        );
        let hanging = r#"<w:ind w:left="360" w:hanging="360"/>"#;
        assert_eq!(read(&[item("", "", "a"), item(hanging, "", "b")]), ["list:p:a|p:b"]);
        let chars = r#"<w:ind w:leftChars="300" w:left="0"/>"#;
        assert_eq!(read(&[item("", "", "a"), item(chars, "", "b")]), ["list:p:a;list:p:b"]);
        let style = r#"<w:pStyle w:val="Deeper"/>"#;
        assert_eq!(read(&[item("", "", "a"), item(style, "", "b")]), ["list:p:a;list:p:b"]);
        let toc = r#"<w:pStyle w:val="TOC1"/>"#;
        assert_eq!(read(&[item(toc, "", "a"), item(toc, "", "b")]), ["p:• a", "p:• b"]);
        let menlo = r#"<w:p><w:r><w:rPr><w:rFonts w:ascii="Menlo"/></w:rPr><w:t>•</w:t><w:tab/>
            <w:t>npm test</w:t></w:r></w:p>"#;
        let prose =
            "<w:p><w:r><w:t>The prose of the document runs on for long enough.</w:t></w:r></w:p>";
        assert_eq!(
            read(&[menlo.to_string(), prose.to_string()]),
            ["list:p:`npm test`", "p:The prose of the document runs on for long enough."]
        );
    }

    /// markitai: a document of `body` with lists numbered as Word, pandoc
    /// and a `none` level number them, and Word's List Paragraph style.
    fn listed(body: &str) -> Vec<String> {
        let numbering = format!(
            r#"<w:numbering {W}>
            <w:abstractNum w:abstractNumId="0">
              <w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/>
                <w:lvlText w:val="%1."/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl>
              <w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="lowerLetter"/>
                <w:lvlText w:val="%2."/><w:pPr><w:ind w:left="1440" w:hanging="360"/></w:pPr></w:lvl>
            </w:abstractNum>
            <w:abstractNum w:abstractNumId="9">
              <w:lvl w:ilvl="0"><w:numFmt w:val="bullet"/><w:lvlText w:val=" "/>
                <w:pPr><w:ind w:left="720" w:hanging="480"/></w:pPr></w:lvl>
              <w:lvl w:ilvl="1"><w:numFmt w:val="bullet"/><w:lvlText w:val=" "/>
                <w:pPr><w:ind w:left="1440" w:hanging="480"/></w:pPr></w:lvl>
            </w:abstractNum>
            <w:abstractNum w:abstractNumId="5">
              <w:lvl w:ilvl="0"><w:numFmt w:val="none"/><w:lvlText w:val=""/>
                <w:pPr><w:ind w:left="720"/></w:pPr></w:lvl>
            </w:abstractNum>
            <w:abstractNum w:abstractNumId="6">
              <w:lvl w:ilvl="0"><w:numFmt w:val="bullet"/><w:lvlText w:val=""/>
                <w:lvlPicBulletId w:val="0"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl>
            </w:abstractNum>
            <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
            <w:num w:numId="1000"><w:abstractNumId w:val="9"/></w:num>
            <w:num w:numId="5"><w:abstractNumId w:val="5"/></w:num>
            <w:num w:numId="6"><w:abstractNumId w:val="6"/></w:num>
            </w:numbering>"#
        );
        let styles = format!(
            r#"<w:styles {W}>
            <w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/>
              <w:pPr><w:ind w:left="720"/></w:pPr></w:style>
            </w:styles>"#
        );
        let document = format!(r#"<w:document {W}><w:body>{body}</w:body></w:document>"#);
        let doc = parse(&docx_parts(&[
            ("word/document.xml", &document),
            ("word/numbering.xml", &numbering),
            ("word/styles.xml", &styles),
        ]))
        .unwrap();
        let mut shown = crate::shared::code::describe(&doc.blocks);
        for block in &doc.blocks {
            if let Block::List(list) = block {
                shown.push(format!("start {}", list.start));
            }
        }
        shown
    }

    /// markitai: a paragraph numbered `numid` at `level` (none: no number),
    /// with `ppr` added to its properties.
    fn para(numid: Option<(u32, u32)>, ppr: &str, text: &str) -> String {
        let num = numid.map_or(String::new(), |(id, level)| {
            format!(r#"<w:numPr><w:ilvl w:val="{level}"/><w:numId w:val="{id}"/></w:numPr>"#)
        });
        format!(r#"<w:p><w:pPr>{ppr}{num}</w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>"#)
    }

    #[test]
    fn a_paragraph_set_in_to_an_items_text_continues_the_item() {
        let list = r#"<w:pStyle w:val="ListParagraph"/>"#;
        let body = [
            para(None, "", "Before the list."),
            para(Some((1, 0)), list, "One."),
            // Enter, then Backspace in Word: the style's indent, no number.
            para(None, list, "More of one."),
            para(Some((1, 0)), list, "Two."),
            para(Some((1, 1)), list, "Two a."),
            para(None, r#"<w:ind w:left="1440"/>"#, "More of two a."),
            // Back at the outer item's text, after its nested list.
            para(None, r#"<w:ind w:left="720"/>"#, "More of two."),
            // The numbering level alone sets this item's text indent.
            para(Some((1, 0)), "", "Three."),
            para(None, r#"<w:ind w:left="720"/>"#, "More of three."),
            para(None, "", "Body text after the list."),
            para(None, r#"<w:ind w:left="720"/>"#, "An indented note after the body."),
        ]
        .concat();
        assert_eq!(
            listed(&body),
            [
                "p:Before the list.",
                "list:p:One.;p:More of one.|p:Two.;list:p:Two a.;p:More of two a.;p:More of two.|p:Three.;p:More of three.",
                "p:Body text after the list.",
                "p:An indented note after the body.",
                "start 1",
            ]
        );
    }

    #[test]
    fn empty_paragraphs_keep_an_item_open_only_for_its_continuation() {
        let list = r#"<w:pStyle w:val="ListParagraph"/>"#;
        let body = [
            para(Some((1, 0)), list, "One."),
            "<w:p/>".to_string(),
            para(None, list, "More of one."),
            "<w:p/><w:p/>".to_string(),
            // An empty paragraph between items closes the list as before;
            // the numbering goes on.
            para(Some((1, 0)), list, "Two."),
            "<w:p/>".to_string(),
            para(None, "", "Body text."),
        ]
        .concat();
        assert_eq!(
            listed(&body),
            ["list:p:One.;p:More of one.", "list:p:Two.", "p:Body text.", "start 1", "start 2"]
        );
    }

    #[test]
    fn a_number_that_shows_nothing_continues_the_item() {
        // pandoc numbers an item's later paragraphs with a bullet of one
        // space, and a `none` level shows no number either.
        let body = [
            para(Some((1, 0)), "", "One."),
            para(Some((1000, 0)), "", "More of one."),
            para(Some((1, 1)), "", "One a."),
            para(Some((1000, 1)), "", "More of one a."),
            para(Some((5, 0)), "", "More of one, at a none level."),
            para(Some((1, 0)), "", "Two."),
            para(None, "", "Body."),
            // Without a list before them they are paragraphs, not items.
            para(Some((1000, 0)), "", "Unmarked, alone."),
            para(Some((5, 0)), "", "None, alone."),
            // A picture bullet shows a picture.
            para(Some((6, 0)), "", "Picture bullet."),
        ]
        .concat();
        assert_eq!(
            listed(&body),
            [
                "list:p:One.;p:More of one.;list:p:One a.;p:More of one a.;p:More of one, at a none level.|p:Two.",
                "p:Body.",
                "p:Unmarked, alone.",
                "p:None, alone.",
                "list:p:Picture bullet.",
                "start 1",
                "start 1",
            ]
        );
    }

    #[test]
    fn body_text_at_the_lists_own_indent_continues_nothing() {
        // A document whose body is set in as far as its lists' text: the
        // paragraph after the list is body text again.
        let body_ind = r#"<w:ind w:left="720"/>"#;
        let item = r#"<w:ind w:left="720" w:hanging="360"/>"#;
        let body = [
            para(None, body_ind, "Body set in."),
            para(Some((1, 0)), item, "One."),
            para(None, body_ind, "Body again."),
            // Nor does a list whose text starts at the margin.
            para(None, "", "Body at the margin."),
            para(Some((1, 0)), r#"<w:ind w:left="0" w:hanging="0"/>"#, "Two."),
            para(None, "", "Margin again."),
        ]
        .concat();
        assert_eq!(
            listed(&body),
            [
                "p:Body set in.",
                "list:p:One.",
                "p:Body again.",
                "p:Body at the margin.",
                "list:p:Two.",
                "p:Margin again.",
                "start 1",
                "start 2",
            ]
        );
    }

    /// markitai: a package of these parts, bytes as they are.
    fn package(parts: &[(&str, &[u8])]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default();
        for (name, body) in parts {
            w.start_file(*name, opts).unwrap();
            w.write_all(body).unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    const R: &str =
        r#"xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships""#;
    const WORD_TYPE: &str =
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml";

    /// markitai: a Word package whose body is `body`, embedding `chunks`
    /// (relationship id, part name under `word/`, content type or none,
    /// bytes), with these footnotes.
    fn with_chunks(
        body: &str,
        chunks: &[(&str, &str, Option<&str>, &[u8])],
        notes: &str,
    ) -> Vec<u8> {
        let document = format!(r#"<w:document {W} {R}><w:body>{body}</w:body></w:document>"#);
        let mut rels = String::from(
            r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#,
        );
        let mut types = String::from(
            r#"<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
            <Default Extension="txt" ContentType="text/plain"/>"#,
        );
        for (id, name, content_type, _) in chunks {
            rels.push_str(&format!(
                r#"<Relationship Id="{id}" Target="{name}"
                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/aFChunk"/>"#
            ));
            if let Some(content_type) = content_type {
                types.push_str(&format!(
                    r#"<Override PartName="/word/{name}" ContentType="{content_type}"/>"#
                ));
            }
        }
        rels.push_str("</Relationships>");
        types.push_str("</Types>");
        let footnotes = format!(
            r#"<w:footnotes {W}><w:footnote w:id="1"><w:p><w:r><w:t>{notes}</w:t></w:r></w:p></w:footnote></w:footnotes>"#
        );
        let names: Vec<String> =
            chunks.iter().map(|(_, name, _, _)| format!("word/{name}")).collect();
        let mut parts: Vec<(&str, &[u8])> = vec![
            ("[Content_Types].xml", types.as_bytes()),
            ("word/document.xml", document.as_bytes()),
            ("word/_rels/document.xml.rels", rels.as_bytes()),
        ];
        if !notes.is_empty() {
            parts.push(("word/footnotes.xml", footnotes.as_bytes()));
        }
        for (name, (_, _, _, bytes)) in names.iter().zip(chunks) {
            parts.push((name, bytes));
        }
        package(&parts)
    }

    #[test]
    fn embedded_parts_are_read_where_they_stand() {
        // markitai: `w:altChunk` parts in HTML, RTF, plain text and Word,
        // nested Word documents down to the depth limit; an MHT part and a
        // missing one add warnings.
        let text = |words: &str| format!("<w:p><w:r><w:t>{words}</w:t></w:r></w:p>");
        let chunk = |id: &str| format!(r#"<w:altChunk r:id="{id}"/>"#);
        let level4 = with_chunks(&text("Level four"), &[], "");
        let level3 = with_chunks(
            &(text("Level three") + &chunk("d")),
            &[("d", "d.docx", None, &level4)],
            "",
        );
        let level2 =
            with_chunks(&(text("Level two") + &chunk("d")), &[("d", "d.docx", None, &level3)], "");
        let noted = r#"<w:p><w:r><w:t>From the nested document</w:t></w:r>
            <w:r><w:footnoteReference w:id="1"/></w:r></w:p>"#;
        let level1 = with_chunks(
            &(noted.to_string() + &chunk("d")),
            &[("d", "afchunk.docx", Some(WORD_TYPE), &level2)],
            "A nested note.",
        );
        let html = br#"<html><head><style>.b{font-weight:bold}</style></head><body>
            <h2>From HTML</h2><p>Some <span class=b>bold</span> text<br>and a <a href="https://e.com">link</a>.
            <p>Second &amp; last<ul><li>one<li>two</ul><img src="data:image/png;base64,iVBORw0KGgo=" alt="dot">
            </body></html>"#;
        // A web archive as html-docx-js writes it: quoted-printable HTML and
        // a base64 picture it shows by location.
        let mht = [
            "MIME-Version: 1.0\r\nContent-Type: multipart/related;\r\n    type=\"text/html\";\r\n",
            "    boundary=\"----=mhtDocumentPart\"\r\n\r\n\r\n------=mhtDocumentPart\r\n",
            "Content-Type: text/html;\r\n    charset=\"utf-8\"\r\nContent-Transfer-Encoding: quoted-printable\r\n",
            "Content-Location: file:///C:/fake/document.html\r\n\r\n",
            "<html><body><p>From the web archive, caf=C3=A9 <img src=3D\"file:///C:/fake/image0.png\">",
            "</p><p>A line split by a soft=\r\n line break.</p></body></html>\r\n\r\n",
            "------=mhtDocumentPart\r\nContent-Type: image/png\r\nContent-Transfer-Encoding: base64\r\n",
            "Content-Location: file:///C:/fake/image0.png\r\n\r\niVBORw0KGgo=\r\n------=mhtDocumentPart--\r\n",
        ]
        .concat();
        let body = [
            text("Before."),
            chunk("h"),
            chunk("r"),
            chunk("t"),
            chunk("w"),
            chunk("m"),
            chunk("x"),
            chunk("gone"),
            format!("<w:tbl><w:tr><w:tc>{}</w:tc></w:tr></w:tbl>", chunk("t")),
            text("After."),
        ]
        .concat();
        let mut parts: Vec<(&str, &str, Option<&str>, &[u8])> = vec![
            ("h", "afchunk1.html", Some("application/xhtml+xml"), html),
            ("r", "afchunk2.rtf", Some("application/rtf"), br"{\rtf1\ansi From {\b RTF}.\par}"),
            ("t", "afchunk3.txt", None, b"Line one\r\n\r\nLine two\n"),
            ("w", "afchunk4.docx", Some(WORD_TYPE), &level1),
            ("m", "afchunk5.mht", Some("message/rfc822"), mht.as_bytes()),
            ("x", "afchunk6.xml", Some("application/xml"), b"<w:document/>"),
        ];
        let bytes = with_chunks(&body, &parts, "");
        let doc = parse(&bytes).unwrap();
        assert_eq!(
            crate::shared::code::describe(&doc.blocks),
            [
                "p:Before.",
                "h2:From HTML",
                "p:Some bold text\nand a link. ",
                "p:Second & last",
                "list:p:one|p:two",
                "p: ",
                "p:From RTF.",
                "p:Line one",
                "p:Line two",
                "p:From the nested document",
                "p:Level two",
                "p:Level three",
                "p:From the web archive, café ",
                "p:A line split by a soft line break.",
                "table:p:Line one;p:Line two",
                "p:After.",
            ]
        );
        let [note] = &doc.notes[..] else { panic!("{:?}", doc.notes) };
        assert_eq!(note.id, "chunk4-fn1");
        assert_eq!(crate::shared::code::describe(&note.blocks), ["p:A nested note."]);
        assert!(format!("{:?}", doc.blocks).contains("NoteRef(\"chunk4-fn1\")"));
        let [data, archived] = &doc.assets[..] else { panic!("{:?}", doc.assets) };
        assert_eq!((data.media_type.as_str(), data.bytes.len()), ("image/png", 8));
        assert_eq!((archived.media_type.as_str(), archived.bytes.len()), ("image/png", 8));
        assert!(format!("{:?}", doc.blocks).contains("Asset(AssetId(1))"));
        assert_eq!(
            doc.warnings,
            [
                "1 embedded part of the document (w:altChunk) is missing from the package; its \
                 content is not in the Markdown.",
                "1 embedded part of the document (w:altChunk) is in a format that is not converted \
                 (application/xml); its content is not in the Markdown.",
                "1 embedded part of the document (w:altChunk) is a Word document nested more than \
                 3 documents deep; its content is not in the Markdown.",
            ]
        );
        // A part that is not what its type says is unreadable, not fatal.
        parts[1].3 = b"not rtf at all";
        parts[3].3 = b"PK\x03\x04 not a zip";
        let doc = parse(&with_chunks(&body, &parts, "")).unwrap();
        assert!(
            doc.warnings[2].starts_with(
                "2 embedded parts of the document (w:altChunk) could not be read; their"
            ),
            "{:?}",
            doc.warnings
        );
        let doc = parse(&with_chunks(&(chunk("a") + &chunk("b")), &[], "")).unwrap();
        assert_eq!(
            doc.warnings,
            ["2 embedded parts of the document (w:altChunk) are missing from the package; their \
              content is not in the Markdown."]
        );
    }

    #[test]
    fn an_embedded_document_reads_against_the_outer_budget() {
        // markitai: what an embedded Word document decompresses counts
        // towards the outer package's total, on top of what it had read.
        let long = format!("<w:p><w:r><w:t>{}</w:t></w:r></w:p>", "word ".repeat(20_000));
        let inner = with_chunks(&long, &[], "");
        let (_, alone) = parse_at(&inner, 0, 0).unwrap();
        assert!(alone > 100_000 && inner.len() < 10_000, "{alone} {}", inner.len());
        let outer = with_chunks(
            r#"<w:altChunk r:id="d"/>"#,
            &[("d", "afchunk.docx", Some(WORD_TYPE), &inner)],
            "",
        );
        let (doc, total) = parse_at(&outer, 0, 0).unwrap();
        assert_eq!(doc.blocks.len(), 1);
        assert!(total >= alone + inner.len() as u64, "{total} < {alone} + {}", inner.len());
    }
}
