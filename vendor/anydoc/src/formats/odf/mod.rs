//! OpenDocument Text (.odt), Spreadsheet (.ods), and Presentation (.odp).

mod styles;
mod table;
mod text;

use crate::error::ConvertError;
use crate::model::{Block, Document, Inline, Note, inlines_are_empty};
use crate::package::Package;
use crate::package::xml::{Element, ns};
use crate::shared::assets::AssetSink;
use crate::shared::code::{MonoShare, drop_line_gutters, listing_tables};
use crate::shared::tabs;
use std::cell::RefCell;
use text::{Ctx, parse_container};

pub fn parse(bytes: &[u8]) -> Result<Document, ConvertError> {
    let pkg = RefCell::new(Package::open(bytes)?);

    if is_encrypted(&pkg)? {
        return Err(ConvertError::Encrypted);
    }

    // styles.xml is optional; corrupt -> skipped (default styling).
    let styles_tree = pkg.borrow_mut().optional_xml_part("styles.xml")?;

    let content_tree = pkg.borrow_mut().required_xml_part("content.xml")?;

    let mut styles = styles::OdfStyles::default();
    if let Some(tree) = &styles_tree {
        styles.collect(tree);
    }
    styles.collect(&content_tree);

    let body = content_tree
        .find(ns::OFFICE, "document-content")
        .and_then(|d| d.find(ns::OFFICE, "body"))
        .ok_or_else(|| ConvertError::malformed_part("content.xml", "no office:body"))?;

    let assets = RefCell::new(AssetSink::new());
    let ctx = Ctx::new(&styles, &pkg, &assets);

    let mut slide_starts = Vec::new();
    let (blocks, notes) = if let Some(text) = body.find(ns::OFFICE, "text") {
        // markitai: a text document's code may be set in a monospaced font,
        // unless that font sets most of its text, which is then read again
        // without code.
        let (blocks, notes, share) = read_text(text, ctx, true)?;
        if share.sets_code_apart() {
            (blocks, notes)
        } else {
            *assets.borrow_mut() = AssetSink::new();
            let (blocks, notes, _) = read_text(text, Ctx::new(&styles, &pkg, &assets), false)?;
            (blocks, notes)
        }
    } else if let Some(sheet) = body.find(ns::OFFICE, "spreadsheet") {
        (table::parse_spreadsheet(sheet, &ctx)?, ctx.notes.into_inner())
    } else if let Some(pres) = body.find(ns::OFFICE, "presentation") {
        (parse_presentation(pres, &ctx, &mut slide_starts)?, ctx.notes.into_inner())
    } else {
        return Err(ConvertError::malformed_part(
            "content.xml",
            "no recognized office body (text, spreadsheet, or presentation)",
        ));
    };

    let assets = std::mem::take(&mut assets.borrow_mut().assets);
    Ok(Document { blocks, notes, assets, slide_starts, warnings: Vec::new() })
}

/// markitai: a text document's body and notes, and how much of its text is
/// set in a monospaced font. Its headings may be set by hand (see
/// [`crate::shared::visual`]), its tables with tab stops
/// ([`crate::shared::tabs`]) and, with `code_fonts`, its code in a
/// monospaced font ([`crate::shared::code`]): a table laying out a listing
/// is its code, and a numbered listing loses its line numbers.
fn read_text(
    text: &Element,
    mut ctx: Ctx,
    code_fonts: bool,
) -> Result<(Vec<Block>, Vec<Note>, MonoShare), ConvertError> {
    ctx.looks = Some(RefCell::default());
    ctx.tabs = Some(RefCell::default());
    ctx.lists = Some(RefCell::default());
    ctx.code_fonts = code_fonts;
    let mut blocks = parse_container(text, &ctx)?;
    let mut notes = ctx.notes.take();
    let rows = ctx.tabs.take().map(RefCell::into_inner).unwrap_or_default();
    // markitai: and its lists typed by hand ([`crate::shared::typed_lists`]).
    let lists = ctx.lists.take().map(RefCell::into_inner).unwrap_or_default();
    tabs::finish(rows, lists, ctx.looks.take().map(RefCell::into_inner), &mut blocks, &mut notes);
    for blocks in std::iter::once(&mut blocks).chain(notes.iter_mut().map(|note| &mut note.blocks))
    {
        listing_tables(blocks);
        drop_line_gutters(blocks);
    }
    Ok((blocks, notes, ctx.share.get()))
}

/// Encrypted ODF packages carry `manifest:encryption-data` elements on file
/// entries. The manifest is parsed properly - substring matching would
/// classify a document as encrypted over a mere comment. An absent or
/// unreadable manifest proves nothing (content parsing decides), but fatal
/// resource-limit errors propagate.
fn is_encrypted(pkg: &RefCell<Package>) -> Result<bool, ConvertError> {
    let Some(tree) = pkg.borrow_mut().optional_xml_part("META-INF/manifest.xml")? else {
        return Ok(false);
    };
    Ok(tree.first_descendant(ns::MANIFEST, "encryption-data").is_some())
}

fn parse_presentation(
    pres: &Element,
    ctx: &Ctx,
    slide_starts: &mut Vec<usize>,
) -> Result<Vec<Block>, ConvertError> {
    let mut blocks = Vec::new();
    for page in pres.find_all(ns::DRAW, "page") {
        // markitai: every `draw:page` is a slide, blank ones included.
        slide_starts.push(blocks.len());
        let mut title = Vec::new();
        let mut body = Vec::new();
        let mut notes = Vec::new();
        walk_shapes(page, ctx, &mut title, &mut body, &mut notes)?;
        blocks.append(&mut title);
        blocks.append(&mut body);
        // Speaker notes are included (fixed policy), set off as a quote.
        if !notes.is_empty() {
            blocks.push(Block::BlockQuote(notes));
        }
    }
    Ok(blocks)
}

/// Walk a page's shapes in document order, recursing into `draw:g` groups.
fn walk_shapes(
    parent: &Element,
    ctx: &Ctx,
    title: &mut Vec<Block>,
    body: &mut Vec<Block>,
    notes: &mut Vec<Block>,
) -> Result<(), ConvertError> {
    for child in parent.child_elems() {
        if child.is(ns::PRESENTATION, "notes") {
            for frame in child.descendants(ns::DRAW, "frame") {
                if let Some(text_box) = frame.find(ns::DRAW, "text-box") {
                    notes.extend(parse_container(text_box, ctx)?);
                }
            }
            continue;
        }
        if child.ns.as_deref().is_none_or(|n| n != ns::DRAW) {
            continue;
        }
        match child.local.as_str() {
            "frame" => {
                let class = child.attr(ns::PRESENTATION, "class").unwrap_or("");
                if matches!(class, "page-number" | "date-time" | "footer" | "header") {
                    continue;
                }
                let mut inner = Vec::new();
                for content in child.child_elems() {
                    if content.is(ns::DRAW, "text-box") {
                        inner.extend(parse_container(content, ctx)?);
                    } else if content.is(ns::TABLE, "table") {
                        let mut tables = table::parse_table(content, ctx)?;
                        // markitai: a slide table styled with a header row
                        // (Impress's default) has its first row as the
                        // header, as PPTX and PPT tables do.
                        if content.attr(ns::TABLE, "use-first-row-styles") == Some("true") {
                            for block in &mut tables {
                                if let Block::Table(table) = block {
                                    table.header_rows = table.header_rows.max(1);
                                }
                            }
                        }
                        inner.extend(tables);
                    } else if content.is(ns::DRAW, "object")
                        && let Some(tex) = text::formula_tex(ctx, content)?
                    {
                        inner.push(Block::Math(tex));
                        break;
                    } else if content.is(ns::DRAW, "object")
                        && let Some(chart) = text::chart_blocks(ctx, content)?
                    {
                        // markitai: a chart's data, not its picture.
                        inner.extend(chart);
                        break;
                    } else if content.is(ns::DRAW, "image") {
                        let mut out = Vec::new();
                        let mut boxes = Vec::new();
                        text::walk_frame(child, ctx, &mut out, &mut boxes)?;
                        if !inlines_are_empty(&out) {
                            inner.push(Block::Paragraph(out));
                        }
                        inner.extend(boxes);
                        break;
                    }
                }
                if class == "title" {
                    push_title_heading(inner, title);
                } else {
                    body.append(&mut inner);
                }
            }
            "g" => walk_shapes(child, ctx, title, body, notes)?,
            "custom-shape" | "rect" | "ellipse" | "polygon" | "path" | "line" | "connector"
            | "caption" => {
                for content in child.child_elems() {
                    if content.is(ns::TEXT, "p") || content.is(ns::TEXT, "list") {
                        body.extend(parse_container(child, ctx)?);
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Collapse a title frame's paragraphs into one slide heading.
fn push_title_heading(inner: Vec<Block>, blocks: &mut Vec<Block>) {
    let mut inlines: Vec<Inline> = Vec::new();
    for block in inner {
        let Block::Paragraph(para) = block else { continue };
        if inlines_are_empty(&para) {
            continue;
        }
        if !inlines.is_empty() {
            inlines.push(Inline::LineBreak);
        }
        inlines.extend(para);
    }
    if !inlines_are_empty(&inlines) {
        let anchor = Some(crate::model::inlines_to_plain_text(&inlines));
        blocks.push(Block::Heading { level: 2, anchor, content: inlines });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    const CONTENT: &[u8] = br#"<office:document-content
        xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
        xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
        <office:body><office:text><text:p>hello</text:p></office:text></office:body>
        </office:document-content>"#;

    fn odt(manifest: &[u8]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default();
        w.start_file("META-INF/manifest.xml", opts).unwrap();
        w.write_all(manifest).unwrap();
        w.start_file("content.xml", opts).unwrap();
        w.write_all(CONTENT).unwrap();
        w.finish().unwrap().into_inner()
    }

    #[test]
    fn corrupt_manifest_does_not_classify_as_encrypted() {
        let doc = parse(&odt(b"<manifest:manifest <<< not xml")).unwrap();
        assert!(!doc.blocks.is_empty());
    }

    fn odt_with_content(content: &str) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        w.start_file("content.xml", zip::write::SimpleFileOptions::default()).unwrap();
        w.write_all(content.as_bytes()).unwrap();
        w.finish().unwrap().into_inner()
    }

    fn table_doc(rows: &str) -> String {
        format!(
            r#"<office:document-content
            xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
            xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
            xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
            <office:body><office:text><table:table>{rows}</table:table></office:text></office:body>
            </office:document-content>"#
        )
    }

    #[test]
    fn repeated_rows_cannot_amplify_text_beyond_the_byte_budget() {
        // H3: the slot budget alone would admit 1000 copies of a 100 KB
        // cell (~100 MB); the duplicated-bytes budget rejects it up front.
        let fat = "x".repeat(100_000);
        let rows = format!(
            r#"<table:table-row table:number-rows-repeated="1000">
            <table:table-cell><text:p>{fat}</text:p></table:table-cell>
            </table:table-row>"#
        );
        let err = parse(&odt_with_content(&table_doc(&rows))).unwrap_err();
        assert!(
            matches!(err, ConvertError::ResourceLimit { limit: "max_expansion_text_bytes", .. }),
            "expected the text-byte budget, got: {err}"
        );
    }

    #[test]
    fn repeated_rows_parse_content_once() {
        // A note inside a repeated row must register once, not per copy.
        let rows = r#"<table:table-row table:number-rows-repeated="3">
            <table:table-cell><text:p>cell<text:note text:id="n1">
            <text:note-body><text:p>note body</text:p></text:note-body>
            </text:note></text:p></table:table-cell>
            </table:table-row>"#;
        let doc = parse(&odt_with_content(&table_doc(rows))).unwrap();
        assert_eq!(doc.notes.len(), 1, "repeated rows must not duplicate notes");
    }

    #[test]
    fn styled_runs_stop_at_tables_and_work_inside_cells() {
        let content = r#"<office:document-content
            xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
            xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
            xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
            xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
            <office:automatic-styles>
              <style:style style:name="Source_20_Code" style:family="paragraph"/>
            </office:automatic-styles>
            <office:body><office:text>
              <text:p text:style-name="Source_20_Code">before table</text:p>
              <table:table><table:table-row><table:table-cell>
                <text:p text:style-name="Source_20_Code">inside cell</text:p>
              </table:table-cell></table:table-row></table:table>
              <text:p>after table</text:p>
            </office:text></office:body>
            </office:document-content>"#;
        let doc = parse(&odt_with_content(content)).unwrap();
        let [Block::CodeBlock { text: before, .. }, Block::Table(table), Block::Paragraph(after)] =
            &doc.blocks[..]
        else {
            panic!("unexpected block order: {:?}", doc.blocks);
        };
        assert_eq!(before, "before table");
        assert_eq!(crate::model::inlines_to_plain_text(after), "after table");
        let crate::model::CellSlot::Origin(cell) = &table.grid[0][0] else {
            panic!("expected an origin cell")
        };
        let [Block::CodeBlock { text: inside, .. }] = &cell.blocks[..] else {
            panic!("unexpected cell blocks: {:?}", cell.blocks)
        };
        assert_eq!(inside, "inside cell");
    }

    #[test]
    fn resource_limit_in_manifest_is_fatal() {
        let mut manifest = Vec::new();
        for _ in 0..(crate::package::limits::MAX_XML_DEPTH + 2) {
            manifest.extend_from_slice(b"<d>");
        }
        let err = parse(&odt(&manifest)).unwrap_err();
        assert!(
            matches!(err, ConvertError::ResourceLimit { limit: "max_xml_depth", .. }),
            "encryption probing must not swallow fatal errors, got: {err}"
        );
    }

    #[test]
    fn form_checkboxes_anchored_in_a_cell_follow_its_content() {
        let content = r#"<office:document-content
            xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
            xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
            xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
            xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
            xmlns:form="urn:oasis:names:tc:opendocument:xmlns:form:1.0"
            xmlns:xml="http://www.w3.org/XML/1998/namespace">
            <office:body><office:spreadsheet><table:table table:name="S">
            <office:forms><form:form>
              <form:checkbox xml:id="c1" form:label="Roof" form:current-state="checked"/>
              <form:checkbox xml:id="c2" form:label="Wall"/>
              <form:checkbox xml:id="c3" form:current-state="unknown"/>
            </form:form></office:forms>
            <table:table-row>
              <table:table-cell><text:p>14</text:p></table:table-cell>
              <table:table-cell><text:p>L/R</text:p><draw:control draw:control="c1"/></table:table-cell>
              <table:table-cell><draw:control draw:control="c2"/><draw:control draw:control="c3"/></table:table-cell>
            </table:table-row>
            </table:table></office:spreadsheet></office:body>
            </office:document-content>"#;
        let doc = parse(&odt_with_content(content)).unwrap();
        let md = crate::render::markdown::document_to_markdown(&doc);
        assert_eq!(md, "|  |  |  |\n| --- | --- | --- |\n| 14 | L/R [x] Roof | [ ] Wall |\n");
    }

    // markitai: shapes are read where they stand in text documents only.
    #[test]
    fn a_shape_anchored_to_a_cell_is_not_the_cells_value() {
        let content = r#"<office:document-content
            xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
            xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
            xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
            xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0">
            <office:body><office:spreadsheet><table:table table:name="S"><table:table-row>
              <table:table-cell><text:p>Value</text:p><draw:custom-shape
                table:end-cell-address="S.C3"><text:p>Callout</text:p></draw:custom-shape></table:table-cell>
            </table:table-row></table:table></office:spreadsheet></office:body>
            </office:document-content>"#;
        let doc = parse(&odt_with_content(content)).unwrap();
        let md = crate::render::markdown::document_to_markdown(&doc);
        assert_eq!(md, "|  |\n| --- |\n| Value |\n");
    }

    // markitai: slide boundaries reach `Document::slide_starts`.
    #[test]
    fn every_slide_starts_at_its_first_block_and_blank_slides_keep_their_place() {
        let content = r#"<office:document-content
            xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
            xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
            xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
            xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0">
            <office:body><office:presentation>
              <draw:page>
                <draw:frame presentation:class="title"><draw:text-box>
                  <text:p>First</text:p></draw:text-box></draw:frame>
                <draw:frame><draw:text-box><text:p>First body</text:p></draw:text-box></draw:frame>
              </draw:page>
              <draw:page/>
              <draw:page>
                <presentation:notes><draw:frame><draw:text-box>
                  <text:p>Note three</text:p></draw:text-box></draw:frame></presentation:notes>
              </draw:page>
              <draw:page>
                <draw:frame><draw:text-box><text:p>Fourth body</text:p></draw:text-box></draw:frame>
              </draw:page>
              <draw:page/>
            </office:presentation></office:body>
            </office:document-content>"#;
        let doc = parse(&odt_with_content(content)).unwrap();
        let [
            Block::Heading { .. },
            Block::Paragraph(_),
            Block::BlockQuote(_),
            Block::Paragraph(fourth),
        ] = &doc.blocks[..]
        else {
            panic!("unexpected blocks: {:?}", doc.blocks);
        };
        assert_eq!(crate::model::inlines_to_plain_text(fourth), "Fourth body");
        // Slide 2 has no blocks, so it starts where slide 3 does, and the
        // notes of the otherwise blank slide 3 stay in it; slide 5 has none
        // and is last, so it starts at the end.
        assert_eq!(doc.slide_starts, [0, 2, 2, 3, 4]);
    }

    #[test]
    fn only_presentations_record_slide_boundaries() {
        assert!(parse(&odt(b"<manifest:manifest/>")).unwrap().slide_starts.is_empty());
    }

    fn text_doc(styles: &str, body: &str) -> String {
        format!(
            r#"<office:document-content
            xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
            xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
            xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
            xmlns:dc="http://purl.org/dc/elements/1.1/">
            <office:automatic-styles>{styles}</office:automatic-styles>
            <office:body><office:text>{body}</office:text></office:body>
            </office:document-content>"#
        )
    }

    fn markdown(content: &str) -> String {
        crate::render::markdown::document_to_markdown(&parse(&odt_with_content(content)).unwrap())
    }

    // markitai: `style:text-position`.
    #[test]
    fn raised_and_lowered_text_uses_unicode_forms_where_it_has_them() {
        let styles = r#"
            <style:style style:name="Sup" style:family="text">
              <style:text-properties style:text-position="super 58%"/></style:style>
            <style:style style:name="Low" style:family="text">
              <style:text-properties style:text-position="-33% 100%"/></style:style>
            <style:style style:name="Note" style:family="text" style:parent-style-name="Sup"/>
            <style:style style:name="Flat" style:family="text">
              <style:text-properties style:text-position="0% 100%"/></style:style>"#;
        let body = r#"<text:p>library<text:span text:style-name="Note">1</text:span>
            H<text:span text:style-name="Low">2</text:span>O
            1<text:span text:style-name="Sup">st</text:span>
            <text:span text:style-name="Sup">10<text:span text:style-name="Flat">2</text:span></text:span></text:p>"#;
        assert_eq!(markdown(&text_doc(styles, body)), "library¹ H₂O 1st ¹⁰2\n");
    }

    // markitai: comments, phonetic guides and hidden text.
    #[test]
    fn comments_ruby_guides_and_hidden_text_stay_out_of_the_text() {
        let styles = r#"
            <style:style style:name="Hidden" style:family="text">
              <style:text-properties text:display="none"/></style:style>
            <style:style style:name="Shown" style:family="text" style:parent-style-name="Hidden">
              <style:text-properties text:display="true"/></style:style>"#;
        let body = r#"<text:p>Before<office:annotation><dc:creator>Ann Author</dc:creator>
            <dc:date>2026-01-01T00:00:00</dc:date><text:p>Check this claim</text:p>
            </office:annotation> after<office:annotation-end/>.</text:p>
            <text:p><text:ruby><text:ruby-base>漢字</text:ruby-base><text:ruby-text>かんじ</text:ruby-text></text:ruby>です</text:p>
            <text:p>Shown<text:span text:style-name="Hidden"> secret<text:span text:style-name="Shown"> visible</text:span></text:span>.</text:p>"#;
        assert_eq!(
            markdown(&text_doc(styles, body)),
            "Before after.\n\n漢字です\n\nShown visible.\n"
        );
    }

    const CHART: &str = r#"<office:document-content
        xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
        xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
        xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
        xmlns:chart="urn:oasis:names:tc:opendocument:xmlns:chart:1.0">
        <office:body><office:chart><chart:chart chart:class="chart:bar">
        <chart:title><text:p>Sales</text:p></chart:title>
        <table:table table:name="local-table">
          <table:table-header-rows><table:table-row>
            <table:table-cell><text:p/></table:table-cell>
            <table:table-cell office:value-type="string"><text:p>2024</text:p></table:table-cell>
          </table:table-row></table:table-header-rows>
          <table:table-rows><table:table-row>
            <table:table-cell office:value-type="string"><text:p>East</text:p></table:table-cell>
            <table:table-cell office:value-type="float" office:value="19.2"><text:p>19.2</text:p></table:table-cell>
          </table:table-row></table:table-rows>
        </table:table></chart:chart></office:chart></office:body></office:document-content>"#;

    fn package(parts: &[(&str, &str)]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, content) in parts {
            w.start_file(*name, zip::write::SimpleFileOptions::default()).unwrap();
            w.write_all(content.as_bytes()).unwrap();
        }
        w.finish().unwrap().into_inner()
    }

    // markitai: chart objects and slide table headers.
    #[test]
    fn a_chart_object_reads_as_its_data_table() {
        let frame = r#"<draw:frame><draw:object xlink:href="./Object 1"/>
            <draw:image xlink:href="./ObjectReplacements/Object 1"/></draw:frame>"#;
        let namespaces = r#"xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
            xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
            xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
            xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
            xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"
            xmlns:xlink="http://www.w3.org/1999/xlink""#;
        let expected = "Sales\n\n|  | 2024 |\n| --- | --- |\n| East | 19.2 |\n";
        // In a text document the chart's data follows its paragraph.
        let odt = format!(
            r#"<office:document-content {namespaces}><office:body><office:text>
            <text:p>See the chart.{frame}</text:p></office:text></office:body></office:document-content>"#
        );
        let doc =
            parse(&package(&[("content.xml", &odt), ("Object 1/content.xml", CHART)])).unwrap();
        assert!(doc.assets.is_empty(), "the replacement picture is not read: {:?}", doc.assets);
        let md = crate::render::markdown::document_to_markdown(&doc);
        assert_eq!(md, format!("See the chart.\n\n{expected}"));
        // On a slide, beside a table styled with a header row: its first
        // row heads it even where its values alone would not say so.
        let slide_table = |styled: &str| {
            format!(
                r#"<office:document-content {namespaces}><office:body><office:presentation>
                <draw:page>{frame}<draw:frame><table:table {styled}>
                  <table:table-row><table:table-cell><text:p>2024</text:p></table:table-cell>
                    <table:table-cell><text:p>2025</text:p></table:table-cell></table:table-row>
                  <table:table-row><table:table-cell><text:p>10</text:p></table:table-cell>
                    <table:table-cell><text:p>12</text:p></table:table-cell></table:table-row>
                </table:table></draw:frame></draw:page>
                </office:presentation></office:body></office:document-content>"#
            )
        };
        let slide = |styled: &str| {
            let odp = slide_table(styled);
            let parts = [("content.xml", odp.as_str()), ("Object 1/content.xml", CHART)];
            crate::render::markdown::document_to_markdown(&parse(&package(&parts)).unwrap())
        };
        assert_eq!(
            slide(r#"table:use-first-row-styles="true""#),
            format!("{expected}\n| 2024 | 2025 |\n| --- | --- |\n| 10 | 12 |\n")
        );
        assert_eq!(
            slide(""),
            format!("{expected}\n|  |  |\n| --- | --- |\n| 2024 | 2025 |\n| 10 | 12 |\n")
        );
        // Any other object keeps its replacement picture path (none here).
        let other = r#"<office:document-content
            xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0">
            <office:body><office:drawing/></office:body></office:document-content>"#;
        let doc =
            parse(&package(&[("content.xml", &odt), ("Object 1/content.xml", other)])).unwrap();
        assert_eq!(crate::render::markdown::document_to_markdown(&doc), "See the chart.\n");
    }

    // markitai: headings set by hand (see `crate::shared::visual`).
    #[test]
    fn bold_paragraphs_set_above_the_body_size_are_headings_ranked_by_size() {
        let prose = r#"<text:p text:style-name="Body">Body text long enough to outweigh every
            heading together, as the text of a document does: a few sentences at one size.</text:p>"#;
        let content = |extra: &str| {
            format!(
                r#"<office:document-content
                xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
                xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
                xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
                xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0">
                <office:automatic-styles>
                  <style:style style:name="Big" style:family="paragraph">
                    <style:text-properties fo:font-size="24.0pt"/></style:style>
                  <style:style style:name="Body" style:family="paragraph">
                    <style:text-properties fo:font-size="12pt"/></style:style>
                  <style:style style:name="Mid" style:family="paragraph">
                    <style:text-properties fo:font-size="18pt"/></style:style>
                  <style:style style:name="Half" style:family="paragraph">
                    <style:text-properties fo:font-size="150%"/></style:style>
                  <style:style style:name="B" style:family="text">
                    <style:text-properties fo:font-weight="bold"/></style:style>
                  <style:style style:name="Tiny" style:family="text">
                    <style:text-properties fo:font-size="9pt" fo:font-weight="bold"/></style:style>
                </office:automatic-styles>
                <office:body><office:text>
                  <text:p text:style-name="Body">By the authors</text:p>
                  <text:p text:style-name="Big"><text:span text:style-name="B">The Title</text:span></text:p>
                  {prose}
                  <text:section text:name="S"><text:p text:style-name="Mid"><text:span
                    text:style-name="B">In a Section</text:span></text:p></text:section>
                  {prose}
                  <text:list><text:list-item><text:p text:style-name="Mid"><text:span
                    text:style-name="B">Listed</text:span></text:p></text:list-item></text:list>
                  <text:p text:style-name="Half"><text:span text:style-name="B">Relative</text:span></text:p>
                  <text:p text:style-name="Mid"><text:span text:style-name="B">Mixed</text:span><text:span
                    text:style-name="Tiny">sizes</text:span></text:p>
                  {prose}{extra}
                </office:text></office:body></office:document-content>"#
            )
        };
        let shapes = |extra: &str| -> Vec<String> {
            parse(&odt_with_content(&content(extra)))
                .unwrap()
                .blocks
                .iter()
                .map(|block| match block {
                    Block::Heading { level, .. } => format!("h{level}"),
                    Block::Paragraph(_) => "p".into(),
                    Block::List(_) => "list".into(),
                    other => format!("{other:?}"),
                })
                .collect()
        };
        // The short first line would take a list paragraph's place if one
        // were taken for the body's.
        assert_eq!(shapes(""), ["p", "h1", "p", "h2", "p", "list", "h2", "p", "p"]);
        let styled = r#"<text:h text:outline-level="1">Real</text:h>"#;
        assert_eq!(shapes(styled), ["p", "p", "p", "p", "p", "list", "p", "p", "p", "h1"]);
    }

    /// markitai: a text document with these font faces and styles, its body
    /// described (see `crate::shared::code::describe`), and its notes'.
    fn described(default_font: &str, body: &str) -> (Vec<String>, Vec<String>) {
        let content = format!(
            r#"<office:document-content
            xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
            xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
            xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
            xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
            xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
            xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"
            xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
            xmlns:xlink="http://www.w3.org/1999/xlink">
            <office:font-face-decls>
              <style:font-face style:name="Menlo" svg:font-family="Menlo"/>
              <style:font-face style:name="Gothic" svg:font-family="'Letter Gothic'"
                style:font-pitch="fixed"/>
              <style:font-face style:name="MS Gothic" svg:font-family="'MS Gothic'"
                style:font-family-generic="modern" style:font-pitch="fixed"/>
              <style:font-face style:name="Times" svg:font-family="Times"/>
            </office:font-face-decls>
            <office:styles>
              <style:default-style style:family="paragraph">
                <style:text-properties style:font-name="{default_font}"/></style:default-style>
              <style:style style:name="Code" style:family="paragraph">
                <style:text-properties style:font-name="Menlo"/></style:style>
              <style:style style:name="Cols" style:family="paragraph"><style:paragraph-properties>
                <style:tab-stops><style:tab-stop style:position="2in"/>
                <style:tab-stop style:position="4in" style:type="right"/></style:tab-stops>
              </style:paragraph-properties></style:style>
              <style:style style:name="Dots" style:family="paragraph"><style:paragraph-properties>
                <style:tab-stops><style:tab-stop style:position="6in" style:type="right"
                  style:leader-style="dotted" style:leader-text="."/></style:tab-stops>
              </style:paragraph-properties></style:style>
            </office:styles>
            <office:automatic-styles>
              <style:style style:name="P1" style:family="paragraph" style:parent-style-name="Code"/>
              <style:style style:name="P2" style:family="paragraph" style:parent-style-name="Cols"/>
              <style:style style:name="T1" style:family="text">
                <style:text-properties style:font-name="Gothic"/></style:style>
              <style:style style:name="T2" style:family="text">
                <style:text-properties fo:font-family="'Iosevka Term', monospace"/></style:style>
              <style:style style:name="T3" style:family="text">
                <style:text-properties style:font-name="MS Gothic"/></style:style>
              <style:style style:name="T4" style:family="text">
                <style:text-properties fo:font-family="Georgia, serif"/></style:style>
              <style:style style:name="T5" style:family="text">
                <style:text-properties style:font-name="Times"/></style:style>
            </office:automatic-styles>
            <office:body><office:text>{body}</office:text></office:body>
            </office:document-content>"#
        );
        let doc = parse(&odt_with_content(&content)).unwrap();
        let notes = doc.notes.iter().flat_map(|note| crate::shared::code::describe(&note.blocks));
        (crate::shared::code::describe(&doc.blocks), notes.collect())
    }

    const PROSE: &str = "<text:p>And then the prose of the document goes on for a while, in the face \
        the document is set in.</text:p>";

    #[test]
    fn text_in_a_monospaced_font_is_code() {
        let body = format!(
            r#"<text:h text:outline-level="1"><text:span text:style-name="T1">Setup</text:span></text:h>
            <text:p>Run <text:span text:style-name="T1">make</text:span> or <text:span
              text:style-name="T2">cargo</text:span>; <text:span text:style-name="T3">kanji</text:span>
              and <text:span text:style-name="T4">serif</text:span> are text.</text:p>
            <text:p text:style-name="P1">fn main() {{</text:p>
            <text:p text:style-name="P1"><text:span text:style-name="T2"/></text:p>
            <text:p text:style-name="P1"><text:s text:c="4"/>go();</text:p>
            <text:p text:style-name="P1">}}</text:p>
            <text:list><text:list-item><text:p text:style-name="P1">listed</text:p></text:list-item></text:list>
            <table:table><table:table-row><table:table-cell><text:p text:style-name="P1">celled</text:p>
              </table:table-cell><table:table-cell><text:p>prose cell</text:p></table:table-cell>
            </table:table-row></table:table>
            <table:table><table:table-row><table:table-cell><text:p text:style-name="P1">1</text:p>
              <text:p text:style-name="P1">2</text:p></table:table-cell><table:table-cell>
              <text:p text:style-name="P1">a = 1</text:p><text:p text:style-name="P1">b = 2</text:p>
            </table:table-cell></table:table-row></table:table>
            <text:p text:style-name="P1">1</text:p><text:p text:style-name="P1">x</text:p>
            <text:p text:style-name="P1">2</text:p><text:p text:style-name="P1">y</text:p>
            <text:p text:style-name="P1">fig.<draw:frame><draw:image xlink:href="Pictures/none.png"/>
              <svg:title>A figure</svg:title></draw:frame></text:p>
            {PROSE}"#
        );
        let (blocks, _) = described("Times", &body);
        assert_eq!(
            blocks,
            [
                "h1:Setup",
                "p:Run `make` or `cargo`; kanji and serif are text.",
                "code:fn main() {\n\n    go();\n}",
                "list:p:`listed`",
                "table:p:`celled`|p:prose cell",
                "code:a = 1\nb = 2",
                "code:x\ny",
                // A paragraph with a picture is no line of code.
                "p:`fig.`",
                &format!("p:{}", &PROSE[8..PROSE.len() - 9]),
            ]
        );
        // A document set in a monospaced font marks no code.
        let typed = format!(r#"<text:p>INT. KITCHEN - NIGHT</text:p>{PROSE}"#);
        let (blocks, _) = described("Menlo", &typed);
        assert_eq!(blocks[0], "p:INT. KITCHEN - NIGHT");
        assert!(!blocks[1].contains('`'), "{blocks:?}");
        // Where the default face is monospaced but the text is mostly set
        // in another, a paragraph in the default face is code.
        let spans = format!(
            r#"<text:p>ls -la</text:p><text:p><text:span text:style-name="T5">{}</text:span></text:p>"#,
            &PROSE[8..PROSE.len() - 9]
        );
        let (blocks, _) = described("Menlo", &spans);
        assert_eq!(blocks[0], "code:ls -la");
    }

    #[test]
    fn columns_set_with_tab_stops_are_a_table() {
        let row = |style: &str, cells: &[&str]| {
            format!(r#"<text:p text:style-name="{style}">{}</text:p>"#, cells.join("<text:tab/>"))
        };
        let body = [
            row("P2", &["Item", "Qty", "Price"]),
            row("P2", &["Apple", "3", "1.20"]),
            row("P2", &["Pear", "12", "0.50"]),
            row("Dots", &["Part 1", "Intro", "Page one"]),
            row("Dots", &["Part 2", "Methods", "Page two"]),
            row("Dots", &["Part 3", "Results", "Page three"]),
            r#"<text:h text:outline-level="1">A<text:tab/>B<text:tab/>C</text:h>"#.into(),
            r#"<text:list><text:list-item><text:p>a<text:tab/>b<text:tab/>c</text:p></text:list-item>
              <text:list-item><text:p>d<text:tab/>e<text:tab/>f</text:p></text:list-item>
              <text:list-item><text:p>g<text:tab/>h<text:tab/>i</text:p></text:list-item></text:list>"#
                .into(),
            r#"<text:p>Noted<text:note><text:note-body><text:p>x<text:tab/>y</text:p></text:note-body>
              </text:note></text:p>"#
                .into(),
        ]
        .concat();
        let (blocks, notes) = described("Times", &body);
        assert_eq!(
            blocks,
            [
                "table:p:Item|p:Qty|p:Price/p:Apple|p:3|p:1.20/p:Pear|p:12|p:0.50",
                "p:Part 1 Intro Page one",
                "p:Part 2 Methods Page two",
                "p:Part 3 Results Page three",
                "h1:A B C",
                "list:p:a b c|p:d e f|p:g h i",
                "p:Noted",
            ]
        );
        assert_eq!(notes, ["p:x y"]);
        // A heading is linked to by its text, tabs read as spaces.
        let content = format!(
            r#"<office:document-content
            xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
            xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
            <office:body><office:text>{}</office:text></office:body></office:document-content>"#,
            r#"<text:h text:outline-level="1">A<text:tab/>B</text:h>"#
        );
        let doc = parse(&odt_with_content(&content)).unwrap();
        let [Block::Heading { anchor: Some(anchor), .. }] = &doc.blocks[..] else {
            panic!("{:?}", doc.blocks)
        };
        assert_eq!(anchor, "A B");
    }

    #[test]
    fn margins_set_the_levels_of_a_list_typed_by_hand() {
        // markitai: a style's `fo:margin-left` and `fo:text-indent`, and a
        // parent style's, place the bullet; a bullet line set in a
        // monospaced face is an item, not code.
        let read = |body: &str| {
            let content = format!(
                r#"<office:document-content
                xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
                xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
                xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
                xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
                xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0">
                <office:font-face-decls>
                  <style:font-face style:name="Menlo" svg:font-family="Menlo"/>
                </office:font-face-decls>
                <office:automatic-styles>
                  <style:style style:name="Hang" style:family="paragraph"><style:paragraph-properties
                    fo:margin-left="0.25in" fo:text-indent="-0.25in"/></style:style>
                  <style:style style:name="Deep" style:family="paragraph"><style:paragraph-properties
                    fo:margin-left="1.27cm"/></style:style>
                  <style:style style:name="Deeper" style:family="paragraph"
                    style:parent-style-name="Deep"/>
                  <style:style style:name="Mono" style:family="text">
                    <style:text-properties style:font-name="Menlo"/></style:style>
                </office:automatic-styles>
                <office:body><office:text>{body}
                <text:p>And then the prose of the document goes on for a while.</text:p>
                </office:text></office:body></office:document-content>"#
            );
            let doc = parse(&odt_with_content(&content)).unwrap();
            crate::shared::code::describe(&doc.blocks)[..doc.blocks.len() - 1].to_vec()
        };
        let item = |style: &str, text: &str| {
            format!(r#"<text:p text:style-name="{style}">•<text:tab/>{text}</text:p>"#)
        };
        assert_eq!(read(&[item("", "a"), item("Hang", "b")].concat()), ["list:p:a|p:b"]);
        assert_eq!(read(&[item("", "a"), item("Deeper", "b")].concat()), ["list:p:a;list:p:b"]);
        let mono = r#"<text:p><text:span text:style-name="Mono">•<text:tab/>npm test</text:span></text:p>"#;
        assert_eq!(read(mono), ["list:p:`npm test`"]);
    }

    /// markitai: a text document's blocks, described, and each list's start.
    fn lists_of(body: &str) -> Vec<String> {
        let styles = r#"<text:list-style style:name="L1">
              <text:list-level-style-number text:level="1" style:num-format="1"/>
              <text:list-level-style-number text:level="2" style:num-format="a"/>
            </text:list-style>"#;
        let doc = parse(&odt_with_content(&text_doc(styles, body))).unwrap();
        let mut shown = crate::shared::code::describe(&doc.blocks);
        for block in &doc.blocks {
            if let Block::List(list) = block {
                shown.push(format!("start {}", list.start));
            }
        }
        shown
    }

    #[test]
    fn an_unnumbered_entry_between_items_stays_in_the_item_before_it() {
        // Several paragraphs of one item, and a paragraph without a number
        // between items as LibreOffice writes it: the list is closed and a
        // list continuing it opens with a list header.
        let body = r#"<text:p>Before.</text:p>
            <text:list xml:id="list1" text:style-name="L1">
              <text:list-item><text:p>One.</text:p><text:p>More of one.</text:p></text:list-item>
              <text:list-item><text:p>Two.</text:p></text:list-item>
            </text:list>
            <text:list xml:id="list2" text:continue-numbering="true" text:style-name="L1">
              <text:list-header><text:p>More of two.</text:p></text:list-header>
              <text:list-item><text:p>Three.</text:p>
                <text:list><text:list-item><text:p>Three a.</text:p></text:list-item></text:list>
              </text:list-item>
            </text:list>
            <text:list text:continue-list="list2" text:style-name="L1">
              <text:list-header><text:list><text:list-header>
                <text:p>More of three a.</text:p>
              </text:list-header></text:list></text:list-header>
              <text:list-item><text:p>Four.</text:p></text:list-item>
            </text:list>
            <text:p>After.</text:p>
            <text:list text:style-name="L1">
              <text:list-header><text:p>A header opening a list of its own.</text:p></text:list-header>
              <text:list-item><text:p>Alone.</text:p></text:list-item>
            </text:list>"#;
        assert_eq!(
            lists_of(body),
            [
                "p:Before.",
                "list:p:One.;p:More of one.|p:Two.;p:More of two.|p:Three.;list:p:Three a.;p:More of three a.|p:Four.",
                "p:After.",
                "p:A header opening a list of its own.",
                "list:p:Alone.",
                "start 1",
                "start 1",
            ]
        );
        // The entry nested one list deep is more of the nested item, not
        // of its parent.
        let doc = parse(&odt_with_content(&text_doc("", body))).unwrap();
        let Some(Block::List(list)) = doc.blocks.get(1) else { panic!("{:?}", doc.blocks) };
        let [_, Block::List(nested)] = &list.items[2].blocks[..] else {
            panic!("{:?}", list.items[2].blocks)
        };
        assert_eq!(nested.items[0].blocks.len(), 2, "{:?}", nested.items[0].blocks);
    }

    /// markitai: a text document with the drawing, form and script namespaces.
    fn drawing_doc(styles: &str, body: &str) -> String {
        format!(
            r#"<office:document-content
            xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
            xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
            xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
            xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
            xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"
            xmlns:xlink="http://www.w3.org/1999/xlink"
            xmlns:script="urn:oasis:names:tc:opendocument:xmlns:script:1.0"
            xmlns:office2="urn:oasis:names:tc:opendocument:xmlns:office:1.0">
            <office:automatic-styles>{styles}</office:automatic-styles>
            <office:body><office:text>{body}</office:text></office:body>
            </office:document-content>"#
        )
    }

    // markitai: frames and shapes anchored to the page, and shapes in a
    // paragraph.
    #[test]
    fn frames_and_shapes_are_read_where_they_stand() {
        let body = r#"
            <draw:frame text:anchor-type="page" text:anchor-page-number="1"><draw:text-box>
              <text:p>Page box.</text:p></draw:text-box></draw:frame>
            <draw:custom-shape text:anchor-type="page"><text:p>Page shape.</text:p>
              <draw:enhanced-geometry draw:type="rectangle"/></draw:custom-shape>
            <draw:g><draw:rect><text:p>Grouped on the page.</text:p></draw:rect></draw:g>
            <text:p>Body.</text:p>
            <text:p>Anchor.<draw:custom-shape text:anchor-type="paragraph"><text:p>Shape words</text:p>
              <draw:enhanced-geometry draw:type="ellipse"/></draw:custom-shape> more.</text:p>
            <text:p>Group:<draw:g><draw:frame><draw:text-box><text:p>Left box</text:p></draw:text-box></draw:frame>
              <draw:ellipse><text:p>Right shape</text:p></draw:ellipse></draw:g></text:p>
            <text:p>Linked <draw:a xlink:href="https://example.com"><draw:frame><svg:title>Logo</svg:title>
              <draw:image xlink:href="Pictures/missing.png"/></draw:frame></draw:a> picture.</text:p>
            <text:p>Empty <draw:line svg:x1="0cm" svg:x2="1cm"/>line.</text:p>"#;
        assert_eq!(
            markdown(&drawing_doc("", body)),
            "Page box.\n\nPage shape.\n\nGrouped on the page.\n\nBody.\n\nAnchor. more.\n\n\
             Shape words\n\nGroup:\n\nLeft box\n\nRight shape\n\nLinked Logo picture.\n\n\
             Empty line.\n"
        );
    }

    // markitai: text set in a symbol font.
    #[test]
    fn text_in_a_symbol_font_reads_as_its_glyphs() {
        let styles = r#"
            <style:style style:name="Sym" style:family="text">
              <style:text-properties style:font-name="Symbol1"/></style:style>
            <style:style style:name="Wing" style:family="text">
              <style:text-properties fo:font-family="'Wingdings', serif"/></style:style>
            <style:style style:name="Open" style:family="text">
              <style:text-properties style:font-name="OpenSymbol"/></style:style>"#;
        let content = drawing_doc(styles, r#"<text:p>Typed <text:span text:style-name="Sym">abg</text:span>, stored <text:span
            text:style-name="Sym">&#xF06D;</text:span>m, tick <text:span text:style-name="Wing">&#xF0FC;</text:span>, unicode <text:span
            text:style-name="Open">&#x2192;</text:span>, plain abg.</text:p>"#)
        .replace(
            "<office:automatic-styles>",
            r#"<office:font-face-decls><style:font-face style:name="Symbol1" svg:font-family="Symbol"
              style:font-charset="x-symbol"/><style:font-face style:name="OpenSymbol"
              svg:font-family="OpenSymbol"/></office:font-face-decls><office:automatic-styles>"#,
        )
        .replace("<office:document-content", r#"<office:document-content xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0""#);
        assert_eq!(markdown(&content), "Typed αβγ, stored μm, tick ✓, unicode →, plain abg.\n");
    }

    // markitai: a heading's number as shown.
    #[test]
    fn a_headings_shown_number_is_its_label() {
        let body = r#"<text:h text:outline-level="1"><text:number>2.</text:number>Methods</text:h>
            <text:h text:outline-level="2"><text:number></text:number>Unnumbered</text:h>"#;
        assert_eq!(markdown(&text_doc("", body)), "# 2. Methods\n\n## Unnumbered\n");
    }

    // markitai: ODF 1.2's numbered paragraphs.
    #[test]
    fn numbered_paragraphs_are_list_items() {
        let styles = r#"<text:list-style style:name="L1">
              <text:list-level-style-number text:level="1" style:num-format="1" style:num-suffix="."/>
              <text:list-level-style-number text:level="2" style:num-format="a" style:num-suffix=")"/>
            </text:list-style>"#;
        let body = r#"<text:p>Intro.</text:p>
            <text:numbered-paragraph text:style-name="L1" text:level="1"><text:number>1.</text:number>
              <text:p>One.</text:p></text:numbered-paragraph>
            <text:numbered-paragraph text:style-name="L1" text:level="1"><text:number>2.</text:number>
              <text:p>Two.</text:p></text:numbered-paragraph>
            <text:numbered-paragraph text:style-name="L1" text:level="2"><text:number>a)</text:number>
              <text:p>Two a.</text:p></text:numbered-paragraph>
            <text:numbered-paragraph text:style-name="L1"><text:number>7.</text:number>
              <text:p>Seven.</text:p></text:numbered-paragraph>
            <text:numbered-paragraph text:style-name="L1"><text:p/></text:numbered-paragraph>
            <text:p>Outro.</text:p>"#;
        assert_eq!(
            markdown(&drawing_doc(styles, body)),
            "Intro.\n\n1. One.\n2. Two.\n\n- a) Two a.\n\n7. Seven.\n\nOutro.\n"
        );
    }

    // markitai: invisible fields, scripts, hidden sections and the other
    // indexes.
    #[test]
    fn invisible_fields_scripts_and_hidden_sections_are_not_text() {
        let body = r#"
            <text:p>Shown <text:variable-set text:name="a" text:display="none" office:value-type="string">secret</text:variable-set><text:variable-set
              text:name="b" office:value-type="string">visible</text:variable-set> <text:user-field-get
              text:name="c" text:display="none">quiet</text:user-field-get><text:script
              script:language="JavaScript">alert("x")</text:script>end.</text:p>
            <text:section text:name="Hidden" text:display="none"><text:p>Hidden section.</text:p></text:section>
            <text:section text:name="Cond" text:display="condition" text:condition="ooow:x"><text:p>Conditional section.</text:p></text:section>
            <text:user-index><text:index-body><text:p>User entry, 2</text:p></text:index-body></text:user-index>
            <text:table-index><text:index-body><text:p>Table 1, 3</text:p></text:index-body></text:table-index>
            <text:object-index><text:index-body><text:p>Object 1, 4</text:p></text:index-body></text:object-index>"#;
        assert_eq!(
            markdown(&drawing_doc("", body)),
            "Shown visible end.\n\nConditional section.\n\nUser entry, 2\n\nTable 1, 3\n\nObject 1, 4\n"
        );
    }
}
