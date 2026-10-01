//! OpenDocument Text (.odt), Spreadsheet (.ods), and Presentation (.odp).

mod styles;
mod table;
mod text;

use crate::error::ConvertError;
use crate::model::{Block, Document, Inline, inlines_are_empty};
use crate::package::Package;
use crate::package::xml::{Element, ns};
use crate::shared::assets::AssetSink;
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
    let blocks = if let Some(text) = body.find(ns::OFFICE, "text") {
        parse_container(text, &ctx)?
    } else if let Some(sheet) = body.find(ns::OFFICE, "spreadsheet") {
        table::parse_spreadsheet(sheet, &ctx)?
    } else if let Some(pres) = body.find(ns::OFFICE, "presentation") {
        parse_presentation(pres, &ctx, &mut slide_starts)?
    } else {
        return Err(ConvertError::malformed_part(
            "content.xml",
            "no recognized office body (text, spreadsheet, or presentation)",
        ));
    };

    let notes = ctx.notes.into_inner();
    let assets = std::mem::take(&mut assets.borrow_mut().assets);
    Ok(Document { blocks, notes, assets, slide_starts })
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
}
