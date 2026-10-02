//! Word documents built in the test from the smallest package that has the
//! feature under test, converted through the whole native path.

use super::*;
use std::io::{Cursor, Write};

const W: &str = r#"xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main""#;

fn docx(body: &str, parts: &[(&str, String)]) -> Vec<u8> {
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default();
    let document = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?><w:document {W}><w:body>{body}</w:body></w:document>"#
    );
    zip.start_file("word/document.xml", options).unwrap();
    zip.write_all(document.as_bytes()).unwrap();
    for (name, content) in parts {
        zip.start_file(*name, options).unwrap();
        zip.write_all(content.as_bytes()).unwrap();
    }
    zip.finish().unwrap().into_inner()
}

fn markdown(body: &str, parts: &[(&str, String)]) -> String {
    extract(&docx(body, parts), "docx").unwrap().markdown
}

fn paragraph(text: &str) -> String {
    format!(r#"<w:p><w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>"#)
}

fn styles() -> (&'static str, String) {
    (
        "word/styles.xml",
        format!(
            r#"<w:styles {W}>
            <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>
            <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/></w:style>
            </w:styles>"#
        ),
    )
}

fn heading(level: u8, inner: &str) -> String {
    format!(r#"<w:p><w:pPr><w:pStyle w:val="Heading{level}"/></w:pPr>{inner}</w:p>"#)
}

fn cell(inner: &str) -> String {
    format!("<w:tc><w:tcPr><w:tcW w:w=\"2000\" w:type=\"dxa\"/></w:tcPr>{inner}</w:tc>")
}

fn table(rows: &[String]) -> String {
    format!(
        "<w:tbl><w:tblGrid><w:gridCol w:w=\"2000\"/><w:gridCol w:w=\"2000\"/></w:tblGrid>{}</w:tbl>",
        rows.concat()
    )
}

#[test]
fn a_note_starts_at_its_text_and_code_paragraphs_keep_their_indentation() {
    // Word writes a space between a note's mark and its text; it is not part
    // of the text. Paragraphs pasted from code keep their leading spaces, and
    // a line that starts `#` is the code's own comment, not a heading, so its
    // `#` is escaped.
    let footnotes = format!(
        r#"<w:footnotes {W}><w:footnote w:id="1"><w:p><w:r><w:footnoteRef/></w:r>
        <w:r><w:t xml:space="preserve"> Note text.</w:t></w:r></w:p></w:footnote></w:footnotes>"#
    );
    let body = format!(
        "{}{}{}",
        paragraph("    indented = 1"),
        paragraph("# a comment"),
        r#"<w:p><w:r><w:t>Cited</w:t></w:r><w:r><w:footnoteReference w:id="1"/></w:r></w:p>"#
    );
    assert_eq!(
        markdown(&body, &[("word/footnotes.xml", footnotes)]),
        "    indented = 1\n\n\\# a comment\n\nCited[^fn1]\n\n[^fn1]: Note text."
    );
}

#[test]
fn a_heading_with_a_soft_return_stays_one_heading() {
    let body = heading(
        1,
        r#"<w:r><w:t>First half</w:t><w:br/><w:t>second half</w:t></w:r>"#,
    );
    assert_eq!(markdown(&body, &[styles()]), "# First half second half");
}

#[test]
fn a_heading_in_a_table_cell_is_emphasis_not_hash_marks() {
    let body = table(&[format!(
        "<w:tr>{}{}</w:tr>",
        cell(&format!(
            "{}{}",
            heading(2, r#"<w:r><w:t>Cell heading</w:t></w:r>"#),
            paragraph("Body")
        )),
        cell(&paragraph("Other"))
    )]);
    assert_eq!(
        markdown(&body, &[styles()]),
        "|  |  |\n| --- | --- |\n| **Cell heading**<br><br>Body | Other |"
    );
}

#[test]
fn only_a_declared_header_row_heads_a_table() {
    let row = |header: bool, a: &str, b: &str| {
        let props = if header {
            "<w:trPr><w:tblHeader/></w:trPr>"
        } else {
            ""
        };
        format!(
            "<w:tr>{props}{}{}</w:tr>",
            cell(&paragraph(a)),
            cell(&paragraph(b))
        )
    };
    let declared = table(&[
        row(true, "Name", "Count"),
        row(false, "Alpha", "1"),
        row(false, "Beta", "2"),
    ]);
    let undeclared = table(&[
        row(false, "Name", "Count"),
        row(false, "Alpha", "1"),
        row(false, "Beta", "2"),
    ]);
    assert_eq!(
        markdown(&declared, &[]),
        "| Name | Count |\n| --- | --- |\n| Alpha | 1 |\n| Beta | 2 |"
    );
    assert_eq!(
        markdown(&undeclared, &[]),
        "|  |  |\n| --- | --- |\n| Name | Count |\n| Alpha | 1 |\n| Beta | 2 |"
    );
}

#[test]
fn a_row_deleted_in_tracked_changes_leaves_no_empty_row() {
    let row = |props: &str, inner: &str| {
        format!(
            "<w:tr><w:trPr>{props}</w:trPr>{}{}</w:tr>",
            cell(inner),
            cell(inner)
        )
    };
    let deleted = r#"<w:p><w:del w:id="2" w:author="A"><w:r><w:delText>Removed</w:delText></w:r></w:del></w:p>"#;
    let body = table(&[
        row("", &paragraph("Kept")),
        row(r#"<w:del w:id="1" w:author="A"/>"#, deleted),
        row("", &paragraph("Last")),
    ]);
    assert_eq!(
        markdown(&body, &[]),
        "|  |  |\n| --- | --- |\n| Kept | Kept |\n| Last | Last |"
    );
}

fn numbering(levels: &str) -> (&'static str, String) {
    (
        "word/numbering.xml",
        format!(
            r#"<w:numbering {W}><w:abstractNum w:abstractNumId="0">{levels}</w:abstractNum>
            <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num></w:numbering>"#
        ),
    )
}

fn item(level: u8, text: &str) -> String {
    format!(
        r#"<w:p><w:pPr><w:numPr><w:ilvl w:val="{level}"/><w:numId w:val="1"/></w:numPr></w:pPr>
        <w:r><w:t>{text}</w:t></w:r></w:p>"#
    )
}

#[test]
fn list_items_whose_labels_markdown_does_not_read_keep_a_line_each() {
    // `a)` starts no Markdown list, so the lines would run together as one
    // paragraph; the hard break keeps each on its own line, as written.
    let parts = [numbering(
        r#"<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/></w:lvl>
        <w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="lowerLetter"/><w:lvlText w:val="%2)"/></w:lvl>"#,
    )];
    let body = [
        item(0, "First"),
        item(1, "sub a"),
        item(1, "sub b"),
        item(0, "Second"),
    ]
    .concat();
    let markdown = markdown(&body, &parts);
    assert!(
        markdown.contains("\n   a) sub a  \n   b) sub b\n2. Second"),
        "{markdown:?}"
    );
}

#[test]
fn nested_content_under_a_label_markdown_does_not_read_is_not_an_indented_code_block() {
    // "(1)" is four columns wide with its space: content indented by that
    // much after a blank line would be a code block, not the item's list.
    let parts = [numbering(
        r#"<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="(%1)"/></w:lvl>
        <w:lvl w:ilvl="1"><w:numFmt w:val="bullet"/><w:lvlText w:val="o"/></w:lvl>"#,
    )];
    let body = [item(0, "Parent"), item(1, "Child")].concat();
    let markdown = markdown(&body, &parts);
    assert!(markdown.starts_with("(1) Parent"), "{markdown:?}");
    assert!(
        !markdown.lines().any(|line| line.starts_with("    ")),
        "{markdown:?}"
    );
    assert!(markdown.contains("* Child"), "{markdown:?}");
}

#[test]
fn chinese_counting_and_circled_numbers_are_written_as_the_document_shows_them() {
    let parts = [numbering(
        r#"<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="chineseCounting"/><w:lvlText w:val="%1、"/></w:lvl>
        <w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="decimalEnclosedCircle"/><w:lvlText w:val="%2"/></w:lvl>"#,
    )];
    let body = [item(0, "总则"), item(1, "范围"), item(0, "定义")].concat();
    let markdown = markdown(&body, &parts);
    assert!(markdown.starts_with("一、 总则"), "{markdown:?}");
    assert!(markdown.contains("① 范围"), "{markdown:?}");
    assert!(markdown.contains("二、 定义"), "{markdown:?}");
}

#[test]
fn ruby_base_text_symbols_and_non_breaking_hyphens_are_kept() {
    let body = concat!(
        r#"<w:p><w:r><w:t>日本</w:t></w:r><w:r><w:ruby><w:rt><w:r><w:t>かんじ</w:t></w:r></w:rt>"#,
        r#"<w:rubyBase><w:r><w:t>漢字</w:t></w:r></w:rubyBase></w:ruby></w:r><w:r><w:t>を読む</w:t></w:r></w:p>"#,
        r#"<w:p><w:r><w:t xml:space="preserve">Done </w:t><w:sym w:font="Wingdings" w:char="F0FC"/>"#,
        r#"<w:t xml:space="preserve"> next </w:t><w:sym w:font="Symbol" w:char="F0AE"/></w:r></w:p>"#,
        r#"<w:p><w:r><w:t>e</w:t><w:noBreakHyphen/><w:t>mail</w:t></w:r></w:p>"#,
    );
    assert_eq!(
        markdown(body, &[]),
        "日本漢字を読む\n\nDone ✓ next →\n\ne-mail"
    );
}

#[test]
fn wordart_text_is_kept() {
    let body = concat!(
        r##"<w:p><w:r><w:pict><v:shape xmlns:v="urn:schemas-microsoft-com:vml" type="#_x0000_t136">"##,
        r#"<v:textpath string="Grand Opening"/></v:shape></w:pict></w:r></w:p>"#,
        r#"<w:p><w:r><w:t>Body.</w:t></w:r></w:p>"#,
    );
    assert_eq!(markdown(body, &[]), "Grand Opening\n\nBody.");
}

#[test]
fn hidden_text_is_left_out_and_raised_text_keeps_its_meaning() {
    let body = concat!(
        r#"<w:p><w:r><w:t xml:space="preserve">Seen </w:t></w:r>"#,
        r#"<w:r><w:rPr><w:vanish/></w:rPr><w:t xml:space="preserve">unseen </w:t></w:r>"#,
        r#"<w:r><w:t>text.</w:t></w:r></w:p>"#,
        r#"<w:p><w:r><w:t>H</w:t></w:r><w:r><w:rPr><w:vertAlign w:val="subscript"/></w:rPr><w:t>2</w:t></w:r>"#,
        r#"<w:r><w:t xml:space="preserve">O at 5 × 10</w:t></w:r>"#,
        r#"<w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:t>-3</w:t></w:r></w:p>"#,
    );
    assert_eq!(markdown(body, &[]), "Seen text.\n\nH₂O at 5 × 10⁻³");
}

fn text(text: &str, bold: bool) -> Inline {
    Inline::Text {
        text: text.into(),
        style: anydoc::model::Style {
            bold,
            ..Default::default()
        },
    }
}

fn renderer() -> Renderer<'static> {
    Renderer {
        asset_names: &[],
        merged_cells: false,
        anchors: BTreeSet::new(),
        extension: "docx",
    }
}

#[test]
fn emphasis_beside_a_letter_keeps_its_punctuation_outside_the_markers() {
    // CommonMark opens a marker only before a character that is not
    // punctuation (unless one of whitespace or punctuation comes first) and
    // closes it only after one; in Chinese, with no spaces between words,
    // `**注意：**请` and `他说**“好”**然后` would show their asterisks.
    let renderer = renderer();
    let cases: [(Vec<Inline>, &str); 7] = [
        (
            vec![text("注意：", true), text("请核对。", false)],
            "**注意**：请核对。",
        ),
        (
            vec![
                text("他说", false),
                text("\u{201c}好\u{201d}", true),
                text("然后离开", false),
            ],
            "他说\u{201c}**好**\u{201d}然后离开",
        ),
        (
            vec![
                text("请看", false),
                text("（注意）", true),
                text("后续", false),
            ],
            "请看（**注意**）后续",
        ),
        // At the start of a paragraph the marker may open before punctuation.
        (
            vec![text("（注意）", true), text("后续", false)],
            "**（注意**）后续",
        ),
        // Beside whitespace or punctuation, or at an end, nothing moves.
        (
            vec![text("Note:", true), text(" text", false)],
            "**Note:** text",
        ),
        (
            vec![text("说：", false), text("“好", true), text("。", false)],
            "说：**“好**。",
        ),
        // Only punctuation: nothing to emphasise.
        (
            vec![text("好", false), text("…", true), text("吧", false)],
            "好…吧",
        ),
    ];
    for (values, expected) in cases {
        assert_eq!(renderer.inlines(&values), expected, "{values:?}");
    }
}

#[test]
fn review_comments_are_reported_as_left_out_of_the_markdown() {
    let comments = |count: usize| {
        let items: String = (0..count)
            .map(|id| {
                format!(
                    r#"<w:comment w:id="{id}" w:author="A"><w:p><w:r><w:t>Remark {id}</w:t></w:r></w:p></w:comment>"#
                )
            })
            .collect();
        (
            "word/comments.xml",
            format!("<w:comments {W}>{items}</w:comments>"),
        )
    };
    let body = paragraph("Reviewed text.");
    let warnings =
        |parts: &[(&str, String)]| extract(&docx(&body, parts), "docx").unwrap().warnings;
    assert!(warnings(&[]).is_empty());
    assert!(warnings(&[comments(0)]).is_empty());
    assert_eq!(
        warnings(&[comments(1)]),
        ["The document has 1 review comment; comments are not included in the Markdown."]
    );
    assert_eq!(
        warnings(&[comments(3)]),
        ["The document has 3 review comments; comments are not included in the Markdown."]
    );
    // The comment text itself stays out of the Markdown.
    assert_eq!(markdown(&body, &[comments(2)]), "Reviewed text.");
}

#[test]
fn embedded_html_rtf_and_web_archive_parts_are_read_and_others_reported() {
    // Report generators embed HTML and RTF with `w:altChunk` for Word to
    // convert on opening, and html-docx-js a web archive (MHT); a part in
    // Word's XML format is not converted and the warning says so.
    let chunk = |id: &str| {
        format!(
            r#"<w:altChunk xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:id="{id}"/>"#
        )
    };
    let body = [
        paragraph("Dear customer,"),
        chunk("html"),
        chunk("rtf"),
        chunk("mht"),
        chunk("xml"),
        paragraph("Kind regards."),
    ]
    .concat();
    let rels = (
        "word/_rels/document.xml.rels",
        [
            r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#,
            r#"<Relationship Id="html" Target="afchunk1.html" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/aFChunk"/>"#,
            r#"<Relationship Id="rtf" Target="afchunk2.rtf" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/aFChunk"/>"#,
            r#"<Relationship Id="mht" Target="afchunk3.mht" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/aFChunk"/>"#,
            r#"<Relationship Id="xml" Target="afchunk4.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/aFChunk"/>"#,
            "</Relationships>",
        ]
        .concat(),
    );
    let types = (
        "[Content_Types].xml",
        r#"<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
        <Default Extension="html" ContentType="text/html"/><Default Extension="rtf" ContentType="application/rtf"/>
        <Default Extension="mht" ContentType="message/rfc822"/>
        <Default Extension="xml" ContentType="application/xml"/></Types>"#
            .to_string(),
    );
    let html = (
        "word/afchunk1.html",
        "<html><body><h2>Your order</h2><p>We shipped <b>3 items</b> on Monday.<table>\
         <tr><th>Item<th>Price<tr><td>Lamp<td>12.00</table></body></html>"
            .to_string(),
    );
    let rtf = (
        "word/afchunk2.rtf",
        r"{\rtf1\ansi Questions? Call us.\par}".to_string(),
    );
    let mht = (
        "word/afchunk3.mht",
        "MIME-Version: 1.0\r\nContent-Type: multipart/related; type=\"text/html\"; \
         boundary=\"----=mhtDocumentPart\"\r\n\r\n------=mhtDocumentPart\r\n\
         Content-Type: text/html; charset=\"utf-8\"\r\nContent-Transfer-Encoding: quoted-printable\r\n\r\n\
         <p>Thank you for choosing our caf=C3=A9.</p>\r\n------=mhtDocumentPart--\r\n"
            .to_string(),
    );
    let xml = ("word/afchunk4.xml", "<w:document/>".to_string());
    let doc = extract(&docx(&body, &[rels, types, html, rtf, mht, xml]), "docx").unwrap();
    assert_eq!(
        doc.markdown,
        "Dear customer,\n\n## Your order\n\nWe shipped **3 items** on Monday.\n\n\
         | Item | Price |\n| --- | --- |\n| Lamp | 12.00 |\n\nQuestions? Call us.\n\n\
         Thank you for choosing our café.\n\nKind regards."
    );
    assert_eq!(
        doc.warnings,
        [
            "1 embedded part of the document (w:altChunk) is in a format that is not converted \
          (application/xml); its content is not in the Markdown."
        ]
    );
}

/// A run set in `font`.
fn run_in(font: &str, text: &str) -> String {
    format!(
        r#"<w:r><w:rPr><w:rFonts w:ascii="{font}" w:hAnsi="{font}"/></w:rPr><w:t xml:space="preserve">{text}</w:t></w:r>"#
    )
}

fn code_line(text: &str) -> String {
    format!("<w:p>{}</w:p>", run_in("Menlo", text))
}

#[test]
fn text_set_in_a_monospaced_font_is_code() {
    // Paragraphs all in Menlo, a blank one among them, are one code block; a
    // Consolas run in prose is inline code; a heading set in Courier is not.
    let body = [
        heading(1, &run_in("Courier New", "Setup")),
        format!(
            "<w:p>{}{}{}</w:p>",
            run_in("Times", "Run "),
            run_in("Consolas", "make_all"),
            run_in("Times", " first.")
        ),
        code_line("fn main() {"),
        code_line("    let x = 1;"),
        code_line(""),
        // A space in the body font and hidden text do not break a code line,
        // and a line that is all link is still one.
        format!(
            r#"<w:p>{}{}<w:r><w:rPr><w:vanish/></w:rPr><w:t>hidden</w:t></w:r>{}</w:p>"#,
            run_in("Menlo", "    let y"),
            run_in("Times", " "),
            run_in("Menlo", "= 2;")
        ),
        format!(
            r#"<w:p><w:hyperlink w:anchor="top">{}</w:hyperlink></w:p>"#,
            run_in("Menlo", "    go();")
        ),
        code_line("}"),
        paragraph("After."),
    ]
    .concat();
    assert_eq!(
        markdown(&body, &[styles()]),
        "# Setup\n\nRun `make_all` first.\n\n```\nfn main() {\n    let x = 1;\n\n    let y = 2;\n    go();\n}\n```\n\nAfter."
    );
}

#[test]
fn a_document_set_in_a_monospaced_font_is_not_code() {
    let body = [
        code_line("INT. KITCHEN - NIGHT"),
        code_line("She opens the door."),
        format!("<w:p>{}</w:p>", run_in("Times", "Page 1")),
    ]
    .concat();
    assert_eq!(
        markdown(&body, &[styles()]),
        "INT. KITCHEN - NIGHT\n\nShe opens the door.\n\nPage 1"
    );
}

#[test]
fn the_normal_style_decides_whether_a_monospaced_font_marks_code() {
    let styles = |defaults: &str, normal: &str| {
        (
            "word/styles.xml",
            format!(
                r#"<w:styles {W}><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="{defaults}"/>
                </w:rPr></w:rPrDefault></w:docDefaults>
                <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>
                <w:rPr><w:rFonts w:ascii="{normal}"/></w:rPr></w:style></w:styles>"#
            ),
        )
    };
    let body = [
        paragraph("Plain words in the body face."),
        format!(
            "<w:p>{}{}</w:p>",
            run_in("Menlo", "cargo"),
            run_in("Times", " builds.")
        ),
    ]
    .concat();
    // Over a monospaced default, the Normal style's face is the body's.
    assert_eq!(
        markdown(&body, &[styles("Courier New", "Times")]),
        "Plain words in the body face.\n\n`cargo` builds."
    );
    // A document whose Normal style is monospaced is set in it, not coded,
    // and so is one whose Normal style leaves a monospaced default alone.
    assert_eq!(
        markdown(&body, &[styles("Times", "Courier New")]),
        "Plain words in the body face.\n\ncargo builds."
    );
    let unset = styles("Courier New", "Times")
        .1
        .replace(r#"<w:rFonts w:ascii="Times"/>"#, "");
    assert_eq!(
        markdown(&body, &[("word/styles.xml", unset)]),
        "Plain words in the body face.\n\ncargo builds."
    );
}

#[test]
fn a_style_sets_the_font_of_runs_that_name_none() {
    let styles = format!(
        r#"<w:styles {W}>
        <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>
          <w:rPr><w:rFonts w:ascii="Times"/></w:rPr></w:style>
        <w:style w:type="paragraph" w:styleId="Listing"><w:name w:val="Listing"/>
          <w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Consolas"/></w:rPr></w:style>
        <w:style w:type="character" w:styleId="Code"><w:name w:val="Code"/>
          <w:rPr><w:rFonts w:hAnsi="Menlo"/></w:rPr></w:style>
        </w:styles>"#
    );
    let body = r#"<w:p><w:pPr><w:pStyle w:val="Listing"/></w:pPr><w:r><w:t>ls -la</w:t></w:r></w:p>
        <w:p><w:r><w:t xml:space="preserve">Call </w:t></w:r><w:r><w:rPr><w:rStyle w:val="Code"/></w:rPr><w:t>open()</w:t></w:r><w:r><w:t xml:space="preserve"> now, then read on in this sentence.</w:t></w:r></w:p>"#;
    assert_eq!(
        markdown(body, &[("word/styles.xml", styles)]),
        "```\nls -la\n```\n\nCall `open()` now, then read on in this sentence."
    );
}

#[test]
fn a_listing_loses_its_line_numbers_and_a_cell_keeps_code_inline() {
    let body = [
        paragraph("Example:"),
        code_line("1"),
        code_line("echo hello"),
        code_line("2"),
        code_line("exit 0"),
        table(&[format!(
            "<w:tr>{}{}</w:tr>",
            cell(&code_line("make")),
            cell(&paragraph("builds it"))
        )]),
    ]
    .concat();
    assert_eq!(
        markdown(&body, &[styles()]),
        "Example:\n\n```\necho hello\nexit 0\n```\n\n|  |  |\n| --- | --- |\n| `make` | builds it |"
    );
}

/// A paragraph of `cells` separated by tabs, at the tab stops `tabs` sets.
fn tab_row(tabs: &str, cells: &[&str]) -> String {
    let runs: Vec<String> = cells
        .iter()
        .map(|cell| format!(r#"<w:r><w:t xml:space="preserve">{cell}</w:t></w:r>"#))
        .collect();
    format!(
        "<w:p><w:pPr>{tabs}</w:pPr>{}</w:p>",
        runs.join("<w:r><w:tab/></w:r>")
    )
}

#[test]
fn columns_set_with_tab_stops_are_a_table_and_other_tabs_stay_spaces() {
    // Three rows at the stops the author set are a table, every row data as
    // in a Word table with no header row. A contents page (a dot leader),
    // a verse indented with a tab and a lone tab stay text, each tab a space.
    let stops =
        r#"<w:tabs><w:tab w:val="left" w:pos="3000"/><w:tab w:val="right" w:pos="6000"/></w:tabs>"#;
    let dots = r#"<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="8000"/></w:tabs>"#;
    let body = [
        paragraph("Price list follows."),
        tab_row(stops, &["Item", "Quantity", "Price"]),
        tab_row(stops, &["Apple", "3", "1.20"]),
        tab_row(stops, &["Banana", "12", "0.50"]),
        tab_row(dots, &["Introduction", "1"]),
        tab_row(dots, &["Results", "15"]),
        tab_row(dots, &["Discussion", "22"]),
        tab_row("", &["", "Whose woods these are I think I know."]),
        tab_row("", &["", "His house is in the village though;"]),
        tab_row("", &["", "He will not see me stopping here"]),
        tab_row("", &["A sentence with", "one tab inside it."]),
    ]
    .concat();
    assert_eq!(
        markdown(&body, &[]),
        "Price list follows.\n\n|  |  |  |\n| --- | --- | --- |\n| Item | Quantity | Price |\n| Apple | 3 | 1.20 |\n| Banana | 12 | 0.50 |\n\nIntroduction 1\n\nResults 15\n\nDiscussion 22\n\n Whose woods these are I think I know.\n\n His house is in the village though;\n\n He will not see me stopping here\n\nA sentence with one tab inside it."
    );
}

#[test]
fn a_listing_laid_out_in_a_table_is_a_code_block() {
    // A highlighter's table: a cell of line numbers, then the code.
    let body = [
        paragraph("Example:"),
        table(&[format!(
            "<w:tr>{}{}</w:tr>",
            cell(&[code_line("1"), code_line("2")].concat()),
            cell(&[code_line("let a = 1;"), code_line("let b = 2;")].concat())
        )]),
        paragraph("The prose of the document goes on in its own face for a while."),
    ]
    .concat();
    assert_eq!(
        markdown(&body, &[styles()]),
        "Example:\n\n```\nlet a = 1;\nlet b = 2;\n```\n\nThe prose of the document goes on in its own face for a while."
    );
}

/// A paragraph typed as a list item: its indent (`w:ind`, as `textutil`
/// writes it) and runs, `\t` pieces as tabs.
fn typed_item(ind: &str, rpr: &str, pieces: &[&str]) -> String {
    let runs: String = pieces
        .iter()
        .map(|piece| match *piece {
            "\t" => format!("<w:r>{rpr}<w:tab/></w:r>"),
            text if text.starts_with("<w:sym") => format!("<w:r>{rpr}{text}</w:r>"),
            text => format!(r#"<w:r>{rpr}<w:t xml:space="preserve">{text}</w:t></w:r>"#),
        })
        .collect();
    format!("<w:p><w:pPr>{ind}</w:pPr>{runs}</w:p>")
}

#[test]
fn lists_typed_by_hand_are_lists() {
    // As `textutil` saves an HTML list: a tab, the bullet, a tab, the text,
    // under a hanging indent that sets the level; an `ol` writes bare
    // numbers. A bullet line set in Menlo is an item with inline code, not
    // a code block. Word's Symbol and Wingdings bullets (`w:sym`) are
    // bullets; numbers before tabs that do not count up stay text.
    let level = |twips: u32| format!(r#"<w:ind w:left="{twips}" w:first-line="-{twips}"/>"#);
    let menlo = r#"<w:rPr><w:rFonts w:ascii="Menlo" w:hAnsi="Menlo"/></w:rPr>"#;
    let hanging = r#"<w:ind w:left="360" w:hanging="360"/>"#;
    let body = [
        paragraph("Features:"),
        typed_item(
            &level(720),
            "",
            &["", "\t", "•", "\t", "", "Written in Lua"],
        ),
        typed_item(&level(1440), "", &["\t", "◦", "\t", "Nested under it"]),
        typed_item(&level(720), "", &["\t", "•", "\t", "Fast"]),
        typed_item(&level(720), menlo, &["\t", "•", "\t", "npm test"]),
        paragraph("Steps:"),
        typed_item(&level(720), "", &["\t", "1", "\t", "Open the box"]),
        typed_item(&level(720), "", &["\t", "2", "\t", "Read the manual"]),
        paragraph("Symbols:"),
        typed_item(
            hanging,
            "",
            &[
                r#"<w:sym w:font="Symbol" w:char="F0B7"/>"#,
                "\t",
                "Symbol bullet",
            ],
        ),
        typed_item(
            hanging,
            "",
            &[
                r#"<w:sym w:font="Wingdings" w:char="F0A7"/>"#,
                "\t",
                "Wingdings square",
            ],
        ),
        paragraph("Stock:"),
        typed_item("", "", &["3", "\t", "Apples"]),
        typed_item("", "", &["12", "\t", "Pears"]),
    ]
    .concat();
    assert_eq!(
        markdown(&body, &[]),
        "Features:\n\n* Written in Lua\n  \n  * Nested under it\n* Fast\n* `npm test`\n\n\
         Steps:\n\n1. Open the box\n2. Read the manual\n\n\
         Symbols:\n\n* Symbol bullet\n* Wingdings square\n\n\
         Stock:\n\n3 Apples\n\n12 Pears"
    );
}

#[test]
fn a_list_typed_by_hand_in_a_word_97_file_is_a_list() {
    // The Word 97 exporter writes an HTML list as typed bullets, as it does
    // in Word documents.
    let doc = extract(include_bytes!("fixtures/textedit-word97.doc"), "doc").unwrap();
    assert!(
        doc.markdown
            .contains("system.\n\n- First listed point\n- Second listed point\n\nThe closing"),
        "{}",
        doc.markdown
    );
    // An `ol` starting at 2 (`<Tab>2<Tab>`: a lone number is an item only
    // before a tab) and a nested `ul` (`<Tab>◦<Tab>` under a deeper hanging
    // indent).
    let doc = extract(include_bytes!("fixtures/textedit-word97-lists.doc"), "doc").unwrap();
    assert!(
        doc.markdown.contains(
            "between tabs.\n\n2. Second step on its own\n\nThe parts of the bicycle:\n\n\
             - Frame\n  \n  - Front wheel\n- Saddle\n\nThe closing"
        ),
        "{}",
        doc.markdown
    );
}

#[test]
fn the_mark_a_word_97_file_keeps_for_a_picture_it_did_not_store_is_not_text() {
    // The macOS exporter writes U+FFFC where `<img>` was and keeps no picture
    // data (no Data stream): the paragraph is empty, not a stray character.
    let bytes = include_bytes!("fixtures/textedit-word97-picture.doc");
    let doc = extract(bytes, "doc").unwrap();
    assert_eq!(doc.markdown, "Before the picture.\n\nAfter the picture.\n");
    assert!(doc.assets.is_empty());
}

fn listed_paragraph(numbered: Option<(u32, u8)>, ppr: &str, text: &str) -> String {
    let num = numbered.map_or(String::new(), |(id, level)| {
        format!(r#"<w:numPr><w:ilvl w:val="{level}"/><w:numId w:val="{id}"/></w:numPr>"#)
    });
    format!(r#"<w:p><w:pPr>{ppr}{num}</w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>"#)
}

#[test]
fn a_paragraph_lined_up_with_an_items_text_continues_the_item() {
    // Word's numbering sets each level's text indent; a paragraph after an
    // item set in as far (Enter, then Backspace, in a List Paragraph) is
    // more of that item, after its nested list when it is back at the outer
    // text; body text at the margin ends the list. pandoc numbers an item's
    // later paragraphs with a bullet of one space, which shows nothing.
    let numbering = (
        "word/numbering.xml",
        format!(
            r#"<w:numbering {W}><w:abstractNum w:abstractNumId="0">
            <w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/>
              <w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl>
            <w:lvl w:ilvl="1"><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/>
              <w:pPr><w:ind w:left="1440" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>
            <w:abstractNum w:abstractNumId="9"><w:lvl w:ilvl="0"><w:numFmt w:val="bullet"/>
              <w:lvlText w:val=" "/><w:pPr><w:ind w:left="720" w:hanging="480"/></w:pPr></w:lvl></w:abstractNum>
            <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
            <w:num w:numId="1000"><w:abstractNumId w:val="9"/></w:num></w:numbering>"#
        ),
    );
    let styles = (
        "word/styles.xml",
        format!(
            r#"<w:styles {W}><w:style w:type="paragraph" w:styleId="ListParagraph">
            <w:name w:val="List Paragraph"/><w:pPr><w:ind w:left="720"/></w:pPr></w:style></w:styles>"#
        ),
    );
    let list = r#"<w:pStyle w:val="ListParagraph"/>"#;
    let body = [
        paragraph("Before the list."),
        listed_paragraph(Some((1, 0)), list, "One."),
        listed_paragraph(None, list, "More of one."),
        listed_paragraph(Some((1, 0)), list, "Two."),
        listed_paragraph(Some((1, 1)), list, "Two a."),
        listed_paragraph(None, r#"<w:ind w:left="1440"/>"#, "More of two a."),
        "<w:p/>".to_string(),
        listed_paragraph(None, r#"<w:ind w:left="720"/>"#, "More of two."),
        listed_paragraph(Some((1, 0)), "", "Three."),
        listed_paragraph(Some((1000, 0)), "", "More of three, as pandoc writes it."),
        paragraph("After the list."),
    ]
    .concat();
    assert_eq!(
        markdown(&body, &[numbering, styles]),
        "Before the list.\n\n1. One.\n   \n   More of one.\n2. Two.\n   \n   * Two a.\n     \n     \
         More of two a.\n   \n   More of two.\n3. Three.\n   \n   More of three, as pandoc writes it.\n\n\
         After the list."
    );
}
