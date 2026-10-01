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
