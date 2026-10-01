//! OpenDocument text and RTF end to end: what the vendored readers recover
//! (see `vendor/anydoc/MARKITAI-PATCH.md`) as the Markdown and warnings a
//! caller receives.

use super::*;
use std::io::{Cursor, Write};

fn odt(parts: &[(&str, &str)]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, content) in parts {
        writer
            .start_file(*name, zip::write::SimpleFileOptions::default())
            .unwrap();
        writer.write_all(content.as_bytes()).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

const NAMESPACES: &str = r#"xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
    xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
    xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
    xmlns:chart="urn:oasis:names:tc:opendocument:xmlns:chart:1.0"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    xmlns:dc="http://purl.org/dc/elements/1.1/""#;

#[test]
fn odt_comments_stay_out_with_a_warning_and_a_chart_reads_as_its_data() {
    let content = format!(
        r#"<office:document-content {NAMESPACES}>
        <office:automatic-styles><style:style style:name="Sup" style:family="text">
          <style:text-properties style:text-position="super 58%"/></style:style>
        </office:automatic-styles>
        <office:body><office:text>
        <text:p>An essay<text:span text:style-name="Sup">1</text:span> on
          <office:annotation><dc:creator>Reviewer</dc:creator><text:p>Cite this</text:p></office:annotation>std::vector&lt;int&gt;.</text:p>
        <text:p>Sales by region:<draw:frame><draw:object xlink:href="./Object 1"/></draw:frame></text:p>
        </office:text></office:body></office:document-content>"#
    );
    let chart = format!(
        r#"<office:document-content {NAMESPACES}><office:body><office:chart><chart:chart>
        <table:table><table:table-header-rows><table:table-row>
          <table:table-cell/><table:table-cell><text:p>2025</text:p></table:table-cell>
        </table:table-row></table:table-header-rows>
        <table:table-rows><table:table-row>
          <table:table-cell><text:p>East</text:p></table:table-cell>
          <table:table-cell office:value-type="float" office:value="19.2"><text:p>19.2</text:p></table:table-cell>
        </table:table-row></table:table-rows></table:table>
        </chart:chart></office:chart></office:body></office:document-content>"#
    );
    let doc = extract(
        &odt(&[("content.xml", &content), ("Object 1/content.xml", &chart)]),
        "odt",
    )
    .unwrap();
    assert_eq!(
        doc.markdown,
        "An essay¹ on std::vector\\<int>.\n\nSales by region:\n\n|  | 2025 |\n| --- | --- |\n| East | 19.2 |\n"
    );
    assert_eq!(
        doc.warnings,
        ["The document has 1 review comment; comments are not included in the Markdown."]
    );
}

#[test]
fn textedit_rtf_reads_like_the_same_page_saved_as_docx() {
    // TextEdit on a Chinese system: `\ansicpg936` over charset-0 fonts whose
    // bytes are Windows-1252; a comment thread nested two tables deep and
    // closed without `\itap2`; a link card spanning paragraphs; a list in a
    // table cell; a footnote mark raised with `\super`.
    let rtf = "{\\rtf1\\ansi\\ansicpg936{\\fonttbl\\f0\\froman\\fcharset0 Times-Roman;}\
{\\*\\listtable{\\list\\listtemplateid1{\\listlevel\\levelnfc23{\\leveltext\\'01\\uc0\\u8226 ;}{\\levelnumbers;}}\\listid1}}\
{\\*\\listoverridetable{\\listoverride\\listid1\\listoverridecount0\\ls1}}\n\
\\f0 Apple\\'92s phone costs \\'bd as much.\n\\fs20 \\super 1\n\\fs24 \\nosupersub \\\n\
\\itap1\\trowd\\cellx8640\n\
\\itap2\\trowd\\cellx8640\n\
\\itap3\\trowd\\cellx4320\\cellx8640\n\
\\pard\\intbl\\itap3 \\nestcell\n\
\\pard\\intbl\\itap3 commenter_one 2 hours ago\\\n\\pard\\intbl\\itap3 I agree.\\nestcell \\lastrow\\nestrow\\nestcell \\lastrow\\nestrow\\cell \\lastrow\\row\n\
\\pard {\\field{\\*\\fldinst{HYPERLINK \"https://example.com/related\"}}{\\fldrslt \\pard \\\n\\pard example.com Related Project\\\n}}\n\
\\pard\\trowd\\cellx8640\n\
\\pard\\intbl Changes:\\\n\
\\ls1\\ilvl0 {\\listtext \\uc0\\u8226 }Preserve linked text\\\n\
\\ls1\\ilvl0 {\\listtext \\uc0\\u8226 }Add a test\\cell \\lastrow\\row\n}";
    let doc = extract(rtf.as_bytes(), "rtf").unwrap();
    assert_eq!(
        doc.markdown,
        "Apple’s phone costs ½ as much.¹\n\n\
         commenter\\_one 2 hours ago\n\nI agree.\n\n\
         [example.com Related Project](https://example.com/related)\n\n\
         Changes:\n\n* Preserve linked text\n* Add a test\n"
    );
}
