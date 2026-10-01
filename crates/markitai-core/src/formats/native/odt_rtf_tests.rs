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
fn odt_code_set_in_a_monospaced_font_and_columns_set_with_tab_stops() {
    // As `textutil` saves a web page: faces named by `style:font-name` (here
    // a generated face name, its family declared apart), a listing one
    // paragraph per line, inline code in prose, a highlighter's
    // table of line numbers and code. Tab-set columns with a bold first
    // row are a table with that row as its header.
    let content = format!(
        r#"<office:document-content {NAMESPACES}
        xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
        xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0">
        <office:font-face-decls><style:font-face style:name="F2" svg:font-family="Courier"/>
          <style:font-face style:name="Times" svg:font-family="Times"/></office:font-face-decls>
        <office:automatic-styles>
          <style:style style:name="P1" style:family="paragraph">
            <style:text-properties style:font-name="Times"/></style:style>
          <style:style style:name="P3" style:family="paragraph">
            <style:text-properties style:font-name="F2"/></style:style>
          <style:style style:name="P4" style:family="paragraph"><style:paragraph-properties>
            <style:tab-stops><style:tab-stop style:position="2in"/>
            <style:tab-stop style:position="4in" style:type="char" style:char="."/></style:tab-stops>
            </style:paragraph-properties></style:style>
          <style:style style:name="T1" style:family="text">
            <style:text-properties fo:font-weight="bold"/></style:style>
          <style:style style:name="T3" style:family="text">
            <style:text-properties style:font-name="F2"/></style:style>
        </office:automatic-styles>
        <office:body><office:text>
        <text:p text:style-name="P1">This script uses the <text:span text:style-name="T3">read</text:span><text:s/>command:</text:p>
        <text:p text:style-name="P3">#!/bin/bash</text:p>
        <text:p text:style-name="P3"><text:span/></text:p>
        <text:p text:style-name="P3"><text:s text:c="2"/>read -p "Name: " name</text:p>
        <table:table><table:table-row><table:table-cell><text:p text:style-name="P3">1</text:p>
          <text:p text:style-name="P3">2</text:p></table:table-cell><table:table-cell>
          <text:p text:style-name="P3">echo "$name"</text:p><text:p text:style-name="P3">exit 0</text:p>
        </table:table-cell></table:table-row></table:table>
        <text:p text:style-name="P4"><text:span text:style-name="T1">Item<text:tab/>Qty<text:tab/>Price</text:span></text:p>
        <text:p text:style-name="P4">Apple<text:tab/>3<text:tab/>1.20</text:p>
        <text:p text:style-name="P4">Pear<text:tab/>12<text:tab/>0.50</text:p>
        <text:p text:style-name="P1">That is all for this page, set in the body face as text is.</text:p>
        </office:text></office:body></office:document-content>"#
    );
    let doc = extract(&odt(&[("content.xml", &content)]), "odt").unwrap();
    assert_eq!(
        doc.markdown,
        "This script uses the `read` command:\n\n```\n#!/bin/bash\n\n  read -p \"Name: \" name\n```\n\n```\necho \"$name\"\nexit 0\n```\n\n| **Item** | **Qty** | **Price** |\n| --- | --- | --- |\n| Apple | 3 | 1.20 |\n| Pear | 12 | 0.50 |\n\nThat is all for this page, set in the body face as text is.\n"
    );
}

#[test]
fn rtf_code_set_in_a_monospaced_font_and_columns_set_with_tab_stops() {
    // TextEdit's font table (PostScript names, one font after another) and
    // Word's tab stops (`\tx`, a right-aligned `\tqr`); a heading set in the
    // code face is no code, and a contents line with a dot leader is text.
    let rtf = "{\\rtf1\\ansi{\\fonttbl\\f0\\froman\\fcharset0 Times-Roman;\\f1\\fnil\\fcharset0 Menlo-Regular;}\n\
{\\stylesheet{\\s1\\outlinelevel0 heading 1;}}\n\
\\pard\\s1\\outlinelevel0\\f1 Usage\\par\n\
\\pard\\f0 Run \\f1 make\\f0  first:\\par\n\
\\pard\\f1 make all\\par\n\
make install\\par\n\
\\pard\\tx2880\\tqr\\tx5760\\f0 Item\\tab Qty\\tab Price\\par\n\
\\pard\\tx2880\\tqr\\tx5760 Apple\\tab 3\\tab 1.20\\par\n\
\\pard\\tx2880\\tqr\\tx5760 Pear\\tab 12\\tab 0.50\\par\n\
\\pard\\tqr\\tldot\\tx8640 Contents\\tab 1\\par\n\
\\pard That is all for this page, set in the body face as text is.\\par}";
    let doc = extract(rtf.as_bytes(), "rtf").unwrap();
    assert_eq!(
        doc.markdown,
        "# Usage\n\nRun `make` first:\n\n```\nmake all\nmake install\n```\n\n|  |  |  |\n| --- | --- | --- |\n| Item | Qty | Price |\n| Apple | 3 | 1.20 |\n| Pear | 12 | 0.50 |\n\nContents 1\n\nThat is all for this page, set in the body face as text is.\n"
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

#[test]
fn odt_lists_typed_by_hand_are_lists() {
    // Indents from the paragraph styles' margins set the levels; a run of
    // dashes that reads as dialogue stays text.
    let content = format!(
        r#"<office:document-content {NAMESPACES}
        xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0">
        <office:automatic-styles>
          <style:style style:name="L1" style:family="paragraph"><style:paragraph-properties
            fo:margin-left="0.5in" fo:text-indent="-0.25in"/></style:style>
          <style:style style:name="L2" style:family="paragraph" style:parent-style-name="L1">
            <style:paragraph-properties fo:margin-left="1in"/></style:style>
        </office:automatic-styles>
        <office:body><office:text>
        <text:p>Packing list:</text:p>
        <text:p text:style-name="L1">•<text:tab/>Clothes</text:p>
        <text:p text:style-name="L2">–<text:tab/>shirts</text:p>
        <text:p text:style-name="L1">•<text:tab/>Books</text:p>
        <text:p>Then they talked.</text:p>
        <text:p>– Are you ready?</text:p>
        <text:p>– Almost!</text:p>
        </office:text></office:body></office:document-content>"#
    );
    let doc = extract(&odt(&[("content.xml", &content)]), "odt").unwrap();
    assert_eq!(
        doc.markdown,
        "Packing list:\n\n- Clothes\n  \n  - shirts\n- Books\n\nThen they talked.\n\n– Are you ready?\n\n– Almost!\n"
    );
}

#[test]
fn rtf_lists_typed_by_hand_are_lists() {
    // `\li` and `\fi` set the levels; `\bullet` is a bullet, and so are the
    // bytes of Word's Wingdings square (`\'a7`) and arrowhead (`\'d8`),
    // which are not `§` and `Ø`. Lowercase letters counting up keep their
    // labels.
    let rtf = "{\\rtf1\\ansi{\\fonttbl{\\f0\\froman\\fcharset0 Times New Roman;}\
{\\f1\\fnil\\fcharset2 Wingdings;}}\n\
\\pard\\f0 Options:\\par\n\
\\pard\\li360\\fi-360 \\bullet\\tab Fast\\par\n\
\\pard\\li1080\\fi-360 {\\f1\\'a7}\\tab cached\\par\n\
\\pard\\li360\\fi-360 {\\f1\\'d8}\\tab Cheap\\par\n\
\\pard Choose:\\par\n\
\\pard a) the first\\par\n\
\\pard b) the second\\par}";
    let doc = extract(rtf.as_bytes(), "rtf").unwrap();
    assert_eq!(
        doc.markdown,
        "Options:\n\n* Fast\n  \n  * cached\n* Cheap\n\nChoose:\n\na) the first  \nb) the second\n"
    );
}
