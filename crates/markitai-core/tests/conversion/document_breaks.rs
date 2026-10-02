//! Word, OpenDocument, RTF and EPUB line breaks and escaping through the
//! whole local conversion, normal output cleanup included: a manual line
//! break stays a hard break, two end the paragraph, and text keeps
//! characters that are not Markdown where they stand.

use super::options;
use markitai_core::convert;
use std::io::Write;

fn package(entries: &[(&str, &str)]) -> Vec<u8> {
    let mut archive = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, body) in entries {
        archive
            .start_file(*name, zip::write::SimpleFileOptions::default())
            .unwrap();
        archive.write_all(body.as_bytes()).unwrap();
    }
    archive.finish().unwrap().into_inner()
}

fn markdown(name: &str, bytes: &[u8]) -> String {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join(name);
    std::fs::write(&input, bytes).unwrap();
    convert(input.to_str().unwrap(), options())
        .unwrap()
        .markdown
}

const EXPECTED: [&str; 4] = [
    "Poem:\\\nRoses are red,\\\nViolets are blue.",
    "One\n\nTwo",
    "Use [!tip] in snake_case files, 2 * 3.",
    "Price\\\n\\# not a heading",
];

#[test]
fn a_word_document_s_line_breaks_and_text_survive_normal_output() {
    let run = |text: &str| format!(r#"<w:r><w:t xml:space="preserve">{text}</w:t></w:r>"#);
    let br = "<w:r><w:br/></w:r>";
    let body = [
        format!(
            "<w:p>{}{br}{}{br}{}</w:p>",
            run("Poem:"),
            run("Roses are red,"),
            run("Violets are blue.")
        ),
        format!("<w:p>{}{br}{br}{}</w:p>", run("One"), run("Two")),
        format!(
            "<w:p>{}</w:p>",
            run("Use [!tip] in snake_case files, 2 * 3.")
        ),
        format!("<w:p>{}{br}{}</w:p>", run("Price"), run("# not a heading")),
    ]
    .concat();
    let document = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>"#
    );
    let output = markdown("breaks.docx", &package(&[("word/document.xml", &document)]));
    for expected in EXPECTED {
        assert!(output.contains(expected), "{expected:?}: {output}");
    }
}

#[test]
fn odt_rtf_and_epub_line_breaks_survive_normal_output() {
    let odt = package(&[
        ("mimetype", "application/vnd.oasis.opendocument.text"),
        (
            "content.xml",
            r#"<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text>
            <text:p>Poem:<text:line-break/>Roses are red,<text:line-break/>Violets are blue.</text:p>
            <text:p>One<text:line-break/><text:line-break/>Two</text:p>
            <text:p>Use [!tip] in snake_case files, 2 * 3.</text:p>
            <text:p>Price<text:line-break/># not a heading</text:p>
            </office:text></office:body></office:document-content>"#,
        ),
    ]);
    let rtf = br"{\rtf1\ansi Poem:\line Roses are red,\line Violets are blue.\par One\line\line Two\par Use [!tip] in snake_case files, 2 * 3.\par Price\line # not a heading\par}";
    let epub = package(&[
        ("mimetype", "application/epub+zip"),
        (
            "META-INF/container.xml",
            "<container><rootfiles><rootfile full-path='OEBPS/content.opf'/></rootfiles></container>",
        ),
        (
            "OEBPS/content.opf",
            "<package><metadata><title>Book</title></metadata><manifest><item id='c1' href='c1.xhtml' media-type='application/xhtml+xml'/></manifest><spine><itemref idref='c1'/></spine></package>",
        ),
        (
            "OEBPS/c1.xhtml",
            "<html xmlns='http://www.w3.org/1999/xhtml'><body><h1>Chapter</h1><p>Poem:<br/>Roses are red,<br/>Violets are blue.</p><p>One<br/><br/>Two</p><p>Use [!tip] in snake_case files, 2 * 3.</p><p>Price<br/># not a heading</p></body></html>",
        ),
    ]);
    for (name, bytes) in [
        ("breaks.odt", odt.as_slice()),
        ("breaks.rtf", rtf.as_slice()),
        ("breaks.epub", epub.as_slice()),
    ] {
        let output = markdown(name, bytes);
        for expected in EXPECTED {
            assert!(output.contains(expected), "{name}: {expected:?}: {output}");
        }
    }
}
