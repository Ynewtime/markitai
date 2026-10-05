//! Web page and EPUB text through the whole local conversion, normal output
//! cleanup included: what a page's forms, footers, line breaks, task lists,
//! ruby and definition lists become.

use super::options;
use markitai_core::convert;
use std::io::Write;

#[test]
fn a_page_s_forms_footers_breaks_and_tasks_survive_normal_output() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("page.html");
    let prose = "The article body has plenty of words so that extraction keeps it. ".repeat(3);
    std::fs::write(
        &input,
        format!(
            r#"<!DOCTYPE html><html><head><meta charset="utf-8"><title>Page</title></head><body>
            <form id="aspnetForm" method="post"><input type="hidden" name="__VIEWSTATE" value="x">
            <article><h1>Page</h1><p>{prose}</p>
            <p>Poem:<br>
            Roses are red,<br>
            Violets are blue.</p>
            <ul><li><input type="checkbox" checked disabled> Done</li><li><input type="checkbox" disabled> Open</li></ul>
            <blockquote><p>Be the change.</p><footer>Mahatma Gandhi</footer></blockquote>
            <p><ruby>漢<rt>kan</rt>字<rt>ji</rt></ruby> and so&shy;ft</p>
            <table><tr><th>Name</th><th>Notes</th></tr><tr><td>A | B</td><td><ul><li>x</li><li>y</li></ul></td></tr></table>
            <dl><dt>Term</dt><dd>Definition text</dd></dl>
            </article><footer>Copyright 2026</footer></form></body></html>"#
        ),
    )
    .unwrap();
    let markdown = convert(input.to_str().unwrap(), options())
        .unwrap()
        .markdown;
    for expected in [
        "plenty of words",
        "Poem:  \nRoses are red,  \nViolets are blue.",
        "* [x] Done\n* [ ] Open",
        "> Be the change.\n>\n> — Mahatma Gandhi",
        "漢字(kanji) and soft",
        "| A \\| B | * x<br>* y |",
        "**Term**\n\nDefinition text",
    ] {
        assert!(markdown.contains(expected), "{expected}: {markdown}");
    }
    assert!(!markdown.contains("Copyright"), "{markdown}");
}

#[test]
fn a_page_s_line_breaks_are_two_space_hard_breaks_in_normal_output() {
    // Shaped like a review page: a `<br>` after a closing quote, after a
    // colon, after blanks, doubled to space paragraphs, at a block's end,
    // and in a list item and a quotation.
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("review.html");
    let prose = "评论正文有足够多的文字，正文提取会保留它。".repeat(8);
    std::fs::write(
        &input,
        format!(
            r#"<!DOCTYPE html><html><head><meta charset="utf-8"><title>书评</title></head><body>
            <article><h1>书评</h1><div class="review-content"><p>{prose}</p>
            <p>“叙述克制，绝不浮夸。&nbsp; ”<br>
            我最迷恋小说的部分是一开头：<br>第一句。 &nbsp; <br><br>第二段。<br></p>
            <ul><li>一<br>二</li></ul><blockquote>甲<br>乙</blockquote>
            <p>{prose}</p></div></article></body></html>"#
        ),
    )
    .unwrap();
    let markdown = convert(input.to_str().unwrap(), options())
        .unwrap()
        .markdown;
    for expected in [
        "“叙述克制，绝不浮夸。\u{a0} ”  \n我最迷恋小说的部分是一开头：  \n第一句。\n\n第二段。\n",
        "* 一  \n  二",
        "> 甲  \n> 乙",
    ] {
        assert!(markdown.contains(expected), "{expected:?}: {markdown:?}");
    }
    assert!(!markdown.contains("\\\n"), "{markdown:?}");
    let lines: Vec<&str> = markdown.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        // No line of blanks only, and a hard break only where a line follows.
        assert!(line.is_empty() || !line.trim().is_empty(), "{markdown:?}");
        if line.ends_with(' ') {
            assert!(
                line.ends_with("  ") && !line.ends_with("   "),
                "{markdown:?}"
            );
            assert!(
                lines.get(index + 1).is_some_and(|next| !next.is_empty()),
                "{markdown:?}"
            );
        }
    }
}

#[test]
fn an_epub_s_ruby_definition_lists_and_written_footnote_marks_read_as_text() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("book.epub");
    let mut archive = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, body) in [
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
            "<html xmlns='http://www.w3.org/1999/xhtml'><body><h1>Chapter</h1><p>Text about <ruby>漢<rt>kan</rt>字<rt>ji</rt></ruby>.</p><dl><dt>Term</dt><dd>Definition text</dd></dl><p>A claim.[^5] More.</p><p>[^5]: The note.</p><p><code>[^6]</code></p></body></html>",
        ),
    ] {
        archive
            .start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        archive.write_all(body.as_bytes()).unwrap();
    }
    std::fs::write(&input, archive.finish().unwrap().into_inner()).unwrap();
    let markdown = convert(input.to_str().unwrap(), options())
        .unwrap()
        .markdown;
    for expected in [
        "Text about 漢字(kanji).",
        "**Term**\n\nDefinition text",
        "A claim.[^5] More.",
        "[^5]: The note.",
        "`[^6]`",
    ] {
        assert!(markdown.contains(expected), "{expected}: {markdown}");
    }
    assert!(!markdown.contains("\\[^"), "{markdown}");
}
