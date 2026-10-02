//! Line breaks and escaping in the document renderer: what each case reads
//! as in CommonMark is noted beside it (checked against a CommonMark parser
//! when the cases were written; see `line.rs`).

use super::*;
use anydoc::model::{Cell, Style, Table, TableKind};

fn renderer(extension: &'static str) -> Renderer<'static> {
    Renderer {
        asset_names: &[],
        merged_cells: false,
        in_cell: false,
        anchors: BTreeSet::new(),
        extension,
    }
}

fn styled(text: &str, apply: impl Fn(&mut Style)) -> Inline {
    let mut style = Style::PLAIN;
    apply(&mut style);
    Inline::Text {
        text: text.into(),
        style,
    }
}

fn bold(text: &str) -> Inline {
    styled(text, |style| style.bold = true)
}

fn link(content: Vec<Inline>) -> Inline {
    Inline::Link {
        content,
        target: LinkTarget::External("https://example.test/x".into()),
    }
}

fn paragraph(values: Vec<Inline>) -> String {
    renderer("docx").blocks(&[Block::Paragraph(values)])
}

fn text(value: &str) -> String {
    paragraph(vec![Inline::plain(value)])
}

#[test]
fn text_that_cannot_be_syntax_where_it_stands_is_written_as_it_is() {
    for literal in [
        "[!tip] Use a fence",
        "Note: [!NOTE] is only text in a paragraph",
        "snake_case, txt2_0 and 00_inbox/",
        r"C:\Users\me and \displaystyle",
        "a * b, 2 * 3 and x*",
        "Price* applies; *Terms below",
        "Name: ________ Date: ________",
        "~5 km, or ~10",
        "a ` lone backtick",
        "see [1] and (2), [a] [b]",
        "a]( without an opening bracket",
        "a < b, x -> y, a<3",
        "AT&T, x&y, &#, &; and a & b",
        "[^ x] and a[^]",
        "C# and F#, issue #12",
        "1+1=2 | not a table",
        "Wow! Look.",
    ] {
        assert_eq!(text(literal), literal, "{literal:?}");
    }
}

#[test]
fn text_that_would_be_syntax_is_escaped() {
    for (literal, written) in [
        // Emphasis needs an opener and a later closer.
        ("5*3 and 2*4", r"5\*3 and 2\*4"),
        ("__init__", r"\_\_init\_\_"),
        ("*not emphasis*", r"\*not emphasis\*"),
        ("~~not struck~~", r"\~\~not struck\~\~"),
        // A code span needs a later backtick run of the same length: the last
        // run of each length has none and stays.
        ("Use `x` here", r"Use \`x` here"),
        ("``a`` and `b", r"\`\`a`` and `b"),
        // An inline link or image, and a footnote mark.
        ("[1](2) and ![a](b)", r"[1\](2) and !\[a\](b)"),
        ("a[^1] mark", r"a\[^1] mark"),
        // Raw HTML and character references.
        ("std::vector<int> v", r"std::vector\<int> v"),
        ("x <y and z> w", r"x \<y and z> w"),
        ("&copy; &#169; &#xA9;", r"\&copy; \&#169; \&#xA9;"),
        // A backslash before punctuation or at the end.
        (r"a\*b", r"a\\*b"),
        (r"C:\", r"C:\\"),
        // Blanks after a backslash are trimmed with the line's end.
        ("a-\\ ", r"a-\\"),
    ] {
        assert_eq!(text(literal), written, "{literal:?}");
    }
}

#[test]
fn a_line_that_starts_like_a_block_is_text() {
    for (literal, written) in [
        ("* item", r"\* item"),
        // A delimiter mark is escaped whole: the rest of the run could still
        // pair with one elsewhere in the paragraph.
        ("***", r"\*\*\*"),
        ("_ _ _", r"\_ \_ \_"),
        ("```", r"\`\`\`"),
        ("~~~ code", r"\~\~\~ code"),
        ("---", r"\---"),
        ("[1]: https://example.test", r"\[1]: https://example.test"),
        ("<div class=x", r"\<div class=x"),
        ("<!-- not a comment", r"\<!-- not a comment"),
        // Literal backticks that would be a code span are escaped first.
        ("``` a``b ```", r"\`\`\` a``b ```"),
    ] {
        assert_eq!(text(literal), written, "{literal:?}");
    }
    // After a hard break a line can still start a block, and a setext
    // underline or a delimiter row would change the line above.
    for (second, written) in [
        ("# heading", r"\# heading"),
        ("- item", r"\- item"),
        ("===", r"\==="),
        ("--", r"\--"),
        ("| --- | --- |", r"\| --- | --- |"),
        ("<p>", r"\<p>"),
    ] {
        assert_eq!(
            paragraph(vec![
                Inline::plain("Line"),
                Inline::LineBreak,
                Inline::plain(second),
            ]),
            format!("Line\\\n{written}"),
            "{second:?}"
        );
    }
}

#[test]
fn text_beside_the_renderer_s_own_markup_is_escaped_where_it_would_join_it() {
    let cases: [(Vec<Inline>, &str); 8] = [
        // A literal marker inside emphasis, or touching its markers.
        (vec![bold("*x")], r"**\*x**"),
        (vec![bold("a*b")], r"**a\*b**"),
        (vec![bold("a * b")], "**a * b**"),
        (vec![Inline::plain("x*"), bold("y")], r"x\***y**"),
        (vec![styled("~a", |style| style.strike = true)], r"~~\~a~~"),
        // `!` before a link would make it an image; normal output's image
        // repairs would read `\![` as one too.
        (
            vec![Inline::plain("Wow!"), link(vec![Inline::plain("site")])],
            "Wow&#33;[site](https://example.test/x)",
        ),
        // A link's text keeps its brackets as text.
        (
            vec![link(vec![Inline::plain("a [b] c")])],
            r"[a \[b\] c](https://example.test/x)",
        ),
        // Outside emphasis a literal `*` cannot take the renderer's markers.
        (
            vec![Inline::plain("5*3 is "), bold("fifteen")],
            "5*3 is **fifteen**",
        ),
    ];
    for (values, written) in cases {
        assert_eq!(paragraph(values.clone()), written, "{values:?}");
    }
}

#[test]
fn cases_a_commonmark_parser_read_wrongly_before_their_fixes() {
    // Each was found by parsing generated documents' output: the parser's
    // reading must be the document's text, emphasised only where it is.
    let cases: [(Vec<Inline>, &str); 4] = [
        // Markers between two punctuation marks can both open and close: as a
        // closer, the bold run's opener would take the literal `**` before it.
        (
            vec![
                Inline::plain("x **^/ a#."),
                bold("|.>"),
                Inline::plain("[)"),
            ],
            r"x \*\*^/ a#.**|.>**[)",
        ),
        // The same, with literal asterisks escaped beside the opener.
        (
            vec![Inline::plain("x **`x`"), bold("** ;[")],
            r"x \*\*\`x`**\*\* ;[**",
        ),
        // Punctuation moved out of emphasis must not leave a space inside it.
        (
            vec![Inline::plain("x .  "), bold("< |"), Inline::plain("b")],
            r"x .  **<** |b",
        ),
        // A rule's whole run is escaped, or its rest would close the
        // underscores on the line above (which then have nothing to pair
        // with and stay).
        (
            vec![
                Inline::plain("x ____[a"),
                Inline::LineBreak,
                Inline::plain("____"),
            ],
            "x ____[a\\\n\\_\\_\\_\\_",
        ),
    ];
    for (values, written) in cases {
        assert_eq!(paragraph(values.clone()), written, "{values:?}");
    }
}

#[test]
fn a_line_break_is_a_hard_break_a_paragraph_break_or_a_space() {
    let poem = paragraph(vec![
        Inline::plain("Poem:"),
        Inline::LineBreak,
        Inline::plain("Roses are red, "),
        Inline::LineBreak,
        Inline::plain("   Violets are blue."),
    ]);
    assert_eq!(poem, "Poem:\\\nRoses are red,\\\nViolets are blue.");
    // Two breaks end the paragraph; breaks at the edges show nothing, and the
    // indentation after a break is not a code block.
    assert_eq!(
        paragraph(vec![
            Inline::LineBreak,
            Inline::plain("One"),
            Inline::LineBreak,
            Inline::plain(" "),
            Inline::LineBreak,
            Inline::plain("     Two"),
            Inline::LineBreak,
        ]),
        "One\n\nTwo"
    );
    // A break beside emphasis leaves its markers closed.
    assert_eq!(
        paragraph(vec![
            bold("bold"),
            Inline::LineBreak,
            Inline::plain("after")
        ]),
        "**bold**\\\nafter"
    );
    // A heading and a link's text are one line.
    let mut renderer = renderer("docx");
    assert_eq!(
        renderer.blocks(&[
            Block::heading(
                2,
                vec![
                    Inline::plain("Long title "),
                    Inline::LineBreak,
                    Inline::plain("continued"),
                ]
            ),
            Block::Paragraph(vec![
                Inline::plain("See "),
                link(vec![
                    Inline::plain("first"),
                    Inline::LineBreak,
                    Inline::plain("second"),
                ]),
                Inline::plain("."),
            ]),
        ]),
        "## Long title continued\n\nSee [first second](https://example.test/x)."
    );
    // A heading's closing `#` is text.
    assert_eq!(
        renderer.blocks(&[Block::heading(2, vec![Inline::plain("Issue #")])]),
        r"## Issue \#"
    );
}

#[test]
fn breaks_at_a_link_s_edges_break_the_line_around_it() {
    let at = |url: &str, content: Vec<Inline>| Inline::Link {
        content,
        target: LinkTarget::External(url.into()),
    };
    // Breaks that end a link's text count with the one after it: three end
    // the paragraph.
    assert_eq!(
        paragraph(vec![
            at(
                "https://example.test/a",
                vec![
                    Inline::plain("Heinrich"),
                    Inline::LineBreak,
                    Inline::LineBreak
                ],
            ),
            Inline::LineBreak,
            Inline::plain("@handle"),
        ]),
        "[Heinrich](https://example.test/a)\n\n@handle"
    );
    // A link not written as one (a `file:` target) leaves its text and its
    // breaks in the line.
    assert_eq!(
        paragraph(vec![
            at(
                "file:///profile",
                vec![Inline::plain("Name"), Inline::LineBreak],
            ),
            Inline::plain("Follows you"),
        ]),
        "Name\\\nFollows you"
    );
    // In a cell the break after each link is a line of the cell.
    let month = |name: &str| {
        at(
            "https://example.test/m",
            vec![Inline::plain(name), Inline::LineBreak],
        )
    };
    assert_eq!(
        renderer("odt").blocks(&[Block::Table(Table::from_rows(
            vec![vec![Cell::from_inlines(vec![
                month("August"),
                month("September")
            ])]],
            0,
            TableKind::Data,
        ))]),
        "|  |\n| --- |\n| [August](https://example.test/m)<br>[September](https://example.test/m) |"
    );
}

#[test]
fn a_footnote_definition_the_reader_writes_as_a_mark_is_kept() {
    // EPUB text `[^1]: …` is a note reference followed by text.
    assert_eq!(
        paragraph(vec![
            Inline::NoteRef("1".into()),
            Inline::plain(": The note."),
        ]),
        "[^1]: The note."
    );
    // The same characters as text are escaped.
    assert_eq!(text("[^1]: The note."), r"\[^1]: The note.");
}

#[test]
fn breaks_in_list_items_quotes_cells_and_notes_keep_their_containers() {
    use anydoc::model::{List, ListItem, MarkerKind};
    let broken = || {
        Block::Paragraph(vec![
            Inline::plain("a"),
            Inline::LineBreak,
            Inline::plain("b"),
        ])
    };
    let mut renderer = renderer("docx");
    let list = Block::List(List {
        marker: MarkerKind::Bullet,
        start: 1,
        items: vec![ListItem {
            blocks: vec![broken()],
            marker_label: None,
        }],
    });
    assert_eq!(renderer.blocks(&[list]), "* a\\\n  b");
    assert_eq!(
        renderer.blocks(&[Block::BlockQuote(vec![broken()])]),
        "> a\\\n> b"
    );
    // A cell's lines are joined with `<br>`, two breaks with an empty line.
    let cell = |values: Vec<Inline>| {
        Block::Table(Table::from_rows(
            vec![vec![Cell::from_inlines(values)]],
            0,
            TableKind::Data,
        ))
    };
    assert_eq!(
        renderer.blocks(&[cell(vec![
            Inline::plain("a"),
            Inline::LineBreak,
            Inline::plain("b"),
            Inline::LineBreak,
            Inline::LineBreak,
            Inline::plain("c | d"),
            Inline::LineBreak,
        ])]),
        "|  |\n| --- |\n| a<br>b<br><br>c \\| d |"
    );
    // A cell holds no blocks: a line starting like one is not escaped.
    assert_eq!(
        renderer.blocks(&[cell(vec![Inline::plain("- 5 # x")])]),
        "|  |\n| --- |\n| - 5 # x |"
    );
}

#[test]
fn a_label_markdown_does_not_read_keeps_its_line_and_a_number_under_a_paragraph_starts_a_list() {
    use anydoc::model::{List, ListItem, MarkerKind};
    let item = |label: &str, blocks: Vec<Block>| ListItem {
        blocks,
        marker_label: Some(label.into()),
    };
    let text = |value: &str| vec![Block::Paragraph(vec![Inline::plain(value)])];
    let mut renderer = renderer("rtf");
    let list = |items| {
        Block::List(List {
            marker: MarkerKind::Decimal,
            start: 1,
            items,
        })
    };
    // `a)` is text: a hard break keeps `b)` off the line before.
    assert_eq!(
        renderer.blocks(&[list(vec![item("a)", text("one")), item("b)", text("two"))])]),
        "a) one\\\nb) two"
    );
    // `2.` cannot start a list under a paragraph's line; a blank line lets it.
    assert_eq!(
        renderer.blocks(&[list(vec![item("a)", text("one")), item("2", text("two"))])]),
        "a) one\n\n2. two"
    );
    // After a fence or a table row a backslash would be text of its own: a
    // blank line instead. In a cell every item is a line.
    let line = |markdown_marker, interrupts, ends_in_paragraph| ListLine {
        text: String::new(),
        markdown_marker,
        interrupts,
        ends_in_paragraph,
    };
    let label = line(false, false, true);
    assert_eq!(
        list_separator(&line(false, false, false), &label, false),
        "\n\n"
    );
    assert_eq!(
        list_separator(&line(true, true, true), &label, false),
        "\\\n"
    );
    assert_eq!(list_separator(&label, &line(true, true, true), false), "\n");
    assert_eq!(list_separator(&label, &label, true), "\n");
}

#[test]
fn an_image_description_is_one_line() {
    let names = vec!["asset-1.png".to_owned()];
    let renderer = Renderer {
        asset_names: &names,
        merged_cells: false,
        in_cell: false,
        anchors: BTreeSet::new(),
        extension: "docx",
    };
    assert_eq!(
        renderer.inlines(&[Inline::Image {
            alt: "A chart\n# of [sales]".into(),
            source: anydoc::model::ImageSource::Asset(anydoc::model::AssetId(0)),
        }]),
        r"![A chart # of \[sales\]](.markitai/assets/asset-1.png)"
    );
}

#[test]
fn text_that_looks_like_an_image_survives_normal_output() {
    // Normal output drops empty images, doubled descriptions and a `)` after
    // an image; literal text of that shape is not an image and stays whole.
    for (values, written) in [
        (
            vec![Inline::plain("Literal ![x]() and ![a]![b](c) here")],
            "Literal !\\[x\\]() and !\\[a]!\\[b\\](c) here\n",
        ),
        (
            vec![
                Inline::plain("(Wow!"),
                link(vec![Inline::plain("site")]),
                Inline::plain(")"),
            ],
            "(Wow&#33;[site](https://example.test/x))\n",
        ),
    ] {
        assert_eq!(
            crate::markdown::normalize(&paragraph(values.clone())),
            written,
            "{values:?}"
        );
    }
}

#[test]
fn a_backtick_beside_a_code_span_is_a_character_reference() {
    // `\`` before a code span would still lengthen its opening backticks.
    let code = styled("code", |style| style.code = true);
    assert_eq!(
        paragraph(vec![Inline::plain("a `b`"), code, Inline::plain("` c")]),
        "a \\`b&#96;`code`&#96; c"
    );
}

#[test]
fn hard_breaks_survive_normal_output() {
    // A literal backslash at a paragraph's end is not a dangling break.
    assert_eq!(
        crate::markdown::normalize(&paragraph(vec![bold("b \\"), Inline::plain("\\ "),])),
        "**b \\\\**\\\\\n"
    );
    let markdown = crate::markdown::normalize(&paragraph(vec![
        Inline::plain("one"),
        Inline::LineBreak,
        link(vec![Inline::plain("two")]),
        Inline::LineBreak,
        Inline::plain("three  "),
    ]));
    assert_eq!(markdown, "one\\\n[two](https://example.test/x)\\\nthree\n");
}
