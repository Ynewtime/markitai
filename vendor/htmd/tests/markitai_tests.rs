// markitai: tests for the port from markup5ever_rcdom to the scraper tree. Every
// expected string is the output of the published htmd 0.5.5 for the same input
// and options (checked with a program linking both versions). The `TreeWriter`
// tests at the end compare its trees with the parser's for the same markup.
use htmd::{
    Element, HtmlToMarkdown, Node, NodeRef, TreeWriter,
    element_handler::Handlers,
    options::{Options, TranslationMode},
};
use pretty_assertions::assert_eq;
use scraper::Html;

fn faithful(html: &str) -> String {
    HtmlToMarkdown::builder()
        .options(Options {
            translation_mode: TranslationMode::Faithful,
            ..Default::default()
        })
        .build()
        .convert(html)
        .unwrap()
}

#[test]
fn similar_adjacent_inline_elements_merge_as_upstream() {
    for (html, markdown) in [
        ("<p><b>p</b><b>q</b></p>", "**pq**"),
        (
            "<p><i>a</i><em>b</em><i>c</i> <b>d</b><strong>e</strong><strong>f</strong></p>",
            "*abc* **def**",
        ),
        ("<p><code>a</code><code>b</code></p>", "`ab`"),
        ("<p><span>a</span><span>b</span></p>", "ab"),
        // Links, different attributes and anything besides one text node stay apart.
        (
            r#"<p><a href="/x">a</a><a href="/x">b</a></p>"#,
            "[a](/x)[b](/x)",
        ),
        (
            r#"<p><b class="k">a</b><b class="k">b</b><b class="other">c</b></p>"#,
            "**ab****c**",
        ),
        ("<p><b>a<!--c--></b><b>b</b></p>", "**a****b**"),
        (
            "<table><tr><th>h</th></tr><tr><td><b>a</b><b>b</b></td></tr></table>",
            "| h      |\n| ------ |\n| **ab** |",
        ),
    ] {
        assert_eq!(markdown, htmd::convert(html).unwrap(), "{html}");
    }
}

#[test]
fn children_that_are_never_walked_are_not_merged() {
    // The ordered-list handler reads its children one by one, so they keep
    // their own markers.
    assert_eq!(
        "1.  x**p****q**",
        htmd::convert("<ol><li>x</li><b>p</b><b>q</b></ol>").unwrap()
    );
}

#[test]
fn merged_text_is_what_handlers_read() {
    assert_eq!(
        "$xy$",
        htmd::convert(
            r#"<p><span class="math math-inline">x</span><span class="math math-inline">y</span></p>"#
        )
        .unwrap()
    );
    // Faithful serialization of a subtree walked before shows the merge too.
    assert_eq!(
        "<pre><code><b>ab</b></code></pre>",
        faithful("<pre><code><b>a</b><b>b</b></code></pre>")
    );
    assert_eq!(
        "<span>ab</span>",
        faithful("<p><span>a</span><span>b</span></p>")
    );
}

#[test]
fn template_contents_are_left_out() {
    assert_eq!(
        "ab",
        htmd::convert("<p>a<template>hidden</template>b</p>").unwrap()
    );
    assert_eq!(
        "a<template></template>b",
        faithful("<p>a<template>hidden</template>b</p>")
    );
    assert_eq!(
        "",
        htmd::convert("<p><template>x</template><template>y</template></p>").unwrap()
    );
}

#[test]
fn converting_leaves_the_tree_unchanged() {
    let converter = HtmlToMarkdown::new();
    let html = converter.html_to_tree("<p><b>p</b><b>q</b></p>").unwrap();
    assert_eq!("**pq**", converter.tree_to_markdown(html.tree.root()));
    assert_eq!("**pq**", converter.tree_to_markdown(html.tree.root()));
    let bold: Vec<String> = html
        .tree
        .root()
        .descendants()
        .filter(|node| matches!(node.value(), Node::Element(element) if element.name() == "b"))
        .map(|node| {
            node.children()
                .filter_map(|text| text.value().as_text())
                .map(|text| text.to_string())
                .collect()
        })
        .collect();
    assert_eq!(vec!["p", "q"], bold);
}

#[test]
fn scripting_option_decides_how_noscript_parses() {
    let html = "<noscript><p>x <b>y</b></p></noscript><p>z</p>";
    assert_eq!(
        "\\<p>x \\<b>y\\</b>\\</p>\n\nz",
        htmd::convert(html).unwrap()
    );
    assert_eq!(
        "x **y**\n\nz",
        HtmlToMarkdown::builder()
            .scripting_enabled(false)
            .build()
            .convert(html)
            .unwrap()
    );
}

#[test]
fn element_attr_matches_local_names() {
    let converter = HtmlToMarkdown::builder()
        .add_handler(vec!["a"], |_: &dyn Handlers, element: Element| {
            Some(element.attr("href").unwrap_or("none").to_owned().into())
        })
        .build();
    assert_eq!(
        "/h",
        converter.convert(r#"<p><a href="/h">x</a></p>"#).unwrap()
    );
    // An SVG link's `xlink:href`, as the anchor handler reads it.
    assert_eq!(
        "https://example.com/x",
        converter
            .convert(r#"<svg><a xlink:href="https://example.com/x">x</a></svg>"#)
            .unwrap()
    );
    assert_eq!("none", converter.convert("<p><a>x</a></p>").unwrap());
    assert_eq!(
        "[svg link](https://example.com/x)",
        htmd::convert(r#"<svg><a xlink:href="https://example.com/x">svg link</a></svg>"#).unwrap()
    );
}

// Unlike the published htmd 0.5.5, which trimmed every line of a list item
// and so joined the lines a `<br>` (two trailing spaces) had broken.
#[test]
fn a_hard_break_in_a_list_item_is_kept() {
    assert_eq!(
        "*   a  \n    b\n*   c",
        htmd::convert("<ul><li>a<br>b<br></li><li>c</li></ul>").unwrap()
    );
    assert_eq!(
        "1.  a  \n    b",
        htmd::convert("<ol><li>a  \n<br>b</li></ol>").unwrap()
    );
}

/// One part of markup, as a `TreeWriter` takes it.
#[derive(Clone, Debug)]
enum Part {
    Start(String, Vec<(String, String)>),
    End(String),
    Text(String),
}

fn start(name: &str) -> Part {
    Part::Start(name.to_owned(), Vec::new())
}

fn start_with(name: &str, attributes: &[(&str, &str)]) -> Part {
    Part::Start(
        name.to_owned(),
        attributes
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect(),
    )
}

fn end(name: &str) -> Part {
    Part::End(name.to_owned())
}

fn text(text: &str) -> Part {
    Part::Text(text.to_owned())
}

fn escaped(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// The markup the parts stand for.
fn markup(parts: &[Part]) -> String {
    let mut markup = String::new();
    for part in parts {
        match part {
            Part::Start(name, attributes) => {
                markup.push('<');
                markup.push_str(name);
                for (name, value) in attributes {
                    markup.push_str(&format!(" {name}=\"{}\"", escaped(value)));
                }
                markup.push('>');
            }
            Part::End(name) => markup.push_str(&format!("</{name}>")),
            Part::Text(text) => markup.push_str(&escaped(text)),
        }
    }
    markup
}

fn written(parts: &[Part]) -> Option<Html> {
    let mut writer = TreeWriter::new();
    for part in parts {
        match part {
            Part::Start(name, attributes) => {
                writer.start(name);
                for (name, value) in attributes {
                    writer.attribute(name, value);
                }
            }
            Part::End(name) => writer.end(name),
            Part::Text(text) => writer.text(text),
        }
    }
    writer.finish()
}

/// A tree as text, one node per line: namespaces, names, attributes and text.
fn dump(html: &Html) -> String {
    fn walk(node: NodeRef<'_>, depth: usize, out: &mut String) {
        out.push_str(&"  ".repeat(depth));
        match node.value() {
            Node::Element(element) => {
                out.push_str(&format!("<{} {}>", element.name.ns, element.name.local));
                for (name, value) in &element.attrs {
                    out.push_str(&format!(" {}:{}={:?}", name.ns, name.local, &**value));
                }
            }
            Node::Text(text) => out.push_str(&format!("{:?}", &**text)),
            other => out.push_str(&format!("{other:?}")),
        }
        out.push('\n');
        for child in node.children() {
            walk(child, depth + 1, out);
        }
    }
    let mut out = String::new();
    walk(html.tree.root(), 0, &mut out);
    out
}

/// Whether the writer built a tree; when it did, it is the parser's.
fn same_as_parsed(parts: &[Part]) -> bool {
    let markup = markup(parts);
    let Some(tree) = written(parts) else {
        return false;
    };
    assert_eq!(
        dump(&Html::parse_document(&markup)),
        dump(&tree),
        "{markup:?}"
    );
    true
}

#[test]
fn tree_writer_builds_the_parsers_tree_for_markup_written_from_a_tree() {
    for parts in [
        // The elements the parser implies, and the head's elements before
        // the body, also with space and a second `<body>` around them.
        vec![],
        vec![text("  "), text("a")],
        vec![
            text("\n  "),
            start_with("meta", &[("name", "x")]),
            start("title"),
            text("T &amp; x"),
            end("title"),
            text("\n"),
            start("link"),
            start_with("body", &[("class", "b")]),
            text("x"),
            end("body"),
        ],
        vec![
            start_with("html", &[("class", "a")]),
            start("markitai-soft-break"),
            end("markitai-soft-break"),
            start_with("body", &[("id", "b"), ("class", "c")]),
            text("x"),
            end("body"),
            end("html"),
        ],
        vec![
            start("html"),
            start("title"),
            text("Mail"),
            end("title"),
            start("meta"),
            start("style"),
            text("p > a {}"),
            end("style"),
            start("p"),
            text("x"),
            end("p"),
            end("html"),
        ],
        // Line feeds right after `<pre>`, `<listing>` and `<textarea>`.
        vec![
            start("pre"),
            text("\n\nx"),
            end("pre"),
            start("listing"),
            text("\ny"),
            end("listing"),
            start("pre"),
            text(""),
            text("\nz"),
            end("pre"),
            start("textarea"),
            text("\nt"),
            end("textarea"),
        ],
        // Tables keep their space; a link in a cell is not inside the one
        // around the table.
        vec![
            start("table"),
            text("\n "),
            start("caption"),
            text("c"),
            end("caption"),
            start("colgroup"),
            text(" "),
            start("col"),
            end("colgroup"),
            start("thead"),
            start("tr"),
            start("th"),
            text("h"),
            end("th"),
            end("tr"),
            end("thead"),
            start("tbody"),
            start("tr"),
            text(" "),
            start("td"),
            start("b"),
            text("1"),
            end("b"),
            start("p"),
            text("x"),
            end("p"),
            end("td"),
            end("tr"),
            end("tbody"),
            end("table"),
        ],
        vec![
            start_with("a", &[("href", "/1")]),
            start("div"),
            start("table"),
            start("tbody"),
            start("tr"),
            start("td"),
            start_with("a", &[("href", "/2")]),
            text("x"),
            end("a"),
            end("td"),
            end("tr"),
            end("tbody"),
            end("table"),
            end("div"),
            end("a"),
        ],
        // Elements the parser closes at once, and their end tags.
        vec![
            start("p"),
            start("br"),
            start_with("img", &[("src", "x")]),
            start("param"),
            end("param"),
            start("track"),
            end("track"),
            start_with("image", &[("src", "y")]),
            end("image"),
            text("x"),
            end("p"),
        ],
        // A button bounds a paragraph: a block inside it leaves the paragraph open.
        vec![
            start("p"),
            start("button"),
            start("div"),
            text("x"),
            end("div"),
            end("button"),
            end("p"),
        ],
        // Names with capitals, lists, ruby, headings, after the body.
        vec![
            start("DIV"),
            start("Span"),
            text("x"),
            end("SPAN"),
            end("div"),
        ],
        vec![
            start("ul"),
            start("li"),
            text("a"),
            start("ul"),
            start("li"),
            text("b"),
            end("li"),
            end("ul"),
            end("li"),
            start("li"),
            end("li"),
            end("ul"),
            start("dl"),
            start("dt"),
            text("t"),
            end("dt"),
            start("dd"),
            text("d"),
            end("dd"),
            end("dl"),
        ],
        vec![
            start("ruby"),
            text("漢"),
            start("rt"),
            text("kan"),
            end("rt"),
            end("ruby"),
        ],
        vec![
            start("h2"),
            start_with("a", &[("href", "#x")]),
            text("x"),
            end("a"),
            end("h2"),
            start("p"),
            start("em"),
            text("y"),
            end("em"),
            end("p"),
        ],
        vec![
            start("body"),
            text("x"),
            end("body"),
            text(" "),
            end("html"),
            text(" y"),
            start("p"),
            text("z"),
            end("p"),
        ],
    ] {
        assert!(same_as_parsed(&parts), "{parts:?}");
    }
}

#[test]
fn tree_writer_reads_text_and_values_as_the_tokenizer_does() {
    for parts in [
        // Carriage returns, also one written before the line feed it pairs with.
        vec![start("p"), text("a\rb\r\nc\r"), text("\nd"), end("p")],
        vec![
            start("pre"),
            text("\r\nx\r"),
            start("b"),
            text("\ny"),
            end("b"),
            end("pre"),
        ],
        vec![
            start("pre"),
            text("\r"),
            text("\n"),
            text("\nx"),
            end("pre"),
        ],
        // A NUL is dropped in text, also in a table, and replaced in
        // attribute values; after `<pre>` it keeps the next line feed.
        vec![start("p"), text("a\0b\r\0\nc"), end("p")],
        vec![start("pre"), text("\0\nx"), end("pre")],
        vec![
            start("table"),
            text(" \0 "),
            start("tbody"),
            end("tbody"),
            end("table"),
        ],
        vec![
            start_with("p", &[("title", "a\0b\r\nc\rd&\"<>")]),
            text("x"),
            end("p"),
        ],
        // A byte order mark is dropped only at the start.
        vec![text("\u{feff}a"), text("\u{feff}b")],
        vec![start("p"), text("\u{feff}a"), end("p")],
        // The first of two attributes with one name; names are lowercased.
        vec![
            start_with(
                "p",
                &[("class", "a"), ("ID", "b"), ("CLASS", "c"), ("id", "d")],
            ),
            text("x"),
            end("p"),
        ],
    ] {
        assert!(same_as_parsed(&parts), "{parts:?}");
    }
}

#[test]
fn tree_writer_reads_raw_text_as_its_markup_spells_it() {
    for parts in [
        // Escapes are read in a title or text area and kept elsewhere; tags
        // inside are text.
        vec![
            start("p"),
            text("a"),
            end("p"),
            start("title"),
            text("x & <y>"),
            start_with("b", &[("class", "c&d\"e")]),
            text("z\r\n"),
            end("b"),
            end("title"),
            start("textarea"),
            text("\nfirst\n"),
            end("textarea"),
        ],
        vec![
            start("xmp"),
            text("x & <y>"),
            start_with("markitai-soft-break", &[("a", "&")]),
            end("markitai-soft-break"),
            text("\0"),
            end("xmp"),
            start("style"),
            text("a > b"),
            end("STYLE"),
            start("noembed"),
            end("noembed"),
            start("noframes"),
            text("f"),
            end("noframes"),
            start("iframe"),
            text("i"),
            end("iframe"),
            start("noscript"),
            text("<n>"),
            end("noscript"),
        ],
        // The first end tag of the element's name ends its text.
        vec![start("title"), start("title"), text("x"), end("title")],
        vec![start("title"), text("x"), end("titlex"), end("title")],
        // Plain text runs to the end.
        vec![
            start("div"),
            start("plaintext"),
            text("a"),
            start("b"),
            end("b"),
            end("plaintext"),
            end("div"),
        ],
        vec![start("title"), text("never closed")],
    ] {
        assert!(same_as_parsed(&parts), "{parts:?}");
    }
}

#[test]
fn tree_writer_leaves_markup_the_parser_rearranges_to_the_parser() {
    for parts in [
        // A block closing an open paragraph, a code block in one.
        vec![
            start("p"),
            text("a"),
            start("div"),
            text("b"),
            end("div"),
            end("p"),
        ],
        vec![
            start("p"),
            start("b"),
            start("pre"),
            text("x"),
            end("pre"),
            end("b"),
            end("p"),
        ],
        // Nested headings, list items, links.
        vec![start("h2"), start("h3"), text("x"), end("h3"), end("h2")],
        vec![
            start("li"),
            start("div"),
            start("li"),
            text("x"),
            end("li"),
            end("div"),
            end("li"),
        ],
        vec![
            start("a"),
            start("div"),
            start("a"),
            text("x"),
            end("a"),
            end("div"),
            end("a"),
        ],
        // Text and elements moved out of a table, implied table parts.
        vec![start("table"), text("x"), end("table")],
        vec![
            start("table"),
            start("markitai-soft-break"),
            end("markitai-soft-break"),
            end("table"),
        ],
        vec![
            start("table"),
            start("tr"),
            start("td"),
            end("td"),
            end("tr"),
            end("table"),
        ],
        // An end tag of an element that is not the current one, `</br>`.
        vec![start("p"), end("div"), end("p")],
        vec![start("p"), end("br"), end("p")],
        // Foreign content, forms, templates, frames, scripts, encodings.
        vec![start("svg"), start("path"), end("path"), end("svg")],
        vec![
            start("math"),
            start("mi"),
            text("x"),
            end("mi"),
            end("math"),
        ],
        vec![
            start("select"),
            start("option"),
            end("option"),
            end("select"),
        ],
        vec![start("form"), end("form")],
        vec![start("template"), text("x"), end("template")],
        vec![start("frameset")],
        vec![start("p"), start("script"), text("x < y"), end("script")],
        vec![
            start_with("meta", &[("charset", "utf-8")]),
            start("p"),
            end("p"),
        ],
        // A NUL before the body.
        vec![text("\0x")],
    ] {
        assert!(written(&parts).is_none(), "{parts:?}");
    }
}

#[test]
fn tree_writer_leaves_names_it_cannot_read_to_the_parser() {
    for parts in [
        vec![start("1a"), text("x")],
        vec![start("p"), end("a b")],
        vec![start("a/b")],
        vec![start_with("p", &[("a=b", "c")])],
        vec![text("x"), Part::Start(String::new(), Vec::new())],
        vec![start("title"), start("a&lt"), end("title")],
    ] {
        assert!(written(&parts).is_none(), "{parts:?}");
    }
    let mut writer = TreeWriter::new();
    writer.text("x");
    writer.attribute("class", "y");
    assert!(writer.finish().is_none());
}

/// xorshift64, for the same documents on every run.
struct Random(u64);

impl Random {
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % bound as u64) as usize
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len())]
    }

    fn string(&mut self, length: usize) -> String {
        const CHARS: &[char] = &[
            'a', 'b', ' ', ' ', '\n', '\r', '\t', '\0', '&', '<', '>', '"', '\'', '=', '/',
            '\u{feff}', 'é', '\u{fffd}',
        ];
        (0..length)
            .map(|_| CHARS[self.below(CHARS.len())])
            .collect()
    }

    fn start(&mut self, name: &str) -> Part {
        const ATTRIBUTES: &[&str] = &["href", "class", "id", "data-x", "CLASS", "xlink:href"];
        let attributes = (0..self.below(3))
            .map(|_| {
                let name = self.pick(ATTRIBUTES).to_owned();
                let length = self.below(4);
                (name, self.string(length))
            })
            .collect();
        Part::Start(name.to_owned(), attributes)
    }
}

const NAMES: &[&str] = &[
    "p",
    "div",
    "span",
    "pre",
    "listing",
    "textarea",
    "table",
    "tbody",
    "tr",
    "td",
    "th",
    "caption",
    "colgroup",
    "col",
    "li",
    "ul",
    "ol",
    "dl",
    "dd",
    "dt",
    "a",
    "b",
    "i",
    "em",
    "code",
    "nobr",
    "h1",
    "h2",
    "h3",
    "svg",
    "foreignObject",
    "math",
    "mi",
    "title",
    "xmp",
    "noembed",
    "noframes",
    "style",
    "plaintext",
    "select",
    "option",
    "br",
    "img",
    "hr",
    "param",
    "wbr",
    "body",
    "html",
    "head",
    "form",
    "button",
    "template",
    "ruby",
    "rt",
    "image",
    "DIV",
    "Td",
    "markitai-soft-break",
    "markitai-footnote",
    "x-y",
    "meta",
    "link",
];

/// Elements written as a tree is: start tags, children, end tags (none for
/// elements without one), mostly where the HTML content model allows them.
fn tree(random: &mut Random, parent: &str, depth: usize, parts: &mut Vec<Part>) {
    const VOID: &[&str] = &["br", "img", "hr", "wbr", "col", "meta", "link"];
    const BLOCKS: &[&str] = &[
        "div",
        "p",
        "ul",
        "ol",
        "dl",
        "table",
        "pre",
        "h2",
        "blockquote",
        "section",
        "title",
        "xmp",
        "textarea",
        "x-y",
    ];
    const INLINE: &[&str] = &[
        "span",
        "a",
        "b",
        "i",
        "em",
        "code",
        "br",
        "img",
        "markitai-soft-break",
        "markitai-footnote",
        "ruby",
        "param",
        "wbr",
    ];
    for _ in 0..random.below(4) {
        let child = match parent {
            "table" => random.pick(&["caption", "colgroup", "tbody", "thead", " "]),
            "tbody" | "thead" => random.pick(&["tr", " "]),
            "tr" => random.pick(&["td", "th", " "]),
            "colgroup" => random.pick(&["col", " "]),
            "ul" | "ol" => random.pick(&["li", " "]),
            "dl" => random.pick(&["dt", "dd"]),
            "ruby" => random.pick(&["rt", "#"]),
            "title" | "xmp" | "textarea" | "pre" => random.pick(&["#", "#", "b"]),
            _ if random.below(12) == 0 => random.pick(NAMES),
            _ if depth > 4 || random.below(2) == 0 => "#",
            "p" | "span" | "a" | "b" | "i" | "em" | "code" | "h2" | "caption" | "rt" | "dt" => {
                random.pick(INLINE)
            }
            _ if random.below(2) == 0 => random.pick(INLINE),
            _ => random.pick(BLOCKS),
        };
        match child {
            "#" | " " => {
                let length = 1 + random.below(5);
                let mut text = random.string(length);
                if child == " " {
                    text.retain(|ch| matches!(ch, ' ' | '\n' | '\t'));
                }
                parts.push(Part::Text(text));
            }
            name => {
                parts.push(random.start(name));
                if !VOID.contains(&name) {
                    tree(random, name, depth + 1, parts);
                    parts.push(Part::End(name.to_owned()));
                }
            }
        }
    }
}

#[test]
fn tree_writer_matches_the_parser_on_generated_markup() {
    let mut random = Random(0x2545_f491_4f6c_dd1d);
    // Markup written from generated trees, inside a document's elements or not.
    let (total, mut built) = (10_000, 0);
    for _ in 0..total {
        let mut parts = Vec::new();
        let root = random.pick(&["#document", "html", "body", "div"]);
        if root == "#document" {
            tree(&mut random, "div", 0, &mut parts);
        } else {
            parts.push(random.start(root));
            tree(&mut random, root, 0, &mut parts);
            parts.push(Part::End(root.to_owned()));
        }
        built += usize::from(same_as_parsed(&parts));
    }
    assert!(built > total * 85 / 100, "{built} of {total} trees built");
    // Any parts in any order: what the writer builds is still the parser's tree.
    let (total, mut built) = (10_000, 0);
    for _ in 0..total {
        let mut parts = Vec::new();
        for _ in 0..random.below(24) {
            let part = match random.below(5) {
                0 | 1 => {
                    let name = random.pick(NAMES);
                    random.start(name)
                }
                2 => Part::End(random.pick(NAMES).to_owned()),
                _ => {
                    let length = random.below(6);
                    Part::Text(random.string(length))
                }
            };
            parts.push(part);
        }
        built += usize::from(same_as_parsed(&parts));
    }
    assert!(built > total / 5, "{built} of {total} sequences built");
}

#[test]
fn nested_identical_emphasis_is_written_once() {
    for (html, markdown) in [
        ("<p><b><strong>text</strong></b></p>", "**text**"),
        ("<p><strong>a <b>b</b> c</strong></p>", "**a b c**"),
        ("<p><em><i>text</i></em></p>", "*text*"),
        (
            r#"<p><b><a href="/x"><strong>t</strong></a></b></p>"#,
            "**[t](/x)**",
        ),
        // Different emphasis nests as CommonMark spells it.
        ("<p><b><i>text</i></b></p>", "***text***"),
        // Siblings stay separate.
        ("<p><b>a</b> <strong>b</strong></p>", "**a** **b**"),
    ] {
        assert_eq!(markdown, htmd::convert(html).unwrap(), "{html}");
    }
}

#[test]
fn image_text_joins_its_lines_and_an_empty_title_is_left_out() {
    for (html, markdown) in [
        (
            "<img src=\"a.png\" alt=\"Two tables: the first\nstack, with its length\n  and a pointer.\">",
            "![Two tables: the first stack, with its length and a pointer.](a.png)",
        ),
        (
            "<img src=\"a.png\" alt=\"x\" title=\"first\nsecond\">",
            "![x](a.png \"first second\")",
        ),
        ("<img src=\"a.png\" alt=\"x\" title=\"\">", "![x](a.png)"),
        (
            "<img src=\"a.png\" alt=\"x\" title=\" \n \">",
            "![x](a.png)",
        ),
        ("<img src=\"a.png\" alt=\"\n\">", "![](a.png)"),
    ] {
        assert_eq!(markdown, htmd::convert(html).unwrap(), "{html}");
    }
}
