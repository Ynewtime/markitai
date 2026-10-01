// markitai: tests for the port from markup5ever_rcdom to the scraper tree. Every
// expected string is the output of the published htmd 0.5.5 for the same input
// and options (checked with a program linking both versions).
use htmd::{
    Element, HtmlToMarkdown, Node,
    element_handler::Handlers,
    options::{Options, TranslationMode},
};
use pretty_assertions::assert_eq;

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
