//! HTML parsing with a bound on element nesting.
//!
//! html5ever's tree builder looks through its stack of open elements for
//! most start tags (whether a `p` is open in scope, for one), so a document
//! of thousands of nested elements costs time quadratic in its depth.
//! Chromium attaches elements past depth 512 to the element at that depth;
//! here, start tags past [`MAX_OPEN`] open elements are dropped before the
//! tree builder sees them, with the end tags that close them, and their text
//! stays in the element at the limit. Content that deep is rendered as plain
//! text anyway (see `MAX_DEPTH`).
//!
//! The depth is followed from the tokens with the parser's own rules for the
//! common cases: end tags close through the elements the parser closes
//! through, and `p`, list items, table cells and rows, options and headings
//! close the way the parser closes them. Elements the parser closes in other
//! ways are counted as still open, so the count errs on the deep side;
//! elements the parser adds on its own (an implied `tbody`, reopened
//! formatting) keep its stack within a small multiple of the limit.

use html5ever::tendril::StrTendril;
use html5ever::tokenizer::{
    BufferQueue, Tag, TagKind, Token, TokenSink, TokenSinkResult, Tokenizer, TokenizerOpts,
};
use html5ever::tree_builder::{TreeBuilder, TreeBuilderOpts, TreeSink};
use html5ever::{LocalName, QualName, TokenizerResult, local_name, ns};
use scraper::{Html, HtmlTreeSink};
use std::cell::RefCell;
use std::collections::HashMap;

/// Open elements a document keeps; deeper start tags are dropped.
pub(crate) const MAX_OPEN: usize = 512;

/// `source` as a document, as `Html::parse_document` reads it, nesting
/// bounded.
pub(crate) fn document(source: &str) -> Html {
    let builder = TreeBuilder::new(
        HtmlTreeSink::new(Html::new_document()),
        TreeBuilderOpts::default(),
    );
    run(
        Tokenizer::new(Bounded::new(builder), TokenizerOpts::default()),
        source,
    )
}

/// `source` as a fragment in a `body`, as `Html::parse_fragment` reads it,
/// nesting bounded.
pub(crate) fn fragment(source: &str) -> Html {
    let sink = HtmlTreeSink::new(Html::new_fragment());
    let context = html5ever::tree_builder::create_element(
        &sink,
        QualName::new(None, ns!(html), local_name!("body")),
        Vec::new(),
    );
    let builder = TreeBuilder::new_for_fragment(sink, context, None, TreeBuilderOpts::default());
    let options = TokenizerOpts {
        initial_state: Some(builder.tokenizer_state_for_context_elem(false)),
        ..TokenizerOpts::default()
    };
    run(Tokenizer::new(Bounded::new(builder), options), source)
}

type Builder = TreeBuilder<<HtmlTreeSink as TreeSink>::Handle, HtmlTreeSink>;

fn run(tokenizer: Tokenizer<Bounded<Builder>>, source: &str) -> Html {
    let input = BufferQueue::default();
    input.push_back(StrTendril::from_slice(source));
    while !matches!(tokenizer.feed(&input), TokenizerResult::Done) {}
    tokenizer.end();
    tokenizer.sink.inner.sink.finish()
}

/// A token sink that passes tokens on, without start tags past the limit.
struct Bounded<S> {
    inner: S,
    open: RefCell<Open>,
}

impl<S> Bounded<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            open: RefCell::default(),
        }
    }
}

impl<S: TokenSink> TokenSink for Bounded<S> {
    type Handle = S::Handle;

    fn process_token(&self, token: Token, line_number: u64) -> TokenSinkResult<S::Handle> {
        if let Token::TagToken(tag) = &token {
            let mut open = self.open.borrow_mut();
            let keep = match tag.kind {
                TagKind::StartTag => open.start(tag),
                TagKind::EndTag => open.end(tag),
            };
            drop(open);
            if !keep {
                return TokenSinkResult::Continue;
            }
        }
        self.inner.process_token(token, line_number)
    }

    fn end(&self) {
        self.inner.end();
    }

    fn adjusted_current_node_present_but_not_in_html_namespace(&self) -> bool {
        self.inner
            .adjusted_current_node_present_but_not_in_html_namespace()
    }
}

/// The elements counted open, and the start tags dropped whose end tags are
/// still to come.
#[derive(Default)]
struct Open {
    elements: Vec<Element>,
    dropped: HashMap<LocalName, usize>,
}

struct Element {
    name: LocalName,
    /// An SVG or MathML element.
    foreign: bool,
}

impl Open {
    /// Counts a start tag; false when it is dropped.
    fn start(&mut self, tag: &Tag) -> bool {
        let name = &*tag.name;
        if matches!(name, "html" | "head" | "body" | "frameset") {
            return true;
        }
        let in_foreign = self
            .elements
            .last()
            .is_some_and(|top| top.foreign && !integration_point(&top.name));
        let foreign = if in_foreign && breaks_out(tag) {
            // The parser leaves foreign content for these.
            while self
                .elements
                .last()
                .is_some_and(|top| top.foreign && !integration_point(&top.name))
            {
                self.elements.pop();
            }
            false
        } else {
            in_foreign || matches!(name, "svg" | "math")
        };
        if !foreign {
            self.close_implied(name);
        }
        let pushed = if foreign {
            !tag.self_closing
        } else {
            !void(name)
        };
        if !pushed {
            return true;
        }
        // Raw text holds no tags, so it never deepens the document; dropping
        // its start tag would read a script as markup.
        if self.elements.len() >= MAX_OPEN && !raw_text(name) {
            *self.dropped.entry(tag.name.clone()).or_default() += 1;
            return false;
        }
        self.elements.push(Element {
            name: tag.name.clone(),
            foreign,
        });
        true
    }

    /// Counts an end tag; false when it closes a dropped start tag.
    fn end(&mut self, tag: &Tag) -> bool {
        if let Some(count) = self.dropped.get_mut(&tag.name)
            && *count > 0
        {
            *count -= 1;
            return false;
        }
        let name = &*tag.name;
        // A special element's end tag closes through anything up to a scope
        // boundary; any other closes through elements that are not special.
        let special_end = special(name);
        let position = self.elements.iter().rposition(|element| {
            element.name == tag.name
                || if special_end {
                    scope_boundary(&element.name)
                } else {
                    special(&element.name) && !element.foreign
                }
        });
        if let Some(position) = position
            && self.elements[position].name == tag.name
        {
            self.elements.truncate(position);
        }
        true
    }

    /// The elements a start tag `name` closes before it opens, as the parser
    /// closes them.
    fn close_implied(&mut self, name: &str) {
        if closes_p(name) {
            self.close_in_scope("p", |element| {
                scope_boundary(element) || element == "button"
            });
        }
        match name {
            "li" => self.close_in_scope("li", |element| {
                special(element) && !matches!(element, "address" | "div" | "p")
            }),
            "dd" | "dt" => {
                self.close_either("dd", "dt", |element| {
                    special(element) && !matches!(element, "address" | "div" | "p")
                });
            }
            "td" | "th" => self.close_either("td", "th", table_boundary),
            "tr" => self.close_in_scope("tr", table_boundary),
            "thead" | "tbody" | "tfoot" => {
                if let Some(position) = self.find(
                    |element| matches!(element, "thead" | "tbody" | "tfoot"),
                    table_boundary,
                ) {
                    self.elements.truncate(position);
                }
            }
            "button" => self.close_in_scope("button", scope_boundary),
            "option" => self.close_top(&["option"]),
            "optgroup" => self.close_top(&["option", "optgroup"]),
            "rb" | "rp" | "rt" | "rtc" => self.close_top(&["rb", "rp", "rt", "rtc"]),
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                self.close_top(&["h1", "h2", "h3", "h4", "h5", "h6"]);
            }
            _ => {}
        }
    }

    fn close_in_scope(&mut self, name: &str, boundary: impl Fn(&str) -> bool) {
        if let Some(position) = self.find(|element| element == name, boundary) {
            self.elements.truncate(position);
        }
    }

    fn close_either(&mut self, first: &str, second: &str, boundary: impl Fn(&str) -> bool) {
        if let Some(position) = self.find(|element| element == first || element == second, boundary)
        {
            self.elements.truncate(position);
        }
    }

    fn close_top(&mut self, names: &[&str]) {
        if self
            .elements
            .last()
            .is_some_and(|top| !top.foreign && names.contains(&&*top.name))
        {
            self.elements.pop();
        }
    }

    /// The nearest open HTML element `wanted` accepts, unless `boundary`
    /// stops the search first.
    fn find(
        &self,
        wanted: impl Fn(&str) -> bool,
        boundary: impl Fn(&str) -> bool,
    ) -> Option<usize> {
        for (position, element) in self.elements.iter().enumerate().rev() {
            if !element.foreign && wanted(&element.name) {
                return Some(position);
            }
            if boundary(&element.name) {
                return None;
            }
        }
        None
    }
}

fn void(name: &str) -> bool {
    matches!(
        name,
        "area"
            | "base"
            | "basefont"
            | "bgsound"
            | "br"
            | "col"
            | "embed"
            | "frame"
            | "hr"
            | "image"
            | "img"
            | "input"
            | "keygen"
            | "link"
            | "meta"
            | "param"
            | "source"
            | "track"
            | "wbr"
    )
}

fn raw_text(name: &str) -> bool {
    matches!(
        name,
        "script"
            | "style"
            | "textarea"
            | "title"
            | "xmp"
            | "iframe"
            | "noembed"
            | "noframes"
            | "noscript"
            | "plaintext"
    )
}

/// Start tags that close an open `p` first.
fn closes_p(name: &str) -> bool {
    matches!(
        name,
        "address"
            | "article"
            | "aside"
            | "blockquote"
            | "center"
            | "details"
            | "dialog"
            | "dir"
            | "div"
            | "dl"
            | "fieldset"
            | "figcaption"
            | "figure"
            | "footer"
            | "form"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "header"
            | "hgroup"
            | "hr"
            | "listing"
            | "main"
            | "menu"
            | "nav"
            | "ol"
            | "p"
            | "plaintext"
            | "pre"
            | "search"
            | "section"
            | "summary"
            | "ul"
            | "xmp"
            | "li"
            | "dd"
            | "dt"
    )
}

/// The elements a search for an open element in scope stops at.
fn scope_boundary(name: &str) -> bool {
    matches!(
        name,
        "applet"
            | "caption"
            | "html"
            | "table"
            | "td"
            | "th"
            | "marquee"
            | "object"
            | "template"
            | "mi"
            | "mo"
            | "mn"
            | "ms"
            | "mtext"
            | "annotation-xml"
            | "foreignobject"
            | "desc"
            | "title"
    )
}

fn table_boundary(name: &str) -> bool {
    matches!(name, "table" | "html" | "template")
}

/// Foreign elements whose children are read as HTML.
fn integration_point(name: &str) -> bool {
    matches!(
        name,
        "foreignobject" | "desc" | "title" | "mi" | "mo" | "mn" | "ms" | "mtext" | "annotation-xml"
    )
}

/// HTML start tags that end foreign content (`font` only with these
/// attributes).
fn breaks_out(tag: &Tag) -> bool {
    if &*tag.name == "font" {
        return tag
            .attrs
            .iter()
            .any(|attr| matches!(&*attr.name.local, "color" | "face" | "size"));
    }
    matches!(
        &*tag.name,
        "b" | "big"
            | "blockquote"
            | "body"
            | "br"
            | "center"
            | "code"
            | "dd"
            | "div"
            | "dl"
            | "dt"
            | "em"
            | "embed"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "head"
            | "hr"
            | "i"
            | "img"
            | "li"
            | "listing"
            | "menu"
            | "meta"
            | "nobr"
            | "ol"
            | "p"
            | "pre"
            | "ruby"
            | "s"
            | "small"
            | "span"
            | "strong"
            | "strike"
            | "sub"
            | "sup"
            | "table"
            | "tt"
            | "u"
            | "ul"
            | "var"
    )
}

/// The parser's special elements (HTML, and the foreign ones it treats so).
fn special(name: &str) -> bool {
    scope_boundary(name)
        || void(name)
        || raw_text(name)
        || closes_p(name)
        || matches!(
            name,
            "body"
                | "button"
                | "colgroup"
                | "frameset"
                | "head"
                | "select"
                | "tbody"
                | "tfoot"
                | "thead"
                | "tr"
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The deepest chain of elements in a parsed tree.
    fn depth(html: &Html) -> usize {
        let mut deepest = 0;
        let mut pending = vec![(html.tree.root(), 0usize)];
        while let Some((node, level)) = pending.pop() {
            deepest = deepest.max(level);
            pending.extend(node.children().map(|child| (child, level + 1)));
        }
        deepest
    }

    #[test]
    fn documents_within_the_limit_parse_as_the_parser_reads_them() {
        for source in [
            "<!doctype html><title>T</title><p>One<p>Two<div>Block</div><ul><li>a<li>b</ul>",
            "<table><tr><td>1<td>2<tr><td>3</table><dl><dt>x<dd>y<dt>z</dl>",
            "<p><b>bold <i>both</b> italic</i> plain</p><select><option>1<option>2</select>",
            "<svg><g><path d='M0'/><circle r='1'/></g><foreignObject><div>html</div></foreignObject></svg><p>after",
            "<math><mi>x</mi><mo>=</mo></math><h1>One<h2>Two</h2><button>a<button>b</button>",
            "<script>if (a < b) { document.write('<div><div>') }</script><div><span>open</div>text",
            "<ruby>漢<rt>kan<rt>ji</ruby><textarea><b>raw</b></textarea><table><td>implied",
        ] {
            assert_eq!(
                document(source).html(),
                Html::parse_document(source).html(),
                "{source}"
            );
            assert_eq!(
                fragment(source).html(),
                Html::parse_fragment(source).html(),
                "{source}"
            );
        }
    }

    #[test]
    fn deep_nesting_stops_at_the_limit_and_keeps_its_text() {
        let depth_of = |source: &str| depth(&document(source));
        let deep = format!(
            "{}innermost{}",
            "<div>".repeat(5_000),
            "</div>".repeat(5_000)
        );
        let parsed = document(&deep);
        assert!(depth(&parsed) <= MAX_OPEN + 4, "{}", depth(&parsed));
        assert!(parsed.html().contains("innermost"));
        // Nesting that only looks shallow to a simpler count: an end tag
        // that crosses a block closes nothing, and SVG closed past many
        // children leaves self-closing tags as HTML.
        let crossed = "<span><div></span>".repeat(5_000);
        assert!(depth_of(&crossed) <= 2 * MAX_OPEN, "{}", depth_of(&crossed));
        let after_svg = format!(
            "<svg>{}</svg>{}",
            "<g>".repeat(600),
            "<x-item/>".repeat(5_000)
        );
        assert!(
            depth_of(&after_svg) <= 2 * MAX_OPEN,
            "{}",
            depth_of(&after_svg)
        );
        let tables = "<table><td>".repeat(3_000);
        assert!(depth_of(&tables) <= 4 * MAX_OPEN, "{}", depth_of(&tables));
        // Text after the dropped elements' end tags is back in place.
        let closed = format!(
            "{}deep{}<p>after</p>",
            "<i>".repeat(2_000),
            "</i>".repeat(2_000)
        );
        let parsed = document(&closed);
        assert!(
            parsed.html().contains("<p>after</p>"),
            "{}",
            parsed.html().len()
        );
    }
}
