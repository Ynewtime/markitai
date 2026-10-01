mod anchor;
mod blockquote;
mod br;
mod caption;
mod code;
mod element_util;
mod emphasis;
mod head_body;
mod headings;
mod hr;
mod html;
mod img;
mod li;
mod list;
mod p;
mod pre;
mod span;
mod table;
mod tbody;
mod td_th;
mod thead;
mod tr;

use crate::{
    NodeRef,
    dom_walker::walk_node,
    element_handler::element_util::serialize_element,
    options::{Options, TranslationMode},
    text_util::concat_strings,
};

use super::Element;
use anchor::AnchorElementHandler;
use blockquote::blockquote_handler;
use br::br_handler;
use caption::caption_handler;
use code::code_handler;
use ego_tree::NodeId;
use emphasis::emphasis_handler;
use head_body::head_body_handler;
use headings::headings_handler;
use hr::hr_handler;
use html::html_handler;
use img::img_handler;
use li::list_item_handler;
use list::list_handler;
use p::p_handler;
use pre::pre_handler;
use scraper::Node;
use span::span_handler;
use std::{
    borrow::Cow,
    cell::{Cell, RefCell},
    collections::{HashMap, HashSet},
};
use table::table_handler;
use tbody::tbody_handler;
use td_th::td_th_handler;
use thead::thead_handler;
use tr::tr_handler;

/// The processing result of an `ElementHandler`.
pub struct HandlerResult {
    /// The converted content.
    pub content: String,
    /// See [`Element::markdown_translated`]
    pub markdown_translated: bool,
}

impl From<String> for HandlerResult {
    fn from(value: String) -> Self {
        HandlerResult {
            content: value,
            markdown_translated: true,
        }
    }
}

impl From<&str> for HandlerResult {
    fn from(value: &str) -> Self {
        HandlerResult {
            content: value.to_string(),
            markdown_translated: true,
        }
    }
}

/// Trait for handling the conversion of a specific HTML element to Markdown.
pub trait ElementHandler: Send + Sync {
    /// Append additional content to the end of the converted Markdown.
    fn append(&self) -> Option<String> {
        None
    }

    /// Handle the conversion of an element.
    fn handle(&self, handlers: &dyn Handlers, element: Element) -> Option<HandlerResult>;
}

impl<F> ElementHandler for F
where
    F: (Fn(&dyn Handlers, Element) -> Option<HandlerResult>) + Send + Sync,
{
    fn handle(&self, handlers: &dyn Handlers, element: Element) -> Option<HandlerResult> {
        self(handlers, element)
    }
}

/// Builtin element handlers
pub(crate) struct ElementHandlers {
    pub(crate) handlers: Vec<Box<dyn ElementHandler>>,
    pub(crate) tag_to_handler_indices: HashMap<String, Vec<usize>>,
    pub(crate) options: Options,
}

impl ElementHandlers {
    pub fn new(options: Options) -> Self {
        let mut handlers = Self {
            handlers: Vec::new(),
            tag_to_handler_indices: HashMap::new(),
            options,
        };

        // img
        handlers.add_handler(vec!["img"], img_handler);

        // a
        handlers.add_handler(vec!["a"], AnchorElementHandler::new());

        // list
        handlers.add_handler(vec!["ol", "ul"], list_handler);

        // li
        handlers.add_handler(vec!["li"], list_item_handler);

        // quote
        handlers.add_handler(vec!["blockquote"], blockquote_handler);

        // code
        handlers.add_handler(vec!["code"], code_handler);

        // strong
        handlers.add_handler(vec!["strong", "b"], bold_handler);

        // italic
        handlers.add_handler(vec!["i", "em"], italic_handler);

        // headings
        handlers.add_handler(vec!["h1", "h2", "h3", "h4", "h5", "h6"], headings_handler);

        // br
        handlers.add_handler(vec!["br"], br_handler);

        // hr
        handlers.add_handler(vec!["hr"], hr_handler);

        // table
        handlers.add_handler(vec!["table"], table_handler);

        // td, th
        handlers.add_handler(vec!["td", "th"], td_th_handler);

        // tr
        handlers.add_handler(vec!["tr"], tr_handler);

        // tbody
        handlers.add_handler(vec!["tbody"], tbody_handler);

        // thead
        handlers.add_handler(vec!["thead"], thead_handler);

        // caption
        handlers.add_handler(vec!["caption"], caption_handler);

        // p
        handlers.add_handler(vec!["p"], p_handler);

        // pre
        handlers.add_handler(vec!["pre"], pre_handler);

        // head, body
        handlers.add_handler(vec!["head", "body"], head_body_handler);

        // html
        handlers.add_handler(vec!["html"], html_handler);

        handlers.add_handler(vec!["span"], span_handler);

        // Other block elements. This is taken from the [CommonMark
        // spec](https://spec.commonmark.org/0.31.2/#html-blocks).
        handlers.add_handler(
            vec![
                "address",
                "article",
                "aside",
                "base",
                "basefont",
                "center",
                "col",
                "colgroup",
                "dd",
                "details",
                "dialog",
                "dir",
                "div",
                "dl",
                "dt",
                "fieldset",
                "figcaption",
                "figure",
                "footer",
                "form",
                "frame",
                "frameset",
                "header",
                "iframe",
                "legend",
                "link",
                "main",
                "menu",
                "menuitem",
                "nav",
                "noframes",
                "optgroup",
                "option",
                "param",
                "script",
                "search",
                "section",
                "style",
                "summary",
                "textarea",
                "tfoot",
                "title",
                "track",
            ],
            block_handler,
        );

        handlers
    }

    pub fn add_handler<Handler>(&mut self, tags: Vec<&str>, handler: Handler)
    where
        Handler: ElementHandler + 'static,
    {
        assert!(!tags.is_empty(), "tags cannot be empty.");
        let handler_idx = self.handlers.len();
        self.handlers.push(Box::new(handler));
        // Update tag to handler indices
        for tag in tags {
            let indices = self
                .tag_to_handler_indices
                .entry(tag.to_owned())
                .or_default();
            indices.push(handler_idx);
        }
    }

    pub(crate) fn find_handler(
        &self,
        tag: &str,
        skipped_handlers: usize,
    ) -> Option<&dyn ElementHandler> {
        let handler_indices = self.tag_to_handler_indices.get(tag)?;
        let idx = handler_indices.iter().rev().nth(skipped_handlers)?;
        Some(self.handlers[*idx].as_ref())
    }
}

/// markitai: one conversion's view of the tree, and the `Handlers` that the
/// element handlers receive (upstream implemented `Handlers` on
/// `ElementHandlers` and edited its `Rc` tree in place).
///
/// While walking a node's children, htmd merged each inline element into an
/// identical previous sibling when both held only text: the second element
/// left the parent's children and its text was appended to the first's. The
/// scraper tree is not edited; the walker records the same edits, and every
/// read of children or text in this crate goes through it. A merge therefore
/// takes effect when the parent's children are first walked, as the edit
/// did, and a node whose children are never walked keeps them all.
pub(crate) struct Walker<'h> {
    pub(crate) handlers: &'h ElementHandlers,
    /// Elements merged into their previous sibling.
    removed: RefCell<HashSet<NodeId>>,
    /// The text of a text node after a merge appended to it.
    texts: RefCell<HashMap<NodeId, String>>,
    /// Whether any merge has happened (skips the lookups until one does).
    edited: Cell<bool>,
}

impl<'h> Walker<'h> {
    pub(crate) fn new(handlers: &'h ElementHandlers) -> Self {
        Self {
            handlers,
            removed: RefCell::new(HashSet::new()),
            texts: RefCell::new(HashMap::new()),
            edited: Cell::new(false),
        }
    }

    /// The children of `node` as htmd's tree held them: without merged-away
    /// elements, and without a template's contents, which scraper keeps in a
    /// fragment child and markup5ever_rcdom kept outside the children.
    pub(crate) fn children<'a>(&self, node: NodeRef<'a>) -> impl Iterator<Item = NodeRef<'a>> {
        node.children().filter(move |child| {
            !matches!(child.value(), Node::Fragment)
                && !(self.edited.get() && self.removed.borrow().contains(&child.id()))
        })
    }

    /// The text of a text node as htmd's tree held it.
    pub(crate) fn text<'a>(&self, node: NodeRef<'a>) -> Cow<'a, str> {
        if self.edited.get()
            && let Some(text) = self.texts.borrow().get(&node.id())
        {
            return Cow::Owned(text.clone());
        }
        match node.value() {
            Node::Text(text) => Cow::Borrowed(&text.text),
            _ => Cow::Borrowed(""),
        }
    }

    /// Records the merges htmd made on entering a node's children: each child
    /// that `can_combine` with the previous kept child is removed and its text
    /// appended to that child's text node. (Upstream compared each pair by
    /// index and stepped back after a removal; keeping the last kept child is
    /// the same sequence of comparisons.)
    pub(crate) fn combine_children(&self, node: NodeRef<'_>) {
        let mut children = self.children(node);
        let Some(mut previous) = children.next() else {
            return;
        };
        for child in children {
            let Some(text) = self.can_combine(previous, child) else {
                previous = child;
                continue;
            };
            let target = self
                .children(previous)
                .next()
                .expect("a combined element holds one text node");
            let mut combined = self.text(target).into_owned();
            combined.push_str(&text);
            self.removed.borrow_mut().insert(child.id());
            self.texts.borrow_mut().insert(target.id(), combined);
            self.edited.set(true);
        }
    }

    // Determine if the two nodes are similar, and should therefore be
    // combined. If so, return the text of the second node to simplify the
    // combining process.
    fn can_combine<'a>(&self, n1: NodeRef<'a>, n2: NodeRef<'a>) -> Option<Cow<'a, str>> {
        // To be combined, both nodes must be elements.
        let Node::Element(element1) = n1.value() else {
            return None;
        };
        let Node::Element(element2) = n2.value() else {
            return None;
        };

        // Only combine inline content; block content (for example, one
        // paragraph following another) repetition is expected and should not
        // be combined.
        if crate::dom_walker::is_block_element(element1.name()) {
            return None;
        }

        // Their children must be a single text element.
        let only_text = |node: NodeRef<'a>| {
            let mut children = self.children(node);
            let first = children.next()?;
            (children.next().is_none() && first.value().is_text()).then_some(first)
        };
        let _text1 = only_text(n1)?;
        let text2 = only_text(n2)?;
        let (name1, name2) = (&element1.name, &element2.name);
        // markup5ever_rcdom gave an HTML `template` element template contents.
        let template = |name: &html5ever::QualName| {
            *name.local == *"template" && *name.ns == *"http://www.w3.org/1999/xhtml"
        };
        // Don't combine adjacent hyperlinks.
        if *name1.local != *"a"
            && (name1 == name2
                // Treat `i` and `em` tags as the same element; likewise for `b` and
                // `strong`.
                || *name1.local == *"i" && *name2.local == *"em"
                || *name1.local == *"em" && *name2.local == *"i"
                || *name1.local == *"b" && *name2.local == *"strong"
                || *name1.local == *"strong" && *name2.local == *"b")
            && !template(name1)
            && !template(name2)
            // Scraper sorts attributes by name: equal sets compare equal
            // whatever their source order. (markup5ever_rcdom's flag for a
            // MathML annotation-xml integration point follows from the name
            // and attributes, so equal ones carry equal flags.)
            && element1.attrs == element2.attrs
        {
            Some(self.text(text2))
        } else {
            None
        }
    }

    pub(crate) fn handle_element<'a>(
        &self,
        node: NodeRef<'a>,
        tag: &'a str,
        attrs: &'a scraper::node::Attributes,
        markdown_translated: bool,
        skipped_handlers: usize,
    ) -> Option<HandlerResult> {
        match self.handlers.find_handler(tag, skipped_handlers) {
            Some(handler) => handler.handle(
                self,
                Element {
                    node,
                    tag,
                    attrs,
                    markdown_translated,
                    skipped_handlers,
                },
            ),
            None => {
                if self.handlers.options.translation_mode == TranslationMode::Faithful {
                    Some(HandlerResult {
                        content: serialize_element(
                            self,
                            &Element {
                                node,
                                tag,
                                attrs,
                                markdown_translated,
                                skipped_handlers: 0,
                            },
                        ),
                        markdown_translated: false,
                    })
                } else {
                    // Default behavior: walk children and return their content
                    Some(self.walk_children(node))
                }
            }
        }
    }
}

/// Provides access to the handlers for processing elements and nodes.
///
/// Handlers can use this to delegate to other handlers or recursively process child nodes.
pub trait Handlers {
    /// Skip the current handler and fall back to the previous handler (earlier in registration order).
    fn fallback(&self, element: Element) -> Option<HandlerResult>;

    /// Process a `markup5ever` node through the handlers.
    fn handle(&self, node: NodeRef<'_>) -> Option<HandlerResult>;

    /// Walks children of a node and returns both content and markdown_translated status.
    fn walk_children(&self, node: NodeRef<'_>) -> HandlerResult;

    /// Get the conversion options.
    fn options(&self) -> &Options;

    /// markitai: the children of a node as this conversion sees them, after
    /// the merges of similar adjacent inline elements made so far (the tree
    /// itself is never edited), without a template's contents.
    fn node_children<'a>(&self, node: NodeRef<'a>) -> Vec<NodeRef<'a>>;

    /// markitai: the text of a text node as this conversion sees it, with the
    /// text of any elements merged into its parent appended.
    fn node_text<'a>(&self, node: NodeRef<'a>) -> Cow<'a, str>;
}

impl Handlers for Walker<'_> {
    fn fallback(&self, element: Element) -> Option<HandlerResult> {
        self.handle_element(
            element.node,
            element.tag,
            element.attrs,
            element.markdown_translated,
            element.skipped_handlers + 1,
        )
    }

    fn handle(&self, node: NodeRef<'_>) -> Option<HandlerResult> {
        let mut output = String::new();
        let markdown_translated = walk_node(node, &mut output, self, None, true, false);
        Some(HandlerResult {
            content: output,
            markdown_translated,
        })
    }

    fn walk_children(&self, node: NodeRef<'_>) -> HandlerResult {
        let mut output = String::new();
        let tag = crate::node_util::get_node_tag_name(node);
        let is_block = tag.is_some_and(crate::dom_walker::is_block_element);
        let is_pre = tag.is_some_and(|t| t == "pre" || t == "code") || is_inside_pre(node);
        let markdown_translated =
            crate::dom_walker::walk_children(node, &mut output, self, is_block, is_pre);
        HandlerResult {
            content: output,
            markdown_translated,
        }
    }

    fn options(&self) -> &Options {
        &self.handlers.options
    }

    fn node_children<'a>(&self, node: NodeRef<'a>) -> Vec<NodeRef<'a>> {
        self.children(node).collect()
    }

    fn node_text<'a>(&self, node: NodeRef<'a>) -> Cow<'a, str> {
        self.text(node)
    }
}

pub(crate) fn is_inside_pre(node: NodeRef<'_>) -> bool {
    let mut current = crate::node_util::get_parent_node(node);
    while let Some(parent) = current {
        if let Some(tag) = crate::node_util::get_node_tag_name(parent)
            && (tag == "pre" || tag == "code")
        {
            return true;
        }
        current = crate::node_util::get_parent_node(parent);
    }
    false
}

fn block_handler(handlers: &dyn Handlers, element: Element) -> Option<HandlerResult> {
    if handlers.options().translation_mode == TranslationMode::Pure {
        let content = handlers.walk_children(element.node).content;
        let content = content.trim_matches('\n');
        Some(concat_strings!("\n\n", content, "\n\n").into())
    } else {
        Some(HandlerResult {
            content: serialize_element(handlers, &element),
            markdown_translated: false,
        })
    }
}

fn bold_handler(handlers: &dyn Handlers, element: Element) -> Option<HandlerResult> {
    emphasis_handler(handlers, element, "**")
}

fn italic_handler(handlers: &dyn Handlers, element: Element) -> Option<HandlerResult> {
    emphasis_handler(handlers, element, "*")
}
