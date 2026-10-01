mod dom_walker;
pub mod element_handler;
mod html_escape;
pub(crate) mod node_util;
pub mod options;
pub(crate) mod text_util;
// markitai: builds the tree of markup from its parts as they are written.
mod tree_writer;

use dom_walker::walk_node;
use element_handler::{ElementHandler, ElementHandlers, Walker};
use options::Options;
use scraper::Html;
// Export publicly, providing an interface to the
// markitai: the tree is scraper's (`ego_tree`) rather than markup5ever_rcdom's.
pub use scraper::Node;

use crate::element_handler::Handlers;
// markitai: see `tree_writer`.
pub use tree_writer::TreeWriter;

/// markitai: a node of the scraper tree, the type handlers receive and walk.
pub type NodeRef<'a> = ego_tree::NodeRef<'a, Node>;

/// Convert HTML to Markdown.
///
/// Example:
///
/// ```
/// use htmd::convert;
///
/// let md = convert("<h1>Hello</h1>").unwrap();
/// assert_eq!("# Hello", md);
/// ```
pub fn convert(html: &str) -> Result<String, std::io::Error> {
    HtmlToMarkdown::new().convert(html)
}

/// The DOM element.
pub struct Element<'a> {
    /// The html5ever node of the element.
    // markitai: a scraper tree node instead of an `Rc` markup5ever_rcdom node.
    pub node: NodeRef<'a>,
    /// The tag name.
    pub tag: &'a str,
    /// The attribute list.
    // markitai: scraper's attributes, which it keeps sorted by name rather
    // than in source order.
    pub attrs: &'a scraper::node::Attributes,
    /// When true, this element's children were all translated using Markdown,
    /// not HTML. This is only needed in faithful translation mode (see the
    /// `Options`): for code blocks, translating a `<pre><code>` sequence to
    /// Markdown, not HTML, requires a Markdown translated `<code>` block;
    /// likewise, translating lists ((`<ol>`/`<ul>`)`<li>`) to Markdown requires
    /// all `<li>` elements are translated to Markdown.
    pub markdown_translated: bool,
    /// The number of handlers to skip for this element.
    pub(crate) skipped_handlers: usize,
}

impl<'a> Element<'a> {
    /// markitai: the value of the first attribute with this local name (in any
    /// namespace), as the built-in handlers look one up in `attrs`.
    pub fn attr(&self, name: &str) -> Option<&'a str> {
        self.attrs
            .iter()
            .find(|(attr, _)| attr.local.as_ref() == name)
            .map(|(_, value)| value.as_ref())
    }
}

/// The html-to-markdown converter.
///
/// # Example
/// ```
/// use htmd::{Element, HtmlToMarkdown};
///
/// // One-liner
/// let md = HtmlToMarkdown::new().convert("<h1>Hello</h1>").unwrap();
/// assert_eq!("# Hello", md);
///
/// // Or use the builder pattern
/// let converter = HtmlToMarkdown::builder()
///     .skip_tags(vec!["img"])
///     .build();
/// let md = converter.convert("<img src=\"https://example.com\">").unwrap();
/// // img is ignored
/// assert_eq!("", md);
/// ```
pub struct HtmlToMarkdown {
    handlers: ElementHandlers,
    // markitai: read only with the `scripting-option` feature.
    #[cfg_attr(not(feature = "scripting-option"), allow(dead_code))]
    scripting_enabled: bool,
}

impl Default for HtmlToMarkdown {
    fn default() -> Self {
        Self::new()
    }
}

impl HtmlToMarkdown {
    /// Create a new converter.
    pub fn new() -> Self {
        let options = Options::default();
        let handlers = ElementHandlers::new(options);
        Self {
            handlers,
            scripting_enabled: true,
        }
    }

    pub(crate) fn from_params(handlers: ElementHandlers, scripting_enabled: bool) -> Self {
        Self {
            handlers,
            scripting_enabled,
        }
    }

    /// Create a new [HtmlToMarkdownBuilder].
    pub fn builder() -> HtmlToMarkdownBuilder {
        HtmlToMarkdownBuilder::new()
    }

    /// Convert HTML to a DOM tree.
    // markitai: returns scraper's document; its root node is `tree.root()`.
    pub fn html_to_tree(&self, html: &str) -> std::io::Result<Html> {
        // markitai: scraper's own entry point when the options are its
        // defaults, so this crate adds no second instance of the parser.
        // Without the `scripting-option` feature they always are.
        #[cfg(feature = "scripting-option")]
        if !self.scripting_enabled {
            use html5ever::tendril::TendrilSink;
            use html5ever::tree_builder::TreeBuilderOpts;
            use html5ever::{ParseOpts, parse_document};
            use scraper::HtmlTreeSink;

            // Feed the whole UTF-8 string to the parser in one shot. This avoids
            // the 4 KiB chunked `read_from` path and the UTF-8 lossy decoder layer
            // used by `from_utf8()`, both of which add overhead when the input is
            // already a valid `&str`.
            let dom = parse_document(
                HtmlTreeSink::new(Html::new_document()),
                ParseOpts {
                    tree_builder: TreeBuilderOpts {
                        scripting_enabled: self.scripting_enabled,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .one(html);

            return Ok(dom);
        }
        Ok(Html::parse_document(html))
    }

    /// Convert a DOM tree to Markdown. For convenience, `Node` is re-exported;
    /// simply `use htmd::Node;` to access this type.
    // markitai: takes any node of a scraper tree (`html.tree.root()` for a
    // whole document) and leaves the tree unchanged.
    pub fn tree_to_markdown(&self, tree: NodeRef<'_>) -> String {
        let mut content = String::new();

        let walker = Walker::new(&self.handlers);
        walk_node(tree, &mut content, &walker, None, true, false);

        // Trim leading/trailing newlines in place instead of allocating a copy.
        let start = content.len() - content.trim_start_matches('\n').len();
        if start > 0 {
            content.drain(..start);
        }
        let end = content.trim_end_matches('\n').len();
        content.truncate(end);

        let mut append = String::new();
        for handler in &self.handlers.handlers {
            let Some(append_content) = handler.append() else {
                continue;
            };
            append.push_str(&append_content);
        }

        content.push_str(append.trim_end_matches('\n'));

        content
    }

    /// Convert HTML to Markdown.
    pub fn convert(&self, html: &str) -> std::io::Result<String> {
        Ok(self.tree_to_markdown(self.html_to_tree(html)?.tree.root()))
    }
}

/// The [HtmlToMarkdown] builder for advanced configurations.
pub struct HtmlToMarkdownBuilder {
    handlers: ElementHandlers,
    scripting_enabled: bool,
}

impl Default for HtmlToMarkdownBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HtmlToMarkdownBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        let options = Options::default();
        let handlers = ElementHandlers::new(options);
        Self {
            handlers,
            scripting_enabled: true,
        }
    }

    /// Set converting options.
    pub fn options(mut self, options: Options) -> Self {
        self.handlers.options = options;
        self
    }

    /// Skip a group of tags when converting.
    pub fn skip_tags(self, tags: Vec<&str>) -> Self {
        self.add_handler(tags, |_: &dyn Handlers, _: Element| None)
    }

    /// Apply a custom element handler for a group of tags.
    ///
    /// # Example
    ///
    /// ```
    /// use htmd::{Element, HtmlToMarkdownBuilder, element_handler::Handlers};
    ///
    /// let mut handlers = HtmlToMarkdownBuilder::new()
    ///    .add_handler(vec!["img"], |_handlers: &dyn Handlers, _: Element| {
    ///        // Skip the img tag when converting.
    ///        None
    ///    })
    ///    .add_handler(vec!["video"], |_handlers: &dyn Handlers, element: Element| {
    ///        // Handle the video tag.
    ///        todo!("Return some text to represent this video element.")
    ///    });
    /// ```
    pub fn add_handler<Handler>(mut self, tags: Vec<&str>, handler: Handler) -> Self
    where
        Handler: ElementHandler + 'static,
    {
        self.handlers.add_handler(tags, handler);
        self
    }

    /// Option for html5ever parsing. If true, the content of <noscript> tags will be converted to raw text.
    /// If false, the content of <noscript> tags will be parsed as normal DOM.
    // markitai: only with the `scripting-option` feature (on by default).
    #[cfg(feature = "scripting-option")]
    pub fn scripting_enabled(mut self, enabled: bool) -> Self {
        self.scripting_enabled = enabled;
        self
    }

    /// Create a new [HtmlToMarkdown].
    pub fn build(self) -> HtmlToMarkdown {
        HtmlToMarkdown::from_params(self.handlers, self.scripting_enabled)
    }
}
