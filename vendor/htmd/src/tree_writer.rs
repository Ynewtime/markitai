// markitai: added. Markitai cleans a page and writes the result as markup for
// this crate to convert; parsing that markup again cost a second run of
// html5ever over every page. This writer takes the markup's parts as they are
// written and builds the tree the parser would build from them.
use ego_tree::NodeId;
use html5ever::tendril::StrTendril;
use html5ever::tree_builder::QuirksMode;
use html5ever::{Attribute, LocalName, QualName, local_name, ns};
use scraper::node::{Element, Text};
use scraper::{Html, Node};

/// The tree that [`Html::parse_document`] (and so
/// [`HtmlToMarkdown::convert`](crate::HtmlToMarkdown::convert)) builds from
/// markup, built from the markup's parts as they are written: start tags
/// ([`start`](Self::start), then [`attribute`](Self::attribute) for each of
/// its attributes), end tags and text, in document order.
///
/// The parts stand for the markup `<name a="v">`, `</name>` and text, text and
/// attribute values written with `&`, `<`, `>` and `"` escaped. The writer
/// applies html5ever's tree construction rules where markup written from a
/// tree needs them: the `html`, `head` and `body` elements the parser
/// implies, elements of the head written before the body, a line feed dropped
/// right after `<pre>`, `<listing>` or `<textarea>`, whitespace kept in table
/// rows, end tags of elements the parser closes at once, and the content of a
/// `title`, `textarea`, `style`, `xmp`, `iframe`, `noembed`, `noframes`,
/// `noscript` or `plaintext` element read as the text its markup spells.
/// Markup the parser would rearrange (a block closing an open paragraph,
/// nested headings, list items or links, text moved out of a table) or read
/// in other ways (foreign content, forms, templates, scripts, an encoding
/// declaration, names it would split) is not built: [`finish`](Self::finish)
/// then returns `None`, and the markup has to be written and parsed.
pub struct TreeWriter {
    html: Html,
    /// The open elements, outermost first, with their names.
    open: Vec<(NodeId, LocalName)>,
    mode: Mode,
    head: Option<NodeId>,
    /// A start tag whose attributes are still being written.
    tag: Option<(LocalName, Vec<Attribute>)>,
    /// The markup read as text since a raw-text start tag.
    raw: Option<RawText>,
    /// A line feed starting the next text is dropped (right after `<pre>`,
    /// `<listing>` and `<textarea>`).
    ignore_lf: bool,
    /// The last character written was a carriage return, which the tokenizer
    /// reads as a line feed together with a line feed right after it.
    after_cr: bool,
    /// Something was written: the tokenizer drops a byte order mark only at
    /// the start of its input.
    started: bool,
    /// The markup needs the parser; nothing more is built.
    unsupported: bool,
}

/// The parser's insertion modes up to the body, and after it. Inside the body
/// the current element decides (a table, a table section, a row, a column
/// group, or anything else).
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Before the `html` element.
    Initial,
    BeforeHead,
    InHead,
    AfterHead,
    InBody,
    /// After `</body>` or `</html>`: content goes on into the body.
    AfterBody,
}

/// The content of a raw-text element, as its markup is written.
struct RawText {
    /// The element whose end tag ends the text; none after `<plaintext>`,
    /// whose text runs to the end.
    end: Option<LocalName>,
    /// Character references are read (`title`, `textarea`): the text is the
    /// markup with the escapes read back, which only names could spell.
    unescape: bool,
    content: String,
    /// A start tag is open in `content`; attributes may follow.
    open: bool,
}

/// No element: the document node is current.
static NONE: LocalName = local_name!("");

impl Default for TreeWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeWriter {
    /// A writer for a new document, parsed with [`Html::parse_document`]'s
    /// options.
    pub fn new() -> Self {
        let mut html = Html::new_document();
        // No doctype is written.
        html.quirks_mode = QuirksMode::Quirks;
        Self {
            html,
            open: Vec::new(),
            mode: Mode::Initial,
            head: None,
            tag: None,
            raw: None,
            ignore_lf: false,
            after_cr: false,
            started: false,
            unsupported: false,
        }
    }

    /// Start a start tag, `<name`; its attributes follow.
    pub fn start(&mut self, name: &str) {
        self.flush();
        if let Some(raw) = &mut self.raw {
            raw.close();
            self.unsupported |= raw.unescape && name.contains('&');
            raw.content.push('<');
            raw.content.push_str(name);
            raw.open = true;
            return;
        }
        self.tag = self.tag_name(name).map(|name| (name, Vec::new()));
    }

    /// An attribute of the start tag started last, ` name="value"`.
    pub fn attribute(&mut self, name: &str, value: &str) {
        if let Some(raw) = self.raw.as_mut().filter(|raw| raw.open) {
            self.unsupported |= raw.unescape && name.contains('&');
            raw.content.push(' ');
            raw.content.push_str(name);
            raw.content.push_str("=\"");
            raw.push(value);
            raw.content.push('"');
            return;
        }
        let Some((_, attributes)) = self.tag.as_mut() else {
            // Not after a start tag: the markup would not be a tag at all.
            self.unsupported = true;
            return;
        };
        if name.is_empty()
            || name.bytes().any(|byte| {
                matches!(
                    byte,
                    b'\t' | b'\n' | b'\x0c' | b'\r' | b' ' | b'/' | b'=' | b'>' | 0
                )
            })
        {
            self.unsupported = true;
            return;
        }
        let name = lowercase(name);
        // The tokenizer keeps the first of two attributes with one name.
        if !attributes
            .iter()
            .any(|attribute| attribute.name.local == name)
        {
            attributes.push(Attribute {
                name: QualName::new(None, ns!(), name),
                value: read(value, '\u{fffd}'),
            });
        }
    }

    /// An end tag, `</name>`.
    pub fn end(&mut self, name: &str) {
        self.flush();
        if let Some(raw) = &mut self.raw {
            raw.close();
            if !raw
                .end
                .as_ref()
                .is_some_and(|end| name.eq_ignore_ascii_case(end))
            {
                raw.content.push_str("</");
                raw.content.push_str(name);
                raw.content.push('>');
                return;
            }
            self.end_raw_text();
            self.after_cr = false;
            return;
        }
        if let Some(name) = self.tag_name(name) {
            self.started = true;
            self.after_cr = false;
            self.ignore_lf = false;
            self.end_tag(name);
        }
    }

    /// Text, as it reads (written escaped, it is the same text).
    pub fn text(&mut self, text: &str) {
        self.flush();
        if let Some(raw) = &mut self.raw {
            raw.close();
            raw.push(text);
            return;
        }
        if text.is_empty() || self.unsupported {
            return;
        }
        let mut text = text;
        if !self.started {
            self.started = true;
            text = text.strip_prefix('\u{feff}').unwrap_or(text);
        }
        if std::mem::take(&mut self.after_cr) {
            text = text.strip_prefix('\n').unwrap_or(text);
        }
        if text.is_empty() {
            return;
        }
        self.after_cr = text.ends_with('\r');
        // A NUL is a token of its own, which ends a line feed's exemption.
        for (index, run) in text.split('\0').enumerate() {
            if index > 0 {
                self.ignore_lf = false;
                self.null_character();
            }
            if run.is_empty() {
                continue;
            }
            let run = read(run, '\0');
            let mut run: &str = &run;
            if std::mem::take(&mut self.ignore_lf) {
                run = run.strip_prefix('\n').unwrap_or(run);
            }
            if !run.is_empty() {
                self.characters(run);
            }
        }
    }

    /// The document, or `None` when its markup has to be parsed.
    pub fn finish(mut self) -> Option<Html> {
        self.flush();
        if self.raw.is_some() {
            self.end_raw_text();
        }
        // The elements the parser implies at the end of the input.
        loop {
            match self.mode {
                Mode::Initial => self.root(Vec::new()),
                Mode::BeforeHead => self.implied_head(),
                Mode::InHead => self.pop_head(),
                Mode::AfterHead => self.implied_body(),
                Mode::InBody | Mode::AfterBody => break,
            }
        }
        (!self.unsupported).then_some(self.html)
    }

    /// Process a start tag whose attributes are complete.
    fn flush(&mut self) {
        if let Some((name, attributes)) = self.tag.take() {
            self.started = true;
            self.after_cr = false;
            self.ignore_lf = false;
            if !self.unsupported {
                self.start_tag(name, attributes);
            }
        }
    }

    fn bail(&mut self) {
        self.unsupported = true;
    }

    fn current(&self) -> &LocalName {
        self.open.last().map_or(&NONE, |(_, name)| name)
    }

    fn current_id(&self) -> NodeId {
        self.open
            .last()
            .map_or_else(|| self.html.tree.root().id(), |(id, _)| *id)
    }

    /// A new element appended to `parent`.
    fn create(&mut self, parent: NodeId, name: LocalName, attributes: Vec<Attribute>) -> NodeId {
        let element = Element::new(QualName::new(None, ns!(html), name), attributes);
        let mut parent = self.html.tree.get_mut(parent).expect("open element");
        parent.append(Node::Element(element)).id()
    }

    /// A new element in the current one, then the current one.
    fn insert(&mut self, name: LocalName, attributes: Vec<Attribute>) {
        let id = self.create(self.current_id(), name.clone(), attributes);
        self.open.push((id, name));
    }

    /// A new element in the current one that the parser closes at once.
    fn insert_void(&mut self, name: LocalName, attributes: Vec<Attribute>) {
        self.create(self.current_id(), name, attributes);
    }

    /// Text appended to the current element, joining text before it.
    fn append_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let current = self.current_id();
        let mut parent = self.html.tree.get_mut(current).expect("open element");
        if let Some(mut last) = parent.last_child()
            && let Node::Text(previous) = last.value()
        {
            previous.text.push_slice(text);
            return;
        }
        parent.append(Node::Text(Text {
            text: StrTendril::from_slice(text),
        }));
    }

    /// The parser adds the attributes an element does not have yet.
    fn add_attributes(&mut self, target: NodeId, attributes: Vec<Attribute>) {
        let mut node = self.html.tree.get_mut(target).expect("open element");
        let Node::Element(element) = node.value() else {
            return;
        };
        for attribute in attributes {
            if let Err(index) = element
                .attrs
                .binary_search_by(|(name, _)| name.cmp(&attribute.name))
            {
                element
                    .attrs
                    .insert(index, (attribute.name, attribute.value));
            }
        }
    }

    fn root(&mut self, attributes: Vec<Attribute>) {
        let id = self.create(self.html.tree.root().id(), local_name!("html"), attributes);
        self.open.push((id, local_name!("html")));
        self.mode = Mode::BeforeHead;
    }

    fn implied_head(&mut self) {
        self.insert(local_name!("head"), Vec::new());
        self.head = Some(self.current_id());
        self.mode = Mode::InHead;
    }

    fn pop_head(&mut self) {
        self.open.pop();
        self.mode = Mode::AfterHead;
    }

    fn implied_body(&mut self) {
        self.insert(local_name!("body"), Vec::new());
        self.mode = Mode::InBody;
    }

    /// Whether `name` is open in html5ever's default scope (with `button`, in
    /// the button scope that bounds paragraphs).
    fn in_scope(&self, name: &LocalName, button: bool) -> bool {
        for (_, open) in self.open.iter().rev() {
            if open == name {
                return true;
            }
            match *open {
                local_name!("applet")
                | local_name!("caption")
                | local_name!("html")
                | local_name!("table")
                | local_name!("td")
                | local_name!("th")
                | local_name!("marquee")
                | local_name!("object")
                | local_name!("select")
                | local_name!("template") => return false,
                local_name!("button") if button => return false,
                _ => {}
            }
        }
        false
    }

    /// Whether a start tag would first close an open paragraph.
    fn closes_paragraph(&self) -> bool {
        self.in_scope(&local_name!("p"), true)
    }

    /// The text of a raw-text element, which then closes.
    fn end_raw_text(&mut self) {
        let Some(mut raw) = self.raw.take() else {
            return;
        };
        raw.close();
        let content = read(&raw.content, '\u{fffd}');
        let mut content: &str = &content;
        if std::mem::take(&mut self.ignore_lf) {
            content = content.strip_prefix('\n').unwrap_or(content);
        }
        self.append_text(content);
        if raw.end.is_some() {
            self.open.pop();
        }
    }

    /// The element of a start tag whose content is raw text.
    fn raw_element(&mut self, name: LocalName, attributes: Vec<Attribute>, unescape: bool) {
        let end = (name != local_name!("plaintext")).then(|| name.clone());
        self.insert(name, attributes);
        self.raw = Some(RawText {
            end,
            unescape,
            content: String::new(),
            open: false,
        });
    }

    fn characters(&mut self, text: &str) {
        let mut text = text;
        loop {
            let whitespace = text.len()
                - text
                    .trim_start_matches(|ch: char| ch.is_ascii_whitespace())
                    .len();
            match self.mode {
                Mode::Initial | Mode::BeforeHead => {
                    text = &text[whitespace..];
                    if text.is_empty() {
                        return;
                    }
                    if self.mode == Mode::Initial {
                        self.root(Vec::new());
                    } else {
                        self.implied_head();
                    }
                }
                Mode::InHead | Mode::AfterHead => {
                    self.append_text(&text[..whitespace]);
                    text = &text[whitespace..];
                    if text.is_empty() {
                        return;
                    }
                    if self.mode == Mode::InHead {
                        self.pop_head();
                    } else {
                        self.implied_body();
                    }
                }
                Mode::AfterBody | Mode::InBody => {
                    if whitespace < text.len() {
                        // Text in a table, a section, a row or a column
                        // group moves before the table unless it is space.
                        if matches!(
                            *self.current(),
                            local_name!("table")
                                | local_name!("tbody")
                                | local_name!("thead")
                                | local_name!("tfoot")
                                | local_name!("tr")
                                | local_name!("colgroup")
                        ) {
                            return self.bail();
                        }
                        self.mode = Mode::InBody;
                    }
                    return self.append_text(text);
                }
            }
        }
    }

    fn null_character(&mut self) {
        // Dropped in the body and in tables; before the body it implies
        // elements, after it reopens the body.
        if self.mode != Mode::InBody {
            self.bail();
        }
    }

    fn start_tag(&mut self, name: LocalName, attributes: Vec<Attribute>) {
        loop {
            match self.mode {
                Mode::Initial => {
                    if name == local_name!("html") {
                        return self.root(attributes);
                    }
                    self.root(Vec::new());
                }
                Mode::BeforeHead => match name {
                    local_name!("html") => return self.add_root_attributes(attributes),
                    local_name!("head") => {
                        self.insert(name, attributes);
                        self.head = Some(self.current_id());
                        self.mode = Mode::InHead;
                        return;
                    }
                    _ => self.implied_head(),
                },
                Mode::InHead => match name {
                    local_name!("html") => return self.add_root_attributes(attributes),
                    local_name!("base")
                    | local_name!("basefont")
                    | local_name!("bgsound")
                    | local_name!("link")
                    | local_name!("meta")
                    | local_name!("title")
                    | local_name!("noframes")
                    | local_name!("style")
                    | local_name!("noscript")
                    | local_name!("script")
                    | local_name!("template") => return self.head_element(name, attributes),
                    local_name!("head") => return,
                    _ => self.pop_head(),
                },
                Mode::AfterHead => match name {
                    local_name!("html") => return self.add_root_attributes(attributes),
                    local_name!("body") => {
                        self.insert(name, attributes);
                        self.mode = Mode::InBody;
                        return;
                    }
                    local_name!("base")
                    | local_name!("basefont")
                    | local_name!("bgsound")
                    | local_name!("link")
                    | local_name!("meta")
                    | local_name!("noframes")
                    | local_name!("script")
                    | local_name!("style")
                    | local_name!("template")
                    | local_name!("title") => {
                        // Into the head, which is no longer open.
                        let Some(head) = self.head else {
                            return self.bail();
                        };
                        let index = self.open.len();
                        self.open.push((head, local_name!("head")));
                        self.head_element(name, attributes);
                        self.open.remove(index);
                        return;
                    }
                    local_name!("head") => return,
                    local_name!("frameset") => return self.bail(),
                    _ => self.implied_body(),
                },
                Mode::AfterBody => self.mode = Mode::InBody,
                Mode::InBody => return self.body_start_tag(name, attributes),
            }
        }
    }

    fn add_root_attributes(&mut self, attributes: Vec<Attribute>) {
        if let Some(&(root, _)) = self.open.first() {
            self.add_attributes(root, attributes);
        }
    }

    /// An element of the head, as the parser reads it wherever it stands.
    fn head_element(&mut self, name: LocalName, attributes: Vec<Attribute>) {
        match name {
            local_name!("base")
            | local_name!("basefont")
            | local_name!("bgsound")
            | local_name!("link")
            | local_name!("meta") => {
                // A `meta` naming an encoding makes the parser's caller
                // read on, where a byte order mark would be dropped.
                if attributes.iter().any(|attribute| {
                    matches!(
                        attribute.name.local,
                        local_name!("charset") | local_name!("http-equiv")
                    )
                }) {
                    return self.bail();
                }
                self.insert_void(name, attributes);
            }
            local_name!("title") => self.raw_element(name, attributes, true),
            local_name!("noframes") | local_name!("style") | local_name!("noscript") => {
                self.raw_element(name, attributes, false)
            }
            _ => self.bail(),
        }
    }

    fn body_start_tag(&mut self, name: LocalName, attributes: Vec<Attribute>) {
        match *self.current() {
            local_name!("table") => {
                return match name {
                    local_name!("caption")
                    | local_name!("colgroup")
                    | local_name!("tbody")
                    | local_name!("tfoot")
                    | local_name!("thead") => self.insert(name, attributes),
                    local_name!("style") => self.raw_element(name, attributes, false),
                    _ => self.bail(),
                };
            }
            local_name!("tbody") | local_name!("thead") | local_name!("tfoot") => {
                return if name == local_name!("tr") {
                    self.insert(name, attributes)
                } else {
                    self.bail()
                };
            }
            local_name!("tr") => {
                return if matches!(name, local_name!("td") | local_name!("th")) {
                    self.insert(name, attributes)
                } else {
                    self.bail()
                };
            }
            local_name!("colgroup") => {
                return if name == local_name!("col") {
                    self.insert_void(name, attributes)
                } else {
                    self.bail()
                };
            }
            _ => {}
        }
        match name {
            local_name!("html") => self.add_root_attributes(attributes),
            local_name!("base")
            | local_name!("basefont")
            | local_name!("bgsound")
            | local_name!("link")
            | local_name!("meta")
            | local_name!("noframes")
            | local_name!("script")
            | local_name!("style")
            | local_name!("template")
            | local_name!("title") => self.head_element(name, attributes),
            local_name!("body") => {
                if let Some(&(body, ref open)) = self.open.get(1)
                    && *open == local_name!("body")
                {
                    self.add_attributes(body, attributes);
                }
            }
            local_name!("address")
            | local_name!("article")
            | local_name!("aside")
            | local_name!("blockquote")
            | local_name!("center")
            | local_name!("details")
            | local_name!("dialog")
            | local_name!("dir")
            | local_name!("div")
            | local_name!("dl")
            | local_name!("fieldset")
            | local_name!("figcaption")
            | local_name!("figure")
            | local_name!("footer")
            | local_name!("header")
            | local_name!("hgroup")
            | local_name!("main")
            | local_name!("menu")
            | local_name!("nav")
            | local_name!("ol")
            | local_name!("p")
            | local_name!("search")
            | local_name!("section")
            | local_name!("summary")
            | local_name!("ul") => {
                if self.closes_paragraph() {
                    return self.bail();
                }
                self.insert(name, attributes);
            }
            local_name!("h1")
            | local_name!("h2")
            | local_name!("h3")
            | local_name!("h4")
            | local_name!("h5")
            | local_name!("h6") => {
                if self.closes_paragraph() || heading(self.current()) {
                    return self.bail();
                }
                self.insert(name, attributes);
            }
            local_name!("pre") | local_name!("listing") => {
                if self.closes_paragraph() {
                    return self.bail();
                }
                self.insert(name, attributes);
                self.ignore_lf = true;
            }
            local_name!("li") | local_name!("dd") | local_name!("dt") => {
                for (_, open) in self.open.iter().rev() {
                    let closes = if name == local_name!("li") {
                        *open == local_name!("li")
                    } else {
                        matches!(*open, local_name!("dd") | local_name!("dt"))
                    };
                    if closes {
                        return self.bail();
                    }
                    if special(open)
                        && !matches!(
                            *open,
                            local_name!("address") | local_name!("div") | local_name!("p")
                        )
                    {
                        break;
                    }
                }
                if self.closes_paragraph() {
                    return self.bail();
                }
                self.insert(name, attributes);
            }
            local_name!("plaintext") | local_name!("xmp") => {
                if self.closes_paragraph() {
                    return self.bail();
                }
                self.raw_element(name, attributes, false);
            }
            local_name!("button") | local_name!("nobr") => {
                if self.in_scope(&name, false) {
                    return self.bail();
                }
                self.insert(name, attributes);
            }
            local_name!("a") => {
                // A link open since the last cell, caption or object.
                for (_, open) in self.open.iter().rev() {
                    match *open {
                        local_name!("a") => return self.bail(),
                        local_name!("applet")
                        | local_name!("object")
                        | local_name!("marquee")
                        | local_name!("template")
                        | local_name!("td")
                        | local_name!("th")
                        | local_name!("caption") => break,
                        _ => {}
                    }
                }
                self.insert(name, attributes);
            }
            local_name!("area")
            | local_name!("br")
            | local_name!("embed")
            | local_name!("img")
            | local_name!("keygen")
            | local_name!("wbr")
            | local_name!("param")
            | local_name!("source")
            | local_name!("track") => self.insert_void(name, attributes),
            local_name!("input") | local_name!("hr") => {
                if self.in_scope(&local_name!("select"), false)
                    || (name == local_name!("hr") && self.closes_paragraph())
                {
                    return self.bail();
                }
                self.insert_void(name, attributes);
            }
            local_name!("image") => self.insert_void(local_name!("img"), attributes),
            local_name!("textarea") => {
                self.raw_element(name, attributes, true);
                self.ignore_lf = true;
            }
            local_name!("iframe") | local_name!("noembed") | local_name!("noscript") => {
                self.raw_element(name, attributes, false)
            }
            local_name!("rb") | local_name!("rtc") | local_name!("rp") | local_name!("rt") => {
                // In a ruby, the parser first closes the elements whose end
                // tags it implies (all but `rtc` for `rp` and `rt`).
                let current = self.current();
                let implied = matches!(
                    *current,
                    local_name!("dd")
                        | local_name!("dt")
                        | local_name!("li")
                        | local_name!("option")
                        | local_name!("optgroup")
                        | local_name!("p")
                        | local_name!("rb")
                        | local_name!("rp")
                        | local_name!("rt")
                ) || (*current == local_name!("rtc")
                    && matches!(name, local_name!("rb") | local_name!("rtc")));
                if implied && self.in_scope(&local_name!("ruby"), false) {
                    return self.bail();
                }
                self.insert(name, attributes);
            }
            // Forms, selects, frames, foreign content, and table parts
            // outside a table.
            local_name!("form")
            | local_name!("frameset")
            | local_name!("select")
            | local_name!("option")
            | local_name!("optgroup")
            | local_name!("math")
            | local_name!("svg")
            | local_name!("caption")
            | local_name!("col")
            | local_name!("colgroup")
            | local_name!("frame")
            | local_name!("head")
            | local_name!("tbody")
            | local_name!("td")
            | local_name!("tfoot")
            | local_name!("th")
            | local_name!("thead")
            | local_name!("tr") => self.bail(),
            _ => self.insert(name, attributes),
        }
    }

    fn end_tag(&mut self, name: LocalName) {
        if self.unsupported {
            return;
        }
        loop {
            match (self.mode, &name) {
                (Mode::InBody, _) => break,
                (Mode::AfterBody, &local_name!("html")) => return,
                (Mode::AfterBody, _) => self.mode = Mode::InBody,
                (Mode::InHead, &local_name!("head")) => return self.pop_head(),
                (_, &local_name!("br")) => return self.bail(),
                // Before the body only these end tags count; they end the
                // head and imply the elements before them.
                (Mode::Initial | Mode::BeforeHead, &local_name!("head"))
                | (_, &local_name!("html") | &local_name!("body")) => match self.mode {
                    Mode::Initial => self.root(Vec::new()),
                    Mode::BeforeHead => self.implied_head(),
                    Mode::InHead => self.pop_head(),
                    _ => self.implied_body(),
                },
                _ => return,
            }
        }
        let current = self.current();
        if matches!(name, local_name!("body") | local_name!("html")) {
            if *current == local_name!("body") {
                self.mode = Mode::AfterBody;
            } else {
                self.bail();
            }
        } else if *current == name {
            if matches!(name, local_name!("form") | local_name!("template")) {
                return self.bail();
            }
            self.open.pop();
        } else if !matches!(
            name,
            local_name!("param")
                | local_name!("track")
                | local_name!("keygen")
                | local_name!("basefont")
                | local_name!("bgsound")
                | local_name!("image")
        ) {
            // Only the end tags of elements the parser closed at once are
            // ignored as the parser ignores them.
            self.bail();
        }
    }

    /// A tag name as the tokenizer reads it, or `None` (and the markup needs
    /// the parser) when it would not read it as one name.
    fn tag_name(&mut self, name: &str) -> Option<LocalName> {
        let bytes = name.as_bytes();
        if !bytes.first().is_some_and(u8::is_ascii_alphabetic)
            || bytes.iter().any(|byte| {
                matches!(
                    byte,
                    b'\t' | b'\n' | b'\x0c' | b'\r' | b' ' | b'/' | b'>' | 0
                )
            })
        {
            self.unsupported = true;
            return None;
        }
        Some(lowercase(name))
    }
}

impl RawText {
    /// Ends the start tag open in `content`.
    fn close(&mut self) {
        if std::mem::take(&mut self.open) {
            self.content.push('>');
        }
    }

    /// Text or an attribute value as its markup reads here.
    fn push(&mut self, text: &str) {
        if self.unescape {
            self.content.push_str(text);
            return;
        }
        for ch in text.chars() {
            match ch {
                '&' => self.content.push_str("&amp;"),
                '<' => self.content.push_str("&lt;"),
                '>' => self.content.push_str("&gt;"),
                '"' => self.content.push_str("&quot;"),
                ch => self.content.push(ch),
            }
        }
    }
}

fn heading(name: &LocalName) -> bool {
    matches!(
        *name,
        local_name!("h1")
            | local_name!("h2")
            | local_name!("h3")
            | local_name!("h4")
            | local_name!("h5")
            | local_name!("h6")
    )
}

/// html5ever's special elements, which bound the elements a list item closes.
fn special(name: &LocalName) -> bool {
    heading(name)
        || matches!(
            *name,
            local_name!("address")
                | local_name!("applet")
                | local_name!("area")
                | local_name!("article")
                | local_name!("aside")
                | local_name!("base")
                | local_name!("basefont")
                | local_name!("bgsound")
                | local_name!("blockquote")
                | local_name!("body")
                | local_name!("br")
                | local_name!("button")
                | local_name!("caption")
                | local_name!("center")
                | local_name!("col")
                | local_name!("colgroup")
                | local_name!("dd")
                | local_name!("details")
                | local_name!("dir")
                | local_name!("div")
                | local_name!("dl")
                | local_name!("dt")
                | local_name!("embed")
                | local_name!("fieldset")
                | local_name!("figcaption")
                | local_name!("figure")
                | local_name!("footer")
                | local_name!("form")
                | local_name!("frame")
                | local_name!("frameset")
                | local_name!("head")
                | local_name!("header")
                | local_name!("hgroup")
                | local_name!("hr")
                | local_name!("html")
                | local_name!("iframe")
                | local_name!("img")
                | local_name!("input")
                | local_name!("isindex")
                | local_name!("li")
                | local_name!("link")
                | local_name!("listing")
                | local_name!("main")
                | local_name!("marquee")
                | local_name!("menu")
                | local_name!("meta")
                | local_name!("nav")
                | local_name!("noembed")
                | local_name!("noframes")
                | local_name!("noscript")
                | local_name!("object")
                | local_name!("ol")
                | local_name!("p")
                | local_name!("param")
                | local_name!("plaintext")
                | local_name!("pre")
                | local_name!("script")
                | local_name!("section")
                | local_name!("select")
                | local_name!("source")
                | local_name!("style")
                | local_name!("summary")
                | local_name!("table")
                | local_name!("tbody")
                | local_name!("td")
                | local_name!("template")
                | local_name!("textarea")
                | local_name!("tfoot")
                | local_name!("th")
                | local_name!("thead")
                | local_name!("title")
                | local_name!("tr")
                | local_name!("track")
                | local_name!("ul")
                | local_name!("wbr")
                | local_name!("xmp")
        )
}

/// Text as the tokenizer reads it: line breaks as line feeds, and a NUL as
/// `nul` (in text, where it is a token of its own, there is none).
fn read(text: &str, nul: char) -> StrTendril {
    if !text.contains(['\r', '\0']) {
        return StrTendril::from_slice(text);
    }
    let mut read = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\r' => {
                chars.next_if_eq(&'\n');
                read.push('\n');
            }
            '\0' => read.push(nul),
            ch => read.push(ch),
        }
    }
    StrTendril::from(read)
}

/// A name with ASCII capitals lowercased, as the tokenizer reads tag and
/// attribute names.
fn lowercase(name: &str) -> LocalName {
    if name.bytes().any(|byte| byte.is_ascii_uppercase()) {
        LocalName::from(name.to_ascii_lowercase())
    } else {
        LocalName::from(name)
    }
}
