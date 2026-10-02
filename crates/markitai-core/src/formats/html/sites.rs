//! Readers for reading sites whose pages the generic reader reads poorly.
//!
//! A reader runs only for a page that identifies itself as one of the site's
//! pages: its own address, or, for a page saved from a browser (which has no
//! address), the `<!-- saved from url=... -->` comment, the canonical link, the
//! `og:url` property or `<base href>` that names the site, and, when a page
//! names no address at all, the site's own markers. It reads what the page
//! itself serves (markup, or the JSON its scripts embed for their own
//! rendering) and builds a clean article page that the generic reader then
//! converts, so tables, code, images and links come out as on every other
//! page. When a reader finds none of its markers, or what it builds converts
//! to nothing, the generic reader reads the original page.

mod bilibili;
mod blogs;
mod wechat;
mod zhihu;

use super::{Attribute, escaped};
use scraper::{ElementRef, Html, Node};
use serde_json::{Map, Value};
use std::sync::OnceLock;
use url::Url;

/// The marker a built page carries, so that it is never read a second time.
const MARKER: &str = "markitai-reader";

/// What a reader made of a page.
pub(super) struct Reading {
    /// A cleaned page for the generic reader, in place of the original.
    pub page: Option<String>,
    /// Metadata that replaces what the generic reader finds.
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Site {
    Zhihu,
    WeChat,
    Cnblogs,
    Jianshu,
    Kr36,
    Bilibili,
    Oschina,
}

/// The site a host belongs to.
fn site_of(host: &str) -> Option<Site> {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    let within = |domain: &str| {
        host == domain
            || host
                .strip_suffix(domain)
                .is_some_and(|rest| rest.ends_with('.'))
    };
    if within("zhihu.com") {
        Some(Site::Zhihu)
    } else if host == "mp.weixin.qq.com" {
        Some(Site::WeChat)
    } else if within("cnblogs.com") {
        Some(Site::Cnblogs)
    } else if within("jianshu.com") {
        Some(Site::Jianshu)
    } else if within("36kr.com") {
        Some(Site::Kr36)
    } else if within("bilibili.com") {
        Some(Site::Bilibili)
    } else if within("oschina.net") {
        Some(Site::Oschina)
    } else {
        None
    }
}

/// The addresses a saved page names for itself (read in one walk over the
/// page's head and the comments before it, which ends at the body) and, for a
/// page that names none, whether it carries a site's own markers.
struct Identity {
    addresses: Vec<Url>,
    zhihu_data: bool,
    wechat_content: bool,
}

fn web_address(value: &str) -> Option<Url> {
    Url::parse(value.trim())
        .ok()
        .filter(|url| matches!(url.scheme(), "http" | "https") && url.host_str().is_some())
}

/// The address in a `saved from url=(0043)https://...` comment, as Chrome,
/// Firefox's "Save Page As" and SingleFile write it.
fn saved_from(comment: &str) -> Option<Url> {
    let lower = comment.to_ascii_lowercase();
    let at = lower.find("saved from url=")? + "saved from url=".len();
    let rest = comment[at..].trim_start();
    let rest = match rest.strip_prefix('(') {
        Some(after) => after.split_once(')')?.1,
        None => rest,
    };
    web_address(rest.split_whitespace().next()?)
}

fn identity(document: &Html) -> Identity {
    let mut found = Identity {
        addresses: Vec::new(),
        zhihu_data: false,
        wechat_content: false,
    };
    for node in document.tree.nodes() {
        match node.value() {
            Node::Comment(comment) => {
                if let Some(url) = saved_from(comment) {
                    found.addresses.push(url);
                }
            }
            Node::Element(element) => match element.name() {
                "body" => break,
                "link" if element.attribute("rel").is_some_and(super::canonical_rel) => {
                    found
                        .addresses
                        .extend(element.attribute("href").and_then(web_address));
                }
                "meta"
                    if element
                        .attribute("property")
                        .or_else(|| element.attribute("name"))
                        .is_some_and(|name| name.trim().eq_ignore_ascii_case("og:url")) =>
                {
                    found
                        .addresses
                        .extend(element.attribute("content").and_then(web_address));
                }
                "base" => {
                    found
                        .addresses
                        .extend(element.attribute("href").and_then(web_address));
                }
                _ => {}
            },
            _ => {}
        }
    }
    // A page that names no address is known by its markers, which follow the head.
    if found.addresses.is_empty() {
        for node in document.tree.nodes() {
            if let Node::Element(element) = node.value() {
                match (element.name(), element.id()) {
                    ("script", Some("js-initialData")) => found.zhihu_data = true,
                    (_, Some("js_content")) => found.wechat_content = true,
                    _ => {}
                }
                if found.zhihu_data || found.wechat_content {
                    break;
                }
            }
        }
    }
    found
}

/// A page's own address and the site it belongs to.
fn recognize(document: &Html, base: Option<&Url>) -> Option<(Site, Option<Url>)> {
    if let Some(base) = base {
        return site_of(base.host_str()?).map(|site| (site, Some(base.clone())));
    }
    let found = identity(document);
    if let Some((site, url)) = found
        .addresses
        .iter()
        .find_map(|url| Some((site_of(url.host_str()?)?, url.clone())))
    {
        return Some((site, Some(url)));
    }
    // A page that names no address is known by its own markers.
    if found.addresses.is_empty() {
        if found.zhihu_data {
            return Some((Site::Zhihu, None));
        }
        if found.wechat_content {
            return Some((Site::WeChat, None));
        }
    }
    None
}

/// What the site's reader makes of the page, or nothing when the page is not
/// one of the site's, or has none of the markers the reader needs.
pub(super) fn read(document: &Html, base: Option<&Url>) -> Option<Reading> {
    let (site, address) = recognize(document, base)?;
    if marked(document) {
        return None;
    }
    let address = address.as_ref();
    match site {
        Site::Zhihu => zhihu::read(document, address).map(Article::build),
        Site::WeChat => wechat::read(document, address).map(Article::build),
        Site::Cnblogs => blogs::cnblogs(document).map(Article::build),
        Site::Jianshu => blogs::jianshu(document).map(Article::build),
        Site::Kr36 => blogs::kr36(document),
        Site::Bilibili => bilibili::read(document).map(Article::build),
        Site::Oschina => blogs::oschina(document).map(Article::build),
    }
}

/// Whether the page was built by a reader.
fn marked(document: &Html) -> bool {
    document.tree.nodes().any(|node| {
        matches!(node.value(), Node::Element(element)
            if element.name() == "meta" && element.attribute("name") == Some(MARKER))
    })
}

/// An article a reader found, as the generic reader will read it.
#[derive(Default)]
struct Article {
    reader: &'static str,
    title: String,
    author: Option<String>,
    published: Option<String>,
    site: Option<String>,
    canonical: Option<String>,
    /// Lines under the title: author, account, date and the like.
    byline: Vec<String>,
    /// The article's own markup.
    body: String,
    metadata: Map<String, Value>,
}

impl Article {
    fn build(self) -> Reading {
        let mut page = String::from("<!doctype html><html><head><meta charset=\"utf-8\">");
        let mut meta = |name: &str, value: &str, property: bool| {
            page.push_str(if property {
                "<meta property=\""
            } else {
                "<meta name=\""
            });
            page.push_str(name);
            page.push_str("\" content=\"");
            escaped(value, &mut page);
            page.push_str("\">");
        };
        meta(MARKER, self.reader, false);
        meta("og:title", &self.title, true);
        if let Some(author) = &self.author {
            meta("author", author, false);
        }
        if let Some(published) = &self.published {
            meta("article:published_time", published, true);
        }
        if let Some(site) = &self.site {
            meta("og:site_name", site, true);
        }
        page.push_str("<title>");
        escaped(&self.title, &mut page);
        page.push_str("</title>");
        if let Some(canonical) = &self.canonical {
            page.push_str("<link rel=\"canonical\" href=\"");
            escaped(canonical, &mut page);
            page.push_str("\">");
        }
        page.push_str("</head><body><article><h1>");
        escaped(&self.title, &mut page);
        page.push_str("</h1>");
        let byline: Vec<&str> = self
            .byline
            .iter()
            .map(String::as_str)
            .filter(|part| !part.is_empty())
            .collect();
        if !byline.is_empty() {
            page.push_str("<p>");
            escaped(&byline.join(" · "), &mut page);
            page.push_str("</p>");
        }
        page.push_str(&self.body);
        page.push_str("</article></body></html>");
        Reading {
            page: Some(page),
            metadata: self.metadata,
        }
    }
}

/// The text of an element with its whitespace collapsed.
fn text_of(element: ElementRef<'_>) -> String {
    element
        .text()
        .flat_map(str::split_whitespace)
        .collect::<Vec<_>>()
        .join(" ")
}

fn first<'a>(scope: ElementRef<'a>, query: &str) -> Option<ElementRef<'a>> {
    scope.select(&super::selector(query)).next()
}

/// The text of the first element the query finds, when it has any.
fn text_at(scope: ElementRef<'_>, query: &str) -> Option<String> {
    first(scope, query)
        .map(text_of)
        .filter(|text| !text.is_empty())
}

/// A time as a page shows it: `2018-07-11 23:46`, or only the day for midnight.
fn show_time(iso: &str) -> String {
    let (day, time) = iso.split_once('T').unwrap_or((iso, ""));
    let time = time.get(..5).unwrap_or_default();
    if time.is_empty() || time == "00:00" {
        day.to_owned()
    } else {
        format!("{day} {time}")
    }
}

/// The first non-empty `content` of the `meta` with this `property` or `name`,
/// its whitespace (a non-breaking space between two names) collapsed.
fn meta_content(document: &Html, key: &str) -> Option<String> {
    document
        .select(&super::selector("meta"))
        .filter(|meta| {
            ["property", "name"].iter().any(|attribute| {
                meta.value()
                    .attribute(attribute)
                    .is_some_and(|name| name.trim().eq_ignore_ascii_case(key))
            })
        })
        .filter_map(|meta| meta.value().attribute("content"))
        .map(|content| content.split_whitespace().collect::<Vec<_>>().join(" "))
        .find(|content| !content.is_empty())
}

// ---- dates -------------------------------------------------------------

/// A Unix time in seconds as an ISO 8601 time in China Standard Time, the
/// zone these sites show their dates in.
fn iso_from_seconds(seconds: i64) -> Option<String> {
    let zone = chrono::FixedOffset::east_opt(8 * 3600)?;
    chrono::DateTime::from_timestamp(seconds, 0).map(|time| {
        time.with_timezone(&zone)
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, false)
    })
}

/// A Unix time written in seconds or in milliseconds.
fn iso_from_unix(value: i64) -> Option<String> {
    iso_from_seconds(if value > 100_000_000_000 {
        value / 1000
    } else {
        value
    })
}

/// A date a page shows (`2018年7月11日 23:46`, `2020-03-27 17:38`,
/// `2022-04-14`) as ISO 8601 in China Standard Time.
fn iso_from_text(text: &str) -> Option<String> {
    static DATE: OnceLock<regex::Regex> = OnceLock::new();
    let date = DATE.get_or_init(|| {
        regex::Regex::new(
            r"(\d{4})\D{1,2}(\d{1,2})\D{1,2}(\d{1,2})\D{0,2}(?:(\d{1,2}):(\d{2})(?::(\d{2}))?)?",
        )
        .expect("static pattern")
    });
    let found = date.captures(text)?;
    let number = |index: usize| found.get(index)?.as_str().parse::<u32>().ok();
    let (year, month, day) = (number(1)?, number(2)?, number(3)?);
    if !(1990..=2100).contains(&year) || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some(format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}+08:00",
        number(4).unwrap_or(0),
        number(5).unwrap_or(0),
        number(6).unwrap_or(0)
    ))
}

// ---- markup ------------------------------------------------------------

/// How a site writes the parts the generic reader cannot read as they are.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dialect {
    Zhihu,
    WeChat,
    Jianshu,
    Cnblogs,
    Bilibili,
    /// Markup with nothing site-specific beyond what every dialect does.
    Plain,
}

const VOID: [&str; 14] = [
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
    "track", "wbr",
];

/// Elements that are never article text.
const DROPPED: &[&str] = &[
    "script", "style", "template", "iframe", "object", "embed", "form", "button", "input",
    "select", "textarea", "svg", "canvas", "audio", "video", "link", "meta", "head", "title",
    "nav", "dialog", "map", "source",
];

fn hidden(element: ElementRef<'_>) -> bool {
    let value = element.value();
    if value.attribute("hidden").is_some() || value.attribute("aria-hidden") == Some("true") {
        return true;
    }
    value.attribute("style").is_some_and(|style| {
        let compact: String = style
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect::<String>()
            .to_ascii_lowercase();
        compact.contains("display:none") || compact.contains("visibility:hidden")
    })
}

fn has_class(element: ElementRef<'_>, class: &str) -> bool {
    element.value().class_names().any(|name| name == class)
}

/// The address an image shows: the lazy-loading attributes sites keep the real
/// address in, before a placeholder `src`.
fn image_source<'a>(element: ElementRef<'a>) -> Option<&'a str> {
    [
        "data-original",
        "data-actualsrc",
        "data-src",
        "data-original-src",
        "data-lazy-src",
        "src",
    ]
    .iter()
    .filter_map(|name| element.value().attribute(name))
    .map(str::trim)
    .find(|value| !value.is_empty() && !value.to_ascii_lowercase().starts_with("data:"))
}

fn alt_text(element: ElementRef<'_>) -> String {
    element
        .value()
        .attribute("alt")
        .or_else(|| element.value().attribute("data-alt"))
        .unwrap_or_default()
        .trim()
        .to_owned()
}

/// The TeX of a formula Zhihu serves as an image or a marked span.
fn formula(element: ElementRef<'_>) -> Option<String> {
    let value = element.value();
    let marked = value.attribute("eeimg").is_some()
        || value
            .attribute("src")
            .is_some_and(|src| src.contains("/equation?tex="))
        || has_class(element, "ztext-math");
    if !marked {
        return None;
    }
    let tex = value
        .attribute("data-tex")
        .map(str::to_owned)
        .or_else(|| (!alt_text(element).is_empty()).then(|| alt_text(element)))
        .or_else(|| {
            let src = value.attribute("src")?;
            Url::parse(src)
                .or_else(|_| Url::parse(&format!("https:{src}")))
                .ok()?
                .query_pairs()
                .find(|(key, _)| key == "tex")
                .map(|(_, tex)| tex.into_owned())
        })?;
    let tex = tex.trim().trim_end_matches("\\\\").trim().to_owned();
    (!tex.is_empty() && !tex.contains('$')).then_some(tex)
}

/// The code of a `pre`, one line per `br`, block or per-line `code` child.
fn code_text(pre: ElementRef<'_>) -> String {
    fn walk(element: ElementRef<'_>, out: &mut String) {
        for child in element.children() {
            match child.value() {
                Node::Text(text) => out.push_str(text),
                Node::Element(value) => {
                    let Some(child) = ElementRef::wrap(child) else {
                        continue;
                    };
                    match value.name() {
                        "br" => out.push('\n'),
                        // A copy button or line-number gutter is not code.
                        _ if is_gutter(child) => {}
                        "script" | "style" | "button" => {}
                        _ => walk(child, out),
                    }
                }
                _ => {}
            }
        }
    }
    let lines: Vec<ElementRef<'_>> = pre
        .children()
        .filter_map(ElementRef::wrap)
        .filter(|child| child.value().name() == "code")
        .collect();
    let mut out = String::new();
    // WeChat writes each line of a block as a `code` of its own.
    if lines.len() > 1 {
        for (index, line) in lines.iter().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            walk(*line, &mut out);
        }
    } else {
        walk(pre, &mut out);
    }
    out.trim_end_matches(['\n', '\r']).to_owned()
}

fn is_gutter(element: ElementRef<'_>) -> bool {
    element.value().class_names().any(|class| {
        class.contains("line-index")
            || class.contains("line-numbers")
            || class == "copy-code-btn"
            || class == "hljs-button"
            || class == "code-block-extension-header"
    })
}

fn code_language(pre: ElementRef<'_>) -> Option<String> {
    let mut candidates = vec![pre];
    candidates.extend(
        pre.children()
            .filter_map(ElementRef::wrap)
            .filter(|child| child.value().name() == "code"),
    );
    candidates.iter().find_map(|element| {
        let value = element.value();
        for name in ["data-lang", "lang", "data-language"] {
            if let Some(lang) = value
                .attribute(name)
                .map(str::trim)
                .filter(|l| !l.is_empty())
            {
                return Some(lang.to_ascii_lowercase());
            }
        }
        value.class_names().find_map(|class| {
            class
                .strip_prefix("language-")
                .or_else(|| class.strip_prefix("lang-"))
                .filter(|lang| !lang.is_empty())
                .map(str::to_ascii_lowercase)
        })
    })
}

struct Cleaner {
    dialect: Dialect,
    out: String,
}

impl Cleaner {
    /// Text between tags. A non-breaking space, which these sites pad lines
    /// and headings with, is a space.
    fn text(&mut self, text: &str) {
        if text.contains('\u{a0}') {
            escaped(&text.replace('\u{a0}', " "), &mut self.out);
        } else {
            escaped(text, &mut self.out);
        }
    }

    fn children(&mut self, element: ElementRef<'_>) {
        for child in element.children() {
            match child.value() {
                Node::Text(text) => self.text(text),
                Node::Element(_) => {
                    if let Some(child) = ElementRef::wrap(child) {
                        self.element(child);
                    }
                }
                _ => {}
            }
        }
    }

    /// A `noscript` that repeats an image its siblings show.
    fn redundant_noscript(element: ElementRef<'_>) -> bool {
        element
            .parent()
            .and_then(ElementRef::wrap)
            .is_some_and(|parent| {
                parent
                    .children()
                    .filter_map(ElementRef::wrap)
                    .any(|sibling| sibling.value().name() == "img")
            })
    }

    fn dropped(&self, element: ElementRef<'_>) -> bool {
        let name = element.value().name();
        if DROPPED.contains(&name) || hidden(element) {
            return true;
        }
        if name == "noscript" {
            return Self::redundant_noscript(element);
        }
        match self.dialect {
            // A code block's line-number gutter. (The `js_img_placeholder`
            // class is on the article's own images, which are not dropped.)
            Dialect::WeChat => has_class(element, "code-snippet__line-index"),
            Dialect::Jianshu => has_class(element, "image-container-fill"),
            // The outlining icons of a collapsed code listing.
            Dialect::Cnblogs => {
                has_class(element, "code_img_closed") || has_class(element, "code_img_opened")
            }
            Dialect::Zhihu | Dialect::Bilibili | Dialect::Plain => false,
        }
    }

    fn element(&mut self, element: ElementRef<'_>) {
        let name = element.value().name();
        if self.dropped(element) {
            return;
        }
        match name {
            "img" => return self.image(element, None),
            "pre" => return self.code(element),
            _ => {}
        }
        if self.dialect == Dialect::Zhihu
            && let Some(tex) = formula(element)
        {
            return self.formula(element, &tex);
        }
        if self.dialect == Dialect::Jianshu && name == "div" && has_class(element, "image-package")
        {
            return self.jianshu_image(element);
        }
        // A custom element (WeChat's cards and players) has no markup to keep;
        // a `picture` is its image.
        if name.contains('-') || name == "noscript" || name == "picture" {
            return self.children(element);
        }
        if name == "a" {
            match element.value().attribute("href") {
                Some(href) => {
                    self.out.push_str("<a href=\"");
                    escaped(href.trim(), &mut self.out);
                    self.out.push_str("\">");
                    self.children(element);
                    self.out.push_str("</a>");
                }
                None => self.children(element),
            }
            return;
        }
        self.out.push('<');
        self.out.push_str(name);
        for attribute in ["colspan", "rowspan", "start"] {
            if let Some(value) = element.value().attribute(attribute) {
                self.out.push(' ');
                self.out.push_str(attribute);
                self.out.push_str("=\"");
                escaped(value, &mut self.out);
                self.out.push('"');
            }
        }
        self.out.push('>');
        if VOID.contains(&name) {
            return;
        }
        self.children(element);
        self.out.push_str("</");
        self.out.push_str(name);
        self.out.push('>');
    }

    fn image(&mut self, element: ElementRef<'_>, alt: Option<&str>) {
        if let Some(tex) = (self.dialect == Dialect::Zhihu)
            .then(|| formula(element))
            .flatten()
        {
            return self.formula(element, &tex);
        }
        let Some(src) = image_source(element) else {
            return;
        };
        // WeChat numbers its images with a fragment, which is not part of the address.
        // Bilibili's image service resizes by a directive after `@`; the address
        // without it is the image as uploaded.
        let src = match self.dialect {
            Dialect::WeChat => src
                .split_once("#imgIndex=")
                .map_or(src, |(address, _)| address),
            Dialect::Bilibili => src.split_once('@').map_or(src, |(address, _)| address),
            _ => src,
        };
        self.out.push_str("<img src=\"");
        escaped(src, &mut self.out);
        self.out.push_str("\" alt=\"");
        escaped(alt.unwrap_or(&alt_text(element)), &mut self.out);
        self.out.push_str("\">");
    }

    /// A formula as text between dollar signs; alone in its paragraph it is a
    /// display formula.
    fn formula(&mut self, element: ElementRef<'_>, tex: &str) {
        let alone = element
            .parent()
            .and_then(ElementRef::wrap)
            .is_some_and(|parent| {
                parent.value().name() == "p"
                    && parent.children().all(|child| match child.value() {
                        Node::Text(text) => text.trim().is_empty(),
                        Node::Element(_) => child.id() == element.id(),
                        _ => true,
                    })
            });
        let delimiter = if alone { "$$" } else { "$" };
        self.out.push_str(delimiter);
        self.text(tex);
        self.out.push_str(delimiter);
    }

    fn code(&mut self, pre: ElementRef<'_>) {
        let code = code_text(pre);
        if code.trim().is_empty() {
            return;
        }
        self.out.push_str("<pre><code");
        if let Some(language) = code_language(pre) {
            self.out.push_str(" class=\"language-");
            escaped(&language, &mut self.out);
            self.out.push('"');
        }
        self.out.push('>');
        escaped(&code, &mut self.out);
        self.out.push_str("</code></pre>");
    }

    /// Jianshu's `image-package`: the image and, unless it is the editor's
    /// default, its caption.
    fn jianshu_image(&mut self, package: ElementRef<'_>) {
        let Some(image) = first(package, "img") else {
            return;
        };
        let caption = text_at(package, ".image-caption")
            .filter(|caption| !caption.eq_ignore_ascii_case("image") && caption != "图片");
        self.out.push_str("<figure>");
        self.image(image, caption.as_deref());
        if let Some(caption) = caption {
            self.out.push_str("<figcaption>");
            self.text(&caption);
            self.out.push_str("</figcaption>");
        }
        self.out.push_str("</figure>");
    }
}

/// The markup of an element's content, cleaned the way the site needs.
fn clean_children(element: ElementRef<'_>, dialect: Dialect) -> String {
    let mut cleaner = Cleaner {
        dialect,
        out: String::new(),
    };
    cleaner.children(element);
    cleaner.out
}

/// The markup of an HTML fragment (JSON a site embeds for its own rendering),
/// cleaned the way the site needs.
fn clean_fragment(fragment: &str, dialect: Dialect) -> String {
    let parsed = Html::parse_fragment(fragment);
    clean_children(parsed.root_element(), dialect)
}

/// Whether there is text or an image to read.
fn substantial(markup: &str) -> bool {
    markup.contains("<img") || markup.contains("<pre") || {
        let fragment = Html::parse_fragment(markup);
        fragment
            .root_element()
            .text()
            .any(|text| !text.trim().is_empty())
    }
}

// ---- links the site's own redirect page wraps ---------------------------

/// The page an address on a site's own redirect page (`link.zhihu.com/?target=`)
/// names: the target address, when it is an http(s) page.
pub(super) fn redirect_target(url: &Url) -> Option<String> {
    let host = url.host_str()?.to_ascii_lowercase();
    let path = url.path().trim_end_matches('/');
    let parameter = match (host.as_str(), path) {
        ("link.zhihu.com", "") => "target",
        ("link.juejin.cn", "") => "target",
        ("link.csdn.net", "") => "target",
        ("links.jianshu.com", "/go") => "to",
        ("www.jianshu.com", "/go-wild") => "url",
        ("www.douban.com", "/link2") => "url",
        ("sspai.com" | "www.sspai.com", "/link") => "target",
        ("gitee.com", "/link") => "target",
        ("www.oschina.net", "/action/GoToLink") => "url",
        ("www.infoq.cn" | "xie.infoq.cn", "/link") => "target",
        _ => return None,
    };
    let target = url
        .query_pairs()
        .find(|(key, _)| key == parameter)
        .map(|(_, value)| value.into_owned())?;
    let target = web_address(&target)?;
    // A redirect page that points at another redirect page is not followed.
    site_redirect_host(target.host_str()?).then(|| target.to_string())
}

/// Whether an address may be the target of a redirect: any host but another
/// redirect page of the same kind.
fn site_redirect_host(host: &str) -> bool {
    !matches!(
        host,
        "link.zhihu.com" | "link.juejin.cn" | "link.csdn.net" | "links.jianshu.com"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosts_name_their_sites_and_only_them() {
        for (host, site) in [
            ("www.zhihu.com", Some(Site::Zhihu)),
            ("zhuanlan.zhihu.com", Some(Site::Zhihu)),
            ("zhihu.com", Some(Site::Zhihu)),
            ("mp.weixin.qq.com", Some(Site::WeChat)),
            ("www.cnblogs.com", Some(Site::Cnblogs)),
            ("www.jianshu.com", Some(Site::Jianshu)),
            ("36kr.com", Some(Site::Kr36)),
            ("www.bilibili.com", Some(Site::Bilibili)),
            ("t.bilibili.com", Some(Site::Bilibili)),
            ("bilibili.com.example.test", None),
            ("my.oschina.net", Some(Site::Oschina)),
            ("notzhihu.com", None),
            ("zhihu.com.example.test", None),
            ("weixin.qq.com", None),
            ("example.test", None),
        ] {
            assert_eq!(site_of(host), site, "{host}");
        }
    }

    #[test]
    fn saved_pages_name_their_address_in_a_comment_or_the_head() {
        let comment =
            saved_from(" saved from url=(0043)https://www.zhihu.com/question/1/answer/2 ");
        assert_eq!(
            comment.map(String::from).as_deref(),
            Some("https://www.zhihu.com/question/1/answer/2")
        );
        assert!(saved_from(" saved from url=(0014)about:blank ").is_none());
        let document = Html::parse_document(
            "<!-- saved from url=(0030)https://mp.weixin.qq.com/s/abc --><html><head></head><body></body></html>",
        );
        let (site, address) = recognize(&document, None).unwrap();
        assert_eq!(site, Site::WeChat);
        assert_eq!(address.unwrap().as_str(), "https://mp.weixin.qq.com/s/abc");
        let document = Html::parse_document(
            r#"<html><head><link rel="canonical" href="https://www.cnblogs.com/u/p/1.html"></head><body></body></html>"#,
        );
        assert_eq!(recognize(&document, None).unwrap().0, Site::Cnblogs);
        let document = Html::parse_document(
            r#"<html><head><meta property="og:url" content="https://www.jianshu.com/p/abc"></head><body></body></html>"#,
        );
        assert_eq!(recognize(&document, None).unwrap().0, Site::Jianshu);
        // Another site's canonical link is not this site, even with its markers.
        let document = Html::parse_document(
            r#"<html><head><link rel="canonical" href="https://example.test/a"></head><body><div id="js_content"><p>x</p></div></body></html>"#,
        );
        assert!(recognize(&document, None).is_none());
        // Markers identify a page that names no address at all.
        let document = Html::parse_document(
            r#"<html><body><div id="js_content"><p>x</p></div></body></html>"#,
        );
        assert_eq!(recognize(&document, None).unwrap().0, Site::WeChat);
    }

    #[test]
    fn dates_are_read_in_china_standard_time() {
        assert_eq!(
            iso_from_text("2018年7月11日 23:46").as_deref(),
            Some("2018-07-11T23:46:00+08:00")
        );
        assert_eq!(
            iso_from_text("发布于 2022-04-14").as_deref(),
            Some("2022-04-14T00:00:00+08:00")
        );
        assert_eq!(
            iso_from_text("2020-03-27 17:38:09").as_deref(),
            Some("2020-03-27T17:38:09+08:00")
        );
        assert!(iso_from_text("yesterday").is_none());
        assert!(iso_from_text("9999-99-99").is_none());
        assert_eq!(
            iso_from_unix(1_531_324_000).as_deref(),
            Some("2018-07-11T23:46:40+08:00")
        );
        assert_eq!(
            iso_from_unix(1_531_324_000_000).as_deref(),
            Some("2018-07-11T23:46:40+08:00")
        );
    }

    #[test]
    fn redirect_pages_give_their_target() {
        for (wrapped, target) in [
            (
                "https://link.zhihu.com/?target=https%3A//example.test/a%3Fb%3D1",
                "https://example.test/a?b=1",
            ),
            (
                "https://link.juejin.cn?target=https%3A%2F%2Fexample.test%2Fx",
                "https://example.test/x",
            ),
            (
                "https://links.jianshu.com/go?to=https%3A%2F%2Fexample.test%2F",
                "https://example.test/",
            ),
            (
                "https://www.douban.com/link2/?url=http%3A%2F%2Fexample.test%2Fp",
                "http://example.test/p",
            ),
        ] {
            let url = Url::parse(wrapped).unwrap();
            assert_eq!(redirect_target(&url).as_deref(), Some(target), "{wrapped}");
        }
        for kept in [
            "https://link.zhihu.com/?target=javascript%3Aalert(1)",
            "https://link.zhihu.com/?target=%2Frelative",
            "https://link.zhihu.com/?other=https%3A%2F%2Fexample.test",
            "https://link.zhihu.com/?target=https%3A%2F%2Flink.zhihu.com%2F%3Ftarget%3Dhttps%3A%2F%2Fa.test",
            "https://www.zhihu.com/?target=https%3A%2F%2Fexample.test",
            "https://example.test/?target=https%3A%2F%2Fother.test",
        ] {
            assert!(
                redirect_target(&Url::parse(kept).unwrap()).is_none(),
                "{kept}"
            );
        }
    }

    #[test]
    fn the_cleaner_keeps_text_code_and_images_and_drops_chrome() {
        let fragment = r#"<p>Text &amp; <b>bold</b> <a href="https://a.test/?x=1&amp;y=2" class="c">link</a></p>
            <script>bad()</script><div style="display: none">hidden</div>
            <pre lang="python"><code>if a &lt; b:
    pass</code></pre>
            <figure><noscript><img src="https://i.test/small.jpg"></noscript>
            <img src="data:image/svg+xml;utf8,&lt;svg/&gt;" data-original="https://i.test/big.jpg" alt="A picture"></figure>
            <table><tr><td colspan="2">cell</td></tr></table>
            <mp-common-profile><span>card text</span></mp-common-profile>"#;
        let markup = clean_fragment(fragment, Dialect::Zhihu);
        assert!(markup.contains("Text &amp; <b>bold</b>"), "{markup}");
        assert!(
            markup.contains(r#"<a href="https://a.test/?x=1&amp;y=2">link</a>"#),
            "{markup}"
        );
        assert!(
            !markup.contains("bad()") && !markup.contains("hidden"),
            "{markup}"
        );
        assert!(
            markup.contains(
                "<pre><code class=\"language-python\">if a &lt; b:\n    pass</code></pre>"
            ),
            "{markup}"
        );
        assert!(markup.contains(r#"<img src="https://i.test/big.jpg" alt="A picture">"#));
        assert!(!markup.contains("small.jpg"), "{markup}");
        assert!(markup.contains(r#"<td colspan="2">cell</td>"#));
        assert!(markup.contains("card text") && !markup.contains("mp-common"));
        assert!(substantial(&markup));
        assert!(!substantial("<p> </p>"));
    }

    #[test]
    fn non_breaking_spaces_become_spaces_outside_code() {
        let markup = clean_fragment(
            "<p>a&nbsp;&nbsp;b</p><pre><code>x&nbsp;y</code></pre>",
            Dialect::Bilibili,
        );
        assert!(markup.contains("<p>a  b</p>"), "{markup:?}");
        assert!(markup.contains("x\u{a0}y"), "{markup:?}");
    }

    #[test]
    fn formulas_become_dollar_delimited_text() {
        let markup = clean_fragment(
            r#"<p>Energy <img src="https://www.zhihu.com/equation?tex=E%3Dmc%5E2" alt="E=mc^2" eeimg="1"> is mass.</p>
               <p><img src="https://www.zhihu.com/equation?tex=x%5E2%5C%5C" alt="x^2\\" eeimg="1"></p>"#,
            Dialect::Zhihu,
        );
        assert!(markup.contains("Energy $E=mc^2$ is mass."), "{markup}");
        assert!(markup.contains("<p>$$x^2$$</p>"), "{markup}");
    }

    #[test]
    fn wechat_code_blocks_join_their_line_elements() {
        let markup = clean_fragment(
            r#"<section class="code-snippet__fix"><ul class="code-snippet__line-index"><li></li><li></li></ul>
               <pre class="code-snippet__js" data-lang="javascript"><code><span leaf="">const a = 1;</span></code><code><span leaf="">log(a);</span></code></pre></section>"#,
            Dialect::WeChat,
        );
        assert!(
            markup.contains(
                "<pre><code class=\"language-javascript\">const a = 1;\nlog(a);</code></pre>"
            ),
            "{markup}"
        );
        assert!(!markup.contains("<li>"), "{markup}");
    }
}
