mod article;
mod callouts;
mod charset;
mod code;
mod facts;
mod furniture;
mod hacker_news;
mod mail;
mod raw_math;
mod sites;
mod social;
mod srcset;
mod stream;

pub(crate) use social::canonical_status_url;

use crate::{Document, Error, Result};
use facts::{Fact, Facts};
use scraper::{ElementRef, Html, Selector};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use url::Url;

fn selector(query: &str) -> Selector {
    Selector::parse(query).expect("static selector")
}

/// The elements inside `element` in document order, as `ElementRef::select`
/// walks them (the element itself is not one of them).
fn inside<'a>(element: ElementRef<'a>) -> impl Iterator<Item = ElementRef<'a>> {
    element.descendants().skip(1).filter_map(ElementRef::wrap)
}

/// `a[href]`: a link.
fn link(element: ElementRef<'_>) -> bool {
    element.value().name() == "a" && element.value().attribute("href").is_some()
}

/// `[id], a[name]`: an element a fragment can name.
fn fragment_target(element: ElementRef<'_>) -> bool {
    let value = element.value();
    value.attribute("id").is_some() || (value.name() == "a" && value.attribute("name").is_some())
}

/// `a[href], sup, span[data-definition], label.footref`: what can mark a
/// note's reference.
fn reference_markup(element: ElementRef<'_>) -> bool {
    let value = element.value();
    link(element)
        || match value.name() {
            "sup" => true,
            "span" => value.attribute("data-definition").is_some(),
            "label" => value.class_names().any(|class| class == "footref"),
            _ => false,
        }
}

/// A `span` of one of these classes (`span.a, span.b`).
fn span_of(element: ElementRef<'_>, classes: &[&str]) -> bool {
    element.value().name() == "span"
        && element
            .value()
            .class_names()
            .any(|class| classes.contains(&class))
}

/// Whether a link's `rel` names the page's canonical address.
fn canonical_rel(rel: &str) -> bool {
    rel.split_whitespace()
        .any(|token| token.eq_ignore_ascii_case("canonical"))
}

/// An element's attribute by name, compared as text. scraper's `Element::attr`
/// gives the same answer (the attribute with this local name and no namespace)
/// but interns the name on every call, which for a name outside html5ever's
/// static set takes a global lock and an allocation; the cleaner asks for
/// attributes of every element several times over.
trait Attribute {
    fn attribute(&self, name: &str) -> Option<&str>;

    /// The class names, split at ASCII whitespace as scraper's
    /// `Element::classes` splits them, in the attribute's order and without
    /// interning each one: `classes` interns them the first time it is asked
    /// about an element, a global lock and an allocation for every name
    /// outside html5ever's static set.
    fn class_names(&self) -> std::str::SplitAsciiWhitespace<'_> {
        self.attribute("class")
            .unwrap_or_default()
            .split_ascii_whitespace()
    }
}

impl Attribute for scraper::node::Element {
    fn attribute(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(key, _)| &*key.local == name && key.prefix.is_none() && key.ns.is_empty())
            .map(|(_, value)| &**value)
    }
}

fn plain(element: ElementRef<'_>) -> String {
    element
        .text()
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// What the page's metadata and the site readers' tests look for on every
/// page, read in one walk over its nodes in the order `Html::select` reads
/// them (the order the parser made them in, nodes taken out of the tree
/// included). Each field stands for the selector in its comment.
struct Landmarks<'a> {
    /// `meta`
    metas: Vec<ElementRef<'a>>,
    /// `script`
    scripts: Vec<ElementRef<'a>>,
    /// The first `title` and the first `h1`.
    title: Option<ElementRef<'a>>,
    heading: Option<ElementRef<'a>>,
    /// The first `link[rel]` whose `rel` names the canonical address.
    canonical: Option<ElementRef<'a>>,
    /// `time[datetime]`
    times: Vec<ElementRef<'a>>,
    /// `[data-partnereventstore]`: Steam's event data.
    events: Vec<ElementRef<'a>>,
    /// Substack: the first `[class*="feedPermalinkUnit"]`, every
    /// `link[href], script[src]`, and whether there is a `div.body.markup`.
    permalink: Option<ElementRef<'a>>,
    assets: Vec<ElementRef<'a>>,
    rendered_body: bool,
    /// X: whether there is a `[data-testid="primaryColumn"]
    /// article[data-testid="tweet"]` or a `main article
    /// [data-engagement-action]` (the 2026 page has no test ids), and every
    /// `img[src], video[poster]`.
    column_post: bool,
    media: Vec<ElementRef<'a>>,
    /// Hacker News: whether there are a `#hnmain` and a `tr.athing, tr.comtr`.
    hn_main: bool,
    hn_rows: bool,
}

impl<'a> Landmarks<'a> {
    fn read(document: &'a Html) -> Self {
        let mut found = Self {
            metas: Vec::new(),
            scripts: Vec::new(),
            title: None,
            heading: None,
            canonical: None,
            times: Vec::new(),
            events: Vec::new(),
            permalink: None,
            assets: Vec::new(),
            rendered_body: false,
            column_post: false,
            media: Vec::new(),
            hn_main: false,
            hn_rows: false,
        };
        let classes = |element: ElementRef<'_>, wanted: &[&str]| {
            wanted
                .iter()
                .all(|class| element.value().class_names().any(|name| name == *class))
        };
        // `Html::select` leaves out an element without a parent (taken out of
        // the tree), but not the elements inside it.
        let elements = document.tree.nodes().filter_map(ElementRef::wrap);
        for element in elements.filter(|element| element.parent().is_some()) {
            let value = element.value();
            let has = |name| value.attribute(name).is_some();
            match value.name() {
                "meta" => found.metas.push(element),
                "script" => {
                    found.scripts.push(element);
                    if has("src") {
                        found.assets.push(element);
                    }
                }
                "link" => {
                    if has("href") {
                        found.assets.push(element);
                    }
                    if found.canonical.is_none()
                        && value.attribute("rel").is_some_and(canonical_rel)
                    {
                        found.canonical = Some(element);
                    }
                }
                "title" if found.title.is_none() => found.title = Some(element),
                "h1" if found.heading.is_none() => found.heading = Some(element),
                "time" if has("datetime") => found.times.push(element),
                "img" if has("src") => found.media.push(element),
                "video" if has("poster") => found.media.push(element),
                "div" if classes(element, &["body", "markup"]) => found.rendered_body = true,
                "tr" if classes(element, &["athing"]) || classes(element, &["comtr"]) => {
                    found.hn_rows = true;
                }
                // A selector's descendant reads element parents only.
                "article"
                    if value.attribute("data-testid") == Some("tweet")
                        && std::iter::successors(element.parent(), |node| node.parent())
                            .map_while(ElementRef::wrap)
                            .any(|parent| {
                                parent.value().attribute("data-testid") == Some("primaryColumn")
                            }) =>
                {
                    found.column_post = true;
                }
                _ => {}
            }
            if has("data-partnereventstore") {
                found.events.push(element);
            }
            if !found.column_post && has("data-engagement-action") {
                found.column_post = social::is_engagement_in_post(element);
            }
            if found.permalink.is_none()
                && value
                    .attribute("class")
                    .is_some_and(|class| class.contains("feedPermalinkUnit"))
            {
                found.permalink = Some(element);
            }
            found.hn_main |= value.attribute("id") == Some("hnmain");
        }
        found
    }
}

/// The first non-empty `content` of a `meta` element whose `property` or `name`
/// is one of `keys`, tried in order (a page that writes both attributes, or
/// an Open Graph value under `name`, is read either way).
fn meta(metas: &[ElementRef<'_>], keys: &[&str]) -> Option<String> {
    for key in keys {
        for element in metas {
            let names = |attribute: &str| {
                element
                    .value()
                    .attribute(attribute)
                    .is_some_and(|name| name.trim().eq_ignore_ascii_case(key))
            };
            if (names("property") || names("name"))
                && let Some(value) = element
                    .value()
                    .attribute("content")
                    .map(str::trim)
                    .filter(|v| !v.is_empty())
            {
                return Some(value.to_owned());
            }
        }
    }
    None
}

fn jsonld_documents(scripts: &[ElementRef<'_>]) -> Vec<Value> {
    fn collect(value: Value, documents: &mut Vec<Value>, depth: usize) {
        if depth > 64 {
            return;
        }
        match value {
            Value::Object(mut object) => {
                let graph = object.remove("@graph");
                documents.push(Value::Object(object));
                if let Some(graph) = graph {
                    collect(graph, documents, depth + 1);
                }
            }
            Value::Array(values) => {
                for value in values {
                    collect(value, documents, depth + 1);
                }
            }
            _ => {}
        }
    }
    let mut documents = Vec::new();
    for script in scripts {
        if script
            .value()
            .attribute("type")
            .is_some_and(|kind| kind.trim().eq_ignore_ascii_case("application/ld+json"))
            && let Ok(value) = serde_json::from_str(&script.text().collect::<String>())
        {
            collect(value, &mut documents, 0);
        }
    }
    documents
}

fn jsonld_text(documents: &[Value], key: &str) -> Option<String> {
    documents.iter().find_map(|document| {
        document
            .get(key)?
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
    })
}

/// Whether a metadata value is a web address, which names a page, not a
/// person (`article:author` often links the author's profile).
fn web_address(value: &str) -> bool {
    let value = value.trim();
    value.starts_with("//") || (value.contains("://") && !value.contains(char::is_whitespace))
}

/// The names in a JSON-LD `author` value: a string, a `Person` or other
/// object with a `name`, an object that points (`@id`) at another object of
/// the page, or an array of them. An array that has persons keeps them only
/// (an organization beside them is the publisher). Web addresses are no names.
fn jsonld_author_names(
    value: &Value,
    documents: &[Value],
    follow: bool,
    depth: usize,
) -> Vec<String> {
    if depth > 4 {
        return Vec::new();
    }
    let usable = |name: &str| {
        let name = name.split_whitespace().collect::<Vec<_>>().join(" ");
        (!name.is_empty() && !web_address(&name)).then_some(name)
    };
    let person = |value: &Value| {
        let kind = value.get("@type");
        kind.and_then(Value::as_str) == Some("Person")
            || kind
                .and_then(Value::as_array)
                .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("Person")))
    };
    match value {
        Value::String(name) => usable(name).into_iter().collect(),
        Value::Object(object) => {
            if let Some(name) = object.get("name").and_then(Value::as_str) {
                return usable(name).into_iter().collect();
            }
            // A reference to an object elsewhere in the page's graph.
            let target = object
                .get("@id")
                .and_then(Value::as_str)
                .filter(|_| follow)
                .and_then(|id| {
                    documents
                        .iter()
                        .find(|document| document.get("@id").and_then(Value::as_str) == Some(id))
                });
            target.map_or_else(Vec::new, |target| {
                jsonld_author_names(target, documents, false, depth + 1)
            })
        }
        Value::Array(values) => {
            let persons = values.iter().any(person);
            let mut names = Vec::new();
            for value in values.iter().filter(|value| !persons || person(value)) {
                for name in jsonld_author_names(value, documents, follow, depth + 1) {
                    if !names.contains(&name) {
                        names.push(name);
                    }
                }
            }
            names
        }
        _ => Vec::new(),
    }
}

/// The JSON-LD `publisher`'s name (a string or an object's `name`): a site's
/// own name, when its meta tags name none.
fn jsonld_publisher(documents: &[Value]) -> Option<String> {
    documents.iter().find_map(|document| {
        let publisher = document.get("publisher")?;
        let name = publisher
            .as_str()
            .or_else(|| publisher.get("name")?.as_str())?
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        (!name.is_empty() && !web_address(&name)).then_some(name)
    })
}

/// The page's author: a `author` meta tag, else a JSON-LD author, else an
/// `article:author` meta tag, whichever is a name: a web address (a profile
/// link) is never written as the author.
fn page_author(metas: &[ElementRef<'_>], jsonld: &[Value]) -> Option<String> {
    let name = |value: String| (!web_address(&value)).then_some(value);
    meta(metas, &["author"])
        .and_then(name)
        .or_else(|| {
            jsonld.iter().find_map(|document| {
                let names = jsonld_author_names(document.get("author")?, jsonld, true, 0);
                (!names.is_empty()).then(|| names.join(", "))
            })
        })
        .or_else(|| meta(metas, &["article:author"]).and_then(name))
}

fn clean_title(value: &str, site: Option<&str>) -> Option<String> {
    let mut title = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(site) = site {
        let site = site.split_whitespace().collect::<Vec<_>>().join(" ");
        loop {
            let mut changed = false;
            for separator in [" | ", " - ", " -- ", " · ", " — ", " – "] {
                if let Some(trimmed) = title.strip_suffix(&format!("{separator}{site}")) {
                    title = trimmed.to_owned();
                    changed = true;
                    break;
                }
                if let Some(trimmed) = title.strip_prefix(&format!("{site}{separator}")) {
                    title = trimmed.to_owned();
                    changed = true;
                    break;
                }
            }
            if !changed {
                break;
            }
        }
    }
    if let Some((end, _)) = title.char_indices().nth(300) {
        title.truncate(end);
        if let Some(boundary) = title.rfind(' ').filter(|index| *index > 0) {
            title.truncate(boundary);
        }
        title.push('…');
    }
    (!title.is_empty()).then_some(title)
}

/// A title's letters and digits in lower case: what two spellings of one title
/// share (`Rust (programming language) - Wikipedia`, `Rust (programming
/// language)`).
fn comparable(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Whether a JSON-LD `headline` is no title of this page: it names nothing the
/// first heading, the document title and the Open Graph title say, while the
/// heading and one of the titles agree with each other. Wikipedia's `headline`
/// is the article's short description (`memory-safe programming language
/// without garbage collection`).
fn headline_describes_instead(
    headline: &str,
    heading: Option<&str>,
    titles: [Option<&str>; 2],
) -> bool {
    let headline = comparable(headline);
    let heading = heading.map(comparable).unwrap_or_default();
    let titles: Vec<String> = titles
        .into_iter()
        .flatten()
        .map(comparable)
        .filter(|title| !title.is_empty())
        .collect();
    let related = |a: &str, b: &str| a.contains(b) || b.contains(a);
    !headline.is_empty()
        && !heading.is_empty()
        && titles.iter().any(|title| related(&heading, title))
        && !related(&headline, &heading)
        && titles.iter().all(|title| !related(&headline, title))
}

fn page_title(page: &Landmarks<'_>, jsonld: &[Value], site: Option<&str>) -> Option<String> {
    let document_title = page.title.map(plain);
    let social = meta(&page.metas, &["og:title", "twitter:title"]);
    let headline = jsonld_text(jsonld, "headline").filter(|headline| {
        !headline_describes_instead(
            headline,
            page.heading.map(plain).as_deref(),
            [document_title.as_deref(), social.as_deref()],
        )
    });
    let mut title = [
        headline.clone(),
        jsonld_text(jsonld, "name"),
        document_title.clone(),
        social.clone(),
    ]
    .into_iter()
    .flatten()
    .find_map(|value| clean_title(&value, site))?;
    if let Some(prefix) = title
        .strip_suffix('…')
        .or_else(|| title.strip_suffix("..."))
    {
        let prefix = prefix.trim_end();
        if let Some(candidate) = [document_title, social]
            .into_iter()
            .flatten()
            .filter_map(|value| clean_title(&value, site))
            .find(|value| {
                !value.ends_with('…')
                    && !value.ends_with("...")
                    && value.len() > title.len()
                    && value.starts_with(prefix)
            })
        {
            title = candidate;
        }
    }
    if title.contains(" | ") {
        let heading = page.heading.map(plain);
        for candidate in [headline, heading].into_iter().flatten() {
            let candidate = candidate.split_whitespace().collect::<Vec<_>>().join(" ");
            if !candidate.is_empty() && title.starts_with(&format!("{candidate} | ")) {
                return Some(candidate);
            }
        }
    }
    Some(title)
}

fn escaped(value: &str, output: &mut String) {
    for ch in value.chars() {
        match ch {
            '&' => output.push_str("&amp;"),
            '<' => output.push_str("&lt;"),
            '>' => output.push_str("&gt;"),
            '"' => output.push_str("&quot;"),
            _ => output.push(ch),
        }
    }
}

/// The element htmd turns into a soft line break.
const SOFT_BREAK: &str = "markitai-soft-break";

/// Where the cleaner writes the sanitized page for htmd: start tags with their
/// attributes, end tags and text, which stand for the markup `<name a="v">`,
/// `</name>` and escaped text. By default they build htmd's tree as they are
/// written, the tree htmd would parse from that markup; the markup itself is
/// written only for a page the tree writer cannot build (`htmd::TreeWriter`).
struct Out {
    tree: Option<htmd::TreeWriter>,
    markup: String,
    /// A start tag is open in `markup`; attributes may follow.
    open: bool,
    /// The last thing written ended a soft line break.
    soft_break: bool,
}

impl Out {
    fn tree() -> Self {
        Self {
            tree: Some(htmd::TreeWriter::new()),
            ..Self::markup()
        }
    }

    fn markup() -> Self {
        Self {
            tree: None,
            markup: String::new(),
            open: false,
            soft_break: false,
        }
    }

    /// `<name`, followed by the element's attributes.
    fn start(&mut self, name: &str) {
        self.soft_break = false;
        if let Some(tree) = &mut self.tree {
            tree.start(name);
        } else {
            self.close();
            self.markup.push('<');
            self.markup.push_str(name);
            self.open = true;
        }
    }

    /// ` name="value"`, an attribute of the element started last.
    fn attribute(&mut self, name: &str, value: &str) {
        if let Some(tree) = &mut self.tree {
            tree.attribute(name, value);
        } else {
            self.markup.push(' ');
            self.markup.push_str(name);
            self.markup.push_str("=\"");
            escaped(value, &mut self.markup);
            self.markup.push('"');
        }
    }

    /// `</name>`.
    fn end(&mut self, name: &str) {
        self.soft_break = name == SOFT_BREAK;
        if let Some(tree) = &mut self.tree {
            tree.end(name);
        } else {
            self.close();
            self.markup.push_str("</");
            self.markup.push_str(name);
            self.markup.push('>');
        }
    }

    fn text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.soft_break = false;
        if let Some(tree) = &mut self.tree {
            tree.text(text);
        } else {
            self.close();
            escaped(text, &mut self.markup);
        }
    }

    /// Ends the start tag open in `markup`.
    fn close(&mut self) {
        if std::mem::take(&mut self.open) {
            self.markup.push('>');
        }
    }

    /// The markup written, complete.
    fn into_markup(mut self) -> String {
        self.close();
        self.markup
    }
}

/// A link's destination: the address as the page gives it, resolved against
/// the page's own, and, for the redirect page a site wraps its outward links
/// in (`link.zhihu.com/?target=...`), the address that page leads to.
fn safe_url(value: &str, base: Option<&Url>) -> Option<String> {
    let destination = resolved_url(value, base)?;
    // Only an address whose query names a target can be such a wrapper.
    if ["target=", "to=", "url="]
        .iter()
        .any(|parameter| destination.contains(parameter))
        && let Ok(url) = Url::parse(&destination)
        && let Some(target) = sites::redirect_target(&url)
    {
        return Some(
            target
                .replace('<', "%3C")
                .replace('>', "%3E")
                .replace('"', "%22"),
        );
    }
    Some(destination)
}

fn resolved_url(value: &str, base: Option<&Url>) -> Option<String> {
    fn destination(value: &str) -> String {
        value
            .replace('<', "%3C")
            .replace('>', "%3E")
            .replace('"', "%22")
    }
    let value = value.trim();
    if value.is_empty() || value.chars().any(char::is_control) {
        return None;
    }
    if value.starts_with('#') {
        return Some(destination(value));
    }
    if let Ok(url) = Url::parse(value) {
        return matches!(url.scheme(), "http" | "https" | "mailto" | "tel")
            .then(|| destination(value));
    }
    // A scheme-relative address (`//host/path`) names another host; without a
    // page to take the scheme from, Markdown would read it as a local path.
    if base.is_none() && value.starts_with("//") {
        return Url::parse(&format!("https:{value}"))
            .ok()
            .filter(|url| url.host_str().is_some())
            .map(|_| destination(&format!("https:{value}")));
    }
    if let Some(base) = base {
        return base
            .join(value)
            .ok()
            .filter(|url| matches!(url.scheme(), "http" | "https"))
            .map(|url| url.to_string());
    }
    (!value.contains(':')).then(|| destination(value))
}

/// A saved page's own address for its relative links: an absolute
/// `<base href>` (which also names the directory), else a canonical link
/// (whose directory is known unless it is only the home page).
fn saved_page_base(document: ElementRef<'_>) -> Option<(Url, bool)> {
    let web = |value: &str| {
        Url::parse(value.trim())
            .ok()
            .filter(|url| matches!(url.scheme(), "http" | "https") && url.host_str().is_some())
    };
    // The first `<base href>` and the first canonical `<link href>`, in one
    // walk over the page.
    let (mut base, mut canonical) = (None, None);
    for element in inside(document) {
        let value = element.value();
        match value.name() {
            "base" if base.is_none() => base = value.attribute("href"),
            "link" if canonical.is_none() && value.attribute("rel").is_some_and(canonical_rel) => {
                canonical = value.attribute("href");
            }
            _ => {}
        }
        if base.is_some() && canonical.is_some() {
            break;
        }
    }
    if let Some(base) = base.and_then(web) {
        return Some((base, true));
    }
    canonical.and_then(web).map(|canonical| {
        let directory = !canonical.path().trim_matches('/').is_empty();
        (canonical, directory)
    })
}

/// A link of a page read without its URL: relative links resolve against the
/// saved page's own address when it has one (root-relative ones only, when
/// that is its home page); fragments stay in the page.
fn saved_page_link(value: &str, link_base: Option<&(Url, bool)>) -> Option<String> {
    let trimmed = value.trim();
    // An absolute address keeps its own spelling: `safe_url` reads it before
    // it would resolve against the base, so it is parsed once.
    let base = link_base
        .filter(|(_, directory)| {
            !trimmed.starts_with('#') && (*directory || trimmed.starts_with('/'))
        })
        .map(|(base, _)| base);
    safe_url(trimmed, base)
}

/// Inline image data is not an addressable resource. Like the reference, keep
/// a `data:<type>...` placeholder so alt text and position survive; active
/// document types are removed, and the payload itself is never retained.
fn image_source(value: &str, base: Option<&Url>) -> Option<String> {
    let value = value.trim();
    if !value
        .get(..5)
        .is_some_and(|scheme| scheme.eq_ignore_ascii_case("data:"))
    {
        return safe_url(value, base);
    }
    let lower = value.to_ascii_lowercase();
    if [
        "data:text/html",
        "data:image/svg+xml",
        "data:text/javascript",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
    {
        return None;
    }
    let meta = value.split(',').next().unwrap_or(value);
    (meta.len() <= 256
        && !meta.chars().any(|ch| {
            ch.is_control()
                || ch.is_whitespace()
                || matches!(ch, '(' | ')' | '<' | '>' | '"' | '\\')
        }))
    .then(|| format!("{meta}..."))
}

/// htmd pads every cell to its column width. The reference writes compact GFM
/// rows (`| a | b |`) with one `---` per column, so a table with a header
/// separator is rewritten to that spelling. Cell text is unchanged (htmd has
/// already replaced literal pipes with `&#124;`); any other output, such as a
/// table kept as HTML, is left alone.
fn compact_table(markdown: &str) -> Option<String> {
    let cells = |line: &str| -> Option<Vec<String>> {
        let inner = line.strip_prefix('|')?.strip_suffix('|')?;
        Some(
            inner
                .split('|')
                .map(|cell| cell.trim().to_owned())
                .collect(),
        )
    };
    let separator = |line: &str| {
        cells(line).is_some_and(|row| {
            !row.is_empty()
                && row
                    .iter()
                    .all(|cell| !cell.is_empty() && cell.bytes().all(|b| b == b'-'))
        })
    };
    let lines: Vec<&str> = markdown.split('\n').collect();
    let header = lines.iter().position(|line| line.starts_with("| "))?;
    if !lines.get(header + 1).is_some_and(|line| separator(line)) {
        return None;
    }
    let mut output = Vec::with_capacity(lines.len());
    for (index, line) in lines.iter().enumerate() {
        if index < header || !line.starts_with("| ") {
            output.push((*line).to_owned());
            continue;
        }
        let row = cells(line)?;
        output.push(if index == header + 1 {
            format!("| {} |", vec!["---"; row.len()].join(" | "))
        } else {
            format!("| {} |", row.join(" | "))
        });
    }
    Some(output.join("\n"))
}

/// Where a link wrapped around blocks (a card's heading and summary) is
/// written, since a Markdown link cannot hold blocks: its first heading, else
/// its first block with text and no block inside. `None` when the link holds
/// no block or no such block. (An HTML parser never nests links.)
fn block_link_target(anchor: ElementRef<'_>) -> Option<ElementRef<'_>> {
    let inside = || {
        anchor
            .descendants()
            .filter_map(ElementRef::wrap)
            .filter(|node| *node != anchor && !is_hidden(*node))
    };
    let block = |node: &ElementRef<'_>| block_tag(node.value().name());
    if !inside().any(|node| block(&node)) {
        return None;
    }
    inside()
        .find(|node| matches!(node.value().name(), "h1" | "h2" | "h3" | "h4" | "h5" | "h6"))
        .or_else(|| {
            inside().find(|node| {
                block(node)
                    && node.text().any(|text| !text.trim().is_empty())
                    && !node
                        .descendants()
                        .filter_map(ElementRef::wrap)
                        .any(|child| child != *node && block(&child))
            })
        })
}

fn block_tag(name: &str) -> bool {
    matches!(
        name,
        "html"
            | "body"
            | "article"
            | "main"
            | "section"
            | "div"
            | "p"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "pre"
            | "ul"
            | "ol"
            | "li"
            | "dl"
            | "dt"
            | "dd"
            | "table"
            | "thead"
            | "tbody"
            | "tfoot"
            | "tr"
            | "td"
            | "th"
            | "blockquote"
            | "figure"
            | "figcaption"
            | "details"
            | "summary"
            | "hr"
            | "header"
            | "footer"
    )
}

fn escaped_text(value: &str, output: &mut Out) {
    // HTML source line boundaries remain soft Markdown line breaks. The marker
    // prevents the renderer from collapsing them with ordinary inline spaces.
    let mut start = 0;
    for (index, ch) in value.char_indices() {
        if ch == '\n' || ch == '\r' {
            let segment = &value[start..index];
            // The first segment can follow an inline element. Its leading
            // space separates words; only subsequent segments contain source
            // indentation following a line boundary.
            output.text(if start == 0 {
                segment.trim_end_matches([' ', '\t'])
            } else {
                segment.trim_matches([' ', '\t'])
            });
            if !output.soft_break {
                output.start(SOFT_BREAK);
                output.end(SOFT_BREAK);
            }
            start = index + ch.len_utf8();
        }
    }
    output.text(if start > 0 {
        value[start..].trim_start_matches([' ', '\t'])
    } else {
        value
    });
}

fn hidden_inline_style(style: &str) -> bool {
    // Parse declaration boundaries before inspecting property names. Semicolons
    // inside strings, comments or function/block values are not separators.
    fn record(declaration: &str, properties: &mut [Option<(bool, bool)>; 2]) {
        let Some((property, value)) = declaration.split_once(':') else {
            return;
        };
        let property = property.trim();
        let index = if property.eq_ignore_ascii_case("display") {
            0
        } else if property.eq_ignore_ascii_case("visibility") {
            1
        } else {
            return;
        };
        let value = value.trim();
        let (value, important) = value
            .rsplit_once('!')
            .filter(|(_, priority)| priority.trim().eq_ignore_ascii_case("important"))
            .map_or((value, false), |(value, _)| (value.trim(), true));
        // Strings and other invalid keyword values cannot create a hiding rule
        // or override an earlier valid declaration. CSS variable evaluation and
        // stylesheets belong to a browser's computed-style model.
        let keyword = value.to_ascii_lowercase();
        let hidden = match (index, keyword.as_str()) {
            (0, "none") | (1, "hidden" | "collapse") => true,
            (
                0,
                "block" | "inline" | "inline-block" | "flow-root" | "flex" | "inline-flex" | "grid"
                | "inline-grid" | "table" | "inline-table" | "table-row" | "table-cell"
                | "table-caption" | "table-column" | "table-column-group" | "table-row-group"
                | "table-header-group" | "table-footer-group" | "list-item" | "contents",
            )
            | (1, "visible")
            | (_, "initial" | "inherit" | "unset" | "revert" | "revert-layer") => false,
            _ => return,
        };
        if properties[index].is_none_or(|(_, previous_important)| important || !previous_important)
        {
            properties[index] = Some((hidden, important));
        }
    }

    let mut properties = [None; 2];
    let mut declaration = String::new();
    let mut quote = None;
    let mut nesting = 0usize;
    let mut characters = style.chars().peekable();
    while let Some(character) = characters.next() {
        if let Some(delimiter) = quote {
            declaration.push(character);
            if character == '\\' {
                if let Some(escaped) = characters.next() {
                    declaration.push(escaped);
                }
            } else if character == delimiter {
                quote = None;
            }
            continue;
        }
        if character == '/' && characters.peek() == Some(&'*') {
            characters.next();
            while let Some(comment) = characters.next() {
                if comment == '*' && characters.peek() == Some(&'/') {
                    characters.next();
                    break;
                }
            }
            declaration.push(' ');
            continue;
        }
        match character {
            '\'' | '"' => quote = Some(character),
            '(' | '[' | '{' => nesting += 1,
            ')' | ']' | '}' => nesting = nesting.saturating_sub(1),
            ';' if nesting == 0 => {
                record(&declaration, &mut properties);
                declaration.clear();
                continue;
            }
            '\\' => {
                declaration.push(character);
                if let Some(escaped) = characters.next() {
                    declaration.push(escaped);
                }
                continue;
            }
            _ => {}
        }
        declaration.push(character);
    }
    if quote.is_none() && nesting == 0 {
        record(&declaration, &mut properties);
    }
    properties.into_iter().flatten().any(|(hidden, _)| hidden)
}

fn is_hidden(element: ElementRef<'_>) -> bool {
    let value = element.value();
    value.attribute("hidden").is_some() || value.attribute("style").is_some_and(hidden_inline_style)
}

fn has_class(element: ElementRef<'_>, class: &str) -> bool {
    has_any_class(element, &[class])
}

/// Whether the element has one of these classes, reading its classes once.
fn has_any_class(element: ElementRef<'_>, classes: &[&str]) -> bool {
    element
        .value()
        .attribute("class")
        .is_some_and(|names| names.split_whitespace().any(|name| classes.contains(&name)))
}

fn tex_script(element: ElementRef<'_>) -> Option<bool> {
    if element.value().name() != "script" {
        return None;
    }
    let mut parts = element.value().attribute("type")?.split(';');
    if !parts.next()?.trim().eq_ignore_ascii_case("math/tex") {
        return None;
    }
    Some(parts.any(|part| {
        part.split_once('=').is_some_and(|(key, value)| {
            key.trim().eq_ignore_ascii_case("mode") && value.trim().eq_ignore_ascii_case("display")
        })
    }))
}

fn math_symbol(value: &str) -> String {
    let symbol = match value {
        "α" => r"\alpha",
        "β" => r"\beta",
        "γ" => r"\gamma",
        "δ" => r"\delta",
        "ε" => r"\epsilon",
        "ζ" => r"\zeta",
        "η" => r"\eta",
        "θ" => r"\theta",
        "ι" => r"\iota",
        "κ" => r"\kappa",
        "λ" => r"\lambda",
        "μ" => r"\mu",
        "ν" => r"\nu",
        "ξ" => r"\xi",
        "π" => r"\pi",
        "ρ" => r"\rho",
        "σ" => r"\sigma",
        "τ" => r"\tau",
        "υ" => r"\upsilon",
        "ϕ" => r"\phi",
        "φ" => r"\varphi",
        "χ" => r"\chi",
        "ψ" => r"\psi",
        "ω" => r"\omega",
        "Γ" => r"\Gamma",
        "Δ" => r"\Delta",
        "Θ" => r"\Theta",
        "Λ" => r"\Lambda",
        "Ξ" => r"\Xi",
        "Π" => r"\Pi",
        "Σ" => r"\Sigma",
        "Φ" => r"\Phi",
        "Ψ" => r"\Psi",
        "Ω" => r"\Omega",
        "≠" => r"\neq",
        "≤" => r"\leq",
        "≥" => r"\geq",
        "±" => r"\pm",
        "∓" => r"\mp",
        "×" => r"\times",
        "÷" => r"\div",
        "∞" => r"\infty",
        "∑" => r"\sum",
        "∏" => r"\prod",
        "∫" => r"\int",
        "∮" => r"\oint",
        "∂" => r"\partial",
        "∇" => r"\nabla",
        "→" => r"\rightarrow",
        "←" => r"\leftarrow",
        "⇒" => r"\Rightarrow",
        "⇐" => r"\Leftarrow",
        "↦" => r"\mapsto",
        "∈" => r"\in",
        "∉" => r"\notin",
        "⊂" => r"\subset",
        "⊃" => r"\supset",
        "⊆" => r"\subseteq",
        "⊇" => r"\supseteq",
        "∪" => r"\cup",
        "∩" => r"\cap",
        "∧" => r"\wedge",
        "∨" => r"\vee",
        "¬" => r"\neg",
        "∅" => r"\emptyset",
        "∀" => r"\forall",
        "∃" => r"\exists",
        "⟹" => r"\implies",
        "⟸" => r"\impliedby",
        "⟺" => r"\iff",
        "⟨" => r"\langle",
        "⟩" => r"\rangle",
        "⌊" => r"\lfloor",
        "⌋" => r"\rfloor",
        "⌈" => r"\lceil",
        "⌉" => r"\rceil",
        "⋯" => r"\cdots",
        "…" => r"\ldots",
        "⋅" | "·" => r"\cdot",
        "−" => "-",
        "⁡" | "⁢" | "⁣" => "",
        _ => value,
    };
    symbol.to_owned()
}

fn inert_math_node(name: &str) -> bool {
    matches!(
        name,
        "script" | "style" | "annotation-xml" | "iframe" | "object" | "embed" | "template"
    )
}

fn math_text(element: ElementRef<'_>, depth: usize) -> Result<String> {
    if inert_math_node(element.value().name()) {
        return Ok(String::new());
    }
    if depth > 64 {
        return Err(Error::Conversion(
            "MathML nesting exceeds 64 elements".into(),
        ));
    }
    let mut result = String::new();
    for child in element.children() {
        if let Some(child) = ElementRef::wrap(child) {
            result.push_str(&math_text(child, depth + 1)?);
        } else if let scraper::Node::Text(text) = child.value() {
            result.push_str(text);
        }
    }
    Ok(result)
}

fn mathml(element: ElementRef<'_>, depth: usize) -> Result<String> {
    if inert_math_node(element.value().name()) {
        return Ok(String::new());
    }
    if depth > 64 {
        return Err(Error::Conversion(
            "MathML nesting exceeds 64 elements".into(),
        ));
    }
    let name = element.value().name();
    let children = element.child_elements().collect::<Vec<_>>();
    let mut parts = Vec::new();
    for child in element.children() {
        if let Some(child) = ElementRef::wrap(child) {
            if !matches!(child.value().name(), "annotation" | "annotation-xml") {
                parts.push(mathml(child, depth + 1)?);
            }
        } else if let scraper::Node::Text(text) = child.value()
            && !text.trim().is_empty()
        {
            parts.push(text.trim().to_owned());
        }
    }
    let joined = parts
        .iter()
        .filter(|part| !part.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    let text = if matches!(name, "mi" | "mn" | "mo" | "mtext" | "ms") {
        math_text(element, depth)?
    } else {
        String::new()
    };
    let text = text.trim();
    let at = |index: usize| parts.get(index).map(String::as_str).unwrap_or("");
    Ok(match name {
        "semantics" => parts.into_iter().next().unwrap_or_default(),
        "annotation" | "annotation-xml" | "none" | "mprescripts" => String::new(),
        "mi" if text.chars().count() > 1 && text.chars().all(char::is_alphabetic) => {
            format!(r"\mathrm{{{text}}}")
        }
        "mi" | "mn" | "mo" => math_symbol(text),
        "mtext" | "ms" => {
            let mut escaped = String::new();
            for character in text.chars() {
                match character {
                    '\\' => escaped.push_str(r"\textbackslash{}"),
                    '{' | '}' | '$' | '&' | '%' | '#' | '_' => {
                        escaped.push('\\');
                        escaped.push(character);
                    }
                    '^' => escaped.push_str(r"\textasciicircum{}"),
                    '~' => escaped.push_str(r"\textasciitilde{}"),
                    _ => escaped.push(character),
                }
            }
            format!(r"\text{{{escaped}}}")
        }
        "mspace" => r"\;".into(),
        "msup" if parts.len() >= 2 => format!("{}^{{{}}}", at(0), at(1)),
        "msub" if parts.len() >= 2 => format!("{}_{{{}}}", at(0), at(1)),
        "msubsup" if parts.len() >= 3 => format!("{}_{{{}}}^{{{}}}", at(0), at(1), at(2)),
        "mfrac" if parts.len() >= 2 => format!(r"\frac{{{}}}{{{}}}", at(0), at(1)),
        "msqrt" => format!(r"\sqrt{{{joined}}}"),
        "mroot" if parts.len() >= 2 => format!(r"\sqrt[{}]{{{}}}", at(1), at(0)),
        "mover" if parts.len() >= 2 => {
            let raw = children
                .get(1)
                .map(|element| math_text(*element, depth + 1))
                .transpose()?
                .unwrap_or_default();
            let accent = match raw.trim() {
                "˙" | "̇" => Some("dot"),
                "¨" | "̈" => Some("ddot"),
                "¯" | "‾" => Some("overline"),
                "^" | "̂" => Some("hat"),
                "~" | "̃" | "˜" => Some("tilde"),
                "→" | "⃗" => Some("vec"),
                "⏞" => Some("overbrace"),
                _ => None,
            };
            accent.map_or_else(
                || format!(r"\overset{{{}}}{{{}}}", at(1), at(0)),
                |accent| format!("\\{accent}{{{}}}", at(0)),
            )
        }
        "munder" if parts.len() >= 2 => format!(r"\underset{{{}}}{{{}}}", at(1), at(0)),
        "munderover" if parts.len() >= 3 => format!("{}_{{{}}}^{{{}}}", at(0), at(1), at(2)),
        "mtable" => format!(r"\begin{{aligned}}{}\end{{aligned}}", parts.join(r" \\ ")),
        "mtr" | "mlabeledtr" => parts.join(" & "),
        "mfenced" => {
            let open = element.value().attribute("open").unwrap_or("(");
            let close = element.value().attribute("close").unwrap_or(")");
            let separators = element
                .value()
                .attribute("separators")
                .unwrap_or(",")
                .chars()
                .filter(|ch| !ch.is_whitespace())
                .collect::<Vec<_>>();
            let mut content = String::new();
            for (index, part) in parts.iter().enumerate() {
                if index > 0 && !separators.is_empty() {
                    content.push(separators[(index - 1).min(separators.len() - 1)]);
                    content.push(' ');
                }
                content.push_str(part);
            }
            let delimiter = |value: &str| match value {
                "" => ".".into(),
                "{" => r"\{".into(),
                "}" => r"\}".into(),
                value => math_symbol(value),
            };
            format!(
                r"\left{}{content}\right{}",
                delimiter(open),
                delimiter(close)
            )
        }
        "menclose" if element.value().attribute("notation") == Some("top") => {
            format!(r"\overline{{{joined}}}")
        }
        "mphantom" => format!(r"\phantom{{{joined}}}"),
        _ => joined,
    })
}

fn math_container(element: ElementRef<'_>) -> bool {
    element.value().name() == "math"
        || element.value().name() == "mjx-container"
        || has_any_class(
            element,
            &[
                "katex",
                "katex-display",
                "math-inline",
                "math-block",
                "hurmet-tex",
                "mwe-math-element",
            ],
        )
}

/// Whether decoded text carries a TeX command (a backslash and two letters),
/// the reference's test for LaTeX in image URLs and alt text.
fn looks_like_latex(text: &str) -> bool {
    text.as_bytes().windows(3).any(|window| {
        window[0] == b'\\' && window[1].is_ascii_alphabetic() && window[2].is_ascii_alphabetic()
    })
}

/// `+` as space, then percent escapes as UTF-8, as the reference decodes.
fn url_component(raw: &str) -> String {
    let raw = raw.replace('+', " ");
    let bytes = raw.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let hex = |byte: u8| (byte as char).to_digit(16);
        if bytes[index] == b'%'
            && let (Some(high), Some(low)) = (
                bytes.get(index + 1).and_then(|b| hex(*b)),
                bytes.get(index + 2).and_then(|b| hex(*b)),
            )
        {
            decoded.push((high * 16 + low) as u8);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

/// The smallest size, in pixels, of an image that can be content: below it
/// on either axis an image is a spacer, a tracking pixel or an icon (the
/// threshold defuddle uses).
const MIN_IMAGE_SIZE: f64 = 33.0;

/// For an `img` or `svg` smaller than [`MIN_IMAGE_SIZE`] on either axis by
/// what it declares (width/height attributes, inline style, an SVG's
/// viewBox when nothing else is given, a `1x` srcset URL's width), the text
/// to write instead: an emoji image's character, else nothing. `None` for
/// any other element, an image that declares no size (or only percentages)
/// and an image rendering an equation.
fn small_image(element: ElementRef<'_>) -> Option<String> {
    let name = element.value().name();
    if !matches!(name, "img" | "svg") {
        return None;
    }
    let pixels = |value: &str| -> Option<f64> {
        let value = value.trim();
        if value.contains('%') {
            return None;
        }
        let number = value.trim_end_matches("px").trim();
        number
            .parse::<f64>()
            .ok()
            .filter(|n| *n > 0.0 && n.is_finite())
    };
    let style = element.value().attribute("style").unwrap_or("");
    let style_size = |property: &str| {
        style.split(';').find_map(|declaration| {
            let (key, value) = declaration.split_once(':')?;
            key.trim()
                .eq_ignore_ascii_case(property)
                .then(|| pixels(value))
                .flatten()
        })
    };
    let mut width = element
        .value()
        .attribute("width")
        .and_then(pixels)
        .or_else(|| style_size("width"));
    let mut height = element
        .value()
        .attribute("height")
        .and_then(pixels)
        .or_else(|| style_size("height"));
    if name == "svg" && width.is_none() && height.is_none() {
        let view_box = element
            .value()
            .attribute("viewBox")
            .or_else(|| element.value().attribute("viewbox"));
        if let Some(parts) = view_box.map(|v| {
            v.split(|c: char| c.is_whitespace() || c == ',')
                .filter(|p| !p.is_empty())
                .collect::<Vec<_>>()
        }) && parts.len() == 4
        {
            width = pixels(parts[2]);
            height = pixels(parts[3]);
        }
    }
    if name == "img" && width.is_none() && height.is_none() {
        // Images served at a tiny width through CDN parameters.
        width = element.value().attribute("srcset").and_then(|srcset| {
            let one_x = srcset
                .split(',')
                .map(str::trim)
                .find(|candidate| candidate.ends_with(" 1x"))?;
            let url = one_x.split_whitespace().next()?;
            ["width=", "width/", "w_", "w=", "w:"]
                .iter()
                .find_map(|key| {
                    let at = url.find(key)? + key.len();
                    let digits: String =
                        url[at..].chars().take_while(char::is_ascii_digit).collect();
                    digits.parse::<f64>().ok().filter(|n| *n > 0.0)
                })
        });
    }
    if width.is_none() && height.is_none() {
        return None;
    }
    let small = |size: Option<f64>| size.is_some_and(|size| size < MIN_IMAGE_SIZE);
    if !small(width) && !small(height) {
        return None;
    }
    if name == "img"
        && (latex_image(element).is_some()
            || element
                .value()
                .attribute("alt")
                .is_some_and(looks_like_latex))
    {
        return None;
    }
    let alt = element.value().attribute("alt").unwrap_or("").trim();
    let emoji = name == "img"
        && !alt.is_empty()
        && alt.chars().count() <= 8
        && !alt.chars().any(|c| c.is_alphanumeric());
    Some(if emoji { alt.to_owned() } else { String::new() })
}

/// LaTeX an image rendered by a TeX image service carries, as the reference
/// finds it: a named query parameter, the whole query, an escaped path
/// segment, then TeX-looking alt text. Other images stay images.
fn latex_image(element: ElementRef<'_>) -> Option<(String, bool)> {
    if element.value().name() != "img" {
        return None;
    }
    let src = element.value().attribute("src")?;
    let decoded = |raw: &str| Some(url_component(raw)).filter(|text| looks_like_latex(text));
    let (path, query) = match src.split_once('?') {
        Some((path, rest)) => (path, Some(rest.split('#').next().unwrap_or(""))),
        None => (src, None),
    };
    let named = query.and_then(|query| {
        ["latex", "chl", "tex", "eq", "math"]
            .into_iter()
            .find_map(|name| {
                query.split('&').find_map(|pair| {
                    let (key, value) = pair.split_once('=')?;
                    (key.eq_ignore_ascii_case(name) && !value.is_empty())
                        .then(|| decoded(value))
                        .flatten()
                })
            })
    });
    let latex = named
        .or_else(|| query.filter(|query| !query.is_empty()).and_then(decoded))
        .or_else(|| {
            path.split('/')
                .rev()
                .filter(|segment| segment.contains("%5C") || segment.contains("%5c"))
                .find_map(decoded)
        })
        .or_else(|| {
            element
                .value()
                .attribute("alt")
                .filter(|alt| looks_like_latex(alt))
                .map(str::to_owned)
        })?;
    let alone = element
        .parent()
        .and_then(ElementRef::wrap)
        .is_some_and(|parent| parent.value().name() == "p" && parent.children().count() == 1);
    let latex = latex.trim().to_owned();
    (!latex.is_empty()).then(|| {
        let block = latex.contains("\\begin{") || alone;
        (latex, block)
    })
}

fn math_expression(element: ElementRef<'_>) -> Result<Option<(String, bool)>> {
    if let Some(math) = latex_image(element) {
        return Ok(Some(math));
    }
    if let Some(block) = tex_script(element) {
        let text = element.text().collect::<String>();
        return Ok((!text.trim().is_empty()).then(|| (text.trim().to_owned(), block)));
    }
    if !math_container(element) {
        return Ok(None);
    }
    let inner = if element.value().name() == "math" {
        Some(element)
    } else {
        element.select(&selector("math")).next()
    };
    let block = has_class(element, "katex-display")
        || has_class(element, "math-block")
        || matches!(element.value().attribute("display"), Some("block" | "true"))
        || inner.is_some_and(|math| math.value().attribute("display") == Some("block"));
    let attribute = |element: ElementRef<'_>| {
        ["data-latex", "data-math", "data-entry", "alttext"]
            .into_iter()
            .find_map(|key| {
                element
                    .value()
                    .attribute(key)
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .map(str::to_owned)
            })
    };
    let mut latex = attribute(element).or_else(|| inner.and_then(attribute));
    if latex.is_none() {
        latex = element
            .select(&selector("annotation"))
            .find(|node| {
                node.value().attribute("encoding").is_some_and(|encoding| {
                    matches!(
                        encoding.to_ascii_lowercase().as_str(),
                        "application/x-tex" | "application/x-latex" | "text/tex"
                    )
                })
            })
            .map(|node| math_text(node, 0))
            .transpose()?
            .filter(|text| !text.trim().is_empty());
    }
    if latex.is_none()
        && let Some(math) = inner
    {
        latex = Some(mathml(math, 0)?);
    }
    if latex.is_none() && has_class(element, "mwe-math-element") {
        latex = element.select(&selector("img[alt]")).find_map(|image| {
            image
                .value()
                .attribute("alt")
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(str::to_owned)
        });
    }
    Ok(latex
        .filter(|text| !text.trim().is_empty())
        .map(|text| (text.trim().to_owned(), block)))
}

fn duplicate_math_preview(element: ElementRef<'_>) -> bool {
    has_any_class(
        element,
        &[
            "MathJax_Preview",
            "MathJax",
            "MathJax_Display",
            "MathJax_SVG",
            "MathJax_MathML",
        ],
    ) && element
        .parent()
        .and_then(ElementRef::wrap)
        .is_some_and(|parent| {
            parent.child_elements().any(|sibling| {
                tex_script(sibling).is_some() && sibling.text().any(|text| !text.trim().is_empty())
            })
        })
}

fn emit_math(latex: &str, block: bool, output: &mut Out) {
    output.start("markitai-math");
    output.attribute("data-latex", latex);
    if block {
        output.attribute("data-display", "block");
    }
    output.end("markitai-math");
}

// DOM element addresses serve only as stable identity keys while this immutable
// parse tree is borrowed. They are never dereferenced through these keys.
fn element_key(element: ElementRef<'_>) -> usize {
    element.value() as *const scraper::node::Element as usize
}

fn contains_element(parent: ElementRef<'_>, child: ElementRef<'_>) -> bool {
    parent == child || child.ancestors().any(|node| node.id() == parent.id())
}

fn note_heading(element: ElementRef<'_>) -> bool {
    matches!(
        element.value().name(),
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
    ) && matches!(
        plain(element).to_ascii_lowercase().as_str(),
        "footnote"
            | "footnotes"
            | "foot notes"
            | "endnotes"
            | "end notes"
            | "note"
            | "notes"
            | "reference"
            | "references"
            | "sidenotes"
    )
}

fn note_number(text: &str) -> Option<String> {
    let number = text.trim().trim_matches(['[', ']', '(', ')', '.']);
    (!number.is_empty()
        && number.len() <= 4
        && number.bytes().all(|byte| byte.is_ascii_digit())
        && !number.starts_with('0'))
    .then(|| number.to_owned())
}

fn first_content(element: ElementRef<'_>) -> Option<ElementRef<'_>> {
    for child in element.children() {
        if let Some(child) = ElementRef::wrap(child) {
            return Some(child);
        }
        if let scraper::Node::Text(text) = child.value()
            && !text.trim().is_empty()
        {
            return None;
        }
    }
    None
}

fn leading_note_marker(element: ElementRef<'_>) -> Option<(String, ElementRef<'_>)> {
    let mut first = first_content(element)?;
    if matches!(first.value().name(), "b" | "strong" | "span")
        && let Some(inner) = first_content(first)
        && inner.value().name() == "sup"
    {
        first = inner;
    }
    matches!(first.value().name(), "sup" | "strong")
        .then(|| note_number(&plain(first)).map(|number| (number, first)))?
}

fn note_context(facts: &Facts, element: ElementRef<'_>) -> bool {
    facts.remember(element, Fact::NoteContext, || {
        let value = element.value();
        value.attribute("role").is_some_and(|role| {
            matches!(
                role,
                "doc-footnote" | "doc-endnote" | "doc-endnotes" | "doc-footnotes"
            )
        }) || value
            .attribute("id")
            .is_some_and(|id| matches!(id, "footnotes" | "endnotes"))
            || value.attribute("data-footnotes").is_some()
            || value.attribute("data-type") == Some("footnote")
            || value.class_names().any(|class| {
                matches!(
                    class,
                    "footnotes"
                        | "footnote"
                        | "footnotes-list"
                        | "references"
                        | "reflist"
                        | "footnote-definition"
                        | "footnote-definitions"
                        | "footdef"
                        | "footnotes-footer"
                        | "footnote-footer"
                        | "easy-footnotes-wrapper"
                        | "wp-block-footnotes"
                        | "footnotes-segment"
                )
            })
    })
}

fn note_fragment(link: ElementRef<'_>, base: Option<&Url>) -> Option<String> {
    let href = link.value().attribute("href")?;
    let (prefix, fragment) = href.rsplit_once('#')?;
    if fragment.is_empty() || safe_url(href, base).is_none() {
        return None;
    }
    // A remote page sharing an anchor name is not a local footnote. Explicit
    // HTMLBook markers and Word's generated export backlinks carry their own
    // structural evidence and are handled separately.
    if !prefix.is_empty()
        && link.value().attribute("data-type") != Some("noteref")
        && !base.is_some_and(|base| {
            base.join(href).is_ok_and(|mut target| {
                let mut page = base.clone();
                page.set_fragment(None);
                target.set_fragment(None);
                target == page
            })
        })
    {
        return None;
    }
    Some(fragment.to_owned())
}

fn reference_candidate(element: ElementRef<'_>) -> bool {
    if element.value().name() != "a" {
        return true;
    }
    if element.value().attribute("data-footnote-ref").is_some()
        || element.value().attribute("data-type") == Some("noteref")
        || element
            .value()
            .attribute("role")
            .is_some_and(|role| matches!(role, "doc-noteref" | "doc-biblioref"))
        || element
            .value()
            .class_names()
            .any(|class| matches!(class, "footnote-ref" | "footnote-anchor" | "noteref"))
    {
        return true;
    }
    // A marker is digits in brackets or note signs; the text of a link that
    // says anything else is not collected.
    if element.text().flat_map(str::chars).any(|ch| {
        !ch.is_whitespace()
            && !matches!(
                ch,
                '0'..='9' | '[' | ']' | '(' | ')' | '.' | '*' | '†' | '‡'
            )
    }) {
        return false;
    }
    let text = plain(element);
    note_number(&text).is_some() || (!text.is_empty() && text.chars().all(|ch| "*†‡".contains(ch)))
}

fn literal_container(element: ElementRef<'_>) -> bool {
    matches!(element.value().name(), "pre" | "code" | "script" | "style")
        || code::is_block(element)
        || math_container(element)
        || duplicate_math_preview(element)
}

fn visible_reference(facts: &Facts, element: ElementRef<'_>, prune_chrome: bool) -> bool {
    std::iter::once(element)
        .chain(element.ancestors().filter_map(ElementRef::wrap))
        .all(|parent| {
            !facts.hidden(parent)
                && !(prune_chrome && article::excluded(facts, parent))
                && !matches!(
                    parent.value().name(),
                    "script"
                        | "style"
                        | "nav"
                        | "footer"
                        | "form"
                        | "button"
                        | "input"
                        | "select"
                        | "textarea"
                        | "iframe"
                        | "object"
                        | "embed"
                        | "head"
                        | "template"
                        | "noscript"
                )
        })
}

fn generic_note_section(facts: &Facts, element: ElementRef<'_>) -> bool {
    let label = |heading: ElementRef<'_>| {
        note_heading(heading)
            || (matches!(
                heading.value().name(),
                "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
            ) && matches!(
                plain(heading).to_ascii_lowercase().as_str(),
                "references and notes" | "notes and references" | "bibliography"
            ))
    };
    note_context(facts, element)
        || ["id", "class"].iter().any(|attr| {
            element.value().attribute(attr).is_some_and(|value| {
                value
                    .split_ascii_whitespace()
                    .any(|token| matches!(token, "note" | "notes" | "endnotes" | "bibliography"))
            })
        })
        || element.child_elements().any(label)
        || element
            .prev_siblings()
            .filter_map(ElementRef::wrap)
            .next()
            .is_some_and(label)
}

fn generic_continuation(element: ElementRef<'_>) -> bool {
    if !matches!(
        element.value().name(),
        "p" | "ul" | "ol" | "blockquote" | "pre"
    ) || element.value().attribute("id").is_some()
        || inside(element).any(fragment_target)
        || element
            .value()
            .class_names()
            .any(|class| matches!(class, "seealso" | "see-also" | "related" | "notetitle"))
    {
        return false;
    }
    let text = plain(element).to_ascii_lowercase();
    if ["see also:", "related:", "update:"]
        .iter()
        .any(|label| text.starts_with(label))
    {
        return false;
    }
    !first_content(element).is_some_and(|first| {
        matches!(first.value().name(), "b" | "strong")
            && plain(first).to_ascii_lowercase().starts_with("update")
    })
}

fn numeric_text_marker(element: ElementRef<'_>, number: &str) -> Option<String> {
    for child in element.children() {
        match child.value() {
            scraper::Node::Comment(_) => continue,
            scraper::Node::Text(text) if text.trim().is_empty() => continue,
            scraper::Node::Text(text) => {
                let tail = text.trim_start().strip_prefix(number)?.strip_prefix('.')?;
                if !tail.is_empty() && !tail.chars().next().is_some_and(char::is_whitespace) {
                    return None;
                }
                let length = text.len() - tail.trim_start().len();
                return Some(text[..length].to_owned());
            }
            _ => return None,
        }
    }
    None
}

fn note_has_content(
    element: ElementRef<'_>,
    markers: &[ElementRef<'_>],
    mut text_marker: Option<&str>,
    depth: usize,
) -> bool {
    if depth > MAX_DEPTH {
        return true;
    } // The serializer keeps that subtree as plain text.
    if markers.contains(&element)
        || (depth > 0 && is_hidden(element))
        || element.value().attribute("role") == Some("doc-backlink")
        || element.value().attribute("data-footnote-backref").is_some()
        || element.value().class_names().any(|class| {
            matches!(
                class,
                "footnote-backref"
                    | "data-footnote-backref"
                    | "mw-cite-backlink"
                    | "easy-footnote-to-top"
            )
        })
    {
        return false;
    }
    let name = element.value().name();
    if name == "script" && tex_script(element).is_some() {
        return !plain(element).is_empty();
    }
    if matches!(
        name,
        "script"
            | "style"
            | "nav"
            | "footer"
            | "form"
            | "button"
            | "input"
            | "select"
            | "textarea"
            | "iframe"
            | "object"
            | "embed"
            | "head"
            | "template"
            | "noscript"
    ) {
        return false;
    }
    if name == "img" {
        return element
            .value()
            .attribute("data-src")
            .or_else(|| element.value().attribute("src"))
            .is_some_and(|src| image_source(src, None).is_some());
    }
    if name == "a"
        && element
            .value()
            .attribute("href")
            .is_some_and(|href| href.starts_with('#'))
        && !plain(element).is_empty()
        && plain(element)
            .chars()
            .all(|ch| "^↩↥↑↵⤴⤵⏎\u{fe0e}\u{fe0f}".contains(ch))
    {
        return false;
    }
    element.children().any(|child| {
        if let Some(child) = ElementRef::wrap(child) {
            note_has_content(child, markers, None, depth + 1)
        } else if let scraper::Node::Text(text) = child.value() {
            let text = if let Some(prefix) = text_marker
                && let Some(rest) = text.strip_prefix(prefix)
            {
                text_marker = None;
                rest
            } else {
                text.as_ref()
            };
            !text.trim().is_empty()
        } else {
            false
        }
    })
}

/// What footnote recovery asks of every element of a page, read once in
/// document order (an element after its ancestors) instead of from each
/// element's ancestors again, and kept by position in the tree.
struct Reach {
    /// Inside a literal container: code or math (whose classification may
    /// read a subtree), `pre`, `code`, `script` or `style`.
    literal: Vec<bool>,
    /// In scope: inside the region, or inside a notes container outside it.
    scope: Vec<bool>,
    /// In a note's context: the element, or one around it reached through
    /// elements in scope.
    context: Vec<bool>,
}

impl Reach {
    /// `elements`: those of `document`, in document order.
    fn read(
        root: ElementRef<'_>,
        document: ElementRef<'_>,
        elements: &[ElementRef<'_>],
        facts: &Facts,
    ) -> Self {
        let nodes = document.tree().values().len();
        let at = |element: ElementRef<'_>| facts::position(element.id());
        let mut reach = Self {
            literal: vec![false; nodes],
            scope: vec![false; nodes],
            context: vec![false; nodes],
        };
        // Inside the region; the region or an element around it; below an
        // element whose content is inert.
        let (mut region, mut around, mut inert) =
            (vec![false; nodes], vec![false; nodes], vec![false; nodes]);
        for node in std::iter::once(*root).chain(root.ancestors()) {
            around[facts::position(node.id())] = true;
        }
        let inert_content = |element: ElementRef<'_>| {
            matches!(
                element.value().name(),
                "head" | "template" | "noscript" | "iframe" | "object" | "embed"
            )
        };
        for &element in elements {
            let index = at(element);
            let parent = element.parent().and_then(ElementRef::wrap);
            reach.literal[index] = parent.is_some_and(|parent| reach.literal[at(parent)])
                || literal_container(element);
            // The closest element around this one; for the document element,
            // one outside the elements read here.
            let up = element.ancestors().find_map(ElementRef::wrap);
            region[index] = element == root || up.is_some_and(|up| region[at(up)]);
            inert[index] = if element == document {
                element
                    .ancestors()
                    .filter_map(ElementRef::wrap)
                    .any(inert_content)
            } else {
                up.is_some_and(|up| inert[at(up)] || inert_content(up))
            };
            // A container of notes outside the region.
            let external = !region[index]
                && !around[index]
                && matches!(
                    element.value().name(),
                    "div" | "section" | "aside" | "ol" | "ul" | "p" | "li"
                )
                && !reach.literal[index]
                && !inert[index]
                && (note_context(facts, element)
                    || (["id", "class"].iter().any(|attr| {
                        element.value().attribute(attr).is_some_and(|value| {
                            value
                                .as_bytes()
                                .windows(8)
                                .any(|word| word.eq_ignore_ascii_case(b"footnote"))
                        })
                    }) && inside(element).any(note_heading)));
            let up = up.map(at);
            reach.scope[index] = region[index] || external || up.is_some_and(|up| reach.scope[up]);
            // (An element in a note's context is in scope.)
            reach.context[index] = reach.scope[index]
                && (note_context(facts, element) || up.is_some_and(|up| reach.context[up]));
        }
        reach
    }

    fn literal(&self, element: ElementRef<'_>) -> bool {
        self.literal[facts::position(element.id())]
    }

    fn in_scope(&self, element: ElementRef<'_>) -> bool {
        self.scope[facts::position(element.id())]
    }

    fn context(&self, element: ElementRef<'_>) -> bool {
        self.context[facts::position(element.id())]
    }
}

struct Footnote<'a> {
    nodes: Vec<ElementRef<'a>>,
    aliases: Vec<String>,
    markers: Vec<ElementRef<'a>>,
    references: Vec<ElementRef<'a>>,
    text_marker: Option<String>,
}

struct Footnotes<'a> {
    facts: &'a Facts,
    prune_chrome: bool,
    definitions: Vec<Footnote<'a>>,
    references: HashMap<usize, usize>,
    removed: HashSet<usize>,
    markers: HashSet<usize>,
    text_markers: HashMap<usize, String>,
    /// In-page tables of contents of a full page, left out.
    contents: HashSet<usize>,
    /// Blocks beside an article's body that are page furniture, and the
    /// quoted history of a web mail message, left out.
    furniture: HashSet<usize>,
    /// Where a saved page's relative links point when no page URL is given:
    /// its `<base href>`, else its canonical address; `false` when that is only
    /// the site's home page, which resolves only root-relative links.
    link_base: Option<(Url, bool)>,
    /// Links around blocks, written as their content ...
    block_links: HashSet<usize>,
    /// ... with the link on the block that names them: target → link.
    link_targets: HashMap<usize, ElementRef<'a>>,
}

impl<'a> Footnotes<'a> {
    fn add(
        &mut self,
        nodes: Vec<ElementRef<'a>>,
        aliases: Vec<String>,
        markers: Vec<ElementRef<'a>>,
        references: Vec<ElementRef<'a>>,
    ) {
        if nodes.is_empty()
            || !nodes
                .iter()
                .any(|node| note_has_content(*node, &markers, None, 0))
            || self.definitions.iter().any(|note| {
                note.nodes
                    .iter()
                    .any(|existing| nodes.iter().any(|node| contains_element(*existing, *node)))
            })
        {
            return;
        }
        self.definitions.push(Footnote {
            nodes,
            aliases,
            markers,
            references,
            text_marker: None,
        });
    }

    /// An in-page table of contents or page furniture of a full page.
    fn left_out(&self, element: ElementRef<'_>) -> bool {
        let key = element_key(element);
        self.contents.contains(&key) || self.furniture.contains(&key)
    }

    /// The element or a block around it is left out.
    fn inside_left_out(&self, element: ElementRef<'_>) -> bool {
        (!self.contents.is_empty() || !self.furniture.is_empty())
            && element
                .ancestors()
                .filter_map(ElementRef::wrap)
                .any(|parent| self.left_out(parent))
    }

    fn inside_definition(&self, element: ElementRef<'_>) -> bool {
        self.definitions.iter().any(|note| {
            note.nodes
                .iter()
                .any(|node| contains_element(*node, element))
        })
    }

    fn collect(
        root: ElementRef<'a>,
        document: ElementRef<'a>,
        base: Option<&Url>,
        prune_chrome: bool,
        facts: &'a Facts,
    ) -> Self {
        let mut notes = Self {
            facts,
            prune_chrome,
            definitions: Vec::new(),
            references: HashMap::new(),
            removed: HashSet::new(),
            markers: HashSet::new(),
            text_markers: HashMap::new(),
            contents: if prune_chrome {
                article::contents(root, document)
                    .into_iter()
                    .map(element_key)
                    .collect()
            } else {
                HashSet::new()
            },
            furniture: if prune_chrome {
                furniture::beside_body(facts, root)
                    .into_iter()
                    .chain(mail::quoted_history(root))
                    .map(element_key)
                    .collect()
            } else {
                HashSet::new()
            },
            link_base: None,
            block_links: HashSet::new(),
            link_targets: HashMap::new(),
        };
        if prune_chrome && base.is_none() {
            notes.link_base = saved_page_base(document);
        }
        // Links around blocks, reference candidates and inline popovers, in
        // one walk over the region.
        let mut references = Vec::new();
        let mut inline = Vec::new();
        for element in inside(root) {
            if link(element)
                && let Some(target) = block_link_target(element)
            {
                notes.block_links.insert(element_key(element));
                notes.link_targets.insert(element_key(target), element);
            }
            if reference_markup(element) && reference_candidate(element) {
                references.push(element);
            }
            if span_of(
                element,
                &[
                    "footnote-container",
                    "sidenote-container",
                    "inline-footnote",
                ],
            ) {
                inline.push(element);
            }
        }
        // Definitions only move when a reference or inline popover resolves.
        // Ordinary documents need no document-wide ID, literal or note indexes.
        if references.is_empty() && inline.is_empty() {
            return notes;
        }
        let elements = document
            .descendants()
            .filter_map(ElementRef::wrap)
            .collect::<Vec<_>>();
        let reach = Reach::read(root, document, &elements, facts);
        let in_literal = |element| reach.literal(element);
        let in_scope = |element| reach.in_scope(element);
        let mut ids = HashMap::new();
        for element in &elements {
            for attribute in ["id", "name"] {
                if let Some(id) = element.value().attribute(attribute) {
                    ids.entry(id).or_insert(*element);
                }
            }
        }
        let references = references
            .into_iter()
            .filter(|element| {
                !in_literal(*element)
                    && visible_reference(facts, *element, prune_chrome)
                    && !notes.inside_left_out(*element)
            })
            .collect::<Vec<_>>();

        // Inline popovers have a definition and a reference at the same DOM
        // position. Only the identified content root bypasses hidden styling.
        for container in inline {
            if in_literal(container) || !visible_reference(facts, container, prune_chrome) {
                continue;
            }
            if let Some(content) = inside(container)
                .find(|node| span_of(*node, &["footnote", "sidenote", "footnoteContent"]))
            {
                if content
                    .ancestors()
                    .filter_map(ElementRef::wrap)
                    .take_while(|ancestor| *ancestor != container)
                    .any(is_hidden)
                {
                    continue;
                }
                notes.add(vec![content], vec![], vec![], vec![container]);
            }
        }
        for reference in &references {
            if let Some(id) = reference.value().attribute("data-definition")
                && let Some(target) = ids.get(id).copied()
                && target.value().name() == "aside"
                && in_scope(target)
            {
                notes.add(vec![target], vec![id.to_owned()], vec![], vec![*reference]);
            }
            if reference.value().name() == "label" && has_class(*reference, "footref") {
                let content = reference
                    .next_siblings()
                    .filter_map(ElementRef::wrap)
                    .find(|node| node.value().name() != "input");
                if let Some(content) = content.filter(|node| has_class(*node, "sidenote")) {
                    let markers = leading_note_marker(content)
                        .map(|(_, marker)| marker)
                        .into_iter()
                        .collect();
                    notes.add(vec![content], vec![], markers, vec![*reference]);
                }
            }
        }

        // Known definition structures are accepted even when there is one note.
        // Their bodies are retained unless a real reference can be resolved.
        for element in elements
            .iter()
            .copied()
            .filter(|element| in_scope(*element) && !in_literal(*element))
        {
            let name = element.value().name();
            let parent = element.parent().and_then(ElementRef::wrap);
            if name == "ol"
                && element.value().attribute("start").is_some()
                && parent.is_some_and(|parent| parent.value().name() == "aside")
                && let Some(number) = element.value().attribute("start").and_then(note_number)
            {
                let items = element
                    .child_elements()
                    .filter(|child| child.value().name() == "li")
                    .collect();
                notes.add(items, vec![format!("num:{number}")], vec![], vec![]);
                continue;
            }
            let context = reach.context(element);
            let id = element.value().attribute("id").unwrap_or("");
            let known_id = ["fn:", "fn-", "fn.", "ftnt", "cite_note-", "footnote-"]
                .iter()
                .any(|prefix| id.starts_with(prefix))
                && !id.contains("ref")
                && !id.ends_with("-link");
            let explicit = matches!(name, "p" | "li" | "div" | "aside")
                && (context || known_id)
                && (name == "li"
                    || (name == "p" && (context || known_id))
                    || has_class(element, "footnote-definition")
                    || has_class(element, "footnote-footer")
                    || element.value().attribute("role") == Some("doc-footnote")
                    || (known_id && !plain(element).is_empty()));
            if !explicit || notes.inside_definition(element) {
                continue;
            }
            let mut aliases = Vec::new();
            let mut markers = Vec::new();
            if !id.is_empty() {
                aliases.push(id.to_owned());
            }
            if let Some((number, marker)) = leading_note_marker(element) {
                aliases.push(format!("num:{number}"));
                markers.push(marker);
            }
            for anchor in inside(element).filter(|node| fragment_target(*node)) {
                if matches!(anchor.value().name(), "a" | "span" | "sup")
                    && (plain(anchor).is_empty() || note_number(&plain(anchor)).is_some())
                    && let Some(id) = anchor
                        .value()
                        .attribute("id")
                        .or_else(|| anchor.value().attribute("name"))
                {
                    aliases.push(id.to_owned());
                    markers.push(anchor);
                }
            }
            if name == "li"
                && let Some(list) = parent
            {
                let index = list
                    .child_elements()
                    .filter(|child| child.value().name() == "li")
                    .position(|child| child == element)
                    .unwrap_or(0);
                let start = list
                    .value()
                    .attribute("start")
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(1);
                if let Some(number) = start.checked_add(index) {
                    aliases.push(format!("num:{number}"));
                }
            }
            if !aliases.is_empty() {
                notes.add(vec![element], aliases, markers, vec![]);
            }
        }

        // Word exports omit target IDs but identify each definition with a
        // generated _ftnref backlink. Keep surrounding prose and external links.
        for paragraph in elements
            .iter()
            .copied()
            .filter(|element| in_scope(*element) && element.value().name() == "p")
        {
            if notes.inside_definition(paragraph) {
                continue;
            }
            if let Some(link) = inside(paragraph).find(|node| link(*node))
                && let Some(number) = link
                    .value()
                    .attribute("href")
                    .and_then(|href| href.rsplit_once('#'))
                    .and_then(|(_, fragment)| fragment.strip_prefix("_ftnref"))
                    .and_then(note_number)
            {
                let marker = link
                    .ancestors()
                    .filter_map(ElementRef::wrap)
                    .take_while(|parent| *parent != paragraph)
                    .find(|parent| parent.value().name() == "sup")
                    .unwrap_or(link);
                notes.add(
                    vec![paragraph],
                    vec![format!("_ftn{number}")],
                    vec![marker],
                    vec![],
                );
            }
        }

        // Generic numeric links need a concentrated definition section, not
        // merely numeric equation/theorem targets scattered through an article.
        let mut generic = Vec::new();
        let mut numeric_references: HashMap<String, Vec<ElementRef<'_>>> = HashMap::new();
        for reference in &references {
            if reference.value().name() != "a" || note_number(&plain(*reference)).is_none() {
                continue;
            }
            let Some(fragment) = note_fragment(*reference, base) else {
                continue;
            };
            numeric_references
                .entry(fragment.clone())
                .or_default()
                .push(*reference);
            let Some(target) = ids
                .get(fragment.as_str())
                .copied()
                .filter(|target| in_scope(*target))
            else {
                continue;
            };
            if notes.inside_definition(target) {
                continue;
            }
            let node = if matches!(target.value().name(), "p" | "li") {
                Some(target)
            } else {
                target
                    .ancestors()
                    .filter_map(ElementRef::wrap)
                    .take_while(|parent| *parent != root)
                    .find(|parent| {
                        matches!(parent.value().name(), "p" | "li")
                            || has_class(*parent, "reference")
                    })
            };
            if let Some(node) = node
                && !inside(node).any(|node| node == *reference && link(node))
                && !generic.iter().any(|(id, _, _)| *id == fragment)
            {
                generic.push((fragment, node, target));
            }
        }
        for container in elements.iter().copied().filter(|element| {
            in_scope(*element)
                && *element != root
                && matches!(
                    element.value().name(),
                    "ol" | "ul" | "div" | "section" | "aside"
                )
        }) {
            let matches = generic
                .iter()
                .filter(|(_, node, _)| contains_element(container, *node))
                .collect::<Vec<_>>();
            if matches.len() < 2 || !generic_note_section(facts, container) {
                continue;
            }
            let external_count = numeric_references
                .values()
                .filter(|references| {
                    references
                        .iter()
                        .any(|reference| !contains_element(container, *reference))
                })
                .count();
            if matches.len() * 4 < external_count * 3 {
                continue;
            }
            // A common wrapper around all article content is not a notes list.
            if references.iter().any(|reference| {
                contains_element(container, *reference)
                    && !matches
                        .iter()
                        .any(|(_, node, _)| contains_element(*node, *reference))
            }) {
                continue;
            }
            for (id, node, target) in matches {
                let mut markers = Vec::new();
                if *node != *target
                    && (plain(*target).is_empty() || note_number(&plain(*target)).is_some())
                {
                    markers.push(*target);
                }
                let mut nodes = vec![*node];
                for sibling in node.next_siblings().filter_map(ElementRef::wrap) {
                    if generic.iter().any(|(_, target, _)| *target == sibling)
                        || !generic_continuation(sibling)
                    {
                        break;
                    }
                    if !is_hidden(sibling) {
                        nodes.push(sibling);
                    }
                }
                let before = notes.definitions.len();
                notes.add(nodes, vec![id.clone()], markers, vec![]);
                if notes.definitions.len() > before
                    && let Some(number) = numeric_references
                        .get(id)
                        .and_then(|links| links.first())
                        .and_then(|link| note_number(&plain(*link)))
                {
                    notes.definitions.last_mut().unwrap().text_marker =
                        numeric_text_marker(*node, &number);
                }
            }
        }

        // Loose numbered paragraphs require corroborating body references.
        // A section delimiter admits continuation blocks; otherwise only the
        // numbered paragraphs move, leaving updates and unrelated prose intact.
        let paragraphs = elements
            .iter()
            .copied()
            .filter(|element| {
                in_scope(*element)
                    && element.value().name() == "p"
                    && !notes.inside_definition(*element)
            })
            .filter_map(|element| {
                leading_note_marker(element).map(|(number, marker)| (element, number, marker))
            })
            .collect::<Vec<_>>();
        let matched = paragraphs
            .iter()
            .filter(|(_, number, _)| {
                references.iter().any(|reference| {
                    reference.value().name() == "sup"
                        && plain(*reference) == *number
                        && !paragraphs
                            .iter()
                            .any(|(paragraph, _, _)| contains_element(*paragraph, *reference))
                        && !notes.inside_definition(*reference)
                })
            })
            .count();
        if matched >= 2 {
            for (paragraph, number, marker) in &paragraphs {
                let siblings = paragraph
                    .parent()
                    .map(|parent| {
                        parent
                            .children()
                            .filter_map(ElementRef::wrap)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let position = siblings
                    .iter()
                    .position(|element| element == paragraph)
                    .unwrap_or(0);
                let boundary = siblings[..position]
                    .iter()
                    .rposition(|element| element.value().name() == "hr" || note_heading(*element));
                let mut nodes = vec![*paragraph];
                if boundary.is_some() {
                    for sibling in &siblings[position + 1..] {
                        if paragraphs.iter().any(|(node, _, _)| node == sibling)
                            || !matches!(sibling.value().name(), "p" | "ul" | "ol" | "blockquote")
                            || first_content(*sibling).is_some_and(|first| {
                                matches!(first.value().name(), "b" | "strong")
                                    && leading_note_marker(*sibling).is_none()
                            })
                        {
                            break;
                        }
                        if !is_hidden(*sibling) {
                            nodes.push(*sibling);
                        }
                    }
                }
                notes.add(nodes, vec![format!("num:{number}")], vec![*marker], vec![]);
            }
        }

        let order = elements
            .iter()
            .enumerate()
            .map(|(index, element)| (element_key(*element), index))
            .collect::<HashMap<_, _>>();
        notes.definitions.sort_by_key(|note| {
            order
                .get(&element_key(note.nodes[0]))
                .copied()
                .unwrap_or(usize::MAX)
        });
        let mut aliases = HashMap::new();
        for (index, note) in notes.definitions.iter().enumerate() {
            for alias in &note.aliases {
                aliases.entry(alias.clone()).or_insert(index);
            }
        }
        for reference in references {
            if notes.inside_definition(reference) || in_literal(reference) {
                continue;
            }
            let alias = if reference.value().name() == "a" {
                note_fragment(reference, base)
            } else if reference.value().name() == "sup"
                && !inside(reference).any(|node| node.value().name() == "a")
            {
                note_number(&plain(reference)).map(|number| format!("num:{number}"))
            } else {
                None
            };
            if let Some(index) = alias.as_ref().and_then(|alias| aliases.get(alias)).copied() {
                let mut replacement = reference;
                for parent in reference.ancestors().filter_map(ElementRef::wrap) {
                    if !matches!(parent.value().name(), "sup" | "span")
                        || plain(parent) != plain(reference)
                    {
                        break;
                    }
                    replacement = parent;
                }
                notes.definitions[index].references.push(replacement);
            }
        }
        notes.definitions.retain(|note| !note.references.is_empty());
        let reference_ids = notes
            .definitions
            .iter()
            .flat_map(|note| note.references.iter())
            .flat_map(|reference| {
                std::iter::once(*reference)
                    .chain(reference.descendants().filter_map(ElementRef::wrap))
            })
            .filter_map(|element| element.value().attribute("id").map(str::to_owned))
            .collect::<HashSet<_>>();
        for note in &mut notes.definitions {
            for root in &note.nodes {
                for element in root.descendants().filter_map(ElementRef::wrap) {
                    let explicit = element.value().attribute("role") == Some("doc-backlink")
                        || element.value().attribute("data-footnote-backref").is_some()
                        || element.value().class_names().any(|class| {
                            matches!(
                                class,
                                "footnote-backref"
                                    | "data-footnote-backref"
                                    | "mw-cite-backlink"
                                    | "easy-footnote-to-top"
                                    | "footnote-definition-label"
                            )
                        });
                    let link_back = element.value().name() == "a"
                        && element
                            .value()
                            .attribute("href")
                            .and_then(|href| href.strip_prefix('#'))
                            .is_some_and(|fragment| {
                                reference_ids.contains(fragment)
                                    || (!plain(element).is_empty()
                                        && plain(element)
                                            .chars()
                                            .all(|ch| "^↩↥↑↵⤴⤵⏎\u{fe0e}\u{fe0f}".contains(ch)))
                            });
                    if explicit || link_back {
                        note.markers.push(element);
                    }
                }
            }
        }
        notes.definitions.retain(|note| {
            note.nodes.iter().enumerate().any(|(index, node)| {
                note_has_content(
                    *node,
                    &note.markers,
                    if index == 0 {
                        note.text_marker.as_deref()
                    } else {
                        None
                    },
                    0,
                )
            })
        });
        for (index, note) in notes.definitions.iter().enumerate() {
            if let Some(prefix) = &note.text_marker {
                notes
                    .text_markers
                    .insert(element_key(note.nodes[0]), prefix.clone());
            }
            for reference in &note.references {
                notes.references.insert(element_key(*reference), index + 1);
            }
            for node in &note.nodes {
                notes.removed.insert(element_key(*node));
            }
            for marker in &note.markers {
                notes.markers.insert(element_key(*marker));
            }
        }
        // Delete separators only when the following structure consists solely
        // of definitions. Notes headings that introduce additional prose remain.
        for element in &elements {
            if !(note_heading(*element) || element.value().name() == "hr") {
                continue;
            }
            let mut carrier = *element;
            while let Some(parent) = carrier.parent().and_then(ElementRef::wrap)
                && parent != root
                && parent.child_elements().count() == 1
                && plain(parent) == plain(carrier)
            {
                carrier = parent;
            }
            if let Some(next) = carrier.next_siblings().filter_map(ElementRef::wrap).next()
                && notes.definition_shell(next, 0)
            {
                notes.removed.insert(element_key(*element));
            }
        }
        // A duplicated sidenote is dropped only when its text matches the
        // adjacent resolved definition; unrelated marginal text stays visible.
        for sidenote in inside(root).filter(|node| span_of(*node, &["sidenote"])) {
            if notes.inside_definition(sidenote) {
                continue;
            }
            if let Some(previous) = sidenote.prev_siblings().filter_map(ElementRef::wrap).next()
                && let Some(number) = notes.references.get(&element_key(previous))
            {
                let content = plain(sidenote);
                let content = content.trim_start_matches(|ch: char| {
                    ch.is_ascii_digit() || ch == '.' || ch.is_whitespace()
                });
                let definition = notes.definitions[*number - 1]
                    .nodes
                    .iter()
                    .map(|node| plain(*node))
                    .collect::<Vec<_>>()
                    .join(" ");
                let definition = definition.trim_start_matches(|ch: char| {
                    ch.is_ascii_digit() || ch == '.' || ch.is_whitespace()
                });
                if content == definition {
                    notes.removed.insert(element_key(sidenote));
                }
            }
        }
        notes
    }

    fn definition_shell(&self, element: ElementRef<'_>, depth: usize) -> bool {
        if depth > MAX_DEPTH {
            return false;
        }
        if self.removed.contains(&element_key(element)) {
            return true;
        }
        let mut has_definition = false;
        for node in element.children() {
            if let Some(child) = ElementRef::wrap(node) {
                if note_heading(child) || child.value().name() == "hr" {
                    continue;
                }
                if !self.definition_shell(child, depth + 1) {
                    return false;
                }
                has_definition = true;
            } else if let scraper::Node::Text(text) = node.value()
                && !text.trim().is_empty()
            {
                return false;
            }
        }
        has_definition
    }
}

const MAX_DEPTH: usize = 256;

thread_local! {
    /// Set when a rendering on this thread kept a subtree below [`MAX_DEPTH`]
    /// as plain text. A rendering is one synchronous recursion on one thread.
    static FLATTENED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The text below an element without recursion, a space at block edges so
/// the words of neighbouring blocks stay apart.
fn flat_text(element: ElementRef<'_>) -> String {
    let block = |element: Option<ElementRef<'_>>| {
        element.is_some_and(|element| block_tag(element.value().name()))
    };
    let mut text = String::new();
    for node in element.descendants() {
        if (block(ElementRef::wrap(node)) || block(node.prev_sibling().and_then(ElementRef::wrap)))
            && !text.ends_with(' ')
        {
            text.push(' ');
        }
        if let scraper::Node::Text(value) = node.value() {
            text.push_str(value);
        }
    }
    text
}

fn serialize_clean(
    element: ElementRef<'_>,
    base: Option<&Url>,
    output: &mut Out,
    depth: usize,
    notes: &Footnotes<'_>,
    definition: bool,
) -> Result<()> {
    // As in the reference, callouts are canonicalized before hidden and chrome
    // removal: a collapsed body is content; hidden elements inside it are not.
    let callout = callouts::detect(element);
    if callout.is_none()
        && (notes.facts.hidden(element)
            || (notes.prune_chrome && article::excluded(notes.facts, element)))
        && !(definition && depth == 0)
    {
        return Ok(());
    }
    if callouts::replaced_title(element) {
        return Ok(());
    }
    // A spacer, tracking pixel or icon carries no content on a page; an
    // emoji drawn as an image keeps its character.
    if notes.prune_chrome
        && let Some(kept) = small_image(element)
    {
        escaped_text(&kept, output);
        return Ok(());
    }
    let key = element_key(element);
    // A note's definition is written at the end, wherever it stood.
    if notes.left_out(element) && !(definition && depth == 0) {
        return Ok(());
    }
    if !definition && let Some(number) = notes.references.get(&key) {
        let visible_text = element
            .descendants()
            .filter_map(|node| {
                let scraper::Node::Text(text) = node.value() else {
                    return None;
                };
                (!node
                    .ancestors()
                    .filter_map(ElementRef::wrap)
                    .take_while(|parent| *parent != element)
                    .any(|parent| {
                        notes.removed.contains(&element_key(parent))
                            || !visible_reference(notes.facts, parent, notes.prune_chrome)
                    }))
                .then_some(&**text)
            })
            .collect::<String>();
        let start = visible_text.len() - visible_text.trim_start().len();
        let end = visible_text.trim_end().len();
        escaped_text(&visible_text[..start], output);
        output.start("markitai-footnote");
        output.attribute("data-number", &number.to_string());
        output.end("markitai-footnote");
        escaped_text(&visible_text[end..], output);
        return Ok(());
    }
    if (notes.removed.contains(&key) && !(definition && depth == 0))
        || (definition && notes.markers.contains(&key))
    {
        return Ok(());
    }
    if depth > MAX_DEPTH {
        // Unclosed legacy tags nest this deep; the rest keeps its words.
        escaped_text(&flat_text(element), output);
        FLATTENED.with(|flattened| flattened.set(true));
        return Ok(());
    }
    if let Some(callout) = callout {
        output.start("blockquote");
        output.start("p");
        output.start("markitai-callout");
        output.attribute(
            "data-marker",
            &format!("[!{}]{}", callout.kind, callout.fold),
        );
        output.end("markitai-callout");
        output.text(" ");
        escaped_text(&callouts::marker_title(&callout), output);
        output.end("p");
        let content_name = callout.content.value().name();
        serialize_children(
            callout.content,
            content_name,
            false,
            None,
            base,
            output,
            depth,
            notes,
            definition,
        )?;
        output.end("blockquote");
        return Ok(());
    }
    if code::render(element, output, depth)? {
        return Ok(());
    }
    let value = element.value();
    let name = value.name();
    // A label that only repeats the language of the code block after it.
    if matches!(name, "div" | "span") && code::repeats_language(element) {
        return Ok(());
    }
    if duplicate_math_preview(element) {
        return Ok(());
    }
    let in_code = element
        .ancestors()
        .filter_map(ElementRef::wrap)
        .any(|ancestor| matches!(ancestor.value().name(), "pre" | "code"));
    if !in_code && let Some((latex, block)) = math_expression(element)? {
        emit_math(&latex, block, output);
        return Ok(());
    }
    if matches!(
        name,
        "script"
            | "style"
            | "nav"
            | "footer"
            | "form"
            | "button"
            | "input"
            | "select"
            | "textarea"
            | "iframe"
            | "object"
            | "embed"
            | "head"
            | "template"
            | "noscript"
    ) {
        return Ok(());
    }
    // A link that shows nothing (an icon drawn by a style sheet, a vote
    // arrow) would be written as `[](url)`.
    if name == "a"
        && value.attribute("href").is_some()
        && shows_nothing(element, notes.prune_chrome)
    {
        return Ok(());
    }
    if notes.prune_chrome {
        // A list item that shows nothing (an icon, a share button) is an
        // empty bullet.
        if name == "li" && shows_nothing_kept(notes.facts, element) {
            return Ok(());
        }
        // A heading's link to its own page (a permalink around its text or an
        // anchor glyph beside it) would not resolve in Markdown.
        if name == "a" && heading_permalink(element) {
            return if plain(element).chars().any(char::is_alphanumeric) {
                serialize_children(
                    element, name, false, None, base, output, depth, notes, definition,
                )
            } else {
                Ok(())
            };
        }
    }
    if name == "table" && serialize_table(element, base, output, depth, notes, definition)? {
        return Ok(());
    }
    // Canonicalize lazy images without retaining arbitrary event or style
    // attributes; the largest `srcset` candidate stands for the image where
    // `src` names a thumbnail.
    let src = if name == "img" {
        srcset::image(value)
    } else {
        None
    };
    let preformatted = in_code || matches!(name, "pre" | "code");
    let styled_code = name == "code"
        && value.attribute("style").is_some_and(|style| {
            style
                .to_ascii_lowercase()
                .replace(char::is_whitespace, "")
                .contains("white-space:pre")
        })
        && !element
            .ancestors()
            .filter_map(ElementRef::wrap)
            .any(|parent| parent.value().name() == "pre");
    if styled_code {
        output.start("pre");
    }
    if notes.block_links.contains(&key) {
        return serialize_children(
            element,
            name,
            preformatted,
            None,
            base,
            output,
            depth,
            notes,
            definition,
        );
    }
    let serialized_name = if definition && depth == 0 && name == "li" {
        "div"
    } else {
        name
    };
    output.start(serialized_name);
    for attribute in [
        "href", "src", "alt", "title", "colspan", "rowspan", "start", "class", "id",
    ] {
        let raw = if attribute == "src" && name == "img" {
            src
        } else {
            value.attribute(attribute)
        };
        if let Some(raw) = raw {
            let normalized = if attribute == "src" && name == "img" {
                image_source(raw, base)
            } else if attribute == "href" && base.is_none() {
                saved_page_link(raw, notes.link_base.as_ref())
            } else if matches!(attribute, "href" | "src") {
                safe_url(raw, base)
            } else {
                Some(raw.to_owned())
            };
            if let Some(normalized) = normalized {
                output.attribute(attribute, &normalized);
            }
        }
    }
    let text_marker = if definition && depth == 0 {
        notes.text_markers.get(&key).map(String::as_str)
    } else {
        None
    };
    let link = notes
        .link_targets
        .get(&key)
        .and_then(|anchor| anchor.value().attribute("href"))
        .and_then(|href| safe_url(href, base));
    if let Some(link) = &link {
        output.start("a");
        output.attribute("href", link);
    }
    serialize_children(
        element,
        name,
        preformatted,
        text_marker,
        base,
        output,
        depth,
        notes,
        definition,
    )?;
    if link.is_some() {
        output.end("a");
    }
    if styled_code {
        output.end("pre");
    }
    if !matches!(
        name,
        "area" | "base" | "br" | "col" | "hr" | "img" | "link" | "meta" | "source" | "wbr"
    ) {
        output.end(serialized_name);
    }
    if !in_code && !block_tag(name) && screen_reader_label(element) {
        label_gap(element, output);
    }
    Ok(())
}

/// Whether an element is text drawn for screen readers only (`visually-hidden`,
/// `sr-only`, `screen-reader-text`, a CSS module's `…-VisuallyHidden`): the
/// label of what follows (`Published`, `By`), kept as the page's text.
fn screen_reader_label(element: ElementRef<'_>) -> bool {
    let Some(classes) = element.value().attribute("class") else {
        return false;
    };
    classes.split_ascii_whitespace().any(|class| {
        [
            "visually-hidden",
            "visuallyhidden",
            "sr-only",
            "screen-reader-text",
            "screen-reader-only",
        ]
        .iter()
        .any(|label| class.eq_ignore_ascii_case(label))
            || (class.len() >= "visuallyhidden".len()
                && class
                    .as_bytes()
                    .windows("visuallyhidden".len())
                    .any(|window| window.eq_ignore_ascii_case(b"visuallyhidden")))
    })
}

/// The space between a screen reader label and the text after it. Markup
/// writes the two without one (`<span class="visually-hidden">Published</span>
/// <span>1 hour ago</span>` with no text between), since a style sheet sets
/// them apart; written as is they would read `Published1 hour ago`.
fn label_gap(label: ElementRef<'_>, output: &mut Out) {
    let next = label
        .next_siblings()
        .find(|node| !matches!(node.value(), scraper::Node::Comment(_)));
    let joined = match next.map(|node| node.value()) {
        Some(scraper::Node::Text(text)) => {
            !text.is_empty() && !text.starts_with(char::is_whitespace)
        }
        Some(scraper::Node::Element(next)) => !block_tag(next.name()),
        _ => false,
    };
    if joined && label.text().any(|text| !text.trim().is_empty()) {
        output.text(" ");
    }
}

/// A link inside a heading to a fragment of the same page.
fn heading_permalink(link: ElementRef<'_>) -> bool {
    link.value()
        .attribute("href")
        .is_some_and(|href| href.trim().starts_with('#'))
        && link
            .ancestors()
            .filter_map(ElementRef::wrap)
            .take(3)
            .any(|parent| {
                matches!(
                    parent.value().name(),
                    "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
                )
            })
}

/// A table's caption and its own rows (not those of nested tables), each with
/// whether it is a `thead` row; hidden rows are left out.
fn table_rows(table: ElementRef<'_>) -> (Option<ElementRef<'_>>, Vec<(ElementRef<'_>, bool)>) {
    let mut caption = None;
    let mut rows = Vec::new();
    for child in table.child_elements() {
        match child.value().name() {
            "caption" if caption.is_none() => caption = Some(child),
            "tr" => rows.push((child, false)),
            section @ ("thead" | "tbody" | "tfoot") => rows.extend(
                child
                    .child_elements()
                    .filter(|row| row.value().name() == "tr")
                    .map(|row| (row, section == "thead")),
            ),
            _ => {}
        }
    }
    rows.retain(|(row, _)| !is_hidden(*row));
    (caption, rows)
}

fn table_cells(row: ElementRef<'_>) -> impl Iterator<Item = ElementRef<'_>> {
    row.child_elements()
        .filter(|cell| matches!(cell.value().name(), "td" | "th"))
}

fn table_span(cell: ElementRef<'_>, attribute: &str) -> usize {
    const MAX_SPAN: usize = 64;
    cell.value()
        .attribute(attribute)
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .map_or(1, |value| value.min(MAX_SPAN))
}

/// A list item of a full page with no text and no image once the blocks that
/// page conversion drops (buttons, forms, hidden or navigation content, an
/// icon) are gone.
fn shows_nothing_kept(facts: &Facts, item: ElementRef<'_>) -> bool {
    let mut stack = vec![item];
    while let Some(element) = stack.pop() {
        if element != item && article::discarded(facts, element) {
            continue;
        }
        if element.value().name() == "img" && small_image(element).is_none() {
            return false;
        }
        for child in element.children() {
            match child.value() {
                scraper::Node::Text(text) if !text.trim().is_empty() => return false,
                scraper::Node::Element(_) => stack.extend(ElementRef::wrap(child)),
                _ => {}
            }
        }
    }
    true
}

/// An element that shows nothing: no text and no image that is kept.
fn shows_nothing(element: ElementRef<'_>, prune_chrome: bool) -> bool {
    is_hidden(element)
        || (element.text().all(|text| text.trim().is_empty())
            && !element
                .descendants()
                .filter_map(ElementRef::wrap)
                .any(|node| {
                    node.value().name() == "img" && !(prune_chrome && small_image(node).is_some())
                }))
}

/// Whether a table is written as a Markdown table: it has header cells, or it
/// reads as data (at least two rows and two columns of short cells, mostly
/// filled, without nested tables or block structure). Other tables lay out a
/// page and are written as their cells' content.
fn table_grid(table: ElementRef<'_>, rows: &[(ElementRef<'_>, bool)], prune_chrome: bool) -> bool {
    const MAX_CELL_WORDS: usize = 40;
    if rows
        .iter()
        .any(|(row, head)| *head || table_cells(*row).any(|cell| cell.value().name() == "th"))
    {
        return true;
    }
    if table.value().attribute("role").is_some_and(|role| {
        let role = role.trim();
        role.eq_ignore_ascii_case("presentation") || role.eq_ignore_ascii_case("none")
    }) {
        return false;
    }
    let (mut filled_rows, mut width, mut cells, mut empty) = (0, 0, 0, 0);
    for (row, _) in rows {
        let (mut row_width, mut filled) = (0, false);
        for cell in table_cells(*row) {
            row_width += table_span(cell, "colspan");
            cells += 1;
            if shows_nothing(cell, prune_chrome) {
                empty += 1;
                continue;
            }
            filled = true;
            let mut paragraphs = 0;
            for node in cell.descendants().filter_map(ElementRef::wrap) {
                match node.value().name() {
                    "table" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "blockquote" | "ul"
                    | "ol" | "hr" => return false,
                    "p" => paragraphs += 1,
                    _ => {}
                }
            }
            if paragraphs > 1
                || cell
                    .text()
                    .flat_map(str::split_whitespace)
                    .nth(MAX_CELL_WORDS)
                    .is_some()
            {
                return false;
            }
        }
        width = width.max(row_width);
        filled_rows += usize::from(filled);
    }
    filled_rows >= 2 && width >= 2 && empty * 3 < cells * 2
}

/// Writes a table for htmd, which keeps only the first row's `th` cells as the
/// header and only `td` cells after it (row headers, a header row's data cells
/// and spanned positions would be dropped or shift the columns), and flattens
/// every table without header cells, including data tables and the data
/// tables inside a layout table.
///
/// A table written as a Markdown table (see [`table_grid`]) becomes a regular
/// grid: one header row (the first `thead` row, a first row of only `th`
/// cells, empty cells when only rows have header cells, or a data table's
/// first row) and body rows of `td` cells; each span's other positions are
/// empty cells, and columns empty in every row are left out. A layout table
/// becomes its cells' content as blocks. htmd writes the text of a table
/// inside a Markdown table's cell into that cell.
fn serialize_table(
    table: ElementRef<'_>,
    base: Option<&Url>,
    output: &mut Out,
    depth: usize,
    notes: &Footnotes<'_>,
    definition: bool,
) -> Result<bool> {
    const MAX_COLUMNS: usize = 256;
    let (caption, mut rows) = table_rows(table);
    if rows.is_empty() {
        return Ok(false);
    }
    let children = |element: ElementRef<'_>, output: &mut Out| {
        let name = element.value().name();
        serialize_children(
            element,
            name,
            false,
            None,
            base,
            output,
            depth + 2,
            notes,
            definition,
        )
    };
    if let Some(caption) = caption.filter(|caption| !is_hidden(*caption)) {
        output.start("p");
        children(caption, output)?;
        output.end("p");
    }
    if !table_grid(table, &rows, notes.prune_chrome) {
        for (row, _) in rows.iter().filter(|(row, _)| !notes.left_out(*row)) {
            for cell in table_cells(*row).filter(|cell| !is_hidden(*cell) && !notes.left_out(*cell))
            {
                output.start("div");
                children(cell, output)?;
                output.end("div");
            }
        }
        return Ok(true);
    }
    let headed = rows
        .iter()
        .any(|(row, head)| *head || table_cells(*row).any(|cell| cell.value().name() == "th"));
    let header = if headed {
        rows.iter().position(|(_, head)| *head).or_else(|| {
            rows.first()
                .is_some_and(|(row, _)| table_cells(*row).all(|cell| cell.value().name() == "th"))
                .then_some(0)
        })
    } else {
        rows.iter().position(|(row, _)| {
            !table_cells(*row).all(|cell| shows_nothing(cell, notes.prune_chrome))
        })
    };
    if !headed && let Some(index) = header {
        rows.drain(..index);
    } else if let Some(index) = header {
        let row = rows.remove(index);
        rows.insert(0, row);
    }
    // `carry[column]`: rows below the current one still covered by a rowspan.
    let mut carry: Vec<usize> = Vec::new();
    let mut grid = Vec::with_capacity(rows.len());
    for (row, _) in &rows {
        let covered: Vec<bool> = carry.iter().map(|rows| *rows > 0).collect();
        for rows in &mut carry {
            *rows = rows.saturating_sub(1);
        }
        let mut line: Vec<Option<ElementRef<'_>>> = Vec::new();
        for cell in table_cells(*row) {
            while covered.get(line.len()).copied().unwrap_or(false) {
                line.push(None);
            }
            let start = line.len();
            line.push(Some(cell));
            line.extend(std::iter::repeat_n(None, table_span(cell, "colspan") - 1));
            if line.len() > MAX_COLUMNS {
                return Ok(false);
            }
            if carry.len() < line.len() {
                carry.resize(line.len(), 0);
            }
            let below = table_span(cell, "rowspan") - 1;
            for rows in &mut carry[start..line.len()] {
                *rows = below;
            }
        }
        grid.push(line);
    }
    // Spacer columns show nothing in any row; an empty column would also
    // give htmd a separator cell without dashes.
    let width = grid.iter().map(Vec::len).max().unwrap_or(0);
    let columns: Vec<usize> = (0..width)
        .filter(|column| {
            grid.iter().any(|line| {
                line.get(*column)
                    .copied()
                    .flatten()
                    .is_some_and(|cell| !shows_nothing(cell, notes.prune_chrome))
            })
        })
        .collect();
    if columns.is_empty() {
        return Ok(false);
    }
    let cell = |tag: &str, cell: Option<ElementRef<'_>>, output: &mut Out| -> Result<()> {
        output.start(tag);
        if let Some(cell) = cell.filter(|cell| !is_hidden(*cell)) {
            children(cell, output)?;
        }
        output.end(tag);
        Ok(())
    };
    output.start("table");
    output.start("thead");
    output.start("tr");
    let body = if header.is_some() {
        for column in &columns {
            cell("th", grid[0].get(*column).copied().flatten(), output)?;
        }
        &grid[1..]
    } else {
        for _ in &columns {
            cell("th", None, output)?;
        }
        &grid[..]
    };
    output.end("tr");
    output.end("thead");
    output.start("tbody");
    for line in body.iter().filter(|line| {
        !line
            .iter()
            .flatten()
            .all(|cell| shows_nothing(*cell, notes.prune_chrome))
    }) {
        output.start("tr");
        for column in &columns {
            cell("td", line.get(*column).copied().flatten(), output)?;
        }
        output.end("tr");
    }
    output.end("tbody");
    output.end("table");
    Ok(true)
}

/// An element's children as sanitized HTML; `name` decides whether text next
/// to block boundaries is trimmed.
#[allow(clippy::too_many_arguments)]
fn serialize_children(
    element: ElementRef<'_>,
    name: &str,
    preformatted: bool,
    mut text_marker: Option<&str>,
    base: Option<&Url>,
    output: &mut Out,
    depth: usize,
    notes: &Footnotes<'_>,
    definition: bool,
) -> Result<()> {
    for child in element.children() {
        if let Some(child) = ElementRef::wrap(child) {
            serialize_clean(child, base, output, depth + 1, notes, definition)?;
        } else if let scraper::Node::Text(text) = child.value() {
            let text = if let Some(prefix) = text_marker
                && let Some(rest) = text.strip_prefix(prefix)
            {
                text_marker = None;
                rest
            } else {
                text.as_ref()
            };
            if preformatted {
                output.text(text);
            } else {
                let previous = child
                    .prev_siblings()
                    .find(|node| !matches!(node.value(), scraper::Node::Comment(_)));
                let next = child
                    .next_siblings()
                    .find(|node| !matches!(node.value(), scraper::Node::Comment(_)));
                let trim_start = previous.map_or(block_tag(name), |node| {
                    ElementRef::wrap(node).is_some_and(|element| block_tag(element.value().name()))
                });
                let trim_end = next.map_or(block_tag(name), |node| {
                    ElementRef::wrap(node).is_some_and(|element| block_tag(element.value().name()))
                });
                let text = if trim_start { text.trim_start() } else { text };
                let text = if trim_end { text.trim_end() } else { text };
                prose(text, element, output);
            }
        }
    }
    Ok(())
}

/// Text of a page's prose: raw TeX delimiters (`$x$`, `\(x\)`, `$$x$$`,
/// `\[x\]`, see [`raw_math`]) are math, written as a recognized wrapper is so
/// that the Markdown escaping leaves their backslashes, brackets and
/// underscores alone; the rest is escaped text. `parent` is the element that
/// holds the text. A display expression (`$$x$$`, `\[x\]`) is a block of its
/// own when it is all its element holds; inside a sentence it is inline math,
/// as the reference spells it. A table cell cannot hold a paragraph break or a
/// line break, so math in one stays inline and on one line.
fn prose(text: &str, parent: ElementRef<'_>, output: &mut Out) {
    let Some(pieces) = raw_math::split(text) else {
        escaped_text(text, output);
        return;
    };
    let in_cell = std::iter::once(parent)
        .chain(parent.ancestors().filter_map(ElementRef::wrap))
        .any(|element| matches!(element.value().name(), "td" | "th"));
    // One expression and nothing else in the text, in an element with no
    // other content.
    let (mut expressions, mut other_text) = (0, false);
    for piece in &pieces {
        match piece {
            raw_math::Piece::Math { .. } => expressions += 1,
            raw_math::Piece::Text(text) => other_text |= !text.trim().is_empty(),
        }
    }
    let alone = expressions == 1
        && !other_text
        && parent
            .children()
            .filter(|node| match node.value() {
                scraper::Node::Text(text) => !text.trim().is_empty(),
                scraper::Node::Comment(_) => false,
                _ => true,
            })
            .count()
            == 1;
    for piece in pieces {
        match piece {
            // The blanks around an expression that is a block of its own.
            raw_math::Piece::Text(_) if alone => {}
            raw_math::Piece::Text(text) => escaped_text(text, output),
            raw_math::Piece::Math { latex, .. } if in_cell => {
                emit_math(&latex.replace(['\n', '\r'], " "), false, output);
            }
            raw_math::Piece::Math { latex, display } => emit_math(latex, display && alone, output),
        }
    }
}

fn render_clean(root: ElementRef<'_>, base: Option<&Url>) -> Result<String> {
    render_with_footnotes(root, root, base, false, &Facts::new(root))
}

fn render_with_footnotes<'a>(
    root: ElementRef<'a>,
    document: ElementRef<'a>,
    base: Option<&Url>,
    prune_chrome: bool,
    facts: &Facts,
) -> Result<String> {
    let notes = Footnotes::collect(root, document, base, prune_chrome, facts);
    let mut markdown =
        render_cleaned(&|output| serialize_clean(root, base, output, 0, &notes, false))?;
    for (index, note) in notes.definitions.iter().enumerate() {
        let content = render_cleaned(&|output| {
            for node in &note.nodes {
                serialize_clean(*node, base, output, 0, &notes, true)?;
            }
            Ok(())
        })?;
        if content.is_empty() {
            continue;
        }
        markdown.push_str(&format!("\n\n[^{}]: ", index + 1));
        let mut lines = content.lines();
        markdown.push_str(lines.next().unwrap_or_default());
        // Continuations remain inside the Markdown definition, including nested
        // lists, quotes and code fences; no source prose is flattened away.
        for line in lines {
            markdown.push('\n');
            if !line.is_empty() {
                markdown.push_str("    ");
                markdown.push_str(line);
            }
        }
    }
    if prune_chrome {
        markdown = drop_empty_sections(&markdown);
    }
    Ok(markdown.trim().to_owned())
}

/// A heading of a full page with nothing under it: no content before the next
/// heading of its level or a higher one, or before the end. Hidden or removed
/// blocks leave such orphans; a heading over deeper headings is not one, unless
/// they are orphans too. The page's first heading, when it is a title, stays
/// even over a section of its own level. Fenced code is never read as headings.
fn drop_empty_sections(markdown: &str) -> String {
    #[derive(Clone, Copy, PartialEq)]
    enum Line {
        Blank,
        /// A heading's level, and whether it labels notes or sources, whose
        /// entries may have moved to the end as footnotes.
        Heading(usize, bool),
        Content,
    }
    let mut lines: Vec<&str> = markdown.split('\n').collect();
    loop {
        let mut fence: Option<(char, usize)> = None;
        let kinds: Vec<Line> = lines
            .iter()
            .map(|line| {
                let trimmed = line.trim_start_matches(' ');
                let marker = trimmed.chars().next().filter(|ch| matches!(ch, '`' | '~'));
                let run = marker.map_or(0, |marker| {
                    trimmed.chars().take_while(|ch| *ch == marker).count()
                });
                if let Some((open, length)) = fence {
                    if marker == Some(open)
                        && run >= length
                        && line.len() - trimmed.len() < 4
                        && trimmed[run..].trim().is_empty()
                    {
                        fence = None;
                    }
                    return Line::Content;
                }
                if run >= 3 && line.len() - trimmed.len() < 4 {
                    fence = marker.map(|marker| (marker, run));
                    return Line::Content;
                }
                let level = line.chars().take_while(|ch| *ch == '#').count();
                if line.trim().is_empty() {
                    Line::Blank
                } else if (1..=6).contains(&level) && line[level..].starts_with(' ') {
                    Line::Heading(level, furniture::scholarly_label(&line[level..]))
                } else {
                    Line::Content
                }
            })
            .collect();
        let title = kinds
            .iter()
            .position(|kind| matches!(kind, Line::Heading(..)));
        let empty: Vec<usize> = (0..lines.len())
            .filter(|&index| {
                let Line::Heading(level, false) = kinds[index] else {
                    return false;
                };
                if level == 1 && Some(index) == title {
                    return false;
                }
                match kinds[index + 1..].iter().find(|kind| **kind != Line::Blank) {
                    None => true,
                    Some(Line::Heading(next, _)) => *next <= level,
                    Some(_) => false,
                }
            })
            .collect();
        if empty.is_empty() {
            return lines.join("\n");
        }
        let mut kept = Vec::with_capacity(lines.len());
        let mut index = 0;
        while index < lines.len() {
            if empty.contains(&index) {
                index += 1 + usize::from(kinds.get(index + 1) == Some(&Line::Blank));
            } else {
                kept.push(lines[index]);
                index += 1;
            }
        }
        lines = kept;
    }
}

/// Empty quoted lines as `>` (per nesting level), as the reference writes them.
fn tight_blockquote(markdown: &str) -> String {
    markdown
        .split('\n')
        .map(|line| {
            if !line.is_empty()
                && line.chars().all(|ch| ch == '>' || ch == ' ')
                && line.contains('>')
            {
                line.trim_end()
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn converter() -> &'static htmd::HtmlToMarkdown {
    // The converter keeps no state between documents, so its handler table is
    // built once per process instead of once per call.
    static CONVERTER: std::sync::OnceLock<htmd::HtmlToMarkdown> = std::sync::OnceLock::new();
    CONVERTER.get_or_init(sanitized_converter)
}

/// The Markdown of what `write` writes: htmd converts the tree built as it is
/// written, and parses the markup only for a page the tree writer cannot build
/// (raw-text elements such as a `title` in the body), writing it again.
fn render_cleaned(write: &dyn Fn(&mut Out) -> Result<()>) -> Result<String> {
    let mut output = Out::tree();
    write(&mut output)?;
    let document = output.tree.take().and_then(htmd::TreeWriter::finish);
    // Every conversion a test makes also parses the markup and requires the
    // same tree.
    #[cfg(test)]
    {
        let mut markup = Out::markup();
        write(&mut markup)?;
        let markup = markup.into_markup();
        match &document {
            Some(document) => assert!(
                tests::same_tree(
                    document.tree.root(),
                    Html::parse_document(&markup).tree.root()
                ),
                "written and parsed trees differ for {markup:?}"
            ),
            None => tests::markup_parsed(),
        }
    }
    if let Some(document) = document {
        return Ok(converter()
            .tree_to_markdown(document.tree.root())
            .trim()
            .to_owned());
    }
    let mut output = Out::markup();
    write(&mut output)?;
    render_sanitized(&output.into_markup())
}

fn render_sanitized(cleaned: &str) -> Result<String> {
    converter()
        .convert(cleaned)
        .map(|markdown| markdown.trim().to_owned())
        .map_err(|error| Error::Conversion(format!("HTML rendering failed: {error}")))
}

fn sanitized_converter() -> htmd::HtmlToMarkdown {
    htmd::HtmlToMarkdown::builder()
        // The reference's list and rule spelling: one space after a list
        // marker (`* item`, `1. item`) and `---` rules.
        .options(htmd::options::Options {
            ul_bullet_spacing: 1,
            ol_number_spacing: 1,
            ..Default::default()
        })
        .skip_tags(vec!["script", "style", "head"])
        .add_handler(
            vec!["markitai-footnote"],
            |_: &dyn htmd::element_handler::Handlers, element: htmd::Element| {
                let number = element.attr("data-number")?.parse::<usize>().ok()?;
                Some(format!("[^{number}]").into())
            },
        )
        .add_handler(
            vec!["markitai-callout"],
            |_: &dyn htmd::element_handler::Handlers, element: htmd::Element| {
                // Generated marker text (`[!type]fold`), never page markup.
                let marker = element.attr("data-marker")?;
                Some(marker.to_owned().into())
            },
        )
        .add_handler(
            vec!["markitai-math"],
            |_: &dyn htmd::element_handler::Handlers, element: htmd::Element| {
                let latex = element.attr("data-latex")?;
                // Emit TeX syntax as data, never HTML supplied in an annotation.
                let latex = latex.replace('<', r"\lt ").replace('>', r"\gt ");
                let block = element.attr("data-display") == Some("block");
                Some(
                    if block {
                        format!("\n\n$${latex}$$\n\n")
                    } else {
                        format!("${latex}$")
                    }
                    .into(),
                )
            },
        )
        .add_handler(
            vec!["markitai-soft-break"],
            |_: &dyn htmd::element_handler::Handlers, _: htmd::Element| Some("\n".into()),
        )
        .add_handler(
            vec!["q"],
            |handlers: &dyn htmd::element_handler::Handlers, element: htmd::Element| {
                Some(format!("\"{}\"", handlers.walk_children(element.node).content).into())
            },
        )
        .add_handler(
            vec!["sub", "sup"],
            |handlers: &dyn htmd::element_handler::Handlers, element: htmd::Element| {
                Some(
                    format!(
                        "<{0}>{1}</{0}>",
                        element.tag,
                        handlers.walk_children(element.node).content.trim()
                    )
                    .into(),
                )
            },
        )
        .add_handler(
            vec![
                "div",
                "section",
                "article",
                "main",
                "figure",
                "figcaption",
                "dl",
                "dt",
                "dd",
                "details",
                "summary",
            ],
            |handlers: &dyn htmd::element_handler::Handlers, element: htmd::Element| {
                Some(
                    format!(
                        "\n\n{}\n\n",
                        handlers.walk_children(element.node).content.trim()
                    )
                    .into(),
                )
            },
        )
        .add_handler(
            vec!["table"],
            |handlers: &dyn htmd::element_handler::Handlers, element: htmd::Element| {
                let mut result = handlers.fallback(element)?;
                if let Some(compact) = compact_table(&result.content) {
                    result.content = compact;
                }
                Some(result)
            },
        )
        .add_handler(
            vec!["hr"],
            |_: &dyn htmd::element_handler::Handlers, _: htmd::Element| Some("\n\n---\n\n".into()),
        )
        .add_handler(
            vec!["blockquote"],
            |handlers: &dyn htmd::element_handler::Handlers, element: htmd::Element| {
                let mut result = handlers.fallback(element)?;
                result.content = tight_blockquote(&result.content);
                Some(result)
            },
        )
        .build()
}

#[derive(Debug)]
enum BbNode {
    Text(String),
    Tag {
        name: String,
        argument: Option<String>,
        children: Vec<BbNode>,
    },
}

struct BbFrame {
    name: String,
    argument: Option<String>,
    children: Vec<BbNode>,
}

fn bbcode_tree(source: &str) -> Result<Vec<BbNode>> {
    fn close(stack: &mut Vec<BbFrame>) {
        let frame = stack.pop().expect("a non-root BBCode frame");
        stack
            .last_mut()
            .expect("BBCode root frame")
            .children
            .push(BbNode::Tag {
                name: frame.name,
                argument: frame.argument,
                children: frame.children,
            });
    }
    let mut stack = vec![BbFrame {
        name: String::new(),
        argument: None,
        children: Vec::new(),
    }];
    let mut offset = 0;
    while offset < source.len() {
        let Some(relative_start) = source[offset..].find('[') else {
            stack
                .last_mut()
                .unwrap()
                .children
                .push(BbNode::Text(source[offset..].to_owned()));
            break;
        };
        let start = offset + relative_start;
        if start > offset {
            stack
                .last_mut()
                .unwrap()
                .children
                .push(BbNode::Text(source[offset..start].to_owned()));
        }
        let mut quote = None;
        let mut end = None;
        // Backslash-quoted delimiters occur in older event-store payloads even
        // after JSON decoding; the token normalizer below handles that form.
        for (index, character) in source[start + 1..].char_indices() {
            if character == '\'' || character == '"' {
                if quote == Some(character) {
                    quote = None;
                } else if quote.is_none() {
                    quote = Some(character);
                }
            } else if character == ']' && quote.is_none() {
                end = Some(start + 1 + index);
                break;
            }
        }
        let Some(end) = end else {
            stack
                .last_mut()
                .unwrap()
                .children
                .push(BbNode::Text(source[start..].to_owned()));
            break;
        };
        offset = end + 1;
        let token = source[start + 1..end]
            .replace(r"\/", "/")
            .replace("\\\"", "\"");
        let token = token.trim();
        let closing = token.starts_with('/');
        let token = token.strip_prefix('/').unwrap_or(token);
        let (name, argument) = token
            .split_once('=')
            .map_or((token, None), |(name, value)| {
                (
                    name,
                    Some(value.trim().trim_matches(['\'', '"']).to_owned()),
                )
            });
        let name = name.trim().to_ascii_lowercase();
        let name = if name == "*" { "li" } else { &name };
        if !matches!(
            name,
            "p" | "b"
                | "i"
                | "u"
                | "s"
                | "url"
                | "img"
                | "h1"
                | "h2"
                | "h3"
                | "h4"
                | "h5"
                | "h6"
                | "list"
                | "li"
                | "code"
                | "quote"
                | "previewyoutube"
        ) {
            stack
                .last_mut()
                .unwrap()
                .children
                .push(BbNode::Text(source[start..offset].to_owned()));
            continue;
        }
        if closing {
            if let Some(position) = stack.iter().rposition(|frame| frame.name == name) {
                while stack.len() > position {
                    close(&mut stack);
                }
            } else {
                stack
                    .last_mut()
                    .unwrap()
                    .children
                    .push(BbNode::Text(source[start..offset].to_owned()));
            }
            continue;
        }
        if name == "code" {
            let tail = &source[offset..];
            let lower = tail.to_ascii_lowercase();
            let close = lower
                .find("[/code]")
                .map(|index| (index, 7))
                .or_else(|| lower.find(r"[\/code]").map(|index| (index, 8)));
            let (length, closing_length) = close.unwrap_or((tail.len(), 0));
            stack.last_mut().unwrap().children.push(BbNode::Tag {
                name: name.into(),
                argument,
                children: vec![BbNode::Text(tail[..length].to_owned())],
            });
            offset += length + closing_length;
            continue;
        }
        if name == "li"
            && let Some(list) = stack.iter().rposition(|frame| frame.name == "list")
            && let Some(item) = stack
                .iter()
                .enumerate()
                .skip(list + 1)
                .find_map(|(index, frame)| (frame.name == "li").then_some(index))
        {
            while stack.len() > item {
                close(&mut stack);
            }
        }
        if stack.len() > 64 {
            return Err(Error::Conversion("BBCode nesting exceeds 64 tags".into()));
        }
        stack.push(BbFrame {
            name: name.into(),
            argument,
            children: Vec::new(),
        });
    }
    while stack.len() > 1 {
        close(&mut stack);
    }
    Ok(stack.pop().unwrap().children)
}

fn bbcode_text(nodes: &[BbNode]) -> String {
    nodes
        .iter()
        .map(|node| match node {
            BbNode::Text(text) => text.clone(),
            BbNode::Tag { children, .. } => bbcode_text(children),
        })
        .collect()
}

fn escaped_bbcode_text(text: &str, output: &mut String) {
    // Keep angle brackets encoded after the HTML renderer's entity decoding,
    // so all displayed BBCode text stays literal in Markdown. Code nodes use
    // a separate fenced path; destinations use URL validation instead.
    escaped(&text.replace('<', "&lt;").replace('>', "&gt;"), output);
}

fn bbcode_html(nodes: &[BbNode], base: Option<&Url>, output: &mut String) {
    for node in nodes {
        match node {
            BbNode::Text(text) => {
                for (index, line) in text.split('\n').enumerate() {
                    if index > 0 {
                        output.push_str("<br>");
                    }
                    escaped_bbcode_text(line, output);
                }
            }
            BbNode::Tag {
                name,
                argument,
                children,
            } => {
                let argument = argument.as_deref().unwrap_or("");
                if name == "code" {
                    output.push_str("<pre><code");
                    if !argument.is_empty()
                        && argument
                            .chars()
                            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
                    {
                        output.push_str(" class=\"language-");
                        escaped(argument, output);
                        output.push('"');
                    }
                    output.push('>');
                    escaped(&bbcode_text(children), output);
                    output.push_str("</code></pre>");
                    continue;
                }
                if matches!(name.as_str(), "url" | "img") {
                    let text = bbcode_text(children);
                    let destination = if argument.is_empty() {
                        text.as_str()
                    } else {
                        argument
                    };
                    let destination = destination.replace(r"\/", "/");
                    if let Some(url) = safe_url(&destination, base) {
                        output.push_str(if name == "img" {
                            "<img src=\""
                        } else {
                            "<a href=\""
                        });
                        escaped(&url, output);
                        output.push_str("\">");
                        if name == "url" {
                            bbcode_html(children, base, output);
                            output.push_str("</a>");
                        }
                    } else {
                        bbcode_html(children, base, output);
                    }
                    continue;
                }
                if name == "previewyoutube" {
                    let id = argument.split(';').next().unwrap_or("").trim();
                    if id.len() == 11
                        && id
                            .chars()
                            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
                    {
                        output.push_str("<img src=\"https://www.youtube.com/watch?v=");
                        escaped(id, output);
                        output.push_str("\">");
                    } else {
                        bbcode_html(children, base, output);
                    }
                    continue;
                }
                let tag = match name.as_str() {
                    "b" => "strong",
                    "i" => "em",
                    "s" => "del",
                    "list" if !argument.is_empty() => "ol",
                    "list" => "ul",
                    "quote" => "blockquote",
                    name => name,
                };
                output.push('<');
                output.push_str(tag);
                output.push('>');
                if name == "quote" && !argument.is_empty() {
                    output.push_str("<p><strong>");
                    escaped_bbcode_text(argument, output);
                    output.push_str("</strong></p>");
                }
                bbcode_html(children, base, output);
                output.push_str("</");
                output.push_str(tag);
                output.push('>');
            }
        }
    }
}

struct Announcement {
    markdown: String,
    metadata: Map<String, Value>,
}

fn structured_announcement(
    page: &Landmarks<'_>,
    base: Option<&Url>,
) -> Result<Option<Announcement>> {
    for carrier in &page.events {
        let raw = carrier
            .value()
            .attribute("data-partnereventstore")
            .unwrap_or("");
        let Ok(data) = serde_json::from_str::<Value>(raw) else {
            continue;
        };
        let entries = match &data {
            Value::Array(entries) => entries.iter().collect::<Vec<_>>(),
            Value::Object(_) => vec![&data],
            _ => continue,
        };
        for entry in entries {
            let Some(body) = entry.get("announcement_body") else {
                continue;
            };
            let Some(text) = body
                .get("body")
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
            else {
                continue;
            };
            let nodes = bbcode_tree(text)?;
            let mut html = String::new();
            bbcode_html(&nodes, base, &mut html);
            let tree = Html::parse_fragment(&html);
            let markdown = render_clean(tree.root_element(), base)?;
            if markdown.is_empty() {
                continue;
            }
            let mut metadata = Map::new();
            if let Some(title) = body
                .get("headline")
                .and_then(Value::as_str)
                .and_then(|value| clean_title(value, None))
            {
                metadata.insert("title".into(), title.into());
            }
            if let Some(time) = body
                .get("posttime")
                .and_then(Value::as_i64)
                .and_then(|seconds| chrono::DateTime::from_timestamp(seconds, 0))
            {
                metadata.insert("published".into(), time.to_rfc3339().into());
            }
            if let Some(group) = carrier
                .value()
                .attribute("data-groupvanityinfo")
                .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                .and_then(|value| {
                    value
                        .as_array()
                        .and_then(|array| array.first())
                        .and_then(|group| group.get("group_name"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .filter(|group| !group.trim().is_empty())
            {
                metadata.insert("author".into(), group.trim().into());
            }
            return Ok(Some(Announcement { markdown, metadata }));
        }
    }
    Ok(None)
}

/// Whether a page is a Substack page: by its host, its permalink container or
/// its CDN assets.
fn substack_page(page: &Landmarks<'_>, base: Option<&Url>) -> bool {
    let substack_host = |host: &str| host == "substack.com" || host.ends_with(".substack.com");
    let cdn_host = |host: &str| host == "substackcdn.com" || host.ends_with(".substackcdn.com");
    base.and_then(Url::host_str).is_some_and(substack_host)
        || page.permalink.is_some()
        || page.assets.iter().any(|asset| {
            asset
                .value()
                .attribute("href")
                .or_else(|| asset.value().attribute("src"))
                .and_then(|source| Url::parse(source).ok())
                .is_some_and(|url| url.host_str().is_some_and(cdn_host))
        })
}

/// A Substack note: the main note's text and its attached image. The feed,
/// recommendations and app promotion around a note are not its content.
/// An article with its own rendered body is left to the ordinary reader.
fn substack_note(
    document: &Html,
    page: &Landmarks<'_>,
    base: Option<&Url>,
) -> Result<Option<Announcement>> {
    let permalink = page.permalink;
    if !substack_page(page, base) || page.rendered_body {
        return Ok(None);
    }
    let note_selector = selector("div.ProseMirror.FeedProseMirror");
    let note = match permalink {
        Some(unit) => unit.select(&note_selector).next(),
        None => document.select(&note_selector).next(),
    };
    let Some(note) = note else {
        return Ok(None);
    };
    let mut html = note.html();
    if let Some(image) = meta(&page.metas, &["og:image"]).or_else(|| substack_note_image(note)) {
        html.push_str("<img alt=\"\" src=\"");
        escaped(&image, &mut html);
        html.push_str("\">");
    }
    let tree = Html::parse_fragment(&html);
    let markdown = render_clean(tree.root_element(), base)?;
    if markdown.is_empty() {
        return Ok(None);
    }
    let mut metadata = Map::new();
    metadata.insert("site".into(), "Substack".into());
    if let Some(title) = meta(&page.metas, &["og:title"]) {
        // "Test User (@testuser)" names the note's author before the handle.
        let author = match title.rfind(" (@") {
            Some(at) if title.ends_with(')') => title[..at].trim().to_owned(),
            _ => title.trim().to_owned(),
        };
        if !author.is_empty() {
            metadata.insert("author".into(), author.into());
        }
        metadata.insert("title".into(), title.into());
    }
    Ok(Some(Announcement { markdown, metadata }))
}

/// The image attached to a note: the largest `srcset` entry of the image
/// grid right after the note's comment body (or after its wrapper).
fn substack_note_image<'a>(note: ElementRef<'a>) -> Option<String> {
    let classes =
        |element: ElementRef<'_>| element.value().attribute("class").unwrap_or("").to_owned();
    let body = note
        .ancestors()
        .filter_map(ElementRef::wrap)
        .find(|parent| {
            let classes = classes(*parent);
            classes.contains("feedCommentBody") && !classes.contains("feedCommentBodyInner")
        })?;
    let next_element = |element: ElementRef<'a>| element.next_siblings().find_map(ElementRef::wrap);
    let grid = [
        next_element(body),
        body.parent()
            .and_then(ElementRef::wrap)
            .and_then(next_element),
    ]
    .into_iter()
    .flatten()
    .find(|element| classes(*element).contains("imageGrid"))?;
    let image = grid.select(&selector("img")).next()?;
    let largest = image.value().attribute("srcset").and_then(srcset::best);
    largest
        .or_else(|| image.value().attribute("src"))
        .map(str::to_owned)
}

pub(super) fn fragment(source: &str) -> Result<String> {
    let document = Html::parse_fragment(source);
    render_clean(document.root_element(), None)
}

/// Replace declarative shadow-root templates (`shadowrootmode` or legacy
/// `shadowroot`, open or closed) with their markup before parsing, innermost
/// first and at most ten levels deep, as the reference does: a browser would
/// attach that markup, while a static parser leaves it inert. Other templates
/// are unchanged.
fn flatten_shadow_roots(source: &str) -> std::borrow::Cow<'_, str> {
    static MENTION: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    static OPENING: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    // The word in any ASCII case, found by the regex engine's literal
    // search: several times faster than comparing every ten bytes of a page.
    let mention =
        MENTION.get_or_init(|| regex::Regex::new("(?i-u)shadowroot").expect("static pattern"));
    if !mention.is_match(source) {
        return std::borrow::Cow::Borrowed(source);
    }
    let opening = OPENING.get_or_init(|| {
        regex::Regex::new(
            r#"(?is)<template\s[^>]*\bshadowroot(?:mode)?\s*=\s*["']?(?:open|closed)["']?[^>]*>"#,
        )
        .expect("static pattern")
    });
    let template_start = |lower: &str, from: usize| -> Option<usize> {
        let mut at = from;
        while let Some(found) = lower[at..].find("<template") {
            let index = at + found;
            let next = lower.as_bytes().get(index + 9);
            if next.is_none_or(|byte| !(byte.is_ascii_alphanumeric() || *byte == b'_')) {
                return Some(index);
            }
            at = index + 9;
        }
        None
    };
    let mut html = source.to_owned();
    for _ in 0..10 {
        let lower = html.to_ascii_lowercase();
        let mut result = String::with_capacity(html.len());
        let mut copied = 0;
        let mut search = 0;
        let mut replaced = false;
        while let Some(open) = opening.find_at(&html, search) {
            let body_start = open.end();
            let Some(close) = lower[body_start..]
                .find("</template>")
                .map(|at| body_start + at)
            else {
                break;
            };
            if template_start(&lower, body_start).is_some_and(|inner| inner < close) {
                // Not innermost: an inner template resolves in this or a later pass.
                search = open.start() + 1;
                continue;
            }
            result.push_str(&html[copied..open.start()]);
            result.push_str(&html[body_start..close]);
            copied = close + "</template>".len();
            search = copied;
            replaced = true;
        }
        if !replaced {
            break;
        }
        result.push_str(&html[copied..]);
        html = result;
    }
    std::borrow::Cow::Owned(html)
}

/// Words as the reference counts them: each CJK character is a word; other
/// text counts whitespace-separated runs after punctuation becomes space.
pub(crate) fn count_words(text: &str) -> usize {
    let cjk = |ch: char| matches!(ch as u32, 0x3040..=0x30ff | 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xac00..=0xd7af | 0xf900..=0xfaff | 0x20000..=0x2a6df);
    let bytes = text.as_bytes();
    let mut words = 0;
    let mut in_word = false;
    let mut at = 0;
    while at < bytes.len() {
        // Most text is ASCII, read a byte at a time: no CJK range to test
        // and no Unicode table to look up.
        let byte = bytes[at];
        let word = if byte.is_ascii() {
            at += 1;
            byte.is_ascii_alphanumeric() || byte == b'_'
        } else {
            let ch = text[at..].chars().next().unwrap_or_default();
            at += ch.len_utf8();
            if cjk(ch) {
                words += 1;
                in_word = false;
                continue;
            }
            ch.is_alphanumeric()
        };
        words += usize::from(word && !in_word);
        in_word = word;
    }
    words
}

pub(crate) use charset::{decode_fetched, legacy_encoding};

/// Extract a local HTML file's bytes. A `<meta>` declaration decides unless a
/// BOM or valid UTF-8 is present (see [`charset`]); byte sequences invalid in
/// the declared encoding become replacement characters with a warning.
/// Undeclared bytes are read like plain text.
pub(super) fn extract_html_bytes(bytes: &[u8]) -> Result<Document> {
    let Some(encoding) = legacy_encoding(bytes) else {
        let (source, warning) = super::text::decode_legacy(bytes)?;
        let mut document = extract_html(&source, None)?;
        document.warnings.extend(warning);
        return Ok(document);
    };
    let (source, malformed) = encoding.decode_without_bom_handling(bytes);
    let mut document = extract_html(&source, None)?;
    if malformed {
        document.warnings.push(format!(
            "HTML declared as {} contains invalid byte sequences; they were replaced with U+FFFD.",
            encoding.name()
        ));
    }
    Ok(document)
}

/// Extract an article candidate, metadata and Markdown without fetching links.
pub fn extract_html(source: &str, base_url: Option<&str>) -> Result<Document> {
    FLATTENED.with(|flattened| flattened.set(false));
    let source = flatten_shadow_roots(source);
    let mut document = Html::parse_document(&source);
    stream::restore(&mut document)?;
    let base = base_url.and_then(|value| Url::parse(value).ok());
    // A reading site's article, read from what the page serves (see `sites`).
    let mut site_metadata = Map::new();
    if let Some(reading) = sites::read(&document, base.as_ref()) {
        match reading.page {
            Some(page) => {
                if let Ok(mut converted) = extract_html(&page, base_url)
                    && !converted.markdown.trim().is_empty()
                {
                    converted.metadata.extend(reading.metadata);
                    return Ok(converted);
                }
            }
            None => site_metadata = reading.metadata,
        }
    }
    let facts = Facts::new(document.root_element());
    let root = article::select(&document, &facts);
    let landmarks = Landmarks::read(&document);
    let mut metadata = Map::new();
    let jsonld = jsonld_documents(&landmarks.scripts);
    let site = meta(&landmarks.metas, &["og:site_name", "application-name"]);
    let title = page_title(&landmarks, &jsonld, site.as_deref());
    if let Some(title) = title.filter(|value| !value.is_empty()) {
        metadata.insert("title".into(), title.into());
    }
    let author = page_author(&landmarks.metas, &jsonld);
    // The title was cleaned with the site the page's meta tags name. Without
    // one, the JSON-LD publisher names the site, as defuddle reads it, and a
    // Substack page that has neither is `Substack`, as the reference's
    // Substack reader has it.
    let site = site
        .or_else(|| jsonld_publisher(&jsonld))
        .or_else(|| substack_page(&landmarks, base.as_ref()).then(|| "Substack".into()));
    let published = meta(&landmarks.metas, &["article:published_time"])
        .or_else(|| jsonld_text(&jsonld, "datePublished"))
        .or_else(|| {
            landmarks.times.iter().find_map(|time| {
                let value = time.value().attribute("datetime")?.trim();
                let bytes = value.as_bytes();
                (bytes.len() >= 7
                    && bytes[..4].iter().all(u8::is_ascii_digit)
                    && bytes[4] == b'-'
                    && bytes[5..7].iter().all(u8::is_ascii_digit))
                .then(|| value.to_owned())
            })
        });
    for (key, value) in [
        ("author", author),
        ("site", site),
        ("published", published),
        (
            "description",
            meta(
                &landmarks.metas,
                &["description", "og:description", "twitter:description"],
            ),
        ),
    ] {
        if let Some(value) = value {
            metadata.insert(key.into(), value.into());
        }
    }
    if let Some(canonical) = landmarks
        .canonical
        .and_then(|node| node.value().attribute("href"))
        .and_then(|value| safe_url(value, base.as_ref()))
        .filter(|value| {
            !base.as_ref().is_some_and(|source| {
                !source.path().trim_matches('/').is_empty()
                    && Url::parse(value)
                        .is_ok_and(|canonical| canonical.path().trim_matches('/').is_empty())
            })
        })
    {
        metadata.insert("canonical_url".into(), canonical.into());
    }
    if let Some(base) = &base {
        metadata.insert("source".into(), base.as_str().into());
        if let Some(host) = base.host_str() {
            metadata.insert("domain".into(), host.into());
        }
    }
    let markdown = if let Some(announcement) = structured_announcement(&landmarks, base.as_ref())? {
        metadata.extend(announcement.metadata);
        announcement.markdown
    } else if let Some(note) = substack_note(&document, &landmarks, base.as_ref())? {
        metadata.extend(note.metadata);
        note.markdown
    } else if let Some(post) = social::post(&document, &landmarks, base.as_ref())? {
        metadata.extend(post.metadata);
        post.markdown
    } else if let Some(page) = hacker_news::page(&document, &landmarks, base.as_ref())? {
        metadata.extend(page.metadata);
        page.markdown
    } else {
        render_with_footnotes(root, document.root_element(), base.as_ref(), true, &facts)?
    };
    if markdown.is_empty() {
        return Err(Error::Conversion(
            "HTML contains no extractable content".into(),
        ));
    }
    if base.is_some() {
        // A web page's extraction fact, as the reference reports it.
        metadata.insert("word_count".into(), count_words(&markdown).into());
    }
    metadata.insert("converter".into(), "native-html".into());
    metadata.extend(site_metadata);
    let mut warnings = Vec::new();
    if FLATTENED.with(|flattened| flattened.replace(false)) {
        warnings.push(format!("HTML is nested deeper than {MAX_DEPTH} elements; the content below that depth was kept as plain text without its formatting."));
    }
    Ok(Document {
        markdown,
        metadata,
        warnings,
        ..Document::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    thread_local! {
        /// Conversions on this thread whose cleaned markup had to be parsed.
        static MARKUP: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    pub(super) fn markup_parsed() {
        MARKUP.with(|count| count.set(count.get() + 1));
    }

    /// The same nodes, names, attributes and text in the same order.
    pub(super) fn same_tree(a: htmd::NodeRef<'_>, b: htmd::NodeRef<'_>) -> bool {
        let same = match (a.value(), b.value()) {
            (scraper::Node::Element(x), scraper::Node::Element(y)) => {
                x.name == y.name && x.attrs == y.attrs
            }
            (scraper::Node::Text(x), scraper::Node::Text(y)) => *x.text == *y.text,
            (x, y) => x == y,
        };
        same && a.children().count() == b.children().count()
            && a.children().zip(b.children()).all(|(x, y)| same_tree(x, y))
    }

    #[test]
    fn nesting_deeper_than_the_limit_keeps_its_words_with_a_warning() {
        let deep = format!(
            "<h1>Title</h1><p>Before</p>{}<p>Deep <b>bold</b></p><p>text</p>{}<p>After</p>",
            "<div>".repeat(300),
            "</div>".repeat(300)
        );
        let document = extract_html(&deep, None).unwrap();
        assert!(document.markdown.starts_with("# Title\n\nBefore"));
        assert!(
            document.markdown.contains("Deep bold text"),
            "{}",
            document.markdown
        );
        assert!(document.markdown.ends_with("After"));
        assert_eq!(document.warnings.len(), 1);
        assert!(document.warnings[0].starts_with("HTML is nested deeper than 256 elements"));
        // Unclosed legacy inline tags nest the same way.
        let fonts: String = (0..400).map(|i| format!("<font size=2>w{i} ")).collect();
        let document = extract_html(&format!("<p>Start</p>{fonts}"), None).unwrap();
        assert!(document.markdown.contains("w0 w1") && document.markdown.ends_with("w399"));
        // The flag does not leak into the next conversion on this thread.
        assert!(
            extract_html("<p>plain</p>", None)
                .unwrap()
                .warnings
                .is_empty()
        );
    }

    #[test]
    fn attribute_lookup_answers_as_scraper_does() {
        let document = Html::parse_document(
            r#"<html xml:lang="en"><body data-x="1" CLASS="a b" id="i">
            <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1 1">
            <a xlink:href="/svg" href="/plain"><title>t</title></a></svg>
            <math><mi definitionURL="u" mathvariant="bold">x</mi></math>
            <img src="s" alt="" data-src="d"><p hidden>h</p></body></html>"#,
        );
        let names = [
            "class",
            "CLASS",
            "id",
            "data-x",
            "src",
            "alt",
            "data-src",
            "hidden",
            "href",
            "xlink:href",
            "xmlns",
            "xmlns:xlink",
            "viewBox",
            "viewbox",
            "definitionURL",
            "definitionurl",
            "mathvariant",
            "xml:lang",
            "lang",
            "",
            "missing",
        ];
        let mut found = 0;
        for element in document
            .root_element()
            .descendants()
            .filter_map(ElementRef::wrap)
        {
            for name in names {
                let value = element.value();
                assert_eq!(
                    value.attribute(name),
                    value.attr(name),
                    "{name} on {}",
                    value.name()
                );
                found += usize::from(value.attribute(name).is_some());
            }
        }
        assert!(found >= 10, "{found}");
    }

    #[test]
    fn consecutive_source_line_breaks_are_one_soft_break() {
        for source in ["<p>first\n\n\nsecond</p>", "<p>first\r\n  \r\nsecond</p>"] {
            assert_eq!(fragment(source).unwrap(), "first\nsecond", "{source:?}");
        }
    }

    #[test]
    fn markup_writer_escapes_values_and_closes_its_last_tag() {
        let mut output = Out::markup();
        output.start("a");
        output.attribute("title", "say \"hi\" & <go>");
        output.text("x < y");
        output.end("a");
        output.start("img");
        assert_eq!(
            output.into_markup(),
            "<a title=\"say &quot;hi&quot; &amp; &lt;go&gt;\">x &lt; y</a><img>"
        );
    }

    /// Converts a source as a page with and without a doctype and as a
    /// fragment; every conversion checks the written tree against the parsed
    /// one (in `render_cleaned`). Returns how often the markup was parsed.
    fn markup_parsed_for(source: &str) -> usize {
        let before = MARKUP.with(std::cell::Cell::get);
        for document in [
            format!("<!doctype html><body>{source}</body>"),
            source.to_owned(),
        ] {
            let _ = extract_html(&document, None);
        }
        let _ = fragment(source);
        MARKUP.with(std::cell::Cell::get) - before
    }

    #[test]
    fn cleaned_pages_build_the_tree_their_markup_parses_into() {
        for source in [
            // Callouts, notes, soft breaks, void and unknown elements.
            "<div class='callout' data-callout='note'><div class='callout-title'><div class='callout-title-inner'>Title</div></div><div class='callout-content'><p>Body\n  line</p></div></div>\
             <p>Claim<sup id='r1'><a href='#n1'>1</a></sup> and <param>p</param><wbr> <custom-el a='1'>c</custom-el></p>\
             <ol><li id='n1'>The note text. <a href='#r1'>↩</a></li></ol>",
            // Line feeds after `<pre>`, and carriage returns written as references.
            "<pre>\n\nfirst line\n</pre><pre>a&#13;b&#13;&#10;c&#13;</pre><p>x&#13;y&#13;&#10;z</p>",
            // Tables written as grids and as blocks, lists and headings.
            "<table><caption>Cap</caption><tr><th>h</th><th>i</th></tr><tr><td>1</td><td rowspan='2'>2</td></tr><tr><td><b>3</b></td></tr></table>\
             <table><tr><td><h2><a href='/x'>Title</a></h2><ul><li>a<ul><li>b</li></ul></li></ul></td><td><p>Text</p></td></tr></table>",
            // A link around a layout table is written on the block inside it.
            "<a href='/1'><div><table><tr><td><a href='/2'>x</a></td><td>y</td></tr></table></div></a>",
            // An email's head elements in its body, and raw text: a title
            // with line breaks written as markers, text elements whose
            // escapes stay as written.
            "<html><head><title>Mail</title><meta name='x'><style>p {}</style></head><body><p>Hi</p></body></html>",
            "<p>a</p><title>Inner &amp; ti\ntle</title><p>b</p>",
            "<p>a</p><xmp>x &lt; y\nz</xmp><noembed>n &amp; m</noembed><noframes>f</noframes>",
            "<p>a</p><plaintext>rest <b>of</b> it",
        ] {
            assert_eq!(markup_parsed_for(source), 0, "{source}");
        }
    }

    /// Markup the parser rearranges (closed, moved or merged elements) or
    /// reads as foreign content is parsed, and the conversion stays the one
    /// of the parsed markup.
    #[test]
    fn cleaned_pages_the_parser_rearranges_parse_their_markup() {
        for source in [
            // A code block inside a paragraph closes the paragraph; the
            // formatting element around it is opened again after it.
            "<p>Use <b>bold <code style='white-space: pre'>x = 1</code> tail</b> now</p>",
            // A link around blocks moves onto its first heading, inside a heading.
            "<h2><a href='/x'><h3>Inner title</h3><p>Summary text of the card</p></a></h2>",
            // A layout table's cells become blocks, and its list items close
            // each other without the cells between them.
            "<ul><li><table><tr><td><li>a</li></td><td>b <a href='/2'>two</a></td></tr></table></li></ul>",
            // A table without rows keeps its text, which the parser moves
            // before the table.
            "<table>\n  <caption>Cap\n  tion</caption>\n  <colgroup><col></colgroup>\n</table>",
            // Foreign content: SVG names with capitals, MathML without TeX.
            "<p>Figure <svg viewBox='0 0 1 1'><foreignObject><div>inside</div></foreignObject><clipPath id='c'></clipPath></svg> after <math><mrow><mi>x</mi></mrow></math></p>",
        ] {
            assert!(markup_parsed_for(source) > 0, "{source}");
        }
        // Script content needs the markup parsed (the cleaner writes none).
        let before = MARKUP.with(std::cell::Cell::get);
        let markdown = render_cleaned(&|output| {
            for (name, text) in [("p", "a"), ("script", "x < y"), ("p", "b")] {
                output.start(name);
                output.text(text);
                output.end(name);
            }
            Ok(())
        });
        assert_eq!(markdown.unwrap(), "a\n\nb");
        assert_eq!(MARKUP.with(std::cell::Cell::get), before + 1);
    }

    #[test]
    fn streamed_article_keeps_metadata_links_and_resolved_notes() {
        let result = extract_html(
            r##"<title>Recovered article</title><main><!--$?--><template id="B:0"></template><p>Loading content</p><!--/$--></main>
            <div hidden id="S:0"><article><h1>Recovered article</h1><p>Readable evidence<a role="doc-noteref" href="#fn-1">1</a> and <a href="/source">its source</a>.</p><aside id="fn-1" role="doc-footnote">The complete supporting note.</aside><p hidden>Still hidden</p></article></div>
            <script>$RC("B:0","S:0");</script>"##,
            Some("https://example.test/article"),
        )
        .unwrap();
        assert_eq!(result.metadata["title"], "Recovered article");
        assert!(result.markdown.contains("Readable evidence[^1]"));
        assert!(
            result
                .markdown
                .contains("[its source](https://example.test/source)")
        );
        assert!(
            result
                .markdown
                .contains("[^1]: The complete supporting note.")
        );
        assert!(!result.markdown.contains("Loading content"));
        assert!(!result.markdown.contains("Still hidden"));
    }

    #[test]
    fn article_pruning_keeps_external_notes_and_drops_widget_only_references() {
        let source = r##"<body><article><p>Evidence from the article remains available together with its source and full explanatory paragraph. This main narrative must survive the removal of unrelated recommendation controls.<a role="doc-noteref" href="#fn-main">1</a></p>
            <div class="related-posts"><h2>Related posts</h2><a href="/other">Other story</a><a role="doc-noteref" href="#fn-widget">2</a></div></article>
            <aside role="doc-footnote" id="fn-main">The retained external source.</aside><aside role="doc-footnote" id="fn-widget">Widget-only definition.</aside></body>"##;
        let result = extract_html(source, None).unwrap();
        assert!(result.markdown.contains("main narrative must survive"));
        assert!(result.markdown.contains("[^1]"));
        assert!(
            result
                .markdown
                .ends_with("[^1]: The retained external source."),
            "{}",
            result.markdown
        );
        assert!(!result.markdown.contains("Other story"));
        assert!(!result.markdown.contains("Widget-only"));
        assert!(!result.markdown.contains("[^2]"));
    }

    #[test]
    fn document_fragments_retain_book_toc_and_related_reading() {
        let result = fragment(r##"<section class="toc"><h2>Contents</h2><ul><li><a href="#chapter">Chapter one</a></li></ul></section><section class="related-posts"><p>Related reading belongs to the book.</p></section><h2 id="chapter">Chapter one</h2><p>Full chapter text.</p>"##).unwrap();
        assert!(result.contains("[Chapter one](#chapter)"));
        assert!(result.contains("Related reading belongs to the book."));
        assert!(result.contains("Full chapter text."));
    }

    #[test]
    fn external_note_roles_do_not_recover_inert_or_literal_definitions() {
        for wrapper in ["pre", "template", "noscript"] {
            let source = format!(
                r##"<article><p>A substantial visible article retains its explanatory text and linked evidence. The referenced source is absent from rendered content and must not be recovered from an inert example.<a role="doc-noteref" href="#fn-example">1</a></p></article><{wrapper}><aside role="doc-footnote" id="fn-example">Inert definition.</aside></{wrapper}>"##
            );
            let result = extract_html(&source, None).unwrap();
            assert!(
                result.markdown.contains("[1](#fn-example)"),
                "{wrapper}: {}",
                result.markdown
            );
            assert!(
                !result.markdown.contains("[^1]"),
                "{wrapper}: {}",
                result.markdown
            );
            assert!(
                !result.markdown.contains("Inert definition"),
                "{wrapper}: {}",
                result.markdown
            );
        }
    }

    #[test]
    fn code_editor_numeric_links_are_literal_before_footnote_collection() {
        let result = extract_html(r##"<article><p>Example:</p><div class="cm-content"><div class="cm-line">goto <a role="doc-noteref" href="#fn-code">1</a></div><div class="cm-line">return 2</div></div><p>Explanation<a role="doc-noteref" href="#fn-real">2</a>.</p><ol class="footnotes"><li id="fn-code">Not a code footnote.</li><li id="fn-real">The prose source.</li></ol></article>"##, None).unwrap();
        assert!(
            result.markdown.contains("```\ngoto 1\nreturn 2\n```"),
            "{}",
            result.markdown
        );
        assert!(
            result.markdown.contains("Explanation[^1]."),
            "{}",
            result.markdown
        );
        assert!(result.markdown.ends_with("[^1]: The prose source."));
        assert!(!result.markdown.contains("[^2]"));
    }

    #[test]
    fn article_drops_navigation_and_resolves_links() {
        let doc = extract_html(r#"<title>Fallback</title><meta property="og:title" content="Article"><nav>menu</nav><article><h1>Article</h1><p>Some <strong>important</strong> text <a href="../next">next</a>.</p><img data-src="/image.png" alt="Photo"><p hidden>hidden</p><script>alert(1)</script></article><footer>footer</footer>"#, Some("https://example.test/posts/one")).unwrap();
        assert_eq!(doc.metadata["title"], "Fallback");
        assert!(doc.markdown.contains("**important**"));
        assert!(doc.markdown.contains("https://example.test/next"));
        assert!(doc.markdown.contains("https://example.test/image.png"));
        for excluded in ["menu", "hidden", "alert", "footer"] {
            assert!(!doc.markdown.contains(excluded));
        }
    }
    #[test]
    fn article_keeps_code_table_unicode_and_safe_text() {
        let doc = extract_html(r#"<main><h1>中文</h1><pre><code class="language-rust">let n = 1 &lt; 2;</code></pre><table><tr><th>A</th><th>B</th></tr><tr><td>x</td><td>y</td></tr></table><a href="javascript:alert(1)">safe label</a></main>"#, None).unwrap();
        assert!(doc.markdown.contains("let n = 1 < 2;"));
        assert!(doc.markdown.contains("|"));
        assert!(doc.markdown.contains("safe label"));
        assert!(!doc.markdown.contains("javascript:"));
    }

    #[test]
    fn structured_metadata_uses_graph_headline_and_author() {
        let doc = extract_html(r#"<title>Wrong title</title><meta property="og:title" content="Wrong social title"><meta name="og:site_name" content="Journal"><script type="application/ld+json">{"@graph":[{"name":"Website"},{"headline":"Journal - Structured headline · Journal","author":{"name":"Ada"},"datePublished":"2026-02-03"}]}</script><main><p>Article text.</p></main>"#, None).unwrap();
        assert_eq!(doc.metadata["title"], "Structured headline");
        assert_eq!(doc.metadata["author"], "Ada");
        assert_eq!(doc.metadata["published"], "2026-02-03");
        assert!(!doc.markdown.contains("Structured headline"));
    }

    #[test]
    fn title_layers_recover_truncation_and_keep_real_subtitles() {
        let doc = extract_html(r#"<title>A complete long headline</title><script type="application/ld+json">{"headline":"A complete…"}</script><p>Body</p>"#, None).unwrap();
        assert_eq!(doc.metadata["title"], "A complete long headline");
        for (title, expected) in [
            ("Article | Site", "Article"),
            ("Article - a retrospective", "Article - a retrospective"),
        ] {
            let doc = extract_html(
                &format!("<title>{title}</title><h1>Article</h1><p>Body</p>"),
                None,
            )
            .unwrap();
            assert_eq!(doc.metadata["title"], expected);
        }
        let long = "界".repeat(350);
        assert_eq!(clean_title(&long, None).unwrap().chars().count(), 301);
    }

    #[test]
    fn malformed_jsonld_and_meta_author_do_not_hide_valid_layers() {
        let doc = extract_html(r#"<meta name="author" content="Meta author"><meta property="article:published_time" content="2026-01-01"><title>Document title</title><meta property="og:title" content="Social title"><script type="application/ld+json">invalid</script><script type="application/ld+json">[{"author":"Schema author","datePublished":"2025-01-01"}]</script><p>Body</p>"#, None).unwrap();
        assert_eq!(doc.metadata["title"], "Document title");
        assert_eq!(doc.metadata["author"], "Meta author");
        assert_eq!(doc.metadata["published"], "2026-01-01");
        let doc = extract_html(
            "<time datetime=PT2H>Duration</time><time datetime=2026-03-04>Today</time>",
            None,
        )
        .unwrap();
        assert_eq!(doc.metadata["published"], "2026-03-04");
    }

    /// The Markdown of `body` as the article of a full page.
    fn article_markdown(body: &str) -> String {
        extract_html(
            &format!("<html><body><article>{body}</article></body></html>"),
            None,
        )
        .unwrap()
        .markdown
    }

    #[test]
    fn an_author_is_a_name_never_a_web_address() {
        // `article:author` is often a profile link; the JSON-LD person is the name.
        let bbc = r#"<meta property="article:author" content="https://www.facebook.com/bbcnews"><script type="application/ld+json">{"@type":"NewsArticle","author":[{"@type":"Person","name":"Katya Adler"},{"@type":"Organization","name":"BBC News"}]}</script><main><p>Body.</p></main>"#;
        let doc = extract_html(bbc, None).unwrap();
        assert_eq!(doc.metadata["author"], "Katya Adler");
        // A web address alone is no author, in a meta tag or in JSON-LD.
        let doc = extract_html(
            r#"<meta name="author" content="https://example.test/people/ada"><meta property="article:author" content="https://example.test/people/ada"><script type="application/ld+json">{"author":{"@type":"Person","name":"https://example.test/people/ada"}}</script><main><p>Body.</p></main>"#,
            None,
        )
        .unwrap();
        assert!(!doc.metadata.contains_key("author"), "{:?}", doc.metadata);
        // A name beats a profile link in either meta tag, and `author` leads.
        let doc = extract_html(
            r#"<meta property="article:author" content="Grace"><meta name="author" content="Ada"><main><p>Body.</p></main>"#,
            None,
        )
        .unwrap();
        assert_eq!(doc.metadata["author"], "Ada");
        let doc = extract_html(
            r#"<meta property="article:author" content="Grace"><main><p>Body.</p></main>"#,
            None,
        )
        .unwrap();
        assert_eq!(doc.metadata["author"], "Grace");
        // A graph's person, found through its `@id`; several persons are joined.
        let doc = extract_html(
            r##"<script type="application/ld+json">{"@graph":[{"@type":"Article","author":{"@id":"https://x.test/#/p/1"}},{"@type":"Person","@id":"https://x.test/#/p/1","name":"Edsger  Dijkstra"}]}</script><main><p>Body.</p></main>"##,
            None,
        )
        .unwrap();
        assert_eq!(doc.metadata["author"], "Edsger Dijkstra");
        let doc = extract_html(
            r#"<script type="application/ld+json">{"author":[{"@type":"Person","name":"A"},{"@type":"Person","name":"B"},{"@type":"Person","name":"A"}]}</script><main><p>Body.</p></main>"#,
            None,
        )
        .unwrap();
        assert_eq!(doc.metadata["author"], "A, B");
    }

    #[test]
    fn a_headline_that_is_a_description_does_not_replace_the_page_title() {
        // Wikipedia's JSON-LD `headline` is the article's short description.
        let wikipedia = |headline: &str| {
            format!(
                r#"<title>Rust (programming language) - Wikipedia</title><meta property="og:title" content="Rust (programming language) - Wikipedia"><script type="application/ld+json">{{"@type":"Article","name":"Rust (programming language)","headline":"{headline}"}}</script><main><h1><span>Rust (programming language)</span></h1><p>Body.</p></main>"#
            )
        };
        let doc = extract_html(
            &wikipedia("memory-safe programming language without garbage collection"),
            None,
        )
        .unwrap();
        assert_eq!(doc.metadata["title"], "Rust (programming language)");
        // A headline that names the page's heading, or part of it, still leads.
        let doc = extract_html(
            &wikipedia("The Rust (programming language) - a short history"),
            None,
        )
        .unwrap();
        assert_eq!(
            doc.metadata["title"],
            "The Rust (programming language) - a short history"
        );
        // With no heading to confirm the other titles, the headline is used.
        let doc = extract_html(
            r#"<title>Site</title><script type="application/ld+json">{"headline":"Real headline"}</script><p>Body.</p>"#,
            None,
        )
        .unwrap();
        assert_eq!(doc.metadata["title"], "Real headline");
    }

    #[test]
    fn the_site_comes_from_either_meta_attribute_and_from_substack_pages() {
        let doc = extract_html(
            r#"<meta name="application-name" property="al:web:url" content="The Site"><main><p>Body.</p></main>"#,
            None,
        )
        .unwrap();
        assert_eq!(doc.metadata["site"], "The Site");
        let doc = extract_html(
            r#"<meta property="og:site_name" content="Journal"><main><p>Body.</p></main>"#,
            None,
        )
        .unwrap();
        assert_eq!(doc.metadata["site"], "Journal");
        // Meta tags lead; the JSON-LD publisher names a site they leave unnamed.
        let publisher = r#"<script type="application/ld+json">{"@type":"NewsArticle","publisher":{"@type":"Organization","name":"12 Grams  of Carbon"}}</script>"#;
        let doc = extract_html(&format!("{publisher}<main><p>Body.</p></main>"), None).unwrap();
        assert_eq!(doc.metadata["site"], "12 Grams of Carbon");
        let doc = extract_html(
            &format!(r#"<meta property="og:site_name" content="Journal">{publisher}<main><p>Body.</p></main>"#),
            None,
        )
        .unwrap();
        assert_eq!(doc.metadata["site"], "Journal");
        // A Substack article that names no site in its markup is `Substack`.
        let article = r#"<html><head><title>Post</title></head><body><div class="body markup"><p>Rendered article body.</p></div></body></html>"#;
        let doc = extract_html(article, Some("https://on.substack.com/p/post")).unwrap();
        assert_eq!(doc.metadata["site"], "Substack");
        let doc = extract_html(article, Some("https://example.org/p/post")).unwrap();
        assert!(!doc.metadata.contains_key("site"));
    }

    #[test]
    fn raw_tex_delimiters_keep_their_backslashes_brackets_and_underscores() {
        let markdown = article_markdown(
            r#"<h1>Attention</h1><p>Say we have a source sequence $\mathbf{x}$ of length $n$ and \(a_i = [b_1, b_2]\) here.</p>
            <p>$$\begin{aligned} \mathbf{x} &amp;= [x_1, x_2, \dots, x_n] \\ y &amp;= z \end{aligned}$$</p>
            <p>Display with brackets: \[ s_t = f(s_{t-1}) \] and more.</p>
            <p>\[F = m_a\]</p>"#,
        );
        assert!(
            markdown.contains(
                r"Say we have a source sequence $\mathbf{x}$ of length $n$ and $a_i = [b_1, b_2]$ here."
            ),
            "{markdown}"
        );
        assert!(
            markdown.contains(
                r"$$\begin{aligned} \mathbf{x} &= [x_1, x_2, \dots, x_n] \\ y &= z \end{aligned}$$"
            ),
            "{markdown}"
        );
        // A display expression inside a sentence is inline math; alone in its
        // element it is a block.
        assert!(
            markdown.contains("Display with brackets: $s_t = f(s_{t-1})$ and more."),
            "{markdown}"
        );
        assert!(
            markdown.contains("\n\n$$F = m_a$$\n\n") || markdown.ends_with("\n\n$$F = m_a$$"),
            "{markdown}"
        );
        assert!(
            !markdown.contains(r"\\mathbf") && !markdown.contains(r"x\_1"),
            "{markdown}"
        );
        // Multi-line display math is kept line by line.
        let markdown =
            article_markdown("<p>$$\n\\begin{aligned}\na &amp;= [x_1]\n\\end{aligned}\n$$</p>");
        assert!(
            markdown.contains("$$\\begin{aligned}\na &= [x_1]\n\\end{aligned}$$"),
            "{markdown}"
        );
        // In a table cell the math stays inline on one line.
        let markdown =
            article_markdown("<table><tr><th>A</th></tr><tr><td>$$\nx_1\n$$</td></tr></table>");
        assert!(markdown.contains("| $x_1$ |"), "{markdown}");
    }

    #[test]
    fn prices_code_and_loose_dollars_are_not_math() {
        let markdown = article_markdown(
            "<p>It costs $5 and $10, or US$5-$10; use $HOME/$USER and a_b_c, [x] too.</p>\
             <p>In code: <code>$a_b$ and \\(x\\)</code></p><pre><code>$$x_1$$\n\\[y\\]</code></pre>",
        );
        assert!(
            markdown.contains(
                r"It costs $5 and $10, or US$5-$10; use $HOME/$USER and a\_b\_c, \[x\] too."
            ),
            "{markdown}"
        );
        assert!(markdown.contains("`$a_b$ and \\(x\\)`"), "{markdown}");
        assert!(
            markdown.contains("```\n$$x_1$$\n\\[y\\]\n```"),
            "{markdown}"
        );
    }

    #[test]
    fn an_image_is_its_largest_srcset_candidate_and_lazy_sources_still_work() {
        let markdown = article_markdown(
            r#"<p><a href="https://m.test/orig/a.webp"><img src="https://m.test/t150/a.webp" srcset="https://m.test/t150/a.webp 150w, https://m.test/t1200/a.webp 1200w, https://m.test/t600/a.webp 600w" width="1920" alt="Unit"></a></p>
            <p><img src="data:image/gif;base64,R0lGODlh" data-src="https://m.test/lazy.jpg" alt="Lazy"></p>
            <p><img src="data:image/gif;base64,R0lGODlh" srcset="data:image/gif;base64,R0lGODlh 1x" data-src="https://m.test/lazy2.jpg" alt="Lazy2"></p>
            <p><img srcset="https://cdn.test/fetch/w_424,c_limit/a.png 424w, https://cdn.test/fetch/w_1456,c_limit/a.png 1456w" alt="Cdn"></p>
            <p><img src="https://m.test/1x.png" srcset="https://m.test/1x.png 1x, https://m.test/2x.png 2x" alt="Dense"></p>
            <p><img src="https://m.test/2560/b.jpg" srcset="https://m.test/400/b.jpg 400w, https://m.test/1920/b.jpg 1920w" width="2560" height="2560" alt="Bbc"></p>
            <p><img src="https://m.test/p640.png" srcset="https://m.test/p160.png 160w" width="640" alt="Portrait"></p>"#,
        );
        for expected in [
            // A `src` wider than the whole set is not downgraded.
            "![Bbc](https://m.test/2560/b.jpg)",
            "![Portrait](https://m.test/p640.png)",
            "[![Unit](https://m.test/t1200/a.webp)](https://m.test/orig/a.webp)",
            "![Lazy](https://m.test/lazy.jpg)",
            "![Lazy2](https://m.test/lazy2.jpg)",
            "![Cdn](https://cdn.test/fetch/w_1456,c_limit/a.png)",
            "![Dense](https://m.test/2x.png)",
        ] {
            assert!(markdown.contains(expected), "{expected}\n{markdown}");
        }
    }

    #[test]
    fn image_text_survives_line_breaks_and_an_empty_title_is_left_out() {
        let doc = extract_html(
            "<html><body><article><p><img src=\"https://m.test/a.svg\" alt=\"Two tables: the first table holds s1 on the\nstack, with its length (5), and a\n  pointer.\" title=\"\"></p><p><img src=\"https://m.test/b.webp\" alt=\"\" title=\"\"></p></article></body></html>",
            None,
        )
        .unwrap();
        let normal = crate::markdown::normalize(&doc.markdown);
        assert!(
            normal.contains("![Two tables: the first table holds s1 on the stack, with its length (5), and a pointer.](https://m.test/a.svg)"),
            "{normal}"
        );
        assert!(normal.contains("![](https://m.test/b.webp)"), "{normal}");
        assert!(!normal.contains("\"\""), "{normal}");
    }

    #[test]
    fn nested_identical_emphasis_is_written_once_on_a_page() {
        let markdown = article_markdown(
            "<p>He said <b><strong>no</strong></b>, <i><em>yes</em></i> and <strong>a <b>b</b></strong>.</p>",
        );
        assert!(
            markdown.contains("He said **no**, *yes* and **a b**."),
            "{markdown}"
        );
        assert!(!markdown.contains("****"), "{markdown}");
    }

    #[test]
    fn a_screen_reader_label_is_set_apart_from_the_text_it_labels() {
        let markdown = article_markdown(
            r#"<ul><li><span class="visually-hidden ssrcss-i2z2ig-VisuallyHidden e1">Published</span><span class="m"><span><time datetime="2026-10-01">1 hour ago</time></span></span></li></ul>
            <div><span class="sr-only">By</span><span>Katya Adler</span> <span class="sr-only">Role</span> Editor</div>
            <p>Read <span class="sr-only">about us</span> now <span class="sr-only">last</span></p>"#,
        );
        assert!(markdown.contains("Published 1 hour ago"), "{markdown}");
        assert!(
            markdown.contains("By Katya Adler Role Editor"),
            "{markdown}"
        );
        // A label followed by a space, or by nothing, gains no space of its own.
        assert!(markdown.ends_with("Read about us now last"), "{markdown}");
        assert!(!markdown.contains("  "), "{markdown}");
    }

    #[test]
    fn a_label_repeating_the_code_languages_name_is_left_out_and_brush_names_it() {
        let mdn = |label: &str, brush: &str| {
            format!(
                r#"<h2>Syntax</h2><div class="code-example"><div class="example-header"><span class="language-name">{label}</span></div><pre class="{brush} notranslate"><code>map(callbackFn)
map(callbackFn, thisArg)
</code></pre></div>"#
            )
        };
        let markdown = article_markdown(&mdn("js", "brush: js"));
        assert!(
            markdown.contains("## Syntax\n\n```js\nmap(callbackFn)\nmap(callbackFn, thisArg)\n```"),
            "{markdown}"
        );
        // Spellings of one language are the same label; other spellings of the
        // brush syntax name it too.
        for (label, brush) in [
            ("JavaScript", "brush:js"),
            ("JS", "brush: js; gutter: false"),
        ] {
            let markdown = article_markdown(&mdn(label, brush));
            assert!(markdown.contains("```js\nmap(callbackFn)"), "{markdown}");
            assert!(!markdown.contains(&format!("\n\n{label}\n")), "{markdown}");
        }
        // A label naming another language, or a heading, is the page's text.
        let markdown = article_markdown(&mdn("css", "brush: js"));
        assert!(markdown.contains("css\n\n```js"), "{markdown}");
        let markdown = article_markdown(
            "<h4>js</h4><pre class=\"brush: js\"><code>f()</code></pre><p>js</p><pre class=\"brush: js\"><code>g()</code></pre>",
        );
        assert!(markdown.contains("#### js\n\n```js\nf()"), "{markdown}");
        assert!(markdown.contains("js\n\n```js\ng()"), "{markdown}");
    }

    #[test]
    fn github_issue_interface_text_is_left_out_and_the_discussion_stays() {
        let page = r##"<html><body><div data-testid="issue-viewer-container"><h1>add support for other text encodings <span>#1</span></h1>
            <a href="https://github.com/login?return_to=https://github.com/o/r/issues/1">New issue</a>
            <div><span class="prc-TooltipV2-Tooltip-cYMVY CopyToClipboardButton-module__tooltip--Dq1IB" aria-label="Copy link" aria-hidden="true" popover="auto">Copy link</span><button>x</button></div>
            <span>Closed</span>
            <div class="header"><a href="https://github.com/BurntSushi">BurntSushi</a> opened <a href="#issue">on Sep 9, 2016</a>
            <span class="prc-TooltipV2-Tooltip-cYMVY" aria-hidden="true" popover="auto">Issue body actions</span></div>
            <div data-testid="issue-body-viewer"><p>Right now, ripgrep only supports reading UTF-8 encoded text.</p></div>
            <span>Reactions are currently unavailable</span>
            <div class="prc-Flash-Flash-3q4Aj SignedOutBanner-module__signedOutBanner--ycf6Y"><a href="/signup?return_to=https://github.com/o/r/issues/1">Sign up for free</a><span><strong> to join this conversation on GitHub.</strong> Already have an account? </span><a href="/login?return_to=https://github.com/o/r/issues/1">Sign in to comment</a></div>
            <p>Reactions are currently unavailable in a reply too.</p></div></body></html>"##;
        let markdown = extract_html(page, Some("https://github.com/o/r/issues/1"))
            .unwrap()
            .markdown;
        for kept in [
            "add support for other text encodings",
            "Closed",
            "BurntSushi",
            "opened",
            "on Sep 9, 2016",
            "Right now, ripgrep only supports reading UTF-8 encoded text.",
            // The labels are chrome only where they stand alone.
            "Reactions are currently unavailable in a reply too.",
        ] {
            assert!(markdown.contains(kept), "{kept}\n{markdown}");
        }
        for left_out in [
            "Copy link",
            "New issue",
            "Issue body actions",
            "Sign up for free",
            "join this conversation",
            "Sign in to comment",
        ] {
            assert!(!markdown.contains(left_out), "{left_out}\n{markdown}");
        }
        assert!(
            !markdown
                .lines()
                .any(|line| line == "Reactions are currently unavailable"),
            "{markdown}"
        );
    }

    #[test]
    fn canonical_homepage_is_not_used_for_article() {
        let html = r#"<link rel="alternate canonical" href="https://example.test"><p>Body</p>"#;
        assert!(
            !extract_html(html, Some("https://example.test/post"))
                .unwrap()
                .metadata
                .contains_key("canonical_url")
        );
        assert_eq!(
            extract_html(html, Some("https://example.test/"))
                .unwrap()
                .metadata["canonical_url"],
            "https://example.test"
        );
    }

    #[test]
    fn inline_quotes_and_source_lines_preserve_reference_spelling() {
        let doc = extract_html("<p>WWF's goal is to:\n    <q>Build a <em>better</em> future.</q>\n</p>\n<a href=\"https://example.com\">Example Link</a>\n<img src=\"https://example.com/test.jpg\" alt=\"Test Image\">", None).unwrap();
        assert_eq!(
            doc.markdown,
            "WWF's goal is to:\n\"Build a *better* future.\"\n\n[Example Link](https://example.com)\n![Test Image](https://example.com/test.jpg)"
        );
        assert_eq!(
            fragment("<p>One <q>two <q>three</q></q> four.</p>").unwrap(),
            "One \"two \"three\"\" four."
        );
    }

    #[test]
    fn urls_are_validated_without_rewriting_absolute_spelling() {
        let base = Url::parse("https://example.test/posts/one").unwrap();
        for value in [
            "https://EXAMPLE.test",
            "https://example.test/%2f?q=a%2Bb#part",
            "mailto:name@example.test",
            "tel:+12025550123",
        ] {
            assert_eq!(safe_url(value, Some(&base)).as_deref(), Some(value));
        }
        assert_eq!(
            safe_url("../two", Some(&base)).as_deref(),
            Some("https://example.test/two")
        );
        assert_eq!(
            safe_url("https://example.test/<tag>\"", None).as_deref(),
            Some("https://example.test/%3Ctag%3E%22")
        );
        for value in [
            "javascript:alert(1)",
            "data:text/html,body",
            "java\nscript:alert(1)",
            "https://exam\tple.test/",
        ] {
            assert!(safe_url(value, Some(&base)).is_none());
        }
    }

    #[test]
    fn row_headers_header_data_cells_and_spans_keep_their_columns() {
        let markdown = |html: &str| extract_html(html, None).unwrap().markdown;
        let row_headers = markdown(
            "<main><table><thead><tr><th></th><th>2023</th><th>2024</th></tr></thead>\
             <tbody><tr><th>Revenue</th><td>10</td><td>12</td></tr><tr><th>Cost</th><td>7</td><td>8</td></tr></tbody></table></main>",
        );
        assert!(
            row_headers.contains(
                "|  | 2023 | 2024 |\n| --- | --- | --- |\n| Revenue | 10 | 12 |\n| Cost | 7 | 8 |"
            ),
            "{row_headers}"
        );
        // Key/value rows without a header row get an empty one; the caption
        // stays a paragraph above the table.
        let infobox = markdown(
            "<main>Lead<table><caption>Tool</caption><tr><td colspan=\"2\">Logo</td></tr>\
             <tr><th>Developer</th><td>Dynalist</td></tr><tr><td></td><td></td></tr><tr><th>License</th><td>Proprietary</td></tr></table></main>",
        );
        assert!(
            infobox.contains("Lead\n\nTool\n\n|  |  |\n| --- | --- |\n| Logo |  |\n| Developer | Dynalist |\n| License | Proprietary |"),
            "{infobox}"
        );
        let spans = markdown(
            "<main><table><thead><tr><th rowspan=\"2\">Region</th><th colspan=\"2\">Sales</th><th>Total</th></tr><tr><th>Q1</th><th>Q2</th></tr></thead>\
             <tbody><tr><th>North</th><td rowspan=\"2\">1</td><td>2</td><td>3</td></tr><tr><th>South</th><td>4</td><td>5</td></tr></tbody></table></main>",
        );
        assert!(
            spans.contains("| Region | Sales |  | Total |\n| --- | --- | --- | --- |\n|  | Q1 | Q2 |  |\n| North | 1 | 2 | 3 |\n| South |  | 4 | 5 |"),
            "{spans}"
        );
        let first_row = markdown(
            "<main><table><tr><th>Key</th><th>Value</th></tr><tr><th>Speed</th><td>fast</td></tr></table></main>",
        );
        assert!(
            first_row.contains("| Key | Value |\n| --- | --- |\n| Speed | fast |"),
            "{first_row}"
        );
        // A table of short cells without header cells is data: its first
        // row is the header.
        let data = markdown(
            "<main><table><tr><td>Name</td><td>Count</td></tr><tr><td>alpha</td><td>1</td></tr></table></main>",
        );
        assert!(
            data.contains("| Name | Count |\n| --- | --- |\n| alpha | 1 |"),
            "{data}"
        );
        // A leading spacer row and a spacer column show nothing.
        let spacers = markdown(
            "<main><table><tr><td><img src=\"s.gif\" width=\"1\" height=\"1\"></td></tr><tr><td>A</td><td></td><td>B</td></tr><tr><td>C</td><td> </td><td>D</td></tr></table></main>",
        );
        assert!(
            spacers.contains("| A | B |\n| --- | --- |\n| C | D |"),
            "{spacers}"
        );
        // A layout table is written as its cells' content, so a data table
        // inside it is still a table.
        let prose = "A long column of prose that lays out the page beside a sidebar. ".repeat(4);
        let layout = markdown(&format!(
            "<main><table><tr><td><p>{prose}</p><table><tr><td>Ctrl+S</td><td>Save</td></tr><tr><td>Ctrl+O</td><td>Open</td></tr></table></td><td><p>Links</p><p>More links</p></td></tr></table></main>"
        ));
        assert!(
            layout.contains("sidebar.\n\n| Ctrl+S | Save |\n| --- | --- |\n| Ctrl+O | Open |\n\nLinks\n\nMore links"),
            "{layout}"
        );
        assert_eq!(layout.matches("Ctrl+S").count(), 1, "{layout}");
        // Mostly empty or single-row tables are layout too.
        for table in [
            "<table><tr><td>1.</td><td></td><td></td></tr><tr><td></td><td></td><td>Item</td></tr></table>",
            "<table><tr><td>Only</td><td>row</td></tr></table>",
            "<table role=\"presentation\"><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>",
        ] {
            let plain = markdown(&format!("<main>{table}</main>"));
            assert!(!plain.contains('|'), "{plain}");
        }
        // So is a table with block structure or long text in a cell.
        let long = "word ".repeat(41);
        for cell in [
            "<table><tr><td>x</td></tr><tr><td>y</td></tr></table>".to_owned(),
            "<p>one</p><p>two</p>".to_owned(),
            "<ul><li>item</li></ul>".to_owned(),
            "<h3>Heading</h3>".to_owned(),
            long,
        ] {
            let table = format!(
                "<table><tr><td>a</td><td>{cell}</td></tr><tr><td>c</td><td>d</td></tr></table>"
            );
            let plain = markdown(&format!("<main>{table}</main>"));
            assert!(!plain.contains('|'), "{plain}");
        }
    }

    #[test]
    fn in_page_tables_of_contents_and_mediawiki_edit_links_are_left_out() {
        let markdown = |html: &str| extract_html(html, None).unwrap().markdown;
        let page = markdown(
            r##"<main><h1>Guide</h1><p>Intro.</p><hr><ul><li><a href="#one">1. One</a><ul><li><a href="#two">Two</a></li></ul></li><li>3. <a href="#%C3%A9t%C3%A9">Été</a></li><li><a href="#three">Three</a></li></ul><hr>
            <div class="box"><div><h2>Contents</h2><span><label></label></span></div><div><ol><li><a href="#one">One</a></li><li><a href="#two">Two</a></li><li><a href="#été">Été</a></li></ol></div></div>
            <div><p>These are the parts of this long guide in order.</p><ul><li><a href="#one">One</a></li><li><a href="#two">Two</a></li><li><a href="#three">Three</a></li></ul></div>
            <div><p><a href="/all">All</a></p><ul><li><a href="#one">One</a></li><li><a href="#two">Two</a></li><li><a href="#three">Three</a></li></ul></div>
            <ul><li><a href="#one">One</a> explains the setup</li><li><a href="#two">Two</a></li><li><a href="#été">Été</a></li></ul>
            <ul><li><a href="#one">One</a></li><li><a href="#two">Two</a></li><li><a href="#note">Note</a></li></ul>
            <ol><li><a href="#one">One</a></li><li><a href="#two">Two</a></li></ol>
            <h2 id="one">One</h2><p>First.</p><section id="two"><h2>Two</h2><p>Second.</p></section><a name="été"></a><h2>Été</h2><p id="note">Third.</p><h3><span id="three">Three</span></h3></main>"##,
        );
        assert!(
            page.contains("Intro.\n\nThese are the parts of this long guide in order.\n\n[All](/all)\n\n* [One](#one) explains the setup"),
            "{page}"
        );
        assert!(page.contains("* [Note](#note)"), "{page}");
        assert!(
            !page.contains("Contents") && !page.contains("---"),
            "{page}"
        );
        assert_eq!(page.matches("[Two](#two)").count(), 3, "{page}");
        let wiki = markdown(
            r#"<main><h1>Obsidian</h1><div id="siteSub">From Wikipedia, the free encyclopedia</div><div id="contentSub"><span>(Redirected from Obs)</span></div><h2><span class="mw-headline" id="History">History</span><span class="mw-editsection"><span class="mw-editsection-bracket">[</span><a href="/w/index.php?action=edit&amp;section=1">edit</a><span class="mw-editsection-bracket">]</span></span></h2><p>Text.</p></main>"#,
        );
        assert!(wiki.contains("# Obsidian\n\n## History\n\nText."), "{wiki}");
    }

    /// `words` words of running text.
    fn prose(words: usize) -> String {
        (0..words)
            .map(|index| format!("word{index}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn furniture_beside_the_body_is_left_out_of_a_full_page_only() {
        let page = format!(
            r##"<main><div class="post"><h1>Title</h1><p>{}</p><p>{}</p><p>{}</p></div>
            <div><p>Jane writes about databases.</p><a href="/jane">Jane</a></div>
            <section><h3>Related</h3><article><h3><a href="/a">Another post</a></h3></article></section></main>"##,
            prose(40),
            prose(40),
            prose(40)
        );
        let full = extract_html(&page, None).unwrap().markdown;
        assert!(
            full.contains("word39") && full.contains("# Title"),
            "{full}"
        );
        for gone in ["Jane", "Related", "Another post"] {
            assert!(!full.contains(gone), "{gone}: {full}");
        }
        // Books and emails keep every block.
        let book = fragment(&page).unwrap();
        for kept in ["Jane", "Related", "Another post"] {
            assert!(book.contains(kept), "{kept}: {book}");
        }
    }

    #[test]
    fn a_web_mail_thread_folds_quoted_history_on_a_full_page_only() {
        let page = r#"<div role="main"><h2 class="hP">Kickoff</h2>
            <div class="adn" data-message-id="1"><span class="gD">Alex Rivera</span><div class="a3s aiL"><div>Are Tuesdays good for you?</div></div></div>
            <div class="adn" data-message-id="2"><span class="gD">Jane Doe</span><div class="a3s aiL"><div>Tuesdays work well.</div>
            <div class="gmail_quote"><div class="gmail_attr">On Wed, Alex Rivera wrote:</div><blockquote class="gmail_quote">Are Tuesdays good for you?</blockquote></div>
            <div class="adL"><div>-- Jane Doe, Example Corp</div></div></div></div></div>"#;
        let full = extract_html(page, None).unwrap().markdown;
        assert_eq!(
            full,
            "## Kickoff\n\nAlex Rivera\n\nAre Tuesdays good for you?\n\nJane Doe\n\nTuesdays work well."
        );
        // A mail fragment keeps the history it quotes.
        let mail = fragment(page).unwrap();
        for kept in [
            "On Wed, Alex Rivera wrote:",
            "> Are Tuesdays",
            "Example Corp",
        ] {
            assert!(mail.contains(kept), "{kept}: {mail}");
        }
    }

    #[test]
    fn page_marks_beside_an_article_are_left_out_of_a_full_page_only() {
        let page = format!(
            r##"<main><div data-block="nav"><ul><li><a href="/">Home</a></li><li><a href="/posts">Posts</a></li><li>A job in an unexpected industry</li></ul></div>
            <article><h1>A job in an unexpected industry</h1><div>8 min read</div><p>{}</p>
            <div class="top-comment"><p>Great story!</p><a href="#comments">View all comments</a></div>
            <p>{}</p><div><a>13 Likes</a></div><p>{}</p></article></main>"##,
            prose(40),
            prose(40),
            prose(40)
        );
        let full = extract_html(&page, None).unwrap().markdown;
        assert!(
            full.starts_with("# A job in an unexpected industry\n\nword0"),
            "{full}"
        );
        for gone in [
            "Home",
            "Posts",
            "min read",
            "Great story",
            "View all",
            "Likes",
        ] {
            assert!(!full.contains(gone), "{gone}: {full}");
        }
        let book = fragment(&page).unwrap();
        for kept in ["Home", "Posts", "8 min read", "Great story", "13 Likes"] {
            assert!(book.contains(kept), "{kept}: {book}");
        }
    }

    #[test]
    fn a_layout_table_loses_a_furniture_cell_and_keeps_its_article_cell() {
        let page = format!(
            r##"<table><tr><td><p>{}</p><p>{}</p><p>{}</p></td>
            <td><b>Links:</b><br><a href="/a">Friend</a><br><a href="/b">Another</a></td></tr></table>"##,
            prose(40),
            prose(40),
            prose(40)
        );
        let full = extract_html(&page, None).unwrap().markdown;
        assert!(full.contains("word39"), "{full}");
        assert!(
            !full.contains("Friend") && !full.contains("Links"),
            "{full}"
        );
        let book = fragment(&page).unwrap();
        assert!(book.contains("Friend") && book.contains("Links"), "{book}");
        // A whole row of furniture below the article's row.
        let rows = format!(
            r##"<table><tr><td><p>{}</p><p>{}</p><p>{}</p></td></tr>
            <tr><td><a href="/a">Friend</a> <a href="/b">Another</a></td></tr></table>"##,
            prose(40),
            prose(40),
            prose(40)
        );
        let full = extract_html(&rows, None).unwrap().markdown;
        assert!(
            full.contains("word39") && !full.contains("Friend"),
            "{full}"
        );
    }

    #[test]
    fn sections_without_content_are_dropped_from_a_full_page() {
        let page = extract_html(
            "<main><h1>Title</h1><p>Intro text.</p><h2>Hidden</h2><p hidden>Gone.</p><h2>Full</h2><p>Body.</p><h2>Empty parent</h2><h3>Empty child</h3><h2>References</h2><h2>Last</h2></main>",
            None,
        )
        .unwrap()
        .markdown;
        assert_eq!(
            page,
            "# Title\n\nIntro text.\n\n## Full\n\nBody.\n\n## References"
        );
        // A chapter of a book may be a title alone.
        assert!(fragment("<h2>Chapter</h2>").unwrap().contains("## Chapter"));
        assert_eq!(
            drop_empty_sections("# T\n\n## A\n\n### B\n\n## C\n\ntext\n\n### D\n").trim_end(),
            "# T\n\n## C\n\ntext"
        );
        assert_eq!(
            drop_empty_sections("## A\n\n### B\n\ntext\n\n```\n# not a heading\n```\n\n## Z")
                .trim_end(),
            "## A\n\n### B\n\ntext\n\n```\n# not a heading\n```"
        );
        // A document's title stays over a section of its own level; later ones go.
        assert_eq!(
            drop_empty_sections("# Title\n\n# Part\n\ntext\n\n# Empty\n\n# Last\n\nmore"),
            "# Title\n\n# Part\n\ntext\n\n# Last\n\nmore"
        );
        assert_eq!(drop_empty_sections("## A\n\n## B\n\ntext"), "## B\n\ntext");
        // Comment lines of a code block are not headings, whatever follows them.
        let code = "text\n\n```sh\n# one\n# two\n```\n\n~~~\n## three\n## four\n~~~";
        assert_eq!(drop_empty_sections(code), code);
    }

    #[test]
    fn a_headings_link_to_its_own_page_and_empty_bullets_are_left_out_of_a_full_page() {
        let html = r##"<main><h2><a href="#one">One</a></h2><p>a</p><h2 id="two">Two<a class="anchor" href="#two">#</a></h2><p>b</p><h2><a href="/blog/x">Post</a></h2><p>c</p>
            <ul><li><a href="/share"><svg viewBox="0 0 1 1"></svg></a></li><li>Real</li><li><button>Share</button></li></ul></main>"##;
        let full = extract_html(html, None).unwrap().markdown;
        assert_eq!(
            full,
            "## One\n\na\n\n## Two\n\nb\n\n## [Post](/blog/x)\n\nc\n\n* Real"
        );
        let book = fragment(html).unwrap();
        assert!(book.contains("## [One](#one)"), "{book}");
        assert!(book.contains("[#](#two)"), "{book}");
    }

    #[test]
    fn framework_hide_classes_hide_parts_of_a_full_page_only() {
        let html = r#"<p>Kept <span class="hidden">gone</span> <span class="md:hidden">mobile</span> <span class="not-machine:hidden">machine</span> <span class="hidden md:inline">desktop</span> <span class="[&amp;_.x]:hidden">arbitrary</span> <span class="isHidden-vzcyV0">module</span> <span class="is-hidden-abc">dashed</span> <span class="invisible">https://</span><span class="hidden"><math><mi>x</mi></math></span> end.</p>"#;
        let page = extract_html(&format!("<main>{html}</main>"), None)
            .unwrap()
            .markdown;
        for kept in ["Kept", "desktop", "arbitrary", "$x$", "end."] {
            assert!(page.contains(kept), "{kept}: {page}");
        }
        for gone in ["gone", "mobile", "machine", "module", "dashed", "https://"] {
            assert!(!page.contains(gone), "{gone}: {page}");
        }
        // A book or an email comes without the site's style sheet.
        let fragment = fragment(html).unwrap();
        assert!(
            fragment.contains("gone") && fragment.contains("mobile"),
            "{fragment}"
        );
    }

    #[test]
    fn links_around_blocks_are_written_on_the_block_that_names_them() {
        let markdown = extract_html(
            r#"<article><a href="/post"><img src="cover.png" alt="Cover"><h3>Title</h3><p>Summary text.</p></a>
            <a href="/card"><div><div>Card name</div><div>Card text</div></div></a>
            <a href="/story"><div>Category</div><h2>Story</h2></a>
            <a href="/media"><div class="thumb"><img src="t.png" alt="Thumb"></div><div>Media title</div></a>
            <p><a href="/inline">inline</a> stays.</p></article>"#,
            None,
        )
        .unwrap()
        .markdown;
        assert!(
            markdown.contains("![Cover](cover.png)\n\n### [Title](/post)\n\nSummary text."),
            "{markdown}"
        );
        assert!(
            markdown.contains("[Card name](/card)\n\nCard text"),
            "{markdown}"
        );
        assert!(markdown.contains("[inline](/inline) stays."), "{markdown}");
        assert!(
            markdown.contains("Category\n\n## [Story](/story)"),
            "{markdown}"
        );
        assert!(
            markdown.contains("![Thumb](t.png)\n\n[Media title](/media)"),
            "{markdown}"
        );
        assert!(
            !markdown.contains("[###") && !markdown.contains("](/post)\n\nSummary text.]"),
            "{markdown}"
        );
    }

    #[test]
    fn saved_pages_resolve_links_against_their_own_address() {
        let body = r##"<main><p><a href="/about">About</a> <a href="next.html">Next</a> <a href="#part">Part</a> <a href="//cdn.example.net/file.pdf">File</a> <img src="page_files/a.png" alt="A"> <img src="//img.example.net/b.png" alt="B"></p></main>"##;
        let page = |head: &str| {
            extract_html(
                &format!("<html><head>{head}</head><body>{body}</body></html>"),
                None,
            )
            .unwrap()
            .markdown
        };
        let canonical = page(r#"<link rel="canonical" href="https://example.org/blog/post/">"#);
        for expected in [
            "[About](https://example.org/about)",
            "[Next](https://example.org/blog/post/next.html)",
            "[Part](#part)",
            "[File](https://cdn.example.net/file.pdf)",
            "![A](page_files/a.png)",
            "![B](https://img.example.net/b.png)",
        ] {
            assert!(canonical.contains(expected), "{expected}: {canonical}");
        }
        // A home-page canonical does not say which directory the page is in.
        let home = page(r#"<link rel="canonical" href="https://example.org/">"#);
        assert!(
            home.contains("[About](https://example.org/about)")
                && home.contains("[Next](next.html)"),
            "{home}"
        );
        let base = page(
            r#"<base href="https://docs.example.org/guide/"><link rel="canonical" href="https://example.org/x/">"#,
        );
        assert!(
            base.contains("[Next](https://docs.example.org/guide/next.html)"),
            "{base}"
        );
        // Without a saved address, and in fragments, links stay as written.
        assert!(page("").contains("[Next](next.html)"));
        let fragment = fragment(
            r#"<link rel="canonical" href="https://example.org/a/"><a href="next.html">Next</a>"#,
        )
        .unwrap();
        assert!(fragment.contains("[Next](next.html)"), "{fragment}");
    }

    #[test]
    fn links_that_show_nothing_are_left_out() {
        let markdown = extract_html(
            r#"<main><p>See <a href="/vote"></a><a href="/share"><i class="icon"></i> </a><a href="/pixel"><img src="p.gif" width="1" height="1"></a>the <a href="/site">site</a> and <a href="/home"><img src="logo.png" alt="Logo"></a>.</p></main>"#,
            None,
        )
        .unwrap()
        .markdown;
        assert!(
            markdown.contains("See the [site](/site) and [![Logo](logo.png)](/home)."),
            "{markdown}"
        );
        assert!(!markdown.contains("[]("), "{markdown}");
    }

    #[test]
    fn tables_use_the_reference_compact_row_spelling() {
        let doc = extract_html(
            "<main><table><thead><tr><th>Station</th><th>Count</th></tr></thead>\
             <tbody><tr><td>North-12</td><td>7</td></tr><tr><td>a|b</td><td></td></tr></tbody></table>\
             <pre><code>| keep   | padded |\n| ------ | ------ |</code></pre></main>",
            None,
        )
        .unwrap();
        assert!(
            doc.markdown
                .contains("| Station | Count |\n| --- | --- |\n| North-12 | 7 |\n| a&#124;b |  |"),
            "{}",
            doc.markdown
        );
        assert!(
            doc.markdown.contains("| keep   | padded |"),
            "{}",
            doc.markdown
        );
        assert_eq!(
            compact_table("\n\n| only | rows |\n| no | header |\n"),
            None
        );
        assert_eq!(compact_table("<table><tr><td>x</td></tr></table>"), None);
    }

    #[test]
    fn inline_image_data_keeps_reference_placeholder_without_payload() {
        let doc = extract_html(
            "<main><p>Before</p><p><img alt=\"blue box\" src=\"data:image/png;base64,iVBORw0KGgo=\"></p>\
             <p><img alt=\"vector\" src=\"DATA:image/svg+xml;base64,PHN2Zz4=\"></p>\
             <p><img alt=\"page\" src=\"data:text/html,<b>x</b>\"></p><p>After</p></main>",
            None,
        )
        .unwrap();
        assert!(
            doc.markdown
                .contains("![blue box](data:image/png;base64...)"),
            "{}",
            doc.markdown
        );
        assert!(!doc.markdown.contains("iVBOR"));
        assert!(!doc.markdown.contains("vector") && !doc.markdown.contains("svg"));
        assert!(!doc.markdown.contains("text/html"));
        assert_eq!(
            image_source("data:image/png;base64", None).as_deref(),
            Some("data:image/png;base64...")
        );
        assert!(image_source(&format!("data:{}", "a".repeat(300)), None).is_none());
        assert!(image_source("data:image/png;x=(1),AAAA", None).is_none());
        // Links never accept data URLs, only image sources do.
        assert!(safe_url("data:image/png;base64,AAAA", None).is_none());
    }

    #[test]
    fn block_boundaries_code_and_chemical_notation_survive() {
        let doc = extract_html("<main><div>First</div><div>Second</div><p>H<sub>2</sub>O and 10<sup>n</sup>.</p><pre><code>one\n  two\n\nthree</code></pre><code style=\"white-space: pre\">a\n  b</code></main>", None).unwrap();
        assert!(
            doc.markdown
                .starts_with("First\n\nSecond\n\nH<sub>2</sub>O and 10<sup>n</sup>.")
        );
        assert!(doc.markdown.contains("one\n  two\n\nthree"));
        assert!(doc.markdown.contains("```\na\n  b\n```"));
        assert!(!doc.markdown.contains("markitai-soft-break"));
    }

    #[test]
    fn candidate_inside_hidden_ancestor_cannot_become_article() {
        let doc = extract_html("<div hidden><article><p>Hidden text that outweighs the visible article by length.</p></article></div><main><p>Visible</p></main>", None).unwrap();
        assert_eq!(doc.markdown, "Visible");
    }

    #[test]
    fn css_custom_properties_preserve_visible_article_content() {
        let doc = extract_html(r#"<div style="--footer-display: none; --graph-controls-display: none;"><article><h1>开发者</h1><p>如果你熟悉 TypeScript 或 CSS，可以开发插件。</p></article></div>"#, None).unwrap();
        assert!(doc.markdown.contains("开发者"));
        assert!(doc.markdown.contains("可以开发插件"));
    }

    #[test]
    fn hidden_style_checks_property_tokens_not_embedded_strings_or_comments() {
        for style in [
            "--foo-display:none; --foo-visibility:hidden",
            "content: 'display:none; visibility:hidden'",
            r#"content: "escaped \"; display:none"; color: red"#,
            "/* display:none; visibility:hidden */ color:red",
            "background:url('x;display:none'); color:red",
            "--theme: { display:none; visibility:hidden }; color:red",
            "display:'none'; visibility:\"hidden\"",
            "dis/**/play:none; visibility:hiddenish",
            "content: 'unfinished; display:none",
            "color:red; /* unfinished comment; display:none",
        ] {
            assert!(!hidden_inline_style(style), "unexpectedly hidden: {style}");
        }
        for style in [
            "DISPLAY : NoNe",
            "display: /* explanation */ none ! IMPORTANT",
            "/* declaration */ visibility: hidden",
            "visibility:collapse",
            "content:'display:block;'; display:none",
            "--theme:{display:block;}; visibility:hidden",
        ] {
            assert!(hidden_inline_style(style), "expected hidden: {style}");
        }
    }

    #[test]
    fn inline_hidden_properties_follow_order_and_importance() {
        for style in [
            "display:none; display:block",
            "visibility:hidden!important; visibility:visible!important",
            "display:none; display:block!important; display:none",
        ] {
            assert!(!hidden_inline_style(style), "unexpectedly hidden: {style}");
        }
        for style in [
            "display:block; display:none",
            "display:none!important; display:block",
            "visibility:hidden; display:block",
            "display:none; display:'block'",
        ] {
            assert!(hidden_inline_style(style), "expected hidden: {style}");
        }
        let doc = extract_html(r#"<main><p style="content:'display:none'">Visible</p><p style="display:/* comment */none">Hidden</p><p hidden style="display:block">Also hidden</p></main>"#, None).unwrap();
        assert_eq!(doc.markdown, "Visible");
    }

    #[test]
    fn mathjax_tex_sources_replace_only_their_duplicate_previews() {
        let doc = extract_html(r#"<main><p>Energy <span class="MathJax_Preview">duplicate preview</span><span class="MathJax">duplicate glyphs</span><script type="math/tex">E_u</script> remains.</p><script type="math/tex; mode=display">\frac{a}{b}</script><script type="application/javascript">alert('bad')</script><script type="math/tex-evil">not math</script><p><span class="MathJax">Visible unannotated text</span></p></main>"#, None).unwrap();
        assert!(doc.markdown.contains("Energy $E_u$ remains."));
        assert!(doc.markdown.contains(r"$$\frac{a}{b}$$"));
        assert!(doc.markdown.contains("Visible unannotated text"));
        for text in ["duplicate", "alert", "not math"] {
            assert!(!doc.markdown.contains(text));
        }
    }

    #[test]
    fn callouts_and_alerts_become_obsidian_blockquotes_like_the_reference() {
        let doc = extract_html(
            r#"<main><p>Before.</p>
            <div class="callout is-collapsible is-collapsed" data-callout="faq"><div class="callout-title"><div class="callout-title-inner">Is <b>this</b> foldable?</div></div><div class="callout-content" style="display:none"><p>Yes, hidden when collapsed.</p><p style="display:none">Secret aside.</p></div></div>
            <div class="markdown-alert markdown-alert-warning"><p class="markdown-alert-title">Warning</p><p>Check the version.</p></div>
            <div class="alert alert-dismissible alert-success"><h4 class="alert-heading">Well done!</h4>Operation completed.</div>
            <aside class="Callout-Tip">Use shortcuts.</aside>
            <div class="admonition tip"><p class="admonition-title">Helpful <em>tip</em></p><div class="admonition-content"><p>Save time.</p></div></div>
            <div class="callout" data-callout="bad type!"><div class="callout-content"><p>Fallback type.</p></div></div>
            <hr><ul><li>One</li><li>Two</li></ul><ol><li>First</li></ol><p>After.</p></main>"#,
            None,
        )
        .unwrap();
        let text = &doc.markdown;
        // Obsidian: title text joined by spaces, fold marker, collapsed body kept,
        // hidden elements inside the body still removed.
        assert!(
            text.contains("> [!faq]- Is this foldable?\n>\n> Yes, hidden when collapsed."),
            "{text}"
        );
        assert!(!text.contains("Secret aside"), "{text}");
        // GitHub, Bootstrap: the title element is replaced by the default title.
        assert!(
            text.contains("> [!warning] Warning\n>\n> Check the version."),
            "{text}"
        );
        assert_eq!(text.matches("Warning").count(), 1, "{text}");
        assert!(
            text.contains("> [!success] Success\n>\n> Operation completed."),
            "{text}"
        );
        assert!(!text.contains("Well done"), "{text}");
        // Asides match case-insensitively; admonitions keep their title.
        assert!(text.contains("> [!tip] Tip\n>\n> Use shortcuts."), "{text}");
        assert!(
            text.contains("> [!tip] Helpful tip\n>\n> Save time."),
            "{text}"
        );
        assert!(
            text.contains("> [!note] Note\n>\n> Fallback type."),
            "{text}"
        );
        // The reference's rule and list spelling.
        assert!(
            text.contains("\n\n---\n\n* One\n* Two\n\n1. First\n\nAfter."),
            "{text}"
        );
    }

    #[test]
    fn web_pages_report_the_reference_word_count() {
        assert_eq!(count_words("Hello, world! **Bold** text_here 42"), 5);
        assert_eq!(count_words("中文 字数 and English"), 6);
        assert_eq!(count_words("  "), 0);
        let page = extract_html(
            "<main><h1>Title</h1><p>Two words.</p></main>",
            Some("https://example.test/a"),
        )
        .unwrap();
        assert_eq!(page.metadata["word_count"], 3);
        let file = extract_html("<main><h1>Title</h1><p>Two words.</p></main>", None).unwrap();
        assert!(!file.metadata.contains_key("word_count"));
    }

    #[test]
    fn declarative_shadow_roots_are_content_while_other_templates_stay_inert() {
        let doc = extract_html(
            r#"<html><body><main><h1>Shadow</h1>
            <div><TEMPLATE ShadowRootMode="OPEN"><p>Outer shadow text.</p><section><template shadowrootmode='closed'><p>Inner closed text.</p></template></section></TEMPLATE></div>
            <div><template shadowroot=open><p>Legacy attribute text.</p></template></div>
            <template><p>Inert template text.</p></template>
            <template shadowrootmode="none"><p>Unknown mode text.</p></template>
            <p>Regular text.</p></main></body></html>"#,
            None,
        )
        .unwrap();
        let text = &doc.markdown;
        for kept in [
            "Outer shadow text.",
            "Inner closed text.",
            "Legacy attribute text.",
            "Regular text.",
        ] {
            assert!(text.contains(kept), "{kept}: {text}");
        }
        for inert in ["Inert template text", "Unknown mode text"] {
            assert!(!text.contains(inert), "{inert}: {text}");
        }
        assert_eq!(flatten_shadow_roots("<p>plain</p>"), "<p>plain</p>");
        // The attribute is found in any ASCII case, and only in ASCII.
        assert_eq!(
            flatten_shadow_roots("<template SHADOWROOT=open><p>x</p></template>"),
            "<p>x</p>"
        );
        let long_s = "<template \u{17f}hadowrootmode=open><p>x</p></template>";
        assert_eq!(flatten_shadow_roots(long_s), long_s);
        // An unclosed shadow template is left for the parser, unchanged.
        let unclosed = r#"<template shadowrootmode="open"><p>Never closed"#;
        assert_eq!(flatten_shadow_roots(unclosed), unclosed);
        // Hoisting is bounded at ten nested levels.
        let deep = format!(
            "{}core{}",
            r#"<template shadowrootmode="open">"#.repeat(12),
            "</template>".repeat(12)
        );
        let flattened = flatten_shadow_roots(&deep);
        assert_eq!(flattened.matches("<template").count(), 2);
    }

    #[test]
    fn tex_image_services_become_math_like_the_reference() {
        let doc = extract_html(
            r#"<main><p>If <img src="https://s0.wp.com/latex.php?latex=%5Clambda%2C+%5Cmu+%5Cin+%5Cmathbb%7BR%7D&amp;bg=ffffff" alt="&#92;lambda, &#92;mu &#92;in &#92;mathbb{R}" class="latex"> and <img src="https://s0.wp.com/latex.php?latex=A%2C+B&amp;bg=ffffff" alt="A, B"> hold, then <img src="https://latex.codecogs.com/svg.image?%5Cfrac%7Ba%7D%7Bb%7D"> and <img src="https://i.upmath.me/svg/%5Csqrt%7Bx%7D"> and <img src="/eq.svg" alt="\operatorname{fn}(x)"> and <img src="/cat.png" alt="A cat">.</p><p><img src="https://chart.googleapis.com/chart?cht=tx&amp;chl=%5Csum_i+x_i"></p><p>Aligned: <img src="/a.svg" alt="\begin{align} a &amp;= b \end{align}"></p></main>"#,
            None,
        )
        .unwrap();
        let text = &doc.markdown;
        // A named parameter, then the whole query, then an escaped path
        // segment, then TeX-looking alt text.
        assert!(
            text.contains(r"If $\lambda, \mu \in \mathbb{R}$ and"),
            "{text}"
        );
        assert!(
            text.contains(r"then $\frac{a}{b}$ and $\sqrt{x}$ and $\operatorname{fn}(x)$ and"),
            "{text}"
        );
        // Without a TeX command the image stays an image.
        assert!(
            text.contains("![A, B](https://s0.wp.com/latex.php?latex=A%2C+B&bg=ffffff)"),
            "{text}"
        );
        assert!(text.contains("![A cat](/cat.png)"), "{text}");
        // Display math: the sole child of a paragraph, or an environment.
        assert!(text.contains(r"$$\sum_i x_i$$"), "{text}");
        assert!(
            text.contains(r"$$\begin{align} a &= b \end{align}$$"),
            "{text}"
        );
    }

    #[test]
    fn katex_annotations_and_data_sources_are_emitted_once() {
        let doc = extract_html(r#"<main><p>A <span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>x</mi><mo>+</mo><mi>y</mi></mrow><annotation encoding="application/x-tex">x+y</annotation></semantics></math></span><span class="katex-html">duplicate</span></span> term.</p><span class="katex-display"><span class="katex"><math><semantics><mi>z</mi><annotation encoding="application/x-tex">z^2</annotation></semantics></math><span class="katex-html">other duplicate</span></span></span><p><span class="math-inline" data-math="\forall x"><span class="katex">visual x</span></span> and <span class="hurmet-tex" data-entry="\vec{F}"><math><mi>F</mi></math></span>.</p></main>"#, None).unwrap();
        assert!(doc.markdown.contains("A $x+y$ term."));
        assert_eq!(doc.markdown.matches("z^2").count(), 1);
        assert!(doc.markdown.contains("$$z^2$$"));
        assert!(doc.markdown.contains(r"$\forall x$ and $\vec{F}$."));
        assert!(!doc.markdown.contains("duplicate"));
        assert!(!doc.markdown.contains("visual"));
    }

    #[test]
    fn hidden_assistive_math_is_recovered_inside_known_visual_wrappers() {
        let doc = extract_html(r#"<p>Quadratic <span class="mwe-math-element"><span style="display:none"><math alttext="ax^2+bx+c=0"><mi>fallback</mi></math></span><img src="https://example.test/equation.svg" alt="ax^2+bx+c=0"></span>.</p><mjx-container display="true"><svg><path></path></svg><mjx-assistive-mml style="display:none"><math><msub><mi>r</mi><mtext>perf</mtext></msub></math></mjx-assistive-mml></mjx-container>"#, None).unwrap();
        assert!(doc.markdown.contains("Quadratic $ax^2+bx+c=0$."));
        assert!(doc.markdown.contains(r"$$r_{\text{perf}}$$"));
        assert!(!doc.markdown.contains("equation.svg"));
        assert!(!doc.markdown.contains("fallback"));
    }

    #[test]
    fn nested_presentation_mathml_keeps_fraction_roots_scripts_and_accents() {
        let doc = extract_html(r#"<math display="block"><mrow><mfrac><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup><mroot><mi>y</mi><mn>3</mn></mroot></mfrac><mo>+</mo><msqrt><mrow><mi>α</mi><mo>×</mo><mi>β</mi></mrow></msqrt><mo>=</mo><mover><mi>F</mi><mo>→</mo></mover></mrow></math>"#, None).unwrap();
        assert_eq!(
            doc.markdown,
            r"$$\frac{x_{1}^{2}}{\sqrt[3]{y}} + \sqrt{\alpha \times \beta} = \vec{F}$$"
        );
        let doc = extract_html(r#"<math><mfenced open="[" close="]"><mi>a</mi><mfrac><mn>1</mn><mi>b</mi></mfrac></mfenced></math>"#, None).unwrap();
        assert_eq!(doc.markdown, r"$\left[a, \frac{1}{b}\right]$");
    }

    #[test]
    fn mathml_tables_preserve_rows_and_columns() {
        let doc = extract_html(r#"<math display="block"><mtable><mtr><mtd><mi>x</mi></mtd><mtd><mo>=</mo></mtd><mtd><mi>y</mi></mtd></mtr><mtr><mtd><mi>u</mi></mtd><mtd><mo>=</mo></mtd><mtd><mi>v</mi></mtd></mtr></mtable></math>"#, None).unwrap();
        assert_eq!(
            doc.markdown,
            r"$$\begin{aligned}x & = & y \\ u & = & v\end{aligned}$$"
        );
    }

    #[test]
    fn math_annotations_cannot_emit_raw_html_or_execute_active_scripts() {
        let doc = extract_html(r#"<p><span class="math-inline" data-math="x &lt;img src=x onerror=alert(1)&gt;">visual</span></p><script>alert(2)</script><pre>&lt;math&gt;$x_1$&lt;/math&gt;</pre>"#, None).unwrap();
        assert!(
            doc.markdown
                .contains(r"$x \lt img src=x onerror=alert(1)\gt $")
        );
        assert!(!doc.markdown.contains("alert(2)"));
        assert!(!doc.markdown.contains("<img"));
        assert!(doc.markdown.contains("<math>$x_1$</math>"));
    }

    #[test]
    fn deep_mathml_fails_explicitly_instead_of_overflowing() {
        let html = format!(
            "<math>{}<mi>x</mi>{}</math>",
            "<mrow>".repeat(66),
            "</mrow>".repeat(66)
        );
        assert!(
            matches!(extract_html(&html, None), Err(Error::Conversion(message)) if message.contains("MathML nesting"))
        );
    }

    #[test]
    fn mathml_skips_active_nodes_and_xml_annotations_inside_nested_text() {
        let doc = extract_html(r#"<math><mrow><mi>x</mi><script>secret_script()</script><style>secret_style</style><annotation-xml encoding="text/html"><mtext>secret_xml</mtext></annotation-xml><mo>+</mo><mtext>safe<script>secret_nested()</script> text</mtext></mrow></math>"#, None).unwrap();
        assert!(doc.markdown.contains("x +"));
        assert!(doc.markdown.contains(r"\text{safe text}"));
        assert!(!doc.markdown.contains("secret"));
    }

    fn announcement_html(body: &str) -> String {
        let data = serde_json::json!([{"announcement_body": {"body": body, "headline": "Patch notes", "posttime": 1736942400}}]);
        let mut html = String::from("<title>Generic title</title><div data-partnereventstore=\"");
        escaped(&data.to_string(), &mut html);
        html.push_str("\" data-groupvanityinfo='[{\"group_name\":\"Example Game\"}]'></div>");
        html
    }

    #[test]
    fn structured_announcement_recovers_metadata_links_and_video_from_json() {
        let html = announcement_html(
            r#"[p]Patch [b]released[/b]. Read [url=\"https:\/\/example.test\/notes\"]the notes[\/url].[\/p][previewyoutube=\"dQw4w9WgXcQ;full\"][\/previewyoutube]"#,
        );
        let doc = extract_html(&html, None).unwrap();
        assert_eq!(doc.metadata["title"], "Patch notes");
        assert_eq!(doc.metadata["author"], "Example Game");
        assert_eq!(doc.metadata["published"], "2025-01-15T12:00:00+00:00");
        assert!(doc.markdown.contains("Patch **released**."));
        assert!(
            doc.markdown
                .contains("[the notes](https://example.test/notes)")
        );
        assert!(
            doc.markdown
                .contains("![](https://www.youtube.com/watch?v=dQw4w9WgXcQ)")
        );
    }

    #[test]
    fn nested_bbcode_lists_quotes_and_literal_code_keep_structure() {
        let html = announcement_html(
            "[quote=Ada][list][*]First [b]bold [i]inner[/i][/b][list][*]Nested[/list][*]Second[/list][/quote][code=rust]let s = \"[b]literal[/b]\";\n  // <script> data\n[/code]",
        );
        let doc = extract_html(&html, None).unwrap();
        for text in [
            "Ada",
            "First",
            "**bold *inner***",
            "Nested",
            "Second",
            "```rust",
            "[b]literal[/b]",
            "  // <script> data",
        ] {
            assert!(
                doc.markdown.contains(text),
                "missing {text}: {}",
                doc.markdown
            );
        }
        assert!(doc.markdown.contains(">"));
    }

    #[test]
    fn bbcode_escapes_html_and_sanitizes_destinations_without_dropping_labels() {
        let html = announcement_html(
            r#"[p]<script>alert(1)</script> [url=javascript:alert(2)]safe label[/url] [img]data:text/html,bad[/img] [unknown]keep me[/unknown][/p]"#,
        );
        let doc = extract_html(&html, None).unwrap();
        assert!(doc.markdown.contains("safe label"));
        assert!(doc.markdown.contains("keep me"));
        assert!(
            doc.markdown
                .contains("&lt;script&gt;alert(1)&lt;/script&gt;")
        );
        assert!(!doc.markdown.contains("<script>"));
        assert!(!doc.markdown.contains("](javascript:"));
        assert!(!doc.markdown.contains("](data:"));
    }

    #[test]
    fn bbcode_protects_all_displayed_text_paths_and_keeps_image_alt_empty() {
        let html = announcement_html(
            r#"[h2]<img src=x onerror=heading>[/h2][quote="<img src=x onerror=author>"]Quoted body[/quote][url=https://example.test]<svg onload=label>Link label</svg>[/url][img]https://example.test/image.png[/img]"#,
        );
        let doc = extract_html(&html, None).unwrap();
        for expected in [
            "## &lt;img src=x onerror=heading&gt;",
            "&lt;img src=x onerror=author&gt;",
            "&lt;svg onload=label&gt;Link label&lt;/svg&gt;",
            "Quoted body",
            "![](https://example.test/image.png)",
        ] {
            assert!(
                doc.markdown.contains(expected),
                "missing {expected}: {}",
                doc.markdown
            );
        }
        assert!(!doc.markdown.contains("<img"));
        assert!(!doc.markdown.contains("<svg"));
    }

    #[test]
    fn invalid_event_data_falls_back_and_depth_limit_is_explicit() {
        let doc = extract_html(
            "<div data-partnereventstore='invalid'></div><p>Visible fallback</p>",
            None,
        )
        .unwrap();
        assert_eq!(doc.markdown, "Visible fallback");
        let body = format!("{}deep{}", "[b]".repeat(66), "[/b]".repeat(66));
        assert!(
            matches!(extract_html(&announcement_html(&body), None), Err(Error::Conversion(message)) if message.contains("BBCode nesting"))
        );
    }

    #[test]
    fn footnotes_repeated_references_preserve_inline_whitespace_and_labels() {
        let doc = extract_html(
            r##"<article><p>Before <a href="#fn-a">[1]</a> after;
            again<sup><a id="ref-a" href="#fn-a">1</a></sup>, done.
            <a href="#fn-a">Read the note</a>.</p>
            <section class="footnotes"><h2>Footnotes</h2><ol><li id="fn-a">
            <p>A <em>useful</em> note. <a href="#ref-a" role="doc-backlink">↩</a></p>
            </li></ol></section></article>"##,
            None,
        )
        .unwrap();
        assert!(
            doc.markdown.contains("Before [^1] after;"),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.contains("again[^1], done."));
        assert!(doc.markdown.contains("[Read the note](#fn-a)"));
        assert_eq!(doc.markdown.matches("[^1]:").count(), 1);
        assert!(doc.markdown.ends_with("[^1]: A *useful* note."));
        assert!(!doc.markdown.contains("Footnotes"));
        assert!(!doc.markdown.contains('↩'));
    }

    #[test]
    fn footnotes_keep_nested_blocks_and_code_inside_the_definition() {
        let doc = extract_html(
            r##"<article><p>Claim<sup><a href="#note">1</a></sup>.</p>
            <ol class="footnotes"><li id="note"><p>First paragraph.</p>
            <ul><li>Nested item.</li></ul><blockquote>Quotation.</blockquote>
            <pre><code>literal [^9] &lt;script&gt;</code></pre><p>Last paragraph.</p>
            </li></ol></article>"##,
            None,
        )
        .unwrap();
        assert!(doc.markdown.contains("Claim[^1]."));
        assert!(
            doc.markdown
                .contains("[^1]: First paragraph.\n\n    * Nested item."),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.contains("    > Quotation."));
        assert!(doc.markdown.contains("    literal [^9] <script>"));
        assert!(doc.markdown.ends_with("    Last paragraph."));
    }

    #[test]
    fn footnotes_restore_only_linked_hidden_definition_and_sanitize_its_body() {
        let doc = extract_html(r##"<article><p>Text<span data-definition="note"><a href="#">*</a></span> continues.</p>
            <aside id="note" style="display:none">Keep <a href="javascript:bad()">label</a>.
            <script>SECRET</script><span hidden>HIDDEN</span><a href="https://example.com/n">safe</a>.</aside>
            <aside hidden>UNRELATED</aside></article>"##, None).unwrap();
        assert!(
            doc.markdown.contains("Text[^1] continues."),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.contains("[^1]: Keep label."));
        assert!(doc.markdown.contains("[safe](https://example.com/n)"));
        assert!(!doc.markdown.contains("javascript:") && !doc.markdown.contains("SECRET"));
        assert!(!doc.markdown.contains("HIDDEN") && !doc.markdown.contains("UNRELATED"));
    }

    #[test]
    fn footnotes_inline_popovers_and_org_labels_move_each_body_once() {
        let doc = extract_html(r#"<article><p>A<span class="inline-footnote">1<span class="footnoteContent" hidden>First note.</span></span> B.</p>
            <p>C<label class="footref" for="fn.2">2</label><input id="fn.2" type="checkbox">
            <span class="sidenote"><sup>2</sup> Second note.</span> D.</p></article>"#, None).unwrap();
        assert!(doc.markdown.contains("A[^1] B."), "{}", doc.markdown);
        assert!(doc.markdown.contains("C[^2]"));
        assert!(doc.markdown.contains(" D."));
        assert!(
            doc.markdown
                .ends_with("[^1]: First note.\n\n[^2]: Second note.")
        );
        assert_eq!(doc.markdown.matches("Second note.").count(), 1);
    }

    #[test]
    fn footnotes_loose_definitions_preserve_bold_attribution_and_trailing_update() {
        let doc = extract_html(
            r#"<article><p>First<sup>1</sup> claim.</p><p>Second<sup>2</sup> claim.</p>
            <p><b><sup>1</sup> Note today:</b> First evidence.</p>
            <p><b><sup>2</sup> Note yesterday:</b> Second evidence.</p>
            <p><b>Update:</b> Keep this in the article.</p></article>"#,
            None,
        )
        .unwrap();
        assert!(
            doc.markdown.contains("First[^1] claim."),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.contains("Second[^2] claim."));
        assert!(
            doc.markdown
                .contains("**Update:** Keep this in the article.\n\n[^1]:")
        );
        assert!(
            doc.markdown
                .contains("[^1]: **Note today:** First evidence.")
        );
        assert!(
            doc.markdown
                .ends_with("[^2]: **Note yesterday:** Second evidence.")
        );
    }

    #[test]
    fn footnotes_labeled_external_section_adopts_continuations_without_unrelated_asides() {
        let doc = extract_html(
            r#"<article><p>A<sup>1</sup> and B<sup>2</sup>.</p></article>
            <aside>Unrelated advertising</aside><div class="page__footnotes"><h3>Footnotes</h3>
            <p><sup>1</sup> First evidence.</p><ul><li>Detail.</li></ul>
            <p><sup>2</sup> Second evidence.</p></div>"#,
            None,
        )
        .unwrap();
        assert!(
            doc.markdown.starts_with("A[^1] and B[^2]."),
            "{}",
            doc.markdown
        );
        assert!(
            doc.markdown
                .contains("[^1]: First evidence.\n\n    * Detail.")
        );
        assert!(doc.markdown.ends_with("[^2]: Second evidence."));
        assert!(!doc.markdown.contains("Unrelated"));
    }

    #[test]
    fn footnotes_generic_named_anchors_require_a_shared_definition_section() {
        let doc = extract_html(
            r##"<article><p>Fact <a href="#one">[1]</a> and <a href="#two">[2]</a>.</p>
            <div class="notes"><p><a name="one"><sup>1</sup></a> First.</p>
            <p><a name="two"><sup>2</sup></a> Second.</p></div></article>"##,
            None,
        )
        .unwrap();
        assert!(
            doc.markdown.starts_with("Fact [^1] and [^2]."),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.ends_with("[^1]: First.\n\n[^2]: Second."));
        let ordinary = extract_html(r##"<article><p>See <a href="#equation">1</a> and <a href="#theorem">2</a>.</p>
            <h2 id="equation">Equation</h2><p>x=2</p><h2 id="theorem">Theorem</h2><p>Proof.</p></article>"##, None).unwrap();
        assert!(!ordinary.markdown.contains("[^"));
        assert!(ordinary.markdown.contains("[1](#equation)"));
        assert!(ordinary.markdown.contains("## Theorem"));
    }

    #[test]
    fn footnotes_missing_targets_and_remote_numeric_links_remain_links() {
        let doc = extract_html(r##"<article><p><a href="#missing">1</a> and <a href="https://elsewhere.test/#fn-2">2</a>.</p>
            <ol class="footnotes"><li id="fn-2">Unreferenced but valid content.</li></ol>
            <pre><code>&lt;a href="#fn-2"&gt;2&lt;/a&gt;</code></pre></article>"##, Some("https://example.com/post")).unwrap();
        assert!(!doc.markdown.contains("[^"));
        assert!(doc.markdown.contains("[1](#missing)"));
        assert!(doc.markdown.contains("Unreferenced but valid content."));
        assert!(doc.markdown.contains("<a href=\"#fn-2\">2</a>"));
    }

    #[test]
    fn footnotes_do_not_delete_unmatched_sidenote_text_or_following_prose() {
        let doc = extract_html(r##"<article><p>Claim<sup><a href="#fn-1">1</a></sup><span class="sidenote">Different marginal remark.</span></p>
            <div class="footnotes"><h2>Footnotes</h2><ol><li id="fn-1">Evidence.</li></ol>
            <p>Independent concluding prose.</p></div></article>"##, None).unwrap();
        assert!(doc.markdown.contains("Different marginal remark."));
        assert!(doc.markdown.contains("Independent concluding prose."));
        assert!(doc.markdown.ends_with("[^1]: Evidence."));
    }

    #[test]
    fn footnotes_aside_start_and_word_backlinks_use_their_explicit_numbers() {
        let doc = extract_html(r##"<article><p>Claim<sup>7</sup> and Word<sup><a href="#_ftn1">[1]</a></sup>.</p>
            <aside><ol start="7"><li>Aside evidence.</li></ol></aside>
            <p><sup><a href="//document-guid#_ftnref1">[1]</a></sup> Word evidence.</p></article>"##, None).unwrap();
        assert!(
            doc.markdown.starts_with("Claim[^1] and Word[^2]."),
            "{}",
            doc.markdown
        );
        assert!(
            doc.markdown
                .ends_with("[^1]: Aside evidence.\n\n[^2]: Word evidence.")
        );
        assert!(!doc.markdown.contains("document-guid"));
    }

    #[test]
    fn a_definition_beside_the_body_moves_to_the_end_and_is_not_furniture() {
        let doc = extract_html(
            &format!(
                r##"<div><div class="post"><p>Claim<sup><a href="#_ftn1">[1]</a></sup> {}</p><p>{}</p><p>{}</p></div>
                <p id="ftn1"><a href="#_ftnref1">[1]</a> Word evidence.</p>
                <div><p>Written by Jane.</p></div></div>"##,
                prose(40),
                prose(40),
                prose(40)
            ),
            None,
        )
        .unwrap();
        assert!(doc.markdown.starts_with("Claim[^1]"), "{}", doc.markdown);
        assert!(
            doc.markdown.ends_with("[^1]: Word evidence."),
            "{}",
            doc.markdown
        );
        assert!(!doc.markdown.contains("Jane"), "{}", doc.markdown);
        // A reference inside furniture that is left out makes no footnote.
        let doc = extract_html(
            &format!(
                r##"<div><div class="post"><p>{}</p><p>{}</p><p>{}</p></div>
                <div><p>Written by Jane<sup><a href="#fn-9">9</a></sup>.</p></div>
                <div class="footnotes"><ol><li id="fn-9">Bio note.</li></ol></div></div>"##,
                prose(40),
                prose(40),
                prose(40)
            ),
            None,
        )
        .unwrap();
        assert!(
            !doc.markdown.contains("[^") && !doc.markdown.contains("Jane"),
            "{}",
            doc.markdown
        );
    }

    #[test]
    fn footnotes_do_not_create_dangling_markers_for_empty_sanitized_bodies() {
        let doc = extract_html(
            r##"<article><p>Claim<sup><a href="#fn-1">1</a></sup>.</p>
            <div class="footnotes"><p id="fn-1"><script>unsafe()</script>
            <a role="doc-backlink" href="#ref-1">↩</a></p></div></article>"##,
            None,
        )
        .unwrap();
        assert!(!doc.markdown.contains("[^"));
        assert!(doc.markdown.contains("#fn-1"));
        assert!(!doc.markdown.contains("unsafe()"));
    }

    #[test]
    fn source_newlines_preserve_separating_spaces_after_inline_elements() {
        let markdown = fragment(
            "<p>First <a href=\"/x\">label</a> tail;\n    next <code>x</code> tail;\n  last.</p>",
        )
        .unwrap();
        assert_eq!(markdown, "First [label](/x) tail;\nnext `x` tail;\nlast.");
    }

    #[test]
    fn hidden_or_discarded_popover_references_do_not_restore_their_definitions() {
        for hidden in [
            r#"<span class="inline-footnote" hidden>1<span class="footnoteContent">Hidden secret</span></span>"#,
            r#"<div hidden><span class="inline-footnote">1<span class="footnoteContent">Hidden secret</span></span></div>"#,
            r#"<div style="display:none"><span class="footnote-container"><span class="footnote">Hidden secret</span></span></div>"#,
            r#"<nav><span class="inline-footnote">1<span class="footnoteContent">Hidden secret</span></span></nav>"#,
            r#"<span class="inline-footnote"><span hidden><span class="footnoteContent">Hidden secret</span></span></span>"#,
        ] {
            let doc =
                extract_html(&format!("<article><p>Visible.</p>{hidden}</article>"), None).unwrap();
            assert_eq!(doc.markdown, "Visible.", "{hidden}");
        }
        let visible = extract_html(r#"<article><p>Visible<span class="inline-footnote">1<span class="footnoteContent" hidden>Intended note.</span></span>.</p></article>"#, None).unwrap();
        assert_eq!(visible.markdown, "Visible[^1].\n\n[^1]: Intended note.");
    }

    #[test]
    fn ordinary_numbered_navigation_does_not_turn_instructions_into_notes() {
        for targets in [
            r#"<section><h2>Instructions</h2><p id="step1">Install app.</p><p id="step2">Launch app.</p></section>"#,
            r#"<ol><li id="step1">Install app.</li><li id="step2">Launch app.</li></ol>"#,
        ] {
            let source = format!(
                r##"<article><p>Steps <a href="#step1">1</a> and <a href="#step2">2</a>.</p>{targets}</article>"##
            );
            let doc = extract_html(&source, None).unwrap();
            assert!(!doc.markdown.contains("[^"), "{}", doc.markdown);
            assert!(doc.markdown.contains("[1](#step1) and [2](#step2)"));
            assert_eq!(doc.markdown.matches("Install app.").count(), 1);
            assert_eq!(doc.markdown.matches("Launch app.").count(), 1);
        }
    }

    #[test]
    fn footnote_detection_excludes_all_recognized_math_wrappers() {
        for wrapper in [
            "math-inline",
            "math-block",
            "katex",
            "katex-display",
            "hurmet-tex",
            "mwe-math-element",
        ] {
            for visual in ["<sup>2</sup>", "x<sup>2</sup>"] {
                let source = format!(
                    r#"<article><p>Formula <span class="{wrapper}" data-math="x^2">{visual}</span>.</p><ol class="footnotes"><li id="fn-1">Unused one.</li><li id="fn-2">Unused two.</li></ol></article>"#
                );
                let doc = extract_html(&source, None).unwrap();
                assert!(!doc.markdown.contains("[^"), "{wrapper}: {}", doc.markdown);
                assert_eq!(doc.markdown.matches("x^2").count(), 1);
                assert!(doc.markdown.contains("Unused two."));
            }
        }
        let doc = extract_html(r#"<article><p><mjx-container data-math="x^2"><sup>2</sup></mjx-container></p><ol class="footnotes"><li id="fn-1">Unused one.</li><li id="fn-2">Unused two.</li></ol></article>"#, None).unwrap();
        assert!(doc.markdown.contains("$x^2$"));
        assert!(!doc.markdown.contains("[^"));
    }

    #[test]
    fn backlink_only_definitions_do_not_leave_dangling_or_skipped_numbers() {
        let doc = extract_html(r##"<article><p>See <a id="ref1" href="#fn-1">1</a> and <a href="#fn-2">2</a>.</p>
            <ol class="footnotes"><li id="fn-1"><a href="#ref1">Back</a></li><li id="fn-2">Real note.</li></ol></article>"##, None).unwrap();
        assert!(
            doc.markdown.contains("[1](#fn-1) and [^1]"),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.ends_with("[^1]: Real note."));
        assert!(!doc.markdown.contains("[^2]"));
        assert!(doc.markdown.contains("[Back](#ref1)"));
    }

    #[test]
    fn oversized_footnote_list_start_does_not_overflow_or_drop_definitions() {
        let source = format!(
            r##"<article><p>A<a href="#a">1</a> and B<a href="#b">2</a>.</p><ol class="footnotes" start="{}"><li id="a">First.</li><li id="b">Second.</li></ol></article>"##,
            usize::MAX
        );
        let doc = extract_html(&source, None).unwrap();
        assert!(
            doc.markdown.contains("A[^1] and B[^2]."),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.ends_with("[^1]: First.\n\n[^2]: Second."));
    }

    #[test]
    fn footnotes_ignore_duplicate_mathjax_previews_with_tex_source() {
        for wrapper in [
            "MathJax_Preview",
            "MathJax",
            "MathJax_Display",
            "MathJax_SVG",
            "MathJax_MathML",
        ] {
            let source = format!(
                r#"<article><p>Formula <span class="{wrapper}"><sup>2</sup></span><script type="math/tex">x^2</script>.</p><ol class="footnotes"><li id="fn-1">Unused one.</li><li id="fn-2">Unused two.</li></ol></article>"#
            );
            let doc = extract_html(&source, None).unwrap();
            assert!(!doc.markdown.contains("[^"), "{wrapper}: {}", doc.markdown);
            assert!(doc.markdown.starts_with("Formula $x^2$."));
            assert_eq!(doc.markdown.matches("Unused two.").count(), 1);
        }
    }

    #[test]
    fn generic_footnote_continuations_remain_with_their_definition() {
        let doc = extract_html(
            r##"<article><p>Claims <a href="#first">1</a> and <a href="#second">2</a>.</p>
            <div class="note"><p id="first">1. First evidence.</p><p>Continuation paragraph.</p>
            <ul><li>Supporting detail.</li></ul><blockquote>Quoted detail.</blockquote>
            <pre><code>literal x &lt; y</code></pre><p>Final continuation.</p>
            <p id="second">2. Second evidence.</p><p class="seealso">See also: another topic.</p>
            <p>Keep this with the related section.</p></div><p>Outside the notes.</p></article>"##,
            None,
        )
        .unwrap();
        assert!(doc.markdown.contains("Claims [^1] and [^2]."));
        let definition = doc.markdown.find("[^1]: First evidence.").unwrap();
        for text in [
            "Continuation paragraph.",
            "Supporting detail.",
            "Quoted detail.",
            "literal x < y",
            "Final continuation.",
        ] {
            assert_eq!(
                doc.markdown.matches(text).count(),
                1,
                "{text}: {}",
                doc.markdown
            );
            assert!(doc.markdown.find(text).unwrap() > definition);
        }
        assert!(
            doc.markdown
                .contains("\n\n    Continuation paragraph.\n\n    * Supporting detail.")
        );
        assert!(doc.markdown.contains("\n    literal x < y\n"));
        assert!(doc.markdown.find("See also: another topic.").unwrap() < definition);
        assert!(doc.markdown.find("Outside the notes.").unwrap() < definition);
        assert!(doc.markdown.ends_with("[^2]: Second evidence."));
    }

    #[test]
    fn generic_footnotes_stop_at_section_id_and_update_boundaries() {
        for boundary in [
            "<h3>Next section</h3>",
            "<p id=\"new-section\">Next section</p>",
            "<p><a name=\"new-section\"></a>Next section</p>",
            "<p><b>Update today:</b> Next section</p>",
        ] {
            let source = format!(
                r##"<article><p><a href="#a">1</a> and <a href="#b">2</a>.</p><div class="notes"><p id="a">1. First.</p><p id="b">2. Second.</p>{boundary}<p>Independent content.</p></div></article>"##
            );
            let doc = extract_html(&source, None).unwrap();
            assert!(
                doc.markdown.find("Independent content.").unwrap()
                    < doc.markdown.find("[^1]:").unwrap(),
                "{}",
                doc.markdown
            );
            assert!(doc.markdown.ends_with("[^1]: First.\n\n[^2]: Second."));
        }
    }

    #[test]
    fn hidden_continuations_are_never_promoted_to_visible_definition_roots() {
        for hidden in [
            "hidden",
            "style=\"display:none\"",
            "style=\"visibility:hidden\"",
        ] {
            for notes in [
                format!(
                    r##"<p><a href="#a">1</a> and <a href="#b">2</a>.</p><div class="notes"><p id="a">1. First.</p><p {hidden}>SECRET</p><p>Visible continuation.</p><p id="b">2. Second.</p></div>"##
                ),
                format!(
                    r#"<p>A<sup>1</sup> B<sup>2</sup>.</p><hr><p><sup>1</sup>First.</p><p {hidden}>SECRET</p><p>Visible continuation.</p><p><sup>2</sup>Second.</p>"#
                ),
            ] {
                let doc = extract_html(&format!("<article>{notes}</article>"), None).unwrap();
                assert!(!doc.markdown.contains("SECRET"), "{}", doc.markdown);
                assert!(
                    doc.markdown
                        .contains("[^1]: First.\n\n    Visible continuation.")
                );
                assert!(doc.markdown.ends_with("[^2]: Second."));
            }
        }
    }

    #[test]
    fn generic_numeric_prefix_removal_requires_the_actual_reference_number() {
        let doc = extract_html(r##"<article><p><a href="#a">1</a>, <a href="#b">2</a>, <a href="#c">3</a>.</p>
            <div class="notes"><p id="a">1. <em>First evidence.</em></p><p id="b">2024. A historical date.</p><p id="c">3.5 is a decimal value.</p></div></article>"##, None).unwrap();
        assert!(
            doc.markdown.contains("[^1]: *First evidence.*"),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.contains("[^2]: 2024\\. A historical date."));
        assert!(doc.markdown.ends_with("[^3]: 3.5 is a decimal value."));
    }

    #[test]
    fn numeric_prefix_and_backlink_only_definition_is_not_emitted() {
        let doc = extract_html(r##"<article><p><a id="ref-a" href="#a">1</a> and <a href="#b">2</a>.</p>
            <div class="notes"><p id="a">1. <a href="#ref-a">Back</a></p><p id="b">2. Real content.</p></div></article>"##, None).unwrap();
        assert!(
            doc.markdown.starts_with("[1](#a) and [^1]."),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.ends_with("[^1]: Real content."));
        assert!(!doc.markdown.contains("[^2]"));
    }

    #[test]
    fn a_substack_note_permalink_keeps_the_note_and_its_image_only() {
        let note = |text: &str| {
            format!(
                r#"<div class="feedCommentBody-x"><div class="feedCommentBodyInner-y"><div class="ProseMirror FeedProseMirror"><p>{text}</p></div></div></div>"#
            )
        };
        let page = format!(
            r#"<html><head><meta property="og:title" content="Test User (@testuser)"><meta property="og:image" content="https://example.com/full.jpg"></head><body><div class="promo"><h4>The app for independent voices</h4><p>Promotion copy.</p></div><div class="feedPermalinkUnit-z">{}</div><div class="feed">{}</div></body></html>"#,
            note("The main note."),
            note("Another unrelated note.")
        );
        let doc = extract_html(&page, Some("https://substack.com/@testuser/note/c-1")).unwrap();
        assert_eq!(
            doc.markdown,
            "The main note.\n\n![](https://example.com/full.jpg)"
        );
        assert_eq!(doc.metadata["author"], "Test User");
        assert_eq!(doc.metadata["site"], "Substack");
        // The same classes on a page that is not Substack's are ordinary markup.
        let elsewhere = page.replace("feedPermalinkUnit-z", "unit");
        let doc = extract_html(&elsewhere, Some("https://example.org/notes")).unwrap();
        assert!(doc.markdown.contains("Promotion copy"), "{}", doc.markdown);
        // An article with its own rendered body is left to the ordinary reader.
        let article = page.replace(
            r#"<div class="feed">"#,
            r#"<div class="body markup"><p>Rendered article body.</p></div><div class="feed">"#,
        );
        let doc = extract_html(&article, Some("https://substack.com/p/post")).unwrap();
        assert!(
            doc.markdown.contains("Rendered article body."),
            "{}",
            doc.markdown
        );
    }

    #[test]
    fn spacer_and_icon_images_are_dropped_but_emoji_and_content_images_stay() {
        let doc = extract_html(
            r#"<html><body><article><h1>Thread</h1><p><img src="s.gif" height="1" width="40">First comment <img src="https://example.com/e.png" alt="😀" width="16" height="16"> text.</p>
            <p><img src="https://example.com/avatar.png" style="width: 20px; height: 20px"> Second comment.</p>
            <p><img src="https://example.com/photo.jpg" width="640" height="480" alt="Photo"></p>
            <p><img src="https://example.com/banner.png" width="100%" height="20%" alt="Banner"></p>
            <p><img src="https://example.com/eq.png" alt="\frac{a}{b}" width="20" height="12"></p>
            <svg viewBox="0 0 16 16"><path d="M0 0h16v16z"/></svg></article></body></html>"#,
            None,
        )
        .unwrap();
        assert!(!doc.markdown.contains("s.gif"), "{}", doc.markdown);
        assert!(!doc.markdown.contains("avatar.png"), "{}", doc.markdown);
        assert!(
            doc.markdown.contains("First comment 😀 text."),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.contains("photo.jpg"), "{}", doc.markdown);
        assert!(doc.markdown.contains("banner.png"), "{}", doc.markdown);
        assert!(doc.markdown.contains("frac"), "{}", doc.markdown);
        // A fragment (an email body, a book) keeps what its author put in.
        let fragment = fragment(
            r#"<p><img src="https://example.com/icon.png" width="16" height="16"> Label</p>"#,
        )
        .unwrap();
        assert!(fragment.contains("icon.png"), "{fragment}");
    }

    #[test]
    fn replacement_references_preserve_only_visible_wrapper_edge_whitespace() {
        let doc = extract_html(
            r##"<article><p><span>Word<span class="reference"> <sup>1</sup> </span></span>Next.</p>
            <ol class="footnotes"><li id="fn-1">Definition.</li></ol></article>"##,
            None,
        )
        .unwrap();
        assert_eq!(doc.markdown, "Word [^1] Next.\n\n[^1]: Definition.");
        let inline = extract_html(r#"<article><p>Before<span class="inline-footnote">1<span class="footnoteContent" hidden>  Hidden definition.  </span></span>After.</p></article>"#, None).unwrap();
        assert_eq!(
            inline.markdown,
            "Before[^1]After.\n\n[^1]: Hidden definition."
        );
    }

    /// Elements of hidden, chrome, note and plain kinds, nested in each other.
    const KINDS: &str = r##"<html><body>
        <div hidden><p>Hidden.</p></div><p style="display:none">None.</p>
        <nav class="md:hidden"><ul><li><a href="/a">A</a></li></ul></nav>
        <div class="hidden md:block">Shown.</div><div role="tooltip">Tip</div>
        <span class="mw-editsection">edit</span><div id="SiteSub">From</div>
        <section class="related-posts"><h2>Related</h2></section>
        <div role="navigation"><a href="/b">B</a></div>
        <aside class="toc"><ol><li><a href="#x">X</a></li></ol></aside>
        <div data-testid="issue-viewer-metadata-pane">Meta</div>
        <div class="related-posts" role="doc-endnotes"><p>Note kept.</p></div>
        <section id="footnotes"><ol><li id="fn1">One.</li></ol></section>
        <div class="footnote-content"><p>Content.</p></div>
        <div><h3>Related stories</h3><article><a href="/c"><h4>C</h4></a></article><article><a href="/d"><h4>D</h4></a></article></div>
        <p class="note">Plain <em>text</em>.</p></body></html>"##;

    #[test]
    fn facts_answer_as_the_tests_they_keep() {
        let document = Html::parse_document(KINDS);
        let root = document.root_element();
        let elements: Vec<_> = root.descendants().filter_map(ElementRef::wrap).collect();
        // Asked children first, twice, against a fresh table for each answer.
        let kept = Facts::new(root);
        let mut answers = [0; 4];
        for _ in 0..2 {
            for &element in elements.iter().rev() {
                let fresh = || Facts::new(root);
                let pairs = [
                    (kept.hidden(element), is_hidden(element)),
                    (
                        article::excluded(&kept, element),
                        article::excluded(&fresh(), element),
                    ),
                    (
                        article::note(&kept, element),
                        article::note(&fresh(), element),
                    ),
                    (
                        note_context(&kept, element),
                        note_context(&fresh(), element),
                    ),
                ];
                for (index, (remembered, computed)) in pairs.into_iter().enumerate() {
                    assert_eq!(
                        remembered,
                        computed,
                        "fact {index} of {:?}",
                        element.value()
                    );
                    answers[index] += usize::from(computed);
                }
            }
        }
        // Each fact holds for some elements and not for others.
        assert!(answers.iter().all(|count| *count > 0), "{answers:?}");
        assert!(answers[1] >= 2 * 9, "{answers:?}");
        // Positions are the tree's own numbering of its nodes.
        for (index, node) in document.tree.nodes().enumerate() {
            assert_eq!(facts::position(node.id()), index);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "an element of another tree")]
    fn facts_are_kept_for_one_tree_only() {
        let (one, other) = (
            Html::parse_document("<p>a</p>"),
            Html::parse_document("<p>b</p>"),
        );
        Facts::new(one.root_element()).hidden(other.root_element());
    }

    #[test]
    fn element_tests_answer_as_the_selectors_they_replace() {
        let document = Html::parse_document(
            r#"<html><body><a>no address</a><a href="">empty</a><a name="n">named</a>
            <span id="i">id</span><div name="d">named div</div><sup>1</sup><span data-definition="d">definition</span>
            <span class="x footnote-container">container</span><span class="sidenote">side</span>
            <span class="Sidenote">case</span><span class="footnoteContent">content</span>
            <span class="sidenote-container inline-footnote">two</span><div class="footnote">div</div>
            <label class="footref">label</label><label class="footrefs">labels</label>
            <span class="a&#9;footnote&#12;b">tab</span><span class="x&#160;sidenote">nbsp</span>
            <svg><a href="/svg">vector link</a><title id="t">vector title</title></svg>
            <template><a href="/t" id="u">template</a><sup>2</sup></template></body></html>"#,
        );
        type Test = fn(ElementRef<'_>) -> bool;
        let cases: [(&str, Test); 6] = [
            ("a[href]", link),
            ("[id],a[name]", fragment_target),
            (
                "a[href], sup, span[data-definition], label.footref",
                reference_markup,
            ),
            (
                "span.footnote-container, span.sidenote-container, span.inline-footnote",
                |element| {
                    span_of(
                        element,
                        &[
                            "footnote-container",
                            "sidenote-container",
                            "inline-footnote",
                        ],
                    )
                },
            ),
            (
                "span.footnote,span.sidenote,span.footnoteContent",
                |element| span_of(element, &["footnote", "sidenote", "footnoteContent"]),
            ),
            ("span.sidenote", |element| span_of(element, &["sidenote"])),
        ];
        let root = document.root_element();
        for (query, test) in cases {
            let selected: Vec<_> = root.select(&Selector::parse(query).unwrap()).collect();
            let walked: Vec<_> = inside(root).filter(|element| test(*element)).collect();
            assert!(!selected.is_empty(), "{query}");
            assert_eq!(walked, selected, "{query}");
        }
    }

    #[test]
    fn class_names_are_scrapers_classes_without_interning() {
        let document = Html::parse_document(
            "<p class=\" a  b\tc\nd\u{c}e a f\u{a0}g \">x</p><p>y</p><p class=\"\">z</p>",
        );
        for element in document
            .root_element()
            .descendants()
            .filter_map(ElementRef::wrap)
        {
            let value = element.value();
            for name in ["a", "b", "c", "d", "e", "f", "g", "f\u{a0}g", "", "x"] {
                assert_eq!(
                    value.class_names().any(|class| class == name),
                    value.classes().any(|class| class == name),
                    "{name:?}"
                );
            }
        }
    }

    /// Two pages: one with every landmark (one of them only in an element
    /// taken out of the tree), one with their look-alikes.
    fn landmark_pages() -> [Html; 2] {
        let mut with = Html::parse_document(
            r#"<html><head><title>First</title><meta name="a" content="1"><meta property="b">
            <link rel="alternate canonical"><link rel="canonical" href="/c"><link href="/s.css">
            <script src="/s.js"></script><script type="application/ld+json">{}</script></head>
            <body><svg><title>Vector</title></svg><h1>One</h1><h1>Two</h1>
            <time>soon</time><time datetime="2026-01-01">then</time>
            <div data-partnereventstore="[]"></div><div class="x feedPermalinkUnit-y">unit</div>
            <div class="markup body">rendered</div><div data-testid="primaryColumn"><section>
            <article data-testid="tweet">post</article></section></div>
            <img src="/i.png"><img alt="none"><video poster="/p.png"></video>
            <table id="hnmain"><tr class="athing"><td>row</td></tr></table>
            <div id="gone"><meta name="moved" content="2"><title>Moved</title></div>
            <meta id="lone" name="lone" content="3"></body></html>"#,
        );
        for id in ["gone", "lone"] {
            let node = with
                .root_element()
                .descendants()
                .filter_map(ElementRef::wrap)
                .find(|element| element.value().attribute("id") == Some(id))
                .unwrap()
                .id();
            with.tree.get_mut(node).unwrap().detach();
        }
        let without = Html::parse_document(
            r#"<html><head><link rel="stylesheet" href="/s.css"></head>
            <body><div class="feedpermalinkunit">unit</div><div class="body">body</div>
            <span class="body markup">not a div</span>
            <div data-testid="primaryColumn"><template><article data-testid="tweet">post</article></template></div>
            <article data-testid="tweet">outside</article><div id="HNmain"></div>
            <article><b data-engagement-action="like"></b></article><main><b data-engagement-action="like"></b></main>
            <tr class="athing"><td>no table</td></tr><table><tr class="comtrs"><td>x</td></tr></table></body></html>"#,
        );
        [with, without]
    }

    #[test]
    fn landmarks_find_what_their_selectors_find() {
        for (index, document) in landmark_pages().iter().enumerate() {
            let page = Landmarks::read(document);
            let all = |query: &str| {
                document
                    .select(&Selector::parse(query).unwrap())
                    .collect::<Vec<_>>()
            };
            let first = |query: &str| all(query).first().copied();
            let canonical = all("link[rel]").into_iter().find(|link| {
                link.value().attribute("rel").is_some_and(|rel| {
                    rel.split_whitespace()
                        .any(|t| t.eq_ignore_ascii_case("canonical"))
                })
            });
            assert_eq!(page.metas, all("meta"), "{index}");
            assert_eq!(page.scripts, all("script"), "{index}");
            assert_eq!(page.title, first("title"), "{index}");
            assert_eq!(page.heading, first("h1"), "{index}");
            assert_eq!(page.canonical, canonical, "{index}");
            assert_eq!(page.times, all("time[datetime]"), "{index}");
            assert_eq!(page.events, all("[data-partnereventstore]"), "{index}");
            assert_eq!(
                page.permalink,
                first(r#"[class*="feedPermalinkUnit"]"#),
                "{index}"
            );
            assert_eq!(page.assets, all("link[href], script[src]"), "{index}");
            assert_eq!(
                page.rendered_body,
                first("div.body.markup").is_some(),
                "{index}"
            );
            assert_eq!(
                page.column_post,
                first(r#"[data-testid="primaryColumn"] article[data-testid="tweet"]"#).is_some()
                    || first("main article [data-engagement-action]").is_some(),
                "{index}"
            );
            assert_eq!(page.media, all("img[src], video[poster]"), "{index}");
            assert_eq!(page.hn_main, first("#hnmain").is_some(), "{index}");
            assert_eq!(
                page.hn_rows,
                first("tr.athing, tr.comtr").is_some(),
                "{index}"
            );
            // Every flag differs between the pages.
            let flags = [
                page.rendered_body,
                page.column_post,
                page.hn_main,
                page.hn_rows,
                page.permalink.is_some(),
                page.canonical.is_some(),
            ];
            assert!(
                flags.iter().all(|flag| *flag == (index == 0)),
                "{index}: {flags:?}"
            );
        }
        // An element taken out of the tree is left out, what it holds is not.
        let [with, _] = landmark_pages();
        let page = Landmarks::read(&with);
        assert_eq!(page.title.map(plain).as_deref(), Some("First"));
        assert!(meta(&page.metas, &["moved"]).is_some());
        assert!(meta(&page.metas, &["lone"]).is_none());
    }

    /// Footnote recovery's questions about every element, asked of its
    /// ancestors as the passes did before they were read once: the elements
    /// (by address) inside a literal container, in scope, in a note's context.
    fn reach_by_ancestors(
        root: ElementRef<'_>,
        document: ElementRef<'_>,
        facts: &Facts,
    ) -> [HashSet<usize>; 3] {
        let elements: Vec<_> = document
            .descendants()
            .filter_map(ElementRef::wrap)
            .collect();
        let mut literal = HashSet::new();
        for &element in &elements {
            if element
                .parent()
                .and_then(ElementRef::wrap)
                .is_some_and(|parent| literal.contains(&element_key(parent)))
                || literal_container(element)
            {
                literal.insert(element_key(element));
            }
        }
        let external: Vec<_> = elements
            .iter()
            .copied()
            .filter(|element| {
                !contains_element(root, *element)
                    && !contains_element(*element, root)
                    && matches!(
                        element.value().name(),
                        "div" | "section" | "aside" | "ol" | "ul" | "p" | "li"
                    )
                    && !literal.contains(&element_key(*element))
                    && !element
                        .ancestors()
                        .filter_map(ElementRef::wrap)
                        .any(|parent| {
                            matches!(
                                parent.value().name(),
                                "head" | "template" | "noscript" | "iframe" | "object" | "embed"
                            )
                        })
                    && (note_context(facts, *element)
                        || (["id", "class"].iter().any(|attr| {
                            element.value().attribute(attr).is_some_and(|value| {
                                value.to_ascii_lowercase().contains("footnote")
                            })
                        }) && element
                            .select(&selector("h1,h2,h3,h4,h5,h6"))
                            .any(note_heading)))
            })
            .collect();
        let in_scope = |element: ElementRef<'_>| {
            contains_element(root, element)
                || external
                    .iter()
                    .any(|container| contains_element(*container, element))
        };
        let scope = elements
            .iter()
            .copied()
            .filter(|element| in_scope(*element))
            .map(element_key)
            .collect();
        let context = elements
            .iter()
            .copied()
            .filter(|element| {
                in_scope(*element)
                    && (note_context(facts, *element)
                        || element
                            .ancestors()
                            .filter_map(ElementRef::wrap)
                            .take_while(|parent| in_scope(*parent))
                            .any(|parent| note_context(facts, parent)))
            })
            .map(element_key)
            .collect();
        [literal, scope, context]
    }

    #[test]
    fn reach_answers_what_each_element_asked_of_its_ancestors() {
        let page = Html::parse_document(
            r##"<html><head><div class="footnotes"><p>In the head.</p></div></head><body>
            <div class="wrap"><article id="root"><p>Text<sup><a href="#fn1">1</a></sup>.</p>
            <div class="footnotes"><ol><li id="r1">Inner.</li></ol></div>
            <pre><div class="footnotes"><p>Literal.</p></div></pre>
            <span class="katex"><span class="footnotes">Math.</span></span></article>
            <section role="doc-endnotes"><ol><li id="fn1"><p>External <b>note</b>.</p></li></ol></section>
            <div id="Footnotes-list"><h3>Notes</h3><p>Named.</p></div>
            <div class="FOOTNOTE-box"><p>No heading.</p></div>
            <template><div class="footnotes"><p>Inert.</p></div></template>
            <noscript><div class="footnotes">raw</div></noscript>
            <aside><div role="doc-footnote"><p>Nested context.</p></div></aside></div></body></html>"##,
        );
        fn root_of<'a>(document: &'a Html, id: &str) -> ElementRef<'a> {
            document
                .root_element()
                .descendants()
                .filter_map(ElementRef::wrap)
                .find(|element| element.value().attribute("id") == Some(id))
                .unwrap()
        }
        // A whole page around its region; a region inside a notes container,
        // which is not one outside it; a fragment inside a template, as a
        // site reader converts one element; a region in a document inside
        // inert content further up.
        let wrapped = Html::parse_document(
            r#"<div id="footnotes"><article id="root"><p>x</p></article><p>Beside.</p></div>"#,
        );
        let template = Html::parse_document(
            r#"<template><div id="doc"><div class="footnotes"><p>x</p></div><pre><b>y</b></pre></div></template>"#,
        );
        let inert = Html::parse_document(
            r#"<template><section><div id="doc"><article id="root"><p>x</p></article><div class="footnotes"><p>y</p></div></div></section></template>"#,
        );
        let cases = [
            (root_of(&page, "root"), page.root_element()),
            (root_of(&wrapped, "root"), wrapped.root_element()),
            (root_of(&template, "doc"), root_of(&template, "doc")),
            (root_of(&inert, "root"), root_of(&inert, "doc")),
        ];
        let mut counts = [0; 3];
        for (root, document) in cases {
            let facts = Facts::new(document);
            let elements: Vec<_> = document
                .descendants()
                .filter_map(ElementRef::wrap)
                .collect();
            let reach = Reach::read(root, document, &elements, &facts);
            let [literal, scope, context] = reach_by_ancestors(root, document, &facts);
            for &element in &elements {
                let name = element
                    .value()
                    .attribute("id")
                    .unwrap_or(element.value().name());
                let key = element_key(element);
                assert_eq!(
                    reach.literal(element),
                    literal.contains(&key),
                    "literal {name}"
                );
                assert_eq!(
                    reach.in_scope(element),
                    scope.contains(&key),
                    "scope {name}"
                );
                if scope.contains(&key) {
                    assert_eq!(
                        reach.context(element),
                        context.contains(&key),
                        "context {name}"
                    );
                }
            }
            counts[0] += literal.len();
            counts[1] += scope.len();
            counts[2] += context.len();
        }
        assert!(counts.iter().all(|count| *count > 3), "{counts:?}");
    }

    #[test]
    fn saved_pages_take_the_first_base_then_the_first_canonical_link() {
        fn by_selectors(document: ElementRef<'_>) -> Option<(Url, bool)> {
            let web = |value: &str| {
                Url::parse(value.trim()).ok().filter(|url| {
                    matches!(url.scheme(), "http" | "https") && url.host_str().is_some()
                })
            };
            if let Some(base) = document
                .select(&selector("base[href]"))
                .next()
                .and_then(|node| web(node.value().attribute("href")?))
            {
                return Some((base, true));
            }
            document
                .select(&selector("link[rel][href]"))
                .find(|node| node.value().attribute("rel").is_some_and(canonical_rel))
                .and_then(|node| web(node.value().attribute("href")?))
                .map(|canonical| {
                    let directory = !canonical.path().trim_matches('/').is_empty();
                    (canonical, directory)
                })
        }
        let mut found = 0;
        for head in [
            r#"<base><base href="https://a.test/dir/"><base href="https://b.test/">"#,
            r#"<base href="/relative"><link rel="canonical" href="https://c.test/post">"#,
            r#"<link rel="canonical"><link rel="Alternate CANONICAL" href=" https://d.test/ ">"#,
            r#"<link rel="icon" href="https://e.test/i"><link href="https://e.test/x" rel="canonical">"#,
            r#"<link rel="canonical" href="ftp://f.test/"><link rel="canonical" href="https://g.test/">"#,
            "<title>No address</title>",
        ] {
            for document in [
                format!("<html><head>{head}</head><body><p>x</p></body></html>"),
                format!("<p>x</p><base href=\"https://h.test/late/\">{head}"),
            ] {
                let document = Html::parse_document(&document);
                let expected = by_selectors(document.root_element());
                assert_eq!(saved_page_base(document.root_element()), expected, "{head}");
                found += usize::from(expected.is_some());
            }
        }
        assert!(found >= 7, "{found}");
    }

    #[test]
    fn saved_page_links_resolve_as_before_with_one_parse() {
        fn twice(value: &str, link_base: Option<&(Url, bool)>) -> Option<String> {
            let trimmed = value.trim();
            match link_base {
                Some((base, directory))
                    if !trimmed.starts_with('#')
                        && Url::parse(trimmed).is_err()
                        && (*directory || trimmed.starts_with('/')) =>
                {
                    safe_url(trimmed, Some(base))
                }
                _ => safe_url(trimmed, None),
            }
        }
        let bases = [
            None,
            Some((Url::parse("https://example.test/dir/page").unwrap(), true)),
            Some((Url::parse("https://example.test/").unwrap(), false)),
        ];
        for value in [
            "#part",
            "",
            "  /root  ",
            "file.html",
            "../up",
            "//cdn.test/a",
            "?q=1",
            "https://Other.test/P?q#f",
            "mailto:a@b.test",
            "tel:+1",
            "javascript:alert(1)",
            "data:text/html,x",
            "a:b",
            "http://exa mple/",
            "x\u{0}y",
            "<tag>\"",
        ] {
            for base in &bases {
                assert_eq!(
                    saved_page_link(value, base.as_ref()),
                    twice(value, base.as_ref()),
                    "{value:?} {base:?}"
                );
            }
        }
    }

    #[test]
    fn word_counts_read_ascii_bytes_and_other_characters_alike() {
        fn by_characters(text: &str) -> usize {
            let cjk = |ch: char| matches!(ch as u32, 0x3040..=0x30ff | 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xac00..=0xd7af | 0xf900..=0xfaff | 0x20000..=0x2a6df);
            let (mut words, mut in_word) = (0, false);
            for ch in text.chars() {
                if cjk(ch) {
                    words += 1;
                    in_word = false;
                } else if ch.is_alphanumeric() || ch == '_' {
                    words += usize::from(!in_word);
                    in_word = true;
                } else {
                    in_word = false;
                }
            }
            words
        }
        for text in [
            "",
            "one",
            "two words",
            "snake_case, camel42 and __x__",
            "naïve café—déjà vu",
            "日本語のテキスト",
            "한국어 문장입니다",
            "mixed日本text and 漢字x",
            "emoji 🎉 between",
            "tab\tnew\nline\r\n",
            "٣٤ digits ²",
            "ǅ title case",
            "\u{20000}\u{2a6df}\u{2a6e0}x",
            "end.",
        ] {
            assert_eq!(count_words(text), by_characters(text), "{text:?}");
        }
        assert_eq!(count_words("日本 word"), 3);
    }

    #[test]
    fn inline_code_keeps_its_line_breaks_as_text() {
        // Code is preformatted itself, not only inside a `pre` or `code`.
        let page = extract_html(
            "<article><p>Run <code>make\n  all</code> now.</p></article>",
            None,
        )
        .unwrap();
        assert_eq!(page.markdown, "Run `make\n  all` now.");
    }

    #[test]
    fn hidden_articles_do_not_compete_with_the_visible_one() {
        let visible = "The visible article explains its subject at some length. ".repeat(6);
        let hidden = "A hidden article that would be another candidate. ".repeat(6);
        let page = extract_html(
            &format!(
                "<body><div hidden><article><p>{hidden}</p></article></div><article><p>{visible}</p></article><div><p>A short line beside it.</p></div></body>"
            ),
            None,
        )
        .unwrap();
        assert!(
            page.markdown.contains("visible article"),
            "{}",
            page.markdown
        );
        assert!(!page.markdown.contains("short line"), "{}", page.markdown);
    }

    #[test]
    fn note_reference_markers_are_numbers_in_brackets_or_note_signs() {
        let candidate = |text: &str| {
            let document = Html::parse_fragment(&format!("<a href=\"#n\">{text}</a>"));
            let link = document.select(&selector("a")).next().unwrap();
            reference_candidate(link)
        };
        for marker in ["1", "[2]", "(3).", " 4 ", "*", "†‡", "[[5]]", "<b>6</b>"] {
            assert!(candidate(marker), "{marker:?}");
        }
        for text in [
            "1a", "see 1", "", "0", "12345", "1 2", "Note", "[1] more", "1\u{a0}x", "[\n7\n]",
        ] {
            assert!(!candidate(text), "{text:?}");
        }
    }
}
