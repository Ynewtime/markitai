use super::Attribute;
use super::facts::{Fact, Facts};
use scraper::{ElementRef, Html, Node};

fn token(element: ElementRef<'_>, attribute: &str, expected: &str) -> bool {
    any_token(element, attribute, &[expected])
}

/// Whether a word of the attribute is one of `expected`, in any case, reading
/// the attribute once.
fn any_token(element: ElementRef<'_>, attribute: &str, expected: &[&str]) -> bool {
    element.value().attribute(attribute).is_some_and(|value| {
        value.split_ascii_whitespace().any(|part| {
            expected
                .iter()
                .any(|expected| part.eq_ignore_ascii_case(expected))
        })
    })
}

/// Whether a class word or the id is one of `names`. The attributes are read
/// once per element, not once per name: the lists are long and every element
/// of a page is asked several times.
fn named(element: ElementRef<'_>, names: &[&str]) -> bool {
    let value = element.value();
    value
        .attribute("class")
        .into_iter()
        .flat_map(str::split_ascii_whitespace)
        .chain(value.attribute("id"))
        .any(|word| names.iter().any(|name| word.eq_ignore_ascii_case(name)))
}

pub(super) fn note(facts: &Facts, element: ElementRef<'_>) -> bool {
    facts.remember(element, Fact::Note, || {
        super::note_context(facts, element)
            || any_token(
                element,
                "role",
                &[
                    "doc-footnote",
                    "doc-endnote",
                    "doc-endnotes",
                    "doc-bibliography",
                ],
            )
            || named(
                element,
                &[
                    "footnotes",
                    "endnotes",
                    "sidenote",
                    "sidenotes",
                    "footnote-content",
                    "footnoteContent",
                ],
            )
    })
}

fn ancillary(facts: &Facts, element: ElementRef<'_>) -> bool {
    if element.value().name() == "aside" || note(facts, element) {
        return true;
    }
    if !matches!(element.value().name(), "div" | "section" | "ul" | "ol") {
        return false;
    }
    first_heading(element).is_some_and(|heading| {
        let mut length = 0usize;
        let label = heading.text().try_fold(String::new(), |mut label, part| {
            length = length.saturating_add(part.len());
            if length > 64 {
                return None;
            }
            label.push_str(part);
            Some(label)
        });
        label.is_some_and(|label| {
            let label = label.trim();
            label.eq_ignore_ascii_case("footnotes") || label.eq_ignore_ascii_case("endnotes")
        })
    })
}

pub(super) fn heading(element: ElementRef<'_>) -> bool {
    matches!(
        element.value().name(),
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
    )
}

fn first_heading(mut element: ElementRef<'_>) -> Option<ElementRef<'_>> {
    // A heading may be wrapped for layout, but arbitrary descendant headings do
    // not label their outer article or page.
    for _ in 0..3 {
        if heading(element) {
            return Some(element);
        }
        if !matches!(element.value().name(), "div" | "section" | "aside") {
            return None;
        }
        let mut first = None;
        for child in element.children() {
            match child.value() {
                Node::Text(text) if !text.trim().is_empty() => return None,
                Node::Element(_) => {
                    let child = ElementRef::wrap(child)?;
                    if super::is_hidden(child) {
                        continue;
                    }
                    first = Some(child);
                    break;
                }
                _ => {}
            }
        }
        element = first?;
    }
    heading(element).then_some(element)
}

fn related_heading(element: ElementRef<'_>) -> bool {
    let Some(heading) = first_heading(element) else {
        return false;
    };
    let mut label = String::new();
    for part in heading.text() {
        if label.len().saturating_add(part.len()) > 100 {
            return false;
        }
        label.push_str(part);
    }
    let label = label.split_whitespace().collect::<Vec<_>>().join(" ");
    [
        "related stories",
        "related posts",
        "related articles",
        "recommended stories",
        "recommended posts",
        "read next",
    ]
    .iter()
    .any(|expected| label.eq_ignore_ascii_case(expected))
}

fn card(element: ElementRef<'_>) -> bool {
    let name = element.value().name();
    let marked = named(
        element,
        &[
            "card",
            "o-card",
            "story-card",
            "post-card",
            "recommendation-card",
        ],
    );
    if !marked && !matches!(name, "article" | "a") {
        return false;
    }
    let mut stack = vec![(element, 0)];
    let mut linked = false;
    let mut title_or_image = false;
    while let Some((node, depth)) = stack.pop() {
        if depth > 5 {
            return false;
        }
        if super::is_hidden(node) {
            continue;
        }
        let tag = node.value().name();
        if !marked && matches!(tag, "p" | "pre" | "table" | "blockquote") {
            return false;
        }
        linked |= tag == "a" && outward(node);
        title_or_image |= heading(node) || tag == "img";
        stack.extend(node.child_elements().map(|child| (child, depth + 1)));
    }
    linked && title_or_image
}

fn related_cards(facts: &Facts, element: ElementRef<'_>) -> bool {
    if !related_heading(element) {
        return false;
    }
    let mut count = 0;
    let mut stack = vec![(element, 0)];
    while let Some((node, depth)) = stack.pop() {
        if depth > 6 {
            return false;
        }
        if super::is_hidden(node) || heading(node) {
            continue;
        }
        if node != element && card(node) {
            count += 1;
            continue;
        }
        if matches!(node.value().name(), "p" | "pre" | "table" | "blockquote") || note(facts, node)
        {
            return false;
        }
        // Unlabelled prose beside cards belongs to the article, not a widget.
        if node
            .children()
            .any(|child| matches!(child.value(), Node::Text(text) if !text.trim().is_empty()))
        {
            return false;
        }
        stack.extend(node.child_elements().map(|child| (child, depth + 1)));
    }
    count >= 2
}

/// Whether a page's framework classes hide an element as defuddle reads them:
/// `hidden` or `invisible`, also behind a variant (`md:hidden`, a site's own
/// `not-machine:hidden`), or a CSS-module `isHidden-…` name, unless a
/// responsive class shows it again (`hidden md:block`). Arbitrary variants
/// (`[&_.x]:hidden`) target other elements. Fragments (books, email) come
/// without the site's style sheet, so this is full-page chrome only.
fn hidden_class(element: ElementRef<'_>) -> bool {
    let Some(classes) = element.value().attribute("class") else {
        return false;
    };
    let mut hidden = false;
    for class in classes.split_ascii_whitespace() {
        if let Some((variant, utility)) = class.split_once(':')
            && (matches!(variant, "sm" | "md" | "lg" | "xl" | "2xl")
                || ((variant.starts_with("min-[") || variant.starts_with("max-["))
                    && variant.ends_with(']')))
            && ["block", "flex", "grid", "inline", "table", "contents"]
                .iter()
                .any(|shown| utility.starts_with(shown))
        {
            return false;
        }
        if class.contains('[') {
            continue;
        }
        let bare = class.rsplit(':').next().unwrap_or(class);
        let module = bare
            .strip_prefix("is")
            .map(|rest| rest.strip_prefix(['-', '_']).unwrap_or(rest))
            .and_then(|rest| {
                rest.strip_prefix("Hidden")
                    .or_else(|| rest.strip_prefix("hidden"))
            })
            .is_some_and(|rest| rest.starts_with(['-', '_']));
        hidden |= matches!(bare, "hidden" | "invisible") || module;
    }
    hidden
}

/// Math keeps its hidden accessible copies (MathML beside a rendering).
fn contains_math(element: ElementRef<'_>) -> bool {
    element
        .descendants()
        .filter_map(ElementRef::wrap)
        .any(|node| {
            node.value().name() == "math"
                || node.value().attribute("data-mathml").is_some()
                || token(node, "class", "katex-mathml")
        })
}

/// Full-page chrome only. Fragment conversion must leave this policy disabled:
/// the same table of contents can be essential text in a book or email.
pub(super) fn excluded(facts: &Facts, element: ElementRef<'_>) -> bool {
    facts.remember(element, Fact::Excluded, || chrome(facts, element))
}

fn chrome(facts: &Facts, element: ElementRef<'_>) -> bool {
    if hidden_class(element) && !contains_math(element) {
        return true;
    }
    // A tooltip labels a control (a download button's "Save"); it is not text.
    if token(element, "role", "tooltip") {
        return true;
    }
    // MediaWiki's section edit links, "From Wikipedia" tagline, redirect
    // note and skip links.
    if named(
        element,
        &[
            "mw-editsection",
            "mw-jump-link",
            "siteSub",
            "contentSub",
            "jump-to-nav",
        ],
    ) {
        return true;
    }
    if note(facts, element)
        || !matches!(
            element.value().name(),
            "div" | "section" | "aside" | "nav" | "ul" | "ol"
        )
    {
        return false;
    }
    if any_token(element, "role", &["doc-toc", "navigation"]) {
        return true;
    }
    if named(
        element,
        &[
            "related-posts",
            "related-articles",
            "related-stories",
            "recommended-posts",
            "recommended-articles",
            "post-related",
            "read-next",
            "jp-relatedposts",
            "newsletter-signup",
            "newsletter-form",
            "subscribe-widget",
            "subscription-widget",
            "subscription-widget-wrap",
            "subscription-box",
            "post-subscribe",
            "subscribe-form",
            "cta-section",
            "share-buttons",
            "social-share",
            "sharing-buttons",
            "post-share",
            "share-tools",
            "article-share",
            "cookie-banner",
            "cookie-consent-banner",
            // GitHub's (Primer) page sidebar: assignees, labels,
            // notifications and sign-up prompts beside an issue.
            "Layout-sidebar",
            "discussion-sidebar",
        ],
    ) {
        return true;
    }
    if named(
        element,
        &["toc", "toc-container", "table-of-contents", "article-toc"],
    ) && element
        .child_elements()
        .any(|child| matches!(child.value().name(), "ul" | "ol"))
    {
        return true;
    }
    if element
        .value()
        .attribute("data-testid")
        .is_some_and(|value| {
            matches!(
                value,
                "issue-metadata-sticky"
                    | "issue-viewer-metadata-container"
                    | "issue-viewer-metadata-pane"
                    | "comment-header-right-side-items"
            )
        })
    {
        return true;
    }
    matches!(element.value().name(), "div" | "section" | "aside") && related_cards(facts, element)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Other,
    Main,
    Content,
    Readme,
    Discussion,
}

fn kind(element: ElementRef<'_>) -> Kind {
    let tag = element.value().name();
    if element.value().attribute("data-testid") == Some("issue-viewer-container") {
        return Kind::Discussion;
    }
    if tag == "article"
        && token(element, "class", "markdown-body")
        && token(element, "class", "entry-content")
        && token(element, "itemprop", "text")
    {
        return Kind::Readme;
    }
    if tag == "main" || token(element, "role", "main") {
        return Kind::Main;
    }
    if tag == "article"
        || token(element, "itemprop", "articleBody")
        || token(element, "role", "article")
        || matches!(tag, "div" | "section")
            && named(
                element,
                &[
                    "article",
                    "article-body",
                    "article-content",
                    "article-text",
                    "entry-content",
                    "post-content",
                    "post-body",
                    "blog-post-content",
                    "story-body",
                    "story-content",
                    // The rest of defuddle's specific entry points.
                    "js-article-content",
                    "article_post",
                    "article-wrapper",
                    "content-article",
                    "instapaper_body",
                ],
            )
    {
        return Kind::Content;
    }
    Kind::Other
}

/// A page whose content region has its own precise boundary (a repository
/// README, a complete issue) and needs no reading of its furniture.
pub(super) fn structured_page(root: ElementRef<'_>) -> bool {
    matches!(kind(root), Kind::Readme | Kind::Discussion)
}

pub(super) fn discarded(facts: &Facts, element: ElementRef<'_>) -> bool {
    facts.hidden(element)
        || excluded(facts, element)
        || matches!(
            element.value().name(),
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
}

/// Words of class and id names that mark page furniture around an article:
/// recommendation and share rails, newsletter and cookie boxes, tables of
/// contents, disclaimers. They only weigh the choice of region; nothing is
/// removed for its name.
const FURNITURE_WORDS: &[&str] = &[
    "recommend",
    "recommended",
    "recommendations",
    "related",
    "share",
    "sharing",
    "social",
    "newsletter",
    "newsletters",
    "subscribe",
    "subscription",
    "disclaimer",
    "disclaimers",
    "promo",
    "sponsor",
    "sponsored",
    "advert",
    "advertisement",
    "cookie",
    "cookies",
    "toc",
    "breadcrumb",
    "breadcrumbs",
    "pagination",
    "sidebar",
    "widget",
    "footer",
    "signup",
    // Google's and Yahoo's markers for text that is not the page's content.
    "nocontent",
];

/// Whether a class or id name, split into words at `-`, `_` and spaces,
/// names page furniture.
fn furniture(element: ElementRef<'_>) -> bool {
    ["class", "id"].iter().any(|attribute| {
        element.value().attribute(attribute).is_some_and(|value| {
            value
                .split(|c: char| c.is_ascii_whitespace() || matches!(c, '-' | '_'))
                .any(|word| {
                    FURNITURE_WORDS
                        .iter()
                        .any(|furniture| word.eq_ignore_ascii_case(furniture))
                })
        })
    })
}

/// Whether a class or id names a featured comment set into an article: a top,
/// featured, hot, best or pinned comment, in any spelling (`top-comment`,
/// `hotComment`, `featured_comments`, the BEM block of `top-comment__body`).
pub(super) fn featured_comment(element: ElementRef<'_>) -> bool {
    ["class", "id"]
        .iter()
        .filter_map(|attribute| element.value().attribute(attribute))
        .flat_map(str::split_ascii_whitespace)
        .any(|token| {
            let block = token.split("__").next().unwrap_or(token);
            if block.len() > 32 || !block.contains("omment") {
                return false;
            }
            let name: String = block
                .chars()
                .filter(|ch| !matches!(ch, '-' | '_'))
                .map(|ch| ch.to_ascii_lowercase())
                .collect();
            name.strip_suffix("comments")
                .or_else(|| name.strip_suffix("comment"))
                .is_some_and(|kind| matches!(kind, "top" | "featured" | "hot" | "best" | "pinned"))
        })
}

/// Whether an article is a teaser card: its title is a link to another page
/// (a linked heading, or a heading inside a link).
pub(super) fn teaser(element: ElementRef<'_>) -> bool {
    element
        .descendants()
        .filter_map(ElementRef::wrap)
        .any(|node| {
            heading(node)
                && (node.select(&TEASER_LINK).next().is_some()
                    || node
                        .ancestors()
                        .filter_map(ElementRef::wrap)
                        .take_while(|parent| *parent != element)
                        .any(|parent| parent.value().name() == "a" && outward(parent)))
                && node.select(&TEASER_LINK).all(outward)
        })
}

/// A link to another page, not a fragment of this one.
fn outward(link: ElementRef<'_>) -> bool {
    link.value().attribute("href").is_some_and(|href| {
        let href = href.trim();
        !href.is_empty() && !href.starts_with('#') && !href.starts_with("javascript:")
    })
}

static TEASER_LINK: std::sync::LazyLock<scraper::Selector> =
    std::sync::LazyLock::new(|| scraper::Selector::parse("a[href]").expect("static selector"));

/// In-page tables of contents under `root`: each outermost list whose links
/// all point to headings of the page, with no other text than numbering,
/// taken together with the wrappers that hold nothing else than a short
/// title for it, and with a pair of rules framing it. The headings already
/// carry that structure, and the links would not resolve in Markdown.
pub(super) fn contents<'a>(root: ElementRef<'a>, document: ElementRef<'a>) -> Vec<ElementRef<'a>> {
    const MIN_LINKS: usize = 3;
    // The page's heading targets, read for the first list that could be a
    // table of contents (most pages have none).
    let mut targets = None;
    let mut found = Vec::new();
    for list in root.descendants().filter_map(ElementRef::wrap) {
        if !matches!(list.value().name(), "ul" | "ol")
            || list
                .ancestors()
                .filter_map(ElementRef::wrap)
                .any(|parent| matches!(parent.value().name(), "ul" | "ol" | "li"))
        {
            continue;
        }
        // Every link names a fragment, and the other text is numbering.
        let mut fragments = Vec::new();
        let fragment_links = list.descendants().all(|node| match node.value() {
            Node::Element(element) if element.name() == "a" => {
                let fragment = element
                    .attribute("href")
                    .and_then(|href| href.trim().strip_prefix('#'));
                fragments.extend(fragment);
                fragment.is_some()
            }
            Node::Text(text) => {
                !text.chars().any(char::is_alphabetic)
                    || node
                        .ancestors()
                        .filter_map(ElementRef::wrap)
                        .take_while(|parent| *parent != list)
                        .any(|parent| parent.value().name() == "a")
            }
            _ => true,
        });
        if !fragment_links || fragments.len() < MIN_LINKS {
            continue;
        }
        let targets = targets.get_or_insert_with(|| heading_targets(document));
        if !fragments
            .iter()
            .all(|fragment| targets.contains(decoded(fragment).as_ref()))
        {
            continue;
        }
        let mut wrapper = list;
        while let Some(parent) = wrapper.parent().and_then(ElementRef::wrap) {
            if parent == root
                || matches!(parent.value().name(), "body" | "html" | "main" | "article")
            {
                break;
            }
            let titles_only = parent
                .children()
                .filter(|child| child.id() != wrapper.id())
                .all(|child| match child.value() {
                    Node::Text(text) => text.trim().is_empty(),
                    Node::Element(_) => ElementRef::wrap(child).is_some_and(|element| {
                        super::is_hidden(element) || contents_title(element)
                    }),
                    _ => true,
                });
            if !titles_only {
                break;
            }
            wrapper = parent;
        }
        let rule = |element: Option<ElementRef<'a>>| {
            element.filter(|element| element.value().name() == "hr")
        };
        if let (Some(before), Some(after)) = (
            rule(wrapper.prev_siblings().find_map(ElementRef::wrap)),
            rule(wrapper.next_siblings().find_map(ElementRef::wrap)),
        ) {
            found.extend([before, after]);
        }
        found.push(wrapper);
    }
    found
}

/// The ids and anchor names of the page that label a heading: on the heading,
/// around it, or on an empty element right before it.
fn heading_targets(document: ElementRef<'_>) -> std::collections::HashSet<&str> {
    let mut targets = std::collections::HashSet::new();
    for node in document.descendants().filter_map(ElementRef::wrap) {
        let value = node.value();
        let Some(id) = value.attribute("id").or_else(|| {
            (value.name() == "a")
                .then(|| value.attribute("name"))
                .flatten()
        }) else {
            continue;
        };
        let labels_heading = heading(node)
            || node
                .ancestors()
                .filter_map(ElementRef::wrap)
                .take(4)
                .any(heading)
            || first_heading(node).is_some()
            || (node.text().all(|text| text.trim().is_empty())
                && node
                    .next_siblings()
                    .find_map(ElementRef::wrap)
                    .is_some_and(heading));
        if labels_heading {
            targets.insert(id);
        }
    }
    targets
}

/// A table of contents' title: a few words without links to other places,
/// images, lists or tables ("Contents", a hide toggle).
fn contents_title(element: ElementRef<'_>) -> bool {
    const MAX_WORDS: usize = 6;
    element
        .text()
        .flat_map(str::split_whitespace)
        .nth(MAX_WORDS)
        .is_none()
        && !element
            .descendants()
            .filter_map(ElementRef::wrap)
            .any(|node| {
                matches!(node.value().name(), "img" | "ul" | "ol" | "table")
                    || (node.value().name() == "a" && outward(node))
            })
}

/// A link fragment with `%XX` escapes decoded, as ids are compared unescaped.
fn decoded(fragment: &str) -> std::borrow::Cow<'_, str> {
    if !fragment.contains('%') {
        return fragment.into();
    }
    let bytes = fragment.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let hex = |offset: usize| {
            bytes
                .get(index + offset)
                .and_then(|byte| (*byte as char).to_digit(16))
        };
        if bytes[index] == b'%'
            && let (Some(high), Some(low)) = (hex(1), hex(2))
        {
            output.push((high * 16 + low) as u8);
            index += 3;
        } else {
            output.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8_lossy(&output).into_owned().into()
}

struct Scored<'a> {
    element: ElementRef<'a>,
    parent: Option<usize>,
    content_parent: Option<usize>,
    end: usize,
    score: usize,
    /// Score inside furniture-named subtrees, counted once per outermost one.
    furniture: usize,
    kind: Kind,
}

/// Choose one coherent content region. Scoring is accumulated once per DOM node;
/// nested main/article candidates never rescan their complete text subtrees.
pub(super) fn select<'a>(document: &'a Html, facts: &Facts) -> ElementRef<'a> {
    let mut nodes: Vec<Scored<'_>> = Vec::new();
    let mut stack = vec![(document.root_element(), None, None, false)];
    let mut body = None;
    while let Some((element, parent, content_parent, ancillary_parent)) = stack.pop() {
        if discarded(facts, element) {
            continue;
        }
        let index = nodes.len();
        let kind = kind(element);
        let ancillary = ancillary_parent || ancillary(facts, element);
        let score = if ancillary {
            0
        } else {
            element
                .children()
                .filter_map(|child| match child.value() {
                    Node::Text(text) => Some(text.trim().len()),
                    _ => None,
                })
                .fold(0usize, usize::saturating_add)
                .saturating_add(if element.value().name() == "p" { 40 } else { 0 })
        };
        nodes.push(Scored {
            element,
            parent,
            content_parent,
            end: index + 1,
            score,
            furniture: 0,
            kind,
        });
        if element.value().name() == "body" {
            body = Some(index);
        }
        let content_parent = if matches!(kind, Kind::Content | Kind::Readme | Kind::Discussion) {
            Some(index)
        } else {
            content_parent
        };
        stack.extend(
            element
                .children()
                .rev()
                .filter_map(ElementRef::wrap)
                .map(|child| (child, Some(index), content_parent, ancillary)),
        );
    }
    for index in (0..nodes.len()).rev() {
        // Children come after their parent, so each subtree is complete here.
        if furniture(nodes[index].element) {
            nodes[index].furniture = nodes[index].score;
        }
        if let Some(parent) = nodes[index].parent {
            nodes[parent].score = nodes[parent].score.saturating_add(nodes[index].score);
            nodes[parent].furniture = nodes[parent]
                .furniture
                .saturating_add(nodes[index].furniture);
            nodes[parent].end = nodes[parent].end.max(nodes[index].end);
        }
    }
    let Some(fallback) = body.or_else(|| (!nodes.is_empty()).then_some(0)) else {
        return document.root_element();
    };
    let root = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.kind == Kind::Main && node.score > 0)
        .max_by_key(|(_, node)| node.score)
        .map_or(fallback, |(index, _)| index);
    let in_root = |index: usize| index >= root && index < nodes[root].end;
    // Explicit repository documentation and the complete issue viewer keep
    // their content together; an isolated comment's markdown-body is not one.
    if let Some((_, node)) = nodes
        .iter()
        .enumerate()
        .filter(|(index, node)| {
            in_root(*index)
                && node.score > 0
                && matches!(node.kind, Kind::Readme | Kind::Discussion)
        })
        .max_by_key(|(_, node)| node.score)
    {
        return node.element;
    }
    let candidates: Vec<&Scored<'_>> = nodes
        .iter()
        .enumerate()
        .filter(|(index, node)| {
            in_root(*index)
                && node.kind == Kind::Content
                && node.score > 0
                && node.content_parent.is_none_or(|parent| !in_root(parent))
        })
        .map(|(_, node)| node)
        .collect();
    // Portfolios, indexes and discussion pages need all sibling articles;
    // teaser cards linking to other articles beside one article do not.
    let (teasers, articles): (Vec<&Scored<'_>>, Vec<&Scored<'_>>) = candidates
        .iter()
        .copied()
        .partition(|node| teaser(node.element));
    let (candidate, teaser_score) = match (articles.as_slice(), candidates.len()) {
        (_, 0) => return nodes[root].element,
        (_, 1) => (candidates[0], 0),
        ([article], _) => (*article, teasers.iter().map(|node| node.score).sum()),
        _ => return nodes[root].element,
    };
    // Keep substantial introductions/conclusions outside the named region;
    // furniture-named rails and boxes outside it are not such text.
    let outside = nodes[root].score.saturating_sub(candidate.score);
    let outside_furniture = nodes[root]
        .furniture
        .saturating_sub(candidate.furniture)
        .saturating_add(teaser_score);
    let weighed = candidate
        .score
        .saturating_add(outside.saturating_sub(outside_furniture));
    if candidate.score.saturating_mul(5) >= weighed.saturating_mul(3) {
        candidate.element
    } else {
        nodes[root].element
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scraper::Selector;

    fn excluded(element: ElementRef<'_>) -> bool {
        super::excluded(&Facts::new(element), element)
    }

    fn select(document: &Html) -> ElementRef<'_> {
        super::select(document, &Facts::new(document.root_element()))
    }

    fn element<'a>(document: &'a Html, query: &str) -> ElementRef<'a> {
        document
            .select(&Selector::parse(query).unwrap())
            .next()
            .unwrap()
    }

    #[test]
    fn names_match_whole_class_words_and_ids_in_any_case() {
        let document = Html::parse_document(
            "<main><p>Body text.</p><div class='Widget RELATED-POSTS'><p>x</p></div>\
             <div id='SiteSub'>From Wikipedia</div><div class='related-postsx'><p>y</p></div>\
             <div id='related-posts extra'><p>z</p></div></main>",
        );
        assert!(excluded(element(&document, ".Widget")));
        assert!(excluded(element(&document, "#SiteSub")));
        assert!(!excluded(element(&document, ".related-postsx")));
        // An id is one name, not words.
        assert!(!excluded(element(&document, "main > div:last-child")));
    }

    #[test]
    fn related_cards_need_both_label_and_repeated_card_structure() {
        let document = Html::parse_document(
            r#"<main><section id="widget"><div><h3>Related Posts</h3></div><div><a href="/one"><article><h3>One</h3><span>3 min</span></article></a></div><div><a href="/two"><article><h3>Two</h3></article></a></div></section><section id="prose"><h2>Related Posts</h2><p>Our article compares publishing systems.</p><a href="/one"><h3>Study one</h3></a><a href="/two"><h3>Study two</h3></a></section><section id="ordinary"><h2>Recommended architecture</h2><a href="/one"><h3>One</h3></a><a href="/two"><h3>Two</h3></a></section></main>"#,
        );
        assert!(excluded(element(&document, "#widget")));
        assert!(!excluded(element(&document, "#prose")));
        assert!(!excluded(element(&document, "#ordinary")));
        assert!(!excluded(element(&document, "main")));
    }

    #[test]
    fn inline_story_widget_does_not_hide_surrounding_article() {
        let document = Html::parse_document(
            r#"<main><article id="story"><p>Opening narrative.</p><div class="injected-story-block"><h2>Related Stories</h2><div class="story-list"><article class="o-card"><a href="/one"><img src="one.png"></a><h3><a href="/one">One</a></h3></article><article class="o-card"><h3><a href="/two">Two</a></h3></article></div></div><p>Concluding evidence.</p></article></main>"#,
        );
        assert_eq!(select(&document).value().attribute("id"), Some("story"));
        assert!(excluded(element(&document, ".injected-story-block")));
        assert!(!excluded(element(&document, "#story")));
    }

    #[test]
    fn a_github_page_sidebar_is_page_chrome() {
        let document = Html::parse_document(
            r#"<body><div class="Layout-main"><p>The issue.</p></div><div class="Layout-sidebar"><div class="sidebar-section"><p>Subscribe</p></div></div></body>"#,
        );
        assert!(excluded(element(&document, ".Layout-sidebar")));
        assert!(!excluded(element(&document, ".Layout-main")));
    }

    #[test]
    fn furniture_rails_do_not_keep_the_page_around_a_named_article() {
        // A short article beside long recommendation, newsletter and
        // disclaimer rails: the rails are not an introduction to keep.
        let rail = "Recommended reading about other subjects entirely. ".repeat(12);
        let document = Html::parse_document(&format!(
            r#"<body><div class="js-article-content" id="story"><h2>Section</h2><p>The article itself is short.</p></div><div class="recommended-footer"><p>{rail}</p></div><div class="newsletter-box"><p>{rail}</p></div><div class="site-disclaimers"><p>{rail}</p></div></body>"#
        ));
        assert_eq!(select(&document).value().attribute("id"), Some("story"));
        // The same rails' text in unnamed blocks is page text to keep.
        let document = Html::parse_document(&format!(
            r#"<body><div class="js-article-content" id="story"><h2>Section</h2><p>The article itself is short.</p></div><div><p>{rail}</p></div></body>"#
        ));
        assert_eq!(select(&document).value().name(), "body");
    }

    #[test]
    fn plain_article_names_and_nocontent_rails_choose_the_article() {
        let rail = "Other stories from this site this week. ".repeat(12);
        for name in ["article", "article-text"] {
            let document = Html::parse_document(&format!(
                r#"<body><div class="hero"><a href="/story">Story</a></div><div class="{name}" id="story"><p>The article itself is short.</p></div><div class="panel nocontent"><p>{rail}</p></div></body>"#
            ));
            assert_eq!(
                select(&document).value().attribute("id"),
                Some("story"),
                "{name}"
            );
        }
    }

    #[test]
    fn portfolio_and_multiple_top_level_articles_are_not_reduced_to_one_card() {
        for wrapper in ["main", "body"] {
            let document = Html::parse_document(&format!(
                "<{wrapper}><h1>Career</h1><article><h2>Engineering</h2><p>First career.</p></article><article><h2>Research</h2><p>Second career with longer text.</p></article></{wrapper}>"
            ));
            assert_eq!(select(&document).value().name(), wrapper);
        }
    }

    #[test]
    fn teaser_cards_beside_one_article_do_not_keep_the_page() {
        // An unlabeled grid of cards linking to other articles.
        let body = "The article explains the subject in some detail. ".repeat(8);
        let summary = "A summary of another article on this site. ".repeat(3);
        let cards = |href: &str| {
            (1..=3)
                .map(|n| {
                    format!(r#"<article><a href="{href}{n}"><figure></figure><h2>Other {n}</h2></a><section><p>{summary}</p></section></article>"#)
                })
                .collect::<String>()
        };
        let page = |cards: &str| {
            Html::parse_document(&format!(
                r#"<main><article id="story"><p>{body}</p></article><div class="list"><div>{cards}</div></div></main>"#
            ))
        };
        assert_eq!(
            select(&page(&cards("/article/"))).value().attribute("id"),
            Some("story")
        );
        // Cards linking within this page, and a second full article, are
        // sibling content to keep.
        assert_eq!(select(&page(&cards("#part"))).value().name(), "main");
        let second = format!(r#"<article><h2>Second</h2><p>{summary}</p></article>"#);
        assert_eq!(
            select(&page(&format!("{}{second}", cards("/article/"))))
                .value()
                .name(),
            "main"
        );
    }

    #[test]
    fn named_content_excludes_independent_cta_without_losing_heading() {
        let document = Html::parse_document(
            "<main><section class='blog-post-content'><h1>Release</h1><p>Full release details.</p></section><section class='cta-section'><h2>Work with us</h2><p>Subscribe for updates.</p></section></main>",
        );
        assert!(token(select(&document), "class", "blog-post-content"));
        assert!(excluded(element(&document, ".cta-section")));
        let article = Html::parse_document(
            "<main><article><header><h1>Title outside content wrapper</h1></header><div class='entry-content'><p>Body.</p></div></article></main>",
        );
        assert_eq!(select(&article).value().name(), "article");
    }

    #[test]
    fn substantive_outside_prose_and_link_only_content_survive() {
        let document = Html::parse_document(
            "<main><p>A substantial independent introduction explains every example on this page.</p><article><p>One short example.</p></article><p>The conclusions compare the examples and record independent findings.</p></main>",
        );
        assert_eq!(select(&document).value().name(), "main");
        let links = Html::parse_document(
            "<article><h1>Reading list</h1><ul><li><a href='/one'>Study one</a></li><li><a href='/two'>Study two</a></li></ul></article>",
        );
        assert_eq!(select(&links).value().name(), "article");
        assert!(!excluded(element(&links, "ul")));
    }

    #[test]
    fn external_notes_and_asides_do_not_outweigh_a_short_explicit_article() {
        for notes in [
            "<div class='page__footnotes'><h3>Footnotes</h3><p>First lengthy definition.</p><ul><li>Evidence and further details.</li></ul><p>Second lengthy definition.</p></div>",
            "<section role='doc-endnotes'><p>First lengthy definition.</p><p>Second lengthy definition.</p></section>",
        ] {
            let document = Html::parse_document(&format!(
                "<article id='short'><p>A<sup>1</sup> and B<sup>2</sup>.</p></article><aside><p>Unrelated advertising exceeds the short article.</p><p>More advertising.</p></aside>{notes}"
            ));
            assert_eq!(select(&document).value().attribute("id"), Some("short"));
        }
    }

    #[test]
    fn ancillary_scoring_does_not_delete_article_asides_or_aside_only_pages() {
        let document = Html::parse_document(
            "<main><article><p>Claim.</p><aside><p>An essential caveat within the article.</p></aside></article></main>",
        );
        assert_eq!(select(&document).value().name(), "article");
        assert!(!excluded(element(&document, "aside")));
        let aside = Html::parse_document("<aside><p>The only content on this page.</p></aside>");
        assert_eq!(select(&aside).value().name(), "body");
        assert!(!excluded(element(&aside, "aside")));
    }

    #[test]
    fn github_readme_has_positive_evidence_and_comment_bodies_are_not_candidates() {
        let document = Html::parse_document(
            "<main><div><p>Repository file listing and controls dominate the surrounding layout.</p></div><article class='markdown-body entry-content' itemprop='text'><h1>README</h1><p>Install the package.</p></article></main>",
        );
        assert_eq!(
            select(&document).value().attribute("itemprop"),
            Some("text")
        );
        let discussion = Html::parse_document(
            "<main><p>Repository controls.</p><div data-testid='issue-viewer-container'><h1>Issue title</h1><div class='markdown-body'><p>Problem.</p></div><div class='react-comments-container'><div class='markdown-body'><p>Maintainer's solution.</p></div></div><div data-testid='issue-viewer-metadata-container'>Sidebar controls</div></div></main>",
        );
        assert_eq!(
            select(&discussion).value().attribute("data-testid"),
            Some("issue-viewer-container")
        );
        assert!(!excluded(element(&discussion, ".react-comments-container")));
        assert!(excluded(element(
            &discussion,
            "[data-testid='issue-viewer-metadata-container']"
        )));
    }

    #[test]
    fn hidden_content_cannot_win_selection_and_notes_override_widget_labels() {
        let document = Html::parse_document(
            "<div hidden><article><p>A long hidden candidate must not displace the visible document.</p></article></div><main><p>Visible.</p></main><section role='doc-endnotes' class='related-posts'><p>Referenced note.</p></section>",
        );
        assert_eq!(select(&document).value().name(), "main");
        assert!(!excluded(element(&document, "[role='doc-endnotes']")));
    }

    #[test]
    fn a_tooltip_labels_a_control_and_is_not_page_text() {
        let document = Html::parse_document(
            r#"<body><span><button aria-label="Save"></button><div role="tooltip">Save to Photos</div></span></body>"#,
        );
        assert!(excluded(element(&document, "[role='tooltip']")));
        assert!(!excluded(element(&document, "span")));
    }

    #[test]
    fn navigation_discussion_and_toc_heading_are_not_chrome_by_words_alone() {
        let document = Html::parse_document(
            "<article><h1>Navigation and subscriptions</h1><section id='toc'><h2>Table of contents design</h2><p>Navigation is an accessibility feature.</p></section><div class='comments'><p>Discuss subscribe widgets here.</p></div><aside class='toc-container'><h2>Contents</h2><ol><li><a href='#toc'>Design</a></li></ol></aside></article>",
        );
        assert!(!excluded(element(&document, "#toc")));
        assert!(!excluded(element(&document, ".comments")));
        assert!(excluded(element(&document, ".toc-container")));
    }
}
