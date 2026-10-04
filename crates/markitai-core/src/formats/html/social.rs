//! Posts on X (Twitter): the post itself, the author's own follow-up posts, its
//! media and the post it quotes, without avatars, player controls, counters,
//! replies from other accounts and timelines.
//!
//! Three page generations are read. The oldest marks its parts with
//! `data-testid` (`tweet`, `User-Name`, `tweetText`, `tweetPhoto`); the next
//! carries `data-tweet-id` on the post and no test ids; the server-rendered
//! page of late 2026 has neither: a post is an `article` holding engagement
//! buttons (`data-engagement-action`), its text is a `div[dir=auto]`, its date
//! the text of its `/status/<id>` link, and a quoted post is a nested `article`
//! in a clickable `div[role=link][data-href]`.
//! Long-form Articles have a separate `x-article-body` beside their heading;
//! a status page can embed the complete Article instead of ordinary post text.

use super::Attribute;
use super::{Announcement, Landmarks, render_clean, selector};
use crate::Result;
use scraper::{ElementRef, Html};
use serde_json::{Map, Value};
use url::Url;

fn inside(element: ElementRef<'_>, container: Option<ElementRef<'_>>) -> bool {
    container.is_some_and(|container| element.ancestors().any(|node| node.id() == container.id()))
}

fn has_engagement(element: ElementRef<'_>) -> bool {
    element
        .descendants()
        .filter_map(ElementRef::wrap)
        .any(|child| child.value().attribute("data-engagement-action").is_some())
}

/// Whether an `article` is a post. The two older generations tag it; the
/// newest has no tag, so its counters and buttons are the evidence, and
/// `untagged` says whether the page vouches for it (a status address or a
/// real column), because the cards of other Articles on an X Article page
/// look the same.
fn is_post(element: ElementRef<'_>, untagged: bool) -> bool {
    element.value().name() == "article"
        && (element.value().attribute("data-testid") == Some("tweet")
            || element.value().attribute("data-tweet-id").is_some()
            || (untagged && has_engagement(element)))
}

/// Whether an element with `data-engagement-action` sits in an `article`
/// inside `main`: a post of the newest page, which has no test ids
/// (`main article [data-engagement-action]`).
pub(super) fn is_engagement_in_post(element: ElementRef<'_>) -> bool {
    let mut in_article = false;
    for parent in element.ancestors().filter_map(ElementRef::wrap) {
        match parent.value().name() {
            "article" => in_article = true,
            "main" if in_article => return true,
            _ => {}
        }
    }
    false
}

/// The quoted post inside a post: the older page's quote card, the newer
/// page's clickable box linking to another status.
fn quoted(post: ElementRef<'_>) -> Option<ElementRef<'_>> {
    post.select(&selector(
        r#"[data-testid="card.wrapper"], div[role="link"][data-href*="/status/"]"#,
    ))
    .next()
}

fn text_of(element: ElementRef<'_>) -> String {
    element
        .text()
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// The digits after `/status/` in a path or address.
fn status_id(text: &str) -> Option<String> {
    let rest = &text[text.find("/status/")? + "/status/".len()..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    (!digits.is_empty()).then_some(digits)
}

/// The account a status path or address belongs to, as `@handle`.
fn permalink_handle(text: &str) -> Option<String> {
    let before = &text[..text.find("/status/")?];
    let user = before.rsplit('/').next()?;
    (!user.is_empty()
        && !["i", "web"].contains(&user)
        && user.len() <= 15
        && user.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'))
    .then(|| format!("@{user}"))
}

/// The status link that names a post itself, not the post it quotes.
fn own_permalink<'a>(post: ElementRef<'a>, exclude: Option<ElementRef<'_>>) -> Option<&'a str> {
    post.select(&selector(r#"a[href*="/status/"]"#))
        .filter(|link| !inside(*link, exclude))
        .filter_map(|link| link.value().attribute("href"))
        .find(|href| status_id(href).is_some())
}

/// The id of a post: its own tag, else its own status link.
fn own_id(post: ElementRef<'_>) -> Option<String> {
    post.value()
        .attribute("data-tweet-id")
        .map(str::to_owned)
        .or_else(|| own_permalink(post, quoted(post)).and_then(status_id))
}

/// The post's account: its display name and `@handle`.
fn author(
    post: ElementRef<'_>,
    exclude: Option<ElementRef<'_>>,
) -> (Option<String>, Option<String>) {
    let scope = post
        .select(&selector(r#"[data-testid="User-Name"]"#))
        .find(|element| !inside(*element, exclude))
        .unwrap_or(post);
    let mut name = None;
    let mut handle = None;
    for link in scope.select(&selector("a[href]")) {
        if inside(link, exclude) || link.select(&selector("img")).next().is_some() {
            continue;
        }
        let href = link.value().attribute("href").unwrap_or("");
        if href.contains("/status/") {
            continue;
        }
        let text = text_of(link);
        if text.is_empty() {
            continue;
        }
        if text.starts_with('@') {
            handle.get_or_insert(text);
        } else {
            name.get_or_insert(text);
        }
        if name.is_some() && handle.is_some() {
            break;
        }
    }
    // A quoted post can name its author in plain spans.
    if name.is_none() && handle.is_none() && scope != post {
        for span in scope.select(&selector("span")) {
            let text = text_of(span);
            if text.is_empty() || span.select(&selector("span")).nth(1).is_some() {
                continue;
            }
            if text.starts_with('@') {
                handle.get_or_insert(text);
            } else {
                name.get_or_insert(text);
            }
        }
    }
    (name, handle)
}

/// The element holding a post's text.
fn text_element<'a>(
    post: ElementRef<'a>,
    exclude: Option<ElementRef<'_>>,
) -> Option<ElementRef<'a>> {
    post.select(&selector(r#"[data-testid="tweetText"], div[dir="auto"]"#))
        .find(|element| !inside(*element, exclude))
}

/// A post's text as Markdown paragraphs, one per line of the post.
fn text_markdown(element: ElementRef<'_>, base: Option<&Url>) -> Result<String> {
    let tree = Html::parse_fragment(&element.html());
    let markdown = render_clean(tree.root_element(), base)?;
    Ok(markdown
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n"))
}

/// A photo's address at a larger size. X serves a photo as `name=` small,
/// medium, large or `orig`, the original bytes, and `orig` exists only in the
/// format the photo was uploaded in: a page that names `jpg` or `png` says what
/// that is, so `name=orig` is asked for. A page that asks for WebP (the server-
/// rendered page always does) does not say, and `orig` in a guessed format was
/// a 404 for a 2014 photo, so it is only raised to `large`, which every photo
/// has. Any other address is left as it is.
fn original_image(url: &str) -> String {
    let Ok(mut parsed) = Url::parse(url) else {
        return url.to_owned();
    };
    if parsed.host_str() != Some("pbs.twimg.com") || !parsed.path().starts_with("/media/") {
        return url.to_owned();
    }
    let mut pairs: Vec<(String, String)> = parsed
        .query_pairs()
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect();
    let uploaded = |extension: &str| ["jpg", "jpeg", "png"].contains(&extension);
    let known = pairs
        .iter()
        .any(|(key, value)| key == "format" && uploaded(value))
        || parsed
            .path()
            .rsplit_once('.')
            .is_some_and(|(_, extension)| uploaded(extension));
    let Some((_, size)) = pairs.iter_mut().find(|(key, _)| key == "name") else {
        return url.to_owned();
    };
    if known {
        "orig".clone_into(size);
    } else if ["thumb", "tiny", "small", "medium"].contains(&size.as_str()) {
        "large".clone_into(size);
    } else {
        return url.to_owned();
    }
    parsed.query_pairs_mut().clear().extend_pairs(pairs);
    parsed.into()
}

/// Images and videos of a post in page order, each once, as Markdown: a photo
/// at its original size, a video as its poster and, when the page carries a
/// real address (not the browser-local `blob:` handle), a link to the file.
fn media(post: ElementRef<'_>, exclude: Option<ElementRef<'_>>) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut add = |line: String| {
        if !lines.contains(&line) {
            lines.push(line);
        }
    };
    for element in post.select(&selector(
        r#"video, [data-testid="tweetPhoto"] img, a[aria-label="Image"] img"#,
    )) {
        if inside(element, exclude) {
            continue;
        }
        let https = |url: Option<&str>| {
            url.map(str::trim)
                .filter(|url| url.starts_with("https://"))
                .map(str::to_owned)
        };
        if element.value().name() == "video" {
            if let Some(poster) = https(element.value().attribute("poster")) {
                add(image_markdown(&poster));
            }
            let source = element.value().attribute("src").or_else(|| {
                element
                    .select(&selector("source[src]"))
                    .find_map(|source| source.value().attribute("src"))
            });
            if let Some(source) = https(source) {
                let label = if source.contains("/tweet_video/") {
                    "GIF"
                } else {
                    "Video"
                };
                add(format!("[{label}]({})", source.replace(' ', "%20")));
            }
        } else if let Some(photo) = https(element.value().attribute("src")) {
            add(image_markdown(&original_image(&photo)));
        }
    }
    lines
}

fn image_markdown(url: &str) -> String {
    format!(
        "![]({})",
        url.replace(' ', "%20")
            .replace('(', "%28")
            .replace(')', "%29")
    )
}

/// The date a post shows: the older page's `time[datetime]`, else the text
/// of its status link ("6:02 AM · Jul 4, 2026", "Jun 19").
fn shown_date(
    post: ElementRef<'_>,
    exclude: Option<ElementRef<'_>>,
) -> (Option<String>, Option<String>) {
    if let Some(time) = post
        .select(&selector("time[datetime]"))
        .find(|element| !inside(*element, exclude))
    {
        return (
            time.value().attribute("datetime").map(str::to_owned),
            Some(text_of(time)),
        );
    }
    let shown = post
        .select(&selector(r#"a[href*="/status/"]"#))
        .filter(|link| !inside(*link, exclude))
        .map(text_of)
        .find(|text| {
            text.chars().any(|c| c.is_ascii_digit())
                && [
                    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                    "Dec",
                ]
                .iter()
                .any(|month| text.contains(month))
        });
    (None, shown)
}

/// The UTC date a post was made on, from its id: since November 2010 an id
/// carries its creation time in milliseconds after 2010-11-04 01:42:54 UTC in
/// every bit above the lowest 22.
fn id_date(id: &str) -> Option<String> {
    let id: u64 = id.parse().ok()?;
    if id < 30_000_000_000 {
        return None;
    }
    let millis = i64::try_from((id >> 22) + 1_288_834_974_657).ok()?;
    chrono::DateTime::from_timestamp_millis(millis).map(|time| time.date_naive().to_string())
}

/// `YYYY-MM-DD`: the page's own timestamp when it has one, else the date in
/// the post's id, which does not depend on the language or time zone the page
/// was rendered in, else the shown text with a year.
fn published(iso: Option<&str>, id: Option<&str>, shown: Option<&str>) -> Option<String> {
    if let Some(iso) = iso
        && let Ok(time) = chrono::DateTime::parse_from_rfc3339(iso)
    {
        return Some(time.date_naive().to_string());
    }
    if let Some(date) = id.and_then(id_date) {
        return Some(date);
    }
    let date = shown?.rsplit('·').next()?.trim();
    chrono::NaiveDate::parse_from_str(date, "%b %d, %Y")
        .ok()
        .map(|date| date.to_string())
}

/// X's own hosts.
fn x_host(host: &str) -> bool {
    let host = host.strip_prefix("www.").unwrap_or(host);
    matches!(host, "x.com" | "twitter.com" | "mobile.twitter.com")
}

/// Third-party mirrors of X posts, which answer a client that is not a
/// browser with a redirect or an embed page instead of the post.
fn mirror_host(host: &str) -> bool {
    let host = host.strip_prefix("www.").unwrap_or(host);
    matches!(
        host,
        "fxtwitter.com" | "vxtwitter.com" | "fixupx.com" | "fixvx.com" | "twittpr.com"
    )
}

/// The canonical `https://x.com/<user>/status/<id>` (or `/article/<id>`)
/// address of a post named through X's hosts or a mirror, without tracking
/// parameters, fragment or a trailing `/photo/1` or language segment. `None`
/// for any other address and for one that already is canonical, so a caller
/// can fetch the post itself rather than a mirror's redirect page.
pub(crate) fn canonical_status_url(url: &Url) -> Option<Url> {
    if !url.username().is_empty() || url.password().is_some() || url.port().is_some() {
        return None;
    }
    let host = url.host_str()?;
    if !x_host(host) && !mirror_host(host) {
        return None;
    }
    let segments: Vec<&str> = url
        .path_segments()?
        .filter(|segment| !segment.is_empty())
        .collect();
    let (user, kind, id) = match segments.as_slice() {
        [user, kind @ ("status" | "article"), id, ..] => (*user, *kind, *id),
        ["i", "web", "status", id, ..] => ("i", "status", *id),
        _ => return None,
    };
    if user.is_empty()
        || user.len() > 15
        || !user.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        || id.is_empty()
        || !id.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let canonical = Url::parse(&format!("https://x.com/{user}/{kind}/{id}")).ok()?;
    (canonical != *url).then_some(canonical)
}

/// Whether a page is an X post page: by its host (a status or Article
/// address), else by X's own column markup or its media host.
fn is_x_page(page: &Landmarks<'_>, base: Option<&Url>) -> bool {
    if let Some(host) = base.and_then(Url::host_str) {
        return x_host(host)
            && base.is_some_and(|url| {
                url.path().contains("/status/") || url.path().contains("/article/")
            });
    }
    // X's own column and post markup, or its media host.
    if page.column_post {
        return true;
    }
    page.media.iter().any(|element| {
        element
            .value()
            .attribute("src")
            .or_else(|| element.value().attribute("poster"))
            .and_then(|source| Url::parse(source).ok())
            .is_some_and(|url| url.host_str() == Some("pbs.twimg.com"))
    })
}

/// A post as the page shows it: its element, the post it quotes and its
/// account.
struct Post<'a> {
    element: ElementRef<'a>,
    quote: Option<ElementRef<'a>>,
    name: Option<String>,
    handle: Option<String>,
}

impl<'a> Post<'a> {
    fn read(element: ElementRef<'a>) -> Self {
        let quote = quoted(element);
        let (name, handle) = author(element, quote);
        // The status link names the account without the page's decoration.
        let handle = own_permalink(element, quote)
            .and_then(permalink_handle)
            .or(handle);
        Self {
            element,
            quote,
            name,
            handle,
        }
    }

    fn same_author(&self, other: &Post<'_>) -> bool {
        match (&self.handle, &other.handle) {
            (Some(a), Some(b)) => a.eq_ignore_ascii_case(b),
            _ => self.name.is_some() && self.name == other.name,
        }
    }

    /// The post's text, media and quoted post as Markdown blocks.
    fn blocks(&self, base: Option<&Url>) -> Result<Vec<String>> {
        let mut blocks = Vec::new();
        if let Some(text) = text_element(self.element, self.quote) {
            let text = text_markdown(text, base)?;
            if !text.is_empty() {
                blocks.push(text);
            }
        }
        blocks.extend(media(self.element, self.quote));
        if let Some(quote) = self.quote
            && let Some(block) = quote_block(quote, base)?
        {
            blocks.push(block);
        }
        Ok(blocks)
    }
}

/// The quoted post as a block quote: `**Name @handle** · date`, its text and
/// media, and a link to it.
fn quote_block(quote: ElementRef<'_>, base: Option<&Url>) -> Result<Option<String>> {
    let (name, handle) = author(quote, None);
    let handle = handle.or_else(|| {
        quote
            .value()
            .attribute("data-href")
            .and_then(permalink_handle)
    });
    // `YYYY-MM-DD` when the quote's timestamp or id says it, which does not
    // change with the reader's time zone; else the text the page shows.
    let (iso, shown) = shown_date(quote, None);
    let id = quote
        .value()
        .attribute("data-href")
        .and_then(status_id)
        .or_else(|| own_id(quote));
    let date = published(iso.as_deref(), id.as_deref(), None).or(shown);
    let mut lines = Vec::new();
    let who = match (&name, &handle) {
        (Some(name), Some(handle)) => format!("{name} {handle}"),
        (Some(one), None) | (None, Some(one)) => one.clone(),
        (None, None) => String::new(),
    };
    let mut header = String::new();
    if !who.is_empty() {
        header.push_str("**");
        header.push_str(&who.replace('*', "\\*"));
        header.push_str("**");
    }
    if let Some(date) = date {
        if !header.is_empty() {
            header.push_str(" · ");
        }
        header.push_str(&date);
    }
    if !header.is_empty() {
        lines.push(header);
    }
    if let Some(text) = text_element(quote, None) {
        let text = text_markdown(text, base)?;
        if !text.is_empty() {
            lines.push(text);
        }
    }
    lines.extend(media(quote, None));
    if let Some(href) = quote.value().attribute("data-href")
        && let Some(url) = Url::parse("https://x.com")
            .ok()
            .and_then(|x| x.join(href).ok())
    {
        lines.push(format!("[{url}]({url})"));
    }
    if lines.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        lines
            .join("\n\n")
            .lines()
            .map(|line| {
                if line.is_empty() {
                    ">".to_owned()
                } else {
                    format!("> {line}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    ))
}

fn reply_slot(article: &ElementRef<'_>) -> bool {
    article
        .ancestors()
        .filter_map(ElementRef::wrap)
        .any(|parent| {
            parent.value().name() == "section"
                || parent.value().attribute("data-testid") == Some("card.wrapper")
                || (parent.value().attribute("role") == Some("link")
                    && parent
                        .value()
                        .attribute("data-href")
                        .is_some_and(|href| href.contains("/status/")))
        })
}

/// An embedded long-form Article, scoped by its own body and the requested
/// post id. The page can repeat it in responsive layouts or surround it with
/// replies containing other Articles. Its author chrome is outside the body.
fn long_article(document: &Html, base: Option<&Url>) -> Result<Option<Announcement>> {
    let wanted = base.and_then(|url| {
        let parts: Vec<_> = url
            .path_segments()?
            .filter(|part| !part.is_empty())
            .collect();
        let id = match parts.as_slice() {
            [_, "status" | "article", id, ..] | ["i", "web", "status", id, ..] => *id,
            _ => return None,
        };
        (!id.is_empty() && id.bytes().all(|byte| byte.is_ascii_digit())).then(|| id.to_owned())
    });
    let candidates: Vec<_> = document
        .select(&selector(".x-article-body"))
        .filter_map(|body| {
            if body
                .ancestors()
                .filter_map(ElementRef::wrap)
                .any(|node| matches!(node.value().name(), "template" | "noscript" | "head"))
            {
                return None;
            }
            let container = body.parent().and_then(ElementRef::wrap)?;
            let heading = container
                .child_elements()
                .find(|child| child.value().name() == "h1" && !text_of(*child).is_empty())?;
            let mut owners = body
                .ancestors()
                .filter_map(ElementRef::wrap)
                .filter(|node| node.value().name() == "article");
            let owner = owners.next();
            if owners.next().is_some() || owner.is_some_and(|owner| inside(body, quoted(owner))) {
                return None;
            }
            // Links cited by the author do not identify the enclosing post.
            let id = owner.and_then(|owner| {
                owner
                    .value()
                    .attribute("data-tweet-id")
                    .map(str::to_owned)
                    .or_else(|| own_permalink(owner, Some(body)).and_then(status_id))
            });
            Some((body, container, heading, owner, id))
        })
        .collect();
    let selected = if let Some(wanted) = &wanted {
        candidates
            .iter()
            .find(|candidate| candidate.4.as_ref() == Some(wanted))
            .or_else(|| {
                // A standalone Article has no enclosing post. Only accept an
                // unambiguous document; an unrelated reply is never a fallback.
                let mut standalone = candidates.iter().filter(|candidate| candidate.3.is_none());
                let first = standalone.next()?;
                standalone.next().is_none().then_some(first)
            })
    } else {
        // Saved pages have no address to identify the post. Match the same
        // main timeline post as the ordinary reader, never the first Article
        // found somewhere in a reply or recommendation.
        let column = document
            .select(&selector(r#"[data-testid="primaryColumn"], main"#))
            .next();
        let scope = column.unwrap_or_else(|| document.root_element());
        let main = scope.select(&selector("article")).find(|article| {
            is_post(*article, column.is_some())
                && !reply_slot(article)
                && !article
                    .ancestors()
                    .filter_map(ElementRef::wrap)
                    .any(|parent| parent.value().name() == "article")
        });
        candidates.iter().find(|candidate| {
            main.is_some() && candidate.3.map(|owner| owner.id()) == main.map(|post| post.id())
        })
    };
    let Some((body, container, heading, owner, id)) = selected else {
        return Ok(None);
    };
    // Preserve the document's block structure; post text_markdown deliberately
    // separates every line and would break nested lists and code in an Article.
    let body_markdown = render_clean(*body, base)?;
    if body_markdown.is_empty() {
        return Ok(None);
    }
    let mut blocks = vec![render_clean(*heading, base)?];
    for cover in container
        .child_elements()
        .filter(|child| child.value().name() == "img")
    {
        let markdown = render_clean(cover, base)?;
        if !markdown.is_empty() {
            blocks.push(markdown);
        }
    }
    blocks.push(body_markdown);
    let mut metadata = Map::new();
    metadata.insert("title".into(), text_of(*heading).into());
    metadata.insert("site".into(), "X (Twitter)".into());
    if let Some(paragraph) = body
        .select(&selector("p"))
        .map(text_of)
        .find(|text| !text.is_empty())
    {
        metadata.insert(
            "description".into(),
            paragraph
                .chars()
                .take(200)
                .collect::<String>()
                .trim_end()
                .into(),
        );
    }
    if let Some(owner) = owner {
        let (name, handle) = author(*owner, Some(*body));
        let handle = own_permalink(*owner, Some(*body))
            .and_then(permalink_handle)
            .or(handle);
        if let Some(who) = handle.as_ref().or(name.as_ref()) {
            metadata.insert("author".into(), who.clone().into());
        }
        if let (Some(name), Some(_)) = (name, handle) {
            metadata.insert("author_name".into(), name.into());
        }
        let (iso, shown) = shown_date(*owner, Some(*body));
        if let Some(date) = published(iso.as_deref(), id.as_deref(), shown.as_deref()) {
            metadata.insert("published".into(), date.into());
        }
    }
    // This is a document, not social_post: keep ordinary LLM enhancement available.
    Ok(Some(Announcement {
        markdown: blocks.join("\n\n"),
        metadata,
    }))
}

/// The post of an X status page as Markdown with its metadata: the post, then
/// the posts the author continued it with, each after a rule. Replies from
/// other accounts, and the author's own replies after the first of them, are
/// the discussion around the post and are left out.
pub(super) fn post(
    document: &Html,
    page: &Landmarks<'_>,
    base: Option<&Url>,
) -> Result<Option<Announcement>> {
    if !is_x_page(page, base) {
        return Ok(None);
    }
    if let Some(article) = long_article(document, base)? {
        return Ok(Some(article));
    }
    // Older Article pages without a dedicated body remain ordinary documents;
    // the surrounding `article` elements are cards of other Articles. Only
    // `/<handle>/article/<id>` names one; an account called `article` does not.
    if base.is_some_and(|url| {
        url.path_segments()
            .is_some_and(|mut parts| parts.nth(1) == Some("article"))
    }) {
        return Ok(None);
    }
    let column = document
        .select(&selector(r#"[data-testid="primaryColumn"], main"#))
        .next();
    let scope = column.unwrap_or_else(|| document.root_element());
    let status = base.and_then(|url| status_id(url.path()));
    // Posts of the page in order; a quoted post is inside another one.
    let timeline: Vec<ElementRef<'_>> = scope
        .select(&selector("article"))
        .filter(|article| {
            is_post(*article, status.is_some() || column.is_some())
                && !article
                    .ancestors()
                    .filter_map(ElementRef::wrap)
                    .any(|parent| parent.value().name() == "article")
        })
        .collect();
    // The main post is the one the address names, else the first that is not
    // a reply (replies sit in a `section` or in a clickable box of their own).
    let Some(position) = status
        .as_deref()
        .and_then(|id| {
            timeline
                .iter()
                .position(|article| own_id(*article).as_deref() == Some(id))
        })
        .or_else(|| timeline.iter().position(|article| !reply_slot(article)))
    else {
        return Ok(None);
    };
    let main = Post::read(timeline[position]);
    let lead = main.blocks(base)?;
    if lead.is_empty() {
        return Ok(None);
    }
    let main_id = own_id(main.element);
    let mut sections = vec![lead.join("\n\n")];
    // Consecutive posts of the author right after the main one continue it;
    // the first post of anyone else ends the thread. A page can repeat the
    // main post, which is not a continuation of itself.
    let mut seen = vec![main_id.clone()];
    for article in &timeline[position + 1..] {
        let id = own_id(*article);
        if id.is_some() && seen.contains(&id) {
            continue;
        }
        let reply = Post::read(*article);
        if !reply.same_author(&main) {
            break;
        }
        seen.push(id);
        let blocks = reply.blocks(base)?;
        if !blocks.is_empty() {
            sections.push(blocks.join("\n\n"));
        }
    }
    let mut metadata = Map::new();
    let who = main.handle.clone().or_else(|| main.name.clone());
    if let Some(who) = &who {
        metadata.insert("title".into(), format!("Post by {who} on X").into());
        metadata.insert("author".into(), who.clone().into());
    }
    if let (Some(name), Some(_)) = (&main.name, &main.handle) {
        metadata.insert("author_name".into(), name.clone().into());
    }
    metadata.insert("site".into(), "X (Twitter)".into());
    let (iso, shown) = shown_date(main.element, main.quote);
    if let Some(date) = published(iso.as_deref(), main_id.as_deref(), shown.as_deref()) {
        metadata.insert("published".into(), date.into());
    }
    let plain = text_element(main.element, main.quote)
        .map(text_of)
        .unwrap_or_default();
    if !plain.is_empty() {
        let mut description: String = plain.chars().take(200).collect();
        description.truncate(description.trim_end().len());
        metadata.insert("description".into(), Value::String(description));
    }
    // The body is already the post; a language model keeps it as it is.
    metadata.insert("content_profile".into(), "social_post".into());
    Ok(Some(Announcement {
        markdown: sections.join("\n\n---\n\n"),
        metadata,
    }))
}

#[cfg(test)]
mod tests {
    use super::super::extract_html;
    use super::{canonical_status_url, id_date, original_image, published};
    use url::Url;

    /// A post of the newest server-rendered page: no test ids, the text in a
    /// `div[dir=auto]`, the date in the status link, counters in
    /// `[data-engagement-action]`.
    fn article(handle: &str, name: &str, id: &str, date: &str, text: &str, extra: &str) -> String {
        let text = if text.is_empty() {
            String::new()
        } else {
            format!(r#"<div dir="auto">{text}</div>"#)
        };
        format!(
            r#"<article><div><a href="/{handle}"><div><img alt="@{handle}" src="https://pbs.twimg.com/profile_images/1/{handle}_normal.jpg"></div></a></div>
            <div><span><a href="/{handle}"><div>{name}</div></a><div><span role="button" aria-label="Verified account"></span></div></span>
            <span><a href="/{handle}"><span>@{handle}</span></a></span></div>
            {text}{extra}
            <div><span><a href="/{handle}/status/{id}">{date}</a></span><span aria-hidden="true">·</span><a href="/{handle}/status/{id}"><div>46.3M</div><div>Views</div></a></div>
            <div><div data-engagement-action="reply"><a aria-label="Reply" href="/{handle}/status/{id}"><span>5.6K</span></a></div>
            <div data-engagement-action="like"><button aria-label="Like"><span>366K</span></button></div>
            <span data-engagement-action="bookmark"><button aria-label="Bookmark"></button></span></div></article>"#
        )
    }

    /// A reply card: the same post in a clickable box that links to it.
    fn card(handle: &str, name: &str, id: &str, text: &str) -> String {
        format!(
            r#"<li><div role="link" data-href="/{handle}/status/{id}">{}</div></li>"#,
            article(handle, name, id, "Feb 25, 2025", text, "")
        )
    }

    fn page(main: &str, replies: &str) -> String {
        format!(
            r#"<html><head><title>Ada on X</title></head><body><main>
            <h1 class="sr-only">Ada on X: "Hello world"</h1><div>{main}</div><div><ul>{replies}</ul></div></main></body></html>"#
        )
    }

    const ID: &str = "1894221971766084049";
    const STATUS: &str = "https://x.com/ada/status/1894221971766084049";

    #[test]
    fn a_2026_post_keeps_text_media_and_its_quote_without_page_controls() {
        let page = r#"<html><body><main><article data-tweet-id="1"><div>
            <a href="https://x.com/ada"><img alt="user avatar" src="https://pbs.twimg.com/profile_images/1/a_normal.jpeg" width="40" height="40"></a>
            <a href="https://x.com/ada">Ada</a><a href="https://x.com/ada">@ada</a></div>
            <div dir="auto"><span>First line of the post.
Second line with </span><a href="https://example.com/tool">example.com/tool</a></div>
            <div><video poster="https://pbs.twimg.com/amplify_video_thumb/9/img/p.jpg" src="blob:https://x.com/abc"></video><div>00:00</div></div>
            <div role="link" data-href="/grace/status/2"><a href="https://x.com/grace">Grace</a><a href="https://x.com/grace">@grace</a><a href="https://x.com/grace/status/2">Jun 19</a>
              <div dir="auto"><span>The quoted post.</span><button>Show more</button></div>
              <a aria-label="Image" href="https://x.com/grace/status/2/photo/1"><img src="https://pbs.twimg.com/media/Q?format=jpg&amp;name=medium"></a></div>
            <a href="https://x.com/ada/status/1">6:02 AM · Jul 4, 2026</a><a href="https://x.com/ada/status/1">32KViews</a>
            </article><section><article data-tweet-id="3"><div dir="auto">A reply.</div></article></section></main></body></html>"#;
        let doc = extract_html(page, Some("https://x.com/ada/status/1")).unwrap();
        assert_eq!(
            doc.markdown,
            "First line of the post.\n\nSecond line with [example.com/tool](https://example.com/tool)\n\n![](https://pbs.twimg.com/amplify_video_thumb/9/img/p.jpg)\n\n> **Grace @grace** · Jun 19\n>\n> The quoted post.\n>\n> ![](https://pbs.twimg.com/media/Q?format=jpg&name=orig)\n>\n> [https://x.com/grace/status/2](https://x.com/grace/status/2)"
        );
        assert_eq!(doc.metadata["title"], "Post by @ada on X");
        assert_eq!(doc.metadata["author"], "@ada");
        assert_eq!(doc.metadata["published"], "2026-07-04");
    }

    #[test]
    fn an_older_post_is_read_by_its_test_ids_and_other_pages_are_not_posts() {
        let page = r#"<html><body><div data-testid="primaryColumn"><article data-testid="tweet"><div data-testid="User-Name"><a href="/ada"><span>Ada</span></a><a href="/ada"><span>@ada</span></a></div>
            <div data-testid="tweetText"><span>The post text.</span></div><div data-testid="tweet-stats"><span>1.2K replies</span></div>
            <time datetime="2025-03-15T08:30:00.000Z">8:30 AM · Mar 15, 2025</time></article>
            <div aria-label="Discover more"><h2>Discover more</h2></div></div></body></html>"#;
        let doc = extract_html(page, None).unwrap();
        assert_eq!(doc.markdown, "The post text.");
        assert_eq!(doc.metadata["published"], "2025-03-15");
        // An article about posts on another site is an ordinary page.
        let elsewhere = page.replace("primaryColumn", "column");
        let doc = extract_html(&elsewhere, Some("https://example.org/post")).unwrap();
        assert!(doc.markdown.contains("1.2K replies"), "{}", doc.markdown);
    }

    #[test]
    fn the_server_rendered_post_is_its_text_once_with_account_date_and_no_counters() {
        let main = article(
            "ada",
            "Ada Lovelace",
            ID,
            "3:04 AM · Feb 25, 2025",
            "Hello world",
            "",
        );
        let doc = extract_html(&page(&main, ""), Some(STATUS)).unwrap();
        assert_eq!(doc.markdown, "Hello world");
        for noise in ["Views", "46.3M", "366K", "Verified", "avatar"] {
            assert!(!doc.markdown.contains(noise), "{noise}: {}", doc.markdown);
        }
        assert_eq!(doc.metadata["title"], "Post by @ada on X");
        assert_eq!(doc.metadata["author"], "@ada");
        assert_eq!(doc.metadata["author_name"], "Ada Lovelace");
        assert_eq!(doc.metadata["site"], "X (Twitter)");
        assert_eq!(doc.metadata["published"], "2025-02-25");
        assert_eq!(doc.metadata["description"], "Hello world");
        assert_eq!(doc.metadata["content_profile"], "social_post");
        // A page saved from the browser has no address: its markup is enough.
        let saved = extract_html(&page(&main, ""), None).unwrap();
        assert_eq!(saved.markdown, "Hello world");
        assert_eq!(saved.metadata["author"], "@ada");
    }

    #[test]
    fn photos_keep_their_original_size_and_a_post_without_text_is_its_media() {
        let photos = r#"<div><div inert=""><a aria-label="Image" href="/ada/status/1894221971766084049/photo/1"><img alt="" src="https://pbs.twimg.com/media/AAA?format=webp&amp;name=large"></a></div>
            <a aria-label="Image" href="/ada/status/1894221971766084049/photo/2"><img alt="" src="https://pbs.twimg.com/media/BBB?format=png&amp;name=small"></a>
            <a aria-label="Image" href="/ada/status/1894221971766084049/photo/1"><img alt="" src="https://pbs.twimg.com/media/AAA?format=webp&amp;name=medium"></a></div>"#;
        let with_text = article("ada", "Ada", ID, "Feb 25, 2025", "Two photos", photos);
        let doc = extract_html(&page(&with_text, ""), Some(STATUS)).unwrap();
        assert_eq!(
            doc.markdown,
            "Two photos\n\n![](https://pbs.twimg.com/media/AAA?format=webp&name=large)\n\n![](https://pbs.twimg.com/media/BBB?format=png&name=orig)"
        );
        let without_text = article("ada", "Ada", ID, "Feb 25, 2025", "", photos);
        let doc = extract_html(&page(&without_text, ""), Some(STATUS)).unwrap();
        assert!(
            doc.markdown
                .starts_with("![](https://pbs.twimg.com/media/AAA"),
            "{}",
            doc.markdown
        );
        assert!(doc.metadata.get("description").is_none());
        assert_eq!(doc.metadata["title"], "Post by @ada on X");
    }

    #[test]
    fn a_video_is_its_poster_once_and_a_link_only_when_the_page_has_a_file() {
        let video = r#"<div><video poster="https://pbs.twimg.com/amplify_video_thumb/9/img/p?format=webp&amp;name=medium" src="blob:https://x.com/abc"></video>
            <div><img alt="" src="https://pbs.twimg.com/amplify_video_thumb/9/img/p?format=webp&amp;name=medium"><button aria-label="Play"></button></div><div>00:00</div></div>
            <video poster="https://pbs.twimg.com/tweet_video_thumb/g.jpg" src="https://video.twimg.com/tweet_video/g.mp4"></video>"#;
        let main = article("ada", "Ada", ID, "Feb 25, 2025", "Clips", video);
        let doc = extract_html(&page(&main, ""), Some(STATUS)).unwrap();
        assert_eq!(
            doc.markdown,
            "Clips\n\n![](https://pbs.twimg.com/amplify_video_thumb/9/img/p?format=webp&name=medium)\n\n![](https://pbs.twimg.com/tweet_video_thumb/g.jpg)\n\n[GIF](https://video.twimg.com/tweet_video/g.mp4)"
        );
        assert!(!doc.markdown.contains("00:00"));
    }

    #[test]
    fn a_quoted_post_is_a_block_quote_and_not_a_second_post() {
        let quote = r#"<div role="link" data-href="/grace/status/1894160928502858201"><article>
            <div><a href="/grace"><div><img alt="@grace" src="https://pbs.twimg.com/profile_images/2/g_normal.jpg"></div></a></div>
            <div><span><a href="/grace"><div>Grace Hopper</div></a></span><span><a href="/grace"><span>@grace</span></a></span>
            <span><a href="/grace/status/1894160928502858201">Feb 25, 2025</a></span></div>
            <div dir="auto">A quoted line.</div>
            <div><video poster="https://pbs.twimg.com/amplify_video_thumb/7/img/q?format=webp&amp;name=medium"></video><div>00:00</div></div>
            </article></div>"#;
        let main = article(
            "ada",
            "Ada",
            ID,
            "3:04 AM · Feb 25, 2025",
            "Good luck!",
            quote,
        );
        let doc = extract_html(&page(&main, ""), Some(STATUS)).unwrap();
        assert_eq!(
            doc.markdown,
            "Good luck!\n\n> **Grace Hopper @grace** · 2025-02-24\n>\n> A quoted line.\n>\n> ![](https://pbs.twimg.com/amplify_video_thumb/7/img/q?format=webp&name=medium)\n>\n> [https://x.com/grace/status/1894160928502858201](https://x.com/grace/status/1894160928502858201)"
        );
        assert_eq!(doc.metadata["author"], "@ada");
        assert_eq!(doc.metadata["published"], "2025-02-25");
        assert_eq!(doc.markdown.matches("Good luck!").count(), 1);
    }

    #[test]
    fn the_authors_own_replies_continue_the_post_and_other_accounts_do_not() {
        let main = article("ada", "Ada", ID, "3:04 AM · Feb 25, 2025", "Part one", "");
        let replies = format!(
            "{}{}{}{}{}",
            card("ada", "Ada", "1894221971766084050", "Part two"),
            card("ada", "Ada", "1894221971766084051", "Part three"),
            card("grace", "Grace", "1894221971766084052", "A comment"),
            card("ada", "Ada", "1894221971766084053", "Thanks Grace"),
            card("linus", "Linus", "1894221971766084054", "Another comment"),
        );
        // The thread's earlier post comes first, as X lists it, and is not
        // part of this post.
        let earlier = card("ada", "Ada", "1894221971766083999", "Earlier");
        let html = page(&main, &replies).replace("<main>", &format!("<main><ul>{earlier}</ul>"));
        let doc = extract_html(&html, Some(STATUS)).unwrap();
        assert_eq!(
            doc.markdown,
            "Part one\n\n---\n\nPart two\n\n---\n\nPart three"
        );
        assert_eq!(doc.metadata["description"], "Part one");
        assert_eq!(doc.metadata["published"], "2025-02-25");
    }

    #[test]
    fn an_older_page_continues_with_the_authors_replies_in_its_section() {
        let post = |name: &str, handle: &str, text: &str| {
            format!(
                r#"<article data-testid="tweet"><div data-testid="User-Name"><a href="/{handle}"><span>{name}</span></a><a href="/{handle}"><span>@{handle}</span></a></div>
                <div data-testid="tweetText"><span>{text}</span></div></article>"#
            )
        };
        let html = format!(
            r#"<html><body><div data-testid="primaryColumn">{}<section>{}{}{}</section></div></body></html>"#,
            post("Ada", "ada", "Opening"),
            post("Ada", "ada", "Continued"),
            post("Grace", "grace", "Question"),
            post("Ada", "ada", "Answer")
        );
        let doc = extract_html(&html, Some("https://twitter.com/ada/status/7")).unwrap();
        assert_eq!(doc.markdown, "Opening\n\n---\n\nContinued");
        assert_eq!(doc.metadata["content_profile"], "social_post");
    }

    #[test]
    fn a_date_in_another_language_still_comes_from_the_post_id() {
        let main = article("ada", "Ada", ID, "上午3:04 · 2025年2月25日", "你好", "");
        let doc = extract_html(&page(&main, ""), Some(STATUS)).unwrap();
        assert_eq!(doc.metadata["published"], "2025-02-25");
        assert_eq!(
            id_date("1894221971766084049").as_deref(),
            Some("2025-02-25")
        );
        // Ids from before November 2010 carry no time.
        assert_eq!(id_date("1234"), None);
        assert_eq!(
            published(None, None, Some("3:04 AM · Feb 25, 2025")).as_deref(),
            Some("2025-02-25")
        );
        assert_eq!(published(None, None, Some("Feb 25")), None);
    }

    #[test]
    fn an_article_page_is_still_read_as_a_document() {
        // Cards of other Articles are plain `article` elements with counters.
        let cards = article("ivan", "Ivan", "1", "Dec 22, 2025", "Another Article", "").repeat(2);
        let body = "The ultimate guide to Articles. Long-form writing gives your ideas room to breathe and truly shine, with headings, lists and images for readers. ".repeat(6);
        let html = format!(
            r#"<html><head><title>Guide</title></head><body><div><h1 dir="auto">The ultimate guide</h1><div dir="auto"><p>{body}</p></div></div>
            <ul>{cards}</ul></body></html>"#
        );
        // The address alone keeps it a document even if the page gains a `main`.
        let in_main = html
            .replace("<body>", "<body><main>")
            .replace("</body>", "</main></body>");
        let article = Some("https://x.com/creators/article/2011957172821737574");
        for (page, base) in [(&html, article), (&html, None), (&in_main, article)] {
            let doc = extract_html(page, base).unwrap();
            assert!(
                doc.markdown.contains("The ultimate guide")
                    && doc.markdown.contains("room to breathe"),
                "{base:?}: {}",
                doc.markdown
            );
            assert!(doc.metadata.get("content_profile").is_none(), "{base:?}");
            assert_ne!(
                doc.metadata.get("author").and_then(|v| v.as_str()),
                Some("@ivan")
            );
        }
    }

    #[test]
    fn an_embedded_article_keeps_document_blocks_and_excludes_the_surrounding_thread() {
        // An authored reduction of the server-rendered layout: the post's short
        // text is empty, its Article is inside a timeline item, and the page
        // repeats the main post for another responsive layout.
        let content = format!(
            r#"<div dir="auto"></div><div><div>
            <img alt="Article cover image" src="https://pbs.twimg.com/media/cover.jpg">
            <h1 dir="auto">A practical guide</h1>
            <div><button data-engagement-action="reply">999</button></div>
            <div class="x-article-body break-words">
              <p>Opening <strong>idea</strong>.</p>
              <h2>First section</h2>
              <ol><li>First step<ul><li>Nested detail</li></ul></li><li>Second step</li></ol>
              <blockquote><p>A useful quotation.</p></blockquote>
              <pre><code>first line\n  second line</code></pre>
              <p><a href="/grace/status/77">A cited post</a></p>
              <a href="/ada/article/{ID}/media/7"><img src="https://pbs.twimg.com/media/diagram.jpg" alt="Diagram"></a>
              <p>Closing paragraph.</p>
            </div></div></div>"#
        )
        .replace("first line\\n", "first line\n");
        let main = article("ada", "Ada", ID, "Feb 25, 2025", "", &content);
        let other = article("grace", "Grace", "77", "Feb 25, 2025", "REPLY_ONLY", "");
        let followup = article("ada", "Ada", "88", "Feb 25, 2025", "FOLLOWUP_ONLY", "");
        let html = format!(
            r#"<html><head><title>Ada on X: https://t.co/example / X</title>
            <meta name="description" content="https://t.co/example"></head>
            <body><main>{main}</main><ul><li>{other}</li><li>{main}</li><li>{followup}</li></ul></body></html>"#
        );
        let saved = extract_html(&html, None).unwrap();
        assert_eq!(saved.metadata["title"], "A practical guide");
        assert_eq!(saved.markdown.matches("Opening **idea**.").count(), 1);
        assert!(!saved.markdown.contains("REPLY_ONLY"));
        for base in [STATUS.to_owned(), format!("https://x.com/ada/article/{ID}")] {
            let doc = extract_html(&html, Some(&base)).unwrap();
            assert!(
                doc.markdown.starts_with("# A practical guide\n"),
                "{}",
                doc.markdown
            );
            assert_eq!(doc.markdown.matches("Opening **idea**.").count(), 1);
            assert_eq!(doc.markdown.matches("cover.jpg").count(), 1);
            assert_eq!(doc.markdown.matches("diagram.jpg").count(), 1);
            assert!(
                doc.markdown.contains("## First section"),
                "{}",
                doc.markdown
            );
            assert!(doc.markdown.contains("1. First step"), "{}", doc.markdown);
            assert!(doc.markdown.contains("Nested detail"), "{}", doc.markdown);
            assert!(
                doc.markdown.contains("> A useful quotation."),
                "{}",
                doc.markdown
            );
            assert!(
                doc.markdown.contains("first line\n  second line"),
                "{}",
                doc.markdown
            );
            assert!(
                doc.markdown
                    .contains("[A cited post](https://x.com/grace/status/77)")
            );
            assert!(
                doc.markdown.ends_with("Closing paragraph."),
                "{}",
                doc.markdown
            );
            for noise in [
                "REPLY_ONLY",
                "FOLLOWUP_ONLY",
                "profile_images",
                "t.co/example",
                "999",
            ] {
                assert!(!doc.markdown.contains(noise), "{noise}: {}", doc.markdown);
            }
            assert_eq!(doc.metadata["title"], "A practical guide");
            assert_eq!(doc.metadata["author"], "@ada");
            assert_eq!(doc.metadata["author_name"], "Ada");
            assert_eq!(doc.metadata["description"], "Opening idea.");
            assert_eq!(doc.metadata["published"], "2025-02-25");
            assert!(doc.metadata.get("content_profile").is_none());
        }
    }

    #[test]
    fn an_unrelated_or_quoted_article_does_not_replace_the_requested_post() {
        let content = r#"<div><h1>OTHER_ARTICLE</h1><div class="x-article-body"><p>OTHER_BODY</p></div></div>"#;
        let other = article("grace", "Grace", "77", "Feb 25, 2025", "Reply", content);
        let main = article("ada", "Ada", ID, "Feb 25, 2025", "The actual post.", "");
        let html = page(&main, &other);
        for base in [
            Some(STATUS),
            None,
            Some("https://x.com/article/status/1894221971766084049"),
        ] {
            let doc = extract_html(&html, base).unwrap();
            assert_eq!(doc.markdown, "The actual post.");
            assert_eq!(doc.metadata["content_profile"], "social_post");
        }
        assert!(
            super::long_article(
                &scraper::Html::parse_document(&html),
                Some(&Url::parse(STATUS).unwrap())
            )
            .unwrap()
            .is_none()
        );

        let doc = extract_html(&html, Some("https://x.com/article/status/77")).unwrap();
        assert_eq!(doc.metadata["title"], "OTHER_ARTICLE");
        assert!(doc.markdown.contains(r"OTHER\_BODY"), "{}", doc.markdown);

        // Even a quote with the requested id cannot become the outer Article.
        let quoted = article("ada", "Ada", ID, "Feb 25, 2025", "", content);
        let wrapper = article("grace", "Grace", "77", "Feb 25, 2025", "Comment", &quoted);
        assert!(
            super::long_article(
                &scraper::Html::parse_document(&wrapper),
                Some(&Url::parse(STATUS).unwrap())
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn a_standalone_article_uses_its_heading_only_on_x() {
        let html = r#"<html><head><title>Original page title</title></head><body>
            <div><h1>Actual article title</h1>
            <div class="x-article-body"><p>Standalone body.</p></div></div></body></html>"#;
        let doc = extract_html(html, Some("https://x.com/ada/article/1234")).unwrap();
        assert_eq!(doc.metadata["title"], "Actual article title");
        assert_eq!(doc.markdown, "# Actual article title\n\nStandalone body.");
        assert!(doc.metadata.get("content_profile").is_none());
        let other = extract_html(html, Some("https://example.org/article/1234")).unwrap();
        assert_eq!(other.metadata["title"], "Original page title");
        assert!(other.metadata.get("author").is_none());
    }

    #[test]
    fn x_posts_named_through_mirrors_or_tracking_have_one_canonical_address() {
        let canonical = |text: &str| {
            canonical_status_url(&Url::parse(text).unwrap()).map(|url| url.to_string())
        };
        let nasa = Some("https://x.com/NASA/status/525397368116895744");
        for (given, expected) in [
            ("https://twitter.com/NASA/status/525397368116895744", nasa),
            (
                "https://mobile.twitter.com/NASA/status/525397368116895744?s=20&t=abc#frag",
                nasa,
            ),
            (
                "https://www.twitter.com/NASA/status/525397368116895744/photo/1",
                nasa,
            ),
            ("http://x.com/NASA/status/525397368116895744", nasa),
            (
                "https://fxtwitter.com/NASA/status/525397368116895744/en",
                nasa,
            ),
            ("https://vxtwitter.com/NASA/status/525397368116895744", nasa),
            ("https://fixupx.com/NASA/status/525397368116895744", nasa),
            ("https://fixvx.com/NASA/status/525397368116895744", nasa),
            ("https://twittpr.com/NASA/status/525397368116895744", nasa),
            (
                "https://twitter.com/i/web/status/525397368116895744",
                Some("https://x.com/i/status/525397368116895744"),
            ),
            (
                "https://fxtwitter.com/XCreators/article/2011957172821737574",
                Some("https://x.com/XCreators/article/2011957172821737574"),
            ),
            // Already canonical, not a post, or not X.
            ("https://x.com/NASA/status/525397368116895744", None),
            ("https://x.com/NASA", None),
            ("https://x.com/NASA/status/not-a-number", None),
            ("https://x.com/search?q=status", None),
            ("https://example.com/NASA/status/525397368116895744", None),
            (
                "https://fxtwitter.com.example.org/NASA/status/525397368116895744",
                None,
            ),
            (
                "https://user:secret@twitter.com/NASA/status/525397368116895744",
                None,
            ),
        ] {
            assert_eq!(canonical(given).as_deref(), expected, "{given}");
        }
    }

    #[test]
    fn photos_get_the_original_only_where_the_page_names_its_format() {
        // The page names the format the photo was uploaded in: ask for `orig`.
        assert_eq!(
            original_image("https://pbs.twimg.com/media/A?format=png&name=small"),
            "https://pbs.twimg.com/media/A?format=png&name=orig"
        );
        assert_eq!(
            original_image("https://pbs.twimg.com/media/A?format=jpg&name=medium&x=1"),
            "https://pbs.twimg.com/media/A?format=jpg&name=orig&x=1"
        );
        assert_eq!(
            original_image("https://pbs.twimg.com/media/A.jpg?name=medium"),
            "https://pbs.twimg.com/media/A.jpg?name=orig"
        );
        // A WebP request does not say it, and `orig` need not exist in that
        // format: the photo only gets as large as every photo can be.
        assert_eq!(
            original_image("https://pbs.twimg.com/media/A?format=webp&name=medium"),
            "https://pbs.twimg.com/media/A?format=webp&name=large"
        );
        for other in [
            "https://pbs.twimg.com/media/A?format=webp&name=large",
            "https://pbs.twimg.com/media/A?format=webp&name=4096x4096",
            "https://pbs.twimg.com/media/A?format=webp",
            "https://pbs.twimg.com/profile_images/1/a_normal.jpg",
            "https://pbs.twimg.com/amplify_video_thumb/9/img/p?format=jpg&name=medium",
            "https://example.com/media/A?format=jpg&name=medium",
            "not a url",
        ] {
            assert_eq!(original_image(other), other);
        }
    }
}
