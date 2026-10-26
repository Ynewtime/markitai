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
use super::hacker_news::escaped;
use super::{Announcement, Landmarks, render_clean, selector};
use crate::Result;
use crate::formats::link_destination;
use regex::Regex;
use scraper::{ElementRef, Html};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
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
    let tree = super::parse::fragment(&element.html());
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

/// The poster address that names a video's picture, whatever size and
/// format the page asks for: `pbs.twimg.com` and the path without its
/// extension (`/media/Eu2ND` for `/media/Eu2ND.jpg` and
/// `/media/Eu2ND?format=webp&name=medium`).
fn poster_key(url: &str) -> Option<String> {
    let url = Url::parse(url.trim()).ok()?;
    if url.host_str() != Some("pbs.twimg.com") {
        return None;
    }
    let path = url.path();
    let stem = match path.rsplit_once('.') {
        Some((stem, extension)) if !extension.contains('/') => stem,
        _ => path,
    };
    Some(stem.to_owned())
}

static MEDIA_URL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""?media_url_https"?\s*:\s*"([^"]+)""#).expect("pattern"));
static MP4_VARIANT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"\{[^{}]*video/mp4[^{}]*\}"#).expect("pattern"));
static VARIANT_URL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""?url"?\s*:\s*"([^"]+)""#).expect("pattern"));
static VARIANT_BITRATE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""?bitrate"?\s*:\s*(\d+)"#).expect("pattern"));

/// The video files a post page names for its videos, by poster (see
/// [`poster_key`]). The page draws a video as its poster and a player that
/// streams it (a browser-local `blob:` address); the post's data, which X
/// serializes into the page's scripts, lists the files as `video/mp4`
/// variants after the media's `media_url_https`. The largest one is kept,
/// and only on X's video host.
fn video_files(document: &Html) -> HashMap<String, String> {
    let mut files = HashMap::new();
    for script in document.select(&selector("script")) {
        let text = script.text().collect::<String>();
        if !text.contains("video/mp4") {
            continue;
        }
        let unescape = |value: &str| {
            value
                .replace("\\/", "/")
                .replace("\\u002F", "/")
                .replace("\\u002f", "/")
                .replace("\\u0026", "&")
        };
        let starts: Vec<_> = MEDIA_URL.captures_iter(&text).collect();
        for (index, captures) in starts.iter().enumerate() {
            let whole = captures.get(0).expect("match");
            let end = starts
                .get(index + 1)
                .map_or(text.len(), |next| next.get(0).expect("match").start());
            let Some(key) = poster_key(&unescape(&captures[1])) else {
                continue;
            };
            let best = MP4_VARIANT
                .find_iter(&text[whole.end()..end])
                .filter_map(|variant| {
                    let variant = variant.as_str();
                    let url = unescape(&VARIANT_URL.captures(variant)?[1]);
                    let bitrate = VARIANT_BITRATE
                        .captures(variant)
                        .and_then(|found| found[1].parse::<u64>().ok())
                        .unwrap_or(0);
                    Url::parse(&url)
                        .is_ok_and(|parsed| {
                            parsed.scheme() == "https"
                                && parsed.host_str() == Some("video.twimg.com")
                        })
                        .then_some((bitrate, url))
                })
                .max_by_key(|(bitrate, _)| *bitrate);
            if let Some((_, url)) = best {
                files.entry(key).or_insert(url);
            }
        }
    }
    files
}

/// The scripts of an X post page that name video files (see
/// [`video_files`]), as markup to add to another copy of the page: the page a
/// browser drew has run and removed them.
pub(crate) fn video_data_scripts(html: &str) -> String {
    let document = super::parse::document(html);
    let mut scripts = String::new();
    for script in document.select(&selector("script")) {
        let text = script.text().collect::<String>();
        // Text of a parsed script cannot close it early.
        if text.contains("video/mp4") && text.contains("media_url_https") {
            scripts.push_str("<script>");
            scripts.push_str(&text);
            scripts.push_str("</script>");
        }
    }
    scripts
}

/// Images and videos of a post in page order, each once, as Markdown: a photo
/// at its original size, a video as its poster linked to its file
/// (`[![Video](poster)](file.mp4)`, as a page's `<video>` is written) when
/// the page names one, either as the element's address or in the post's data
/// (`files`), and as its poster alone when it only has the browser-local
/// `blob:` handle.
fn media(
    post: ElementRef<'_>,
    exclude: Option<ElementRef<'_>>,
    files: &HashMap<String, String>,
) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut add = |line: String| {
        if !lines.contains(&line) {
            lines.push(line);
        }
    };
    let https = |url: Option<&str>| {
        url.map(str::trim)
            .filter(|url| url.starts_with("https://"))
            .map(str::to_owned)
    };
    // A player draws its poster as a picture too, which is not a photo.
    let posters: HashSet<String> = post
        .select(&selector("video[poster]"))
        .filter_map(|video| poster_key(video.value().attribute("poster")?))
        .collect();
    for element in post.select(&selector(
        r#"video, [data-testid="tweetPhoto"] img, a[aria-label="Image"] img"#,
    )) {
        if inside(element, exclude) {
            continue;
        }
        if element.value().name() == "video" {
            let poster = https(element.value().attribute("poster"));
            let source = element.value().attribute("src").or_else(|| {
                element
                    .select(&selector("source[src]"))
                    .find_map(|source| source.value().attribute("src"))
            });
            let file = https(source).or_else(|| {
                poster
                    .as_deref()
                    .and_then(poster_key)
                    .and_then(|key| files.get(&key).cloned())
            });
            let label = if file
                .as_deref()
                .is_some_and(|file| file.contains("/tweet_video/"))
            {
                "GIF"
            } else {
                "Video"
            };
            match (poster, file) {
                (Some(poster), Some(file)) => add(format!(
                    "[{}]({})",
                    image_markdown(&poster).replacen("![]", &format!("![{label}]"), 1),
                    link_destination(&file)
                )),
                (Some(poster), None) => {
                    add(image_markdown(&poster).replacen("![]", &format!("![{label}]"), 1));
                }
                (None, Some(file)) => add(format!("[{label}]({})", link_destination(&file))),
                (None, None) => {}
            }
        } else if let Some(photo) = https(element.value().attribute("src"))
            && !poster_key(&photo).is_some_and(|key| posters.contains(&key))
        {
            add(image_markdown(&original_image(&photo)));
        }
    }
    lines
}

fn image_markdown(url: &str) -> String {
    format!("![]({})", link_destination(url))
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
    status_address(url).filter(|canonical| canonical != url)
}

/// The canonical `https://x.com/<user>/status/<id>` (or `/article/<id>`)
/// address of a post named through X's hosts or a mirror, canonical or not.
pub(crate) fn status_address(url: &Url) -> Option<Url> {
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
    Url::parse(&format!("https://x.com/{user}/{kind}/{id}")).ok()
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
    fn blocks(&self, base: Option<&Url>, files: &HashMap<String, String>) -> Result<Vec<String>> {
        let mut blocks = Vec::new();
        if let Some(text) = text_element(self.element, self.quote) {
            let text = text_markdown(text, base)?;
            if !text.is_empty() {
                blocks.push(text);
            }
        }
        blocks.extend(media(self.element, self.quote, files));
        if let Some(quote) = self.quote
            && let Some(block) = quote_block(quote, base, files)?
        {
            blocks.push(block);
        }
        Ok(blocks)
    }
}

/// The quoted post as a block quote: `**Name @handle** · date`, its text and
/// media, and a link to it.
fn quote_block(
    quote: ElementRef<'_>,
    base: Option<&Url>,
    files: &HashMap<String, String>,
) -> Result<Option<String>> {
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
    lines.extend(media(quote, None, files));
    if let Some(href) = quote.value().attribute("data-href")
        && let Some(url) = Url::parse("https://x.com")
            .ok()
            .and_then(|x| super::safe_url(href, Some(&x)))
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

/// An Article's title and cover pictures beside its body: the `h1` and
/// pictures around an `x-article-body`, or, on the 2026 page, the
/// `twitter-article-title` and the photo above `twitterArticleRichTextView` in
/// their `twitterArticleReadView`.
fn article_heading(body: ElementRef<'_>) -> Option<(ElementRef<'_>, Vec<ElementRef<'_>>)> {
    if body
        .value()
        .classes()
        .any(|class| class == "x-article-body")
    {
        let container = body.parent().and_then(ElementRef::wrap)?;
        let heading = container
            .child_elements()
            .find(|child| child.value().name() == "h1" && !text_of(*child).is_empty())?;
        let covers = container
            .child_elements()
            .filter(|child| child.value().name() == "img")
            .collect();
        return Some((heading, covers));
    }
    let view = body
        .ancestors()
        .filter_map(ElementRef::wrap)
        .find(|node| node.value().attribute("data-testid") == Some("twitterArticleReadView"))?;
    let heading = view
        .select(&selector(r#"[data-testid="twitter-article-title"]"#))
        .find(|title| !text_of(*title).is_empty())?;
    let covers = view
        .select(&selector(r#"[data-testid="tweetPhoto"] img"#))
        .filter(|photo| !inside(*photo, Some(body)))
        .take(1)
        .collect();
    Some((heading, covers))
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
        .select(&selector(
            r#".x-article-body, [data-testid="twitterArticleRichTextView"]"#,
        ))
        .filter_map(|body| {
            if body
                .ancestors()
                .filter_map(ElementRef::wrap)
                .any(|node| matches!(node.value().name(), "template" | "noscript" | "head"))
            {
                return None;
            }
            let (heading, covers) = article_heading(body)?;
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
            Some((body, heading, covers, owner, id))
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
    let Some((body, heading, covers, owner, id)) = selected else {
        return Ok(None);
    };
    // Preserve the document's block structure; post text_markdown deliberately
    // separates every line and would break nested lists and code in an Article.
    let body_markdown = render_clean(*body, base)?;
    if body_markdown.is_empty() {
        return Ok(None);
    }
    let mut blocks = vec![if heading.value().name() == "h1" {
        render_clean(*heading, base)?
    } else {
        format!("# {}", text_of(*heading).replace('#', "\\#"))
    }];
    for cover in covers {
        let markdown = if heading.value().name() == "h1" {
            render_clean(*cover, base)?
        } else {
            cover
                .value()
                .attribute("src")
                .filter(|src| src.trim().starts_with("https://"))
                .map(|src| image_markdown(&original_image(src.trim())))
                .unwrap_or_default()
        };
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

/// A reply under `## Comments`, as the reference writes it: `**Name @handle**
/// · [date](permalink)`, then the reply's text, media and quoted post. The
/// post's author's own reply says `(author)` after the name.
fn comment(
    reply: &Post<'_>,
    by_author: bool,
    base: Option<&Url>,
    files: &HashMap<String, String>,
) -> Result<Option<Vec<String>>> {
    let mut header = Vec::new();
    let who = match (&reply.name, &reply.handle) {
        (Some(name), Some(handle)) => format!("{name} {handle}"),
        (Some(one), None) | (None, Some(one)) => one.clone(),
        (None, None) => String::new(),
    };
    if !who.is_empty() {
        let mark = if by_author { " (author)" } else { "" };
        header.push(format!("**{}**{mark}", escaped(&who)));
    }
    let (iso, shown) = shown_date(reply.element, reply.quote);
    let id = own_id(reply.element);
    if let Some(date) = published(iso.as_deref(), id.as_deref(), None).or(shown) {
        let permalink = own_permalink(reply.element, reply.quote).and_then(|href| {
            Url::parse("https://x.com")
                .ok()
                .and_then(|x| super::safe_url(href, Some(&x)))
        });
        header.push(match permalink {
            Some(url) => format!("[{date}]({url})"),
            None => date,
        });
    }
    let blocks = reply.blocks(base, files)?;
    if header.is_empty() && blocks.is_empty() {
        return Ok(None);
    }
    let mut lines = Vec::new();
    if !header.is_empty() {
        lines.push(header.join(" · "));
    }
    for block in blocks {
        if !lines.is_empty() {
            lines.push(String::new());
        }
        lines.extend(block.lines().map(str::to_owned));
    }
    Ok(Some(lines))
}

/// A thread part's number opening a post: `2/`, `2/n`, `2/5` or `(2/5)`.
static THREAD_NUMBER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*\(?(\d{1,2})\s*/\s*(\d{1,2}|n)?\)?(?:\s|$)").expect("thread number pattern")
});

/// Whether `text` opens with a thread part's number: a small `n` (1 to 50),
/// and in `n/m` no more than `m`. A date such as `10/24` at the end, or
/// `13/5` at the start, is no part number.
fn thread_number(text: &str) -> bool {
    THREAD_NUMBER.captures(text).is_some_and(|found| {
        let part: u32 = found[1].parse().unwrap_or(0);
        let total = found
            .get(2)
            .and_then(|total| total.as_str().parse::<u32>().ok());
        (1..=50).contains(&part) && total.is_none_or(|total| part <= total)
    })
}

/// Whether the author's reply `next`, after their post or part `previous`,
/// continues the thread rather than answering under it: either opens with a
/// part number (`2/`, `(2/5)`), the post opens a thread (`🧵`, `thread:`, `thread 👇`), the
/// earlier text breaks off (`,` `:` `;` `…` or a dash), or the reply picks
/// up mid-sentence (a lowercase letter or `…`). X's markup does not say
/// which replies are the self-thread, so a reply without these is a
/// comment, like a "tag them with feedback" note.
fn continues_thread(main: &str, previous: &str, next: &str) -> bool {
    let opens = |text: &str| {
        let lower = text.to_lowercase();
        text.contains('\u{1F9F5}')
            || [
                "thread:",
                "thread\u{1F447}",
                "thread \u{1F447}",
                "thread below",
            ]
            .iter()
            .any(|cue| lower.contains(cue))
    };
    let previous = previous.trim_end();
    let next = next.trim_start();
    thread_number(next)
        || thread_number(main)
        || opens(main)
        || previous.ends_with([',', ':', ';', '…', '-', '–', '—'])
        || previous.ends_with("...")
        || next.starts_with('…')
        || next.starts_with("...")
        || next.chars().next().is_some_and(char::is_lowercase)
}

/// The posts that follow `main` in the conversation: those before the first
/// heading outside a post after it, where X starts "Discover more" and other
/// timelines, as the reference stops there.
fn conversation(scope: ElementRef<'_>, main: ElementRef<'_>) -> HashSet<usize> {
    let mut posts = HashSet::new();
    let mut after_main = false;
    for node in scope.descendants().filter_map(ElementRef::wrap) {
        if node == main {
            after_main = true;
        } else if after_main {
            match node.value().name() {
                "article" => {
                    posts.insert(super::element_key(node));
                }
                "h2" if !node
                    .ancestors()
                    .filter_map(ElementRef::wrap)
                    .any(|parent| parent.value().name() == "article") =>
                {
                    break;
                }
                _ => {}
            }
        }
    }
    posts
}

/// The post of an X status page as Markdown with its metadata: the post, then
/// the posts the author continued it with (see [`continues_thread`]), each
/// after a rule. Replies from other accounts, and the author's own replies
/// that do not continue it or come after the first comment, are the
/// discussion around the post: they follow under `## Comments` in the order
/// the page lists them, as x.com shows them: X puts the author's own replies
/// first and ranks the others itself, and an answer to a reply follows it.
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
    let files = video_files(document);
    let lead = main.blocks(base, &files)?;
    if lead.is_empty() {
        return Ok(None);
    }
    let main_id = own_id(main.element);
    let mut sections = vec![lead.join("\n\n")];
    // Consecutive posts of the author right after the main one continue it
    // when they read as its parts; the first other post ends the thread. A
    // page can repeat the main post, which is not a continuation of itself.
    let mut seen = vec![main_id.clone()];
    let in_conversation = conversation(scope, main.element);
    let mut comments: Vec<Vec<String>> = Vec::new();
    let mut thread_ended = false;
    let plain_text = |post: &Post<'_>| {
        text_element(post.element, post.quote)
            .map(text_of)
            .unwrap_or_default()
    };
    let main_text = plain_text(&main);
    let mut previous_text = main_text.clone();
    for article in &timeline[position + 1..] {
        let id = own_id(*article);
        if id.is_some() && seen.contains(&id) {
            continue;
        }
        if !in_conversation.contains(&super::element_key(*article)) {
            break;
        }
        seen.push(id);
        let reply = Post::read(*article);
        let by_author = reply.same_author(&main);
        if !thread_ended {
            let text = plain_text(&reply);
            thread_ended = !by_author || !continues_thread(&main_text, &previous_text, &text);
            previous_text = text;
        }
        if thread_ended {
            if let Some(lines) = comment(&reply, by_author, base, &files)? {
                comments.push(lines);
            }
            continue;
        }
        let blocks = reply.blocks(base, &files)?;
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
    let plain = main_text;
    if !plain.is_empty() {
        let mut description: String = plain.chars().take(200).collect();
        description.truncate(description.trim_end().len());
        metadata.insert("description".into(), Value::String(description));
    }
    // The body is already the post; a language model keeps it as it is.
    metadata.insert("content_profile".into(), "social_post".into());
    let mut markdown = sections.join("\n\n---\n\n");
    if !comments.is_empty() {
        markdown.push_str("\n\n---\n\n## Comments\n\n");
        let quoted: Vec<String> = comments
            .iter()
            .map(|lines| {
                lines
                    .iter()
                    .map(|line| {
                        if line.is_empty() {
                            ">".to_owned()
                        } else {
                            format!("> {line}")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .collect();
        markdown.push_str(&quoted.join("\n\n"));
    }
    Ok(Some(Announcement { markdown, metadata }))
}

#[cfg(test)]
mod tests {
    use super::super::extract_html;
    use super::{
        canonical_status_url, continues_thread, id_date, image_markdown, original_image, published,
    };
    use url::Url;

    #[test]
    fn a_post_image_address_is_written_as_one_destination() {
        assert_eq!(
            image_markdown("https://pbs.example/media/a b(c)<d>.jpg"),
            "![](https://pbs.example/media/a%20b%28c%29%3Cd%3E.jpg)"
        );
    }

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
            "First line of the post.\n\nSecond line with [example.com/tool](https://example.com/tool)\n\n![Video](https://pbs.twimg.com/amplify_video_thumb/9/img/p.jpg)\n\n> **Grace @grace** · Jun 19\n>\n> The quoted post.\n>\n> ![](https://pbs.twimg.com/media/Q?format=jpg&name=orig)\n>\n> [https://x.com/grace/status/2](https://x.com/grace/status/2)\n\n---\n\n## Comments\n\n> A reply."
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
            "Clips\n\n![Video](https://pbs.twimg.com/amplify_video_thumb/9/img/p?format=webp&name=medium)\n\n[![GIF](https://pbs.twimg.com/tweet_video_thumb/g.jpg)](https://video.twimg.com/tweet_video/g.mp4)"
        );
        assert!(!doc.markdown.contains("00:00"));
    }

    #[test]
    fn a_video_links_the_largest_file_the_posts_data_names() {
        // X's server-rendered page: the player is a poster (also drawn as a
        // picture), and the post's data in a script lists the files.
        let state = r#"<script>window.state={media_entities2:[{id_str:"5",media_url_https:"https://pbs.twimg.com/media/Lamp5.jpg",type:"video",video_info:{variants:[{content_type:"application/x-mpegURL",url:"https://video.twimg.com/amplify_video/5/pl/a.m3u8"},{bitrate:288000,content_type:"video/mp4",url:"https://video.twimg.com/amplify_video/5/vid/480x270/s.mp4?tag=1"},{bitrate:2176000,content_type:"video/mp4",url:"https://video.twimg.com/amplify_video/5/vid/1280x720/l.mp4?tag=1"}]}},{media_url_https:"https://pbs.twimg.com/media/Other6.jpg",type:"video",video_info:{variants:[{bitrate:9,content_type:"video/mp4",url:"https://elsewhere.example/v.mp4"}]}}]}</script>"#;
        let player = r#"<div data-testid="tweetPhoto"><video poster="https://pbs.twimg.com/media/Lamp5?format=webp&amp;name=medium"></video>
            <img alt="" src="https://pbs.twimg.com/media/Lamp5?format=webp&amp;name=medium"></div>
            <video poster="https://pbs.twimg.com/media/Other6?format=jpg&amp;name=small" src="blob:https://x.com/9"></video>"#;
        let main = article("ada", "Ada", ID, "Feb 25, 2025", "Lamp test", player);
        let doc = extract_html(&format!("{}{state}", page(&main, "")), Some(STATUS)).unwrap();
        assert_eq!(
            doc.markdown,
            "Lamp test\n\n[![Video](https://pbs.twimg.com/media/Lamp5?format=webp&name=medium)](https://video.twimg.com/amplify_video/5/vid/1280x720/l.mp4?tag=1)\n\n![Video](https://pbs.twimg.com/media/Other6?format=jpg&name=small)"
        );
        // The data a browser's copy of the page has lost is taken from the
        // server-rendered page's scripts alone.
        let scripts = super::video_data_scripts(&format!(
            "<html><body><script>a=1</script>{state}</body></html>"
        ));
        assert!(scripts.starts_with("<script>window.state=") && !scripts.contains("a=1"));
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
            "Good luck!\n\n> **Grace Hopper @grace** · 2025-02-24\n>\n> A quoted line.\n>\n> ![Video](https://pbs.twimg.com/amplify_video_thumb/7/img/q?format=webp&name=medium)\n>\n> [https://x.com/grace/status/1894160928502858201](https://x.com/grace/status/1894160928502858201)"
        );
        assert_eq!(doc.metadata["author"], "@ada");
        assert_eq!(doc.metadata["published"], "2025-02-25");
        assert_eq!(doc.markdown.matches("Good luck!").count(), 1);
    }

    #[test]
    fn a_quoted_post_link_with_an_unsafe_scheme_is_not_written() {
        let quote = r#"<div role="link" data-href="javascript:alert(1)//status/1"><article>
            <div dir="auto">A quoted line.</div>
            </article></div>"#;
        let main = article("ada", "Ada", ID, "3:04 AM · Feb 25, 2025", "Hi", quote);
        let doc = extract_html(&page(&main, ""), Some(STATUS)).unwrap();
        assert!(
            doc.markdown.contains("> A quoted line."),
            "{}",
            doc.markdown
        );
        assert!(!doc.markdown.contains("javascript:"), "{}", doc.markdown);
    }

    #[test]
    fn the_authors_own_replies_continue_the_post_and_the_rest_are_comments() {
        let main = article(
            "ada",
            "Ada",
            ID,
            "3:04 AM · Feb 25, 2025",
            "1/3 Part one",
            "",
        );
        let replies = format!(
            "{}{}{}{}{}",
            card("ada", "Ada", "1894221971766084050", "2/3 Part two"),
            card("ada", "Ada", "1894221971766084051", "3/3 Part three"),
            card("grace", "Grace", "1894221971766084052", "A comment"),
            card("ada", "Ada", "1894221971766084053", "Thanks Grace"),
            card("linus", "Linus", "1894221971766084054", "Another comment"),
        );
        // The thread's earlier post comes first, as X lists it, and is not
        // part of this post.
        let earlier = card("ada", "Ada", "1894221971766083999", "Earlier");
        let html = page(&main, &replies).replace("<main>", &format!("<main><ul>{earlier}</ul>"));
        let doc = extract_html(&html, Some(STATUS)).unwrap();
        // The first reply from another account ends the thread: it and every
        // reply after it, the author's own included, are comments, in the
        // page's order: the author's answer stays under the reply it answers.
        assert_eq!(
            doc.markdown,
            "1/3 Part one\n\n---\n\n2/3 Part two\n\n---\n\n3/3 Part three\n\n---\n\n## Comments\n\n\
             > **Grace @grace** · [2025-02-25](https://x.com/grace/status/1894221971766084052)\n>\n> A comment\n\n\
             > **Ada @ada** (author) · [2025-02-25](https://x.com/ada/status/1894221971766084053)\n>\n> Thanks Grace\n\n\
             > **Linus @linus** · [2025-02-25](https://x.com/linus/status/1894221971766084054)\n>\n> Another comment"
        );
        assert_eq!(doc.metadata["description"], "1/3 Part one");
        assert_eq!(doc.metadata["published"], "2025-02-25");
    }

    #[test]
    fn comments_keep_the_order_x_ranks_them_in_not_their_post_ids() {
        let main = article("ada", "Ada", ID, "3:04 AM · Feb 25, 2025", "Big news.", "");
        // X lists the author's reply first and ranks a later reply above an
        // earlier one; the author's answer to Grace follows her reply.
        let replies = format!(
            "{}{}{}{}",
            card(
                "ada",
                "Ada",
                "1894221971766084062",
                "Tag Grace with feedback."
            ),
            card("linus", "Linus", "1894221971766084060", "Most relevant"),
            card(
                "grace",
                "Grace",
                "1894221971766084052",
                "Earlier, less relevant"
            ),
            card("ada", "Ada", "1894221971766084061", "Answer to Grace"),
        );
        let doc = extract_html(&page(&main, &replies), Some(STATUS)).unwrap();
        let order: Vec<&str> = doc
            .markdown
            .lines()
            .filter_map(|line| {
                line.strip_prefix("> ")
                    .filter(|text| !text.starts_with("**"))
            })
            .collect();
        assert_eq!(
            order,
            [
                "Tag Grace with feedback.",
                "Most relevant",
                "Earlier, less relevant",
                "Answer to Grace"
            ],
            "{}",
            doc.markdown
        );
    }

    /// A static fetch of a 2026 post (fixtures/x-author-reply.html, saved as
    /// fetched): the author's "tag them with feedback" reply is neither
    /// numbered nor continues the post, so it is a comment, marked as the
    /// author's and listed first, as the page (and x.com) lists it, though
    /// the other reply was posted before it. The page holds only these two
    /// replies.
    #[test]
    fn an_authors_reply_that_does_not_continue_the_post_is_a_comment() {
        let html = include_str!("fixtures/x-author-reply.html");
        let doc = extract_html(
            html,
            Some("https://x.com/amorriscode/status/2113972174524674247"),
        )
        .unwrap();
        let (body, comments) = doc
            .markdown
            .split_once("\n\n---\n\n## Comments\n\n")
            .unwrap();
        assert!(
            body.starts_with("Yesterday was my last day at Anthropic."),
            "{body}"
        );
        assert!(body.ends_with("I'll share what's next soon!"), "{body}");
        assert!(!body.contains("---") && !body.contains("Tag "), "{body}");
        assert_eq!(
            comments,
            "> **Anthony Morris ツ @amorriscode** (author) · [2026-10-24](https://x.com/amorriscode/status/2114004396518650193)\n>\n\
             > Tag [@alicelovescake1](https://x.com/alicelovescake1) and [@BenjaminPasero](https://x.com/BenjaminPasero) with any CCD feedback going forward. They're the best!\n\n\
             > **Thariq @trq212** · [2026-10-24](https://x.com/trq212/status/2114001556895072741)\n>\n> \u{1F410}"
        );
    }

    #[test]
    fn numbered_or_broken_off_replies_continue_the_thread() {
        for (main, reply, continues) in [
            ("Shipping today 🧵", "The details.", true),
            ("Here is how it works:", "First, the parser.", true),
            ("It started small", "and then it grew.", true),
            ("A long story…", "…that ends here.", true),
            ("Big news.", "(2/4) More news.", true),
            ("1/ Big news.", "More news.", true),
            ("Big news.", "2/n More news.", true),
            ("See you on 10/24", "Doors open at six.", false),
            ("Big news.", "Moved to 10/24.", false),
            ("Big news.", "13/5 is the score.", false),
            ("Big news.", "2. More news.", false),
            ("Big news.", "Tag Grace with feedback.", false),
            ("Big news!", "Thanks, everyone.", false),
        ] {
            assert_eq!(
                continues_thread(main, main, reply),
                continues,
                "{main} / {reply}"
            );
        }
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
            post("Ada", "ada", "Opening, and"),
            post("Ada", "ada", "continued"),
            post("Grace", "grace", "Question"),
            post("Ada", "ada", "Answer")
        );
        let doc = extract_html(&html, Some("https://twitter.com/ada/status/7")).unwrap();
        assert_eq!(
            doc.markdown,
            "Opening, and\n\n---\n\ncontinued\n\n---\n\n## Comments\n\n> **Grace @grace**\n>\n> Question\n\n> **Ada @ada** (author)\n>\n> Answer"
        );
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
            assert_eq!(
                doc.markdown,
                "The actual post.\n\n---\n\n## Comments\n\n> **Grace @grace** · [Feb 25, 2025](https://x.com/grace/status/77)\n>\n> Reply"
            );
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
    fn a_reply_in_a_localized_timeline_is_a_comment_and_discover_more_is_not() {
        // The main post is in the conversation's `section` too, and the
        // timeline's label is not English.
        let post = |name: &str, handle: &str, id: &str, text: &str| {
            format!(
                r#"<div data-testid="cellInnerDiv"><article data-testid="tweet"><div data-testid="User-Name"><a href="/{handle}">{name}</a><a href="/{handle}">@{handle}</a></div>
                <a href="/{handle}/status/{id}"><time datetime="2026-05-16T12:00:00.000Z">5月16日</time></a><div data-testid="tweetText">{text}</div></article></div>"#
            )
        };
        let html = format!(
            r#"<html lang="zh-Hans"><body><main><section aria-label="时间线：对话">{}<div data-testid="cellInnerDiv"><div role="separator"></div></div>{}<div data-testid="cellInnerDiv"><h2>发现更多</h2></div>{}</section></main></body></html>"#,
            post("Main", "main_user", "1234567890", "主帖"),
            post("Reply", "reply_user", "1234567891", "回复"),
            post("Other", "other", "1234567892", "推荐"),
        );
        let doc = extract_html(&html, Some("https://x.com/main_user/status/1234567890")).unwrap();
        assert_eq!(
            doc.markdown,
            "主帖\n\n---\n\n## Comments\n\n> **Reply @reply\\_user** · [2026-05-16](https://x.com/reply_user/status/1234567891)\n>\n> 回复"
        );
        assert_eq!(doc.metadata["title"], "Post by @main_user on X");
    }

    #[test]
    fn a_2026_article_in_its_status_page_is_its_title_cover_and_whole_body() {
        // The read view of early 2026: title and cover above the rich text,
        // counters between them, the body in Draft.js blocks.
        let view = r#"<div data-testid="twitterArticleReadView">
            <a href="/ada/article/1894221971766084049/media/5"><div data-testid="tweetPhoto">
              <img src="https://pbs.twimg.com/media/C?format=jpg&amp;name=medium"></div></a>
            <div dir="auto" data-testid="twitter-article-title"><span>Notes on notes</span></div>
            <div role="group" aria-label="3 replies"><button data-testid="reply">3</button></div>
            <div data-testid="twitterArticleRichTextView"><div data-testid="longformRichTextComponent">
              <div class="longform-unstyled"><div class="public-DraftStyleDefault-block"><span>First paragraph.</span></div></div>
              <div dir="ltr"><h2 class="longform-header-two"><div><span>A section</span></div></h2></div>
              <ul><li><div><span>one</span></div></li><li><div><span>two</span></div></li></ul>
              <div data-testid="markdown-code-block"><div><div dir="ltr"><span>markdown</span></div>
                <div><button aria-label="Copy to clipboard"></button></div></div>
                <pre><code class="language-markdown"><span>
vault/
└── notes/</span></code></pre></div>
              <div class="longform-unstyled"><div><span>Last paragraph.</span></div></div>
            </div></div></div>"#;
        let main = article("ada", "Ada", ID, "Feb 25, 2025", "", view);
        let html = page(&main, "");
        let doc = extract_html(&html, Some(STATUS)).unwrap();
        assert_eq!(
            doc.markdown,
            "# Notes on notes\n\n![](https://pbs.twimg.com/media/C?format=jpg&name=orig)\n\nFirst paragraph.\n\n## A section\n\n* one\n\n* two\n\n```markdown\nvault/\n└── notes/\n```\n\nLast paragraph."
        );
        assert_eq!(doc.metadata["title"], "Notes on notes");
        assert_eq!(doc.metadata["author"], "@ada");
        assert_eq!(doc.metadata["site"], "X (Twitter)");
        assert!(doc.metadata.get("content_profile").is_none());
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
