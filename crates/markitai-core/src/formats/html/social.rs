//! Posts on X (Twitter): the post itself, its media and the post it quotes,
//! without the page's avatars, player controls, counters and timelines.
//!
//! Both page generations are read: the older one marks its parts with
//! `data-testid` (`tweet`, `User-Name`, `tweetText`, `tweetPhoto`), the 2026
//! one carries `data-tweet-id` on the post and no test ids at all.

use super::Attribute;
use super::{Announcement, render_clean, selector};
use crate::Result;
use scraper::{ElementRef, Html};
use serde_json::{Map, Value};
use url::Url;

fn inside(element: ElementRef<'_>, container: Option<ElementRef<'_>>) -> bool {
    container.is_some_and(|container| element.ancestors().any(|node| node.id() == container.id()))
}

fn is_post(element: ElementRef<'_>) -> bool {
    element.value().name() == "article"
        && (element.value().attribute("data-testid") == Some("tweet")
            || element.value().attribute("data-tweet-id").is_some())
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

/// The post's author: its display name and `@handle`.
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

/// Images and video posters of a post, in page order, each once.
fn media(post: ElementRef<'_>, exclude: Option<ElementRef<'_>>) -> Vec<String> {
    let mut urls: Vec<String> = Vec::new();
    for element in post.select(&selector(
        r#"video[poster], [data-testid="tweetPhoto"] img, a[aria-label="Image"] img"#,
    )) {
        if inside(element, exclude) {
            continue;
        }
        let url = if element.value().name() == "video" {
            element.value().attribute("poster")
        } else {
            element.value().attribute("src")
        };
        if let Some(url) = url.map(str::trim).filter(|url| url.starts_with("https://"))
            && !urls.iter().any(|seen| seen == url)
        {
            urls.push(url.to_owned());
        }
    }
    urls
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

/// `YYYY-MM-DD` from an ISO timestamp or shown text with a year.
fn published(iso: Option<&str>, shown: Option<&str>) -> Option<String> {
    if let Some(iso) = iso
        && let Ok(time) = chrono::DateTime::parse_from_rfc3339(iso)
    {
        return Some(time.date_naive().to_string());
    }
    let shown = shown?;
    let date = shown.rsplit('·').next()?.trim();
    chrono::NaiveDate::parse_from_str(date, "%b %d, %Y")
        .ok()
        .map(|date| date.to_string())
}

/// Whether a page is an X post page: by its host (a status URL), else by
/// X's own test ids or its media host.
fn is_x_page(document: &Html, base: Option<&Url>) -> bool {
    let x_host = |host: &str| {
        matches!(
            host,
            "x.com" | "twitter.com" | "mobile.twitter.com" | "www.x.com" | "www.twitter.com"
        )
    };
    if let Some(host) = base.and_then(Url::host_str) {
        return x_host(host) && base.is_some_and(|url| url.path().contains("/status/"));
    }
    // X's own column and post test ids, or its media host.
    if document
        .select(&selector(
            r#"[data-testid="primaryColumn"] article[data-testid="tweet"]"#,
        ))
        .next()
        .is_some()
    {
        return true;
    }
    document
        .select(&selector("img[src], video[poster]"))
        .any(|element| {
            element
                .value()
                .attribute("src")
                .or_else(|| element.value().attribute("poster"))
                .and_then(|source| Url::parse(source).ok())
                .is_some_and(|url| url.host_str() == Some("pbs.twimg.com"))
        })
}

/// The post of an X status page as Markdown with its metadata.
pub(super) fn post(document: &Html, base: Option<&Url>) -> Result<Option<Announcement>> {
    if !is_x_page(document, base) {
        return Ok(None);
    }
    let column = document
        .select(&selector(r#"[data-testid="primaryColumn"], main"#))
        .next()
        .unwrap_or_else(|| document.root_element());
    // The main post: not a quoted post and not a reply in the timeline below.
    let Some(main) = column.select(&selector("article")).find(|article| {
        is_post(*article)
            && !article
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
    }) else {
        return Ok(None);
    };
    let quote = quoted(main);
    let (name, handle) = author(main, quote);
    let Some(text) = text_element(main, quote) else {
        return Ok(None);
    };
    let body = text_markdown(text, base)?;
    if body.is_empty() {
        return Ok(None);
    }
    let mut blocks = vec![body];
    blocks.extend(media(main, quote).iter().map(|url| image_markdown(url)));
    if let Some(quote) = quote {
        let (quote_name, quote_handle) = author(quote, None);
        let (_, quote_date) = shown_date(quote, None);
        let mut lines = Vec::new();
        let who = match (&quote_name, &quote_handle) {
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
        if let Some(date) = quote_date {
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
        lines.extend(media(quote, None).iter().map(|url| image_markdown(url)));
        if let Some(href) = quote.value().attribute("data-href")
            && let Some(url) = Url::parse("https://x.com")
                .ok()
                .and_then(|x| x.join(href).ok())
        {
            lines.push(format!("[{url}]({url})"));
        }
        if !lines.is_empty() {
            let quoted = lines
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
                .join("\n");
            blocks.push(quoted);
        }
    }
    let mut metadata = Map::new();
    let who = handle.clone().or_else(|| name.clone());
    if let Some(who) = &who {
        metadata.insert("title".into(), format!("Post by {who} on X").into());
        metadata.insert("author".into(), who.clone().into());
    }
    metadata.insert("site".into(), "X (Twitter)".into());
    let (iso, shown) = shown_date(main, quote);
    if let Some(date) = published(iso.as_deref(), shown.as_deref()) {
        metadata.insert("published".into(), date.into());
    }
    let plain = text_of(text);
    if !plain.is_empty() {
        let mut description: String = plain.chars().take(200).collect();
        description.truncate(description.trim_end().len());
        metadata.insert("description".into(), Value::String(description));
    }
    Ok(Some(Announcement {
        markdown: blocks.join("\n\n"),
        metadata,
    }))
}

#[cfg(test)]
mod tests {
    use super::super::extract_html;

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
            "First line of the post.\n\nSecond line with [example.com/tool](https://example.com/tool)\n\n![](https://pbs.twimg.com/amplify_video_thumb/9/img/p.jpg)\n\n> **Grace @grace** · Jun 19\n>\n> The quoted post.\n>\n> ![](https://pbs.twimg.com/media/Q?format=jpg&name=medium)\n>\n> [https://x.com/grace/status/2](https://x.com/grace/status/2)"
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
}
