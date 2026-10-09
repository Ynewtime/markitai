//! YouTube watch pages (`youtube.com/watch?v=<id>`, `youtu.be/<id>`).
//!
//! A page a browser rendered (the local browser's result, or a page saved from
//! a browser) carries the video's title, channel, description, counts and the
//! comments loaded so far in its `ytd-*` elements; the guide, the masthead and
//! the like, dislike, share and save buttons beside them are not part of it.
//! A page as the server sends it has none of that yet: the
//! `ytInitialPlayerResponse` its scripts render from names the title, channel,
//! description, view count and publication date.

use super::{Article, escaped, first, text_at, text_of};
use scraper::{ElementRef, Html};
use serde_json::Value;

/// The player response is read from at most this many bytes of its script.
const MAX_SCRIPT: usize = 4 * 1024 * 1024;
/// At most this many loaded comments are kept.
const MAX_COMMENTS: usize = 200;

/// A description as paragraphs, its line breaks kept.
fn description(text: &str, body: &mut String) {
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    for paragraph in lines.split(|line| line.is_empty()) {
        if paragraph.is_empty() {
            continue;
        }
        body.push_str("<p>");
        for (index, line) in paragraph.iter().enumerate() {
            if index > 0 {
                body.push_str("<br>");
            }
            escaped(line, body);
        }
        body.push_str("</p>");
    }
}

/// The text of an element with its line breaks, each line's whitespace
/// collapsed.
fn lines_of(element: ElementRef<'_>) -> String {
    element
        .text()
        .collect::<String>()
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
}

fn rendered(document: &Html) -> Option<Article> {
    let root = document.root_element();
    let metadata = first(root, "ytd-watch-metadata")?;
    let title = text_at(metadata, "h1").or_else(|| super::meta_content(document, "og:title"))?;
    let channel = first(metadata, "#channel-name a, ytd-channel-name a");
    let author = channel.map(text_of).filter(|name| !name.is_empty());
    let mut body = String::new();
    if let Some(channel) = channel
        && let Some(name) = &author
    {
        body.push_str("<p>");
        match channel.value().attr("href") {
            Some(href) => {
                body.push_str("<a href=\"");
                escaped(href, &mut body);
                body.push_str("\">");
                escaped(name, &mut body);
                body.push_str("</a>");
            }
            None => escaped(name, &mut body),
        }
        if let Some(subscribers) = text_at(metadata, "#owner-sub-count") {
            body.push_str(" · ");
            escaped(&subscribers, &mut body);
        }
        body.push_str("</p>");
    }
    if let Some(text) = first(
        metadata,
        "#description-inner, ytd-text-inline-expander, #description",
    ) {
        description(&lines_of(text), &mut body);
    }
    // The counts and dates under the description.
    for panel in root.select(&super::super::selector(
        "ytd-watch-info-panel-renderer, #info-strings",
    )) {
        for line in lines_of(panel).lines().filter(|line| !line.is_empty()) {
            body.push_str("<p>");
            escaped(line, &mut body);
            body.push_str("</p>");
        }
    }
    let comments: Vec<_> = root
        .select(&super::super::selector("ytd-comment-thread-renderer"))
        .filter_map(|thread| {
            let text = text_at(thread, "#content-text")?;
            Some((text_at(thread, "#author-text"), text))
        })
        .take(MAX_COMMENTS)
        .collect();
    if !comments.is_empty() {
        body.push_str("<h2>Comments</h2>");
        for (author, text) in comments {
            body.push_str("<p>");
            if let Some(author) = author {
                body.push_str("<strong>");
                escaped(&author, &mut body);
                body.push_str("</strong>: ");
            }
            escaped(&text, &mut body);
            body.push_str("</p>");
        }
    }
    Some(Article {
        reader: "youtube",
        title,
        author,
        site: Some("YouTube".into()),
        body,
        ..Article::default()
    })
}

/// The `ytInitialPlayerResponse` object a server-sent page assigns.
fn player_response(document: &Html) -> Option<Value> {
    document
        .select(&super::super::selector("script"))
        .find_map(|script| {
            let text: String = script.text().collect();
            let at = text.find("ytInitialPlayerResponse")?;
            let rest = &text[at..];
            let start = rest.find('{')?;
            // The read limit may fall inside a character; end before it.
            let mut end = rest.len().min(start + MAX_SCRIPT);
            while !rest.is_char_boundary(end) {
                end -= 1;
            }
            let rest = &rest[start..end];
            serde_json::Deserializer::from_str(rest)
                .into_iter::<Value>()
                .next()?
                .ok()
        })
}

fn served(document: &Html) -> Option<Article> {
    let response = player_response(document)?;
    let details = response.get("videoDetails")?;
    let text = |value: &Value, key: &str| {
        value
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
    };
    let title = text(details, "title")?;
    let author = text(details, "author");
    let microformat = response.pointer("/microformat/playerMicroformatRenderer");
    let published =
        microformat.and_then(|m| text(m, "publishDate").or_else(|| text(m, "uploadDate")));
    let mut body = String::new();
    if let Some(name) = &author {
        body.push_str("<p>");
        match text(details, "channelId") {
            Some(id) => {
                body.push_str("<a href=\"https://www.youtube.com/channel/");
                escaped(&id, &mut body);
                body.push_str("\">");
                escaped(name, &mut body);
                body.push_str("</a>");
            }
            None => escaped(name, &mut body),
        }
        body.push_str("</p>");
    }
    if let Some(text) = text(details, "shortDescription") {
        description(&text, &mut body);
    }
    if let Some(views) = text(details, "viewCount") {
        body.push_str("<p>Views: ");
        escaped(&views, &mut body);
        body.push_str("</p>");
    }
    let mut byline = vec![author.clone().unwrap_or_default()];
    byline.push(
        published
            .as_deref()
            .map(super::show_time)
            .unwrap_or_default(),
    );
    Some(Article {
        reader: "youtube",
        title,
        author,
        published,
        byline,
        site: Some("YouTube".into()),
        body,
        ..Article::default()
    })
}

pub(super) fn read(document: &Html) -> Option<Article> {
    rendered(document).or_else(|| served(document))
}

#[cfg(test)]
mod tests {
    use crate::formats::html::extract_html;

    #[test]
    fn a_rendered_watch_page_is_read_without_its_buttons_and_guide() {
        let page = r#"<html><head><meta property="og:url" content="https://www.youtube.com/watch?v=abc"><title>Video - YouTube</title></head><body>
            <div id="masthead-container"><nav>YouTube Navigation</nav></div>
            <ytd-watch-flexy><div id="primary"><ytd-watch-metadata>
              <h1><yt-formatted-string>The Video Title</yt-formatted-string></h1>
              <div id="info"><yt-formatted-string id="channel-name"><a href="/channel/UC1">The Channel</a></yt-formatted-string>
                <span id="owner-sub-count">1.2M subscribers</span></div>
              <div id="menu-container"><yt-formatted-string id="like-button">42K</yt-formatted-string><span>Dislike</span><span>Share</span><span>Save</span></div>
              <div id="description-inner"><yt-attributed-string>
                First line of the description.
                Chapters:
                00:00 Introduction

                - A link line
              </yt-attributed-string></div>
            </ytd-watch-metadata>
            <ytd-watch-info-panel-renderer><div>Views: 1,234</div><div>Published: Mar 15, 2024</div></ytd-watch-info-panel-renderer></div>
            <div id="secondary"><ytd-comments><ytd-comment-thread-renderer><div id="comment"><span id="author-text">@fan</span>
              <div id="content-text">Great explanation!</div></div></ytd-comment-thread-renderer></ytd-comments></div></ytd-watch-flexy>
            <div id="masthead-portal"><div id="guide"><a href="/">Home</a><a href="/feed/trending">Trending</a></div></div>
            </body></html>"#;
        let document = extract_html(page, None).unwrap();
        let markdown = &document.markdown;
        assert!(markdown.starts_with("# The Video Title"), "{markdown}");
        assert!(
            markdown.contains("[The Channel](/channel/UC1) · 1.2M subscribers"),
            "{markdown}"
        );
        assert!(
            markdown.contains("First line of the description.  \nChapters:  \n00:00 Introduction"),
            "{markdown}"
        );
        assert!(markdown.contains("Views: 1,234"), "{markdown}");
        assert!(markdown.contains("Published: Mar 15, 2024"), "{markdown}");
        assert!(
            markdown.contains("**@fan**: Great explanation!"),
            "{markdown}"
        );
        for chrome in [
            "Dislike",
            "Share",
            "Save",
            "42K",
            "Trending",
            "Home",
            "YouTube Navigation",
        ] {
            assert!(!markdown.contains(chrome), "{chrome}: {markdown}");
        }
        assert_eq!(document.metadata["author"], "The Channel");
    }

    #[test]
    fn a_served_watch_page_is_read_from_its_player_response() {
        let page = r#"<html><head><title>Video - YouTube</title></head><body><div id="shell">Before you continue</div>
            <script>var ytInitialPlayerResponse = {"videoDetails":{"videoId":"abc","title":"Served Title","author":"Served Channel","channelId":"UC2","shortDescription":"Line one.\nLine two.","viewCount":"42"},"microformat":{"playerMicroformatRenderer":{"publishDate":"2024-03-15T07:00:00-07:00"}}};var meta = {};</script>
            </body></html>"#;
        let document = extract_html(page, Some("https://www.youtube.com/watch?v=abc")).unwrap();
        let markdown = &document.markdown;
        assert!(markdown.starts_with("# Served Title"), "{markdown}");
        assert!(
            markdown.contains("[Served Channel](https://www.youtube.com/channel/UC2)"),
            "{markdown}"
        );
        assert!(markdown.contains("Line one.  \nLine two."), "{markdown}");
        assert!(markdown.contains("Views: 42"), "{markdown}");
        assert!(!markdown.contains("Before you continue"), "{markdown}");
        assert_eq!(document.metadata["published"], "2024-03-15T07:00:00-07:00");
    }

    #[test]
    fn a_player_response_longer_than_the_read_limit_is_cut_at_a_character() {
        // The limit falls inside a two-byte character of an oversized script.
        let filler = "é".repeat(super::MAX_SCRIPT / 2 + 1);
        let page = format!(
            r#"<html><body><p>Page text</p><script>var ytInitialPlayerResponse = {{"ab":"{filler}"}};</script></body></html>"#
        );
        let document = extract_html(&page, Some("https://www.youtube.com/watch?v=abc")).unwrap();
        assert!(document.markdown.contains("Page text"), "{}", document.markdown);
    }
}
