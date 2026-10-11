//! A GitHub issue: its opening post, then its discussion. GitHub's issue
//! viewer (React, tagged with `data-testid`) surrounds every post with an
//! avatar, a heading only screen readers see ("octocat commented on May 26,
//! 2025"), the "opened on … · edited by …" line, badges and edit-history and
//! reaction controls. The reader writes the opening post's body, then each
//! comment under `## Comments` as a block quote headed `**author** ·
//! [date](permalink)`, the format of the Hacker News and X readers and of the
//! reference's comment threads. Timeline events (labels, references, closing)
//! are not comments and are left out.

use super::Attribute;
use super::hacker_news::escaped;
use super::{Announcement, Landmarks, render_clean, safe_url, selector};
use crate::Result;
use scraper::ElementRef;
use serde_json::Map;
use url::Url;

fn first<'a>(element: ElementRef<'a>, query: &str) -> Option<ElementRef<'a>> {
    element.select(&selector(query)).next()
}

fn text_of(element: ElementRef<'_>) -> String {
    element
        .text()
        .flat_map(str::split_whitespace)
        .collect::<Vec<_>>()
        .join(" ")
}

/// The UTC day (`YYYY-MM-DD`) of a `relative-time` stamp, which does not
/// change with the time zone the page was saved in.
fn day(time: ElementRef<'_>) -> Option<String> {
    let stamp = time.value().attribute("datetime")?;
    chrono::DateTime::parse_from_rfc3339(stamp.trim())
        .ok()
        .map(|time| time.naive_utc().date().to_string())
}

/// A post's author and the day and permalink of its date link.
struct Header {
    author: Option<String>,
    day: Option<String>,
    permalink: Option<String>,
}

fn header(post: ElementRef<'_>, author: &str, link: &str, base: Option<&Url>) -> Header {
    let author = first(post, author)
        .map(text_of)
        .filter(|name| !name.is_empty());
    let link = first(post, link);
    let time = link
        .and_then(|link| first(link, "relative-time"))
        .or_else(|| first(post, "relative-time"));
    Header {
        author,
        day: time.and_then(day),
        permalink: link
            .and_then(|link| link.value().attribute("href"))
            .and_then(|href| safe_url(href, base)),
    }
}

/// A post's Markdown body: the first `.markdown-body`, written once (the
/// viewer nests one inside another).
fn body(post: ElementRef<'_>, base: Option<&Url>) -> Result<String> {
    match first(post, ".markdown-body") {
        Some(body) => render_clean(body, base),
        None => Ok(String::new()),
    }
}

/// A comment as block-quote lines: its header, a blank line, its body.
fn comment(post: ElementRef<'_>, base: Option<&Url>) -> Result<Option<Vec<String>>> {
    let text = body(post, base)?;
    if text.is_empty() {
        return Ok(None);
    }
    let header = header(
        post,
        r#"a[data-testid="avatar-link"]"#,
        r##"a[href*="#issuecomment-"]"##,
        base,
    );
    let mut parts = Vec::new();
    if let Some(author) = &header.author {
        parts.push(format!("**{}**", escaped(author)));
    }
    if let Some(day) = header.day {
        parts.push(match header.permalink {
            Some(url) => format!("[{day}]({url})"),
            None => day,
        });
    }
    let mut lines = Vec::new();
    if !parts.is_empty() {
        lines.push(parts.join(" · "));
        lines.push(String::new());
    }
    lines.extend(text.lines().map(str::to_owned));
    Ok(Some(lines))
}

/// The issue of a page that GitHub's issue viewer renders, as Markdown with
/// its author and date.
pub(super) fn issue(page: &Landmarks<'_>, base: Option<&Url>) -> Result<Option<Announcement>> {
    let Some(opening) = page.issue else {
        return Ok(None);
    };
    let Some(viewer) = first(opening, r#"[data-testid="issue-body-viewer"]"#) else {
        return Ok(None);
    };
    let text = body(viewer, base)?;
    if text.is_empty() {
        return Ok(None);
    }
    let mut metadata = Map::new();
    let header = header(
        opening,
        r#"a[data-testid="issue-body-header-author"]"#,
        r#"a[data-testid="issue-body-header-link"]"#,
        base,
    );
    if let Some(author) = header.author {
        metadata.insert("author".into(), author.into());
    }
    if let Some(day) = header.day {
        metadata.insert("published".into(), day.into());
    }
    let mut markdown = text;
    let comments = opening
        .ancestors()
        .filter_map(ElementRef::wrap)
        .find(|parent| parent.value().attribute("data-testid") == Some("issue-viewer-container"))
        .map(|viewer| {
            viewer
                .select(&selector(".react-issue-comment"))
                .filter(|post| !post.ancestors().any(|parent| parent.id() == opening.id()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut quoted = Vec::new();
    for post in comments {
        if let Some(lines) = comment(post, base)? {
            quoted.push(
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
                    .join("\n"),
            );
        }
    }
    if !quoted.is_empty() {
        markdown.push_str("\n\n---\n\n## Comments\n\n");
        markdown.push_str(&quoted.join("\n\n"));
    }
    Ok(Some(Announcement { markdown, metadata }))
}

#[cfg(test)]
mod tests {
    use super::super::extract_html;

    /// A post's header as the viewer writes it, then its body.
    fn post(author: &str, author_id: &str, link: &str, stamp: &str, text: &str) -> String {
        format!(
            r#"<a href="https://github.com/{author}" aria-label="@{author}'s profile"><img data-testid="github-avatar" src="https://avatars.githubusercontent.com/u/1?v=4&amp;size=80"></a>
            <div><h3 class="sr-only">{author} commented <relative-time datetime="{stamp}">on May 26, 2025</relative-time></h3>
            <a data-testid="{author_id}" href="/{author}">{author}</a>
            <span>opened <a {link}><relative-time datetime="{stamp}">on May 26, 2025</relative-time></a></span>
            <span><span>·</span><span>edited by <a href="https://github.com/{author}">{author}</a></span></span>
            <span data-testid="comment-author-association">Contributor</span><button aria-label="Open edits">Edits</button></div>
            <div data-testid="issue-body-viewer"><div data-testid="markdown-body" class="markdown-body"><div class="markdown-body">{text}</div></div></div>
            <div role="toolbar" aria-label="Reactions"><button>👍 2</button></div>"#
        )
    }

    #[test]
    fn an_issue_is_its_opening_post_then_its_comments_without_the_viewer_chrome() {
        let opening = post(
            "ada",
            "issue-body-header-author",
            r##"data-testid="issue-body-header-link" href="https://github.com/o/r/issues/5#issue-1""##,
            "2025-05-25T20:35:48.000Z",
            "<p>The <code>parse</code> call fails.</p>",
        );
        let reply = |author: &str, id: &str, stamp: &str, text: &str| {
            format!(
                r#"<div data-testid="timeline-row-border-{id}"><div class="react-issue-comment">{}</div></div>"#,
                post(
                    author,
                    "avatar-link",
                    &format!(r##"href="https://github.com/o/r/issues/5#issuecomment-{id}""##),
                    stamp,
                    text
                )
            )
        };
        let page = format!(
            r#"<html><head><meta property="og:url" content="https://github.com/o/r/issues/5"></head><body>
            <div data-testid="issue-viewer-container"><h1 data-testid="issue-title">Parse fails <span>#5</span></h1><span data-testid="header-state">Open</span>
            <div data-testid="issue-viewer-issue-container"><h2 class="sr-only">Description</h2>{opening}</div>
            <div data-testid="issue-timeline-container"><h2 class="sr-only">Activity</h2>{}
            <div><a href="/grace">grace</a> added the <span>bug</span> label</div>{}</div></div></body></html>"#,
            reply(
                "grace",
                "11",
                "2025-05-25T22:14:08.000Z",
                "<p>Reproduced *here*.</p>"
            ),
            reply(
                "ada",
                "12",
                "2025-06-02T08:00:00.000Z",
                r#"<p>Fixed in <a href="https://github.com/o/r/pull/6">#6</a>.</p>"#
            ),
        );
        let doc = extract_html(&page, Some("https://github.com/o/r/issues/5")).unwrap();
        assert_eq!(
            doc.markdown,
            "The `parse` call fails.\n\n---\n\n## Comments\n\n\
             > **grace** · [2025-05-25](https://github.com/o/r/issues/5#issuecomment-11)\n>\n> Reproduced \\*here\\*.\n\n\
             > **ada** · [2025-06-02](https://github.com/o/r/issues/5#issuecomment-12)\n>\n> Fixed in [#6](https://github.com/o/r/pull/6)."
        );
        assert_eq!(doc.metadata["author"], "ada");
        assert_eq!(doc.metadata["published"], "2025-05-25");
        // A page saved without its address is read by the viewer's markup.
        let saved = extract_html(&page, None).unwrap();
        assert_eq!(saved.markdown, doc.markdown);
    }
}
