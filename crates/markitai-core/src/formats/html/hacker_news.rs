//! Hacker News pages: a story or comment with its discussion, or a list of
//! stories. The pages lay everything out in nested tables, so the reply
//! structure of a discussion is only visible in each comment's indent.

use super::Attribute;
use super::{Announcement, Landmarks, render_clean, selector};
use crate::Result;
use scraper::{ElementRef, Html};
use serde_json::{Map, Value};
use url::Url;

const SITE: &str = "https://news.ycombinator.com/";
/// Pixels per reply level in the spacer image of older comment markup.
const INDENT_PIXELS: usize = 40;
const TITLE_EXCERPT: usize = 50;

fn is_hacker_news(page: &Landmarks<'_>, base: Option<&Url>) -> bool {
    base.and_then(Url::host_str) == Some("news.ycombinator.com") || (page.hn_main && page.hn_rows)
}

fn text_of(element: ElementRef<'_>) -> String {
    element
        .text()
        .flat_map(str::split_whitespace)
        .collect::<Vec<_>>()
        .join(" ")
}

fn first<'a>(element: ElementRef<'a>, query: &str) -> Option<ElementRef<'a>> {
    element.select(&selector(query)).next()
}

/// Markdown-escaped plain text.
fn escaped(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    for character in text.chars() {
        if matches!(character, '\\' | '*' | '_' | '[' | ']' | '`') {
            output.push('\\');
        }
        output.push(character);
    }
    output
}

/// The date (`YYYY-MM-DD`) of an age element's timestamp title.
fn date(age: ElementRef<'_>) -> Option<String> {
    let stamp = age.value().attribute("title")?.split_whitespace().next()?;
    let day = stamp.split('T').next()?;
    (day.len() == 10 && day.as_bytes()[4] == b'-' && day.as_bytes()[7] == b'-')
        .then(|| day.to_owned())
}

fn link(site: &Url, href: &str) -> Option<String> {
    site.join(href.trim()).ok().map(String::from)
}

/// A reply level from `td.ind`: its `indent` attribute, or the width of its
/// spacer image in older markup.
fn depth(row: ElementRef<'_>) -> usize {
    let Some(indent) = first(row, "td.ind") else {
        return 0;
    };
    if let Some(level) = indent
        .value()
        .attribute("indent")
        .and_then(|value| value.trim().parse().ok())
    {
        return level;
    }
    first(indent, "img")
        .and_then(|image| image.value().attribute("width"))
        .and_then(|width| width.trim().parse::<usize>().ok())
        .map_or(0, |width| width / INDENT_PIXELS)
}

struct Comment {
    depth: usize,
    lines: Vec<String>,
}

/// A comment's header (`**user** · [date](permalink) · points`) and text,
/// or `None` for a deleted comment without either.
fn comment(container: ElementRef<'_>, site: &Url, linked: bool) -> Result<Option<Vec<String>>> {
    let user = first(container, ".hnuser").map(text_of);
    let age = first(container, ".age");
    let mut header = Vec::new();
    if let Some(user) = &user {
        header.push(format!("**{}**", escaped(user)));
    }
    if let Some(age) = age
        && let Some(day) = date(age)
    {
        let permalink = first(age, "a")
            .and_then(|anchor| anchor.value().attribute("href"))
            .and_then(|href| link(site, href));
        header.push(match permalink.filter(|_| linked) {
            Some(url) => format!("[{day}]({url})"),
            None => day,
        });
    }
    if let Some(score) = first(container, ".score").map(text_of)
        && !score.is_empty()
    {
        header.push(escaped(&score));
    }
    let text = match first(container, ".commtext") {
        Some(text) => render_clean(text, Some(site))?,
        None => String::new(),
    };
    if header.is_empty() && text.is_empty() {
        return Ok(None);
    }
    let mut lines = Vec::new();
    if !header.is_empty() {
        lines.push(header.join(" · "));
    }
    if !text.is_empty() {
        if !lines.is_empty() {
            lines.push(String::new());
        }
        lines.extend(text.lines().map(str::to_owned));
    }
    Ok(Some(lines))
}

/// Comments as blockquotes, a reply one level deeper than its parent; each
/// top-level comment starts a separate quote.
fn discussion(comments: &[Comment]) -> String {
    let quote = |level: usize, line: &str| {
        let prefix = "> ".repeat(level);
        if line.is_empty() {
            prefix.trim_end().to_owned()
        } else {
            format!("{prefix}{line}")
        }
    };
    let mut output: Vec<String> = Vec::new();
    let mut previous: Option<usize> = None;
    for comment in comments {
        let level = comment.depth + 1;
        match previous {
            Some(before) if comment.depth > 0 => output.push(quote(before.min(level), "")),
            Some(_) => output.push(String::new()),
            None => {}
        }
        output.extend(comment.lines.iter().map(|line| quote(level, line)));
        previous = Some(level);
    }
    output.join("\n")
}

fn thread(document: &Html, site: &Url) -> Result<Vec<Comment>> {
    let mut comments = Vec::new();
    let mut root = None;
    for row in document.select(&selector("tr.comtr")) {
        let level = depth(row);
        // Levels are relative to the first comment shown.
        let base = *root.get_or_insert(level);
        if let Some(lines) = comment(row, site, true)? {
            comments.push(Comment {
                depth: level.saturating_sub(base),
                lines,
            });
        }
    }
    // A reply cannot be more than one level below the comment before it.
    let mut allowed = 0;
    for comment in &mut comments {
        comment.depth = comment.depth.min(allowed);
        allowed = comment.depth + 1;
    }
    Ok(comments)
}

fn excerpt(text: &str) -> String {
    let mut excerpt: String = text.chars().take(TITLE_EXCERPT).collect();
    if excerpt.len() < text.len() {
        excerpt.push_str("...");
    }
    excerpt
}

fn item(document: &Html, item: ElementRef<'_>, site: &Url) -> Result<Announcement> {
    let mut metadata = Map::new();
    metadata.insert("site".into(), "Hacker News".into());
    let mut blocks = Vec::new();
    let story = first(item, ".titleline > a");
    if let Some(story) = story {
        let title = text_of(story);
        if !title.is_empty() {
            metadata.insert("title".into(), title.clone().into());
        }
        // A text post links to itself.
        if let Some(url) = story
            .value()
            .attribute("href")
            .filter(|href| !href.trim_start().starts_with("item?"))
            .and_then(|href| link(site, href))
        {
            blocks.push(format!("[{url}]({url})"));
        }
        if let Some(user) = first(item, ".subtext .hnuser, .subline .hnuser").map(text_of) {
            metadata.insert("author".into(), user.into());
        }
        if let Some(day) = first(item, ".subtext .age, .subline .age").and_then(date) {
            metadata.insert("published".into(), day.into());
        }
        // A text post's body, in older markup a comment text of the story.
        if let Some(text) =
            first(document.root_element(), ".toptext").or_else(|| first(item, ".commtext"))
        {
            let text = render_clean(text, Some(site))?;
            if !text.is_empty() {
                blocks.push(text);
            }
        }
    } else if let Some(lines) = comment(item, site, false)? {
        let user = first(item, ".hnuser").map(text_of);
        let text = first(item, ".commtext").map(text_of).unwrap_or_default();
        if let Some(user) = &user {
            metadata.insert("author".into(), user.clone().into());
            let title = if text.is_empty() {
                format!("Comment by {user}")
            } else {
                format!("Comment by {user}: {}", excerpt(&text))
            };
            metadata.insert("title".into(), title.into());
        }
        if let Some(day) = first(item, ".age").and_then(date) {
            metadata.insert("published".into(), day.into());
        }
        if !text.is_empty() {
            metadata.insert("description".into(), excerpt(&text).into());
        }
        blocks.push(lines.join("\n"));
    }
    let comments = thread(document, site)?;
    if !comments.is_empty() {
        if !blocks.is_empty() {
            blocks.push("---".into());
        }
        blocks.push("## Comments".into());
        blocks.push(discussion(&comments));
    }
    Ok(Announcement {
        markdown: blocks.join("\n\n"),
        metadata,
    })
}

/// A story list: each story as a numbered item with its source site, then
/// its points, submitter and comment link.
fn listing(document: &Html, site: &Url) -> Option<Announcement> {
    let mut items = Vec::new();
    for story in document.select(&selector("tr.athing:not(.comtr)")) {
        let Some(title) = first(story, ".titleline > a") else {
            continue;
        };
        let Some(url) = title
            .value()
            .attribute("href")
            .and_then(|href| link(site, href))
        else {
            continue;
        };
        let mut line = format!("[{}]({url})", escaped(&text_of(title)));
        if let Some(source) = first(story, ".sitestr").map(text_of) {
            line.push_str(&format!(" ({})", escaped(&source)));
        }
        let details = story
            .next_siblings()
            .filter_map(ElementRef::wrap)
            .next()
            .and_then(|row| first(row, ".subtext"));
        let mut facts = Vec::new();
        if let Some(details) = details {
            if let Some(score) = first(details, ".score").map(text_of) {
                facts.push(escaped(&score));
            }
            if let Some(user) = first(details, ".hnuser").map(text_of) {
                facts.push(format!("by {}", escaped(&user)));
            }
            if let Some((label, href)) = details.select(&selector("a[href]")).find_map(|anchor| {
                let label = text_of(anchor);
                let href = anchor.value().attribute("href")?;
                (label.ends_with("comments") || label.ends_with("comment") || label == "discuss")
                    .then_some((label, href))
            }) && let Some(url) = link(site, href)
            {
                facts.push(format!("[{}]({url})", escaped(&label)));
            }
        }
        if !facts.is_empty() {
            line.push_str("  \n   ");
            line.push_str(&facts.join(" · "));
        }
        items.push(line);
    }
    if items.is_empty() {
        return None;
    }
    let mut markdown = items
        .iter()
        .enumerate()
        .map(|(index, item)| format!("{}. {item}", index + 1))
        .collect::<Vec<_>>()
        .join("\n");
    if let Some(url) = document
        .select(&selector("a.morelink"))
        .next()
        .and_then(|more| more.value().attribute("href"))
        .and_then(|href| link(site, href))
    {
        markdown.push_str(&format!("\n\n[More]({url})"));
    }
    let mut metadata = Map::new();
    metadata.insert("site".into(), Value::String("Hacker News".into()));
    Some(Announcement { markdown, metadata })
}

pub(super) fn page(
    document: &Html,
    landmarks: &Landmarks<'_>,
    base: Option<&Url>,
) -> Result<Option<Announcement>> {
    if !is_hacker_news(landmarks, base) {
        return Ok(None);
    }
    let site = base
        .filter(|base| base.host_str() == Some("news.ycombinator.com"))
        .cloned()
        .or_else(|| Url::parse(SITE).ok());
    let Some(site) = site else {
        return Ok(None);
    };
    // A discussion page's item: the item table, or in older markup the
    // table holding the story row.
    let discussion = document
        .select(&selector("tr.comtr, table.comment-tree"))
        .next()
        .is_some();
    let fat = document
        .select(&selector("table.fatitem"))
        .next()
        .or_else(|| {
            document
                .select(&selector("tr.athing:not(.comtr)"))
                .next()
                .filter(|_| discussion)
                .and_then(|row| {
                    row.ancestors()
                        .filter_map(ElementRef::wrap)
                        .find(|parent| parent.value().name() == "table")
                })
        });
    if let Some(fat) = fat {
        let announcement = item(document, fat, &site)?;
        return Ok((!announcement.markdown.is_empty()).then_some(announcement));
    }
    Ok(listing(document, &site))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: u32, indent: &str, user: &str, text: &str) -> String {
        format!(
            r#"<tr class="athing comtr" id="{id}"><td><table><tr><td class="ind" {indent}></td><td class="default"><div class="comhead"><a class="hnuser" href="user?id={user}">{user}</a> <span class="age" title="2025-01-15T10:00:00 1736935200"><a href="item?id={id}">2 hours ago</a></span></div><div class="comment"><div class="commtext c00">{text}</div><div class="reply"><a href="reply?id={id}">reply</a></div></div></td></tr></table></td></tr>"#
        )
    }

    fn page(document: &Html, base: Option<&Url>) -> Result<Option<Announcement>> {
        super::page(document, &Landmarks::read(document), base)
    }

    fn convert(html: &str) -> Announcement {
        page(&Html::parse_document(html), None).unwrap().unwrap()
    }

    #[test]
    fn a_story_keeps_its_link_text_and_reply_structure() {
        let comments = [
            row(2, r#"indent="0""#, "one", "First<p>Second paragraph"),
            row(
                3,
                r#"><img src="s.gif" width="40" height="1""#,
                "two",
                "Reply",
            ),
            // A jump of several levels is one level below the comment before.
            row(4, r#"indent="4""#, "three", "Deep"),
            row(5, r#"indent="1""#, "four", "Back"),
            row(
                6,
                r#"><img src="s.gif" width="0" height="1""#,
                "five",
                "Next thread",
            ),
        ]
        .concat();
        let page = convert(&format!(
            r#"<table id="hnmain"><tr><td><table class="fatitem"><tr class="athing" id="1"><td class="title"><span class="titleline"><a href="https://example.com/a">A story</a></span></td></tr><tr><td class="subtext"><span class="score">5 points</span> by <a class="hnuser" href="user?id=poster">poster</a> <span class="age" title="2025-01-14T09:00:00"><a href="item?id=1">1 day ago</a></span></td></tr><tr><td></td><td><div class="toptext">Story text</div></td></tr></table><table class="comment-tree">{comments}</table></td></tr></table>"#
        ));
        assert_eq!(
            page.markdown,
            "[https://example.com/a](https://example.com/a)\n\nStory text\n\n---\n\n## Comments\n\n\
             > **one** · [2025-01-15](https://news.ycombinator.com/item?id=2)\n>\n> First\n>\n> Second paragraph\n>\n\
             > > **two** · [2025-01-15](https://news.ycombinator.com/item?id=3)\n> >\n> > Reply\n> >\n\
             > > > **three** · [2025-01-15](https://news.ycombinator.com/item?id=4)\n> > >\n> > > Deep\n> >\n\
             > > **four** · [2025-01-15](https://news.ycombinator.com/item?id=5)\n> >\n> > Back\n\n\
             > **five** · [2025-01-15](https://news.ycombinator.com/item?id=6)\n>\n> Next thread"
        );
        assert_eq!(page.metadata["title"], "A story");
        assert_eq!(page.metadata["author"], "poster");
        assert_eq!(page.metadata["published"], "2025-01-14");
        assert_eq!(page.metadata["site"], "Hacker News");
    }

    #[test]
    fn a_comment_page_is_titled_by_its_author_and_text() {
        let text = "A comment long enough to be cut in its title, well past fifty characters.";
        let page = convert(&format!(
            r#"<table id="hnmain"><tr><td><table class="fatitem"><tr class="athing" id="9"><td class="default"><span class="comhead"><a class="hnuser" href="user?id=writer">writer</a> <span class="age" title="2025-06-15T12:00:00 1750000000"><a href="item?id=9">1 day ago</a></span></span><div class="comment"><div class="commtext c00">{text}</div></div></td></tr></table><table class="comment-tree"></table></td></tr></table>"#
        ));
        assert_eq!(page.markdown, format!("**writer** · 2025-06-15\n\n{text}"));
        assert_eq!(
            page.metadata["title"],
            "Comment by writer: A comment long enough to be cut in its title, well..."
        );
    }

    #[test]
    fn a_text_post_has_no_self_link_and_reply_levels_start_at_the_first_shown() {
        let replies = [
            row(2, r#"indent="2""#, "one", "Top"),
            row(3, r#"indent="3""#, "two", "Reply"),
            row(4, r#"indent="2""#, "three", "Sibling"),
        ]
        .concat();
        let page = convert(&format!(
            r#"<table id="hnmain"><tr><td><table class="fatitem"><tr class="athing" id="1"><td class="title"><span class="titleline"><a href="item?id=1">Ask HN: Why?</a></span></td></tr><tr><td></td><td><div class="toptext">Question</div></td></tr></table><table class="comment-tree">{replies}</table></td></tr></table>"#
        ));
        assert!(
            page.markdown
                .starts_with("Question\n\n---\n\n## Comments\n\n> **one**"),
            "{}",
            page.markdown
        );
        assert!(
            page.markdown.contains("> > Reply\n\n> **three**"),
            "{}",
            page.markdown
        );
    }

    #[test]
    fn older_markup_without_an_item_table_is_a_discussion() {
        let comments = row(2, r#"indent="0""#, "one", "Reply");
        let page = convert(&format!(
            r#"<table id="hnmain"><tr><td><table><tr class="athing" id="1"><td class="title"><span class="titleline"><a href="item?id=1">Ask HN: How?</a></span></td></tr><tr><td class="subtext"><span class="subline"><span class="score">3 points</span> by <a class="hnuser" href="user?id=asker">asker</a></span></td></tr><tr><td><div class="comment"><span class="commtext c00">Question text</span></div></td></tr></table></td></tr><tr><td><table class="comment-tree">{comments}</table></td></tr></table>"#
        ));
        assert!(
            page.markdown
                .starts_with("Question text\n\n---\n\n## Comments\n\n> **one**"),
            "{}",
            page.markdown
        );
        assert_eq!(page.markdown.matches("Question text").count(), 1);
        assert_eq!(page.metadata["author"], "asker");
    }

    #[test]
    fn a_story_list_is_numbered_with_its_details() {
        let page = convert(
            r#"<table id="hnmain"><tr><td><table><tr class="athing" id="1"><td class="title"><span class="rank">1.</span></td><td class="votelinks"><a href="vote?id=1"><div class="votearrow"></div></a></td><td class="title"><span class="titleline"><a href="https://example.com/x">X_Y</a><span class="sitebit"> (<a href="from?site=example.com"><span class="sitestr">example.com</span></a>)</span></span></td></tr><tr><td colspan="2"></td><td class="subtext"><span class="score">9 points</span> by <a class="hnuser" href="user?id=a">a</a> | <a href="hide?id=1">hide</a> | <a href="item?id=1">3&nbsp;comments</a></td></tr><tr class="athing" id="2"><td class="title"><span class="titleline"><a href="item?id=2">Ask HN: Z</a></span></td></tr><tr><td class="subtext"><a href="item?id=2">discuss</a></td></tr></table><a class="morelink" href="news?p=2">More</a></td></tr></table>"#,
        );
        assert_eq!(
            page.markdown,
            "1. [X\\_Y](https://example.com/x) (example.com)  \n   9 points · by a · [3 comments](https://news.ycombinator.com/item?id=1)\n\
             2. [Ask HN: Z](https://news.ycombinator.com/item?id=2)  \n   [discuss](https://news.ycombinator.com/item?id=2)\n\n\
             [More](https://news.ycombinator.com/news?p=2)"
        );
    }

    #[test]
    fn other_pages_are_not_read_as_hacker_news() {
        let document = Html::parse_document(
            r#"<table><tr class="athing"><td><span class="titleline"><a href="/a">A</a></span></td></tr></table>"#,
        );
        assert!(page(&document, None).unwrap().is_none());
        let site = Url::parse("https://news.ycombinator.com/user?id=a").unwrap();
        // On the site itself the story rows are enough.
        assert!(page(&document, Some(&site)).unwrap().is_some());
        let profile = Html::parse_document("<table><tr><td>user: a</td></tr></table>");
        assert!(page(&profile, Some(&site)).unwrap().is_none());
    }
}
