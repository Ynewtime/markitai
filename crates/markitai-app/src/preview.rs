//! The Markdown preview every app shows (desktop/DESIGN.md, sections 3.3 and
//! 15): one Rust function, so the three apps show the same bytes.

use std::path::Path;

use pulldown_cmark::{CowStr, Event, Options, Parser, Tag, TagEnd, html};

/// The preview's stylesheet, the same in every app.
pub const PREVIEW_CSS: &str = include_str!("preview.css");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Theme {
    Light,
    Dark,
}

impl Theme {
    fn name(self) -> &'static str {
        match self {
            Theme::Light => "light",
            Theme::Dark => "dark",
        }
    }
}

/// One complete HTML document showing `markdown` in `theme`: no scripts,
/// nothing loaded but `data:` images and files inside `base_dir` (the
/// conversion's folder), raw HTML shown as text. Every other image is shown
/// as its alt text.
pub fn render(markdown: &str, theme: Theme, base_dir: Option<&Path>) -> String {
    let options = Options::ENABLE_TABLES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_YAML_STYLE_METADATA_BLOCKS;
    let mut events = Vec::new();
    let mut parser = Parser::new_ext(markdown, options);
    while let Some(event) = parser.next() {
        match event {
            Event::Html(raw) | Event::InlineHtml(raw) => events.push(Event::Text(raw)),
            Event::Start(Tag::Image {
                link_type,
                dest_url,
                title,
                id,
            }) => match image_url(&dest_url, base_dir) {
                Some(url) => events.push(Event::Start(Tag::Image {
                    link_type,
                    dest_url: url,
                    title,
                    id,
                })),
                None => {
                    let alt = alt_text(&mut parser);
                    let shown = if alt.is_empty() {
                        dest_url.to_string()
                    } else {
                        alt
                    };
                    events.push(Event::Html(
                        format!(
                            "<span class=\"img-remote\" title=\"{}\">{}</span>",
                            escape(&dest_url),
                            escape(&shown)
                        )
                        .into(),
                    ));
                }
            },
            event => events.push(event),
        }
    }
    let mut body = String::with_capacity(markdown.len() * 3 / 2);
    html::push_html(&mut body, events.into_iter());
    let color = theme.name();
    format!(
        "<!doctype html>\n<html data-theme=\"{color}\">\n<head>\n<meta charset=\"utf-8\">\n\
<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; img-src file: data:; style-src 'unsafe-inline'\">\n\
<meta name=\"referrer\" content=\"no-referrer\">\n<meta name=\"color-scheme\" content=\"{color}\">\n\
<style>{PREVIEW_CSS}</style>\n</head>\n<body><main>\n{body}</main></body>\n</html>\n"
    )
}

/// The image's text up to its end, consumed from `parser`.
fn alt_text(parser: &mut Parser) -> String {
    let mut alt = String::new();
    let mut depth = 0;
    for event in parser.by_ref() {
        match event {
            Event::Start(Tag::Image { .. }) => depth += 1,
            Event::End(TagEnd::Image) if depth == 0 => break,
            Event::End(TagEnd::Image) => depth -= 1,
            Event::Text(text) | Event::Code(text) => alt.push_str(&text),
            _ => {}
        }
    }
    alt
}

/// What the preview loads for an image source: `data:` as written, and a
/// `file:` or relative source as the `file://` URL of the file it names,
/// if that file is inside `base_dir`. Anything else loads nothing.
fn image_url<'a>(source: &CowStr<'a>, base_dir: Option<&Path>) -> Option<CowStr<'a>> {
    let trimmed = source.trim_start();
    let scheme = |name: &str| {
        trimmed
            .get(..name.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(name))
    };
    if scheme("data:") {
        return Some(source.clone());
    }
    let base_url = file_url(base_dir?);
    let base_url = base_url.strip_suffix('/').unwrap_or(&base_url);
    let base = segments(Vec::new(), &base_url["file://".len()..]);
    let named = if scheme("file:") {
        segments(Vec::new(), local_path(&trimmed["file:".len()..])?)
    } else if has_scheme(source) || source.starts_with("//") || source.starts_with(['/', '\\']) {
        return None;
    } else {
        segments(
            base.clone(),
            source.split(['?', '#']).next().unwrap_or_default(),
        )
    };
    let rest = named.strip_prefix(base.as_slice())?;
    if rest.is_empty() || rest.iter().any(|segment| segment.contains(':')) {
        return None;
    }
    let mut url = base_url.to_owned();
    for segment in rest {
        url.push('/');
        push_encoded(&mut url, segment);
    }
    Some(url.into())
}

/// The path of a `file:` URL (the text after `file:`) on this computer:
/// no host or `localhost`, query and fragment dropped.
fn local_path(url: &str) -> Option<&str> {
    let url = url.split(['?', '#']).next().unwrap_or_default();
    match url.strip_prefix("//") {
        Some(rest) => {
            let (host, path) = rest.split_at(rest.find('/')?);
            (host.is_empty() || host.eq_ignore_ascii_case("localhost")).then_some(path)
        }
        None => url.starts_with('/').then_some(url),
    }
}

/// `start` followed by the segments of a URL path, decoded, with `.` and
/// `..` resolved as a browser resolves them.
fn segments(mut start: Vec<String>, path: &str) -> Vec<String> {
    for part in percent_decode(path).split(['/', '\\']) {
        match part {
            "" | "." => {}
            ".." => {
                start.pop();
            }
            part => start.push(part.to_owned()),
        }
    }
    start
}

fn has_scheme(url: &str) -> bool {
    let Some(end) = url.find(':') else {
        return false;
    };
    let candidate = &url[..end];
    // A drive letter (C:) is a path, not a scheme.
    candidate.len() > 1
        && candidate.starts_with(|c: char| c.is_ascii_alphabetic())
        && candidate
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
}

/// `file://` URL of an absolute path, percent-encoding what a URL path may
/// not hold. Windows paths become `file:///C:/...`.
pub fn file_url(path: &Path) -> String {
    let text = path.to_string_lossy().replace('\\', "/");
    let mut url = String::from("file://");
    if !text.starts_with('/') {
        url.push('/');
    }
    push_encoded(&mut url, &text);
    url
}

fn push_encoded(url: &mut String, text: &str) {
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || b"/-._~:!$&'()*+,;=@".contains(&byte) {
            url.push(byte as char);
        } else {
            url.push_str(&format!("%{byte:02X}"));
        }
    }
}

pub(crate) fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && let Some(hex) = text.get(i + 1..i + 3)
            && let Ok(byte) = u8::from_str_radix(hex, 16)
        {
            out.push(byte);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '&' => escaped.push_str("&amp;"),
            '"' => escaped.push_str("&quot;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '\'' => escaped.push_str("&#39;"),
            c => escaped.push(c),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(html: &str) -> &str {
        &html[html.find("<main>").unwrap()..html.rfind("</main>").unwrap()]
    }

    #[test]
    fn raw_html_is_shown_as_text() {
        let html = render(
            "<script>alert(1)</script>\n\nhi <img src=x onerror=alert(2)>\n",
            Theme::Light,
            None,
        );
        let body = body(&html);
        assert!(!body.contains("<script"), "{body}");
        assert!(!body.contains("<img src=x"), "{body}");
        assert!(body.contains("&lt;script&gt;"), "{body}");
        assert!(html.contains("default-src 'none'; img-src file: data:;"));
        assert!(!html.contains("<base"));
    }

    #[test]
    fn gfm_renders_and_front_matter_is_left_out() {
        let html = render(
            "---\ntitle: T\n---\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n~~gone~~\n\n- [x] done\n\nNote[^1].\n\n[^1]: Foot.\n",
            Theme::Dark,
            None,
        );
        let body = body(&html);
        assert!(body.contains("<table>"), "{body}");
        assert!(body.contains("<del>gone</del>"), "{body}");
        assert!(body.contains("type=\"checkbox\""), "{body}");
        assert!(body.contains("class=\"footnote-definition\""), "{body}");
        assert!(body.contains("class=\"footnote-reference\""), "{body}");
        assert!(!body.contains("title: T"), "{body}");
        assert!(html.contains("<html data-theme=\"dark\">"));
        assert!(html.contains("content=\"dark\""));
    }

    #[test]
    fn images_load_only_from_inside_the_folder() {
        let base = Path::new("/out/Quarterly Report");
        let html = render(
            "![chart](.markitai/assets/a%20b.png) ![up](../secret.png) ![net](https://example.com/x.png) ![](//cdn.example/y.png) ![abs](/etc/z.png) ![d](data:image/png;base64,AA==) ![js](javascript:alert(1))",
            Theme::Light,
            Some(base),
        );
        let body = body(&html);
        assert!(
            body.contains("src=\"file:///out/Quarterly%20Report/.markitai/assets/a%20b.png\""),
            "{body}"
        );
        assert!(
            body.contains("src=\"data:image/png;base64,AA==\""),
            "{body}"
        );
        for remote in [
            "../secret.png",
            "https://example.com/x.png",
            "//cdn.example/y.png",
            "/etc/z.png",
            "javascript:alert(1)",
        ] {
            assert!(
                body.contains(&format!("title=\"{remote}\"")),
                "{remote}: {body}"
            );
            assert!(
                !body.contains(&format!("src=\"{remote}\"")),
                "{remote}: {body}"
            );
        }
        assert!(body.contains(">net</span>"), "{body}");
        assert!(body.contains(">//cdn.example/y.png</span>"), "{body}");
        let none = render("![a](assets/a.png)", Theme::Light, None);
        let none = &none[none.find("<main>").unwrap()..];
        assert!(none.contains("class=\"img-remote\""));
    }

    fn image(source: &str, base: Option<&Path>) -> String {
        let html = render(&format!("![i]({source})"), Theme::Light, base);
        body(&html).to_owned()
    }

    #[test]
    fn file_images_load_only_from_inside_the_folder() {
        let base = Some(Path::new("/out/Quarterly Report"));
        for (source, loaded) in [
            (
                "file:///out/Quarterly%20Report/.markitai/assets/a%20b.png",
                "file:///out/Quarterly%20Report/.markitai/assets/a%20b.png",
            ),
            (
                "FILE://localhost/out/Quarterly%20Report/./x.png?v=1#top",
                "file:///out/Quarterly%20Report/x.png",
            ),
            (
                "file:/out/Quarterly%20Report/sub/../x.png",
                "file:///out/Quarterly%20Report/x.png",
            ),
            (
                "file:///out/Quarterly%20Report/%e6%8a%a5.png",
                "file:///out/Quarterly%20Report/%E6%8A%A5.png",
            ),
            (
                "../Quarterly%20Report/x.png",
                "file:///out/Quarterly%20Report/x.png",
            ),
        ] {
            let body = image(source, base);
            assert!(
                body.contains(&format!("src=\"{loaded}\"")),
                "{source}: {body}"
            );
        }
        for outside in [
            "file:///etc/hosts.png",
            "file:///out/Quarterly%20Report/../secret.png",
            "file:///out/Quarterly%20Report/%2e%2E/secret.png",
            "file:///out/Quarterly%20Report/..%2Fsecret.png",
            r"file:///out/Quarterly%20Report/..\secret.png",
            "file:///out/Quarterly%20Report%20Old/x.png",
            "file:///out/Quarterly%20Report",
            "file://example.com/out/Quarterly%20Report/x.png",
            "file:x.png",
            "file:///out/Quarterly%20Report/c:x.png",
        ] {
            let body = image(outside, base);
            assert!(
                body.contains(&format!("title=\"{outside}\"")),
                "{outside}: {body}"
            );
            assert!(!body.contains("src="), "{outside}: {body}");
        }
        let body = image("file:///out/Quarterly%20Report/x.png", None);
        assert!(body.contains("class=\"img-remote\""), "{body}");
    }

    #[test]
    fn windows_drive_folders_hold_their_images() {
        let base = Some(Path::new(r"C:\Users\example\Markitai\Report"));
        for (source, loaded) in [
            (
                "file:///C:/Users/example/Markitai/Report/a.png",
                "file:///C:/Users/example/Markitai/Report/a.png",
            ),
            (
                r".markitai\assets\a.png",
                "file:///C:/Users/example/Markitai/Report/.markitai/assets/a.png",
            ),
        ] {
            let body = image(source, base);
            assert!(
                body.contains(&format!("src=\"{loaded}\"")),
                "{source}: {body}"
            );
        }
        for outside in [
            "file:///C:/Users/example/Markitai/secret.png",
            "file:///D:/Users/example/Markitai/Report/a.png",
            "file:///C:/Users/example/Markitai/Report/../../secret.png",
            r"..\\..\\secret.png",
        ] {
            let body = image(outside, base);
            assert!(!body.contains("src="), "{outside}: {body}");
        }
    }

    #[test]
    fn windows_folders_become_file_urls() {
        assert_eq!(
            file_url(Path::new(r"C:\Users\example\Documents\Markitai\报告")),
            "file:///C:/Users/example/Documents/Markitai/%E6%8A%A5%E5%91%8A"
        );
    }

    #[test]
    fn the_same_request_gives_the_same_bytes() {
        let markdown = "# T\n\n| a |\n|:-:|\n| 1 |\n";
        assert_eq!(
            render(markdown, Theme::Light, None),
            render(markdown, Theme::Light, None)
        );
    }
}
