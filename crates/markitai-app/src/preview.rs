//! The Markdown preview every app shows (native/DESIGN.md, sections 3.3 and
//! 15): one Rust function, so the three apps show the same bytes.

use std::path::{Component, Path, PathBuf};

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
/// nothing loaded but `file:` and `data:` images, raw HTML shown as text.
/// Relative images resolve inside `base_dir` (the conversion's folder);
/// every other image is shown as its alt text.
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

/// What the preview loads for an image source: `data:` and `file:` as
/// written, a relative path as a `file://` URL inside `base_dir`, else none.
fn image_url<'a>(source: &CowStr<'a>, base_dir: Option<&Path>) -> Option<CowStr<'a>> {
    let lower = source.trim_start().to_ascii_lowercase();
    if lower.starts_with("data:") || lower.starts_with("file:") {
        return Some(source.clone());
    }
    if has_scheme(source) || source.starts_with("//") || source.starts_with(['/', '\\']) {
        return None;
    }
    let base = base_dir?;
    let path = source.split(['?', '#']).next().unwrap_or_default();
    let mut inside = PathBuf::new();
    for part in percent_decode(path).split(['/', '\\']) {
        match Path::new(part).components().next() {
            None | Some(Component::CurDir) => {}
            Some(Component::Normal(_)) if !part.contains(':') => inside.push(part),
            Some(Component::ParentDir) if inside.pop() => {}
            _ => return None,
        }
    }
    Some(file_url(&base.join(inside)).into())
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
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || b"/-._~:!$&'()*+,;=@".contains(&byte) {
            url.push(byte as char);
        } else {
            url.push_str(&format!("%{byte:02X}"));
        }
    }
    url
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
