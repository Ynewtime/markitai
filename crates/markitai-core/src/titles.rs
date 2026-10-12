//! What may be a document's `title`: plain text that names something.
//!
//! A title comes from a converter's metadata (a PDF's Info title, an EPUB's
//! `dc:title`, a page's `<title>`), the most prominent text of a PDF's first
//! pages, a heading of the Markdown, or the file name. Whatever the source,
//! the front matter gets plain text ([`clean_title`]: no HTML, no Markdown
//! emphasis, no private-use glyphs a PDF font maps its bullets to) and only a
//! candidate that names the document ([`meaningful_title`]): not an
//! application's default name, not a number (`20240315000127`, `2025`), not
//! a page number. A candidate that fails is passed over for the next source,
//! down to the file name. These are Python markitai 1.2.1's rules
//! (`utils/titles.py`), so both implementations title a document alike.

use regex::Regex;
use std::sync::LazyLock;
use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

fn pattern(source: &str) -> Regex {
    Regex::new(source).expect("static title pattern")
}

static TAG: LazyLock<Regex> =
    LazyLock::new(|| pattern(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>"));
static BREAK: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?i)<(?:br|/p|/div|/li)\b[^<>]*>"));
// A footnote mark set as superscript: `Outlook<sup>1</sup>`, `Data<sup>*,†</sup>`.
static MARK: LazyLock<Regex> = LazyLock::new(|| pattern(r"(?i)<sup>\s*[\d*†‡§¶,\s]{1,8}</sup>"));
static LINK: LazyLock<Regex> = LazyLock::new(|| pattern(r"!?\[([^\]]*)\]\([^)]*\)"));
static FOOTNOTE: LazyLock<Regex> = LazyLock::new(|| pattern(r"\[\^[^\]]*\]"));
static HEADING_MARKS: LazyLock<Regex> = LazyLock::new(|| pattern(r"^#{1,6}\s+"));
static ESCAPE: LazyLock<Regex> = LazyLock::new(|| pattern(r"\\([\\`*_{}\[\]()#+\-.!|<>~])"));
static EMPHASIS: LazyLock<Regex> = LazyLock::new(|| pattern(r"\*\*(.+?)\*\*|__(.+?)__|~~(.+?)~~"));
// `Page 3`, `第 3 页`, `3 / 12`, `- 3 -`, `p. 7`, `Seite 2`.
static PAGE_NUMBER: LazyLock<Regex> = LazyLock::new(|| {
    pattern(concat!(
        r"(?i)^(?:(?:page|p\.|pg\.?|seite|página|pagina)\s*\d+(?:\s*(?:of|/)\s*\d+)?",
        r"|第\s*\d+\s*[页頁]",
        r"|[-–—]?\s*\d+\s*(?:/\s*\d+)?\s*[-–—]?)$",
    ))
});

const OFFICE_PREFIXES: [&str; 3] = [
    "microsoft word - ",
    "microsoft powerpoint - ",
    "microsoft excel - ",
];

/// Not a control, format, private-use, surrogate or unassigned code point,
/// nor the replacement character. A PDF font maps its bullets and icons to
/// the private-use area (`U+F043`): text extraction keeps them, a reader
/// sees a symbol.
fn visible(ch: char) -> bool {
    ch != '\u{fffd}' && ch.general_category_group() != GeneralCategoryGroup::Other
}

fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// HTML character references decoded as in text (`&amp;`, `&#x4E2D;`).
pub(crate) fn unescape_html(text: &str) -> String {
    if !text.contains('&') {
        return text.to_owned();
    }
    let mut output = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        output.push_str(&rest[..at]);
        rest = &rest[at..];
        match crate::output_profiles::html_entity(rest) {
            Some((consumed, chars)) => {
                output.extend(chars.into_iter().filter(|ch| *ch != '\0'));
                rest = &rest[consumed..];
            }
            None => {
                output.push('&');
                rest = &rest[1..];
            }
        }
    }
    output.push_str(rest);
    output
}

/// `text` as a plain one-line title: no markup, no invisible characters.
///
/// Drops every HTML tag (`<sup>`, `<mark>`, `<br>` …) and decodes entities,
/// keeps a link's or an image's text, drops footnote references and marks
/// (`[^1]`, `<sup>1</sup>`), heading marks, Markdown emphasis and backslash
/// escapes, and collapses whitespace.
pub(crate) fn clean_title(text: &str) -> String {
    let text: String = text
        .chars()
        .map(|ch| {
            if visible(ch) || matches!(ch, '\t' | '\n' | '\r') {
                ch
            } else {
                ' '
            }
        })
        .collect();
    let text = collapse(&text);
    let text = HEADING_MARKS.replace(&text, "");
    let text = LINK.replace_all(&text, "$1");
    let text = FOOTNOTE.replace_all(&text, "");
    let text = MARK.replace_all(&text, "");
    let text = BREAK.replace_all(&text, " ");
    let text = TAG.replace_all(&text, "");
    // Entities can name invisible characters too (`&#xf043;`).
    let text: String = unescape_html(&text)
        .chars()
        .map(|ch| if visible(ch) { ch } else { ' ' })
        .collect();
    let text = EMPHASIS.replace_all(&text, "$1$2$3");
    let text = ESCAPE.replace_all(&text, "$1");
    let mut text = collapse(&text);
    loop {
        let stripped = ["**", "__", "~~", "`", "*", "_"]
            .into_iter()
            .find_map(|wrapper| {
                (text.len() > wrapper.len() * 2)
                    .then(|| text.strip_prefix(wrapper)?.strip_suffix(wrapper))
                    .flatten()
                    .map(|inner| inner.trim().to_owned())
            });
        match stripped {
            Some(inner) => text = inner,
            None => break text,
        }
    }
}

/// Whether a title is only the name an application gives a document nobody
/// named: an Office print driver's prefix (`Microsoft Word - report.doc`), a
/// file name with a document or image extension, a single lowercase slug
/// (`code`, `layout-table`: a browser printing a page without `<title>`), or
/// a default name (see [`default_name`]). Another title with an upper-case
/// letter or a space and no extension is kept.
pub(crate) fn placeholder_title(title: &str) -> bool {
    const EXTENSIONS: &[&str] = &[
        "htm", "html", "xhtml", "mht", "mhtml", "pdf", "doc", "docx", "docm", "dot", "dotx", "odt",
        "ott", "rtf", "txt", "md", "tex", "ppt", "pptx", "pps", "ppsx", "odp", "key", "xls",
        "xlsx", "ods", "csv", "pages", "numbers", "indd", "ps", "eps", "dvi", "png", "jpg", "jpeg",
        "gif", "tif", "tiff", "bmp", "svg",
    ];
    let lower = title.trim().to_lowercase();
    if OFFICE_PREFIXES
        .iter()
        .any(|prefix| lower.starts_with(prefix))
    {
        return true;
    }
    if let Some((stem, extension)) = lower.rsplit_once('.')
        && !stem.trim().is_empty()
        && EXTENSIONS.contains(&extension)
    {
        return true;
    }
    default_name(&lower)
        || title.chars().any(|c| c.is_ascii_lowercase())
            && title
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '-' | '_'))
}

/// Whether `lower` (a lower-cased title) is a name an application gives a
/// new document, with a number or separators after it: `Untitled`,
/// `Untitled Document`, `Untitled-2`, `PowerPoint Presentation`, `Title`, and
/// in a few languages (`Sans titre`, `无标题`); or, numbered only, Word's
/// `Document1`, PowerPoint's `Presentation1`, Excel's `Book1` and their
/// Chinese names (`文档1`, `演示文稿1`, `工作簿1`).
fn default_name(lower: &str) -> bool {
    const NAMES: &[&str] = &[
        "untitled",
        "untitled document",
        "untitled presentation",
        "untitled spreadsheet",
        "untitled form",
        "untitled design",
        "untitled drawing",
        "title",
        "document",
        "presentation",
        "new document",
        "word document",
        "microsoft word document",
        "new microsoft word document",
        "powerpoint presentation",
        "microsoft powerpoint presentation",
        "sans titre",
        "unbenannt",
        "sin título",
        "sem título",
        "senza titolo",
        "无标题",
        "未命名",
        "無題",
        "無標題",
    ];
    const NUMBERED: &[&str] = &[
        "document",
        "presentation",
        "book",
        "workbook",
        "drawing",
        "publication",
        "doc",
        "文档",
        "演示文稿",
        "工作簿",
    ];
    let stem = lower
        .trim_end_matches(|c: char| c.is_ascii_digit() || matches!(c, ' ' | '-' | '_' | '(' | ')'));
    let numbered = lower[stem.len()..].chars().any(|c| c.is_ascii_digit());
    NAMES.contains(&stem) || numbered && NUMBERED.contains(&stem)
}

/// Whether `title` (already [`clean_title`]d) names the document: two
/// letters at least (`2025`, `—`, `1.2.3` are no names), digits at most 70 %
/// of its letters and digits (`20240315000127E` is an id; `2025年报` and
/// `Report 2024` are names), not a page number, not an application's
/// default name ([`placeholder_title`]; its file-name and slug rules only
/// for a `metadata` title).
pub(crate) fn meaningful_title(title: &str, metadata: bool) -> bool {
    if title.is_empty() {
        return false;
    }
    let letters = title.chars().filter(|c| c.is_alphabetic()).count();
    let digits = title.chars().filter(|c| c.is_numeric()).count();
    if letters < 2 || digits as f64 > 0.7 * (letters + digits) as f64 {
        return false;
    }
    if PAGE_NUMBER.is_match(title) {
        return false;
    }
    if metadata {
        return !placeholder_title(title);
    }
    let lower = title.to_lowercase();
    !default_name(lower.trim())
        && !OFFICE_PREFIXES
            .iter()
            .any(|prefix| lower.starts_with(prefix))
}

/// `text` cleaned when it is a meaningful title.
pub(crate) fn usable_title(text: &str, metadata: bool) -> Option<String> {
    let title = clean_title(text);
    meaningful_title(&title, metadata).then_some(title)
}

/// The first Markdown heading, of any level, that names the document: a
/// line starting with `# ` or `##` (the historical permissive rule, which
/// also reads a heading in a code block), whitespace before the first
/// nonempty line ignored. A heading that is no name is passed over.
pub(crate) fn heading_title(markdown: &str) -> Option<String> {
    markdown.trim_start().lines().find_map(|line| {
        (line.starts_with("# ") || line.starts_with("##"))
            .then(|| usable_title(line.trim_start_matches('#'), false))
            .flatten()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn titles_are_plain_text() {
        for (raw, clean) in [
            // The owner's LLM-path title in Python: glyphs a PDF font maps to
            // the private-use area, then a superscript tag.
            (
                "\u{f043} \u{f020}<sup>申请人享有的主要权利</sup>",
                "申请人享有的主要权利",
            ),
            ("**<mark>办理须知</mark>**", "办理须知"),
            (
                "Regional Climate Outlook<sup>1</sup>",
                "Regional Climate Outlook",
            ),
            ("<sup>申请人</sup>权利", "申请人权利"),
            ("A&amp;B<br/>Report", "A&B Report"),
            ("[Annual Report](https://x.test) 2024", "Annual Report 2024"),
            ("Results[^1]", "Results"),
            ("## **Bold** and _soft_", "Bold and _soft_"),
            (r"C\# Guide", "C# Guide"),
            (r"ZZZ\_Sheet\_1", "ZZZ_Sheet_1"),
            ("  spaced\n  out ", "spaced out"),
            ("&#xf043;Notice", "Notice"),
            ("`code`", "code"),
        ] {
            assert_eq!(clean_title(raw), clean, "{raw:?}");
        }
    }

    #[test]
    fn names_and_no_names() {
        for title in [
            "电子登记单",
            "2025年报",
            "Annual Report 2024",
            "Book",
            "Untitled Love Song",
            "Document Management Policy",
            "第一章 概述",
        ] {
            assert!(meaningful_title(title, false), "{title}");
            assert!(meaningful_title(title, true), "{title}");
        }
        for title in [
            "2025",
            "20240315000127",
            "20240315000127E",
            "1.2.3",
            "—",
            "Page 3",
            "第 3 页",
            "3 / 12",
            "Untitled",
            "Untitled-2",
            "Document1",
            "无标题",
            "文档1",
            "Microsoft Word - report.docx",
            "",
        ] {
            assert!(!meaningful_title(title, false), "{title}");
            assert!(!meaningful_title(title, true), "{title}");
        }
        // An application's name for the file: only a metadata title.
        for title in ["report_v3.docx", "scan0001.pdf", "layout-table", "code"] {
            assert!(placeholder_title(title), "{title}");
            assert!(!meaningful_title(title, true), "{title}");
        }
        assert_eq!(usable_title("<sup></sup>20260521", false), None);
    }

    #[test]
    fn a_heading_that_is_no_name_is_passed_over() {
        let markdown = "<!-- p1 -->\n\n## 2025\n\n### Page 3\n\n## Real Title\n\n# Later";
        assert_eq!(heading_title(markdown).as_deref(), Some("Real Title"));
        assert_eq!(
            heading_title("text\n\n#   <sup>申请人享有的主要权利</sup>\n").as_deref(),
            Some("申请人享有的主要权利")
        );
        assert_eq!(heading_title("plain text\n\n## 12345\n"), None);
        assert_eq!(
            heading_title("#hashtag\n\n# **Bold**").as_deref(),
            Some("Bold")
        );
    }
}
