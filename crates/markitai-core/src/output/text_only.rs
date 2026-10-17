//! `--text-only` (`output.text_only`): for image inputs and PDFs, the written
//! document is the recognised text and nothing else. No front matter, no
//! title heading of an image, no picture links, no page-image comments; the
//! text keeps the structure its reader or model gave it (headings, tables).

use crate::ConversionOutput;
use regex::Regex;
use std::sync::LazyLock;

static PICTURE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"!\[(?:[^\]\\]|\\.)*\]\([^)\s]*(?:\s+"[^"]*")?\)"#).expect("picture compiles")
});
static COMMENT_PICTURE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^<!--\s*!\[[^\]]*\]\([^)\s]+\)\s*-->$").expect("comment picture compiles")
});
/// A line that only links an owned file (a multi-page TIFF's original).
static ASSET_LINK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\[(?:[^\]\\]|\\.)*\]\((?:\.markitai/)?assets/[^)\s]+\)$").expect("link compiles")
});

/// The text of `markdown` without pictures and page-image comments, and
/// without the `# title` heading an image document opens with.
pub(crate) fn strip(markdown: &str, title: Option<&str>) -> String {
    let heading = title.map(|title| format!("# {title}"));
    let mut lines: Vec<String> = Vec::new();
    let mut fence: Option<(char, usize)> = None;
    let mut seen_text = false;
    for line in markdown.lines() {
        let trimmed = line.trim();
        let marker = trimmed.chars().next().filter(|c| *c == '`' || *c == '~');
        if let Some(kind) = marker {
            let run = trimmed.chars().take_while(|c| *c == kind).count();
            if run >= 3 {
                match fence {
                    None => fence = Some((kind, run)),
                    Some((open, length)) if open == kind && run >= length => fence = None,
                    _ => {}
                }
                lines.push(line.to_owned());
                seen_text = true;
                continue;
            }
        }
        if fence.is_some() {
            lines.push(line.to_owned());
            continue;
        }
        if trimmed == "<!-- Page images for reference -->"
            || COMMENT_PICTURE.is_match(trimmed)
            || ASSET_LINK.is_match(trimmed)
        {
            continue;
        }
        if !seen_text && !trimmed.is_empty() && heading.as_deref() == Some(trimmed) {
            seen_text = true;
            continue;
        }
        let kept = PICTURE.replace_all(line, "");
        if kept.trim().is_empty() && !trimmed.is_empty() {
            continue;
        }
        if !trimmed.is_empty() {
            seen_text = true;
        }
        lines.push(kept.trim_end().to_owned());
    }
    let mut text = String::new();
    let mut blank = false;
    for line in lines {
        if line.trim().is_empty() {
            blank = !text.is_empty();
            continue;
        }
        if blank {
            text.push('\n');
            blank = false;
        }
        text.push_str(&line);
        text.push('\n');
    }
    text
}

/// Reduce a finished conversion to its text. `title` names the heading an
/// image document opens with. A body the image analysis already wrote as
/// text only is left as it is.
pub(crate) fn apply(result: &mut ConversionOutput, title: Option<&str>) {
    result.markdown = strip(&result.markdown, title);
    if !result.text_only
        && let Some(markdown) = &mut result.llm_markdown
    {
        *markdown = strip(markdown, title);
    }
    // Nothing links a picture any more: no picture is published or described.
    result.images.clear();
    result.text_only = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_image_document_keeps_only_its_text() {
        let base = "# english\n\n![english](.markitai/assets/image.jpg)\n\nMARKITAI OCR\nLOCAL TEXT ONLY\n2026\n";
        assert_eq!(
            strip(base, Some("english")),
            "MARKITAI OCR\nLOCAL TEXT ONLY\n2026\n"
        );
        // Nothing but the picture: an empty document.
        assert_eq!(
            strip(
                "# photo\n\n![photo](.markitai/assets/image.png)\n",
                Some("photo")
            ),
            ""
        );
    }

    #[test]
    fn a_multipage_tiff_loses_its_original_link_and_page_pictures() {
        let base = "# scan\n\n[Original TIFF](.markitai/assets/original.tiff)\n\n<!-- Page number: 1 -->\n\n![Page 1](.markitai/assets/page0001.png)\n\nFirst page.\n\n<!-- Page number: 2 -->\n\n![Page 2](.markitai/assets/page0002.png)\n\nSecond page.\n";
        assert_eq!(
            strip(base, Some("scan")),
            "<!-- Page number: 1 -->\n\nFirst page.\n\n<!-- Page number: 2 -->\n\nSecond page.\n"
        );
    }

    #[test]
    fn a_pdf_keeps_its_structure_without_pictures_or_screenshot_comments() {
        let llm = "# Report\n\nIntro ![chart](.markitai/assets/a.png) text.\n\n<!-- ![Page 1](.markitai/screenshots/r.pdf.page0001.jpg) -->\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```\n![kept in code](x.png)\n```\n\n<!-- Page images for reference -->\n<!-- ![Page 2](.markitai/screenshots/r.pdf.page0002.jpg) -->\n";
        assert_eq!(
            strip(llm, None),
            "# Report\n\nIntro  text.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```\n![kept in code](x.png)\n```\n"
        );
        // A PDF's own first heading is text, not an image title.
        assert_eq!(strip("# Report\n\nBody.\n", None), "# Report\n\nBody.\n");
    }

    #[test]
    fn the_title_heading_goes_only_when_it_opens_the_document() {
        assert_eq!(
            strip("Text first.\n\n# english\n", Some("english")),
            "Text first.\n\n# english\n"
        );
    }
}
