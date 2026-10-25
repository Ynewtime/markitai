//! The one-line description of a release: the first sentence of the lead
//! paragraph in its `RELEASE-NOTES.md` asset (what `release_build.py notes`
//! writes), as plain text.

/// At most this many bytes of the asset are read.
pub(crate) const LIMIT: u64 = 16 * 1024;
/// A description longer than this is cut at a word with an ellipsis.
const WIDTH: usize = 120;

pub(crate) fn lead(markdown: &str) -> Option<String> {
    let mut paragraph = Vec::new();
    for line in markdown.lines() {
        let line = line.trim();
        if line.is_empty() {
            if paragraph.is_empty() {
                continue;
            }
            break;
        }
        // Headings and lists are not the lead; a subsection ends it.
        if line.starts_with('#')
            || line.starts_with("- ")
            || line.starts_with("* ")
            || line.starts_with('>')
            || line.starts_with("```")
        {
            if paragraph.is_empty() {
                continue;
            }
            break;
        }
        paragraph.push(line);
    }
    let text = plain(&paragraph.join(" "));
    let sentence = first_sentence(&text);
    let sentence = sentence.trim();
    (!sentence.is_empty()).then(|| shorten(sentence))
}

/// Markdown links become their text; emphasis and code marks go; control
/// characters (a terminal escape in a crafted asset) are dropped.
fn plain(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text
        .chars()
        .filter(|character| !character.is_control())
        .peekable();
    while let Some(character) = chars.next() {
        match character {
            '[' => {
                let mut label = String::new();
                let mut closed = false;
                for next in chars.by_ref() {
                    if next == ']' {
                        closed = true;
                        break;
                    }
                    label.push(next);
                }
                out.push_str(&label);
                if closed && chars.peek() == Some(&'(') {
                    for next in chars.by_ref() {
                        if next == ')' {
                            break;
                        }
                    }
                }
            }
            '*' | '`' => {}
            character => out.push(character),
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn first_sentence(text: &str) -> &str {
    let bytes = text.as_bytes();
    for (index, character) in text.char_indices() {
        match character {
            '。' | '！' | '？' => return &text[..index + character.len_utf8()],
            '.' | '!' | '?' if bytes.get(index + 1).is_none_or(|next| *next == b' ') => {
                return &text[..=index];
            }
            _ => {}
        }
    }
    text
}

fn shorten(text: &str) -> String {
    if text.chars().count() <= WIDTH {
        return text.to_owned();
    }
    let cut: String = text.chars().take(WIDTH).collect();
    let cut = cut.rsplit_once(' ').map_or(cut.as_str(), |(head, _)| head);
    format!("{}…", cut.trim_end_matches([',', ';', ':']))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lead_is_the_first_sentence_of_the_first_paragraph() {
        let notes = "Adds Kimi Code, a few workbench features and fixes for 2.0.0. Update the way you \
             installed: run the setup command again.\n\n### Added\n\n- **Kimi Code** provider\n";
        assert_eq!(
            lead(notes).as_deref(),
            Some("Adds Kimi Code, a few workbench features and fixes for 2.0.0.")
        );
        // Markup goes, link text stays; a version's dot does not end the sentence.
        assert_eq!(
            lead("## [2.2.0]\n\nFaster **PDF** tables in `serve` and [EPUB](https://x.test/a) tables of contents\n").as_deref(),
            Some("Faster PDF tables in serve and EPUB tables of contents")
        );
        assert_eq!(
            lead("PDF 表格识别更准，新增 EPUB 目录导出。其余修复。").as_deref(),
            Some("PDF 表格识别更准，新增 EPUB 目录导出。")
        );
        // Only sections: no lead.
        assert_eq!(lead("### Fixed\n\n- one\n"), None);
        assert_eq!(lead(""), None);
    }

    #[test]
    fn a_crafted_asset_cannot_reach_the_terminal_with_escapes_or_length() {
        let lead = lead("Fine\u{1b}[31m red\u{7} text.").unwrap();
        assert!(
            lead.starts_with("Fine") && !lead.chars().any(char::is_control),
            "{lead:?}"
        );
        let long = "word ".repeat(80);
        let cut = super::lead(&long).unwrap();
        assert!(
            cut.ends_with('…') && cut.chars().count() <= WIDTH + 1,
            "{cut}"
        );
    }
}
