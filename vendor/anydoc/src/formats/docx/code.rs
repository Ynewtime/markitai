//! markitai: code set in a monospaced font.
//!
//! Word has no notion of code: a writer, or an exporter such as macOS
//! `textutil`, sets a snippet in Courier, Consolas or Menlo. A paragraph set
//! entirely in such a font is read as a line of a code block and a run of it
//! inside prose as inline code - unless monospace is what the document is
//! set in (a typewriter-style manuscript), where it says nothing.

use super::styles::{Styles, run_font};
use crate::error::ConvertError;
use crate::package::xml::{Element, ns};

/// Share of a document's text, in characters, above which a monospaced font
/// is the document's typeface rather than a marker of code.
pub const BODY_SHARE: f64 = 0.75;

/// Monospaced families met in documents, matched case-insensitively. A name
/// with the word `Mono` or `Monospace` (`Roboto Mono`, `SF Mono`) also
/// counts; `Monotype Corsiva` does not.
const FAMILIES: &[&str] = &[
    "courier",
    "courier new",
    "consolas",
    "menlo",
    "monaco",
    "lucida console",
    "lucida sans typewriter",
    "inconsolata",
    "source code pro",
    "fira code",
    "cascadia code",
    "hack",
    "anonymous pro",
    "fixedsys",
    "terminal",
];

/// Whether a font family name is one of the monospaced faces code is set in.
/// CJK faces with fixed-width Latin (MS Gothic, NSimSun) are body text, not
/// code, and are not listed.
pub fn is_monospace(font: &str) -> bool {
    let name = font.trim().to_ascii_lowercase();
    FAMILIES.contains(&name.as_str())
        || name
            .split([' ', '-', '_'])
            .any(|word| matches!(word, "mono" | "monospace" | "monospaced"))
}

/// Whether a monospaced font marks code in this body: it sets at most
/// [`BODY_SHARE`] of the text's visible characters (by each run's own font,
/// else its character style's, else its paragraph's).
pub fn sets_code_apart(body: &Element, styles: &Styles) -> Result<bool, ConvertError> {
    let (mut mono, mut total) = (0usize, 0usize);
    for paragraph in body.descendants(ns::W, "p") {
        let style = paragraph
            .find(ns::W, "pPr")
            .and_then(|pr| pr.find(ns::W, "pStyle"))
            .and_then(|e| e.attr(ns::W, "val"))
            .or(styles.default_paragraph);
        let paragraph_font = match style {
            Some(id) => styles.style_font(id)?,
            None => None,
        }
        .or(styles.default_font);
        for run in paragraph.child_elems().filter(|c| c.is(ns::W, "r")) {
            let rpr = run.find(ns::W, "rPr");
            let font = match rpr.and_then(run_font) {
                Some(font) => Some(font),
                None => match rpr
                    .and_then(|rpr| rpr.find(ns::W, "rStyle"))
                    .and_then(|e| e.attr(ns::W, "val"))
                {
                    Some(id) => styles.style_font(id)?,
                    None => None,
                }
                .or(paragraph_font),
            };
            let visible: usize = run
                .child_elems()
                .filter(|c| c.is(ns::W, "t"))
                .map(|t| t.text().chars().filter(|c| !c.is_whitespace()).count())
                .sum();
            total += visible;
            if font.is_some_and(is_monospace) {
                mono += visible;
            }
        }
    }
    Ok(mono as f64 <= BODY_SHARE * total as f64)
}

/// A code block that carries its line numbers (a web page's numbered
/// listing saved as a document), as the code alone. The numbers either come
/// first as a column (`1`, `2`, then the two lines) or alternate with the
/// lines (`1`, the first line, `2`, ...). They count up from the first line
/// and number every line of code; a lone line counts only from 1. `None`
/// when the block does not have that shape.
pub fn without_line_gutter(text: &str) -> Option<String> {
    let lines: Vec<&str> = text.split('\n').collect();
    let number = |line: &str| line.trim().parse::<u64>().ok();
    let start = number(lines[0])?;
    let counts = |i: usize, line: &str| number(line) == start.checked_add(i as u64);
    let half = lines.len() / 2;
    if lines.len() % 2 != 0 || (half < 2 && start != 1) {
        return None;
    }
    let column = lines[..half].iter().enumerate().all(|(i, line)| counts(i, line))
        && !counts(half, lines[half]);
    if column {
        return Some(lines[half..].join("\n"));
    }
    let alternating = lines.iter().step_by(2).enumerate().all(|(i, line)| counts(i, line));
    alternating.then(|| lines.iter().skip(1).step_by(2).copied().collect::<Vec<_>>().join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_faces_are_monospace_and_body_faces_are_not() {
        for font in ["Courier New", "Consolas", "Menlo", " monaco ", "Roboto Mono", "PT-Mono"] {
            assert!(is_monospace(font), "{font}");
        }
        for font in ["Times", "Calibri", "Arial", "MS Gothic", "NSimSun", "Monotype Corsiva"] {
            assert!(!is_monospace(font), "{font}");
        }
    }

    #[test]
    fn numbered_lines_lose_their_numbers_only_when_every_line_is_counted() {
        let stripped = |text| without_line_gutter(text);
        // Alternating with the lines, from any first number.
        assert_eq!(stripped("1\nfn main() {\n2\n}").as_deref(), Some("fn main() {\n}"));
        assert_eq!(stripped("7\na\n8\n    b\n9\nc").as_deref(), Some("a\n    b\nc"));
        // As a column before them.
        assert_eq!(stripped("1\n2\n3\nx\n  y\nz").as_deref(), Some("x\n  y\nz"));
        assert_eq!(stripped("1\necho hello").as_deref(), Some("echo hello"));
        // A skipped number, an odd count, a lone line not counted from 1, a
        // block of numbers or a first line that is no number stay as they are.
        for text in ["1\na\n3\nb", "1\na\n2", "5\na", "1\n2\n3\n4", "x\n1\ny\n2", "1\n2\n3\na"] {
            assert_eq!(stripped(text), None, "{text:?}");
        }
    }
}
