//! How a conversion's warnings read on the terminal.
//!
//! The converters emit one warning per page or image. That is the right grain
//! for a report or `--json`, which keep every one, but a 40-page scan would
//! print 40 near-identical lines and bury the one thing the user can do about
//! it. This layer works only on the text of the warnings it is given, so a
//! changed converter message merely stops being collapsed.

/// A note about how the output was produced rather than something to act on.
/// The terminal shows it with `-v`; reports and `--json` always carry it.
fn is_explanatory(warning: &str) -> bool {
    warning.starts_with("PDF images are placed after their page's text")
        || (warning.starts_with("PDF page ") && warning.contains(": read as plain text because "))
}

/// `(page, reason)` of a per-page "native text was not recovered" warning.
fn page_without_text(warning: &str) -> Option<(usize, &str)> {
    let rest = warning.strip_prefix("PDF page ")?;
    let (page, rest) = rest.split_once(": native text was not recovered (")?;
    let reason = rest.strip_suffix("); OCR is required for this page.")?;
    Some((page.parse().ok()?, reason))
}

/// Page numbers as ranges: `1-3, 5, 7-9`. Past `LIMIT` ranges the rest is
/// counted instead of listed.
fn ranges(pages: &[usize]) -> String {
    const LIMIT: usize = 12;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &page in pages {
        match runs.last_mut() {
            Some((_, end)) if *end + 1 == page => *end = page,
            _ => runs.push((page, page)),
        }
    }
    let text = |&(start, end): &(usize, usize)| match end - start {
        0 => start.to_string(),
        _ => format!("{start}-{end}"),
    };
    let mut shown = runs
        .iter()
        .take(LIMIT)
        .map(text)
        .collect::<Vec<_>>()
        .join(", ");
    if runs.len() > LIMIT {
        let rest: usize = runs[LIMIT..]
            .iter()
            .map(|(start, end)| end - start + 1)
            .sum();
        shown.push_str(&format!(" and {rest} more"));
    }
    shown
}

/// The warnings of one item as the terminal shows them: pages whose text was
/// not recovered merge into one line, at the place of the first, that says
/// which pages and what to do; notes that only explain are left out unless
/// `verbose`.
pub(super) fn present(warnings: &[String], verbose: bool) -> Vec<String> {
    let mut lines = Vec::new();
    let mut pages: Vec<usize> = Vec::new();
    let mut reasons: Vec<&str> = Vec::new();
    let mut merged_at = None;
    for warning in warnings {
        if let Some((page, reason)) = page_without_text(warning) {
            if merged_at.is_none() {
                merged_at = Some(lines.len());
                lines.push(String::new());
            }
            pages.push(page);
            if !reasons.contains(&reason) {
                reasons.push(reason);
            }
        } else if verbose || !is_explanatory(warning) {
            lines.push(warning.clone());
        }
    }
    if let Some(at) = merged_at {
        lines[at] = summary(pages, &reasons);
    }
    lines
}

/// The one line that stands for a document's pages without recovered text.
fn summary(mut pages: Vec<usize>, reasons: &[&str]) -> String {
    pages.sort_unstable();
    pages.dedup();
    let (noun, pronoun) = if pages.len() == 1 {
        ("page", "it")
    } else {
        ("pages", "them")
    };
    format!(
        "PDF {noun} {}: native text was not recovered ({}); OCR is required to read {pronoun}. Run again with --ocr.",
        ranges(&pages),
        reasons.join(", ")
    )
}

/// An error message that embeds the per-page warnings of a document with no
/// readable text at all, with each run of them condensed the same way.
pub(super) fn condense(message: &str) -> String {
    const START: &str = "PDF page ";
    const END: &str = "); OCR is required for this page.";
    // (start, end, page, reason) of every well-formed per-page warning.
    let mut found: Vec<(usize, usize, usize, &str)> = Vec::new();
    let mut from = 0;
    while let Some(offset) = message[from..].find(START) {
        let start = from + offset;
        from = start + START.len();
        let Some(length) = message[start..].find(END) else {
            break;
        };
        let end = start + length + END.len();
        if let Some((page, reason)) = page_without_text(&message[start..end]) {
            found.push((start, end, page, reason));
            from = end;
        }
    }
    let mut out = String::new();
    let mut copied = 0;
    let mut index = 0;
    while index < found.len() {
        // A run: warnings separated by nothing but a space.
        let mut last = index;
        while last + 1 < found.len() && found[last + 1].0 <= found[last].1 + 1 {
            last += 1;
        }
        let run = &found[index..=last];
        out.push_str(&message[copied..run[0].0]);
        let mut reasons: Vec<&str> = Vec::new();
        for &(_, _, _, reason) in run {
            if !reasons.contains(&reason) {
                reasons.push(reason);
            }
        }
        out.push_str(&summary(
            run.iter().map(|entry| entry.2).collect(),
            &reasons,
        ));
        copied = run[run.len() - 1].1;
        index = last + 1;
    }
    out.push_str(&message[copied..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    #[test]
    fn pages_without_text_become_one_line_that_names_the_ranges_and_ocr() {
        let page = |number: usize, reason: &str| {
            format!(
                "PDF page {number}: native text was not recovered ({reason}); OCR is required for this page."
            )
        };
        let mut warnings = vec!["Other warning".to_owned()];
        warnings.extend((1..=3).map(|n| page(n, "scanned")));
        warnings.push(page(5, "image-only"));
        warnings.extend((7..=9).map(|n| page(n, "scanned")));
        warnings.push("Last warning".into());
        assert_eq!(
            present(&warnings, false),
            [
                "Other warning",
                "PDF pages 1-3, 5, 7-9: native text was not recovered (scanned, image-only); OCR is required to read them. Run again with --ocr.",
                "Last warning",
            ]
        );
        assert_eq!(
            present(&[page(4, "scanned")], false),
            [
                "PDF page 4: native text was not recovered (scanned); OCR is required to read it. Run again with --ocr."
            ]
        );
        // A long list of scattered pages stays one short line.
        let scattered: Vec<_> = (1..=80).step_by(2).map(|n| page(n, "scanned")).collect();
        let merged = present(&scattered, false);
        assert_eq!(merged.len(), 1);
        assert!(
            merged[0].contains("1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23 and 28 more"),
            "{}",
            merged[0]
        );
    }

    #[test]
    fn an_error_that_embeds_the_page_warnings_is_condensed_too() {
        let page = |number: usize, reason: &str| {
            format!(
                "PDF page {number}: native text was not recovered ({reason}); OCR is required for this page."
            )
        };
        let message = format!(
            "Native PDF conversion failed: no reliable native text or extractable images; {} {} {}",
            page(1, "empty native text extraction"),
            page(2, "empty native text extraction"),
            page(3, "scanned")
        );
        assert_eq!(
            condense(&message),
            "Native PDF conversion failed: no reliable native text or extractable images; PDF pages 1-3: native text was not recovered (empty native text extraction, scanned); OCR is required to read them. Run again with --ocr."
        );
        // Any other message is returned untouched.
        for same in [
            "Native PDF conversion failed: password required",
            "PDF page 4: unreadable",
            "",
        ] {
            assert_eq!(condense(same), same);
        }
    }

    #[test]
    fn explanatory_notes_wait_for_verbose_and_everything_else_stays() {
        let warnings = lines(&[
            "PDF images are placed after their page's text; their exact position and vector graphics are not reconstructed.",
            "PDF page 5: read as plain text because it looked like a scan but draws no image. Reading order, paragraph breaks and text styling may differ.",
            "PDF images are placed after their page's text and vector graphics are not reconstructed; the page screenshots keep each page's appearance.",
            "Merged table cells are represented by their origin cell with empty covered cells in Markdown.",
        ]);
        assert_eq!(
            present(&warnings, false),
            [
                "Merged table cells are represented by their origin cell with empty covered cells in Markdown."
            ]
        );
        assert_eq!(present(&warnings, true), warnings);
        assert!(present(&[], false).is_empty());
    }
}
