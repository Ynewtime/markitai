//! Tables continued across page breaks, joined in the assembled Markdown.
//!
//! A table that a page break cuts is read page by page, so each page's part
//! comes out as a table of its own. The part opening the next page continues
//! the table when layout geometry says so (the same column borders, see the
//! layout reader), or when it repeats the header row of the table it
//! follows, as a printed table repeats its header on every page. Only
//! running headers and footers may stand between the two parts; a page
//! recognized by OCR ends the table.
use super::PdfPage;
use std::ops::Range;

/// The cells of a Markdown table row (`|a|b\|c|`), split at unescaped pipes.
fn cells(line: &str) -> Option<Vec<&str>> {
    let inner = line.trim().strip_prefix('|')?.strip_suffix('|')?;
    let mut cells = Vec::new();
    let mut start = 0;
    let mut escaped = false;
    for (at, ch) in inner.char_indices() {
        if ch == '|' && !escaped {
            cells.push(inner[start..at].trim());
            start = at + 1;
        }
        escaped = ch == '\\' && !escaped;
    }
    cells.push(inner[start..].trim());
    Some(cells)
}

/// The rows of a table block (header, separator, body rows), when the block
/// is one: two lines or more, every row with the header's column count.
fn rows(block: &str) -> Option<Vec<&str>> {
    let rows: Vec<&str> = block.lines().collect();
    let columns = cells(rows.first()?)?.len();
    let separator = cells(rows.get(1)?)?
        .iter()
        .all(|cell| cell.len() >= 3 && cell.bytes().all(|b| b == b'-'));
    (separator
        && rows
            .iter()
            .all(|row| cells(row).is_some_and(|cells| cells.len() == columns)))
    .then_some(rows)
}

/// The byte ranges of `text`'s blocks (text between blank lines), without
/// surrounding whitespace.
fn blocks(text: &str) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut push = |start: usize, end: usize| {
        let part = &text[start..end];
        let lead = part.len() - part.trim_start().len();
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            ranges.push(start + lead..start + lead + trimmed.len());
        }
    };
    let mut start = 0;
    for (at, _) in text.match_indices("\n\n") {
        push(start, at);
        start = at + 2;
    }
    push(start, text.len());
    ranges
}

/// Running headers and footers: a one-line block (not a table row) that
/// opens, or closes, two pages or more and half of the pages with two
/// blocks or more, its digits (dates, folios) set aside.
struct Furniture {
    /// The key of each block that opens (`h…`) or closes (`f…`) a page,
    /// sorted, so a key's count is the length of its run.
    keys: Vec<String>,
    pages: usize,
}

impl Furniture {
    /// The block's text with each run of digits (a folio of any length, a
    /// date) as one `0`, after its end's letter.
    fn key(end: char, block: &str) -> Option<String> {
        if block.contains('\n') || block.len() > 200 || block.starts_with('|') {
            return None;
        }
        let mut key = String::with_capacity(block.len() + 1);
        key.push(end);
        let mut digits = false;
        for c in block.chars() {
            if !(c.is_ascii_digit() && digits) {
                key.push(if c.is_ascii_digit() { '0' } else { c });
            }
            digits = c.is_ascii_digit();
        }
        Some(key)
    }

    fn new(pages: &[PdfPage]) -> Self {
        let mut keys = Vec::new();
        let mut counted = 0;
        for page in pages.iter().filter(|page| native(page)) {
            let ranges = blocks(&page.markdown);
            let (Some(first), Some(last)) = (ranges.first(), ranges.last()) else {
                continue;
            };
            if ranges.len() < 2 {
                continue;
            }
            counted += 1;
            for (end, range) in [('h', first), ('f', last)] {
                if let Some(key) = Self::key(end, &page.markdown[range.clone()]) {
                    keys.push(key);
                }
            }
        }
        keys.sort_unstable();
        Self {
            keys,
            pages: counted,
        }
    }

    fn is(&self, end: char, block: &str) -> bool {
        let Some(key) = Self::key(end, block) else {
            return false;
        };
        let count =
            self.keys.partition_point(|k| *k <= key) - self.keys.partition_point(|k| *k < key);
        count >= 2 && count * 2 >= self.pages
    }

    fn head(&self, block: &str) -> bool {
        self.is('h', block)
    }

    fn foot(&self, block: &str) -> bool {
        self.is('f', block)
    }
}

/// Whether a page holds native text that tables can continue through.
fn native(page: &PdfPage) -> bool {
    !page.needs_ocr && !page.ocr_completed && !page.markdown.trim().is_empty()
}

/// The table ending a page's Markdown, which the next page may continue.
struct Open {
    /// The page holding it.
    page: usize,
    /// Where it ends, before any running footer.
    end: usize,
    /// Its header row's cells.
    header: Vec<String>,
    /// Whether running footers follow it.
    footed: bool,
}

/// The table ending `text` (on page `page`), before any running footer.
fn ending_table(page: usize, text: &str, furniture: &Furniture) -> Option<Open> {
    let ranges = blocks(text);
    let footers = ranges
        .iter()
        .rev()
        .take_while(|range| furniture.foot(&text[(*range).clone()]))
        .count();
    let range = ranges.get(ranges.len().checked_sub(footers + 1)?)?.clone();
    let header = cells(rows(&text[range.clone()])?[0])?
        .into_iter()
        .map(str::to_owned)
        .collect();
    Some(Open {
        page,
        end: range.end,
        header,
        footed: footers > 0,
    })
}

/// Appends to the open table, in `owner`, the rows of the table opening
/// `next` when that continues it, and removes them from `next`. A repeated
/// header row is dropped; otherwise the continuation's first row is a body
/// row. `geometric` (layout found the same column borders) counts only when
/// no running header or footer stands between the two. Returns whether
/// `next` holds nothing but running headers and footers afterwards.
fn join(
    open: &mut Open,
    owner: &mut String,
    next: &mut String,
    geometric: bool,
    furniture: &Furniture,
) -> Option<bool> {
    let ranges = blocks(next);
    let headers = ranges
        .iter()
        .take_while(|range| furniture.head(&next[(*range).clone()]))
        .count();
    let opening = ranges.get(headers)?.clone();
    let table = rows(&next[opening.clone()])?;
    let first = cells(table[0])?;
    let header = &open.header;
    let repeated = header.len() == first.len()
        && header.iter().zip(&first).all(|(a, b)| a == b)
        && header.iter().any(|cell| !cell.is_empty());
    let geometric = geometric && !open.footed && headers == 0;
    if header.len() != first.len() || header.len() < 2 || !(repeated || geometric) {
        return None;
    }
    let mut appended = String::new();
    for row in std::iter::once(table[0])
        .filter(|_| !repeated)
        .chain(table[2..].iter().copied())
    {
        appended.push('\n');
        appended.push_str(row);
    }
    // Only a running footer, if any, follows the open table.
    owner.insert_str(open.end, &appended);
    open.end += appended.len();
    let removed = opening.start..ranges.get(headers + 1).map_or(next.len(), |r| r.start);
    next.replace_range(removed, "");
    *next = next.trim().to_owned();
    Some(blocks(next).iter().all(|range| {
        let block = &next[range.clone()];
        furniture.head(block) || furniture.foot(block)
    }))
}

/// Joins tables continued across page breaks: the rows of a page's opening
/// table move to the table ending the previous page, or the page before it
/// when the pages between gave all their table rows to it. The open table's
/// end is kept, so a table over many pages is not read again for each.
pub(super) fn join_tables(pages: &mut [PdfPage]) {
    let furniture = Furniture::new(pages);
    let mut open: Option<Open> = None;
    for index in 0..pages.len() {
        if !native(&pages[index]) {
            open = None;
            continue;
        }
        if let Some(table) = open.as_mut() {
            let (before, after) = pages.split_at_mut(index);
            let page = &mut after[0];
            let geometric = page.continues_table;
            if join(
                table,
                &mut before[table.page].markdown,
                &mut page.markdown,
                geometric,
                &furniture,
            ) == Some(true)
            {
                continue;
            }
        }
        open = ending_table(index, &pages[index].markdown, &furniture);
    }
}

#[cfg(test)]
mod tests {
    use super::super::{PdfPage, PdfPages};
    use crate::Document;
    use std::collections::BTreeMap;

    fn page(number: usize, markdown: &str, continues_table: bool) -> PdfPage {
        PdfPage {
            number,
            markdown: markdown.into(),
            needs_ocr: false,
            ocr_reason: None,
            asset_names: Vec::new(),
            asset_ocr: BTreeMap::new(),
            screenshot_name: None,
            visibility_suspect: false,
            ocr_completed: false,
            omitted_text: None,
            warning_index: 0,
            continues_table,
            ocr_layer: None,
        }
    }

    fn assemble(pages: Vec<PdfPage>) -> Document {
        PdfPages {
            pages,
            document: Document::default(),
            comments: Default::default(),
            ocr_layer_producer: None,
            unverified_visibility: Default::default(),
        }
        .finish()
        .unwrap()
    }

    const HEAD: &str = "|Name|Count|\n|---|---|";

    #[test]
    fn a_repeated_header_row_joins_the_parts_and_is_dropped() {
        let document = assemble(vec![
            page(1, &format!("Intro.\n\n{HEAD}\n|a|1|\n|b|2|"), false),
            page(2, &format!("{HEAD}\n|c|3|"), false),
            page(3, &format!("{HEAD}\n|d|4|\n\nAfter the table."), false),
        ]);
        assert_eq!(
            document.markdown,
            format!(
                "<!-- Page number: 1 -->\n\nIntro.\n\n{HEAD}\n|a|1|\n|b|2|\n|c|3|\n|d|4|\n\n\
                 <!-- Page number: 2 -->\n\n<!-- Page number: 3 -->\n\nAfter the table."
            )
        );
        // A page whose rows all moved is no missing text.
        assert!(document.warnings.is_empty(), "{:?}", document.warnings);
    }

    #[test]
    fn geometry_joins_a_part_without_a_header_and_keeps_its_first_row() {
        let pages = || {
            vec![
                page(1, &format!("{HEAD}\n|a|1|"), false),
                page(2, "|b|2|\n|---|---|\n|c|3|", true),
            ]
        };
        assert_eq!(
            assemble(pages()).markdown,
            format!(
                "<!-- Page number: 1 -->\n\n{HEAD}\n|a|1|\n|b|2|\n|c|3|\n\n<!-- Page number: 2 -->"
            )
        );
        // Without that evidence, a different first row starts a new table.
        let mut separate = pages();
        separate[1].continues_table = false;
        assert_eq!(
            assemble(separate).markdown,
            format!(
                "<!-- Page number: 1 -->\n\n{HEAD}\n|a|1|\n\n\
                 <!-- Page number: 2 -->\n\n|b|2|\n|---|---|\n|c|3|"
            )
        );
    }

    #[test]
    fn running_headers_and_footers_may_stand_between_the_parts() {
        let document = assemble(
            (1..=3)
                .map(|n| {
                    let body = match n {
                        1 => format!("Intro.\n\n{HEAD}\n|a|1|"),
                        2 => format!("{HEAD}\n|b|2|"),
                        _ => format!("{HEAD}\n|c|3|\n\nAfter the table."),
                    };
                    page(
                        n,
                        &format!("Annual report 2025\n\n{body}\n\nPage {n} of 3"),
                        false,
                    )
                })
                .collect(),
        );
        assert_eq!(
            document.markdown,
            format!(
                "<!-- Page number: 1 -->\n\nAnnual report 2025\n\nIntro.\n\n{HEAD}\n|a|1|\n|b|2|\n|c|3|\n\n\
                 Page 1 of 3\n\n<!-- Page number: 2 -->\n\nAnnual report 2025\n\nPage 2 of 3\n\n\
                 <!-- Page number: 3 -->\n\nAnnual report 2025\n\nAfter the table.\n\nPage 3 of 3"
            )
        );
        // Geometry alone does not reach across them.
        let document = assemble(vec![
            page(1, &format!("Annual report\n\n{HEAD}\n|a|1|"), false),
            page(2, "Annual report\n\n|b|2|\n|---|---|", true),
        ]);
        assert_eq!(document.markdown.matches("|---|---|").count(), 2);
    }

    #[test]
    fn other_text_between_the_parts_or_a_recognized_page_keeps_them_apart() {
        for between in ["Closing remark.", "## Next", "![Chart](chart.png)"] {
            let document = assemble(vec![
                page(1, &format!("{HEAD}\n|a|1|\n\n{between}"), false),
                page(2, &format!("{HEAD}\n|b|2|"), false),
            ]);
            assert_eq!(document.markdown.matches(HEAD).count(), 2, "{between}");
        }
        // A closing line that differs on each page is no running footer.
        let document = assemble(vec![
            page(1, "Opening.\n\nAlpha note.", false),
            page(2, "Opening.\n\nBeta note.", false),
            page(3, &format!("{HEAD}\n|a|1|\n\nGamma note."), false),
            page(4, &format!("{HEAD}\n|b|2|"), false),
        ]);
        assert_eq!(document.markdown.matches(HEAD).count(), 2);
        // Recognized text replaces the page's own, table or not.
        let mut recognized = page(2, &format!("{HEAD}\n|b|2|"), false);
        recognized.ocr_completed = true;
        let document = assemble(vec![
            page(1, &format!("{HEAD}\n|a|1|"), false),
            recognized,
            page(3, &format!("{HEAD}\n|c|3|"), true),
        ]);
        assert_eq!(document.markdown.matches(HEAD).count(), 3);
    }

    #[test]
    fn only_tables_of_one_shape_join() {
        // A different column count, a header of empty cells, one column.
        for (first, second) in [
            (
                format!("{HEAD}\n|a|1|"),
                "|Name|Count|Note|\n|---|---|---|\n|b|2|x|".to_owned(),
            ),
            (
                "| | |\n|---|---|\n|a|1|".to_owned(),
                "| | |\n|---|---|\n|b|2|".to_owned(),
            ),
            (
                "|Name|\n|---|\n|a|".to_owned(),
                "|Name|\n|---|\n|b|".to_owned(),
            ),
        ] {
            let document = assemble(vec![page(1, &first, false), page(2, &second, false)]);
            assert!(
                document
                    .markdown
                    .contains(&format!("<!-- Page number: 2 -->\n\n{second}")),
                "{}",
                document.markdown
            );
        }
        // Geometry joins only a part of the same column count.
        let document = assemble(vec![
            page(1, &format!("{HEAD}\n|a|1|"), false),
            page(2, "|b|2|x|\n|---|---|---|", true),
        ]);
        assert_eq!(
            document
                .markdown
                .matches("<!-- Page number: 2 -->\n\n|b|")
                .count(),
            1
        );
        // Escaped pipes are cell text, not borders.
        let document = assemble(vec![
            page(1, "|A\\|B|C|\n|---|---|\n|a|1|", false),
            page(2, "|A\\|B|C|\n|---|---|\n|b\\|c|2|", false),
        ]);
        assert!(
            document
                .markdown
                .contains("|a|1|\n|b\\|c|2|\n\n<!-- Page number: 2 -->"),
            "{}",
            document.markdown
        );
    }

    #[test]
    fn a_table_over_many_pages_keeps_every_row_in_order() {
        // Each page repeats the header and closes with a running footer.
        let pages = (1..=1500)
            .map(|n| {
                let rows: String = (0..30).map(|k| format!("\n|r{n}-{k}|{k}|")).collect();
                page(n, &format!("{HEAD}{rows}\n\nReport {n}"), false)
            })
            .collect();
        let markdown = assemble(pages).markdown;
        assert_eq!(markdown.matches(HEAD).count(), 1);
        let rows: Vec<&str> = markdown.lines().filter(|l| l.starts_with("|r")).collect();
        assert_eq!(rows.len(), 1500 * 30);
        assert_eq!((rows[0], rows[44_999]), ("|r1-0|0|", "|r1500-29|29|"));
        assert!(
            markdown.contains("|r1500-29|29|\n\nReport 1\n\n<!-- Page number: 2 -->\n\nReport 2")
        );
    }

    #[test]
    fn text_before_the_owning_table_is_kept_byte_for_byte() {
        let code = "```\nfirst\n\n\n  indented\n```";
        let document = assemble(vec![
            page(1, &format!("{code}\n\n\n{HEAD}\n|a|1|"), false),
            page(2, &format!("{HEAD}\n|b|2|"), false),
        ]);
        assert_eq!(
            document.markdown,
            format!(
                "<!-- Page number: 1 -->\n\n{code}\n\n\n{HEAD}\n|a|1|\n|b|2|\n\n<!-- Page number: 2 -->"
            )
        );
    }
}
