//! Running page headers and footers.
//!
//! A conference name, a journal title or a chapter title printed at the top
//! of the pages, or a document's title printed at their foot, is page
//! furniture, like the page number the page reader already removes. Each
//! page's Markdown would otherwise open or end with it. A header or footer
//! is recognized from the Markdown of the pages it marks and confirmed by
//! where it is printed; nothing else is removed.
use super::geometry;
use lopdf::ObjectId;
use regex::Regex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::LazyLock;

/// The band at the top or the foot of a page, as a fraction of its height,
/// in which a running header or footer is printed (a fifth, as Python
/// markitai 1.3.0 takes page chrome).
const BAND: f32 = 0.2;
/// The baseline difference within which two pages' lines stand at the same
/// height.
const SAME_HEIGHT: f32 = 1.5;
/// Characters a header's text has, without its page number.
const LENGTH: std::ops::RangeInclusive<usize> = 8..=160;

/// The text a page's first block shows, for comparison with other pages':
/// emphasis, underline and escapes removed, white space collapsed, and
/// digits and separators at either end (a running page number) trimmed.
/// `None` when the block is no candidate: more than one line, a heading,
/// list item, quotation, table, code fence, image or HTML block, or text
/// outside [`LENGTH`].
fn key(block: &str) -> Option<String> {
    let block = block.trim();
    if block.contains('\n')
        || block.starts_with(['#', '|', '>', '`', '!', '<', '['])
        || ["- ", "* ", "+ "]
            .iter()
            .any(|marker| block.starts_with(marker))
        || block.starts_with(|c: char| c.is_ascii_digit())
            && block
                .trim_start_matches(|c: char| c.is_ascii_digit())
                .starts_with(['.', ')'])
    {
        return None;
    }
    let plain = block
        .replace("**", "")
        .replace("<u>", "")
        .replace("</u>", "")
        .replace(['*', '\\'], "");
    let plain = plain.split_whitespace().collect::<Vec<_>>().join(" ");
    let separators = |c: char| c.is_ascii_digit() || c.is_whitespace() || "|·•–—-/".contains(c);
    let core = plain
        .trim_start_matches(separators)
        .trim_end_matches(separators);
    // Other numbers in the line (a chapter's, `Page 3 of 9`) change from
    // page to page too.
    (LENGTH.contains(&core.chars().count()) && core.chars().any(char::is_alphabetic))
        .then(|| core.replace(|c: char| c.is_ascii_digit(), "#"))
}

/// A page number in the forms the page reader leaves (it removes `3`,
/// `- 3 -` and `Page 3 of 9`): `3 / 9`, `第 3 页`, `第3页，共9页`, `共9页 第3页`,
/// `3 页`, `— 3 —`.
static FOLIO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(?:[-–—~·•]\s*\d{1,4}\s*[-–—~·•]|\d{1,4}\s*/\s*\d{1,4}|\d{1,4}\s*[页頁]|第\s*\d{1,4}\s*[页頁](?:\s*[,，、/]?\s*共\s*\d{1,4}\s*[页頁])?|共\s*\d{1,4}\s*[页頁]\s*[,，、]?\s*第\s*\d{1,4}\s*[页頁])$",
    )
    .expect("folio pattern")
});

/// The block or line as a page number, white space removed, when it is one.
fn folio(text: &str) -> Option<String> {
    let plain = text.replace("**", "").replace(['*', '\\'], "");
    let plain = plain.trim();
    (!plain.contains('\n') && FOLIO.is_match(plain))
        .then(|| plain.chars().filter(|c| !c.is_whitespace()).collect())
}

/// Each page's first block of Markdown: the page's text up to its first
/// blank line.
fn first_block(markdown: &str) -> &str {
    let text = markdown.trim_start();
    text.split("\n\n").next().unwrap_or(text)
}

/// Each page's last block of Markdown: the page's text after its last blank
/// line.
fn last_block(markdown: &str) -> &str {
    let text = markdown.trim_end();
    text.rsplit("\n\n").next().unwrap_or(text)
}

/// Where a running line is printed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Edge {
    Top,
    Bottom,
}

/// The pages whose Markdown opens with a running header (`Edge::Top`) or
/// ends with a running footer (`Edge::Bottom`); see [`key`] for the text
/// compared. Such a line opens (or ends) two pages with text or more, and on
/// each page where it is removed it is the page's topmost (or bottommost)
/// text, all of it on one baseline in the top (or bottom) [`BAND`] of the
/// page and smaller than the body text, or, for a footer set apart from the
/// text above it, at the height of the same footer on another page; `position` reads those pages'
/// positioned text. A page number in a form the page reader leaves
/// ([`folio`]) goes from any page where it is that line. A page whose position is not known (rotated, or not in
/// `frames`) keeps its text.
pub(super) fn marked(
    edge: Edge,
    pages: &[(u32, &str)],
    frames: &BTreeMap<u32, geometry::Frame>,
    position: &dyn Fn(&HashSet<u32>) -> Vec<pdf_inspector::TextItem>,
) -> HashSet<u32> {
    if pages.len() < 2 {
        return HashSet::new();
    }
    let mut by_key: HashMap<String, Vec<u32>> = HashMap::new();
    // A page number goes on any page where it is printed alone in the band.
    let mut folios: HashMap<u32, String> = HashMap::new();
    for &(number, markdown) in pages {
        let block = match edge {
            Edge::Top => first_block(markdown),
            Edge::Bottom => last_block(markdown),
        };
        if let Some(key) = key(block) {
            by_key.entry(key).or_default().push(number);
        } else if let Some(folio) = folio(block) {
            folios.insert(number, folio);
        }
    }
    let enough = |count: usize| count >= 2;
    by_key.retain(|_, numbers| enough(numbers.len()));
    if by_key.is_empty() && folios.is_empty() {
        return HashSet::new();
    }
    let candidates: HashSet<u32> = by_key
        .values()
        .flatten()
        .chain(folios.keys())
        .copied()
        .collect();
    let mut items_by_page: BTreeMap<u32, Vec<pdf_inspector::TextItem>> = BTreeMap::new();
    for item in position(&candidates) {
        if matches!(item.item_type, pdf_inspector::types::ItemType::Text)
            && !item.text.trim().is_empty()
        {
            items_by_page.entry(item.page).or_default().push(item);
        }
    }
    // A page that holds little besides the line (a page of pictures) is
    // measured against the body text of the other pages.
    let all: Vec<pdf_inspector::TextItem> = items_by_page.values().flatten().cloned().collect();
    let document_body = body_size(&all);
    let mut stripped = HashSet::new();
    for (line, numbers) in by_key {
        // Each page's printed line: its height and whether it is set smaller
        // than the body text.
        let printed: Vec<(u32, f32, bool, bool)> = numbers
            .into_iter()
            .filter_map(|number| {
                let (frame, items) = (frames.get(&number)?, items_by_page.get(&number)?);
                let (y, small, apart) = printed_at(edge, items, frame, &line, document_body)?;
                Some((number, y, small, apart))
            })
            .collect();
        // A footer may be set at the body size when it stands apart from
        // the text above it and at the same height on another page; a header
        // must be smaller (a paragraph can open every page).
        let confirmed: Vec<u32> = printed
            .iter()
            .filter(|(number, y, small, apart)| {
                *small
                    || (edge == Edge::Bottom
                        && *apart
                        && printed.iter().any(|(other, other_y, _, _)| {
                            other != number && (other_y - y).abs() <= SAME_HEIGHT
                        }))
            })
            .map(|(number, _, _, _)| *number)
            .collect();
        if enough(confirmed.len()) {
            stripped.extend(confirmed);
        }
    }
    for (number, shown) in folios {
        if let (Some(frame), Some(items)) = (frames.get(&number), items_by_page.get(&number))
            && extreme_line(edge, items, frame)
                .is_some_and(|(line, _)| folio(&line_text(&line)).as_ref() == Some(&shown))
        {
            stripped.insert(number);
        }
    }
    stripped
}

/// The topmost (or bottommost) line of a page, when it lies in the page's
/// top (or bottom) [`BAND`], and the page's other items.
fn extreme_line<'a>(
    edge: Edge,
    items: &'a [pdf_inspector::TextItem],
    frame: &geometry::Frame,
) -> Option<(
    Vec<&'a pdf_inspector::TextItem>,
    Vec<&'a pdf_inspector::TextItem>,
)> {
    let at = items
        .iter()
        .map(|item| item.y)
        .fold(None, |at: Option<f32>, y| {
            Some(at.map_or(y, |at| match edge {
                Edge::Top => at.max(y),
                Edge::Bottom => at.min(y),
            }))
        })?;
    let inside = match edge {
        Edge::Top => at >= frame.height * (1. - BAND),
        Edge::Bottom => at <= frame.height * BAND,
    };
    if !at.is_finite() || !inside {
        return None;
    }
    let (mut line, rest): (Vec<_>, Vec<_>) = items
        .iter()
        .partition(|item| (item.y - at).abs() <= item.font_size.max(1.) * 0.2);
    line.sort_by(|a, b| a.x.total_cmp(&b.x));
    Some((line, rest))
}

/// A line's text, its items in reading order.
fn line_text(line: &[&pdf_inspector::TextItem]) -> String {
    line.iter()
        .map(|item| item.text.trim())
        .collect::<Vec<_>>()
        .join(" ")
}

/// The size most of a page's characters are set in.
fn body_size(items: &[pdf_inspector::TextItem]) -> f32 {
    let mut sizes: BTreeMap<i32, usize> = BTreeMap::new();
    for item in items {
        *sizes
            .entry((item.font_size * 10.).round() as i32)
            .or_default() += item.text.chars().filter(|c| !c.is_whitespace()).count();
    }
    sizes
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map_or(0., |(size, _)| size as f32 / 10.)
}

/// The baseline of the topmost (or bottommost) text of a page when that is
/// `shown`, on one baseline in the page's top (or bottom) band, whether it
/// is set smaller than the page's body text, and whether it stands apart
/// from the page's other text. A page with less text
/// besides the line than the line itself is measured against
/// `document_body`.
fn printed_at(
    edge: Edge,
    items: &[pdf_inspector::TextItem],
    frame: &geometry::Frame,
    shown: &str,
    document_body: f32,
) -> Option<(f32, bool, bool)> {
    let (line, rest) = extreme_line(edge, items, frame)?;
    let letters = |items: &[&pdf_inspector::TextItem]| {
        items
            .iter()
            .map(|item| item.text.chars().filter(|c| !c.is_whitespace()).count())
            .sum::<usize>()
    };
    let body = if letters(&rest) > letters(&line) {
        body_size(items)
    } else {
        document_body
    };
    let small = line.iter().all(|item| item.font_size < body * 0.95);
    // A line at the body size must stand apart from the text beside it, as
    // page chrome does: further from it than the page's lines are from each
    // other (its median baseline pitch), or two of its sizes when the page
    // has no run of lines to measure.
    let size = line
        .iter()
        .map(|item| item.font_size)
        .fold(0f32, f32::max)
        .max(1.);
    let gap = rest
        .iter()
        .map(|item| (item.y - line[0].y).abs())
        .fold(f32::INFINITY, f32::min);
    let apart = gap >= pitch(&rest).map_or(size * 2., |pitch| pitch * 1.4);
    key(&line_text(&line))
        .is_some_and(|key| same_line(&key, shown))
        .then(|| (line[0].y, small, apart))
}

/// The median distance between the baselines of consecutive lines of
/// `items`, when there are a few lines close enough to be running text.
fn pitch(items: &[&pdf_inspector::TextItem]) -> Option<f32> {
    let mut baselines: Vec<f32> = items.iter().map(|item| item.y).collect();
    baselines.sort_by(|a, b| b.total_cmp(a));
    baselines.dedup_by(|a, b| (*a - *b).abs() <= 1.);
    let mut gaps: Vec<f32> = baselines
        .windows(2)
        .map(|pair| pair[0] - pair[1])
        .filter(|gap| *gap > 0. && *gap < 60.)
        .collect();
    if gaps.len() < 3 {
        return None;
    }
    gaps.sort_by(f32::total_cmp);
    Some(gaps[gaps.len() / 2])
}

/// Whether the key of a printed line is `shown`, the key of the Markdown
/// block: the same text, or that text with the `Page N` the page reader
/// removed from the block (`Page 3 Publication 501 (2025)`).
fn same_line(printed: &str, shown: &str) -> bool {
    printed == shown
        || [printed.strip_prefix(shown), printed.strip_suffix(shown)]
            .into_iter()
            .flatten()
            .any(|rest| {
                let rest = rest.trim_matches(|c: char| c.is_whitespace() || "|·•–—-/#".contains(c));
                rest.eq_ignore_ascii_case("page") || rest.eq_ignore_ascii_case("p.")
            })
}

/// Remove the first (`Edge::Top`) or last (`Edge::Bottom`) block of
/// `markdown` (see [`marked`]).
pub(super) fn strip(edge: Edge, markdown: &mut String) {
    *markdown = match edge {
        Edge::Top => match markdown.trim_start().split_once("\n\n") {
            Some((_, rest)) => rest.trim_start().to_owned(),
            None => String::new(),
        },
        Edge::Bottom => match markdown.trim_end().rsplit_once("\n\n") {
            Some((rest, _)) => rest.trim_end().to_owned(),
            None => String::new(),
        },
    };
}

/// The visible page box of each listed page, for [`marked`].
pub(super) fn frames(
    pdf: &lopdf::Document,
    page_ids: &BTreeMap<u32, ObjectId>,
    pages: &[u32],
) -> BTreeMap<u32, geometry::Frame> {
    pages
        .iter()
        .filter_map(|&number| {
            let id = page_ids.get(&number)?;
            geometry::frame(pdf, *id).map(|frame| (number, frame))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_header_key_ignores_its_page_number_and_styling() {
        assert_eq!(
            key("Proc. of the Conference on Probes 2026").as_deref(),
            Some("Proc. of the Conference on Probes")
        );
        assert_eq!(
            key("**Chapter 3 — Methods** | 17").as_deref(),
            Some("Chapter # — Methods")
        );
        for block in [
            "# A heading",
            "- an item",
            "1. an item",
            "|a|b|",
            "line one\nline two",
            "42",
            "Short",
        ] {
            assert_eq!(key(block), None, "{block}");
        }
    }

    #[test]
    fn a_printed_line_may_keep_the_page_number_its_block_lost() {
        let shown = key("**Publication 501 (2025)**").unwrap();
        assert!(same_line(
            &key("Page 3 Publication 501 (2025)").unwrap(),
            &shown
        ));
        assert!(same_line(
            &key("Publication 501 (2025) Page 4").unwrap(),
            &shown
        ));
        assert!(!same_line(
            &key("Draft Publication 501 (2025)").unwrap(),
            &shown
        ));
    }

    #[test]
    fn page_numbers_the_reader_leaves_are_recognized() {
        for text in [
            "3 / 9",
            "第 3 页",
            "第3页，共9页",
            "共 9 页 第 3 页",
            "— 3 —",
            "**12 页**",
        ] {
            assert!(folio(text).is_some(), "{text}");
        }
        assert_eq!(folio("第 3 页").as_deref(), Some("第3页"));
        for text in [
            "3/9 of the budget",
            "第三章",
            "Page 3 explains it",
            "1 / 2 cup",
            "2026",
        ] {
            assert!(folio(text).is_none(), "{text}");
        }
    }

    #[test]
    fn strip_removes_only_the_first_or_the_last_block() {
        let mut markdown = "Running head\n\nFirst paragraph.\n\nSecond.".to_owned();
        strip(Edge::Top, &mut markdown);
        assert_eq!(markdown, "First paragraph.\n\nSecond.");
        strip(Edge::Bottom, &mut markdown);
        assert_eq!(markdown, "First paragraph.");
        let mut alone = "Running head".to_owned();
        strip(Edge::Top, &mut alone);
        assert_eq!(alone, "");
        let mut alone = "Running foot\n".to_owned();
        strip(Edge::Bottom, &mut alone);
        assert_eq!(alone, "");
    }
}
