//! Running page headers.
//!
//! A conference name, a journal title or a chapter title printed at the top
//! of every page is page furniture, like the page number at the bottom that
//! the page reader already removes. Each page's Markdown would otherwise
//! open with it. A header is recognized from the Markdown of the pages it
//! heads and confirmed by where it is printed; nothing else is removed.
use super::geometry;
use lopdf::ObjectId;
use std::collections::{BTreeMap, HashMap, HashSet};

/// The top band of a page, as a fraction of its height, in which a running
/// header is printed.
const TOP_BAND: f32 = 0.15;
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
    (LENGTH.contains(&core.chars().count()) && core.chars().any(char::is_alphabetic))
        .then(|| core.to_owned())
}

/// Each page's first block of Markdown: the page's text up to its first
/// blank line.
fn first_block(markdown: &str) -> &str {
    let text = markdown.trim_start();
    text.split("\n\n").next().unwrap_or(text)
}

/// The pages whose Markdown opens with a running header (see [`key`] for
/// the text compared). A header opens more than half of the pages with text,
/// three at least, and on each page where it is removed it is the page's
/// topmost text, all of it on one baseline in the top [`TOP_BAND`] of the
/// page and smaller than the page's body text; `position` reads those
/// pages' positioned text. A page whose position is not known (rotated, or
/// not in `frames`) keeps its text.
pub(super) fn headers(
    pages: &[(u32, &str)],
    frames: &BTreeMap<u32, geometry::Frame>,
    position: &dyn Fn(&HashSet<u32>) -> Vec<pdf_inspector::TextItem>,
) -> HashSet<u32> {
    if pages.len() < 3 {
        return HashSet::new();
    }
    let mut by_key: HashMap<String, Vec<u32>> = HashMap::new();
    for &(number, markdown) in pages {
        if let Some(key) = key(first_block(markdown)) {
            by_key.entry(key).or_default().push(number);
        }
    }
    let enough = |count: usize| count >= 3 && count * 2 > pages.len();
    by_key.retain(|_, numbers| enough(numbers.len()));
    if by_key.is_empty() {
        return HashSet::new();
    }
    let candidates: HashSet<u32> = by_key.values().flatten().copied().collect();
    let mut items_by_page: BTreeMap<u32, Vec<pdf_inspector::TextItem>> = BTreeMap::new();
    for item in position(&candidates) {
        if matches!(item.item_type, pdf_inspector::types::ItemType::Text)
            && !item.text.trim().is_empty()
        {
            items_by_page.entry(item.page).or_default().push(item);
        }
    }
    let mut stripped = HashSet::new();
    for (header, numbers) in by_key {
        let confirmed: Vec<u32> = numbers
            .into_iter()
            .filter(|number| {
                let (Some(frame), Some(items)) = (frames.get(number), items_by_page.get(number))
                else {
                    return false;
                };
                printed_on_top(items, frame, &header)
            })
            .collect();
        if enough(confirmed.len()) {
            stripped.extend(confirmed);
        }
    }
    stripped
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

/// Whether the topmost text of a page is `header`, on one baseline in the
/// page's top band and set smaller than the page's body text (a paragraph
/// that happens to open every page is body-sized).
fn printed_on_top(
    items: &[pdf_inspector::TextItem],
    frame: &geometry::Frame,
    header: &str,
) -> bool {
    let Some(top) = items
        .iter()
        .map(|item| item.y)
        .fold(None, |top: Option<f32>, y| {
            Some(top.map_or(y, |t| t.max(y)))
        })
    else {
        return false;
    };
    if !top.is_finite() || top < frame.height * (1. - TOP_BAND) {
        return false;
    }
    let mut line: Vec<&pdf_inspector::TextItem> = items
        .iter()
        .filter(|item| (item.y - top).abs() <= item.font_size.max(1.) * 0.2)
        .collect();
    let body = body_size(items);
    if line.iter().any(|item| item.font_size >= body * 0.95) {
        return false;
    }
    line.sort_by(|a, b| a.x.total_cmp(&b.x));
    let text = line
        .iter()
        .map(|item| item.text.trim())
        .collect::<Vec<_>>()
        .join(" ");
    key(&text).is_some_and(|shown| shown == header)
}

/// Remove the first block of `markdown` (see [`headers`]).
pub(super) fn strip(markdown: &mut String) {
    let text = markdown.trim_start();
    *markdown = match text.split_once("\n\n") {
        Some((_, rest)) => rest.trim_start().to_owned(),
        None => String::new(),
    };
}

/// The visible page box of each listed page, for [`headers`].
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
            Some("Chapter 3 — Methods")
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
    fn strip_removes_only_the_first_block() {
        let mut markdown = "Running head\n\nFirst paragraph.\n\nSecond.".to_owned();
        strip(&mut markdown);
        assert_eq!(markdown, "First paragraph.\n\nSecond.");
        let mut alone = "Running head".to_owned();
        strip(&mut alone);
        assert_eq!(alone, "");
    }
}
