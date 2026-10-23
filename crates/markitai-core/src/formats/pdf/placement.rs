//! Where an embedded image goes in its page's text. The page reader writes
//! the text alone; each image is placed by where it is drawn among the
//! page's text lines: before the text when it lies above every line, after
//! the paragraph that holds the nearest line above it, or after the text when
//! it lies below every line. An image whose place cannot be found in the
//! page's Markdown follows the page's text, as before.

use super::named_resource;
use lopdf::{Document, Object, ObjectId};
use pdf_inspector::{TextItem, types::ItemType};
use std::collections::{BTreeMap, HashSet};

/// Lines further apart than this baseline difference are different lines.
const SAME_LINE: f32 = 2.0;
/// The fewest letters the line above an image must leave to be found again;
/// a Han, kana or Hangul character counts as two.
const MIN_ANCHOR: usize = 12;
/// The letters of that line's end looked for in the page's Markdown.
const ANCHOR: usize = 40;
/// Pages with more positioned items than this keep their images at the end.
const MAX_ITEMS: usize = 50_000;

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Place {
    /// Above every text line: before the page's text.
    Start,
    /// After the paragraph holding this line, the nearest above the image.
    After(String),
    /// Below every text line: after the page's text.
    End,
}

struct Line {
    y: f32,
    x0: f32,
    x1: f32,
    text: String,
}

fn lines(items: &[&TextItem]) -> Vec<Line> {
    let mut items: Vec<_> = items
        .iter()
        .filter(|item| {
            matches!(item.item_type, ItemType::Text)
                && item.rotation == 0.0
                && !item.text.trim().is_empty()
                && item.x.is_finite()
                && item.y.is_finite()
        })
        .collect();
    items.sort_by(|a, b| b.y.total_cmp(&a.y).then(a.x.total_cmp(&b.x)));
    let mut lines: Vec<Line> = Vec::new();
    let mut members: Vec<Vec<&&TextItem>> = Vec::new();
    for item in items {
        match lines.last_mut() {
            Some(line) if (line.y - item.y).abs() <= SAME_LINE => {
                line.x0 = line.x0.min(item.x);
                line.x1 = line.x1.max(item.x + item.width);
                members.last_mut().expect("one per line").push(item);
            }
            _ => {
                lines.push(Line {
                    y: item.y,
                    x0: item.x,
                    x1: item.x + item.width,
                    text: String::new(),
                });
                members.push(vec![item]);
            }
        }
    }
    for (line, mut items) in lines.iter_mut().zip(members) {
        items.sort_by(|a, b| a.x.total_cmp(&b.x));
        line.text = items.iter().map(|item| item.text.as_str()).collect();
    }
    lines
}

/// The page-level image XObject a `[Image: NAME]` item names.
fn image_object(pdf: &Document, resources: &[&lopdf::Dictionary], name: &str) -> Option<ObjectId> {
    let id = named_resource(pdf, resources, b"XObject", name.as_bytes())?
        .as_reference()
        .ok()?;
    let stream = pdf.get_object(id).ok()?.as_stream().ok()?;
    (stream.dict.get(b"Subtype").and_then(Object::as_name).ok() == Some(b"Image")).then_some(id)
}

/// Each image's place on each page, for the images drawn on the page itself
/// (not inside a form) with a known position. `items` are the pages'
/// positioned text and image items.
pub(super) fn places(
    pdf: &Document,
    pages: &BTreeMap<u32, ObjectId>,
    items: &[TextItem],
) -> BTreeMap<u32, BTreeMap<ObjectId, Place>> {
    let mut by_page: BTreeMap<u32, Vec<&TextItem>> = BTreeMap::new();
    for item in items {
        by_page.entry(item.page).or_default().push(item);
    }
    let mut out = BTreeMap::new();
    for (number, items) in by_page {
        let Some(&id) = pages.get(&number) else {
            continue;
        };
        if items.len() > MAX_ITEMS {
            continue;
        }
        let Ok(Some(resources)) = super::sanitize::page_resources(pdf, id) else {
            continue;
        };
        let resources = [resources];
        let lines = lines(&items);
        if lines.is_empty() {
            continue;
        }
        let mut seen = HashSet::new();
        let mut places = BTreeMap::new();
        for image in items
            .iter()
            .filter(|item| matches!(item.item_type, ItemType::Image))
        {
            let Some(name) = image
                .text
                .strip_prefix("[Image: ")
                .and_then(|rest| rest.strip_suffix(']'))
            else {
                continue;
            };
            let Some(object) = image_object(pdf, &resources, name) else {
                continue;
            };
            // An image drawn twice has no single place.
            if !seen.insert(object) {
                places.remove(&object);
                continue;
            }
            let (bottom, top) = (image.y, image.y + image.height);
            let place = if lines.iter().all(|line| line.y < bottom) {
                Some(Place::Start)
            } else if lines.iter().all(|line| line.y > top) {
                Some(Place::End)
            } else {
                // The nearest line above the image that shares its columns.
                lines
                    .iter()
                    .filter(|line| {
                        line.y > top && line.x0 < image.x + image.width && line.x1 > image.x
                    })
                    .min_by(|a, b| a.y.total_cmp(&b.y))
                    .map(|line| Place::After(line.text.clone()))
            };
            if let Some(place) = place {
                places.insert(object, place);
            }
        }
        if !places.is_empty() {
            out.insert(number, places);
        }
    }
    out
}

/// Letters and digits only: what a line keeps through the Markdown's
/// emphasis, escapes, joined hyphenation and spacing. A radical form that
/// stands for an ideograph (Chrome's 同⼀套) is that ideograph.
fn letter(ch: char) -> Option<char> {
    super::radicals::ideograph(ch).or_else(|| ch.is_alphanumeric().then_some(ch))
}

fn letters(text: &str) -> impl Iterator<Item = char> + '_ {
    text.chars().filter_map(letter)
}

/// How much of an anchor `letters` make: an ideograph, kana or Hangul
/// syllable says about as much as two Latin letters, so a short CJK heading
/// ("同一套转换内核") is found again as surely as "One conversion core".
fn weight(letters: &[char]) -> usize {
    letters.iter().map(|&ch| if wide(ch) { 2 } else { 1 }).sum()
}

fn wide(ch: char) -> bool {
    matches!(ch as u32, 0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF | 0xF900..=0xFAFF | 0x20000..=0x3FFFD)
}

/// Whether a Markdown line starts a list item (`- `, `* `, `+ `, `1. `,
/// `1) `), after its indentation.
fn list_item(line: &str) -> bool {
    let line = line.trim_start_matches(' ');
    let digits = line.bytes().take_while(u8::is_ascii_digit).count();
    let rest = if (1..=9).contains(&digits) && line[digits..].starts_with(['.', ')']) {
        &line[digits + 1..]
    } else if line.starts_with(['-', '*', '+']) {
        &line[1..]
    } else {
        return false;
    };
    rest.is_empty() || rest.starts_with([' ', '\t'])
}

/// The byte offset in `section` just after the paragraph that holds `line`
/// (after the list item, when the line is a list item's: a picture drawn
/// between two items goes between them), when the end of that line occurs
/// exactly once in it.
pub(super) fn after_paragraph(section: &str, line: &str) -> Option<usize> {
    let wanted: Vec<char> = letters(line).collect();
    if weight(&wanted) < MIN_ANCHOR {
        return None;
    }
    let wanted = &wanted[wanted.len().saturating_sub(ANCHOR)..];
    let (chars, ends): (Vec<char>, Vec<usize>) = section
        .char_indices()
        .filter_map(|(at, ch)| letter(ch).map(|letter| (letter, at + ch.len_utf8())))
        .unzip();
    let mut found = chars
        .windows(wanted.len())
        .enumerate()
        .filter(|(_, window)| *window == wanted)
        .map(|(start, _)| ends[start + wanted.len() - 1]);
    let end = found.next()?;
    if found.next().is_some() {
        return None;
    }
    let paragraph_end = section[end..]
        .find("\n\n")
        .map_or(section.len(), |gap| end + gap);
    let line_start = section[..end].rfind('\n').map_or(0, |at| at + 1);
    if !list_item(&section[line_start..]) {
        return Some(paragraph_end);
    }
    // The item ends where the next one starts, within its list.
    let mut at = end;
    while let Some(newline) = section[at..paragraph_end].find('\n') {
        let next = at + newline;
        if list_item(&section[next + 1..]) {
            return Some(next);
        }
        at = next + 1;
    }
    Some(paragraph_end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_end_is_found_once_through_markup() {
        let section = "<!-- Page number: 4 -->\n\nFirst paragraph ends **here, in bold**.\n\nAliquam ante quam,\npellentesque ut dignissim.\n\nLast.";
        let at = after_paragraph(section, "Aliquam ante quam,").unwrap();
        assert_eq!(&section[at..], "\n\nLast.");
        let at = after_paragraph(section, "First paragraph ends here, in bold.").unwrap();
        assert!(section[at..].starts_with("\n\nAliquam"));
        // Too short, absent or repeated: no place.
        assert_eq!(after_paragraph(section, "Last."), None);
        assert_eq!(after_paragraph(section, "Something else entirely"), None);
        assert_eq!(
            after_paragraph("repeated line\n\nrepeated line", "repeated line"),
            None
        );
    }

    #[test]
    fn a_short_cjk_line_is_an_anchor_through_radical_forms() {
        // Seven ideographs, one of them Chrome's radical form ⼀ (U+2F00)
        let section =
            "# 文件和网页转成 Markdown\n\n## 同⼀套转换内核\n\n图 1. 命令⾏和 MCP 服务共⽤⼀条流程";
        let at = after_paragraph(section, "同⼀套转换内核").unwrap();
        assert!(section[at..].starts_with("\n\n图 1."));
        assert_eq!(after_paragraph(section, "同一套转换内核"), Some(at));
        // Still too short: three ideographs, or six Latin letters
        assert_eq!(after_paragraph("## 能转什么\n\nOffice", "能转什么"), None);
        assert_eq!(after_paragraph("## Office\n\nPDF", "Office"), None);
    }

    #[test]
    fn a_line_of_a_list_item_places_after_that_item() {
        let section = "Intro.\n\n- 2、Choose a server in the Proxy group\n  of the list\n- 3、Press start in the corner\n\nEnd.";
        let at = after_paragraph(section, "2、Choose a server in the Proxy group").unwrap();
        assert_eq!(&section[at..], "\n- 3、Press start in the corner\n\nEnd.");
        let at = after_paragraph(section, "3、Press start in the corner").unwrap();
        assert_eq!(&section[at..], "\n\nEnd.");
        assert!(list_item("12) twelve") && list_item("  * nested") && list_item("-"));
        assert!(!list_item("-5 degrees") && !list_item("1.5 m"));
    }
}
