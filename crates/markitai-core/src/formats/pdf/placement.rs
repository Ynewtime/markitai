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
/// The fewest letters the line above an image must leave to be found again.
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
/// emphasis, escapes, joined hyphenation and spacing.
fn letters(text: &str) -> impl Iterator<Item = char> + '_ {
    text.chars().filter(|ch| ch.is_alphanumeric())
}

/// The byte offset in `section` just after the paragraph that holds `line`,
/// when the end of that line occurs exactly once in it.
pub(super) fn after_paragraph(section: &str, line: &str) -> Option<usize> {
    let wanted: Vec<char> = letters(line).collect();
    if wanted.len() < MIN_ANCHOR {
        return None;
    }
    let wanted = &wanted[wanted.len().saturating_sub(ANCHOR)..];
    let (chars, ends): (Vec<char>, Vec<usize>) = section
        .char_indices()
        .filter(|(_, ch)| ch.is_alphanumeric())
        .map(|(at, ch)| (ch, at + ch.len_utf8()))
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
    Some(
        section[end..]
            .find("\n\n")
            .map_or(section.len(), |gap| end + gap),
    )
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
}
