//! List bullets painted as shapes (markitai).
//!
//! A browser draws a list item's marker as a small filled disc or square,
//! or a stroked ring, instead of showing a bullet character, so the text
//! layer holds no marker and the page reader runs the items into one
//! paragraph. A caller that reads the page's vector graphics passes such
//! shapes in ([`LoadedPdf::pages_markdown_with_marks`](crate::LoadedPdf));
//! a shape that marks the start of a text line becomes a `•` item where it
//! is painted, which the reader's list detection takes like a bullet
//! character. [`targets`] is the rule, also for a caller's own layout.

use crate::types::{ItemType, TextItem};

/// A compact painted shape that may be a list bullet: a filled shape or a
/// stroked ring (a stroked shape with straight sides is a checkbox or a
/// frame), in the coordinates of the text items it is matched against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PaintedMark {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    /// The colour it is painted in: its fill, or its stroke when it is only
    /// stroked.
    pub color: [u8; 3],
}

/// Text that ends this close before a mark, in em, makes the mark part of
/// its line (an inline icon or separator) rather than the line's start.
const CLEAR_LEFT: f32 = 3.0;
/// Colour channels of a bullet and of its text agree within this: a
/// browser paints the marker in the list item's colour, while legend
/// swatches and status dots have colours of their own.
const SAME_COLOUR: u8 = 32;
/// A mark no lighter than this in any channel, and with channels this
/// close, is a dark neutral (black or dark grey), the colour of most lists
/// whatever colour their items' links are.
const DARK: u8 = 128;
const NEUTRAL: u8 = 24;
/// Size bounds of a bullet against its text, in em.
const MAX_SIZE: f32 = 0.5;
const MIN_SIZE: f32 = 0.15;

fn is_text(item: &TextItem) -> bool {
    matches!(item.item_type, ItemType::Text) && !item.text.trim().is_empty() && item.is_upright()
}

fn same_colour(a: [u8; 3], b: [u8; 3]) -> bool {
    a.iter().zip(b).all(|(a, b)| a.abs_diff(b) <= SAME_COLOUR)
}

fn dark_neutral([r, g, b]: [u8; 3]) -> bool {
    r.max(g).max(b) <= DARK && r.max(g).max(b) - r.min(g).min(b) <= NEUTRAL
}

/// The colour most of the text is set in, when known.
fn body_colour(items: &[TextItem], order: &[usize]) -> Option<[u8; 3]> {
    let mut counts: Vec<([u8; 3], usize)> = Vec::new();
    for item in order.iter().map(|&i| &items[i]) {
        let Some(colour) = item.fill_color else {
            continue;
        };
        let chars = item.text.chars().count();
        match counts.iter_mut().find(|(c, _)| *c == colour) {
            Some((_, count)) => *count += chars,
            None => counts.push((colour, chars)),
        }
    }
    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(colour, _)| colour)
}

/// The marks that are bullets of text lines, each with the index of the
/// first item of its line: a mark at most half that item's size and at
/// least 0.15 of it (both extents), ending no more than two em before the
/// item and at most half a point inside it, centred between the item's
/// baseline and 0.8 em above it (a browser centres its disc near the
/// x-height), and painted in a dark neutral, in the item's colour or in the
/// colour of most of the text (a link's bullet keeps the list's colour; a
/// coloured swatch or status dot has a colour of its own), with no other text
/// on that baseline ending within three em before the mark or between the
/// mark and the item. A mark is a bullet of the nearest such item; an item
/// takes one mark. Sorted by item index.
pub fn targets(items: &[TextItem], marks: &[PaintedMark]) -> Vec<(usize, usize)> {
    if marks.is_empty() {
        return Vec::new();
    }
    // Text items by baseline, so each mark looks at nearby lines only.
    let mut order: Vec<usize> = (0..items.len()).filter(|&i| is_text(&items[i])).collect();
    crate::sort::stable(&mut order, &mut |a: &usize, b: &usize| {
        items[*a].line_y().total_cmp(&items[*b].line_y())
    });
    let baselines: Vec<f32> = order.iter().map(|&i| items[i].line_y()).collect();
    let body = body_colour(items, &order);
    let mut taken = vec![false; items.len()];
    let within = |low: f32, high: f32| {
        let start = baselines.partition_point(|&y| y < low);
        let end = baselines.partition_point(|&y| y <= high);
        order[start..end].iter().copied()
    };
    let mut found: Vec<(usize, usize)> = Vec::new();
    for (index, mark) in marks.iter().enumerate() {
        let (width, height) = (mark.x1 - mark.x0, mark.y1 - mark.y0);
        if !(width > 0. && height > 0.) {
            continue;
        }
        let centre = (mark.y0 + mark.y1) / 2.;
        // The largest text this mark can be a bullet of sets how far below
        // its centre that text's baseline can be.
        let largest = width.min(height) / MIN_SIZE;
        let target = within(centre - largest * 0.8, centre)
            .filter(|&i| {
                let item = &items[i];
                let size = item.font_size;
                let baseline = item.line_y();
                width.max(height) <= size * MAX_SIZE
                    && width.min(height) >= size * MIN_SIZE
                    && mark.x1 <= item.x + 0.5
                    && item.x - mark.x1 <= size * 2.
                    && centre >= baseline
                    && centre <= baseline + size * 0.8
                    && (dark_neutral(mark.color)
                        || [item.fill_color, body]
                            .into_iter()
                            .flatten()
                            .any(|colour| same_colour(colour, mark.color)))
            })
            .min_by(|&a, &b| items[a].x.total_cmp(&items[b].x));
        let Some(target) = target else {
            continue;
        };
        let item = &items[target];
        let size = item.font_size;
        let baseline = item.line_y();
        let tolerance = (size * 0.2).min(2.);
        let crowded = within(baseline - tolerance, baseline + tolerance).any(|other| {
            let o = &items[other];
            other != target && o.x < item.x && o.x + o.width > mark.x0 - size * CLEAR_LEFT
        });
        if !crowded && !taken[target] {
            taken[target] = true;
            found.push((index, target));
        }
    }
    crate::sort::total(&mut found, &mut |a: &(usize, usize), b: &(usize, usize)| {
        a.1.cmp(&b.1)
    });
    found
}

/// The `•` item a mark that is the bullet of `item`'s line reads as: placed
/// where the mark is painted, on the line's baseline, otherwise like `item`
/// so it joins that line, in a tagged PDF its marked content too.
pub(crate) fn bullet(item: &TextItem, mark: &PaintedMark) -> TextItem {
    let mut bullet = item.clone();
    bullet.text = "\u{2022}".into();
    bullet.x = mark.x0;
    bullet.y = item.line_y();
    bullet.width = mark.x1 - mark.x0;
    bullet.fill_color = Some(mark.color);
    bullet.is_underline = false;
    bullet.is_strikeout = false;
    bullet.baseline_shift = 0.;
    bullet
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ItemType;
    use crate::LoadedPdf;
    use lopdf::{dictionary, Document, Object, Stream};

    const BLACK: [u8; 3] = [0, 0, 0];
    const BLUE: [u8; 3] = [0, 0, 238];

    fn item(text: &str, x: f32, y: f32, colour: [u8; 3]) -> TextItem {
        TextItem {
            text: text.into(),
            x,
            y,
            width: text.len() as f32 * 6.,
            height: 12.,
            font: "Helvetica".into(),
            font_tag: String::new(),
            legacy_symbol_rewrite: false,
            font_size: 12.,
            page: 1,
            is_bold: false,
            is_italic: false,
            font_weight: None,
            bold_source: None,
            fixed_pitch: None,
            fill_color: Some(colour),
            stroke_color: None,
            render_mode: None,
            is_underline: false,
            is_strikeout: false,
            rotation: 0.,
            advance_known: true,
            item_type: ItemType::Text,
            mcid: None,
            baseline_shift: 0.,
        }
    }

    /// A 4.5pt disc whose centre is `rise` above the baseline `y`, ending
    /// `gap` before `x`.
    fn disc(x: f32, y: f32, gap: f32, rise: f32, color: [u8; 3]) -> PaintedMark {
        let (x1, centre) = (x - gap, y + rise);
        PaintedMark {
            x0: x1 - 4.5,
            y0: centre - 2.25,
            x1,
            y1: centre + 2.25,
            color,
        }
    }

    #[test]
    fn a_compact_mark_before_a_line_in_the_list_colour_is_its_bullet() {
        let items = vec![
            item("Most of the text is set in black.", 40., 760., BLACK),
            item("First item", 64., 700., BLACK),
            item("A linked item", 64., 680., BLUE),
        ];
        let marks = [
            disc(64., 700., 9., 4., BLACK),
            disc(64., 680., 9., 4., BLACK),
        ];
        assert_eq!(targets(&items, &marks), [(0, 1), (1, 2)]);
        // A coloured mark only before text of its colour: a legend swatch or
        // a status dot is no bullet.
        let red = [221, 51, 51];
        assert_eq!(targets(&items, &[disc(64., 700., 9., 4., red)]), []);
        let mut red_text = items.clone();
        red_text[1].fill_color = Some(red);
        assert_eq!(
            targets(&red_text, &[disc(64., 700., 9., 4., red)]),
            [(0, 1)]
        );
        // A blue body colour admits a blue mark before black text.
        let mut blue_body = items.clone();
        blue_body[0].fill_color = Some(BLUE);
        blue_body[0].text = "Most of this page's text is blue, more than the rest.".into();
        assert_eq!(
            targets(&blue_body, &[disc(64., 700., 9., 4., BLUE)]),
            [(0, 1)]
        );
        // A black bullet before a link on a page set in dark grey: neither
        // the item's colour nor the body's, but a dark neutral.
        let mut grey_body = items.clone();
        grey_body[0].fill_color = Some([80, 80, 80]);
        assert_eq!(
            targets(&grey_body, &[disc(64., 680., 9., 4., BLACK)]),
            [(0, 2)]
        );
    }

    #[test]
    fn size_position_and_neighbours_decide() {
        let first = || vec![item("First item", 64., 700., BLACK)];
        let check = |items: &[TextItem], mark: PaintedMark| targets(items, &[mark]).len();
        assert_eq!(check(&first(), disc(64., 700., 9., 4., BLACK)), 1);
        // Larger than half the text or smaller than 0.15 of it.
        let big = PaintedMark {
            x0: 48.,
            y0: 700.,
            x1: 55.,
            y1: 707.,
            color: BLACK,
        };
        let tiny = PaintedMark {
            x0: 54.,
            y0: 703.,
            x1: 55.5,
            y1: 704.5,
            color: BLACK,
        };
        assert_eq!(check(&first(), big), 0);
        assert_eq!(check(&first(), tiny), 0);
        // More than two em before the text, overlapping it, below the
        // baseline or above 0.8 em.
        assert_eq!(check(&first(), disc(64., 700., 25., 4., BLACK)), 0);
        assert_eq!(check(&first(), disc(64., 700., -3., 4., BLACK)), 0);
        assert_eq!(check(&first(), disc(64., 700., 9., -1., BLACK)), 0);
        assert_eq!(check(&first(), disc(64., 700., 9., 10., BLACK)), 0);
        // Text close before the mark: a separator inside a line.
        let mut inline = first();
        inline.push(item("words", 20., 700., BLACK));
        assert_eq!(check(&inline, disc(64., 700., 9., 4., BLACK)), 0);
        // Text of another column, three em or more to the left, is no
        // obstacle.
        let column = vec![
            item("First item", 164., 700., BLACK),
            item("side", 90., 700., BLACK),
        ];
        assert_eq!(check(&column, disc(164., 700., 9., 4., BLACK)), 1);
        let near = vec![
            item("First item", 164., 700., BLACK),
            item("side", 100., 700., BLACK),
        ];
        assert_eq!(check(&near, disc(164., 700., 9., 4., BLACK)), 0);
        // One mark per line.
        let marks = [
            disc(64., 700., 9., 4., BLACK),
            disc(64., 700., 8., 4., BLACK),
        ];
        assert_eq!(targets(&first(), &marks), [(0, 0)]);
    }

    /// One page: a sentence, then three lines each after a 4.5pt disc.
    fn listed_page() -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let pages = doc.new_object_id();
        let font = doc.add_object(
            dictionary! { "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica" },
        );
        let resources = doc.add_object(dictionary! { "Font" => dictionary! { "F1" => font } });
        let mut content =
            String::from("BT /F1 12 Tf 40 740 Td (The kit holds three tools.) Tj ET\n");
        for (index, tool) in ["A hammer", "A saw", "A level"].iter().enumerate() {
            let y = 710 - 20 * index as i32;
            content.push_str(&format!("BT /F1 12 Tf 64 {y} Td ({tool}) Tj ET\n"));
        }
        let contents = doc.add_object(Stream::new(dictionary! {}, content.into_bytes()));
        let page = doc.add_object(dictionary! {
            "Type" => "Page", "Parent" => pages, "Contents" => contents, "Resources" => resources
        });
        doc.objects.insert(
            pages,
            dictionary! {
                "Type" => "Pages", "Count" => 1, "Kids" => vec![Object::Reference(page)],
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()]
            }
            .into(),
        );
        let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages });
        doc.trailer.set("Root", catalog);
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn the_page_reader_lists_lines_after_caller_supplied_bullets() {
        let bytes = listed_page();
        let loaded = LoadedPdf::load_mem(&bytes).unwrap();
        let plain = loaded.pages_markdown(None).unwrap().pages[0]
            .markdown
            .clone();
        assert!(!plain.contains("- A saw"), "{plain}");
        let marks = |page: u32| -> Vec<PaintedMark> {
            (0..3)
                .filter(|_| page == 1)
                .map(|index| disc(64., 710. - 20. * index as f32, 9., 4., BLACK))
                .collect()
        };
        let listed = loaded
            .pages_markdown_with_marks(None, &marks)
            .unwrap()
            .pages[0]
            .markdown
            .clone();
        assert!(
            listed.contains("- A hammer\n- A saw\n- A level"),
            "{listed}"
        );
        assert!(listed.starts_with("The kit holds three tools."), "{listed}");
        // Marks of another page change nothing here.
        let elsewhere = |page: u32| marks(page - 1);
        assert_eq!(
            loaded
                .pages_markdown_with_marks(None, &elsewhere)
                .unwrap()
                .pages[0]
                .markdown,
            plain
        );
    }
}
