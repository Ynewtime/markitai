//! markitai: the base direction of lines holding right-to-left text, read
//! from the way the lines are aligned.
//!
//! A line's glyphs are painted in display order, and a left-to-right and a
//! right-to-left paragraph can display the same glyphs for two different
//! texts: a Latin word at the left end of a line of Persian is the first
//! word of a left-to-right paragraph and the last of a right-to-left one,
//! and a full stop at the left end closes a right-to-left sentence but opens
//! a left-to-right line. Rules P2 and P3 of the Unicode Bidirectional
//! Algorithm take a paragraph's direction from its first strong character
//! in reading order, which is the very order that is unknown here, and a
//! producer may set the direction regardless of the letters (a browser lays
//! a paragraph out left to right unless the page says otherwise, whatever
//! script it holds). The layout tells: a paragraph's lines start at an edge
//! they share and end where their words run out, so the side on which the
//! lines of a paragraph share their edges is the side they start from.
//!
//! The evidence is taken where it is unambiguous. Lines of one paragraph —
//! each linked to the nearest line below it that overlaps it horizontally,
//! is set in a size within a fifth of its own, follows within two and a
//! half em and shares its left or its right edge, unless the step is a
//! quarter wider than the step above or below it — vote pairwise: a pair sharing its right edges whose left edges
//! lie a word apart votes right to left, the mirror case left to right. A
//! paragraph reads the way its pairs vote by a majority. An indented first
//! line votes against the paragraph's last line in justified text, so such
//! a paragraph ties and keeps the direction its letters give, as before. A
//! line that is a paragraph of its own (a heading, a one-line paragraph, a
//! list item) is compared in the same way with the few lines above and
//! below it that overlap it, so a heading set flush with the column's start
//! edge reads in the direction of the text around it. Lines on which the
//! layout says nothing are left to the reader's letter-based rule
//! (`text_utils::rtl_line_base`).

use crate::types::{ItemType, TextItem};

/// Edges closer than this, in em of the smaller font of two lines, are one
/// edge.
const SHARED_EM: f32 = 0.35;
/// Edges at least this far apart, in em, are a ragged line end: about a
/// word.
const RAGGED_EM: f32 = 1.5;
/// Furthest baseline step between two lines of one paragraph, in em of the
/// larger font.
const PARAGRAPH_STEP_EM: f32 = 2.5;
/// Largest ratio between the font sizes of two lines of one paragraph.
const PARAGRAPH_SIZE_RATIO: f32 = 1.2;
/// A baseline step this much wider than the next step above or below is
/// a paragraph's end.
const PARAGRAPH_GAP_RATIO: f32 = 1.25;
/// Lines above and below that a one-line paragraph is compared with.
const NEIGHBOURS: usize = 3;
/// Furthest baseline distance of those comparisons, in em of the larger
/// font.
const NEIGHBOUR_REACH_EM: f32 = 8.0;
/// A gap between two runs of a line from which they stand in two columns
/// rather than a word apart, in em of the line's largest font.
const COLUMN_GAP_EM: f32 = 3.0;

/// One line as the decision reads it: its page, baseline, horizontal extent
/// and largest font size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LineBox {
    pub(crate) page: u32,
    pub(crate) y: f32,
    pub(crate) left: f32,
    pub(crate) right: f32,
    pub(crate) size: f32,
}

impl LineBox {
    /// The box of a line's visible, upright text runs; `None` for a line
    /// without any, holding a turned run, or with a gap of a column's width
    /// between two runs: where such a line starts says nothing about its
    /// direction (the row of a table can share the right edge of a column
    /// of figures whatever its script).
    pub(crate) fn of<'a>(items: impl IntoIterator<Item = &'a TextItem>) -> Option<Self> {
        let mut found: Option<Self> = None;
        let mut spans: Vec<(f32, f32)> = Vec::new();
        for item in items {
            if !matches!(item.item_type, ItemType::Text) || item.text.trim().is_empty() {
                continue;
            }
            if !item.is_upright() {
                return None;
            }
            let (left, right) = (item.x, item.x + item.width.max(0.0));
            spans.push((left, right));
            let line = found.get_or_insert(Self {
                page: item.page,
                y: item.line_y(),
                left,
                right,
                size: item.font_size,
            });
            line.left = line.left.min(left);
            line.right = line.right.max(right);
            line.size = line.size.max(item.font_size);
        }
        let line = found
            .filter(|line| line.size > 0.0 && line.left.is_finite() && line.right.is_finite())?;
        crate::sort::stable(&mut spans, &mut |a, b| a.0.total_cmp(&b.0));
        let mut reach = f32::NEG_INFINITY;
        for (left, right) in spans {
            if reach > f32::NEG_INFINITY && left - reach > COLUMN_GAP_EM * line.size {
                return None;
            }
            reach = reach.max(right);
        }
        Some(line)
    }
}

/// What two lines' edges say: `1` right to left (right edges shared, left
/// edges a word apart), `-1` left to right, `0` nothing.
fn vote(a: &LineBox, b: &LineBox) -> i32 {
    let em = a.size.min(b.size).max(1.0);
    let left = (a.left - b.left).abs();
    let right = (a.right - b.right).abs();
    if right <= SHARED_EM * em && left >= RAGGED_EM * em {
        1
    } else if left <= SHARED_EM * em && right >= RAGGED_EM * em {
        -1
    } else {
        0
    }
}

/// Whether two lines share their left or their right edge: two lines of
/// one paragraph do, at the edge their lines start from (and both edges
/// when it is justified, but for an indent and its last line).
fn shares_edge(a: &LineBox, b: &LineBox) -> bool {
    let em = a.size.min(b.size).max(1.0);
    (a.left - b.left).abs() <= SHARED_EM * em || (a.right - b.right).abs() <= SHARED_EM * em
}

fn overlaps(a: &LineBox, b: &LineBox) -> bool {
    a.left.max(b.left) < a.right.min(b.right)
}

fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// The base direction each line's alignment gives (`Some(true)` right to
/// left), parallel to `lines`; `None` where the layout says nothing or the
/// line has no box. The lines may come in any order and from several pages.
pub(crate) fn aligned_bases(lines: &[Option<LineBox>]) -> Vec<Option<bool>> {
    let mut order: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].is_some()).collect();
    let line = |i: usize| lines[i].expect("only lines with a box are ordered");
    crate::sort::stable(&mut order, &mut |&a, &b| {
        let (a, b) = (line(a), line(b));
        a.page.cmp(&b.page).then(b.y.total_cmp(&a.y))
    });

    // Link each line to the nearest line below it that overlaps it, when
    // the two can be lines of one paragraph, and count the votes of the
    // links of each paragraph. A step a quarter wider than the step of the
    // link above or below it is the space between two paragraphs, which
    // may be set in different directions.
    let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
    for (k, &i) in order.iter().enumerate() {
        let a = line(i);
        for &j in &order[k + 1..] {
            let b = line(j);
            if b.page != a.page || a.y - b.y > PARAGRAPH_STEP_EM * a.size.max(b.size) {
                break;
            }
            if !overlaps(&a, &b) {
                continue;
            }
            if a.size.max(b.size) <= PARAGRAPH_SIZE_RATIO * a.size.min(b.size)
                && shares_edge(&a, &b)
            {
                candidates.push((i, j, a.y - b.y));
            }
            break;
        }
    }
    let mut step_into = vec![None; lines.len()];
    let mut step_out = vec![None; lines.len()];
    for &(i, j, step) in &candidates {
        step_out[i] = Some(step);
        step_into[j] = Some(step);
    }
    let mut parent: Vec<usize> = (0..lines.len()).collect();
    let mut links: Vec<(usize, i32)> = Vec::new();
    for &(i, j, step) in &candidates {
        let wider =
            |other: Option<f32>| other.is_some_and(|other| step > PARAGRAPH_GAP_RATIO * other);
        if wider(step_into[i]) || wider(step_out[j]) {
            continue;
        }
        let (ri, rj) = (root(&mut parent, i), root(&mut parent, j));
        parent[rj] = ri;
        links.push((i, vote(&line(i), &line(j))));
    }
    let roots: Vec<usize> = (0..lines.len()).map(|i| root(&mut parent, i)).collect();
    let mut tally = vec![0i32; lines.len()];
    for &(i, v) in &links {
        tally[roots[i]] += v;
    }

    // A paragraph whose own lines say nothing — one line, or lines of one
    // width — is compared with the lines of other paragraphs around it that
    // overlap it.
    let mut around = vec![0i32; lines.len()];
    for (k, &i) in order.iter().enumerate() {
        if tally[roots[i]] != 0 {
            continue;
        }
        let a = line(i);
        let within = |j: &&usize| {
            let b = line(**j);
            b.page == a.page && (a.y - b.y).abs() <= NEIGHBOUR_REACH_EM * a.size.max(b.size)
        };
        let near = |j: &&usize| roots[**j] != roots[i] && overlaps(&a, &line(**j));
        let above = order[..k]
            .iter()
            .rev()
            .take_while(within)
            .filter(near)
            .take(NEIGHBOURS);
        let below = order[k + 1..]
            .iter()
            .take_while(within)
            .filter(near)
            .take(NEIGHBOURS);
        around[roots[i]] += above.chain(below).map(|&j| vote(&a, &line(j))).sum::<i32>();
    }

    order.iter().fold(vec![None; lines.len()], |mut bases, &i| {
        let r = roots[i];
        let score = if tally[r] != 0 { tally[r] } else { around[r] };
        bases[i] = match score.signum() {
            1 => Some(true),
            -1 => Some(false),
            _ => None,
        };
        bases
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(y: f32, left: f32, right: f32, size: f32) -> Option<LineBox> {
        Some(LineBox {
            page: 1,
            y,
            left,
            right,
            size,
        })
    }

    #[test]
    fn ragged_left_lines_read_right_to_left_and_ragged_right_ones_left_to_right() {
        // A right-aligned paragraph of three lines, the last one short.
        let rtl = [
            line(700.0, 100.0, 500.0, 12.0),
            line(683.0, 130.0, 500.0, 12.0),
            line(666.0, 380.0, 500.0, 12.0),
        ];
        assert_eq!(aligned_bases(&rtl), [Some(true); 3]);
        // The same paragraph flush left.
        let ltr = [
            line(700.0, 100.0, 500.0, 12.0),
            line(683.0, 100.0, 470.0, 12.0),
            line(666.0, 100.0, 220.0, 12.0),
        ];
        assert_eq!(aligned_bases(&ltr), [Some(false); 3]);
    }

    #[test]
    fn a_full_line_takes_the_direction_of_its_paragraph() {
        // Justified: every line full but the last, which is flush right.
        let lines = [
            line(700.0, 100.0, 500.0, 12.0),
            line(683.0, 100.0, 500.0, 12.0),
            line(666.0, 100.0, 500.0, 12.0),
            line(649.0, 100.0, 500.0, 12.0),
            line(632.0, 410.0, 500.0, 12.0),
        ];
        assert_eq!(aligned_bases(&lines), [Some(true); 5]);
    }

    #[test]
    fn an_indented_first_line_and_a_short_last_line_tie() {
        // Justified with a first-line indent: the indent votes one way, the
        // last line the other, and the layout decides nothing.
        let lines = [
            line(700.0, 124.0, 500.0, 12.0),
            line(683.0, 100.0, 500.0, 12.0),
            line(666.0, 100.0, 500.0, 12.0),
            line(649.0, 100.0, 260.0, 12.0),
        ];
        assert_eq!(aligned_bases(&lines), [None; 4]);
    }

    #[test]
    fn a_heading_reads_by_the_edge_it_shares_with_the_text_below() {
        // A heading flush left over flush-left text; a centred one over the
        // same text; a heading over right-aligned text.
        let flush_left = [
            line(740.0, 100.0, 300.0, 24.0),
            line(700.0, 100.0, 500.0, 12.0),
            line(683.0, 100.0, 460.0, 12.0),
            line(666.0, 100.0, 220.0, 12.0),
        ];
        assert_eq!(aligned_bases(&flush_left)[0], Some(false));
        let centred = [
            line(740.0, 200.0, 400.0, 24.0),
            flush_left[1],
            flush_left[2],
            flush_left[3],
        ];
        assert_eq!(aligned_bases(&centred)[0], None);
        let flush_right = [
            line(740.0, 300.0, 500.0, 24.0),
            line(700.0, 100.0, 500.0, 12.0),
            line(683.0, 140.0, 500.0, 12.0),
        ];
        assert_eq!(aligned_bases(&flush_right)[0], Some(true));
    }

    #[test]
    fn columns_lines_far_apart_and_other_pages_are_no_neighbours() {
        // Two columns: each column's paragraph decides for itself.
        let lines = [
            line(700.0, 300.0, 500.0, 12.0),
            line(700.0, 50.0, 250.0, 12.0),
            line(683.0, 360.0, 500.0, 12.0),
            line(683.0, 50.0, 180.0, 12.0),
        ];
        assert_eq!(
            aligned_bases(&lines),
            [Some(true), Some(false), Some(true), Some(false)]
        );
        // A line far below, or on the next page, is not of the paragraph.
        let apart = [
            line(700.0, 100.0, 500.0, 12.0),
            line(600.0, 100.0, 300.0, 12.0),
        ];
        assert_eq!(aligned_bases(&apart), [None, None]);
        let mut next_page = line(683.0, 100.0, 300.0, 12.0);
        next_page.as_mut().unwrap().page = 2;
        assert_eq!(
            aligned_bases(&[line(700.0, 100.0, 500.0, 12.0), next_page]),
            [None, None]
        );
        // A line of a quite different size is a paragraph of its own.
        let sizes = [
            line(700.0, 100.0, 500.0, 18.0),
            line(683.0, 100.0, 300.0, 12.0),
        ];
        assert_eq!(aligned_bases(&sizes), [Some(false), Some(false)]);
        assert_eq!(
            aligned_bases(&[line(700.0, 100.0, 500.0, 12.0), None]),
            [None, None]
        );
    }

    #[test]
    fn paragraphs_set_apart_or_sharing_no_edge_are_not_linked() {
        // A one-line right-to-left paragraph (flush right) a paragraph's
        // spacing above a left-to-right one whose first line is full: the
        // two share the right edge but stand a wider step apart than the
        // lines of the paragraph below, so the left-to-right paragraph's
        // votes do not reach the line above, which the full line's right
        // edge sets apart from its own ragged left.
        let lines = [
            line(649.0, 127.0, 540.0, 16.0),
            line(612.0, 72.0, 538.7, 16.0),
            line(591.0, 72.0, 124.0, 16.0),
        ];
        assert_eq!(
            aligned_bases(&lines),
            [Some(true), Some(false), Some(false)]
        );
        // At the paragraph's own step they are one paragraph, whose
        // contradicting votes tie.
        let one = [
            lines[0],
            line(628.0, 72.0, 538.7, 16.0),
            line(607.0, 72.0, 124.0, 16.0),
        ];
        assert_eq!(aligned_bases(&one), [None; 3]);
        // Lines that share neither edge are no paragraph (a centred block).
        let centred = [
            line(700.0, 200.0, 400.0, 12.0),
            line(683.0, 150.0, 450.0, 12.0),
        ];
        assert_eq!(aligned_bases(&centred), [None, None]);
        // A centred line over a flush-left paragraph is no line of it, and
        // takes none of its votes.
        let over = [
            line(700.0, 200.0, 400.0, 12.0),
            line(683.0, 100.0, 500.0, 12.0),
            line(666.0, 100.0, 300.0, 12.0),
        ];
        assert_eq!(aligned_bases(&over), [None, Some(false), Some(false)]);
        // Nor is a heading of another size, though at the paragraph's step:
        // flush left over a right-aligned paragraph, it reads by the edge
        // it shares with the paragraph's full line.
        let heading = [
            line(717.0, 100.0, 300.0, 24.0),
            line(700.0, 100.0, 500.0, 12.0),
            line(683.0, 200.0, 500.0, 12.0),
            line(666.0, 300.0, 500.0, 12.0),
        ];
        assert_eq!(
            aligned_bases(&heading),
            [Some(false), Some(true), Some(true), Some(true)]
        );
    }

    #[test]
    fn interleaved_columns_keep_their_own_paragraphs() {
        // Two columns whose baselines interleave: a justified left-to-right
        // paragraph on the left, a ragged right-to-left one on the right.
        // Each line's paragraph is found below it in its own column, so the
        // full lines of the justified paragraph read as its last line does.
        let mut lines = Vec::new();
        for k in 0..6 {
            let right = if k == 5 { 120.0 } else { 250.0 };
            lines.push(line(700.0 - 17.0 * k as f32, 50.0, right, 12.0));
            let left = [320.0, 350.0, 310.0, 360.0, 330.0, 450.0][k];
            lines.push(line(691.0 - 17.0 * k as f32, left, 500.0, 12.0));
        }
        let bases = aligned_bases(&lines);
        for (k, base) in bases.iter().enumerate() {
            assert_eq!(*base, Some(k % 2 == 1), "line {k}");
        }
    }

    #[test]
    fn a_line_of_runs_a_column_apart_has_no_box() {
        // The row of a table: its runs stand columns apart.
        let row = [
            item("1.2.0", 100.0, 30.0, 12.0),
            item("x", 200.0, 30.0, 12.0),
        ];
        assert_eq!(LineBox::of(&row), None);
        let words = [item("a", 100.0, 30.0, 12.0), item("b", 160.0, 30.0, 12.0)];
        assert!(LineBox::of(&words).is_some());
    }

    pub(crate) fn item(text: &str, x: f32, width: f32, size: f32) -> TextItem {
        TextItem {
            text: text.into(),
            x,
            y: 700.0,
            width,
            height: size,
            font: "TestFont".into(),
            font_tag: String::new(),
            legacy_symbol_rewrite: false,
            font_size: size,
            page: 1,
            is_bold: false,
            is_italic: false,
            font_weight: None,
            bold_source: None,
            fixed_pitch: None,
            fill_color: None,
            stroke_color: None,
            render_mode: None,
            is_underline: false,
            is_strikeout: false,
            rotation: 0.0,
            advance_known: true,
            item_type: ItemType::Text,
            mcid: None,
            baseline_shift: 0.0,
        }
    }

    #[test]
    fn a_box_holds_the_visible_upright_runs() {
        let items = [
            item("a", 120.0, 30.0, 12.0),
            item("b", 100.0, 10.0, 14.0),
            item(" ", 300.0, 5.0, 30.0),
        ];
        let line = LineBox::of(&items).unwrap();
        assert_eq!(
            (line.left, line.right, line.size, line.y),
            (100.0, 150.0, 14.0, 700.0)
        );
        let mut turned = items.clone();
        turned[0].rotation = 90.0;
        assert_eq!(LineBox::of(&turned), None);
        assert_eq!(LineBox::of(&items[2..]), None);
    }
}
