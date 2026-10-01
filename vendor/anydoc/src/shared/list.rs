//! Nested-list assembly with numbering identity.
//!
//! Frontends resolve each list paragraph to a [`ListEntry`] carrying its
//! indent level, its list identity ([`ListKey`]), and its computed effective
//! number. Assembly splits runs whenever the identity or marker kind changes
//! at a level, or an ordered sequence is non-contiguous (a restart), so the
//! renderer's `start + index` numbering reproduces the source exactly.

pub use crate::model::MarkerKind;
use crate::model::{Block, List, ListItem};

/// Identity of a resolved list at one level: which list instance the entry
/// belongs to and what marker its level uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListKey {
    /// Stable per-instance id (DOCX `numId`, RTF `\lsN`, DOC list identity
    /// `lsid`, a counter for HTML/ODF lists).
    pub instance: u64,
    pub marker: MarkerKind,
}

/// One flat, fully resolved list paragraph (plus any blocks attached to the
/// same item, like text-box content anchored in it).
#[derive(Debug)]
pub struct ListEntry {
    pub level: usize,
    pub key: ListKey,
    /// Effective item number at this entry (ignored for bullets).
    pub number: u64,
    /// Literal marker text when the source number text is not reproducible
    /// from the marker kind and number alone (composite number text).
    pub label: Option<String>,
    pub blocks: Vec<Block>,
    /// markitai: where the item's lines of text start, in twips from the
    /// left margin (its left indent), when the reader knows it; see
    /// [`continuation_level`].
    pub indent: Option<i32>,
    /// markitai: the entry is no item of its own but more paragraphs of the
    /// item open at `level` before it (see [`ListEntry::continuation`]).
    pub continues: bool,
}

impl ListEntry {
    /// markitai: paragraphs that continue the item open at `level`: they go
    /// into that item after what it holds so far (its text, its nested
    /// lists), take no number and split no list.
    pub fn continuation(level: usize, blocks: Vec<Block>) -> ListEntry {
        ListEntry {
            level,
            key: ListKey { instance: u64::MAX, marker: MarkerKind::Bullet },
            number: 0,
            label: None,
            blocks,
            indent: None,
            continues: true,
        }
    }
}

/// markitai: how far apart, in twips (5 points), two indents may be and
/// still line up.
pub const ALIGNED: i32 = 100;

/// markitai: the items of a run a following paragraph can continue: the
/// last item at each level still open at the run's end, outermost first,
/// with the indent of its text.
fn open_items(run: &[ListEntry]) -> Vec<(usize, Option<i32>)> {
    let mut open = Vec::new();
    let mut ceiling = usize::MAX;
    for entry in run.iter().rev().filter(|entry| !entry.continues) {
        if entry.level < ceiling {
            open.push((entry.level, entry.indent));
            ceiling = entry.level;
        }
    }
    open.reverse();
    open
}

/// markitai: the level at which a paragraph with no number of its own
/// continues an item of `run`, judged by where its lines start: `left`, in
/// twips. The list's text must sit further right than the body text before
/// it (`body_left`, by at least twice [`ALIGNED`]), and the paragraph at
/// least as far right as the outermost open item's text; it continues the
/// deepest open item whose text starts no further right than it does. Body
/// text set back at the body's indent continues nothing, nor does anything
/// after a list whose text starts where the body's does.
pub fn continuation_level(run: &[ListEntry], left: i32, body_left: i32) -> Option<usize> {
    let open = open_items(run);
    let first = open.first()?.1?;
    if first < body_left.saturating_add(2 * ALIGNED) {
        return None;
    }
    // Set back further than the outermost text, no item's text starts
    // within reach.
    open.iter()
        .rev()
        .find(|(_, indent)| indent.is_some_and(|indent| indent <= left.saturating_add(ALIGNED)))
        .map(|&(level, _)| level)
}

/// markitai: the level at which a paragraph numbered at `level` with a
/// marker that shows nothing (Word's `none` format, or a bullet of only
/// spaces, as pandoc writes an item's later paragraphs) continues an item of
/// `run`: the deepest open item at or above that level.
pub fn unmarked_level(run: &[ListEntry], level: usize) -> Option<usize> {
    open_items(run).iter().rev().find(|&&(open, _)| open <= level).map(|&(open, _)| open)
}

/// Pop the accumulated run of list paragraphs into list blocks; one block
/// per identity segment.
pub fn flush_list(blocks: &mut Vec<Block>, run: &mut Vec<ListEntry>) {
    let entries = std::mem::take(run);
    if entries.is_empty() {
        return;
    }
    blocks.extend(build_lists(entries));
}

/// Fold a flat run into nested lists, splitting at identity/marker changes
/// and ordered-sequence discontinuities.
fn build_lists(entries: Vec<ListEntry>) -> Vec<Block> {
    let Some(min_lvl) = entries.iter().map(|e| e.level).min() else {
        return Vec::new();
    };
    let mut out: Vec<Block> = Vec::new();
    let mut current: Option<(List, ListKey, u64)> = None; // list, key, last number

    let flush_current = |current: &mut Option<(List, ListKey, u64)>, out: &mut Vec<Block>| {
        if let Some((list, _, _)) = current.take()
            && !list.items.is_empty()
        {
            out.push(Block::List(list));
        }
    };

    let mut iter = entries.into_iter().peekable();
    while let Some(&ListEntry { level, key, number, .. }) = iter.peek() {
        if level <= min_lvl {
            let entry = iter.next().unwrap();
            // markitai: a continuation goes into the item open at its
            // level, after the item's own blocks and nested lists; with no
            // such item (which a reader does not produce) its blocks stand
            // between the lists.
            if entry.continues {
                match current.as_mut().and_then(|(list, _, _)| list.items.last_mut()) {
                    Some(item) => item.blocks.extend(entry.blocks),
                    None => {
                        flush_current(&mut current, &mut out);
                        out.extend(entry.blocks);
                    }
                }
                continue;
            }
            let split = match &current {
                Some((_, cur_key, last_number)) => {
                    *cur_key != key
                        || (key.marker.ordered() && last_number.checked_add(1) != Some(number))
                }
                None => true,
            };
            if split {
                flush_current(&mut current, &mut out);
                let list = List {
                    marker: key.marker,
                    start: if key.marker.ordered() { number } else { 1 },
                    items: Vec::new(),
                };
                current = Some((list, key, number));
            }
            let (list, _, last) = current.as_mut().unwrap();
            list.items.push(ListItem { blocks: entry.blocks, marker_label: entry.label });
            *last = number;
        } else {
            let mut sub = Vec::new();
            while iter.peek().is_some_and(|e| e.level > min_lvl) {
                sub.push(iter.next().unwrap());
            }
            let sublists = build_lists(sub);
            if sublists.is_empty() {
                continue;
            }
            if current.is_none() {
                // Sub-level content with no parent item yet: host it in an
                // anonymous item so nesting is preserved.
                let key = ListKey { instance: u64::MAX, marker: MarkerKind::Bullet };
                current = Some((
                    List { marker: MarkerKind::Bullet, start: 1, items: Vec::new() },
                    key,
                    0,
                ));
            }
            let (list, _, _) = current.as_mut().unwrap();
            if list.items.is_empty() {
                list.items.push(ListItem::default());
            }
            list.items.last_mut().unwrap().blocks.extend(sublists);
        }
    }
    flush_current(&mut current, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Inline;

    fn entry(
        level: usize,
        instance: u64,
        marker: MarkerKind,
        number: u64,
        text: &str,
    ) -> ListEntry {
        ListEntry {
            level,
            key: ListKey { instance, marker },
            number,
            label: None,
            blocks: vec![Block::Paragraph(vec![Inline::plain(text)])],
            indent: None,
            continues: false,
        }
    }

    /// markitai: an item whose text starts `indent` twips in.
    fn at(indent: i32, entry: ListEntry) -> ListEntry {
        ListEntry { indent: Some(indent), ..entry }
    }

    fn more(level: usize, text: &str) -> ListEntry {
        ListEntry::continuation(level, vec![Block::Paragraph(vec![Inline::plain(text)])])
    }

    /// markitai: each item's marker, then its blocks one level deeper.
    fn shown(blocks: &[Block], depth: usize, out: &mut Vec<String>) {
        for block in blocks {
            match block {
                Block::Paragraph(inlines) => out.push(format!(
                    "{}{}",
                    "  ".repeat(depth),
                    crate::model::inlines_to_plain_text(inlines)
                )),
                Block::List(list) => {
                    for (i, item) in list.items.iter().enumerate() {
                        let marker = list.marker.label(list.start + i as u64);
                        out.push(format!("{}{marker}", "  ".repeat(depth)));
                        shown(&item.blocks, depth + 1, out);
                    }
                }
                _ => out.push("?".into()),
            }
        }
    }

    fn lines(entries: Vec<ListEntry>) -> Vec<String> {
        let mut out = Vec::new();
        shown(&lists(entries), 0, &mut out);
        out
    }

    fn lists(entries: Vec<ListEntry>) -> Vec<Block> {
        build_lists(entries)
    }

    #[test]
    fn contiguous_numbers_stay_one_list() {
        let out = lists(vec![
            entry(0, 1, MarkerKind::Decimal, 1, "a"),
            entry(0, 1, MarkerKind::Decimal, 2, "b"),
        ]);
        assert_eq!(out.len(), 1);
        let Block::List(l) = &out[0] else { panic!() };
        assert!(l.ordered());
        assert_eq!((l.start, l.items.len()), (1, 2));
    }

    #[test]
    fn restart_splits_with_new_start() {
        let out = lists(vec![
            entry(0, 1, MarkerKind::Decimal, 1, "a"),
            entry(0, 1, MarkerKind::Decimal, 10, "restarted"),
        ]);
        assert_eq!(out.len(), 2);
        let Block::List(l) = &out[1] else { panic!() };
        assert_eq!(l.start, 10);
    }

    #[test]
    fn distinct_instances_split() {
        let out = lists(vec![
            entry(0, 1, MarkerKind::Decimal, 1, "a"),
            entry(0, 2, MarkerKind::Decimal, 1, "b"),
        ]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn marker_change_splits() {
        let out = lists(vec![
            entry(0, 1, MarkerKind::Decimal, 1, "a"),
            entry(0, 1, MarkerKind::Bullet, 0, "b"),
        ]);
        assert_eq!(out.len(), 2);
        let Block::List(l) = &out[1] else { panic!() };
        assert!(!l.ordered());
    }

    #[test]
    fn nesting_preserved() {
        let out = lists(vec![
            entry(0, 1, MarkerKind::Decimal, 1, "outer"),
            entry(1, 1, MarkerKind::LowerRoman, 1, "inner"),
            entry(0, 1, MarkerKind::Decimal, 2, "outer2"),
        ]);
        assert_eq!(out.len(), 1);
        let Block::List(l) = &out[0] else { panic!() };
        assert_eq!(l.items.len(), 2);
        assert!(matches!(l.items[0].blocks.last(), Some(Block::List(sub)) if sub.ordered()));
    }

    #[test]
    fn a_continuation_goes_into_its_item_after_its_nested_list() {
        let d = MarkerKind::Decimal;
        let shown = lines(vec![
            entry(0, 1, d, 1, "one"),
            more(0, "more of one"),
            entry(0, 1, d, 2, "two"),
            entry(1, 1, MarkerKind::Bullet, 0, "under two"),
            more(1, "more under two"),
            more(0, "more of two"),
            entry(0, 1, d, 3, "three"),
        ]);
        assert_eq!(
            shown,
            [
                "1.",
                "  one",
                "  more of one",
                "2.",
                "  two",
                "  -",
                "    under two",
                "    more under two",
                "  more of two",
                "3.",
                "  three",
            ]
        );
    }

    #[test]
    fn a_continuation_with_no_item_open_stands_between_the_lists() {
        let shown = lines(vec![more(0, "loose"), entry(0, 1, MarkerKind::Decimal, 1, "one")]);
        assert_eq!(shown, ["loose", "1.", "  one"]);
    }

    #[test]
    fn indents_decide_which_item_a_paragraph_continues() {
        let d = MarkerKind::Decimal;
        let run = vec![at(720, entry(0, 1, d, 1, "one")), at(1440, entry(1, 2, d, 1, "a"))];
        // At the nested item's text, under it; back at the outer item's
        // text (or between the two), under the outer item.
        assert_eq!(continuation_level(&run, 1440, 0), Some(1));
        assert_eq!(continuation_level(&run, 1500, 0), Some(1));
        assert_eq!(continuation_level(&run, 2880, 0), Some(1));
        assert_eq!(continuation_level(&run, 720, 0), Some(0));
        assert_eq!(continuation_level(&run, 660, 0), Some(0));
        assert_eq!(continuation_level(&run, 1100, 0), Some(0));
        // Set back to where the markers hang, or to the body: no item.
        assert_eq!(continuation_level(&run, 360, 0), None);
        assert_eq!(continuation_level(&run, 0, 0), None);
        // A list whose text starts at the body's indent continues nothing.
        assert_eq!(continuation_level(&run, 720, 720), None);
        assert_eq!(continuation_level(&run, 720, 600), None);
        assert_eq!(continuation_level(&run, 720, 500), Some(0));
        // Nor does an item whose indent the reader does not know.
        assert_eq!(continuation_level(&[entry(0, 1, d, 1, "x")], 720, 0), None);
        // A continuation already read changes nothing.
        let run = vec![at(720, entry(0, 1, d, 1, "one")), more(0, "more")];
        assert_eq!(continuation_level(&run, 720, 0), Some(0));
        // A closed sibling is not open.
        let run = vec![
            at(720, entry(0, 1, d, 1, "one")),
            at(1440, entry(1, 2, d, 1, "a")),
            at(720, entry(0, 1, d, 2, "two")),
        ];
        assert_eq!(continuation_level(&run, 1440, 0), Some(0));
    }

    #[test]
    fn an_unmarked_paragraph_continues_the_deepest_item_at_or_above_its_level() {
        let d = MarkerKind::Decimal;
        let run = vec![entry(0, 1, d, 1, "one"), entry(1, 1, d, 1, "a")];
        assert_eq!(unmarked_level(&run, 1), Some(1));
        assert_eq!(unmarked_level(&run, 4), Some(1));
        assert_eq!(unmarked_level(&run, 0), Some(0));
        assert_eq!(unmarked_level(&[entry(1, 1, d, 1, "a")], 0), None);
        assert_eq!(unmarked_level(&[], 0), None);
    }
}
