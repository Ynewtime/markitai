//! markitai: lists typed by hand.
//!
//! Many documents never use their word processor's lists: each item is a
//! paragraph that starts with a bullet character ("•", "–", Word's Symbol
//! bullet) or a number ("1.", "a)", "一、") and a tab or a space, sometimes
//! after a tab of indentation. macOS `textutil` saves every HTML list that
//! way in Word documents ("Tab • Tab text", "Tab 1 Tab text"), and text
//! pasted from elsewhere arrives so. Read as text, such a list was a run of
//! paragraphs starting with "•".
//!
//! The readers note, while they read, which top-level paragraphs are plain
//! (not a heading, list item, table cell, text box, styled container or
//! code) and how they are indented. [`TypedLists::lists`] then turns each
//! run of consecutive plain paragraphs that read as items into a list:
//!
//! - an item starts, after any spaces or tabs, with a marker and then a tab
//!   or a space (a bullet such as "•", or a label closed the CJK way, may
//!   touch its text), and has a letter, digit, image, formula or note
//!   after it (`* * *` is no item);
//! - a bullet that is only ever a bullet (•, ◦, ▪, ➢, ・, Word's Symbol and
//!   Wingdings bullets, a ballot box, a tick) makes an item by itself;
//! - a character that also starts sentences ("-", "–", "—", "*", "+", "o",
//!   an arrow) makes one only when a tab follows it, under an item of the
//!   run that is one, or when at least two items at the same level start
//!   with it and no other such character starts one there without a tab
//!   (`+ fast`, `- loud` say something). An em dash, which opens dialogue
//!   in the languages that set dialogue with dashes, never makes one that
//!   way, nor do hyphens and en dashes when an item reads as dialogue: it
//!   ends with "?", "!", "…" or a quotation mark, or a dash after a word's
//!   closing punctuation sets the narration apart (`– Oui, – dit-elle.`);
//! - a number makes an item in a run of at least two numbered at the same
//!   level that count up by one in the same form (`1.` `2.`; `a)` `b)`;
//!   `(i)` `(ii)`; `一、` `二、`; deeper items between them belong to
//!   them); a decimal number followed by a tab (`1<Tab>`, as `textutil`
//!   writes an `ol`) makes one by itself, unless a numbered paragraph
//!   stands beside it at its level (`3<Tab>Apples`, `12<Tab>Pears`) or its
//!   text is all bold (a numbered heading typed by hand). A number before a
//!   space (`3 apples`), a capital with a full stop before a space (`A.
//!   Smith`), a year (more than three digits) or an outline number (`1.2`)
//!   never starts an item;
//! - the level comes from where the marker sits: the paragraph's left and
//!   first-line indent, then each leading tab (to the hanging indent, else
//!   the next half-inch stop) and space. An item set further right than the
//!   one before it is a level under it, one set back returns to the level
//!   open at its place, and markers within five points of each other are at
//!   one place; no item is more than one level below the item before it;
//! - a decimal number with a full stop, a closing parenthesis or nothing
//!   after it is an ordered list counting from the first item's number;
//!   other labels are kept as written (`- a) …`, `- （一） …`), bullets
//!   become `- `, a ballot box becomes a task-list checkbox and a tick or
//!   cross stays at the start of the item's text.
//!
//! A paragraph that is a row of a table set with tab stops, or that the
//! heading guess took, is no item; a paragraph opening with a bullet that is
//! only ever a bullet is never taken for a heading, and the readers do not
//! take one set all in a monospaced font for a line of code
//! ([`opens_with_bullet`]).

use crate::model::{Block, Inline, MarkerKind, inlines_are_empty, inlines_to_plain_text};
use crate::shared::list::{ALIGNED, ListEntry, ListKey, continuation_level, flush_list};
use std::ops::Range;

/// Where a paragraph's lines start, in twips (twentieths of a point): the
/// indent of its lines from the left margin and the first line's offset
/// from that (negative for a hanging indent).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Indent {
    pub left: i32,
    pub first_line: i32,
}

/// A length in twips: a number with a unit (`0.5in`, `1.27cm`, `36pt`; ODF
/// and strict OOXML), or a bare number of twips (Word).
pub fn twips(length: &str) -> Option<i32> {
    let length = length.trim();
    let split = length.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(length.len());
    let number: f64 = length[..split].trim().parse().ok()?;
    let per_unit = match &length[split..] {
        "" => 1.0,
        "pt" => 20.0,
        "pc" => 240.0,
        "in" => 1440.0,
        "cm" => 1440.0 / 2.54,
        "mm" => 144.0 / 2.54,
        "px" => 15.0,
        _ => return None,
    };
    let twips = (number * per_unit).round();
    (f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&twips).then_some(twips as i32)
}

/// A default tab stop, in twips: every half inch.
const DEFAULT_STOP: i32 = 720;

/// The width a leading space takes, in twips: a quarter of a 12-point em.
const SPACE: i32 = 60;

/// Markers set this close, in twips, share a level.
const SAME_LEVEL: i32 = 100;

/// Most leading characters looked at for the marker of an item.
const PREFIX: usize = 48;

/// The plain top-level paragraphs of a document and their indents,
/// gathered while it is read.
#[derive(Debug, Default)]
pub struct TypedLists {
    paragraphs: Vec<(usize, Indent)>,
}

impl TypedLists {
    /// `blocks[index]` is a plain top-level paragraph indented `indent`.
    pub fn paragraph(&mut self, index: usize, indent: Indent) {
        self.paragraphs.push((index, indent));
    }

    /// The plain paragraphs that start with a bullet that is only ever a
    /// bullet: none of them is a heading.
    pub(crate) fn bullets(&self, blocks: &[Block]) -> Vec<usize> {
        self.paragraphs
            .iter()
            .filter(|&&(index, _)| {
                matches!(blocks.get(index), Some(Block::Paragraph(inlines)) if opens_with_bullet(inlines))
            })
            .map(|&(index, _)| index)
            .collect()
    }

    /// Each run of the paragraphs that reads as a list typed by hand: the
    /// blocks it takes and the list blocks that replace them. A paragraph
    /// `taken` names (a row of a table) is no item.
    pub(crate) fn lists(
        &self,
        blocks: &[Block],
        taken: impl Fn(usize) -> bool,
    ) -> Vec<(Range<usize>, Vec<Block>)> {
        let mut found = Vec::new();
        let mut run = Run::default();
        // markitai: where the last paragraph of body text starts its lines.
        let mut body_left = 0;
        for &(index, indent) in &self.paragraphs {
            let paragraph = match blocks.get(index) {
                Some(Block::Paragraph(inlines)) if !taken(index) => Some(inlines),
                _ => None,
            };
            let item = paragraph.and_then(|inlines| item(inlines)).map(|item| Item {
                index,
                indent,
                ..item
            });
            let next = run.last_index().is_some_and(|last| last + 1 == index);
            // markitai: a paragraph right after an item, with words and no
            // marker, set in as far as an item's text continues it.
            if item.is_none()
                && next
                && let Some(inlines) = paragraph
                && has_words(inlines)
                && run.may_continue(indent.left)
            {
                run.more.push((run.items.len(), More { index, indent, content: inlines.clone() }));
                continue;
            }
            if run.last_index().is_some() && (item.is_none() || !next) {
                close(std::mem::take(&mut run), &mut found);
            }
            match item {
                Some(item) => {
                    if run.items.is_empty() {
                        run.body_left = body_left;
                    }
                    run.items.push(item);
                }
                None => body_left = indent.left,
            }
        }
        close(run, &mut found);
        found
    }
}

/// Whether a paragraph opens with a bullet that is only ever a bullet, a
/// ballot box or a tick, then its text: an item of a list typed by hand
/// even when it is set all in a monospaced font (`• npm test`), which is
/// otherwise a line of code.
pub fn opens_with_bullet(inlines: &[Inline]) -> bool {
    item(inlines).is_some_and(
        |item| matches!(item.marker, Marker::Bullet { kind, .. } if kind != Bullet::Weak),
    )
}

/// What a bullet character says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bullet {
    /// Only ever a bullet (•, ◦, ▪, ➢, a symbol font's private-use code).
    Only,
    /// A ballot box, checked or not: a task-list checkbox.
    Box(bool),
    /// A tick or a cross: a bullet that says something, kept in the text.
    Kept,
    /// A character that also starts sentences: a dash, `*`, `+`, `o`, an
    /// arrow.
    Weak,
}

fn bullet(c: char) -> Option<Bullet> {
    Some(match c {
        '•' | '◦' | '▪' | '▫' | '●' | '○' | '■' | '□' | '◆' | '◇' | '♦' | '❖' | '‣' | '⁃' | '∙'
        | '·' | '・' | '･' | '◘' | '⦁' | '➢' | '➤' | '►' | '▶' | '▸' | '▹' | '❑' | '❒' => {
            Bullet::Only
        }
        // Symbol and Wingdings codes in the private-use block show their
        // font's glyph: in Word documents almost always a bullet.
        '\u{f000}'..='\u{f0ff}' => Bullet::Only,
        '☐' => Bullet::Box(false),
        '☑' | '☒' => Bullet::Box(true),
        '✓' | '✔' | '✗' | '✘' => Bullet::Kept,
        '-' | '–' | '—' | '*' | '+' | 'o' | '→' | '⇒' | '➔' | '➜' => Bullet::Weak,
        _ => return None,
    })
}

/// How a number is counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Count {
    Decimal,
    LowerAlpha,
    UpperAlpha,
    LowerRoman,
    UpperRoman,
    Cjk,
    Circled,
}

/// A number label's punctuation: an opening parenthesis and what closes it
/// (`None` for a bare number or a circled digit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Form {
    open: bool,
    close: Option<char>,
}

#[derive(Debug, Clone, PartialEq)]
enum Marker {
    Bullet {
        c: char,
        kind: Bullet,
    },
    /// The ways the label reads (`i` is a letter or a Roman one), its form
    /// and its text.
    Number {
        readings: Vec<(Count, u64)>,
        form: Form,
        label: String,
    },
}

/// A paragraph that starts like an item.
#[derive(Debug, Clone)]
struct Item {
    index: usize,
    indent: Indent,
    marker: Marker,
    /// The whitespace before the marker.
    leading: Vec<char>,
    /// Whether a tab separates the marker from the text.
    tab_after: bool,
    /// The paragraph's content without its marker.
    content: Vec<Inline>,
}

impl Item {
    /// Where the marker sits, in twips from the left margin.
    fn position(&self) -> i32 {
        let Indent { left, first_line } = self.indent;
        let mut position = left.saturating_add(first_line);
        for &c in &self.leading {
            position = if c != '\t' {
                position.saturating_add(SPACE)
            } else if position < left {
                // A hanging indent is the first stop.
                left
            } else {
                (position.div_euclid(DEFAULT_STOP) + 1).saturating_mul(DEFAULT_STOP)
            };
        }
        position
    }
}

/// The leading text of a paragraph: the characters of its first text
/// inlines (a tab is one; a bookmark between them is nothing), up to
/// [`PREFIX`].
fn prefix(inlines: &[Inline]) -> Vec<char> {
    let mut chars = Vec::new();
    for inline in inlines {
        match inline {
            Inline::Text { text, .. } => chars.extend(text.chars().take(PREFIX - chars.len())),
            Inline::Anchor(_) => {}
            _ => break,
        }
        if chars.len() >= PREFIX {
            break;
        }
    }
    chars
}

/// The paragraph as an item, if it starts like one (its index and indent
/// are filled in by the caller).
fn item(inlines: &[Inline]) -> Option<Item> {
    let chars = prefix(inlines);
    let start = chars.iter().position(|c| !c.is_whitespace())?;
    let (marker, end) = marker(&chars, start)?;
    let after = chars[end..].iter().take_while(|c| c.is_whitespace()).count();
    let separated = after > 0;
    let tab_after = chars[end..end + after].contains(&'\t');
    let touching_allowed = match &marker {
        Marker::Bullet { kind, c } => {
            *kind != Bullet::Weak && chars.get(end).is_none_or(|next| next != c)
        }
        Marker::Number { form, readings, .. } => {
            matches!(form.close, Some('、' | '）' | '．'))
                || (form.open && form.close.is_some())
                || readings.iter().any(|&(count, _)| count == Count::Circled)
        }
    };
    if !separated && !touching_allowed {
        return None;
    }
    let mut marker = marker;
    if let Marker::Number { form, readings, .. } = &mut marker {
        // A bare number labels an item only before a tab (`3 apples`
        // counts apples), and a capital before a full stop and a space is
        // an initial (`A. Smith`), though `I.` may still count in Roman.
        if form.close.is_none()
            && !readings.iter().any(|&(c, _)| c == Count::Circled)
            && (!tab_after || readings.iter().any(|&(c, _)| c != Count::Decimal))
        {
            return None;
        }
        if form.close == Some('.') && !form.open && !tab_after {
            readings.retain(|&(c, _)| c != Count::UpperAlpha);
            if readings.is_empty() {
                return None;
            }
        }
    }
    let mut content = without_prefix(inlines, end + after);
    if let Marker::Bullet { kind: Bullet::Kept, c } = marker {
        content.insert(0, Inline::plain(format!("{c} ")));
    }
    if !has_words(&content) {
        return None;
    }
    if let Marker::Bullet { kind: Bullet::Box(checked), .. } = marker {
        // Renderers set the box apart from the text after it.
        content.insert(0, Inline::Checkbox(checked));
    }
    Some(Item {
        index: 0,
        indent: Indent::default(),
        marker,
        leading: chars[..start].to_vec(),
        tab_after,
        content,
    })
}

/// The marker starting at `chars[start]` and where it ends.
fn marker(chars: &[char], start: usize) -> Option<(Marker, usize)> {
    let c = chars[start];
    let next = chars.get(start + 1);
    match bullet(c) {
        // `o` is a bullet only on its own, not the start of a word or `o)`.
        // (Other weak marks need the space after them that `--` and `**`
        // lack.)
        Some(Bullet::Weak) if c == 'o' && next.is_some_and(|n| !n.is_whitespace()) => {
            number(chars, start)
        }
        Some(kind) => Some((Marker::Bullet { c, kind }, start + 1)),
        None => number(chars, start),
    }
}

/// A number label at `chars[start]`: `1.`, `2)`, `(3)`, `a.`, `iv)`,
/// `一、`, `（二）`, `①`, a bare `1`.
fn number(chars: &[char], start: usize) -> Option<(Marker, usize)> {
    let mut i = start;
    let open = matches!(chars.get(i), Some('(' | '（'));
    if open {
        i += 1;
    }
    let body_start = i;
    while i < chars.len() && !is_label_end(chars[i]) && i - body_start < 6 {
        i += 1;
    }
    let body: String = chars[body_start..i].iter().collect();
    let readings = readings(&body);
    if readings.is_empty() {
        return None;
    }
    let close = match chars.get(i) {
        Some(&c @ (')' | '）')) => Some(c),
        Some(&c @ ('.' | '、' | '．')) if !open => Some(c),
        _ if open => return None,
        _ => None,
    };
    if close.is_some() {
        i += 1;
    }
    // `1.2`, `1.5 kg` and `a.m.` have no space after the stop, which
    // `item` requires.
    let label: String = chars[start..i].iter().collect();
    Some((Marker::Number { readings, form: Form { open, close }, label }, i))
}

fn is_label_end(c: char) -> bool {
    c.is_whitespace() || matches!(c, '.' | ')' | '）' | '、' | '．' | '(' | '（')
}

/// The ways a label's characters count: a decimal of at most three digits
/// (half- or full-width), one letter (`i`, `v`, `x` also Roman), a Roman
/// numeral, Chinese numerals or a circled digit.
fn readings(body: &str) -> Vec<(Count, u64)> {
    let digits: Option<String> = body
        .chars()
        .map(|c| match c {
            '0'..='9' => Some(c),
            '０'..='９' => char::from_u32(c as u32 - '０' as u32 + '0' as u32),
            _ => None,
        })
        .collect();
    let count = body.chars().count();
    if let Some(digits) = digits.filter(|_| (1..=3).contains(&count)) {
        return digits.parse().map(|n| vec![(Count::Decimal, n)]).unwrap_or_default();
    }
    let mut found = Vec::new();
    if count == 1 {
        let c = body.chars().next().unwrap_or_default();
        if let Some(n) = ('①'..='⑳').contains(&c).then(|| u64::from(c as u32 - '①' as u32 + 1))
        {
            return vec![(Count::Circled, n)];
        }
        if c.is_ascii_lowercase() {
            found.push((Count::LowerAlpha, u64::from(c as u32 - 'a' as u32 + 1)));
        } else if c.is_ascii_uppercase() {
            found.push((Count::UpperAlpha, u64::from(c as u32 - 'A' as u32 + 1)));
        }
    }
    if let Some(n) = roman(body) {
        let count = if body.starts_with(|c: char| c.is_ascii_lowercase()) {
            Count::LowerRoman
        } else {
            Count::UpperRoman
        };
        found.push((count, n));
    }
    if let Some(n) = cjk(body) {
        found.push((Count::Cjk, n));
    }
    found
}

/// The value of a Roman numeral up to 39 in one case (`iv`, `XII`).
fn roman(text: &str) -> Option<u64> {
    let lower = text.to_ascii_lowercase();
    if text.is_empty()
        || text.len() > 6
        || !(text == lower || text == text.to_ascii_uppercase())
        || !lower.chars().all(|c| matches!(c, 'i' | 'v' | 'x'))
    {
        return None;
    }
    (1..40).find(|&n| MarkerKind::LowerRoman.ordinal(n) == lower)
}

/// The value of one to three Chinese numerals (`三`, `十二`, `二十`).
fn cjk(text: &str) -> Option<u64> {
    let digit = |c: char| "〇一二三四五六七八九".chars().position(|d| d == c).map(|n| n as u64);
    let chars: Vec<char> = text.chars().collect();
    match chars[..] {
        [one] => digit(one).filter(|&n| n > 0).or((one == '十').then_some(10)),
        ['十', one] => Some(10 + digit(one).filter(|&n| n > 0)?),
        [tens, '十'] => Some(10 * digit(tens).filter(|&n| n > 1)?),
        [tens, '十', one] => {
            Some(10 * digit(tens).filter(|&n| n > 1)? + digit(one).filter(|&n| n > 0)?)
        }
        _ => None,
    }
}

/// `inlines` without their first `count` characters of text and the
/// whitespace then at their start.
fn without_prefix(inlines: &[Inline], count: usize) -> Vec<Inline> {
    let mut left = count;
    let mut out = Vec::with_capacity(inlines.len());
    let mut leading = true;
    for inline in inlines {
        match inline {
            Inline::Text { text, style } if leading => {
                let skip = text.chars().count().min(left);
                left -= skip;
                let rest: String = text.chars().skip(skip).collect();
                let rest = if left == 0 { rest.trim_start().to_string() } else { rest };
                if !rest.is_empty() {
                    leading = false;
                    out.push(Inline::Text { text: rest, style: *style });
                }
            }
            Inline::Anchor(_) => out.push(inline.clone()),
            other => {
                leading = false;
                out.push(other.clone());
            }
        }
    }
    out
}

/// Whether an item's content holds words: a letter or digit, or an image,
/// a formula or a note.
fn has_words(inlines: &[Inline]) -> bool {
    !inlines_are_empty(inlines)
        && (inlines_to_plain_text(inlines).chars().any(char::is_alphanumeric)
            || inlines.iter().any(|inline| {
                matches!(inline, Inline::Image { .. } | Inline::Math(_) | Inline::NoteRef(_))
            }))
}

/// Levels for the items of a run, as an outline indents: an item set
/// further right than the one before it opens a level under it, one set
/// back to an open level's position returns to that level; markers within
/// [`SAME_LEVEL`] of each other are at one position. No item is more than a
/// level below the one before it.
fn levels(items: &[Item]) -> Vec<usize> {
    levels_at(&items.iter().map(Item::position).collect::<Vec<_>>())
}

/// The levels of items whose markers sit at `positions` (see [`levels`]).
fn levels_at(positions: &[i32]) -> Vec<usize> {
    let mut open: Vec<i32> = Vec::new();
    positions
        .iter()
        .map(|&position| {
            while open.last().is_some_and(|&last| last - position >= SAME_LEVEL) {
                open.pop();
            }
            if open.last().is_none_or(|&last| position - last >= SAME_LEVEL) {
                open.push(position);
            }
            open.len() - 1
        })
        .collect()
}

/// Which items of a run read as items (see the module documentation).
fn accepted(items: &[Item]) -> Vec<bool> {
    let levels = levels(items);
    let mut accepted = vec![false; items.len()];
    // Numbers: runs at one level counting up by one in one form.
    for (i, item) in items.iter().enumerate() {
        let Marker::Number { readings, form, .. } = &item.marker else { continue };
        let chain = chain(items, &levels, i);
        let single_tab = chain.len() == 1
            && item.tab_after
            && !form.open
            && readings.iter().any(|&(c, _)| c == Count::Decimal)
            && !numbered_neighbour(items, &levels, i)
            && !all_bold(&item.content);
        if chain.len() >= 2 || single_tab {
            for &k in &chain {
                accepted[k] = true;
            }
        }
    }
    for (k, item) in items.iter().enumerate() {
        if matches!(item.marker, Marker::Bullet { kind, .. } if kind != Bullet::Weak) {
            accepted[k] = true;
        }
    }
    // Weak bullets: a tab after, a sibling with the same mark, or a parent.
    for (k, item) in items.iter().enumerate() {
        let Marker::Bullet { c, kind: Bullet::Weak } = item.marker else { continue };
        let untabbed = |j: &usize| {
            levels[*j] == levels[k]
                && !items[*j].tab_after
                && matches!(items[*j].marker, Marker::Bullet { kind: Bullet::Weak, .. })
        };
        let siblings: Vec<usize> = (0..items.len())
            .filter(|&j| {
                levels[j] == levels[k]
                    && matches!(items[j].marker, Marker::Bullet { c: other, .. } if other == c)
            })
            .collect();
        // Marks that differ side by side say something (`+ fast`, `- loud`).
        let mixed = (0..items.len())
            .filter(untabbed)
            .any(|j| !matches!(items[j].marker, Marker::Bullet { c: other, .. } if other == c));
        let dialogue = matches!(c, '-' | '–')
            && siblings.iter().any(|&j| !items[j].tab_after && reads_as_dialogue(&items[j]));
        // The em dash opens dialogue in most languages that set it apart.
        let by_siblings = c != '—' && siblings.len() >= 2 && !mixed && !dialogue;
        let parent = (0..k).rev().find(|&j| levels[j] < levels[k]).is_some_and(|j| accepted[j]);
        accepted[k] = item.tab_after || by_siblings || parent;
    }
    accepted
}

/// The items at `items[first]`'s level counting on from it by one in its
/// form: deeper items between them belong to them; a shallower item or a
/// bullet at the level ends the count.
fn chain(items: &[Item], levels: &[usize], first: usize) -> Vec<usize> {
    let Marker::Number { readings, form, .. } = &items[first].marker else {
        return Vec::new();
    };
    let level = levels[first];
    let mut chain = vec![first];
    let mut readings = readings.clone();
    for k in first + 1..items.len() {
        if levels[k] > level {
            continue;
        }
        if levels[k] < level {
            break;
        }
        let Marker::Number { readings: next, form: next_form, .. } = &items[k].marker else {
            break;
        };
        let counting: Vec<(Count, u64)> = next
            .iter()
            .copied()
            .filter(|&(count, n)| readings.iter().any(|&(c, m)| c == count && m + 1 == n))
            .collect();
        if next_form != form || counting.is_empty() {
            break;
        }
        readings = counting;
        chain.push(k);
    }
    chain
}

/// Whether the item before or after `items[k]` at its level (deeper items
/// between them aside) is numbered too: numbers side by side that do not
/// count up are no list (`3<Tab>Apples`, `12<Tab>Pears`).
fn numbered_neighbour(items: &[Item], levels: &[usize], k: usize) -> bool {
    let numbered = |j: usize| matches!(items[j].marker, Marker::Number { .. });
    let sibling = |j: &usize| levels[*j] <= levels[k];
    let before = (0..k).rev().find(sibling).filter(|&j| levels[j] == levels[k]);
    let after = (k + 1..items.len()).find(sibling).filter(|&j| levels[j] == levels[k]);
    before.is_some_and(numbered) || after.is_some_and(numbered)
}

/// Whether all of an item's text is bold: a numbered heading typed by hand
/// (`1<Tab>Introduction`), when it stands alone.
fn all_bold(inlines: &[Inline]) -> bool {
    inlines.iter().all(|inline| match inline {
        Inline::Text { text, style } => style.bold || text.trim().is_empty(),
        Inline::Link { content, .. } => all_bold(content),
        _ => true,
    })
}

/// Whether a dash item reads as a line of dialogue: it ends as speech does,
/// or a dash after a word's closing punctuation sets the narration apart
/// (`— Oui, — dit-il.`; a dash after a word, `Paris – the capital`, is a
/// list's separator).
fn reads_as_dialogue(item: &Item) -> bool {
    let text = inlines_to_plain_text(&item.content);
    let words: Vec<&str> = text.split_whitespace().collect();
    words.last().is_some_and(|last| last.ends_with(['?', '!', '…', '»', '«', '”', '“', '"', '\'']))
        || words.windows(2).any(|pair| {
            pair[0].ends_with([',', '.', '!', '?', '…', ';'])
                && pair[1].starts_with(['-', '–', '—'])
                && !pair[1].starts_with("--")
        })
        || speech_incise(&text)
}

/// markitai: speech verbs as an incise after a line of dialogue gives them
/// (`– Je pars, dit Paul.`, `– Ya voy, dijo.`, `– Vi ses, sa han.`), in
/// the languages that set dialogue with dashes, and English; words that
/// also say other things in a list (`added`, `called`, `fit`) are left out.
const SPEECH_VERBS: &[&str] = &[
    "said",
    "asked",
    "replied",
    "answered",
    "cried",
    "shouted",
    "whispered",
    "muttered",
    "murmured",
    "exclaimed",
    "sighed",
    "dit",
    "demanda",
    "répondit",
    "ajouta",
    "murmura",
    "cria",
    "reprit",
    "souffla",
    "lança",
    "s'écria",
    "s'exclama",
    "déclara",
    "avoua",
    "dijo",
    "preguntó",
    "respondió",
    "contestó",
    "añadió",
    "exclamó",
    "gritó",
    "murmuró",
    "susurró",
    "disse",
    "perguntou",
    "respondeu",
    "retrucou",
    "acrescentou",
    "gritou",
    "exclamou",
    "murmurou",
    "chiese",
    "rispose",
    "aggiunse",
    "esclamò",
    "gridò",
    "mormorò",
    "sussurrò",
    "сказал",
    "сказала",
    "спросил",
    "спросила",
    "ответил",
    "ответила",
    "крикнул",
    "крикнула",
    "прошептал",
    "прошептала",
    "добавил",
    "добавила",
    "сказав",
    "спитав",
    "спитала",
    "відповів",
    "відповіла",
    "powiedział",
    "powiedziała",
    "zapytał",
    "zapytała",
    "odpowiedział",
    "odpowiedziała",
    "dodał",
    "dodała",
    "krzyknął",
    "krzyknęła",
    "szepnął",
    "szepnęła",
    "řekl",
    "řekla",
    "zeptal",
    "zeptala",
    "odpověděl",
    "odpověděla",
    "sa",
    "sade",
    "sagde",
    "spurte",
    "spurgte",
    "frågade",
    "svarade",
    "svarte",
    "svarede",
    "ropade",
    "sanoi",
    "kysyi",
    "vastasi",
    "huusi",
    "sagte",
    "fragte",
    "antwortete",
    "rief",
    "flüsterte",
];

/// markitai: personal pronouns a speech verb follows in an incise (`he
/// said`, `she asked`).
const SPEECH_PRONOUNS: &[&str] = &["he", "she", "i", "they", "we", "you", "it"];

/// markitai: whether a clause after a comma is a speech incise: at most
/// three words, the first a speech verb or a French inversion (`dit-il`,
/// `répondit-elle`, `dis-je`, `a-t-on`), or a pronoun and then a speech
/// verb. On the 21,825 dash and `<li>` list items of the local Markdown and
/// HTML (crate READMEs and changelogs, these docs, the reference's pages)
/// no clause reads so; written dialogue in fourteen languages mostly does.
fn speech_incise(text: &str) -> bool {
    let verb = |word: &str| SPEECH_VERBS.contains(&word);
    text.split(',').skip(1).any(|after| {
        let clause = after.trim_start().trim_start_matches(['-', '–', '—']);
        let clause = clause.split(['.', '!', '?', ';', '…']).next().unwrap_or("");
        let words: Vec<String> = clause.split_whitespace().map(str::to_lowercase).collect();
        let [first, rest @ ..] = &words[..] else { return false };
        if rest.len() > 2 {
            return false;
        }
        let inverted =
            ["-t-il", "-t-elle", "-t-on", "-il", "-elle", "-ils", "-elles", "-on", "-je"]
                .iter()
                .find_map(|pronoun| first.strip_suffix(pronoun))
                .is_some_and(|verb| !verb.is_empty() && verb.chars().all(char::is_alphabetic));
        verb(first.split('-').next().unwrap_or(""))
            || inverted
            || (SPEECH_PRONOUNS.contains(&first.as_str()) && rest.first().is_some_and(|w| verb(w)))
    })
}

/// markitai: a paragraph with no marker right after an item, set in as far
/// as an item's text: more of that item.
#[derive(Debug, Clone)]
struct More {
    index: usize,
    indent: Indent,
    content: Vec<Inline>,
}

/// markitai: consecutive paragraphs that start like items, the paragraphs
/// that may continue them (each after the number of items before it), and
/// where the body text before them starts its lines.
#[derive(Debug, Default)]
struct Run {
    items: Vec<Item>,
    more: Vec<(usize, More)>,
    body_left: i32,
}

impl Run {
    /// The block index of the run's last paragraph.
    fn last_index(&self) -> Option<usize> {
        let item = self.items.last().map(|item| item.index);
        let more = self.more.last().map(|(_, more)| more.index);
        item.max(more)
    }

    /// Whether a paragraph starting its lines `left` twips in may continue
    /// an item of the run (see [`continuation_level`], which decides once
    /// the items' levels are known).
    fn may_continue(&self, left: i32) -> bool {
        self.items.iter().filter_map(|item| text_indent(item.indent)).any(|text| {
            text >= self.body_left.saturating_add(2 * ALIGNED) && left >= text - ALIGNED
        })
    }
}

/// markitai: where an item's text lines start, when a hanging indent says
/// so (the marker hangs out to the left of them); `None` otherwise, as an
/// item with no hanging indent shows no column a paragraph could line up
/// with.
fn text_indent(indent: Indent) -> Option<i32> {
    (indent.first_line < 0).then_some(indent.left)
}

/// End a run of paragraphs that start like items, keeping the lists it
/// reads as. markitai: a paragraph continuing a rejected item stays text.
fn close(run: Run, found: &mut Vec<(Range<usize>, Vec<Block>)>) {
    let Run { items, more, body_left } = run;
    if items.is_empty() {
        return;
    }
    let accepted = accepted(&items);
    let mut more = more.into_iter().peekable();
    let mut part: Vec<Member> = Vec::new();
    for (k, (item, accepted)) in items.into_iter().zip(accepted).enumerate() {
        if accepted {
            part.push(Member::Item(item));
        } else {
            list(std::mem::take(&mut part), body_left, found);
        }
        while let Some((_, paragraph)) = more.next_if(|(after, _)| *after == k + 1) {
            if accepted {
                part.push(Member::More(paragraph));
            }
        }
    }
    list(part, body_left, found);
}

/// markitai: a paragraph of a list typed by hand: an item, or more of one.
enum Member {
    Item(Item),
    More(More),
}

/// The list blocks a run of items makes. markitai: with the paragraphs
/// that continue them; one that lines up with no open item's text ends the
/// list and stays text, and the members after it make their own.
fn list(members: Vec<Member>, body_left: i32, found: &mut Vec<(Range<usize>, Vec<Block>)>) {
    let positions: Vec<i32> = members
        .iter()
        .filter_map(|member| match member {
            Member::Item(item) => Some(item.position()),
            Member::More(_) => None,
        })
        .collect();
    let mut levels = levels_at(&positions).into_iter();
    // The run's items are assembled on their own: one identity serves, and
    // a change of marker kind or a gap in the count still splits a list.
    let instance = 0;
    let mut entries: Vec<ListEntry> = Vec::new();
    let mut range: Option<Range<usize>> = None;
    let mut members = members.into_iter();
    while let Some(member) = members.next() {
        let item = match member {
            Member::Item(item) => item,
            Member::More(more) => match continuation_level(&entries, more.indent.left, body_left) {
                Some(level) => {
                    let blocks = vec![Block::Paragraph(more.content)];
                    entries.push(ListEntry::continuation(level, blocks));
                    range = range.map(|range| range.start..more.index + 1);
                    continue;
                }
                None => {
                    if let Some(range) = range {
                        let mut lists = Vec::new();
                        flush_list(&mut lists, &mut entries);
                        found.push((range, lists));
                    }
                    return list(members.collect(), body_left, found);
                }
            },
        };
        let (marker, number, label) = match item.marker {
            Marker::Bullet { .. } => (MarkerKind::Bullet, 0, None),
            Marker::Number { readings, form, label } => {
                match readings.iter().find(|&&(count, _)| count == Count::Decimal) {
                    Some(&(_, n)) if !form.open && matches!(form.close, None | Some('.' | ')')) => {
                        (MarkerKind::Decimal, n, None)
                    }
                    _ => (MarkerKind::Bullet, 0, Some(label)),
                }
            }
        };
        range = Some(range.map_or(item.index, |range| range.start)..item.index + 1);
        entries.push(ListEntry {
            level: levels.next().unwrap_or_default(),
            key: ListKey { instance, marker },
            number,
            label,
            blocks: vec![Block::Paragraph(item.content)],
            indent: text_indent(item.indent),
            continues: false,
        });
    }
    if let Some(range) = range {
        let mut lists = Vec::new();
        flush_list(&mut lists, &mut entries);
        found.push((range, lists));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{List, Style};
    use crate::shared::tabs::{TAB, tab};

    /// A paragraph of text pieces; `\t` pieces are tabs.
    fn para(pieces: &[&str]) -> Block {
        Block::Paragraph(
            pieces
                .iter()
                .map(|piece| if *piece == TAB { tab(Style::PLAIN) } else { Inline::plain(*piece) })
                .collect(),
        )
    }

    /// The blocks of `paragraphs`, each plain at `indent`, after the lists
    /// are placed.
    fn read(paragraphs: Vec<(Block, Indent)>) -> Vec<Block> {
        let mut lists = TypedLists::default();
        let mut blocks = Vec::new();
        for (index, (block, indent)) in paragraphs.into_iter().enumerate() {
            blocks.push(block);
            lists.paragraph(index, indent);
        }
        let found = lists.lists(&blocks, |_| false);
        for (range, replacement) in found.into_iter().rev() {
            blocks.splice(range, replacement);
        }
        blocks
    }

    fn flat(paragraphs: Vec<Block>) -> Vec<Block> {
        read(paragraphs.into_iter().map(|block| (block, Indent::default())).collect())
    }

    /// Each block as a line: a paragraph's text, or a list's items with
    /// their markers and nesting.
    fn shown(blocks: &[Block]) -> Vec<String> {
        fn items(list: &List, depth: usize, out: &mut Vec<String>) {
            for (i, item) in list.items.iter().enumerate() {
                let marker = match (&item.marker_label, list.marker) {
                    (Some(label), _) => format!("- {label}"),
                    (None, MarkerKind::Bullet) => "-".to_string(),
                    (None, kind) => kind.label(list.start + i as u64),
                };
                for block in &item.blocks {
                    match block {
                        Block::Paragraph(inlines) => out.push(format!(
                            "{}{marker} {}",
                            "  ".repeat(depth),
                            crate::shared::tabs::plain_text(inlines)
                        )),
                        Block::List(inner) => items(inner, depth + 1, out),
                        other => out.push(format!("{other:?}")),
                    }
                }
            }
        }
        let mut out = Vec::new();
        for block in blocks {
            match block {
                Block::Paragraph(inlines) => out.push(inlines_to_plain_text(inlines)),
                Block::List(l) => {
                    out.push("<list>".into());
                    items(l, 0, &mut out);
                }
                other => out.push(format!("{other:?}")),
            }
        }
        out
    }

    fn hanging(left: i32) -> Indent {
        Indent { left, first_line: -left }
    }

    #[test]
    fn textutil_lists_nest_by_indent() {
        let blocks = read(vec![
            (para(&["Features"]), Indent::default()),
            (para(&["", TAB, "•", TAB, "", "Written in Lua"]), hanging(720)),
            (para(&[TAB, "◦", TAB, "Nested"]), hanging(1440)),
            (para(&[TAB, "•", TAB, "Fast"]), hanging(720)),
            (para(&["After."]), Indent::default()),
            (para(&[TAB, "1", TAB, "One"]), hanging(720)),
            (para(&[TAB, "2", TAB, "Two"]), hanging(720)),
        ]);
        assert_eq!(
            shown(&blocks),
            [
                "Features",
                "<list>",
                "- Written in Lua",
                "  - Nested",
                "- Fast",
                "After.",
                "<list>",
                "1. One",
                "2. Two",
            ]
        );
    }

    #[test]
    fn leading_tabs_and_spaces_set_the_level() {
        let blocks = flat(vec![
            para(&["•", TAB, "Top"]),
            para(&[TAB, "–", TAB, "Under it"]),
            para(&["    o Deeper by spaces"]),
            para(&["• Top again"]),
            para(&["•No space"]),
        ]);
        assert_eq!(
            shown(&blocks),
            [
                "<list>",
                "- Top",
                "  - Under it",
                "  - Deeper by spaces",
                "- Top again",
                "- No space"
            ]
        );
    }

    #[test]
    fn symbol_font_bullets_boxes_and_ticks() {
        // A bookmark before the bullet is no text, and stays.
        let mut bookmarked = vec![Inline::Anchor("_Ref1".into())];
        let Block::Paragraph(rest) = para(&["• ", "Marked"]) else { unreachable!() };
        bookmarked.extend(rest);
        let blocks = flat(vec![Block::Paragraph(bookmarked), para(&["Text."])]);
        let Block::List(list) = &blocks[0] else { panic!("{blocks:?}") };
        assert!(matches!(
            &list.items[0].blocks[..],
            [Block::Paragraph(inlines)] if matches!(&inlines[..],
                [Inline::Anchor(_), Inline::Text { text, .. }] if text == "Marked")
        ));
        let blocks = flat(vec![
            para(&["\u{f0b7}", TAB, "Symbol bullet"]),
            para(&["☐ Open task"]),
            para(&["☒ Done task"]),
            para(&["✓ Checked"]),
        ]);
        let [Block::List(list)] = &blocks[..] else { panic!("{blocks:?}") };
        let first = |k: usize| match &list.items[k].blocks[0] {
            Block::Paragraph(inlines) => inlines.clone(),
            _ => unreachable!(),
        };
        assert_eq!(inlines_to_plain_text(&first(0)), "Symbol bullet");
        assert!(matches!(first(1)[0], Inline::Checkbox(false)));
        assert!(matches!(first(2)[0], Inline::Checkbox(true)));
        assert_eq!(inlines_to_plain_text(&first(3)), "✓ Checked");
    }

    #[test]
    fn a_lone_dash_or_dialogue_stays_text() {
        let unchanged = |paragraphs: Vec<Block>| {
            let expected = shown(&paragraphs);
            assert_eq!(shown(&flat(paragraphs)), expected);
        };
        // A sentence opening with a dash, an attribution, a scene break.
        unchanged(vec![para(&["- and then it rained."]), para(&["Next."])]);
        unchanged(vec![para(&["- Where are you going?"]), para(&["- Home!"])]);
        // A weak mark touching its text is no marker (negative numbers).
        unchanged(vec![para(&["-5 °C at night"]), para(&["-3 °C by day"])]);
        unchanged(vec![para(&["“Fear itself.”"]), para(&["— F. D. Roosevelt"])]);
        unchanged(vec![para(&["* * *"]), para(&["* * *"])]);
        // Dialogue: speech endings, or a dash between spaces inside.
        unchanged(vec![
            para(&["— Bonjour, dit-il."]),
            para(&["— Vous partez déjà ?"]),
            para(&["— Oui."]),
        ]);
        unchanged(vec![para(&["– Oui, – dit-elle, – demain."]), para(&["– Bien."])]);
        // An em dash needs a tab: dialogue whose lines end as statements.
        unchanged(vec![para(&["— Bonjour, dit-il."]), para(&["— Bonsoir, répondit-elle."])]);
        // Different marks side by side say something.
        unchanged(vec![
            para(&["+ fast"]),
            para(&["+ cheap"]),
            para(&["- heavy"]),
            para(&["- loud"]),
        ]);
        // Two different weak marks, a doubled one, a word starting with o.
        unchanged(vec![para(&["- one"]), para(&["* two"])]);
        unchanged(vec![para(&["-- comment"]), para(&["-- another"])]);
        unchanged(vec![para(&["o dear"]), para(&["only once"])]);
    }

    #[test]
    fn dialogue_with_a_speech_incise_stays_text() {
        // markitai: hyphens and en dashes whose lines end in full stops are
        // dialogue when a line's clause after a comma is a speech incise.
        let unchanged = |paragraphs: Vec<Block>| {
            let expected = shown(&paragraphs);
            assert_eq!(shown(&flat(paragraphs)), expected);
        };
        unchanged(vec![para(&["– Je pars demain, dit-il."]), para(&["– Je sais."])]);
        unchanged(vec![para(&["- Il est tard, murmura-t-elle, viens."]), para(&["- Oui."])]);
        unchanged(vec![para(&["– Je n'en sais rien, dis-je."]), para(&["– Moi non plus."])]);
        unchanged(vec![para(&["- Llegaremos tarde, dijo."]), para(&["- No importa."])]);
        unchanged(vec![para(&["– Vi ses i morgon, sa han."]), para(&["– Kom i tid."])]);
        unchanged(vec![para(&["- Я подожду, сказал он."]), para(&["- Хорошо."])]);
        unchanged(vec![para(&["- I'll be there, she said."]), para(&["- Fine."])]);
        unchanged(vec![para(&["- Ready, – he asked."]), para(&["- Yes."])]);
        // Commas in list items that are no incise keep the list.
        let listed = |paragraphs: Vec<Block>| {
            let blocks = flat(paragraphs);
            assert!(matches!(&blocks[..], [Block::List(_)]), "{:?}", shown(&blocks));
        };
        listed(vec![para(&["- Fixed the parser, added tests."]), para(&["- Faster builds."])]);
        listed(vec![para(&["- Shares fell, analysts said."]), para(&["- Oil rose."])]);
        listed(vec![para(&["- Revenue rose, the company said."]), para(&["- Costs fell."])]);
        listed(vec![para(&["- Apples, pears and plums."]), para(&["- Milk, then bread."])]);
        listed(vec![para(&["- Si oui, peut-on le changer plus tard."]), para(&["- Non."])]);
        assert!(speech_incise("Bien, s'écria-t-il."));
        assert!(!speech_incise("No comma at all, really"));
        assert!(!speech_incise("Nothing after the comma,"));
    }

    #[test]
    fn weak_marks_need_a_tab_a_sibling_or_a_parent() {
        let blocks = flat(vec![
            para(&["-", TAB, "Tabbed dash"]),
            para(&["Text."]),
            para(&["- milk"]),
            para(&["- eggs"]),
            para(&["Text."]),
            para(&["• Parent"]),
            para(&["    – child"]),
        ]);
        assert_eq!(
            shown(&blocks),
            [
                "<list>",
                "- Tabbed dash",
                "Text.",
                "<list>",
                "- milk",
                "- eggs",
                "Text.",
                "<list>",
                "- Parent",
                "  - child",
            ]
        );
    }

    #[test]
    fn numbers_count_up_in_one_form() {
        let blocks = flat(vec![
            para(&["1. First"]),
            para(&["2. Second"]),
            para(&["Text."]),
            para(&["a) one"]),
            para(&["b) two"]),
            para(&["Text."]),
            para(&["一、总则"]),
            para(&["二、范围"]),
            para(&["Text."]),
            para(&["(iv) four"]),
            para(&["(v) five"]),
            para(&["Text."]),
            para(&["(1) one"]),
            para(&["(2) two"]),
            para(&["Text."]),
            para(&["n) fourteenth"]),
            para(&["o) fifteenth"]),
            para(&["Text."]),
            para(&["3.", TAB, "Continues at three"]),
        ]);
        assert_eq!(
            shown(&blocks),
            [
                "<list>",
                "1. First",
                "2. Second",
                "Text.",
                "<list>",
                "- a) one",
                "- b) two",
                "Text.",
                "<list>",
                "- 一、 总则",
                "- 二、 范围",
                "Text.",
                "<list>",
                "- (iv) four",
                "- (v) five",
                "Text.",
                "<list>",
                "- (1) one",
                "- (2) two",
                "Text.",
                "<list>",
                "- n) fourteenth",
                "- o) fifteenth",
                "Text.",
                "<list>",
                "3. Continues at three",
            ]
        );
    }

    #[test]
    fn numbers_that_label_nothing_stay_text() {
        let unchanged = |paragraphs: Vec<Block>| {
            let expected = shown(&paragraphs);
            assert_eq!(shown(&flat(paragraphs)), expected);
        };
        // A lone numbered paragraph, a gap in the count, mixed forms.
        unchanged(vec![para(&["1. Introduction"]), para(&["Text."])]);
        unchanged(vec![para(&["1. One"]), para(&["3. Three"])]);
        unchanged(vec![para(&["1. One"]), para(&["2) Two"])]);
        // Counts, initials, years, outline numbers, decimals.
        unchanged(vec![para(&["3 apples"]), para(&["4 pears"])]);
        unchanged(vec![para(&["A. Smith wrote it."]), para(&["B. Jones agreed."])]);
        unchanged(vec![para(&["2019", TAB, "Joined"]), para(&["2020", TAB, "Left"])]);
        unchanged(vec![para(&["1.1 Scope"]), para(&["1.2 Terms"])]);
        unchanged(vec![para(&["1.5 kg of flour"]), para(&["2.5 kg of sugar"])]);
        // A bare letter before a tab (questions and answers).
        unchanged(vec![para(&["Q", TAB, "Why?"]), para(&["A", TAB, "Because."])]);
        // Numbers before tabs side by side that do not count up (an
        // inventory), and a numbered heading typed by hand in bold.
        unchanged(vec![para(&["3", TAB, "Apples"]), para(&["12", TAB, "Pears"])]);
        let bold = Inline::Text {
            text: "Introduction".into(),
            style: Style { bold: true, ..Style::PLAIN },
        };
        unchanged(vec![
            Block::Paragraph(vec![Inline::plain("1"), tab(Style::PLAIN), bold]),
            para(&["Text."]),
        ]);
        // Nothing after the marker.
        unchanged(vec![para(&[TAB, "1", TAB, TAB]), para(&["•"])]);
    }

    #[test]
    fn nested_bullets_keep_a_numbered_count_going() {
        let blocks = read(vec![
            (para(&[TAB, "1", TAB, "Step one"]), hanging(720)),
            (para(&[TAB, "•", TAB, "detail"]), hanging(1440)),
            (para(&[TAB, "2", TAB, "Step two"]), hanging(720)),
        ]);
        assert_eq!(shown(&blocks), ["<list>", "1. Step one", "  - detail", "2. Step two"]);
    }

    #[test]
    fn labels_and_values() {
        assert_eq!(readings("12"), [(Count::Decimal, 12)]);
        assert_eq!(readings("１２"), [(Count::Decimal, 12)]);
        assert!(readings("2024").is_empty());
        assert_eq!(readings("i"), [(Count::LowerAlpha, 9), (Count::LowerRoman, 1)]);
        assert_eq!(readings("XIV"), [(Count::UpperRoman, 14)]);
        assert!(readings("iiii").is_empty() && readings("Iv").is_empty());
        assert_eq!(readings("二十三"), [(Count::Cjk, 23)]);
        assert_eq!(readings("十"), [(Count::Cjk, 10)]);
        assert!(readings("一十").is_empty());
        assert_eq!(readings("③"), [(Count::Circled, 3)]);
        let hanging = Item {
            index: 0,
            indent: hanging(720),
            marker: Marker::Bullet { c: '•', kind: Bullet::Only },
            leading: vec!['\t'],
            tab_after: true,
            content: Vec::new(),
        };
        assert_eq!(hanging.position(), 720);
        let typed =
            Item { indent: Indent::default(), leading: vec!['\t', '\t'], ..hanging.clone() };
        assert_eq!(typed.position(), 1440);
        let spaced =
            Item { indent: Indent { left: 360, first_line: 0 }, leading: vec![' '; 3], ..hanging };
        assert_eq!(spaced.position(), 540);
    }

    /// markitai: a paragraph whose lines start `left` twips in.
    fn at(left: i32) -> Indent {
        Indent { left, first_line: 0 }
    }

    #[test]
    fn a_paragraph_set_in_to_an_items_text_continues_it() {
        let blocks = read(vec![
            (para(&["Before the list."]), at(0)),
            (para(&[TAB, "1", TAB, "One"]), hanging(720)),
            (para(&["More of one."]), at(720)),
            (para(&[TAB, "2", TAB, "Two"]), hanging(720)),
            (para(&[TAB, "◦", TAB, "Two a"]), hanging(1440)),
            (para(&["More of two a."]), at(1440)),
            (para(&["More of two."]), at(720)),
            (para(&[TAB, "3", TAB, "Three"]), hanging(720)),
            (para(&["Body after."]), at(0)),
            (para(&["Indented after the body."]), at(720)),
        ]);
        assert_eq!(
            crate::shared::code::describe(&blocks),
            [
                "p:Before the list.",
                "list:p:One;p:More of one.|p:Two;list:p:Two a;p:More of two a.;p:More of two.|p:Three",
                "p:Body after.",
                "p:Indented after the body.",
            ]
        );
        let [_, Block::List(list), ..] = &blocks[..] else { panic!("{blocks:?}") };
        assert_eq!((list.start, list.items.len()), (1, 3));
    }

    #[test]
    fn only_a_paragraph_lined_up_with_an_items_text_continues_it() {
        let shown = |paragraphs| crate::shared::code::describe(&read(paragraphs));
        // Body set in as far as the list's text, before and after it.
        assert_eq!(
            shown(vec![
                (para(&["Body set in."]), at(720)),
                (para(&[TAB, "•", TAB, "One"]), hanging(720)),
                (para(&["Body again."]), at(720)),
            ]),
            ["p:Body set in.", "list:p:One", "p:Body again."]
        );
        // Set back to where the bullet hangs.
        assert_eq!(
            shown(vec![
                (para(&[TAB, "•", TAB, "One"]), hanging(720)),
                (para(&["Under the bullet."]), at(360)),
            ]),
            ["list:p:One", "p:Under the bullet."]
        );
        // An item with no hanging indent shows no column of text.
        assert_eq!(
            shown(vec![(para(&["•", TAB, "One"]), at(0)), (para(&["Set in."]), at(720))]),
            ["list:p:One", "p:Set in."]
        );
        assert_eq!(
            shown(vec![(para(&["•", TAB, "One"]), at(720)), (para(&["Set in."]), at(720))]),
            ["list:p:One", "p:Set in."]
        );
        // Body text set in between two dashes keeps them apart: neither is
        // an item alone.
        assert_eq!(
            shown(vec![
                (para(&["Body set in."]), at(720)),
                (para(&["- one"]), hanging(720)),
                (para(&["Body again."]), at(720)),
                (para(&["- two"]), hanging(720)),
            ]),
            ["p:Body set in.", "p:- one", "p:Body again.", "p:- two"]
        );
        // After an item that is no item (dialogue), it stays text too.
        assert_eq!(
            shown(vec![
                (para(&["- Where are you going?"]), hanging(720)),
                (para(&["Narration set in."]), at(720)),
                (para(&["- Home!"]), hanging(720)),
            ]),
            ["p:- Where are you going?", "p:Narration set in.", "p:- Home!"]
        );
        // An empty paragraph or one with only a mark ends the list.
        assert_eq!(
            shown(vec![
                (para(&[TAB, "•", TAB, "One"]), hanging(720)),
                (para(&[" "]), at(720)),
                (para(&["Set in, after a blank."]), at(720)),
            ]),
            ["list:p:One", "p: ", "p:Set in, after a blank."]
        );
    }
}
