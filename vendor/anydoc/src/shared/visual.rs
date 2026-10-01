//! markitai: headings a document shows only by how a paragraph looks.
//!
//! Many documents never use a heading style: a title is a short paragraph set
//! in bold at a size above the body text, and nothing else marks it. Word
//! processors saving a web page (macOS TextEdit and `textutil` write an `h2`
//! as `<w:b/><w:sz w:val="36"/>`) and people formatting by hand both do this,
//! and read as plain text such a title became a bold paragraph.
//!
//! The readers report, while they read, the size of every piece of visible
//! body text and which top-level paragraphs are plain (no heading style or
//! outline level, not a list item, not in a table, cell, text box or styled
//! container). [`Looks::apply`] then turns into headings the plain paragraphs
//! that look like one, and only in a document with no heading at all, so it
//! never competes with headings the author marked.
//!
//! The rules were set against web pages saved by `textutil` as DOCX, ODT, RTF
//! and Word 97 documents, whose HTML headings are known, and checked on pages
//! held out from that choice (see `docs/formats.md`):
//!
//! - the body size is the size most visible characters are set in;
//! - all of a candidate's visible text has one size: at least a point above
//!   the body and bold, or a third above it without bold (13-point code
//!   under 12-point text is never bold, and is not a heading);
//! - bold text at the body size is not enough (table headers, labels and
//!   lead-ins look the same), nor is anything smaller than the body;
//! - at most [`MAX_WORDS`] words (two CJK characters count as one), at least
//!   one letter (a number drawn large is a figure), not ending like running
//!   prose (`.`, `,`, `;`, `:` and their full-width forms), and no image;
//! - sizes rank the levels: the largest candidate size is level 1, the next
//!   level 2, and so on to 6, so a title set once above the section
//!   headings is the one level-1 heading.

use crate::model::{Block, CellSlot, Inline, Style, inlines_to_plain_text};
use std::collections::HashMap;

/// A font size in half-points, the unit of Word's `w:sz` and RTF's `\fs`.
pub type Size = u32;

/// Longest heading set by hand, in words. The longest of the corpus's has 13.
const MAX_WORDS: usize = 15;

/// The one size of a paragraph's visible text, gathered run by run.
#[derive(Debug, Clone, Copy, Default)]
pub struct ParaSize {
    size: Option<Size>,
    mixed: bool,
}

impl ParaSize {
    fn add(&mut self, size: Size) {
        match self.size {
            None if !self.mixed => self.size = Some(size),
            Some(seen) if seen != size => {
                self.size = None;
                self.mixed = true;
            }
            _ => {}
        }
    }

    /// Take in the sizes of text read apart (a link's, a field's).
    pub fn merge(&mut self, other: ParaSize) {
        if other.mixed {
            self.size = None;
            self.mixed = true;
        } else if let Some(size) = other.size {
            self.add(size);
        }
    }

    /// The size every visible character is set in; `None` when the text
    /// mixes sizes or has none.
    pub fn uniform(self) -> Option<Size> {
        if self.mixed { None } else { self.size }
    }
}

/// What a reader saw of a document's body text, for [`Looks::apply`].
#[derive(Debug, Default)]
pub struct Looks {
    /// Visible characters set at each size, and those of the latest text's
    /// size not yet counted there (text mostly keeps the size before it).
    chars: HashMap<Size, usize>,
    latest: Option<(Size, usize)>,
    /// Plain top-level paragraphs: the index of their block and the one size
    /// of their text.
    paragraphs: Vec<(usize, Option<Size>)>,
}

impl Looks {
    /// Text set at `size` was read, inside the paragraph whose sizes `para`
    /// gathers. Whitespace shows no size.
    pub fn text(&mut self, para: &mut ParaSize, size: Size, text: &str) {
        let visible = text.chars().filter(|c| !c.is_whitespace()).count();
        if visible == 0 {
            return;
        }
        match &mut self.latest {
            Some((latest, count)) if *latest == size => *count += visible,
            latest => {
                if let Some((size, count)) = latest.replace((size, visible)) {
                    *self.chars.entry(size).or_default() += count;
                }
            }
        }
        para.add(size);
    }

    /// `blocks[index]` is a plain top-level paragraph whose text gathered
    /// `size`.
    pub fn paragraph(&mut self, index: usize, size: ParaSize) {
        self.paragraphs.push((index, size.uniform()));
    }

    /// Leave out of [`Self::apply`] the plain paragraphs at the indices
    /// `skip` names (the rows of a table set with tab stops; see
    /// [`crate::shared::tabs`]).
    pub fn skip(&mut self, skip: impl Fn(usize) -> bool) {
        self.paragraphs.retain(|&(index, _)| !skip(index));
    }

    /// The body text size: the size most visible characters are set in (the
    /// larger one on a tie, which finds fewer headings).
    fn body_size(&mut self) -> Option<Size> {
        if let Some((size, count)) = self.latest.take() {
            *self.chars.entry(size).or_default() += count;
        }
        self.chars.iter().max_by_key(|&(size, count)| (*count, *size)).map(|(size, _)| *size)
    }

    /// Turn the plain paragraphs of `blocks` that look like headings into
    /// headings, unless the document has a heading already.
    pub fn apply(mut self, blocks: &mut [Block]) {
        if blocks.iter().any(has_heading) {
            return;
        }
        let Some(body) = self.body_size() else {
            return;
        };
        let found: Vec<(usize, Size)> = self
            .paragraphs
            .iter()
            .filter_map(|&(index, size)| match blocks.get(index) {
                Some(Block::Paragraph(inlines)) => {
                    size.filter(|&size| looks_like_heading(inlines, size, body)).map(|s| (index, s))
                }
                _ => None,
            })
            .collect();
        let mut sizes: Vec<Size> = found.iter().map(|&(_, size)| size).collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        sizes.dedup();
        for (index, size) in found {
            let rank = sizes.iter().position(|&s| s == size).unwrap_or(0);
            let level = (rank + 1).min(6) as u8;
            if let Block::Paragraph(inlines) = &mut blocks[index] {
                let mut content = std::mem::take(inlines);
                // A heading's weight is its level, not emphasis markers, and
                // the line breaks around a title only spaced it on the page.
                without_bold(&mut content);
                trim_breaks(&mut content);
                blocks[index] = Block::heading(level, content);
            }
        }
    }
}

/// Whether a plain paragraph whose visible text is all set at `size` looks
/// like a heading over body text set at `body`.
fn looks_like_heading(inlines: &[Inline], size: Size, body: Size) -> bool {
    let larger = if all_bold(inlines) { size >= body + 2 } else { size * 3 >= body * 4 };
    if !larger || has_image(inlines) {
        return false;
    }
    let text = inlines_to_plain_text(inlines);
    let text = text.trim();
    text.chars().any(char::is_alphabetic)
        && word_count(text) <= MAX_WORDS
        && !text.ends_with(['.', ',', ';', ':', '。', '，', '；', '：'])
}

/// Words of `text`; two CJK characters count as one.
fn word_count(text: &str) -> usize {
    let mut words = 0;
    let mut cjk: usize = 0;
    for token in text.split_whitespace() {
        let mut in_word = false;
        for c in token.chars() {
            if is_cjk(c) {
                cjk += 1;
                words += usize::from(in_word);
                in_word = false;
            } else if c.is_alphanumeric() {
                in_word = true;
            }
        }
        words += usize::from(in_word);
    }
    words + cjk.div_ceil(2)
}

/// Han, kana and Hangul: scripts written without spaces between words.
fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{3040}'..='\u{30ff}'
        | '\u{3400}'..='\u{4dbf}'
        | '\u{4e00}'..='\u{9fff}'
        | '\u{ac00}'..='\u{d7af}'
        | '\u{f900}'..='\u{faff}'
        | '\u{20000}'..='\u{2fa1f}')
}

/// Whether every piece of visible text is bold.
fn all_bold(inlines: &[Inline]) -> bool {
    inlines.iter().all(|inline| match inline {
        Inline::Text { text, style } => style.bold || text.trim().is_empty(),
        Inline::Link { content, .. } => all_bold(content),
        _ => true,
    })
}

fn has_image(inlines: &[Inline]) -> bool {
    inlines.iter().any(|inline| match inline {
        Inline::Image { .. } => true,
        Inline::Link { content, .. } => has_image(content),
        _ => false,
    })
}

fn without_bold(inlines: &mut [Inline]) {
    for inline in inlines {
        match inline {
            Inline::Text { style, .. } => *style = Style { bold: false, ..*style },
            Inline::Link { content, .. } => without_bold(content),
            _ => {}
        }
    }
}

/// Drop the line breaks and blank text that lead or trail `inlines`.
fn trim_breaks(inlines: &mut Vec<Inline>) {
    let blank = |inline: &Inline| match inline {
        Inline::LineBreak => true,
        Inline::Text { text, .. } => text.trim().is_empty(),
        _ => false,
    };
    let start = inlines.iter().position(|inline| !blank(inline)).unwrap_or(inlines.len());
    inlines.drain(..start);
    let end = inlines.iter().rposition(|inline| !blank(inline)).map_or(0, |last| last + 1);
    inlines.truncate(end);
}

/// Whether a heading appears anywhere in `block`.
fn has_heading(block: &Block) -> bool {
    match block {
        Block::Heading { .. } => true,
        Block::BlockQuote(blocks) => blocks.iter().any(has_heading),
        Block::List(list) => list.items.iter().any(|item| item.blocks.iter().any(has_heading)),
        Block::Table(table) => table.grid.iter().flatten().any(|slot| match slot {
            CellSlot::Origin(cell) => cell.blocks.iter().any(has_heading),
            CellSlot::Covered { .. } => false,
        }),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bold(text: &str) -> Inline {
        Inline::Text { text: text.into(), style: Style { bold: true, ..Style::PLAIN } }
    }

    /// A document read as paragraphs of (text, bold, size in half-points).
    fn read(paragraphs: &[(&str, bool, Size)]) -> Vec<Block> {
        let mut looks = Looks::default();
        let mut blocks = Vec::new();
        for &(text, is_bold, size) in paragraphs {
            let mut para = ParaSize::default();
            looks.text(&mut para, size, text);
            blocks.push(Block::Paragraph(vec![if is_bold {
                bold(text)
            } else {
                Inline::plain(text)
            }]));
            looks.paragraph(blocks.len() - 1, para);
        }
        looks.apply(&mut blocks);
        blocks
    }

    fn levels(blocks: &[Block]) -> Vec<Option<u8>> {
        blocks
            .iter()
            .map(|block| match block {
                Block::Heading { level, .. } => Some(*level),
                _ => None,
            })
            .collect()
    }

    /// Body text, long enough to outweigh every heading in these tests.
    const BODY: &str = "Body text runs on for long enough to set the size most of the document \
        uses: it is the text the headings stand out from, so there has to be more of it than of \
        all of them together, which takes a few sentences. Here they are, plainly set, with \
        nothing bold and nothing larger, to be read as the paragraphs they are.";

    #[test]
    fn sizes_above_the_body_rank_the_levels() {
        let blocks = read(&[
            ("The Title", true, 48),
            (BODY, false, 24),
            ("A Section", true, 36),
            (BODY, false, 24),
            ("A Subsection", true, 28),
            (BODY, false, 24),
            ("Another Section", true, 36),
            (BODY, false, 24),
        ]);
        assert_eq!(levels(&blocks), [Some(1), None, Some(2), None, Some(3), None, Some(2), None]);
        let Block::Heading { content, .. } = &blocks[0] else { unreachable!() };
        assert!(matches!(&content[0], Inline::Text { style, .. } if !style.bold));
    }

    #[test]
    fn bold_at_the_body_size_and_plain_text_slightly_larger_are_body_text() {
        let blocks = read(&[
            ("Framework", true, 24),
            ("let x = 1", false, 26),
            (BODY, false, 24),
            ("Large Plain Title", false, 32),
            ("Small Print", true, 20),
            ("Half a Point Up", true, 25),
            ("2.0", true, 36),
        ]);
        assert_eq!(levels(&blocks), [None, None, None, Some(1), None, None, None]);
    }

    #[test]
    fn prose_and_long_lines_stay_paragraphs() {
        let blocks = read(&[
            ("Here's the latest.", true, 36),
            ("Highlights:", true, 36),
            (
                "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen",
                true,
                36,
            ),
            ("第一章 总则", true, 36),
            (BODY, false, 24),
        ]);
        assert_eq!(levels(&blocks), [None, None, None, Some(1), None]);
    }

    #[test]
    fn a_mixed_size_paragraph_is_not_a_candidate() {
        let mut looks = Looks::default();
        let mut para = ParaSize::default();
        looks.text(&mut para, 36, "Title");
        looks.text(&mut para, 24, "small");
        assert_eq!(para.uniform(), None);
        let mut other = ParaSize::default();
        looks.text(&mut other, 36, "  ");
        assert_eq!(other.uniform(), None, "whitespace has no size");
        other.merge(para);
        assert_eq!(other.uniform(), None);
    }

    #[test]
    fn a_styled_heading_anywhere_turns_the_guess_off() {
        let mut looks = Looks::default();
        let mut para = ParaSize::default();
        looks.text(&mut para, 48, "Looks Like a Title");
        let mut body = ParaSize::default();
        looks.text(&mut body, 24, BODY);
        let mut blocks = vec![
            Block::Paragraph(vec![bold("Looks Like a Title")]),
            Block::Paragraph(vec![Inline::plain(BODY)]),
            Block::BlockQuote(vec![Block::heading(2, vec![Inline::plain("Styled")])]),
        ];
        looks.paragraph(0, para);
        looks.paragraph(1, body);
        looks.apply(&mut blocks);
        assert!(matches!(blocks[0], Block::Paragraph(_)));
    }

    #[test]
    fn line_breaks_around_a_title_are_dropped() {
        let mut looks = Looks::default();
        let mut para = ParaSize::default();
        looks.text(&mut para, 48, "Spaced Title");
        let mut body = ParaSize::default();
        looks.text(&mut body, 24, BODY);
        let mut blocks = vec![
            Block::Paragraph(vec![
                Inline::LineBreak,
                Inline::plain(" "),
                bold("Spaced Title"),
                Inline::LineBreak,
            ]),
            Block::Paragraph(vec![Inline::plain(BODY)]),
        ];
        looks.paragraph(0, para);
        looks.paragraph(1, body);
        looks.apply(&mut blocks);
        let Block::Heading { level: 1, content, .. } = &blocks[0] else { panic!("{blocks:?}") };
        assert!(matches!(&content[..], [Inline::Text { text, .. }] if text == "Spaced Title"));
    }

    #[test]
    fn cjk_words_are_counted_in_pairs() {
        assert_eq!(word_count("第一章 总则"), 3);
        assert_eq!(word_count("Rust 编程语言"), 3);
        assert_eq!(word_count("Node.js and CPU"), 3);
    }
}
