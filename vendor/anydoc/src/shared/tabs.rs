//! markitai: columns set with tab stops.
//!
//! Word processors line columns up with tabs: "Item<Tab>Quantity<Tab>Price",
//! one paragraph per row, at tab stops the author set or at the default
//! ones. Read as text, each tab became a space and the columns ran
//! together. The Word, OpenDocument and RTF readers now keep each tab of the
//! body as a text inline of its own ([`TAB`]) and note which top-level
//! paragraphs could be a row - plain body paragraphs, not a heading, list
//! item, table cell, text box or styled container - with the tab stops they
//! are set at. [`finish`] makes a table of each run of such paragraphs that
//! reads as one, and writes every other tab back as the space it was, so
//! nothing else changes.
//!
//! The rules keep a missed table as the text it was rather than guess:
//!
//! - at least [`MIN_ROWS`] consecutive paragraphs at the same tab stops,
//!   each with the same number of tab-separated cells, and no line break
//!   inside; a tab inside a link splits nothing;
//! - at the default tab stops a run of tabs separates one pair of columns
//!   (the author pressed Tab until the text lined up), so empty cells are
//!   dropped; at stops the author set each tab moves to the next column;
//! - a column no row fills is dropped: a tab before every row's text
//!   indents it (a verse, a quotation) and makes no column;
//! - two columns only at tab stops the author set, and not after a list
//!   label (`1.`, `a)`) or a field label (`Date:`); three or more at any;
//! - not a list typed by hand (a first column of bullets), a table of
//!   contents (a tab stop with a leader, or page numbers counting up in the
//!   last column with no header above them) or prose (a row longer than
//!   [`MAX_ROW_CHARS`], which no longer fits a line);
//! - the first row is the header only when it, and not every row, is bold;
//!   otherwise every row is data, as in a Word table with no header row.

use crate::model::{
    Block, Cell, CellSlot, Inline, Note, Style, Table, TableKind, inlines_are_empty,
    inlines_to_plain_text,
};
use crate::shared::visual::Looks;
use std::ops::Range;

/// A tab, as a reader keeps it in text until [`finish`] places it: a text
/// inline of exactly this. Being whitespace, it reads as the space it was
/// wherever a reader looks at the text before then.
pub const TAB: &str = "\t";

/// Fewest rows a table of tab-separated columns has.
const MIN_ROWS: usize = 3;

/// Most columns a table of tab-separated columns has.
const MAX_COLUMNS: usize = 12;

/// Longest row, in characters: a row of tab-set columns is one line (about
/// a hundred characters at 11 or 12 points on a portrait page), and a
/// longer paragraph wraps under its first column like prose.
const MAX_ROW_CHARS: usize = 100;

/// A tab in text set in `style`.
pub fn tab(style: Style) -> Inline {
    Inline::Text { text: TAB.to_string(), style }
}

fn is_tab(inline: &Inline) -> bool {
    matches!(inline, Inline::Text { text, .. } if text == TAB)
}

/// Whether a paragraph's own inline content holds a tab.
pub fn has_tab(inlines: &[Inline]) -> bool {
    inlines.iter().any(is_tab)
}

/// The text of inline content, its tabs written as spaces (as
/// [`inlines_to_plain_text`] gives it otherwise).
pub fn plain_text(inlines: &[Inline]) -> String {
    let mut out = String::new();
    for inline in inlines {
        match inline {
            tab if is_tab(tab) => out.push(' '),
            Inline::Link { content, .. } => out.push_str(&plain_text(content)),
            other => out.push_str(&inlines_to_plain_text(std::slice::from_ref(other))),
        }
    }
    out
}

/// Write every tab in inline content as the space it was.
pub fn spaces(inlines: &mut [Inline]) {
    for inline in inlines {
        match inline {
            Inline::Text { text, .. } if text == TAB => *text = " ".to_string(),
            Inline::Link { content, .. } => spaces(content),
            _ => {}
        }
    }
}

fn spaces_in_blocks(blocks: &mut [Block]) {
    for block in blocks {
        match block {
            Block::Heading { content, .. } | Block::Paragraph(content) => spaces(content),
            Block::BlockQuote(inner) => spaces_in_blocks(inner),
            Block::List(list) => {
                for item in &mut list.items {
                    spaces_in_blocks(&mut item.blocks);
                }
            }
            Block::Table(table) => {
                for slot in table.grid.iter_mut().flatten() {
                    if let CellSlot::Origin(cell) = slot {
                        spaces_in_blocks(&mut cell.blocks);
                    }
                }
            }
            Block::CodeBlock { .. } | Block::Rule | Block::Math(_) => {}
        }
    }
}

/// The tab stops a paragraph is set at, kept as a fingerprint: paragraphs
/// whose stops are equal line their columns up.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stops {
    /// How many stops the paragraph sets (none: the default ones only).
    set: u16,
    /// Whether a stop draws a leader (dots or a line up to the text).
    leader: bool,
    hash: u64,
}

impl Stops {
    /// A tab stop at `position` (in the format's own unit), aligning text
    /// `align`, with a leader or not. Stops are added in position order.
    pub fn add(&mut self, position: impl std::fmt::Display, align: &str, leader: bool) {
        self.set = self.set.saturating_add(1);
        self.leader |= leader;
        // FNV-1a over the stop.
        let mut hash = if self.hash == 0 { 0xcbf2_9ce4_8422_2325 } else { self.hash };
        for byte in format!("{position}/{align}/{leader};").bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
        self.hash = hash;
    }

    /// Whether the paragraph uses the default stops only.
    fn default_only(self) -> bool {
        self.set == 0
    }
}

/// The paragraphs of a document that could be rows of a table set with tab
/// stops, gathered while it is read.
#[derive(Debug, Default)]
pub struct TabRows {
    paragraphs: Vec<(usize, Stops)>,
}

impl TabRows {
    /// `blocks[index]` is a plain top-level paragraph holding a tab, set at
    /// `stops`.
    pub fn paragraph(&mut self, index: usize, stops: Stops) {
        self.paragraphs.push((index, stops));
    }

    /// Each run of the paragraphs that reads as a table: the blocks it
    /// takes and the table.
    fn tables(&self, blocks: &[Block]) -> Vec<(Range<usize>, Table)> {
        let mut found = Vec::new();
        let mut run: Vec<(usize, Vec<Vec<Inline>>)> = Vec::new();
        let mut run_stops = Stops::default();
        for &(index, stops) in &self.paragraphs {
            let row = match blocks.get(index) {
                Some(Block::Paragraph(inlines)) => cells(inlines, stops),
                _ => None,
            };
            let continues = |row: &Vec<Vec<Inline>>| {
                run.last().is_some_and(|(last, cells)| {
                    *last + 1 == index && run_stops == stops && cells.len() == row.len()
                })
            };
            if !row.as_ref().is_some_and(continues) {
                close(&mut run, run_stops, &mut found);
            }
            if let Some(row) = row {
                run.push((index, row));
                run_stops = stops;
            }
        }
        close(&mut run, run_stops, &mut found);
        found
    }
}

/// End a run of rows, keeping its table if it reads as one.
fn close(
    run: &mut Vec<(usize, Vec<Vec<Inline>>)>,
    stops: Stops,
    found: &mut Vec<(Range<usize>, Table)>,
) {
    let rows = std::mem::take(run);
    if rows.len() < MIN_ROWS {
        return;
    }
    let range = rows[0].0..rows[rows.len() - 1].0 + 1;
    if let Some(table) = table(rows.into_iter().map(|(_, cells)| cells).collect(), stops) {
        found.push((range, table));
    }
}

/// A paragraph's inline content split at each of its tabs; `None` when it
/// cannot be a row (no tab of its own, or a line break inside it).
fn cells(inlines: &[Inline], stops: Stops) -> Option<Vec<Vec<Inline>>> {
    let start = inlines.iter().position(|inline| !matches!(inline, Inline::LineBreak))?;
    let end = inlines.iter().rposition(|inline| !matches!(inline, Inline::LineBreak))? + 1;
    let inlines = &inlines[start..end];
    if !has_tab(inlines) || inlines.iter().any(|inline| matches!(inline, Inline::LineBreak)) {
        return None;
    }
    let mut cells = vec![Vec::new()];
    for inline in inlines {
        if is_tab(inline) {
            cells.push(Vec::new());
        } else if let Some(cell) = cells.last_mut() {
            cell.push(inline.clone());
        }
    }
    if stops.default_only() {
        // Tabs pressed until the text lines up separate one pair of
        // columns; a tab with nothing after it separates nothing.
        let first = cells.remove(0);
        cells.retain(|cell| !inlines_are_empty(cell));
        cells.insert(0, first);
    }
    Some(cells)
}

/// The table a run of rows makes, if it reads as one (see the module
/// documentation).
fn table(mut rows: Vec<Vec<Vec<Inline>>>, stops: Stops) -> Option<Table> {
    if stops.leader {
        return None;
    }
    // A column no row fills goes, the indentation of every row included.
    let width = rows[0].len();
    let filled: Vec<bool> =
        (0..width).map(|c| rows.iter().any(|row| !inlines_are_empty(&row[c]))).collect();
    for row in &mut rows {
        let mut column = 0;
        row.retain(|_| {
            column += 1;
            filled[column - 1]
        });
    }
    let width = rows[0].len();
    if !(2..=MAX_COLUMNS).contains(&width) {
        return None;
    }
    let texts: Vec<Vec<String>> = rows
        .iter()
        .map(|row| row.iter().map(|cell| plain_text(cell).trim().to_string()).collect())
        .collect();
    if texts
        .iter()
        .any(|row| row.iter().map(|text| text.chars().count()).sum::<usize>() > MAX_ROW_CHARS)
    {
        return None;
    }
    let column = |c: usize| texts.iter().map(move |row| row[c].as_str());
    if column(0).all(is_bullet) {
        return None;
    }
    if width == 2
        && (stops.default_only()
            || column(0).all(is_number_label)
            || column(0).all(|text| text.ends_with([':', '：'])))
    {
        return None;
    }
    let last = width - 1;
    if column(last).all(is_page_number) && counts_up(column(last)) {
        return None;
    }
    let header = all_bold(&rows[0]) && !rows.iter().all(|row| all_bold(row));
    let rows = rows
        .into_iter()
        .map(|row| row.into_iter().map(|cell| Cell::from_inlines(trimmed(cell))).collect())
        .collect();
    Some(Table::from_rows(rows, usize::from(header), TableKind::Data))
}

/// A cell's content without the spaces at its edges.
fn trimmed(mut cell: Vec<Inline>) -> Vec<Inline> {
    if let Some(Inline::Text { text, .. }) = cell.first_mut() {
        *text = text.trim_start().to_string();
    }
    if let Some(Inline::Text { text, .. }) = cell.last_mut() {
        *text = text.trim_end().to_string();
    }
    cell.retain(|inline| !matches!(inline, Inline::Text { text, .. } if text.is_empty()));
    cell
}

/// A bullet a list typed by hand starts its items with.
fn is_bullet(text: &str) -> bool {
    let mut chars = text.chars();
    let (Some(c), None) = (chars.next(), chars.next()) else {
        return false;
    };
    matches!(
        c,
        '•' | '◦'
            | '▪'
            | '▫'
            | '●'
            | '○'
            | '■'
            | '□'
            | '·'
            | '‣'
            | '⁃'
            | '–'
            | '—'
            | '-'
            | '*'
            | '+'
            | 'o'
            | '→'
            | '➢'
            | '➤'
            | '✓'
            | '✔'
            | '§'
    ) || ('\u{f000}'..='\u{f0ff}').contains(&c)
}

/// A number a list typed by hand starts its items with: `1`, `1.`, `2.1`,
/// `a)`, `(iv)`, `一、`.
fn is_number_label(text: &str) -> bool {
    let label = text.strip_prefix(['(', '（']).unwrap_or(text);
    let label = label.strip_suffix(['.', ')', '）', '、']).unwrap_or(label);
    let count = label.chars().count();
    (1..=8).contains(&count)
        && (label
            .split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
            || (count == 1 && label.chars().all(|c| c.is_ascii_alphabetic()))
            || is_roman(label)
            || label.chars().all(|c| "〇零一二三四五六七八九十百".contains(c)))
}

fn is_roman(text: &str) -> bool {
    (1..=7).contains(&text.len())
        && text.chars().all(|c| matches!(c.to_ascii_lowercase(), 'i' | 'v' | 'x' | 'l' | 'c'))
}

/// A page number of a table of contents: digits, or a front-matter Roman
/// numeral.
fn is_page_number(text: &str) -> bool {
    ((1..=4).contains(&text.len()) && text.bytes().all(|b| b.is_ascii_digit())) || is_roman(text)
}

/// Whether the numbers among `pages` never go down.
fn counts_up<'t>(pages: impl Iterator<Item = &'t str>) -> bool {
    let numbers: Vec<u32> = pages.filter_map(|page| page.parse().ok()).collect();
    numbers.windows(2).all(|pair| pair[0] <= pair[1])
}

/// Whether every piece of visible text in a row is bold.
fn all_bold(row: &[Vec<Inline>]) -> bool {
    fn bold(inlines: &[Inline]) -> bool {
        inlines.iter().all(|inline| match inline {
            Inline::Text { text, style } => style.bold || text.trim().is_empty(),
            Inline::Link { content, .. } => bold(content),
            _ => true,
        })
    }
    row.iter().any(|cell| !inlines_are_empty(cell)) && row.iter().all(|cell| bold(cell))
}

/// Place the tables a document's tabs set out, write every other tab back
/// as a space, and then find the headings its paragraphs show by their looks
/// (see [`crate::shared::visual`]); a row of a table is no heading.
pub fn finish(rows: TabRows, looks: Option<Looks>, blocks: &mut Vec<Block>, notes: &mut [Note]) {
    let tables = rows.tables(blocks);
    spaces_in_blocks(blocks);
    for note in notes {
        spaces_in_blocks(&mut note.blocks);
    }
    if let Some(mut looks) = looks {
        looks.skip(|index| tables.iter().any(|(range, _)| range.contains(&index)));
        looks.apply(blocks);
    }
    for (range, table) in tables.into_iter().rev() {
        let _rows: Vec<Block> = blocks.splice(range, [Block::Table(table)]).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::visual::ParaSize;

    /// A paragraph of `cells` joined by tabs.
    fn row(cells: &[&str]) -> Block {
        let mut inlines = Vec::new();
        for (i, cell) in cells.iter().enumerate() {
            if i > 0 {
                inlines.push(tab(Style::PLAIN));
            }
            if !cell.is_empty() {
                inlines.push(Inline::plain(*cell));
            }
        }
        Block::Paragraph(inlines)
    }

    fn custom() -> Stops {
        let mut stops = Stops::default();
        stops.add(3000, "left", false);
        stops.add(6000, "right", false);
        stops
    }

    /// The blocks of a document of `paragraphs`, each recorded at `stops`
    /// when it holds a tab, after [`finish`].
    fn read(paragraphs: Vec<Block>, stops: Stops) -> Vec<Block> {
        let mut rows = TabRows::default();
        for (index, block) in paragraphs.iter().enumerate() {
            if matches!(block, Block::Paragraph(inlines) if has_tab(inlines)) {
                rows.paragraph(index, stops);
            }
        }
        let mut blocks = paragraphs;
        finish(rows, None, &mut blocks, &mut []);
        blocks
    }

    /// Each block as a line: a paragraph's text, or a table's rows of cells.
    fn shown(blocks: &[Block]) -> Vec<String> {
        blocks
            .iter()
            .map(|block| match block {
                Block::Paragraph(inlines) => inlines_to_plain_text(inlines),
                Block::Table(table) => table
                    .grid
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|slot| match slot {
                                CellSlot::Origin(cell) => match &cell.blocks[..] {
                                    [Block::Paragraph(inlines)] => inlines_to_plain_text(inlines),
                                    _ => "?".into(),
                                },
                                CellSlot::Covered { .. } => "^".into(),
                            })
                            .collect::<Vec<_>>()
                            .join("|")
                    })
                    .collect::<Vec<_>>()
                    .join(" / "),
                other => format!("{other:?}"),
            })
            .collect()
    }

    #[test]
    fn rows_at_the_same_stops_make_a_table() {
        let blocks = read(
            vec![
                Block::Paragraph(vec![Inline::plain("Price list follows.")]),
                row(&["Item", "Quantity", "Price"]),
                row(&["Apple", "3", "1.20"]),
                row(&["Banana", "", "0.50"]),
                row(&["Cherry ", " 100", "12.00"]),
                Block::Paragraph(vec![Inline::plain("After\tit.")]),
            ],
            custom(),
        );
        assert_eq!(
            shown(&blocks),
            [
                "Price list follows.",
                "Item|Quantity|Price / Apple|3|1.20 / Banana||0.50 / Cherry|100|12.00",
                "After\tit.",
            ]
        );
        let Block::Table(table) = &blocks[1] else { unreachable!() };
        assert_eq!(table.header_rows, 0);
    }

    #[test]
    fn at_the_default_stops_a_run_of_tabs_is_one_column_break() {
        let blocks = read(
            vec![
                row(&["Name", "", "Team", "Room"]),
                row(&["Alexandra", "Ops", "", "12"]),
                row(&["Bo", "", "Sales", "7", ""]),
            ],
            Stops::default(),
        );
        assert_eq!(shown(&blocks), ["Name|Team|Room / Alexandra|Ops|12 / Bo|Sales|7"]);
    }

    #[test]
    fn a_bold_first_row_is_the_header() {
        let bold = |text: &str| Inline::Text {
            text: text.into(),
            style: Style { bold: true, ..Style::PLAIN },
        };
        let bold_row = |cells: [&str; 2]| {
            Block::Paragraph(vec![bold(cells[0]), tab(Style::PLAIN), bold(cells[1])])
        };
        let blocks = read(
            vec![bold_row(["Item", "Price"]), row(&["Apple", "1.20"]), row(&["Pear", "0.90"])],
            custom(),
        );
        let [Block::Table(table)] = &blocks[..] else { panic!("{blocks:?}") };
        assert_eq!(table.header_rows, 1);
        // Bold throughout, the first row is no header.
        let blocks = read(
            vec![
                bold_row(["Item", "Price"]),
                bold_row(["Apple", "1.20"]),
                bold_row(["Pear", "0.90"]),
            ],
            custom(),
        );
        let [Block::Table(table)] = &blocks[..] else { panic!("{blocks:?}") };
        assert_eq!(table.header_rows, 0);
    }

    #[test]
    fn rows_are_consecutive_paragraphs() {
        // A paragraph with no tab between two rows ends the run.
        let blocks = read(
            vec![
                row(&["a", "b", "c"]),
                row(&["d", "e", "f"]),
                Block::Paragraph(vec![Inline::plain("between")]),
                row(&["g", "h", "i"]),
            ],
            custom(),
        );
        assert_eq!(shown(&blocks), ["a b c", "d e f", "between", "g h i"]);
        // A column every row leaves empty goes.
        let blocks = read(
            vec![
                row(&["Apple", "", "1.20"]),
                row(&["Pear", "", "0.90"]),
                row(&["Plum", "", "0.20"]),
            ],
            custom(),
        );
        assert_eq!(shown(&blocks), ["Apple|1.20 / Pear|0.90 / Plum|0.20"]);
    }

    #[test]
    fn indentation_lists_contents_and_prose_stay_text() {
        let unchanged = |paragraphs: Vec<Block>, stops: Stops| {
            let expected: Vec<String> =
                shown(&paragraphs).iter().map(|t| t.replace('\t', " ")).collect();
            assert_eq!(shown(&read(paragraphs, stops)), expected);
        };
        // Two rows are not enough.
        unchanged(vec![row(&["a", "b", "c"]), row(&["d", "e", "f"])], custom());
        // A verse indented with a tab, at any stops.
        let verse =
            || vec![row(&["", "Whose woods"]), row(&["", "His house"]), row(&["", "He will"])];
        unchanged(verse(), Stops::default());
        unchanged(verse(), custom());
        // A list typed by hand (as macOS textutil saves an HTML list).
        unchanged(
            vec![row(&["", "•", "One"]), row(&["", "•", "Two"]), row(&["", "•", "Three"])],
            Stops::default(),
        );
        unchanged(
            vec![row(&["•", "a", "x"]), row(&["•", "b", "y"]), row(&["•", "c", "z"])],
            custom(),
        );
        unchanged(vec![row(&["1.", "One"]), row(&["2.", "Two"]), row(&["3.", "Three"])], custom());
        // Field labels.
        unchanged(
            vec![row(&["To:", "All"]), row(&["From:", "Me"]), row(&["Re:", "Tabs"])],
            custom(),
        );
        // Two columns at the default stops.
        unchanged(
            vec![row(&["North", "1"]), row(&["South", "2"]), row(&["West", "3"])],
            Stops::default(),
        );
        // A table of contents: a leader, or page numbers counting up.
        let mut dotted = Stops::default();
        dotted.add(9000, "right", true);
        unchanged(
            vec![row(&["Introduction", "1"]), row(&["Methods", "7"]), row(&["Results", "15"])],
            dotted,
        );
        unchanged(
            vec![
                row(&["1", "Introduction", "1"]),
                row(&["2", "Methods", "7"]),
                row(&["3", "Results", "15"]),
            ],
            Stops::default(),
        );
        // Rows that wrap like prose.
        let long = "word ".repeat(30);
        unchanged(
            vec![row(&["a", "b", &long]), row(&["c", "d", "e"]), row(&["f", "g", "h"])],
            custom(),
        );
        // Different stops, or a different number of cells, end a run.
        let mut other = custom();
        other.add(8000, "left", false);
        let mut blocks = vec![row(&["a", "b", "c"]), row(&["d", "e", "f"]), row(&["g", "h", "i"])];
        let mut rows = TabRows::default();
        rows.paragraph(0, custom());
        rows.paragraph(1, custom());
        rows.paragraph(2, other);
        finish(rows, None, &mut blocks, &mut []);
        assert_eq!(shown(&blocks), ["a b c", "d e f", "g h i"]);
        unchanged(vec![row(&["a", "b", "c"]), row(&["d", "e"]), row(&["g", "h", "i"])], custom());
    }

    #[test]
    fn a_tab_inside_a_link_or_beside_a_line_break_splits_nothing() {
        let link = Block::Paragraph(vec![Inline::Link {
            content: vec![Inline::plain("a"), tab(Style::PLAIN), Inline::plain("b")],
            target: crate::model::LinkTarget::Anchor("x".into()),
        }]);
        let mut broken = row(&["c", "d", "e"]);
        if let Block::Paragraph(inlines) = &mut broken {
            inlines.insert(1, Inline::LineBreak);
        }
        let blocks =
            read(vec![link, broken, row(&["f", "g", "h"]), row(&["i", "j", "k"])], custom());
        assert_eq!(shown(&blocks), ["a b", "c\n d e", "f g h", "i j k"]);
    }

    #[test]
    fn a_row_is_no_heading_and_tabs_elsewhere_are_spaces() {
        // A first row set large would be a heading set by hand, the largest
        // one; as the row of a table it is not, and the bold title is level
        // 1. A note's tab is a space again.
        let mut blocks = vec![
            Block::Paragraph(vec![Inline::plain("Body text that sets the size.")]),
            Block::Paragraph(vec![Inline::Text {
                text: "Region".into(),
                style: Style { bold: true, ..Style::PLAIN },
            }]),
            row(&["A", "B", "C"]),
            row(&["D", "E", "F"]),
            row(&["G", "H", "I"]),
        ];
        let mut looks = Looks::default();
        let mut rows = TabRows::default();
        for (index, size, repeat) in [(0, 24, 20), (1, 36, 1), (2, 48, 1), (3, 24, 1), (4, 24, 1)] {
            let Block::Paragraph(inlines) = &blocks[index] else { unreachable!() };
            let mut para = ParaSize::default();
            looks.text(&mut para, size, &inlines_to_plain_text(inlines).repeat(repeat));
            looks.paragraph(index, para);
            if index >= 2 {
                rows.paragraph(index, custom());
            }
        }
        let mut notes = [Note {
            id: "1".into(),
            kind: crate::model::NoteKind::Footnote,
            blocks: vec![row(&["n", "o"])],
        }];
        finish(rows, Some(looks), &mut blocks, &mut notes);
        assert!(matches!(blocks[1], Block::Heading { level: 1, .. }), "{blocks:?}");
        assert!(matches!(blocks[2], Block::Table(_)), "{blocks:?}");
        assert_eq!(blocks.len(), 3);
        assert_eq!(shown(&notes[0].blocks), ["n o"]);
    }

    #[test]
    fn code_lines_and_plain_text_read_tabs_as_spaces() {
        let inlines = vec![
            Inline::plain("a\tb"),
            tab(Style::PLAIN),
            Inline::Link {
                content: vec![tab(Style::PLAIN)],
                target: crate::model::LinkTarget::Anchor("x".into()),
            },
        ];
        assert_eq!(plain_text(&inlines), "a\tb  ");
    }

    #[test]
    fn labels_bullets_and_page_numbers() {
        for label in ["1", "1.", "2.1", "a)", "(iv)", "B.", "一、", "（二）"] {
            assert!(is_number_label(label), "{label}");
        }
        for text in ["Apple", "1.20 kg", "12 kg", "", "Total:"] {
            assert!(!is_number_label(text), "{text}");
        }
        for bullet in ["•", "◦", "-", "\u{f0b7}"] {
            assert!(is_bullet(bullet), "{bullet}");
        }
        assert!(!is_bullet("ab") && !is_bullet("1"));
        assert!(is_page_number("15") && is_page_number("xii") && !is_page_number("1.20"));
        assert!(counts_up(["1", "7", "7", "15"].into_iter()));
        assert!(!counts_up(["3", "12", "1"].into_iter()));
    }
}
