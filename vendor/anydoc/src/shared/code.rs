//! markitai: code set in a monospaced font.
//!
//! Word processors have no notion of code: a writer, or an exporter such as
//! macOS `textutil`, sets a snippet in Courier, Consolas or Menlo. A
//! paragraph set entirely in such a font is read as a line of a code block
//! and a run of it inside prose as inline code - unless monospace is what
//! the document is set in (a typewriter-style manuscript), where it says
//! nothing. The Word, OpenDocument and RTF readers resolve each run's font
//! their own way and share these decisions.

use crate::model::{Block, CellSlot, Inline};

/// Share of a document's text, in characters, above which a monospaced font
/// is the document's typeface rather than a marker of code.
pub const BODY_SHARE: f64 = 0.75;

/// Monospaced families met in documents, matched case-insensitively. A name
/// with the word `Mono` or `Monospace` (`Roboto Mono`, `SF Mono`) also
/// counts; `Monotype Corsiva` does not.
const FAMILIES: &[&str] = &[
    "courier",
    "courier new",
    "consolas",
    "menlo",
    "monaco",
    "lucida console",
    "lucida sans typewriter",
    "inconsolata",
    "source code pro",
    "fira code",
    "cascadia code",
    "hack",
    "anonymous pro",
    "fixedsys",
    "terminal",
];

/// Whether a font family name is one of the monospaced faces code is set in.
/// CJK faces with fixed-width Latin (MS Gothic, NSimSun) are body text, not
/// code, and are not listed. A PostScript name (`Menlo-Regular`,
/// `Courier-Bold`, `CourierNewPSMT`, `SFMono-Regular`, as macOS writes in
/// RTF) names its family before the hyphen.
pub fn is_monospace(font: &str) -> bool {
    let name = font.trim().trim_matches(['\'', '"']).trim().to_ascii_lowercase();
    if FAMILIES.contains(&name.as_str())
        || name
            .split([' ', '-', '_'])
            .any(|word| matches!(word, "mono" | "monospace" | "monospaced"))
    {
        return true;
    }
    // A PostScript name: the family, without spaces, before the face.
    let family = name.split('-').next().unwrap_or_default();
    let family = ["psmt", "ps", "mt"]
        .iter()
        .find_map(|suffix| family.strip_suffix(suffix))
        .unwrap_or(family);
    FAMILIES.iter().any(|known| known.replace(' ', "") == family)
        || (family.len() > 4 && family.ends_with("mono"))
}

/// CJK faces whose Latin letters are fixed-width: the format may call them
/// fixed-pitch (RTF `\fmodern`, ODF `style:font-pitch="fixed"`), yet they
/// set body text.
const CJK_FACES: &[&str] = &[
    "ms gothic",
    "ms mincho",
    "msgothic",
    "msmincho",
    "ms pgothic",
    "ms pmincho",
    "simsun",
    "nsimsun",
    "simhei",
    "fangsong",
    "kaiti",
    "mingliu",
    "pmingliu",
    "batang",
    "gulim",
    "dotum",
    "gungsuh",
    "osaka",
    "hiragino",
    "yu gothic",
    "yu mincho",
    "meiryo",
    "malgun",
    "microsoft yahei",
    "microsoft jhenghei",
    "dfkai",
    "songti",
    "heiti",
    "stsong",
    "stheiti",
    "stkaiti",
    "stfangsong",
    "pingfang",
    "apple sd gothic",
    "applegothic",
    "applemyungjo",
    "noto sans cjk",
    "noto serif cjk",
    "source han",
];

/// Whether a face the document only declares fixed-pitch (RTF `\fmodern` or
/// `\fprq1`, ODF `style:font-pitch="fixed"`) is a code face: any such face
/// with a Latin name, except the CJK faces whose Latin letters happen to be
/// fixed-width.
pub fn is_fixed_pitch_code_face(font: &str) -> bool {
    let name = font.trim().trim_matches(['\'', '"']).trim().to_ascii_lowercase();
    !name.is_empty() && name.is_ascii() && !CJK_FACES.iter().any(|face| name.starts_with(face))
}

/// How much of a document's text, in visible characters, is set in a
/// monospaced font.
#[derive(Debug, Clone, Copy, Default)]
pub struct MonoShare {
    mono: usize,
    total: usize,
}

impl MonoShare {
    /// Text was read, in a monospaced font or not.
    pub fn text(&mut self, text: &str, mono: bool) {
        let visible = text.chars().filter(|c| !c.is_whitespace()).count();
        self.total += visible;
        if mono {
            self.mono += visible;
        }
    }

    /// Whether a monospaced font marks code: it sets at most
    /// [`BODY_SHARE`] of the text.
    pub fn sets_code_apart(self) -> bool {
        self.mono as f64 <= BODY_SHARE * self.total as f64
    }
}

/// The runs of a paragraph that carry text, and how many of them are set in
/// a monospaced font; runs of only spaces are counted apart.
#[derive(Debug, Default, Clone, Copy)]
pub struct RunFonts {
    text: usize,
    mono: usize,
    blank: usize,
    blank_mono: usize,
}

impl RunFonts {
    /// A run with some text (`blank` when it is all whitespace).
    pub fn run(&mut self, blank: bool, mono: bool) {
        if blank {
            self.blank += 1;
            self.blank_mono += usize::from(mono);
        } else {
            self.text += 1;
            self.mono += usize::from(mono);
        }
    }

    /// Take in the runs of content read apart (a link's, a field's).
    pub fn add(&mut self, other: RunFonts) {
        self.text += other.text;
        self.mono += other.mono;
        self.blank += other.blank;
        self.blank_mono += other.blank_mono;
    }

    /// Whether every run with text, or with none every blank run, is
    /// monospaced; `None` without either.
    pub fn all_mono(self) -> Option<bool> {
        if self.text > 0 {
            Some(self.mono == self.text)
        } else if self.blank > 0 {
            Some(self.blank_mono == self.blank)
        } else {
            None
        }
    }
}

/// Inline content with no code styling (a heading set in a monospaced font
/// is typography, not code).
pub fn without_code(inlines: &mut [Inline]) {
    for inline in inlines {
        match inline {
            Inline::Text { style, .. } => style.code = false,
            Inline::Link { content, .. } => without_code(content),
            _ => {}
        }
    }
}

/// Whether inline content holds an image (a paragraph with one is never a
/// line of code: a code block would keep only its alt text).
pub fn has_image(inlines: &[Inline]) -> bool {
    inlines.iter().any(|inline| match inline {
        Inline::Image { .. } => true,
        Inline::Link { content, .. } => has_image(content),
        _ => false,
    })
}

/// A code block that carries its line numbers (a web page's numbered
/// listing saved as a document), as the code alone. The numbers either come
/// first as a column (`1`, `2`, then the two lines) or alternate with the
/// lines (`1`, the first line, `2`, ...). They count up from the first line
/// and number every line of code; a lone line counts only from 1. `None`
/// when the block does not have that shape.
pub fn without_line_gutter(text: &str) -> Option<String> {
    let lines: Vec<&str> = text.split('\n').collect();
    let number = |line: &str| line.trim().parse::<u64>().ok();
    let start = number(lines[0])?;
    let counts = |i: usize, line: &str| number(line) == start.checked_add(i as u64);
    let half = lines.len() / 2;
    if !lines.len().is_multiple_of(2) || (half < 2 && start != 1) {
        return None;
    }
    let column = lines[..half].iter().enumerate().all(|(i, line)| counts(i, line))
        && !counts(half, lines[half]);
    if column {
        return Some(lines[half..].join("\n"));
    }
    let alternating = lines.iter().step_by(2).enumerate().all(|(i, line)| counts(i, line));
    alternating.then(|| lines.iter().skip(1).step_by(2).copied().collect::<Vec<_>>().join("\n"))
}

/// Tables that only lay out a code listing, as code blocks. A syntax
/// highlighter sets a numbered listing as a table: one row holding the code,
/// after a cell of its line numbers (Chroma, Pygments, Rouge, Hexo), or one
/// row per line, its number then the line (GitHub). Saving such a page,
/// `textutil` keeps the table in ODT and RTF (and flattens it in DOCX); its
/// code cells are inline code, as a table cell holds no code block. A table
/// whose rows hold such cells and nothing else becomes the code, without the
/// numbers; a single cell of code (a code box) also does. A table with a
/// header row, more than two cells in a row, or any cell that is not all
/// code or line numbers stays a table.
pub fn listing_tables(blocks: &mut [Block]) {
    for block in blocks {
        match block {
            Block::Table(table) => {
                if let Some(text) = listing(table) {
                    *block = Block::CodeBlock { lang: None, text };
                }
            }
            Block::BlockQuote(inner) => listing_tables(inner),
            Block::List(list) => {
                for item in &mut list.items {
                    listing_tables(&mut item.blocks);
                }
            }
            _ => {}
        }
    }
}

/// The code a table lays out (see [`listing_tables`]).
fn listing(table: &crate::model::Table) -> Option<String> {
    if table.header_rows > 0 {
        return None;
    }
    let rows: Vec<Vec<&crate::model::Cell>> = table
        .grid
        .iter()
        .map(|row| {
            row.iter()
                .filter_map(|slot| match slot {
                    CellSlot::Origin(cell) if !cell.is_empty() => Some(cell),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .filter(|row| !row.is_empty())
        .collect();
    // One row of code, or one numbered line per row.
    if rows.is_empty() || rows.iter().any(|row| row.len() > 2 || (rows.len() > 1 && row.len() != 2))
    {
        return None;
    }
    let mut code = Vec::new();
    for row in &rows {
        let (cell, gutter) = match row[..] {
            [cell] => (cell, None),
            [gutter, cell] => (cell, Some(gutter)),
            _ => return None,
        };
        if let Some(gutter) = gutter {
            let numbers = cell_lines(gutter, false)?;
            if !numbers
                .iter()
                .all(|line| line.trim().is_empty() || line.trim().parse::<u64>().is_ok())
            {
                return None;
            }
        }
        code.extend(cell_lines(cell, true)?);
    }
    let first = code.iter().position(|line| !line.trim().is_empty())?;
    let last = code.iter().rposition(|line| !line.trim().is_empty())?;
    Some(code[first..=last].join("\n"))
}

/// The lines of a cell of paragraphs (a line break inside one starts a new
/// line); with `code`, `None` unless every piece of its visible text is
/// inline code.
fn cell_lines(cell: &crate::model::Cell, code: bool) -> Option<Vec<String>> {
    fn line(inlines: &[Inline], code: bool, lines: &mut Vec<String>) -> Option<()> {
        for inline in inlines {
            match inline {
                Inline::Text { text, style } => {
                    if code && !style.code && !text.trim().is_empty() {
                        return None;
                    }
                    lines.last_mut()?.push_str(text);
                }
                Inline::LineBreak => lines.push(String::new()),
                Inline::Link { content, .. } => line(content, code, lines)?,
                Inline::Anchor(_) => {}
                _ => return None,
            }
        }
        Some(())
    }
    let mut lines = Vec::new();
    for block in &cell.blocks {
        let Block::Paragraph(inlines) = block else { return None };
        lines.push(String::new());
        line(inlines, code, &mut lines)?;
    }
    Some(lines)
}

/// Every code block in `blocks`, at any depth, without the line numbers it
/// carries (see [`without_line_gutter`]).
pub fn drop_line_gutters(blocks: &mut [Block]) {
    for block in blocks {
        match block {
            Block::CodeBlock { text, .. } => {
                if let Some(code) = without_line_gutter(text) {
                    *text = code;
                }
            }
            Block::BlockQuote(inner) => drop_line_gutters(inner),
            Block::List(list) => {
                for item in &mut list.items {
                    drop_line_gutters(&mut item.blocks);
                }
            }
            Block::Table(table) => {
                for slot in table.grid.iter_mut().flatten() {
                    if let CellSlot::Origin(cell) = slot {
                        drop_line_gutters(&mut cell.blocks);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Blocks as short strings for the readers' tests: `h1:text`, `p:text`
/// (inline code in backticks), `code:text`, `list:item|item` and
/// `table:cell|cell/cell|cell`, a cell's or item's blocks joined by `;`.
#[cfg(test)]
pub(crate) fn describe(blocks: &[Block]) -> Vec<String> {
    fn text(inlines: &[Inline]) -> String {
        let mut out = String::new();
        for inline in inlines {
            match inline {
                Inline::Text { text, style } if style.code && !text.trim().is_empty() => {
                    out.push('`');
                    out.push_str(text);
                    out.push('`');
                }
                Inline::Text { text, .. } => out.push_str(text),
                Inline::Link { content, .. } => out.push_str(&text(content)),
                Inline::LineBreak => out.push('\n'),
                _ => {}
            }
        }
        out
    }
    let joined = |blocks: &[Block]| describe(blocks).join(";");
    blocks
        .iter()
        .map(|block| match block {
            Block::Heading { level, content, .. } => format!("h{level}:{}", text(content)),
            Block::Paragraph(inlines) => format!("p:{}", text(inlines)),
            Block::CodeBlock { text, .. } => format!("code:{text}"),
            Block::List(list) => format!(
                "list:{}",
                list.items.iter().map(|item| joined(&item.blocks)).collect::<Vec<_>>().join("|")
            ),
            Block::Table(table) => format!(
                "table:{}",
                table
                    .grid
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|slot| match slot {
                                CellSlot::Origin(cell) => joined(&cell.blocks),
                                CellSlot::Covered { .. } => "^".into(),
                            })
                            .collect::<Vec<_>>()
                            .join("|")
                    })
                    .collect::<Vec<_>>()
                    .join("/")
            ),
            other => format!("{other:?}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_table_laying_out_a_listing_is_its_code() {
        use crate::model::{Cell, Style, Table, TableKind};
        let code = |text: &str| Inline::Text {
            text: text.into(),
            style: Style { code: true, ..Style::PLAIN },
        };
        let lines = |texts: &[&str]| -> Cell {
            Cell::new(texts.iter().map(|text| Block::Paragraph(vec![code(text)])).collect())
        };
        let table = |rows: Vec<Vec<Cell>>| Block::Table(Table::from_rows(rows, 0, TableKind::Data));
        let mut blocks = vec![
            // A cell of line numbers, then the code (blank lines kept).
            table(vec![vec![lines(&["1", "2", "3"]), lines(&["a = 1", " ", "b = 2"])]]),
            // One numbered line per row, a line break inside a cell.
            table(vec![
                vec![
                    lines(&["1"]),
                    Cell::new(vec![Block::Paragraph(vec![
                        code("x"),
                        Inline::LineBreak,
                        code("y"),
                    ])]),
                ],
                vec![lines(&["2"]), lines(&["z"])],
            ]),
            // A code box.
            table(vec![vec![lines(&["make"])]]),
            // A gutter that is not numbers, prose in a cell, a column of
            // code cells or a header stay tables.
            table(vec![vec![lines(&["a"]), lines(&["b"])]]),
            table(vec![vec![lines(&["1"]), Cell::from_inlines(vec![Inline::plain("prose")])]]),
            table(vec![vec![lines(&["ls"])], vec![lines(&["cd"])]]),
            Block::Table(Table::from_rows(vec![vec![lines(&["make"])]], 1, TableKind::Data)),
        ];
        listing_tables(&mut blocks);
        assert_eq!(
            describe(&blocks),
            [
                "code:a = 1\n \nb = 2",
                "code:x\ny\nz",
                "code:make",
                "table:p:`a`|p:`b`",
                "table:p:`1`|p:prose",
                "table:p:`ls`/p:`cd`",
                "table:p:`make`",
            ]
        );
    }

    #[test]
    fn code_faces_are_monospace_and_body_faces_are_not() {
        for font in [
            "Courier New",
            "Consolas",
            "Menlo",
            " monaco ",
            "Roboto Mono",
            "PT-Mono",
            "'Liberation Mono'",
            // PostScript names, as macOS writes them in RTF.
            "Menlo-Regular",
            "Courier-Bold",
            "CourierNewPSMT",
            "SFMono-Regular",
            "JetBrainsMono-Italic",
        ] {
            assert!(is_monospace(font), "{font}");
        }
        for font in [
            "Times",
            "Times-Roman",
            "Calibri",
            "Arial",
            "MS Gothic",
            "NSimSun",
            "Monotype Corsiva",
            "MonotypeCorsiva",
            "HelveticaNeue",
            "Helvetica-Bold",
        ] {
            assert!(!is_monospace(font), "{font}");
        }
    }

    #[test]
    fn a_fixed_pitch_face_is_code_unless_it_sets_cjk_text() {
        for font in ["Courier 10 Pitch", "OCR A Extended", "Iosevka", "'Letter Gothic'"] {
            assert!(is_fixed_pitch_code_face(font), "{font}");
        }
        for font in
            ["MS Gothic", "MS Mincho Western", "SimSun", "ＭＳ ゴシック", "宋体", "", "Osaka"]
        {
            assert!(!is_fixed_pitch_code_face(font), "{font}");
        }
    }

    #[test]
    fn monospace_marks_code_up_to_three_quarters_of_the_text() {
        let share = |plain: &str, mono: &str| {
            let mut share = MonoShare::default();
            share.text(plain, false);
            share.text(mono, true);
            share.sets_code_apart()
        };
        // Whitespace is not counted: 8 of 12 characters, then 3 of 4.
        assert!(share("ab cd", "efgh ijkl"));
        assert!(share("a", "b c\td"));
        // 8 of 10 is the document's typeface.
        assert!(!share("a b", "efgh ijkl"));
        assert!(MonoShare::default().sets_code_apart());
    }

    #[test]
    fn a_paragraph_is_mono_by_its_text_runs_else_its_blank_ones() {
        let mut fonts = RunFonts::default();
        assert_eq!(fonts.all_mono(), None);
        fonts.run(true, false);
        assert_eq!(fonts.all_mono(), Some(false));
        fonts.run(false, true);
        assert_eq!(fonts.all_mono(), Some(true), "a body-font space between code runs");
        let mut other = RunFonts::default();
        other.run(false, false);
        fonts.add(other);
        assert_eq!(fonts.all_mono(), Some(false));
    }

    #[test]
    fn numbered_lines_lose_their_numbers_only_when_every_line_is_counted() {
        let stripped = |text| without_line_gutter(text);
        // Alternating with the lines, from any first number.
        assert_eq!(stripped("1\nfn main() {\n2\n}").as_deref(), Some("fn main() {\n}"));
        assert_eq!(stripped("7\na\n8\n    b\n9\nc").as_deref(), Some("a\n    b\nc"));
        // As a column before them.
        assert_eq!(stripped("1\n2\n3\nx\n  y\nz").as_deref(), Some("x\n  y\nz"));
        assert_eq!(stripped("1\necho hello").as_deref(), Some("echo hello"));
        // A skipped number, an odd count, a lone line not counted from 1, a
        // block of numbers or a first line that is no number stay as they are.
        for text in ["1\na\n3\nb", "1\na\n2", "5\na", "1\n2\n3\n4", "x\n1\ny\n2", "1\n2\n3\na"] {
            assert_eq!(stripped(text), None, "{text:?}");
        }
    }

    #[test]
    fn numbered_listings_lose_their_numbers_at_any_depth() {
        let code = |text: &str| Block::CodeBlock { lang: None, text: text.into() };
        let mut blocks =
            vec![code("1\nx"), Block::BlockQuote(vec![code("1\n2\na\nb")]), code("not\nnumbered")];
        drop_line_gutters(&mut blocks);
        let texts: Vec<&str> = blocks
            .iter()
            .map(|block| match block {
                Block::CodeBlock { text, .. } => text.as_str(),
                Block::BlockQuote(inner) => match &inner[0] {
                    Block::CodeBlock { text, .. } => text.as_str(),
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(texts, ["x", "a\nb", "not\nnumbered"]);
    }
}
