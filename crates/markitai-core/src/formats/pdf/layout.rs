//! Layout improvements only for pages with a reliable, upright text layer.
use super::geometry::{Frame, Grid, Mark};
use pdf_inspector::{TextItem, types::ItemType};
use std::collections::{BTreeMap, HashSet};

#[path = "links.rs"]
mod links;
#[cfg(test)]
#[path = "structure_tests.rs"]
mod structure_tests;
#[path = "unruled.rs"]
mod unruled;

const MAX_ITEMS: usize = 250_000;
const MAX_TEXT: usize = 16 * 1024 * 1024;
const MAX_PAGE_ITEMS: usize = 20_000;

pub(super) struct Layout {
    pages: BTreeMap<u32, Vec<TextItem>>,
    faces: Faces,
    carry: Carry,
}

/// The document's type: its heading sizes, and the faces of its text, the
/// body's and those its headings are set in. A browser sets `<h4>`–`<h6>` at
/// or below the body size in a bold face, so size alone does not find them;
/// the face does, when the same face sets a heading by size elsewhere or is
/// bold.
#[derive(Default)]
struct Faces {
    /// Heading sizes, largest first.
    headings: Vec<f32>,
    /// The body text's size.
    body: f32,
    /// The face with most characters at the body size.
    body_font: Option<String>,
    /// Faces other than the body's that set text at a heading size, and bold
    /// faces.
    heading_fonts: HashSet<String>,
}

/// What a page passes to the next one it renders.
#[derive(Default)]
struct Carry {
    /// The prose line pitch, in em, of the last page that showed one.
    pitch: Option<f32>,
    /// The page whose last block was a table that may continue, with what
    /// its continuation keeps to.
    table: Option<(u32, Ending)>,
}

/// A table that ends a page, low enough on it to continue on the next.
enum Ending {
    /// A ruled table's column borders and header cells.
    Ruled { xs: Vec<f32>, header: Vec<String> },
    /// A borderless table's columns and line pitch.
    Unruled(unruled::Shape),
}

/// A table continues on the next page only when it ends in the lowest
/// fifth of its page: a table the page break did not cut ends higher, and a
/// table at the top of the next page is then a new one.
const CONTINUED_BAND: f32 = 0.2;
/// Column borders of a table and of its continuation agree within this many
/// points.
const SAME_BORDER: f32 = 1.5;

/// What borderless-table detection on a page needs beyond the page's text.
struct Tables<'a> {
    /// The table that ended the previous page.
    continued: Option<&'a Ending>,
    /// The running text's line pitch on the last page that showed one;
    /// this page's own replaces it.
    pitch: &'a mut Option<f32>,
    /// Whether the page's structure tree holds table cells: the page reader
    /// reads those tables from the tags, and geometry defers to them.
    tagged: &'a dyn Fn() -> bool,
}

impl Layout {
    /// The selected pages' positioned text from the document the page
    /// reader loaded; `None` when it could not load the file.
    pub(super) fn read(
        pdf: Option<&pdf_inspector::LoadedPdf>,
        selected: &HashSet<u32>,
        geometry: &BTreeMap<u32, (Frame, Vec<Grid>, Vec<Mark>)>,
    ) -> std::result::Result<Self, &'static str> {
        let Some(Ok((items, rotations))) = pdf.map(|pdf| {
            pdf.text_with_positions_and_rotations(
                Some(selected),
                pdf_inspector::PositionOptions::new().bold_from_weight(true),
            )
        }) else {
            return Err("positioned text could not be decoded");
        };
        if items.len() > MAX_ITEMS || items.iter().map(|i| i.text.len()).sum::<usize>() > MAX_TEXT {
            return Err("positioned text exceeds the layout budget");
        }
        let mut pages: BTreeMap<u32, Vec<TextItem>> = BTreeMap::new();
        let mut sizes = BTreeMap::<i32, usize>::new();
        // Characters per size and face, and the bold faces.
        let mut fonts = BTreeMap::<(i32, String), usize>::new();
        let mut bold_fonts = HashSet::new();
        for item in items {
            if rotations.contains_key(&item.page) {
                continue;
            }
            // Table cells often use a smaller size than prose; counting them
            // would make ordinary paragraphs look like headings.
            let in_table = geometry.get(&item.page).is_some_and(|(_, grids, _)| {
                let (x, y) = (item.x + item.width / 2., item.y + item.height / 2.);
                grids.iter().any(|grid| {
                    x >= grid.xs[0]
                        && x <= grid.xs[grid.xs.len() - 1]
                        && y >= grid.ys[0]
                        && y <= grid.ys[grid.ys.len() - 1]
                })
            });
            // Fixed-pitch code blocks do not define the body size either.
            let code = item.fixed_pitch == Some(true);
            if matches!(item.item_type, ItemType::Text) && valid(&item) && !in_table && !code {
                let key = (item.font_size * 10.).round() as i32;
                let count = item.text.chars().filter(|c| !c.is_whitespace()).count();
                *sizes.entry(key).or_default() += count;
                *fonts.entry((key, item.font.clone())).or_default() += count;
                if item.is_bold && !item.is_italic {
                    bold_fonts.insert(item.font.clone());
                }
            }
            pages.entry(item.page).or_default().push(item);
        }
        let body = sizes
            .iter()
            .max_by_key(|(_, count)| **count)
            .map_or(12., |(size, _)| *size as f32 / 10.);
        let mut headings = Vec::<f32>::new();
        for (&size, _) in sizes.iter().rev() {
            let size = size as f32 / 10.;
            if size > body * 1.15
                && headings
                    .last()
                    .is_none_or(|previous| (*previous - size).abs() > size * 0.05)
            {
                headings.push(size);
            }
        }
        let body_key = (body * 10.).round() as i32;
        let body_font = fonts
            .iter()
            .filter(|((key, _), _)| *key == body_key)
            .max_by_key(|(_, count)| **count)
            .map(|((_, font), _)| font.clone());
        let mut heading_fonts: HashSet<String> = fonts
            .keys()
            .filter(|(key, _)| *key as f32 / 10. > body * 1.15)
            .map(|(_, font)| font.clone())
            .collect();
        heading_fonts.extend(bold_fonts);
        if let Some(body_font) = &body_font {
            heading_fonts.remove(body_font);
        }
        Ok(Self {
            pages,
            faces: Faces {
                headings,
                body,
                body_font,
                heading_fonts,
            },
            carry: Carry::default(),
        })
    }

    /// Pages are rendered in ascending order: a table ending one page may
    /// continue at the top of the next. `tagged` tells whether the page's
    /// structure tree holds table cells; it is asked only when a borderless
    /// table is found. Also whether the page's Markdown starts with a table
    /// that continues the table ending the previous page's Markdown.
    pub(super) fn page(
        &mut self,
        number: u32,
        frame: Frame,
        grids: Vec<Grid>,
        marks: &[Mark],
        baseline: &str,
        tagged: &dyn Fn() -> bool,
    ) -> Option<(String, bool)> {
        let items = self.pages.remove(&number)?;
        let continued = self
            .carry
            .table
            .take()
            .filter(|(page, _)| page + 1 == number)
            .map(|(_, shape)| shape);
        let mut tables = Tables {
            continued: continued.as_ref(),
            pitch: &mut self.carry.pitch,
            tagged,
        };
        let page = render(
            items,
            &self.faces,
            frame,
            grids,
            marks,
            baseline,
            &mut tables,
        )?;
        self.carry.table = page
            .ending
            .filter(|(bottom, _)| *bottom <= frame.height * CONTINUED_BAND)
            .map(|(_, ending)| (number, ending));
        Some((page.markdown, page.continues))
    }
}

fn valid(item: &TextItem) -> bool {
    [
        item.x,
        item.y,
        item.width,
        item.height,
        item.font_size,
        item.rotation,
        item.baseline_shift,
    ]
    .iter()
    .all(|n| n.is_finite())
        && item.width >= 0.
        && item.height > 0.
        && item.font_size > 1.
        && item.font_size < 1000.
        && item.rotation.abs() < 0.1
        // A super/subscript run is measured from its anchor's baseline.
        && item.baseline_shift.abs() < item.font_size * 1.5
        && !matches!(item.render_mode, Some(3 | 7))
        && !item.text.chars().any(|c| {
            c == '\u{fffd}'
                || (c.is_control() && !c.is_whitespace())
                || matches!(c as u32, 0xe000..=0xf8ff | 0xf0000..=0xffffd | 0x100000..=0x10fffd)
        })
}

/// `text` without the destinations of links the page reader made from bare
/// URLs (`[https://a.b](https://a.b)`): the URL is page text once.
fn without_url_destinations(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains("](") {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut output = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("](") {
        let label = rest[..at].rfind('[').map(|open| &rest[open + 1..at]);
        let tail = &rest[at + 2..];
        match (label, tail.find(')')) {
            (Some(label), Some(close)) if tail[..close] == *label => {
                output.push_str(&rest[..at + 1]);
                rest = &tail[close + 1..];
            }
            _ => {
                output.push_str(&rest[..at + 2]);
                rest = tail;
            }
        }
    }
    output.push_str(rest);
    std::borrow::Cow::Owned(output)
}

/// A letter of a right-to-left script: Hebrew, Arabic, Syriac, Thaana, N'Ko,
/// Samaritan, Mandaic and their presentation forms.
fn is_right_to_left(c: char) -> bool {
    matches!(c as u32, 0x0590..=0x08FF | 0xFB1D..=0xFDFF | 0xFE70..=0xFEFF | 0x10800..=0x10FFF | 0x1E800..=0x1EFFF)
}

fn character_counts(text: &str) -> BTreeMap<char, usize> {
    let mut counts = BTreeMap::new();
    // These are generated decorations in the original native Markdown, not
    // arbitrary source HTML to interpret. Unknown constructs fail agreement.
    let text = without_url_destinations(text)
        .replace("<u>", "")
        .replace("</u>", "")
        .replace("<s>", "")
        .replace("</s>", "")
        .replace("<sup>", "")
        .replace("</sup>", "")
        .replace("<sub>", "")
        .replace("</sub>", "")
        .replace("<br>", "");
    for c in text.chars().filter(|c| c.is_alphanumeric()) {
        *counts.entry(c).or_default() += 1;
    }
    counts
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Style {
    bold: bool,
    italic: bool,
    underline: bool,
    strike: bool,
    code: bool,
    /// Raised (`1`) or lowered (`-1`) off the anchor's baseline.
    script: i8,
}

impl From<&TextItem> for Style {
    fn from(item: &TextItem) -> Self {
        let text = item.text.trim();
        Self {
            bold: item.is_bold,
            italic: item.is_italic,
            underline: item.is_underline,
            strike: item.is_strikeout,
            // A fixed-pitch run in prose is an inline code literal; a mono
            // link or bare URL is link styling, and a run holding a
            // backtick stays text rather than being escaped.
            code: item.fixed_pitch == Some(true)
                && !item.is_underline
                && !is_bare_url(text)
                && !text.contains('`')
                && item.baseline_shift == 0.,
            script: if item.baseline_shift > 0. {
                1
            } else if item.baseline_shift < 0. {
                -1
            } else {
                0
            },
        }
    }
}

/// Whether a run is nothing but a URL: one word starting with a scheme
/// (`https://…`) or `www.`, optionally in angle brackets. A word that only
/// contains one (`baseUrl="https://…"`) is code or prose, not a link.
fn is_bare_url(text: &str) -> bool {
    let text = text.strip_prefix('<').unwrap_or(text);
    if text.contains(char::is_whitespace) {
        return false;
    }
    if text.starts_with("www.") {
        return true;
    }
    text.find("://").is_some_and(|at| {
        let scheme = &text[..at];
        scheme.starts_with(|c: char| c.is_ascii_alphabetic())
            && scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '.' | '-'))
    })
}

struct Run {
    text: String,
    style: Style,
    /// The target of the link annotation over the run (see [`links::apply`]).
    link: Option<std::sync::Arc<str>>,
}

impl Run {
    /// Whether `next` continues this run: the same style and link.
    fn joins(&self, style: Style, link: Option<&std::sync::Arc<str>>) -> bool {
        self.style == style && self.link.as_ref() == link
    }
}

struct Line {
    items: Vec<TextItem>,
    y: f32,
    size: f32,
}

/// The baseline an item's line is set on: a script run's is its anchor's.
fn line_y(item: &TextItem) -> f32 {
    item.y - item.baseline_shift
}

/// A stable sort of text items. Its comparator is a trait object, so the two
/// orders `lines` sorts by share one compiled sort.
fn sort_items(
    items: &mut [TextItem],
    compare: &mut dyn FnMut(&TextItem, &TextItem) -> std::cmp::Ordering,
) {
    items.sort_by(|a, b| compare(a, b));
}

fn lines(mut items: Vec<TextItem>) -> Vec<Line> {
    sort_items(&mut items, &mut |a, b| {
        line_y(b).total_cmp(&line_y(a)).then(a.x.total_cmp(&b.x))
    });
    let mut result: Vec<Line> = Vec::new();
    for item in items {
        let y = line_y(&item);
        if let Some(line) = result.last_mut()
            && (line.y - y).abs() <= (line.size.min(item.font_size) * 0.2).min(2.)
        {
            line.size = line.size.max(item.font_size);
            line.items.push(item);
        } else {
            result.push(Line {
                y,
                size: item.font_size,
                items: vec![item],
            });
        }
    }
    for line in &mut result {
        sort_items(&mut line.items, &mut |a, b| a.x.total_cmp(&b.x));
    }
    attach_raised(&mut result);
    result
}

/// Short runs raised off a line that the extractor did not read as
/// superscripts (a reference's `^ a b` back-links, a footnote mark in
/// another face) join the line they are raised from: on a baseline of their
/// own they would form a line between it and the line above, and join the
/// paragraph above. Such a line has at most twelve characters, all smaller
/// than the line below, whose baseline is at most 0.6 of that line's size
/// below theirs (a superscript's raise), and stands within that line's extent, or two em before or
/// after it (a footnote's leading number), without overlapping any of its
/// runs. Its runs keep their raise, as script runs do.
fn attach_raised(lines: &mut Vec<Line>) {
    let mut index = 0;
    while index + 1 < lines.len() {
        let (raised, below) = (&lines[index], &lines[index + 1]);
        let rise = raised.y - below.y;
        let chars = raised
            .items
            .iter()
            .map(|i| i.text.trim().chars().count())
            .sum::<usize>();
        let start = below.items.first().map_or(0., |i| i.x);
        let end = below
            .items
            .iter()
            .map(|i| i.x + i.width)
            .fold(f32::NEG_INFINITY, f32::max);
        let fits = raised.items.iter().all(|item| {
            item.font_size <= below.size * 0.9
                && item.baseline_shift == 0.
                && item.x >= start - below.size * 2.
                && item.x + item.width <= end + below.size * 2.
                && below.items.iter().all(|other| {
                    item.x + item.width <= other.x + 0.5 || other.x + other.width <= item.x + 0.5
                })
        });
        if (1..=12).contains(&chars) && rise > 0. && rise <= below.size * 0.6 && fits {
            let raised = lines.remove(index);
            let below = &mut lines[index];
            for mut item in raised.items {
                item.baseline_shift = item.y - below.y;
                below.items.push(item);
            }
            sort_items(&mut below.items, &mut |a, b| a.x.total_cmp(&b.x));
        } else {
            index += 1;
        }
    }
}

fn runs(line: &Line) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    let mut previous: Option<&TextItem> = None;
    for item in &line.items {
        let text = item.text.trim();
        if text.is_empty() {
            continue;
        }
        let space = previous.is_some_and(|p| {
            p.text.ends_with(char::is_whitespace)
                || item.text.starts_with(char::is_whitespace)
                || item.x - (p.x + p.width) > item.font_size.min(p.font_size) * 0.12
        });
        let style = Style::from(item);
        let link = match &item.item_type {
            ItemType::Link(target) => Some(std::sync::Arc::<str>::from(target.as_str())),
            _ => None,
        };
        if let Some(last) = out.last_mut()
            && last.joins(style, link.as_ref())
        {
            if space {
                last.text.push(' ');
            }
            last.text.push_str(text);
        } else {
            if space && let Some(last) = out.last_mut() {
                last.text.push(' ');
            }
            out.push(Run {
                text: text.into(),
                style,
                link,
            });
        }
        previous = Some(item);
    }
    out
}

fn append_runs(output: &mut Vec<Run>, source: Vec<Run>) {
    for (index, run) in source.into_iter().enumerate() {
        if let Some(last) = output.last_mut()
            && last.joins(run.style, run.link.as_ref())
        {
            if index == 0 && !last.text.ends_with(char::is_whitespace) {
                last.text.push(' ');
            }
            last.text.push_str(&run.text);
        } else {
            if index == 0
                && let Some(last) = output.last_mut()
                && !last.text.ends_with(char::is_whitespace)
            {
                last.text.push(' ');
            }
            output.push(run);
        }
    }
}

/// The Markdown of a block's runs. Text is escaped as the other readers
/// escape it (`super::super::escape`): `<` only where it would open an HTML
/// tag, comment or entity, so `a < b`, `x -> y` and `Vec<String>` read as
/// written. Consecutive runs under one link annotation form one link, which
/// its underline is the styling of; a link whose text is its target is an
/// autolink.
fn markdown(runs: &[Run]) -> String {
    let mut output = String::new();
    // The link open in `output`: its target, where its text starts and that
    // text as the page shows it.
    let mut open: Option<(&std::sync::Arc<str>, usize, String)> = None;
    let close = |output: &mut String,
                 (target, start, shown): (&std::sync::Arc<str>, usize, String)| {
        let trimmed = output.trim_end().len();
        let space = trimmed < output.len();
        output.truncate(trimmed);
        if shown.trim() == &**target {
            output.replace_range(start.., &format!("<{target}>"));
        } else {
            output.insert(start, '[');
            output.push_str("](");
            output.push_str(target);
            output.push(')');
        }
        if space {
            output.push(' ');
        }
    };
    for run in runs {
        let text = run.text.trim();
        if text.is_empty() {
            continue;
        }
        if open
            .as_ref()
            .is_some_and(|(target, _, _)| run.link.as_ref() != Some(*target))
        {
            close(&mut output, open.take().expect("a link is open"));
        }
        if run.text.starts_with(char::is_whitespace) && !output.is_empty() && !output.ends_with(' ')
        {
            output.push(' ');
        }
        match (&mut open, &run.link) {
            (Some((_, _, shown)), _) => {
                if run.text.starts_with(char::is_whitespace) && !shown.is_empty() {
                    shown.push(' ');
                }
                shown.push_str(text);
            }
            (None, Some(target)) => open = Some((target, output.len(), text.to_owned())),
            (None, None) => {}
        }
        let style = run.style;
        // Code is verbatim and exclusive: no emphasis or escaping inside.
        if style.code {
            output.push('`');
            output.push_str(text);
            output.push('`');
            if run.text.ends_with(char::is_whitespace) {
                output.push(' ');
            }
            continue;
        }
        // A link's underline is how the page shows it is a link.
        let underline = style.underline && run.link.is_none();
        if style.bold {
            output.push_str("**");
        }
        if style.italic {
            output.push('*');
        }
        if underline {
            output.push_str("<u>");
        }
        if style.strike {
            output.push_str("<s>");
        }
        let tag = match style.script {
            1 => Some("sup"),
            -1 => Some("sub"),
            _ => None,
        };
        if let Some(tag) = tag {
            output.push_str(&format!("<{tag}>"));
        }
        output.push_str(&super::super::escape(text));
        if let Some(tag) = tag {
            output.push_str(&format!("</{tag}>"));
        }
        if style.strike {
            output.push_str("</s>");
        }
        if underline {
            output.push_str("</u>");
        }
        if style.italic {
            output.push('*');
        }
        if style.bold {
            output.push_str("**");
        }
        if run.text.ends_with(char::is_whitespace) {
            output.push(' ');
            if let Some((_, _, shown)) = &mut open {
                shown.push(' ');
            }
        }
    }
    if let Some(link) = open.take() {
        close(&mut output, link);
    }
    let output = output.trim();
    // A block whose text starts with `>` would be a quotation.
    if output.starts_with('>') {
        format!("\\{output}")
    } else {
        output.to_owned()
    }
}

fn list_prefix(text: &str) -> Option<&str> {
    let text = text.trim_start();
    for bullet in ["•", "●", "◦", "▪", "- ", "* "] {
        if let Some(rest) = text.strip_prefix(bullet) {
            return Some(rest.trim_start());
        }
    }
    None
}

fn heading_level(line: &Line, sizes: &[f32]) -> usize {
    // A large initial does not turn its entire paragraph into a heading.
    let large = line
        .items
        .iter()
        .filter(|i| i.font_size >= line.size * 0.9)
        .map(|i| i.text.chars().count())
        .sum::<usize>();
    let all = line
        .items
        .iter()
        .map(|i| i.text.chars().count())
        .sum::<usize>();
    if large * 5 < all * 4 {
        return 0;
    }
    sizes
        .iter()
        .position(|s| (line.size - s).abs() <= s * 0.05)
        .map_or(0, |n| (n + 1).min(6))
}

/// Baseline gap, in the larger line's size, from which a line starts a new
/// block in the paragraph flow.
const NEW_BLOCK: f32 = 1.8;

/// Which lines are headings set at the body size: a block of one or two
/// lines whose runs are all in a heading face (see [`Faces`]), neither
/// italic nor fixed-pitch, at the body size, set off from the text before
/// and after it (or next to a heading by size, or at the start or end of
/// the flow), and reading as a title: up to 14 words and 120 characters,
/// with a letter, not ending like a sentence (`.`, `,` or `;`) or, past
/// three words, like a lead-in (`:`). Such a block ranks below every
/// heading size (`<h4>` under `<h3>`). The extractor names a run merged
/// from several faces after its first, so a bold label opening a line
/// (`**1** The first approach…`) can look like a line in that face: the
/// lead-in rule and the set-off around the block keep such lines text.
fn same_size_headings(lines: &[Line], headings: &[f32], faces: &Faces) -> Vec<bool> {
    let face = |line: &Line| {
        heading_level(line, headings) == 0
            && line.size >= faces.body * 0.95
            && line.size <= faces.body * 1.15 + 0.05
            && line.items.iter().all(|item| {
                !item.is_italic
                    && item.fixed_pitch != Some(true)
                    && item.baseline_shift == 0.
                    && faces.body_font.as_deref() != Some(item.font.as_str())
                    && faces.heading_fonts.contains(&item.font)
            })
    };
    let faced: Vec<bool> = lines.iter().map(face).collect();
    let apart = |above: &Line, below: &Line| {
        above.y - below.y > above.size.max(below.size) * NEW_BLOCK
            || heading_level(above, headings) > 0
            || heading_level(below, headings) > 0
    };
    let mut result = vec![false; lines.len()];
    let mut start = 0;
    while start < lines.len() {
        if !faced[start] {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < lines.len()
            && end - start < 3
            && faced[end]
            && !apart(&lines[end - 1], &lines[end])
        {
            end += 1;
        }
        let before = start == 0 || apart(&lines[start - 1], &lines[start]);
        let after = end == lines.len() || apart(&lines[end - 1], &lines[end]);
        let text = lines[start..end]
            .iter()
            .flat_map(|line| &line.items)
            .map(|item| item.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        let words = text.split_whitespace().count();
        if end - start <= 2
            && before
            && after
            && (1..=14).contains(&words)
            && text.chars().count() <= 120
            && text.chars().any(char::is_alphabetic)
            && !text.ends_with(['.', ',', ';'])
            && !(text.ends_with(':') && words > 3)
            && list_prefix(&text).is_none()
        {
            result[start..end].fill(true);
        }
        start = end;
    }
    result
}

/// For each line, the farthest right edge reached by the lines of `lines`
/// starting within a point and a half of it: the right edge of its column.
fn column_rights(lines: &[Line]) -> Vec<f32> {
    let edge = |line: &Line| {
        let start = line.items.first().map_or(0., |i| i.x);
        let end = line
            .items
            .iter()
            .map(|i| i.x + i.width)
            .fold(f32::NEG_INFINITY, f32::max);
        (start, end)
    };
    let mut buckets: BTreeMap<i32, f32> = BTreeMap::new();
    for line in lines {
        let (start, end) = edge(line);
        let bucket = buckets.entry((start * 2.).round() as i32).or_insert(end);
        *bucket = bucket.max(end);
    }
    lines
        .iter()
        .map(|line| {
            let key = (edge(line).0 * 2.).round() as i32;
            buckets
                .range(key - 3..=key + 3)
                .map(|(_, end)| *end)
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect()
}

/// Whether line `above` of `lines` ended short of its column (`right`) by
/// more than the first word of line `below` and an em and a quarter: wrapped
/// text fills each line but a paragraph's last, so `below` starts a new
/// block even without a gap (rows of a link list, lines a `<br>` breaks,
/// paragraphs set without spacing). A line ending in a hyphen continues, as
/// do a label of its own (four characters at most, no letter: a footnote's
/// number), a line followed by one starting in lower case, and a line of five
/// words or more ending within an em of where a neighbouring line of its
/// paragraph ends (the line below, or the line above at the text's pitch
/// from the same left edge): those lines keep a measure of their own, as
/// text beside a floated figure or in a narrower box does.
fn ended_early(lines: &[Line], above: usize, below: usize, right: f32) -> bool {
    let (upper, lower) = (&lines[above], &lines[below]);
    let (Some(last), Some(first)) = (upper.items.last(), lower.items.first()) else {
        return false;
    };
    if last.text.trim_end().ends_with('-') {
        return false;
    }
    // A label of its own (a footnote's number, a marker) heads the line
    // below it.
    let label: String = upper.items.iter().map(|i| i.text.trim()).collect();
    if label.chars().count() <= 4 && !label.chars().any(char::is_alphabetic) {
        return false;
    }
    let end = |line: &Line| {
        line.items
            .iter()
            .map(|i| i.x + i.width)
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let start = |line: &Line| line.items.first().map_or(0., |i| i.x);
    let close = |other: &Line| (end(other) - end(upper)).abs() <= upper.size;
    // Only a line of running text keeps a measure: a short line (a link,
    // a name, a label) ends where its words do.
    let words: usize = upper
        .items
        .iter()
        .map(|i| i.text.split_whitespace().count())
        .sum();
    if words >= 5 && close(lower) {
        return false;
    }
    if words >= 5
        && let Some(before) = above.checked_sub(1).map(|i| &lines[i])
        && before.y - upper.y <= before.size.max(upper.size) * NEW_BLOCK
        && (start(before) - start(upper)).abs() <= 1.5
        && close(before)
    {
        return false;
    }
    let text = first.text.trim_start();
    // A line going on in lower case continues the sentence above.
    if text.starts_with(char::is_lowercase) {
        return false;
    }
    let chars = text.chars().count().max(1);
    let word = text
        .split_whitespace()
        .next()
        .map_or(0, |w| w.chars().count());
    let word_width = first.width * word as f32 / chars as f32;
    right - end(upper) > word_width + lower.size.max(upper.size) * 1.25
}

/// A symbol a line may start with as a list marker that stays in the text
/// (`✅ True HEPA filter`): check marks, crosses, arrows, stars, shapes and
/// pictographs, followed by a space.
fn symbol_marker(text: &str) -> Option<char> {
    let mut chars = text.trim_start().chars();
    let symbol = chars.next()?;
    let next = chars.next()?;
    let symbol_range = matches!(symbol as u32,
        0x2190..=0x21FF | 0x2300..=0x23FF | 0x25A0..=0x25FF | 0x2600..=0x27BF
        | 0x27F0..=0x27FF | 0x2900..=0x297F | 0x2B00..=0x2BFF | 0x1F300..=0x1FAFF);
    // A variation selector may follow an emoji before the space.
    let spaced = next.is_whitespace()
        || (matches!(next, '\u{FE0E}' | '\u{FE0F}')
            && chars.next().is_some_and(char::is_whitespace));
    (symbol_range && spaced).then_some(symbol)
}

/// Gap between baselines, in the larger line's size, from which a line is
/// set off from the text above: prose leading is 1.2–1.5 em, a paragraph
/// or `<pre>` gap about two line heights.
const SET_OFF: f32 = 1.7;
/// Columns beyond which a run's position is layout, not indentation.
const MAX_CODE_COLUMN: usize = 160;

/// Whether a line is code: nine in ten of its characters set in a
/// fixed-pitch face. Opening a block, a mono-set link or bare URL is a link
/// style (a sidebar of references), not code — a command naming a URL is;
/// inside a block, a URL in a string literal is code too.
fn is_code(line: &Line, inside: bool) -> bool {
    let (mut mono, mut all) = (0, 0);
    for item in &line.items {
        let text = item.text.trim();
        let count = text.chars().count();
        all += count;
        let link = !inside && (item.is_underline || is_bare_url(text));
        if item.fixed_pitch == Some(true) && !link {
            mono += count;
        }
    }
    all > 0 && mono * 10 >= all * 9
}

/// A code block's lines (`None` for a blank line between them).
#[derive(Default)]
struct Code<'a> {
    lines: Vec<Option<&'a Line>>,
    pitch: Option<f32>,
}

impl<'a> Code<'a> {
    fn push(&mut self, line: &'a Line) {
        if let Some(Some(last)) = self.lines.last() {
            let gap = last.y - line.y;
            // A blank line leaves half again the block's pitch once one
            // is known, two em before: the first gap can be the blank line,
            // and a loosely set block's pitch is 1.75 em.
            let blank = match self.pitch {
                Some(pitch) => gap > pitch * 1.5,
                None => gap > line.size * 2.,
            };
            if blank {
                self.lines.push(None);
            } else if gap > 0. {
                self.pitch = Some(self.pitch.map_or(gap, |pitch| pitch.min(gap)));
            }
        }
        self.lines.push(Some(line));
    }

    /// The fenced block, each run placed at its column: the text lost its
    /// indentation and alignment to trimming and run splitting, and a
    /// fixed-pitch face gives both back as glyph advances from the block's
    /// left edge.
    fn fence(&mut self) -> Option<String> {
        let lines = std::mem::take(&mut self.lines);
        self.pitch = None;
        let written = lines.iter().flatten();
        let left = written
            .clone()
            .flat_map(|line| line.items.first())
            .map(|i| i.x)
            .fold(f32::INFINITY, f32::min);
        let (width, glyphs) = written
            .flat_map(|line| &line.items)
            .filter(|item| item.fixed_pitch == Some(true))
            .fold((0., 0), |(width, glyphs), item| {
                (width + item.width, glyphs + item.text.chars().count())
            });
        let advance = if glyphs > 0 {
            width / glyphs as f32
        } else {
            0.
        };
        let mut text = Vec::new();
        for line in &lines {
            let Some(line) = line else {
                text.push(String::new());
                continue;
            };
            let mut row = String::new();
            let mut previous: Option<&TextItem> = None;
            for item in &line.items {
                let columns = row.chars().count();
                let column = if advance > 0. {
                    ((item.x - left) / advance).round().max(0.) as usize
                } else {
                    0
                };
                if column > columns && column <= MAX_CODE_COLUMN {
                    row.extend(std::iter::repeat_n(' ', column - columns));
                } else if previous.is_some_and(|p| {
                    !p.text.ends_with(char::is_whitespace)
                        && !item.text.starts_with(char::is_whitespace)
                        && item.x - (p.x + p.width) > item.font_size.min(p.font_size) * 0.12
                }) {
                    row.push(' ');
                }
                row.push_str(&item.text);
                previous = Some(item);
            }
            text.push(row.trim_end().to_owned());
        }
        while text.last().is_some_and(String::is_empty) {
            text.pop();
        }
        let body = text.join("\n");
        if body.trim().is_empty() {
            return None;
        }
        // A fence longer than any backtick run inside the code.
        let mut longest = 0;
        let mut run = 0;
        for c in body.chars() {
            run = if c == '`' { run + 1 } else { 0 };
            longest = longest.max(run);
        }
        let fence = "`".repeat(longest.max(2) + 1);
        Some(format!("{fence}\n{body}\n{fence}"))
    }
}

/// A list item's marker: a bullet character, a painted bullet or a number.
struct Marker {
    /// Where the marker starts, for nesting.
    x: f32,
    /// `None` for a bullet, else the number with its delimiter (`3.`).
    number: Option<String>,
}

/// The painted bullets of a page's lines, by the position of the first item
/// of each bulleted line: where its bullet starts. The rule is the one the
/// page reader is given marks for (`pdf_inspector::painted_bullets::targets`):
/// a compact filled shape or ring at most half the text's size, ending no
/// more than two em before the line's first run with no other text close
/// before it, centred on the lower part of the text (a browser centres its
/// disc near the x-height), and dark or in the text's colour.
struct Bullets(Vec<((u32, u32), f32)>);

impl Bullets {
    fn new(items: &[TextItem], marks: &[Mark]) -> Self {
        Self(
            pdf_inspector::painted_bullets::targets(items, marks)
                .into_iter()
                .map(|(mark, item)| {
                    let item = &items[item];
                    ((item.x.to_bits(), item.y.to_bits()), marks[mark].x0)
                })
                .collect(),
        )
    }

    fn of(&self, line: &Line) -> Option<f32> {
        let first = line.items.first()?;
        let at = (first.x.to_bits(), first.y.to_bits());
        self.0.iter().find(|(item, _)| *item == at).map(|(_, x)| *x)
    }
}

/// Whether `line` stands beside `above` rather than below it: within three
/// quarters of an em, with no horizontal overlap. Two columns whose
/// baselines interleave (a bulleted sidebar beside an article) read as one
/// otherwise, each line of one joining the other's paragraph. A marker on
/// a line of its own, a short run such as a footnote mark set off its
/// line's baseline, or code (a highlighted listing's runs need not share a
/// baseline) is no column.
fn beside(above: &Line, line: &Line) -> bool {
    let short = |line: &Line| {
        line.items
            .iter()
            .flat_map(|i| i.text.chars())
            .filter(|c| !c.is_whitespace())
            .count()
            < 4
    };
    if short(above) || short(line) || is_code(above, true) || is_code(line, true) {
        return false;
    }
    let span = |line: &Line| {
        let start = line.items.first().map_or(0., |i| i.x);
        let end = line
            .items
            .iter()
            .map(|i| i.x + i.width)
            .fold(f32::NEG_INFINITY, f32::max);
        (start, end)
    };
    let marker = |line: &Line| {
        let text: String = line.items.iter().map(|i| i.text.as_str()).collect();
        list_prefix(&text).is_some_and(str::is_empty)
            || number_prefix(&text).is_some_and(|(_, rest)| rest.is_empty())
    };
    let ((a0, a1), (b0, b1)) = (span(above), span(line));
    above.y - line.y < above.size.min(line.size) * 0.75
        && (b0 >= a1 || a0 >= b1)
        && !marker(above)
        && !marker(line)
}

/// `1.`/`1)` (up to three digits) and the text after it, when a run starts
/// with a number followed by a space or nothing more.
fn number_prefix(text: &str) -> Option<(&str, &str)> {
    let text = text.trim_start();
    let digits = text.bytes().take_while(u8::is_ascii_digit).count();
    if !(1..=3).contains(&digits) {
        return None;
    }
    let rest = &text[digits..];
    let delimiter = rest.chars().next().filter(|c| matches!(c, '.' | ')'))?;
    let after = &rest[1..];
    (after.is_empty() || after.starts_with(char::is_whitespace))
        .then(|| (&text[..digits + delimiter.len_utf8()], after.trim_start()))
}

fn flow(lines: &[Line], headings: &[f32], faces: &Faces, bullets: &Bullets) -> Option<String> {
    // Blocks with whether each is a list item: consecutive items are one
    // tight list.
    let mut blocks: Vec<(String, bool)> = Vec::new();
    let mut paragraph = Vec::new();
    let mut previous: Option<&Line> = None;
    // The index of `previous` in `lines` (see `ended_early`).
    let mut previous_index = 0;
    let same_size = same_size_headings(lines, headings, faces);
    let rights = column_rights(lines);
    // Lines starting with a symbol that two lines or more of the flow start
    // with are items of a list that keeps the symbol (`- ✅ HEPA filter`).
    let symbols: Vec<Option<char>> = lines
        .iter()
        .map(|line| {
            line.items
                .first()
                .and_then(|item| symbol_marker(&item.text))
        })
        .collect();
    let symbol_list = |index: usize| {
        symbols[index].filter(|symbol| symbols.iter().filter(|s| **s == Some(*symbol)).count() >= 2)
    };
    let mut previous_level = 0;
    let mut item: Option<String> = None;
    let mut list_left = 0f32;
    // A marker set on a line of its own, waiting for its item's text.
    let mut pending: Option<Marker> = None;
    let mut code = Code::default();
    let flush = |blocks: &mut Vec<(String, bool)>,
                 paragraph: &mut Vec<Run>,
                 level: usize,
                 item: Option<&str>| {
        if paragraph.is_empty() {
            return;
        }
        // A heading keeps its links' text without their targets: a link
        // there is navigation (a site's name, a post's permalink), and the
        // heading names the document or section.
        if level > 0 {
            let mut unlinked: Vec<Run> = Vec::with_capacity(paragraph.len());
            for mut run in paragraph.drain(..) {
                run.link = None;
                match unlinked.last_mut() {
                    Some(last) if last.style == run.style => last.text.push_str(&run.text),
                    _ => unlinked.push(run),
                }
            }
            *paragraph = unlinked;
        }
        let content = markdown(paragraph);
        let prefix = if level > 0 {
            format!("{} ", "#".repeat(level))
        } else {
            item.unwrap_or_default().to_owned()
        };
        // A paragraph's own text starting like a heading, list item or
        // quotation (`# 1`, `+ note`, `2026. The year`) stays text.
        let content = if level == 0 && item.is_none() {
            super::super::literal_heading_marks(&content)
        } else {
            content
        };
        blocks.push((format!("{prefix}{content}"), item.is_some()));
        paragraph.clear();
    };
    for (index, line) in lines.iter().enumerate() {
        let mut current = runs(line);
        if current.is_empty() {
            continue;
        }
        if index > 0 && beside(&lines[index - 1], line) {
            return None;
        }
        // Code opens where a paragraph would: at the start, after a
        // heading, or set off from the text above. A single mono-set line
        // at ordinary leading continues its paragraph — an inline literal
        // wrapped onto a line of its own — but a run of two or more is a
        // block set without margins (AppKit prints `<pre>` that way).
        let code_line = if code.lines.is_empty() {
            is_code(line, false)
                && (previous.is_none_or(|prev| {
                    previous_level > 0 || prev.y - line.y > prev.size.max(line.size) * SET_OFF
                }) || lines.get(index + 1).is_some_and(|next| {
                    let gap = line.y - next.y;
                    is_code(next, true) && gap > 0. && gap <= line.size.max(next.size) * 2.5
                }))
        } else {
            is_code(line, true)
        };
        if code_line {
            flush(&mut blocks, &mut paragraph, previous_level, item.as_deref());
            item = None;
            if let Some(marker) = pending.take() {
                blocks.push((marker.number.unwrap_or_else(|| "-".into()), false));
            }
            code.push(line);
            previous = Some(line);
            previous_index = index;
            previous_level = 0;
            continue;
        }
        if !code.lines.is_empty() {
            blocks.extend(code.fence().map(|fence| (fence, false)));
            // The prose after a block starts its own paragraph.
            previous = None;
        }
        let first_x = line.items[0].x;
        // A numbered or painted marker never turns a heading into an item
        // ("## 1. Start Here" keeps its level).
        let heading = match heading_level(line, headings) {
            0 if same_size[index] => (headings.len() + 1).min(6),
            level => level,
        };
        let marker = if heading == 0
            && let Some(marker) = pending.take()
        {
            Some(marker)
        } else if let Some(rest) = list_prefix(&current[0].text) {
            current[0].text = rest.to_owned();
            if current[0].text.is_empty() {
                current.remove(0);
            }
            Some(Marker {
                x: first_x,
                number: None,
            })
        } else if heading > 0 {
            None
        } else if let Some(x) = bullets.of(line) {
            Some(Marker { x, number: None })
        } else if symbol_list(index).is_some() {
            Some(Marker {
                x: first_x,
                number: None,
            })
        } else if let Some((number, rest)) = number_prefix(&current[0].text)
            // A number opens an item only where a new line of the text
            // could: at the start, in a list, after a gap or a heading, or
            // shifted off the text above. A wrapped line of prose that
            // happens to start with "2. " stays prose.
            && previous.is_none_or(|prev| {
                item.is_some()
                    || previous_level > 0
                    || prev.y - line.y > prev.size.max(line.size) * 1.8
                    || (first_x - prev.items[0].x).abs() > line.size * 0.75
            })
        {
            let number = number.to_owned();
            current[0].text = rest.to_owned();
            if current[0].text.is_empty() {
                current.remove(0);
            }
            Some(Marker {
                x: first_x,
                number: Some(number),
            })
        } else {
            None
        };
        if let Some(marker) = pending.take() {
            // The line after a lone marker was a heading.
            blocks.push((marker.number.unwrap_or_else(|| "-".into()), false));
        }
        if current.is_empty() {
            // Only the marker: its item's text is on the next line.
            flush(&mut blocks, &mut paragraph, previous_level, item.as_deref());
            item = None;
            pending = marker;
            previous = Some(line);
            previous_index = index;
            previous_level = 0;
            continue;
        }
        let is_list = marker.is_some();
        // Side-by-side prose and unruled tables need a reading-order model.
        // Keep the previous reader instead of concatenating their columns;
        // a list item beside other text is no exception (a bulleted sidebar
        // beside an article). Only a marker set as an item of its own may
        // stand apart from its text.
        let marker_item = is_list
            && line.items.first().is_some_and(|first| {
                list_prefix(&first.text).is_some_and(str::is_empty)
                    || number_prefix(&first.text).is_some_and(|(_, rest)| rest.is_empty())
            });
        if line
            .items
            .windows(2)
            .skip(usize::from(marker_item))
            .any(|p| p[1].x - (p[0].x + p[0].width) > line.size * 3.)
        {
            return None;
        }
        let level = if is_list { 0 } else { heading };
        let in_list = item.is_some();
        let new_block = previous.is_some_and(|prev| {
            let gap = prev.y - line.y;
            level != previous_level
                || is_list
                || gap > prev.size.max(line.size) * NEW_BLOCK
                || ended_early(lines, previous_index, index, rights[previous_index])
                || (in_list && line.items[0].x + 2. < prev.items[0].x)
                || (level == 0
                    && !in_list
                    && (line.items[0].x - prev.items[0].x).abs() > line.size * 2.5)
        });
        if new_block {
            flush(&mut blocks, &mut paragraph, previous_level, item.as_deref());
            item = None;
        }
        if let Some(marker) = marker {
            // Nesting is measured from the list's leftmost marker; a list
            // starts again after any other block.
            let continuing = blocks.last().is_some_and(|(_, list)| *list);
            list_left = if continuing {
                list_left.min(marker.x)
            } else {
                marker.x
            };
            // A browser indents each nested list by 40px (2.5 em at 12pt);
            // right-aligned numbers stay within half an em of their column.
            let depth = (((marker.x - list_left) + line.size * 0.5) / (line.size * 2.))
                .floor()
                .clamp(0., 5.) as usize;
            let indent = "    ".repeat(depth);
            item = Some(match marker.number {
                Some(number) => format!("{indent}{number} "),
                None => format!("{indent}- "),
            });
        }
        append_runs(&mut paragraph, current);
        previous = Some(line);
        previous_index = index;
        previous_level = level;
    }
    flush(&mut blocks, &mut paragraph, previous_level, item.as_deref());
    blocks.extend(code.fence().map(|fence| (fence, false)));
    if let Some(marker) = pending {
        blocks.push((marker.number.unwrap_or_else(|| "-".into()), false));
    }
    let mut output = String::new();
    for (index, (block, list)) in blocks.iter().enumerate() {
        if index > 0 {
            let tight = *list && blocks[index - 1].1;
            output.push_str(if tight { "\n" } else { "\n\n" });
        }
        output.push_str(block);
    }
    Some(output)
}

struct Table {
    top: f32,
    bottom: f32,
    markdown: String,
    /// Column borders, left to right.
    xs: Vec<f32>,
    /// The Markdown of the cells of its first (header) row.
    header: Vec<String>,
    /// Whether that row is set apart as a header: bold throughout, above a
    /// row that is not.
    header_styled: bool,
}

impl Table {
    /// Whether this table, at the top of its page, continues `ending`: the
    /// same column borders, and a first row that repeats the header there
    /// or is no header of its own.
    fn continues(&self, ending: &Ending) -> bool {
        let Ending::Ruled { xs, header } = ending else {
            return false;
        };
        xs.len() == self.xs.len()
            && xs
                .iter()
                .zip(&self.xs)
                .all(|(a, b)| (a - b).abs() <= SAME_BORDER)
            && (*header == self.header || !self.header_styled)
    }
}

fn table(grid: &Grid, items: &mut Vec<TextItem>) -> Option<Table> {
    let rows = grid.ys.len() - 1;
    let columns = grid.xs.len() - 1;
    let left = grid.xs[0];
    let right = grid.xs[columns];
    let bottom = grid.ys[0];
    let top = grid.ys[rows];
    let inside = |item: &TextItem| {
        item.y >= bottom
            && item.y + item.height <= top + 2.
            && item.x >= left - 1.
            && item.x + item.width <= right + 1.
    };
    let mut cells: Vec<Vec<TextItem>> = (0..rows * columns).map(|_| Vec::new()).collect();
    let mut selected = Vec::new();
    for (index, item) in items.iter().enumerate() {
        if !inside(item) {
            continue;
        }
        let x = item.x + item.width / 2.;
        let y = item.y + item.height / 2.;
        let column = grid.xs.windows(2).position(|w| x >= w[0] && x <= w[1])?;
        let row = grid.ys.windows(2).position(|w| y >= w[0] && y <= w[1])?;
        // A spanning run is evidence against a rectangular cell partition.
        if item.x < grid.xs[column] - 1.
            || item.x + item.width > grid.xs[column + 1] + 1.
            || item.y < grid.ys[row] - 1.
            || item.y + item.height > grid.ys[row + 1] + 2.
        {
            return None;
        }
        cells[(rows - row - 1) * columns + column].push(item.clone());
        selected.push(index);
    }
    if selected.len() < 3
        || (0..rows)
            .filter(|&row| {
                cells[row * columns..(row + 1) * columns]
                    .iter()
                    .any(|c| !c.is_empty())
            })
            .count()
            < 2
    {
        return None;
    }
    // Text straddling a border must not silently remain outside the table.
    if items
        .iter()
        .any(|i| i.y >= bottom && i.y <= top && i.x < right && i.x + i.width > left && !inside(i))
    {
        return None;
    }
    // Each row's cells, and whether its text is bold throughout.
    let mut rendered: Vec<(Vec<String>, bool)> = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut values = Vec::with_capacity(columns);
        let mut bold = true;
        for column in 0..columns {
            let cell: Vec<Vec<Run>> = lines(std::mem::take(&mut cells[row * columns + column]))
                .iter()
                .map(runs)
                .collect();
            bold &= cell.iter().flatten().all(|run| run.style.bold);
            values.push(
                cell.iter()
                    .map(|runs| markdown_cell(runs))
                    .collect::<Vec<_>>()
                    .join("<br>"),
            );
        }
        rendered.push((values, bold));
    }
    // An empty header row heads nothing: the first row with text is the
    // header. Two rows hold text, so one follows it.
    let first = rendered
        .iter()
        .position(|(values, _)| values.iter().any(|v| !v.is_empty()))?;
    let rendered = &rendered[first..];
    let mut markdown = String::new();
    for (index, (values, _)) in rendered.iter().enumerate() {
        markdown.push('|');
        for value in values {
            markdown.push_str(value);
            markdown.push('|');
        }
        markdown.push('\n');
        if index == 0 {
            markdown.push('|');
            markdown.push_str(&"---|".repeat(columns));
            markdown.push('\n');
        }
    }
    let header_styled = rendered[0].1 && rendered.get(1).is_some_and(|(_, bold)| !bold);
    let mut index = 0;
    let selected = selected.into_iter().collect::<HashSet<_>>();
    items.retain(|_| {
        let keep = !selected.contains(&index);
        index += 1;
        keep
    });
    Some(Table {
        top,
        bottom,
        markdown: markdown.trim_end().into(),
        xs: grid.xs.clone(),
        header: rendered[0].0.clone(),
        header_styled,
    })
}

fn markdown_cell(runs: &[Run]) -> String {
    markdown(runs).replace('|', "\\|")
}

/// What `segment` makes of the lines between ruled tables.
struct Segment {
    markdown: String,
    /// Whether its first block is a table continuing the previous page's.
    continues: bool,
    /// The bottom and shape of a borderless table that is its last block.
    ending: Option<(f32, unruled::Shape)>,
}

/// Lines between ruled tables: borderless tables found among them, and the
/// paragraph flow around those.
fn segment(
    lines: &[Line],
    headings: &[f32],
    faces: &Faces,
    bullets: &Bullets,
    context: &Tables,
    continued: Option<&unruled::Shape>,
) -> Option<Segment> {
    let mut found = unruled::find(lines, *context.pitch, headings, continued);
    if !found.is_empty() && (context.tagged)() {
        found.clear();
    }
    let mut blocks = Vec::new();
    let mut start = 0;
    let mut ending = None;
    let mut continues = false;
    for table in found {
        if table.lines.start > start {
            blocks.push(flow(
                &lines[start..table.lines.start],
                headings,
                faces,
                bullets,
            )?);
        }
        continues |= table.continued && blocks.is_empty();
        blocks.push(table.markdown);
        start = table.lines.end;
        ending = Some((table.bottom, table.shape));
    }
    if start < lines.len() {
        blocks.push(flow(&lines[start..], headings, faces, bullets)?);
        ending = None;
    }
    Some(Segment {
        markdown: blocks.join("\n\n"),
        continues,
        ending,
    })
}

/// A page's Markdown, whether it starts with a table continuing the
/// previous page's, and the bottom of a table ending it, with what a
/// continuation of that table keeps to.
struct Rendered {
    markdown: String,
    continues: bool,
    ending: Option<(f32, Ending)>,
}

fn render(
    mut items: Vec<TextItem>,
    faces: &Faces,
    frame: Frame,
    grids: Vec<Grid>,
    marks: &[Mark],
    baseline: &str,
    context: &mut Tables,
) -> Option<Rendered> {
    let headings = &faces.headings[..];
    // Link annotations carry a target, not page text: they mark the runs
    // they cover (see `links::apply`). Form-field values are page text whose
    // semantics this reconstruction does not know.
    if items.len() > MAX_PAGE_ITEMS
        || items
            .iter()
            .any(|i| matches!(i.item_type, ItemType::FormField))
    {
        return None;
    }
    let links = links::LinkBox::collect(&items);
    items.retain(|i| matches!(i.item_type, ItemType::Text) && !i.text.trim().is_empty());
    if items.is_empty()
        || items.iter().any(|i| {
            !valid(i)
                || i.x < -1.
                || i.y < -1.
                || i.x + i.width > frame.width + 2.
                || i.y + i.height > frame.height + 2.
        })
    {
        return None;
    }
    let all_text = items.iter().map(|i| i.text.as_str()).collect::<String>();
    // Lines are assembled left to right here. Right-to-left text needs the
    // page reader's bidirectional ordering, which puts a line's runs in
    // reading order; a reading order is worth more than list or code
    // structure.
    if all_text.chars().any(is_right_to_left) {
        return None;
    }
    if character_counts(&all_text) != character_counts(baseline) {
        return None;
    }
    let bullets = Bullets::new(&items, marks);
    // After the bullets, which are read from text items only.
    links::apply(&mut items, &links);
    let mut tables = Vec::new();
    for grid in grids {
        if let Some(table) = table(&grid, &mut items) {
            tables.push(table);
        }
    }
    tables.sort_by(|a, b| b.top.total_cmp(&a.top));
    if tables.windows(2).any(|t| t[0].bottom < t[1].top) {
        return None;
    }
    let all_lines = lines(items);
    if let Some(own) = unruled::prose_pitch(&all_lines) {
        *context.pitch = Some(own);
    }
    // A borderless table ending the previous page continues only in the
    // page's first lines.
    let unruled_continued = |start: usize, blocks: &[String]| match context.continued {
        Some(Ending::Unruled(shape)) if start == 0 && blocks.is_empty() => Some(shape),
        _ => None,
    };
    let mut start = 0;
    let mut blocks: Vec<String> = Vec::new();
    let mut ending = None;
    let mut continues = false;
    for table in tables {
        let end = start + all_lines[start..].partition_point(|line| line.y > table.top);
        if end > start {
            let continued = unruled_continued(start, &blocks);
            let part = segment(
                &all_lines[start..end],
                headings,
                faces,
                &bullets,
                context,
                continued,
            )?;
            continues |= part.continues;
            blocks.push(part.markdown);
        }
        // Any non-table text beside it is ambiguous multi-column layout.
        if all_lines
            .get(end)
            .is_some_and(|line| line.y >= table.bottom)
        {
            return None;
        }
        // A ruled table opening the page may continue the previous page's.
        continues |= blocks.iter().all(|b| b.trim().is_empty())
            && context.continued.is_some_and(|e| table.continues(e));
        blocks.push(table.markdown);
        start = end;
        ending = Some((
            table.bottom,
            Ending::Ruled {
                xs: table.xs,
                header: table.header,
            },
        ));
    }
    if start < all_lines.len() {
        let continued = unruled_continued(start, &blocks);
        let part = segment(
            &all_lines[start..],
            headings,
            faces,
            &bullets,
            context,
            continued,
        )?;
        continues |= part.continues;
        blocks.push(part.markdown);
        ending = part
            .ending
            .map(|(bottom, shape)| (bottom, Ending::Unruled(shape)));
    }
    let markdown = blocks.join("\n\n");
    (!markdown.is_empty()).then_some(Rendered {
        markdown,
        continues,
        ending,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::{
        Dictionary, Object, Stream,
        content::{Content, Operation},
        dictionary,
    };

    fn text(font: &str, size: i64, x: i64, y: i64, value: &str) -> Vec<Operation> {
        vec![
            Operation::new("BT", vec![]),
            Operation::new(
                "Tf",
                vec![Object::Name(font.as_bytes().into()), size.into()],
            ),
            Operation::new("Td", vec![x.into(), y.into()]),
            Operation::new("Tj", vec![Object::string_literal(value)]),
            Operation::new("ET", vec![]),
        ]
    }

    fn pdf(pages: Vec<Vec<Operation>>, rotate: Option<i64>) -> Vec<u8> {
        let mut pdf = lopdf::Document::with_version("1.7");
        let pages_id = pdf.new_object_id();
        let regular =
            pdf.add_object(dictionary! {"Type"=>"Font","Subtype"=>"Type1","BaseFont"=>"Helvetica"});
        let bold = pdf.add_object(
            dictionary! {"Type"=>"Font","Subtype"=>"Type1","BaseFont"=>"Helvetica-Bold"},
        );
        let mono =
            pdf.add_object(dictionary! {"Type"=>"Font","Subtype"=>"Type1","BaseFont"=>"Courier"});
        let resources =
            pdf.add_object(dictionary! {"Font"=>dictionary!{"F1"=>regular,"F2"=>bold,"F3"=>mono}});
        let mut kids = Vec::new();
        for operations in pages {
            let content = Content { operations }.encode().unwrap();
            let stream = pdf.add_object(Stream::new(Dictionary::new(), content));
            let mut page = dictionary! {"Type"=>"Page","Parent"=>pages_id,"Contents"=>stream,"Resources"=>resources};
            if let Some(rotate) = rotate {
                page.set("Rotate", rotate);
            }
            let id = pdf.add_object(page);
            kids.push(Object::Reference(id));
        }
        pdf.objects.insert(pages_id,dictionary!{"Type"=>"Pages","Count"=>kids.len() as i64,"Kids"=>kids,"MediaBox"=>vec![0.into(),0.into(),600.into(),800.into()]}.into());
        let catalog = pdf.add_object(dictionary! {"Type"=>"Catalog","Pages"=>pages_id});
        pdf.trailer.set("Root", catalog);
        let mut bytes = Vec::new();
        pdf.save_to(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn a_printed_listing_is_fenced_with_its_indentation_and_blank_lines() {
        // A browser prints `<pre>` two line heights below its introduction,
        // in a smaller fixed-pitch face: 10pt Courier advances 6pt a glyph,
        // so a line starting 24pt further right is indented four columns.
        let mut page = text(
            "F1",
            12,
            40,
            700,
            "Here is a small program that prints a greeting:",
        );
        page.extend(text("F3", 10, 40, 675, "fn main() {"));
        page.extend(text("F3", 10, 64, 663, "println!(\"<hi>\");"));
        page.extend(text("F3", 10, 40, 651, "}"));
        page.extend(text("F3", 10, 40, 627, "// see https://example.com/a"));
        page.extend(text(
            "F1",
            12,
            40,
            600,
            "The program takes no arguments and prints one line",
        ));
        page.extend(text(
            "F1",
            12,
            40,
            586,
            "to its standard output before it exits normally.",
        ));
        let output = super::super::extract(&pdf(vec![page], None)).unwrap();
        assert!(
            output.markdown.contains(
                "greeting:\n\n```\nfn main() {\n    println!(\"<hi>\");\n}\n\n// see https://example.com/a\n```\n\nThe program takes"
            ),
            "{}",
            output.markdown
        );
    }

    /// A filled `size`-point square with its lower left corner at (x, y):
    /// the bullet a browser paints for a `square` list item.
    fn bullet(x: f32, y: f32, size: f32) -> Vec<Operation> {
        vec![
            Operation::new("re", vec![x.into(), y.into(), size.into(), size.into()]),
            Operation::new("f", vec![]),
        ]
    }

    #[test]
    fn painted_bullets_and_numbers_make_lists() {
        // A browser paints list bullets as shapes, not characters: a
        // 4.5pt mark centred near the x-height, 0.75 em before the text. A
        // nested list is indented 30pt; a number may sit on a line of its
        // own above its item's text.
        let mut page = text("F1", 12, 40, 740, "Use contain when:");
        page.extend(bullet(56.5, 712.0, 4.5));
        page.extend(text(
            "F1",
            12,
            70,
            710,
            "You want to preserve the full image",
        ));
        page.extend(bullet(56.5, 695.5, 4.5));
        page.extend(text(
            "F1",
            12,
            70,
            693,
            "The image might have a different ratio",
        ));
        page.extend(bullet(86.5, 679.0, 4.5));
        page.extend(text("F1", 12, 100, 677, "Even inside a narrow card"));
        page.extend(text("F1", 12, 40, 645, "The steps are:"));
        page.extend(text("F1", 12, 52, 620, "1. Measure the box"));
        page.extend(text("F1", 12, 52, 604, "2."));
        page.extend(text(
            "F1",
            12,
            64,
            588,
            "Pick a mode that keeps what matters",
        ));
        page.extend(text(
            "F1",
            12,
            40,
            556,
            "Both modes keep the original file unchanged.",
        ));
        let output = super::super::extract(&pdf(vec![page], None)).unwrap();
        assert!(
            output.markdown.contains(
                "Use contain when:\n\n- You want to preserve the full image\n- The image might have a different ratio\n    - Even inside a narrow card\n\nThe steps are:\n\n1. Measure the box\n2. Pick a mode that keeps what matters\n\nBoth modes keep"
            ),
            "{}",
            output.markdown
        );
    }

    /// The layout pass's Markdown for a one-page PDF, with the page's own
    /// rule grids and marks; `None` when it leaves the page to the reader.
    fn layout_page(ops: Vec<Operation>) -> Option<String> {
        let bytes = pdf(vec![ops], None);
        let loaded = pdf_inspector::LoadedPdf::load_mem(&bytes).unwrap();
        let doc = lopdf::Document::load_mem(&bytes).unwrap();
        let id = doc.get_pages()[&1];
        let frame = super::super::geometry::frame(&doc, id).unwrap();
        let (_, content) = super::super::inspect_page(&doc, id);
        let (grids, marks) = super::super::geometry::page_shapes(
            &content.unwrap().operations,
            frame,
            &super::super::geometry::rule_resources(&doc, id),
        );
        let baseline = loaded.pages_markdown(None).unwrap();
        let mut layout =
            Layout::read(Some(&loaded), &HashSet::from([1]), &BTreeMap::new()).unwrap();
        layout
            .page(
                1,
                frame,
                grids,
                &marks,
                &baseline.pages[0].markdown,
                &|| false,
            )
            .map(|(markdown, _)| markdown)
    }

    const SIDEBAR: [&str; 4] = [
        "River lantern",
        "Lantern orchard",
        "Orchard copper",
        "Copper meadow",
    ];

    /// A bulleted list at x 64, a line every 15pt from 720 down.
    fn sidebar() -> Vec<Operation> {
        let mut page = Vec::new();
        for (index, item) in SIDEBAR.iter().enumerate() {
            let y = 720 - 15 * index as i64;
            page.extend(bullet(52., y as f32 + 2., 4.5));
            page.extend(text("F1", 12, 64, y, item));
        }
        page
    }

    #[test]
    fn list_items_beside_other_text_leave_the_page_to_the_reader() {
        let words = "Main column words run on across the page.";
        assert!(layout_page(sidebar()).is_some_and(|m| m.starts_with("- River lantern\n- ")));
        // The article's lines interleave with the sidebar's, three points
        // lower (two would put them on its lines), or share their baselines.
        for offset in [3, 0] {
            let mut page = sidebar();
            for index in 0..12 {
                page.extend(text("F1", 12, 220, 720 - offset - 15 * index, words));
            }
            assert_eq!(layout_page(page), None, "offset {offset}");
        }
    }

    #[test]
    fn the_page_reader_reads_the_painted_bullets_of_a_page_left_to_it() {
        // Two runs far apart on one line keep the page with the page
        // reader, whose text is in the page's own coordinates: the page box
        // starts at (100, 100).
        let mut ops = vec![
            Operation::new("q", vec![]),
            Operation::new("cm", [1, 0, 0, 1, 100, 100].map(Object::from).to_vec()),
        ];
        ops.extend(sidebar());
        ops.extend(text("F1", 12, 40, 600, "Total"));
        ops.extend(text("F1", 12, 400, 600, "Four trees"));
        ops.push(Operation::new("Q", vec![]));
        let mut doc = lopdf::Document::load_mem(&pdf(vec![ops], None)).unwrap();
        let pages = doc
            .catalog()
            .unwrap()
            .get(b"Pages")
            .unwrap()
            .as_reference()
            .unwrap();
        doc.get_dictionary_mut(pages)
            .unwrap()
            .set("MediaBox", [100, 100, 700, 900].map(Object::from).to_vec());
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        let markdown = super::super::extract(&bytes).unwrap().markdown;
        let items = SIDEBAR.map(|item| format!("- {item}")).join("\n");
        assert!(markdown.contains(&items), "{markdown}");
    }

    /// A stroked circle of radius `r` around (`x`, `y`), drawn with curves.
    fn ring(x: f32, y: f32, r: f32) -> Vec<Operation> {
        let k = 0.5523 * r;
        let curve = |p: [f32; 6]| Operation::new("c", p.iter().map(|&v| v.into()).collect());
        vec![
            Operation::new("m", vec![(x + r).into(), y.into()]),
            curve([x + r, y + k, x + k, y + r, x, y + r]),
            curve([x - k, y + r, x - r, y + k, x - r, y]),
            curve([x - r, y - k, x - k, y - r, x, y - r]),
            curve([x + k, y - r, x + r, y - k, x + r, y]),
            Operation::new("h", vec![]),
            Operation::new("S", vec![]),
        ]
    }

    #[test]
    fn only_a_mark_in_the_list_colour_shaped_as_a_bullet_makes_an_item() {
        let lines = ["Alder and ash", "Birch and beech", "Cedar and cypress"];
        let page = |mark: &dyn Fn(f32) -> Vec<Operation>| {
            let mut page = text("F1", 12, 40, 750, "Three kinds of tree grow here.");
            for (index, line) in lines.iter().enumerate() {
                let y = 720 - 15 * index as i64;
                page.extend(mark(y as f32));
                page.extend(text("F1", 12, 64, y, line));
            }
            page
        };
        let items = lines.map(|line| format!("- {line}")).join("\n");
        // A ring, as a nested list's bullet, makes items.
        let rings = layout_page(page(&|y| ring(54.25, y + 4.25, 2.25))).unwrap();
        assert!(rings.contains(&items), "{rings}");
        // A coloured swatch, a stroked box (a checkbox), or a square inside
        // the line does not.
        let red = |y: f32| {
            let mut ops = vec![Operation::new(
                "rg",
                vec![0.87.into(), 0.2.into(), 0.2.into()],
            )];
            ops.extend(bullet(52., y + 2., 4.5));
            ops.push(Operation::new("rg", vec![0.into(), 0.into(), 0.into()]));
            ops
        };
        let boxed = |y: f32| {
            vec![
                Operation::new(
                    "re",
                    vec![52.into(), (y + 2.).into(), 4.5.into(), 4.5.into()],
                ),
                Operation::new("S", vec![]),
            ]
        };
        let inline = |y: f32| bullet(130., y + 2., 4.5);
        for mark in [&red as &dyn Fn(f32) -> Vec<Operation>, &boxed, &inline] {
            let markdown = layout_page(page(mark)).unwrap();
            assert!(!markdown.contains("- "), "{markdown}");
        }
    }

    #[test]
    fn a_marker_set_apart_from_its_text_is_no_column() {
        // A numbered list with a wide hanging indent: the number is an item
        // of its own, four em before its text.
        let mut page = text("F1", 12, 40, 750, "Follow the steps.");
        for (index, step) in ["Open the valve", "Close the valve"].iter().enumerate() {
            let y = 720 - 15 * index as i64;
            page.extend(text("F1", 12, 40, y, &format!("{}.", index + 1)));
            page.extend(text("F1", 12, 100, y, step));
        }
        let markdown = layout_page(page).unwrap();
        assert!(
            markdown.contains("1. Open the valve\n2. Close the valve"),
            "{markdown}"
        );
    }

    #[test]
    fn a_footnote_mark_or_a_listing_off_the_baseline_is_no_column() {
        // A raised mark the extractor keeps as a line of its own, and a
        // highlighted listing whose runs stand on two baselines.
        let mut page = text("F1", 12, 40, 720, "A sentence that ends with a mark.");
        page.extend(text("F1", 8, 230, 725, "[a]"));
        page.extend(text("F1", 12, 40, 690, "The listing reads:"));
        page.extend(text("F3", 10, 40, 660, "#include"));
        page.extend(text("F3", 10, 100, 663, "<stdio.h>"));
        page.extend(text("F3", 10, 40, 645, "int main(void);"));
        assert!(layout_page(page).is_some());
    }

    #[test]
    fn a_raised_marker_stays_on_its_line_as_a_superscript() {
        // A browser `<sup>` (0.83 em, raised 0.4 em) holding a letter is a
        // script run of its own; a digit would fuse into its word.
        let mut page = text("F1", 12, 40, 740, "The claim holds");
        page.extend(text("F1", 10, 125, 745, "a"));
        page.extend(text("F1", 12, 131, 740, " in every case we measured."));
        page.extend(text(
            "F1",
            12,
            40,
            726,
            "A second sentence closes the paragraph.",
        ));
        // The painted bullet below needs the native layout, so the page
        // cannot pass through the page reader's own script rendering.
        page.extend(bullet(56.5, 697.0, 4.5));
        page.extend(text(
            "F1",
            12,
            70,
            695,
            "One listed point follows the paragraph",
        ));
        let output = super::super::extract(&pdf(vec![page], None)).unwrap();
        assert!(
            output.markdown.contains(
                "The claim holds<sup>a</sup> in every case we measured. A second sentence closes the paragraph.\n\n- One listed point"
            ),
            "{}",
            output.markdown
        );
    }

    #[test]
    fn a_line_starting_with_a_number_inside_prose_is_not_an_item() {
        let mut page = text(
            "F1",
            12,
            40,
            740,
            "The report counts every visitor, and in the year",
        );
        page.extend(text(
            "F1",
            12,
            40,
            726,
            "2. edition it counted twice as many as the first.",
        ));
        page.extend(text(
            "F1",
            12,
            40,
            712,
            "The third edition repeats the method unchanged.",
        ));
        let output = super::super::extract(&pdf(vec![page], None)).unwrap();
        assert!(
            output
                .markdown
                .contains("in the year 2. edition it counted twice"),
            "{}",
            output.markdown
        );
    }

    #[test]
    fn consecutive_listing_lines_without_margins_are_fenced() {
        // AppKit prints `<pre>` without margins: the listing sits at the
        // prose's own leading, so only its run of fixed-pitch lines shows it.
        let mut page = text("F1", 12, 40, 700, "Call the setup function like this:");
        page.extend(text("F3", 10, 40, 686, "setup({"));
        page.extend(text("F3", 10, 52, 674, "logging = true,"));
        page.extend(text("F3", 10, 40, 662, "})"));
        page.extend(text(
            "F1",
            12,
            40,
            648,
            "The call takes effect at once and returns nothing.",
        ));
        let output = super::super::extract(&pdf(vec![page], None)).unwrap();
        assert!(
            output
                .markdown
                .contains("like this:\n\n```\nsetup({\n  logging = true,\n})\n```\n\nThe call"),
            "{}",
            output.markdown
        );
    }

    #[test]
    fn inline_fixed_pitch_words_are_code_and_a_browser_h3_is_a_heading() {
        // Chrome's `<h3>` is 1.17 em; a `<code>` word in prose is set in a
        // fixed-pitch face beside the proportional text.
        let mut page = text("F2", 14, 40, 740, "Choosing a mode");
        let mut line = text("F1", 12, 40, 710, "Set");
        line.extend(text("F3", 12, 63, 710, "mode"));
        line.extend(text(
            "F1",
            12,
            95,
            710,
            "to contain when the whole picture must stay visible.",
        ));
        page.extend(line);
        page.extend(text(
            "F1",
            12,
            40,
            696,
            "Cover crops the picture to the box it is given instead.",
        ));
        let output = super::super::extract(&pdf(vec![page], None)).unwrap();
        assert!(
            output.markdown.contains("# **Choosing a mode**"),
            "{}",
            output.markdown
        );
        assert!(
            output.markdown.contains("Set `mode` to contain when"),
            "{}",
            output.markdown
        );
    }

    #[test]
    fn real_pdf_uses_document_heading_sizes_and_geometric_paragraph_breaks() {
        let mut first = text("F2", 28, 40, 750, "A document title");
        first.extend(text("F2", 18, 40, 710, "Section on first page"));
        first.extend(text(
            "F1",
            12,
            40,
            675,
            "This first paragraph has a physical line wrap and",
        ));
        first.extend(text(
            "F1",
            12,
            40,
            661,
            "its continuation belongs to the same paragraph.",
        ));
        first.extend(text(
            "F1",
            12,
            40,
            625,
            "A separate paragraph starts after a larger gap.",
        ));
        first.extend(text(
            "F1",
            12,
            40,
            611,
            "Keep this second continuation with that paragraph.",
        ));
        let mut second = text("F2", 18, 40, 750, "Section on second page");
        second.extend(text(
            "F1",
            12,
            40,
            710,
            "Body text on another page must not promote its section",
        ));
        second.extend(text(
            "F1",
            12,
            40,
            696,
            "to the same heading level as the document title.",
        ));
        let output = super::super::extract(&pdf(vec![first, second], None)).unwrap();
        assert!(
            output.markdown.contains("# **A document title**"),
            "{}",
            output.markdown
        );
        assert!(output.markdown.contains("## **Section on first page**"));
        assert!(output.markdown.contains("## **Section on second page**"));
        assert!(output.markdown.contains("physical line wrap and its continuation belongs to the same paragraph.\n\nA separate paragraph"));
    }

    #[test]
    fn real_pdf_joins_continuous_bold_across_lines_without_styling_plain_neighbors() {
        let mut ops = text(
            "F1",
            12,
            40,
            750,
            "Regular paragraph with enough text to establish its body font.",
        );
        ops.extend(text("F2", 12, 40, 714, "Bold words continue onto"));
        ops.extend(text("F2", 12, 40, 700, "the next physical line."));
        ops.extend(text(
            "F1",
            12,
            40,
            660,
            "Plain text after the bold paragraph remains plain text.",
        ));
        let result = super::super::extract(&pdf(vec![ops], None)).unwrap();
        assert!(
            result
                .markdown
                .contains("**Bold words continue onto the next physical line.**"),
            "{}",
            result.markdown
        );
        assert!(!result.markdown.contains("**Plain text"));
    }

    /// The page reader reads bold from a font's name and flags, the layout
    /// reader also from its weight class, from the text runs both share.
    #[test]
    fn a_heavier_weight_alone_makes_layout_text_bold() {
        let mut pdf = lopdf::Document::with_version("1.7");
        let pages_id = pdf.new_object_id();
        let regular =
            pdf.add_object(dictionary! {"Type"=>"Font","Subtype"=>"Type1","BaseFont"=>"Helvetica"});
        let descriptor = pdf.add_object(dictionary! {"Type"=>"FontDescriptor","FontName"=>"Helvetica","Flags"=>32,"ItalicAngle"=>0,"FontWeight"=>600});
        let semibold = pdf.add_object(dictionary! {"Type"=>"Font","Subtype"=>"Type1","BaseFont"=>"Helvetica","FontDescriptor"=>descriptor});
        let resources =
            pdf.add_object(dictionary! {"Font"=>dictionary!{"F1"=>regular,"F4"=>semibold}});
        let mut ops = text(
            "F1",
            12,
            40,
            750,
            "Regular paragraph with enough text to establish its body font.",
        );
        ops.extend(text(
            "F4",
            12,
            40,
            714,
            "Semibold words set apart by weight.",
        ));
        ops.extend(text(
            "F1",
            12,
            40,
            680,
            "Plain text after the semibold line remains plain text.",
        ));
        let content = Content { operations: ops }.encode().unwrap();
        let stream = pdf.add_object(Stream::new(Dictionary::new(), content));
        let page = pdf.add_object(dictionary! {"Type"=>"Page","Parent"=>pages_id,"Contents"=>stream,"Resources"=>resources});
        pdf.objects.insert(pages_id,dictionary!{"Type"=>"Pages","Count"=>1,"Kids"=>vec![page.into()],"MediaBox"=>vec![0.into(),0.into(),600.into(),800.into()]}.into());
        let catalog = pdf.add_object(dictionary! {"Type"=>"Catalog","Pages"=>pages_id});
        pdf.trailer.set("Root", catalog);
        let mut bytes = Vec::new();
        pdf.save_to(&mut bytes).unwrap();
        let baseline = pdf_inspector::extract_pages_markdown_mem(&bytes, None).unwrap();
        assert!(!baseline.pages[0].markdown.contains("**"));
        let markdown = super::super::extract(&bytes).unwrap().markdown;
        assert!(
            markdown.contains("**Semibold words set apart by weight.**"),
            "{markdown}"
        );
        assert!(!markdown.contains("**Plain text"), "{markdown}");
    }

    #[test]
    fn real_pdf_ruled_table_keeps_header_blank_column_and_multiline_cell() {
        let mut ops = text(
            "F1",
            12,
            40,
            700,
            "This paragraph comes before a ruled table with four columns.",
        );
        for x in [40, 80, 330, 410, 520] {
            ops.push(Operation::new("m", vec![x.into(), 400.into()]));
            ops.push(Operation::new("l", vec![x.into(), 520.into()]));
            ops.push(Operation::new("S", vec![]));
        }
        for y in [400, 440, 480, 520] {
            ops.push(Operation::new("m", vec![40.into(), y.into()]));
            ops.push(Operation::new("l", vec![520.into(), y.into()]));
            ops.push(Operation::new("S", vec![]));
        }
        for (x, y, value) in [
            (86, 502, "Product"),
            (336, 502, "Count"),
            (416, 502, "Note"),
            (46, 462, "1"),
            (86, 462, "Long description"),
            (86, 448, "continues in the same cell"),
            (336, 462, "8"),
            (46, 422, "2"),
            (86, 422, "Short description"),
            (336, 422, "3"),
        ] {
            ops.extend(text("F1", 12, x, y, value));
        }
        ops.extend(text(
            "F1",
            12,
            40,
            350,
            "This paragraph comes after the table without losing any cells.",
        ));
        let result = super::super::extract(&pdf(vec![ops], None)).unwrap();
        assert!(result.markdown.contains("||Product|Count|Note|\n|---|---|---|---|\n|1|Long description<br>continues in the same cell|8||\n|2|Short description|3||"),"{}",result.markdown);
        assert!(
            result.markdown.find("before a ruled table").unwrap()
                < result.markdown.find("|Product|").unwrap()
        );
        assert!(
            result.markdown.find("|2|").unwrap() < result.markdown.find("after the table").unwrap()
        );
    }

    #[test]
    fn hidden_and_rotated_real_pdf_pages_keep_the_existing_reader() {
        for hidden in [false, true] {
            let mut ops = text(
                "F1",
                12,
                40,
                700,
                "Visible content remains on the existing page reader when layout is unsafe.",
            );
            if hidden {
                ops.push(Operation::new("Tr", vec![3.into()]));
                ops.extend(text(
                    "F1",
                    12,
                    40,
                    650,
                    "HIDDEN SECRET MUST NEVER ENTER THE RECONSTRUCTED TEXT",
                ));
            }
            let bytes = pdf(vec![ops], if hidden { None } else { Some(90) });
            let original = pdf_inspector::extract_pages_markdown_mem(&bytes, None).unwrap();
            let result = super::super::extract(&bytes).unwrap();
            assert_eq!(
                result.markdown,
                format!(
                    "<!-- Page number: 1 -->\n\n{}",
                    original.pages[0].markdown.trim()
                )
            );
            if hidden {
                // Both extraction paths exclude the nonpainting text. The
                // conservative layout gate still leaves the page unchanged.
                assert!(!original.pages[0].markdown.contains("HIDDEN SECRET"));
                assert!(!result.markdown.contains("HIDDEN SECRET"));
                assert!(
                    result.warnings.iter().any(|warning| warning
                        .contains("invisible text rendering mode")
                        && warning.contains("complete hidden-text filtering is not established"))
                );
            }
        }
    }

    /// Detection context for a page without a previous table or tags.
    fn untagged(pitch: &mut Option<f32>) -> Tables<'_> {
        Tables {
            continued: None,
            pitch,
            tagged: &|| false,
        }
    }

    #[test]
    fn malformed_geometry_and_disagreeing_text_decline_refinement() {
        let bytes = pdf(
            vec![text(
                "F1",
                12,
                40,
                700,
                "Valid native words make the independently authored source readable.",
            )],
            None,
        );
        let items = pdf_inspector::extract_text_with_positions_mem(&bytes).unwrap();
        let baseline = items
            .iter()
            .filter(|i| matches!(i.item_type, ItemType::Text))
            .map(|i| i.text.as_str())
            .collect::<String>();
        let frame = Frame {
            x: 0.,
            y: 0.,
            width: 600.,
            height: 800.,
        };
        assert!(
            render(
                items.clone(),
                &Faces::default(),
                frame,
                vec![],
                &[],
                "Different text from another decoder.",
                &mut untagged(&mut None)
            )
            .is_none()
        );
        let mut hebrew = items.clone();
        hebrew[0].text = "שלום עולם".into();
        let hebrew_baseline = hebrew.iter().map(|i| i.text.as_str()).collect::<String>();
        assert!(
            render(
                hebrew,
                &Faces::default(),
                frame,
                vec![],
                &[],
                &hebrew_baseline,
                &mut untagged(&mut None)
            )
            .is_none()
        );
        for (bad_x, bad_rotation) in [(f32::NAN, 0.), (-100., 0.), (40., 45.)] {
            let mut bad = items.clone();
            bad[0].x = bad_x;
            bad[0].rotation = bad_rotation;
            assert!(
                render(
                    bad,
                    &Faces::default(),
                    frame,
                    vec![],
                    &[],
                    &baseline,
                    &mut untagged(&mut None)
                )
                .is_none()
            );
        }
    }

    #[test]
    fn separate_prose_columns_do_not_become_one_paragraph_or_a_table() {
        let mut ops = Vec::new();
        for y in [700, 686, 672] {
            ops.extend(text("F1", 12, 40, y, "Left column paragraph."));
            ops.extend(text("F1", 12, 350, y, "Right column paragraph."));
        }
        let bytes = pdf(vec![ops], None);
        let loaded = pdf_inspector::LoadedPdf::load_mem(&bytes).unwrap();
        let mut layout =
            Layout::read(Some(&loaded), &HashSet::from([1]), &BTreeMap::new()).unwrap();
        let doc = lopdf::Document::load_mem(&bytes).unwrap();
        let id = doc.get_pages()[&1];
        let frame = super::super::geometry::frame(&doc, id).unwrap();
        let (_, content) = super::super::inspect_page(&doc, id);
        let grids = super::super::geometry::grids(
            &content.unwrap().operations,
            frame,
            &super::super::geometry::rule_resources(&doc, id),
        );
        let baseline = pdf_inspector::extract_pages_markdown_mem(&bytes, None).unwrap();
        assert!(
            layout
                .page(1, frame, grids, &[], &baseline.pages[0].markdown, &|| false)
                .is_none()
        );
    }

    /// A browser-printed ruled table continuing across pages, drawn like
    /// Chrome's print: a page background, per-cell border rectangles under a
    /// neutral graphics state, two-line notes around each row's baseline, and
    /// each page's last row reaching the bottom band where running folios live.
    #[test]
    fn printed_table_across_pages_keeps_every_row_and_edge_value() {
        let mut pdf = lopdf::Document::with_version("1.7");
        let pages_id = pdf.new_object_id();
        let font =
            pdf.add_object(dictionary! {"Type"=>"Font","Subtype"=>"Type1","BaseFont"=>"Helvetica"});
        let neutral = pdf.add_object(dictionary! {"ca"=>1,"BM"=>"Normal"});
        let resources = pdf.add_object(
            dictionary! {"Font"=>dictionary!{"F1"=>font},"ExtGState"=>dictionary!{"G3"=>neutral}},
        );
        let fill = |ops: &mut Vec<Operation>, color: f32, [x, y, w, h]: [f32; 4]| {
            ops.push(Operation::new(
                "rg",
                vec![color.into(), color.into(), color.into()],
            ));
            ops.push(Operation::new(
                "re",
                vec![x.into(), y.into(), w.into(), h.into()],
            ));
            ops.push(Operation::new("f", vec![]));
        };
        let words = |ops: &mut Vec<Operation>, x: f32, y: f32, value: &str| {
            ops.extend([
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec![Object::Name(b"F1".to_vec()), 9.into()]),
                Operation::new("Td", vec![x.into(), y.into()]),
                Operation::new("Tj", vec![Object::string_literal(value)]),
                Operation::new("ET", vec![]),
            ]);
        };
        let columns = [(51., 165.), (216., 165.), (381., 164.25)];
        let mut kids = Vec::new();
        let mut row = 0;
        for _ in 0..3 {
            let mut ops = vec![
                Operation::new("q", vec![]),
                Operation::new("gs", vec![Object::Name(b"G3".to_vec())]),
            ];
            fill(&mut ops, 0.96, [51., 48.63, 494.25, 743.25]);
            fill(&mut ops, 1., [51., 48.63, 494.25, 743.25]);
            let mut rows = vec![(764.88f32, 791.88f32)];
            rows.extend(
                (0..18).map(|n| (764.88 - 39.75 * (n + 1) as f32, 764.88 - 39.75 * n as f32)),
            );
            for (index, &(bottom, top)) in rows.iter().enumerate() {
                for (x, width) in columns {
                    if index == 0 {
                        fill(
                            &mut ops,
                            0.96,
                            [x + 0.75, bottom, width - 0.75, top - bottom - 0.75],
                        );
                    }
                    fill(&mut ops, 0.88, [x, bottom, 0.75, top - bottom]);
                    fill(&mut ops, 0.88, [x, top - 0.75, width, 0.75]);
                }
                fill(&mut ops, 0.88, [544.5, bottom, 0.75, top - bottom]);
            }
            for (x, width) in columns {
                fill(&mut ops, 0.88, [x, rows[rows.len() - 1].0, width, 0.75]);
            }
            ops.push(Operation::new("Q", vec![]));
            ops.push(Operation::new("rg", vec![0.into(), 0.into(), 0.into()]));
            for (x, value) in [(120.6, "Name"), (282.1, "Square"), (449.3, "Notes")] {
                words(&mut ops, x, 773., value);
            }
            for &(bottom, top) in &rows[1..] {
                row += 1;
                let baseline = (bottom + top) / 2. - 3.;
                words(&mut ops, 60.8, baseline, &format!("Row {row}"));
                words(&mut ops, 225.2, baseline, &(row * row).to_string());
                words(&mut ops, 389.7, baseline + 6.7, "long cell text long");
                words(&mut ops, 389.7, baseline - 6.7, "cell text");
            }
            let content = Content { operations: ops }.encode().unwrap();
            let stream = pdf.add_object(Stream::new(Dictionary::new(), content));
            let id = pdf.add_object(dictionary! {"Type"=>"Page","Parent"=>pages_id,"Contents"=>stream,"Resources"=>resources});
            kids.push(Object::Reference(id));
        }
        pdf.objects.insert(pages_id,dictionary!{"Type"=>"Pages","Count"=>3,"Kids"=>kids,"MediaBox"=>vec![0.into(),0.into(),595.92.into(),842.88.into()]}.into());
        let catalog = pdf.add_object(dictionary! {"Type"=>"Catalog","Pages"=>pages_id});
        pdf.trailer.set("Root", catalog);
        let mut bytes = Vec::new();
        pdf.save_to(&mut bytes).unwrap();
        let markdown = super::super::extract(&bytes).unwrap().markdown;
        // Each page repeats the header row: one table, its header once, and
        // the next pages keep only their markers.
        let rows = (1..=54)
            .map(|row| format!("|Row {row}|{}|long cell text long<br>cell text|", row * row))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            markdown.contains(&format!(
                "|Name|Square|Notes|\n|---|---|---|\n{rows}\n\n\
                 <!-- Page number: 2 -->\n\n<!-- Page number: 3 -->"
            )),
            "{markdown}"
        );
        assert_eq!(markdown.matches("|Name|Square|Notes|").count(), 1);
    }

    /// A ruled three-column table between the column borders `xs` from
    /// `top` down, rows 30pt high, each row's cells in its font.
    fn ruled(xs: [i64; 4], top: i64, rows: &[(&str, [String; 3])]) -> Vec<Operation> {
        let bottom = top - 30 * rows.len() as i64;
        let mut ops = Vec::new();
        let mut rule = |from: (i64, i64), to: (i64, i64)| {
            ops.push(Operation::new("m", vec![from.0.into(), from.1.into()]));
            ops.push(Operation::new("l", vec![to.0.into(), to.1.into()]));
            ops.push(Operation::new("S", vec![]));
        };
        for x in xs {
            rule((x, bottom), (x, top));
        }
        for k in 0..=rows.len() as i64 {
            rule((xs[0], top - 30 * k), (xs[3], top - 30 * k));
        }
        for (index, (font, values)) in rows.iter().enumerate() {
            let y = top - 20 - 30 * index as i64;
            for (x, value) in xs.iter().zip(values) {
                ops.extend(text(font, 12, x + 6, y, value));
            }
        }
        ops
    }

    const XS: [i64; 4] = [40, 200, 360, 520];

    /// `Row n | n² | note n` for each `n`, in `font`.
    fn numbered(font: &str, rows: std::ops::Range<i64>) -> Vec<(&str, [String; 3])> {
        rows.map(|n| {
            (
                font,
                [format!("Row {n}"), (n * n).to_string(), format!("note {n}")],
            )
        })
        .collect()
    }

    /// A ruled table of `XS` from `top` down: a bold header row when given,
    /// then the numbered rows `rows`.
    fn ruled_table(
        top: i64,
        header: Option<[&str; 3]>,
        rows: std::ops::Range<i64>,
    ) -> Vec<Operation> {
        let mut all: Vec<(&str, [String; 3])> = header
            .map(|header| ("F2", header.map(String::from)))
            .into_iter()
            .collect();
        all.extend(numbered("F1", rows));
        ruled(XS, top, &all)
    }

    const HEADER: [&str; 3] = ["Name", "Square", "Notes"];

    fn rows_markdown(rows: std::ops::Range<i64>) -> String {
        rows.map(|n| format!("|Row {n}|{}|note {n}|", n * n))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn a_ruled_table_cut_by_a_page_break_is_one_table() {
        // The table runs into the page's lowest fifth (it ends at 150 of
        // 800pt), and the next page opens with the same columns.
        let mut first = text("F1", 12, 40, 770, "The register follows.");
        first.extend(ruled_table(750, Some(HEADER), 1..20));
        let header = "|**Name**|**Square**|**Notes**|\n|---|---|---|";
        // The header row repeated, or no header row: either continues.
        for repeated in [Some(HEADER), None] {
            let mut second = ruled_table(770, repeated, 20..24);
            second.extend(text("F1", 12, 40, 560, "The register ends here."));
            let pages =
                super::super::extract_pages(&pdf(vec![first.clone(), second], None)).unwrap();
            assert!(pages.pages[1].continues_table);
            let markdown = pages.finish().unwrap().markdown;
            assert_eq!(
                markdown,
                format!(
                    "<!-- Page number: 1 -->\n\nThe register follows.\n\n{header}\n{}\n\n\
                     <!-- Page number: 2 -->\n\nThe register ends here.",
                    rows_markdown(1..24)
                )
            );
        }
    }

    #[test]
    fn a_new_table_at_the_top_of_the_next_page_stays_a_table_of_its_own() {
        let mut first = text("F1", 12, 40, 770, "The register follows.");
        first.extend(ruled_table(750, Some(HEADER), 1..20));
        // A header of its own, with the same columns.
        let second = ruled_table(770, Some(["City", "Code", "Region"]), 20..24);
        let pages = super::super::extract_pages(&pdf(vec![first.clone(), second], None)).unwrap();
        assert!(!pages.pages[1].continues_table);
        let markdown = pages.finish().unwrap().markdown;
        assert!(
            markdown.contains(&format!(
                "<!-- Page number: 2 -->\n\n|**City**|**Code**|**Region**|\n|---|---|---|\n{}",
                rows_markdown(20..24)
            )),
            "{markdown}"
        );
        // Other column borders, or a caption above it: a new table.
        for second in [
            ruled([40, 240, 360, 520], 770, &numbered("F1", 20..24)),
            [
                text("F1", 12, 40, 775, "Cities by code."),
                ruled(XS, 750, &numbered("F1", 20..24)),
            ]
            .concat(),
        ] {
            let pages =
                super::super::extract_pages(&pdf(vec![first.clone(), second], None)).unwrap();
            assert!(!pages.pages[1].continues_table);
            let markdown = pages.finish().unwrap().markdown;
            assert_eq!(markdown.matches("|---|---|---|").count(), 2, "{markdown}");
        }
        // A first row in bold above other bold rows is not set apart as a
        // header: the table continues.
        let second = ruled(XS, 770, &numbered("F2", 20..24));
        let pages = super::super::extract_pages(&pdf(vec![first.clone(), second], None)).unwrap();
        assert!(pages.pages[1].continues_table);
        // A table ending well above the page's foot was not cut: a table
        // without a header at the top of the next page starts anew.
        let mut first = text("F1", 12, 40, 770, "The register follows.");
        first.extend(ruled_table(750, Some(HEADER), 1..6));
        let second = ruled_table(770, None, 20..24);
        let pages = super::super::extract_pages(&pdf(vec![first, second], None)).unwrap();
        assert!(!pages.pages[1].continues_table);
        let markdown = pages.finish().unwrap().markdown;
        assert!(
            markdown.contains(&format!(
                "{}\n\n<!-- Page number: 2 -->\n\n|Row 20|400|note 20|\n|---|---|---|\n{}",
                rows_markdown(1..6),
                rows_markdown(21..24)
            )),
            "{markdown}"
        );
    }

    #[test]
    fn an_empty_first_row_of_a_ruled_table_is_no_header() {
        // A row with nothing in it above the header row.
        let mut page = text("F1", 12, 40, 770, "The register follows.");
        page.extend(ruled_table(750, Some(["", "", ""]), 1..1));
        page.extend(ruled_table(720, Some(HEADER), 1..4));
        let markdown = super::super::extract(&pdf(vec![page], None))
            .unwrap()
            .markdown;
        assert!(
            markdown.contains(&format!(
                "The register follows.\n\n|**Name**|**Square**|**Notes**|\n|---|---|---|\n{}",
                rows_markdown(1..4)
            )),
            "{markdown}"
        );
    }

    /// Link annotations do not block refinement, and neither table cells nor
    /// a fixed-pitch code block define the body size that headings exceed.
    #[test]
    fn linked_page_with_small_table_and_code_keeps_prose_as_paragraphs() {
        let mut pdf = lopdf::Document::with_version("1.7");
        let pages_id = pdf.new_object_id();
        let regular =
            pdf.add_object(dictionary! {"Type"=>"Font","Subtype"=>"Type1","BaseFont"=>"Helvetica"});
        let bold = pdf.add_object(
            dictionary! {"Type"=>"Font","Subtype"=>"Type1","BaseFont"=>"Helvetica-Bold"},
        );
        let code =
            pdf.add_object(dictionary! {"Type"=>"Font","Subtype"=>"Type1","BaseFont"=>"Courier"});
        let resources =
            pdf.add_object(dictionary! {"Font"=>dictionary!{"F1"=>regular,"F2"=>bold,"F3"=>code}});
        let mut ops = text("F2", 24, 40, 760, "Print acceptance");
        ops.extend(text(
            "F1",
            11,
            40,
            730,
            "A document with a small table and a code block.",
        ));
        ops.extend(text(
            "F1",
            11,
            40,
            710,
            "Visit the external site for details.",
        ));
        for y in [690, 680, 670, 660] {
            ops.extend(text(
                "F3",
                8,
                40,
                y,
                "code code code code code code code code code code code",
            ));
        }
        for x in [40, 200, 360, 520] {
            ops.push(Operation::new("m", vec![x.into(), 400.into()]));
            ops.push(Operation::new("l", vec![x.into(), 640.into()]));
            ops.push(Operation::new("S", vec![]));
        }
        for y in (400..=640).step_by(20) {
            ops.push(Operation::new("m", vec![40.into(), y.into()]));
            ops.push(Operation::new("l", vec![520.into(), y.into()]));
            ops.push(Operation::new("S", vec![]));
        }
        for (row, y) in (0..12).map(|index| 626 - 20 * index).enumerate() {
            ops.extend(text("F1", 9, 46, y, &format!("Row {row} label text")));
            ops.extend(text("F1", 9, 206, y, &format!("value {row} value")));
            ops.extend(text("F1", 9, 366, y, "long cell text long"));
        }
        let content = Content { operations: ops }.encode().unwrap();
        let stream = pdf.add_object(Stream::new(Dictionary::new(), content));
        let link = pdf.add_object(dictionary! {"Type"=>"Annot","Subtype"=>"Link","Rect"=>vec![40.into(),706.into(),240.into(),720.into()],"A"=>dictionary!{"S"=>"URI","URI"=>Object::string_literal("https://example.test/")}});
        let page = pdf.add_object(dictionary! {"Type"=>"Page","Parent"=>pages_id,"Contents"=>stream,"Resources"=>resources,"Annots"=>vec![link.into()]});
        pdf.objects.insert(pages_id,dictionary!{"Type"=>"Pages","Count"=>1,"Kids"=>vec![page.into()],"MediaBox"=>vec![0.into(),0.into(),600.into(),800.into()]}.into());
        let catalog = pdf.add_object(dictionary! {"Type"=>"Catalog","Pages"=>pages_id});
        pdf.trailer.set("Root", catalog);
        let mut bytes = Vec::new();
        pdf.save_to(&mut bytes).unwrap();
        let markdown = super::super::extract(&bytes).unwrap().markdown;
        assert!(markdown.contains("# **Print acceptance**"), "{markdown}");
        for prose in ["A document with a small table", "Visit the external site"] {
            let line = markdown.lines().find(|line| line.contains(prose)).unwrap();
            assert!(!line.starts_with('#'), "{line}\n{markdown}");
        }
        assert!(
            markdown.contains("|Row 0 label text|value 0 value|long cell text long|"),
            "{markdown}"
        );
    }

    #[test]
    fn real_pdf_keeps_formula_punctuation_and_literal_markdown_metacharacters() {
        let mut ops = text(
            "F1",
            12,
            40,
            700,
            "Formula prose: a - b + c = (d / e); [literal] * stars_under.",
        );
        ops.extend(text(
            "F1",
            12,
            40,
            686,
            "Keep signs and parentheses, then finish with punctuation!",
        ));
        let result = super::super::extract(&pdf(vec![ops], None)).unwrap();
        assert!(
            result
                .markdown
                .contains("a - b + c = (d / e); \\[literal\\] \\* stars\\_under."),
            "{}",
            result.markdown
        );
        assert!(
            result
                .markdown
                .contains("parentheses, then finish with punctuation!")
        );
    }

    /// Text runs at `(x, y)` in 12pt Helvetica.
    fn runs_at(cells: &[(i64, i64, &str)]) -> Vec<Operation> {
        cells
            .iter()
            .flat_map(|&(x, y, value)| text("F1", 12, x, y, value))
            .collect()
    }

    /// Four lines of running text from `top`, 15pt apart (1.25 em).
    fn prose_from(top: i64) -> Vec<Operation> {
        runs_at(&[
            (
                40,
                top,
                "The plans below differ in price, storage and the support they",
            ),
            (
                40,
                top - 15,
                "include. Each plan can be changed at the end of any month, and",
            ),
            (
                40,
                top - 30,
                "a change takes effect on the first day of the following month",
            ),
            (
                40,
                top - 45,
                "without any charge for the switch itself or for the new plan.",
            ),
        ])
    }

    /// The borderless tables the detector finds on one authored page.
    fn borderless(page: Vec<Operation>) -> Vec<String> {
        let bytes = pdf(vec![page], None);
        let items = pdf_inspector::extract_text_with_positions_mem(&bytes)
            .unwrap()
            .into_iter()
            .filter(|i| matches!(i.item_type, ItemType::Text) && !i.text.trim().is_empty())
            .collect();
        let lines = lines(items);
        unruled::find(&lines, unruled::prose_pitch(&lines), &[], None)
            .into_iter()
            .map(|table| table.markdown)
            .collect()
    }

    #[test]
    fn wrapped_top_aligned_cells_without_rules_are_rows_by_their_spacing() {
        // Each cell's lines follow at the running text's pitch (15pt); rows
        // are 6pt further apart. The header cells wrap too, and a note a
        // paragraph below starts at the first column without being a row.
        let mut page = prose_from(740);
        page.extend(runs_at(&[
            (200, 650, "Monthly"),
            (300, 650, "Storage"),
            (400, 650, "Support"),
            (200, 635, "fee"),
            (300, 635, "included"),
            (40, 614, "Starter plan for"),
            (200, 614, "9 EUR"),
            (300, 614, "10 GB"),
            (400, 614, "Email"),
            (40, 599, "small teams"),
            (40, 578, "Growth"),
            (200, 578, "29 EUR"),
            (300, 578, "100 GB"),
            (400, 578, "Chat and"),
            (400, 563, "email"),
            (40, 542, "Enterprise"),
            (200, 542, "Custom"),
            (300, 542, "Unlimited"),
            (400, 542, "Phone"),
            (40, 527, "agreement"),
            (40, 491, "Prices exclude tax."),
        ]));
        let markdown = super::super::extract(&pdf(vec![page], None))
            .unwrap()
            .markdown;
        assert!(
            markdown.contains(
                "for the new plan.\n\n\
                 ||Monthly fee|Storage included|Support|\n\
                 |---|---|---|---|\n\
                 |Starter plan for small teams|9 EUR|10 GB|Email|\n\
                 |Growth|29 EUR|100 GB|Chat and email|\n\
                 |Enterprise agreement|Custom|Unlimited|Phone|\n\nPrices exclude tax."
            ),
            "{markdown}"
        );
    }

    #[test]
    fn centred_rows_whose_labels_always_wrap_keep_their_label_column() {
        // No running text gives a pitch: a value centred between a label's
        // two lines shows they are one cell. No label shares a baseline with
        // the values, so the label column comes from the lines between them.
        let mut page = text("F2", 18, 40, 770, "Climate by region");
        page.extend(runs_at(&[
            (40, 740, "Measured over ten years at regional stations."),
            // A caption just above the table keeps to its first column.
            (40, 718, "Regional averages"),
            (220, 700, "North"),
            (320, 700, "South"),
            (420, 700, "East"),
            (40, 680, "Annual rainfall"),
            (220, 673, "812"),
            (320, 673, "640"),
            (420, 673, "455"),
            (40, 666, "in millimetres"),
            (40, 646, "Days of frost"),
            (220, 639, "31"),
            (320, 639, "18"),
            (420, 639, "9"),
            (40, 632, "per winter"),
            (40, 612, "Mean summer"),
            (220, 605, "17"),
            (320, 605, "21"),
            (420, 605, "24"),
            (40, 598, "temperature"),
            (40, 560, "Stations report every hour."),
        ]));
        let markdown = super::super::extract(&pdf(vec![page], None))
            .unwrap()
            .markdown;
        assert!(
            markdown.contains(
                "regional stations.\n\nRegional averages\n\n\
                 ||North|South|East|\n\
                 |---|---|---|---|\n\
                 |Annual rainfall in millimetres|812|640|455|\n\
                 |Days of frost per winter|31|18|9|\n\
                 |Mean summer temperature|17|21|24|\n\nStations report every hour."
            ),
            "{markdown}"
        );
    }

    #[test]
    fn a_small_cells_pitch_a_pixel_off_is_still_one_cell() {
        // A browser sets lines on whole pixels: in 9pt text one cell's line
        // pitch measures 10.5pt and another's 11.2pt, more than 6% apart.
        let mut page = Vec::new();
        let mut put = |x: f32, y: f32, value: &str| {
            page.extend([
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec![Object::Name(b"F1".to_vec()), 9.into()]),
                Operation::new("Td", vec![x.into(), y.into()]),
                Operation::new("Tj", vec![Object::string_literal(value)]),
                Operation::new("ET", vec![]),
            ])
        };
        for (x, value) in [(200., "Grade"), (300., "Width"), (400., "Depth")] {
            put(x, 700., value);
        }
        for (top, step, label, values) in [
            (680., 10.5, ["Oak board", "planed"], ["A", "180", "22"]),
            (655.5, 11.2, ["Pine strip", "rough sawn"], ["B", "95", "19"]),
            (630.3, 10.5, ["Ash panel", "kiln dried"], ["A", "240", "28"]),
        ] {
            put(40., top, label[0]);
            put(40., top - step, label[1]);
            for (x, value) in [200., 300., 400.].into_iter().zip(values) {
                put(x, top - step / 2., value);
            }
        }
        assert_eq!(
            borderless(page),
            [
                "||Grade|Width|Depth|\n|---|---|---|---|\n|Oak board planed|A|180|22|\n|Pine strip rough sawn|B|95|19|\n|Ash panel kiln dried|A|240|28|"
            ]
        );
    }

    #[test]
    fn side_by_side_paragraphs_are_not_a_borderless_table() {
        // Three columns of running text, their paragraphs a few points
        // apart and starting at different heights: aligned stretches of text
        // with gutters, but cells that read as sentences.
        let mut page = Vec::new();
        for (column, x, lengths) in [(0, 40, [4, 3, 5]), (1, 220, [3, 5, 4]), (2, 400, [5, 4, 3])] {
            let mut y = 720 - column * 8;
            for (paragraph, length) in lengths.into_iter().enumerate() {
                for line in 0..length {
                    let words = [
                        "harbour lanterns glow late",
                        "over quiet copper roofs",
                        "while the morning tide",
                        "turns along the old wall",
                    ];
                    page.extend(text(
                        "F1",
                        12,
                        x,
                        y,
                        words[(paragraph + line + column as usize) % 4],
                    ));
                    y -= 14;
                }
                y -= 8;
            }
        }
        assert_eq!(borderless(page), Vec::<String>::new());
    }

    #[test]
    fn rows_of_single_lines_are_left_to_the_page_reader() {
        // No cell of the grid wraps, so nothing in it shows which spacing is
        // a cell's own (the note below, at the text's pitch, is no row):
        // the page reader's alignment-based tables cover such grids.
        let mut page = prose_from(760);
        for row in 0..6 {
            let y = 680 - row * 21;
            page.extend(runs_at(&[
                (40, y, "Item"),
                (200, y, "North"),
                (300, y, "South"),
                (400, y, "East"),
            ]));
        }
        page.extend(runs_at(&[(40, 554, "Values are"), (40, 539, "rounded.")]));
        assert_eq!(borderless(page), Vec::<String>::new());
    }

    /// Centred rows as in `centred_rows_whose_labels_always_wrap_keep_their_label_column`,
    /// without the caption and the paragraphs around: the header, then rows
    /// whose two-line labels start at `top` and step 34pt down.
    fn centred(rows: &[(&str, &str, [&str; 3])]) -> Vec<Operation> {
        let mut page = runs_at(&[(220, 700, "North"), (320, 700, "South"), (420, 700, "East")]);
        for (index, (first, second, values)) in rows.iter().enumerate() {
            let top = 680 - 34 * index as i64;
            page.extend(runs_at(&[(40, top, first), (40, top - 14, second)]));
            for (x, value) in [220, 320, 420].into_iter().zip(values) {
                page.extend(runs_at(&[(x, top - 7, value)]));
            }
        }
        page
    }

    const RAINFALL: (&str, &str, [&str; 3]) =
        ("Annual rainfall", "in millimetres", ["812", "640", "455"]);
    const FROST: (&str, &str, [&str; 3]) = ("Days of frost", "per winter", ["31", "18", "9"]);
    const SUMMER: (&str, &str, [&str; 3]) = ("Mean summer", "temperature", ["17", "21", "24"]);

    #[test]
    fn a_borderless_table_needs_three_rows_each_of_two_cells() {
        // The rows of the centred example are found, but a row of a lone
        // label (a heading or a stray line) is no row.
        assert_eq!(borderless(centred(&[RAINFALL, FROST, SUMMER])).len(), 1);
        let closed = ("Closed for", "the winter", ["", "", ""]);
        assert_eq!(
            borderless(centred(&[RAINFALL, closed, FROST, SUMMER])),
            Vec::<String>::new()
        );
        // A header and one row are not enough either.
        let mut page = prose_from(780);
        page.extend(runs_at(&[
            (40, 690, "Plan"),
            (200, 690, "Monthly"),
            (300, 690, "Storage"),
            (400, 690, "Support"),
            (40, 669, "Starter plan for"),
            (200, 669, "9 EUR"),
            (300, 669, "10 GB"),
            (400, 669, "Email"),
            (40, 654, "small teams"),
        ]));
        assert_eq!(borderless(page), Vec::<String>::new());
    }

    #[test]
    fn rows_of_two_cells_under_a_wide_header_are_not_a_table() {
        // Only the header fills three cells; each row below fills two.
        let mut page = prose_from(780);
        page.extend(runs_at(&[
            (40, 690, "Station"),
            (220, 690, "Mean"),
            (320, 690, "Peak"),
            (420, 690, "Days of"),
            (40, 675, "name"),
            (220, 675, "rain"),
            (320, 675, "wind"),
            (420, 675, "frost"),
            (40, 654, "North"),
            (220, 654, "812"),
            (40, 639, "ridge"),
            (40, 618, "South"),
            (320, 618, "41"),
            (40, 603, "valley"),
            (40, 582, "East"),
            (420, 582, "9"),
            (40, 567, "coast"),
        ]));
        assert_eq!(borderless(page), Vec::<String>::new());
    }

    #[test]
    fn cells_that_read_as_paragraphs_are_not_a_borderless_table() {
        // Two of every row's four cells hold a sentence of 13 words or more.
        let mut page = prose_from(780);
        page.extend(runs_at(&[
            (160, 690, "Free plan"),
            (330, 690, "Pro plan"),
            (500, 690, "Price"),
        ]));
        for (top, label, free, pro, price) in [
            (
                669,
                "Members",
                [
                    "Five members can open",
                    "and edit shared files",
                    "from any of their devices",
                ],
                [
                    "Fifty members can open",
                    "and edit shared files",
                    "with history of every day",
                ],
                "9 EUR",
            ),
            (
                618,
                "Storage",
                [
                    "Ten gigabytes of space",
                    "for all the files the",
                    "team keeps in the folders",
                ],
                [
                    "One terabyte of space",
                    "for all the files the",
                    "team keeps in the folders",
                ],
                "19 EUR",
            ),
            (
                567,
                "Support",
                [
                    "Support by email within",
                    "two working days of the",
                    "first message from a user",
                ],
                [
                    "Support by phone within",
                    "two working hours of the",
                    "first message from a user",
                ],
                "29 EUR",
            ),
        ] {
            for (line, (free, pro)) in free.into_iter().zip(pro).enumerate() {
                let y = top - 15 * line as i64;
                page.extend(runs_at(&[(160, y, free), (330, y, pro)]));
            }
            page.extend(runs_at(&[(40, top - 15, label), (500, top - 15, price)]));
        }
        assert_eq!(borderless(page), Vec::<String>::new());
    }

    #[test]
    fn a_cell_spanning_two_rows_declines_the_table() {
        // The note's two lines overlap the rows of both Alpha and Beta.
        let mut page = prose_from(780);
        page.extend(runs_at(&[
            (200, 690, "Group"),
            (300, 690, "Score"),
            (400, 690, "Note"),
            (40, 669, "Alpha"),
            (200, 669, "A"),
            (300, 669, "12"),
            (400, 666, "Both joined"),
            (400, 651, "in March"),
            (40, 648, "Beta"),
            (200, 648, "A"),
            (300, 648, "15"),
            (40, 627, "Gamma"),
            (200, 627, "B"),
            (300, 627, "9"),
            (400, 627, "New"),
            (40, 606, "Delta"),
            (200, 606, "B"),
            (300, 606, "11"),
            (400, 606, "New"),
        ]));
        assert_eq!(borderless(page), Vec::<String>::new());
    }

    #[test]
    fn cells_whose_left_edges_drift_do_not_make_columns() {
        // The counts are centred: their left edges drift with their width
        // and no two rows share one. Taken as no column of its own, each
        // count would join the kind beside it ("Hard 7"); too few cells start
        // at a shared edge to tell the columns apart.
        let mut page = prose_from(780);
        page.extend(runs_at(&[
            (200, 690, "Kind"),
            (304, 690, "Count"),
            (400, 690, "Grade"),
        ]));
        for (top, label, kind, (x, count), grade) in [
            (669, ["Oak board", "planed"], "Hard", (330, "7"), "A"),
            (633, ["Pine strip", "rough"], "Soft", (324, "12"), "B"),
            (597, ["Ash panel", "dried"], "Hard", (318, "123"), "A"),
            (561, ["Elm beam", "oiled"], "Hard", (312, "1234"), "C"),
        ] {
            page.extend(runs_at(&[
                (40, top, label[0]),
                (200, top, kind),
                (x, top, count),
                (400, top, grade),
                (40, top - 15, label[1]),
            ]));
        }
        assert_eq!(borderless(page), Vec::<String>::new());
    }

    #[test]
    fn columns_without_a_clear_gutter_are_not_a_borderless_table() {
        // Every gap between cells is 1.5 em: words of a line can be as far
        // apart. ("Oak" is 22pt wide at 12pt, "X" and "A" 8pt.)
        let mut page = runs_at(&[(80, 700, "A"), (106, 700, "B"), (132, 700, "C")]);
        for (index, (first, second)) in [("Oak", "red"), ("Ash", "dry"), ("Elm", "wet")]
            .into_iter()
            .enumerate()
        {
            let top = 680 - 34 * index as i64;
            page.extend(runs_at(&[
                (40, top, first),
                (40, top - 14, second),
                (80, top - 7, "X"),
                (106, top - 7, "X"),
                (132, top - 7, "X"),
            ]));
        }
        assert_eq!(borderless(page), Vec::<String>::new());
    }

    #[test]
    fn a_row_step_between_a_cells_pitch_and_a_rows_is_no_evidence() {
        // Rows 16pt apart against a 15pt line pitch: neither the same cell
        // nor clearly a new row.
        let mut page = prose_from(760);
        page.extend(runs_at(&[
            (40, 680, "Starter plan for"),
            (200, 680, "9 EUR"),
            (300, 680, "10 GB"),
            (400, 680, "Email"),
            (40, 665, "small teams"),
            (40, 649, "Growth"),
            (200, 649, "29 EUR"),
            (300, 649, "100 GB"),
            (400, 649, "Chat and"),
            (400, 634, "email"),
            (40, 618, "Enterprise"),
            (200, 618, "Custom"),
            (300, 618, "Unlimited"),
            (400, 618, "Phone"),
        ]));
        assert_eq!(borderless(page), Vec::<String>::new());
    }
}
