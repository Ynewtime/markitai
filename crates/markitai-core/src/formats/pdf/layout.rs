//! Layout improvements only for pages with a reliable, upright text layer.
use super::geometry::{Frame, Grid};
use pdf_inspector::{TextItem, types::ItemType};
use std::collections::{BTreeMap, HashSet};

const MAX_ITEMS: usize = 250_000;
const MAX_TEXT: usize = 16 * 1024 * 1024;
const MAX_PAGE_ITEMS: usize = 20_000;

pub(super) struct Layout {
    pages: BTreeMap<u32, Vec<TextItem>>,
    headings: Vec<f32>,
}

impl Layout {
    pub(super) fn read(
        bytes: &[u8],
        selected: &HashSet<u32>,
    ) -> std::result::Result<Self, &'static str> {
        let (items, rotations) =
            pdf_inspector::extract_text_with_positions_and_rotations_mem_with_options(
                bytes,
                Some(selected),
                pdf_inspector::PositionOptions::new().bold_from_weight(true),
            )
            .map_err(|_| "positioned text could not be decoded")?;
        if items.len() > MAX_ITEMS || items.iter().map(|i| i.text.len()).sum::<usize>() > MAX_TEXT {
            return Err("positioned text exceeds the layout budget");
        }
        let mut pages: BTreeMap<u32, Vec<TextItem>> = BTreeMap::new();
        let mut sizes = BTreeMap::<i32, usize>::new();
        for item in items {
            if rotations.contains_key(&item.page) {
                continue;
            }
            if matches!(item.item_type, ItemType::Text) && valid(&item) {
                *sizes
                    .entry((item.font_size * 10.).round() as i32)
                    .or_default() += item.text.chars().filter(|c| !c.is_whitespace()).count();
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
            if size > body * 1.2
                && headings
                    .last()
                    .is_none_or(|previous| (*previous - size).abs() > size * 0.05)
            {
                headings.push(size);
            }
        }
        Ok(Self { pages, headings })
    }

    pub(super) fn page(
        &mut self,
        number: u32,
        frame: Frame,
        grids: Vec<Grid>,
        baseline: &str,
    ) -> Option<String> {
        let items = self.pages.remove(&number)?;
        render(items, &self.headings, frame, grids, baseline)
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
        && item.baseline_shift.abs() < 0.1
        && !matches!(item.render_mode, Some(3 | 7))
        && !item.text.chars().any(|c| {
            c == '\u{fffd}'
                || (c.is_control() && !c.is_whitespace())
                || matches!(c as u32, 0xe000..=0xf8ff | 0xf0000..=0xffffd | 0x100000..=0x10fffd)
        })
}

fn character_counts(text: &str) -> BTreeMap<char, usize> {
    let mut counts = BTreeMap::new();
    // These are generated decorations in the original native Markdown, not
    // arbitrary source HTML to interpret. Unknown constructs fail agreement.
    let text = text
        .replace("<u>", "")
        .replace("</u>", "")
        .replace("<s>", "")
        .replace("</s>", "")
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
}

impl From<&TextItem> for Style {
    fn from(item: &TextItem) -> Self {
        Self {
            bold: item.is_bold,
            italic: item.is_italic,
            underline: item.is_underline,
            strike: item.is_strikeout,
        }
    }
}

struct Run {
    text: String,
    style: Style,
}
struct Line {
    items: Vec<TextItem>,
    y: f32,
    size: f32,
}

fn lines(mut items: Vec<TextItem>) -> Vec<Line> {
    items.sort_by(|a, b| b.y.total_cmp(&a.y).then(a.x.total_cmp(&b.x)));
    let mut result: Vec<Line> = Vec::new();
    for item in items {
        if let Some(line) = result.last_mut()
            && (line.y - item.y).abs() <= (line.size.min(item.font_size) * 0.2).min(2.)
        {
            line.size = line.size.max(item.font_size);
            line.items.push(item);
        } else {
            result.push(Line {
                y: item.y,
                size: item.font_size,
                items: vec![item],
            });
        }
    }
    for line in &mut result {
        line.items.sort_by(|a, b| a.x.total_cmp(&b.x));
    }
    result
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
        if let Some(last) = out.last_mut()
            && last.style == style
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
            });
        }
        previous = Some(item);
    }
    out
}

fn append_runs(output: &mut Vec<Run>, source: Vec<Run>) {
    for (index, run) in source.into_iter().enumerate() {
        if let Some(last) = output.last_mut()
            && last.style == run.style
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

fn markdown(runs: &[Run]) -> String {
    let mut output = String::new();
    for run in runs {
        let text = run.text.trim();
        if text.is_empty() {
            continue;
        }
        if run.text.starts_with(char::is_whitespace) && !output.is_empty() && !output.ends_with(' ')
        {
            output.push(' ');
        }
        let style = run.style;
        if style.bold {
            output.push_str("**");
        }
        if style.italic {
            output.push('*');
        }
        if style.underline {
            output.push_str("<u>");
        }
        if style.strike {
            output.push_str("<s>");
        }
        output.push_str(
            &super::super::escape(text)
                .replace('<', "&lt;")
                .replace('>', "&gt;"),
        );
        if style.strike {
            output.push_str("</s>");
        }
        if style.underline {
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
        }
    }
    output.trim().to_owned()
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

fn flow(lines: &[Line], headings: &[f32]) -> Option<String> {
    let mut blocks = Vec::new();
    let mut paragraph = Vec::new();
    let mut previous: Option<&Line> = None;
    let mut previous_level = 0;
    let mut in_list = false;
    let flush = |blocks: &mut Vec<String>, paragraph: &mut Vec<Run>, level: usize, list: bool| {
        if paragraph.is_empty() {
            return;
        }
        let content = markdown(paragraph);
        let prefix = if level > 0 {
            format!("{} ", "#".repeat(level))
        } else if list {
            "- ".into()
        } else {
            String::new()
        };
        blocks.push(format!("{prefix}{content}"));
        paragraph.clear();
    };
    for line in lines {
        let mut current = runs(line);
        if current.is_empty() {
            continue;
        }
        let is_list = list_prefix(&current[0].text).is_some();
        if is_list {
            current[0].text = list_prefix(&current[0].text).unwrap().to_owned();
        }
        // Side-by-side prose and unruled tables need a reading-order model.
        // Keep the previous reader instead of concatenating their columns.
        if !is_list
            && line
                .items
                .windows(2)
                .any(|p| p[1].x - (p[0].x + p[0].width) > line.size * 3.)
        {
            return None;
        }
        let level = if is_list {
            0
        } else {
            heading_level(line, headings)
        };
        let new_block = previous.is_some_and(|prev| {
            let gap = prev.y - line.y;
            level != previous_level
                || is_list
                || gap > prev.size.max(line.size) * 1.8
                || (in_list && line.items[0].x + 2. < prev.items[0].x)
                || (level == 0
                    && !in_list
                    && (line.items[0].x - prev.items[0].x).abs() > line.size * 2.5)
        });
        if new_block {
            flush(&mut blocks, &mut paragraph, previous_level, in_list);
            in_list = false;
        }
        if is_list {
            in_list = true;
        }
        append_runs(&mut paragraph, current);
        previous = Some(line);
        previous_level = level;
    }
    flush(&mut blocks, &mut paragraph, previous_level, in_list);
    Some(blocks.join("\n\n"))
}

struct Table {
    top: f32,
    bottom: f32,
    markdown: String,
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
    let mut markdown = String::new();
    for row in 0..rows {
        markdown.push('|');
        for column in 0..columns {
            let value = lines(std::mem::take(&mut cells[row * columns + column]))
                .iter()
                .map(|line| markdown_cell(&runs(line)))
                .collect::<Vec<_>>()
                .join("<br>");
            markdown.push_str(&value);
            markdown.push('|');
        }
        markdown.push('\n');
        if row == 0 {
            markdown.push('|');
            for _ in 0..columns {
                markdown.push_str("---|");
            }
            markdown.push('\n');
        }
    }
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
    })
}

fn markdown_cell(runs: &[Run]) -> String {
    markdown(runs).replace('|', "\\|")
}

fn render(
    mut items: Vec<TextItem>,
    headings: &[f32],
    frame: Frame,
    grids: Vec<Grid>,
    baseline: &str,
) -> Option<String> {
    if items.len() > MAX_PAGE_ITEMS
        || items
            .iter()
            .any(|i| matches!(i.item_type, ItemType::Link(_) | ItemType::FormField))
    {
        return None;
    }
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
    if character_counts(&all_text) != character_counts(baseline) {
        return None;
    }
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
    let mut start = 0;
    let mut blocks = Vec::new();
    for table in tables {
        let end = start + all_lines[start..].partition_point(|line| line.y > table.top);
        if end > start {
            blocks.push(flow(&all_lines[start..end], headings)?);
        }
        // Any non-table text beside it is ambiguous multi-column layout.
        if all_lines
            .get(end)
            .is_some_and(|line| line.y >= table.bottom)
        {
            return None;
        }
        blocks.push(table.markdown);
        start = end;
    }
    if start < all_lines.len() {
        blocks.push(flow(&all_lines[start..], headings)?);
    }
    let output = blocks.join("\n\n");
    (!output.is_empty()).then_some(output)
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
        let resources = pdf.add_object(dictionary! {"Font"=>dictionary!{"F1"=>regular,"F2"=>bold}});
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
                &[],
                frame,
                vec![],
                "Different text from another decoder."
            )
            .is_none()
        );
        for (bad_x, bad_rotation) in [(f32::NAN, 0.), (-100., 0.), (40., 45.)] {
            let mut bad = items.clone();
            bad[0].x = bad_x;
            bad[0].rotation = bad_rotation;
            assert!(render(bad, &[], frame, vec![], &baseline).is_none());
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
        let mut layout = Layout::read(&bytes, &HashSet::from([1])).unwrap();
        let doc = lopdf::Document::load_mem(&bytes).unwrap();
        let id = doc.get_pages()[&1];
        let frame = super::super::geometry::frame(&doc, id).unwrap();
        let (_, content) = super::super::inspect_page(&doc, id);
        let grids = super::super::geometry::grids(
            &content.unwrap(),
            frame,
            &super::super::geometry::neutral_states(&doc, id),
        );
        let baseline = pdf_inspector::extract_pages_markdown_mem(&bytes, None).unwrap();
        assert!(
            layout
                .page(1, frame, grids, &baseline.pages[0].markdown)
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
        for row in 1..=54 {
            assert!(
                markdown.contains(&format!(
                    "|Row {row}|{}|long cell text long<br>cell text|",
                    row * row
                )),
                "row {row}: {markdown}"
            );
        }
        assert_eq!(
            markdown
                .matches("|Name|Square|Notes|\n|---|---|---|")
                .count(),
            3,
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
}
