//! Tables drawn without rules whose cells wrap.
//!
//! A browser centres (or tops) each cell's lines in its row, so the baselines
//! of one row interleave and no single baseline marks a row. Columns are the
//! left edges that the table's rows share; rows are told apart by their
//! spacing: one cell's lines follow at the table's line pitch, while cell
//! padding sets rows further apart. Only a table with that evidence is
//! reconstructed; anything else stays with the caller's paragraph flow, which
//! declines side-by-side text and leaves the page to the page reader.
use super::{Line, Run, heading_level, line_y, lines, markdown_cell, runs};
use pdf_inspector::TextItem;
use std::ops::Range;

/// Gap, in em, that separates two cells' text on a line: wider than any word
/// space, and the extractor merges runs closer than half an em.
const CELL_GAP: f32 = 1.;
/// Gap, in em, that a row starting a table must show at least once.
const GUTTER: f32 = 2.;
/// Rows starting a table are at most this far apart, in em.
const SEED_REACH: f32 = 10.;
/// A line further than this from the table, in em, does not extend it.
const EXTEND_REACH: f32 = 3.;
/// One cell's lines follow at most this fraction above the table's line
/// pitch, or a pixel when that is more.
const SAME_CELL: f32 = 0.06;
/// Rows follow at least this fraction above it, or two pixels.
const NEW_ROW: f32 = 0.12;
/// One CSS pixel, in points: a browser sets lines on whole pixels, so one
/// pitch can measure a pixel apart.
const PIXEL: f32 = 0.75;
/// Words from which a cell reads as running text.
const LONG_CELL: usize = 12;

/// A table found in a run of lines.
pub(super) struct Found {
    /// The lines it takes, in the slice it was found in.
    pub(super) lines: Range<usize>,
    pub(super) markdown: String,
    /// What a continuation on the next page keeps to.
    pub(super) shape: Shape,
    /// The bottom of its last line.
    pub(super) bottom: f32,
    /// Whether it continues the table that ended the previous page.
    pub(super) continued: bool,
}

/// A table's column left edges and line pitch (in em).
#[derive(Clone)]
pub(super) struct Shape {
    pub(super) columns: Vec<f32>,
    pub(super) pitch: f32,
}

/// The line pitch of the running text, in em: the median baseline distance
/// between consecutive long lines that share their left edge and size.
pub(super) fn prose_pitch(lines: &[Line]) -> Option<f32> {
    let unbroken = |line: &Line| {
        line.items.iter().all(|i| i.fixed_pitch != Some(true))
            && line
                .items
                .windows(2)
                .all(|w| w[1].x - (w[0].x + w[0].width) < line.size * CELL_GAP)
    };
    let mut pitches = Vec::new();
    for pair in lines.windows(2) {
        let (upper, lower) = (&pair[0], &pair[1]);
        let (Some(first), Some(last)) = (upper.items.first(), upper.items.last()) else {
            continue;
        };
        let pitch = (upper.y - lower.y) / upper.size;
        if (upper.size - lower.size).abs() <= 0.5
            && last.x + last.width - first.x >= upper.size * 20.
            && lower
                .items
                .first()
                .is_some_and(|i| (i.x - first.x).abs() <= 1.)
            && (1. ..=2.).contains(&pitch)
            && unbroken(upper)
            && unbroken(lower)
        {
            pitches.push(pitch);
        }
    }
    if pitches.len() < 3 {
        return None;
    }
    pitches.sort_by(f32::total_cmp);
    Some(pitches[pitches.len() / 2])
}

/// Borderless tables in `lines` (top to bottom), in order. `prose` is the
/// running text's line pitch, when known. `continued` is the shape of a
/// table that ended the previous page: rows at the top of this page that
/// keep to it continue that table.
pub(super) fn find(
    lines: &[Line],
    prose: Option<f32>,
    headings: &[f32],
    continued: Option<&Shape>,
) -> Vec<Found> {
    let mut found = Vec::new();
    let mut floor = 0;
    if let Some(shape) = continued
        && let Some(table) = continuation(lines, shape, headings)
    {
        floor = table.lines.end;
        found.push(table);
    }
    let seeds: Vec<usize> = (floor..lines.len())
        .filter(|&i| starts(&lines[i]).is_some())
        .collect();
    // Seed rows near each other form one candidate, processed top down.
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for &seed in &seeds {
        match groups.last_mut() {
            Some(group)
                if lines[group[group.len() - 1]].y - lines[seed].y
                    <= lines[seed].size * SEED_REACH =>
            {
                group.push(seed)
            }
            _ => groups.push(vec![seed]),
        }
    }
    groups.reverse();
    while let Some(group) = groups.pop() {
        if group.len() < 2 {
            continue;
        }
        let (first, last) = (group[0], group[group.len() - 1]);
        let Some(columns) = columns(lines, &group) else {
            continue;
        };
        // A line between the seed rows that keeps to no column (a paragraph
        // across them, a heading) separates two candidates.
        if let Some(bad) = (first..=last).find(|&i| {
            heading_level(&lines[i], headings) > 0 || fit(&lines[i], &columns, true).is_none()
        }) {
            let (above, below): (Vec<usize>, Vec<usize>) = group
                .into_iter()
                .filter(|&s| s != bad)
                .partition(|&s| s < bad);
            groups.push(below);
            groups.push(above);
            continue;
        }
        // Lines just above and below that keep to the columns: the rest of
        // a header's or a last row's wrapped cells. A line whose box overlaps
        // its neighbour's (the interleaved lines of centred cells) is in the
        // same row and may hold a run of two cells, as the rows between the
        // seeds may.
        let attached = |near: &Line, line: &Line| {
            let size = near.size.max(line.size);
            let distance = (near.y - line.y).abs();
            distance <= size * EXTEND_REACH
                && heading_level(line, headings) == 0
                && starts(line).is_none()
                && fit(line, &columns, distance < size).is_some()
        };
        let mut top = first;
        while top > floor && attached(&lines[top], &lines[top - 1]) {
            top -= 1;
        }
        let mut bottom = last + 1;
        while bottom < lines.len() && attached(&lines[bottom - 1], &lines[bottom]) {
            bottom += 1;
        }
        if let Some(table) = build(lines, top..bottom, &columns, Pitch::Measure(prose)) {
            floor = table.lines.end;
            found.push(table);
            // Seed rows already taken by this table start no other one.
            for group in &mut groups {
                group.retain(|&s| s >= floor);
            }
        }
    }
    found
}

/// The left edges of a line's cells when it can start a table: three or more
/// stretches of text at least a cell gap apart, one gap a clear gutter, no
/// fixed-pitch text (aligned code is not a table).
fn starts(line: &Line) -> Option<Vec<f32>> {
    let first = line.items.first()?;
    if line.items.iter().any(|i| i.fixed_pitch == Some(true)) {
        return None;
    }
    let mut lefts = vec![first.x];
    let mut end = first.x + first.width;
    let mut widest = 0f32;
    for item in &line.items[1..] {
        let gap = item.x - end;
        if gap >= line.size * CELL_GAP {
            lefts.push(item.x);
            widest = widest.max(gap);
        }
        end = end.max(item.x + item.width);
    }
    (lefts.len() >= 3 && widest >= line.size * GUTTER).then_some(lefts)
}

/// Clusters of left edges (with their lines) no wider than `tolerance`: each
/// cluster's leftmost edge, how many edges it holds and the lines they come
/// from.
fn clusters(lefts: &[(f32, usize)], tolerance: f32) -> Vec<(f32, usize, Vec<usize>)> {
    // Only the edges are sorted, as floats are elsewhere in the reader.
    let mut edges: Vec<f32> = lefts.iter().map(|(x, _)| *x).collect();
    edges.sort_by(f32::total_cmp);
    let mut clusters: Vec<(f32, usize, Vec<usize>)> = Vec::new();
    for x in edges {
        if clusters
            .last()
            .is_none_or(|(left, ..)| x - left > tolerance)
        {
            clusters.push((x, 0, Vec::new()));
        }
    }
    for &(x, line) in lefts {
        let index = clusters.partition_point(|(left, ..)| *left <= x) - 1;
        let (_, members, lines) = &mut clusters[index];
        *members += 1;
        if !lines.contains(&line) {
            lines.push(line);
        }
    }
    clusters
}

/// Column left edges of the candidate: the edges at which two or more of its
/// seed rows start a cell, nine in ten of their cells starting at one; and a
/// label column further left, where two or more of the lines between the
/// seed rows start (a wrapped label of a vertically centred row shares no
/// baseline with the row's other cells). Each seed row has three cells or
/// more, each at an edge of its own, so nine in ten of them start at three
/// shared edges or more.
fn columns(lines: &[Line], seeds: &[usize]) -> Option<Vec<f32>> {
    let tolerance = lines[seeds[0]].size * 0.25;
    let lefts: Vec<(f32, usize)> = seeds
        .iter()
        .flat_map(|&seed| {
            starts(&lines[seed])
                .unwrap_or_default()
                .into_iter()
                .map(move |x| (x, seed))
        })
        .collect();
    let count = lefts.len();
    let shared: Vec<(f32, usize, Vec<usize>)> = clusters(&lefts, tolerance)
        .into_iter()
        .filter(|(_, _, rows)| rows.len() >= 2)
        .collect();
    let aligned: usize = shared.iter().map(|(_, members, _)| members).sum();
    if aligned * 10 < count * 9 {
        return None;
    }
    let mut columns: Vec<f32> = shared.iter().map(|(left, _, _)| *left).collect();
    let first = *columns.first()?;
    let (top, bottom) = (seeds[0], seeds[seeds.len() - 1]);
    let labels: Vec<(f32, usize)> = (top..=bottom)
        .filter(|i| !seeds.contains(i))
        .filter_map(|i| lines[i].items.first().map(|item| (item.x, i)))
        .filter(|(x, _)| *x < first - tolerance)
        .collect();
    if let Some((left, _, rows)) = clusters(&labels, tolerance)
        .into_iter()
        .max_by_key(|(_, members, _)| *members)
        && rows.len() >= 2
        // A label keeps clear of the first seed column.
        && (top..=bottom).all(|i| {
            lines[i]
                .items
                .iter()
                .filter(|item| item.x < first - tolerance)
                .all(|item| item.x + item.width <= first - item.font_size * CELL_GAP)
        })
    {
        columns.insert(0, left);
    }
    Some(columns)
}

/// The cells of a line's items: the column whose left edge each starts at or
/// after. A run crossing into the next column only is split at the space
/// nearest that column's edge when `split` allows (the extractor merges
/// neighbouring cells' words when their gap is narrow); any other crossing
/// text does not keep to the columns.
fn fit(line: &Line, columns: &[f32], split: bool) -> Option<Vec<(usize, TextItem)>> {
    let mut parts = Vec::new();
    for item in &line.items {
        if item.fixed_pitch == Some(true) {
            return None;
        }
        let tolerance = item.font_size * 0.25;
        let column = columns
            .iter()
            .rposition(|&left| item.x >= left - tolerance)?;
        let end = item.x + item.width;
        match columns.get(column + 1) {
            Some(&next) if end > next + 0.5 => {
                if !split
                    || columns
                        .get(column + 2)
                        .is_some_and(|&after| end > after + 0.5)
                {
                    return None;
                }
                let (left, right) = split_at_column(item, next)?;
                parts.push((column, left));
                parts.push((column + 1, right));
            }
            _ => parts.push((column, item.clone())),
        }
    }
    Some(parts)
}

/// `item` split at the space after which its text is estimated to reach
/// `edge`, its glyphs taken as equally wide; within an em of it.
fn split_at_column(item: &TextItem, edge: f32) -> Option<(TextItem, TextItem)> {
    let count = item.text.chars().count() as f32;
    let (distance, at) = item
        .text
        .char_indices()
        .enumerate()
        .filter(|(_, (_, c))| *c == ' ')
        .map(|(index, (at, _))| {
            let estimate = item.x + item.width * (index + 1) as f32 / count;
            ((estimate - edge).abs(), at)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))?;
    let (before, after) = (item.text[..at].trim_end(), item.text[at + 1..].trim_start());
    if distance > item.font_size || before.is_empty() || after.is_empty() {
        return None;
    }
    let end = item.x + item.width;
    let mut left = item.clone();
    left.text = before.into();
    left.width = item.width * before.chars().count() as f32 / count;
    let mut right = item.clone();
    right.text = after.into();
    right.x = edge;
    right.width = (end - edge).max(0.);
    Some((left, right))
}

/// One line of a cell: the parts of one page line in one column, measured on
/// its anchor text rather than on raised or lowered scripts.
struct CellLine {
    baseline: f32,
    size: f32,
    line: usize,
    items: Vec<TextItem>,
}

impl CellLine {
    fn new(line: usize, items: Vec<TextItem>) -> Self {
        let anchored = items.iter().any(|i| i.baseline_shift == 0.);
        let (baseline, size) = items
            .iter()
            .filter(|i| !anchored || i.baseline_shift == 0.)
            .fold((f32::MIN, 0f32), |(baseline, size), i| {
                (baseline.max(line_y(i)), size.max(i.font_size))
            });
        Self {
            baseline,
            size,
            line,
            items,
        }
    }
}

struct Cell {
    column: usize,
    top: f32,
    bottom: f32,
    lines: Vec<CellLine>,
}

/// Where a table's line pitch comes from.
enum Pitch {
    /// Measured in the table, and confirmed by a vertically centred line of
    /// another column between two of a cell's lines or by the running
    /// text's pitch (given, when known).
    Measure(Option<f32>),
    /// The pitch of the table this one continues.
    Continue(f32),
}

/// The largest step, in em at `size`, within `margin` (a fraction) of
/// `pitch` or within `pixels` of it when that is more.
fn reach(pitch: f32, size: f32, margin: f32, pixels: f32) -> f32 {
    pitch + (pitch * margin).max(pixels * PIXEL / size)
}

/// The line pitch, in em, of the cells in `by_column`: the smallest step
/// between consecutive lines of a column, when it is confirmed.
fn measure(by_column: &[Vec<CellLine>], prose: Option<f32>) -> Option<f32> {
    let steps = || {
        by_column.iter().enumerate().flat_map(|(column, lines)| {
            lines.windows(2).map(move |pair| {
                let size = pair[0].size.max(pair[1].size);
                let step = (pair[0].baseline - pair[1].baseline) / size;
                (step, size, column, &pair[0], &pair[1])
            })
        })
    };
    let (pitch, size, ..) = steps().min_by(|a, b| a.0.total_cmp(&b.0))?;
    if prose.is_some_and(|prose| reach(prose.min(pitch), size, SAME_CELL, 1.) >= prose.max(pitch)) {
        return Some(pitch);
    }
    let centred = steps().any(|(step, size, column, upper, lower)| {
        let middle = (upper.baseline + lower.baseline) / 2.;
        step <= reach(pitch, size, SAME_CELL, 1.)
            && by_column.iter().enumerate().any(|(other, lines)| {
                other != column
                    && lines
                        .iter()
                        .any(|l| (l.baseline - middle).abs() <= l.size.max(upper.size) * 0.15)
            })
    });
    centred.then_some(pitch)
}

/// The table in `range`, if its rows and cells hold the evidence.
fn build(lines: &[Line], range: Range<usize>, columns: &[f32], pitch: Pitch) -> Option<Found> {
    let mut by_column: Vec<Vec<(usize, Vec<TextItem>)>> =
        (0..columns.len()).map(|_| Vec::new()).collect();
    for index in range {
        for (column, item) in fit(&lines[index], columns, true)? {
            let column = &mut by_column[column];
            match column.last_mut() {
                Some((line, items)) if *line == index => items.push(item),
                _ => column.push((index, vec![item])),
            }
        }
    }
    let by_column: Vec<Vec<CellLine>> = by_column
        .into_iter()
        .map(|lines| {
            lines
                .into_iter()
                .map(|(line, items)| CellLine::new(line, items))
                .collect()
        })
        .collect();
    let continued = matches!(pitch, Pitch::Continue(_));
    let pitch = match pitch {
        Pitch::Measure(prose) => measure(&by_column, prose)?,
        Pitch::Continue(pitch) => pitch,
    };
    // A cell's lines continue at the table's pitch; a clearly wider step is a
    // new row. A step between the two is no evidence either way.
    let mut cells: Vec<Vec<Cell>> = Vec::with_capacity(columns.len());
    for (column, cell_lines) in by_column.into_iter().enumerate() {
        let mut ours = Vec::new();
        let mut current: Option<Cell> = None;
        for cell_line in cell_lines {
            if let Some(cell) = &mut current {
                let previous = &cell.lines[cell.lines.len() - 1];
                let size = previous.size.max(cell_line.size);
                let step = (previous.baseline - cell_line.baseline) / size;
                if step <= reach(pitch, size, SAME_CELL, 1.) {
                    cell.bottom = cell_line.baseline - cell_line.size * 0.2;
                    cell.lines.push(cell_line);
                    continue;
                }
                if step < reach(pitch, size, NEW_ROW, 2.) {
                    return None;
                }
                ours.extend(current.take());
            }
            current = Some(Cell {
                column,
                top: cell_line.baseline + cell_line.size * 0.8,
                bottom: cell_line.baseline - cell_line.size * 0.2,
                lines: vec![cell_line],
            });
        }
        ours.extend(current);
        // Bottom first, so the top cell is the last.
        ours.reverse();
        cells.push(ours);
    }
    // Rows: cells whose text overlaps vertically, taken from the top (the
    // highest of the columns' top cells, the leftmost of equals). Two cells
    // of one column in a row would be a spanning cell or a misread pitch.
    let mut rows: Vec<(f32, Vec<Option<Cell>>)> = Vec::new();
    loop {
        let mut highest: Option<(usize, f32)> = None;
        for (column, ours) in cells.iter().enumerate() {
            if let Some(cell) = ours.last()
                && highest.is_none_or(|(_, top)| cell.top > top)
            {
                highest = Some((column, cell.top));
            }
        }
        let Some(cell) = highest.and_then(|(column, _)| cells[column].pop()) else {
            break;
        };
        match rows.last_mut() {
            Some((bottom, row)) if cell.top > *bottom => {
                *bottom = bottom.min(cell.bottom);
                let column = cell.column;
                if row[column].replace(cell).is_some() {
                    return None;
                }
            }
            _ => {
                let mut row: Vec<Option<Cell>> = (0..columns.len()).map(|_| None).collect();
                let (column, bottom) = (cell.column, cell.bottom);
                row[column] = Some(cell);
                rows.push((bottom, row));
            }
        }
    }
    let filled = |row: &[Option<Cell>]| row.iter().flatten().count();
    // A line of one cell just above or below belongs to the text around; a
    // continuation starts at the top of its page.
    while rows.last().is_some_and(|(_, row)| filled(row) < 2) {
        rows.pop();
    }
    while !continued && rows.first().is_some_and(|(_, row)| filled(row) < 2) {
        rows.remove(0);
    }
    let cells: Vec<&Cell> = rows
        .iter()
        .flat_map(|(_, row)| row.iter().flatten())
        .collect();
    let words = |cell: &Cell| {
        cell.lines
            .iter()
            .flat_map(|l| &l.items)
            .map(|i| i.text.split_whitespace().count())
            .sum::<usize>()
    };
    let long = cells.iter().filter(|cell| words(cell) > LONG_CELL).count();
    // Every column keeps a cell: a seed row has three or more, and the rows
    // left are never trimmed seed rows.
    if rows.len() < if continued { 1 } else { 3 }
        || rows.iter().any(|(_, row)| filled(row) < 2)
        || long * 3 > cells.len()
        || (!continued
            && (rows.iter().filter(|(_, row)| filled(row) >= 3).count() < 2
                || !cells.iter().any(|cell| cell.lines.len() > 1)))
    {
        return None;
    }
    let lines = || cells.iter().flat_map(|c| &c.lines).map(|l| l.line);
    let (start, end) = (lines().min()?, lines().max()? + 1);
    let bottom = cells
        .iter()
        .flat_map(|c| &c.lines)
        .map(|l| l.baseline - l.size * 0.2)
        .fold(f32::INFINITY, f32::min);
    // A continuation's header stayed on the previous page; standing alone,
    // its first row heads it, as for any table. Joined to the table it
    // continues, the first row is a body row again.
    let mut markdown = String::new();
    for (index, (_, row)) in rows.into_iter().enumerate() {
        markdown.push('|');
        for cell in row {
            if let Some(cell) = cell {
                markdown.push_str(&cell_markdown(cell));
            }
            markdown.push('|');
        }
        markdown.push('\n');
        if index == 0 {
            markdown.push('|');
            markdown.push_str(&"---|".repeat(columns.len()));
            markdown.push('\n');
        }
    }
    Some(Found {
        lines: start..end,
        markdown: markdown.trim_end().into(),
        shape: Shape {
            columns: columns.to_vec(),
            pitch,
        },
        bottom,
        continued,
    })
}

/// Rows at the top of a page that keep to the shape of the table ending the
/// previous page, with no header of their own.
fn continuation(lines: &[Line], shape: &Shape, headings: &[f32]) -> Option<Found> {
    let mut end = 0;
    while end < lines.len()
        && heading_level(&lines[end], headings) == 0
        && fit(&lines[end], &shape.columns, true).is_some()
        && (end == 0
            || lines[end - 1].y - lines[end].y
                <= lines[end - 1].size.max(lines[end].size) * EXTEND_REACH)
    {
        end += 1;
    }
    // Its rows are not trimmed at the top: one of a single cell fails it.
    build(lines, 0..end, &shape.columns, Pitch::Continue(shape.pitch))
}

/// A cell's text: its lines rejoined, as the column width wrapped them. A
/// line ending in a hyphen after a letter joins the next without a space.
fn cell_markdown(cell: Cell) -> String {
    let items: Vec<TextItem> = cell.lines.into_iter().flat_map(|l| l.items).collect();
    let mut output: Vec<Run> = Vec::new();
    for line in lines(items) {
        let mut current = runs(&line).into_iter();
        let Some(first) = current.next() else {
            continue;
        };
        let hyphenated = output.last().is_some_and(|last| {
            let mut tail = last.text.chars().rev();
            tail.next() == Some('-') && tail.next().is_some_and(char::is_alphabetic)
        }) && first.text.starts_with(char::is_alphabetic);
        match output.last_mut() {
            Some(last) => {
                if !hyphenated && !last.text.ends_with(char::is_whitespace) {
                    last.text.push(' ');
                }
                if last.joins(first.style, first.link.as_ref()) {
                    last.text.push_str(&first.text);
                } else {
                    output.push(first);
                }
            }
            None => output.push(first),
        }
        output.extend(current);
    }
    markdown_cell(&output)
}
