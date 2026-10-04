//! Choose an inert carrier's origin inside existing visible literal data.
//! This is not an estimate of Calc's print origin: a chart may start earlier.
use super::{Result, check_deadline, failure, xml};
use std::{collections::BTreeSet, time::Instant};
const NS: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Anchor {
    pub col: u32,
    pub row: u32,
}
fn coordinate(value: &str) -> Result<Anchor> {
    let split = value.bytes().take_while(u8::is_ascii_uppercase).count();
    if split == 0 || split > 3 {
        return Err(failure("invalid worksheet cell coordinate"));
    }
    let mut col = 0u32;
    for b in value[..split].bytes() {
        col = col * 26 + u32::from(b - b'A' + 1);
    }
    let digits = &value[split..];
    if digits.starts_with('0') || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(failure("invalid worksheet cell coordinate"));
    }
    let row = digits
        .parse::<u32>()
        .map_err(|_| failure("invalid worksheet row"))?;
    if !(1..=16384).contains(&col) || !(1..=1048576).contains(&row) {
        return Err(failure("worksheet coordinate exceeds grid"));
    }
    Ok(Anchor {
        col: col - 1,
        row: row - 1,
    })
}
fn boolean(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0" | "false") => Ok(false),
        Some("1" | "true") => Ok(true),
        _ => Err(failure("invalid worksheet layout boolean")),
    }
}
fn number(value: Option<&str>, default: f64) -> Result<f64> {
    let v = value
        .map_or(Ok(default), str::parse::<f64>)
        .map_err(|_| failure("invalid worksheet layout measurement"))?;
    if !v.is_finite() || v < 0. {
        return Err(failure("invalid worksheet layout measurement"));
    }
    Ok(v)
}
fn children(nodes: &[xml::Node], i: usize) -> impl Iterator<Item = &xml::Node> {
    nodes[i + 1..]
        .iter()
        .take_while(move |n| n.start < nodes[i].close_start)
}
fn scalar<'a>(bytes: &'a [u8], n: &xml::Node) -> Result<std::borrow::Cow<'a, str>> {
    let raw = bytes
        .get(n.open_end..n.close_start)
        .ok_or_else(|| failure("invalid XML text range"))?;
    if raw.len() > 4096 || raw.contains(&b'<') {
        return Err(failure("unsupported worksheet scalar"));
    }
    quick_xml::escape::unescape(
        std::str::from_utf8(raw).map_err(|_| failure("invalid worksheet text"))?,
    )
    .map_err(|_| failure("invalid worksheet text reference"))
}
fn text_present(bytes: &[u8], n: &xml::Node, deadline: Instant) -> Result<bool> {
    use quick_xml::events::Event;
    let mut reader = quick_xml::Reader::from_reader(&bytes[n.open_end..n.close_start]);
    let mut present = false;
    loop {
        check_deadline(deadline)?;
        match reader
            .read_event()
            .map_err(|_| failure("invalid worksheet text"))?
        {
            Event::Text(t) => {
                let t = t.decode().map_err(|_| failure("invalid worksheet text"))?;
                present |= !t.trim().is_empty();
            }
            Event::CData(t) => {
                present |= !t
                    .decode()
                    .map_err(|_| failure("invalid worksheet text"))?
                    .trim()
                    .is_empty()
            }
            Event::GeneralRef(t) => {
                let t = t
                    .decode()
                    .map_err(|_| failure("invalid worksheet text reference"))?;
                let reference = format!("&{t};");
                present |= !quick_xml::escape::unescape(&reference)
                    .map_err(|_| failure("invalid worksheet text reference"))?
                    .trim()
                    .is_empty();
            }
            Event::Eof => break,
            _ => return Err(failure("unsupported nested worksheet text")),
        }
    }
    Ok(present)
}
fn rich_present(nodes: &[xml::Node], i: usize, bytes: &[u8], deadline: Instant) -> Result<bool> {
    // Only standard direct text and rich runs count; phonetic annotations do not.
    let parent = &nodes[i];
    let mut run_end = 0;
    let mut present = false;
    for n in children(nodes, i) {
        if n.namespace != NS {
            continue;
        }
        if n.depth == parent.depth + 1 && n.local == "r" {
            run_end = n.close_start;
        }
        if n.local == "t"
            && (n.depth == parent.depth + 1 || (n.depth == parent.depth + 2 && n.start < run_end))
        {
            present |= text_present(bytes, n, deadline)?;
        }
    }
    Ok(present)
}
pub(super) fn shared_nonempty(bytes: &[u8], deadline: Instant) -> Result<Vec<bool>> {
    let nodes = xml::index(bytes, deadline)?;
    if nodes[0].namespace != NS || nodes[0].local != "sst" {
        return Err(failure("invalid shared string root"));
    }
    let mut result = Vec::new();
    for (i, n) in nodes.iter().enumerate() {
        check_deadline(deadline)?;
        if n.namespace == NS && n.local == "si" && n.depth == 1 {
            result.push(rich_present(&nodes, i, bytes, deadline)?);
        }
    }
    Ok(result)
}
fn container<'a>(
    nodes: &'a [xml::Node],
    name: &str,
    deadline: Instant,
) -> Result<Option<&'a xml::Node>> {
    let mut found = None;
    for n in nodes {
        check_deadline(deadline)?;
        if n.namespace == NS && n.depth == 1 && n.local == name {
            if found.is_some() {
                return Err(failure("duplicate worksheet layout container"));
            }
            found = Some(n);
        }
    }
    Ok(found)
}
fn child_of(n: &xml::Node, parent: Option<&xml::Node>) -> bool {
    parent
        .is_some_and(|p| n.depth == p.depth + 1 && n.start >= p.open_end && n.end <= p.close_start)
}
// The grid has 16,384 columns. Row events and a Fenwick difference tree bound
// merge qualification to O((cells + merges) log columns), including overlaps.
// A merge's top-left is excluded from its covered region, not from other merges.
fn uncovered(
    mut candidates: Vec<Anchor>,
    merges: &[(Anchor, Anchor)],
    deadline: Instant,
) -> Result<Vec<Anchor>> {
    const COLUMNS: usize = 16384;
    let mut events = Vec::with_capacity(merges.len() * 4);
    for &(lo, hi) in merges {
        check_deadline(deadline)?;
        events.push((lo.row, lo.col, hi.col, 1i32));
        events.push((hi.row + 1, lo.col, hi.col, -1));
        events.push((lo.row, lo.col, lo.col, -1));
        events.push((lo.row + 1, lo.col, lo.col, 1));
    }
    events.sort_unstable_by_key(|e| e.0);
    candidates.sort_unstable_by_key(|a| (a.row, a.col));
    let mut tree = vec![0i32; COLUMNS + 2];
    let mut next = 0;
    let mut result = Vec::new();
    for cell in candidates {
        check_deadline(deadline)?;
        while let Some(&(row, lo, hi, delta)) = events.get(next) {
            if row > cell.row {
                break;
            }
            check_deadline(deadline)?;
            for (index, change) in [(lo as usize + 1, delta), (hi as usize + 2, -delta)] {
                let mut i = index;
                while i < tree.len() {
                    tree[i] += change;
                    i += 1usize << i.trailing_zeros();
                }
            }
            next += 1;
        }
        let mut count = 0;
        let mut i = cell.col as usize + 1;
        while i != 0 {
            count += tree[i];
            i &= i - 1;
        }
        if count == 0 {
            result.push(cell);
        }
    }
    Ok(result)
}
pub(super) fn select(
    nodes: &[xml::Node],
    bytes: &[u8],
    shared: &[bool],
    deadline: Instant,
) -> Result<Anchor> {
    if nodes[0].namespace != NS || nodes[0].local != "worksheet" {
        return Err(failure(
            "only transitional OOXML worksheets support overflow repair",
        ));
    }
    let data = container(nodes, "sheetData", deadline)?;
    let cols = container(nodes, "cols", deadline)?;
    let merged = container(nodes, "mergeCells", deadline)?;
    let views = container(nodes, "sheetViews", deadline)?;
    let format = container(nodes, "sheetFormatPr", deadline)?;
    let mut default_height = 15.;
    let mut default_width = 8.;
    let mut default_hidden = false;
    let mut columns = Vec::new();
    let mut merges = Vec::new();
    for n in nodes.iter().filter(|n| n.namespace == NS) {
        check_deadline(deadline)?;
        match n.local.as_str() {
            "sheetView" if child_of(n, views) && boolean(n.attr("rightToLeft"))? => {
                return Err(failure("RTL worksheet overflow mapping is unsupported"));
            }
            "sheetFormatPr" if format.is_some_and(|f| f.start == n.start) => {
                default_height = number(n.attr("defaultRowHeight"), 15.)?;
                default_width = number(n.attr("defaultColWidth").or(n.attr("baseColWidth")), 8.)?;
                default_hidden = boolean(n.attr("zeroHeight"))?;
            }
            "col" if child_of(n, cols) => {
                let lo = n
                    .attr("min")
                    .and_then(|s| s.parse::<u32>().ok())
                    .filter(|v| (1..=16384).contains(v))
                    .ok_or_else(|| failure("invalid worksheet column range"))?;
                let hi = n
                    .attr("max")
                    .and_then(|s| s.parse::<u32>().ok())
                    .filter(|v| (lo..=16384).contains(v))
                    .ok_or_else(|| failure("invalid worksheet column range"))?;
                columns.push((
                    lo - 1,
                    hi - 1,
                    boolean(n.attr("hidden"))?,
                    n.attr("width").map(str::to_owned),
                ));
            }
            "mergeCell" if child_of(n, merged) => {
                let (a, b) = n
                    .attr("ref")
                    .and_then(|s| s.split_once(':'))
                    .ok_or_else(|| failure("invalid worksheet merge range"))?;
                let (a, b) = (coordinate(a)?, coordinate(b)?);
                if a.col > b.col || a.row > b.row {
                    return Err(failure("invalid worksheet merge range"));
                }
                if merges.len() == 16_384 {
                    return Err(failure(
                        "worksheet merge qualification exceeds range budget",
                    ));
                }
                merges.push((a, b));
            }
            _ => {}
        }
    }
    columns.sort_by_key(|r| r.0);
    if columns.windows(2).any(|v| v[0].1 >= v[1].0) {
        return Err(failure("overlapping worksheet column definitions"));
    }
    // A grid-sized mask bounds lookup cost independently of cell and range count.
    let mut visible_columns = vec![default_width >= 1.; 16384];
    for (lo, hi, hidden, width) in columns {
        let visible = !hidden && number(width.as_deref(), default_width)? >= 1.;
        visible_columns[lo as usize..=hi as usize].fill(visible);
    }
    let mut row = None;
    let mut seen_rows = BTreeSet::new();
    let mut seen_cells = BTreeSet::new();
    let mut candidates = Vec::new();
    for (i, n) in nodes.iter().enumerate().filter(|(_, n)| n.namespace == NS) {
        check_deadline(deadline)?;
        if n.local == "row" && child_of(n, data) {
            let r = n
                .attr("r")
                .and_then(|s| s.parse::<u32>().ok())
                .filter(|v| (1..=1048576).contains(v))
                .ok_or_else(|| failure("worksheet row has no explicit valid index"))?;
            if !seen_rows.insert(r) {
                return Err(failure("duplicate worksheet row"));
            }
            let hidden = n
                .attr("hidden")
                .map_or(Ok(default_hidden), |v| boolean(Some(v)))?;
            row = Some((
                r - 1,
                !hidden && number(n.attr("ht"), default_height)? >= 2.,
                n.close_start,
            ));
        }
        if n.local != "c"
            || n.depth != 3
            || !data.is_some_and(|d| n.start >= d.open_end && n.end <= d.close_start)
        {
            continue;
        }
        let cell = coordinate(
            n.attr("r")
                .ok_or_else(|| failure("worksheet cell has no explicit coordinate"))?,
        )?;
        if !seen_cells.insert((cell.row, cell.col)) {
            return Err(failure("duplicate worksheet cell"));
        }
        let Some((r, visible, end)) = row else {
            return Err(failure("worksheet cell has no row"));
        };
        if r != cell.row || n.end > end {
            return Err(failure("worksheet row and cell disagree"));
        }
        if !visible || !visible_columns[cell.col as usize] {
            continue;
        }
        if children(nodes, i).any(|v| v.namespace == NS && v.local == "f" && v.depth == n.depth + 1)
        {
            continue;
        }
        let kind = n.attr("t").unwrap_or("n");
        let mut occupied = false;
        if kind == "inlineStr" {
            for (j, v) in nodes[i + 1..]
                .iter()
                .enumerate()
                .take_while(|(_, v)| v.start < n.close_start)
            {
                if v.namespace == NS && v.local == "is" && v.depth == n.depth + 1 {
                    occupied |= rich_present(nodes, i + 1 + j, bytes, deadline)?;
                }
            }
        } else {
            let mut values = children(nodes, i)
                .filter(|v| v.namespace == NS && v.local == "v" && v.depth == n.depth + 1);
            if let Some(v) = values.next() {
                if values.next().is_some() {
                    return Err(failure("duplicate worksheet cell value"));
                }
                let value = scalar(bytes, v)?;
                let value = value.trim();
                occupied = match kind {
                    "s" => *shared
                        .get(
                            value
                                .parse::<usize>()
                                .map_err(|_| failure("invalid shared string index"))?,
                        )
                        .ok_or_else(|| failure("missing shared string index"))?,
                    "n" => value.parse::<f64>().is_ok_and(f64::is_finite),
                    "b" => matches!(value, "0" | "1"),
                    "e" => matches!(
                        value,
                        "#NULL!" | "#DIV/0!" | "#VALUE!" | "#REF!" | "#NAME?" | "#NUM!" | "#N/A"
                    ),
                    _ => false,
                };
            }
        }
        if occupied {
            candidates.push(cell);
        }
    }
    let candidates = uncovered(candidates, &merges, deadline)?;
    let a = candidates.into_iter().reduce(|a,b| Anchor {col:a.col.min(b.col),row:a.row.min(b.row)})
        .ok_or_else(|| failure("worksheet has no safe visible literal data anchor (formula-only and unsupported values do not qualify)"))?;
    if merges.iter().any(|(lo, hi)| {
        a.col >= lo.col && a.col <= hi.col && a.row >= lo.row && a.row <= hi.row && a != *lo
    }) {
        return Err(failure("worksheet data anchor is covered by a merged cell"));
    }
    Ok(a)
}
