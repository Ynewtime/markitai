//! In-house SpreadsheetML reader (.xlsx / .xlsm): the workbook's visible
//! sheets, shared strings, cell number formats from `xl/styles.xml`, and
//! merge regions. Rows, columns, and sheets the source hides are omitted,
//! and merge regions are remapped onto the surviving grid.

use super::controls::{Checkboxes, cell_inlines, read_vml_checkboxes};
use super::notes;
use super::numfmt::{DateParts, DatePiece, NumberFormat, Rendered, builtin_code};
use super::{format_duration_days, format_float, format_time_of_day};
use crate::error::ConvertError;
use crate::model::{
    Block, Cell, Document, GridBuilder, Inline, LinkTarget, List, ListItem, MarkerKind, Style,
    Table, TableKind,
};
use crate::package::limits;
use crate::package::relationships::{Relationships, read_rels, rel_type, rels_part_for};
use crate::package::xml::{Element, ns};
use crate::package::{Package, path};
use crate::shared::assets::AssetSink;
use crate::shared::header::resolve_header_rows;
use crate::shared::text::{clean_cell_text, clean_text};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

pub(super) const SHARED_STRINGS_REL: &str =
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings";

/// The grid bounds the format defines; a reference outside them is not a
/// real cell.
pub(super) const MAX_ROWS: u32 = 1_048_576;
pub(super) const MAX_COLS: u32 = 16_384;

pub(super) fn parse(pkg: &mut Package, wb_part: &str) -> Result<Document, ConvertError> {
    let workbook = pkg.required_xml_part(wb_part)?;
    // Any OOXML main part resolves here; a document or presentation would
    // otherwise convert to an empty workbook.
    if !workbook.child_elems().next().is_some_and(|e| e.is(ns::SML, "workbook")) {
        return Err(ConvertError::malformed("main part is not a workbook"));
    }
    let wb_rels = read_rels(pkg, &rels_part_for(wb_part))?;
    let date1904 = workbook
        .first_descendant(ns::SML, "workbookPr")
        .and_then(|e| e.attr_unqualified("date1904"))
        .is_some_and(bool_attr);

    let shared =
        match sibling_part(pkg, &wb_rels, wb_part, SHARED_STRINGS_REL, "sharedStrings.xml")? {
            Some(root) => shared_strings(&root),
            None => Vec::new(),
        };
    let styles =
        Styles::read(sibling_part(pkg, &wb_rels, wb_part, rel_type::STYLES, "styles.xml")?);

    // Visible sheets in workbook order; hidden and veryHidden sheets are
    // omitted entirely, heading included.
    let mut sheets: Vec<(String, String)> = Vec::new();
    for sheet in workbook
        .first_descendant(ns::SML, "sheets")
        .into_iter()
        .flat_map(|s| s.find_all(ns::SML, "sheet"))
    {
        if matches!(sheet.attr_unqualified("state"), Some("hidden" | "veryHidden")) {
            continue;
        }
        let name = sheet.attr_unqualified("name").unwrap_or_default().to_string();
        let Some(target) =
            sheet.attr_qualified(ns::R, "id").and_then(|rid| wb_rels.internal_target(rid))
        else {
            log::warn!("skipping sheet {name:?} with no worksheet relationship");
            continue;
        };
        match path::resolve(wb_part, target) {
            Ok(t) => sheets.push((name, t.path)),
            Err(e) => log::warn!("skipping sheet {name:?} with unresolvable target: {e}"),
        }
    }

    let multi_sheet = sheets.len() > 1;
    let mut doc = Document::default();
    let mut failed = 0usize;
    // One budget for the workbook, so sheets cannot multiply the cap.
    let mut slots = 0u64;
    // markitai: formulas whose cell carries no cached value, workbook-wide.
    let mut uncached = 0usize;
    let mut people = None;
    let mut assets = AssetSink::new();
    for (name, part) in &sheets {
        let worksheet = pkg.optional_xml_part(part)?;
        let Some(worksheet) = worksheet.as_ref().and_then(|r| r.find(ns::SML, "worksheet")) else {
            log::warn!("skipping unreadable sheet {name:?}");
            failed += 1;
            continue;
        };
        let mut content = read_sheet(worksheet, &shared, &styles, date1904);
        content.checkboxes = read_vml_checkboxes(pkg, part)?;
        // markitai: cell hyperlinks and notes.
        let sheet_rels = read_rels(pkg, &rels_part_for(part))?;
        content.links = notes::hyperlinks(worksheet, &sheet_rels);
        let people = match &people {
            Some(people) => people,
            None => people.insert(notes::people(pkg, &wb_rels, wb_part)?),
        };
        content.notes = notes::cell_notes(pkg, part, &sheet_rels, people)?;
        uncached += content.uncached_formulas;
        let pictures = super::drawings::xml(
            pkg,
            worksheet,
            part,
            &sheet_rels,
            &content,
            &mut assets,
            &mut doc.warnings,
        )?;
        push_sheet_images(&mut doc, name, multi_sheet, build_table(content, &mut slots)?, pictures);
    }
    if !sheets.is_empty() && failed == sheets.len() {
        return Err(ConvertError::malformed("no sheet in the workbook could be read"));
    }
    if uncached > 0 {
        doc.warnings.push(if uncached == 1 {
            "1 formula in the workbook has no cached value, so its formula text is shown; open and save the workbook in Excel or LibreOffice to compute it.".to_string()
        } else {
            format!("{uncached} formulas in the workbook have no cached value, so their formula text is shown; open and save the workbook in Excel or LibreOffice to compute them.")
        });
    }
    doc.assets = assets.assets;
    Ok(doc)
}

/// markitai: a built sheet: its grid, the notes of its visible cells, and how
/// many hidden rows and columns held content.
#[derive(Default)]
pub(super) struct Built {
    pub(super) table: Option<Table>,
    pub(super) notes: Vec<(u32, u32, String)>,
    pub(super) hidden_rows: usize,
    pub(super) hidden_cols: usize,
}

/// markitai: append a built sheet to the workbook document, as every
/// container does: the sheet's name heading when the workbook shows several,
/// its table, then its cell notes as a list under a `Notes` heading, each
/// item led by the cell's reference (`B3: text`). Hidden rows and columns
/// that held content are omitted as hidden sheets are, with a warning naming
/// how many.
pub(super) fn push_sheet(doc: &mut Document, name: &str, multi_sheet: bool, built: Built) {
    push_sheet_images(doc, name, multi_sheet, built, Vec::new());
}

pub(super) fn push_sheet_images(
    doc: &mut Document,
    name: &str,
    multi_sheet: bool,
    built: Built,
    pictures: Vec<Inline>,
) {
    let Built { table, notes, hidden_rows, hidden_cols } = built;
    if hidden_rows + hidden_cols > 0 {
        let count = |n: usize, what: &str| match n {
            0 => None,
            1 => Some(format!("1 hidden {what}")),
            n => Some(format!("{n} hidden {what}s")),
        };
        let parts: Vec<String> = [count(hidden_rows, "row"), count(hidden_cols, "column")]
            .into_iter()
            .flatten()
            .collect();
        let pronoun = if hidden_rows + hidden_cols == 1 { "it is" } else { "they are" };
        doc.warnings.push(format!(
            "Worksheet {name:?} has {} holding content; {pronoun} omitted by the native spreadsheet reader.",
            parts.join(" and ")
        ));
    }
    if table.is_none() && notes.is_empty() && pictures.is_empty() {
        return;
    }
    if multi_sheet {
        doc.blocks.push(Block::heading(2, vec![Inline::plain(name.to_string())]));
    }
    if let Some(table) = table {
        doc.blocks.push(Block::Table(table));
    }
    for image in pictures {
        doc.blocks.push(Block::Paragraph(vec![image]));
    }
    if !notes.is_empty() {
        doc.blocks.push(Block::heading(3, vec![Inline::plain("Notes")]));
        let items = notes
            .into_iter()
            .map(|(row, col, text)| ListItem {
                blocks: vec![Block::Paragraph(vec![Inline::plain(format!(
                    "{}: {text}",
                    cell_reference(row, col)
                ))])],
                marker_label: None,
            })
            .collect();
        doc.blocks.push(Block::List(List { marker: MarkerKind::Bullet, start: 1, items }));
    }
}

/// markitai: a zero-based (row, column) as an A1 reference (`B3`).
pub(super) fn cell_reference(row: u32, col: u32) -> String {
    let mut letters = Vec::new();
    let mut n = col + 1;
    while n > 0 {
        let rem = (n - 1) % 26;
        letters.push(char::from(b'A' + rem as u8));
        n = (n - 1) / 26;
    }
    letters.iter().rev().collect::<String>() + &(row + 1).to_string()
}

/// Part name for a workbook-level sibling: the relationship of the given
/// type when present, else the conventional name next to the workbook part.
pub(super) fn sibling_part_name(
    rels: &Relationships,
    base: &str,
    rel: &str,
    conventional: &str,
) -> String {
    rels.first_of_type(rel)
        .and_then(|r| path::resolve(base, &r.target).ok())
        .map(|t| t.path)
        .unwrap_or_else(|| match base.rsplit_once('/') {
            Some((dir, _)) => format!("{dir}/{conventional}"),
            None => conventional.to_string(),
        })
}

/// Load a workbook-level XML part by relationship type, falling back to the
/// conventional name next to the workbook part.
fn sibling_part(
    pkg: &mut Package,
    rels: &Relationships,
    base: &str,
    rel: &str,
    conventional: &str,
) -> Result<Option<Element>, ConvertError> {
    pkg.optional_xml_part(&sibling_part_name(rels, base, rel, conventional))
}

/// The shared string table, one cleaned entry per `si` in order.
fn shared_strings(root: &Element) -> Vec<String> {
    let Some(sst) = root.find(ns::SML, "sst") else {
        return Vec::new();
    };
    // markitai: a cell's line breaks are content.
    sst.find_all(ns::SML, "si").map(|si| clean_cell_text(&rich_text(si))).collect()
}

/// Text of an `si` or `is`: a single `t`, or rich-text `r` runs
/// concatenated. Phonetic guides (`rPh`) are not content.
pub(super) fn rich_text(item: &Element) -> String {
    let mut out = String::new();
    for child in item.child_elems() {
        if child.is(ns::SML, "t") {
            out.push_str(&child.text());
        } else if child.is(ns::SML, "r")
            && let Some(t) = child.find(ns::SML, "t")
        {
            out.push_str(&t.text());
        }
    }
    out
}

/// A cell's resolved number format: General, or a parsed format code.
/// Shared with the BIFF reader - .xls XF/FORMAT records resolve into the
/// same representation.
#[derive(Clone)]
pub(super) enum CellFormat {
    General,
    Fmt(Rc<NumberFormat>),
}

/// `xl/styles.xml` reduced to what rendering needs: the ordered `cellXfs`
/// list, each entry's numFmtId resolved to a parsed format.
struct Styles {
    xfs: Vec<CellFormat>,
}

impl Styles {
    fn read(root: Option<Element>) -> Styles {
        let Some(root) = root else {
            return Styles { xfs: Vec::new() };
        };
        let mut custom: HashMap<u32, &str> = HashMap::new();
        for fmts in root.descendants(ns::SML, "numFmts") {
            for nf in fmts.find_all(ns::SML, "numFmt") {
                if let (Some(id), Some(code)) = (
                    nf.attr_unqualified("numFmtId").and_then(|v| v.parse().ok()),
                    nf.attr_unqualified("formatCode"),
                ) {
                    custom.insert(id, code);
                }
            }
        }
        let mut cache: HashMap<u32, CellFormat> = HashMap::new();
        let xfs = root
            .first_descendant(ns::SML, "cellXfs")
            .map(|xfs| {
                xfs.find_all(ns::SML, "xf")
                    .map(|xf| {
                        let id = xf
                            .attr_unqualified("numFmtId")
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(0u32);
                        cache.entry(id).or_insert_with(|| resolve_format(id, &custom)).clone()
                    })
                    .collect()
            })
            .unwrap_or_default();
        Styles { xfs }
    }

    /// The format for a cell's `s` attribute, an index into `cellXfs`
    /// (default 0).
    fn for_cell(&self, s: Option<&str>) -> &CellFormat {
        let i = s.and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        self.xfs.get(i).unwrap_or(&CellFormat::General)
    }
}

/// A numFmtId's format: the file's own `numFmt` entries first, then the
/// built-in table. Unknown ids and unsupported codes fall back to General -
/// never to a guess.
pub(super) fn resolve_format(id: u32, custom: &HashMap<u32, &str>) -> CellFormat {
    let code = custom.get(&id).copied().or_else(|| builtin_code(id));
    match code {
        Some(code) => match NumberFormat::parse(code) {
            Some(f) => CellFormat::Fmt(Rc::new(f)),
            None => {
                log::debug!("unsupported number format {code:?}, rendering as General");
                CellFormat::General
            }
        },
        None => {
            if id != 0 {
                log::debug!("numFmtId {id} has no resolvable code, rendering as General");
            }
            CellFormat::General
        }
    }
}

/// One worksheet, parsed but not yet filtered or gridded. Shared with the
/// BIFF reader, which fills it from records instead of XML.
#[derive(Default)]
pub(super) struct SheetContent {
    /// Raw OfficeArt records of this BIFF sheet, including split shapes.
    pub(super) drawings: Vec<u8>,
    /// Rendered text by zero-based (row, col); empty results are absent. A
    /// line break in a cell's text is `\n` (markitai).
    pub(super) cells: HashMap<(u32, u32), String>,
    /// markitai: cells whose text is a formula, written because the cell
    /// caches no value.
    pub(super) formulas: HashSet<(u32, u32)>,
    /// markitai: formulas without a cached value, shown or not (a shared
    /// formula's later cells carry no text of their own).
    pub(super) uncached_formulas: usize,
    /// markitai: the web or mail address a cell links to.
    pub(super) links: HashMap<(u32, u32), String>,
    /// markitai: cell notes and comments by cell, in reading order.
    pub(super) notes: Vec<(u32, u32, String)>,
    /// Form control checkboxes by the cell they are anchored in.
    pub(super) checkboxes: Checkboxes,
    pub(super) hidden_rows: HashSet<u32>,
    /// Inclusive zero-based column ranges hidden by `cols/col` entries.
    pub(super) hidden_cols: Vec<(u32, u32)>,
    /// Inclusive zero-based merge regions (r1, c1, r2, c2), area > 1.
    pub(super) merges: Vec<(u32, u32, u32, u32)>,
}

fn read_sheet(
    worksheet: &Element,
    shared: &[String],
    styles: &Styles,
    date1904: bool,
) -> SheetContent {
    let mut out = SheetContent::default();
    for cols in worksheet.find_all(ns::SML, "cols") {
        for col in cols.find_all(ns::SML, "col") {
            if !col.attr_unqualified("hidden").is_some_and(bool_attr) {
                continue;
            }
            let bound = |name| {
                col.attr_unqualified(name)
                    .and_then(|v| v.parse::<u32>().ok())
                    .and_then(|v| v.checked_sub(1))
            };
            if let (Some(min), Some(max)) = (bound("min"), bound("max"))
                && min <= max
            {
                out.hidden_cols.push((min, max.min(MAX_COLS - 1)));
            }
        }
    }
    let mut next_row: u32 = 0;
    for row in worksheet.find_all(ns::SML, "sheetData").flat_map(|sd| sd.find_all(ns::SML, "row")) {
        let r = row
            .attr_unqualified("r")
            .and_then(|v| v.parse::<u32>().ok())
            .and_then(|v| v.checked_sub(1))
            .unwrap_or(next_row);
        if r >= MAX_ROWS {
            continue;
        }
        next_row = r + 1;
        if row.attr_unqualified("hidden").is_some_and(bool_attr) {
            out.hidden_rows.insert(r);
        }
        let mut next_col: u32 = 0;
        for c in row.find_all(ns::SML, "c") {
            // Position comes from the cell's own reference: a row may skip
            // cells entirely, so iteration order says nothing.
            let (cr, cc) = match c.attr_unqualified("r").map(parse_ref) {
                Some(Some(rc)) => rc,
                Some(None) => continue,
                None => (r, next_col),
            };
            next_col = cc + 1;
            if cr >= MAX_ROWS || cc >= MAX_COLS {
                continue;
            }
            // markitai: a formula cell saved without its value (openpyxl and
            // other writers that do not calculate) shows its formula. An
            // empty value caches nothing but for a string result, which may
            // be empty.
            let (mut formula, mut value) = (None, None);
            for child in c.child_elems() {
                if child.is(ns::SML, "f") {
                    formula = formula.or(Some(child));
                } else if child.is(ns::SML, "v") {
                    value = value.or(Some(child));
                }
            }
            let uncached = || match value {
                None => true,
                Some(v) => {
                    !matches!(c.attr_unqualified("t"), Some("str" | "s" | "inlineStr"))
                        && v.text().trim().is_empty()
                }
            };
            if let Some(formula) = formula
                && uncached()
            {
                out.uncached_formulas += 1;
                let formula = clean_text(&formula.text());
                let formula = formula.trim();
                if !formula.is_empty() {
                    out.cells.insert((cr, cc), format!("={formula}"));
                    out.formulas.insert((cr, cc));
                }
                continue;
            }
            let text = cell_text(c, value, shared, styles, date1904);
            if !text.is_empty() {
                out.cells.insert((cr, cc), text);
            }
        }
    }
    for merge in
        worksheet.find_all(ns::SML, "mergeCells").flat_map(|mc| mc.find_all(ns::SML, "mergeCell"))
    {
        let Some(region) = merge.attr_unqualified("ref").and_then(parse_region) else {
            log::debug!("skipping unparseable merge reference");
            continue;
        };
        let (r1, c1, r2, c2) = region;
        if r1 != r2 || c1 != c2 {
            out.merges.push(region);
        }
    }
    out
}

/// A cell's rendered text, per its `t` type and resolved number format.
/// markitai: `v` is the cell's first `v` child, which the caller has found.
fn cell_text(
    c: &Element,
    v: Option<&Element>,
    shared: &[String],
    styles: &Styles,
    date1904: bool,
) -> String {
    let fmt = styles.for_cell(c.attr_unqualified("s"));
    let value = || v.map(|v| v.text()).unwrap_or_default();
    match c.attr_unqualified("t").unwrap_or("n") {
        "s" => {
            let v = value();
            match v.trim().parse::<usize>().ok().and_then(|i| shared.get(i)) {
                Some(text) => format_as_text(fmt, text),
                None => {
                    log::debug!("shared string index {v:?} out of range");
                    String::new()
                }
            }
        }
        "str" => format_as_text(fmt, &clean_cell_text(&value())),
        "inlineStr" => {
            let text = c.find(ns::SML, "is").map(|is| clean_cell_text(&rich_text(is)));
            format_as_text(fmt, &text.unwrap_or_default())
        }
        "b" => match value().trim() {
            "1" | "true" => "TRUE".to_string(),
            "0" | "false" => "FALSE".to_string(),
            _ => String::new(),
        },
        "e" | "d" => clean_text(&value()),
        _ => {
            let v = value();
            let v = v.trim();
            if v.is_empty() {
                return String::new();
            }
            let Ok(n) = v.parse::<f64>() else {
                log::debug!("unparseable numeric cell value {v:?}");
                return String::new();
            };
            render_number(fmt, n, date1904)
        }
    }
}

/// Render a numeric cell through its resolved format. Shared with the BIFF
/// and binary readers so every container formats identically.
pub(super) fn render_number(fmt: &CellFormat, n: f64, date1904: bool) -> String {
    let text = match fmt {
        CellFormat::General => format_float(n),
        CellFormat::Fmt(f) => match f.format_number(n) {
            Rendered::General { value, prefix, suffix } => {
                format!("{prefix}{}{suffix}", format_float(value))
            }
            Rendered::Text(s) => s,
            Rendered::DateTime(parts) => render_serial(n, parts, date1904),
            Rendered::Spelled(pieces) => render_spelled(n, pieces, date1904),
        },
    };
    clean_text(&text)
}

pub(super) fn format_as_text(fmt: &CellFormat, text: &str) -> String {
    match fmt {
        CellFormat::Fmt(f) => match f.format_text(text) {
            Some(s) => clean_text(&s),
            None => text.to_string(),
        },
        CellFormat::General => text.to_string(),
    }
}

/// Materialize a sheet's grid: visibility filtering, the populated extent
/// widened to cover intersecting merge regions (a merge anchored on the
/// only populated cell must survive at full size), and merges remapped onto
/// the surviving rows and columns.
pub(super) fn build_table(mut sheet: SheetContent, slots: &mut u64) -> Result<Built, ConvertError> {
    // Hidden coordinates as sorted lists: lookups and first-visible scans
    // stay logarithmic, so an adversarial pile of hidden rows or column
    // ranges cannot force quadratic work.
    let hidden_rows = {
        let mut rows: Vec<u32> = sheet.hidden_rows.iter().copied().collect();
        rows.sort_unstable();
        rows
    };
    let hidden_cols = expand_ranges(&mut sheet.hidden_cols);
    let hidden_row = |r: u32| hidden_rows.binary_search(&r).is_ok();
    let hidden_col = |c: u32| hidden_cols.binary_search(&c).is_ok();

    // Cell text and anchored checkboxes become one inline map; the rest of
    // the assembly no longer cares which was which.
    let mut cells: HashMap<(u32, u32), Vec<Inline>> = HashMap::new();
    for (at, boxes) in sheet.checkboxes.drain() {
        if at.0 < MAX_ROWS && at.1 < MAX_COLS {
            cells.insert(at, cell_inlines(sheet.cells.remove(&at), &boxes));
        }
    }
    for (at, text) in sheet.cells.drain() {
        // markitai: a formula shown for its missing value reads as code, a
        // line break stays one, and a linked cell is its link.
        let mut inlines = if sheet.formulas.contains(&at) {
            vec![Inline::Text { text, style: Style { code: true, ..Style::PLAIN } }]
        } else {
            cell_lines(text)
        };
        if let Some(url) = sheet.links.get(&at) {
            inlines =
                vec![Inline::Link { content: inlines, target: LinkTarget::External(url.clone()) }];
        }
        cells.insert(at, inlines);
    }
    // A merge with no surviving row or column disappears with its content.
    // One whose origin is hidden keeps its content at the first surviving
    // position it covers, so the value is not lost.
    sheet.merges.retain(|&(r1, c1, r2, c2)| {
        let vr = first_visible(&hidden_rows, r1, r2);
        let vc = first_visible(&hidden_cols, c1, c2);
        let (Some(vr), Some(vc)) = (vr, vc) else {
            return false;
        };
        if (vr, vc) != (r1, c1)
            && let Some(text) = cells.remove(&(r1, c1))
        {
            cells.insert((vr, vc), text);
        }
        true
    });

    // markitai: the hidden rows and columns that hold content, which the
    // caller reports, and the notes of the cells that stay visible.
    let mut built = Built::default();
    {
        let mut rows = HashSet::new();
        let mut cols = HashSet::new();
        for &(r, c) in cells.keys() {
            if hidden_row(r) {
                rows.insert(r);
            }
            if hidden_col(c) {
                cols.insert(c);
            }
        }
        built.hidden_rows = rows.len();
        built.hidden_cols = cols.len();
    }
    built.notes = std::mem::take(&mut sheet.notes)
        .into_iter()
        .filter(|&(r, c, _)| !hidden_row(r) && !hidden_col(c))
        .collect();

    // Populated extent over visible cells only.
    let mut bounds: Option<(u32, u32, u32, u32)> = None;
    for &(r, c) in cells.keys() {
        if hidden_row(r) || hidden_col(c) {
            continue;
        }
        bounds = Some(match bounds {
            None => (r, c, r, c),
            Some((r1, c1, r2, c2)) => (r1.min(r), c1.min(c), r2.max(r), c2.max(c)),
        });
    }
    let Some((mut r1, mut c1, mut r2, mut c2)) = bounds else {
        return Ok(built);
    };
    // Merge regions touching the populated extent widen it to their full
    // size; the rest are dropped, so a crafted merge list can neither force
    // unbounded materialization nor saturate onto (0,0).
    sheet.merges.retain(|&(mr1, mc1, mr2, mc2)| mr1 <= r2 && mr2 >= r1 && mc1 <= c2 && mc2 >= c1);
    for &(mr1, mc1, mr2, mc2) in &sheet.merges {
        (r1, c1, r2, c2) = (r1.min(mr1), c1.min(mc1), r2.max(mr2), c2.max(mc2));
    }

    let row_map: Vec<u32> = (r1..=r2).filter(|&r| !hidden_row(r)).collect();
    let col_map: Vec<u32> = (c1..=c2).filter(|&c| !hidden_col(c)).collect();
    if row_map.is_empty() || col_map.is_empty() {
        return Ok(built);
    }
    // Charged before materializing, and across the workbook rather than per
    // sheet: the extent comes from cell coordinates, so a handful of cells
    // describes a whole sheet and a handful of sheets multiplies it.
    *slots = slots.saturating_add(row_map.len() as u64 * col_map.len() as u64);
    if *slots > limits::MAX_GRID_SLOTS {
        return Err(ConvertError::ResourceLimit {
            limit: "max_grid_slots",
            detail: format!("workbook extent covers {slots} grid positions"),
        });
    }

    // Remap merges onto the surviving coordinates. The covered-position set
    // is charged against the expansion budget up front, before any
    // insertion work, mirroring what placement would charge.
    let visible_span = |map: &[u32], lo: u32, hi: u32| {
        let a = map.partition_point(|&x| x < lo);
        let b = map.partition_point(|&x| x <= hi);
        (a, b - a)
    };
    let mut origins: HashMap<(usize, usize), (u32, u32)> = HashMap::new();
    let mut covered: HashSet<(usize, usize)> = HashSet::new();
    let mut expansion = 0u64;
    for &(mr1, mc1, mr2, mc2) in &sheet.merges {
        let (r0, rn) = visible_span(&row_map, mr1, mr2);
        let (c0, cn) = visible_span(&col_map, mc1, mc2);
        if rn * cn <= 1 {
            continue;
        }
        expansion = expansion.saturating_add((rn as u64) * (cn as u64) - 1);
        if expansion > limits::MAX_EXPANSION {
            return Err(ConvertError::ResourceLimit {
                limit: "max_expansion",
                detail: "merge region expansion exceeds the content budget".into(),
            });
        }
        origins.insert((r0, c0), (cn as u32, rn as u32));
        for r in r0..r0 + rn {
            for c in c0..c0 + cn {
                if (r, c) != (r0, c0) {
                    covered.insert((r, c));
                }
            }
        }
    }

    let mut builder = GridBuilder::new();
    // A merge is real extent: trailing rows it covers stay in the grid.
    builder.keep_covered_tail();
    for (ri, &row) in row_map.iter().enumerate() {
        builder.next_row();
        for (ci, &col) in col_map.iter().enumerate() {
            if covered.contains(&(ri, ci)) {
                builder.covered();
                continue;
            }
            let cell = match cells.remove(&(row, col)) {
                Some(inlines) => Cell::from_inlines(inlines),
                None => Cell::default(),
            };
            match origins.get(&(ri, ci)) {
                Some(&(col_span, row_span)) => {
                    builder.place(Cell::spanning(cell.blocks, col_span, row_span))?
                }
                None => builder.place(cell)?,
            }
        }
    }
    // A spreadsheet marks no header row, so the shape of the data decides.
    let mut table = builder.finish(TableKind::Data);
    if table.grid.is_empty() {
        return Ok(built);
    }
    table.header_rows = resolve_header_rows(&table, 0);
    built.table = Some(table);
    Ok(built)
}

/// markitai: a cell's text as inlines, each line break kept as one.
fn cell_lines(text: String) -> Vec<Inline> {
    if !text.contains('\n') {
        return vec![Inline::plain(text)];
    }
    let mut inlines = Vec::new();
    for (i, line) in text.split('\n').enumerate() {
        if i > 0 {
            inlines.push(Inline::LineBreak);
        }
        if !line.is_empty() {
            inlines.push(Inline::plain(line));
        }
    }
    inlines
}

/// Flatten inclusive ranges into a sorted, deduplicated coordinate list.
/// Coalescing before expansion bounds the output by the coordinate space,
/// not by the range count.
fn expand_ranges(ranges: &mut [(u32, u32)]) -> Vec<u32> {
    ranges.sort_unstable();
    let mut out = Vec::new();
    let mut next = 0u32;
    for &(a, b) in ranges.iter() {
        out.extend(a.max(next)..=b);
        next = next.max(b.saturating_add(1));
    }
    out
}

/// First coordinate in `lo..=hi` absent from the sorted hidden list. The
/// consecutive hidden run starting at `lo` is measured by binary search, so
/// a long run cannot force a linear scan per query.
fn first_visible(hidden: &[u32], lo: u32, hi: u32) -> Option<u32> {
    let tail = &hidden[hidden.partition_point(|&h| h < lo)..];
    // `tail[j] - j` never decreases, so "the run still holds at j" is a
    // prefix property.
    let (mut a, mut b) = (0usize, tail.len());
    while a < b {
        let mid = (a + b) / 2;
        if tail[mid] == lo + mid as u32 {
            a = mid + 1;
        } else {
            b = mid;
        }
    }
    let first = lo.checked_add(u32::try_from(a).ok()?)?;
    (first <= hi).then_some(first)
}

/// Render a date/time serial the way the crate always has: elapsed formats
/// as a duration, sub-day serials as a time of day, everything else as an
/// ISO-like date with the midnight time omitted.
fn render_serial(serial: f64, parts: DateParts, date1904: bool) -> String {
    if !serial.is_finite() {
        return format_float(serial);
    }
    if parts.elapsed {
        return format_duration_days(serial, parts.span, parts.second_fraction);
    }
    if !parts.date {
        return format_time_of_day(serial.fract(), parts.seconds);
    }
    // A serial carrying no whole day has no date for a date format to show;
    // the clock a combined format also names still renders.
    if serial.abs() < 1.0 {
        return if parts.time {
            format_time_of_day(serial, parts.seconds)
        } else {
            format_float(serial)
        };
    }
    // Out of the representable date range (through 9999-12-31): the serial
    // is not a date, show the number.
    if !(0.0..2_958_466.0).contains(&serial) {
        return format_float(serial);
    }
    let mut days = serial.trunc() as i64;
    // The fictitious 1900-02-29 has no date to render and would otherwise
    // collapse onto serial 59. Tested before the seconds carry, so a value
    // late on serial 59 still resolves as the date it belongs to.
    if !date1904 && days == 60 {
        return format_float(serial);
    }
    let mut secs = (serial.fract() * 86_400.0).round() as i64;
    if secs >= 86_400 {
        secs = 0;
        days += 1;
    }
    let civil_days = if date1904 {
        days + days_from_civil(1904, 1, 1)
    } else {
        // 1900 system: serial 1 is 1900-01-01, and the fictitious
        // 1900-02-29 (serial 60) offsets everything after it by one day.
        days - i64::from(days >= 60) + days_from_civil(1899, 12, 31)
    };
    let (y, m, d) = civil_from_days(civil_days);
    if !(1..=9999).contains(&y) {
        return format_float(serial);
    }
    let mut out = format!("{y:04}-{m:02}-{d:02}");
    if parts.time && secs != 0 {
        out.push(' ');
        out.push_str(&format_time_of_day(secs as f64 / 86_400.0, parts.seconds));
    }
    out
}

/// markitai: a date serial written through a spelled date format's pieces
/// (`dddd, mmmm d, yyyy` → `Wednesday, March 4, 2026`), with English names.
/// A serial outside the date range shows the number, as [`render_serial`]
/// does.
fn render_spelled(serial: f64, pieces: &[DatePiece], date1904: bool) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const DAYS: [&str; 7] =
        ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    if !serial.is_finite() || !(1.0..2_958_466.0).contains(&serial) {
        return format_float(serial);
    }
    let mut days = serial.trunc() as i64;
    if !date1904 && days == 60 {
        return format_float(serial);
    }
    let mut secs = (serial.fract() * 86_400.0).round() as i64;
    if secs >= 86_400 {
        secs = 0;
        days += 1;
    }
    let civil = if date1904 {
        days + days_from_civil(1904, 1, 1)
    } else {
        days - i64::from(days >= 60) + days_from_civil(1899, 12, 31)
    };
    let (y, m, d) = civil_from_days(civil);
    if !(1..=9999).contains(&y) {
        return format_float(serial);
    }
    // 1970-01-01 was a Thursday.
    let weekday = (civil + 4).rem_euclid(7) as usize;
    let clock12 = pieces.iter().any(|p| matches!(p, DatePiece::AmPm(_)));
    let hour = secs / 3600;
    let pad = |value: i64, width: usize| {
        if width >= 2 { format!("{value:02}") } else { value.to_string() }
    };
    let mut out = String::new();
    for piece in pieces {
        match piece {
            DatePiece::Literal(text) => out.push_str(text),
            DatePiece::Year(n) if *n <= 2 => out.push_str(&format!("{:02}", y % 100)),
            DatePiece::Year(_) => out.push_str(&format!("{y:04}")),
            DatePiece::Month(n) => match n {
                1 | 2 => out.push_str(&pad(i64::from(m), *n)),
                3 => out.push_str(&MONTHS[m as usize - 1][..3]),
                _ => out.push_str(MONTHS[m as usize - 1]),
            },
            DatePiece::Day(n) => match n {
                1 | 2 => out.push_str(&pad(i64::from(d), *n)),
                3 => out.push_str(&DAYS[weekday][..3]),
                _ => out.push_str(DAYS[weekday]),
            },
            DatePiece::Hour(n) => {
                let shown = if clock12 { (hour + 11) % 12 + 1 } else { hour };
                out.push_str(&pad(shown, *n));
            }
            DatePiece::Minute(n) => out.push_str(&pad(secs / 60 % 60, *n)),
            DatePiece::Second(n) => out.push_str(&pad(secs % 60, *n)),
            DatePiece::AmPm(full) => out.push_str(match (hour < 12, full) {
                (true, true) => "AM",
                (false, true) => "PM",
                (true, false) => "A",
                (false, false) => "P",
            }),
        }
    }
    out
}

/// Days from 1970-01-01 to a civil date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = y - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = i64::from((m + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// A civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (y + i64::from(m <= 2), m, d)
}

/// A cell reference (`C3`) as zero-based (row, column); the column letters
/// are bijective base-26.
pub(super) fn parse_ref(r: &str) -> Option<(u32, u32)> {
    let digits_at = r.find(|c: char| c.is_ascii_digit())?;
    let (letters, digits) = r.split_at(digits_at);
    if letters.is_empty() {
        return None;
    }
    let mut col: u32 = 0;
    for ch in letters.chars() {
        if !ch.is_ascii_alphabetic() {
            return None;
        }
        col = col.checked_mul(26)?.checked_add(ch.to_ascii_uppercase() as u32 - 'A' as u32 + 1)?;
        if col > MAX_COLS {
            return None;
        }
    }
    let row: u32 = digits.parse().ok()?;
    if !(1..=MAX_ROWS).contains(&row) {
        return None;
    }
    Some((row - 1, col - 1))
}

/// A merge reference (`F1:O3`, or a single cell) as an inclusive normalized
/// region.
pub(super) fn parse_region(r: &str) -> Option<(u32, u32, u32, u32)> {
    let (a, b) = r.split_once(':').unwrap_or((r, r));
    let (r1, c1) = parse_ref(a.trim())?;
    let (r2, c2) = parse_ref(b.trim())?;
    Some((r1.min(r2), c1.min(c2), r1.max(r2), c1.max(c2)))
}

/// XML schema boolean attribute.
fn bool_attr(v: &str) -> bool {
    matches!(v.trim(), "1" | "true")
}

#[cfg(test)]
mod tests {
    use super::super::numfmt::Unit;
    use super::*;
    use crate::model::{CellSlot, inlines_to_plain_text};
    use std::io::Write;

    const SML: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
    const R: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
    const PKG_RELS: &str = "http://schemas.openxmlformats.org/package/2006/relationships";
    const WS_REL: &str =
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet";
    const STYLES_REL: &str =
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles";

    /// Route through the shared container dispatch, the only entry a caller
    /// has.
    fn parse(bytes: &[u8]) -> Result<Document, ConvertError> {
        super::super::parse(bytes)
    }

    /// Assemble a workbook: (name, state, worksheet body) per sheet, plus
    /// optional styleSheet and sst parts.
    #[derive(Default)]
    struct Wb<'a> {
        sheets: Vec<(&'a str, &'a str, &'a str)>,
        styles: Option<&'a str>,
        shared: Option<&'a str>,
        date1904: bool,
        /// Further parts, verbatim: (name, body).
        extra: Vec<(&'a str, &'a str)>,
    }

    impl Wb<'_> {
        fn build(&self) -> Vec<u8> {
            let mut sheets = String::new();
            let mut rels = String::new();
            for (i, (name, state, _)) in self.sheets.iter().enumerate() {
                let id = i + 1;
                let state =
                    if state.is_empty() { String::new() } else { format!(" state=\"{state}\"") };
                sheets.push_str(&format!(
                    r#"<sheet name="{name}" sheetId="{id}"{state} r:id="rId{id}"/>"#
                ));
                rels.push_str(&format!(
                    r#"<Relationship Id="rId{id}" Type="{WS_REL}" Target="worksheets/sheet{id}.xml"/>"#
                ));
            }
            if self.styles.is_some() {
                rels.push_str(&format!(
                    r#"<Relationship Id="rId90" Type="{STYLES_REL}" Target="styles.xml"/>"#
                ));
            }
            if self.shared.is_some() {
                rels.push_str(&format!(
                    r#"<Relationship Id="rId91" Type="{SHARED_STRINGS_REL}" Target="sharedStrings.xml"/>"#
                ));
            }
            let pr = if self.date1904 { r#"<workbookPr date1904="1"/>"# } else { "" };
            let workbook = format!(
                r#"<?xml version="1.0"?><workbook xmlns="{SML}" xmlns:r="{R}">{pr}<sheets>{sheets}</sheets></workbook>"#
            );
            let rels = format!(
                r#"<?xml version="1.0"?><Relationships xmlns="{PKG_RELS}">{rels}</Relationships>"#
            );
            let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
            let opts = zip::write::SimpleFileOptions::default();
            let mut add = |name: &str, body: &str| {
                zip.start_file(name, opts).unwrap();
                zip.write_all(body.as_bytes()).unwrap();
            };
            add("xl/workbook.xml", &workbook);
            add("xl/_rels/workbook.xml.rels", &rels);
            for (i, (_, _, body)) in self.sheets.iter().enumerate() {
                add(
                    &format!("xl/worksheets/sheet{}.xml", i + 1),
                    &format!(r#"<?xml version="1.0"?><worksheet xmlns="{SML}">{body}</worksheet>"#),
                );
            }
            if let Some(styles) = self.styles {
                add(
                    "xl/styles.xml",
                    &format!(
                        r#"<?xml version="1.0"?><styleSheet xmlns="{SML}">{styles}</styleSheet>"#
                    ),
                );
            }
            if let Some(shared) = self.shared {
                add(
                    "xl/sharedStrings.xml",
                    &format!(r#"<?xml version="1.0"?><sst xmlns="{SML}">{shared}</sst>"#),
                );
            }
            for (name, body) in &self.extra {
                add(name, body);
            }
            zip.finish().unwrap().into_inner()
        }
    }

    fn one_sheet(body: &str) -> Wb<'_> {
        Wb { sheets: vec![("S", "", body)], ..Wb::default() }
    }

    fn first_table(doc: &Document) -> &Table {
        match doc.blocks.iter().find_map(|b| match b {
            Block::Table(t) => Some(t),
            _ => None,
        }) {
            Some(t) => t,
            None => panic!("expected a table, got {:?}", doc.blocks),
        }
    }

    fn texts(table: &Table) -> Vec<Vec<String>> {
        table
            .grid
            .iter()
            .map(|row| {
                row.iter()
                    .map(|slot| match slot {
                        CellSlot::Origin(cell) => cell
                            .blocks
                            .iter()
                            .filter_map(|b| match b {
                                Block::Paragraph(i) => Some(inlines_to_plain_text(i)),
                                _ => None,
                            })
                            .collect(),
                        CellSlot::Covered { .. } => "<covered>".to_string(),
                    })
                    .collect()
            })
            .collect()
    }

    fn covered_count(table: &Table) -> usize {
        table.grid.iter().flatten().filter(|s| matches!(s, CellSlot::Covered { .. })).count()
    }

    #[test]
    fn number_formats_apply_to_stored_values() {
        // The issue #27 case: a percent renders as its display value, not
        // its stored fraction; currency keeps its symbol and grouping; a
        // date format keeps the unambiguous ISO rendering.
        let wb = Wb {
            styles: Some(
                r#"<numFmts><numFmt numFmtId="164" formatCode="0.0%"/><numFmt numFmtId="165" formatCode="&quot;$&quot;#,##0.00"/><numFmt numFmtId="166" formatCode="mm/dd/yyyy"/></numFmts><cellXfs><xf numFmtId="0"/><xf numFmtId="164"/><xf numFmtId="165"/><xf numFmtId="166"/></cellXfs>"#,
            ),
            ..one_sheet(
                r#"<sheetData><row r="1"><c r="A1" s="1"><v>0.075</v></c><c r="B1" s="2"><v>1234.5</v></c><c r="C1" s="3"><v>46096</v></c></row></sheetData>"#,
            )
        };
        let doc = parse(&wb.build()).unwrap();
        assert_eq!(texts(first_table(&doc)), vec![vec!["7.5%", "$1,234.50", "2026-03-15"]]);
    }

    #[test]
    fn unresolvable_numfmt_ids_render_general() {
        // Id 23 is not an implied built-in and id 30 is locale-specific:
        // with no numFmt element the code is unknown, and guessing (a date
        // shape) would be worse than General.
        let wb = Wb {
            styles: Some(r#"<cellXfs><xf numFmtId="23"/><xf numFmtId="30"/></cellXfs>"#),
            ..one_sheet(
                r#"<sheetData><row r="1"><c r="A1" s="0"><v>1234.5</v></c><c r="B1" s="1"><v>1234.5</v></c></row></sheetData>"#,
            )
        };
        let doc = parse(&wb.build()).unwrap();
        assert_eq!(texts(first_table(&doc)), vec![vec!["1234.5", "1234.5"]]);
    }

    // markitai: the currency and accounting ids openpyxl writes without a
    // code: grouping, decimals and negative parentheses, but no currency
    // symbol, which depends on a locale the file does not name. A code that
    // writes its own symbol keeps it.
    #[test]
    fn currency_and_accounting_ids_without_a_code_show_parentheses() {
        let wb = Wb {
            styles: Some(
                r#"<numFmts><numFmt numFmtId="164" formatCode="&quot;$&quot;#,##0.00_);[Red](&quot;$&quot;#,##0.00)"/><numFmt numFmtId="165" formatCode="[$€-407] #,##0.00"/></numFmts><cellXfs><xf numFmtId="8"/><xf numFmtId="7"/><xf numFmtId="5"/><xf numFmtId="44"/><xf numFmtId="43"/><xf numFmtId="164"/><xf numFmtId="165"/></cellXfs>"#,
            ),
            ..one_sheet(
                r#"<sheetData><row r="1"><c r="A1" s="0"><v>-1234.5</v></c><c r="B1" s="1"><v>1234.5</v></c><c r="C1" s="2"><v>-1234.5</v></c><c r="D1" s="3"><v>-1234.5</v></c><c r="E1" s="4"><v>1234.5</v></c><c r="F1" s="5"><v>-1234.5</v></c><c r="G1" s="6"><v>1234.5</v></c></row></sheetData>"#,
            )
        };
        let doc = parse(&wb.build()).unwrap();
        let row: Vec<String> =
            texts(first_table(&doc))[0].iter().map(|cell| cell.trim().to_string()).collect();
        assert_eq!(
            row,
            [
                "(1,234.50)",
                "1,234.50",
                "(1,235)",
                "(1,234.50)",
                "1,234.50",
                "($1,234.50)",
                "€ 1,234.50"
            ]
        );
    }

    // markitai: what openpyxl and other writers that do not calculate leave
    // in a formula cell, and what a sheet attaches to its cells.
    #[test]
    fn formulas_without_a_cached_value_show_their_text_once_warned() {
        let wb = one_sheet(
            r#"<sheetData><row r="1"><c r="A1"><v>2</v></c><c r="B1"><f>A1*3</f><v></v></c><c r="C1"><f>A1*4</f></c><c r="D1" t="str"><f>""</f><v></v></c><c r="E1"><f>A1*5</f><v>10</v></c><c r="F1"><f t="shared" si="0"/></c></row></sheetData>"#,
        );
        let doc = parse(&wb.build()).unwrap();
        let table = first_table(&doc);
        // A shared formula's later cell has no text of its own to show.
        assert_eq!(texts(table), vec![vec!["2", "=A1*3", "=A1*4", "", "10"]]);
        let CellSlot::Origin(cell) = &table.grid[0][1] else { panic!() };
        assert!(matches!(&cell.blocks[0], Block::Paragraph(i)
            if matches!(&i[0], Inline::Text { style, .. } if style.code)));
        assert_eq!(
            doc.warnings,
            [
                "3 formulas in the workbook have no cached value, so their formula text is shown; open and save the workbook in Excel or LibreOffice to compute them."
            ]
        );
        // A cached workbook warns about nothing.
        let cached = one_sheet(
            r#"<sheetData><row r="1"><c r="A1"><f>1+1</f><v>2</v></c></row></sheetData>"#,
        );
        assert!(parse(&cached.build()).unwrap().warnings.is_empty());
    }

    const HYPERLINK_REL: &str =
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink";
    const COMMENTS_REL: &str =
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments";

    #[test]
    fn hyperlinks_notes_hidden_cells_and_line_breaks() {
        let rels = format!(
            r##"<?xml version="1.0"?><Relationships xmlns="{PKG_RELS}"><Relationship Id="rId1" Type="{HYPERLINK_REL}" Target="https://example.com/page" TargetMode="External"/><Relationship Id="rId2" Type="{HYPERLINK_REL}" Target="#Sheet2!A1" TargetMode="External"/><Relationship Id="rId3" Type="{HYPERLINK_REL}" Target="mailto:bob@example.com" TargetMode="External"/><Relationship Id="rId4" Type="{HYPERLINK_REL}" Target="file:///etc/passwd" TargetMode="External"/><Relationship Id="rId5" Type="{COMMENTS_REL}" Target="/xl/comments1.xml"/></Relationships>"##
        );
        let comments = format!(
            r#"<?xml version="1.0"?><comments xmlns="{SML}"><authors><author>Ann</author></authors><commentList><comment ref="B2" authorId="0"><text><r><rPr><b/></rPr><t>Ann:</t></r><r><t xml:space="preserve">
Check this
value</t></r></text></comment><comment ref="A1" authorId="0"><text><t>First</t></text></comment><comment ref="C3" authorId="0"><text><t>On a hidden row</t></text></comment></commentList></comments>"#
        );
        let wb = Wb {
            sheets: vec![(
                "S",
                "",
                r#"<cols><col min="6" max="6" hidden="1"/></cols><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Name</t></is></c><c r="B1" t="inlineStr"><is><t>Site</t></is></c><c r="F1" t="inlineStr"><is><t>secret</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>line one
line two</t></is></c><c r="B2" t="inlineStr"><is><t>example.com</t></is></c><c r="C2" t="inlineStr"><is><t>Sheet2</t></is></c><c r="D2" t="inlineStr"><is><t>mail</t></is></c><c r="E2" t="inlineStr"><is><t>file</t></is></c></row><row r="3" hidden="1"><c r="C3" t="inlineStr"><is><t>hidden</t></is></c></row></sheetData><hyperlinks xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><hyperlink ref="B2" r:id="rId1"/><hyperlink ref="C2" r:id="rId2"/><hyperlink ref="D2" r:id="rId3"/><hyperlink ref="E2" r:id="rId4"/></hyperlinks>"#,
            )],
            extra: vec![
                ("xl/worksheets/_rels/sheet1.xml.rels", &rels),
                ("xl/comments1.xml", &comments),
            ],
            ..Wb::default()
        };
        let doc = parse(&wb.build()).unwrap();
        let table = first_table(&doc);
        let CellSlot::Origin(link) = &table.grid[1][1] else { panic!() };
        assert!(matches!(&link.blocks[0], Block::Paragraph(i) if matches!(&i[0],
            Inline::Link { target: LinkTarget::External(url), .. } if url == "https://example.com/page")));
        // Links into the workbook and to other schemes keep their text only.
        let CellSlot::Origin(mail) = &table.grid[1][3] else { panic!() };
        assert!(matches!(&mail.blocks[0], Block::Paragraph(i) if matches!(&i[0],
            Inline::Link { target: LinkTarget::External(url), .. } if url == "mailto:bob@example.com")));
        for col in [2, 4] {
            let CellSlot::Origin(plain) = &table.grid[1][col] else { panic!() };
            assert!(
                matches!(&plain.blocks[0], Block::Paragraph(i) if matches!(&i[0], Inline::Text { .. }))
            );
        }
        let CellSlot::Origin(lines) = &table.grid[1][0] else { panic!() };
        assert!(matches!(&lines.blocks[0], Block::Paragraph(i)
            if matches!(i.as_slice(), [Inline::Text { .. }, Inline::LineBreak, Inline::Text { .. }])));
        // The hidden row's note goes with it; the rest follow the table in
        // reading order, on one line each.
        let Some(Block::List(notes)) = doc.blocks.last() else { panic!("{:?}", doc.blocks) };
        let notes: Vec<String> = notes
            .items
            .iter()
            .map(|item| match &item.blocks[0] {
                Block::Paragraph(i) => inlines_to_plain_text(i),
                _ => String::new(),
            })
            .collect();
        assert_eq!(notes, ["A1: First", "B2: Ann: Check this value"]);
        assert!(matches!(&doc.blocks[doc.blocks.len() - 2], Block::Heading { level: 3, .. }));
        assert_eq!(
            doc.warnings,
            [
                "Worksheet \"S\" has 1 hidden row and 1 hidden column holding content; they are omitted by the native spreadsheet reader."
            ]
        );
    }

    #[test]
    fn threaded_comments_replace_their_legacy_placeholder() {
        let rels = format!(
            r#"<?xml version="1.0"?><Relationships xmlns="{PKG_RELS}"><Relationship Id="rId1" Type="{COMMENTS_REL}" Target="../comments1.xml"/><Relationship Id="rId2" Type="http://schemas.microsoft.com/office/2017/10/relationships/threadedComment" Target="../threadedComments/threadedComment1.xml"/></Relationships>"#
        );
        let legacy = format!(
            r#"<?xml version="1.0"?><comments xmlns="{SML}"><authors><author>tc={{1}}</author></authors><commentList><comment ref="A1" authorId="0"><text><t>[Threaded comment] Your version of Excel allows you to read this threaded comment</t></text></comment><comment ref="A2" authorId="0"><text><t>A plain note</t></text></comment></commentList></comments>"#
        );
        let threaded = r#"<?xml version="1.0"?><ThreadedComments xmlns="http://schemas.microsoft.com/office/spreadsheetml/2018/threadedcomments"><threadedComment ref="A1" personId="{P1}" id="{1}"><text>Is this right?</text></threadedComment><threadedComment ref="A1" personId="{P2}" id="{2}" parentId="{1}"><text>Yes.</text></threadedComment></ThreadedComments>"#;
        let persons = r#"<?xml version="1.0"?><personList xmlns="http://schemas.microsoft.com/office/spreadsheetml/2018/threadedcomments"><person displayName="Ann Lee" id="{P1}"/><person displayName="Bo" id="{P2}"/></personList>"#;
        let mut book = one_sheet(
            r#"<sheetData><row r="1"><c r="A1"><v>1</v></c></row><row r="2"><c r="A2"><v>2</v></c></row></sheetData>"#,
        );
        book.extra = vec![
            ("xl/worksheets/_rels/sheet1.xml.rels", &rels),
            ("xl/comments1.xml", &legacy),
            ("xl/threadedComments/threadedComment1.xml", threaded),
            ("xl/persons/person.xml", persons),
        ];
        let mut bytes = book.build();
        // The persons part hangs off the workbook's own relationships.
        bytes = with_workbook_rel(
            &bytes,
            r#"<Relationship Id="rId99" Type="http://schemas.microsoft.com/office/2017/10/relationships/person" Target="persons/person.xml"/>"#,
        );
        let doc = parse(&bytes).unwrap();
        let Some(Block::List(notes)) = doc.blocks.last() else { panic!("{:?}", doc.blocks) };
        let notes: Vec<String> = notes
            .items
            .iter()
            .map(|item| match &item.blocks[0] {
                Block::Paragraph(i) => inlines_to_plain_text(i),
                _ => String::new(),
            })
            .collect();
        assert_eq!(notes, ["A1: Ann Lee: Is this right?", "A1: Bo: Yes.", "A2: A plain note"]);
    }

    /// A built workbook with one more relationship in its workbook part's
    /// relationships.
    fn with_workbook_rel(bytes: &[u8], rel: &str) -> Vec<u8> {
        use std::io::Read;
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default();
        for i in 0..archive.len() {
            let mut file = archive.by_index(i).unwrap();
            let name = file.name().to_string();
            let mut body = String::new();
            file.read_to_string(&mut body).unwrap();
            if name == "xl/_rels/workbook.xml.rels" {
                body = body.replace("</Relationships>", &format!("{rel}</Relationships>"));
            }
            zip.start_file(name, opts).unwrap();
            zip.write_all(body.as_bytes()).unwrap();
        }
        zip.finish().unwrap().into_inner()
    }

    #[test]
    fn spelled_dates_write_names_and_a_twelve_hour_clock() {
        let pieces = |code: &str| match NumberFormat::parse(code).unwrap().format_number(0.0) {
            Rendered::Spelled(p) => p.to_vec(),
            other => panic!("{other:?}"),
        };
        // 2026-03-04 15:30, a Wednesday.
        let serial = 46_085.0 + 15.5 / 24.0;
        let spell = |code: &str| render_spelled(serial, &pieces(code), false);
        assert_eq!(spell("dddd, mmmm d, yyyy"), "Wednesday, March 4, 2026");
        assert_eq!(spell("ddd dd mmm yyyy"), "Wed 04 Mar 2026");
        assert_eq!(spell("mmm d, yyyy h:mm AM/PM"), "Mar 4, 2026 3:30 PM");
        assert_eq!(spell("yyyy\"年\"m\"月\"d\"日\""), "2026年3月4日");
        assert_eq!(render_spelled(46_085.0, &pieces("d mmmm yyyy"), true), "5 March 2030");
        assert_eq!(render_spelled(60.0, &pieces("d mmmm yyyy"), false), "60");
        assert_eq!(render_spelled(0.5, &pieces("d mmmm yyyy"), false), "0.5");
    }

    #[test]
    fn value_types_render_by_their_t_attribute() {
        let wb = one_sheet(
            r#"<sheetData><row r="1"><c r="A1" t="b"><v>1</v></c><c r="B1" t="e"><v>#DIV/0!</v></c><c r="C1" t="str"><v>=sum</v></c><c r="D1" t="d"><v>2026-03-15</v></c><c r="E1" t="inlineStr"><is><r><t>in</t></r><r><t>line</t></r></is></c></row></sheetData>"#,
        );
        let doc = parse(&wb.build()).unwrap();
        assert_eq!(
            texts(first_table(&doc)),
            vec![vec!["TRUE", "#DIV/0!", "=sum", "2026-03-15", "inline"]]
        );
    }

    #[test]
    fn shared_strings_resolve_including_rich_text_runs() {
        let wb = Wb {
            shared: Some(
                r#"<si><t>plain</t></si><si><r><t>ri</t></r><r><t>ch</t></r><rPh sb="0" eb="1"><t>ignored</t></rPh></si>"#,
            ),
            ..one_sheet(
                r#"<sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row></sheetData>"#,
            )
        };
        let doc = parse(&wb.build()).unwrap();
        assert_eq!(texts(first_table(&doc)), vec![vec!["plain", "rich"]]);
    }

    #[test]
    fn date1904_serials_shift_epoch() {
        let wb = Wb {
            styles: Some(
                r#"<numFmts><numFmt numFmtId="164" formatCode="yyyy-mm-dd"/></numFmts><cellXfs><xf numFmtId="164"/></cellXfs>"#,
            ),
            date1904: true,
            ..one_sheet(r#"<sheetData><row r="1"><c r="A1" s="0"><v>100</v></c></row></sheetData>"#)
        };
        let doc = parse(&wb.build()).unwrap();
        assert_eq!(texts(first_table(&doc)), vec![vec!["1904-04-10"]]);
    }

    #[test]
    fn merge_extends_past_the_populated_range() {
        // Issue #8: the only populated cell anchors F1:O3, so the grid must
        // widen to the merge's full 3x10 extent instead of clipping to the
        // 1x1 populated range.
        let wb = one_sheet(
            r#"<sheetData><row r="1"><c r="F1" t="inlineStr"><is><t>wide</t></is></c></row></sheetData><mergeCells count="1"><mergeCell ref="F1:O3"/></mergeCells>"#,
        );
        let doc = parse(&wb.build()).unwrap();
        let table = first_table(&doc);
        assert_eq!(table.grid.len(), 3);
        assert_eq!(table.grid[1].len(), 10);
        let CellSlot::Origin(cell) = &table.grid[0][0] else {
            panic!("expected the merge origin at (0,0)");
        };
        assert_eq!((cell.col_span, cell.row_span), (10, 3));
        assert_eq!(covered_count(table), 29);
    }

    /// Minimal sheet with a used range at D11:E12 and the given merge.
    fn sheet_with_merge(merge_ref: &str) -> Vec<u8> {
        one_sheet(&format!(
            r#"<sheetData><row r="11"><c r="D11" t="inlineStr"><is><t>x</t></is></c><c r="E11" t="inlineStr"><is><t>y</t></is></c></row><row r="12"><c r="D12" t="inlineStr"><is><t>z</t></is></c><c r="E12" t="inlineStr"><is><t>w</t></is></c></row></sheetData><mergeCells count="1"><mergeCell ref="{merge_ref}"/></mergeCells>"#
        ))
        .build()
    }

    #[test]
    fn merge_inside_the_used_range_covers_cells() {
        let doc = parse(&sheet_with_merge("D11:E11")).unwrap();
        assert_eq!(covered_count(first_table(&doc)), 1);
    }

    #[test]
    fn merge_outside_the_used_columns_is_ignored() {
        // The merge overlaps the used rows but not the used columns; it
        // must neither cover cells nor drag the grid out to column A.
        let doc = parse(&sheet_with_merge("A1:B12")).unwrap();
        let table = first_table(&doc);
        assert_eq!(covered_count(table), 0, "out-of-range merge must not cover cells");
        assert_eq!(table.grid[0].len(), 2);
    }

    #[test]
    fn hidden_rows_columns_and_sheets_are_omitted() {
        // Hidden content is invisible to someone opening the workbook, so
        // passing it on would make it look authoritative. One visible sheet
        // remains, so no sheet heading is emitted either.
        let visible = r#"<cols><col min="2" max="2" hidden="1"/></cols><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>a</t></is></c><c r="B1" t="inlineStr"><is><t>hidden col</t></is></c><c r="C1" t="inlineStr"><is><t>c</t></is></c></row><row r="2" hidden="1"><c r="A2" t="inlineStr"><is><t>hidden row</t></is></c></row><row r="3"><c r="A3" t="inlineStr"><is><t>d</t></is></c></row></sheetData>"#;
        let secret = r#"<sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>secret</t></is></c></row></sheetData>"#;
        let wb = Wb {
            sheets: vec![("Shown", "", visible), ("Secret", "hidden", secret)],
            ..Wb::default()
        };
        let doc = parse(&wb.build()).unwrap();
        assert_eq!(doc.blocks.len(), 1, "hidden sheet must add no heading and no table");
        assert_eq!(texts(first_table(&doc)), vec![vec!["a", "c"], vec!["d", ""]]);
    }

    #[test]
    fn merges_remap_across_hidden_columns() {
        // Dropping a hidden column renumbers the grid: a merge spanning it
        // comes out one column narrower, not applied at stale indices.
        let wb = one_sheet(
            r#"<cols><col min="2" max="2" hidden="1"/></cols><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>m</t></is></c><c r="D1" t="inlineStr"><is><t>x</t></is></c></row></sheetData><mergeCells count="1"><mergeCell ref="A1:C1"/></mergeCells>"#,
        );
        let doc = parse(&wb.build()).unwrap();
        let table = first_table(&doc);
        let CellSlot::Origin(cell) = &table.grid[0][0] else {
            panic!("expected the merge origin at (0,0)");
        };
        assert_eq!((cell.col_span, cell.row_span), (2, 1));
        assert_eq!(table.grid[0].len(), 3);
    }

    #[test]
    fn merge_origin_in_a_hidden_row_keeps_its_content() {
        // The origin row is hidden but the merge survives: its value moves
        // to the first surviving position it covers instead of being lost.
        let wb = one_sheet(
            r#"<sheetData><row r="1" hidden="1"><c r="A1" t="inlineStr"><is><t>kept</t></is></c></row><row r="2"/><row r="3"><c r="B3" t="inlineStr"><is><t>x</t></is></c></row></sheetData><mergeCells count="1"><mergeCell ref="A1:A3"/></mergeCells>"#,
        );
        let doc = parse(&wb.build()).unwrap();
        let table = first_table(&doc);
        let CellSlot::Origin(cell) = &table.grid[0][0] else {
            panic!("expected the merge origin at (0,0)");
        };
        assert_eq!(cell.row_span, 2);
        assert_eq!(texts(table)[0][0], "kept");
    }

    #[test]
    fn the_grid_budget_spans_the_whole_workbook() {
        // Two cells at opposite corners describe the whole sheet, so the
        // extent has to be charged before any position is built.
        let wb = one_sheet(
            r#"<sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>a</t></is></c></row><row r="1048576"><c r="XFD1048576" t="inlineStr"><is><t>b</t></is></c></row></sheetData>"#,
        );
        let err = parse(&wb.build()).unwrap_err();
        assert!(
            matches!(err, ConvertError::ResourceLimit { limit: "max_grid_slots", .. }),
            "got {err:?}"
        );

        // Sheets accumulate, so a pile of them cannot each sit under the cap.
        let half = r#"<sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>a</t></is></c></row><row r="150000"><c r="P150000" t="inlineStr"><is><t>b</t></is></c></row></sheetData>"#;
        let wb = Wb { sheets: vec![("A", "", half), ("B", "", half)], ..Wb::default() };
        let err = parse(&wb.build()).unwrap_err();
        assert!(
            matches!(err, ConvertError::ResourceLimit { limit: "max_grid_slots", .. }),
            "got {err:?}"
        );
    }

    const DATE_ONLY: DateParts = DateParts {
        date: true,
        time: false,
        elapsed: false,
        seconds: false,
        span: (Unit::Hour, Unit::Hour),
        second_fraction: 0,
    };

    #[test]
    fn the_fictitious_leap_day_keeps_its_own_value() {
        // Serial 60 is 1900-02-29, a day that never existed, and mapping it
        // onto a real date would make it indistinguishable from serial 59.
        assert_eq!(render_serial(59.0, DATE_ONLY, false), "1900-02-28");
        assert_eq!(render_serial(60.0, DATE_ONLY, false), "60");
        assert_eq!(render_serial(61.0, DATE_ONLY, false), "1900-03-01");
        // A value late on serial 59 still belongs to its own day.
        assert_eq!(render_serial(59.9999999, DATE_ONLY, false), "1900-02-28");
        // The 1904 system has no such day.
        assert_eq!(render_serial(60.0, DATE_ONLY, true), "1904-03-01");
    }

    #[test]
    fn a_sub_day_serial_keeps_the_clock_a_combined_format_names() {
        let both = DateParts { date: true, time: true, seconds: true, ..DateParts::default() };
        assert_eq!(render_serial(0.5, both, false), "12:00:00");
        // A clock without seconds drops them (markitai).
        let minutes = DateParts { seconds: false, ..both };
        assert_eq!(render_serial(46_095.396_5, minutes, false), "2026-03-14 09:30");
        assert_eq!(render_serial(0.396_5, minutes, false), "09:30");
        // A date-only format still has nothing to show but the number.
        assert_eq!(render_serial(0.5, DATE_ONLY, false), "0.5");
    }

    #[test]
    fn fractional_elapsed_formats_preserve_precision_and_carry_across_units() {
        for (code, seconds, expected) in [
            ("[h]:mm:ss.00", 95445.25, "26:30:45.25"),
            ("[h]:mm:ss.00", 95445.0, "26:30:45.00"),
            ("[h]:mm:ss.00", 95445.999, "26:30:46.00"),
            ("[h]:mm:ss.00", 59.999, "0:01:00.00"),
            ("[h]:mm:ss.00", 3599.999, "1:00:00.00"),
            ("[h]:mm:ss.00", 86399.999, "24:00:00.00"),
            ("[h]:mm:ss.00", 5.255, "0:00:05.26"),
            ("[h]:mm:ss.00", -95445.25, "-26:30:45.25"),
            ("[h]:mm:ss.00", 0.0, "0:00:00.00"),
            ("[m]:ss.00", 3599.999, "60:00.00"),
            ("[m]:ss.000", 95445.125, "1590:45.125"),
            ("[s].00", 95445.25, "95445.25"),
            ("[ss].00", -95445.25, "-95445.25"),
            ("[s].0", 0.0, "0.0"),
            // Existing formats stay on their previous whole-second path.
            ("[h]", 95445.25, "26"),
            ("[m]", 95445.25, "1590"),
            ("[s]", 95445.25, "95445"),
            ("[h]:mm", 95445.25, "26:30"),
            (r"[h]:mm:ss\.00", 95445.25, "26:30:45"),
            (r#"[h]:mm:ss".00""#, 95445.25, "26:30:45"),
        ] {
            let format = CellFormat::Fmt(Rc::new(NumberFormat::parse(code).unwrap()));
            assert_eq!(
                render_number(&format, seconds / 86400.0, false),
                expected,
                "{code}, {seconds}"
            );
        }
        let date = CellFormat::Fmt(Rc::new(NumberFormat::parse("yyyy-mm-dd hh:mm:ss.00").unwrap()));
        assert_eq!(render_number(&date, 46095.5, false), "2026-03-14 12:00:00");
    }

    #[test]
    fn fractional_duration_survives_the_actual_xlsx_styles_and_cell_reader() {
        let wb = Wb {
            styles: Some(
                r#"<numFmts><numFmt numFmtId="164" formatCode="[h]:mm:ss.00"/></numFmts><cellXfs><xf numFmtId="164"/></cellXfs>"#,
            ),
            ..one_sheet(
                r#"<sheetData><row r="1"><c r="A1" s="0"><v>1.1046903935185185185</v></c><c r="B1" s="0"><v>-1.1046903935185185185</v></c><c r="C1" s="0"><v>0</v></c></row></sheetData>"#,
            )
        };
        let doc = parse(&wb.build()).unwrap();
        assert_eq!(
            texts(first_table(&doc)),
            vec![vec!["26:30:45.25", "-26:30:45.25", "0:00:00.00"]]
        );
    }

    #[test]
    fn an_elapsed_span_shows_the_seconds_its_format_names() {
        // markitai: 27 hours 5 minutes, as `[h]:mm` and `[h]:mm:ss` show it.
        let serial = (27.0 * 60.0 + 5.0) / 1_440.0;
        let span = DateParts {
            date: false,
            time: true,
            elapsed: true,
            seconds: true,
            span: (Unit::Hour, Unit::Second),
            second_fraction: 0,
        };
        assert_eq!(render_serial(serial, span, false), "27:05:00");
        let minutes = DateParts { seconds: false, span: (Unit::Hour, Unit::Minute), ..span };
        assert_eq!(render_serial(serial, minutes, false), "27:05");
        // markitai: `[mm]:ss` and `[s]` carry the whole span in their unit.
        let total = DateParts { span: (Unit::Minute, Unit::Second), ..span };
        assert_eq!(render_serial(serial, total, false), "1625:00");
        let total = DateParts { span: (Unit::Second, Unit::Second), ..span };
        assert_eq!(render_serial(serial, total, false), "97500");
    }

    const VML_REL: &str =
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/vmlDrawing";

    /// A legacy drawing with a checked captioned box over B1, an unchecked
    /// bare one over C1, a hidden one over D1, and a cell note.
    const VML: &str = r##"<xml xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:x="urn:schemas-microsoft-com:office:excel">
        <v:shape id="_x0000_s1025" type="#_x0000_t201" style="position:absolute;margin-left:1pt">
          <v:textbox>
            <div style="text-align:left"><font face="Tahoma">Roof</font></div>
          </v:textbox>
          <x:ClientData ObjectType="Checkbox"><x:Anchor>1, 5, 0, 2, 2, 10, 1, 1</x:Anchor><x:Checked>1</x:Checked></x:ClientData>
        </v:shape>
        <v:shape id="_x0000_s1026" type="#_x0000_t201" style="position:absolute">
          <v:textbox><div><font></font></div></v:textbox>
          <x:ClientData ObjectType="Checkbox"><x:Anchor>2, 5, 0, 2, 3, 10, 1, 1</x:Anchor></x:ClientData>
        </v:shape>
        <v:shape id="_x0000_s1027" type="#_x0000_t201" style="position:absolute;visibility:hidden">
          <x:ClientData ObjectType="Checkbox"><x:Anchor>3, 5, 0, 2, 4, 10, 1, 1</x:Anchor><x:Checked>1</x:Checked></x:ClientData>
        </v:shape>
        <v:shape id="_x0000_s1028" type="#_x0000_t202" style="position:absolute">
          <v:textbox><div>a note</div></v:textbox>
          <x:ClientData ObjectType="Note"><x:Anchor>4, 5, 0, 2, 5, 10, 1, 1</x:Anchor></x:ClientData>
        </v:shape>
        </xml>"##;

    #[test]
    fn form_control_checkboxes_land_in_their_anchor_cell() {
        let rels = format!(
            r#"<?xml version="1.0"?><Relationships xmlns="{PKG_RELS}"><Relationship Id="rId1" Type="{VML_REL}" Target="../drawings/vmlDrawing1.vml"/></Relationships>"#
        );
        let wb = Wb {
            sheets: vec![(
                "S",
                "",
                r#"<sheetData><row r="1"><c r="A1" t="str"><v>14</v></c><c r="B1" t="str"><v>L/R</v></c></row></sheetData><legacyDrawing r:id="rId1"/>"#,
            )],
            extra: vec![
                ("xl/worksheets/_rels/sheet1.xml.rels", &rels),
                ("xl/drawings/vmlDrawing1.vml", VML),
            ],
            ..Wb::default()
        };
        let doc = parse(&wb.build()).unwrap();
        assert_eq!(texts(first_table(&doc)), vec![vec!["14", "L/R [x] Roof", "[ ]"]]);
    }
}
