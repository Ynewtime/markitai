//! Numbers tables from bounded ZIP or directory packages.

mod container;
#[cfg(test)]
mod tests;

use crate::{Document, Error, Result};
use iwork::table::{Cell, CellFormat, CellValue, Decimal, Table};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

const MAX_TEXT: usize = 1024 * 1024;
const MAX_OUTPUT: usize = 32 * 1024 * 1024;

fn error(message: impl std::fmt::Display) -> Error {
    Error::Conversion(format!("Numbers conversion failed: {message}"))
}

pub(super) fn extract(bytes: &[u8]) -> Result<Document> {
    let (archive, expected_tables) = container::open(bytes)?;
    render(archive, expected_tables)
}

pub(super) fn extract_directory(path: &std::path::Path) -> Result<Document> {
    let (archive, expected_tables) = container::open_directory(path)?;
    render(archive, expected_tables)
}

fn render(archive: iwork::Document, expected_tables: usize) -> Result<Document> {
    let tables = archive.tables();
    if tables.len() != expected_tables || tables.is_empty() {
        return Err(error(
            "no complete table model was found, or a table could not be decoded",
        ));
    }
    let sheets = archive.sheets();
    if sheets.is_empty() || sheets.len() > 1024 {
        return Err(error("missing or excessive sheet records"));
    }
    let by_id: HashMap<_, _> = tables
        .iter()
        .map(|table| (table.identifier, table))
        .collect();
    let mut seen = HashSet::new();
    let mut sheet_ids = HashSet::new();
    let mut output = Document::default();
    let mut metadata = Vec::new();
    let mut flags = Flags::default();
    for sheet in &sheets {
        if !sheet_ids.insert(sheet.identifier) {
            return Err(error("duplicate sheet reference"));
        }
        heading(&mut output.markdown, 1, &sheet.name)?;
        for id in &sheet.drawables {
            let Some(table) = by_id.get(id) else {
                flags.canvas = true;
                continue;
            };
            if !seen.insert(*id) {
                return Err(error("a table is referenced more than once"));
            }
            append_table(&mut output.markdown, table, &mut flags)?;
            metadata.push(table_metadata(table, &sheet.name));
        }
    }
    // Do not silently drop a decoded table with missing canvas membership.
    for table in &tables {
        if seen.insert(table.identifier) {
            flags.unplaced = true;
            append_table(&mut output.markdown, table, &mut flags)?;
            metadata.push(table_metadata(table, table.sheet.as_deref().unwrap_or("")));
        }
    }
    output
        .metadata
        .insert("sheet_count".into(), sheets.len().into());
    output
        .metadata
        .insert("table_count".into(), tables.len().into());
    output
        .metadata
        .insert("numbers_tables".into(), metadata.into());
    output
        .metadata
        .insert("numbers_value_mode".into(), "saved-values".into());
    if flags.formulas {
        output.warnings.push(
            "Numbers formulas use their last saved values; formulas are not recalculated.".into(),
        );
    }
    if flags.formats {
        output.warnings.push("Numbers display formatting is normalized: basic decimal, percentage and currency formats are supported; rich-text styling, bullets, locale, custom formats, date patterns and interactive controls may differ from Numbers.".into());
    }
    if flags.hidden {
        output.warnings.push(
            "Numbers hidden or filtered rows and columns are included in their stored order."
                .into(),
        );
    }
    if flags.canvas {
        output.warnings.push("Numbers non-table canvas objects (including charts, shapes, text boxes and images) are not extracted.".into());
    }
    if flags.unplaced {
        output.warnings.push(
            "Numbers tables missing a sheet reference are appended after the listed sheets.".into(),
        );
    }
    Ok(output)
}

#[derive(Default)]
struct Flags {
    formulas: bool,
    formats: bool,
    hidden: bool,
    canvas: bool,
    unplaced: bool,
}

fn table_metadata(table: &Table, sheet: &str) -> serde_json::Value {
    json!({
        "sheet":sheet, "name":table.name, "rows":table.rows, "columns":table.columns,
        "header_rows":table.header_rows, "header_columns":table.header_columns,
        "footer_rows":table.footer_rows,
        "cached_formula_cells":table.cells().iter().filter(|c| c.has_formula).count(),
        "merges":table.merges.iter().map(|m| json!({"row":m.row + 1,"column":m.column + 1,"rows":m.rows,"columns":m.columns})).collect::<Vec<_>>()
    })
}

fn heading(out: &mut String, level: usize, name: &str) -> Result<()> {
    if name.len() > 4096 {
        return Err(error("sheet or table name exceeds the size limit"));
    }
    for _ in 0..level {
        out.push('#');
    }
    out.push(' ');
    escaped(out, name, false);
    out.push_str("\n\n");
    limit(out)
}

fn append_table(out: &mut String, table: &Table, flags: &mut Flags) -> Result<()> {
    if !table.problems.is_empty() {
        return Err(error("table contains unsupported or damaged cell storage"));
    }
    let area = table
        .rows
        .checked_mul(table.columns)
        .filter(|n| *n <= container::MAX_CELLS)
        .ok_or_else(|| error("table dimensions exceed the limit"))?;
    if table.cells().len() > area
        || table.header_rows as usize > table.rows
        || table.header_columns as usize > table.columns
        || table.footer_rows as usize > table.rows
    {
        return Err(error("inconsistent table dimensions"));
    }
    let mut previous = None;
    for cell in table.cells() {
        let position = (cell.row, cell.column);
        if cell.row >= table.rows || cell.column >= table.columns || previous == Some(position) {
            return Err(error("duplicate or out-of-range table cell"));
        }
        previous = Some(position);
        if matches!(&cell.value, CellValue::Text(text) | CellValue::RichText(text) if text.len() > MAX_TEXT)
        {
            return Err(error("cell text exceeds the 1 MiB limit"));
        }
        if matches!(cell.value, CellValue::Unknown(_)) {
            return Err(error("unsupported cell value type"));
        }
        if let Some(key) = cell.record.string_id
            && !table.side_tables().strings.entries.contains_key(&key)
        {
            return Err(error("cell references missing text"));
        }
    }
    flags.hidden |= table
        .row_extents
        .iter()
        .chain(&table.column_extents)
        .any(|e| e.hidden())
        || table.hidden_rows != 0
        || table.hidden_columns != 0
        || table.filtered_rows != 0;
    heading(out, 2, &table.name)?;
    if table.merges.is_empty() && table.header_rows == 1 {
        markdown_table(out, table, flags)?;
    } else {
        html_table(out, table, flags, area)?;
    }
    out.push('\n');
    limit(out)
}

fn markdown_table(out: &mut String, table: &Table, flags: &mut Flags) -> Result<()> {
    for row in 0..table.rows {
        out.push('|');
        for column in 0..table.columns {
            out.push(' ');
            if let Some(cell) = table.cell(row, column) {
                escaped(out, &cell_text(cell, table, flags)?, false);
            }
            out.push_str(" |");
            limit(out)?;
        }
        out.push('\n');
        if row == 0 {
            out.push('|');
            for _ in 0..table.columns {
                out.push_str(" --- |");
            }
            out.push('\n');
        }
        limit(out)?;
    }
    Ok(())
}

fn html_table(out: &mut String, table: &Table, flags: &mut Flags, area: usize) -> Result<()> {
    // One bounded bitmap prevents quadratic scans of all merges per cell.
    let mut covered = vec![false; area];
    let mut anchors = HashMap::new();
    for merge in &table.merges {
        let end_row = merge
            .row
            .checked_add(merge.rows)
            .filter(|end| *end <= table.rows);
        let end_column = merge
            .column
            .checked_add(merge.columns)
            .filter(|end| *end <= table.columns);
        let (Some(end_row), Some(end_column)) = (end_row, end_column) else {
            return Err(error("merged range falls outside the table"));
        };
        if merge.rows == 0 || merge.columns == 0 {
            return Err(error("empty merged range"));
        }
        for row in merge.row..end_row {
            for column in merge.column..end_column {
                let slot = &mut covered[row * table.columns + column];
                if *slot {
                    return Err(error("overlapping merged ranges"));
                }
                *slot = true;
            }
        }
        anchors.insert((merge.row, merge.column), merge);
    }
    out.push_str("<table>\n");
    for row in 0..table.rows {
        out.push_str("<tr>");
        for column in 0..table.columns {
            let merge = anchors.get(&(row, column));
            if covered[row * table.columns + column] && merge.is_none() {
                continue;
            }
            let tag = if row < table.header_rows as usize || column < table.header_columns as usize
            {
                "th"
            } else {
                "td"
            };
            write!(out, "<{tag}").unwrap();
            if let Some(merge) = merge {
                if merge.rows > 1 {
                    write!(out, " rowspan=\"{}\"", merge.rows).unwrap();
                }
                if merge.columns > 1 {
                    write!(out, " colspan=\"{}\"", merge.columns).unwrap();
                }
            }
            out.push('>');
            if let Some(cell) = table.cell(row, column) {
                escaped(out, &cell_text(cell, table, flags)?, true);
            }
            write!(out, "</{tag}>").unwrap();
            limit(out)?;
        }
        out.push_str("</tr>\n");
        limit(out)?;
    }
    out.push_str("</table>\n");
    Ok(())
}

fn cell_text(cell: &Cell, table: &Table, flags: &mut Flags) -> Result<String> {
    flags.formulas |= cell.has_formula;
    flags.formats |= !matches!(cell.format, CellFormat::Automatic | CellFormat::Text);
    flags.formats |= matches!(cell.value, CellValue::RichText(_));
    let format = cell
        .record
        .applicable_format()
        .and_then(|slot| cell.record.format_id_in(slot))
        .and_then(|id| table.side_tables().formats.entries.get(&id))
        .and_then(|entry| entry.format.as_ref());
    let decimals = format
        .and_then(|f| f.varint(2))
        .filter(|n| *n <= 12)
        .map(|n| n as usize);
    match &cell.value {
        CellValue::Number(value) | CellValue::Currency(value) => {
            // Preserve decimal128's integer mantissa; do not round through f64.
            let mut value = *value;
            if cell.format == CellFormat::Percentage {
                value.exponent += 2;
            }
            let mut text = decimal(value, decimals);
            if cell.format == CellFormat::Percentage {
                text.push('%');
            }
            if cell.format == CellFormat::Currency || matches!(cell.value, CellValue::Currency(_)) {
                if let Some(code) = format
                    .and_then(|f| f.bytes(3))
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .filter(|s| s.len() == 3 && s.bytes().all(|b| b.is_ascii_alphabetic()))
                {
                    text = format!("{code} {text}");
                } else {
                    // The numeric value remains exact; guessing a currency is unsafe.
                    flags.formats = true;
                }
            }
            Ok(text)
        }
        CellValue::Date(value) | CellValue::Duration(value) if !value.is_finite() => {
            Err(error("non-finite date or duration"))
        }
        CellValue::Date(value) => {
            // Stored dates carry no timezone. Do not attach the upstream
            // convenience formatter's artificial UTC suffix.
            if !(-63_114_076_800.0..=252_423_993_599.0).contains(value) {
                return Err(error("date lies outside the supported calendar range"));
            }
            let mut text = iwork::table::format_date(*value);
            text.pop();
            Ok(text)
        }
        CellValue::Error => Ok("#ERROR!".into()),
        _ => Ok(cell.value.to_text()),
    }
}

fn decimal(value: Decimal, places: Option<usize>) -> String {
    let exact = || {
        if !(-40..=32).contains(&value.exponent) {
            format!("{}e{}", value.mantissa, value.exponent)
        } else {
            value.to_string()
        }
    };
    let Some(places) = places else {
        return exact();
    };
    let mut digits = value.mantissa.unsigned_abs().to_string();
    let shift = i64::from(value.exponent) + places as i64;
    if !(-80..=80).contains(&shift) {
        return exact();
    }
    if shift >= 0 {
        digits.extend(std::iter::repeat_n('0', shift as usize));
    } else {
        let remove = (-shift) as usize;
        if remove > digits.len() {
            digits = "0".into();
        } else {
            let keep = digits.len() - remove;
            let round = digits.as_bytes()[keep] >= b'5';
            digits.truncate(keep);
            if digits.is_empty() {
                digits.push('0');
            }
            if round {
                let mut bytes = digits.into_bytes();
                let mut carry = true;
                for byte in bytes.iter_mut().rev() {
                    if *byte == b'9' {
                        *byte = b'0';
                    } else {
                        *byte += 1;
                        carry = false;
                        break;
                    }
                }
                if carry {
                    bytes.insert(0, b'1');
                }
                digits = String::from_utf8(bytes).unwrap();
            }
        }
    }
    if digits.len() <= places {
        digits = format!("{}{}", "0".repeat(places + 1 - digits.len()), digits);
    }
    if places > 0 {
        digits.insert(digits.len() - places, '.');
    }
    if value.mantissa < 0 && digits.bytes().any(|b| (b'1'..=b'9').contains(&b)) {
        digits.insert(0, '-');
    }
    digits
}

/// Text written escaped for a Markdown or an HTML table: its lines joined
/// with one `<br>`, none for an empty line or at its edges (a raw line feed
/// would end a Markdown table's row).
fn escaped(out: &mut String, text: &str, html: bool) {
    let joined;
    let text = if text.contains(super::text::LINE_BREAKS) {
        joined = text
            .split(super::text::LINE_BREAKS)
            .filter(|line| !line.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        &joined
    } else {
        text
    };
    for character in text.chars() {
        match character {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\n' => out.push_str("<br>"),
            '\r' => {}
            '|' if !html => out.push_str("\\|"),
            '\\' | '*' | '_' | '`' | '[' | ']' if !html => {
                out.push('\\');
                out.push(character);
            }
            _ => out.push(character),
        }
    }
}

fn limit(out: &str) -> Result<()> {
    if out.len() > MAX_OUTPUT {
        Err(error("rendered tables exceed the 32 MiB limit"))
    } else {
        Ok(())
    }
}
