//! markitai: what a SpreadsheetML worksheet attaches to its cells beside
//! their values: hyperlinks (`hyperlinks/hyperlink`, resolved through the
//! worksheet's relationships) and notes, both the legacy comments part
//! (`comments/commentList/comment`), which Excel now calls notes, and the
//! threaded comments of Excel 365 (`ThreadedComments/threadedComment`, with
//! their authors in the workbook's persons part).
//!
//! A threaded comment is also written to the legacy part, as boilerplate
//! telling an older Excel it cannot edit it; a cell with a threaded comment
//! takes its text from the threaded part instead.

use super::xlsx::{parse_ref, parse_region, rich_text};
use crate::error::ConvertError;
use crate::package::relationships::{Relationships, TargetMode};
use crate::package::xml::{Element, ns};
use crate::package::{Package, path};
use crate::shared::text::{clean_text, collapse_ws};
use std::collections::{BTreeMap, HashMap, HashSet};

const HYPERLINK_REL: &str =
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink";
const COMMENTS_REL: &str =
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments";
const THREADED_REL: &str =
    "http://schemas.microsoft.com/office/2017/10/relationships/threadedComment";
const PERSON_REL: &str = "http://schemas.microsoft.com/office/2017/10/relationships/person";

/// Cells one hyperlink range is applied to at most; a link over a whole
/// column applies to its first cell.
const MAX_LINK_CELLS: u64 = 10_000;
/// Notes kept per sheet.
const MAX_NOTES: usize = 10_000;

/// The web and mail addresses the worksheet's cells link to. Links to a place
/// in the workbook (`location` alone, or a `#Sheet2!A1` target) and other
/// schemes keep the cell's text only.
pub(super) fn hyperlinks(worksheet: &Element, rels: &Relationships) -> HashMap<(u32, u32), String> {
    let mut out = HashMap::new();
    for link in worksheet
        .find_all(ns::SML, "hyperlinks")
        .flat_map(|links| links.find_all(ns::SML, "hyperlink"))
    {
        let Some(rel) = link.attr_qualified(ns::R, "id").and_then(|id| rels.get(id)) else {
            continue;
        };
        if rel.rel_type != HYPERLINK_REL || rel.mode != TargetMode::External {
            continue;
        }
        let target = rel.target.trim();
        let scheme = target.split_once(':').map(|(scheme, _)| scheme.to_ascii_lowercase());
        if !matches!(scheme.as_deref(), Some("http" | "https" | "mailto")) {
            continue;
        }
        let url = match link.attr_unqualified("location").filter(|l| !l.is_empty()) {
            Some(location) if !target.contains('#') => format!("{target}#{location}"),
            _ => target.to_string(),
        };
        let Some((r1, c1, r2, c2)) = link.attr_unqualified("ref").and_then(parse_region) else {
            continue;
        };
        let area = u64::from(r2 - r1 + 1) * u64::from(c2 - c1 + 1);
        if area > MAX_LINK_CELLS {
            out.entry((r1, c1)).or_insert(url);
            continue;
        }
        for r in r1..=r2 {
            for c in c1..=c2 {
                out.entry((r, c)).or_insert_with(|| url.clone());
            }
        }
    }
    out
}

/// The display names of the workbook's persons, by id, for threaded
/// comments.
pub(super) fn people(
    pkg: &mut Package,
    wb_rels: &Relationships,
    wb_part: &str,
) -> Result<HashMap<String, String>, ConvertError> {
    let mut out = HashMap::new();
    let Some(part) =
        wb_rels.first_of_type(PERSON_REL).and_then(|rel| path::resolve(wb_part, &rel.target).ok())
    else {
        return Ok(out);
    };
    if let Some(root) = pkg.optional_xml_part(&part.path)? {
        for person in root.descendants_any("person") {
            if let (Some(id), Some(name)) =
                (person.attr_unqualified("id"), person.attr_unqualified("displayName"))
            {
                out.insert(id.to_string(), one_line(name));
            }
        }
    }
    Ok(out)
}

/// The sheet's notes in reading order (row, then column, then the order the
/// parts list them), each as the text a reader sees: a legacy note's own
/// text, which Excel begins with its author's name, or a threaded comment's
/// author and text, one item per message of the thread.
pub(super) fn cell_notes(
    pkg: &mut Package,
    sheet_part: &str,
    rels: &Relationships,
    people: &HashMap<String, String>,
) -> Result<Vec<(u32, u32, String)>, ConvertError> {
    let mut notes: BTreeMap<(u32, u32), Vec<String>> = BTreeMap::new();
    let mut threaded: HashSet<(u32, u32)> = HashSet::new();
    let mut count = 0usize;
    for part in targets(rels, sheet_part, THREADED_REL) {
        let Some(root) = pkg.optional_xml_part(&part)? else {
            continue;
        };
        for comment in root.descendants_any("threadedComment") {
            let Some(at) = comment.attr_unqualified("ref").and_then(parse_ref) else {
                continue;
            };
            let text =
                comment.child_elems().find(|e| e.local == "text").map(|t| one_line(&t.text()));
            let Some(text) = text.filter(|t| !t.is_empty()) else {
                continue;
            };
            let text = match comment.attr_unqualified("personId").and_then(|id| people.get(id)) {
                Some(name) if !name.is_empty() => format!("{name}: {text}"),
                _ => text,
            };
            threaded.insert(at);
            if count < MAX_NOTES {
                notes.entry(at).or_default().push(text);
                count += 1;
            }
        }
    }
    for part in targets(rels, sheet_part, COMMENTS_REL) {
        let Some(root) = pkg.optional_xml_part(&part)? else {
            continue;
        };
        for comment in root.descendants(ns::SML, "comment") {
            let Some(at) = comment.attr_unqualified("ref").and_then(parse_ref) else {
                continue;
            };
            if threaded.contains(&at) {
                continue;
            }
            let Some(text) = comment
                .find(ns::SML, "text")
                .map(|t| one_line(&rich_text(t)))
                .filter(|t| !t.is_empty())
            else {
                continue;
            };
            if count < MAX_NOTES {
                notes.entry(at).or_default().push(text);
                count += 1;
            }
        }
    }
    Ok(notes
        .into_iter()
        .flat_map(|((row, col), texts)| texts.into_iter().map(move |text| (row, col, text)))
        .collect())
}

/// Internal targets of the sheet's relationships of one type, sorted.
fn targets(rels: &Relationships, sheet_part: &str, rel_type: &str) -> Vec<String> {
    let mut out: Vec<String> = rels
        .iter()
        .filter(|(_, r)| r.rel_type == rel_type && r.mode == TargetMode::Internal)
        .filter_map(|(_, r)| path::resolve(sheet_part, &r.target).ok())
        .map(|t| t.path)
        .collect();
    out.sort();
    out.dedup();
    out
}

/// A note's text on one line: a note box wraps, so its line breaks are not
/// structure.
fn one_line(text: &str) -> String {
    collapse_ws(&clean_text(text)).trim().to_string()
}
