//! Count workbook sheets independently of PDF pagination, with bounded ZIP/XML reads.

use super::{Result, failure};
use quick_xml::{events::Event, name::ResolveResult};

const MAX_XML: u64 = 32 * 1024 * 1024;
const ODF_OFFICE: &[u8] = b"urn:oasis:names:tc:opendocument:xmlns:office:1.0";
const ODF_TABLE: &[u8] = b"urn:oasis:names:tc:opendocument:xmlns:table:1.0";
const OOXML: &[u8] = b"http://schemas.openxmlformats.org/spreadsheetml/2006/main";
const STRICT_OOXML: &[u8] = b"http://purl.oclc.org/ooxml/spreadsheetml/main";

fn part(bytes: &[u8], name: &str) -> Result<Vec<u8>> {
    crate::opc::Zip::open(bytes, crate::opc::MAX_ENTRIES)
        .and_then(|mut zip| zip.read(name, MAX_XML))
        .map_err(|e| failure(&format!("workbook package: {e}")))?
        .ok_or_else(|| failure("workbook sheet-index part is missing"))
}

pub(super) fn xlsx(bytes: &[u8]) -> Result<usize> {
    count(&part(bytes, "xl/workbook.xml")?, false)
}

pub(super) fn ods(bytes: &[u8]) -> Result<usize> {
    count(&part(bytes, "content.xml")?, true)
}

fn count(xml: &[u8], odf: bool) -> Result<usize> {
    let mut reader = quick_xml::NsReader::from_reader(xml);
    let mut depth = 0usize;
    let mut container = None;
    let mut saw_container = false;
    let mut count = 0usize;
    loop {
        let event = reader
            .read_event()
            .map_err(|_| failure("invalid workbook sheet-index XML"))?;
        let empty = matches!(event, Event::Empty(_));
        match event {
            Event::Start(element) | Event::Empty(element) => {
                if depth >= 128 {
                    return Err(failure("workbook XML nesting exceeds 128"));
                }
                let namespace = reader.resolver().resolve_element(element.name()).0;
                let local = element.local_name();
                let is_container = if odf {
                    local.as_ref() == b"spreadsheet"
                        && matches!(namespace, ResolveResult::Bound(uri) if uri.as_ref() == ODF_OFFICE)
                } else {
                    local.as_ref() == b"sheets"
                        && matches!(namespace, ResolveResult::Bound(uri) if matches!(uri.as_ref(), OOXML | STRICT_OOXML))
                };
                if is_container {
                    if saw_container {
                        return Err(failure("workbook has multiple sheet containers"));
                    }
                    saw_container = true;
                    if !empty {
                        container = Some(depth);
                    }
                }
                let is_sheet = if odf {
                    local.as_ref() == b"table"
                        && matches!(namespace, ResolveResult::Bound(uri) if uri.as_ref() == ODF_TABLE)
                } else {
                    local.as_ref() == b"sheet"
                        && matches!(namespace, ResolveResult::Bound(uri) if matches!(uri.as_ref(), OOXML | STRICT_OOXML))
                };
                if container.is_some_and(|parent| depth == parent + 1) && is_sheet {
                    count += 1;
                    if count > super::MAX_PAGES {
                        return Err(failure("workbook exceeds 1,000 sheets"));
                    }
                }
                if !empty {
                    depth += 1;
                }
            }
            Event::End(_) => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| failure("unbalanced workbook XML"))?;
                if container == Some(depth) {
                    container = None;
                }
            }
            Event::DocType(_) => return Err(failure("workbook document types are not accepted")),
            Event::Eof => break,
            _ => {}
        }
    }
    if count == 0 || depth != 0 {
        return Err(failure("workbook contains no complete sheets"));
    }
    Ok(count)
}

pub(super) fn validate_pdf(bytes: &[u8], sheets: usize) -> Result<usize> {
    let pages = super::validate_pdf(bytes, None)?;
    if pages != sheets {
        return Err(failure(
            "complete-sheet PDF page count does not match every workbook sheet, including hidden and empty sheets",
        ));
    }
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_workbooks_count_hidden_and_empty_sheets_without_counting_cells() {
        assert_eq!(
            xlsx(include_bytes!("fixtures/whole-workbook.xlsx")).unwrap(),
            4
        );
        assert_eq!(
            ods(include_bytes!("fixtures/whole-workbook.ods")).unwrap(),
            4
        );
        let nested = br#"<o:document-content xmlns:o="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:t="urn:oasis:names:tc:opendocument:xmlns:table:1.0"><o:body><o:spreadsheet><t:table><t:table-row><t:table-cell><t:table/></t:table-cell></t:table-row></t:table><t:table/></o:spreadsheet></o:body></o:document-content>"#;
        assert_eq!(count(nested, true).unwrap(), 2);
    }

    #[test]
    fn invalid_or_excessive_sheet_indexes_are_explicit_errors() {
        for xml in [
            "<!DOCTYPE workbook><workbook/>",
            "<workbook xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><sheets/></workbook>",
            "<workbook><sheets><sheet/></sheets></workbook>",
            "<workbook xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><sheets><sheet/></sheets><sheets><sheet/></sheets></workbook>",
            "<workbook xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><sheets><sheet/></workbook>",
        ] {
            assert!(count(xml.as_bytes(), false).is_err(), "{xml}");
        }
        let xml = format!(
            "<workbook xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><sheets>{}</sheets></workbook>",
            "<sheet/>".repeat(1001)
        );
        assert!(
            count(xml.as_bytes(), false)
                .unwrap_err()
                .to_string()
                .contains("1,000")
        );
        let xml = format!("{}{}", "<a>".repeat(129), "</a>".repeat(129));
        assert!(
            count(xml.as_bytes(), false)
                .unwrap_err()
                .to_string()
                .contains("128")
        );
        assert!(xlsx(b"not a ZIP").is_err());
    }

    #[test]
    fn strict_ooxml_sheets_are_counted() {
        assert_eq!(count(br#"<workbook xmlns="http://purl.oclc.org/ooxml/spreadsheetml/main"><sheets><sheet/><sheet/></sheets></workbook>"#, false).unwrap(), 2);
    }

    #[test]
    fn truncated_export_cannot_report_complete_sheet_success() {
        assert_eq!(validate_pdf(&super::super::tests::pdf(4), 4).unwrap(), 4);
        assert!(
            validate_pdf(&super::super::tests::pdf(3), 4)
                .unwrap_err()
                .to_string()
                .contains("every workbook sheet")
        );
    }
}
