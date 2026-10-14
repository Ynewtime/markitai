//! Line breaks inside the cells of sheets and documents, built in the test:
//! a pipe table's row is one line, so every break in a cell is one `<br>`.

use super::*;
use std::io::{Cursor, Write};

/// A one-sheet workbook whose rows hold shared strings, as Excel writes a
/// cell typed with Alt+Enter (a line feed in the string).
fn xlsx(rows: &[&[&str]]) -> Vec<u8> {
    let mut strings = Vec::new();
    let mut sheet = String::new();
    for (row, values) in rows.iter().enumerate() {
        sheet.push_str(&format!("<row r=\"{}\">", row + 1));
        for (column, value) in values.iter().enumerate() {
            let reference = format!("{}{}", char::from(b'A' + column as u8), row + 1);
            sheet.push_str(&format!(
                "<c r=\"{reference}\" t=\"s\"><v>{}</v></c>",
                strings.len()
            ));
            strings.push(format!(
                "<si><t xml:space=\"preserve\">{}</t></si>",
                value.replace('&', "&amp;").replace('<', "&lt;")
            ));
        }
        sheet.push_str("</row>");
    }
    let main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
    let relationships = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
    let parts = [
        (
            "[Content_Types].xml",
            r#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>"#.to_owned(),
        ),
        (
            "_rels/.rels",
            format!(r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="{relationships}/officeDocument" Target="xl/workbook.xml"/></Relationships>"#),
        ),
        (
            "xl/workbook.xml",
            format!(r#"<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{main}" xmlns:r="{relationships}"><sheets><sheet name="Plan" sheetId="1" r:id="rId1"/></sheets></workbook>"#),
        ),
        (
            "xl/_rels/workbook.xml.rels",
            format!(r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="{relationships}/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="{relationships}/sharedStrings" Target="sharedStrings.xml"/></Relationships>"#),
        ),
        (
            "xl/worksheets/sheet1.xml",
            format!(r#"<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="{main}"><sheetData>{sheet}</sheetData></worksheet>"#),
        ),
        (
            "xl/sharedStrings.xml",
            format!(
                r#"<?xml version="1.0" encoding="UTF-8"?><sst xmlns="{main}" count="{0}" uniqueCount="{0}">{1}</sst>"#,
                strings.len(),
                strings.concat()
            ),
        ),
    ];
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, body) in parts {
        zip.start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(body.as_bytes()).unwrap();
    }
    zip.finish().unwrap().into_inner()
}

#[test]
fn a_sheet_cell_typed_over_several_lines_stays_one_table_row() {
    let doc = extract(
        &xlsx(&[
            &["Task", "Notes"],
            &["Plant bulbs", "Bring gloves\nand a trowel"],
            &["Water beds", "Mornings\r\n\r\n\nonly | if dry"],
            &["Rake leaves", "\nInto bags\u{2028}by the shed\u{2029}\n"],
            &["Literal", "type <br> here"],
            &["Sweep", "last row"],
        ]),
        "xlsx",
    )
    .unwrap();
    assert_eq!(
        doc.markdown,
        "## Plan\n\
         | Task | Notes |\n\
         | --- | --- |\n\
         | Plant bulbs | Bring gloves<br>and a trowel |\n\
         | Water beds | Mornings<br>only \\| if dry |\n\
         | Rake leaves | Into bags<br>by the shed |\n\
         | Literal | type \\<br> here |\n\
         | Sweep | last row |"
    );
}

#[test]
fn every_kind_of_line_break_in_a_cell_is_one_br() {
    for separator in [
        "\n", "\r\n", "\r", "\u{0b}", "\u{85}", "\u{2028}", "\u{2029}",
    ] {
        let text = format!("Bring gloves{separator}and a trowel");
        assert_eq!(
            table_cell(&text),
            "Bring gloves<br>and a trowel",
            "{separator:?}"
        );
        assert_eq!(
            super::super::text::cell(&text),
            "Bring gloves<br>and a trowel",
            "{separator:?}"
        );
    }
}

#[test]
fn excel_s_escaped_control_characters_are_decoded_and_its_breaks_are_br() {
    let doc = extract(
        &xlsx(&[
            &["Task", "Notes"],
            &["Rake leaves", "Into bags_x000B_by the shed"],
            &["Water", "Mornings_x000D_\nonly"],
            &["Literal", "_x005F_x000D_ stays, _xZZZZ_ too"],
        ]),
        "xlsx",
    )
    .unwrap();
    assert_eq!(
        doc.markdown,
        "## Plan\n\
         | Task | Notes |\n\
         | --- | --- |\n\
         | Rake leaves | Into bags<br>by the shed |\n\
         | Water | Mornings<br>only |\n\
         | Literal | \\_x000D\\_ stays, \\_xZZZZ\\_ too |"
    );
}
