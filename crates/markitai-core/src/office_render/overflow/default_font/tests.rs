use super::*;
use std::{
    collections::BTreeMap,
    io::{Cursor, Read, Write},
    time::Duration,
};
fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(10)
}
fn parts(bytes: &[u8]) -> BTreeMap<String, Vec<u8>> {
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).unwrap();
    (0..zip.len())
        .map(|i| {
            let mut f = zip.by_index(i).unwrap();
            let n = f.name().to_owned();
            let mut b = Vec::new();
            f.read_to_end(&mut b).unwrap();
            (n, b)
        })
        .collect()
}
fn archive(parts: &BTreeMap<String, Vec<u8>>) -> Vec<u8> {
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, bytes) in parts {
        zip.start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(bytes).unwrap();
    }
    zip.finish().unwrap().into_inner()
}
fn fixture() -> BTreeMap<String, Vec<u8>> {
    parts(include_bytes!("../../fixtures/whole-workbook.xlsx"))
}
fn edit(parts: &mut BTreeMap<String, Vec<u8>>, name: &str, from: &str, to: &str) {
    let original = std::str::from_utf8(&parts[name]).unwrap();
    assert!(original.contains(from));
    parts.insert(name.into(), original.replace(from, to).into_bytes());
}
fn normalized(parts: &BTreeMap<String, Vec<u8>>) -> Option<Vec<u8>> {
    normalize(&archive(parts), deadline(), 100_000_000).unwrap()
}
#[test]
fn missing_color_is_a_private_document_policy_preserving_other_parts() {
    let original = fixture();
    let source = archive(&original);
    let saved = source.clone();
    let changed = parts(
        &normalize(&source, deadline(), 100_000_000)
            .unwrap()
            .unwrap(),
    );
    assert_eq!(source, saved);
    assert_eq!(changed.len(), original.len());
    for (name, bytes) in &original {
        if name != "xl/styles.xml" {
            assert_eq!(&changed[name], bytes, "{name}");
        }
    }
    let styles = std::str::from_utf8(&original["xl/styles.xml"]).unwrap();
    assert_eq!(
        changed["xl/styles.xml"],
        styles
            .replace("</font>", &format!("{COLOR}</font>"))
            .as_bytes()
    );
    assert!(
        normalized(&changed).is_none(),
        "idempotent when colors are already declared"
    );
}
#[test]
fn every_authored_color_form_is_untouched() {
    for color in [
        "rgb=\"FFFFFFFF\"",
        "rgb=\"FFFF0000\"",
        "theme=\"4\"",
        "indexed=\"64\"",
        "auto=\"1\"",
        "auto=\"0\"",
        "tint=\"0.2\"",
        "unknown=\"author\"",
    ] {
        let mut input = fixture();
        edit(
            &mut input,
            "xl/styles.xml",
            "</font>",
            &format!("<color {color}/></font>"),
        );
        assert!(normalized(&input).is_none(), "{color}");
    }
}
#[test]
fn mixed_missing_and_explicit_fonts_preserve_author_bytes() {
    let mut input = fixture();
    edit(
        &mut input,
        "xl/styles.xml",
        "<fonts count=\"1\">",
        "<fonts count=\"2\">",
    );
    let author = "<font><color rgb=\"FFFF0000\"/><sz val=\"11\"/></font>";
    edit(
        &mut input,
        "xl/styles.xml",
        "</fonts>",
        &format!("{author}</fonts>"),
    );
    let changed = parts(&normalized(&input).unwrap());
    let styles = std::str::from_utf8(&changed["xl/styles.xml"]).unwrap();
    assert!(styles.contains(author));
    assert_eq!(styles.matches(COLOR).count(), 1);
}
#[test]
fn theme_parts_and_nonstandard_theme_relationships_skip() {
    let mut input = fixture();
    input.insert("xl/theme/theme1.xml".into(), b"<theme/>".to_vec());
    assert!(normalized(&input).is_none());
    let mut input = fixture();
    edit(
        &mut input,
        "xl/_rels/workbook.xml.rels",
        "</Relationships>",
        &format!(
            "<Relationship Id=\"theme\" Type=\"{REL}/theme\" Target=\"colors.xml\"/></Relationships>"
        ),
    );
    assert!(normalized(&input).is_none());
}
#[test]
fn conditional_rich_extended_and_table_models_skip() {
    for addition in [
        "<conditionalFormatting/>",
        "<extLst/>",
        "<tableParts/>",
        "<r><t>Rich</t></r>",
    ] {
        let mut input = fixture();
        edit(
            &mut input,
            "xl/worksheets/sheet1.xml",
            "</worksheet>",
            &format!("{addition}</worksheet>"),
        );
        assert!(normalized(&input).is_none(), "{addition}");
    }
    let mut input = fixture();
    input.insert(
        "xl/sharedStrings.xml".into(),
        format!("<sst xmlns=\"{SHEET}\"><si><r><t>Rich</t></r></si></sst>").into_bytes(),
    );
    assert!(normalized(&input).is_none());
}
#[test]
fn font_inheritance_ambiguity_and_invalid_references_skip() {
    for (from, to) in [
        ("fontId=\"0\"", "fontId=\"17\""),
        ("xfId=\"0\"", "xfId=\"9\""),
        ("<fonts count=\"1\">", "<fonts count=\"2\">"),
        ("<sz val=\"18\"/>", "<scheme val=\"minor\"/>"),
    ] {
        let mut input = fixture();
        edit(&mut input, "xl/styles.xml", from, to);
        assert!(normalized(&input).is_none(), "{to}");
    }
    let mut input = fixture();
    edit(
        &mut input,
        "xl/styles.xml",
        "<fonts count=\"1\">",
        "<fonts count=\"2\">",
    );
    edit(
        &mut input,
        "xl/styles.xml",
        "</fonts>",
        "<font><color rgb=\"FFFF0000\"/></font></fonts>",
    );
    edit(
        &mut input,
        "xl/styles.xml",
        "<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\"",
        "<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"1\"",
    );
    assert!(
        normalized(&input).is_none(),
        "local and parent fonts differ without applyFont"
    );
}
#[test]
fn namespaced_and_empty_fonts_insert_in_the_correct_namespace() {
    let mut input = fixture();
    edit(
        &mut input,
        "xl/styles.xml",
        "<font><sz val=\"18\"/><name val=\"Liberation Sans\"/></font>",
        &format!("<q:font xmlns:q=\"{SHEET}\"/>"),
    );
    let changed = parts(&normalized(&input).unwrap());
    assert!(
        std::str::from_utf8(&changed["xl/styles.xml"])
            .unwrap()
            .contains(&format!(">{COLOR}</q:font>"))
    );
}
#[test]
fn malformed_xml_and_exhausted_budgets_stay_errors() {
    let mut input = fixture();
    input.insert("xl/styles.xml".into(), b"<!DOCTYPE a><a/>".to_vec());
    assert!(normalize(&archive(&input), deadline(), 100_000_000).is_err());
    let source = archive(&fixture());
    assert!(normalize(&source, deadline(), 1).is_err());
    assert!(
        normalize(
            &source,
            Instant::now() - Duration::from_secs(1),
            100_000_000
        )
        .is_err()
    );
}

#[test]
fn legal_utf16_custom_xml_declines_policy_before_utf8_indexing() {
    let mut input = fixture();
    let mut bytes = vec![0xff, 0xfe];
    for word in
        "<?xml version=\"1.0\" encoding=\"UTF-16\"?><custom xmlns=\"urn:author\">value</custom>"
            .encode_utf16()
    {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    input.insert("customXml/item1.xml".into(), bytes);
    assert!(normalized(&input).is_none());
}

#[test]
fn unknown_utf8_xml_role_declines_without_changing_the_package() {
    let mut input = fixture();
    input.insert(
        "customXml/item1.xml".into(),
        b"<custom xmlns=\"urn:author\">value</custom>".to_vec(),
    );
    let original = archive(&input);
    let saved = original.clone();
    assert!(
        normalize(&original, deadline(), 100_000_000)
            .unwrap()
            .is_none()
    );
    assert_eq!(original, saved);
}

fn utf16(text: &str, little_endian: bool, bom: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    if bom {
        bytes.extend_from_slice(if little_endian {
            &[0xff, 0xfe]
        } else {
            &[0xfe, 0xff]
        });
    }
    for word in text.encode_utf16() {
        bytes.extend_from_slice(&if little_endian {
            word.to_le_bytes()
        } else {
            word.to_be_bytes()
        });
    }
    bytes
}

#[test]
fn known_styles_with_unsupported_encoding_preserve_original_renderer_behavior() {
    for little_endian in [false, true] {
        for bom in [false, true] {
            let mut input = fixture();
            let styles = format!(
                "<?xml version=\"1.0\" encoding=\"UTF-16\"?>{}",
                std::str::from_utf8(&input["xl/styles.xml"]).unwrap()
            );
            input.insert("xl/styles.xml".into(), utf16(&styles, little_endian, bom));
            assert!(normalized(&input).is_none());
        }
    }
    let mut input = fixture();
    let styles = format!(
        "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>{}",
        std::str::from_utf8(&input["xl/styles.xml"]).unwrap()
    );
    input.insert("xl/styles.xml".into(), styles.into_bytes());
    assert!(normalized(&input).is_none());
}

#[test]
fn later_known_part_encoding_declines_but_supported_utf8_errors_remain_errors() {
    let mut input = fixture();
    let chart = "<?xml version=\"1.0\" encoding=\"UTF-16\"?><chartSpace xmlns=\"http://schemas.openxmlformats.org/drawingml/2006/chart\"/>";
    input.insert("xl/charts/chart1.xml".into(), utf16(chart, false, false));
    assert!(normalized(&input).is_none());
    for invalid in [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><!DOCTYPE a><a/>",
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><a>",
        "<?xml version=\"1.0\" encoding=\"ISO-8859-1\" encoding=\"UTF-8\"?><a/>",
        "<?xml version=\"1.0\" encoding=\"\"?><a/>",
    ] {
        input.insert("xl/charts/chart1.xml".into(), invalid.as_bytes().to_vec());
        assert!(
            normalize(&archive(&input), deadline(), 100_000_000).is_err(),
            "{invalid}"
        );
    }
}
