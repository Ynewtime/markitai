use super::*;
use std::{
    collections::BTreeMap,
    io::{Cursor, Read, Write},
    time::Duration,
};
fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(10)
}
fn extension() -> Extension {
    Extension {
        page: 1,
        width: 1400.,
        height: 744.,
        right: 1454.844,
    }
}
fn fixture() -> &'static [u8] {
    include_bytes!("../fixtures/whole-workbook.xlsx")
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
    let mut z = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (n, b) in parts {
        z.start_file(n, zip::write::SimpleFileOptions::default())
            .unwrap();
        z.write_all(b).unwrap();
    }
    z.finish().unwrap().into_inner()
}
fn text_pdf(x: f32, width: f32, text: &str) -> Vec<u8> {
    use lopdf::{Document, Object, Stream, dictionary};
    let mut pdf = Document::with_version("1.5");
    let pages = pdf.new_object_id();
    let font =
        pdf.add_object(dictionary! {"Type"=>"Font","Subtype"=>"Type1","BaseFont"=>"Helvetica"});
    let contents = pdf.add_object(Stream::new(
        dictionary! {},
        format!("BT /F1 12 Tf {x} 20 Td ({text}) Tj ET").into_bytes(),
    ));
    let page=pdf.add_object(dictionary!{"Type"=>"Page","Parent"=>pages,"MediaBox"=>vec![0.into(),0.into(),Object::Real(width),Object::Real(100.)],"Resources"=>dictionary!{"Font"=>dictionary!{"F1"=>font}},"Contents"=>contents});
    pdf.objects.insert(
        pages,
        dictionary! {"Type"=>"Pages","Kids"=>vec![page.into()],"Count"=>1}.into(),
    );
    let catalog = pdf.add_object(dictionary! {"Type"=>"Catalog","Pages"=>pages});
    pdf.trailer.set("Root", catalog);
    let mut b = Vec::new();
    pdf.save_to(&mut b).unwrap();
    b
}
#[test]
fn a_workbook_the_font_policy_cannot_read_keeps_its_fonts_until_the_deadline() {
    let mut malformed = parts(fixture());
    malformed.insert("xl/styles.xml".into(), b"<!DOCTYPE a><a/>".to_vec());
    for bytes in [archive(&malformed), b"not a zip".to_vec()] {
        assert!(
            normalize_default_font(&bytes, deadline(), 100_000_000)
                .unwrap()
                .is_none()
        );
    }
    assert!(normalize_default_font(fixture(), Instant::now(), 100_000_000).is_err());
}
#[test]
fn no_overflow_is_exact_no_op() {
    let pdf = text_pdf(10., 100., "HELLO");
    assert!(inspect(&pdf, 1, deadline()).unwrap().extensions.is_empty());
    assert_eq!(rewrite(fixture(), &[], deadline(), 1).unwrap(), fixture());
}
#[test]
fn real_positioned_text_supplies_dynamic_right_edge() {
    let pdf = text_pdf(90., 100., "HELLO");
    let p = inspect(&pdf, 1, deadline()).unwrap();
    assert_eq!(p.extensions.len(), 1);
    let e = &p.extensions[0];
    assert_eq!(e.page, 1);
    assert!(e.right > 125. && e.right < 140.);
    assert_eq!(e.width, 100.);
}
#[test]
fn occupied_neighbor_like_visible_runs_within_page_do_not_expand() {
    let pdf = text_pdf(1260., 1540., "OUTSIDE PRINTBLOCK");
    assert!(inspect(&pdf, 1, deadline()).unwrap().extensions.is_empty());
}
#[test]
fn geometry_deadline_and_page_count_are_bounded() {
    assert!(dimensions(1e9, 100.).is_err());
    assert!(dimensions(f64::NAN, 1.).is_err());
    assert!(inspect(&text_pdf(10., 100., "hi"), 2, deadline()).is_err());
    assert!(inspect(&text_pdf(10., 100., "hi"), 1, Instant::now()).is_err());
    assert!(rewrite(fixture(), &[extension()], deadline(), 1).is_err());
    assert!(rewrite(fixture(), &[extension()], Instant::now(), 10_000_000).is_err());
}
#[test]
fn transparent_carrier_preserves_source_cells_styles_and_sheets() {
    let original = parts(fixture());
    let changed = parts(&rewrite(fixture(), &[extension()], deadline(), 10_000_000).unwrap());
    for (name, bytes) in &original {
        if !matches!(
            name.as_str(),
            "xl/worksheets/sheet1.xml" | "[Content_Types].xml"
        ) {
            assert_eq!(&changed[name], bytes, "{name}");
        }
    }
    let sheet = std::str::from_utf8(&changed["xl/worksheets/sheet1.xml"]).unwrap();
    assert!(sheet.contains("OUTSIDE PRINT AREA"));
    assert!(sheet.contains("width=\"14\""));
    let drawing = changed
        .iter()
        .find(|(n, _)| n.starts_with("xl/drawings/markitai-bounds-"))
        .unwrap()
        .1;
    let text = std::str::from_utf8(drawing).unwrap();
    assert!(
        text.contains("<a:noFill/>") && text.contains("hidden=\"0\"") && !text.contains("<a:t>")
    );
    assert_eq!(fixture(), include_bytes!("../fixtures/whole-workbook.xlsx"));
}
#[test]
fn existing_drawing_and_relationships_are_preserved_without_id_collision() {
    let mut b = parts(fixture());
    let s=String::from_utf8(b["xl/worksheets/sheet1.xml"].clone()).unwrap().replace("</worksheet>","<drawing xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" r:id=\"chart\"/></worksheet>");
    b.insert("xl/worksheets/sheet1.xml".into(), s.into_bytes());
    b.insert("xl/worksheets/_rels/sheet1.xml.rels".into(),br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="chart" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" Target="../drawings/chart.xml"/></Relationships>"#.to_vec());
    let existing=br#"<q:wsDr xmlns:q="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"><q:twoCellAnchor><q:cNvPr id="7" name="Markitai layout bounds 1"/></q:twoCellAnchor></q:wsDr>"#;
    b.insert("xl/drawings/chart.xml".into(), existing.to_vec());
    b.insert(
        "xl/drawings/_rels/chart.xml.rels".into(),
        b"unchanged chart relationship bytes".to_vec(),
    );
    let changed = parts(&rewrite(&archive(&b), &[extension()], deadline(), 10_000_000).unwrap());
    for (n, v) in &b {
        if n != "xl/drawings/chart.xml" {
            assert_eq!(&changed[n], v, "{n}");
        }
    }
    let d = std::str::from_utf8(&changed["xl/drawings/chart.xml"]).unwrap();
    assert!(d.contains("id=\"8\""));
    assert!(d.contains("name=\"Markitai layout bounds 2\""));
    assert!(d.contains(
        "<q:twoCellAnchor><q:cNvPr id=\"7\" name=\"Markitai layout bounds 1\"/></q:twoCellAnchor>"
    ));
}
#[test]
fn names_and_relationship_ids_are_allocated_without_overwriting() {
    let mut b = parts(fixture());
    b.insert(
        "xl/drawings/markitai-bounds-1.xml".into(),
        b"preserve me".to_vec(),
    );
    b.insert("xl/worksheets/_rels/sheet1.xml.rels".into(),br#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdMarkitaiBounds1" Type="other" Target="image.png"/></Relationships>"#.to_vec());
    let changed = parts(&rewrite(&archive(&b), &[extension()], deadline(), 10_000_000).unwrap());
    assert_eq!(changed["xl/drawings/markitai-bounds-1.xml"], b"preserve me");
    assert!(changed.contains_key("xl/drawings/markitai-bounds-2.xml"));
    assert!(
        std::str::from_utf8(&changed["xl/worksheets/_rels/sheet1.xml.rels"])
            .unwrap()
            .contains("Id=\"rIdMarkitaiBounds2\"")
    );
}
#[test]
fn external_duplicate_and_escaping_relationships_are_rejected() {
    for mutation in [
        "TargetMode=\"External\" Target=\"https://invalid.example/a\"",
        "Target=\"../../../escape.xml\"",
    ] {
        let mut b = parts(fixture());
        let rel = String::from_utf8(b["xl/_rels/workbook.xml.rels"].clone())
            .unwrap()
            .replace("Target=\"worksheets/sheet1.xml\"", mutation);
        b.insert("xl/_rels/workbook.xml.rels".into(), rel.into_bytes());
        assert!(rewrite(&archive(&b), &[extension()], deadline(), 10_000_000).is_err());
    }
    let mut b = parts(fixture());
    let rel = String::from_utf8(b["xl/_rels/workbook.xml.rels"].clone())
        .unwrap()
        .replace("Id=\"rId2\"", "Id=\"rId1\"");
    b.insert("xl/_rels/workbook.xml.rels".into(), rel.into_bytes());
    assert!(rewrite(&archive(&b), &[extension()], deadline(), 10_000_000).is_err());
}
#[test]
fn invalid_zip_xml_and_ambiguous_pages_are_rejected() {
    assert!(rewrite(b"not zip", &[extension()], deadline(), 10_000_000).is_err());
    let mut b = parts(fixture());
    b.insert("../escape".into(), vec![]);
    assert!(rewrite(&archive(&b), &[extension()], deadline(), 10_000_000).is_err());
    assert!(xml::index(b"<!DOCTYPE x><x/>", deadline()).is_err());
    assert!(xml::index(b"<x/><x/>", deadline()).is_err());
    assert!(
        rewrite(
            fixture(),
            &[extension(), extension()],
            deadline(),
            10_000_000
        )
        .is_err()
    );
}
#[test]
fn workbook_relationship_order_selects_sheet_not_filename() {
    let mut b = parts(fixture());
    let book = String::from_utf8(b["xl/workbook.xml"].clone())
        .unwrap()
        .replace("r:id=\"rId1\"", "r:id=\"swap\"")
        .replace("r:id=\"rId4\"", "r:id=\"rId1\"")
        .replace("r:id=\"swap\"", "r:id=\"rId4\"");
    b.insert("xl/workbook.xml".into(), book.into_bytes());
    let changed = parts(&rewrite(&archive(&b), &[extension()], deadline(), 10_000_000).unwrap());
    assert_eq!(
        b["xl/worksheets/sheet1.xml"],
        changed["xl/worksheets/sheet1.xml"]
    );
    assert_ne!(
        b["xl/worksheets/sheet4.xml"],
        changed["xl/worksheets/sheet4.xml"]
    );
}

#[test]
fn shared_sheet_alias_is_rejected_but_missing_a1_selects_real_data() {
    let mut b = parts(fixture());
    let book = String::from_utf8(b["xl/workbook.xml"].clone())
        .unwrap()
        .replace("r:id=\"rId2\"", "r:id=\"rId1\"");
    b.insert("xl/workbook.xml".into(), book.into_bytes());
    assert!(rewrite(&archive(&b), &[extension()], deadline(), 10_000_000).is_err());
    let mut b = parts(fixture());
    let sheet = String::from_utf8(b["xl/worksheets/sheet1.xml"].clone())
        .unwrap()
        .replace("VISIBLE FIRST", "");
    b.insert("xl/worksheets/sheet1.xml".into(), sheet.into_bytes());
    let changed = parts(&rewrite(&archive(&b), &[extension()], deadline(), 10_000_000).unwrap());
    let drawing = changed
        .iter()
        .find(|(n, _)| n.starts_with("xl/drawings/markitai-bounds-"))
        .unwrap()
        .1;
    let text = std::str::from_utf8(drawing).unwrap();
    assert!(text.contains("<mt:col>9</mt:col>") && text.contains("<mt:row>30</mt:row>"));
}
#[test]
fn xml_insertion_preserves_bom_and_rejects_text_outside_root() {
    let xml = b"\xef\xbb\xbf<r xmlns='urn:test'/>";
    let nodes = super::xml::index(xml, deadline()).unwrap();
    let changed = super::xml::append(xml, &nodes[0], "<child/>").unwrap();
    assert_eq!(changed, b"\xef\xbb\xbf<r xmlns='urn:test'><child/></r>");
    assert!(super::xml::index(b"junk<r/>", deadline()).is_err());
}

#[test]
fn repair_verification_preserves_existing_other_sheet_uncertainty() {
    let before = Plan {
        extensions: vec![extension()],
        uncertain_pages: vec![2],
        warnings: vec!["sheet 2 unchanged".into()],
    };
    let after = Plan {
        uncertain_pages: vec![2],
        warnings: vec!["sheet 2 unchanged".into()],
        ..Plan::default()
    };
    assert!(verify_repair(&before, &after).is_ok());
    let changed = Plan {
        uncertain_pages: vec![1, 2],
        ..Plan::default()
    };
    assert!(verify_repair(&before, &changed).is_err());
    let remaining = Plan {
        extensions: vec![extension()],
        ..Plan::default()
    };
    assert!(verify_repair(&before, &remaining).is_err());
}

#[test]
fn actual_two_page_measurement_keeps_other_rotated_page_warning() {
    use lopdf::{Object, dictionary};
    let mut pdf = lopdf::Document::load_mem(&text_pdf(90., 100., "HELLO")).unwrap();
    let page_id = *pdf.get_pages().values().next().unwrap();
    let parent = pdf
        .get_dictionary(page_id)
        .unwrap()
        .get(b"Parent")
        .unwrap()
        .as_reference()
        .unwrap();
    let mut other = pdf.get_dictionary(page_id).unwrap().clone();
    other.set("Rotate", 90);
    let other_id = pdf.add_object(other);
    pdf.objects.insert(
        parent,
        dictionary! {"Type"=>"Pages", "Kids"=>vec![page_id.into(),other_id.into()], "Count"=>2}
            .into(),
    );
    let mut original = Vec::new();
    pdf.save_to(&mut original).unwrap();
    let before = inspect(&original, 2, deadline()).unwrap();
    assert_eq!(before.extensions.len(), 1);
    assert_eq!(before.uncertain_pages, [2]);
    pdf.get_dictionary_mut(page_id).unwrap().set(
        "MediaBox",
        vec![0.into(), 0.into(), Object::Real(150.), Object::Real(100.)],
    );
    let mut repaired = Vec::new();
    pdf.save_to(&mut repaired).unwrap();
    let after = inspect(&repaired, 2, deadline()).unwrap();
    assert_eq!(after.uncertain_pages, [2]);
    assert_eq!(before.warnings, after.warnings);
    assert!(after.extensions.is_empty());
    assert!(verify_repair(&before, &after).is_ok());
}

#[test]
fn macro_parts_are_not_modified_by_ooxml_repair() {
    let mut source = parts(fixture());
    let content = String::from_utf8(source["[Content_Types].xml"].clone())
        .unwrap()
        .replace(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
            "application/vnd.ms-excel.sheet.macroEnabled.main+xml",
        );
    source.insert("[Content_Types].xml".into(), content.into_bytes());
    source.insert(
        "xl/vbaProject.bin".into(),
        b"authored inert macro-part bytes, never executed".to_vec(),
    );
    let rel = String::from_utf8(source["xl/_rels/workbook.xml.rels"].clone()).unwrap().replace("</Relationships>", "<Relationship Id=\"macro\" Type=\"http://schemas.microsoft.com/office/2006/relationships/vbaProject\" Target=\"vbaProject.bin\"/></Relationships>");
    source.insert("xl/_rels/workbook.xml.rels".into(), rel.into_bytes());
    let changed =
        parts(&rewrite(&archive(&source), &[extension()], deadline(), 10_000_000).unwrap());
    assert_eq!(changed["xl/vbaProject.bin"], source["xl/vbaProject.bin"]);
    assert_eq!(
        changed["xl/_rels/workbook.xml.rels"],
        source["xl/_rels/workbook.xml.rels"]
    );
    assert!(
        std::str::from_utf8(&changed["[Content_Types].xml"])
            .unwrap()
            .contains("macroEnabled.main+xml")
    );
}

fn anchor_xml(body: &str, shared: &[bool]) -> Result<anchor::Anchor> {
    let bytes = format!(r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">{body}</worksheet>"#).into_bytes();
    anchor::select(&xml::index(&bytes, deadline())?, &bytes, shared, deadline())
}
#[test]
fn data_anchor_handles_non_a1_and_split_minimum_without_dimension_guess() {
    let a=anchor_xml(r#"<dimension ref="A1:XFD1048576"/><sheetData><row r="5"><c r="F5" t="inlineStr"><is><t>FIRST ROW</t></is></c></row><row r="10"><c r="C10"><v>0</v></c></row><row r="31"><c r="J31" t="inlineStr"><is><t>OUTSIDE</t></is></c></row></sheetData>"#,&[]).unwrap();
    assert_eq!(a, anchor::Anchor { col: 2, row: 4 });
}
#[test]
fn hidden_prefix_and_hidden_values_do_not_define_anchor() {
    let a=anchor_xml(r#"<cols><col min="1" max="2" hidden="1"/></cols><sheetData><row r="1" hidden="1"><c r="A1"><v>7</v></c></row><row r="5"><c r="A5"><v>8</v></c><c r="C5"><v>9</v></c></row></sheetData>"#,&[]).unwrap();
    assert_eq!(a, anchor::Anchor { col: 2, row: 4 });
}
#[test]
fn shared_string_zero_is_not_assumed_nonempty_and_phonetics_are_not_data() {
    let bytes=br#"<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t> </t><rPh sb="0" eb="1"><t>phonetic only</t></rPh></si><si><r><t>&amp; text</t></r></si><si><t><![CDATA[ real ]]></t></si><si><t>&#32;</t></si></sst>"#;
    let shared = anchor::shared_nonempty(bytes, deadline()).unwrap();
    assert_eq!(shared, vec![false, true, true, false]);
    let a=anchor_xml(r#"<sheetData><row r="1"><c r="A1" t="s"><v>0</v></c></row><row r="5"><c r="C5" t="s"><v>1</v></c></row></sheetData>"#,&shared).unwrap();
    assert_eq!(a, anchor::Anchor { col: 2, row: 4 });
}
#[test]
fn formula_only_is_explicit_limit_but_formula_bytes_with_other_data_remain_eligible() {
    let only = r#"<sheetData><row r="5"><c r="C5"><f>1+1</f><v>2</v></c></row></sheetData>"#;
    assert!(anchor_xml(only, &[]).is_err());
    let mixed = only.replace("</row>", "<c r=\"D5\"><v>0</v></c></row>");
    assert_eq!(
        anchor_xml(&mixed, &[]).unwrap(),
        anchor::Anchor { col: 3, row: 4 }
    );
}
#[test]
fn anchor_rejects_ambiguous_and_unmappable_layout() {
    for body in [
        r#"<sheetViews><sheetView rightToLeft="1"/></sheetViews><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData>"#,
        r#"<sheetData><row r="1"><c r="A2"><v>1</v></c></row></sheetData>"#,
        r#"<sheetData><row r="1"><c r="A1"><v>1</v></c><c r="A1"><v>2</v></c></row></sheetData>"#,
        r#"<sheetData><row r="1"><c r="XFE1"><v>1</v></c></row></sheetData>"#,
        r#"<sheetData><row r="1" ht="1"><c r="A1"><v>1</v></c></row></sheetData>"#,
        r#"<cols><col min="1" max="1" width="0.2"/></cols><sheetData><row r="1"><c r="A1"><v>1</v></c></row></sheetData>"#,
        r#"<sheetData><row r="1"><c r="B1"><v>1</v></c></row></sheetData><mergeCells><mergeCell ref="A1:B1"/></mergeCells>"#,
    ] {
        assert!(anchor_xml(body, &[]).is_err(), "{body}");
    }
}
#[test]
fn shared_string_package_relationship_is_resolved_and_preserved() {
    let mut original = parts(fixture());
    let sheet = String::from_utf8(original["xl/worksheets/sheet1.xml"].clone())
        .unwrap()
        .replace(
            "t=\"inlineStr\"><is><t>VISIBLE FIRST</t></is>",
            "t=\"s\"><v>0</v>",
        );
    assert!(sheet.contains("t=\"s\"><v>0</v>"));
    original.insert("xl/worksheets/sheet1.xml".into(), sheet.into_bytes());
    let relations=String::from_utf8(original["xl/_rels/workbook.xml.rels"].clone()).unwrap().replace("</Relationships>",r#"<Relationship Id="strings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="strings/custom.xml"/></Relationships>"#);
    original.insert("xl/_rels/workbook.xml.rels".into(), relations.into_bytes());
    original.insert("xl/strings/custom.xml".into(),br#"<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>VISIBLE FIRST</t></si></sst>"#.to_vec());
    let changed =
        parts(&rewrite(&archive(&original), &[extension()], deadline(), 10_000_000).unwrap());
    for name in ["xl/_rels/workbook.xml.rels", "xl/strings/custom.xml"] {
        assert_eq!(original[name], changed[name]);
    }
    let drawing = changed
        .iter()
        .find(|(n, _)| n.starts_with("xl/drawings/markitai-bounds-"))
        .unwrap()
        .1;
    let drawing = std::str::from_utf8(drawing).unwrap();
    assert!(drawing.contains("<mt:col>0</mt:col>") && drawing.contains("<mt:row>0</mt:row>"));
    assert!(drawing.contains("<mt:oneCellAnchor") && !drawing.contains("absoluteAnchor"));
    assert!(drawing.contains("<mt:colOff>12700</mt:colOff>"));
}

#[test]
fn cached_formula_is_preserved_but_does_not_supply_anchor_proof() {
    let mut original = parts(fixture());
    let sheet = String::from_utf8(original["xl/worksheets/sheet1.xml"].clone())
        .unwrap()
        .replace(
            r#"t="inlineStr"><is><t>VISIBLE FIRST</t></is>"#,
            r#"><f>SUM(B1:C1)</f><v>99</v>"#,
        );
    assert!(sheet.contains("<f>SUM(B1:C1)</f>"));
    original.insert("xl/worksheets/sheet1.xml".into(), sheet.as_bytes().to_vec());
    let changed =
        parts(&rewrite(&archive(&original), &[extension()], deadline(), 10_000_000).unwrap());
    let index = xml::index(sheet.as_bytes(), deadline()).unwrap();
    let old = index.iter().find(|n| n.local == "sheetData").unwrap();
    let data = &sheet.as_bytes()[old.start..old.end];
    assert!(
        changed["xl/worksheets/sheet1.xml"]
            .windows(data.len())
            .any(|window| window == data)
    );
    let drawing = changed
        .iter()
        .find(|(n, _)| n.starts_with("xl/drawings/markitai-bounds-"))
        .unwrap()
        .1;
    let text = std::str::from_utf8(drawing).unwrap();
    assert!(text.contains("<mt:col>9</mt:col>") && text.contains("<mt:row>30</mt:row>"));
}

#[test]
fn merge_covered_source_cell_cannot_prove_a_synthetic_anchor_outside_merge() {
    let body = r#"<sheetData><row r="1"><c r="F1"><v>1</v></c></row><row r="5"><c r="C5"><v>999</v></c></row></sheetData><mergeCells><mergeCell ref="B5:C6"/></mergeCells>"#;
    // C5 is ignored by Calc. Combining it with F1 used to invent C1 outside
    // the merge, so merely checking the final synthesized point was insufficient.
    assert_eq!(
        anchor_xml(body, &[]).unwrap(),
        anchor::Anchor { col: 5, row: 0 }
    );
    let top_left = body.replace("r=\"C5\"", "r=\"B5\"");
    assert_eq!(
        anchor_xml(&top_left, &[]).unwrap(),
        anchor::Anchor { col: 1, row: 0 }
    );
}
#[test]
fn overlapping_merges_do_not_exempt_another_merges_covered_top_left() {
    let body = r#"<sheetData><row r="1"><c r="F1"><v>1</v></c></row><row r="5"><c r="B5"><v>999</v></c></row></sheetData><mergeCells><mergeCell ref="A4:C6"/><mergeCell ref="B5:D7"/></mergeCells>"#;
    assert_eq!(
        anchor_xml(body, &[]).unwrap(),
        anchor::Anchor { col: 5, row: 0 }
    );
}
#[test]
fn only_canonical_unique_containers_supply_cells_columns_and_merges() {
    let body = r#"<other><row r="1"><c r="A1"><v>999</v></c></row><col min="6" max="6" hidden="1"/><mergeCell ref="A1:G8"/></other><sheetData><row r="5"><c r="F5"><v>1</v></c></row></sheetData>"#;
    assert_eq!(
        anchor_xml(body, &[]).unwrap(),
        anchor::Anchor { col: 5, row: 4 }
    );
    assert!(
        anchor_xml(
            r#"<other><row r="1"><c r="A1"><v>999</v></c></row></other>"#,
            &[]
        )
        .is_err()
    );
    assert!(anchor_xml(&format!("{body}<sheetData/>"), &[]).is_err());
    assert!(anchor_xml(&format!("{body}<cols/><cols/>"), &[]).is_err());
    assert!(anchor_xml(&format!("{body}<mergeCells/><mergeCells/>"), &[]).is_err());
    let fake_child = r#"<sheetData><unknown r="1"><c r="A1"><v>999</v></c></unknown></sheetData>"#;
    assert!(anchor_xml(fake_child, &[]).is_err());
}
#[test]
fn merge_qualification_is_bounded_and_keeps_top_left_literals() {
    let mut body = String::from("<sheetData>");
    for row in 1..=2000 {
        body.push_str(&format!(
            r#"<row r="{row}"><c r="B{row}"><v>7</v></c></row>"#
        ));
    }
    body.push_str("</sheetData><mergeCells>");
    for row in 1..=2000 {
        body.push_str(&format!(r#"<mergeCell ref="A{row}:B{row}"/>"#));
    }
    body.push_str("</mergeCells>");
    assert!(anchor_xml(&body, &[]).is_err());
    let top_left = body.replace("<c r=\"B1\">", "<c r=\"A1\">");
    assert_eq!(
        anchor_xml(&top_left, &[]).unwrap(),
        anchor::Anchor { col: 0, row: 0 }
    );
}
