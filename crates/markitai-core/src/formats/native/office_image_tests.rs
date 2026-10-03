//! Actual ordinary-user source relationships, including an unused PPT bank
//! slot. Raw extraction and Markdown references are both required.
use super::*;

const DIAGRAM: &[u8] = include_bytes!("fixtures/office-images/diagram.png");
fn assert_diagram(bytes: &[u8], extension: &str) -> Document {
    let actual = extract(bytes, extension).unwrap();
    assert_eq!(actual.assets.len(), 1, "{}", actual.markdown);
    assert_eq!(
        actual.assets[0].bytes, DIAGRAM,
        "the referenced diagram must keep its original bytes"
    );
    assert_eq!(
        actual
            .markdown
            .matches(&format!("(.markitai/assets/{})", actual.assets[0].name))
            .count(),
        1,
        "{}",
        actual.markdown
    );
    actual
}
fn budget(markdown: &str, rate: bool) {
    for (label, value) in [
        ("North", "1250.50"),
        ("South", "900.25"),
        ("Grand total", "2150.75"),
    ] {
        assert!(
            markdown
                .lines()
                .any(|line| line.contains(label) && line.contains(value)),
            "{label}/{value}\n{markdown}"
        );
    }
    assert!(markdown.contains("00127"));
    if rate {
        assert!(markdown.contains("12.5%"));
    }
    assert!(markdown.contains("2026-10-03"));
}
#[test]
fn ooxml_diagram_stays_in_budget_sheet_and_original_cells_survive() {
    let d = assert_diagram(
        include_bytes!("fixtures/office-images/xlsx-complex.xlsx"),
        "xlsx",
    );
    budget(&d.markdown, true);
    let image = d.markdown.find("(.markitai/assets/").unwrap();
    assert!(d.markdown.find("## Budget").unwrap() < image);
    assert!(d.markdown.find("## Facts").unwrap() < d.markdown.find("## Budget").unwrap());
    assert!(!d.markdown.contains("Hidden notes"));
}
#[test]
fn biff_template_diagram_resolves_its_embedded_slot_without_changing_cached_values() {
    let d = assert_diagram(
        include_bytes!("fixtures/office-images/xlt-complex.xlt"),
        "xlt",
    );
    budget(&d.markdown, true);
    assert!(d.markdown.find("## Budget").unwrap() < d.markdown.find("(.markitai/assets/").unwrap());
    assert!(!d.markdown.contains("Hidden notes"));
}
#[test]
fn legacy_presentation_references_only_the_second_slide_diagram_not_the_orphan_bank_slot() {
    let d = assert_diagram(
        include_bytes!("fixtures/office-images/pps-complex.pps"),
        "pps",
    );
    let slides: Vec<_> = d.markdown.split("<!-- Slide number: ").collect();
    assert_eq!(slides.len(), 3, "{}", d.markdown);
    assert!(slides[1].contains("Quarterly review"));
    assert!(!slides[1].contains("(.markitai/assets/"));
    assert!(slides[2].contains("(.markitai/assets/"));
    for (label, value) in [
        ("North", "1250.50"),
        ("South", "900.25"),
        ("Grand total", "2150.75"),
    ] {
        assert!(
            slides[2]
                .lines()
                .any(|line| line.contains(label) && line.contains(value)),
            "{}",
            d.markdown
        );
    }
}
#[test]
fn odf_compatibility_body_picture_keeps_the_original_workbook_content() {
    let d = assert_diagram(
        include_bytes!("fixtures/office-images/ods-complex.ods"),
        "ods",
    );
    budget(&d.markdown, false);
}
