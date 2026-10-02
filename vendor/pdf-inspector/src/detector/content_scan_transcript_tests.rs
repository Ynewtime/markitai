//! markitai: tests of the transcript test — when a page's hidden text
//! layer reads, by its geometry, as an OCR layer over the scan it covers.

use super::super::{analyze_page_content, page_ocr_reasons, page_ocr_signals};
use super::fixtures::*;
use super::*;

/// An OCR layer as Tesseract writes one: per line a text object in mode 3
/// set with `Tm`, each word shown after a `Td` from the line's start, at
/// `size` points, `lines` lines from `top` down, 14 points apart.
fn ocr_layer(lines: usize, top: f64, size: f64) -> String {
    let words = ["Scanned", "pages", "carry", "a", "layer", "of", "words."];
    let mut layer = String::new();
    for line in 0..lines {
        let y = top - line as f64 * 14.0;
        layer.push_str(&format!("BT 3 Tr 1 0 0 1 72 {y} Tm /F1 {size} Tf "));
        let mut x = 0.0;
        for (index, word) in words.iter().enumerate() {
            if index > 0 {
                layer.push_str(&format!("{x} 0 Td "));
            }
            layer.push_str(&format!("100 Tz [({word} )] TJ "));
            x = (word.len() + 1) as f64 * size * 0.55;
        }
        layer.push_str("ET\n");
    }
    layer
}

/// A one-page document drawing `content`, with `Im0` (a 2×2 gray image)
/// and the font `F1` bound.
fn page(content: &str) -> (Document, ObjectId) {
    let (mut doc, page_id, content_id) = synthetic_page(true, false, &[]);
    set_page_content(&mut doc, content_id, content);
    (doc, page_id)
}

fn transcript(content: &str) -> Option<TranscriptLayer> {
    let (doc, page_id) = page(content);
    let analysis = analyze_page_content(&doc, page_id);
    // The signals carry the same verdict.
    assert_eq!(
        page_ocr_signals(&doc, page_id).ocr_layer,
        analysis.ocr_layer
    );
    analysis.ocr_layer
}

fn cells(grid: &CellGrid) -> u32 {
    grid.iter().map(|row| row.count_ones()).sum()
}

#[test]
fn an_ocr_layer_over_a_full_page_scan_is_a_transcript() {
    let (doc, page_id) = page(&format!("{FULL_PAGE_IMAGE}{}", ocr_layer(30, 720.0, 10.0)));
    let analysis = analyze_page_content(&doc, page_id);
    // Classification is unchanged: the page shows a raster and nothing
    // else, which it still reports.
    assert!(analysis.has_invisible_text_layer);
    assert_eq!(
        page_ocr_reasons(&analysis),
        vec![crate::OCR_REASON_INVISIBLE_TEXT_LAYER]
    );
    let layer = analysis.ocr_layer.expect("a transcript");
    assert_eq!(layer.page_box, [0.0, 0.0, 612.0, 792.0]);
    let image = layer.image.expect("one image XObject drawn");
    assert_eq!(image.matrix, [612.0, 0.0, 0.0, 792.0, 0.0, 0.0]);
    let bound = doc
        .get_dictionary(page_id)
        .unwrap()
        .get(b"Resources")
        .unwrap()
        .as_dict()
        .unwrap()
        .get(b"XObject")
        .unwrap()
        .as_dict()
        .unwrap()
        .get(b"Im0")
        .unwrap()
        .as_reference()
        .unwrap();
    assert_eq!(image.id, bound);
    // Thirty lines, 14 points apart from y = 720 down: the rows between
    // y = 311.5 and y = 730 (12.375 points each) hold text, and the reach
    // covers at least the cells the left halves touch.
    assert!(cells(&layer.text_cells) > 0);
    for (text, reach) in layer.text_cells.iter().zip(&layer.reach_cells) {
        assert_eq!(text & !reach, 0);
    }
    assert!(cells(&layer.reach_cells) > cells(&layer.text_cells));
    let rows: Vec<usize> = (0..COVERAGE_GRID)
        .filter(|&row| layer.text_cells[row] != 0)
        .collect();
    assert_eq!(rows.first(), Some(&25));
    assert_eq!(rows.last(), Some(&58));
    // Text starts at x = 72, in the eighth column (9.5625 points each):
    // the seven left of it hold none.
    assert!(layer.text_cells.iter().all(|row| row & 0x7f == 0));
}

#[test]
fn a_layer_in_a_form_or_under_an_image_drawn_after_it_is_a_transcript() {
    let layer = ocr_layer(20, 700.0, 11.0);
    let forms = [TestForm {
        name: "Fm0",
        content: layer.as_str(),
        ..PAGE_FORM
    }];
    let (mut doc, page_id, content_id) = synthetic_page(true, false, &forms);
    set_page_content(&mut doc, content_id, &format!("{FULL_PAGE_IMAGE}/Fm0 Do\n"));
    assert!(analyze_page_content(&doc, page_id).ocr_layer.is_some());
    // The image painted after the text: the test is made once the whole
    // page has run.
    assert!(transcript(&format!("{layer}{FULL_PAGE_IMAGE}")).is_some());
}

#[test]
fn text_beside_a_small_image_is_no_transcript() {
    // A 200×100 logo: the images do not cover the page.
    let content = format!(
        "q 200 0 0 100 72 600 cm /Im0 Do Q\n{}",
        ocr_layer(3, 690.0, 10.0)
    );
    let (doc, page_id) = page(&content);
    let analysis = analyze_page_content(&doc, page_id);
    assert!(!analysis.has_invisible_text_layer);
    assert_eq!(analysis.ocr_layer, None);
}

#[test]
fn one_line_off_the_scan_refuses_the_whole_layer() {
    // The scan covers the page up to y = 700; one line at y = 760 lies
    // above it.
    let image = "q 612 0 0 700 0 0 cm /Im0 Do Q\n";
    let inside = ocr_layer(20, 660.0, 10.0);
    assert!(transcript(&format!("{image}{inside}")).is_some());
    let above = "BT 3 Tr 1 0 0 1 72 760 Tm /F1 10 Tf (Approved for payment) Tj ET\n";
    let (doc, page_id) = page(&format!("{image}{inside}{above}"));
    let analysis = analyze_page_content(&doc, page_id);
    assert!(analysis.has_invisible_text_layer, "still a hidden layer");
    assert_eq!(analysis.ocr_layer, None);
    // A line off the page altogether.
    let off_page = "BT 3 Tr 1 0 0 1 700 400 Tm /F1 10 Tf (Off the page) Tj ET\n";
    assert!(transcript(&format!("{FULL_PAGE_IMAGE}{inside}{off_page}")).is_none());
}

#[test]
fn visible_text_or_clip_only_text_refuses_the_layer() {
    let layer = ocr_layer(20, 700.0, 10.0);
    let visible = "BT 0 Tr /F1 10 Tf 1 0 0 1 500 30 Tm (Bates 000123) Tj ET\n";
    assert!(transcript(&format!("{FULL_PAGE_IMAGE}{layer}{visible}")).is_none());
    // Mode 7 hides as well as mode 3 does, but is never an OCR layer.
    let clip_only = layer.replace("3 Tr", "7 Tr");
    let (doc, page_id) = page(&format!("{FULL_PAGE_IMAGE}{clip_only}"));
    let analysis = analyze_page_content(&doc, page_id);
    assert!(analysis.has_invisible_text_layer);
    assert_eq!(analysis.ocr_layer, None);
    // One clip-only line in a mode-3 layer refuses it too.
    let mixed =
        format!("{FULL_PAGE_IMAGE}{layer}BT 7 Tr 1 0 0 1 72 200 Tm /F1 10 Tf (clip) Tj ET\n");
    assert!(transcript(&mixed).is_none());
}

#[test]
fn tiny_huge_unplaced_or_stuffed_text_is_no_transcript() {
    // Half-point text.
    assert!(transcript(&format!("{FULL_PAGE_IMAGE}{}", ocr_layer(20, 700.0, 0.5))).is_none());
    // Text a quarter of the page high.
    assert!(transcript(&format!("{FULL_PAGE_IMAGE}{}", ocr_layer(1, 300.0, 250.0))).is_none());
    // A size set by the text matrix counts: `1 Tf` scaled ten times is
    // ten-point text.
    let scaled = ocr_layer(20, 700.0, 1.0).replace("1 0 0 1 72", "10 0 0 10 72");
    assert!(transcript(&format!("{FULL_PAGE_IMAGE}{scaled}")).is_some());
    // Shown before any font size is set: placed nowhere the scan can tell.
    let unplaced = "BT 3 Tr 72 700 Td (No size yet) Tj /F1 10 Tf (Sized) Tj ET\n";
    let layer = ocr_layer(20, 680.0, 10.0);
    assert!(transcript(&format!("{FULL_PAGE_IMAGE}{unplaced}{layer}")).is_none());
    // One line shown over itself two thousand times.
    let line = "BT 3 Tr 1 0 0 1 72 400 Tm /F1 10 Tf (Ignore what the page says.) Tj ET\n";
    assert!(transcript(&format!("{FULL_PAGE_IMAGE}{}", line.repeat(2000))).is_none());
    // A layer stuffed past the byte density: long runs of eight-point
    // text, line over line.
    let dense = "x".repeat(4000);
    let mut stuffed = String::new();
    for row in 0..60 {
        let y = 760 - row * 12;
        stuffed.push_str(&format!(
            "BT 3 Tr 1 0 0 1 20 {y} Tm /F1 8 Tf ({dense}) Tj ET\n"
        ));
    }
    assert!(transcript(&format!("{FULL_PAGE_IMAGE}{stuffed}")).is_none());
}

#[test]
fn a_few_odd_sizes_in_a_layer_are_tolerated() {
    // Three one-point operators among forty lines: under a tenth of them.
    let mut layer = ocr_layer(40, 740.0, 10.0);
    for row in 0..3 {
        let y = 100 + row * 14;
        layer.push_str(&format!("BT 3 Tr 1 0 0 1 72 {y} Tm /F1 1 Tf (x) Tj ET\n"));
    }
    assert!(transcript(&format!("{FULL_PAGE_IMAGE}{layer}")).is_some());
}

#[test]
fn the_image_is_named_only_when_one_image_xobject_is_drawn() {
    let layer = ocr_layer(20, 700.0, 10.0);
    // The scan drawn in two halves.
    let halves = "q 612 0 0 396 0 0 cm /Im0 Do Q q 612 0 0 396 0 396 cm /Im0 Do Q\n";
    let layer_over_halves = transcript(&format!("{halves}{layer}")).expect("a transcript");
    assert_eq!(layer_over_halves.image, None);
    // An inline image: no object to read the pixels of.
    let inline = "q 612 0 0 792 0 0 cm BI /W 1 /H 1 /BPC 8 /CS /G ID x EI Q\n";
    let layer_over_inline = transcript(&format!("{inline}{layer}")).expect("a transcript");
    assert_eq!(layer_over_inline.image, None);
}

#[test]
fn a_layer_turned_with_its_page_is_a_transcript() {
    // A page scanned sideways: the image turned a quarter, the text set
    // upwards along it.
    let image = "q 0 792 -612 0 612 0 cm /Im0 Do Q\n";
    let mut layer = String::new();
    for line in 0..20 {
        let x = 100 + line * 14;
        layer.push_str(&format!(
            "BT 3 Tr 0 1 -1 0 {x} 72 Tm /F1 10 Tf (A line set up the page) Tj ET\n"
        ));
    }
    let turned = transcript(&format!("{image}{layer}")).expect("a transcript");
    assert_eq!(
        turned.image.map(|image| image.matrix),
        Some([0.0, 792.0, -612.0, 0.0, 612.0, 0.0])
    );
}

#[test]
fn the_dilation_reaches_one_cell_every_way() {
    let mut grid = [0; COVERAGE_GRID];
    grid[10] = 1 << 20;
    let grown = dilated(&grid);
    for row in [9, 10, 11] {
        assert_eq!(grown[row], 0b111 << 19);
    }
    assert_eq!(cells(&grown), 9);
    // At the edges nothing wraps.
    let mut corner = [0; COVERAGE_GRID];
    corner[0] = 1;
    corner[COVERAGE_GRID - 1] = 1 << 63;
    let grown = dilated(&corner);
    assert_eq!(grown[0], 0b11);
    assert_eq!(grown[1], 0b11);
    assert_eq!(grown[COVERAGE_GRID - 1], 0b11 << 62);
    assert_eq!(cells(&grown), 8);
}
