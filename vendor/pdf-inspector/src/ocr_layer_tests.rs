//! markitai: the per-page Markdown of pages whose invisible layer
//! transcribes their scan (see `PageOcrLayer`).

use super::*;
use lopdf::{dictionary, Object, Stream};

/// Pages of 612×792 points, each drawing its content with the font `F1`
/// (Helvetica) and the image `Im0` (2×2 gray) bound.
fn pdf(pages: &[String]) -> Vec<u8> {
    let mut doc = Document::with_version("1.5");
    let tree = doc.new_object_id();
    let font = doc.add_object(
        dictionary! { "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica" },
    );
    let image = doc.add_object(Stream::new(
        dictionary! {
            "Type" => "XObject", "Subtype" => "Image", "Width" => 2, "Height" => 2,
            "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8
        },
        vec![250, 40, 40, 250],
    ));
    let mut kids = Vec::new();
    for content in pages {
        let contents = doc.add_object(Stream::new(dictionary! {}, content.as_bytes().to_vec()));
        kids.push(Object::Reference(doc.add_object(dictionary! {
            "Type" => "Page", "Parent" => tree, "Contents" => contents,
            "Resources" => dictionary! {
                "Font" => dictionary! { "F1" => font },
                "XObject" => dictionary! { "Im0" => image }
            }
        })));
    }
    doc.objects.insert(
        tree,
        dictionary! {
            "Type" => "Pages", "Count" => kids.len() as i64, "Kids" => kids,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()]
        }
        .into(),
    );
    let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => tree });
    doc.trailer.set("Root", catalog);
    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).unwrap();
    bytes
}

const SCAN: &str = "q 612 0 0 792 0 0 cm /Im0 Do Q\n";

/// `lines` invisible lines of words from y = 700 down, as an OCR layer
/// sets them: `size` points, one text object per line.
fn layer(lines: &[&str], size: f64) -> String {
    layer_from(700.0, lines, size)
}

/// [`layer`] from y = `top` down.
fn layer_from(top: f64, lines: &[&str], size: f64) -> String {
    let mut layer = String::new();
    for (index, line) in lines.iter().enumerate() {
        let y = top - index as f64 * size * 1.4;
        layer.push_str(&format!(
            "BT 3 Tr 1 0 0 1 72 {y} Tm /F1 {size} Tf ({line}) Tj ET\n"
        ));
    }
    layer
}

const LINES: &[&str] = &[
    "Quarterly report of the harbour authority",
    "The harbour handled more ships this quarter than in any",
    "quarter before it, and the new berth opened in March.",
    "Repairs to the north pier are planned for the autumn.",
];

#[test]
fn a_scan_with_an_ocr_layer_reads_as_its_layer() {
    // The first line set larger, as an OCR layer sets a title's line.
    let mut content = SCAN.to_string();
    content.push_str(&layer(&LINES[..1], 24.0));
    content.push_str(&layer_from(650.0, &LINES[1..], 11.0));
    let bytes = pdf(&[content]);
    let result = extract_pages_markdown_mem(&bytes, None).unwrap();
    let page = &result.pages[0];
    assert!(!page.needs_ocr, "{page:?}");
    assert_eq!(page.ocr_reason, None);
    assert!(
        page.markdown
            .starts_with("Quarterly report of the harbour authority\n\nThe harbour handled"),
        "{}",
        page.markdown
    );
    assert!(page.markdown.contains("the new berth opened in March."));
    // No heading is read from the layer's sizes.
    assert!(!page.markdown.contains('#'), "{}", page.markdown);
    assert!(result.pages_needing_ocr.is_empty());
    let [layer] = result.ocr_layer_by_page.as_slice() else {
        panic!("one layer page: {:?}", result.ocr_layer_by_page);
    };
    assert_eq!(layer.page, 1);
    assert_eq!(layer.page_box, [0.0, 0.0, 612.0, 792.0]);
    assert_eq!(
        layer.image.map(|(_, matrix)| matrix),
        Some([612.0, 0.0, 0.0, 792.0, 0.0, 0.0])
    );
    // A loaded document reads the same, through its page-run cache.
    let loaded = LoadedPdf::load_mem(&bytes).unwrap();
    let again = loaded.pages_markdown(None).unwrap();
    assert_eq!(again.pages[0].markdown, page.markdown);
    assert_eq!(again.ocr_layer_by_page, result.ocr_layer_by_page);
    // The positioned text of the other readings leaves the layer out, as
    // it always has.
    let (items, _) = loaded
        .text_with_positions_and_rotations(None, PositionOptions::new())
        .unwrap();
    assert!(!items.iter().any(|item| item.text.contains("harbour")));
}

#[test]
fn a_layer_that_does_not_read_as_text_leaves_the_page_a_scan() {
    for lines in [
        // Too few letters and digits for an OCR layer: a stamp's words.
        vec!["Received by the clerk"],
        // Punctuation only.
        vec!["~~~ ### ;;; ||| ~~~ ### ;;; ||| ~~~ ### ;;; |||"; 12],
    ] {
        let bytes = pdf(&[format!("{SCAN}{}", layer(&lines, 11.0))]);
        let result = extract_pages_markdown_mem(&bytes, None).unwrap();
        let page = &result.pages[0];
        assert!(page.needs_ocr, "{lines:?}: {page:?}");
        assert_eq!(page.markdown, "");
        assert_eq!(
            page.ocr_reason.as_deref(),
            Some(OCR_REASON_INVISIBLE_TEXT_LAYER)
        );
        assert!(result.ocr_layer_by_page.is_empty());
        assert_eq!(result.pages_needing_ocr, vec![1]);
    }
}

#[test]
fn a_layer_page_does_not_set_the_size_other_pages_are_read_against() {
    // A native page: a 16-point heading over 12-point body text.
    let native = "BT /F1 16 Tf 72 720 Td (Harbour notes) Tj ET\n\
        BT /F1 12 Tf 72 690 Td (The body of the notes is set in twelve points, line after line.) Tj ET\n\
        BT /F1 12 Tf 72 675 Td (A second line of the body keeps the size the reader expects.) Tj ET\n"
        .to_string();
    // A scan whose layer sets twenty lines in 18 points: were its sizes
    // counted, they would be the document's body size.
    let many: Vec<&str> = LINES.iter().copied().cycle().take(20).collect();
    let scan = format!("{SCAN}{}", layer_from(760.0, &many, 18.0));
    let alone = extract_pages_markdown_mem(&pdf(std::slice::from_ref(&native)), None).unwrap();
    let mixed = extract_pages_markdown_mem(&pdf(&[native.clone(), scan]), None).unwrap();
    assert!(
        alone.pages[0].markdown.starts_with('#'),
        "{}",
        alone.pages[0].markdown
    );
    assert_eq!(mixed.pages[0].markdown, alone.pages[0].markdown);
    assert!(!mixed.pages[1].needs_ocr);
    assert_eq!(
        mixed
            .ocr_layer_by_page
            .iter()
            .map(|layer| layer.page)
            .collect::<Vec<_>>(),
        vec![2]
    );
    // Only the pages asked for are read, and their layers reported.
    let first = extract_pages_markdown_mem(
        &pdf(&[native, format!("{SCAN}{}", layer(LINES, 11.0))]),
        Some(&[1]),
    )
    .unwrap();
    assert_eq!(first.pages.len(), 1);
    assert_eq!(first.ocr_layer_by_page[0].page, 2);
    assert!(first.pages[0].markdown.contains("harbour"));
}

#[test]
fn a_hidden_layer_that_is_no_transcript_keeps_the_page_a_scan() {
    // A layer half off the scan: the image covers the lower half only.
    let half = "q 612 0 0 396 0 0 cm /Im0 Do Q\n";
    let bytes = pdf(&[format!("{half}{}", layer(LINES, 11.0))]);
    let result = extract_pages_markdown_mem(&bytes, None).unwrap();
    assert!(result.pages[0].needs_ocr);
    assert_eq!(result.pages[0].markdown, "");
    assert!(result.ocr_layer_by_page.is_empty());
    // Visible text beside the layer: the visible text is read, the layer
    // is not.
    let visible =
        "BT 0 Tr /F1 12 Tf 72 100 Td (A visible caption stays the page's own text.) Tj ET\n";
    let bytes = pdf(&[format!("{SCAN}{}{visible}", layer(LINES, 11.0))]);
    let result = extract_pages_markdown_mem(&bytes, None).unwrap();
    assert!(result.ocr_layer_by_page.is_empty());
    assert!(!result.pages[0].markdown.contains("harbour"));
}
