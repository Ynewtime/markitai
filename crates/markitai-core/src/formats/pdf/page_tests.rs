//! Authored page streams exercise the extraction/assembly boundary without OCR.
use super::*;
use lopdf::dictionary;

const NATIVE: &str = "Visible native paragraph retained exactly while other pages require OCR.";
const LIMITATION: &str = "PDF images are appended to their source page; exact placement, page screenshots, vector graphics and local OCR are not implemented.";

fn fixture(streams: &[String], masked_image: bool) -> (Vec<u8>, String) {
    let mut pdf = lopdf::Document::with_version("1.7");
    let tree = pdf.new_object_id();
    let font = pdf.add_object(dictionary! {
        "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica"
    });
    let mut image = Stream::new(
        dictionary! {
            "Type" => "XObject", "Subtype" => "Image", "Width" => 1,
            "Height" => 1, "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8
        },
        vec![255, 0, 0],
    );
    if masked_image {
        image
            .dict
            .set("Mask", vec![Object::Integer(0), Object::Integer(0)]);
    }
    let image_id = pdf.add_object(image);
    let mut kids = Vec::new();
    for stream in streams {
        let mut resources = dictionary! { "Font" => dictionary! { "F1" => font } };
        if stream.contains("/Im Do") {
            resources.set("XObject", dictionary! { "Im" => image_id });
        }
        let content = pdf.add_object(Stream::new(Dictionary::new(), stream.as_bytes().to_vec()));
        kids.push(Object::Reference(pdf.add_object(dictionary! {
            "Type" => "Page", "Parent" => tree, "Resources" => resources,
            "Contents" => content
        })));
    }
    pdf.objects.insert(
        tree,
        dictionary! {
            "Type" => "Pages", "Count" => kids.len() as i64, "Kids" => kids,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()]
        }
        .into(),
    );
    let catalog = pdf.add_object(dictionary! { "Type" => "Catalog", "Pages" => tree });
    let info = pdf.add_object(dictionary! { "Title" => Object::string_literal("  Typed pages  ") });
    pdf.trailer.set("Root", catalog);
    pdf.trailer.set("Info", info);
    let mut bytes = Vec::new();
    pdf.save_to(&mut bytes).unwrap();
    (
        bytes,
        format!("pdf-image-{}-{}.png", image_id.0, image_id.1),
    )
}

fn text(value: &str) -> String {
    format!("BT /F1 12 Tf 1 0 0 1 40 700 Tm ({value}) Tj ET")
}

#[test]
fn typed_pages_keep_body_metadata_asset_deduplication_and_default_markers() {
    let (bytes, name) = fixture(
        &[
            text(NATIVE),
            String::new(),
            "/Im Do".into(),
            "/Im Do".into(),
        ],
        false,
    );
    let pages = extract_pages(&bytes).unwrap();
    assert_eq!(pages.pages.len(), 4);
    assert!(pages.document.markdown.is_empty());
    assert_eq!(pages.document.metadata["title"], "Typed pages");
    assert_eq!(pages.document.metadata["converter"], "pdf-inspector");
    assert_eq!(pages.document.assets.len(), 1);
    assert_eq!(pages.document.assets[0].name, name);
    let image = image::load_from_memory(&pages.document.assets[0].bytes)
        .unwrap()
        .to_rgb8();
    assert_eq!(image.dimensions(), (1, 1));
    assert_eq!(image.as_raw(), &[255, 0, 0]);
    assert!(pages.pages[0].markdown.contains(NATIVE));
    assert!(!pages.pages[0].needs_ocr);
    assert!(pages.pages[0].asset_names.is_empty());
    assert!(pages.pages[1].needs_ocr || pages.pages[1].markdown.trim().is_empty());
    for (index, page) in pages.pages.iter().enumerate() {
        assert_eq!(page.number, index + 1);
        assert!(!page.markdown.contains("<!-- Page number:"));
        assert!(!page.markdown.contains(".markitai/assets/"));
        assert!(!page.ocr_completed);
        assert!(page.asset_ocr.is_empty());
        assert!(page.screenshot_name.is_none());
    }
    assert_eq!(
        pages.pages[2].asset_names.as_slice(),
        std::slice::from_ref(&name)
    );
    assert_eq!(
        pages.pages[3].asset_names.as_slice(),
        std::slice::from_ref(&name)
    );
    let native_body = pages.pages[0].markdown.trim().to_owned();
    let document = pages.finish().unwrap();
    assert!(document.markdown.starts_with(&format!(
        "<!-- Page number: 1 -->\n\n{native_body}\n\n<!-- Page number: 2 -->"
    )));
    assert_eq!(document.markdown.matches("<!-- Page number:").count(), 4);
    assert_eq!(
        document
            .markdown
            .matches(&format!(".markitai/assets/{name}"))
            .count(),
        2
    );
    assert_eq!(document.assets.len(), 1);
    assert_eq!(document.warnings.last().unwrap(), LIMITATION);
    assert!(
        document
            .warnings
            .iter()
            .any(|warning| warning.starts_with("PDF page 2: native text was not recovered"))
    );
}

#[test]
fn blank_and_unextractable_image_pages_are_available_before_default_finish_fails() {
    for stream in [String::new(), "/Im Do".into()] {
        let (bytes, _) = fixture(&[stream], true);
        let pages = extract_pages(&bytes).unwrap();
        assert_eq!(pages.pages.len(), 1);
        assert!(pages.pages[0].needs_ocr || pages.pages[0].markdown.trim().is_empty());
        assert!(pages.document.assets.is_empty());
        assert!(
            !pages
                .document
                .warnings
                .iter()
                .any(|warning| warning.contains("OCR is required"))
        );
        let message = pages.finish().unwrap_err().to_string();
        assert!(message.contains("no reliable native text or extractable images"));
        assert!(message.contains("PDF page 1: native text was not recovered"));
        assert!(!message.contains(LIMITATION));
        assert_eq!(extract(&bytes).unwrap_err().to_string(), message);
    }
    let (bytes, _) = fixture(&["/Im Do".into()], false);
    let pages = extract_pages(&bytes).unwrap();
    assert_eq!(pages.document.assets.len(), 1);
    assert!(
        pages.finish().is_ok(),
        "an extractable image preserves default success"
    );
}

#[test]
fn recovering_one_page_retains_native_body_and_inspection_warning_order() {
    let (bytes, _) = fixture(&[text(NATIVE), "/Im Do".into(), "/Im Do".into()], true);
    let mut pages = extract_pages(&bytes).unwrap();
    let native = pages.pages[0].markdown.clone();
    assert_eq!(
        pages
            .document
            .warnings
            .iter()
            .filter(|warning| warning.contains("was not extracted"))
            .count(),
        1
    );
    pages.pages[1].markdown = "Recovered raster text.".into();
    pages.pages[1].needs_ocr = false;
    pages.pages[1].ocr_completed = true;
    pages
        .document
        .metadata
        .insert("ocr_used".into(), true.into());
    pages
        .document
        .warnings
        .push("Explicit caller diagnostic.".into());
    let document = pages.finish().unwrap();
    assert!(document.markdown.contains(native.trim()));
    assert!(
        document
            .markdown
            .contains("<!-- Page number: 2 -->\n\nRecovered raster text.")
    );
    assert_eq!(document.metadata["ocr_used"], true);
    assert!(
        !document
            .warnings
            .iter()
            .any(|warning| warning.starts_with("PDF page 2: native text was not recovered"))
    );
    let image_warning = document
        .warnings
        .iter()
        .position(|warning| warning.contains("was not extracted"))
        .unwrap();
    let missing_warning = document
        .warnings
        .iter()
        .position(|warning| warning.starts_with("PDF page 3: native text was not recovered"))
        .unwrap();
    let caller_warning = document
        .warnings
        .iter()
        .position(|warning| warning == "Explicit caller diagnostic.")
        .unwrap();
    assert!(image_warning < missing_warning && missing_warning < caller_warning);
    assert_eq!(document.warnings.last().unwrap(), LIMITATION);
}

#[test]
fn default_missing_warning_precedes_inspection_failure_and_shared_image_is_not_retried() {
    let (bytes, _) = fixture(&[text(NATIVE), "/Im Do".into(), "/Im Do".into()], true);
    let document = extract(&bytes).unwrap();
    let second = document
        .warnings
        .iter()
        .position(|warning| warning.starts_with("PDF page 2: native text was not recovered"))
        .unwrap();
    let image = document
        .warnings
        .iter()
        .position(|warning| warning.contains("was not extracted"))
        .unwrap();
    let third = document
        .warnings
        .iter()
        .position(|warning| warning.starts_with("PDF page 3: native text was not recovered"))
        .unwrap();
    assert!(second < image && image < third);
    assert_eq!(
        document
            .warnings
            .iter()
            .filter(|warning| warning.contains("was not extracted"))
            .count(),
        1
    );
}

#[test]
fn completed_blank_ocr_does_not_claim_another_ocr_attempt_is_required() {
    let (bytes, _) = fixture(&[String::new()], false);
    let mut pages = extract_pages(&bytes).unwrap();
    pages.pages[0].needs_ocr = false;
    pages.pages[0].ocr_completed = true;
    pages
        .document
        .warnings
        .push("PDF page 1: local OCR completed with no recognized text.".into());
    let document = pages.finish_with_media().unwrap();
    assert_eq!(document.markdown, "<!-- Page number: 1 -->");
    assert!(
        document
            .warnings
            .iter()
            .any(|warning| warning.contains("no recognized text"))
    );
    assert!(
        !document
            .warnings
            .iter()
            .any(|warning| warning.contains("OCR is required"))
    );
    let mut pages = extract_pages(&bytes).unwrap();
    pages.pages[0].needs_ocr = false;
    pages.pages[0].ocr_completed = true;
    assert!(
        pages.finish().is_err(),
        "the default empty-content guard is unchanged"
    );
}

#[test]
fn asset_ocr_is_appended_only_to_its_page_reference_without_changing_native_body() {
    let (bytes, name) = fixture(&[text(NATIVE), "/Im Do".into(), "/Im Do".into()], false);
    let mut pages = extract_pages(&bytes).unwrap();
    let native = pages.pages[0].markdown.clone();
    pages.pages[1]
        .asset_ocr
        .insert(name.clone(), " Recognized image words.\n".into());
    pages.pages[1]
        .asset_ocr
        .insert("not-a-page-asset.png".into(), "MUST NOT APPEAR".into());
    let document = pages.finish().unwrap();
    assert!(document.markdown.contains(native.trim()));
    assert!(document.markdown.contains(&format!("![Image on page 2](.markitai/assets/{name})\n\nRecognized image words.\n\n<!-- Page number: 3 -->")));
    assert_eq!(
        document.markdown.matches("Recognized image words.").count(),
        1
    );
    assert!(!document.markdown.contains("MUST NOT APPEAR"));
    assert_eq!(document.assets.len(), 1);
}

#[test]
fn visibility_routing_uses_inspected_graphics_state_even_when_native_text_survives() {
    let stream = format!("{} q 3 Tr {} Q", text(NATIVE), text("INVISIBLE TEXT"));
    let (bytes, _) = fixture(&[stream, text(NATIVE)], false);
    let pages = extract_pages(&bytes).unwrap();
    assert!(pages.pages[0].visibility_suspect);
    assert!(!pages.pages[1].visibility_suspect);
    assert!(pages.pages[0].markdown.contains(NATIVE));
    assert!(!pages.pages[0].markdown.contains("INVISIBLE TEXT"));
    let document = pages.finish().unwrap();
    assert!(
        document
            .warnings
            .iter()
            .any(|warning| warning.contains("invisible text rendering mode"))
    );
    assert!(document.markdown.contains(NATIVE));
}

#[test]
fn screenshot_comments_follow_their_page_assets_and_keep_the_final_published_name() {
    let (bytes, name) = fixture(&[text(NATIVE), "/Im Do".into()], false);
    let mut pages = extract_pages(&bytes).unwrap();
    pages.pages[0].screenshot_name = Some("page-1.v2.jpg".into());
    pages.pages[1].screenshot_name = Some("page-2.jpg".into());
    pages.pages[1]
        .asset_ocr
        .insert(name.clone(), "Embedded image text.".into());
    let document = pages.finish_with_media().unwrap();
    assert!(document.markdown.contains(
        "<!-- ![Page 1](.markitai/screenshots/page-1.v2.jpg) -->\n\n<!-- Page number: 2 -->"
    ));
    assert!(document.markdown.ends_with(&format!("![Image on page 2](.markitai/assets/{name})\n\nEmbedded image text.\n\n<!-- ![Page 2](.markitai/screenshots/page-2.jpg) -->")));
    assert_eq!(document.markdown.matches("<!-- ![Page ").count(), 2);
}

#[test]
fn screenshot_basename_is_encoded_once_without_comment_or_uri_injection() {
    let (bytes, _) = fixture(&[String::new()], false);
    let mut pages = extract_pages(&bytes).unwrap();
    pages.pages[0].screenshot_name = Some("页 1(50%)-->?#\n.jpg".into());
    let document = pages.finish_with_media().unwrap();
    assert!(document.markdown.ends_with(
        "<!-- ![Page 1](.markitai/screenshots/%E9%A1%B5%201%2850%25%29--%3E%3F%23%0A.jpg) -->"
    ));
    assert_eq!(document.markdown.matches(" -->").count(), 2);
    assert!(!document.markdown.contains("?#"));
}

#[test]
fn bounded_page_extraction_rejects_before_processing_and_accepts_the_exact_limit() {
    let (bytes, _) = fixture(&[text(NATIVE), String::new()], false);
    for limit in [0, 1] {
        assert!(
            matches!(extract_pages_bounded(&bytes, limit), Err(Error::InvalidInput(message)) if message == format!("PDF page count exceeds the {limit}-page limit"))
        );
    }
    let bounded = extract_pages_bounded(&bytes, 2).unwrap().finish().unwrap();
    let default = extract_pages(&bytes).unwrap().finish().unwrap();
    assert_eq!(bounded.markdown, default.markdown);
    assert_eq!(bounded.metadata, default.metadata);
    assert_eq!(bounded.warnings, default.warnings);
    assert_eq!(bounded.assets.len(), default.assets.len());
    assert!(bounded.markdown.contains(NATIVE));
    assert!(bounded.markdown.contains("<!-- Page number: 2 -->"));
    // Empty/unreadable bodies remain reachable at the exact limit, rather than
    // being mistaken for an extraction failure before the media caller runs.
    let (blank, _) = fixture(&[String::new()], false);
    assert_eq!(extract_pages_bounded(&blank, 1).unwrap().pages.len(), 1);
}

#[test]
fn shared_screenshot_reference_uses_the_same_encoded_filename_as_page_assembly() {
    assert_eq!(
        screenshot_reference(12, "图 1(50%)-->?.jpg"),
        "<!-- ![Page 12](.markitai/screenshots/%E5%9B%BE%201%2850%25%29--%3E%3F.jpg) -->"
    );
}

#[test]
fn a_tagged_borderless_table_is_read_from_the_structure_tree() {
    // Wrapped, vertically centred cells drawn without rules: the text's
    // alignment alone splits them into many broken rows.
    let document = extract(include_bytes!("fixtures/borderless-table.pdf")).unwrap();
    assert!(
        document.markdown.contains(
            "|Attribute|Northwind Cold Archive tier €4.00 per terabyte|Contoso Warm Object tier €19.00 per terabyte|Fabrikam Hot Block tier €41.00 per terabyte|\n\
             |---|---|---|---|\n\
             |Retrieval latency in practice|Hours|Minutes|Instant|\n\
             |Durability guarantee|High|Higher|Available on some plans|\n\
             |Minimum storage duration|Ninety days|Thirty days|None|"
        ),
        "{}",
        document.markdown
    );
    assert!(
        document
            .markdown
            .contains("Cold storage was chosen for records older than a year.")
    );
}

#[test]
fn a_tagged_default_style_table_keeps_each_cell_apart() {
    // Cells a few pixels apart: their runs would merge into one text item
    // that keeps only the first cell's marked content.
    let document = extract(include_bytes!("fixtures/default-table.pdf")).unwrap();
    assert!(
        document.markdown.contains(
            "|Language|Year|Typing|Primary Use|\n|---|---|---|---|\n|Python|1991|Dynamic|General purpose|\n|Rust|2015|Static|Systems programming|\n|TypeScript|2012|Static|Web development|"
        ),
        "{}",
        document.markdown
    );
}

#[test]
fn a_tagged_table_among_long_paragraphs_keeps_its_header() {
    // The table holds far less than half of its band's text, and the rest of
    // its own region is its cells: it is fully tagged.
    let document = extract(include_bytes!("fixtures/table-beside-prose.pdf")).unwrap();
    assert!(
        document.markdown.contains(
            "|Service|Peak (MB)|Idle (MB)|\n|---|---|---|\n|Alpha|45|12|\n|Bravo|120|38|\n|Charlie|200|95|"
        ),
        "{}",
        document.markdown
    );
}
