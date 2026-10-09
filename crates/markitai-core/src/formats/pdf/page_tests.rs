//! Authored page streams exercise the extraction/assembly boundary without OCR.
use super::*;
use lopdf::dictionary;

const NATIVE: &str = "Visible native paragraph retained exactly while other pages require OCR.";
const LIMITATION: &str = crate::pdf_media::IMAGE_PLACEMENT;

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
    // The image was not extracted, so none is out of place.
    assert!(document.assets.is_empty());
    assert_eq!(
        document.warnings.last().unwrap(),
        "Explicit caller diagnostic."
    );
}

#[test]
fn only_a_pdf_with_extracted_images_warns_about_their_placement() {
    let (bytes, _) = fixture(&[text(NATIVE)], false);
    let document = extract(&bytes).unwrap();
    assert!(document.assets.is_empty());
    assert!(
        !document
            .warnings
            .iter()
            .any(|warning| warning == LIMITATION),
        "{:?}",
        document.warnings
    );
    let (bytes, _) = fixture(&[text(NATIVE), "/Im Do".into()], false);
    let document = extract(&bytes).unwrap();
    assert!(!document.assets.is_empty());
    assert!(
        document
            .warnings
            .iter()
            .any(|warning| warning == LIMITATION),
        "{:?}",
        document.warnings
    );
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

#[test]
fn a_wrapped_table_without_tagged_cells_continues_on_the_next_page() {
    // No header cells and no rules: Chrome tags the table as layout, so only
    // its geometry gives the rows. Labels and header cells wrap, values are
    // centred between their lines, two header words share one text run,
    // and the last four rows are printed on the next page without a header.
    let pages = extract_pages(include_bytes!("fixtures/wrapped-table.pdf")).unwrap();
    // On its own, the next page's part is headed by its first row.
    assert!(pages.pages[1].continues_table);
    assert!(
        pages.pages[1].markdown.starts_with(
            "|Fibre cement board|X|X||X|X|\n|---|---|---|---|---|---|\n|Zinc||X|X|X|X|\n"
        ),
        "{}",
        pages.pages[1].markdown
    );
    // Assembled, its rows join the table.
    let document = pages.finish().unwrap();
    assert!(
        document.markdown.contains(
            "Blank cells mean the property was not tested.\n\n\
             ||Low thermal conductivity|Corrosion resistant|Recyclable at end of life|Fire rated|Weight class|\n\
             |---|---|---|---|---|---|\n\
             |Brick|X|X|X|X||\n\
             |Stainless steel sheet||X|X|X|X|\n\
             |Cross-laminated timber panel|X||X||X|\n\
             |Glass||X|X|X||\n\
             |Fibre cement board|X|X||X|X|\n\
             |Zinc||X|X|X|X|\n\
             |Recycled plastic composite|X|X|||X|\n\
             |Copper||X|X|X||\n\n\
             <!-- Page number: 2 -->\n\n\
             Brick and zinc were shortlisted for the street facade."
        ),
        "{}",
        document.markdown
    );
}

#[test]
fn a_one_line_paragraph_after_a_table_is_not_a_heading() {
    // Set off by paragraph spacing only, in the body's size and weight and
    // closing a sentence: the page reader took it for a title.
    let document = extract(include_bytes!("fixtures/table-beside-prose.pdf")).unwrap();
    let sentence = "Memory use varies widely between services.";
    let carrying: Vec<&str> = document
        .markdown
        .lines()
        .filter(|line| line.contains(sentence))
        .collect();
    assert_eq!(carrying, [sentence], "{}", document.markdown);
    // The page's real headings keep their sizes' levels.
    for heading in ["# Quarterly storage review", "## Memory usage"] {
        assert!(
            document.markdown.lines().any(|line| line == heading),
            "{heading}\n{}",
            document.markdown
        );
    }
}

#[test]
fn right_to_left_text_reads_in_the_direction_its_lines_are_set() {
    // Arabic in the browser's default direction, flush left: the heading
    // and the paragraph read left to right, the Latin words at their ends
    // where they stand and the full stop last. Then a right-to-left
    // section: its brackets and quotation marks as written, and the cells
    // of its table with their word spaces.
    let document = extract(include_bytes!("fixtures/rtl-text.pdf")).unwrap();
    let markdown = &document.markdown;
    for line in [
        "# مقدمة عن Markitai في Linux",
        "Markitai أداة لتحويل المستندات إلى نص. تدعم ملفات PDF و DOCX وتعمل بسرعة كبيرة على \
         كل الأنظمة، وتكتب النتيجة بصيغة Markdown بشكل منظم.",
        "قال المطور: «النص العربي يقرأ بالترتيب الصحيح» ثم أضاف أن الاختبارات (وعددها 120) \
         نجحت كلها، فابدأ اليوم (الأمر سهل جداً).",
        "|1.3.0|إعادة كتابة بلغة Rust|",
    ] {
        assert!(
            markdown.lines().any(|candidate| candidate == line),
            "{line}\n{markdown}"
        );
    }
}

fn text_at(y: i32, value: &str) -> String {
    format!("BT /F1 12 Tf 1 0 0 1 40 {y} Tm ({value}) Tj ET")
}

#[test]
fn isolated_body_size_lines_are_headings_only_when_they_do_not_read_as_text() {
    // One-line paragraphs of the body's size and weight, apart from the
    // paragraphs around them. A bare title stays a heading; a sentence, the
    // lead-in of a block and the tail of a sentence do not.
    let paragraph = |top: i32| {
        (0..3)
            .map(|line| {
                text_at(
                    top - 14 * line,
                    "The measurements were repeated on every node and each run was recorded twice.",
                )
            })
            .collect::<Vec<_>>()
    };
    let alone = [
        "Experimental Setup",
        "Memory use varies widely between services.",
        "Here is the generated assembly:",
        "with the product",
    ];
    let mut rows = Vec::new();
    for (index, line) in alone.into_iter().enumerate() {
        let top = 720 - index as i32 * 148;
        rows.extend(paragraph(top));
        rows.push(text_at(top - 14 * 3 - 34, line));
    }
    rows.extend(paragraph(720 - alone.len() as i32 * 148));
    let (bytes, _) = fixture(&[rows.join("\n")], false);
    let page = pdf_inspector::extract_pages_markdown_mem(&bytes, None)
        .unwrap()
        .pages
        .remove(0);
    let markdown = &page.markdown;
    let heading = |line: &str| {
        markdown
            .lines()
            .any(|candidate| candidate.starts_with('#') && candidate.ends_with(line))
    };
    assert!(heading(alone[0]), "{markdown}");
    for line in &alone[1..] {
        assert!(markdown.contains(line), "{line}\n{markdown}");
        assert!(!heading(line), "{line}\n{markdown}");
    }
}

#[test]
fn clipped_vector_chart_becomes_an_asset_without_the_screenshot_option() {
    let mut stream = text("An ordinary opening paragraph before the chart.");
    stream.push_str("\nq 70 300 340 200 re W n\n0 G 0.5 w\n");
    for y in [330, 380, 430, 480] {
        stream.push_str(&format!("100 {y} m 320 {y} l S\n"));
    }
    stream.push_str("100 330 m 100 480 l S\n");
    for (x, height) in [(115, 50), (160, 100), (205, 70), (250, 130)] {
        stream.push_str(&format!("0.1 0.3 0.7 rg {x} 330 15 {height} re f\n"));
    }
    for (x, y, label) in [
        (80, 474, "15"),
        (80, 424, "10"),
        (80, 374, "5"),
        (80, 324, "0"),
        (120, 310, "Category"),
        (335, 400, "Series"),
    ] {
        stream.push_str(&format!(
            "0 g BT /F1 9 Tf 1 0 0 1 {x} {y} Tm ({label}) Tj ET\n"
        ));
    }
    stream.push_str(
        "Q\nBT /F1 12 Tf 1 0 0 1 40 250 Tm (The paragraph after the chart stays text.) Tj ET",
    );
    let (bytes, _) = fixture(&[stream], false);
    let document = extract_pages(&bytes).unwrap().finish().unwrap();
    let asset = document
        .assets
        .iter()
        .find(|a| a.name.starts_with("pdf-chart-1-"))
        .unwrap();
    let image = image::load_from_memory(&asset.bytes).unwrap().to_rgb8();
    assert_eq!(image.dimensions(), (680, 400));
    assert!(!document.warnings.iter().any(|w| w == LIMITATION));
    let filtered = extract_pages_policy(&bytes, None, sanitize::Mode::Remove)
        .unwrap()
        .finish()
        .unwrap();
    assert!(
        !filtered
            .assets
            .iter()
            .any(|a| a.name.starts_with("pdf-chart-"))
    );
    assert!(filtered.markdown.contains("Category"));
    assert!(
        image
            .pixels()
            .any(|p| i16::from(p[2]) > i16::from(p[0]) + 50)
    );
    let reference = format!("![Chart](.markitai/assets/{})", asset.name);
    assert!(
        document.markdown.contains(&reference),
        "{}",
        document.markdown
    );
    assert!(!document.markdown.contains("Category"));
    assert!(
        document.markdown.find("opening paragraph").unwrap()
            < document.markdown.find(&reference).unwrap()
    );
    assert!(
        document.markdown.find(&reference).unwrap()
            < document.markdown.find("paragraph after").unwrap()
    );
}

#[test]
fn text_drawn_outside_the_crop_box_is_left_out_with_a_warning() {
    // One line inside the CropBox, one sentinel below and left of it: a
    // viewer shows only the first, upright or turned.
    let stream = [
        text_at(500, "Visible line inside the crop box."),
        "BT /F1 8 Tf 1 0 0 1 5 10 Tm (OUTSIDE CROP MUST NOT APPEAR) Tj ET".into(),
    ]
    .join("\n");
    let (bytes, _) = fixture(&[stream], false);
    for rotate in [0, 90] {
        let mut pdf = lopdf::Document::load_mem(&bytes).unwrap();
        let id = pdf.get_pages()[&1];
        let page = pdf.get_dictionary_mut(id).unwrap();
        page.set(
            "CropBox",
            vec![30.into(), 60.into(), 580.into(), 760.into()],
        );
        page.set("Rotate", rotate);
        let mut cropped = Vec::new();
        pdf.save_to(&mut cropped).unwrap();
        let pages = extract_pages(&cropped).unwrap();
        let markdown = &pages.pages[0].markdown;
        assert!(
            markdown.contains("Visible line inside the crop box."),
            "{markdown}"
        );
        assert!(!markdown.contains("OUTSIDE CROP"), "{rotate}: {markdown}");
        assert!(
            pages
                .document
                .warnings
                .iter()
                .any(|w| w
                    .starts_with("PDF page 1: 1 text item(s) drawn outside the visible page area")),
            "{:?}",
            pages.document.warnings
        );
    }
}

#[test]
fn images_are_placed_where_they_are_drawn_among_the_lines() {
    // A banner above the text, a figure between two paragraphs and a
    // footer image below the text, drawn in the reverse order.
    let mut pdf = lopdf::Document::with_version("1.7");
    let tree = pdf.new_object_id();
    let font = pdf.add_object(dictionary! {
        "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica"
    });
    let mut xobjects = Dictionary::new();
    for (index, rgb) in [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        .into_iter()
        .enumerate()
    {
        let image = pdf.add_object(Stream::new(
            dictionary! {
                "Type" => "XObject", "Subtype" => "Image", "Width" => 1,
                "Height" => 1, "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8
            },
            rgb.to_vec(),
        ));
        xobjects.set(format!("Im{index}"), image);
    }
    let stream = [
        "q 300 0 0 40 40 100 cm /Im2 Do Q".to_owned(),
        "q 300 0 0 60 40 520 cm /Im1 Do Q".to_owned(),
        "q 300 0 0 50 40 730 cm /Im0 Do Q".to_owned(),
        text_at(
            700,
            "The opening paragraph starts right below the banner image.",
        ),
        text_at(
            686,
            "It continues on a second line before the figure is drawn.",
        ),
        text_at(
            480,
            "The closing paragraph follows the figure in the reading order.",
        ),
    ]
    .join("\n");
    let content = pdf.add_object(Stream::new(Dictionary::new(), stream.into_bytes()));
    let page = pdf.add_object(dictionary! {
        "Type" => "Page", "Parent" => tree, "Contents" => content,
        "Resources" => dictionary! { "Font" => dictionary! { "F1" => font }, "XObject" => xobjects }
    });
    pdf.objects.insert(
        tree,
        dictionary! {
            "Type" => "Pages", "Count" => 1, "Kids" => vec![page.into()],
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()]
        }
        .into(),
    );
    let catalog = pdf.add_object(dictionary! { "Type" => "Catalog", "Pages" => tree });
    pdf.trailer.set("Root", catalog);
    let mut bytes = Vec::new();
    pdf.save_to(&mut bytes).unwrap();
    let document = extract_pages(&bytes).unwrap().finish().unwrap();
    let markdown = &document.markdown;
    let at = |needle: &str| {
        markdown
            .find(needle)
            .unwrap_or_else(|| panic!("{needle}\n{markdown}"))
    };
    let image = |index: u8| {
        let names: Vec<_> = document
            .assets
            .iter()
            .map(|asset| asset.name.as_str())
            .collect();
        at(&format!(".markitai/assets/{}", names[index as usize]))
    };
    // Assets are named in drawing order: footer, figure, banner.
    assert!(image(2) < at("The opening paragraph"), "{markdown}");
    assert!(at("before the figure is drawn.") < image(1), "{markdown}");
    assert!(image(1) < at("The closing paragraph"), "{markdown}");
    assert!(at("in the reading order.") < image(0), "{markdown}");
    assert!(
        !document.warnings.iter().any(|w| w == LIMITATION),
        "{:?}",
        document.warnings
    );
}

#[test]
fn a_justified_word_break_is_one_space() {
    // Justified text widens each break with an offset before its space
    // glyph; a double space between words is two glyphs and stays.
    let stream = [
        "BT /F1 12 Tf 1 0 0 1 40 700 Tm [(Vivamus) -400 ( dapibus) -400 ( sodales) -400 ( ex,)] TJ ET",
        "BT /F1 12 Tf 1 0 0 1 40 680 Tm [(Two  spaces) ( stay  apart.)] TJ ET",
    ]
    .join("\n");
    let (bytes, _) = fixture(&[stream], false);
    let pages = extract_pages(&bytes).unwrap();
    let markdown = &pages.pages[0].markdown;
    assert!(
        markdown.contains("Vivamus dapibus sodales ex,"),
        "{markdown:?}"
    );
    assert!(
        markdown.contains("Two  spaces stay  apart."),
        "{markdown:?}"
    );
}

#[test]
fn a_photo_the_page_binds_but_never_draws_does_not_make_it_a_scan() {
    // Every page of a document binds the same large photo; this one shows
    // only a few lines of text and never draws it.
    let mut pdf = lopdf::Document::with_version("1.7");
    let tree = pdf.new_object_id();
    let font = pdf.add_object(dictionary! {
        "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica"
    });
    let photo = pdf.add_object(Stream::new(
        dictionary! {
            "Type" => "XObject", "Subtype" => "Image", "Width" => 1000,
            "Height" => 1000, "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8
        },
        vec![128; 1000 * 1000],
    ));
    let stream = [
        text_at(700, "A page of ordinary text that happens to bind a photo."),
        text_at(686, "Its content never draws that photo anywhere at all."),
    ]
    .join("\n");
    let content = pdf.add_object(Stream::new(Dictionary::new(), stream.into_bytes()));
    let page = pdf.add_object(dictionary! {
        "Type" => "Page", "Parent" => tree, "Contents" => content,
        "Resources" => dictionary! {
            "Font" => dictionary! { "F1" => font },
            "XObject" => dictionary! { "Photo" => photo }
        }
    });
    pdf.objects.insert(
        tree,
        dictionary! {
            "Type" => "Pages", "Count" => 1, "Kids" => vec![page.into()],
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()]
        }
        .into(),
    );
    let catalog = pdf.add_object(dictionary! { "Type" => "Catalog", "Pages" => tree });
    pdf.trailer.set("Root", catalog);
    let mut bytes = Vec::new();
    pdf.save_to(&mut bytes).unwrap();
    let pages = extract_pages(&bytes).unwrap();
    assert!(!pages.pages[0].needs_ocr);
    assert!(pages.pages[0].markdown.contains("happens to bind a photo"));
    assert!(
        !pages
            .document
            .warnings
            .iter()
            .any(|w| w.contains("looked like a scan")),
        "{:?}",
        pages.document.warnings
    );
}
