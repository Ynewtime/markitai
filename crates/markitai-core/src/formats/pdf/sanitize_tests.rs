//! Authored operator fixtures; expected visible content is independent of
//! another extractor or a post-layout string replacement.
use super::*;
use lopdf::dictionary;
use serde_json::json;

const INTRO: &str = "The visible harbour report contains readable ordinary paragraphs.";

fn text(y: usize, value: &str) -> String {
    format!("BT /F1 12 Tf 1 0 0 1 40 {y} Tm ({value}) Tj ET\n")
}

pub(super) fn fixture(
    chunks: &[Vec<u8>],
    forms: &[(&str, Vec<u8>, Option<Dictionary>)],
) -> Vec<u8> {
    let mut pdf = lopdf::Document::with_version("1.7");
    let tree = pdf.new_object_id();
    let font = pdf.add_object(
        dictionary! {"Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica"},
    );
    let bold = pdf.add_object(
        dictionary! {"Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica-Bold"},
    );
    let zero = pdf.add_object(dictionary! {"Type" => "ExtGState", "ca" => 0.0});
    let stroke_zero = pdf.add_object(dictionary! {"Type" => "ExtGState", "CA" => 0.0});
    let opaque = pdf.add_object(dictionary! {"Type" => "ExtGState", "ca" => 1.0, "CA" => 1.0});
    let font_only =
        pdf.add_object(dictionary! {"Type" => "ExtGState", "Font" => vec![font.into(), 12.into()]});
    let unknown = pdf.add_object(dictionary! {"Type" => "ExtGState", "BM" => "Multiply"});
    let mut resources = dictionary! {
        "Font" => dictionary! {"F1" => font, "F2" => bold},
        "ExtGState" => dictionary! {"Zero" => zero, "StrokeZero" => stroke_zero, "Opaque" => opaque, "FontOnly" => font_only, "Unknown" => unknown}
    };
    let mut xobjects = Dictionary::new();
    let ids: Vec<_> = forms.iter().map(|_| pdf.new_object_id()).collect();
    for ((name, _, _), id) in forms.iter().zip(&ids) {
        xobjects.set(*name, *id);
    }
    resources.set("XObject", xobjects);
    for ((_, bytes, own), id) in forms.iter().zip(ids) {
        let mut dictionary = dictionary! {"Type" => "XObject", "Subtype" => "Form", "BBox" => vec![0.into(),0.into(),612.into(),792.into()]};
        if let Some(own) = own {
            dictionary.set("Resources", own.clone());
        }
        pdf.objects
            .insert(id, Stream::new(dictionary, bytes.clone()).into());
    }
    let streams: Vec<Object> = chunks
        .iter()
        .map(|chunk| {
            let mut stream = Stream::new(Dictionary::new(), chunk.clone());
            stream.compress().unwrap();
            Object::Reference(pdf.add_object(stream))
        })
        .collect();
    let page =
        pdf.add_object(dictionary! {"Type" => "Page", "Parent" => tree, "Contents" => streams});
    // Resources inherited from Pages deliberately cover the page/resource path.
    pdf.objects.insert(tree, dictionary! {"Type" => "Pages", "Count" => 1, "Kids" => vec![page.into()], "Resources" => resources, "MediaBox" => vec![0.into(),0.into(),612.into(),792.into()]}.into());
    let catalog = pdf.add_object(dictionary! {"Type" => "Catalog", "Pages" => tree});
    pdf.trailer.set("Root", catalog);
    let mut bytes = Vec::new();
    pdf.save_to(&mut bytes).unwrap();
    bytes
}

fn convert(bytes: &[u8], mode: &str) -> Document {
    extract_with_config(bytes, &json!({"security":{"pdf_sanitize":mode}})).unwrap()
}

#[test]
fn off_warn_and_remove_have_distinct_policy_without_deleting_visible_duplicates() {
    let repeated = "Shared words occur visibly in the ordinary report.";
    let content = format!(
        "{}{}q 1 g {}Q q /Zero gs {}Q BT /F1 0.5 Tf 1 0 0 1 40 560 Tm (TINY UNIQUE HIDDEN TOKEN) Tj ET\n{}",
        text(740, INTRO),
        text(700, repeated),
        text(660, repeated),
        text(620, "TRANSPARENT UNIQUE HIDDEN TOKEN"),
        text(
            520,
            "The final visible paragraph must remain in every policy."
        )
    );
    let bytes = fixture(&[content.into_bytes()], &[]);
    let original = bytes.clone();
    let off = convert(&bytes, "off");
    let warn = convert(&bytes, "warn");
    let removed = convert(&bytes, "remove");
    assert_eq!(off.markdown, warn.markdown);
    assert_eq!(
        off.markdown.matches(repeated).count(),
        2,
        "{}",
        off.markdown
    );
    assert_eq!(
        removed.markdown.matches(repeated).count(),
        1,
        "{}",
        removed.markdown
    );
    assert!(off.markdown.contains("TRANSPARENT UNIQUE HIDDEN TOKEN"));
    assert!(!removed.markdown.contains("TRANSPARENT UNIQUE HIDDEN TOKEN"));
    assert!(!removed.markdown.contains("TINY UNIQUE HIDDEN TOKEN"));
    assert!(removed.markdown.contains("final visible paragraph"));
    assert!(!off.warnings.iter().any(|w| w.contains("hidden-text")
        || w.contains("contains white text")
        || w.contains("transparent text graphics state")));
    assert!(warn.warnings.iter().any(|w| w.contains("white text")));
    assert!(
        removed
            .warnings
            .iter()
            .any(|w| w.contains("pdf_sanitize=remove filtered"))
    );
    assert_eq!(bytes, original);
}

#[test]
fn black_background_white_text_is_actually_preserved_in_all_policies() {
    let content = format!(
        "{}q 0 g 30 600 540 80 re f 1 g {}Q {}",
        text(740, INTRO),
        text(640, "VISIBLE WHITE ON BLACK PANEL"),
        text(
            540,
            "The ordinary report continues below the highlighted panel."
        )
    );
    let bytes = fixture(&[content.into_bytes()], &[]);
    for mode in ["off", "warn", "remove"] {
        let document = convert(&bytes, mode);
        assert!(
            document.markdown.contains("VISIBLE WHITE ON BLACK PANEL"),
            "{mode}: {}",
            document.markdown
        );
        if mode == "remove" {
            assert!(document.warnings.iter().any(|w| {
                w.contains("white text over a painted or unknown background was retained")
            }));
        }
    }
}

#[test]
fn opacity_channels_and_missing_extgstate_entries_inherit_without_hiding_visible_strokes() {
    let content = format!(
        "{}q /Zero gs /FontOnly gs {}Q q /Zero gs 1 Tr {}Q q /StrokeZero gs 0 Tr {}Q q /Zero gs /StrokeZero gs 2 Tr {}Q {}",
        text(750, INTRO),
        text(710, "HIDDEN AFTER FONT ONLY STATE"),
        text(670, "VISIBLE STROKE WITH TRANSPARENT FILL"),
        text(630, "VISIBLE FILL WITH TRANSPARENT STROKE"),
        text(590, "HIDDEN BOTH PAINT CHANNELS"),
        text(550, "VISIBLE AFTER GRAPHICS RESTORE")
    );
    let bytes = fixture(&[content.into_bytes()], &[]);
    let removed = convert(&bytes, "remove");
    assert!(!removed.markdown.contains("HIDDEN AFTER FONT ONLY STATE"));
    assert!(!removed.markdown.contains("HIDDEN BOTH PAINT CHANNELS"));
    for text in [
        "VISIBLE STROKE WITH TRANSPARENT FILL",
        "VISIBLE FILL WITH TRANSPARENT STROKE",
        "VISIBLE AFTER GRAPHICS RESTORE",
    ] {
        assert!(
            removed.markdown.contains(text),
            "{text}: {}",
            removed.markdown
        );
    }
}

#[test]
fn shared_forms_are_filtered_per_call_and_nested_local_resources_override_inheritance() {
    let content = format!(
        "{}q /Zero gs /Shared Do Q q 1 0 0 1 0 -80 cm /Shared Do Q {}",
        text(740, INTRO),
        text(
            500,
            "The last visible paragraph follows both Form invocations."
        )
    );
    let form = text(680, "SHARED FORM TEXT MUST REMAIN ONCE").into_bytes();
    let bytes = fixture(&[content.into_bytes()], &[("Shared", form, None)]);
    let off = convert(&bytes, "off");
    assert_eq!(
        off.markdown
            .matches("SHARED FORM TEXT MUST REMAIN ONCE")
            .count(),
        2
    );
    let removed = convert(&bytes, "remove");
    assert_eq!(
        removed
            .markdown
            .matches("SHARED FORM TEXT MUST REMAIN ONCE")
            .count(),
        1,
        "{}",
        removed.markdown
    );
    let own = dictionary! {"Font" => dictionary! {"F1" => dictionary! {"Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica"}}, "ExtGState" => dictionary! {"Zero" => dictionary! {"Type" => "ExtGState", "ca" => 1.0}}};
    let outer = format!("{} /Inner Do", text(680, "HIDDEN IN OUTER FORM")).into_bytes();
    let inner = format!("/Zero gs {}", text(640, "VISIBLE LOCAL RESOURCE OVERRIDE")).into_bytes();
    let content = format!(
        "{}q /Zero gs /Outer Do Q {}",
        text(740, INTRO),
        text(540, "VISIBLE AFTER NESTED FORMS")
    );
    let bytes = fixture(
        &[content.into_bytes()],
        &[("Outer", outer, None), ("Inner", inner, Some(own))],
    );
    let removed = convert(&bytes, "remove");
    assert!(!removed.markdown.contains("HIDDEN IN OUTER FORM"));
    assert!(
        removed.markdown.contains("VISIBLE LOCAL RESOURCE OVERRIDE"),
        "{}",
        removed.markdown
    );
    assert!(removed.markdown.contains("VISIBLE AFTER NESTED FORMS"));
}

#[test]
fn scaled_normal_text_and_show_spacing_survive_compressed_stream_boundaries() {
    let first = format!(
        "{}q /Zero gs BT /F1 12 Tf 1 0 0 1 40 690 Tm (HIDDEN BETWEEN STREAMS) Tj ET\n",
        text(750, INTRO)
    );
    let second = "Q BT /F1 1 Tf 12 0 0 12 40 640 Tm (VISIBLE SCALED NORMAL TEXT) Tj ET\nBT /F1 12 Tf 1 0 0 1 40 580 Tm (Visible before ) Tj q /Zero gs [(HIDDEN) -150 (ADVANCE)] TJ Q /F2 12 Tf (VISIBLE AFTER ADVANCE) Tj ET\n";
    let bytes = fixture(&[first.into_bytes(), second.as_bytes().to_vec()], &[]);
    let removed = convert(&bytes, "remove");
    assert!(!removed.markdown.contains("HIDDEN"), "{}", removed.markdown);
    assert!(removed.markdown.contains("VISIBLE SCALED NORMAL TEXT"));
    assert!(removed.markdown.contains("VISIBLE AFTER ADVANCE"));
    // The larger bold line is a heading in the actual native layout reader.
    assert!(
        removed
            .markdown
            .contains("## Visible before VISIBLE AFTER ADVANCE")
    );
}

#[test]
fn invisible_modes_and_actualtext_do_not_reenter_a_visible_body() {
    for rendering in [3, 7] {
        let content = format!(
            "{}q {rendering} Tr BT /F1 12 Tf 1 0 0 1 40 640 Tm /Span << /ActualText (HIDDEN REPLACEMENT TOKEN) >> BDC (HIDDEN GLYPH TOKEN) Tj EMC ET Q {}",
            text(740, INTRO),
            text(540, "Visible after the nonpainting marked content.")
        );
        let bytes = fixture(&[content.into_bytes()], &[]);
        for mode in ["off", "warn", "remove"] {
            let document = convert(&bytes, mode);
            assert!(
                !document.markdown.contains("HIDDEN"),
                "{rendering} {mode}: {}",
                document.markdown
            );
            assert!(document.markdown.contains("Visible after"));
        }
    }
}

#[test]
fn unknown_visibility_warns_honestly_without_claiming_a_complete_cleaning() {
    let content = format!(
        "{}q /Unknown gs {}Q",
        text(740, INTRO),
        text(
            640,
            "Visible multiply-blended paragraph has an unknown backdrop."
        )
    );
    let bytes = fixture(&[content.into_bytes()], &[]);
    let off = convert(&bytes, "off");
    let warn = convert(&bytes, "warn");
    let removed = convert(&bytes, "remove");
    assert_eq!(off.markdown, warn.markdown);
    assert_eq!(warn.markdown, removed.markdown);
    assert!(
        !off.warnings
            .iter()
            .any(|w| w.contains("hidden-text inspection"))
    );
    assert!(
        warn.warnings
            .iter()
            .any(|w| w.contains("hidden-text inspection could not determine"))
    );
    assert!(!removed.warnings.iter().any(|w| w.contains("filtered")));
}

#[test]
fn bounded_filtering_keeps_original_body_when_form_inspection_is_incomplete() {
    let mut forms = Vec::new();
    let names: Vec<_> = (0..34).map(|n| format!("Nested{n}")).collect();
    for n in 0..34 {
        let body = if n == 33 {
            text(640, "DEEP HIDDEN TEXT").into_bytes()
        } else {
            format!("/{} Do", names[n + 1]).into_bytes()
        };
        forms.push((names[n].as_str(), body, None));
    }
    let content = format!("{}q /Zero gs /Nested0 Do Q", text(740, INTRO));
    let bytes = fixture(&[content.into_bytes()], &forms);
    let original = convert(&bytes, "warn");
    let removed = convert(&bytes, "remove");
    assert_eq!(original.markdown, removed.markdown);
    assert!(
        removed
            .warnings
            .iter()
            .any(|w| w.contains("content inspection is incomplete")),
        "{:?}",
        removed.warnings
    );
}

#[test]
fn invalid_policy_is_rejected_without_changing_input() {
    let bytes = fixture(&[text(740, INTRO).into_bytes()], &[]);
    assert!(matches!(
        extract_with_config(&bytes, &json!({"security":{"pdf_sanitize":"erase"}})),
        Err(Error::Config(_))
    ));
}

#[test]
fn nearest_resources_replace_the_whole_ancestor_dictionary() {
    let mut pdf =
        lopdf::Document::load_mem(&fixture(&[text(740, INTRO).into_bytes()], &[])).unwrap();
    let page_id = pdf.get_pages()[&1];
    let parent_id = pdf
        .get_dictionary(page_id)
        .unwrap()
        .get(b"Parent")
        .unwrap()
        .as_reference()
        .unwrap();
    let original = pdf
        .get_dictionary(parent_id)
        .unwrap()
        .get(b"Resources")
        .unwrap()
        .as_dict()
        .unwrap();
    let fonts = original.get(b"Font").unwrap().clone();
    let local = dictionary! {"Font" => fonts};
    pdf.get_dictionary_mut(page_id)
        .unwrap()
        .set("Resources", local.clone());
    let nearest = sanitize::page_resources(&pdf, page_id).unwrap().unwrap();
    assert!(!nearest.has(b"ExtGState"));
    assert!(!nearest.has(b"XObject"));
    assert!(
        sanitize::normalize_inherited_resources(&pdf, 4096)
            .unwrap()
            .is_none()
    );
    let mut bytes = Vec::new();
    pdf.save_to(&mut bytes).unwrap();
    assert!(convert(&bytes, "off").markdown.contains(INTRO));
    let content = format!(
        "{}q /Zero gs {}Q",
        text(740, INTRO),
        text(650, "UNKNOWN LOCAL RESOURCE STATE MUST REMAIN")
    );
    let stream = pdf.add_object(Stream::new(Dictionary::new(), content.into_bytes()));
    pdf.get_dictionary_mut(page_id)
        .unwrap()
        .set("Contents", stream);
    bytes.clear();
    pdf.save_to(&mut bytes).unwrap();
    let result = convert(&bytes, "remove");
    assert!(
        result
            .markdown
            .contains("UNKNOWN LOCAL RESOURCE STATE MUST REMAIN")
    );
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("unresolved graphics state"))
    );
}

#[test]
fn invalid_nearest_resources_and_parent_cycles_are_bounded_and_do_not_borrow_ancestors() {
    let mut pdf =
        lopdf::Document::load_mem(&fixture(&[text(740, INTRO).into_bytes()], &[])).unwrap();
    let page_id = pdf.get_pages()[&1];
    pdf.get_dictionary_mut(page_id).unwrap().set("Resources", 3);
    assert!(sanitize::page_resources(&pdf, page_id).is_err());
    assert!(
        sanitize::normalize_inherited_resources(&pdf, 4096)
            .unwrap()
            .is_none()
    );
    let (inspection, _) = inspect_page(&pdf, page_id);
    assert!(inspection.incomplete);
    pdf.get_dictionary_mut(page_id)
        .unwrap()
        .remove(b"Resources");
    pdf.get_dictionary_mut(page_id)
        .unwrap()
        .set("Parent", page_id);
    assert!(
        sanitize::page_resources(&pdf, page_id)
            .unwrap_err()
            .contains("cycle")
    );
    let mut long = Dictionary::new();
    for _ in 0..70 {
        let id = pdf.add_object(long);
        long = dictionary! {"Parent"=>id};
    }
    let id = pdf.add_object(long);
    assert!(
        sanitize::page_resources(&pdf, id)
            .unwrap_err()
            .contains("limit")
    );
}

#[test]
fn recursive_form_and_invalid_xobject_keep_an_honest_inspection_boundary() {
    let content = format!("{}q /Zero gs /Recursive Do Q", text(740, INTRO));
    let bytes = fixture(
        &[content.into_bytes()],
        &[("Recursive", b"/Recursive Do".to_vec(), None)],
    );
    let pdf = lopdf::Document::load_mem(&bytes).unwrap();
    let (inspection, _) = inspect_page(&pdf, pdf.get_pages()[&1]);
    assert!(inspection.incomplete);
    assert!(
        inspection
            .warnings
            .iter()
            .any(|w| w.contains("Recursive Form"))
    );
    let bytes = fixture(
        &[format!("{} /Missing Do", text(740, INTRO)).into_bytes()],
        &[],
    );
    let result = convert(&bytes, "remove");
    assert!(result.markdown.contains(INTRO));
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("resource could not be resolved"))
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("remove filtered"))
    );
}

#[test]
fn standalone_authored_policy_and_scan_fixtures_pass_the_actual_native_reader() {
    let bytes = include_bytes!("fixtures/sanitize/policy-text.pdf");
    let off = convert(bytes, "off");
    let warn = convert(bytes, "warn");
    let removed = convert(bytes, "remove");
    assert_eq!(off.markdown, warn.markdown);
    assert_eq!(
        off.markdown.matches("Shared words occur visibly").count(),
        2
    );
    assert_eq!(
        removed
            .markdown
            .matches("Shared words occur visibly")
            .count(),
        1
    );
    assert!(!removed.markdown.contains("HIDDEN TOKEN"));
    let scan = include_bytes!("fixtures/sanitize/policy-searchable-scan.pdf");
    for mode in ["off", "warn", "remove"] {
        let doc = convert(scan, mode);
        assert_eq!(doc.metadata["ocr_layer_pages"], json!([1]));
        assert!(doc.markdown.contains("The harbour handled more ships"));
        assert!(!doc.warnings.iter().any(|w| w.contains("remove filtered")));
    }
}

#[test]
fn filtering_an_extra_hidden_run_preserves_real_tagged_table_cells_and_prose() {
    for original in [
        include_bytes!("fixtures/default-table.pdf").as_slice(),
        include_bytes!("fixtures/borderless-table.pdf").as_slice(),
        include_bytes!("fixtures/table-beside-prose.pdf").as_slice(),
    ] {
        let expected = convert(original, "warn").markdown;
        let mut pdf = lopdf::Document::load_mem(original).unwrap();
        let page = pdf.get_pages()[&1];
        let mut resources = sanitize::page_resources(&pdf, page)
            .unwrap()
            .unwrap()
            .clone();
        let font = pdf
            .add_object(dictionary! {"Type"=>"Font", "Subtype"=>"Type1", "BaseFont"=>"Helvetica"});
        let zero = pdf.add_object(dictionary! {"Type"=>"ExtGState", "ca"=>0.0});
        let mut fonts = pdf.get_dict_in_dict(&resources, b"Font").unwrap().clone();
        fonts.set("MarkitaiTestFont", font);
        resources.set("Font", fonts);
        let mut states = pdf
            .get_dict_in_dict(&resources, b"ExtGState")
            .cloned()
            .unwrap_or_default();
        states.set("MarkitaiTestZero", zero);
        resources.set("ExtGState", states);
        let content = pdf
            .get_page_content_with_limit(page, MAX_STREAM_BYTES)
            .unwrap();
        let mut combined = content;
        combined.extend_from_slice(b"\nq /MarkitaiTestZero gs BT /MarkitaiTestFont 12 Tf 1 0 0 1 40 10 Tm (HIDDEN TABLE INJECTION) Tj ET Q\n");
        let stream = pdf.add_object(Stream::new(Dictionary::new(), combined));
        let dict = pdf.get_dictionary_mut(page).unwrap();
        dict.set("Contents", stream);
        dict.set("Resources", resources);
        let mut bytes = Vec::new();
        pdf.save_to(&mut bytes).unwrap();
        let removed = convert(&bytes, "remove");
        assert!(!removed.markdown.contains("HIDDEN TABLE INJECTION"));
        assert_eq!(
            removed.markdown, expected,
            "visible table/prose changed: {:?}",
            removed.warnings
        );
    }
}
