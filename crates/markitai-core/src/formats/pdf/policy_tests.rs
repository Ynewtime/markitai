//! Independently authored PDFs for the native reader's content-retention policy.
use super::*;
use lopdf::dictionary;

fn pdf(pages: &[Vec<u8>], forms: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut doc = lopdf::Document::with_version("1.7");
    let tree = doc.new_object_id();
    let font = doc.add_object(dictionary! {
        "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica"
    });
    let ids = forms
        .iter()
        .map(|_| doc.new_object_id())
        .collect::<Vec<_>>();
    let mut xobjects = Dictionary::new();
    for ((name, _), id) in forms.iter().zip(&ids) {
        xobjects.set(*name, *id);
    }
    let resources = dictionary! {
        "Font" => dictionary! { "F1" => font }, "XObject" => xobjects
    };
    for ((_, content), id) in forms.iter().zip(ids) {
        doc.objects.insert(
            id,
            Stream::new(
                dictionary! {
                    "Type" => "XObject", "Subtype" => "Form", "FormType" => 1,
                    "BBox" => vec![0.into(),0.into(),612.into(),792.into()],
                    "Resources" => resources.clone()
                },
                content.clone(),
            )
            .into(),
        );
    }
    let mut kids = Vec::new();
    for content in pages {
        let mut stream = Stream::new(Dictionary::new(), content.clone());
        stream.compress().unwrap();
        let stream = doc.add_object(stream);
        let page = doc.add_object(dictionary! {
            "Type" => "Page", "Parent" => tree, "Contents" => stream,
            "Resources" => resources.clone()
        });
        kids.push(Object::Reference(page));
    }
    doc.objects.insert(
        tree,
        dictionary! {
            "Type" => "Pages", "Count" => kids.len() as i64, "Kids" => kids,
            "MediaBox" => vec![0.into(),0.into(),612.into(),792.into()]
        }
        .into(),
    );
    let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => tree });
    doc.trailer.set("Root", catalog);
    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).unwrap();
    bytes
}

fn text(y: usize, text: &str) -> String {
    format!("BT /F1 10 Tf 1 0 0 1 40 {y} Tm ({text}) Tj ET\n")
}

#[test]
fn every_page_prefixed_paragraph_survives_two_and_forty_page_documents() {
    for count in [2, 40] {
        let pages = (1..=count)
            .map(|page| {
                let mut content =
                    "BT /F1 16 Tf 1 0 0 1 40 760 Tm (Decode benchmark) Tj ET\n".to_owned();
                for row in 1..=36 {
                    content.push_str(&text(
                        730 - (row - 1) * 18,
                        &format!("Page {page} row {row}: the complete visible source paragraph."),
                    ));
                }
                content.into_bytes()
            })
            .collect::<Vec<_>>();
        let document = extract(&pdf(&pages, &[])).unwrap();
        let mut offset = 0;
        for page in 1..=count {
            for row in 1..=36 {
                let expected =
                    format!("Page {page} row {row}: the complete visible source paragraph.");
                let position = document.markdown[offset..]
                    .find(&expected)
                    .unwrap_or_else(|| panic!("missing {expected}"));
                offset += position + expected.len();
                assert_eq!(document.markdown.matches(&expected).count(), 1);
            }
        }
        assert_eq!(
            document
                .markdown
                .matches("complete visible source paragraph.")
                .count(),
            count * 36
        );
        assert_eq!(
            document.markdown.matches("<!-- Page number:").count(),
            count
        );
    }
}

#[test]
fn complete_folios_are_removed_but_page_references_remain_substantive_prose() {
    let pages = (1..=3)
        .map(|page| {
            let mut content = text(
                700,
                "Page 42 explains the result and must remain visible prose.",
            );
            content.push_str(&text(
                650,
                "Page 3 of 10 contains the experimental method, not a folio.",
            ));
            content.push_str(&text(
                600,
                "Page of 10 examples is an awkward but meaningful sentence.",
            ));
            content.push_str(&text(20, &format!("Page {page} of 3")));
            content.into_bytes()
        })
        .collect::<Vec<_>>();
    let bytes = pdf(&pages, &[]);
    let baseline = pdf_inspector::extract_pages_markdown_mem(&bytes, None).unwrap();
    let document = extract(&bytes).unwrap();
    assert_eq!(
        document
            .markdown
            .matches("Page 42 explains the result")
            .count(),
        3
    );
    assert_eq!(
        document.markdown.matches("Page 3 of 10 contains").count(),
        3
    );
    assert_eq!(document.markdown.matches("Page of 10 examples").count(), 3);
    for page in 1..=3 {
        assert!(
            !document.markdown.contains(&format!("Page {page} of 3")),
            "baseline={baseline:?}; final={}",
            document.markdown
        );
    }
}

#[test]
fn rendering_modes_persist_across_text_objects_and_nested_graphics_states() {
    for mode in [3, 7] {
        let content = format!(
            "{}q {mode} Tr {}q 0 Tr {}Q {}Q {}",
            text(740, "Visible before the hidden state."),
            text(700, "HIDDEN BEFORE NESTED STATE"),
            text(660, "Visible nested state is explicitly restored."),
            text(620, "HIDDEN AFTER NESTED STATE"),
            text(580, "Visible after restoring the outer state.")
        );
        let bytes = pdf(&[content.into_bytes()], &[]);
        let document = extract(&bytes).unwrap();
        assert!(
            !document.markdown.contains("HIDDEN"),
            "mode {mode}: {}",
            document.markdown
        );
        assert!(document.markdown.contains("Visible before"));
        assert!(document.markdown.contains("Visible nested"));
        assert!(document.markdown.contains("Visible after"));
        assert!(
            document
                .warnings
                .iter()
                .any(|warning| warning.contains("invisible text rendering mode"))
        );
    }
}

#[test]
fn hidden_page_and_form_text_operators_preserve_the_following_glyph_position() {
    for mode in [3, 7] {
        for operator in [
            "(SECRET) Tj",
            "[(SE) -250 (CRET)] TJ",
            "(SECRET) '",
            "0 0 (SECRET) \"",
        ] {
            for in_form in [false, true] {
                let make = |mode| {
                    let content = format!(
                        "BT /F1 12 Tf 14 TL 1 0 0 1 40 700 Tm (Before) Tj q {mode} Tr {operator} Q /F1 16 Tf (After) Tj ET"
                    );
                    if in_form {
                        pdf(&[b"/Inner Do".to_vec()], &[("Inner", content.into_bytes())])
                    } else {
                        pdf(&[content.into_bytes()], &[])
                    }
                };
                let visible = pdf_inspector::extract_text_with_positions_mem(&make(0)).unwrap();
                let hidden = pdf_inspector::extract_text_with_positions_mem(&make(mode)).unwrap();
                assert!(
                    !hidden.iter().any(|item| item.text.contains("SECRET")),
                    "mode={mode}, operator={operator}, form={in_form}"
                );
                let position = |items: &[pdf_inspector::TextItem]| {
                    let item = items
                        .iter()
                        .find(|item| item.text == "After")
                        .unwrap_or_else(|| panic!("following visible run: {items:?}"));
                    (item.x, item.y)
                };
                assert_eq!(
                    position(&hidden),
                    position(&visible),
                    "mode={mode}, operator={operator}, form={in_form}"
                );
            }
        }
    }
}

#[test]
fn nested_forms_inherit_hidden_state_without_leaking_it_to_the_caller() {
    for mode in [3, 7] {
        let outer = format!(
            "{}q 0 Tr {}Q /Inner Do",
            text(700, "HIDDEN OUTER FORM"),
            text(660, "Visible nested form content.")
        );
        let main = format!(
            "{}q {mode} Tr /Outer Do Q {}",
            text(740, "Visible page content before forms."),
            text(540, "Visible page content after forms.")
        );
        let document = extract(&pdf(
            &[main.into_bytes()],
            &[
                ("Outer", outer.into_bytes()),
                ("Inner", text(620, "HIDDEN INNER FORM").into_bytes()),
            ],
        ))
        .unwrap();
        assert!(!document.markdown.contains("HIDDEN"));
        assert!(document.markdown.contains("Visible nested form"));
        assert!(document.markdown.contains("Visible page content before"));
        assert!(document.markdown.contains("Visible page content after"));
    }
}

#[test]
fn invisible_actual_text_does_not_restore_nonpainting_glyphs() {
    for mode in [3, 7] {
        let content = format!(
            "{}q {mode} Tr BT /F1 12 Tf 40 650 Td /Span << /ActualText (HIDDEN ACTUAL TEXT) >> BDC (hidden glyphs) Tj EMC ET Q {}",
            text(700, "Visible content before an inaccessible hidden span."),
            text(600, "Visible content after the marked span.")
        );
        let document = extract(&pdf(&[content.into_bytes()], &[])).unwrap();
        assert!(!document.markdown.contains("HIDDEN"));
        assert!(!document.markdown.contains("hidden glyphs"));
        assert!(document.markdown.contains("Visible content before"));
        assert!(document.markdown.contains("Visible content after"));
    }
}

#[test]
fn an_invisible_only_page_cannot_pass_the_plain_text_fallback() {
    for mode in [3, 7] {
        let content = format!(
            "{mode} Tr {}",
            text(
                700,
                "Hidden-only native text must not become a successful visible document."
            )
        );
        let error = extract(&pdf(&[content.into_bytes()], &[]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("no reliable native text or extractable images"));
    }
}

/// Four pages of body text under a running header (`size` points, at the
/// top of a 792pt page) that names the page number after it.
fn headed_pages(size: usize) -> Vec<Vec<u8>> {
    (1..=4)
        .map(|page| {
            let mut content = format!(
                "BT /F1 {size} Tf 1 0 0 1 40 770 Tm (Proc. of the Conference on Probes 2026  {page}) Tj ET\n"
            );
            for row in 0..8 {
                content.push_str(&text(
                    700 - row * 40,
                    &format!("Body paragraph {row} of page {page} keeps its own words."),
                ));
            }
            content.into_bytes()
        })
        .collect()
}

#[test]
fn a_running_header_on_every_page_is_removed_like_a_folio() {
    let document = extract(&pdf(&headed_pages(8), &[])).unwrap();
    assert!(
        !document.markdown.contains("Proc. of the Conference"),
        "{}",
        document.markdown
    );
    for page in 1..=4 {
        assert!(document.markdown.contains(&format!(
            "Body paragraph 0 of page {page} keeps its own words."
        )));
    }
    // Set at the body size it is text that happens to open each page.
    let document = extract(&pdf(&headed_pages(10), &[])).unwrap();
    assert_eq!(
        document.markdown.matches("Proc. of the Conference").count(),
        4
    );
}

#[test]
fn an_info_title_that_is_a_file_name_or_a_default_name_is_not_the_document_title() {
    for title in [
        "multi.html",
        "report.docx",
        "Microsoft Word - notes.doc",
        "layout-table",
        "code",
        "Untitled",
        "untitled-3",
        "Untitled Document",
        "Untitled (2)",
        "PowerPoint Presentation",
        "Document1",
        "Document 12",
        "Presentation1",
        "Book1",
        "Title",
        "Sans titre",
        "无标题",
        "演示文稿1",
    ] {
        assert!(placeholder_title(title), "{title}");
    }
    for title in [
        "Placeholder topic - Wikipedia",
        "Settings",
        "A Study of Reading Order",
        "Defuddle on Cloudflare Workers · Issue #56",
        "v1.2",
        "Untitled Love Song",
        "Document Management Policy",
        "Book",
        "The Presentation of Self",
        "Real Title Here",
    ] {
        assert!(!placeholder_title(title), "{title}");
    }
}

/// The Info title of a document nobody named gives way to its first heading.
#[test]
fn a_default_info_title_gives_way_to_the_first_heading() {
    for (info, expected) in [
        ("Untitled", None),
        ("Quarterly Results", Some("Quarterly Results")),
    ] {
        let mut doc = lopdf::Document::with_version("1.7");
        let tree = doc.new_object_id();
        let font = doc.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica"
        });
        let content = doc.add_object(Stream::new(
            Dictionary::new(),
            b"BT /F1 12 Tf 72 700 Td (A body paragraph long enough to be read as page text.) Tj ET"
                .to_vec(),
        ));
        let page = doc.add_object(dictionary! {
            "Type" => "Page", "Parent" => tree, "Contents" => content,
            "Resources" => dictionary! { "Font" => dictionary! { "F1" => font } }
        });
        doc.objects.insert(
            tree,
            dictionary! {
                "Type" => "Pages", "Count" => 1, "Kids" => vec![page.into()],
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()]
            }
            .into(),
        );
        let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => tree });
        doc.trailer.set("Root", catalog);
        let meta = doc.add_object(dictionary! { "Title" => Object::string_literal(info) });
        doc.trailer.set("Info", meta);
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        let document = extract(&bytes).unwrap();
        assert_eq!(
            document.metadata.get("title").and_then(|t| t.as_str()),
            expected,
            "{info}"
        );
    }
}

/// A form's values read beside the questions that label them, under no
/// internal field name, and its password unread.
#[test]
fn form_values_read_beside_their_labels() {
    let mut doc = lopdf::Document::with_version("1.7");
    let tree = doc.new_object_id();
    let page = doc.new_object_id();
    let font = doc.add_object(dictionary! {
        "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica",
        "Encoding" => "WinAnsiEncoding",
    });
    let content = doc.add_object(Stream::new(
        Dictionary::new(),
        b"BT /F1 16 Tf 60 762 Td (Employee Information) Tj ET \
          BT /F1 11 Tf 60 712 Td (1. First name) Tj ET \
          BT /F1 11 Tf 60 676 Td (2. Last name) Tj ET \
          BT /F1 11 Tf 60 640 Td (3. PIN) Tj ET"
            .to_vec(),
    ));
    let widget = |doc: &mut lopdf::Document, y: i64, name: &str, value: &str, flags: i64| {
        doc.add_object(dictionary! {
            "Type" => "Annot", "Subtype" => "Widget", "FT" => "Tx", "Ff" => flags, "P" => page,
            "T" => Object::string_literal(name), "V" => Object::string_literal(value),
            "Rect" => vec![220.into(), (y - 6).into(), 500.into(), (y + 14).into()],
        })
    };
    let fields = [
        widget(
            &mut doc,
            712,
            "topmostSubform[0].Page1[0].f1_01[0]",
            "Maria",
            0,
        ),
        widget(
            &mut doc,
            676,
            "topmostSubform[0].Page1[0].f1_02[0]",
            "Garcia",
            0,
        ),
        widget(
            &mut doc,
            640,
            "topmostSubform[0].Page1[0].f1_03[0]",
            "4321",
            1 << 13,
        ),
    ];
    let annots: Vec<Object> = fields.iter().map(|&id| Object::Reference(id)).collect();
    doc.objects.insert(
        page,
        dictionary! {
            "Type" => "Page", "Parent" => tree, "Contents" => content, "Annots" => annots.clone(),
            "Resources" => dictionary! { "Font" => dictionary! { "F1" => font } }
        }
        .into(),
    );
    doc.objects.insert(
        tree,
        dictionary! {
            "Type" => "Pages", "Count" => 1, "Kids" => vec![page.into()],
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()]
        }
        .into(),
    );
    let catalog = doc.add_object(dictionary! {
        "Type" => "Catalog", "Pages" => tree, "AcroForm" => dictionary! { "Fields" => annots },
    });
    doc.trailer.set("Root", catalog);
    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).unwrap();
    let markdown = extract(&bytes).unwrap().markdown;
    assert!(markdown.contains("1. First name: Maria"), "{markdown}");
    assert!(markdown.contains("2. Last name: Garcia"), "{markdown}");
    assert!(!markdown.contains("topmostSubform"), "{markdown}");
    assert!(!markdown.contains("4321"), "{markdown}");
}
