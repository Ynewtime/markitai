//! markitai: the form contents a `FontStyleCache` keeps for a document's
//! walks (see `FormContents`).

use super::*;
use crate::extractor::content_stream::extract_page_text_items;
use crate::types::TextItem;
use lopdf::{Stream, dictionary};

/// Pages each showing `page_content`, with the forms `forms` bound as
/// `X1`, `X2`, … and the font `F1` (Helvetica) in every resource
/// dictionary. Returns the document, the pages and the forms.
fn doc_with_forms(
    pages: usize,
    page_content: &[u8],
    forms: &[&[u8]],
) -> (Document, Vec<ObjectId>, Vec<ObjectId>) {
    let mut doc = Document::with_version("1.4");
    let font = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });
    let form_ids: Vec<ObjectId> = forms
        .iter()
        .map(|content| {
            doc.add_object(Stream::new(
                dictionary! {
                    "Type" => "XObject",
                    "Subtype" => "Form",
                    "BBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
                    "Resources" => dictionary! {
                        "Font" => dictionary! { "F1" => Object::Reference(font) },
                    },
                },
                content.to_vec(),
            ))
        })
        .collect();
    let mut xobjects = lopdf::Dictionary::new();
    for (index, id) in form_ids.iter().enumerate() {
        xobjects.set(format!("X{}", index + 1), Object::Reference(*id));
    }
    let pages_id = doc.new_object_id();
    let page_ids: Vec<ObjectId> = (0..pages)
        .map(|_| {
            let content = doc.add_object(Stream::new(dictionary! {}, page_content.to_vec()));
            doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => Object::Reference(pages_id),
                "Contents" => Object::Reference(content),
                "Resources" => dictionary! {
                    "Font" => dictionary! { "F1" => Object::Reference(font) },
                    "XObject" => xobjects.clone(),
                },
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            })
        })
        .collect();
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Count" => pages as i64,
            "Kids" => page_ids.iter().map(|id| Object::Reference(*id)).collect::<Vec<_>>(),
        }),
    );
    let catalog = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id),
    });
    doc.trailer.set("Root", Object::Reference(catalog));
    (doc, page_ids, form_ids)
}

fn page_items(doc: &Document, page_id: ObjectId, cache: &mut FontStyleCache) -> Vec<TextItem> {
    let ((items, _, _), _, _, _) = extract_page_text_items(
        doc,
        page_id,
        1,
        &FontCMaps::from_doc(doc),
        false,
        cache,
        &mut FormWalkBudget::new(),
    )
    .expect("page");
    items
}

fn stream_of(doc: &Document, id: ObjectId) -> &lopdf::Stream {
    doc.get_object(id).unwrap().as_stream().unwrap()
}

#[test]
fn a_form_drawn_again_is_decoded_once_and_read_as_before() {
    let form: &[u8] = b"BT /F1 12 Tf 72 700 Td (kept) Tj ET";
    let (doc, pages, forms) = doc_with_forms(2, b"/X1 Do 1 0 0 1 0 -100 cm /X1 Do", &[form]);
    let mut shared = FontStyleCache::new();
    for &page in &pages {
        let items = page_items(&doc, page, &mut shared);
        let fresh = page_items(&doc, page, &mut FontStyleCache::new());
        assert_eq!(format!("{items:?}"), format!("{fresh:?}"));
        let texts: Vec<&str> = items.iter().map(|item| item.text.as_str()).collect();
        assert_eq!(texts, ["kept", "kept"]);
    }
    assert_eq!(shared.forms.by_form.len(), 1);
    assert_eq!(shared.forms.bytes, form.len());
    let stream = stream_of(&doc, forms[0]);
    let first = shared.forms.read(forms[0], stream).expect("content");
    let again = shared.forms.read(forms[0], stream).expect("content");
    assert!(Arc::ptr_eq(&first, &again));
    assert_eq!(first.operations.len(), 5);
}

#[test]
fn a_form_the_walk_skips_is_kept_as_skipped() {
    let mut oversized = b"BT /F1 12 Tf 72 700 Td (lost) Tj ET".to_vec();
    oversized.resize(
        super::super::content_decode::MAX_PAGE_CONTENT_BYTES + 1,
        b' ',
    );
    let (doc, pages, forms) = doc_with_forms(
        1,
        b"/X1 Do /X1 Do /X2 Do",
        &[&oversized, b"BT /F1 12 Tf 72 600 Td (kept) Tj ET"],
    );
    let mut cache = FontStyleCache::new();
    let items = page_items(&doc, pages[0], &mut cache);
    let texts: Vec<&str> = items.iter().map(|item| item.text.as_str()).collect();
    assert_eq!(texts, ["kept"]);
    assert!(matches!(cache.forms.by_form.get(&forms[0]), Some(None)));
    assert!(matches!(cache.forms.by_form.get(&forms[1]), Some(Some(_))));
}

#[test]
fn form_contents_stop_at_their_bound() {
    let (doc, _, forms) = doc_with_forms(1, b"", &[b"q Q", b"q Q "]);
    let mut contents = FormContents {
        bytes: FORM_CONTENTS_MAX_BYTES - 3,
        ..FormContents::default()
    };
    // Four bytes do not fit, and are read all the same.
    let wide = contents.read(forms[1], stream_of(&doc, forms[1]));
    assert_eq!(wide.map(|content| content.operations.len()), Some(2));
    assert!(contents.by_form.is_empty());
    // Three do, and reach the bound.
    assert!(contents.read(forms[0], stream_of(&doc, forms[0])).is_some());
    assert_eq!(contents.by_form.len(), 1);
    assert_eq!(contents.bytes, FORM_CONTENTS_MAX_BYTES);
}
