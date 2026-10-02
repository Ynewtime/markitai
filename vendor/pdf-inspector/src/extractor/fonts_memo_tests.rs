//! markitai: the font readings a `FontStyleCache` keeps per font object
//! (see `FontReadings`): which fonts are kept and under which id, that a
//! kept reading is the reading of the font listed, and the bound.

use super::*;
use crate::extractor::content_stream::extract_page_text_items;
use crate::extractor::FormWalkBudget;
use crate::types::TextItem;
use lopdf::{dictionary, Stream};

/// A simple font over Helvetica whose `/Differences` read code `A` as
/// `glyph`, `A` as wide as `width`.
fn differences_font(glyph: &[u8], width: i64) -> lopdf::Dictionary {
    dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
        "FirstChar" => 65,
        "LastChar" => 65,
        "Widths" => vec![Object::Integer(width)],
        "Encoding" => dictionary! {
            "Type" => "Encoding",
            "Differences" => vec![65.into(), Object::Name(glyph.to_vec())],
        },
    }
}

/// A document whose pages are `pages` under one `/Pages` node with the
/// indirect resources `inherited`; each page is given its parent.
fn doc_of_pages(
    mut doc: Document,
    pages: Vec<lopdf::Dictionary>,
    inherited: Option<lopdf::Dictionary>,
) -> (Document, Vec<ObjectId>) {
    let pages_id = doc.new_object_id();
    let ids: Vec<ObjectId> = pages
        .into_iter()
        .map(|mut page| {
            page.set("Type", "Page");
            page.set("Parent", Object::Reference(pages_id));
            page.set("MediaBox", vec![0.into(), 0.into(), 612.into(), 792.into()]);
            doc.add_object(page)
        })
        .collect();
    let mut node = dictionary! {
        "Type" => "Pages",
        "Count" => ids.len() as i64,
        "Kids" => ids.iter().map(|id| Object::Reference(*id)).collect::<Vec<_>>(),
    };
    if let Some(inherited) = inherited {
        node.set("Resources", Object::Reference(doc.add_object(inherited)));
    }
    doc.objects.insert(pages_id, Object::Dictionary(node));
    let catalog = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id),
    });
    doc.trailer.set("Root", Object::Reference(catalog));
    (doc, ids)
}

/// Three pages under a `/Pages` node whose indirect resources list `F1`,
/// font A (code `A` reads `B`), and `F3`, font C (WinAnsi):
/// - page 1 lists A as `F1` in resources of its own, and draws a form
///   that lists A again as `F2`;
/// - page 2 has no resources of its own and inherits both;
/// - page 3 lists in its own resources an `F1` written in place, font D
///   (code `A` reads `C`), which shadows the inherited one.
///
/// Every page shows `(A)` in its `F1`, page 2 also in `F3`. Returns the
/// document, the pages, and the ids of fonts A and C.
fn shared_fonts_doc() -> (Document, [ObjectId; 3], ObjectId, ObjectId) {
    let mut doc = Document::with_version("1.4");
    let font_a = doc.add_object(differences_font(b"B", 700));
    let font_c = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
        "Encoding" => "WinAnsiEncoding",
    });
    let form = doc.add_object(Stream::new(
        dictionary! {
            "Type" => "XObject",
            "Subtype" => "Form",
            "BBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            "Resources" => dictionary! {
                "Font" => dictionary! { "F2" => Object::Reference(font_a) },
            },
        },
        b"BT /F2 12 Tf 72 600 Td (A) Tj ET".to_vec(),
    ));
    let mut content = |bytes: &[u8]| {
        Object::Reference(doc.add_object(Stream::new(dictionary! {}, bytes.to_vec())))
    };
    let pages = vec![
        dictionary! {
            "Contents" => content(b"BT /F1 12 Tf 72 700 Td (A) Tj ET /Fm Do"),
            "Resources" => dictionary! {
                "Font" => dictionary! { "F1" => Object::Reference(font_a) },
                "XObject" => dictionary! { "Fm" => Object::Reference(form) },
            },
        },
        dictionary! {
            "Contents" => content(b"BT /F1 12 Tf 72 700 Td (A) Tj 0 -100 Td /F3 12 Tf (A) Tj ET"),
        },
        dictionary! {
            "Contents" => content(b"BT /F1 12 Tf 72 700 Td (A) Tj ET"),
            "Resources" => dictionary! {
                "Font" => dictionary! { "F1" => differences_font(b"C", 500) },
            },
        },
    ];
    let inherited = dictionary! {
        "Font" => dictionary! {
            "F1" => Object::Reference(font_a),
            "F3" => Object::Reference(font_c),
        },
    };
    let (doc, ids) = doc_of_pages(doc, pages, Some(inherited));
    (doc, [ids[0], ids[1], ids[2]], font_a, font_c)
}

/// A TrueType font whose `/Differences` name codes 1 and 2 by glyph index
/// alone, with a ToUnicode CMap of `bfchar` lines when one is given.
fn glyph_index_font(doc: &mut Document, bfchar: Option<&str>) -> ObjectId {
    let mut font = dictionary! {
        "Type" => "Font",
        "Subtype" => "TrueType",
        "BaseFont" => "ABCDEF+OpenSymbol",
        "Encoding" => dictionary! {
            "Type" => "Encoding",
            "Differences" => vec![
                1.into(),
                Object::Name(b"gid1283".to_vec()),
                Object::Name(b"gid1464".to_vec()),
            ],
        },
    };
    if let Some(bfchar) = bfchar {
        let lines = bfchar.lines().count();
        let cmap = format!(
            "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
             1 begincodespacerange\n<00> <FF>\nendcodespacerange\n\
             {lines} beginbfchar\n{bfchar}\nendbfchar\nendcmap\n\
             CMapName currentdict /CMap defineresource pop\nend\nend"
        );
        let cmap = doc.add_object(Stream::new(dictionary! {}, cmap.into_bytes()));
        font.set("ToUnicode", Object::Reference(cmap));
    }
    doc.add_object(font)
}

/// The fonts a listing holds: each name with the dictionary's address.
fn listed(fonts: &FontResources<'_>) -> Vec<(Vec<u8>, usize)> {
    fonts
        .iter()
        .map(|(name, dict)| (name.clone(), std::ptr::from_ref(*dict) as usize))
        .collect()
}

fn names(ids: &FontObjectIds<'_>) -> Vec<(String, ObjectId)> {
    let mut names: Vec<(String, ObjectId)> = ids
        .iter()
        .map(|(name, id)| (String::from_utf8_lossy(name).into_owned(), *id))
        .collect();
    names.sort();
    names
}

/// The items of page `page_id` read with `cache`.
fn page_items(
    doc: &Document,
    page_id: ObjectId,
    cmaps: &FontCMaps,
    cache: &mut FontStyleCache,
) -> Vec<TextItem> {
    let ((items, _, _), _, _, _) = extract_page_text_items(
        doc,
        page_id,
        1,
        cmaps,
        false,
        cache,
        &mut FormWalkBudget::new(),
    )
    .expect("page");
    items
}

#[test]
fn page_fonts_give_an_id_only_to_the_dictionary_listed() {
    let (doc, [page_1, page_2, page_3], font_a, font_c) = shared_fonts_doc();
    for page in [page_1, page_2, page_3] {
        let (fonts, _) = page_fonts(&doc, page);
        assert_eq!(listed(&fonts), listed(&doc.get_page_fonts(page).unwrap()));
    }
    // A name the page's own resources list is reached there first, then
    // in the resources it inherits.
    let (_, ids) = page_fonts(&doc, page_1);
    assert_eq!(names(&ids), [("F1".into(), font_a), ("F3".into(), font_c)]);
    let (_, ids) = page_fonts(&doc, page_2);
    assert_eq!(names(&ids), [("F1".into(), font_a), ("F3".into(), font_c)]);
    // The `F1` written in place shadows the inherited reference, which is
    // not its id.
    let (fonts, ids) = page_fonts(&doc, page_3);
    let own = doc
        .get_dictionary(page_3)
        .unwrap()
        .get(b"Resources")
        .unwrap();
    let own_f1 = own.as_dict().unwrap().get(b"Font").unwrap();
    let own_f1 = own_f1
        .as_dict()
        .unwrap()
        .get(b"F1")
        .unwrap()
        .as_dict()
        .unwrap();
    assert!(std::ptr::eq(fonts[b"F1".as_slice()], own_f1));
    assert_eq!(names(&ids), [("F3".into(), font_c)]);
}

#[test]
fn page_fonts_take_the_id_that_reaches_the_font_listed() {
    // The page's own `F1` refers to an object that is missing: lopdf lists
    // the inherited `F1`, and that is the id given.
    let (mut doc, [page_1, ..], font_a, _) = shared_fonts_doc();
    let missing = doc.new_object_id();
    doc.get_dictionary_mut(page_1)
        .unwrap()
        .get_mut(b"Resources")
        .and_then(Object::as_dict_mut)
        .unwrap()
        .set("Font", dictionary! { "F1" => Object::Reference(missing) });
    let (fonts, ids) = page_fonts(&doc, page_1);
    assert_eq!(listed(&fonts), listed(&doc.get_page_fonts(page_1).unwrap()));
    assert_eq!(ids.get(b"F1".as_slice()), Some(&font_a));
    // Resources that do not resolve list no font, and give no id.
    let (mut doc, [page_1, ..], _, _) = shared_fonts_doc();
    let parent = doc
        .get_dictionary(page_1)
        .and_then(|page| page.get(b"Parent"))
        .and_then(Object::as_reference)
        .unwrap();
    doc.get_dictionary_mut(parent)
        .unwrap()
        .set("Parent", Object::Reference(parent));
    assert!(doc.get_page_fonts(page_1).is_err());
    let (fonts, ids) = page_fonts(&doc, page_1);
    assert!(fonts.is_empty() && ids.is_empty());
}

#[test]
fn form_fonts_give_an_id_to_the_fonts_that_are_objects() {
    let (mut doc, _, font_a, _) = shared_fonts_doc();
    let form = doc.add_object(Stream::new(
        dictionary! {
            "Subtype" => "Form",
            "Resources" => dictionary! {
                "Font" => dictionary! {
                    "F2" => Object::Reference(font_a),
                    "F9" => differences_font(b"C", 500),
                },
            },
        },
        Vec::new(),
    ));
    let dict = &doc.get_object(form).unwrap().as_stream().unwrap().dict;
    let (fonts, ids) = crate::extractor::xobjects::get_form_fonts(&doc, dict);
    assert_eq!(fonts.len(), 2);
    assert_eq!(names(&ids), [("F2".into(), font_a)]);
}

#[test]
fn a_font_listed_by_several_pages_and_forms_is_read_once_as_each_reads_it() {
    let (doc, pages, font_a, font_c) = shared_fonts_doc();
    let cmaps = FontCMaps::from_doc(&doc);
    let mut shared = FontStyleCache::new();
    let texts: Vec<Vec<String>> = pages
        .iter()
        .map(|&page| {
            let items = page_items(&doc, page, &cmaps, &mut shared);
            let fresh = page_items(&doc, page, &cmaps, &mut FontStyleCache::new());
            assert_eq!(format!("{items:?}"), format!("{fresh:?}"));
            items.into_iter().map(|item| item.text).collect()
        })
        .collect();
    // Font A reads `A` as `B` on the pages and in the form; font C reads
    // it as `A`, and the font written in place on page 3 as `C`.
    assert_eq!(texts, [vec!["B", "B"], vec!["B", "A"], vec!["C"]]);
    // Fonts A and C are kept, the font written in place is not.
    let kept = |ids: Vec<ObjectId>| {
        let mut ids = ids;
        ids.sort();
        ids
    };
    assert_eq!(
        kept(shared.readings.encodings.keys().copied().collect()),
        [font_a, font_c]
    );
    assert_eq!(
        kept(shared.readings.widths.keys().copied().collect()),
        [font_a, font_c]
    );
}

#[test]
fn every_listing_of_a_kept_font_takes_the_reading_kept() {
    let (doc, [page_1, page_2, _], _, _) = shared_fonts_doc();
    let cmaps = FontCMaps::from_doc(&doc);
    let mut cache = FontStyleCache::new();
    let mut read = |page| {
        let (fonts, ids) = page_fonts(&doc, page);
        let (encodings, _) = build_font_encodings(&doc, &fonts, &ids, &cmaps, &mut cache);
        let widths = build_font_widths(&doc, &fonts, &ids, &mut cache);
        (encodings, widths, cache.readings.weight)
    };
    let (encodings_1, widths_1, weight_1) = read(page_1);
    let (encodings_2, widths_2, weight_2) = read(page_2);
    assert!(Arc::ptr_eq(&encodings_1["F1"], &encodings_2["F1"]));
    assert_eq!(widths_1["F1"].widths, widths_2["F1"].widths);
    assert_eq!(widths_2["F1"].widths.get(&65), Some(&700));
    assert_eq!(weight_1, weight_2);
    // The reading kept is the font's own.
    let (fonts, _) = page_fonts(&doc, page_1);
    let (fresh, _) = build_font_encodings(
        &doc,
        &fonts,
        &FontObjectIds::new(),
        &cmaps,
        &mut FontStyleCache::new(),
    );
    assert_eq!(fresh["F1"].differences, encodings_1["F1"].differences);
    assert_eq!(fresh["F1"].named_codes, encodings_1["F1"].named_codes);
}

#[test]
fn a_page_is_flagged_for_any_kept_font_whose_glyph_indexes_stay_unread() {
    // `F1` names its codes by glyph index and has no CMap to read them;
    // `F2`, listed after it, reads. The page is read twice, the second
    // time from the readings kept.
    let mut doc = Document::with_version("1.4");
    let unread = glyph_index_font(&mut doc, None);
    let plain = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });
    let page = dictionary! {
        "Resources" => dictionary! {
            "Font" => dictionary! {
                "F1" => Object::Reference(unread),
                "F2" => Object::Reference(plain),
            },
        },
    };
    let (doc, pages) = doc_of_pages(doc, vec![page], None);
    let cmaps = FontCMaps::from_doc(&doc);
    let mut cache = FontStyleCache::new();
    for _ in 0..2 {
        let (fonts, ids) = page_fonts(&doc, pages[0]);
        assert_eq!(ids.len(), 2);
        let (_, has_gid_fonts) = build_font_encodings(&doc, &fonts, &ids, &cmaps, &mut cache);
        assert!(has_gid_fonts);
    }
}

#[test]
fn readings_asked_for_with_other_cmaps_start_over() {
    // The glyph-index names read through the font's ToUnicode CMap when
    // the CMaps hold it, and stay unread without it.
    let mut doc = Document::with_version("1.4");
    let font = glyph_index_font(&mut doc, Some("<01> <2022>\n<02> <25E6>"));
    let page = dictionary! {
        "Resources" => dictionary! {
            "Font" => dictionary! { "F1" => Object::Reference(font) },
        },
    };
    let (doc, pages) = doc_of_pages(doc, vec![page], None);
    let with_cmap = FontCMaps::from_doc(&doc);
    let without = FontCMaps::default();
    let mut cache = FontStyleCache::new();
    let (fonts, ids) = page_fonts(&doc, pages[0]);
    for (cmaps, flagged) in [(&with_cmap, false), (&without, true), (&with_cmap, false)] {
        let (_, has_gid_fonts) = build_font_encodings(&doc, &fonts, &ids, cmaps, &mut cache);
        assert_eq!(has_gid_fonts, flagged);
    }
}

#[test]
fn font_readings_stop_at_their_bound() {
    let mut readings = FontReadings {
        weight: FONT_READINGS_MAX_WEIGHT - 1,
        ..FontReadings::default()
    };
    let widths = FontWidthInfo {
        widths: HashMap::from([(65, 700)]),
        default_width: 0,
        space_width: 250,
        is_cid: false,
        units_scale: 0.001,
        wmode: 0,
        cid_codes: None,
    };
    // A font and its one width are two entries: past the bound.
    readings.keep_widths((1, 0), &Some(widths));
    assert!(readings.widths.is_empty());
    assert_eq!(readings.weight, FONT_READINGS_MAX_WEIGHT - 1);
    // A font without a table is one: within it, and the bound is reached.
    readings.keep_widths((2, 0), &None);
    readings.keep_encoding((3, 0), &None, false);
    assert_eq!(readings.widths.len(), 1);
    assert!(readings.encodings.is_empty());
    assert_eq!(readings.weight, FONT_READINGS_MAX_WEIGHT);
}

#[test]
fn lopdf_encodings_resolve_as_resolving_every_font_up_front_did() {
    let doc = Document::with_version("1.4");
    let font = |encoding: &str| {
        dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
            "Encoding" => Object::Name(encoding.as_bytes().to_vec()),
        }
    };
    let win_ansi = font("WinAnsiEncoding");
    let mac_roman = font("MacRomanEncoding");
    let not_a_font = dictionary! { "Type" => "XObject" };
    let read = |fonts: &[(&[u8], &lopdf::Dictionary)], name: &str| {
        let fonts: FontResources<'_> = fonts
            .iter()
            .map(|(name, font)| (name.to_vec(), *font))
            .collect();
        let encodings = LopdfEncodings::new(&doc, &fonts);
        encodings
            .get(name)
            .map(|encoding| Document::decode_text(encoding, &[0x80]).unwrap())
    };
    // Two names that read as the same text: the last whose encoding
    // resolves gives it.
    let shadowed = "F\u{FFFD}";
    assert_eq!(
        read(&[(b"F\xfe", &win_ansi), (b"F\xff", &not_a_font)], shadowed).as_deref(),
        Some("\u{20AC}")
    );
    assert_eq!(
        read(&[(b"F\xfe", &win_ansi), (b"F\xff", &mac_roman)], shadowed).as_deref(),
        Some("\u{C4}")
    );
    // A font lopdf does not resolve, and a name not listed, have none.
    assert_eq!(read(&[(b"F1", &not_a_font)], "F1"), None);
    assert_eq!(read(&[(b"F1", &win_ansi)], "F2"), None);
}
