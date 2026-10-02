//! Type0 fonts without a ToUnicode CMap whose encoding is a predefined CMap,
//! and pages where some text runs cannot be decoded (markitai).

use crate::{extract_pages_markdown_mem, extract_text_with_positions_mem, PageOmittedText};
use lopdf::{dictionary, Dictionary, Document, Object, Stream};

/// A one-page PDF showing each run — font resource, baseline, string bytes —
/// in the fonts `fonts` adds to the document.
fn page_pdf(
    fonts: impl FnOnce(&mut Document) -> Vec<(&'static str, Object)>,
    runs: &[(&str, i32, Vec<u8>)],
) -> Vec<u8> {
    let mut doc = Document::with_version("1.4");
    let font_entries = fonts(&mut doc);
    let mut font_dict = Dictionary::new();
    for (name, font) in font_entries {
        font_dict.set(name, font);
    }
    let mut content = String::new();
    for (font, y, bytes) in runs {
        let hex: String = bytes.iter().map(|byte| format!("{byte:02X}")).collect();
        content.push_str(&format!("BT /{font} 14 Tf 72 {y} Td <{hex}> Tj ET\n"));
    }
    let contents = doc.add_object(Stream::new(dictionary! {}, content.into_bytes()));
    let pages_id = doc.new_object_id();
    let page = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference(pages_id),
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        "Contents" => Object::Reference(contents),
        "Resources" => dictionary! { "Font" => font_dict },
    });
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![Object::Reference(page)],
            "Count" => 1,
        }),
    );
    let catalog = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference(pages_id),
    });
    doc.trailer.set("Root", Object::Reference(catalog));
    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).unwrap();
    bytes
}

fn helvetica(doc: &mut Document) -> Object {
    Object::Reference(doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
        "Encoding" => "WinAnsiEncoding",
    }))
}

/// A Type0 font with no embedded program: `encoding` names its CMap, its
/// descendant is a `subtype` CIDFont of the Adobe `ordering` collection.
fn cid_font(
    doc: &mut Document,
    encoding: &str,
    subtype: &str,
    ordering: &str,
    widths: Option<Vec<Object>>,
    to_unicode: Option<&str>,
) -> Object {
    let descriptor = doc.add_object(dictionary! {
        "Type" => "FontDescriptor",
        "FontName" => "MS-Mincho",
        "Flags" => 4,
        "FontBBox" => vec![0.into(), Object::Integer(-200), 1000.into(), 900.into()],
        "ItalicAngle" => 0,
        "Ascent" => 900,
        "Descent" => -200,
        "CapHeight" => 700,
        "StemV" => 80,
    });
    let mut descendant = dictionary! {
        "Type" => "Font",
        "Subtype" => subtype,
        "BaseFont" => "MS-Mincho",
        "CIDSystemInfo" => dictionary! {
            "Registry" => Object::string_literal("Adobe"),
            "Ordering" => Object::string_literal(ordering),
            "Supplement" => 2,
        },
        "FontDescriptor" => Object::Reference(descriptor),
        "DW" => 1000,
    };
    if let Some(widths) = widths {
        descendant.set("W", widths);
    }
    let descendant = doc.add_object(descendant);
    let mut font = dictionary! {
        "Type" => "Font",
        "Subtype" => "Type0",
        "BaseFont" => "MS-Mincho",
        "Encoding" => encoding,
        "DescendantFonts" => vec![Object::Reference(descendant)],
    };
    if let Some(cmap) = to_unicode {
        let stream = doc.add_object(Stream::new(dictionary! {}, cmap.as_bytes().to_vec()));
        font.set("ToUnicode", Object::Reference(stream));
    }
    Object::Reference(doc.add_object(font))
}

/// A simple font that names its glyphs `gidNNNN` and has no ToUnicode
/// CMap: what codes 1 and 2 show cannot be identified.
fn glyph_index_font(doc: &mut Document) -> Object {
    let encoding = doc.add_object(dictionary! {
        "Type" => "Encoding",
        "Differences" => vec![
            1.into(),
            Object::Name(b"gid1283".to_vec()),
            Object::Name(b"gid1464".to_vec()),
        ],
    });
    Object::Reference(doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "TrueType",
        "BaseFont" => "ABCDEF+Logo",
        "Encoding" => Object::Reference(encoding),
    }))
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

fn utf16(text: &str) -> Vec<u8> {
    text.encode_utf16().flat_map(u16::to_be_bytes).collect()
}

const BEFORE: &str = "English line before";
const AFTER: &str = "English line after";
/// "ABC 123 吾輩は猫である。" in Shift-JIS: one-byte ASCII among
/// two-byte ideographs.
const SHIFT_JIS: &str = "41424320313233208ce1947982cd944c82c582a082e98142";
const CHINESE: &str = "这是一个简体中文测试。文档转换质量检查。";

#[test]
fn a_page_in_a_predefined_cmap_keeps_every_line() {
    let pdf = page_pdf(
        |doc| {
            vec![
                (
                    "F1",
                    cid_font(doc, "90ms-RKSJ-H", "CIDFontType2", "Japan1", None, None),
                ),
                ("F2", helvetica(doc)),
            ]
        },
        &[
            ("F2", 700, BEFORE.as_bytes().to_vec()),
            ("F1", 670, hex(SHIFT_JIS)),
            ("F2", 640, AFTER.as_bytes().to_vec()),
        ],
    );
    let result = extract_pages_markdown_mem(&pdf, None).unwrap();
    let page = &result.pages[0];
    assert!(!page.needs_ocr, "{:?}", page.ocr_reason);
    for line in [BEFORE, "ABC 123 吾輩は猫である。", AFTER] {
        assert!(page.markdown.contains(line), "{line}: {}", page.markdown);
    }
    assert!(result.omitted_text_by_page.is_empty());
}

#[test]
fn a_utf16_cmap_reads_through_its_collection() {
    let pdf = page_pdf(
        |doc| {
            vec![
                (
                    "F1",
                    cid_font(doc, "UniGB-UTF16-H", "CIDFontType0", "GB1", None, None),
                ),
                ("F2", helvetica(doc)),
            ]
        },
        &[
            ("F2", 700, BEFORE.as_bytes().to_vec()),
            ("F1", 670, utf16(CHINESE)),
            ("F2", 640, AFTER.as_bytes().to_vec()),
        ],
    );
    let page = &extract_pages_markdown_mem(&pdf, None).unwrap().pages[0];
    assert!(!page.needs_ocr, "{:?}", page.ocr_reason);
    assert!(page.markdown.contains(CHINESE), "{}", page.markdown);
}

#[test]
fn widths_are_those_of_the_cids_the_encoding_selects() {
    // "AB吾輩": CIDs 264, 265, 1943 and 3344 in Adobe-Japan1, given widths
    // the default would not.
    let widths = || {
        Some(vec![
            264.into(),
            vec![500.into(), 600.into()].into(),
            1943.into(),
            vec![1000.into()].into(),
            3344.into(),
            vec![900.into()].into(),
        ])
    };
    let shift_jis = page_pdf(
        |doc| {
            vec![(
                "F1",
                cid_font(doc, "90ms-RKSJ-H", "CIDFontType2", "Japan1", widths(), None),
            )]
        },
        &[("F1", 700, hex("41428ce19479"))],
    );
    let identity_cmap = "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n\
1 begincodespacerange <0000> <FFFF> endcodespacerange\n\
4 beginbfchar <0108> <0041> <0109> <0042> <0797> <543E> <0D10> <8F29> endbfchar\n\
endcmap end end";
    let identity = page_pdf(
        |doc| {
            vec![(
                "F1",
                cid_font(
                    doc,
                    "Identity-H",
                    "CIDFontType2",
                    "Japan1",
                    widths(),
                    Some(identity_cmap),
                ),
            )]
        },
        &[("F1", 700, hex("0108010907970d10"))],
    );
    let item = |pdf: &[u8]| {
        let items = extract_text_with_positions_mem(pdf).unwrap();
        assert_eq!(items.len(), 1, "{items:?}");
        assert_eq!(items[0].text, "AB吾輩");
        items[0].width
    };
    let expected = (500.0 + 600.0 + 1000.0 + 900.0) / 1000.0 * 14.0;
    assert!(
        (item(&shift_jis) - expected).abs() < 0.01,
        "{}",
        item(&shift_jis)
    );
    assert!((item(&identity) - item(&shift_jis)).abs() < 0.01);
}

/// A page with `cjk` runs of a font no reading decodes — Identity-H over
/// Adobe-GB1 CIDs that are in fact UTF-16 code units, with no program and
/// no ToUnicode CMap — and `english` runs that decode.
fn undecodable_page(cjk: usize, english: &[&str]) -> Vec<u8> {
    let mut runs = Vec::new();
    let mut y = 740;
    for line in english {
        runs.push(("F2", y, line.as_bytes().to_vec()));
        y -= 20;
    }
    for _ in 0..cjk {
        runs.push(("F1", y, utf16(CHINESE)));
        y -= 20;
    }
    page_pdf(
        |doc| {
            vec![
                (
                    "F1",
                    cid_font(doc, "Identity-H", "CIDFontType2", "GB1", None, None),
                ),
                ("F2", helvetica(doc)),
            ]
        },
        &runs,
    )
}

#[test]
fn an_undecodable_run_costs_its_page_that_run_alone() {
    let result = extract_pages_markdown_mem(&undecodable_page(1, &[BEFORE, AFTER]), None).unwrap();
    let page = &result.pages[0];
    assert!(!page.needs_ocr, "{:?}", page.ocr_reason);
    assert!(page.markdown.contains(BEFORE) && page.markdown.contains(AFTER));
    assert!(!page.markdown.contains('\u{FFFD}'), "{}", page.markdown);
    assert_eq!(
        result.omitted_text_by_page,
        [PageOmittedText {
            page: 1,
            runs: 1,
            chars: 20,
            replacement_chars: 0,
            unidentified_glyphs: false,
        }]
    );
}

#[test]
fn a_page_whose_text_is_mostly_undecodable_still_needs_ocr() {
    let english = "A single readable line of English text";
    let result = extract_pages_markdown_mem(&undecodable_page(4, &[english]), None).unwrap();
    let page = &result.pages[0];
    assert!(page.needs_ocr);
    assert_eq!(page.ocr_reason.as_deref(), Some("suspected_garbled_text"));
    assert!(result.omitted_text_by_page.is_empty());
    // Too little readable text left stands no more than lost text does.
    let result = extract_pages_markdown_mem(&undecodable_page(1, &["Short line"]), None).unwrap();
    assert!(result.pages[0].needs_ocr);
}

#[test]
fn a_logo_in_a_font_without_glyph_identities_leaves_the_body() {
    let body = "The body of this page reads as it should, in a font that decodes.";
    let pdf = |text: &str| {
        page_pdf(
            |doc| vec![("F1", glyph_index_font(doc)), ("F2", helvetica(doc))],
            &[
                ("F1", 740, vec![1, 2]),
                ("F2", 700, text.as_bytes().to_vec()),
            ],
        )
    };
    let result = extract_pages_markdown_mem(&pdf(body), None).unwrap();
    let page = &result.pages[0];
    assert!(!page.needs_ocr, "{:?}", page.ocr_reason);
    assert!(page.markdown.contains(body), "{}", page.markdown);
    assert_eq!(
        result.omitted_text_by_page,
        [PageOmittedText {
            page: 1,
            runs: 0,
            chars: 0,
            replacement_chars: 0,
            unidentified_glyphs: true,
        }]
    );
    // A page that has little else is still the font's.
    let result = extract_pages_markdown_mem(&pdf("Page 1"), None).unwrap();
    assert!(result.pages[0].needs_ocr);
}

#[test]
fn a_code_without_a_reading_keeps_its_run_marked() {
    // 0x817F is a Shift-JIS code 90ms-RKSJ-H assigns no CID.
    let pdf = page_pdf(
        |doc| {
            vec![
                (
                    "F1",
                    cid_font(doc, "90ms-RKSJ-H", "CIDFontType2", "Japan1", None, None),
                ),
                ("F2", helvetica(doc)),
            ]
        },
        &[
            ("F2", 700, BEFORE.as_bytes().to_vec()),
            ("F1", 670, hex("8ce1817f947982cd944c82c582a082e98142")),
            ("F2", 640, AFTER.as_bytes().to_vec()),
        ],
    );
    let result = extract_pages_markdown_mem(&pdf, None).unwrap();
    let page = &result.pages[0];
    assert!(!page.needs_ocr, "{:?}", page.ocr_reason);
    assert!(
        page.markdown.contains("吾\u{FFFD}輩は猫である。"),
        "{}",
        page.markdown
    );
    assert_eq!(
        result.omitted_text_by_page,
        [PageOmittedText {
            page: 1,
            runs: 0,
            chars: 0,
            replacement_chars: 1,
            unidentified_glyphs: false,
        }]
    );
}

#[test]
fn a_sparse_tounicode_cmap_gives_way_to_the_encodings_reading() {
    // A ToUnicode CMap of one entry over a Shift-JIS font: the font's
    // strings read through its encoding and collection instead.
    let to_unicode = "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n\
2 begincodespacerange <00> <80> <8140> <9FFC> endcodespacerange\n\
1 beginbfchar <41> <0041> endbfchar\nendcmap end end";
    let pdf = page_pdf(
        |doc| {
            vec![
                (
                    "F1",
                    cid_font(
                        doc,
                        "90ms-RKSJ-H",
                        "CIDFontType2",
                        "Japan1",
                        None,
                        Some(to_unicode),
                    ),
                ),
                ("F2", helvetica(doc)),
            ]
        },
        &[
            ("F2", 700, BEFORE.as_bytes().to_vec()),
            ("F1", 670, hex(SHIFT_JIS)),
            ("F2", 640, AFTER.as_bytes().to_vec()),
        ],
    );
    let page = &extract_pages_markdown_mem(&pdf, None).unwrap().pages[0];
    assert!(!page.needs_ocr, "{:?}", page.ocr_reason);
    assert!(
        page.markdown.contains("ABC 123 吾輩は猫である。"),
        "{}",
        page.markdown
    );
}
