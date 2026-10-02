//! Every assertion runs against each backend the build contains: CoreGraphics
//! on macOS, hayro elsewhere, and both on macOS with `portable-media`.

use super::*;
use image::RgbImage;
use lopdf::{Document, Object, Stream, dictionary};

const MIXED: &[u8] = include_bytes!("fixtures/mixed-native-scanned-blank.pdf");

fn open(bytes: &[u8], backend: Backend) -> PdfRasterSession {
    PdfRasterSession::open_with(bytes, backend)
        .unwrap_or_else(|error| panic!("{}: {error}", backend.name()))
}

fn assert_rgb(image: &RgbImage, x: u32, y: u32, expected: [u8; 3], backend: Backend) {
    let actual = image.get_pixel(x, y).0;
    assert!(
        actual.iter().zip(expected).all(|(a, b)| a.abs_diff(b) <= 4),
        "{}: pixel ({x}, {y}): {actual:?} != {expected:?}",
        backend.name()
    );
}

fn has_ink(image: &RgbImage, x: u32, y: u32, width: u32, height: u32) -> bool {
    (y..y + height)
        .any(|y| (x..x + width).any(|x| image.get_pixel(x, y).0.iter().any(|v| *v < 200)))
}

fn save(document: &mut Document) -> Vec<u8> {
    let mut bytes = Vec::new();
    document.save_to(&mut bytes).unwrap();
    bytes
}

fn one_page(content: &[u8], resources: lopdf::Dictionary, media: [i32; 4]) -> Document {
    let mut document = Document::with_version("1.7");
    let pages = document.new_object_id();
    let stream = document.add_object(Stream::new(dictionary! {}, content.to_vec()));
    let page = document.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages, "Contents" => stream,
        "Resources" => resources,
        "MediaBox" => media.into_iter().map(Object::from).collect::<Vec<_>>()
    });
    document.objects.insert(
        pages,
        dictionary! {
            "Type" => "Pages", "Kids" => vec![Object::Reference(page)], "Count" => 1
        }
        .into(),
    );
    let catalog = document.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages });
    document.trailer.set("Root", catalog);
    document
}

fn encrypted(bytes: &[u8], user_password: &str) -> Vec<u8> {
    let mut document = Document::load_mem(bytes).unwrap();
    document.trailer.set(
        "ID",
        vec![
            Object::string_literal("pdf-raster-test-id"),
            Object::string_literal("pdf-raster-test-id"),
        ],
    );
    let encryption = lopdf::EncryptionState::try_from(lopdf::EncryptionVersion::V1 {
        document: &document,
        owner_password: "owner-secret",
        user_password,
        permissions: lopdf::Permissions::all(),
    })
    .unwrap();
    document.encrypt(&encryption).unwrap();
    save(&mut document)
}

#[test]
fn session_owns_input_and_renders_native_scan_and_blank_pages() {
    for backend in Backend::compiled() {
        let session = {
            let temporary_input = MIXED.to_vec();
            open(&temporary_input, backend)
        };
        assert_eq!(session.pages(), 3);
        assert_eq!(session.dimensions(1, 150.).unwrap(), (1275, 1650));
        let native = session.render(1, 150.).unwrap();
        assert_eq!(native.dimensions(), (1275, 1650));
        assert!(has_ink(&native, 60, 60, 1150, 800), "{}", backend.name());
        let scan = session.render(2, 150.).unwrap();
        assert!(has_ink(&scan, 60, 100, 1150, 800), "{}", backend.name());
        assert_ne!(native, scan);
        let blank = session.render(3, 150.).unwrap();
        assert!(
            blank.as_raw().iter().all(|v| *v == 255),
            "{}",
            backend.name()
        );
        assert_eq!(session.render(2, 150.).unwrap(), scan);
        assert!(session.render(0, 150.).is_err());
        assert!(session.render(4, 150.).is_err());
        assert!(session.dimensions(4, 150.).is_err());
    }
}

#[test]
fn crop_and_all_quarter_rotations_keep_upright_corner_pixels() {
    let fixtures: [(&[u8], u32, u32); 4] = [
        (include_bytes!("fixtures/rotate-crop-0.pdf"), 1250, 1667),
        (include_bytes!("fixtures/rotate-crop-90.pdf"), 1667, 1250),
        (include_bytes!("fixtures/rotate-crop-180.pdf"), 1250, 1667),
        (include_bytes!("fixtures/rotate-crop-270.pdf"), 1667, 1250),
    ];
    for backend in Backend::compiled() {
        for (fixture, width, height) in fixtures {
            let session = open(fixture, backend);
            assert_eq!(session.dimensions(1, 150.).unwrap(), (width, height));
            let image = session.render(1, 150.).unwrap();
            assert_eq!(image.dimensions(), (width, height));
            let inset = 73; // 35 page points at 150 DPI, well inside each colored square.
            assert_rgb(&image, inset, inset, [255, 0, 0], backend);
            assert_rgb(&image, width - inset, inset, [0, 255, 0], backend);
            assert_rgb(&image, inset, height - inset, [0, 0, 255], backend);
            assert_rgb(&image, width - inset, height - inset, [0, 0, 0], backend);
            // No outside-CropBox sentinel or MediaBox margin may appear at the edge.
            for x in 0..width {
                assert_eq!(image.get_pixel(x, 5).0, [255; 3], "{}", backend.name());
                assert_eq!(
                    image.get_pixel(x, height - 6).0,
                    [255; 3],
                    "{}",
                    backend.name()
                );
            }
        }
    }
}

#[test]
fn nested_forms_vectors_and_zero_alpha_are_drawn_as_visible_pixels() {
    for backend in Backend::compiled() {
        let session = open(
            include_bytes!("fixtures/nested-form-vector-alpha.pdf"),
            backend,
        );
        let image = session.render(1, 72.).unwrap();
        assert_eq!(image.dimensions(), (500, 400));
        assert_rgb(&image, 125, 225, [0, 179, 0], backend);
        assert_rgb(&image, 230, 220, [0, 0, 255], backend);
        assert!(has_ink(&image, 85, 105, 300, 30), "{}", backend.name());
        assert!(
            !has_ink(&image, 40, 30, 420, 50),
            "{}: zero-alpha text became visible",
            backend.name()
        );
    }
}

#[test]
fn image_soft_mask_composites_transparency_onto_white() {
    let mut document = one_page(
        b"q 240 0 0 100 30 50 cm /Im Do Q",
        dictionary! {},
        [0, 0, 300, 200],
    );
    let mask = document.add_object(Stream::new(
        dictionary! {
            "Type" => "XObject", "Subtype" => "Image", "Width" => 3, "Height" => 1,
            "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8, "Interpolate" => false
        },
        vec![0, 128, 255],
    ));
    let image = document.add_object(Stream::new(
        dictionary! {
            "Type" => "XObject", "Subtype" => "Image", "Width" => 3, "Height" => 1,
            "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8,
            "SMask" => mask, "Interpolate" => false
        },
        vec![255, 0, 0, 255, 0, 0, 255, 0, 0],
    ));
    let page = *document.get_pages().get(&1).unwrap();
    document
        .get_object_mut(page)
        .unwrap()
        .as_dict_mut()
        .unwrap()
        .set(
            "Resources",
            dictionary! { "XObject" => dictionary! { "Im" => image } },
        );
    let bytes = save(&mut document);
    for backend in Backend::compiled() {
        let rendered = open(&bytes, backend).render(1, 72.).unwrap();
        assert_rgb(&rendered, 70, 100, [255, 255, 255], backend);
        assert_rgb(&rendered, 150, 100, [255, 127, 127], backend);
        assert_rgb(&rendered, 230, 100, [255, 0, 0], backend);
        assert_rgb(&rendered, 10, 100, [255, 255, 255], backend);
    }
}

#[test]
fn crop_is_intersected_with_media_and_clips_outside_marks() {
    let mut document = one_page(
        b"0 0 1 rg -50 -50 40 40 re f 1 0 0 rg 0 0 25 50 re f",
        dictionary! {},
        [0, 0, 100, 100],
    );
    let page = *document.get_pages().get(&1).unwrap();
    document
        .get_object_mut(page)
        .unwrap()
        .as_dict_mut()
        .unwrap()
        .set(
            "CropBox",
            vec![Object::from(-50), (-50).into(), 50.into(), 50.into()],
        );
    let bytes = save(&mut document);
    for backend in Backend::compiled() {
        let session = open(&bytes, backend);
        assert_eq!(session.dimensions(1, 72.).unwrap(), (50, 50));
        let image = session.render(1, 72.).unwrap();
        assert_rgb(&image, 10, 25, [255, 0, 0], backend);
        assert_rgb(&image, 40, 25, [255, 255, 255], backend);
        assert!(
            !image.pixels().any(|pixel| pixel.0 == [0, 0, 255]),
            "{}",
            backend.name()
        );
    }
}

#[test]
fn fractional_page_boxes_keep_the_size_their_decimals_give() {
    // 593.76 x 245.28 points are exactly 1237 x 511 pixels at 150 DPI; read
    // as f32 they would round up to one pixel more.
    let mut document = one_page(b"", dictionary! {}, [0, 0, 1, 1]);
    let page = *document.get_pages().get(&1).unwrap();
    document
        .get_object_mut(page)
        .unwrap()
        .as_dict_mut()
        .unwrap()
        .set(
            "MediaBox",
            vec![
                Object::from(0),
                0.into(),
                Object::Real(593.76),
                Object::Real(245.28),
            ],
        );
    let bytes = save(&mut document);
    for backend in Backend::compiled() {
        let session = open(&bytes, backend);
        assert_eq!(
            session.dimensions(1, 150.).unwrap(),
            (1237, 511),
            "{}",
            backend.name()
        );
        assert_eq!(session.render(1, 150.).unwrap().dimensions(), (1237, 511));
    }
}

#[test]
fn malformed_locked_and_oversized_documents_fail_explicitly() {
    let mut huge = one_page(b"", dictionary! {}, [0, 0, 1_000_000, 1_000_000]);
    let huge_bytes = save(&mut huge);
    let locked = encrypted(MIXED, "user-secret");
    for backend in Backend::compiled() {
        // Rejected before either backend sees them.
        for bytes in [b"".as_slice(), b"not a PDF"] {
            assert!(PdfRasterSession::open_with(bytes, backend).is_err());
        }
        // CoreGraphics rejects a header without a document. Under Rosetta its
        // failed open crashes the process intermittently (SIGSEGV, SIGILL or
        // SIGBUS inside CGPDFDocumentCreateWithProvider, reproduced without
        // Rust); production first parses the bytes, so only this call is
        // skipped there.
        #[cfg(target_os = "macos")]
        let skip = backend == Backend::CoreGraphics && crate::system_frameworks::translated();
        #[cfg(not(target_os = "macos"))]
        let skip = false;
        if skip {
            eprintln!(
                "skipping CoreGraphics' rejection of a malformed PDF: it crashes intermittently \
                 under Rosetta translation (docs/validation/macos-x86_64-rosetta.md)"
            );
        } else {
            assert!(
                PdfRasterSession::open_with(b"%PDF-1.7\ninvalid document", backend).is_err(),
                "{}",
                backend.name()
            );
        }
        let huge = open(&huge_bytes, backend);
        assert!(
            huge.dimensions(1, 150.)
                .unwrap_err()
                .to_string()
                .contains("32 million")
        );
        assert!(huge.render(1, 150.).is_err());
        let normal = open(MIXED, backend);
        for dpi in [0., -1., f64::NAN, f64::INFINITY, f64::MAX] {
            assert!(normal.dimensions(1, dpi).is_err());
            assert!(normal.render(1, dpi).is_err());
        }
        let error = PdfRasterSession::open_with(&locked, backend)
            .err()
            .expect("password-protected PDF must fail");
        assert!(
            error.to_string().contains("locked"),
            "{}: {error}",
            backend.name()
        );
    }
}

#[test]
fn an_owner_password_only_document_renders_like_its_plain_original() {
    let protected = encrypted(MIXED, "");
    for backend in Backend::compiled() {
        let plain = open(MIXED, backend);
        let session = open(&protected, backend);
        assert_eq!(session.pages(), 3);
        for page in 1..=3 {
            assert_eq!(
                session.render(page, 72.).unwrap(),
                plain.render(page, 72.).unwrap(),
                "{}: page {page}",
                backend.name()
            );
        }
    }
}

#[test]
fn a_non_embedded_standard_font_is_drawn() {
    let mut document = one_page(
        b"BT /F1 36 Tf 20 40 Td (Standard) Tj ET",
        dictionary! {
            "Font" => dictionary! {
                "F1" => dictionary! {
                    "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica"
                }
            }
        },
        [0, 0, 300, 100],
    );
    let bytes = save(&mut document);
    for backend in Backend::compiled() {
        let image = open(&bytes, backend).render(1, 72.).unwrap();
        assert!(has_ink(&image, 20, 30, 200, 40), "{}", backend.name());
        assert!(!has_ink(&image, 250, 0, 50, 100), "{}", backend.name());
    }
}

#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
mod portable_only {
    use super::*;

    /// A Type0 font without a font program, Adobe-GB1 with a predefined
    /// UCS-2 CMap: what many Chinese PDFs written with Acrobat's Asian font
    /// packs contain. The portable renderer draws it with a host CJK face.
    fn non_embedded_chinese() -> Vec<u8> {
        let mut document = one_page(
            // 中文 in UniGB-UCS2-H: two-byte UCS-2 codes.
            b"BT /F1 48 Tf 20 30 Td <4E2D6587> Tj ET",
            dictionary! {},
            [0, 0, 200, 100],
        );
        let descriptor = document.add_object(dictionary! {
            "Type" => "FontDescriptor", "FontName" => "STSong-Light", "Flags" => 6,
            "FontBBox" => vec![Object::from(-25), (-254).into(), 1000.into(), 880.into()],
            "ItalicAngle" => 0, "Ascent" => 880, "Descent" => -120, "CapHeight" => 880,
            "StemV" => 80
        });
        let descendant = document.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "CIDFontType0", "BaseFont" => "STSong-Light",
            "CIDSystemInfo" => dictionary! {
                "Registry" => Object::string_literal("Adobe"),
                "Ordering" => Object::string_literal("GB1"),
                "Supplement" => 4
            },
            "FontDescriptor" => descriptor, "DW" => 1000
        });
        let font = document.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "Type0", "BaseFont" => "STSong-Light",
            "Encoding" => "UniGB-UCS2-H", "DescendantFonts" => vec![Object::Reference(descendant)]
        });
        let page = *document.get_pages().get(&1).unwrap();
        document
            .get_object_mut(page)
            .unwrap()
            .as_dict_mut()
            .unwrap()
            .set(
                "Resources",
                dictionary! { "Font" => dictionary! { "F1" => font } },
            );
        save(&mut document)
    }

    #[test]
    fn a_non_embedded_cjk_font_is_drawn_with_a_host_face_when_one_exists() {
        let session = open(&non_embedded_chinese(), Backend::Portable);
        let image = session.render(1, 72.).unwrap();
        let host_has_cjk = super::super::portable::host_has_cjk_face();
        if !host_has_cjk {
            eprintln!("skipping: this host has none of the listed Chinese faces");
            return;
        }
        assert!(has_ink(&image, 20, 20, 100, 50));
        assert_eq!(session.portable_warnings(), Some((0, 0)));
    }

    #[test]
    fn a_page_side_beyond_sixteen_bits_is_refused_before_drawing() {
        let mut wide = one_page(b"", dictionary! {}, [0, 0, 70_000, 10]);
        let session = open(&save(&mut wide), Backend::Portable);
        let error = session.dimensions(1, 72.).unwrap_err().to_string();
        assert!(error.contains("65,535"), "{error}");
        assert!(session.render(1, 72.).is_err());
        assert_eq!(session.dimensions(1, 36.).unwrap(), (35_000, 5));
    }

    #[test]
    fn encrypted_marker_in_plain_content_is_harmless() {
        let mut document = one_page(
            b"% /Encrypt is only a comment here\n",
            dictionary! {},
            [0, 0, 50, 50],
        );
        let session = open(&save(&mut document), Backend::Portable);
        assert!(
            session
                .render(1, 72.)
                .unwrap()
                .as_raw()
                .iter()
                .all(|v| *v == 255)
        );
    }
}
