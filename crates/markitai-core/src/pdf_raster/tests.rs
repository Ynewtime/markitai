use super::*;
use lopdf::{Document, Object, Stream, dictionary};

const MIXED: &[u8] = include_bytes!("fixtures/mixed-native-scanned-blank.pdf");

fn assert_rgb(image: &RgbImage, x: u32, y: u32, expected: [u8; 3]) {
    let actual = image.get_pixel(x, y).0;
    assert!(
        actual.iter().zip(expected).all(|(a, b)| a.abs_diff(b) <= 4),
        "pixel ({x}, {y}): {actual:?} != {expected:?}"
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

#[test]
fn session_owns_input_and_renders_native_scan_and_blank_pages() {
    let session = {
        let temporary_input = MIXED.to_vec();
        PdfRasterSession::open(&temporary_input).unwrap()
    };
    assert_eq!(session.pages(), 3);
    assert_eq!(session.dimensions(1, 150.).unwrap(), (1275, 1650));
    let native = session.render(1, 150.).unwrap();
    assert_eq!(native.dimensions(), (1275, 1650));
    assert!(has_ink(&native, 60, 60, 1150, 800));
    let scan = session.render(2, 150.).unwrap();
    assert!(has_ink(&scan, 60, 100, 1150, 800));
    assert_ne!(native, scan);
    let blank = session.render(3, 150.).unwrap();
    assert!(blank.as_raw().iter().all(|v| *v == 255));
    assert_eq!(session.render(2, 150.).unwrap(), scan);
    assert!(session.render(0, 150.).is_err());
    assert!(session.render(4, 150.).is_err());
    assert!(session.dimensions(4, 150.).is_err());
}

#[test]
fn crop_and_all_quarter_rotations_keep_upright_corner_pixels() {
    let fixtures: [(&[u8], u32, u32); 4] = [
        (include_bytes!("fixtures/rotate-crop-0.pdf"), 1250, 1667),
        (include_bytes!("fixtures/rotate-crop-90.pdf"), 1667, 1250),
        (include_bytes!("fixtures/rotate-crop-180.pdf"), 1250, 1667),
        (include_bytes!("fixtures/rotate-crop-270.pdf"), 1667, 1250),
    ];
    for (fixture, width, height) in fixtures {
        let session = PdfRasterSession::open(fixture).unwrap();
        assert_eq!(session.dimensions(1, 150.).unwrap(), (width, height));
        let image = session.render(1, 150.).unwrap();
        assert_eq!(image.dimensions(), (width, height));
        let inset = 73; // 35 page points at 150 DPI, well inside each colored square.
        assert_rgb(&image, inset, inset, [255, 0, 0]);
        assert_rgb(&image, width - inset, inset, [0, 255, 0]);
        assert_rgb(&image, inset, height - inset, [0, 0, 255]);
        assert_rgb(&image, width - inset, height - inset, [0, 0, 0]);
        // No outside-CropBox sentinel or MediaBox margin may appear at the edge.
        for x in 0..width {
            assert_eq!(image.get_pixel(x, 5).0, [255; 3]);
            assert_eq!(image.get_pixel(x, height - 6).0, [255; 3]);
        }
    }
}

#[test]
fn nested_forms_vectors_and_zero_alpha_are_drawn_as_visible_pixels() {
    let session =
        PdfRasterSession::open(include_bytes!("fixtures/nested-form-vector-alpha.pdf")).unwrap();
    let image = session.render(1, 72.).unwrap();
    assert_eq!(image.dimensions(), (500, 400));
    assert_rgb(&image, 125, 225, [0, 179, 0]);
    assert_rgb(&image, 230, 220, [0, 0, 255]);
    assert!(has_ink(&image, 85, 105, 300, 30));
    assert!(
        !has_ink(&image, 40, 30, 420, 50),
        "zero-alpha text became visible"
    );
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
    let rendered = PdfRasterSession::open(&save(&mut document))
        .unwrap()
        .render(1, 72.)
        .unwrap();
    assert_rgb(&rendered, 70, 100, [255, 255, 255]);
    assert_rgb(&rendered, 150, 100, [255, 127, 127]);
    assert_rgb(&rendered, 230, 100, [255, 0, 0]);
    assert_rgb(&rendered, 10, 100, [255, 255, 255]);
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
    let session = PdfRasterSession::open(&save(&mut document)).unwrap();
    assert_eq!(session.dimensions(1, 72.).unwrap(), (50, 50));
    let image = session.render(1, 72.).unwrap();
    assert_rgb(&image, 10, 25, [255, 0, 0]);
    assert_rgb(&image, 40, 25, [255, 255, 255]);
    assert!(!image.pixels().any(|pixel| pixel.0 == [0, 0, 255]));
}

#[test]
fn malformed_locked_and_oversized_documents_fail_explicitly() {
    // Rejected before CoreGraphics sees them.
    for bytes in [b"".as_slice(), b"not a PDF"] {
        assert!(PdfRasterSession::open(bytes).is_err());
    }
    // CoreGraphics rejects a header without a document. Under Rosetta its
    // failed open crashes the process intermittently (SIGSEGV, SIGILL or SIGBUS
    // inside CGPDFDocumentCreateWithProvider, reproduced without Rust);
    // production first parses the bytes, so only this call is skipped there.
    if crate::system_frameworks::translated() {
        eprintln!(
            "skipping CoreGraphics' rejection of a malformed PDF: it crashes intermittently under \
             Rosetta translation (docs/validation/macos-x86_64-rosetta.md)"
        );
    } else {
        assert!(PdfRasterSession::open(b"%PDF-1.7\ninvalid document").is_err());
    }
    let mut huge = one_page(b"", dictionary! {}, [0, 0, 1_000_000, 1_000_000]);
    let huge = PdfRasterSession::open(&save(&mut huge)).unwrap();
    assert!(
        huge.dimensions(1, 150.)
            .unwrap_err()
            .to_string()
            .contains("32 million")
    );
    assert!(huge.render(1, 150.).is_err());
    let normal = PdfRasterSession::open(MIXED).unwrap();
    for dpi in [0., -1., f64::NAN, f64::INFINITY, f64::MAX] {
        assert!(normal.dimensions(1, dpi).is_err());
        assert!(normal.render(1, dpi).is_err());
    }
    let mut locked = Document::load_mem(MIXED).unwrap();
    locked.trailer.set(
        "ID",
        vec![
            Object::string_literal("pdf-raster-test-id"),
            Object::string_literal("pdf-raster-test-id"),
        ],
    );
    let encryption = lopdf::EncryptionState::try_from(lopdf::EncryptionVersion::V1 {
        document: &locked,
        owner_password: "owner-secret",
        user_password: "user-secret",
        permissions: lopdf::Permissions::all(),
    })
    .unwrap();
    locked.encrypt(&encryption).unwrap();
    let error = PdfRasterSession::open(&save(&mut locked))
        .err()
        .expect("password-protected PDF must fail");
    assert!(error.to_string().contains("locked"), "{error}");
}

#[test]
fn invalid_box_geometry_is_rejected_before_sdk_drawing() {
    for rect in [
        CGRect::new(CGPoint::new(0., 0.), CGSize::new(0., 1.)),
        CGRect::new(CGPoint::new(f64::NAN, 0.), CGSize::new(1., 1.)),
        CGRect::new(CGPoint::new(0., 0.), CGSize::new(f64::INFINITY, 1.)),
        CGRect::new(CGPoint::new(f64::MAX, 0.), CGSize::new(f64::MAX, 1.)),
    ] {
        assert!(checked_rect(rect).is_err());
    }
    let first = CGRect::new(CGPoint::new(0., 0.), CGSize::new(10., 10.));
    let disjoint = CGRect::new(CGPoint::new(20., 20.), CGSize::new(10., 10.));
    assert!(intersection(first, disjoint).is_err());
}
