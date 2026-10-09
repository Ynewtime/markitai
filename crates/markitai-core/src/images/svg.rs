//! Static SVG rendering with explicit resource and external-reference boundaries.

use super::{MAX_PIXELS, error};
use crate::Result;
use base64::Engine;
use image::{DynamicImage, ImageFormat, RgbaImage};
use resvg::{tiny_skia, usvg};
use std::io::Cursor;
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

const MAX_SOURCE: usize = 8 * 1024 * 1024;
const MAX_NODES: u32 = 50_000;
const MAX_DEPTH: usize = 64;
const VISION_WIDTH: u32 = 2048;

fn check_structure(source: &str) -> Result<()> {
    use quick_xml::events::Event;
    let mut reader = quick_xml::Reader::from_str(source);
    let mut depth = 0_usize;
    let mut nodes = 0_u32;
    loop {
        let event = reader
            .read_event()
            .map_err(|cause| error(format!("Invalid SVG XML: {cause}")))?;
        match event {
            Event::Start(_) | Event::Empty(_) => {
                nodes += 1;
                if depth >= MAX_DEPTH {
                    return Err(error("SVG nesting exceeds 64 levels"));
                }
                if matches!(event, Event::Start(_)) {
                    depth += 1;
                }
            }
            Event::End(_) => depth = depth.saturating_sub(1),
            Event::Text(_) | Event::CData(_) | Event::Comment(_) | Event::GeneralRef(_) => {
                nodes += 1
            }
            Event::DocType(_) => return Err(error("SVG document types are not supported")),
            Event::PI(_) => return Err(error("SVG processing instructions are not supported")),
            Event::Eof => break,
            _ => {}
        }
        if nodes >= MAX_NODES {
            return Err(error("SVG exceeds the 50000 node limit"));
        }
    }
    if depth != 0 {
        return Err(error("Incomplete SVG XML"));
    }
    Ok(())
}

fn xml(bytes: &[u8]) -> Result<usvg::roxmltree::Document<'_>> {
    if bytes.len() > MAX_SOURCE {
        return Err(error("SVG exceeds the 8 MiB input limit"));
    }
    let source = std::str::from_utf8(bytes).map_err(|_| error("SVG must be UTF-8 XML"))?;
    // roxmltree's tokenizer recurses through elements. Enforce depth with an
    // iterative reader before constructing its document, not afterwards.
    check_structure(source)?;
    usvg::roxmltree::Document::parse_with_options(
        source,
        usvg::roxmltree::ParsingOptions {
            allow_dtd: false,
            nodes_limit: MAX_NODES,
            ..Default::default()
        },
    )
    .map_err(|cause| error(format!("Invalid SVG XML: {cause}")))
}

pub(super) fn is_svg(bytes: &[u8]) -> bool {
    // This branch is only consulted after raster magic detection fails.
    bytes
        .first()
        .is_some_and(|byte| matches!(byte, b'<' | b' ' | b'\t' | b'\n' | b'\r' | 0xef))
        && xml(bytes).is_ok_and(|document| svg_root(&document))
}

fn svg_root(document: &usvg::roxmltree::Document<'_>) -> bool {
    let root = document.root_element().tag_name();
    root.name() == "svg"
        && root
            .namespace()
            .is_none_or(|namespace| namespace == "http://www.w3.org/2000/svg")
}

fn preflight(document: &usvg::roxmltree::Document<'_>) -> Result<bool> {
    if !svg_root(document) {
        return Err(error("Expected an SVG root element"));
    }
    let mut text = false;
    for node in document.descendants() {
        if node.is_pi() {
            return Err(error("SVG processing instructions are not supported"));
        }
        if !node.is_element() {
            continue;
        }
        let name = node.tag_name().name();
        if matches!(
            name,
            "script"
                | "foreignObject"
                | "animate"
                | "animateMotion"
                | "animateTransform"
                | "set"
                | "discard"
        ) {
            return Err(error(format!(
                "SVG element '{name}' requires unsupported dynamic or embedded content"
            )));
        }
        text |= name == "text";
        for attribute in node.attributes() {
            if attribute.name().starts_with("on") {
                return Err(error("SVG event handlers are not supported"));
            }
            if attribute.name() != "href" || name == "a" {
                continue;
            }
            let target = attribute.value().trim();
            if target.starts_with('#') {
                continue;
            }
            if matches!(name, "image" | "feImage") && target.starts_with("data:") {
                let (header, data) = target
                    .split_once(',')
                    .ok_or_else(|| error("Invalid SVG image data URL"))?;
                if !matches!(
                    header,
                    "data:image/png;base64"
                        | "data:image/jpeg;base64"
                        | "data:image/jpg;base64"
                        | "data:image/gif;base64"
                        | "data:image/webp;base64"
                ) || base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .is_err()
                {
                    return Err(error(
                        "SVG image resources require valid base64 PNG, JPEG, GIF or WebP data",
                    ));
                }
                continue;
            }
            return Err(error(
                "SVG external resource references are disabled; embed raster images as data URLs",
            ));
        }
    }
    Ok(text)
}

fn fonts() -> Arc<usvg::fontdb::Database> {
    static FONTS: OnceLock<Arc<usvg::fontdb::Database>> = OnceLock::new();
    FONTS
        .get_or_init(|| {
            let mut database = usvg::fontdb::Database::new();
            database.load_system_fonts();
            set_generic_fallback(&mut database);
            Arc::new(database)
        })
        .clone()
}

// Font availability varies by host. Keep generic text usable even when the
// platform has no face with fontdb's default family names, but never hand it to
// an icon font: without a face covering basic Latin the text error stays explicit.
fn set_generic_fallback(database: &mut usvg::fontdb::Database) {
    let missing = [
        usvg::fontdb::Family::Serif,
        usvg::fontdb::Family::SansSerif,
        usvg::fontdb::Family::Monospace,
    ]
    .into_iter()
    .filter(|generic| {
        database
            .query(&usvg::fontdb::Query {
                families: &[*generic],
                ..Default::default()
            })
            .is_none()
    })
    .collect::<Vec<_>>();
    // Checking coverage reads font files, so skip it when nothing is missing.
    if missing.is_empty() {
        return;
    }
    let fallback_family = database
        .faces()
        .find(|face| has_latin_letters(database, face.id))
        .and_then(|face| face.families.first())
        .map(|item| item.0.clone());
    if let Some(family) = fallback_family {
        for generic in missing {
            match generic {
                usvg::fontdb::Family::Serif => database.set_serif_family(family.clone()),
                usvg::fontdb::Family::SansSerif => database.set_sans_serif_family(family.clone()),
                _ => database.set_monospace_family(family.clone()),
            }
        }
    }
}

fn has_latin_letters(database: &usvg::fontdb::Database, id: usvg::fontdb::ID) -> bool {
    database
        .with_face_data(id, |data, index| {
            skrifa::FontRef::from_index(data, index).is_ok_and(|font| {
                let charmap = skrifa::charmap::Charmap::new(&font);
                ('A'..='Z')
                    .chain('a'..='z')
                    .all(|letter| charmap.map(letter).is_some())
            })
        })
        .unwrap_or(false)
}

/// The width an SVG is rendered at.
#[derive(Clone, Copy)]
enum Width {
    Pixels(u32),
    /// A multiple of the SVG's own width, at most `VISION_WIDTH`.
    Times(f64),
}

/// For local OCR, an SVG renders at twice its own size (as the reference
/// rasterizes it), at most `VISION_WIDTH` wide: its text is then a size the
/// recognizer reads, without detecting text on a canvas ten times the
/// drawing.
const OCR_SCALE: f64 = 2.0;

pub(super) fn render(bytes: &[u8]) -> Result<DynamicImage> {
    render_with(bytes, Width::Pixels(VISION_WIDTH))
}

pub(super) fn render_for_ocr(bytes: &[u8]) -> Result<DynamicImage> {
    render_with(bytes, Width::Times(OCR_SCALE))
}

#[cfg(test)]
fn render_at_width(bytes: &[u8], width: u32) -> Result<DynamicImage> {
    render_with(bytes, Width::Pixels(width))
}

fn render_with(bytes: &[u8], width: Width) -> Result<DynamicImage> {
    let document = xml(bytes)?;
    let has_text = preflight(&document)?;
    let fontdb = if has_text {
        fonts()
    } else {
        Arc::new(usvg::fontdb::Database::new())
    };
    render_document(&document, width, fontdb)
}

fn render_document(
    document: &usvg::roxmltree::Document<'_>,
    pixel_width: Width,
    fontdb: Arc<usvg::fontdb::Database>,
) -> Result<DynamicImage> {
    let rejected_image = AtomicBool::new(false);
    let rejected_font = AtomicBool::new(false);
    let embedded_pixels = AtomicU64::new(0);
    let select_font = usvg::FontResolver::default_font_selector();
    let select_fallback = usvg::FontResolver::default_fallback_selector();
    let options = usvg::Options {
        fontdb,
        image_href_resolver: usvg::ImageHrefResolver {
            resolve_string: Box::new(|_, _| {
                rejected_image.store(true, Ordering::Relaxed);
                None
            }),
            resolve_data: Box::new(|_, bytes, _| {
                let decoded = (|| -> Result<_> {
                    if !matches!(
                        image::guess_format(&bytes),
                        Ok(ImageFormat::Png
                            | ImageFormat::Jpeg
                            | ImageFormat::Gif
                            | ImageFormat::WebP)
                    ) {
                        return Err(error("Unsupported SVG data image"));
                    }
                    let (image, _) = super::decode(&bytes)?;
                    let pixels = u64::from(image.width()) * u64::from(image.height());
                    // Rust 1.99 renames this `try_update`; the minimum supported Rust predates the new name.
                    #[allow(deprecated)]
                    embedded_pixels
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                            used.checked_add(pixels).filter(|sum| *sum <= MAX_PIXELS)
                        })
                        .map_err(|_| error("SVG embedded images exceed the pixel limit"))?;
                    let mut png = Cursor::new(Vec::new());
                    image.write_to(&mut png, ImageFormat::Png).map_err(error)?;
                    Ok(usvg::ImageKind::PNG(Arc::new(png.into_inner())))
                })();
                match decoded {
                    Ok(image) => Some(image),
                    Err(_) => {
                        rejected_image.store(true, Ordering::Relaxed);
                        None
                    }
                }
            }),
        },
        font_resolver: usvg::FontResolver {
            select_font: Box::new(|font, database| {
                let selected = select_font(font, database);
                if selected.is_none() {
                    rejected_font.store(true, Ordering::Relaxed);
                }
                selected
            }),
            select_fallback: Box::new(|character, used, database| {
                let selected = select_fallback(character, used, database);
                if selected.is_none() && !character.is_whitespace() {
                    rejected_font.store(true, Ordering::Relaxed);
                }
                selected
            }),
        },
        ..Default::default()
    };
    let tree = usvg::Tree::from_xmltree(document, &options).map_err(error)?;
    if rejected_image.load(Ordering::Relaxed) {
        return Err(error(
            "SVG contains an unsupported, invalid or oversized image resource",
        ));
    }
    if rejected_font.load(Ordering::Relaxed) {
        return Err(error(
            "SVG text requires a font or glyph unavailable on this host",
        ));
    }
    let width = tree.size().width();
    let height = tree.size().height();
    if f64::from(width).ceil() * f64::from(height).ceil() > MAX_PIXELS as f64 {
        return Err(error("SVG canvas exceeds 32 million pixels"));
    }
    let pixel_width = match pixel_width {
        Width::Pixels(pixels) => pixels,
        Width::Times(times) => (f64::from(width) * times)
            .ceil()
            .clamp(1.0, f64::from(VISION_WIDTH)) as u32,
    };
    let scale = f64::from(pixel_width) / f64::from(width);
    let scaled_height = (f64::from(height) * scale).ceil().max(1.0);
    if pixel_width == 0 || scaled_height * f64::from(pixel_width) > MAX_PIXELS as f64 {
        return Err(error("SVG vision canvas exceeds 32 million pixels"));
    }
    let pixel_height = scaled_height as u32;
    let mut pixmap = tiny_skia::Pixmap::new(pixel_width, pixel_height)
        .ok_or_else(|| error("Cannot allocate SVG canvas"))?;
    resvg::render(
        &tree,
        tiny_skia::Transform::from_scale(scale as f32, scale as f32),
        &mut pixmap.as_mut(),
    );
    if !pixmap.pixels().iter().any(|pixel| pixel.alpha() != 0) {
        return Err(error("SVG has no visible rendered content"));
    }
    let rgba = RgbaImage::from_raw(pixel_width, pixel_height, pixmap.take_demultiplied())
        .ok_or_else(|| error("Invalid SVG pixel buffer"))?;
    Ok(DynamicImage::ImageRgba8(rgba))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config;

    fn picture(body: &str) -> String {
        format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="80">{body}</svg>"#)
    }

    #[test]
    fn local_ocr_renders_twice_the_drawing_at_most_the_vision_width() {
        let small = r#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100"><rect width="200" height="100" fill="blue"/></svg>"#;
        let size = |image: DynamicImage| (image.width(), image.height());
        assert_eq!(size(render_for_ocr(small.as_bytes()).unwrap()), (400, 200));
        assert_eq!(size(render(small.as_bytes()).unwrap()), (2048, 1024));
        let large = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 3000 1500"><rect width="3000" height="1500" fill="blue"/></svg>"#;
        assert_eq!(
            size(render_for_ocr(large.as_bytes()).unwrap()),
            (2048, 1024)
        );
    }

    #[test]
    fn geometry_alpha_and_viewbox_are_rendered_to_straight_rgba() {
        let source = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 80"><rect x="10" y="20" width="40" height="30" fill="red" fill-opacity="0.5"/></svg>"#;
        let image = render_at_width(source.as_bytes(), 100).unwrap().to_rgba8();
        assert_eq!(image.dimensions(), (100, 80));
        assert_eq!(image.get_pixel(0, 0).0, [0, 0, 0, 0]);
        let red = image.get_pixel(20, 30).0;
        assert_eq!(&red[..3], &[255, 0, 0]);
        assert!((127..=128).contains(&red[3]));
    }

    #[test]
    fn original_svg_asset_and_png_vision_are_separate() {
        let temp = tempfile::tempdir().unwrap();
        let source = picture(r#"<rect width="100" height="80" fill="blue"/>"#);
        let cfg =
            config::normalize(&serde_json::json!({"image":{"max_width":25,"compress":false}}))
                .unwrap();
        for name in ["input.svg", "actually-vector.png", "actually-vector.heic"] {
            let path = temp.path().join(name);
            std::fs::write(&path, &source).unwrap();
            let (document, vision) = super::super::extract(&path, &cfg, false).unwrap();
            assert_eq!(document.assets.len(), 1);
            assert_eq!(document.assets[0].bytes, source.as_bytes());
            assert_eq!(document.assets[0].name, "image.svg");
            assert!(document.markdown.contains(".markitai/assets/image.svg"));
            assert_eq!(vision.len(), 1);
            let vision = &vision[0];
            assert_eq!(vision.mime, "image/png");
            let pixels = image::load_from_memory(&vision.bytes).unwrap().to_rgba8();
            assert_eq!(pixels.dimensions(), (2048, 1639));
            assert_eq!(pixels.get_pixel(10, 10).0, [0, 0, 255, 255]);
        }
    }

    #[test]
    fn text_uses_available_fonts_and_missing_fonts_are_explicit() {
        let source = picture(
            r#"<text x="4" y="40" font-size="24" font-family="sans-serif">Hello SVG</text>"#,
        );
        let document = xml(source.as_bytes()).unwrap();
        let empty = Arc::new(usvg::fontdb::Database::new());
        assert!(
            render_document(&document, Width::Pixels(100), empty)
                .unwrap_err()
                .to_string()
                .contains("font")
        );
        // Hosts with only icon fonts have faces but no usable generic family.
        if fonts()
            .query(&usvg::fontdb::Query {
                families: &[usvg::fontdb::Family::SansSerif],
                ..Default::default()
            })
            .is_some()
        {
            let image = render_at_width(source.as_bytes(), 100).unwrap().to_rgba8();
            let visible = image.pixels().filter(|pixel| pixel[3] != 0).count();
            assert!(
                visible > 50 && visible < 4_000,
                "visible glyph pixels: {visible}"
            );
        }
    }

    /// A face with only `cmap` and `name` tables: enough for fontdb and charmaps.
    fn synthetic_font(family: &str, ranges: &[(char, char)]) -> Vec<u8> {
        let mut cmap = [0u16, 1, 3, 10]
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect::<Vec<_>>();
        cmap.extend(12u32.to_be_bytes());
        cmap.extend([12u16, 0].iter().flat_map(|value| value.to_be_bytes()));
        let length = 16 + 12 * ranges.len() as u32;
        for value in [length, 0, ranges.len() as u32] {
            cmap.extend(value.to_be_bytes());
        }
        let mut glyph = 1;
        for (start, end) in ranges {
            for value in [*start as u32, *end as u32, glyph] {
                cmap.extend(value.to_be_bytes());
            }
            glyph += *end as u32 - *start as u32 + 1;
        }
        let text = family
            .encode_utf16()
            .flat_map(u16::to_be_bytes)
            .collect::<Vec<_>>();
        let mut name = [0u16, 2, 30]
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect::<Vec<_>>();
        for name_id in [1u16, 6] {
            for value in [3, 1, 0x0409, name_id, text.len() as u16, 0] {
                name.extend(u16::to_be_bytes(value));
            }
        }
        name.extend(&text);
        let mut font = [1u16, 0, 2, 32, 1, 0]
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect::<Vec<_>>();
        let mut offset = 12 + 2 * 16;
        for (tag, table) in [(b"cmap", &cmap), (b"name", &name)] {
            font.extend(tag);
            for value in [0, offset, table.len() as u32] {
                font.extend(value.to_be_bytes());
            }
            offset += table.len() as u32;
        }
        font.extend(cmap);
        font.extend(name);
        font
    }

    #[test]
    fn generic_text_fallback_skips_faces_without_latin_letters() {
        let generic = |database: &usvg::fontdb::Database, family| {
            let id = database.query(&usvg::fontdb::Query {
                families: &[family],
                ..Default::default()
            })?;
            Some(database.face(id)?.families[0].0.clone())
        };
        let mut database = usvg::fontdb::Database::new();
        database.load_font_data(synthetic_font("Icons", &[('\u{f000}', '\u{f2ff}')]));
        // Ligature icon fonts map lowercase letters only.
        database.load_font_data(synthetic_font("Ligatures", &[('a', 'z')]));
        assert_eq!(database.len(), 2);

        let mut icons = database.clone();
        set_generic_fallback(&mut icons);
        assert_eq!(generic(&icons, usvg::fontdb::Family::SansSerif), None);
        let source = picture(
            r#"<text x="4" y="40" font-size="24" font-family="sans-serif">Hello SVG</text>"#,
        );
        let document = xml(source.as_bytes()).unwrap();
        assert!(
            render_document(&document, Width::Pixels(100), Arc::new(icons))
                .unwrap_err()
                .to_string()
                .contains("font or glyph unavailable")
        );

        database.load_font_data(synthetic_font("Text", &[('A', 'Z'), ('a', 'z')]));
        database.load_font_data(synthetic_font("Later Text", &[('A', 'Z'), ('a', 'z')]));
        set_generic_fallback(&mut database);
        for family in [
            usvg::fontdb::Family::Serif,
            usvg::fontdb::Family::SansSerif,
            usvg::fontdb::Family::Monospace,
        ] {
            assert_eq!(generic(&database, family).as_deref(), Some("Text"));
        }
    }

    #[test]
    fn external_files_network_entities_and_active_content_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("secret.png");
        std::fs::write(&file, "not an image, never load it").unwrap();
        for target in [
            file.to_str().unwrap(),
            "file:///etc/passwd",
            "http://127.0.0.1:1/no-request",
            "missing.png",
        ] {
            let source = picture(&format!(
                r#"<image href="{target}" width="100" height="80"/>"#
            ));
            assert!(
                render_at_width(source.as_bytes(), 100)
                    .unwrap_err()
                    .to_string()
                    .contains("external resource")
            );
        }
        for source in [
            "<!DOCTYPE svg [<!ENTITY x SYSTEM 'file:///etc/passwd'>]><svg>&x;</svg>".to_owned(),
            picture("<foreignObject><div>Hidden loss</div></foreignObject>"),
            picture("<script>untrusted()</script>"),
            picture("<animate attributeName='x'/>"),
            picture("<rect width='100' height='80' onclick='run()'/>"),
        ] {
            assert!(render_at_width(source.as_bytes(), 100).is_err());
        }
    }

    #[test]
    fn embedded_raster_data_is_bounded_and_bad_data_never_becomes_blank_success() {
        use base64::Engine;
        let mut buffer = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, image::Rgba([0, 255, 0, 255])))
            .write_to(&mut buffer, ImageFormat::Png)
            .unwrap();
        let data = base64::engine::general_purpose::STANDARD.encode(buffer.into_inner());
        let source = picture(&format!(
            r#"<image href="data:image/png;base64,{data}" width="100" height="80"/>"#
        ));
        let pixels = render_at_width(source.as_bytes(), 100).unwrap().to_rgba8();
        assert_eq!(pixels.get_pixel(50, 40).0, [0, 255, 0, 255]);
        for data in [
            "data:image/png;base64,AQID",
            "data:image/svg+xml,%3Csvg/%3E",
            "data:image/png;base64,!invalid!",
        ] {
            let source = picture(&format!(
                r#"<rect width="100" height="80"/><image href="{data}" width="10" height="10"/>"#
            ));
            assert!(
                render_at_width(source.as_bytes(), 100)
                    .unwrap_err()
                    .to_string()
                    .contains("image resource")
            );
        }
    }

    #[test]
    fn malformed_empty_large_and_deep_inputs_fail() {
        for source in [
            "<svg>",
            "<not-svg/>",
            "<svg/>",
            "<svg width='100000' height='100000'><rect width='1' height='1'/></svg>",
        ] {
            assert!(render_at_width(source.as_bytes(), 100).is_err(), "{source}");
        }
        let deep = format!(
            "<svg>{}<rect width='1' height='1'/>{}</svg>",
            "<g>".repeat(129),
            "</g>".repeat(129)
        );
        assert!(
            render_at_width(deep.as_bytes(), 100)
                .unwrap_err()
                .to_string()
                .contains("nesting")
        );
        assert!(
            xml(&vec![b' '; MAX_SOURCE + 1])
                .unwrap_err()
                .to_string()
                .contains("8 MiB")
        );
        let nested = format!(
            "<svg width='2' height='2'>{}<rect width='2' height='2'/>{}</svg>",
            "<g>".repeat(16),
            "</g>".repeat(16)
        );
        assert!(render_at_width(nested.as_bytes(), 2).is_ok());
        // SVG + 62 groups + the empty rectangle are exactly 64 element levels.
        let boundary = format!(
            "<svg width='2' height='2'>{}<rect width='2' height='2'/>{}</svg>",
            "<g>".repeat(MAX_DEPTH - 2),
            "</g>".repeat(MAX_DEPTH - 2)
        );
        assert_eq!(
            render_at_width(boundary.as_bytes(), 2)
                .unwrap()
                .to_rgba8()
                .get_pixel(0, 0)
                .0,
            [0, 0, 0, 255]
        );
        let over_boundary = boundary
            .replacen("<rect", "<g><rect", 1)
            .replacen("/>", "/></g>", 1);
        assert!(
            xml(over_boundary.as_bytes())
                .unwrap_err()
                .to_string()
                .contains("nesting")
        );
        let tall = b"<svg width='1' height='100'><rect width='1' height='100'/></svg>";
        assert!(
            render(tall)
                .unwrap_err()
                .to_string()
                .contains("vision canvas")
        );
        let many = format!("<svg>{}</svg>", "<g/>".repeat(MAX_NODES as usize));
        assert!(xml(many.as_bytes()).is_err());
    }
}
