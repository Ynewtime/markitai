//! Bounded image decoding and shared asset preparation.

mod heif;
mod svg;
mod tiff;

use crate::{Asset, Document, Error, Result, config, output_profiles};
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Rgb, RgbImage};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{BufRead, Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;

const MAX_PIXELS: u64 = 32_000_000;
const MAX_DECODED: u64 = 256 * 1024 * 1024;

pub(crate) struct VisionImage {
    pub mime: &'static str,
    pub bytes: Vec<u8>,
}

/// Image extensions (without the dot) read as images.
pub const IMAGE_EXTENSIONS: &[&str] = &[
    "jpeg", "jpg", "png", "webp", "gif", "bmp", "tiff", "tif", "svg", "heic", "heif", "avif",
];

pub fn is_image_extension(extension: &str) -> bool {
    let extension = extension.trim_start_matches('.').to_ascii_lowercase();
    IMAGE_EXTENSIONS.contains(&extension.as_str())
}

fn error(error: impl std::fmt::Display) -> Error {
    Error::Conversion(format!("Cannot decode or encode image: {error}"))
}

#[derive(Debug, Clone, Copy)]
enum DecodedFormat {
    Raster(ImageFormat),
    Heif(heif::Info),
}

fn decode(bytes: &[u8]) -> Result<(DynamicImage, DecodedFormat)> {
    if heif::signature(bytes) {
        let decoded = heif::decode(bytes)?;
        return Ok((decoded.image, DecodedFormat::Heif(decoded.info)));
    }
    let mut reader = ImageReader::new(ImageBytes::new(bytes)).with_guessed_format()?;
    let format = reader
        .format()
        .ok_or_else(|| error("unrecognized image format"))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_PIXELS as u32);
    limits.max_image_height = Some(MAX_PIXELS as u32);
    limits.max_alloc = Some(MAX_DECODED);
    reader.limits(limits);
    let mut decoder = reader.into_decoder().map_err(error)?;
    let (width, height) = decoder.dimensions();
    if u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(error("decoded image exceeds 32 million pixels"));
    }
    let orientation = decoder.orientation().map_err(error)?;
    let mut image = DynamicImage::from_decoder(decoder).map_err(error)?;
    image.apply_orientation(orientation);
    Ok((image, DecodedFormat::Raster(format)))
}

fn dimension(cfg: &Value, key: &str, fallback: u32) -> u32 {
    cfg.pointer(key)
        .and_then(Value::as_u64)
        .unwrap_or(u64::from(fallback))
        .min(u64::from(u32::MAX)) as u32
}

fn filtered(image: &DynamicImage, cfg: &Value) -> bool {
    image.width() < dimension(cfg, "/image/filter/min_width", 50)
        || image.height() < dimension(cfg, "/image/filter/min_height", 50)
        || u64::from(image.width()) * u64::from(image.height())
            < cfg
                .pointer("/image/filter/min_area")
                .and_then(Value::as_u64)
                .unwrap_or(5000)
}

fn rgb_on_white(image: &DynamicImage) -> RgbImage {
    if !image.color().has_alpha() {
        return image.to_rgb8();
    }
    let rgba = image.to_rgba8();
    RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let pixel = rgba.get_pixel(x, y).0;
        let alpha = u32::from(pixel[3]);
        Rgb([0, 1, 2].map(|channel| {
            ((u32::from(pixel[channel]) * alpha + 255 * (255 - alpha) + 127) / 255) as u8
        }))
    })
}

fn encode(
    image: &DynamicImage,
    cfg: &Value,
    force_png: bool,
) -> Result<(Vec<u8>, &'static str, &'static str)> {
    encode_limited(image, cfg, force_png, tiff::MAX_ENCODED)
}

fn encode_limited(
    image: &DynamicImage,
    cfg: &Value,
    force_png: bool,
    limit: usize,
) -> Result<(Vec<u8>, &'static str, &'static str)> {
    let width = dimension(cfg, "/image/max_width", 1920).max(1);
    let height = dimension(cfg, "/image/max_height", 99999).max(1);
    let resized;
    let image = if !force_png && (image.width() > width || image.height() > height) {
        resized = image.resize(width, height, image::imageops::FilterType::Lanczos3);
        &resized
    } else {
        image
    };
    let format = if force_png {
        "png"
    } else {
        cfg.pointer("/image/format")
            .and_then(Value::as_str)
            .unwrap_or("jpeg")
    };
    let mut buffer = EncodedBuffer {
        inner: Cursor::new(Vec::new()),
        limit,
    };
    let (extension, mime) = match format {
        "png" => {
            image
                .write_to(&mut buffer, ImageFormat::Png)
                .map_err(error)?;
            ("png", "image/png")
        }
        "webp" => {
            // The Rust encoder is lossless; callers disclose this quality difference.
            image
                .write_to(&mut buffer, ImageFormat::WebP)
                .map_err(error)?;
            ("webp", "image/webp")
        }
        _ => {
            let quality = dimension(cfg, "/image/quality", 75).clamp(1, 100) as u8;
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buffer, quality)
                .encode_image(&rgb_on_white(image))
                .map_err(error)?;
            ("jpg", "image/jpeg")
        }
    };
    Ok((buffer.inner.into_inner(), extension, mime))
}

struct EncodedBuffer {
    inner: Cursor<Vec<u8>>,
    limit: usize,
}
impl Write for EncodedBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self
            .inner
            .position()
            .checked_add(bytes.len() as u64)
            .is_none_or(|end| end > self.limit as u64)
        {
            return Err(std::io::Error::other("Encoded image byte budget exceeded"));
        }
        self.inner.write(bytes)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
impl Seek for EncodedBuffer {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(position)
    }
}

/// The reader every image and TIFF decode reads through: a byte slice whose
/// first bytes may be replaced by a TIFF header that points at another page's
/// directory (see `tiff::page_reader`). The decoders are generic over their
/// reader and compiled again for every reader type, so one type for all of
/// them keeps one copy of each, where a `Cursor` here beside the TIFF page
/// reader kept two (about 60 KB of code).
pub(crate) struct ImageBytes<'a> {
    bytes: &'a [u8],
    header: [u8; 16],
    header_len: usize,
    position: u64,
}

impl<'a> ImageBytes<'a> {
    /// The bytes as they are, read as a `Cursor` over them reads.
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            header: [0; 16],
            header_len: 0,
            position: 0,
        }
    }
}

impl Read for ImageBytes<'_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let source = self.fill_buf()?;
        let count = out.len().min(source.len());
        out[..count].copy_from_slice(&source[..count]);
        self.consume(count);
        Ok(count)
    }

    /// Copies the rest at once, as a `Cursor` does, rather than through the
    /// default's growing reads.
    fn read_to_end(&mut self, out: &mut Vec<u8>) -> std::io::Result<usize> {
        let start = out.len();
        loop {
            let source = self.fill_buf()?;
            if source.is_empty() {
                return Ok(out.len() - start);
            }
            out.extend_from_slice(source);
            let count = source.len();
            self.consume(count);
        }
    }
}

impl BufRead for ImageBytes<'_> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        let position = usize::try_from(self.position).unwrap_or(usize::MAX);
        if position < self.header_len {
            return Ok(&self.header[position..self.header_len]);
        }
        Ok(self.bytes.get(position..).unwrap_or_default())
    }

    fn consume(&mut self, amount: usize) {
        self.position = self.position.saturating_add(amount as u64);
    }
}

impl Seek for ImageBytes<'_> {
    fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
        let position = match from {
            SeekFrom::Start(position) => Some(position),
            SeekFrom::End(delta) => (self.bytes.len() as u64).checked_add_signed(delta),
            SeekFrom::Current(delta) => self.position.checked_add_signed(delta),
        }
        .ok_or_else(|| {
            // The messages a `Cursor` and the TIFF page reader gave.
            let message = if self.header_len == 0 {
                "invalid seek to a negative or overflowing position"
            } else {
                "invalid TIFF seek"
            };
            std::io::Error::new(std::io::ErrorKind::InvalidInput, message)
        })?;
        self.position = position;
        Ok(position)
    }
}

fn notice(doc: &mut Document, message: &str) {
    if !doc.warnings.iter().any(|existing| existing == message) {
        doc.warnings.push(message.into());
    }
}

/// Prepare embedded raster assets without touching the input or fetching URLs.
pub(crate) fn prepare_assets(doc: &mut Document, cfg: &Value) {
    let mut prepared = Vec::with_capacity(doc.assets.len());
    let mut seen: HashMap<_, String> = HashMap::new();
    let mut replacements = HashMap::with_capacity(doc.assets.len());
    for asset in std::mem::take(&mut doc.assets) {
        let from = format!(".markitai/assets/{}", asset.name);
        let extension = Path::new(&asset.name)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if !is_image_extension(extension)
            && image::guess_format(&asset.bytes).is_err()
            && !heif::signature(&asset.bytes)
        {
            replacements.entry(from.clone()).or_insert(from);
            prepared.push(asset);
            continue;
        }
        let digest = Sha256::digest(&asset.bytes);
        if config::enabled(cfg, "/image/filter/deduplicate")
            && let Some(target) = seen.get(&digest)
        {
            replacements.entry(from).or_insert_with(|| target.clone());
            continue;
        }
        // Compression applies to page previews, never the TIFF document itself.
        if tiff::signature(&asset.bytes) {
            match tiff::multiple(&asset.bytes) {
                Ok(false) => {}
                result => {
                    if result.is_err() {
                        notice(
                            doc,
                            "A TIFF asset could not be decoded; it was preserved unchanged and was not filtered or compressed.",
                        );
                    }
                    replacements.entry(from.clone()).or_insert(from.clone());
                    seen.entry(digest).or_insert(from);
                    prepared.push(asset);
                    continue;
                }
            }
        }
        let (image, format) = match decode(&asset.bytes) {
            Ok(value) => value,
            Err(_) => {
                notice(
                    doc,
                    "An embedded image could not be decoded within native limits; its original asset was retained without filtering or compression.",
                );
                replacements.entry(from.clone()).or_insert(from);
                prepared.push(asset);
                continue;
            }
        };
        if let DecodedFormat::Heif(info) = format {
            primary_notice(doc, info);
        }
        if filtered(&image, cfg) {
            replacements.entry(from).or_default();
            continue;
        }
        let compress = config::enabled(cfg, "/image/compress");
        let asset = if compress || matches!(format, DecodedFormat::Heif(_)) {
            match encode(&image, cfg, !compress) {
                Ok((bytes, extension, _)) => Asset {
                    name: format!("{}.{}", asset.name, extension),
                    bytes,
                },
                Err(_) => {
                    notice(
                        doc,
                        "An embedded image could not be encoded; its original asset was retained.",
                    );
                    asset
                }
            }
        } else {
            // Correct mislabeled embedded payloads even when preserving their bytes.
            let DecodedFormat::Raster(format) = format else {
                unreachable!()
            };
            let suffix = format.extensions_str()[0];
            Asset {
                name: format!("{}.{}", asset.name, suffix),
                ..asset
            }
        };
        if config::enabled(cfg, "/image/compress") && cfg["image"]["format"] == "webp" {
            notice(
                doc,
                "WebP output uses lossless native encoding; image.quality does not affect WebP in this build.",
            );
        }
        let target = format!(".markitai/assets/{}", asset.name);
        replacements.entry(from).or_insert_with(|| target.clone());
        seen.insert(digest, target);
        prepared.push(asset);
    }
    // Every key refers to the original document. A prepared name may also be
    // another input name, so applying replacements one by one would cascade.
    // Ambiguous duplicate source names retain the first asset's decision, even
    // when it keeps its original name or is filtered out.
    if replacements.iter().any(|(before, after)| before != after) {
        doc.markdown = output_profiles::rewrite_asset_references(&doc.markdown, &replacements);
    }
    doc.assets = prepared;
}

pub(crate) fn extract(
    path: &Path,
    cfg: &Value,
    local_ocr: bool,
) -> Result<(Document, Vec<VisionImage>)> {
    let mut bytes = Vec::new();
    // The shared input policy bounds ordinary files. A growing input cannot
    // bypass that ceiling while being read.
    std::fs::File::open(path)?
        .take(500 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 500 * 1024 * 1024 {
        return Err(error("image input exceeds 500 MiB"));
    }
    if tiff::signature(&bytes) && tiff::multiple(&bytes)? {
        return extract_tiff(path, bytes, cfg, local_ocr);
    }
    if heif::signature(&bytes) {
        return extract_heif(path, &bytes, cfg, local_ocr);
    }
    let (image, svg) = single_vision(&bytes, path, cfg)?;
    let (name, asset) = if svg {
        ("image.svg".into(), bytes.clone())
    } else {
        (
            format!("image.{}", mime_extension(image.mime)),
            image.bytes.clone(),
        )
    };
    let mut doc = image_document(path, &name, asset);
    if svg {
        doc.metadata.insert("format".into(), "SVG".into());
    }
    if !svg {
        webp_notice(&mut doc, cfg);
    }
    if local_ocr {
        let recognized = if tiff::signature(&bytes) {
            crate::ocr::recognize_rgb(rgb_on_white(&tiff::Pages::new(&bytes)?.decode(0)?), cfg)?
        } else {
            crate::ocr::recognize(if svg { &image.bytes } else { &bytes }, cfg)?
        };
        if recognized.text.trim().is_empty() {
            doc.warnings.push(
                "Local OCR found no readable text; the output retains the image reference.".into(),
            );
        } else {
            doc.markdown.push('\n');
            doc.markdown.push_str(&recognized.text);
            if !doc.markdown.ends_with('\n') {
                doc.markdown.push('\n');
            }
        }
        ocr_metadata(&mut doc, cfg);
        Ok((doc, Vec::new()))
    } else {
        Ok((doc, vec![image]))
    }
}

/// Prepare every document page for one image-analysis request. Animated image
/// formats retain their existing first-frame policy when transcoding.
pub(crate) fn prepare_vision(bytes: &[u8], name: &str, cfg: &Value) -> Result<Vec<VisionImage>> {
    if tiff::signature(bytes) {
        let pages = tiff::Pages::new(bytes)?;
        pages.check_vision_limit(cfg)?;
        if pages.len() > 1 {
            let mut images = Vec::with_capacity(pages.len());
            let mut remaining = tiff::MAX_ENCODED;
            for index in 0..pages.len() {
                let image = pages.decode(index)?;
                let (bytes, _, mime) = encode_limited(
                    &image,
                    cfg,
                    !config::enabled(cfg, "/image/compress"),
                    remaining,
                )?;
                remaining -= bytes.len();
                images.push(VisionImage { mime, bytes });
            }
            return Ok(images);
        }
    }
    Ok(vec![single_vision(bytes, Path::new(name), cfg)?.0])
}

fn single_vision(bytes: &[u8], path: &Path, cfg: &Value) -> Result<(VisionImage, bool)> {
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let detected = image::guess_format(bytes)
        .ok()
        .or_else(|| tiff::signature(bytes).then_some(ImageFormat::Tiff));
    if detected.is_none() && !heif::signature(bytes) && (extension == "svg" || svg::is_svg(bytes)) {
        let rendered = svg::render(bytes)?;
        let (bytes, _, mime) = encode(&rendered, cfg, true)?;
        return Ok((VisionImage { mime, bytes }, true));
    }
    let (image, format) = if detected == Some(ImageFormat::Tiff) {
        (
            tiff::Pages::new(bytes)?.decode(0)?,
            DecodedFormat::Raster(ImageFormat::Tiff),
        )
    } else {
        decode(bytes)?
    };
    let preserve = !config::enabled(cfg, "/image/compress");
    let (bytes, mime) = if preserve
        && matches!(
            format,
            DecodedFormat::Raster(
                ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP | ImageFormat::Gif
            )
        ) {
        (
            bytes.to_vec(),
            match format {
                DecodedFormat::Raster(ImageFormat::Jpeg) => "image/jpeg",
                DecodedFormat::Raster(ImageFormat::WebP) => "image/webp",
                DecodedFormat::Raster(ImageFormat::Gif) => "image/gif",
                _ => "image/png",
            },
        )
    } else {
        let (bytes, _, mime) = encode(&image, cfg, preserve)?;
        (bytes, mime)
    };
    Ok((VisionImage { mime, bytes }, false))
}

fn primary_notice(doc: &mut Document, info: heif::Info) {
    if info.images > 1 {
        notice(
            doc,
            &format!(
                "{} contains {} images; only primary image {} was converted, matching the primary-image policy.",
                info.format,
                info.images,
                info.primary + 1
            ),
        );
    }
}

fn extract_heif(
    path: &Path,
    bytes: &[u8],
    cfg: &Value,
    local_ocr: bool,
) -> Result<(Document, Vec<VisionImage>)> {
    let decoded = heif::decode(bytes)?;
    let (encoded, extension, mime) = encode(
        &decoded.image,
        cfg,
        !config::enabled(cfg, "/image/compress"),
    )?;
    let mut doc = image_document(path, &format!("image.{extension}"), encoded.clone());
    doc.metadata
        .insert("format".into(), decoded.info.format.into());
    doc.metadata
        .insert("image_count".into(), decoded.info.images.into());
    doc.metadata
        .insert("primary_image".into(), (decoded.info.primary + 1).into());
    primary_notice(&mut doc, decoded.info);
    webp_notice(&mut doc, cfg);
    if local_ocr {
        let recognized = crate::ocr::recognize_rgb(rgb_on_white(&decoded.image), cfg)?;
        if recognized.text.trim().is_empty() {
            doc.warnings.push(
                "Local OCR found no readable text; the output retains the image reference.".into(),
            );
        } else {
            doc.markdown.push('\n');
            doc.markdown.push_str(&recognized.text);
            if !doc.markdown.ends_with('\n') {
                doc.markdown.push('\n');
            }
        }
        ocr_metadata(&mut doc, cfg);
        Ok((doc, Vec::new()))
    } else {
        Ok((
            doc,
            vec![VisionImage {
                mime,
                bytes: encoded,
            }],
        ))
    }
}

fn mime_extension(mime: &str) -> &'static str {
    match mime {
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        _ => "png",
    }
}

fn webp_notice(doc: &mut Document, cfg: &Value) {
    if config::enabled(cfg, "/image/compress")
        && cfg.pointer("/image/format").and_then(Value::as_str) == Some("webp")
    {
        notice(
            doc,
            "WebP output uses the native lossless encoder; image.quality applies to JPEG output.",
        );
    }
}

fn ocr_metadata(doc: &mut Document, cfg: &Value) {
    doc.metadata.insert("ocr_used".into(), true.into());
    doc.metadata
        .insert("ocr_path".into(), crate::ocr::backend().into());
    if config::enabled(cfg, "/llm/enabled") {
        doc.warnings.push(
            "VLM OCR is disabled; only locally recognized text is sent for LLM enhancement.".into(),
        );
    }
}

fn extract_tiff(
    path: &Path,
    bytes: Vec<u8>,
    cfg: &Value,
    local_ocr: bool,
) -> Result<(Document, Vec<VisionImage>)> {
    let pages = tiff::Pages::new(&bytes)?;
    if !local_ocr {
        pages.check_vision_limit(cfg)?;
    }
    let mut doc = image_document(path, "original.tiff", Vec::new());
    let title = doc.metadata["title"].as_str().unwrap_or_default();
    doc.markdown = format!("# {title}\n\n[Original TIFF](.markitai/assets/original.tiff)\n");
    doc.assets.clear();
    doc.metadata.insert("format".into(), "TIFF".into());
    doc.metadata.insert("page_count".into(), pages.len().into());
    let mut vision = Vec::with_capacity(if local_ocr { 0 } else { pages.len() });
    let mut remaining = tiff::MAX_ENCODED - bytes.len();
    for index in 0..pages.len() {
        let image = pages.decode(index)?;
        let (encoded, extension, mime) = encode_limited(
            &image,
            cfg,
            !config::enabled(cfg, "/image/compress"),
            remaining / if local_ocr { 1 } else { 2 },
        )?;
        let name = format!("page{:04}.{extension}", index + 1);
        doc.markdown.push_str(&format!(
            "\n<!-- Page number: {} -->\n\n![Page {}](.markitai/assets/{name})\n",
            index + 1,
            index + 1
        ));
        if local_ocr {
            let recognized = crate::ocr::recognize_rgb(rgb_on_white(&image), cfg)?;
            if recognized.text.trim().is_empty() {
                doc.warnings.push(format!("Local OCR found no readable text on TIFF page {}; the output retains its image reference.", index + 1));
            } else {
                if doc.markdown.len().saturating_add(recognized.text.len()) > 64 * 1024 * 1024 {
                    return Err(error("TIFF OCR text exceeds 64 MiB"));
                }
                doc.markdown.push('\n');
                doc.markdown.push_str(&recognized.text);
                if !doc.markdown.ends_with('\n') {
                    doc.markdown.push('\n');
                }
            }
        } else {
            remaining -= encoded.len();
            vision.push(VisionImage {
                mime,
                bytes: encoded.clone(),
            });
        }
        remaining -= encoded.len();
        doc.assets.push(Asset {
            name,
            bytes: encoded,
        });
    }
    // Move the original only after the borrowing decoder session has finished.
    doc.assets.insert(
        0,
        Asset {
            name: "original.tiff".into(),
            bytes,
        },
    );
    if local_ocr {
        ocr_metadata(&mut doc, cfg);
    }
    webp_notice(&mut doc, cfg);
    Ok((doc, vision))
}

fn image_document(path: &Path, asset_name: &str, bytes: Vec<u8>) -> Document {
    let title = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .replace(['\r', '\n'], " ");
    let alt = title
        .replace('\\', "\\\\")
        .replace('[', "\\[")
        .replace(']', "\\]");
    let mut doc = Document {
        markdown: format!("# {title}\n\n![{alt}](.markitai/assets/{asset_name})\n"),
        assets: vec![Asset {
            name: asset_name.into(),
            bytes,
        }],
        ..Default::default()
    };
    doc.metadata.insert("title".into(), title.into());
    doc
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageEncoder;
    use serde_json::json;

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        DynamicImage::new_rgba8(width, height)
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .unwrap();
        bytes
    }

    #[test]
    fn transparent_jpeg_has_white_background_and_respects_dimensions() {
        let cfg = config::normalize(&json!({"image":{"max_width":20,"max_height":20}})).unwrap();
        let original = DynamicImage::new_rgba8(100, 50);
        let (bytes, extension, mime) = encode(&original, &cfg, false).unwrap();
        let (output, _) = decode(&bytes).unwrap();
        assert_eq!((output.width(), output.height()), (20, 10));
        assert_eq!((extension, mime), ("jpg", "image/jpeg"));
        assert!(output.to_rgb8().get_pixel(0, 0).0.iter().all(|v| *v >= 250));
    }

    #[test]
    fn exif_orientation_is_applied_before_dimension_checks_and_encoding() {
        let mut bytes = Vec::new();
        let mut encoder = image::codecs::jpeg::JpegEncoder::new(&mut bytes);
        // Little-endian TIFF header, one SHORT orientation entry (rotate 90 CW).
        encoder
            .set_exif_metadata(vec![
                73, 73, 42, 0, 8, 0, 0, 0, 1, 0, 18, 1, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
            ])
            .unwrap();
        encoder.encode_image(&RgbImage::new(60, 90)).unwrap();
        let (image, _) = decode(&bytes).unwrap();
        assert_eq!((image.width(), image.height()), (90, 60));
    }

    #[test]
    fn filtering_and_duplicates_preserve_literal_examples_and_shared_references() {
        let big = png(100, 100);
        let mut doc = Document {
            markdown: "![a](.markitai/assets/a.png)\n![b](.markitai/assets/b.png)\n![tiny](.markitai/assets/tiny.png)\n`![example](.markitai/assets/tiny.png)`\n".into(),
            assets: vec![Asset{name:"a.png".into(),bytes:big.clone()},Asset{name:"b.png".into(),bytes:big.clone()},Asset{name:"tiny.png".into(),bytes:png(2,2)}],
            ..Default::default()
        };
        let cfg = config::normalize(&json!({"image":{"compress":false}})).unwrap();
        prepare_assets(&mut doc, &cfg);
        assert_eq!(doc.assets.len(), 1);
        assert_eq!(doc.assets[0].bytes, big);
        assert!(!doc.markdown.contains("![tiny]"));
        assert!(
            doc.markdown
                .contains("`![example](.markitai/assets/tiny.png)`")
        );
        assert_eq!(
            doc.markdown.matches(".markitai/assets/a.png.png").count(),
            2
        );
    }

    #[test]
    fn chained_asset_names_keep_distinct_images_and_duplicates_point_to_final_asset() {
        for compress in [false, true] {
            let first = png(100, 100);
            let second = png(120, 100);
            let mut doc = Document {
                markdown: "![first](.markitai/assets/a.png)\n![second](.markitai/assets/a.png.png)\n![duplicate](.markitai/assets/copy.png)\n`![literal](.markitai/assets/a.png)`\n".into(),
                assets: vec![
                    Asset { name: "a.png".into(), bytes: first.clone() },
                    Asset { name: "a.png.png".into(), bytes: second },
                    Asset { name: "copy.png".into(), bytes: first },
                ],
                ..Default::default()
            };
            let cfg =
                config::normalize(&json!({"image":{"compress":compress,"format":"png"}})).unwrap();
            prepare_assets(&mut doc, &cfg);
            assert_eq!(
                doc.markdown,
                "![first](.markitai/assets/a.png.png)\n![second](.markitai/assets/a.png.png.png)\n![duplicate](.markitai/assets/a.png.png)\n`![literal](.markitai/assets/a.png)`\n"
            );
            assert_eq!(doc.assets.len(), 2);
            assert_eq!(doc.assets[0].name, "a.png.png");
            assert_eq!(doc.assets[1].name, "a.png.png.png");
            assert_eq!(decode(&doc.assets[0].bytes).unwrap().0.width(), 100);
            assert_eq!(decode(&doc.assets[1].bytes).unwrap().0.width(), 120);
        }
    }

    #[test]
    fn filtering_an_original_name_does_not_remove_a_prepared_image_or_its_duplicate() {
        let big = png(100, 100);
        let mut doc = Document {
            markdown: "![kept](.markitai/assets/a.png)\n![duplicate](.markitai/assets/copy.png)\n![tiny](.markitai/assets/a.png.png)\n`![literal](.markitai/assets/a.png.png)`\n".into(),
            assets: vec![
                Asset { name: "a.png".into(), bytes: big.clone() },
                Asset { name: "copy.png".into(), bytes: big.clone() },
                Asset { name: "a.png.png".into(), bytes: png(2, 2) },
            ],
            ..Default::default()
        };
        let cfg = config::normalize(&json!({"image":{"compress":false}})).unwrap();
        prepare_assets(&mut doc, &cfg);
        assert_eq!(doc.assets.len(), 1);
        assert_eq!(doc.assets[0].name, "a.png.png");
        assert_eq!(doc.assets[0].bytes, big);
        assert!(doc.markdown.contains("![kept](.markitai/assets/a.png.png)"));
        assert!(
            doc.markdown
                .contains("![duplicate](.markitai/assets/a.png.png)")
        );
        assert!(!doc.markdown.contains("![tiny]"));
        assert!(
            doc.markdown
                .contains("`![literal](.markitai/assets/a.png.png)`")
        );
        assert!(doc.warnings.is_empty());
    }

    #[test]
    fn duplicate_source_names_keep_the_first_filter_decision() {
        let big = png(100, 100);
        let tiny = png(2, 2);
        for keep_first in [true, false] {
            let input =
                "![same](.markitai/assets/same.png)\n`![literal](.markitai/assets/same.png)`\n";
            let bytes = if keep_first {
                [big.clone(), tiny.clone()]
            } else {
                [tiny.clone(), big.clone()]
            };
            let mut doc = Document {
                markdown: input.into(),
                assets: bytes
                    .into_iter()
                    .map(|bytes| Asset {
                        name: "same.png".into(),
                        bytes,
                    })
                    .collect(),
                ..Default::default()
            };
            let cfg = config::normalize(&json!({"image":{"compress":false}})).unwrap();
            prepare_assets(&mut doc, &cfg);
            assert_eq!(doc.assets.len(), 1);
            assert_eq!(doc.assets[0].bytes, big);
            assert_eq!(
                doc.markdown
                    .contains("![same](.markitai/assets/same.png.png)"),
                keep_first
            );
            assert!(
                doc.markdown
                    .contains("`![literal](.markitai/assets/same.png)`")
            );
            if !keep_first {
                assert!(!doc.markdown.contains("![same]"));
            }
        }
    }

    #[test]
    fn retained_first_asset_keeps_its_reference_when_a_later_namesake_is_an_image() {
        for name in ["same.png", "same.bin"] {
            let retained = vec![0, 1, 2];
            let input = format!("![same](.markitai/assets/{name})\n");
            let mut doc = Document {
                markdown: input.clone(),
                assets: vec![
                    Asset {
                        name: name.into(),
                        bytes: retained.clone(),
                    },
                    Asset {
                        name: name.into(),
                        bytes: png(100, 100),
                    },
                ],
                ..Default::default()
            };
            let cfg = config::normalize(&json!({"image":{"compress":false}})).unwrap();
            prepare_assets(&mut doc, &cfg);
            assert_eq!(doc.markdown, input);
            assert_eq!(doc.assets.len(), 2);
            assert_eq!(doc.assets[0].name, name);
            assert_eq!(doc.assets[0].bytes, retained);
            assert_eq!(doc.assets[1].name, format!("{name}.png"));
            assert_eq!(decode(&doc.assets[1].bytes).unwrap().0.width(), 100);
            assert_eq!(doc.warnings.len(), usize::from(name.ends_with(".png")));
        }
    }

    #[test]
    fn malformed_embedded_assets_are_retained_with_warning() {
        let mut doc = Document {
            assets: vec![Asset {
                name: "bad.png".into(),
                bytes: vec![0, 1, 2],
            }],
            ..Default::default()
        };
        prepare_assets(&mut doc, &config::defaults());
        assert_eq!(doc.assets.len(), 1);
        assert_eq!(doc.warnings.len(), 1);
        assert!(decode(&[0, 1, 2]).is_err());
    }

    #[test]
    fn actual_raster_content_takes_precedence_over_mislabeled_extensions() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = png(80, 80);
        let cfg = config::normalize(&json!({"image":{"compress":false}})).unwrap();
        for extension in ["tiff", "heic", "svg", "avif"] {
            let path = dir.path().join(format!("renamed.{extension}"));
            std::fs::write(&path, &bytes).unwrap();
            let (doc, vision) = extract(&path, &cfg, false).unwrap();
            assert_eq!(doc.assets[0].bytes, bytes);
            assert_eq!(vision[0].mime, "image/png");
        }
    }

    #[test]
    fn disabling_compression_preserves_vision_bytes_and_transcoded_dimensions() {
        let dir = tempfile::tempdir().unwrap();
        let cfg =
            config::normalize(&json!({"image":{"compress":false,"max_width":10,"max_height":10}}))
                .unwrap();
        for (extension, format) in [("jpg", ImageFormat::Jpeg), ("bmp", ImageFormat::Bmp)] {
            let path = dir.path().join(format!("wide.{extension}"));
            DynamicImage::new_rgb8(200, 100)
                .save_with_format(&path, format)
                .unwrap();
            let (_, vision) = extract(&path, &cfg, false).unwrap();
            let (image, _) = decode(&vision[0].bytes).unwrap();
            assert_eq!((image.width(), image.height()), (200, 100));
            if extension == "jpg" {
                assert_eq!(vision[0].bytes, std::fs::read(&path).unwrap());
            }
        }
    }

    #[test]
    fn oversized_header_is_rejected_before_pixel_allocation() {
        let mut bytes = Vec::new();
        let mut encoder = png::Encoder::new(&mut bytes, 100_000, 100_000);
        encoder.set_color(png::ColorType::Rgb);
        let mut writer = encoder.write_header().unwrap();
        writer
            .write_chunk(
                png::chunk::IDAT,
                &[120, 156, 99, 96, 96, 96, 0, 0, 0, 4, 0, 1],
            )
            .unwrap();
        writer.finish().unwrap();
        let message = decode(&bytes).unwrap_err().to_string();
        assert!(
            message.contains("32 million") || message.contains("limit"),
            "{message}"
        );
    }

    #[test]
    fn svg_conversion_sends_2048_pixel_png_and_publishes_original_vector_asset() {
        use base64::Engine;
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::time::{Duration, Instant};

        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("vector.svg");
        let source = br#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50"><rect width="100" height="50" fill="red"/></svg>"#;
        std::fs::write(&input, source).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && Instant::now() < deadline =>
                    {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("SVG vision request not received: {error}"),
                }
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request_reader = bounded_fixture_io::Reader::new(
                &stream,
                std::time::Instant::now() + Duration::from_secs(5),
            );
            stream
                .set_write_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut bytes = Vec::new();
            let request = loop {
                let mut buffer = [0; 4096];
                let count = request_reader.read(&mut buffer).unwrap();
                assert_ne!(count, 0);
                bytes.extend_from_slice(&buffer[..count]);
                assert!(bytes.len() < 2 * 1024 * 1024);
                if let Some(end) = bytes.windows(4).position(|value| value == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..end]);
                    assert!(headers.starts_with("POST /v1/chat/completions "));
                    let length = headers
                        .lines()
                        .find_map(|line| {
                            let (key, value) = line.split_once(':')?;
                            key.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap();
                    if bytes.len() >= end + 4 + length {
                        break serde_json::from_slice::<Value>(&bytes[end + 4..end + 4 + length])
                            .unwrap();
                    }
                }
            };
            assert!(
                request["messages"][0]["content"]
                    .as_str()
                    .unwrap()
                    .contains("MARKITAI_VISION_JSON_V1")
            );
            let protected = request["messages"][1]["content"][0]["text"]
                .as_str()
                .unwrap();
            let response = json!({"choices":[{"message":{"content":json!({
                "cleaned_markdown":format!("{protected}\n\n# Read vector\n\nA red rectangle."),
                "frontmatter":{"description":"A red rectangle in SVG", "tags":["svg"]}
            }).to_string()}}]})
            .to_string();
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).unwrap();
            request
        });
        let converted = crate::convert(
            input.to_str().unwrap(),
            crate::ConvertOptions {
                output_dir: Some(directory.path().join("out")),
                config: Some(json!({
                    "cache":{"enabled":false},"history":{"record":false},
                    "prompts":{"dir":directory.path().join("prompts")},"log":{"dir":null},
                    "ocr":{"enabled":false},"screenshot":{"enabled":false},
                    "image":{"compress":false,"max_width":25,"alt_enabled":false,"desc_enabled":false},
                    "llm":{"enabled":true,"keep_base":true,"on_failure":"fail","router_settings":{"num_retries":0},
                        "model_list":[{"model_name":"local-svg-vision","litellm_params":{"model":"openai/test-svg","api_base":base,"api_key":"local-fixture-only"},"model_info":{"supports_vision":true}}]}
                })),
                ..Default::default()
            },
        );
        let request = server.join().unwrap();
        let output = converted.unwrap();
        let data_url = request["messages"][1]["content"]
            .as_array()
            .unwrap()
            .iter()
            .find_map(|block| block["image_url"]["url"].as_str())
            .unwrap();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data_url.strip_prefix("data:image/png;base64,").unwrap())
            .unwrap();
        let pixels = image::load_from_memory(&bytes).unwrap().to_rgba8();
        assert_eq!(pixels.dimensions(), (2048, 1024));
        assert_eq!(pixels.get_pixel(1024, 512).0, [255, 0, 0, 255]);
        assert_eq!(output.usage.requests, 1);
        assert_eq!(output.assets.len(), 1);
        assert_eq!(output.assets[0].extension().unwrap(), "svg");
        assert_eq!(std::fs::read(&output.assets[0]).unwrap(), source);
        assert_eq!(std::fs::read(&input).unwrap(), source);
        assert!(output.markdown.contains(".svg)"));
        assert!(output.llm_markdown.unwrap().contains("A red rectangle."));
        assert!(output.output_path.unwrap().is_file());
    }
    #[cfg(target_os = "macos")]
    #[test]
    fn embedded_heif_and_avif_are_real_pngs_with_primary_notice_and_alpha() {
        let mut doc = Document {
            markdown:
                "![primary](.markitai/assets/primary.heic) ![circle](.markitai/assets/circle.avif)"
                    .into(),
            assets: vec![
                Asset {
                    name: "primary.heic".into(),
                    bytes: include_bytes!("images/fixtures/heif/primary-second.heic").to_vec(),
                },
                Asset {
                    name: "circle.avif".into(),
                    bytes: include_bytes!("images/fixtures/heif/circle_custom_properties.avif")
                        .to_vec(),
                },
            ],
            ..Default::default()
        };
        let cfg=config::normalize(&json!({"image":{"compress":false,"filter":{"min_width":0,"min_height":0,"min_area":0}}})).unwrap();
        prepare_assets(&mut doc, &cfg);
        assert_eq!(doc.assets.len(), 2);
        assert!(
            doc.markdown.contains("primary.heic.png") && doc.markdown.contains("circle.avif.png")
        );
        assert!(
            doc.warnings
                .iter()
                .any(|w| w.contains("only primary image 2"))
        );
        let primary = image::load_from_memory(&doc.assets[0].bytes)
            .unwrap()
            .to_rgba8();
        assert!(primary.get_pixel(30, 20)[1] > 240);
        let circle = image::load_from_memory(&doc.assets[1].bytes)
            .unwrap()
            .to_rgba8();
        assert_eq!(circle.get_pixel(0, 0)[3], 0);
        for asset in &doc.assets {
            assert_eq!(image::guess_format(&asset.bytes).unwrap(), ImageFormat::Png);
        }
        let vision = prepare_vision(
            include_bytes!("images/fixtures/heif/quadrants-orientation6.heic"),
            "misleading.svg",
            &cfg,
        )
        .unwrap();
        assert_eq!(vision[0].mime, "image/png");
        let decoded = image::load_from_memory(&vision[0].bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (80, 120));
    }

    /// Every step a decoder takes gives what a `Cursor` over the same bytes
    /// gives: data, positions and errors (kind and message).
    #[test]
    fn image_bytes_reads_and_seeks_as_a_cursor_does() {
        use std::io::BufRead;
        let bytes: Vec<u8> = (0..1000u32).map(|i| (i * 7 % 251) as u8).collect();
        let mut ours = ImageBytes::new(&bytes);
        let mut cursor = Cursor::new(bytes.as_slice());
        let error = |e: std::io::Error| (e.kind(), e.to_string());
        for step in 0..14 {
            match step {
                0 | 6 | 11 => {
                    let (mut a, mut b) = ([0; 7], [0; 7]);
                    assert_eq!(ours.read(&mut a).unwrap(), cursor.read(&mut b).unwrap());
                    assert_eq!(a, b);
                }
                1 => assert_eq!(ours.fill_buf().unwrap(), cursor.fill_buf().unwrap()),
                2 => {
                    ours.consume(3);
                    cursor.consume(3);
                }
                3 => assert_eq!(
                    ours.seek(SeekFrom::Current(-5)).unwrap(),
                    cursor.seek(SeekFrom::Current(-5)).unwrap()
                ),
                4 => {
                    let (mut a, mut b) = (vec![9], vec![9]);
                    assert_eq!(
                        ours.read_to_end(&mut a).unwrap(),
                        cursor.read_to_end(&mut b).unwrap()
                    );
                    assert_eq!(a, b);
                    assert_eq!(a.len(), 1 + bytes.len() - 5);
                }
                5 => assert_eq!(
                    ours.seek(SeekFrom::End(-10)).unwrap(),
                    cursor.seek(SeekFrom::End(-10)).unwrap()
                ),
                7 => assert_eq!(
                    error(ours.seek(SeekFrom::Current(-2000)).unwrap_err()),
                    error(cursor.seek(SeekFrom::Current(-2000)).unwrap_err())
                ),
                8 => assert_eq!(
                    error(ours.seek(SeekFrom::End(-2000)).unwrap_err()),
                    error(cursor.seek(SeekFrom::End(-2000)).unwrap_err())
                ),
                9 => {
                    let (mut a, mut b) = ([0; 20], [0; 20]);
                    assert_eq!(
                        error(ours.read_exact(&mut a).unwrap_err()),
                        error(cursor.read_exact(&mut b).unwrap_err())
                    );
                }
                10 => assert_eq!(
                    ours.seek(SeekFrom::Start(5000)).unwrap(),
                    cursor.seek(SeekFrom::Start(5000)).unwrap()
                ),
                12 => assert_eq!(
                    ours.seek(SeekFrom::Start(2)).unwrap(),
                    cursor.seek(SeekFrom::Start(2)).unwrap()
                ),
                _ => {
                    let (mut a, mut b) = ([0; 64], [0; 64]);
                    ours.read_exact(&mut a).unwrap();
                    cursor.read_exact(&mut b).unwrap();
                    assert_eq!(a, b);
                }
            }
            assert_eq!(
                ours.stream_position().unwrap(),
                cursor.stream_position().unwrap(),
                "step {step}"
            );
        }
    }

    /// Every raster format decodes through `ImageBytes` to the pixels it
    /// decodes to from a `Cursor`.
    #[test]
    fn image_bytes_decode_every_format_as_a_cursor_does() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_fn(37, 23, |x, y| {
            Rgb([(x * 7) as u8, (y * 11) as u8, ((x + y) * 5) as u8])
        }));
        for format in [
            ImageFormat::Png,
            ImageFormat::Jpeg,
            ImageFormat::Gif,
            ImageFormat::Bmp,
            ImageFormat::Tiff,
            ImageFormat::WebP,
        ] {
            let mut encoded = Cursor::new(Vec::new());
            image.write_to(&mut encoded, format).unwrap();
            let encoded = encoded.into_inner();
            let through_cursor = ImageReader::new(Cursor::new(encoded.as_slice()))
                .with_guessed_format()
                .unwrap()
                .decode()
                .unwrap();
            let (decoded, found) = decode(&encoded).unwrap();
            assert!(
                matches!(found, DecodedFormat::Raster(f) if f == format),
                "{format:?}"
            );
            assert_eq!(decoded, through_cursor, "{format:?}");
            assert_eq!((decoded.width(), decoded.height()), (37, 23), "{format:?}");
        }
    }
}

#[cfg(test)]
mod bounded_fixture_io {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/bounded_read.rs"
    ));
}
