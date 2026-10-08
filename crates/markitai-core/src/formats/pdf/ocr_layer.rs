//! Pages read from the OCR text layer a scanner laid, invisible, over the
//! page image (`pdf_inspector::PageOcrLayer`). The page reader accepts a
//! layer by its geometry: all of the page's text in render mode 3, on the
//! image that covers the page, at plausible sizes and density. Before the
//! layer's text stands for the page, this module checks it against what the
//! core's own inspection saw and, when the image can be read, against the
//! image itself: the layer's text must lie on the image's ink, and the
//! image's text-like ink under the layer's text. A layer that passes may
//! still misread the scan; the page's warning says so.

use super::{MAX_IMAGE_PIXELS, PageInspection, Space, color_space, decoded};
use lopdf::{Object, Stream};

/// Cells per side of the grids the reader reports (`PageOcrLayer`).
const GRID: usize = 64;

/// A pixel is ink when it is darker than this share of the paper's
/// luminance, the paper being the 90th percentile of the image's.
const INK_SHARE_OF_PAPER: f64 = 0.6;

/// An image whose paper is darker than this is not read for ink.
const MIN_PAPER_LUMINANCE: f64 = 80.0;

/// A cell holds text-like ink when at least the first share of its pixels
/// is ink and less than the second: more is a solid area — a photograph's
/// shadows, a filled shape, a scanner's black border — rather than print.
const INK_CELL_MIN_SHARE: f64 = 0.01;
const SOLID_CELL_MIN_SHARE: f64 = 0.5;

/// Pixels a cell needs before its share of ink means anything.
const MIN_CELL_SAMPLES: u32 = 4;

/// At least this share of the cells where the layer surely puts text must
/// lie within one cell of text-like ink...
///
/// Measured on 59 Tesseract 5.5 layers over 200–300 dpi gray, RGB JPEG and
/// bitonal renderings of printed pages: 0.978 at the least. Another page's
/// layer over a page of the same layout gave 0.867, words other than the
/// print's set on its lines 0.890: both are refused here, where the 0.8
/// first proposed let them pass.
const MIN_TEXT_ON_INK: f64 = 0.9;

/// ...and at least this share of the text-like ink cells within one cell
/// of where the layer's text may reach: 1.0 on every measured layer, while
/// a page's print with a short layer, a figure or handwriting beside it has
/// ink no text reaches.
const MIN_INK_UNDER_TEXT: f64 = 0.6;

/// The pixels of an image sampled at most; a larger image is sampled on a
/// coarser lattice (every second pixel of a 200 dpi letter page, every
/// third at 300 dpi), fine enough for strokes two or three pixels wide.
const MAX_SAMPLES: usize = 1_000_000;

/// Luminance histogram buckets per cell, of `256 / BUCKETS` levels each.
const BUCKETS: usize = 32;

/// How a page's OCR layer was checked against its image, for a page whose
/// text the layer is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LayerCheck {
    /// The layer's text lies on the image's text.
    Aligned,
    /// The image could not be read; why.
    Unverified(&'static str),
}

/// What comparing a layer with its image found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Alignment {
    Aligned,
    /// The shares of the layer's text on ink and of the ink under its text.
    Misaligned {
        text_on_ink: f64,
        ink_under_text: f64,
    },
    Unverified(&'static str),
}

/// Whether the core's own inspection of the page agrees that the layer is
/// all the page shows as text: it read the whole page, and the only
/// visibility signal it raised is the invisible rendering mode.
pub(super) fn inspection_agrees(inspection: &PageInspection) -> bool {
    !inspection.incomplete
        && inspection.signals.len() == 1
        && inspection.signals.contains(super::INVISIBLE_RENDERING)
}

/// The layer compared with the image under it.
pub(super) fn check(pdf: &lopdf::Document, layer: &pdf_inspector::PageOcrLayer) -> Alignment {
    let Some((id, matrix)) = layer.image else {
        return Alignment::Unverified("the page draws more than one image, or an inline one");
    };
    let Ok(stream) = pdf.get_object(id).and_then(Object::as_stream) else {
        return Alignment::Unverified("its image could not be read");
    };
    let luma = match Luma::decode(pdf, stream) {
        Ok(luma) => luma,
        Err(why) => return Alignment::Unverified(why),
    };
    match ink_cells(&luma, matrix, layer.page_box) {
        Some(ink) => compare(&layer.text_cells, &layer.reach_cells, &ink),
        None => Alignment::Unverified("the page image is too dark to tell its text from its paper"),
    }
}

type Grid = [u64; GRID];

fn count(grid: &Grid) -> u32 {
    grid.iter().map(|row| row.count_ones()).sum()
}

fn both(a: &Grid, b: &Grid) -> u32 {
    a.iter().zip(b).map(|(a, b)| (a & b).count_ones()).sum()
}

/// `grid` grown by one cell every way, diagonals included.
fn dilated(grid: &Grid) -> Grid {
    let spread = |row: u64| row | (row << 1) | (row >> 1);
    let mut out = [0; GRID];
    for (index, cell) in out.iter_mut().enumerate() {
        let below = index.checked_sub(1).map_or(0, |row| grid[row]);
        let above = grid.get(index + 1).copied().unwrap_or(0);
        *cell = spread(below | grid[index] | above);
    }
    out
}

fn compare(text: &Grid, reach: &Grid, ink: &Grid) -> Alignment {
    let text_cells = count(text);
    let ink_cells = count(ink);
    if text_cells == 0 || ink_cells == 0 {
        return Alignment::Misaligned {
            text_on_ink: 0.0,
            ink_under_text: 0.0,
        };
    }
    let text_on_ink = f64::from(both(text, &dilated(ink))) / f64::from(text_cells);
    let ink_under_text = f64::from(both(ink, &dilated(reach))) / f64::from(ink_cells);
    if text_on_ink >= MIN_TEXT_ON_INK && ink_under_text >= MIN_INK_UNDER_TEXT {
        Alignment::Aligned
    } else {
        Alignment::Misaligned {
            text_on_ink,
            ink_under_text,
        }
    }
}

/// The cells of the page grid holding text-like ink: the image drawn under
/// `matrix` (which places its unit square on the page, the image's first
/// row at the top), sampled into the 64 × 64 cells of `page_box`. `None`
/// when the image is too dark for its paper to be told from its ink.
fn ink_cells(luma: &Luma, matrix: [f64; 6], page_box: [f64; 4]) -> Option<Grid> {
    let [a, b, c, d, e, f] = matrix;
    let [x0, y0, x1, y1] = page_box;
    let (cell_w, cell_h) = ((x1 - x0) / GRID as f64, (y1 - y0) / GRID as f64);
    if !(cell_w > 0.0 && cell_h > 0.0) {
        return None;
    }
    let mut histograms = vec![[0u32; BUCKETS]; GRID * GRID];
    let (width, height) = (luma.width as f64, luma.height as f64);
    // A sample's cell, in cells from the page box's corner: the image's unit
    // square at (s, t) lands at (a s + c t + e, b s + d t + f), and moves
    // by a fixed step from one sampled column to the next.
    let (ds, grid) = (luma.step as f64 / width, GRID as f64);
    let columns = luma.columns();
    for (index, samples) in luma.pixels.chunks_exact(columns).enumerate() {
        let t = 1.0 - ((index * luma.step) as f64 + 0.5) / height;
        let s = 0.5 / width;
        let mut col = (a * s + c * t + e - x0) / cell_w;
        let mut line = (b * s + d * t + f - y0) / cell_h;
        let (col_step, line_step) = (a * ds / cell_w, b * ds / cell_h);
        for &level in samples {
            if (0.0..grid).contains(&col) && (0.0..grid).contains(&line) {
                let cell = line as usize * GRID + col as usize;
                histograms[cell][usize::from(level) * BUCKETS / 256] += 1;
            }
            col += col_step;
            line += line_step;
        }
    }
    let mut total = [0u64; BUCKETS];
    for histogram in &histograms {
        for (sum, count) in total.iter_mut().zip(histogram) {
            *sum += u64::from(*count);
        }
    }
    let samples: u64 = total.iter().sum();
    if samples == 0 {
        return None;
    }
    // The paper: the level below which nine tenths of the samples lie.
    let mut seen = 0;
    let paper_bucket = total
        .iter()
        .position(|&count| {
            seen += count;
            seen * 10 >= samples * 9
        })
        .unwrap_or(BUCKETS - 1);
    let bucket_levels = 256 / BUCKETS;
    let paper = ((paper_bucket + 1) * bucket_levels) as f64;
    if paper < MIN_PAPER_LUMINANCE {
        return None;
    }
    // Buckets wholly below the ink threshold.
    let ink_buckets = (INK_SHARE_OF_PAPER * paper) as usize / bucket_levels;
    let mut ink = [0u64; GRID];
    for (cell, histogram) in histograms.iter().enumerate() {
        let samples: u32 = histogram.iter().sum();
        if samples < MIN_CELL_SAMPLES {
            continue;
        }
        let dark: u32 = histogram[..ink_buckets.min(BUCKETS)].iter().sum();
        let share = f64::from(dark) / f64::from(samples);
        if (INK_CELL_MIN_SHARE..SOLID_CELL_MIN_SHARE).contains(&share) {
            ink[cell / GRID] |= 1 << (cell % GRID);
        }
    }
    Some(ink)
}

/// An image's luminance, 8 bits a sample, sampled every `step` pixels
/// along its rows and columns (see `MAX_SAMPLES`), its first row at the
/// top: `pixels` holds rows of `columns()` samples.
struct Luma {
    width: usize,
    height: usize,
    step: usize,
    pixels: Vec<u8>,
}

/// How an image's samples are read for their luminance.
enum Samples {
    Gray {
        invert: bool,
    },
    /// A stencil mask: a sample of 0 paints, unless `/Decode [1 0]`.
    Mask {
        paint: u8,
    },
    Rgb,
    Cmyk,
    /// RGB palette entries.
    Indexed(Vec<u8>),
}

/// The sampling stride for an image of `width` × `height` pixels.
fn sample_step(width: usize, height: usize) -> usize {
    (((width * height) as f64 / MAX_SAMPLES as f64).sqrt().ceil() as usize).max(1)
}

impl Luma {
    /// Samples per stored row.
    fn columns(&self) -> usize {
        self.width.div_ceil(self.step)
    }

    /// The luminance of an image XObject: JPEG, or samples behind the
    /// general-purpose filters, of 1 to 8 (or 16) bits in a gray, RGB,
    /// CMYK or indexed space, or a stencil mask. Soft masks and colour
    /// management are not applied: the check needs where the ink lies,
    /// not its colour. The error says why an image is not read.
    fn decode(pdf: &lopdf::Document, stream: &Stream) -> Result<Self, &'static str> {
        let dict = &stream.dict;
        let dimension = |key: &[u8]| {
            dict.get(key)
                .and_then(Object::as_i64)
                .ok()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
        };
        let (Some(width), Some(height)) = (dimension(b"Width"), dimension(b"Height")) else {
            return Err("its image has no valid size");
        };
        if width
            .checked_mul(height)
            .is_none_or(|pixels| pixels > MAX_IMAGE_PIXELS)
        {
            return Err("its image is past the pixel limit");
        }
        let filters: Vec<&[u8]> = if dict.has(b"Filter") {
            stream
                .filters()
                .map_err(|_| "its image's filters could not be read")?
        } else {
            Vec::new()
        };
        for filter in &filters {
            match *filter {
                b"JBIG2Decode" => {
                    return Err("its image is JBIG2-compressed, which is not decoded");
                }
                b"CCITTFaxDecode" | b"CCF" => {
                    return Err("its image is CCITT fax-compressed, which is not decoded");
                }
                b"JPXDecode" => return Err("its image is JPEG 2000, which is not decoded"),
                _ => {}
            }
        }
        if filters.last() == Some(&b"DCTDecode".as_slice())
            || filters.last() == Some(&b"DCT".as_slice())
        {
            if filters.len() != 1 {
                return Err("its JPEG image is filtered further, which is not decoded");
            }
            return Self::jpeg(&stream.content);
        }
        if filters.iter().any(|filter| {
            !matches!(
                *filter,
                b"FlateDecode"
                    | b"Fl"
                    | b"LZWDecode"
                    | b"LZW"
                    | b"RunLengthDecode"
                    | b"RL"
                    | b"ASCII85Decode"
                    | b"A85"
                    | b"ASCIIHexDecode"
                    | b"AHx"
            )
        }) {
            return Err("its image's filter is not decoded");
        }
        let mask = dict
            .get(b"ImageMask")
            .and_then(Object::as_bool)
            .unwrap_or(false);
        let decode = dict
            .get(b"Decode")
            .ok()
            .and_then(|value| pdf.dereference(value).ok())
            .and_then(|(_, value)| value.as_array().ok())
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.as_float().unwrap_or(f32::NAN))
                    .collect::<Vec<_>>()
            });
        let inverted = match decode.as_deref() {
            None => false,
            Some([low, high]) if *low == 0.0 && *high == 1.0 => false,
            Some([low, high]) if *low == 1.0 && *high == 0.0 => true,
            Some(_) => return Err("its image remaps its samples"),
        };
        let (samples, components, bits) = if mask {
            (
                Samples::Mask {
                    paint: u8::from(inverted),
                },
                1,
                1,
            )
        } else {
            let bits = dict
                .get(b"BitsPerComponent")
                .and_then(Object::as_i64)
                .unwrap_or(0);
            let space = dict
                .get(b"ColorSpace")
                .map_err(|_| "its image has no colour space")?;
            let samples = if cmyk(pdf, space) {
                Samples::Cmyk
            } else {
                match color_space(pdf, space, false) {
                    Ok(Space::Gray) => Samples::Gray { invert: inverted },
                    Ok(Space::Rgb) => Samples::Rgb,
                    Ok(Space::Indexed(palette)) => Samples::Indexed(palette),
                    Err(_) => return Err("its image's colour space is not read"),
                }
            };
            if decode.is_some() && !matches!(samples, Samples::Gray { .. }) {
                return Err("its image remaps its samples");
            }
            let components = match samples {
                Samples::Rgb => 3,
                Samples::Cmyk => 4,
                _ => 1,
            };
            let supported = match samples {
                Samples::Indexed(_) => matches!(bits, 1 | 2 | 4 | 8),
                Samples::Gray { .. } => matches!(bits, 1 | 2 | 4 | 8 | 16),
                _ => matches!(bits, 8 | 16),
            };
            if !supported {
                return Err("its image's sample depth is not read");
            }
            (samples, components, bits as usize)
        };
        let data = decoded(stream).map_err(|_| "its image data could not be decompressed")?;
        let row = (width * components * bits).div_ceil(8);
        if data.len() < row * height {
            return Err("its image data is shorter than its size");
        }
        let step = sample_step(width, height);
        let mut pixels = Vec::with_capacity(width.div_ceil(step) * height.div_ceil(step));
        let max = ((1u32 << bits.min(8)) - 1) as f64;
        for line in data.chunks_exact(row).take(height).step_by(step) {
            for column in (0..width).step_by(step) {
                let sample = |component: usize| -> u8 {
                    let index = column * components + component;
                    match bits {
                        16 => line[index * 2],
                        8 => line[index],
                        _ => {
                            let bit = index * bits;
                            let byte = line[bit / 8];
                            let shift = 8 - bits - bit % 8;
                            (byte >> shift) & ((1u8 << bits) - 1)
                        }
                    }
                };
                let scaled = |value: u8| -> f64 {
                    if bits >= 8 {
                        f64::from(value)
                    } else {
                        f64::from(value) * 255.0 / max
                    }
                };
                let level = match &samples {
                    Samples::Gray { invert } => {
                        let level = scaled(sample(0));
                        if *invert { 255.0 - level } else { level }
                    }
                    Samples::Mask { paint } => {
                        if sample(0) == *paint {
                            0.0
                        } else {
                            255.0
                        }
                    }
                    Samples::Rgb => rgb_luminance(
                        f64::from(sample(0)),
                        f64::from(sample(1)),
                        f64::from(sample(2)),
                    ),
                    Samples::Cmyk => {
                        let k = f64::from(sample(3));
                        let channel = |value: u8| (255.0 - f64::from(value) - k).max(0.0);
                        rgb_luminance(channel(sample(0)), channel(sample(1)), channel(sample(2)))
                    }
                    Samples::Indexed(palette) => {
                        let entries = palette.len() / 3;
                        let index = usize::from(sample(0)).min(entries.saturating_sub(1)) * 3;
                        match palette.get(index..index + 3) {
                            Some([r, g, b]) => {
                                rgb_luminance(f64::from(*r), f64::from(*g), f64::from(*b))
                            }
                            _ => 255.0,
                        }
                    }
                };
                pixels.push(level.round().clamp(0.0, 255.0) as u8);
            }
        }
        Ok(Self {
            width,
            height,
            step,
            pixels,
        })
    }

    fn jpeg(bytes: &[u8]) -> Result<Self, &'static str> {
        use image::ImageDecoder;
        let mut reader = image::ImageReader::with_format(
            crate::images::ImageBytes::new(bytes),
            image::ImageFormat::Jpeg,
        );
        // At most four bytes a pixel (CMYK) within the pixel limit.
        let mut limits = image::Limits::default();
        limits.max_alloc = Some(4 * MAX_IMAGE_PIXELS as u64);
        reader.limits(limits);
        let decoder = reader
            .into_decoder()
            .map_err(|_| "its JPEG image could not be decoded")?;
        // The size in the JPEG's own header, which the dictionary's may not
        // match, is checked before the pixels are decoded.
        let (width, height) = decoder.dimensions();
        let (width, height) = (width as usize, height as usize);
        if width == 0 || height == 0 {
            return Err("its JPEG image has no valid size");
        }
        if width * height > MAX_IMAGE_PIXELS {
            return Err("its image is past the pixel limit");
        }
        let image = image::DynamicImage::from_decoder(decoder)
            .map_err(|_| "its JPEG image could not be decoded")?
            .into_luma8();
        let step = sample_step(width, height);
        let full = image.into_raw();
        let pixels = if step == 1 {
            full
        } else {
            full.chunks_exact(width)
                .step_by(step)
                .flat_map(|row| row.iter().step_by(step).copied())
                .collect()
        };
        Ok(Self {
            width,
            height,
            step,
            pixels,
        })
    }
}

fn rgb_luminance(r: f64, g: f64, b: f64) -> f64 {
    0.299 * r + 0.587 * g + 0.114 * b
}

/// Whether `space` is CMYK: the device space, or an ICC profile of four
/// components.
fn cmyk(pdf: &lopdf::Document, space: &Object) -> bool {
    let Ok((_, space)) = pdf.dereference(space) else {
        return false;
    };
    match space {
        Object::Name(name) => name == b"DeviceCMYK",
        Object::Array(items) => match items.as_slice() {
            [Object::Name(family), profile] if family == b"ICCBased" => {
                pdf.dereference(profile)
                    .ok()
                    .and_then(|(_, profile)| profile.as_stream().ok())
                    .and_then(|profile| profile.dict.get(b"N").and_then(Object::as_i64).ok())
                    == Some(4)
            }
            _ => false,
        },
        _ => false,
    }
}

/// The document's producer as a warning names it: `/Producer`, or else
/// `/Creator`, of the information dictionary, on one line, at most 60
/// characters.
pub(super) fn producer(pdf: &lopdf::Document) -> Option<String> {
    let info = pdf
        .trailer
        .get(b"Info")
        .and_then(Object::as_reference)
        .and_then(|id| pdf.get_dictionary(id))
        .ok()?;
    [b"Producer".as_slice(), b"Creator"].iter().find_map(|key| {
        let text = info.get(key).and_then(lopdf::decode_text_string).ok()?;
        let text: String = text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .filter(|ch| !ch.is_control())
            .collect();
        let text = text.trim();
        (!text.is_empty()).then(|| {
            if text.chars().count() > 60 {
                format!("{}…", text.chars().take(59).collect::<String>())
            } else {
                text.to_string()
            }
        })
    })
}

/// Page numbers as ranges: `1-3, 5, 7-9`.
pub(super) fn ranges(pages: &[usize]) -> String {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &page in pages {
        match runs.last_mut() {
            Some((_, end)) if *end + 1 == page => *end = page,
            _ => runs.push((page, page)),
        }
    }
    runs.iter()
        .map(|&(start, end)| {
            if start == end {
                start.to_string()
            } else {
                format!("{start}-{end}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The warning for the pages whose text is their OCR layer, checked alike.
pub(super) fn warning(pages: &[usize], check: LayerCheck, producer: Option<&str>) -> String {
    let (label, image) = if pages.len() == 1 {
        ("page", "the page image")
    } else {
        ("pages", "each page image")
    };
    let laid = match producer {
        Some(producer) => format!("the invisible OCR text layer laid over {image} ({producer})"),
        None => format!("the invisible OCR text layer laid over {image}"),
    };
    let checked = match check {
        LayerCheck::Aligned => {
            "it lines up with the text in the image, but Markitai did not recognize the text itself"
                .to_string()
        }
        LayerCheck::Unverified(why) => format!("it could not be checked against the image ({why})"),
    };
    format!(
        "PDF {label} {}: the text was read from {laid}; {checked}, so recognition errors in the layer are kept.",
        ranges(pages)
    )
}

/// The warning for a page whose layer was not used because it does not
/// line up with the text in its image.
pub(super) fn misaligned(page: u32, text_on_ink: f64, ink_under_text: f64) -> String {
    format!(
        "PDF page {page}: an invisible OCR text layer lies over the page image, but it does not line up with the text in the image ({:.0}% of its text lies on print, {:.0}% of the print lies under its text), so it was not used.",
        text_on_ink * 100.0,
        ink_under_text * 100.0
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(cells: &[(usize, usize)]) -> Grid {
        let mut grid = [0; GRID];
        for &(row, col) in cells {
            grid[row] |= 1 << col;
        }
        grid
    }

    #[test]
    fn text_must_lie_on_ink_and_ink_under_text() {
        let line: Vec<(usize, usize)> = (8..40).map(|col| (50, col)).collect();
        let text = grid(&line);
        // Ink one row below the text: within a cell of it.
        let ink = grid(&line.iter().map(|&(_, col)| (49, col)).collect::<Vec<_>>());
        assert_eq!(compare(&text, &text, &ink), Alignment::Aligned);
        // Ink elsewhere on the page only.
        let elsewhere = grid(&(8..40).map(|col| (10, col)).collect::<Vec<_>>());
        assert!(matches!(
            compare(&text, &text, &elsewhere),
            Alignment::Misaligned { text_on_ink, .. } if text_on_ink == 0.0
        ));
        // Text on a tenth of the ink: a short layer over a full page.
        let mut page_ink = Vec::new();
        for row in 10..60 {
            page_ink.extend((8..40).map(|col| (row, col)));
        }
        let page_ink = grid(&page_ink);
        assert!(matches!(
            compare(&text, &text, &page_ink),
            Alignment::Misaligned { text_on_ink, ink_under_text }
                if text_on_ink == 1.0 && ink_under_text < 0.1
        ));
        // No ink at all.
        assert!(matches!(
            compare(&text, &text, &[0; GRID]),
            Alignment::Misaligned { .. }
        ));
    }

    #[test]
    fn page_ranges_read_as_a_person_writes_them() {
        assert_eq!(ranges(&[1, 2, 3, 5, 7, 8, 9]), "1-3, 5, 7-9");
        assert_eq!(ranges(&[4]), "4");
        assert_eq!(
            warning(&[2], LayerCheck::Aligned, Some("Tesseract 5.5.3")),
            "PDF page 2: the text was read from the invisible OCR text layer laid over the page image (Tesseract 5.5.3); it lines up with the text in the image, but Markitai did not recognize the text itself, so recognition errors in the layer are kept."
        );
        assert_eq!(
            warning(
                &[1, 2],
                LayerCheck::Unverified("its image is JBIG2-compressed, which is not decoded"),
                None
            ),
            "PDF pages 1-2: the text was read from the invisible OCR text layer laid over each page image; it could not be checked against the image (its image is JBIG2-compressed, which is not decoded), so recognition errors in the layer are kept."
        );
    }

    #[test]
    fn ink_is_read_from_the_cells_the_image_lies_on() {
        // A white 640 × 64 image with its first row dark, drawn over the
        // page: a cell is 10 × 1 pixels, and those of the top row are all
        // ink — solid, not print.
        let mut pixels = vec![255u8; 64 * 640];
        for pixel in &mut pixels[..640] {
            *pixel = 0;
        }
        let luma = Luma {
            step: 1,
            width: 640,
            height: 64,
            pixels,
        };
        let ink = ink_cells(
            &luma,
            [640.0, 0.0, 0.0, 640.0, 0.0, 0.0],
            [0.0, 0.0, 640.0, 640.0],
        )
        .unwrap();
        assert_eq!(count(&ink), 0);
        // At 640 × 640 a cell is 10 × 10 pixels: one dark row in ten is
        // print.
        let mut pixels = vec![255u8; 640 * 640];
        for pixel in &mut pixels[..640] {
            *pixel = 0;
        }
        let luma = Luma {
            step: 1,
            width: 640,
            height: 640,
            pixels,
        };
        let ink = ink_cells(
            &luma,
            [640.0, 0.0, 0.0, 640.0, 0.0, 0.0],
            [0.0, 0.0, 640.0, 640.0],
        )
        .unwrap();
        assert_eq!(
            ink[GRID - 1],
            u64::MAX,
            "the image's first row is the page's top"
        );
        assert_eq!(count(&ink), 64);
        // Turned a quarter: the first row runs down the page's left edge.
        let turned = ink_cells(
            &luma,
            [0.0, 640.0, -640.0, 0.0, 640.0, 0.0],
            [0.0, 0.0, 640.0, 640.0],
        )
        .unwrap();
        assert!(turned.iter().all(|row| *row == 1), "{turned:?}");
        // A black image has no paper.
        let dark = Luma {
            step: 1,
            width: 8,
            height: 8,
            pixels: vec![20; 64],
        };
        assert_eq!(
            ink_cells(
                &dark,
                [640.0, 0.0, 0.0, 640.0, 0.0, 0.0],
                [0.0, 0.0, 640.0, 640.0]
            ),
            None
        );
    }

    fn image(dict: lopdf::Dictionary, data: Vec<u8>) -> Stream {
        let mut stream = Stream::new(dict, data);
        stream.allows_compression = false;
        stream
    }

    fn decoded_levels(stream: &Stream) -> Result<Vec<u8>, &'static str> {
        Luma::decode(&lopdf::Document::with_version("1.5"), stream).map(|luma| luma.pixels)
    }

    #[test]
    fn samples_of_every_read_depth_and_space_become_luminance() {
        use lopdf::dictionary;
        // One-bit gray, 0 black: a row of 10 pixels in two bytes.
        let gray = |decode: Option<Vec<Object>>| {
            let mut dict = dictionary! {
                "Width" => 10, "Height" => 1, "ColorSpace" => "DeviceGray",
                "BitsPerComponent" => 1
            };
            if let Some(decode) = decode {
                dict.set("Decode", decode);
            }
            image(dict, vec![0b1010_0000, 0b1100_0000])
        };
        assert_eq!(
            decoded_levels(&gray(None)).unwrap(),
            [255, 0, 255, 0, 0, 0, 0, 0, 255, 255]
        );
        // `/Decode [1 0]` inverts.
        assert_eq!(
            decoded_levels(&gray(Some(vec![1.into(), 0.into()]))).unwrap(),
            [0, 255, 0, 255, 255, 255, 255, 255, 0, 0]
        );
        assert_eq!(
            decoded_levels(&gray(Some(vec![0.5.into(), 1.into()]))),
            Err("its image remaps its samples")
        );
        // A stencil mask paints where its samples are 0.
        let mask = image(
            dictionary! { "Width" => 4, "Height" => 1, "ImageMask" => true },
            vec![0b0110_0000],
        );
        assert_eq!(decoded_levels(&mask).unwrap(), [0, 255, 255, 0]);
        // Four-bit indexed through an RGB palette.
        let indexed = image(
            dictionary! {
                "Width" => 2, "Height" => 1, "BitsPerComponent" => 4,
                "ColorSpace" => vec![
                    "Indexed".into(), "DeviceRGB".into(), 1.into(),
                    Object::String(vec![255, 255, 255, 0, 0, 0], lopdf::StringFormat::Hexadecimal)
                ]
            },
            vec![0x01],
        );
        assert_eq!(decoded_levels(&indexed).unwrap(), [255, 0]);
        // CMYK: no ink is white, full black is black.
        let cmyk = image(
            dictionary! {
                "Width" => 2, "Height" => 1, "BitsPerComponent" => 8, "ColorSpace" => "DeviceCMYK"
            },
            vec![0, 0, 0, 0, 0, 0, 0, 255],
        );
        assert_eq!(decoded_levels(&cmyk).unwrap(), [255, 0]);
        // Sixteen-bit RGB reads its high bytes.
        let rgb16 = image(
            dictionary! {
                "Width" => 1, "Height" => 1, "BitsPerComponent" => 16, "ColorSpace" => "DeviceRGB"
            },
            vec![255, 0, 255, 0, 255, 0],
        );
        assert_eq!(decoded_levels(&rgb16).unwrap(), [255]);
        // Data shorter than the size says.
        let short = image(
            dictionary! {
                "Width" => 4, "Height" => 4, "BitsPerComponent" => 8, "ColorSpace" => "DeviceGray"
            },
            vec![0; 15],
        );
        assert_eq!(
            decoded_levels(&short),
            Err("its image data is shorter than its size")
        );
        // Deflated samples are inflated first.
        let mut deflated = Stream::new(
            dictionary! {
                "Width" => 2, "Height" => 1, "BitsPerComponent" => 8, "ColorSpace" => "DeviceGray"
            },
            vec![10, 200],
        );
        deflated.compress().unwrap();
        assert_eq!(decoded_levels(&deflated).unwrap(), [10, 200]);
    }

    #[test]
    fn jpeg_images_decode_and_other_codecs_are_named() {
        use lopdf::dictionary;
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 95)
            .encode(&[255u8; 16 * 8], 16, 8, image::ExtendedColorType::L8)
            .unwrap();
        let stream = image(
            dictionary! {
                "Width" => 16, "Height" => 8, "BitsPerComponent" => 8,
                "ColorSpace" => "DeviceGray", "Filter" => "DCTDecode"
            },
            jpeg,
        );
        let pixels = decoded_levels(&stream).unwrap();
        assert_eq!(pixels.len(), 16 * 8);
        assert!(pixels.iter().all(|&level| level > 240));
        for (filter, why) in [
            (
                "JBIG2Decode",
                "its image is JBIG2-compressed, which is not decoded",
            ),
            (
                "CCITTFaxDecode",
                "its image is CCITT fax-compressed, which is not decoded",
            ),
            ("JPXDecode", "its image is JPEG 2000, which is not decoded"),
        ] {
            let stream = image(
                dictionary! {
                    "Width" => 16, "Height" => 8, "BitsPerComponent" => 1,
                    "ColorSpace" => "DeviceGray", "Filter" => filter
                },
                vec![0; 16],
            );
            assert_eq!(decoded_levels(&stream), Err(why));
        }
        // A JPEG's own header gives its size, whatever the dictionary says:
        // one past the pixel limit is refused before it is decoded.
        let mut large = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut large, 95)
            .encode(&[255u8; 16 * 8], 16, 8, image::ExtendedColorType::L8)
            .unwrap();
        let frame = large
            .windows(2)
            .position(|marker| marker == [0xFF, 0xC0])
            .unwrap();
        // Frame header: marker, length, precision, then height and width.
        large[frame + 5..frame + 9].copy_from_slice(&[0x1F, 0x40, 0x1F, 0x40]);
        let stream = image(
            dictionary! {
                "Width" => 16, "Height" => 8, "BitsPerComponent" => 8,
                "ColorSpace" => "DeviceGray", "Filter" => "DCTDecode"
            },
            large,
        );
        assert_eq!(
            decoded_levels(&stream),
            Err("its image is past the pixel limit")
        );
        let lab = image(
            dictionary! {
                "Width" => 1, "Height" => 1, "BitsPerComponent" => 8,
                "ColorSpace" => vec!["Lab".into(), dictionary! {}.into()]
            },
            vec![0, 0, 0],
        );
        assert_eq!(
            decoded_levels(&lab),
            Err("its image's colour space is not read")
        );
    }
}
