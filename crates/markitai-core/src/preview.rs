//! Previews of an original document for a side-by-side view: PDF page boxes,
//! rendered pages and positioned text lines, and raster images scaled to a
//! width. They use the conversion's own renderers, decoders and pixel limits.
use crate::{Error, Result, images, pdf_raster::PdfRasterSession};
use image::{DynamicImage, ImageEncoder, RgbImage};

/// Lines a page's text answer holds at most.
pub const MAX_LINES: usize = 5000;
/// The largest side the portable renderer draws.
const MAX_SIDE: f64 = u16::MAX as f64;

/// A page box: PDF points (1/72 inch) for a PDF page, pixels for an image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PageSize {
    pub width: f64,
    pub height: f64,
}

/// One line of a page's text and its box, `[x0, y0, x1, y1]` in points from
/// the page's top-left corner with `y` growing down.
#[derive(Clone, Debug, PartialEq)]
pub struct TextLine {
    pub text: String,
    pub bbox: [f64; 4],
}

/// One open PDF. Like the renderers, it stays on its caller's thread.
pub struct PdfPreview {
    session: PdfRasterSession,
}

fn failure(message: &str) -> Error {
    Error::Conversion(format!("Page preview: {message}"))
}

impl PdfPreview {
    pub fn open(bytes: &[u8]) -> Result<Self> {
        PdfRasterSession::open(bytes).map(|session| Self { session })
    }

    pub fn page_count(&self) -> usize {
        self.session.pages()
    }

    /// The visible box (CropBox ∩ MediaBox) of page `page` (1-based) after
    /// its rotation, in points.
    pub fn page_size(&self, page: usize) -> Result<PageSize> {
        let (width, height) = self.session.page_size(page)?;
        Ok(PageSize { width, height })
    }

    /// Page `page` drawn on white as a PNG `width` pixels wide, or narrower
    /// when that would exceed the renderers' pixel limits.
    pub fn render_png(&self, page: usize, width: u32) -> Result<Vec<u8>> {
        let size = self.page_size(page)?;
        // Pixels per point; the renderers round each side up.
        let pixels = (images::MAX_PIXELS as f64 / (size.width * size.height)).sqrt() * 0.995;
        let scale = (f64::from(width.max(1)) / size.width)
            .min(pixels)
            .min((MAX_SIDE - 1.) / size.width)
            .min((MAX_SIDE - 1.) / size.height);
        // A hair less, so rounding up the width lands on `width`, not past it.
        png(&self.session.render(page, scale * 72. * (1. - 1e-9))?)
    }
}

fn png(image: &RgbImage) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    image::codecs::png::PngEncoder::new_with_quality(
        &mut bytes,
        image::codecs::png::CompressionType::Fast,
        image::codecs::png::FilterType::Adaptive,
    )
    .write_image(
        image.as_raw(),
        image.width(),
        image.height(),
        image::ExtendedColorType::Rgb8,
    )
    .map_err(|error| failure(&error.to_string()))?;
    Ok(bytes)
}

/// The positioned text of page `page` of a PDF whose page box is `size`
/// (from [`PdfPreview::page_size`]): runs on one baseline that follow each
/// other within an em joined into a line, in reading order, at most
/// [`MAX_LINES`]. A scanned page without a text layer has none.
pub fn pdf_page_text(bytes: &[u8], page: usize, size: PageSize) -> Result<Vec<TextLine>> {
    pdf_page_layout(bytes, page, size).map(|layout| layout.lines)
}

/// Pictures a page's answer lists at most.
pub const MAX_IMAGES: usize = 500;
/// A picture smaller than this on either side (points) is a rule or a dot.
const MIN_IMAGE_SIDE: f64 = 4.;

/// What a page shows where: its text lines (as [`pdf_page_text`]) and the
/// boxes its pictures are drawn in, both `[x0, y0, x1, y1]` in points from
/// the page's top-left corner, the pictures in drawing order.
#[derive(Clone, Debug, PartialEq)]
pub struct PageLayout {
    pub lines: Vec<TextLine>,
    pub images: Vec<[f64; 4]>,
}

/// The text lines and picture boxes of page `page` of a PDF whose page box is `size`.
pub fn pdf_page_layout(bytes: &[u8], page: usize, size: PageSize) -> Result<PageLayout> {
    let number = u32::try_from(page).map_err(|_| failure("page number is out of range"))?;
    let items = pdf_inspector::extract_text_with_positions_mem_in_frame(
        bytes,
        Some(&std::collections::HashSet::from([number])),
        pdf_inspector::PositionFrame::Display,
    )
    .map_err(|error| failure(&error.to_string()))?;
    let lines = lines(
        items
            .iter()
            .filter(|item| {
                item.page == number
                    && matches!(item.item_type, pdf_inspector::types::ItemType::Text)
            })
            .map(|item| Run {
                text: &item.text,
                x: f64::from(item.x),
                y: f64::from(item.y),
                width: f64::from(item.width),
                height: f64::from(item.height),
                upright: item.rotation.abs() < 1. || item.rotation > 359.,
            }),
        size,
    );
    let images = items
        .iter()
        .filter(|item| {
            item.page == number && matches!(item.item_type, pdf_inspector::types::ItemType::Image)
        })
        .filter_map(|item| {
            let (x, y) = (f64::from(item.x), f64::from(item.y));
            let (width, height) = (f64::from(item.width), f64::from(item.height));
            image_box(x, y, width, height, size)
        })
        .take(MAX_IMAGES)
        .collect();
    Ok(PageLayout { lines, images })
}

/// A picture's box from its corner and size in y-up page points, in the
/// top-left frame and clipped to the page; none when it is not finite, is a
/// rule or a dot, or lies outside the page.
fn image_box(x: f64, y: f64, width: f64, height: f64, size: PageSize) -> Option<[f64; 4]> {
    if ![x, y, width, height].iter().all(|value| value.is_finite()) {
        return None;
    }
    let (left, right) = (x.min(x + width).max(0.), x.max(x + width).min(size.width));
    let (bottom, top) = (
        y.min(y + height).max(0.),
        y.max(y + height).min(size.height),
    );
    if right - left < MIN_IMAGE_SIDE || top - bottom < MIN_IMAGE_SIDE {
        return None;
    }
    Some([left, size.height - top, right, size.height - bottom])
}

/// A positioned text run: `y` is its baseline (bottom edge) from the page's
/// bottom, `height` its em size.
struct Run<'a> {
    text: &'a str,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    upright: bool,
}

struct Line {
    text: String,
    baseline: f64,
    em: f64,
    upright: bool,
    // y-up page coordinates
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
}

impl Line {
    fn new(run: &Run<'_>, text: &str) -> Self {
        Self {
            text: text.to_owned(),
            baseline: run.y,
            em: run.height,
            upright: run.upright,
            x0: run.x,
            y0: run.y,
            x1: run.x + run.width,
            y1: run.y + run.height,
        }
    }

    /// Whether `run` continues this line: upright text on the same baseline
    /// that starts no more than an em after it ends.
    fn continues(&self, run: &Run<'_>) -> bool {
        let em = self.em.max(run.height).max(1.);
        let gap = run.x - self.x1;
        self.upright
            && run.upright
            && (run.y - self.baseline).abs() <= em * 0.25
            && (-em * 0.5..=em).contains(&gap)
    }

    fn extend(&mut self, run: &Run<'_>, text: &str) {
        let em = self.em.max(run.height).max(1.);
        if run.x - self.x1 > em * 0.15 && !self.text.ends_with(' ') && !text.starts_with(' ') {
            self.text.push(' ');
        }
        self.text.push_str(text);
        self.em = self.em.max(run.height);
        self.x0 = self.x0.min(run.x);
        self.y0 = self.y0.min(run.y);
        self.x1 = self.x1.max(run.x + run.width);
        self.y1 = self.y1.max(run.y + run.height);
    }

    fn finish(self, size: PageSize) -> Option<TextLine> {
        let text = self.text.split_whitespace().collect::<Vec<_>>().join(" ");
        if text.is_empty() {
            return None;
        }
        let x = |value: f64| value.clamp(0., size.width);
        let y = |value: f64| (size.height - value).clamp(0., size.height);
        Some(TextLine {
            text,
            bbox: [x(self.x0), y(self.y1), x(self.x1), y(self.y0)],
        })
    }
}

fn lines<'a>(runs: impl Iterator<Item = Run<'a>>, size: PageSize) -> Vec<TextLine> {
    let mut result = Vec::new();
    let mut current: Option<Line> = None;
    for run in runs {
        let finite = [run.x, run.y, run.width, run.height]
            .iter()
            .all(|value| value.is_finite());
        if !finite || run.text.trim().is_empty() {
            continue;
        }
        let text = run.text.replace(char::is_control, " ");
        match &mut current {
            Some(line) if line.continues(&run) => line.extend(&run, &text),
            _ => {
                if let Some(done) = current.replace(Line::new(&run, &text))
                    && let Some(line) = done.finish(size)
                {
                    result.push(line);
                    if result.len() == MAX_LINES {
                        return result;
                    }
                }
            }
        }
    }
    if let Some(line) = current.and_then(|line| line.finish(size)) {
        result.push(line);
    }
    result
}

/// The displayed size, in pixels, of a raster image (png, jpeg, gif, webp,
/// bmp, tiff, heic/heif, avif): a TIFF's first page or a container's primary
/// image, after its orientation.
pub fn image_size(bytes: &[u8]) -> Result<PageSize> {
    let (width, height) = images::preview_size(bytes)?;
    Ok(PageSize {
        width: f64::from(width),
        height: f64::from(height),
    })
}

/// That image as a PNG at most `width` pixels wide, aspect kept.
pub fn image_png(bytes: &[u8], width: u32) -> Result<Vec<u8>> {
    let image = images::decode_preview(bytes)?;
    let image = if image.width() > width.max(1) {
        let (shown, wide) = (u64::from(image.height()), u64::from(image.width()));
        let height = ((shown * u64::from(width.max(1)) + wide / 2) / wide).max(1) as u32;
        image.resize_exact(width.max(1), height, image::imageops::FilterType::Triangle)
    } else {
        image
    };
    let mut bytes = std::io::Cursor::new(Vec::new());
    match image {
        DynamicImage::ImageRgb8(_) | DynamicImage::ImageRgba8(_) => {
            image.write_to(&mut bytes, image::ImageFormat::Png)
        }
        _ => {
            DynamicImage::ImageRgba8(image.to_rgba8()).write_to(&mut bytes, image::ImageFormat::Png)
        }
    }
    .map_err(|error| failure(&error.to_string()))?;
    Ok(bytes.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(text: &str, x: f64, y: f64, width: f64) -> Run<'_> {
        Run {
            text,
            x,
            y,
            width,
            height: 10.,
            upright: true,
        }
    }
    const LETTER: PageSize = PageSize {
        width: 612.,
        height: 792.,
    };

    #[test]
    fn runs_on_one_baseline_join_and_distant_cells_stay_apart() {
        let lines = lines(
            [
                run("Quarterly", 72., 700., 45.),
                run("report", 120., 700.5, 30.),
                // A table row: cells far apart on one baseline.
                run("Revenue", 72., 650., 40.),
                run("1,200", 300., 650., 25.),
                // The next line down.
                run("Costs", 72., 636., 30.),
                run("   ", 110., 636., 5.),
            ]
            .into_iter(),
            LETTER,
        );
        let texts: Vec<_> = lines.iter().map(|line| line.text.as_str()).collect();
        assert_eq!(texts, ["Quarterly report", "Revenue", "1,200", "Costs"]);
        // Boxes are in points from the top-left corner.
        assert_eq!(lines[0].bbox, [72., 792. - 710.5, 150., 92.]);
        assert_eq!(lines[3].bbox, [72., 792. - 646., 102., 156.]);
    }

    #[test]
    fn rotated_runs_are_lines_of_their_own_and_boxes_stay_on_the_page() {
        let mut stamp = run("DRAFT", 590., 300., 12.);
        stamp.upright = false;
        let lines = lines(
            [
                run("Body", 10., 300., 20.),
                stamp,
                run("edge", -5., 795., 30.),
            ]
            .into_iter(),
            LETTER,
        );
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[2].bbox, [0., 0., 25., 0.]);
    }

    #[test]
    fn a_page_holds_at_most_the_line_limit() {
        let texts: Vec<String> = (0..MAX_LINES + 10).map(|n| format!("line {n}")).collect();
        let runs = texts
            .iter()
            .enumerate()
            .map(|(n, text)| run(text, 0., n as f64 * 20., 10.));
        assert_eq!(lines(runs, LETTER).len(), MAX_LINES);
    }

    const MIXED: &[u8] = include_bytes!("pdf_raster/fixtures/mixed-native-scanned-blank.pdf");

    #[test]
    fn pdf_pages_report_their_visible_box_after_rotation() {
        let preview = PdfPreview::open(MIXED).unwrap();
        assert_eq!(preview.page_count(), 3);
        assert_eq!(preview.page_size(1).unwrap(), LETTER);
        assert!(preview.page_size(4).is_err());
        let turned =
            PdfPreview::open(include_bytes!("pdf_raster/fixtures/rotate-crop-90.pdf")).unwrap();
        assert_eq!(
            turned.page_size(1).unwrap(),
            PageSize {
                width: 800.,
                height: 600.
            }
        );
        let png = preview.render_png(1, 400).unwrap();
        let page = image::load_from_memory(&png).unwrap();
        assert_eq!((page.width(), page.height()), (400, 518));
        // The top-left corner is the page's white background.
        assert_eq!(page.to_rgb8().get_pixel(0, 0).0, [255, 255, 255]);
    }

    #[test]
    fn a_picture_box_is_measured_from_the_top_left_and_clipped_to_the_page() {
        let size = PageSize {
            width: 600.,
            height: 800.,
        };
        assert_eq!(
            image_box(40., 700., 120., 60., size),
            Some([40., 40., 160., 100.])
        );
        // A flipped placement has a negative size; one over the edge is clipped.
        assert_eq!(
            image_box(160., 760., -120., -60., size),
            Some([40., 40., 160., 100.])
        );
        assert_eq!(
            image_box(550., -20., 100., 40., size),
            Some([550., 780., 600., 800.])
        );
        assert_eq!(image_box(40., 700., 300., 1., size), None, "a rule");
        assert_eq!(image_box(700., 100., 50., 50., size), None, "off the page");
        assert_eq!(image_box(f64::NAN, 100., 50., 50., size), None);
    }

    #[test]
    fn pdf_page_text_is_positioned_from_the_top_left() {
        let lines = pdf_page_text(MIXED, 1, LETTER).unwrap();
        let heading = lines
            .iter()
            .find(|line| line.text.contains("NATIVE PAGE ONE"))
            .unwrap_or_else(|| panic!("{lines:?}"));
        let [x0, y0, x1, y1] = heading.bbox;
        assert!(0. <= x0 && x0 < x1 && x1 <= LETTER.width, "{heading:?}");
        assert!(
            0. <= y0 && y0 < y1 && y1 < LETTER.height / 2.,
            "{heading:?}"
        );
        let later = lines
            .iter()
            .position(|line| line.text.contains("Every paragraph remains"))
            .unwrap();
        assert!(lines[later].bbox[1] > y1);
        // The scanned page has no text layer of its own, only its picture.
        let scanned = pdf_page_layout(MIXED, 2, LETTER).unwrap();
        assert!(scanned.lines.is_empty());
        let [x0, y0, x1, y1] = *scanned
            .images
            .first()
            .unwrap_or_else(|| panic!("{scanned:?}"));
        assert!(
            x0 < x1 && y0 < y1 && x1 - x0 > LETTER.width / 2.,
            "{scanned:?}"
        );
        assert!(pdf_page_layout(MIXED, 1, LETTER).unwrap().lines == lines);
    }

    #[test]
    fn an_image_is_measured_from_its_header_and_scaled_only_down() {
        let mut encoded = std::io::Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(RgbImage::new(300, 120))
            .write_to(&mut encoded, image::ImageFormat::Png)
            .unwrap();
        let bytes = encoded.into_inner();
        assert_eq!(
            image_size(&bytes).unwrap(),
            PageSize {
                width: 300.,
                height: 120.
            }
        );
        let scaled = image::load_from_memory(&image_png(&bytes, 160).unwrap()).unwrap();
        assert_eq!((scaled.width(), scaled.height()), (160, 64));
        let kept = image::load_from_memory(&image_png(&bytes, 2400).unwrap()).unwrap();
        assert_eq!((kept.width(), kept.height()), (300, 120));
        assert!(image_size(b"not an image").is_err());
    }

    #[test]
    fn tiff_and_heic_previews_are_oriented_pngs() {
        let mut tiff = std::io::Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(RgbImage::new(200, 50))
            .write_to(&mut tiff, image::ImageFormat::Tiff)
            .unwrap();
        let tiff = tiff.into_inner();
        assert_eq!(
            image_size(&tiff).unwrap(),
            PageSize {
                width: 200.,
                height: 50.
            }
        );
        let page = image::load_from_memory(&image_png(&tiff, 160).unwrap()).unwrap();
        assert_eq!((page.width(), page.height()), (160, 40));
        // Turned a quarter for display: 120 by 80 coded, 80 by 120 shown.
        let heic = include_bytes!("images/fixtures/heif/quadrants-orientation6.heic");
        assert_eq!(
            image_size(heic).unwrap(),
            PageSize {
                width: 80.,
                height: 120.
            }
        );
        let png = image_png(heic, 2400).unwrap();
        assert_eq!(image::guess_format(&png).unwrap(), image::ImageFormat::Png);
        let shown = image::load_from_memory(&png).unwrap();
        assert_eq!((shown.width(), shown.height()), (80, 120));
    }
}
