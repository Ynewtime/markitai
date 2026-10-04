//! Preserve high-confidence vector charts as pixels, never inferred data values.
use super::geometry::{Chart, Frame};
use crate::{Asset, pdf_raster::PdfRasterSession};
use pdf_inspector::{TextItem, types::ItemType};

pub(super) struct Figure {
    pub bounds: Chart,
    pub name: String,
}

/// Only isolated horizontal bands can replace text without changing its order.
/// Require short numeric ticks and labels inside the complete clipping region;
/// refuse prose, crossing text, side columns and overlapping candidate regions.
pub(super) fn regions(items: &[TextItem], charts: &[Chart]) -> Vec<Chart> {
    if charts.len() > 4 || items.len() > 20_000 {
        return Vec::new();
    }
    let mut accepted = Vec::new();
    for &chart in charts {
        let mut numeric = 0;
        let mut labels = 0;
        let mut bytes = 0;
        let mut safe = true;
        for item in items
            .iter()
            .filter(|i| matches!(i.item_type, ItemType::Text) && !i.text.trim().is_empty())
        {
            let low = item.y - item.font_size * 0.25;
            let high = item.y + item.height;
            if high < chart.y0 || low > chart.y1 {
                continue;
            }
            if ![item.x, item.y, item.width, item.height, item.font_size]
                .iter()
                .all(|v| v.is_finite())
                || item.x < chart.x0 - 1.
                || item.x + item.width > chart.x1 + 1.
                || low < chart.y0 - 1.
                || high > chart.y1 + 1.
                || item.text.chars().count() > 80
                || item.text.split_whitespace().count() > 10
            {
                safe = false;
                break;
            }
            bytes += item.text.len();
            if item.text.trim().parse::<f64>().is_ok_and(f64::is_finite) {
                numeric += 1;
            } else if item.text.chars().any(char::is_alphabetic) {
                labels += 1;
            }
        }
        if safe && numeric >= 3 && labels >= 2 && bytes <= 2048 {
            accepted.push(chart);
        }
    }
    accepted.sort_by(|a, b| b.y1.total_cmp(&a.y1));
    if accepted.windows(2).any(|pair| pair[0].y0 <= pair[1].y1) {
        return Vec::new();
    }
    accepted
}

#[derive(Default)]
pub(super) struct Raster {
    session: Option<std::result::Result<PdfRasterSession, ()>>,
    pixels: u64,
}

impl Raster {
    /// One lazy document session, one raster per candidate page. No screenshot
    /// option, OCR, provider call or external executable is involved.
    pub(super) fn render(
        &mut self,
        bytes: &[u8],
        number: u32,
        frame: Frame,
        charts: &[Chart],
    ) -> std::result::Result<Vec<(Figure, Asset)>, &'static str> {
        let session = self
            .session
            .get_or_insert_with(|| PdfRasterSession::open(bytes).map_err(|_| ()));
        let session = session
            .as_ref()
            .map_err(|_| "page renderer could not open the PDF")?;
        let dpi = 144.;
        let (width, height) = session
            .dimensions(number as usize, dpi)
            .map_err(|_| "page geometry could not be rendered")?;
        let pixels = u64::from(width) * u64::from(height);
        if pixels > 8_000_000 || self.pixels + pixels > 64_000_000 || charts.len() > 4 {
            return Err("chart rendering exceeds the pixel budget");
        }
        // Upright frame only; tolerate the renderer's integer rounding.
        if width == 0
            || height == 0
            || frame.width <= 0.
            || frame.height <= 0.
            || (width as f64 / frame.width as f64 - height as f64 / frame.height as f64).abs()
                > 0.01
        {
            return Err("chart and rendered page coordinates disagree");
        }
        self.pixels += pixels;
        let image = session
            .render(number as usize, dpi)
            .map_err(|_| "chart page rendering failed")?;
        if image.dimensions() != (width, height) {
            return Err("rendered page dimensions changed");
        }
        let mut output = Vec::new();
        for (index, chart) in charts.iter().enumerate() {
            let (x, y, w, h) = crop(*chart, frame, width, height)
                .ok_or("chart bounds do not fit the rendered page")?;
            let pixels = image::imageops::crop_imm(&image, x, y, w, h).to_image();
            let mut png = Vec::new();
            {
                let mut encoder = png::Encoder::new(&mut png, w, h);
                encoder.set_color(png::ColorType::Rgb);
                encoder.set_depth(png::BitDepth::Eight);
                let mut writer = encoder
                    .write_header()
                    .map_err(|_| "chart PNG header failed")?;
                writer
                    .write_image_data(pixels.as_raw())
                    .map_err(|_| "chart PNG encoding failed")?;
            }
            let name = format!("pdf-chart-{number}-{}.png", index + 1);
            output.push((
                Figure {
                    bounds: *chart,
                    name: name.clone(),
                },
                Asset { name, bytes: png },
            ));
        }
        Ok(output)
    }
}

fn crop(chart: Chart, frame: Frame, width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
    if ![
        chart.x0,
        chart.x1,
        chart.y0,
        chart.y1,
        frame.width,
        frame.height,
    ]
    .iter()
    .all(|n| n.is_finite())
        || frame.width <= 0.
        || frame.height <= 0.
        || chart.x0 < 0.
        || chart.y0 < 0.
        || chart.x1 > frame.width
        || chart.y1 > frame.height
        || chart.x0 >= chart.x1
        || chart.y0 >= chart.y1
    {
        return None;
    }
    let sx = width as f64 / frame.width as f64;
    let sy = height as f64 / frame.height as f64;
    let x0 = (chart.x0 as f64 * sx).floor() as u32;
    let x1 = (chart.x1 as f64 * sx).ceil().min(width as f64) as u32;
    let y0 = ((frame.height - chart.y1) as f64 * sy).floor() as u32;
    let y1 = ((frame.height - chart.y0) as f64 * sy)
        .ceil()
        .min(height as f64) as u32;
    (x1 > x0 && y1 > y0).then_some((x0, y0, x1 - x0, y1 - y0))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crop_uses_top_origin_and_page_scale_without_clamping_invalid_regions() {
        let frame = Frame {
            x: 10.,
            y: 20.,
            width: 200.,
            height: 300.,
        };
        let chart = Chart {
            x0: 20.,
            y0: 40.,
            x1: 120.,
            y1: 140.,
        };
        assert_eq!(crop(chart, frame, 400, 600), Some((40, 320, 200, 200)));
        assert_eq!(crop(Chart { x0: -1., ..chart }, frame, 400, 600), None);
        assert_eq!(
            crop(
                Chart {
                    y1: f32::NAN,
                    ..chart
                },
                frame,
                400,
                600
            ),
            None
        );
    }
}
