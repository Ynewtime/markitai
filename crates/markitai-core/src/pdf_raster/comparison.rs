//! CoreGraphics/hayro comparison over a list of PDFs, for
//! `docs/validation/drivers/portable-raster-r1/compare.py`. Ignored by
//! default; it needs macOS, the `portable-media` feature and these variables:
//!
//! - `MARKITAI_RASTER_COMPARE_INPUTS`: a file listing one PDF path per line.
//! - `MARKITAI_RASTER_COMPARE_OUT`: a new directory for `documents.jsonl`,
//!   `pages.jsonl` and side-by-side PNGs of dissimilar pages (`pages/`).
//! - `MARKITAI_RASTER_COMPARE_DPI` (150), `MARKITAI_RASTER_COMPARE_PAGES`
//!   (pages per document, 40) and `MARKITAI_RASTER_COMPARE_PICTURE_STEP`
//!   (pixel step of the saved pictures, 2; 1 keeps full resolution).
//!
//! Each page is drawn by both backends at one DPI. Similarity is measured on
//! luma: SSIM over 8x8 blocks (all blocks, and blocks with ink in either
//! image), intersection over union of ink (luma below 160) in 3x3 cells, so a
//! sub-pixel glyph offset is not counted as a difference, and the mean
//! absolute channel difference.

use super::{Backend, PdfRasterSession};
use image::RgbImage;
use serde_json::{Value, json};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const BACKENDS: [Backend; 2] = [Backend::CoreGraphics, Backend::Portable];

fn variable(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

fn luma(image: &RgbImage) -> Vec<f32> {
    image
        .pixels()
        .map(|pixel| {
            let [r, g, b] = pixel.0.map(f32::from);
            0.299 * r + 0.587 * g + 0.114 * b
        })
        .collect()
}

struct Similarity {
    ssim: f64,
    ssim_ink: Option<f64>,
    ink_iou: Option<f64>,
    mean_abs: f64,
    differing: f64,
    /// Share of pixels with luma below 160 in each image: which draws heavier.
    ink: (f64, f64),
}

fn similarity(first: &RgbImage, second: &RgbImage) -> Similarity {
    let (width, height) = first.dimensions();
    let (a, b) = (luma(first), luma(second));
    let (c1, c2) = ((0.01f64 * 255.).powi(2), (0.03f64 * 255.).powi(2));
    let (mut all, mut all_n, mut ink, mut ink_n) = (0f64, 0usize, 0f64, 0usize);
    for by in (0..height.saturating_sub(7)).step_by(8) {
        for bx in (0..width.saturating_sub(7)).step_by(8) {
            let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0f64, 0f64, 0f64, 0f64, 0f64);
            let mut inked = false;
            for y in by..by + 8 {
                for x in bx..bx + 8 {
                    let index = (y * width + x) as usize;
                    let (p, q) = (f64::from(a[index]), f64::from(b[index]));
                    inked |= p < 160. || q < 160.;
                    sa += p;
                    sb += q;
                    saa += p * p;
                    sbb += q * q;
                    sab += p * q;
                }
            }
            let n = 64.;
            let (ma, mb) = (sa / n, sb / n);
            let (va, vb) = (saa / n - ma * ma, sbb / n - mb * mb);
            let cov = sab / n - ma * mb;
            let value = ((2. * ma * mb + c1) * (2. * cov + c2))
                / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            all += value;
            all_n += 1;
            if inked {
                ink += value;
                ink_n += 1;
            }
        }
    }
    let (cells_x, cells_y) = (width.div_ceil(3), height.div_ceil(3));
    let mut cells = vec![(false, false); (cells_x * cells_y) as usize];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let cell = &mut cells[((y / 3) * cells_x + x / 3) as usize];
            cell.0 |= a[index] < 160.;
            cell.1 |= b[index] < 160.;
        }
    }
    let union = cells.iter().filter(|(p, q)| *p || *q).count();
    let both = cells.iter().filter(|(p, q)| *p && *q).count();
    let (mut sum, mut differing) = (0u64, 0usize);
    for (p, q) in first.pixels().zip(second.pixels()) {
        let mut largest = 0;
        for channel in 0..3 {
            let delta = p.0[channel].abs_diff(q.0[channel]);
            sum += u64::from(delta);
            largest = largest.max(delta);
        }
        differing += usize::from(largest > 64);
    }
    let pixels = (width as usize * height as usize).max(1);
    let share = |values: &[f32]| {
        values.iter().filter(|value| **value < 160.).count() as f64 / pixels as f64
    };
    Similarity {
        ssim: if all_n == 0 { 1. } else { all / all_n as f64 },
        ssim_ink: (ink_n > 0).then(|| ink / ink_n as f64),
        ink_iou: (union > 0).then(|| both as f64 / union as f64),
        mean_abs: sum as f64 / (pixels * 3) as f64,
        differing: differing as f64 / pixels as f64,
        ink: (share(&a), share(&b)),
    }
}

/// CoreGraphics, hayro and the absolute difference side by side, sampled
/// every `step` pixels.
fn save_side_by_side(path: &Path, first: &RgbImage, second: &RgbImage, step: u32) {
    let (width, height) = first.dimensions();
    let (half_w, half_h) = (width.div_ceil(step), height.div_ceil(step));
    let mut canvas = RgbImage::from_pixel(half_w * 3 + 8, half_h, image::Rgb([255, 0, 255]));
    for y in 0..half_h {
        for x in 0..half_w {
            let (sx, sy) = ((x * step).min(width - 1), (y * step).min(height - 1));
            let (p, q) = (first.get_pixel(sx, sy), second.get_pixel(sx, sy));
            canvas.put_pixel(x, y, *p);
            canvas.put_pixel(x + half_w + 4, y, *q);
            let delta = (0..3).map(|c| p.0[c].abs_diff(q.0[c])).max().unwrap_or(0);
            canvas.put_pixel(
                x + 2 * half_w + 8,
                y,
                image::Rgb([255, 255 - delta, 255 - delta]),
            );
        }
    }
    canvas.save(path).unwrap();
}

fn cause(message: &str) -> &'static str {
    [
        ("locked", "password required"),
        ("encryption", "encryption"),
        ("cannot open", "unreadable document"),
        ("no readable pages", "no pages"),
        ("32 million", "pixel limit"),
        ("65,535", "side limit"),
        ("page box", "page box"),
        ("failed on this page", "renderer panic"),
        ("failed while reading", "renderer panic"),
        ("no PDF header", "not a PDF"),
        ("500 MiB", "size limit"),
    ]
    .into_iter()
    .find(|(needle, _)| message.contains(needle))
    .map_or("other", |(_, cause)| cause)
}

fn line(file: &mut fs::File, value: &Value) {
    writeln!(file, "{value}").unwrap();
}

#[test]
#[ignore = "driven by docs/validation/drivers/portable-raster-r1/compare.py"]
fn coregraphics_and_portable_rendering_of_a_corpus() {
    let inputs =
        variable("MARKITAI_RASTER_COMPARE_INPUTS").expect("set MARKITAI_RASTER_COMPARE_INPUTS");
    let out = PathBuf::from(
        variable("MARKITAI_RASTER_COMPARE_OUT").expect("set MARKITAI_RASTER_COMPARE_OUT"),
    );
    let dpi = variable("MARKITAI_RASTER_COMPARE_DPI").map_or(150., |value| value.parse().unwrap());
    let cap = variable("MARKITAI_RASTER_COMPARE_PAGES").map_or(40, |value| value.parse().unwrap());
    let step =
        variable("MARKITAI_RASTER_COMPARE_PICTURE_STEP").map_or(2, |value| value.parse().unwrap());
    fs::create_dir(&out).expect("the output directory must be new");
    fs::create_dir(out.join("pages")).unwrap();
    let mut documents = fs::File::create(out.join("documents.jsonl")).unwrap();
    let mut pages_out = fs::File::create(out.join("pages.jsonl")).unwrap();
    let list = fs::read_to_string(inputs).unwrap();
    for (number, path) in list
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .enumerate()
    {
        let bytes = fs::read(path).unwrap();
        let mut opened = Vec::new();
        let mut record = json!({"input": path, "bytes": bytes.len()});
        for backend in BACKENDS {
            let started = Instant::now();
            let session = PdfRasterSession::open_with(&bytes, backend);
            let elapsed = started.elapsed().as_secs_f64() * 1e3;
            record[backend.name()] = match &session {
                Ok(session) => json!({"open_ms": elapsed, "pages": session.pages()}),
                Err(error) => {
                    json!({"open_ms": elapsed, "error": error.to_string(), "cause": cause(&error.to_string())})
                }
            };
            opened.push(session.ok());
        }
        let [Some(quartz), Some(portable)] = [opened[0].as_ref(), opened[1].as_ref()] else {
            line(&mut documents, &record);
            continue;
        };
        if quartz.pages() != portable.pages() {
            record["page_count_mismatch"] = json!(true);
        }
        let count = quartz.pages().min(portable.pages()).min(cap);
        for page in 1..=count {
            let mut row = json!({"input": path, "page": page});
            let mut images = Vec::new();
            for (backend, session) in BACKENDS.into_iter().zip([quartz, portable]) {
                let started = Instant::now();
                let rendered = session
                    .dimensions(page, dpi)
                    .and_then(|_| session.render(page, dpi));
                let elapsed = started.elapsed().as_secs_f64() * 1e3;
                row[backend.name()] = match &rendered {
                    Ok(image) => json!({"ms": elapsed, "size": [image.width(), image.height()]}),
                    Err(error) => {
                        json!({"ms": elapsed, "error": error.to_string(), "cause": cause(&error.to_string())})
                    }
                };
                images.push(rendered.ok());
            }
            if let [Some(first), Some(second)] = [&images[0], &images[1]] {
                if first.dimensions() != second.dimensions() {
                    row["size_mismatch"] = json!(true);
                } else {
                    let found = similarity(first, second);
                    row["ssim"] = json!(found.ssim);
                    row["ssim_ink"] = json!(found.ssim_ink);
                    row["ink_iou"] = json!(found.ink_iou);
                    row["mean_abs"] = json!(found.mean_abs);
                    row["differing"] = json!(found.differing);
                    row["ink"] = json!([found.ink.0, found.ink.1]);
                    if found.ssim_ink.is_some_and(|value| value < 0.85)
                        || found.ink_iou.is_some_and(|value| value < 0.85)
                    {
                        let name = format!("{number:04}-p{page:03}.png");
                        save_side_by_side(&out.join("pages").join(&name), first, second, step);
                        row["picture"] = json!(name);
                    }
                }
            }
            line(&mut pages_out, &row);
        }
        record["compared_pages"] = json!(count);
        if let Some((fonts, images)) = portable.portable_warnings() {
            record["portable_warnings"] = json!({"fonts": fonts, "images": images});
        }
        line(&mut documents, &record);
    }
}

#[test]
fn identical_images_are_fully_similar_and_a_shift_is_tolerated_by_ink_cells() {
    let mut page = RgbImage::from_pixel(64, 64, image::Rgb([255; 3]));
    for y in 20..40 {
        for x in 10..50 {
            page.put_pixel(x, y, image::Rgb([0; 3]));
        }
    }
    let same = similarity(&page, &page);
    assert!((same.ssim - 1.).abs() < 1e-9 && same.ink_iou == Some(1.) && same.mean_abs == 0.);
    let mut shifted = RgbImage::from_pixel(64, 64, image::Rgb([255; 3]));
    for y in 20..40 {
        for x in 11..51 {
            shifted.put_pixel(x, y, image::Rgb([0; 3]));
        }
    }
    let moved = similarity(&page, &shifted);
    assert!(moved.ink_iou.unwrap() > 0.85, "{:?}", moved.ink_iou);
    let blank = RgbImage::from_pixel(64, 64, image::Rgb([255; 3]));
    let missing = similarity(&page, &blank);
    assert_eq!(missing.ink_iou, Some(0.));
    assert_eq!(
        cause("Native PDF rendering: PDF is locked; ..."),
        "password required"
    );
}
