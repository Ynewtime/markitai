//! Local, bounded OCR. Platform engines receive pixels, never source paths.

use crate::{Error, Result};
use serde_json::Value;

#[cfg(any(target_os = "macos", test))]
mod cjk;
#[cfg(any(target_os = "macos", test))]
mod pixels;
#[cfg(target_os = "macos")]
mod vision;

#[derive(Debug)]
pub(crate) struct OcrResult {
    pub text: String,
    /// Mean confidence of nonempty recognized lines.
    #[cfg(any(target_os = "macos", test))]
    pub confidence: f32,
    /// Pixel rectangles [left, top, right, bottom], after orientation correction.
    #[cfg(any(target_os = "macos", test))]
    pub boxes: Vec<[f32; 4]>,
    /// How many times the image was enlarged for the reading used; 1 when it
    /// was read at its own size. Only macOS tests read it.
    #[cfg(any(target_os = "macos", test))]
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub scale: f32,
}

pub(crate) fn available() -> bool {
    #[cfg(target_os = "macos")]
    return objc2::available!(macos = 11.0);
    #[cfg(not(target_os = "macos"))]
    false
}

pub(crate) fn backend() -> &'static str {
    if available() { "vision" } else { "unavailable" }
}

pub(crate) fn recognize(bytes: &[u8], cfg: &Value) -> Result<OcrResult> {
    if !available() {
        return Err(Error::Unsupported(
            "Local OCR requires macOS 11 or later; no local OCR backend is available on this platform"
                .into(),
        ));
    }
    #[cfg(target_os = "macos")]
    {
        let language = language(cfg)?;
        read(pixels::prepare(bytes)?, &language)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (bytes, cfg);
        Err(Error::Unsupported(
            "Local OCR backend is unavailable".into(),
        ))
    }
}

/// Recognize upright RGB pixels already composited over white by a renderer.
pub(crate) fn recognize_rgb(image: image::RgbImage, cfg: &Value) -> Result<OcrResult> {
    if !available() {
        return Err(Error::Unsupported(
            "Local OCR requires macOS 11 or later; no local OCR backend is available on this platform"
                .into(),
        ));
    }
    #[cfg(target_os = "macos")]
    {
        let language = language(cfg)?;
        read(pixels::prepare_rgb(image)?, &language)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (image, cfg);
        Err(Error::Unsupported(
            "Local OCR backend is unavailable".into(),
        ))
    }
}

/// Recognize prepared pixels. Small Chinese text is read a second time from
/// an enlarged copy, and that reading replaces the first.
#[cfg(target_os = "macos")]
fn read(image: pixels::Prepared, language: &str) -> Result<OcrResult> {
    let space = (image.width, image.height);
    let first = vision::recognize(&image, language, space, true)?;
    let (lines, scale) = match first.enlarge {
        Some(factor) => {
            let larger = pixels::enlarge(&image, factor)?;
            drop(image);
            (
                vision::recognize(&larger, language, space, false)?.lines,
                factor,
            )
        }
        None => (first.lines, 1.0),
    };
    let mut result = assemble(lines, space.0, space.1)?;
    result.scale = scale;
    Ok(result)
}

#[cfg(any(target_os = "macos", test))]
const MAX_LINES: usize = 10_000;
#[cfg(any(target_os = "macos", test))]
const MAX_TEXT: usize = 8 * 1024 * 1024;

#[cfg(any(target_os = "macos", test))]
#[derive(Debug)]
struct Line {
    text: String,
    confidence: f32,
    bounds: [f32; 4],
}

#[cfg(any(target_os = "macos", test))]
fn failure(message: &str) -> Error {
    Error::Conversion(format!("Local OCR: {message}"))
}

#[cfg(any(target_os = "macos", test))]
fn language(cfg: &Value) -> Result<String> {
    let configured = cfg
        .pointer("/ocr/lang")
        .and_then(Value::as_str)
        .unwrap_or("en");
    if configured.len() > 48 {
        return Err(Error::Config(
            "ocr.lang is not a supported language identifier".into(),
        ));
    }
    let normalized = configured.trim().to_ascii_lowercase().replace('_', "-");
    if normalized.is_empty()
        || !normalized
            .split('-')
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_alphanumeric()))
    {
        return Err(Error::Config(
            "ocr.lang is not a supported language identifier".into(),
        ));
    }
    Ok(match normalized.as_str() {
        "zh" | "zh-cn" | "cn" | "ch" | "zh-hans" => "zh-Hans",
        "zh-tw" | "cht" | "chinese-cht" | "zh-hant" => "zh-Hant",
        "ja" | "jp" | "japan" => "ja-JP",
        "ko" | "korean" => "ko-KR",
        "ar" | "arabic" => "ar-SA",
        "en" => "en-US",
        "fr" | "french" => "fr-FR",
        "de" | "german" => "de-DE",
        "es" => "es-ES",
        "it" => "it-IT",
        "pt" => "pt-BR",
        _ => &normalized,
    }
    .to_owned())
}

#[cfg(any(target_os = "macos", test))]
fn supported_language(requested: &str, supported: &[String]) -> Result<String> {
    if let Some(exact) = supported
        .iter()
        .find(|tag| tag.eq_ignore_ascii_case(requested))
    {
        return Ok(exact.clone());
    }
    // Short ISO codes can select the engine's regional spelling; an explicitly
    // requested region/script is never silently replaced by another language.
    if !requested.contains('-') {
        let prefix = format!("{}-", requested.to_ascii_lowercase());
        if let Some(regional) = supported
            .iter()
            .find(|tag| tag.to_ascii_lowercase().starts_with(&prefix))
        {
            return Ok(regional.clone());
        }
    }
    Err(Error::Unsupported(
        "ocr.lang is not supported by the installed macOS Vision text recognizer".into(),
    ))
}

/// Names the cause when Vision fails in a process translated by Rosetta.
#[cfg(any(target_os = "macos", test))]
const ROSETTA: &str = "under Rosetta translation";

/// The error for a failed Vision request. `reported` is the request's error
/// description, or `None` when Vision returned failure without one. Under
/// Rosetta (an x86_64 build on Apple silicon) every measured system fails the
/// request without an error, so the message names that cause and the remedy;
/// it does not claim anything about physical Intel Macs.
#[cfg(any(target_os = "macos", test))]
fn recognition_failure(reported: Option<&str>, translated: bool) -> Error {
    let detail = reported.unwrap_or("Vision reported failure without an error");
    if translated {
        failure(&format!(
            "Vision text recognition failed {ROSETTA} ({detail}); this x86_64 build is running on \
             Apple silicon, so use the native arm64 build for local OCR"
        ))
    } else {
        failure(&format!("Vision text recognition failed: {detail}"))
    }
}

#[cfg(any(target_os = "macos", test))]
fn pixel_bounds(rectangle: [f64; 4], width: u32, height: u32) -> Result<[f32; 4]> {
    let [x, y, w, h] = rectangle;
    if rectangle.iter().any(|n| !n.is_finite()) || w < 0.0 || h < 0.0 {
        return Err(failure("recognizer returned invalid text geometry"));
    }
    Ok([
        (x.clamp(0.0, 1.0) * f64::from(width)) as f32,
        ((1.0 - y - h).clamp(0.0, 1.0) * f64::from(height)) as f32,
        ((x + w).clamp(0.0, 1.0) * f64::from(width)) as f32,
        ((1.0 - y).clamp(0.0, 1.0) * f64::from(height)) as f32,
    ])
}

#[cfg(any(target_os = "macos", test))]
fn assemble(mut lines: Vec<Line>, width: u32, height: u32) -> Result<OcrResult> {
    if lines.len() > MAX_LINES {
        return Err(failure("recognized line count exceeds the safety limit"));
    }
    lines.retain(|line| !line.text.trim().is_empty());
    let mut bytes = 0usize;
    for line in &mut lines {
        let [left, top, right, bottom] = line.bounds;
        if !line.confidence.is_finite()
            || !(0.0..=1.0).contains(&line.confidence)
            || line.bounds.iter().any(|n| !n.is_finite())
            || left < 0.0
            || top < 0.0
            || right < left
            || bottom < top
            || right > width as f32
            || bottom > height as f32
        {
            return Err(failure("recognizer returned invalid text observations"));
        }
        // Text is ordinary document content; do not synthesize Markdown markup.
        line.text = line.text.trim().to_owned();
        bytes = bytes
            .checked_add(line.text.len() + 2)
            .filter(|total| *total <= MAX_TEXT)
            .ok_or_else(|| failure("recognized text exceeds the safety limit"))?;
    }
    let mut text = String::with_capacity(bytes);
    let mut boxes = Vec::with_capacity(lines.len());
    let mut confidence = 0.0;
    for block in columns(lines, 0) {
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        rows(block, &mut text, &mut boxes, &mut confidence);
    }
    let confidence = if boxes.is_empty() {
        0.0
    } else {
        confidence / boxes.len() as f32
    };
    let result = OcrResult {
        text,
        confidence,
        boxes,
        scale: 1.0,
    };
    // Keep diagnostics private to the native API, while validating the engine's
    // blank-image distinction before exposing text to the conversion pipeline.
    if (result.text.is_empty() != result.boxes.is_empty()) || !result.confidence.is_finite() {
        return Err(failure(
            "recognizer returned inconsistent text observations",
        ));
    }
    Ok(result)
}

/// Top edge, then left edge: the order `columns` and `rows` start from. One
/// named comparator, so the two sorts share one compiled sort.
#[cfg(any(target_os = "macos", test))]
fn top_then_left(a: &Line, b: &Line) -> std::cmp::Ordering {
    a.bounds[1]
        .total_cmp(&b.bounds[1])
        .then_with(|| a.bounds[0].total_cmp(&b.bounds[0]))
}

/// Lines in reading order as blocks: a page of text columns is read column
/// by column. A column gutter is the widest horizontal gap between the lines
/// narrower than three fifths of the text, at least one line height wide,
/// with at least three lines of prose (a median of 12 characters) on each side
/// overlapping by two line heights; short cells side by side, such as a
/// receipt's items and prices, stay rows. Lines crossing the gutter (a title)
/// separate sections, each read left column first. Nested columns are found
/// in each side, a few levels deep.
#[cfg(any(target_os = "macos", test))]
fn columns(mut lines: Vec<Line>, depth: usize) -> Vec<Vec<Line>> {
    const MAX_DEPTH: usize = 4;
    const MIN_COLUMN_LINES: usize = 3;
    const MIN_PROSE_CHARS: usize = 12;
    const SPANNING: f32 = 0.6;
    lines.sort_by(top_then_left);
    if depth >= MAX_DEPTH || lines.len() < 2 * MIN_COLUMN_LINES {
        return vec![lines];
    }
    let height = |line: &Line| line.bounds[3] - line.bounds[1];
    let mut heights: Vec<f32> = lines.iter().map(height).collect();
    heights.sort_by(f32::total_cmp);
    let line_height = heights[heights.len() / 2];
    let left = lines
        .iter()
        .map(|line| line.bounds[0])
        .fold(f32::INFINITY, f32::min);
    let right = lines
        .iter()
        .map(|line| line.bounds[2])
        .fold(f32::NEG_INFINITY, f32::max);
    let spanning = |line: &Line| line.bounds[2] - line.bounds[0] > SPANNING * (right - left);
    let mut spans: Vec<(f32, f32)> = lines
        .iter()
        .filter(|line| !spanning(line))
        .map(|line| (line.bounds[0], line.bounds[2]))
        .collect();
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut gutter: Option<(f32, f32)> = None;
    let mut reach = f32::NEG_INFINITY;
    for (start, end) in spans {
        if reach.is_finite()
            && start - reach >= line_height
            && gutter.is_none_or(|(a, b)| start - reach > b - a)
        {
            gutter = Some((reach, start));
        }
        reach = reach.max(end);
    }
    let Some((gap_left, gap_right)) = gutter else {
        return vec![lines];
    };
    let side = |on_left: bool| -> Vec<&Line> {
        lines
            .iter()
            .filter(|line| {
                if on_left {
                    line.bounds[2] <= gap_left
                } else {
                    line.bounds[0] >= gap_right
                }
            })
            .collect()
    };
    let prose = |side: &[&Line]| {
        let mut lengths: Vec<usize> = side.iter().map(|line| line.text.chars().count()).collect();
        lengths.sort_unstable();
        side.len() >= MIN_COLUMN_LINES && lengths[lengths.len() / 2] >= MIN_PROSE_CHARS
    };
    let (on_left, on_right) = (side(true), side(false));
    let band = |side: &[&Line]| {
        side.iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(top, bottom), line| {
                (top.min(line.bounds[1]), bottom.max(line.bounds[3]))
            })
    };
    let ((left_top, left_bottom), (right_top, right_bottom)) = (band(&on_left), band(&on_right));
    let overlap = left_bottom.min(right_bottom) - left_top.max(right_top);
    if !prose(&on_left) || !prose(&on_right) || overlap < 2.0 * line_height {
        return vec![lines];
    }
    let mut blocks = Vec::new();
    let (mut left_lines, mut right_lines) = (Vec::new(), Vec::new());
    let flush =
        |left_lines: &mut Vec<Line>, right_lines: &mut Vec<Line>, blocks: &mut Vec<Vec<Line>>| {
            for side in [std::mem::take(left_lines), std::mem::take(right_lines)] {
                if !side.is_empty() {
                    blocks.extend(columns(side, depth + 1));
                }
            }
        };
    for line in lines {
        if line.bounds[2] <= gap_left {
            left_lines.push(line);
        } else if line.bounds[0] >= gap_right {
            right_lines.push(line);
        } else {
            flush(&mut left_lines, &mut right_lines, &mut blocks);
            blocks.push(vec![line]);
        }
    }
    flush(&mut left_lines, &mut right_lines, &mut blocks);
    blocks
}

/// Appends one block's lines as text rows: lines sharing most of a vertical
/// band form a left-to-right row, and a wide vertical gap starts a paragraph.
#[cfg(any(target_os = "macos", test))]
fn rows(mut lines: Vec<Line>, text: &mut String, boxes: &mut Vec<[f32; 4]>, confidence: &mut f32) {
    lines.sort_by(top_then_left);
    let mut index = 0;
    let mut previous: Option<[f32; 4]> = None;
    while index < lines.len() {
        let first = lines[index].bounds;
        let mut end = index + 1;
        // Group only lines sharing most of the same vertical band. Each line is
        // visited once; wide multi-column pages do not trigger pairwise scans.
        while end < lines.len() {
            let next = lines[end].bounds;
            let overlap = first[3].min(next[3]) - first[1].max(next[1]);
            let min_height = (first[3] - first[1]).min(next[3] - next[1]);
            if overlap <= 0.0 || overlap < min_height * 0.6 {
                break;
            }
            end += 1;
        }
        lines[index..end].sort_by(|a, b| a.bounds[0].total_cmp(&b.bounds[0]));
        let mut row = first;
        if let Some(last) = previous {
            text.push('\n');
            if first[1] - last[3] > 0.8 * (first[3] - first[1]).max(last[3] - last[1]) {
                text.push('\n');
            }
        }
        for (offset, line) in lines[index..end].iter().enumerate() {
            if offset > 0 {
                text.push(' ');
            }
            text.push_str(&line.text);
            boxes.push(line.bounds);
            *confidence += line.confidence;
            row[1] = row[1].min(line.bounds[1]);
            row[3] = row[3].max(line.bounds[3]);
        }
        previous = Some(row);
        index = end;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn vision_failures_name_rosetta_only_in_a_translated_process() {
        let native = recognition_failure(None, false).to_string();
        assert!(native.contains("failed: Vision reported failure without an error"));
        assert!(!native.contains("Rosetta") && !native.contains("arm64"));
        let reported = recognition_failure(Some("The request was cancelled."), false).to_string();
        assert!(
            reported.ends_with("failed: The request was cancelled."),
            "{reported}"
        );
        let translated = recognition_failure(None, true);
        assert!(matches!(translated, Error::Conversion(_)));
        let translated = translated.to_string();
        assert!(translated.starts_with("Local OCR: "), "{translated}");
        for part in [ROSETTA, "without an error", "x86_64", "native arm64 build"] {
            assert!(translated.contains(part), "{part}: {translated}");
        }
    }

    #[test]
    fn language_aliases_preserve_script_and_reject_unavailable_languages() {
        for (input, expected) in [
            ("zh_cn", "zh-Hans"),
            ("chinese_cht", "zh-Hant"),
            ("JP", "ja-JP"),
            ("korean", "ko-KR"),
            ("french", "fr-FR"),
        ] {
            assert_eq!(language(&json!({"ocr":{"lang":input}})).unwrap(), expected);
        }
        let supported = vec!["en-US".into(), "zh-Hans".into(), "vi-VN".into()];
        assert_eq!(supported_language("vi", &supported).unwrap(), "vi-VN");
        assert!(supported_language("zh-Hant", &supported).is_err());
        assert!(supported_language("auto", &supported).is_err());
        assert!(language(&json!({"ocr":{"lang":"../../en"}})).is_err());
        assert!(language(&json!({"ocr":{"lang":""}})).is_err());
    }

    #[test]
    fn normalized_boxes_use_top_left_pixels_and_reject_invalid_geometry() {
        assert_eq!(
            pixel_bounds([0.1, 0.6, 0.5, 0.2], 100, 200).unwrap(),
            [10., 40., 60., 80.]
        );
        assert!(pixel_bounds([f64::NAN, 0., 1., 1.], 10, 10).is_err());
        assert!(pixel_bounds([0., 0., -1., 1.], 10, 10).is_err());
    }

    /// Lines of Chinese drawn by the in-process SVG renderer in a macOS system
    /// font, black on white, `size` pixels per em.
    #[cfg(target_os = "macos")]
    fn chinese_lines(lines: &[&str], size: u32) -> image::RgbImage {
        use resvg::{tiny_skia, usvg};
        let family = "Hiragino Sans GB";
        let longest = lines.iter().map(|l| l.chars().count()).max().unwrap() as u32;
        let (width, pitch) = (size * (longest + 2), size * 3 / 2);
        let height = size * 2 + pitch * lines.len() as u32;
        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">"#
        );
        svg.push_str(r#"<rect width="100%" height="100%" fill="white"/>"#);
        for (row, line) in lines.iter().enumerate() {
            let baseline = size * 2 + pitch * row as u32;
            svg.push_str(&format!(
                r#"<text x="{size}" y="{baseline}" font-family="{family}" font-size="{size}">{line}</text>"#
            ));
        }
        svg.push_str("</svg>");
        let mut options = usvg::Options::default();
        options.fontdb_mut().load_system_fonts();
        assert!(
            options
                .fontdb
                .faces()
                .any(|face| face.families.iter().any(|(name, _)| name == family)),
            "macOS provides {family}"
        );
        let tree = usvg::Tree::from_str(&svg, &options).unwrap();
        let mut pixmap = tiny_skia::Pixmap::new(width, height).unwrap();
        resvg::render(&tree, tiny_skia::Transform::default(), &mut pixmap.as_mut());
        image::RgbImage::from_fn(width, height, |x, y| {
            let pixel = pixmap.pixel(x, y).unwrap();
            image::Rgb([pixel.red(), pixel.green(), pixel.blue()])
        })
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn vision_reads_small_chinese_text_again_enlarged_in_original_coordinates() {
        let lines = [
            "含链接注释、图片、曲线标记或圆角裁剪的页面也能重建；",
            "宽度相同的边框块不再并入表格网格。",
        ];
        let config = json!({"ocr":{"lang":"zh"}});
        let small = chinese_lines(&lines, 13);
        let (width, height) = small.dimensions();
        let result = recognize_rgb(small, &config);
        if vision_unavailable_under_rosetta(&result) {
            return;
        }
        let result = result.unwrap();
        assert!(result.scale > 1.25, "{}", result.scale);
        assert_eq!(
            result.text.split_whitespace().collect::<String>(),
            lines.concat()
        );
        // Rectangles describe the image that was passed in, not the enlarged copy.
        assert_eq!(result.boxes.len(), 2);
        for [left, top, right, bottom] in result.boxes {
            assert!(right <= width as f32 && bottom <= height as f32);
            assert!(left < right && bottom - top < 24.0, "{top} {bottom}");
        }
        // Ordinary sizes and other languages are read once.
        let large = recognize_rgb(chinese_lines(&lines, 25), &config).unwrap();
        assert_eq!(large.scale, 1.0);
        assert_eq!(
            large.text.split_whitespace().collect::<String>(),
            lines.concat()
        );
        let english = recognize_rgb(
            chinese_lines(&["Small print from 2026", "Local text only"], 13),
            &json!({"ocr":{"lang":"en"}}),
        )
        .unwrap();
        assert_eq!(english.scale, 1.0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn vision_recovers_a_character_dropped_inside_a_wide_box() {
        // This system's recognizer drops 的 from the second line at this size
        // and stretches the box of 版 over it.
        let lines = [
            "静态获取与原生浏览器支持读取操作系统手动代理",
            "并采用参考版的单一环境代理顺序与回环直连。",
        ];
        let result = recognize_rgb(chinese_lines(&lines, 25), &json!({"ocr":{"lang":"zh"}}));
        if vision_unavailable_under_rosetta(&result) {
            return;
        }
        let result = result.unwrap();
        assert_eq!(result.scale, 1.0);
        assert_eq!(
            result.text.split_whitespace().collect::<String>(),
            lines.concat()
        );
    }

    #[test]
    fn lines_keep_rows_paragraphs_confidence_and_blank_image_distinction() {
        let line = |text: &str, bounds, confidence| Line {
            text: text.into(),
            bounds,
            confidence,
        };
        let result = assemble(
            vec![
                line(" second paragraph ", [0., 70., 90., 80.], 0.8),
                line("right", [55., 10., 95., 20.], 0.6),
                line("left", [0., 11., 40., 21.], 1.0),
                line("next line", [0., 25., 90., 35.], 0.6),
            ],
            100,
            100,
        )
        .unwrap();
        assert_eq!(result.text, "left right\nnext line\n\nsecond paragraph");
        assert!((result.confidence - 0.75).abs() < 0.001);
        assert_eq!(result.boxes.len(), 4);
        let blank = assemble(Vec::new(), 100, 100).unwrap();
        assert!(blank.text.is_empty() && blank.boxes.is_empty());
        assert_eq!(blank.confidence, 0.0);
        assert!(assemble(vec![line("bad", [0., 0., 1000., 1.], 1.)], 10, 10).is_err());
        assert!(assemble(vec![line("bad", [0., 0., 1., 1.], f32::NAN)], 10, 10).is_err());
    }

    #[test]
    fn text_columns_are_read_column_by_column_and_short_cells_stay_rows() {
        let line = |text: &str, left: f32, top: f32, right: f32| Line {
            text: text.into(),
            bounds: [left, top, right, top + 10.],
            confidence: 1.0,
        };
        let prose = |column: &str, row: usize| format!("{column} column prose line {row}");
        let mut page = vec![line(
            "A title across both columns of the page",
            0.,
            0.,
            300.,
        )];
        for row in 0..4 {
            let top = 20. + 14. * row as f32;
            page.push(line(&prose("Left", row), 0., top, 140.));
            page.push(line(&prose("Right", row), 160., top, 300.));
        }
        page.push(line("A closing line across both columns", 0., 80., 300.));
        let text = assemble(page, 300, 100).unwrap().text;
        let left = (0..4)
            .map(|row| prose("Left", row))
            .collect::<Vec<_>>()
            .join("\n");
        let right = (0..4)
            .map(|row| prose("Right", row))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            text,
            format!(
                "A title across both columns of the page\n\n{left}\n\n{right}\n\nA closing line across both columns"
            )
        );
        // Three columns: each gutter is found in turn.
        let mut three = Vec::new();
        for row in 0..3 {
            let top = 14. * row as f32;
            for (index, name) in ["First", "Second", "Third"].iter().enumerate() {
                let x = 110. * index as f32;
                three.push(line(&prose(name, row), x, top, x + 90.));
            }
        }
        let text = assemble(three, 330, 60).unwrap().text;
        let order: Vec<&str> = text
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.split(' ').next().unwrap())
            .collect();
        assert_eq!(
            order,
            [
                "First", "First", "First", "Second", "Second", "Second", "Third", "Third", "Third"
            ]
        );
        // A receipt's items and prices, and two short prose lines, stay rows.
        let mut receipt = Vec::new();
        for (row, item) in ["Coffee beans", "Oat milk", "Sourdough loaf", "Delivery"]
            .iter()
            .enumerate()
        {
            let top = 14. * row as f32;
            receipt.push(line(item, 0., top, 100.));
            receipt.push(line("$4.50", 200., top, 240.));
        }
        let text = assemble(receipt, 240, 60).unwrap().text;
        assert!(
            text.starts_with("Coffee beans $4.50\nOat milk $4.50"),
            "{text}"
        );
        let short = vec![
            line(&prose("Left", 0), 0., 0., 140.),
            line(&prose("Right", 0), 160., 0., 300.),
            line(&prose("Left", 1), 0., 14., 140.),
            line(&prose("Right", 1), 160., 14., 300.),
        ];
        let text = assemble(short, 300, 30).unwrap().text;
        assert_eq!(text.lines().count(), 2, "{text}");
        // A gap narrower than a line is a word space, not a gutter.
        let mut narrow = Vec::new();
        for row in 0..4 {
            let top = 14. * row as f32;
            narrow.push(line(&prose("Left", row), 0., top, 140.));
            narrow.push(line(&prose("Right", row), 145., top, 300.));
        }
        let text = assemble(narrow, 300, 60).unwrap().text;
        assert!(
            text.starts_with(&format!("{} {}", prose("Left", 0), prose("Right", 0))),
            "{text}"
        );
        // Blocks stacked rather than side by side keep top-to-bottom order.
        let mut stacked = Vec::new();
        for row in 0..3 {
            stacked.push(line(&prose("Right", row), 160., 14. * row as f32, 300.));
            stacked.push(line(&prose("Left", row), 0., 60. + 14. * row as f32, 140.));
        }
        let text = assemble(stacked, 300, 100).unwrap().text;
        assert!(text.starts_with(&prose("Right", 0)), "{text}");
    }

    /// True, after saying why, when `result` is Vision's explicit failure in a
    /// process translated by Rosetta; the caller then skips its assertions
    /// about recognized text. Never true in a native process, and any other
    /// failure is left for the caller to report.
    #[cfg(target_os = "macos")]
    pub(crate) fn vision_unavailable_under_rosetta<T>(result: &Result<T>) -> bool {
        let unavailable = crate::system_frameworks::translated()
            && matches!(result, Err(Error::Conversion(message)) if message.contains(ROSETTA));
        if let (true, Err(error)) = (unavailable, result) {
            eprintln!("skipping recognized-text assertions: {error}");
        }
        unavailable
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn vision_recognizes_authored_pixels_and_reports_a_real_blank_image() {
        let config = json!({"ocr":{"lang":"en"}});
        let result = recognize(include_bytes!("ocr/fixtures/english.png"), &config);
        let rejected = || {
            assert!(matches!(
                recognize(b"not an image", &config),
                Err(Error::Conversion(_))
            ));
            assert!(matches!(
                recognize(
                    include_bytes!("ocr/fixtures/english.png"),
                    &json!({"ocr":{"lang":"not-a-language"}})
                ),
                Err(Error::Unsupported(_))
            ));
        };
        if vision_unavailable_under_rosetta(&result) {
            // Inputs rejected before the recognition request keep their errors.
            rejected();
            return;
        }
        let result = result.unwrap();
        let expected = include_str!("ocr/fixtures/english.txt");
        assert_eq!(
            result.text.split_whitespace().collect::<Vec<_>>(),
            expected.split_whitespace().collect::<Vec<_>>()
        );
        assert!(result.confidence > 0.3 && result.confidence <= 1.0);
        assert!(result.boxes.len() >= 3);
        let rgb = image::load_from_memory(include_bytes!("ocr/fixtures/english.png"))
            .unwrap()
            .into_rgb8();
        let rendered = recognize_rgb(rgb, &config).unwrap();
        assert_eq!(rendered.text, result.text);
        assert_eq!(rendered.boxes, result.boxes);
        assert_eq!(rendered.confidence, result.confidence);
        let blank = pixels::encode_test_image(image::DynamicImage::ImageRgb8(
            image::RgbImage::from_pixel(400, 200, image::Rgb([255; 3])),
        ));
        let result = recognize(&blank, &config).unwrap();
        assert!(result.text.is_empty() && result.boxes.is_empty());
        assert_eq!(result.confidence, 0.0);
        rejected();
    }
}
