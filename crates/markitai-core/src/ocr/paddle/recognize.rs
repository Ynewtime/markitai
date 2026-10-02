//! Text line recognition with a CTC model: each detected region is cut out
//! of the image upright, scaled to the model's 48-pixel height and read as a
//! sequence of per-column character probabilities, which collapse into text.

use super::super::{Result, failure};
use image::{Rgb, RgbImage};

/// The model reads lines this many pixels tall.
pub(super) const HEIGHT: u32 = 48;
/// Line inputs are padded to widths in steps of this many pixels, so that
/// lines of similar length share an input shape.
pub(super) const BUCKET: u32 = 32;
/// Lines are padded to at least this width, as the reference configuration
/// does (its batches are at least 320 pixels wide).
pub(super) const MIN_WIDTH: u32 = 320;
/// A line wider than this (100 heights) is squeezed to fit.
pub(super) const MAX_WIDTH: u32 = 4800;
/// A cut-out region at least this many times taller than wide holds text
/// running down: it is turned a quarter counter-clockwise to be read.
pub(super) const UPRIGHT_RATIO: f32 = 1.5;

/// The characters a recognizer's outputs stand for: output 0 is the CTC
/// blank, outputs 1 to n the model's dictionary in order, and output n + 1 a
/// space.
#[derive(Debug)]
pub(super) struct Dictionary {
    symbols: Vec<Box<str>>,
}

impl Dictionary {
    /// The dictionary a model stores in its `character` metadata, one
    /// character per line.
    pub fn parse(listing: &str) -> Result<Dictionary> {
        let mut symbols: Vec<Box<str>> = listing
            .split('\n')
            .map(|line| line.strip_suffix('\r').unwrap_or(line))
            .map(Box::from)
            .collect();
        // A final newline ends the last line rather than adding one.
        if listing.ends_with('\n') {
            symbols.pop();
        }
        if listing.is_empty() || symbols.is_empty() || symbols.len() > 100_000 {
            return Err(failure(
                "recognition model has no usable character dictionary",
            ));
        }
        symbols.push(" ".into());
        Ok(Dictionary { symbols })
    }

    /// How many outputs a model with this dictionary has.
    pub fn outputs(&self) -> usize {
        self.symbols.len() + 1
    }

    fn symbol(&self, index: usize) -> Option<&str> {
        index
            .checked_sub(1)
            .and_then(|i| self.symbols.get(i))
            .map(|s| &**s)
    }
}

/// One line read from the model's outputs.
#[derive(Debug, PartialEq)]
pub(super) struct Reading {
    pub text: String,
    /// The mean of the highest probability of each character kept: how sure
    /// the model is of the line (0 when it read nothing).
    pub confidence: f32,
    /// For each character of `text`, the middle of the output steps that
    /// read it, in steps from the line's left edge.
    pub centers: Vec<f32>,
}

/// Collapse `steps` rows of `classes` probabilities (row-major) into text:
/// each step's most probable output, repeats merged, blanks dropped.
pub(super) fn decode(
    probabilities: &[f32],
    classes: usize,
    dictionary: &Dictionary,
) -> Result<Reading> {
    if classes != dictionary.outputs() || !probabilities.len().is_multiple_of(classes) {
        return Err(failure(
            "recognition model outputs do not match its character dictionary",
        ));
    }
    let mut text = String::new();
    let (mut sum, mut kept) = (0.0f64, 0usize);
    let mut previous = usize::MAX;
    // Each kept symbol's first and last step, and its character count.
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    let mut in_symbol = false;
    for (number, step) in probabilities.chunks_exact(classes).enumerate() {
        let (index, best) =
            step.iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |(i, b), (j, p)| {
                    if *p > b { (j, *p) } else { (i, b) }
                });
        if !best.is_finite() {
            return Err(failure("recognition model returned invalid probabilities"));
        }
        if index != previous {
            in_symbol = false;
            if let Some(symbol) = dictionary.symbol(index) {
                text.push_str(symbol);
                sum += f64::from(best);
                kept += 1;
                runs.push((number, number, symbol.chars().count()));
                in_symbol = true;
            }
        } else if in_symbol && let Some(run) = runs.last_mut() {
            run.1 = number;
        }
        previous = index;
    }
    let confidence = if kept == 0 {
        0.0
    } else {
        (sum / kept as f64) as f32
    };
    let centers = runs
        .iter()
        .flat_map(|&(first, last, count)| {
            std::iter::repeat_n((first + last) as f32 / 2.0 + 0.5, count)
        })
        .collect();
    Ok(Reading {
        text,
        confidence: confidence.clamp(0.0, 1.0),
        centers,
    })
}

/// Text read left to right from a right-to-left script, as stored: the
/// characters reversed, and runs of left-to-right text (Latin letters and
/// numbers, with the spaces and marks between them) kept in their order.
pub(super) fn logical(visual: &str) -> String {
    let rtl = |c: char| matches!(c, '\u{0590}'..='\u{08ff}' | '\u{fb1d}'..='\u{fdff}' | '\u{fe70}'..='\u{feff}');
    let ltr = |c: char| c.is_alphanumeric() && !rtl(c);
    let mut characters: Vec<char> = visual.chars().collect();
    characters.reverse();
    let mut i = 0;
    while i < characters.len() {
        if !ltr(characters[i]) {
            i += 1;
            continue;
        }
        // A run from this character to the last left-to-right one before
        // the next right-to-left letter.
        let mut end = i;
        let mut j = i;
        while j < characters.len() && !rtl(characters[j]) {
            if ltr(characters[j]) {
                end = j;
            }
            j += 1;
        }
        characters[i..=end].reverse();
        i = end + 1;
    }
    characters.into_iter().collect()
}

/// The region of `image` inside `corners` (top-left, top-right,
/// bottom-right, bottom-left), cut out as a rectangle: each output pixel
/// samples the image bilinearly through the perspective that maps the
/// rectangle's corners onto the region's, with edge pixels repeated.
pub(super) fn cut(image: &RgbImage, corners: &[[f32; 2]; 4]) -> Result<RgbImage> {
    let distance =
        |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let width = distance(corners[0], corners[1]).max(distance(corners[2], corners[3])) as u32;
    let height = distance(corners[0], corners[3]).max(distance(corners[1], corners[2])) as u32;
    let (width, height) = (width.max(1), height.max(1));
    if u64::from(width) * u64::from(height) > 32_000_000 {
        return Err(failure("a text region exceeds the pixel limit"));
    }
    let target = [
        [0.0, 0.0],
        [width as f32, 0.0],
        [width as f32, height as f32],
        [0.0, height as f32],
    ];
    let transform = perspective(&target, corners)
        .ok_or_else(|| failure("a text region has degenerate corners"))?;
    let (iw, ih) = image.dimensions();
    let sample = |x: f64, y: f64| -> Rgb<u8> {
        let x = x.clamp(0.0, f64::from(iw - 1));
        let y = y.clamp(0.0, f64::from(ih - 1));
        let (x0, y0) = (x.floor() as u32, y.floor() as u32);
        let (x1, y1) = ((x0 + 1).min(iw - 1), (y0 + 1).min(ih - 1));
        let (fx, fy) = (x - f64::from(x0), y - f64::from(y0));
        let p = |x: u32, y: u32| image.get_pixel(x, y).0.map(f64::from);
        let (a, b, c, d) = (p(x0, y0), p(x1, y0), p(x0, y1), p(x1, y1));
        Rgb([0, 1, 2].map(|i| {
            let top = a[i] + (b[i] - a[i]) * fx;
            let bottom = c[i] + (d[i] - c[i]) * fx;
            (top + (bottom - top) * fy).round().clamp(0.0, 255.0) as u8
        }))
    };
    Ok(RgbImage::from_fn(width, height, |u, v| {
        let (u, v) = (f64::from(u), f64::from(v));
        let w = transform[6] * u + transform[7] * v + 1.0;
        let x = (transform[0] * u + transform[1] * v + transform[2]) / w;
        let y = (transform[3] * u + transform[4] * v + transform[5]) / w;
        sample(x, y)
    }))
}

/// The perspective transform (row-major 3 by 3 with the last entry 1, as
/// eight numbers) mapping each `from` point onto the `to` point.
fn perspective(from: &[[f32; 2]; 4], to: &[[f32; 2]; 4]) -> Option<[f64; 8]> {
    // Each pair gives two rows of A h = b.
    let mut a = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let (x, y) = (f64::from(from[i][0]), f64::from(from[i][1]));
        let (u, v) = (f64::from(to[i][0]), f64::from(to[i][1]));
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    // Gaussian elimination with partial pivoting.
    for column in 0..8 {
        let pivot =
            (column..8).max_by(|&i, &j| a[i][column].abs().total_cmp(&a[j][column].abs()))?;
        if a[pivot][column].abs() < 1e-9 {
            return None;
        }
        a.swap(column, pivot);
        let pivot_row = a[column];
        for (row, values) in a.iter_mut().enumerate() {
            if row != column {
                let factor = values[column] / pivot_row[column];
                for (value, pivot) in values[column..].iter_mut().zip(&pivot_row[column..]) {
                    *value -= factor * pivot;
                }
            }
        }
    }
    let mut h = [0.0; 8];
    for i in 0..8 {
        h[i] = a[i][8] / a[i][i];
    }
    h.iter().all(|n| n.is_finite()).then_some(h)
}

/// A cut-out line turned a quarter counter-clockwise: text that ran down
/// the page now runs left to right.
pub(super) fn turn_counter_clockwise(image: &RgbImage) -> RgbImage {
    image::imageops::rotate270(image)
}

/// The model's input for one line: the line scaled to [`HEIGHT`] pixels,
/// keeping its proportions, on a width in steps of [`BUCKET`] (at least
/// `min_width`), as BGR planes normalized to [-1, 1] and padded with zeros.
/// Returns the data and the input's width.
/// Also returns the width the line itself takes in the input.
pub(super) fn input(line: &RgbImage, min_width: u32) -> (Vec<f32>, u32, u32) {
    let (w, h) = line.dimensions();
    let scaled_width =
        ((HEIGHT as f32 * w as f32 / h.max(1) as f32).ceil() as u32).clamp(1, MAX_WIDTH);
    let width = scaled_width.div_ceil(BUCKET).max(1) * BUCKET;
    let width = width.max(min_width).min(MAX_WIDTH);
    let scaled = super::resize(line, scaled_width.min(width), HEIGHT);
    (
        super::planes(&scaled, width, HEIGHT),
        width,
        scaled_width.min(width),
    )
}

/// `reading`'s text with the spaces the line's pixels show between a
/// Chinese or Japanese letter and a Latin letter or digit, which the
/// recognizer leaves out: where its text has none, the widest ink-free run of
/// pixel columns between the two characters' centers is measured over the
/// cut-out line, and a run of [`super::super::layout::SPACE`] line heights or
/// more is a space, as for Vision. `pixels_per_step` maps the reading's
/// character centers onto the line's pixels.
pub(super) fn spaced(line: &RgbImage, reading: &Reading, pixels_per_step: f32) -> String {
    use super::super::layout;
    let characters: Vec<char> = reading.text.chars().collect();
    if characters.len() != reading.centers.len() || !pixels_per_step.is_finite() {
        return reading.text.clone();
    }
    let junctions = layout::junctions(&characters);
    let (width, height) = (line.width() as f32, line.height() as f32);
    let Some(background) = (!junctions.is_empty())
        .then(|| layout::background(line, [0.0, 0.0, width, height]))
        .flatten()
    else {
        return reading.text.clone();
    };
    let x = |index: usize| reading.centers[index] * pixels_per_step;
    let mut text = String::with_capacity(reading.text.len() + junctions.len());
    let mut next = junctions.iter().peekable();
    for (index, character) in characters.iter().enumerate() {
        text.push(*character);
        if next.peek() == Some(&&index) {
            next.next();
            if layout::spaced(line, [0.0, height], [x(index), x(index + 1)], background) {
                text.push(' ');
            }
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dictionary() -> Dictionary {
        Dictionary::parse("a\nb\nc\n").unwrap()
    }

    /// One step of probabilities with `index` most probable at `p`.
    fn step(index: usize, p: f32, classes: usize) -> Vec<f32> {
        let mut row = vec![(1.0 - p) / (classes - 1) as f32; classes];
        row[index] = p;
        row
    }

    #[test]
    fn the_dictionary_adds_the_blank_and_a_space() {
        let d = dictionary();
        assert_eq!(d.outputs(), 5);
        assert_eq!(d.symbol(0), None);
        assert_eq!(d.symbol(1), Some("a"));
        assert_eq!(d.symbol(3), Some("c"));
        assert_eq!(d.symbol(4), Some(" "));
        assert_eq!(d.symbol(5), None);
        // Without a final newline the last line still counts; a line that is
        // a space is a symbol.
        assert_eq!(Dictionary::parse("x\n \ny").unwrap().outputs(), 5);
        assert!(Dictionary::parse("").is_err());
    }

    #[test]
    fn ctc_merges_repeats_drops_blanks_and_averages_kept_probabilities() {
        let d = dictionary();
        let classes = d.outputs();
        let steps: Vec<f32> = [
            step(1, 0.9, classes),
            step(1, 0.8, classes),
            step(0, 0.99, classes),
            step(1, 0.7, classes),
            step(4, 0.6, classes),
            step(2, 0.5, classes),
            step(2, 0.95, classes),
        ]
        .concat();
        let reading = decode(&steps, classes, &d).unwrap();
        assert_eq!(reading.text, "aa b");
        // Character centers: the first a spans steps 0 and 1, the b steps 5 and 6.
        assert_eq!(reading.centers, [1.0, 3.5, 4.5, 6.0]);
        // Kept: 0.9 (first a), 0.7 (second a), 0.6 (space), 0.5 (b).
        assert!(
            (reading.confidence - 0.675).abs() < 1e-6,
            "{}",
            reading.confidence
        );
        let blank = decode(&step(0, 0.9, classes), classes, &d).unwrap();
        assert_eq!((blank.text.as_str(), blank.confidence), ("", 0.0));
        assert!(decode(&[0.5; 6], 6, &d).is_err());
        assert!(decode(&[f32::NAN; 5], 5, &d).is_err());
    }

    #[test]
    fn a_wide_ink_free_gap_between_chinese_and_latin_letters_is_a_space() {
        // Two 30-pixel glyph blocks on a 40-pixel line, `gap` pixels apart.
        let line = |gap: u32| {
            RgbImage::from_fn(140, 40, |x, y| {
                let ink = (5..35).contains(&y)
                    && ((10..40).contains(&x) || (40 + gap..70 + gap).contains(&x));
                Rgb([if ink { 0 } else { 255 }; 3])
            })
        };
        let reading = |gap: u32, text: &str| Reading {
            text: text.into(),
            confidence: 0.99,
            centers: vec![25.0, 55.0 + gap as f32],
        };
        // 0.22 line heights of 40 pixels is 8.8.
        assert_eq!(spaced(&line(16), &reading(16, "在C"), 1.0), "在 C");
        assert_eq!(spaced(&line(4), &reading(4, "在C"), 1.0), "在C");
        // Two Latin letters, and a reading whose centers do not match, stay.
        assert_eq!(spaced(&line(16), &reading(16, "ab"), 1.0), "ab");
        let mut short = reading(16, "在C");
        short.centers.pop();
        assert_eq!(spaced(&line(16), &short, 1.0), "在C");
    }

    #[test]
    fn right_to_left_text_is_stored_in_reading_order_with_numbers_kept() {
        // Read left to right off the image: "123" then Arabic letters.
        assert_eq!(logical("123 ابج"), "جبا 123");
        assert_eq!(logical("ابج"), "جبا");
        assert_eq!(logical("PDF 2.5 ابج"), "جبا PDF 2.5");
        assert_eq!(logical("plain"), "plain");
    }

    #[test]
    fn a_region_is_cut_upright_and_tall_lines_turned() {
        // A white image with a black block at x 20..40, y 10..20.
        let image = RgbImage::from_fn(80, 40, |x, y| {
            if (20..40).contains(&x) && (10..20).contains(&y) {
                Rgb([0; 3])
            } else {
                Rgb([255; 3])
            }
        });
        let cut = cut(
            &image,
            &[[20.0, 10.0], [40.0, 10.0], [40.0, 20.0], [20.0, 20.0]],
        )
        .unwrap();
        assert_eq!(cut.dimensions(), (20, 10));
        assert_eq!(cut.get_pixel(0, 0).0, [0; 3]);
        assert_eq!(cut.get_pixel(19, 9).0, [0; 3]);
        // Sampling outside the image repeats its edge.
        let edge = super::cut(
            &image,
            &[[70.0, 0.0], [79.0, 0.0], [79.0, 39.0], [70.0, 39.0]],
        )
        .unwrap();
        assert_eq!(edge.dimensions(), (9, 39));
        let turned = turn_counter_clockwise(&edge);
        assert_eq!(turned.dimensions(), (39, 9));
        assert!(super::cut(&image, &[[1.0, 1.0]; 4]).is_err());
    }

    #[test]
    fn line_inputs_keep_proportions_on_bucketed_widths() {
        let line = RgbImage::from_pixel(200, 24, Rgb([255; 3]));
        let (data, width, scaled) = input(&line, MIN_WIDTH);
        // 200 x 24 scales to 400 x 48, padded to 416.
        assert_eq!((width, scaled), (416, 400));
        assert_eq!(data.len(), (3 * HEIGHT * width) as usize);
        // White is 1.0, padding 0.0.
        assert_eq!(data[0], 1.0);
        assert_eq!(data[(width - 1) as usize], 0.0);
        let short = RgbImage::from_pixel(10, 24, Rgb([255; 3]));
        assert_eq!(input(&short, MIN_WIDTH).1, MIN_WIDTH);
        assert_eq!(input(&short, 0).1, 32);
        let long = RgbImage::from_pixel(10_000, 10, Rgb([255; 3]));
        assert_eq!(input(&long, MIN_WIDTH).1, MAX_WIDTH);
    }
}
