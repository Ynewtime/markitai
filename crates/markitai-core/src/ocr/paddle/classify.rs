//! The text direction classifier: whether a cut-out line is upside down.

use super::super::{Result, failure};
use image::RgbImage;

/// The classifier reads lines scaled to this size, in batches of this many.
pub(super) const HEIGHT: u32 = 48;
pub(super) const WIDTH: u32 = 192;
pub(super) const BATCH: usize = 6;
/// A line is turned over only when the classifier is at least this sure.
const SURE: f32 = 0.9;

/// The classifier's input for one line, as BGR planes normalized to [-1, 1]
/// on a [`WIDTH`]-pixel input padded with zeros.
pub(super) fn input(line: &RgbImage) -> Vec<f32> {
    let (w, h) = line.dimensions();
    let scaled = ((HEIGHT as f32 * w as f32 / h.max(1) as f32).ceil() as u32).clamp(1, WIDTH);
    super::planes(&super::resize(line, scaled, HEIGHT), WIDTH, HEIGHT)
}

/// Whether each line of a batch is upside down, from the classifier's
/// `[upright, upside down]` probabilities per line.
pub(super) fn upside_down(probabilities: &[f32], lines: usize) -> Result<Vec<bool>> {
    if probabilities.len() < lines * 2 || probabilities.iter().any(|p| !p.is_finite()) {
        return Err(failure("direction classifier returned invalid output"));
    }
    Ok(probabilities
        .as_chunks::<2>()
        .0
        .iter()
        .take(lines)
        .map(|pair| pair[1] > pair[0] && pair[1] > SURE)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_sure_upside_down_answer_turns_a_line() {
        let answers = [0.95, 0.05, 0.05, 0.95, 0.15, 0.85, 0.5, 0.5];
        assert_eq!(
            upside_down(&answers, 4).unwrap(),
            [false, true, false, false]
        );
        // Padding rows of a batch are ignored.
        assert_eq!(upside_down(&answers, 2).unwrap(), [false, true]);
        assert!(upside_down(&answers[..3], 2).is_err());
        assert!(upside_down(&[f32::NAN, 0.0], 1).is_err());
    }

    #[test]
    fn lines_are_scaled_into_the_input_keeping_proportions() {
        let wide = RgbImage::from_pixel(400, 20, image::Rgb([255; 3]));
        let data = input(&wide);
        assert_eq!(data.len(), (3 * HEIGHT * WIDTH) as usize);
        // Squeezed to the full width: no padding.
        assert_eq!(data[(WIDTH - 1) as usize], 1.0);
        let narrow = RgbImage::from_pixel(20, 20, image::Rgb([255; 3]));
        let data = input(&narrow);
        assert_eq!(data[47], 1.0);
        assert_eq!(data[48], 0.0);
    }
}
