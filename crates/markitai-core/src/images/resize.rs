//! Lanczos3 downscaling of 8-bit RGB that gives the bytes `image`'s
//! `DynamicImage::resize` does, with the same arithmetic, on several threads
//! and one row at a time. `image` resizes pixel by pixel through generic
//! accessors and keeps a full-size `f32` RGBA intermediate (70 MB for a
//! 2458 x 1766 picture); the tests below hold this to its output exactly.

use image::RgbImage;
use std::f32::consts::PI;

/// Most threads one resize uses.
const THREADS: usize = 4;

fn sinc(t: f32) -> f32 {
    let a = t * PI;
    if t == 0.0 { 1.0 } else { a.sin() / a }
}

fn lanczos3(x: f32) -> f32 {
    if x.abs() < 3.0 {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// The source pixels `left..left + weights.len()` that output pixel `n`
/// along one axis is the weighted sum of.
struct Taps {
    left: usize,
    weights: Vec<f32>,
}

fn taps(length: u32, new_length: u32) -> Vec<Taps> {
    let ratio = length as f32 / new_length as f32;
    let sratio = if ratio < 1.0 { 1.0 } else { ratio };
    let support = 3.0 * sratio;
    (0..new_length)
        .map(|n| {
            let input = (n as f32 + 0.5) * ratio;
            let left = ((input - support).floor() as i64).clamp(0, i64::from(length) - 1);
            let right = ((input + support).ceil() as i64).clamp(left + 1, i64::from(length));
            let input = input - 0.5;
            let mut weights: Vec<f32> = (left..right)
                .map(|i| lanczos3((i as f32 - input) / sratio))
                .collect();
            let mut sum = 0.0;
            for weight in &weights {
                sum += weight;
            }
            for weight in &mut weights {
                *weight /= sum;
            }
            Taps {
                left: left as usize,
                weights,
            }
        })
        .collect()
}

/// The size `DynamicImage::resize` gives an image fitted inside
/// `width` x `height` (aspect ratio kept, never a zero side).
pub(super) fn fit(from: (u32, u32), width: u32, height: u32) -> (u32, u32) {
    let ratio = f64::min(
        f64::from(width) / f64::from(from.0),
        f64::from(height) / f64::from(from.1),
    );
    let side =
        |n: u32| ((f64::from(n) * ratio).round() as u64).clamp(1, u64::from(u32::MAX)) as u32;
    (side(from.0), side(from.1))
}

/// `pixels` (`width` x `height`, `channels` bytes per pixel, the first three
/// of them red, green and blue) resized to `new_width` x `new_height`.
pub(super) fn lanczos3_rgb(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: usize,
    new_width: u32,
    new_height: u32,
) -> RgbImage {
    let vertical = taps(height, new_height);
    let horizontal = taps(width, new_width);
    let stride = width as usize * channels;
    let row_bytes = new_width as usize * 3;
    let mut out = vec![0u8; row_bytes * new_height as usize];
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(THREADS)
        .min(new_height as usize / 16 + 1);
    let rows_each = (new_height as usize).div_ceil(threads);
    std::thread::scope(|scope| {
        for (band, rows) in out.chunks_mut(rows_each * row_bytes).enumerate() {
            let (vertical, horizontal) = (&vertical, &horizontal);
            scope.spawn(move || {
                let mut column = vec![0f32; width as usize * 3];
                for (n, row) in rows.chunks_mut(row_bytes).enumerate() {
                    column.fill(0.0);
                    let tap = &vertical[band * rows_each + n];
                    for (i, weight) in tap.weights.iter().enumerate() {
                        let source = &pixels[(tap.left + i) * stride..][..stride];
                        for (sum, pixel) in column
                            .as_chunks_mut::<3>()
                            .0
                            .iter_mut()
                            .zip(source.chunks_exact(channels))
                        {
                            sum[0] += f32::from(pixel[0]) * weight;
                            sum[1] += f32::from(pixel[1]) * weight;
                            sum[2] += f32::from(pixel[2]) * weight;
                        }
                    }
                    for (out, tap) in row.as_chunks_mut::<3>().0.iter_mut().zip(horizontal) {
                        let mut t = [0f32; 3];
                        for (i, weight) in tap.weights.iter().enumerate() {
                            let at = (tap.left + i) * 3;
                            t[0] += column[at] * weight;
                            t[1] += column[at + 1] * weight;
                            t[2] += column[at + 2] * weight;
                        }
                        for (byte, value) in out.iter_mut().zip(t) {
                            *byte = value.clamp(0.0, 255.0).round() as u8;
                        }
                    }
                }
            });
        }
    });
    RgbImage::from_raw(new_width, new_height, out).expect("rows fill the buffer")
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;
    use image::imageops::FilterType;

    /// A busy picture: smooth ramps with hard edges and noise.
    fn picture(width: u32, height: u32, channels: usize) -> Vec<u8> {
        let mut state = 0x2545_f491_u32;
        let mut bytes = Vec::new();
        for y in 0..height {
            for x in 0..width {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let edge = if (x / 7 + y / 5) % 3 == 0 { 255 } else { 0 };
                let mut pixel = [
                    (x * 255 / width) as u8,
                    ((y * 255 / height) as u8).wrapping_add((state >> 28) as u8),
                    edge,
                    255,
                ];
                pixel[2] = pixel[2].wrapping_add((state >> 24) as u8 & 3);
                bytes.extend_from_slice(&pixel[..channels]);
            }
        }
        bytes
    }

    #[test]
    fn resizing_gives_the_bytes_of_the_image_crates_lanczos3() {
        for (width, height, bound_w, bound_h) in [
            (301, 203, 128, 99_999),
            (97, 131, 50, 40),
            (1000, 30, 640, 99_999),
            (64, 900, 99_999, 128),
            (500, 400, 499, 399),
            (33, 33, 1, 1),
            (20, 600, 7, 99_999),
        ] {
            for channels in [3, 4] {
                let bytes = picture(width, height, channels);
                let expected = if channels == 3 {
                    DynamicImage::ImageRgb8(
                        RgbImage::from_raw(width, height, bytes.clone()).unwrap(),
                    )
                } else {
                    DynamicImage::ImageRgba8(
                        image::RgbaImage::from_raw(width, height, bytes.clone()).unwrap(),
                    )
                }
                .resize(bound_w, bound_h, FilterType::Lanczos3)
                .to_rgb8();
                let (new_width, new_height) = fit((width, height), bound_w, bound_h);
                assert_eq!(
                    (new_width, new_height),
                    expected.dimensions(),
                    "{width}x{height} in {bound_w}x{bound_h}"
                );
                let ours = lanczos3_rgb(&bytes, width, height, channels, new_width, new_height);
                assert!(
                    ours.as_raw() == expected.as_raw(),
                    "{width}x{height} in {bound_w}x{bound_h}, {channels} channels"
                );
            }
        }
    }
}
