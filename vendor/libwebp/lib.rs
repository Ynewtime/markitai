//! libwebp's lossy encoder behind one C function (`markitai_webp.c`).
#![no_std]

use core::ffi::{c_int, c_void};

/// Receives the encoded file in pieces; returning 0 stops the encoder.
pub type Sink = unsafe extern "C" fn(data: *const u8, size: usize, context: *mut c_void) -> c_int;

/// libwebp's `VP8_ENC_ERROR_BAD_WRITE`: the sink returned 0.
pub const ERROR_BAD_WRITE: c_int = 8;
/// libwebp's `VP8_ENC_ERROR_BAD_DIMENSION`: a side over 16383 pixels.
pub const ERROR_BAD_DIMENSION: c_int = 5;

unsafe extern "C" {
    /// Encodes `width` x `height` pixels, RGBA when `alpha` is non-zero, else
    /// RGB, rows `stride` bytes apart, as lossy WebP at `quality` (0-100) with
    /// Pillow's settings, and hands the file to `sink`. Returns 0, or libwebp's
    /// `WebPEncodingError`.
    pub fn markitai_webp_encode(
        pixels: *const u8,
        width: c_int,
        height: c_int,
        stride: c_int,
        alpha: c_int,
        quality: f32,
        sink: Sink,
        context: *mut c_void,
    ) -> c_int;
}
