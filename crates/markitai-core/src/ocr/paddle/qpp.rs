//! An experimental opt-in local OCR engine, `MARKITAI_OCR_ENGINE=qppocr`:
//! `qppocr`, a pure-Rust runner of the PP-OCRv6 models with its own AVX2/FMA
//! and NEON kernels. It reads the multilingual recognizer's languages (the
//! default language and the Chinese, Japanese and Latin-script spellings);
//! `tract`, the default, reads the rest (the Korean and other single-script
//! recognizers use operators qppocr lacks) and takes over for what `engine`
//! lists. Measured in the whole pipeline it was no faster than tract.
//!
//! One engine per process, decided once: [`engine`]. qppocr's kernels are
//! chosen at run time (`is_x86_feature_detected!`), so a processor without
//! AVX2 and FMA never executes an instruction it lacks.

use super::{Line, Result, models};
use image::RgbImage;
use qppocr::{Advanced, DeviceChoice, Engine, ModelSource, Tier};
use std::sync::{Mutex, OnceLock};

/// The environment variable that chooses the engine: `qppocr`, `tract` or
/// `auto` (the default).
const VARIABLE: &str = "MARKITAI_OCR_ENGINE";

/// qppocr's own 0/180 line classifier (PP-LCNet_x0_25_textline_ori); the
/// v2 classifier of the tract path turns over upright lines it should not.
pub(super) const CLASSIFIER: &str = "ppocr-lcnet-textline-ori";

/// The loaded engine. qppocr spreads one image over every thread of the OCR
/// budget, so images are read one at a time; the pages in flight overlap
/// with their rendering and layout instead.
pub(super) struct Qpp {
    engine: Engine,
    gate: Mutex<()>,
    backend: String,
}

/// Lines read from one image, with what the languages after the first
/// reading need.
pub(super) struct Found {
    pub lines: Vec<Line>,
    /// Each line's region: top-left, top-right, bottom-right, bottom-left,
    /// in image pixels.
    pub corners: Vec<[[f32; 2]; 4]>,
    /// The middle of each character of each line, in image pixels.
    pub centers: Vec<Vec<[f32; 2]>>,
    /// The region was found upside down (or, for a tall region, turned the
    /// other way) and read turned over.
    pub flipped: Vec<bool>,
    /// The image had at least [`crate::ocr::MAX_TEXT_REGIONS`] text regions.
    pub capped: bool,
}

impl Found {
    pub(super) fn empty() -> Found {
        Found {
            lines: Vec::new(),
            corners: Vec::new(),
            centers: Vec::new(),
            flipped: Vec::new(),
            capped: false,
        }
    }

    /// The lines of a part of an image, `map`ped from the part's pixels to
    /// those of the image (`width` by `height`), added to these.
    pub(super) fn extend_mapped(
        &mut self,
        other: Found,
        map: impl Fn([f32; 2]) -> [f32; 2],
        (width, height): (u32, u32),
    ) {
        self.capped |= other.capped;
        for (mut line, (corners, (centers, flipped))) in other.lines.into_iter().zip(
            other
                .corners
                .into_iter()
                .zip(other.centers.into_iter().zip(other.flipped)),
        ) {
            let corners = corners.map(&map);
            line.bounds = bounds_of(&corners, width, height);
            self.lines.push(line);
            self.corners.push(corners);
            self.centers.push(centers.into_iter().map(&map).collect());
            self.flipped.push(flipped);
        }
    }
}

/// The pixel rectangle `[left, top, right, bottom]` of a region's corners,
/// within an image of `width` by `height`.
fn bounds_of(corners: &[[f32; 2]; 4], width: u32, height: u32) -> [f32; 4] {
    let xs = corners.map(|p| p[0]);
    let ys = corners.map(|p| p[1]);
    let low = |v: [f32; 4], limit: u32| {
        v.into_iter()
            .fold(f32::MAX, f32::min)
            .clamp(0.0, limit as f32)
    };
    let high = |v: [f32; 4], limit: u32| {
        v.into_iter()
            .fold(f32::MIN, f32::max)
            .clamp(0.0, limit as f32)
    };
    [
        low(xs, width),
        low(ys, height),
        high(xs, width),
        high(ys, height),
    ]
}

/// The engine this process reads with, `None` for tract. The first call
/// decides; a bad `MARKITAI_OCR_ENGINE` is an error every time.
pub(super) fn engine() -> Result<Option<&'static Qpp>> {
    static ENGINE: OnceLock<std::result::Result<Option<Qpp>, String>> = OnceLock::new();
    ENGINE
        .get_or_init(decide)
        .as_ref()
        .map(Option::as_ref)
        .map_err(|message| crate::Error::Config(message.clone()))
}

/// The engine this process would read with, by the same rules as [`engine`]
/// but without loading anything: for `doctor`.
pub(super) fn planned() -> &'static str {
    match request(&std::env::var(VARIABLE).unwrap_or_default()) {
        Ok(Request::Qppocr) => "qppocr",
        _ => "tract",
    }
}

/// Whether this processor runs qppocr's fast kernels: AVX2 and FMA on
/// x86_64, always NEON on aarch64.
fn fast_cpu() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

fn notice(text: &str) {
    crate::console::line(&format!("Local OCR: {text}"));
}

/// What `MARKITAI_OCR_ENGINE` asks for.
#[derive(Debug, PartialEq, Eq)]
enum Request {
    Auto,
    Qppocr,
    Tract,
}

fn request(text: &str) -> std::result::Result<Request, String> {
    match text.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Ok(Request::Auto),
        "qppocr" => Ok(Request::Qppocr),
        "tract" => Ok(Request::Tract),
        _ => Err(format!("{VARIABLE} must be qppocr, tract or auto")),
    }
}

fn decide() -> std::result::Result<Option<Qpp>, String> {
    let forced = match request(&std::env::var(VARIABLE).unwrap_or_default())? {
        Request::Auto => false,
        Request::Qppocr => true,
        Request::Tract => {
            notice(&format!(
                "{VARIABLE}=tract: text is read with the tract engine."
            ));
            return Ok(None);
        }
    };
    if !forced {
        // qppocr is opt-in: in the full pipeline it measured no faster than
        // tract on x86_64 laptops and slower on aarch64.
        return Ok(None);
    }
    if !fast_cpu() {
        notice(&format!(
            "{VARIABLE}=qppocr on a processor without AVX2 and FMA: qppocr runs its portable \
             path, which is very slow."
        ));
    }
    match start() {
        Ok(qpp) => {
            super::super::trace::line(format_args!(
                "engine=qppocr kernels={} threads={}",
                qpp.backend,
                super::super::thread_budget()
            ));
            Ok(Some(qpp))
        }
        Err(error) => {
            notice(&format!(
                "the qppocr engine could not start ({error}), so text is read with the slower \
                 tract engine."
            ));
            Ok(None)
        }
    }
}

/// Rec shards (qppocr reads lines on several small thread pools) for a
/// budget of `threads`: pools of two threads at least, none above four, so
/// the threads busy at once stay within the budget.
fn shards(threads: usize) -> usize {
    match (threads / 2).min(4) {
        0 | 1 => 0,
        n => n,
    }
}

pub(super) fn start() -> std::result::Result<Qpp, String> {
    Ok(Qpp {
        engine: build()?,
        gate: Mutex::new(()),
        backend: format!("{:?}", qppocr::detect_backend()).to_lowercase(),
    })
}

/// An engine on the verified model files.
fn build() -> std::result::Result<Engine, String> {
    let bytes = |name: &str| -> std::result::Result<Vec<u8>, String> {
        models::named(name)
            .and_then(|model| models::load(model, true))
            .map_err(|error| error.to_string())
    };
    let source = ModelSource::Bytes {
        det: bytes(super::DETECTOR)?,
        rec: bytes(super::MULTILINGUAL)?,
        cls: Some(bytes(CLASSIFIER)?),
        // The recognizer file carries its character list.
        dict: None,
    };
    let threads = super::super::thread_budget();
    let rec_shards = shards(threads);
    Engine::builder()
        .tier(Tier::Small)
        .threads(threads)
        .device(DeviceChoice::Cpu)
        .advanced(move |advanced: &mut Advanced| {
            advanced.rec_shards = rec_shards;
            // Spaces between Chinese or Japanese and Latin letters are
            // added by this crate's own rule (`spaced`), as for tract.
            advanced.rec_space_gap = 0.0;
        })
        .build_with(source)
        .map_err(|error| error.to_string())
}

/// qppocr said no to this image once; the rest of the process is told so
/// once.
fn failed_once(error: &dyn std::fmt::Display) {
    static TOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !TOLD.swap(true, std::sync::atomic::Ordering::Relaxed) {
        notice(&format!(
            "the qppocr engine failed on an image ({error}); such images are read with the slower \
             tract engine."
        ));
    }
}

impl Qpp {
    /// Detect, orient and read the text lines of `image`; `None` when
    /// qppocr failed on it (said once), which the caller reads with tract.
    pub(super) fn read(&self, image: &RgbImage) -> Option<Found> {
        let (width, height) = image.dimensions();
        let pixels = qppocr::rgb_from_bytes(width, height, image.as_raw().clone())
            .map_err(|error| failed_once(&error))
            .ok()?;
        let result = {
            let _turn = self.gate.lock().unwrap_or_else(|e| e.into_inner());
            self.engine.run(&pixels)
        };
        drop(pixels);
        let result = result.map_err(|error| failed_once(&error)).ok()?;
        let timings = result.timings;
        super::super::trace::line(format_args!(
            "engine=qppocr det={:.0}ms (pre={:.0} infer={:.0} post={:.0}) crop={:.0}ms cls={:.0}ms \
             rec={:.0}ms (pre={:.0} infer={:.0} post={:.0}) total={:.0}ms regions={}",
            timings.det_pre_ms + timings.det_infer_ms + timings.det_post_ms,
            timings.det_pre_ms,
            timings.det_infer_ms,
            timings.det_post_ms,
            timings.crop_ms,
            timings.cls_ms,
            timings.rec_pre_ms + timings.rec_infer_ms + timings.rec_post_ms,
            timings.rec_pre_ms,
            timings.rec_infer_ms,
            timings.rec_post_ms,
            timings.total_ms,
            result.lines.len(),
        ));
        let mut found = Found::empty();
        found.capped = result.num_boxes >= crate::ocr::MAX_TEXT_REGIONS;
        for line in result.lines {
            let [tl, tr, _, bl] = line.pts;
            let bounds = bounds_of(&line.pts, width, height);
            // The text runs along the top edge of a wide region and up the
            // left edge of a tall one (qppocr turns those clockwise); a line
            // it found upside down runs the other way.
            let wide =
                |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
            let direction = if wide(tl, bl) >= 1.5 * wide(tl, tr) {
                [tl[0] - bl[0], tl[1] - bl[1]]
            } else {
                [tr[0] - tl[0], tr[1] - tl[1]]
            };
            let flipped = line.rotation == 180;
            found.lines.push(Line {
                text: line.text,
                confidence: if line.confidence.is_finite() {
                    line.confidence.clamp(0.0, 1.0)
                } else {
                    0.0
                },
                bounds,
                direction: if flipped {
                    direction.map(|n| -n)
                } else {
                    direction
                },
                marks: Default::default(),
            });
            found.corners.push(line.pts);
            found.centers.push(
                line.chars
                    .iter()
                    .map(|c| {
                        let mean = |axis: usize| c.pts.iter().map(|p| p[axis]).sum::<f32>() / 4.0;
                        [mean(0), mean(1)]
                    })
                    .collect(),
            );
            found.flipped.push(flipped);
        }
        Some(found)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_variable_names_an_engine() {
        assert_eq!(request(""), Ok(Request::Auto));
        assert_eq!(request(" Auto "), Ok(Request::Auto));
        assert_eq!(request("QPPOCR"), Ok(Request::Qppocr));
        assert_eq!(request("tract"), Ok(Request::Tract));
        assert!(
            request("ort")
                .unwrap_err()
                .contains("qppocr, tract or auto")
        );
    }

    #[test]
    fn rec_shards_keep_the_busy_threads_within_the_budget() {
        for threads in 1..=64 {
            let shards = shards(threads);
            assert!(shards == 0 || (2..=4).contains(&shards));
            // qppocr gives each shard (threads / shards, at least 2) threads.
            if shards > 0 {
                assert!(shards * (threads / shards).max(2) <= threads, "{threads}");
            }
        }
        assert_eq!(shards(1), 0);
        assert_eq!(shards(4), 2);
        assert_eq!(shards(16), 4);
    }
}
