//! Portable local OCR: PaddleOCR's PP-OCR models (text detection, line
//! direction and line recognition), run in this process by the pure-Rust
//! `tract` inference engine. Windows and Linux use it; macOS builds it only
//! with the `portable-media` feature, where Vision stays the default.
//!
//! An image is read in three steps. The detector finds text regions as
//! rotated rectangles; each region is cut out upright (turned a quarter when
//! it is tall, then over when the classifier finds it upside down); and a
//! recognizer reads each cut-out line. The lines then take the same layout
//! steps as Vision's ([`super::assemble`]): turned pages, columns, rows,
//! paragraphs, code and mended zeros.
//!
//! The default language reads with the multilingual recognizer (Latin
//! script, Chinese, Japanese) and, when that reading looks like another
//! script, with the Korean one too, reusing the detected regions; see
//! [`read_default`].

mod classify;
mod detect;
pub(crate) mod models;
mod recognize;

use super::{Line, OcrResult, Result, failure};
use image::RgbImage;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tract_onnx::prelude::*;

type Plan = Arc<TypedRunnableModel>;

/// The multilingual recognizer the default language reads with.
const MULTILINGUAL: &str = "ppocrv6-rec-small";
/// The recognizer of Hangul.
const KOREAN: &str = "korean-ppocrv5-rec-mobile";
const DETECTOR: &str = "ppocrv6-det-small";
const CLASSIFIER: &str = "ppocr-cls-mobile-v2";
/// Lines the recognizer is less sure of than this are dropped from the text,
/// as the reference's RapidOCR configuration does.
const TEXT_SCORE: f32 = 0.5;

/// Scale `image` to `width` by `height` with a triangle filter, which
/// averages the pixels it covers when it shrinks.
fn resize(image: &RgbImage, width: u32, height: u32) -> RgbImage {
    if image.dimensions() == (width, height) {
        return image.clone();
    }
    image::imageops::resize(
        image,
        width.max(1),
        height.max(1),
        image::imageops::FilterType::Triangle,
    )
}

/// `image` as BGR planes normalized to [-1, 1], on a `width` by `height`
/// input padded with zeros on the right and below.
fn planes(image: &RgbImage, width: u32, height: u32) -> Vec<f32> {
    let (w, h) = (width as usize, height as usize);
    let plane = w * h;
    let mut data = vec![0.0f32; plane * 3];
    let (iw, ih) = image.dimensions();
    for y in 0..(ih as usize).min(h) {
        for x in 0..(iw as usize).min(w) {
            let pixel = image.get_pixel(x as u32, y as u32).0;
            for channel in 0..3 {
                data[(2 - channel) * plane + y * w + x] = f32::from(pixel[channel]) / 127.5 - 1.0;
            }
        }
    }
    data
}

fn engine_error(what: &str, error: impl std::fmt::Display) -> crate::Error {
    failure(&format!("{what}: {error}"))
}

/// A model of `role` parsed from its verified file, without input shapes.
fn parse(name: &str, role: models::Role) -> Result<InferenceModel> {
    let model = models::named(name)?;
    if model.role != role {
        return Err(failure(&format!(
            "the model {name} is not a {role:?} model"
        )));
    }
    let bytes = models::load(model, true)?;
    tract_onnx::onnx()
        .with_ignore_output_shapes(true)
        .with_ignore_value_info(true)
        .model_for_read(&mut std::io::Cursor::new(bytes))
        .map_err(|error| engine_error(&format!("cannot read the model {name}"), error))
}

/// `model` optimized for inputs of `shape`, where `None` is a dimension that
/// varies from run to run.
fn plan(mut model: InferenceModel, shape: [Option<usize>; 4], name: &str) -> Result<Plan> {
    let dims: TVec<TDim> = shape
        .iter()
        .enumerate()
        .map(|(axis, n)| match n {
            Some(n) => TDim::from(*n as i64),
            None => model.sym(&format!("d{axis}")).into(),
        })
        .collect();
    let built = model
        .set_input_fact(0, f32::fact(dims).into())
        .and_then(|_| model.into_optimized())
        .and_then(|typed| typed.into_runnable());
    built.map_err(|error| engine_error(&format!("cannot prepare the model {name}"), error))
}

/// Run a plan on one input of `shape` and return its first output's values
/// and shape.
fn run(plan: &Plan, shape: &[usize], data: Vec<f32>, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let input = Tensor::from_shape(shape, &data)
        .map_err(|error| engine_error(&format!("cannot pass an input to {name}"), error))?;
    // from_shape copies the planes; do not retain both allocations during inference.
    drop(data);
    let outputs = plan
        .run(tvec!(input.into()))
        .map_err(|error| engine_error(&format!("the model {name} failed"), error))?;
    let output = outputs
        .first()
        .ok_or_else(|| failure(&format!("the model {name} returned no output")))?;
    let values = output
        .try_as_plain_ram()
        .and_then(|view| view.as_slice::<f32>().map(<[f32]>::to_vec))
        .map_err(|error| engine_error(&format!("the model {name} returned no numbers"), error))?;
    Ok((values, output.shape().to_vec()))
}

/// The detector: the parsed model, and plans for the input sizes it read
/// last (each new size costs an optimization, and the pages of one
/// document usually share a size).
struct Detector {
    model: InferenceModel,
    plans: Mutex<Vec<((u32, u32), Plan)>>,
}

const DETECTOR_PLANS: usize = 4;

impl Detector {
    fn plan(&self, width: u32, height: u32) -> Result<Plan> {
        let mut plans = self.plans.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(index) = plans.iter().position(|(size, _)| *size == (width, height)) {
            let entry = plans.remove(index);
            let plan = entry.1.clone();
            plans.push(entry);
            return Ok(plan);
        }
        // Keep the cache locked while building: simultaneous cold requests for
        // one shape must not optimize and retain duplicate plans.
        let plan = plan(
            self.model.clone(),
            [
                Some(1),
                Some(3),
                Some(height as usize),
                Some(width as usize),
            ],
            DETECTOR,
        )?;
        if plans.len() >= DETECTOR_PLANS {
            plans.remove(0);
        }
        plans.push(((width, height), plan.clone()));
        Ok(plan)
    }

    fn regions(&self, image: &RgbImage) -> Result<Vec<detect::Region>> {
        // Bound cold optimization and input planes as well as the inference itself.
        let _turn = DetectionTurn::take();
        let layout = detect::Layout::of(image.width(), image.height());
        let plan = self.plan(layout.width, layout.height)?;
        let shape = [1, 3, layout.height as usize, layout.width as usize];
        let data = detect::input(image, &layout);
        let (map, out) = run(&plan, &shape, data, DETECTOR)?;
        if !detection_shape(&out, map.len(), layout.width, layout.height) {
            return Err(failure("detection model returned a map of the wrong size"));
        }
        detect::regions(
            &map[..layout.width as usize * layout.height as usize],
            &layout,
        )
    }
}

fn detection_shape(shape: &[usize], values: usize, width: u32, height: u32) -> bool {
    shape == [1, 1, height as usize, width as usize]
        && (width as usize).checked_mul(height as usize) == Some(values)
}

/// Detections that may run at once in a process. A detection holds about
/// 300 bytes per input pixel while it runs (some 0.6 GB for a 1600 by 1200
/// page), so a batch converting many images in parallel takes turns.
const DETECTIONS_AT_ONCE: usize = 2;

/// One of the [`DETECTIONS_AT_ONCE`] turns, given back when dropped.
struct DetectionTurn;

static DETECTING: (Mutex<usize>, std::sync::Condvar) = (Mutex::new(0), std::sync::Condvar::new());

impl DetectionTurn {
    fn take() -> DetectionTurn {
        let (count, freed) = &DETECTING;
        let mut running = count.lock().unwrap_or_else(|e| e.into_inner());
        while *running >= DETECTIONS_AT_ONCE {
            running = freed.wait(running).unwrap_or_else(|e| e.into_inner());
        }
        *running += 1;
        DetectionTurn
    }
}

impl Drop for DetectionTurn {
    fn drop(&mut self) {
        let (count, freed) = &DETECTING;
        let mut running = count.lock().unwrap_or_else(|e| e.into_inner());
        *running = running.saturating_sub(1);
        freed.notify_one();
    }
}

/// A recognizer: a plan for one line at a time (tract runs a batch of lines
/// slower than the same lines one by one) and its dictionary.
struct Recognizer {
    name: &'static str,
    plan: Plan,
    dictionary: recognize::Dictionary,
    right_to_left: bool,
}

impl Recognizer {
    fn load(name: &'static str) -> Result<Recognizer> {
        let model = parse(name, models::Role::Recognize)?;
        let listing = model
            .properties
            .get("onnx.metadata_props.character")
            .and_then(|tensor| {
                tensor
                    .try_as_plain_ram()
                    .and_then(|view| view.to_scalar::<String>().cloned())
                    .ok()
            })
            .ok_or_else(|| failure(&format!("the model {name} has no character dictionary")))?;
        let dictionary = recognize::Dictionary::parse(&listing)?;
        let plan = plan(
            model,
            [Some(1), Some(3), Some(recognize::HEIGHT as usize), None],
            name,
        )?;
        Ok(Recognizer {
            name,
            plan,
            dictionary,
            right_to_left: models::named(name)?.right_to_left,
        })
    }

    fn read(&self, line: &RgbImage) -> Result<recognize::Reading> {
        let (data, width, scaled) = recognize::input(line, recognize::MIN_WIDTH);
        let shape = [1, 3, recognize::HEIGHT as usize, width as usize];
        let (probabilities, out) = run(&self.plan, &shape, data, self.name)?;
        let classes = *out.last().unwrap_or(&0);
        let mut reading = recognize::decode(&probabilities, classes, &self.dictionary)?;
        if self.right_to_left {
            reading.text = recognize::logical(&reading.text);
        } else {
            // Output steps cover the input evenly; the line takes `scaled` of
            // its `width` pixels, `line.width()` of its own.
            let steps = out.get(1).copied().unwrap_or(1).max(1) as f32;
            let pixels = width as f32 / steps * line.width() as f32 / scaled.max(1) as f32;
            reading.text = recognize::spaced(line, &reading, pixels);
        }
        Ok(reading)
    }
}

/// The detector and the direction classifier, loaded once per process.
struct Engine {
    detector: Detector,
    classifier: Plan,
}

fn engine() -> Result<Arc<Engine>> {
    static ENGINE: Mutex<Option<Arc<Engine>>> = Mutex::new(None);
    let mut slot = ENGINE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(engine) = slot.as_ref() {
        return Ok(engine.clone());
    }
    let detector = Detector {
        model: parse(DETECTOR, models::Role::Detect)?,
        plans: Mutex::new(Vec::new()),
    };
    let classifier = plan(
        parse(CLASSIFIER, models::Role::Classify)?,
        [
            Some(classify::BATCH),
            Some(3),
            Some(classify::HEIGHT as usize),
            Some(classify::WIDTH as usize),
        ],
        CLASSIFIER,
    )?;
    let engine = Arc::new(Engine {
        detector,
        classifier,
    });
    *slot = Some(engine.clone());
    Ok(engine)
}

/// Values loaded once per process, by name. Each name loads under its own
/// lock, so a slow load (a first download) holds up only the threads that
/// need the same value, not those reading with one already loaded.
struct Loads<T> {
    slots: Mutex<Vec<(&'static str, Slot<T>)>>,
}

type Slot<T> = Arc<Mutex<Load<T>>>;

enum Load<T> {
    Pending,
    Ready(Arc<T>),
    /// An optional load failed: it is not tried again in this process.
    Failed,
}

impl<T> Loads<T> {
    const fn new() -> Self {
        Loads {
            slots: Mutex::new(Vec::new()),
        }
    }

    fn slot(&self, name: &'static str) -> Slot<T> {
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, slot)) = slots.iter().find(|(n, _)| *n == name) {
            return slot.clone();
        }
        let slot = Arc::new(Mutex::new(Load::Pending));
        slots.push((name, slot.clone()));
        slot
    }

    /// The value `name`, loaded by `load` unless it was loaded before. A
    /// failure is returned and tried again on the next call.
    fn get(&self, name: &'static str, load: impl FnOnce() -> Result<T>) -> Result<Arc<T>> {
        let slot = self.slot(name);
        let mut state = slot.lock().unwrap_or_else(|e| e.into_inner());
        if let Load::Ready(value) = &*state {
            return Ok(value.clone());
        }
        let value = Arc::new(load()?);
        *state = Load::Ready(value.clone());
        Ok(value)
    }

    /// The value `name` when it can be loaded; a failure is kept, so a
    /// model that cannot be downloaded (offline, say) is tried once.
    fn optional(&self, name: &'static str, load: impl FnOnce() -> Result<T>) -> Option<Arc<T>> {
        let slot = self.slot(name);
        let mut state = slot.lock().unwrap_or_else(|e| e.into_inner());
        match &*state {
            Load::Ready(value) => return Some(value.clone()),
            Load::Failed => return None,
            Load::Pending => {}
        }
        match load() {
            Ok(value) => {
                let value = Arc::new(value);
                *state = Load::Ready(value.clone());
                Some(value)
            }
            Err(_) => {
                *state = Load::Failed;
                None
            }
        }
    }
}

static RECOGNIZERS: Loads<Recognizer> = Loads::new();

/// The recognizer `name`, loaded once per process.
fn recognizer(name: &'static str) -> Result<Arc<Recognizer>> {
    RECOGNIZERS.get(name, || Recognizer::load(name))
}

/// The recognizer `name` when it can be loaded, tried once per process.
fn optional_recognizer(name: &'static str) -> Option<Arc<Recognizer>> {
    RECOGNIZERS.optional(name, || Recognizer::load(name))
}

/// A detected line, cut out upright.
struct Cut {
    image: RgbImage,
    /// The region's pixel rectangle in the image.
    bounds: [f32; 4],
    /// The direction the text runs in the image.
    direction: [f32; 2],
    /// The classifier found the line upside down. It is read both ways and
    /// the more confident reading kept: the classifier turns over a few
    /// upright lines of ordinary prose (one of 24 rendered 300 DPI pages
    /// lost a line that way).
    flagged: bool,
}

/// The text lines of `image`, cut out upright.
fn cuts(engine: &Engine, image: &RgbImage) -> Result<Vec<Cut>> {
    let regions = engine.detector.regions(image)?;
    let mut cuts = Vec::with_capacity(regions.len());
    for region in &regions {
        let [tl, tr, _, bl] = region.corners;
        let mut cut = recognize::cut(image, &region.corners)?;
        let mut direction = [tr[0] - tl[0], tr[1] - tl[1]];
        if cut.height() as f32 >= recognize::UPRIGHT_RATIO * cut.width() as f32 {
            cut = recognize::turn_counter_clockwise(&cut);
            direction = [bl[0] - tl[0], bl[1] - tl[1]];
        }
        let xs = region.corners.map(|p| p[0]);
        let ys = region.corners.map(|p| p[1]);
        let (w, h) = (image.width() as f32, image.height() as f32);
        let bounds = [
            xs.iter().copied().fold(f32::MAX, f32::min).clamp(0.0, w),
            ys.iter().copied().fold(f32::MAX, f32::min).clamp(0.0, h),
            xs.iter().copied().fold(f32::MIN, f32::max).clamp(0.0, w),
            ys.iter().copied().fold(f32::MIN, f32::max).clamp(0.0, h),
        ];
        cuts.push(Cut {
            image: cut,
            bounds,
            direction,
            flagged: false,
        });
    }
    let size = classify::BATCH * 3 * (classify::HEIGHT * classify::WIDTH) as usize;
    for batch in cuts.chunks_mut(classify::BATCH) {
        let mut data = Vec::with_capacity(size);
        for cut in batch.iter() {
            data.extend(classify::input(&cut.image));
        }
        data.resize(size, 0.0);
        let shape = [
            classify::BATCH,
            3,
            classify::HEIGHT as usize,
            classify::WIDTH as usize,
        ];
        let (answers, _) = run(&engine.classifier, &shape, data, CLASSIFIER)?;
        let turned = classify::upside_down(&answers, batch.len())?;
        for (cut, over) in batch.iter_mut().zip(turned) {
            cut.flagged = over;
        }
    }
    Ok(cuts)
}

/// Threads reading lines in this process, beyond the callers' own.
static HELPERS: AtomicUsize = AtomicUsize::new(0);
const MAX_HELPERS_PER_IMAGE: usize = 7;

/// Take up to `wanted` helper threads within the processor count.
fn take_helpers(wanted: usize) -> usize {
    let budget = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .saturating_sub(1);
    let mut active = HELPERS.load(Ordering::Relaxed);
    loop {
        let take = wanted.min(budget.saturating_sub(active));
        if take == 0 {
            return 0;
        }
        match HELPERS.compare_exchange(active, active + take, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return take,
            Err(now) => active = now,
        }
    }
}

/// One cut-out line read with `recognizer`: as cut, and also turned over
/// when the classifier flagged it, keeping the more confident reading.
fn read_cut(recognizer: &Recognizer, cut: &Cut) -> Result<Line> {
    let mut reading = recognizer.read(&cut.image)?;
    let mut direction = cut.direction;
    if cut.flagged {
        let turned = image::imageops::rotate180(&cut.image);
        let over = recognizer.read(&turned)?;
        if over.confidence > reading.confidence {
            reading = over;
            direction = direction.map(|n| -n);
        }
    }
    Ok(Line {
        text: reading.text,
        confidence: reading.confidence,
        bounds: cut.bounds,
        direction,
    })
}

/// Read each cut-out line with `recognizer`, on this thread and up to
/// [`MAX_HELPERS_PER_IMAGE`] more; one line per cut, in order.
fn read_lines(recognizer: &Recognizer, cuts: &[&Cut]) -> Result<Vec<Line>> {
    let next = AtomicUsize::new(0);
    let work = || -> Result<Vec<(usize, Line)>> {
        let mut done = Vec::new();
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(cut) = cuts.get(index) else {
                return Ok(done);
            };
            done.push((index, read_cut(recognizer, cut)?));
        }
    };
    let helpers = take_helpers(MAX_HELPERS_PER_IMAGE.min(cuts.len().saturating_sub(1)));
    let results: Vec<Result<Vec<(usize, Line)>>> = std::thread::scope(|scope| {
        let spawned: Vec<_> = (0..helpers).map(|_| scope.spawn(work)).collect();
        let mut results = vec![work()];
        for handle in spawned {
            results.push(
                handle
                    .join()
                    .unwrap_or_else(|_| Err(failure("a recognition thread failed"))),
            );
        }
        results
    });
    HELPERS.fetch_sub(helpers, Ordering::AcqRel);
    let mut lines: Vec<Option<Line>> = (0..cuts.len()).map(|_| None).collect();
    for result in results {
        for (index, line) in result? {
            lines[index] = Some(line);
        }
    }
    lines
        .into_iter()
        .map(|line| line.ok_or_else(|| failure("a line was not read")))
        .collect()
}

/// What `ocr.lang` asks of the portable engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Choice {
    /// The default language: see [`read_default`].
    Default,
    /// One recognizer.
    Model(&'static str),
}

/// The languages the multilingual recognizer reads (the reference's
/// PP-OCRv6 list), by the spellings `ocr.lang` accepts for them.
const MULTILINGUAL_LANGUAGES: &[&str] = &[
    "ch",
    "chinese-cht",
    "en",
    "japan",
    "af",
    "az",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "es",
    "et",
    "eu",
    "fi",
    "fr",
    "ga",
    "gl",
    "hr",
    "hu",
    "id",
    "is",
    "it",
    "ku",
    "la",
    "lb",
    "lt",
    "lv",
    "mi",
    "ms",
    "mt",
    "nl",
    "no",
    "nb",
    "nn",
    "oc",
    "pl",
    "pt",
    "qu",
    "rm",
    "ro",
    "rs-latin",
    "sk",
    "sl",
    "sq",
    "sv",
    "sw",
    "tl",
    "tr",
    "uz",
    "vi",
    "french",
    "german",
    "zh",
    "cn",
    "cht",
    "ja",
    "jp",
    "yue",
    "english",
];

/// Spellings of the single-language recognizers.
const SINGLE_LANGUAGES: &[(&str, &[&str])] = &[
    (KOREAN, &["ko", "korean"]),
    (
        "arabic-ppocrv5-rec-mobile",
        &["ar", "arabic", "ars", "fa", "ur", "ug"],
    ),
    ("th-ppocrv5-rec-mobile", &["th", "thai"]),
    ("el-ppocrv5-rec-mobile", &["el", "greek"]),
    (
        "eslav-ppocrv5-rec-mobile",
        &["eslav", "ru", "uk", "be", "russian", "ukrainian"],
    ),
    (
        "cyrillic-ppocrv5-rec-mobile",
        &["cyrillic", "bg", "mk", "kk", "ky", "mn", "tg", "sr-cyrl"],
    ),
    ("latin-ppocrv5-rec-mobile", &["latin"]),
    (
        "devanagari-ppocrv5-rec-mobile",
        &["devanagari", "hi", "mr", "ne", "sa"],
    ),
    ("ta-ppocrv5-rec-mobile", &["ta", "tamil"]),
    ("te-ppocrv5-rec-mobile", &["te", "telugu"]),
];

/// The recognizer a normalized `ocr.lang` spelling (lower case, hyphens)
/// names: the reference's RapidOCR names and short codes, and the Vision
/// spellings (`zh-Hans`, `en-US`, `ko-KR`), so that one configuration works
/// on every system. A region or script after the language is accepted when
/// the language is: `pt-br` reads as `pt`.
pub(super) fn choice(spelling: &str) -> Result<Choice> {
    if spelling == "en" {
        return Ok(Choice::Default);
    }
    let lookup = |name: &str| -> Option<Choice> {
        if let Some((model, _)) = SINGLE_LANGUAGES
            .iter()
            .find(|(_, names)| names.contains(&name))
        {
            return Some(Choice::Model(model));
        }
        MULTILINGUAL_LANGUAGES
            .contains(&name)
            .then_some(Choice::Model(MULTILINGUAL))
    };
    if let Some(found) = lookup(spelling) {
        return Ok(found);
    }
    match spelling.split_once('-') {
        Some((language, _)) if !language.is_empty() => lookup(language),
        _ => None,
    }
    .ok_or_else(|| {
        crate::Error::Unsupported(format!(
            "ocr.lang is not a language the portable OCR models read ({spelling}); use en (the \
             default: Latin script, Chinese and Japanese, then Korean), zh, zh_tw, ja, ko, ar, th, \
             el, ru, latin, cyrillic, devanagari, ta, te, or a code such as fr, de or es"
        ))
    })
}

/// Recognize upright RGB pixels in the language `spelling` names.
pub(super) fn read(image: &RgbImage, spelling: &str) -> Result<OcrResult> {
    let choice = choice(spelling)?;
    models::preflight(&needed(spelling)?)?;
    let engine = engine()?;
    let cuts = cuts(&engine, image)?;
    let (lines, language, unread) = match choice {
        Choice::Model(name) => {
            let all: Vec<&Cut> = cuts.iter().collect();
            (read_lines(&*recognizer(name)?, &all)?, name, false)
        }
        Choice::Default => read_default(&cuts)?,
    };
    finish(lines, image, language, unread)
}

/// The default language. The multilingual recognizer reads Latin script,
/// Chinese (Simplified and Traditional) and Japanese, the scripts the
/// reference's default reads; it returns nothing, or a doubtful reading, for
/// a line of Hangul. Each line it is not sure of is read again with the
/// Korean recognizer, from the same cut-out, and that reading replaces it
/// when it is Hangul read with more confidence, or the first reading covers
/// little of the line (two Hanja of a Korean line). A Korean recognizer that
/// cannot be loaded (offline, say) leaves the first reading, and is not
/// tried again in this process. When half of
/// the lines or more stay doubtful, the image holds text of a script neither
/// reads: only the sure lines are kept, and the result is marked unread.
fn read_default(cuts: &[Cut]) -> Result<(Vec<Line>, &'static str, bool)> {
    let all: Vec<&Cut> = cuts.iter().collect();
    let mut lines = read_lines(&*recognizer(MULTILINGUAL)?, &all)?;
    let doubtful: Vec<usize> = (0..lines.len()).filter(|&i| !sure(&lines[i])).collect();
    if doubtful.is_empty() {
        return Ok((lines, MULTILINGUAL, false));
    }
    // Lines read with confidence by either recognizer.
    let mut read: Vec<bool> = lines.iter().map(sure).collect();
    let mut korean_lines = 0;
    if let Some(korean) = optional_recognizer(KOREAN) {
        let again: Vec<&Cut> = doubtful.iter().map(|&i| &cuts[i]).collect();
        if let Ok(readings) = read_lines(&korean, &again) {
            for (&i, reading) in doubtful.iter().zip(readings) {
                // A first reading that covers little of its line (Hanja read
                // from a line of Hangul) loses whatever its confidence.
                if korean_replaces(&lines[i], &reading) {
                    lines[i] = reading;
                    read[i] = true;
                    korean_lines += 1;
                }
            }
        }
    }
    let language = if korean_lines * 2 > lines.len() {
        KOREAN
    } else {
        MULTILINGUAL
    };
    let unread = unread(&lines, &read);
    if unread {
        let mut flags = read.into_iter();
        lines.retain(|_| flags.next().unwrap_or(false));
    }
    Ok((lines, language, unread))
}

/// A line the recognizers are at least this sure of is text of a script
/// they read. Measured on the corpora of `docs/ocr.md`.
const SURE: f32 = 0.9;

/// A reading with fewer characters than this per line height of its region's
/// length left most of the region unread: the multilingual recognizer reads a
/// line of Hangul as its final period alone, confidently.
const SPARSE: f32 = 0.4;

/// Whether a reading covers its region: it has characters, at least
/// [`SPARSE`] per line height of the region's length.
fn covers(line: &Line) -> bool {
    let [left, top, right, bottom] = line.bounds;
    let (long, short) = (
        (right - left).max(bottom - top),
        (right - left).min(bottom - top),
    );
    let characters = line.text.chars().filter(|c| !c.is_whitespace()).count();
    characters > 0 && characters as f32 >= SPARSE * long / short.max(1.0)
}

/// Whether a line was read with confidence and covers its region.
fn sure(line: &Line) -> bool {
    line.confidence >= SURE && covers(line)
}

fn korean_replaces(first: &Line, korean: &Line) -> bool {
    korean.confidence >= TEXT_SCORE
        && hangul(&korean.text)
        && (korean.confidence > first.confidence || !covers(first))
}

/// Whether a line reads as Hangul: at least two syllables (one in a line of
/// a single letter), a third or more of its letters.
fn hangul(text: &str) -> bool {
    let letters = text.chars().filter(|c| c.is_alphabetic()).count();
    let syllables = text
        .chars()
        .filter(|c| matches!(c, '\u{ac00}'..='\u{d7a3}'))
        .count();
    syllables > 0 && syllables >= letters.min(2) && syllables * 3 >= letters
}

/// Whether the default language found text it could not read: half of the
/// lines or more were not `read` (with confidence by the multilingual
/// recognizer, or as Hangul by the Korean one), and one of them is shaped
/// like a line of text (at least three times as wide as tall). A blank page
/// or a picture has no lines; a table of short cells has no wide unread one.
fn unread(lines: &[Line], read: &[bool]) -> bool {
    let doubtful: Vec<&Line> = lines
        .iter()
        .zip(read)
        .filter(|(_, read)| !**read)
        .map(|(line, _)| line)
        .collect();
    let wide = |line: &&Line| {
        let [left, top, right, bottom] = line.bounds;
        right - left >= 3.0 * (bottom - top)
    };
    !lines.is_empty() && doubtful.len() * 2 >= lines.len() && doubtful.iter().any(wide)
}

/// The result of the reading used: lines below the text score dropped, the
/// rest laid out.
fn finish(
    mut lines: Vec<Line>,
    image: &RgbImage,
    language: &str,
    unread: bool,
) -> Result<OcrResult> {
    lines.retain(|line| line.confidence >= TEXT_SCORE);
    let mut result = super::assemble(lines, image.width(), image.height())?;
    result.language = language.to_owned();
    result.unread = unread;
    Ok(result)
}

/// The models the configured language needs, for `doctor`: the default set
/// for the default language, else the detector, the classifier and the
/// language's recognizer.
pub(crate) fn needed(spelling: &str) -> Result<Vec<&'static models::Model>> {
    match choice(spelling)? {
        Choice::Default => Ok(models::defaults().collect()),
        Choice::Model(name) => [DETECTOR, CLASSIFIER, name]
            .into_iter()
            .map(models::named)
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_map_reference_and_vision_spellings_to_recognizers() {
        assert_eq!(choice("en").unwrap(), Choice::Default);
        for multilingual in [
            "en-us",
            "zh",
            "zh-cn",
            "ch",
            "zh-hans",
            "zh-tw",
            "chinese-cht",
            "zh-hant",
            "ja",
            "japan",
            "ja-jp",
            "fr",
            "fr-fr",
            "german",
            "pt-br",
            "vi",
            "yue-hant",
        ] {
            assert_eq!(
                choice(multilingual).unwrap(),
                Choice::Model(MULTILINGUAL),
                "{multilingual}"
            );
        }
        for (spelling, model) in [
            ("ko", KOREAN),
            ("korean", KOREAN),
            ("ko-kr", KOREAN),
            ("ar", "arabic-ppocrv5-rec-mobile"),
            ("ar-sa", "arabic-ppocrv5-rec-mobile"),
            ("th-th", "th-ppocrv5-rec-mobile"),
            ("ru-ru", "eslav-ppocrv5-rec-mobile"),
            ("uk", "eslav-ppocrv5-rec-mobile"),
            ("cyrillic", "cyrillic-ppocrv5-rec-mobile"),
            ("latin", "latin-ppocrv5-rec-mobile"),
            ("hi-in", "devanagari-ppocrv5-rec-mobile"),
            ("el", "el-ppocrv5-rec-mobile"),
            ("ta", "ta-ppocrv5-rec-mobile"),
            ("te", "te-ppocrv5-rec-mobile"),
        ] {
            assert_eq!(
                choice(spelling).unwrap(),
                Choice::Model(model),
                "{spelling}"
            );
            assert!(models::named(model).is_ok(), "{model}");
        }
        for unknown in ["he", "xx", "auto", "-en", "klingon-x"] {
            let error = choice(unknown).unwrap_err();
            assert!(matches!(error, crate::Error::Unsupported(_)), "{unknown}");
            assert!(error.to_string().contains("zh_tw"), "{error}");
        }
        assert_eq!(needed("en").unwrap().len(), 4);
        assert_eq!(
            needed("ar")
                .unwrap()
                .iter()
                .map(|m| m.name.as_str())
                .collect::<Vec<_>>(),
            [DETECTOR, CLASSIFIER, "arabic-ppocrv5-rec-mobile"]
        );
    }

    fn line(text: &str, confidence: f32, bounds: [f32; 4]) -> Line {
        Line {
            text: text.into(),
            confidence,
            bounds,
            direction: [1.0, 0.0],
        }
    }

    #[test]
    fn a_sure_line_is_confident_and_covers_its_region() {
        let wide = [0.0, 0.0, 500.0, 50.0];
        assert!(sure(&line("The quick brown fox", 0.95, wide)));
        // Doubtful, empty, or a lone period for ten line heights of Hangul.
        assert!(!sure(&line("The quick brown fox", 0.8, wide)));
        assert!(!sure(&line("  ", 0.99, wide)));
        assert!(!sure(&line(".", 0.94, wide)));
        assert!(!sure(&line("abc", 0.99, wide)));
        assert!(sure(&line("abcd", 0.99, wide)));
        // A lone digit in a narrow cell, and text running down the page.
        assert!(sure(&line("7", 0.99, [0.0, 0.0, 30.0, 50.0])));
        assert!(sure(&line("一二三四", 0.99, [0.0, 0.0, 50.0, 200.0])));
        assert!(!sure(&line("一", 0.99, [0.0, 0.0, 50.0, 200.0])));
    }

    #[test]
    fn hangul_readings_replace_only_lines_mostly_of_hangul() {
        assert!(hangul("회의실 예약 시스템이 새로워져서"));
        assert!(hangul("버전 1.3.0을 HTML로"));
        assert!(hangul("수"));
        assert!(!hangul("Local OCR 한"));
        assert!(!hangul("..."));
        assert!(!hangul("本地文字识别"));
    }

    #[test]
    fn weak_korean_fallback_does_not_discard_the_first_reading() {
        let wide = [0.0, 0.0, 500.0, 50.0];
        let sparse = line("漢字", 0.95, wide);
        assert!(!korean_replaces(&sparse, &line("한국어", 0.3, wide)));
        assert!(!korean_replaces(&sparse, &line("한국어", f32::NAN, wide)));
        assert!(korean_replaces(&sparse, &line("한국어", 0.86, wide)));
        let latin = line("A readable English line", 0.86, wide);
        assert!(!korean_replaces(&latin, &line("한국어", 0.7, wide)));
        assert!(!korean_replaces(&sparse, &line("Local OCR 한", 0.99, wide)));
    }

    #[test]
    fn detection_maps_require_one_complete_batch_and_channel() {
        assert!(detection_shape(&[1, 1, 32, 64], 2048, 64, 32));
        assert!(!detection_shape(&[1, 1, 32, 64], 2047, 64, 32));
        assert!(!detection_shape(&[2, 1, 32, 64], 4096, 64, 32));
        assert!(!detection_shape(&[1, 2, 32, 64], 4096, 64, 32));
        assert!(!detection_shape(&[32, 64], 2048, 64, 32));
    }

    #[test]
    fn half_the_lines_unread_and_one_long_is_unread() {
        let wide = [0.0, 0.0, 500.0, 50.0];
        let narrow = [0.0, 0.0, 60.0, 50.0];
        let text = |s: &str| line(s, 0.99, wide);
        assert!(!unread(&[], &[]));
        assert!(!unread(&[text("A line read with confidence")], &[true]));
        assert!(unread(
            &[text("A line"), line("", 0.0, wide)],
            &[true, false]
        ));
        // A line the Korean recognizer read counts as read.
        assert!(!unread(
            &[text("A line"), line("한국어", 0.86, wide)],
            &[true, true]
        ));
        let three = [text("A line"), text("Another line"), line("", 0.0, wide)];
        assert!(!unread(&three, &[true, true, false]));
        // Short cells nobody read do not make the image unread.
        let cells = [line("7", 0.6, narrow), line("", 0.0, narrow)];
        assert!(!unread(&cells, &[false, false]));
    }

    fn words(text: &str) -> Vec<&str> {
        text.split_whitespace().collect()
    }

    /// The authored fixture read with the installed models, as drawn and
    /// upside down, and a blank image. Run with the default models under
    /// `MARKITAI_HOME` (`markitai doctor --fix`); it never downloads.
    #[test]
    #[ignore = "needs the PaddleOCR models installed under MARKITAI_HOME"]
    fn the_installed_models_read_the_english_fixture_either_way_up() {
        for model in models::defaults() {
            assert!(models::present(model), "install {} first", model.name);
        }
        let fixture = include_bytes!("../fixtures/english.png");
        let expected = include_str!("../fixtures/english.txt");
        let rgb = super::super::pixels::upright(fixture).unwrap();
        let result = read(&rgb, "en").unwrap();
        assert_eq!(words(&result.text), words(expected), "{}", result.text);
        assert!(!result.unread && result.confidence > 0.9);
        assert_eq!(result.boxes.len(), 3);
        let turned = image::imageops::rotate180(&rgb);
        assert_eq!(words(&read(&turned, "en").unwrap().text), words(expected));
        let blank = RgbImage::from_pixel(400, 200, image::Rgb([255; 3]));
        let blank = read(&blank, "en").unwrap();
        assert!(blank.text.is_empty() && blank.boxes.is_empty() && !blank.unread);
        // A written language reads with its recognizer alone.
        assert_eq!(read(&rgb, "en-us").unwrap().language, MULTILINGUAL);
    }

    #[test]
    fn planes_are_bgr_and_padded_with_zeros() {
        let image = RgbImage::from_pixel(2, 1, image::Rgb([255, 0, 0]));
        let data = planes(&image, 3, 2);
        assert_eq!(data.len(), 18);
        // Blue plane first: -1 where the image is, 0 for padding.
        assert_eq!(&data[0..6], &[-1.0, -1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&data[12..15], &[1.0, 1.0, 0.0]);
        assert_eq!(resize(&image, 2, 1), image);
        assert_eq!(resize(&image, 4, 2).dimensions(), (4, 2));
    }

    #[test]
    fn helper_threads_stay_within_the_processor_count() {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let first = take_helpers(usize::MAX);
        assert!(first < cores);
        HELPERS.fetch_sub(first, Ordering::AcqRel);
        assert_eq!(take_helpers(0), 0);
    }

    #[test]
    fn a_failed_optional_load_is_kept_and_a_required_one_tried_again() {
        let loads: Loads<u32> = Loads::new();
        let calls = AtomicUsize::new(0);
        let fail = || -> Result<u32> {
            calls.fetch_add(1, Ordering::Relaxed);
            Err(failure("offline"))
        };
        assert!(loads.optional("korean", fail).is_none());
        assert!(loads.optional("korean", fail).is_none());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(loads.get("multilingual", fail).is_err());
        assert!(loads.get("multilingual", fail).is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        // Asked for by name, a model whose optional load failed is tried again.
        assert_eq!(*loads.get("korean", || Ok(7)).unwrap(), 7);
        assert_eq!(*loads.optional("korean", fail).unwrap(), 7);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn a_slow_load_does_not_hold_up_a_loaded_value() {
        use std::sync::mpsc;
        use std::time::Duration;
        let loads: Loads<u32> = Loads::new();
        assert_eq!(*loads.get("multilingual", || Ok(1)).unwrap(), 1);
        let (started, began) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let (answer, answered) = mpsc::channel();
        let loads = &loads;
        std::thread::scope(|scope| {
            scope.spawn(move || {
                loads.optional("korean", || {
                    started.send(()).unwrap();
                    released.recv().unwrap();
                    Err(failure("offline"))
                })
            });
            began.recv().unwrap();
            scope.spawn(move || answer.send(*loads.get("multilingual", || Ok(2)).unwrap()));
            let found = answered.recv_timeout(Duration::from_secs(60));
            release.send(()).unwrap();
            assert_eq!(found, Ok(1), "a loaded value waited for another's load");
        });
    }
}
