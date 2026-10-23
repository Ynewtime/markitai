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
//! Two engines run these models. `tract` is the default. `qppocr` ([`qpp`]) is
//! an experimental opt-in (`MARKITAI_OCR_ENGINE=qppocr`): it detects, orients
//! and reads the lines of the multilingual recognizer's languages in one pass,
//! and tract reads the rest, takes over for a model or a failure qppocr cannot
//! handle, and re-reads doubtful lines of the default language with the
//! Korean recognizer.
//!
//! The default language reads with the multilingual recognizer (Latin
//! script, Chinese, Japanese) and, when that reading looks like another
//! script, with the Korean one too, reusing the detected regions; see
//! [`read_default`].

mod classify;
mod depthwise;
mod detect;
pub(crate) mod models;
mod qpp;
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
/// varies from run to run, with its depthwise convolutions spread over the
/// OCR threads ([`depthwise`]).
fn plan(mut model: InferenceModel, shape: [Option<usize>; 4], name: &str) -> Result<Plan> {
    let dims: TVec<TDim> = shape
        .iter()
        .enumerate()
        .map(|(axis, n)| match n {
            Some(n) => TDim::from(*n as i64),
            None => model.sym(&format!("d{axis}")).into(),
        })
        .collect();
    let mut lap = super::trace::Lap::start();
    let built = model
        .set_input_fact(0, f32::fact(dims).into())
        .and_then(|_| model.into_typed())
        .and_then(|typed| typed.into_decluttered())
        .and_then(|mut typed| {
            depthwise::split(&mut typed)?;
            typed.into_optimized()
        })
        .and_then(|typed| {
            if super::trace::enabled() {
                super::trace::line(format_args!(
                    "optimized {name} in {:.0}ms: {}",
                    lap.ms(),
                    operators(&typed)
                ));
            }
            typed.into_runnable()
        });
    built.map_err(|error| engine_error(&format!("cannot prepare the model {name}"), error))
}

/// The operators of an optimized model with how many of each, for the trace:
/// the matrix products name the kernels tract chose for them.
fn operators(model: &TypedModel) -> String {
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for node in model.nodes() {
        let op = node.op();
        let name = op.name();
        let detail = format!("{op:?}");
        let key = match detail.find("mmm") {
            // The kernel names appear as `fma_mmm_f32_16x6`, `generic_f32_4x4`...
            Some(_) => kernels(&detail),
            None => name.to_string(),
        };
        *counts.entry(key).or_default() += 1;
    }
    counts
        .iter()
        .map(|(name, count)| format!("{name}x{count}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The kernel names in the debug text of a matrix product.
fn kernels(detail: &str) -> String {
    let mut found = Vec::new();
    for word in detail.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if (word.contains("_mmm_") || word.contains("_f32_")) && !found.contains(&word) {
            found.push(word);
        }
    }
    if found.is_empty() {
        "matmul".into()
    } else {
        found.join("+")
    }
}

/// `plan` run on `input` with each operator's time added to the trace's
/// totals.
fn profiled(plan: &Plan, input: Tensor, name: &str) -> TractResult<TVec<TValue>> {
    plan.spawn()?
        .run_plan_with_eval(tvec!(input.into()), |context, state, node, inputs| {
            let start = std::time::Instant::now();
            let outputs = tract_onnx::tract_core::plan::eval(context, state, node, inputs);
            super::trace::operator(name, &node.op().name(), start.elapsed());
            outputs
        })
}

/// Run a plan on one input of `shape` and return its first output's values
/// and shape.
fn run(plan: &Plan, shape: &[usize], data: Vec<f32>, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let input = Tensor::from_shape(shape, &data)
        .map_err(|error| engine_error(&format!("cannot pass an input to {name}"), error))?;
    // from_shape copies the planes; do not retain both allocations during inference.
    drop(data);
    let outputs = if super::trace::enabled() {
        profiled(plan, input, name)
    } else {
        plan.run(tvec!(input.into()))
    }
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

    fn regions(&self, image: &RgbImage, box_threshold: f32) -> Result<(Vec<detect::Region>, bool)> {
        // Bound cold optimization and input planes as well as the inference itself.
        let _turn = DetectionTurn::take();
        let layout = detect::Layout::of(image.width(), image.height());
        let plan = self.plan(layout.width, layout.height)?;
        let shape = [1, 3, layout.height as usize, layout.width as usize];
        let data = detect::input(image, &layout);
        let (map, out) = tract_linalg::multithread::multithread_tract_scope(executor(), || {
            run(&plan, &shape, data, DETECTOR)
        })?;
        if !detection_shape(&out, map.len(), layout.width, layout.height) {
            return Err(failure("detection model returned a map of the wrong size"));
        }
        detect::regions_scoring(
            &map[..layout.width as usize * layout.height as usize],
            &layout,
            box_threshold,
        )
    }
}

/// The one pool of compute threads OCR uses, as many as
/// [`super::thread_budget`]: a page's detection splits its convolutions over
/// it and the lines of every page in flight are read on it, so the threads
/// busy at once never exceed the budget however many pages are in flight, and
/// a page left with the last lines does not read them alone while the others
/// idle. `None` on a single thread.
fn pool() -> Option<&'static Arc<rayon::ThreadPool>> {
    static POOL: std::sync::OnceLock<Option<Arc<rayon::ThreadPool>>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let threads = super::thread_budget();
        if threads < 2 {
            return None;
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|n| format!("markitai-ocr-{n}"))
            .build()
            .ok()
            .map(Arc::new)
    })
    .as_ref()
}

/// tract's matrix kernels on [`pool`]: a page's convolutions split over its
/// rows, with the same results as on one thread.
fn executor() -> tract_linalg::multithread::Executor {
    pool().map_or(tract_linalg::multithread::Executor::SingleThread, |pool| {
        tract_linalg::multithread::Executor::MultiThread(pool.clone())
    })
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

/// The detections running, and those waiting by page number (an image is a
/// first page) and then arrival.
struct Turns {
    running: usize,
    waiting: std::collections::BTreeSet<(usize, u64)>,
    arrivals: u64,
}

static DETECTING: (Mutex<Turns>, std::sync::Condvar) = (
    Mutex::new(Turns {
        running: 0,
        waiting: std::collections::BTreeSet::new(),
        arrivals: 0,
    }),
    std::sync::Condvar::new(),
);

impl DetectionTurn {
    /// The next free turn goes to the waiting page with the lowest number in
    /// its document, then to the first to come: a page of a short document
    /// (the one page of a scan in a batch) waits for the detections running,
    /// not for a long document's later pages queued before it.
    fn take() -> DetectionTurn {
        let page = super::trace::page().unwrap_or(1);
        let (turns, freed) = &DETECTING;
        let mut state = turns.lock().unwrap_or_else(|e| e.into_inner());
        let me = (page, state.arrivals);
        state.arrivals += 1;
        state.waiting.insert(me);
        while state.running >= DETECTIONS_AT_ONCE || state.waiting.first() != Some(&me) {
            state = freed.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        state.waiting.remove(&me);
        state.running += 1;
        // The next waiter may fit too.
        freed.notify_all();
        DetectionTurn
    }
}

impl Drop for DetectionTurn {
    fn drop(&mut self) {
        let (turns, freed) = &DETECTING;
        let mut state = turns.lock().unwrap_or_else(|e| e.into_inner());
        state.running = state.running.saturating_sub(1);
        freed.notify_all();
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

/// The detector, loaded once per process.
struct Engine {
    detector: Detector,
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
    let engine = Arc::new(Engine { detector });
    *slot = Some(engine.clone());
    Ok(engine)
}

static CLASSIFIERS: Loads<Plan> = Loads::new();

/// The direction classifier, loaded once per process.
fn classifier() -> Result<Arc<Plan>> {
    CLASSIFIERS.get(CLASSIFIER, || {
        plan(
            parse(CLASSIFIER, models::Role::Classify)?,
            [
                Some(classify::BATCH),
                Some(3),
                Some(classify::HEIGHT as usize),
                Some(classify::WIDTH as usize),
            ],
            CLASSIFIER,
        )
    })
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
    /// The cut-out was turned a quarter because the region is tall.
    turned: bool,
    /// The classifier found the line upside down. It is read both ways and
    /// the more confident reading kept: the classifier turns over a few
    /// upright lines of ordinary prose (one of 24 rendered 300 DPI pages
    /// lost a line that way).
    flagged: bool,
}

/// `region` of `image`, cut out upright.
fn cut_of(image: &RgbImage, region: &detect::Region) -> Result<Cut> {
    let [tl, tr, _, bl] = region.corners;
    let mut cut = recognize::cut(image, &region.corners)?;
    let mut direction = [tr[0] - tl[0], tr[1] - tl[1]];
    let turned = cut.height() as f32 >= recognize::UPRIGHT_RATIO * cut.width() as f32;
    if turned {
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
    Ok(Cut {
        image: cut,
        bounds,
        direction,
        turned,
        flagged: false,
    })
}

/// The text lines of `image`, cut out upright, and whether it had more than
/// [`detect::MAX_REGIONS`]. The classifier and the recognizer `reader` load
/// on their own threads while the detector reads the image, the first time
/// in a process: optimizing a model takes a tenth of a second or more.
fn cuts(engine: &Engine, image: &RgbImage, reader: &'static str) -> Result<(Vec<Cut>, bool)> {
    let mut lap = super::trace::Lap::start();
    let (detected, classifier) = std::thread::scope(|scope| {
        let classifier = scope.spawn(classifier);
        // A failure is reported when the lines are read with it.
        let reader = scope.spawn(move || recognizer(reader).map(drop));
        let detected = engine.detector.regions(image, detect::BOX_THRESHOLD);
        let _ = reader.join();
        let classifier = classifier
            .join()
            .unwrap_or_else(|_| Err(failure("the classifier failed to load")));
        (detected, classifier)
    });
    let (regions, capped) = detected?;
    let detect_ms = lap.ms();
    let cuts = classified(&*classifier?, image, &regions)?;
    super::trace::line(format_args!(
        "det={detect_ms:.0}ms cls={:.0}ms regions={}",
        lap.ms(),
        regions.len()
    ));
    Ok((cuts, capped))
}

/// `regions` of `image` cut out upright and direction-classified. Batches of
/// lines are cut out and classified side by side.
fn classified(classifier: &Plan, image: &RgbImage, regions: &[detect::Region]) -> Result<Vec<Cut>> {
    let batches = spread(regions.len().div_ceil(classify::BATCH), |batch| {
        let start = batch * classify::BATCH;
        let regions = &regions[start..(start + classify::BATCH).min(regions.len())];
        let mut cuts = regions
            .iter()
            .map(|region| cut_of(image, region))
            .collect::<Result<Vec<Cut>>>()?;
        let size = classify::BATCH * 3 * (classify::HEIGHT * classify::WIDTH) as usize;
        let mut data = Vec::with_capacity(size);
        for cut in &cuts {
            data.extend(classify::input(&cut.image));
        }
        data.resize(size, 0.0);
        let shape = [
            classify::BATCH,
            3,
            classify::HEIGHT as usize,
            classify::WIDTH as usize,
        ];
        let (answers, _) = run(classifier, &shape, data, CLASSIFIER)?;
        let turned = classify::upside_down(&answers, cuts.len())?;
        for (cut, over) in cuts.iter_mut().zip(turned) {
            cut.flagged = over;
        }
        Ok(cuts)
    })?;
    Ok(batches.into_iter().flatten().collect())
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
        marks: Default::default(),
    })
}

/// Read each cut-out line with `recognizer`; one line per cut, in order.
fn read_lines(recognizer: &Recognizer, cuts: &[&Cut]) -> Result<Vec<Line>> {
    // One or two lines (a label, a caption) share the detector's threads;
    // more are read side by side, one thread each.
    if cuts.len() <= 2 {
        return tract_linalg::multithread::multithread_tract_scope(executor(), || {
            cuts.iter().map(|cut| read_cut(recognizer, cut)).collect()
        });
    }
    spread(cuts.len(), |index| read_cut(recognizer, cuts[index]))
}

/// Read single upright text lines (the PDF glyph check's crops) in the
/// language `spelling` names, with its recognizer only: no detection,
/// classification or second reading. The recognizer loads here when no page
/// has needed it yet. One text per image, in order.
pub(super) fn read_line_images(images: &[RgbImage], spelling: &str) -> Result<Vec<String>> {
    let name = match choice(spelling)? {
        Choice::Default => MULTILINGUAL,
        Choice::Model(name) => name,
    };
    models::preflight(&[models::named(name)?])?;
    let recognizer = recognizer(name)?;
    let read = |index: usize| recognizer.read(&images[index]).map(|reading| reading.text);
    if images.len() <= 2 {
        return tract_linalg::multithread::multithread_tract_scope(executor(), || {
            (0..images.len()).map(read).collect()
        });
    }
    spread(images.len(), read)
}

/// `work` of each index below `count`, in order, computed on [`pool`] next
/// to every other page's work (on this thread without a pool). `work` must
/// not wait for another page's work: the pool's threads would all wait.
fn spread<T: Send>(
    count: usize,
    work: impl Fn(usize) -> Result<T> + Send + Sync,
) -> Result<Vec<T>> {
    use rayon::prelude::*;
    match pool().filter(|_| count > 1) {
        Some(pool) => pool.install(|| (0..count).into_par_iter().map(work).collect()),
        None => (0..count).map(work).collect(),
    }
}

/// `work` of each index below `count`, in order, computed on this thread and
/// one more, each taking the next index. For work that takes a
/// [`DetectionTurn`], which would hold up the threads of [`pool`]; the
/// detections themselves use the pool.
fn spread_detections<T: Send>(
    count: usize,
    work: impl Fn(usize) -> Result<T> + Sync,
) -> Result<Vec<T>> {
    let next = AtomicUsize::new(0);
    let worker = || -> Result<Vec<(usize, T)>> {
        let mut done = Vec::new();
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= count {
                return Ok(done);
            }
            done.push((index, work(index)?));
        }
    };
    let results: Vec<Result<Vec<(usize, T)>>> = std::thread::scope(|scope| {
        let helper = (count > 1).then(|| scope.spawn(worker));
        let mut results = vec![worker()];
        if let Some(helper) = helper {
            results.push(
                helper
                    .join()
                    .unwrap_or_else(|_| Err(failure("a detection thread failed"))),
            );
        }
        results
    });
    let mut values: Vec<Option<T>> = (0..count).map(|_| None).collect();
    for result in results {
        for (index, value) in result? {
            values[index] = Some(value);
        }
    }
    values
        .into_iter()
        .map(|value| value.ok_or_else(|| failure("a tile was not searched")))
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

/// Recognize upright RGB pixels in the language `spelling` names. A
/// full-image pass that keeps no text is tried again on overlapping tiles
/// ([`tiled`]), as the reference does.
pub(super) fn read(image: &RgbImage, spelling: &str) -> Result<OcrResult> {
    let choice = choice(spelling)?;
    models::preflight(&needed(spelling)?)?;
    if matches!(choice, Choice::Default | Choice::Model(MULTILINGUAL))
        && let Some(qpp) = qpp::engine()?
        && let Some(found) = qpp.read(image)
    {
        return read_qpp(qpp, image, choice, found);
    }
    let engine = engine()?;
    let reader = match choice {
        Choice::Model(name) => name,
        Choice::Default => MULTILINGUAL,
    };
    let mut lap = super::trace::Lap::start();
    let (cuts, capped) = cuts(&engine, image, reader)?;
    lap.ms();
    let (lines, language, unread) = recognize(choice, &cuts)?;
    super::trace::line(format_args!(
        "engine=tract rec={:.0}ms lines={}",
        lap.ms(),
        cuts.len()
    ));
    if !lines.iter().any(|line| line.confidence >= TEXT_SCORE)
        && let Some(((lines, language, unread), capped)) = tiled(image, choice)?
    {
        let tile_ms = lap.ms();
        let mut result = finish(lines, image, language, unread, TILE_TEXT_SCORE)?;
        result.capped = capped;
        super::trace::line(format_args!("tiled={tile_ms:.0}ms post={:.0}ms", lap.ms()));
        return Ok(result);
    }
    let mut result = finish(lines, image, language, unread, TEXT_SCORE)?;
    result.capped = capped;
    super::trace::line(format_args!("post={:.0}ms", lap.ms()));
    super::trace::report(10);
    Ok(result)
}

/// `image` read by qppocr: its lines, then (for the default language) the
/// Korean pass on the doubtful ones and the tiled retry of a sparse image,
/// both with tract.
fn read_qpp(
    qpp: &qpp::Qpp,
    image: &RgbImage,
    choice: Choice,
    mut found: qpp::Found,
) -> Result<OcrResult> {
    let mut lap = super::trace::Lap::start();
    let capped = found.capped;
    let (lines, language, unread) = match choice {
        Choice::Default => qpp_default(image, found)?,
        Choice::Model(name) => {
            qpp_spaces(image, &mut found);
            (found.lines, name, false)
        }
    };
    let second_ms = lap.ms();
    if !lines.iter().any(|line| line.confidence >= TEXT_SCORE) {
        let retried = match tiled_qpp(image, choice, qpp)? {
            Some(retried) => retried,
            None => tiled(image, choice)?,
        };
        if let Some(((lines, language, unread), capped)) = retried {
            let tile_ms = lap.ms();
            let mut result = finish(lines, image, language, unread, TILE_TEXT_SCORE)?;
            result.capped = capped;
            super::trace::line(format_args!(
                "engine=qppocr second={second_ms:.0}ms tiled={tile_ms:.0}ms post={:.0}ms",
                lap.ms()
            ));
            return Ok(result);
        }
    }
    let mut result = finish(lines, image, language, unread, TEXT_SCORE)?;
    result.capped = capped;
    super::trace::line(format_args!(
        "engine=qppocr second={second_ms:.0}ms post={:.0}ms",
        lap.ms()
    ));
    Ok(result)
}

/// Whether a region is a block, not a line of text: at least 200 pixels and a
/// fifth of the image's shorter side on its short side, and less than three
/// times as long as that.
fn blocky(corners: &[[f32; 2]; 4], (width, height): (u32, u32)) -> bool {
    let side = |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let (one, other) = (side(corners[0], corners[1]), side(corners[0], corners[3]));
    let (short, long) = (one.min(other), one.max(other));
    short >= 200.0 && short >= 0.2 * width.min(height) as f32 && long < 3.0 * short
}

/// The spaces between a Chinese or Japanese letter and a Latin letter or
/// digit that the line's pixels show, which the recognizer leaves out, put
/// into the text of each of qppocr's lines that has such a junction: the same
/// rule as for tract's readings ([`recognize::spaced`]), on the line cut out
/// of `image`, with the characters' places projected onto the line.
fn qpp_spaces(image: &RgbImage, found: &mut qpp::Found) {
    for index in 0..found.lines.len() {
        let characters: Vec<char> = found.lines[index].text.chars().collect();
        let places = &found.centers[index];
        if characters.len() < 2
            || characters.len() != places.len()
            || super::layout::junctions(&characters).is_empty()
        {
            continue;
        }
        let corners = found.corners[index];
        let region = detect::Region {
            corners,
            score: 1.0,
        };
        let Ok(cut) = cut_of(image, &region) else {
            continue;
        };
        let [tl, tr, _, bl] = corners;
        let axis = if cut.turned {
            [bl[0] - tl[0], bl[1] - tl[1]]
        } else {
            [tr[0] - tl[0], tr[1] - tl[1]]
        };
        let length = (axis[0] * axis[0] + axis[1] * axis[1]).max(1.0);
        let width = cut.image.width() as f32;
        let mut centers: Vec<f32> = places
            .iter()
            .map(|p| ((p[0] - tl[0]) * axis[0] + (p[1] - tl[1]) * axis[1]) / length * width)
            .collect();
        // Characters of a line read upside down come in the other direction.
        let mut line = cut.image;
        if centers.first() > centers.last() {
            line = image::imageops::rotate180(&line);
            for center in &mut centers {
                *center = width - *center;
            }
        }
        let reading = recognize::Reading {
            text: found.lines[index].text.clone(),
            confidence: found.lines[index].confidence,
            centers,
        };
        found.lines[index].text = recognize::spaced(&line, &reading, 1.0);
    }
}

/// The default language's lines from qppocr: the doubtful ones are cut out of
/// `image` and read again by [`second_reading`].
fn qpp_default(image: &RgbImage, mut found: qpp::Found) -> Result<Reading> {
    qpp_spaces(image, &mut found);
    let mut doubtful = Vec::new();
    let mut cuts = Vec::new();
    for index in (0..found.lines.len()).filter(|&i| !sure(&found.lines[i])) {
        let corners = found.corners[index];
        // A page-sized block (a spine shadow, a photo) is no line of Hangul.
        if blocky(&corners, image.dimensions()) {
            continue;
        }
        let region = detect::Region {
            corners,
            score: 1.0,
        };
        // A region that cannot be cut out keeps its first reading.
        let Ok(mut cut) = cut_of(image, &region) else {
            continue;
        };
        let side =
            |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        let tall =
            side(corners[0], corners[3]) >= recognize::UPRIGHT_RATIO * side(corners[0], corners[1]);
        // qppocr turned the line over, or turned a tall one the other way:
        // read it both ways.
        cut.flagged = found.flipped[index] || tall;
        doubtful.push(index);
        cuts.push(cut);
    }
    let again: Vec<&Cut> = cuts.iter().collect();
    second_reading(found.lines, &doubtful, &again)
}

/// The lines read, the recognizer that read them, and whether the default
/// language found text it cannot read.
type Reading = (Vec<Line>, &'static str, bool);

/// Every cut read in the chosen language.
fn recognize(choice: Choice, cuts: &[Cut]) -> Result<Reading> {
    match choice {
        Choice::Model(name) => {
            let all: Vec<&Cut> = cuts.iter().collect();
            Ok((read_lines(&*recognizer(name)?, &all)?, name, false))
        }
        Choice::Default => read_default(cuts),
    }
}

/// The reference's detection side: RapidOCR scales an input's shorter side
/// to this. A tile is scaled up toward it (by at most [`TILE_UPSCALE`]), so
/// text a few pixels high on a mostly empty page is large enough to find.
/// When several tiles are searched, two at a time, each is enlarged at most
/// so that the two hold no more pixels than the full-image pass did: the
/// retry then needs about that pass's memory, not twice it.
const TILE_SHORT_SIDE: f32 = 736.0;
const TILE_UPSCALE: f32 = 2.0;
/// The reference's thresholds for its tiled retry: a region, and a line,
/// may be a little less certain than in a full-image pass.
const TILE_BOX_THRESHOLD: f32 = 0.35;
const TILE_TEXT_SCORE: f32 = 0.45;

/// The tiled retry of a sparse image ([`detect::tiles`]): each tile is
/// scaled up and searched for text, its regions are cut from the image itself
/// and read, and a line two overlapping tiles both found is kept once (the
/// more confident reading). `None` when the image is too small to tile.
fn tiled(image: &RgbImage, choice: Choice) -> Result<Option<(Reading, bool)>> {
    let Some((tiles, budget)) = tile_windows(image) else {
        return Ok(None);
    };
    let engine = engine()?;
    let found = spread_detections(tiles.len(), |index| {
        let [x0, y0, x1, y1] = tiles[index];
        let scaled = scaled_tile(image, tiles[index], budget);
        let (found, more) = engine.detector.regions(&scaled, TILE_BOX_THRESHOLD)?;
        let (scale_x, scale_y) = (
            scaled.width() as f32 / (x1 - x0) as f32,
            scaled.height() as f32 / (y1 - y0) as f32,
        );
        let regions: Vec<detect::Region> = found
            .into_iter()
            .map(|region| detect::Region {
                corners: region
                    .corners
                    .map(|[x, y]| [x / scale_x + x0 as f32, y / scale_y + y0 as f32]),
                score: region.score,
            })
            .collect();
        Ok((regions, more))
    })?;
    let capped = found.iter().any(|(_, more)| *more);
    let regions: Vec<detect::Region> = found.into_iter().flat_map(|(regions, _)| regions).collect();
    if regions.is_empty() {
        return Ok(None);
    }
    let cuts = classified(&*classifier()?, image, &regions)?;
    let (lines, language, unread) = recognize(choice, &cuts)?;
    Ok(Some(((distinct(lines), language, unread), capped)))
}

/// The tiles of a sparse image worth searching (those with a sharp edge: a
/// gradient or a blank margin holds no text), and by how much each may be
/// enlarged; `None` when there are none.
fn tile_windows(image: &RgbImage) -> Option<(Vec<[u32; 4]>, f32)> {
    let tiles: Vec<[u32; 4]> = detect::tiles(image.width(), image.height())
        .into_iter()
        .filter(|tile| detect::has_edges(image, *tile))
        .collect();
    if tiles.is_empty() {
        return None;
    }
    let budget = if tiles.len() > 1 {
        let full = detect::Layout::of(image.width(), image.height());
        let window = tiles
            .iter()
            .map(|[x0, y0, x1, y1]| f64::from(x1 - x0) * f64::from(y1 - y0))
            .fold(1.0, f64::max);
        (f64::from(full.width) * f64::from(full.height) / (2.0 * window)).sqrt() as f32
    } else {
        f32::INFINITY
    };
    Some((tiles, budget))
}

/// `tile` of `image`, enlarged toward [`TILE_SHORT_SIDE`] by at most
/// [`TILE_UPSCALE`] and `budget`.
fn scaled_tile(image: &RgbImage, [x0, y0, x1, y1]: [u32; 4], budget: f32) -> RgbImage {
    let tile = image::imageops::crop_imm(image, x0, y0, x1 - x0, y1 - y0).to_image();
    let short = tile.width().min(tile.height()).max(1) as f32;
    let factor = (TILE_SHORT_SIDE / short)
        .min(TILE_UPSCALE)
        .min(budget)
        .max(1.0);
    if factor > 1.0 {
        let side = |n: u32| ((n as f32 * factor).round() as u32).max(1);
        resize(&tile, side(tile.width()), side(tile.height()))
    } else {
        tile
    }
}

/// [`tiled`] read by qppocr, tile after tile, with no tract model loaded.
/// `None` when it cannot (a tile it fails on): the caller reads with tract.
fn tiled_qpp(
    image: &RgbImage,
    choice: Choice,
    qpp: &qpp::Qpp,
) -> Result<Option<Option<(Reading, bool)>>> {
    let Some((tiles, budget)) = tile_windows(image) else {
        return Ok(Some(None));
    };
    let mut found = qpp::Found::empty();
    for tile in tiles {
        let [x0, y0, x1, y1] = tile;
        let scaled = scaled_tile(image, tile, budget);
        let Some(part) = qpp.read(&scaled) else {
            return Ok(None);
        };
        let (scale_x, scale_y) = (
            scaled.width() as f32 / (x1 - x0) as f32,
            scaled.height() as f32 / (y1 - y0) as f32,
        );
        found.extend_mapped(
            part,
            |[x, y]| [x / scale_x + x0 as f32, y / scale_y + y0 as f32],
            image.dimensions(),
        );
    }
    if found.lines.is_empty() {
        return Ok(Some(None));
    }
    let capped = found.capped;
    // A line two overlapping tiles both found is kept once.
    let kept = distinct_indices(&found.lines);
    let mut keep = vec![false; found.lines.len()];
    for index in kept {
        keep[index] = true;
    }
    let mut flags = keep.iter();
    found.lines.retain(|_| *flags.next().unwrap_or(&false));
    let mut flags = keep.iter();
    found.corners.retain(|_| *flags.next().unwrap_or(&false));
    let mut flags = keep.iter();
    found.centers.retain(|_| *flags.next().unwrap_or(&false));
    let mut flags = keep.iter();
    found.flipped.retain(|_| *flags.next().unwrap_or(&false));
    let reading = match choice {
        Choice::Default => qpp_default(image, found)?,
        Choice::Model(name) => {
            qpp_spaces(image, &mut found);
            (found.lines, name, false)
        }
    };
    Ok(Some(Some((reading, capped))))
}

/// `lines` without repeats: a line whose text matches a kept one (case and
/// spacing aside) and overlaps it by a third or more of the smaller box is
/// the same line, found again by an overlapping tile.
fn distinct(lines: Vec<Line>) -> Vec<Line> {
    let kept = distinct_indices(&lines);
    let mut lines: Vec<Option<Line>> = lines.into_iter().map(Some).collect();
    kept.into_iter()
        .filter_map(|index| lines[index].take())
        .collect()
}

/// The indices of the lines [`distinct`] keeps, in order.
fn distinct_indices(lines: &[Line]) -> Vec<usize> {
    let key = |line: &Line| {
        line.text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    let overlap = |a: [f32; 4], b: [f32; 4]| {
        let width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
        let height = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
        let area = |r: [f32; 4]| ((r[2] - r[0]) * (r[3] - r[1])).max(1.0);
        width * height / area(a).min(area(b))
    };
    let mut kept: Vec<usize> = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let text = key(line);
        match kept.iter_mut().find(|other| {
            let other = &lines[**other];
            key(other) == text && overlap(other.bounds, line.bounds) >= 0.35
        }) {
            Some(other) if line.confidence > lines[*other].confidence => *other = index,
            Some(_) => {}
            None => kept.push(index),
        }
    }
    kept
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
fn read_default(cuts: &[Cut]) -> Result<Reading> {
    let all: Vec<&Cut> = cuts.iter().collect();
    let lines = read_lines(&*recognizer(MULTILINGUAL)?, &all)?;
    let doubtful: Vec<usize> = (0..lines.len()).filter(|&i| !sure(&lines[i])).collect();
    let again: Vec<&Cut> = doubtful.iter().map(|&i| &cuts[i]).collect();
    second_reading(lines, &doubtful, &again)
}

/// The rest of [`read_default`] after the multilingual recognizer read
/// `lines`: the `doubtful` ones are read again, from the cut-outs `again`
/// (one per doubtful line), with the Korean recognizer.
fn second_reading(mut lines: Vec<Line>, doubtful: &[usize], again: &[&Cut]) -> Result<Reading> {
    // Lines the first reading was not sure of, whether or not they could be
    // cut out again.
    let unsure = lines.iter().filter(|line| !sure(line)).count();
    if unsure == 0 {
        return Ok((lines, MULTILINGUAL, false));
    }
    // Lines read with confidence by either recognizer.
    let mut read: Vec<bool> = lines.iter().map(sure).collect();
    let mut korean_lines = 0;
    if !again.is_empty()
        && let Some(korean) = optional_recognizer(KOREAN)
        && let Ok(readings) = read_lines(&korean, again)
    {
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

/// The result of the reading used: lines below `score` dropped, the rest
/// laid out.
fn finish(
    mut lines: Vec<Line>,
    image: &RgbImage,
    language: &str,
    unread: bool,
    score: f32,
) -> Result<OcrResult> {
    lines.retain(|line| line.confidence >= score);
    let mut result = super::assemble(lines, image.width(), image.height())?;
    result.language = language.to_owned();
    result.unread = unread;
    Ok(result)
}

/// The engine this process reads with (`qppocr`, which falls back to tract, or
/// `tract`), for `doctor`.
pub(crate) fn engine_name() -> &'static str {
    qpp::planned()
}

/// Load what reading `spelling` takes before the first image needs it: the
/// verified models, and the detector's plans for images of `sizes` (pages
/// rendered for OCR). A long-running process (`markitai serve`) then does
/// not make its first OCR job wait for them. Never downloads: a model that
/// is not installed is an error, before anything loads. qppocr is not
/// loaded.
pub(super) fn warm(spelling: &str, sizes: &[(u32, u32)]) -> Result<()> {
    let choice = choice(spelling)?;
    for model in needed(spelling)? {
        if models::observation(model).0 != crate::ocr::LocalOcrModelState::Ready {
            return Err(failure(&format!(
                "the model {} is not installed",
                model.name
            )));
        }
    }
    let engine = engine()?;
    classifier()?;
    match choice {
        Choice::Default => {
            recognizer(MULTILINGUAL)?;
            optional_recognizer(KOREAN);
        }
        Choice::Model(name) => {
            recognizer(name)?;
        }
    }
    for &(width, height) in sizes {
        let layout = detect::Layout::of(width, height);
        engine.detector.plan(layout.width, layout.height)?;
    }
    Ok(())
}

/// The models the configured language needs, for `doctor`: the default set
/// for the default language, else the detector, the classifier and the
/// language's recognizer.
pub(crate) fn needed(spelling: &str) -> Result<Vec<&'static models::Model>> {
    match choice(spelling)? {
        Choice::Default => {
            let mut needed: Vec<_> = models::defaults().collect();
            if qpp::planned() == "qppocr" {
                needed.push(models::named(qpp::CLASSIFIER)?);
            }
            Ok(needed)
        }
        Choice::Model(name) => {
            let mut needed = vec![DETECTOR, CLASSIFIER, name];
            if name == MULTILINGUAL && qpp::planned() == "qppocr" {
                needed.push(qpp::CLASSIFIER);
            }
            needed.into_iter().map(models::named).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With every turn taken, the next free one goes to the waiting page with
    /// the lowest number, not to the first to wait: a short document's first
    /// page goes before a long document's fifth.
    #[test]
    fn a_free_detection_turn_goes_to_the_earliest_page_waiting() {
        let waiting = || {
            DETECTING
                .0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .waiting
                .len()
        };
        let wait_for = |count: usize| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while waiting() < count {
                assert!(
                    std::time::Instant::now() < deadline,
                    "waiters did not queue"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        };
        // Other tests may detect meanwhile: count the waiters already there.
        let mut held: Vec<DetectionTurn> = (0..DETECTIONS_AT_ONCE)
            .map(|_| DetectionTurn::take())
            .collect();
        let before = waiting();
        let (order, turns) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel::<()>();
        let released = Arc::new(Mutex::new(released));
        let waiter = |page: usize| {
            let order = order.clone();
            let released = released.clone();
            std::thread::spawn(move || {
                super::super::trace::set_page(page);
                let _turn = DetectionTurn::take();
                order.send(page).unwrap();
                let _ = released.lock().unwrap().recv();
            })
        };
        let fifth = waiter(5);
        wait_for(before + 1);
        let first = waiter(1);
        wait_for(before + 2);
        // One turn frees: the first page takes it, though it came second.
        held.pop();
        assert_eq!(turns.recv().unwrap(), 1);
        held.clear();
        assert_eq!(turns.recv().unwrap(), 5);
        release.send(()).unwrap();
        release.send(()).unwrap();
        fifth.join().unwrap();
        first.join().unwrap();
    }

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
            marks: Default::default(),
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

    /// qppocr itself, whatever engine the process defaults to: the English
    /// fixture upright and turned over, and the same spacing as tract's.
    #[test]
    #[ignore = "needs the PaddleOCR models installed under MARKITAI_HOME"]
    fn qppocr_reads_the_english_fixture_either_way_up() {
        for model in models::defaults() {
            assert!(models::present(model), "install {} first", model.name);
        }
        assert!(
            models::present(models::named(qpp::CLASSIFIER).unwrap()),
            "install the qppocr classifier first"
        );
        let qpp = qpp::start().unwrap();
        let fixture = include_bytes!("../fixtures/english.png");
        let expected = include_str!("../fixtures/english.txt");
        let rgb = super::super::pixels::upright(fixture).unwrap();
        for image in [rgb.clone(), image::imageops::rotate180(&rgb)] {
            let found = qpp.read(&image).unwrap();
            let (lines, language, unread) = qpp_default(&image, found).unwrap();
            let result = finish(lines, &image, language, unread, TEXT_SCORE).unwrap();
            assert_eq!(words(&result.text), words(expected), "{}", result.text);
            assert!(!result.unread && result.confidence > 0.9);
            assert_eq!(result.boxes.len(), 3);
        }
    }

    /// A caption a few pixels high on a large, mostly empty image (white ink
    /// on a dark gradient, like the corpus's `sample.jpg`): the full-image
    /// pass finds nothing, the tiled retry reads it, as the reference does.
    #[test]
    #[ignore = "needs the PaddleOCR models installed under MARKITAI_HOME"]
    fn a_small_caption_on_a_mostly_empty_image_is_read_on_tiles() {
        for model in models::defaults() {
            assert!(models::present(model), "install {} first", model.name);
        }
        let fixture =
            super::super::pixels::upright(include_bytes!("../fixtures/english.png")).unwrap();
        let expected = include_str!("../fixtures/english.txt");
        let small = |n: u32| (n as f32 * 0.12).round() as u32;
        let caption = image::imageops::resize(
            &fixture,
            small(fixture.width()),
            small(fixture.height()),
            image::imageops::FilterType::Triangle,
        );
        let mut page = RgbImage::from_fn(1024, 768, |x, y| {
            image::Rgb([
                (229.5 * x as f32 / 1023.0) as u8,
                (30.0 + 120.0 * y as f32 / 767.0) as u8,
                (60.0 + 120.0 * (x + y) as f32 / 1790.0) as u8,
            ])
        });
        // The fixture's dark letters become white ink over the gradient.
        for (x, y, pixel) in caption.enumerate_pixels() {
            let ink = 1.0 - pixel.0.iter().map(|&c| f32::from(c)).sum::<f32>() / 765.0;
            let target = page.get_pixel_mut(40 + x, 40 + y);
            for channel in &mut target.0 {
                *channel = (f32::from(*channel) * (1.0 - ink) + 255.0 * ink).round() as u8;
            }
        }
        let (cuts, _) = cuts(&engine().unwrap(), &page, MULTILINGUAL).unwrap();
        let (lines, _, _) = recognize(Choice::Default, &cuts).unwrap();
        assert!(
            !lines.iter().any(|line| line.confidence >= TEXT_SCORE),
            "the full-image pass already reads the caption, so this does not test the retry"
        );
        let result = read(&page, "en").unwrap();
        assert_eq!(words(&result.text), words(expected), "{}", result.text);
        assert!(!result.unread);
    }

    #[test]
    fn a_page_sized_region_is_a_block_and_a_line_is_not() {
        let rectangle = |w: f32, h: f32| [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
        assert!(blocky(&rectangle(875.0, 1241.0), (875, 1241)));
        assert!(!blocky(&rectangle(700.0, 40.0), (875, 1241)));
        // A tall strip of a line, a small image that is one line.
        assert!(!blocky(&rectangle(40.0, 700.0), (875, 1241)));
        assert!(!blocky(&rectangle(300.0, 60.0), (300, 60)));
    }

    #[test]
    fn a_line_two_overlapping_tiles_found_is_kept_once() {
        let kept = distinct(vec![
            line("Total  Due", 0.8, [10.0, 10.0, 100.0, 20.0]),
            // The same line found again by the next tile, more confidently.
            line("total due", 0.95, [40.0, 10.0, 100.0, 20.0]),
            // The same words elsewhere on the page are another line.
            line("Total Due", 0.7, [10.0, 300.0, 100.0, 310.0]),
            line("Other", 0.6, [10.0, 10.0, 100.0, 20.0]),
        ]);
        let texts: Vec<(&str, f32)> = kept
            .iter()
            .map(|line| (line.text.as_str(), line.confidence))
            .collect();
        assert_eq!(
            texts,
            [("total due", 0.95), ("Total Due", 0.7), ("Other", 0.6)]
        );
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
    fn lines_are_read_in_order_on_the_pool_of_the_thread_budget() {
        let names = Mutex::new(std::collections::HashSet::new());
        let read = spread(64, |index| {
            names
                .lock()
                .unwrap()
                .insert(std::thread::current().name().map(str::to_owned));
            std::thread::sleep(std::time::Duration::from_millis(2));
            Ok(index)
        })
        .unwrap();
        assert_eq!(read, (0..64).collect::<Vec<_>>());
        let threads = names.lock().unwrap().len();
        assert!(threads <= super::super::thread_budget().max(1), "{threads}");
    }

    #[test]
    fn a_failed_line_fails_the_page() {
        let failed = spread(8, |index| {
            if index == 5 {
                Err(failure("no"))
            } else {
                Ok(index)
            }
        });
        assert!(failed.is_err());
        let failed = spread_detections(8, |index| {
            if index == 5 {
                Err(failure("no"))
            } else {
                Ok(index)
            }
        });
        assert!(failed.is_err());
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
