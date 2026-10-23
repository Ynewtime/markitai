//! Local, bounded OCR. Platform engines receive pixels, never source paths.
//!
//! Every platform reads with the portable engine (`paddle`) by default:
//! PaddleOCR models run in process by a pure-Rust inference engine. macOS
//! also has the system's Vision framework (`vision`): `MARKITAI_OCR_BACKEND`
//! (`vision` or `paddle`) chooses one, and Vision reads when the portable
//! engine's models cannot be downloaded. (Vision's first use of a binary
//! compiles its models for some 23 seconds, and its default reading of a page
//! of Chinese, Japanese and Korean text lost lines, table rows and characters
//! that the portable engine reads: see docs/ocr.md.) A macOS build without the
//! `portable-media` feature has Vision only.

use crate::{Error, Result};
use serde_json::Value;

#[cfg(any(target_os = "macos", test))]
mod auto;
mod cjk;
mod layout;
#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
pub(crate) mod paddle;
mod page;
mod pixels;
pub(crate) mod trace;
#[cfg(target_os = "macos")]
mod vision;

#[derive(Debug, Default)]
pub(crate) struct OcrResult {
    pub text: String,
    /// Mean confidence of nonempty recognized lines.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub confidence: f32,
    /// Pixel rectangles [left, top, right, bottom], after orientation correction.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub boxes: Vec<[f32; 4]>,
    /// How many times the image was enlarged for the reading used; 1 when it
    /// was read at its own size. Only macOS tests read it.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub scale: f32,
    /// The Vision language, or the portable engine's recognizer, of the
    /// reading used. Only tests read it.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub language: String,
    /// Set only under the default language: its readings found text lines
    /// and none could read them. [`unread_warning`] says so.
    pub unread: bool,
    /// The portable engine found more text regions than it reads and kept
    /// the first ones from the top. [`capped_warning`] says so.
    pub capped: bool,
    /// `text` as blocks, with its headings' sizes and its pictures' places.
    pub blocks: Vec<Block>,
    /// The pictures cut out of a PDF page ([`recognize_page`]).
    pub figures: Vec<Figure>,
    /// The recognized lines in the order `text` reads them, with where they
    /// are; none for an image read turned.
    pub lines: Vec<Recognized>,
    /// The image's width and height in pixels.
    pub size: (u32, u32),
    /// The lines as the recognizer read them, for assembling them again
    /// with the image's pixels.
    raw: Vec<Line>,
}

impl OcrResult {
    /// A reading of `text` alone, as one block (none when empty).
    #[cfg(test)]
    pub(crate) fn of_text(text: String) -> Self {
        Self {
            blocks: if text.is_empty() {
                Vec::new()
            } else {
                vec![Block::Text(text.clone())]
            },
            text,
            confidence: 1.0,
            scale: 1.0,
            language: "test".into(),
            ..Self::default()
        }
    }
}

/// Text regions the portable engine reads per image, at most; specks and
/// faint regions do not count.
pub(crate) const MAX_TEXT_REGIONS: usize = 1000;

/// The warning for an image or page with more text regions than the portable
/// engine reads ([`MAX_TEXT_REGIONS`]); `subject` names it.
pub(crate) fn capped_warning(subject: &str) -> String {
    format!(
        "Local OCR read only the first {} text regions of {subject}, from the top; the text \
         below them is missing.",
        thousands(MAX_TEXT_REGIONS)
    )
}

/// `n` with a comma between each group of three digits: 1000 is "1,000".
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// The warning for an image or page that the default language policy could
/// not read; `subject` names it ("Local OCR could not read ...").
#[cfg(all(target_os = "macos", not(feature = "portable-media")))]
pub(crate) fn unread_warning(subject: &str) -> String {
    let languages = vision::languages();
    unread_message(subject, &languages)
}

/// The warning for an image or page that the default language policy could
/// not read; `subject` names it ("Local OCR could not read ...").
#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
pub(crate) fn unread_warning(subject: &str) -> String {
    #[cfg(target_os = "macos")]
    if selected().is_ok_and(|backend| backend == Backend::Vision) {
        return unread_message(subject, &vision::languages());
    }
    paddle_unread_message(subject)
}

/// [`unread_warning`] for the portable engine.
#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
fn paddle_unread_message(subject: &str) -> String {
    format!(
        "Local OCR could not read {subject}: the default reading (Latin script, Chinese and \
         Japanese) could not confidently read all the text. Possible causes include an \
         unsupported script, low image quality, or an unavailable optional Korean model, so \
         only the lines read with confidence are kept. Set ocr.lang to the language of the \
         text, one that the portable OCR models read: ar (Arabic, Persian, Urdu), th, el, ru \
         (also uk and be), cyrillic, hi (Devanagari), ta, te, ko or latin. Run `markitai doctor \
         --fix` to check and repair the required models."
    )
}

/// [`unread_warning`] for a recognizer that reads `languages`.
#[cfg(any(target_os = "macos", test))]
fn unread_message(subject: &str, languages: &[String]) -> String {
    let mut message = format!(
        "Local OCR could not read {subject}: it looks like text in a script that the default \
         English reading does not cover, and Chinese, Japanese and Korean readings found nothing \
         better, so only the lines read with confidence are kept. Set ocr.lang to the language \
         of the text"
    );
    if languages.is_empty() {
        message.push('.');
    } else {
        message.push_str(&format!(
            ", one that this system's Vision recognizer reads: {}.",
            languages.join(", ")
        ));
    }
    message
}

#[cfg(all(target_os = "macos", not(feature = "portable-media")))]
pub(crate) fn available() -> bool {
    objc2::available!(macos = 11.0)
}

#[cfg(all(target_os = "macos", not(feature = "portable-media")))]
pub(crate) fn backend() -> &'static str {
    if available() { "vision" } else { "unavailable" }
}

#[cfg(all(target_os = "macos", not(feature = "portable-media")))]
pub(crate) fn recognize(bytes: &[u8], cfg: &Value) -> Result<OcrResult> {
    if !available() {
        return Err(Error::Unsupported(
            "Local OCR requires macOS 11 or later; no local OCR backend is available on this platform"
                .into(),
        ));
    }
    let spelling = configured(cfg)?;
    let image = pixels::upright(bytes)?;
    with_pixels(
        read(&pixels::prepare_rgb(image.clone())?, &spelling),
        &image,
        false,
    )
}

/// Recognize upright RGB pixels already composited over white by a renderer.
#[cfg(all(target_os = "macos", not(feature = "portable-media")))]
pub(crate) fn recognize_rgb(image: image::RgbImage, cfg: &Value) -> Result<OcrResult> {
    recognize_pixels(image, cfg, false)
}

/// [`recognize_rgb`] for a rendered PDF page: its pictures are also cut out
/// ([`OcrResult::figures`]) and their labels left out of its text.
#[cfg(all(target_os = "macos", not(feature = "portable-media")))]
pub(crate) fn recognize_page(image: image::RgbImage, cfg: &Value) -> Result<OcrResult> {
    recognize_pixels(image, cfg, true)
}

#[cfg(all(target_os = "macos", not(feature = "portable-media")))]
fn recognize_pixels(image: image::RgbImage, cfg: &Value, figures: bool) -> Result<OcrResult> {
    if !available() {
        return Err(Error::Unsupported(
            "Local OCR requires macOS 11 or later; no local OCR backend is available on this platform"
                .into(),
        ));
    }
    let spelling = configured(cfg)?;
    with_pixels(
        read(&pixels::prepare_rgb(image.clone())?, &spelling),
        &image,
        figures,
    )
}

/// Read single upright text lines, one text per image in order: the crops
/// of the PDF glyph check (`formats::pdf::glyph_check`). The portable engine
/// reads them with its recognizer only; Vision recognizes each crop.
#[cfg(all(target_os = "macos", not(feature = "portable-media")))]
pub(crate) fn read_lines(images: Vec<image::RgbImage>, cfg: &Value) -> Result<Vec<String>> {
    images
        .into_iter()
        .map(|image| recognize_rgb(image, cfg).map(|result| result.text))
        .collect()
}

/// Integrity and path safety observed without loading an ONNX graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalOcrModelState {
    Missing,
    Ready,
    Corrupt,
    Unsafe,
}

/// A model file the portable OCR engine reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalOcrModel {
    pub name: String,
    pub path: std::path::PathBuf,
    pub bytes: u64,
    /// Compatibility projection: true only when state is Ready.
    pub installed: bool,
    pub state: LocalOcrModelState,
    /// The concrete damaged file or unsafe path reason, when applicable.
    pub detail: Option<String>,
}

impl LocalOcrModel {
    pub fn new(
        name: String,
        path: std::path::PathBuf,
        bytes: u64,
        state: LocalOcrModelState,
        detail: Option<String>,
    ) -> Self {
        Self {
            name,
            path,
            bytes,
            installed: state == LocalOcrModelState::Ready,
            state,
            detail,
        }
    }
}

/// The portable engine this process reads with: `qppocr` (tract takes over
/// where it cannot read) or `tract`.
pub(crate) fn portable_engine() -> &'static str {
    #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
    {
        paddle::engine_name()
    }
    #[cfg(not(any(not(target_os = "macos"), feature = "portable-media")))]
    {
        "tract"
    }
}

/// The model files the portable engine needs for `cfg`'s `ocr.lang`, or
/// `None` when this process reads with another engine or has none. Never
/// downloads or creates anything.
pub(crate) fn portable_models(cfg: &Value) -> Result<Option<Vec<LocalOcrModel>>> {
    #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
    if selected().is_ok_and(|backend| backend == Backend::Paddle) {
        let models = paddle::needed(&configured(cfg)?)?;
        return Ok(Some(
            models
                .into_iter()
                .map(|model| {
                    let (state, detail) = paddle::models::observation(model);
                    LocalOcrModel::new(
                        model.name.clone(),
                        paddle::models::path(model),
                        model.bytes,
                        state,
                        detail,
                    )
                })
                .collect(),
        ));
    }
    let _ = cfg;
    Ok(None)
}

/// Explicitly repair missing or safely damaged [`portable_models`] from their official
/// mirror, each verified by size and SHA-256; returns the files installed.
pub(crate) fn install_portable_models(cfg: &Value) -> Result<Vec<std::path::PathBuf>> {
    #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
    if selected().is_ok_and(|backend| backend == Backend::Paddle) {
        let models = paddle::needed(&configured(cfg)?)?;
        return paddle::models::install(&models, false);
    }
    let _ = cfg;
    Err(Error::Unsupported(
        "This process does not read with the portable OCR engine; it needs no model files".into(),
    ))
}

/// The sizes of a page the PDF reader renders for OCR at 150 DPI: A4 and
/// Letter, upright.
#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
const PAGE_SIZES: [(u32, u32); 2] = [(1240, 1755), (1275, 1650)];

/// Load the portable engine for `cfg`'s language and prepare its detection
/// for [`PAGE_SIZES`] (see [`paddle::warm`]); whether it is ready.
pub(crate) fn warm(cfg: &Value) -> bool {
    #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
    if selected().is_ok_and(|backend| backend == Backend::Paddle) {
        return configured(cfg).is_ok_and(|spelling| paddle::warm(&spelling, &PAGE_SIZES).is_ok());
    }
    let _ = cfg;
    false
}

/// The engines a build can read with.
#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    #[cfg(target_os = "macos")]
    Vision,
    Paddle,
}

/// The environment variable that sets [`thread_budget`].
const THREADS_VARIABLE: &str = "MARKITAI_OCR_THREADS";

/// Threads local OCR may keep busy at once, all pages together: the
/// processors this process may run on, or `MARKITAI_OCR_THREADS` (a positive
/// number) when set. Pages in flight share these threads instead of each
/// bringing its own, so a busy machine is not oversubscribed.
pub(crate) fn thread_budget() -> usize {
    static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        budget_from(
            std::env::var(THREADS_VARIABLE).ok().as_deref(),
            std::thread::available_parallelism().map_or(1, |n| n.get()),
        )
    })
}

fn budget_from(requested: Option<&str>, processors: usize) -> usize {
    requested
        .and_then(|text| text.trim().parse::<usize>().ok())
        .filter(|n| (1..=1024).contains(n))
        .unwrap_or(processors)
}

/// The environment variable that chooses an engine.
#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
const BACKEND_VARIABLE: &str = "MARKITAI_OCR_BACKEND";

/// The engine this process reads with: `MARKITAI_OCR_BACKEND` when set, else
/// the portable engine.
#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
fn selected() -> Result<Backend> {
    let requested = std::env::var(BACKEND_VARIABLE).unwrap_or_default();
    #[cfg(target_os = "macos")]
    let vision = objc2::available!(macos = 11.0);
    match requested.trim().to_ascii_lowercase().as_str() {
        "" => Ok(Backend::Paddle),
        "paddle" => Ok(Backend::Paddle),
        "vision" => {
            #[cfg(target_os = "macos")]
            if vision {
                return Ok(Backend::Vision);
            }
            Err(Error::Unsupported(format!(
                "{BACKEND_VARIABLE}=vision selects macOS Vision, which needs macOS 11 or later; \
                 unset it or choose paddle"
            )))
        }
        _ => Err(Error::Config(format!(
            "{BACKEND_VARIABLE} must be vision or paddle"
        ))),
    }
}

/// Whether a local OCR engine is built and chosen. The portable engine's
/// models may still need a download (`paddle::models`).
#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
pub(crate) fn available() -> bool {
    selected().is_ok()
}

#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
pub(crate) fn backend() -> &'static str {
    match selected() {
        #[cfg(target_os = "macos")]
        Ok(Backend::Vision) => "vision",
        Ok(Backend::Paddle) => "paddle",
        Err(_) => "unavailable",
    }
}

#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
pub(crate) fn recognize(bytes: &[u8], cfg: &Value) -> Result<OcrResult> {
    let backend = selected()?;
    let spelling = configured(cfg)?;
    let image = pixels::upright(bytes)?;
    let read = match backend {
        #[cfg(target_os = "macos")]
        Backend::Vision => read(&pixels::prepare_rgb(image.clone())?, &spelling),
        Backend::Paddle => {
            let read_models = paddle::read(&image, &spelling);
            #[cfg(target_os = "macos")]
            if let Err(error) = &read_models
                && vision_instead(error)
            {
                return with_pixels(
                    read(&pixels::prepare_rgb(image.clone())?, &spelling),
                    &image,
                    false,
                );
            }
            read_models
        }
    };
    with_pixels(read, &image, false)
}

/// Whether Vision should read in place of the portable engine, which is the
/// default on macOS: only when its models could not be downloaded, the engine
/// was not chosen by the user and Vision works here.
#[cfg(all(target_os = "macos", feature = "portable-media"))]
fn vision_instead(error: &Error) -> bool {
    static TOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let download_failed = matches!(error, Error::Unsupported(message)
        if message.starts_with("Local OCR download of"));
    let chosen = std::env::var(BACKEND_VARIABLE).is_ok_and(|value| !value.trim().is_empty());
    let works = objc2::available!(macos = 11.0) && !crate::system_frameworks::translated();
    if !download_failed || chosen || !works {
        return false;
    }
    if !TOLD.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "Local OCR: the portable engine's models could not be downloaded; reading with macOS Vision instead \
             (the first Vision use of this program takes about 20 seconds). `markitai doctor --fix` installs the models."
        );
    }
    true
}

/// Recognize upright RGB pixels already composited over white by a renderer.
#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
pub(crate) fn recognize_rgb(image: image::RgbImage, cfg: &Value) -> Result<OcrResult> {
    recognize_pixels(image, cfg, false)
}

/// [`recognize_rgb`] for a rendered PDF page: its pictures are also cut out
/// ([`OcrResult::figures`]) and their labels left out of its text.
#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
pub(crate) fn recognize_page(image: image::RgbImage, cfg: &Value) -> Result<OcrResult> {
    recognize_pixels(image, cfg, true)
}

#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
fn recognize_pixels(image: image::RgbImage, cfg: &Value, figures: bool) -> Result<OcrResult> {
    let backend = selected()?;
    let spelling = configured(cfg)?;
    let read = match backend {
        #[cfg(target_os = "macos")]
        Backend::Vision => read(&pixels::prepare_rgb(image.clone())?, &spelling),
        Backend::Paddle => {
            pixels::check_rgb(&image)?;
            let read_models = paddle::read(&image, &spelling);
            #[cfg(target_os = "macos")]
            if let Err(error) = &read_models
                && vision_instead(error)
            {
                return with_pixels(
                    read(&pixels::prepare_rgb(image.clone())?, &spelling),
                    &image,
                    figures,
                );
            }
            read_models
        }
    };
    with_pixels(read, &image, figures)
}

/// A reading assembled again with the pixels it was read from: bold and
/// italic words, the bullets the recognizer dropped, ruled tables and, with
/// `figures`, the pictures cut out. The reading's own verdicts stay.
fn with_pixels(
    read: Result<OcrResult>,
    image: &image::RgbImage,
    figures: bool,
) -> Result<OcrResult> {
    let read = read?;
    if read.raw.is_empty() {
        return Ok(read);
    }
    let (width, height) = image.dimensions();
    let mut lap = trace::Lap::start();
    let mut again = assemble_with(
        read.raw.clone(),
        width,
        height,
        Some(Pixels { image, figures }),
    )?;
    trace::line(format_args!("pixels={:.0}ms", lap.ms()));
    again.confidence = read.confidence;
    again.scale = read.scale;
    again.language = read.language;
    again.unread = read.unread;
    again.capped = read.capped;
    Ok(again)
}

/// Read single upright text lines, one text per image in order: the crops
/// of the PDF glyph check (`formats::pdf::glyph_check`). The portable engine
/// reads them with its recognizer only; Vision recognizes each crop.
#[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
pub(crate) fn read_lines(images: Vec<image::RgbImage>, cfg: &Value) -> Result<Vec<String>> {
    let backend = selected()?;
    let spelling = configured(cfg)?;
    match backend {
        #[cfg(target_os = "macos")]
        Backend::Vision => images
            .into_iter()
            .map(|image| Ok(read(&pixels::prepare_rgb(image)?, &spelling)?.text))
            .collect(),
        Backend::Paddle => {
            for image in &images {
                pixels::check_rgb(image)?;
            }
            paddle::read_line_images(&images, &spelling)
        }
    }
}

/// One reading of an image in one language, after the enlarged second reading
/// of small text and the recovery of dropped characters.
#[cfg(target_os = "macos")]
struct Pass {
    lines: Vec<Line>,
    /// How many times the image was enlarged for these lines; 1 when read as is.
    scale: f32,
}

/// Read prepared pixels in one Vision language. Small Chinese or Japanese
/// text, and Chinese, Japanese or Korean text that the first reading missed,
/// are read a second time from an enlarged copy, and that reading replaces
/// the first.
#[cfg(target_os = "macos")]
fn pass(image: &pixels::Prepared, language: &str) -> Result<Pass> {
    let space = (image.width, image.height);
    let mut lap = trace::Lap::start();
    let read = pass_with(image, |image, may_enlarge| {
        vision::recognize(image, language, space, may_enlarge)
    });
    trace::line(format_args!("vision pass {language} ms={:.0}", lap.ms()));
    read
}

/// [`pass`] with all of Chinese, Japanese, Korean and English asked for at once
/// and the language detected by Vision.
#[cfg(target_os = "macos")]
fn pass_detecting(image: &pixels::Prepared) -> Result<Pass> {
    let space = (image.width, image.height);
    let mut lap = trace::Lap::start();
    let read = pass_with(image, |image, may_enlarge| {
        vision::recognize_detecting(image, space, may_enlarge)
    });
    trace::line(format_args!("vision pass detected ms={:.0}", lap.ms()));
    read
}

#[cfg(target_os = "macos")]
fn pass_with(
    image: &pixels::Prepared,
    read: impl Fn(&pixels::Prepared, bool) -> Result<vision::Reading>,
) -> Result<Pass> {
    let first = read(image, true)?;
    Ok(match first.enlarge {
        Some(factor) => {
            let larger = pixels::enlarge(image, factor)?;
            Pass {
                lines: read(&larger, false)?.lines,
                scale: factor,
            }
        }
        None => Pass {
            lines: first.lines,
            scale: 1.0,
        },
    })
}

/// The result of the reading used. Table cells it missed and numbers it
/// garbled are read again first, unless the page is turned.
#[cfg(target_os = "macos")]
fn finish(image: &pixels::Prepared, pass: Pass, language: &str, unread: bool) -> Result<OcrResult> {
    let mut lines = pass.lines;
    if !unread && layout::orientation(&lines) == layout::Turn::Upright {
        vision::reread(image, language, &mut lines);
    }
    let mut result = assemble(lines, image.width, image.height)?;
    result.scale = pass.scale;
    result.language = language.to_owned();
    result.unread = unread;
    Ok(result)
}

/// Recognize prepared pixels in the language `ocr.lang` names (`spelling`, as
/// [`configured`] normalized it). Only the default `en` is a policy: see
/// [`read_default`]. Every other language, English with a region included,
/// is read as that one language.
#[cfg(target_os = "macos")]
fn read(image: &pixels::Prepared, spelling: &str) -> Result<OcrResult> {
    if spelling == "en" {
        // `MARKITAI_OCR_VISION_POLICY=detect`: one request asking for all the
        // languages with detection, instead of the passes below (for
        // measuring what the passes cost and add).
        if std::env::var("MARKITAI_OCR_VISION_POLICY").is_ok_and(|value| value.trim() == "detect") {
            return finish(image, pass_detecting(image)?, CHINESE, false);
        }
        return read_default(image);
    }
    let language = tag(spelling);
    finish(image, pass(image, &language)?, &language, false)
}

#[cfg(target_os = "macos")]
const ENGLISH: &str = "en-US";
#[cfg(target_os = "macos")]
const CHINESE: &str = "zh-Hans";
#[cfg(target_os = "macos")]
const KOREAN: &str = "ko-KR";
#[cfg(target_os = "macos")]
const JAPANESE: &str = "ja-JP";

/// The default language policy ([`auto`]): English first, as for `en-US`. When
/// that reading found no text, or doubts a quarter of its lines, or is of one
/// or two confident lines (English returns confident Latin fragments for lines
/// that mix Latin with Chinese), the image is read as Chinese, which also reads
/// Latin, Japanese and Traditional Chinese. When English found no text or
/// doubts half of its lines it is read as Korean too, unless the Chinese
/// reading has enough credible letters to settle it (Vision's Chinese
/// recognizer makes confident Han of Hangul); the reading with the most text
/// of its script is kept. Text holding kana is read as Japanese once more,
/// which has its own aids. A reading that fails is not an error: the English
/// reading stands, as it always did, and the result is marked unread when text
/// was seen and not read.
#[cfg(target_os = "macos")]
fn read_default(image: &pixels::Prepared) -> Result<OcrResult> {
    use auto::Verdict;
    use cjk::Script;
    let english = pass(image, ENGLISH)?;
    let verdict = auto::judge(&english.lines);
    trace_pass(ENGLISH, &Ok(&english), Some(verdict));
    if verdict == Verdict::Sound {
        return finish(image, english, ENGLISH, false);
    }
    // The strongest reading so far: its language, lines and strength.
    let mut best: Option<(&str, Pass, f32)> = None;
    let mut elsewhere = false;
    let mut settled = false;
    // A Vision language this system lacks, or a failed reading, is not a
    // failure of the conversion: English has read the image.
    let chinese = pass(image, CHINESE);
    trace_pass(CHINESE, &chinese.as_ref(), None);
    if let Ok(chinese) = chinese {
        elsewhere = auto::saw(&chinese.lines, &english.lines);
        if auto::reads(&chinese.lines, Script::Japanese) {
            settled = auto::credible(&chinese.lines, Script::Japanese) >= auto::CONVINCING;
            let strength = auto::strength(&chinese.lines, Script::Japanese);
            best = Some((CHINESE, chinese, strength));
        }
    }
    let korean = (verdict == Verdict::Failed && !settled).then(|| pass(image, KOREAN));
    if let Some(korean) = &korean {
        trace_pass(KOREAN, &korean.as_ref(), None);
    }
    if let Some(Ok(korean)) = korean {
        elsewhere |= auto::saw(&korean.lines, &english.lines);
        let strength = auto::strength(&korean.lines, Script::Korean);
        if auto::reads(&korean.lines, Script::Korean)
            && best.as_ref().is_none_or(|(_, _, other)| strength > *other)
        {
            best = Some((KOREAN, korean, strength));
        }
    }
    let kana = matches!(&best, Some((CHINESE, pass, _)) if auto::japanese(&pass.lines));
    if kana && let Ok(japanese) = pass(image, JAPANESE) {
        let strength = auto::strength(&japanese.lines, Script::Japanese);
        let floor = best.as_ref().map_or(0.0, |(_, _, other)| *other);
        if auto::reads(&japanese.lines, Script::Japanese) && strength >= floor {
            best = Some((JAPANESE, japanese, strength));
        }
    }
    // No single language read the image, yet it is not plainly English: a page
    // that mixes Chinese, Japanese, Korean and Latin, or text that one
    // language's recognizer gives up on, is read once more with all of them
    // asked for and Vision detecting the language.
    if best.is_none() && verdict != Verdict::Sound {
        let all = pass_detecting(image);
        trace_pass("detected", &all.as_ref(), None);
        if let Ok(all) = all {
            let strength = [Script::Chinese, Script::Japanese, Script::Korean]
                .into_iter()
                .filter(|script| auto::reads(&all.lines, *script))
                .map(|script| auto::strength(&all.lines, script))
                .fold(0.0, f32::max);
            if strength > 0.0 || auto::judge(&all.lines) == Verdict::Sound {
                elsewhere = true;
                best = Some((CHINESE, all, strength));
            }
        }
    }
    match best {
        Some((language, pass, _)) => finish(image, pass, language, false),
        None => {
            let unread = auto::unread(verdict, &english.lines, elsewhere);
            let mut english = english;
            if unread {
                // What English makes of another script is not text: keep
                // only the lines it is sure of.
                english.lines.retain(auto::sure);
            }
            finish(image, english, ENGLISH, unread)
        }
    }
}

/// One line of `MARKITAI_OCR_TRACE` for a reading of the default policy: what a
/// language read, or why it could not, so that a page left unread can be told
/// from a language this system lacks.
#[cfg(target_os = "macos")]
fn trace_pass(
    language: &str,
    pass: &std::result::Result<&Pass, &Error>,
    verdict: Option<auto::Verdict>,
) {
    if !trace::enabled() {
        return;
    }
    match pass {
        Ok(pass) => {
            let sure = pass.lines.iter().filter(|line| auto::sure(line)).count();
            trace::line(format_args!(
                "engine=vision language={language} lines={} sure={sure} scale={} verdict={verdict:?}",
                pass.lines.len(),
                pass.scale
            ));
        }
        Err(error) => trace::line(format_args!(
            "engine=vision language={language} failed: {error}"
        )),
    }
}

const MAX_LINES: usize = 10_000;
const MAX_TEXT: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug)]
struct Line {
    text: String,
    confidence: f32,
    bounds: [f32; 4],
    /// The direction the text runs, from the line's top-left to its top-right
    /// corner, in top-left pixel coordinates: `[1, 0]` for upright text.
    direction: [f32; 2],
    /// What the page's pixels tell of it ([`page`]); nothing without them.
    marks: page::Marks,
}

fn failure(message: &str) -> Error {
    Error::Conversion(format!("Local OCR: {message}"))
}

/// The spelling of `ocr.lang` after case, surrounding whitespace and
/// underscores are normalized; an absent value is the default `en`. The
/// configuration materializes its default, so a default `en` and one written
/// out cannot be told apart here: both are the default language policy.
fn configured(cfg: &Value) -> Result<String> {
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
    Ok(normalized)
}

/// The Vision language a normalized `ocr.lang` spelling asks for.
#[cfg(any(target_os = "macos", test))]
fn tag(spelling: &str) -> String {
    match spelling {
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
        other => other,
    }
    .to_owned()
}

#[cfg(test)]
fn language(cfg: &Value) -> Result<String> {
    Ok(tag(&configured(cfg)?))
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
    Err(Error::Unsupported(format!(
        "ocr.lang is not supported by the installed macOS Vision text recognizer ({requested}); \
         it reads {}",
        supported.join(", ")
    )))
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

/// What assembling an image's lines may also read: its pixels (for bold and
/// italic words, dropped bullets, ruled tables), and whether pictures are
/// cut out of it (a PDF page's charts and photographs).
#[derive(Clone, Copy)]
pub(super) struct Pixels<'a> {
    pub image: &'a image::RgbImage,
    pub figures: bool,
}

fn assemble(lines: Vec<Line>, width: u32, height: u32) -> Result<OcrResult> {
    assemble_with(lines, width, height, None)
}

fn assemble_with(
    mut lines: Vec<Line>,
    width: u32,
    height: u32,
    pixels: Option<Pixels<'_>>,
) -> Result<OcrResult> {
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
        // Text is ordinary document content: Markdown markup is synthesized
        // only for headings, lists, emphasis, tables, code and pictures (see
        // `rows`).
        line.text = line.text.trim().to_owned();
        if let Some(mended) = layout::digits(&line.text) {
            line.text = mended;
        }
        line.marks = page::Marks::default();
        bytes = bytes
            .checked_add(line.text.len() + 2)
            .filter(|total| *total <= MAX_TEXT)
            .ok_or_else(|| failure("recognized text exceeds the safety limit"))?;
    }
    let raw = lines.clone();
    // A page whose text runs sideways or upside down is read upright.
    let turn = layout::orientation(&lines);
    layout::upright(&mut lines, turn, width, height);
    // The pixels speak only for an upright page of the lines' size.
    let page = pixels
        .filter(|pixels| {
            turn == layout::Turn::Upright && pixels.image.dimensions() == (width, height)
        })
        .and_then(|pixels| Some((page::Page::new(pixels.image)?, pixels.figures)));
    if let Some((page, _)) = &page {
        for line in &mut lines {
            line.marks = page.marks(line);
        }
    }
    let sizes = sizes(&lines);
    let mut style = Style::default();
    let mut regions = Vec::new();
    let mut figures = Vec::new();
    if let (Some((page, cut)), Some(sizes)) = (&page, sizes) {
        aligned_bullets(&mut lines, sizes.body);
        style = Style::of(&lines, sizes.body);
        regions = self::regions(page, &mut lines, sizes.body, *cut, &mut figures);
    }
    let mut out = Out {
        lines: Vec::with_capacity(lines.len()),
        ..Out::default()
    };
    for (block, regions) in place(columns(lines, 0), regions) {
        rows(block, regions, sizes, &style, &mut out);
    }
    let confidence = if out.boxes.is_empty() {
        0.0
    } else {
        out.confidence / out.boxes.len() as f32
    };
    let text = render(&out.blocks, &tiers(&out.blocks), &|_| None);
    let result = OcrResult {
        text,
        confidence,
        boxes: out.boxes,
        scale: 1.0,
        language: String::new(),
        unread: false,
        capped: false,
        blocks: out.blocks,
        figures,
        lines: if turn == layout::Turn::Upright {
            out.lines
        } else {
            Vec::new()
        },
        size: (width, height),
        raw,
    };
    // Keep diagnostics private to the native API, while validating the engine's
    // blank-image distinction before exposing text to the conversion pipeline.
    if (result.blocks.is_empty() != result.boxes.is_empty()) || !result.confidence.is_finite() {
        return Err(failure(
            "recognizer returned inconsistent text observations",
        ));
    }
    Ok(result)
}

/// One block of an image's Markdown: blocks stand apart by a blank line.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Block {
    /// Markdown as written: a paragraph, list, table or code.
    Text(String),
    /// A heading: its text's height in body text heights, and its text.
    Heading { size: f32, text: String },
    /// The picture of that index in [`OcrResult::figures`].
    Figure(usize),
}

/// A picture cut out of a page: where it is, and its pixels.
#[derive(Debug)]
pub(crate) struct Figure {
    pub bounds: [f32; 4],
    pub image: image::RgbImage,
}

/// A recognized line as read, where it is on the image.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Recognized {
    pub text: String,
    pub bounds: [f32; 4],
}

/// Heading sizes this share of a larger one apart start a level of their own:
/// 16 pt under 18 pt (0.89) is another level. Type sizes measured on the ink
/// hold to about 4 %.
const TIER: f32 = 0.92;

/// The heading levels of `blocks`' sizes: the largest headings are of the
/// first level, those less than [`TIER`] of them of the second, and so on.
/// The bottom size of each level, largest first.
pub(crate) fn tiers(blocks: &[Block]) -> Vec<f32> {
    let mut sizes: Vec<f32> = blocks
        .iter()
        .filter_map(|block| match block {
            Block::Heading { size, .. } => Some(*size),
            _ => None,
        })
        .collect();
    tiers_of(&mut sizes)
}

/// [`tiers`] of heading sizes from several images (the pages of a PDF).
pub(crate) fn tiers_of(sizes: &mut [f32]) -> Vec<f32> {
    sizes.sort_by(|a, b| b.total_cmp(a));
    let mut tiers: Vec<(f32, f32)> = Vec::new();
    for &size in sizes.iter() {
        match tiers.last_mut() {
            Some((top, bottom)) if size >= TIER * *top => *bottom = size,
            _ => tiers.push((size, size)),
        }
    }
    tiers.into_iter().map(|(_, bottom)| bottom).collect()
}

/// The level of a heading of `size` among `tiers`, from 1 to 6.
fn level(tiers: &[f32], size: f32) -> usize {
    (tiers
        .iter()
        .position(|bottom| size >= *bottom - 1e-3)
        .unwrap_or(tiers.len().saturating_sub(1))
        + 1)
    .clamp(1, 6)
}

/// `blocks` as Markdown, headings levelled by `tiers`, each picture as
/// `figure` writes it (left out when it writes nothing).
pub(crate) fn render(
    blocks: &[Block],
    tiers: &[f32],
    figure: &dyn Fn(usize) -> Option<String>,
) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(blocks.len());
    for block in blocks {
        match block {
            Block::Text(text) => parts.push(text.clone()),
            Block::Heading { size, text } => {
                parts.push(format!("{} {text}", "#".repeat(level(tiers, *size))));
            }
            Block::Figure(index) => parts.extend(figure(*index)),
        }
    }
    parts.join("\n\n")
}

/// What the rows of an image's blocks add up to.
#[derive(Default)]
struct Out {
    blocks: Vec<Block>,
    boxes: Vec<[f32; 4]>,
    confidence: f32,
    lines: Vec<Recognized>,
}

impl Out {
    fn line(&mut self, line: &Line) {
        self.boxes.push(line.bounds);
        self.confidence += line.confidence;
        self.lines.push(Recognized {
            text: line.text.clone(),
            bounds: line.bounds,
        });
    }

    /// A region's lines, then the region: a picture's labels are not lines
    /// its Markdown reads.
    fn region(&mut self, region: Region) {
        for line in &region.lines {
            if matches!(region.block, Block::Figure(_)) {
                self.boxes.push(line.bounds);
                self.confidence += line.confidence;
            } else {
                self.line(line);
            }
        }
        self.blocks.push(region.block);
    }
}

/// A part of the page written whole where it stands: a ruled table, or a
/// picture with its labels.
struct Region {
    bounds: [f32; 4],
    block: Block,
    lines: Vec<Line>,
}

/// Bullets the pixels found, kept only beside another one at the same
/// indentation: a list has two items at least, and a speck of dirt left
/// of a line is no bullet.
fn aligned_bullets(lines: &mut [Line], body: f32) {
    let found: Vec<f32> = lines.iter().filter_map(|line| line.marks.bullet).collect();
    for line in lines {
        if let Some(x) = line.marks.bullet
            && found
                .iter()
                .filter(|other| (**other - x).abs() <= 0.5 * body)
                .count()
                < 2
        {
            line.marks.bullet = None;
        }
    }
}

/// The stroke of the page's body text, which a bold word's exceeds.
#[derive(Default)]
struct Style {
    body: Option<f32>,
}

/// A word stands for bold or italic alone only with this many letters or
/// digits: a short word's stroke and slant are a few pixels' worth.
const MIN_STYLED_LETTERS: usize = 4;

impl Style {
    /// The median stroke of lines of the body's height, from three of them.
    fn of(lines: &[Line], body: f32) -> Self {
        let mut strokes: Vec<f32> = lines
            .iter()
            .filter(|line| {
                let height = line.bounds[3] - line.bounds[1];
                (0.8 * body..=1.25 * body).contains(&height)
            })
            .filter_map(|line| line.marks.stroke)
            .collect();
        if strokes.len() < 3 {
            return Self::default();
        }
        strokes.sort_by(f32::total_cmp);
        Self {
            body: Some(strokes[strokes.len() / 2]).filter(|stroke| *stroke > 0.0),
        }
    }

    /// Whether each word of `line` is bold and italic, when its words were
    /// measured; `before` is the word before the line's first (the end of
    /// the line above).
    fn words(&self, line: &Line, before: Option<(bool, bool)>) -> Option<Vec<(bool, bool)>> {
        let body = self.body?;
        let words = line.marks.words.as_ref()?;
        let tokens: Vec<&str> = line.text.split_whitespace().collect();
        if words.len() != tokens.len() {
            return None;
        }
        let n = words.len();
        let ratio = |i: usize| words[i].stroke / body;
        let short = |i: usize| {
            words[i].stroke == 0.0
                || tokens[i].chars().filter(|c| c.is_alphanumeric()).count() < MIN_STYLED_LETTERS
        };
        let strong: Vec<bool> = (0..n).map(|i| ratio(i) >= page::BOLD).collect();
        let bold: Vec<bool> = (0..n)
            .map(|i| {
                strong[i]
                    || (ratio(i) >= page::BOLD_NEXT
                        && ((i > 0 && strong[i - 1]) || (i + 1 < n && strong[i + 1])))
            })
            .collect();
        let italic: Vec<bool> = words.iter().map(|word| word.italic).collect();
        let settle = |mut flags: Vec<bool>, before: bool| {
            let left = |flags: &[bool], i: usize| if i > 0 { flags[i - 1] } else { before };
            // A short word (too small to measure well) between two emphasized
            // words goes with them.
            for i in 0..n {
                if !flags[i] && short(i) && left(&flags, i) && i + 1 < n && flags[i + 1] {
                    flags[i] = true;
                }
            }
            // A short word does not stand emphasized alone.
            let alone: Vec<bool> = (0..n)
                .map(|i| flags[i] && short(i) && !left(&flags, i) && !(i + 1 < n && flags[i + 1]))
                .collect();
            for (flag, alone) in flags.iter_mut().zip(alone) {
                *flag &= !alone;
            }
            flags
        };
        let (bold_before, italic_before) = before.unwrap_or_default();
        let bold = settle(bold, bold_before);
        let italic = settle(italic, italic_before);
        Some(bold.into_iter().zip(italic).collect())
    }

    /// Whether all of `line` is bold: each of its measured words, or its
    /// whole stroke when its words were not told apart.
    fn bold(&self, line: &Line) -> bool {
        match (self.words(line, None), self.body, line.marks.stroke) {
            (Some(words), ..) => {
                // A short word's few pixels do not tell (`In` of a bold line
                // measures light), unless the line has no other.
                let letters = |word: &str| word.chars().filter(|c| c.is_alphanumeric()).count();
                let flags: Vec<bool> = line
                    .text
                    .split_whitespace()
                    .zip(&words)
                    .filter(|(word, _)| letters(word) >= MIN_STYLED_LETTERS)
                    .map(|(_, (bold, _))| *bold)
                    .collect();
                if flags.is_empty() {
                    !words.is_empty() && words.iter().all(|(bold, _)| *bold)
                } else {
                    flags.iter().all(|bold| *bold)
                }
            }
            (None, Some(body), Some(stroke)) => stroke >= page::BOLD * body,
            _ => false,
        }
    }
}

/// The bullet a recognizer read at the start of a line, with the space
/// after it: its length in bytes.
fn bullet_glyph(text: &str) -> Option<usize> {
    let mut characters = text.char_indices();
    let (_, first) = characters.next()?;
    let (at, space) = characters.next()?;
    (matches!(
        first,
        '•' | '●'
            | '▪'
            | '■'
            | '◦'
            | '‣'
            | '∙'
            | '○'
            | '□'
            | '➢'
            | '►'
            | '▶'
            | '-'
            | '–'
            | '*'
    ) && space.is_whitespace())
    .then_some(at + space.len_utf8())
}

/// Caption lines stay text beside the picture they name.
fn caption(text: &str) -> bool {
    let lower = text.trim_start().to_lowercase();
    [
        "fig", "figure", "table", "chart", "source", "图", "圖", "表", "출처", "図",
    ]
    .iter()
    .any(|word| lower.starts_with(word))
}

/// Labels of a picture are short: at most this many characters.
const MAX_LABEL_CHARS: usize = 24;
/// A line of this many characters inside a marked area makes it a box of
/// text, not a picture.
const MIN_PROSE_CHARS: usize = 30;

/// The ruled tables of the page and, when `cut`, its pictures, each taken
/// out of `lines` with the lines it holds (a table's cells, a picture's
/// labels); the pictures' pixels go to `figures`.
fn regions(
    page: &page::Page<'_>,
    lines: &mut Vec<Line>,
    body: f32,
    cut: bool,
    figures: &mut Vec<Figure>,
) -> Vec<Region> {
    let center = |b: [f32; 4]| [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0];
    let inside = |b: [f32; 4], area: [f32; 4]| {
        let [x, y] = center(b);
        x >= area[0] && x <= area[2] && y >= area[1] && y <= area[3]
    };
    let mut regions = Vec::new();
    for grid in page.grids(body) {
        let held: Vec<usize> = (0..lines.len())
            .filter(|&i| inside(lines[i].bounds, grid.bounds))
            .collect();
        let cells = grid_cells(page, &grid, held.iter().map(|&i| &lines[i]));
        let filled = cells
            .iter()
            .flatten()
            .filter(|cell| !cell.is_empty())
            .count();
        if cells.len() < 2 || filled < 2 {
            continue;
        }
        let mut held = take(lines, &held);
        crate::sort::by(&mut held, top_then_left);
        regions.push(Region {
            bounds: grid.bounds,
            block: Block::Text(table_markdown(&cells)),
            lines: held,
        });
    }
    if !cut {
        return regions;
    }
    let mut skip: Vec<[f32; 4]> = regions.iter().map(|region| region.bounds).collect();
    skip.extend(lines.iter().filter_map(|line| {
        let x = line.marks.bullet?;
        let [l, t, _, b] = line.bounds;
        Some([x - 1.0, t, l, b])
    }));
    let chars = |line: &Line| line.text.chars().filter(|c| !c.is_whitespace()).count();
    // Each picture's area and the lines it holds (its labels).
    let mut found: Vec<([f32; 4], Vec<usize>)> = Vec::new();
    for mut area in page.pictures(lines, &skip, body) {
        // A box around prose (a framed note, a page border) is no picture.
        if lines.iter().any(|line| {
            inside(line.bounds, area)
                && chars(line) >= MIN_PROSE_CHARS
                && line.bounds[2] - line.bounds[0] >= 0.5 * (area[2] - area[0])
        }) {
            continue;
        }
        // Its labels: short lines inside it or within a line height of it.
        let mut held: Vec<usize> = Vec::new();
        for _ in 0..3 {
            let reach = [
                area[0] - body,
                area[1] - body,
                area[2] + body,
                area[3] + body,
            ];
            let mut grew = false;
            for (i, line) in lines.iter().enumerate() {
                let b = line.bounds;
                let near = b[0] < reach[2] && b[2] > reach[0] && b[1] < reach[3] && b[3] > reach[1];
                if held.contains(&i)
                    || !(inside(b, area) || near)
                    || chars(line) > MAX_LABEL_CHARS
                    || caption(&line.text)
                    || line.marks.bullet.is_some()
                {
                    continue;
                }
                held.push(i);
                area = union(area, b);
                grew = true;
            }
            if !grew {
                break;
            }
        }
        found.push((area, held));
    }
    // Pieces of one picture (a chart's bar groups, its labels between
    // them) less than a line height apart are one picture.
    let mut merged = true;
    while merged {
        merged = false;
        'pairs: for i in 0..found.len() {
            for j in i + 1..found.len() {
                let (a, b) = (found[i].0, found[j].0);
                if b[0] < a[2] + body
                    && a[0] < b[2] + body
                    && b[1] < a[3] + body
                    && a[1] < b[3] + body
                {
                    let (area, held) = found.swap_remove(j);
                    found[i].0 = union(found[i].0, area);
                    found[i].1.extend(held);
                    merged = true;
                    break 'pairs;
                }
            }
        }
    }
    // Lines a merged picture now covers are its own too.
    let mut taken: Vec<usize> = Vec::new();
    for (area, held) in &mut found {
        for (i, line) in lines.iter().enumerate() {
            if inside(line.bounds, *area) && !held.contains(&i) && chars(line) <= MAX_LABEL_CHARS {
                held.push(i);
            }
        }
        held.retain(|i| !taken.contains(i));
        taken.extend(held.iter().copied());
    }
    crate::sort::by(&mut found, |a, b| a.0[1].total_cmp(&b.0[1]));
    let mut pictures: Vec<([f32; 4], Vec<Line>)> = Vec::new();
    let mut gone: Vec<usize> = found
        .iter()
        .flat_map(|(_, held)| held.iter().copied())
        .collect();
    gone.sort_unstable();
    gone.dedup();
    let mut kept = take(lines, &gone);
    for (area, held) in &found {
        let mut own: Vec<Line> = held
            .iter()
            .map(|i| kept[gone.binary_search(i).expect("a taken line")].clone())
            .collect();
        crate::sort::by(&mut own, top_then_left);
        pictures.push((*area, own));
    }
    kept.clear();
    for (area, own) in pictures {
        let pad = 0.3 * body;
        let bounds = [area[0] - pad, area[1] - pad, area[2] + pad, area[3] + pad];
        let Some(image) = page.crop(bounds) else {
            lines.extend(own);
            continue;
        };
        let [l, t, ..] = bounds.map(|v| v.max(0.0));
        let bounds = [l, t, l + image.width() as f32, t + image.height() as f32];
        regions.push(Region {
            bounds,
            block: Block::Figure(figures.len()),
            lines: own,
        });
        figures.push(Figure { bounds, image });
    }
    regions
}

fn union(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

/// The lines of `indices` (ascending), taken out of `lines`.
fn take(lines: &mut Vec<Line>, indices: &[usize]) -> Vec<Line> {
    let mut taken = Vec::with_capacity(indices.len());
    for &index in indices.iter().rev() {
        taken.push(lines.remove(index));
    }
    taken.reverse();
    taken
}

/// The cells of a ruled table: each line goes in the cell under its centre;
/// a line across a column rule is split at the word nearest it. Rows and
/// columns with no text are left out.
fn grid_cells<'a>(
    page: &page::Page<'_>,
    grid: &page::Grid,
    lines: impl Iterator<Item = &'a Line>,
) -> Vec<Vec<String>> {
    let (columns, rows) = (grid.columns.len() - 1, grid.rows.len() - 1);
    let mut pieces: Vec<(usize, usize, f32, f32, String)> = Vec::new();
    let band = |edges: &[f32], at: f32| {
        edges
            .windows(2)
            .position(|pair| at >= pair[0] && at < pair[1])
            .unwrap_or(edges.len().saturating_sub(2))
    };
    for line in lines {
        let [l, t, r, b] = line.bounds;
        let row = band(&grid.rows, (t + b) / 2.0);
        let height = b - t;
        let crossed: Vec<usize> = (0..columns)
            .filter(|&c| r.min(grid.columns[c + 1]) - l.max(grid.columns[c]) > 0.3 * height)
            .collect();
        let tokens: Vec<&str> = line.text.split_whitespace().collect();
        if crossed.len() < 2 || tokens.len() < 2 {
            let column = band(&grid.columns, (l + r) / 2.0);
            pieces.push((row, column, t, l, line.text.clone()));
            continue;
        }
        // Where each word stands: the ink's words when they match the
        // text's, else by its characters' share of the line.
        let spans = page.word_spans(line.bounds);
        let centres: Vec<f32> = if spans.len() == tokens.len() {
            spans.iter().map(|[a, b]| (a + b) / 2.0).collect()
        } else {
            let total = line.text.chars().count().max(1) as f32;
            let mut at = 0usize;
            let mut centres = Vec::new();
            for token in line.text.split(' ') {
                let length = token.chars().count();
                if !token.is_empty() {
                    centres.push(l + (r - l) * (at as f32 + length as f32 / 2.0) / total);
                }
                at += length + 1;
            }
            centres
        };
        for (token, centre) in tokens.iter().zip(centres) {
            let column = band(&grid.columns, centre);
            match pieces.last_mut() {
                Some(last) if last.0 == row && last.1 == column && last.2 == t => {
                    last.4.push(' ');
                    last.4.push_str(token);
                }
                _ => pieces.push((row, column, t, centre, (*token).to_owned())),
            }
        }
    }
    crate::sort::by(&mut pieces, |a, b| {
        (a.0, a.1)
            .cmp(&(b.0, b.1))
            .then(a.2.total_cmp(&b.2))
            .then(a.3.total_cmp(&b.3))
    });
    let mut cells = vec![vec![String::new(); columns]; rows];
    for (row, column, _, _, text) in pieces {
        let cell = &mut cells[row][column];
        let joins = cell
            .chars()
            .last()
            .is_some_and(|c| !(cjk::han(c) || cjk::kana(c)))
            || text
                .chars()
                .next()
                .is_some_and(|c| !(cjk::han(c) || cjk::kana(c)));
        if !cell.is_empty() && joins {
            cell.push(' ');
        }
        cell.push_str(&text);
    }
    cells.retain(|row| row.iter().any(|cell| !cell.is_empty()));
    let used: Vec<bool> = (0..columns)
        .map(|c| cells.iter().any(|row| !row[c].is_empty()))
        .collect();
    for row in &mut cells {
        let mut flags = used.iter();
        row.retain(|_| *flags.next().unwrap_or(&false));
    }
    cells
}

/// Each block of lines with the regions that stand among them: a region goes
/// with the block it overlaps most across, or, beside every block, with the
/// nearest one above it (the first block when none is).
fn place(blocks: Vec<Vec<Line>>, regions: Vec<Region>) -> Vec<(Vec<Line>, Vec<Region>)> {
    let spans: Vec<Option<[f32; 4]>> = blocks
        .iter()
        .map(|lines| {
            lines.iter().map(|line| line.bounds).reduce(|a, b| {
                [
                    a[0].min(b[0]),
                    a[1].min(b[1]),
                    a[2].max(b[2]),
                    a[3].max(b[3]),
                ]
            })
        })
        .collect();
    let mut placed: Vec<(Vec<Line>, Vec<Region>)> = blocks
        .into_iter()
        .map(|lines| (lines, Vec::new()))
        .collect();
    if placed.is_empty() {
        placed.push((Vec::new(), Vec::new()));
    }
    for region in regions {
        let b = region.bounds;
        let overlap = |span: &[f32; 4]| {
            let across = span[2].min(b[2]) - span[0].max(b[0]);
            let down = span[3].min(b[3] + (b[3] - b[1])) - span[1].max(b[1] - (b[3] - b[1]));
            (across > 0.0 && down > 0.0).then_some(across)
        };
        let best = spans
            .iter()
            .enumerate()
            .filter_map(|(i, span)| Some((i, overlap(span.as_ref()?)?)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
            .or_else(|| {
                spans
                    .iter()
                    .enumerate()
                    .filter(|(_, span)| span.is_some_and(|span| span[1] <= b[1]))
                    .max_by(|a, b| {
                        a.1.map_or(0.0, |s| s[1])
                            .total_cmp(&b.1.map_or(0.0, |s| s[1]))
                    })
                    .map(|(i, _)| i)
            })
            .unwrap_or(0);
        placed[best].1.push(region);
    }
    placed
}

/// Where a run of emphasis closes at the end of one line of a paragraph and
/// opens again at the start of the next, it goes on instead.
fn joined_emphasis(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut from = 0;
    let mut index = 0;
    while let Some(offset) = text[index..].find('\n') {
        let at = index + offset;
        let closing = bytes[..at].iter().rev().take_while(|b| **b == b'*').count();
        let mut open = at + 1;
        while bytes.get(open) == Some(&b' ') {
            open += 1;
        }
        let opening = bytes[open..].iter().take_while(|b| **b == b'*').count();
        let before = at.checked_sub(closing + 1).map(|i| bytes[i]);
        let after = bytes.get(open + opening).copied();
        if (1..=3).contains(&closing)
            && closing == opening
            && before.is_some_and(|b| b != b' ' && b != b'*')
            && after.is_some_and(|b| b != b' ' && b != b'*')
        {
            out.push_str(&text[from..at - closing]);
            out.push_str(&text[at..open]);
            from = open + opening;
        }
        index = at + 1;
    }
    out.push_str(&text[from..]);
    out
}

/// A row's words with their emphasis: runs of bold, italic or both words
/// between `**`, `*` or `***`, and the emphasis of its last word. `None`
/// when no word of it was measured. `before` is the emphasis of the word
/// before it (the end of the row above).
fn styled(
    lines: &[Line],
    style: &Style,
    skip_first: bool,
    mut before: Option<(bool, bool)>,
) -> Option<(String, Option<(bool, bool)>)> {
    let mut words: Vec<(&str, (bool, bool))> = Vec::new();
    let mut measured = false;
    for line in lines {
        match style.words(line, before) {
            Some(flags) => {
                measured = true;
                before = flags.last().copied();
                words.extend(line.text.split_whitespace().zip(flags));
            }
            None => {
                before = None;
                words.extend(line.text.split_whitespace().map(|w| (w, (false, false))));
            }
        }
    }
    if !measured {
        return None;
    }
    if skip_first && !words.is_empty() {
        words.remove(0);
    }
    let mut out = String::new();
    for run in words.chunk_by(|a, b| a.1 == b.1) {
        let marker = match run[0].1 {
            (true, true) => "***",
            (true, false) => "**",
            (false, true) => "*",
            (false, false) => "",
        };
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(marker);
        out.push_str(&run.iter().map(|(w, _)| *w).collect::<Vec<_>>().join(" "));
        out.push_str(marker);
    }
    Some((out, before))
}

/// Top edge, then left edge: the order `columns` and `rows` start from. One
/// named comparator, so the two sorts share one compiled sort.
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
fn columns(mut lines: Vec<Line>, depth: usize) -> Vec<Vec<Line>> {
    const MAX_DEPTH: usize = 4;
    const MIN_COLUMN_LINES: usize = 3;
    const MIN_PROSE_CHARS: usize = 12;
    const SPANNING: f32 = 0.6;
    crate::sort::by(&mut lines, top_then_left);
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
    crate::sort::by(&mut spans, |a, b| a.0.total_cmp(&b.0));
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

/// Text heights on a page, for telling its headings: the median height of
/// its lines of prose (a picture's short labels, a photo's key caps, do not
/// count), and, with the page's pixels, the median type size of its long
/// lines (`page::Marks::size`). `None` on a page of too few lines to tell.
#[derive(Clone, Copy)]
struct Sizes {
    body: f32,
    ink: Option<f32>,
}

impl Sizes {
    /// A line's size in body sizes: its type measured on the ink when both
    /// were, else its box's height. A recognizer's boxes grow and shrink with
    /// descenders and commas (one 16 pt heading measured 1.03–1.28 times the
    /// body by box, 1.41 times every time on the ink).
    fn of(&self, line: &Line) -> f32 {
        match (self.ink, line.marks.size) {
            (Some(body), Some(size)) => size / body,
            _ => (line.bounds[3] - line.bounds[1]) / self.body,
        }
    }
}

/// Lines of at least this many characters (not spaces) are prose, whose
/// median height is the body's: [`sizes`].
const MIN_BODY_CHARS: usize = 12;
/// Lines of at least this many letters and digits give the body's type size.
const MIN_INK_BODY_CHARS: usize = 20;

/// A page needs this many lines before a taller one is a heading.
const MIN_SIZED_LINES: usize = 4;
/// A heading's line is at least this many times the median line's height.
const HEADING: f32 = 1.25;
/// Rows at least this many times the median line's height, one after the
/// other, are one paragraph (a wrapped title in a larger or bold face).
const LARGE: f32 = 1.15;
/// A heading holds at most this many characters per row...
const MAX_HEADING_CHARS: usize = 60;
/// ...and this many rows.
const MAX_HEADING_ROWS: usize = 3;
/// A bold row opening a paragraph is a heading from this many times the
/// median line's height.
const BOLD_HEADING: f32 = 1.1;

fn sizes(lines: &[Line]) -> Option<Sizes> {
    if lines.len() < MIN_SIZED_LINES {
        return None;
    }
    let median = |mut values: Vec<f32>| {
        values.sort_by(f32::total_cmp);
        values[values.len() / 2]
    };
    let height = |line: &Line| line.bounds[3] - line.bounds[1];
    let prose: Vec<f32> = lines
        .iter()
        .filter(|line| line.text.chars().filter(|c| !c.is_whitespace()).count() >= MIN_BODY_CHARS)
        .map(height)
        .collect();
    let body = if prose.len() >= 3 {
        median(prose)
    } else {
        median(lines.iter().map(height).collect())
    };
    let inked: Vec<f32> = lines
        .iter()
        .filter(|line| {
            line.text.chars().filter(|c| c.is_alphanumeric()).count() >= MIN_INK_BODY_CHARS
        })
        .filter_map(|line| line.marks.size)
        .collect();
    let ink = (inked.len() >= 3)
        .then(|| median(inked))
        .filter(|ink| *ink > 0.0);
    (body > 0.0).then_some(Sizes { body, ink })
}

/// Lines of a row at least this many line heights apart are pieces of their
/// own (table cells, if the rows around them agree): the recognizer finds
/// words this far apart as separate lines.
const PIECE_GAP: f32 = 0.6;

/// One row of a block: its text, its rectangle, its tallest line's height,
/// and its pieces: runs of its lines less than [`PIECE_GAP`] apart, with
/// their left and right edges.
struct Row {
    text: String,
    rect: [f32; 4],
    height: f32,
    pieces: Vec<(String, [f32; 2])>,
    /// All its words are bold.
    bold: bool,
    /// It follows a bullet (read, then left out of `text`, or drawn).
    bullet: bool,
    /// Its text with its bold and italic words emphasized.
    styled: Option<String>,
    /// Its largest line's size in body sizes ([`Sizes::of`]).
    size: f32,
}

/// What a row is written as.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Text,
    Code,
    Heading,
    /// The rest of the heading above, on its line.
    HeadingMore,
    /// A list item, after its bullet.
    ListItem,
    /// The rest of the list item above.
    ListMore,
    /// A row of the table of that index.
    Table(usize),
}

/// Rows of cells a table needs: laid out in columns, three; divided by the
/// rules the recognizer read as `|`, two.
const MIN_TABLE_ROWS: usize = 3;
const MIN_RULED_ROWS: usize = 2;

/// A table: its rows (`start..end` of the block's) as cells.
struct Table {
    start: usize,
    end: usize,
    cells: Vec<Vec<String>>,
}

/// Index of the column of `columns` (sorted, apart) that the span `x` overlaps most.
fn column_of(columns: &[[f32; 2]], x: [f32; 2]) -> Option<usize> {
    columns
        .iter()
        .enumerate()
        .map(|(index, column)| (index, x[1].min(column[1]) - x[0].max(column[0])))
        .filter(|(_, overlap)| *overlap > 0.0)
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(index, _)| index)
}

/// The tables among `rows`. Laid out: a run of rows of two or more pieces,
/// whose pieces fall in columns (the spans that the pieces of its rows of
/// the most common number of pieces overlap), one piece per column and row;
/// a row of one piece within one column continues the cell above it (a
/// wrapped cell). Ruled: a run of
/// rows of one piece each that its rules or wide gaps split into the same
/// number of cells, with a rule in one of them at least.
fn tables(rows: &[Row], coded: &[bool]) -> Vec<Table> {
    let mut tables = Vec::new();
    let mut number = 0;
    while number < rows.len() {
        if coded[number] || rows[number].pieces.len() < 2 {
            number += 1;
            continue;
        }
        let mut spans: Vec<[f32; 2]> = Vec::new();
        let mut last = number;
        let mut end = number;
        while end < rows.len() && !coded[end] {
            let row = &rows[end];
            if row.pieces.len() >= 2 {
                spans.extend(row.pieces.iter().map(|(_, x)| *x));
                last = end;
            } else {
                // One piece: the rest of a cell, within one column so far.
                let columns = merged(spans.clone());
                let x = row.pieces[0].1;
                let within = columns
                    .iter()
                    .filter(|column| x[1].min(column[1]) > x[0].max(column[0]))
                    .count()
                    == 1;
                let near = row.rect[1] - rows[end - 1].rect[3] <= row.height;
                if !within || !near {
                    break;
                }
            }
            end += 1;
        }
        let end = last + 1;
        let laid_out = (number..end).filter(|&i| rows[i].pieces.len() >= 2);
        // The columns are those of the rows of the most common number of
        // pieces: the recognizer sometimes reads two cells as one line, and
        // that piece goes in the column it overlaps most.
        let mut counts: Vec<usize> = laid_out.clone().map(|i| rows[i].pieces.len()).collect();
        counts.sort_unstable();
        let common = counts
            .chunk_by(|a, b| a == b)
            .max_by_key(|run| run.len())
            .map_or(0, |run| run[0]);
        let columns = merged(
            laid_out
                .clone()
                .filter(|&i| rows[i].pieces.len() == common)
                .flat_map(|i| rows[i].pieces.iter().map(|(_, x)| *x))
                .collect(),
        );
        let distinct = laid_out.clone().all(|i| {
            let mut seen: Vec<usize> = rows[i]
                .pieces
                .iter()
                .filter_map(|(_, x)| column_of(&columns, *x))
                .collect();
            let count = seen.len();
            seen.dedup();
            count == rows[i].pieces.len() && seen.len() == count
        });
        if columns.len() >= 2 && laid_out.count() >= MIN_TABLE_ROWS && distinct {
            let mut cells: Vec<Vec<String>> = Vec::new();
            for row in &rows[number..end] {
                if row.pieces.len() >= 2 {
                    let mut line = vec![String::new(); columns.len()];
                    for (piece, x) in &row.pieces {
                        let column = column_of(&columns, *x).expect("a placed piece");
                        line[column] = piece.clone();
                    }
                    cells.push(line);
                } else if let (Some(line), Some(column)) =
                    (cells.last_mut(), column_of(&columns, row.pieces[0].1))
                {
                    let cell = &mut line[column];
                    if !cell.is_empty() {
                        cell.push(' ');
                    }
                    cell.push_str(&row.pieces[0].0);
                }
            }
            tables.push(Table {
                start: number,
                end,
                cells,
            });
        }
        // A run whose pieces do not fall in columns is no table, nor any part of it.
        number = end;
    }
    // Ruled rows, outside the laid-out tables.
    let free = |index: usize| {
        !coded[index]
            && rows[index].pieces.len() == 1
            && !tables
                .iter()
                .any(|table| (table.start..table.end).contains(&index))
    };
    let split: Vec<Option<(Vec<String>, bool)>> = (0..rows.len())
        .map(|index| {
            free(index)
                .then(|| layout::table_cells(&rows[index].text))
                .filter(|(cells, _)| cells.len() >= 2)
        })
        .collect();
    let mut start = 0;
    while start < rows.len() {
        let Some((first, _)) = &split[start] else {
            start += 1;
            continue;
        };
        let columns = first.len();
        let end = (start..rows.len())
            .find(|&index| {
                split[index]
                    .as_ref()
                    .is_none_or(|(cells, _)| cells.len() != columns)
            })
            .unwrap_or(rows.len());
        let ruled = split[start..end].iter().flatten().any(|(_, ruled)| *ruled);
        if ruled && end - start >= MIN_RULED_ROWS {
            tables.push(Table {
                start,
                end,
                cells: split[start..end]
                    .iter()
                    .flatten()
                    .map(|(cells, _)| cells.clone())
                    .collect(),
            });
        }
        start = end;
    }
    tables
}

/// Spans merged where they overlap, left to right.
fn merged(mut spans: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    crate::sort::by(&mut spans, |a, b| a[0].total_cmp(&b[0]));
    let mut columns: Vec<[f32; 2]> = Vec::new();
    for span in spans {
        match columns.last_mut() {
            Some(column) if span[0] < column[1] => column[1] = column[1].max(span[1]),
            _ => columns.push(span),
        }
    }
    columns
}

/// A table in Markdown, its first row the header: number cells without
/// their inner spaces, letter-spaced cells closed up, rules escaped.
fn table_markdown(cells: &[Vec<String>]) -> String {
    let row = |cells: &[String]| {
        let cells: Vec<String> = cells
            .iter()
            .map(|cell| {
                layout::number_cell(cell)
                    .or_else(|| layout::closed_up(cell))
                    .unwrap_or_else(|| cell.clone())
                    .replace('|', "\\|")
            })
            .collect();
        format!("| {} |", cells.join(" | "))
    };
    let width = cells.first().map_or(0, Vec::len);
    let mut lines = Vec::with_capacity(cells.len() + 1);
    for (index, cells) in cells.iter().enumerate() {
        lines.push(row(cells));
        if index == 0 {
            lines.push(row(&vec!["---".to_owned(); width]));
        }
    }
    lines.join("\n")
}

/// Appends one block's lines as Markdown blocks: lines sharing most of a
/// vertical band form a left-to-right row; a row starts a paragraph after a
/// wide vertical gap, and after a row that stops short of the block's right
/// edge by more than the next row's first word needs (only rows that run to
/// that edge, or end with no room left for that word, in a block where two
/// or more do, wrap on to the next: ragged-right prose keeps its
/// paragraphs), unless both are in a larger face. Lines meet
/// without a space where Chinese or Japanese text touches the next line, and
/// letter-spaced text is closed up. A short row much taller than the page's
/// text, or a little taller and bold, opening a paragraph, is a heading
/// (with the rows of its size right below it); a row after a bullet (read,
/// or found in the pixels) is a list item, the rows indented under it its
/// continuation; rows of cells in columns, or divided by the rules the
/// recognizer read as `|`, are a table ([`tables`]). Rows of code in a
/// fixed-pitch font ([`layout::code`]) are fenced, each indented as far as
/// its left edge. Words the pixels show bold or italic are emphasized.
/// `regions` (ruled tables, pictures) are written where they stand among
/// the rows.
fn rows(
    mut lines: Vec<Line>,
    mut regions: Vec<Region>,
    sizes: Option<Sizes>,
    style: &Style,
    out: &mut Out,
) {
    crate::sort::by(&mut lines, top_then_left);
    crate::sort::by(&mut regions, |a, b| a.bounds[1].total_cmp(&b.bounds[1]));
    // Each row's lines.
    let mut groups = Vec::new();
    let mut index = 0;
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
        crate::sort::by(&mut lines[index..end], |a, b| {
            a.bounds[0].total_cmp(&b.bounds[0])
        });
        groups.push(index..end);
        index = end;
    }
    let code = layout::code(&lines, &groups);
    let coded: Vec<bool> = (0..groups.len())
        .map(|number| {
            code.as_ref()
                .is_some_and(|code| code.rows.contains(&number))
        })
        .collect();
    // The emphasis of the last word of the row above.
    let mut carry = None;
    let rows: Vec<Row> = groups
        .iter()
        .zip(&coded)
        .map(|(group, &coded)| {
            let first = &lines[group.start];
            let mut row = Row {
                text: String::new(),
                rect: first.bounds,
                height: 0.0,
                pieces: Vec::new(),
                bold: false,
                bullet: false,
                styled: None,
                size: 1.0,
            };
            for offset in group.clone() {
                let line = &lines[offset];
                let height = line.bounds[3] - line.bounds[1];
                let x = [line.bounds[0], line.bounds[2]];
                if offset > group.start {
                    let previous = &lines[offset - 1];
                    let joined = !layout::touching(previous, line);
                    if joined {
                        row.text.push(' ');
                    }
                    match row.pieces.last_mut() {
                        Some((piece, span))
                            if line.bounds[0] - previous.bounds[2] < PIECE_GAP * height =>
                        {
                            if joined {
                                piece.push(' ');
                            }
                            piece.push_str(&line.text);
                            span[1] = x[1];
                        }
                        _ => row.pieces.push((line.text.clone(), x)),
                    }
                } else {
                    row.pieces.push((line.text.clone(), x));
                }
                row.text.push_str(&line.text);
                row.rect[0] = row.rect[0].min(line.bounds[0]);
                row.rect[1] = row.rect[1].min(line.bounds[1]);
                row.rect[2] = row.rect[2].max(line.bounds[2]);
                row.rect[3] = row.rect[3].max(line.bounds[3]);
                row.height = row.height.max(height);
                if let Some(sizes) = sizes {
                    row.size = row.size.max(sizes.of(line));
                }
            }
            if coded {
                carry = None;
                return row;
            }
            let members = &lines[group.clone()];
            let glyph = bullet_glyph(&row.text);
            row.bullet = glyph.is_some() || first.marks.bullet.is_some();
            row.bold = members.iter().all(|line| style.bold(line));
            let touching = members
                .windows(2)
                .any(|pair| layout::touching(&pair[0], &pair[1]));
            if let Some(closed) = layout::closed_up(&row.text) {
                row.text = closed;
                carry = None;
            } else if !touching
                && let Some((text, last)) = styled(members, style, glyph.is_some(), carry)
            {
                row.styled = Some(text);
                carry = last;
            } else {
                carry = None;
            }
            if let Some(length) = glyph {
                row.text = row.text[length..].trim_start().to_owned();
            }
            row
        })
        .collect();
    let tables = tables(&rows, &coded);
    let right = rows
        .iter()
        .zip(&coded)
        .filter(|(_, coded)| !**coded)
        .fold(f32::NEG_INFINITY, |r, (row, _)| r.max(row.rect[2]));
    // How near the block's right edge a row must come to reach it.
    let at_edge = |row: &Row| {
        row.rect[2] >= right - sizes.map_or(row.height, |sizes| sizes.body.max(row.height * 0.5))
    };
    // Whether `above` ran on into `below`: it reaches the block's right
    // edge, or stops short of it by less than `below`'s first word needs
    // (left-aligned prose breaks before a word that does not fit; a row
    // ending further short ends its paragraph). Chinese and Japanese text
    // breaks between any two characters.
    let wrapped = |above: &Row, below: &Row| {
        if at_edge(above) {
            return true;
        }
        let text = below.text.trim_start();
        let chars = text.chars().count().max(1);
        // Up to two characters into CJK text (`·安排` before `十月…`).
        let first = text.split_whitespace().next().unwrap_or_default();
        let word = match first.chars().position(|c| cjk::han(c) || cjk::kana(c)) {
            Some(at) => (at + 2).min(first.chars().count()),
            None => first.chars().count(),
        };
        let width = (below.rect[2] - below.rect[0]) * word as f32 / chars as f32;
        right - above.rect[2] < width + 0.6 * below.height.max(above.height)
    };
    // The block's text wraps: two rows or more reach its right edge, or run
    // on into the next.
    let wraps = rows.iter().filter(|row| at_edge(row)).count() >= 2
        || (1..rows.len())
            .filter(|&n| !coded[n - 1] && !coded[n] && wrapped(&rows[n - 1], &rows[n]))
            .count()
            >= 2;
    let gap = |above: &Row, below: &Row| {
        below.rect[1] - above.rect[3] > 0.8 * below.height.max(above.height)
    };
    let mut kinds: Vec<Kind> = Vec::with_capacity(rows.len());
    // The left edge of the text of the list item being read, and the rows
    // of the heading being read.
    let mut item_left = 0.0f32;
    let mut heading_rows = 0usize;
    for (number, row) in rows.iter().enumerate() {
        if coded[number] {
            kinds.push(Kind::Code);
            continue;
        }
        if let Some(table) = tables
            .iter()
            .position(|table| (table.start..table.end).contains(&number))
        {
            kinds.push(Kind::Table(table));
            continue;
        }
        let previous = number.checked_sub(1).map(|n| (&rows[n], kinds[n]));
        // A heading goes on over the rows of its size right below it.
        if let Some((above, Kind::Heading | Kind::HeadingMore)) = previous
            && heading_rows < MAX_HEADING_ROWS
            && !gap(above, row)
            && !row.bullet
            && (sizes.is_some() && row.size >= LARGE || (row.bold && above.bold))
            && (0.8..=1.2).contains(&(row.size / above.size))
        {
            heading_rows += 1;
            kinds.push(Kind::HeadingMore);
            continue;
        }
        if row.bullet {
            item_left = row.rect[0];
            kinds.push(Kind::ListItem);
            continue;
        }
        if let Some((above, Kind::ListItem | Kind::ListMore)) = previous
            && !gap(above, row)
            && row.rect[0] >= item_left - 0.5 * row.height
            && !(row.bold && !above.bold)
        {
            kinds.push(Kind::ListMore);
            continue;
        }
        let opens = previous.is_none_or(|(above, kind)| {
            kind != Kind::Text || gap(above, row) || !(wraps && wrapped(above, row))
        });
        let sized = sizes.is_some()
            && (row.size >= HEADING || (row.bold && opens && row.size >= BOLD_HEADING));
        if !sized {
            kinds.push(Kind::Text);
            continue;
        }
        let characters = row.text.chars().filter(|c| !c.is_whitespace()).count();
        kinds.push(
            if row.pieces.len() > 1
                || !(1..=MAX_HEADING_CHARS).contains(&characters)
                || row.text.ends_with([',', '，', ';', '；', '、'])
            {
                Kind::Text
            } else {
                heading_rows = 1;
                Kind::Heading
            },
        );
    }
    // A fence longer than any run of backticks in the code.
    let fence = code.as_ref().map(|code| {
        let longest = lines[groups[code.rows.start].start..groups[code.rows.end - 1].end]
            .iter()
            .flat_map(|line| line.text.split(|c| c != '`').map(str::len))
            .max()
            .unwrap_or(0);
        "`".repeat(longest.max(2) + 1)
    });
    let large = |row: &Row| sizes.is_some() && row.size >= LARGE;
    let mut pending: Option<Block> = None;
    let flush = |pending: &mut Option<Block>, out: &mut Out| match pending.take() {
        Some(Block::Text(text)) => out.blocks.push(Block::Text(joined_emphasis(&text))),
        Some(block) => out.blocks.push(block),
        None => {}
    };
    let mut regions = regions.into_iter().peekable();
    // A region was written since the last row: the next row opens a block.
    let mut fresh = false;
    for (number, (group, row)) in groups.into_iter().zip(&rows).enumerate() {
        while let Some(region) = regions.next_if(|region| region.bounds[1] < row.rect[1]) {
            flush(&mut pending, out);
            out.region(region);
            fresh = true;
        }
        for line in &lines[group.clone()] {
            out.line(line);
        }
        let kind = kinds[number];
        // A table is written whole at its first row.
        if let Kind::Table(table) = kind
            && tables[table].start != number
        {
            continue;
        }
        let edge = code
            .as_ref()
            .is_some_and(|code| code.rows.start == number || code.rows.end == number);
        let separator = if number == 0 || fresh {
            "\n\n"
        } else {
            let last = &rows[number - 1];
            match (kinds[number - 1], kind) {
                _ if edge => "\n\n",
                (_, Kind::HeadingMore) => " ",
                (Kind::Code, Kind::Code) if !gap(last, row) => "\n",
                (Kind::ListItem | Kind::ListMore, Kind::ListItem) => "\n",
                (Kind::ListItem | Kind::ListMore, Kind::ListMore) => "\n  ",
                (Kind::Text, Kind::Text)
                    if !gap(last, row)
                        && ((wraps && wrapped(last, row)) || (large(last) && large(row))) =>
                {
                    "\n"
                }
                _ => "\n\n",
            }
        };
        fresh = false;
        let body = row.styled.as_deref().unwrap_or(&row.text);
        let piece = match kind {
            Kind::Code => {
                let code = code.as_ref().expect("a code row");
                let fence = fence.as_ref().expect("a fence");
                let mut piece = String::new();
                if code.rows.start == number {
                    piece.push_str(fence);
                    piece.push('\n');
                }
                let indent = layout::indent(code, lines[group.start].bounds[0]);
                piece.extend(std::iter::repeat_n(' ', indent));
                piece.push_str(&row.text);
                if code.rows.end == number + 1 {
                    piece.push('\n');
                    piece.push_str(fence);
                }
                piece
            }
            Kind::Table(table) => table_markdown(&tables[table].cells),
            Kind::Heading | Kind::HeadingMore => row.text.clone(),
            Kind::ListItem => format!("- {body}"),
            Kind::ListMore | Kind::Text => body.to_owned(),
        };
        if separator == "\n\n" {
            flush(&mut pending, out);
            pending = Some(match kind {
                Kind::Heading | Kind::HeadingMore => Block::Heading {
                    size: row.size,
                    text: piece,
                },
                _ => Block::Text(piece),
            });
        } else {
            match &mut pending {
                Some(Block::Heading { size, text }) => {
                    *size = size.max(row.size);
                    text.push_str(separator);
                    text.push_str(&piece);
                }
                Some(Block::Text(text)) => {
                    text.push_str(separator);
                    text.push_str(&piece);
                }
                _ => pending = Some(Block::Text(piece)),
            }
        }
    }
    flush(&mut pending, out);
    for region in regions {
        out.region(region);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    // The macOS tests below are about Vision, which is no longer what the
    // public entry points read with by default.
    #[cfg(target_os = "macos")]
    fn recognize_rgb(image: image::RgbImage, cfg: &Value) -> Result<OcrResult> {
        read(&pixels::prepare_rgb(image)?, &configured(cfg)?)
    }

    #[cfg(target_os = "macos")]
    fn recognize(bytes: &[u8], cfg: &Value) -> Result<OcrResult> {
        read(&pixels::prepare(bytes)?, &configured(cfg)?)
    }

    #[test]
    fn the_thread_budget_is_the_processors_unless_a_valid_number_is_set() {
        assert_eq!(budget_from(None, 8), 8);
        assert_eq!(budget_from(Some(" 3 "), 8), 3);
        for bad in ["", "0", "-2", "many", "5000"] {
            assert_eq!(budget_from(Some(bad), 8), 8, "{bad}");
        }
    }

    #[test]
    fn the_capped_warning_names_the_region_limit() {
        assert_eq!(
            capped_warning("this image"),
            "Local OCR read only the first 1,000 text regions of this image, from the top; \
             the text below them is missing."
        );
        assert!(capped_warning("x").contains(&thousands(MAX_TEXT_REGIONS)));
        for (n, grouped) in [
            (0, "0"),
            (999, "999"),
            (1000, "1,000"),
            (1_234_567, "1,234,567"),
        ] {
            assert_eq!(thousands(n), grouped);
        }
    }

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

    /// Only `en` itself, absent or written in any case, is the default language
    /// policy; a region, or any other language, is read as that language alone.
    #[test]
    fn only_a_bare_en_is_the_default_language() {
        let spelling = |lang: Value| configured(&json!({"ocr":{"lang":lang}})).unwrap();
        assert_eq!(configured(&json!({})).unwrap(), "en");
        assert_eq!(configured(&json!({"ocr":{}})).unwrap(), "en");
        for default in ["en", "EN", " En "] {
            assert_eq!(spelling(json!(default)), "en");
        }
        for explicit in ["en-US", "en_us", "en-GB", "zh", "ja", "ko", "fr", "english"] {
            assert_ne!(spelling(json!(explicit)), "en", "{explicit}");
        }
        assert_eq!(tag(&spelling(json!("en_US"))), "en-us");
        assert_eq!(tag("en"), "en-US");
    }

    #[test]
    #[cfg(any(not(target_os = "macos"), feature = "portable-media"))]
    fn portable_unread_warning_explains_uncertainty_and_model_repair() {
        let warning = paddle_unread_message("PDF page 3");
        assert!(warning.starts_with("Local OCR could not read PDF page 3:"));
        for cause in [
            "unsupported script",
            "low image quality",
            "optional Korean model",
        ] {
            assert!(warning.contains(cause), "{warning}");
        }
        assert!(warning.contains("only the lines read with confidence are kept"));
        assert!(warning.contains("Set ocr.lang"));
        assert!(warning.contains("markitai doctor --fix"));
        assert!(!warning.contains("Korean reading found nothing better"));
    }

    #[test]
    fn the_unread_warning_names_its_subject_and_the_setting_to_change() {
        let warning = unread_warning("PDF page 3");
        assert!(warning.starts_with("Local OCR could not read PDF page 3: "));
        assert!(warning.contains("only the lines read with confidence are kept"));
        assert!(warning.contains("Set ocr.lang to the language of the text"));
        assert!(!unread_warning("this image").contains("PDF page"));
        // The languages this system reads are listed when they are known.
        let languages = ["en-US".to_owned(), "th-TH".to_owned()];
        assert!(unread_message("this image", &languages).ends_with(
            "the language of the text, one that this system's Vision recognizer reads: \
                 en-US, th-TH."
        ));
        assert!(unread_message("this image", &[]).ends_with("the language of the text."));
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
        system_font_lines("Hiragino Sans GB", lines, size)
    }

    /// Lines drawn by the in-process SVG renderer in the macOS system font
    /// `family`, black on white, `size` pixels per em.
    #[cfg(target_os = "macos")]
    fn system_font_lines(family: &str, lines: &[&str], size: u32) -> image::RgbImage {
        use resvg::{tiny_skia, usvg};
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

    #[cfg(target_os = "macos")]
    #[test]
    fn vision_reads_small_japanese_text_enlarged_and_recovers_a_dropped_kanji() {
        // Read as drawn, this system's recognizer drops 環 from 環境変数 at
        // both sizes and stretches a neighbouring box over it.
        let lines = [
            "キャッシュの保存先は環境変数で変更できます。テストの際",
            "には一時ディレクトリを指定してください。",
        ];
        let config = json!({"ocr":{"lang":"ja"}});
        let small = recognize_rgb(system_font_lines("Hiragino Sans", &lines, 16), &config);
        if vision_unavailable_under_rosetta(&small) {
            return;
        }
        let small = small.unwrap();
        assert!(small.scale > 1.25, "{}", small.scale);
        assert_eq!(
            small.text.split_whitespace().collect::<String>(),
            lines.concat()
        );
        // At 17 pixels per em the lines are read once and 環 is recovered.
        let larger =
            recognize_rgb(system_font_lines("Hiragino Sans", &lines, 17), &config).unwrap();
        assert_eq!(larger.scale, 1.0);
        assert_eq!(
            larger.text.split_whitespace().collect::<String>(),
            lines.concat()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn vision_reads_korean_text_it_missed_again_enlarged_but_not_small_hangul() {
        // At 16 pixels per em this system's recognizer returns no text for
        // these lines; a fast reading finds them, and an enlarged copy is read.
        let lines = [
            "오늘 아침은 조금 쌀쌀해서 역 앞 카페에서 따뜻한 라테를",
            "주문했습니다. 직원이 새로운 메뉴도 있다고 알려 주어서",
            "다음에는 유자차를 마셔 볼 생각입니다.",
        ];
        let expected: String = lines.concat().split_whitespace().collect();
        let config = json!({"ocr":{"lang":"ko"}});
        let missed = system_font_lines("Arial Unicode MS", &lines, 16);
        let (width, height) = missed.dimensions();
        let missed = recognize_rgb(missed, &config);
        if vision_unavailable_under_rosetta(&missed) {
            return;
        }
        let missed = missed.unwrap();
        // The fast reading's lines, about 15 pixels tall, set the factor.
        assert!(
            missed.scale > 1.25 && missed.scale < 3.0,
            "{}",
            missed.scale
        );
        assert_eq!(missed.text.split_whitespace().collect::<String>(), expected);
        assert_eq!(missed.boxes.len(), 3);
        for [_, top, right, bottom] in missed.boxes {
            assert!(right <= width as f32 && bottom <= height as f32 && bottom - top < 24.0);
        }
        // One pixel smaller the lines are read, and small Hangul is read once.
        let small =
            recognize_rgb(system_font_lines("Arial Unicode MS", &lines, 15), &config).unwrap();
        assert_eq!(small.scale, 1.0);
        assert_eq!(small.text.split_whitespace().collect::<String>(), expected);
        // At 11 pixels the recognizer drops 돗 from 돗자리를; it is recovered.
        let lines = [
            "봄이 되면 강변의 벚꽃길에 꽃이 한꺼번에 피어납니다. 밤에는",
            "조명이 켜지고, 돗자리를 펴고 앉아 꽃구경을 즐기는 사람들로",
            "붐빕니다.",
        ];
        let recovered =
            recognize_rgb(system_font_lines("Arial Unicode MS", &lines, 11), &config).unwrap();
        assert_eq!(recovered.scale, 1.0);
        assert_eq!(
            recovered.text.split_whitespace().collect::<String>(),
            lines.concat().split_whitespace().collect::<String>()
        );
        // A blank image stays blank.
        let blank = image::RgbImage::from_pixel(400, 200, image::Rgb([255; 3]));
        let blank = recognize_rgb(blank, &config).unwrap();
        assert!(blank.text.is_empty() && blank.boxes.is_empty());
        assert_eq!(blank.scale, 1.0);
    }

    #[cfg(target_os = "macos")]
    fn squeezed(text: &str) -> String {
        text.split_whitespace().collect()
    }

    /// Reads the lines drawn in `font` at `size` under the default language
    /// (no `ocr.lang` set) and requires the Vision `language` to have read them
    /// exactly, as `explicit`, written out as `ocr.lang`, reads them.
    #[cfg(target_os = "macos")]
    fn default_reads(font: &str, lines: &[&str], size: u32, language: &str, explicit: &str) {
        let image = system_font_lines(font, lines, size);
        let result = recognize_rgb(image.clone(), &json!({"ocr":{"enabled":true}}));
        if vision_unavailable_under_rosetta(&result) {
            return;
        }
        let result = result.unwrap();
        assert_eq!(result.language, language, "{font} {size}");
        assert_eq!(
            squeezed(&result.text),
            squeezed(&lines.concat()),
            "{font} {size}"
        );
        assert!(!result.unread);
        // The default adds no step of its own to a language's reading.
        let written = recognize_rgb(image, &json!({"ocr":{"lang":explicit}})).unwrap();
        assert_eq!(written.text, result.text, "{font} {size}");
        assert_eq!(written.scale, result.scale, "{font} {size}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn the_default_language_reads_chinese_japanese_and_korean_without_configuration() {
        let chinese = [
            "含链接注释、图片、曲线标记或圆角裁剪的页面也能重建；",
            "宽度相同的边框块不再并入表格网格。",
        ];
        // Small and ordinary text, with the enlarged second reading.
        default_reads("Hiragino Sans GB", &chinese, 13, "zh-Hans", "zh");
        default_reads("Hiragino Sans GB", &chinese, 25, "zh-Hans", "zh");
        let japanese = [
            "キャッシュの保存先は環境変数で変更できます。テストの際",
            "には一時ディレクトリを指定してください。",
        ];
        // Kana: Japanese is read once more, with its own aids.
        default_reads("Hiragino Sans", &japanese, 16, "ja-JP", "ja");
        default_reads("Hiragino Sans", &japanese, 24, "ja-JP", "ja");
        let korean = [
            "오늘 아침은 조금 쌀쌀해서 역 앞 카페에서 따뜻한 라테를",
            "주문했습니다. 직원이 새로운 메뉴도 있다고 알려 주어서",
            "다음에는 유자차를 마셔 볼 생각입니다.",
        ];
        // Vision's Chinese recognizer makes Han of Hangul; Korean is read too.
        default_reads("Arial Unicode MS", &korean, 24, "ko-KR", "ko");
        default_reads("Arial Unicode MS", &korean, 16, "ko-KR", "ko");
    }

    /// A scan-like page of Chinese with a table: drawn in a system font, then
    /// blurred and recompressed as a poor JPEG, the way the parity corpus's
    /// scanned pages are. Vision cannot run in a macOS virtual machine, so the
    /// runners skip this test by name.
    #[cfg(target_os = "macos")]
    #[test]
    fn vision_reads_a_degraded_scan_of_chinese_with_a_table_by_default() {
        let lines = [
            "第一章 概述",
            "本文档用于测试中文文本提取、表格识别和光学字符识别。",
            "Markitai 将各种文档格式转换为 Markdown，并可选地使用大语言模型进行增强",
            "第二章 数据表",
            "项目    数量    单价",
            "苹果    12    3.50",
            "香蕉    30    1.20",
            "结论：所有字符都应该被正确保留，没有乱码。",
        ];
        let page = system_font_lines("Hiragino Sans GB", &lines, 22);
        let blurred = image::imageops::blur(&page, 0.8);
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 55)
            .encode_image(&blurred)
            .unwrap();
        let scan = image::load_from_memory(&jpeg).unwrap().to_rgb8();
        let result = recognize_rgb(scan, &json!({"ocr":{"enabled":true}}));
        if vision_unavailable_under_rosetta(&result) {
            return;
        }
        let result = result.unwrap();
        let text = squeezed(&result.text);
        for phrase in [
            "第一章概述",
            "本文档用于测试中文文本提取",
            "项目数量单价",
            "苹果",
            "所有字符都应该被正确保留",
        ] {
            assert!(text.contains(phrase), "{phrase}: {}", result.text);
        }
        assert!(!result.unread);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn the_default_language_reads_a_line_of_chinese_that_english_returns_latin_words_for() {
        // English alone returns `* PDF. Word, Excel SEAT.`, with full
        // confidence, for this line of Chinese with three Latin words.
        let line = ["支持 PDF、Word、Excel 与图片转换。"];
        let image = system_font_lines("Hiragino Sans GB", &line, 24);
        let english = recognize_rgb(image.clone(), &json!({"ocr":{"lang":"en-US"}}));
        if vision_unavailable_under_rosetta(&english) {
            return;
        }
        let english = english.unwrap();
        assert!(!english.text.contains('持') && english.language.eq_ignore_ascii_case("en-US"));
        let result = recognize_rgb(image, &json!({"ocr":{"enabled":true,"lang":"en"}})).unwrap();
        assert_eq!(result.language, "zh-Hans");
        assert_eq!(squeezed(&result.text), squeezed(&line.concat()));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn the_default_language_keeps_english_as_english_reads_it_and_a_written_language_stands() {
        let default = json!({"ocr":{"enabled":true}});
        let written = ["en-US", "en_us"].map(|spelling| json!({"ocr":{"lang":spelling}}));
        let three = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs 1234567890.",
            "Local OCR runs on this machine without any network.",
        ];
        // Three lines, one line, and a short line: English stands.
        for lines in [&three[..], &three[..1], &["Hello World"][..]] {
            let image = system_font_lines("Helvetica", lines, 24);
            let result = recognize_rgb(image.clone(), &default);
            if vision_unavailable_under_rosetta(&result) {
                return;
            }
            let result = result.unwrap();
            assert_eq!(result.language, "en-US");
            assert_eq!((result.scale, result.unread), (1.0, false));
            assert_eq!(squeezed(&result.text), squeezed(&lines.concat()));
            for config in &written {
                let same = recognize_rgb(image.clone(), config).unwrap();
                assert_eq!(same.text, result.text);
            }
        }
        // `en-US` names English alone: Chinese is not read, and nothing replaces it.
        let chinese = ["宽度相同的边框块不再并入表格网格。"];
        let chinese = system_font_lines("Hiragino Sans GB", &chinese, 25);
        for config in &written {
            let result = recognize_rgb(chinese.clone(), config).unwrap();
            assert!(result.language.eq_ignore_ascii_case("en-US"));
            assert!(!result.text.chars().any(cjk::han), "{}", result.text);
            assert!(!result.unread);
        }
        // A blank image stays blank, and is not reported as unread.
        let blank = image::RgbImage::from_pixel(400, 200, image::Rgb([255; 3]));
        let blank = recognize_rgb(blank, &default).unwrap();
        assert!(blank.text.is_empty() && blank.boxes.is_empty() && !blank.unread);
    }

    #[test]
    fn lines_keep_rows_paragraphs_confidence_and_blank_image_distinction() {
        let line = |text: &str, bounds, confidence| Line {
            text: text.into(),
            bounds,
            confidence,
            direction: [1.0, 0.0],
            marks: Default::default(),
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
            direction: [1.0, 0.0],
            marks: Default::default(),
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
        // A receipt's items and prices stay rows, of a table; two short
        // prose lines stay rows.
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
            text.starts_with("| Coffee beans | $4.50 |\n| --- | --- |\n| Oat milk | $4.50 |"),
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

    /// Draw a line of `words` (text, stroke, shear) from `left` at `top`,
    /// each word one stem per letter (at most six), and return it as read.
    fn drawn(image: &mut image::RgbImage, left: u32, top: u32, words: &[(&str, u32, f32)]) -> Line {
        let mut x = left;
        for (text, stroke, shear) in words {
            let letters = text
                .chars()
                .filter(|c| c.is_alphanumeric())
                .count()
                .clamp(1, 6) as u32;
            x = page::tests::word(image, x, top, letters, *stroke, *shear) + 12;
        }
        let text: Vec<&str> = words.iter().map(|(text, ..)| *text).collect();
        sized(
            &text.join(" "),
            [left as f32, top as f32, (x - 12) as f32, (top + 28) as f32],
        )
    }

    /// A printed page as the recognizer reads it and as its pixels show it:
    /// a title, a paragraph with bold and italic words over two lines, a
    /// list whose bullets the recognizer dropped, a ruled table whose header
    /// it read as one line across two cells, and a chart with its labels.
    #[test]
    fn a_printed_page_reads_with_emphasis_bullets_a_ruled_table_and_a_chart() {
        let mut image = page::tests::paper(1000, 1300);
        let plain = |text| (text, 2, 0.0);
        let bold = |text| (text, 4, 0.0);
        let italic = |text| (text, 2, 0.3);
        let mut lines = vec![sized("Report", [100., 30., 400., 100.])];
        // Lines of the paragraph run to the page's right edge (justified).
        for (top, words) in [
            (140, vec![plain("alpha"), bold("beta"), bold("gamma")]),
            (
                172,
                vec![bold("delta"), plain("eps"), italic("zeta"), italic("theta")],
            ),
        ] {
            let mut line = drawn(&mut image, 100, top, &words);
            line.bounds[2] = 700.;
            lines.push(line);
        }
        lines.push(drawn(
            &mut image,
            100,
            204,
            &[plain("closing"), plain("words"), plain("here")],
        ));
        for (top, text) in [(260, "first"), (300, "second")] {
            page::tests::fill(&mut image, [150, top + 10, 158, top + 18], [0; 3]);
            lines.push(drawn(&mut image, 180, top, &[plain(text), plain("item")]));
        }
        // The table: three rows of two cells.
        for y in [400, 440, 480, 520] {
            page::tests::fill(&mut image, [100, y, 700, y + 1], [180; 3]);
        }
        for x in [100, 400, 699] {
            page::tests::fill(&mut image, [x, 400, x + 1, 521], [180; 3]);
        }
        let header = drawn(&mut image, 110, 406, &[plain("Name")]);
        let right = drawn(&mut image, 410, 406, &[plain("Value")]);
        lines.push(sized(
            "Name Value",
            [header.bounds[0], 406., right.bounds[2], 434.],
        ));
        lines.push(drawn(&mut image, 110, 446, &[plain("one")]));
        lines.push(drawn(&mut image, 410, 446, &[plain("first")]));
        lines.push(drawn(&mut image, 110, 486, &[plain("two")]));
        // The chart: bars, one pale yellow, with axis numbers and a label.
        page::tests::fill(&mut image, [300, 650, 340, 850], [0, 60, 140]);
        page::tests::fill(&mut image, [350, 700, 390, 850], [255, 204, 0]);
        page::tests::fill(&mut image, [400, 600, 440, 850], [255, 69, 0]);
        lines.push(drawn(&mut image, 250, 640, &[plain("10")]));
        lines.push(drawn(&mut image, 250, 830, &[plain("0")]));
        lines.push(drawn(&mut image, 320, 860, &[plain("Row"), plain("1")]));
        lines.push(drawn(
            &mut image,
            100,
            950,
            &[plain("after"), plain("the"), plain("chart")],
        ));
        let result = assemble_with(
            lines,
            1000,
            1300,
            Some(Pixels {
                image: &image,
                figures: true,
            }),
        )
        .unwrap();
        assert_eq!(
            result.text,
            "# Report\n\nalpha **beta gamma\ndelta** eps *zeta theta*\nclosing words here\n\n\
             - first item\n- second item\n\n\
             | Name | Value |\n| --- | --- |\n| one | first |\n| two |  |\n\n\
             after the chart"
        );
        assert_eq!(result.figures.len(), 1);
        assert!(result.blocks.contains(&Block::Figure(0)));
        let [l, t, r, b] = result.figures[0].bounds;
        assert!(
            l <= 250. && t <= 600. && r >= 440. && b >= 888.,
            "{:?}",
            result.figures[0].bounds
        );
        // The chart's labels are the picture's, not text the Markdown reads.
        let read: Vec<&str> = result.lines.iter().map(|line| line.text.as_str()).collect();
        assert!(
            !read.contains(&"Row 1") && read.contains(&"Name Value"),
            "{read:?}"
        );
        // Without pictures asked for (an image input), the labels stay text.
        let again = assemble_with(
            result.raw.clone(),
            1000,
            1300,
            Some(Pixels {
                image: &image,
                figures: false,
            }),
        )
        .unwrap();
        assert!(
            again.text.contains("Row 1") && again.figures.is_empty(),
            "{}",
            again.text
        );
    }

    #[test]
    fn heading_levels_are_tiers_of_sizes_and_blocks_render_as_markdown() {
        assert_eq!(tiers_of(&mut [1.4, 2.2, 1.2, 1.45, 1.38]), [2.2, 1.38, 1.2]);
        let sizes = [2.2, 1.38, 1.2];
        assert_eq!(
            [2.3, 2.2, 1.5, 1.38, 1.25, 1.0].map(|size| level(&sizes, size)),
            [1, 1, 2, 2, 3, 3]
        );
        let blocks = vec![
            Block::Heading {
                size: 1.4,
                text: "Section".into(),
            },
            Block::Text("Body.".into()),
            Block::Figure(0),
            Block::Figure(1),
        ];
        let figure = |index: usize| (index == 0).then(|| "![Chart](a.jpg)".to_owned());
        assert_eq!(
            render(&blocks, &sizes, &figure),
            "## Section\n\nBody.\n\n![Chart](a.jpg)"
        );
        // Alone on its page, a heading is of the first level.
        assert_eq!(
            render(&blocks[..1], &tiers(&blocks), &|_| None),
            "# Section"
        );
    }

    /// The body is the prose's height, not a photo's key labels'; with the
    /// pixels, sizes are the type's, steady where boxes vary; 16 pt under
    /// 18 pt (0.89) is a level of its own; a short word does not make a bold
    /// line light.
    #[test]
    fn sizes_come_from_prose_and_type_and_short_words_do_not_unbold_a_line() {
        let mut lines: Vec<Line> = (0..12)
            .map(|i| sized("Esc", [10. * i as f32, 0., 10. * i as f32 + 8., 15.]))
            .collect();
        for top in [40., 80., 120.] {
            lines.push(sized(
                "A line of prose long enough",
                [0., top, 300., top + 28.],
            ));
        }
        assert_eq!(sizes(&lines).unwrap().body, 28.0);
        assert!(sizes(&lines).unwrap().ink.is_none());
        for line in &mut lines[12..] {
            line.marks.size = Some(20.0);
        }
        let mut heading = sized("Heading with a comma,", [0., 160., 300., 190.]);
        heading.marks.size = Some(28.2);
        let found = sizes(&lines).unwrap();
        assert_eq!(found.ink, Some(20.0));
        assert!((found.of(&heading) - 1.41).abs() < 0.001);
        assert_eq!(tiers_of(&mut [1.58, 1.41, 2.35]), [2.35, 1.58, 1.41]);
        let style = Style { body: Some(2.0) };
        let word = |stroke| page::Word {
            stroke,
            italic: false,
        };
        let mut bold = sized("In eleifend velit", [0., 0., 300., 28.]);
        bold.marks.words = Some(vec![word(2.1), word(3.2), word(3.1)]);
        assert!(style.bold(&bold));
        bold.marks.words = Some(vec![word(3.2), word(2.0), word(3.1)]);
        assert!(!style.bold(&bold));
    }

    #[test]
    fn emphasis_closed_at_a_line_end_and_opened_on_the_next_goes_on() {
        assert_eq!(joined_emphasis("a **b c**\n**d** e"), "a **b c\nd** e");
        assert_eq!(joined_emphasis("*a*\n  *b*"), "*a\n  b*");
        assert_eq!(joined_emphasis("***a***\n***b***"), "***a\nb***");
        // Different runs, a list's items and a bold end before italic stay.
        for kept in ["**a**\n*b*", "- **a**\n- **b**", "**a** \n**b**", "a\nb"] {
            assert_eq!(joined_emphasis(kept), kept);
        }
    }

    #[test]
    fn read_bullets_start_list_items_and_captions_stay_text() {
        assert_eq!(bullet_glyph("• item"), Some("• ".len()));
        assert_eq!(bullet_glyph("- item"), Some(2));
        assert_eq!(bullet_glyph("-item"), None);
        assert_eq!(bullet_glyph("item"), None);
        assert!(caption("Figure 2: sales") && caption("图 1 销量") && !caption("Row 1"));
    }

    fn sized(text: &str, bounds: [f32; 4]) -> Line {
        Line {
            text: text.into(),
            bounds,
            confidence: 1.0,
            direction: [1.0, 0.0],
            marks: Default::default(),
        }
    }

    /// The lines the portable engine reads on the parity corpus's scanned
    /// page of Chinese, Japanese, Korean and a table (150 DPI), with their
    /// rectangles: the page comes out as headings, paragraphs and a table,
    /// with its letter-spaced Latin closed up, as the text PDF it was printed
    /// from reads.
    #[test]
    fn a_scanned_page_reads_as_headings_paragraphs_and_a_table() {
        let page = vec![
            sized("第一章 概述", [145., 114., 378., 158.]),
            sized(
                "本文档用于测试中文文本提取、表格识别和光学字符识别。",
                [149., 202., 785., 230.],
            ),
            sized(
                "M a r k i t a i 将各种文档格式转换为 Ma r k d o w n，并可选地使用大语言模型进行增强",
                [147., 252., 1239., 281.],
            ),
            sized(
                "日本語のテキストも含まれています:東京は日本の首都です。",
                [150., 302., 836., 330.],
            ),
            sized(
                "한국어 문장도 포함합니다: 서울은 대한민국의 수도입니다.",
                [148., 350., 910., 381.],
            ),
            sized("第二章 数据表", [147., 391., 417., 432.]),
            sized("项目  数量  单价", [147., 475., 455., 508.]),
            sized("苹果 | 1 2 | 3.50", [146., 525., 494., 558.]),
            sized("香蕉 | 30 | 1. 20", [148., 576., 493., 607.]),
            sized(
                "结论：所有字符都应该被正确保留，没有乱码。",
                [148., 626., 663., 657.],
            ),
        ];
        assert_eq!(
            assemble(page, 1240, 1755).unwrap().text,
            "# 第一章 概述\n\n\
             本文档用于测试中文文本提取、表格识别和光学字符识别。\n\n\
             Markitai 将各种文档格式转换为 Markdown，并可选地使用大语言模型进行增强\n\n\
             日本語のテキストも含まれています:東京は日本の首都です。\n\n\
             한국어 문장도 포함합니다: 서울은 대한민국의 수도입니다.\n\n\
             # 第二章 数据表\n\n\
             | 项目 | 数量 | 单价 |\n\
             | --- | --- | --- |\n\
             | 苹果 | 12 | 3.50 |\n\
             | 香蕉 | 30 | 1.20 |\n\n\
             结论：所有字符都应该被正确保留，没有乱码。"
        );
    }

    /// Wrapped prose stays one paragraph; a heading wrapped over two rows is
    /// one heading, and a smaller one is of the second level.
    #[test]
    fn wrapped_rows_and_wrapped_headings_stay_together() {
        let mut page = vec![
            sized("A Title", [100., 0., 300., 40.]),
            sized("A section heading that runs on", [0., 60., 380., 86.]),
            sized("to a second row", [0., 87., 200., 112.]),
        ];
        for (row, words) in [
            "First prose row runs to the edge",
            "and wraps on to this row",
            "A sentence alone.",
        ]
        .iter()
        .enumerate()
        {
            let right = if row == 0 { 400. } else { 250. };
            page.push(sized(
                words,
                [0., 130. + 22. * row as f32, right, 148. + 22. * row as f32],
            ));
        }
        page.push(sized("Another sentence alone.", [0., 196., 240., 214.]));
        page.push(sized(
            "A last row runs to the edge again",
            [0., 218., 398., 236.],
        ));
        assert_eq!(
            assemble(page, 400, 240).unwrap().text,
            "# A Title\n\n\
             ## A section heading that runs on to a second row\n\n\
             First prose row runs to the edge\nand wraps on to this row\n\n\
             A sentence alone.\n\n\
             Another sentence alone.\n\n\
             A last row runs to the edge again"
        );
    }

    /// Ragged-right prose: each row stops short of the widest by less than
    /// the next row's first word, so it wrapped there; the paragraphs are
    /// set apart by a gap, and a row with room left for the next word ends
    /// its paragraph.
    #[test]
    fn ragged_right_rows_stay_one_paragraph() {
        let paragraphs: [&[&str]; 3] = [
            &[
                "The allotment committee met in the old seed shed to",
                "review the state of the plots, the water barrels and",
                "borrow during the growing season.",
            ],
            &[
                "Members agreed that the gate should be repainted soon",
                "before the spring fair, and that the compost bins need",
                "emptying.",
            ],
            &["Short line.", "Another short line after it."],
        ];
        let mut page = Vec::new();
        let mut top = 0.;
        for paragraph in paragraphs {
            for row in paragraph.iter() {
                let right = row.chars().count() as f32 * 9.;
                page.push(sized(row, [0., top, right, top + 18.]));
                top += 24.;
            }
            top += 20.;
        }
        assert_eq!(
            assemble(page, 520, 320).unwrap().text,
            "The allotment committee met in the old seed shed to\n\
             review the state of the plots, the water barrels and\n\
             borrow during the growing season.\n\n\
             Members agreed that the gate should be repainted soon\n\
             before the spring fair, and that the compost bins need\n\
             emptying.\n\n\
             Short line.\n\n\
             Another short line after it."
        );
    }

    /// A table without rules: cells in columns, one read as part of its
    /// neighbour, a wrapped cell, and a header without a first cell.
    #[test]
    fn cells_in_columns_are_a_table_with_wrapped_and_empty_cells() {
        let mut page = vec![
            sized("Name", [40., 0., 100., 20.]),
            sized("Note", [200., 0., 260., 20.]),
        ];
        let rows = [
            ("1", "Apples", Some("Red")),
            ("2", "Pears and more Green", None),
            ("3", "Plums", Some("Blue")),
        ];
        for (row, (number, name, note)) in rows.iter().enumerate() {
            let top = 30. + 40. * row as f32;
            page.push(sized(number, [0., top, 10., top + 20.]));
            let right = if note.is_some() { 120. } else { 260. };
            page.push(sized(name, [40., top, right, top + 20.]));
            if let Some(note) = note {
                page.push(sized(note, [200., top, 240., top + 20.]));
            }
        }
        page.push(sized("cont.", [40., 92., 80., 110.]));
        assert_eq!(
            assemble(page, 300, 160).unwrap().text,
            "|  | Name | Note |\n\
             | --- | --- | --- |\n\
             | 1 | Apples | Red |\n\
             | 2 | Pears and more Green cont. |  |\n\
             | 3 | Plums | Blue |"
        );
        // Two rows divided by rules are a table; two in columns are not.
        let ruled = vec![
            sized("a | b", [0., 0., 100., 20.]),
            sized("1 | 2", [0., 30., 100., 50.]),
        ];
        assert_eq!(
            assemble(ruled, 100, 60).unwrap().text,
            "| a | b |\n| --- | --- |\n| 1 | 2 |"
        );
        let laid_out = vec![
            sized("Left", [0., 0., 40., 20.]),
            sized("Right", [100., 0., 150., 20.]),
            sized("Left", [0., 30., 40., 50.]),
            sized("Right", [100., 30., 150., 50.]),
        ];
        assert!(!assemble(laid_out, 150, 60).unwrap().text.contains('|'));
    }

    /// The gutter is looked for left to right: a right column set a little
    /// higher than the left one is still a column.
    #[test]
    fn a_right_column_starting_higher_is_still_read_after_the_left_one() {
        let line = |text: String, left: f32, top: f32, right: f32| Line {
            text,
            bounds: [left, top, right, top + 10.],
            confidence: 1.0,
            direction: [1.0, 0.0],
            marks: Default::default(),
        };
        let mut page = Vec::new();
        for row in 0..4 {
            let top = 14. * row as f32;
            page.push(line(
                format!("Left column prose line {row}"),
                0.,
                top + 3.,
                140.,
            ));
            page.push(line(
                format!("Right column prose line {row}"),
                160.,
                top,
                300.,
            ));
        }
        let text = assemble(page, 300, 60).unwrap().text;
        let left = (0..4)
            .map(|row| format!("Left column prose line {row}"))
            .collect::<Vec<_>>()
            .join("\n");
        let right = (0..4)
            .map(|row| format!("Right column prose line {row}"))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(text, format!("{left}\n\n{right}"));
    }

    /// The reading order comes from the lines' positions alone: the same page
    /// handed over in any order reads the same.
    #[test]
    fn reading_order_does_not_depend_on_the_recognizer_order() {
        let line = |text: String, left: f32, top: f32, right: f32| Line {
            text,
            bounds: [left, top, right, top + 10.],
            confidence: 1.0,
            direction: [1.0, 0.0],
            marks: Default::default(),
        };
        let mut page = vec![line(
            "A title across both columns of the page".into(),
            0.,
            0.,
            300.,
        )];
        for row in 0..4 {
            let top = 20. + 14. * row as f32;
            page.push(line(format!("Left column prose line {row}"), 0., top, 140.));
            page.push(line(
                format!("Right column prose line {row}"),
                160.,
                top,
                300.,
            ));
        }
        page.push(line(
            "A closing line across both columns".into(),
            0.,
            80.,
            300.,
        ));
        page.push(line("Last paragraph".into(), 0., 160., 120.));
        let copy = |lines: &[Line]| -> Vec<Line> {
            lines
                .iter()
                .map(|l| line(l.text.clone(), l.bounds[0], l.bounds[1], l.bounds[2]))
                .collect()
        };
        let expected = assemble(copy(&page), 300, 200).unwrap().text;
        assert!(
            expected.starts_with(
                "A title across both columns of the page\n\nLeft column prose line 0\n"
            ) && expected.ends_with("A closing line across both columns\n\nLast paragraph"),
            "{expected}"
        );
        let mut state = 7_u64;
        for round in 0..6 {
            let mut shuffled = copy(&page);
            if round == 0 {
                shuffled.reverse();
            } else {
                for index in (1..shuffled.len()).rev() {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    shuffled.swap(index, (state >> 33) as usize % (index + 1));
                }
            }
            assert_eq!(
                assemble(shuffled, 300, 200).unwrap().text,
                expected,
                "round {round}"
            );
        }
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
        // `en` is the default policy; this sound English reading is its first.
        assert_eq!(result.language, "en-US");
        assert!(!result.unread);
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
