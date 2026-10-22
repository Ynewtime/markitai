//! Whether a page's text layer says what its glyphs draw (with `--ocr`).
//!
//! A legacy or custom-encoded font stores other codes than the letters it
//! draws: Latin codes for Devanagari or CJK glyphs, a shifted alphabet. The
//! text can look plausible enough to pass every text-only check of
//! [`text_trust`], and only the image tells. A few short stretches of each
//! font's text are drawn from the page, read by the OCR recognizer alone (no
//! detection) and compared with the layer by writing system and by
//! characters. A font that fails does so on every page, so each font is
//! sampled a few times per document, not on each page: the cost does not
//! grow with the page count. A page whose failing fonts carry
//! [`MAX_BAD_LETTER_SHARE`] of its letters is read by OCR instead.
//!
//! As Python markitai 1.2.1's `converter/pdf_text_trust.py` `GlyphCheck`
//! (8a757175). Never fatal: when drawing or reading fails, the check stops and
//! every page keeps the verdict of the text-only checks.

use super::text_trust::{self, Failure};
use super::{PdfPage, PdfPages, geometry};
use crate::pdf_raster::PdfRasterSession;
use crate::{Result, ocr};
use image::RgbImage;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use unicode_normalization::UnicodeNormalization;

/// Characters cut from a run for one sample, and the least letters (and
/// share of letters) a sample needs to say anything.
const SAMPLE_CHARS: usize = 20;
const SAMPLE_MIN_LETTERS: usize = 6;
const SAMPLE_MIN_LETTER_SHARE: f64 = 0.6;
/// Samples per font per document, and crops per page at most.
const SAMPLES_PER_FONT: usize = 2;
const MAX_CROPS_PER_PAGE: usize = 4;
/// A crop is drawn so its height is this many pixels: the recognizer's input
/// height, so it is not scaled again; at most eight times 72 dpi.
const CROP_HEIGHT_PX: f64 = 48.;
const MAX_ZOOM: f64 = 8.;
/// Below this similarity (difflib's ratio over letters and digits) a sample
/// does not say what its glyphs draw. OCR of clean text scores 0.9 and up, a
/// shifted or permuted alphabet under 0.3.
const MIN_SIMILARITY: f64 = 0.5;
/// A page fails when fonts whose samples disagree carry this share of its
/// letters.
const MAX_BAD_LETTER_SHARE: f64 = 0.3;
/// The kind of a page's failure ([`Failure::kind`]): pages that fail alike
/// share one warning.
pub(crate) const KIND: &str = "text that is not what its glyphs draw";

/// What OCR of a crop says of the layer's text for it.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Agreement {
    Agrees,
    /// Why not.
    Disagrees(String),
    /// OCR read too little to tell.
    Unsure,
}

/// Letters and digits only, NFKC and lower case: what OCR and the layer
/// should share (OCR reads punctuation and spacing in its own way).
fn comparable(text: &str) -> Vec<char> {
    text.nfkc()
        .flat_map(char::to_lowercase)
        .filter(|ch| ch.is_alphanumeric())
        .collect()
}

/// The longest common run of `a[a0..a1]` and `b[b0..b1]`, earliest in `a`
/// then in `b` on ties, as difflib's `find_longest_match` without junk.
fn longest_match(
    a: &[char],
    b: &[char],
    (a0, a1): (usize, usize),
    (b0, b1): (usize, usize),
) -> (usize, usize, usize) {
    let (mut best_a, mut best_b, mut best) = (a0, b0, 0);
    let mut lengths = vec![0usize; b1 - b0 + 1];
    for (i, ch) in a.iter().enumerate().take(a1).skip(a0) {
        let mut next = vec![0usize; b1 - b0 + 1];
        for (j, other) in b.iter().enumerate().take(b1).skip(b0) {
            if ch == other {
                let length = lengths[j - b0] + 1;
                next[j - b0 + 1] = length;
                if length > best {
                    (best_a, best_b, best) = (i + 1 - length, j + 1 - length, length);
                }
            }
        }
        lengths = next;
    }
    (best_a, best_b, best)
}

/// difflib's `SequenceMatcher(None, a, b, autojunk=False).ratio()`: twice
/// the characters of the matching runs (found as Ratcliff and Obershelp
/// do, longest first, then on either side) over both lengths.
fn similarity(a: &[char], b: &[char]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.;
    }
    let mut matched = 0;
    let mut queue = vec![((0, a.len()), (0, b.len()))];
    while let Some(((a0, a1), (b0, b1))) = queue.pop() {
        let (i, j, length) = longest_match(a, b, (a0, a1), (b0, b1));
        if length == 0 {
            continue;
        }
        matched += length;
        if a0 < i && b0 < j {
            queue.push(((a0, i), (b0, j)));
        }
        if i + length < a1 && j + length < b1 {
            queue.push(((i + length, a1), (j + length, b1)));
        }
    }
    2. * matched as f64 / (a.len() + b.len()) as f64
}

/// Does the text the layer gives for a crop (`layer`) say what OCR reads in
/// it (`seen`)?
pub(crate) fn compare_sample(layer: &str, seen: &str) -> Agreement {
    if seen.chars().filter(|ch| ch.is_alphabetic()).count() < 2 {
        return Agreement::Unsure;
    }
    if let (Some(layer_script), Some(seen_script)) = (
        text_trust::dominant_script(layer),
        text_trust::dominant_script(seen),
    ) && layer_script != seen_script
    {
        return Agreement::Disagrees(format!(
            "the layer reads {layer_script}, the page shows {seen_script}"
        ));
    }
    let (layer, seen) = (comparable(layer), comparable(seen));
    if !layer.is_empty() && !seen.is_empty() && similarity(&layer, &seen) < MIN_SIMILARITY {
        return Agreement::Disagrees("the layer's text is not what its glyphs draw".into());
    }
    Agreement::Agrees
}

#[derive(Default)]
struct Tally {
    agree: usize,
    disagree: usize,
    why: String,
    /// The texts sampled so far (a running header is sampled once).
    seen: HashSet<String>,
}

impl Tally {
    fn settled(&self) -> bool {
        self.agree + self.disagree >= SAMPLES_PER_FONT
    }

    fn bad(&self) -> bool {
        self.disagree > self.agree
    }
}

struct Sample {
    font: String,
    /// Its font is drawn through its ToUnicode map (see [`Renderers`]).
    drawn_by_map: bool,
    text: String,
    /// `[left, top, right, bottom]` in points from the page's top-left corner.
    rect: [f64; 4],
}

/// A font's name without its subset tag (`ABCDEF+Arial` is `Arial`).
fn font_name(name: &str) -> &str {
    match name.split_once('+') {
        Some((tag, rest)) if tag.len() == 6 && tag.bytes().all(|b| b.is_ascii_uppercase()) => rest,
        _ => name,
    }
}

/// A page font, by resource tag.
struct PageFont {
    /// The key its samples are counted under: the font's name and a
    /// fingerprint of how it maps its codes (its ToUnicode stream, else its
    /// encoding). The same map on every page is one font, sampled once per
    /// document even when each page has its own copy; another map of the
    /// same name is another font.
    key: String,
    /// A composite font drawn from a host face through its ToUnicode map
    /// (see [`Renderers`]).
    drawn_by_map: bool,
}

fn page_fonts(pdf: &lopdf::Document, page: lopdf::ObjectId) -> HashMap<Vec<u8>, PageFont> {
    let mut fonts = HashMap::new();
    for (tag, font) in pdf.get_page_fonts(page).unwrap_or_default() {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        match font
            .get(b"ToUnicode")
            .and_then(|value| pdf.dereference(value))
        {
            Ok((_, lopdf::Object::Stream(stream))) => stream.content.hash(&mut hasher),
            _ => format!("{:?}", font.get(b"Encoding").ok()).hash(&mut hasher),
        }
        let name = font
            .get(b"BaseFont")
            .and_then(lopdf::Object::as_name)
            .map(|name| font_name(&String::from_utf8_lossy(name)).to_owned())
            .unwrap_or_else(|_| "?".into());
        let key = format!("{name}#{:016x}", hasher.finish());
        let drawn_by_map = drawn_by_map(pdf, font);
        fonts.insert(tag, PageFont { key, drawn_by_map });
    }
    fonts
}

/// The first [`SAMPLE_CHARS`] characters of a run, when they are mostly
/// letters of a script OCR reads, with where they stand on a page `height`
/// points high and `width` wide.
fn sample(
    font: &PageFont,
    item: &pdf_inspector::TextItem,
    (width, height): (f64, f64),
    readable: &[&str],
) -> Option<Sample> {
    if !item.advance_known || item.width <= 0. || item.rotation.abs() > 0.5 {
        return None;
    }
    let chars: Vec<char> = item.text.chars().collect();
    let start = chars.iter().take_while(|ch| ch.is_whitespace()).count();
    let mut end = (start + SAMPLE_CHARS).min(chars.len());
    while end > start && chars[end - 1].is_whitespace() {
        end -= 1;
    }
    let text: String = chars[start..end].iter().collect();
    let visible = text.chars().filter(|ch| !ch.is_whitespace()).count();
    let letters = text.chars().filter(|ch| ch.is_alphabetic()).count();
    if letters < SAMPLE_MIN_LETTERS || (letters as f64) < SAMPLE_MIN_LETTER_SHARE * visible as f64 {
        return None;
    }
    if !text_trust::dominant_script(&text).is_some_and(|script| readable.contains(&script)) {
        return None;
    }
    // Where the characters stand along the run, by their share of it.
    let step = f64::from(item.width) / chars.len() as f64;
    let left = f64::from(item.x) + start as f64 * step;
    let right = f64::from(item.x) + end as f64 * step;
    // The glyphs' box: from a descender below the baseline to an ascender
    // above it.
    let size = f64::from(item.height);
    let top = height - (f64::from(item.y) + 0.9 * size);
    let bottom = height - (f64::from(item.y) - 0.25 * size);
    let box_height = bottom - top;
    if box_height < 3. || right <= left {
        return None;
    }
    let (pad_x, pad_y) = (0.3 * box_height, 0.15 * box_height);
    let rect = [left - pad_x, top - pad_y, right + pad_x, bottom + pad_y];
    if rect[0] < 0. || rect[1] < 0. || rect[2] > width || rect[3] > height {
        return None;
    }
    Some(Sample {
        font: font.key.clone(),
        drawn_by_map: font.drawn_by_map,
        text,
        rect,
    })
}

/// Reads line images (the crops), one text per image in order.
pub(crate) type ReadLines<'a> = dyn Fn(Vec<RgbImage>) -> Result<Vec<String>> + 'a;

/// Compares a document's text layer with what its glyphs draw, per font,
/// pages in order. See the module.
struct GlyphCheck<'a> {
    readable: &'a [&'a str],
    read: &'a ReadLines<'a>,
    fonts: HashMap<String, Tally>,
    /// Line images read so far.
    crops: usize,
    /// The pages found failing so far, by number.
    failures: BTreeMap<usize, Failure>,
}

impl GlyphCheck<'_> {
    /// Why the page's text layer does not say what its glyphs draw, or
    /// `None`. `items` are the page's text runs, `keys` its fonts' keys by
    /// resource tag, `size` its visible box in points.
    fn judge(
        &mut self,
        renderers: &mut Renderers<'_>,
        number: usize,
        items: &[&pdf_inspector::TextItem],
        fonts: &HashMap<Vec<u8>, PageFont>,
        size: (f64, f64),
    ) -> Result<Option<Failure>> {
        let mut letters: Vec<(String, usize)> = Vec::new();
        let mut candidates: Vec<(String, Vec<Sample>)> = Vec::new();
        for item in items {
            let count = item.text.chars().filter(|ch| ch.is_alphabetic()).count();
            if count == 0 {
                continue;
            }
            let font = fonts
                .get(item.font_tag.as_bytes())
                .map(|font| PageFont {
                    key: font.key.clone(),
                    drawn_by_map: font.drawn_by_map,
                })
                .unwrap_or_else(|| PageFont {
                    key: font_name(&item.font).to_owned(),
                    drawn_by_map: false,
                });
            match letters.iter_mut().find(|(name, _)| *name == font.key) {
                Some((_, total)) => *total += count,
                None => letters.push((font.key.clone(), count)),
            }
            let tally = self.fonts.entry(font.key.clone()).or_default();
            if tally.settled() {
                continue;
            }
            if let Some(found) = sample(&font, item, size, self.readable)
                && !tally.seen.contains(&found.text)
            {
                match candidates.iter_mut().find(|(name, _)| *name == font.key) {
                    Some((_, samples)) => samples.push(found),
                    None => candidates.push((font.key, vec![found])),
                }
            }
        }
        let mut picked: Vec<Sample> = Vec::new();
        for (font, found) in candidates {
            let tally = self.fonts.get_mut(&font).expect("a tally per font");
            let mut need = SAMPLES_PER_FONT.saturating_sub(tally.agree + tally.disagree);
            // Spread over the page: every `step`th stretch, then the rest.
            let step = (found.len() / need.max(1)).max(1);
            let order: Vec<usize> = (0..found.len())
                .step_by(step)
                .chain(0..found.len())
                .collect();
            let mut found: Vec<Option<Sample>> = found.into_iter().map(Some).collect();
            for index in order {
                if need == 0 {
                    break;
                }
                let Some(sample) = found[index].take() else {
                    continue;
                };
                // A running header, a repeat.
                if !tally.seen.insert(sample.text.clone()) {
                    continue;
                }
                picked.push(sample);
                need -= 1;
            }
        }
        picked.truncate(MAX_CROPS_PER_PAGE);
        if !picked.is_empty() {
            let mut images = Vec::with_capacity(picked.len());
            for sample in &picked {
                let height = sample.rect[3] - sample.rect[1];
                let zoom = (CROP_HEIGHT_PX / height).clamp(1., MAX_ZOOM);
                let session = renderers.session(sample.drawn_by_map)?;
                images.push(session.render_region(number, 72. * zoom, sample.rect)?);
            }
            self.crops += images.len();
            let texts = (self.read)(images)?;
            for (sample, seen) in picked.iter().zip(&texts) {
                let agreement = compare_sample(&sample.text, seen);
                ocr::trace::line(format_args!(
                    "glyph-check font={} layer={:?} ocr={seen:?} -> {agreement:?}",
                    sample.font, sample.text
                ));
                let tally = self.fonts.get_mut(&sample.font).expect("a tally per font");
                match agreement {
                    Agreement::Agrees => tally.agree += 1,
                    Agreement::Disagrees(why) => {
                        tally.disagree += 1;
                        tally.why = why;
                    }
                    Agreement::Unsure => {}
                }
            }
        }
        let total: usize = letters.iter().map(|(_, count)| count).sum();
        let bad: Vec<&(String, usize)> = letters
            .iter()
            .filter(|(font, _)| self.fonts[font].bad())
            .collect();
        let bad_letters: usize = bad.iter().map(|(_, count)| count).sum();
        if total == 0 || (bad_letters as f64) < MAX_BAD_LETTER_SHARE * total as f64 {
            return Ok(None);
        }
        let (worst, _) = bad
            .iter()
            .max_by_key(|(_, count)| *count)
            .expect("a bad font");
        let name = worst.split('#').next().unwrap_or(worst);
        Ok(Some(Failure {
            kind: KIND,
            detail: format!("{} (font {name})", self.fonts[worst].why),
            script: None,
        }))
    }
}

/// Whether the glyph check has a say on `page`: a text layer the text-only
/// checks trusted, of enough text, that OCR would not read anyway.
fn checked(page: &PdfPage) -> bool {
    !page.needs_ocr
        && page.layer_failure.is_none()
        && page.ocr_layer.is_none()
        && !page.visibility_suspect
        && page.omitted_text.is_none()
        && page.markdown.trim().chars().count() >= text_trust::MIN_CHARS
}

/// What [`check`] did.
#[derive(Default)]
pub(crate) struct GlyphReport {
    /// Line images read.
    pub crops: usize,
    /// When a page failed and a font was drawn from its character collection
    /// rather than its ToUnicode map (see [`Renderers`]): the document to
    /// draw the pages from for OCR, so they show the glyphs checked.
    pub render: Option<Vec<u8>>,
}

/// Run the glyph check over the trusted pages of `pages` (the PDF `bytes`,
/// read for `readable` writing systems by `read`): a page that fails is
/// sent to OCR, with its failure. Never fatal: an error stops the check
/// where it is.
pub(crate) fn check(
    bytes: &[u8],
    pages: &mut PdfPages,
    readable: &[&str],
    read: &ReadLines<'_>,
) -> GlyphReport {
    let numbers: HashSet<u32> = pages
        .pages
        .iter()
        .filter(|page| checked(page))
        .map(|page| page.number as u32)
        .collect();
    if numbers.is_empty() {
        return GlyphReport::default();
    }
    let mut check = GlyphCheck {
        readable,
        read,
        fonts: HashMap::new(),
        crops: 0,
        failures: BTreeMap::new(),
    };
    // What failed before an error keeps its verdict.
    let render = match run(bytes, &numbers, &mut check) {
        Ok(render) => render,
        Err(error) => {
            ocr::trace::line(format_args!("glyph-check stopped: {error}"));
            None
        }
    };
    let failed = !check.failures.is_empty();
    for (number, failure) in std::mem::take(&mut check.failures) {
        if let Some(page) = pages.pages.iter_mut().find(|page| page.number == number) {
            page.needs_ocr = true;
            page.ocr_reason = Some(failure.reason());
            page.layer_failure = Some(failure);
        }
    }
    GlyphReport {
        crops: check.crops,
        render: render.filter(|_| failed),
    }
}

/// [`check`]'s pages `numbers` judged in order, their failures kept in
/// `check`. Returns the document drawn from character collections when one
/// was made.
fn run(
    bytes: &[u8],
    numbers: &HashSet<u32>,
    check: &mut GlyphCheck<'_>,
) -> Result<Option<Vec<u8>>> {
    let failed = |error: pdf_inspector::PdfError| crate::Error::Conversion(error.to_string());
    let loaded = pdf_inspector::LoadedPdf::load_mem(bytes).map_err(failed)?;
    let pdf = loaded.document();
    let ids = pdf.get_pages();
    let (items, rotations) = loaded
        .text_with_positions_and_rotations(Some(numbers), pdf_inspector::PositionOptions::new())
        .map_err(failed)?;
    let mut by_page: BTreeMap<u32, Vec<&pdf_inspector::TextItem>> = BTreeMap::new();
    for item in &items {
        if matches!(item.item_type, pdf_inspector::types::ItemType::Text) {
            by_page.entry(item.page).or_default().push(item);
        }
    }
    let mut renderers = Renderers {
        pdf,
        bytes,
        original: None,
        collection: None,
    };
    for (number, items) in by_page {
        // A page turned by its /Rotate, or whose text is: the items are not
        // where the page is drawn upright.
        if rotations.contains_key(&number) {
            continue;
        }
        let Some(&id) = ids.get(&number) else {
            continue;
        };
        let Some(frame) = geometry::frame(pdf, id) else {
            continue;
        };
        let size = (f64::from(frame.width), f64::from(frame.height));
        let fonts = page_fonts(pdf, id);
        if let Some(failure) = check.judge(&mut renderers, number as usize, &items, &fonts, size)? {
            check.failures.insert(number as usize, failure);
        }
    }
    Ok(renderers.collection.map(|(bytes, _)| bytes))
}

/// The page renderers of the check, opened when a crop first needs them:
/// one of the document, and one of a copy whose composite fonts drawn from
/// a host face have no ToUnicode map. The portable renderer picks such a
/// font's glyphs through Unicode: from the ToUnicode map when there is one,
/// else from the character collection's UCS2 map. A legacy font's map, the
/// very thing checked, would draw the letters it claims; without it the
/// glyphs are those of the CIDs, as MuPDF and CoreGraphics draw them.
struct Renderers<'a> {
    pdf: &'a lopdf::Document,
    bytes: &'a [u8],
    original: Option<PdfRasterSession>,
    collection: Option<(Vec<u8>, PdfRasterSession)>,
}

impl Renderers<'_> {
    fn session(&mut self, drawn_by_map: bool) -> Result<&PdfRasterSession> {
        if drawn_by_map {
            if self.collection.is_none() {
                let copy = drawn_by_collection(self.pdf)?;
                let session = PdfRasterSession::open(&copy)?;
                self.collection = Some((copy, session));
            }
            return Ok(&self.collection.as_ref().expect("opened").1);
        }
        if self.original.is_none() {
            self.original = Some(PdfRasterSession::open(self.bytes)?);
        }
        Ok(self.original.as_ref().expect("opened"))
    }
}

/// The document with no ToUnicode map in its fonts [`drawn_by_map`].
fn drawn_by_collection(pdf: &lopdf::Document) -> Result<Vec<u8>> {
    let fonts: Vec<lopdf::ObjectId> = pdf
        .objects
        .iter()
        .filter(|(_, object)| object.as_dict().is_ok_and(|font| drawn_by_map(pdf, font)))
        .map(|(&id, _)| id)
        .collect();
    let mut copy = pdf.clone();
    for id in fonts {
        if let Ok(font) = copy.get_dictionary_mut(id) {
            font.remove(b"ToUnicode");
        }
    }
    let mut bytes = Vec::new();
    copy.save_to(&mut bytes)
        .map_err(|error| crate::Error::Conversion(error.to_string()))?;
    Ok(bytes)
}

/// Whether `font` is a composite font a renderer draws from a host face
/// through its ToUnicode map, where its character collection's UCS2 map
/// would draw the glyphs of its CIDs: not embedded, with a ToUnicode map,
/// of a collection with such a map (Adobe-GB1, -CNS1, -Japan1, -Korea1).
fn drawn_by_map(pdf: &lopdf::Document, font: &lopdf::Dictionary) -> bool {
    if font.get(b"Subtype").and_then(lopdf::Object::as_name).ok() != Some(b"Type0".as_slice())
        || !font.has(b"ToUnicode")
    {
        return false;
    }
    let resolve = |value: &lopdf::Object| -> Option<lopdf::Object> {
        pdf.dereference(value).ok().map(|(_, value)| value.clone())
    };
    let Some(descendant) = font
        .get(b"DescendantFonts")
        .ok()
        .and_then(resolve)
        .and_then(|value| value.as_array().ok()?.first().cloned())
        .and_then(|value| resolve(&value))
    else {
        return false;
    };
    let Ok(descendant) = descendant.as_dict() else {
        return false;
    };
    let text = |dict: &lopdf::Dictionary, key: &[u8]| -> Option<Vec<u8>> {
        dict.get(key)
            .ok()
            .and_then(resolve)
            .and_then(|value| value.as_str().ok().map(<[u8]>::to_vec))
    };
    let known = descendant
        .get(b"CIDSystemInfo")
        .ok()
        .and_then(resolve)
        .and_then(|value| value.as_dict().ok().cloned())
        .is_some_and(|info| {
            text(&info, b"Registry").as_deref() == Some(b"Adobe".as_slice())
                && text(&info, b"Ordering").is_some_and(|ordering| {
                    [b"GB1".as_slice(), b"CNS1", b"Japan1", b"Korea1"]
                        .contains(&ordering.as_slice())
                })
        });
    known && !embedded(pdf, descendant)
}

/// Whether a descendant font embeds its program (or has no descriptor to
/// say).
fn embedded(pdf: &lopdf::Document, descendant: &lopdf::Dictionary) -> bool {
    descendant
        .get(b"FontDescriptor")
        .ok()
        .and_then(|value| pdf.dereference(value).ok())
        .and_then(|(_, value)| value.as_dict().ok())
        .is_none_or(|descriptor| {
            [b"FontFile".as_slice(), b"FontFile2", b"FontFile3"]
                .iter()
                .any(|key| descriptor.has(key))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ratio(a: &str, b: &str) -> f64 {
        similarity(
            &a.chars().collect::<Vec<_>>(),
            &b.chars().collect::<Vec<_>>(),
        )
    }

    #[test]
    fn similarity_is_difflibs_ratio() {
        // SequenceMatcher(None, a, b, autojunk=False).ratio()
        assert_eq!(ratio("abcd", "bcde"), 0.75);
        assert_eq!(ratio("", ""), 1.);
        assert_eq!(ratio("abc", ""), 0.);
        // A shifted alphabet shares a few letters by chance.
        assert!((ratio("thecommitteemetor", "uifdpnnjuuffnfups") - 1. / 17.).abs() < 1e-9);
        // "qabxcd" and "abycdf": blocks "ab" and "cd".
        assert!((ratio("qabxcd", "abycdf") - 8. / 12.).abs() < 1e-9);
    }

    #[test]
    fn samples_are_compared_by_writing_system_and_by_characters() {
        assert_eq!(
            compare_sample("the committee met", "The committee met"),
            Agreement::Agrees
        );
        // OCR's own spacing, case and punctuation do not count.
        assert_eq!(
            compare_sample("Ｔｈｅ ﬁrst", "the first,"),
            Agreement::Agrees
        );
        assert_eq!(
            compare_sample("uif dpnnjuuff nfu", "the committee met"),
            Agreement::Disagrees("the layer's text is not what its glyphs draw".into())
        );
        assert_eq!(
            compare_sample("gvuijomwshcaakpezdon", "本报告介绍了城市公共图书馆"),
            Agreement::Disagrees("the layer reads Latin, the page shows CJK".into())
        );
        // Too little read to tell.
        assert_eq!(compare_sample("the committee met", "1."), Agreement::Unsure);
    }

    #[test]
    fn a_subset_tag_is_not_part_of_a_fonts_name() {
        assert_eq!(font_name("ABCDEF+Arial-Bold"), "Arial-Bold");
        assert_eq!(font_name("Abcdef+Arial"), "Abcdef+Arial");
        assert_eq!(font_name("Helvetica"), "Helvetica");
    }

    #[test]
    fn a_non_embedded_cjk_font_with_a_map_is_drawn_from_its_collection() {
        use lopdf::{Object, dictionary};
        let font = |ordering: &str, embedded: bool| {
            let mut pdf = lopdf::Document::with_version("1.5");
            let mut descriptor = dictionary! { "Type" => "FontDescriptor", "FontName" => "Heiti" };
            if embedded {
                let program = pdf.add_object(lopdf::Stream::new(dictionary! {}, vec![0]));
                descriptor.set("FontFile2", program);
            }
            let descriptor = pdf.add_object(descriptor);
            let descendant = pdf.add_object(dictionary! {
                "Type" => "Font", "Subtype" => "CIDFontType0", "BaseFont" => "Heiti",
                "CIDSystemInfo" => dictionary! {
                    "Registry" => Object::string_literal("Adobe"),
                    "Ordering" => Object::string_literal(ordering),
                    "Supplement" => 2,
                },
                "FontDescriptor" => descriptor,
            });
            let map = pdf.add_object(lopdf::Stream::new(dictionary! {}, b"map".to_vec()));
            let id = pdf.add_object(dictionary! {
                "Type" => "Font", "Subtype" => "Type0", "BaseFont" => "Heiti",
                "Encoding" => "UniGB-UCS2-H",
                "DescendantFonts" => vec![Object::Reference(descendant)],
                "ToUnicode" => map,
            });
            (pdf, id)
        };
        let (pdf, id) = font("GB1", false);
        assert!(drawn_by_map(&pdf, pdf.get_dictionary(id).unwrap()));
        let copy = lopdf::Document::load_mem(&drawn_by_collection(&pdf).unwrap()).unwrap();
        assert!(!copy.get_dictionary(id).unwrap().has(b"ToUnicode"));
        // An embedded program draws its own glyphs; an Identity font has no
        // collection to draw from.
        for (ordering, embedded) in [("GB1", true), ("Identity", false)] {
            let (pdf, id) = font(ordering, embedded);
            assert!(!drawn_by_map(&pdf, pdf.get_dictionary(id).unwrap()));
        }
    }
}
