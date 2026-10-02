use super::{
    Line, MAX_LINES, MAX_TEXT, Result, failure, pixel_bounds, pixels::Prepared,
    recognition_failure, supported_language,
};
use crate::system_frameworks::{self, Framework};
use objc2::{AnyThread, rc::Retained, rc::autoreleasepool};
use objc2_foundation::{NSArray, NSData, NSDictionary, NSString};
use objc2_vision::{
    VNImageRequestHandler, VNRecognizeTextRequest, VNRequest, VNRequestTextRecognitionLevel,
};

// Chinese, Japanese and Korean recognition aids.
use super::{cjk, layout, pixels, pixels::MAX_PIXELS};
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_foundation::NSRange;
use objc2_vision::VNRecognizedText;

/// One recognition of an image.
pub(super) struct Reading {
    /// Lines with pixel rectangles in the requested coordinate space.
    pub lines: Vec<Line>,
    /// For small Chinese or Japanese text, or Chinese, Japanese or Korean
    /// text that the reading missed entirely, the factor to read the image
    /// again at; the lines are then this first reading, without recovered
    /// characters.
    pub enlarge: Option<f32>,
}

/// The languages the accurate recognizer of this system supports, asked once
/// per process (each recognition still checks its own request's list).
pub(super) fn languages() -> Vec<String> {
    static SUPPORTED: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    if let Some(languages) = SUPPORTED.get() {
        return languages.clone();
    }
    if system_frameworks::open(Framework::Vision).is_err() {
        return Vec::new();
    }
    let found = autoreleasepool(|_| {
        let request = VNRecognizeTextRequest::new();
        request.setRecognitionLevel(VNRequestTextRecognitionLevel::Accurate);
        supported(&request).unwrap_or_default()
    });
    if found.is_empty() {
        return found;
    }
    SUPPORTED.get_or_init(|| found).clone()
}

fn supported(request: &VNRecognizeTextRequest) -> Result<Vec<String>> {
    // SAFETY: The caller checks macOS >= 11, the request is initialized and
    // confined to this call, and the selector has no additional preconditions.
    let languages =
        unsafe { request.supportedRecognitionLanguagesAndReturnError() }.map_err(|error| {
            failure(&format!(
                "cannot obtain supported recognition languages: {}",
                error.localizedDescription()
            ))
        })?;
    Ok((0..languages.len())
        .map(|index| languages.objectAtIndex(index).to_string())
        .collect())
}

/// An accurate request with language correction for `languages`.
fn request(languages: &NSArray<NSString>) -> Retained<VNRecognizeTextRequest> {
    let request = VNRecognizeTextRequest::new();
    request.setRecognitionLevel(VNRequestTextRecognitionLevel::Accurate);
    request.setUsesLanguageCorrection(true);
    request.setRecognitionLanguages(languages);
    request
}

fn perform(handler: &VNImageRequestHandler, request: &VNRecognizeTextRequest) -> Result<()> {
    let requests = NSArray::<VNRequest>::from_slice(&[request]);
    handler.performRequests_error(&requests).map_err(|error| {
        // objc2 substitutes an error in this domain when the method
        // returns failure without setting one.
        let reported = (error.domain().to_string() != "__objc2.missingError")
            .then(|| error.localizedDescription().to_string());
        recognition_failure(reported.as_deref(), system_frameworks::translated())
    })
}

/// The lines of a completed request, with rectangles in a `space` of pixels,
/// each with its top candidate and its normalized rectangle.
#[allow(clippy::type_complexity)]
fn observed(
    request: &VNRecognizeTextRequest,
    space: (u32, u32),
) -> Result<(Vec<Line>, Vec<(Retained<VNRecognizedText>, CGRect)>)> {
    let observations = request
        .results()
        .ok_or_else(|| failure("Vision returned no recognition result"))?;
    if observations.len() > MAX_LINES {
        return Err(failure("recognized line count exceeds the safety limit"));
    }
    let mut lines = Vec::with_capacity(observations.len());
    let mut recognized = Vec::with_capacity(observations.len());
    let mut text_bytes = 0usize;
    for index in 0..observations.len() {
        let observation = observations.objectAtIndex(index);
        let candidates = observation.topCandidates(1);
        let Some(candidate) = candidates.firstObject() else {
            continue;
        };
        let text = candidate.string();
        text_bytes = text_bytes
            .checked_add(text.len() + 2)
            .filter(|size| *size <= MAX_TEXT)
            .ok_or_else(|| failure("recognized text exceeds the safety limit"))?;
        // SAFETY: This initialized observation was returned by the completed
        // Vision request. Its immutable geometry getters are valid here.
        let (rectangle, left, right) = unsafe {
            (
                observation.boundingBox(),
                observation.topLeft(),
                observation.topRight(),
            )
        };
        let bounds = pixel_bounds(
            [
                rectangle.origin.x,
                rectangle.origin.y,
                rectangle.size.width,
                rectangle.size.height,
            ],
            space.0,
            space.1,
        )?;
        // The direction the text runs, in top-left pixel coordinates.
        let direction = [
            ((right.x - left.x) * f64::from(space.0)) as f32,
            ((left.y - right.y) * f64::from(space.1)) as f32,
        ];
        lines.push(Line {
            text: text.to_string(),
            confidence: candidate.confidence(),
            bounds,
            direction,
        });
        recognized.push((candidate, rectangle));
    }
    Ok((lines, recognized))
}

/// Recognize text, reporting rectangles in a `space` of pixels (the original
/// image when `image` is an enlarged copy). For Chinese, Japanese and Korean,
/// `may_enlarge` lets a reading of small text, or one that found no text at
/// all, stop early with an enlargement factor; otherwise characters dropped
/// inside wide character boxes are recovered by reading those regions again,
/// and, for Chinese and Japanese, spaces that the image shows between their
/// letters and Latin ones are kept.
pub(super) fn recognize(
    image: &Prepared,
    requested: &str,
    space: (u32, u32),
    may_enlarge: bool,
) -> Result<Reading> {
    // Vision and Foundation classes are looked up by name below.
    system_frameworks::open(Framework::Vision).map_err(|message| failure(&message))?;
    autoreleasepool(|_| {
        let probe = VNRecognizeTextRequest::new();
        probe.setRecognitionLevel(VNRequestTextRecognitionLevel::Accurate);
        let selected = supported_language(requested, &supported(&probe)?)?;
        let languages = NSArray::from_retained_slice(&[NSString::from_str(&selected)]);
        let request = request(&languages);
        let data = NSData::with_bytes(&image.png);
        let handler = VNImageRequestHandler::initWithData_options(
            VNImageRequestHandler::alloc(),
            &data,
            &NSDictionary::new(),
        );
        perform(&handler, &request)?;
        let (mut lines, recognized) = observed(&request, space)?;
        let Some(script) = cjk::script(requested) else {
            return Ok(Reading {
                lines,
                enlarge: None,
            });
        };
        if may_enlarge {
            // The first reading's line heights are in the image's own pixels.
            debug_assert_eq!(space, (image.width, image.height));
            let factor = if lines.is_empty() {
                // Vision occasionally finds no text in small print that it
                // reads at other sizes; a fast reading still finds the lines.
                let found = fast_lines(&handler, image.width, image.height);
                cjk::missed(&found, image.width, image.height, MAX_PIXELS)
            } else {
                cjk::enlargement(&lines, script, image.width, image.height, MAX_PIXELS)
            };
            if factor.is_some() {
                return Ok(Reading {
                    lines,
                    enlarge: factor,
                });
            }
        }
        let mut budget = cjk::MAX_REREADS;
        let mut ink = Ink::new(image);
        for (line, (candidate, rectangle)) in lines.iter_mut().zip(&recognized) {
            let mut insertions =
                recover(&handler, &languages, script, line, candidate, &mut budget);
            if script != cjk::Script::Korean {
                insertions.extend(spaces(&mut ink, line, candidate, *rectangle));
            }
            if !insertions.is_empty() {
                // Later positions first; at one position the space goes in
                // first, so that a recovered letter stays before it.
                crate::sort::by_key(&mut insertions, |&(index, c)| {
                    (std::cmp::Reverse(index), c != ' ')
                });
                let mut characters: Vec<char> = line.text.chars().collect();
                for (index, character) in insertions {
                    characters.insert(index, character);
                }
                line.text = characters.into_iter().collect();
            }
        }
        Ok(Reading {
            lines,
            enlarge: None,
        })
    })
}

/// Read the regions of a line's suspiciously wide letter boxes again and
/// return the letters that the second reading places there, as insertions
/// into the line's characters. A failed second reading inserts nothing.
fn recover(
    handler: &VNImageRequestHandler,
    languages: &NSArray<NSString>,
    script: cjk::Script,
    line: &Line,
    candidate: &VNRecognizedText,
    budget: &mut usize,
) -> Vec<(usize, char)> {
    let characters: Vec<char> = line.text.chars().collect();
    let mut insertions: Vec<(usize, char)> = Vec::new();
    if *budget == 0 || characters.iter().filter(|c| script.letter(**c)).count() < cjk::MIN_LETTERS {
        return insertions;
    }
    let boxes: Vec<Option<[f64; 4]>> = (0..characters.len())
        .map(|index| char_box(candidate, &characters, index))
        .collect();
    for suspect in cjk::suspects(&characters, &boxes, script) {
        if *budget == 0 {
            break;
        }
        *budget -= 1;
        let request = request(languages);
        let [x, y, width, height] = suspect.region;
        // SAFETY: The region is a nonempty rectangle inside the unit square, in
        // Vision's normalized lower-left coordinates, as the property requires.
        unsafe {
            request
                .setRegionOfInterest(CGRect::new(CGPoint::new(x, y), CGSize::new(width, height)));
        }
        let requests = NSArray::<VNRequest>::from_slice(&[&request]);
        if handler.performRequests_error(&requests).is_err() {
            continue;
        }
        let Some(observations) = request.results() else {
            continue;
        };
        let mut parts: Vec<(f64, String)> = (0..observations.len())
            .filter_map(|index| {
                let observation = observations.objectAtIndex(index);
                let text = observation.topCandidates(1).firstObject()?.string();
                // SAFETY: As for the line observations above.
                let left = unsafe { observation.boundingBox() }.origin.x;
                Some((left, text.to_string()))
            })
            .collect();
        crate::sort::by(&mut parts, |a, b| a.0.total_cmp(&b.0));
        let reading: String = parts.into_iter().map(|(_, text)| text).collect();
        if let Some(found) = cjk::insertion(&characters, suspect.index, &reading, script)
            && !insertions.iter().any(|(index, _)| *index == found.0)
        {
            insertions.push(found);
        }
    }
    insertions
}

/// The normalized `[x, y, width, height]` box of `characters[index]` in a
/// candidate whose string they are, when Vision gives one.
fn char_box(candidate: &VNRecognizedText, characters: &[char], index: usize) -> Option<[f64; 4]> {
    let character = *characters.get(index)?;
    if character.is_whitespace() {
        return None;
    }
    let offset: usize = characters[..index].iter().map(|c| c.len_utf16()).sum();
    if offset + character.len_utf16() > candidate.string().length() {
        return None;
    }
    let range = NSRange::new(offset, character.len_utf16());
    // SAFETY: The range lies within the candidate's UTF-16 string, and the
    // candidate belongs to a completed request.
    let rectangle = unsafe { candidate.boundingBoxForRange_error(range) }.ok()?;
    // SAFETY: The rectangle observation was just returned by Vision.
    let b = unsafe { rectangle.boundingBox() };
    let b = [b.origin.x, b.origin.y, b.size.width, b.size.height];
    b.iter().all(|n| n.is_finite() && *n >= 0.0).then_some(b)
}

/// The read image's pixels, decoded when a line first needs them.
struct Ink<'a> {
    image: &'a Prepared,
    pixels: Option<Option<image::RgbImage>>,
}

impl<'a> Ink<'a> {
    fn new(image: &'a Prepared) -> Self {
        Ink {
            image,
            pixels: None,
        }
    }

    fn pixels(&mut self) -> Option<&image::RgbImage> {
        self.pixels
            .get_or_insert_with(|| pixels::decode(self.image).ok())
            .as_ref()
    }
}

/// Spaces between a Chinese or Japanese letter and a Latin letter or digit
/// that the reading left out although the image shows one: an ink-free gap
/// of at least [`layout::SPACE`] line heights between the two characters.
/// Spaces the reading has are kept: none was found where the text has none.
fn spaces(
    ink: &mut Ink,
    line: &Line,
    candidate: &VNRecognizedText,
    rectangle: CGRect,
) -> Vec<(usize, char)> {
    let characters: Vec<char> = line.text.chars().collect();
    let junctions = layout::junctions(&characters);
    if junctions.is_empty() {
        return Vec::new();
    }
    let (width, height) = (f64::from(ink.image.width), f64::from(ink.image.height));
    let band = [
        (1.0 - rectangle.origin.y - rectangle.size.height) * height,
        (1.0 - rectangle.origin.y) * height,
    ]
    .map(|n| n as f32);
    let area = [
        (rectangle.origin.x * width) as f32,
        band[0],
        ((rectangle.origin.x + rectangle.size.width) * width) as f32,
        band[1],
    ];
    let Some(pixels) = ink.pixels() else {
        return Vec::new();
    };
    let Some(background) = layout::background(pixels, area) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for index in junctions {
        let (Some(a), Some(b)) = (
            char_box(candidate, &characters, index),
            char_box(candidate, &characters, index + 1),
        ) else {
            continue;
        };
        let centre = |b: [f64; 4]| ((b[0] + b[2] / 2.0) * width) as f32;
        if layout::spaced(pixels, band, [centre(a), centre(b)], background) {
            found.push((index + 1, ' '));
        }
    }
    found
}

/// Readings of regions per image, at most.
const MAX_REGION_READS: usize = 24;

/// Read the regions that [`layout::rereads`] finds in a `language` reading
/// again, enlarged as [`layout::Reread::attempts`] lists, and add the missing
/// table cells and mended numbers they read. Lines keep the original image's
/// pixels. At most [`MAX_REGION_READS`] readings per image; a failed reading
/// changes nothing.
pub(super) fn reread(image: &Prepared, language: &str, lines: &mut Vec<Line>) {
    let latin = cjk::script(language).is_none();
    let layout::Found { rereads, digits } =
        layout::rereads(lines, image.width, image.height, latin);
    for (index, digit) in digits {
        lines[index].text = digit.to_string();
    }
    if rereads.is_empty() || system_frameworks::open(Framework::Vision).is_err() {
        return;
    }
    let Ok(selected) = supported_language(language, &languages()) else {
        return;
    };
    let Ok(rgb) = pixels::decode(image) else {
        return;
    };
    let mut budget = MAX_REGION_READS;
    autoreleasepool(|_| {
        let languages = NSArray::from_retained_slice(&[NSString::from_str(&selected)]);
        for reread in rereads {
            for (region, factor, margin) in reread.attempts() {
                if budget == 0 {
                    return;
                }
                budget -= 1;
                let Some(read) = read_region(&rgb, region, factor, margin, &languages) else {
                    continue;
                };
                match reread.target {
                    layout::Target::Cell {
                        span,
                        band,
                        max_chars,
                        numeric,
                    } => {
                        if let Some(cell) =
                            layout::cell(read, span, band, max_chars, numeric, latin)
                        {
                            lines.push(cell);
                            break;
                        }
                    }
                    layout::Target::Mend { index } => {
                        let reading: Vec<&str> = read.iter().map(|l| l.text.as_str()).collect();
                        if let Some(text) = layout::mend(&lines[index].text, &reading.join(" ")) {
                            lines[index].text = text;
                            break;
                        }
                    }
                }
            }
        }
    });
}

/// The lines a copy of `region` of `rgb`, enlarged `factor` times inside a
/// white `margin` (in original pixels), reads, left to right, with
/// rectangles in the original image's pixels.
fn read_region(
    rgb: &image::RgbImage,
    region: [f32; 4],
    factor: f32,
    margin: f32,
    languages: &NSArray<NSString>,
) -> Option<Vec<Line>> {
    let border = (margin * factor).round() as u32;
    let copy = pixels::crop(rgb, region, factor, border).ok()?;
    let request = request(languages);
    let handler = VNImageRequestHandler::initWithData_options(
        VNImageRequestHandler::alloc(),
        &NSData::with_bytes(&copy.png),
        &NSDictionary::new(),
    );
    perform(&handler, &request).ok()?;
    let (mut lines, _) = observed(&request, (copy.width, copy.height)).ok()?;
    // Vision may read a lone digit upside down (6 as 9): only upright lines count.
    lines.retain(|line| line.direction[0] > line.direction[1].abs());
    let [x0, y0, ..] = region.map(f32::floor);
    let border = border as f32;
    // Rectangles stay inside the original image, as assembly requires.
    let (width, height) = (rgb.width() as f32, rgb.height() as f32);
    for line in &mut lines {
        let [l, t, r, b] = line.bounds.map(|n| n - border);
        line.bounds = [
            (x0 + l / factor).clamp(0.0, width),
            (y0 + t / factor).clamp(0.0, height),
            (x0 + r / factor).clamp(0.0, width),
            (y0 + b / factor).clamp(0.0, height),
        ];
    }
    crate::sort::by(&mut lines, |a, b| a.bounds[0].total_cmp(&b.bounds[0]));
    Some(lines)
}

/// The pixel `[width, height]` of each text line that a fast reading finds in
/// a `width` by `height` image. Only the geometry is used: the fast recognizer
/// reads Latin script, so its text for these languages is meaningless. A
/// failed reading finds no lines.
fn fast_lines(handler: &VNImageRequestHandler, width: u32, height: u32) -> Vec<[f32; 2]> {
    let request = VNRecognizeTextRequest::new();
    request.setRecognitionLevel(VNRequestTextRecognitionLevel::Fast);
    request.setUsesLanguageCorrection(false);
    request.setRecognitionLanguages(&NSArray::from_retained_slice(&[NSString::from_str(
        "en-US",
    )]));
    let requests = NSArray::<VNRequest>::from_slice(&[&request]);
    if handler.performRequests_error(&requests).is_err() {
        return Vec::new();
    }
    let Some(observations) = request.results() else {
        return Vec::new();
    };
    (0..observations.len().min(MAX_LINES))
        .filter_map(|index| {
            // SAFETY: As for the line observations in `recognize`.
            let rectangle = unsafe { observations.objectAtIndex(index).boundingBox() };
            let line = [
                rectangle.size.width * f64::from(width),
                rectangle.size.height * f64::from(height),
            ];
            line.iter()
                .all(|side| side.is_finite() && *side > 0.0)
                .then(|| line.map(|side| side as f32))
        })
        .collect()
}
