use super::{
    Line, MAX_LINES, MAX_TEXT, Result, failure, pixel_bounds, pixels::Prepared,
    recognition_failure, supported_language,
};
use crate::system_frameworks::{self, Framework};
use objc2::{AnyThread, rc::autoreleasepool};
use objc2_foundation::{NSArray, NSData, NSDictionary, NSString};
use objc2_vision::{
    VNImageRequestHandler, VNRecognizeTextRequest, VNRequest, VNRequestTextRecognitionLevel,
};

// Chinese, Japanese and Korean recognition aids.
use super::{cjk, pixels::MAX_PIXELS};
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

/// Recognize text, reporting rectangles in a `space` of pixels (the original
/// image when `image` is an enlarged copy). For Chinese, Japanese and Korean,
/// `may_enlarge` lets a reading of small text, or one that found no text at
/// all, stop early with an enlargement factor; otherwise characters dropped
/// inside wide character boxes are recovered by reading those regions again.
pub(super) fn recognize(
    image: &Prepared,
    requested: &str,
    space: (u32, u32),
    may_enlarge: bool,
) -> Result<Reading> {
    // Vision and Foundation classes are looked up by name below.
    system_frameworks::open(Framework::Vision).map_err(|message| failure(&message))?;
    autoreleasepool(|_| {
        let request = VNRecognizeTextRequest::new();
        request.setRecognitionLevel(VNRequestTextRecognitionLevel::Accurate);
        request.setUsesLanguageCorrection(true);
        // SAFETY: The caller checks macOS >= 11, the request is initialized and
        // confined to this call, and the selector has no additional preconditions.
        let languages =
            unsafe { request.supportedRecognitionLanguagesAndReturnError() }.map_err(|error| {
                failure(&format!(
                    "cannot obtain supported recognition languages: {}",
                    error.localizedDescription()
                ))
            })?;
        let supported: Vec<String> = (0..languages.len())
            .map(|index| languages.objectAtIndex(index).to_string())
            .collect();
        let selected = supported_language(requested, &supported)?;
        let languages = NSArray::from_retained_slice(&[NSString::from_str(&selected)]);
        request.setRecognitionLanguages(&languages);
        let data = NSData::with_bytes(&image.png);
        let handler = VNImageRequestHandler::initWithData_options(
            VNImageRequestHandler::alloc(),
            &data,
            &NSDictionary::new(),
        );
        let requests = NSArray::<VNRequest>::from_slice(&[&request]);
        handler.performRequests_error(&requests).map_err(|error| {
            // objc2 substitutes an error in this domain when the method
            // returns failure without setting one.
            let reported = (error.domain().to_string() != "__objc2.missingError")
                .then(|| error.localizedDescription().to_string());
            recognition_failure(reported.as_deref(), system_frameworks::translated())
        })?;
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
            // Vision request. Its immutable rectangle getter is valid here.
            let rectangle = unsafe { observation.boundingBox() };
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
            lines.push(Line {
                text: text.to_string(),
                confidence: candidate.confidence(),
                bounds,
            });
            recognized.push(candidate);
        }
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
        for (line, candidate) in lines.iter_mut().zip(&recognized) {
            recover(&handler, &languages, script, line, candidate, &mut budget);
        }
        Ok(Reading {
            lines,
            enlarge: None,
        })
    })
}

/// Read the regions of a line's suspiciously wide letter boxes again and
/// insert a letter that the second reading places there. A failed
/// second reading leaves the line as it was.
fn recover(
    handler: &VNImageRequestHandler,
    languages: &NSArray<NSString>,
    script: cjk::Script,
    line: &mut Line,
    candidate: &VNRecognizedText,
    budget: &mut usize,
) {
    let characters: Vec<char> = line.text.chars().collect();
    if *budget == 0 || characters.iter().filter(|c| script.letter(**c)).count() < cjk::MIN_LETTERS {
        return;
    }
    let length = candidate.string().length();
    let mut offset = 0;
    let boxes: Vec<Option<[f64; 4]>> = characters
        .iter()
        .map(|character| {
            let range = NSRange::new(offset, character.len_utf16());
            offset += character.len_utf16();
            if character.is_whitespace() || offset > length {
                return None;
            }
            // SAFETY: The range lies within the candidate's UTF-16 string, and
            // the candidate belongs to a completed request.
            let rectangle = unsafe { candidate.boundingBoxForRange_error(range) }.ok()?;
            // SAFETY: The rectangle observation was just returned by Vision.
            let b = unsafe { rectangle.boundingBox() };
            let b = [b.origin.x, b.origin.y, b.size.width, b.size.height];
            b.iter().all(|n| n.is_finite() && *n >= 0.0).then_some(b)
        })
        .collect();
    let mut insertions: Vec<(usize, char)> = Vec::new();
    for suspect in cjk::suspects(&characters, &boxes, script) {
        if *budget == 0 {
            break;
        }
        *budget -= 1;
        let request = VNRecognizeTextRequest::new();
        request.setRecognitionLevel(VNRequestTextRecognitionLevel::Accurate);
        request.setUsesLanguageCorrection(true);
        request.setRecognitionLanguages(languages);
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
        parts.sort_by(|a, b| a.0.total_cmp(&b.0));
        let reading: String = parts.into_iter().map(|(_, text)| text).collect();
        if let Some(found) = cjk::insertion(&characters, suspect.index, &reading, script)
            && !insertions.iter().any(|(index, _)| *index == found.0)
        {
            insertions.push(found);
        }
    }
    if !insertions.is_empty() {
        let mut characters = characters;
        insertions.sort_by_key(|&(index, _)| std::cmp::Reverse(index));
        for (index, character) in insertions {
            characters.insert(index, character);
        }
        line.text = characters.into_iter().collect();
    }
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
