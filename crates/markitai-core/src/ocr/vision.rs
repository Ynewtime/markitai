use super::{
    Line, MAX_LINES, MAX_TEXT, Result, failure, pixel_bounds, pixels::Prepared, supported_language,
};
use objc2::{AnyThread, rc::autoreleasepool};
use objc2_foundation::{NSArray, NSData, NSDictionary, NSString};
use objc2_vision::{
    VNImageRequestHandler, VNRecognizeTextRequest, VNRequest, VNRequestTextRecognitionLevel,
};

pub(super) fn recognize(image: &Prepared, requested: &str) -> Result<Vec<Line>> {
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
            failure(&format!(
                "Vision text recognition failed: {}",
                error.localizedDescription()
            ))
        })?;
        let observations = request
            .results()
            .ok_or_else(|| failure("Vision returned no recognition result"))?;
        if observations.len() > MAX_LINES {
            return Err(failure("recognized line count exceeds the safety limit"));
        }
        let mut lines = Vec::with_capacity(observations.len());
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
                image.width,
                image.height,
            )?;
            lines.push(Line {
                text: text.to_string(),
                confidence: candidate.confidence(),
                bounds,
            });
        }
        Ok(lines)
    })
}
