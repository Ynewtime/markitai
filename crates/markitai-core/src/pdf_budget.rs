//! The decompression budget of one PDF. Everything lopdf decodes for a
//! document (its object streams while it loads, and every content, font or
//! image stream read from it afterwards on this thread) draws from one
//! total, so a small file of many inflating streams cannot exhaust memory:
//! a limit per stream bounds none of that. The total is generous, 64 times
//! the file and at least 1 GiB (a stream compresses its text a few times
//! over, and a document is read in a few passes); `MARKITAI_PDF_DECOMPRESS_MIB`
//! sets it in mebibytes.

/// Names the budget in mebibytes when set to a positive whole number.
const VARIABLE: &str = "MARKITAI_PDF_DECOMPRESS_MIB";
const FLOOR: usize = 1 << 30;
const FACTOR: usize = 64;

/// The budget of a PDF of `file_len` bytes.
pub(crate) fn limit(file_len: usize) -> usize {
    std::env::var(VARIABLE)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&mebibytes| mebibytes > 0)
        .map_or_else(
            || file_len.saturating_mul(FACTOR).max(FLOOR),
            |mebibytes| mebibytes.saturating_mul(1 << 20),
        )
}

/// Runs `work`, which reads the PDF `bytes`, under its budget (or under the
/// one already in force, when it reads part of a document being converted).
pub(crate) fn within<T>(bytes: &[u8], work: impl FnOnce() -> T) -> T {
    lopdf::with_decompress_budget(limit(bytes.len()), work)
}

#[cfg(test)]
mod tests {
    use lopdf::{Document, Object, Stream, dictionary};

    /// A one-page PDF whose content is `copies` compressed streams that
    /// each inflate to `inflated` bytes of PDF white space and some text.
    pub(crate) fn inflating(copies: usize, inflated: usize) -> Vec<u8> {
        let mut pdf = Document::with_version("1.7");
        let tree = pdf.new_object_id();
        let font = pdf.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica"
        });
        let mut content = b"BT /F1 12 Tf 72 700 Td (Words of the page.) Tj ET".to_vec();
        content.resize(inflated, b' ');
        let streams: Vec<Object> = (0..copies)
            .map(|_| {
                let mut stream = Stream::new(dictionary! {}, content.clone());
                stream.compress().unwrap();
                pdf.add_object(stream).into()
            })
            .collect();
        let page = pdf.add_object(dictionary! {
            "Type" => "Page", "Parent" => tree, "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            "Contents" => streams,
            "Resources" => dictionary! { "Font" => dictionary! { "F1" => font } },
        });
        pdf.objects.insert(
            tree,
            dictionary! { "Type" => "Pages", "Kids" => vec![page.into()], "Count" => 1 }.into(),
        );
        let catalog = pdf.add_object(dictionary! { "Type" => "Catalog", "Pages" => tree });
        pdf.trailer.set("Root", catalog);
        let mut bytes = Vec::new();
        pdf.save_to(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn a_document_draws_every_stream_from_one_budget() {
        // Eight streams of 2 MiB each from a file of a few kilobytes.
        let bytes = inflating(8, 2 << 20);
        assert!(bytes.len() < 64 * 1024, "{}", bytes.len());
        let document = Document::load_mem(&bytes).unwrap();
        let page = *document.get_pages().values().next().unwrap();
        let decode_all = || {
            document
                .get_page_contents(page)
                .into_iter()
                .map(|id| {
                    document
                        .get_object(id)
                        .and_then(Object::as_stream)
                        .and_then(Stream::decompressed_content)
                        .map(|content| content.len())
                })
                .collect::<Vec<_>>()
        };
        let total = decode_all();
        assert!(
            total
                .iter()
                .all(|decoded| decoded.as_ref().is_ok_and(|&len| len == 2 << 20))
        );
        // Under a 5 MiB budget the first two streams decode and the rest
        // fail, though each alone is within it.
        let bounded = lopdf::with_decompress_budget(5 << 20, decode_all);
        assert_eq!(bounded.iter().filter(|decoded| decoded.is_ok()).count(), 2);
        let refused = bounded
            .iter()
            .find_map(|decoded| decoded.as_ref().err())
            .unwrap();
        assert!(
            matches!(
                refused,
                lopdf::Error::Decompress(lopdf::DecompressError::MemoryLimitExceeded { limit })
                    if *limit == 5 << 20
            ),
            "{refused:?}"
        );
        // The default budget converts the document.
        let markdown = super::within(&bytes, || crate::formats::extract_pdf_pages(&bytes))
            .unwrap()
            .finish()
            .unwrap()
            .markdown;
        assert!(markdown.contains("Words of the page."), "{markdown}");
    }

    #[test]
    fn the_budget_is_generous_and_can_be_named() {
        assert_eq!(super::limit(1024), 1 << 30);
        assert_eq!(super::limit(100 << 20), 6400 << 20);
    }
}
