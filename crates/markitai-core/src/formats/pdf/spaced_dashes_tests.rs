//! A reportlab newsletter page of spaced and in-word hyphens
//! (`fixtures/spaced-dashes`): a hyphen or a dash set between spaces stays
//! spaced as printed, as Python markitai 1.2.0 reads it, whether it sits in
//! a run with its words or in a run of its own; in-word hyphens and a word
//! broken at a line end read as before.
use super::*;

#[test]
fn spaced_hyphens_and_dashes_stay_as_printed() {
    let markdown = extract(include_bytes!("fixtures/spaced-dashes/spaced-dashes.pdf"))
        .unwrap()
        .markdown;
    for text in [
        "Harbour Gazette - Spring Issue",
        "open Monday - Friday, and the ferry runs North - South twice a day.",
        "Buses run Inland – Coastal in summer — not in winter.",
        "Tide tables - Winter Edition",
        // In-word hyphens, in one run or as runs of their own.
        "fit lower-energy lamps and a well-known pump",
        "the nineteen-sixties.",
        "Fit a lower-cost lamp and a long-life bulb.",
        // No other use of `conversion` on the page: the break is kept.
        "check the conver- sion of the old net shed",
    ] {
        assert!(markdown.contains(text), "{text}: {markdown}");
    }
    for joined in [
        "Gazette-Spring",
        "Monday-Friday",
        "North-South",
        "tables-Winter",
    ] {
        assert!(!markdown.contains(joined), "{joined}: {markdown}");
    }
}
