//! A reportlab newsletter page of two columns under a centred title
//! (`fixtures/two-column`): a wrapped line of the left column ending in a
//! full stop short of the column's edge continues its paragraph when the
//! next word could not have fitted on it, as in a single column.
use super::*;

#[test]
fn a_full_stop_at_the_end_of_a_column_line_does_not_split_its_paragraph() {
    let markdown = extract(include_bytes!("fixtures/two-column/two-column.pdf"))
        .unwrap()
        .markdown;
    for text in [
        "A second item concerned the lights on the pier. Several lamps have failed. The board will replace",
        "Finally, the board discussed the summer regatta. Entries are already up on last year.",
        "The cost estimate came in lower than expected. Work will start once the spring tides have passed. Section 3 ends here.",
    ] {
        assert!(markdown.contains(text), "{text}: {markdown}");
    }
    // Each section is one paragraph under its heading.
    let paragraphs = markdown
        .split("\n\n")
        .filter(|block| !block.starts_with('#') && block.contains("ends here"))
        .count();
    assert_eq!(paragraphs, 5, "{markdown}");
}
