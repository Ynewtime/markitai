//! ReportLab lists (`fixtures/reportlab-lists`): ListFlowable draws each
//! marker as a run of its own, larger than the text and a little lower, with
//! the text hanging to its right. The reader once wrote `## •Item` headings
//! and `1Item` paragraphs, and glued the paragraph after the list to its
//! last item.
use super::*;

#[test]
fn reportlab_bullets_and_numbers_are_markdown_lists() {
    let markdown = extract(include_bytes!(
        "fixtures/reportlab-lists/reportlab-lists.pdf"
    ))
    .unwrap()
    .markdown;
    for text in [
        "# **Ridge Walk**\n",
        "\n## **Packing list**\n\n- A waterproof jacket with a hood, because the ridge path is exposed and the forecast changes quickly in the afternoon.\n- Two litres of water and a flask of tea.\n- A paper map of the valley and a compass, in case the phone loses its signal behind the northern shoulder of the hill.\n\nEveryone carries their own lunch; the café at the top closes at three.\n",
        "\n## **Route**\n\n1. Park at the village hall and follow the lane past the mill.\n2. Cross the footbridge and climb the stone steps to the gate.\n3. Turn left along the wall until the cairn comes into view.\n\nThe walk takes about four hours with a stop at the cairn.",
    ] {
        assert!(markdown.contains(text), "{text:?} in {markdown}");
    }
    assert!(!markdown.contains('•'), "{markdown}");
}

#[test]
fn reportlabs_anonymous_title_is_no_title() {
    // ReportLab writes `(anonymous)` when a document is given no title: the
    // fixture with its Info title set so, padded to keep every offset.
    let original = include_bytes!("fixtures/reportlab-lists/reportlab-lists.pdf");
    let (named, anonymous) = (
        &b"/Title (Ridge Walk)"[..],
        &b"/Title (\\(anonymous\\))"[..],
    );
    let at = original
        .windows(named.len())
        .position(|window| window == named)
        .unwrap();
    let keywords = b"/Keywords () ";
    let dropped = original[..at]
        .windows(keywords.len())
        .rposition(|window| window == keywords)
        .unwrap();
    let mut bytes = [
        &original[..dropped],
        &original[dropped + keywords.len()..at],
    ]
    .concat();
    bytes.extend_from_slice(anonymous);
    bytes.resize(at + named.len(), b' ');
    bytes.extend_from_slice(&original[at + named.len()..]);
    assert_eq!(bytes.len(), original.len());

    let document = extract(&bytes).unwrap();
    // The page's own title stands in, as for a PDF without one.
    assert_eq!(
        document
            .metadata
            .get("title")
            .and_then(|title| title.as_str()),
        Some("Ridge Walk")
    );
}
