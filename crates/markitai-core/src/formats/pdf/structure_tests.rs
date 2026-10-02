//! Block structure, escaping and links of the layout pass.
use super::*;
use lopdf::{
    Dictionary, Object, Stream,
    content::{Content, Operation},
    dictionary,
};

/// A positioned run as the extractor reports one: `width` in points from
/// `x`, the baseline at `y`.
fn run(text: &str, x: f32, y: f32, width: f32, size: f32, font: &str) -> TextItem {
    TextItem {
        text: text.into(),
        x,
        y,
        width,
        height: size,
        font: font.into(),
        font_tag: String::new(),
        legacy_symbol_rewrite: false,
        font_size: size,
        page: 1,
        is_bold: font.ends_with("Bold"),
        is_italic: false,
        font_weight: None,
        bold_source: None,
        fixed_pitch: Some(false),
        fill_color: None,
        stroke_color: None,
        render_mode: None,
        is_underline: false,
        is_strikeout: false,
        rotation: 0.,
        advance_known: true,
        item_type: ItemType::Text,
        mcid: None,
        baseline_shift: 0.,
    }
}

/// A 12pt run in the body face, its width estimated from its characters.
fn body(text: &str, x: f32, y: f32) -> TextItem {
    run(text, x, y, text.chars().count() as f32 * 6., 12., "Body")
}

fn faces() -> Faces {
    Faces {
        headings: vec![18.],
        body: 12.,
        body_font: Some("Body".into()),
        heading_fonts: HashSet::from(["Head-Bold".to_owned()]),
    }
}

/// The flow of `items` laid out in lines, with 18pt headings known.
fn flowed(items: Vec<TextItem>) -> String {
    flow(&lines(items), &[18.], &faces(), &Bullets(Vec::new())).unwrap()
}

#[test]
fn a_bold_line_at_the_body_size_set_apart_is_a_heading_below_the_sizes() {
    // The title links to the site's home page: a heading keeps the text.
    let mut title = run("Section title", 40., 760., 120., 18., "Head-Bold");
    title.item_type = ItemType::Link("https://example.com/".into());
    let markdown = flowed(vec![
        title,
        body(
            "Opening paragraph of the section, set in the body face.",
            40.,
            730.,
        ),
        run("Installation steps", 40., 700., 110., 12., "Head-Bold"),
        body(
            "Run the installer and follow the prompts it shows.",
            40.,
            676.,
        ),
        // A bold sentence, a long lead-in and a bold run inside a body line
        // stay text.
        run(
            "This whole sentence is bold for emphasis.",
            40.,
            646.,
            240.,
            12.,
            "Head-Bold",
        ),
        body("Then more body text follows here as usual.", 40., 622.),
        run(
            "Before you start, check the following items:",
            40.,
            592.,
            260.,
            12.,
            "Head-Bold",
        ),
        body("A list or paragraph follows the lead-in.", 40., 568.),
    ]);
    assert!(markdown.starts_with("# **Section title**"), "{markdown}");
    assert!(
        markdown.contains("\n\n## **Installation steps**\n\n"),
        "{markdown}"
    );
    for text in ["This whole sentence", "Before you start"] {
        let line = markdown.lines().find(|l| l.contains(text)).unwrap();
        assert!(!line.starts_with('#'), "{markdown}");
    }
}

#[test]
fn a_line_ending_short_of_its_column_ends_its_paragraph() {
    let markdown = flowed(vec![
        body(
            "This paragraph wraps across the full measure of the column",
            40.,
            700.,
        ),
        body("and ends here.", 40., 686.),
        // No extra spacing before the next paragraph, or between rows of a
        // byline block.
        body(
            "Another paragraph begins on the very next line of the page",
            40.,
            672.,
        ),
        body("without any spacing at all.", 40., 658.),
        body("Written by", 40., 644.),
        body("Jane Smith", 40., 630.),
        // A lower-case line goes on with the sentence, and a footnote's
        // number heads the line after it.
        body("Short line", 40., 600.),
        body("continues the sentence.", 40., 586.),
        body("1", 40., 556.),
        body("The footnote text.", 40., 542.),
    ]);
    assert!(
        markdown.contains("and ends here.\n\nAnother paragraph"),
        "{markdown}"
    );
    assert!(
        markdown.contains("at all.\n\nWritten by\n\nJane Smith"),
        "{markdown}"
    );
    assert!(
        markdown.contains("Short line continues the sentence."),
        "{markdown}"
    );
    assert!(markdown.contains("1 The footnote text."), "{markdown}");
}

#[test]
fn text_beside_a_floated_figure_keeps_its_paragraph() {
    // Lines of running text wrapped at a narrower measure than the page's.
    let mut items = vec![body(
        "A full width line of running text sets the right edge of the column",
        40.,
        720.,
    )];
    for (row, text) in [
        "Beside the figure each line of text",
        "Wraps at the figure's edge instead",
        "Of the column, which is not a break",
    ]
    .iter()
    .enumerate()
    {
        items.push(body(text, 40., 690. - row as f32 * 14.));
    }
    let markdown = flowed(items);
    assert!(
        markdown.contains("text Wraps at the figure's edge instead Of the column"),
        "{markdown}"
    );
}

#[test]
fn lines_starting_with_the_same_symbol_are_list_items_keeping_it() {
    let markdown = flowed(vec![
        body("A good purifier should ideally have:", 40., 700.),
        body("\u{2705} True HEPA filter", 40., 676.),
        body("\u{2705} Quiet sleep mode", 40., 656.),
        body("\u{2192} a single arrow line stays text", 40., 620.),
    ]);
    assert!(
        markdown.contains("- \u{2705} True HEPA filter\n- \u{2705} Quiet sleep mode"),
        "{markdown}"
    );
    assert!(!markdown.contains("- \u{2192}"), "{markdown}");
}

#[test]
fn a_raised_back_reference_joins_the_line_it_is_raised_from() {
    let markdown = flowed(vec![
        body(
            "3. ^ Author, C. (2015). Signal processing applications. Applied",
            49.,
            270.,
        ),
        body("50-65.", 64., 254.),
        body("4. ^", 49., 235.5),
        run("a b", 74.5, 240.25, 16.4, 10.3, "Small"),
        body("Smith, D.; Jones, E. (2020). Frameworks.", 95., 235.25),
    ]);
    assert!(markdown.contains("4. ^ <sup>a b</sup> Smith"), "{markdown}");
    assert!(!markdown.contains("50-65. a b"), "{markdown}");
}

#[test]
fn angle_brackets_and_leading_quote_marks_read_as_written() {
    let markdown = flowed(vec![
        body(
            "Compare a < b and c > d, x -> y and Vec<String> or <div> tags.",
            40.,
            700.,
        ),
        body("> not a quotation", 40., 660.),
    ]);
    assert!(
        markdown.contains("Compare a < b and c > d, x -> y and Vec\\<String> or \\<div> tags."),
        "{markdown}"
    );
    assert!(markdown.contains("\n\n\\> not a quotation"), "{markdown}");
    assert!(
        !markdown.contains("&lt;") && !markdown.contains("&gt;"),
        "{markdown}"
    );
}

/// A one-page PDF of Helvetica runs (`x`, `y`, text) with link annotations
/// (`[x0, y0, x1, y1]`, URI).
fn linked_pdf(runs: &[(i64, i64, &str)], links: &[([i64; 4], &str)]) -> Vec<u8> {
    let mut pdf = lopdf::Document::with_version("1.7");
    let pages_id = pdf.new_object_id();
    let font =
        pdf.add_object(dictionary! {"Type"=>"Font","Subtype"=>"Type1","BaseFont"=>"Helvetica"});
    let resources = pdf.add_object(dictionary! {"Font"=>dictionary!{"F1"=>font}});
    let mut operations = Vec::new();
    for &(x, y, text) in runs {
        operations.extend([
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 12.into()]),
            Operation::new("Td", vec![x.into(), y.into()]),
            Operation::new("Tj", vec![Object::string_literal(text)]),
            Operation::new("ET", vec![]),
        ]);
    }
    let content = Content { operations }.encode().unwrap();
    let stream = pdf.add_object(Stream::new(Dictionary::new(), content));
    let annots: Vec<Object> = links
        .iter()
        .map(|(rect, uri)| {
            // Quartz writes the action and its URI as indirect objects.
            let uri = pdf.add_object(Object::string_literal(*uri));
            let action = pdf.add_object(dictionary! {"S"=>"URI","URI"=>uri});
            pdf.add_object(dictionary! {
                "Type"=>"Annot",
                "Subtype"=>"Link",
                "Rect"=>rect.iter().map(|&n| Object::from(n)).collect::<Vec<_>>(),
                "A"=>action,
            })
            .into()
        })
        .collect();
    let page = pdf.add_object(dictionary! {"Type"=>"Page","Parent"=>pages_id,"Contents"=>stream,"Resources"=>resources,"Annots"=>annots});
    pdf.objects.insert(
        pages_id,
        dictionary! {"Type"=>"Pages","Count"=>1,"Kids"=>vec![page.into()],"MediaBox"=>vec![0.into(),0.into(),600.into(),800.into()]}.into(),
    );
    let catalog = pdf.add_object(dictionary! {"Type"=>"Catalog","Pages"=>pages_id});
    pdf.trailer.set("Root", catalog);
    let mut bytes = Vec::new();
    pdf.save_to(&mut bytes).unwrap();
    bytes
}

#[test]
fn link_annotations_over_runs_become_markdown_links() {
    // "manual" is a run of its own (Helvetica 12pt: 39.3pt wide); "this
    // report" sits inside a run of a sentence; the script link carries no
    // target into Markdown.
    let bytes = linked_pdf(
        &[
            (40, 700, "Read the"),
            (100, 700, "manual"),
            (145, 700, "for details."),
            (40, 660, "See this report for supporting analysis."),
            (40, 620, "Do not"),
            (85, 620, "click"),
            (120, 620, "this, or write to us."),
        ],
        &[
            ([99, 697, 140, 711], "https://example.com/manual"),
            // "See " is 24.7pt and "this report" 53.4pt wide.
            ([64, 657, 119, 671], "https://example.com/report"),
            ([84, 617, 112, 631], "javascript:alert(1)"),
        ],
    );
    let markdown = super::super::extract(&bytes).unwrap().markdown;
    assert!(
        markdown.contains("Read the [manual](https://example.com/manual) for details."),
        "{markdown}"
    );
    assert!(
        markdown.contains("See [this report](https://example.com/report) for supporting analysis."),
        "{markdown}"
    );
    assert!(
        markdown.contains("Do not click this, or write to us."),
        "{markdown}"
    );
    assert!(!markdown.contains("javascript"), "{markdown}");
}
