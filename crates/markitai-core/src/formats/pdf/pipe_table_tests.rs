//! Rows of text divided by `|` are a Markdown table, and a line that runs
//! past the page's right edge no longer sends its page to the plain reader.
use super::*;

/// A one-page PDF (Helvetica, 595 x 842) with each `(x, y, size, text)`.
fn pdf(lines: &[(i32, i32, i32, &str)]) -> Vec<u8> {
    let mut stream = String::from("BT\n");
    for (x, y, size, text) in lines {
        stream.push_str(&format!("/F1 {size} Tf 1 0 0 1 {x} {y} Tm ({text}) Tj\n"));
    }
    stream.push_str("ET\n");
    let objects = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_owned(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".to_owned(),
        format!("<< /Length {} >>\nstream\n{stream}endstream", stream.len()),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_owned(),
    ];
    let mut bytes = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (index, object) in objects.iter().enumerate() {
        offsets.push(bytes.len());
        bytes.extend(format!("{} 0 obj\n{object}\nendobj\n", index + 1).bytes());
    }
    let xref = bytes.len();
    bytes.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).bytes());
    for offset in offsets {
        bytes.extend(format!("{offset:010} 00000 n \n").bytes());
    }
    bytes.extend(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        )
        .bytes(),
    );
    bytes
}

fn page() -> Vec<(i32, i32, i32, &'static str)> {
    vec![
        (72, 770, 18, "Section Two Data"),
        (72, 734, 12, "A sentence before the table, long enough to read as text."),
        // Longer than the page is wide from x = 72 (the width is an estimate).
        (72, 710, 12, "This line is deliberately far too long to fit between its left margin and the right edge of the page, and a producer need not wrap it."),
        (72, 686, 12, "Item | Qty | Price"),
        (72, 662, 12, "Apple | 12 | 3.50"),
        (72, 638, 12, "Banana | 30 | 1.20"),
        (72, 614, 12, "Closing remark after the table."),
    ]
}

#[test]
fn pipe_rows_are_a_table_and_their_neighbours_stay_paragraphs() {
    let markdown = extract(&pdf(&page())).unwrap().markdown;
    assert!(
        markdown.contains(
            "| Item | Qty | Price |\n| --- | --- | --- |\n| Apple | 12 | 3.50 |\n| Banana | 30 | 1.20 |"
        ),
        "{markdown}"
    );
    // Not run together: the rows' neighbours are paragraphs of their own.
    assert!(!markdown.contains("1.20 Closing"), "{markdown}");
    assert!(!markdown.contains("3.50 Banana"), "{markdown}");
    assert!(
        markdown.contains("\n\nClosing remark after the table."),
        "{markdown}"
    );
    assert!(
        markdown.contains("A sentence before the table, long enough to read as text.\n\n"),
        "{markdown}"
    );
}

#[test]
fn one_line_with_pipes_is_a_paragraph_of_its_own() {
    let lines = [
        (72, 770, 12, "Before the line."),
        (72, 746, 12, "Left | Right"),
        (72, 722, 12, "After the line."),
    ];
    let markdown = extract(&pdf(&lines)).unwrap().markdown;
    assert!(!markdown.contains("| ---"), "{markdown}");
    assert!(markdown.contains("Left | Right"), "{markdown}");
}
