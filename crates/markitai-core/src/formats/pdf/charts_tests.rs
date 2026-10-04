use super::super::{RuleResources, page_charts};
use super::*;
use lopdf::content::Content;

fn frame() -> Frame {
    Frame {
        x: 0.,
        y: 0.,
        width: 600.,
        height: 800.,
    }
}

fn chart() -> String {
    let mut stream = String::from("q 90 80 340 180 re W n 0 G 1 w\n");
    for y in [100, 150, 200, 250] {
        stream.push_str(&format!("100 {y} m 350 {y} l S\n"));
    }
    stream.push_str("100 100 m 100 250 l S\n0.2 0.4 0.7 rg\n");
    for (x, top) in [(110, 160), (170, 200), (230, 140), (290, 230)] {
        // Office exports rectangles as closed polygons rather than `re`.
        stream.push_str(&format!(
            "{x} 100 m {} 100 l {} {top} l {x} {top} l {x} 100 l h f\n",
            x + 20,
            x + 20
        ));
    }
    stream.push_str("Q\n");
    stream
}

fn detect_stream(stream: &str) -> Vec<Chart> {
    page_charts(
        &Content::decode(stream.as_bytes()).unwrap(),
        frame(),
        &RuleResources::default(),
    )
}

#[test]
fn clipped_bar_plot_keeps_whole_label_and_legend_region() {
    let found = detect_stream(&chart());
    assert_eq!(found.len(), 1);
    let r = found[0];
    assert_eq!([r.x0, r.y0, r.x1, r.y1], [90., 80., 430., 260.]);
}

#[test]
fn rectangles_and_implicitly_closed_paths_both_work() {
    let points = [[10., 20.], [30., 20.], [30., 80.], [10., 80.]];
    let open: Vec<_> = points.windows(2).map(|p| (p[0], p[1])).collect();
    assert!(rectangle(&points, &open, [10., 20., 30., 80.]));
    let mut closed = open;
    closed.push((points[3], points[0]));
    assert!(rectangle(&points, &closed, [10., 20., 30., 80.]));
    let triangle = [[10., 20.], [30., 20.], [30., 80.]];
    let segments: Vec<_> = triangle.windows(2).map(|p| (p[0], p[1])).collect();
    assert!(!rectangle(&triangle, &segments, [10., 20., 30., 80.]));
}

#[test]
fn page_clip_decoration_and_ungridded_bars_are_not_charts() {
    assert!(detect_stream(&chart().replace("90 80 340 180 re", "0 0 600 800 re")).is_empty());
    let no_rules = chart()
        .lines()
        .filter(|s| !s.ends_with(" l S"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(detect_stream(&no_rules).is_empty());
    let equal = chart()
        .replace("160 l", "180 l")
        .replace("200 l", "180 l")
        .replace("140 l", "180 l")
        .replace("230 l", "180 l");
    assert!(detect_stream(&equal).is_empty());
}

#[test]
fn unknown_visibility_curves_and_overlapping_bars_abstain() {
    assert!(detect_stream(&chart().replace("0 G 1 w", "/Missing gs 0 G 1 w")).is_empty());
    let curved = chart().replace("110 100 m 130 100 l", "110 100 m 115 95 125 95 130 100 c");
    assert!(detect_stream(&curved).is_empty());
    let overlap = chart().replace(
        "170 100 m 190 100 l 190 200 l 170 200 l 170 100 l",
        "120 100 m 140 100 l 140 200 l 120 200 l 120 100 l",
    );
    assert!(detect_stream(&overlap).is_empty());
}

#[test]
fn translations_are_applied_to_chart_bounds() {
    let stream = format!("q 1 0 0 1 20 30 cm\n{}Q", chart());
    let found = detect_stream(&stream);
    assert_eq!(found.len(), 1);
    let r = found[0];
    assert_eq!([r.x0, r.y0, r.x1, r.y1], [110., 110., 450., 290.]);
}

#[test]
fn tick_marks_do_not_move_the_vertical_axis() {
    let mut stream = chart();
    // Appended outside q/Q so use the same page coordinate frame.
    for y in [100, 150, 200, 250] {
        stream.push_str(&format!("96 {y} m 100 {y} l S\n"));
    }
    assert_eq!(detect_stream(&stream).len(), 1);
}
