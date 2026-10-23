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
    // Under a page-wide clip the chart is the plot, its labels to be found
    // from the page's text.
    let found = detect_stream(&chart().replace("90 80 340 180 re", "0 0 600 800 re"));
    assert_eq!(found.len(), 1);
    let r = found[0];
    assert_eq!([r.x0, r.y0, r.x1, r.y1], [100., 100., 350., 250.]);
    assert!(r.labels);
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

/// A LibreOffice chart: a white background and a grey frame drawn as closed
/// paths, evenly spaced gridlines without a vertical axis, two series of
/// bars side by side, all under the sheet's print-area clip.
fn office_chart(series: &str) -> String {
    let mut stream = String::from("q 0 0 600 800 re W* n\n");
    let frame = "250 300 m 40 300 l 40 520 l 460 520 l 460 300 l 250 300 l h";
    stream.push_str(&format!("1 1 1 rg {frame} f*\n0.85 G {frame} S\n0.53 G\n"));
    for y in [340, 380, 420, 460] {
        stream.push_str(&format!("400 {y} m 90 {y} l S\n"));
    }
    stream.push_str(series);
    stream.push_str("Q\n");
    stream
}

#[test]
fn a_framed_plot_takes_its_frame_and_a_line_series_counts() {
    let mut bars = String::from(
        "0.3 0.5 0.75 rg
",
    );
    for (x, top) in [(100, 380), (170, 420), (240, 450)] {
        bars.push_str(&format!(
            "{x} 340 m {} 340 l {} {top} l {x} {top} l {x} 340 l h f*
",
            x + 18,
            x + 18
        ));
    }
    let found = detect_stream(&office_chart(&bars));
    assert_eq!(found.len(), 1, "{found:?}");
    let r = found[0];
    assert_eq!([r.x0, r.y0, r.x1, r.y1,], [40., 300., 460., 520.]);
    assert!(!r.labels);
    let line = "0.29 0.49 0.73 RG 100 450 m 180 420 230 360 300 350 c 340 360 l S\n";
    let found = detect_stream(&office_chart(line));
    assert_eq!(found.len(), 1, "{found:?}");
    // A filled curve on the plot, or a series too short, is no chart.
    let area = "0.29 0.49 0.73 rg 100 340 m 180 420 230 360 300 340 c h f\n";
    assert!(detect_stream(&office_chart(area)).is_empty());
    let short = "0.29 0.49 0.73 RG 100 450 m 120 420 l S\n";
    assert!(detect_stream(&office_chart(short)).is_empty());
}

#[test]
fn frames_and_axes_alone_are_no_plot() {
    // Two framed boxes stacked, a curve in one: their edges share a span
    // and a left side, but a frame has a top edge.
    let mut stream = String::new();
    for bottom in [100, 400] {
        let top = bottom + 200;
        stream.push_str(&format!(
            "0 G 40 {bottom} m 460 {bottom} l 460 {top} l 40 {top} l h S\n"
        ));
    }
    stream.push_str("0 0 1 RG 100 450 m 180 420 230 360 300 350 c S\n");
    assert!(detect_stream(&stream).is_empty());
    // Axes alone, with bars on them: a plot, its labels from the text.
    let mut axes =
        String::from("0 G 100 542 m 100 342 l S 100 342 m 450 342 l S\n0.2 0.4 0.8 rg\n");
    for (x, height) in [(120, 40), (185, 80), (250, 150)] {
        axes.push_str(&format!("{x} 342 40 {height} re h f\n"));
    }
    let found = detect_stream(&axes);
    assert_eq!(found.len(), 1, "{found:?}");
    let r = found[0];
    assert_eq!([r.x0, r.y0, r.x1, r.y1], [100., 342., 450., 542.]);
    assert!(r.labels);
}
