//! Conservative preservation of vector bar and line charts.
//!
//! This does not infer data values. A matching chart is rasterized by the
//! caller, including its labels and legend, instead of flattening those labels
//! into prose. Uncertain drawings remain on the existing extraction path.
//! [`detect`] finds locally clipped, gridded vertical bar charts by their
//! clip; [`plots`] finds a plot by its axes and series wherever it is drawn
//! (LibreOffice and hand-made charts set no clip of their own).
use super::{Bar, Chart, Edge, Frame, TOLERANCE, Trace};

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() <= TOLERANCE
}

/// A bar's rectangle: [`closed_rectangle`], 2 to 80 points wide and at
/// least 4 tall.
pub(super) fn rectangle(
    points: &[[f32; 2]],
    segments: &[([f32; 2], [f32; 2])],
    bounds: [f32; 4],
) -> bool {
    let [x0, y0, x1, y1] = bounds;
    (2.0..=80.0).contains(&(x1 - x0))
        && y1 - y0 >= 4.0
        && closed_rectangle(points, segments, bounds)
}

/// Whether a path of at most eight points is the closed axis-aligned
/// rectangle `bounds`.
pub(super) fn closed_rectangle(
    points: &[[f32; 2]],
    segments: &[([f32; 2], [f32; 2])],
    [x0, y0, x1, y1]: [f32; 4],
) -> bool {
    let width = x1 - x0;
    let height = y1 - y0;
    if points.len() > 8 {
        return false;
    }
    // Require all four corners and reject diagonal, inset and open paths.
    // Filled paths close implicitly, so the last-to-first segment counts too.
    let close = points.last().copied().zip(points.first().copied());
    let edges = segments.iter().copied().chain(close);
    let mut perimeter = 0.;
    for (a, b) in edges {
        let horizontal =
            (a[1] - b[1]).abs() < 0.1 && ((a[1] - y0).abs() < 0.1 || (a[1] - y1).abs() < 0.1);
        let vertical =
            (a[0] - b[0]).abs() < 0.1 && ((a[0] - x0).abs() < 0.1 || (a[0] - x1).abs() < 0.1);
        if !horizontal && !vertical {
            return false;
        }
        perimeter += (a[0] - b[0]).abs() + (a[1] - b[1]).abs();
    }
    // `h` and `re` already supply closure; the implicit edge must not count it
    // twice when the points vector does not repeat its starting point.
    let explicit_closed = segments
        .last()
        .is_some_and(|(_, b)| points.first().is_some_and(|a| a == b));
    if explicit_closed && let Some((a, b)) = close {
        perimeter -= (a[0] - b[0]).abs() + (a[1] - b[1]).abs();
    }
    (perimeter - 2. * (width + height)).abs() < 0.5
        && [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            .iter()
            .all(|corner| {
                points
                    .iter()
                    .any(|p| (p[0] - corner[0]).abs() < 0.1 && (p[1] - corner[1]).abs() < 0.1)
            })
}

pub(super) fn detect(edges: &[Edge], bars: &[Bar], frame: Frame) -> Vec<Chart> {
    let mut groups: Vec<Vec<&Bar>> = Vec::new();
    for bar in bars {
        let [x0, y0, x1, y1] = bar.clip;
        let area = (x1 - x0) * (y1 - y0);
        // A whole-page clip says nothing about where a chart ends. Requiring a
        // small local clip avoids guessing how much nearby prose to swallow.
        if area <= 0. || area > frame.width * frame.height * 0.5 || x1 - x0 < 60. || y1 - y0 < 40. {
            continue;
        }
        if let Some(group) = groups.iter_mut().find(|group| {
            group[0].clip.iter().zip(bar.clip).all(|(a, b)| near(*a, b))
                && near(group[0].bounds[1], bar.bounds[1])
                && near(
                    group[0].bounds[2] - group[0].bounds[0],
                    bar.bounds[2] - bar.bounds[0],
                )
        }) {
            group.push(bar);
        } else {
            if groups.len() >= 128 {
                return Vec::new();
            }
            groups.push(vec![bar]);
        }
    }
    let mut result: Vec<Chart> = Vec::new();
    for mut group in groups {
        if group.len() < 4 {
            continue;
        }
        group.sort_by(|a, b| a.bounds[0].total_cmp(&b.bounds[0]));
        if group
            .windows(2)
            .any(|pair| pair[0].bounds[2] > pair[1].bounds[0] + 0.2)
        {
            continue;
        }
        let first = group[0];
        let left = first.bounds[0];
        let right = group.last().unwrap().bounds[2];
        let bottom = first.bounds[1];
        let top = group.iter().map(|b| b.bounds[3]).fold(bottom, f32::max);
        let lowest = group.iter().map(|b| b.bounds[3]).fold(top, f32::min);
        if right - left < 50. || top - bottom < 30. || top - lowest < 8. {
            continue;
        }
        // A shared baseline, a tall axis and at least three equally spanning
        // horizontal rules distinguish the plot from colored table cells or
        // a row of decorative rectangles. Exclude shapes outside the clip.
        let clip = first.clip;
        let rules: Vec<_> = edges
            .iter()
            .filter(|e| {
                e.horizontal
                    && e.start >= clip[0] - TOLERANCE
                    && e.end <= clip[2] + TOLERANCE
                    && e.start <= left
                    && e.end >= right
                    && e.position >= bottom - TOLERANCE
                    && e.position <= clip[3]
            })
            .collect();
        let Some(base) = rules.iter().find(|e| near(e.position, bottom)) else {
            continue;
        };
        let mut levels: Vec<f32> = rules
            .iter()
            .filter(|e| near(e.start, base.start) && near(e.end, base.end))
            .map(|e| e.position)
            .collect();
        levels.sort_by(f32::total_cmp);
        levels.dedup_by(|a, b| near(*a, *b));
        let Some(&axis_top) = levels.last() else {
            continue;
        };
        if levels.len() < 3
            || axis_top < top - TOLERANCE
            || !edges.iter().any(|e| {
                !e.horizontal
                    // Tick marks can extend the merged horizontal rules a
                    // little left of their actual vertical axis.
                    && e.position >= base.start - TOLERANCE
                    && e.position <= (base.start + 8.).min(left)
                    && e.start <= bottom + TOLERANCE
                    && e.end >= axis_top - TOLERANCE
            })
        {
            continue;
        }
        // The clip is the chart's box only when it is not much larger than
        // the plot (a spreadsheet's print area clips the whole sheet).
        let plot = (right - base.start.min(left)) * (axis_top - bottom);
        if (clip[2] - clip[0]) * (clip[3] - clip[1]) > plot * FRAME_AREA {
            continue;
        }
        let candidate = Chart {
            x0: clip[0],
            y0: clip[1],
            x1: clip[2],
            y1: clip[3],
            labels: false,
        };
        if result.iter().any(|r| {
            candidate.x0 < r.x1 && candidate.x1 > r.x0 && candidate.y0 < r.y1 && candidate.y1 > r.y0
        }) {
            continue;
        }
        result.push(candidate);
        if result.len() >= 16 {
            break;
        }
    }
    result
}

/// The plot's frame extends at least this far left of and below its axes
/// (room for tick numbers and category names), in points.
const FRAME_MARGIN: f32 = 10.;
/// A frame holds at most this many times the plot's area.
const FRAME_AREA: f32 = 6.;

/// Charts found by their plot: three or more evenly spaced rules of one span
/// at least 60 points long (gridlines, the lowest the baseline), or a lone
/// rule of that length with a vertical axis rising from its left end (a
/// frame has a top edge of the same span); at least 30 points
/// tall; and a series drawn on it: two or more bars of different heights
/// standing on the baseline, side by side, or a stroked line (a curve or
/// slanted segments) across at least 30% of the plot's width, and no filled
/// curve on it. Its rectangle is the
/// smallest closed rectangle (a frame, a background) around the plot with
/// room for labels left of and below it, else the plot alone, its labels to
/// be found from the text (`Chart::labels`); in either case within the
/// clip the series was drawn in. Charts overlapping `found` are skipped.
pub(super) fn plots(
    edges: &[Edge],
    bars: &[Bar],
    traces: &[Trace],
    boxes: &[[f32; 4]],
    frame: Frame,
    found: &mut Vec<Chart>,
) {
    // Horizontal rules of one span, bottom first.
    let mut spans: Vec<Vec<&Edge>> = Vec::new();
    for edge in edges
        .iter()
        .filter(|e| e.horizontal && e.end - e.start >= 60.)
    {
        if let Some(span) = spans
            .iter_mut()
            .find(|span| near(span[0].start, edge.start) && near(span[0].end, edge.end))
        {
            span.push(edge);
        } else if spans.len() < 256 {
            spans.push(vec![edge]);
        } else {
            return;
        }
    }
    for mut span in spans {
        span.sort_by(|a, b| a.position.total_cmp(&b.position));
        let (left, right, base) = (span[0].start, span[0].end, span[0].position);
        let axis = edges
            .iter()
            .filter(|e| {
                !e.horizontal
                    && near(e.position, left)
                    && near(e.start, base)
                    && e.end - base >= 30.
            })
            .map(|e| e.end)
            .reduce(f32::max);
        let steps: Vec<f32> = span
            .windows(2)
            .map(|pair| pair[1].position - pair[0].position)
            .collect();
        let gridded = steps.len() >= 2
            && steps.iter().all(|step| *step >= 4.)
            && steps.iter().copied().fold(0., f32::max)
                <= steps.iter().copied().fold(f32::INFINITY, f32::min) * 1.1;
        // Without gridlines, the baseline is the span's only rule: a frame
        // has a top edge too.
        let top = match axis {
            _ if gridded => axis.unwrap_or(base).max(span[span.len() - 1].position),
            Some(top) if span.len() == 1 => top,
            _ => continue,
        };
        if top - base < 30. {
            continue;
        }
        let inside = |b: [f32; 4]| {
            b[0] >= left - TOLERANCE
                && b[2] <= right + TOLERANCE
                && b[1] >= base - TOLERANCE
                && b[3] <= top + TOLERANCE
        };
        // A filled curve on the plot is a drawing this does not read.
        if traces.iter().any(|t| {
            t.filled
                && t.bounds[0] < right
                && t.bounds[2] > left
                && t.bounds[1] < top
                && t.bounds[3] > base
        }) {
            continue;
        }
        let mut standing: Vec<&Bar> = bars
            .iter()
            .filter(|bar| near(bar.bounds[1], base) && inside(bar.bounds))
            .collect();
        standing.sort_by(|a, b| a.bounds[0].total_cmp(&b.bounds[0]));
        if standing
            .windows(2)
            .any(|pair| pair[0].bounds[2] > pair[1].bounds[0] + 0.2)
        {
            continue;
        }
        let heights = standing
            .iter()
            .map(|bar| bar.bounds[3])
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), y| {
                (low.min(y), high.max(y))
            });
        let series = traces
            .iter()
            .find(|trace| {
                !trace.filled
                    && inside(trace.bounds)
                    && trace.bounds[2] - trace.bounds[0] >= 0.3 * (right - left)
            })
            .map(|trace| trace.clip);
        let clip = if standing.len() >= 2 && heights.1 - heights.0 >= 4. {
            standing[0].clip
        } else if let Some(clip) = series {
            clip
        } else {
            continue;
        };
        let area = (right - left) * (top - base);
        let framed = boxes
            .iter()
            .filter(|b| {
                b[0] <= left - FRAME_MARGIN
                    && b[1] <= base - FRAME_MARGIN
                    && b[2] >= right - TOLERANCE
                    && b[3] >= top - TOLERANCE
                    && (b[2] - b[0]) * (b[3] - b[1]) <= area * FRAME_AREA
            })
            .min_by(|a, b| {
                ((a[2] - a[0]) * (a[3] - a[1])).total_cmp(&((b[2] - b[0]) * (b[3] - b[1])))
            });
        let bounds = framed.copied().unwrap_or([left, base, right, top]);
        let candidate = Chart {
            x0: bounds[0].max(clip[0]).max(0.),
            y0: bounds[1].max(clip[1]).max(0.),
            x1: bounds[2].min(clip[2]).min(frame.width),
            y1: bounds[3].min(clip[3]).min(frame.height),
            labels: framed.is_none(),
        };
        if candidate.x1 - candidate.x0 < 50.
            || candidate.y1 - candidate.y0 < 30.
            || found.iter().any(|r| {
                candidate.x0 < r.x1
                    && candidate.x1 > r.x0
                    && candidate.y0 < r.y1
                    && candidate.y1 > r.y0
            })
        {
            continue;
        }
        found.push(candidate);
        if found.len() >= 16 {
            return;
        }
    }
}

#[cfg(test)]
#[path = "charts_tests.rs"]
mod tests;
