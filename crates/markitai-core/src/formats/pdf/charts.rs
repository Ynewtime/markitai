//! Conservative preservation of locally clipped vertical bar charts.
//!
//! This does not infer data values. A matching chart is rasterized by the
//! caller, including its labels and legend, instead of flattening those labels
//! into prose. Uncertain drawings remain on the existing extraction path.
use super::{Bar, Chart, Edge, Frame, TOLERANCE};

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() <= TOLERANCE
}

pub(super) fn rectangle(
    points: &[[f32; 2]],
    segments: &[([f32; 2], [f32; 2])],
    [x0, y0, x1, y1]: [f32; 4],
) -> bool {
    let width = x1 - x0;
    let height = y1 - y0;
    if !(2.0..=80.0).contains(&width) || height < 4.0 || points.len() > 8 {
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
        let candidate = Chart {
            x0: clip[0],
            y0: clip[1],
            x1: clip[2],
            y1: clip[3],
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

#[cfg(test)]
#[path = "charts_tests.rs"]
mod tests;
