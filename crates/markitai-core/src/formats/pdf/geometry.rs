//! Conservative geometry for complete, axis-aligned ruled tables.
use lopdf::{Object, ObjectId, content::Content};
use std::collections::HashSet;

const TOLERANCE: f32 = 1.5;
const MAX_EDGES: usize = 8192;

#[derive(Clone, Copy, Debug)]
pub(super) struct Frame {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

fn inherited<'a>(pdf: &'a lopdf::Document, mut id: ObjectId, name: &[u8]) -> Option<&'a Object> {
    for _ in 0..64 {
        let dict = pdf.get_dictionary(id).ok()?;
        if let Ok(value) = dict.get(name) {
            return pdf.dereference(value).ok().map(|(_, value)| value);
        }
        id = dict.get(b"Parent").and_then(Object::as_reference).ok()?;
    }
    None
}

fn rectangle(value: &Object) -> Option<[f32; 4]> {
    let values = value.as_array().ok()?;
    if values.len() != 4 {
        return None;
    }
    let mut result = [0.0; 4];
    for (to, from) in result.iter_mut().zip(values) {
        *to = from.as_float().ok()?;
    }
    (result
        .iter()
        .all(|v| v.is_finite() && v.abs() < 1_000_000.0)
        && result[0] < result[2]
        && result[1] < result[3])
        .then_some(result)
}

pub(super) fn frame(pdf: &lopdf::Document, id: ObjectId) -> Option<Frame> {
    // Positioned items may be rotated into a synthetic frame. Pages whose
    // displayed orientation differs are kept on the original reader's path.
    if inherited(pdf, id, b"Rotate")
        .is_some_and(|v| v.as_i64().map_or(true, |n| n.rem_euclid(360) != 0))
    {
        return None;
    }
    let mut bounds = rectangle(inherited(pdf, id, b"MediaBox")?)?;
    if let Some(crop) = inherited(pdf, id, b"CropBox").and_then(rectangle) {
        let intersection = [
            bounds[0].max(crop[0]),
            bounds[1].max(crop[1]),
            bounds[2].min(crop[2]),
            bounds[3].min(crop[3]),
        ];
        if intersection[0] < intersection[2] && intersection[1] < intersection[3] {
            bounds = intersection;
        }
    }
    Some(Frame {
        x: bounds[0],
        y: bounds[1],
        width: bounds[2] - bounds[0],
        height: bounds[3] - bounds[1],
    })
}

#[derive(Clone, Copy)]
struct State {
    matrix: [f32; 6],
    clip: [f32; 4],
    white_fill: bool,
    white_stroke: bool,
}

impl State {
    fn point(self, x: f32, y: f32) -> Option<[f32; 2]> {
        let [a, b, c, d, e, f] = self.matrix;
        let p = [a * x + c * y + e, b * x + d * y + f];
        p.iter()
            .all(|v| v.is_finite() && v.abs() < 1_000_000.0)
            .then_some(p)
    }
    fn inside(self, p: [f32; 2]) -> bool {
        p[0] >= self.clip[0] - TOLERANCE
            && p[0] <= self.clip[2] + TOLERANCE
            && p[1] >= self.clip[1] - TOLERANCE
            && p[1] <= self.clip[3] + TOLERANCE
    }
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    horizontal: bool,
    position: f32,
    start: f32,
    end: f32,
}

fn edge(a: [f32; 2], b: [f32; 2]) -> Option<Edge> {
    if (a[1] - b[1]).abs() < 0.3 && (a[0] - b[0]).abs() > 3.0 {
        Some(Edge {
            horizontal: true,
            position: (a[1] + b[1]) / 2.0,
            start: a[0].min(b[0]),
            end: a[0].max(b[0]),
        })
    } else if (a[0] - b[0]).abs() < 0.3 && (a[1] - b[1]).abs() > 3.0 {
        Some(Edge {
            horizontal: false,
            position: (a[0] + b[0]) / 2.0,
            start: a[1].min(b[1]),
            end: a[1].max(b[1]),
        })
    } else {
        None
    }
}

/// Names of the page's graphics states that change nothing a rule verdict
/// depends on: full opacity, normal blending, no soft mask. Line and rendering
/// parameters are allowed; everything else keeps the page unsafe.
pub(super) fn neutral_states(pdf: &lopdf::Document, id: ObjectId) -> HashSet<Vec<u8>> {
    let Ok((direct, ids)) = pdf.get_page_resources(id) else {
        return HashSet::new();
    };
    let resolve = |value: &'_ Object| pdf.dereference(value).ok().map(|(_, value)| value.clone());
    let mut neutral = HashSet::new();
    let mut seen = HashSet::new();
    for resources in direct
        .into_iter()
        .chain(ids.iter().filter_map(|id| pdf.get_dictionary(*id).ok()))
    {
        let Some(states) = resources
            .get(b"ExtGState")
            .ok()
            .and_then(&resolve)
            .and_then(|value| value.as_dict().ok().cloned())
        else {
            continue;
        };
        for (name, value) in states.iter() {
            // The nearest dictionary defining a name decides it, even when that
            // definition is malformed; an inherited neutral state cannot mask it.
            if !seen.insert(name.clone()) {
                continue;
            }
            let Some(Ok(state)) = resolve(value).map(|value| value.as_dict().cloned()) else {
                continue;
            };
            let safe = state.iter().all(|(key, value)| {
                let value = resolve(value);
                match key.as_slice() {
                    b"Type" | b"LW" | b"LC" | b"LJ" | b"ML" | b"D" | b"RI" | b"FL" | b"SM"
                    | b"SA" | b"OP" | b"op" | b"OPM" => true,
                    b"CA" | b"ca" => value
                        .and_then(|value| value.as_float().ok())
                        .is_some_and(|alpha| alpha >= 0.999),
                    b"BM" => value.is_some_and(|value| {
                        value
                            .as_name()
                            .is_ok_and(|name| matches!(name, b"Normal" | b"Compatible"))
                    }),
                    b"SMask" => {
                        value.is_some_and(|value| value.as_name().is_ok_and(|name| name == b"None"))
                    }
                    b"AIS" => value.is_some_and(|value| value.as_bool().is_ok_and(|flag| !flag)),
                    _ => false,
                }
            });
            if safe {
                neutral.insert(name.clone());
            }
        }
    }
    neutral
}

fn edges(content: &Content, frame: Frame, neutral: &HashSet<Vec<u8>>) -> Option<Vec<Edge>> {
    if content.operations.len() > 200_000 {
        return None;
    }
    let mut state = State {
        matrix: [1., 0., 0., 1., -frame.x, -frame.y],
        clip: [0., 0., frame.width, frame.height],
        white_fill: false,
        white_stroke: false,
    };
    let mut stack = Vec::new();
    let mut points = Vec::<[f32; 2]>::new();
    let mut segments = Vec::new();
    let mut current = None;
    let mut start = None;
    let mut rectangular = false;
    let mut pending_clip = false;
    let mut output = Vec::new();
    for op in &content.operations {
        let numbers = || {
            op.operands
                .iter()
                .map(Object::as_float)
                .collect::<std::result::Result<Vec<_>, _>>()
                .ok()
        };
        match op.operator.as_str() {
            "q" => {
                if stack.len() >= 64 {
                    return None;
                }
                stack.push(state);
            }
            "Q" => {
                state = stack.pop()?;
            }
            "cm" => {
                let n = numbers()?;
                let [a, b, c, d, e, f] = n.as_slice() else {
                    return None;
                };
                let [g, h, i, j, k, l] = state.matrix;
                state.matrix = [
                    g * a + i * b,
                    h * a + j * b,
                    g * c + i * d,
                    h * c + j * d,
                    g * e + i * f + k,
                    h * e + j * f + l,
                ];
            }
            "g" | "G" | "rg" | "RG" | "k" | "K" => {
                let n = numbers()?;
                let white = match n.as_slice() {
                    [v] => *v >= 0.98,
                    [r, g, b] => r.min(*g).min(*b) >= 0.98,
                    [c, m, y, k] => c.max(*m).max(*y).max(*k) <= 0.02,
                    _ => return None,
                };
                if op.operator.chars().all(char::is_uppercase) {
                    state.white_stroke = white;
                } else {
                    state.white_fill = white;
                }
            }
            "m" | "l" => {
                let n = numbers()?;
                let [x, y] = n.as_slice() else {
                    return None;
                };
                let p = state.point(*x, *y)?;
                if op.operator == "m" {
                    start = Some(p);
                } else if let Some(previous) = current {
                    segments.push((previous, p));
                }
                current = Some(p);
                points.push(p);
                rectangular = false;
            }
            "re" => {
                let n = numbers()?;
                let [x, y, w, h] = n.as_slice() else {
                    return None;
                };
                let p = [
                    state.point(*x, *y)?,
                    state.point(x + w, *y)?,
                    state.point(x + w, y + h)?,
                    state.point(*x, y + h)?,
                ];
                rectangular = points.is_empty();
                for i in 0..4 {
                    segments.push((p[i], p[(i + 1) % 4]));
                }
                points.extend(p);
                current = Some(p[0]);
                start = current;
            }
            "h" => {
                if let (Some(a), Some(b)) = (current, start) {
                    segments.push((a, b));
                    current = Some(b);
                }
            }
            "c" | "v" | "y" => {
                // A curve's control points do not describe a table rule.
                // Never infer a closed cell from the straight portions only.
                return None;
            }
            "W" | "W*" => {
                pending_clip = true;
            }
            "S" | "s" | "f" | "F" | "f*" | "B" | "B*" | "b" | "b*" | "n" => {
                let bounds = points.iter().fold(
                    [
                        f32::INFINITY,
                        f32::INFINITY,
                        f32::NEG_INFINITY,
                        f32::NEG_INFINITY,
                    ],
                    |a, p| {
                        [
                            a[0].min(p[0]),
                            a[1].min(p[1]),
                            a[2].max(p[0]),
                            a[3].max(p[1]),
                        ]
                    },
                );
                if pending_clip {
                    if !rectangular {
                        return None;
                    }
                    state.clip = [
                        state.clip[0].max(bounds[0]),
                        state.clip[1].max(bounds[1]),
                        state.clip[2].min(bounds[2]),
                        state.clip[3].min(bounds[3]),
                    ];
                    pending_clip = false;
                }
                let fill = matches!(
                    op.operator.as_str(),
                    "f" | "F" | "f*" | "B" | "B*" | "b" | "b*"
                ) && !state.white_fill;
                let stroke = matches!(op.operator.as_str(), "S" | "s" | "B" | "B*" | "b" | "b*")
                    && !state.white_stroke;
                if stroke {
                    for &(a, b) in &segments {
                        if state.inside(a)
                            && state.inside(b)
                            && let Some(e) = edge(a, b)
                        {
                            output.push(e);
                        }
                    }
                }
                // Office exporters often paint rules as narrow polygons,
                // including bevelled corners, rather than stroke a line.
                if fill && points.iter().all(|&p| state.inside(p)) {
                    let [x0, y0, x1, y1] = bounds;
                    if y1 - y0 <= 1.5 && x1 - x0 >= 3.0 {
                        output.push(Edge {
                            horizontal: true,
                            position: (y0 + y1) / 2.0,
                            start: x0,
                            end: x1,
                        });
                    }
                    if x1 - x0 <= 1.5 && y1 - y0 >= 3.0 {
                        output.push(Edge {
                            horizontal: false,
                            position: (x0 + x1) / 2.0,
                            start: y0,
                            end: y1,
                        });
                    }
                }
                if output.len() > MAX_EDGES {
                    return None;
                }
                points.clear();
                segments.clear();
                current = None;
                start = None;
                rectangular = false;
            }
            "gs" if op
                .operands
                .first()
                .and_then(|name| name.as_name().ok())
                .is_some_and(|name| neutral.contains(name)) => {}
            // Unknown resource colours, transparency, Forms and shading make
            // a geometrical table verdict unsafe. Text can still be recovered.
            "gs" | "cs" | "CS" | "sc" | "SC" | "scn" | "SCN" | "sh" | "Do" | "BI" => return None,
            _ => {}
        }
        if points.len() > MAX_EDGES || segments.len() > MAX_EDGES {
            return None;
        }
    }
    Some(output)
}

fn merge(mut edges: Vec<Edge>) -> Vec<Edge> {
    edges.sort_by(|a, b| {
        a.horizontal
            .cmp(&b.horizontal)
            .then(a.position.total_cmp(&b.position))
            .then(a.start.total_cmp(&b.start))
    });
    let mut bands: Vec<Vec<Edge>> = Vec::new();
    for e in edges {
        if let Some(band) = bands.last_mut()
            && band[0].horizontal == e.horizontal
            && (band[0].position - e.position).abs() <= TOLERANCE
        {
            band.push(e);
        } else {
            bands.push(vec![e]);
        }
    }
    let mut out: Vec<Edge> = Vec::new();
    for mut band in bands {
        band.sort_by(|a, b| a.start.total_cmp(&b.start));
        let position = band[0].position;
        for mut e in band {
            e.position = position;
            if let Some(last) = out.last_mut()
                && last.horizontal == e.horizontal
                && last.position == e.position
                && e.start <= last.end + TOLERANCE
            {
                last.end = last.end.max(e.end);
            } else {
                out.push(e);
            }
        }
    }
    out
}

#[derive(Debug)]
pub(super) struct Grid {
    pub xs: Vec<f32>,
    pub ys: Vec<f32>,
}

pub(super) fn grids(content: &Content, frame: Frame, neutral: &HashSet<Vec<u8>>) -> Vec<Grid> {
    let Some(edges) = edges(content, frame, neutral).map(merge) else {
        return Vec::new();
    };
    detect_grids(&edges)
}

fn detect_grids(edges: &[Edge]) -> Vec<Grid> {
    let mut groups: Vec<Vec<Edge>> = Vec::new();
    for &e in edges
        .iter()
        .filter(|e| e.horizontal && e.end - e.start >= 40.0)
    {
        if let Some(group) = groups.iter_mut().find(|group| {
            (group[0].start - e.start).abs() <= TOLERANCE
                && (group[0].end - e.end).abs() <= TOLERANCE
        }) {
            group.push(e);
        } else {
            groups.push(vec![e]);
        }
        if groups.len() > 128 {
            return Vec::new();
        }
    }
    let mut grids = Vec::new();
    for group in groups {
        if !(3..=513).contains(&group.len()) {
            continue;
        }
        let mut ys = group.iter().map(|e| e.position).collect::<Vec<_>>();
        ys.sort_by(f32::total_cmp);
        let first = group[0];
        let mut xs = edges
            .iter()
            .filter(|e| {
                !e.horizontal
                    && e.position >= first.start - TOLERANCE
                    && e.position <= first.end + TOLERANCE
                    && e.start <= ys[0] + TOLERANCE
                    && e.end >= ys[ys.len() - 1] - TOLERANCE
            })
            .map(|e| e.position)
            .collect::<Vec<_>>();
        xs.sort_by(f32::total_cmp);
        if !(3..=33).contains(&xs.len())
            || (xs.len() - 1) * (ys.len() - 1) > 4096
            || (xs[0] - first.start).abs() > TOLERANCE
            || (xs[xs.len() - 1] - first.end).abs() > TOLERANCE
            || xs.windows(2).any(|w| w[1] - w[0] < 4.0)
            || ys.windows(2).any(|w| w[1] - w[0] < 4.0)
        {
            continue;
        }
        grids.push(Grid { xs, ys });
        if grids.len() > 32 {
            return Vec::new();
        }
    }
    grids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed_edges(bytes: &[u8], frame: Frame) -> Option<Vec<Edge>> {
        edges(&Content::decode(bytes).unwrap(), frame, &HashSet::new())
    }
    #[test]
    fn separate_strokes_and_thin_fills_form_one_complete_grid() {
        let frame = Frame {
            x: 10.,
            y: 20.,
            width: 300.,
            height: 400.,
        };
        let bytes=b"q 1 0 0 1 10 20 cm 0 g 0 G 20 50 m 220 50 l S 20 100 200 0.2 re f 20 150 m 220 150 l S 20 50 m 20 150 l S 100 50 0.2 100 re f 220 50 m 220 150 l S Q";
        let grids = detect_grids(&merge(parsed_edges(bytes, frame).unwrap()));
        assert_eq!(grids.len(), 1);
        assert_eq!(grids[0].xs.len(), 3);
        assert_eq!(grids[0].ys.len(), 3);
        assert!((grids[0].xs[0] - 20.).abs() < 0.1);
    }
    #[test]
    fn only_neutral_graphics_states_keep_rule_geometry() {
        use lopdf::{Dictionary, Stream, dictionary};
        let mut pdf = lopdf::Document::with_version("1.7");
        let soft_mask = pdf.add_object(dictionary! {"Type"=>"Mask","S"=>"Luminosity"});
        let states = dictionary! {
            "Plain"=>dictionary!{"Type"=>"ExtGState","ca"=>1,"CA"=>1.0,"BM"=>"Normal","LW"=>2,"SA"=>true,"SMask"=>"None"},
            "Faded"=>dictionary!{"ca"=>0.5},
            "Multiply"=>dictionary!{"BM"=>"Multiply"},
            "Masked"=>dictionary!{"SMask"=>soft_mask},
            "Font"=>dictionary!{"Font"=>vec![]},
        };
        let resources = pdf.add_object(dictionary! {"ExtGState"=>states});
        let content = pdf.add_object(Stream::new(Dictionary::new(), vec![]));
        let page =
            pdf.add_object(dictionary! {"Type"=>"Page","Contents"=>content,"Resources"=>resources});
        let neutral = neutral_states(&pdf, page);
        assert_eq!(neutral, HashSet::from([b"Plain".to_vec()]));
        // A page's own unsafe or malformed definition shadows an inherited
        // neutral state of the same name.
        let inherited = pdf.add_object(dictionary! {"ExtGState"=>dictionary!{"Near"=>dictionary!{"ca"=>1},"Broken"=>dictionary!{"ca"=>1},"Far"=>dictionary!{"ca"=>1}}});
        let parent = pdf.add_object(dictionary! {"Type"=>"Pages","Resources"=>inherited});
        let own = pdf.add_object(
            dictionary! {"ExtGState"=>dictionary!{"Near"=>dictionary!{"ca"=>0.2},"Broken"=>3}},
        );
        let child = pdf.add_object(
            dictionary! {"Type"=>"Page","Parent"=>parent,"Contents"=>content,"Resources"=>own},
        );
        assert_eq!(
            neutral_states(&pdf, child),
            HashSet::from([b"Far".to_vec()])
        );
        let frame = Frame {
            x: 0.,
            y: 0.,
            width: 300.,
            height: 400.,
        };
        let rules = b"20 50 m 220 50 l S 20 100 m 220 100 l S 20 150 m 220 150 l S 20 50 m 20 150 l S 100 50 m 100 150 l S 220 50 m 220 150 l S";
        for (state, kept) in [
            ("Plain", true),
            ("Faded", false),
            ("Multiply", false),
            ("Masked", false),
            ("Font", false),
            ("Missing", false),
        ] {
            let mut bytes = format!("/{state} gs ").into_bytes();
            bytes.extend_from_slice(rules);
            let found = edges(&Content::decode(&bytes).unwrap(), frame, &neutral)
                .map(merge)
                .map(|edges| detect_grids(&edges).len());
            assert_eq!(found.is_some(), kept, "{state}");
            if kept {
                assert_eq!(found, Some(1));
            }
        }
    }
    #[test]
    fn clipped_incomplete_or_transparent_rules_do_not_invent_tables() {
        let frame = Frame {
            x: 0.,
            y: 0.,
            width: 300.,
            height: 400.,
        };
        let bytes=b"0 0 10 10 re W n 20 50 m 220 50 l S 20 100 m 220 100 l S 20 150 m 220 150 l S 20 50 m 20 150 l S 100 50 m 100 150 l S 220 50 m 220 150 l S";
        assert!(detect_grids(&merge(parsed_edges(bytes, frame).unwrap())).is_empty());
        assert!(parsed_edges(b"/GS gs", frame).is_none());
        assert!(parsed_edges(b"0 0 m 30 0 l 40 0 40 0.2 30 0.2 c h f", frame).is_none());
        assert!(parsed_edges(b"0 0 m 30 40 l W n", frame).is_none());
    }
}
