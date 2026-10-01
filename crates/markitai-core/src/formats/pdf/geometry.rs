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
    /// Inside a curved or polygonal clip, visible rule extents are unknown;
    /// such a scope contributes no edges until its graphics state is restored.
    shaped_clip: bool,
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

/// Page resources a rule verdict may pass over: graphics states that change
/// nothing it depends on (full opacity, normal blending, no soft mask; line and
/// rendering parameters are allowed) and image XObjects, which paint pixels but
/// no rules. Forms, other states and unknown names keep the page unsafe.
#[derive(Debug, Default)]
pub(super) struct RuleResources {
    pub states: HashSet<Vec<u8>>,
    pub images: HashSet<Vec<u8>>,
}

fn resolved<'a>(pdf: &'a lopdf::Document, value: &'a Object) -> Option<&'a Object> {
    pdf.dereference(value).ok().map(|(_, value)| value)
}

fn neutral_state(pdf: &lopdf::Document, state: &lopdf::Dictionary) -> bool {
    state.iter().all(|(key, value)| {
        let value = resolved(pdf, value);
        match key.as_slice() {
            b"Type" | b"LW" | b"LC" | b"LJ" | b"ML" | b"D" | b"RI" | b"FL" | b"SM" | b"SA"
            | b"OP" | b"op" | b"OPM" => true,
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
    })
}

pub(super) fn rule_resources(pdf: &lopdf::Document, id: ObjectId) -> RuleResources {
    let mut found = RuleResources::default();
    let Ok((direct, ids)) = pdf.get_page_resources(id) else {
        return found;
    };
    let mut seen_states = HashSet::new();
    let mut seen_objects = HashSet::new();
    for resources in direct
        .into_iter()
        .chain(ids.iter().filter_map(|id| pdf.get_dictionary(*id).ok()))
    {
        // The nearest dictionary defining a name decides it, even when that
        // definition is malformed; an inherited safe entry cannot mask it.
        if let Some(states) = resources
            .get(b"ExtGState")
            .ok()
            .and_then(|value| resolved(pdf, value))
            .and_then(|value| value.as_dict().ok())
        {
            for (name, value) in states.iter() {
                if seen_states.insert(name.clone())
                    && resolved(pdf, value)
                        .and_then(|value| value.as_dict().ok())
                        .is_some_and(|state| neutral_state(pdf, state))
                {
                    found.states.insert(name.clone());
                }
            }
        }
        if let Some(objects) = resources
            .get(b"XObject")
            .ok()
            .and_then(|value| resolved(pdf, value))
            .and_then(|value| value.as_dict().ok())
        {
            for (name, value) in objects.iter() {
                if seen_objects.insert(name.clone())
                    && resolved(pdf, value)
                        .and_then(|value| value.as_stream().ok())
                        .is_some_and(|stream| {
                            stream
                                .dict
                                .get(b"Subtype")
                                .and_then(Object::as_name)
                                .is_ok_and(|kind| kind == b"Image")
                        })
                {
                    found.images.insert(name.clone());
                }
            }
        }
    }
    found
}

/// A small painted shape — the disc, ring or square a browser draws as a
/// list bullet instead of a bullet character — in frame coordinates.
#[derive(Clone, Copy, Debug)]
pub(super) struct Mark {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

const MAX_MARKS: usize = 4096;
/// Extent, in points, of a shape that can be a list bullet: a browser's
/// disc is about 0.35 em, so this spans 5–20pt type.
const MARK_MIN: f32 = 1.0;
const MARK_MAX: f32 = 8.0;

fn shapes(
    content: &Content,
    frame: Frame,
    resources: &RuleResources,
) -> Option<(Vec<Edge>, Vec<Mark>)> {
    if content.operations.len() > 200_000 {
        return None;
    }
    let mut state = State {
        matrix: [1., 0., 0., 1., -frame.x, -frame.y],
        clip: [0., 0., frame.width, frame.height],
        white_fill: false,
        white_stroke: false,
        shaped_clip: false,
    };
    let mut stack = Vec::new();
    let mut points = Vec::<[f32; 2]>::new();
    let mut segments = Vec::new();
    let mut current = None;
    let mut start = None;
    let mut rectangular = false;
    let mut pending_clip = false;
    let mut curved = false;
    let mut output = Vec::new();
    let mut marks = Vec::new();
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
                // A curve's control points do not describe a table rule. Never
                // infer a closed cell from the straight portions only: the
                // whole path contributes no edges, and cannot be a clip.
                let n = numbers()?;
                let [.., x, y] = n.as_slice() else {
                    return None;
                };
                let p = state.point(*x, *y)?;
                current = Some(p);
                points.push(p);
                curved = true;
                rectangular = false;
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
                let clipping = pending_clip;
                if pending_clip {
                    if !rectangular || curved {
                        state.shaped_clip = true;
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
                if stroke && !curved && !state.shaped_clip {
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
                if fill && !curved && !state.shaped_clip && points.iter().all(|&p| state.inside(p))
                {
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
                // A compact painted shape, curved or not, may be a bullet.
                let [x0, y0, x1, y1] = bounds;
                let (width, height) = (x1 - x0, y1 - y0);
                if (fill || stroke)
                    && !clipping
                    && marks.len() < MAX_MARKS
                    && (MARK_MIN..=MARK_MAX).contains(&width)
                    && (MARK_MIN..=MARK_MAX).contains(&height)
                    && width.max(height) <= width.min(height) * 1.5
                    && points.iter().all(|&p| state.inside(p))
                {
                    marks.push(Mark { x0, y0, x1, y1 });
                }
                if output.len() > MAX_EDGES {
                    return None;
                }
                points.clear();
                segments.clear();
                current = None;
                start = None;
                rectangular = false;
                curved = false;
            }
            "gs" if op
                .operands
                .first()
                .and_then(|name| name.as_name().ok())
                .is_some_and(|name| resources.states.contains(name)) => {}
            "Do" if op
                .operands
                .first()
                .and_then(|name| name.as_name().ok())
                .is_some_and(|name| resources.images.contains(name)) => {}
            // Unknown resource colours, transparency, Forms and shading make
            // a geometrical table verdict unsafe. Text can still be recovered.
            "gs" | "cs" | "CS" | "sc" | "SC" | "scn" | "SCN" | "sh" | "Do" | "BI" => return None,
            _ => {}
        }
        if points.len() > MAX_EDGES || segments.len() > MAX_EDGES {
            return None;
        }
    }
    Some((output, marks))
}

#[cfg(test)]
fn edges(content: &Content, frame: Frame, resources: &RuleResources) -> Option<Vec<Edge>> {
    shapes(content, frame, resources).map(|(edges, _)| edges)
}

/// A stable sort of edges. Its comparator is a trait object, so the two
/// orders `merge` sorts by share one compiled sort.
fn sort_edges(edges: &mut [Edge], compare: &mut dyn FnMut(&Edge, &Edge) -> std::cmp::Ordering) {
    edges.sort_by(|a, b| compare(a, b));
}

fn merge(mut edges: Vec<Edge>) -> Vec<Edge> {
    sort_edges(&mut edges, &mut |a, b| {
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
        sort_edges(&mut band, &mut |a, b| a.start.total_cmp(&b.start));
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

#[cfg(test)]
pub(super) fn grids(content: &Content, frame: Frame, resources: &RuleResources) -> Vec<Grid> {
    page_shapes(content, frame, resources).0
}

/// The page's complete ruled tables and its bullet-sized marks, from one
/// walk of its content. A page whose paint cannot be judged has neither.
pub(super) fn page_shapes(
    content: &Content,
    frame: Frame,
    resources: &RuleResources,
) -> (Vec<Grid>, Vec<Mark>) {
    let Some((edges, marks)) = shapes(content, frame, resources) else {
        return (Vec::new(), Vec::new());
    };
    (detect_grids(&merge(edges)), marks)
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
        let first = group[0];
        let verticals = edges
            .iter()
            .filter(|e| {
                !e.horizontal
                    && e.position >= first.start - TOLERANCE
                    && e.position <= first.end + TOLERANCE
            })
            .collect::<Vec<_>>();
        let mut ys = group.iter().map(|e| e.position).collect::<Vec<_>>();
        ys.sort_by(f32::total_cmp);
        // Separately ruled boxes of the same width (a table and a bordered
        // block below it) share a width group; only rules joined by a
        // vertical border belong to one grid.
        let mut runs: Vec<Vec<f32>> = Vec::new();
        for y in ys {
            if let Some(run) = runs.last_mut()
                && verticals
                    .iter()
                    .any(|e| e.start <= run[run.len() - 1] + TOLERANCE && e.end >= y - TOLERANCE)
            {
                run.push(y);
            } else {
                runs.push(vec![y]);
            }
        }
        for ys in runs {
            if !(3..=513).contains(&ys.len()) {
                continue;
            }
            let mut xs = verticals
                .iter()
                .filter(|e| e.start <= ys[0] + TOLERANCE && e.end >= ys[ys.len() - 1] - TOLERANCE)
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
    }
    grids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed_edges(bytes: &[u8], frame: Frame) -> Option<Vec<Edge>> {
        edges(
            &Content::decode(bytes).unwrap(),
            frame,
            &RuleResources::default(),
        )
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
        let neutral = rule_resources(&pdf, page);
        assert_eq!(neutral.states, HashSet::from([b"Plain".to_vec()]));
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
            rule_resources(&pdf, child).states,
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
    fn image_invocations_and_curved_marks_leave_the_rule_grid_intact() {
        use lopdf::{Dictionary, Stream, dictionary};
        let mut pdf = lopdf::Document::with_version("1.7");
        let image = pdf.add_object(Stream::new(
            dictionary! {"Subtype"=>"Image","Width"=>1,"Height"=>1},
            vec![0],
        ));
        let form = pdf.add_object(Stream::new(dictionary! {"Subtype"=>"Form"}, vec![]));
        let resources = pdf.add_object(
            dictionary! {"XObject"=>dictionary!{"Im0"=>image,"Fm0"=>form,"Im1"=>image}},
        );
        let parent_resources = pdf.add_object(dictionary! {"XObject"=>dictionary!{"Fm1"=>image}});
        let parent = pdf.add_object(dictionary! {"Type"=>"Pages","Resources"=>parent_resources});
        let own = pdf.add_object(dictionary! {"XObject"=>dictionary!{"Fm1"=>form}});
        let content = pdf.add_object(Stream::new(Dictionary::new(), vec![]));
        let page =
            pdf.add_object(dictionary! {"Type"=>"Page","Contents"=>content,"Resources"=>resources});
        let child = pdf.add_object(
            dictionary! {"Type"=>"Page","Parent"=>parent,"Contents"=>content,"Resources"=>own},
        );
        let found = rule_resources(&pdf, page);
        assert_eq!(
            found.images,
            HashSet::from([b"Im0".to_vec(), b"Im1".to_vec()])
        );
        // A nearer Form of the same name shadows an inherited image.
        assert!(rule_resources(&pdf, child).images.is_empty());
        let frame = Frame {
            x: 0.,
            y: 0.,
            width: 300.,
            height: 400.,
        };
        let rules = "20 50 m 220 50 l S 20 100 m 220 100 l S 20 150 m 220 150 l S 20 50 m 20 150 l S 100 50 m 100 150 l S 220 50 m 220 150 l S";
        // A bullet drawn with curves and an image beside the table.
        let marks = "q 10 0 0 10 240 60 cm /Im0 Do Q 5 5 m 5 8 8 8 8 5 c 8 2 5 2 5 5 c f";
        for (extra, grids) in [(marks, Some(1)), ("/Fm0 Do", None), ("/Missing Do", None)] {
            let bytes = format!("{rules} {extra}");
            let found = edges(
                &Content::decode(bytes.as_bytes()).unwrap(),
                frame,
                &rule_resources(&pdf, page),
            )
            .map(|edges| detect_grids(&merge(edges)).len());
            assert_eq!(found, grids, "{extra}");
        }
    }

    #[test]
    fn a_same_width_box_below_a_table_is_a_separate_region() {
        let frame = Frame {
            x: 0.,
            y: 0.,
            width: 300.,
            height: 400.,
        };
        // A three-row ruled table, then a bordered block of the same width
        // below it whose rules are not joined to the table's.
        let table = "20 300 m 220 300 l S 20 330 m 220 330 l S 20 360 m 220 360 l S 20 390 m 220 390 l S 20 300 m 20 390 l S 120 300 m 120 390 l S 220 300 m 220 390 l S";
        let block = "20.4 200 m 219.6 200 l S 20.4 280 m 219.6 280 l S 20.4 200 m 20.4 280 l S 219.6 200 m 219.6 280 l S";
        let bytes = format!("{table} {block}");
        let grids = detect_grids(&merge(parsed_edges(bytes.as_bytes(), frame).unwrap()));
        assert_eq!(grids.len(), 1);
        assert_eq!(grids[0].ys.len(), 4);
        assert!((grids[0].ys[0] - 300.).abs() < 0.1);
        assert_eq!(grids[0].xs.len(), 3);
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
        // A curved path contributes no edges instead of declining the page,
        // and cannot define a clip.
        assert!(
            parsed_edges(b"0 0 m 30 0 l 40 0 40 0.2 30 0.2 c h f", frame)
                .unwrap()
                .is_empty()
        );
        // Rules inside a curved or polygonal clip are withheld until the
        // graphics state is restored; rules after it still count.
        let rules = "20 50 m 220 50 l S 20 100 m 220 100 l S 20 150 m 220 150 l S 20 50 m 20 150 l S 100 50 m 100 150 l S 220 50 m 220 150 l S";
        for clip in [
            "0 0 m 300 0 l 300 390 300 400 290 400 c 0 400 l h W n",
            "0 0 m 300 0 l 0 400 l h W n",
        ] {
            let inside = format!("q {clip} {rules} Q");
            let edges = parsed_edges(inside.as_bytes(), frame).unwrap();
            assert!(detect_grids(&merge(edges)).is_empty(), "{clip}");
            let after = format!("q {clip} Q {rules}");
            let edges = parsed_edges(after.as_bytes(), frame).unwrap();
            assert_eq!(detect_grids(&merge(edges)).len(), 1, "{clip}");
        }
        assert!(parsed_edges(b"/Im0 Do", frame).is_none());
        assert!(
            parsed_edges(b"0 0 m 30 40 l W n", frame)
                .unwrap()
                .is_empty()
        );
    }
}
