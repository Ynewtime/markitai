//! What a page's pixels add to the lines a recognizer read from it: the
//! weight and slant of each word (bold, italic), the list bullets a
//! recognizer drops (a lone dot is not text to it), the ruled grids of
//! tables, and the pictures (charts, photographs) whose labels are not body
//! text. Measured on 150 DPI renderings of printed pages (scans of a
//! LibreOffice document, 2026-10-23): the stroke of a word in a regular face
//! is 1.6–2.3 pixels, of the same word in bold 2.5–3.3; an italic word's
//! columns line up best sheared by a quarter, with 1.17–1.55 times the
//! upright profile's peak, an upright word's by none (at most 1.02).

use super::Line;
use image::RgbImage;

/// A pixel darker than this share of the paper's luminance is ink (text).
const INK: f32 = 0.6;
/// A pixel darker than this share, or colored, is marked: the light gray
/// rules of a table and a picture's pale areas are marked, not ink.
const MARK: f32 = 0.88;
/// A pixel whose channels differ by this much is colored.
const CHROMA: i16 = 48;
/// A page whose paper is darker than this is not read for marks.
const MIN_PAPER: f32 = 80.0;

/// A word whose stroke is this many times the body's is bold; next to a
/// bold word, [`BOLD_NEXT`] is enough.
pub(super) const BOLD: f32 = 1.5;
pub(super) const BOLD_NEXT: f32 = 1.35;
/// The shears tried for a word's slant, and the share its best one must
/// sharpen the column profile by to be italic.
const SHEARS: [f32; 6] = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3];
const ITALIC_GAIN: f32 = 1.12;
/// A word needs this many ink pixels before its stroke and slant count.
const MIN_WORD_INK: u32 = 30;

/// What the pixels tell of one recognized line.
#[derive(Clone, Debug, Default)]
pub(super) struct Marks {
    /// The line's stroke width, in pixels.
    pub stroke: Option<f32>,
    /// Its words' stroke and slant, one per whitespace-separated word of its
    /// text; none when the ink's words do not match the text's.
    pub words: Option<Vec<Word>>,
    /// The left edge of a bullet drawn left of it, which the recognizer
    /// left out.
    pub bullet: Option<f32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Word {
    pub stroke: f32,
    pub italic: bool,
}

/// A page's pixels, as luminance, with the paper's level.
pub(super) struct Page<'a> {
    image: &'a RgbImage,
    width: u32,
    height: u32,
    luma: Vec<u8>,
    ink: u8,
    mark: u8,
}

/// Rules of one table: its box, the x of its column edges and the y of its
/// row edges, each sorted, the box's own edges included.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Grid {
    pub bounds: [f32; 4],
    pub columns: Vec<f32>,
    pub rows: Vec<f32>,
}

impl<'a> Page<'a> {
    /// `None` for a dark or empty image: its ink cannot be told.
    pub(super) fn new(image: &'a RgbImage) -> Option<Self> {
        let (width, height) = image.dimensions();
        if width < 8 || height < 8 {
            return None;
        }
        let luma: Vec<u8> = image
            .pixels()
            .map(|p| {
                ((299 * u32::from(p[0]) + 587 * u32::from(p[1]) + 114 * u32::from(p[2])) / 1000)
                    as u8
            })
            .collect();
        let mut histogram = [0u64; 256];
        for &value in luma.iter().step_by(3) {
            histogram[usize::from(value)] += 1;
        }
        let total: u64 = histogram.iter().sum();
        let mut seen = 0;
        let mut paper = 255;
        for (level, count) in histogram.iter().enumerate() {
            seen += count;
            if seen * 10 >= total * 9 {
                paper = level;
                break;
            }
        }
        let paper = paper as f32;
        (paper >= MIN_PAPER).then_some(Page {
            image,
            width,
            height,
            luma,
            ink: (INK * paper) as u8,
            mark: (MARK * paper) as u8,
        })
    }

    fn at(&self, x: u32, y: u32) -> u8 {
        self.luma[(y * self.width + x) as usize]
    }

    fn ink(&self, x: u32, y: u32) -> bool {
        self.at(x, y) < self.ink
    }

    fn marked(&self, x: u32, y: u32) -> bool {
        if self.at(x, y) < self.mark {
            return true;
        }
        let p = self.image.get_pixel(x, y);
        let (high, low) = (p[0].max(p[1]).max(p[2]), p[0].min(p[1]).min(p[2]));
        i16::from(high) - i16::from(low) >= CHROMA
    }

    /// The whole pixels of `bounds` within the page; none when empty.
    fn area(&self, [l, t, r, b]: [f32; 4]) -> Option<[u32; 4]> {
        let clamp = |v: f32, max: u32| (v.max(0.0) as u32).min(max);
        let area = [
            clamp(l.floor(), self.width),
            clamp(t.floor(), self.height),
            clamp(r.ceil(), self.width),
            clamp(b.ceil(), self.height),
        ];
        (area[0] < area[2] && area[1] < area[3]).then_some(area)
    }

    /// The stroke width of the ink in `bounds`: twice its area over its
    /// edges (a stroke of width w and length L has area wL and 2L edges).
    fn stroke_in(&self, [l, t, r, b]: [u32; 4]) -> Option<(f32, u32)> {
        let (mut area, mut edges) = (0u32, 0u32);
        for y in t..b {
            for x in l..r {
                let ink = self.ink(x, y);
                area += u32::from(ink);
                if x + 1 < r && ink != self.ink(x + 1, y) {
                    edges += 1;
                }
                if y + 1 < b && ink != self.ink(x, y + 1) {
                    edges += 1;
                }
            }
        }
        (area >= MIN_WORD_INK && edges > 0).then(|| (2.0 * area as f32 / edges as f32, area))
    }

    /// Whether the ink in `bounds` stands slanted: some shear right lines its
    /// columns up [`ITALIC_GAIN`] times better than none.
    fn italic_in(&self, [l, t, r, b]: [u32; 4]) -> bool {
        // Every shear moves a row left by less than this.
        let margin = (0.35 * (b - t) as f32).ceil() as usize + 1;
        let mut best = 0u64;
        let mut upright = 0u64;
        let mut profile = vec![0u32; (r - l) as usize + margin];
        for shear in SHEARS {
            profile.iter_mut().for_each(|count| *count = 0);
            for y in t..b {
                // Higher rows lean right in italic: shift them back left.
                let shift = (shear * (b - 1 - y) as f32).round() as usize;
                for x in l..r {
                    if self.ink(x, y) {
                        profile[(x - l) as usize + margin - shift.min(margin)] += 1;
                    }
                }
            }
            let score: u64 = profile.iter().map(|&c| u64::from(c) * u64::from(c)).sum();
            if shear == 0.0 {
                upright = score;
            }
            best = best.max(score);
        }
        upright > 0 && best as f32 >= ITALIC_GAIN * upright as f32
    }

    /// The x spans of the words in `area`: runs of columns with ink, apart
    /// by gaps of a fifth of the line's height or more.
    fn words_in(&self, [l, t, r, b]: [u32; 4]) -> Vec<[u32; 2]> {
        let gap = (((b - t) as f32 * 0.2).round() as u32).max(2);
        let mut words: Vec<[u32; 2]> = Vec::new();
        let mut empty = 0;
        for x in l..r {
            if (t..b).any(|y| self.ink(x, y)) {
                match words.last_mut() {
                    Some(word) if empty < gap => word[1] = x + 1,
                    _ => words.push([x, x + 1]),
                }
                empty = 0;
            } else {
                empty += 1;
            }
        }
        words
    }

    /// What the pixels tell of `line`.
    pub(super) fn marks(&self, line: &Line) -> Marks {
        let Some(area) = self.area(line.bounds) else {
            return Marks::default();
        };
        let stroke = self.stroke_in(area).map(|(stroke, _)| stroke);
        let tokens = line.text.split_whitespace().count();
        let spans = self.words_in(area);
        let words = (tokens > 0 && spans.len() == tokens).then(|| {
            spans
                .iter()
                .map(|&[l, r]| {
                    let word = [l, area[1], r, area[3]];
                    match self.stroke_in(word) {
                        Some((stroke, _)) => Word {
                            stroke,
                            italic: self.italic_in(word),
                        },
                        None => Word::default(),
                    }
                })
                .collect()
        });
        Marks {
            stroke,
            words,
            bullet: self.bullet(line.bounds).map(|bullet| bullet[0]),
        }
    }

    /// A bullet drawn left of a line: the one solid mark (of a few pixels at
    /// least) within three and a half line heights of its left edge, from a
    /// tenth to three fifths of the line's height on each side, about as
    /// wide as tall, at least a third filled, around the middle of the line.
    pub(super) fn bullet(&self, [l, t, r, b]: [f32; 4]) -> Option<[f32; 4]> {
        let h = b - t;
        if h < 6.0 || r <= l {
            return None;
        }
        let area = self.area([l - 3.5 * h, t, l - 0.1 * h, b])?;
        let parts = self.parts(area, |page, x, y| page.ink(x, y), 4)?;
        let [part] = parts.as_slice() else {
            return None;
        };
        let ([pl, pt, pr, pb], count) = *part;
        let (w, ht) = ((pr - pl) as f32, (pb - pt) as f32);
        let middle = (pt + pb) as f32 / 2.0;
        let sized = |side: f32| (0.1 * h..=0.6 * h).contains(&side);
        (sized(w)
            && sized(ht)
            && (0.5..=2.0).contains(&(w / ht))
            && count as f32 >= 0.3 * w * ht
            && middle > t + 0.15 * h
            && middle < b - 0.1 * h)
            .then_some([pl as f32, pt as f32, pr as f32, pb as f32])
    }

    /// The 8-connected parts of the pixels `on` in `area`, with their pixel
    /// counts, ignoring specks of fewer than `min` pixels; `None` past 16
    /// parts.
    #[allow(clippy::type_complexity)]
    fn parts(
        &self,
        [l, t, r, b]: [u32; 4],
        on: impl Fn(&Self, u32, u32) -> bool,
        min: u32,
    ) -> Option<Vec<([u32; 4], u32)>> {
        let (w, h) = ((r - l) as usize, (b - t) as usize);
        let mut seen = vec![false; w * h];
        let mut parts = Vec::new();
        let mut stack = Vec::new();
        for y in 0..h {
            for x in 0..w {
                if seen[y * w + x] || !on(self, l + x as u32, t + y as u32) {
                    continue;
                }
                seen[y * w + x] = true;
                stack.push((x, y));
                let (mut bounds, mut count) = ([x, y, x + 1, y + 1], 0u32);
                while let Some((px, py)) = stack.pop() {
                    count += 1;
                    bounds = [
                        bounds[0].min(px),
                        bounds[1].min(py),
                        bounds[2].max(px + 1),
                        bounds[3].max(py + 1),
                    ];
                    for ny in py.saturating_sub(1)..(py + 2).min(h) {
                        for nx in px.saturating_sub(1)..(px + 2).min(w) {
                            if !seen[ny * w + nx] && on(self, l + nx as u32, t + ny as u32) {
                                seen[ny * w + nx] = true;
                                stack.push((nx, ny));
                            }
                        }
                    }
                }
                if count >= min {
                    parts.push((
                        [
                            l + bounds[0] as u32,
                            t + bounds[1] as u32,
                            l + bounds[2] as u32,
                            t + bounds[3] as u32,
                        ],
                        count,
                    ));
                    if parts.len() > 16 {
                        return None;
                    }
                }
            }
        }
        Some(parts)
    }

    /// Thin straight marks at least `long` pixels long, as `[x0, y0, x1,
    /// y1]`: horizontal ones (`across`) or vertical ones. A run may skip two
    /// unmarked pixels (a rule's JPEG noise, a crossing); runs on next rows
    /// that overlap are one rule, kept when at most `thick` pixels thick.
    fn rules(&self, across: bool, long: u32, thick: u32) -> Vec<[u32; 4]> {
        let (lines, length) = if across {
            (self.height, self.width)
        } else {
            (self.width, self.height)
        };
        let marked = |line: u32, at: u32| {
            if across {
                self.marked(at, line)
            } else {
                self.marked(line, at)
            }
        };
        // Open rules: [start, end, first line, last line].
        let mut open: Vec<[u32; 4]> = Vec::new();
        let mut done: Vec<[u32; 4]> = Vec::new();
        for line in 0..lines {
            let mut runs = Vec::new();
            let (mut start, mut last) = (None::<u32>, 0u32);
            for at in 0..length {
                if marked(line, at) {
                    if start.is_none() {
                        start = Some(at);
                    }
                    last = at;
                } else if let Some(s) = start
                    && at - last > 2
                {
                    if last + 1 - s >= long {
                        runs.push([s, last + 1]);
                    }
                    start = None;
                }
            }
            if let Some(s) = start
                && last + 1 - s >= long
            {
                runs.push([s, last + 1]);
            }
            let mut next = Vec::new();
            for run in runs {
                match open
                    .iter()
                    .position(|rule| rule[3] + 1 == line && run[0] < rule[1] && rule[0] < run[1])
                {
                    Some(index) => {
                        let mut rule = open.swap_remove(index);
                        rule = [rule[0].min(run[0]), rule[1].max(run[1]), rule[2], line];
                        next.push(rule);
                    }
                    None => next.push([run[0], run[1], line, line]),
                }
            }
            done.append(&mut open);
            open = next;
            if done.len() + open.len() > 4096 {
                break;
            }
        }
        done.append(&mut open);
        done.into_iter()
            .filter(|rule| rule[3] - rule[2] < thick)
            .map(|[s, e, first, last]| {
                if across {
                    [s, first, e, last + 1]
                } else {
                    [first, s, last + 1, e]
                }
            })
            .collect()
    }

    /// The ruled tables of the page: horizontal and vertical rules that
    /// cross, two rows and two columns of cells at least, `body` being the
    /// height of a line of text.
    pub(super) fn grids(&self, body: f32) -> Vec<Grid> {
        let thick = ((0.15 * body) as u32).max(3) + 1;
        let across = self.rules(true, ((3.0 * body) as u32).max(self.width / 16), thick);
        let down = self.rules(false, ((1.2 * body) as u32).max(8), thick);
        if across.len() < 2 || down.len() < 2 || across.len() + down.len() > 2000 {
            return Vec::new();
        }
        let tolerance = (0.3 * body).max(4.0) as u32;
        let rules: Vec<[u32; 4]> = across.iter().chain(&down).copied().collect();
        let mut group: Vec<usize> = (0..rules.len()).collect();
        fn root(group: &mut [usize], mut i: usize) -> usize {
            while group[i] != i {
                group[i] = group[group[i]];
                i = group[i];
            }
            i
        }
        for (a, h) in across.iter().enumerate() {
            for (b, v) in down.iter().enumerate() {
                let crosses = v[0] + tolerance >= h[0]
                    && v[2] <= h[2] + tolerance
                    && h[1] + tolerance >= v[1]
                    && h[3] <= v[3] + tolerance;
                if crosses {
                    let (x, y) = (root(&mut group, a), root(&mut group, across.len() + b));
                    group[x] = y;
                }
            }
        }
        let mut grids = Vec::new();
        let roots: std::collections::BTreeSet<usize> =
            (0..rules.len()).map(|i| root(&mut group, i)).collect();
        for top in roots {
            let members: Vec<usize> = (0..rules.len())
                .filter(|&i| root(&mut group, i) == top)
                .collect();
            let (hs, vs): (Vec<usize>, Vec<usize>) =
                members.iter().partition(|&&i| i < across.len());
            if hs.len() < 2 || vs.len() < 2 {
                continue;
            }
            let bounds = members.iter().fold([u32::MAX, u32::MAX, 0, 0], |b, &i| {
                let r = rules[i];
                [
                    b[0].min(r[0]),
                    b[1].min(r[1]),
                    b[2].max(r[2]),
                    b[3].max(r[3]),
                ]
            });
            let edges = |values: Vec<f32>, low: u32, high: u32| {
                let mut values = values;
                values.push(low as f32);
                values.push(high as f32);
                values.sort_by(f32::total_cmp);
                let mut kept: Vec<f32> = Vec::new();
                for value in values {
                    match kept.last() {
                        Some(&last) if value - last <= tolerance as f32 => {}
                        _ => kept.push(value),
                    }
                }
                kept
            };
            // Rules across most of the table divide its rows; rules down
            // most of a row band divide its columns.
            let width = (bounds[2] - bounds[0]) as f32;
            let rows = edges(
                hs.iter()
                    .map(|&i| rules[i])
                    .filter(|r| (r[2] - r[0]) as f32 >= 0.5 * width)
                    .map(|r| (r[1] + r[3]) as f32 / 2.0)
                    .collect(),
                bounds[1],
                bounds[3],
            );
            let columns = edges(
                vs.iter()
                    .map(|&i| rules[i])
                    .map(|r| (r[0] + r[2]) as f32 / 2.0)
                    .collect(),
                bounds[0],
                bounds[2],
            );
            if rows.len() >= 3 && columns.len() >= 3 && rows.len() <= 500 && columns.len() <= 64 {
                grids.push(Grid {
                    bounds: bounds.map(|v| v as f32),
                    columns,
                    rows,
                });
            }
        }
        grids
    }

    /// The pictures of the page: marked areas that no text line, `skip`
    /// box or rule-thin mark explains, of at least two and a half `body`
    /// line heights each way. Their raw boxes, before labels join them.
    pub(super) fn pictures(&self, lines: &[Line], skip: &[[f32; 4]], body: f32) -> Vec<[f32; 4]> {
        let cell = ((body / 3.0).round() as u32).max(3);
        let (cw, ch) = (self.width.div_ceil(cell), self.height.div_ceil(cell));
        let (cw, ch) = (cw as usize, ch as usize);
        let mut busy = vec![false; cw * ch];
        let threshold = (cell * cell / 20).max(2);
        for cy in 0..ch {
            for cx in 0..cw {
                let (x0, y0) = (cx as u32 * cell, cy as u32 * cell);
                let mut count = 0;
                'cell: for y in y0..(y0 + cell).min(self.height) {
                    for x in x0..(x0 + cell).min(self.width) {
                        if self.marked(x, y) {
                            count += 1;
                            if count >= threshold {
                                break 'cell;
                            }
                        }
                    }
                }
                busy[cy * cw + cx] = count >= threshold;
            }
        }
        let pad = 0.15 * body;
        let mut clear = |[l, t, r, b]: [f32; 4]| {
            let to = |v: f32, max: usize| ((v.max(0.0) / cell as f32) as usize).min(max);
            for cy in to(t, ch)..to(b, ch - 1) + 1 {
                for cx in to(l, cw)..to(r, cw - 1) + 1 {
                    if cy < ch && cx < cw {
                        busy[cy * cw + cx] = false;
                    }
                }
            }
        };
        for line in lines {
            let [l, t, r, b] = line.bounds;
            clear([l - pad, t - pad, r + pad, b + pad]);
        }
        for &area in skip {
            clear(area);
        }
        // Parts of busy cells, two cells apart at most.
        const REACH: usize = 2;
        let mut part = vec![usize::MAX; cw * ch];
        let mut boxes: Vec<([usize; 4], usize)> = Vec::new();
        let mut stack = Vec::new();
        for start in 0..cw * ch {
            if !busy[start] || part[start] != usize::MAX {
                continue;
            }
            let id = boxes.len();
            part[start] = id;
            stack.push(start);
            let (sx, sy) = (start % cw, start / cw);
            let mut bounds = [sx, sy, sx + 1, sy + 1];
            let mut count = 0;
            while let Some(at) = stack.pop() {
                count += 1;
                let (x, y) = (at % cw, at / cw);
                bounds = [
                    bounds[0].min(x),
                    bounds[1].min(y),
                    bounds[2].max(x + 1),
                    bounds[3].max(y + 1),
                ];
                for ny in y.saturating_sub(REACH)..(y + REACH + 1).min(ch) {
                    for nx in x.saturating_sub(REACH)..(x + REACH + 1).min(cw) {
                        let next = ny * cw + nx;
                        if busy[next] && part[next] == usize::MAX {
                            part[next] = id;
                            stack.push(next);
                        }
                    }
                }
            }
            boxes.push((bounds, count));
            if boxes.len() > 4096 {
                return Vec::new();
            }
        }
        let side = 2.5 * body;
        let mut pictures: Vec<[f32; 4]> = boxes
            .into_iter()
            .filter(|(_, count)| *count >= 12)
            .map(|(b, _)| {
                [
                    (b[0] as u32 * cell) as f32,
                    (b[1] as u32 * cell) as f32,
                    ((b[2] as u32 * cell).min(self.width)) as f32,
                    ((b[3] as u32 * cell).min(self.height)) as f32,
                ]
            })
            .filter(|b| b[2] - b[0] >= side && b[3] - b[1] >= side)
            .collect();
        pictures.truncate(32);
        pictures
    }

    /// The pixels of `bounds`.
    pub(super) fn crop(&self, bounds: [f32; 4]) -> Option<RgbImage> {
        let [l, t, r, b] = self.area(bounds)?;
        Some(image::imageops::crop_imm(self.image, l, t, r - l, b - t).to_image())
    }

    /// The x spans of the words of `bounds`, for splitting a line across
    /// table cells.
    pub(super) fn word_spans(&self, bounds: [f32; 4]) -> Vec<[f32; 2]> {
        self.area(bounds).map_or_else(Vec::new, |area| {
            self.words_in(area)
                .into_iter()
                .map(|[l, r]| [l as f32, r as f32])
                .collect()
        })
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use image::Rgb;

    pub(crate) fn paper(width: u32, height: u32) -> RgbImage {
        RgbImage::from_pixel(width, height, Rgb([255; 3]))
    }

    pub(crate) fn fill(image: &mut RgbImage, [l, t, r, b]: [u32; 4], color: [u8; 3]) {
        for y in t..b {
            for x in l..r {
                image.put_pixel(x, y, Rgb(color));
            }
        }
    }

    /// A word of `letters` letters drawn as stems `stroke` pixels wide,
    /// leaning right by `shear` per row; returns its right edge.
    pub(crate) fn word(
        image: &mut RgbImage,
        left: u32,
        top: u32,
        letters: u32,
        stroke: u32,
        shear: f32,
    ) -> u32 {
        let mut x = left;
        for _ in 0..letters {
            for y in top + 4..top + 22 {
                let lean = (shear * (top + 22 - y) as f32) as u32;
                fill(image, [x + lean, y, x + lean + stroke, y + 1], [0; 3]);
            }
            // A bar across the letter's foot.
            fill(image, [x, top + 20, x + 8, top + 22], [0; 3]);
            x += 10;
        }
        x
    }

    fn line(text: &str, bounds: [f32; 4]) -> Line {
        Line {
            text: text.into(),
            confidence: 1.0,
            bounds,
            direction: [1.0, 0.0],
            marks: Marks::default(),
        }
    }

    #[test]
    fn words_are_bold_or_italic_by_their_stroke_and_slant() {
        let mut image = paper(400, 40);
        let mut x = 10;
        x = word(&mut image, x, 4, 5, 2, 0.0) + 8;
        x = word(&mut image, x, 4, 5, 4, 0.0) + 8;
        word(&mut image, x, 4, 5, 2, 0.3);
        let page = Page::new(&image).unwrap();
        let marks = page.marks(&line("plain bold slanted", [8., 2., 390., 30.]));
        let words = marks.words.expect("three words, as the text");
        assert!(words[1].stroke >= BOLD * words[0].stroke, "{words:?}");
        assert!(
            !words[0].italic && !words[1].italic && words[2].italic,
            "{words:?}"
        );
        // A text of another word count gets no word marks.
        assert!(
            page.marks(&line("two words", [8., 2., 390., 30.]))
                .words
                .is_none()
        );
    }

    #[test]
    fn a_lone_dot_left_of_a_line_is_its_bullet() {
        let mut image = paper(300, 60);
        fill(&mut image, [20, 12, 28, 20], [0; 3]);
        word(&mut image, 40, 2, 4, 2, 0.0);
        // Text, not a dot: the line beside a word.
        word(&mut image, 100, 32, 3, 2, 0.0);
        let page = Page::new(&image).unwrap();
        assert!(page.bullet([40., 2., 80., 30.]).is_some());
        assert!(page.bullet([140., 32., 200., 58.]).is_none());
        // A long dash is not a bullet.
        let mut dash = paper(300, 40);
        fill(&mut dash, [10, 14, 34, 17], [0; 3]);
        let page = Page::new(&dash).unwrap();
        assert!(page.bullet([40., 2., 80., 30.]).is_none());
    }

    #[test]
    fn light_rules_that_cross_are_a_grid() {
        let mut image = paper(600, 400);
        for y in [50, 110, 170, 230] {
            fill(&mut image, [40, y, 560, y + 1], [190; 3]);
        }
        for x in [40, 200, 559] {
            fill(&mut image, [x, 50, x + 1, 231], [190; 3]);
        }
        let page = Page::new(&image).unwrap();
        let grids = page.grids(28.0);
        assert_eq!(grids.len(), 1, "{grids:?}");
        assert_eq!(grids[0].rows.len(), 4);
        assert_eq!(grids[0].columns.len(), 3);
        // Rules that never cross (underlines) are no grid.
        let mut underlines = paper(600, 400);
        for y in [50, 110] {
            fill(&mut underlines, [40, y, 560, y + 1], [0; 3]);
        }
        assert!(Page::new(&underlines).unwrap().grids(28.0).is_empty());
    }

    #[test]
    fn colored_shapes_without_text_are_a_picture() {
        let mut image = paper(800, 800);
        // Bars of a chart, one pale yellow.
        fill(&mut image, [200, 300, 240, 500], [0, 60, 140]);
        fill(&mut image, [250, 350, 290, 500], [255, 204, 0]);
        fill(&mut image, [300, 250, 340, 500], [255, 69, 0]);
        // A text line drawn and recognized.
        word(&mut image, 100, 100, 30, 2, 0.0);
        let text = line("words", [100., 100., 400., 128.]);
        let page = Page::new(&image).unwrap();
        let pictures = page.pictures(&[text], &[], 28.0);
        assert_eq!(pictures.len(), 1, "{pictures:?}");
        let [l, t, r, b] = pictures[0];
        assert!(
            l <= 200. && t <= 250. && r >= 340. && b >= 500.,
            "{pictures:?}"
        );
        assert!(b - t < 300.);
    }
}
