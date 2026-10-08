//! Text detection with a DB (differentiable binarization) model: the image
//! is scaled to the model's input, the model returns a probability map of
//! text, and the map's connected regions above a threshold become rotated
//! rectangles, scored by their mean probability and grown back to the text's
//! full extent (DB shrinks text regions during training by the same rule).

use super::super::{Result, failure};
use image::RgbImage;

/// Pixels whose text probability exceeds this are text.
const THRESHOLD: f32 = 0.3;
/// A region whose mean probability is below this is not text.
const BOX_THRESHOLD: f32 = 0.5;
/// How far a region grows back: its area times this, over its perimeter.
const UNCLIP_RATIO: f32 = 1.6;
/// Regions with a shorter side than this (in map pixels) are specks.
const MIN_SIDE: f32 = 3.0;
/// Text regions read per image, at most; specks and faint regions do not
/// count. [`crate::ocr::capped_warning`] names this number.
pub(super) const MAX_REGIONS: usize = 1000;
/// The model reads sides that are multiples of this.
pub(super) const STRIDE: u32 = 32;
/// The longer side of an image is scaled down to this before detection. The
/// reference uses 2,000; on the 300 DPI and full-page corpora 1,600 reads the
/// same (two images one edit better, two one worse) with a third fewer
/// pixels to hold and compute.
const MAX_LONG_SIDE: f32 = 1600.0;
/// A small image (shorter side below this) is scaled up toward it, by at
/// most [`MAX_UPSCALE`]. The reference scales every image up to a 736-pixel
/// short side; measured on the rendered corpora that costs up to three times
/// the time and reads no better (English and numbers equal, Chinese and
/// Japanese at 72 and 96 DPI 0.2 to 0.4 points worse), while a 1.5 times
/// enlargement of small images reads them best.
const MIN_SHORT_SIDE: f32 = 400.0;
const MAX_UPSCALE: f32 = 1.5;
/// Detection inputs hold at most this many pixels; larger ones are scaled down.
const MAX_INPUT_PIXELS: f64 = 3_200_000.0;
/// A strip at least this many times wider than tall, or at most this many
/// pixels tall, is padded above and below before detection: the model finds
/// a lone line poorly when it touches the image's edges.
const STRIP_RATIO: f32 = 8.0;
const STRIP_HEIGHT: u32 = 30;

/// How an image is laid out on the detection model's input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Layout {
    /// The input's size, multiples of [`STRIDE`].
    pub width: u32,
    pub height: u32,
    /// Rows of background above the image, in input pixels.
    pub top: u32,
    /// Input pixels per image pixel, horizontally and vertically.
    pub scale_x: f32,
    pub scale_y: f32,
    /// The image's own size.
    pub image_width: u32,
    pub image_height: u32,
}

impl Layout {
    /// The layout of a `width` by `height` image.
    pub fn of(width: u32, height: u32) -> Layout {
        let (w, h) = (width.max(1) as f32, height.max(1) as f32);
        // A strip is padded to a height of an eighth of its width (at least
        // 60 pixels), centered.
        let padded = if height <= STRIP_HEIGHT || w / h > STRIP_RATIO {
            ((w / STRIP_RATIO).max(STRIP_HEIGHT as f32) * 2.0).max(h)
        } else {
            h
        };
        let pad = (padded - h) / 2.0;
        let mut scale = 1.0f32;
        if w.max(padded) > MAX_LONG_SIDE {
            scale = MAX_LONG_SIDE / w.max(padded);
        }
        if w.min(padded) * scale < MIN_SHORT_SIDE {
            scale = (MIN_SHORT_SIDE / w.min(padded)).min(MAX_UPSCALE.max(scale));
        }
        let pixels = f64::from(w) * f64::from(padded) * f64::from(scale).powi(2);
        if pixels > MAX_INPUT_PIXELS {
            scale *= (MAX_INPUT_PIXELS / pixels).sqrt() as f32;
        }
        let side = |n: f32| ((n / STRIDE as f32).round().max(1.0) as u32) * STRIDE;
        let width_in = side(w * scale);
        let height_in = side(padded * scale);
        let scale_x = width_in as f32 / w;
        let scale_y = height_in as f32 / padded;
        Layout {
            width: width_in,
            height: height_in,
            top: (pad * scale_y).round() as u32,
            scale_x,
            scale_y,
            image_width: width,
            image_height: height,
        }
    }

    /// An input pixel's position in the image.
    fn image_point(&self, [x, y]: [f32; 2]) -> [f32; 2] {
        [x / self.scale_x, (y - self.top as f32) / self.scale_y]
    }
}

/// The detection input: the image scaled into `layout` on its background,
/// as BGR planes normalized to [-1, 1], the order and range the model was
/// trained on.
pub(super) fn input(image: &RgbImage, layout: &Layout) -> Vec<f32> {
    let (w, h) = (layout.width as usize, layout.height as usize);
    let plane = w * h;
    let background = background(image);
    let mut data = vec![0.0f32; plane * 3];
    let fill = background.map(normalize);
    for (channel, value) in fill.iter().enumerate() {
        data[(2 - channel) * plane..(3 - channel) * plane].fill(*value);
    }
    let inner_height = ((layout.image_height as f32 * layout.scale_y).round() as u32)
        .clamp(1, layout.height.saturating_sub(layout.top).max(1));
    let scaled = super::resize(image, layout.width, inner_height);
    for y in 0..inner_height as usize {
        let row = (y + layout.top as usize) * w;
        if y + layout.top as usize >= h {
            break;
        }
        for x in 0..w {
            let pixel = scaled.get_pixel(x as u32, y as u32).0;
            for channel in 0..3 {
                data[(2 - channel) * plane + row + x] = normalize(pixel[channel]);
            }
        }
    }
    data
}

fn normalize(value: u8) -> f32 {
    f32::from(value) / 127.5 - 1.0
}

/// The image's background: the median of its border pixels, per channel.
pub(super) fn background(image: &RgbImage) -> [u8; 3] {
    let (w, h) = image.dimensions();
    if w == 0 || h == 0 {
        return [255; 3];
    }
    let mut channels: [Vec<u8>; 3] = Default::default();
    let mut add = |x: u32, y: u32| {
        let pixel = image.get_pixel(x, y).0;
        for channel in 0..3 {
            channels[channel].push(pixel[channel]);
        }
    };
    let step = |n: u32| (n / 256).max(1) as usize;
    for x in (0..w).step_by(step(w)) {
        add(x, 0);
        add(x, h - 1);
    }
    for y in (0..h).step_by(step(h)) {
        add(0, y);
        add(w - 1, y);
    }
    channels.map(|mut values| {
        values.sort_unstable();
        values[values.len() / 2]
    })
}

/// A detected text region: its corners in image pixels, clockwise from the
/// top-left as the text would be read if it runs along the longer side.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Region {
    pub corners: [[f32; 2]; 4],
    pub score: f32,
}

/// The text regions of a probability map (`width` by `height`, row-major)
/// for an image laid out as `layout`: the first [`MAX_REGIONS`] from the top,
/// and whether there were more.
pub(super) fn regions(map: &[f32], layout: &Layout) -> Result<(Vec<Region>, bool)> {
    let (width, height) = (layout.width as usize, layout.height as usize);
    if map.len() != width * height {
        return Err(failure("detection model returned a map of the wrong size"));
    }
    if map.iter().any(|p| !p.is_finite()) {
        return Err(failure("detection model returned invalid probabilities"));
    }
    let mask = dilate(&threshold(map), width, height);
    let mut found = Vec::new();
    for points in components(&mask, width, height) {
        let Some(rect) = Rect::enclosing(&points) else {
            continue;
        };
        if rect.short_side() < MIN_SIDE {
            continue;
        }
        let score = mean_inside(map, width, height, &rect.corners());
        if score < BOX_THRESHOLD {
            continue;
        }
        let grown = rect.grown(rect.area() * UNCLIP_RATIO / rect.perimeter());
        if grown.short_side() < MIN_SIDE + 2.0 {
            continue;
        }
        let corners = grown.corners().map(|point| {
            let [x, y] = layout.image_point(point);
            [
                x.round()
                    .clamp(0.0, layout.image_width.saturating_sub(1) as f32),
                y.round()
                    .clamp(0.0, layout.image_height.saturating_sub(1) as f32),
            ]
        });
        let corners = clockwise(corners);
        let side =
            |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        if side(corners[0], corners[1]) <= 3.0 || side(corners[0], corners[3]) <= 3.0 {
            continue;
        }
        if found.len() == MAX_REGIONS {
            return Ok((found, true));
        }
        found.push(Region { corners, score });
    }
    Ok((found, false))
}

fn threshold(map: &[f32]) -> Vec<bool> {
    map.iter().map(|p| *p > THRESHOLD).collect()
}

/// The mask grown by one pixel right and down (a 2 by 2 dilation), which
/// joins characters a hair apart.
fn dilate(mask: &[bool], width: usize, height: usize) -> Vec<bool> {
    let mut grown = mask.to_vec();
    for y in 0..height {
        for x in 0..width {
            if mask[y * width + x] {
                continue;
            }
            let left = x > 0 && mask[y * width + x - 1];
            let up = y > 0 && mask[(y - 1) * width + x];
            let diagonal = x > 0 && y > 0 && mask[(y - 1) * width + x - 1];
            grown[y * width + x] = left || up || diagonal;
        }
    }
    grown
}

/// The 8-connected regions of a mask, each as the end pixels of its runs
/// (the pixels its convex hull can touch), in order of first appearance.
fn components(mask: &[bool], width: usize, height: usize) -> Vec<Vec<[f32; 2]>> {
    // Runs of each row: (row, start, end inclusive, label).
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    let mut row_start = Vec::with_capacity(height + 1);
    for y in 0..height {
        row_start.push(runs.len());
        let row = &mask[y * width..(y + 1) * width];
        let mut x = 0;
        while x < width {
            if !row[x] {
                x += 1;
                continue;
            }
            let start = x;
            while x < width && row[x] {
                x += 1;
            }
            runs.push((y, start, x - 1));
        }
    }
    row_start.push(runs.len());
    let mut parent: Vec<usize> = (0..runs.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for y in 1..height {
        let (mut above, above_end) = (row_start[y - 1], row_start[y]);
        for current in row_start[y]..row_start[y + 1] {
            let (_, start, end) = runs[current];
            // Runs above that touch this one, diagonals included.
            while above < above_end && runs[above].2 + 1 < start {
                above += 1;
            }
            let mut probe = above;
            while probe < above_end && runs[probe].1 <= end + 1 {
                let (a, b) = (root(&mut parent, probe), root(&mut parent, current));
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
                probe += 1;
            }
        }
    }
    let mut index_of = vec![usize::MAX; runs.len()];
    let mut groups: Vec<Vec<[f32; 2]>> = Vec::new();
    for (i, &(y, start, end)) in runs.iter().enumerate() {
        let r = root(&mut parent, i);
        if index_of[r] == usize::MAX {
            index_of[r] = groups.len();
            groups.push(Vec::new());
        }

        let group = &mut groups[index_of[r]];
        group.push([start as f32, y as f32]);
        if end != start {
            group.push([end as f32, y as f32]);
        }
    }
    groups
}

/// A rotated rectangle: its center, its two half extents, and the unit
/// direction of its first side.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    center: [f32; 2],
    half: [f32; 2],
    axis: [f32; 2],
}

impl Rect {
    /// The smallest rectangle around `points` (rotating calipers over their
    /// convex hull).
    fn enclosing(points: &[[f32; 2]]) -> Option<Rect> {
        let hull = hull(points);
        match hull.len() {
            0 => return None,
            1 => {
                return Some(Rect {
                    center: hull[0],
                    half: [0.0, 0.0],
                    axis: [1.0, 0.0],
                });
            }
            _ => {}
        }
        let mut best: Option<(f32, Rect)> = None;
        for i in 0..hull.len() {
            let (a, b) = (hull[i], hull[(i + 1) % hull.len()]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let length = (dx * dx + dy * dy).sqrt();
            if length == 0.0 {
                continue;
            }
            let axis = [dx / length, dy / length];
            let normal = [-axis[1], axis[0]];
            let (mut u0, mut u1, mut v0, mut v1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
            for p in &hull {
                let u = p[0] * axis[0] + p[1] * axis[1];
                let v = p[0] * normal[0] + p[1] * normal[1];
                u0 = u0.min(u);
                u1 = u1.max(u);
                v0 = v0.min(v);
                v1 = v1.max(v);
            }
            let area = (u1 - u0) * (v1 - v0);
            if best.as_ref().is_none_or(|(smallest, _)| area < *smallest) {
                let (u, v) = ((u0 + u1) / 2.0, (v0 + v1) / 2.0);
                best = Some((
                    area,
                    Rect {
                        center: [u * axis[0] + v * normal[0], u * axis[1] + v * normal[1]],
                        half: [(u1 - u0) / 2.0, (v1 - v0) / 2.0],
                        axis,
                    },
                ));
            }
        }
        best.map(|(_, rect)| rect)
    }

    fn short_side(&self) -> f32 {
        2.0 * self.half[0].min(self.half[1])
    }

    fn area(&self) -> f32 {
        4.0 * self.half[0] * self.half[1]
    }

    fn perimeter(&self) -> f32 {
        4.0 * (self.half[0] + self.half[1])
    }

    /// The rectangle with every side moved out by `distance`: the smallest
    /// rectangle around the region offset by that distance.
    fn grown(&self, distance: f32) -> Rect {
        let distance = if distance.is_finite() { distance } else { 0.0 };
        Rect {
            half: [self.half[0] + distance, self.half[1] + distance],
            ..*self
        }
    }

    fn corners(&self) -> [[f32; 2]; 4] {
        let [cx, cy] = self.center;
        let u = [self.axis[0] * self.half[0], self.axis[1] * self.half[0]];
        let v = [-self.axis[1] * self.half[1], self.axis[0] * self.half[1]];
        [
            [cx - u[0] - v[0], cy - u[1] - v[1]],
            [cx + u[0] - v[0], cy + u[1] - v[1]],
            [cx + u[0] + v[0], cy + u[1] + v[1]],
            [cx - u[0] + v[0], cy - u[1] + v[1]],
        ]
    }
}

/// The convex hull of `points`, counter-clockwise in image coordinates
/// (Andrew's monotone chain), without collinear points.
fn hull(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    sorted.dedup();
    if sorted.len() < 3 {
        return sorted;
    }
    let cross = |o: [f32; 2], a: [f32; 2], b: [f32; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut hull: Vec<[f32; 2]> = Vec::with_capacity(sorted.len() * 2);
    for pass in 0..2 {
        let start = hull.len();
        let ordered: Box<dyn Iterator<Item = &[f32; 2]>> = if pass == 0 {
            Box::new(sorted.iter())
        } else {
            Box::new(sorted.iter().rev())
        };
        for &p in ordered {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull
}

/// The mean probability of the map's pixels inside the convex polygon
/// `corners` (pixel centers on or inside its edges).
fn mean_inside(map: &[f32], width: usize, height: usize, corners: &[[f32; 2]; 4]) -> f32 {
    let x0 = corners
        .iter()
        .map(|p| p[0])
        .fold(f32::MAX, f32::min)
        .floor()
        .max(0.0) as usize;
    let x1 = (corners.iter().map(|p| p[0]).fold(f32::MIN, f32::max).ceil() as usize).min(width - 1);
    let y0 = corners
        .iter()
        .map(|p| p[1])
        .fold(f32::MAX, f32::min)
        .floor()
        .max(0.0) as usize;
    let y1 =
        (corners.iter().map(|p| p[1]).fold(f32::MIN, f32::max).ceil() as usize).min(height - 1);
    // The sign each edge's cross product takes for points inside.
    let area: f32 = (0..4)
        .map(|i| {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            a[0] * b[1] - a[1] * b[0]
        })
        .sum();
    let orientation = if area >= 0.0 { 1.0 } else { -1.0 };
    let inside = |x: f32, y: f32| {
        (0..4).all(|i| {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            orientation * ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])) >= -0.5
        })
    };
    let (mut sum, mut count) = (0.0f64, 0usize);
    if x0 > x1 || y0 > y1 {
        return 0.0;
    }
    for y in y0..=y1 {
        for x in x0..=x1 {
            if inside(x as f32, y as f32) {
                sum += f64::from(map[y * width + x]);
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

/// Corners ordered top-left, top-right, bottom-right, bottom-left: the two
/// leftmost are the left side, each side ordered top to bottom.
fn clockwise(corners: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
    let mut by_x = corners;
    by_x.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let (mut left, mut right) = ([by_x[0], by_x[1]], [by_x[2], by_x[3]]);
    left.sort_by(|a, b| a[1].total_cmp(&b[1]));
    right.sort_by(|a, b| a[1].total_cmp(&b[1]));
    [left[0], right[0], right[1], left[1]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_cap_the_long_side_enlarge_only_small_images_and_keep_the_stride() {
        // A screenshot is read as it is; A4 pages at 150 and 300 DPI are
        // scaled to a 1600-pixel long side; a small image is enlarged 1.5
        // times.
        let screen = Layout::of(1440, 900);
        assert_eq!((screen.width, screen.height, screen.top), (1440, 896, 0));
        let page = Layout::of(1240, 1754);
        assert_eq!((page.width, page.height), (1120, 1600));
        let large = Layout::of(2480, 3508);
        assert_eq!((large.width, large.height), (1120, 1600));
        let small = Layout::of(600, 400);
        assert_eq!((small.width, small.height), (608, 416));
        let tiny = Layout::of(320, 68);
        assert_eq!((tiny.width, tiny.height, tiny.scale_x), (480, 96, 1.5));
        for layout in [screen, page, large, small] {
            assert_eq!(layout.width % STRIDE, 0);
            assert_eq!(layout.height % STRIDE, 0);
        }
        // A strip is padded above and below, centered.
        let strip = Layout::of(1600, 100);
        assert!(strip.top > 0);
        let back = strip.image_point([0.0, strip.top as f32]);
        assert!(back[1].abs() < 1.0, "{back:?}");
        assert_eq!(
            strip.image_point([strip.width as f32, 0.0])[0].round(),
            1600.0
        );
        // Huge inputs stay within the pixel budget.
        let wide = Layout::of(30_000, 1_000);
        assert!(f64::from(wide.width) * f64::from(wide.height) <= MAX_INPUT_PIXELS * 1.1);
    }

    #[test]
    fn the_input_is_bgr_normalized_with_a_padded_background() {
        let image = RgbImage::from_fn(64, 4, |x, _| {
            if x < 32 {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([255, 255, 255])
            }
        });
        let layout = Layout::of(64, 4);
        let data = input(&image, &layout);
        let plane = (layout.width * layout.height) as usize;
        assert_eq!(data.len(), plane * 3);
        // Padding above is the border's median (white), in every plane.
        assert_eq!(data[0], 1.0);
        assert_eq!(data[plane], 1.0);
        // Red is the last plane: blue, green, red.
        let row = (layout.top + 1) as usize * layout.width as usize;
        assert!((data[2 * plane + row] - 1.0).abs() < 0.02);
        assert!((data[row] + 1.0).abs() < 0.02, "{}", data[row]);
    }

    /// A map of `width` by `height` with the given rectangles at `p`.
    fn map(width: usize, height: usize, boxes: &[([usize; 4], f32)]) -> Vec<f32> {
        let mut map = vec![0.0; width * height];
        for ([x0, y0, x1, y1], p) in boxes {
            for y in *y0..*y1 {
                for x in *x0..*x1 {
                    map[y * width + x] = *p;
                }
            }
        }
        map
    }

    fn layout(width: u32, height: u32) -> Layout {
        Layout {
            width,
            height,
            top: 0,
            scale_x: 1.0,
            scale_y: 1.0,
            image_width: width,
            image_height: height,
        }
    }

    #[test]
    fn a_region_of_text_becomes_a_grown_rectangle_in_reading_order() {
        let found = regions(&map(128, 64, &[([10, 20, 90, 30], 0.9)]), &layout(128, 64))
            .unwrap()
            .0;
        assert_eq!(found.len(), 1);
        let [tl, tr, br, bl] = found[0].corners;
        // The mask spans x 10..=90 and y 20..=30 after dilation; the
        // rectangle grows by area * 1.6 / perimeter = 7.1 pixels.
        assert_eq!(tl, [3.0, 13.0]);
        assert_eq!(br, [97.0, 37.0]);
        assert!(tr[0] > tl[0] && bl[1] > tl[1]);
        // Its mean includes the dilated edge: 720 / 891.
        assert!((found[0].score - 0.808).abs() < 0.01, "{}", found[0].score);
    }

    #[test]
    fn specks_do_not_count_toward_the_region_limit() {
        // More specks above a line of text than regions are read: one-pixel
        // dots four pixels apart, in rows down to y = 68.
        let mut boxes = Vec::new();
        for y in (0..=68).step_by(4) {
            for x in (0..256).step_by(4) {
                boxes.push(([x, y, x + 1, y + 1], 0.95));
            }
        }
        assert!(boxes.len() > MAX_REGIONS);
        boxes.push(([10, 90, 200, 102], 0.9));
        let (found, capped) = regions(&map(256, 128, &boxes), &layout(256, 128)).unwrap();
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].corners[0][1] > 80.0);
        assert!(!capped);
    }

    #[test]
    fn regions_beyond_the_limit_are_reported() {
        // 32 by 32 blocks of text, more than are read.
        let mut boxes = Vec::new();
        for y in (0..256).step_by(8) {
            for x in (0..256).step_by(8) {
                boxes.push(([x, y, x + 5, y + 5], 0.9));
            }
        }
        let (found, capped) = regions(&map(256, 256, &boxes), &layout(256, 256)).unwrap();
        assert_eq!(found.len(), MAX_REGIONS);
        assert!(capped);
        let (found, capped) =
            regions(&map(256, 256, &boxes[..MAX_REGIONS]), &layout(256, 256)).unwrap();
        assert_eq!(found.len(), MAX_REGIONS);
        assert!(!capped);
    }

    #[test]
    fn faint_specks_and_thin_regions_are_not_text() {
        // Below the box threshold, too thin, and a dot.
        let found = regions(
            &map(
                128,
                64,
                &[
                    ([10, 10, 60, 20], 0.4),
                    ([10, 40, 100, 41], 0.9),
                    ([110, 50, 111, 51], 0.95),
                ],
            ),
            &layout(128, 64),
        )
        .unwrap()
        .0;
        assert!(found.is_empty(), "{found:?}");
        assert!(regions(&[0.0; 10], &layout(128, 64)).is_err());
        let mut nan = map(32, 32, &[]);
        nan[3] = f32::NAN;
        assert!(regions(&nan, &layout(32, 32)).is_err());
    }

    #[test]
    fn separate_lines_are_separate_regions_and_touching_runs_join() {
        let found = regions(
            &map(
                256,
                96,
                &[
                    ([10, 10, 100, 22], 0.9),
                    ([110, 10, 200, 22], 0.9),
                    ([10, 50, 200, 62], 0.9),
                ],
            ),
            &layout(256, 96),
        )
        .unwrap()
        .0;
        assert_eq!(found.len(), 3);
        // Diagonal neighbours belong to one region.
        let mut mask = vec![false; 16];
        mask[0] = true;
        mask[5] = true;
        mask[10] = true;
        assert_eq!(components(&mask, 4, 4).len(), 1);
        mask[3] = true;
        assert_eq!(components(&mask, 4, 4).len(), 2);
    }

    #[test]
    fn a_tilted_region_keeps_its_angle() {
        // A thick diagonal band.
        let (width, height) = (200usize, 200usize);
        let mut probability = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 40..160 {
                if (y as f32 - 0.25 * x as f32 - 60.0).abs() < 6.0 {
                    probability[y * width + x] = 0.9;
                }
            }
        }
        let found = regions(&probability, &layout(200, 200)).unwrap().0;
        assert_eq!(found.len(), 1);
        let [tl, tr, _, _] = found[0].corners;
        let slope = (tr[1] - tl[1]) / (tr[0] - tl[0]);
        assert!((slope - 0.25).abs() < 0.05, "{slope}");
    }

    #[test]
    fn rectangles_and_hulls_of_simple_point_sets() {
        let square = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [2.0, 2.0]];
        assert_eq!(hull(&square).len(), 4);
        let rect = Rect::enclosing(&square).unwrap();
        assert!((rect.area() - 16.0).abs() < 1e-4);
        assert!((rect.short_side() - 4.0).abs() < 1e-4);
        let line = [[0.0, 0.0], [10.0, 0.0]];
        assert_eq!(Rect::enclosing(&line).unwrap().short_side(), 0.0);
        assert!(Rect::enclosing(&[]).is_none());
        assert_eq!(
            clockwise([[10.0, 0.0], [0.0, 5.0], [0.0, 0.0], [10.0, 5.0]]),
            [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]
        );
    }
}
