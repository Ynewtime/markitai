//! `TSD` — everything placed on a page, a sheet or a slide.
//!
//! A drawable is an image, a shape, a text box, a line, a movie, a group, a
//! table or a chart: anything with a position and a size. All three apps write
//! the same archives — `TSD` type ids live in the common registry, not in a
//! per-app one — so a shape made in Keynote and a shape imported into Pages are
//! the same object.
//!
//! Three things about the wire format decide how everything here is written.
//!
//! **Inheritance is nesting, and it goes deep.** `super` is always field 1 and
//! always a submessage, so a Keynote title placeholder is
//! `KN.PlaceholderArchive { 1: TSWP.ShapeInfoArchive { 1: TSD.ShapeArchive {
//! 1: TSD.DrawableArchive { 1: geometry … } } } }` — four levels, every one of
//! them at field 1. Nothing here assumes a depth: [`drawable_path`] walks
//! field 1 until it finds the message whose field 1 is a geometry, and every
//! read and write goes through the path it returns.
//!
//! **A masked image is two objects.** `TSD.ImageArchive` carries the picture's
//! own rectangle; `TSD.MaskArchive` carries the window it is seen through, and
//! the mask's geometry is in the *image's* coordinate space, because the mask's
//! parent is the image. What the app calls the object's position and size is
//! therefore `image.position + mask.position` and `mask.size` — verified
//! against Pages, which reports 60 × 123, 475 × 383 for an image whose archive
//! says 33.86 × 66.28, 511.86 × 466.13 and whose mask says 25.89 × 56.52,
//! 475 × 383. [`Drawable::frame`] is that composition.
//!
//! **A container holds its drawables, and one of the three does not.** A
//! Keynote slide and a Numbers sheet each list what is on them and are named
//! as each drawable's `parent`; a Pages page lists them in a page group of
//! `TP.FloatingDrawablesArchive` and is named by nothing, so a drawable
//! floating on a page has **no parent at all**. [`add_shape`], [`add_image`]
//! and the table writer all resolve the container once and let it decide, for
//! exactly that reason.
//!
//! **The style is a separate object, and it inherits.** Opacity, fill, stroke,
//! shadow and reflection live in a `TSD.ShapeStyleArchive` (or
//! `TSD.MediaStyleArchive`, whose property numbering is *different*), usually
//! wrapped in a `TSWP.ShapeStyleArchive`, and a property the style does not
//! carry comes from its parent. Setting a shape's opacity in Keynote makes the
//! app write a new variation style whose only properties are the ones that
//! changed — observed, and the reason [`crate::Document::object_style`] walks the
//! chain rather than reading one object.

use std::collections::BTreeMap;

use crate::pb::{decode_nested, Message, Value};
use crate::style::{get_path, reference_at};

/// `TSD.DrawableArchive` — the abstract base. Never a payload on its own here.
pub const TYPE_DRAWABLE: u32 = 3002;
/// `TSD.ContainerArchive` — geometry, parent and children, but *not* a drawable.
pub const TYPE_CONTAINER: u32 = 3003;
/// `TSD.ShapeArchive`.
pub const TYPE_SHAPE: u32 = 3004;
/// `TSD.ImageArchive`.
pub const TYPE_IMAGE: u32 = 3005;
/// `TSD.MaskArchive` — the window a masked image is seen through.
pub const TYPE_MASK: u32 = 3006;
/// `TSD.MovieArchive` — video and audio.
pub const TYPE_MOVIE: u32 = 3007;
/// `TSD.GroupArchive`.
pub const TYPE_GROUP: u32 = 3008;
/// `TSD.ConnectionLineArchive`.
pub const TYPE_CONNECTION_LINE: u32 = 3009;
/// `TSD.ShapeStyleArchive`.
pub const TYPE_SHAPE_STYLE: u32 = 3015;
/// `TSD.MediaStyleArchive` — images and movies. Different field numbers.
pub const TYPE_MEDIA_STYLE: u32 = 3016;
/// `TSD.PencilAnnotationArchive`.
pub const TYPE_PENCIL_ANNOTATION: u32 = 3086;
/// `TSD.PencilAnnotationStorageArchive` — note the id is not in the 3000s.
pub const TYPE_PENCIL_STORAGE: u32 = 242;
/// `TSD.StandinCaptionArchive` — an empty message meaning "a caption could go
/// here". Every drawable in a Keynote theme has two.
pub const TYPE_STANDIN_CAPTION: u32 = 3097;
/// `TSWP.DrawableAttachmentArchive` — a drawable anchored in text.
pub const TYPE_DRAWABLE_ATTACHMENT: u32 = 2003;
/// `TSWP.ShapeInfoArchive` — the real class of most shapes and text boxes.
pub const TYPE_SHAPE_INFO: u32 = 2011;
/// `TSWP.CommentInfoArchive`.
pub const TYPE_COMMENT_INFO: u32 = 2014;
/// `TSWP.EquationInfoArchive`.
pub const TYPE_EQUATION_INFO: u32 = 2015;
/// `TSWP.ShapeStyleArchive` — wraps `TSD.ShapeStyleArchive` at field 1.
pub const TYPE_WP_SHAPE_STYLE: u32 = 2025;
/// `TSWP.TOCInfoArchive`.
pub const TYPE_TOC_INFO: u32 = 2240;
/// `TSA.CaptionInfoArchive`.
pub const TYPE_CAPTION_INFO: u32 = 633;
/// `TST.TableInfoArchive` — a table is a drawable.
pub const TYPE_TABLE_INFO: u32 = 6000;
/// `TST.WPTableInfoArchive`.
pub const TYPE_WP_TABLE_INFO: u32 = 6007;
/// `TSCH.PreUFF.ChartInfoArchive`.
pub const TYPE_CHART_INFO: u32 = 5000;
/// `TSCH.ChartDrawableArchive` — a chart is a drawable.
pub const TYPE_CHART_DRAWABLE: u32 = 5021;
/// `KN.PlaceholderArchive` / `TN.PlaceholderArchive` / `TP.PlaceholderArchive`.
pub const TYPE_PLACEHOLDER: u32 = 7;
/// `KN.PlaceholderArchive`, registered twice.
pub const TYPE_PLACEHOLDER_ALIAS: u32 = 12;

/// Field numbers of `TSD.DrawableArchive`, once [`drawable_path`] has found it.
pub mod field {
    pub const GEOMETRY: u32 = 1;
    pub const PARENT: u32 = 2;
    pub const TEXT_WRAP: u32 = 3;
    pub const HYPERLINK: u32 = 4;
    pub const LOCKED: u32 = 5;
    pub const COMMENT: u32 = 6;
    pub const ASPECT_RATIO_LOCKED: u32 = 7;
    pub const DESCRIPTION: u32 = 8;
    pub const PENCIL_ANNOTATIONS: u32 = 9;
    pub const TITLE: u32 = 10;
    pub const CAPTION: u32 = 11;
    pub const TITLE_HIDDEN: u32 = 12;
    pub const CAPTION_HIDDEN: u32 = 13;
}

/// Field numbers of `TSD.ImageArchive`, relative to the archive itself.
pub mod image_field {
    pub const STYLE: u32 = 3;
    pub const ORIGINAL_SIZE: u32 = 4;
    pub const MASK: u32 = 5;
    pub const FLAGS: u32 = 7;
    pub const NATURAL_SIZE: u32 = 9;
    pub const INSTANT_ALPHA_PATH: u32 = 10;
    pub const DATA: u32 = 11;
    pub const THUMBNAIL_DATA: u32 = 12;
    pub const ORIGINAL_DATA: u32 = 13;
    pub const ADJUSTMENTS: u32 = 14;
    pub const ADJUSTED_DATA: u32 = 15;
    pub const THUMBNAIL_ADJUSTED_DATA: u32 = 16;
    pub const ENHANCED_DATA: u32 = 17;
    pub const UNTAGGED_IS_GENERIC: u32 = 18;
    pub const TRACED_PATH: u32 = 19;
    pub const ATTRIBUTION: u32 = 20;
    pub const BACKGROUND_REMOVED: u32 = 22;
    pub const ORIGINAL_SVG_DATA: u32 = 23;
}

/// `TSD.ImageArchive.flags` / `TSD.MovieArchive.flags`, from the two states
/// `TSD.MediaFlagsCommandArchive` names.
///
/// Both bits were read off documents: a placeholder image in a Keynote theme
/// carries 1, and an image whose bytes the app replaced comes back carrying 2.
pub mod media_flag {
    pub const PLACEHOLDER: u32 = 1;
    pub const WAS_REPLACED: u32 = 2;
}

/// What kind of thing a drawable is, as far as its message type says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Shape,
    Image,
    Mask,
    Movie,
    Group,
    ConnectionLine,
    Table,
    Chart,
    Placeholder,
    Comment,
    Equation,
    Caption,
    Contents,
    Other,
}

impl Kind {
    pub fn of(message_type: u32) -> Option<Kind> {
        Some(match message_type {
            TYPE_SHAPE | TYPE_SHAPE_INFO => Kind::Shape,
            TYPE_IMAGE => Kind::Image,
            TYPE_MASK => Kind::Mask,
            TYPE_MOVIE => Kind::Movie,
            TYPE_GROUP => Kind::Group,
            TYPE_CONNECTION_LINE => Kind::ConnectionLine,
            TYPE_TABLE_INFO | TYPE_WP_TABLE_INFO => Kind::Table,
            TYPE_CHART_INFO | TYPE_CHART_DRAWABLE => Kind::Chart,
            TYPE_PLACEHOLDER | TYPE_PLACEHOLDER_ALIAS => Kind::Placeholder,
            TYPE_COMMENT_INFO => Kind::Comment,
            TYPE_EQUATION_INFO => Kind::Equation,
            TYPE_CAPTION_INFO => Kind::Caption,
            TYPE_TOC_INFO => Kind::Contents,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Shape => "shape",
            Kind::Image => "image",
            Kind::Mask => "mask",
            Kind::Movie => "movie",
            Kind::Group => "group",
            Kind::ConnectionLine => "connection line",
            Kind::Table => "table",
            Kind::Chart => "chart",
            Kind::Placeholder => "placeholder",
            Kind::Comment => "comment",
            Kind::Equation => "equation",
            Kind::Caption => "caption",
            Kind::Contents => "table of contents",
            Kind::Other => "drawable",
        }
    }

    /// Media drawables keep an `originalSize` beside their geometry, and the
    /// app maintains the two together.
    pub fn is_media(self) -> bool {
        matches!(self, Kind::Image | Kind::Movie)
    }
}

/// `TSD.GeometryArchive` — where a drawable sits and how big it is.
///
/// Position is the top-left corner **in the parent's coordinate space**, in
/// points, y growing downward. Size is the untransformed size: rotation is not
/// baked into it. `angle` is degrees counter-clockwise about the centre —
/// confirmed by drawing a line from (100, 600) to (500, 700), which Keynote
/// stored as a 412.31-point wide geometry at 345.96°.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    /// Bit field. 3 on an ordinary object, 1 on a shape that sizes itself to
    /// its text, 0 on a zero-size one. The individual bits are not decoded here
    /// — nothing in the corpus separates them — and are carried through.
    pub flags: u32,
    pub angle: f32,
}

impl Geometry {
    pub fn decode(message: &Message) -> Geometry {
        let point = |field: u32| -> (f32, f32) {
            message
                .bytes(field)
                .and_then(decode_nested)
                .map(|p| (float(&p, 1), float(&p, 2)))
                .unwrap_or((0.0, 0.0))
        };
        let (x, y) = point(1);
        let (width, height) = point(2);
        Geometry {
            x,
            y,
            width,
            height,
            flags: message.varint(3).unwrap_or(0) as u32,
            angle: match message.get(4) {
                Some(Value::Fixed32(b)) => f32::from_le_bytes(*b),
                _ => 0.0,
            },
        }
    }

    /// Write position and size back into a geometry message, leaving `flags`,
    /// `angle` and anything else it carries exactly as they were.
    ///
    /// Fields are set in place rather than rebuilt, because a geometry this
    /// crate composed from scratch would hold only the fields it knows about —
    /// and the whole point of the wire-level rule is that it never has to know
    /// them all.
    /// Does this shape size itself to its text?
    ///
    /// Flag 1 with a zero height is what Keynote wrote for a text box, and the
    /// app then reported a height of 115 points and a position 58 above the
    /// stored one — half the height it had laid out. Anything reading such an
    /// object's rectangle is reading an anchor, not a box.
    pub fn fits_its_text(&self) -> bool {
        self.flags & 1 != 0 && self.height == 0.0
    }

    pub fn write_into(&self, message: &mut Message) {
        for (field, (a, b)) in [(1, (self.x, self.y)), (2, (self.width, self.height))] {
            let mut point = message
                .bytes(field)
                .and_then(decode_nested)
                .unwrap_or_default();
            point.set_in_order(1, Value::Fixed32(a.to_le_bytes()));
            point.set_in_order(2, Value::Fixed32(b.to_le_bytes()));
            message.set_in_order(field, Value::Bytes(point.encode()));
        }
    }
}

/// Extent of a rectangle rotated about its own centre.
///
/// `|w·cosθ| + |h·sinθ|` by `|w·sinθ| + |h·cosθ|` — the standard bounding box,
/// and what the apps report an object's position against.
pub fn rotated_extent(width: f32, height: f32, degrees: f32) -> (f32, f32) {
    if degrees == 0.0 {
        return (width, height);
    }
    let radians = degrees.to_radians();
    let (sin, cos) = (radians.sin().abs(), radians.cos().abs());
    (width * cos + height * sin, width * sin + height * cos)
}

/// A rectangle in the coordinate space of a drawable's parent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Which of the six generators draws a shape's outline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathSourceKind {
    /// Arrows, stars, plus signs — one control point.
    Point,
    /// Rounded rectangle, polygon, chevron — one scalar.
    Scalar,
    /// A baked path.
    Bezier,
    /// A speech bubble.
    Callout,
    /// A connector between two objects.
    ConnectionLine,
    /// A user-editable node list.
    EditableBezier,
}

/// A shape's path source: which generator, and how big the path it generates is
/// before it is scaled into the geometry.
#[derive(Debug, Clone)]
pub struct PathSource {
    pub kind: PathSourceKind,
    pub natural_size: Option<(f32, f32)>,
    /// Elements of the baked path, when there is one. A line drawn by Keynote
    /// is a two-element bezier; a rectangle is six, because iWork writes a
    /// redundant `moveTo(0, 0)` after the closing element and both public
    /// writers reproduce it deliberately.
    pub elements: usize,
    pub horizontal_flip: bool,
    pub vertical_flip: bool,
}

impl PathSource {
    fn decode(message: &Message) -> Option<PathSource> {
        let (kind, field) = [
            (PathSourceKind::Point, 3u32),
            (PathSourceKind::Scalar, 4),
            (PathSourceKind::Bezier, 5),
            (PathSourceKind::Callout, 6),
            (PathSourceKind::ConnectionLine, 7),
            (PathSourceKind::EditableBezier, 8),
        ]
        .into_iter()
        .find(|(_, field)| message.get(*field).is_some())?;
        let body = message.bytes(field).and_then(decode_nested)?;
        // Natural size is field 3 for the point and scalar sources, 2 for the
        // bezier and editable-bezier ones, 1 for a callout.
        let natural = [3u32, 2, 1]
            .into_iter()
            .find_map(|f| body.bytes(f).and_then(decode_nested).filter(is_point))
            .map(|p| (float(&p, 1), float(&p, 2)));
        let elements = body
            .bytes(3)
            .and_then(decode_nested)
            .map(|path| path.all(1).count())
            .unwrap_or(0);
        Some(PathSource {
            kind,
            natural_size: natural,
            elements,
            horizontal_flip: message.varint(1).unwrap_or(0) != 0,
            vertical_flip: message.varint(2).unwrap_or(0) != 0,
        })
    }

    /// A path of two elements is a straight line — what the app calls a line
    /// object. Inferred from the shape of what Keynote wrote for `make new
    /// line`, not from anything that says so.
    pub fn looks_like_a_line(&self) -> bool {
        self.kind == PathSourceKind::Bezier && self.elements == 2
    }
}

/// `TSP.Color` — channels are 0.0–1.0, not 0–255.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub red: f32,
    pub green: f32,
    pub blue: f32,
    /// Alpha defaults to 1 when absent, and iWork's own writers always emit it.
    pub alpha: f32,
}

impl Color {
    /// Read the channels of a `TSP.Color`, if that is what this message is.
    pub fn decode(message: &Message) -> Option<Color> {
        if !crate::style::is_color(message) {
            return None;
        }
        Some(Color {
            red: float(message, 3),
            green: float(message, 4),
            blue: float(message, 5),
            alpha: match message.get(6) {
                Some(Value::Fixed32(b)) => f32::from_le_bytes(*b),
                _ => 1.0,
            },
        })
    }
}

impl std::fmt::Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let byte = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
        write!(
            f,
            "#{:02x}{:02x}{:02x}",
            byte(self.red),
            byte(self.green),
            byte(self.blue)
        )?;
        if self.alpha < 1.0 {
            write!(f, " {:.0}%", self.alpha * 100.0)?;
        }
        Ok(())
    }
}

/// `TSD.FillArchive` — a tagged union with exactly one arm set. An empty
/// message is a fill that is there and paints nothing.
#[derive(Debug, Clone)]
pub enum Fill {
    None,
    Color(Color),
    Gradient { stops: usize, angle: Option<f32> },
    Image { data: Option<u64>, technique: u32 },
}

impl Fill {
    pub(crate) fn decode(message: &Message) -> Fill {
        if let Some(colour) = message.bytes(1).and_then(decode_nested) {
            if let Some(colour) = Color::decode(&colour) {
                return Fill::Color(colour);
            }
        }
        if let Some(gradient) = message.bytes(2).and_then(decode_nested) {
            let angle = gradient
                .bytes(5)
                .and_then(decode_nested)
                .map(|a| float(&a, 2));
            return Fill::Gradient {
                stops: gradient.all(2).count(),
                angle,
            };
        }
        if let Some(image) = message.bytes(3).and_then(decode_nested) {
            return Fill::Image {
                data: image.bytes(6).and_then(reference),
                technique: image.varint(2).unwrap_or(0) as u32,
            };
        }
        Fill::None
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Fill::None => "none",
            Fill::Color(_) => "colour",
            Fill::Gradient { .. } => "gradient",
            Fill::Image { .. } => "image",
        }
    }
}

/// How a stroke's dash pattern reads.
///
/// Dashes and dots are the *same* `TSDPattern` type — the difference is that a
/// dotted pattern's first entry is smaller than 1. And `count` is not
/// `pattern.len()`: iWork writes six floats whatever the count says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrokePattern {
    /// A solid line.
    Solid,
    /// No line at all.
    Empty,
    Dashed,
    Dotted,
}

/// `TSD.StrokeArchive`.
#[derive(Debug, Clone)]
pub struct Stroke {
    pub color: Option<Color>,
    pub width: f32,
    pub pattern: StrokePattern,
}

impl Stroke {
    fn decode(message: &Message) -> Stroke {
        let pattern = message
            .bytes(6)
            .and_then(decode_nested)
            .map(|p| {
                let entries: Vec<f32> = p
                    .all(4)
                    .map(|value| match value {
                        Value::Fixed32(b) => f32::from_le_bytes(*b),
                        _ => 0.0,
                    })
                    .collect();
                let count = p.varint(3).unwrap_or(0) as usize;
                match p.varint(1).unwrap_or(0) {
                    1 => StrokePattern::Solid,
                    2 => StrokePattern::Empty,
                    _ if entries.first().is_some_and(|first| *first < 1.0) && count > 0 => {
                        StrokePattern::Dotted
                    }
                    _ if count > 0 => StrokePattern::Dashed,
                    _ => StrokePattern::Solid,
                }
            })
            .unwrap_or(StrokePattern::Solid);
        Stroke {
            color: message
                .bytes(1)
                .and_then(decode_nested)
                .and_then(|c| Color::decode(&c)),
            width: float(message, 2),
            pattern,
        }
    }
}

/// `TSD.ShadowArchive`. Every field has a non-zero default, so a shadow read
/// as a zeroed struct renders differently from the one iWork drew.
#[derive(Debug, Clone)]
pub struct Shadow {
    pub color: Option<Color>,
    pub angle: f32,
    pub offset: f32,
    /// Blur radius, and an integer in the archive rather than a float.
    pub radius: i32,
    pub opacity: f32,
    pub enabled: bool,
}

impl Shadow {
    fn decode(message: &Message) -> Shadow {
        Shadow {
            color: message
                .bytes(1)
                .and_then(decode_nested)
                .and_then(|c| Color::decode(&c)),
            angle: float_or(message, 2, 315.0),
            offset: float_or(message, 3, 5.0),
            radius: message.varint(4).unwrap_or(1) as i32,
            opacity: float_or(message, 5, 1.0),
            // Absent means enabled: the default is true.
            enabled: message.varint(6).unwrap_or(1) != 0,
        }
    }
}

/// A drawable's object style, resolved up the parent chain.
///
/// Property numbering differs between the two style classes and the difference
/// is silent: a shape's properties are `fill` 1, `stroke` 2, `opacity` 3,
/// `shadow` 4, `reflection` 5, while a media style has no fill and numbers them
/// `stroke` 1, `opacity` 2, `shadow` 3, `reflection` 4. Reading a media style
/// with a shape style's numbering reports the stroke as a fill.
#[derive(Debug, Clone, Default)]
pub struct ObjectStyle {
    /// The style object the drawable points at.
    pub identifier: u64,
    /// Its name, when it is a named style rather than a variation.
    pub name: Option<String>,
    /// How many properties the style overrides locally — `TSS`'s own
    /// bookkeeping, and the only thing separating "the named style" from "the
    /// named style plus changes".
    pub override_count: Option<u32>,
    pub fill: Option<Fill>,
    pub stroke: Option<Stroke>,
    /// 0.0–1.0. Keynote wrote 0.5 for an object the script set to 50%.
    pub opacity: Option<f32>,
    pub shadow: Option<Shadow>,
    /// Reflection opacity, 0.0–1.0, defaulting to 0.5 when the archive is
    /// present and empty. Keynote wrote 0.4 for "reflection value 40".
    pub reflection: Option<f32>,
    /// The chain walked to resolve it, nearest first.
    pub inherited_from: Vec<u64>,
}

/// Whether a style archive numbers its properties as a shape or as media.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StyleShape {
    Shape,
    Media,
}

/// Everything sitting between an image's stored pixels and what is drawn.
///
/// This is the state PLAN.md warns about: replace the bytes under it and the
/// document opens, the app reports the same frame, and the picture renders
/// wrong. [`crate::Document::replace_media`] refuses rather than produce that,
/// and this is what it looks at.
#[derive(Debug, Clone, Default)]
pub struct EditState {
    /// The mask object, when there is one.
    pub mask: Option<u64>,
    /// The mask's window, in the image's own coordinate space.
    pub mask_frame: Option<Frame>,
    /// True when the mask really crops — its window is not the whole picture.
    pub crops: bool,
    /// True when the mask's outline is not a plain rectangle: masked with a
    /// shape rather than cropped.
    pub mask_is_a_shape: bool,
    /// An Instant Alpha knockout path.
    pub instant_alpha: bool,
    /// Tone and colour adjustments that are not at their default.
    pub adjustments: Vec<(&'static str, f32)>,
    /// Data references to renderings derived from the original pixels —
    /// `thumbnailImageData`, `originalData`, `adjustedImageData`,
    /// `thumbnailAdjustedImageData`, `enhancedImageData`. Their presence means
    /// the displayed image is not simply the stored one.
    pub derived: Vec<&'static str>,
    /// A traced path that is not the plain rectangle of the picture's natural
    /// size — a real trace of PDF content.
    pub traced_shape: bool,
    pub background_removed: bool,
}

impl EditState {
    /// What a caller must be told before the bytes are swapped. Empty means
    /// nothing stands between the stored pixels and the render.
    pub fn objections(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.crops {
            let frame = self.mask_frame.unwrap_or(Frame {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
            });
            out.push(format!(
                "it is cropped: the mask shows {:.1} × {:.1} at {:.1}, {:.1} of the picture",
                frame.width, frame.height, frame.x, frame.y
            ));
        }
        if self.mask_is_a_shape {
            out.push("it is masked with a shape, not a rectangle".to_string());
        }
        if self.instant_alpha {
            out.push("it carries an Instant Alpha knockout path".to_string());
        }
        if !self.adjustments.is_empty() {
            let list: Vec<String> = self
                .adjustments
                .iter()
                .map(|(name, value)| format!("{name} {value}"))
                .collect();
            out.push(format!("it carries adjustments: {}", list.join(", ")));
        }
        if !self.derived.is_empty() {
            out.push(format!(
                "it keeps renderings derived from the old pixels: {}",
                self.derived.join(", ")
            ));
        }
        if self.traced_shape {
            out.push("it carries a traced outline of the old content".to_string());
        }
        if self.background_removed {
            out.push("its background was removed".to_string());
        }
        out
    }

    pub fn is_clean(&self) -> bool {
        self.objections().is_empty()
    }
}

/// Names of `TSD.ImageAdjustmentsArchive`'s fields, with the value that means
/// "untouched". `top_level` defaults to 1, not 0.
const ADJUSTMENTS: &[(u32, &str, f32)] = &[
    (1, "exposure", 0.0),
    (2, "saturation", 0.0),
    (3, "contrast", 0.0),
    (4, "highlights", 0.0),
    (5, "shadows", 0.0),
    (6, "sharpness", 0.0),
    (7, "denoise", 0.0),
    (8, "temperature", 0.0),
    (9, "tint", 0.0),
    (10, "bottom level", 0.0),
    (11, "top level", 1.0),
    (12, "gamma", 0.0),
];

/// The picture or film a drawable shows.
#[derive(Debug, Clone, Default)]
pub struct Media {
    /// `TSP.DataReference` into the package's media registry — the displayed
    /// bytes.
    pub data: Option<u64>,
    /// Poster frame of a movie.
    pub poster: Option<u64>,
    /// Intrinsic size of the stored picture, in points (which is its pixel size
    /// at 72 dpi).
    pub natural_size: Option<(f32, f32)>,
    /// `originalSize` (field 4). For an **unmasked** image this is the
    /// drawable's own size, which the app rewrites as the object is resized.
    /// For a **masked** image it is not the picture's own size at all — the app
    /// fills it with the mask window (the visible size), and the corpus is not
    /// even self-consistent about that. It is therefore never the basis for the
    /// crop test (see [`EditState`]) and is left untouched when a masked image
    /// is resized.
    pub original_size: Option<(f32, f32)>,
    /// `flags` — see [`media_flag`].
    pub flags: u32,
    /// A movie's trim points, in seconds into the film.
    pub trim: Option<(f32, f32)>,
    /// Which frame of a movie is shown while it is not playing.
    pub poster_time: Option<f32>,
    /// 0.0–1.0.
    pub volume: Option<f32>,
    /// `MovieLoopOption`: 0 none, 1 repeat, 2 back and forth.
    pub loop_option: Option<u32>,
    pub audio_only: bool,
    /// A Keynote live-video source rather than a film. Ground rule 8 material:
    /// read it, carry it, never author it.
    pub live_video: bool,
    /// A movie that streams from a URL and stores no bytes.
    pub remote_url: Option<String>,
}

impl Media {
    pub fn is_placeholder(&self) -> bool {
        self.flags & media_flag::PLACEHOLDER != 0
    }

    pub fn was_replaced(&self) -> bool {
        self.flags & media_flag::WAS_REPLACED != 0
    }
}

/// Where a drawable sits, as the document says rather than as it is drawn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Placement {
    /// A Numbers sheet.
    Sheet(String),
    /// A Keynote slide, by the stream that holds it.
    Slide(String),
    /// A Pages section template — the page furniture a template puts down.
    SectionTemplate(u64),
    /// Pages' floating drawables: on the page, out of the text flow, on the
    /// page the archive files them under.
    Floating {
        page: u32,
    },
    /// Anchored in or inline with text, at a character index.
    InText {
        storage: u64,
        character: u64,
    },
    /// A child of a group.
    Group(u64),
    /// The mask of an image, or an image's caption or title.
    PartOf(u64),
    Unknown,
}

impl Placement {
    pub fn as_str(&self) -> String {
        match self {
            Placement::Sheet(name) => format!("sheet {name}"),
            Placement::Slide(stream) => format!("slide {stream}"),
            Placement::SectionTemplate(owner) => format!("section template {owner}"),
            Placement::Floating { page } => format!("floating on page {}", page + 1),
            Placement::InText { storage, character } => {
                format!("anchored in storage {storage} at character {character}")
            }
            Placement::Group(owner) => format!("in group {owner}"),
            Placement::PartOf(owner) => format!("part of {owner}"),
            Placement::Unknown => "unplaced".to_string(),
        }
    }
}

/// One placed object.
#[derive(Debug, Clone)]
pub struct Drawable {
    pub identifier: u64,
    pub stream: String,
    pub message_type: u32,
    pub kind: Kind,
    /// Field path from the object's payload to its `TSD.DrawableArchive`. `[1]`
    /// for an image, `[1, 1]` for a shape info, `[1, 1, 1]` for a Keynote
    /// placeholder. Every read and write here goes through it.
    pub path: Vec<u32>,
    pub geometry: Geometry,
    /// `super.parent` — the upward half of a containment iWork stores twice.
    pub parent: Option<u64>,
    pub locked: bool,
    pub aspect_ratio_locked: bool,
    pub hyperlink: Option<String>,
    /// VoiceOver description.
    pub description: Option<String>,
    /// An object-anchored comment.
    pub comment: Option<u64>,
    /// The object's style, unresolved. [`crate::Document::object_style`]
    /// resolves it.
    pub style: Option<u64>,
    /// Text storage owned by a shape or text box.
    pub text: Option<u64>,
    pub path_source: Option<PathSource>,
    /// Children of a group, in z-order, back to front.
    pub children: Vec<u64>,
    pub media: Option<Media>,
    pub edit_state: Option<EditState>,
    pub placement: Placement,
    /// Foreign fields grafted onto this archive by another framework's proto2
    /// extension — a gallery, a web video, a 3D object, a live video source, a
    /// freehand drawing, an equation. Their presence is what says the object is
    /// one of those things; the contents are carried through untouched.
    pub extensions: Vec<&'static str>,
    /// Position in the owning container's list — the z-order, back to front.
    pub z: usize,
    /// Pencil annotations drawn over the object.
    pub pencil_annotations: Vec<u64>,
}

impl Drawable {
    /// The object's rectangle before rotation, in its parent's space.
    ///
    /// For everything but a masked image that is the geometry. For a masked
    /// image it is the mask's window moved into the parent's space, because the
    /// mask hangs off the image and its position is relative to it.
    pub fn base_rect(&self, mask: Option<&Drawable>) -> Frame {
        match mask {
            Some(mask) => Frame {
                x: self.geometry.x + mask.geometry.x,
                y: self.geometry.y + mask.geometry.y,
                width: mask.geometry.width,
                height: mask.geometry.height,
            },
            None => Frame {
                x: self.geometry.x,
                y: self.geometry.y,
                width: self.geometry.width,
                height: self.geometry.height,
            },
        }
    }

    /// The rectangle the app reports for this object.
    ///
    /// Two corrections sit between the archive and what Keynote and Pages say,
    /// and both were found by asking them:
    ///
    /// * **A masked image is reported as its mask.** Pages says 60 × 123,
    ///   475 × 383 for a photo whose own geometry is 33.86 × 66.28,
    ///   511.86 × 466.13 and whose mask is 25.89 × 56.52, 475 × 383 — the
    ///   position is the sum, the size is the mask's.
    /// * **Position is the origin of the *rotated* bounding box; size is not
    ///   rotated.** A 220 × 180 shape stored at 100, 100 and turned 30° is
    ///   reported at 470, 57 — which is the centre, 610 × 190, minus half of
    ///   `(220·cos30 + 180·sin30) × (220·sin30 + 180·cos30)` — and still
    ///   220 × 180. A line stored at 93.84 × 650, 412.31 wide at 346°, comes
    ///   back at exactly 100 × 600.
    ///
    /// What this cannot correct for is a shape that sizes itself to its text:
    /// its stored height is 0 and its stored position is the centre of a box
    /// whose real height only exists once the text has been laid out. Keynote
    /// reports such a text box 58 points above where the archive puts it and
    /// 115 points tall. [`Geometry::fits_its_text`] says when that is the case.
    pub fn frame(&self, mask: Option<&Drawable>) -> Frame {
        let base = self.base_rect(mask);
        let (width, height) = rotated_extent(base.width, base.height, self.geometry.angle);
        Frame {
            x: base.x + base.width / 2.0 - width / 2.0,
            y: base.y + base.height / 2.0 - height / 2.0,
            width: base.width,
            height: base.height,
        }
    }

    /// The mask object, when this drawable has one.
    pub fn mask(&self) -> Option<u64> {
        self.edit_state.as_ref().and_then(|state| state.mask)
    }
}

// -- reading -----------------------------------------------------------------

/// Does this message look like a `TSP.Point` or `TSP.Size` — two floats?
fn is_point(message: &Message) -> bool {
    !message.fields.is_empty()
        && message
            .fields
            .iter()
            .all(|f| matches!(f.number, 1 | 2) && matches!(f.value, Value::Fixed32(_)))
}

/// Does this message look like a `TSD.GeometryArchive`?
///
/// The test is structural rather than positional because the whole point of
/// [`drawable_path`] is not to assume how deep the drawable sits: a geometry is
/// a message whose first two fields are a point and a size, whose field 3 is a
/// small varint and whose field 4 is a float.
fn is_geometry(message: &Message) -> bool {
    let point_at = |field: u32| {
        message
            .bytes(field)
            .and_then(decode_nested)
            .is_some_and(|p| is_point(&p))
    };
    if !point_at(1) && !point_at(2) {
        return false;
    }
    message.fields.iter().all(|f| match f.number {
        1 | 2 => matches!(f.value, Value::Bytes(_)),
        3 => matches!(f.value, Value::Varint(_)),
        4 => matches!(f.value, Value::Fixed32(_)),
        _ => false,
    })
}

/// Field path from an object's payload to the `TSD.DrawableArchive` inside it.
///
/// Returns `Some(vec![])` when the payload *is* the drawable archive. `super`
/// is field 1 everywhere, so this walks field 1 until it finds a message whose
/// own field 1 is a geometry. Four levels is the deepest the corpus goes
/// (`KN.PlaceholderArchive`), and the walk is bounded so a cycle or a
/// misidentified payload cannot run away.
pub fn drawable_path(archive: &Message) -> Option<Vec<u32>> {
    fn walk(message: &Message, depth: usize) -> Option<Vec<u32>> {
        if depth == 0 {
            return None;
        }
        let inner = message.bytes(1).and_then(decode_nested)?;
        if is_geometry(&inner) {
            return Some(Vec::new());
        }
        let mut path = walk(&inner, depth - 1)?;
        path.insert(0, 1);
        Some(path)
    }
    walk(archive, 8)
}

fn float(message: &Message, number: u32) -> f32 {
    match message.get(number) {
        Some(Value::Fixed32(b)) => f32::from_le_bytes(*b),
        _ => 0.0,
    }
}

fn float_or(message: &Message, number: u32, default: f32) -> f32 {
    match message.get(number) {
        Some(Value::Fixed32(b)) => f32::from_le_bytes(*b),
        _ => default,
    }
}

/// A `TSP.Reference` or `TSP.DataReference` — both are `{1: identifier}`.
fn reference(bytes: &[u8]) -> Option<u64> {
    decode_nested(bytes)?.varint(1)
}

fn string(message: &Message, number: u32) -> Option<String> {
    let bytes = message.bytes(number)?;
    std::str::from_utf8(bytes).ok().map(str::to_string)
}

fn size(message: &Message, number: u32) -> Option<(f32, f32)> {
    let point = message.bytes(number).and_then(decode_nested)?;
    is_point(&point).then(|| (float(&point, 1), float(&point, 2)))
}

/// Is this the rectangle iWork writes for an untouched image — the full picture
/// at its natural size?
///
/// The shape is `moveTo(0,0) lineTo(w,0) lineTo(w,h) lineTo(0,h) close
/// moveTo(0,0)`: six elements, the last of them the redundant `moveTo` every
/// iWork writer emits. Anything else is a real trace and must not be rewritten.
fn is_natural_rectangle(path: &Message, width: f32, height: f32) -> bool {
    let corners: Vec<(u64, f32, f32)> = path
        .all(1)
        .filter_map(|value| match value {
            Value::Bytes(raw) => decode_nested(raw),
            _ => None,
        })
        .map(|element| {
            let point = element.bytes(2).and_then(decode_nested).unwrap_or_default();
            (
                element.varint(1).unwrap_or(0),
                float(&point, 1),
                float(&point, 2),
            )
        })
        .collect();
    let close = |a: f32, b: f32| (a - b).abs() <= (a.abs().max(b.abs()) * 1e-4).max(1e-3);
    corners.len() == 6
        && corners[0] == (1, 0.0, 0.0)
        && close(corners[1].1, width)
        && corners[1].0 == 2
        && close(corners[2].1, width)
        && close(corners[2].2, height)
        && close(corners[3].2, height)
        && corners[4].0 == 5
        && corners[5].0 == 1
}

/// The rectangle path iWork writes for a picture of this size, with the
/// trailing `moveTo` and all.
pub fn natural_rectangle(width: f32, height: f32) -> Message {
    let element = |kind: u64, x: f32, y: f32| {
        let mut point = Message::default();
        point.set_in_order(1, Value::Fixed32(x.to_le_bytes()));
        point.set_in_order(2, Value::Fixed32(y.to_le_bytes()));
        let mut element = Message::default();
        element.set_in_order(1, Value::Varint(kind));
        if kind != 5 {
            element.set_in_order(2, Value::Bytes(point.encode()));
        }
        Value::Bytes(element.encode())
    };
    let mut path = Message::default();
    for value in [
        element(1, 0.0, 0.0),
        element(2, width, 0.0),
        element(2, width, height),
        element(2, 0.0, height),
        element(5, 0.0, 0.0),
        element(1, 0.0, 0.0),
    ] {
        path.append_in_order(1, value);
    }
    path
}

/// Read every drawable in a document.
pub fn drawables(document: &crate::Document) -> Vec<Drawable> {
    let containers = containers(document);
    let mut out = Vec::new();

    for (stream, object) in document.objects() {
        let message_type = object.message_type();
        let Some(kind) = Kind::of(message_type) else {
            continue;
        };
        let Ok(archive) = Message::decode(object.payload()) else {
            continue;
        };
        let Some(path) = drawable_path(&archive) else {
            continue;
        };
        let Some(drawable) = (if path.is_empty() {
            Some(archive.clone())
        } else {
            get_path(&archive, &path).and_then(|value| match value {
                Value::Bytes(raw) => decode_nested(&raw),
                _ => None,
            })
        }) else {
            continue;
        };
        let geometry = drawable
            .bytes(field::GEOMETRY)
            .and_then(decode_nested)
            .map(|g| Geometry::decode(&g))
            .unwrap_or(Geometry {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
                flags: 0,
                angle: 0.0,
            });

        // The message one level above the drawable archive is the concrete
        // class — the shape, the image, the movie — and is where everything
        // kind-specific lives.
        let body = if path.is_empty() {
            archive.clone()
        } else {
            get_path(&archive, &path[..path.len() - 1])
                .and_then(|value| match value {
                    Value::Bytes(raw) => decode_nested(&raw),
                    _ => None,
                })
                .unwrap_or_else(|| archive.clone())
        };

        let (media, edit_state) = match kind {
            Kind::Image => {
                let media = Media {
                    data: body.bytes(image_field::DATA).and_then(reference),
                    poster: body.bytes(image_field::THUMBNAIL_DATA).and_then(reference),
                    natural_size: size(&body, image_field::NATURAL_SIZE),
                    original_size: size(&body, image_field::ORIGINAL_SIZE),
                    flags: body.varint(image_field::FLAGS).unwrap_or(0) as u32,
                    trim: None,
                    poster_time: None,
                    volume: None,
                    loop_option: None,
                    audio_only: false,
                    live_video: false,
                    remote_url: None,
                };
                (Some(media), Some(edit_state(document, &body)))
            }
            Kind::Movie => {
                let time = |field: u32| match body.get(field) {
                    Some(Value::Fixed32(bytes)) => Some(f32::from_le_bytes(*bytes)),
                    _ => None,
                };
                let media = Media {
                    data: body.bytes(14).and_then(reference),
                    poster: body.bytes(15).and_then(reference),
                    natural_size: size(&body, 21),
                    original_size: size(&body, 20),
                    flags: body.varint(13).unwrap_or(0) as u32,
                    trim: time(3).zip(time(4)),
                    poster_time: time(5),
                    volume: time(7),
                    // Field 6 is the deprecated integer; field 24 is the enum
                    // that replaced it, and both may be written.
                    loop_option: body.varint(24).or(body.varint(6)).map(|v| v as u32),
                    audio_only: body.varint(9).unwrap_or(0) != 0,
                    live_video: body.varint(30).unwrap_or(0) != 0,
                    remote_url: string(&body, 17),
                };
                (Some(media), None)
            }
            _ => (None, None),
        };

        let style = match kind {
            Kind::Image => body.bytes(image_field::STYLE).and_then(reference),
            Kind::Movie => body.bytes(19).and_then(reference),
            // A mask's field 2 is its path source, not a style — it has none.
            Kind::Mask | Kind::Group => None,
            // A table's field 2 is its `TST.TableModelArchive` (6001), not a
            // style: a `TableInfoArchive` has no object style. Reading it as one
            // reported a bogus style — and `object_style` then read field 10 of
            // the *model* as an override count — for every table in the corpus.
            Kind::Table => None,
            _ => body.bytes(2).and_then(reference),
        };

        let path_source = match kind {
            Kind::Shape | Kind::ConnectionLine => body.bytes(3).and_then(decode_nested),
            Kind::Mask => body.bytes(2).and_then(decode_nested),
            _ => None,
        }
        .as_ref()
        .and_then(PathSource::decode);

        let children: Vec<u64> = if kind == Kind::Group {
            body.all(2)
                .filter_map(|value| match value {
                    Value::Bytes(raw) => reference(raw),
                    _ => None,
                })
                .collect()
        } else {
            Vec::new()
        };

        // A shape's own text storage hangs off the ShapeInfo, one level above
        // the shape archive.
        let text = (message_type == TYPE_SHAPE_INFO
            || message_type == TYPE_PLACEHOLDER
            || message_type == TYPE_PLACEHOLDER_ALIAS
            || message_type == TYPE_COMMENT_INFO)
            .then(|| shape_info(&archive, &path).and_then(|info| info.bytes(2).and_then(reference)))
            .flatten();

        let parent = drawable.bytes(field::PARENT).and_then(reference);
        // No container list names it: fall back to what the object itself says
        // its parent is. A mask's parent is the image it clips, and a Keynote
        // slide keeps its slide-number placeholder outside the drawable list.
        let placement = containers
            .get(&object.identifier)
            .map(|(placement, _)| placement.clone())
            .or_else(|| parent.map(Placement::PartOf))
            .unwrap_or(Placement::Unknown);
        let z = containers
            .get(&object.identifier)
            .map(|(_, z)| *z)
            .unwrap_or(0);

        out.push(Drawable {
            identifier: object.identifier,
            stream: stream.to_string(),
            message_type,
            kind,
            path,
            geometry,
            parent,
            locked: drawable.varint(field::LOCKED).unwrap_or(0) != 0,
            aspect_ratio_locked: drawable.varint(field::ASPECT_RATIO_LOCKED).unwrap_or(0) != 0,
            hyperlink: string(&drawable, field::HYPERLINK),
            description: string(&drawable, field::DESCRIPTION),
            comment: drawable.bytes(field::COMMENT).and_then(reference),
            style,
            text,
            path_source,
            children,
            media,
            edit_state,
            placement,
            extensions: extensions(kind, &body),
            z,
            pencil_annotations: drawable
                .all(field::PENCIL_ANNOTATIONS)
                .filter_map(|value| match value {
                    Value::Bytes(raw) => reference(raw),
                    _ => None,
                })
                .collect(),
        });
    }

    out.sort_by_key(|d| (d.stream.clone(), d.placement.as_str(), d.z, d.identifier));
    out
}

/// The `TSWP.ShapeInfoArchive` inside an object.
///
/// It is two levels above the drawable archive — shape info, shape, drawable —
/// so for a bare `TSWP.ShapeInfoArchive` that is the payload itself, and for a
/// `KN.PlaceholderArchive` it is one level down.
fn shape_info(archive: &Message, path: &[u32]) -> Option<Message> {
    let at = &path[..path.len().saturating_sub(2)];
    if at.is_empty() {
        return Some(archive.clone());
    }
    get_path(archive, at).and_then(|value| match value {
        Value::Bytes(raw) => decode_nested(&raw),
        _ => None,
    })
}

/// Which framework has grafted fields onto this archive.
///
/// Proto2 extensions look like plain high-numbered fields on the wire, so the
/// only way to name one is to know which host message it belongs to — the same
/// number means different things on an image and on a movie.
fn extensions(kind: Kind, body: &Message) -> Vec<&'static str> {
    let mut out = Vec::new();
    match kind {
        Kind::Image => {
            if body.get(200).is_some() {
                out.push("gallery");
            }
            if body.get(300).is_some() {
                out.push("web video");
            }
            if (100..=103).any(|f| body.get(f).is_some()) {
                out.push("equation");
            }
        }
        Kind::Movie => {
            if body.get(100).is_some() {
                out.push("live video");
            }
            if body.get(200).is_some() {
                out.push("3D object");
            }
        }
        Kind::Group if body.get(100).is_some() => out.push("freehand drawing"),
        _ => {}
    }
    out
}

/// Read an image's non-destructive edit state.
fn edit_state(document: &crate::Document, image: &Message) -> EditState {
    let mut state = EditState {
        mask: image.bytes(image_field::MASK).and_then(reference),
        instant_alpha: image.get(image_field::INSTANT_ALPHA_PATH).is_some(),
        background_removed: image.varint(image_field::BACKGROUND_REMOVED).unwrap_or(0) != 0,
        ..EditState::default()
    };

    for (field, name) in [
        // A separately stored downscale of the *old* picture. 65 of the 69
        // corpus images carry it, and replacing the bytes leaves it lying —
        // the same objection as the other derived renderings.
        (image_field::THUMBNAIL_DATA, "thumbnailImageData"),
        (image_field::ORIGINAL_DATA, "originalData"),
        (image_field::ADJUSTED_DATA, "adjustedImageData"),
        (
            image_field::THUMBNAIL_ADJUSTED_DATA,
            "thumbnailAdjustedImageData",
        ),
        (image_field::ENHANCED_DATA, "enhancedImageData"),
    ] {
        if image.get(field).is_some() {
            state.derived.push(name);
        }
    }

    if let Some(adjustments) = image
        .bytes(image_field::ADJUSTMENTS)
        .and_then(decode_nested)
    {
        for (number, name, default) in ADJUSTMENTS {
            let Some(Value::Fixed32(bytes)) = adjustments.get(*number) else {
                continue;
            };
            let value = f32::from_le_bytes(*bytes);
            if (value - default).abs() > f32::EPSILON {
                state.adjustments.push((name, value));
            }
        }
        if adjustments.varint(13).unwrap_or(0) != 0 {
            state.adjustments.push(("enhance", 1.0));
        }
    }

    let natural = size(image, image_field::NATURAL_SIZE);
    if let (Some(path), Some((width, height))) = (
        image
            .bytes(image_field::TRACED_PATH)
            .and_then(decode_nested),
        natural,
    ) {
        state.traced_shape = !is_natural_rectangle(&path, width, height);
    }

    // The mask crops unless its window is the whole drawn picture at the
    // origin — the identity mask the app installs when it replaces an image,
    // which hides nothing. The picture's drawn rectangle is the image's own
    // geometry. It is emphatically **not** `originalSize` (field 4): for a
    // masked image the app fills that field with the mask window itself
    // (keynote-shapes 160×160 image / 160×120 originalSize / 160×120 mask;
    // pages-book 324×486 / 324×216 / 324×216), so comparing the mask size to
    // originalSize is a tautology that lets a real crop slid to (0, 0) pass as
    // an identity.
    if let Some(mask) = state.mask {
        if let Some(archive) = document
            .object(mask)
            .and_then(|(_, object)| Message::decode(object.payload()).ok())
        {
            let drawable = archive.bytes(1).and_then(decode_nested).unwrap_or_default();
            let mask_geometry = drawable
                .bytes(field::GEOMETRY)
                .and_then(decode_nested)
                .map(|g| Geometry::decode(&g));
            let image_rect = image
                .bytes(1)
                .and_then(decode_nested)
                .and_then(|d| d.bytes(field::GEOMETRY).and_then(decode_nested))
                .map(|g| Geometry::decode(&g));
            if let Some(mask_geometry) = mask_geometry {
                state.mask_frame = Some(Frame {
                    x: mask_geometry.x,
                    y: mask_geometry.y,
                    width: mask_geometry.width,
                    height: mask_geometry.height,
                });
                let same = |a: f32, b: f32| (a - b).abs() < 0.01;
                state.crops = !(same(mask_geometry.x, 0.0)
                    && same(mask_geometry.y, 0.0)
                    && image_rect.is_some_and(|r| {
                        same(mask_geometry.width, r.width) && same(mask_geometry.height, r.height)
                    }));
            }
            // A "Mask with Shape" — a triangle, a diamond — is not a crop, and
            // the element count cannot tell one from a plain rectangle: a
            // triangle is five elements, a diamond six, and iWork's rectangle is
            // six too. The exact test is whether the bezier *is* the natural
            // rectangle of its own size; [`is_natural_rectangle`] is that test.
            if let Some(source_msg) = archive.bytes(2).and_then(decode_nested) {
                if let Some(source) = PathSource::decode(&source_msg) {
                    state.mask_is_a_shape = match source.kind {
                        PathSourceKind::Bezier => !source_msg
                            .bytes(5)
                            .and_then(decode_nested)
                            .and_then(|bezier| {
                                let path = bezier.bytes(3).and_then(decode_nested)?;
                                let (w, h) = source.natural_size?;
                                Some(is_natural_rectangle(&path, w, h))
                            })
                            .unwrap_or(false),
                        _ => true,
                    };
                }
            }
        }
    }
    state
}

/// Which container holds each drawable, and at what depth in its list.
///
/// Containment is written twice — downward from the sheet, slide or page, and
/// upward as each drawable's `parent` — and the downward list is the one that
/// carries **z-order**, back to front. The lists are in different archives per
/// app, all of them found by message type:
///
/// | App | Archive | Field |
/// |---|---|---|
/// | Numbers | `TN.SheetArchive` (2) | 2 |
/// | Keynote | `KN.SlideArchive` (5) | 7 |
/// | Pages | `TP.FloatingDrawablesArchive` (10010) | 1 |
/// | Pages | `TP.SectionTemplateArchive` (10143) | 3 |
/// | any | `TSD.GroupArchive` (3008) | 2 |
/// | any | `TSWP.StorageArchive` (2001) attachment table | 9 |
///
/// Pages is the app that needed care. Its `TP.DrawablesZOrderArchive` (10015)
/// lists *every* drawable in one document-wide order — the body storage and the
/// image anchored inside it side by side — so it answers "how deep" and nothing
/// about "where". Reading it as the floating list reports an anchored image as
/// floating, which it did until the attachment table was checked against it.
fn containers(document: &crate::Document) -> BTreeMap<u64, (Placement, usize)> {
    let mut out: BTreeMap<u64, (Placement, usize)> = BTreeMap::new();
    let mut depth: BTreeMap<u64, usize> = BTreeMap::new();
    let mut record = |target: u64, placement: Placement, z: usize| {
        out.entry(target).or_insert((placement, z));
    };

    for (stream, object) in document.objects() {
        let Ok(archive) = Message::decode(object.payload()) else {
            continue;
        };
        let listed = |field: u32| -> Vec<u64> {
            archive
                .all(field)
                .filter_map(|value| match value {
                    Value::Bytes(raw) => reference(raw),
                    _ => None,
                })
                .collect()
        };
        match (object.message_type(), document.kind()) {
            (2, crate::Kind::Numbers) => {
                let name = string(&archive, 1).unwrap_or_default();
                for (z, target) in listed(2).into_iter().enumerate() {
                    record(target, Placement::Sheet(name.clone()), z);
                }
            }
            (5, crate::Kind::Keynote) => {
                let slide = stream
                    .rsplit('/')
                    .next()
                    .unwrap_or(stream)
                    .trim_end_matches(".iwa")
                    .to_string();
                for (z, target) in listed(7).into_iter().enumerate() {
                    record(target, Placement::Slide(slide.clone()), z);
                }
            }
            (10015, crate::Kind::Pages) => {
                for (z, target) in listed(1).into_iter().enumerate() {
                    depth.insert(target, z);
                }
            }
            (10010, crate::Kind::Pages) => {
                // **Two levels, not one.** The archive is one entry *per page*
                // — `{1: page index, 4: [{1: reference}]}` — and reading field
                // 1 as a list of references reads the page number as an object
                // identifier, so every floating drawable in every Pages
                // document came back `Unknown`. Found by a chart: the only
                // drawable in `pages-numbering` that is neither anchored in
                // text nor part of a section template.
                for page_entry in archive.all(1) {
                    let Value::Bytes(raw) = page_entry else {
                        continue;
                    };
                    let Some(page) = decode_nested(raw) else {
                        continue;
                    };
                    let number = page.varint(1).unwrap_or(0) as u32;
                    for (z, value) in page.all(4).enumerate() {
                        let Value::Bytes(entry) = value else {
                            continue;
                        };
                        // Each entry is a wrapper around the reference, not
                        // the reference itself: `{1: {1: identifier}}`.
                        if let Some(target) = decode_nested(entry)
                            .and_then(|wrapper| wrapper.bytes(1).and_then(reference))
                        {
                            record(target, Placement::Floating { page: number }, z);
                        }
                    }
                }
            }
            (10143, crate::Kind::Pages) => {
                for (z, target) in listed(3).into_iter().enumerate() {
                    record(target, Placement::SectionTemplate(object.identifier), z);
                }
            }
            (TYPE_GROUP, _) => {
                for (z, target) in listed(2).into_iter().enumerate() {
                    record(target, Placement::Group(object.identifier), z);
                }
            }
            (crate::TYPE_STORAGE, _) => {
                // Field 9 is the attachment table: `{1: character index,
                // 2: -> TSWP.DrawableAttachmentArchive}`, and the attachment's
                // own field 1 is the drawable. That indirection is what makes a
                // Pages image "anchored at character 12" rather than "on the
                // page".
                let Some(table) = archive.bytes(9).and_then(decode_nested) else {
                    continue;
                };
                for (z, value) in table.all(1).enumerate() {
                    let Value::Bytes(raw) = value else { continue };
                    let Some(entry) = decode_nested(raw) else {
                        continue;
                    };
                    let character = entry.varint(1).unwrap_or(0);
                    let Some(attachment) = entry.bytes(2).and_then(reference) else {
                        continue;
                    };
                    let Some(target) = document
                        .object(attachment)
                        .and_then(|(_, o)| Message::decode(o.payload()).ok())
                        .and_then(|a| a.bytes(1).and_then(reference))
                    else {
                        continue;
                    };
                    record(
                        target,
                        Placement::InText {
                            storage: object.identifier,
                            character,
                        },
                        z,
                    );
                }
            }
            _ => {}
        }
    }
    // Pages' document-wide order refines the depth of anything it names,
    // whatever container the object turned out to belong to.
    for (target, z) in depth {
        if let Some(entry) = out.get_mut(&target) {
            entry.1 = z;
        } else {
            out.insert(target, (Placement::Unknown, z));
        }
    }
    out
}

/// What [`crate::Document::set_geometry`] did.
#[derive(Debug, Clone)]
pub struct GeometryChange {
    pub drawable: u64,
    /// The rectangle the app reported before and after — the mask's window
    /// where there is one.
    pub before: Frame,
    pub after: Frame,
    /// The mask that was scaled with the picture, if any.
    pub mask: Option<u64>,
    /// Every object whose archive was rewritten.
    pub rewritten: Vec<u64>,
}

/// Resolve a style object, walking `parent` for every property it does not
/// carry itself.
pub fn object_style(document: &crate::Document, identifier: u64) -> Option<ObjectStyle> {
    let mut style = ObjectStyle {
        identifier,
        ..ObjectStyle::default()
    };
    let mut next = Some(identifier);
    let mut seen = Vec::new();
    while let Some(current) = next {
        if seen.contains(&current) || seen.len() > 8 {
            break;
        }
        seen.push(current);
        let Some((_, object)) = document.object(current) else {
            break;
        };
        let Ok(archive) = Message::decode(object.payload()) else {
            break;
        };
        // A `TSWP.ShapeStyleArchive` wraps the TSD one at field 1; a bare
        // `TSD.ShapeStyleArchive` or `TSD.MediaStyleArchive` is the TSD one.
        let (base, shape) = match object.message_type() {
            TYPE_WP_SHAPE_STYLE => (vec![1u32], StyleShape::Shape),
            TYPE_MEDIA_STYLE => (Vec::new(), StyleShape::Media),
            _ => (Vec::new(), StyleShape::Shape),
        };
        let at = |extra: &[u32]| -> Vec<u32> {
            let mut path = base.clone();
            path.extend_from_slice(extra);
            path
        };
        let tsd = get_path(&archive, &base)
            .and_then(|value| match value {
                Value::Bytes(raw) => decode_nested(&raw),
                _ => None,
            })
            .unwrap_or_else(|| archive.clone());

        if style.name.is_none() {
            style.name = crate::style::string_at(&archive, &at(&[1, 1]));
        }
        if style.override_count.is_none() {
            style.override_count = tsd.varint(10).map(|n| n as u32);
        }
        if current != identifier {
            style.inherited_from.push(current);
        }

        if let Some(properties) = tsd.bytes(11).and_then(decode_nested) {
            let (fill, stroke, opacity, shadow, reflection) = match shape {
                StyleShape::Shape => (Some(1u32), 2u32, 3u32, 4u32, 5u32),
                StyleShape::Media => (None, 1, 2, 3, 4),
            };
            if let Some(fill) = fill {
                if style.fill.is_none() {
                    if let Some(message) = properties.bytes(fill).and_then(decode_nested) {
                        style.fill = Some(Fill::decode(&message));
                    } else if properties.get(fill).is_some() {
                        style.fill = Some(Fill::None);
                    }
                }
            }
            if style.stroke.is_none() {
                if let Some(message) = properties.bytes(stroke).and_then(decode_nested) {
                    style.stroke = Some(Stroke::decode(&message));
                }
            }
            if style.opacity.is_none() {
                if let Some(Value::Fixed32(bytes)) = properties.get(opacity) {
                    style.opacity = Some(f32::from_le_bytes(*bytes));
                }
            }
            if style.shadow.is_none() {
                if let Some(message) = properties.bytes(shadow).and_then(decode_nested) {
                    style.shadow = Some(Shadow::decode(&message));
                }
            }
            if style.reflection.is_none() {
                if let Some(message) = properties.bytes(reflection).and_then(decode_nested) {
                    style.reflection = Some(float_or(&message, 1, 0.5));
                }
            }
        }

        // `TSS.StyleArchive.parent` is `{3: {1: identifier}}`, so the path ends
        // in the reference's own field — [1, 3, 1] from the style archive,
        // which is [1, 1, 3, 1] when a `TSWP.ShapeStyleArchive` wraps it.
        next = reference_at(&archive, &at(&[1, 3, 1]));
    }
    Some(style)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pb::Field;

    fn point(x: f32, y: f32) -> Message {
        let mut message = Message::default();
        message.set_in_order(1, Value::Fixed32(x.to_le_bytes()));
        message.set_in_order(2, Value::Fixed32(y.to_le_bytes()));
        message
    }

    fn geometry(x: f32, y: f32, w: f32, h: f32, flags: u32, angle: f32) -> Message {
        let mut message = Message::default();
        message.set_in_order(1, Value::Bytes(point(x, y).encode()));
        message.set_in_order(2, Value::Bytes(point(w, h).encode()));
        message.set_in_order(3, Value::Varint(u64::from(flags)));
        message.set_in_order(4, Value::Fixed32(angle.to_le_bytes()));
        message
    }

    #[test]
    fn geometry_roundtrips() {
        let encoded = geometry(956.0, 536.0, 8.0, 8.0, 3, 0.0);
        let decoded = Geometry::decode(&encoded);
        assert_eq!(decoded.x, 956.0);
        assert_eq!(decoded.width, 8.0);
        assert_eq!(decoded.flags, 3);

        let mut again = encoded.clone();
        decoded.write_into(&mut again);
        assert_eq!(
            again.encode(),
            encoded.encode(),
            "a no-op write changes nothing"
        );
    }

    /// Writing position and size must leave flags and angle alone: they are
    /// the fields nothing here understands well enough to rebuild.
    #[test]
    fn writing_geometry_keeps_the_rest() {
        let mut message = geometry(0.0, 0.0, 10.0, 10.0, 3, 30.0);
        Geometry {
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
            flags: 0,
            angle: 0.0,
        }
        .write_into(&mut message);
        let after = Geometry::decode(&message);
        assert_eq!(
            (after.x, after.y, after.width, after.height),
            (1.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(after.flags, 3, "flags survive");
        assert_eq!(after.angle, 30.0, "angle survives");
    }

    /// The whole enumeration hangs off finding the drawable archive at an
    /// unknown depth. Four levels is what a Keynote placeholder needs.
    #[test]
    fn the_drawable_is_found_at_any_depth() {
        let mut drawable = Message::default();
        drawable.set_in_order(
            1,
            Value::Bytes(geometry(1.0, 2.0, 3.0, 4.0, 3, 0.0).encode()),
        );

        let mut level = drawable.clone();
        for depth in 0..4 {
            let path = drawable_path(&level).expect("found");
            assert_eq!(path.len(), depth);
            let mut outer = Message::default();
            outer.set_in_order(1, Value::Bytes(level.encode()));
            level = outer;
        }
    }

    #[test]
    fn a_message_with_no_geometry_is_not_a_drawable() {
        let mut message = Message::default();
        message.set_in_order(1, Value::Bytes(b"not a message".to_vec()));
        assert_eq!(drawable_path(&message), None);
        assert_eq!(drawable_path(&Message::default()), None);
    }

    /// A point is two floats; a geometry holds two of them. Neither must be
    /// mistaken for the other, or the walk stops one level too early.
    #[test]
    fn a_point_is_not_a_geometry() {
        assert!(is_point(&point(1.0, 2.0)));
        assert!(!is_geometry(&point(1.0, 2.0)));
        assert!(is_geometry(&geometry(0.0, 0.0, 1.0, 1.0, 3, 0.0)));
    }

    #[test]
    fn the_natural_rectangle_is_the_one_iwork_writes() {
        let path = natural_rectangle(750.0, 683.0);
        assert!(is_natural_rectangle(&path, 750.0, 683.0));
        assert!(!is_natural_rectangle(&path, 100.0, 683.0));
        // Five elements — the trailing moveTo dropped — is not it.
        let mut short = path.clone();
        short.fields.pop();
        assert!(!is_natural_rectangle(&short, 750.0, 683.0));
    }

    #[test]
    fn a_stroke_pattern_tells_dots_from_dashes() {
        let pattern = |kind: u64, count: u64, first: f32| {
            let mut p = Message::default();
            p.set_in_order(1, Value::Varint(kind));
            p.set_in_order(3, Value::Varint(count));
            for value in [first, 2.0, 0.0, 0.0, 0.0, 0.0] {
                p.fields.push(Field {
                    number: 4,
                    value: Value::Fixed32(value.to_le_bytes()),
                });
            }
            let mut stroke = Message::default();
            stroke.set_in_order(6, Value::Bytes(p.encode()));
            Stroke::decode(&stroke).pattern
        };
        assert_eq!(pattern(1, 0, 0.0), StrokePattern::Solid);
        assert_eq!(pattern(2, 0, 0.0), StrokePattern::Empty);
        assert_eq!(pattern(0, 2, 2.0), StrokePattern::Dashed);
        assert_eq!(pattern(0, 2, 0.0001), StrokePattern::Dotted);
    }
}

// -- adding a drawable --------------------------------------------------------

/// Where a new drawable can go, and how the container holds it.
pub(crate) enum Container {
    /// A Keynote slide: `owned_drawables` (7) and `drawables_z_order` (42).
    Slide { identifier: u64 },
    /// A Numbers sheet: `drawable_infos` (2).
    Sheet { identifier: u64 },
    /// A Pages page. Nothing owns the drawable — the page group in
    /// `TP.FloatingDrawablesArchive` names it, and `TP.DrawablesZOrderArchive`
    /// says where in the stack it sits.
    Page {
        floating: u64,
        zorder: Option<u64>,
        page: u64,
    },
}

impl Container {
    /// The object the drawable's `parent` points at, if the container is one
    /// that owns its drawables. A Pages page is not.
    pub(crate) fn parent(&self) -> Option<u64> {
        match self {
            Container::Slide { identifier } | Container::Sheet { identifier } => Some(*identifier),
            Container::Page { .. } => None,
        }
    }

    /// The object a new drawable is written beside, so it lands in the stream
    /// the app expects to read it from.
    pub(crate) fn neighbour(&self) -> u64 {
        match self {
            Container::Slide { identifier } | Container::Sheet { identifier } => *identifier,
            Container::Page { floating, .. } => *floating,
        }
    }
}

/// Find what a caller means by a container: a slide (by its archive or its
/// node), a sheet (by identifier or name), or a page (`page 2`).
///
/// Slides first, then sheets, then pages, because only the last is ambiguous
/// with a bare number — and a document with slides or sheets has no pages to
/// put anything on.
pub(crate) fn container_of(
    document: &crate::Document,
    wanted: &str,
) -> Result<Container, crate::Error> {
    let by_id: Option<u64> = wanted.parse().ok();
    if let Some(show) = document.show() {
        for slide in &show.slides {
            if by_id == Some(slide.identifier) || by_id == Some(slide.node) {
                return Ok(Container::Slide {
                    identifier: slide.identifier,
                });
            }
        }
    }
    for (_, object) in document.objects() {
        if object.message_type() != crate::table::TYPE_SHEET {
            continue;
        }
        let Ok(archive) = Message::decode(object.payload()) else {
            continue;
        };
        let named = archive
            .bytes(crate::table::sheet_field::NAME)
            .is_some_and(|raw| String::from_utf8_lossy(raw) == wanted);
        if by_id == Some(object.identifier) || named {
            return Ok(Container::Sheet {
                identifier: object.identifier,
            });
        }
    }
    if let Some(page) = page_of(document, wanted)? {
        return Ok(page);
    }
    Err(crate::Error::Format(format!(
        "no slide, sheet or page {wanted:?} to put a drawable on"
    )))
}

/// A Pages page, named `page 2` or just `2`, counting from one.
///
/// Nothing here knows how many pages a document has — laying the text out is
/// the app's job, and this crate does not do it. So the number is taken on
/// trust, and a box put past the end is a box the app will not show.
fn page_of(document: &crate::Document, wanted: &str) -> Result<Option<Container>, crate::Error> {
    let number = wanted
        .strip_prefix("page ")
        .unwrap_or(wanted)
        .trim()
        .parse::<u64>();
    let Ok(page) = number else { return Ok(None) };
    let Some((_, floating)) = document
        .objects()
        .find(|(_, object)| object.message_type() == crate::pages::TYPE_FLOATING_DRAWABLES)
    else {
        return Ok(None);
    };
    if page == 0 {
        return Err(crate::Error::Format(
            "pages are counted from one, so there is no page 0".into(),
        ));
    }
    let floating = floating.identifier;
    let zorder = document
        .archive(crate::create::ROOT)
        .ok()
        .and_then(|root| reference_at(&root, &[crate::pages::document_field::DRAWABLES_ZORDER, 1]));
    Ok(Some(Container::Page {
        floating,
        zorder,
        page: page - 1,
    }))
}

/// A style a text box can be drawn with, from the document rather than invented.
///
/// A shape already on the page is the best answer — the new box then looks like
/// the ones beside it. Failing that, the theme's text-box preset, which is what
/// the app itself reaches for when it makes one.
fn text_box_style(document: &crate::Document) -> Option<u64> {
    for drawable in document.drawables() {
        if drawable.text.is_some() {
            if let Some(style) = drawable.style {
                return Some(style);
            }
        }
    }
    document
        .objects()
        .find(|(_, object)| object.message_type() == crate::create::TYPE_SHAPE_STYLE)
        .map(|(_, object)| object.identifier)
}

/// The styles a new text storage points at: a paragraph style and a list style
/// the document already has.
pub(crate) fn text_styles(document: &crate::Document) -> Option<(u64, u64, u64)> {
    let stylesheet = document
        .objects()
        .find(|(_, object)| object.message_type() == crate::create::TYPE_STYLESHEET)
        .map(|(_, object)| object.identifier)?;
    let paragraph = document
        .objects()
        .find(|(_, object)| object.message_type() == crate::style::TYPE_PARAGRAPH_STYLE)
        .map(|(_, object)| object.identifier)?;
    let list = document
        .objects()
        .find(|(_, object)| object.message_type() == crate::style::TYPE_LIST_STYLE)
        .map(|(_, object)| object.identifier)?;
    Some((stylesheet, paragraph, list))
}

/// The outline a new shape is drawn along.
///
/// Only three, because these are the three whose path this crate can write
/// from nothing and check: everything else the app offers — a star, a
/// chevron, a rounded rectangle — is a `TSD.PointPathSourceArchive` or a
/// `TSD.ScalarPathSourceArchive` whose parameters nothing here has decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Outline {
    /// The box itself, which is what a text box is.
    #[default]
    Rectangle,
    /// An ellipse inscribed in the box.
    Ellipse,
    /// A straight line from the top left of the box to the bottom right, so a
    /// height of zero is a horizontal rule.
    Line,
}

/// Put a text box on a Keynote slide, a Numbers sheet or a Pages page.
///
/// Everything it points at comes from the document: the style of a text shape
/// already there (or the theme's text-box preset), the stylesheet, a paragraph
/// style and a list style. Nothing is invented, which is why the box looks like
/// the document it lands in.
///
/// Returns the `TSWP.ShapeInfoArchive` — the object `iwork drawables` prints
/// and [`crate::Document::set_geometry`] moves.
pub fn add_text_box(
    document: &mut crate::Document,
    container: &str,
    text: &str,
    position: (f32, f32),
    size: (f32, f32),
) -> Result<u64, crate::Error> {
    add_shape(
        document,
        container,
        Outline::Rectangle,
        text,
        position,
        size,
    )
}

/// Put a shape on a Keynote slide, a Numbers sheet or a Pages page.
///
/// A shape is a text box that is not a rectangle and need hold no text: the
/// archive is the same one, and the only difference is the path. See
/// [`add_text_box`] for what it borrows from the document.
pub fn add_shape(
    document: &mut crate::Document,
    container: &str,
    outline: Outline,
    text: &str,
    position: (f32, f32),
    size: (f32, f32),
) -> Result<u64, crate::Error> {
    let flat = size.0 <= 0.0 || size.1 <= 0.0;
    // A line is allowed to be flat — that is what a horizontal rule is — but
    // it still has to go somewhere, and a shape with no area is invisible.
    let empty = match outline {
        Outline::Line => size.0 == 0.0 && size.1 == 0.0,
        _ => flat,
    };
    if empty || size.0 < 0.0 || size.1 < 0.0 {
        return Err(crate::Error::Format(format!(
            "a {} needs a size, and {} × {} is not one",
            match outline {
                Outline::Rectangle => "text box",
                Outline::Ellipse => "shape",
                Outline::Line => "line",
            },
            size.0,
            size.1
        )));
    }
    let container = container_of(document, container)?;
    let style = text_box_style(document).ok_or_else(|| {
        crate::Error::Format(
            "this document has no shape style to draw a text box with, and inventing one here \
             would be a style the document never defined"
                .into(),
        )
    })?;
    let (stylesheet, paragraph, list) = text_styles(document).ok_or_else(|| {
        crate::Error::Format("this document has no text styles to write text with".into())
    })?;

    let parent = container.parent();
    let neighbour = container.neighbour();

    let mut grow = crate::create::Grow::new(document);
    let storage = grow.allocate();
    let shape = grow.allocate();
    grow.beside(
        neighbour,
        storage,
        crate::TYPE_STORAGE,
        &crate::create::text_box_storage(stylesheet, paragraph, list, text),
    )?;
    grow.beside(
        neighbour,
        shape,
        crate::create::TYPE_SHAPE_INFO,
        &crate::create::text_box(
            parent,
            style,
            storage,
            outline,
            position,
            size,
            matches!(container, Container::Page { .. }),
        ),
    )?;
    grow.finish()?;

    hold(document, &container, shape)?;
    document.declare_external_references();
    Ok(shape)
}

/// The downward half of containment: the container's own list of what is on
/// it, which is also the z-order.
pub(crate) fn hold(
    document: &mut crate::Document,
    container: &Container,
    shape: u64,
) -> Result<(), crate::Error> {
    match container {
        Container::Slide { identifier } => {
            let identifier = *identifier;
            let mut archive = document.archive(identifier)?;
            archive.append_in_order(
                crate::keynote::slide_field::OWNED_DRAWABLES,
                Value::Bytes(crate::create::reference_bytes(shape)),
            );
            archive.append_in_order(
                crate::keynote::slide_field::DRAWABLES_Z_ORDER,
                Value::Bytes(crate::create::reference_bytes(shape)),
            );
            document.set_archive_of(identifier, &archive)?;
        }
        Container::Sheet { identifier } => {
            let identifier = *identifier;
            let mut archive = document.archive(identifier)?;
            archive.append_in_order(
                crate::table::sheet_field::DRAWABLES,
                Value::Bytes(crate::create::reference_bytes(shape)),
            );
            document.set_archive_of(identifier, &archive)?;
        }
        Container::Page {
            floating,
            zorder,
            page,
        } => {
            let (floating, zorder, page) = (*floating, *zorder, *page);
            let mut archive = document.archive(floating)?;
            put_on_page(&mut archive, page, shape);
            document.set_archive_of(floating, &archive)?;
            if let Some(zorder) = zorder {
                let mut stack = document.archive(zorder)?;
                stack.append_in_order(1, Value::Bytes(crate::create::reference_bytes(shape)));
                document.set_archive_of(zorder, &stack)?;
            }
        }
    }
    Ok(())
}

/// A media style the document already has, for an image to be drawn with.
///
/// An image already placed is the best answer; failing that the theme's
/// image preset. A `TSD.MediaStyleArchive` is not a shape style and the two
/// number their properties differently, so a shape style will not do.
fn image_style(document: &crate::Document) -> Option<u64> {
    for drawable in document.drawables() {
        if drawable.kind == Kind::Image {
            if let Some(style) = drawable.style {
                return Some(style);
            }
        }
    }
    document
        .objects()
        .find(|(_, object)| object.message_type() == TYPE_MEDIA_STYLE)
        .map(|(_, object)| object.identifier)
}

/// Put an image on a Keynote slide, a Numbers sheet or a Pages page.
///
/// The bytes go into the package under `Data/`, a `TSP.DataInfo` is added to
/// the registry with their SHA-1 and their length, and the drawable points at
/// it. `size` is the rectangle on the page, in points; passing `None` uses the
/// picture's own pixel size, which is what the app does when a picture is
/// dropped at full size.
///
/// Only PNG and JPEG, because the pixel size has to be known — the registry
/// records it and the drawable's `naturalSize` must agree with it, and this
/// crate reads headers rather than decoding pictures.
pub fn add_image(
    document: &mut crate::Document,
    container: &str,
    bytes: &[u8],
    preferred_name: &str,
    position: (f32, f32),
    size: Option<(f32, f32)>,
) -> Result<u64, crate::Error> {
    let natural = crate::media::pixel_size(bytes).ok_or_else(|| {
        crate::Error::Format(
            "the picture is not a PNG or a JPEG, so its pixel size cannot be read here — and              an image whose size the registry does not know is one the app draws wrong"
                .into(),
        )
    })?;
    let size = size.unwrap_or(natural);
    if size.0 <= 0.0 || size.1 <= 0.0 {
        return Err(crate::Error::Format(format!(
            "an image needs a size, and {} × {} is not one",
            size.0, size.1
        )));
    }
    let container = container_of(document, container)?;
    let style = image_style(document).ok_or_else(|| {
        crate::Error::Format(
            "this document has no media style to draw an image with, and inventing one here              would be a style the document never defined"
                .into(),
        )
    })?;

    let parent = container.parent();
    let neighbour = container.neighbour();

    // **One stored file per distinct content.** iWork's media registry is
    // keyed by digest, and a package holding the same bytes twice under two
    // names is one the app aborts on: Keynote dies in `TSPersistence` before
    // it draws anything, and no error reaches the user. So the digest decides
    // whether this is a new file or another reference to one already here.
    let digest = crate::media::sha1(bytes);
    let existing = document
        .data_files()
        .into_iter()
        .find(|file| file.digest == digest)
        .map(|file| file.identifier);

    let mut grow = crate::create::Grow::new(document);
    // The data identifier comes out of the same counter as the objects. A
    // `DataReference` and an object reference are told apart by where they
    // sit, not by their value, so sharing the counter is what keeps a data
    // identifier from ever being read as an object.
    let data = match existing {
        Some(identifier) => identifier,
        None => grow.allocate(),
    };
    let image = grow.allocate();
    grow.beside_with(
        neighbour,
        image,
        TYPE_IMAGE,
        &crate::create::image(
            parent,
            style,
            data,
            position,
            size,
            natural,
            matches!(container, Container::Page { .. }),
        ),
        &[data],
    )?;
    grow.finish()?;

    if existing.is_none() {
        document.register_media(data, bytes, preferred_name, natural)?;
    }
    hold(document, &container, image)?;
    document.declare_external_references();
    Ok(image)
}

/// Name a drawable in a `TP.FloatingDrawablesArchive` page group, making the
/// group if the page has none yet.
///
/// The groups are a repeated field carrying the page index inside them, so
/// they are not in any particular order and a page with nothing on it has no
/// group at all — which is why every word-processing document this crate has
/// read has an *empty* floating archive.
fn put_on_page(archive: &mut Message, page: u64, shape: u64) {
    const PAGE_GROUPS: u32 = 1;
    const PAGE_INDEX: u32 = 1;
    const DRAWABLES: u32 = 4;

    let entry = crate::create::nested(DRAWABLES, vec![crate::create::reference(1, shape)]);
    for field in archive
        .fields
        .iter_mut()
        .filter(|f| f.number == PAGE_GROUPS)
    {
        let Value::Bytes(raw) = &field.value else {
            continue;
        };
        let Ok(mut group) = Message::decode(raw) else {
            continue;
        };
        if group.varint(PAGE_INDEX).unwrap_or(0) != page {
            continue;
        }
        group.append_in_order(DRAWABLES, entry.value);
        field.value = Value::Bytes(group.encode());
        return;
    }
    let group = crate::create::message(vec![crate::create::varint(PAGE_INDEX, page), entry]);
    archive.append_in_order(PAGE_GROUPS, Value::Bytes(group.encode()));
}

// -- writing a style ---------------------------------------------------------

/// A property of an object style this crate can write.
///
/// The numbering differs between a shape and a media style — a media style has
/// no fill, and everything after it shifts down by one — which is why this is
/// an enum resolved against the archive rather than a bare field number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Property {
    Fill,
    Stroke,
    Opacity,
}

/// The path from a style archive's root to one property, or a refusal naming
/// what the object turned out to be.
///
/// A `TSWP.ShapeStyleArchive` wraps the `TSD` one at field 1; a bare
/// `TSD.ShapeStyleArchive` or `TSD.MediaStyleArchive` *is* the `TSD` one. The
/// property bag is field 11 of whichever it is.
fn property_path(
    document: &crate::Document,
    style: u64,
    property: Property,
) -> Result<Vec<u32>, crate::Error> {
    let Some((_, object)) = document.object(style) else {
        return Err(crate::Error::refused(
            crate::Refusal::NotFound,
            format!("object {style} is not in this document"),
        ));
    };
    let (mut path, shape) = match object.message_type() {
        TYPE_WP_SHAPE_STYLE => (vec![1u32], StyleShape::Shape),
        TYPE_SHAPE_STYLE => (Vec::new(), StyleShape::Shape),
        TYPE_MEDIA_STYLE => (Vec::new(), StyleShape::Media),
        other => {
            return Err(crate::Error::refused(
                crate::Refusal::WrongSlot,
                format!(
                    "object {style} is message type {other}, not a shape or media style — \
                     a drawable's style identifier is the one to pass here"
                ),
            ))
        }
    };
    let field = match (shape, property) {
        (StyleShape::Media, Property::Fill) => {
            return Err(crate::Error::refused(
                crate::Refusal::WrongSlot,
                format!("style {style} is a media style, and a media style has no fill"),
            ))
        }
        (StyleShape::Shape, Property::Fill) => 1,
        (StyleShape::Shape, Property::Stroke) => 2,
        (StyleShape::Shape, Property::Opacity) => 3,
        (StyleShape::Media, Property::Stroke) => 1,
        (StyleShape::Media, Property::Opacity) => 2,
    };
    path.extend_from_slice(&[11, field]);
    Ok(path)
}

/// Write one property into a style archive, refusing rather than inventing the
/// message it would have to live in.
fn set_property(
    document: &mut crate::Document,
    style: u64,
    property: Property,
    value: Option<Value>,
) -> Result<(), crate::Error> {
    let path = property_path(document, style, property)?;
    let mut archive = document.archive(style)?;
    crate::style::set_path(&mut archive, &path, value).map_err(|why| {
        crate::Error::refused(
            crate::Refusal::Missing,
            format!("style {style}: {why} (writing {property:?})"),
        )
    })?;

    // **`override_count` is not decoration.** `TSS` keeps the number of
    // properties a style sets locally, and a style whose bag has entries while
    // its count says zero is one the app treats as overriding nothing:
    // Keynote, handed a variation with a colour and a count of 0, saved the
    // document back with the properties stripped out. The count is maintained
    // here from the bag itself, so the two can never disagree.
    let bag = path[..path.len() - 1].to_vec();
    let overrides = crate::style::get_path(&archive, &bag)
        .and_then(|value| match value {
            Value::Bytes(raw) => decode_nested(&raw),
            _ => None,
        })
        .map(|properties| properties.fields.len() as u64)
        .unwrap_or(0);
    let mut count_path = bag[..bag.len() - 1].to_vec();
    count_path.push(10);
    let _ = crate::style::set_path(&mut archive, &count_path, Some(Value::Varint(overrides)));
    document.set_archive_for(style, &archive)
}

/// Build the `TSP.Color` every colour in the corpus is built as: model 1,
/// channels 3–6, and the two trailing fields iWork always writes with it.
pub(crate) fn colour_message(colour: Color) -> Message {
    let mut message = Message::default();
    message.set_in_order(1, Value::Varint(1));
    crate::style::set_channels(
        &mut message,
        colour.red,
        colour.green,
        colour.blue,
        colour.alpha,
    );
    message.set_in_order(12, Value::Varint(1));
    message.set_in_order(13, Value::Fixed32(1.0f32.to_le_bytes()));
    message
}

/// Which field of a drawable's concrete class holds its style reference.
fn style_field(kind: Kind) -> Option<u32> {
    match kind {
        Kind::Image => Some(image_field::STYLE),
        Kind::Movie => Some(19),
        Kind::Mask | Kind::Group | Kind::Table => None,
        _ => Some(2),
    }
}

/// How many drawables in the document point at this style.
fn users_of(document: &crate::Document, style: u64) -> usize {
    document
        .drawables()
        .iter()
        .filter(|d| d.style == Some(style))
        .count()
}

/// Whether a style archive is a *variation* — one that exists to hold one
/// object's changes and names the style it varies as its parent.
fn is_variation(document: &crate::Document, style: u64) -> bool {
    let Some((_, object)) = document.object(style) else {
        return false;
    };
    let base: Vec<u32> = match object.message_type() {
        TYPE_WP_SHAPE_STYLE => vec![1, 1, 3, 1],
        _ => vec![1, 3, 1],
    };
    document
        .archive(style)
        .ok()
        .and_then(|archive| reference_at(&archive, &base))
        .is_some()
}

/// The style to paint this drawable through, giving it one of its own if it
/// is currently sharing.
///
/// **Why a drawable cannot simply have its shared style painted.** A document
/// from nothing points every shape at a *theme preset* — `line-style-preset-0`
/// and its siblings — and a preset is the theme's, not the object's. Painting
/// one was measured: Keynote opened the deck, and on saving it wrote the
/// preset back out as a stub with the colour gone, because it regenerates its
/// presets from the theme.
///
/// What Keynote does instead is give the object its own archive naming the
/// preset as parent, carrying only what differs:
///
/// ```text
/// 1.1.3 -> the style it varies      1.1.4  1      (this is a variation)
/// 1.1.5 -> the stylesheet           1.10   2      (two properties differ)
/// 1.11  { 3: 0.5, 5: { 1: 0.4 } }                 (opacity and reflection)
/// ```
///
/// That is what this makes, once, the first time a drawable is painted.
fn own_style(document: &mut crate::Document, drawable: u64) -> Result<u64, crate::Error> {
    let Some(found) = document.drawable(drawable) else {
        return Err(crate::Error::refused(
            crate::Refusal::NotFound,
            format!("no drawable {drawable} in this document"),
        ));
    };
    let Some(current) = found.style else {
        return Err(crate::Error::refused(
            crate::Refusal::NotDrawn,
            format!(
                "drawable {drawable} is a {:?}, which has no object style to paint",
                found.kind
            ),
        ));
    };
    // Already its own, and nobody else's: paint it where it stands.
    if users_of(document, current) == 1 && is_variation(document, current) {
        return Ok(current);
    }

    let Some(field) = style_field(found.kind) else {
        return Err(crate::Error::refused(
            crate::Refusal::NotDrawn,
            format!("a {:?} carries no style reference", found.kind),
        ));
    };
    let Some((_, object)) = document.object(current) else {
        return Err(crate::Error::refused(
            crate::Refusal::NotFound,
            format!("style {current} is not in this document"),
        ));
    };
    let message_type = object.message_type();
    let wrapped = message_type == TYPE_WP_SHAPE_STYLE;

    // The stylesheet the parent belongs to, carried over rather than guessed.
    let parent_archive = document.archive(current)?;
    let stylesheet = reference_at(
        &parent_archive,
        if wrapped { &[1, 1, 5, 1] } else { &[1, 5, 1] },
    );

    // `super`: the parent, the variation flag, and the stylesheet.
    let mut sup = Message::default();
    sup.set_in_order(3, Value::Bytes(crate::style::reference(current).encode()));
    sup.set_in_order(4, Value::Varint(1));
    if let Some(stylesheet) = stylesheet {
        sup.set_in_order(
            5,
            Value::Bytes(crate::style::reference(stylesheet).encode()),
        );
    }

    let mut tsd = Message::default();
    tsd.set_in_order(1, Value::Bytes(sup.encode()));
    tsd.set_in_order(10, Value::Varint(0));
    // The bag starts empty: a variation carries only what differs.
    tsd.set_in_order(11, Value::Bytes(Message::default().encode()));

    let archive = if wrapped {
        let mut outer = Message::default();
        outer.set_in_order(1, Value::Bytes(tsd.encode()));
        outer
    } else {
        tsd
    };

    let mut grow = crate::create::Grow::new(document);
    let variation = grow.allocate();
    grow.beside(current, variation, message_type, &archive)?;
    grow.finish()?;

    // Point the drawable at it. The style sits on the concrete class, which is
    // the message one level above the drawable archive.
    let mut owner = document.archive(drawable)?;
    let mut path = found.path.clone();
    path.pop();
    path.push(field);
    crate::style::set_path(
        &mut owner,
        &path,
        Some(Value::Bytes(crate::style::reference(variation).encode())),
    )
    .map_err(|why| {
        crate::Error::refused(
            crate::Refusal::Missing,
            format!("drawable {drawable}: {why} (pointing at its own style)"),
        )
    })?;
    document.set_archive_for(drawable, &owner)?;
    // The variation lands beside the style it varies, in the stylesheet, and
    // the drawable is somewhere else — a slide, a sheet, a page. A reference
    // across components has to be declared or the app is handed an object it
    // was never told to load.
    document.declare_external_references();
    Ok(variation)
}

/// Paint a drawable, or with `None` leave it painting nothing.
///
/// The argument is the **drawable**, not its style. A document from nothing
/// points every shape at a shared theme preset, and painting a preset does
/// not survive: Keynote regenerates its presets on save and the colour is
/// gone. So the first paint gives the drawable a style of its own, naming the
/// preset as its parent and carrying only what differs, and paints that —
/// which is what the app does. A second paint reuses it.
pub fn set_fill(
    document: &mut crate::Document,
    drawable: u64,
    colour: Option<Color>,
) -> Result<(), crate::Error> {
    let style = own_style(document, drawable)?;
    let fill = match colour {
        None => Message::default(),
        Some(colour) => {
            let mut fill = Message::default();
            fill.set_in_order(1, Value::Bytes(colour_message(colour).encode()));
            fill
        }
    };
    set_property(
        document,
        style,
        Property::Fill,
        Some(Value::Bytes(fill.encode())),
    )
}

/// Outline a drawable: colour and width in points, solid.
///
/// A width of `0.0` is what iWork writes for "no outline"; the stroke archive
/// stays, because removing it altogether is a shape without the field every
/// other shape in the corpus carries.
pub fn set_stroke(
    document: &mut crate::Document,
    drawable: u64,
    colour: Color,
    width: f32,
) -> Result<(), crate::Error> {
    if !width.is_finite() || width < 0.0 {
        return Err(crate::Error::refused(
            crate::Refusal::UnwritableValue,
            format!("a stroke is {width} points wide"),
        ));
    }
    let style = own_style(document, drawable)?;
    let mut stroke = Message::default();
    stroke.set_in_order(1, Value::Bytes(colour_message(colour).encode()));
    stroke.set_in_order(2, Value::Fixed32(width.to_le_bytes()));
    // `{1: 1}` is the solid pattern, with no dash entries and no count.
    let mut pattern = Message::default();
    pattern.set_in_order(1, Value::Varint(1));
    stroke.set_in_order(6, Value::Bytes(pattern.encode()));
    set_property(
        document,
        style,
        Property::Stroke,
        Some(Value::Bytes(stroke.encode())),
    )
}

/// How opaque the drawable is, 0.0 to 1.0.
pub fn set_opacity(
    document: &mut crate::Document,
    drawable: u64,
    opacity: f32,
) -> Result<(), crate::Error> {
    if !(0.0..=1.0).contains(&opacity) {
        return Err(crate::Error::refused(
            crate::Refusal::UnwritableValue,
            format!("opacity is 0.0 to 1.0, not {opacity}"),
        ));
    }
    let style = own_style(document, drawable)?;
    set_property(
        document,
        style,
        Property::Opacity,
        Some(Value::Fixed32(opacity.to_le_bytes())),
    )
}
