//! Inline SVG charts and diagrams kept as drawings. A drawing that shows text
//! (axis labels, a diagram's boxes) carries its meaning in its markup; written
//! as text it would be loose labels ("Jan. Feb. March $0 $50"). As the
//! reference does, such a drawing is written as SVG markup in the Markdown,
//! with the paint its lost style sheets gave it: CSS variables, `light-dark()`
//! and Tailwind colour classes resolved to colours, and shapes that only a
//! style sheet painted given a visible fallback.
//!
//! The markup is rebuilt from an allow-list, never copied: drawing elements and
//! presentation attributes only, references only to fragments of the drawing
//! (`#id`), no scripts, event handlers, animations, images, style sheets or
//! filters, and plain formatting inside `foreignObject`; values call only
//! colour, transform and arithmetic functions, and every `id` (with the
//! references to it) gets a prefix of its own drawing, so two drawings cannot
//! take each other's gradients and an `id` cannot shadow a script's global.
//! It is written on one line, and its text escapes Markdown's punctuation as
//! character references, so a Markdown reader that does not take the line as
//! HTML still reads no links or emphasis in a label. A drawing nested deeper
//! than [`MAX_DEPTH`], with more than [`MAX_ELEMENTS`] elements or longer than
//! [`MAX_MARKUP`] is not kept.

use super::{ElementRef, escaped};
use scraper::Node;

/// The longest drawing kept, in bytes of rebuilt markup.
const MAX_MARKUP: usize = 512 * 1024;
/// The deepest element nesting a kept drawing has.
const MAX_DEPTH: usize = 64;
/// The most elements a kept drawing has.
const MAX_ELEMENTS: usize = 20_000;

/// Elements written with their content.
const DRAWING: &[&str] = &[
    "svg",
    "g",
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "text",
    "tspan",
    "textPath",
    "title",
    "desc",
    "defs",
    "linearGradient",
    "radialGradient",
    "stop",
    "clipPath",
    "mask",
    "pattern",
    "marker",
    "symbol",
    "use",
    "switch",
    "foreignObject",
];

/// Presentation and geometry attributes written as they are (after the checks
/// in [`safe_value`]); `style`, `class` and `href` are read separately.
const ATTRIBUTES: &[&str] = &[
    "viewBox",
    "width",
    "height",
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "fx",
    "fy",
    "fr",
    "d",
    "points",
    "pathLength",
    "transform",
    "preserveAspectRatio",
    "xmlns",
    "version",
    "role",
    "aria-label",
    "aria-hidden",
    "id",
    "fill",
    "fill-opacity",
    "fill-rule",
    "stroke",
    "stroke-width",
    "stroke-opacity",
    "stroke-dasharray",
    "stroke-dashoffset",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-miterlimit",
    "opacity",
    "color",
    "visibility",
    "display",
    "vector-effect",
    "shape-rendering",
    "text-rendering",
    "paint-order",
    "clip-path",
    "clip-rule",
    "mask",
    "marker-start",
    "marker-mid",
    "marker-end",
    "text-anchor",
    "dominant-baseline",
    "alignment-baseline",
    "baseline-shift",
    "font-size",
    "font-family",
    "font-weight",
    "font-style",
    "font-variant",
    "letter-spacing",
    "word-spacing",
    "text-decoration",
    "writing-mode",
    "dx",
    "dy",
    "rotate",
    "textLength",
    "lengthAdjust",
    "startOffset",
    "side",
    "offset",
    "stop-color",
    "stop-opacity",
    "flood-color",
    "lighting-color",
    "gradientUnits",
    "gradientTransform",
    "spreadMethod",
    "patternUnits",
    "patternContentUnits",
    "patternTransform",
    "markerWidth",
    "markerHeight",
    "markerUnits",
    "refX",
    "refY",
    "orient",
    "clipPathUnits",
    "maskUnits",
    "maskContentUnits",
    "systemLanguage",
];

/// CSS properties kept in a `style` attribute.
const PROPERTIES: &[&str] = &[
    "fill",
    "fill-opacity",
    "fill-rule",
    "stroke",
    "stroke-width",
    "stroke-opacity",
    "stroke-dasharray",
    "stroke-dashoffset",
    "stroke-linecap",
    "stroke-linejoin",
    "opacity",
    "color",
    "visibility",
    "display",
    "font",
    "font-size",
    "font-family",
    "font-weight",
    "font-style",
    "letter-spacing",
    "text-anchor",
    "text-align",
    "text-decoration",
    "dominant-baseline",
    "line-height",
    "white-space",
    "transform",
    "transform-origin",
    "background-color",
    "stop-color",
    "stop-opacity",
];

/// HTML elements kept inside a `foreignObject`; others write only their
/// content, and [`LEFT_OUT`] ones nothing.
const FORMATTING: &[&str] = &[
    "div", "span", "p", "br", "b", "strong", "i", "em", "code", "sub", "sup", "small", "u", "s",
    "ul", "ol", "li",
];

/// Elements left out with their content, in the drawing or inside it.
const LEFT_OUT: &[&str] = &[
    "script", "style", "noscript", "template", "iframe", "object", "embed", "img", "image",
    "video", "audio", "canvas", "math", "input", "button", "select", "textarea", "form",
];

const FILLED: &[&str] = &["path", "rect", "circle", "ellipse", "polygon"];
const STROKED: &[&str] = &["line", "polyline"];
const NOT_RENDERED: &[&str] = &["defs", "clipPath", "mask", "pattern", "marker"];
const COLOR_ATTRIBUTES: &[&str] = &[
    "fill",
    "stroke",
    "color",
    "stop-color",
    "flood-color",
    "lighting-color",
];

/// What one bounded walk over a drawing finds before it is written.
struct Survey {
    /// Text in a `text` or `foreignObject` element: the reference keeps only
    /// such a drawing. One without (a logo, an icon, a picture) says nothing
    /// as markup in Markdown.
    labels: bool,
    /// A filled shape with a class the palette does not resolve and no fill
    /// of its own, outside definitions: a style sheet painted the drawing, so
    /// the fallbacks of [`fallback`] apply.
    lost: bool,
    /// The prefix of the drawing's `id`s, from a hash of its markup.
    prefix: String,
}

/// One walk over the drawing, in document order, that gives up past
/// [`MAX_DEPTH`] or [`MAX_ELEMENTS`] before anything else reads it.
fn survey(svg: ElementRef<'_>) -> Option<Survey> {
    let mut found = Survey {
        labels: false,
        lost: false,
        prefix: String::new(),
    };
    // FNV-1a over names, attributes and text.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    };
    let mut elements = 0;
    // (node, depth, inside a label, inside definitions)
    let mut stack = vec![(*svg, 0, false, false)];
    while let Some((node, depth, label, defs)) = stack.pop() {
        match node.value() {
            Node::Text(text) => {
                found.labels |= label && !text.trim().is_empty();
                mix(text.as_bytes());
            }
            Node::Element(value) => {
                elements += 1;
                if depth > MAX_DEPTH || elements > MAX_ELEMENTS {
                    return None;
                }
                let name = value.name();
                mix(name.as_bytes());
                for (key, attribute) in value.attrs() {
                    mix(key.as_bytes());
                    mix(attribute.as_bytes());
                }
                if !found.lost
                    && !defs
                    && FILLED.contains(&name)
                    && let Some(shape) = ElementRef::wrap(node)
                {
                    let paint = paint(shape, false, "");
                    found.lost =
                        paint.class_left && paint.get("fill").is_none() && !paint.style_has("fill");
                }
                let label = label || matches!(name, "text" | "foreignObject");
                let defs = defs || NOT_RENDERED.contains(&name);
                for child in node.children().rev() {
                    stack.push((child, depth + 1, label, defs));
                }
            }
            _ => {}
        }
    }
    found.prefix = format!("m{:08x}-", hash >> 32);
    Some(found)
}

/// Whether a drawing stands in the flow of blocks: every element around it,
/// up to the body, is a block container, so its markup is a block of its own
/// in Markdown, never inside a table cell, a link, a heading or a sentence.
pub(super) fn in_block_flow(svg: ElementRef<'_>) -> bool {
    svg.ancestors().filter_map(ElementRef::wrap).all(|parent| {
        matches!(
            parent.value().name(),
            "html"
                | "body"
                | "article"
                | "main"
                | "section"
                | "div"
                | "figure"
                | "header"
                | "footer"
                | "aside"
                | "blockquote"
                | "dl"
                | "dd"
                | "details"
                | "ul"
                | "ol"
                | "li"
                | "center"
        )
    })
}

/// The drawing as one line of sanitized SVG markup, or `None` for a drawing
/// that shows no text, nests deeper than [`MAX_DEPTH`], has more than
/// [`MAX_ELEMENTS`] elements or is longer than [`MAX_MARKUP`].
pub(super) fn markup(svg: ElementRef<'_>) -> Option<String> {
    let survey = survey(svg)?;
    if !survey.labels {
        return None;
    }
    let mut output = String::new();
    let context = Context {
        lost: survey.lost,
        prefix: &survey.prefix,
    };
    write(svg, &mut output, &context, false, false, false, 0)?;
    Some(output)
}

/// What every element of one drawing is written with.
struct Context<'a> {
    lost: bool,
    prefix: &'a str,
}

/// An element's attributes as they will be written, and whether classes the
/// palette does not resolve remain (styling the drawing lost with its sheets).
struct Paint {
    attributes: Vec<(String, String)>,
    class_left: bool,
}

impl Paint {
    fn get(&self, name: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    fn set(&mut self, name: &str, value: String) {
        match self.attributes.iter_mut().find(|(key, _)| key == name) {
            Some((_, old)) => *old = value,
            None => self.attributes.push((name.to_owned(), value)),
        }
    }

    fn style_has(&self, property: &str) -> bool {
        self.get("style").is_some_and(|style| {
            style.split(';').any(|declaration| {
                declaration
                    .split_once(':')
                    .is_some_and(|(key, _)| key.trim().eq_ignore_ascii_case(property))
            })
        })
    }
}

/// CSS functions a kept value may call: colours, transforms and arithmetic.
/// Anything else (`image-set()`, `image()`, `cross-fade()`, `src()`,
/// `element()`, `expression()`) could load or run something.
const FUNCTIONS: &[&str] = &[
    "",
    "url",
    "rgb",
    "rgba",
    "hsl",
    "hsla",
    "hwb",
    "lab",
    "lch",
    "oklab",
    "oklch",
    "color",
    "var",
    "light-dark",
    "translate",
    "translatex",
    "translatey",
    "rotate",
    "scale",
    "scalex",
    "scaley",
    "matrix",
    "skewx",
    "skewy",
    "calc",
    "min",
    "max",
    "clamp",
];

/// Whether an attribute value is plain drawing data: no script or data
/// address, import, escape, function outside [`FUNCTIONS`], or `url()` other
/// than a reference to a fragment of the drawing.
fn safe_value(value: &str) -> bool {
    let compact: String = value
        .chars()
        .filter(|character| !character.is_whitespace() && !character.is_control())
        .flat_map(char::to_lowercase)
        .collect();
    if ["javascript:", "vbscript:", "data:", "@import", "\\", "<"]
        .iter()
        .any(|bad| compact.contains(bad))
    {
        return false;
    }
    let bytes = compact.as_bytes();
    compact.match_indices('(').all(|(at, _)| {
        let start = bytes[..at]
            .iter()
            .rposition(|byte| !(byte.is_ascii_alphanumeric() || *byte == b'-'))
            .map_or(0, |before| before + 1);
        let function = &compact[start..at];
        FUNCTIONS.contains(&function)
            && (function != "url"
                || compact[at + 1..]
                    .trim_start_matches(['"', '\''])
                    .starts_with('#'))
    })
}

/// A value with every `url(#a)` reference to the drawing given the drawing's
/// prefix, as its `id`s are.
fn local_references(value: &str, prefix: &str) -> String {
    let lower = value.to_ascii_lowercase();
    let mut output = String::with_capacity(value.len() + prefix.len());
    let mut copied = 0;
    for (at, _) in lower.match_indices("url(") {
        let inside = at + 4;
        let hash = value[inside..]
            .find(|character: char| !(character.is_whitespace() || matches!(character, '"' | '\'')))
            .map(|offset| inside + offset)
            .filter(|position| value[*position..].starts_with('#'));
        if let Some(hash) = hash {
            output.push_str(&value[copied..=hash]);
            output.push_str(prefix);
            copied = hash + 1;
        }
    }
    output.push_str(&value[copied..]);
    output
}

/// Tailwind's default palette: (name, the hex colours of shades 50, 100, 200
/// … 900, 950), from Tailwind CSS 3.4.19 (MIT, Copyright (c) Tailwind Labs,
/// Inc.). Its license and source identity are in `licenses/tailwind/`, whose
/// digest of these values `scripts/tailwind_attribution.py` checks.
const PALETTE: &[(&str, [&str; 11])] = &[
    (
        "slate",
        [
            "#f8fafc", "#f1f5f9", "#e2e8f0", "#cbd5e1", "#94a3b8", "#64748b", "#475569", "#334155",
            "#1e293b", "#0f172a", "#020617",
        ],
    ),
    (
        "gray",
        [
            "#f9fafb", "#f3f4f6", "#e5e7eb", "#d1d5db", "#9ca3af", "#6b7280", "#4b5563", "#374151",
            "#1f2937", "#111827", "#030712",
        ],
    ),
    (
        "zinc",
        [
            "#fafafa", "#f4f4f5", "#e4e4e7", "#d4d4d8", "#a1a1aa", "#71717a", "#52525b", "#3f3f46",
            "#27272a", "#18181b", "#09090b",
        ],
    ),
    (
        "neutral",
        [
            "#fafafa", "#f5f5f5", "#e5e5e5", "#d4d4d4", "#a3a3a3", "#737373", "#525252", "#404040",
            "#262626", "#171717", "#0a0a0a",
        ],
    ),
    (
        "stone",
        [
            "#fafaf9", "#f5f5f4", "#e7e5e4", "#d6d3d1", "#a8a29e", "#78716c", "#57534e", "#44403c",
            "#292524", "#1c1917", "#0c0a09",
        ],
    ),
    (
        "red",
        [
            "#fef2f2", "#fee2e2", "#fecaca", "#fca5a5", "#f87171", "#ef4444", "#dc2626", "#b91c1c",
            "#991b1b", "#7f1d1d", "#450a0a",
        ],
    ),
    (
        "orange",
        [
            "#fff7ed", "#ffedd5", "#fed7aa", "#fdba74", "#fb923c", "#f97316", "#ea580c", "#c2410c",
            "#9a3412", "#7c2d12", "#431407",
        ],
    ),
    (
        "amber",
        [
            "#fffbeb", "#fef3c7", "#fde68a", "#fcd34d", "#fbbf24", "#f59e0b", "#d97706", "#b45309",
            "#92400e", "#78350f", "#451a03",
        ],
    ),
    (
        "yellow",
        [
            "#fefce8", "#fef9c3", "#fef08a", "#fde047", "#facc15", "#eab308", "#ca8a04", "#a16207",
            "#854d0e", "#713f12", "#422006",
        ],
    ),
    (
        "lime",
        [
            "#f7fee7", "#ecfccb", "#d9f99d", "#bef264", "#a3e635", "#84cc16", "#65a30d", "#4d7c0f",
            "#3f6212", "#365314", "#1a2e05",
        ],
    ),
    (
        "green",
        [
            "#f0fdf4", "#dcfce7", "#bbf7d0", "#86efac", "#4ade80", "#22c55e", "#16a34a", "#15803d",
            "#166534", "#14532d", "#052e16",
        ],
    ),
    (
        "emerald",
        [
            "#ecfdf5", "#d1fae5", "#a7f3d0", "#6ee7b7", "#34d399", "#10b981", "#059669", "#047857",
            "#065f46", "#064e3b", "#022c22",
        ],
    ),
    (
        "teal",
        [
            "#f0fdfa", "#ccfbf1", "#99f6e4", "#5eead4", "#2dd4bf", "#14b8a6", "#0d9488", "#0f766e",
            "#115e59", "#134e4a", "#042f2e",
        ],
    ),
    (
        "cyan",
        [
            "#ecfeff", "#cffafe", "#a5f3fc", "#67e8f9", "#22d3ee", "#06b6d4", "#0891b2", "#0e7490",
            "#155e75", "#164e63", "#083344",
        ],
    ),
    (
        "sky",
        [
            "#f0f9ff", "#e0f2fe", "#bae6fd", "#7dd3fc", "#38bdf8", "#0ea5e9", "#0284c7", "#0369a1",
            "#075985", "#0c4a6e", "#082f49",
        ],
    ),
    (
        "blue",
        [
            "#eff6ff", "#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8",
            "#1e40af", "#1e3a8a", "#172554",
        ],
    ),
    (
        "indigo",
        [
            "#eef2ff", "#e0e7ff", "#c7d2fe", "#a5b4fc", "#818cf8", "#6366f1", "#4f46e5", "#4338ca",
            "#3730a3", "#312e81", "#1e1b4b",
        ],
    ),
    (
        "violet",
        [
            "#f5f3ff", "#ede9fe", "#ddd6fe", "#c4b5fd", "#a78bfa", "#8b5cf6", "#7c3aed", "#6d28d9",
            "#5b21b6", "#4c1d95", "#2e1065",
        ],
    ),
    (
        "purple",
        [
            "#faf5ff", "#f3e8ff", "#e9d5ff", "#d8b4fe", "#c084fc", "#a855f7", "#9333ea", "#7e22ce",
            "#6b21a8", "#581c87", "#3b0764",
        ],
    ),
    (
        "fuchsia",
        [
            "#fdf4ff", "#fae8ff", "#f5d0fe", "#f0abfc", "#e879f9", "#d946ef", "#c026d3", "#a21caf",
            "#86198f", "#701a75", "#4a044e",
        ],
    ),
    (
        "pink",
        [
            "#fdf2f8", "#fce7f3", "#fbcfe8", "#f9a8d4", "#f472b6", "#ec4899", "#db2777", "#be185d",
            "#9d174d", "#831843", "#500724",
        ],
    ),
    (
        "rose",
        [
            "#fff1f2", "#ffe4e6", "#fecdd3", "#fda4af", "#fb7185", "#f43f5e", "#e11d48", "#be123c",
            "#9f1239", "#881337", "#4c0519",
        ],
    ),
];

/// A palette colour by name and shade (`amber`, `600`).
fn palette(name: &str, shade: &str) -> Option<&'static str> {
    let index = match shade {
        "50" => 0,
        "100" => 1,
        "200" => 2,
        "300" => 3,
        "400" => 4,
        "500" => 5,
        "600" => 6,
        "700" => 7,
        "800" => 8,
        "900" => 9,
        "950" => 10,
        _ => return None,
    };
    PALETTE
        .iter()
        .find(|(color, _)| *color == name)
        .map(|(_, shades)| shades[index])
}

/// `light-dark(a, b)` as `a`, the colour a light page shows.
fn light(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(at) = rest.find("light-dark(") {
        output.push_str(&rest[..at]);
        let inner = &rest[at + "light-dark(".len()..];
        // The first argument ends at its top-level comma.
        let mut depth = 0usize;
        let mut comma = None;
        let mut close = None;
        for (index, character) in inner.char_indices() {
            match character {
                '(' => depth += 1,
                ')' if depth == 0 => {
                    close = Some(index);
                    break;
                }
                ')' => depth -= 1,
                ',' if depth == 0 && comma.is_none() => comma = Some(index),
                _ => {}
            }
        }
        let (Some(comma), Some(close)) = (comma, close) else {
            output.push_str(&rest[at..]);
            return output;
        };
        output.push_str(inner[..comma].trim());
        rest = &inner[close + 1..];
    }
    output.push_str(rest);
    output
}

/// The `var(…)` calls of a value: where each starts and ends (after its
/// closing parenthesis, matched by depth), its variable's name and fallback.
fn variables(value: &str) -> Vec<(usize, usize, String, Option<String>)> {
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(offset) = value[from..].find("var(") {
        let start = from + offset;
        let inner = start + 4;
        let mut depth = 0usize;
        let mut comma = None;
        let mut close = None;
        for (index, character) in value[inner..].char_indices() {
            match character {
                '(' => depth += 1,
                ')' if depth == 0 => {
                    close = Some(inner + index);
                    break;
                }
                ')' => depth -= 1,
                ',' if depth == 0 && comma.is_none() => comma = Some(inner + index),
                _ => {}
            }
        }
        let Some(close) = close else {
            break;
        };
        let (name, fallback) = match comma {
            Some(comma) => (
                &value[inner..comma],
                Some(value[comma + 1..close].trim().to_owned()).filter(|f| !f.is_empty()),
            ),
            None => (&value[inner..close], None),
        };
        found.push((
            start,
            close + 1,
            name.trim().trim_start_matches("--").to_ascii_lowercase(),
            fallback,
        ));
        from = close + 1;
    }
    found
}

/// Every `var(…)` of a value replaced by its fallback, or `None` when one has
/// none: without the page's style sheets, only the author's fallback is known.
fn fallbacks(value: &str) -> Option<String> {
    let mut output = String::with_capacity(value.len());
    let mut copied = 0;
    for (start, end, _, fallback) in variables(value) {
        output.push_str(&value[copied..start]);
        output.push_str(&fallbacks(&fallback?)?);
        copied = end;
    }
    output.push_str(&value[copied..]);
    Some(output)
}

/// A colour that names CSS variables, without the page's style sheets. A value
/// that is one variable is its fallback, a Tailwind palette variable
/// (`--color-amber-600`), black or white by name, then a guess by name (a
/// background is `Canvas`, a border `#ccc`, a muted colour `#888`), else
/// `currentColor`; a value that uses variables inside it (`rgb(var(--r) 0 0)`)
/// takes their fallbacks, else it is `currentColor`.
fn resolve_color(value: &str) -> String {
    let value = light(value);
    let calls = variables(&value);
    let Some((start, end, name, fallback)) = calls.first() else {
        return value;
    };
    if calls.len() > 1 || value[..*start].trim().len() + value[*end..].trim().len() > 0 {
        return fallbacks(&value).unwrap_or_else(|| "currentColor".into());
    }
    if let Some(fallback) = fallback {
        return resolve_color(fallback);
    }
    if let Some((rest, shade)) = name.rsplit_once('-')
        && let Some(color) = rest.rsplit('-').next()
        && let Some(hex) = palette(color, shade)
    {
        return hex.to_owned();
    }
    if name.ends_with("-black") || name == "black" {
        return "#000".into();
    }
    if name.ends_with("-white") || name == "white" {
        return "#fff".into();
    }
    let has = |words: &[&str]| words.iter().any(|word| name.contains(word));
    if has(&["background", "card", "surface", "bg"]) {
        "Canvas".into()
    } else if has(&["border", "divider", "separator"]) {
        "#ccc".into()
    } else if has(&["muted", "subtle", "secondary", "placeholder"]) {
        "#888".into()
    } else {
        "currentColor".into()
    }
}

/// A Tailwind class as paint: `fill-amber-500` (with `/50` opacity),
/// `stroke-black`, `text-[14px]`, `font-semibold`. `Ok` is an attribute,
/// `Err` a style declaration.
fn tailwind(class: &str) -> Option<Result<(&'static str, String), String>> {
    for (prefix, property) in [("fill-", "fill"), ("stroke-", "stroke")] {
        let Some(rest) = class.strip_prefix(prefix) else {
            continue;
        };
        let special = match rest {
            "black" => Some("#000"),
            "white" => Some("#fff"),
            "transparent" => Some("transparent"),
            "current" => Some("currentColor"),
            _ => None,
        };
        if let Some(color) = special {
            return Some(Ok((property, color.to_owned())));
        }
        let (shaded, opacity) = match rest.split_once('/') {
            Some((shaded, opacity)) => (shaded, Some(opacity)),
            None => (rest, None),
        };
        let (name, shade) = shaded.split_once('-')?;
        if !name.bytes().all(|b| b.is_ascii_lowercase()) {
            return None;
        }
        let hex = palette(name, shade)?;
        return Some(Ok((
            property,
            match opacity {
                Some(opacity) => {
                    let alpha = opacity.parse::<u32>().ok().filter(|_| {
                        !opacity.is_empty() && opacity.bytes().all(|b| b.is_ascii_digit())
                    })?;
                    let channel = |at: usize| u8::from_str_radix(&hex[at..at + 2], 16).unwrap_or(0);
                    format!(
                        "rgba({},{},{},{})",
                        channel(1),
                        channel(3),
                        channel(5),
                        f64::from(alpha) / 100.0
                    )
                }
                None => hex.to_owned(),
            },
        )));
    }
    if let Some(size) = class
        .strip_prefix("text-[")
        .and_then(|rest| rest.strip_suffix(']'))
        .filter(|size| !size.is_empty() && !["#", "rgb", "hsl"].iter().any(|p| size.starts_with(p)))
    {
        return Some(Err(format!("font-size:{size}")));
    }
    let font = match class {
        "font-semibold" => "font-weight:600",
        "font-bold" => "font-weight:700",
        "font-medium" => "font-weight:500",
        "font-mono" => "font-family:monospace",
        _ => return None,
    };
    Some(Err(font.to_owned()))
}

/// The declarations of a style value that are kept: allowed properties with
/// safe values, a colour's `var()` and `light-dark()` resolved, another
/// property's variables replaced by their fallbacks (without one, the
/// declaration is dropped), and references given the drawing's prefix.
fn kept_style(style: &str, prefix: &str) -> String {
    let mut kept: Vec<String> = Vec::new();
    for declaration in style.split(';') {
        let Some((property, value)) = declaration.split_once(':') else {
            continue;
        };
        let property = property.trim().to_ascii_lowercase();
        if !PROPERTIES.contains(&property.as_str()) {
            continue;
        }
        let value = value.trim();
        let value =
            if COLOR_ATTRIBUTES.contains(&property.as_str()) || property == "background-color" {
                Some(resolve_color(value))
            } else {
                fallbacks(&light(value))
            };
        if let Some(value) = value.filter(|value| !value.is_empty() && safe_value(value)) {
            kept.push(format!("{property}:{}", local_references(&value, prefix)));
        }
    }
    kept.join(";")
}

fn paint(element: ElementRef<'_>, html: bool, prefix: &str) -> Paint {
    let mut paint = Paint {
        attributes: Vec::new(),
        class_left: false,
    };
    let mut classes = "";
    let mut style = String::new();
    for (name, value) in element.value().attrs() {
        match name {
            "class" => classes = value,
            "style" => style = value.to_owned(),
            "href" if !html && value.trim().starts_with('#') && safe_value(value) => {
                let fragment = value.trim().trim_start_matches('#');
                paint.set("href", format!("#{prefix}{fragment}"));
            }
            _ if html => {}
            "id" => paint.set("id", format!("{prefix}{}", value.trim())),
            _ if ATTRIBUTES.contains(&name) && safe_value(value) => {
                let value = if COLOR_ATTRIBUTES.contains(&name)
                    && (value.contains("var(") || value.contains("light-dark("))
                {
                    resolve_color(value)
                } else {
                    value.to_owned()
                };
                if safe_value(&value) {
                    paint.set(name, local_references(&value, prefix));
                }
            }
            _ => {}
        }
    }
    let mut extra: Vec<String> = Vec::new();
    for class in classes.split_ascii_whitespace() {
        match tailwind(class).filter(|_| !html) {
            Some(Ok((property, color))) => paint.set(property, color),
            Some(Err(declaration)) => extra.push(declaration),
            None => paint.class_left = true,
        }
    }
    if !extra.is_empty() {
        if !style.trim().is_empty() && !style.trim_end().ends_with(';') {
            style.push(';');
        }
        style.push_str(&extra.join(";"));
    }
    let style = kept_style(&style, prefix);
    if !style.is_empty() {
        paint.set("style", style);
    }
    paint
}

/// Visible paint for a shape a lost style sheet painted: text in the current
/// colour, a filled shape without fill unfilled, a line or an open path drawn
/// in the current colour (a grid line faintly).
fn fallback(element: ElementRef<'_>, paint: &mut Paint, defs: bool) {
    let name = element.value().name();
    if !paint.class_left || defs {
        return;
    }
    if matches!(name, "text" | "tspan") {
        if paint.get("fill").is_none() && !paint.style_has("fill") {
            paint.set("fill", "currentColor".into());
        }
        return;
    }
    let filled = FILLED.contains(&name);
    let stroked = STROKED.contains(&name);
    if !filled && !stroked {
        return;
    }
    let has_fill = paint.get("fill").is_some_and(|fill| fill != "none");
    let has_stroke = paint.get("stroke").is_some() || paint.style_has("stroke");
    if filled && paint.get("fill").is_none() && !paint.style_has("fill") {
        paint.set("fill", "none".into());
    }
    if has_stroke {
        return;
    }
    if stroked {
        paint.set("stroke", "currentColor".into());
        if paint.get("stroke-opacity").is_none() {
            paint.set("stroke-opacity", "0.2".into());
        }
    } else if filled && !has_fill {
        let closed = paint
            .get("d")
            .is_some_and(|d| d.trim_end().ends_with(['z', 'Z']));
        if !closed {
            paint.set("stroke", "currentColor".into());
        }
    }
}

/// An attribute value on one line, quoted.
fn escaped_attribute(value: &str, output: &mut String) {
    let line = value
        .split(|character: char| character.is_whitespace() || character.is_control())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    escaped(&line, output);
}

/// Text on one line, its runs of white space one space, and the punctuation
/// Markdown reads (`[`, `*`, `_`, a backtick …) as character references.
fn text(value: &str, output: &mut String) {
    let mut collapsed = String::with_capacity(value.len());
    let mut space = false;
    for character in value.chars() {
        if character.is_whitespace() || character.is_control() {
            space = true;
        } else {
            if space {
                collapsed.push(' ');
                space = false;
            }
            collapsed.push(character);
        }
    }
    if space {
        collapsed.push(' ');
    }
    let mut html = String::with_capacity(collapsed.len());
    escaped(&collapsed, &mut html);
    for character in html.chars() {
        match character {
            '\\' | '`' | '*' | '_' | '[' | ']' | '~' | '|' | '$' => {
                output.push_str(&format!("&#{};", u32::from(character)));
            }
            character => output.push(character),
        }
    }
}

/// Write an element of the drawing (or, `html`, of a `foreignObject`'s
/// content); `None` past the limits.
fn write(
    element: ElementRef<'_>,
    output: &mut String,
    context: &Context<'_>,
    html: bool,
    in_text: bool,
    defs: bool,
    depth: usize,
) -> Option<()> {
    if depth > MAX_DEPTH || output.len() > MAX_MARKUP {
        return None;
    }
    let name = element.value().name();
    if LEFT_OUT.contains(&name)
        || name.starts_with("fe") && !html
        || name.starts_with("animate")
        || name == "set"
    {
        return Some(());
    }
    let (kept, html_inside) = if html {
        (FORMATTING.contains(&name), true)
    } else if name == "foreignObject" {
        (true, true)
    } else if DRAWING.contains(&name) {
        (true, false)
    } else if name == "a" {
        // A link's drawing is drawn; the link is not followed.
        (false, false)
    } else {
        // Anything else draws nothing.
        return Some(());
    };
    let in_text =
        in_text || html || matches!(name, "text" | "tspan" | "textPath" | "title" | "desc");
    if kept {
        let mut paint = paint(element, html, context.prefix);
        if context.lost && !html {
            fallback(element, &mut paint, defs);
        }
        output.push('<');
        output.push_str(name);
        for (key, value) in &paint.attributes {
            output.push(' ');
            output.push_str(key);
            output.push_str("=\"");
            escaped_attribute(value, output);
            output.push('"');
        }
        output.push('>');
        if name == "br" {
            return Some(());
        }
    }
    let defs = defs || NOT_RENDERED.contains(&name);
    for child in element.children() {
        match child.value() {
            Node::Element(_) => {
                if let Some(child) = ElementRef::wrap(child) {
                    write(
                        child,
                        output,
                        context,
                        html_inside,
                        in_text,
                        defs,
                        depth + 1,
                    )?;
                }
            }
            Node::Text(value) if in_text => text(value, output),
            _ => {}
        }
    }
    if kept {
        output.push_str("</");
        output.push_str(name);
        output.push('>');
    }
    (output.len() <= MAX_MARKUP).then_some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scraper::{Html, Selector};

    /// The kept markup, its drawing's `id` prefix written `P-`.
    fn kept(source: &str) -> Option<String> {
        let html = Html::parse_fragment(source);
        let svg = html
            .select(&Selector::parse("svg").unwrap())
            .next()
            .unwrap();
        let prefix = survey(svg).map(|survey| survey.prefix).unwrap_or_default();
        markup(svg).map(|markup| markup.replace(&prefix, "P-"))
    }

    #[test]
    fn a_chart_keeps_its_drawing_on_one_line_and_an_icon_is_not_kept() {
        assert_eq!(
            kept(
                "<svg viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'>\n  <circle cx='50' cy='50' r='40' fill='none' stroke='black' />\n  <text x='50' y='55'>\n   Node  label </text>\n</svg>"
            )
            .as_deref(),
            Some(
                r#"<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" fill="none" r="40" stroke="black"></circle><text x="50" y="55"> Node label </text></svg>"#
            )
        );
        assert_eq!(
            kept("<svg viewBox='0 0 24 24'><path d='M0 0h24'/></svg>"),
            None
        );
        assert_eq!(kept("<svg><title>Logo</title><path d='M0 0'/></svg>"), None);
    }

    #[test]
    fn scripts_events_links_images_and_styles_are_not_written() {
        let markup = kept(
            r##"<svg onload="alert(1)" viewBox="0 0 10 10"><script>alert(2)</script><style>p{display:none}</style>
            <a href="javascript:alert(3)"><text x="1" onclick="alert(4)">Label</text></a>
            <image href="https://tracker.example/p.png"/><use href="#shape"/><use href="https://example.com/x.svg#a"/>
            <set attributeName="href" to="javascript:alert(5)"/><animate attributeName="x" values="0;1"/>
            <rect fill="url(https://example.com/x)" stroke="url(#g)" style="fill:red;background:url(https://e.x/a);behavior:x"/>
            <foreignObject><div style="color:blue" onclick="x()"><img src="x" onerror="y()">Box <b>bold</b><iframe src="https://e.x"></iframe></div></foreignObject></svg>"##,
        )
        .unwrap();
        assert_eq!(
            markup,
            r##"<svg viewBox="0 0 10 10"><text x="1">Label</text><use href="#P-shape"></use><use></use><rect stroke="url(#P-g)" style="fill:red"></rect><foreignObject><div style="color:blue">Box <b>bold</b></div></foreignObject></svg>"##
        );
    }

    #[test]
    fn variables_light_dark_and_tailwind_classes_become_colors() {
        let markup = kept(
            r##"<svg viewBox="0 0 10 10"><circle fill="var(--background-color-card)" stroke="var(--color-amber-600)"/>
            <path stroke="light-dark(var(--color-slate-400), var(--color-slate-300))" d="M0 0"/>
            <line class="stroke-zinc-400" stroke-width="1.5"/><circle class="fill-orange-500/50"/>
            <rect fill="var(--missing, #123456)" style="stroke: var(--border-color)"/>
            <text class="fill-amber-500 text-[14px] font-semibold">-0.50</text></svg>"##,
        )
        .unwrap();
        assert_eq!(
            markup,
            r##"<svg viewBox="0 0 10 10"><circle fill="Canvas" stroke="#d97706"></circle><path d="M0 0" stroke="#94a3b8"></path><line stroke-width="1.5" stroke="#a1a1aa"></line><circle fill="rgba(249,115,22,0.5)"></circle><rect fill="#123456" style="stroke:#ccc"></rect><text fill="#f59e0b" style="font-size:14px;font-weight:600">-0.50</text></svg>"##
        );
    }

    #[test]
    fn shapes_a_lost_style_sheet_painted_get_a_visible_fallback() {
        let markup = kept(
            r#"<svg viewBox="0 0 400 200"><g class="axis"><line class="gridline" x1="0" x2="350"/><text x="0" y="27">Jan.</text></g>
            <path class="area" d="M0,120L350,50 L350,170 Z"/><path class="line" d="M0,120L350,50"/>
            <defs><clipPath id="c"><rect class="clip" width="1"/></clipPath></defs><text class="label">t</text></svg>"#,
        )
        .unwrap();
        assert_eq!(
            markup,
            r#"<svg viewBox="0 0 400 200"><g><line x1="0" x2="350" stroke="currentColor" stroke-opacity="0.2"></line><text x="0" y="27">Jan.</text></g><path d="M0,120L350,50 L350,170 Z" fill="none"></path><path d="M0,120L350,50" fill="none" stroke="currentColor"></path><defs><clipPath id="P-c"><rect width="1"></rect></clipPath></defs><text fill="currentColor">t</text></svg>"#
        );
        // Paint of its own leaves a drawing as it is.
        assert_eq!(
            kept(r#"<svg><rect class="bar" fill="blue"/><text class="label">t</text></svg>"#)
                .as_deref(),
            Some(r#"<svg><rect fill="blue"></rect><text>t</text></svg>"#)
        );
    }

    #[test]
    fn label_text_reads_as_no_markdown_and_values_call_no_loading_function() {
        // Markdown's punctuation in a label is a character reference: no link,
        // picture or emphasis even where a reader takes the line as a paragraph.
        assert_eq!(
            kept(r#"<svg><text>[click](javascript:alert(1)) ![x](https://t.example/p.png) *up* `c` a_b | $5 ~</text></svg>"#)
                .as_deref(),
            Some(
                "<svg><text>&#91;click&#93;(javascript:alert(1)) !&#91;x&#93;(https://t.example/p.png) &#42;up&#42; &#96;c&#96; a&#95;b &#124; &#36;5 &#126;</text></svg>"
            )
        );
        // A mask, fill or style that names an image by another function loads
        // it; only colour, transform and arithmetic functions are kept.
        assert_eq!(
            kept(
                r#"<svg><rect mask="image-set('https://t.example/m.png' 1x)" fill="-webkit-image-set(&quot;https://t.example/f.png&quot; 1x)" stroke="rgb(1, 2, 3)" transform="translate(2 3) rotate(45)" style="fill:cross-fade(url(#a), url(#b));stroke:hsl(10 20% 30%)"/><text>t</text></svg>"#
            )
            .as_deref(),
            Some(
                r#"<svg><rect stroke="rgb(1, 2, 3)" transform="translate(2 3) rotate(45)" style="stroke:hsl(10 20% 30%)"></rect><text>t</text></svg>"#
            )
        );
    }

    #[test]
    fn variables_match_their_parentheses() {
        assert_eq!(resolve_color("var(--x, rgb(1,2,3))"), "rgb(1,2,3)");
        assert_eq!(resolve_color("var(--x, var(--y, #123))"), "#123");
        assert_eq!(resolve_color("rgb(var(--r, 10) 0 0)"), "rgb(10 0 0)");
        assert_eq!(resolve_color("rgb(var(--r) 0 0)"), "currentColor");
        assert_eq!(resolve_color("var(--color-amber-600)"), "#d97706");
        assert_eq!(resolve_color("var(--x"), "var(--x");
        // Another property's variable needs a fallback, else it is dropped.
        assert_eq!(
            kept_style("font-size: var(--s, 12px); opacity: var(--o)", "P-"),
            "font-size:12px"
        );
    }

    #[test]
    fn ids_and_references_belong_to_their_drawing() {
        let markup = |label: &str| {
            let source = format!(
                r##"<svg><defs><linearGradient id="g"><stop offset="0"/></linearGradient><clipPath id="clip0"><rect width="9"/></clipPath></defs>
                <rect fill="url(#g)" clip-path="url( '#clip0' )" style="stroke:url(#g)"/><use href="#g"/><text id="config">{label}</text></svg>"##
            );
            let html = Html::parse_fragment(&source);
            let svg = html
                .select(&Selector::parse("svg").unwrap())
                .next()
                .unwrap();
            markup(svg).unwrap()
        };
        let (one, two) = (markup("one"), markup("two"));
        let prefix = |markup: &str| markup[markup.find("id=\"").unwrap() + 4..][..10].to_owned();
        assert_ne!(prefix(&one), prefix(&two));
        let p = prefix(&one);
        assert!(p.starts_with('m') && p.ends_with('-'), "{p}");
        for reference in [
            format!(r#"id="{p}g""#),
            format!(r#"id="{p}clip0""#),
            format!(r##"fill="url(#{p}g)""##),
            format!(r##"clip-path="url( '#{p}clip0' )""##),
            format!("stroke:url(#{p}g)"),
            format!(r##"href="#{p}g""##),
            format!(r#"id="{p}config""#),
        ] {
            assert!(one.contains(&reference), "{reference} in {one}");
        }
    }

    #[test]
    fn a_drawing_too_deep_too_large_or_too_long_is_not_kept() {
        let deep = format!(
            "<svg>{}<text>t</text>{}</svg>",
            "<g>".repeat(MAX_DEPTH + 1),
            "</g>".repeat(MAX_DEPTH + 1)
        );
        assert_eq!(kept(&deep), None);
        // Thousands of nested labels or definitions are given up on in one
        // walk, before anything reads a label's text or a shape's paint.
        for (open, close) in [
            ("<text> ", "</text>"),
            ("<defs><g><rect class=\"a\"/>", "</g></defs>"),
        ] {
            let nested = format!("<svg>{}{}</svg>", open.repeat(40_000), close.repeat(40_000));
            let started = std::time::Instant::now();
            assert_eq!(kept(&nested), None);
            assert!(started.elapsed() < std::time::Duration::from_secs(5));
        }
        let wide = format!(
            "<svg><text>t</text>{}</svg>",
            "<g></g>".repeat(MAX_ELEMENTS)
        );
        assert_eq!(kept(&wide), None);
        let long = format!(
            "<svg><text>t</text>{}</svg>",
            format!("<path d=\"{}\"/>", "L1 1".repeat(40)).repeat(MAX_MARKUP / 160 + 1)
        );
        assert_eq!(kept(&long), None);
    }
}
