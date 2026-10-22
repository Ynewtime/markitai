//! Text styles — the objects a run of text points at.
//!
//! A `TSWP.StorageArchive` carries no formatting of its own. What it carries is
//! attribute tables, and every one of them has the same shape: entries of
//! `{1: character_index, 2: reference}`, strictly increasing by index, each run
//! reaching to the next entry. The objects those references land on are the
//! styles:
//!
//! | Storage field | Points at | Message type |
//! |---|---|---|
//! | 5 | paragraph styles | 2022 `TSWP.ParagraphStyleArchive` |
//! | 7 | list styles | 2023 `TSWP.ListStyleArchive` |
//! | 8 | character styles | 2021 `TSWP.CharacterStyleArchive` |
//!
//! The tables are the solid part. Their shape is asserted by the test suite,
//! it is the same in all three apps, and everything in this module that
//! *changes* which style applies to which text works on them alone.
//!
//! What a style archive **contains** was worked out by experiment rather than
//! assumed — see [`property`]. What is not in that list is left unnamed rather
//! than guessed at: a wrong field number writes wrong bytes, where a wrong name
//! in [`crate::registry`] only prints wrong. So a style is handled as what it
//! demonstrably is — a tree of wire fields you can read, address by path and
//! rewrite — and new styles are made by **copying one that already works**,
//! which is the rule `FORMAT.md` gives for whole documents and for the same
//! reason.
//!
//! `iwork style <file> <id>` prints that tree with the path of every field, so
//! the numbers can be discovered from the document at hand rather than assumed.

use std::ops::Range;

use crate::pb::{self, Field, Message, Value};

/// `TSWP.CharacterStyleArchive`.
///
/// Note the numbering against [`TYPE_PARAGRAPH_STYLE`]: the character archive is
/// the *lower* number. Public prior art has these the other way round and this
/// crate copied that, until 241 styles across six documents were asked what they
/// call themselves — every type 2021 identifies as `character-style-…`, every
/// type 2022 as `…-paragraphstyle-…`.
pub const TYPE_CHARACTER_STYLE: u32 = 2021;
/// `TSWP.ParagraphStyleArchive`. See [`TYPE_CHARACTER_STYLE`].
pub const TYPE_PARAGRAPH_STYLE: u32 = 2022;
/// `TSWP.ListStyleArchive`.
pub const TYPE_LIST_STYLE: u32 = 2023;

/// How far to descend into nested messages before giving up.
///
/// Style archives nest a handful of levels; anything deeper is either not a
/// message at all or not something this crate should be walking.
const MAX_DEPTH: usize = 16;

/// Which of the three text-style archives an object is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StyleKind {
    Character,
    Paragraph,
    List,
}

impl StyleKind {
    pub fn from_message_type(message_type: u32) -> Option<StyleKind> {
        match message_type {
            TYPE_CHARACTER_STYLE => Some(StyleKind::Character),
            TYPE_PARAGRAPH_STYLE => Some(StyleKind::Paragraph),
            TYPE_LIST_STYLE => Some(StyleKind::List),
            _ => None,
        }
    }

    pub fn message_type(self) -> u32 {
        match self {
            StyleKind::Character => TYPE_CHARACTER_STYLE,
            StyleKind::Paragraph => TYPE_PARAGRAPH_STYLE,
            StyleKind::List => TYPE_LIST_STYLE,
        }
    }

    /// Field of `TSWP.StorageArchive` whose attribute table points at styles of
    /// this kind.
    ///
    /// Field 5 holds the **paragraph** table and field 8 the **character** one,
    /// which is the opposite of what this crate assumed and of what the field
    /// order suggests. The tell was there from the beginning: `FORMAT.md`
    /// recorded that field 5's run indices "are exactly the paragraph starts"
    /// and called it the character table anyway.
    pub fn attribute_table(self) -> u32 {
        match self {
            StyleKind::Paragraph => 5,
            StyleKind::List => 7,
            StyleKind::Character => 8,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            StyleKind::Character => "character",
            StyleKind::Paragraph => "paragraph",
            StyleKind::List => "list",
        }
    }
}

/// The style's name as the app shows it, e.g. `"Titel"`.
///
/// Absent on *variation* styles — the anonymous ones iWork makes when text is
/// formatted directly rather than by picking a named style. Those carry
/// [`PARENT`] instead and inherit everything they do not override.
pub const NAME: &[u32] = &[1, 1];
/// The style's internal identifier, e.g. `"text-1-paragraphstyle-Title"`. This
/// is the key the stylesheet's keyed entries use.
pub const STYLE_IDENTIFIER: &[u32] = &[1, 2];
/// Reference to the style this one inherits from.
pub const PARENT: &[u32] = &[1, 3, 1];
/// Reference to the stylesheet the style belongs to.
pub const STYLESHEET: &[u32] = &[1, 5, 1];
/// Set on a variation style — the anonymous kind. Naming one of these and
/// listing it among the named styles is what Pages refuses to open.
pub const IS_VARIATION: &[u32] = &[1, 4];
/// How many properties the style overrides. Not maintained by this crate.
pub const OVERRIDE_COUNT: &[u32] = &[10];

/// A string carried somewhere inside a style archive, with the field path it
/// was found at.
///
/// This is the exploratory view — every readable string in the archive, wherever
/// it sits. It is how [`NAME`] was pinned down, and it is worth keeping around:
/// a style archive holds several strings that are easy to confuse, and taking
/// simply the first one is wrong. An unnamed variation style has no field 1.1
/// at all, so its first string is the *font name* in the property bag, and
/// treating that as the style's name would rename the font.
#[derive(Debug, Clone, PartialEq)]
pub struct Label {
    /// Field numbers from the root of the archive down to the string.
    pub path: Vec<u32>,
    pub text: String,
}

/// One text style in a document.
#[derive(Debug, Clone)]
pub struct TextStyle {
    /// Object identifier — the handle every style method on
    /// [`crate::Document`] takes.
    pub identifier: u64,
    /// Stream the object lives in.
    pub stream: String,
    pub kind: StyleKind,
    /// Name from [`NAME`]. `None` on a variation style.
    pub name: Option<String>,
    /// Internal identifier from [`STYLE_IDENTIFIER`].
    pub style_identifier: Option<String>,
    /// Style this one inherits from, from [`PARENT`].
    ///
    /// Worth knowing before editing: changing a named style does nothing to
    /// text whose runs point at a variation that overrides the same field.
    pub parent: Option<u64>,
    /// Stylesheet the style belongs to, from [`STYLESHEET`]. This is the object
    /// a copy gets listed in.
    pub stylesheet: Option<u64>,
    /// Every readable string the archive carries, with its path.
    pub labels: Vec<Label>,
    /// The style archive exactly as it sits in the document.
    pub archive: Message,
}

impl TextStyle {
    /// Name if it has one, else the internal identifier, else the first string
    /// in the archive — for display only, never for editing.
    pub fn label(&self) -> Option<&str> {
        self.name
            .as_deref()
            .or(self.style_identifier.as_deref())
            .or_else(|| self.labels.first().map(|l| l.text.as_str()))
    }
}

/// Field paths inside a style's property bag.
///
/// Field **11** is the character bag. It appears in both kinds of style,
/// because a paragraph style carries character properties too. Field **12** is
/// the paragraph bag and only paragraph styles have one. Field **10** is a
/// count of how many properties the style overrides.
///
/// Most of these were established by a controlled experiment rather than by
/// correlation: a document was built in which each paragraph differed from a
/// baseline in exactly one property, with values chosen to be unmistakable —
/// 37pt, `#123456`, 175%, 17pt — and imported into Pages. Diffing each
/// resulting style against the baseline's leaves exactly one changed field per
/// probe, and the value is right there. Where a colour was asked for, all four
/// channels come back to the byte: `#123456` is `0.070588, 0.203922, 0.337255`,
/// which is `18/255, 52/255, 86/255`.
///
/// A few entries below were *not* reached by that experiment and carry names
/// from public prior art instead; each says so. The distinction is the same one
/// [`crate::registry`] draws, and it matters for the same reason — a probed
/// field has been seen to do what it says, a borrowed name has not.
///
/// ### Clearing a property is not the same as removing it
///
/// Several properties come in pairs: the value, and a boolean saying the value
/// is deliberately *none*. Removing the field means "inherit from the parent
/// style"; setting the companion means "explicitly nothing". They are different
/// documents. The companions this module knows are `FONT_COLOR_NULL`,
/// `TEXT_BACKGROUND_NULL`, `SHADOW_NULL` and `PARAGRAPH_BACKGROUND_NULL`.
pub mod property {
    // -- character properties, field 11 --------------------------------------
    /// Bold **toggle**, `0` or `1` — not "is this text bold".
    ///
    /// Independent of the font's own weight: importing bold text sets both this
    /// and a `-Bold` font name, and table-label styles in real documents use
    /// `HelveticaNeue-Bold` with this left at `0`.
    pub const BOLD: &[u32] = &[11, 1];
    /// Italic toggle, `0` or `1`.
    pub const ITALIC: &[u32] = &[11, 2];
    /// Font size in points. Asked for 37pt, got `37`.
    pub const FONT_SIZE: &[u32] = &[11, 3];
    /// PostScript font name — `"Helvetica-BoldOblique"`, `"CourierNewPSMT"`.
    pub const FONT_NAME: &[u32] = &[11, 5];
    /// The font colour is deliberately none. See the module note on clearing.
    pub const FONT_COLOR_NULL: &[u32] = &[11, 6];
    /// Font colour: `{1: model, 3: r, 4: g, 5: b, 6: a}`, channels `0.0..=1.0`.
    pub const FONT_COLOR: &[u32] = &[11, 7];
    pub const RED: &[u32] = &[11, 7, 3];
    pub const GREEN: &[u32] = &[11, 7, 4];
    pub const BLUE: &[u32] = &[11, 7, 5];
    pub const ALPHA: &[u32] = &[11, 7, 6];
    /// Language tag, e.g. `"fr"`.
    pub const LANGUAGE: &[u32] = &[11, 9];
    /// Vertical position: `1` superscript, `2` subscript.
    pub const SUPERSCRIPT: &[u32] = &[11, 10];
    /// Underline: `1` single, `2` double.
    pub const UNDERLINE: &[u32] = &[11, 11];
    /// Strikethrough: `1` single.
    pub const STRIKETHROUGH: &[u32] = &[11, 12];
    /// Capitalisation: `1` all caps, `2` small caps.
    pub const CAPITALISATION: &[u32] = &[11, 13];
    /// Baseline shift. The probe asked for 6pt up and stored `12`.
    pub const BASELINE_SHIFT: &[u32] = &[11, 14];
    /// Kerning. Not reached by the probe; name from prior art.
    pub const KERNING: &[u32] = &[11, 15];
    /// The shadow is deliberately none.
    pub const SHADOW_NULL: &[u32] = &[11, 20];
    /// Text shadow: colour, angle, offset, opacity. The probe produced
    /// `45°, offset 1, opacity 0.5`.
    pub const SHADOW: &[u32] = &[11, 21];
    /// Strikethrough colour, and its width. Both appeared alongside a
    /// strikethrough in the probe, taking the text colour.
    pub const STRIKETHROUGH_COLOR: &[u32] = &[11, 23];
    pub const STRIKETHROUGH_WIDTH: &[u32] = &[11, 24];
    /// The text background is deliberately none.
    pub const TEXT_BACKGROUND_NULL: &[u32] = &[11, 25];
    /// Highlight behind the characters. `#ABCDEF` came back exact.
    pub const TEXT_BACKGROUND: &[u32] = &[11, 26];
    /// Tracking, as a fraction of the font size: 3pt on 12pt gave `0.25`.
    pub const TRACKING: &[u32] = &[11, 27];
    /// Underline colour, and its width. Both appeared alongside an underline.
    pub const UNDERLINE_COLOR: &[u32] = &[11, 29];
    pub const UNDERLINE_WIDTH: &[u32] = &[11, 30];
    /// Strike or underline words only, skipping the spaces between them. Both
    /// appeared set to `0` alongside their decoration.
    pub const WORD_STRIKETHROUGH: &[u32] = &[11, 31];
    pub const WORD_UNDERLINE: &[u32] = &[11, 32];
    /// Stroke drawn around the glyphs — what "outline" text is. The probe
    /// produced a colour and a width of `0.36`, with [`TEXT_FILL_NULL`] set.
    pub const TEXT_STROKE: &[u32] = &[11, 44];
    pub const TEXT_FILL_NULL: &[u32] = &[11, 45];
    /// Fill drawn inside the glyphs, and the colour inside it.
    pub const TEXT_FILL: &[u32] = &[11, 46];
    pub const TEXT_FILL_COLOR: &[u32] = &[11, 46, 1];

    /// Every place a style keeps the colour of its own text.
    ///
    /// A style does not have one text colour, it has up to four, and they are
    /// expected to agree. Choosing a colour in Pages writes it to all of them;
    /// setting only [`FONT_COLOR`] leaves the fill behind, and **the fill is
    /// what gets drawn** — a title whose `11.7` was set to red and whose
    /// `11.46.1` stayed black renders black.
    ///
    /// [`crate::Document::set_text_style_color`] writes the lot. The text
    /// background at [`TEXT_BACKGROUND`] and the outline colour inside
    /// [`TEXT_STROKE`] are deliberately not here: those are different colours
    /// that happen to share a shape.
    pub const TEXT_COLOR_PATHS: &[&[u32]] = &[
        FONT_COLOR,
        STRIKETHROUGH_COLOR,
        UNDERLINE_COLOR,
        TEXT_FILL_COLOR,
    ];

    // -- paragraph properties, field 12 --------------------------------------
    /// Alignment: `1` right, `2` centre, `3` justified. Left is the absence.
    pub const ALIGNMENT: &[u32] = &[12, 1];
    /// The paragraph fill is deliberately none.
    pub const PARAGRAPH_BACKGROUND_NULL: &[u32] = &[12, 5];
    /// Paragraph background fill. `#FEDCBA` came back exact.
    pub const PARAGRAPH_BACKGROUND: &[u32] = &[12, 6];
    /// First-line indent, points.
    pub const FIRST_LINE_INDENT: &[u32] = &[12, 7];
    /// Hyphenate. Not reached by the probe; name from prior art.
    pub const HYPHENATE: &[u32] = &[12, 8];
    /// Keep the paragraph's lines on one page. Not reached by the probe.
    pub const KEEP_LINES_TOGETHER: &[u32] = &[12, 9];
    /// Keep with the next paragraph.
    pub const KEEP_WITH_NEXT: &[u32] = &[12, 10];
    /// Left indent, points.
    pub const LEFT_INDENT: &[u32] = &[12, 11];
    /// Line spacing. The multiple sits at `12.13.2`: 175% gave `1.75`.
    pub const LINE_SPACING: &[u32] = &[12, 13];
    pub const LINE_SPACING_AMOUNT: &[u32] = &[12, 13, 2];
    /// Page break before the paragraph.
    pub const PAGE_BREAK_BEFORE: &[u32] = &[12, 14];
    /// Which edges carry a rule, with its offset and width beside it.
    pub const BORDERS: &[u32] = &[12, 15];
    pub const RULE_OFFSET: &[u32] = &[12, 17];
    pub const RULE_WIDTH: &[u32] = &[12, 18];
    /// Right indent, points.
    pub const RIGHT_INDENT: &[u32] = &[12, 19];
    /// Space after and before the paragraph, points.
    pub const SPACE_AFTER: &[u32] = &[12, 20];
    pub const SPACE_BEFORE: &[u32] = &[12, 21];
    /// Tab stops. The first stop's position sits at `12.25.1.1`; 144pt gave
    /// `144`.
    pub const TABS: &[u32] = &[12, 25];
    /// Widow and orphan control.
    pub const WIDOW_CONTROL: &[u32] = &[12, 26];
    /// Outline depth, for a heading. Not reached by the probe.
    pub const OUTLINE_LEVEL: &[u32] = &[12, 27];
    /// Stroke used to draw the paragraph's rule.
    pub const PARAGRAPH_STROKE: &[u32] = &[12, 32];
    /// The list style is deliberately none.
    pub const LIST_STYLE_NULL: &[u32] = &[12, 39];
    /// Reference to the list style a bulleted or numbered paragraph uses.
    pub const LIST_STYLE: &[u32] = &[12, 40, 1];

    /// How far each name below can be trusted, in the same terms
    /// [`crate::registry`] uses.
    pub use crate::registry::Confidence;

    /// Every named path, with what backs the name.
    pub const BY_NAME: &[(&str, &[u32], Confidence)] = &[
        ("bold", BOLD, Confidence::Confirmed),
        ("italic", ITALIC, Confidence::Confirmed),
        ("font-size", FONT_SIZE, Confidence::Confirmed),
        ("font-name", FONT_NAME, Confidence::Confirmed),
        ("font-color-null", FONT_COLOR_NULL, Confidence::Unverified),
        ("font-color", FONT_COLOR, Confidence::Confirmed),
        ("text-fill", TEXT_FILL, Confidence::Confirmed),
        ("text-stroke", TEXT_STROKE, Confidence::Confirmed),
        ("shadow", SHADOW, Confidence::Confirmed),
        ("red", RED, Confidence::Confirmed),
        ("green", GREEN, Confidence::Confirmed),
        ("blue", BLUE, Confidence::Confirmed),
        ("alpha", ALPHA, Confidence::Confirmed),
        ("language", LANGUAGE, Confidence::Confirmed),
        ("superscript", SUPERSCRIPT, Confidence::Confirmed),
        ("underline", UNDERLINE, Confidence::Confirmed),
        ("strikethrough", STRIKETHROUGH, Confidence::Confirmed),
        ("capitalisation", CAPITALISATION, Confidence::Confirmed),
        ("baseline-shift", BASELINE_SHIFT, Confidence::Confirmed),
        ("kerning", KERNING, Confidence::Unverified),
        (
            "strikethrough-width",
            STRIKETHROUGH_WIDTH,
            Confidence::Inferred,
        ),
        ("text-background", TEXT_BACKGROUND, Confidence::Confirmed),
        ("tracking", TRACKING, Confidence::Confirmed),
        ("underline-width", UNDERLINE_WIDTH, Confidence::Inferred),
        (
            "word-strikethrough",
            WORD_STRIKETHROUGH,
            Confidence::Inferred,
        ),
        ("word-underline", WORD_UNDERLINE, Confidence::Inferred),
        ("alignment", ALIGNMENT, Confidence::Confirmed),
        (
            "paragraph-background",
            PARAGRAPH_BACKGROUND,
            Confidence::Confirmed,
        ),
        (
            "first-line-indent",
            FIRST_LINE_INDENT,
            Confidence::Confirmed,
        ),
        ("hyphenate", HYPHENATE, Confidence::Unverified),
        (
            "keep-lines-together",
            KEEP_LINES_TOGETHER,
            Confidence::Unverified,
        ),
        ("keep-with-next", KEEP_WITH_NEXT, Confidence::Confirmed),
        ("left-indent", LEFT_INDENT, Confidence::Confirmed),
        ("line-spacing", LINE_SPACING_AMOUNT, Confidence::Confirmed),
        (
            "page-break-before",
            PAGE_BREAK_BEFORE,
            Confidence::Confirmed,
        ),
        ("right-indent", RIGHT_INDENT, Confidence::Confirmed),
        ("space-after", SPACE_AFTER, Confidence::Confirmed),
        ("space-before", SPACE_BEFORE, Confidence::Confirmed),
        ("widow-control", WIDOW_CONTROL, Confidence::Confirmed),
        ("outline-level", OUTLINE_LEVEL, Confidence::Unverified),
        ("tabs", TABS, Confidence::Confirmed),
        ("paragraph-stroke", PARAGRAPH_STROKE, Confidence::Confirmed),
        ("borders", BORDERS, Confidence::Inferred),
        ("name", super::NAME, Confidence::Confirmed),
        (
            "style-identifier",
            super::STYLE_IDENTIFIER,
            Confidence::Confirmed,
        ),
    ];

    pub fn path(name: &str) -> Option<&'static [u32]> {
        BY_NAME
            .iter()
            .find(|(known, _, _)| *known == name)
            .map(|(_, path, _)| *path)
    }
}

/// Does this message look like a colour — `{1: model, 3: r, 4: g, 5: b, 6: a}`
/// with the channels as floats?
pub fn is_color(message: &Message) -> bool {
    [3, 4, 5]
        .iter()
        .all(|c| matches!(message.get(*c), Some(Value::Fixed32(_))))
}

/// Set the channels of a colour message, leaving its model, colour space and
/// anything else it carries alone.
pub fn set_channels(colour: &mut Message, red: f32, green: f32, blue: f32, alpha: f32) {
    for (field, value) in [(3, red), (4, green), (5, blue), (6, alpha)] {
        colour.set_in_order(field, Value::Fixed32(value.to_le_bytes()));
    }
}

/// Read a string field at `path`, if it holds readable text.
pub fn string_at(archive: &Message, path: &[u32]) -> Option<String> {
    match get_path(archive, path)? {
        Value::Bytes(raw) => readable(&raw),
        _ => None,
    }
}

/// Read a reference field at `path`, if it holds one.
pub fn reference_at(archive: &Message, path: &[u32]) -> Option<u64> {
    match get_path(archive, path)? {
        Value::Varint(identifier) => Some(identifier),
        _ => None,
    }
}

/// Where a style is used: one run of one storage's attribute table.
#[derive(Debug, Clone)]
pub struct StyleUse {
    /// Object identifier of the `TSWP.StorageArchive` holding the run.
    pub storage: u64,
    pub stream: String,
    /// Field of the storage whose attribute table the run belongs to.
    pub table: u32,
    /// Character range the run covers, in UTF-16 code units. It ends at the
    /// next entry in the table, or at the end of the text.
    pub range: Range<u64>,
}

/// What [`crate::Document::create_text_style`] did.
#[derive(Debug, Clone)]
pub struct CreatedStyle {
    pub identifier: u64,
    /// Style that was copied to make it.
    pub template: u64,
    pub stream: String,
    /// Stylesheet entries that were cloned alongside the object, so the new
    /// style is listed everywhere the template was.
    pub registrations_cloned: usize,
    /// The name the copy was given, or `None` when the template was a variation
    /// style and the requested name was therefore not applied. See
    /// [`crate::Document::create_text_style`].
    pub name: Option<String>,
}

/// What [`crate::Document::delete_text_style`] did.
#[derive(Debug, Clone)]
pub struct StyleDeletion {
    pub identifier: u64,
    /// Runs pointed at the replacement style.
    pub runs_repointed: usize,
    /// Runs removed outright, letting the preceding run extend over them.
    pub runs_dropped: usize,
    /// References dropped from stylesheets.
    pub registrations_removed: usize,
}

// -- references --------------------------------------------------------------

/// A `TSP.Reference` — always `{1: <object identifier>}`, everywhere in the
/// format. There are no file offsets; resolution is by identifier across the
/// whole package.
pub fn reference(identifier: u64) -> Message {
    let mut message = Message::default();
    message.set(1, Value::Varint(identifier));
    message
}

/// The object a message refers to, if it is a reference and nothing else.
///
/// The "and nothing else" matters: a bare `{1: n}` is a reference, but a
/// message that merely *starts* with a varint field 1 is something else with
/// its own meaning, and must not be mistaken for one.
pub fn reference_target(message: &Message) -> Option<u64> {
    match message.fields.as_slice() {
        [Field {
            number: 1,
            value: Value::Varint(identifier),
        }] => Some(*identifier),
        _ => None,
    }
}

/// Is this message a reference to `identifier`?
pub fn is_reference_to(message: &Message, identifier: u64) -> bool {
    reference_target(message) == Some(identifier)
}

/// How many references to `identifier` this message holds, at any depth.
pub fn count_references(message: &Message, identifier: u64) -> usize {
    count_references_to(message, identifier, MAX_DEPTH)
}

fn count_references_to(message: &Message, identifier: u64, depth: usize) -> usize {
    if depth == 0 {
        return 0;
    }
    let mut found = 0;
    for field in &message.fields {
        let Value::Bytes(raw) = &field.value else {
            continue;
        };
        let Some(nested) = pb::decode_nested(raw) else {
            continue;
        };
        if is_reference_to(&nested, identifier) {
            found += 1;
        } else {
            found += count_references_to(&nested, identifier, depth - 1);
        }
    }
    found
}

/// Every object this message refers to, at any depth, without duplicates.
///
/// Used to reconcile a component's declared `external_references`
/// ([`crate::Document::declare_external_references`]). The detector is
/// [`reference_target`]'s — a bare `{1: n}` and nothing else — which is strict
/// enough to be safe in practice: run over five Pages documents and one Keynote
/// document, every reference it found that crossed a component boundary was
/// already declared by iWork itself, and it found no others.
pub fn references(message: &Message) -> Vec<u64> {
    let mut out = Vec::new();
    collect_references(message, &mut out, MAX_DEPTH);
    out.sort_unstable();
    out.dedup();
    out
}

fn collect_references(message: &Message, out: &mut Vec<u64>, depth: usize) {
    if depth == 0 {
        return;
    }
    for field in &message.fields {
        let Value::Bytes(raw) = &field.value else {
            continue;
        };
        let Some(nested) = pb::decode_nested(raw) else {
            continue;
        };
        match reference_target(&nested) {
            Some(target) => out.push(target),
            None => collect_references(&nested, out, depth - 1),
        }
    }
}

// -- labels ------------------------------------------------------------------

/// Every readable string in an archive, with the path it sits at.
pub fn labels(archive: &Message) -> Vec<Label> {
    let mut out = Vec::new();
    collect_labels(archive, &mut Vec::new(), &mut out, MAX_DEPTH);
    out
}

fn collect_labels(message: &Message, path: &mut Vec<u32>, out: &mut Vec<Label>, depth: usize) {
    if depth == 0 {
        return;
    }
    for field in &message.fields {
        let Value::Bytes(raw) = &field.value else {
            continue;
        };
        path.push(field.number);
        // A payload that round-trips as a message is treated as one; only what
        // is left over can be a string. Getting this the other way round would
        // mean walking into text and inventing structure inside it.
        match pb::decode_nested(raw) {
            Some(nested) => collect_labels(&nested, path, out, depth - 1),
            None => {
                if let Some(text) = readable(raw) {
                    out.push(Label {
                        path: path.clone(),
                        text,
                    });
                }
            }
        }
        path.pop();
    }
}

/// UTF-8 with no control characters and at least one letter or digit — enough
/// to tell a style name from a packed array that happens to be valid UTF-8.
fn readable(raw: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(raw).ok()?;
    let usable = !text.is_empty()
        && text.chars().all(|c| !c.is_control())
        && text.chars().any(char::is_alphanumeric);
    usable.then(|| text.to_string())
}

// -- field paths -------------------------------------------------------------

/// Read the field at `path`, descending through nested messages.
///
/// Repeated fields resolve to their first occurrence, which is what style
/// archives need — the repeated fields in this format are attribute tables and
/// stylesheet lists, and those have their own operations below. Returned by
/// value because everything below the top level has to be decoded to be
/// reached.
pub fn get_path(message: &Message, path: &[u32]) -> Option<Value> {
    match path.split_first()? {
        (head, []) => message.get(*head).cloned(),
        (head, rest) => get_path(&pb::decode_nested(message.bytes(*head)?)?, rest),
    }
}

/// Set the field at `path`, or remove it when `value` is `None`.
///
/// **Every message on the way must already exist.** Setting `11.7.3` in a style
/// that has no `11.7` fails rather than inventing one, because a message this
/// crate invents is a message with only the fields it happened to set — and
/// that is not a valid archive. Asking Pages to render a colour built as
/// `{3: r, 4: g, 5: b}`, with no model and no alpha, crashes it on opening; a
/// real colour carries `{1: model, 3: r, 4: g, 5: b, 6: a, 12: space}` and
/// there is no way to know that from the path alone.
///
/// A leaf may be created freely — adding a field to a bag that is already there
/// is safe, and is how a style gains a property it did not have. It is the
/// *containers* that must be inherited rather than fabricated.
///
/// To set a property whose container is missing, get the container from a style
/// that has one: copy that style
/// ([`crate::Document::create_text_style`]) and edit the copy, or lift the
/// subtree across with [`crate::Document::copy_text_style_property`].
pub fn set_path(message: &mut Message, path: &[u32], value: Option<Value>) -> Result<(), String> {
    let Some((head, rest)) = path.split_first() else {
        return Err("empty field path".into());
    };
    if rest.is_empty() {
        match value {
            Some(value) => message.set_in_order(*head, value),
            None => {
                message.clear(*head);
            }
        }
        return Ok(());
    }

    let position = message.fields.iter().position(|f| f.number == *head);
    let Some(position) = position else {
        if value.is_none() {
            return Ok(()); // nothing to clear
        }
        return Err(format!(
            "field {head} does not exist, and a message this crate invents \
             would carry only the fields it was asked for — copy a style that \
             has one instead"
        ));
    };
    let Value::Bytes(raw) = &message.fields[position].value else {
        return Err(format!("field {head} is not a nested message"));
    };
    let mut nested = if raw.is_empty() {
        Message::default()
    } else {
        pb::decode_nested(raw).ok_or_else(|| format!("field {head} is not a nested message"))?
    };

    set_path(&mut nested, rest, value)?;
    message.fields[position].value = Value::Bytes(nested.encode());
    Ok(())
}

// -- attribute tables --------------------------------------------------------

/// One entry of an attribute table.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    /// Character index the run starts at, in UTF-16 code units.
    pub start: u64,
    /// Object the run points at, when the entry carries a reference.
    pub style: Option<u64>,
}

/// The entries of an attribute table, in order.
pub fn runs(table: &Message) -> Vec<Run> {
    entries(table)
        .into_iter()
        .map(|(start, entry)| Run {
            start,
            style: entry_style(&entry),
        })
        .collect()
}

/// Split a table into its entries and everything else, so a rewrite can put
/// back what it does not understand.
fn entries(table: &Message) -> Vec<(u64, Message)> {
    let mut out = Vec::new();
    for field in &table.fields {
        let Value::Bytes(raw) = &field.value else {
            continue;
        };
        let Some(entry) = pb::decode_nested(raw) else {
            continue;
        };
        let Some(start) = entry.varint(1) else {
            continue;
        };
        out.push((start, entry));
    }
    out
}

fn non_entries(table: &Message) -> Vec<Field> {
    table
        .fields
        .iter()
        .filter(|field| match &field.value {
            Value::Bytes(raw) => pb::decode_nested(raw)
                .map(|entry| entry.varint(1).is_none())
                .unwrap_or(true),
            _ => true,
        })
        .cloned()
        .collect()
}

fn entry_style(entry: &Message) -> Option<u64> {
    reference_target(&pb::decode_nested(entry.bytes(2)?)?)
}

/// Field number the entries sit under. Observed as 1; taken from the table
/// itself when it already has entries, so a document that disagrees still
/// round-trips.
fn entry_field(table: &Message) -> u32 {
    table
        .fields
        .iter()
        .find(|field| match &field.value {
            Value::Bytes(raw) => pb::decode_nested(raw)
                .map(|entry| entry.varint(1).is_some())
                .unwrap_or(false),
            _ => false,
        })
        .map(|field| field.number)
        .unwrap_or(1)
}

/// Put entries back into a table, keeping it strictly increasing and free of
/// runs that only repeat the run before them.
///
/// A run saying the same thing as its predecessor is not *wrong* — the text
/// still ends up styled the same way — but it is not what a table looks like
/// when iWork writes one, and it accumulates: style a word, then re-point the
/// style, and the storage keeps entries that draw no boundary. Two entries
/// coalesce only when they are identical apart from where they start, so an
/// entry carrying something this crate does not understand is never merged
/// away on the strength of its style reference alone.
///
/// Fields that are not entries keep their order but move ahead of the entries.
/// Every table in the samples is entries and nothing else, so this is a
/// precaution against losing a field rather than a reshuffling anything has
/// been observed to notice.
fn rebuild(table: &mut Message, keep: Vec<Field>, entries: Vec<(u64, Message)>, field: u32) {
    table.fields = keep;
    let mut previous: Option<(u64, Vec<u8>)> = None;
    for (start, entry) in entries {
        let shape = {
            let mut without_index = entry.clone();
            without_index.clear(1);
            without_index.encode()
        };
        if let Some((last_start, last_shape)) = &previous {
            if start <= *last_start || shape == *last_shape {
                continue;
            }
        }
        table.fields.push(Field {
            number: field,
            value: Value::Bytes(entry.encode()),
        });
        previous = Some((start, shape));
    }
}

/// Point `range` at `style`, leaving the rest of the table as it was.
///
/// The run in effect at the end of the range is re-established immediately
/// after it, so applying a style to the middle of a paragraph does not bleed
/// into what follows. Entries are cloned from whichever run was already in
/// effect, so anything else an entry carries survives the edit.
///
/// `text_len` is the length of the storage's text in UTF-16 code units; the
/// range is clamped into it, and a range that reaches the end needs no
/// re-establishing entry after it.
///
/// A table always comes back with an entry at 0, because a table that begins
/// late leaves the characters in front of it with no attribute at all — see the
/// nil head entry below.
pub fn apply(table: &mut Message, range: Range<u64>, style: u64, text_len: u64) {
    let start = range.start.min(text_len);
    let end = range.end.min(text_len);
    if start >= end {
        return;
    }

    let field = entry_field(table);
    let keep = non_entries(table);
    let existing = entries(table);

    let in_effect = |at: u64| existing.iter().rev().find(|(index, _)| *index <= at);

    let mut head = in_effect(start).map(|(_, e)| e.clone()).unwrap_or_default();
    head.set(1, Value::Varint(start));
    head.set(2, Value::Bytes(reference(style).encode()));

    let tail = in_effect(end).map(|(_, e)| e.clone());
    let resumes_already = existing.iter().any(|(index, _)| *index == end);

    let mut out: Vec<(u64, Message)> = existing
        .iter()
        .filter(|(index, _)| *index < start)
        .cloned()
        .collect();
    // A run or paragraph table describes the text from its first entry onward,
    // so a table whose first entry is at `start` leaves the characters before it
    // with no attribute at all — which `iwork check` reports and which iWork
    // never writes. Styling a range in the middle of a storage that had no table
    // of this kind at all is exactly that case, and Pages answers it with a
    // **head entry at 0 carrying nothing**: made to set a bold 22pt font over
    // characters 19–29 of a storage with no character table, it wrote
    // `[0 nil, 19 bold, 30 nil]` (FORMAT.md §Text, the second probe fixture). A
    // nil entry is the format's "whatever was in force here", and it is what
    // this crate writes for a new paragraph too.
    if start > 0 && out.first().map(|(index, _)| *index) != Some(0) {
        let mut nil = Message::default();
        nil.set(1, Value::Varint(0));
        out.insert(0, (0, nil));
    }
    out.push((start, head));
    if end < text_len && !resumes_already {
        if let Some(mut tail) = tail {
            tail.set(1, Value::Varint(end));
            out.push((end, tail));
        }
    }
    out.extend(existing.iter().filter(|(index, _)| *index >= end).cloned());

    rebuild(table, keep, out, field);
}

/// Point every run that uses `from` at `to`, or drop those runs when `to` is
/// `None`, letting the preceding run extend over them.
///
/// **The entry at index 0 is not dropped, it is emptied.** "Let the preceding
/// run extend over it" has no meaning at the start of the text, where there is
/// no preceding run: removing that entry leaves the table beginning at the next
/// one, and the characters in front of it with no attribute at all — the
/// invariant `iwork check` enforces, broken by the writer that is supposed to
/// keep it (`delete-style pages-styled.pages 1732648` reported `0..12`
/// uncovered). What the format has for "there is deliberately nothing here" is
/// an entry carrying its index and no object, which is what Pages writes at 0
/// for an unstyled first run and what this leaves behind.
///
/// Returns how many entries were touched.
pub fn repoint(table: &mut Message, from: u64, to: Option<u64>) -> usize {
    let field = entry_field(table);
    let keep = non_entries(table);
    let mut touched = 0;
    let mut out = Vec::new();
    for (start, mut entry) in entries(table) {
        if entry_style(&entry) != Some(from) {
            out.push((start, entry));
            continue;
        }
        touched += 1;
        match to {
            Some(to) => {
                entry.set(2, Value::Bytes(reference(to).encode()));
                out.push((start, entry));
            }
            None if start == 0 => {
                entry.clear(2);
                out.push((start, entry));
            }
            None => {}
        }
    }
    rebuild(table, keep, out, field);
    touched
}

/// Clone every top-level bare reference to `old` as a reference to `new`.
///
/// This is how a duplicated style gets listed wherever its template was: the
/// document stylesheet lists its styles as repeated `TSP.Reference` fields, so
/// the entry to add is exactly the entry already there with one number changed
/// — no knowledge of the stylesheet's schema required, and nothing to get
/// wrong.
///
/// **Call this on the style's own stylesheet and nowhere else.** A bare
/// reference is a good signature for "listed here", but on its own it is not a
/// good enough one, and the caller cannot tell by message type either — the
/// document stylesheet is type `401`, in the TSP/TSK range rather than the TSS
/// range its name suggests. What identifies it is that the style points at it:
/// every style archive carries [`STYLESHEET`], so the list to add to is named
/// by the style being copied. [`crate::Document::create_text_style`] resolves
/// it that way.
///
/// The alternative — cloning every top-level bare reference anywhere — looked
/// equivalent and is not. A Keynote `KN.SlideArchive` holds five bare style
/// references in field 31, one per outline level, and they are a *positional
/// array*: adding a sixth does not list a style, it corrupts the mapping from
/// level to style. Repeatedness cannot separate the two cases, because both are
/// repeated fields. Provenance can.
///
/// Keyed entries `{1: "text-1-paragraphstyle-Title", 2: reference}`, mapping a
/// well-known identifier to a style, are left alone: duplicating one would
/// either collide on the key or invent one, and a second style claiming to be
/// the document's title style is worse than a style that is merely listed.
///
/// The grouping entries `{1: parent reference, 2: child reference…}` are not
/// left alone — see [`clone_sibling`], which the caller must also run.
pub fn clone_registrations(stylesheet: &mut Message, old: u64, new: u64) -> usize {
    let mut additions = Vec::new();
    for field in &stylesheet.fields {
        let Value::Bytes(raw) = &field.value else {
            continue;
        };
        let Some(nested) = pb::decode_nested(raw) else {
            continue;
        };
        if is_reference_to(&nested, old) {
            additions.push(Field {
                number: field.number,
                value: Value::Bytes(reference(new).encode()),
            });
        }
    }
    let cloned = additions.len();
    for field in additions {
        stylesheet.append_in_order(field.number, field.value);
    }
    cloned
}

/// List `new` beside `old` in the entry that groups `old` under `parent`.
///
/// A stylesheet does not only list its styles; it also records, for each style
/// that has children, an entry shaped `{1: parent, 2: child, 2: child, …}`.
/// Every style in the five Pages documents and the Keynote document this crate
/// was checked against that names a parent *and* is listed in a stylesheet
/// appears in its parent's entry — 109 styles, no exceptions. A copy that is
/// listed but not grouped is a shape those documents never take.
///
/// The entry is found by its contents rather than its field number: the one
/// whose first field is a reference to `parent` and which already lists `old`.
/// Nothing else in a stylesheet has that shape, and being wrong about it would
/// mean adding a style to the wrong family.
///
/// Returns whether an entry was found and extended.
pub fn clone_sibling(stylesheet: &mut Message, parent: u64, old: u64, new: u64) -> bool {
    for field in &mut stylesheet.fields {
        let Value::Bytes(raw) = &field.value else {
            continue;
        };
        let Some(mut entry) = pb::decode_nested(raw) else {
            continue;
        };
        let heads_the_family = entry
            .fields
            .first()
            .and_then(|f| match &f.value {
                Value::Bytes(raw) => pb::decode_nested(raw),
                _ => None,
            })
            .is_some_and(|head| is_reference_to(&head, parent));
        if !heads_the_family {
            continue;
        }
        let Some(position) = entry.fields.iter().position(|f| match &f.value {
            Value::Bytes(raw) => pb::decode_nested(raw).is_some_and(|r| is_reference_to(&r, old)),
            _ => false,
        }) else {
            continue;
        };
        let number = entry.fields[position].number;
        entry.fields.insert(
            position + 1,
            Field {
                number,
                value: Value::Bytes(reference(new).encode()),
            },
        );
        field.value = Value::Bytes(entry.encode());
        return true;
    }
    false
}

/// Undo [`clone_sibling`]: drop `identifier` from the family entries.
///
/// An entry whose *head* is `identifier` describes that style's own children
/// and goes with it; an entry that merely lists it loses one child. Returns the
/// number of references dropped.
pub fn remove_sibling(stylesheet: &mut Message, identifier: u64) -> usize {
    let mut dropped = 0;
    stylesheet.fields.retain_mut(|field| {
        let Value::Bytes(raw) = &field.value else {
            return true;
        };
        let Some(mut entry) = pb::decode_nested(raw) else {
            return true;
        };
        let refers = |f: &Field| match &f.value {
            Value::Bytes(raw) => {
                pb::decode_nested(raw).is_some_and(|r| is_reference_to(&r, identifier))
            }
            _ => false,
        };
        // A family entry is headed by the parent's reference. A keyed entry is
        // headed by a string, and is not ours to edit — see
        // [`clone_registrations`].
        let heads_a_family = entry.fields.first().is_some_and(|f| match &f.value {
            Value::Bytes(raw) => {
                pb::decode_nested(raw).is_some_and(|head| reference_target(&head).is_some())
            }
            _ => false,
        });
        if !heads_a_family || entry.fields.len() < 2 || !entry.fields.iter().any(refers) {
            return true;
        }
        if entry.fields.first().is_some_and(refers) {
            dropped += entry.fields.iter().filter(|f| refers(f)).count();
            return false;
        }
        let before = entry.fields.len();
        entry.fields.retain(|f| !refers(f));
        dropped += before - entry.fields.len();
        field.value = Value::Bytes(entry.encode());
        true
    });
    dropped
}

/// Drop every top-level bare reference to `identifier`.
pub fn remove_registrations(stylesheet: &mut Message, identifier: u64) -> usize {
    let before = stylesheet.fields.len();
    stylesheet.fields.retain(|field| match &field.value {
        Value::Bytes(raw) => match pb::decode_nested(raw) {
            Some(nested) => !is_reference_to(&nested, identifier),
            None => true,
        },
        _ => true,
    });
    before - stylesheet.fields.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(start: u64, style: u64) -> Message {
        let mut entry = Message::default();
        entry.set(1, Value::Varint(start));
        entry.set(2, Value::Bytes(reference(style).encode()));
        entry
    }

    fn table(runs: &[(u64, u64)]) -> Message {
        let mut table = Message::default();
        for (start, style) in runs {
            table.fields.push(Field {
                number: 1,
                value: Value::Bytes(entry(*start, *style).encode()),
            });
        }
        table
    }

    fn shape(table: &Message) -> Vec<(u64, Option<u64>)> {
        runs(table)
            .into_iter()
            .map(|r| (r.start, r.style))
            .collect()
    }

    #[test]
    fn a_reference_is_a_lone_varint_field_one() {
        assert_eq!(reference_target(&reference(3712)), Some(3712));
        let mut two_fields = reference(3712);
        two_fields.set(2, Value::Varint(1));
        assert_eq!(
            reference_target(&two_fields),
            None,
            "an entry is not a reference"
        );
        assert_eq!(reference_target(&Message::default()), None);
    }

    #[test]
    fn counts_references_at_any_depth() {
        let mut inner = Message::default();
        inner.set(4, Value::Bytes(reference(9).encode()));
        let mut outer = Message::default();
        outer.set(1, Value::Bytes(inner.encode()));
        outer.set(2, Value::Bytes(reference(9).encode()));
        outer.set(3, Value::Bytes(reference(10).encode()));
        assert_eq!(count_references(&outer, 9), 2);
        assert_eq!(count_references(&outer, 10), 1);
        assert_eq!(count_references(&outer, 11), 0);
    }

    #[test]
    fn labels_carry_the_path_they_were_found_at() {
        let mut base = Message::default();
        base.set(1, Value::Bytes(b"Grosse Uberschrift".to_vec()));
        base.set(2, Value::Bytes(b"heading-1".to_vec()));
        let mut archive = Message::default();
        archive.set(1, Value::Bytes(base.encode()));
        archive.set(11, Value::Varint(1));

        let found = labels(&archive);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].text, "Grosse Uberschrift");
        assert_eq!(found[0].path, vec![1, 1]);
        assert_eq!(found[1].path, vec![1, 2]);
    }

    #[test]
    fn set_path_sets_and_clears_leaves_in_a_bag_that_exists() {
        let mut archive = Message::default();
        archive.set(11, Value::Bytes(Message::default().encode()));

        set_path(
            &mut archive,
            &[11, 12],
            Some(Value::Fixed32(18.0f32.to_le_bytes())),
        )
        .unwrap();
        assert_eq!(
            get_path(&archive, &[11, 12]),
            Some(Value::Fixed32(18.0f32.to_le_bytes()))
        );

        set_path(&mut archive, &[11, 1], Some(Value::Varint(1))).unwrap();
        assert_eq!(get_path(&archive, &[11, 1]), Some(Value::Varint(1)));

        set_path(&mut archive, &[11, 12], None).unwrap();
        assert_eq!(get_path(&archive, &[11, 12]), None);
        assert_eq!(get_path(&archive, &[11, 1]), Some(Value::Varint(1)));
    }

    /// A container this crate invents holds only what it was asked for, and a
    /// colour of `{3: r, 4: g, 5: b}` — no model, no alpha — crashes Pages.
    #[test]
    fn set_path_will_not_invent_a_container() {
        let mut archive = Message::default();
        assert!(set_path(&mut archive, &[11, 7, 3], Some(Value::Varint(1))).is_err());
        assert!(archive.fields.is_empty(), "and nothing was written");

        // Clearing something that is not there is a no-op, not an error.
        set_path(&mut archive, &[11, 7, 3], None).unwrap();
        assert!(archive.fields.is_empty());
    }

    /// A path that would have to tunnel through a string is an error, not a
    /// silent overwrite.
    #[test]
    fn set_path_refuses_to_tunnel_through_a_string() {
        let mut archive = Message::default();
        archive.set(1, Value::Bytes(b"Grosse Uberschrift".to_vec()));
        assert!(set_path(&mut archive, &[1, 2], Some(Value::Varint(1))).is_err());
        archive.set(2, Value::Varint(7));
        assert!(set_path(&mut archive, &[2, 1], Some(Value::Varint(1))).is_err());
    }

    #[test]
    fn applying_to_the_middle_restores_what_followed() {
        let mut t = table(&[(0, 100), (40, 200)]);
        apply(&mut t, 10..20, 300, 60);
        assert_eq!(
            shape(&t),
            vec![
                (0, Some(100)),
                (10, Some(300)),
                (20, Some(100)),
                (40, Some(200)),
            ]
        );
    }

    #[test]
    fn applying_across_runs_replaces_them() {
        let mut t = table(&[(0, 100), (10, 200), (20, 300), (40, 400)]);
        apply(&mut t, 5..30, 999, 60);
        assert_eq!(
            shape(&t),
            vec![
                (0, Some(100)),
                (5, Some(999)),
                (30, Some(300)),
                (40, Some(400)),
            ]
        );
    }

    #[test]
    fn applying_to_the_tail_adds_no_trailing_entry() {
        let mut t = table(&[(0, 100), (10, 200)]);
        apply(&mut t, 10..60, 300, 60);
        assert_eq!(shape(&t), vec![(0, Some(100)), (10, Some(300))]);
    }

    #[test]
    fn applying_to_everything_leaves_one_run() {
        let mut t = table(&[(0, 100), (10, 200), (20, 300)]);
        apply(&mut t, 0..60, 400, 60);
        assert_eq!(shape(&t), vec![(0, Some(400))]);
    }

    #[test]
    fn applying_to_an_empty_table_starts_one() {
        let mut t = Message::default();
        apply(&mut t, 0..10, 500, 10);
        assert_eq!(shape(&t), vec![(0, Some(500))]);
    }

    /// A range ending exactly where a run already starts must not gain a
    /// duplicate entry at that index.
    #[test]
    fn applying_up_to_an_existing_boundary_adds_nothing_extra() {
        let mut t = table(&[(0, 100), (20, 200)]);
        apply(&mut t, 5..20, 300, 60);
        assert_eq!(
            shape(&t),
            vec![(0, Some(100)), (5, Some(300)), (20, Some(200))]
        );
    }

    #[test]
    fn an_empty_or_backwards_range_changes_nothing() {
        let mut t = table(&[(0, 100), (10, 200)]);
        let before = t.clone();
        apply(&mut t, 5..5, 300, 60);
        // Built rather than written as a literal: a backwards range is exactly
        // what this asserts is harmless, and clippy rejects the literal form.
        let backwards = Range { start: 20, end: 10 };
        apply(&mut t, backwards, 300, 60);
        assert_eq!(t, before);
    }

    #[test]
    fn the_range_is_clamped_into_the_text() {
        let mut t = table(&[(0, 100)]);
        apply(&mut t, 5..900, 300, 10);
        assert_eq!(shape(&t), vec![(0, Some(100)), (5, Some(300))]);
    }

    /// Whatever else an entry carries has to survive being re-pointed.
    #[test]
    fn applying_preserves_fields_the_crate_does_not_understand() {
        let mut t = Message::default();
        let mut rich = entry(0, 100);
        rich.set(7, Value::Varint(42));
        t.fields.push(Field {
            number: 1,
            value: Value::Bytes(rich.encode()),
        });
        apply(&mut t, 2..4, 200, 10);

        let kept: Vec<_> = entries(&t).iter().map(|(_, e)| e.varint(7)).collect();
        assert_eq!(kept, vec![Some(42), Some(42), Some(42)]);
    }

    /// Styling a range that already has that style leaves no trace, and a run
    /// re-pointed onto its neighbour's style merges into it.
    #[test]
    fn runs_that_repeat_the_run_before_them_are_dropped() {
        let mut t = table(&[(0, 100), (10, 200)]);
        apply(&mut t, 2..6, 100, 60);
        assert_eq!(shape(&t), vec![(0, Some(100)), (10, Some(200))]);

        let mut t = table(&[(0, 100), (10, 200), (20, 300)]);
        assert_eq!(repoint(&mut t, 200, Some(100)), 1);
        assert_eq!(shape(&t), vec![(0, Some(100)), (20, Some(300))]);
    }

    /// Coalescing goes by the whole entry, not by the style reference, so an
    /// entry that differs in a field this crate does not model survives.
    #[test]
    fn runs_differing_in_an_unmodelled_field_are_kept_apart() {
        let mut t = Message::default();
        t.fields.push(Field {
            number: 1,
            value: Value::Bytes(entry(0, 100).encode()),
        });
        let mut annotated = entry(10, 100);
        annotated.set(7, Value::Varint(1));
        t.fields.push(Field {
            number: 1,
            value: Value::Bytes(annotated.encode()),
        });

        apply(&mut t, 30..40, 200, 60);
        assert_eq!(
            shape(&t),
            vec![
                (0, Some(100)),
                (10, Some(100)),
                (30, Some(200)),
                (40, Some(100))
            ]
        );
    }

    #[test]
    fn a_run_starting_where_the_text_ends_is_dropped() {
        let mut t = table(&[(0, 100), (10, 200)]);
        apply(&mut t, 5..10, 200, 10);
        assert_eq!(shape(&t), vec![(0, Some(100)), (5, Some(200))]);
    }

    #[test]
    fn repoint_redirects_or_drops() {
        let mut t = table(&[(0, 100), (10, 200), (20, 100)]);
        assert_eq!(repoint(&mut t, 100, Some(999)), 2);
        assert_eq!(
            shape(&t),
            vec![(0, Some(999)), (10, Some(200)), (20, Some(999))]
        );

        // Dropping lets the run before extend over the one that went — except
        // at 0, where there is no run before and the table would begin late,
        // leaving the first characters with no attribute at all. What is left
        // there is the entry with its object taken off, which is the format's
        // "whatever was in force here".
        let mut t = table(&[(0, 100), (10, 200), (20, 100)]);
        assert_eq!(repoint(&mut t, 100, None), 2);
        assert_eq!(shape(&t), vec![(0, None), (10, Some(200))]);
    }

    #[test]
    fn registrations_clone_and_remove_only_bare_references() {
        let mut sheet = Message::default();
        sheet.fields.push(Field {
            number: 2,
            value: Value::Bytes(reference(70).encode()),
        });
        // A keyed entry: "body" -> style 70. Must be left alone by both.
        let mut keyed = Message::default();
        keyed.set(1, Value::Bytes(b"body".to_vec()));
        keyed.set(2, Value::Bytes(reference(70).encode()));
        sheet.fields.push(Field {
            number: 3,
            value: Value::Bytes(keyed.encode()),
        });

        assert_eq!(clone_registrations(&mut sheet, 70, 71), 1);
        assert_eq!(count_references(&sheet, 71), 1);
        assert_eq!(sheet.fields.len(), 3);

        assert_eq!(remove_registrations(&mut sheet, 70), 1);
        assert_eq!(sheet.fields.len(), 2, "the keyed entry stays");
        assert_eq!(count_references(&sheet, 70), 1);
    }
}
