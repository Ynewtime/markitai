//! `KN` — the Keynote show: slides, layouts, placeholders, notes, transitions.
//!
//! Only Keynote writes these archives, and the numbering is **app-scoped**: a
//! Keynote type 5 is `KN.SlideArchive` and a Numbers type 5 is something else
//! entirely. [`crate::registry`] resolves them per document kind; nothing here
//! may be used on a document [`crate::Document::kind`] does not call Keynote.
//!
//! ```text
//!  KN.DocumentArchive (1)
//!    └── show (2) ─── KN.ShowArchive (2)
//!          ├── theme (2) ──── KN.ThemeArchive (10)
//!          │                    ├── super.name — "21_BasicWhite"
//!          │                    └── templates (2): the slide layouts, in order,
//!          │                        each a KN.SlideNodeArchive
//!          ├── slideTree (3) — inline KN.SlideTreeArchive
//!          │                    └── slides (2): the deck, in order, each a
//!          │                        KN.SlideNodeArchive
//!          ├── size (4), stylesheet (5), slideNumbersVisible (6)
//!          └── soundtrack (17)
//!
//!  KN.SlideNodeArchive (4)      one per slide *and* one per layout
//!    ├── slide (2) ──────────── KN.SlideArchive (5)
//!    ├── isSkipped (4), hasNote (8), hasTransition (7)
//!    ├── thumbnails (16) / thumbnailSizes (10) / thumbnailsAreDirty (14)
//!    └── template_slide_id (29): the *layout's* UUID, not the node's
//!
//!  KN.SlideArchive (5)          a slide and a layout are the same archive
//!    ├── style (1), transition (4), template_slide (17), note (27)
//!    ├── titlePlaceholder (5), bodyPlaceholder (6),
//!    │   slideNumberPlaceholder (20), objectPlaceholder (30)
//!    ├── owned_drawables (7), drawables_z_order (42)
//!    ├── builds (2), buildChunks (43)
//!    └── name (10) — only a layout has one
//! ```
//!
//! ## A slide is a component
//!
//! Each slide is its own `Index/Slide*.iwa` stream and its own component, whose
//! identifier is the `KN.SlideArchive`'s own; each layout is an
//! `Index/TemplateSlide-*.iwa`. The slide's *node* lives in `Index/Document.iwa`
//! with the show. That split is what makes duplicating a slide a package
//! operation rather than an object one — see [`duplicate_slide`].
//!
//! ## What "showing" means
//!
//! Keynote's `title showing` and `body showing` are not flags. A placeholder is
//! shown when the slide **owns** it — when it appears in `owned_drawables` (7) —
//! and the reference at field 5 or 6 stays whatever the layout gave it either
//! way. That is measured, not assumed: over the six slides of `keynote-deck`
//! the app's `title showing` / `body showing` agree with membership of field 7
//! twelve times out of twelve, including the "Statement" slide whose title
//! placeholder still holds text the app does not draw.
//!
//! ## Slide numbers
//!
//! A skipped slide has no number. The app answers `slide number` with **-1** for
//! it, and numbers the rest 1, 2, 3 … skipping over it — so the number is a
//! function of the deck, not a field on the slide. [`Slide::number`] is `None`
//! for a skipped slide for exactly that reason.

use std::collections::{BTreeMap, BTreeSet};

use crate::pb::{decode_nested, Message, Value};
use crate::style::reference_target;
use crate::Error;

/// `KN.DocumentArchive`.
pub const TYPE_DOCUMENT: u32 = 1;
/// `KN.ShowArchive`.
pub const TYPE_SHOW: u32 = 2;
/// `KN.UIStateArchive`.
pub const TYPE_UI_STATE: u32 = 3;
/// `KN.SlideNodeArchive` — one per slide and one per layout.
pub const TYPE_SLIDE_NODE: u32 = 4;
/// `KN.SlideArchive` — a slide *or* a layout.
pub const TYPE_SLIDE: u32 = 5;
/// `KN.PlaceholderArchive`.
pub const TYPE_PLACEHOLDER: u32 = 7;
/// `KN.BuildArchive` — one animation of one drawable.
pub const TYPE_BUILD: u32 = 8;
/// `KN.SlideStyleArchive`.
pub const TYPE_SLIDE_STYLE: u32 = 9;
/// `KN.ThemeArchive`.
pub const TYPE_THEME: u32 = 10;
/// `KN.NoteArchive` — the presenter notes wrapper; field 1 is the storage.
pub const TYPE_NOTE: u32 = 15;
/// `KN.RecordingArchive` — a recorded presentation. Read and passed through,
/// never authored (ground rule 8).
pub const TYPE_RECORDING: u32 = 16;
/// `KN.ClassicStylesheetRecordArchive`.
pub const TYPE_CLASSIC_STYLESHEET_RECORD: u32 = 19;
/// `KN.Soundtrack`.
pub const TYPE_SOUNDTRACK: u32 = 21;
/// `KN.SlideNumberAttachmentArchive`.
pub const TYPE_SLIDE_NUMBER_ATTACHMENT: u32 = 22;
/// `KN.MotionBackgroundStyleArchive`.
pub const TYPE_MOTION_BACKGROUND_STYLE: u32 = 26;
/// `KN.BuildChunkArchive` — one stage of a build.
pub const TYPE_BUILD_CHUNK: u32 = 153;

/// Field numbers of `KN.DocumentArchive`.
pub mod document_field {
    pub const SHOW: u32 = 2;
    /// `TSA.DocumentArchive`. **Field 3, not 15 or 8** — see
    /// [`crate::metadata::super_field`].
    pub const SUPER: u32 = 3;
    pub const CUSTOM_FORMAT_LIST: u32 = 4;
}

/// Field numbers of `KN.ShowArchive`.
pub mod show_field {
    pub const UI_STATE: u32 = 1;
    pub const THEME: u32 = 2;
    /// An inline `KN.SlideTreeArchive`, not a reference.
    pub const SLIDE_TREE: u32 = 3;
    pub const SIZE: u32 = 4;
    pub const STYLESHEET: u32 = 5;
    pub const SLIDE_NUMBERS_VISIBLE: u32 = 6;
    pub const RECORDING: u32 = 7;
    pub const LOOP: u32 = 8;
    pub const MODE: u32 = 9;
    pub const AUTOPLAY_TRANSITION_DELAY: u32 = 10;
    pub const AUTOPLAY_BUILD_DELAY: u32 = 11;
    pub const IDLE_TIMER_ACTIVE: u32 = 15;
    pub const IDLE_TIMER_DELAY: u32 = 16;
    pub const SOUNDTRACK: u32 = 17;
    pub const PLAYS_ON_OPEN: u32 = 18;
    pub const SLIDE_LIST: u32 = 19;
}

/// Field numbers of `KN.SlideTreeArchive`, which sits inline in the show.
pub mod slide_tree_field {
    /// `rootSlideNode`, deprecated and absent from every deck here.
    pub const ROOT: u32 = 1;
    /// The deck, in order. **This repeated field is the slide order** — see
    /// [`crate::Document::move_slide`].
    pub const SLIDES: u32 = 2;
}

/// Field numbers of `KN.SlideNodeArchive`.
pub mod node_field {
    pub const CHILDREN: u32 = 1;
    pub const SLIDE: u32 = 2;
    /// `isSkipped` — the whole of "skip this slide".
    pub const SKIPPED: u32 = 4;
    pub const HAS_BUILDS: u32 = 6;
    pub const HAS_TRANSITION: u32 = 7;
    pub const HAS_NOTE: u32 = 8;
    pub const THUMBNAIL_SIZES: u32 = 10;
    pub const COPY_FROM_SLIDE: u32 = 12;
    pub const THUMBNAILS_DIRTY: u32 = 14;
    pub const BUILD_EVENT_COUNT: u32 = 15;
    pub const THUMBNAILS: u32 = 16;
    pub const SLIDE_NUMBER_VISIBLE: u32 = 18;
    pub const HAS_EXPLICIT_BUILDS: u32 = 20;
    pub const DEPTH: u32 = 21;
    /// `template_slide_id` — a `TSP.UUID`, and the *layout's*, so two slides on
    /// one layout carry the same value.
    pub const TEMPLATE_SLIDE_ID: u32 = 29;
}

/// Field numbers of `KN.SlideArchive`.
pub mod slide_field {
    pub const STYLE: u32 = 1;
    pub const BUILDS: u32 = 2;
    pub const TRANSITION: u32 = 4;
    pub const TITLE_PLACEHOLDER: u32 = 5;
    pub const BODY_PLACEHOLDER: u32 = 6;
    /// `owned_drawables` — and, as measured, what "showing" means.
    pub const OWNED_DRAWABLES: u32 = 7;
    pub const NAME: u32 = 10;
    pub const TEMPLATE_SLIDE: u32 = 17;
    pub const IN_DOCUMENT: u32 = 19;
    pub const SLIDE_NUMBER_PLACEHOLDER: u32 = 20;
    pub const NOTE: u32 = 27;
    pub const SAGE_TAG_MAP: u32 = 28;
    pub const OBJECT_PLACEHOLDER: u32 = 30;
    pub const BODY_PARAGRAPH_STYLES: u32 = 31;
    pub const BODY_LIST_STYLES: u32 = 35;
    pub const GUIDE_STORAGE: u32 = 36;
    pub const DRAWABLES_Z_ORDER: u32 = 42;
    pub const BUILD_CHUNKS: u32 = 43;
    pub const INSTRUCTIONAL_TEXT: u32 = 45;
}

/// Field numbers of `KN.ThemeArchive`.
pub mod theme_field {
    /// `TSS.ThemeArchive`; its field 3 is the theme's stored name.
    pub const SUPER: u32 = 1;
    /// The slide layouts, in the order the app lists them, each a node.
    pub const TEMPLATES: u32 = 2;
    pub const UUID: u32 = 3;
    pub const DEFAULT_TEMPLATE_NODE: u32 = 5;
    pub const LIVE_VIDEO_SOURCES: u32 = 9;
    pub const MOTION_BACKGROUND_PRESETS: u32 = 10;
    /// Inside the `TSS.ThemeArchive` super: the theme's stored name.
    pub const SUPER_NAME: u32 = 3;
    /// Inside the super: the document stylesheet.
    pub const SUPER_STYLESHEET: u32 = 4;
}

/// Field numbers of `KN.PlaceholderArchive`.
pub mod placeholder_field {
    /// `TSWP.ShapeInfoArchive`.
    pub const SUPER: u32 = 1;
    pub const KIND: u32 = 2;
}

/// The chain from a slide to what a transition is made of:
/// `KN.TransitionArchive.attributes (2)` → `KN.TransitionAttributesArchive`,
/// whose field 8 is the `KN.AnimationAttributesArchive` shared with builds and
/// whose fields 9–20 are the per-effect `custom_*` block.
pub mod transition_field {
    pub const ATTRIBUTES: u32 = 2;

    // -- KN.TransitionAttributesArchive -------------------------------------
    pub const ANIMATION_ATTRIBUTES: u32 = 8;
    pub const CUSTOM_TWIST: u32 = 9;
    pub const CUSTOM_MOSAIC_SIZE: u32 = 10;
    pub const CUSTOM_MOSAIC_TYPE: u32 = 11;
    pub const CUSTOM_BOUNCE: u32 = 12;
    pub const CUSTOM_MAGIC_MOVE_FADE_UNMATCHED: u32 = 13;
    pub const CUSTOM_TIMING_CURVE: u32 = 15;
    pub const CUSTOM_TEXT_DELIVERY: u32 = 16;
    pub const CUSTOM_MOTION_BLUR: u32 = 17;
    pub const CUSTOM_TRAVEL_DISTANCE: u32 = 18;
    pub const CUSTOM_ANGLE: u32 = 19;
    pub const CUSTOM_BLUR_AMOUNT: u32 = 20;

    // -- KN.AnimationAttributesArchive --------------------------------------
    pub const ANIMATION_TYPE: u32 = 1;
    pub const EFFECT: u32 = 2;
    pub const DURATION: u32 = 3;
    pub const DIRECTION: u32 = 4;
    pub const DELAY: u32 = 5;
    pub const AUTOMATIC: u32 = 6;
    pub const COLOR: u32 = 7;
    pub const SEED: u32 = 11;
    pub const CUSTOM_DETAIL: u32 = 12;
    pub const RTL: u32 = 16;

    /// Every field of `KN.TransitionAttributesArchive` the 15.3.1 schema names:
    /// 1–7 are the deprecated `database_*` copies of the animation attributes,
    /// 8 is the animation attributes themselves, and the rest is the `custom_*`
    /// block. **14 is a hole** — the schema skips it — which is why a decoder
    /// that reports what it did not recognise has something to say about it.
    pub const NAMED: [u32; 19] = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
    ];
}

/// Field numbers of `KN.BuildArchive` (8) and its attributes.
///
/// **Partly measured at last.** `keynote-builds.key` carries eight builds —
/// four Dissolve build-ins and four Disappear build-outs — made from the
/// Animate inspector on an unlocked screen, so `delivery`, `eventTrigger`, the
/// animation attributes and the `animation_type` that tells an "In" from an
/// "Out" are Confirmed against the fixture. None of the 182 bundled themes has
/// a build, and Keynote's dictionary still has no build vocabulary, so the
/// action-build, motion-path, by-bullet-group and chunk-timing fields below
/// remain schema-only — decoded anyway so a deck from outside is *reported*,
/// and pinned by [`tests`](crate::keynote) against the fixture's eight.
pub mod build_field {
    pub const DRAWABLE: u32 = 1;
    /// `required string delivery` — carried verbatim, because nothing here
    /// knows what its values look like.
    pub const DELIVERY: u32 = 2;
    pub const ATTRIBUTES: u32 = 4;
    pub const CHUNK_ID_SEED: u32 = 5;

    // -- KN.BuildAttributesArchive ------------------------------------------
    pub const EVENT_TRIGGER: u32 = 4;
    pub const ANIMATION_ATTRIBUTES: u32 = 18;
    pub const START_OFFSET: u32 = 27;
    pub const END_OFFSET: u32 = 28;
    pub const CUSTOM_TEXT_DELIVERY: u32 = 20;
    pub const CUSTOM_DELIVERY_OPTION: u32 = 21;
    pub const ACTION_MOTION_PATH: u32 = 22;

    // -- KN.BuildChunkArchive (153) -----------------------------------------
    pub const CHUNK_BUILD: u32 = 1;
    pub const CHUNK_DELAY: u32 = 3;
    pub const CHUNK_DURATION: u32 = 4;
    pub const CHUNK_AUTOMATIC: u32 = 5;
    pub const CHUNK_REFERENT: u32 = 6;
}

/// Field numbers of `KN.Soundtrack` (21).
pub mod soundtrack_field {
    pub const VOLUME: u32 = 1;
    pub const MODE: u32 = 2;
    /// `repeated TSP.DataReference movie_media` — **an ordered track list**,
    /// each entry a bare `{1: data identifier}` into the media registry.
    pub const MOVIE_MEDIA: u32 = 3;
}

/// Field numbers of `KN.RecordingArchive` (16) — read and passed through,
/// never authored (ground rule 8).
pub mod recording_field {
    pub const EVENT_TRACKS: u32 = 1;
    pub const MOVIE_TRACK: u32 = 2;
    pub const DURATION: u32 = 3;
    pub const MODIFICATION_DATE: u32 = 5;
}

/// Field numbers of `KN.LiveVideoSource` (184) and its collection (185).
pub mod live_video_field {
    pub const NAME: u32 = 1;
    pub const POSTER_IMAGE: u32 = 4;
    pub const IS_DEFAULT: u32 = 8;
    pub const COLLECTION_SOURCES: u32 = 1;
    pub const COLLECTION_DEFAULT: u32 = 2;
}

/// What a `KN.PlaceholderArchive` stands in for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlaceholderKind {
    /// `kKindPlaceholder` — a placeholder with no special role.
    Generic,
    SlideNumber,
    Title,
    Body,
    /// `kKindObjectPlaceholder` — the "drag an image here" well.
    Object,
    /// A value the 15.3.1 schema does not name.
    Unknown(u64),
}

impl PlaceholderKind {
    pub fn of(value: u64) -> PlaceholderKind {
        match value {
            0 => PlaceholderKind::Generic,
            1 => PlaceholderKind::SlideNumber,
            2 => PlaceholderKind::Title,
            3 => PlaceholderKind::Body,
            4 => PlaceholderKind::Object,
            other => PlaceholderKind::Unknown(other),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            PlaceholderKind::Generic => "placeholder",
            PlaceholderKind::SlideNumber => "slide number",
            PlaceholderKind::Title => "title",
            PlaceholderKind::Body => "body",
            PlaceholderKind::Object => "object",
            PlaceholderKind::Unknown(_) => "unknown",
        }
    }
}

/// What a text storage on a slide is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    Title,
    Body,
    SlideNumber,
    ObjectPlaceholder,
    /// A placeholder the theme left unclassified.
    Placeholder,
    /// Presenter notes — a storage of kind 4, reached from `KN.NoteArchive`.
    Notes,
    /// A text item or shape the user put on the slide.
    TextBox,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Title => "title",
            Role::Body => "body",
            Role::SlideNumber => "slide number",
            Role::ObjectPlaceholder => "object placeholder",
            Role::Placeholder => "placeholder",
            Role::Notes => "notes",
            Role::TextBox => "text box",
        }
    }
}

/// One `KN.PlaceholderArchive` and the text it holds.
#[derive(Debug, Clone)]
pub struct Placeholder {
    pub identifier: u64,
    pub kind: PlaceholderKind,
    /// The `TSWP.StorageArchive` it lays out. This is what
    /// [`crate::Document::set_text`] takes.
    pub storage: Option<u64>,
    pub text: String,
    /// Whether the slide owns it — the app's `title showing` / `body showing`.
    pub shown: bool,
}

/// One text storage on a slide, and what it is for.
#[derive(Debug, Clone)]
pub struct SlideText {
    pub storage: u64,
    pub role: Role,
    /// The drawable that lays it out, when one does. Notes have none.
    pub drawable: Option<u64>,
    pub text: String,
}

/// Every transition effect Keynote 15.3.1 has, as `(the app's name, the
/// identifier on the wire)`.
///
/// The names are the `transition effects` enumeration of the scripting
/// dictionary and the identifiers are the `cocoa string-value` beside each one;
/// the pairing is the app's own. It is not taken on trust either:
/// `keynote-transitions` sets all 44 from a script, and
/// `tests/keynote.rs` makes Keynote read every one of them back and compares
/// the name it gives with the identifier this table maps it to.
///
/// The order is the dictionary's, which is the order the app's Animate
/// inspector lists them in: the four text-and-object effects first, then the
/// object effects, then the slide effects.
pub const EFFECTS: [(&str, &str); 44] = [
    ("no transition effect", "none"),
    ("magic move", "apple:magic-move-implied-motion-path"),
    ("shimmer", "apple:ca-text-shimmer"),
    ("sparkle", "apple:ca-text-sparkle"),
    ("swing", "apple:ca-swing"),
    ("object cube", "apple:ca-cube"),
    ("object flip", "apple:ca-dissolve-and-flip"),
    ("object pop", "apple:ca-pop"),
    ("object push", "apple:ca-push"),
    ("object revolve", "apple:ca-revolve"),
    ("object zoom", "apple:ca-zoom"),
    ("perspective", "apple:ca-isometric"),
    ("clothesline", "apple:ClotheslinePush"),
    ("confetti", "com.apple.iWork.Keynote.KLNConfetti"),
    ("dissolve", "apple:dissolve"),
    ("drop", "apple:bounce"),
    ("droplet", "apple:droplet"),
    (
        "fade through color",
        "com.apple.iWork.Keynote.BLTFadeThruColor",
    ),
    ("grid", "apple:apple-grid"),
    ("iris", "apple:wipe-iris"),
    ("move in", "apple:slide"),
    ("push", "apple:push"),
    ("reveal", "apple:reveal"),
    ("switch", "apple:FlipThrough"),
    ("wipe", "apple:wipe"),
    ("blinds", "com.apple.iWork.Keynote.BLTBlinds"),
    ("color planes", "com.apple.iWork.Keynote.KLNColorPlanes"),
    ("cube", "apple:3D-cube"),
    ("doorway", "apple:doorway"),
    ("fall", "apple:fall"),
    ("flip", "apple:revolve"),
    ("flop", "com.apple.iWork.Keynote.BUKFlop"),
    ("mosaic", "com.apple.iWork.Keynote.BLTMosaicFlip"),
    ("page flip", "apple:pageflip"),
    ("pivot", "apple:pivot"),
    ("reflection", "com.apple.iWork.Keynote.BLTReflection"),
    ("revolving door", "com.apple.iWork.Keynote.BLTRevolvingDoor"),
    ("scale", "apple:scale"),
    ("swap", "com.apple.iWork.Keynote.KLNSwap"),
    ("swoosh", "com.apple.iWork.Keynote.BLTSwoosh"),
    ("twirl", "apple:twirl"),
    ("twist", "com.apple.iWork.Keynote.BUKTwist"),
    ("fade and move", "apple:fade-and-move"),
    ("radial wipe", "apple:radial wipe"),
];

/// What the app calls this effect identifier, if it is one the app has.
///
/// **`apple:revolve` is "flip" and `apple:ca-revolve` is "object revolve"** —
/// three of the identifiers differ from their names by nothing a reader could
/// guess, which is why the table is the app's and not a transformation.
pub fn effect_name(identifier: &str) -> Option<&'static str> {
    EFFECTS
        .iter()
        .find(|(_, wire)| *wire == identifier)
        .map(|(name, _)| *name)
}

/// Which way a transition travels — `AnimationAttributesArchive.direction` (4).
///
/// **Keynote's own writer never emits this field.** Every one of the 44 effects
/// set through `transition properties` leaves it absent, which is 0: "whatever
/// this effect does by default". The values below were got the only way this
/// corpus can get them — by handing Keynote a PowerPoint file with a direction
/// on it and reading back what the importer wrote (see FORMAT.md §13). Two
/// families, orthogonal and diagonal, and they are consistent across `apple:push`,
/// `apple:wipe`, `apple:slide` and `com.apple.iWork.Keynote.BLTBlinds`.
pub fn direction_name(direction: u64) -> &'static str {
    match direction {
        0 => "the effect's own default",
        11 => "left to right",
        12 => "right to left",
        13 => "top to bottom",
        14 => "bottom to top",
        21 => "top left to bottom right",
        22 => "top right to bottom left",
        23 => "bottom left to top right",
        24 => "bottom right to top left",
        _ => "unknown",
    }
}

/// `TransitionCustomAttributesTimingCurveType` — the "Acceleration" popup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingCurve {
    Linear,
    EaseIn,
    EaseOut,
    EaseInEaseOut,
    Custom,
    Unknown(u64),
}

impl TimingCurve {
    pub fn of(value: u64) -> TimingCurve {
        match value {
            1 => TimingCurve::Linear,
            2 => TimingCurve::EaseIn,
            3 => TimingCurve::EaseOut,
            4 => TimingCurve::EaseInEaseOut,
            5 => TimingCurve::Custom,
            other => TimingCurve::Unknown(other),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            TimingCurve::Linear => "linear",
            TimingCurve::EaseIn => "ease in",
            TimingCurve::EaseOut => "ease out",
            TimingCurve::EaseInEaseOut => "ease in, ease out",
            TimingCurve::Custom => "custom",
            TimingCurve::Unknown(_) => "unknown",
        }
    }
}

/// `TransitionCustomAttributesTextDeliveryType` — Magic Move's text
/// granularity, the app's "Text: Blend / Word / Character" popup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextDelivery {
    /// Blend the whole block. The default Keynote writes for Magic Move.
    ByObject,
    ByWord,
    ByCharacter,
    ByLine,
    Unknown(u64),
}

impl TextDelivery {
    pub fn of(value: u64) -> TextDelivery {
        match value {
            1 => TextDelivery::ByObject,
            2 => TextDelivery::ByWord,
            3 => TextDelivery::ByCharacter,
            4 => TextDelivery::ByLine,
            other => TextDelivery::Unknown(other),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            TextDelivery::ByObject => "by object",
            TextDelivery::ByWord => "by word",
            TextDelivery::ByCharacter => "by character",
            TextDelivery::ByLine => "by line",
            TextDelivery::Unknown(_) => "unknown",
        }
    }
}

/// The `custom_*` block of `KN.TransitionAttributesArchive` — the parameters
/// that belong to the *effect* rather than to the timing.
///
/// Every one is an `Option` because **Keynote writes only the parameters the
/// chosen effect has**, and it writes them whatever their value: `apple:scale`
/// and `apple:ca-revolve` both carry `custom_bounce` = false, which an absent-
/// means-false decoder would have reported as "no such parameter". Measured by
/// setting all 44 effects on 44 otherwise identical slides — `keynote-transitions`
/// — and diffing; the two control slides prove the block does not move when the
/// duration, the delay or the automatic flag do.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TransitionParameters {
    /// `custom_twist` (9) — degrees. 3.3 on `BUKTwist`.
    pub twist: Option<f32>,
    /// `custom_mosaic_size` (10) and `custom_mosaic_type` (11). Named from the
    /// schema; `BLTMosaicFlip` was set from a script and wrote neither.
    pub mosaic_size: Option<u64>,
    pub mosaic_type: Option<u64>,
    /// `custom_bounce` (12) — the "Bounce" checkbox of the object effects.
    pub bounce: Option<bool>,
    /// `custom_magic_move_fade_unmatched_objects` (13) — Magic Move's "Fade
    /// Unmatched Objects", true by default.
    pub magic_move_fade_unmatched: Option<bool>,
    /// `custom_timing_curve` (15) — Magic Move's Acceleration.
    pub timing_curve: Option<TimingCurve>,
    /// `custom_text_delivery_type` (16) — Magic Move's text granularity.
    pub text_delivery: Option<TextDelivery>,
    /// `custom_motion_blur` (17). Named from the schema; unwritten by any of
    /// the 44 effects.
    pub motion_blur: Option<bool>,
    /// `custom_travel_distance` (18) — 1.0 on `apple:fade-and-move`.
    pub travel_distance: Option<f32>,
    /// `custom_angle` (19) — degrees; 90 on `apple:radial wipe`.
    pub angle: Option<f32>,
    /// `custom_blur_amount` (20) — 0.5 on `apple:radial wipe`.
    pub blur_amount: Option<f32>,
}

impl TransitionParameters {
    /// Did the effect bring any parameters of its own?
    pub fn is_empty(&self) -> bool {
        *self == TransitionParameters::default()
    }

    /// One line per parameter that is present, for a listing.
    pub fn describe(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(v) = self.twist {
            out.push(format!("twist {v}°"));
        }
        if let Some(v) = self.mosaic_size {
            out.push(format!("mosaic size {v}"));
        }
        if let Some(v) = self.mosaic_type {
            out.push(format!("mosaic type {v}"));
        }
        if let Some(v) = self.bounce {
            out.push(format!("bounce {v}"));
        }
        if let Some(v) = self.magic_move_fade_unmatched {
            out.push(format!("fade unmatched {v}"));
        }
        if let Some(v) = self.timing_curve {
            out.push(format!("acceleration {}", v.as_str()));
        }
        if let Some(v) = self.text_delivery {
            out.push(format!("text {}", v.as_str()));
        }
        if let Some(v) = self.motion_blur {
            out.push(format!("motion blur {v}"));
        }
        if let Some(v) = self.travel_distance {
            out.push(format!("travel {v}"));
        }
        if let Some(v) = self.angle {
            out.push(format!("angle {v}°"));
        }
        if let Some(v) = self.blur_amount {
            out.push(format!("blur {v}"));
        }
        out
    }
}

/// A slide's transition, in full: the animation attributes shared with builds
/// and the `custom_*` block that belongs to the effect.
///
/// **A transition belongs to the slide it leaves.** Keynote plays a slide's
/// transition on the way *out* of it; PowerPoint's belongs to the slide being
/// entered, and Keynote's own importer shifts the whole deck by one to say so —
/// which is how the direction values in [`direction_name`] were measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    /// `AnimationAttributesArchive.effect` — `"none"`, or the identifier the
    /// app's dictionary lists: `"apple:dissolve"`, `"apple:magic-move…"`,
    /// `"com.apple.iWork.Keynote.BLTFadeThruColor"`.
    pub effect: String,
    /// `animation_type` — `"Transition"` on every slide transition seen, and
    /// `"In"` / `"Out"` on a build, which is what tells a build-in from a
    /// build-out (`keynote-builds.key`: four dissolves say `"In"`, four
    /// disappears say `"Out"`).
    pub animation_type: String,
    pub duration: f64,
    pub delay: f64,
    /// Advance without a click.
    pub automatic: bool,
    /// Which way it travels; 0 is the effect's own default, and 0 is what the
    /// app writes. [`direction_name`] names the values.
    pub direction: u64,
    /// `random_number_seed` — different on every slide, and copied verbatim by
    /// the app's own duplicate.
    pub seed: u64,
    /// `color` (7) — the colour `com.apple.iWork.Keynote.BLTFadeThruColor`
    /// fades through, and the only effect of the 44 that writes one.
    pub color: Option<crate::drawable::Color>,
    /// `writing_direction_is_rtl` (16), 0 everywhere here.
    pub rtl: bool,
    /// The per-effect parameters.
    pub parameters: TransitionParameters,
    /// Field numbers of `KN.TransitionAttributesArchive` the 15.3.1 schema does
    /// not name — **the tripwire**. Empty over every deck and every bundled
    /// theme; a non-empty one means this table has gone out of date.
    pub unknown_parameters: Vec<u32>,
}

impl Transition {
    /// Does the slide have a transition at all? `"none"` is the empty one.
    pub fn is_none(&self) -> bool {
        self.effect.is_empty() || self.effect == "none"
    }

    /// Which way it travels, in words.
    pub fn direction_name(&self) -> &'static str {
        direction_name(self.direction)
    }
}

impl Default for Transition {
    fn default() -> Transition {
        Transition {
            effect: "none".into(),
            animation_type: String::new(),
            duration: 0.0,
            delay: 0.0,
            automatic: false,
            direction: 0,
            seed: 0,
            color: None,
            rtl: false,
            parameters: TransitionParameters::default(),
            unknown_parameters: Vec::new(),
        }
    }
}

/// One `KN.BuildArchive` (8) — one animation of one drawable.
///
/// **Measured, for the parts a fixture could reach.** `keynote-builds.key`
/// carries eight builds, and against them `delivery`, `event_trigger`, the
/// animation attributes and the `animation_type` string (`"In"` / `"Out"`,
/// where a transition writes `"Transition"`) are Confirmed. The effect
/// identifiers are not the menu names — Dissolve stores `apple:dissolve
/// character` and Disappear stores `apple:bc-appear`. The remaining fields
/// (action builds, motion paths, by-bullet-group ranges) stay schema-only:
/// none of the seven decks here nor the 182 bundled themes exercises them, and
/// Keynote's dictionary has no build vocabulary, so `iwork slides` reports what
/// a build says and `tests/keynote.rs` pins the fixture's eight.
#[derive(Debug, Clone, PartialEq)]
pub struct Build {
    pub identifier: u64,
    /// `drawable` (1) — what is animated.
    pub drawable: Option<u64>,
    /// `delivery` (2), a required string. Carried verbatim: nothing here knows
    /// what its values look like.
    pub delivery: String,
    /// `attributes.eventTrigger` (4.4) — On Click / With Build / After Build.
    pub event_trigger: Option<u64>,
    /// `attributes.animationAttributes` (4.18) — the same
    /// `KN.AnimationAttributesArchive` a transition carries, so the effect name,
    /// duration, delay, direction and automatic flag read the same way.
    pub animation: Transition,
    /// `attributes.custom_textDelivery` (4.20) and `custom_deliveryOption`
    /// (4.21) — by-word/by-bullet text builds and their order.
    pub text_delivery: Option<u64>,
    pub delivery_option: Option<u64>,
    /// `attributes.action_motionPathSource` (4.22) — an action build's path.
    pub has_motion_path: bool,
    /// `attributes` field numbers the 15.3.1 schema does not name.
    pub unknown_attributes: Vec<u32>,
}

/// One `KN.BuildChunkArchive` (153) — one stage of a build. The fixture's eight
/// builds each have a single chunk, so the shape is Confirmed but the chunk
/// timing that would carry "With Build 2, after 0.3 s" is still schema-only.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildChunk {
    pub identifier: u64,
    pub build: Option<u64>,
    pub delay: f64,
    pub duration: f64,
    pub automatic: bool,
    pub referent: bool,
}

/// `KN.RecordingArchive` (16) — a recorded presentation.
///
/// **Read and passed through, never authored** (ground rule 8). Nothing in this
/// corpus has one: Play ▸ Record Slideshow is menu-only and the dictionary has
/// no term for it, so this is identify-and-report. `KN.ShowArchive.recording`
/// (7) is where it hangs; its event tracks are `KN.RecordingEventTrackArchive`
/// (17) and its movie track a `KN.RecordingMovieTrackArchive` (18).
#[derive(Debug, Clone, PartialEq)]
pub struct Recording {
    pub identifier: u64,
    pub event_tracks: Vec<u64>,
    pub movie_track: Option<u64>,
    pub duration: f64,
}

/// One `KN.LiveVideoSource` (184) — a camera the theme knows about.
///
/// Every deck here carries exactly one, `"Default Camera"`, named as the
/// collection's `default_source` and listed in no `sources`. Ground rule 8's
/// other never-author case: a live-video feed is a device, not a document.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveVideoSource {
    pub identifier: u64,
    pub name: String,
    pub is_default: bool,
    /// Whether it is a member of the collection's `sources` (1) rather than
    /// only its `default_source` (2).
    pub listed: bool,
}

/// Which slide — by its position in the deck, or by an identifier.
///
/// A deck is an ordered thing, so position is the natural way in: `0` is the
/// first slide the show plays, skipped slides included. An identifier is the
/// `KN.SlideArchive`'s or its node's, which is what [`crate::Document::slides`]
/// prints and what the CLI takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlideRef {
    Index(usize),
    Identifier(u64),
}

/// A bare number is a **position**: `doc.slide_mut(0)` is the first slide.
///
/// An identifier is spelled — `SlideRef::Identifier(2652498)` — rather than
/// given its own conversion. Two integer conversions would make a literal
/// ambiguous, and the ambiguity would not be an error at the call site but a
/// silent choice between "the third slide" and "slide 3", which is not a
/// mistake this crate is willing to let a caller make.
impl From<usize> for SlideRef {
    fn from(index: usize) -> SlideRef {
        SlideRef::Index(index)
    }
}

impl std::fmt::Display for SlideRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlideRef::Index(index) => write!(f, "at {index}"),
            SlideRef::Identifier(id) => write!(f, "{id}"),
        }
    }
}

/// One slide of the deck.
#[derive(Debug, Clone)]
pub struct Slide {
    /// The `KN.SlideArchive`, which is also the component's identifier.
    pub identifier: u64,
    /// The `KN.SlideNodeArchive` that puts it in the deck.
    pub node: u64,
    /// `Index/Slide*.iwa`.
    pub stream: String,
    /// Position in the deck, from 0.
    pub index: usize,
    /// The number the app shows. `None` for a skipped slide, which the app
    /// answers `-1` for.
    pub number: Option<usize>,
    pub skipped: bool,
    /// `KN.SlideNodeArchive.isSlideNumberVisible` (18) — and **this is where
    /// the app's document-level `slide numbers showing` lives**. Turning it on
    /// in Keynote sets the flag on every node and leaves
    /// `KN.ShowArchive.slideNumbersVisible` (6) absent; the document property
    /// is "every slide shows its number", not a switch of its own.
    pub number_visible: bool,
    /// The layout it is built on — a `KN.SlideArchive` in a
    /// `Index/TemplateSlide-*.iwa`.
    pub layout: Option<u64>,
    pub layout_name: String,
    pub title: Option<Placeholder>,
    pub body: Option<Placeholder>,
    pub slide_number: Option<Placeholder>,
    pub object: Option<Placeholder>,
    /// The `KN.NoteArchive`, when the slide has one.
    pub note: Option<u64>,
    /// The presenter-notes storage — what an edit takes.
    pub note_storage: Option<u64>,
    pub notes: String,
    /// `owned_drawables` (7).
    pub drawables: Vec<u64>,
    /// `drawables_z_order` (42), back to front.
    pub z_order: Vec<u64>,
    pub transition: Transition,
    /// The `KN.BuildArchive`s the slide lists (2). Empty everywhere but
    /// `keynote-builds.key`, whose four animated objects carry eight between
    /// them — see [`Build`].
    pub builds: Vec<Build>,
    /// The `KN.BuildChunkArchive`s it lists (43), in order.
    pub build_chunks: Vec<BuildChunk>,
    pub style: Option<u64>,
    /// Every text storage on the slide, with its role.
    pub texts: Vec<SlideText>,
}

impl Slide {
    /// The title text, or the empty string.
    pub fn title_text(&self) -> &str {
        self.title.as_ref().map(|p| p.text.as_str()).unwrap_or("")
    }

    /// The body text, or the empty string.
    pub fn body_text(&self) -> &str {
        self.body.as_ref().map(|p| p.text.as_str()).unwrap_or("")
    }

    /// The app's `title showing`.
    pub fn title_showing(&self) -> bool {
        self.title.as_ref().is_some_and(|p| p.shown)
    }

    /// The app's `body showing`.
    pub fn body_showing(&self) -> bool {
        self.body.as_ref().is_some_and(|p| p.shown)
    }
}

/// One slide layout — what the app calls a `slide layout` and 2013 called a
/// master slide.
#[derive(Debug, Clone)]
pub struct Layout {
    pub identifier: u64,
    pub node: u64,
    pub stream: String,
    /// Position in the theme's list, from 0. The app numbers them from 1.
    pub index: usize,
    /// `KN.SlideArchive.name` (10) — "Title & Bullets", "Statement", "Blank".
    pub name: String,
    pub placeholders: Vec<Placeholder>,
    pub drawables: usize,
    /// `bodyParagraphStyles` (31): five style references, one per outline
    /// level. A **positional array** — an entry added to it shifts the mapping
    /// from level to style rather than listing another style.
    pub body_paragraph_styles: Vec<u64>,
    pub body_list_styles: Vec<u64>,
    pub transition: Transition,
}

/// `KN.Soundtrack` (21) — the deck's own audio, the app's Document ▸ Audio.
///
/// One per deck, in `Index/Document.iwa`, in every deck this crate has seen —
/// **and empty in all of them**: volume 1, mode "play once", no tracks, eleven
/// bytes on the wire. Nothing can fill one in. There is no soundtrack term
/// anywhere in Keynote's sdef; `audio clip` is an element of a *slide*, and
/// `make new audio clip` is accepted and then does nothing at all — the app
/// answers no error, the slide's `audio clips` still count zero, and the saved
/// package holds no new `Data/` entry.
///
/// So the *shape* of a filled one is decoded and unexercised: `movie_media` (3)
/// is a repeated `TSP.DataReference`, each a bare `{1: data identifier}` into
/// the same registry `iwork media` lists, **in play order**.
#[derive(Debug, Clone)]
pub struct Soundtrack {
    pub identifier: u64,
    pub volume: f64,
    /// 0 play once, 1 loop, 2 do not play.
    pub mode: u64,
    /// The track list, in order: media identifiers into the document's
    /// `TSP.DataInfo` table.
    pub media: Vec<u64>,
}

impl Soundtrack {
    pub fn mode_name(&self) -> &'static str {
        match self.mode {
            0 => "play once",
            1 => "loop",
            2 => "do not play",
            _ => "unknown",
        }
    }

    /// How many tracks it names.
    pub fn tracks(&self) -> usize {
        self.media.len()
    }
}

/// Everything this module reads about one Keynote document.
#[derive(Debug, Clone)]
pub struct Show {
    pub identifier: u64,
    pub theme: Option<u64>,
    /// The theme's *stored* name — `"21_BasicWhite"`. The app shows a
    /// localised display name ("Basic White") that is nowhere in the document.
    pub theme_name: String,
    pub theme_uuid: String,
    /// Slide size in points. Keynote stores 16:9 as 1920 × 1080, which is the
    /// number the app's `width` and `height` report.
    pub width: f32,
    pub height: f32,
    /// `KN.ShowArchive.slideNumbersVisible` (6). **Absent from every deck in
    /// this corpus, including the one whose numbers are on**: the app writes
    /// the per-slide flag instead. See [`Slide::number_visible`] and
    /// [`Show::numbers_shown_on`].
    pub slide_numbers_visible: bool,
    /// `loop_presentation` (8) — the app's `auto loop`. **Measured**: true in
    /// `keynote-playback` and false in every other deck, which is the only
    /// thing about it that differs from them.
    pub loop_presentation: bool,
    /// `mode` (9): 0 normal, 1 auto-play (self-playing), 2 hyperlinks only.
    /// Written explicitly, at 0, by every deck here. **Schema-only** — the
    /// presentation type has no scripting term, so nothing can move it.
    pub mode: u64,
    /// `autoplay_transition_delay` (10), default 5 s, and
    /// `autoplay_build_delay` (11), default 2 s: the two waits a self-playing
    /// show uses. Written explicitly at their defaults by every deck here and
    /// **schema-only** for the same reason `mode` is.
    pub autoplay_transition_delay: f64,
    pub autoplay_build_delay: f64,
    /// `idle_timer_active` (15) — the app's `auto restart`. **Measured.**
    pub idle_timer_active: bool,
    /// `idle_timer_delay` (16) — **in seconds**, default 900. The dictionary's
    /// `maximum idle duration` is in *minutes*: setting it to 137 wrote 8220.
    pub idle_timer_delay: f64,
    /// `automatically_plays_upon_open` (18) — the app's `auto play`, despite
    /// the name; the *self-playing* mode is field 9. **Measured.**
    pub plays_on_open: bool,
    pub stylesheet: Option<u64>,
    pub soundtrack: Option<Soundtrack>,
    /// A `KN.RecordingArchive`, when the deck carries a recorded presentation.
    /// Read and passed through; never authored.
    pub recording: Option<Recording>,
    /// The theme's live-video sources — one `"Default Camera"` in every deck
    /// here. Identify-and-report, like the recording.
    pub live_video_sources: Vec<LiveVideoSource>,
    pub slides: Vec<Slide>,
    pub layouts: Vec<Layout>,
}

impl Show {
    pub fn mode_name(&self) -> &'static str {
        match self.mode {
            0 => "normal",
            1 => "auto-play",
            2 => "hyperlinks only",
            _ => "unknown",
        }
    }

    /// How many slides show their number — the app's document-level `slide
    /// numbers showing`, which is true when this is every slide.
    pub fn numbers_shown_on(&self) -> usize {
        self.slides.iter().filter(|s| s.number_visible).count()
    }

    /// The slide with this identifier, whichever way it was named — by the
    /// slide archive or by its node.
    pub fn slide(&self, identifier: u64) -> Option<&Slide> {
        self.slides
            .iter()
            .find(|s| s.identifier == identifier || s.node == identifier)
    }

    /// `maximum idle duration` as the app reports it — **minutes**, where
    /// [`Show::idle_timer_delay`] is the seconds on the wire.
    pub fn idle_minutes(&self) -> f64 {
        self.idle_timer_delay / 60.0
    }

    /// How many builds the whole deck carries. Zero in every deck but
    /// `keynote-builds.key`, which has eight — the fixture [`Build`] was
    /// measured against.
    pub fn build_count(&self) -> usize {
        self.slides.iter().map(|s| s.builds.len()).sum()
    }

    /// The layout a slide is built on.
    pub fn layout_of(&self, slide: &Slide) -> Option<&Layout> {
        slide
            .layout
            .and_then(|id| self.layouts.iter().find(|l| l.identifier == id))
    }
}

// -- reading -----------------------------------------------------------------

fn float(message: &Message, number: u32) -> f32 {
    match message.get(number) {
        Some(Value::Fixed32(bytes)) => f32::from_le_bytes(*bytes),
        Some(Value::Fixed64(bytes)) => f64::from_le_bytes(*bytes) as f32,
        _ => 0.0,
    }
}

fn double(message: &Message, number: u32, default: f64) -> f64 {
    match message.get(number) {
        Some(Value::Fixed64(bytes)) => f64::from_le_bytes(*bytes),
        Some(Value::Fixed32(bytes)) => f32::from_le_bytes(*bytes) as f64,
        _ => default,
    }
}

fn flag(message: &Message, number: u32, default: bool) -> bool {
    message.varint(number).map(|v| v != 0).unwrap_or(default)
}

fn text(message: &Message, number: u32) -> String {
    message
        .bytes(number)
        .map(|raw| String::from_utf8_lossy(raw).into_owned())
        .unwrap_or_default()
}

fn reference(message: &Message, number: u32) -> Option<u64> {
    decode_nested(message.bytes(number)?).and_then(|r| reference_target(&r))
}

fn references(message: &Message, number: u32) -> Vec<u64> {
    message
        .all(number)
        .filter_map(|value| match value {
            Value::Bytes(raw) => decode_nested(raw).and_then(|r| reference_target(&r)),
            _ => None,
        })
        .collect()
}

fn float32(message: &Message, number: u32) -> Option<f32> {
    match message.get(number)? {
        Value::Fixed32(bytes) => Some(f32::from_le_bytes(*bytes)),
        Value::Fixed64(bytes) => Some(f64::from_le_bytes(*bytes) as f32),
        _ => None,
    }
}

/// The transition a slide or a layout carries, parameters and all.
pub fn transition(slide: &Message) -> Transition {
    let attributes = slide
        .bytes(slide_field::TRANSITION)
        .and_then(decode_nested)
        .and_then(|t| {
            t.bytes(transition_field::ATTRIBUTES)
                .and_then(decode_nested)
        });
    let Some(attributes) = attributes else {
        return Transition::default();
    };
    transition_attributes(&attributes)
}

/// Decode a `KN.TransitionAttributesArchive` — field 8 and the `custom_*` block
/// beside it.
///
/// The same message sits at `KN.SlideStylePropertiesArchive.transition` (2),
/// which is how a *theme* gives a layout a default transition, so this is the
/// one place that knows the layout.
pub fn transition_attributes(attributes: &Message) -> Transition {
    // The animation attributes are what a transition *is*; the parameters
    // beside them are read whether or not there are any, because a tripwire
    // that only fires on well-formed input is not a tripwire.
    let animation = attributes
        .bytes(transition_field::ANIMATION_ATTRIBUTES)
        .and_then(decode_nested)
        .unwrap_or_default();
    let mut unknown_parameters: Vec<u32> = attributes
        .fields
        .iter()
        .map(|field| field.number)
        .filter(|number| !transition_field::NAMED.contains(number))
        .collect();
    unknown_parameters.sort_unstable();
    unknown_parameters.dedup();
    let effect = text(&animation, transition_field::EFFECT);
    Transition {
        // An attributes message with no effect at all is the same "no
        // transition" the app writes as `"none"`, and saying so keeps
        // [`effect_name`] answering for every transition a deck can hold.
        effect: if effect.is_empty() {
            "none".into()
        } else {
            effect
        },
        animation_type: text(&animation, transition_field::ANIMATION_TYPE),
        duration: double(&animation, transition_field::DURATION, 0.0),
        delay: double(&animation, transition_field::DELAY, 0.0),
        automatic: flag(&animation, transition_field::AUTOMATIC, false),
        direction: animation.varint(transition_field::DIRECTION).unwrap_or(0),
        seed: animation.varint(transition_field::SEED).unwrap_or(0),
        color: animation
            .bytes(transition_field::COLOR)
            .and_then(decode_nested)
            .as_ref()
            .and_then(crate::drawable::Color::decode),
        rtl: flag(&animation, transition_field::RTL, false),
        parameters: TransitionParameters {
            twist: float32(attributes, transition_field::CUSTOM_TWIST),
            mosaic_size: attributes.varint(transition_field::CUSTOM_MOSAIC_SIZE),
            mosaic_type: attributes.varint(transition_field::CUSTOM_MOSAIC_TYPE),
            bounce: attributes
                .varint(transition_field::CUSTOM_BOUNCE)
                .map(|v| v != 0),
            magic_move_fade_unmatched: attributes
                .varint(transition_field::CUSTOM_MAGIC_MOVE_FADE_UNMATCHED)
                .map(|v| v != 0),
            timing_curve: attributes
                .varint(transition_field::CUSTOM_TIMING_CURVE)
                .map(TimingCurve::of),
            text_delivery: attributes
                .varint(transition_field::CUSTOM_TEXT_DELIVERY)
                .map(TextDelivery::of),
            motion_blur: attributes
                .varint(transition_field::CUSTOM_MOTION_BLUR)
                .map(|v| v != 0),
            travel_distance: float32(attributes, transition_field::CUSTOM_TRAVEL_DISTANCE),
            angle: float32(attributes, transition_field::CUSTOM_ANGLE),
            blur_amount: float32(attributes, transition_field::CUSTOM_BLUR_AMOUNT),
        },
        unknown_parameters,
    }
}

/// Every field number of `KN.BuildAttributesArchive` the 15.3.1 schema names.
const BUILD_ATTRIBUTE_FIELDS: [u32; 34] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 33, 34, 35, 36,
];

/// One `KN.BuildArchive`. Schema-only — see [`Build`].
fn build(identifier: u64, archive: &Message) -> Build {
    let attributes = archive
        .bytes(build_field::ATTRIBUTES)
        .and_then(decode_nested)
        .unwrap_or_default();
    let mut unknown_attributes: Vec<u32> = attributes
        .fields
        .iter()
        .map(|field| field.number)
        .filter(|number| !BUILD_ATTRIBUTE_FIELDS.contains(number) && !(37..=41).contains(number))
        .collect();
    unknown_attributes.sort_unstable();
    unknown_attributes.dedup();
    let animation = attributes
        .bytes(build_field::ANIMATION_ATTRIBUTES)
        .and_then(decode_nested);
    Build {
        identifier,
        drawable: reference(archive, build_field::DRAWABLE),
        delivery: text(archive, build_field::DELIVERY),
        event_trigger: attributes.varint(build_field::EVENT_TRIGGER),
        // A build's animation attributes are a transition's, so they are read
        // by the same code and the `custom_*` block simply stays empty.
        animation: match animation {
            Some(animation) => {
                let mut wrapper = Message::default();
                wrapper.set(
                    transition_field::ANIMATION_ATTRIBUTES,
                    Value::Bytes(animation.encode()),
                );
                transition_attributes(&wrapper)
            }
            None => Transition::default(),
        },
        text_delivery: attributes.varint(build_field::CUSTOM_TEXT_DELIVERY),
        delivery_option: attributes.varint(build_field::CUSTOM_DELIVERY_OPTION),
        has_motion_path: attributes.get(build_field::ACTION_MOTION_PATH).is_some(),
        unknown_attributes,
    }
}

/// One `KN.BuildChunkArchive`. Schema-only — see [`Build`].
fn build_chunk(identifier: u64, archive: &Message) -> BuildChunk {
    BuildChunk {
        identifier,
        build: reference(archive, build_field::CHUNK_BUILD),
        delay: double(archive, build_field::CHUNK_DELAY, 0.0),
        duration: double(archive, build_field::CHUNK_DURATION, 0.0),
        automatic: flag(archive, build_field::CHUNK_AUTOMATIC, false),
        referent: flag(archive, build_field::CHUNK_REFERENT, false),
    }
}

/// The storage a `TSWP.ShapeInfoArchive` — or anything nesting one — lays out.
///
/// A placeholder is a `KN.PlaceholderArchive` whose field 1 is a shape info,
/// whose field 2 is the storage; a plain text item is the shape info itself.
/// One recursive look for field 2 covers both without knowing which it has.
///
/// **The target has to be a storage.** Field 2 of a `TSD.ImageArchive` is not
/// one, and a walk that trusts the field number alone reports an image slide as
/// carrying a text box whose "storage" is the slide itself — which is exactly
/// what it did before this check was added.
fn shape_storage(
    archives: &BTreeMap<u64, (u32, Message)>,
    archive: &Message,
    depth: usize,
) -> Option<u64> {
    if depth == 0 {
        return None;
    }
    if let Some(storage) = reference(archive, 2) {
        if archives
            .get(&storage)
            .is_some_and(|(message_type, _)| *message_type == crate::TYPE_STORAGE)
        {
            return Some(storage);
        }
    }
    let nested = archive.bytes(1).and_then(decode_nested)?;
    shape_storage(archives, &nested, depth - 1)
}

fn placeholder(
    archives: &BTreeMap<u64, (u32, Message)>,
    document: &crate::Document,
    identifier: u64,
    owned: &BTreeSet<u64>,
) -> Option<Placeholder> {
    let (message_type, archive) = archives.get(&identifier)?;
    let kind = if *message_type == TYPE_PLACEHOLDER {
        PlaceholderKind::of(archive.varint(placeholder_field::KIND).unwrap_or(0))
    } else {
        PlaceholderKind::Generic
    };
    let storage = shape_storage(archives, archive, 4);
    Some(Placeholder {
        identifier,
        kind,
        storage,
        text: storage
            .and_then(|id| document.storage_text(id).ok())
            .unwrap_or_default(),
        shown: owned.contains(&identifier),
    })
}

/// Read the whole `KN` structure of a document.
///
/// Returns `None` for anything that is not a Keynote document: a Pages file
/// handed here has no `KN.ShowArchive` and says so rather than guessing.
pub fn show(document: &crate::Document) -> Option<Show> {
    if document.kind() != crate::Kind::Keynote {
        return None;
    }
    let mut archives: BTreeMap<u64, (u32, Message)> = BTreeMap::new();
    let mut stream_of: BTreeMap<u64, String> = BTreeMap::new();
    for (stream, object) in document.objects() {
        if let Ok(message) = Message::decode(object.payload()) {
            archives.insert(object.identifier, (object.message_type(), message));
            stream_of.insert(object.identifier, stream.to_string());
        }
    }

    let (identifier, show_archive) = archives
        .iter()
        .find(|(_, (message_type, _))| *message_type == TYPE_SHOW)
        .map(|(id, (_, m))| (*id, m.clone()))?;

    // -- the theme, and the layouts it lists ---------------------------------
    let theme = reference(&show_archive, show_field::THEME);
    let theme_archive = theme
        .and_then(|id| archives.get(&id))
        .map(|(_, m)| m.clone());
    let theme_super = theme_archive
        .as_ref()
        .and_then(|t| t.bytes(theme_field::SUPER).and_then(decode_nested));
    let layout_nodes = theme_archive
        .as_ref()
        .map(|t| references(t, theme_field::TEMPLATES))
        .unwrap_or_default();

    let mut layouts = Vec::new();
    for (index, node) in layout_nodes.iter().enumerate() {
        let Some((_, node_archive)) = archives.get(node) else {
            continue;
        };
        let Some(identifier) = reference(node_archive, node_field::SLIDE) else {
            continue;
        };
        let Some((_, archive)) = archives.get(&identifier) else {
            continue;
        };
        let owned: BTreeSet<u64> = references(archive, slide_field::OWNED_DRAWABLES)
            .into_iter()
            .collect();
        let mut placeholders = Vec::new();
        for field in [
            slide_field::TITLE_PLACEHOLDER,
            slide_field::BODY_PLACEHOLDER,
            slide_field::SLIDE_NUMBER_PLACEHOLDER,
            slide_field::OBJECT_PLACEHOLDER,
        ] {
            if let Some(id) = reference(archive, field) {
                if let Some(p) = placeholder(&archives, document, id, &owned) {
                    placeholders.push(p);
                }
            }
        }
        layouts.push(Layout {
            identifier,
            node: *node,
            stream: stream_of.get(&identifier).cloned().unwrap_or_default(),
            index,
            name: text(archive, slide_field::NAME),
            placeholders,
            drawables: owned.len(),
            body_paragraph_styles: references(archive, slide_field::BODY_PARAGRAPH_STYLES),
            body_list_styles: references(archive, slide_field::BODY_LIST_STYLES),
            transition: transition(archive),
        });
    }
    let layout_name: BTreeMap<u64, String> = layouts
        .iter()
        .map(|l| (l.identifier, l.name.clone()))
        .collect();

    // -- the deck ------------------------------------------------------------
    let slide_nodes = show_archive
        .bytes(show_field::SLIDE_TREE)
        .and_then(decode_nested)
        .map(|tree| references(&tree, slide_tree_field::SLIDES))
        .unwrap_or_default();

    let mut slides = Vec::new();
    let mut number = 0usize;
    for (index, node) in slide_nodes.iter().enumerate() {
        let Some((_, node_archive)) = archives.get(node) else {
            continue;
        };
        let Some(identifier) = reference(node_archive, node_field::SLIDE) else {
            continue;
        };
        let Some((_, archive)) = archives.get(&identifier) else {
            continue;
        };
        let skipped = flag(node_archive, node_field::SKIPPED, false);
        let number_visible = flag(node_archive, node_field::SLIDE_NUMBER_VISIBLE, false);
        // The app numbers the slides it will show, and answers -1 for the rest.
        let visible = if skipped {
            None
        } else {
            number += 1;
            Some(number)
        };

        let drawables = references(archive, slide_field::OWNED_DRAWABLES);
        let owned: BTreeSet<u64> = drawables.iter().copied().collect();
        let named = |field: u32| {
            reference(archive, field).and_then(|id| placeholder(&archives, document, id, &owned))
        };
        let title = named(slide_field::TITLE_PLACEHOLDER);
        let body = named(slide_field::BODY_PLACEHOLDER);
        let slide_number = named(slide_field::SLIDE_NUMBER_PLACEHOLDER);
        let object = named(slide_field::OBJECT_PLACEHOLDER);

        let note = reference(archive, slide_field::NOTE);
        let note_storage = note
            .and_then(|id| archives.get(&id))
            .and_then(|(_, n)| reference(n, 1));

        // Every storage on the slide, named by what points at it.
        let mut texts = Vec::new();
        let mut claimed: BTreeMap<u64, Role> = BTreeMap::new();
        for (place, role) in [
            (&title, Role::Title),
            (&body, Role::Body),
            (&slide_number, Role::SlideNumber),
            (&object, Role::ObjectPlaceholder),
        ] {
            if let Some(p) = place {
                if let Some(storage) = p.storage {
                    claimed.insert(storage, role);
                    texts.push(SlideText {
                        storage,
                        role,
                        drawable: Some(p.identifier),
                        text: p.text.clone(),
                    });
                }
            }
        }
        if let Some(storage) = note_storage {
            claimed.insert(storage, Role::Notes);
            texts.push(SlideText {
                storage,
                role: Role::Notes,
                drawable: None,
                text: document.storage_text(storage).unwrap_or_default(),
            });
        }
        for drawable in &drawables {
            let Some((message_type, info)) = archives.get(drawable) else {
                continue;
            };
            let Some(storage) = shape_storage(&archives, info, 4) else {
                continue;
            };
            if claimed.contains_key(&storage) {
                continue;
            }
            let role = if *message_type == TYPE_PLACEHOLDER {
                Role::Placeholder
            } else {
                Role::TextBox
            };
            claimed.insert(storage, role);
            texts.push(SlideText {
                storage,
                role,
                drawable: Some(*drawable),
                text: document.storage_text(storage).unwrap_or_default(),
            });
        }

        slides.push(Slide {
            identifier,
            node: *node,
            stream: stream_of.get(&identifier).cloned().unwrap_or_default(),
            index,
            number: visible,
            skipped,
            number_visible,
            layout: reference(archive, slide_field::TEMPLATE_SLIDE),
            layout_name: reference(archive, slide_field::TEMPLATE_SLIDE)
                .and_then(|id| layout_name.get(&id).cloned())
                .unwrap_or_default(),
            notes: note_storage
                .and_then(|id| document.storage_text(id).ok())
                .unwrap_or_default(),
            title,
            body,
            slide_number,
            object,
            note,
            note_storage,
            drawables,
            z_order: references(archive, slide_field::DRAWABLES_Z_ORDER),
            transition: transition(archive),
            builds: references(archive, slide_field::BUILDS)
                .into_iter()
                .filter_map(|id| archives.get(&id).map(|(_, m)| build(id, m)))
                .collect(),
            build_chunks: references(archive, slide_field::BUILD_CHUNKS)
                .into_iter()
                .filter_map(|id| archives.get(&id).map(|(_, m)| build_chunk(id, m)))
                .collect(),
            style: reference(archive, slide_field::STYLE),
            texts,
        });
    }

    let size = show_archive
        .bytes(show_field::SIZE)
        .and_then(decode_nested)
        .unwrap_or_default();

    let soundtrack = reference(&show_archive, show_field::SOUNDTRACK)
        .and_then(|id| archives.get(&id).map(|(_, m)| (id, m)))
        .map(|(identifier, m)| Soundtrack {
            identifier,
            volume: double(m, soundtrack_field::VOLUME, 0.0),
            mode: m.varint(soundtrack_field::MODE).unwrap_or(0),
            media: references(m, soundtrack_field::MOVIE_MEDIA),
        });

    let recording = reference(&show_archive, show_field::RECORDING)
        .and_then(|id| archives.get(&id).map(|(_, m)| (id, m)))
        .map(|(identifier, m)| Recording {
            identifier,
            event_tracks: references(m, recording_field::EVENT_TRACKS),
            movie_track: reference(m, recording_field::MOVIE_TRACK),
            duration: double(m, recording_field::DURATION, 0.0),
        });

    // The theme names one collection; the collection names its sources and,
    // separately, its default — which in every deck here is the only one there
    // is, listed nowhere else.
    let mut live_video_sources = Vec::new();
    if let Some(collection) = theme_archive
        .as_ref()
        .and_then(|t| reference(t, theme_field::LIVE_VIDEO_SOURCES))
        .and_then(|id| archives.get(&id))
        .map(|(_, m)| m)
    {
        let listed = references(collection, live_video_field::COLLECTION_SOURCES);
        let mut wanted = listed.clone();
        if let Some(default) = reference(collection, live_video_field::COLLECTION_DEFAULT) {
            if !wanted.contains(&default) {
                wanted.push(default);
            }
        }
        for identifier in wanted {
            let Some((_, source)) = archives.get(&identifier) else {
                continue;
            };
            live_video_sources.push(LiveVideoSource {
                identifier,
                name: text(source, live_video_field::NAME),
                is_default: flag(source, live_video_field::IS_DEFAULT, false),
                listed: listed.contains(&identifier),
            });
        }
    }

    Some(Show {
        identifier,
        theme,
        theme_name: theme_super
            .as_ref()
            .map(|s| text(s, theme_field::SUPER_NAME))
            .unwrap_or_default(),
        theme_uuid: theme_archive
            .as_ref()
            .map(|t| text(t, theme_field::UUID))
            .unwrap_or_default(),
        width: float(&size, 1),
        height: float(&size, 2),
        slide_numbers_visible: flag(&show_archive, show_field::SLIDE_NUMBERS_VISIBLE, false),
        loop_presentation: flag(&show_archive, show_field::LOOP, false),
        mode: show_archive.varint(show_field::MODE).unwrap_or(0),
        autoplay_transition_delay: double(
            &show_archive,
            show_field::AUTOPLAY_TRANSITION_DELAY,
            5.0,
        ),
        autoplay_build_delay: double(&show_archive, show_field::AUTOPLAY_BUILD_DELAY, 2.0),
        idle_timer_active: flag(&show_archive, show_field::IDLE_TIMER_ACTIVE, false),
        idle_timer_delay: double(&show_archive, show_field::IDLE_TIMER_DELAY, 900.0),
        plays_on_open: flag(&show_archive, show_field::PLAYS_ON_OPEN, false),
        stylesheet: reference(&show_archive, show_field::STYLESHEET),
        soundtrack,
        recording,
        live_video_sources,
        slides,
        layouts,
    })
}

// -- writing -----------------------------------------------------------------

/// The node of a slide, whichever way the slide was named.
fn node_of(document: &crate::Document, slide: u64) -> Result<(u64, Message), Error> {
    let mut by_slide = None;
    for (_, object) in document.objects() {
        if object.message_type() != TYPE_SLIDE_NODE {
            continue;
        }
        let Ok(archive) = Message::decode(object.payload()) else {
            continue;
        };
        if object.identifier == slide {
            return Ok((object.identifier, archive));
        }
        if reference(&archive, node_field::SLIDE) == Some(slide) {
            by_slide = Some((object.identifier, archive));
        }
    }
    by_slide.ok_or(Error::NoSuchObject(slide))
}

/// The node order held in a `KN.SlideTreeArchive`, **refusing the read if any
/// entry is not a bare `{1: node}` reference**.
///
/// `move_slide` and `insert_into_slide_tree` rebuild the repeated `slides`
/// field from these ids, re-emitting each as a fresh `{1: id}` message — which
/// reproduces a bare reference byte for byte but would silently drop an entry
/// that carried anything else, and its slide with it. Every entry Keynote
/// writes is a bare reference, so refusing the odd one costs nothing and keeps
/// the rebuild honest.
fn slide_tree_order(tree: &Message) -> Result<Vec<u64>, Error> {
    let mut order = Vec::new();
    for value in tree.all(slide_tree_field::SLIDES) {
        let Value::Bytes(raw) = value else {
            return Err(Error::Format(
                "KN.SlideTreeArchive has a slide entry that is not a message; \
                 refusing to rewrite a tree this crate would not reproduce"
                    .into(),
            ));
        };
        let target = decode_nested(raw)
            .as_ref()
            .and_then(reference_target)
            .ok_or_else(|| {
                Error::Format(
                    "KN.SlideTreeArchive has a slide entry that is not a bare reference; \
                     refusing to rewrite a tree this crate would not reproduce"
                        .into(),
                )
            })?;
        order.push(target);
    }
    Ok(order)
}

/// The show, its slide tree, and the node order — the reading every reorder
/// starts from. Errors if the document is not a deck, has no tree, or holds a
/// tree entry `slide_tree_order` will not reproduce.
fn read_slide_tree(document: &crate::Document) -> Result<(u64, Message, Message, Vec<u64>), Error> {
    let (show_id, show_archive) = document
        .objects()
        .find(|(_, o)| o.message_type() == TYPE_SHOW)
        .map(|(_, o)| (o.identifier, Message::decode(o.payload())))
        .ok_or_else(|| Error::Format("no KN.ShowArchive: not a Keynote document".into()))?;
    let show_archive = show_archive.map_err(|e| Error::Format(format!("KN.ShowArchive: {e}")))?;
    let tree_raw = show_archive
        .bytes(show_field::SLIDE_TREE)
        .ok_or_else(|| Error::Format("KN.ShowArchive has no slide tree".into()))?;
    let tree = decode_nested(tree_raw)
        .ok_or_else(|| Error::Format("KN.ShowArchive slide tree does not decode".into()))?;
    let order = slide_tree_order(&tree)?;
    Ok((show_id, show_archive, tree, order))
}

/// Write a new node order back into the show, keeping every other field of the
/// `KN.SlideTreeArchive` verbatim. The entries are re-emitted as bare
/// references, which reproduces what `slide_tree_order` accepted byte for
/// byte, so an order that ends where it began reproduces the original stream.
fn write_slide_order(
    document: &mut crate::Document,
    show_id: u64,
    mut show_archive: Message,
    tree: &Message,
    order: &[u64],
) -> Result<(), Error> {
    let mut rebuilt = Message::default();
    for field in &tree.fields {
        if field.number != slide_tree_field::SLIDES {
            rebuilt.fields.push(field.clone());
        }
    }
    for id in order {
        let mut entry = Message::default();
        entry.set(1, Value::Varint(*id));
        rebuilt.append_in_order(slide_tree_field::SLIDES, Value::Bytes(entry.encode()));
    }
    show_archive.set(show_field::SLIDE_TREE, Value::Bytes(rebuilt.encode()));
    document.set_archive_of(show_id, &show_archive)
}

/// The node of a slide that is actually **in the show's deck** — refusing a
/// slide layout, whose node lives in the theme's `templates`, not the show's
/// slide tree.
///
/// [`node_of`] resolves a layout id perfectly well, because a layout has a
/// `KN.SlideNodeArchive` too; membership of the slide tree is the only thing
/// that tells a slide from a template. Without this guard, `set_slide_skipped`
/// writes `isSkipped` into a theme node and the CLI reports "slide N is
/// skipped" about a layout, silently.
fn slide_node_in_deck(document: &crate::Document, slide: u64) -> Result<(u64, Message), Error> {
    let (node, archive) = node_of(document, slide)?;
    let (_, _, _, order) = read_slide_tree(document)?;
    if !order.contains(&node) {
        return Err(Error::Format(format!(
            "object {slide} is not a slide in the show's deck — a slide layout, or a node \
             the slide tree does not list; refusing to treat it as a slide"
        )));
    }
    Ok((node, archive))
}

/// Skip a slide, or stop skipping it.
///
/// The whole of the feature is `KN.SlideNodeArchive.isSkipped` (4), which the
/// app's `skipped` property reads and writes. A skipped slide keeps its
/// number's *place* in nothing: the app renumbers the deck around it and
/// answers `slide number` with -1.
///
/// Refuses an id that is not a slide in the show's deck — a slide layout has a
/// node too, and writing `isSkipped` into a theme template is a silent
/// mis-edit. See `slide_node_in_deck`.
///
/// Returns whether the flag changed.
pub fn set_slide_skipped(
    document: &mut crate::Document,
    slide: u64,
    skipped: bool,
) -> Result<bool, Error> {
    let (node, mut archive) = slide_node_in_deck(document, slide)?;
    let was = flag(&archive, node_field::SKIPPED, false);
    if was == skipped {
        return Ok(false);
    }
    // `isSkipped` is a *required* field of the message and every node in the
    // corpus writes it, false included — so this sets it rather than removing
    // it when it goes back to false.
    archive.set(node_field::SKIPPED, Value::Varint(u64::from(skipped)));
    document.set_archive_of(node, &archive)?;
    Ok(true)
}

/// Move a slide to another position in the deck.
///
/// `to` is the index the slide ends up at, counting the deck **after** the
/// slide has been taken out — which is what `move slide n to after slide m`
/// means to the app. Reordering is exactly a permutation of
/// `KN.ShowArchive.slideTree.slides`: Keynote's own `move` was watched doing
/// that and touching nothing else, every node and every slide component byte
/// for byte where it was.
///
/// Refuses a slide the deck does not list (a slide layout among them) and an
/// out-of-range target — a clamp would report "now at position 5" for a
/// `move … 99` on a six-slide deck, which is a lie about where the slide went.
/// The permutation only ever re-emits entries `slide_tree_order` accepted as
/// bare references, so it is byte-for-byte lossless or it is refused.
pub fn move_slide(document: &mut crate::Document, slide: u64, to: usize) -> Result<usize, Error> {
    let (node, _) = node_of(document, slide)?;
    let (show_id, show_archive, tree, mut order) = read_slide_tree(document)?;
    let from = order.iter().position(|id| *id == node).ok_or_else(|| {
        Error::Format(format!(
            "object {slide} is not a slide in the show's deck — a slide layout, perhaps; \
             refusing to move it"
        ))
    })?;
    if to >= order.len() {
        return Err(Error::Format(format!(
            "target position {to} is past the end of a {}-slide deck; \
             a move must land on an existing position (0..{})",
            order.len(),
            order.len() - 1
        )));
    }
    if from == to {
        return Ok(from);
    }
    let moved = order.remove(from);
    order.insert(to, moved);
    write_slide_order(document, show_id, show_archive, &tree, &order)?;
    Ok(to)
}

/// What [`duplicate_slide`] made.
#[derive(Debug, Clone)]
pub struct SlideCopy {
    /// The new `KN.SlideArchive`, which is also the new component's identifier.
    pub identifier: u64,
    /// The new `KN.SlideNodeArchive`.
    pub node: u64,
    /// The slide it was copied from.
    pub source: u64,
    /// The new `Index/Slide-<id>.iwa`.
    pub stream: String,
    /// Position in the deck, from 0.
    pub index: usize,
    /// How many objects were copied.
    pub objects: usize,
    /// How many `Data/` entries the copy now shares with the original.
    pub media: usize,
    /// Declarations added by [`crate::Document::declare_external_references`].
    pub declarations: usize,
}

/// Copy a slide, and put the copy straight after it.
///
/// **Copy, don't synthesise, taken all the way.** Keynote's own `duplicate` was
/// watched first, by saving before and after: it writes a new
/// `Index/Slide-<id>.iwa` holding the same objects as the source stream, in the
/// same order, at the same sizes, with every reference *inside* the stream
/// remapped and every reference *out* of it — the layout, the stylesheet, the
/// slide style — left pointing where it did; a new `KN.SlideNodeArchive` beside
/// the original in `Index/Document.iwa`, identical to it but for the slide it
/// names and `thumbnailsAreDirty`; a new `TSP.ComponentInfo` carrying the same
/// external references, the same data references with the *user* object
/// remapped, and fresh object UUIDs; and one more entry in the show's slide
/// tree. This does that, and nothing else.
///
/// What it deliberately does not do: invent a thumbnail (the node's
/// `thumbnails` are copied and marked dirty, which is what the app itself
/// leaves behind), or copy the `Data/` bytes (a duplicated image slide shares
/// the original's media — the app shares it too, and the registry is
/// refcounted per component, not per package).
pub fn duplicate_slide(document: &mut crate::Document, slide: u64) -> Result<SlideCopy, Error> {
    // -- find the slide, its node, its stream -------------------------------
    let (stream, source_id) = {
        let (stream, object) = document.object(slide).ok_or(Error::NoSuchObject(slide))?;
        if object.message_type() != TYPE_SLIDE {
            // Naming the node is the friendlier call and costs one lookup.
            let (node, node_archive) = node_of(document, slide)?;
            let target = reference(&node_archive, node_field::SLIDE)
                .ok_or_else(|| Error::Format(format!("slide node {node} names no slide")))?;
            let (stream, _) = document.object(target).ok_or(Error::NoSuchObject(target))?;
            (stream.to_string(), target)
        } else {
            (stream.to_string(), slide)
        }
    };
    // What tells a slide from a layout is deck membership, not the stream name:
    // a layout has a node and a `TemplateSlide` stream, but its node is not in
    // the show's slide tree. Keynote's own dictionary will not copy a layout
    // either.
    let (node_id, node_archive) = slide_node_in_deck(document, source_id)?;

    // -- allocate identifiers for every object in the stream ----------------
    let source_objects: Vec<u64> = document
        .objects()
        .filter(|(name, _)| *name == stream)
        .map(|(_, object)| object.identifier)
        .collect();
    if source_objects.is_empty() {
        return Err(Error::Format(format!("slide {source_id} has no stream")));
    }
    let mut next = document.next_object_identifier();
    let mut map: BTreeMap<u64, u64> = BTreeMap::new();
    // The slide archive first, so the copy's component identifier is its slide
    // archive's — the convention every component in a deck follows.
    map.insert(source_id, next);
    next += 1;
    for id in &source_objects {
        if *id == source_id {
            continue;
        }
        map.insert(*id, next);
        next += 1;
    }
    let new_slide = map[&source_id];
    let new_node = next;
    next += 1;
    document.set_last_object_identifier(next - 1)?;

    // -- the stream ----------------------------------------------------------
    let mut copied = document.stream_objects(&stream);
    for object in &mut copied {
        object.identifier = map[&object.identifier];
        for message in &mut object.messages {
            if let Ok(mut archive) = Message::decode(&message.payload) {
                if remap_references(&mut archive, &map, MAX_DEPTH) {
                    message.payload = archive.encode();
                }
            }
        }
    }
    let new_stream = format!("Index/Slide-{new_slide}.iwa");
    let objects = copied.len();
    document.add_stream(&new_stream, copied)?;

    // -- the node ------------------------------------------------------------
    let mut node = node_archive.clone();
    let mut slide_ref = Message::default();
    slide_ref.set(1, Value::Varint(new_slide));
    node.set(node_field::SLIDE, Value::Bytes(slide_ref.encode()));
    // The copy's thumbnails are the original's until the app redraws them,
    // which is exactly what Keynote's own duplicate leaves behind.
    node.set(node_field::THUMBNAILS_DIRTY, Value::Varint(1));
    document.add_object_after(node_id, new_node, TYPE_SLIDE_NODE, &node)?;

    // -- the slide tree ------------------------------------------------------
    let index = insert_into_slide_tree(document, node_id, new_node)?;

    // -- the component -------------------------------------------------------
    let media = clone_component(document, source_id, new_slide, &new_stream, &map)?;
    let declarations = document.declare_external_references();

    Ok(SlideCopy {
        identifier: new_slide,
        node: new_node,
        source: source_id,
        stream: new_stream,
        index,
        objects,
        media,
        declarations,
    })
}

pub(crate) const MAX_DEPTH: usize = 24;

/// Rewrite every `TSP.Reference` whose target is in `map`, at any depth.
///
/// The detector is [`crate::style::reference_target`]'s — a message that is a
/// bare `{1: n}` and nothing else — and only targets *in the map* are touched,
/// so a reference out of the stream, a `TSP.DataReference` and any one-field
/// message that is not a reference at all are all left exactly as written.
/// Returns whether anything changed, so an untouched object keeps its bytes.
pub(crate) fn remap_references(
    message: &mut Message,
    map: &BTreeMap<u64, u64>,
    depth: usize,
) -> bool {
    if depth == 0 {
        return false;
    }
    let mut changed = false;
    for field in &mut message.fields {
        let Value::Bytes(raw) = &field.value else {
            continue;
        };
        let Some(mut nested) = decode_nested(raw) else {
            continue;
        };
        match reference_target(&nested) {
            Some(target) => {
                if let Some(replacement) = map.get(&target) {
                    nested.set(1, Value::Varint(*replacement));
                    field.value = Value::Bytes(nested.encode());
                    changed = true;
                }
            }
            None => {
                if remap_references(&mut nested, map, depth - 1) {
                    field.value = Value::Bytes(nested.encode());
                    changed = true;
                }
            }
        }
    }
    changed
}

/// Put `new_node` straight after `after` in the show's slide tree, and say
/// where it landed.
///
/// Errors if `after` is not in the tree, rather than appending at the end: a
/// duplicate whose anchor the deck does not list is a slide inserted nowhere in
/// particular, which is not an operation worth guessing at.
fn insert_into_slide_tree(
    document: &mut crate::Document,
    after: u64,
    new_node: u64,
) -> Result<usize, Error> {
    let (show_id, show_archive, tree, mut order) = read_slide_tree(document)?;
    let index = order
        .iter()
        .position(|id| *id == after)
        .map(|position| position + 1)
        .ok_or_else(|| {
            Error::Format(format!(
                "slide node {after} is not in the show's slide tree; \
                 refusing to insert a copy at the end instead"
            ))
        })?;
    order.insert(index, new_node);
    write_slide_order(document, show_id, show_archive, &tree, &order)?;
    Ok(index)
}

/// Clone a component entry in `TSP.PackageMetadata` for the copied stream.
///
/// Everything comes from the source component: the external references it
/// already declares, its data references with the *using object* remapped, its
/// version fields, its save token. Only three things differ — the identifier,
/// the locator, and the object UUIDs, which have to be new because two
/// components claiming the same object UUID is the one thing a copy must not
/// do. Returns how many `Data/` entries the copy shares with the original.
fn clone_component(
    document: &mut crate::Document,
    source: u64,
    copy: u64,
    stream: &str,
    map: &BTreeMap<u64, u64>,
) -> Result<usize, Error> {
    let locator = stream
        .strip_prefix("Index/")
        .and_then(|name| name.strip_suffix(".iwa"))
        .unwrap_or(stream)
        .to_string();
    // The source must have a `TSP.ComponentInfo` to copy. Without one, the new
    // stream is declared by nobody: `declare_external_references` adds zero
    // declarations — the `Document → new-slide` one included — and Keynote
    // opens the deck, counts the extra slide, and answers `missing value` for
    // its layout, its title and its body. A named error beats an `Ok` that
    // reports `media: 0` and leaves that quiet failure behind.
    if !document.components().iter().any(|c| c.identifier == source) {
        return Err(Error::Format(format!(
            "slide {source} has no TSP.ComponentInfo to copy; refusing to duplicate a slide \
             whose component the package does not declare"
        )));
    }
    let mut media = 0usize;
    document.update_package_metadata(|metadata| {
        let Some(entry) = metadata
            .all(3)
            .filter_map(|value| match value {
                Value::Bytes(raw) => Message::decode(raw).ok(),
                _ => None,
            })
            .find(|info| info.varint(1) == Some(source))
        else {
            return;
        };
        let mut info = Message::default();
        for field in &entry.fields {
            match field.number {
                // identifier and locator, the two things that must differ
                1 => info.set(1, Value::Varint(copy)),
                2 => info.fields.push(field.clone()),
                3 => {}
                // data references: same media, the user object remapped
                7 => {
                    let Value::Bytes(raw) = &field.value else {
                        continue;
                    };
                    let Some(mut reference) = decode_nested(raw) else {
                        continue;
                    };
                    let mut users = Vec::new();
                    for value in reference.all(2) {
                        let Value::Bytes(raw) = value else { continue };
                        let Some(mut user) = decode_nested(raw) else {
                            continue;
                        };
                        if let Some(object) = user.varint(1) {
                            if let Some(replacement) = map.get(&object) {
                                user.set(1, Value::Varint(*replacement));
                            }
                        }
                        users.push(user.encode());
                    }
                    reference.fields.retain(|f| f.number != 2);
                    for user in users {
                        reference.append_in_order(2, Value::Bytes(user));
                    }
                    media += 1;
                    info.append_in_order(7, Value::Bytes(reference.encode()));
                }
                // object UUIDs: the same shape, new values
                11 => {
                    let Value::Bytes(raw) = &field.value else {
                        continue;
                    };
                    let Some(mut uuid_entry) = decode_nested(raw) else {
                        continue;
                    };
                    let Some(object) = uuid_entry.varint(1) else {
                        continue;
                    };
                    let Some(replacement) = map.get(&object) else {
                        continue;
                    };
                    let (lower, upper) = derived_uuid(&uuid_entry, *replacement);
                    let mut uuid = Message::default();
                    uuid.set(1, Value::Varint(lower));
                    uuid.set(2, Value::Varint(upper));
                    uuid_entry.set(1, Value::Varint(*replacement));
                    uuid_entry.set(2, Value::Bytes(uuid.encode()));
                    info.append_in_order(11, Value::Bytes(uuid_entry.encode()));
                }
                _ => info.fields.push(field.clone()),
            }
        }
        // The locator is only written when it differs from the preferred one,
        // which for a copy it always does. Field 3 was skipped as the source's
        // fields were copied above, so it is inserted in ascending order here
        // rather than appended past field 12 — the crate's `set_in_order` rule.
        info.set_in_order(3, Value::Bytes(locator.as_bytes().to_vec()));
        metadata.append_in_order(3, Value::Bytes(info.encode()));
    })?;
    Ok(media)
}

/// A fresh object UUID, derived rather than drawn.
///
/// A copy must not claim the original's object UUIDs — two components saying
/// the same object is the same object is the one way this operation could
/// corrupt a document that still opens. The value is a SHA-1 of the original
/// UUID and the new identifier, which makes the whole duplicate reproducible
/// (the same input gives the same file, so byte-identity tests still mean
/// something) without a random-number dependency.
fn derived_uuid(entry: &Message, identifier: u64) -> (u64, u64) {
    let mut seed = Vec::with_capacity(32);
    if let Some(raw) = entry.bytes(2) {
        seed.extend_from_slice(raw);
    }
    seed.extend_from_slice(b"iwork-rs duplicate-slide");
    seed.extend_from_slice(&identifier.to_le_bytes());
    let digest = crate::media::sha1(&seed);
    let lower = u64::from_le_bytes(digest[0..8].try_into().expect("sha1 is 20 bytes"));
    let upper = u64::from_le_bytes(digest[8..16].try_into().expect("sha1 is 20 bytes"));
    (lower, upper)
}

/// Add a slide to the end of a deck, drawn from one of its layouts.
///
/// The pieces are the ones `Document::new(Kind::Keynote)` writes for its first
/// slide, and the difference is that everything they point at already exists:
/// the master is the deck's own layout rather than a new one, and the style is
/// that layout's. So a slide added to a deck made from *any* theme looks like
/// the theme, which a slide carrying styles invented here would not.
///
/// `layout` is the layout to draw it from — `None` takes the deck's first,
/// which is what Keynote's own New Slide does.
pub fn add_slide(
    document: &mut crate::Document,
    layout: Option<u64>,
) -> Result<crate::keynote::Slide, Error> {
    let deck = show(document).ok_or_else(|| Error::Format("not a Keynote document".into()))?;
    let master = match layout {
        Some(wanted) => deck
            .layouts
            .iter()
            .find(|l| l.identifier == wanted || l.node == wanted)
            .ok_or_else(|| Error::Format(format!("no layout {wanted} in this deck")))?,
        None => deck
            .layouts
            .first()
            .ok_or_else(|| Error::Format("this deck has no layout to draw a slide from".into()))?,
    };
    // The layout's own style, which is what a slide drawn from it uses.
    let master_archive = document.archive(master.identifier)?;
    let style = reference(&master_archive, slide_field::STYLE)
        .ok_or_else(|| Error::Format(format!("layout {} names no style", master.identifier)))?;

    let last_node = deck.slides.last().map(|slide| slide.node);
    let master_id = master.identifier;

    let mut grow = crate::create::Grow::new(document);
    let slide = grow.allocate();
    let node = grow.allocate();
    grow.component(
        "Slide",
        true,
        vec![(
            slide,
            TYPE_SLIDE,
            crate::create::keynote_slide(style, Some(master_id)),
        )],
    )?;
    // The node goes beside the last slide's node, which is in `Document`.
    let neighbour = last_node
        .ok_or_else(|| Error::Format("this deck has no slide to put a new one after".into()))?;
    let node_archive = crate::create::keynote_slide_node(slide, node);
    grow.beside(neighbour, node, TYPE_SLIDE_NODE, &node_archive)?;
    grow.finish()?;

    insert_into_slide_tree(document, neighbour, node)?;

    show(document)
        .map(|deck| deck.slides)
        .unwrap_or_default()
        .into_iter()
        .find(|s| s.identifier == slide)
        .ok_or_else(|| Error::Format("the new slide is not in the deck".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pb::Field;

    fn message(fields: Vec<(u32, Value)>) -> Message {
        Message {
            fields: fields
                .into_iter()
                .map(|(number, value)| Field { number, value })
                .collect(),
        }
    }

    fn reference_bytes(id: u64) -> Value {
        let mut r = Message::default();
        r.set(1, Value::Varint(id));
        Value::Bytes(r.encode())
    }

    #[test]
    fn placeholder_kinds_are_the_schemas_five() {
        assert_eq!(PlaceholderKind::of(0), PlaceholderKind::Generic);
        assert_eq!(PlaceholderKind::of(1), PlaceholderKind::SlideNumber);
        assert_eq!(PlaceholderKind::of(2), PlaceholderKind::Title);
        assert_eq!(PlaceholderKind::of(3), PlaceholderKind::Body);
        assert_eq!(PlaceholderKind::of(4), PlaceholderKind::Object);
        assert_eq!(PlaceholderKind::of(9), PlaceholderKind::Unknown(9));
    }

    /// Only the targets in the map move. A reference out of the stream is the
    /// whole point of a slide component — the layout, the stylesheet, the
    /// slide style all live elsewhere — and remapping one would break it.
    #[test]
    fn remapping_leaves_references_out_of_the_map_alone() {
        let mut archive = message(vec![
            (1, reference_bytes(100)),
            (5, reference_bytes(200)),
            (7, reference_bytes(300)),
        ]);
        let map = BTreeMap::from([(200u64, 999u64)]);
        assert!(remap_references(&mut archive, &map, MAX_DEPTH));
        let target = |number: u32| {
            decode_nested(archive.bytes(number).unwrap())
                .and_then(|r| reference_target(&r))
                .unwrap()
        };
        assert_eq!(target(1), 100, "outside the map, untouched");
        assert_eq!(target(5), 999, "inside the map, moved");
        assert_eq!(target(7), 300);
    }

    /// An object with nothing to remap keeps its bytes, which is what lets a
    /// duplicate leave every untouched stream byte-identical.
    #[test]
    fn remapping_nothing_reports_nothing() {
        let mut archive = message(vec![(1, reference_bytes(100))]);
        let before = archive.encode();
        assert!(!remap_references(&mut archive, &BTreeMap::new(), MAX_DEPTH));
        assert_eq!(archive.encode(), before);
    }

    /// References nest: a slide's transition holds its attributes three levels
    /// down, and a shape's storage sits under its super.
    #[test]
    fn remapping_reaches_into_nested_messages() {
        let inner = message(vec![(2, reference_bytes(7))]);
        let mut archive = message(vec![(1, Value::Bytes(inner.encode()))]);
        let map = BTreeMap::from([(7u64, 70u64)]);
        assert!(remap_references(&mut archive, &map, MAX_DEPTH));
        let nested = decode_nested(archive.bytes(1).unwrap()).unwrap();
        assert_eq!(
            decode_nested(nested.bytes(2).unwrap()).and_then(|r| reference_target(&r)),
            Some(70)
        );
    }

    /// The same copy twice gives the same UUID, so a duplicate is reproducible
    /// and the byte-identity tests still say something.
    #[test]
    fn a_derived_uuid_is_a_function_of_its_inputs() {
        let mut uuid = Message::default();
        uuid.set(1, Value::Varint(11));
        uuid.set(2, Value::Varint(22));
        let entry = message(vec![
            (1, Value::Varint(5)),
            (2, Value::Bytes(uuid.encode())),
        ]);
        let once = derived_uuid(&entry, 4242);
        assert_eq!(once, derived_uuid(&entry, 4242));
        assert_ne!(once, derived_uuid(&entry, 4243), "and it depends on the id");
        assert_ne!(once, (11, 22), "and it is not the one it came from");
    }

    #[test]
    fn a_transition_with_no_attributes_is_none() {
        assert!(transition(&Message::default()).is_none());
        let t = Transition {
            effect: "apple:dissolve".into(),
            ..Transition::default()
        };
        assert!(!t.is_none());
    }

    /// The effect table is the app's own pairing, and neither half of it
    /// repeats — a duplicate identifier would make [`effect_name`] answer for
    /// the wrong effect.
    #[test]
    fn the_effect_table_is_a_bijection() {
        let names: BTreeSet<&str> = EFFECTS.iter().map(|(name, _)| *name).collect();
        let wires: BTreeSet<&str> = EFFECTS.iter().map(|(_, wire)| *wire).collect();
        assert_eq!(names.len(), EFFECTS.len(), "no name appears twice");
        assert_eq!(wires.len(), EFFECTS.len(), "no identifier appears twice");
        assert_eq!(effect_name("none"), Some("no transition effect"));
        // The three the reader could not have guessed.
        assert_eq!(effect_name("apple:revolve"), Some("flip"));
        assert_eq!(effect_name("apple:ca-revolve"), Some("object revolve"));
        assert_eq!(effect_name("apple:slide"), Some("move in"));
        assert_eq!(effect_name("apple:not-a-real-effect"), None);
    }

    /// Only the parameters that are on the wire come back. Keynote writes the
    /// ones the effect has and no others, so absent has to stay absent — an
    /// effect without a bounce is not an effect whose bounce is off.
    #[test]
    fn a_parameter_that_is_not_written_is_not_read() {
        let mut animation = Message::default();
        animation.set(
            transition_field::EFFECT,
            Value::Bytes(b"apple:scale".into()),
        );
        let mut attributes = Message::default();
        attributes.set(
            transition_field::ANIMATION_ATTRIBUTES,
            Value::Bytes(animation.encode()),
        );
        attributes.set(transition_field::CUSTOM_BOUNCE, Value::Varint(0));

        let decoded = transition_attributes(&attributes);
        assert_eq!(decoded.effect, "apple:scale");
        assert_eq!(decoded.parameters.bounce, Some(false), "written false");
        assert_eq!(decoded.parameters.twist, None, "not written at all");
        assert!(!decoded.parameters.is_empty());
        assert!(decoded.unknown_parameters.is_empty());
    }

    /// The tripwire fires on a field the 15.3.1 schema does not name — 14,
    /// which is a hole in the message, and anything past 20.
    #[test]
    fn an_unnamed_parameter_field_is_reported() {
        let mut attributes = Message::default();
        attributes.set(
            transition_field::ANIMATION_ATTRIBUTES,
            Value::Bytes(Message::default().encode()),
        );
        attributes.set(14, Value::Varint(1));
        attributes.set(31, Value::Varint(1));
        assert_eq!(
            transition_attributes(&attributes).unknown_parameters,
            vec![14, 31]
        );
    }

    /// The two enums of `KN.TransitionAttributesArchive`, and what a value the
    /// schema does not list does.
    #[test]
    fn the_transition_enums_are_the_schemas() {
        assert_eq!(TimingCurve::of(1), TimingCurve::Linear);
        assert_eq!(TimingCurve::of(4), TimingCurve::EaseInEaseOut);
        assert_eq!(TimingCurve::of(5), TimingCurve::Custom);
        assert_eq!(TimingCurve::of(9), TimingCurve::Unknown(9));
        assert_eq!(TimingCurve::of(9).as_str(), "unknown");
        assert_eq!(TextDelivery::of(1), TextDelivery::ByObject);
        assert_eq!(TextDelivery::of(2), TextDelivery::ByWord);
        assert_eq!(TextDelivery::of(3), TextDelivery::ByCharacter);
        assert_eq!(TextDelivery::of(4), TextDelivery::ByLine);
        assert_eq!(TextDelivery::of(0), TextDelivery::Unknown(0));
    }

    /// Direction 0 is what the app writes and it means "the effect's default";
    /// the named values are the two families the PowerPoint importer produced.
    #[test]
    fn directions_are_two_families_of_four() {
        assert_eq!(direction_name(0), "the effect's own default");
        assert_eq!(direction_name(11), "left to right");
        assert_eq!(direction_name(14), "bottom to top");
        assert_eq!(direction_name(21), "top left to bottom right");
        assert_eq!(direction_name(24), "bottom right to top left");
        assert_eq!(direction_name(1), "unknown", "1 is not 11");
        assert_eq!(direction_name(15), "unknown");
    }

    /// A soundtrack's `movie_media` is an ordered list of data identifiers, and
    /// order is the play order — so it is read as a list, not as a set.
    #[test]
    fn a_soundtrack_track_list_keeps_its_order() {
        let mut archive = Message::default();
        archive.set(
            soundtrack_field::VOLUME,
            Value::Fixed64(0.5f64.to_le_bytes()),
        );
        archive.set(soundtrack_field::MODE, Value::Varint(1));
        for id in [30u64, 10, 20] {
            archive.append_in_order(soundtrack_field::MOVIE_MEDIA, reference_bytes(id));
        }
        let track = Soundtrack {
            identifier: 1,
            volume: double(&archive, soundtrack_field::VOLUME, 0.0),
            mode: archive.varint(soundtrack_field::MODE).unwrap_or(0),
            media: references(&archive, soundtrack_field::MOVIE_MEDIA),
        };
        assert_eq!(track.media, vec![30, 10, 20]);
        assert_eq!(track.tracks(), 3);
        assert_eq!(track.mode_name(), "loop");
        assert_eq!(track.volume, 0.5);
    }

    /// A build is decoded from the schema and has never been seen; what this
    /// asserts is that the decode does not invent one. An empty archive is a
    /// build with no drawable, no delivery and no effect — not a panic, and not
    /// a build that claims to know something.
    #[test]
    fn a_build_decodes_without_claiming_anything() {
        let empty = build(7, &Message::default());
        assert_eq!(empty.identifier, 7);
        assert_eq!(empty.drawable, None);
        assert!(empty.delivery.is_empty());
        assert_eq!(empty.event_trigger, None);
        assert!(empty.animation.is_none());
        assert_eq!(empty.animation.effect, "none");
        assert!(!empty.has_motion_path);
        assert!(empty.unknown_attributes.is_empty());

        // A build's animation attributes are a transition's, read by the same
        // code — so an effect on a build reads out the same way.
        let mut animation = Message::default();
        animation.set(
            transition_field::EFFECT,
            Value::Bytes(b"apple:move-in".into()),
        );
        animation.set(
            transition_field::DURATION,
            Value::Fixed64(2.0f64.to_le_bytes()),
        );
        let mut attributes = Message::default();
        attributes.set(
            build_field::ANIMATION_ATTRIBUTES,
            Value::Bytes(animation.encode()),
        );
        attributes.set(build_field::EVENT_TRIGGER, Value::Varint(2));
        attributes.set(99, Value::Varint(1));
        let mut archive = Message::default();
        archive.set(build_field::DELIVERY, Value::Bytes(b"byParagraph".into()));
        archive.set(build_field::DRAWABLE, reference_bytes(42));
        archive.set(build_field::ATTRIBUTES, Value::Bytes(attributes.encode()));

        let decoded = build(8, &archive);
        assert_eq!(decoded.drawable, Some(42));
        assert_eq!(decoded.delivery, "byParagraph");
        assert_eq!(decoded.event_trigger, Some(2));
        assert_eq!(decoded.animation.effect, "apple:move-in");
        assert_eq!(decoded.animation.duration, 2.0);
        assert_eq!(decoded.unknown_attributes, vec![99]);
    }
}

// -- writing a transition -----------------------------------------------------

/// What a transition is to become. `None` in a field keeps what the slide has,
/// or takes the app's own default when it has none.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TransitionEdit {
    /// The effect, by the name [`effect_name`] prints or by the identifier the
    /// archive holds — `"dissolve"` and `"apple:dissolve"` are the same thing.
    /// `"none"` removes the transition.
    pub effect: String,
    /// Seconds. The app writes 1.0 for every one of the 44 effects.
    pub duration: Option<f64>,
    /// Seconds before it starts.
    pub delay: Option<f64>,
    /// Advance without a click.
    pub automatic: Option<bool>,
    /// Which way it travels. **Keynote's own writer never emits this**, and 0 —
    /// the effect's own default — is what every effect set from a script
    /// carries; the non-zero values are known only from decks its PowerPoint
    /// importer wrote. Setting one is therefore writing a field the app does
    /// not, which is why it is opt-in rather than defaulted.
    pub direction: Option<u64>,
}

/// The duration the app gives every one of the 44 effects.
const DEFAULT_TRANSITION_DURATION: f64 = 1.0;

/// Give a slide a transition, or take its transition away.
///
/// **A transition belongs to the slide it leaves** — Keynote plays it on the
/// way *out* — so this is the slide the audience is looking at when the effect
/// runs, not the one it reveals.
///
/// Two objects change: the `KN.SlideArchive`'s `transition` (4), and the
/// `KN.SlideNodeArchive`'s `has_transition` (7), which is the flag the app's
/// own navigator reads. Writing one without the other gives a deck that plays
/// a transition the outline does not show, or shows one it does not play.
///
/// **The parameters belong to the effect.** `custom_bounce`, `custom_twist`,
/// the Magic Move settings and the rest sit beside the animation attributes and
/// mean nothing to a different effect, so changing the effect drops them rather
/// than carrying them across; keeping the same effect keeps them. The random
/// seed is kept when there is one and minted when there is not — the app copies
/// it verbatim through its own duplicate, so it is an identity, not a nonce.
pub fn set_transition(
    document: &mut crate::Document,
    slide: u64,
    edit: &TransitionEdit,
) -> Result<Transition, Error> {
    // Naming a slide by its node is what `iwork slides` prints and what every
    // other write here accepts, and the *slide* archive is what carries the
    // transition — so resolve one to the other before touching anything.
    let (node, _) = slide_node_in_deck(document, slide)?;
    let slide = reference(&document.archive(node)?, node_field::SLIDE)
        .ok_or_else(|| Error::Format(format!("slide node {node} names no slide")))?;
    let mut archive = document.archive(slide)?;
    let wire = transition_identifier(&edit.effect)?;

    let was = transition(&archive);
    let existing = archive
        .bytes(slide_field::TRANSITION)
        .and_then(decode_nested)
        .and_then(|t| {
            t.bytes(transition_field::ATTRIBUTES)
                .and_then(decode_nested)
        });

    let set_has_transition = |document: &mut crate::Document, has: bool| -> Result<(), Error> {
        let mut node_archive = document.archive(node)?;
        node_archive.set(node_field::HAS_TRANSITION, Value::Varint(u64::from(has)));
        document.set_archive_of(node, &node_archive)
    };

    // **`transition` is a required field of `KN.SlideArchive`.** Removing it is
    // not "no transition": Keynote refuses to parse the slide at all — "Cannot
    // parse message of type KN.SlideArchive because it is missing required
    // fields: transition", and then the whole component fails to load. What
    // the app writes for a slide with no transition is the same message with
    // the effect `"none"` in it, which is what this writes too.

    // The animation attributes, in the order and the shape the app writes them.
    let mut animation = Message::default();
    animation.set_in_order(
        transition_field::ANIMATION_TYPE,
        Value::Bytes(b"Transition".to_vec()),
    );
    animation.set_in_order(
        transition_field::EFFECT,
        Value::Bytes(wire.as_bytes().to_vec()),
    );
    animation.set_in_order(
        transition_field::DURATION,
        Value::Fixed64(
            edit.duration
                .or((!was.is_none()).then_some(was.duration))
                .unwrap_or(DEFAULT_TRANSITION_DURATION)
                .to_le_bytes(),
        ),
    );
    if let Some(direction) = edit.direction {
        animation.set_in_order(transition_field::DIRECTION, Value::Varint(direction));
    }
    animation.set_in_order(
        transition_field::DELAY,
        Value::Fixed64(edit.delay.unwrap_or(was.delay).to_le_bytes()),
    );
    animation.set_in_order(
        transition_field::AUTOMATIC,
        Value::Varint(u64::from(edit.automatic.unwrap_or(was.automatic))),
    );
    animation.set_in_order(
        transition_field::SEED,
        Value::Varint(match was.seed {
            0 => transition_seed(slide),
            seed => seed,
        }),
    );
    animation.set_in_order(transition_field::RTL, Value::Varint(u64::from(was.rtl)));

    let mut attributes = Message::default();
    attributes.set_in_order(
        transition_field::ANIMATION_ATTRIBUTES,
        Value::Bytes(animation.encode()),
    );
    // The `custom_*` block: kept only when the effect is the one it was written
    // for. A bounce on a dissolve is not a bounce, it is a field the app has to
    // ignore or misread — and "no transition" carries none at all.
    if wire != "none" && was.effect == wire_of(&edit.effect) {
        if let Some(existing) = &existing {
            for field in &existing.fields {
                if field.number != transition_field::ANIMATION_ATTRIBUTES {
                    attributes.set_in_order(field.number, field.value.clone());
                }
            }
        }
    }

    let mut holder = Message::default();
    holder.set_in_order(
        transition_field::ATTRIBUTES,
        Value::Bytes(attributes.encode()),
    );
    archive.set_in_order(slide_field::TRANSITION, Value::Bytes(holder.encode()));
    document.set_archive_of(slide, &archive)?;
    set_has_transition(document, wire != "none")?;

    Ok(transition(&archive))
}

/// The archive identifier an effect is written as, from either spelling.
fn transition_identifier(effect: &str) -> Result<String, Error> {
    let wanted = effect.trim();
    if let Some((_, wire)) = EFFECTS
        .iter()
        .find(|(name, wire)| name.eq_ignore_ascii_case(wanted) || *wire == wanted)
    {
        return Ok((*wire).to_string());
    }
    Err(Error::Format(format!(
        "{effect:?} is not one of the {} effects Keynote offers — `iwork effects` lists them, \
         by name and by the identifier the archive holds",
        EFFECTS.len()
    )))
}

/// The same lookup, for comparing against what a slide already carries.
fn wire_of(effect: &str) -> String {
    transition_identifier(effect).unwrap_or_else(|_| effect.to_string())
}

/// A `random_number_seed` for a slide that has none.
///
/// The app copies this verbatim through its own duplicate, so it identifies the
/// transition rather than randomising it; deriving it from the slide keeps a
/// deck this crate writes twice byte-identical.
fn transition_seed(slide: u64) -> u64 {
    let mut z = slide
        .wrapping_mul(0x2545_F491_4F6C_DD1D)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    // The app's seeds are 32-bit-ish (36861517 on the fixture), so this stays
    // in the range one of its own would occupy.
    (z ^ (z >> 31)) & 0x7FFF_FFFF
}

// -- writing a build ----------------------------------------------------------

/// Whether a build brings a drawable on or takes it off.
///
/// This is `animation_type` on the animation attributes, and it is the whole of
/// the distinction: `keynote-builds.key`'s four build-ins say `"In"` and its
/// four build-outs say `"Out"`, and nothing else differs in kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BuildKind {
    #[default]
    In,
    Out,
}

impl BuildKind {
    fn animation_type(self) -> &'static str {
        match self {
            BuildKind::In => "In",
            BuildKind::Out => "Out",
        }
    }

    /// The effect the Animate inspector gives a new build of this kind, and the
    /// only two build effects this corpus has measured.
    pub fn default_effect(self) -> &'static str {
        match self {
            BuildKind::In => "apple:dissolve character",
            BuildKind::Out => "apple:bc-appear",
        }
    }
}

/// What a build is to be.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildEdit {
    pub kind: BuildKind,
    /// The build effect's identifier. **Not a transition effect** — the two
    /// enumerations are different, `"apple:dissolve character"` is a build and
    /// `"apple:dissolve"` is a transition — and only the two in
    /// [`BuildKind::default_effect`] have been seen written by the app. Empty
    /// takes the kind's default.
    pub effect: String,
    pub duration: Option<f64>,
    pub delay: Option<f64>,
}

impl Default for BuildEdit {
    fn default() -> BuildEdit {
        BuildEdit {
            kind: BuildKind::In,
            effect: String::new(),
            duration: None,
            delay: None,
        }
    }
}

/// The duration and event trigger the Animate inspector gives a new build.
const DEFAULT_BUILD_DURATION: f64 = 1.0;
/// `eventTrigger` 1 — On Click, which is what all eight builds of the fixture
/// carry.
const BUILD_ON_CLICK: u64 = 1;

/// Animate a drawable on or off a slide.
///
/// **A build is two objects and two lists.** The `KN.BuildArchive` (8) says
/// what is animated and how, and goes in the slide's `builds` (2); a
/// `KN.BuildChunkArchive` (153) says *when*, and goes in the slide's
/// `buildChunks` (43), which is the order the app plays them in. The node's
/// `has_builds` (6), `has_explicit_builds` (20) and `build_event_count` (15)
/// are what the app's own navigator reads, and a deck with builds the node does
/// not count is a deck whose outline disagrees with what it plays.
///
/// Two fields are written because `keynote-builds.key` has them and **neither
/// has been measured**: the build attributes' field 17 (`60`) and the chunk's
/// fields 6, 7 and 8, a flag and a pair of UUIDs. They are reproduced in the
/// shape the app wrote them, with the UUIDs derived from the chunk's own
/// identifier the way every other identity here is. Nothing pretends to know
/// what they mean.
///
/// The effect is an identifier of the **build** enumeration, not the
/// transition one — `"apple:dissolve character"` rather than `"apple:dissolve"`
/// — and only the two in [`BuildKind::default_effect`] have been seen written
/// by the app, so anything else is passed through and taken on trust.
pub fn add_build(
    document: &mut crate::Document,
    slide: u64,
    drawable: u64,
    edit: &BuildEdit,
) -> Result<u64, Error> {
    let (node, _) = slide_node_in_deck(document, slide)?;
    let slide = reference(&document.archive(node)?, node_field::SLIDE)
        .ok_or_else(|| Error::Format(format!("slide node {node} names no slide")))?;
    let mut archive = document.archive(slide)?;

    // The drawable has to be *on this slide*: a build naming a shape the slide
    // does not own animates nothing, and says nothing about it either.
    let owned: Vec<u64> = references(&archive, slide_field::OWNED_DRAWABLES);
    if !owned.contains(&drawable) {
        return Err(Error::Format(format!(
            "drawable {drawable} is not on slide {slide} — a build animates something the \
             slide owns, and this one owns {}",
            match owned.len() {
                0 => "nothing".to_string(),
                n => format!("{n} drawable(s)"),
            }
        )));
    }
    let effect = match edit.effect.trim() {
        "" => edit.kind.default_effect().to_string(),
        given => given.to_string(),
    };

    let mut grow = crate::create::Grow::new(document);
    let build = grow.allocate();
    let chunk = grow.allocate();
    grow.beside(
        slide,
        build,
        TYPE_BUILD,
        &build_archive(drawable, &effect, edit, build),
    )?;
    grow.beside(
        slide,
        chunk,
        TYPE_BUILD_CHUNK,
        &chunk_archive(build, edit, chunk),
    )?;
    grow.finish()?;

    archive.append_in_order(
        slide_field::BUILDS,
        Value::Bytes(crate::create::reference_bytes(build)),
    );
    archive.append_in_order(
        slide_field::BUILD_CHUNKS,
        Value::Bytes(crate::create::reference_bytes(chunk)),
    );
    document.set_archive_of(slide, &archive)?;

    // The node's three counters, which is what the navigator shows.
    let chunks = references(&archive, slide_field::BUILD_CHUNKS).len() as u64;
    let mut node_archive = document.archive(node)?;
    node_archive.set(node_field::HAS_BUILDS, Value::Varint(1));
    node_archive.set(node_field::HAS_EXPLICIT_BUILDS, Value::Varint(1));
    node_archive.set(node_field::BUILD_EVENT_COUNT, Value::Varint(chunks));
    document.set_archive_of(node, &node_archive)?;
    document.declare_external_references();
    Ok(build)
}

/// `KN.BuildArchive`, in the shape the Animate inspector writes one.
fn build_archive(drawable: u64, effect: &str, edit: &BuildEdit, seed: u64) -> Message {
    use crate::create::{double, message, nested, reference, string, varint};
    message(vec![
        reference(build_field::DRAWABLE, drawable),
        // `delivery` is a *required* string and this is the value all eight
        // builds of the fixture carry.
        string(build_field::DELIVERY, "All at Once"),
        double(3, 0.0),
        nested(
            build_field::ATTRIBUTES,
            vec![
                varint(build_field::EVENT_TRIGGER, BUILD_ON_CLICK),
                // Unmeasured, and 60 on every build the app wrote.
                double(17, 60.0),
                nested(
                    build_field::ANIMATION_ATTRIBUTES,
                    vec![
                        string(transition_field::ANIMATION_TYPE, edit.kind.animation_type()),
                        string(transition_field::EFFECT, effect),
                        double(
                            transition_field::DURATION,
                            edit.duration.unwrap_or(DEFAULT_BUILD_DURATION),
                        ),
                        double(transition_field::DELAY, edit.delay.unwrap_or(0.0)),
                        varint(transition_field::SEED, transition_seed(seed)),
                        varint(transition_field::RTL, 0),
                    ],
                ),
                varint(build_field::CUSTOM_TEXT_DELIVERY, 1),
                varint(build_field::CUSTOM_DELIVERY_OPTION, 1),
            ],
        ),
        varint(build_field::CHUNK_ID_SEED, 1),
    ])
}

/// `KN.BuildChunkArchive` — when the build plays.
fn chunk_archive(build: u64, edit: &BuildEdit, seed: u64) -> Message {
    use crate::create::{double, message, nested, reference, varint};
    let uuid = || {
        vec![
            varint(1, transition_seed(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            varint(
                2,
                transition_seed(seed ^ 1).wrapping_mul(0xD1B5_4A32_D192_ED03),
            ),
        ]
    };
    message(vec![
        reference(build_field::CHUNK_BUILD, build),
        double(build_field::CHUNK_DELAY, edit.delay.unwrap_or(0.0)),
        double(
            build_field::CHUNK_DURATION,
            edit.duration.unwrap_or(DEFAULT_BUILD_DURATION),
        ),
        varint(build_field::CHUNK_AUTOMATIC, 0),
        // 6, 7 and 8 are the app's and are unmeasured; the shape is the
        // fixture's and the identities are derived, as every identity here is.
        varint(build_field::CHUNK_REFERENT, 1),
        nested(7, vec![nested(1, uuid()), varint(2, 1)]),
        nested(8, uuid()),
    ])
}

/// Give a slide presenter notes it does not have yet, and hand back the storage
/// an edit takes.
///
/// A slide made from nothing has none: the note is not a field of the slide but
/// **two objects** — a `KN.NoteArchive`, which is `{1: → storage}` and nothing
/// else, and the `TSWP.StorageArchive` of **kind 4** it points at. Both go in
/// the slide's own stream, the slide's `note` (27) points at the archive, and
/// the node's `hasNote` follows the text.
///
/// The storage is the one `create::text_box_storage` writes with its
/// kind changed: the field set is identical — `[1, 2, 3, 5, 6, 7, 10, 14, 24]`
/// — which was read off `keynote-deck.key`'s own note rather than assumed.
pub fn add_note(document: &mut crate::Document, slide: u64) -> Result<u64, crate::Error> {
    use crate::{Error, Refusal};

    let show = document
        .show()
        .ok_or_else(|| Error::refused(Refusal::NotFound, "not a Keynote document"))?;
    let found = show
        .slide(slide)
        .ok_or_else(|| Error::refused(Refusal::NotFound, format!("no slide {slide}")))?;
    if let Some(storage) = found.note_storage {
        return Ok(storage);
    }
    let slide = found.identifier;
    let (stylesheet, paragraph, list) =
        crate::drawable::text_styles(document).ok_or_else(|| {
            Error::refused(
                Refusal::Missing,
                "this document has no text styles to write notes with",
            )
        })?;

    let mut grow = crate::create::Grow::new(document);
    let storage = grow.allocate();
    let note = grow.allocate();
    grow.beside(
        slide,
        storage,
        crate::TYPE_STORAGE,
        &crate::create::storage_of_kind(
            crate::create::STORAGE_NOTE,
            stylesheet,
            paragraph,
            list,
            "",
        ),
    )?;
    grow.beside(
        slide,
        note,
        TYPE_NOTE,
        &crate::create::message(vec![crate::create::reference(1, storage)]),
    )?;
    grow.finish()?;

    let mut archive = document.archive(slide)?;
    archive.set_in_order(
        slide_field::NOTE,
        crate::pb::Value::Bytes(crate::create::reference_bytes(note)),
    );
    document.set_archive_of(slide, &archive)?;
    Ok(storage)
}
