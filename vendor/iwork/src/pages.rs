//! `TP` — the Pages document spine: modes, sections, headers, footers, page
//! templates, footnotes, contents and flows.
//!
//! Only Pages writes these archives, and only Pages has this shape:
//!
//! ```text
//!  TP.DocumentArchive
//!    ├── body_storage (4) ──── TSWP.StorageArchive, kind = body
//!    │                           └── table_section (17): {char index → section}
//!    ├── settings (7) ──────── TP.SettingsArchive: body? headers? footnotes?
//!    ├── page_templates (48) ─ TP.PageTemplateArchive …   (page layout only)
//!    ├── toc_styles (14) ───── TSWP.TOCSettingsArchive …
//!    └── flow_info_container (47) ─ TSWP.FlowInfoContainerArchive
//!                                     └── TSWP.FlowInfoArchive: a linked
//!                                         text-box thread
//!
//!  TP.SectionArchive
//!    ├── first / even / odd section-template page (23, 24, 25)
//!    └── each a TP.SectionTemplateArchive
//!          ├── headers (1): exactly three storages — left, centre, right
//!          ├── footers (2): exactly three more
//!          └── section_template_drawables (3)
//! ```
//!
//! **The schema is twelve years newer than the published one.** Everything
//! called a "page master" in 2013 is a *section template* in 15.3.1 — the type
//! at registry id 10143 is `TP.SectionTemplateArchive`, not the
//! `TP.PageLayoutArchive` this crate used to call it — and six `TP` commands
//! changed superclass. Names here come from the descriptors carved out of the
//! installed 15.3.1 binaries (`reference/protos-15.3/pages/TPArchives.proto`),
//! never from the 2013 tables.
//!
//! ## Where a section begins
//!
//! A section is an entry in the body storage's `table_section` (field 17),
//! anchored like a paragraph: at the character **after** the `U+0004` that
//! begins it, not on the break itself. The first section starts at 0 with no
//! break in front of it. So section *i* covers `[start_i, start_{i+1} - 1)` —
//! the break belongs to no section's text — and the last runs to the end.
//!
//! That is checked against the app: Pages reports `body text of section 1…3`
//! of `pages-report` as 145, 923 and 432 characters, and the entries are at 0,
//! 146 and 1070 in a 1502-unit storage. 146−0−1 = 145, 1070−146−1 = 923,
//! 1502−1070 = 432.

use std::collections::BTreeMap;

use crate::pb::{decode_nested, Message, Value};
use crate::style::reference_target;

/// `TP.DocumentArchive`.
pub const TYPE_DOCUMENT: u32 = 10000;
/// `TP.ThemeArchive`.
pub const TYPE_THEME: u32 = 10001;
/// `TP.FloatingDrawablesArchive`.
pub const TYPE_FLOATING_DRAWABLES: u32 = 10010;
/// `TP.SectionArchive`.
pub const TYPE_SECTION: u32 = 10011;
/// `TP.SettingsArchive`.
pub const TYPE_SETTINGS: u32 = 10012;
/// `TP.DrawablesZOrderArchive`.
pub const TYPE_ZORDER: u32 = 10015;
/// `TP.UserDefinedGuideMapArchive`.
pub const TYPE_GUIDE_MAP: u32 = 10016;
/// `TP.PageTemplateArchive` — page-layout documents only.
pub const TYPE_PAGE_TEMPLATE: u32 = 10017;
/// `TP.SectionTemplateArchive` — 2013's "page master".
pub const TYPE_SECTION_TEMPLATE: u32 = 10143;
/// `TSWP.TOCSettingsArchive` — which paragraph styles a contents list gathers.
pub const TYPE_TOC_SETTINGS: u32 = 2051;
/// `TSWP.TOCEntryInstanceArchive` — one line of a contents list, as laid out.
pub const TYPE_TOC_ENTRY_INSTANCE: u32 = 2052;
/// `TSWP.TOCInfoArchive` — the contents list itself, which is a drawable.
pub const TYPE_TOC_INFO: u32 = 2240;
/// `TSWP.FlowInfoArchive` — one linked-text-box thread.
pub const TYPE_FLOW_INFO: u32 = 2410;
/// `TSWP.FlowInfoContainerArchive`.
pub const TYPE_FLOW_INFO_CONTAINER: u32 = 2411;
/// `TSWP.FootnoteReferenceAttachmentArchive` — the mark in the text, and the
/// only route from a body storage to a note's own storage.
pub const TYPE_FOOTNOTE_REFERENCE: u32 = 2008;
/// `TSWP.BookmarkFieldArchive`.
pub const TYPE_BOOKMARK_FIELD: u32 = 2035;
/// `TSWP.NumberAttachmentArchive` — a page number, a page count or a footnote
/// mark, standing behind a `U+FFFC`.
pub const TYPE_NUMBER_ATTACHMENT: u32 = 2043;

/// Field numbers of `TP.DocumentArchive`.
pub mod document_field {
    pub const BODY_STORAGE: u32 = 4;
    pub const SECTION: u32 = 5;
    pub const SETTINGS: u32 = 7;
    pub const TOC_STYLES: u32 = 14;
    pub const SUPER: u32 = 15;
    pub const CHANGE_SESSIONS: u32 = 16;
    pub const MOST_RECENT_CHANGE_SESSION: u32 = 17;
    pub const DRAWABLES_ZORDER: u32 = 20;
    pub const USES_SINGLE_HEADER_FOOTER: u32 = 21;
    pub const PAGE_WIDTH: u32 = 30;
    pub const PAGE_HEIGHT: u32 = 31;
    pub const LEFT_MARGIN: u32 = 32;
    pub const RIGHT_MARGIN: u32 = 33;
    pub const TOP_MARGIN: u32 = 34;
    pub const BOTTOM_MARGIN: u32 = 35;
    pub const HEADER_MARGIN: u32 = 36;
    pub const FOOTER_MARGIN: u32 = 37;
    pub const PAGE_SCALE: u32 = 38;
    pub const LAYS_OUT_BODY_VERTICALLY: u32 = 39;
    pub const CHANGE_TRACKING_ENABLED: u32 = 40;
    pub const ORIENTATION: u32 = 42;
    pub const PRINTER_ID: u32 = 43;
    pub const PAPER_ID: u32 = 44;
    pub const FLOW_INFO_CONTAINER: u32 = 47;
    pub const PAGE_TEMPLATES: u32 = 48;
}

/// Change tracking, as `TP` records it — the *document's* half of the feature.
///
/// The text's half is `table_insertion` (21) and `table_deletion` (22), and
/// [`crate::annotations`] has both and what they cost an edit.
///
/// **`pages-tracked.pages` is the one document that has this on.** Track
/// Changes was turned on there from the menu (an unlocked screen — Pages'
/// dictionary has no change-tracking property), so `enabled` is true and
/// `sessions` is 1, and its body storage carries the text half too — a
/// `table_deletion` an edit is refused through, which is what
/// [`tests`](crate::annotations) pins. Every other document in the corpus, and
/// all 640 bundled templates, is at the schema defaults: `enabled` false,
/// `sessions` empty, the six display switches at rest. Those switch values are
/// still Confirmed-to-exist and Unverified-in-every-other-setting, because
/// nothing here moves one on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChangeTracking {
    /// `TP.DocumentArchive.change_tracking_enabled` (40).
    pub enabled: bool,
    /// How many `TSWP.ChangeSessionArchive` the document lists (16).
    pub sessions: usize,
    /// Whether it names a most recent one (17).
    pub has_recent_session: bool,
    /// `show_ct_markup` (12), default true.
    pub show_markup: bool,
    /// `show_ct_deletions` (13), default true.
    pub show_deletions: bool,
    /// `ct_bubbles_visibility` (14) — an `int32` with no documented default.
    pub bubbles_visibility: u64,
    /// `change_bars_visible` (15), default true.
    pub change_bars: bool,
    /// `format_changes_visible` (16), default true.
    pub format_changes: bool,
    /// `annotations_visible` (17), default true. The comment layer's switch,
    /// not change tracking's.
    pub annotations_visible: bool,
}

impl ChangeTracking {
    /// Is this the state every document but `pages-tracked.pages` is in?
    pub fn is_default(&self) -> bool {
        !self.enabled
            && self.sessions == 0
            && !self.has_recent_session
            && self.show_markup
            && self.show_deletions
            && self.bubbles_visibility == 0
            && self.change_bars
            && self.format_changes
            && self.annotations_visible
    }
}

/// Field numbers of `TP.SettingsArchive`.
pub mod settings_field {
    /// `body` — **and the document mode**. False is a page-layout document.
    pub const BODY: u32 = 1;
    pub const HEADERS: u32 = 2;
    pub const FOOTERS: u32 = 3;
    pub const HYPHENATION: u32 = 9;
    /// The six change-tracking *display* switches. Present in every document;
    /// `enabled` is exercised by `pages-tracked.pages` but these six sit at
    /// their schema defaults everywhere, because nothing in Pages' dictionary
    /// moves one.
    pub const SHOW_CT_MARKUP: u32 = 12;
    pub const SHOW_CT_DELETIONS: u32 = 13;
    pub const CT_BUBBLES_VISIBILITY: u32 = 14;
    pub const CHANGE_BARS_VISIBLE: u32 = 15;
    pub const FORMAT_CHANGES_VISIBLE: u32 = 16;
    pub const ANNOTATIONS_VISIBLE: u32 = 17;
    pub const DOCUMENT_IS_RTL: u32 = 18;
    pub const LANGUAGE: u32 = 21;
    pub const ORIG_TEMPLATE: u32 = 25;
    pub const FOOTNOTE_KIND: u32 = 30;
    pub const FOOTNOTE_FORMAT: u32 = 31;
    pub const FOOTNOTE_NUMBERING: u32 = 32;
    pub const FOOTNOTE_GAP: u32 = 33;
    pub const FACING_PAGES: u32 = 34;
    pub const SECTION_AUTHORING: u32 = 40;
}

/// Field numbers of `TP.SectionArchive`. Fields 1–16 are all `OBSOLETE_`.
pub mod section_field {
    pub const INHERIT_PREVIOUS_HEADER_FOOTER: u32 = 17;
    pub const FIRST_PAGE_DIFFERENT: u32 = 18;
    pub const EVEN_ODD_PAGES_DIFFERENT: u32 = 19;
    pub const START_KIND: u32 = 20;
    pub const PAGE_NUMBER_KIND: u32 = 21;
    pub const PAGE_NUMBER_START: u32 = 22;
    pub const FIRST_TEMPLATE_PAGE: u32 = 23;
    pub const EVEN_TEMPLATE_PAGE: u32 = 24;
    pub const ODD_TEMPLATE_PAGE: u32 = 25;
    pub const NAME: u32 = 26;
    pub const FIRST_PAGE_HIDES_HEADER_FOOTER: u32 = 28;
    pub const USER_DEFINED_GUIDE_STORAGE: u32 = 29;
    pub const BACKGROUND_FILL: u32 = 30;
    pub const HYPERLINK_UUID: u32 = 31;
}

/// Whether the document has a text body — Pages' two document modes.
///
/// Pages calls them "word processing" and "page layout"; the format calls it
/// `TP.SettingsArchive.body`, and the app's dictionary calls it `document
/// body`, a read-only boolean. All three agree.
///
/// A second, independent signal says the same thing, and it was found by
/// counting rather than by reading a schema: of the 640 bundled Pages
/// templates, **the 388 whose `body` is false are exactly the 388 that carry a
/// `TP.PageTemplateArchive`** — the sets are equal, with no exception either
/// way. A page-layout document has named page templates because that is what
/// its pages are made of; a word-processing document has none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mode {
    WordProcessing,
    PageLayout,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::WordProcessing => "word processing",
            Mode::PageLayout => "page layout",
        }
    }
}

/// How footnotes are gathered, as `TP.SettingsArchive` records it.
///
/// **`pages-footnotes.pages` has two footnotes**, so the containment
/// [`Structure::footnotes`] reads is exercised at last — but these four
/// *settings* are at their defaults in that document too (footnotes, numeric,
/// continuous, a 10-point gap), and at the same defaults in every other
/// document and all 640 bundled Pages templates. So the values below are
/// Confirmed and the named alternatives are still Unverified — the enumerators
/// of the 15.3.1 schema, not observations — because nothing here moves the
/// mode, the format or the restart rule off its default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FootnoteSettings {
    /// 0 footnotes, 1 document endnotes, 2 section endnotes.
    pub kind: u64,
    /// 0 numeric, 1 roman, 2 symbolic, 3 Japanese numeric, 4 Japanese
    /// ideographic, 5 Arabic numeric.
    pub format: u64,
    /// 0 continuous, 1 restart each page, 2 restart each section.
    pub numbering: u64,
    /// Space between the body and the notes, in points.
    pub gap: u64,
}

impl FootnoteSettings {
    pub fn kind_name(self) -> &'static str {
        match self.kind {
            0 => "footnotes",
            1 => "document endnotes",
            2 => "section endnotes",
            _ => "unknown",
        }
    }

    pub fn format_name(self) -> &'static str {
        match self.format {
            0 => "1, 2, 3",
            1 => "i, ii, iii",
            2 => "*, †, ‡",
            3 => "Japanese numeric",
            4 => "Japanese ideographic",
            5 => "Arabic numeric",
            _ => "unknown",
        }
    }

    pub fn numbering_name(self) -> &'static str {
        match self.numbering {
            0 => "continuous",
            1 => "restart on each page",
            2 => "restart in each section",
            _ => "unknown",
        }
    }
}

/// Paper, margins and the document-wide switches.
#[derive(Debug, Clone)]
pub struct PageSetup {
    pub width: f32,
    pub height: f32,
    pub left_margin: f32,
    pub right_margin: f32,
    pub top_margin: f32,
    pub bottom_margin: f32,
    pub header_margin: f32,
    pub footer_margin: f32,
    pub scale: f32,
    /// `TP.DocumentArchive.orientation`; 0 in every document seen.
    pub orientation: u64,
    pub paper_id: String,
    pub printer_id: String,
    /// `TP.SettingsArchive.facing_pages` — left and right pages differ.
    pub facing_pages: bool,
    /// `TP.DocumentArchive.uses_single_header_footer`.
    pub single_header_footer: bool,
    pub headers_shown: bool,
    pub footers_shown: bool,
    pub body_vertical: bool,
    pub rtl: bool,
    pub language: String,
    /// `TP.SettingsArchive.orig_template` — the template the document came from.
    pub template: String,
}

impl PageSetup {
    /// Is the page taller than it is wide?
    ///
    /// `orientation` is 0 in every document in this corpus and in all 640
    /// bundled templates, including the landscape ones, so the field does not
    /// say — the page *size* does, and Pages swaps width and height rather
    /// than setting a flag.
    pub fn portrait(&self) -> bool {
        self.height >= self.width
    }
}

/// Which of a section's three template pages a header or footer belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TemplatePage {
    /// `first_section_template_page` (23) — used when
    /// `section_template_first_page_different` is set.
    First,
    /// `even_section_template_page` (24) — the left-hand page of a spread.
    Even,
    /// `odd_section_template_page` (25) — the ordinary page.
    Odd,
}

impl TemplatePage {
    pub fn as_str(self) -> &'static str {
        match self {
            TemplatePage::First => "first",
            TemplatePage::Even => "even",
            TemplatePage::Odd => "odd",
        }
    }

    pub fn field(self) -> u32 {
        match self {
            TemplatePage::First => section_field::FIRST_TEMPLATE_PAGE,
            TemplatePage::Even => section_field::EVEN_TEMPLATE_PAGE,
            TemplatePage::Odd => section_field::ODD_TEMPLATE_PAGE,
        }
    }

    pub const ALL: [TemplatePage; 3] = [TemplatePage::First, TemplatePage::Even, TemplatePage::Odd];
}

/// Where in the three-zone strip a header or footer storage sits.
///
/// Every `TP.SectionTemplateArchive` in this corpus carries **exactly three
/// headers and exactly three footers** — 3144 of them across the 640 bundled
/// templates and 66 across this corpus, with no other shape — which is Pages'
/// three header and three footer fields.
///
/// The zone order is `left, centre, right` and is **Inferred**: nothing in the
/// archive names them, and all three zones of a strip point at the same
/// paragraph style, so alignment does not say either. The evidence is a
/// mirror pair. `08_Journal_Newsletter` puts its date in header zone 2 and its
/// page number in footer zone 2; `08_Newsletter_RTL`, the same design laid out
/// right to left, puts both in zone 0. Content that moves to the other end of
/// the strip when the design is mirrored is content addressed by *side*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Zone {
    Left,
    Centre,
    Right,
}

impl Zone {
    pub fn as_str(self) -> &'static str {
        match self {
            Zone::Left => "left",
            Zone::Centre => "centre",
            Zone::Right => "right",
        }
    }

    fn of(index: usize) -> Option<Zone> {
        match index {
            0 => Some(Zone::Left),
            1 => Some(Zone::Centre),
            2 => Some(Zone::Right),
            _ => None,
        }
    }
}

/// A page number, page count or footnote mark standing in a header, a footer
/// or the body — a `TSWP.NumberAttachmentArchive` (2043) behind a `U+FFFC`.
///
/// This is where the *format* of a page number lives. A section says whether
/// numbering continues or restarts and at what
/// ([`Section::page_number_kind`], [`Section::page_number_start`]); what the
/// number is drawn as is here, on the attachment, and a document can therefore
/// number one section in roman and another in arabic without either section
/// archive saying so.
///
/// Across the 640 bundled Pages templates there are **129 of these and every
/// one is `kind` 0, `format` 0, `"decimal"`** — so the other kinds and formats
/// are named from the 15.3.1 schema and are Unverified.
#[derive(Debug, Clone)]
pub struct NumberField {
    /// The storage it stands in.
    pub storage: u64,
    /// Character index of the `U+FFFC`.
    pub index: u64,
    /// The `TSWP.NumberAttachmentArchive`.
    pub identifier: u64,
    /// `TSWP.TextualAttachmentArchive.kind`: 0 page number, 1 page count,
    /// 2 footnote mark.
    pub kind: u64,
    /// `number_format`, 0 everywhere it has been seen.
    pub format: u64,
    /// `number_format_name` — `"decimal"` in all 129 observed.
    pub format_name: String,
}

impl NumberField {
    pub fn kind_name(&self) -> &'static str {
        match self.kind {
            0 => "page number",
            1 => "page count",
            2 => "footnote mark",
            _ => "unknown",
        }
    }
}

/// One header or footer storage, with everything needed to name it.
#[derive(Debug, Clone)]
pub struct HeaderFooter {
    /// Object identifier of the `TSWP.StorageArchive`. This is what
    /// [`crate::Document::set_text`] takes.
    pub storage: u64,
    /// Index of the section it belongs to, in document order.
    pub section: usize,
    pub section_identifier: u64,
    pub template: TemplatePage,
    pub section_template: u64,
    pub zone: Zone,
    /// Header or footer.
    pub footer: bool,
    pub text: String,
    /// The page numbers and page counts standing in it, if any. A footer whose
    /// text is a lone `U+FFFC` usually has exactly one.
    pub numbers: Vec<NumberField>,
}

impl HeaderFooter {
    pub fn kind(&self) -> &'static str {
        if self.footer {
            "footer"
        } else {
            "header"
        }
    }
}

/// One `TP.SectionArchive`, with the range of body text it covers.
#[derive(Debug, Clone)]
pub struct Section {
    pub identifier: u64,
    /// Position in document order, from 0.
    pub index: usize,
    /// Apple's name for the section — "Blank", "Chapter Opener", "Cover" …
    pub name: String,
    /// First character of the section's text, in UTF-16 code units.
    pub start: u64,
    /// One past its last character. The `U+0004` that begins the *next*
    /// section is not part of either.
    pub end: u64,
    /// Take headers and footers from the section before this one.
    pub inherits_header_footer: bool,
    pub first_page_different: bool,
    pub even_odd_different: bool,
    /// `section_start_kind` — where the section is allowed to begin. 0
    /// everywhere in this corpus; the other values are Unverified.
    pub start_kind: u64,
    /// 0 continue from the previous section, 1 start at
    /// [`Section::page_number_start`]. Both observed.
    pub page_number_kind: u64,
    /// The number to start at. The schema gives the field no default, so a
    /// section that does not carry one reads as 0 rather than as an invented
    /// 1 — though every section in this corpus and in all 640 bundled
    /// templates writes it, and 1045 of the 1048 write exactly 1.
    pub page_number_start: u64,
    /// Hide the header and footer on the section's first page.
    pub hides_header_footer_on_first_page: bool,
    /// The section paints its own background.
    pub has_background: bool,
    /// `section_hyperlink_uuid` — what a link to this section points at.
    pub hyperlink_uuid: bool,
    /// Object identifiers of the first, even and odd `TP.SectionTemplateArchive`s.
    pub templates: [Option<u64>; 3],
}

impl Section {
    /// How the page numbering of this section reads.
    pub fn numbering(&self) -> String {
        match self.page_number_kind {
            0 => "continue from the previous section".to_string(),
            1 => format!("start at {}", self.page_number_start),
            other => format!("kind {other}, start {}", self.page_number_start),
        }
    }

    pub fn length(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
}

/// One `TP.PageTemplateArchive` — a named page a page-layout document is built
/// from. Word-processing documents have none.
#[derive(Debug, Clone)]
pub struct PageTemplate {
    pub identifier: u64,
    pub name: String,
    /// `headers_footers_match_previous_page` — the "Match previous page"
    /// switch, and the only *required* field of the message.
    pub matches_previous_page: bool,
    pub hides_headers_footers: bool,
    pub has_background: bool,
    pub drawables: usize,
    /// `placeholder_drawables`, each a `{tag, drawable, z_index}`.
    pub placeholders: usize,
}

/// One linked-text-box thread: a `TSWP.FlowInfoArchive`.
///
/// A thread is **numbered, not named**: the archive's only identity is
/// `user_interface_identifier`, which is what Pages shows as "Text Box 1". The
/// storage is shared by every box in the thread, which is the case Phase 4's
/// "a storage is not one-to-one with a drawable" was written for.
#[derive(Debug, Clone)]
pub struct Thread {
    pub identifier: u64,
    /// `user_interface_identifier` — the number the app puts on the thread.
    pub number: u64,
    /// The one storage all the boxes lay out.
    pub storage: Option<u64>,
    /// The boxes, in the order the text flows through them.
    pub boxes: Vec<u64>,
}

/// One rule of a table of contents: a paragraph style, and whether a paragraph
/// in it appears.
#[derive(Debug, Clone)]
pub struct ContentsRule {
    pub paragraph_style: Option<u64>,
    pub entry_style: Option<u64>,
    pub show: bool,
}

/// A `TSWP.TOCSettingsArchive` and the entries it gathered.
#[derive(Debug, Clone)]
pub struct Contents {
    pub identifier: u64,
    pub name: String,
    /// `toc_scope`: 0 on the document's own settings, 1 on the copy a placed
    /// contents list carries. Observed in `pages-toc`, which has one of each.
    pub scope: u64,
    pub rules: Vec<ContentsRule>,
    /// The `TSWP.TOCInfoArchive` that owns this copy of the settings, when one
    /// does. The document-level settings have no owner.
    pub placed_in: Option<u64>,
    /// `(heading, page number)` for each line the last layout produced.
    pub entries: Vec<(String, u64)>,
}

/// A footnote as this crate reads one.
///
/// Observed at last: `pages-footnotes.pages`, made through Insert > Footnote
/// on an unlocked screen, carries two — `table_footnote` entries at
/// characters 26 and 77, each referencing a
/// `TSWP.FootnoteReferenceAttachmentArchive` whose contained storage is the
/// note's own text, a storage of kind 2. The schema guess held; what no
/// fixture shows yet is an endnote mode, a custom mark, or a restarting
/// numbering — those fields stay Inferred.
#[derive(Debug, Clone)]
pub struct Footnote {
    /// Storage the mark is in.
    pub storage: u64,
    /// Character index of the mark — a `U+000E` in the text, not the
    /// `U+FFFC` an ordinary attachment sits on.
    pub index: u64,
    /// The `TSWP.FootnoteReferenceAttachmentArchive`.
    pub attachment: Option<u64>,
    /// Its `contained_storage` (field 2): the note's own text, a storage of
    /// kind 2.
    pub body: Option<u64>,
    /// `custom_mark_string` (field 3), when the mark is not automatic.
    pub mark: Option<String>,
    pub text: String,
}

/// Everything this module reads about one Pages document.
#[derive(Debug, Clone)]
pub struct Structure {
    pub mode: Mode,
    pub setup: PageSetup,
    pub footnote_settings: FootnoteSettings,
    pub change_tracking: ChangeTracking,
    pub sections: Vec<Section>,
    pub header_footers: Vec<HeaderFooter>,
    pub page_templates: Vec<PageTemplate>,
    pub threads: Vec<Thread>,
    pub contents: Vec<Contents>,
    pub footnotes: Vec<Footnote>,
    /// Object identifier of the body storage, when the document has one.
    pub body_storage: Option<u64>,
    /// Bookmark anchors — `table_bookmark` (field 15) entries, as
    /// `(storage, index, TSWP.BookmarkFieldArchive)`. Terminator entries (no
    /// reference — where a bookmark's run ends) are not listed.
    ///
    /// Observed at last: `pages-bookmarks.pages`, made through Insert >
    /// Bookmark on an unlocked screen, carries two archives of shape
    /// `{1.1: UUID string, 3: varint, 4: varint}` — a bookmark stores no
    /// name of its own.
    pub bookmarks: Vec<(u64, u64, Option<u64>)>,
}

/// Read a float field, whatever width it was written at.
fn float(message: &Message, number: u32) -> f32 {
    match message.get(number) {
        Some(Value::Fixed32(bytes)) => f32::from_le_bytes(*bytes),
        Some(Value::Fixed64(bytes)) => f64::from_le_bytes(*bytes) as f32,
        _ => 0.0,
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

/// Every reference held in a repeated field, in order.
fn references(message: &Message, number: u32) -> Vec<u64> {
    message
        .all(number)
        .filter_map(|value| match value {
            Value::Bytes(raw) => decode_nested(raw).and_then(|r| reference_target(&r)),
            _ => None,
        })
        .collect()
}

/// Every reference in a repeated field, each paired with **its index in the
/// full field** — not its position among the ones that decoded.
///
/// The plain [`references`] filter drops any entry that is not a bare
/// reference, so a header strip with a nil or unrecognised middle entry would
/// slide the entries after it one slot to the left: the right-hand storage
/// would come back labelled Centre. Keeping each survivor's true index labels
/// zones by where they actually sit, and an entry past the third simply has no
/// zone rather than displacing the ones before it.
fn indexed_references(message: &Message, number: u32) -> Vec<(usize, u64)> {
    message
        .all(number)
        .enumerate()
        .filter_map(|(index, value)| match value {
            Value::Bytes(raw) => decode_nested(raw)
                .and_then(|r| reference_target(&r))
                .map(|target| (index, target)),
            _ => None,
        })
        .collect()
}

/// Read the whole `TP` structure of a document.
///
/// Returns `None` for a document that is not a Pages one — the caller has
/// [`crate::Document::kind`] to check first, but a Numbers file handed here
/// simply has no `TP.DocumentArchive` and says so.
pub fn structure(document: &crate::Document) -> Option<Structure> {
    let mut archives: BTreeMap<u64, (u32, Message)> = BTreeMap::new();
    for (_, object) in document.objects() {
        if let Ok(message) = Message::decode(object.payload()) {
            archives.insert(object.identifier, (object.message_type(), message));
        }
    }

    let (_, root) = archives
        .values()
        .find(|(message_type, _)| *message_type == TYPE_DOCUMENT)
        .cloned()?;

    let settings = reference(&root, document_field::SETTINGS)
        .and_then(|id| archives.get(&id))
        .map(|(_, m)| m.clone())
        .unwrap_or_default();

    let mode = if flag(&settings, settings_field::BODY, true) {
        Mode::WordProcessing
    } else {
        Mode::PageLayout
    };

    let setup = PageSetup {
        width: float(&root, document_field::PAGE_WIDTH),
        height: float(&root, document_field::PAGE_HEIGHT),
        left_margin: float(&root, document_field::LEFT_MARGIN),
        right_margin: float(&root, document_field::RIGHT_MARGIN),
        top_margin: float(&root, document_field::TOP_MARGIN),
        bottom_margin: float(&root, document_field::BOTTOM_MARGIN),
        header_margin: float(&root, document_field::HEADER_MARGIN),
        footer_margin: float(&root, document_field::FOOTER_MARGIN),
        // A document with no `page_scale` is at 100%, not at 0%.
        scale: match root.get(document_field::PAGE_SCALE) {
            Some(_) => float(&root, document_field::PAGE_SCALE),
            None => 1.0,
        },
        orientation: root.varint(document_field::ORIENTATION).unwrap_or(0),
        paper_id: text(&root, document_field::PAPER_ID),
        printer_id: text(&root, document_field::PRINTER_ID),
        facing_pages: flag(&settings, settings_field::FACING_PAGES, false),
        single_header_footer: flag(&root, document_field::USES_SINGLE_HEADER_FOOTER, false),
        headers_shown: flag(&settings, settings_field::HEADERS, true),
        footers_shown: flag(&settings, settings_field::FOOTERS, true),
        body_vertical: flag(&root, document_field::LAYS_OUT_BODY_VERTICALLY, false),
        rtl: flag(&settings, settings_field::DOCUMENT_IS_RTL, false),
        language: text(&settings, settings_field::LANGUAGE),
        template: text(&settings, settings_field::ORIG_TEMPLATE),
    };

    let footnote_settings = FootnoteSettings {
        kind: settings.varint(settings_field::FOOTNOTE_KIND).unwrap_or(0),
        format: settings
            .varint(settings_field::FOOTNOTE_FORMAT)
            .unwrap_or(0),
        numbering: settings
            .varint(settings_field::FOOTNOTE_NUMBERING)
            .unwrap_or(0),
        gap: settings.varint(settings_field::FOOTNOTE_GAP).unwrap_or(0),
    };

    let change_tracking = ChangeTracking {
        enabled: flag(&root, document_field::CHANGE_TRACKING_ENABLED, false),
        sessions: root
            .fields
            .iter()
            .filter(|f| f.number == document_field::CHANGE_SESSIONS)
            .count(),
        has_recent_session: root
            .get(document_field::MOST_RECENT_CHANGE_SESSION)
            .is_some(),
        show_markup: flag(&settings, settings_field::SHOW_CT_MARKUP, true),
        show_deletions: flag(&settings, settings_field::SHOW_CT_DELETIONS, true),
        bubbles_visibility: settings
            .varint(settings_field::CT_BUBBLES_VISIBILITY)
            .unwrap_or(0),
        change_bars: flag(&settings, settings_field::CHANGE_BARS_VISIBLE, true),
        format_changes: flag(&settings, settings_field::FORMAT_CHANGES_VISIBLE, true),
        annotations_visible: flag(&settings, settings_field::ANNOTATIONS_VISIBLE, true),
    };

    let body_storage = reference(&root, document_field::BODY_STORAGE);
    let body_text = body_storage
        .and_then(|id| document.storage_text(id).ok())
        .unwrap_or_default();
    let body_length = crate::text::length(&body_text);

    // -- sections, from the body storage's section table ---------------------
    let mut starts: Vec<(u64, u64)> = Vec::new();
    if let Some(storage) = body_storage.and_then(|id| archives.get(&id)) {
        if let Some(table) = storage
            .1
            .bytes(crate::text::SECTION_TABLE)
            .and_then(decode_nested)
        {
            starts = crate::text::entry_indices(&table, crate::text::Anchoring::Paragraph)
                .into_iter()
                .filter_map(|(index, object)| object.map(|o| (index, o)))
                .collect();
        }
    }
    // Last resort, and **never taken**: over all 640 bundled Pages templates
    // the number of sections reported here equals the number of
    // `TP.SectionArchive` objects exactly, and no two of them start at 0 — so
    // every document, page-layout ones included, has a body storage with a
    // complete section table. `TP.DocumentArchive.section` (5) exists in the
    // schema and is absent from every document seen. If a document ever turns
    // up without the table, the archives are still listed, in object order,
    // and their ranges read as empty rather than as a guess.
    //
    // An empty range is a visible non-claim — `iwork sections` prints
    // "0 unit(s)" — which is the point; the alternative is a range invented
    // from nothing.
    if starts.is_empty() {
        let mut loose: Vec<u64> = archives
            .iter()
            .filter(|(_, (message_type, _))| *message_type == TYPE_SECTION)
            .map(|(id, _)| *id)
            .collect();
        loose.sort_unstable();
        starts = loose.into_iter().map(|id| (0, id)).collect();
    }

    let mut sections = Vec::new();
    for (index, (start, identifier)) in starts.iter().enumerate() {
        let Some((_, archive)) = archives.get(identifier) else {
            continue;
        };
        // The break character belongs to no section's text, so a section stops
        // one short of the next one's start.
        let end = match starts.get(index + 1) {
            Some((next, _)) => next.saturating_sub(1),
            None => body_length,
        };
        let mut templates = [None; 3];
        for (slot, page) in TemplatePage::ALL.iter().enumerate() {
            templates[slot] = reference(archive, page.field());
        }
        sections.push(Section {
            identifier: *identifier,
            index,
            name: text(archive, section_field::NAME),
            start: *start,
            end: end.max(*start),
            inherits_header_footer: flag(
                archive,
                section_field::INHERIT_PREVIOUS_HEADER_FOOTER,
                false,
            ),
            first_page_different: flag(archive, section_field::FIRST_PAGE_DIFFERENT, false),
            even_odd_different: flag(archive, section_field::EVEN_ODD_PAGES_DIFFERENT, false),
            start_kind: archive.varint(section_field::START_KIND).unwrap_or(0),
            page_number_kind: archive.varint(section_field::PAGE_NUMBER_KIND).unwrap_or(0),
            page_number_start: archive
                .varint(section_field::PAGE_NUMBER_START)
                .unwrap_or(0),
            hides_header_footer_on_first_page: flag(
                archive,
                section_field::FIRST_PAGE_HIDES_HEADER_FOOTER,
                false,
            ),
            has_background: archive.get(section_field::BACKGROUND_FILL).is_some(),
            hyperlink_uuid: archive.get(section_field::HYPERLINK_UUID).is_some(),
            templates,
        });
    }

    // -- headers and footers -------------------------------------------------
    let mut header_footers = Vec::new();
    for section in &sections {
        for (slot, page) in TemplatePage::ALL.iter().enumerate() {
            let Some(template) = section.templates[slot] else {
                continue;
            };
            let Some((_, archive)) = archives.get(&template) else {
                continue;
            };
            for (field, footer) in [(1u32, false), (2u32, true)] {
                for (position, storage) in indexed_references(archive, field) {
                    let Some(zone) = Zone::of(position) else {
                        continue;
                    };
                    header_footers.push(HeaderFooter {
                        storage,
                        section: section.index,
                        section_identifier: section.identifier,
                        template: *page,
                        section_template: template,
                        zone,
                        footer,
                        text: document.storage_text(storage).unwrap_or_default(),
                        numbers: number_fields(&archives, storage),
                    });
                }
            }
        }
    }

    // -- page templates ------------------------------------------------------
    let mut page_templates = Vec::new();
    for identifier in references(&root, document_field::PAGE_TEMPLATES) {
        let Some((_, archive)) = archives.get(&identifier) else {
            continue;
        };
        page_templates.push(PageTemplate {
            identifier,
            name: text(archive, 1),
            matches_previous_page: flag(archive, 4, false),
            hides_headers_footers: flag(archive, 5, false),
            has_background: archive.get(6).is_some(),
            drawables: archive.all(2).count(),
            placeholders: archive.all(3).count(),
        });
    }

    // -- linked text boxes ---------------------------------------------------
    let mut threads = Vec::new();
    let flow_identifiers = match reference(&root, document_field::FLOW_INFO_CONTAINER)
        .and_then(|id| archives.get(&id))
    {
        Some((_, container)) => references(container, 1),
        None => Vec::new(),
    };
    for identifier in flow_identifiers {
        let Some((_, archive)) = archives.get(&identifier) else {
            continue;
        };
        threads.push(Thread {
            identifier,
            number: archive.varint(3).unwrap_or(0),
            storage: reference(archive, 1),
            boxes: references(archive, 2),
        });
    }

    // -- tables of contents --------------------------------------------------
    //
    // Two places hold a `TSWP.TOCSettingsArchive`: the document's `toc_styles`
    // list, which is the style-inclusion map the whole document is judged by,
    // and each placed contents list's own copy. `pages-toc` has one of each and
    // they disagree — the document's names two styles and the placed list's
    // names six — so reading only one of them is reading the wrong one.
    let mut owner_of: BTreeMap<u64, u64> = BTreeMap::new();
    let mut instances: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for (identifier, (message_type, archive)) in &archives {
        if *message_type != TYPE_TOC_INFO {
            continue;
        }
        if let Some(settings) = reference(archive, 2) {
            owner_of.insert(settings, *identifier);
            instances.insert(settings, references(archive, 3));
        }
    }
    let mut contents = Vec::new();
    for (identifier, (message_type, archive)) in &archives {
        if *message_type != TYPE_TOC_SETTINGS {
            continue;
        }
        let rules = archive
            .all(3)
            .filter_map(|value| match value {
                Value::Bytes(raw) => decode_nested(raw),
                _ => None,
            })
            .map(|entry| ContentsRule {
                paragraph_style: reference(&entry, 1),
                entry_style: reference(&entry, 2),
                show: flag(&entry, 3, false),
            })
            .collect();
        let entries = instances
            .get(identifier)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| archives.get(id))
                    .map(|(_, entry)| (text(entry, 4), entry.varint(2).unwrap_or(0)))
                    .collect()
            })
            .unwrap_or_default();
        contents.push(Contents {
            identifier: *identifier,
            name: text(archive, 1),
            scope: archive.varint(2).unwrap_or(0),
            rules,
            placed_in: owner_of.get(identifier).copied(),
            entries,
        });
    }
    contents.sort_by_key(|c| (c.placed_in.is_some(), c.identifier));

    // -- footnotes and bookmarks, wherever they are ---------------------------
    let mut footnotes = Vec::new();
    let mut bookmarks = Vec::new();
    for (identifier, (message_type, archive)) in &archives {
        if *message_type != crate::TYPE_STORAGE {
            continue;
        }
        if let Some(table) = archive
            .bytes(crate::text::FOOTNOTE_TABLE)
            .and_then(decode_nested)
        {
            for (index, attachment) in
                crate::text::entry_indices(&table, crate::text::Anchoring::Character)
            {
                let mark_archive = attachment.and_then(|id| archives.get(&id)).map(|(_, m)| m);
                let body = mark_archive.and_then(|m| reference(m, 2));
                footnotes.push(Footnote {
                    storage: *identifier,
                    index,
                    attachment,
                    body,
                    mark: mark_archive
                        .and_then(|m| m.bytes(3))
                        .map(|raw| String::from_utf8_lossy(raw).into_owned()),
                    text: body
                        .and_then(|id| document.storage_text(id).ok())
                        .unwrap_or_default(),
                });
            }
        }
        if let Some(table) = archive
            .bytes(crate::text::BOOKMARK_TABLE)
            .and_then(decode_nested)
        {
            for (index, object) in crate::text::entry_indices(&table, crate::text::Anchoring::Run) {
                // An entry with no reference ends the run before it — it is
                // where a bookmark *stops*, not a bookmark. `pages-bookmarks`
                // has two `TSWP.BookmarkFieldArchive`s and a terminator, and
                // counting the terminator reported three.
                if object.is_some() {
                    bookmarks.push((*identifier, index, object));
                }
            }
        }
    }
    footnotes.sort_by_key(|f| (f.storage, f.index));
    bookmarks.sort();

    Some(Structure {
        mode,
        setup,
        footnote_settings,
        change_tracking,
        sections,
        header_footers,
        page_templates,
        threads,
        contents,
        footnotes,
        body_storage,
        bookmarks,
    })
}

/// The page numbers and page counts standing in one storage.
///
/// They are `table_attachment` (field 9) entries like an anchored image, told
/// apart by what they point at: a `TSWP.NumberAttachmentArchive` rather than a
/// `TSWP.DrawableAttachmentArchive`.
fn number_fields(archives: &BTreeMap<u64, (u32, Message)>, storage: u64) -> Vec<NumberField> {
    let Some((_, archive)) = archives.get(&storage) else {
        return Vec::new();
    };
    let Some(table) = archive.bytes(9).and_then(decode_nested) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (index, object) in crate::text::entry_indices(&table, crate::text::Anchoring::Character) {
        let Some(identifier) = object else { continue };
        let Some((message_type, attachment)) = archives.get(&identifier) else {
            continue;
        };
        if *message_type != TYPE_NUMBER_ATTACHMENT {
            continue;
        }
        out.push(NumberField {
            storage,
            index,
            identifier,
            kind: attachment
                .bytes(1)
                .and_then(decode_nested)
                .and_then(|super_| super_.varint(2))
                .unwrap_or(0),
            format: attachment.varint(2).unwrap_or(0),
            format_name: text(attachment, 4),
        });
    }
    out
}

/// One paragraph range and the column layout in force over it.
///
/// Columns are not a property of a section: they are a
/// `TSWP.ColumnStyleArchive` reached from the body storage's
/// `table_layout_style` (field 12), which is anchored per paragraph. A
/// document whose sections have different column counts has several entries in
/// that one table, and the ranges are what say where each applies. In this
/// corpus only the *body* storage carries the table — a text box in a thread
/// has none.
///
/// **Widths and gaps are fractions of the text width, not points.** The one
/// non-equal layout in the whole install, in `02_ResearchPaper_JP`, reads
/// `first 0.26090077`, `gap 0.035152942`, `width 0.7039463` — which sum to
/// exactly 1.0 — and its equal-column neighbour has `gap 0.03527747` on a page
/// whose text is 515 points wide. A gap of three hundredths of a point would
/// be no gap at all.
#[derive(Debug, Clone)]
pub struct ColumnLayout {
    pub storage: u64,
    /// First character the layout applies to.
    pub start: u64,
    /// One past the last — the next entry's start, or the end of the text.
    pub end: u64,
    pub style: Option<u64>,
    /// `(count, gap)` when the columns are equal. The gap is a fraction of the
    /// text width.
    pub equal: Option<(u64, f32)>,
    /// `(first width, [(gap, width) …])` when the columns are not equal, all
    /// of them fractions of the text width.
    pub unequal: Option<(f32, Vec<(f32, f32)>)>,
    /// `columns_null` — the style deliberately asserts no columns.
    pub none: bool,
}

impl ColumnLayout {
    /// How many columns the text is set in.
    pub fn count(&self) -> u64 {
        if let Some((count, _)) = self.equal {
            return count;
        }
        match &self.unequal {
            Some((_, following)) => following.len() as u64 + 1,
            None => 1,
        }
    }

    /// The column and gap fractions in order, left to right, which should add
    /// up to the whole text width.
    pub fn fractions(&self) -> Vec<f32> {
        match (&self.equal, &self.unequal) {
            (Some((count, gap)), _) if *count > 0 => {
                let gaps = count.saturating_sub(1) as f32;
                let width = (1.0 - gap * gaps) / *count as f32;
                let mut out = vec![width];
                for _ in 1..*count {
                    out.push(*gap);
                    out.push(width);
                }
                out
            }
            (_, Some((first, following))) => {
                let mut out = vec![*first];
                for (gap, width) in following {
                    out.push(*gap);
                    out.push(*width);
                }
                out
            }
            _ => vec![1.0],
        }
    }
}

/// Column layouts of one storage, in order.
pub fn column_layouts(document: &crate::Document, storage: u64) -> Vec<ColumnLayout> {
    let Ok(archive) = document.archive(storage) else {
        return Vec::new();
    };
    let Some(table) = archive
        .bytes(crate::text::LAYOUT_TABLE)
        .and_then(decode_nested)
    else {
        return Vec::new();
    };
    let length = crate::text::length(&document.storage_text(storage).unwrap_or_default());
    let entries = crate::text::entry_indices(&table, crate::text::Anchoring::Paragraph);
    let mut out = Vec::new();
    for (position, (start, style)) in entries.iter().enumerate() {
        let end = entries
            .get(position + 1)
            .map(|(next, _)| *next)
            .unwrap_or(length);
        let properties = style
            .and_then(|id| document.archive(id).ok())
            .and_then(|s| s.bytes(11).and_then(decode_nested));
        let columns = properties
            .as_ref()
            .and_then(|p| p.bytes(7).and_then(decode_nested));
        let equal = columns
            .as_ref()
            .and_then(|c| c.bytes(1).and_then(decode_nested))
            .map(|e| (e.varint(1).unwrap_or(1), float(&e, 2)));
        let unequal = columns
            .as_ref()
            .and_then(|c| c.bytes(2).and_then(decode_nested))
            .map(|u| {
                let following = u
                    .all(2)
                    .filter_map(|value| match value {
                        Value::Bytes(raw) => decode_nested(raw),
                        _ => None,
                    })
                    .map(|pair| (float(&pair, 1), float(&pair, 2)))
                    .collect();
                (float(&u, 1), following)
            });
        out.push(ColumnLayout {
            storage,
            start: *start,
            end: end.max(*start),
            style: *style,
            equal,
            unequal,
            none: properties
                .as_ref()
                .map(|p| flag(p, 6, false))
                .unwrap_or(false),
        });
    }
    out
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

    #[test]
    fn a_zone_is_one_of_three() {
        assert_eq!(Zone::of(0), Some(Zone::Left));
        assert_eq!(Zone::of(1), Some(Zone::Centre));
        assert_eq!(Zone::of(2), Some(Zone::Right));
        assert_eq!(Zone::of(3), None, "no template has a fourth");
    }

    #[test]
    fn the_three_template_pages_are_fields_23_to_25() {
        let fields: Vec<u32> = TemplatePage::ALL.iter().map(|p| p.field()).collect();
        assert_eq!(fields, vec![23, 24, 25]);
    }

    /// The numbering a section reports is the one the app's inspector shows.
    #[test]
    fn page_numbering_reads_as_the_inspector_says_it() {
        let mut section = Section {
            identifier: 1,
            index: 0,
            name: "Blank".into(),
            start: 0,
            end: 10,
            inherits_header_footer: true,
            first_page_different: false,
            even_odd_different: false,
            start_kind: 0,
            page_number_kind: 0,
            page_number_start: 1,
            hides_header_footer_on_first_page: false,
            has_background: false,
            hyperlink_uuid: false,
            templates: [None; 3],
        };
        assert_eq!(section.numbering(), "continue from the previous section");
        section.page_number_kind = 1;
        section.page_number_start = 2;
        assert_eq!(section.numbering(), "start at 2");
    }

    /// Landscape is a page wider than it is tall; `orientation` is 0 even on
    /// the landscape templates, so it cannot be the answer.
    #[test]
    fn orientation_comes_from_the_page_size() {
        let mut setup = PageSetup {
            width: 595.0,
            height: 842.0,
            left_margin: 0.0,
            right_margin: 0.0,
            top_margin: 0.0,
            bottom_margin: 0.0,
            header_margin: 0.0,
            footer_margin: 0.0,
            scale: 1.0,
            orientation: 0,
            paper_id: "iso-a4".into(),
            printer_id: String::new(),
            facing_pages: false,
            single_header_footer: false,
            headers_shown: true,
            footers_shown: true,
            body_vertical: false,
            rtl: false,
            language: "en".into(),
            template: String::new(),
        };
        assert!(setup.portrait());
        std::mem::swap(&mut setup.width, &mut setup.height);
        assert!(!setup.portrait());
    }

    #[test]
    fn a_column_count_comes_from_whichever_shape_is_there() {
        let mut layout = ColumnLayout {
            storage: 1,
            start: 0,
            end: 10,
            style: None,
            equal: None,
            unequal: None,
            none: false,
        };
        assert_eq!(layout.count(), 1, "no columns archive is one column");
        layout.equal = Some((3, 12.0));
        assert_eq!(layout.count(), 3);
        layout.equal = None;
        layout.unequal = Some((100.0, vec![(10.0, 90.0), (10.0, 90.0)]));
        assert_eq!(layout.count(), 3);
    }

    #[test]
    fn a_missing_float_reads_as_zero_and_a_missing_flag_as_its_default() {
        let empty = message(vec![]);
        assert_eq!(float(&empty, 30), 0.0);
        assert!(flag(&empty, 1, true));
        assert!(!flag(&empty, 1, false));
        let set = message(vec![(1, Value::Varint(0))]);
        assert!(
            !flag(&set, 1, true),
            "an explicit false wins over a default"
        );
    }
}
