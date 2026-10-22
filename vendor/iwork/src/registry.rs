//! Message-type registry.
//!
//! Every object payload in an iWork document is an unlabelled protobuf message;
//! the only thing identifying its schema is `MessageInfo.type`, an index into a
//! registry that lives inside the iWork binaries and is not published.
//!
//! This table was built by observation, so each entry carries how much it can
//! be trusted. Nothing in this crate *depends* on the table — it is used for
//! human-readable output only. Parsing and rewriting work at the wire level and
//! are unaffected by a wrong or missing name.

/// How much a registry entry is worth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    /// The payload was decoded and behaves as the name says — e.g. type 2001
    /// really does hold the UTF-8 text of the document.
    Confirmed,
    /// Name inferred from position in the object graph, payload shape and the
    /// framework's number range. Plausible, not proven.
    Inferred,
    /// Carried over from public prior art (numbers-parser, keynote-parser and
    /// friends) without a local sample to check it against.
    Unverified,
}

/// Which app an entry applies to.
///
/// Most message types mean the same thing everywhere — the frameworks are
/// shared, and `TSWP.StorageArchive` is 2001 in all three apps. The app-level
/// archives are not: **Numbers and Keynote both number their document archive
/// 1**, so a table keyed on the number alone has to be wrong for one of them.
/// A Keynote deck was reported as holding a `TN.DocumentArchive` until this
/// existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum App {
    /// The message type means the same thing in all three apps.
    Any,
    Pages,
    Numbers,
    Keynote,
}

impl App {
    /// The app-specific entry a document of this kind should prefer.
    fn of(kind: Kind) -> Option<App> {
        match kind {
            Kind::Pages => Some(App::Pages),
            Kind::Numbers => Some(App::Numbers),
            Kind::Keynote => Some(App::Keynote),
            Kind::Unknown => None,
        }
    }
}

pub struct Entry {
    pub message_type: u32,
    pub name: &'static str,
    pub confidence: Confidence,
    /// App this entry is for. Entries with a specific app are only chosen for
    /// documents of that kind.
    pub app: App,
}

use crate::document::Kind;
use App::{Keynote as InKeynote, Numbers as InNumbers, Pages as InPages};
use Confidence::*;

/// Framework prefixes, by number range.
///
/// Rebuilt from the type registries carved out of the installed 15.3.1 binaries
/// — three dumps, one per app, agreeing on every range. The old table had two
/// ranges wrong: **5000–5999 is `TSCH`, not `TSS`** (every document carries the
/// theme's six chart-style presets, and they were being reported as
/// stylesheets), and 1000–1999 is not assigned at all. `TSS` is 400–499 and
/// `TSA` 600–699.
pub fn framework(message_type: u32) -> &'static str {
    match message_type {
        1..=199 => "app",
        200..=399 => "TSK",
        400..=599 => "TSS",
        600..=999 => "TSA",
        2000..=2999 => "TSWP",
        3000..=3999 => "TSD",
        4000..=4999 => "TSCE",
        5000..=5999 => "TSCH",
        6000..=6999 => "TST",
        10000..=10999 => "app",
        11000..=11999 => "TSP package",
        12000..=12999 => "app",
        _ => "?",
    }
}

const ENTRIES: &[Entry] = &[
    // -- package level -------------------------------------------------------
    // Every custom cell format in the document, in one archive. Every document
    // has it and it is empty in most of them; the one in `numbers-rules.numbers`
    // holds the template's "Millions" format, `#,###.##M`.
    Entry {
        message_type: 222,
        name: "TSK.CustomFormatListArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 11006,
        name: "TSP.PackageMetadata",
        confidence: Confirmed,
        app: App::Any,
    },
    // **Not the annotation authors this table used to call them.** 11014/11015
    // were carried over from prior art as `TSP.AnnotationAuthorArchive` and its
    // storage, and every document in the corpus appeared to have three authors
    // and a list of them. It has neither. The 15.3.1 registry names them
    // `TSP.DataMetadata` and `TSP.DataMetadataMap`, and the payloads agree
    // exactly: the map is `repeated {data_identifier, data_metadata}` whose
    // identifiers are the media ids of the *theme assets* — the images a
    // document names but does not carry — and each metadata is one
    // `fallback_color`, the colour drawn where the asset is not there.
    // `pages-report` maps 11, 10 and 14, which are its three unstored assets.
    //
    // The real annotation authors are 212/213, below, and there are none.
    Entry {
        message_type: 11014,
        name: "TSP.DataMetadata",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 11015,
        name: "TSP.DataMetadataMap",
        confidence: Confirmed,
        app: App::Any,
    },
    // -- annotations ---------------------------------------------------------
    // The author of a comment or of a tracked change: a name, a colour and the
    // collaboration identity it belongs to. For a long time nothing in the
    // corpus had one; `pages-comments.pages` (made through the UI on an
    // unlocked screen) finally carries a real author, name and colour exactly
    // where the schema said.
    Entry {
        message_type: 212,
        name: "TSK.AnnotationAuthorArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // Every document carries exactly one, in `Index/AnnotationAuthorStorage*`,
    // and in all 924 it is an **empty payload**: zero authors. That the object
    // exists is Confirmed; that field 1 lists authors is the schema's word.
    Entry {
        message_type: 213,
        name: "TSK.AnnotationAuthorStorageArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // A comment's text, date, author and replies. Hangs off
    // `TSD.DrawableArchive.comment` (field 6) for an object comment, off a
    // `TSWP.CommentInfoArchive` for one in text, and off a
    // `TST.CommentStorageWrapperArchive` for one on a cell.
    Entry {
        message_type: 3056,
        name: "TSD.CommentStorageArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // The floating shape a text comment is drawn as — a `TSWP.ShapeInfoArchive`
    // with a comment storage hung off it.
    Entry {
        message_type: 2014,
        name: "TSWP.CommentInfoArchive",
        confidence: Unverified,
        app: App::Any,
    },
    // The anchor a comment puts into text: `table_highlight` (23) and
    // `table_overlapping_highlight` (25) both point at one.
    Entry {
        message_type: 2013,
        name: "TSWP.HighlightArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // -- change tracking -----------------------------------------------------
    // One insertion or one deletion. `table_insertion` (21) and
    // `table_deletion` (22) point at these. The kind has **no zero value** —
    // insertion is 1, deletion 2 — so an all-default archive is invalid rather
    // than an insertion.
    Entry {
        message_type: 2060,
        name: "TSWP.ChangeArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // The author-and-time an editing session belongs to;
    // `TP.DocumentArchive.change_sessions` (16) lists them.
    Entry {
        message_type: 2062,
        name: "TSWP.ChangeSessionArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // A pre-collaboration change author, kept for documents old enough to have
    // one. Its Objective-C class is `TSKAnnotationAuthor`, the same class 212
    // unarchives to.
    Entry {
        message_type: 2061,
        name: "TSK.DeprecatedChangeAuthorArchive",
        confidence: Unverified,
        app: App::Any,
    },
    // -- word processing -----------------------------------------------------
    Entry {
        message_type: 2001,
        name: "TSWP.StorageArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // The class most "shapes" really are: a `TSD.ShapeArchive` at field 1 plus
    // the text storage a shape can hold at field 2. Confirmed by making Keynote
    // create a shape, a text item and a line, all three of which came back as
    // this type. `TSWP.SelectionArchive`, which this entry used to name, is
    // 2002.
    Entry {
        message_type: 2011,
        name: "TSWP.ShapeInfoArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // 2021 and 2022 are the other way round from what public prior art says.
    // Settled by asking the styles: across six documents, every type 2021 has an
    // internal identifier of the form `character-style-…` (12 of them) and every
    // type 2022 `…-paragraphstyle-…` (229). A type 2022 also carries paragraph
    // properties — alignment, indents, tab stops — which a character style has
    // no use for.
    Entry {
        message_type: 2021,
        name: "TSWP.CharacterStyleArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 2022,
        name: "TSWP.ParagraphStyleArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 2023,
        name: "TSWP.ListStyleArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 2003,
        name: "TSWP.DrawableAttachmentArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 2024,
        name: "TSWP.ColumnStyleArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 2025,
        name: "TSWP.ShapeStyleArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 2026,
        name: "TSWP.TOCEntryStyleArchive",
        confidence: Unverified,
        app: App::Any,
    },
    // The table of contents, in four pieces. 2051 is the style-inclusion map —
    // "which paragraph styles become entries" — and a document has one of its
    // own plus one per placed list; `pages-toc` has both and they disagree,
    // the document's naming two styles and the list's six. 2240 is the placed
    // list, which is a drawable. 2052 is one line of it as last laid out,
    // carrying the heading text and the page number. Only two of the 640
    // bundled Pages templates have a 2240 at all.
    Entry {
        message_type: 2051,
        name: "TSWP.TOCSettingsArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 2052,
        name: "TSWP.TOCEntryInstanceArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 2240,
        name: "TSWP.TOCInfoArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // Linked text boxes. 2410 is one thread — a storage and the boxes it flows
    // through, in order — and 2411 is the document's list of them. Nineteen of
    // the 640 Pages templates have a thread; every one of them has the
    // container, empty or not.
    Entry {
        message_type: 2410,
        name: "TSWP.FlowInfoArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 2411,
        name: "TSWP.FlowInfoContainerArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // The footnote mark in the text, and the anchor half of a bookmark.
    // **Neither exists anywhere that can be reached from here.** All 901
    // templates the three apps ship were scanned for both, and for a storage of
    // kind 2: zero, zero and zero. Both names come from the 15.3.1 schema and
    // nothing has decoded one.
    Entry {
        message_type: 2008,
        name: "TSWP.FootnoteReferenceAttachmentArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 2035,
        name: "TSWP.BookmarkFieldArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // The smart fields, all of which are anchored from `StorageArchive` field
    // 11 and all of which wrap a `TSWP.SmartFieldArchive` carrying a UUID.
    // 2031 is the placeholder — the commonest by far, 9335 of them across the
    // installed templates, and what every "Company Name" in a Pages template
    // is. 2032 is the hyperlink and its field 2 is the URL: three Numbers
    // templates carry five between them, and none of the 640 Pages templates
    // or 182 Keynote themes has one at all.
    Entry {
        message_type: 2031,
        name: "TSWP.PlaceholderSmartFieldArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 2032,
        name: "TSWP.HyperlinkFieldArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 2034,
        name: "TSWP.DateTimeSmartFieldArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 2036,
        name: "TSWP.MergeSmartFieldArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // A page number, a page count or a footnote mark, behind a `U+FFFC` in an
    // ordinary attachment table. **This is where a page number's format
    // lives** — not on the section, which only says whether numbering
    // continues or restarts. 129 of them across the 640 bundled Pages
    // templates and every one is `kind` 0, `number_format` 0, `"decimal"`.
    Entry {
        message_type: 2043,
        name: "TSWP.NumberAttachmentArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // -- drawables -----------------------------------------------------------
    //
    // `TSD` ids live in the *common* registry: the same number means the same
    // thing in all three apps, which the three 15.3.1 registry dumps confirm
    // entry for entry. Two of the five entries this block used to hold were
    // wrong — 3016 is the media style, not a theme, and 3047 is the guide
    // storage — and both were Inferred, which is what Inferred is for.
    Entry {
        message_type: 3002,
        name: "TSD.DrawableArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 3003,
        name: "TSD.ContainerArchive",
        confidence: Unverified,
        app: App::Any,
    },
    // Every shape in the corpus is a `TSWP.ShapeInfoArchive` (2011) wrapping
    // one of these, rather than one of these on its own.
    Entry {
        message_type: 3004,
        name: "TSD.ShapeArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 3005,
        name: "TSD.ImageArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // The window a masked image is seen through: its own geometry in the
    // image's coordinate space plus a path source. Confirmed by the frames
    // Pages reports for a cropped photo.
    Entry {
        message_type: 3006,
        name: "TSD.MaskArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // Read from the two live-video placeholders a Keynote theme ships: poster
    // image data at 15, style at 19, `is_live_video` at 30 and the Keynote-only
    // `KN.LiveVideoInfo` extension at 100.
    Entry {
        message_type: 3007,
        name: "TSD.MovieArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 3008,
        name: "TSD.GroupArchive",
        confidence: Unverified,
        app: App::Any,
    },
    Entry {
        message_type: 3009,
        name: "TSD.ConnectionLineArchive",
        confidence: Unverified,
        app: App::Any,
    },
    Entry {
        message_type: 3015,
        name: "TSD.ShapeStyleArchive",
        confidence: Unverified,
        app: App::Any,
    },
    // Confirmed by what it holds — stroke, opacity, shadow and reflection, in
    // *that* numbering, one field lower than a shape style because media has no
    // fill. Thirteen of them in every document.
    Entry {
        message_type: 3016,
        name: "TSD.MediaStyleArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 3045,
        name: "TSD.CanvasSelectionArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 3047,
        name: "TSD.GuideStorageArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 3061,
        name: "TSD.DrawableSelectionArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 3091,
        name: "TSD.FreehandDrawingToolkitUIState",
        confidence: Inferred,
        app: App::Any,
    },
    // An empty message — zero payload bytes — meaning "a caption could go here
    // and none has been written". A Keynote deck carries 178 of them.
    Entry {
        message_type: 3097,
        name: "TSD.StandinCaptionArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // -- charts --------------------------------------------------------------
    // The 5000s are `TSCH`, not `TSS`. Every document in the corpus carries the
    // theme's chart-style presets — six of 5020, six of 5022, six of 5024,
    // eighteen of 5026 (three axes each) and thirty-six of 5028 (six series
    // each) — and they were being reported as stylesheets and themes.
    //
    // **The whole block is a two-field sandwich.** Every archive from 5021 down
    // to 5031 is `{1: super, 10000: the real message}`, and the schema of field
    // 10000 is decided by the *shell's* type id and by nothing in the bytes —
    // which is why they are named here even where nothing decodes them. 5021's
    // extension is `TSCH.ChartArchive`, a message with **no type id of its
    // own**; the ten style shells' extensions are the matching
    // `TSCH.Generated.*Archive`s.
    //
    // Counts across the corpus: 33 of 5021, 138 of 5020, 140 of 5022, 33 of
    // 5023, 140 of 5024, 33 of 5025, 423 of 5026, 99 of 5027, 849 of 5028, 49
    // of 5029 and 139 of 5030. Confirmed entries are the ones this crate
    // decodes and asserts about; the rest are named from position and shape.
    Entry {
        message_type: 5000,
        name: "TSCH.PreUFF.ChartInfoArchive",
        confidence: Unverified,
        app: App::Any,
    },
    Entry {
        message_type: 5002,
        name: "TSCH.PreUFF.ChartGridArchive",
        confidence: Unverified,
        app: App::Any,
    },
    Entry {
        message_type: 5004,
        name: "TSCH.ChartMediatorArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 5020,
        name: "TSCH.ChartStylePreset",
        confidence: Inferred,
        app: App::Any,
    },
    // The chart on the canvas: a `TSD.DrawableArchive` at field 1 and the whole
    // chart model at extension 10000. Decoded, printed by `iwork charts` and
    // asserted against the grid Keynote was told to plot.
    Entry {
        message_type: 5021,
        name: "TSCH.ChartDrawableArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 5022,
        name: "TSCH.ChartStyleArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 5023,
        name: "TSCH.ChartNonStyleArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 5024,
        name: "TSCH.LegendStyleArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 5025,
        name: "TSCH.LegendNonStyleArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 5026,
        name: "TSCH.ChartAxisStyleArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 5027,
        name: "TSCH.ChartAxisNonStyleArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 5028,
        name: "TSCH.ChartSeriesStyleArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 5029,
        name: "TSCH.ChartSeriesNonStyleArchive",
        confidence: Inferred,
        app: App::Any,
    },
    // 5030 is in every document in the corpus and 5031 in none of them: a
    // reference line's *style* is part of the theme's presets, its non-style is
    // written only when a chart has a reference line, and no fixture and no
    // bundled template has one.
    Entry {
        message_type: 5030,
        name: "TSCH.ReferenceLineStyleArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 5031,
        name: "TSCH.ReferenceLineNonStyleArchive",
        confidence: Unverified,
        app: App::Any,
    },
    // Numbers' subclass of the mediator, and the only place a chart's *live*
    // data references live: `{1: super, 2: entity_id, 3: formula storage,
    // 4: columns_are_series}`, with one `TSCE.FormulaArchive` per series and
    // per label. Decoded and printed as `fed by …`.
    Entry {
        message_type: 12006,
        name: "TN.ChartMediatorArchive",
        confidence: Confirmed,
        app: InNumbers,
    },
    // The stylesheet a document's styles belong to really is 401, and the
    // theme 402 — the 400s are where `TSS` lives.
    Entry {
        message_type: 401,
        name: "TSS.StylesheetArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 402,
        name: "TSS.ThemeArchive",
        confidence: Inferred,
        app: App::Any,
    },
    // -- media ---------------------------------------------------------------
    // Not a message type but the field the media registry lives in: the
    // package metadata's `datas`, one `TSP.DataInfo` per file under `Data/`.
    // The digest is a raw SHA-1 of the bytes, checked against `shasum` over
    // every stored file in the corpus.
    Entry {
        message_type: 242,
        name: "TSD.PencilAnnotationStorageArchive",
        confidence: Unverified,
        app: App::Any,
    },
    // -- tables (Numbers, and tables embedded in Pages/Keynote) --------------
    // 6001–6006 were guessed here before anything read a table, and two of them
    // were guessed wrong: 6004 is the cell style and 6005 the interned data
    // list, one off from what this table used to say. Everything marked
    // Confirmed below was decoded and then agreed with by Numbers itself, cell
    // by cell — see `tests/tables.rs`.
    Entry {
        message_type: 6000,
        name: "TST.TableInfoArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 6001,
        name: "TST.TableModelArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 6002,
        name: "TST.Tile",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 6003,
        name: "TST.TableStyleArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 6004,
        name: "TST.CellStyleArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 6005,
        name: "TST.TableDataList",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 6006,
        name: "TST.HeaderStorageBucket",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 6007,
        name: "TST.WPTableInfoArchive",
        confidence: Unverified,
        app: App::Any,
    },
    Entry {
        message_type: 6008,
        name: "TST.TableStylePresetArchive",
        confidence: Inferred,
        app: App::Any,
    },
    // The second registration of the data list. Both ids decode as the same
    // message, so the table has to be many-to-one.
    Entry {
        message_type: 6201,
        name: "TST.TableDataList",
        confidence: Unverified,
        app: App::Any,
    },
    Entry {
        message_type: 6204,
        name: "TST.HiddenStateFormulaOwnerArchive",
        confidence: Inferred,
        app: App::Any,
    },
    // The item list behind a pop-up-menu cell. One per pop-up cell in the
    // formats fixture, named by `CellSpecArchive.chooser_control_popup_model`.
    Entry {
        message_type: 6206,
        name: "TST.PopUpMenuModel",
        confidence: Inferred,
        app: App::Any,
    },
    // Reached from a rich-text cell's key: field 1 is the `TSWP.StorageArchive`
    // holding the text, which is how a Pages table's cells read back.
    Entry {
        message_type: 6218,
        name: "TST.RichTextPayloadArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // The ordered conditional-highlighting rules for a range of cells. Reached
    // by key from the CONDITIONAL_STYLE data list; `numbers-rules.numbers` has
    // two, and their four rules are the four the template's inspector shows.
    Entry {
        message_type: 6010,
        name: "TST.ConditionalStyleSetArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // Every table carries one whether or not it filters anything — three
    // tables, seven archives, all eight bytes long. The one in
    // `numbers-rules.numbers` that has a rule hides nine rows, and they are the
    // nine whose `hidingState` is 2.
    Entry {
        message_type: 6220,
        name: "TST.FilterSetArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 6247,
        name: "TST.TableStyleNetworkArchive",
        confidence: Inferred,
        app: App::Any,
    },
    // UUID ↔ index for a table's rows and columns. Confirmed by round-tripping
    // every column of three documents through it, and by resolving a pivot's
    // fields to the column headings the app drew for them.
    Entry {
        message_type: 6267,
        name: "TST.ColumnRowUIDMapArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // -- categories, summaries and pivots -----------------------------------
    // Read out of documents made from Apple's `Categories`, `Pivot Table
    // Basics`, `My Stocks` and `Note Taking Colourful Log` templates: no
    // AppleScript command creates any of these, so nothing else in this
    // repository can produce one. Two ObjC-name traps live here — 6372
    // unarchives as `TSTCategoryOwner` and 6369 as `TSTPivotRowColumnOrder` —
    // so a table keyed on the class name rather than the message name is wrong
    // about both.
    Entry {
        message_type: 6316,
        name: "TST.SummaryModelArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 6317,
        name: "TST.SummaryCellVendorArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 6318,
        name: "TST.CategoryOrderArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 6369,
        name: "TST.PivotOrderArchive",
        confidence: Inferred,
        app: App::Any,
    },
    // The pivot rules: source table, row/column/value fields, summary
    // functions, grand-total switches. Decoded and checked against the pivot
    // table Numbers drew from them in the same document.
    Entry {
        message_type: 6370,
        name: "TST.PivotOwnerArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // The by-reference category owner. Its one repeated field is a list of
    // references to 6373, and following it reaches the categories the app
    // shows.
    Entry {
        message_type: 6372,
        name: "TST.CategoryOwnerRefArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // One category: grouping columns, summary assignments, the group tree and
    // the on/off switch. The groups it names hold exactly the rows whose
    // grouping column carries the group's value.
    Entry {
        message_type: 6373,
        name: "TST.GroupByArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 6374,
        name: "TST.PivotGroupingColumnOptionsMapArchive",
        confidence: Unverified,
        app: App::Any,
    },
    // The out-of-line halves of the group tree. 6383 skips field 2 — the
    // number does not exist in the schema — and carries its children twice
    // over, inline at 3 and by reference at 10.
    Entry {
        message_type: 6382,
        name: "TST.GroupByArchive.AggregatorArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 6383,
        name: "TST.GroupByArchive.GroupNodeArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // -- calculation engine --------------------------------------------------
    //
    // Both entries this block used to have were wrong, and wrong in the same
    // way: they named 4008 the engine and 4009 the formula archive, one slot
    // late each. **`TSCE.FormulaArchive` has no type id at all** — it is never
    // an object, only a field of one (a `TST.TableDataList` entry, a filter
    // predicate, a tracked reference, a chart mediator). The twelve ids below
    // are the whole of `TSCE`, from the registry dumped out of the installed
    // 15.3.1 binaries, and 4000, 4003, 4004, 4005 and 4008 have all been
    // decoded out of documents here.
    //
    // `TSCE` is in the *shared* registry: every Pages document and every
    // Keynote deck carries a calculation engine, an owner-dependencies archive
    // and a named-reference manager, empty or not.
    Entry {
        message_type: 4000,
        name: "TSCE.CalculationEngineArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 4001,
        name: "TSCE.FormulaRewriteCommandArchive",
        confidence: Unverified,
        app: App::Any,
    },
    Entry {
        message_type: 4003,
        name: "TSCE.NamedReferenceManagerArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    // Renamed from `TSCE.ReferenceTrackerArchive` at 11.2, same id. Holds one
    // AST per tracked reference; in this corpus they are the header-cell
    // references a name resolves through.
    Entry {
        message_type: 4004,
        name: "TSCE.TrackedReferenceStoreArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 4005,
        name: "TSCE.TrackedReferenceArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 4007,
        name: "TSCE.RemoteDataStoreArchive",
        confidence: Inferred,
        app: App::Any,
    },
    // The per-owner hub, and the one a cross-table reference is resolved
    // through: the archive whose `owner_kind` is 35 maps a table's haunted
    // owner UUID to the `base_owner_uid` every AST writes.
    Entry {
        message_type: 4008,
        name: "TSCE.FormulaOwnerDependenciesArchive",
        confidence: Confirmed,
        app: App::Any,
    },
    Entry {
        message_type: 4009,
        name: "TSCE.CellRecordTileArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 4010,
        name: "TSCE.RangePrecedentsTileArchive",
        confidence: Unverified,
        app: App::Any,
    },
    Entry {
        message_type: 4011,
        name: "TSCE.ReferencesToDirtyArchive",
        confidence: Unverified,
        app: App::Any,
    },
    // The document-wide cache of header names, and the fragments it is tiled
    // into. Reached from `CalculationEngineArchive.header_name_manager`.
    Entry {
        message_type: 6365,
        name: "TST.HeaderNameMgrTileArchive",
        confidence: Inferred,
        app: App::Any,
    },
    Entry {
        message_type: 6366,
        name: "TST.HeaderNameMgrArchive",
        confidence: Inferred,
        app: App::Any,
    },
    // -- Pages ---------------------------------------------------------------
    //
    // This block was wrong in four of its six entries, and all four were the
    // same mistake: the only published `TP` tables are a 2013 snapshot, and
    // Apple renumbered and renamed most of this range afterwards. The registry
    // dumped out of the installed 15.3.1 binaries
    // (`reference/protos-15.3/registry-pages.tsv`) settles it, and every entry
    // below has been decoded out of a document as well — 10012's booleans are
    // the Document inspector's switches, 10143 really does hold three headers
    // and three footers, 10017 really does name a page.
    //
    // The headline rename is **"page master" → "section template"**: 2013's
    // `TP.PageMasterArchive` is 15.3.1's `TP.SectionTemplateArchive`, and the
    // new `TP.PageTemplateArchive` at 10017 is a different thing again. Seven
    // ids in the whole 730-id table carry a different message than the mined
    // references claim, and all seven are `TP`.
    Entry {
        message_type: 10000,
        name: "TP.DocumentArchive",
        confidence: Confirmed,
        app: InPages,
    },
    Entry {
        message_type: 10001,
        name: "TP.ThemeArchive",
        confidence: Inferred,
        app: InPages,
    },
    Entry {
        message_type: 10010,
        name: "TP.FloatingDrawablesArchive",
        confidence: Inferred,
        app: InPages,
    },
    // Named by its `name` field — "Blank", "Chapter Opener" — and by the three
    // section templates it points at through fields 23, 24 and 25.
    Entry {
        message_type: 10011,
        name: "TP.SectionArchive",
        confidence: Confirmed,
        app: InPages,
    },
    // Was `TP.ThemeArchive` here, which is 10001. Field 1 is `body`, the
    // word-processing/page-layout switch; 30–33 are the footnote settings and
    // 34 is `facing_pages`, all of which the document inspector shows.
    Entry {
        message_type: 10012,
        name: "TP.SettingsArchive",
        confidence: Confirmed,
        app: InPages,
    },
    // Was `TP.SettingsArchive`. It is one repeated reference field listing
    // every drawable in the document, which is what Phase 3 found it to be.
    Entry {
        message_type: 10015,
        name: "TP.DrawablesZOrderArchive",
        confidence: Confirmed,
        app: InPages,
    },
    // Was `TP.BodyStorageArchive`, a message that does not exist in any
    // version of the schema. It holds `{page index, guide storage}` pairs.
    Entry {
        message_type: 10016,
        name: "TP.UserDefinedGuideMapArchive",
        confidence: Inferred,
        app: InPages,
    },
    // Page-layout documents only, and there exactly: of the 640 bundled Pages
    // templates the 388 that carry one are precisely the 388 whose
    // `TP.SettingsArchive.body` is false.
    Entry {
        message_type: 10017,
        name: "TP.PageTemplateArchive",
        confidence: Confirmed,
        app: InPages,
    },
    // Was `TP.PageLayoutArchive` — the 2013 name for this id was
    // `TP.PageMasterArchive` and neither is right. Field 1 is three header
    // storages and field 2 three footer storages, in every one of the 3144
    // instances the bundled templates carry.
    Entry {
        message_type: 10143,
        name: "TP.SectionTemplateArchive",
        confidence: Confirmed,
        app: InPages,
    },
    // 10133 and 10147 swapped places between 2013 and 15.3.1: what the mined
    // table calls `TP.ViewStateArchive` at 10133 is now `TP.UIStateArchive`,
    // and 10147 — `TP.UIStateArchive` in that table — is now the root that
    // points at the layout state and the view state.
    Entry {
        message_type: 10131,
        name: "TP.LayoutStateArchive",
        confidence: Inferred,
        app: InPages,
    },
    Entry {
        message_type: 10133,
        name: "TP.UIStateArchive",
        confidence: Inferred,
        app: InPages,
    },
    Entry {
        message_type: 10147,
        name: "TP.ViewStateRootArchive",
        confidence: Inferred,
        app: InPages,
    },
    // -- Numbers -------------------------------------------------------------
    // The root object (identifier 1) is type 1, not 10000. That distinguishes a
    // Numbers object graph from a Pages one — but *not* from a Keynote one,
    // which numbers its own document archive 1 as well. See `App`.
    Entry {
        message_type: 1,
        name: "TN.DocumentArchive",
        confidence: Confirmed,
        app: InNumbers,
    },
    Entry {
        message_type: 2,
        name: "TN.SheetArchive",
        confidence: Unverified,
        app: InNumbers,
    },
    // The only object in any document here that carries version patches, and
    // the only reason `type == 0` needed settling before a table could be
    // written. One per spreadsheet, in `Index/ViewState*.iwa`, always with
    // three patches — for 11.0, 10.1 and 10.0 — each dropping field 28 from the
    // base and supplying its own. Pages and Keynote have no equivalent. See
    // FORMAT.md §3 and `Document::patched_objects`.
    Entry {
        message_type: 12026,
        name: "TN.UIStateArchive",
        confidence: Inferred,
        app: InNumbers,
    },
    // -- Keynote -------------------------------------------------------------
    // Derived from one deck: 1204 objects, 19 masters, 5 slides, 30 streams.
    // The app-level numbering starts at 1 and collides with Numbers throughout,
    // which is what `App` exists to keep straight.
    Entry {
        message_type: 1,
        name: "KN.DocumentArchive",
        confidence: Confirmed,
        app: InKeynote,
    },
    // Object 1 field 2 points at it, and it is the presentation: field 2 the
    // theme, field 3 the slide tree, field 4 the slide size (1920 × 1080 in the
    // sample), field 5 the document stylesheet.
    Entry {
        message_type: 2,
        name: "KN.ShowArchive",
        confidence: Confirmed,
        app: InKeynote,
    },
    // One per slide, in the show's slide tree; field 2 points at the slide.
    Entry {
        message_type: 4,
        name: "KN.SlideNodeArchive",
        confidence: Confirmed,
        app: InKeynote,
    },
    // The slide. Carries its transition — `{"Transition", "none", 1.0, 0.5}` —
    // and a reference to the master it is built on, which is what makes this a
    // presentation object rather than anything shared with the other two apps.
    Entry {
        message_type: 5,
        name: "KN.SlideArchive",
        confidence: Confirmed,
        app: InKeynote,
    },
    // The document's view state — canvas scale and offset, which slide is being
    // edited, which are collapsed. One per deck, in `Index/ViewState.iwa`.
    Entry {
        message_type: 3,
        name: "KN.UIStateArchive",
        confidence: Confirmed,
        app: InKeynote,
    },
    // The four fields 5, 6, 20 and 30 of a slide point at these, and field 2
    // says which of the five kinds it is: 1 slide number, 2 title, 3 body,
    // 4 object well, 0 unclassified. Every one of the 86 in `keynote-deck` is a
    // `TSWP.ShapeInfoArchive` at field 1 with that one byte beside it.
    Entry {
        message_type: 7,
        name: "KN.PlaceholderArchive",
        confidence: Confirmed,
        app: InKeynote,
    },
    // `KN.SlideArchive.builds` (2) points at these. **No document in this
    // corpus has one**, because nothing in Keynote's dictionary makes a build —
    // so the type is named from the 15.3.1 registry, its fields are decoded from
    // the 15.3.1 schema, and neither has ever been measured. `keynote::Build`
    // reports one if it ever meets one and `tests/keynote.rs` fails the day a
    // fixture grows one, which is the honest form of "unverified".
    Entry {
        message_type: 8,
        name: "KN.BuildArchive",
        confidence: Confirmed,
        app: InKeynote,
    },
    // What a slide's field 1 points at, one per layout and shared by every
    // slide on it. It lives in the document stylesheet, not in the layout's own
    // stream, which is why counting them per `Index/TemplateSlide-*.iwa` gave
    // the wrong idea of what it was. Named by the 15.3.1 registry; the shape
    // agrees — a `TSS.StyleArchive` at field 1.
    Entry {
        message_type: 9,
        name: "KN.SlideStyleArchive",
        confidence: Confirmed,
        app: InKeynote,
    },
    // Names itself: field 1.3 is the theme name, `"58_Startup_Simple_PM"`.
    Entry {
        message_type: 10,
        name: "KN.ThemeArchive",
        confidence: Confirmed,
        app: InKeynote,
    },
    // The presenter notes, and nothing but: one required reference to the
    // `TSWP.StorageArchive` of kind 4 that holds them. Seven bytes on the wire.
    Entry {
        message_type: 15,
        name: "KN.NoteArchive",
        confidence: Confirmed,
        app: InKeynote,
    },
    // A recorded presentation — ground rule 8's "read and pass through, never
    // author". Nothing here has one; `KN.ShowArchive.recording` (7) is where it
    // would hang. Play ▸ Record Slideshow is menu-only and the dictionary has no
    // term for it, so no probe can make one; `keynote::Recording` identifies and
    // reports rather than pretending to understand.
    Entry {
        message_type: 16,
        name: "KN.RecordingArchive",
        confidence: Inferred,
        app: InKeynote,
    },
    // Its two track kinds, listed at `KN.RecordingArchive` fields 1 and 2: the
    // event track holds navigation, laser, pause and movie events with start
    // times; the movie track holds `KN.MovieSegmentArchive`s naming `Data/`
    // entries. Unseen for the same reason the recording is.
    Entry {
        message_type: 17,
        name: "KN.RecordingEventTrackArchive",
        confidence: Inferred,
        app: InKeynote,
    },
    Entry {
        message_type: 18,
        name: "KN.RecordingMovieTrackArchive",
        confidence: Inferred,
        app: InKeynote,
    },
    // One per slide layout, holding the layout's identifier-to-style map.
    Entry {
        message_type: 19,
        name: "KN.ClassicStylesheetRecordArchive",
        confidence: Inferred,
        app: InKeynote,
    },
    // `KN.ShowArchive.soundtrack` (17) points at it, and every deck here has
    // one: volume 1, mode "play once", no movie media, eleven bytes on the wire.
    // The *track list* (3, a repeated `TSP.DataReference`) is decoded and
    // unexercised — there is no soundtrack term anywhere in Keynote's sdef, and
    // `make new audio clip` is accepted and then does nothing at all.
    Entry {
        message_type: 21,
        name: "KN.Soundtrack",
        confidence: Confirmed,
        app: InKeynote,
    },
    // The three `Index/ViewState.iwa` companions of `KN.UIStateArchive`.
    Entry {
        message_type: 23,
        name: "KN.DesktopUILayoutArchive",
        confidence: Confirmed,
        app: InKeynote,
    },
    Entry {
        message_type: 24,
        name: "KN.CanvasSelectionArchive",
        confidence: Confirmed,
        app: InKeynote,
    },
    Entry {
        message_type: 25,
        name: "KN.SlideCollectionSelectionArchive",
        confidence: Confirmed,
        app: InKeynote,
    },
    // Eighteen in the document stylesheet of `keynote-deck` — the theme's
    // motion-background presets, listed from `KN.ThemeArchive` field 10.
    Entry {
        message_type: 26,
        name: "KN.MotionBackgroundStyleArchive",
        confidence: Confirmed,
        app: InKeynote,
    },
    // The theme's live-video sources, and the collection field 9 names. Every
    // deck here carries exactly one, `"Default Camera"`, `is_default_source`
    // true — and it is the collection's `default_source` (2) while its `sources`
    // (1) is empty, so a reader that only walked the list would find none. A
    // camera is a device rather than a document: identify and report, never
    // author.
    Entry {
        message_type: 184,
        name: "KN.LiveVideoSource",
        confidence: Confirmed,
        app: InKeynote,
    },
    Entry {
        message_type: 185,
        name: "KN.LiveVideoSourceCollection",
        confidence: Confirmed,
        app: InKeynote,
    },
    // One stage of a build; `KN.SlideArchive.buildChunks` (43) lists them, in
    // order, each with its own delay, duration and automatic flag — which is
    // where "With Build 2, after 0.3s" would live. Unseen for the same reason as
    // the builds themselves, and decoded on the same terms.
    Entry {
        message_type: 153,
        name: "KN.BuildChunkArchive",
        confidence: Inferred,
        app: InKeynote,
    },
    // Seven of them, every one identifying itself as `dropcap-style-N` or
    // `drop-cap-style-default` in the TSS base at field 1.2. The archive is a
    // TSWP one rather than a KN one, so the number being in the app range is the
    // app registering a framework message rather than defining its own.
    Entry {
        message_type: 10024,
        name: "TSWP.DropCapStyleArchive",
        confidence: Inferred,
        app: InKeynote,
    },
];

/// The entry for a message type in a document of `kind`.
///
/// An app-specific entry wins over a shared one, so Keynote's type 1 resolves
/// to `KN.DocumentArchive` and Numbers' to `TN.DocumentArchive`.
pub fn lookup_in(kind: Kind, message_type: u32) -> Option<&'static Entry> {
    let matching = |app: App| {
        ENTRIES
            .iter()
            .find(|e| e.message_type == message_type && e.app == app)
    };
    App::of(kind)
        .and_then(matching)
        .or_else(|| matching(App::Any))
}

/// The entry for a message type when the document's kind is not known.
///
/// Only entries that mean the same thing in all three apps can be resolved this
/// way; an app-specific number resolves to nothing rather than to whichever app
/// happens to be listed first.
pub fn lookup(message_type: u32) -> Option<&'static Entry> {
    ENTRIES
        .iter()
        .find(|e| e.message_type == message_type && e.app == App::Any)
}

/// Human-readable label for a message type, e.g.
/// `"TSWP.StorageArchive"` or `"TSWP #2042"` when the type is unknown.
pub fn describe(message_type: u32) -> String {
    label(lookup(message_type), message_type)
}

/// Human-readable label for a message type in a document of a known kind.
///
/// Prefer this: without a kind, every app-level archive is unnameable.
pub fn describe_in(kind: Kind, message_type: u32) -> String {
    label(lookup_in(kind, message_type), message_type)
}

fn label(entry: Option<&'static Entry>, message_type: u32) -> String {
    match entry {
        Some(entry) => match entry.confidence {
            Confirmed => entry.name.to_string(),
            Inferred => format!("{}?", entry.name),
            Unverified => format!("{}??", entry.name),
        },
        None => format!("{} #{message_type}", framework(message_type)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reason `App` exists: one number, two apps, two meanings.
    #[test]
    fn type_one_depends_on_the_app() {
        assert_eq!(describe_in(Kind::Numbers, 1), "TN.DocumentArchive");
        assert_eq!(describe_in(Kind::Keynote, 1), "KN.DocumentArchive");
        // With no kind to go on, saying nothing beats saying the wrong one.
        // Below 200 the number belongs to whichever app wrote the file, so the
        // framework it falls back to is "app", not a shared one.
        assert_eq!(describe(1), "app #1");
        assert_eq!(describe_in(Kind::Unknown, 1), "app #1");
    }

    #[test]
    fn shared_types_resolve_for_every_app() {
        for kind in [Kind::Pages, Kind::Numbers, Kind::Keynote, Kind::Unknown] {
            assert_eq!(describe_in(kind, 2001), "TSWP.StorageArchive");
        }
        assert_eq!(describe(2001), "TSWP.StorageArchive");
    }

    #[test]
    fn confidence_shows_in_the_label() {
        assert_eq!(describe_in(Kind::Keynote, 5), "KN.SlideArchive");
        // 8 was the example here until keynote-builds.key confirmed it.
        assert_eq!(describe_in(Kind::Keynote, 8), "KN.BuildArchive");
        assert_eq!(describe(2024), "TSWP.ColumnStyleArchive?");
        assert_eq!(describe_in(Kind::Numbers, 2), "TN.SheetArchive??");
        assert_eq!(describe(999999), "? #999999");
    }

    /// No `(message_type, app)` pair is registered twice. A duplicate is dead
    /// weight at best — `lookup_in` returns the first match — and a
    /// contradiction at worst, which is exactly how type 401 came to be listed
    /// once as "the 400s are where TSS lives" and again as "not in the TSS
    /// range, despite the name".
    #[test]
    fn no_type_and_app_pair_is_registered_twice() {
        let mut seen = std::collections::HashSet::new();
        for entry in ENTRIES {
            assert!(
                seen.insert((entry.message_type, entry.app)),
                "type {} is registered twice for app {:?}",
                entry.message_type,
                entry.app
            );
        }
    }

    /// A Pages type must not be offered for a Keynote document, and vice versa.
    #[test]
    fn app_specific_entries_do_not_leak_across_apps() {
        assert_eq!(describe_in(Kind::Keynote, 10000), "app #10000");
        assert_eq!(describe_in(Kind::Pages, 10000), "TP.DocumentArchive");
        assert_eq!(describe_in(Kind::Pages, 10024), "app #10024");
        assert_eq!(
            describe_in(Kind::Keynote, 10024),
            "TSWP.DropCapStyleArchive?"
        );
    }
}
