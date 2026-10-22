//! Documents from nothing.
//!
//! [`crate::Document::from_template`] answers "create a document" by copying one
//! that works, which is the rule the rest of this crate lives by — and which
//! needs Apple's software installed to have something to copy. This module owes
//! the other half of the README's promise: a caller with this crate and nothing
//! else gets [`crate::Document::new`], and the bytes come from here.
//!
//! # Why this is allowed to synthesise
//!
//! FORMAT.md's rule is *copy, don't synthesise*, and it is there because
//! inventing a message out of a schema crashed Pages every time it was tried.
//! Nothing below repeals it. What is written here was instead **measured**: a
//! document each app made was reduced — one field, one object, one stream
//! deleted at a time, the app asked after every deletion whether it still opened
//! the result and still read the text back — until nothing more could go. Every
//! object this module writes is one the app was watched insisting on, and
//! everything the app was watched not needing is not written. See FORMAT.md
//! §"The least document each app will open".
//!
//! # The shape of a document
//!
//! Four things, in this order:
//!
//! 1. **Objects**, each an `ArchiveInfo` and its payload, grouped into
//!    **components** — one `Index/<name>.iwa` stream apiece.
//! 2. **The component index**, `Index/Metadata.iwa`: a `TSP.PackageMetadata`
//!    naming every component and declaring every reference that leaves one.
//! 3. **The identity**, `Metadata/Properties.plist` and
//!    `Metadata/DocumentIdentifier`: the UUIDs that say which document this is.
//! 4. **Nothing else.** No previews (the app draws its own), no build history,
//!    no view state — all three apps open a document that has none.

use std::collections::BTreeMap;

use crate::iwa::{ArchiveMessage, ArchiveObject};
use crate::pb::{Field, Message, Value};
use crate::{Kind, Package};

/// The archive version every object this crate writes carries.
///
/// `MessageInfo.version`, and it is `[1, 0, 5]` on every object of every
/// document in the corpus — the apps write one number for the whole file.
pub(crate) const VERSION: [u32; 3] = [1, 0, 5];

/// `TSP.PackageMetadata` field 7 — `[26, 3, 1]`, packed, and the same in every
/// document of all three apps.
///
/// See [`Blueprint::component_index`] for what leaving it out does, which is not
/// what leaving out a version usually does.
const PACKAGE_VERSION: [u8; 3] = [26, 3, 1];

/// `TSP.ComponentInfo.save_token` / `read_version` — `[2, 0, 0]`, packed.
const COMPONENT_VERSION: [u8; 3] = [2, 0, 0];

/// `TSP.ComponentInfo` field 12, and `TSP.PackageMetadata` field 8: the version
/// of the format the component was written in.
///
/// **It is per app, and that matters.** Pages writes 834 and 835, Numbers 535,
/// 536 and 537, Keynote 2385 and 2386 — the highest of each being what the
/// package itself claims in field 8. Writing Pages' number into a Numbers
/// document was watched being refused: everything else about the document was
/// right, every object had been grafted from one the app wrote, and Numbers
/// still would not open it. Nothing in this crate reads the number for meaning;
/// what it does is write back the one the app that has to read it uses.
fn component_generation(kind: Kind) -> u64 {
    match kind {
        Kind::Pages => 835,
        Kind::Numbers => 537,
        Kind::Keynote => 2386,
        Kind::Unknown => 0,
    }
}

/// The root object of a document, and the root of its metadata component.
///
/// Both are fixed by convention rather than by anything in the bytes: every
/// document in the corpus numbers its document archive 1 and its
/// `DocumentMetadata` 71, and a reader that has neither open has nothing to
/// resolve an identifier against ([`crate::Document::undeclared_references`]).
pub(crate) const ROOT: u64 = 1;
pub(crate) const DOCUMENT_METADATA: u64 = 71;
/// The `TSP.PackageMetadata` itself, which is object 2 in every document.
const PACKAGE_METADATA: u64 = 2;

/// Where the identifiers this module hands out begin.
///
/// Above the four the format fixes (1, 2, 71, and the data-metadata map), and
/// low enough that a document written here is readable: an object numbered
/// 1_732_539 is Pages counting from wherever its session had got to, not
/// something the format asks for.
const FIRST_IDENTIFIER: u64 = 1000;

// -- building blocks ----------------------------------------------------------

/// A protobuf message from its fields.
pub(crate) fn message(fields: Vec<Field>) -> Message {
    Message { fields }
}

pub(crate) fn varint(number: u32, value: u64) -> Field {
    Field {
        number,
        value: Value::Varint(value),
    }
}

pub(crate) fn string(number: u32, text: &str) -> Field {
    Field {
        number,
        value: Value::Bytes(text.as_bytes().to_vec()),
    }
}

pub(crate) fn float(number: u32, value: f32) -> Field {
    Field {
        number,
        value: Value::Fixed32(value.to_le_bytes()),
    }
}

#[allow(dead_code)]
pub(crate) fn double(number: u32, value: f64) -> Field {
    Field {
        number,
        value: Value::Fixed64(value.to_le_bytes()),
    }
}

#[allow(dead_code)]
/// Raw bytes as a length-delimited field.
pub(crate) fn bytes(number: u32, value: Vec<u8>) -> Field {
    Field {
        number,
        value: Value::Bytes(value),
    }
}

/// A nested message as a length-delimited field.
pub(crate) fn nested(number: u32, fields: Vec<Field>) -> Field {
    Field {
        number,
        value: Value::Bytes(message(fields).encode()),
    }
}

/// A `TSP.Reference` — `{1: identifier}`, the whole of it.
pub(crate) fn reference(number: u32, target: u64) -> Field {
    nested(number, vec![varint(1, target)])
}

/// An attribute table with a single entry covering the whole storage.
pub(crate) fn attribute_table(number: u32, target: u64) -> Field {
    Field {
        number,
        value: Value::Bytes(
            message(vec![nested(1, vec![varint(1, 0), reference(2, target)])]).encode(),
        ),
    }
}

// -- the blueprint ------------------------------------------------------------

/// A document being assembled: components, the objects in them, and the
/// identifiers handed out so far.
pub(crate) struct Blueprint {
    kind: Kind,
    components: Vec<Component>,
    next: u64,
}

/// One component — one `Index/*.iwa` stream, and the objects it holds.
struct Component {
    /// `ComponentInfo.identifier`, which is also its root object's.
    identifier: u64,
    /// `ComponentInfo.preferred_name`: "Document", "DocumentStylesheet".
    name: String,
    /// `ComponentInfo.locator`, when the stream is not named for the component
    /// — `CalculationEngine-1234` for a `CalculationEngine`.
    locator: Option<String>,
    objects: Vec<ArchiveObject>,
    /// Set on `DocumentMetadata` alone, in every document in the corpus.
    always_loaded: bool,
}

/// Which component an object goes in, handed back by [`Blueprint::component`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ComponentId(usize);

impl Blueprint {
    /// A blueprint with the two components every document has.
    pub(crate) fn new(kind: Kind) -> Blueprint {
        let mut blueprint = Blueprint {
            kind,
            components: Vec::new(),
            next: FIRST_IDENTIFIER,
        };
        blueprint.push_component("Document", ROOT, None, false);
        blueprint.push_component("DocumentMetadata", DOCUMENT_METADATA, None, true);
        blueprint
    }

    fn push_component(
        &mut self,
        name: &str,
        identifier: u64,
        locator: Option<String>,
        always_loaded: bool,
    ) -> ComponentId {
        self.components.push(Component {
            identifier,
            name: name.to_string(),
            locator,
            objects: Vec::new(),
            always_loaded,
        });
        ComponentId(self.components.len() - 1)
    }

    /// The `Document` component, which every blueprint has.
    pub(crate) fn document(&self) -> ComponentId {
        ComponentId(0)
    }

    fn metadata(&self) -> ComponentId {
        ComponentId(1)
    }

    /// A new component, rooted at a fresh identifier.
    ///
    /// `locator` is the stream name when it is not the component's name; iWork
    /// writes `CalculationEngine-1732610` for the calculation engine and the
    /// plain name for everything else.
    pub(crate) fn component(&mut self, name: &str, numbered: bool) -> (ComponentId, u64) {
        let identifier = self.allocate();
        let locator = numbered.then(|| format!("{name}-{identifier}"));
        (
            self.push_component(name, identifier, locator, false),
            identifier,
        )
    }

    #[allow(dead_code)]
    /// A component holding one object, which is its root.
    ///
    /// How Numbers stores a table: every tile, every interning list and every
    /// header bucket is a component of its own, named `Tables/Tile-1007` and
    /// the like, so the app can load one without loading the rest.
    pub(crate) fn in_own_component(
        &mut self,
        name: &str,
        message_type: u32,
        archive: Message,
    ) -> u64 {
        let (component, identifier) = self.component(name, true);
        self.put(component, identifier, message_type, archive);
        identifier
    }

    /// The next unused object identifier.
    pub(crate) fn allocate(&mut self) -> u64 {
        let identifier = self.next;
        self.next += 1;
        identifier
    }

    /// Put an object in a component under an identifier already allocated.
    pub(crate) fn put(
        &mut self,
        component: ComponentId,
        identifier: u64,
        message_type: u32,
        archive: Message,
    ) {
        self.components[component.0].objects.push(ArchiveObject {
            identifier,
            messages: vec![ArchiveMessage {
                message_type,
                version: VERSION.to_vec(),
                extra: Vec::new(),
                payload: archive.encode(),
            }],
            extra: Vec::new(),
        });
    }

    /// Allocate an identifier and put an object under it.
    pub(crate) fn add(
        &mut self,
        component: ComponentId,
        message_type: u32,
        archive: Message,
    ) -> u64 {
        let identifier = self.allocate();
        self.put(component, identifier, message_type, archive);
        identifier
    }

    /// Every object, whichever component it is in — for the reference walk that
    /// writes the component index.
    fn objects(&self) -> impl Iterator<Item = (&Component, &ArchiveObject)> {
        self.components
            .iter()
            .flat_map(|component| component.objects.iter().map(move |o| (component, o)))
    }

    /// Assemble the package: object streams, the component index, the identity.
    pub(crate) fn finish(mut self) -> Package {
        self.write_document_metadata();

        let mut package = Package {
            entries: Vec::new(),
            form: crate::package::Form::SingleFile,
        };
        for component in &self.components {
            package.set(
                &stream_name(component),
                crate::iwa::serialize(&component.objects),
            );
        }
        package.set(
            "Index/Metadata.iwa",
            crate::iwa::serialize(&[self.component_index()]),
        );
        crate::metadata::write_identity(&mut package, self.kind);
        package
    }

    /// `TSP.DocumentMetadata` — the object every reader opens second.
    ///
    /// Field 3 is a repeated save token; the app writes one per save, carrying a
    /// digest of what it wrote. A document that has never been saved by an app
    /// has none, and Pages, Numbers and Keynote all open one that way.
    fn write_document_metadata(&mut self) {
        let component = self.metadata();
        self.put(
            component,
            DOCUMENT_METADATA,
            TYPE_DOCUMENT_METADATA,
            message(vec![varint(1, 0)]),
        );
    }

    /// `TSP.PackageMetadata`: every component, and every reference that leaves
    /// the component it is written in.
    fn component_index(&self) -> ArchiveObject {
        // Which component each object lives in, so a reference can be told from
        // one that stays at home.
        let mut home: BTreeMap<u64, u64> = BTreeMap::new();
        for (component, object) in self.objects() {
            home.insert(object.identifier, component.identifier);
        }

        // Field 1 is the high-water mark: the highest identifier ever handed
        // out in this package. iWork allocates above it rather than above what
        // it can see, so a document that understates it hands out an identifier
        // something already has. `Document::problems` checks it, and did so
        // against the first document this module wrote.
        let highest = self
            .objects()
            .map(|(_, object)| object.identifier)
            .max()
            .unwrap_or(DOCUMENT_METADATA)
            .max(self.next);
        let mut fields = vec![
            varint(1, highest),
            nested(2, vec![string(2, &crate::metadata::uuid()), varint(3, 0)]),
        ];
        for component in &self.components {
            let mut info = vec![varint(1, component.identifier), string(2, &component.name)];
            if let Some(locator) = &component.locator {
                info.push(string(3, locator));
            }
            info.push(Field {
                number: 4,
                value: Value::Bytes(COMPONENT_VERSION.to_vec()),
            });
            info.push(Field {
                number: 5,
                value: Value::Bytes(COMPONENT_VERSION.to_vec()),
            });

            // The declarations. `Document` and `DocumentMetadata` are open
            // before anything else is read, so a reference into either needs no
            // declaring; every other crossing does, root objects included.
            let mut declared: Vec<u64> = Vec::new();
            for object in &component.objects {
                let Ok(archive) = Message::decode(object.payload()) else {
                    continue;
                };
                for target in crate::style::references(&archive) {
                    let Some(&owner) = home.get(&target) else {
                        continue;
                    };
                    if owner == component.identifier
                        || target == ROOT
                        || target == DOCUMENT_METADATA
                    {
                        continue;
                    }
                    declared.push(target);
                }
            }
            declared.sort_unstable();
            declared.dedup();
            for target in declared {
                let owner = home[&target];
                // Two forms, and they are not interchangeable: `{component}`
                // alone declares that component's **root**, whose identifier is
                // the component's own, and `{component, object}` declares one
                // object inside it. iWork writes the short form for a root and
                // the long form for everything else.
                info.push(match target == owner {
                    true => nested(6, vec![varint(1, owner)]),
                    false => nested(6, vec![varint(1, owner), varint(2, target)]),
                });
            }

            // Every object in the component, given a UUID of its own.
            //
            // Pages does not need these — the reduction deleted all 2,450 of
            // them and the document still opened. Numbers does: nothing in its
            // component index could be deleted at all, this included, and a
            // document written without them is refused. FORMAT.md's rule 23
            // already said a *new* component needs fresh
            // `object_uuid_map_entries`; this says a new document does too.
            for object in &component.objects {
                let (lower, upper) = object_uuid();
                info.push(nested(
                    11,
                    vec![
                        varint(1, object.identifier),
                        nested(2, vec![varint(1, lower), varint(2, upper)]),
                    ],
                ));
            }

            info.push(varint(10, 0));
            info.push(varint(12, component_generation(self.kind)));
            if component.always_loaded {
                info.push(varint(21, 1));
            }
            fields.push(nested(3, info));
        }
        // The package's own versions and generation. Three separate numbers,
        // and each was learned by leaving it out:
        //
        // * **Field 5** is the package's save version. Left out, the document is
        //   refused outright. `[2, 0, 0]`, which is what a Pages package carries
        //   and what a Numbers one accepts — Numbers' own says `[3, 2, 10]` and
        //   Keynote's `[2, 4, 0]`, and neither is needed.
        // * **Field 7** is the one that cost the most. Left out, Numbers opens
        //   the document, finds the sheet, finds the table — and answers
        //   `missing value` for every cell in it. Nothing is refused, nothing is
        //   reported, the document is simply empty. It is `[26, 3, 1]` in Pages,
        //   Numbers and Keynote alike, so it is written as the constant it is.
        // * **Field 8** is the format version, and is per app
        //   ([`component_generation`]).
        fields.push(Field {
            number: 5,
            value: Value::Bytes(COMPONENT_VERSION.to_vec()),
        });
        fields.push(Field {
            number: 7,
            value: Value::Bytes(PACKAGE_VERSION.to_vec()),
        });
        fields.push(varint(8, component_generation(self.kind)));

        ArchiveObject {
            identifier: PACKAGE_METADATA,
            messages: vec![ArchiveMessage {
                message_type: TYPE_PACKAGE_METADATA,
                version: VERSION.to_vec(),
                extra: Vec::new(),
                payload: message(fields).encode(),
            }],
            extra: Vec::new(),
        }
    }
}

/// A `TSP.UUID` for one object, as two 64-bit halves.
///
/// Uniqueness inside the document is the whole requirement — two components
/// claiming the same object UUID is how a document that still opens ends up
/// corrupted (FORMAT.md rule 23) — so the entropy comes from the same place the
/// document's own UUIDs do.
fn object_uuid() -> (u64, u64) {
    let hex = crate::metadata::uuid().replace('-', "");
    let half = |range: std::ops::Range<usize>| u64::from_str_radix(&hex[range], 16).unwrap_or(0);
    (half(0..16), half(16..32))
}

fn stream_name(component: &Component) -> String {
    format!(
        "Index/{}.iwa",
        component.locator.as_deref().unwrap_or(&component.name)
    )
}

/// `TSP.PackageMetadata`.
const TYPE_PACKAGE_METADATA: u32 = 11006;
/// `TSP.DocumentMetadata`.
const TYPE_DOCUMENT_METADATA: u32 = 11011;

// -- Pages --------------------------------------------------------------------

/// `TSWP.StorageArchive`.
const TYPE_STORAGE: u32 = 2001;
/// `TSS.StylesheetArchive`.
pub(crate) const TYPE_STYLESHEET: u32 = 401;
/// `TSWP.ColumnStyleArchive`.
const TYPE_COLUMN_STYLE: u32 = 2024;
/// `TP.DocumentArchive`.
const TYPE_PAGES_DOCUMENT: u32 = 10000;

/// The paper a new Pages document starts on.
///
/// Pages ships its Blank template twice — `ISO.template` and
/// `Traditional.template` — and the only difference between the two documents
/// is this: the page, the margins and the paper's name. Both were read out of
/// those templates rather than looked up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Paper {
    /// 595.28 × 841.89 pt, 56.69 pt margins. What Pages calls `iso-a4`.
    #[default]
    A4,
    /// 612 × 792 pt, 72 pt margins. What Pages calls `na-letter`.
    Letter,
}

impl Paper {
    /// Width, height, margin, header inset, footer inset — all in points.
    fn measurements(self) -> (f32, f32, f32, f32, f32) {
        match self {
            Paper::A4 => (595.28, 841.89, 56.692917, 35.43307, 42.519684),
            Paper::Letter => (612.0, 792.0, 72.0, 36.0, 43.2),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Paper::A4 => "iso-a4",
            Paper::Letter => "na-letter",
        }
    }
}

/// A Pages document with one empty body.
///
/// Eleven objects, which is what the reduction left of the 569 in a document
/// Pages made from its own Blank template:
///
/// ```text
/// 1     TP.DocumentArchive      the page, the margins, the locale
///  └ body  TSWP.StorageArchive  the text, and the tables anchored into it
/// 401   TSS.StylesheetArchive   in its own component
///  ├ TSWP.ParagraphStyleArchive "Body"
///  ├ TSWP.ListStyleArchive      "None"
///  └ TSWP.ColumnStyleArchive    "None", and the variation the body points at
/// 71    TSP.DocumentMetadata
/// ```
///
/// What is *not* here is as measured as what is: no theme, no section, no
/// section templates, no headers or footers, no view state, no calculation
/// engine, no annotation authors, no previews, no build history. Pages opens
/// this and reads the text back out of it.
pub(crate) fn pages(paper: Paper) -> Blueprint {
    let mut blueprint = Blueprint::new(Kind::Pages);
    let document = blueprint.document();
    let (styles, stylesheet) = blueprint.component("DocumentStylesheet", false);

    // -- the stylesheet's styles, which have to exist before it lists them ----

    let list = blueprint.add(
        styles,
        crate::style::TYPE_LIST_STYLE,
        list_style(stylesheet),
    );
    let body_style = blueprint.add(
        styles,
        crate::style::TYPE_PARAGRAPH_STYLE,
        paragraph_style(stylesheet, list),
    );
    let column = blueprint.add(styles, TYPE_COLUMN_STYLE, column_style(stylesheet));
    let body_column = blueprint.add(
        styles,
        TYPE_COLUMN_STYLE,
        column_variation(stylesheet, column),
    );
    // A text box has to be drawn with something. The app's own themes carry a
    // whole set of presets; a document with no theme carries the one this
    // crate can put to use, named the way a theme names it.
    let text_box_style = blueprint.add(
        styles,
        TYPE_SHAPE_STYLE,
        message(vec![nested(
            1,
            vec![
                nested(
                    1,
                    vec![string(2, TEXT_BOX_IDENTIFIER), reference(5, stylesheet)],
                ),
                // The property bag, with a fill that is *present and paints
                // nothing* — `{11: {1: <empty>}}`, byte for byte what a shape
                // style in a document Keynote wrote carries. Without it there
                // is nowhere for a colour to go, and this crate refuses to
                // invent the message rather than write into one it has seen.
                nested(11, vec![bytes(1, Vec::new())]),
            ],
        )]),
    );
    // And one for an image, which is drawn with a media style rather than a
    // shape style — a different archive with different field numbers.
    let image_style = blueprint.add(
        styles,
        TYPE_MEDIA_STYLE,
        message(vec![
            nested(
                1,
                vec![string(2, IMAGE_IDENTIFIER), reference(5, stylesheet)],
            ),
            // The same bag, one level up because a media style is the `TSD`
            // archive rather than a wrapper around it, and numbered one lower
            // because it has no fill: field 1 here is the stroke.
            nested(11, vec![bytes(1, Vec::new())]),
        ]),
    );
    // A calculation engine, which every Pages document the app makes carries
    // whether or not it has a table — `pages-plain` and `pages-styled` have one
    // apiece with nothing in it. It is also *where a table goes*: a Pages
    // document keeps its `TST.TableInfoArchive` and `TST.TableModelArchive` in
    // the engine's component, so without one there is nowhere to put a table.
    let (engine_component, engine) = blueprint.component("CalculationEngine", false);
    blueprint.put(
        engine_component,
        engine,
        TYPE_CALCULATION_ENGINE,
        message(vec![nested(2, Vec::new())]),
    );

    // The styles a *table* is drawn with. A Pages document has no table when
    // it is made, and carries them anyway — a blank document the app makes
    // holds 102 cell styles with not a table in sight — which is what lets one
    // be added later: `add_table` finds them by the names registered here.
    //
    // They are **not listed in the stylesheet**, and that is not an oversight:
    // adding them to its field 8 the way a Numbers stylesheet lists its styles
    // made Pages refuse the document outright — measured, by bisecting a
    // document it would not open. The styles are found by name among the
    // objects instead, which is what `add_table` does.
    blueprint.add(
        styles,
        TYPE_TABLE_STYLE,
        named_style("table-0-tableStyle", stylesheet),
    );
    for area in CELL_AREAS {
        blueprint.add(
            styles,
            TYPE_CELL_STYLE,
            cell_style(&format!("tableCell-0-{area}"), stylesheet),
        );
    }
    for area in TEXT_AREAS {
        blueprint.add(
            styles,
            crate::style::TYPE_PARAGRAPH_STYLE,
            table_text_style(
                &format!("text-0-paragraphstyle-{area}"),
                area,
                stylesheet,
                list,
            ),
        );
    }
    blueprint.put(
        styles,
        stylesheet,
        TYPE_STYLESHEET,
        stylesheet_archive(column, body_column, body_style, text_box_style, image_style),
    );

    // -- the body, and the document that points at it -------------------------

    let body = blueprint.add(
        document,
        TYPE_STORAGE,
        body_storage(stylesheet, body_style, list, body_column),
    );
    // Nothing floats on a new page, but the archive that would hold it exists
    // anyway — every Pages document in the corpus has one, empty, and it is
    // where a text box put on a page has to be named.
    let floating = blueprint.add(
        document,
        crate::pages::TYPE_FLOATING_DRAWABLES,
        message(vec![]),
    );
    // The body flow is itself in the stack, at the bottom: that is what Pages
    // writes into a document with no drawables at all.
    let zorder = blueprint.add(
        document,
        crate::pages::TYPE_ZORDER,
        message(vec![reference(1, body)]),
    );
    blueprint.put(
        document,
        ROOT,
        TYPE_PAGES_DOCUMENT,
        pages_document(stylesheet, body, floating, zorder, paper),
    );
    blueprint
}

/// `TST.ColumnRowUIDMapArchive` — one UUID per column and one per row.
///
/// Three arrays per axis, and the third is the inverse of the second: entry
/// *i* holds `uuid[i]` at `index[i]`, and `uid_for_index[index[i]] == i`. The
/// app emits them **sorted by the 128-bit UUID**, high half first, which is
/// why the indices read out of order in a document Numbers wrote.
fn column_row_uid_map(rows: usize, columns: usize, seed: u64) -> Message {
    let axis = |fields: &mut Vec<Field>, base: u32, count: usize, salt: u64| {
        let mut pairs: Vec<((u64, u64), usize)> = (0..count)
            .map(|index| {
                let mut state = seed
                    ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
                let mut next = move || {
                    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    z ^ (z >> 31)
                };
                ((next(), next()), index)
            })
            .collect();
        // Sorted the way the app keeps them: by the 128-bit value, high first.
        pairs.sort_by_key(|((lower, upper), _)| (*upper, *lower));
        for ((lower, upper), _) in &pairs {
            fields.push(nested(base, vec![varint(1, *lower), varint(2, *upper)]));
        }
        for (_, index) in &pairs {
            fields.push(varint(base + 1, *index as u64));
        }
        let mut position_of = vec![0u64; count];
        for (position, (_, index)) in pairs.iter().enumerate() {
            position_of[*index] = position as u64;
        }
        for position in position_of {
            fields.push(varint(base + 2, position));
        }
    };
    let mut fields = Vec::new();
    axis(&mut fields, 1, columns, 1);
    axis(&mut fields, 4, rows, 2);
    message(fields)
}

/// `TST.ColumnRowUIDMapArchive`.
pub(crate) const TYPE_UID_MAP: u32 = 6267;

/// `TSS.StylesheetArchive`: the styles a document offers by name.
///
/// Field 5 is the pair of column styles the body's layout comes from; field 8 is
/// the list of named styles, as references and again as `{identifier, style}`
/// entries — the keyed form is what the app looks a style up by when it applies
/// one from the menu.
fn stylesheet_archive(
    column: u64,
    body_column: u64,
    body_style: u64,
    text_box_style: u64,
    image_style: u64,
) -> Message {
    message(vec![
        varint(4, 0),
        nested(5, vec![reference(1, column), reference(2, body_column)]),
        varint(6, 1),
        nested(
            8,
            vec![
                reference(1, body_style),
                reference(1, text_box_style),
                reference(1, image_style),
                nested(
                    2,
                    vec![string(1, BODY_IDENTIFIER), reference(2, body_style)],
                ),
                nested(
                    2,
                    vec![string(1, TEXT_BOX_IDENTIFIER), reference(2, text_box_style)],
                ),
                nested(
                    2,
                    vec![string(1, IMAGE_IDENTIFIER), reference(2, image_style)],
                ),
            ],
        ),
    ])
}

/// The internal name of the shape style a text box is drawn with, in the shape
/// a theme names its presets.
const TEXT_BOX_IDENTIFIER: &str = "textbox-style-preset-0";

/// The same, for the media style an image is drawn with. A
/// `TSD.MediaStyleArchive` numbers its properties one lower than a shape
/// style, because it has no fill.
const IMAGE_IDENTIFIER: &str = "image-style-preset-0";

/// The internal name of the one paragraph style a new document has.
///
/// The shape is the app's — `text-<n>-paragraphstyle-<name>` — and the number
/// is a theme's index, which a document with no theme does not have. 0 is what
/// the bundled themes number their first.
const BODY_IDENTIFIER: &str = "text-0-paragraphstyle-Body";

/// `TSWP.ParagraphStyleArchive` — "Body", 11 pt Helvetica Neue, black, ranged
/// left.
///
/// Field 11 is the character bag and field 12 the paragraph bag; the reduction
/// deleted **both** and Pages still opened the document, so everything here is
/// chosen rather than required. What is chosen is the smallest set that makes
/// the result a document rather than a shrug: a font, a size, a colour, an
/// alignment, and the list style the paragraph is not in.
fn paragraph_style(stylesheet: u64, list: u64) -> Message {
    use crate::style::property;
    message(vec![
        nested(
            1,
            vec![
                string(1, "Body"),
                string(2, BODY_IDENTIFIER),
                reference(5, stylesheet),
            ],
        ),
        nested(
            property::FONT_SIZE[0],
            vec![
                varint(property::BOLD[1], 0),
                varint(property::ITALIC[1], 0),
                float(property::FONT_SIZE[1], 11.0),
                string(property::FONT_NAME[1], "HelveticaNeue"),
                nested(property::FONT_COLOR[1], black()),
            ],
        ),
        nested(
            property::ALIGNMENT[0],
            vec![
                // 4 is "natural" — left in a left-to-right language, and what
                // every Body style in the corpus carries.
                varint(property::ALIGNMENT[1], 4),
                reference(property::LIST_STYLE[1], list),
            ],
        ),
    ])
}

/// `TSP.Color`, in the sRGB form every style in the corpus uses: a model, three
/// components, an alpha — and field 12, which is the colour space and is 1 on
/// every colour Pages 15.3.1 writes.
fn black() -> Vec<Field> {
    vec![
        varint(1, 1),
        float(3, 0.0),
        float(4, 0.0),
        float(5, 0.0),
        float(6, 1.0),
        varint(12, 1),
        float(13, 1.0),
    ]
}

/// `TSWP.ListStyleArchive` — "None", the list a paragraph is in when it is in
/// no list.
///
/// The nine repeated values are one per indent level: field 11 the label type
/// (0, none), field 12 the label indent and field 13 the text indent, 36 pt
/// apart. Nine is what Pages writes; nothing here has asked whether it is a
/// limit or a habit.
fn list_style(stylesheet: u64) -> Message {
    let mut fields = vec![
        nested(
            1,
            vec![
                string(1, "None"),
                string(2, "text-0-liststyle-None"),
                reference(5, stylesheet),
            ],
        ),
        varint(10, 4),
    ];
    for _ in 0..9 {
        fields.push(varint(11, 0));
    }
    for _ in 0..9 {
        fields.push(float(12, 0.0));
    }
    for level in 0..9 {
        fields.push(float(13, 36.0 * level as f32));
    }
    message(fields)
}

/// `TSWP.ColumnStyleArchive` — "None": one column, no gutter.
fn column_style(stylesheet: u64) -> Message {
    message(vec![
        nested(
            1,
            vec![
                string(1, "None"),
                string(2, "column-style-default"),
                reference(5, stylesheet),
            ],
        ),
        varint(10, 9),
        nested(
            11,
            vec![
                // One column, equal widths, no gap.
                varint(1, 1),
                varint(2, 0),
                varint(3, 0),
                float(4, 0.0),
                varint(5, 0),
                nested(7, vec![reference(1, ROOT)]),
            ],
        ),
    ])
}

/// The anonymous column style the body actually points at.
///
/// A *variation*: no name, a parent, and field 4 set — the archive iWork writes
/// when something is formatted directly rather than by picking a named style.
/// The body storage points at this one, and this one inherits everything from
/// the style above.
fn column_variation(stylesheet: u64, parent: u64) -> Message {
    message(vec![
        nested(
            1,
            vec![reference(3, parent), varint(4, 1), reference(5, stylesheet)],
        ),
        varint(10, 1),
    ])
}

/// `TSWP.StorageArchive` for the body: no text, and one entry in every
/// attribute table.
///
/// A table entry is `{1: character index, 2: value}` and the first is always at
/// 0 ([`crate::style`]). An empty storage still needs them: they are what says
/// which paragraph style the first paragraph is in, and Pages writes exactly
/// this — one entry each — for the empty body of a new document.
fn body_storage(stylesheet: u64, paragraph: u64, list: u64, column: u64) -> Message {
    message(vec![
        // kind 0: the body of a word-processing document.
        varint(1, 0),
        reference(2, stylesheet),
        attribute_table(5, paragraph),
        // table_para_data — `{index, first, second}`, the list level and its
        // number, both zero for a paragraph in no list.
        nested(
            6,
            vec![nested(1, vec![varint(1, 0), varint(2, 0), varint(3, 0)])],
        ),
        attribute_table(7, list),
        varint(10, 1),
        attribute_table(12, column),
        // table_para_starts and table_para_bidi: one entry, zero.
        nested(
            14,
            vec![nested(1, vec![varint(1, 0), varint(2, 0), varint(3, 0)])],
        ),
        nested(
            24,
            vec![nested(1, vec![varint(1, 0), varint(2, 0), varint(3, 0)])],
        ),
    ])
}

/// `TP.DocumentArchive`: the page, and what is on it.
///
/// Field 15 is the `TSA.DocumentArchive` every app's document archive is built
/// on — the locale, the language, and which template this came from. A document
/// this crate made came from no template, so it says so by leaving field 9 out.
fn pages_document(stylesheet: u64, body: u64, floating: u64, zorder: u64, paper: Paper) -> Message {
    let (width, height, margin, header, footer) = paper.measurements();
    message(vec![
        reference(2, stylesheet),
        reference(3, floating),
        reference(4, body),
        nested(
            15,
            vec![nested(
                1,
                vec![
                    // TSK.DocumentArchive: the two locales, both unset, and the
                    // flags that say they were never chosen by a user.
                    varint(10, 1),
                    varint(12, 0),
                    varint(15, 1),
                    varint(16, 1),
                ],
            )],
        ),
        reference(20, zorder),
        float(30, width),
        float(31, height),
        float(32, margin),
        float(33, margin),
        float(34, margin),
        float(35, margin),
        float(36, header),
        float(37, footer),
        float(38, 1.0),
        varint(39, 0),
        varint(40, 0),
        varint(42, 0),
        // The printer this was laid out for, and the paper in it. A space is
        // what the bundled templates carry — the app fills in a real one the
        // first time the document meets a printer.
        string(43, " "),
        string(44, paper.name()),
    ])
}

// -- Numbers ------------------------------------------------------------------

/// `TN.DocumentArchive` — the root of a Numbers document, and object 1.
const TYPE_NUMBERS_DOCUMENT: u32 = 1;
/// `TN.SheetArchive`.
const TYPE_SHEET: u32 = 2;
/// `TST.TableInfoArchive` — the drawable a table hangs off.
const TYPE_TABLE_INFO: u32 = 6000;
/// `TST.TableModelArchive` — the table itself.
const TYPE_TABLE_MODEL: u32 = 6001;
/// `TST.Tile` — the cells, 256 rows at a time.
const TYPE_TILE: u32 = 6002;
/// `TST.TableDataList` — the interning tables cells refer into.
const TYPE_DATA_LIST: u32 = 6005;
/// `TST.HeaderStorageBucket` — per-row and per-column sizes and cell counts.
const TYPE_HEADER_BUCKET: u32 = 6006;

/// A tile covers this many rows, and `TileStorage.tile_size` says so.
const TILE_SIZE: u64 = 256;
/// What Numbers gives a new table, in points.
const DEFAULT_ROW_HEIGHT: f64 = 19.929931640625;
const DEFAULT_COLUMN_WIDTH: f64 = 98.0;

/// A Numbers document with one sheet and one empty table.
///
/// Numbers opens it, reports the sheet and the table, and — told to save —
/// writes the whole thing back with the cells intact. Eighty-two objects, and
/// every one of them is here because the app was watched insisting on it:
///
/// * A Numbers root needs four things a Pages root does not — a stylesheet, a
///   theme, a `TSK` 205 (empty, and required all the same) and a calculation
///   engine. Deleting any one of them from a document Numbers wrote makes
///   Numbers refuse it.
/// * `ComponentInfo` field 12 and `PackageMetadata` field 8 are **per app**:
///   Pages writes 835, Numbers 537, Keynote 2386. Field 7 is `[26, 3, 1]` in
///   all three, and a document without it opens *empty*.
/// * Nothing in a Numbers component index is deletable — twenty-five probes,
///   not one accepted — where a Pages one gives up 2,450 object-UUID entries at
///   once. So the UUID map is written for every document now.
/// * Every tile, interning list and header bucket is a component of its own,
///   and the table's info and model live in the `CalculationEngine` component.
/// * The stylesheet must say `is_locked = false`, and the theme must carry the
///   style presets the app reaches for when it finds a style missing.
/// * A `TST.TableModelArchive` names a table style, seventeen cell styles and
///   eight paragraph styles, one per area of the table; its `DataStore` names
///   eight interning lists; and its `category_owner_deprecated.owner_uid` has
///   `lower` and `upper` that are `required` even when they are zero.
pub(crate) fn numbers(
    sheet_name: &str,
    table_name: &str,
    rows: usize,
    columns: usize,
) -> Blueprint {
    let mut blueprint = Blueprint::new(Kind::Numbers);
    let document = blueprint.document();

    // The interning tables every cell refers into. A new table has interned
    // nothing, so each is a list type and a next key of 1 — keys start at 1
    // because a stored key of 0 means "none".
    // Eight of them, which is what a table Numbers made carries: one per kind
    // of thing a cell can point at. A new table has interned nothing, so each
    // is a list type and a next key of 1 — keys start at 1 because a stored key
    // of 0 means "none".
    let strings = blueprint.in_own_component("Tables/DataList", TYPE_DATA_LIST, data_list(1));
    let formulas = blueprint.in_own_component("Tables/DataList", TYPE_DATA_LIST, data_list(3));
    let conditional = blueprint.in_own_component("Tables/DataList", TYPE_DATA_LIST, data_list(2));
    let list_10 = blueprint.in_own_component("Tables/DataList", TYPE_DATA_LIST, data_list(10));
    let list_11 = blueprint.in_own_component("Tables/DataList", TYPE_DATA_LIST, data_list(11));
    let controls = blueprint.in_own_component("Tables/DataList", TYPE_DATA_LIST, data_list(12));
    // The format list starts with one entry rather than none: the automatic
    // format, which every slot can point at. `set_cell` gives a cell its format
    // by borrowing one the table already uses, and a table with no cells has
    // none to borrow — so a new table would be one no value could be written
    // into. Its refcount is 0 until a cell takes it.
    let formats =
        blueprint.in_own_component("Tables/DataList", TYPE_DATA_LIST, automatic_format_list());
    let styles = blueprint.in_own_component("Tables/DataList", TYPE_DATA_LIST, data_list(4));
    // The list style every paragraph style needs to say it is in no list.
    let (style_component, stylesheet) = blueprint.component("DocumentStylesheet", false);
    let list = blueprint.add(
        style_component,
        crate::style::TYPE_LIST_STYLE,
        list_style(stylesheet),
    );
    // What the stylesheet will list. Numbers, unlike Pages, does not open a
    // document whose stylesheet is empty: replacing the stylesheet of a
    // document Numbers wrote with an empty one — and changing nothing else —
    // is refused.
    let mut named: Vec<(String, u64)> = vec![("text-0-liststyle-None".to_string(), list)];

    // One entry per row and per column: the default size (a literal 0 means
    // "still the table's default"), visible, and no cells yet.
    let row_bucket = blueprint.in_own_component(
        "Tables/HeaderStorageBucket",
        TYPE_HEADER_BUCKET,
        header_bucket(rows),
    );
    let column_bucket = blueprint.in_own_component(
        "Tables/HeaderStorageBucket",
        TYPE_HEADER_BUCKET,
        header_bucket(columns),
    );
    // One tile per 256 rows, each with a `TileRowInfo` per row.
    //
    // Numbers writes no `TileRowInfo` for a row with no cells, and a table made
    // that way is one no caller can then write into: `set_cell` finds a cell by
    // its row's entry, and giving a row its first entry is a job this crate has
    // not taken on. Writing the empty entries costs 522 bytes a row and makes
    // every cell of a new table writable, which is the whole point of making
    // one.
    let tiles: Vec<(u64, u64)> = (0..rows.div_ceil(TILE_SIZE as usize))
        .map(|index| {
            let first = index * TILE_SIZE as usize;
            let count = (rows - first).min(TILE_SIZE as usize);
            (
                index as u64,
                blueprint.in_own_component("Tables/Tile", TYPE_TILE, tile(count, columns)),
            )
        })
        .collect();

    // What the reduction found a Numbers root cannot do without, beyond its
    // sheets: a stylesheet, a theme, a `TSK` 205 (which is *empty* in the
    // document it was measured in), and a calculation engine.
    let sheet_style = blueprint.add(style_component, TYPE_SHEET_STYLE, sheet_style(stylesheet));
    named.push(("sheet-0-sheetStyle".to_string(), sheet_style));
    let presets = theme_presets(&mut blueprint, style_component, stylesheet, &mut named);
    let theme = blueprint.add(
        document,
        TYPE_NUMBERS_THEME,
        numbers_theme(stylesheet, presets),
    );
    let support = blueprint.add(document, TYPE_DOCUMENT_SUPPORT, message(Vec::new()));
    // The calculation engine's component, which is also where the table lives:
    // a document Numbers wrote keeps its `TST.TableInfoArchive` and
    // `TST.TableModelArchive` in `Index/CalculationEngine.iwa`, not in
    // `Document`. The engine is the component's root.
    let (engine_component, engine) = blueprint.component("CalculationEngine", false);
    blueprint.put(
        engine_component,
        engine,
        TYPE_CALCULATION_ENGINE,
        message(vec![nested(2, Vec::new())]),
    );

    // The styles the table model names. A `TST.TableModelArchive` points at a
    // table style, four cell styles (body, header row, header column, footer),
    // the text styles for the same four areas, and a shape style — and Numbers,
    // unlike Pages, does not open a document whose table names none of them.
    let table_style = blueprint.add(
        style_component,
        TYPE_TABLE_STYLE,
        named_style("table-0-tableStyle", stylesheet),
    );
    named.push(("table-0-tableStyle".to_string(), table_style));

    // Seventeen cell styles and eight paragraph styles, because that is how
    // many a `TST.TableModelArchive` names: one per area of the table (body,
    // header row, header column, footer, the category levels) and one per area
    // of text in it. They are made in a loop because what they are *for* is the
    // field that points at them; the archives themselves differ only in name.
    let table_cells: Vec<u64> = CELL_AREAS
        .iter()
        .map(|area| {
            let identifier = format!("tableCell-0-{area}");
            let style = blueprint.add(
                style_component,
                TYPE_CELL_STYLE,
                cell_style(&identifier, stylesheet),
            );
            named.push((identifier, style));
            style
        })
        .collect();
    let table_text: Vec<u64> = TEXT_AREAS
        .iter()
        .map(|area| {
            let identifier = format!("text-0-paragraphstyle-{area}");
            let style = blueprint.add(
                style_component,
                crate::style::TYPE_PARAGRAPH_STYLE,
                table_text_style(&identifier, area, stylesheet, list),
            );
            named.push((identifier, style));
            style
        })
        .collect();
    blueprint.put(
        style_component,
        stylesheet,
        TYPE_STYLESHEET,
        numbers_stylesheet(&named),
    );

    // A Numbers sheet is a page as well as a container: it carries three header
    // storages and three footer storages for printing, and a guide storage,
    // exactly as a Pages section template does. A sheet Numbers wrote has all
    // seven.

    // One number the whole table's identities come from, so that two documents
    // made in the same second are still two documents.
    let seed = {
        let hex = crate::metadata::uuid().replace('-', "");
        u64::from_str_radix(&hex[0..16], 16).unwrap_or(0x9E37_79B9_7F4A_7C15)
    };

    // A Numbers sheet is a page as well as a container: three header storages,
    // three footer storages and a guide storage, exactly as a Pages section
    // template has.
    let guides = blueprint.add(document, TYPE_GUIDE_STORAGE, message(Vec::new()));
    let headers: Vec<u64> = (0..6)
        .map(|_| {
            blueprint.add(
                document,
                TYPE_STORAGE,
                page_storage(stylesheet, table_text[1], list),
            )
        })
        .collect();

    let sheet = blueprint.allocate();
    let model = blueprint.allocate();
    let uid_map = blueprint.add(
        engine_component,
        TYPE_UID_MAP,
        column_row_uid_map(rows, columns, seed),
    );
    let info = blueprint.add(
        engine_component,
        TYPE_TABLE_INFO,
        table_info(Some(sheet), model, rows, columns, (0.0, 0.0), seed),
    );
    blueprint.put(
        engine_component,
        model,
        TYPE_TABLE_MODEL,
        table_model(TableParts {
            name: table_name,
            rows,
            columns,
            tiles: &tiles,
            row_bucket,
            column_bucket,
            strings,
            formats,
            styles,
            formulas,
            conditional,
            list_10,
            list_11,
            controls,
            table_style,
            cell_styles: &table_cells,
            text_styles: &table_text,
            uid_map,
            seed,
        }),
    );
    blueprint.put(
        document,
        sheet,
        TYPE_SHEET,
        sheet_archive(sheet_name, info, sheet_style, guides, &headers),
    );
    blueprint.put(
        document,
        ROOT,
        TYPE_NUMBERS_DOCUMENT,
        numbers_document(sheet, stylesheet, support, theme, engine),
    );
    blueprint
}

/// The `FORMAT` list of a new table: one entry, the automatic format.
///
/// A `ListEntry` is `{1: key, 2: refcount}` and a payload field, which for a
/// format list is field 6 — a `TSK.FormatStructArchive`, whose field 1 is the
/// format type. 260 is automatic, and it is what Numbers writes for a cell
/// nobody has formatted, text cells included.
fn automatic_format_list() -> Message {
    message(vec![
        varint(1, 2),
        varint(2, 2),
        nested(
            3,
            vec![varint(1, 1), varint(2, 0), nested(6, vec![varint(1, 260)])],
        ),
    ])
}

/// `TST.TableDataList` — `{1: listType, 2: nextListID}` and no entries.
fn data_list(list_type: u64) -> Message {
    message(vec![varint(1, list_type), varint(2, 1)])
}

/// `TST.HeaderStorageBucket` — one entry per row, or per column.
///
/// `{1: index, 2: size, 3: hidingState, 4: numberOfCells}`, and a size of a
/// literal `0` is what a row still at the table's default height carries. The
/// entry for a row with no cells is optional in documents Numbers writes; it is
/// written here because a row that exists and says nothing about itself is
/// harder to reason about than one that does.
fn header_bucket(count: usize) -> Message {
    let mut fields = vec![varint(1, 1)];
    for index in 0..count {
        fields.push(nested(
            2,
            vec![
                varint(1, index as u64),
                float(2, 0.0),
                varint(3, 0),
                varint(4, 0),
            ],
        ));
    }
    message(fields)
}

/// `TST.Tile` holding `rows` rows, every cell of them empty.
///
/// `TileRowInfo` fields 3 and 4 are the pre-BNC buffer and its offsets: dead
/// weight in this storage version, `required` in the schema, and present on
/// every row of every tile in the corpus. They are written because a `required`
/// field a parser cannot find is a parse error, not a default.
fn tile(rows: usize, columns: usize) -> Message {
    let mut fields = vec![
        varint(1, 0),
        varint(2, 0),
        varint(3, 0),
        varint(4, rows as u64),
    ];
    for index in 0..rows {
        fields.push(nested(
            5,
            vec![
                varint(1, index as u64),
                varint(2, 0),
                bytes(3, Vec::new()),
                bytes(4, empty_offsets(columns)),
                varint(5, 5),
                bytes(6, Vec::new()),
                bytes(7, empty_offsets(columns)),
            ],
        ));
    }
    // storage_version 5 and last_saved_in_BNC — every tile in the corpus
    // carries both, and a reader is entitled to refuse one that does not.
    fields.push(varint(6, 5));
    fields.push(varint(7, 1));
    message(fields)
}

/// A cell-offset array with every column empty.
///
/// `int16[]`, little-endian and signed, `-1` for a column with no cell. Numbers
/// pads the array well past the table's width — 255 entries for a five-column
/// table — and the padding is what leaves room for a column to be given a cell
/// later, so it is written the same way here.
fn empty_offsets(columns: usize) -> Vec<u8> {
    let slots = columns.max(OFFSET_SLOTS);
    (-1i16).to_le_bytes().repeat(slots)
}

/// How many offsets a row carries whatever its width, as Numbers writes them.
pub(crate) const OFFSET_SLOTS: usize = 255;

/// `TST.TableInfoArchive` — the drawable the table is drawn as.
///
/// Field 1 is the `TSD.DrawableArchive` every placed object begins with: a
/// geometry and the thing it hangs off, which for a Numbers table is the sheet.
fn table_info(
    parent: Option<u64>,
    model: u64,
    rows: usize,
    columns: usize,
    position: (f32, f32),
    seed: u64,
) -> Message {
    let width = columns as f32 * DEFAULT_COLUMN_WIDTH as f32;
    let height = rows as f32 * DEFAULT_ROW_HEIGHT as f32;
    let mut drawable = vec![nested(
        1,
        vec![
            nested(1, vec![float(1, position.0), float(2, position.1)]),
            nested(2, vec![float(1, width), float(2, height)]),
            varint(3, 3),
            float(4, 0.0),
        ],
    )];
    // A sheet owns its tables and is named as the parent; a Pages page owns
    // nothing, and a table floating on one names none — the same rule as
    // every other drawable.
    if let Some(parent) = parent {
        drawable.push(reference(2, parent));
    }
    message(vec![
        nested(1, drawable),
        reference(2, model),
        // `group_by_uuid` and `hidden_states_uuid`: the identities a category
        // or a filter would hang off. A table that has neither still names
        // them, and a table info without them is one the app tries to *upgrade*
        // while reading — which it then complains about, the document being
        // written in the current format already.
        nested(7, uuid_pair(seed, 7)),
        nested(8, uuid_pair(seed, 8)),
        // `formula_coord_space`.
        varint(10, 0),
    ])
}

/// Everything [`table_model`] needs, so its signature stays readable.
struct TableParts<'a> {
    name: &'a str,
    rows: usize,
    columns: usize,
    tiles: &'a [(u64, u64)],
    row_bucket: u64,
    column_bucket: u64,
    strings: u64,
    formats: u64,
    styles: u64,
    formulas: u64,
    conditional: u64,
    list_10: u64,
    list_11: u64,
    controls: u64,
    table_style: u64,
    /// Seventeen, in the order [`CELL_AREAS`] names them.
    cell_styles: &'a [u64],
    /// Eight, in the order [`TEXT_AREAS`] names them.
    text_styles: &'a [u64],
    /// `TST.ColumnRowUIDMapArchive` — the stable identity of every row and
    /// column. A table without one is a table nothing can insert a row into,
    /// because a new row's identity has nowhere to go.
    uid_map: u64,
    /// Where the table's UIDs come from.
    seed: u64,
}

/// The areas of a table that carry a cell style of their own, in the order the
/// model's fields name them.
pub(crate) const CELL_AREAS: &[&str] = &[
    "bodyStyle",
    "headerRowStyle",
    "headerColumnStyle",
    "footerRowStyle",
    "categoryLevel1Row",
    "categoryLevel2Row",
    "categoryLevel3Row",
    "categoryLevel4Row",
    "categoryLevel5Row",
    "groupLevel1Style",
    "groupLevel2Style",
    "groupLevel3Style",
    "groupLevel4Style",
    "groupLevel5Style",
    "pivotHeaderStyle",
    "pivotValueStyle",
    "pivotTotalStyle",
];

/// The same for the text in those areas.
pub(crate) const TEXT_AREAS: &[&str] = &[
    "Table Header",
    "Table Body",
    "Table Footer",
    "Table Group 1",
    "Table Group 2",
    "Table Group 3",
    "Table Group 4",
    "Table Group 5",
];

/// `TST.TableModelArchive` — the table, with its `TST.DataStore` inline.
///
/// The row/column asymmetry at the data store is Apple's: rows get a *list* of
/// bucket references and columns get a single one.
fn table_model(parts: TableParts) -> Message {
    message(vec![
        string(1, &table_id()),
        nested(
            4,
            vec![
                nested(1, vec![varint(1, 1), reference(2, parts.row_bucket)]),
                reference(2, parts.column_bucket),
                nested(3, {
                    let mut storage: Vec<Field> = parts
                        .tiles
                        .iter()
                        .map(|(index, tile)| {
                            nested(1, vec![varint(1, *index), reference(2, *tile)])
                        })
                        .collect();
                    storage.push(varint(2, TILE_SIZE));
                    storage
                }),
                reference(4, parts.strings),
                reference(5, parts.styles),
                reference(6, parts.formulas),
                varint(7, 1),
                varint(8, 0),
                nested(9, vec![nested(1, vec![varint(1, 0), varint(2, 0)])]),
                bytes(10, Vec::new()),
                reference(11, parts.conditional),
                varint(14, 4),
                reference(19, parts.list_10),
                reference(20, parts.list_11),
                reference(21, parts.controls),
                reference(22, parts.formats),
            ],
        ),
        varint(6, parts.rows as u64),
        varint(7, parts.columns as u64),
        string(8, parts.name),
        // One header row and one header column, which is what Numbers gives a
        // new table, and both frozen.
        varint(9, 1),
        varint(10, 1),
        varint(11, 0),
        varint(12, 1),
        varint(13, 1),
        double(16, DEFAULT_ROW_HEIGHT),
        double(17, DEFAULT_COLUMN_WIDTH),
        reference(3, parts.table_style),
        // The style of every area of the table, in the order the app names
        // them: four for the table proper, five per category level, five per
        // group level, three for a pivot.
        reference(18, parts.cell_styles[0]),
        reference(19, parts.cell_styles[1]),
        reference(20, parts.cell_styles[2]),
        reference(21, parts.cell_styles[3]),
        varint(22, 1),
        reference(24, parts.text_styles[0]),
        reference(25, parts.text_styles[1]),
        reference(26, parts.text_styles[1]),
        reference(27, parts.text_styles[1]),
        varint(29, 1),
        reference(30, parts.text_styles[1]),
        varint(31, 0),
        varint(32, 1),
        double(33, 0.0),
        // The table's own identity, and the identity of the thing that owns its
        // formulas. Both are minted here: nothing outside this document refers
        // to them, and the calculation engine that would is empty.
        nested(39, uid(parts.seed, 1)),
        reference(44, 0),
        reference(46, parts.uid_map),
        nested(
            47,
            vec![nested(1, uid(parts.seed, 2)), nested(2, vec![varint(2, 0)])],
        ),
        reference(60, parts.cell_styles[4]),
        reference(61, parts.cell_styles[5]),
        reference(62, parts.cell_styles[6]),
        reference(63, parts.cell_styles[7]),
        reference(64, parts.cell_styles[8]),
        reference(65, parts.text_styles[1]),
        reference(66, parts.text_styles[1]),
        reference(67, parts.text_styles[1]),
        reference(68, parts.text_styles[1]),
        reference(69, parts.text_styles[1]),
        reference(71, parts.cell_styles[9]),
        reference(72, parts.cell_styles[10]),
        reference(73, parts.cell_styles[11]),
        reference(74, parts.cell_styles[12]),
        reference(75, parts.cell_styles[13]),
        reference(76, parts.text_styles[3]),
        reference(77, parts.text_styles[4]),
        reference(78, parts.text_styles[5]),
        reference(79, parts.text_styles[6]),
        reference(80, parts.text_styles[7]),
        nested(
            81,
            vec![
                // `CategoryOwnerArchive.owner_uid`, a `TSP.UUID` whose `lower`
                // and `upper` are **required** — the app says so by name in
                // the unified log when they are not there.
                nested(1, vec![varint(1, 0), varint(2, 0)]),
                nested(
                    2,
                    vec![
                        nested(1, uuid_pair(parts.seed, 3)),
                        nested(
                            3,
                            vec![nested(1, vec![varint(1, 1), varint(2, 0)]), string(6, "")],
                        ),
                        varint(6, 0),
                        nested(7, vec![varint(2, 0), varint(3, 0)]),
                        nested(8, vec![varint(2, 1), varint(3, 0)]),
                        nested(9, vec![varint(2, 3), varint(3, 0)]),
                        nested(10, vec![varint(2, 2), varint(3, 0)]),
                        nested(11, vec![varint(2, 4), varint(3, 0)]),
                        nested(12, vec![varint(2, 5), varint(3, 0)]),
                        nested(13, vec![varint(2, 6), varint(3, 0)]),
                        varint(14, 8),
                        nested(16, vec![varint(2, 7), varint(3, 0)]),
                    ],
                ),
            ],
        ),
        nested(
            82,
            vec![nested(1, uid(parts.seed, 4)), nested(2, vec![varint(2, 0)])],
        ),
        nested(84, vec![nested(1, uuid_pair(parts.seed, 5))]),
        reference(87, parts.cell_styles[14]),
        reference(88, parts.cell_styles[15]),
        reference(89, parts.cell_styles[16]),
        nested(93, vec![nested(1, uuid_pair(parts.seed, 6))]),
    ])
}

/// A four-word `TSCE` UID, derived from the table's seed so that a document is
/// internally consistent and two documents do not share one.
fn uid(seed: u64, which: u64) -> Vec<Field> {
    let word = |n: u64| ((seed.wrapping_mul(0x9E37_79B9) ^ (which << 8) ^ n) & 0xFFFF_FFFF) + 1;
    vec![
        varint(2, word(1)),
        varint(3, word(2)),
        varint(4, word(3)),
        varint(5, word(4)),
    ]
}

/// A two-word `TSP.UUID`, from the same seed.
fn uuid_pair(seed: u64, which: u64) -> Vec<Field> {
    let word = |n: u64| seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (which << 32) ^ n;
    vec![varint(1, word(1)), varint(2, word(2))]
}

/// A paragraph style for one area of a table: bold in a header, plain in the
/// body, 10 pt throughout, which is what the app's own table styles carry.
fn table_text_style(identifier: &str, name: &str, stylesheet: u64, list: u64) -> Message {
    use crate::style::property;
    message(vec![
        nested(
            1,
            vec![
                string(1, name),
                string(2, identifier),
                reference(5, stylesheet),
            ],
        ),
        nested(
            property::FONT_SIZE[0],
            vec![
                float(property::FONT_SIZE[1], 10.0),
                string(
                    property::FONT_NAME[1],
                    match name.contains("Header") {
                        true => "HelveticaNeue-Bold",
                        false => "HelveticaNeue",
                    },
                ),
                nested(property::FONT_COLOR[1], black()),
            ],
        ),
        nested(
            property::ALIGNMENT[0],
            vec![
                varint(property::ALIGNMENT[1], 4),
                reference(property::LIST_STYLE[1], list),
            ],
        ),
    ])
}

/// A style that is nothing but a name and the stylesheet it belongs to.
fn named_style(identifier: &str, stylesheet: u64) -> Message {
    message(vec![nested(
        1,
        vec![string(2, identifier), reference(5, stylesheet)],
    )])
}

/// `TST.CellStyleArchive` — a cell's padding and its fill.
fn cell_style(identifier: &str, stylesheet: u64) -> Message {
    message(vec![
        nested(1, vec![string(2, identifier), reference(5, stylesheet)]),
        nested(
            11,
            vec![
                bytes(1, Vec::new()),
                varint(3, 1),
                varint(8, 0),
                // The four insets Numbers gives a cell, in points.
                nested(
                    9,
                    vec![float(1, 4.0), float(2, 4.0), float(3, 4.0), float(4, 4.0)],
                ),
            ],
        ),
    ])
}

/// `TST.TableStyleArchive`.
pub(crate) const TYPE_TABLE_STYLE: u32 = 6003;
/// `TST.CellStyleArchive`.
pub(crate) const TYPE_CELL_STYLE: u32 = 6004;
/// `TSWP.ShapeStyleArchive`.
pub(crate) const TYPE_SHAPE_STYLE: u32 = 2025;

/// `TST.TableModelArchive.table_id` — an uppercase UUID, as Numbers writes it.
fn table_id() -> String {
    crate::metadata::uuid()
}

/// `TN.SheetArchive` — a name, what is on it, and how it is laid out.
///
/// The layout fields are the ones a sheet Numbers made carries, kept because a
/// sheet is a *page* as well as a container: field 7 is its zoom, 13 and 14 its
/// print margins, 22 the style it is drawn with.
fn sheet_archive(name: &str, table: u64, style: u64, guides: u64, headers: &[u64]) -> Message {
    let mut fields = vec![
        string(1, name),
        reference(2, table),
        varint(3, 1),
        varint(5, 1),
        varint(6, 0),
        float(7, 0.72),
        varint(8, 0),
        varint(11, 0),
        varint(12, 1),
        float(13, 20.0),
        float(14, 20.0),
        varint(20, 0),
        varint(21, 0),
        reference(22, style),
        varint(23, 1),
        varint(24, 0),
    ];
    fields.push(reference(17, guides));
    for header in headers.iter().take(3) {
        fields.push(reference(18, *header));
    }
    for footer in headers.iter().skip(3) {
        fields.push(reference(19, *footer));
    }
    message(fields)
}

/// One of a sheet's six header and footer storages: empty, and pointing at the
/// styles that say what it would look like if it held anything.
fn page_storage(stylesheet: u64, paragraph: u64, list: u64) -> Message {
    message(vec![
        // kind 1: a header.
        varint(1, 1),
        reference(2, stylesheet),
        attribute_table(5, paragraph),
        nested(
            6,
            vec![nested(1, vec![varint(1, 0), varint(2, 0), varint(3, 0)])],
        ),
        attribute_table(7, list),
        varint(10, 1),
        nested(
            14,
            vec![nested(1, vec![varint(1, 0), varint(2, 0), varint(3, 0)])],
        ),
        nested(
            24,
            vec![nested(1, vec![varint(1, 0), varint(2, 0), varint(3, 0)])],
        ),
    ])
}

/// `TSD.GuideStorageArchive` — a sheet's user guides, of which a new sheet has
/// none.
const TYPE_GUIDE_STORAGE: u32 = 3047;

/// `TN.SheetStyleArchive` — a white sheet.
fn sheet_style(stylesheet: u64) -> Message {
    message(vec![
        nested(
            1,
            vec![string(2, "sheet-0-sheetStyle"), reference(5, stylesheet)],
        ),
        varint(2, 2),
        nested(
            3,
            vec![nested(1, vec![nested(1, white())]), bytes(2, Vec::new())],
        ),
    ])
}

/// `TSP.Color`, white and opaque.
fn white() -> Vec<Field> {
    vec![
        varint(1, 1),
        float(3, 1.0),
        float(4, 1.0),
        float(5, 1.0),
        float(6, 1.0),
        varint(12, 1),
        float(13, 1.0),
    ]
}

/// `TN.SheetStyleArchive`.
const TYPE_SHEET_STYLE: u32 = 12050;

/// `TN.DocumentArchive` — object 1, and type 1.
///
/// Every reference here is one the reduction could not delete: a Numbers
/// document with no stylesheet, no theme, no `TSK` 205 or no calculation
/// engine is refused, where a Pages document needs none of the four.
fn numbers_document(sheet: u64, stylesheet: u64, support: u64, theme: u64, engine: u64) -> Message {
    message(vec![
        reference(1, sheet),
        reference(4, stylesheet),
        reference(5, support),
        reference(6, theme),
        nested(8, vec![nested(1, Vec::new()), reference(4, engine)]),
    ])
}

/// `TSS.StylesheetArchive` as Numbers writes one: every style twice.
///
/// Field 1 lists the styles as plain references; field 2 lists them again as
/// `{identifier, style}`, which is how the app looks one up by name. Pages
/// keeps the same pair nested under field 8 and opens a document with neither;
/// Numbers keeps them at the top level and does not.
fn numbers_stylesheet(named: &[(String, u64)]) -> Message {
    let mut fields: Vec<Field> = named
        .iter()
        .map(|(_, style)| reference(1, *style))
        .collect();
    for (identifier, style) in named {
        fields.push(nested(2, vec![string(1, identifier), reference(2, *style)]));
    }
    // `is_locked`, and its default is **true**. A document whose stylesheet
    // does not say otherwise is one the app cannot add a style to, and it needs
    // to: opening this document without the field, Numbers got as far as
    // "Adding style (TSDMediaStyle*) to locked stylesheet" and stopped. Nothing
    // in the file says that; the app's own log does.
    fields.push(varint(4, 0));
    message(fields)
}

/// The style presets a theme carries, and the objects they name.
///
/// `TSD.ThemePresetsArchive`, field 100 of the theme: six line styles, six
/// shape styles, one text-box style, six image styles, six movie styles and one
/// drawing line style. They are not decoration either — a theme without them
/// sends the app looking for `presetOfKind:index:` at load, and what it says
/// then is
///
/// ```text
/// Attempt to request TSSLineStylePresetKind preset for out of bounds index 0.
/// invalid nil value for 'presetStyle'
/// Failed to initialize object context
/// ```
///
/// A line, shape, text-box or drawing-line preset is a `TSWP.ShapeStyleArchive`
/// — whose `super` is a `TSD.ShapeStyleArchive` whose `super` is the
/// `TSS.StyleArchive`, two levels down — and an image or movie preset is a
/// `TSD.MediaStyleArchive`, one level down. Getting that wrong is what
/// "missing required fields: super.super" means.
fn theme_presets(
    blueprint: &mut Blueprint,
    component: ComponentId,
    stylesheet: u64,
    named: &mut Vec<(String, u64)>,
) -> Field {
    let shape =
        |blueprint: &mut Blueprint, named: &mut Vec<(String, u64)>, kind: &str, index: usize| {
            let identifier = format!("{kind}-preset-{index}");
            let style = blueprint.add(
                component,
                TYPE_SHAPE_STYLE,
                message(vec![nested(
                    1,
                    vec![
                        nested(1, vec![string(2, &identifier), reference(5, stylesheet)]),
                        // The property bag, carrying a fill that is present
                        // and paints nothing — what a shape style in a deck
                        // Keynote wrote carries, and the only place a colour
                        // written later has to go.
                        nested(11, vec![bytes(1, Vec::new())]),
                    ],
                )]),
            );
            named.push((identifier, style));
            style
        };
    let mut presets = Vec::new();
    for index in 0..6 {
        presets.push(reference(4, shape(blueprint, named, "line-style", index)));
    }
    for index in 0..6 {
        presets.push(reference(5, shape(blueprint, named, "shape-style", index)));
    }
    presets.push(reference(6, shape(blueprint, named, "textbox-style", 0)));
    let media =
        |blueprint: &mut Blueprint, named: &mut Vec<(String, u64)>, kind: &str, index: usize| {
            let identifier = format!("{kind}-preset-{index}");
            let style = blueprint.add(
                component,
                TYPE_MEDIA_STYLE,
                message(vec![
                    nested(1, vec![string(2, &identifier), reference(5, stylesheet)]),
                    // The same bag one level up, numbered one lower: a media
                    // style has no fill, so field 1 here is the stroke.
                    nested(11, vec![bytes(1, Vec::new())]),
                ]),
            );
            named.push((identifier, style));
            style
        };
    for index in 0..6 {
        presets.push(reference(7, media(blueprint, named, "image-style", index)));
    }
    for index in 0..6 {
        presets.push(reference(8, media(blueprint, named, "movie-style", index)));
    }
    presets.push(reference(
        9,
        shape(blueprint, named, "drawing-line-style", 0),
    ));
    nested(100, presets)
}

/// `TSD.MediaStyleArchive`.
const TYPE_MEDIA_STYLE: u32 = 3016;

/// The theme: a stylesheet, and the palette every style picks its colours from.
///
/// The palette is not decoration. Reducing a theme Numbers wrote gives up its
/// name, its default-style map and its font map, and stops at **27 colours** —
/// three of the thirty go and the rest do not. A palette is indexed by
/// position, which is the obvious reason a shorter one would not do, so this
/// writes the count the app was watched insisting on.
fn numbers_theme(stylesheet: u64, presets: Field) -> Message {
    numbers_theme_with(stylesheet, presets, Vec::new())
}

/// The same, with whatever the app puts outside the `TSS.ThemeArchive` — for
/// Keynote, the master slides.
fn numbers_theme_with(stylesheet: u64, presets: Field, extra: Vec<Field>) -> Message {
    let mut theme = vec![reference(4, stylesheet), presets];
    for (red, green, blue) in PALETTE {
        theme.push(nested(
            10,
            vec![
                varint(1, 1),
                float(3, *red),
                float(4, *green),
                float(5, *blue),
                float(6, 1.0),
                varint(12, 1),
                float(13, 1.0),
            ],
        ));
    }
    let mut fields = vec![nested(1, theme)];
    fields.extend(extra);
    message(fields)
}

/// Twenty-seven colours: a greyscale ramp and two rows of hues, which is the
/// shape of the palette the apps ship.
const PALETTE: &[(f32, f32, f32)] = &[
    (1.0, 1.0, 1.0),
    (0.84, 0.84, 0.84),
    (0.57, 0.57, 0.57),
    (0.37, 0.37, 0.37),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.34, 0.76, 1.0),
    (0.0, 0.63, 1.0),
    (0.0, 0.46, 0.73),
    (0.0, 0.32, 0.51),
    (0.85, 0.33, 0.31),
    (0.92, 0.49, 0.19),
    (0.95, 0.76, 0.20),
    (0.44, 0.68, 0.28),
    (0.27, 0.45, 0.77),
    (0.51, 0.30, 0.63),
    (0.75, 0.31, 0.51),
    (0.40, 0.40, 0.40),
    (0.65, 0.65, 0.65),
    (0.90, 0.90, 0.90),
    (0.98, 0.85, 0.85),
    (0.85, 0.98, 0.85),
    (0.85, 0.85, 0.98),
    (0.98, 0.98, 0.85),
    (0.85, 0.98, 0.98),
    (0.98, 0.85, 0.98),
    (0.50, 0.50, 0.50),
];

/// `TN.ThemeArchive`.
const TYPE_NUMBERS_THEME: u32 = 12009;
/// `TSK` 205 — empty in every document measured, and required all the same.
const TYPE_DOCUMENT_SUPPORT: u32 = 205;
/// `TSCE.CalculationEngineArchive`.
const TYPE_CALCULATION_ENGINE: u32 = 4000;

// -- Keynote ------------------------------------------------------------------

/// `KN.DocumentArchive` — the root of a deck, and object 1.
const TYPE_KEYNOTE_DOCUMENT: u32 = 1;
/// `KN.ShowArchive`.
const TYPE_SHOW: u32 = 2;
/// `KN.SlideNodeArchive` — a slide's place in the show's tree.
const TYPE_SLIDE_NODE: u32 = 4;
/// `KN.SlideArchive`.
const TYPE_SLIDE: u32 = 5;
/// `KN.SlideStyleArchive`.
const TYPE_SLIDE_STYLE: u32 = 9;
/// `KN.ThemeArchive`.
const TYPE_KEYNOTE_THEME: u32 = 10;

/// A Keynote deck with one empty slide, and the master it is drawn from.
///
/// **What makes a slide archive a master is its `name`.** The deck died for an
/// afternoon on
///
/// ```text
/// Caught NSInvalidArgumentException while running finalize handler:
/// -[KNSlide generateObjectPlaceholderIfNecessary]: unrecognized selector
/// ```
///
/// — the app holding a show slide where it wanted a master — and none of the
/// obvious answers was the answer. Not the message type: 5 and 6 carry the same
/// message and Keynote's own decks write masters at 5. Not `inDocument`, true
/// on both. Not the component name, `TemplateSlide` against `Slide`, which this
/// already wrote. Not the placeholders. What separates a master from a show
/// slide in every deck in the corpus is that a master has `name` (field 10) and
/// names no `template_slide`, and a show slide is the other way round — and
/// giving the master a name is what made Keynote open this.
///
/// The schemas carved out of 15.3.1 (`reference/protos-15.3`) name the required
/// fields, so this is the one of the three written by reading rather than by
/// deleting: a `KN.ShowArchive` needs its theme, its slide tree, its size and
/// its stylesheet; a `KN.SlideNodeArchive` needs to say whether it is skipped,
/// has builds and has a transition; a `KN.SlideArchive` needs a style, a
/// transition and to say it is in the document.
pub(crate) fn keynote(slide_size: (f32, f32)) -> Blueprint {
    let mut blueprint = Blueprint::new(Kind::Keynote);
    let document = blueprint.document();
    let seed = {
        let hex = crate::metadata::uuid().replace('-', "");
        u64::from_str_radix(&hex[0..16], 16).unwrap_or(0x9E37_79B9_7F4A_7C15)
    };
    let (style_component, stylesheet) = blueprint.component("DocumentStylesheet", false);

    let slide_style = blueprint.add(
        style_component,
        TYPE_SLIDE_STYLE,
        named_style("slide-style-default", stylesheet),
    );
    // The two styles any text on a slide needs: what a paragraph looks like,
    // and the list it is not in.
    let list = blueprint.add(
        style_component,
        crate::style::TYPE_LIST_STYLE,
        list_style(stylesheet),
    );
    let body = blueprint.add(
        style_component,
        crate::style::TYPE_PARAGRAPH_STYLE,
        paragraph_style(stylesheet, list),
    );
    let mut named = vec![
        ("slide-style-default".to_string(), slide_style),
        ("text-0-liststyle-None".to_string(), list),
        (BODY_IDENTIFIER.to_string(), body),
    ];

    // The styles a *table* is drawn with, which a deck carries whether or not
    // it has one — `keynote-deck.key` holds 102 cell styles and no table. They
    // are objects and are not listed in the sheet: `add_table` finds them by
    // the names they carry, and listing them is what made Pages refuse a
    // document outright.
    blueprint.add(
        style_component,
        TYPE_TABLE_STYLE,
        named_style("table-0-tableStyle", stylesheet),
    );
    for area in CELL_AREAS {
        blueprint.add(
            style_component,
            TYPE_CELL_STYLE,
            cell_style(&format!("tableCell-0-{area}"), stylesheet),
        );
    }
    for area in TEXT_AREAS {
        blueprint.add(
            style_component,
            crate::style::TYPE_PARAGRAPH_STYLE,
            table_text_style(
                &format!("text-0-paragraphstyle-{area}"),
                area,
                stylesheet,
                list,
            ),
        );
    }

    // …and the calculation engine, which every deck the app makes carries and
    // which is the component a table's model and info live in.
    let (engine_component, engine) = blueprint.component("CalculationEngine", false);
    blueprint.put(
        engine_component,
        engine,
        TYPE_CALCULATION_ENGINE,
        message(vec![nested(2, Vec::new())]),
    );

    let presets = theme_presets(&mut blueprint, style_component, stylesheet, &mut named);
    let theme = blueprint.allocate();
    blueprint.put(
        style_component,
        stylesheet,
        TYPE_STYLESHEET,
        numbers_stylesheet(&named),
    );
    // Both slides live in components of their own: a slide node holds a *lazy*
    // reference to its slide, and a slide in `Document` is one the app reports
    // as "Failed to load lazy slide reference".
    //
    // The template slide is the master this one is drawn from. A deck without
    // one loads and then says "invalid nil value for 'masterSlide'".
    // The master's object placeholder — the frame a slide's own content lands
    // in. A master without one sends the app looking for
    // `generateObjectPlaceholderIfNecessary`, which is a method the show-slide
    // class does not have, and the deck dies on the selector.
    let (template_component, template) = blueprint.component("TemplateSlide", true);
    // All three of them: a master has a title, a body and an object
    // placeholder, and a slide drawn from it inherits the frames.
    let placeholders: Vec<u64> = [2u64, 3, 4]
        .iter()
        .map(|kind| {
            blueprint.add(
                template_component,
                TYPE_PLACEHOLDER,
                object_placeholder(template, slide_size, *kind),
            )
        })
        .collect();
    blueprint.put(
        template_component,
        template,
        TYPE_SLIDE,
        master_archive(slide_style, &placeholders),
    );
    let template_node = blueprint.add(
        document,
        TYPE_SLIDE_NODE,
        keynote_slide_node(template, seed),
    );
    let slide = blueprint.in_own_component(
        "Slide",
        TYPE_SLIDE,
        keynote_slide(slide_style, Some(template)),
    );
    let node = blueprint.add(document, TYPE_SLIDE_NODE, keynote_slide_node(slide, seed));
    blueprint.put(
        document,
        theme,
        TYPE_KEYNOTE_THEME,
        numbers_theme_with(
            stylesheet,
            presets,
            vec![
                // `templates`, and the one to draw a new slide from.
                reference(2, template_node),
                string(3, &crate::metadata::uuid()),
                reference(5, template_node),
                reference(6, template_node),
            ],
        ),
    );
    let show = blueprint.add(
        document,
        TYPE_SHOW,
        show_archive(theme, stylesheet, node, slide_size),
    );
    blueprint.put(
        document,
        ROOT,
        TYPE_KEYNOTE_DOCUMENT,
        message(vec![
            reference(2, show),
            nested(3, vec![nested(1, Vec::new())]),
        ]),
    );
    blueprint
}

/// `KN.ShowArchive` — everything about the deck that is not a slide.
fn show_archive(theme: u64, stylesheet: u64, node: u64, size: (f32, f32)) -> Message {
    message(vec![
        reference(2, theme),
        // The slide tree: `{2: repeated slide node}`.
        nested(3, vec![reference(2, node)]),
        nested(4, vec![float(1, size.0), float(2, size.1)]),
        reference(5, stylesheet),
    ])
}

/// `KN.SlideNodeArchive` — where a slide sits in the show.
///
/// Three of these are `required` in the schema and two more are `optional` and
/// insisted on anyway: Keynote's own loader says "Missing isSlideNumberVisible
/// on slide node" and "Slide background alpha expected in document saved at or
/// after version …" for a node that leaves out 18 and 28. An optional field
/// with a default is not always a field you may omit.
pub(crate) fn keynote_slide_node(slide: u64, seed: u64) -> Message {
    message(vec![
        reference(2, slide),
        // `isSkipped`, `hasBuilds`, `hasTransition`, `hasNote`.
        varint(4, 0),
        varint(6, 0),
        varint(7, 0),
        varint(8, 0),
        varint(14, 1),
        // `isSlideNumberVisible`.
        varint(18, 0),
        varint(20, 0),
        // `depth`: a slide at the top level of the outline.
        varint(21, 1),
        varint(26, 4294967295),
        varint(27, 2),
        // `background_is_no_fill_or_color_fill_with_alpha`.
        varint(28, 0),
        // `template_slide_id`.
        nested(29, uuid_pair(seed, 9)),
    ])
}

/// The master slide: a style, a transition, and the object placeholder every
/// slide drawn from it inherits.
fn master_archive(style: u64, placeholders: &[u64]) -> Message {
    message(vec![
        reference(1, style),
        nested(4, vec![nested(2, Vec::new())]),
        // Title, body and object, at the fields that name each.
        reference(5, placeholders[0]),
        reference(6, placeholders[1]),
        reference(30, placeholders[2]),
        // A master has a name and a show slide does not, which is one of the
        // two things that tell them apart in a document Keynote wrote — the
        // other being that a show slide names its `template_slide` and a master
        // names none.
        string(10, "Title"),
        varint(19, 1),
        varint(41, 0),
    ])
}

/// `KN.PlaceholderArchive` for the object placeholder: a shape covering the
/// slide, and nothing in it.
///
/// Four archives deep — `KN.Placeholder` over `TSWP.ShapeInfo` over `TSD.Shape`
/// over `TSD.Drawable` — because each one's `super` is `required` and the app
/// says so by name when it is not there.
fn object_placeholder(slide: u64, size: (f32, f32), kind: u64) -> Message {
    message(vec![
        nested(
            1,
            vec![nested(
                1,
                vec![nested(
                    1,
                    vec![
                        nested(
                            1,
                            vec![
                                nested(1, vec![float(1, 0.0), float(2, 0.0)]),
                                nested(2, vec![float(1, size.0), float(2, size.1)]),
                                varint(3, 0),
                                float(4, 0.0),
                            ],
                        ),
                        reference(2, slide),
                    ],
                )],
            )],
        ),
        // 2 title, 3 body, 4 object.
        varint(2, kind),
    ])
}

/// `KN.PlaceholderArchive`.
const TYPE_PLACEHOLDER: u32 = 7;

/// `KN.SlideArchive` — the slide itself, with nothing on it.
///
/// `template_slide` is what makes it a slide rather than a master: a slide in
/// the show names the master it is drawn from, and the master names none.
pub(crate) fn keynote_slide(style: u64, template: Option<u64>) -> Message {
    let mut fields = vec![
        reference(1, style),
        // A transition, which every slide has whether or not it does anything.
        nested(4, vec![nested(2, Vec::new())]),
        // `inDocument`, and it is true on a master as well — what tells the
        // two apart is the component each lives in, `Slide-1031` against
        // `TemplateSlide-1029`.
        varint(19, 1),
    ];
    if let Some(template) = template {
        fields.push(reference(17, template));
    }
    message(fields)
}

/// Somewhere objects can be put: a package being assembled, or a document being
/// grown.
///
/// The archives are the same either way — a tile is a tile — and the only thing
/// that differs is where the objects land and how the component index hears
/// about them. Everything that builds more than one object goes through this,
/// so a table added to a spreadsheet is the table `Document::new` makes.
pub(crate) trait Site {
    fn allocate(&mut self) -> u64;
    /// A component holding the objects given, the first of which is its root.
    fn component(
        &mut self,
        name: &str,
        numbered: bool,
        objects: Vec<(u64, u32, Message)>,
    ) -> Result<u64, crate::Error>;
}

// -- growing a document that already exists -----------------------------------

/// Adding objects and components to a document that is already there.
///
/// [`Blueprint`] assembles a package from nothing, where every identifier is
/// free and the component index is written once at the end. Adding a sheet to a
/// spreadsheet somebody else made is the same job under three constraints: the
/// identifiers have to be ones the document has not used, each new component
/// needs its `TSP.ComponentInfo` written into an index that already exists, and
/// the high-water mark has to move so the app does not hand out an identifier
/// this crate has just taken.
///
/// What it does not do is decide *what* to add — that is the caller's, and the
/// archives come from the same functions [`Blueprint`] uses.
pub(crate) struct Grow<'a> {
    document: &'a mut crate::Document,
    next: u64,
    /// Components made here, so `finish` can register them in one pass.
    added: Vec<NewComponent>,
    /// Objects added to components that already existed, as
    /// `(component root, object)` — they need object-UUID entries too.
    joined: Vec<(u64, u64)>,
}

struct NewComponent {
    identifier: u64,
    name: String,
    locator: String,
    objects: Vec<u64>,
}

impl Site for Grow<'_> {
    fn allocate(&mut self) -> u64 {
        Grow::allocate(self)
    }

    fn component(
        &mut self,
        name: &str,
        numbered: bool,
        objects: Vec<(u64, u32, Message)>,
    ) -> Result<u64, crate::Error> {
        Grow::component(self, name, numbered, objects)
    }
}

impl Site for Blueprint {
    fn allocate(&mut self) -> u64 {
        Blueprint::allocate(self)
    }

    fn component(
        &mut self,
        name: &str,
        numbered: bool,
        objects: Vec<(u64, u32, Message)>,
    ) -> Result<u64, crate::Error> {
        let root = objects
            .first()
            .map(|(identifier, _, _)| *identifier)
            .ok_or_else(|| crate::Error::Format("a component needs a root object".into()))?;
        let component = self.push_component(
            name,
            root,
            numbered.then(|| format!("{name}-{root}")),
            false,
        );
        for (identifier, message_type, archive) in objects {
            self.put(component, identifier, message_type, archive);
        }
        Ok(root)
    }
}

impl<'a> Grow<'a> {
    pub(crate) fn new(document: &'a mut crate::Document) -> Grow<'a> {
        let next = document.next_object_identifier();
        Grow {
            document,
            next,
            added: Vec::new(),
            joined: Vec::new(),
        }
    }

    pub(crate) fn allocate(&mut self) -> u64 {
        let identifier = self.next;
        self.next += 1;
        identifier
    }

    /// Put an object into the stream a neighbour is already in.
    pub(crate) fn beside(
        &mut self,
        neighbour: u64,
        identifier: u64,
        message_type: u32,
        archive: &Message,
    ) -> Result<(), crate::Error> {
        self.beside_with(neighbour, identifier, message_type, archive, &[])
    }

    /// The same, for an object that names media: the identifiers go into the
    /// object's own `data_references`, which is how the app knows what to load.
    pub(crate) fn beside_with(
        &mut self,
        neighbour: u64,
        identifier: u64,
        message_type: u32,
        archive: &Message,
        data_references: &[u64],
    ) -> Result<(), crate::Error> {
        self.document.add_object_after_with(
            neighbour,
            identifier,
            message_type,
            archive,
            data_references,
        )?;
        if let Some(root) = self.component_of(neighbour) {
            self.joined.push((root, identifier));
        }
        Ok(())
    }

    /// A new component holding the objects given, the first of which is its
    /// root — and whose identifier is therefore the component's.
    ///
    /// `numbered` names the stream `Tables/Tile-1007` rather than `Tables/Tile`,
    /// which is what the apps do for everything they may have more than one of.
    pub(crate) fn component(
        &mut self,
        name: &str,
        numbered: bool,
        objects: Vec<(u64, u32, Message)>,
    ) -> Result<u64, crate::Error> {
        let root = objects
            .first()
            .map(|(identifier, _, _)| *identifier)
            .ok_or_else(|| crate::Error::Format("a component needs a root object".into()))?;
        let locator = match numbered {
            true => format!("{name}-{root}"),
            false => name.to_string(),
        };
        let stream = format!("Index/{locator}.iwa");
        let identifiers: Vec<u64> = objects.iter().map(|(id, _, _)| *id).collect();
        let objects: Vec<ArchiveObject> = objects
            .into_iter()
            .map(|(identifier, message_type, archive)| ArchiveObject {
                identifier,
                messages: vec![ArchiveMessage {
                    message_type,
                    version: VERSION.to_vec(),
                    extra: Vec::new(),
                    payload: archive.encode(),
                }],
                extra: Vec::new(),
            })
            .collect();
        self.document.add_stream(&stream, objects)?;
        self.added.push(NewComponent {
            identifier: root,
            name: name.to_string(),
            locator,
            objects: identifiers,
        });
        Ok(root)
    }

    /// Which component an object lives in, by its stream.
    fn component_of(&self, object: u64) -> Option<u64> {
        let (stream, _) = self.document.object(object)?;
        self.document
            .components()
            .into_iter()
            .find(|component| component.stream_name() == stream)
            .map(|component| component.identifier)
    }

    /// Register the new components, give every new object a UUID, raise the
    /// high-water mark and declare what the new objects point at.
    ///
    /// Returns the number of declarations added, which is what a caller reports.
    pub(crate) fn finish(self) -> Result<usize, crate::Error> {
        let Grow {
            document,
            next,
            added,
            joined,
        } = self;
        document.set_last_object_identifier(next.saturating_sub(1))?;

        let kind = document.kind();
        document.update_package_metadata(|metadata| {
            for component in &added {
                let mut info = vec![varint(1, component.identifier), string(2, &component.name)];
                if component.locator != component.name {
                    info.push(string(3, &component.locator));
                }
                info.push(Field {
                    number: 4,
                    value: Value::Bytes(COMPONENT_VERSION.to_vec()),
                });
                info.push(Field {
                    number: 5,
                    value: Value::Bytes(COMPONENT_VERSION.to_vec()),
                });
                for object in &component.objects {
                    info.push(object_uuid_entry(*object));
                }
                info.push(varint(10, 0));
                info.push(varint(12, component_generation(kind)));
                metadata.append_in_order(3, Value::Bytes(message(info).encode()));
            }

            // Objects that joined a component that was already there: the
            // component keeps its entry, and gains a UUID for each of them.
            for (root, object) in &joined {
                let mut rewritten = Vec::new();
                for value in metadata.all(3) {
                    let Value::Bytes(raw) = value else { continue };
                    let Ok(mut info) = Message::decode(raw) else {
                        continue;
                    };
                    if info.varint(1) != Some(*root) {
                        continue;
                    }
                    info.fields.push(object_uuid_entry(*object));
                    rewritten.push((raw.clone(), info.encode()));
                }
                for (before, after) in rewritten {
                    for field in metadata.fields.iter_mut() {
                        if field.number == 3 && field.value == Value::Bytes(before.clone()) {
                            field.value = Value::Bytes(after.clone());
                            break;
                        }
                    }
                }
            }
        })?;
        Ok(document.declare_external_references())
    }
}

/// The styles a table model names, however they were come by.
pub(crate) struct TableStyles {
    pub(crate) table: u64,
    /// Seventeen, in the order [`CELL_AREAS`] names them.
    pub(crate) cells: Vec<u64>,
    /// Eight, in the order [`TEXT_AREAS`] names them.
    pub(crate) text: Vec<u64>,
}

/// Build a table: its tiles, its interning lists, its header buckets, its model
/// and the drawable the sheet holds it by.
///
/// Returns `(table info, table model)`. Everything that can be its own
/// component is one, because Numbers refuses a document whose tile is not — see
/// FORMAT.md §14 — and the model and the info are handed back for the caller to
/// place, since where *they* go differs: a new document puts them in the
/// calculation engine's component, and a document being grown puts them beside
/// the table that is already there.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_table(
    site: &mut impl Site,
    parent: Option<u64>,
    name: &str,
    rows: usize,
    columns: usize,
    styles: &TableStyles,
    seed: u64,
    position: (f32, f32),
) -> Result<BuiltTable, crate::Error> {
    let data_list_component =
        |site: &mut dyn Site, archive: Message| -> Result<u64, crate::Error> {
            let identifier = site.allocate();
            site.component(
                "Tables/DataList",
                true,
                vec![(identifier, TYPE_DATA_LIST, archive)],
            )
        };
    let strings = data_list_component(site, data_list(1))?;
    let formats = data_list_component(site, automatic_format_list())?;
    let styles_list = data_list_component(site, data_list(4))?;
    let formulas = data_list_component(site, data_list(3))?;
    let conditional = data_list_component(site, data_list(2))?;
    let list_10 = data_list_component(site, data_list(10))?;
    let list_11 = data_list_component(site, data_list(11))?;
    let controls = data_list_component(site, data_list(12))?;

    let bucket = |site: &mut dyn Site, count: usize| -> Result<u64, crate::Error> {
        let identifier = site.allocate();
        site.component(
            "Tables/HeaderStorageBucket",
            true,
            vec![(identifier, TYPE_HEADER_BUCKET, header_bucket(count))],
        )
    };
    let row_bucket = bucket(site, rows)?;
    let column_bucket = bucket(site, columns)?;

    let mut tiles: Vec<(u64, u64)> = Vec::new();
    for index in 0..rows.div_ceil(TILE_SIZE as usize) {
        let first = index * TILE_SIZE as usize;
        let count = (rows - first).min(TILE_SIZE as usize);
        let identifier = site.allocate();
        let tile = site.component(
            "Tables/Tile",
            true,
            vec![(identifier, TYPE_TILE, tile(count, columns))],
        )?;
        tiles.push((index as u64, tile));
    }

    let model = site.allocate();
    let info = site.allocate();
    let uid_map = site.allocate();
    let model_archive = table_model(TableParts {
        name,
        rows,
        columns,
        tiles: &tiles,
        row_bucket,
        column_bucket,
        strings,
        formats,
        styles: styles_list,
        formulas,
        conditional,
        list_10,
        list_11,
        controls,
        table_style: styles.table,
        cell_styles: &styles.cells,
        text_styles: &styles.text,
        uid_map,
        seed,
    });
    let info_archive = table_info(parent, model, rows, columns, position, seed);
    Ok(BuiltTable {
        info,
        model,
        uid_map,
        info_archive,
        model_archive,
        uid_map_archive: column_row_uid_map(rows, columns, seed),
    })
}

/// What [`build_table`] makes: three objects, and the archives to store them
/// with. The caller decides which stream they land in.
pub(crate) struct BuiltTable {
    pub(crate) info: u64,
    pub(crate) model: u64,
    pub(crate) uid_map: u64,
    pub(crate) info_archive: Message,
    pub(crate) model_archive: Message,
    pub(crate) uid_map_archive: Message,
}

/// A whole message as a length-delimited field.
pub(crate) fn nested_message(number: u32, inner: Message) -> Field {
    Field {
        number,
        value: Value::Bytes(inner.encode()),
    }
}

/// A `TSP.Reference` to one object, encoded — for a caller appending one to a
/// repeated field.
pub(crate) fn reference_bytes(target: u64) -> Vec<u8> {
    message(vec![varint(1, target)]).encode()
}

/// One `object_uuid_map_entries` entry: an object and a UUID of its own.
fn object_uuid_entry(object: u64) -> Field {
    let (lower, upper) = object_uuid();
    nested(
        11,
        vec![
            varint(1, object),
            nested(2, vec![varint(1, lower), varint(2, upper)]),
        ],
    )
}

// -- drawables ----------------------------------------------------------------

/// `TSWP.ShapeInfoArchive` — a shape that holds text, which is what a text box
/// is.
pub(crate) const TYPE_SHAPE_INFO: u32 = 2011;

/// The elements of the path a shape is drawn along, in its own coordinates.
///
/// Element kinds are `TSP.Path.ElementType`: 1 move, 2 line, 4 curve (two
/// control points and an end point), 5 close. A rectangle is what the app
/// writes for a text box — move, three lines, close, move — and the curve form
/// is copied off a freehand drawing Pages wrote, which is where the
/// three-points-per-curve shape was read.
fn outline_path(outline: crate::drawable::Outline, width: f32, height: f32) -> Vec<Field> {
    let point = |x: f32, y: f32| nested(2, vec![float(1, x), float(2, y)]);
    let element = |kind: u64, x: f32, y: f32| nested(1, vec![varint(1, kind), point(x, y)]);
    let curve = |c1: (f32, f32), c2: (f32, f32), to: (f32, f32)| {
        nested(
            1,
            vec![
                varint(1, 4),
                point(c1.0, c1.1),
                point(c2.0, c2.1),
                point(to.0, to.1),
            ],
        )
    };
    match outline {
        crate::drawable::Outline::Rectangle => vec![
            element(1, 0.0, 0.0),
            element(2, width, 0.0),
            element(2, width, height),
            element(2, 0.0, height),
            nested(1, vec![varint(1, 5)]),
            element(1, 0.0, 0.0),
        ],
        crate::drawable::Outline::Line => vec![element(1, 0.0, 0.0), element(2, width, height)],
        crate::drawable::Outline::Ellipse => {
            // The circle-to-Bézier constant: four arcs, each pulled
            // 0.5522847 of the way along the box, is an ellipse to within a
            // ten-thousandth of its radius.
            const K: f32 = 0.552_284_8;
            let (a, b) = (width / 2.0, height / 2.0);
            let (kx, ky) = (a * K, b * K);
            vec![
                element(1, width, b),
                curve((width, b + ky), (a + kx, height), (a, height)),
                curve((a - kx, height), (0.0, b + ky), (0.0, b)),
                curve((0.0, b - ky), (a - kx, 0.0), (a, 0.0)),
                curve((a + kx, 0.0), (width, b - ky), (width, b)),
                nested(1, vec![varint(1, 5)]),
                element(1, width, b),
            ]
        }
    }
}

/// A text box: a rectangle, a style, and a storage holding the words.
///
/// Four archives deep, like every drawable — `TSWP.ShapeInfo` over `TSD.Shape`
/// over `TSD.Drawable` — with the rectangle written twice, as the geometry's
/// size and as the path the shape is drawn along. FORMAT.md §6's rule 14 is
/// exactly this: a size that appears twice must be written twice.
#[allow(clippy::too_many_arguments)]
pub(crate) fn text_box(
    parent: Option<u64>,
    style: u64,
    storage: u64,
    outline: crate::drawable::Outline,
    position: (f32, f32),
    size: (f32, f32),
    wrap: bool,
) -> Message {
    let (width, height) = size;
    let mut drawable = vec![nested(
        1,
        vec![
            nested(1, vec![float(1, position.0), float(2, position.1)]),
            nested(2, vec![float(1, width), float(2, height)]),
            varint(3, 3),
            float(4, 0.0),
        ],
    )];
    // A slide or a sheet owns its drawables and is named as the parent; a
    // Pages page does not — the page group holds the reference instead, and a
    // parent pointing at nothing is what makes Pages refuse the document.
    if let Some(parent) = parent {
        drawable.push(reference(2, parent));
    }
    // Only Pages flows text around a drawable, and only if it says how:
    // `TSD.ExteriorTextWrapArchive`, copied field for field off a floating box
    // Pages itself wrote — around both sides, 12pt of margin.
    if wrap {
        drawable.push(nested(
            3,
            vec![
                varint(1, 1),
                varint(2, 2),
                varint(3, 0),
                float(4, 12.0),
                float(5, 0.0),
                varint(6, 0),
            ],
        ));
    }
    message(vec![
        nested(
            1,
            vec![
                nested(1, drawable),
                reference(2, style),
                // The path the shape is drawn along.
                nested(
                    3,
                    vec![
                        varint(1, 0),
                        varint(2, 0),
                        nested(
                            5,
                            vec![
                                nested(2, vec![float(1, width), float(2, height)]),
                                nested(3, outline_path(outline, width, height)),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        // The storage twice: the field the app reads and the deprecated one it
        // still writes.
        reference(2, storage),
        reference(4, storage),
        varint(6, 1),
    ])
}

/// `TSD.ImageArchive` — a picture, and the rectangle it is drawn in.
///
/// One level shallower than a shape: `{1: TSD.DrawableArchive, 3: media style,
/// 4: originalSize, 7: flags, 9: naturalSize, 11: → the data}`. `naturalSize`
/// is the picture's own pixels and `originalSize` the rectangle it was placed
/// at, which for an image placed from nothing are the two things a resize
/// later has to keep in step (§6).
pub(crate) fn image(
    parent: Option<u64>,
    style: u64,
    data: u64,
    position: (f32, f32),
    size: (f32, f32),
    natural: (f32, f32),
    wrap: bool,
) -> Message {
    let mut drawable = vec![nested(
        1,
        vec![
            nested(1, vec![float(1, position.0), float(2, position.1)]),
            nested(2, vec![float(1, size.0), float(2, size.1)]),
            varint(3, 3),
            float(4, 0.0),
        ],
    )];
    if let Some(parent) = parent {
        drawable.push(reference(2, parent));
    }
    if wrap {
        drawable.push(nested(
            3,
            vec![
                varint(1, 1),
                varint(2, 2),
                varint(3, 0),
                float(4, 12.0),
                float(5, 0.0),
                varint(6, 0),
            ],
        ));
    }
    // Every image in the corpus has its aspect ratio locked, and the apps will
    // not resize one non-proportionally. A new image is written the same way.
    drawable.push(varint(7, 1));
    message(vec![
        nested(1, drawable),
        reference(3, style),
        nested(4, vec![float(1, size.0), float(2, size.1)]),
        varint(7, 0),
        nested(9, vec![float(1, natural.0), float(2, natural.1)]),
        reference(11, data),
        varint(18, 0),
    ])
}

/// The `TSWP.StorageArchive` a text box owns: kind 3, and the text in it.
pub(crate) fn text_box_storage(stylesheet: u64, paragraph: u64, list: u64, text: &str) -> Message {
    storage_of_kind(STORAGE_TEXT_BOX, stylesheet, paragraph, list, text)
}

/// `TSWP.StorageArchive.kind` 3 — a text box.
pub(crate) const STORAGE_TEXT_BOX: u64 = 3;
/// …and 4, a slide's presenter notes. "Presenter notes are the kind-4 storages,
/// and nothing else is" (`FORMAT.md` §13).
pub(crate) const STORAGE_NOTE: u64 = 4;

/// A storage of a given kind, which is the only thing that differs between a
/// text box's and a note's: the field set is identical, and was read off the
/// note of `keynote-deck.key` to make sure — `[1, 2, 3, 5, 6, 7, 10, 14, 24]`
/// in both.
pub(crate) fn storage_of_kind(
    kind: u64,
    stylesheet: u64,
    paragraph: u64,
    list: u64,
    text: &str,
) -> Message {
    let mut fields = vec![varint(1, kind), reference(2, stylesheet)];
    if !text.is_empty() {
        fields.push(string(3, text));
    }
    fields.extend([
        attribute_table(5, paragraph),
        nested(
            6,
            vec![nested(1, vec![varint(1, 0), varint(2, 0), varint(3, 0)])],
        ),
        attribute_table(7, list),
        varint(10, 1),
        nested(
            14,
            vec![nested(1, vec![varint(1, 0), varint(2, 0), varint(3, 0)])],
        ),
        nested(
            24,
            vec![nested(1, vec![varint(1, 0), varint(2, 0), varint(3, 0)])],
        ),
    ]);
    message(fields)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::CellValue;
    use crate::Document;

    fn document(blueprint: Blueprint) -> Document {
        let mut document =
            Document::from_package(blueprint.finish()).expect("a blueprint has to parse");
        document.declare_external_references();
        document
    }

    /// The Pages blueprint, from the inside: the objects the format fixes are
    /// where the format fixes them, and nothing dangles.
    #[test]
    fn a_pages_blueprint_is_a_document() {
        let doc = document(pages(Paper::A4));
        assert_eq!(doc.object(ROOT).expect("root").1.message_type(), 10000);
        assert!(doc.problems().is_empty(), "{:?}", doc.problems());
        assert!(doc.undeclared_references().is_empty());
    }

    /// The Numbers blueprint is not offered by [`crate::Document::new`], so
    /// this is what stands in for it: everything about the table that can be
    /// checked without Numbers in the room.
    ///
    /// It is worth keeping sharp. When the app finally accepts one of these,
    /// the difference between this and that is the answer.
    #[test]
    fn a_numbers_blueprint_holds_a_table_that_can_be_written_into() {
        let mut doc = document(numbers("Sheet 1", "Budget", 12, 3));
        assert!(doc.problems().is_empty(), "{:?}", doc.problems());
        assert!(doc.undeclared_references().is_empty());

        let table = doc.table("Budget").expect("no table");
        assert_eq!((table.rows, table.columns), (12, 3));
        assert_eq!(table.cells().len(), 0, "a new table has no cells");

        // Every cell of it is writable, which is the reason the tile carries a
        // row entry per row rather than none: `set_cell` finds a cell through
        // its row's entry, and Numbers writes no entry for an empty row.
        doc.set_cell("Budget", 0, 0, CellValue::Text("Region".into()))
            .expect("A1");
        doc.set_cell(
            "Budget",
            11,
            2,
            CellValue::Number(crate::table::Decimal::parse("42").expect("a number")),
        )
        .expect("C12");
        let table = doc.table("Budget").expect("no table");
        assert_eq!(table.value(0, 0).to_text(), "Region");
        assert_eq!(table.value(11, 2).to_text(), "42");
        assert!(doc.problems().is_empty(), "{:?}", doc.problems());
    }

    /// A table wider than one tile gets more than one, and the rows in the
    /// second tile are writable too.
    #[test]
    fn a_table_taller_than_a_tile_gets_more_than_one() {
        let mut doc = document(numbers("Sheet 1", "Long", 300, 2));
        let tiles = doc
            .objects()
            .filter(|(_, object)| object.message_type() == TYPE_TILE)
            .count();
        assert_eq!(tiles, 2, "300 rows is two tiles of 256");
        doc.set_cell("Long", 299, 0, CellValue::Text("last".into()))
            .expect("the last row");
        assert_eq!(doc.table("Long").unwrap().value(299, 0).to_text(), "last");
    }

    /// The version number in a component index is the app's, not this crate's.
    #[test]
    fn the_component_generation_is_the_apps_own() {
        assert_eq!(component_generation(Kind::Pages), 835);
        assert_eq!(component_generation(Kind::Numbers), 537);
        assert_eq!(component_generation(Kind::Keynote), 2386);
    }
}
