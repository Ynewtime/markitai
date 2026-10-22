//! Writing to the calculation engine — `TSCE`.
//!
//! Everything else in this crate reads `TSCE` ([`crate::formula`]). This is the
//! one place that writes it, and it writes exactly one thing: the record that
//! says *this cell holds a formula, and these are the cells it reads*.
//!
//! # Why that record is the whole problem
//!
//! Numbers does not recalculate a formula when a document opens. The value in
//! the cell record is what the app shows, and it goes on showing it: a cell
//! given a formula whose cached value belongs to another row was watched
//! reporting that other row's answer, through an open, an edit and a save. Nor
//! does editing a cell the formula reads help — **the engine recalculates what
//! its dependency graph reaches, and a cell nothing registered is not in it**.
//! Watched: with `B2` changed from 1 to 10, the app's own `=SUM($B$2:$B$4)`
//! went from 6 to 15 and the copy of it this crate had put in the next column
//! stayed at 6.
//!
//! # What the app writes for one formula
//!
//! Measured by giving Numbers a blank document, setting `A3` to `=A1+A2`
//! through its own scripting interface, and diffing what came back. Three
//! things changed, and nothing else:
//!
//! ```text
//! owner.cell_dependencies (4) += CellRecordExpandedArchive {
//!     column: 0, row: 2,                          ← A3, the formula's cell
//!     expanded_edges: { rows: [0, 1], columns: [0, 0] }   ← A1 and A2
//! }
//! owner.tiled_cell_dependencies (13) += a new TSCE.CellRecordTileArchive
//!     holding the same record, with the owner id and the tile's origin
//! engine.dependency_tracker.number_of_formulas (2.5) += 1
//! ```
//!
//! The edges are the formula's **precedents**, resolved against the cell that
//! holds it — which is why a relative reference is a different edge in every
//! row, and why the crate has to resolve the AST rather than copy the source's
//! record.
//!
//! What does *not* work, and was tried first: `TSCE.ReferencesToDirtyArchive`,
//! the engine's list of what to recompute, which every document carries and
//! which is empty in every one of them. Naming a cell there — by its owner id,
//! its column and its row — changed nothing at all: the app opened the document
//! and showed the stale value exactly as before. Whatever that list is for, it
//! is not a request.
//!
//! The owner is the table's **cell owner**: of the several
//! `TSCE.FormulaOwnerDependenciesArchive` a table has, the one whose
//! `formula_owner_uid` is the table's `base_owner_uid` — `owner_kind` 1, and
//! the UID every other owner of that table names as its base.

use std::collections::BTreeSet;

use crate::pb::{Message, Value};
use crate::table::Table;
use crate::Error;

/// `TSCE.CalculationEngineArchive`.
pub const TYPE_ENGINE: u32 = 4000;
/// `TSCE.FormulaOwnerDependenciesArchive`.
pub const TYPE_OWNER_DEPENDENCIES: u32 = 4008;
/// `TSCE.CellRecordTileArchive`.
pub const TYPE_CELL_RECORD_TILE: u32 = 4009;

/// How many rows and columns one dependency tile covers.
///
/// Every tile in the corpus begins at `(0, 0)`, so the tiling is never
/// exercised there; 256 is the number `TST.TileStorage` uses for the cells
/// themselves and the one used here for want of a document that shows another.
const TILE_SIZE: usize = 256;

/// Field numbers this module writes.
mod field {
    /// `CalculationEngineArchive.dependency_tracker`.
    pub const TRACKER: u32 = 2;
    /// `DependencyTrackerArchive.number_of_formulas`.
    pub const FORMULA_COUNT: u32 = 5;
    /// `FormulaOwnerDependenciesArchive.formula_owner_uid`.
    pub const OWNER_UID: u32 = 1;
    /// …`.internal_formula_owner_id`.
    pub const OWNER_ID: u32 = 2;
    /// …`.cell_dependencies`, the expanded form.
    pub const CELL_DEPENDENCIES: u32 = 4;
    /// …`.tiled_cell_dependencies`, the same records in their own objects.
    pub const TILED: u32 = 13;
    /// `CellDependenciesExpandedArchive.cell_records`, and
    /// `CellRecordTileArchive.cell_records`.
    pub const RECORDS: u32 = 4;
    /// `CellRecordExpandedArchive.expanded_edges`.
    pub const EDGES: u32 = 6;
}

/// Where a formula's cell and its precedents live, once resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dependency {
    pub row: usize,
    pub column: usize,
    /// The cells the formula reads, in no particular order and without
    /// duplicates.
    pub precedents: BTreeSet<(usize, usize)>,
}

/// The cells a formula reads, resolved against the cell that will hold it.
///
/// Only references this crate can resolve to a cell **in the same table** are
/// returned; `None` says the formula reaches somewhere that would need an edge
/// this module does not write — another table, a whole row or column, a stored
/// `#REF!` — and the caller should refuse rather than register half a graph.
pub fn precedents_of(
    formula: &crate::formula::Formula,
    row: usize,
    column: usize,
) -> Option<BTreeSet<(usize, usize)>> {
    use crate::formula::Axis;
    let mut cells = BTreeSet::new();
    for node in &formula.ast.nodes {
        let Some(reference) = node.reference() else {
            continue;
        };
        if reference.is_error || reference.table.is_some() {
            return None;
        }
        let resolve = |axis: Axis, host: usize| -> Option<usize> {
            match axis {
                Axis::Unbounded => None,
                other => other
                    .resolve(host as i64)
                    .and_then(|v| usize::try_from(v).ok()),
            }
        };
        let (first_row, first_column) = (
            resolve(reference.row, row)?,
            resolve(reference.column, column)?,
        );
        let (last_row, last_column) = (
            resolve(reference.row_end, row)?,
            resolve(reference.column_end, column)?,
        );
        if last_row < first_row || last_column < first_column {
            return None;
        }
        // A range is every cell in it: the app writes one edge per cell, which
        // is what the `=SUM($B$2:$B$4)` case in the corpus shows.
        for r in first_row..=last_row {
            for c in first_column..=last_column {
                cells.insert((r, c));
            }
        }
    }
    Some(cells)
}

/// Register a formula in the engine, so that the app recalculates the cell when
/// anything it reads changes.
///
/// Adds the record in both places the app keeps it — the owner's expanded list
/// and a tile of its own — and raises the engine's formula count. Returns
/// `Ok(false)` when the document has no engine to register in, which is a
/// document with no formulas in it and no `TSCE` at all.
pub fn register_formula(
    document: &mut crate::Document,
    table: &Table,
    dependency: &Dependency,
) -> Result<bool, Error> {
    let Some(engine) = object_of_type(document, TYPE_ENGINE) else {
        return Ok(false);
    };
    let Some((owner_object, owner_id)) = cell_owner(document, table)? else {
        return Err(Error::Format(format!(
            "table {}: the calculation engine has no cell owner for it, so a formula written \
             here would never be recalculated",
            table.name
        )));
    };

    let record = expanded_record(dependency);

    // 1. The owner's expanded list.
    let mut owner = document.archive(owner_object)?;
    let mut dependencies = owner
        .bytes(field::CELL_DEPENDENCIES)
        .and_then(crate::pb::decode_nested)
        .unwrap_or_default();
    dependencies.append_in_order(field::RECORDS, Value::Bytes(record.encode()));
    owner.set_in_order(
        field::CELL_DEPENDENCIES,
        Value::Bytes(dependencies.encode()),
    );

    // 2. The same record in the tiled form, which is what the app reads.
    //
    // **Into the tile that is already there**, when one covers this cell. A
    // second tile with the same owner and the same origin is not a second page
    // of records: the app finds one of the two and reports every cell in the
    // other as "Unexpected missing or corrupt cell record", which is what a
    // first attempt did to two cells sixty rows away from the one it added.
    let tiled = owner
        .bytes(field::TILED)
        .and_then(crate::pb::decode_nested)
        .unwrap_or_default();
    let origin = (
        dependency.column / TILE_SIZE * TILE_SIZE,
        dependency.row / TILE_SIZE * TILE_SIZE,
    );
    let existing = tiled
        .all(1)
        .filter_map(|value| match value {
            Value::Bytes(raw) => crate::table::reference(raw),
            _ => None,
        })
        .find(|tile| {
            document.archive(*tile).ok().is_some_and(|archive| {
                archive.varint(1) == Some(owner_id)
                    && archive.varint(2).unwrap_or(0) as usize == origin.0
                    && archive.varint(3).unwrap_or(0) as usize == origin.1
            })
        });
    match existing {
        Some(tile) => {
            let mut archive = document.archive(tile)?;
            archive.append_in_order(field::RECORDS, Value::Bytes(record.encode()));
            document.set_archive_of(tile, &archive)?;
        }
        None => {
            let tile = crate::create::message(vec![
                crate::create::varint(1, owner_id),
                crate::create::varint(2, origin.0 as u64),
                crate::create::varint(3, origin.1 as u64),
                crate::create::nested_message(field::RECORDS, record),
            ]);
            let mut grow = crate::create::Grow::new(document);
            let identifier = grow.allocate();
            grow.beside(owner_object, identifier, TYPE_CELL_RECORD_TILE, &tile)?;
            grow.finish()?;
            let mut tiled = tiled;
            tiled.append_in_order(1, Value::Bytes(crate::create::reference_bytes(identifier)));
            owner.set_in_order(field::TILED, Value::Bytes(tiled.encode()));
        }
    }
    document.set_archive_of(owner_object, &owner)?;

    // 3. The engine's count of what it has to work out.
    let mut archive = document.archive(engine)?;
    if let Some(mut tracker) = archive
        .bytes(field::TRACKER)
        .and_then(crate::pb::decode_nested)
    {
        let count = tracker.varint(field::FORMULA_COUNT).unwrap_or(0) + 1;
        tracker.set_in_order(field::FORMULA_COUNT, Value::Varint(count));
        archive.set_in_order(field::TRACKER, Value::Bytes(tracker.encode()));
        document.set_archive_of(engine, &archive)?;
    }
    Ok(true)
}

/// `TSCE.CellRecordExpandedArchive` — the cell, and the cells it reads.
fn expanded_record(dependency: &Dependency) -> Message {
    let mut edges = Vec::new();
    for (row, _) in &dependency.precedents {
        edges.push(crate::create::varint(1, *row as u64));
    }
    for (_, column) in &dependency.precedents {
        edges.push(crate::create::varint(2, *column as u64));
    }
    crate::create::message(vec![
        crate::create::varint(1, dependency.column as u64),
        crate::create::varint(2, dependency.row as u64),
        crate::create::nested(field::EDGES, edges),
    ])
}

/// The `FormulaOwnerDependenciesArchive` that owns a table's cells, and its id.
fn cell_owner(document: &crate::Document, table: &Table) -> Result<Option<(u64, u64)>, Error> {
    for (_, object) in document.objects() {
        if object.message_type() != TYPE_OWNER_DEPENDENCIES {
            continue;
        }
        let Ok(archive) = Message::decode(object.payload()) else {
            continue;
        };
        let Some(uid) = archive
            .bytes(field::OWNER_UID)
            .and_then(crate::pb::decode_nested)
        else {
            continue;
        };
        let matches = uid.varint(1) == Some(table.base_uid.lower)
            && uid.varint(2) == Some(table.base_uid.upper);
        if matches {
            let Some(id) = archive.varint(field::OWNER_ID) else {
                continue;
            };
            return Ok(Some((object.identifier, id)));
        }
    }
    Ok(None)
}

fn object_of_type(document: &crate::Document, message_type: u32) -> Option<u64> {
    document
        .objects()
        .find(|(_, object)| object.message_type() == message_type)
        .map(|(_, object)| object.identifier)
}

// -- giving a new table its owners -------------------------------------------

/// `owner_kind` of the owner that stands for the table itself, the one a
/// `base_owner_uid` is reached through.
const OWNER_KIND_HAUNTED: u64 = 35;
/// `owner_kind` of the owner that holds a table's *cells* — the one
/// [`register_formula`] writes a dependency into.
const OWNER_KIND_CELLS: u64 = 1;

/// Give every table that has none the `TSCE` owners a formula needs.
///
/// A table this crate builds from nothing has a `HauntedOwnerArchive` in its
/// model (field 84) and nothing else: no `FormulaOwnerDependenciesArchive`
/// anywhere, so `base_owner_uid` resolves to nothing, `cell_owner` finds
/// nothing, and every formula written into it was refused by name. This is what
/// closes that — two owners per table, copied from the ones Numbers writes:
///
/// * **kind 35, the haunted owner.** Its `formula_owner_uid` is the table's
///   `haunted_uid` and its `base_owner_uid` (12) is the UUID this mints. That
///   pair is the whole join: it is how a reader gets from the table to the
///   identity every cross-table reference writes.
/// * **kind 1, the cell owner.** Its `formula_owner_uid` *is* that
///   `base_owner_uid`, and its field 11 points at the table model. This is the
///   one a formula's dependency record goes into.
///
/// The app writes about eleven owners per table — kinds 3, 4, 5, 6, 8, 9, 10,
/// 11, 12 besides these two — and what each is for is not established here.
/// They are not written: an owner whose purpose is unknown is an object
/// invented rather than copied, and the two that *are* written are the two
/// whose job this crate can state. The rest of each archive is the empty shape
/// every owner in the corpus carries, sentinels included.
///
/// Both owners are registered in the engine's dependency tracker the way the
/// app registers one: an entry in its owner list (field 3) and a reference to
/// the object (field 6). Nothing happens to a table that already has a cell
/// owner, so this is safe to call after every write that makes a table.
pub fn give_tables_their_owners(document: &mut crate::Document) -> Result<usize, Error> {
    let Some(engine) = object_of_type(document, TYPE_ENGINE) else {
        // No calculation engine: a document that holds no formulas and cannot
        // be given one. Pages and Keynote documents without tables are here.
        return Ok(0);
    };

    let wanted: Vec<(u64, crate::table::Uuid)> = document
        .tables()
        .into_iter()
        .filter(|table| {
            table.haunted_uid != crate::table::Uuid::default()
                && cell_owner(document, table).ok().flatten().is_none()
        })
        .map(|table| (table.model, table.haunted_uid))
        .collect();
    if wanted.is_empty() {
        return Ok(0);
    }

    // Every UID and every internal id already spoken for, so the new ones
    // collide with nothing.
    let mut taken: BTreeSet<(u64, u64)> = BTreeSet::new();
    let mut highest = 0u64;
    for (_, object) in document.objects() {
        if object.message_type() != TYPE_OWNER_DEPENDENCIES {
            continue;
        }
        let Ok(archive) = Message::decode(object.payload()) else {
            continue;
        };
        if let Some(uid) = archive
            .bytes(field::OWNER_UID)
            .and_then(crate::pb::decode_nested)
        {
            taken.insert((uid.varint(1).unwrap_or(0), uid.varint(2).unwrap_or(0)));
        }
        highest = highest.max(archive.varint(field::OWNER_ID).unwrap_or(0));
    }

    let mut registered: Vec<(u64, u64)> = Vec::new(); // (object, internal id)
    let mut grow = crate::create::Grow::new(document);
    for (model, haunted) in &wanted {
        // The app numbers a table's owners as offsets from one base, and its
        // haunted owner sits 35 above the base. That arithmetic is *not* the
        // join — the join is the lookup through field 12 — but following it
        // keeps a document this crate writes looking like one the app wrote,
        // and any free value serves when it is taken.
        let mut base = crate::table::Uuid {
            lower: haunted.lower.wrapping_sub(OWNER_KIND_HAUNTED),
            upper: haunted.upper,
        };
        while taken.contains(&(base.lower, base.upper)) || base == *haunted {
            base.lower = base.lower.wrapping_sub(1);
        }
        taken.insert((base.lower, base.upper));
        taken.insert((haunted.lower, haunted.upper));

        highest += 1;
        let haunted_id = highest;
        let haunted_object = grow.allocate();
        highest += 1;
        let cells_id = highest;
        let cells_object = grow.allocate();

        let mut haunted_archive = empty_owner(*haunted, haunted_id, OWNER_KIND_HAUNTED);
        haunted_archive.set_in_order(
            12,
            Value::Bytes(
                crate::create::message(vec![
                    crate::create::varint(1, base.lower),
                    crate::create::varint(2, base.upper),
                ])
                .encode(),
            ),
        );
        let mut cells_archive = empty_owner(base, cells_id, OWNER_KIND_CELLS);
        cells_archive.set_in_order(
            11,
            Value::Bytes(crate::create::message(vec![crate::create::varint(1, *model)]).encode()),
        );

        grow.beside(
            engine,
            haunted_object,
            TYPE_OWNER_DEPENDENCIES,
            &haunted_archive,
        )?;
        grow.beside(
            engine,
            cells_object,
            TYPE_OWNER_DEPENDENCIES,
            &cells_archive,
        )?;
        registered.push((haunted_object, haunted_id));
        registered.push((cells_object, cells_id));
    }
    grow.finish()?;

    // The engine's own index of them: one entry per internal id, one reference
    // per object, both in the dependency tracker.
    let mut archive = document.archive(engine)?;
    let mut tracker = archive
        .bytes(field::TRACKER)
        .and_then(crate::pb::decode_nested)
        .unwrap_or_default();
    let mut owners = tracker
        .bytes(3)
        .and_then(crate::pb::decode_nested)
        .unwrap_or_default();
    for (object, id) in &registered {
        owners.append_in_order(
            1,
            Value::Bytes(
                crate::create::message(vec![
                    crate::create::varint(1, *id),
                    crate::create::nested(2, Vec::new()),
                ])
                .encode(),
            ),
        );
        tracker.append_in_order(
            6,
            Value::Bytes(crate::create::message(vec![crate::create::varint(1, *object)]).encode()),
        );
    }
    tracker.set_in_order(3, Value::Bytes(owners.encode()));
    archive.set_in_order(field::TRACKER, Value::Bytes(tracker.encode()));
    document.set_archive_of(engine, &archive)?;
    Ok(wanted.len())
}

/// A `FormulaOwnerDependenciesArchive` with nothing in it.
///
/// The shape every owner in the corpus carries: the two saturation sentinels in
/// 7 and 8 — a range that names no rows and no columns — the six empty
/// sub-lists of field 6, and the five empty fields around them. An owner
/// written without them is one whose `required` fields a parser cannot find.
fn empty_owner(uid: crate::table::Uuid, id: u64, kind: u64) -> Message {
    use crate::create::{message, nested, varint};
    let sentinels = || {
        let extent = || {
            nested(
                0,
                vec![
                    varint(1, 32767),
                    varint(2, 2147483647),
                    varint(3, 32767),
                    varint(4, 2147483647),
                ],
            )
            .value
        };
        message(vec![
            crate::pb::Field {
                number: 2,
                value: extent(),
            },
            crate::pb::Field {
                number: 3,
                value: extent(),
            },
        ])
    };
    let empty = |number: u32| crate::pb::Field {
        number,
        value: Value::Bytes(Vec::new()),
    };
    message(vec![
        nested(1, vec![varint(1, uid.lower), varint(2, uid.upper)]),
        varint(2, id),
        varint(3, kind),
        empty(4),
        empty(5),
        nested(
            6,
            vec![empty(1), empty(2), empty(3), empty(4), empty(5), empty(7)],
        ),
        crate::pb::Field {
            number: 7,
            value: Value::Bytes(sentinels().encode()),
        },
        crate::pb::Field {
            number: 8,
            value: Value::Bytes(sentinels().encode()),
        },
        nested(9, vec![empty(1)]),
        empty(10),
        empty(13),
        empty(14),
        empty(15),
        empty(16),
    ])
}

/// Put a chart mediator in the engine, with the owner that makes it live.
///
/// The mediator archive itself is [`crate::chart::bind_chart`]'s; this is the
/// half that belongs to the calculation engine — where the object goes, the
/// `owner_kind` **2** owner keyed by the mediator's entity id, and the two
/// registrations in the dependency tracker that every owner has.
///
/// One owner per mediator, which is what the corpus shows: twelve mediators,
/// twelve kind-2 owners, and eleven owners apiece for the eight tables beside
/// them.
pub fn add_chart_mediator(
    document: &mut crate::Document,
    chart: u64,
    archive: &Message,
    uid: crate::table::Uuid,
) -> Result<u64, Error> {
    let engine = object_of_type(document, TYPE_ENGINE).ok_or_else(|| {
        Error::refused(
            crate::Refusal::Missing,
            "this document has no calculation engine, so a chart cannot be made to follow a \
             table"
                .to_string(),
        )
    })?;

    let mut highest = 0u64;
    for (_, object) in document.objects() {
        if object.message_type() != TYPE_OWNER_DEPENDENCIES {
            continue;
        }
        if let Ok(owner) = Message::decode(object.payload()) {
            highest = highest.max(owner.varint(field::OWNER_ID).unwrap_or(0));
        }
    }
    let owner_id = highest + 1;

    let mut grow = crate::create::Grow::new(document);
    let mediator = grow.allocate();
    let owner_object = grow.allocate();
    // `owner_kind` 2, keyed by the mediator's entity id, pointing at the chart
    // — the shape every kind-2 owner in the corpus has, its `6` holding one
    // empty list where a table's owner holds six.
    let mut owner = empty_owner(uid, owner_id, OWNER_KIND_MEDIATOR);
    owner.clear(6);
    owner.set_in_order(
        6,
        Value::Bytes(
            crate::create::message(vec![crate::pb::Field {
                number: 5,
                value: Value::Bytes(Vec::new()),
            }])
            .encode(),
        ),
    );
    owner.set_in_order(
        11,
        Value::Bytes(crate::create::message(vec![crate::create::varint(1, chart)]).encode()),
    );

    grow.beside(
        engine,
        mediator,
        crate::chart::TYPE_TN_CHART_MEDIATOR,
        archive,
    )?;
    grow.beside(engine, owner_object, TYPE_OWNER_DEPENDENCIES, &owner)?;
    grow.finish()?;

    register_owner(document, engine, owner_object, owner_id)?;
    Ok(mediator)
}

/// `owner_kind` of the owner that stands for a chart's mediator.
const OWNER_KIND_MEDIATOR: u64 = 2;

/// Add one owner to the engine's dependency tracker: its internal id in the
/// owner list, and a reference to the object beside it.
fn register_owner(
    document: &mut crate::Document,
    engine: u64,
    object: u64,
    id: u64,
) -> Result<(), Error> {
    let mut archive = document.archive(engine)?;
    let mut tracker = archive
        .bytes(field::TRACKER)
        .and_then(crate::pb::decode_nested)
        .unwrap_or_default();
    let mut owners = tracker
        .bytes(3)
        .and_then(crate::pb::decode_nested)
        .unwrap_or_default();
    owners.append_in_order(
        1,
        Value::Bytes(
            crate::create::message(vec![
                crate::create::varint(1, id),
                crate::create::nested(2, Vec::new()),
            ])
            .encode(),
        ),
    );
    tracker.append_in_order(
        6,
        Value::Bytes(crate::create::message(vec![crate::create::varint(1, object)]).encode()),
    );
    tracker.set_in_order(3, Value::Bytes(owners.encode()));
    archive.set_in_order(field::TRACKER, Value::Bytes(tracker.encode()));
    document.set_archive_of(engine, &archive)
}
