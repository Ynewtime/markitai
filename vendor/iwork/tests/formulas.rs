//! Formulas, read — and printed back at the app that wrote them.
//!
//! Three halves, which is one too many but they are genuinely different kinds
//! of evidence.
//!
//! 1. **Structure.** Every formula archive reachable in the corpus decodes,
//!    every field of every node is one the 15.3.1 schema has, and re-encoding
//!    reproduces the bytes exactly. That needs no app and is the claim the
//!    decoder rests on.
//! 2. **Meaning.** The formula zoo — `numbers-formulas.numbers`, ninety-five
//!    formulas built by Numbers itself, one per node type, operator, reference
//!    shape and literal kind — is asserted case by case: which node the app
//!    wrote for which spelling, and what this crate prints for it.
//! 3. **The oracle**, behind `IWORK_APP_CHECK=1`: `formula of cell` is the
//!    formula text the user would see in the formula bar, and every formula in
//!    every fixture is compared against it **character for character**.
//!
//! Without `tests/fixtures/generated`, all three pass having asserted nothing
//! and say so — `scripts/make-fixtures.sh` builds it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use iwork::formula::{self, node, Ast, Formula, Site};
use iwork::pb::{decode_nested, Message, Value};
use iwork::Document;

fn generated(name: &str) -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/generated")
        .join(name);
    path.exists().then_some(path)
}

fn open(name: &str) -> Option<Document> {
    let path = generated(name)?;
    Some(Document::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

macro_rules! fixture {
    ($name:expr) => {
        match open($name) {
            Some(doc) => doc,
            None => {
                eprintln!("no {} — skipping (run scripts/make-fixtures.sh)", $name);
                return;
            }
        }
    };
}

/// Every document the corpus has, whatever app wrote it.
/// A password-protected package, which is not a fixture any test here can use:
/// its object streams are ciphertext and `Document::open` refuses it by design.
/// `tests/fixtures.rs` is where that refusal is asserted.
fn encrypted(path: &Path) -> bool {
    iwork::Package::read(path).is_ok_and(|package| package.contains(".iwpv2"))
}

fn corpus() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/generated");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("pages") | Some("numbers") | Some("key")
            )
        })
        .filter(|path| !encrypted(path))
        .collect();
    out.sort();
    out
}

/// The formula text of every cell of every table, keyed by table name and A1.
fn our_formulas(doc: &Document) -> BTreeMap<(String, String), String> {
    iwork::table::formulas(&doc.tables())
        .into_iter()
        .map(|f| ((f.table_name, f.reference), f.text))
        .collect()
}

// -- structure ---------------------------------------------------------------

/// Walk every object of a document and hand back every message that is an
/// `ASTNodeArrayArchive`, with the bytes it came from.
///
/// The selection is the decoder's own: an array whose every element is a node
/// the 15.3.1 schema accepts *and* whose stream evaluates as an RPN program.
/// Both halves are needed. A chart's grid values are `{1: varint, 2: message}`
/// and `AST_function_node_index` is a varint, so wire types alone reject those;
/// but a `TSCE.CellRecordExpandedArchive` is `{1: column, 2: row}`, which is a
/// perfectly legal node shape and a nonsensical program.
fn node_arrays(doc: &Document) -> Vec<(Vec<u8>, Ast)> {
    fn walk(message: &Message, depth: usize, out: &mut Vec<(Vec<u8>, Ast)>) {
        if depth > 16 {
            return;
        }
        for field in &message.fields {
            let Value::Bytes(bytes) = &field.value else {
                continue;
            };
            let Some(inner) = decode_nested(bytes) else {
                continue;
            };
            if let Some(ast) = Ast::decode(&inner) {
                if !ast.nodes.is_empty() && ast.validate().is_ok() && ast.is_well_formed() {
                    out.push((bytes.clone(), ast));
                    continue;
                }
            }
            walk(&inner, depth + 1, out);
        }
    }
    let mut out = Vec::new();
    for (_, object) in doc.objects() {
        let Ok(message) = Message::decode(object.payload()) else {
            continue;
        };
        walk(&message, 0, &mut out);
    }
    out
}

/// The claim the whole decoder rests on: decode, and the bytes come back.
///
/// `encode(decode(x)) == x` over every node array in every fixture, plus the
/// stricter statement that every field of every node is one the schema has at
/// the wire type the schema gives it — which is what `Ast::validate` checks and
/// what makes a misparse an error rather than a plausible wrong answer.
#[test]
fn every_formula_archive_re_encodes_to_its_own_bytes() {
    let mut arrays = 0usize;
    let mut nodes = 0usize;
    let mut documents = 0usize;
    for path in corpus() {
        let doc = Document::open(&path).unwrap();
        documents += 1;
        for (bytes, ast) in node_arrays(&doc) {
            ast.validate()
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert_eq!(
                ast.encode(),
                bytes,
                "{}: a node array did not re-encode to its own bytes",
                path.display()
            );
            arrays += 1;
            nodes += ast.nodes.len();
        }
    }
    if documents == 0 {
        eprintln!("no fixtures — skipping (run scripts/make-fixtures.sh)");
        return;
    }
    eprintln!("{arrays} node arrays, {nodes} nodes, over {documents} documents");
    assert!(arrays > 0, "the corpus has formulas in it");
}

/// A structural claim the corpus round-trip cannot make, because that test
/// pre-filters on `validate() && is_well_formed()` and then asserts them: a
/// hand-built `=1+2` decodes, validates, evaluates as a program, re-encodes to
/// its own bytes and prints the text it should. Any of those can fail.
#[test]
fn a_hand_built_ast_round_trips_and_prints() {
    let leaf = |fields: &[(u32, Value)]| {
        let mut message = Message::default();
        for (number, value) in fields {
            message.fields.push(iwork::pb::Field {
                number: *number,
                value: value.clone(),
            });
        }
        Value::Bytes(message.encode())
    };
    let number = |value: f64| {
        leaf(&[
            (1, Value::Varint(u64::from(node::NUMBER))),
            (4, Value::Fixed64(value.to_le_bytes())),
        ])
    };
    let mut array = Message::default();
    for element in [
        number(1.0),
        number(2.0),
        leaf(&[(1, Value::Varint(u64::from(node::ADDITION)))]),
    ] {
        array.fields.push(iwork::pb::Field {
            number: 1,
            value: element,
        });
    }
    let bytes = array.encode();
    let ast = Ast::decode(&array).expect("a node array");
    ast.validate().expect("every field is one the schema has");
    assert!(ast.is_well_formed(), "1 2 + leaves exactly one value");
    assert_eq!(ast.encode(), bytes, "re-encodes to its own bytes");
    let names = formula::Names::default();
    assert_eq!(ast.text(Site::anonymous(&names)), "1+2");
}

/// Every node type the corpus uses is one this crate names, and every function
/// id resolves — except the two that deliberately do not.
///
/// **337** is the internal function behind a spilled cell: `=SEQUENCE(1,3)`
/// spills into the cells beside it and each of them holds `337(<origin>)`.
/// Numbers itself prints `(null)` for it, which is what this crate prints too.
/// **175** is a chart's: every formula in a `TN.ChartMediatorArchive` wraps its
/// one operand in it. Both sit in the published function table's 169–184 and
/// 326–340 holes, so neither has a name to give them, and inventing one would
/// be worse than saying so.
#[test]
fn every_node_type_and_function_id_in_the_corpus_is_known() {
    let mut kinds: BTreeSet<u32> = BTreeSet::new();
    let mut functions: BTreeSet<u32> = BTreeSet::new();
    let mut seen = false;
    for path in corpus() {
        let doc = Document::open(&path).unwrap();
        for (_, ast) in node_arrays(&doc) {
            seen = true;
            for node in &ast.nodes {
                kinds.insert(node.kind);
                if let Some((index, _)) = node.function() {
                    functions.insert(index);
                }
            }
        }
    }
    if !seen {
        eprintln!("no fixtures — skipping");
        return;
    }
    let unnamed: Vec<u32> = kinds
        .iter()
        .copied()
        .filter(|k| node::name(*k) == "?")
        .collect();
    assert!(unnamed.is_empty(), "unnamed node types: {unnamed:?}");

    let unnamed: Vec<u32> = functions
        .iter()
        .copied()
        .filter(|f| formula::function_name(*f).is_none())
        .collect();
    assert_eq!(
        unnamed,
        vec![175, 337],
        "the only function ids with no published name are the chart one and the spill one"
    );
    eprintln!(
        "{} node types, {} function ids over the corpus",
        kinds.len(),
        functions.len()
    );
}

/// Charts carry `TSCE` formulas of their own, and Phase 6 will want to know
/// where: every one of them lives in a `TN.ChartMediatorArchive` and wraps its
/// single operand in function 175.
#[test]
fn a_chart_keeps_its_references_in_its_mediator() {
    let doc = fixture!("numbers-rules.numbers");
    let mut mediators = 0usize;
    let mut elsewhere = 0usize;
    for (_, object) in doc.objects() {
        let Ok(message) = Message::decode(object.payload()) else {
            continue;
        };
        let mut found = Vec::new();
        fn walk(message: &Message, depth: usize, out: &mut Vec<Ast>) {
            if depth > 16 {
                return;
            }
            for field in &message.fields {
                let Value::Bytes(bytes) = &field.value else {
                    continue;
                };
                let Some(inner) = decode_nested(bytes) else {
                    continue;
                };
                if let Some(ast) = Ast::decode(&inner) {
                    if !ast.nodes.is_empty() && ast.validate().is_ok() && ast.is_well_formed() {
                        out.push(ast);
                        continue;
                    }
                }
                walk(&inner, depth + 1, out);
            }
        }
        walk(&message, 0, &mut found);
        for ast in &found {
            if ast.nodes.iter().any(|n| n.function() == Some((175, 1))) {
                if object.message_type() == 12006 {
                    mediators += 1;
                } else {
                    elsewhere += 1;
                }
            }
        }
    }
    assert!(mediators > 0, "the stocks fixture has charts");
    assert_eq!(elsewhere, 0, "function 175 lives only in a chart mediator");
    eprintln!("{mediators} chart formulas in TN.ChartMediatorArchive objects");
}

/// A pivot's stored table is its **base**, and the app shows a larger *view*.
///
/// `Sales Pivot` is 7×5 on disk and 10×6 to Numbers: the grand-total row and
/// the grand-total column exist only in the view, which `TSCE.CoordMapperArchive`
/// maps base coordinates into. Seventeen of the app's thirty-two formulas for
/// that table are at positions that have no cell record at all.
#[test]
fn a_pivots_stored_table_is_smaller_than_the_one_the_app_shows() {
    let doc = fixture!("numbers-pivot.numbers");
    let table = doc.table("Sales Pivot").expect("the pivot");
    assert!(table.pivot.is_some());
    assert_eq!((table.rows, table.columns), (7, 5));
}

/// A formula stored in a table's FORMULA list has **no host cell of its own**,
/// which is what lets several cells share one entry.
///
/// `=SEQUENCE(1,3)` spills into the two cells beside it, and all three of them
/// key the same formula: the two spill children carry function 337 against an
/// absolute reference to their origin.
#[test]
fn a_formula_carries_no_host_and_can_be_shared() {
    let doc = fixture!("numbers-formulas.numbers");
    let tables = doc.tables();
    for table in &tables {
        for (_, _, formula) in table.formula_cells() {
            assert!(
                formula.host.is_none(),
                "a formula in a data list should carry no host cell"
            );
        }
    }
    let zoo = doc.table("Zoo").expect("the zoo table");
    let mut keys: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for cell in zoo.cells() {
        if let Some(key) = cell.record.formula_id {
            keys.entry(key).or_default().push(format!(
                "{}{}",
                column_letters(cell.column),
                cell.row + 1
            ));
        }
    }
    let shared: Vec<&Vec<String>> = keys.values().filter(|cells| cells.len() > 1).collect();
    assert!(
        !shared.is_empty(),
        "the spilled SEQUENCE() shares one formula key with its children: {keys:?}"
    );
}

fn column_letters(column: usize) -> String {
    formula::column_letters(column as i64)
}

// -- the reference model -----------------------------------------------------

/// The zoo, case by case: which reference shape the app wrote, and what this
/// crate makes of it.
///
/// The interesting column is the *second*: the same `=B25` is a relative
/// reference and `=$B$2` an absolute one, and the file distinguishes them by
/// one boolean per axis rather than by anything in the text.
#[test]
fn the_reference_model_is_read_axis_by_axis() {
    let doc = fixture!("numbers-formulas.numbers");
    let zoo = doc.table("Zoo").expect("the zoo table");
    let case = |label: &str| -> (usize, &formula::Formula) {
        let row = (0..zoo.rows)
            .find(|r| matches!(zoo.value(*r, 0), iwork::CellValue::Text(t) if t == label))
            .unwrap_or_else(|| panic!("no case called {label}"));
        (row, zoo.formula(row, 2).expect("a formula"))
    };

    // Relative on both axes: the offsets are signed and count from the cell
    // that holds the key.
    let (row, formula) = case("relativ");
    let reference = formula.ast.nodes[0].reference().expect("a reference");
    assert_eq!(reference.column, formula::Axis::Relative(-1));
    assert_eq!(reference.row, formula::Axis::Relative(0));
    let resolved = reference.resolve((2, row as i64));
    assert_eq!((resolved.column, resolved.row), (Some(1), Some(row as i64)));

    // `$B$2`, `B$2`, `$B2` — one flag per axis, and an absolute index is the
    // index itself rather than an offset.
    let (_, formula) = case("absolut-beide");
    let reference = formula.ast.nodes[0].reference().unwrap();
    assert_eq!(reference.column, formula::Axis::Absolute(1));
    assert_eq!(reference.row, formula::Axis::Absolute(1));
    let (_, formula) = case("absolut-zeile");
    let reference = formula.ast.nodes[0].reference().unwrap();
    assert!(!reference.column.is_absolute() && reference.row.is_absolute());
    let (_, formula) = case("absolut-spalte");
    let reference = formula.ast.nodes[0].reference().unwrap();
    assert!(reference.column.is_absolute() && !reference.row.is_absolute());

    // A whole column has no row at all, and a whole row no column.
    let (_, formula) = case("ganze-spalte");
    let reference = formula.ast.nodes[0].reference().unwrap();
    assert_eq!(reference.row, formula::Axis::Unbounded);
    assert!(matches!(reference.column, formula::Axis::Relative(_)));
    let (_, formula) = case("ganze-zeile");
    let reference = formula.ast.nodes[0].reference().unwrap();
    assert_eq!(reference.column, formula::Axis::Unbounded);

    // `B` and `B:B` are the same archive — the app accepts both spellings and
    // writes one.
    let (_, letter) = case("ganze-spalte-buchstabe");
    let (_, bare) = case("ganze-spalte");
    assert_eq!(
        letter.ast.nodes[0].reference(),
        bare.ast.nodes[0].reference(),
        "a whole column is one shape however it was typed"
    );

    // A range is a COLON_TRACT_NODE, and a mixed range keeps both lists.
    let (_, formula) = case("bereich-gemischt");
    let node = &formula.ast.nodes[0];
    assert_eq!(node.kind, node::COLON_TRACT);
    let reference = node.reference().unwrap();
    assert!(reference.is_range);
    assert!(reference.column.is_absolute(), "$B2 anchors the column");
    assert!(!reference.row.is_absolute(), "$B2 leaves the row relative");
    assert!(!reference.column_end.is_absolute(), "B$4 leaves the column");
    assert!(reference.row_end.is_absolute(), "B$4 anchors the row");

    // A deleted precedent leaves a stored reference error, with both axes at
    // their saturation sentinel — and they are different sentinels.
    let (_, formula) = case("referenzfehler");
    let node = &formula.ast.nodes[0];
    assert_eq!(node.kind, node::REFERENCE_ERROR_WITH_UIDS);
    assert!(node.reference().unwrap().is_error);
}

/// Cross-table references resolve by **identity, not by name** — proven by a
/// table that was renamed after the formula was written.
///
/// The fixture writes `=Alt::A1` and then renames `Alt` to `Neu` before saving.
/// The string `Alt` therefore appears nowhere in the document, the AST carries
/// a UUID, and the only way to print `Neu::A1` is to follow it.
#[test]
fn a_cross_table_reference_survives_a_rename() {
    let doc = fixture!("numbers-formulas.numbers");
    let tables = doc.tables();
    assert!(
        tables.iter().any(|t| t.name == "Neu"),
        "the fixture renames Alt to Neu"
    );
    assert!(
        !tables.iter().any(|t| t.name == "Alt"),
        "and nothing is called Alt any more"
    );

    let zoo = doc.table("Zoo").unwrap();
    let row = (0..zoo.rows)
        .find(|r| matches!(zoo.value(*r, 0), iwork::CellValue::Text(t) if t == "umbenannt"))
        .expect("the umbenannt case");
    let formula = zoo.formula(row, 2).expect("a formula");
    let reference = formula.ast.nodes[0].reference().expect("a reference");
    let uid = reference
        .table
        .expect("a cross-table reference carries a UUID");

    let target = tables
        .iter()
        .find(|t| t.base_uid == uid)
        .expect("the UUID names a table in this document");
    assert_eq!(target.name, "Neu");
    // And the UUID is the base owner's, not the haunted owner's: reading the
    // wrong one finds nothing.
    assert_ne!(target.haunted_uid, uid);
    assert!(
        tables.iter().all(|t| t.haunted_uid != uid),
        "no table's haunted UUID is what an AST writes"
    );

    let names = iwork::table::names(&tables);
    let position = tables.iter().position(|t| t.name == "Zoo").unwrap();
    let at = Site::new(&names, Some(position), (2, row as i64));
    assert_eq!(formula.text(at), "=Neu::A1");
}

/// The header-name model, which is what makes a formula readable and is
/// entirely absent from the file: a name is the *text of a header cell*.
#[test]
fn a_reference_is_named_by_the_header_cells_of_its_table() {
    let doc = fixture!("numbers-formulas.numbers");
    let tables = doc.tables();
    let formulas = our_formulas(&doc);
    let zoo = |case: &str| {
        let table = doc.table("Zoo").unwrap();
        let row = (0..table.rows)
            .find(|r| matches!(table.value(*r, 0), iwork::CellValue::Text(t) if t == case))
            .unwrap_or_else(|| panic!("no case {case}"));
        formulas
            .get(&("Zoo".to_string(), format!("C{}", row + 1)))
            .cloned()
            .unwrap_or_default()
    };

    // A body cell of a table with a header row *and* a header column is named
    // by both, column first.
    assert_eq!(zoo("relativ"), "=Wert relativ");
    // The `$` of an absolute axis sits in front of that axis's name.
    assert_eq!(zoo("absolut-beide"), "=$Wert $addition");
    assert_eq!(zoo("absolut-zeile"), "=Wert $addition");
    assert_eq!(zoo("absolut-spalte"), "=$Wert addition");
    // A range keeps A1 notation even where every row and column is named.
    assert_eq!(zoo("bereich"), "=SUM(B2:B4)");
    // A whole column or row takes the one name it has.
    assert_eq!(zoo("ganze-spalte"), "=SUM(Wert)");
    assert_eq!(zoo("ganze-zeile"), "=SUM(addition)");
    // A table with no headers cannot name anything.
    assert_eq!(zoo("kopflose-tabelle"), "=Kopflos::B2");
    // Two header rows: the *last* one names the column.
    assert_eq!(zoo("zwei-kopfzeilen"), "=Unten Zeile3");
    // A cell inside the header row has no row name, so the whole reference
    // falls back to A1.
    assert_eq!(zoo("kopfzelle"), "=B1");
    // A name unique in the document needs no table prefix even across tables;
    // the same name in a second table does.
    assert_eq!(zoo("kreuztabelle"), "=Menge Schrauben");
    assert_eq!(zoo("kopfname"), "=SUM(Menge)");
    assert_eq!(zoo("mehrdeutiger-kopfname"), "=SUM(Daten2::Menge)");
    // Quoting: an operator character forces single quotes and an embedded
    // apostrophe is doubled; a space and a function's name do not.
    assert_eq!(zoo("kopfname-operator"), "='A+B' normal");
    assert_eq!(zoo("kopfname-leerzeichen"), "=x y normal");
    assert_eq!(zoo("kopfname-apostroph"), "='it''s' normal");
    assert_eq!(zoo("kopfname-klammern"), "='Preis (netto)' normal");
    assert_eq!(zoo("kopfname-funktionsname"), "=SUM normal");
    assert_eq!(zoo("kopfname-zeile-apostroph"), "='A+B' 'mit''Hoch'");

    // And a name that names more than one row is not a name: every row of the
    // invoice fixture's header column reads "Item name".
    let names = iwork::table::names(&tables);
    let daten = names
        .tables
        .iter()
        .find(|t| t.name == "Daten")
        .expect("Daten");
    assert_eq!(daten.column_name(1), Some("Menge"));
    assert_eq!(daten.row_name(1), Some("Schrauben"));
    assert_eq!(daten.column_name(0), None, "column A is the header column");
    assert_eq!(daten.row_name(0), None, "row 1 is the header row");
}

/// Ambiguous header names, in the one document that has them: a name shared by
/// eleven rows names nothing, and the app falls back to A1.
#[test]
fn a_name_that_names_more_than_one_row_is_not_used() {
    let doc = fixture!("numbers-links.numbers");
    let formulas = our_formulas(&doc);
    assert_eq!(
        formulas.get(&("Budget-1".to_string(), "E2".to_string())),
        Some(&"=C2×D2".to_string()),
        "every row's header cell reads \"Item name\", so no row has a name"
    );
    assert_eq!(
        formulas.get(&("Budget-1-1-1".to_string(), "D1".to_string())),
        Some(&"=SUM(Cost)".to_string()),
        "a column name in another table, unique in the document, needs no prefix"
    );
}

/// The `LET` trap, which is the reason the field table in `formula.rs` exists.
///
/// `AST_let_is_continuation` is field **36** and a **varint** from 14.4
/// onwards; up to 13.1 that number held a nested `ASTLetNodeWhitespace`
/// message. 15.3.1 writes the new shape, and a second binding in one `LET` is
/// exactly what sets the flag.
#[test]
fn a_let_binding_uses_the_post_14_4_shape_of_fields_34_to_37() {
    let doc = fixture!("numbers-formulas.numbers");
    let zoo = doc.table("Zoo").unwrap();
    let case = |label: &str| {
        let row = (0..zoo.rows)
            .find(|r| matches!(zoo.value(*r, 0), iwork::CellValue::Text(t) if t == label))
            .unwrap_or_else(|| panic!("no case {label}"));
        zoo.formula(row, 2).expect("a formula").clone()
    };

    let single = case("let");
    let binds: Vec<&formula::Node> = single
        .ast
        .nodes
        .iter()
        .filter(|n| n.kind == node::LET_BIND)
        .collect();
    assert_eq!(binds.len(), 1);
    assert_eq!(binds[0].let_binding(), Some(("x".to_string(), false)));
    assert_eq!(binds[0].symbol(), Some(1));
    // Field 36 really is a varint here — a message would fail validation.
    assert!(binds[0].validate().is_ok());
    assert!(matches!(binds[0].message().get(36), Some(Value::Varint(0))));

    let multi = case("let-mehrfach");
    let binds: Vec<(String, bool)> = multi
        .ast
        .nodes
        .iter()
        .filter(|n| n.kind == node::LET_BIND)
        .filter_map(|n| n.let_binding())
        .collect();
    assert_eq!(
        binds,
        vec![("x".to_string(), false), ("y".to_string(), true)],
        "the second binding of one LET continues the first"
    );
    assert_eq!(
        multi
            .ast
            .nodes
            .iter()
            .filter(|n| n.kind == node::END_SCOPE)
            .count(),
        2,
        "one END_SCOPE_NODE per binding"
    );
}

/// A filter rule's condition is a `TSCE` formula, and Phase 1b could only say
/// "predicate 37 against a formula". This is the same rule, decoded.
#[test]
fn a_filter_condition_reads_as_the_test_it_is() {
    let doc = fixture!("numbers-rules.numbers");
    let tables = doc.tables();
    let names = iwork::table::names(&tables);
    let mut conditions = Vec::new();
    for (position, table) in tables.iter().enumerate() {
        let Some(filter) = &table.filter else {
            continue;
        };
        for rule in &filter.rules {
            if let Some(text) = iwork::table::predicate_text(
                &rule.predicate,
                &names,
                position,
                rule.column.unwrap_or(0),
            ) {
                conditions.push(text);
            }
        }
    }
    assert_eq!(conditions.len(), 1, "the fixture has one filter rule");
    let condition = &conditions[0];
    assert!(
        condition.starts_with("=IF(LEN(") && condition.contains("FIND.CASEINSENSITIVE"),
        "the rule tests where a character occurs in the cell: {condition}"
    );

    // Conditional highlighting is the same machinery, and its subject is the
    // cell being styled — a node with no coordinates at all.
    let mut highlights = Vec::new();
    for (position, table) in tables.iter().enumerate() {
        for set in &table.conditional_styles {
            for rule in &set.rules {
                if let Some(text) =
                    iwork::table::predicate_text(&rule.predicate, &names, position, 0)
                {
                    highlights.push(text);
                }
            }
        }
    }
    assert!(
        highlights.iter().any(|t| t == "=#CELL>0"),
        "a greater-than-zero highlight: {highlights:?}"
    );
}

/// Pages tables carry formulas too, and no dictionary will report them: Pages
/// has no table property at all. The decoder is the only reader there is.
#[test]
fn pages_tables_carry_formulas_of_their_own() {
    let doc = fixture!("pages-report.pages");
    let formulas = our_formulas(&doc);
    assert_eq!(
        formulas.get(&("Details".to_string(), "D2".to_string())),
        Some(&"=B2×C2".to_string())
    );
    assert!(
        formulas.values().any(|t| t == "=SUM(D)"),
        "and a whole-column reference: {formulas:?}"
    );
}

/// Numeric literals are stored twice and the decimal is the one that is right.
#[test]
fn a_number_literal_comes_from_its_decimal_and_not_from_the_double() {
    let doc = fixture!("numbers-formulas.numbers");
    let formulas = our_formulas(&doc);
    let zoo = doc.table("Zoo").unwrap();
    let text_of = |label: &str| {
        let row = (0..zoo.rows)
            .find(|r| matches!(zoo.value(*r, 0), iwork::CellValue::Text(t) if t == label))
            .unwrap();
        formulas
            .get(&("Zoo".to_string(), format!("C{}", row + 1)))
            .cloned()
            .unwrap()
    };
    assert_eq!(text_of("zahl-dezimal"), "=0.1+0.2");
    assert_eq!(text_of("zahl-klein"), "=0.00001×2");
    assert_eq!(text_of("zahl-gross"), "=1000000×3");
    assert_eq!(text_of("funktion-viele-argumente"), "=SUM(1,2,3,4,5)");
}

// -- the oracle --------------------------------------------------------------

/// Off unless `IWORK_APP_CHECK=1`: every formula in every Numbers fixture,
/// compared with what the app prints in its formula bar, **character for
/// character**.
///
/// The one documented exception is a **pivot table**, for two reasons that both
/// belong to it: its formulas are category references — a reference to a
/// *group* rather than to a rectangle, which this crate decodes and does not
/// spell the way the app does — and its stored table is smaller than the table
/// the app shows, because the grand-total row and column are view-only. Every
/// cell of a pivot is therefore counted as deferred and named, never skipped,
/// and nothing outside a pivot is allowed to be.
#[test]
fn every_formula_matches_the_app() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the oracle comparison");
        return;
    }
    let mut compared = 0usize;
    let mut agreed = 0usize;
    let mut deferred = 0usize;
    let mut wrong: Vec<String> = Vec::new();
    for name in [
        "numbers-values.numbers",
        "numbers-formulas.numbers",
        "numbers-large.numbers",
        "numbers-links.numbers",
        "numbers-rules.numbers",
        "numbers-sorted.numbers",
        "numbers-pivot.numbers",
    ] {
        let Some(path) = generated(name) else {
            eprintln!("no {name} — skipping");
            continue;
        };
        let doc = Document::open(&path).unwrap();
        let pivots: BTreeSet<String> = doc
            .tables()
            .iter()
            .filter(|t| t.pivot.is_some())
            .map(|t| t.name.clone())
            .collect();
        let ours = our_formulas(&doc);
        let theirs = read_oracle(&path);
        for (key, app) in &theirs {
            compared += 1;
            let mine = ours.get(key);
            if mine.map(String::as_str) == Some(app.as_str()) {
                agreed += 1;
            } else if pivots.contains(&key.0) {
                deferred += 1;
            } else {
                wrong.push(format!("{name} {key:?}: app {app:?}, ours {mine:?}"));
            }
        }
        for key in ours.keys() {
            assert!(
                theirs.contains_key(key),
                "{name}: we found a formula at {key:?} that the app does not report"
            );
        }
    }
    eprintln!("{agreed}/{compared} formulas match the app character for character, {deferred} deferred (all in pivot tables)");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert!(compared > 0, "the oracle reported formulas");
}

/// `formula of cell` for every cell of every table, keyed the way
/// [`our_formulas`] keys it.
fn read_oracle(document: &Path) -> BTreeMap<(String, String), String> {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/table-oracle.sh");
    let output = std::process::Command::new(&script)
        .arg(document)
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "{}: Numbers would not report on {}:\n{}",
        script.display(),
        document.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    let mut out = BTreeMap::new();
    let mut table = String::new();
    for line in text.lines() {
        let field: Vec<&str> = line.split('\t').collect();
        match field.first() {
            Some(&"table") if field.len() >= 7 => table = field[1].to_string(),
            Some(&"cell") if field.len() >= 7 && !field[6].is_empty() => {
                out.insert((table.clone(), field[1].to_string()), field[6].to_string());
            }
            _ => {}
        }
    }
    out
}

/// A no-op formula site, so the printer can be exercised without a document.
#[allow(dead_code)]
fn nowhere() -> formula::Names {
    formula::Names::default()
}

#[allow(dead_code)]
fn text_of(formula: &Formula, names: &formula::Names) -> String {
    formula.text(Site::anonymous(names))
}
