//! `TSCE` — the calculation engine: formulas, their ASTs and their text.
//!
//! A cell does not hold a formula; it holds a **key** into its table's FORMULA
//! `TST.TableDataList`, and the entry there is a `TSCE.FormulaArchive` whose
//! only required field is an `ASTNodeArrayArchive` — a post-order (RPN) stream
//! of nodes. Evaluating that stream onto a stack, with operators popping their
//! operands, reconstructs the expression the user typed.
//!
//! Two consequences of the key indirection are worth stating up front, because
//! both are load-bearing here:
//!
//! * **A formula archive stored in a data list carries no host cell.** The
//!   `host_column`/`host_row` fields exist in the schema and 15.3.1 leaves them
//!   out of every entry in this corpus. Relative references therefore resolve
//!   against *the cell holding the key*, which is why one entry can be shared
//!   by many cells — the spill children of one `SEQUENCE()` share a key, and so
//!   does every row of a filled-down column.
//! * **A reference is not stored as text.** It is a column/row pair of signed
//!   offsets (or absolute indices) plus, for anything outside the host table, a
//!   UUID. Table names never appear in the file, which is why renaming a table
//!   does not touch a single formula.
//!
//! Everything below works at the wire level, like the rest of the crate: a node
//! is kept as its protobuf message, validated field by field against the 15.3.1
//! schema, and read through typed accessors. `Node::validate` is what makes the
//! decode trustworthy — it rejects any field number or wire type the schema
//! does not have, so a misparse is an error rather than a plausible wrong
//! answer, and re-encoding a decoded node reproduces its bytes exactly.

use std::collections::BTreeMap;

use crate::pb::{decode_nested, Message, Value};
pub use crate::table::Uuid;

// -- node types --------------------------------------------------------------

/// `ASTNodeArrayArchive.ASTNodeType`, the discriminator at field 1.
///
/// The numbering has holes (37–44, 47, 49–51, 58–62) and three values were
/// renamed without changing meaning; the names here are 15.3.1's.
pub mod node {
    pub const ADDITION: u32 = 1;
    pub const SUBTRACTION: u32 = 2;
    pub const MULTIPLICATION: u32 = 3;
    pub const DIVISION: u32 = 4;
    pub const POWER: u32 = 5;
    pub const CONCATENATION: u32 = 6;
    pub const GREATER_THAN: u32 = 7;
    pub const GREATER_THAN_OR_EQUAL: u32 = 8;
    pub const LESS_THAN: u32 = 9;
    pub const LESS_THAN_OR_EQUAL: u32 = 10;
    pub const EQUAL_TO: u32 = 11;
    pub const NOT_EQUAL_TO: u32 = 12;
    pub const NEGATION: u32 = 13;
    pub const PLUS_SIGN: u32 = 14;
    pub const PERCENT: u32 = 15;
    pub const FUNCTION: u32 = 16;
    pub const NUMBER: u32 = 17;
    pub const BOOLEAN: u32 = 18;
    pub const STRING: u32 = 19;
    pub const DATE: u32 = 20;
    pub const DURATION: u32 = 21;
    pub const EMPTY_ARGUMENT: u32 = 22;
    pub const TOKEN: u32 = 23;
    pub const ARRAY: u32 = 24;
    pub const LIST: u32 = 25;
    pub const THUNK: u32 = 26;
    pub const LOCAL_CELL_REFERENCE: u32 = 27;
    pub const CROSS_TABLE_CELL_REFERENCE: u32 = 28;
    pub const COLON: u32 = 29;
    pub const REFERENCE_ERROR: u32 = 30;
    pub const UNKNOWN_FUNCTION: u32 = 31;
    pub const APPEND_WHITESPACE: u32 = 32;
    pub const PREPEND_WHITESPACE: u32 = 33;
    pub const BEGIN_THUNK: u32 = 34;
    pub const END_THUNK: u32 = 35;
    pub const CELL_REFERENCE: u32 = 36;
    pub const COLON_WITH_UIDS: u32 = 45;
    pub const REFERENCE_ERROR_WITH_UIDS: u32 = 46;
    pub const UID_REFERENCE: u32 = 48;
    pub const LET_BIND: u32 = 52;
    pub const VAR: u32 = 53;
    pub const END_SCOPE: u32 = 54;
    pub const LAMBDA: u32 = 55;
    pub const BEGIN_LAMBDA_THUNK: u32 = 56;
    pub const END_LAMBDA_THUNK: u32 = 57;
    pub const LINKED_CELL_REF: u32 = 63;
    pub const LINKED_COLUMN_REF: u32 = 64;
    pub const LINKED_ROW_REF: u32 = 65;
    pub const CATEGORY_REF: u32 = 66;
    pub const COLON_TRACT: u32 = 67;
    pub const VIEW_TRACT_REF: u32 = 68;
    pub const INTERSECTION: u32 = 69;
    pub const SPILL_RANGE: u32 = 70;

    /// Whether the 15.3.1 schema has this node type.
    ///
    /// The enum is closed and its numbering has holes (37–44, 47, 49–51,
    /// 58–62), so an out-of-range discriminator means the message is not an AST
    /// node — which is what tells a node array apart from the many other
    /// repeated-field-1 messages a document holds. A genuinely newer node type
    /// would land here too; `upgrade_node_type` (field 47) is the schema's own
    /// answer to that and nothing in this corpus carries one.
    pub fn is_known(kind: u32) -> bool {
        name(kind) != "?"
    }

    /// The schema's name for a node type, for diagnostics.
    pub fn name(kind: u32) -> &'static str {
        match kind {
            ADDITION => "ADDITION_NODE",
            SUBTRACTION => "SUBTRACTION_NODE",
            MULTIPLICATION => "MULTIPLICATION_NODE",
            DIVISION => "DIVISION_NODE",
            POWER => "POWER_NODE",
            CONCATENATION => "CONCATENATION_NODE",
            GREATER_THAN => "GREATER_THAN_NODE",
            GREATER_THAN_OR_EQUAL => "GREATER_THAN_OR_EQUAL_TO_NODE",
            LESS_THAN => "LESS_THAN_NODE",
            LESS_THAN_OR_EQUAL => "LESS_THAN_OR_EQUAL_TO_NODE",
            EQUAL_TO => "EQUAL_TO_NODE",
            NOT_EQUAL_TO => "NOT_EQUAL_TO_NODE",
            NEGATION => "NEGATION_NODE",
            PLUS_SIGN => "PLUS_SIGN_NODE",
            PERCENT => "PERCENT_NODE",
            FUNCTION => "FUNCTION_NODE",
            NUMBER => "NUMBER_NODE",
            BOOLEAN => "BOOLEAN_NODE",
            STRING => "STRING_NODE",
            DATE => "DATE_NODE",
            DURATION => "DURATION_NODE",
            EMPTY_ARGUMENT => "EMPTY_ARGUMENT_NODE",
            TOKEN => "TOKEN_NODE",
            ARRAY => "ARRAY_NODE",
            LIST => "LIST_NODE",
            THUNK => "THUNK_NODE",
            LOCAL_CELL_REFERENCE => "LOCAL_CELL_REFERENCE_NODE",
            CROSS_TABLE_CELL_REFERENCE => "CROSS_TABLE_CELL_REFERENCE_NODE",
            COLON => "COLON_NODE",
            REFERENCE_ERROR => "REFERENCE_ERROR_NODE",
            UNKNOWN_FUNCTION => "UNKNOWN_FUNCTION_NODE",
            APPEND_WHITESPACE => "APPEND_WHITESPACE_NODE",
            PREPEND_WHITESPACE => "PREPEND_WHITESPACE_NODE",
            BEGIN_THUNK => "BEGIN_THUNK_NODE",
            END_THUNK => "END_THUNK_NODE",
            CELL_REFERENCE => "CELL_REFERENCE_NODE",
            COLON_WITH_UIDS => "COLON_NODE_WITH_UIDS",
            REFERENCE_ERROR_WITH_UIDS => "REFERENCE_ERROR_WITH_UIDS",
            UID_REFERENCE => "UID_REFERENCE_NODE",
            LET_BIND => "LET_BIND_NODE",
            VAR => "VAR_NODE",
            END_SCOPE => "END_SCOPE_NODE",
            LAMBDA => "LAMBDA_NODE",
            BEGIN_LAMBDA_THUNK => "BEGIN_LAMBDA_THUNK_NODE",
            END_LAMBDA_THUNK => "END_LAMBDA_THUNK_NODE",
            LINKED_CELL_REF => "LINKED_CELL_REF_NODE",
            LINKED_COLUMN_REF => "LINKED_COLUMN_REF_NODE",
            LINKED_ROW_REF => "LINKED_ROW_REF_NODE",
            CATEGORY_REF => "CATEGORY_REF_NODE",
            COLON_TRACT => "COLON_TRACT_NODE",
            VIEW_TRACT_REF => "VIEW_TRACT_REF_NODE",
            INTERSECTION => "INTERSECTION_NODE",
            SPILL_RANGE => "SPILL_RANGE_NODE",
            _ => "?",
        }
    }
}

// -- the schema --------------------------------------------------------------

/// Wire types, as the schema constrains them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Wire {
    Varint,
    Fixed64,
    Bytes,
}

fn wire_of(value: &Value) -> Wire {
    match value {
        Value::Varint(_) => Wire::Varint,
        Value::Fixed64(_) => Wire::Fixed64,
        Value::Bytes(_) => Wire::Bytes,
        Value::Fixed32(_) => Wire::Varint, // never legal here; validate() rejects
    }
}

/// `ASTNodeArrayArchive.ASTNodeArchive`, every field of the 15.3.1 schema.
///
/// **Fields 35 and 36 are the trap this table exists for.** Up to 13.1 field 35
/// was a nested `ASTNodeArrayArchive` and field 36 a nested
/// `ASTLetNodeWhitespace`; at 14.4 they became a `string` and a `bool` *in
/// place*. Field 35 keeps wire type LEN, so an old schema parses a whitespace
/// string as a nested AST and says nothing; field 36 goes LEN → varint, so an
/// old schema throws. 15.3.1 writes the new shape — `numbers-formulas.numbers`
/// has a `LET_BIND_NODE` carrying `34 = "x"`, `36 = 0` (a varint) and `37 = 1`
/// — and this table is keyed on that, never on the wire type found.
const NODE_FIELDS: &[(u32, Wire, &str)] = &[
    (1, Wire::Varint, "AST_node_type"),
    (2, Wire::Varint, "AST_function_node_index"),
    (3, Wire::Varint, "AST_function_node_numArgs"),
    (4, Wire::Fixed64, "AST_number_node_number"),
    (5, Wire::Varint, "AST_boolean_node_boolean"),
    (6, Wire::Bytes, "AST_string_node_string"),
    (7, Wire::Fixed64, "AST_date_node_dateNum"),
    (8, Wire::Fixed64, "AST_duration_node_unitNum"),
    (9, Wire::Varint, "AST_duration_node_unit"),
    (10, Wire::Varint, "AST_token_node_boolean"),
    (11, Wire::Varint, "AST_array_node_numCol"),
    (12, Wire::Varint, "AST_array_node_numRow"),
    (13, Wire::Varint, "AST_list_node_numArgs"),
    (14, Wire::Bytes, "AST_thunk_node_array"),
    (15, Wire::Bytes, "AST_local_cell_reference_node_reference"),
    (
        16,
        Wire::Bytes,
        "AST_cross_table_cell_reference_node_reference",
    ),
    (17, Wire::Bytes, "AST_unknown_function_node_string"),
    (18, Wire::Varint, "AST_unknown_function_node_numArgs"),
    (19, Wire::Varint, "AST_date_node_suppress_date_format"),
    (20, Wire::Varint, "AST_date_node_suppress_time_format"),
    (21, Wire::Bytes, "AST_date_node_date_time_format"),
    (22, Wire::Varint, "AST_duration_node_style"),
    (23, Wire::Varint, "AST_duration_node_duration_unit_largest"),
    (24, Wire::Varint, "AST_duration_node_duration_unit_smallest"),
    (25, Wire::Bytes, "AST_whitespace"),
    (26, Wire::Bytes, "AST_column"),
    (27, Wire::Bytes, "AST_row"),
    (28, Wire::Bytes, "AST_cross_table_reference_extra_info"),
    (29, Wire::Varint, "AST_duration_node_use_automatic_units"),
    (30, Wire::Bytes, "AST_uid_coordinate"),
    (33, Wire::Bytes, "AST_sticky_bits"),
    (34, Wire::Bytes, "AST_let_identifier"),
    (35, Wire::Bytes, "AST_let_whitespace"),
    (36, Wire::Varint, "AST_let_is_continuation"),
    (37, Wire::Varint, "AST_symbol"),
    (38, Wire::Bytes, "AST_tract_list"),
    (39, Wire::Bytes, "AST_category_ref"),
    (40, Wire::Bytes, "AST_colon_tract"),
    (41, Wire::Bytes, "AST_frozen_sticky_bits"),
    (42, Wire::Varint, "AST_number_node_decimal_low"),
    (43, Wire::Varint, "AST_number_node_decimal_high"),
    (44, Wire::Bytes, "AST_category_levels"),
    (45, Wire::Bytes, "AST_lambda_idents"),
    (46, Wire::Varint, "AST_range_context"),
    (47, Wire::Varint, "upgrade_node_type"),
];

/// `TSCE.FormulaArchive`.
const FORMULA_FIELDS: &[(u32, Wire, &str)] = &[
    (1, Wire::Bytes, "AST_node_array"),
    (2, Wire::Varint, "host_column"),
    (3, Wire::Varint, "host_row"),
    (4, Wire::Varint, "host_column_is_negative"),
    (5, Wire::Varint, "host_row_is_negative"),
    (6, Wire::Bytes, "translation_flags"),
    (7, Wire::Bytes, "host_table_uid"),
    (8, Wire::Bytes, "host_column_uid"),
    (9, Wire::Bytes, "host_row_uid"),
];

fn field_schema(table: &[(u32, Wire, &str)], number: u32) -> Option<Wire> {
    table
        .iter()
        .find(|(n, _, _)| *n == number)
        .map(|(_, wire, _)| *wire)
}

/// How deeply a `THUNK_NODE`'s nested arrays may nest before every
/// thunk-descending path — validation, symbol collection and printing — gives
/// up and returns a bounded result rather than recursing further.
///
/// A thunk holds a lazily evaluated argument, so genuine nesting is a handful of
/// levels deep: an `IF` inside an `IF`, a `LAMBDA` body, a spilled `SEQUENCE`.
/// Nothing the app writes comes near this. The bound exists only so a hostile
/// document — the reviewer built one 40000 thunks deep in 574KB — turns into an
/// error or a `#DEPTH!` marker instead of a stack overflow that `catch_unwind`
/// cannot catch. 256 is far past any real formula and far below the depth a
/// recursion of this shape needs to exhaust the stack, matching how the rest of
/// the crate bounds itself (the varint's 10-byte cap, the document walk's 16
/// levels).
const THUNK_DEPTH_LIMIT: usize = 256;

// -- the AST -----------------------------------------------------------------

/// One `ASTNodeArchive`, kept as its message and read through accessors.
///
/// Keeping the wire form is what makes "unknown fields verbatim" free: a node
/// re-encodes to the bytes it came from, whatever this crate does or does not
/// understand about it.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub kind: u32,
    raw: Message,
}

impl Node {
    pub fn decode(message: Message) -> Option<Node> {
        let kind = message.varint(1)? as u32;
        node::is_known(kind).then_some(Node { kind, raw: message })
    }

    pub fn message(&self) -> &Message {
        &self.raw
    }

    pub fn encode(&self) -> Vec<u8> {
        self.raw.encode()
    }

    /// Every field is one the 15.3.1 schema has, at the wire type the schema
    /// gives it, and so is every submessage this module descends into.
    ///
    /// Returns the offending field's number and a reason. This is the check
    /// that turns "the bytes parsed" into "the bytes are an AST node".
    pub fn validate(&self) -> Result<(), String> {
        self.validate_at(0)
    }

    /// The body of [`Node::validate`], carrying the thunk-nesting depth so a
    /// runaway chain of `THUNK_NODE`s becomes a bounded error rather than a
    /// stack overflow.
    fn validate_at(&self, depth: usize) -> Result<(), String> {
        for field in &self.raw.fields {
            let Some(wire) = field_schema(NODE_FIELDS, field.number) else {
                return Err(format!(
                    "node type {} has no field {} in the 15.3.1 schema",
                    self.kind, field.number
                ));
            };
            if wire_of(&field.value) != wire || matches!(field.value, Value::Fixed32(_)) {
                return Err(format!(
                    "node type {} field {} is the wrong wire type",
                    self.kind, field.number
                ));
            }
            if wire == Wire::Bytes {
                if let Value::Bytes(bytes) = &field.value {
                    if let Some(reason) = validate_submessage(field.number, bytes, depth) {
                        return Err(reason);
                    }
                }
            }
        }
        Ok(())
    }

    // -- typed operands ------------------------------------------------------

    /// Function index (field 2) and argument count (field 3).
    pub fn function(&self) -> Option<(u32, u32)> {
        let index = self.raw.varint(2)? as u32;
        Some((index, self.raw.varint(3).unwrap_or(0) as u32))
    }

    /// A number literal, exactly as stored.
    ///
    /// Numbers writes every literal **twice**: a lossy binary double at field 4
    /// and an IEEE-754 decimal128 at fields 42/43. The decimal is authoritative
    /// — `0.1` is exactly a tenth there and is not in the double — so this
    /// prefers it and falls back to the double when it is absent.
    pub fn number(&self) -> Option<Number> {
        match (self.raw.varint(42), self.raw.varint(43)) {
            (Some(low), Some(high)) => Some(Number::Decimal { low, high }),
            _ => self.double(4).map(Number::Double),
        }
    }

    pub fn boolean(&self) -> Option<bool> {
        self.raw.varint(5).map(|v| v != 0)
    }

    /// `TOKEN_NODE`'s boolean, which lives at field 10 and not field 5.
    pub fn token_boolean(&self) -> Option<bool> {
        self.raw.varint(10).map(|v| v != 0)
    }

    /// A string literal, stored **unescaped** — a `"` inside it is one
    /// character here and two in the formula text.
    pub fn string(&self) -> Option<String> {
        self.raw
            .bytes(6)
            .map(|b| String::from_utf8_lossy(b).into_owned())
    }

    /// Seconds since 2001-01-01, timezone-naive — the same epoch a date cell
    /// uses.
    pub fn date(&self) -> Option<f64> {
        self.double(7)
    }

    /// Duration magnitude (field 8) and unit code (field 9).
    pub fn duration(&self) -> Option<(f64, i64)> {
        Some((
            self.double(8)?,
            self.raw.varint(9).map(|v| v as i32 as i64).unwrap_or(0),
        ))
    }

    /// `ARRAY_NODE`'s shape: columns then rows.
    pub fn array_shape(&self) -> Option<(u32, u32)> {
        Some((
            self.raw.varint(11).unwrap_or(1) as u32,
            self.raw.varint(12).unwrap_or(1) as u32,
        ))
    }

    pub fn list_args(&self) -> Option<u32> {
        self.raw.varint(13).map(|v| v as u32)
    }

    /// An unrecognised function's name (field 17) and arity (field 18) — what
    /// the app writes so a formula it cannot resolve survives a round trip.
    pub fn unknown_function(&self) -> Option<(String, u32)> {
        let name = String::from_utf8_lossy(self.raw.bytes(17)?).into_owned();
        Some((name, self.raw.varint(18).unwrap_or(0) as u32))
    }

    pub fn whitespace(&self) -> Option<String> {
        self.raw
            .bytes(25)
            .map(|b| String::from_utf8_lossy(b).into_owned())
    }

    /// `LET_BIND_NODE`'s identifier, and whether it continues the binding list
    /// of the `LET` before it rather than opening a new one.
    pub fn let_binding(&self) -> Option<(String, bool)> {
        let name = String::from_utf8_lossy(self.raw.bytes(34)?).into_owned();
        Some((name, self.raw.varint(36).unwrap_or(0) != 0))
    }

    /// Symbol-table index of a `VAR_NODE` or a `LET_BIND_NODE`.
    pub fn symbol(&self) -> Option<u32> {
        self.raw.varint(37).map(|v| v as u32)
    }

    /// `LAMBDA_NODE`'s parameter names and the symbol index the first one has.
    pub fn lambda_idents(&self) -> Option<(Vec<String>, u32)> {
        let idents = self.raw.bytes(45).and_then(decode_nested)?;
        let names = idents
            .all(1)
            .filter_map(|v| match v {
                Value::Bytes(b) => Some(String::from_utf8_lossy(b).into_owned()),
                _ => None,
            })
            .collect();
        Some((names, idents.varint(2).unwrap_or(0) as u32))
    }

    /// A nested AST, for `THUNK_NODE`'s lazily evaluated argument.
    pub fn thunk(&self) -> Option<Ast> {
        Ast::decode(&self.raw.bytes(14).and_then(decode_nested)?)
    }

    /// The reference this node makes, when it makes one.
    pub fn reference(&self) -> Option<Reference> {
        Reference::of(self)
    }

    /// `CATEGORY_REF_NODE`'s payload — a reference into a categorised or pivot
    /// table, which addresses a *group* and not a rectangle.
    pub fn category_reference(&self) -> Option<CategoryReference> {
        let outer = self.raw.bytes(39).and_then(decode_nested)?;
        let archive = outer.bytes(1).and_then(decode_nested)?;
        let levels = self.raw.bytes(44).and_then(decode_nested);
        Some(CategoryReference {
            group_by: archive.bytes(1).and_then(uuid_of),
            column: archive.bytes(2).and_then(uuid_of),
            aggregate: archive.varint(3).unwrap_or(0) as u32,
            group_level: archive.varint(4).map(unzigzag).unwrap_or(0),
            groups: archive
                .bytes(6)
                .and_then(decode_nested)
                .map(|list| list.all(1).filter_map(bytes_uuid).collect())
                .unwrap_or_default(),
            show_aggregate_name: archive.varint(14).unwrap_or(0) != 0,
            column_group_level: levels.as_ref().and_then(|l| l.varint(1)).unwrap_or(0) as u32,
            row_group_level: levels.as_ref().and_then(|l| l.varint(2)).unwrap_or(0) as u32,
        })
    }

    /// How many operands this node takes off the stack, and how many it puts
    /// back.
    pub fn stack_effect(&self) -> (usize, usize) {
        match self.kind {
            node::ADDITION
            | node::SUBTRACTION
            | node::MULTIPLICATION
            | node::DIVISION
            | node::POWER
            | node::CONCATENATION
            | node::GREATER_THAN
            | node::GREATER_THAN_OR_EQUAL
            | node::LESS_THAN
            | node::LESS_THAN_OR_EQUAL
            | node::EQUAL_TO
            | node::NOT_EQUAL_TO
            | node::COLON
            | node::COLON_WITH_UIDS
            | node::INTERSECTION => (2, 1),
            node::NEGATION
            | node::PLUS_SIGN
            | node::PERCENT
            | node::SPILL_RANGE
            | node::APPEND_WHITESPACE
            | node::PREPEND_WHITESPACE
            | node::END_SCOPE
            | node::LAMBDA => (1, 1),
            node::LET_BIND => (1, 0),
            node::FUNCTION => (self.raw.varint(3).unwrap_or(0) as usize, 1),
            node::UNKNOWN_FUNCTION => (self.raw.varint(18).unwrap_or(0) as usize, 1),
            node::LIST => (self.raw.varint(13).unwrap_or(0) as usize, 1),
            node::ARRAY => {
                // Raw varints, not the u32-truncated `array_shape`: a malformed
                // node can carry a u64::MAX dimension, and multiplying two of
                // them must not overflow-panic. A saturated product underflows
                // the stack in `depth()`, which is the well-formed answer — an
                // array that claims more cells than the file could hold is
                // malformed, not a program.
                let columns = self.raw.varint(11).unwrap_or(1) as usize;
                let rows = self.raw.varint(12).unwrap_or(1) as usize;
                (columns.saturating_mul(rows), 1)
            }
            node::BEGIN_THUNK
            | node::END_THUNK
            | node::BEGIN_LAMBDA_THUNK
            | node::END_LAMBDA_THUNK => (0, 0),
            _ => (0, 1),
        }
    }

    fn double(&self, number: u32) -> Option<f64> {
        match self.raw.get(number) {
            Some(Value::Fixed64(bytes)) => Some(f64::from_le_bytes(*bytes)),
            _ => None,
        }
    }
}

/// Which submessages this module knows the shape of, and what it requires of
/// them. A field not listed is carried through unvalidated — and unread.
///
/// A submessage this module *does* claim to know is a validation failure when it
/// does not decode: the trust claim (see `Node::validate`) is that a decoded
/// node is really an AST node, and a field the schema says is a nested message
/// that will not parse as one is exactly the kind of misparse the check exists
/// to catch. Returning `None` — "valid" — for it would be the silent pass this
/// crate promises never to give.
fn validate_submessage(number: u32, bytes: &[u8], depth: usize) -> Option<String> {
    let expect: &[(u32, Wire, &str)] = match number {
        14 => {
            // A nested node array: recurse, which is the only way a thunk's
            // contents get the same guarantee as its parent's — but no deeper
            // than the thunk-depth limit, so a hostile chain of thunks is a
            // bounded error rather than a stack overflow.
            if depth >= THUNK_DEPTH_LIMIT {
                return Some(format!(
                    "thunk nesting exceeds the limit of {THUNK_DEPTH_LIMIT}"
                ));
            }
            let Some(array) = decode_nested(bytes) else {
                return Some("field 14's thunk array does not decode".to_string());
            };
            let Some(ast) = Ast::decode(&array) else {
                return Some("field 14's thunk array is not an AST node array".to_string());
            };
            return ast.validate_at(depth + 1).err();
        }
        26 | 27 => &[
            (1, Wire::Varint, "column_or_row"),
            (2, Wire::Varint, "absolute"),
        ],
        28 => &[
            (1, Wire::Bytes, "table_id"),
            (2, Wire::Bytes, "whitespace_after_sheet_name"),
            (3, Wire::Bytes, "whitespace_before_table_name"),
            (4, Wire::Bytes, "whitespace_after_table_name"),
            (5, Wire::Bytes, "whitespace_before_cell_address"),
        ],
        33 | 41 => &[
            (1, Wire::Varint, "begin_row_is_absolute"),
            (2, Wire::Varint, "begin_column_is_absolute"),
            (3, Wire::Varint, "end_row_is_absolute"),
            (4, Wire::Varint, "end_column_is_absolute"),
        ],
        40 => &[
            (1, Wire::Bytes, "relative_column"),
            (2, Wire::Bytes, "relative_row"),
            (3, Wire::Bytes, "absolute_column"),
            (4, Wire::Bytes, "absolute_row"),
            (5, Wire::Varint, "preserve_rectangular"),
        ],
        45 => &[
            (1, Wire::Bytes, "AST_identifier_string"),
            (2, Wire::Varint, "AST_first_symbol"),
            (3, Wire::Bytes, "AST_whitespace_before_idents"),
            (4, Wire::Bytes, "AST_whitespace_after_idents"),
        ],
        _ => return None,
    };
    let Some(message) = decode_nested(bytes) else {
        return Some(format!("field {number}'s submessage does not decode"));
    };
    for field in &message.fields {
        match field_schema(expect, field.number) {
            Some(wire) if wire == wire_of(&field.value) => {}
            _ => {
                return Some(format!(
                    "field {number}'s submessage has no field {} of that wire type",
                    field.number
                ))
            }
        }
    }
    None
}

fn uuid_of(bytes: &[u8]) -> Option<Uuid> {
    decode_nested(bytes).map(|m| Uuid::decode(&m))
}

fn bytes_uuid(value: &Value) -> Option<Uuid> {
    match value {
        Value::Bytes(bytes) => uuid_of(bytes),
        _ => None,
    }
}

/// `TSCE.CategoryReferenceArchive` — what a pivot cell's formula points at.
///
/// It names a *group* of the source table rather than a rectangle: a group-by
/// owner, the column being summarised, the aggregate applied and the path of
/// group UUIDs down the category tree. **This crate decodes it and does not
/// print it the way the app does** — see `FORMAT.md` §9. The 32 formulas of
/// `numbers-pivot.numbers` are the only ones in this corpus and Numbers spells
/// them `Source::$Units $January::Electric::Bicycles (Sum)`, which needs the
/// source table's group tree, its aggregate names and the `$` rules for a
/// group path; none of those has more than one document behind it here.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoryReference {
    pub group_by: Option<Uuid>,
    pub column: Option<Uuid>,
    /// `aggregate_type`. Apple publishes no names; 2 is Sum.
    pub aggregate: u32,
    pub group_level: i64,
    /// The group path, outermost first.
    pub groups: Vec<Uuid>,
    pub show_aggregate_name: bool,
    pub column_group_level: u32,
    pub row_group_level: u32,
}

/// A numeric literal as the archive stores it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Number {
    /// IEEE-754 decimal128, split into its two halves. Exact.
    Decimal { low: u64, high: u64 },
    /// The legacy binary double, when no decimal is present.
    Double(f64),
}

/// The decimal128 exponent bias, and the `high` word of an exact integer.
const DECIMAL128_BIAS: i32 = 6176;
const DECIMAL128_INTEGER_HIGH: u64 = (DECIMAL128_BIAS as u64) << 49;

impl Number {
    /// The literal as the app prints it.
    ///
    /// A decimal128 is rendered from its own digits rather than through a
    /// float, so `0.1` prints as `0.1` and not as the nearest double. The
    /// common case — an exact integer — has `high == 0x3040000000000000` and
    /// prints `low` verbatim, which is what keeps `1` from becoming `1.0`.
    pub fn text(&self) -> String {
        match *self {
            Number::Decimal { low, high } if high == DECIMAL128_INTEGER_HIGH => low.to_string(),
            Number::Decimal { low, high } => {
                let negative = high >> 63 != 0;
                let exponent = ((high >> 49) & 0x3fff) as i32 - DECIMAL128_BIAS;
                let mantissa = (u128::from(high & ((1u64 << 49) - 1)) << 64) | u128::from(low);
                let mut text = decimal_text(mantissa, exponent);
                if negative && mantissa != 0 {
                    text.insert(0, '-');
                }
                text
            }
            Number::Double(value) => plain_double(value),
        }
    }

    /// The value as a float, for callers that want a number rather than text.
    pub fn value(&self) -> f64 {
        match *self {
            Number::Decimal { low, high } if high == DECIMAL128_INTEGER_HIGH => low as f64,
            Number::Decimal { .. } => self.text().parse().unwrap_or(f64::NAN),
            Number::Double(value) => value,
        }
    }
}

/// `mantissa × 10^exponent` as plain digits — never exponent notation, which
/// no formula text uses.
fn decimal_text(mantissa: u128, exponent: i32) -> String {
    let digits = mantissa.to_string();
    if exponent >= 0 {
        let mut out = digits;
        for _ in 0..exponent {
            out.push('0');
        }
        return out;
    }
    let shift = (-exponent) as usize;
    let mut whole = String::new();
    let mut fraction = String::new();
    if digits.len() > shift {
        whole.push_str(&digits[..digits.len() - shift]);
        fraction.push_str(&digits[digits.len() - shift..]);
    } else {
        whole.push('0');
        for _ in 0..shift - digits.len() {
            fraction.push('0');
        }
        fraction.push_str(&digits);
    }
    while fraction.ends_with('0') {
        fraction.pop();
    }
    if fraction.is_empty() {
        whole
    } else {
        format!("{whole}.{fraction}")
    }
}

/// A double without exponent notation. Rust's `Display` for `f64` never uses
/// one, so this is `to_string` with `-0` normalised.
fn plain_double(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    value.to_string()
}

/// `ASTNodeArrayArchive` — the RPN node stream.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Ast {
    pub nodes: Vec<Node>,
}

impl Ast {
    /// Decode a node array. Returns `None` if any element is not a node.
    pub fn decode(array: &Message) -> Option<Ast> {
        let mut nodes = Vec::new();
        for field in &array.fields {
            if field.number != 1 {
                return None;
            }
            let Value::Bytes(bytes) = &field.value else {
                return None;
            };
            nodes.push(Node::decode(decode_nested(bytes)?)?);
        }
        Some(Ast { nodes })
    }

    pub fn validate(&self) -> Result<(), String> {
        self.validate_at(0)
    }

    /// [`Ast::validate`] carrying the thunk-nesting depth, so the recursion
    /// through nested thunk arrays is bounded.
    fn validate_at(&self, depth: usize) -> Result<(), String> {
        for node in &self.nodes {
            node.validate_at(depth)?;
        }
        Ok(())
    }

    /// Whether the node stream is a well-formed RPN program: no operator ever
    /// pops an operand that is not there, and exactly one value is left at the
    /// end.
    ///
    /// This is the check that tells an AST apart from the other things a
    /// document is full of. A `TSCE.CellRecordExpandedArchive` is
    /// `{1: column, 2: row}` and a `TSP.Reference` is `{1: identifier}`; both
    /// parse as nodes and pass the field-level schema check, and both are
    /// nonsense as programs — `{1: 3, 2: 0}` is a multiplication with an empty
    /// stack. Requiring the stream to *evaluate* is what makes a walk of a
    /// whole document safe.
    pub fn is_well_formed(&self) -> bool {
        self.depth() == Some(1)
    }

    /// Stack depth after the whole stream, or `None` if it ever underflows.
    fn depth(&self) -> Option<usize> {
        let mut depth = 0usize;
        for node in &self.nodes {
            let (pops, pushes) = node.stack_effect();
            depth = depth.checked_sub(pops)?;
            depth += pushes;
        }
        Some(depth)
    }

    /// The node stream as text, without the leading `=`.
    ///
    /// [`Formula::text`] is this with the `=` in front. It is separate because
    /// a chart's data references are ASTs that are not whole formulas: each one
    /// is a reference wrapped in function 175, and what a reader wants printed
    /// is the reference, not `=(null)(…)`.
    pub fn text(&self, at: Site<'_>) -> String {
        let mut printer = Printer::new(at);
        printer.run(self);
        printer.finish()
    }

    /// Every value the node stream leaves on the stack, in order.
    ///
    /// A whole formula leaves exactly one, which is what [`Ast::text`] returns.
    /// A *fragment* need not: a chart's data reference is a function of index
    /// 175 over any number of operands, and dropping that node leaves one
    /// string per operand — three of them where a series is fed by three
    /// separate columns.
    pub fn texts(&self, at: Site<'_>) -> Vec<String> {
        let mut printer = Printer::new(at);
        printer.run(self);
        printer.stack
    }

    /// Re-encode to the `ASTNodeArrayArchive` bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut array = Message::default();
        for node in &self.nodes {
            array.fields.push(crate::pb::Field {
                number: 1,
                value: Value::Bytes(node.encode()),
            });
        }
        array.encode()
    }
}

/// `TSCE.FormulaArchive` — an AST plus, when it has one, its host cell.
#[derive(Debug, Clone, PartialEq)]
pub struct Formula {
    pub ast: Ast,
    /// `host_column`/`host_row`, with the two sign bools applied. Absent from
    /// every formula 15.3.1 writes into a table's FORMULA list: the host is the
    /// cell that holds the key.
    pub host: Option<(i64, i64)>,
    /// `translation_flags`, kept whole — an Excel import or a percent-formatted
    /// result is a property of the formula, not of the AST.
    pub flags: Option<Message>,
    raw: Message,
}

impl Formula {
    pub fn decode(archive: &Message) -> Option<Formula> {
        let ast = Ast::decode(&archive.bytes(1).and_then(decode_nested)?)?;
        let host = match (archive.varint(2), archive.varint(3)) {
            (Some(column), Some(row)) => {
                let column = column as i64;
                let row = row as i64;
                Some((
                    if archive.varint(4).unwrap_or(0) != 0 {
                        -column
                    } else {
                        column
                    },
                    if archive.varint(5).unwrap_or(0) != 0 {
                        -row
                    } else {
                        row
                    },
                ))
            }
            _ => None,
        };
        Some(Formula {
            ast,
            host,
            flags: archive.bytes(6).and_then(decode_nested),
            raw: archive.clone(),
        })
    }

    /// A formula from a node stream and nothing else.
    ///
    /// Which is exactly what a stored formula is: **every** `TSCE.FormulaArchive`
    /// in a table's FORMULA list in this corpus carries field 1 and no other
    /// field — no host cell, no translation flags, no host UUIDs. The host is
    /// the cell that holds the key, which is why one entry can serve a whole
    /// filled column.
    pub fn from_ast(ast: Ast) -> Formula {
        let mut raw = Message::default();
        raw.set_in_order(1, Value::Bytes(ast.encode()));
        Formula {
            ast,
            host: None,
            flags: None,
            raw,
        }
    }

    pub fn message(&self) -> &Message {
        &self.raw
    }

    /// Re-encode the archive, honouring an edit to [`Formula::ast`].
    ///
    /// The node array (field 1) is re-emitted from `ast`, so a caller that
    /// rewrites the AST gets the rewrite back rather than the bytes it started
    /// with. Every other field — the host cell (2–5), the translation flags (6)
    /// and the host UUIDs (7–9) — is preserved verbatim from the archive this
    /// formula decoded, which is what keeps `encode(decode(x)) == x` for an
    /// unedited formula (`ast.encode()` reproduces field 1 byte for byte).
    ///
    /// `host` and `flags` are **decoded read views**, not inputs to this: they
    /// are conveniences for inspecting the archive, and reconstructing a signed
    /// host cell from `Option<(i64, i64)>` is lossy where the app has not been
    /// observed to go. To change them, edit the archive through [`Formula::ast`]
    /// and the fields carried on `raw`. Nothing in this corpus carries either.
    pub fn encode(&self) -> Vec<u8> {
        let mut archive = self.raw.clone();
        archive.set_in_order(1, Value::Bytes(self.ast.encode()));
        archive.encode()
    }

    pub fn validate(&self) -> Result<(), String> {
        for field in &self.raw.fields {
            match field_schema(FORMULA_FIELDS, field.number) {
                Some(wire) if wire == wire_of(&field.value) => {}
                _ => {
                    return Err(format!(
                        "formula archive has no field {} of that wire type",
                        field.number
                    ))
                }
            }
        }
        self.ast.validate()
    }

    /// The formula as text, in the spelling Numbers uses.
    pub fn text(&self, at: Site<'_>) -> String {
        format!("={}", self.ast.text(at))
    }
}

// -- references --------------------------------------------------------------

/// A `TSP.CFUUIDArchive` as the `TSP.UUID` the rest of the crate uses.
///
/// The two are the same 128-bit value in different clothes: the CFUUID form —
/// which is the only form an AST uses — splits it into four 32-bit words, so
/// `lower = w0 | w1 << 32` and `upper = w2 | w3 << 32`. Confirmed word for word
/// against the `base_owner_uid` of every table in `numbers-formulas.numbers`.
pub fn uuid_from_cfuuid(message: &Message) -> Option<Uuid> {
    let word = |n: u32| message.varint(n).map(|v| v as u32);
    Some(Uuid {
        lower: u64::from(word(2)?) | (u64::from(word(3)?) << 32),
        upper: u64::from(word(4)?) | (u64::from(word(5)?) << 32),
    })
}

/// One axis of a reference: absolute index, offset from the host, or the whole
/// axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Absolute(i64),
    Relative(i64),
    /// The axis is not constrained — a whole-column reference has no row and a
    /// whole-row reference has no column.
    Unbounded,
}

impl Axis {
    /// Resolve against the host cell's index on the same axis.
    pub fn resolve(self, host: i64) -> Option<i64> {
        match self {
            Axis::Absolute(index) => Some(index),
            Axis::Relative(offset) => Some(host + offset),
            Axis::Unbounded => None,
        }
    }

    pub fn is_absolute(self) -> bool {
        matches!(self, Axis::Absolute(_))
    }
}

/// A reference made by one node: a cell, a range, or an unbounded strip.
#[derive(Debug, Clone, PartialEq)]
pub struct Reference {
    pub column: Axis,
    pub row: Axis,
    /// End of a range; equal to the start for a single cell.
    pub column_end: Axis,
    pub row_end: Axis,
    /// The table's `base_owner_uid`, when the reference leaves its own table.
    pub table: Option<Uuid>,
    /// A range rather than a single cell — which is what decides whether the
    /// app prints header names or A1 notation.
    pub is_range: bool,
    /// Both axes saturated: a stored `#REF!`.
    pub is_error: bool,
}

/// Rows saturate at 31 bits and columns at 15 — the two sentinels differ, and
/// using the row one for a column breaks every whole-row reference.
const ROW_MAX: i64 = 0x7fff_ffff;
const COLUMN_MAX: i64 = 0x7fff;

impl Reference {
    fn of(node: &Node) -> Option<Reference> {
        let table = node
            .raw
            .bytes(28)
            .and_then(decode_nested)
            .and_then(|info| info.bytes(1).and_then(decode_nested))
            .and_then(|id| uuid_from_cfuuid(&id));
        match node.kind {
            node::CELL_REFERENCE | node::REFERENCE_ERROR_WITH_UIDS | node::UID_REFERENCE => {
                let column = coordinate(&node.raw, 26);
                let row = coordinate(&node.raw, 27);
                if column == Axis::Unbounded && row == Axis::Unbounded {
                    return None;
                }
                // A stored `#REF!` saturates **both** axes at once, each at its
                // own sentinel — that is how the app writes a fully-deleted
                // precedent, and it is what FORMAT.md §9 records. A single
                // saturated axis is not enough: a whole-column reference leaves
                // its row `Unbounded` (never `Absolute(max)`), so requiring both
                // never mistakes one for an error.
                let is_error = node.kind == node::REFERENCE_ERROR_WITH_UIDS
                    || (saturated(column, COLUMN_MAX) && saturated(row, ROW_MAX));
                Some(Reference {
                    column,
                    row,
                    column_end: column,
                    row_end: row,
                    table,
                    is_range: false,
                    is_error,
                })
            }
            node::COLON_TRACT => {
                let tract = node.raw.bytes(40).and_then(decode_nested)?;
                let sticky = node
                    .raw
                    .bytes(33)
                    .and_then(decode_nested)
                    .unwrap_or_default();
                let begin_row_absolute = sticky.varint(1).unwrap_or(0) != 0;
                let begin_column_absolute = sticky.varint(2).unwrap_or(0) != 0;
                let end_row_absolute = sticky.varint(3).unwrap_or(0) != 0;
                let end_column_absolute = sticky.varint(4).unwrap_or(0) != 0;
                let (column, column_end) =
                    tract_axis(&tract, 1, 3, begin_column_absolute, end_column_absolute);
                let (row, row_end) = tract_axis(&tract, 2, 4, begin_row_absolute, end_row_absolute);
                // A tract endpoint sitting on its axis's saturation sentinel is
                // a `#REF!`, exactly as for a single cell: the row's is
                // `0x7fffffff`, the column's `0x7fff`. Without this a range
                // ending at the sentinel would print a spurious column letter
                // and a nine-digit row rather than the error it stands for.
                let is_error = saturated(column, COLUMN_MAX)
                    || saturated(column_end, COLUMN_MAX)
                    || saturated(row, ROW_MAX)
                    || saturated(row_end, ROW_MAX);
                Some(Reference {
                    column,
                    row,
                    column_end,
                    row_end,
                    table,
                    is_range: true,
                    is_error,
                })
            }
            _ => None,
        }
    }

    /// Resolve to zero-based indices against a host cell, where the reference
    /// is bounded on that axis.
    pub fn resolve(&self, host: (i64, i64)) -> ResolvedReference {
        ResolvedReference {
            column: self.column.resolve(host.0),
            row: self.row.resolve(host.1),
            column_end: self.column_end.resolve(host.0),
            row_end: self.row_end.resolve(host.1),
        }
    }
}

/// A reference with its host applied. `None` on an axis means "the whole
/// column" or "the whole row".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedReference {
    pub column: Option<i64>,
    pub row: Option<i64>,
    pub column_end: Option<i64>,
    pub row_end: Option<i64>,
}

fn saturated(axis: Axis, max: i64) -> bool {
    matches!(axis, Axis::Absolute(v) | Axis::Relative(v) if v == max)
}

/// `ASTColumnCoordinateArchive` / `ASTRowCoordinateArchive`. The index is a
/// **zigzag** `sint32` — unlike a colon tract's offsets, which are plain
/// `int32` varints. Mixing the two is the classic silent-corruption bug here.
fn coordinate(node: &Message, number: u32) -> Axis {
    // The whole-column / whole-row signal is the **submessage** being absent —
    // a `CELL_REFERENCE_NODE` with no `AST_row` (27) is a whole column. A
    // submessage that is present but carries no index is a different thing: it
    // is offset 0 (`unzigzag(0)`), relative or absolute per field 2, not
    // unbounded. Numbers writes the index explicitly, but a zero index that
    // relied on the varint default would otherwise read as a whole axis.
    let Some(message) = node.bytes(number).and_then(decode_nested) else {
        return Axis::Unbounded;
    };
    let index = unzigzag(message.varint(1).unwrap_or(0));
    if message.varint(2).unwrap_or(0) != 0 {
        Axis::Absolute(index)
    } else {
        Axis::Relative(index)
    }
}

fn unzigzag(value: u64) -> i64 {
    ((value >> 1) ^ (0u64.wrapping_sub(value & 1))) as i64
}

/// One axis of an `ASTColonTractArchive`.
///
/// Both lists are written when the two ends disagree about absoluteness, and
/// the sticky bits say which one to read for each end. An absent `range_end`
/// means "the same as `range_begin`", not zero and not unbounded.
fn tract_axis(
    tract: &Message,
    relative: u32,
    absolute: u32,
    begin_absolute: bool,
    end_absolute: bool,
) -> (Axis, Axis) {
    let relative_range = range(tract, relative);
    let absolute_range = range(tract, absolute);
    let begin = if begin_absolute {
        absolute_range.map(|(begin, _)| Axis::Absolute(begin))
    } else {
        relative_range.map(|(begin, _)| Axis::Relative(begin))
    };
    let end = if end_absolute {
        absolute_range.map(|(_, end)| Axis::Absolute(end))
    } else {
        relative_range.map(|(_, end)| Axis::Relative(end))
    };
    // An axis with neither list is the whole axis. An endpoint that lands on
    // the saturation sentinel is not handled here — it is a stored `#REF!`, and
    // `Reference::of` flags it after both axes are read.
    (
        begin.unwrap_or(Axis::Unbounded),
        end.unwrap_or(Axis::Unbounded),
    )
}

/// `{range_begin, range_end?}` as signed values, whichever list it came from.
fn range(tract: &Message, number: u32) -> Option<(i64, i64)> {
    let Value::Bytes(bytes) = tract.get(number)? else {
        return None;
    };
    let entry = decode_nested(bytes)?;
    let begin = entry.varint(1)? as i32 as i64;
    let end = entry.varint(2).map(|v| v as i32 as i64).unwrap_or(begin);
    Some((begin, end))
}

// -- the function table ------------------------------------------------------

include!("formula_functions.rs");

/// The published name of a built-in function.
///
/// **Function names are not localised.** The oracle on a German-locale machine
/// reports `SUM`, `VLOOKUP` and `IFERROR`, and separates arguments with a
/// comma; the only spellings Numbers localises in a formula are its *operators*
/// (`×`, `÷`, `−`, `≥`, `≤`, `≠`), and those are the same in every locale too.
pub fn function_name(index: u32) -> Option<&'static str> {
    FUNCTIONS
        .binary_search_by_key(&index, |(id, _)| *id)
        .ok()
        .map(|at| FUNCTIONS[at].1)
}

/// The index of a built-in function by name, for writing one.
///
/// The inverse of [`function_name`], and case-insensitive because a caller
/// types `sum(` as readily as `SUM(`. Names are not localised — see above.
pub fn function_index(name: &str) -> Option<u32> {
    FUNCTIONS
        .iter()
        .find(|(_, known)| known.eq_ignore_ascii_case(name))
        .map(|(index, _)| *index)
}

// -- rendering ---------------------------------------------------------------

/// What a table looks like to a formula: its identity, its name, and the header
/// cells that give its rows and columns names.
#[derive(Debug, Clone, Default)]
pub struct TableNames {
    /// `base_owner_uid` — the identity every cross-table reference is written
    /// with. Not the table's name and not its `table_id`.
    pub uid: Uuid,
    pub name: String,
    pub sheet: Option<String>,
    pub header_rows: usize,
    pub header_columns: usize,
    /// Text of the naming header cell of each column, where it has one.
    pub column_names: Vec<Option<String>>,
    /// Text of the naming header cell of each row.
    pub row_names: Vec<Option<String>>,
}

impl TableNames {
    /// Drop every name that more than one row or column of this table carries.
    ///
    /// **A name has to identify one row or column of its table or the app will
    /// not use it.** `numbers-links.numbers` has eleven rows whose header cell
    /// all read `Item name`, and Numbers prints `=C2×D2` there rather than
    /// `=Qty Item name` — which would name ten cells at once. Rows whose header
    /// is unique keep their name in the same table.
    pub fn drop_ambiguous(&mut self) {
        deduplicate(&mut self.column_names);
        deduplicate(&mut self.row_names);
    }

    /// The name of a column, which only body columns of a table with header
    /// rows have.
    pub fn column_name(&self, column: i64) -> Option<&str> {
        if self.header_rows == 0 || column < self.header_columns as i64 {
            return None;
        }
        self.column_names
            .get(usize::try_from(column).ok()?)?
            .as_deref()
    }

    pub fn row_name(&self, row: i64) -> Option<&str> {
        if self.header_columns == 0 || row < self.header_rows as i64 {
            return None;
        }
        self.row_names.get(usize::try_from(row).ok()?)?.as_deref()
    }
}

fn deduplicate(names: &mut [Option<String>]) {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for name in names.iter().flatten() {
        *counts.entry(name.as_str()).or_default() += 1;
    }
    let duplicated: Vec<String> = counts
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .map(|(name, _)| name.to_string())
        .collect();
    for name in names.iter_mut() {
        if name
            .as_deref()
            .is_some_and(|n| duplicated.iter().any(|d| d == n))
        {
            *name = None;
        }
    }
}

/// Every table in the document, and which of them owns each header name.
#[derive(Debug, Clone, Default)]
pub struct Names {
    pub tables: Vec<TableNames>,
    /// Header name → the table that gets to use it without a prefix.
    owners: BTreeMap<String, usize>,
}

impl Names {
    /// Build the index. The order of `tables` is the order names are claimed
    /// in: where two tables carry the same header name, the **first** keeps it
    /// bare and the others are prefixed with their table's name.
    ///
    /// Evidence: `numbers-formulas.numbers` has `Menge` as a column header of
    /// both `Daten` and `Daten2`, and Numbers prints `SUM(Menge)` for the first
    /// and `SUM(Daten2::Menge)` for the second.
    pub fn new(mut tables: Vec<TableNames>) -> Names {
        for table in &mut tables {
            table.drop_ambiguous();
        }
        let mut owners: BTreeMap<String, usize> = BTreeMap::new();
        for (index, table) in tables.iter().enumerate() {
            for name in table
                .column_names
                .iter()
                .chain(table.row_names.iter())
                .flatten()
            {
                owners.entry(name.clone()).or_insert(index);
            }
        }
        Names { tables, owners }
    }

    pub fn by_uid(&self, uid: Uuid) -> Option<usize> {
        self.tables.iter().position(|t| t.uid == uid)
    }

    fn owns(&self, table: usize, name: &str) -> bool {
        self.owners.get(name) == Some(&table)
    }
}

/// Where a formula sits: the document's tables, which one holds this formula,
/// and the cell the relative references count from.
#[derive(Debug, Clone, Copy)]
pub struct Site<'a> {
    pub names: &'a Names,
    /// Index into `names.tables`, when the formula belongs to a table.
    pub table: Option<usize>,
    /// Host cell, zero-based. A formula in a data list has none of its own and
    /// takes the position of the cell that keys it.
    pub host: (i64, i64),
}

impl<'a> Site<'a> {
    pub fn new(names: &'a Names, table: Option<usize>, host: (i64, i64)) -> Site<'a> {
        Site { names, table, host }
    }

    /// A site with no tables at all — enough to print a formula that makes no
    /// reference, and honest about the ones it cannot resolve.
    pub fn anonymous(names: &'a Names) -> Site<'a> {
        Site {
            names,
            table: None,
            host: (0, 0),
        }
    }
}

/// The stack machine that turns the node stream back into text.
struct Printer<'a> {
    at: Site<'a>,
    stack: Vec<String>,
    /// Open `LET` bindings, innermost last: name, value, and whether this
    /// binding continues the `LET` outside it rather than opening its own.
    scopes: Vec<(String, String, bool)>,
    /// Bindings closed by an `END_SCOPE_NODE` that continued an outer `LET`,
    /// waiting for the scope that opened it.
    pending: Vec<(String, String)>,
    /// Symbol index → identifier, collected before rendering because a
    /// `VAR_NODE` is written *inside* the thunk that its `LAMBDA_NODE` names.
    symbols: BTreeMap<u32, String>,
    /// Nodes this printer did not understand, by type.
    pub unsupported: Vec<u32>,
    /// How many thunks deep this printer is nested, so a runaway chain of
    /// `THUNK_NODE`s stops at [`THUNK_DEPTH_LIMIT`] instead of overflowing.
    depth: usize,
}

impl<'a> Printer<'a> {
    fn new(at: Site<'a>) -> Printer<'a> {
        Printer {
            at,
            stack: Vec::new(),
            scopes: Vec::new(),
            pending: Vec::new(),
            symbols: BTreeMap::new(),
            unsupported: Vec::new(),
            depth: 0,
        }
    }

    fn finish(mut self) -> String {
        self.stack.pop().unwrap_or_default()
    }

    fn pop(&mut self, count: usize) -> Vec<String> {
        let at = self.stack.len().saturating_sub(count);
        self.stack.split_off(at)
    }

    /// Gather every symbol name in the whole tree — including the ones written
    /// inside thunks, because a `VAR_NODE` is emitted inside the thunk its
    /// `LAMBDA_NODE` names — in a **single** descent, bounded by depth.
    ///
    /// This runs once, from [`Printer::run`] at the top. The thunk printers it
    /// feeds inherit the finished map (see [`Printer::node`]) and never collect
    /// again: re-collecting per level is what made printing nested thunks
    /// super-linear, re-decoding the whole remaining payload at every level.
    fn collect_symbols(&mut self, ast: &Ast, depth: usize) {
        if depth >= THUNK_DEPTH_LIMIT {
            return;
        }
        for node in &ast.nodes {
            match node.kind {
                node::LET_BIND => {
                    if let (Some((name, _)), Some(symbol)) = (node.let_binding(), node.symbol()) {
                        self.symbols.insert(symbol, name);
                    }
                }
                node::LAMBDA => {
                    if let Some((names, first)) = node.lambda_idents() {
                        for (offset, name) in names.into_iter().enumerate() {
                            self.symbols.insert(first + offset as u32, name);
                        }
                    }
                }
                node::THUNK => {
                    if let Some(inner) = node.thunk() {
                        self.collect_symbols(&inner, depth + 1);
                    }
                }
                _ => {}
            }
        }
    }

    /// Collect the symbols of the whole tree once, then print it.
    fn run(&mut self, ast: &Ast) {
        self.collect_symbols(ast, 0);
        self.print(ast);
    }

    /// Print a node stream onto the stack, with the symbols already collected.
    /// Thunk printers call this rather than [`Printer::run`] so the single
    /// symbol pass is never repeated.
    fn print(&mut self, ast: &Ast) {
        for node in &ast.nodes {
            self.node(node);
        }
    }

    fn binary(&mut self, operator: &str) {
        let operands = self.pop(2);
        let left = operands.first().cloned().unwrap_or_default();
        let right = operands.get(1).cloned().unwrap_or_default();
        self.stack.push(format!("{left}{operator}{right}"));
    }

    fn node(&mut self, node: &Node) {
        match node.kind {
            node::ADDITION => self.binary("+"),
            node::SUBTRACTION => self.binary("\u{2212}"),
            node::MULTIPLICATION => self.binary("\u{d7}"),
            node::DIVISION => self.binary("\u{f7}"),
            node::POWER => self.binary("^"),
            node::CONCATENATION => self.binary("&"),
            node::GREATER_THAN => self.binary(">"),
            node::GREATER_THAN_OR_EQUAL => self.binary("\u{2265}"),
            node::LESS_THAN => self.binary("<"),
            node::LESS_THAN_OR_EQUAL => self.binary("\u{2264}"),
            node::EQUAL_TO => self.binary("="),
            node::NOT_EQUAL_TO => self.binary("\u{2260}"),
            node::NEGATION => {
                let operand = self.pop(1).pop().unwrap_or_default();
                self.stack.push(format!("\u{2212}{operand}"));
            }
            node::PLUS_SIGN => {
                let operand = self.pop(1).pop().unwrap_or_default();
                self.stack.push(format!("+{operand}"));
            }
            node::PERCENT => {
                let operand = self.pop(1).pop().unwrap_or_default();
                self.stack.push(format!("{operand}%"));
            }
            node::FUNCTION => {
                let (index, arity) = node.function().unwrap_or((0, 0));
                let arguments = self.pop(arity as usize);
                // A function with no published name is what Numbers itself
                // prints as `(null)` — id 337, the internal function behind a
                // spilled cell, is the one in this corpus.
                let name = function_name(index).unwrap_or("(null)");
                self.stack.push(format!("{name}({})", arguments.join(",")));
            }
            node::NUMBER => {
                let text = node.number().map(|n| n.text()).unwrap_or_default();
                self.stack.push(text);
            }
            node::BOOLEAN => {
                let value = node.boolean().unwrap_or(false);
                self.stack
                    .push(if value { "TRUE" } else { "FALSE" }.to_string());
            }
            node::TOKEN => {
                let value = node.token_boolean().unwrap_or(false);
                self.stack
                    .push(if value { "TRUE" } else { "FALSE" }.to_string());
            }
            node::STRING => {
                let literal = node.string().unwrap_or_default().replace('"', "\"\"");
                self.stack.push(format!("\"{literal}\""));
            }
            node::DATE => {
                // No document in this corpus has a date literal; the text is
                // the stored format's own, and this is the honest fallback.
                let seconds = node.date().unwrap_or(0.0);
                self.stack.push(format!("DATE({seconds})"));
            }
            node::DURATION => {
                let (magnitude, unit) = node.duration().unwrap_or((0.0, 0));
                self.stack
                    .push(format!("DURATION({},{unit})", plain_double(magnitude)));
            }
            node::EMPTY_ARGUMENT => self.stack.push(String::new()),
            node::ARRAY => {
                let (columns, rows) = node.array_shape().unwrap_or((1, 1));
                let cells = self.pop((columns as usize).saturating_mul(rows as usize));
                let lines: Vec<String> = cells
                    .chunks(columns.max(1) as usize)
                    .map(|row| row.join(","))
                    .collect();
                self.stack.push(format!("{{{}}}", lines.join(";")));
            }
            node::LIST => {
                let arity = node.list_args().unwrap_or(1) as usize;
                let items = self.pop(arity);
                self.stack.push(format!("({})", items.join(",")));
            }
            node::THUNK => {
                if self.depth >= THUNK_DEPTH_LIMIT {
                    // A runaway chain of thunks stops here with a bounded
                    // marker rather than recursing into a stack overflow.
                    self.unsupported.push(node::THUNK);
                    self.stack.push("#DEPTH!".to_string());
                    return;
                }
                let text = node
                    .thunk()
                    .map(|inner| {
                        // The symbols are already collected for the whole tree,
                        // so print without re-collecting — walking once is what
                        // keeps nested thunks linear rather than super-linear.
                        let mut printer = Printer::new(self.at);
                        printer.symbols = self.symbols.clone();
                        printer.depth = self.depth + 1;
                        printer.print(&inner);
                        let body = printer.stack.pop().unwrap_or_default();
                        self.unsupported.extend(printer.unsupported);
                        body
                    })
                    .unwrap_or_default();
                self.stack.push(text);
            }
            node::COLON => self.binary(":"),
            node::REFERENCE_ERROR | node::REFERENCE_ERROR_WITH_UIDS => {
                self.stack.push("#REF!".to_string())
            }
            node::UNKNOWN_FUNCTION => {
                let (name, arity) = node.unknown_function().unwrap_or_default();
                let arguments = self.pop(arity as usize);
                self.stack.push(format!("{name}({})", arguments.join(",")));
            }
            node::APPEND_WHITESPACE => {
                let space = node.whitespace().unwrap_or_default();
                if let Some(top) = self.stack.last_mut() {
                    top.push_str(&space);
                }
            }
            node::PREPEND_WHITESPACE => {
                let space = node.whitespace().unwrap_or_default();
                if let Some(top) = self.stack.last_mut() {
                    top.insert_str(0, &space);
                }
            }
            node::BEGIN_THUNK
            | node::END_THUNK
            | node::BEGIN_LAMBDA_THUNK
            | node::END_LAMBDA_THUNK => {}
            node::CELL_REFERENCE | node::COLON_TRACT | node::UID_REFERENCE => {
                let text = match node.reference() {
                    Some(reference) => self.reference_text(&reference),
                    None => "#REF!".to_string(),
                };
                self.stack.push(text);
            }
            node::LET_BIND => {
                let (name, continuation) = node.let_binding().unwrap_or_default();
                let value = self.pop(1).pop().unwrap_or_default();
                self.scopes.push((name, value, continuation));
            }
            node::VAR => {
                let name = node
                    .symbol()
                    .and_then(|symbol| self.symbols.get(&symbol).cloned())
                    .unwrap_or_else(|| "?".to_string());
                self.stack.push(name);
            }
            node::END_SCOPE => {
                // One `END_SCOPE_NODE` per binding, innermost first. A binding
                // marked as a continuation belongs to the `LET` outside it, so
                // it waits here until the scope that opened that `LET` closes.
                // `=LET(x,2,y,3,x×y)` is two binds and two end-scopes.
                let body = self.pop(1).pop().unwrap_or_default();
                let Some((name, value, continuation)) = self.scopes.pop() else {
                    self.stack.push(body);
                    return;
                };
                self.pending.insert(0, (name, value));
                if continuation {
                    self.stack.push(body);
                    return;
                }
                let mut parts: Vec<String> = Vec::new();
                for (name, value) in self.pending.drain(..) {
                    parts.push(name);
                    parts.push(value);
                }
                parts.push(body);
                self.stack.push(format!("LET({})", parts.join(",")));
            }
            node::LAMBDA => {
                let body = self.pop(1).pop().unwrap_or_default();
                let (names, _) = node.lambda_idents().unwrap_or_default();
                let mut parts = names;
                parts.push(body);
                self.stack.push(format!("LAMBDA({})", parts.join(",")));
            }
            node::INTERSECTION => self.binary(" "),
            node::SPILL_RANGE => {
                let operand = self.pop(1).pop().unwrap_or_default();
                self.stack.push(format!("{operand}#"));
            }
            node::LINKED_CELL_REF | node::LINKED_COLUMN_REF | node::LINKED_ROW_REF => {
                // A conditional-highlighting rule is written about *the cell
                // it styles*, and this is how that cell is named: a node with
                // no coordinates at all, only the table it is linked to. No
                // dictionary reports a conditional rule, so there is no app
                // spelling to match; these three are the crate's own, and
                // `=#CELL>0` reads the way the rule reads.
                let name = match node.kind {
                    node::LINKED_COLUMN_REF => "#COLUMN",
                    node::LINKED_ROW_REF => "#ROW",
                    _ => "#CELL",
                };
                self.stack.push(name.to_string());
            }
            node::CATEGORY_REF | node::VIEW_TRACT_REF => {
                // A reference into a categorised or pivot table names group
                // values, not coordinates, and the app prints those names. It
                // is decoded (the archive is preserved) and not rendered.
                self.unsupported.push(node.kind);
                self.stack.push("#CATEGORY!".to_string());
            }
            other => {
                self.unsupported.push(other);
                self.stack.push(format!("#NODE{other}"));
            }
        }
    }

    /// A reference in the app's own spelling: header names where the target has
    /// them, A1 notation where it does not, and a table prefix only where one
    /// is needed.
    fn reference_text(&self, reference: &Reference) -> String {
        if reference.is_error {
            return "#REF!".to_string();
        }
        let target = match reference.table {
            // A reference that carries an owner UUID but names no table this
            // document holds is a broken cross-table reference — a deleted
            // table, or an owner this crate could not follow. It must not fall
            // through to a bare local reference, which would be indistinguishable
            // from a same-table one; every other unknown here prints a marker.
            Some(uid) => match self.at.names.by_uid(uid) {
                Some(target) => Some(target),
                None => return "#TABLE!".to_string(),
            },
            None => self.at.table,
        };
        let Some(target) = target else {
            return self.a1_text(reference, None);
        };
        let table = &self.at.names.tables[target];
        let resolved = reference.resolve(self.at.host);

        // Names, when the target has them. A range never takes them: the app
        // prints `SUM(B2:B4)` even in a table whose columns and rows are all
        // named.
        if !reference.is_range || resolved.column.is_none() || resolved.row.is_none() {
            let column_name = resolved.column.and_then(|c| table.column_name(c));
            let row_name = resolved.row.and_then(|r| table.row_name(r));
            let name = match (resolved.column, resolved.row, column_name, row_name) {
                (Some(_), Some(_), Some(column), Some(row)) => Some(format!(
                    "{}{} {}{}",
                    dollar(reference.column),
                    quote_name(column),
                    dollar(reference.row),
                    quote_name(row)
                )),
                (Some(_), None, Some(column), _) => Some(format!(
                    "{}{}",
                    dollar(reference.column),
                    quote_name(column)
                )),
                (None, Some(_), _, Some(row)) => {
                    Some(format!("{}{}", dollar(reference.row), quote_name(row)))
                }
                _ => None,
            };
            if let Some(name) = name {
                let bare = Some(target) == self.at.table
                    || self
                        .at
                        .names
                        .owns(target, column_or_row_key(table, &resolved));
                return if bare {
                    name
                } else {
                    format!("{}::{name}", table.name)
                };
            }
        }
        let prefix = (Some(target) != self.at.table).then(|| table.name.clone());
        self.a1_text(reference, prefix)
    }

    fn a1_text(&self, reference: &Reference, prefix: Option<String>) -> String {
        let resolved = reference.resolve(self.at.host);
        let begin = a1(
            resolved.column,
            resolved.row,
            reference.column.is_absolute(),
            reference.row.is_absolute(),
        );
        let mut text = if reference.is_range
            && (resolved.column != resolved.column_end || resolved.row != resolved.row_end)
        {
            let end = a1(
                resolved.column_end,
                resolved.row_end,
                reference.column_end.is_absolute(),
                reference.row_end.is_absolute(),
            );
            format!("{begin}:{end}")
        } else {
            begin
        };
        if let Some(prefix) = prefix {
            text = format!("{prefix}::{text}");
        }
        text
    }
}

/// The name a reference is looked up under when deciding whether it needs a
/// table prefix: the column name for a whole column, the row name for a whole
/// row, and the column name for a cell.
fn column_or_row_key<'a>(table: &'a TableNames, resolved: &ResolvedReference) -> &'a str {
    match (resolved.column, resolved.row) {
        (Some(column), _) => table.column_name(column).unwrap_or_default(),
        (None, Some(row)) => table.row_name(row).unwrap_or_default(),
        _ => "",
    }
}

fn dollar(axis: Axis) -> &'static str {
    if axis.is_absolute() {
        "$"
    } else {
        ""
    }
}

/// Characters that make Numbers wrap a header name in single quotes, with an
/// embedded `'` doubled.
///
/// Proven by `numbers-formulas.numbers`: `A+B` becomes `'A+B'`, `Preis
/// (netto)` becomes `'Preis (netto)'`, `it's` becomes `'it''s'` and
/// `groesser-gleich` becomes `'groesser-gleich'` — while `x y` and `SUM` are
/// printed bare, so neither a space nor a function's name forces quoting. The
/// rest of the set below is the formula grammar's other operators and is
/// Inferred.
const NEEDS_QUOTING: &[char] = &[
    '+', '-', '*', '/', '^', '&', '%', '=', '<', '>', '(', ')', ',', ';', ':', '"', '\'', '$', '#',
    '{', '}', '\u{d7}', '\u{f7}', '\u{2212}', '\u{2260}', '\u{2265}', '\u{2264}',
];

fn quote_name(name: &str) -> String {
    if name.chars().any(|c| NEEDS_QUOTING.contains(&c)) {
        format!("'{}'", name.replace('\'', "''"))
    } else {
        name.to_string()
    }
}

/// A1 notation from zero-based indices. An unbounded axis drops out, which is
/// how `B` (a whole column) and `2:2` (a whole row) are written.
fn a1(column: Option<i64>, row: Option<i64>, column_absolute: bool, row_absolute: bool) -> String {
    // A resolved index below zero is off the top or left edge of the table —
    // a relative reference counted from a host too near the origin. It is a
    // `#REF!`, not a hybrid of a spliced marker and a zero/negative row.
    if column.is_some_and(|c| c < 0) || row.is_some_and(|r| r < 0) {
        return "#REF!".to_string();
    }
    let mut text = String::new();
    if let Some(column) = column {
        if column_absolute {
            text.push('$');
        }
        text.push_str(&column_letters(column));
    }
    if let Some(row) = row {
        if row_absolute {
            text.push('$');
        }
        text.push_str(&(row + 1).to_string());
    }
    if text.is_empty() {
        "#REF!".to_string()
    } else {
        text
    }
}

/// Zero-based column index as letters: 0 is A, 25 is Z, 26 is AA.
pub fn column_letters(column: i64) -> String {
    if column < 0 {
        return "#REF!".to_string();
    }
    let mut index = column + 1;
    let mut letters = Vec::new();
    while index > 0 {
        let remainder = ((index - 1) % 26) as u8;
        letters.push((b'A' + remainder) as char);
        index = (index - 1) / 26;
    }
    letters.iter().rev().collect()
}

/// An A1 reference back to indices — the inverse of [`column_letters`] plus a
/// row, used by the CLI.
pub fn parse_a1(text: &str) -> Option<(usize, usize)> {
    let letters: String = text
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    let digits: String = text[letters.len()..].to_string();
    if letters.is_empty() || digits.is_empty() {
        return None;
    }
    // Checked, so an over-long reference from the CLI (twenty letters overflow
    // a `usize`) is rejected rather than panicking.
    let mut column = 0usize;
    for c in letters.chars() {
        column = column
            .checked_mul(26)?
            .checked_add(c.to_ascii_uppercase() as usize - 'A' as usize + 1)?;
    }
    let row: usize = digits.parse().ok()?;
    (row > 0).then(|| (row - 1, column - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node_of(fields: &[(u32, Value)]) -> Node {
        let mut message = Message::default();
        for (number, value) in fields {
            message.fields.push(crate::pb::Field {
                number: *number,
                value: value.clone(),
            });
        }
        Node::decode(message).unwrap()
    }

    fn sub(fields: &[(u32, Value)]) -> Value {
        let mut message = Message::default();
        for (number, value) in fields {
            message.fields.push(crate::pb::Field {
                number: *number,
                value: value.clone(),
            });
        }
        Value::Bytes(message.encode())
    }

    #[test]
    fn column_letters_count_the_way_the_app_does() {
        assert_eq!(column_letters(0), "A");
        assert_eq!(column_letters(25), "Z");
        assert_eq!(column_letters(26), "AA");
        assert_eq!(column_letters(51), "AZ");
        assert_eq!(column_letters(52), "BA");
        assert_eq!(column_letters(701), "ZZ");
        assert_eq!(column_letters(702), "AAA");
    }

    #[test]
    fn a1_parses_back() {
        assert_eq!(parse_a1("A1"), Some((0, 0)));
        assert_eq!(parse_a1("B3"), Some((2, 1)));
        assert_eq!(parse_a1("AA10"), Some((9, 26)));
        assert_eq!(parse_a1("A0"), None);
        assert_eq!(parse_a1("12"), None);
    }

    /// The decimal128 shortcut: an exact integer has `high` equal to the bias
    /// shifted, and `low` is the integer.
    #[test]
    fn an_integer_literal_prints_without_a_point() {
        let number = Number::Decimal {
            low: 42,
            high: DECIMAL128_INTEGER_HIGH,
        };
        assert_eq!(number.text(), "42");
        assert_eq!(DECIMAL128_INTEGER_HIGH, 0x3040_0000_0000_0000);
    }

    #[test]
    fn a_decimal_literal_prints_its_own_digits() {
        // 1 × 10^-1, as Numbers writes 0.1.
        let tenth = Number::Decimal {
            low: 1,
            high: ((DECIMAL128_BIAS - 1) as u64) << 49,
        };
        assert_eq!(tenth.text(), "0.1");
        // 1 × 10^-5 — the case a float would print as 1e-5.
        let small = Number::Decimal {
            low: 1,
            high: ((DECIMAL128_BIAS - 5) as u64) << 49,
        };
        assert_eq!(small.text(), "0.00001");
        // 3 × 10^2, an integer written with an exponent.
        let three_hundred = Number::Decimal {
            low: 3,
            high: ((DECIMAL128_BIAS + 2) as u64) << 49,
        };
        assert_eq!(three_hundred.text(), "300");
    }

    /// The two coordinate encodings, which are different and adjacent: a
    /// cell reference's index is zigzag, a colon tract's offset is not.
    #[test]
    fn a_cell_coordinate_is_zigzag_and_a_tract_offset_is_not() {
        let node = node_of(&[
            (1, Value::Varint(u64::from(node::CELL_REFERENCE))),
            (26, sub(&[(1, Value::Varint(1)), (2, Value::Varint(0))])),
            (27, sub(&[(1, Value::Varint(3)), (2, Value::Varint(0))])),
        ]);
        let reference = node.reference().unwrap();
        assert_eq!(reference.column, Axis::Relative(-1));
        assert_eq!(reference.row, Axis::Relative(-2));

        let tract = node_of(&[
            (1, Value::Varint(u64::from(node::COLON_TRACT))),
            (
                33,
                sub(&[
                    (1, Value::Varint(0)),
                    (2, Value::Varint(0)),
                    (3, Value::Varint(0)),
                    (4, Value::Varint(0)),
                ]),
            ),
            (
                40,
                sub(&[
                    (1, sub(&[(1, Value::Varint((-1i64) as u64))])),
                    (
                        2,
                        sub(&[
                            (1, Value::Varint((-3i64) as u64)),
                            (2, Value::Varint((-1i64) as u64)),
                        ]),
                    ),
                ]),
            ),
        ]);
        let reference = tract.reference().unwrap();
        assert_eq!(reference.column, Axis::Relative(-1));
        assert_eq!(reference.row, Axis::Relative(-3));
        assert_eq!(reference.row_end, Axis::Relative(-1));
    }

    /// An omitted `range_end` means "the same as `range_begin`" — not zero,
    /// and not unbounded.
    #[test]
    fn an_omitted_range_end_repeats_the_beginning() {
        let mut tract = Message::default();
        tract.set(3, sub(&[(1, Value::Varint(2))]));
        assert_eq!(range(&tract, 3), Some((2, 2)));
        tract.set(4, sub(&[(1, Value::Varint(2)), (2, Value::Varint(5))]));
        assert_eq!(range(&tract, 4), Some((2, 5)));
    }

    #[test]
    fn a_name_is_quoted_only_when_it_has_to_be() {
        assert_eq!(quote_name("Wert"), "Wert");
        assert_eq!(quote_name("x y"), "x y");
        assert_eq!(quote_name("SUM"), "SUM");
        assert_eq!(quote_name("A+B"), "'A+B'");
        assert_eq!(quote_name("groesser-gleich"), "'groesser-gleich'");
        assert_eq!(quote_name("it's"), "'it''s'");
        assert_eq!(quote_name("Preis (netto)"), "'Preis (netto)'");
    }

    /// Every field a node carries has to be one the schema has, at the wire
    /// type the schema gives it — the check that makes a decode trustworthy.
    #[test]
    fn validation_rejects_a_field_the_schema_does_not_have() {
        let good = node_of(&[
            (1, Value::Varint(u64::from(node::NUMBER))),
            (4, Value::Fixed64(1.0f64.to_le_bytes())),
        ]);
        assert!(good.validate().is_ok());

        let hole = node_of(&[
            (1, Value::Varint(u64::from(node::NUMBER))),
            (31, Value::Varint(1)),
        ]);
        assert!(
            hole.validate().is_err(),
            "31 and 32 are holes in the schema"
        );

        // Field 36 is a varint from 14.4 onwards. A message there is the
        // ≤13.1 shape and must not be accepted quietly.
        let old_let = node_of(&[
            (1, Value::Varint(u64::from(node::LET_BIND))),
            (36, sub(&[(1, Value::Varint(0))])),
        ]);
        assert!(old_let.validate().is_err());
    }

    fn msg(fields: &[(u32, Value)]) -> Message {
        let mut message = Message::default();
        for (number, value) in fields {
            message.fields.push(crate::pb::Field {
                number: *number,
                value: value.clone(),
            });
        }
        message
    }

    fn zigzag(value: i64) -> u64 {
        ((value << 1) ^ (value >> 63)) as u64
    }

    /// A `THUNK_NODE` whose array holds another, `depth` levels deep, around a
    /// single number — the shape of the reviewer's stack-overflow document.
    fn nested_thunk_ast(depth: usize) -> Ast {
        let number = node_of(&[
            (1, Value::Varint(u64::from(node::NUMBER))),
            (4, Value::Fixed64(1.0f64.to_le_bytes())),
        ]);
        let mut array = msg(&[(1, Value::Bytes(number.encode()))]);
        for _ in 0..depth {
            let thunk = node_of(&[
                (1, Value::Varint(u64::from(node::THUNK))),
                (14, Value::Bytes(array.encode())),
            ]);
            array = msg(&[(1, Value::Bytes(thunk.encode()))]);
        }
        Ast::decode(&array).unwrap()
    }

    /// A hostile chain of thunks is a bounded error and a bounded string, never
    /// a stack overflow (which `catch_unwind` cannot catch).
    #[test]
    fn a_runaway_thunk_chain_is_bounded() {
        let names = Names::default();
        let deep = nested_thunk_ast(5000);
        let error = deep
            .validate()
            .expect_err("deep thunk nesting must be rejected, not overflow the stack");
        assert!(error.contains("thunk nesting exceeds"), "{error}");
        let text = deep.text(Site::anonymous(&names));
        assert!(
            text.contains("#DEPTH!"),
            "printing stops at the depth limit with a marker: {text}"
        );

        // A chain just under the limit still validates and prints clean.
        let shallow = nested_thunk_ast(THUNK_DEPTH_LIMIT - 1);
        assert!(shallow.validate().is_ok());
        assert!(!shallow.text(Site::anonymous(&names)).contains("#DEPTH!"));
    }

    /// A malformed `ARRAY_NODE` with `u64::MAX` dimensions must not overflow the
    /// stack-effect multiply; it is simply not a well-formed program.
    #[test]
    fn an_over_large_array_does_not_overflow() {
        let node = node_of(&[
            (1, Value::Varint(u64::from(node::ARRAY))),
            (11, Value::Varint(u64::MAX)),
            (12, Value::Varint(u64::MAX)),
        ]);
        assert_eq!(node.stack_effect(), (usize::MAX, 1));
        let ast = Ast { nodes: vec![node] };
        assert!(!ast.is_well_formed());
    }

    /// `COLON_NODE_WITH_UIDS` (45) pops two operands like `COLON` (29) and
    /// `INTERSECTION` (69) — not zero, which made a formula using it read as not
    /// well-formed.
    #[test]
    fn colon_with_uids_takes_two_operands() {
        let colon = node_of(&[(1, Value::Varint(u64::from(node::COLON_WITH_UIDS)))]);
        assert_eq!(colon.stack_effect(), (2, 1));
        let cell = || {
            node_of(&[
                (1, Value::Varint(u64::from(node::CELL_REFERENCE))),
                (26, sub(&[(1, Value::Varint(0)), (2, Value::Varint(1))])),
                (27, sub(&[(1, Value::Varint(0)), (2, Value::Varint(1))])),
            ])
        };
        let ast = Ast {
            nodes: vec![cell(), cell(), colon],
        };
        assert!(ast.is_well_formed());
    }

    /// A coordinate submessage that is *present* but carries no index is offset
    /// 0, not a whole axis — the whole-axis signal is the submessage being
    /// absent.
    #[test]
    fn a_present_coordinate_with_no_index_is_offset_zero() {
        let node = node_of(&[
            (1, Value::Varint(u64::from(node::CELL_REFERENCE))),
            (26, sub(&[(2, Value::Varint(0))])),
            (27, sub(&[(2, Value::Varint(0))])),
        ]);
        let reference = node.reference().unwrap();
        assert_eq!(reference.column, Axis::Relative(0));
        assert_eq!(reference.row, Axis::Relative(0));

        let whole = node_of(&[
            (1, Value::Varint(u64::from(node::CELL_REFERENCE))),
            (26, sub(&[(1, Value::Varint(2)), (2, Value::Varint(0))])),
        ]);
        assert_eq!(
            whole.reference().unwrap().row,
            Axis::Unbounded,
            "field 27 absent is a whole column"
        );
    }

    /// A stored `#REF!` saturates both axes; one saturated axis alone is not an
    /// error. (The corpus's only saturated node is `REFERENCE_ERROR_WITH_UIDS`,
    /// which is an error by kind, so this is the check that exercises the axes.)
    #[test]
    fn a_cell_reference_is_an_error_only_when_both_axes_saturate() {
        let reference = |column: i64, row: i64| {
            node_of(&[
                (1, Value::Varint(u64::from(node::CELL_REFERENCE))),
                (
                    26,
                    sub(&[(1, Value::Varint(zigzag(column))), (2, Value::Varint(1))]),
                ),
                (
                    27,
                    sub(&[(1, Value::Varint(zigzag(row))), (2, Value::Varint(1))]),
                ),
            ])
            .reference()
            .unwrap()
        };
        assert!(reference(COLUMN_MAX, ROW_MAX).is_error);
        assert!(
            !reference(COLUMN_MAX, 3).is_error,
            "only the column saturated"
        );
        assert!(!reference(3, ROW_MAX).is_error, "only the row saturated");
    }

    /// A range whose endpoint lands on its axis's saturation sentinel is a
    /// `#REF!`, exactly as for a single cell.
    #[test]
    fn a_tract_ending_on_the_sentinel_is_an_error() {
        let tract = node_of(&[
            (1, Value::Varint(u64::from(node::COLON_TRACT))),
            (
                33,
                sub(&[
                    (1, Value::Varint(1)),
                    (2, Value::Varint(1)),
                    (3, Value::Varint(1)),
                    (4, Value::Varint(1)),
                ]),
            ),
            (
                40,
                sub(&[
                    (3, sub(&[(1, Value::Varint(1))])),
                    (
                        4,
                        sub(&[(1, Value::Varint(1)), (2, Value::Varint(ROW_MAX as u64))]),
                    ),
                ]),
            ),
        ]);
        assert!(tract.reference().unwrap().is_error);
    }

    /// A reference that carries an owner UUID no table in the document has
    /// prints a marker, never a bare local reference that would look like a
    /// same-table one.
    #[test]
    fn an_unresolvable_cross_table_owner_prints_a_marker() {
        let cfuuid = sub(&[
            (2, Value::Varint(1)),
            (3, Value::Varint(0)),
            (4, Value::Varint(0)),
            (5, Value::Varint(0)),
        ]);
        let node = node_of(&[
            (1, Value::Varint(u64::from(node::CELL_REFERENCE))),
            (26, sub(&[(1, Value::Varint(0)), (2, Value::Varint(1))])),
            (27, sub(&[(1, Value::Varint(0)), (2, Value::Varint(1))])),
            (28, sub(&[(1, cfuuid)])),
        ]);
        let names = Names::default();
        let ast = Ast { nodes: vec![node] };
        assert_eq!(ast.text(Site::anonymous(&names)), "#TABLE!");
    }

    /// A relative reference resolving before the origin is a clean `#REF!`, not
    /// a spliced marker joined to a zero or negative row.
    #[test]
    fn a_reference_resolving_negative_is_a_clean_ref_error() {
        let node = node_of(&[
            (1, Value::Varint(u64::from(node::CELL_REFERENCE))),
            (
                26,
                sub(&[(1, Value::Varint(zigzag(-1))), (2, Value::Varint(0))]),
            ),
            (27, sub(&[(1, Value::Varint(0)), (2, Value::Varint(0))])),
        ]);
        let names = Names::default();
        let ast = Ast { nodes: vec![node] };
        assert_eq!(ast.text(Site::anonymous(&names)), "#REF!");
    }

    /// An over-long A1 reference from the CLI is rejected, not a panic.
    #[test]
    fn parse_a1_rejects_an_over_long_reference() {
        let huge = format!("{}1", "A".repeat(20));
        assert_eq!(parse_a1(&huge), None);
        assert_eq!(parse_a1("ZZ100"), Some((99, 701)));
    }

    /// Editing `Formula::ast` is reflected by `encode`; an unedited formula
    /// re-encodes to its own bytes.
    #[test]
    fn editing_the_ast_is_reflected_by_formula_encode() {
        let literal = |value: f64| {
            node_of(&[
                (1, Value::Varint(u64::from(node::NUMBER))),
                (4, Value::Fixed64(value.to_le_bytes())),
            ])
        };
        let archive = msg(&[(
            1,
            Value::Bytes(msg(&[(1, Value::Bytes(literal(1.0).encode()))]).encode()),
        )]);
        let original = archive.encode();
        let mut formula = Formula::decode(&archive).unwrap();
        assert_eq!(formula.encode(), original, "unedited: byte-identical");

        formula.ast = Ast::decode(&msg(&[(1, Value::Bytes(literal(2.0).encode()))])).unwrap();
        let bytes = formula.encode();
        assert_ne!(bytes, original, "the edit changed the bytes");
        let round = Formula::decode(&Message::decode(&bytes).unwrap()).unwrap();
        assert_eq!(round.ast.nodes[0].number().unwrap().value(), 2.0);
    }
}
