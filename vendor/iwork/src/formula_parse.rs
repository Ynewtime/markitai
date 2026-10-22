//! Formulas from text — `=SUM(B2:B4)` to the node stream Numbers writes.
//!
//! [`crate::formula`] reads a `TSCE` formula and prints it; this is the other
//! direction, and it is the only place in this crate that builds an AST from
//! nothing. Ground rule 3 says copy rather than synthesise, and every node here
//! is copied — from the shapes `numbers-formulas.numbers` carries, dumped node
//! by node and reproduced field for field:
//!
//! ```text
//! =B2+1        CELL_REFERENCE{26:{1:1,2:0}, 27:{1:0,2:0}}  NUMBER  ADDITION
//! =(B2+1)*2    … ADDITION  LIST{13:1}  NUMBER  MULTIPLICATION
//! =SUM(B2:B4)  COLON_TRACT{33:{0,0,0,0}, 40:{…, 5:1}}  FUNCTION{2:168, 3:1}
//! =LEFT("abcdef", 3)  STRING  NUMBER  FUNCTION{2:76, 3:2}
//! ```
//!
//! Three details are easy to get wrong and are what the fixture settles:
//!
//! * **A parenthesis is a node.** `(x)` writes a `LIST_NODE` with one argument
//!   after the expression, not nothing at all.
//! * **The two coordinate encodings differ.** `AST_column`/`AST_row` are zigzag
//!   `sint32`; the colon tract's lists are plain `int32`, so −1 is ten bytes.
//! * **A number literal is written twice** — a `double` at field 4 and the
//!   decimal128 halves at 42/43 — and the decimal is the authoritative one.
//!
//! What is *not* parsed is what could not then be registered in the
//! calculation engine (see [`crate::calc`]): a reference to another table, a
//! whole row or column, a header name, an array, `LET`/`LAMBDA`. Each is
//! refused by name rather than parsed into a formula the app would recalculate
//! wrongly.

use crate::formula::{node, Ast, Node};
use crate::pb::{Message, Value};

/// Parse a formula, with the host cell the references are relative to.
///
/// The leading `=` is optional. `host` is `(column, row)`, zero-based, the same
/// order [`crate::formula::Formula::host`] uses.
pub fn parse(text: &str, host: (i64, i64)) -> Result<Ast, String> {
    let tokens = lex(text.strip_prefix('=').unwrap_or(text))?;
    let mut parser = Parser {
        tokens,
        at: 0,
        host,
        nodes: Vec::new(),
    };
    parser.expression(0)?;
    if let Some(token) = parser.tokens.get(parser.at) {
        return Err(format!("unexpected {} after the formula", token.describe()));
    }
    let ast = Ast {
        nodes: parser.nodes,
    };
    // The same two checks `Table::audit` applies to a formula this crate reads:
    // a node stream that is not a well-formed program is one the app would
    // refuse, and one this crate must not write.
    ast.validate()?;
    if !ast.is_well_formed() {
        return Err("the formula is not a well-formed expression".into());
    }
    Ok(ast)
}

// -- tokens ------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// A number literal, kept as written so the decimal is exact.
    Number(String),
    Text(String),
    /// A bare word: a function name, a reference, or a name this crate refuses.
    Word(String),
    Punct(&'static str),
}

impl Token {
    fn describe(&self) -> String {
        match self {
            Token::Number(text) => format!("number {text}"),
            Token::Text(text) => format!("string {text:?}"),
            Token::Word(word) => format!("'{word}'"),
            Token::Punct(text) => format!("'{text}'"),
        }
    }
}

/// The operators, longest first so `<=` is not read as `<` and `=`.
const PUNCT: [&str; 20] = [
    "<=", ">=", "<>", "≤", "≥", "≠", "+", "-", "−", "*", "×", "/", "÷", "^", "&", "=", "<", ">",
    "%", ",",
];

fn lex(text: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut rest = text;
    'next: while !rest.is_empty() {
        let first = rest.chars().next().expect("not empty");
        if first.is_whitespace() {
            rest = &rest[first.len_utf8()..];
            continue;
        }
        if first == '"' {
            // A `""` inside a string is one quote — the app stores the
            // unescaped text and escapes it again when it prints the formula.
            let mut value = String::new();
            let mut chars = rest[1..].char_indices();
            loop {
                let Some((at, c)) = chars.next() else {
                    return Err("a string literal is missing its closing quote".into());
                };
                if c != '"' {
                    value.push(c);
                    continue;
                }
                match chars.clone().next() {
                    Some((_, '"')) => {
                        value.push('"');
                        chars.next();
                    }
                    _ => {
                        rest = &rest[1 + at + 1..];
                        break;
                    }
                }
            }
            tokens.push(Token::Text(value));
            continue;
        }
        if first.is_ascii_digit()
            || (first == '.' && rest[1..].starts_with(|c: char| c.is_ascii_digit()))
        {
            let mut end = 0;
            let bytes = rest.as_bytes();
            while end < bytes.len() && (bytes[end].is_ascii_digit() || bytes[end] == b'.') {
                end += 1;
            }
            // An exponent, but only when it is one: `1e3` is a number and `A1`
            // is a reference, so the `e` has to be followed by digits.
            if end < bytes.len() && (bytes[end] | 0x20) == b'e' {
                let mut after = end + 1;
                if after < bytes.len() && (bytes[after] == b'+' || bytes[after] == b'-') {
                    after += 1;
                }
                if after < bytes.len() && bytes[after].is_ascii_digit() {
                    end = after;
                    while end < bytes.len() && bytes[end].is_ascii_digit() {
                        end += 1;
                    }
                }
            }
            tokens.push(Token::Number(rest[..end].to_string()));
            rest = &rest[end..];
            continue;
        }
        if first == '$' || first.is_alphabetic() || first == '_' {
            let end = rest
                .find(|c: char| !(c.is_alphanumeric() || c == '$' || c == '_' || c == '.'))
                .unwrap_or(rest.len());
            tokens.push(Token::Word(rest[..end].to_string()));
            rest = &rest[end..];
            continue;
        }
        if first == '(' || first == ')' || first == ':' || first == ';' {
            // A semicolon separates arguments in some locales' spelling; the
            // app itself writes commas and this accepts both.
            let text = match first {
                '(' => "(",
                ')' => ")",
                ':' => ":",
                _ => ",",
            };
            tokens.push(Token::Punct(text));
            rest = &rest[1..];
            continue;
        }
        for symbol in PUNCT {
            if let Some(tail) = rest.strip_prefix(symbol) {
                tokens.push(Token::Punct(match symbol {
                    "−" => "-",
                    "×" => "*",
                    "÷" => "/",
                    "≤" => "<=",
                    "≥" => ">=",
                    "≠" => "<>",
                    other => other,
                }));
                rest = tail;
                continue 'next;
            }
        }
        return Err(format!("'{first}' is not something a formula can hold"));
    }
    Ok(tokens)
}

// -- the parser --------------------------------------------------------------

struct Parser {
    tokens: Vec<Token>,
    at: usize,
    host: (i64, i64),
    nodes: Vec<Node>,
}

/// Binding power and node type of each infix operator, loosest first.
const INFIX: [(&str, u8, u32); 14] = [
    ("=", 1, node::EQUAL_TO),
    ("<>", 1, node::NOT_EQUAL_TO),
    ("<", 1, node::LESS_THAN),
    ("<=", 1, node::LESS_THAN_OR_EQUAL),
    (">", 1, node::GREATER_THAN),
    (">=", 1, node::GREATER_THAN_OR_EQUAL),
    ("&", 2, node::CONCATENATION),
    ("+", 3, node::ADDITION),
    ("-", 3, node::SUBTRACTION),
    ("*", 4, node::MULTIPLICATION),
    ("/", 4, node::DIVISION),
    ("×", 4, node::MULTIPLICATION),
    ("÷", 4, node::DIVISION),
    ("^", 6, node::POWER),
];

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.at)
    }

    fn eat(&mut self, punct: &str) -> bool {
        if self.peek() == Some(&Token::Punct(leak(punct))) {
            self.at += 1;
            return true;
        }
        false
    }

    fn push(&mut self, node: Message) {
        self.nodes
            .push(Node::decode(node).expect("every node built here has a known type"));
    }

    /// Precedence climbing. `power` is the lowest binding power this call will
    /// consume, which is how `2+3*4` binds the multiplication tighter.
    fn expression(&mut self, power: u8) -> Result<(), String> {
        self.unary()?;
        while let Some(Token::Punct(symbol)) = self.peek() {
            let Some(&(_, binding, kind)) = INFIX.iter().find(|(text, _, _)| text == symbol) else {
                break;
            };
            if binding < power {
                break;
            }
            self.at += 1;
            // `^` is right-associative, every other operator left.
            let next = if kind == node::POWER {
                binding
            } else {
                binding + 1
            };
            self.expression(next)?;
            self.push(just(kind));
        }
        Ok(())
    }

    fn unary(&mut self) -> Result<(), String> {
        if self.eat("-") {
            self.unary()?;
            self.push(just(node::NEGATION));
            return Ok(());
        }
        if self.eat("+") {
            self.unary()?;
            self.push(just(node::PLUS_SIGN));
            return Ok(());
        }
        self.primary()?;
        while self.eat("%") {
            self.push(just(node::PERCENT));
        }
        Ok(())
    }

    fn primary(&mut self) -> Result<(), String> {
        let Some(token) = self.peek().cloned() else {
            return Err("the formula ends where a value was expected".into());
        };
        self.at += 1;
        match token {
            Token::Number(text) => {
                let value = crate::table::Decimal::parse(&text)
                    .ok_or_else(|| format!("{text} is not a number this crate can write"))?;
                self.push(number(value)?);
                Ok(())
            }
            Token::Text(value) => {
                let mut message = just(node::STRING);
                message.set_in_order(6, Value::Bytes(value.into_bytes()));
                self.push(message);
                Ok(())
            }
            Token::Punct("(") => {
                self.expression(0)?;
                if !self.eat(")") {
                    return Err("a '(' is missing its ')'".into());
                }
                // The parenthesis is a node of its own: a one-argument list,
                // which is what the app writes for `=(B2+1)*2`.
                let mut message = just(node::LIST);
                message.set_in_order(13, Value::Varint(1));
                self.push(message);
                Ok(())
            }
            Token::Word(word) => self.word(&word),
            other => Err(format!("{} is not a value", other.describe())),
        }
    }

    /// A word is a function call, a reference, or something refused by name.
    fn word(&mut self, word: &str) -> Result<(), String> {
        if self.peek() == Some(&Token::Punct("(")) {
            self.at += 1;
            return self.call(word);
        }
        if word.contains('.') {
            return Err(format!(
                "{word}: a reference to another table is refused — the calculation engine \
                 would have to be told about a cell this crate cannot resolve, and a formula \
                 the engine only half knows about is one the app recalculates wrongly"
            ));
        }
        let begin = Coordinate::parse(word).ok_or_else(|| {
            format!(
                "'{word}' is neither a cell reference nor a function this crate knows — \
                 a header name, a defined name and a bare TRUE are all refused rather than \
                 guessed at"
            )
        })?;
        if !self.eat(":") {
            self.push(begin.cell(self.host)?);
            return Ok(());
        }
        let Some(Token::Word(second)) = self.peek().cloned() else {
            return Err("a ':' with no cell after it".into());
        };
        self.at += 1;
        let end = Coordinate::parse(&second)
            .ok_or_else(|| format!("'{second}' is not a cell reference"))?;
        self.push(begin.range(&end, self.host)?);
        Ok(())
    }

    fn call(&mut self, name: &str) -> Result<(), String> {
        let index = crate::formula::function_index(name).ok_or_else(|| {
            format!(
                "{name} is not a function this crate knows by name — an unknown function is \
                 written as a name and an arity, which the app resolves or does not, and \
                 guessing which is not this crate's to do"
            )
        })?;
        let mut arguments = 0u32;
        if !self.eat(")") {
            loop {
                self.expression(0)?;
                arguments += 1;
                if self.eat(",") {
                    continue;
                }
                if self.eat(")") {
                    break;
                }
                return Err(format!("{name}(: an argument is missing its ',' or ')'"));
            }
        }
        let mut message = just(node::FUNCTION);
        message.set_in_order(2, Value::Varint(u64::from(index)));
        message.set_in_order(3, Value::Varint(u64::from(arguments)));
        self.push(message);
        Ok(())
    }
}

/// `&'static str` for the punctuation table, which only ever holds literals.
fn leak(text: &str) -> &'static str {
    PUNCT
        .iter()
        .chain(["(", ")", ":", ","].iter())
        .find(|symbol| **symbol == text)
        .copied()
        .unwrap_or("")
}

// -- nodes -------------------------------------------------------------------

fn just(kind: u32) -> Message {
    let mut message = Message::default();
    message.set_in_order(1, Value::Varint(u64::from(kind)));
    message
}

/// A number literal, written the way the app writes one: the lossy double at
/// field 4 *and* the decimal128 halves at 42/43, which are authoritative.
fn number(value: crate::table::Decimal) -> Result<Message, String> {
    let bytes = crate::table::encode_decimal128(value)?;
    let low = u64::from_le_bytes(bytes[..8].try_into().expect("eight bytes"));
    let high = u64::from_le_bytes(bytes[8..].try_into().expect("eight bytes"));
    let mut message = just(node::NUMBER);
    message.set_in_order(4, Value::Fixed64(value.to_f64().to_le_bytes()));
    message.set_in_order(42, Value::Varint(low));
    message.set_in_order(43, Value::Varint(high));
    Ok(message)
}

/// One end of a reference as it was written: `$B$2`, `B2`, `B` or `2`.
struct Coordinate {
    column: Option<(i64, bool)>,
    row: Option<(i64, bool)>,
}

impl Coordinate {
    fn parse(word: &str) -> Option<Coordinate> {
        let (column_absolute, rest) = match word.strip_prefix('$') {
            Some(rest) => (true, rest),
            None => (false, word),
        };
        let letters: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphabetic())
            .collect();
        let rest = &rest[letters.len()..];
        let (row_absolute, digits) = match rest.strip_prefix('$') {
            Some(rest) => (true, rest),
            None => (false, rest),
        };
        if !digits.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        if letters.is_empty() && digits.is_empty() {
            return None;
        }
        // Numbers' widest table is three letters' worth of columns, so a longer
        // run of letters is a *name* — a header, or something defined — and not
        // a column at all. The difference only shows in which refusal the
        // caller is told about, and being told the right one matters.
        if letters.len() > 3 {
            return None;
        }
        // `$` on an axis that is not there is not a reference.
        if letters.is_empty() && column_absolute {
            return None;
        }
        let column = (!letters.is_empty()).then(|| {
            let mut index: i64 = 0;
            for letter in letters.bytes() {
                index = index * 26 + i64::from((letter | 0x20) - b'a') + 1;
            }
            (index - 1, column_absolute)
        });
        let row = match digits.is_empty() {
            true => None,
            false => Some((digits.parse::<i64>().ok()?.checked_sub(1)?, row_absolute)),
        };
        if row.is_some_and(|(index, _)| index < 0) {
            return None;
        }
        Some(Coordinate { column, row })
    }

    /// A single cell — `CELL_REFERENCE_NODE`, one submessage per axis.
    fn cell(&self, host: (i64, i64)) -> Result<Message, String> {
        let (Some(column), Some(row)) = (self.column, self.row) else {
            return Err(
                "a whole row or column is refused: its dependency edge is not one this crate \
                 writes, and the app would recalculate the formula without it"
                    .into(),
            );
        };
        let mut message = just(node::CELL_REFERENCE);
        message.set_in_order(26, Value::Bytes(axis(column, host.0).encode()));
        message.set_in_order(27, Value::Bytes(axis(row, host.1).encode()));
        Ok(message)
    }

    /// A range — `COLON_TRACT_NODE`, with the sticky bits and up to four lists.
    fn range(&self, end: &Coordinate, host: (i64, i64)) -> Result<Message, String> {
        let (Some(begin_column), Some(begin_row), Some(end_column), Some(end_row)) =
            (self.column, self.row, end.column, end.row)
        else {
            return Err(
                "a whole row or column is refused: its dependency edge is not one this crate \
                 writes, and the app would recalculate the formula without it"
                    .into(),
            );
        };

        let mut sticky = Message::default();
        for (number, (_, absolute)) in [
            (1, begin_row),
            (2, begin_column),
            (3, end_row),
            (4, end_column),
        ] {
            sticky.set_in_order(number, Value::Varint(u64::from(absolute)));
        }

        let mut tract = Message::default();
        // The tract's lists are plain `int32`, not zigzag — the one decoding
        // difference between them and the single-cell coordinates above.
        for (relative_field, absolute_field, begin, end, host) in [
            (1, 3, begin_column, end_column, host.0),
            (2, 4, begin_row, end_row, host.1),
        ] {
            let stored = |(index, absolute): (i64, bool)| {
                if absolute {
                    index
                } else {
                    index - host
                }
            };
            for (field, wanted) in [(relative_field, false), (absolute_field, true)] {
                let ends: Vec<i64> = [begin, end]
                    .iter()
                    .filter(|(_, absolute)| *absolute == wanted)
                    .map(|end| stored(*end))
                    .collect();
                let (Some(&first), last) = (ends.first(), ends.last().copied()) else {
                    continue;
                };
                let mut list = Message::default();
                list.set_in_order(1, Value::Varint(first as u64));
                // An omitted `range_end` means it equals `range_begin`, which
                // is how the app writes a range whose ends agree.
                if last.is_some_and(|last| last != first) {
                    list.set_in_order(2, Value::Varint(last.expect("checked") as u64));
                }
                tract.set_in_order(field, Value::Bytes(list.encode()));
            }
        }
        // Field 5 is `1` on every colon tract in the corpus, written here for
        // the same reason the pre-BNC row fields are: the app writes it, and
        // nothing this crate knows says what a tract without it means.
        tract.set_in_order(5, Value::Varint(1));

        let mut message = just(node::COLON_TRACT);
        message.set_in_order(33, Value::Bytes(sticky.encode()));
        message.set_in_order(40, Value::Bytes(tract.encode()));
        Ok(message)
    }
}

/// One axis of a `CELL_REFERENCE_NODE`: `{1: zigzag index, 2: absolute}`.
///
/// An absolute axis stores the index itself; a relative one stores a signed
/// offset from the host cell. Numbers writes field 2 explicitly rather than
/// relying on the varint default, and so does this.
fn axis((index, absolute): (i64, bool), host: i64) -> Message {
    let stored = if absolute { index } else { index - host };
    let mut message = Message::default();
    message.set_in_order(1, Value::Varint(zigzag(stored)));
    message.set_in_order(2, Value::Varint(u64::from(absolute)));
    message
}

fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shapes the fixture carries, reproduced field for field. Each of
    /// these is a formula `numbers-formulas.numbers` holds, dumped from the
    /// document the app wrote — so this is a comparison against Numbers, not
    /// against this parser's own idea of the format.
    #[test]
    fn nodes_match_what_the_app_writes() {
        // `=B2+1` in C2 (host column 2, row 1): a relative reference one column
        // left and no rows away, then the literal, then the operator.
        let ast = parse("=B2+1", (2, 1)).unwrap();
        assert_eq!(ast.nodes.len(), 3);
        assert_eq!(ast.nodes[0].kind, node::CELL_REFERENCE);
        assert_eq!(
            ast.nodes[0].message().bytes(26).unwrap(),
            &[8, 1, 16, 0],
            "column: zigzag(-1) = 1, relative"
        );
        assert_eq!(ast.nodes[0].message().bytes(27).unwrap(), &[8, 0, 16, 0]);
        assert_eq!(ast.nodes[1].kind, node::NUMBER);
        assert_eq!(ast.nodes[1].message().varint(42), Some(1));
        assert_eq!(
            ast.nodes[1].message().varint(43),
            Some(3476778912330022912),
            "the decimal128 high half of an integer 1"
        );
        assert_eq!(ast.nodes[2].kind, node::ADDITION);

        // `=$B$2` — the index itself, and the flag set.
        let ast = parse("=$B$2", (2, 1)).unwrap();
        assert_eq!(ast.nodes[0].message().bytes(26).unwrap(), &[8, 2, 16, 1]);
        assert_eq!(ast.nodes[0].message().bytes(27).unwrap(), &[8, 2, 16, 1]);

        // A parenthesis is a one-argument list node.
        let ast = parse("=(B2+1)*2", (2, 1)).unwrap();
        let kinds: Vec<u32> = ast.nodes.iter().map(|n| n.kind).collect();
        assert_eq!(
            kinds,
            vec![
                node::CELL_REFERENCE,
                node::NUMBER,
                node::ADDITION,
                node::LIST,
                node::NUMBER,
                node::MULTIPLICATION
            ]
        );
        assert_eq!(ast.nodes[3].message().varint(13), Some(1));

        // A function over a range: the tract's absolute lists and the `5: 1`.
        let ast = parse("=SUM($B$2:$B$4)", (2, 1)).unwrap();
        assert_eq!(ast.nodes[0].kind, node::COLON_TRACT);
        assert_eq!(
            ast.nodes[0].message().bytes(33).unwrap(),
            &[8, 1, 16, 1, 24, 1, 32, 1],
            "all four sticky bits set"
        );
        assert_eq!(
            ast.nodes[0].message().bytes(40).unwrap(),
            &[26, 2, 8, 1, 34, 4, 8, 1, 16, 3, 40, 1],
            "absolute_column {{1}}, absolute_row {{1, 3}}, and field 5"
        );
        assert_eq!(ast.nodes[1].kind, node::FUNCTION);
        assert_eq!(ast.nodes[1].message().varint(2), Some(168), "SUM");
        assert_eq!(ast.nodes[1].message().varint(3), Some(1));
    }

    #[test]
    fn precedence_and_associativity() {
        let kinds = |text: &str| -> Vec<u32> {
            parse(text, (0, 0))
                .unwrap()
                .nodes
                .iter()
                .map(|n| n.kind)
                .collect()
        };
        // 2+3*4 multiplies first.
        assert_eq!(
            kinds("=2+3*4"),
            vec![
                node::NUMBER,
                node::NUMBER,
                node::NUMBER,
                node::MULTIPLICATION,
                node::ADDITION
            ]
        );
        // 8-3-2 is (8-3)-2: left-associative.
        assert_eq!(
            kinds("=8-3-2"),
            vec![
                node::NUMBER,
                node::NUMBER,
                node::SUBTRACTION,
                node::NUMBER,
                node::SUBTRACTION
            ]
        );
        // 2^3^2 is 2^(3^2): right-associative.
        assert_eq!(
            kinds("=2^3^2"),
            vec![
                node::NUMBER,
                node::NUMBER,
                node::NUMBER,
                node::POWER,
                node::POWER
            ]
        );
        // Unary minus, and percent as a postfix.
        assert_eq!(kinds("=-5"), vec![node::NUMBER, node::NEGATION]);
        assert_eq!(kinds("=50%"), vec![node::NUMBER, node::PERCENT]);
    }

    #[test]
    fn strings_keep_their_quotes() {
        let ast = parse(r#"="Er sagte ""hallo""""#, (0, 0)).unwrap();
        assert_eq!(
            ast.nodes[0].string().as_deref(),
            Some(r#"Er sagte "hallo""#)
        );
    }

    /// Everything refused by name, because a formula this crate cannot register
    /// in the calculation engine is one the app would recalculate wrongly.
    #[test]
    fn what_is_refused() {
        for (text, expected) in [
            ("=Tabelle.B2", "another table"),
            ("=SUM(B)", "whole row or column"),
            ("=SUM(B2:B)", "whole row or column"),
            ("=NICHTDA(1)", "not a function"),
            ("=Umsatz", "neither a cell reference nor a function"),
            ("=TRUE", "neither a cell reference nor a function"),
            ("=1+", "ends where a value was expected"),
            ("=(1+2", "missing its ')'"),
            ("=SUM(1", "missing its ',' or ')'"),
            ("=1 2", "unexpected number 2"),
            (r#"="abc"#, "missing its closing quote"),
            ("=1#2", "not something a formula can hold"),
        ] {
            let error = parse(text, (0, 0)).expect_err(&format!("{text} was not refused"));
            assert!(
                error.contains(expected),
                "{text}: {error:?} does not mention {expected:?}"
            );
        }
    }
}
