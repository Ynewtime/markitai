//! Binary property lists, version 0 — the format the three `Metadata/*.plist`
//! entries of an iWork package are written in.
//!
//! A package is not all protobuf. Beside `Index/*.iwa` sit
//! `Metadata/Properties.plist`, `Metadata/BuildVersionHistory.plist` and
//! `Metadata/DocumentIdentifier`, and the first two are `bplist00` — Apple's
//! binary plist, which macOS reads before it opens a single object stream.
//! Nothing else in this crate needs a plist, so rather than take a dependency
//! this module reads and writes the one version those files use.
//!
//! ## The layout
//!
//! ```text
//!   "bplist00"                       8 bytes
//!   objects, back to back            each one a marker byte and its payload
//!   offset table                     one big-endian integer per object
//!   trailer                          32 bytes, the last of them in the file
//!     …5 unused, sort version, offset int size, object ref size,
//!       object count, root object, offset table position — all big-endian
//! ```
//!
//! An object's marker byte is a type in the high nibble and, for the sized
//! types, a length in the low one; a low nibble of `0xF` means the length is
//! the *next* object-marker-encoded integer instead. Containers hold indices
//! into the object table, not objects, which is how a plist shares one string
//! between several keys — and iWork's Properties.plist does exactly that, so a
//! writer that assumes one object per value is writing a different file from
//! the one it read.
//!
//! ## What is deliberately not here
//!
//! Sets (`0xC`), UIDs (`0x8`), 16-byte integers and version 1 plists. None
//! occurs in an iWork package, and a decoder that guessed at them would be
//! claiming to understand something it has never seen — [`Error::Format`] says
//! so instead.

use std::cell::Cell;
use std::collections::BTreeMap;

use crate::Error;

/// A property-list value.
///
/// A dictionary keeps its keys in the order the file had them, so a read and a
/// write reproduce a file's own ordering rather than an alphabetical one.
#[derive(Debug, Clone, PartialEq)]
pub enum Plist {
    Bool(bool),
    Integer(i64),
    Real(f64),
    /// Seconds from 2001-01-01, as `CFDate` counts them.
    Date(f64),
    String(String),
    Data(Vec<u8>),
    Array(Vec<Plist>),
    Dictionary(Vec<(String, Plist)>),
}

impl Plist {
    pub fn get(&self, key: &str) -> Option<&Plist> {
        match self {
            Plist::Dictionary(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Plist::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Plist::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Plist]> {
        match self {
            Plist::Array(items) => Some(items),
            _ => None,
        }
    }

    /// Replace the value at `key`, or add it at the end if it is not there.
    pub fn set(&mut self, key: &str, value: Plist) {
        if let Plist::Dictionary(entries) = self {
            match entries.iter_mut().find(|(k, _)| k == key) {
                Some(entry) => entry.1 = value,
                None => entries.push((key.to_string(), value)),
            }
        }
    }

    /// Every key of a dictionary, in file order.
    pub fn keys(&self) -> Vec<&str> {
        match self {
            Plist::Dictionary(entries) => entries.iter().map(|(k, _)| k.as_str()).collect(),
            _ => Vec::new(),
        }
    }
}

// -- reading -----------------------------------------------------------------

const HEADER: &[u8] = b"bplist00";
const TRAILER: usize = 32;

struct Reader<'a> {
    bytes: &'a [u8],
    offsets: Vec<usize>,
    ref_size: usize,
    /// How many more values may be materialised before the read is refused. The
    /// object table is a DAG, not a tree — a container holds *indices*, so a
    /// leaf shared by several keys is legitimate — but that same sharing lets a
    /// ~140-byte file describe an exponential fan-out (four references per node,
    /// a dozen levels deep) that walks to hundreds of megabytes of `Plist`. The
    /// depth limit bounds the path, not the count; this bounds the count. A
    /// budget rather than a visited-set, because visiting a shared leaf twice is
    /// what a real plist does and must stay allowed.
    budget: Cell<usize>,
}

/// The most `Plist` values a single binary plist may expand to. Real
/// `Properties.plist` files carry a couple of dozen; this is high enough that
/// no genuine iWork metadata approaches it and low enough that the fan-out bomb
/// is refused in well under a second.
const MAX_NODES: usize = 1 << 20;

/// Read a property list in either of the two forms an iWork package uses.
///
/// **The package has both, and which is which is not a detail.**
/// `Metadata/Properties.plist` is `bplist00`; `Metadata/BuildVersionHistory.plist`
/// is **XML**, with the Apple DTD and all. A reader that assumes one form and
/// meets the other reports a corrupt document.
pub fn parse(bytes: &[u8]) -> Result<Plist, Error> {
    if !bytes.starts_with(HEADER) {
        if bytes.starts_with(b"<?xml") || bytes.starts_with(b"<plist") {
            return xml::parse(bytes);
        }
        return Err(Error::Format(
            "not a property list: neither a bplist00 header nor XML".into(),
        ));
    }
    parse_binary(bytes)
}

fn parse_binary(bytes: &[u8]) -> Result<Plist, Error> {
    if !bytes.starts_with(HEADER) {
        return Err(Error::Format(
            "not a binary property list: no bplist00 header".into(),
        ));
    }
    if bytes.len() < HEADER.len() + TRAILER {
        return Err(Error::Format("binary property list is truncated".into()));
    }
    let trailer = &bytes[bytes.len() - TRAILER..];
    let offset_size = trailer[6] as usize;
    let ref_size = trailer[7] as usize;
    let count = be(&trailer[8..16]) as usize;
    let root = be(&trailer[16..24]) as usize;
    let table = be(&trailer[24..32]) as usize;

    if offset_size == 0 || offset_size > 8 || ref_size == 0 || ref_size > 8 {
        return Err(Error::Format(format!(
            "binary property list: offset size {offset_size}, reference size {ref_size}"
        )));
    }
    let end = table.saturating_add(count.saturating_mul(offset_size));
    if end > bytes.len() {
        return Err(Error::Format(
            "binary property list: offset table runs past the end".into(),
        ));
    }
    let offsets = (0..count)
        .map(|i| be(&bytes[table + i * offset_size..table + (i + 1) * offset_size]) as usize)
        .collect();

    let reader = Reader {
        bytes,
        offsets,
        ref_size,
        budget: Cell::new(MAX_NODES),
    };
    reader.object(root, 0)
}

impl Reader<'_> {
    fn object(&self, index: usize, depth: usize) -> Result<Plist, Error> {
        if depth > 32 {
            return Err(Error::Format(
                "binary property list nests too deeply".into(),
            ));
        }
        // Spend one from the total-nodes budget. A DAG can name far more values
        // than it has objects, so this — not the depth limit — is what stops a
        // fan-out bomb.
        match self.budget.get() {
            0 => {
                return Err(Error::Format(
                    "binary property list expands to too many values".into(),
                ))
            }
            remaining => self.budget.set(remaining - 1),
        }
        let start = *self
            .offsets
            .get(index)
            .ok_or_else(|| Error::Format(format!("binary property list: no object {index}")))?;
        let marker = *self
            .bytes
            .get(start)
            .ok_or_else(|| Error::Format("binary property list: object past the end".into()))?;
        let (kind, low) = (marker >> 4, (marker & 0x0f) as usize);
        let body = start + 1;

        match kind {
            0x0 => match low {
                0x08 => Ok(Plist::Bool(false)),
                0x09 => Ok(Plist::Bool(true)),
                other => Err(Error::Format(format!(
                    "binary property list: singleton marker 0x0{other:x}"
                ))),
            },
            0x1 => {
                let width = 1usize << low;
                // The module doc promises 16-byte integers are refused, and the
                // accumulator behind `be` is a `u64`: a width past eight would
                // silently truncate a value it cannot hold — `0x14` and the
                // 32-, 64- … -byte widths above it all read wrong rather than
                // erroring. Refuse them, as promised.
                if width > 8 {
                    return Err(Error::Format(format!(
                        "binary property list: {width}-byte integer is wider than this crate reads"
                    )));
                }
                let raw = self.slice(body, width)?;
                // Integers are signed only at eight bytes wide; narrower ones
                // are unsigned, which is why this is not a single sign-extend.
                Ok(Plist::Integer(if width == 8 {
                    i64::from_be_bytes(raw.try_into().unwrap())
                } else {
                    be(raw) as i64
                }))
            }
            0x2 => {
                let width = 1usize << low;
                let raw = self.slice(body, width)?;
                match width {
                    4 => Ok(Plist::Real(
                        f32::from_be_bytes(raw.try_into().unwrap()) as f64
                    )),
                    8 => Ok(Plist::Real(f64::from_be_bytes(raw.try_into().unwrap()))),
                    _ => Err(Error::Format(format!(
                        "binary property list: {width}-byte real"
                    ))),
                }
            }
            0x3 => {
                let raw = self.slice(body, 8)?;
                Ok(Plist::Date(f64::from_be_bytes(raw.try_into().unwrap())))
            }
            0x4 => {
                let (length, at) = self.length(low, body)?;
                Ok(Plist::Data(self.slice(at, length)?.to_vec()))
            }
            0x5 => {
                let (length, at) = self.length(low, body)?;
                let raw = self.slice(at, length)?;
                Ok(Plist::String(
                    std::str::from_utf8(raw)
                        .map_err(|_| {
                            Error::Format("binary property list: ASCII string is not UTF-8".into())
                        })?
                        .to_string(),
                ))
            }
            0x6 => {
                let (length, at) = self.length(low, body)?;
                let raw = self.slice(at, double(length)?)?;
                let units: Vec<u16> = raw
                    .chunks_exact(2)
                    .map(|c| u16::from_be_bytes([c[0], c[1]]))
                    .collect();
                String::from_utf16(&units).map(Plist::String).map_err(|_| {
                    Error::Format("binary property list: UTF-16 string is not valid".into())
                })
            }
            0xa => {
                let (length, at) = self.length(low, body)?;
                let refs = self.references(at, length)?;
                let mut items = Vec::with_capacity(refs.len());
                for reference in refs {
                    items.push(self.object(reference, depth + 1)?);
                }
                Ok(Plist::Array(items))
            }
            0xd => {
                let (length, at) = self.length(low, body)?;
                let refs = self.references(at, double(length)?)?;
                let mut entries = Vec::with_capacity(length);
                for i in 0..length {
                    let key = match self.object(refs[i], depth + 1)? {
                        Plist::String(key) => key,
                        other => {
                            return Err(Error::Format(format!(
                                "binary property list: dictionary key is {other:?}"
                            )))
                        }
                    };
                    entries.push((key, self.object(refs[length + i], depth + 1)?));
                }
                Ok(Plist::Dictionary(entries))
            }
            other => Err(Error::Format(format!(
                "binary property list: marker 0x{other:x} is a type this crate has not seen"
            ))),
        }
    }

    /// The count in a marker's low nibble, or the integer that follows when it
    /// is `0xF`. Returns the count and where the payload starts.
    fn length(&self, low: usize, body: usize) -> Result<(usize, usize), Error> {
        if low != 0x0f {
            return Ok((low, body));
        }
        let marker = *self
            .bytes
            .get(body)
            .ok_or_else(|| Error::Format("binary property list: truncated length".into()))?;
        if marker >> 4 != 0x1 {
            return Err(Error::Format(
                "binary property list: extended length is not an integer".into(),
            ));
        }
        let width = 1usize << (marker & 0x0f);
        Ok((be(self.slice(body + 1, width)?) as usize, body + 1 + width))
    }

    fn references(&self, at: usize, count: usize) -> Result<Vec<usize>, Error> {
        let width = count.checked_mul(self.ref_size).ok_or_else(|| {
            Error::Format("binary property list: reference count overflows".into())
        })?;
        let raw = self.slice(at, width)?;
        Ok(raw
            .chunks_exact(self.ref_size)
            .map(|c| be(c) as usize)
            .collect())
    }

    /// Every length in a binary plist is a number read out of the file, and the
    /// biggest of them is eight bytes wide — so `at + length` is an addition of
    /// two attacker-chosen integers and it wraps. Checked, so a length of
    /// `0xFFFF_FFFF_FFFF_FFFF` is "truncated" rather than an arithmetic panic
    /// in a debug build and a slice of the wrong memory in a release one.
    fn slice(&self, at: usize, length: usize) -> Result<&[u8], Error> {
        let end = at
            .checked_add(length)
            .ok_or_else(|| Error::Format("binary property list: length overflows".into()))?;
        self.bytes
            .get(at..end)
            .ok_or_else(|| Error::Format("binary property list is truncated".into()))
    }
}

/// Twice a count read out of the file: a UTF-16 string's bytes, or a
/// dictionary's keys *and* its values. Both wrap for a large enough count.
fn double(count: usize) -> Result<usize, Error> {
    count
        .checked_mul(2)
        .ok_or_else(|| Error::Format("binary property list: length overflows".into()))
}

fn be(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0u64, |acc, b| (acc << 8) | u64::from(*b))
}

/// XML property lists — read only.
///
/// `Metadata/BuildVersionHistory.plist` is one, and nothing here ever needs to
/// write one back: the app leaves that file alone when it saves a copy under a
/// new identity, so this crate does too.
mod xml {
    use super::Plist;
    use crate::Error;

    pub fn parse(bytes: &[u8]) -> Result<Plist, Error> {
        let text = std::str::from_utf8(bytes)
            .map_err(|_| Error::Format("XML property list is not UTF-8".into()))?;
        let mut cursor = Cursor { text, at: 0 };
        cursor.skip_prologue();
        let value = cursor.value(0)?;
        Ok(value)
    }

    struct Cursor<'a> {
        text: &'a str,
        at: usize,
    }

    impl<'a> Cursor<'a> {
        /// Everything before the document element: the declaration, the
        /// doctype, and the `<plist version="1.0">` wrapper itself.
        fn skip_prologue(&mut self) {
            if let Some(open) = self.text.find("<plist") {
                if let Some(close) = self.text[open..].find('>') {
                    self.at = open + close + 1;
                }
            }
        }

        fn value(&mut self, depth: usize) -> Result<Plist, Error> {
            if depth > 32 {
                return Err(Error::Format("XML property list nests too deeply".into()));
            }
            let tag = self.open_tag()?;
            match tag.as_str() {
                "true/" => Ok(Plist::Bool(true)),
                "false/" => Ok(Plist::Bool(false)),
                "string" => Ok(Plist::String(unescape(&self.until_close("string")?))),
                "key" => Ok(Plist::String(unescape(&self.until_close("key")?))),
                "integer" => self
                    .until_close("integer")?
                    .trim()
                    .parse()
                    .map(Plist::Integer)
                    .map_err(|_| Error::Format("XML property list: bad <integer>".into())),
                "real" => self
                    .until_close("real")?
                    .trim()
                    .parse()
                    .map(Plist::Real)
                    .map_err(|_| Error::Format("XML property list: bad <real>".into())),
                "data" => Ok(Plist::Data(base64_decode(&self.until_close("data")?)?)),
                "array" => {
                    let mut items = Vec::new();
                    while !self.at_close("array") {
                        items.push(self.value(depth + 1)?);
                    }
                    self.consume_close("array")?;
                    Ok(Plist::Array(items))
                }
                "dict" => {
                    let mut entries = Vec::new();
                    while !self.at_close("dict") {
                        let Plist::String(key) = self.value(depth + 1)? else {
                            return Err(Error::Format(
                                "XML property list: dictionary key is not a <key>".into(),
                            ));
                        };
                        entries.push((key, self.value(depth + 1)?));
                    }
                    self.consume_close("dict")?;
                    Ok(Plist::Dictionary(entries))
                }
                "array/" => Ok(Plist::Array(Vec::new())),
                "dict/" => Ok(Plist::Dictionary(Vec::new())),
                "string/" | "key/" => Ok(Plist::String(String::new())),
                other => Err(Error::Format(format!(
                    "XML property list: <{other}> is a type this crate has not seen"
                ))),
            }
        }

        /// Read the next start tag, returning its name — with a trailing `/`
        /// when it closed itself, which is how `<true/>` and `<array/>` arrive.
        fn open_tag(&mut self) -> Result<String, Error> {
            self.skip_space();
            let rest = &self.text[self.at..];
            if !rest.starts_with('<') {
                return Err(Error::Format("XML property list: expected a tag".into()));
            }
            let close = rest
                .find('>')
                .ok_or_else(|| Error::Format("XML property list: unterminated tag".into()))?;
            let inner = &rest[1..close];
            self.at += close + 1;
            let name = inner.split_whitespace().next().unwrap_or("");
            Ok(if inner.ends_with('/') {
                format!("{}/", name.trim_end_matches('/'))
            } else {
                name.to_string()
            })
        }

        fn until_close(&mut self, name: &str) -> Result<String, Error> {
            let end = format!("</{name}>");
            let rest = &self.text[self.at..];
            let at = rest
                .find(&end)
                .ok_or_else(|| Error::Format(format!("XML property list: no {end}")))?;
            let body = rest[..at].to_string();
            self.at += at + end.len();
            Ok(body)
        }

        fn at_close(&mut self, name: &str) -> bool {
            self.skip_space();
            self.text[self.at..].starts_with(&format!("</{name}>"))
        }

        fn consume_close(&mut self, name: &str) -> Result<(), Error> {
            self.skip_space();
            let end = format!("</{name}>");
            if !self.text[self.at..].starts_with(&end) {
                return Err(Error::Format(format!("XML property list: no {end}")));
            }
            self.at += end.len();
            Ok(())
        }

        fn skip_space(&mut self) {
            while let Some(c) = self.text[self.at..].chars().next() {
                if c.is_whitespace() {
                    self.at += c.len_utf8();
                } else {
                    break;
                }
            }
        }
    }

    /// Decode the base64 body of an XML `<data>` element to the bytes it stands
    /// for. CoreFoundation writes the payload line-wrapped and indented, so
    /// whitespace is skipped; padding `=` carries no bits and is skipped too.
    ///
    /// A tiny hand-rolled decoder rather than a dependency — the crate takes
    /// none — and it refuses a character outside the alphabet rather than
    /// returning the base64 *text* as though it were the bytes, which is what
    /// this branch used to do.
    fn base64_decode(text: &str) -> Result<Vec<u8>, Error> {
        fn sextet(c: u8) -> Option<u32> {
            match c {
                b'A'..=b'Z' => Some(u32::from(c - b'A')),
                b'a'..=b'z' => Some(u32::from(c - b'a') + 26),
                b'0'..=b'9' => Some(u32::from(c - b'0') + 52),
                b'+' => Some(62),
                b'/' => Some(63),
                _ => None,
            }
        }
        let mut out = Vec::new();
        let mut acc = 0u32;
        let mut bits = 0u32;
        for &c in text.as_bytes() {
            if c == b'=' || c.is_ascii_whitespace() {
                continue;
            }
            let value = sextet(c)
                .ok_or_else(|| Error::Format("XML property list: <data> is not base64".into()))?;
            acc = (acc << 6) | value;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push((acc >> bits) as u8);
            }
        }
        Ok(out)
    }

    fn unescape(text: &str) -> String {
        if !text.contains('&') {
            return text.to_string();
        }
        text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            // Last, or an escaped ampersand would be unescaped twice.
            .replace("&amp;", "&")
    }
}

// -- writing -----------------------------------------------------------------

/// Serialise a value as a `bplist00`.
///
/// **Not byte-for-byte what CoreFoundation writes, and deliberately so.** A
/// `Properties.plist` written by Pages carries 21 objects for 20 values: the
/// three `false`s are *written* three times and *referenced* once, leaving two
/// objects in the file that nothing points at, while the three equal UUID
/// strings are written out in full three times over. This writer shares equal
/// strings and leaves no orphans, which makes the same ten keys 443 bytes
/// instead of 525. Sharing is invisible to a reader — a plist is a value tree,
/// and `plutil -p` prints the two files identically — and every entry this
/// crate does not touch keeps its original bytes, so the difference only ever
/// appears in a file whose identity has just been changed on purpose.
pub fn write(value: &Plist) -> Vec<u8> {
    let mut objects: Vec<Node> = Vec::new();
    let mut interned: BTreeMap<String, usize> = BTreeMap::new();
    let root = flatten(value, &mut objects, &mut interned);

    let ref_size = width_for(objects.len().saturating_sub(1) as u64);
    let mut body = Vec::new();
    let mut offsets = Vec::with_capacity(objects.len());
    for object in &objects {
        offsets.push(HEADER.len() + body.len());
        encode(object, &mut body, ref_size);
    }

    let table_at = HEADER.len() + body.len();
    let offset_size = width_for(table_at as u64);
    let mut out = Vec::with_capacity(table_at + objects.len() * offset_size + TRAILER);
    out.extend_from_slice(HEADER);
    out.extend_from_slice(&body);
    for offset in &offsets {
        push_be(&mut out, *offset as u64, offset_size);
    }
    out.extend_from_slice(&[0; 5]);
    out.push(0); // sort version
    out.push(offset_size as u8);
    out.push(ref_size as u8);
    push_be(&mut out, objects.len() as u64, 8);
    push_be(&mut out, root as u64, 8);
    push_be(&mut out, table_at as u64, 8);
    out
}

/// One object of the flattened table: a leaf, or a container holding the
/// *indices* of its children rather than the children themselves.
enum Node {
    Bool(bool),
    Integer(i64),
    Real(f64),
    Date(f64),
    String(String),
    Data(Vec<u8>),
    Array(Vec<usize>),
    Dictionary(Vec<(usize, usize)>),
}

/// Walk the tree, giving every value an object index. Strings — which includes
/// dictionary keys — are shared.
fn flatten(
    value: &Plist,
    objects: &mut Vec<Node>,
    interned: &mut BTreeMap<String, usize>,
) -> usize {
    match value {
        Plist::Bool(b) => push(objects, Node::Bool(*b)),
        Plist::Integer(n) => push(objects, Node::Integer(*n)),
        Plist::Real(v) => push(objects, Node::Real(*v)),
        Plist::Date(v) => push(objects, Node::Date(*v)),
        Plist::Data(bytes) => push(objects, Node::Data(bytes.clone())),
        Plist::String(text) => match interned.get(text) {
            Some(index) => *index,
            None => {
                let index = push(objects, Node::String(text.clone()));
                interned.insert(text.clone(), index);
                index
            }
        },
        Plist::Array(items) => {
            let index = push(objects, Node::Array(Vec::new()));
            let children = items
                .iter()
                .map(|item| flatten(item, objects, interned))
                .collect();
            objects[index] = Node::Array(children);
            index
        }
        Plist::Dictionary(entries) => {
            let index = push(objects, Node::Dictionary(Vec::new()));
            // Keys before values, so a dictionary's own strings come first —
            // the order CFPropertyList writes, and the one that keeps a
            // re-serialised file the same shape as the original.
            let keys: Vec<usize> = entries
                .iter()
                .map(|(key, _)| flatten(&Plist::String(key.clone()), objects, interned))
                .collect();
            let values: Vec<usize> = entries
                .iter()
                .map(|(_, value)| flatten(value, objects, interned))
                .collect();
            objects[index] = Node::Dictionary(keys.into_iter().zip(values).collect());
            index
        }
    }
}

fn push(objects: &mut Vec<Node>, node: Node) -> usize {
    objects.push(node);
    objects.len() - 1
}

fn encode(value: &Node, out: &mut Vec<u8>, ref_size: usize) {
    match value {
        Node::Bool(false) => out.push(0x08),
        Node::Bool(true) => out.push(0x09),
        Node::Integer(n) => {
            // Anything that fits unsigned goes narrow; a negative goes wide,
            // because only the eight-byte form is signed.
            let (width, bits) = match *n {
                n if n < 0 => (8usize, n as u64),
                n if n <= 0xff => (1, n as u64),
                n if n <= 0xffff => (2, n as u64),
                n if n <= 0xffff_ffff => (4, n as u64),
                n => (8, n as u64),
            };
            out.push(0x10 | width.trailing_zeros() as u8);
            push_be(out, bits, width);
        }
        Node::Real(v) => {
            out.push(0x23);
            out.extend_from_slice(&v.to_be_bytes());
        }
        Node::Date(v) => {
            out.push(0x33);
            out.extend_from_slice(&v.to_be_bytes());
        }
        Node::Data(bytes) => {
            marker(out, 0x40, bytes.len());
            out.extend_from_slice(bytes);
        }
        Node::String(text) => {
            if text.is_ascii() {
                marker(out, 0x50, text.len());
                out.extend_from_slice(text.as_bytes());
            } else {
                let units: Vec<u16> = text.encode_utf16().collect();
                marker(out, 0x60, units.len());
                for unit in units {
                    out.extend_from_slice(&unit.to_be_bytes());
                }
            }
        }
        Node::Array(items) => {
            marker(out, 0xa0, items.len());
            for item in items {
                push_be(out, *item as u64, ref_size);
            }
        }
        Node::Dictionary(entries) => {
            marker(out, 0xd0, entries.len());
            for (key, _) in entries {
                push_be(out, *key as u64, ref_size);
            }
            for (_, value) in entries {
                push_be(out, *value as u64, ref_size);
            }
        }
    }
}

fn marker(out: &mut Vec<u8>, kind: u8, length: usize) {
    if length < 0x0f {
        out.push(kind | length as u8);
        return;
    }
    out.push(kind | 0x0f);
    let width = width_for(length as u64);
    out.push(0x10 | width.trailing_zeros() as u8);
    push_be(out, length as u64, width);
}

fn width_for(value: u64) -> usize {
    match value {
        v if v <= 0xff => 1,
        v if v <= 0xffff => 2,
        v if v <= 0xffff_ffff => 4,
        _ => 8,
    }
}

fn push_be(out: &mut Vec<u8>, value: u64, width: usize) {
    out.extend_from_slice(&value.to_be_bytes()[8 - width..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(value: &Plist) -> Plist {
        parse(&write(value)).expect("the writer produced something the reader refuses")
    }

    #[test]
    fn a_dictionary_of_strings_and_booleans_survives() {
        let value = Plist::Dictionary(vec![
            ("revision".into(), Plist::String("0::ABC".into())),
            ("isMultiPage".into(), Plist::Bool(true)),
            ("hasRemote".into(), Plist::Bool(false)),
            ("count".into(), Plist::Integer(70000)),
        ]);
        assert_eq!(roundtrip(&value), value);
    }

    /// `Properties.plist` gives the same UUID to `documentUUID`, `shareUUID`
    /// and `stableDocumentUUID` whenever a document has never been copied, so
    /// three of its ten keys carry one value. Whether that value is stored once
    /// or three times is the writer's business — Apple stores it three times,
    /// this stores it once — and neither is visible to a reader.
    #[test]
    fn equal_strings_are_written_once_and_referenced_three_times() {
        let uuid = "1291ACAB-6D20-4208-A032-2C990FC9109B";
        let value = Plist::Dictionary(vec![
            ("documentUUID".into(), Plist::String(uuid.into())),
            ("shareUUID".into(), Plist::String(uuid.into())),
            ("stableDocumentUUID".into(), Plist::String(uuid.into())),
        ]);
        let written = write(&value);
        assert_eq!(roundtrip(&value), value);
        assert_eq!(
            written
                .windows(uuid.len())
                .filter(|w| *w == uuid.as_bytes())
                .count(),
            1,
            "the UUID should be stored once"
        );
    }

    #[test]
    fn an_array_of_strings_survives() {
        let value = Plist::Array(vec![
            Plist::String("Template: 21_BasicWhite (dev/15.3)".into()),
            Plist::String("M15.3.1-7050.1.1-2".into()),
        ]);
        assert_eq!(roundtrip(&value), value);
    }

    /// A key or a value longer than fourteen characters needs the extended
    /// length form, and iWork's own keys — `hasExternalReferenceOrMissingData`
    /// is thirty-three — are all past it.
    #[test]
    fn a_long_string_uses_the_extended_length_marker() {
        let value = Plist::Dictionary(vec![(
            "hasExternalReferenceOrMissingData".into(),
            Plist::Bool(false),
        )]);
        assert_eq!(roundtrip(&value), value);
    }

    #[test]
    fn non_ascii_goes_out_as_utf16() {
        let value = Plist::String("Schülerblatt — 図".into());
        assert_eq!(roundtrip(&value), value);
        assert!(write(&value).contains(&0x6f), "a UTF-16 marker");
    }

    #[test]
    fn nested_containers_survive() {
        let value = Plist::Dictionary(vec![
            (
                "history".into(),
                Plist::Array(vec![Plist::String("a".into()), Plist::Integer(-1)]),
            ),
            (
                "inner".into(),
                Plist::Dictionary(vec![("k".into(), Plist::Real(1.5))]),
            ),
        ]);
        assert_eq!(roundtrip(&value), value);
    }

    #[test]
    fn a_file_that_is_not_a_plist_is_refused_by_name() {
        assert!(matches!(parse(b"PK\x03\x04"), Err(Error::Format(_))));
        assert!(matches!(parse(b"bplist00"), Err(Error::Format(_))));
    }

    /// The object table is a DAG, and a container holds *indices*: a handful of
    /// arrays, each referencing the next one several times, describes an
    /// exponential number of values in a few dozen bytes. The depth limit does
    /// not catch it — the tree is shallow — so the total-nodes budget must, and
    /// quickly.
    #[test]
    fn a_fan_out_bomb_is_refused_and_bounded() {
        const FAN: usize = 8;
        const LEVELS: usize = 7; // 8^7 = 2^21 values, past MAX_NODES = 2^20

        let mut body = Vec::new();
        let mut offsets = Vec::new();
        for level in 0..LEVELS {
            offsets.push(HEADER.len() + body.len());
            body.push(0xa0 | FAN as u8); // an array of FAN references
            for _ in 0..FAN {
                body.push((level + 1) as u8); // all pointing at the next object
            }
        }
        offsets.push(HEADER.len() + body.len());
        body.push(0x08); // the shared leaf: false

        let mut bytes = HEADER.to_vec();
        bytes.extend_from_slice(&body);
        let table_at = bytes.len();
        for offset in &offsets {
            bytes.push(*offset as u8);
        }
        bytes.extend_from_slice(&[0, 0, 0, 0, 0]);
        bytes.push(0); // sort version
        bytes.push(1); // offset size
        bytes.push(1); // ref size
        bytes.extend_from_slice(&(offsets.len() as u64).to_be_bytes());
        bytes.extend_from_slice(&0u64.to_be_bytes()); // root
        bytes.extend_from_slice(&(table_at as u64).to_be_bytes());
        assert!(
            bytes.len() < 200,
            "the bomb is small: {} bytes",
            bytes.len()
        );

        let start = std::time::Instant::now();
        let error = parse(&bytes).unwrap_err();
        assert!(matches!(error, Error::Format(_)), "{error}");
        assert!(
            start.elapsed().as_secs() < 5,
            "the fan-out was materialised rather than bounded"
        );
    }

    /// A 16-byte integer marker (`0x14`) is refused, as the module doc promises:
    /// the `u64` accumulator behind it would keep only the low eight bytes and
    /// report a value it never held.
    #[test]
    fn a_sixteen_byte_integer_is_refused_not_truncated() {
        let mut bytes = HEADER.to_vec();
        let object_at = bytes.len();
        bytes.push(0x14); // integer, width 1 << 4 = 16
        bytes.extend_from_slice(&[0xff; 16]); // a value far past i64::MAX
        let table_at = bytes.len();
        bytes.push(object_at as u8);
        bytes.extend_from_slice(&[0, 0, 0, 0, 0]);
        bytes.push(0); // sort version
        bytes.push(1); // offset size
        bytes.push(1); // ref size
        bytes.extend_from_slice(&1u64.to_be_bytes()); // count
        bytes.extend_from_slice(&0u64.to_be_bytes()); // root
        bytes.extend_from_slice(&(table_at as u64).to_be_bytes());

        let error = parse(&bytes).unwrap_err();
        assert!(
            matches!(error, Error::Format(_)),
            "a 16-byte integer decoded rather than being refused: {error}"
        );
    }

    /// An XML `<data>` element carries base64, and it is decoded to the bytes it
    /// stands for — not returned as the base64 text, which is what it used to be.
    #[test]
    fn xml_data_is_base64_decoded() {
        let xml = br#"<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><data>
SGVsbG8sIHdvcmxk
</data></plist>"#;
        assert_eq!(parse(xml).unwrap(), Plist::Data(b"Hello, world".to_vec()));
    }

    #[test]
    fn xml_data_that_is_not_base64_is_refused() {
        let xml = br#"<plist version="1.0"><data>not base64 !!</data></plist>"#;
        assert!(matches!(parse(xml), Err(Error::Format(_))));
    }
}
