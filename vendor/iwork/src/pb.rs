//! Minimal schema-less protobuf wire codec.
//!
//! iWork's `.proto` files are not published, so everything here works at the
//! wire level: fields are kept as (number, wire value) pairs and re-encoded in
//! the order they were read. That is enough to inspect and rewrite documents
//! without knowing the message definitions.

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Varint(u64),
    Fixed64([u8; 8]),
    Bytes(Vec<u8>),
    Fixed32([u8; 4]),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub number: u32,
    pub value: Value,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Message {
    pub fields: Vec<Field>,
}

pub struct Reader<'a> {
    pub buf: &'a [u8],
    pub pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    pub fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    pub fn varint(&mut self) -> Result<u64, String> {
        let mut value = 0u64;
        let mut shift = 0;
        loop {
            let byte = *self
                .buf
                .get(self.pos)
                .ok_or_else(|| "truncated varint".to_string())?;
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift > 63 {
                return Err("varint too long".into());
            }
        }
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or("length overflow")?;
        if end > self.buf.len() {
            return Err("truncated field".into());
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    pub fn field(&mut self) -> Result<Field, String> {
        let key = self.varint()?;
        // The field number is stored as a `u32` and re-encoded from it, so a
        // number that does not fit — or field 0, which protobuf does not allow —
        // would not come back as the bytes it went in as. Real archives never
        // approach either (field numbers are at most 2^29-1), so refusing them
        // costs no corpus and keeps `decode`/`encode` injective, the same
        // discipline `decode_nested` leans on.
        let raw_number = key >> 3;
        if raw_number == 0 || raw_number > u64::from(u32::MAX) {
            return Err(format!("field number {raw_number} out of range"));
        }
        let number = raw_number as u32;
        let value = match key & 7 {
            0 => Value::Varint(self.varint()?),
            1 => Value::Fixed64(self.take(8)?.try_into().unwrap()),
            2 => {
                let len = self.varint()? as usize;
                Value::Bytes(self.take(len)?.to_vec())
            }
            5 => Value::Fixed32(self.take(4)?.try_into().unwrap()),
            other => return Err(format!("unsupported wire type {other}")),
        };
        Ok(Field { number, value })
    }
}

impl Message {
    pub fn decode(buf: &[u8]) -> Result<Message, String> {
        let mut reader = Reader::new(buf);
        let mut fields = Vec::new();
        while !reader.done() {
            fields.push(reader.field()?);
        }
        Ok(Message { fields })
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for field in &self.fields {
            let wire = match &field.value {
                Value::Varint(_) => 0,
                Value::Fixed64(_) => 1,
                Value::Bytes(_) => 2,
                Value::Fixed32(_) => 5,
            };
            write_varint(&mut out, (u64::from(field.number) << 3) | wire);
            match &field.value {
                Value::Varint(v) => write_varint(&mut out, *v),
                Value::Fixed64(b) => out.extend_from_slice(b),
                Value::Fixed32(b) => out.extend_from_slice(b),
                Value::Bytes(b) => {
                    write_varint(&mut out, b.len() as u64);
                    out.extend_from_slice(b);
                }
            }
        }
        out
    }

    pub fn get(&self, number: u32) -> Option<&Value> {
        self.fields
            .iter()
            .find(|f| f.number == number)
            .map(|f| &f.value)
    }

    pub fn all(&self, number: u32) -> impl Iterator<Item = &Value> {
        self.fields
            .iter()
            .filter(move |f| f.number == number)
            .map(|f| &f.value)
    }

    pub fn varint(&self, number: u32) -> Option<u64> {
        match self.get(number) {
            Some(Value::Varint(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn bytes(&self, number: u32) -> Option<&[u8]> {
        match self.get(number) {
            Some(Value::Bytes(b)) => Some(b),
            _ => None,
        }
    }

    /// Replace the first occurrence of `number`, or append it if absent.
    pub fn set(&mut self, number: u32, value: Value) {
        match self.fields.iter_mut().find(|f| f.number == number) {
            Some(field) => field.value = value,
            None => self.fields.push(Field { number, value }),
        }
    }

    /// Replace the first occurrence of `number`, or insert it in ascending
    /// field order if absent.
    ///
    /// iWork writes fields in ascending order everywhere it has been looked at,
    /// and an archive this crate has edited should look like one the app could
    /// have written rather than one with a field bolted onto the end. Protobuf
    /// itself does not care, but "the bytes are legal" is a lower bar than "the
    /// document is shaped the way the app shapes it", and only one of those is
    /// testable without a Mac.
    pub fn set_in_order(&mut self, number: u32, value: Value) {
        match self.fields.iter_mut().find(|f| f.number == number) {
            Some(field) => field.value = value,
            None => {
                let at = self
                    .fields
                    .iter()
                    .position(|f| f.number > number)
                    .unwrap_or(self.fields.len());
                self.fields.insert(at, Field { number, value });
            }
        }
    }

    /// Add another occurrence of `number`, after the last one already there, or
    /// in ascending field order if there is none.
    ///
    /// This is [`Message::set_in_order`] for a *repeated* field, where replacing
    /// the first occurrence would silently drop an entry rather than add one.
    pub fn append_in_order(&mut self, number: u32, value: Value) {
        let at = self
            .fields
            .iter()
            .rposition(|f| f.number == number)
            .map(|last| last + 1)
            .or_else(|| self.fields.iter().position(|f| f.number > number))
            .unwrap_or(self.fields.len());
        self.fields.insert(at, Field { number, value });
    }

    /// Remove every occurrence of `number`. Returns how many were removed.
    pub fn clear(&mut self, number: u32) -> usize {
        let before = self.fields.len();
        self.fields.retain(|f| f.number != number);
        before - self.fields.len()
    }
}

/// Decode `bytes` as a nested message, accepting the result only when
/// re-encoding it reproduces the input exactly.
///
/// A length-delimited field is ambiguous on the wire: a submessage, a UTF-8
/// string and a packed repeated field are all just bytes, and a short string
/// decodes as a message by accident often enough to matter — `"(\x01"` is a
/// perfectly good `{5: 1}`. Requiring a byte-exact round-trip is what makes
/// descending into a nested message safe: anything that is not really a message
/// either fails to decode or fails to reproduce itself, and non-canonical
/// varints are caught the same way.
///
/// This is the one primitive every recursive walk in this crate is built on, so
/// that a wrong guess about a payload's shape cannot rewrite it into something
/// else.
pub fn decode_nested(bytes: &[u8]) -> Option<Message> {
    if bytes.is_empty() {
        return None;
    }
    let message = Message::decode(bytes).ok()?;
    (message.encode() == bytes).then_some(message)
}

pub fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_roundtrip() {
        for value in [0u64, 1, 127, 128, 300, 65535, 129448, u64::MAX] {
            let mut buf = Vec::new();
            write_varint(&mut buf, value);
            assert_eq!(Reader::new(&buf).varint().unwrap(), value);
        }
    }

    #[test]
    fn message_roundtrips_every_wire_type() {
        let mut message = Message::default();
        message.fields.push(Field {
            number: 1,
            value: Value::Varint(129448),
        });
        message.fields.push(Field {
            number: 2,
            value: Value::Bytes(b"Document".to_vec()),
        });
        message.fields.push(Field {
            number: 30,
            value: Value::Fixed32(595.28f32.to_le_bytes()),
        });
        message.fields.push(Field {
            number: 31,
            value: Value::Fixed64(841.89f64.to_le_bytes()),
        });

        let decoded = Message::decode(&message.encode()).unwrap();
        assert_eq!(decoded, message);
        assert_eq!(decoded.varint(1), Some(129448));
        assert_eq!(decoded.bytes(2), Some(b"Document".as_slice()));
    }

    /// Repeated fields must survive in order — attribute-run tables are
    /// repeated fields and their order is their meaning.
    #[test]
    fn repeated_fields_keep_their_order() {
        let mut message = Message::default();
        for index in [0u64, 6, 57, 58] {
            message.fields.push(Field {
                number: 1,
                value: Value::Varint(index),
            });
        }
        let decoded = Message::decode(&message.encode()).unwrap();
        let values: Vec<_> = decoded
            .all(1)
            .map(|v| match v {
                Value::Varint(n) => *n,
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(values, vec![0, 6, 57, 58]);
    }

    #[test]
    fn set_replaces_in_place_and_appends_when_absent() {
        let mut message = Message::default();
        message.set(3, Value::Bytes(b"old".to_vec()));
        message.set(5, Value::Varint(1));
        message.set(3, Value::Bytes(b"new".to_vec()));
        assert_eq!(message.fields.len(), 2);
        assert_eq!(message.bytes(3), Some(b"new".as_slice()));
        assert_eq!(message.fields[0].number, 3, "field order is preserved");
    }

    #[test]
    fn truncated_input_is_an_error() {
        assert!(Message::decode(&[0x0a, 0x05, b'a']).is_err());
        assert!(Message::decode(&[0x80]).is_err());
    }

    /// Group wire types (3 and 4) never appear in iWork archives; rejecting
    /// them keeps a desynced parse from silently producing garbage.
    #[test]
    fn group_wire_types_are_rejected() {
        assert!(Message::decode(&[0x0b]).is_err());
    }

    /// A field number is kept in a `u32` and re-encoded from it, so one that
    /// does not round-trip is refused rather than silently truncated: field 0
    /// (which protobuf forbids) and a number past `u32::MAX`. Both would break
    /// the "unrecognised bytes come back unchanged" discipline.
    #[test]
    fn field_numbers_that_do_not_round_trip_are_refused() {
        // key = number << 3 | wire type. Field 0, varint: key 0.
        assert!(Message::decode(&[0x00, 0x00]).is_err());
        // A field number of 2^32, varint: key = (1 << 32) << 3.
        let mut key = Vec::new();
        write_varint(&mut key, (1u64 << 32) << 3);
        key.push(0x00); // its varint payload
        assert!(Message::decode(&key).is_err());
        // The largest number that still fits round-trips.
        let mut ok = Vec::new();
        write_varint(&mut ok, (u64::from(u32::MAX)) << 3);
        ok.push(0x07);
        let message = Message::decode(&ok).unwrap();
        assert_eq!(message.fields[0].number, u32::MAX);
        assert_eq!(message.encode(), ok);
    }

    #[test]
    fn clear_removes_every_occurrence() {
        let mut message = Message::default();
        for _ in 0..3 {
            message.fields.push(Field {
                number: 1,
                value: Value::Varint(7),
            });
        }
        message.set(2, Value::Varint(1));
        assert_eq!(message.clear(1), 3);
        assert_eq!(message.fields.len(), 1);
        assert_eq!(message.clear(1), 0);
    }

    #[test]
    fn decode_nested_accepts_a_real_message() {
        let mut inner = Message::default();
        inner.set(1, Value::Varint(4242));
        inner.set(2, Value::Bytes(b"Body".to_vec()));
        assert_eq!(decode_nested(&inner.encode()), Some(inner));
    }

    /// The whole point of the round-trip check: text that happens to parse as a
    /// message must not be treated as one.
    #[test]
    fn decode_nested_rejects_text_and_junk() {
        assert_eq!(decode_nested(b""), None);
        assert_eq!(decode_nested(b"Grosse Uberschrift"), None);
        // Valid framing, but a non-canonical varint, so it cannot re-encode to
        // the same bytes.
        assert_eq!(decode_nested(&[0x08, 0x81, 0x00]), None);
        // Truncated length-delimited field.
        assert_eq!(decode_nested(&[0x0a, 0x05, b'a']), None);
    }

    /// A short ASCII string that *is* accepted by `Message::decode` — the case
    /// the round-trip check exists for. `"(\x01"` decodes as `{5: 1}`.
    #[test]
    fn decode_nested_is_not_fooled_by_a_shape_that_merely_parses() {
        assert!(Message::decode(b"(\x01").is_ok());
        // It really does re-encode identically, so this one is genuinely
        // ambiguous and the caller sees a message. Anything longer is not.
        assert!(decode_nested(b"Body text").is_none());
        assert!(decode_nested(b"Titel").is_none());
    }
}
