//! IWA (iWork Archive) container: Snappy-framed streams of protobuf objects.
//!
//! Layout of a `.iwa` file:
//!
//! ```text
//! repeat until EOF:
//!   u8   0x00                 marker
//!   u24  compressed_length    little endian
//!   ..   snappy raw block     (decompresses to <= 64 KiB)
//! ```
//!
//! Concatenating every decompressed block yields the object stream. Object
//! framing is independent of block boundaries, so the blocks must be joined
//! before parsing:
//!
//! ```text
//! repeat until end of stream:
//!   varint  archive_info_length
//!   ..      ArchiveInfo message
//!   ..      payload of each MessageInfo, back to back, in declaration order
//! ```

use crate::pb::{write_varint, Message, Reader, Value};

pub const BLOCK_SIZE: usize = 65536;

/// One `TSP.MessageInfo` plus the payload bytes it describes.
#[derive(Debug, Clone)]
pub struct ArchiveMessage {
    /// `MessageInfo.type` — index into iWork's global message-type registry.
    pub message_type: u32,
    /// `MessageInfo.version` — repeated uint32, e.g. `[1, 0, 5]`.
    pub version: Vec<u32>,
    /// Fields of the MessageInfo we do not interpret, preserved for round-trips.
    pub extra: Vec<crate::pb::Field>,
    pub payload: Vec<u8>,
}

/// One `TSP.ArchiveInfo` and its messages.
#[derive(Debug, Clone)]
pub struct ArchiveObject {
    /// Package-unique object identifier.
    pub identifier: u64,
    pub messages: Vec<ArchiveMessage>,
    pub extra: Vec<crate::pb::Field>,
}

impl ArchiveObject {
    /// The first message's payload, or nothing.
    ///
    /// An `ArchiveInfo` with no `MessageInfo` in it is not something any app
    /// writes — every object in the corpus has at least one — but it is three
    /// bytes to write by hand, and indexing `messages[0]` for it was a panic
    /// in the middle of an otherwise ordinary document. `message_type` has
    /// always answered 0 for the same object; this answers no bytes.
    pub fn payload(&self) -> &[u8] {
        self.messages
            .first()
            .map(|m| m.payload.as_slice())
            .unwrap_or_default()
    }

    pub fn message_type(&self) -> u32 {
        self.messages.first().map(|m| m.message_type).unwrap_or(0)
    }
}

pub fn decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        if data.len() - pos < 4 {
            return Err("truncated IWA chunk header".into());
        }
        if data[pos] != 0x00 {
            return Err(format!("unexpected chunk marker {:#04x}", data[pos]));
        }
        let len = u32::from_le_bytes([data[pos + 1], data[pos + 2], data[pos + 3], 0]) as usize;
        pos += 4;
        let end = pos + len;
        if end > data.len() {
            return Err("truncated IWA chunk body".into());
        }
        // What the block *says* it decompresses to, before anything allocates
        // that much. A raw Snappy block begins with the uncompressed length as
        // a varint, and nothing stops a hostile one from being four gigabytes
        // in five bytes: `snap` believes it and allocates. 24,358 blocks — the
        // 26 fixtures and all 901 bundled templates — are at most 65,536 bytes
        // and not one is over, which is the block size Apple's writer and this
        // one both use.
        match snap::raw::decompress_len(&data[pos..end]) {
            Ok(size) if size > BLOCK_SIZE => {
                return Err(format!(
                    "IWA chunk claims to decompress to {size} bytes, over the {BLOCK_SIZE}-byte \
                     block size"
                ))
            }
            Ok(_) => {}
            Err(e) => return Err(format!("snappy: {e}")),
        }
        let block = snap::raw::Decoder::new()
            .decompress_vec(&data[pos..end])
            .map_err(|e| format!("snappy: {e}"))?;
        out.extend_from_slice(&block);
        pos = end;
    }
    Ok(out)
}

pub fn compress(stream: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for block in stream.chunks(BLOCK_SIZE) {
        let compressed = snap::raw::Encoder::new()
            .compress_vec(block)
            .expect("snappy encode is infallible for in-memory buffers");
        out.push(0x00);
        out.extend_from_slice(&(compressed.len() as u32).to_le_bytes()[..3]);
        out.extend_from_slice(&compressed);
    }
    out
}

pub fn parse(data: &[u8]) -> Result<Vec<ArchiveObject>, String> {
    let stream = decompress(data)?;
    let mut reader = Reader::new(&stream);
    let mut objects = Vec::new();

    while !reader.done() {
        let info_len = reader.varint()? as usize;
        let info = Message::decode(reader.take(info_len)?)?;

        let mut identifier = 0u64;
        let mut messages = Vec::new();
        let mut extra = Vec::new();
        let mut claimed = 0usize;

        for field in &info.fields {
            match (field.number, &field.value) {
                (1, Value::Varint(v)) => identifier = *v,
                (2, Value::Bytes(raw)) => {
                    let mi = Message::decode(raw)?;
                    let mut message_type = 0u32;
                    let mut version = Vec::new();
                    let mut len = 0usize;
                    let mut mi_extra = Vec::new();
                    for f in &mi.fields {
                        match (f.number, &f.value) {
                            (1, Value::Varint(v)) => message_type = *v as u32,
                            (2, Value::Bytes(b)) => {
                                let mut r = Reader::new(b);
                                while !r.done() {
                                    version.push(r.varint()? as u32);
                                }
                            }
                            (3, Value::Varint(v)) => len = *v as usize,
                            _ => mi_extra.push(f.clone()),
                        }
                    }
                    // `length` is a varint out of the file and the payload it
                    // describes has to be *in* the file: a stream saying its
                    // next message is 2^60 bytes long would otherwise be a
                    // request to allocate 2^60 bytes, and the reader would die
                    // of a corrupt integer rather than report one. The bytes
                    // are read below; this only refuses to reserve room for
                    // more than the stream can possibly hold.
                    let remaining = stream.len() - reader.pos;
                    claimed = claimed.saturating_add(len);
                    if claimed > remaining {
                        return Err(format!(
                            "object {identifier}: its messages claim {claimed} bytes of payload \
                             and {remaining} remain in the stream"
                        ));
                    }
                    messages.push(ArchiveMessage {
                        message_type,
                        version,
                        extra: mi_extra,
                        payload: vec![0; len],
                    });
                }
                _ => extra.push(field.clone()),
            }
        }

        for message in &mut messages {
            let len = message.payload.len();
            message.payload.copy_from_slice(reader.take(len)?);
        }

        objects.push(ArchiveObject {
            identifier,
            messages,
            extra,
        });
    }

    Ok(objects)
}

/// Frame objects into an object stream, without compressing it.
///
/// Kept separate from [`serialize`] so a caller can ask whether re-encoding
/// would actually change anything before paying for Snappy — and, more to the
/// point, before replacing bytes that did not need replacing. See
/// [`crate::Document::save`].
pub fn serialize_stream(objects: &[ArchiveObject]) -> Vec<u8> {
    let mut stream = Vec::new();
    for object in objects {
        let mut info = Message::default();
        info.fields.push(crate::pb::Field {
            number: 1,
            value: Value::Varint(object.identifier),
        });
        for message in &object.messages {
            let mut mi = Message::default();
            mi.fields.push(crate::pb::Field {
                number: 1,
                value: Value::Varint(u64::from(message.message_type)),
            });
            if !message.version.is_empty() {
                let mut packed = Vec::new();
                for v in &message.version {
                    write_varint(&mut packed, u64::from(*v));
                }
                mi.fields.push(crate::pb::Field {
                    number: 2,
                    value: Value::Bytes(packed),
                });
            }
            mi.fields.push(crate::pb::Field {
                number: 3,
                value: Value::Varint(message.payload.len() as u64),
            });
            mi.fields.extend(message.extra.iter().cloned());
            info.fields.push(crate::pb::Field {
                number: 2,
                value: Value::Bytes(mi.encode()),
            });
        }
        info.fields.extend(object.extra.iter().cloned());

        let encoded = info.encode();
        write_varint(&mut stream, encoded.len() as u64);
        stream.extend_from_slice(&encoded);
        for message in &object.messages {
            stream.extend_from_slice(&message.payload);
        }
    }
    stream
}

/// Frame objects into an object stream and compress it, ready to be a package
/// entry.
pub fn serialize(objects: &[ArchiveObject]) -> Vec<u8> {
    compress(&serialize_stream(objects))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(identifier: u64, message_type: u32, payload: Vec<u8>) -> ArchiveObject {
        ArchiveObject {
            identifier,
            messages: vec![ArchiveMessage {
                message_type,
                version: vec![1, 0, 5],
                extra: Vec::new(),
                payload,
            }],
            extra: Vec::new(),
        }
    }

    #[test]
    fn framing_roundtrips() {
        let objects = vec![
            object(1, 10000, b"hello".to_vec()),
            object(4242, 2001, b"some text payload".to_vec()),
        ];
        let parsed = parse(&serialize(&objects)).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].identifier, 1);
        assert_eq!(parsed[0].message_type(), 10000);
        assert_eq!(parsed[1].payload(), b"some text payload");
        assert_eq!(parsed[1].messages[0].version, vec![1, 0, 5]);
    }

    /// Objects are framed independently of Snappy blocks, so one must be able
    /// to straddle a block boundary. Parsing block by block instead of over the
    /// concatenated stream is the classic way to get this wrong.
    #[test]
    fn objects_span_block_boundaries() {
        // Incompressible payloads, so blocks stay full and the boundary lands
        // inside an object rather than between two.
        let payloads: Vec<Vec<u8>> = (0..8)
            .map(|n: u32| {
                (0..20_000u32)
                    .flat_map(|i| i.wrapping_mul(2654435761).wrapping_add(n).to_le_bytes())
                    .collect()
            })
            .collect();
        let objects: Vec<_> = payloads
            .iter()
            .enumerate()
            .map(|(i, p)| object(i as u64 + 1, 2001, p.clone()))
            .collect();

        let encoded = serialize(&objects);
        assert!(
            decompress(&encoded).unwrap().len() > BLOCK_SIZE * 3,
            "test needs a multi-block stream"
        );

        let parsed = parse(&encoded).unwrap();
        assert_eq!(parsed.len(), objects.len());
        for (i, o) in parsed.iter().enumerate() {
            assert_eq!(o.identifier, i as u64 + 1);
            assert_eq!(o.payload(), payloads[i].as_slice());
        }
    }

    #[test]
    fn no_block_exceeds_the_limit() {
        let big = object(1, 2001, vec![7u8; BLOCK_SIZE * 3 + 11]);
        let encoded = serialize(&[big]);
        let mut pos = 0;
        while pos < encoded.len() {
            let len = u32::from_le_bytes([encoded[pos + 1], encoded[pos + 2], encoded[pos + 3], 0])
                as usize;
            let block = snap::raw::Decoder::new()
                .decompress_vec(&encoded[pos + 4..pos + 4 + len])
                .unwrap();
            assert!(block.len() <= BLOCK_SIZE);
            pos += 4 + len;
        }
    }

    #[test]
    fn rejects_a_bad_chunk_marker() {
        let mut encoded = serialize(&[object(1, 2001, b"x".to_vec())]);
        encoded[0] = 0x01;
        assert!(parse(&encoded).is_err());
    }

    #[test]
    fn empty_stream_has_no_objects() {
        assert!(parse(&[]).unwrap().is_empty());
    }
}
