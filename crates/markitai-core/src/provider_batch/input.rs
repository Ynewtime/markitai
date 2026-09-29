use super::types::{ENDPOINT, Error, Limits, custom_id};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub(super) struct Input {
    pub file: File,
    pub model: String,
    pub custom_ids: Vec<String>,
    pub bytes: u64,
    pub sha256: String,
}

/// Read at most one bounded line. A cap+1 byte is consumed only to detect overflow.
pub(super) fn line(reader: &mut impl BufRead, cap: usize) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::new();
    reader
        .take(cap.saturating_add(1) as u64)
        .read_until(b'\n', &mut bytes)
        .map_err(|_| Error::Transport)?;
    if bytes.len() > cap {
        return Err(Error::Limit("Batch JSONL line exceeds its byte limit"));
    }
    Ok(bytes)
}

/// Validate the same anonymous, private snapshot later supplied to multipart.
/// A mutable source pathname is never reopened after this function returns.
pub(super) fn snapshot(path: &Path, limits: Limits) -> Result<Input, Error> {
    let before = std::fs::symlink_metadata(path).map_err(|_| Error::Input)?;
    if !before.is_file() || before.file_type().is_symlink() {
        return Err(Error::Invalid(
            "Batch input must be a regular non-symlink file",
        ));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path).map_err(|_| Error::Input)?;
    let metadata = file.metadata().map_err(|_| Error::Input)?;
    if !metadata.is_file() || metadata.len() > limits.upload_bytes as u64 {
        return Err(Error::Limit(
            "Batch input exceeds its byte limit or is not regular",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != metadata.dev() || before.ino() != metadata.ino() {
            return Err(Error::Input);
        }
    }
    let mut source = BufReader::new(file);
    let mut staged = tempfile::tempfile().map_err(|_| Error::Input)?;
    let mut bytes = 0usize;
    let mut hash = Sha256::new();
    let mut ids = Vec::new();
    let mut unique = HashSet::new();
    let mut model: Option<String> = None;
    loop {
        let row = line(
            &mut source,
            limits.line_bytes.min(limits.upload_bytes - bytes),
        )
        .map_err(|error| match error {
            Error::Transport => Error::Input,
            other => other,
        })?;
        if row.is_empty() {
            break;
        }
        bytes = bytes
            .checked_add(row.len())
            .ok_or(Error::Limit("Batch input is too large"))?;
        if bytes > limits.upload_bytes || ids.len() >= limits.requests {
            return Err(Error::Limit(
                "Batch input exceeds its request or byte limit",
            ));
        }
        let value: Value = serde_json::from_slice(&row)
            .map_err(|_| Error::Invalid("Batch input contains invalid JSONL"))?;
        let object = value
            .as_object()
            .ok_or(Error::Invalid("Batch input row must be an object"))?;
        if object.len() != 4 || value["method"] != "POST" || value["url"] != ENDPOINT {
            return Err(Error::Invalid(
                "Batch input must contain only custom_id, method, url and body for Chat Completions",
            ));
        }
        let id = value["custom_id"]
            .as_str()
            .filter(|id| custom_id(id))
            .ok_or(Error::Invalid("Batch custom_id is invalid"))?;
        if !unique.insert(id.to_owned()) {
            return Err(Error::Invalid(
                "Batch input contains duplicate custom_id values",
            ));
        }
        let body = value["body"]
            .as_object()
            .ok_or(Error::Invalid("Batch request body must be an object"))?;
        let name = body
            .get("model")
            .and_then(Value::as_str)
            .filter(|name| {
                !name.trim().is_empty() && name.len() <= 256 && !name.chars().any(char::is_control)
            })
            .ok_or(Error::Invalid("Batch model is invalid"))?;
        if model.as_deref().is_some_and(|prior| prior != name) {
            return Err(Error::Invalid("Batch input must use one model"));
        }
        if !body
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|v| !v.is_empty())
            || body.get("stream").is_some_and(|v| v != &Value::Bool(false))
        {
            return Err(Error::Invalid(
                "Batch messages are required and streaming is forbidden",
            ));
        }
        model.get_or_insert_with(|| name.to_owned());
        ids.push(id.to_owned());
        staged.write_all(&row).map_err(|_| Error::Input)?;
        hash.update(&row);
    }
    if ids.is_empty() {
        return Err(Error::Invalid("Batch input contains no requests"));
    }
    staged.flush().map_err(|_| Error::Input)?;
    staged.seek(SeekFrom::Start(0)).map_err(|_| Error::Input)?;
    Ok(Input {
        file: staged,
        model: model.expect("validated request"),
        custom_ids: ids,
        bytes: bytes as u64,
        sha256: crate::hex(hash.finalize()),
    })
}
