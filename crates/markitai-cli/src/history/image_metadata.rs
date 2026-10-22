//! Relocate only indexed, copied assets; image descriptions keep their other fields.
use super::{assets, invalid};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_ENTRIES: usize = 100_000;

#[derive(Default)]
pub(super) struct Indexes {
    targets: indexmap::IndexMap<PathBuf, Value>,
    read_bytes: usize,
    entries: usize,
    filtered: usize,
}

impl Indexes {
    pub(super) fn merge(
        &mut self,
        root: &Path,
        mapping: &HashMap<String, String>,
        skipped: &HashSet<String>,
        indexes: Vec<(PathBuf, PathBuf)>,
        allow_symlinks: bool,
    ) -> io::Result<()> {
        for (source, target) in indexes {
            let bytes = assets::read_index(
                &source,
                (MAX_BYTES - self.read_bytes) as u64,
                allow_symlinks,
            )?;
            self.read_bytes += bytes.len();
            let mut value: Value = serde_json::from_slice(&bytes)
                .map_err(|_| invalid("History image index is invalid JSON"))?;
            let object = value
                .as_object_mut()
                .ok_or_else(|| invalid("History image index must be an object"))?;
            let entries = match object.remove("images") {
                Some(Value::Array(entries)) => entries,
                _ => return Err(invalid("History image index entries must be an array")),
            };
            object.insert("images".into(), Value::Array(Vec::new()));
            let destination = self.targets.entry(target).or_insert(value);
            let destination = destination["images"].as_array_mut().unwrap();
            for mut entry in entries {
                self.entries += 1;
                if self.entries > MAX_ENTRIES {
                    return Err(invalid("History image index exceeds the entry limit"));
                }
                let relative = entry
                    .get("path")
                    .and_then(Value::as_str)
                    .and_then(|path| root_relative(path, root));
                let Some(relocated) = relative.as_ref().and_then(|path| mapping.get(path)) else {
                    // Another document's picture is left out on purpose.
                    if !relative.is_some_and(|path| skipped.contains(&path)) {
                        self.filtered += 1;
                    }
                    continue;
                };
                let relocated = relocated.clone();
                entry["path"] = relocated.into();
                // Different analyses of one deduplicated asset are still distinct
                // records. Replacing by path here would lose source descriptions.
                destination.push(entry);
            }
        }
        Ok(())
    }

    pub(super) fn publish(
        self,
        stage_out: &Path,
        final_out: &Path,
        budget: &mut assets::Budget,
        allow_symlinks: bool,
    ) -> io::Result<()> {
        let final_out = crate::report_store::resolve_path(final_out)?;
        for (target, mut value) in self.targets {
            let images = value["images"].as_array_mut().unwrap();
            for entry in images {
                let relative = entry["path"].as_str().unwrap();
                // Mapping values are actual files copied into this private stage.
                let staged = markitai_core::output::join_relative(stage_out, relative);
                let metadata = std::fs::symlink_metadata(&staged)?;
                if !metadata.is_file() {
                    return Err(invalid(
                        "History indexed asset is not a regular copied file",
                    ));
                }
                // One separator throughout: `\` on Windows.
                entry["path"] = markitai_core::output::join_relative(&final_out, relative)
                    .to_string_lossy()
                    .into_owned()
                    .into();
            }
            let mut bytes = super::BoundedJson(Vec::new());
            serde_json::to_writer_pretty(&mut bytes, &value).map_err(io::Error::other)?;
            bytes.write_all(b"\n")?;
            assets::write_index(&target, &bytes.0, budget, allow_symlinks)?;
        }
        if self.filtered > 0 {
            eprintln!(
                "Warning: History omitted {} image metadata records without an asset in this archive.",
                self.filtered
            );
        }
        Ok(())
    }
}

/// An indexed path relative to its asset root, `/`-separated.
fn root_relative(path: &str, root: &Path) -> Option<String> {
    if path.is_empty() || path.contains('\0') || path.contains("://") || path.starts_with("data:") {
        return None;
    }
    let path = Path::new(path);
    let joined = if path.is_absolute() {
        path.to_owned()
    } else {
        root.join(path)
    };
    let resolved = crate::report_store::resolve_path(&joined).ok()?;
    let relative = resolved.strip_prefix(root).ok()?;
    let relative = relative
        .iter()
        .map(|part| part.to_str())
        .collect::<Option<Vec<_>>>()?
        .join("/");
    Some(relative)
}
