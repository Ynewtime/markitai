use super::{
    Checkpoint, Entry, Error, Event, Fence, ItemKey, Limits, Mode, Result, Scope, Snapshot, Status,
};
use indexmap::IndexMap;
use serde::de::{self, DeserializeSeed, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

fn foreign(message: &str) -> Error {
    Error::ForeignScope(message.into())
}

fn object<'a>(value: &'a Value, message: &str) -> Result<&'a Map<String, Value>> {
    value.as_object().ok_or_else(|| invalid(message))
}

fn parse(bytes: &[u8]) -> Result<Value> {
    // A line reader distinguishes invalid UTF-8 from skippable JSON syntax too.
    std::str::from_utf8(bytes).map_err(|_| invalid("JSON is not UTF-8"))?;
    serde_json::from_slice(bytes).map_err(|error| {
        let location = format!("syntax at line {}, column {}", error.line(), error.column());
        if error.is_syntax() || error.is_eof() {
            Error::JsonSyntax(location)
        } else {
            invalid("JSON value cannot be decoded")
        }
    })
}

#[derive(Default)]
struct RawSnapshot {
    version: Option<String>,
    options: Option<IndexMap<String, Value>>,
    documents: IndexMap<String, Value>,
    urls: IndexMap<String, Value>,
    checkpoint: Option<Checkpoint>,
}

struct SnapshotSeed {
    remaining: usize,
}

impl<'de> DeserializeSeed<'de> for SnapshotSeed {
    type Value = RawSnapshot;
    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for SnapshotSeed {
    type Value = RawSnapshot;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a recovery state object")
    }
    fn visit_map<A: MapAccess<'de>>(
        mut self,
        mut map: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut raw = RawSnapshot::default();
        let mut seen = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom("duplicate base field"));
            }
            match key.as_str() {
                "version" => raw.version = Some(map.next_value()?),
                "options" => {
                    let mut remaining = usize::MAX;
                    raw.options = Some(map.next_value_seed(EntriesSeed(&mut remaining))?);
                }
                "documents" => {
                    raw.documents = map.next_value_seed(EntriesSeed(&mut self.remaining))?
                }
                "urls" => raw.urls = map.next_value_seed(EntriesSeed(&mut self.remaining))?,
                "_markitai" => raw.checkpoint = Some(map.next_value()?),
                _ => {
                    map.next_value::<de::IgnoredAny>()?;
                }
            }
        }
        Ok(raw)
    }
}

struct EntriesSeed<'a>(&'a mut usize);

impl<'de> DeserializeSeed<'de> for EntriesSeed<'_> {
    type Value = IndexMap<String, Value>;
    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for EntriesSeed<'_> {
    type Value = IndexMap<String, Value>;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("an item map")
    }
    fn visit_map<A: MapAccess<'de>>(
        self,
        mut map: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut entries = IndexMap::new();
        while let Some(key) = map.next_key::<String>()? {
            if *self.0 == 0 {
                return Err(de::Error::custom("state entry limit"));
            }
            if entries.contains_key(&key) {
                return Err(de::Error::custom("duplicate item key"));
            }
            *self.0 -= 1;
            entries.insert(key, map.next_value()?);
        }
        Ok(entries)
    }
}

fn optional_text(value: Option<&Value>, field: &str) -> Result<Option<String>> {
    match value {
        None | Some(Value::Null | Value::Bool(false)) => Ok(None),
        Some(Value::String(text)) if text.is_empty() => Ok(None),
        Some(Value::String(text)) if !text.contains('\0') => Ok(Some(text.clone())),
        _ => Err(invalid(&format!("{field} must be a string or null"))),
    }
}

fn status(value: Option<&Value>) -> Result<Status> {
    match value {
        None => Ok(Status::Pending),
        Some(Value::String(text)) => match text.as_str() {
            "pending" => Ok(Status::Pending),
            "in_progress" => Ok(Status::InProgress),
            "completed" => Ok(Status::Completed),
            "failed" => Ok(Status::Failed),
            _ => Err(invalid("unknown item status")),
        },
        _ => Err(invalid("item status must be a string")),
    }
}

fn generation(value: &str) -> Result<()> {
    uuid::Uuid::parse_str(value)
        .map(|_| ())
        .map_err(|_| invalid("checkpoint generation must be a UUID"))
}

fn check_scope(scope: &Scope, allow_symlinks: bool) -> Result<()> {
    if !scope.input.is_absolute() || !scope.output.is_absolute() {
        return Err(foreign("expected scope must use resolved absolute paths"));
    }
    if super::paths::resolve(&scope.input)? != scope.input
        || super::paths::resolve(&scope.output)? != scope.output
    {
        return Err(foreign(
            "expected scope no longer resolves to its recorded paths",
        ));
    }
    check_symlinks(scope.output_spelling(), allow_symlinks)
}

fn check_symlinks(path: &Path, allow_symlinks: bool) -> Result<()> {
    match super::paths::symlinks_permitted(path, allow_symlinks) {
        Ok(true) => Ok(()),
        _ => Err(foreign("saved path violates the symlink policy")),
    }
}

fn validate_options(options: &IndexMap<String, Value>, scope: &Scope) -> Result<()> {
    for (field, expected) in [
        (
            "input_dir",
            if scope.mode == Mode::Directory {
                scope.input.as_path()
            } else {
                scope
                    .input
                    .parent()
                    .ok_or_else(|| foreign("URL list has no parent"))?
            },
        ),
        ("output_dir", scope.output.as_path()),
    ] {
        if let Some(text) = optional_text(options.get(field), field)?
            && super::paths::resolve(Path::new(&text))? != expected
        {
            return Err(foreign("legacy options identify another input or output"));
        }
    }
    Ok(())
}

fn validate_checkpoint(checkpoint: &Checkpoint, scope: &Scope) -> Result<()> {
    generation(&checkpoint.generation)?;
    if checkpoint.scope != *scope {
        return Err(foreign("native checkpoint identifies another run"));
    }
    Ok(())
}

fn metadata_component(path: &Path) -> bool {
    path.components().any(|part| matches!(part, Component::Normal(name) if name.to_str().is_some_and(|name| name.eq_ignore_ascii_case(".markitai"))))
}

fn file_key(key: &str, scope: &Scope, allow_symlinks: bool) -> Result<String> {
    if scope.mode != Mode::Directory {
        return Err(foreign("URL-list state cannot contain documents"));
    }
    if key.is_empty() || key.contains('\0') {
        return Err(foreign("document key is empty or invalid"));
    }
    let path = Path::new(key);
    if path.components().any(|part| part == Component::ParentDir) {
        return Err(foreign("document key contains parent traversal"));
    }
    let relative = if path.is_absolute() {
        check_symlinks(path, allow_symlinks)?;
        super::paths::resolve(path)?
            .strip_prefix(&scope.input)
            .map_err(|_| foreign("absolute document key is outside input"))?
            .to_owned()
    } else {
        path.to_owned()
    };
    let normalized: PathBuf = relative
        .components()
        .filter(|part| *part != Component::CurDir)
        .collect();
    if normalized.as_os_str().is_empty()
        || normalized.is_absolute()
        || metadata_component(&normalized)
    {
        return Err(foreign("document key is not a normal input file"));
    }
    let original = scope.input.join(&normalized);
    check_symlinks(&original, allow_symlinks)?;
    if !super::paths::resolve(&original)?.starts_with(&scope.input) {
        return Err(foreign("document key resolves outside input"));
    }
    normalized
        .to_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid("document key is not UTF-8"))
}

fn url_identity(key: &str, explicit: Option<&str>) -> Result<String> {
    let (bare, name) = key
        .split_once(' ')
        .map_or((key, None), |(url, name)| (url, Some(name)));
    if !(bare.starts_with("http://") || bare.starts_with("https://"))
        || bare.chars().any(char::is_whitespace)
        || bare.contains('\0')
        || name.is_some_and(|name| name.is_empty() || name.contains('\0'))
    {
        return Err(invalid("URL state key is invalid"));
    }
    if explicit.is_some_and(|url| url != bare) {
        return Err(foreign("URL entry identity differs from its key"));
    }
    Ok(bare.into())
}

pub(crate) fn entry_parent(
    key: &ItemKey,
    entry: &Entry,
    scope: &Scope,
    allow_symlinks: bool,
) -> Result<PathBuf> {
    // One consistent filesystem observation for this pure validation.
    let _paths = super::paths::Scope::enter();
    let relative_parent = match key {
        ItemKey::File(key) => {
            let normalized = file_key(key, scope, allow_symlinks)?;
            Path::new(&normalized)
                .parent()
                .unwrap_or(Path::new(""))
                .to_owned()
        }
        ItemKey::Url(key) => {
            url_identity(key, entry.url.as_deref())?;
            if let Some(source) = entry
                .source_file
                .as_deref()
                .filter(|source| !source.is_empty())
            {
                let source = Path::new(source);
                check_symlinks(source, allow_symlinks)?;
                let source = super::paths::resolve(source)?;
                match scope.mode {
                    Mode::UrlList => {
                        if source != scope.input {
                            return Err(foreign("URL source does not match the input list"));
                        }
                        PathBuf::new()
                    }
                    Mode::Directory => {
                        let relative = source
                            .strip_prefix(&scope.input)
                            .map_err(|_| foreign("URL source is outside input"))?;
                        if relative.as_os_str().is_empty() || metadata_component(relative) {
                            return Err(foreign("URL source is not an input list"));
                        }
                        relative.parent().unwrap_or(Path::new("")).to_owned()
                    }
                }
            } else {
                // Legacy states may omit source provenance; do not invent it.
                PathBuf::new()
            }
        }
    };
    let parent = scope.output.join(relative_parent);
    check_symlinks(&parent, allow_symlinks)?;
    let parent = super::paths::resolve(&parent)?;
    if !parent.starts_with(&scope.output) {
        return Err(foreign("mirrored output parent resolves outside output"));
    }
    Ok(parent)
}

fn destination(
    path: &Path,
    expected_parent: &Path,
    scope: &Scope,
    allow_symlinks: bool,
) -> Result<PathBuf> {
    if path.as_os_str().is_empty() {
        return Err(foreign("saved destination is empty"));
    }
    check_symlinks(path, allow_symlinks)?;
    let resolved = super::paths::resolve(path)?;
    let spelling = lexical_absolute(path)?;
    if super::paths::resolve(&spelling)? != resolved {
        return Err(foreign(
            "lexical destination changes the resolved symlink target",
        ));
    }
    let relative = resolved
        .strip_prefix(&scope.output)
        .map_err(|_| foreign("saved destination is outside output"))?;
    if relative.as_os_str().is_empty()
        || metadata_component(relative)
        || resolved.parent() != Some(expected_parent)
    {
        return Err(foreign(
            "saved destination does not match the mirrored output parent",
        ));
    }
    if let Ok(metadata) = std::fs::metadata(&resolved)
        && !metadata.is_file()
    {
        return Err(foreign("saved destination is not a normal file"));
    }
    Ok(spelling)
}

fn lexical_absolute(path: &Path) -> Result<PathBuf> {
    let mut result = PathBuf::new();
    for component in std::path::absolute(path)?.components() {
        match component {
            Component::CurDir => (),
            Component::ParentDir => {
                result.pop();
            }
            _ => result.push(component.as_os_str()),
        }
    }
    Ok(result)
}

fn validate_entry(
    key: &ItemKey,
    entry: &mut Entry,
    scope: &Scope,
    allow_symlinks: bool,
) -> Result<()> {
    let parent = entry_parent(key, entry, scope, allow_symlinks)?;
    for value in [&mut entry.output, &mut entry.target].into_iter().flatten() {
        *value = destination(value, &parent, scope, allow_symlinks)?;
    }
    Ok(())
}

fn decode_entry(
    value: &Value,
    key: &ItemKey,
    scope: &Scope,
    allow_symlinks: bool,
    native: bool,
) -> Result<Entry> {
    let data = object(value, "item must be an object")?;
    let mut entry = Entry {
        status: status(data.get("status"))?,
        source_file: match key {
            ItemKey::File(_) => None,
            ItemKey::Url(_) => match data.get("source_file") {
                None => Some(String::new()),
                Some(Value::Null) => None,
                Some(Value::String(text)) if !text.contains('\0') => Some(text.clone()),
                _ => return Err(invalid("source_file must be a string or null")),
            },
        },
        url: optional_text(data.get("url"), "url")?,
        output: optional_text(data.get("output"), "output")?.map(PathBuf::from),
        target: optional_text(data.get("target"), "target")?.map(PathBuf::from),
        error: optional_text(data.get("error"), "error")?,
        observations: Map::new(),
    };
    if let ItemKey::Url(key) = key {
        entry.url = Some(url_identity(key, entry.url.as_deref())?);
    }
    for name in [
        "started_at",
        "completed_at",
        "duration",
        "images",
        "screenshots",
        "cost_usd",
        "llm_usage",
        "cache_hit",
        "fetch_strategy",
        "warnings",
        "skip_reason",
    ] {
        if let Some(value) = data.get(name) {
            entry.observations.insert(name.into(), value.clone());
        }
    }
    if native && let Some(diagnostics) = attempt_diagnostics(data.get("diagnostics"))? {
        entry.observations.insert(
            "diagnostics".into(),
            serde_json::to_value(diagnostics)
                .map_err(|_| invalid("invalid native attempt diagnostics"))?,
        );
    }
    validate_entry(key, &mut entry, scope, allow_symlinks)?;
    Ok(entry)
}

fn count(snapshot: &Snapshot, limits: Limits) -> Result<()> {
    if snapshot.documents.len().saturating_add(snapshot.urls.len()) > limits.entries {
        return Err(Error::Limit("entry count"));
    }
    Ok(())
}

pub(crate) fn decode(
    bytes: &[u8],
    scope: &Scope,
    allow_symlinks: bool,
    limits: Limits,
) -> Result<Snapshot> {
    // One consistent filesystem observation for this pure validation.
    let _paths = super::paths::Scope::enter();
    if bytes.len() > limits.base_bytes {
        return Err(Error::Limit("base bytes"));
    }
    check_scope(scope, allow_symlinks)?;
    std::str::from_utf8(bytes).map_err(|_| invalid("JSON is not UTF-8"))?;
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let raw = SnapshotSeed {
        remaining: limits.entries,
    }
    .deserialize(&mut deserializer)
    .map_err(|error| {
        if error.is_syntax() || error.is_eof() {
            Error::JsonSyntax(format!(
                "syntax at line {}, column {}",
                error.line(),
                error.column()
            ))
        } else if error.to_string().starts_with("state entry limit") {
            Error::Limit("entry count")
        } else {
            invalid("invalid or duplicate base fields")
        }
    })?;
    deserializer
        .end()
        .map_err(|_| Error::JsonSyntax("trailing content after base".into()))?;
    if raw
        .version
        .as_deref()
        .is_some_and(|version| version != "1.0")
    {
        return Err(invalid("unsupported state version"));
    }
    let options = raw.options.unwrap_or_default();
    validate_options(&options, scope)?;
    let checkpoint = raw.checkpoint;
    if let Some(checkpoint) = &checkpoint {
        validate_checkpoint(checkpoint, scope)?;
    }
    let mut snapshot = Snapshot {
        options,
        checkpoint,
        ..Snapshot::default()
    };
    for (key, value) in raw.documents {
        let key = file_key(&key, scope, allow_symlinks)?;
        let entry = decode_entry(
            &value,
            &ItemKey::File(key.clone()),
            scope,
            allow_symlinks,
            snapshot.checkpoint.is_some(),
        )?;
        if snapshot.documents.insert(key, entry).is_some() {
            return Err(invalid("document keys normalize to the same identity"));
        }
    }
    for (key, value) in raw.urls {
        let entry = decode_entry(
            &value,
            &ItemKey::Url(key.clone()),
            scope,
            allow_symlinks,
            snapshot.checkpoint.is_some(),
        )?;
        snapshot.urls.insert(key, entry);
    }
    Ok(snapshot)
}

// This is a native extension; legacy minimal encodings and replay keep their
// reference behavior. Null explicitly clears an observation for a new attempt.
fn attempt_diagnostics(
    value: Option<&Value>,
) -> Result<Option<crate::diagnostics::AttemptDiagnostics>> {
    value
        .filter(|value| !value.is_null())
        .map(|value| {
            crate::diagnostics::AttemptDiagnostics::from_value(value)
                .map_err(|_| invalid("invalid native attempt diagnostics"))
        })
        .transpose()
}

#[derive(Serialize)]
struct WireEntry<'a> {
    status: Status,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_file: Option<Option<&'a str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diagnostics: Option<crate::diagnostics::AttemptDiagnostics>,
}

fn wire_entry<'a>(
    key: &ItemKey,
    entry: &'a Entry,
    scope: &Scope,
    allow_symlinks: bool,
    native: bool,
) -> Result<WireEntry<'a>> {
    let parent = entry_parent(key, entry, scope, allow_symlinks)?;
    let output = entry
        .output
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty())
        .map(|path| destination(path, &parent, scope, allow_symlinks))
        .transpose()?;
    let target = entry
        .target
        .as_deref()
        .filter(|path| !path.as_os_str().is_empty())
        .map(|path| destination(path, &parent, scope, allow_symlinks))
        .transpose()?;
    let is_url = matches!(key, ItemKey::Url(_));
    let named = matches!(key, ItemKey::Url(key) if key.contains(' '));
    let url = if named {
        match key {
            ItemKey::Url(key) => Some(
                entry
                    .url
                    .as_deref()
                    .unwrap_or_else(|| key.split_once(' ').unwrap().0)
                    .to_owned(),
            ),
            _ => None,
        }
    } else {
        None
    };
    let diagnostics = if native {
        attempt_diagnostics(entry.observations.get("diagnostics"))?
    } else {
        None
    };
    Ok(WireEntry {
        diagnostics,
        status: entry.status,
        source_file: is_url.then_some(entry.source_file.as_deref()),
        url,
        output: (entry.status == Status::Completed)
            .then_some(output)
            .flatten(),
        error: (entry.status == Status::Failed)
            .then_some(entry.error.as_deref().filter(|text| !text.is_empty()))
            .flatten(),
        target: (entry.status != Status::Completed)
            .then_some(target)
            .flatten(),
    })
}

struct OrderedOptions<'a>(&'a IndexMap<String, Value>);

impl Serialize for OrderedOptions<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let options = self.0;
        let preferred = [
            "concurrency",
            "llm",
            "cache",
            "ocr",
            "screenshot",
            "alt",
            "desc",
            "fetch_strategy",
            "models",
            "input_dir",
            "output_dir",
        ];
        let mut map = serializer.serialize_map(Some(options.len()))?;
        for key in preferred {
            if let Some(value) = options.get(key) {
                map.serialize_entry(key, value)?;
            }
        }
        for (key, value) in options {
            if !preferred.contains(&key.as_str()) {
                map.serialize_entry(key, value)?;
            }
        }
        map.end()
    }
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
}

impl Write for BoundedWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if data.len() > self.limit.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(io::Error::other("state serialization limit"));
        }
        self.bytes.extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn serialized(value: &impl Serialize, limit: usize, pretty: bool) -> Result<Vec<u8>> {
    let mut writer = BoundedWriter {
        bytes: Vec::new(),
        limit,
        exceeded: false,
    };
    let result = if pretty {
        serde_json::to_writer_pretty(&mut writer, value)
    } else {
        serde_json::to_writer(&mut writer, value)
    };
    if writer.exceeded {
        return Err(Error::Limit("serialized bytes"));
    }
    result.map_err(|_| invalid("state cannot be serialized"))?;
    Ok(writer.bytes)
}

pub(crate) fn encode(
    snapshot: &Snapshot,
    scope: &Scope,
    allow_symlinks: bool,
    limits: Limits,
) -> Result<Vec<u8>> {
    // One consistent filesystem observation for this pure validation.
    let _paths = super::paths::Scope::enter();
    check_scope(scope, allow_symlinks)?;
    count(snapshot, limits)?;
    validate_options(&snapshot.options, scope)?;
    if let Some(checkpoint) = &snapshot.checkpoint {
        validate_checkpoint(checkpoint, scope)?;
    }
    let mut documents = BTreeMap::new();
    for (key, entry) in &snapshot.documents {
        let normalized = file_key(key, scope, allow_symlinks)?;
        if documents
            .insert(
                normalized,
                wire_entry(
                    &ItemKey::File(key.clone()),
                    entry,
                    scope,
                    allow_symlinks,
                    snapshot.checkpoint.is_some(),
                )?,
            )
            .is_some()
        {
            return Err(invalid("document keys normalize to the same identity"));
        }
    }
    let mut urls = IndexMap::new();
    for (key, entry) in &snapshot.urls {
        urls.insert(
            key,
            wire_entry(
                &ItemKey::Url(key.clone()),
                entry,
                scope,
                allow_symlinks,
                snapshot.checkpoint.is_some(),
            )?,
        );
    }
    #[derive(Serialize)]
    struct WireSnapshot<'a> {
        version: &'static str,
        options: OrderedOptions<'a>,
        documents: BTreeMap<String, WireEntry<'a>>,
        urls: IndexMap<&'a String, WireEntry<'a>>,
        #[serde(rename = "_markitai", skip_serializing_if = "Option::is_none")]
        checkpoint: &'a Option<Checkpoint>,
    }
    serialized(
        &WireSnapshot {
            version: "1.0",
            options: OrderedOptions(&snapshot.options),
            documents,
            urls,
            checkpoint: &snapshot.checkpoint,
        },
        limits.base_bytes,
        true,
    )
}

#[cfg(test)]
pub(crate) fn decode_event(line: &[u8]) -> Result<Option<Event>> {
    decode_event_limited(line, Limits::default().line_bytes)
}

pub(crate) fn decode_event_limited(line: &[u8], limit: usize) -> Result<Option<Event>> {
    if line.len() > limit {
        return Err(Error::Limit("journal line"));
    }
    let value = parse(line)?;
    let object = object(&value, "journal event must be an object")?;
    let kind = match object.get("type") {
        Some(Value::String(kind)) if kind == "file" || kind == "url" => kind,
        None | Some(Value::String(_)) => return Ok(None),
        _ => return Err(invalid("journal event type must be a string")),
    };
    let key = object
        .get("key")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("journal event key must be a string"))?;
    let data = object
        .get("data")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    // Legacy replay ignores unknown keys before accessing their data. Leave
    // semantic validation to apply_event once an existing item is identified.
    let fence = object
        .get("_markitai")
        .map(|value| {
            serde_json::from_value::<Fence>(value.clone())
                .map_err(|_| invalid("invalid journal fence fields"))
        })
        .transpose()?;
    if let Some(fence) = &fence {
        generation(&fence.generation)?;
    }
    Ok(Some(Event {
        key: if kind == "file" {
            ItemKey::File(key.into())
        } else {
            ItemKey::Url(key.into())
        },
        data,
        fence,
    }))
}

#[cfg(test)]
pub(crate) fn encode_event(event: &Event) -> Result<Vec<u8>> {
    encode_event_limited(event, Limits::default().line_bytes)
}

pub(crate) fn encode_event_limited(event: &Event, limit: usize) -> Result<Vec<u8>> {
    object(&event.data, "journal event data must be an object")?;
    if let Some(fence) = &event.fence {
        generation(&fence.generation)?;
    }
    #[derive(Serialize)]
    struct WireEvent<'a> {
        #[serde(rename = "type")]
        kind: &'static str,
        key: &'a str,
        data: &'a Value,
        #[serde(rename = "_markitai", skip_serializing_if = "Option::is_none")]
        fence: &'a Option<Fence>,
    }
    let (kind, key) = match &event.key {
        ItemKey::File(key) => ("file", key),
        ItemKey::Url(key) => ("url", key),
    };
    serialized(
        &WireEvent {
            kind,
            key,
            data: &event.data,
            fence: &event.fence,
        },
        limit,
        false,
    )
}

fn updated_entry(
    snapshot: &Snapshot,
    event: &Event,
    scope: &Scope,
    allow_symlinks: bool,
) -> Result<Option<Entry>> {
    match (&snapshot.checkpoint, &event.fence) {
        (Some(checkpoint), Some(fence)) => {
            validate_checkpoint(checkpoint, scope)?;
            generation(&fence.generation)?;
            if checkpoint.generation != fence.generation
                || fence.sequence <= checkpoint.applied_sequence
            {
                return Ok(None);
            }
        }
        (Some(_), None) => return Ok(None),
        (None, Some(_)) => {
            return Err(Error::Sequence(
                "native event cannot be applied to an untagged base".into(),
            ));
        }
        (None, None) => (),
    }
    let original = match &event.key {
        ItemKey::File(key) => snapshot.documents.get(key),
        ItemKey::Url(key) => snapshot.urls.get(key),
    };
    let Some(original) = original else {
        return Ok(None);
    };
    let data = object(&event.data, "journal event data must be an object")?;
    let mut entry = original.clone();
    entry.status = status(data.get("status"))?;
    for (field, destination) in [("output", &mut entry.output), ("target", &mut entry.target)] {
        if data.contains_key(field) {
            *destination = optional_text(data.get(field), field)?.map(PathBuf::from);
        }
    }
    if data.contains_key("error") {
        entry.error = optional_text(data.get("error"), "error")?;
    }
    if snapshot.checkpoint.is_some() && data.contains_key("diagnostics") {
        match attempt_diagnostics(data.get("diagnostics"))? {
            Some(diagnostics) => {
                entry.observations.insert(
                    "diagnostics".into(),
                    serde_json::to_value(diagnostics)
                        .map_err(|_| invalid("invalid native attempt diagnostics"))?,
                );
            }
            None => {
                entry.observations.remove("diagnostics");
            }
        }
    }
    // Reference replay updates only status/output/error/target. Validate explicit
    // immutable provenance without adopting it from an event.
    if let ItemKey::Url(key) = &event.key {
        if let Some(url) = optional_text(data.get("url"), "url")? {
            url_identity(key, Some(&url))?;
        }
        if let Some(source) = optional_text(data.get("source_file"), "source_file")? {
            let probe = Entry {
                source_file: Some(source.clone()),
                ..entry.clone()
            };
            entry_parent(&event.key, &probe, scope, allow_symlinks)?;
            if let Some(old) = entry.source_file.as_deref().filter(|old| !old.is_empty())
                && super::paths::resolve(Path::new(old))?
                    != super::paths::resolve(Path::new(&source))?
            {
                return Err(foreign("journal event changes URL source provenance"));
            }
        }
    }
    check_scope(scope, allow_symlinks)?;
    validate_entry(&event.key, &mut entry, scope, allow_symlinks)?;
    Ok(Some(entry))
}

pub(crate) fn prepare_event(
    snapshot: &Snapshot,
    mut event: Event,
    scope: &Scope,
    allow_symlinks: bool,
) -> Result<Event> {
    // One consistent filesystem observation for this pure validation.
    let _paths = super::paths::Scope::enter();
    let entry = updated_entry(snapshot, &event, scope, allow_symlinks)?
        .ok_or_else(|| invalid("journal mutation is stale or has no known item"))?;
    let data = event
        .data
        .as_object_mut()
        .ok_or_else(|| invalid("journal event data must be an object"))?;
    // Move the supplied event instead of cloning an arbitrary data payload.
    // Only explicit paths are changed; absent fields and clear operations keep
    // their distinct replay meanings.
    for (field, path) in [("output", entry.output), ("target", entry.target)] {
        if data.contains_key(field)
            && let Some(path) = path
        {
            let text = path
                .into_os_string()
                .into_string()
                .map_err(|_| invalid("journal destination is not UTF-8"))?;
            data.insert(field.into(), Value::String(text));
        }
    }
    if matches!(event.key, ItemKey::Url(_))
        && let Some(source) = data
            .get("source_file")
            .and_then(Value::as_str)
            .filter(|source| !source.is_empty())
    {
        let source = Path::new(source);
        let spelling = lexical_absolute(source)?;
        if super::paths::resolve(&spelling)? != super::paths::resolve(source)? {
            return Err(foreign(
                "lexical source path changes the resolved symlink target",
            ));
        }
        let source = spelling
            .into_os_string()
            .into_string()
            .map_err(|_| invalid("journal URL source is not UTF-8"))?;
        data.insert("source_file".into(), Value::String(source));
    }
    Ok(event)
}

pub(crate) fn prepare_checkpoint(
    mut snapshot: Snapshot,
    scope: &Scope,
    allow_symlinks: bool,
) -> Result<Snapshot> {
    // One consistent filesystem observation for this pure validation.
    let _paths = super::paths::Scope::enter();
    check_scope(scope, allow_symlinks)?;
    validate_options(&snapshot.options, scope)?;
    if let Some(checkpoint) = &snapshot.checkpoint {
        validate_checkpoint(checkpoint, scope)?;
    }
    for (key, entry) in &mut snapshot.documents {
        validate_entry(&ItemKey::File(key.clone()), entry, scope, allow_symlinks)?;
    }
    for (key, entry) in &mut snapshot.urls {
        validate_entry(&ItemKey::Url(key.clone()), entry, scope, allow_symlinks)?;
        if let Some(source) = entry
            .source_file
            .as_deref()
            .filter(|source| !source.is_empty())
        {
            let source = Path::new(source);
            let resolved = super::paths::resolve(source)?;
            let spelling = lexical_absolute(source)?;
            if super::paths::resolve(&spelling)? != resolved {
                return Err(foreign(
                    "lexical source path changes the resolved symlink target",
                ));
            }
            entry.source_file = Some(
                spelling
                    .into_os_string()
                    .into_string()
                    .map_err(|_| invalid("checkpoint URL source is not UTF-8"))?,
            );
        }
    }
    // Legacy round trips preserve their original fields. Only the native begin
    // path calls this preparation, so a saved checkpoint no longer depends on
    // the working directory in which it was first opened.
    for field in ["input_dir", "output_dir"] {
        if let Some(text) = optional_text(snapshot.options.get(field), field)? {
            let resolved = super::paths::resolve(Path::new(&text))?;
            let text = resolved
                .into_os_string()
                .into_string()
                .map_err(|_| invalid("checkpoint scope option is not UTF-8"))?;
            snapshot.options.insert(field.into(), Value::String(text));
        }
    }
    Ok(snapshot)
}

pub(crate) fn apply_event(
    snapshot: &mut Snapshot,
    event: &Event,
    scope: &Scope,
    allow_symlinks: bool,
) -> Result<bool> {
    // One consistent filesystem observation for this pure validation.
    let _paths = super::paths::Scope::enter();
    let Some(entry) = updated_entry(snapshot, event, scope, allow_symlinks)? else {
        return Ok(false);
    };
    match &event.key {
        ItemKey::File(key) => {
            snapshot.documents.insert(key.clone(), entry);
        }
        ItemKey::Url(key) => {
            snapshot.urls.insert(key.clone(), entry);
        }
    }
    if let (Some(checkpoint), Some(fence)) = (&mut snapshot.checkpoint, &event.fence) {
        checkpoint.applied_sequence = fence.sequence;
    }
    Ok(true)
}

pub(crate) fn normalize_interrupted(snapshot: &mut Snapshot) {
    for entry in snapshot
        .documents
        .values_mut()
        .chain(snapshot.urls.values_mut())
    {
        if entry.status == Status::InProgress {
            entry.status = Status::Failed;
        }
    }
}

pub(crate) fn task_hash(scope: &Scope, options: &Value) -> Result<String> {
    // One consistent filesystem observation for this pure validation.
    let _paths = super::paths::Scope::enter();
    let options = object(options, "task options must be an object")?;
    let directory_keys = [
        "llm",
        "ocr",
        "screenshot",
        "alt",
        "desc",
        "llm_enabled",
        "ocr_enabled",
        "screenshot_enabled",
        "image_alt_enabled",
        "image_desc_enabled",
        "scan_max_depth",
        "glob_patterns",
    ];
    let keys = if scope.mode == Mode::Directory {
        &directory_keys[..]
    } else {
        &directory_keys[..5]
    };
    let selected = options
        .iter()
        .filter(|(key, _)| keys.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    crate::report_store::task_hash(&scope.input, &scope.output, &Value::Object(selected))
        .map_err(Error::from)
}

pub(crate) fn merge(
    snapshot: &mut Snapshot,
    discovered: &Snapshot,
    url_order: &[String],
    limits: Limits,
) -> Result<()> {
    // One consistent filesystem observation for this pure validation.
    let _paths = super::paths::Scope::enter();
    count(snapshot, limits)?;
    count(discovered, limits)?;
    let wanted: BTreeSet<_> = url_order.iter().collect();
    if url_order.len() != discovered.urls.len()
        || wanted.len() != url_order.len()
        || !wanted.iter().all(|key| discovered.urls.contains_key(*key))
    {
        return Err(invalid(
            "URL discovery order must cover every discovered key exactly once",
        ));
    }
    if let Some(new) = &discovered.checkpoint
        && snapshot.checkpoint.as_ref() != Some(new)
    {
        return Err(foreign(
            "discovered snapshot has a different native checkpoint",
        ));
    }
    let mut merged = snapshot.clone();
    for (key, entry) in &discovered.documents {
        if !merged.documents.contains_key(key) {
            if merged.documents.len().saturating_add(merged.urls.len()) >= limits.entries {
                return Err(Error::Limit("entry count"));
            }
            merged.documents.insert(key.clone(), entry.clone());
        }
    }
    for key in url_order {
        let discovered_entry = &discovered.urls[key];
        let bare = url_identity(key, discovered_entry.url.as_deref())?;
        if merged.urls.contains_key(key) {
            continue;
        }
        let adopted = if key != &bare && !discovered.urls.contains_key(&bare) {
            merged.urls.shift_remove(&bare)
        } else {
            None
        };
        let mut entry = if let Some(entry) = adopted {
            entry
        } else {
            if merged.documents.len().saturating_add(merged.urls.len()) >= limits.entries {
                return Err(Error::Limit("entry count"));
            }
            discovered_entry.clone()
        };
        entry.url = Some(bare);
        merged.urls.insert(key.clone(), entry);
    }
    *snapshot = merged;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const GENERATION: &str = "77e74191-48ba-4c70-bd28-e65fbd33b2a4";

    fn fixture(mode: Mode) -> (tempfile::TempDir, Scope) {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input");
        std::fs::create_dir(&input).unwrap();
        let input = if mode == Mode::UrlList {
            input.join("pages.urls")
        } else {
            input
        };
        let scope = Scope::new(mode, &input, &root.path().join("out")).unwrap();
        (root, scope)
    }

    fn decode_value(value: Value, scope: &Scope) -> Result<Snapshot> {
        decode(
            &serde_json::to_vec(&value).unwrap(),
            scope,
            false,
            Limits::default(),
        )
    }

    fn document_snapshot(scope: &Scope) -> Snapshot {
        decode_value(json!({"version":"1.0", "documents":{"note.txt":{"status":"in_progress","target":scope.output.join("note.txt.md")}}}), scope).unwrap()
    }

    fn event(key: &str, data: Value, sequence: Option<u64>) -> Event {
        Event {
            key: ItemKey::File(key.into()),
            data,
            fence: sequence.map(|sequence| Fence {
                generation: GENERATION.into(),
                sequence,
            }),
        }
    }

    fn tag(snapshot: &mut Snapshot, scope: &Scope, sequence: u64) {
        snapshot.checkpoint = Some(Checkpoint {
            generation: GENERATION.into(),
            applied_sequence: sequence,
            scope: scope.clone(),
        });
    }

    fn paid_diagnostics(status: &str, requests: u64) -> Value {
        json!({"last_attempt":{"operation":"convert","status":status,
            "error":if status=="error" { Some("authored error") } else { None },
            "usage":{"requests":requests,"input_tokens":0,"output_tokens":0,"cost_usd":0.0,
                "by_model":{"fixture":{"requests":requests,"input_tokens":0,"output_tokens":0,"cost_usd":0.0}}}}})
    }

    #[test]
    fn native_diagnostics_round_trip_replace_and_clear_without_double_accounting() {
        let (_root, scope) = fixture(Mode::Directory);
        let mut snapshot = document_snapshot(&scope);
        tag(&mut snapshot, &scope, 0);
        let failed = paid_diagnostics("error", 3);
        apply_event(
            &mut snapshot,
            &event(
                "note.txt",
                json!({"status":"failed","error":"authored error","diagnostics":failed}),
                Some(1),
            ),
            &scope,
            false,
        )
        .unwrap();
        let encoded = encode(&snapshot, &scope, false, Limits::default()).unwrap();
        let mut recovered = decode(&encoded, &scope, false, Limits::default()).unwrap();
        assert_eq!(
            recovered.documents["note.txt"].observations["diagnostics"],
            failed
        );
        apply_event(
            &mut recovered,
            &event(
                "note.txt",
                json!({"status":"in_progress","diagnostics":null}),
                Some(2),
            ),
            &scope,
            false,
        )
        .unwrap();
        assert!(
            !recovered.documents["note.txt"]
                .observations
                .contains_key("diagnostics")
        );
        // A prepared job that never dispatches may restore its saved observation.
        apply_event(
            &mut recovered,
            &event(
                "note.txt",
                json!({"status":"failed","diagnostics":failed}),
                Some(3),
            ),
            &scope,
            false,
        )
        .unwrap();
        assert_eq!(
            recovered.documents["note.txt"].observations["diagnostics"],
            failed
        );
        let done = paid_diagnostics("done", 1);
        apply_event(&mut recovered, &event("note.txt", json!({"status":"completed","error":null,"output":scope.output.join("note.txt.md"),"diagnostics":done}), Some(4)), &scope, false).unwrap();
        let compacted = encode(&recovered, &scope, false, Limits::default()).unwrap();
        let again = decode(&compacted, &scope, false, Limits::default()).unwrap();
        assert_eq!(
            again.documents["note.txt"].observations["diagnostics"],
            done
        );
        assert_eq!(again.checkpoint.unwrap().applied_sequence, 4);
    }

    #[test]
    fn malformed_native_diagnostics_fail_before_mutation_and_payload_is_not_echoed() {
        let (_root, scope) = fixture(Mode::Directory);
        let mut snapshot = document_snapshot(&scope);
        tag(&mut snapshot, &scope, 0);
        let before = snapshot.clone();
        let malformed = json!({"last_attempt":{"operation":"secret-provider-token"}});
        let error = apply_event(
            &mut snapshot,
            &event(
                "note.txt",
                json!({"status":"failed","diagnostics":malformed}),
                Some(1),
            ),
            &scope,
            false,
        )
        .unwrap_err();
        assert_eq!(snapshot, before);
        assert!(!error.to_string().contains("secret-provider"));
        let mut value: Value =
            serde_json::from_slice(&encode(&snapshot, &scope, false, Limits::default()).unwrap())
                .unwrap();
        value["documents"]["note.txt"]["diagnostics"] = malformed;
        assert!(decode_value(value, &scope).is_err());
        snapshot
            .documents
            .get_mut("note.txt")
            .unwrap()
            .observations
            .insert(
                "diagnostics".into(),
                json!({"last_attempt":{"usage":{"requests":-1}}}),
            );
        assert!(encode(&snapshot, &scope, false, Limits::default()).is_err());
    }

    #[test]
    fn legacy_unknown_diagnostics_keep_the_old_minimal_contract() {
        let (_root, scope) = fixture(Mode::Directory);
        let mut snapshot = decode_value(json!({"version":"1.0","documents":{"note.txt":{"status":"failed","error":"old error","diagnostics":"opaque old extension"}}}), &scope).unwrap();
        assert!(
            !snapshot.documents["note.txt"]
                .observations
                .contains_key("diagnostics")
        );
        apply_event(
            &mut snapshot,
            &event(
                "note.txt",
                json!({"status":"failed","diagnostics":{"unknown":"legacy"}}),
                None,
            ),
            &scope,
            false,
        )
        .unwrap();
        let value: Value =
            serde_json::from_slice(&encode(&snapshot, &scope, false, Limits::default()).unwrap())
                .unwrap();
        assert_eq!(
            value["documents"]["note.txt"],
            json!({"status":"failed","error":"old error"})
        );
    }

    #[test]
    fn detailed_legacy_retains_observations_but_serializes_only_resume_fields() {
        let (_root, scope) = fixture(Mode::Directory);
        let mut snapshot = decode_value(json!({
            "version":"1.0", "options":{"input_dir":scope.input,"output_dir":scope.output},
            "documents":{
                "note.txt":{"status":"completed","output":scope.output.join("note.txt.llm.md"),"target":scope.output.join("note.txt.md"),"duration":1.25,"images":4,"llm_usage":{"m":{"input_tokens":7}}},
                "wait.txt":{"status":"in_progress","target":scope.output.join("wait.txt.md")},
                "fail.txt":{"status":"failed","error":"conversion failed","target":scope.output.join("fail.txt.md")}
            }
        }), &scope).unwrap();
        assert_eq!(snapshot.documents["wait.txt"].status, Status::InProgress);
        assert_eq!(snapshot.documents["note.txt"].observations["images"], 4);
        assert!(
            !snapshot.documents["wait.txt"]
                .observations
                .contains_key("images")
        );
        let value: Value =
            serde_json::from_slice(&encode(&snapshot, &scope, false, Limits::default()).unwrap())
                .unwrap();
        assert_eq!(
            value["documents"]["note.txt"],
            json!({"status":"completed","output":scope.output.join("note.txt.llm.md")})
        );
        assert_eq!(
            value["documents"]["fail.txt"],
            json!({"status":"failed","error":"conversion failed","target":scope.output.join("fail.txt.md")})
        );
        normalize_interrupted(&mut snapshot);
        assert_eq!(snapshot.documents["wait.txt"].status, Status::Failed);
        assert_eq!(
            snapshot.documents["wait.txt"].target,
            Some(scope.output.join("wait.txt.md"))
        );
    }

    #[test]
    fn url_and_unknown_option_order_survive_minimal_round_trip() {
        let (_root, scope) = fixture(Mode::UrlList);
        let bytes = br#"{"version":"1.0","options":{"z-extra":1,"llm":false,"a-extra":2},"urls":{"https://example.test/z name":{"status":"pending","url":"https://example.test/z","source_file":null},"https://example.test/a":{"status":"pending"}}}"#;
        let snapshot = decode(bytes, &scope, false, Limits::default()).unwrap();
        assert_eq!(
            snapshot.urls.keys().map(String::as_str).collect::<Vec<_>>(),
            ["https://example.test/z name", "https://example.test/a"]
        );
        assert_eq!(
            snapshot.urls["https://example.test/z name"].source_file,
            None
        );
        assert_eq!(
            snapshot.urls["https://example.test/a"]
                .source_file
                .as_deref(),
            Some("")
        );
        let encoded = encode(&snapshot, &scope, false, Limits::default()).unwrap();
        let text = std::str::from_utf8(&encoded).unwrap();
        assert!(text.find("\"llm\"").unwrap() < text.find("\"z-extra\"").unwrap());
        assert!(text.find("\"z-extra\"").unwrap() < text.find("\"a-extra\"").unwrap());
        assert!(
            text.find("https://example.test/z name").unwrap()
                < text.find("https://example.test/a").unwrap()
        );
        let value: Value = serde_json::from_slice(&encoded).unwrap();
        assert!(value["urls"]["https://example.test/z name"]["source_file"].is_null());
        assert_eq!(value["urls"]["https://example.test/a"]["source_file"], "");
        assert_eq!(
            value["urls"]["https://example.test/z name"]["url"],
            "https://example.test/z"
        );
        assert!(value["urls"]["https://example.test/a"].get("url").is_none());
    }

    #[test]
    fn presence_only_replay_and_null_clearing_do_not_create_measurements() {
        let (_root, scope) = fixture(Mode::Directory);
        let mut snapshot = document_snapshot(&scope);
        let target = snapshot.documents["note.txt"].target.clone();
        assert!(
            apply_event(
                &mut snapshot,
                &event("note.txt", json!({"error":"retry"}), None),
                &scope,
                false
            )
            .unwrap()
        );
        assert_eq!(snapshot.documents["note.txt"].status, Status::Pending);
        assert_eq!(snapshot.documents["note.txt"].target, target);
        assert_eq!(
            snapshot.documents["note.txt"].error.as_deref(),
            Some("retry")
        );
        assert!(apply_event(&mut snapshot, &event("note.txt", json!({"status":"completed","output":scope.output.join("note.txt.md"),"error":null,"target":null,"duration":9}), None), &scope, false).unwrap());
        let entry = &snapshot.documents["note.txt"];
        assert_eq!(entry.status, Status::Completed);
        assert!(entry.target.is_none() && entry.error.is_none() && entry.observations.is_empty());
        normalize_interrupted(&mut snapshot);
        assert_eq!(snapshot.documents["note.txt"].status, Status::Completed);
    }

    #[test]
    fn invalid_event_is_atomic_and_errors_do_not_echo_payload() {
        let (_root, scope) = fixture(Mode::Directory);
        let mut snapshot = document_snapshot(&scope);
        tag(&mut snapshot, &scope, 7);
        let before = snapshot.clone();
        let error = apply_event(&mut snapshot, &event("note.txt", json!({"status":"completed","output":scope.input.join("secret-token.md"),"error":"provider-secret"}), Some(8)), &scope, false).unwrap_err();
        assert!(matches!(error, Error::ForeignScope(_)));
        assert!(!error.to_string().contains("secret"));
        assert_eq!(snapshot, before);
        assert!(
            apply_event(
                &mut snapshot,
                &event("note.txt", json!({"status":"unrecognized-secret"}), Some(8)),
                &scope,
                false
            )
            .is_err()
        );
        assert_eq!(snapshot, before);
    }

    #[test]
    fn replay_fence_prevents_old_log_regression_and_legacy_generation_mixing() {
        let (_root, scope) = fixture(Mode::Directory);
        let mut snapshot = document_snapshot(&scope);
        snapshot.documents.get_mut("note.txt").unwrap().status = Status::Completed;
        tag(&mut snapshot, &scope, 5);
        let before = snapshot.clone();
        let stale = event("note.txt", json!({"status":"in_progress"}), Some(5));
        assert!(!apply_event(&mut snapshot, &stale, &scope, false).unwrap());
        let mut other = event("note.txt", json!({"status":"in_progress"}), Some(6));
        other.fence.as_mut().unwrap().generation = uuid::Uuid::new_v4().to_string();
        assert!(!apply_event(&mut snapshot, &other, &scope, false).unwrap());
        assert!(
            !apply_event(
                &mut snapshot,
                &event("note.txt", json!({"status":"in_progress"}), None),
                &scope,
                false
            )
            .unwrap()
        );
        assert_eq!(snapshot, before);
        assert!(
            !apply_event(
                &mut snapshot,
                &event("unknown.txt", json!({"status":"pending"}), Some(6)),
                &scope,
                false
            )
            .unwrap()
        );
        assert_eq!(snapshot.checkpoint.as_ref().unwrap().applied_sequence, 5);
        assert!(
            apply_event(
                &mut snapshot,
                &event("note.txt", json!({"status":"failed"}), Some(6)),
                &scope,
                false
            )
            .unwrap()
        );
        assert_eq!(snapshot.checkpoint.as_ref().unwrap().applied_sequence, 6);
        let mut legacy = document_snapshot(&scope);
        assert!(matches!(
            apply_event(&mut legacy, &stale, &scope, false),
            Err(Error::Sequence(_))
        ));
    }

    #[test]
    fn event_codec_separates_syntax_semantics_and_unknown_types() {
        assert!(matches!(decode_event(b"{bad"), Err(Error::JsonSyntax(_))));
        assert!(matches!(decode_event(b"[]"), Err(Error::Invalid(_))));
        assert!(matches!(decode_event(b"\xff"), Err(Error::Invalid(_))));
        assert!(
            decode_event(br#"{"type":"future","data":7}"#)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            decode_event(br#"{"type":"file","key":"note.txt","data":7}"#)
                .unwrap()
                .unwrap()
                .data,
            json!(7)
        );
        assert!(decode_event(br#"{"type":"file","key":"note.txt","_markitai":{"generation":"secret","sequence":1}}"#).is_err());
        let sample = event(
            "note.txt",
            json!({"status":"pending","output":null}),
            Some(1),
        );
        let bytes = encode_event(&sample).unwrap();
        assert!(!bytes.ends_with(b"\n"));
        assert_eq!(decode_event(&bytes).unwrap(), Some(sample));
        let large = event("note.txt", json!({"error":"x".repeat(4096)}), Some(2));
        assert!(matches!(
            encode_event_limited(&large, 80),
            Err(Error::Limit(_))
        ));
    }

    #[test]
    fn legacy_unknown_keys_ignore_arbitrary_data_before_semantic_validation() {
        let (_root, scope) = fixture(Mode::Directory);
        let mut snapshot = document_snapshot(&scope);
        let before = snapshot.clone();
        for data in [
            json!(7),
            Value::Null,
            json!([]),
            json!("opaque"),
            json!(false),
        ] {
            for (kind, key) in [
                ("file", "missing.txt"),
                ("url", "https://example.test/missing"),
            ] {
                let bytes =
                    serde_json::to_vec(&json!({"type":kind,"key":key,"data":data})).unwrap();
                let decoded = decode_event(&bytes).unwrap().unwrap();
                assert!(!apply_event(&mut snapshot, &decoded, &scope, false).unwrap());
                assert_eq!(snapshot, before);
            }
            let bytes =
                serde_json::to_vec(&json!({"type":"file","key":"note.txt","data":data})).unwrap();
            let decoded = decode_event(&bytes).unwrap().unwrap();
            assert!(matches!(
                apply_event(&mut snapshot, &decoded, &scope, false),
                Err(Error::Invalid(_))
            ));
            assert_eq!(snapshot, before);
            assert!(matches!(encode_event(&decoded), Err(Error::Invalid(_))));
        }
        let completion = event(
            "note.txt",
            json!({"status":"completed","output":scope.output.join("note.txt.md")}),
            None,
        );
        assert!(apply_event(&mut snapshot, &completion, &scope, false).unwrap());
        assert_eq!(snapshot.documents["note.txt"].status, Status::Completed);
    }

    #[test]
    fn prepared_journal_paths_are_absolute_without_losing_presence_or_mutating_state() {
        let cwd = std::env::current_dir().unwrap();
        let root = tempfile::tempdir_in(&cwd).unwrap();
        let scope = Scope::new(
            Mode::Directory,
            &root.path().join("input"),
            &root.path().join("out"),
        )
        .unwrap();
        let mut snapshot = document_snapshot(&scope);
        tag(&mut snapshot, &scope, 4);
        let before = snapshot.clone();
        let relative = root
            .path()
            .join("out/sub/../note.txt.md")
            .strip_prefix(&cwd)
            .unwrap()
            .to_owned();
        let raw = event(
            "note.txt",
            json!({
                "status":"completed", "output":relative, "target":null,
                "extension":{"opaque":true}
            }),
            Some(5),
        );
        let prepared = prepare_event(&snapshot, raw, &scope, false).unwrap();
        assert_eq!(snapshot, before);
        assert_eq!(prepared.key, ItemKey::File("note.txt".into()));
        assert_eq!(
            prepared.fence,
            Some(Fence {
                generation: GENERATION.into(),
                sequence: 5
            })
        );
        assert_eq!(
            prepared.data,
            json!({
                "status":"completed", "output":scope.output.join("note.txt.md"),
                "target":null, "extension":{"opaque":true}
            })
        );
        assert!(matches!(
            encode_event_limited(&prepared, 16),
            Err(Error::Limit(_))
        ));
        assert_eq!(snapshot, before);
        let encoded = encode_event(&prepared).unwrap();
        let replay = decode_event(&encoded).unwrap().unwrap();
        assert!(Path::new(replay.data["output"].as_str().unwrap()).is_absolute());
        assert!(apply_event(&mut snapshot, &replay, &scope, false).unwrap());
        assert_eq!(
            snapshot.documents["note.txt"].output,
            Some(scope.output.join("note.txt.md"))
        );
        assert!(snapshot.documents["note.txt"].target.is_none());

        let before = snapshot.clone();
        let relative = root
            .path()
            .join("out/note.txt.md")
            .strip_prefix(&cwd)
            .unwrap()
            .to_owned();
        let prepared = prepare_event(
            &snapshot,
            event("note.txt", json!({"target":relative,"error":null}), Some(6)),
            &scope,
            false,
        )
        .unwrap();
        assert_eq!(
            prepared.data,
            json!({"target":scope.output.join("note.txt.md"),"error":null})
        );
        assert_eq!(snapshot, before);
        assert!(matches!(
            prepare_event(
                &snapshot,
                event(
                    "note.txt",
                    json!({"status":"pending","output":scope.input.join("outside.md")}),
                    Some(6)
                ),
                &scope,
                false,
            ),
            Err(Error::ForeignScope(_))
        ));
        assert_eq!(snapshot, before);
        assert!(
            prepare_event(
                &snapshot,
                event("unknown.txt", json!({}), Some(6)),
                &scope,
                false
            )
            .is_err()
        );
        assert_eq!(snapshot, before);
    }

    #[test]
    fn configured_event_limit_is_shared_by_encoding_and_replay() {
        let (_root, scope) = fixture(Mode::Directory);
        let mut snapshot = document_snapshot(&scope);
        let length = Limits::default().line_bytes + 64;
        let limit = length + 1024;
        let mutation = event(
            "note.txt",
            json!({"status":"failed","error":"x".repeat(length)}),
            None,
        );
        assert!(matches!(encode_event(&mutation), Err(Error::Limit(_))));
        let encoded = encode_event_limited(&mutation, limit).unwrap();
        assert!(encoded.len() > Limits::default().line_bytes);
        assert!(matches!(decode_event(&encoded), Err(Error::Limit(_))));
        assert!(matches!(
            decode_event_limited(&encoded, encoded.len() - 1),
            Err(Error::Limit(_))
        ));
        let decoded = decode_event_limited(&encoded, limit).unwrap().unwrap();
        assert!(apply_event(&mut snapshot, &decoded, &scope, false).unwrap());
        assert_eq!(snapshot.documents["note.txt"].status, Status::Failed);
        assert_eq!(
            snapshot.documents["note.txt"].error.as_ref().unwrap().len(),
            length
        );
    }

    #[test]
    fn native_checkpoint_preparation_anchors_scope_and_url_provenance_only_at_begin() {
        let cwd = std::env::current_dir().unwrap();
        let root = tempfile::tempdir_in(&cwd).unwrap();
        let input = root.path().join("input/pages.urls");
        let output = root.path().join("out");
        let scope = Scope::new(Mode::UrlList, &input, &output).unwrap();
        let relative_input_dir = input.parent().unwrap().strip_prefix(&cwd).unwrap();
        let relative_output = output.strip_prefix(&cwd).unwrap();
        let relative_source = input.strip_prefix(&cwd).unwrap();
        let relative_document = relative_output.join("name.md");
        let mut snapshot = Snapshot::default();
        snapshot.options.insert("extra_z".into(), json!(1));
        snapshot
            .options
            .insert("input_dir".into(), json!(relative_input_dir));
        snapshot.options.insert("extra_a".into(), json!(2));
        snapshot
            .options
            .insert("output_dir".into(), json!(relative_output));
        snapshot.urls.insert(
            "https://example.test/p name".into(),
            Entry {
                status: Status::Completed,
                source_file: Some(relative_source.to_str().unwrap().into()),
                url: Some("https://example.test/p".into()),
                output: Some(relative_document.clone()),
                target: Some(relative_document),
                ..Entry::default()
            },
        );
        tag(&mut snapshot, &scope, 12);
        let legacy: Value =
            serde_json::from_slice(&encode(&snapshot, &scope, false, Limits::default()).unwrap())
                .unwrap();
        assert_eq!(legacy["options"]["input_dir"], json!(relative_input_dir));
        assert_eq!(
            legacy["urls"]["https://example.test/p name"]["source_file"],
            json!(relative_source)
        );
        let prepared = prepare_checkpoint(snapshot.clone(), &scope, false).unwrap();
        assert_eq!(prepared.checkpoint, snapshot.checkpoint);
        assert_eq!(
            prepared.options.keys().collect::<Vec<_>>(),
            snapshot.options.keys().collect::<Vec<_>>()
        );
        assert_eq!(
            prepared.options["input_dir"],
            json!(scope.input.parent().unwrap())
        );
        assert_eq!(prepared.options["output_dir"], json!(scope.output));
        let entry = &prepared.urls["https://example.test/p name"];
        assert_eq!(entry.source_file.as_deref(), input.to_str());
        assert_eq!(entry.output, Some(output.join("name.md")));
        assert_eq!(entry.target, Some(output.join("name.md")));
        assert_eq!(
            prepare_checkpoint(prepared.clone(), &scope, false).unwrap(),
            prepared
        );
        assert_eq!(snapshot.options["input_dir"], json!(relative_input_dir));
        let mutation = prepare_event(
            &prepared,
            Event {
                key: ItemKey::Url("https://example.test/p name".into()),
                data: json!({"status":"failed","source_file":relative_source,"error":"retry"}),
                fence: Some(Fence {
                    generation: GENERATION.into(),
                    sequence: 13,
                }),
            },
            &scope,
            false,
        )
        .unwrap();
        assert_eq!(mutation.data["source_file"], json!(input));
        let decoded = decode_event(&encode_event(&mutation).unwrap())
            .unwrap()
            .unwrap();
        let mut replayed = prepared.clone();
        assert!(apply_event(&mut replayed, &decoded, &scope, false).unwrap());
        assert_eq!(
            replayed.urls["https://example.test/p name"].status,
            Status::Failed
        );
        assert_eq!(
            replayed.urls["https://example.test/p name"]
                .source_file
                .as_deref(),
            input.to_str()
        );

        let mut absent = Snapshot::default();
        absent.options.insert("input_dir".into(), Value::Null);
        absent.urls.insert(
            "https://example.test/empty".into(),
            Entry {
                source_file: Some(String::new()),
                ..Entry::default()
            },
        );
        absent
            .urls
            .insert("https://example.test/null".into(), Entry::default());
        let unchanged = prepare_checkpoint(absent.clone(), &scope, false).unwrap();
        assert_eq!(unchanged, absent);
        assert!(!unchanged.options.contains_key("output_dir"));
        assert!(unchanged.options["input_dir"].is_null());
    }

    #[test]
    fn checkpoint_preparation_rejects_original_foreign_provenance_and_generation() {
        let (root, scope) = fixture(Mode::Directory);
        let mut snapshot = document_snapshot(&scope);
        snapshot
            .options
            .insert("input_dir".into(), json!(root.path().join("foreign")));
        assert!(matches!(
            prepare_checkpoint(snapshot, &scope, false),
            Err(Error::ForeignScope(_))
        ));
        let mut snapshot = document_snapshot(&scope);
        snapshot.urls.insert(
            "https://example.test/p".into(),
            Entry {
                source_file: Some(
                    root.path()
                        .join("foreign.urls")
                        .to_string_lossy()
                        .into_owned(),
                ),
                ..Entry::default()
            },
        );
        assert!(matches!(
            prepare_checkpoint(snapshot, &scope, false),
            Err(Error::ForeignScope(_))
        ));
        let mut snapshot = document_snapshot(&scope);
        tag(&mut snapshot, &scope, 3);
        snapshot.checkpoint.as_mut().unwrap().generation.clear();
        assert!(matches!(
            prepare_checkpoint(snapshot, &scope, false),
            Err(Error::Invalid(_))
        ));
        let mut snapshot = document_snapshot(&scope);
        tag(&mut snapshot, &scope, 3);
        snapshot.checkpoint.as_mut().unwrap().scope.output = root.path().join("foreign");
        assert!(matches!(
            prepare_checkpoint(snapshot, &scope, false),
            Err(Error::ForeignScope(_))
        ));
    }

    #[test]
    fn foreign_scope_and_unsafe_mirrored_destinations_are_distinct_errors() {
        let (root, scope) = fixture(Mode::Directory);
        for options in [
            json!({"input_dir":root.path().join("other")}),
            json!({"output_dir":root.path().join("other")}),
        ] {
            assert!(matches!(
                decode_value(json!({"options":options}), &scope),
                Err(Error::ForeignScope(_))
            ));
        }
        for (key, output) in [
            ("../note.txt", scope.output.join("note.md")),
            ("nested/note.txt", scope.output.join("note.md")),
            ("note.txt", scope.output.join(".markitai/state.md")),
            ("note.txt", root.path().join("outside.md")),
        ] {
            assert!(matches!(
                decode_value(
                    json!({"documents":{key:{"status":"completed","output":output}}}),
                    &scope
                ),
                Err(Error::ForeignScope(_))
            ));
        }
        let decoded = decode_value(json!({"documents":{scope.input.join("nested/note.txt").to_string_lossy():{"status":"completed","output":scope.output.join("nested/note.md")}}}), &scope).unwrap();
        assert!(decoded.documents.contains_key("nested/note.txt"));
    }

    #[test]
    fn relative_destinations_use_cwd_instead_of_the_output_directory() {
        let current = std::env::current_dir().unwrap();
        let root = tempfile::tempdir_in(&current).unwrap();
        let scope = Scope::new(
            Mode::Directory,
            &root.path().join("input"),
            &root.path().join("out"),
        )
        .unwrap();
        let relative = root
            .path()
            .join("out/note.md")
            .strip_prefix(&current)
            .unwrap()
            .to_owned();
        let snapshot = decode_value(
            json!({"documents":{"note.txt":{"status":"completed","output":relative}}}),
            &scope,
        )
        .unwrap();
        assert_eq!(
            snapshot.documents["note.txt"].output,
            Some(scope.output.join("note.md"))
        );
        assert!(matches!(
            decode_value(
                json!({"documents":{"note.txt":{"status":"completed","output":"note.md"}}}),
                &scope
            ),
            Err(Error::ForeignScope(_))
        ));
    }

    #[test]
    fn url_scope_checks_source_list_parent_and_named_identity() {
        let (_root, directory) = fixture(Mode::Directory);
        let source = directory.input.join("nested/list.urls");
        let base = json!({"urls":{"https://example.test/p name":{"status":"failed","url":"https://example.test/p","source_file":source,"target":directory.output.join("nested/name.md")}}});
        assert!(decode_value(base.clone(), &directory).is_ok());
        let mut wrong = base.clone();
        wrong["urls"]["https://example.test/p name"]["target"] =
            json!(directory.output.join("name.md"));
        assert!(matches!(
            decode_value(wrong, &directory),
            Err(Error::ForeignScope(_))
        ));
        let list = Scope::new(Mode::UrlList, &source, &directory.output).unwrap();
        let mut wrong = base.clone();
        wrong["urls"]["https://example.test/p name"]["source_file"] =
            json!(directory.input.join("other.urls"));
        assert!(matches!(
            decode_value(wrong, &list),
            Err(Error::ForeignScope(_))
        ));
        let mut wrong = base;
        wrong["urls"]["https://example.test/p name"]["url"] = json!("https://example.test/other");
        assert!(matches!(
            decode_value(wrong, &directory),
            Err(Error::ForeignScope(_))
        ));
    }

    #[test]
    fn malformed_native_scope_generation_and_version_are_rejected() {
        let (_root, scope) = fixture(Mode::Directory);
        let mut snapshot = document_snapshot(&scope);
        tag(&mut snapshot, &scope, 9);
        let bytes = encode(&snapshot, &scope, false, Limits::default()).unwrap();
        assert_eq!(
            decode(&bytes, &scope, false, Limits::default())
                .unwrap()
                .checkpoint,
            snapshot.checkpoint
        );
        let original: Value = serde_json::from_slice(&bytes).unwrap();
        let mut value = original.clone();
        value["_markitai"]["scope"]["mode"] = json!("url_list");
        assert!(matches!(
            decode_value(value, &scope),
            Err(Error::ForeignScope(_))
        ));
        let mut value = original.clone();
        value["_markitai"]["generation"] = json!("");
        assert!(matches!(
            decode_value(value, &scope),
            Err(Error::Invalid(_))
        ));
        let mut value = original;
        value["_markitai"]["applied_sequence"] = json!(-1);
        assert!(matches!(
            decode_value(value, &scope),
            Err(Error::Invalid(_))
        ));
        assert!(decode_value(json!({"version":"2.0"}), &scope).is_err());
        assert!(decode_value(json!({"_markitai":null}), &scope).is_err());
    }

    #[test]
    fn entry_and_byte_limits_apply_without_mutating_the_snapshot() {
        let (_root, scope) = fixture(Mode::Directory);
        let snapshot = document_snapshot(&scope);
        let bytes = encode(&snapshot, &scope, false, Limits::default()).unwrap();
        let mut limits = Limits {
            entries: 0,
            ..Limits::default()
        };
        assert!(matches!(
            decode(&bytes, &scope, false, limits),
            Err(Error::Limit(_))
        ));
        assert!(matches!(
            encode(&snapshot, &scope, false, limits),
            Err(Error::Limit(_))
        ));
        limits = Limits {
            base_bytes: bytes.len() - 1,
            ..Limits::default()
        };
        assert!(matches!(
            decode(&bytes, &scope, false, limits),
            Err(Error::Limit(_))
        ));
        assert!(matches!(
            encode(&snapshot, &scope, false, limits),
            Err(Error::Limit(_))
        ));
        let duplicate = br#"{"documents":{"note.txt":{},"note.txt":{}}}"#;
        assert!(matches!(
            decode(duplicate, &scope, false, Limits::default()),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn merge_preserves_history_and_adopts_bare_url_once_in_discovery_order() {
        let (_root, scope) = fixture(Mode::UrlList);
        let mut saved = decode_value(json!({"urls":{"https://example.test/p":{"status":"completed","source_file":scope.input,"output":scope.output.join("old.md")}}}), &scope).unwrap();
        let discovered = decode_value(
            json!({"urls":{
                "https://example.test/p a":{"source_file":scope.input},
                "https://example.test/p z":{"source_file":scope.input}
            }}),
            &scope,
        )
        .unwrap();
        let order: Vec<String> = vec![
            "https://example.test/p z".into(),
            "https://example.test/p a".into(),
        ];
        merge(&mut saved, &discovered, &order, Limits::default()).unwrap();
        assert_eq!(
            saved.urls.keys().collect::<Vec<_>>(),
            order.iter().collect::<Vec<_>>()
        );
        assert_eq!(saved.urls[&order[0]].status, Status::Completed);
        assert_eq!(saved.urls[&order[1]].status, Status::Pending);
        assert_eq!(
            saved.urls[&order[0]].output,
            Some(scope.output.join("old.md"))
        );
        assert!(!saved.urls.contains_key("https://example.test/p"));
        assert!(encode(&saved, &scope, false, Limits::default()).is_ok());
    }

    #[test]
    fn bare_claimant_blocks_adoption_and_invalid_merge_is_atomic() {
        let (_root, scope) = fixture(Mode::UrlList);
        let mut saved = decode_value(
            json!({"urls":{"https://example.test/p":{"status":"completed"}}}),
            &scope,
        )
        .unwrap();
        let discovered = decode_value(
            json!({"urls":{"https://example.test/p name":{},"https://example.test/p":{}}}),
            &scope,
        )
        .unwrap();
        let order: Vec<String> = vec![
            "https://example.test/p name".into(),
            "https://example.test/p".into(),
        ];
        let before = saved.clone();
        assert!(
            merge(
                &mut saved,
                &discovered,
                &[order[0].clone(), order[0].clone()],
                Limits::default()
            )
            .is_err()
        );
        assert_eq!(saved, before);
        assert!(matches!(
            merge(
                &mut saved,
                &discovered,
                &order,
                Limits {
                    entries: 1,
                    ..Limits::default()
                }
            ),
            Err(Error::Limit(_))
        ));
        assert_eq!(saved, before);
        merge(&mut saved, &discovered, &order, Limits::default()).unwrap();
        assert_eq!(
            saved.urls["https://example.test/p"].status,
            Status::Completed
        );
        assert_eq!(
            saved.urls["https://example.test/p name"].status,
            Status::Pending
        );
    }

    #[test]
    fn task_identity_uses_mode_specific_whitelist_and_distinguishes_absence() {
        let (_root, scope) = fixture(Mode::Directory);
        let opts = json!({"llm":false,"ocr":true,"llm_enabled":true,"scan_max_depth":8,"glob_patterns":["**/*.txt"],"secret":"not-hashed","models":["ignored"]});
        let expected = crate::report_store::task_hash(&scope.input, &scope.output, &json!({"llm":false,"ocr":true,"llm_enabled":true,"scan_max_depth":8,"glob_patterns":["**/*.txt"]})).unwrap();
        assert_eq!(task_hash(&scope, &opts).unwrap(), expected);
        assert_ne!(
            task_hash(&scope, &json!({})).unwrap(),
            task_hash(&scope, &json!({"llm":false})).unwrap()
        );
        let list =
            Scope::new(Mode::UrlList, &scope.input.join("中文.urls"), &scope.output).unwrap();
        let state = task_hash(&list, &opts).unwrap();
        assert_eq!(
            state,
            crate::report_store::task_hash(
                &list.input,
                &list.output,
                &json!({"llm":false,"ocr":true})
            )
            .unwrap()
        );
        assert_ne!(
            state,
            crate::report_store::task_hash(
                &list.output,
                &list.output,
                &json!({"llm":false,"ocr":true})
            )
            .unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_cannot_escape_or_hide_forbidden_output_spelling() {
        use std::os::unix::fs::symlink;
        let (root, scope) = fixture(Mode::Directory);
        std::fs::create_dir(&scope.output).unwrap();
        let outside = root.path().join("elsewhere");
        std::fs::create_dir(&outside).unwrap();
        symlink(&outside, scope.output.join("nested")).unwrap();
        assert!(matches!(
            decode_value(
                json!({"documents":{"nested/note.txt":{"target":scope.output.join("nested/note.md")}}}),
                &scope
            ),
            Err(Error::ForeignScope(_))
        ));
        let bytes = serde_json::to_vec(&json!({"documents":{"nested/note.txt":{"target":scope.output.join("nested/note.md")}}})).unwrap();
        assert!(matches!(
            decode(&bytes, &scope, true, Limits::default()),
            Err(Error::ForeignScope(_))
        ));
        let link = root.path().join("alias");
        symlink(&scope.output, &link).unwrap();
        let alias = Scope::new(Mode::Directory, &scope.input, &link).unwrap();
        assert!(matches!(
            decode_value(json!({}), &alias),
            Err(Error::ForeignScope(_))
        ));
        assert!(decode(b"{}", &alias, true, Limits::default()).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn lexical_output_alias_survives_and_symlink_parent_reinterpretation_fails() {
        use std::os::unix::fs::symlink;
        let (root, scope) = fixture(Mode::Directory);
        std::fs::create_dir(&scope.output).unwrap();
        let alias = root.path().join("output-alias");
        symlink(&scope.output, &alias).unwrap();
        let spelling = alias.join("sub/../note.md");
        let bytes = serde_json::to_vec(
            &json!({"documents":{"note.txt":{"status":"completed","output":spelling}}}),
        )
        .unwrap();
        let snapshot = decode(&bytes, &scope, true, Limits::default()).unwrap();
        assert_eq!(
            snapshot.documents["note.txt"].output,
            Some(alias.join("note.md"))
        );
        let encoded: Value =
            serde_json::from_slice(&encode(&snapshot, &scope, true, Limits::default()).unwrap())
                .unwrap();
        assert_eq!(
            encoded["documents"]["note.txt"]["output"],
            json!(alias.join("note.md"))
        );
        let nested = scope.output.join("nested");
        std::fs::create_dir(&nested).unwrap();
        symlink(&nested, scope.output.join("shortcut")).unwrap();
        // This ordinary in-root symlink has the same resolved and lexical parent.
        let okay = serde_json::to_vec(
            &json!({"documents":{"note.txt":{"target":scope.output.join("shortcut/../note.md")}}}),
        )
        .unwrap();
        assert!(decode(&okay, &scope, true, Limits::default()).is_ok());
        let deeper = nested.join("deeper");
        std::fs::create_dir(&deeper).unwrap();
        symlink(&deeper, scope.output.join("deep-shortcut")).unwrap();
        let bad = serde_json::to_vec(&json!({"documents":{"nested/note.txt":{"target":scope.output.join("deep-shortcut/../note.md")}}})).unwrap();
        assert!(matches!(
            decode(&bad, &scope, true, Limits::default()),
            Err(Error::ForeignScope(_))
        ));
    }
}
