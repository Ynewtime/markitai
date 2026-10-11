//! Persistent extracted pages. HTTP validation remains the fetcher's decision.

use crate::{Error, Result, config};
use globset::GlobBuilder;
use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior, params};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_CAPACITY: i64 = 512 * 1024 * 1024;
const MAX_CONTENT: usize = 100 * 1024 * 1024;
const MAX_METADATA: usize = 4 * 1024 * 1024;
const MAX_LABEL: usize = 64 * 1024;
const MAX_HEADER: usize = 16 * 1024;
const BUSY_TIMEOUT: Duration = Duration::from_secs(30);
const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS fetch_cache (
    key TEXT PRIMARY KEY, url TEXT NOT NULL, content TEXT NOT NULL,
    strategy_used TEXT NOT NULL, title TEXT, final_url TEXT, metadata TEXT,
    created_at INTEGER NOT NULL, accessed_at INTEGER NOT NULL, size_bytes INTEGER NOT NULL,
    etag TEXT, last_modified TEXT, screenshot_path TEXT, static_content TEXT, browser_content TEXT
); CREATE INDEX IF NOT EXISTS idx_fetch_accessed ON fetch_cache(accessed_at);
CREATE INDEX IF NOT EXISTS idx_fetch_url ON fetch_cache(url);";

#[derive(Clone, Debug, Default)]
pub(crate) struct Entry {
    pub content: String,
    pub metadata: Map<String, Value>,
    pub strategy_used: String,
    pub title: Option<String>,
    pub final_url: Option<String>,
    pub etag: Option<String>,
    pub last_modified: Option<String>,
    pub created_at: i64,
}

impl Entry {
    fn has_validators(&self) -> bool {
        [&self.etag, &self.last_modified]
            .iter()
            .any(|value| value.as_ref().is_some_and(|value| !value.is_empty()))
    }
}

pub(crate) struct Cache {
    path: PathBuf,
    capacity: i64,
    ttl: u64,
    skip_read: bool,
    patterns: Vec<String>,
}

fn unavailable() -> Error {
    Error::Conversion("Persistent URL fetch cache is unavailable".into())
}

/// How long an operation is tried again after transient failures, counted
/// from its first attempt. Lock waits have their own `BUSY_TIMEOUT` inside
/// each attempt, so an attempt that waited that long is not repeated.
const RETRY_WINDOW: Duration = Duration::from_secs(5);
const FIRST_RETRY_WAIT: Duration = Duration::from_millis(20);
const LONGEST_RETRY_WAIT: Duration = Duration::from_millis(500);

/// Why one attempt at the store failed. Only the operation's caller sees
/// `unavailable()`; the cause decides whether the attempt is repeated.
#[derive(Debug)]
struct Fault {
    transient: bool,
    /// What failed, for the tests' report; it can name paths and SQL, so
    /// the error callers get never carries it.
    #[cfg_attr(not(test), allow(dead_code))]
    detail: String,
}

type Attempt<T> = std::result::Result<T, Fault>;

impl Fault {
    /// A failure that another attempt cannot change: an invalid entry, a
    /// store that is not one, a link or directory in its place.
    fn permanent(detail: impl Into<String>) -> Self {
        Self {
            transient: false,
            detail: detail.into(),
        }
    }
}

impl From<rusqlite::Error> for Fault {
    fn from(error: rusqlite::Error) -> Self {
        Self {
            transient: transient_sqlite(&error),
            detail: error.to_string(),
        }
    }
}

impl From<std::io::Error> for Fault {
    fn from(error: std::io::Error) -> Self {
        Self {
            transient: transient_io(&error),
            detail: error.to_string(),
        }
    }
}

/// Lock contention SQLite did not wait out (some lock steps return busy
/// without calling the busy handler), a WAL lock protocol failure under
/// load, and I/O errors: SQLite reports a file another process holds for a
/// moment (Windows sharing and lock violations, a virus scanner opening a
/// new file, a side file being deleted by the last connection) as an I/O
/// error once its own short retries are spent, and on Windows as a file it
/// cannot open.
fn transient_sqlite(error: &rusqlite::Error) -> bool {
    use rusqlite::ErrorCode;
    let rusqlite::Error::SqliteFailure(error, _) = error else {
        return false;
    };
    matches!(
        error.code,
        ErrorCode::DatabaseBusy
            | ErrorCode::DatabaseLocked
            | ErrorCode::FileLockingProtocolFailed
            | ErrorCode::SystemIoFailure
    ) || (cfg!(windows) && error.code == ErrorCode::CannotOpen)
}

/// The file errors that pass: on Windows a file that is open without
/// sharing, locked, being deleted or held by a scanner (access denied); on
/// every system an interrupted or would-block call.
fn transient_io(error: &std::io::Error) -> bool {
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{
            ERROR_ACCESS_DENIED, ERROR_DELETE_PENDING, ERROR_LOCK_VIOLATION,
            ERROR_SHARING_VIOLATION,
        };
        if error.raw_os_error().is_some_and(|code| {
            [
                ERROR_ACCESS_DENIED,
                ERROR_SHARING_VIOLATION,
                ERROR_LOCK_VIOLATION,
                ERROR_DELETE_PENDING,
            ]
            .contains(&(code as u32))
        }) {
            return true;
        }
    }
    matches!(
        error.kind(),
        std::io::ErrorKind::Interrupted | std::io::ErrorKind::WouldBlock
    )
}

/// Run `attempt` until it succeeds, a failure is not transient, or the
/// retry window is spent, waiting twice as long after each transient
/// failure. Every attempt is one complete transaction (or read), so a
/// failed one changed nothing.
fn retrying<T>(mut attempt: impl FnMut() -> Attempt<T>) -> Result<T> {
    let deadline = Instant::now() + RETRY_WINDOW;
    let mut wait = FIRST_RETRY_WAIT;
    loop {
        match attempt() {
            Ok(value) => return Ok(value),
            Err(fault) if fault.transient && Instant::now() + wait < deadline => {
                std::thread::sleep(wait);
                wait = (wait * 2).min(LONGEST_RETRY_WAIT);
            }
            Err(_fault) => {
                #[cfg(test)]
                eprintln!(
                    "fetch cache unavailable ({}): {}",
                    if _fault.transient {
                        "transient, retries spent"
                    } else {
                        "permanent"
                    },
                    _fault.detail
                );
                return Err(unavailable());
            }
        }
    }
}

fn path(cfg: &Value) -> PathBuf {
    config::state_path(Path::new(
        cfg.pointer("/cache/global_dir")
            .and_then(Value::as_str)
            .unwrap_or("~/.markitai"),
    ))
    .join("fetch_cache.db")
}

fn capacity(cfg: &Value) -> i64 {
    let size = cfg.pointer("/cache/max_size_bytes");
    size.and_then(Value::as_i64)
        .or_else(|| {
            size.and_then(Value::as_u64)
                .map(|v| v.min(i64::MAX as u64) as i64)
        })
        .unwrap_or(DEFAULT_CAPACITY)
        .max(0)
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .min(i64::MAX as u64) as i64
}

/// The extraction a cached page belongs to. Rows hold extracted Markdown, so
/// a page is replayed only by the extraction that produced it: the package
/// version changes with every release, and the revision is bumped when
/// extraction changes between releases (decision 0003). Older rows stay
/// readable and expire by TTL or capacity.
const NAMESPACE: &str = concat!("native-fetch-", env!("CARGO_PKG_VERSION"), "-r5");

fn key(url: &str, explicit_strategy: Option<&str>) -> String {
    let mut hash = Sha256::new();
    hash.update(NAMESPACE.as_bytes());
    hash.update(b"\0");
    hash.update(url.as_bytes());
    if let Some(strategy) = explicit_strategy.filter(|value| *value != "auto") {
        hash.update(b"\0");
        hash.update(strategy.as_bytes());
    }
    crate::hex(hash.finalize())[..32].to_owned()
}

fn lower_authority(value: &str) -> String {
    let (scheme, separator, rest) = value
        .split_once("://")
        .map_or(("", "", value), |(scheme, rest)| (scheme, "://", rest));
    let (authority, tail) = rest.split_at(rest.find(['/', '?', '#']).unwrap_or(rest.len()));
    format!(
        "{}{separator}{}{tail}",
        scheme.to_lowercase(),
        authority.to_lowercase()
    )
}

// Python fnmatch treats slashes and braces literally, with wildcards spanning
// slashes. Escape globset's brace and backslash syntax rather than expanding it.
fn glob_pattern(value: &str) -> String {
    let chars: Vec<_> = value.chars().collect();
    let mut result = String::new();
    for (index, ch) in chars.iter().enumerate() {
        match ch {
            '{' | '}' | '\\' => {
                result.push('\\');
                result.push(*ch);
            }
            '[' => {
                let mut end = index + 1;
                if chars.get(end) == Some(&'!') {
                    end += 1;
                }
                if chars.get(end) == Some(&']') {
                    end += 1;
                }
                if chars[end..].contains(&']') {
                    result.push('[');
                } else {
                    result.push_str("\\[");
                }
            }
            _ => result.push(*ch),
        }
    }
    result
}

fn matches(candidate: &str, pattern: &str) -> bool {
    !candidate.is_empty()
        && GlobBuilder::new(&glob_pattern(pattern))
            .literal_separator(false)
            .backslash_escape(true)
            .case_insensitive(false)
            .build()
            .is_ok_and(|glob| glob.compile_matcher().is_match(candidate))
}

pub(crate) fn url_matches_patterns(url: &str, patterns: &[String]) -> bool {
    let normalized = lower_authority(url);
    let without_scheme = normalized
        .split_once("://")
        .map_or(normalized.as_str(), |(_, v)| v);
    let rest = url.split_once("://").map_or(url, |(_, v)| v);
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    let host_port = authority.rsplit('@').next().unwrap_or("");
    let hostname = if let Some(ipv6) = host_port.strip_prefix('[') {
        ipv6.split(']').next().unwrap_or("")
    } else {
        host_port.split(':').next().unwrap_or("")
    };
    let path = rest[authority_end..].split(['?', '#']).next().unwrap_or("");
    let last = path.trim_end_matches('/').rsplit('/').next().unwrap_or("");
    let hosts = [authority.to_lowercase(), hostname.to_lowercase()];
    patterns
        .iter()
        .map(|pattern| pattern.trim())
        .filter(|pattern| !pattern.is_empty())
        .any(|pattern| {
            [Some(pattern), pattern.strip_prefix("**/")]
                .into_iter()
                .flatten()
                .any(|variant| {
                    let folded = lower_authority(variant);
                    matches(&normalized, &folded)
                        || matches(without_scheme, &folded)
                        || hosts
                            .iter()
                            .any(|host| matches(host, &variant.to_lowercase()))
                        || matches(last, variant)
                })
        })
}

impl Cache {
    pub(crate) fn from_config(cfg: &Value) -> Option<Self> {
        if cfg.pointer("/cache/enabled").and_then(Value::as_bool) == Some(false) {
            return None;
        }
        Some(Self {
            path: path(cfg),
            capacity: capacity(cfg),
            ttl: cfg
                .pointer("/cache/fetch_ttl_seconds")
                .and_then(Value::as_u64)
                .unwrap_or(86_400),
            skip_read: cfg.pointer("/cache/no_cache").and_then(Value::as_bool) == Some(true),
            patterns: cfg
                .pointer("/cache/no_cache_patterns")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect(),
        })
    }

    pub(crate) fn should_read(&self, url: &str) -> bool {
        !self.skip_read && !url_matches_patterns(url, &self.patterns)
    }

    pub(crate) fn get(&self, url: &str, scope: Option<&str>) -> Result<Option<Entry>> {
        self.get_at(url, scope, now())
    }

    fn get_at(&self, url: &str, scope: Option<&str>, timestamp: i64) -> Result<Option<Entry>> {
        if !self.should_read(url) || url.len() > MAX_LABEL {
            return Ok(None);
        }
        retrying(|| self.read(url, scope, timestamp))
    }

    fn read(&self, url: &str, scope: Option<&str>, timestamp: i64) -> Attempt<Option<Entry>> {
        let Some(mut connection) = existing(&self.path, false)? else {
            return Ok(None);
        };
        let columns = columns(&connection)?;
        // Opening the first writer makes the file visible before its schema
        // transaction commits. Only an actually empty database is a cache miss.
        if columns.is_empty() && empty_database(&connection)? {
            return Ok(None);
        }
        let optional = |name: &'static str| if columns.contains(name) { name } else { "NULL" };
        // Bounds run inside SQLite before strings are transferred to Rust.
        let query = format!(
            "SELECT content,metadata,strategy_used,title,final_url,{etag},{modified},created_at
             FROM fetch_cache WHERE key=?1
             AND length(CAST(content AS BLOB))<=?2
             AND length(CAST(COALESCE(metadata,'') AS BLOB))<=?3
             AND length(CAST(strategy_used AS BLOB))<=?4
             AND length(CAST(COALESCE(title,'') AS BLOB))<=?4
             AND length(CAST(COALESCE(final_url,'') AS BLOB))<=?4
             AND length(CAST(COALESCE({etag},'') AS BLOB))<=?5
             AND length(CAST(COALESCE({modified},'') AS BLOB))<=?5",
            etag = optional("etag"),
            modified = optional("last_modified")
        );
        let key = key(url, scope);
        let row = connection
            .query_row(
                &query,
                params![
                    key,
                    MAX_CONTENT as i64,
                    MAX_METADATA as i64,
                    MAX_LABEL as i64,
                    MAX_HEADER as i64
                ],
                |row| {
                    Ok((
                        Entry {
                            content: row.get(0)?,
                            metadata: Map::new(),
                            strategy_used: row.get(2)?,
                            title: row.get(3)?,
                            final_url: row.get(4)?,
                            etag: row.get(5)?,
                            last_modified: row.get(6)?,
                            created_at: row.get(7)?,
                        },
                        row.get::<_, Option<String>>(1)?,
                    ))
                },
            )
            .optional()?;
        let Some((mut entry, metadata)) = row else {
            return Ok(None);
        };
        if entry.content.trim().is_empty()
            || (!entry.has_validators()
                && (self.ttl == 0
                    || timestamp.saturating_sub(entry.created_at).max(0) as u64 >= self.ttl))
        {
            return Ok(None);
        }
        if let Some(metadata) = metadata.filter(|text| !text.is_empty()) {
            entry.metadata = serde_json::from_str(&metadata)
                .map_err(|_| Fault::permanent("stored metadata is not a JSON object"))?;
        }
        touch_connection(&mut connection, &key, timestamp)?;
        Ok(Some(entry))
    }

    pub(crate) fn touch(&self, url: &str, scope: Option<&str>) -> Result<()> {
        let (key, timestamp) = (key(url, scope), now());
        retrying(|| {
            if let Some(mut connection) = existing(&self.path, false)? {
                touch_connection(&mut connection, &key, timestamp)?;
            }
            Ok(())
        })
    }

    /// Invalidate an old page after a successful response becomes ineligible.
    /// Failed fetches must retain the last good entry instead.
    pub(crate) fn remove(&self, url: &str, scope: Option<&str>) -> Result<()> {
        let key = key(url, scope);
        retrying(|| {
            if let Some(mut connection) = existing(&self.path, false)? {
                if columns(&connection)?.is_empty() && empty_database(&connection)? {
                    return Ok(());
                }
                let transaction =
                    connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
                transaction.execute("DELETE FROM fetch_cache WHERE key=?1", [&key])?;
                transaction.commit()?;
            }
            Ok(())
        })
    }

    pub(crate) fn set(&self, url: &str, scope: Option<&str>, entry: &Entry) -> Result<()> {
        self.set_at(url, scope, entry, now())
    }

    fn set_at(&self, url: &str, scope: Option<&str>, entry: &Entry, timestamp: i64) -> Result<()> {
        if entry.content.trim().is_empty() {
            return Ok(());
        }
        if url.len() > MAX_LABEL
            || entry.strategy_used.len() > MAX_LABEL
            || [&entry.title, &entry.final_url]
                .iter()
                .any(|v| v.as_ref().is_some_and(|v| v.len() > MAX_LABEL))
            || [&entry.etag, &entry.last_modified]
                .iter()
                .any(|v| v.as_ref().is_some_and(|v| v.len() > MAX_HEADER))
        {
            return Err(unavailable());
        }
        let size = i64::try_from(entry.content.len()).map_err(|_| unavailable())?;
        let admissible = size <= self.capacity && entry.content.len() <= MAX_CONTENT;
        // Rejected replacements invalidate their own old key, without creating
        // a new empty store or serializing their metadata.
        if !admissible {
            self.remove(url, scope)?;
            return Ok(());
        }
        let metadata = metadata_json(&entry.metadata)?;
        let key = key(url, scope);
        retrying(|| self.write(&key, url, entry, &metadata, size, timestamp))
    }

    /// One attempt at storing `entry`, creating the store on first use.
    fn write(
        &self,
        key: &str,
        url: &str,
        entry: &Entry,
        metadata: &str,
        size: i64,
        timestamp: i64,
    ) -> Attempt<()> {
        if let Some(parent) = self.path.parent() {
            crate::platform::private_directory()
                .recursive(true)
                .create(parent)?;
        }
        // Create the file before SQLite does, so it and the side files SQLite
        // copies its mode to are private from the start.
        match crate::platform::private_file(
            std::fs::OpenOptions::new().write(true).create_new(true),
        )
        .open(&self.path)
        {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
        private(&self.path)?;
        let mut connection = open(
            &self.path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )?;
        configure_wal(&connection)?;
        connection.pragma_update(None, "synchronous", "NORMAL")?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute_batch(SCHEMA)?;
        let present = columns(&transaction)?;
        for name in [
            "etag",
            "last_modified",
            "screenshot_path",
            "static_content",
            "browser_content",
        ] {
            if !present.contains(name) {
                transaction
                    .execute_batch(&format!("ALTER TABLE fetch_cache ADD COLUMN {name} TEXT"))?;
            }
        }
        transaction.execute("DELETE FROM fetch_cache WHERE key=?1", [key])?;
        let mut total: i64 = transaction.query_row(
            "SELECT COALESCE(SUM(size_bytes),0) FROM fetch_cache",
            [],
            |row| row.get(0),
        )?;
        while total > self.capacity - size {
            let oldest: Option<(String,i64)> = transaction.query_row("SELECT key,size_bytes FROM fetch_cache ORDER BY accessed_at ASC,rowid ASC LIMIT 1", [], |row| Ok((row.get(0)?,row.get(1)?))).optional()?;
            let Some((key, removed)) = oldest else { break };
            transaction.execute("DELETE FROM fetch_cache WHERE key=?1", [key])?;
            total = total.saturating_sub(removed);
        }
        transaction.execute("INSERT INTO fetch_cache (key,url,content,strategy_used,title,final_url,metadata,created_at,accessed_at,size_bytes,etag,last_modified,screenshot_path,static_content,browser_content) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?8,?9,?10,?11,NULL,NULL,NULL)",
            params![key,url,entry.content,entry.strategy_used,entry.title,entry.final_url,metadata,timestamp,size,entry.etag,entry.last_modified])?;
        transaction.commit()?;
        Ok(())
    }
}

struct MetadataBuffer(Vec<u8>);
impl Write for MetadataBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_METADATA.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other("Metadata size limit"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn metadata_json(metadata: &Map<String, Value>) -> Result<String> {
    let mut buffer = MetadataBuffer(Vec::new());
    serde_json::to_writer(&mut buffer, metadata).map_err(|_| unavailable())?;
    String::from_utf8(buffer.0).map_err(|_| unavailable())
}

fn columns(connection: &Connection) -> Attempt<HashSet<String>> {
    let mut query = connection.prepare("PRAGMA table_info(fetch_cache)")?;
    Ok(query
        .query_map([], |row| row.get(1))?
        .collect::<std::result::Result<HashSet<String>, _>>()?)
}

fn touch_connection(connection: &mut Connection, key: &str, timestamp: i64) -> Attempt<()> {
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute(
        "UPDATE fetch_cache SET accessed_at=?2 WHERE key=?1",
        params![key, timestamp],
    )?;
    transaction.commit()?;
    Ok(())
}

#[cfg(test)]
thread_local! {
    /// Opens on this thread that fail as a file held by another process.
    static HELD_OPENS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

fn open(path: &Path, flags: OpenFlags) -> Attempt<Connection> {
    #[cfg(test)]
    if HELD_OPENS.with(|left| left.replace(left.get().saturating_sub(1))) > 0 {
        // ERROR_SHARING_VIOLATION on Windows; an interrupted call elsewhere.
        #[cfg(windows)]
        let error = std::io::Error::from_raw_os_error(32);
        #[cfg(not(windows))]
        let error = std::io::Error::from(std::io::ErrorKind::Interrupted);
        return Err(error.into());
    }
    let connection = Connection::open_with_flags(path, flags)?;
    connection.busy_timeout(BUSY_TIMEOUT)?;
    connection.pragma_update(None, "trusted_schema", "OFF")?;
    Ok(connection)
}

fn empty_database(connection: &Connection) -> Attempt<bool> {
    Ok(connection.query_row(
        "SELECT NOT EXISTS(SELECT 1 FROM sqlite_schema WHERE name NOT GLOB 'sqlite_*')",
        [],
        |row| row.get(0),
    )?)
}

fn configure_wal(connection: &Connection) -> Attempt<()> {
    let deadline = Instant::now() + BUSY_TIMEOUT;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(Fault::permanent(
                "WAL mode stayed busy for the whole busy timeout",
            ));
        }
        connection.busy_timeout(remaining)?;
        match connection.query_row("PRAGMA journal_mode=WAL", [], |row| row.get::<_, String>(0)) {
            Ok(mode) if mode.eq_ignore_ascii_case("wal") => {
                connection.busy_timeout(BUSY_TIMEOUT)?;
                return Ok(());
            }
            // SQLite may decline to invoke the busy handler while promoting
            // locks for WAL. Retry the complete pragma after its statement ends.
            Err(rusqlite::Error::SqliteFailure(error, _))
                if matches!(
                    error.code,
                    rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
                ) =>
            {
                std::thread::sleep(
                    Duration::from_millis(10)
                        .min(deadline.saturating_duration_since(Instant::now())),
                );
            }
            Ok(mode) => return Err(Fault::permanent(format!("journal mode stayed {mode}"))),
            Err(error) => return Err(error.into()),
        }
    }
}

/// The store holds URLs with their query tokens and credential-fetched pages.
/// Writers narrow a store, and the SQLite side files present, to their owner,
/// since earlier versions created them with the default mode. Links are refused.
fn private(path: &Path) -> Attempt<()> {
    for suffix in ["", "-journal", "-wal", "-shm"] {
        let mut name = path.as_os_str().to_owned();
        name.push(suffix);
        let file = PathBuf::from(name);
        match std::fs::symlink_metadata(&file) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Ok(metadata) if metadata.is_file() => {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mode = metadata.permissions().mode();
                    if mode & 0o077 != 0 {
                        crate::platform::open_no_follow(
                            std::fs::OpenOptions::new().read(true),
                            &file,
                        )
                        .and_then(|opened| {
                            opened.set_permissions(std::fs::Permissions::from_mode(mode & 0o700))
                        })?;
                    }
                }
            }
            Ok(_) => return Err(Fault::permanent("a store file is not a regular file")),
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn existing(path: &Path, readonly: bool) -> Attempt<Option<Connection>> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.is_file() => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Ok(_) => return Err(Fault::permanent("the store is not a regular file")),
        Err(error) => return Err(error.into()),
    }
    if !readonly {
        private(path)?;
    }
    open(
        path,
        if readonly {
            OpenFlags::SQLITE_OPEN_READ_ONLY
        } else {
            OpenFlags::SQLITE_OPEN_READ_WRITE
        },
    )
    .map(Some)
}

fn megabytes(bytes: i64) -> f64 {
    (bytes as f64 / (1024.0 * 1024.0) * 100.0).round() / 100.0
}

/// Missing databases remain absent; callers retain the combined error envelope.
pub fn stats(cfg: &Value) -> Result<Value> {
    let path = path(cfg);
    let counted = retrying(|| {
        let Some(connection) = existing(&path, true)? else {
            return Ok(None);
        };
        let counts: (i64, i64) = connection.query_row(
            "SELECT COUNT(*),COALESCE(SUM(size_bytes),0) FROM fetch_cache",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        Ok(Some(counts))
    })?;
    let Some((count, size)) = counted else {
        return Ok(Value::Null);
    };
    Ok(
        json!({"count":count,"size_bytes":size,"size_mb":megabytes(size),"max_size_mb":megabytes(capacity(cfg)),"db_path":path}),
    )
}

/// Check an existing store's table and write transaction without changing rows.
pub fn preflight_clear(cfg: &Value) -> Result<()> {
    let path = path(cfg);
    retrying(|| {
        let Some(mut connection) = existing(&path, false)? else {
            return Ok(());
        };
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.query_row("SELECT COUNT(*) FROM fetch_cache", [], |row| {
            row.get::<_, i64>(0)
        })?;
        transaction.prepare("DELETE FROM fetch_cache")?;
        transaction.rollback()?;
        Ok(())
    })
}

/// Clear one store transactionally. Cross-store coordination belongs to the CLI.
pub fn clear(cfg: &Value) -> Result<u64> {
    let path = path(cfg);
    retrying(|| {
        let Some(mut connection) = existing(&path, false)? else {
            return Ok(0);
        };
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let removed = transaction.execute("DELETE FROM fetch_cache", [])?;
        transaction.commit()?;
        Ok(removed as u64)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    fn cfg(directory: &Path, capacity: i64, ttl: u64) -> Value {
        json!({"cache":{"enabled":true,"global_dir":directory,"max_size_bytes":capacity,"fetch_ttl_seconds":ttl}})
    }

    fn entry(content: &str) -> Entry {
        Entry {
            content: content.into(),
            strategy_used: "static".into(),
            ..Default::default()
        }
    }

    #[test]
    fn key_keeps_raw_url_and_explicit_strategy_separate() {
        let url = "https://Example.com/a?q=one#section";
        assert_eq!(key(url, None), key(url, Some("auto")));
        assert_ne!(key(url, None), key(url, Some("static")));
        assert_ne!(key(url, Some("static")), key(url, Some("jina")));
        assert_ne!(key(url, None), key(&url.replace("one", "two"), None));
        assert_ne!(key(url, None), key(&url.replace("#section", ""), None));
        assert_ne!(
            key(url, None),
            key(&url.replace("Example", "example"), None)
        );
        let reference = crate::hex(Sha256::digest(format!("2\0{url}").as_bytes()));
        assert_ne!(key(url, None), reference[..32]);
        // Rows of an earlier extraction are not replayed by this one.
        let first = crate::hex(Sha256::digest(format!("native-fetch-v1\0{url}").as_bytes()));
        assert_ne!(key(url, None), first[..32]);
        assert!(NAMESPACE.starts_with(concat!("native-fetch-", env!("CARGO_PKG_VERSION"))));
        assert_eq!(key(url, None).len(), 32);
    }

    #[test]
    fn url_patterns_preserve_path_case_and_match_authority_or_basename() {
        for (url, pattern) in [
            (
                "HTTPS://Example.COM/docs/page.html",
                "https://example.com/*",
            ),
            ("https://Example.COM/docs/page.html", "example.com/docs/*"),
            (
                "https://www.Example.com:8443/docs/page.html?token=secret",
                "*.example.com",
            ),
            (
                "https://example.com:8443/docs/page.html",
                "example.com:8443",
            ),
            ("https://example.com/docs/page.html?x=1#x", "*.html"),
            ("https://example.com/docs/page.html/", "page.html"),
            ("https://example.com/docs/page.html", "**/docs/*"),
            ("https://example.com/page.html", "**/page.html"),
            ("https://example.com/a3.html", "a[0-9].html"),
            ("https://example.com/a{old,new}.html", "a{old,new}.html"),
            ("https://example.com/unclosed[one.html", "unclosed[*.html"),
            ("http://[::1]:1234/a", "[[]::1]:1234"),
        ] {
            assert!(
                url_matches_patterns(url, &[pattern.into()]),
                "{url} {pattern}"
            );
        }
        for (url, pattern) in [
            ("https://example.com/Docs/page", "example.com/docs/*"),
            ("https://example.com/report.PDF", "*.pdf"),
            ("https://example.com/aold.html", "a{old,new}.html"),
            ("https://example.com/a", "  "),
            ("https://example.com/a", "other.com"),
            (
                "https://EXAMPLE.com?Token=ABC",
                "https://example.com?Token=abc",
            ),
            ("https://EXAMPLE.com#Section", "https://example.com#section"),
        ] {
            assert!(
                !url_matches_patterns(url, &[pattern.into()]),
                "{url} {pattern}"
            );
        }
        assert!(url_matches_patterns(
            "https://EXAMPLE.com?Token=ABC",
            &["https://example.com?Token=ABC".into()]
        ));
        assert!(url_matches_patterns(
            "https://EXAMPLE.com#Section",
            &["https://example.com#Section".into()]
        ));
    }

    #[test]
    fn inspection_disabled_and_oversized_first_write_do_not_create_state() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("absent");
        let mut cfg = cfg(&directory, 1, 60);
        let cache = Cache::from_config(&cfg).unwrap();
        assert!(cache.get("https://example.com/a", None).unwrap().is_none());
        assert_eq!(stats(&cfg).unwrap(), Value::Null);
        preflight_clear(&cfg).unwrap();
        assert_eq!(clear(&cfg).unwrap(), 0);
        cache
            .set("https://example.com/a", None, &entry("too large"))
            .unwrap();
        cache
            .set("https://example.com/a", None, &entry(" \n "))
            .unwrap();
        assert!(!directory.exists());
        cfg["cache"]["enabled"] = json!(false);
        assert!(Cache::from_config(&cfg).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn store_is_private_and_narrows_shared_files_from_earlier_versions() {
        use std::os::unix::fs::PermissionsExt;
        let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("state");
        let cache = Cache::from_config(&cfg(&directory, 1024, 60)).unwrap();
        let url = "https://example.com/a?token=secret";
        cache.set(url, None, &entry("page")).unwrap();
        assert_eq!(mode(&directory), 0o700);
        assert_eq!(mode(&cache.path), 0o600);

        // A store left shared by an earlier version is narrowed, with the
        // side files SQLite created from its mode, before the next write.
        let shared = |path: &Path| {
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o644)).unwrap()
        };
        shared(&cache.path);
        let reader = Connection::open(&cache.path).unwrap();
        reader
            .query_row("SELECT COUNT(*) FROM fetch_cache", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap();
        let side = |suffix: &str| directory.join(format!("fetch_cache.db{suffix}"));
        assert_eq!(mode(&side("-wal")), 0o644);
        cache.set(url, None, &entry("newer")).unwrap();
        for path in [cache.path.clone(), side("-wal"), side("-shm")] {
            assert_eq!(mode(&path), 0o600, "{}", path.display());
        }
        drop(reader);
        shared(&cache.path);
        assert!(cache.get(url, None).unwrap().is_some());
        assert_eq!(mode(&cache.path), 0o600);

        // A linked store is refused rather than written through.
        let linked = root.path().join("linked");
        std::fs::create_dir(&linked).unwrap();
        let target = root.path().join("elsewhere.db");
        std::fs::write(&target, b"").unwrap();
        std::os::unix::fs::symlink(&target, linked.join("fetch_cache.db")).unwrap();
        let cfg = cfg(&linked, 1024, 60);
        let cache = Cache::from_config(&cfg).unwrap();
        assert!(cache.set(url, None, &entry("page")).is_err());
        assert!(cache.get(url, None).is_err());
        assert!(stats(&cfg).is_err());
        assert_eq!(std::fs::metadata(&target).unwrap().len(), 0);
    }

    #[test]
    fn ttl_is_based_on_creation_and_validators_survive_expiration() {
        let root = tempfile::tempdir().unwrap();
        let cfg = cfg(root.path(), 1024, 10);
        let cache = Cache::from_config(&cfg).unwrap();
        let url = "https://example.com/a";
        cache.set_at(url, None, &entry("old"), 100).unwrap();
        assert!(cache.get_at(url, None, 109).unwrap().is_some());
        assert!(cache.get_at(url, None, 110).unwrap().is_none());
        assert_eq!(stats(&cfg).unwrap()["count"], 1);
        let connection = existing(&cache.path, true).unwrap().unwrap();
        let timestamps: (i64, i64) = connection
            .query_row(
                "SELECT created_at,accessed_at FROM fetch_cache",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(timestamps, (100, 109));
        for (etag, modified) in [
            (Some("v1"), None),
            (None, Some("Mon, 28 Sep 2026 01:00:00 GMT")),
        ] {
            let mut candidate = entry("validated");
            candidate.etag = etag.map(str::to_owned);
            candidate.last_modified = modified.map(str::to_owned);
            cache.set_at(url, None, &candidate, 100).unwrap();
            assert!(cache.get_at(url, None, 1_000_000).unwrap().is_some());
        }
        let mut no_ttl = cfg.clone();
        no_ttl["cache"]["fetch_ttl_seconds"] = json!(0);
        let zero = Cache::from_config(&no_ttl).unwrap();
        assert!(zero.get_at(url, None, 1_000_001).unwrap().is_some());
        zero.set_at(url, None, &entry("fresh"), 100).unwrap();
        assert!(zero.get_at(url, None, 100).unwrap().is_none());
        assert!(zero.get_at(url, None, 99).unwrap().is_none());
    }

    #[test]
    fn bypass_refresh_reopen_and_metadata_roundtrip() {
        let root = tempfile::tempdir().unwrap();
        let mut cfg = cfg(root.path(), 1024, 86_400);
        let cache = Cache::from_config(&cfg).unwrap();
        let url = "https://example.com/a";
        let mut page = entry("中文正文");
        page.metadata =
            serde_json::from_value(json!({"author":"作者","nested":{"items":[1,true]}})).unwrap();
        page.title = Some("Page".into());
        page.final_url = Some("https://example.com/final".into());
        cache.set(url, Some("static"), &page).unwrap();
        assert!(cache.get(url, None).unwrap().is_none());
        let restored = cache.get(url, Some("static")).unwrap().unwrap();
        assert_eq!(restored.metadata, page.metadata);
        assert_eq!(restored.final_url, page.final_url);
        assert_eq!(restored.title, page.title);
        cfg["cache"]["no_cache"] = json!(true);
        let bypass = Cache::from_config(&cfg).unwrap();
        assert!(!bypass.should_read(url));
        assert!(bypass.get(url, Some("static")).unwrap().is_none());
        bypass
            .set(url, Some("static"), &entry("refreshed"))
            .unwrap();
        assert_eq!(
            cache.get(url, Some("static")).unwrap().unwrap().content,
            "refreshed"
        );
        cfg["cache"]["no_cache"] = json!(false);
        cfg["cache"]["no_cache_patterns"] = json!(["Example.COM"]);
        let pattern = Cache::from_config(&cfg).unwrap();
        assert!(!pattern.should_read(url));
        pattern
            .set(url, Some("static"), &entry("pattern refresh"))
            .unwrap();
        cache.touch(url, Some("static")).unwrap();
        assert_eq!(
            cache.get(url, Some("static")).unwrap().unwrap().content,
            "pattern refresh"
        );
    }

    #[test]
    fn legacy_rows_are_inspectable_and_write_migrates_only_missing_columns() {
        let root = tempfile::tempdir().unwrap();
        let cfg = cfg(root.path(), 1024, 60);
        let path = path(&cfg);
        let connection = Connection::open(&path).unwrap();
        connection.execute_batch("CREATE TABLE fetch_cache (key TEXT PRIMARY KEY,url TEXT NOT NULL,content TEXT NOT NULL,strategy_used TEXT NOT NULL,title TEXT,final_url TEXT,metadata TEXT,created_at INTEGER NOT NULL,accessed_at INTEGER NOT NULL,size_bytes INTEGER NOT NULL); INSERT INTO fetch_cache VALUES ('legacy','https://example.com/a','Python page','static',NULL,NULL,NULL,100,100,11)").unwrap();
        assert_eq!(stats(&cfg).unwrap()["count"], 1);
        preflight_clear(&cfg).unwrap();
        assert!(!columns(&connection).unwrap().contains("etag"));
        let cache = Cache::from_config(&cfg).unwrap();
        assert!(cache.get("https://example.com/a", None).unwrap().is_none());
        cache
            .set("https://example.com/a", None, &entry("native page"))
            .unwrap();
        let names = columns(&connection).unwrap();
        for name in [
            "etag",
            "last_modified",
            "screenshot_path",
            "static_content",
            "browser_content",
        ] {
            assert!(names.contains(name));
        }
        assert_eq!(stats(&cfg).unwrap()["count"], 2);
        assert_eq!(clear(&cfg).unwrap(), 2);
        assert_eq!(stats(&cfg).unwrap()["count"], 0);
    }

    #[test]
    fn content_capacity_lru_oversize_and_insert_failure_are_atomic() {
        let root = tempfile::tempdir().unwrap();
        let cfg = cfg(root.path(), 6, 1000);
        let cache = Cache::from_config(&cfg).unwrap();
        cache.set_at("a", None, &entry("aaa"), 100).unwrap();
        cache.set_at("b", None, &entry("bbb"), 101).unwrap();
        cache.get_at("a", None, 102).unwrap();
        cache.set_at("c", None, &entry("ccc"), 103).unwrap();
        assert!(cache.get_at("b", None, 104).unwrap().is_none());
        cache.set_at("a", None, &entry("too large"), 104).unwrap();
        assert!(cache.get_at("a", None, 105).unwrap().is_none());
        assert!(cache.get_at("c", None, 105).unwrap().is_some());
        cache.set_at("a", None, &entry("aaa"), 106).unwrap();
        let connection = existing(&cache.path, false).unwrap().unwrap();
        connection.execute_batch("CREATE TRIGGER reject_new BEFORE INSERT ON fetch_cache BEGIN SELECT RAISE(ABORT,'private SQL URL and credentials'); END").unwrap();
        let error = cache
            .set_at("new", None, &entry("xxxx"), 107)
            .unwrap_err()
            .to_string();
        assert_eq!(error, "Persistent URL fetch cache is unavailable");
        assert_eq!(stats(&cfg).unwrap()["count"], 2);
        assert!(cache.get_at("c", None, 108).unwrap().is_some());
        assert!(cache.get_at("a", None, 108).unwrap().is_some());
    }

    #[test]
    fn corrupt_and_oversized_metadata_or_validators_never_replay() {
        let root = tempfile::tempdir().unwrap();
        let cfg = cfg(root.path(), 1024, 60);
        let cache = Cache::from_config(&cfg).unwrap();
        cache.set("a", None, &entry("valid")).unwrap();
        let connection = existing(&cache.path, false).unwrap().unwrap();
        connection
            .execute("UPDATE fetch_cache SET metadata='[]'", [])
            .unwrap();
        assert!(cache.get("a", None).is_err());
        connection
            .execute(
                "UPDATE fetch_cache SET metadata=CAST(zeroblob(?1) AS TEXT)",
                [MAX_METADATA as i64 + 1],
            )
            .unwrap();
        assert!(cache.get("a", None).unwrap().is_none());
        connection
            .execute(
                "UPDATE fetch_cache SET metadata=NULL,etag=CAST(zeroblob(?1) AS TEXT)",
                [MAX_HEADER as i64 + 1],
            )
            .unwrap();
        assert!(cache.get("a", None).unwrap().is_none());
        let mut bad = entry("valid");
        bad.etag = Some("x".repeat(MAX_HEADER + 1));
        assert!(cache.set("b", None, &bad).is_err());
        bad.etag = None;
        bad.metadata
            .insert("large".into(), json!("x".repeat(MAX_METADATA + 1)));
        assert!(cache.set("b", None, &bad).is_err());
        assert_eq!(stats(&cfg).unwrap()["count"], 1);
        drop(connection);
        std::fs::write(&cache.path, "private corrupt store").unwrap();
        for result in [
            stats(&cfg).map(|_| ()),
            preflight_clear(&cfg),
            clear(&cfg).map(|_| ()),
        ] {
            assert_eq!(
                result.unwrap_err().to_string(),
                "Persistent URL fetch cache is unavailable"
            );
        }
    }

    #[test]
    fn first_reader_can_observe_a_database_before_its_schema_is_committed() {
        let root = tempfile::tempdir().unwrap();
        let cfg = cfg(root.path(), 1024, 60);
        let cache = Cache::from_config(&cfg).unwrap();
        let mut writer = open(
            &cache.path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )
        .unwrap();
        writer.pragma_update(None, "journal_mode", "WAL").unwrap();
        let transaction = writer
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        transaction.execute_batch(SCHEMA).unwrap();
        assert!(cache.get("first", None).unwrap().is_none());
        cache.remove("ineligible", None).unwrap();
        transaction.commit().unwrap();
        cache.set("first", None, &entry("page")).unwrap();
        assert_eq!(cache.get("first", None).unwrap().unwrap().content, "page");
    }

    #[test]
    fn concurrent_first_writes_create_one_complete_cache_without_errors() {
        for round in 0..12 {
            let root = tempfile::tempdir().unwrap();
            let cfg = cfg(&root.path().join("new-state"), 1024, 60);
            let barrier = Arc::new(Barrier::new(8));
            let handles: Vec<_> = (0..8)
                .map(|number| {
                    let cfg = cfg.clone();
                    let barrier = barrier.clone();
                    std::thread::spawn(move || {
                        barrier.wait();
                        let cache = Cache::from_config(&cfg).unwrap();
                        let url = number.to_string();
                        cache.set(&url, None, &entry("page"))?;
                        // Readers meet the same first-use contention.
                        cache.get(&url, None)
                    })
                })
                .collect();
            let results: Vec<_> = handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect();
            assert!(
                results
                    .iter()
                    .all(|result| matches!(result, Ok(Some(entry)) if entry.content == "page")),
                "round {round}: {results:?}"
            );
            assert_eq!(stats(&cfg).unwrap()["count"], 8);
        }
    }

    #[test]
    fn a_store_file_held_for_a_moment_delays_reads_and_writes_without_losing_them() {
        let root = tempfile::tempdir().unwrap();
        let cfg = cfg(&root.path().join("state"), 1024, 60);
        let cache = Cache::from_config(&cfg).unwrap();
        HELD_OPENS.with(|left| left.set(3));
        cache.set("page", None, &entry("first")).unwrap();
        HELD_OPENS.with(|left| left.set(3));
        assert_eq!(cache.get("page", None).unwrap().unwrap().content, "first");
        HELD_OPENS.with(|left| left.set(2));
        cache.remove("page", None).unwrap();
        assert!(cache.get("page", None).unwrap().is_none());
        assert_eq!(HELD_OPENS.with(std::cell::Cell::get), 0);
    }

    #[test]
    fn transient_failures_are_retried_for_a_bounded_time_and_others_fail_at_once() {
        let busy = || Fault {
            transient: true,
            detail: "database is locked".into(),
        };
        let mut calls = 0;
        let value = retrying(|| {
            calls += 1;
            if calls < 3 { Err(busy()) } else { Ok(calls) }
        });
        assert_eq!(value.unwrap(), 3);
        let mut calls = 0;
        let error = retrying(|| -> Attempt<()> {
            calls += 1;
            Err(Fault::permanent("file is not a database"))
        })
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Persistent URL fetch cache is unavailable"
        );
        assert_eq!(calls, 1);
        let started = Instant::now();
        let mut calls = 0;
        assert!(
            retrying(|| -> Attempt<()> {
                calls += 1;
                Err(busy())
            })
            .is_err()
        );
        assert!(calls > 3, "{calls}");
        assert!(started.elapsed() < RETRY_WINDOW + LONGEST_RETRY_WAIT);
    }

    #[test]
    fn lock_contention_and_io_errors_are_transient_but_a_damaged_store_is_not() {
        use rusqlite::ffi;
        let sqlite = |code| rusqlite::Error::SqliteFailure(ffi::Error::new(code), None);
        for code in [
            ffi::SQLITE_BUSY,
            ffi::SQLITE_BUSY_RECOVERY,
            ffi::SQLITE_LOCKED,
            ffi::SQLITE_PROTOCOL,
            ffi::SQLITE_IOERR_SHMLOCK,
            ffi::SQLITE_IOERR_DELETE,
        ] {
            assert!(transient_sqlite(&sqlite(code)), "{code}");
        }
        for code in [
            ffi::SQLITE_NOTADB,
            ffi::SQLITE_CORRUPT,
            ffi::SQLITE_CONSTRAINT_TRIGGER,
            ffi::SQLITE_FULL,
            ffi::SQLITE_READONLY,
            ffi::SQLITE_ERROR,
        ] {
            assert!(!transient_sqlite(&sqlite(code)), "{code}");
        }
        // Windows names a file a scanner or a closing connection holds as one
        // it cannot open; elsewhere that is a lasting permission problem.
        assert_eq!(
            transient_sqlite(&sqlite(ffi::SQLITE_CANTOPEN)),
            cfg!(windows)
        );
        assert!(!transient_io(&std::io::Error::from(
            std::io::ErrorKind::PermissionDenied
        )));
        assert!(transient_io(&std::io::Error::from(
            std::io::ErrorKind::Interrupted
        )));
        #[cfg(windows)]
        for code in [5, 32, 33, 303] {
            // Access denied, sharing and lock violations, delete pending.
            assert!(
                transient_io(&std::io::Error::from_raw_os_error(code)),
                "{code}"
            );
        }
        #[cfg(unix)]
        assert!(!transient_io(&std::io::Error::from_raw_os_error(
            libc::EACCES
        )));
    }

    #[test]
    fn missing_cache_schema_in_an_unrelated_database_is_still_an_error() {
        let root = tempfile::tempdir().unwrap();
        let cfg = cfg(root.path(), 1024, 60);
        let cache = Cache::from_config(&cfg).unwrap();
        let connection = Connection::open(&cache.path).unwrap();
        connection
            .execute_batch("CREATE TABLE unrelated (value TEXT)")
            .unwrap();
        assert!(cache.get("page", None).is_err());
        assert!(cache.remove("page", None).is_err());
        connection
            .execute_batch("DROP TABLE unrelated; CREATE TABLE sqliteXother (value TEXT)")
            .unwrap();
        assert!(cache.get("page", None).is_err());
        assert!(cache.remove("page", None).is_err());
        let memory = Connection::open_in_memory().unwrap();
        assert!(configure_wal(&memory).is_err());
    }

    #[test]
    fn independent_connections_enforce_capacity_and_clear_preflight_is_read_only() {
        let root = tempfile::tempdir().unwrap();
        let cfg = cfg(root.path(), 15, 60);
        Cache::from_config(&cfg)
            .unwrap()
            .set("initial", None, &entry("aaa"))
            .unwrap();
        let barrier = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|n| {
                let cfg = cfg.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    Cache::from_config(&cfg)
                        .unwrap()
                        .set(&n.to_string(), None, &entry("bbb"))
                        .unwrap();
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        let before = stats(&cfg).unwrap();
        preflight_clear(&cfg).unwrap();
        assert_eq!(stats(&cfg).unwrap(), before);
        assert_eq!(before["count"], 5);
        assert_eq!(before["size_bytes"], 15);
        assert_eq!(clear(&cfg).unwrap(), 5);
    }

    #[test]
    fn removal_after_ineligible_response_only_invalidates_its_url_and_scope() {
        let root = tempfile::tempdir().unwrap();
        let cfg = cfg(&root.path().join("state"), 1024, 60);
        let cache = Cache::from_config(&cfg).unwrap();
        let url = "https://example.com/a";
        cache.remove(url, None).unwrap();
        assert!(!cache.path.exists());
        cache.set(url, None, &entry("auto page")).unwrap();
        cache
            .set(url, Some("static"), &entry("explicit page"))
            .unwrap();
        cache
            .set("https://example.com/b", None, &entry("another page"))
            .unwrap();
        cache.remove(url, None).unwrap();
        assert!(cache.get(url, None).unwrap().is_none());
        assert_eq!(
            cache.get(url, Some("static")).unwrap().unwrap().content,
            "explicit page"
        );
        assert!(cache.get("https://example.com/b", None).unwrap().is_some());
        assert_eq!(stats(&cfg).unwrap()["count"], 2);
    }

    #[test]
    fn clear_preflight_rejects_views_and_execution_failures_roll_back() {
        let root = tempfile::tempdir().unwrap();
        let view_cfg = cfg(root.path(), 1024, 60);
        let connection = Connection::open(path(&view_cfg)).unwrap();
        connection.execute_batch("CREATE TABLE pages(content TEXT); INSERT INTO pages VALUES ('saved'); CREATE VIEW fetch_cache AS SELECT content FROM pages").unwrap();
        assert_eq!(
            preflight_clear(&view_cfg).unwrap_err().to_string(),
            "Persistent URL fetch cache is unavailable"
        );
        assert_eq!(
            connection
                .query_row("SELECT COUNT(*) FROM pages", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            1
        );

        let table_cfg = cfg(&root.path().join("table"), 1024, 60);
        let cache = Cache::from_config(&table_cfg).unwrap();
        cache.set("a", None, &entry("aaa")).unwrap();
        cache.set("b", None, &entry("bbb")).unwrap();
        let connection = existing(&cache.path, false).unwrap().unwrap();
        connection.execute_batch("CREATE TRIGGER stop_clear BEFORE DELETE ON fetch_cache WHEN OLD.content='bbb' BEGIN SELECT RAISE(ABORT,'private URL in execution failure'); END").unwrap();
        preflight_clear(&table_cfg).unwrap();
        assert_eq!(stats(&table_cfg).unwrap()["count"], 2);
        assert_eq!(
            clear(&table_cfg).unwrap_err().to_string(),
            "Persistent URL fetch cache is unavailable"
        );
        assert_eq!(stats(&table_cfg).unwrap()["count"], 2);
        assert_eq!(cache.get("a", None).unwrap().unwrap().content, "aaa");
        assert_eq!(cache.get("b", None).unwrap().unwrap().content, "bbb");
    }
}
