//! Persistent Markdown answers, isolated by prompt/content/model fingerprints.

use crate::{Error, Result, config};
use globset::GlobBuilder;
use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior, params};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_CAPACITY: i64 = 512 * 1024 * 1024;
const MAX_ENTRY_BYTES: i64 = 100 * 1024 * 1024;
const MAX_LIST_ENTRIES: usize = 1000;
const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY, value TEXT NOT NULL, model TEXT DEFAULT '',
    created_at INTEGER NOT NULL, accessed_at INTEGER NOT NULL, size_bytes INTEGER NOT NULL
); CREATE INDEX IF NOT EXISTS idx_accessed ON cache(accessed_at);
CREATE INDEX IF NOT EXISTS idx_created ON cache(created_at);";

pub(crate) struct Cache {
    path: PathBuf,
    capacity: i64,
    skip_read: bool,
}

fn unavailable() -> Error {
    // SQLite errors may contain file paths, SQL or cached content.
    Error::Conversion("Persistent LLM cache is unavailable".into())
}

fn directory(cfg: &Value) -> PathBuf {
    config::state_path(Path::new(
        cfg.pointer("/cache/global_dir")
            .and_then(Value::as_str)
            .unwrap_or("~/.markitai"),
    ))
}

fn enabled(cfg: &Value) -> bool {
    cfg.pointer("/cache/enabled")
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

fn capacity(cfg: &Value) -> i64 {
    let value = cfg.pointer("/cache/max_size_bytes");
    value
        .and_then(Value::as_i64)
        .or_else(|| {
            value
                .and_then(Value::as_u64)
                .map(|n| n.min(i64::MAX as u64) as i64)
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

/// Cache keys do not persist raw prompts, input paths, endpoints or credentials.
pub(crate) fn key(content: &str, prompt_scope: &str, model_scope: &str) -> String {
    let content_hash = crate::hex(Sha256::digest(content.as_bytes()));
    let combined = format!("native-markdown-v1:{prompt_scope}|{model_scope}|{content_hash}");
    crate::hex(Sha256::digest(combined.as_bytes()))[..32].to_owned()
}

/// Typed document rows cannot collide with legacy Markdown-only answers.
pub(crate) fn document_key(content: &str, prompt_scope: &str, model_scope: &str) -> String {
    let content_hash = crate::hex(Sha256::digest(content.as_bytes()));
    let combined = format!("native-document-v1:{prompt_scope}|{model_scope}|{content_hash}");
    crate::hex(Sha256::digest(combined.as_bytes()))[..32].to_owned()
}

/// Framed bytes make page order, MIME and changed pixels part of cache identity.
pub(crate) fn vision_key<'a>(
    content: &str,
    prompt_scope: &str,
    model_scope: &str,
    images: impl IntoIterator<Item = (usize, &'a str, &'a [u8])>,
) -> String {
    let mut digest = Sha256::new();
    for part in ["native-vision-v1", prompt_scope, model_scope, content] {
        digest.update((part.len() as u64).to_le_bytes());
        digest.update(part.as_bytes());
    }
    for (number, mime, bytes) in images {
        digest.update((number as u64).to_le_bytes());
        digest.update((mime.len() as u64).to_le_bytes());
        digest.update(mime.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    crate::hex(digest.finalize())[..32].to_owned()
}

pub(crate) fn model_scope<'a>(models: impl IntoIterator<Item = &'a str>) -> String {
    let mut models: Vec<_> = models
        .into_iter()
        .filter(|model| !model.is_empty())
        .collect();
    models.sort_unstable();
    models.dedup();
    if models.is_empty() {
        "pool:none".into()
    } else {
        let digest = crate::hex(Sha256::digest(models.join("\n").as_bytes()));
        format!("pool:{}", &digest[..16])
    }
}

/// Digest templates before substituting timestamps and local source labels.
pub(crate) fn prompt_scope(parts: &[&str]) -> String {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update((part.len() as u64).to_le_bytes());
        digest.update(part.as_bytes());
    }
    crate::hex(digest.finalize())
}

// fnmatch-style patterns treat slashes as ordinary characters and braces as
// literals. Normalize Windows separators before handing the pattern to globset.
fn glob_pattern(pattern: &str) -> String {
    let chars: Vec<_> = pattern.chars().collect();
    let mut pattern = String::new();
    let mut index = 0;
    while index < chars.len() {
        match chars[index] {
            '{' | '}' | '\\' => {
                pattern.push('\\');
                pattern.push(chars[index]);
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
                    pattern.push('[');
                } else {
                    pattern.push_str("\\[");
                }
            }
            ch => pattern.push(ch),
        }
        index += 1;
    }
    pattern
}

fn glob_matches(path: &str, pattern: &str) -> bool {
    GlobBuilder::new(&glob_pattern(pattern))
        .literal_separator(false)
        .backslash_escape(true)
        .case_insensitive(cfg!(windows))
        .build()
        .is_ok_and(|glob| glob.compile_matcher().is_match(path))
}

fn pattern_matches(path: &str, pattern: &str) -> bool {
    glob_matches(path, pattern)
        || pattern
            .strip_prefix("**/")
            .is_some_and(|rest| glob_matches(path, rest))
        || (!pattern.starts_with("**/")
            && pattern.contains("**/")
            && glob_matches(path, &pattern.replacen("**/", "", 1)))
}

pub(crate) fn bypasses(cfg: &Value, context: &str) -> bool {
    if cfg.pointer("/cache/no_cache").and_then(Value::as_bool) == Some(true) {
        return true;
    }
    if context.is_empty() {
        return false;
    }
    let path = context.replace('\\', "/");
    // Stage suffixes follow the path, except for a Windows drive's colon.
    let after_drive = usize::from(path.as_bytes().get(1) == Some(&b':')) * 2;
    let path_end = path[after_drive..]
        .find(':')
        .map_or(path.len(), |index| after_drive + index);
    let filename = if path.starts_with("http://") || path.starts_with("https://") {
        path.split(['?', '#'])
            .next()
            .unwrap_or(&path)
            .rsplit('/')
            .next()
            .unwrap_or("")
    } else {
        path[..path_end].rsplit('/').next().unwrap_or("")
    };
    cfg.pointer("/cache/no_cache_patterns")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|pattern| pattern.replace('\\', "/"))
        .any(|pattern| pattern_matches(&path, &pattern) || pattern_matches(filename, &pattern))
}

impl Cache {
    pub(crate) fn configured(cfg: &Value, context: &str) -> Option<Self> {
        enabled(cfg).then(|| Self {
            path: directory(cfg).join("cache.db"),
            capacity: capacity(cfg),
            skip_read: bypasses(cfg, context),
        })
    }

    pub(crate) fn get(&self, key: &str) -> Result<Option<String>> {
        match self.get_json(key)? {
            Some(Value::String(text)) if !text.trim().is_empty() => Ok(Some(text)),
            Some(Value::String(_)) | None => Ok(None),
            Some(_) => Err(unavailable()),
        }
    }

    pub(crate) fn get_json(&self, key: &str) -> Result<Option<Value>> {
        if self.skip_read {
            return Ok(None);
        }
        let Some(mut connection) = existing(&self.path, false)? else {
            return Ok(None);
        };
        // A corrupt/malicious row cannot force an unbounded answer allocation.
        let value: Option<Option<String>> = connection
            .query_row(
                "SELECT CASE WHEN length(CAST(value AS BLOB)) <= ?2 THEN value ELSE NULL END FROM cache WHERE key = ?1",
                params![key, MAX_ENTRY_BYTES],
                |row| row.get(0),
            )
            .optional()
            .map_err(|_| unavailable())?;
        let Some(value) = value.flatten() else {
            return Ok(None);
        };
        let answer: Value = serde_json::from_str(&value).map_err(|_| unavailable())?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| unavailable())?;
        transaction
            .execute(
                "UPDATE cache SET accessed_at = ?2 WHERE key = ?1",
                params![key, now()],
            )
            .map_err(|_| unavailable())?;
        transaction.commit().map_err(|_| unavailable())?;
        Ok(Some(answer))
    }

    pub(crate) fn set(&self, key: &str, model: &str, answer: &str) -> Result<()> {
        if answer.trim().is_empty() {
            return Ok(());
        }
        self.set_json(key, model, &Value::String(answer.to_owned()))
    }

    pub(crate) fn set_json(&self, key: &str, model: &str, answer: &Value) -> Result<()> {
        let value = serde_json::to_string(answer).map_err(|_| unavailable())?;
        let size = i64::try_from(value.len()).map_err(|_| unavailable())?;
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|_| unavailable())?;
        }
        let mut connection = initialize_writer(&self.path)?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| unavailable())?;
        transaction
            .execute("DELETE FROM cache WHERE key = ?1", [key])
            .map_err(|_| unavailable())?;
        // Remove a stale replacement, but leave unrelated rows intact when the
        // new result alone cannot fit. Admission and all evictions are atomic.
        if size <= self.capacity && size <= MAX_ENTRY_BYTES {
            let mut total: i64 = transaction
                .query_row(
                    "SELECT COALESCE(SUM(size_bytes), 0) FROM cache",
                    [],
                    |row| row.get(0),
                )
                .map_err(|_| unavailable())?;
            while total > self.capacity - size {
                let oldest: Option<(String, i64)> = transaction
                    .query_row("SELECT key, size_bytes FROM cache ORDER BY accessed_at ASC, rowid ASC LIMIT 1", [], |row| Ok((row.get(0)?, row.get(1)?)))
                    .optional().map_err(|_| unavailable())?;
                let Some((key, removed)) = oldest else { break };
                transaction
                    .execute("DELETE FROM cache WHERE key = ?1", [key])
                    .map_err(|_| unavailable())?;
                total = total.saturating_sub(removed);
            }
            let timestamp = now();
            transaction.execute(
                "INSERT INTO cache (key,value,model,created_at,accessed_at,size_bytes) VALUES (?1,?2,?3,?4,?4,?5)",
                params![key, value, model, timestamp, size],
            ).map_err(|_| unavailable())?;
        }
        transaction.commit().map_err(|_| unavailable())?;
        Ok(())
    }
}

// WAL migration upgrades a read lock and can return BUSY without invoking
// SQLite's busy handler. Retrying a fresh connection releases that read lock;
// retrying the insertion transaction would have a different correctness scope.
fn initialize_writer(path: &Path) -> Result<Connection> {
    initialize_writer_with_wait(path, Duration::from_secs(30), &mut std::thread::sleep)
}

fn initialize_writer_with_wait(
    path: &Path,
    timeout: Duration,
    wait: &mut dyn FnMut(Duration),
) -> Result<Connection> {
    let started = Instant::now();
    loop {
        let remaining = timeout.saturating_sub(started.elapsed());
        // No single internal busy wait can outlive the total initialization
        // deadline. Short attempts also release migration locks promptly.
        let result = initialize_writer_once(path, remaining.min(Duration::from_millis(100)));
        match result {
            Ok(connection) => {
                // Ordinary row transactions retain their existing busy policy;
                // only the idempotent initialization above is replayed.
                connection
                    .busy_timeout(Duration::from_secs(30))
                    .map_err(|_| unavailable())?;
                return Ok(connection);
            }
            Err(rusqlite::Error::SqliteFailure(error, _))
                if matches!(
                    error.code,
                    rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
                ) =>
            {
                let remaining = timeout.saturating_sub(started.elapsed());
                if remaining.is_zero() {
                    return Err(unavailable());
                }
                wait(remaining.min(Duration::from_millis(10)));
            }
            Err(_) => return Err(unavailable()),
        }
    }
}

fn initialize_writer_once(path: &Path, timeout: Duration) -> rusqlite::Result<Connection> {
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
    )?;
    connection.busy_timeout(timeout)?;
    connection.pragma_update(None, "trusted_schema", "OFF")?;
    let mode: String = connection.pragma_query_value(None, "journal_mode", |row| row.get(0))?;
    if !mode.eq_ignore_ascii_case("wal") {
        let mode: String =
            connection.pragma_update_and_check(None, "journal_mode", "WAL", |row| row.get(0))?;
        if !mode.eq_ignore_ascii_case("wal") {
            return Err(rusqlite::Error::InvalidQuery);
        }
    }
    connection.pragma_update(None, "synchronous", "NORMAL")?;
    connection.execute_batch(SCHEMA)?;
    Ok(connection)
}

fn open(path: &Path, flags: OpenFlags) -> Result<Connection> {
    let connection = Connection::open_with_flags(path, flags).map_err(|_| unavailable())?;
    connection
        .busy_timeout(Duration::from_secs(30))
        .map_err(|_| unavailable())?;
    connection
        .pragma_update(None, "trusted_schema", "OFF")
        .map_err(|_| unavailable())?;
    Ok(connection)
}

fn existing(path: &Path, readonly: bool) -> Result<Option<Connection>> {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        _ => return Err(unavailable()),
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

fn timestamp(seconds: i64) -> Value {
    chrono::DateTime::from_timestamp(seconds, 0)
        .map(|value| json!(value.to_rfc3339_opts(chrono::SecondsFormat::Secs, false)))
        .unwrap_or(Value::Null)
}

fn preview(raw: &str) -> String {
    let (kind, text) = match serde_json::from_str::<Value>(raw) {
        Ok(Value::String(text)) => ("text", text),
        Ok(value) if value.get("caption").is_some() => {
            ("image", value["caption"].as_str().unwrap_or("").into())
        }
        Ok(value) if value.get("cleaned_markdown").is_some() => (
            "document",
            value["cleaned_markdown"].as_str().unwrap_or("").into(),
        ),
        Ok(value) if value.get("title").is_some() => {
            ("frontmatter", value["title"].as_str().unwrap_or("").into())
        }
        Ok(value) => ("text", value.to_string()),
        Err(_) => ("text", raw.to_owned()),
    };
    format!("{kind}: {}...", text.chars().take(40).collect::<String>())
}

/// Reference-shaped LLM statistics. An absent cache is not created by inspection.
pub fn stats(cfg: &Value, verbose: bool, limit: usize) -> Result<Value> {
    let directory = directory(cfg);
    let path = directory.join("cache.db");
    let fetch = crate::fetch_cache::stats(cfg)
        .unwrap_or_else(|_| json!({"error":"Persistent URL fetch cache is unavailable"}));
    let cache = inspect(cfg, &path, verbose, limit)
        .unwrap_or_else(|_| json!({"error":"Persistent LLM cache is unavailable"}));
    Ok(json!({"cache":cache,"enabled":enabled(cfg),"fetch_cache":fetch}))
}

fn inspect(cfg: &Value, path: &Path, verbose: bool, limit: usize) -> Result<Value> {
    let Some(connection) = existing(path, true)? else {
        return Ok(Value::Null);
    };
    // Hold one read snapshot so verbose groups and totals agree even while
    // other processes refresh or evict entries.
    let connection = connection
        .unchecked_transaction()
        .map_err(|_| unavailable())?;
    let (count, bytes): (i64, i64) = connection
        .query_row(
            "SELECT COUNT(*),COALESCE(SUM(size_bytes),0) FROM cache",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|_| unavailable())?;
    let mut details = json!({"count":count,"size_bytes":bytes,"size_mb":megabytes(bytes),"max_size_mb":megabytes(capacity(cfg)),"db_path":path});
    if verbose {
        let mut groups = Map::new();
        let mut statement = connection.prepare("SELECT COALESCE(NULLIF(model,''),'unknown'),COUNT(*),COALESCE(SUM(size_bytes),0) FROM cache GROUP BY 1 ORDER BY 3 DESC").map_err(|_| unavailable())?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })
            .map_err(|_| unavailable())?;
        for row in rows {
            let (model, count, size) = row.map_err(|_| unavailable())?;
            groups.insert(
                model,
                json!({"count":count,"size_bytes":size,"size_mb":megabytes(size)}),
            );
        }
        details["by_model"] = Value::Object(groups);
        let mut statement = connection.prepare("SELECT key,model,size_bytes,created_at,accessed_at,substr(value,1,200) FROM cache ORDER BY accessed_at DESC,rowid DESC LIMIT ?1").map_err(|_| unavailable())?;
        let rows = statement
            .query_map([limit.min(MAX_LIST_ENTRIES) as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, String>(5)?,
                ))
            })
            .map_err(|_| unavailable())?;
        let mut entries = Vec::new();
        for row in rows {
            let (key, model, size, created, accessed, value) = row.map_err(|_| unavailable())?;
            entries.push(json!({"key":key,"model":model.filter(|name| !name.is_empty()).unwrap_or_else(|| "unknown".into()),"size_bytes":size,"created_at":timestamp(created),"accessed_at":timestamp(accessed),"preview":preview(&value)}));
        }
        details["entries"] = Value::Array(entries);
    }
    Ok(details)
}

/// Validate a writable store before starting a combined cache clear.
pub fn preflight_clear(cfg: &Value) -> Result<()> {
    let Some(mut connection) = existing(&directory(cfg).join("cache.db"), false)? else {
        return Ok(());
    };
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| unavailable())?;
    transaction
        .prepare("DELETE FROM cache")
        .map_err(|_| unavailable())?;
    transaction.rollback().map_err(|_| unavailable())
}

/// Clear LLM entries only; the caller owns any fetch-cache/SPA preflight.
pub fn clear(cfg: &Value) -> Result<u64> {
    let Some(mut connection) = existing(&directory(cfg).join("cache.db"), false)? else {
        return Ok(0);
    };
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| unavailable())?;
    let removed = transaction
        .execute("DELETE FROM cache", [])
        .map_err(|_| unavailable())?;
    transaction.commit().map_err(|_| unavailable())?;
    Ok(removed as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    fn configuration(directory: &Path, size: i64) -> Value {
        json!({"cache":{"enabled":true,"global_dir":directory,"max_size_bytes":size}})
    }

    #[test]
    fn rollback_lock_during_wal_initialization_is_retried_after_release() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("cache.db");
        let holder = Connection::open(&path).unwrap();
        holder.execute_batch("CREATE TABLE existing(value TEXT); INSERT INTO existing VALUES ('retained'); BEGIN IMMEDIATE;").unwrap();
        let mut retries = 0;
        let writer = initialize_writer_with_wait(&path, Duration::from_secs(2), &mut |_| {
            retries += 1;
            assert_eq!(retries, 1);
            holder.execute_batch("COMMIT").unwrap();
        })
        .unwrap();
        assert_eq!(
            retries, 1,
            "held rollback lock must exercise migration contention"
        );
        let mode: String = writer
            .pragma_query_value(None, "journal_mode", |row| row.get(0))
            .unwrap();
        assert_eq!(mode, "wal");
        assert_eq!(
            writer
                .query_row("SELECT value FROM existing", [], |row| row
                    .get::<_, String>(0))
                .unwrap(),
            "retained"
        );
        drop(writer);
        let cfg = configuration(root.path(), 10_000);
        let cache = Cache::configured(&cfg, "").unwrap();
        let value = json!({"cleaned_markdown":"body","description":"description","tags":["tag"]});
        cache.set_json("new", "pool", &value).unwrap();
        assert_eq!(cache.get_json("new").unwrap(), Some(value));
    }

    #[test]
    fn initialization_contention_deadline_and_corruption_remain_errors() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("cache.db");
        let holder = Connection::open(&path).unwrap();
        holder
            .execute_batch("CREATE TABLE existing(value TEXT); BEGIN IMMEDIATE;")
            .unwrap();
        let failure = initialize_writer_with_wait(&path, Duration::ZERO, &mut |_| {
            panic!("expired deadline must not wait")
        });
        assert!(failure.is_err());
        holder.execute_batch("ROLLBACK").unwrap();
        let bad = root.path().join("malformed.db");
        std::fs::write(&bad, b"private malformed database bytes").unwrap();
        let failure = initialize_writer_with_wait(&bad, Duration::from_secs(30), &mut |_| {
            panic!("corruption must not retry")
        });
        assert_eq!(
            failure.unwrap_err().to_string(),
            "Persistent LLM cache is unavailable"
        );
        assert_eq!(
            std::fs::read(bad).unwrap(),
            b"private malformed database bytes"
        );
    }

    #[test]
    #[ignore = "child process for first-write cache coordination"]
    fn fresh_cache_process_writer() {
        let root = PathBuf::from(
            std::env::var_os("MARKITAI_CACHE_TEST_ROOT").expect("private test directory"),
        );
        let key = std::env::var("MARKITAI_CACHE_TEST_KEY").expect("test key");
        std::fs::write(root.join(format!("ready-{key}")), b"ready").unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !root.join("release").exists() {
            assert!(
                Instant::now() < deadline,
                "parent did not release cache writers"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
        let cache = Cache::configured(&configuration(&root, 1_000_000), "").unwrap();
        cache
            .set_json(
                &key,
                "pool",
                &json!({"cleaned_markdown":key,"description":"child","tags":["topic"]}),
            )
            .unwrap();
    }

    #[test]
    fn independent_processes_create_one_cache_without_losing_rows() {
        struct Workers(Vec<std::process::Child>);
        impl Drop for Workers {
            fn drop(&mut self) {
                for child in &mut self.0 {
                    if child.try_wait().ok().flatten().is_none() {
                        let _ = child.kill();
                    }
                    let _ = child.wait();
                }
            }
        }
        let root = tempfile::tempdir().unwrap();
        let mut workers = Workers(Vec::new());
        for index in 0..4 {
            workers.0.push(
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "llm_cache::tests::fresh_cache_process_writer",
                        "--ignored",
                        "--nocapture",
                    ])
                    .env("MARKITAI_CACHE_TEST_ROOT", root.path())
                    .env("MARKITAI_CACHE_TEST_KEY", index.to_string())
                    .stdout(std::process::Stdio::null())
                    .spawn()
                    .unwrap(),
            );
        }
        let deadline = Instant::now() + Duration::from_secs(15);
        while !(0..4).all(|index| root.path().join(format!("ready-{index}")).exists()) {
            assert!(
                Instant::now() < deadline,
                "cache child did not become ready"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(!root.path().join("cache.db").exists());
        std::fs::write(root.path().join("release"), b"release").unwrap();
        for child in &mut workers.0 {
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success(), "cache child failed: {status}");
                    break;
                }
                assert!(Instant::now() < deadline, "cache child exceeded deadline");
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        let cache = Cache::configured(&configuration(root.path(), 1_000_000), "").unwrap();
        for index in 0..4 {
            let key = index.to_string();
            assert_eq!(
                cache.get_json(&key).unwrap().unwrap()["cleaned_markdown"],
                key
            );
        }
    }

    #[test]
    fn simultaneous_first_writes_preserve_every_typed_chunk_without_prewarming() {
        let root = tempfile::tempdir().unwrap();
        let cfg = configuration(root.path(), 1_000_000);
        let barrier = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8).map(|i| {
            let cfg = cfg.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                let cache = Cache::configured(&cfg, "").unwrap();
                let answer = json!({"cleaned_markdown":format!("chunk {i}"),"description":"summary","tags":["topic"]});
                cache.set_json(&i.to_string(), "pool", &answer).unwrap();
            })
        }).collect();
        for handle in handles {
            handle.join().unwrap();
        }
        let cache = Cache::configured(&cfg, "").unwrap();
        for i in 0..8 {
            assert_eq!(
                cache.get_json(&i.to_string()).unwrap().unwrap()["cleaned_markdown"],
                format!("chunk {i}")
            );
        }
        assert_eq!(stats(&cfg, false, 20).unwrap()["cache"]["count"], 8);
    }

    #[test]
    fn typed_rows_are_isolated_preserved_and_url_globs_use_url_basename() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = configuration(dir.path(), 10_000);
        let cache = Cache::configured(&cfg, "https://example.test/article.md?q=1").unwrap();
        let legacy = key("body", "prompt", "pool");
        let typed = document_key("body", "prompt", "pool");
        assert_ne!(legacy, typed);
        cache.set(&legacy, "pool", "legacy Markdown").unwrap();
        let value = json!({"cleaned_markdown":"body","description":"summary","tags":["topic"]});
        cache.set_json(&typed, "pool", &value).unwrap();
        assert_eq!(cache.get_json(&typed).unwrap(), Some(value));
        assert_eq!(
            cache.get(&legacy).unwrap().as_deref(),
            Some("legacy Markdown")
        );
        assert!(cache.get(&typed).is_err());
        let bypass = json!({"cache":{"no_cache_patterns":["article.md"]}});
        assert!(bypasses(&bypass, "https://example.test/article.md?q=1"));
        assert!(!bypasses(&bypass, "https://example.test/other.md"));
    }

    #[test]
    fn missing_disabled_and_inspection_do_not_create_state() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("not-created");
        let mut cfg = configuration(&target, 100);
        assert_eq!(stats(&cfg, true, usize::MAX).unwrap()["cache"], Value::Null);
        preflight_clear(&cfg).unwrap();
        assert_eq!(clear(&cfg).unwrap(), 0);
        assert_eq!(
            Cache::configured(&cfg, "file.md")
                .unwrap()
                .get("missing")
                .unwrap(),
            None
        );
        assert!(!target.exists());
        cfg["cache"]["enabled"] = json!(false);
        assert!(Cache::configured(&cfg, "file.md").is_none());
        assert!(!target.exists());
    }

    #[test]
    fn namespaced_keys_hash_middle_changes_prompts_and_model_sets() {
        let body = format!("{}original{}", "a".repeat(30_000), "z".repeat(30_000));
        assert_ne!(
            key(&body, "p", "pool:a"),
            key(&body.replace("original", "changed"), "p", "pool:a")
        );
        assert_ne!(key(&body, "p", "pool:a"), key(&body, "p2", "pool:a"));
        assert_ne!(key(&body, "p", "pool:a"), key(&body, "p", "pool:b"));
        assert_eq!(model_scope(["b", "a", "b"]), model_scope(["a", "b"]));
        assert_eq!(model_scope([]), "pool:none");
        assert_ne!(prompt_scope(&["ab", "c"]), prompt_scope(&["a", "bc"]));
    }

    #[test]
    fn patterns_handle_basename_stages_globstars_and_literal_braces() {
        for (pattern, context) in [
            ("*.md", "/nested/file.md"),
            ("**/*.pdf", "file.pdf"),
            ("dir/**/*.pdf", "dir/file.pdf"),
            ("dir/**/*.pdf", "dir/sub/file.pdf"),
            ("*.pdf", "C:\\docs\\file.pdf:images"),
            ("a[0-9].md", "a3.md"),
            ("name{old,new}.md", "name{old,new}.md"),
            ("unclosed[*.md", "unclosed[one.md"),
        ] {
            let cfg = json!({"cache":{"no_cache_patterns":[pattern]}});
            assert!(bypasses(&cfg, context), "{pattern}: {context}");
        }
        assert!(!bypasses(
            &json!({"cache":{"no_cache_patterns":["name{old,new}.md"]}}),
            "nameold.md"
        ));
        assert!(!bypasses(
            &json!({"cache":{"no_cache_patterns":["*.pdf"]}}),
            "file.md"
        ));
    }

    #[test]
    fn refresh_read_bypass_and_reopen_preserve_successful_markdown() {
        let root = tempfile::tempdir().unwrap();
        let mut cfg = configuration(root.path(), 1024);
        let cache = Cache::configured(&cfg, "file.md").unwrap();
        cache.set("key", "pool:test", "旧正文").unwrap();
        assert_eq!(cache.get("key").unwrap().as_deref(), Some("旧正文"));
        cfg["cache"]["no_cache"] = json!(true);
        let refresh = Cache::configured(&cfg, "file.md").unwrap();
        assert!(refresh.get("key").unwrap().is_none());
        refresh.set("key", "pool:test", "fresh").unwrap();
        assert_eq!(cache.get("key").unwrap().as_deref(), Some("fresh"));
        let stats = stats(&cfg, true, 20).unwrap();
        assert_eq!(stats["cache"]["count"], 1);
        assert_eq!(stats["cache"]["by_model"]["pool:test"]["count"], 1);
        assert_eq!(stats["cache"]["entries"][0]["preview"], "text: fresh...");
        assert_eq!(clear(&cfg).unwrap(), 1);
        assert!(cache.get("key").unwrap().is_none());
    }

    #[test]
    fn capacity_eviction_is_lru_and_oversize_keeps_unrelated_rows() {
        let root = tempfile::tempdir().unwrap();
        let cfg = configuration(root.path(), 10);
        let cache = Cache::configured(&cfg, "file.md").unwrap();
        cache.set("a", "pool", "aaa").unwrap();
        cache.set("b", "pool", "bbb").unwrap();
        let connection = existing(&cache.path, false).unwrap().unwrap();
        connection
            .execute("UPDATE cache SET accessed_at=1 WHERE key='a'", [])
            .unwrap();
        connection
            .execute("UPDATE cache SET accessed_at=2 WHERE key='b'", [])
            .unwrap();
        cache.get("a").unwrap();
        cache.set("c", "pool", "ccc").unwrap();
        assert!(cache.get("b").unwrap().is_none());
        assert!(cache.get("a").unwrap().is_some());
        cache.set("a", "pool", "does not fit").unwrap();
        assert!(cache.get("a").unwrap().is_none());
        assert_eq!(cache.get("c").unwrap().as_deref(), Some("ccc"));
        cache.set("blank", "pool", " \n ").unwrap();
        assert_eq!(stats(&cfg, false, 20).unwrap()["cache"]["count"], 1);
    }

    #[test]
    fn failed_insert_rolls_back_replacement_and_eviction() {
        let root = tempfile::tempdir().unwrap();
        let cfg = configuration(root.path(), 10);
        let cache = Cache::configured(&cfg, "file.md").unwrap();
        cache.set("a", "pool", "aaa").unwrap();
        cache.set("b", "pool", "bbb").unwrap();
        let connection = existing(&cache.path, false).unwrap().unwrap();
        connection.execute_batch("CREATE TRIGGER fail_write BEFORE INSERT ON cache WHEN NEW.key='c' BEGIN SELECT RAISE(ABORT,'secret SQL text'); END;").unwrap();
        let error = cache.set("c", "pool", "ccc").unwrap_err().to_string();
        assert!(!error.contains("secret"));
        assert!(cache.get("a").unwrap().is_some());
        assert!(cache.get("b").unwrap().is_some());
    }

    #[test]
    fn parallel_connections_stay_within_transactional_capacity() {
        let root = tempfile::tempdir().unwrap();
        let cfg = configuration(root.path(), 25);
        Cache::configured(&cfg, "")
            .unwrap()
            .set("initial", "pool", "aaa")
            .unwrap();
        let barrier = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let cfg = cfg.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    Cache::configured(&cfg, "")
                        .unwrap()
                        .set(&i.to_string(), "pool", "bbb")
                        .unwrap();
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        let stats = stats(&cfg, false, 20).unwrap();
        assert_eq!(stats["cache"]["count"], 5);
        assert_eq!(stats["cache"]["size_bytes"], 25);
    }

    #[test]
    fn malformed_cache_and_fetch_statistics_are_truthful_and_sanitized() {
        let root = tempfile::tempdir().unwrap();
        let cfg = configuration(root.path(), 1024);
        std::fs::write(root.path().join("fetch_cache.db"), b"existing").unwrap();
        assert!(stats(&cfg, false, 20).unwrap()["fetch_cache"]["error"].is_string());
        std::fs::write(root.path().join("cache.db"), b"private corrupt contents").unwrap();
        let error = Cache::configured(&cfg, "")
            .unwrap()
            .get("key")
            .unwrap_err()
            .to_string();
        assert!(!error.contains("private"));
        assert!(!error.contains(&root.path().to_string_lossy().to_string()));
        assert!(stats(&cfg, false, 20).unwrap()["cache"]["error"].is_string());
    }
}
