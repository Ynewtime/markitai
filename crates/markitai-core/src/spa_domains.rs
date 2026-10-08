//! Bounded routing knowledge, independent of cached page content or browser state.
use crate::{Error, Result, config};
use chrono::{DateTime, SecondsFormat, Utc};
use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior, params};
use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use url::Url;

const MAX_DOMAINS: usize = 4096;
const MAX_AUTHORITY: usize = 1024;
const MAX_DATABASE: u64 = 16 * 1024 * 1024;
const TTL: i64 = 30 * 24 * 60 * 60;
const TABLE: &str = "CREATE TABLE IF NOT EXISTS native_spa_domains_v1 (
    domain TEXT PRIMARY KEY NOT NULL CHECK(length(CAST(domain AS BLOB)) BETWEEN 1 AND 1024),
    learned_at INTEGER NOT NULL CHECK(learned_at>=0),
    last_hit INTEGER NOT NULL CHECK(last_hit>=learned_at),
    hits INTEGER NOT NULL CHECK(hits>=1)
);";

#[derive(Clone, Debug, Serialize)]
pub struct DomainInfo {
    pub domain: String,
    pub learned_at: String,
    pub hits: u64,
    pub last_hit: String,
    pub expired: bool,
}

fn unavailable() -> Error {
    Error::Conversion("Learned browser-domain store is unavailable".into())
}

fn path(cfg: &Value) -> PathBuf {
    config::state_path(Path::new(
        cfg.pointer("/cache/global_dir")
            .and_then(Value::as_str)
            .unwrap_or("~/.markitai"),
    ))
    .join("learned_spa_domains.db")
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .min(i64::MAX as u64) as i64
}

fn authority(url: &Url) -> Option<String> {
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return None;
    }
    let domain = url[url::Position::BeforeHost..url::Position::AfterPort].to_ascii_lowercase();
    (!domain.is_empty() && domain.len() <= MAX_AUTHORITY).then_some(domain)
}

fn check_file(path: &Path) -> Result<bool> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Ok(metadata)
            if metadata.is_file()
                && !metadata.file_type().is_symlink()
                && metadata.len() <= MAX_DATABASE =>
        {
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if metadata.nlink() != 1 || metadata.mode() & 0o077 != 0 {
                    return Err(unavailable());
                }
            }
            Ok(true)
        }
        _ => Err(unavailable()),
    }
}

fn open(cfg: &Value, create: bool, write: bool) -> Result<Option<Connection>> {
    let path = path(cfg);
    crate::output::check_path(&path, false).map_err(|_| unavailable())?;
    if !check_file(&path)? {
        if !create {
            return Ok(None);
        }
        let parent = path.parent().ok_or_else(unavailable)?;
        crate::output::create_private_dir(parent).map_err(|_| unavailable())?;
        crate::output::check_path(&path, false).map_err(|_| unavailable())?;
        let temporary = tempfile::NamedTempFile::new_in(parent).map_err(|_| unavailable())?;
        match crate::platform::persist_noclobber(temporary, &path) {
            Ok(_) => {}
            Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(_) => return Err(unavailable()),
        }
        if !check_file(&path)? {
            return Err(unavailable());
        }
    }
    // The shared policy permits root-owned system aliases such as macOS /var.
    // SQLite NOFOLLOW rejects those ancestors too, so resolve only the checked
    // parent and keep the database basename subject to final-file validation.
    crate::output::check_path(&path, false).map_err(|_| unavailable())?;
    let path = path
        .parent()
        .ok_or_else(unavailable)?
        .canonicalize()
        .map_err(|_| unavailable())?
        .join(path.file_name().ok_or_else(unavailable)?);
    if !check_file(&path)? {
        return Err(unavailable());
    }
    for suffix in ["-journal", "-wal", "-shm"] {
        let sibling = path.with_file_name(format!("learned_spa_domains.db{suffix}"));
        check_file(&sibling)?;
    }
    let flags = if write {
        OpenFlags::SQLITE_OPEN_READ_WRITE
    } else {
        OpenFlags::SQLITE_OPEN_READ_ONLY
    };
    let connection = Connection::open_with_flags(path, flags | OpenFlags::SQLITE_OPEN_NOFOLLOW)
        .map_err(|_| unavailable())?;
    connection
        .busy_timeout(Duration::from_secs(3))
        .map_err(|_| unavailable())?;
    connection
        .pragma_update(None, "trusted_schema", "OFF")
        .map_err(|_| unavailable())?;
    if create {
        // The native store uses bounded rollback journals, not a growing WAL.
        let page_size: i64 = connection
            .pragma_query_value(None, "page_size", |row| row.get(0))
            .map_err(|_| unavailable())?;
        let page_size = u64::try_from(page_size)
            .ok()
            .filter(|size| *size > 0)
            .ok_or_else(unavailable)?;
        let pages = MAX_DATABASE
            .checked_div(page_size)
            .filter(|pages| *pages > 0)
            .and_then(|pages| i64::try_from(pages).ok())
            .ok_or_else(unavailable)?;
        connection
            .pragma_update(None, "max_page_count", pages)
            .map_err(|_| unavailable())?;
        connection.execute_batch(TABLE).map_err(|_| unavailable())?;
    }
    Ok(Some(connection))
}

fn timestamp(value: i64) -> Result<String> {
    DateTime::<Utc>::from_timestamp(value, 0)
        .filter(|_| value >= 0)
        .map(|value| value.to_rfc3339_opts(SecondsFormat::Secs, false))
        .ok_or_else(unavailable)
}

fn read_rows(connection: &Connection, time: i64) -> Result<Vec<DomainInfo>> {
    let mut statement = connection.prepare(
        "SELECT CASE WHEN length(CAST(domain AS BLOB))<=1024 THEN domain ELSE NULL END,
         learned_at,hits,last_hit FROM native_spa_domains_v1 ORDER BY hits DESC,domain ASC LIMIT 4097"
    ).map_err(|_| unavailable())?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })
        .map_err(|_| unavailable())?;
    let mut domains = Vec::new();
    for row in rows {
        let (domain, learned, hits, last) = row.map_err(|_| unavailable())?;
        if domains.len() >= MAX_DOMAINS
            || hits < 1
            || last < learned
            || !["http", "https"].iter().any(|scheme| {
                Url::parse(&format!("{scheme}://{domain}/"))
                    .ok()
                    .as_ref()
                    .and_then(authority)
                    .as_deref()
                    == Some(domain.as_str())
            })
        {
            return Err(unavailable());
        }
        domains.push(DomainInfo {
            domain,
            learned_at: timestamp(learned)?,
            hits: hits as u64,
            last_hit: timestamp(last)?,
            expired: time.saturating_sub(last) > TTL,
        });
    }
    Ok(domains)
}

/// Inspect routing records without creating a missing store or changing its age.
pub fn list(cfg: &Value) -> Result<Vec<DomainInfo>> {
    open(cfg, false, false)?.map_or_else(
        || Ok(Vec::new()),
        |connection| read_rows(&connection, now()),
    )
}

/// Validate an existing store before a coordinated multi-store clear.
pub fn preflight_clear(cfg: &Value) -> Result<()> {
    if let Some(mut connection) = open(cfg, false, true)? {
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| unavailable())?;
        read_rows(&transaction, now())?;
        transaction.rollback().map_err(|_| unavailable())?;
    }
    Ok(())
}

/// Remove learned routes only. Cached documents and browser identities are separate.
pub fn clear(cfg: &Value) -> Result<u64> {
    let Some(mut connection) = open(cfg, false, true)? else {
        return Ok(0);
    };
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| unavailable())?;
    read_rows(&transaction, now())?;
    let count = transaction
        .execute("DELETE FROM native_spa_domains_v1", [])
        .map_err(|_| unavailable())?;
    transaction.commit().map_err(|_| unavailable())?;
    Ok(count as u64)
}

/// Consume one routing hint, aging and updating it atomically across processes.
pub(crate) fn take_hint(cfg: &Value, url: &Url) -> Result<bool> {
    take_hint_at(cfg, url, now())
}

fn take_hint_at(cfg: &Value, url: &Url, time: i64) -> Result<bool> {
    let Some(domain) = authority(url) else {
        return Ok(false);
    };
    let Some(mut connection) = open(cfg, false, true)? else {
        return Ok(false);
    };
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| unavailable())?;
    let entry: Option<(i64, i64, i64)> = transaction
        .query_row(
            "SELECT learned_at,last_hit,hits FROM native_spa_domains_v1 WHERE domain=?1",
            [&domain],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()
        .map_err(|_| unavailable())?;
    let Some((learned, last, hits)) = entry else {
        return Ok(false);
    };
    if learned < 0
        || last < learned
        || hits < 1
        || DateTime::<Utc>::from_timestamp(last, 0).is_none()
    {
        return Err(unavailable());
    }
    if time.saturating_sub(last) > TTL {
        transaction
            .execute(
                "DELETE FROM native_spa_domains_v1 WHERE domain=?1",
                [&domain],
            )
            .map_err(|_| unavailable())?;
        transaction.commit().map_err(|_| unavailable())?;
        return Ok(false);
    }
    transaction
        .execute(
            "UPDATE native_spa_domains_v1 SET last_hit=?2,hits=?3 WHERE domain=?1",
            params![domain, time.max(last), hits.saturating_add(1)],
        )
        .map_err(|_| unavailable())?;
    transaction.commit().map_err(|_| unavailable())?;
    Ok(true)
}

pub(crate) fn record_success(cfg: &Value, url: &Url) -> Result<()> {
    record_at(cfg, url, now())
}

fn record_at(cfg: &Value, url: &Url, time: i64) -> Result<()> {
    let Some(domain) = authority(url) else {
        return Ok(());
    };
    let mut connection = open(cfg, true, true)?.ok_or_else(unavailable)?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| unavailable())?;
    // A bounded scan also rejects malformed external rows before an update.
    read_rows(&transaction, time)?;
    transaction
        .execute(
            "DELETE FROM native_spa_domains_v1 WHERE last_hit < ?1",
            [time.saturating_sub(TTL)],
        )
        .map_err(|_| unavailable())?;
    transaction.execute(
        "INSERT INTO native_spa_domains_v1(domain,learned_at,last_hit,hits) VALUES (?1,?2,?2,1)
         ON CONFLICT(domain) DO UPDATE SET last_hit=max(last_hit,excluded.last_hit),hits=min(hits,9223372036854775806)+1",
        params![domain, time],
    ).map_err(|_| unavailable())?;
    transaction.execute(
        "DELETE FROM native_spa_domains_v1 WHERE domain IN (
         SELECT domain FROM native_spa_domains_v1 ORDER BY last_hit DESC,learned_at DESC,domain ASC LIMIT -1 OFFSET 4096)", []
    ).map_err(|_| unavailable())?;
    transaction.commit().map_err(|_| unavailable())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(root: &Path) -> Value {
        serde_json::json!({"cache":{"global_dir":root,"enabled":false,"no_cache":true,"no_cache_patterns":["*"]}})
    }
    fn url(value: &str) -> Url {
        Url::parse(value).unwrap()
    }

    #[test]
    fn missing_management_and_lookups_do_not_create_state() {
        let temp = tempfile::tempdir().unwrap();
        let cfg = cfg(&temp.path().join("absent"));
        assert!(list(&cfg).unwrap().is_empty());
        preflight_clear(&cfg).unwrap();
        assert_eq!(clear(&cfg).unwrap(), 0);
        assert!(!take_hint(&cfg, &url("https://example.test/a")).unwrap());
        assert!(!temp.path().join("absent").exists());
    }

    #[test]
    fn routing_state_is_independent_of_document_cache_flags_and_persists() {
        let temp = tempfile::tempdir().unwrap();
        let cfg = cfg(temp.path());
        record_at(
            &cfg,
            &url("https://Example.test:8443/a?secret=not-stored"),
            10,
        )
        .unwrap();
        assert!(take_hint_at(&cfg, &url("http://example.test:8443/b"), 12).unwrap());
        let rows = list(&cfg).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].domain, "example.test:8443");
        assert_eq!(rows[0].hits, 2);
        assert_eq!(rows[0].learned_at, "1970-01-01T00:00:10+00:00");
        assert_eq!(rows[0].last_hit, "1970-01-01T00:00:12+00:00");
        assert!(rows[0].expired);
        let serialized = serde_json::to_string(&rows).unwrap();
        assert!(!serialized.contains("not-stored"));
        assert!(!path(&cfg).with_file_name("fetch_cache.db").exists());
        assert_eq!(clear(&cfg).unwrap(), 1);
        assert!(list(&cfg).unwrap().is_empty());
    }

    #[test]
    fn expiration_is_idle_and_strictly_after_thirty_days() {
        let temp = tempfile::tempdir().unwrap();
        let cfg = cfg(temp.path());
        let page = url("https://example.test/a");
        record_at(&cfg, &page, 10).unwrap();
        assert!(take_hint_at(&cfg, &page, 10 + TTL).unwrap());
        assert!(take_hint_at(&cfg, &page, 10 + TTL * 2).unwrap());
        assert!(!take_hint_at(&cfg, &page, 11 + TTL * 3).unwrap());
        assert!(list(&cfg).unwrap().is_empty());
    }

    #[test]
    fn authorities_reject_credentials_and_preserve_ports_and_ipv6() {
        let temp = tempfile::tempdir().unwrap();
        let cfg = cfg(temp.path());
        for value in ["file:///tmp/a", "https://user:password@example.test/a"] {
            record_at(&cfg, &url(value), 10).unwrap();
            assert!(!take_hint_at(&cfg, &url(value), 11).unwrap());
        }
        assert!(!path(&cfg).exists());
        for value in [
            "https://example.test:80/a",
            "http://example.test:443/a",
            "http://[::1]:1234/a",
        ] {
            record_at(&cfg, &url(value), 10).unwrap();
            assert!(take_hint_at(&cfg, &url(value), 11).unwrap());
        }
        assert_eq!(list(&cfg).unwrap().len(), 3);
        assert!(!take_hint_at(&cfg, &url("https://example.test:81/b"), 12).unwrap());
    }

    #[test]
    fn capacity_evicts_oldest_without_dropping_newest_record() {
        let temp = tempfile::tempdir().unwrap();
        let cfg = cfg(temp.path());
        record_at(&cfg, &url("https://oldest.test/a"), 1).unwrap();
        let mut connection = open(&cfg, false, true).unwrap().unwrap();
        let transaction = connection.transaction().unwrap();
        for index in 1..MAX_DOMAINS {
            transaction
                .execute(
                    "INSERT INTO native_spa_domains_v1 VALUES (?1,?2,?2,1)",
                    params![format!("site{index}.test"), index as i64 + 1],
                )
                .unwrap();
        }
        transaction.commit().unwrap();
        record_at(&cfg, &url("https://newest.test/a"), 10_000).unwrap();
        let rows = list(&cfg).unwrap();
        assert_eq!(rows.len(), MAX_DOMAINS);
        assert!(rows.iter().any(|entry| entry.domain == "newest.test"));
        assert!(!rows.iter().any(|entry| entry.domain == "oldest.test"));
    }

    #[test]
    fn concurrent_learning_and_hits_do_not_lose_updates() {
        let temp = tempfile::tempdir().unwrap();
        let cfg = cfg(temp.path());
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let cfg = &cfg;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let page = url("https://example.test/a");
                    record_at(cfg, &page, 10).unwrap();
                    assert!(take_hint_at(cfg, &page, 11).unwrap());
                });
            }
        });
        assert_eq!(list(&cfg).unwrap()[0].hits, 16);
    }

    #[test]
    fn failed_insert_preserves_eviction_and_existing_rows_atomically() {
        let temp = tempfile::tempdir().unwrap();
        let cfg = cfg(temp.path());
        record_at(&cfg, &url("https://old.test/a"), 10).unwrap();
        let connection = open(&cfg, false, true).unwrap().unwrap();
        connection.execute_batch("CREATE TRIGGER reject_new BEFORE INSERT ON native_spa_domains_v1 BEGIN SELECT RAISE(ABORT,'private SQL and credentials'); END").unwrap();
        let error = record_at(&cfg, &url("https://new.test/a"), TTL * 2)
            .unwrap_err()
            .to_string();
        assert_eq!(error, "Learned browser-domain store is unavailable");
        assert_eq!(list(&cfg).unwrap()[0].domain, "old.test");
        assert_eq!(clear(&cfg).unwrap(), 1);
    }

    #[test]
    fn corrupt_or_excessive_rows_are_not_trusted() {
        let temp = tempfile::tempdir().unwrap();
        let cfg = cfg(temp.path());
        record_at(&cfg, &url("https://example.test/a"), 10).unwrap();
        let connection = open(&cfg, false, true).unwrap().unwrap();
        connection
            .execute(
                "UPDATE native_spa_domains_v1 SET domain='private.test/path?credential=secret'",
                [],
            )
            .unwrap();
        assert_eq!(
            list(&cfg).unwrap_err().to_string(),
            "Learned browser-domain store is unavailable"
        );
        drop(connection);
        std::fs::write(path(&cfg), b"private broken SQLite bytes").unwrap();
        assert_eq!(
            clear(&cfg).unwrap_err().to_string(),
            "Learned browser-domain store is unavailable"
        );
    }

    #[cfg(unix)]
    #[test]
    fn private_permissions_and_symlink_rejection_protect_state_files() {
        use std::os::unix::fs::{MetadataExt, symlink};
        let temp = tempfile::tempdir().unwrap();
        let cfg = cfg(&temp.path().join("state"));
        record_at(&cfg, &url("https://example.test/a"), 10).unwrap();
        assert_eq!(std::fs::metadata(path(&cfg)).unwrap().mode() & 0o077, 0);
        assert_eq!(
            std::fs::metadata(path(&cfg).parent().unwrap())
                .unwrap()
                .mode()
                & 0o077,
            0
        );
        let mut physical = cfg.clone();
        physical["cache"]["global_dir"] =
            serde_json::json!(path(&cfg).parent().unwrap().canonicalize().unwrap());
        assert_eq!(
            list(&physical).unwrap()[0].domain,
            list(&cfg).unwrap()[0].domain
        );
        let alias = temp.path().join("user-alias");
        symlink(path(&cfg).parent().unwrap(), &alias).unwrap();
        let mut aliased = cfg.clone();
        aliased["cache"]["global_dir"] = serde_json::json!(alias);
        assert!(list(&aliased).is_err());
        assert!(record_at(&aliased, &url("https://other.test/a"), 11).is_err());
        assert_eq!(list(&cfg).unwrap().len(), 1);
        let original = path(&cfg).with_extension("original");
        std::fs::rename(path(&cfg), &original).unwrap();
        symlink(&original, path(&cfg)).unwrap();
        assert!(list(&cfg).is_err());
        assert!(clear(&cfg).is_err());
        assert!(original.is_file());
    }
}
