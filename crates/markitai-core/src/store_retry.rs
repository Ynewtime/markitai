//! Bounded retries for the SQLite stores (the fetch cache, learned browser
//! domains): one attempt is one complete transaction or read, and an attempt
//! that failed for a passing reason is made again for a short while instead of
//! losing the entry it was saving or reading.
use crate::{Error, Result};
use std::time::{Duration, Instant};

/// How long an operation is tried again after transient failures, counted
/// from its first attempt. Lock waits have the store's own busy timeout inside
/// each attempt, so an attempt that waited that long is not repeated.
pub(crate) const RETRY_WINDOW: Duration = Duration::from_secs(5);
const FIRST_RETRY_WAIT: Duration = Duration::from_millis(20);
pub(crate) const LONGEST_RETRY_WAIT: Duration = Duration::from_millis(500);

/// Why one attempt at a store failed. Only the operation's caller sees the
/// store's fixed "unavailable" error; the cause decides whether the attempt is
/// repeated.
#[derive(Debug)]
pub(crate) struct Fault {
    pub(crate) transient: bool,
    /// What failed, for the tests' report; it can name paths and SQL, so
    /// the error callers get never carries it.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) detail: String,
}

pub(crate) type Attempt<T> = std::result::Result<T, Fault>;

impl Fault {
    /// A failure that another attempt cannot change: an invalid entry, a
    /// store that is not one, a link or directory in its place.
    pub(crate) fn permanent(detail: impl Into<String>) -> Self {
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
/// failed one changed nothing. A failure gives `unavailable()`; `_store`
/// names the store in the tests' report.
pub(crate) fn retrying<T>(
    _store: &str,
    unavailable: fn() -> Error,
    mut attempt: impl FnMut() -> Attempt<T>,
) -> Result<T> {
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
                    "{_store} unavailable ({}): {}",
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

#[cfg(test)]
mod tests {
    use super::*;

    fn unavailable() -> Error {
        Error::Conversion("Test store is unavailable".into())
    }

    #[test]
    fn transient_failures_are_retried_for_a_bounded_time_and_others_fail_at_once() {
        let busy = || Fault {
            transient: true,
            detail: "database is locked".into(),
        };
        let mut calls = 0;
        let value = retrying("test store", unavailable, || {
            calls += 1;
            if calls < 3 { Err(busy()) } else { Ok(calls) }
        });
        assert_eq!(value.unwrap(), 3);
        let mut calls = 0;
        let error = retrying("test store", unavailable, || -> Attempt<()> {
            calls += 1;
            Err(Fault::permanent("file is not a database"))
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "Test store is unavailable");
        assert_eq!(calls, 1);
        let started = Instant::now();
        let mut calls = 0;
        assert!(
            retrying("test store", unavailable, || -> Attempt<()> {
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
}
