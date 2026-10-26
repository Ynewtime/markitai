//! A running `markitai serve` and its restart.
//!
//! The server writes a per-launch secret, its address and version to
//! `MARKITAI_HOME/serve/control` (owner-only). Whoever can read that file is
//! the person running the server: `markitai update` sends the secret to the
//! server's loopback address to ask it how much work is running and to
//! restart into the new version (`POST /api/update/restart`). The file
//! permission is the authorisation, the same on every system (Windows: an
//! ACL for the user alone, not the folder's inherited one).
//!
//! A restart into a new version leaves `MARKITAI_HOME/update-pending.json`
//! until the new server is ready: what was swapped and where the old program
//! waits. A new server that fails before it is ready (or a crash that leaves
//! the marker for the next start) puts the old version back. The marker names
//! the server it is for: only that one (handed the marker's id) acts on it
//! while it runs, so another `serve` of the same home leaves it alone.
use super::install::Swapped;
use serde::{Deserialize, Serialize};
use std::io;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

const FORMAT: u32 = 1;
/// The marker id a restarted server is handed: the marker is its own.
pub(crate) const UPDATE_ID: &str = "MARKITAI_SERVE_UPDATE_ID";
/// The header the control requests carry the secret in.
pub(crate) const HEADER: &str = "x-markitai-control";
/// The last release without restart handover: a server of that version or
/// older cannot be restarted in place, and is never exec'ed with handover
/// state (it would try to bind the port it already holds).
pub(crate) const LAST_WITHOUT_HANDOVER: &str = "2.1.0";

pub(crate) fn supports_handover(version: &super::version::Version) -> bool {
    super::version::Version::parse(LAST_WITHOUT_HANDOVER).is_some_and(|last| *version > last)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Control {
    pub(crate) format: u32,
    pub(crate) pid: u32,
    /// The address the server listens on.
    pub(crate) address: String,
    pub(crate) version: String,
    pub(crate) secret: String,
    /// The program the server runs (and restarts into); absent from an older server.
    #[serde(default)]
    pub(crate) exe: Option<PathBuf>,
}

pub(crate) fn path(home: &Path) -> PathBuf {
    home.join("serve").join("control")
}

fn write_private(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    markitai_core::output::create_private_dir(dir)?;
    let mut temporary = tempfile::NamedTempFile::new_in(dir)?;
    // Before the secret is written: 0600 on Unix, the user's own ACL on Windows.
    markitai_core::platform::restrict_to_owner(temporary.path())?;
    io::Write::write_all(&mut temporary, bytes)?;
    temporary.as_file().sync_all()?;
    markitai_core::platform::persist(temporary, path)
        .map(drop)
        .map_err(|error| error.error)
}

fn read_json<T: for<'a> Deserialize<'a>>(path: &Path) -> Option<T> {
    let file = markitai_core::platform::open_read(path, false).ok()?;
    let bytes = markitai_core::platform::read_limited(file, 64 * 1024).ok()??;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    if value["format"].as_u64() != Some(u64::from(FORMAT)) {
        return None;
    }
    serde_json::from_value(value).ok()
}

impl Control {
    pub(crate) fn new(address: SocketAddr, secret: String) -> Self {
        Self {
            format: FORMAT,
            pid: std::process::id(),
            address: address.to_string(),
            version: super::current_identity().0,
            secret,
            exe: std::env::current_exe().ok(),
        }
    }

    /// The same, naming the program the server restarts into.
    pub(crate) fn with_exe(mut self, exe: PathBuf) -> Self {
        self.exe = Some(exe);
        self
    }

    /// Whether the server runs `program` (this install), so a restart starts the new version.
    pub(crate) fn runs(&self, program: &Path) -> bool {
        self.exe.as_deref().is_some_and(|exe| {
            match (std::fs::canonicalize(exe), std::fs::canonicalize(program)) {
                (Ok(left), Ok(right)) => left == right,
                _ => exe == program,
            }
        })
    }

    pub(crate) fn write(&self, home: &Path) -> io::Result<()> {
        write_private(
            &path(home),
            &serde_json::to_vec_pretty(self).map_err(io::Error::other)?,
        )
    }

    pub(crate) fn read(home: &Path) -> Option<Self> {
        read_json(&path(home))
    }

    /// Remove the file if it is still this server's.
    pub(crate) fn remove(&self, home: &Path) {
        if Self::read(home).is_some_and(|found| found.secret == self.secret) {
            let _ = std::fs::remove_file(path(home));
        }
    }

    /// Where this computer reaches the server: its own address when that
    /// is loopback, loopback for a server on every interface; `None` for a
    /// server on one other interface, which only answers that address.
    pub(crate) fn local_address(&self) -> Option<SocketAddr> {
        let mut address: SocketAddr = self.address.parse().ok()?;
        if address.ip().is_unspecified() {
            address.set_ip(match address.ip() {
                std::net::IpAddr::V4(_) => std::net::Ipv4Addr::LOCALHOST.into(),
                std::net::IpAddr::V6(_) => std::net::Ipv6Addr::LOCALHOST.into(),
            });
        }
        address.ip().is_loopback().then_some(address)
    }
}

/// What a running server says about itself.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub(crate) struct Status {
    pub(crate) version: String,
    pub(crate) pid: u32,
    /// Conversions running or queued: they finish before a restart.
    pub(crate) waiting_jobs: u64,
    /// The update this server undid as it started (its failure file is
    /// gone by then).
    #[serde(default)]
    pub(crate) undone: Option<Failed>,
}

fn client(timeout: Duration) -> Result<reqwest::blocking::Client, String> {
    markitai_core::http_client_builder()
        .map_err(|error| error.to_string())?
        .no_proxy()
        .connect_timeout(Duration::from_secs(2))
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
        .user_agent(super::check::USER_AGENT)
        .build()
        .map_err(|error| error.to_string())
}

/// The running server this home's control file names, if it answers.
pub(crate) fn running(home: &Path) -> Option<(Control, Status)> {
    let control = Control::read(home)?;
    let address = control.local_address()?;
    let response = client(Duration::from_secs(3))
        .ok()?
        .get(format!("http://{address}/api/update/control"))
        .header(HEADER, &control.secret)
        .send()
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    let status: Status = response.json().ok()?;
    (status.pid == control.pid).then_some((control, status))
}

/// Ask the server to finish its conversions and restart.
pub(crate) fn request_restart(control: &Control) -> Result<(), String> {
    let address = control
        .local_address()
        .ok_or_else(|| "the server is not on a loopback address".to_owned())?;
    let response = client(Duration::from_secs(5))?
        .post(format!("http://{address}/api/update/restart"))
        .header(HEADER, &control.secret)
        .send()
        .map_err(|error| error.to_string())?;
    if response.status().is_success() {
        Ok(())
    } else {
        Err(format!(
            "the server answered HTTP {}",
            response.status().as_u16()
        ))
    }
}

/// How waiting for a restart ended.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Restart {
    /// A server of the new version answers.
    Done(Status),
    /// The new version did not start; the old one is back.
    RolledBack(String),
    /// Still waiting (the time ran out or the wait was cancelled); the
    /// server restarts by itself once its conversions finish.
    Pending,
    /// A server of another version answers: not this install's (no rollback happened).
    Other(String),
}

/// Wait until a server of `version` answers in place of `before`, telling
/// `waiting(count)` how many conversions the old one still finishes.
pub(crate) fn wait_for_restart(
    home: &Path,
    before: &Control,
    version: &str,
    timeout: Duration,
    waiting: &mut dyn FnMut(u64),
    cancelled: &dyn Fn() -> bool,
) -> Restart {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline && !cancelled() {
        if let Some(failure) = Pending::failure(home)
            && failure.to == version
        {
            return Restart::RolledBack(failure.reason);
        }
        if let Some((control, status)) = running(home) {
            if control.secret != before.secret {
                if let Some(failure) = status
                    .undone
                    .as_ref()
                    .filter(|failure| failure.to == version)
                {
                    return Restart::RolledBack(failure.reason.clone());
                }
                if status.version == version {
                    return Restart::Done(status);
                }
                return Restart::Other(status.version);
            }
            waiting(status.waiting_jobs);
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    Restart::Pending
}

/// An update whose new server has not reported ready yet.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Pending {
    pub(crate) format: u32,
    pub(crate) from: String,
    pub(crate) to: String,
    /// The program to start (the install's path).
    pub(crate) exe: PathBuf,
    pub(crate) swapped: Swapped,
    /// Set by the new server as it starts: a second start that finds it set
    /// means the first never got ready.
    pub(crate) attempted: bool,
    /// Handed to the restarted server (`UPDATE_ID`).
    #[serde(default)]
    pub(crate) id: String,
    /// The process the marker is for: the server restarting, then the new
    /// server once it has started. While it runs, no other start touches the marker.
    #[serde(default)]
    pub(crate) server: Option<u32>,
}

/// Why the last update was undone, for the old server to tell.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Failed {
    pub(crate) format: u32,
    pub(crate) from: String,
    pub(crate) to: String,
    pub(crate) reason: String,
}

pub(crate) const PENDING: &str = "update-pending.json";
const FAILED: &str = "update-failed.json";

impl Pending {
    /// The marker of an update `server` (a PID) restarts into.
    pub(crate) fn new(from: &str, to: &str, exe: PathBuf, swapped: Swapped, server: u32) -> Self {
        Self {
            format: FORMAT,
            from: from.into(),
            to: to.into(),
            exe,
            swapped,
            attempted: false,
            id: super::install::new_id(),
            server: Some(server),
        }
    }

    pub(crate) fn save(&self, home: &Path) -> io::Result<()> {
        write_private(
            &home.join(PENDING),
            &serde_json::to_vec_pretty(self).map_err(io::Error::other)?,
        )
    }

    pub(crate) fn load(home: &Path) -> Option<Self> {
        read_json(&home.join(PENDING))
    }

    pub(crate) fn clear(home: &Path) {
        let _ = std::fs::remove_file(home.join(PENDING));
    }

    /// The new server is ready: the old files go.
    pub(crate) fn commit(&self, home: &Path) {
        self.swapped.commit();
        Self::clear(home);
        let _ = std::fs::remove_file(home.join(FAILED));
    }

    /// Put the old version back and record why.
    pub(crate) fn roll_back(&self, home: &Path, reason: &str) -> io::Result<()> {
        let restored = self.swapped.rollback();
        Self::clear(home);
        let failed = Failed {
            format: FORMAT,
            from: self.from.clone(),
            to: self.to.clone(),
            reason: reason.to_owned(),
        };
        let _ = write_private(
            &home.join(FAILED),
            &serde_json::to_vec_pretty(&failed).map_err(io::Error::other)?,
        );
        restored
    }

    /// The failure an earlier rollback recorded (read once by the old
    /// server as it starts again).
    pub(crate) fn failure(home: &Path) -> Option<Failed> {
        read_json(&home.join(FAILED))
    }

    pub(crate) fn take_failure(home: &Path) -> Option<Failed> {
        let failed = Self::failure(home);
        let _ = std::fs::remove_file(home.join(FAILED));
        failed
    }
}

/// What a server that starts finds in the marker.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Start {
    /// No update in flight.
    Normal,
    /// This is the new version's first start: carry on, commit when ready.
    New(Pending),
    /// This new version already failed to get ready once: roll back.
    Crashed(Pending),
}

/// Read and advance the marker for a server of `version` starting, handed
/// the marker id `handed` by the server it restarts.
pub(crate) fn on_start(home: &Path, version: &str, handed: Option<&str>) -> Start {
    let Some(mut pending) = Pending::load(home) else {
        return Start::Normal;
    };
    let own = handed.is_some_and(|id| !id.is_empty() && id == pending.id);
    let me = std::process::id();
    if !own && pending.server.is_some_and(|pid| pid != me && alive(pid)) {
        // Another server's update, still in its hands.
        return Start::Normal;
    }
    if pending.to != version {
        // The old version runs again (rolled back by hand, or reinstalled).
        Pending::clear(home);
        return Start::Normal;
    }
    if pending.attempted {
        return Start::Crashed(pending);
    }
    pending.attempted = true;
    pending.server = Some(me);
    if pending.save(home).is_err() {
        return Start::Normal;
    }
    Start::New(pending)
}

/// Whether process `pid` still runs.
fn alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        let Ok(pid) = libc::pid_t::try_from(pid) else {
            return false;
        };
        // SAFETY: signal 0 only checks that the process exists.
        let found = unsafe { libc::kill(pid, 0) } == 0;
        found || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_TIMEOUT};
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject,
        };
        // SAFETY: the handle is checked, waited on without blocking and closed once.
        unsafe {
            let process = OpenProcess(PROCESS_SYNCHRONIZE, 0, pid);
            if process.is_null() {
                return false;
            }
            let running = WaitForSingleObject(process, 0) == WAIT_TIMEOUT;
            CloseHandle(process);
            running
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_control_file_is_owner_only_and_removed_only_by_its_server() {
        let home = tempfile::tempdir().unwrap();
        let control = Control::new("127.0.0.1:3600".parse().unwrap(), "s1".into());
        control.write(home.path()).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(path(home.path()))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        assert_eq!(Control::read(home.path()), Some(control.clone()));
        let other = Control::new("127.0.0.1:3601".parse().unwrap(), "s2".into());
        other.write(home.path()).unwrap();
        control.remove(home.path());
        assert_eq!(Control::read(home.path()), Some(other.clone()));
        other.remove(home.path());
        assert_eq!(Control::read(home.path()), None);
    }

    #[test]
    fn only_loopback_or_every_interface_is_reached_locally() {
        let at = |address: &str| Control {
            address: address.into(),
            ..Control::new("127.0.0.1:1".parse().unwrap(), String::new())
        };
        assert_eq!(
            at("0.0.0.0:3600").local_address(),
            Some("127.0.0.1:3600".parse().unwrap())
        );
        assert_eq!(
            at("[::]:3600").local_address(),
            Some("[::1]:3600".parse().unwrap())
        );
        assert_eq!(
            at("127.0.0.1:3600").local_address(),
            Some("127.0.0.1:3600".parse().unwrap())
        );
        assert_eq!(at("192.168.1.20:3600").local_address(), None);
    }

    #[test]
    fn a_new_server_that_never_got_ready_is_rolled_back_on_its_next_start() {
        let home = tempfile::tempdir().unwrap();
        let program = home.path().join("markitai");
        let kept = home.path().join(".markitai.old-x");
        std::fs::write(&program, "new").unwrap();
        std::fs::write(&kept, "old").unwrap();
        let pending = Pending::new(
            "2.1.0",
            "2.2.0",
            program.clone(),
            Swapped {
                aside: vec![(program.clone(), kept.clone())],
                placed: Vec::new(),
                old_program: Some(kept.clone()),
            },
            std::process::id(),
        );
        pending.save(home.path()).unwrap();
        // The first start of 2.2.0 carries on.
        let Start::New(first) = on_start(home.path(), "2.2.0", None) else {
            panic!()
        };
        assert!(first.attempted);
        // It crashed before ready; the next start rolls back.
        let Start::Crashed(again) = on_start(home.path(), "2.2.0", None) else {
            panic!()
        };
        again.roll_back(home.path(), "crashed").unwrap();
        assert_eq!(std::fs::read_to_string(&program).unwrap(), "old");
        assert!(!kept.exists());
        assert_eq!(on_start(home.path(), "2.2.0", None), Start::Normal);
        let failed = Pending::take_failure(home.path()).unwrap();
        assert_eq!(
            (failed.to.as_str(), failed.reason.as_str()),
            ("2.2.0", "crashed")
        );
        assert_eq!(Pending::take_failure(home.path()), None);
        // A ready server commits: the old program goes, the marker too.
        std::fs::write(&kept, "old").unwrap();
        pending.save(home.path()).unwrap();
        let Start::New(ready) = on_start(home.path(), "2.2.0", None) else {
            panic!()
        };
        ready.commit(home.path());
        assert!(!kept.exists() && Pending::load(home.path()).is_none());
        // The old version starting with a marker drops it.
        pending.save(home.path()).unwrap();
        assert_eq!(on_start(home.path(), "2.1.0", None), Start::Normal);
        assert!(Pending::load(home.path()).is_none());
    }

    #[test]
    fn another_servers_marker_is_left_alone_while_that_server_runs() {
        let home = tempfile::tempdir().unwrap();
        let program = home.path().join("markitai");
        let kept = home.path().join(".markitai.old-x");
        std::fs::write(&program, "new").unwrap();
        std::fs::write(&kept, "old").unwrap();
        // A running process that is not this one: the server the marker is for.
        let mut other = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "update::control::tests::helper_waits_for_its_input",
                "--test-threads=1",
            ])
            .env("MARKITAI_TEST_CONTROL_HELPER", "1")
            .stdout(std::process::Stdio::null())
            .stdin(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let swapped = Swapped {
            aside: vec![(program.clone(), kept.clone())],
            placed: Vec::new(),
            old_program: Some(kept.clone()),
        };
        let pending = Pending::new("2.1.0", "2.2.0", program.clone(), swapped, other.id());
        pending.save(home.path()).unwrap();
        // A second serve of either version neither takes nor drops it.
        assert_eq!(on_start(home.path(), "2.2.0", None), Start::Normal);
        assert_eq!(
            on_start(home.path(), "2.1.0", Some("another")),
            Start::Normal
        );
        assert_eq!(Pending::load(home.path()), Some(pending.clone()));
        // The server handed its id does.
        let Start::New(first) = on_start(home.path(), "2.2.0", Some(&pending.id)) else {
            panic!()
        };
        assert_eq!(first.server, Some(std::process::id()));
        // Once the server it names is gone, the next start may undo it.
        let mut stale = pending.clone();
        stale.attempted = true;
        stale.server = Some(other.id());
        stale.save(home.path()).unwrap();
        assert_eq!(on_start(home.path(), "2.2.0", None), Start::Normal);
        let _ = other.kill();
        let _ = other.wait();
        assert!(matches!(
            on_start(home.path(), "2.2.0", None),
            Start::Crashed(_)
        ));
        assert_eq!(std::fs::read_to_string(&kept).unwrap(), "old");
    }

    /// The process the test above stands in for a running server.
    #[test]
    fn helper_waits_for_its_input() {
        if std::env::var_os("MARKITAI_TEST_CONTROL_HELPER").is_some() {
            let _ = std::io::Read::read_to_end(&mut std::io::stdin(), &mut Vec::new());
        }
    }

    #[test]
    fn handover_is_only_offered_to_versions_that_have_it() {
        let v = |text| super::super::version::Version::parse(text).unwrap();
        assert!(!supports_handover(&v("2.1.0")));
        assert!(!supports_handover(&v("2.0.1")));
        assert!(supports_handover(&v("2.1.1")));
        assert!(supports_handover(&v("2.2.0")));
    }
}
