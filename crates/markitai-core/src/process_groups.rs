//! Process groups that this library starts for external runtimes.
//!
//! Chromium, LibreOffice and the official subscription runtimes each run in a
//! new process group so that their whole tree can be stopped. A terminal
//! interrupt is delivered to the foreground group only, so it never reaches
//! them. A host about to terminate calls [`terminate_all`] first. It only loads
//! atomics and calls kill(2), both async-signal-safe, so a signal handler may
//! call it. Caller-held browser sessions are never registered.

use std::process::Child;
use std::sync::atomic::{AtomicI32, Ordering};

const SLOTS: usize = 128;
const RESERVED: i32 = -1;
static GROUPS: [AtomicI32; SLOTS] = [const { AtomicI32::new(0) }; SLOTS];

/// One registry entry, reserved before spawning so an unregistered group
/// never runs. Dropping it releases the entry.
pub(crate) struct Slot(usize);

impl Slot {
    pub(crate) fn reserve() -> Option<Self> {
        GROUPS
            .iter()
            .position(|slot| {
                slot.compare_exchange(0, RESERVED, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            })
            .map(Self)
    }

    /// Address the new group led by `child`. The leader's id cannot be reused
    /// until it is reaped, so [`Slot::retire`] must precede every reap.
    pub(crate) fn publish(&self, child: &Child) {
        if let Ok(pid) = i32::try_from(child.id())
            && pid > 0
        {
            GROUPS[self.0].store(pid, Ordering::Release);
        }
    }

    /// Stop addressing the group, keeping the entry reserved.
    pub(crate) fn retire(&self) {
        GROUPS[self.0].store(RESERVED, Ordering::Release);
    }

    /// Whether the leader has exited, without reaping it.
    #[cfg_attr(not(unix), allow(dead_code))]
    pub(crate) fn exited(&self, child: &Child) -> bool {
        leader_exited(child)
    }

    /// Nonblocking reap that retires the entry first. A leader that has not
    /// exited stays registered.
    pub(crate) fn try_reap(
        &self,
        child: &mut Child,
    ) -> std::io::Result<Option<std::process::ExitStatus>> {
        if !leader_exited(child) {
            return Ok(None);
        }
        self.retire();
        child.try_wait()
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        GROUPS[self.0].store(0, Ordering::Release);
    }
}

/// Whether the leader has exited, leaving it unreaped (Unix `WNOWAIT`).
/// Elsewhere it reports false and callers fall back to their blocking paths.
fn leader_exited(child: &Child) -> bool {
    #[cfg(unix)]
    {
        let pid: libc::id_t = child.id();
        loop {
            // A report sets si_signo to SIGCHLD; no waitable child leaves zeros.
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            let flags = libc::WEXITED | libc::WNOHANG | libc::WNOWAIT;
            if unsafe { libc::waitid(libc::P_PID, pid, &mut info, flags) } == 0 {
                return info.si_signo == libc::SIGCHLD;
            }
            if std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
                // Already reaped (ECHILD) or unknown: nothing is left to address.
                return true;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = child;
        false
    }
}

/// Immediately kill every registered process group. Async-signal-safe.
pub fn terminate_all() {
    #[cfg(unix)]
    for slot in &GROUPS {
        let pid = slot.load(Ordering::Acquire);
        if pid > 0 {
            unsafe {
                libc::kill(-pid, libc::SIGKILL);
            }
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::process::CommandExt;
    use std::time::{Duration, Instant};

    fn alive(pid: i32) -> bool {
        unsafe { libc::kill(pid, 0) == 0 }
    }

    #[test]
    fn exited_leader_is_retired_before_reap_and_live_leader_stays_registered() {
        let slot = Slot::reserve().unwrap();
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .process_group(0)
            .spawn()
            .unwrap();
        slot.publish(&child);
        let pid = child.id() as i32;
        assert_eq!(GROUPS[slot.0].load(Ordering::Acquire), pid);
        assert!(slot.try_reap(&mut child).unwrap().is_none());
        assert_eq!(GROUPS[slot.0].load(Ordering::Acquire), pid);
        // Only this test's own group is signalled; never the global registry.
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        let status = loop {
            if let Some(status) = slot.try_reap(&mut child).unwrap() {
                break status;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(!status.success());
        assert_eq!(GROUPS[slot.0].load(Ordering::Acquire), RESERVED);
        drop(slot);
        assert!(!alive(pid));
    }

    #[test]
    fn terminate_all_kills_a_registered_tree_in_an_isolated_process() {
        // terminate_all is process-wide; run it only in a dedicated child test
        // process so concurrently running tests keep their own runtimes.
        const NAME: &str =
            "process_groups::tests::terminate_all_kills_a_registered_tree_in_an_isolated_process";
        if std::env::var_os("MARKITAI_PROCESS_GROUP_CHILD").is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", NAME, "--test-threads", "1", "--nocapture"])
                .env("MARKITAI_PROCESS_GROUP_CHILD", "1")
                .output()
                .unwrap();
            let text = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success(),
                "{text}{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(text.contains("1 passed"), "{text}");
            return;
        }
        let slot = Slot::reserve().unwrap();
        // The leader starts a grandchild in the same group and waits on it.
        let mut child = std::process::Command::new("sh")
            .args(["-c", "sleep 30 & echo $! ; wait"])
            .stdout(std::process::Stdio::piped())
            .process_group(0)
            .spawn()
            .unwrap();
        slot.publish(&child);
        let mut line = String::new();
        std::io::BufRead::read_line(
            &mut std::io::BufReader::new(child.stdout.take().unwrap()),
            &mut line,
        )
        .unwrap();
        let grandchild: i32 = line.trim().parse().unwrap();
        assert!(alive(grandchild));
        terminate_all();
        let deadline = Instant::now() + Duration::from_secs(10);
        while slot.try_reap(&mut child).unwrap().is_none() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(10));
        }
        // The grandchild was reparented; poll until the kernel removes it.
        while alive(grandchild) {
            assert!(Instant::now() < deadline, "grandchild survived");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn registry_capacity_is_bounded_and_released() {
        const NAME: &str = "process_groups::tests::registry_capacity_is_bounded_and_released";
        if std::env::var_os("MARKITAI_PROCESS_GROUP_CHILD").is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", NAME, "--test-threads", "1"])
                .env("MARKITAI_PROCESS_GROUP_CHILD", "1")
                .output()
                .unwrap();
            let text = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success(),
                "{text}{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(text.contains("1 passed"), "{text}");
            return;
        }
        let slots: Vec<_> = std::iter::from_fn(Slot::reserve).take(SLOTS + 1).collect();
        assert_eq!(slots.len(), SLOTS);
        assert!(Slot::reserve().is_none());
        drop(slots);
        assert!(Slot::reserve().is_some());
    }
}
