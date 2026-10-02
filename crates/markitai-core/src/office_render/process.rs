use super::{Result, failure};
use std::{
    fs,
    path::Path,
    process::{Child, Command, Stdio},
    sync::{Condvar, Mutex},
    time::{Duration, Instant},
};

static ACTIVE: Mutex<usize> = Mutex::new(0);
static READY: Condvar = Condvar::new();

pub(super) struct Permit;
impl Drop for Permit {
    fn drop(&mut self) {
        let mut active = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
        *active -= 1;
        READY.notify_one();
    }
}

pub(super) fn acquire(deadline: Instant) -> Result<Permit> {
    let mut active = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
    while *active >= 2 {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| failure("timed out waiting for an Office export slot"))?;
        let (next, status) = READY
            .wait_timeout(active, remaining)
            .unwrap_or_else(|e| e.into_inner());
        active = next;
        if status.timed_out() {
            return Err(failure("timed out waiting for an Office export slot"));
        }
    }
    *active += 1;
    Ok(Permit)
}

struct Running {
    child: Child,
    group: crate::process_groups::Slot,
    reaped: bool,
}
impl Drop for Running {
    fn drop(&mut self) {
        if self.reaped {
            return;
        }
        // A dedicated tree also ends the real soffice behind a launcher or wrapper.
        self.group.kill_tree(&self.child);
        self.group.retire();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn check_output(directory: &Path, limit: u64) -> Result<()> {
    let mut bytes = 0u64;
    let mut entries = 0usize;
    for entry in fs::read_dir(directory)? {
        entries += 1;
        let metadata = fs::symlink_metadata(entry?.path())?;
        if entries > 8 || !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(failure("export produced unexpected non-regular output"));
        }
        bytes = bytes
            .checked_add(metadata.len())
            .ok_or_else(|| failure("export size overflow"))?;
        if bytes > limit {
            return Err(failure("export exceeded its byte limit"));
        }
    }
    Ok(())
}

pub(super) fn convert(
    program: &Path,
    input: &Path,
    output: &Path,
    profile: &Path,
    filter: &str,
    deadline: Instant,
    limit: u64,
) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(failure("Office export timed out"));
    }
    let mut command = private_command(program, profile)?;
    command
        .arg("--convert-to")
        .arg(filter)
        .arg("--outdir")
        .arg(output)
        .arg(input);
    wait(command, Some((output, limit)), deadline)
}

pub(super) fn diagnose(program: &Path, profile: &Path, deadline: Instant) -> Result<()> {
    let mut command = private_command(program, profile)?;
    command.arg("--version");
    wait(command, None, deadline)
}

fn private_command(program: &Path, profile: &Path) -> Result<Command> {
    let profile_url = url::Url::from_directory_path(profile)
        .map_err(|_| failure("invalid private profile path"))?;
    let mut command = Command::new(program);
    command
        .args([
            "--headless",
            "--nologo",
            "--nodefault",
            "--nolockcheck",
            "--nofirststartwizard",
            "--norestore",
        ])
        .arg(format!("-env:UserInstallation={profile_url}"))
        .current_dir(profile)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // Do not inherit credentials, model settings, Python hooks or the user's LO profile.
    // PATH is needed by platform launchers; HOME itself is never reassigned.
    let path = std::env::var_os("PATH");
    let system_root = std::env::var_os("SystemRoot");
    command.env_clear();
    if let Some(value) = path {
        command.env("PATH", value);
    }
    if let Some(value) = system_root {
        command.env("SystemRoot", value);
    }
    // The profile itself is `-env:UserInstallation` on every platform. The
    // temporary variables cover LibreOffice and its libraries everywhere; on
    // Unix, libraries such as fontconfig also write caches under
    // XDG_CACHE_HOME, which would otherwise fall back to the unset HOME.
    command
        .env("TMPDIR", profile)
        .env("TMP", profile)
        .env("TEMP", profile);
    #[cfg(unix)]
    command.env("XDG_CACHE_HOME", profile.join("cache"));
    Ok(command)
}

fn wait(mut command: Command, output: Option<(&Path, u64)>, deadline: Instant) -> Result<()> {
    if Instant::now() >= deadline {
        return Err(failure("Office export timed out"));
    }
    let group = crate::process_groups::Slot::reserve()
        .ok_or_else(|| failure("too many external runtime process groups are active"))?;
    let child = group
        .spawn(&mut command)
        .map_err(|_| failure("LibreOffice could not be started"))?;
    let mut running = Running {
        child,
        group,
        reaped: false,
    };
    loop {
        if let Some((directory, limit)) = output {
            check_output(directory, limit)?;
        }
        if let Some(status) = running
            .group
            .try_reap(&mut running.child)
            .map_err(|_| failure("LibreOffice process status is unavailable"))?
        {
            running.reaped = true;
            if !status.success() {
                return Err(failure("LibreOffice export failed"));
            }
            if let Some((directory, limit)) = output {
                check_output(directory, limit)?;
            }
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(failure("Office export timed out"));
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

#[cfg(test)]
mod private_profile_tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn every_platform_passes_the_private_profile_and_no_inherited_environment() {
        let workspace = tempfile::tempdir().unwrap();
        let profile = workspace.path().join("profile");
        fs::create_dir(&profile).unwrap();
        let command = private_command(Path::new("soffice"), &profile).unwrap();
        let expected = url::Url::from_directory_path(&profile).unwrap();
        let installation = format!("-env:UserInstallation={expected}");
        assert!(
            expected.as_str().starts_with("file:///") && expected.as_str().ends_with("/profile/")
        );
        assert!(
            command
                .get_args()
                .any(|arg| arg == OsStr::new(&installation))
        );
        let envs = command
            .get_envs()
            .filter_map(|(key, value)| Some((key.to_str()?.to_owned(), value?.to_owned())))
            .collect::<std::collections::BTreeMap<_, _>>();
        for key in ["TMPDIR", "TMP", "TEMP"] {
            assert_eq!(
                envs.get(key).map(|value| value.as_os_str()),
                Some(profile.as_os_str())
            );
        }
        assert_eq!(envs.contains_key("XDG_CACHE_HOME"), cfg!(unix));
        let allowed = [
            "PATH",
            "SystemRoot",
            "TMPDIR",
            "TMP",
            "TEMP",
            "XDG_CACHE_HOME",
        ];
        assert!(
            envs.keys().all(|key| allowed.contains(&key.as_str())),
            "{envs:?}"
        );
        assert_eq!(command.get_current_dir(), Some(profile.as_path()));
    }
}
