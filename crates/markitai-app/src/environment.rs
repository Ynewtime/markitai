//! The process environment an app gives the core before it starts its UI
//! (native/DESIGN.md, section 3.3).

use std::path::Path;

pub enum Login {
    #[cfg_attr(
        not(unix),
        allow(dead_code, reason = "only Unix imports a login shell's environment")
    )]
    Imported,
    Inherited,
    #[cfg_attr(
        not(unix),
        allow(dead_code, reason = "only Unix imports a login shell's environment")
    )]
    Failed(String),
}

/// Create `data_dir`, make it the working directory, so the core never reads
/// a `markitai.json` or `.env` from the folder that launched the app, then
/// import the login shell's environment.
///
/// # Safety
/// Sets environment variables: call it before the process starts any other
/// thread that reads or writes the environment.
pub unsafe fn prepare(data_dir: &Path) -> std::io::Result<Login> {
    std::fs::create_dir_all(data_dir)?;
    std::env::set_current_dir(data_dir)?;
    Ok(unsafe { import_login_environment() })
}

#[cfg(not(unix))]
unsafe fn import_login_environment() -> Login {
    Login::Inherited
}

/// Finder and desktop launchers start apps with a bare PATH and none of the
/// user's startup files, which would leave the core without Homebrew,
/// ~/.local/bin, the subscription CLIs, exported keys and MARKITAI_HOME. An
/// app started from a terminal (TERM is set) already has that environment.
/// On failure the usual tool folders join PATH.
#[cfg(unix)]
unsafe fn import_login_environment() -> Login {
    if std::env::var_os("TERM").is_some_and(|term| !term.is_empty()) {
        return Login::Inherited;
    }
    let shell = std::env::var("SHELL")
        .ok()
        .filter(|shell| !shell.is_empty())
        .unwrap_or_else(|| {
            if cfg!(target_os = "macos") {
                "/bin/zsh"
            } else {
                "/bin/sh"
            }
            .into()
        });
    match unix::login_env(&shell) {
        Ok(variables) => {
            for (key, value) in variables {
                if !matches!(key.as_str(), "_" | "PWD" | "OLDPWD" | "SHLVL" | "TERM") {
                    unsafe { std::env::set_var(key, value) };
                }
            }
            Login::Imported
        }
        Err(why) => {
            unsafe { unix::extend_path() };
            Login::Failed(format!("{shell}: {why}"))
        }
    }
}

#[cfg(unix)]
mod unix {
    use std::io::{Read, Write};
    use std::os::unix::process::CommandExt;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::sync::{Arc, Mutex, mpsc};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    /// As long as VS Code waits for the same environment
    /// (application.shellEnvironmentResolutionTimeout).
    const TIMEOUT: Duration = Duration::from_secs(10);

    /// The environment `shell` has as an interactive login shell, so that it
    /// reads the startup files a terminal's does.
    pub fn login_env(shell: &str) -> Result<Vec<(String, String)>, String> {
        login_env_with(shell, None)
    }

    /// [`login_env`], with `home` as the shell's HOME (and ZDOTDIR) when given.
    fn login_env_with(shell: &str, home: Option<&Path>) -> Result<Vec<(String, String)>, String> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let marker = format!("markitai-env-{:x}{:x}", nanos, std::process::id());
        let script = format!("printf %s {marker}; env -0; printf %s {marker}");
        let name = Path::new(shell)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        let csh = matches!(name, "csh" | "tcsh");
        let mut command = Command::new(shell);
        if csh {
            // They take -l only as their one flag, read .cshrc in every
            // shell, and the script from stdin.
            command.arg("-l").stdin(Stdio::piped());
        } else {
            command
                .args(["-l", "-i", "-c", &script])
                .stdin(Stdio::null());
        }
        command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0);
        if let Some(home) = home {
            command.env("HOME", home).env("ZDOTDIR", home);
        }
        let mut child = command.spawn().map_err(|e| e.to_string())?;
        if csh && let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(format!("{script}\n").as_bytes());
        }
        let out = Drained::start(child.stdout.take().expect("piped"));
        let err = Drained::start(child.stderr.take().expect("piped"));
        let deadline = Instant::now() + TIMEOUT;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(20))
                }
                _ => {
                    // The whole group: an interactive shell's own children too.
                    unsafe { libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL) };
                    let _ = child.wait();
                    return Err(format!(
                        "its startup files did not finish within {} s",
                        TIMEOUT.as_secs()
                    ));
                }
            }
        };
        // A startup file may leave a background job holding the pipes open:
        // the group's are killed, one in a session of its own is not waited
        // for beyond a moment, as the environment has been printed.
        unsafe { libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL) };
        let out = out.take(Duration::from_secs(1));
        let err = err.take(Duration::from_millis(100));
        if !status.success() {
            let err = String::from_utf8_lossy(&err);
            return Err(match err.trim().lines().last() {
                Some(line) => format!("{status}: {line}"),
                None => status.to_string(),
            });
        }
        parse(&out, &marker)
    }

    /// A pipe read to its end on a thread of its own, so that a holder that
    /// never closes it does not hold the caller.
    struct Drained {
        bytes: Arc<Mutex<Vec<u8>>>,
        ended: mpsc::Receiver<()>,
    }

    impl Drained {
        fn start(mut pipe: impl Read + Send + 'static) -> Self {
            let bytes = Arc::new(Mutex::new(Vec::new()));
            let (end, ended) = mpsc::channel();
            let shared = Arc::clone(&bytes);
            std::thread::spawn(move || {
                let mut chunk = [0; 8192];
                while let Ok(n @ 1..) = pipe.read(&mut chunk) {
                    shared.lock().unwrap().extend_from_slice(&chunk[..n]);
                }
                let _ = end.send(());
            });
            Self { bytes, ended }
        }

        /// What was read once the pipe ends, or after `grace` if it does not.
        fn take(self, grace: Duration) -> Vec<u8> {
            let _ = self.ended.recv_timeout(grace);
            std::mem::take(&mut *self.bytes.lock().unwrap())
        }
    }

    /// `env -0` output a login shell printed between two markers, ignoring
    /// whatever its startup files printed around them. The end marker is the
    /// first NUL-separated field that starts with it: a variable holding the
    /// marker, as `$_` holds the word before `env -0` in some shells, is a
    /// field that starts with its name.
    pub fn parse(out: &[u8], marker: &str) -> Result<Vec<(String, String)>, String> {
        let marker = marker.as_bytes();
        let start = out
            .windows(marker.len())
            .position(|w| w == marker)
            .ok_or("the login shell printed no environment")?;
        let mut variables = Vec::new();
        for field in out[start + marker.len()..].split(|&b| b == 0) {
            if field.starts_with(marker) {
                if !variables
                    .iter()
                    .any(|(k, _): &(String, String)| k == "PATH")
                {
                    return Err("the login shell's environment has no PATH".into());
                }
                return Ok(variables);
            }
            let field = String::from_utf8_lossy(field);
            if let Some((key, value)) = field.split_once('=')
                && !key.is_empty()
            {
                variables.push((key.to_owned(), value.to_owned()));
            }
        }
        Err("the login shell's environment was cut short".into())
    }

    /// PATH with the usual tool folders added, when the login shell failed.
    ///
    /// # Safety
    /// As [`super::prepare`].
    pub unsafe fn extend_path() {
        let mut path: Vec<PathBuf> = std::env::var_os("PATH")
            .map(|p| std::env::split_paths(&p).collect())
            .unwrap_or_default();
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_default();
        for dir in [
            PathBuf::from("/opt/homebrew/bin"),
            PathBuf::from("/usr/local/bin"),
            home.join(".local/bin"),
        ] {
            if !path.contains(&dir) {
                path.push(dir);
            }
        }
        if let Ok(joined) = std::env::join_paths(path) {
            unsafe { std::env::set_var("PATH", joined) };
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_marker_in_a_variable_does_not_end_the_environment() {
            let parsed = parse(b"motd\nMPATH=/bin\0_=M\0HOME=/h\0Mjunk", "M").unwrap();
            assert_eq!(
                parsed,
                vec![
                    ("PATH".into(), "/bin".into()),
                    ("_".into(), "M".into()),
                    ("HOME".into(), "/h".into())
                ]
            );
            assert!(parse(b"MHOME=/h\0M", "M").unwrap_err().contains("no PATH"));
            assert!(
                parse(b"nothing", "M")
                    .unwrap_err()
                    .contains("no environment")
            );
            assert!(
                parse(b"MPATH=/bin\0", "M")
                    .unwrap_err()
                    .contains("cut short")
            );
        }

        #[test]
        fn a_real_login_shell_gives_its_path() {
            let shell = if Path::new("/bin/zsh").exists() {
                "/bin/zsh"
            } else {
                "/bin/sh"
            };
            let home = tempfile::tempdir().unwrap();
            std::fs::write(
                home.path().join(".zprofile"),
                "export MARKITAI_ENV_PROBE=from-profile\n",
            )
            .unwrap();
            std::fs::write(
                home.path().join(".profile"),
                "export MARKITAI_ENV_PROBE=from-profile\n",
            )
            .unwrap();
            let variables = login_env_with(shell, Some(home.path())).unwrap();
            assert!(
                variables
                    .iter()
                    .any(|(k, v)| k == "MARKITAI_ENV_PROBE" && v == "from-profile")
            );
        }

        #[test]
        fn a_job_left_holding_the_output_does_not_hold_the_app() {
            let shell = if Path::new("/bin/zsh").exists() {
                "/bin/zsh"
            } else {
                "/bin/sh"
            };
            if !Path::new("/usr/bin/perl").exists() {
                return;
            }
            let home = tempfile::tempdir().unwrap();
            let profile = "export MARKITAI_ENV_PROBE=kept\n/usr/bin/perl -MPOSIX -e 'POSIX::setsid(); sleep 6' &\n";
            std::fs::write(home.path().join(".zprofile"), profile).unwrap();
            std::fs::write(home.path().join(".profile"), profile).unwrap();
            let started = Instant::now();
            let variables = login_env_with(shell, Some(home.path())).unwrap();
            assert!(
                started.elapsed() < Duration::from_secs(4),
                "{:?}",
                started.elapsed()
            );
            assert!(
                variables
                    .iter()
                    .any(|(k, v)| k == "MARKITAI_ENV_PROBE" && v == "kept")
            );
            assert!(variables.iter().any(|(k, v)| k == "PATH" && !v.is_empty()));
        }
    }
}
