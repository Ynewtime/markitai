//! How this copy of markitai was installed, which decides what `update` may
//! do: a setup-script install is replaced in place; every other method only
//! gets the command that updates it, so markitai never fights the tool that
//! owns its files.
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The setup script's record of an install (`receipt`), kept line by line so
/// rewriting it keeps every field this version does not know.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Receipt {
    lines: Vec<(String, String)>,
}

impl Receipt {
    pub(crate) fn parse(text: &str) -> Self {
        Self {
            lines: text
                .lines()
                .filter_map(|line| {
                    let line = line.trim_end_matches('\r');
                    let (key, value) = line.split_once('=')?;
                    Some((key.trim().to_owned(), value.to_owned()))
                })
                .collect(),
        }
    }

    pub(crate) fn get(&self, key: &str) -> Option<&str> {
        self.lines
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.as_str())
    }

    /// Every value of a repeated key (`file=` in the Windows receipt).
    pub(crate) fn all(&self, key: &str) -> Vec<&str> {
        self.lines
            .iter()
            .filter(|(name, _)| name == key)
            .map(|(_, value)| value.as_str())
            .collect()
    }

    /// Replace the first `key` (or append it); later duplicates go.
    pub(crate) fn set(&mut self, key: &str, value: &str) {
        let mut seen = false;
        self.lines.retain_mut(|(name, old)| {
            if name != key {
                return true;
            }
            if seen {
                return false;
            }
            seen = true;
            *old = value.to_owned();
            true
        });
        if !seen {
            self.lines.push((key.to_owned(), value.to_owned()));
        }
    }

    /// Replace every value of a repeated key, where the first one was.
    pub(crate) fn set_all(&mut self, key: &str, values: &[String]) {
        let at = self.lines.iter().position(|(name, _)| name == key);
        self.lines.retain(|(name, _)| name != key);
        let at = at.unwrap_or(self.lines.len()).min(self.lines.len());
        for (offset, value) in values.iter().enumerate() {
            self.lines
                .insert(at + offset, (key.to_owned(), value.clone()));
        }
    }

    pub(crate) fn render(&self) -> String {
        self.lines
            .iter()
            .map(|(key, value)| format!("{key}={value}\n"))
            .collect()
    }

    /// A receipt the setup script wrote: it always starts with the version.
    fn valid(&self) -> bool {
        self.get("version").is_some_and(|value| !value.is_empty())
    }
}

/// Which setup script made an install.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Script {
    /// `setup.sh`: the program in `bin_dir`, notices and receipt in
    /// `~/.local/share/markitai/current`.
    Shell,
    /// `setup.ps1`: one folder holding the EXEs, the notices and the receipt.
    PowerShell,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SetupInstall {
    pub(crate) script: Script,
    /// The running program as installed (`bin_dir/markitai`, `markitai.exe`).
    pub(crate) exe: PathBuf,
    /// `setup.sh`: the command folder; `setup.ps1`: the install folder.
    pub(crate) dir: PathBuf,
    /// The folder holding `receipt`: `…/markitai/current`, or `dir`.
    pub(crate) receipt_dir: PathBuf,
    pub(crate) receipt: Receipt,
}

impl SetupInstall {
    /// The release target this install takes archives for: the receipt's,
    /// which setup chose (the native one under Rosetta), else this build's.
    pub(crate) fn target(&self) -> String {
        self.receipt
            .get("target")
            .filter(|target| !target.is_empty() && !target.contains(['/', '\\']))
            .unwrap_or(markitai_core::build_info::TARGET)
            .to_owned()
    }
}

/// What the Python wheel's entry point knows about its environment
/// (`bindings/python/markitai/_cli.py` passes it to `run_python`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PythonInstall {
    /// `sys.prefix`.
    pub(crate) prefix: PathBuf,
    /// `sys.executable`.
    pub(crate) executable: PathBuf,
    /// The distribution's `INSTALLER` (`pip`, `uv`, …), when it has one.
    pub(crate) installer: Option<String>,
    /// Whether the distribution sits in the user site (`pip install --user`).
    pub(crate) user: bool,
}

impl PythonInstall {
    pub(crate) fn from_map(facts: &HashMap<String, String>) -> Option<Self> {
        let prefix = facts.get("prefix").filter(|value| !value.is_empty())?;
        Some(Self {
            prefix: PathBuf::from(prefix),
            executable: PathBuf::from(facts.get("executable").cloned().unwrap_or_default()),
            installer: facts
                .get("installer")
                .map(|value| value.trim().to_ascii_lowercase())
                .filter(|value| !value.is_empty()),
            user: facts.get("user").is_some_and(|value| value == "1"),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Method {
    Setup(SetupInstall),
    UvTool,
    Pipx,
    Pip {
        python: PathBuf,
        user: bool,
    },
    UvPip {
        python: PathBuf,
    },
    Cargo,
    /// A package manager: its name, and its update command when known.
    Packaged {
        name: String,
        command: Option<String>,
    },
    /// A pre-release version or a build with local changes: never checked.
    Source,
    Unknown {
        path: Option<PathBuf>,
    },
}

/// The `cargo install` command; the package name is required because
/// vendored crates also carry binaries.
pub(crate) const CARGO_COMMAND: &str =
    "cargo install --locked --git https://github.com/Ynewtime/markitai markitai-cli";

impl Method {
    /// The `method` value of `--json` and the API.
    pub(crate) fn key(&self) -> &'static str {
        match self {
            Self::Setup(_) => "setup",
            Self::UvTool => "uv",
            Self::Pipx => "pipx",
            Self::Pip { .. } => "pip",
            Self::UvPip { .. } => "uv-pip",
            Self::Cargo => "cargo",
            Self::Packaged { .. } => "packaged",
            Self::Source => "source",
            Self::Unknown { .. } => "unknown",
        }
    }

    /// How a person names the method (`安装方式：uv tool`).
    pub(crate) fn label(&self) -> String {
        match self {
            Self::Setup(install) => match install.script {
                Script::Shell => "setup.sh".into(),
                Script::PowerShell => "setup.ps1".into(),
            },
            Self::UvTool => "uv tool".into(),
            Self::Pipx => "pipx".into(),
            Self::Pip { .. } => "pip".into(),
            Self::UvPip { .. } => "uv pip".into(),
            Self::Cargo => "cargo install".into(),
            Self::Packaged { name, .. } => name.clone(),
            Self::Source => "source".into(),
            Self::Unknown { .. } => "unknown".into(),
        }
    }

    /// The command that updates this install, when there is one.
    pub(crate) fn command(&self) -> Option<String> {
        Some(match self {
            Self::Setup(_) => "markitai update".into(),
            Self::UvTool => "uv tool upgrade markitai".into(),
            Self::Pipx => "pipx upgrade markitai".into(),
            Self::Pip { python, user } => format!(
                "{} -m pip install -U{} markitai",
                shell_word(python),
                if *user { " --user" } else { "" }
            ),
            Self::UvPip { python } => {
                format!("uv pip install -U markitai --python {}", shell_word(python))
            }
            Self::Cargo => CARGO_COMMAND.into(),
            Self::Packaged { command, .. } => command.clone()?,
            Self::Source | Self::Unknown { .. } => return None,
        })
    }

    pub(crate) fn in_place(&self) -> bool {
        matches!(self, Self::Setup(_))
    }
}

/// A path as one word of a shell command: quoted when it holds a space.
fn shell_word(path: &Path) -> String {
    let text = path.display().to_string();
    if text.contains([' ', '\'', '"', '$', '`', '&', '(', ')']) {
        format!("\"{text}\"")
    } else {
        text
    }
}

/// Everything detection reads, gathered once so tests can describe any
/// machine.
pub(crate) struct Facts {
    pub(crate) version: String,
    pub(crate) dirty: bool,
    /// `std::env::current_exe()`.
    pub(crate) exe: Option<PathBuf>,
    pub(crate) python: Option<PythonInstall>,
    /// `HOME` (or `USERPROFILE`).
    pub(crate) home: Option<PathBuf>,
    pub(crate) env: HashMap<String, String>,
    /// `MARKITAI_PACKAGED_BY` at build time: `Name` or `Name|command`.
    pub(crate) packaged_by: Option<String>,
    pub(crate) windows: bool,
}

impl Facts {
    pub(crate) fn current() -> Self {
        let env = markitai_core::config::environment();
        let home = ["HOME", "USERPROFILE"]
            .iter()
            .find_map(|name| env.get(*name).filter(|value| !value.is_empty()))
            .map(PathBuf::from);
        let (version, dirty) = super::current_identity();
        Self {
            version,
            dirty,
            exe: std::env::current_exe().ok(),
            python: python_install(),
            home,
            env,
            packaged_by: option_env!("MARKITAI_PACKAGED_BY").map(str::to_owned),
            windows: cfg!(windows),
        }
    }

    fn var(&self, name: &str) -> Option<&str> {
        self.env
            .get(name)
            .map(String::as_str)
            .filter(|value| !value.is_empty())
    }
}

static PYTHON: std::sync::OnceLock<PythonInstall> = std::sync::OnceLock::new();

/// Called once by the wheel's entry point before the program runs.
pub(crate) fn set_python_install(install: PythonInstall) {
    let _ = PYTHON.set(install);
}

fn python_install() -> Option<PythonInstall> {
    PYTHON.get().cloned()
}

pub(crate) fn detect(facts: &Facts) -> Method {
    let source = super::version::Version::parse(&facts.version)
        .is_none_or(|version| version.is_prerelease());
    if source || facts.dirty {
        return Method::Source;
    }
    if let Some(python) = &facts.python {
        return python_method(python);
    }
    let Some(exe) = &facts.exe else {
        return Method::Unknown { path: None };
    };
    if let Some(install) = setup_install(facts, exe) {
        return Method::Setup(install);
    }
    if let Some(packaged) = facts.packaged_by.as_deref().filter(|text| !text.is_empty()) {
        let (name, command) = match packaged.split_once('|') {
            Some((name, command)) => (name.trim(), Some(command.trim().to_owned())),
            None => (packaged.trim(), None),
        };
        return Method::Packaged {
            name: name.to_owned(),
            command: command.filter(|command| !command.is_empty()),
        };
    }
    if cargo_install(facts, exe) {
        return Method::Cargo;
    }
    if let Some(method) = packaged_path(exe, facts.windows) {
        return method;
    }
    Method::Unknown {
        path: Some(exe.clone()),
    }
}

fn python_method(python: &PythonInstall) -> Method {
    if python.prefix.join("uv-receipt.toml").is_file() {
        return Method::UvTool;
    }
    // pipx may install with uv (and then INSTALLER says uv): its own
    // metadata decides first.
    if python.prefix.join("pipx_metadata.json").is_file() {
        return Method::Pipx;
    }
    match python.installer.as_deref() {
        Some("uv") => Method::UvPip {
            python: python.executable.clone(),
        },
        _ => Method::Pip {
            python: python.executable.clone(),
            user: python.user,
        },
    }
}

/// The same file, by identity where it can be read; by name otherwise.
fn same_file(left: &Path, right: &Path) -> bool {
    match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

fn read_receipt(dir: &Path) -> Option<Receipt> {
    let path = dir.join("receipt");
    let metadata = std::fs::symlink_metadata(&path).ok()?;
    if !metadata.is_file() || metadata.len() > 64 * 1024 {
        return None;
    }
    let receipt = Receipt::parse(&std::fs::read_to_string(path).ok()?);
    receipt.valid().then_some(receipt)
}

fn setup_install(facts: &Facts, exe: &Path) -> Option<SetupInstall> {
    if facts.windows {
        // setup.ps1: the receipt beside markitai.exe lists it.
        let dir = exe.parent()?;
        let receipt = read_receipt(dir)?;
        let names = receipt.all("file");
        if !names
            .iter()
            .any(|name| name.eq_ignore_ascii_case("markitai.exe"))
        {
            return None;
        }
        let installed = dir.join("markitai.exe");
        if !same_file(&installed, exe) {
            return None;
        }
        return Some(SetupInstall {
            script: Script::PowerShell,
            exe: installed,
            dir: dir.to_owned(),
            receipt_dir: dir.to_owned(),
            receipt,
        });
    }
    // setup.sh: ~/.local/share/markitai/current/receipt, whose bin_dir holds
    // this very program.
    let data = facts
        .var("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| facts.home.as_ref().map(|home| home.join(".local/share")))?;
    let receipt_dir = data.join("markitai").join("current");
    let receipt = read_receipt(&receipt_dir)?;
    let dir = PathBuf::from(receipt.get("bin_dir").filter(|dir| !dir.is_empty())?);
    let installed = dir.join("markitai");
    if !same_file(&installed, exe) {
        return None;
    }
    Some(SetupInstall {
        script: Script::Shell,
        exe: installed,
        dir,
        receipt_dir,
        receipt,
    })
}

fn cargo_install(facts: &Facts, exe: &Path) -> bool {
    let root = facts
        .var("CARGO_INSTALL_ROOT")
        .or_else(|| facts.var("CARGO_HOME"))
        .map(PathBuf::from)
        .or_else(|| facts.home.as_ref().map(|home| home.join(".cargo")));
    let Some(root) = root else { return false };
    if !exe
        .parent()
        .is_some_and(|dir| same_file(dir, &root.join("bin")))
    {
        return false;
    }
    std::fs::read_to_string(root.join(".crates2.json")).is_ok_and(|text| {
        serde_json::from_str::<serde_json::Value>(&text).is_ok_and(|value| {
            value["installs"]
                .as_object()
                .is_some_and(|installs| installs.keys().any(|key| key.starts_with("markitai-cli ")))
        })
    })
}

/// A package manager recognised by where it keeps programs.
fn packaged_path(exe: &Path, windows: bool) -> Option<Method> {
    let text = exe.display().to_string();
    let packaged = |name: &str, command: Option<&str>| {
        Some(Method::Packaged {
            name: name.to_owned(),
            command: command.map(str::to_owned),
        })
    };
    if windows {
        let lower = text.to_ascii_lowercase().replace('/', "\\");
        if lower.contains("\\scoop\\apps\\") {
            return packaged("Scoop", Some("scoop update markitai"));
        }
        if lower.contains("\\microsoft\\winget\\packages\\") || lower.contains("\\windowsapps\\") {
            return packaged("winget", Some("winget upgrade markitai"));
        }
        return None;
    }
    if text.starts_with("/nix/store/") {
        return packaged("Nix", None);
    }
    if text.contains("/Cellar/")
        || text.starts_with("/opt/homebrew/")
        || text.contains("/.linuxbrew/")
    {
        return packaged("Homebrew", Some("brew upgrade markitai"));
    }
    if ["/usr/bin/", "/usr/lib/", "/usr/libexec/", "/bin/"]
        .iter()
        .any(|prefix| text.starts_with(prefix))
    {
        return packaged("the system package manager", None);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(root: &Path, exe: Option<PathBuf>) -> Facts {
        Facts {
            version: "2.1.0".into(),
            dirty: false,
            exe,
            python: None,
            home: Some(root.join("home")),
            env: HashMap::new(),
            packaged_by: None,
            windows: false,
        }
    }

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn receipts_keep_unknown_fields_and_their_order() {
        let mut receipt =
            Receipt::parse("version=2.1.0\r\ntarget=x\nfile=a\nfile=b\nfuture=1\nbroken line\n");
        assert_eq!(receipt.get("version"), Some("2.1.0"));
        assert_eq!(receipt.all("file"), ["a", "b"]);
        receipt.set("version", "2.2.0");
        receipt.set("base", "https://example.test/releases");
        receipt.set_all("file", &["c".into(), "d".into(), "e".into()]);
        assert_eq!(
            receipt.render(),
            "version=2.2.0\ntarget=x\nfile=c\nfile=d\nfile=e\nfuture=1\nbase=https://example.test/releases\n"
        );
        assert!(!Receipt::parse("target=x\n").valid());
    }

    #[test]
    fn a_setup_sh_install_is_its_receipt_naming_this_program() {
        let root = tempfile::tempdir().unwrap();
        let bin = root.path().join("home/.local/bin");
        write(&bin.join("markitai"), "binary");
        let receipt = root
            .path()
            .join("home/.local/share/markitai/current/receipt");
        write(
            &receipt,
            &format!(
                "version=2.1.0\ntarget=x86_64-unknown-linux-gnu\nbin_dir={}\n",
                bin.display()
            ),
        );
        let found = detect(&facts(root.path(), Some(bin.join("markitai"))));
        let Method::Setup(install) = &found else {
            panic!("{found:?}")
        };
        assert_eq!(install.script, Script::Shell);
        assert_eq!(install.dir, bin);
        assert_eq!(install.target(), "x86_64-unknown-linux-gnu");
        assert_eq!(found.command().as_deref(), Some("markitai update"));
        assert!(found.in_place());
        // Another copy of markitai is not that install.
        write(&root.path().join("opt/markitai"), "binary");
        assert_eq!(
            detect(&facts(root.path(), Some(root.path().join("opt/markitai")))),
            Method::Unknown {
                path: Some(root.path().join("opt/markitai"))
            }
        );
        // XDG_DATA_HOME moves the receipt.
        let mut moved = facts(root.path(), Some(bin.join("markitai")));
        moved.env.insert(
            "XDG_DATA_HOME".into(),
            root.path().join("xdg").display().to_string(),
        );
        assert!(matches!(detect(&moved), Method::Unknown { .. }));
    }

    #[test]
    fn a_setup_ps1_install_is_a_receipt_beside_the_exe_that_lists_it() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("Programs/markitai");
        write(&dir.join("markitai.exe"), "exe");
        write(
            &dir.join("receipt"),
            "version=2.1.0\ntarget=x86_64-pc-windows-msvc\npath_added=1\nfile=markitai.exe\nfile=mkai.exe\n",
        );
        let mut windows = facts(root.path(), Some(dir.join("markitai.exe")));
        windows.windows = true;
        let Method::Setup(install) = detect(&windows) else {
            panic!()
        };
        assert_eq!(
            (install.script, install.dir.clone()),
            (Script::PowerShell, dir.clone())
        );
        // A receipt that does not list markitai.exe is someone else's.
        write(&dir.join("receipt"), "version=2.1.0\nfile=other.exe\n");
        assert!(matches!(detect(&windows), Method::Unknown { .. }));
    }

    #[test]
    fn python_installs_are_told_apart_by_what_the_environment_holds() {
        let root = tempfile::tempdir().unwrap();
        let prefix = root.path().join("venv");
        std::fs::create_dir_all(&prefix).unwrap();
        let python = PythonInstall {
            prefix: prefix.clone(),
            executable: prefix.join("bin/python"),
            installer: Some("pip".into()),
            user: false,
        };
        let mut with = facts(root.path(), None);
        with.python = Some(python.clone());
        let pip = detect(&with);
        assert_eq!(pip.key(), "pip");
        assert_eq!(
            pip.command().unwrap(),
            format!(
                "{} -m pip install -U markitai",
                prefix.join("bin/python").display()
            )
        );
        with.python = Some(PythonInstall {
            user: true,
            ..python.clone()
        });
        assert!(
            detect(&with)
                .command()
                .unwrap()
                .ends_with("-U --user markitai")
        );
        with.python = Some(PythonInstall {
            installer: Some("uv".into()),
            ..python.clone()
        });
        assert_eq!(detect(&with).key(), "uv-pip");
        // pipx writes INSTALLER=uv when it installs with uv; its metadata wins.
        write(&prefix.join("pipx_metadata.json"), "{}");
        assert_eq!(detect(&with), Method::Pipx);
        write(&prefix.join("uv-receipt.toml"), "");
        assert_eq!(detect(&with), Method::UvTool);
        assert_eq!(
            Method::UvTool.command().as_deref(),
            Some("uv tool upgrade markitai")
        );
        let map: HashMap<String, String> = [
            ("prefix", "/p"),
            ("executable", "/p/bin/python"),
            ("installer", "UV\n"),
            ("user", "1"),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), value.into()))
        .collect();
        assert_eq!(
            PythonInstall::from_map(&map),
            Some(PythonInstall {
                prefix: "/p".into(),
                executable: "/p/bin/python".into(),
                installer: Some("uv".into()),
                user: true
            })
        );
        assert_eq!(PythonInstall::from_map(&HashMap::new()), None);
    }

    #[test]
    fn cargo_packagers_and_source_builds_are_recognised() {
        let root = tempfile::tempdir().unwrap();
        let cargo_bin = root.path().join("home/.cargo/bin");
        write(&cargo_bin.join("markitai"), "binary");
        write(
            &root.path().join("home/.cargo/.crates2.json"),
            r#"{"installs":{"markitai-cli 2.1.0 (git+https://github.com/Ynewtime/markitai#abc)":{}}}"#,
        );
        let cargo = detect(&facts(root.path(), Some(cargo_bin.join("markitai"))));
        assert_eq!(cargo, Method::Cargo);
        assert_eq!(cargo.command().as_deref(), Some(CARGO_COMMAND));

        let mut packaged = facts(root.path(), Some(root.path().join("x/markitai")));
        packaged.packaged_by = Some("Homebrew|brew upgrade markitai".into());
        assert_eq!(
            detect(&packaged).command().as_deref(),
            Some("brew upgrade markitai")
        );
        packaged.packaged_by = Some("Nix".into());
        assert_eq!(
            detect(&packaged),
            Method::Packaged {
                name: "Nix".into(),
                command: None
            }
        );
        let nix = facts(
            root.path(),
            Some("/nix/store/abc-markitai/bin/markitai".into()),
        );
        assert_eq!(detect(&nix).label(), "Nix");
        let brew = facts(root.path(), Some("/opt/homebrew/bin/markitai".into()));
        assert_eq!(
            detect(&brew).command().as_deref(),
            Some("brew upgrade markitai")
        );
        let mut scoop = facts(
            root.path(),
            Some(r"C:\Users\demo\scoop\apps\markitai\current\markitai.exe".into()),
        );
        scoop.windows = true;
        assert_eq!(detect(&scoop).label(), "Scoop");

        for (version, dirty) in [("2.2.0-dev", false), ("2.1.0", true), ("garbage", false)] {
            let mut source = facts(root.path(), Some(cargo_bin.join("markitai")));
            source.version = version.into();
            source.dirty = dirty;
            assert_eq!(detect(&source), Method::Source, "{version} {dirty}");
        }
    }
}
