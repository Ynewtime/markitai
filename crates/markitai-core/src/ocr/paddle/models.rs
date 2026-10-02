//! The model files the portable engine reads. They are not part of the
//! executable: `models.json` names each official download with its size,
//! SHA-256 and license, and a model lives in
//! `MARKITAI_HOME/models/ocr/<name>-<first 8 digits of its SHA-256>/<file>`.
//! `markitai doctor --fix` installs the default set, and the first OCR that
//! needs a missing model downloads it, saying so once. Every load checks the
//! file's size and digest.

use crate::ocr::LocalOcrModelState as State;
use crate::private_install::{self, Contention};
use crate::{Error, Result, config, platform};
use serde::Deserialize;
#[cfg(test)]
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: u8,
    #[allow(dead_code)] // Documentation for readers of the manifest.
    source: String,
    models: Vec<Model>,
}

/// What a model does.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Role {
    Detect,
    Classify,
    Recognize,
}

/// One downloadable model.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Model {
    pub name: String,
    pub role: Role,
    pub file: String,
    pub url: String,
    pub sha256: String,
    pub bytes: u64,
    #[allow(dead_code)] // Provenance for readers; the manifest test checks it.
    pub license: String,
    /// Part of the set `doctor --fix` installs: what the default language
    /// reads with.
    #[serde(default)]
    pub default: bool,
    /// A recognizer of a right-to-left script.
    #[serde(default)]
    pub right_to_left: bool,
}

/// Models larger than this are refused.
const MAX_MODEL: u64 = 256 * 1024 * 1024;

fn parse(text: &str) -> std::result::Result<Vec<Model>, String> {
    let manifest: Manifest = serde_json::from_str(text).map_err(|error| error.to_string())?;
    if manifest.schema != 1 {
        return Err("unknown manifest schema".into());
    }
    let mut names = std::collections::HashSet::new();
    for model in &manifest.models {
        let safe = |s: &str| {
            !s.is_empty()
                && s.len() <= 96
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
                && !s.starts_with('.')
        };
        if !safe(&model.name) || !safe(&model.file) || !names.insert(model.name.as_str()) {
            return Err(format!("invalid or duplicate model name {}", model.name));
        }
        if model.sha256.len() != 64
            || !model
                .sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return Err(format!("{}: invalid SHA-256", model.name));
        }
        if !model.url.starts_with("https://") || model.bytes == 0 || model.bytes > MAX_MODEL {
            return Err(format!("{}: invalid download", model.name));
        }
    }
    Ok(manifest.models)
}

/// Every model of the manifest.
pub(crate) fn manifest() -> &'static [Model] {
    static MODELS: OnceLock<Vec<Model>> = OnceLock::new();
    MODELS.get_or_init(|| {
        // The manifest is part of the source and checked by a test.
        parse(include_str!("models.json")).unwrap_or_default()
    })
}

/// The model of this name.
pub(crate) fn named(name: &str) -> Result<&'static Model> {
    manifest()
        .iter()
        .find(|model| model.name == name)
        .ok_or_else(|| failure(&format!("the model {name} is not in the manifest")))
}

/// The models `doctor --fix` installs.
pub(crate) fn defaults() -> impl Iterator<Item = &'static Model> {
    manifest().iter().filter(|model| model.default)
}

fn failure(message: &str) -> Error {
    Error::Conversion(format!("Local OCR models: {message}"))
}

/// Where models live.
pub(crate) fn root() -> PathBuf {
    config::home().join("models").join("ocr")
}

/// Where a model's file lives under a models `root`.
fn path_in(root: &Path, model: &Model) -> PathBuf {
    root.join(format!("{}-{}", model.name, &model.sha256[..8]))
        .join(&model.file)
}

/// Where a model's file lives.
pub(crate) fn path(model: &Model) -> PathBuf {
    path_in(&root(), model)
}

struct Observation {
    state: State,
    detail: Option<String>,
    file: Option<private_install::HeldFile>,
    directories: Option<private_install::Directories>,
}

impl Observation {
    fn unsafe_path(error: impl std::fmt::Display) -> Self {
        Self {
            state: State::Unsafe,
            detail: Some(error.to_string()),
            file: None,
            directories: None,
        }
    }
    fn validate(&self) -> Result<()> {
        if let Some(dirs) = &self.directories {
            dirs.validate(failure)?;
        }
        if let Some(file) = &self.file {
            file.validate(failure)?;
        }
        Ok(())
    }
}

fn model_home(root: &Path) -> &Path {
    root.parent().and_then(Path::parent).unwrap_or(root)
}

struct OpenModel {
    file: Option<private_install::HeldFile>,
    directories: Option<private_install::Directories>,
}
impl OpenModel {
    fn validate(&self) -> Result<()> {
        if let Some(dirs) = &self.directories {
            dirs.validate(failure)?;
        }
        if let Some(file) = &self.file {
            file.validate(failure)?;
        }
        Ok(())
    }
}

/// Open and validate paths without hashing, creating state, or waiting for a
/// lock. This also guards the full required set before the first OCR download.
fn open_model_in(root: &Path, model: &Model) -> Result<OpenModel> {
    let target = path_in(root, model);
    let directory = target.parent().expect("manifest has a model file");
    match private_install::directories(model_home(root), root, false, failure)? {
        Some(root_dirs) => {
            let lock = root.join("install.lock");
            match platform::status(&lock) {
                Ok(_) => {
                    private_install::HeldFile::open(&lock, failure)?;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(failure(&format!("{}: {error}", lock.display()))),
            }
            root_dirs.validate(failure)?;
        }
        None => {
            return Ok(OpenModel {
                file: None,
                directories: None,
            });
        }
    }
    let dirs = match private_install::directories(model_home(root), directory, false, failure)? {
        Some(dirs) => dirs,
        None => {
            return Ok(OpenModel {
                file: None,
                directories: None,
            });
        }
    };
    let file = match platform::status(&target) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(failure(&format!("{}: {error}", target.display()))),
        Ok(_) => Some(private_install::HeldFile::open(&target, failure)?),
    };
    let opened = OpenModel {
        file,
        directories: Some(dirs),
    };
    opened.validate()?;
    Ok(opened)
}

/// Only path safety is checked here. Missing models are allowed, while safe
/// damaged contents are checked exactly once by load; no ONNX graph is loaded.
pub(crate) fn preflight(models: &[&Model]) -> Result<()> {
    paths_preflight_in(&root(), models)
}
fn paths_preflight_in(root: &Path, models: &[&Model]) -> Result<()> {
    for model in models {
        open_model_in(root, model)?.validate()?;
    }
    Ok(())
}

/// Read-only integrity observation, bounded by each published size.
fn observe_in(root: &Path, model: &Model) -> Observation {
    inspect_in(root, model, |file| file.digest(model.bytes, failure))
}

/// Hash the one exact read: streaming for doctor, retained bytes for loading.
fn inspect_in(
    root: &Path,
    model: &Model,
    inspect: impl FnOnce(&mut private_install::HeldFile) -> Result<String>,
) -> Observation {
    let target = path_in(root, model);
    let opened = match open_model_in(root, model) {
        Ok(opened) => opened,
        Err(error) => return Observation::unsafe_path(error),
    };
    let Some(mut file) = opened.file else {
        return Observation {
            state: State::Missing,
            detail: None,
            file: None,
            directories: opened.directories,
        };
    };
    let (state, detail) = if file.named.metadata().len() != model.bytes {
        (
            State::Corrupt,
            Some(format!(
                "{} has {} bytes instead of its published {} bytes",
                target.display(),
                file.named.metadata().len(),
                model.bytes
            )),
        )
    } else {
        match inspect(&mut file) {
            Ok(digest) => {
                if digest == model.sha256 {
                    (State::Ready, None)
                } else {
                    (
                        State::Corrupt,
                        Some(format!(
                            "{} does not match its published SHA-256",
                            target.display()
                        )),
                    )
                }
            }
            Err(error) => return Observation::unsafe_path(error),
        }
    };
    let observed = Observation {
        state,
        detail,
        file: Some(file),
        directories: opened.directories,
    };
    if let Err(error) = observed.validate() {
        return Observation::unsafe_path(error);
    }
    observed
}

pub(crate) fn observation(model: &Model) -> (State, Option<String>) {
    let observed = observe_in(&root(), model);
    (observed.state, observed.detail)
}

#[cfg(test)]
pub(crate) fn present(model: &Model) -> bool {
    observe_in(&root(), model).state == State::Ready
}

#[cfg(test)]
fn present_in(root: &Path, model: &Model) -> bool {
    observe_in(root, model).state == State::Ready
}

/// How the error for a missing model tells the user to install it.
fn remedy(root: &Path, model: &Model) -> String {
    format!(
        "run `markitai doctor --fix` where the network is reachable, or download {} \
         (SHA-256 {}, {} bytes) to {}",
        model.url,
        model.sha256,
        model.bytes,
        path_in(root, model).display()
    )
}

/// A model's verified bytes. A missing model is downloaded first when
/// `download` is set.
pub(crate) fn load(model: &Model, download: bool) -> Result<Vec<u8>> {
    load_from(&root(), model, download)
}

fn state_error(root: &Path, model: &Model, observed: &Observation) -> Error {
    match observed.state {
        State::Missing => Error::Unsupported(format!(
            "Local OCR needs the model {}, which is not installed: {}",
            model.name,
            remedy(root, model)
        )),
        State::Corrupt => failure(&format!(
            "{}; run `markitai doctor --fix` to replace this damaged managed model; ordinary OCR will not overwrite it",
            observed.detail.as_deref().unwrap_or("model is damaged")
        )),
        State::Unsafe => failure(observed.detail.as_deref().unwrap_or("unsafe model path")),
        State::Ready => failure("model observation changed"),
    }
}

fn load_from(root: &Path, model: &Model, download: bool) -> Result<Vec<u8>> {
    fn inspect_bytes(root: &Path, model: &Model) -> (Observation, Option<Vec<u8>>) {
        let mut bytes = None;
        let observed = inspect_in(root, model, |file| {
            let data = file.read(model.bytes, failure)?;
            let digest = crate::hex(<sha2::Sha256 as sha2::Digest>::digest(&data));
            bytes = Some(data);
            Ok(digest)
        });
        (observed, bytes)
    }
    let (mut observed, mut bytes) = inspect_bytes(root, model);
    if observed.state == State::Missing && download {
        install_into(root, &[model], true, false, &client()?, |model| {
            model.url.clone()
        })?;
        (observed, bytes) = inspect_bytes(root, model);
    }
    if observed.state != State::Ready {
        return Err(state_error(root, model, &observed));
    }
    observed.validate()?;
    Ok(bytes.expect("ready model's exact read was verified"))
}

/// Whether the download notice was shown in this process.
static ANNOUNCED: AtomicBool = AtomicBool::new(false);

/// Explicit repair may replace only a safe damaged managed leaf. A normal
/// OCR calls install_into with repair=false and never overwrites corruption.
pub(crate) fn install(models: &[&Model], announce: bool) -> Result<Vec<PathBuf>> {
    // Reject unsafe paths before building a client, acquiring/creating state or
    // making any public download request.
    install_preflight(&root(), models, true)?;
    install_into(&root(), models, announce, true, &client()?, |model| {
        model.url.clone()
    })
}

fn install_preflight(root: &Path, models: &[&Model], repair: bool) -> Result<Vec<Observation>> {
    let observed: Vec<_> = models.iter().map(|model| observe_in(root, model)).collect();
    for (model, state) in models.iter().zip(&observed) {
        if state.state == State::Unsafe || (state.state == State::Corrupt && !repair) {
            return Err(state_error(root, model, state));
        }
    }
    Ok(observed)
}

fn install_into(
    root: &Path,
    models: &[&Model],
    announce: bool,
    repair: bool,
    client: &reqwest::blocking::Client,
    url: impl Fn(&Model) -> String,
) -> Result<Vec<PathBuf>> {
    install_preflight(root, models, repair)?;
    private_install::directories(model_home(root), root, true, failure)?;
    let lock = private_install::lock(root, Contention::Wait, failure)?;
    // Observe the entire set again under the lock before the first download.
    let observations = install_preflight(root, models, repair)?;
    let needed: Vec<_> = models
        .iter()
        .zip(observations)
        .filter(|(_, state)| state.state != State::Ready)
        .collect();
    if needed.is_empty() {
        return Ok(Vec::new());
    }
    if announce && !ANNOUNCED.swap(true, Ordering::Relaxed) {
        let bytes: u64 = needed.iter().map(|(model, _)| model.bytes).sum();
        eprintln!(
            "Local OCR: downloading {} model file(s) ({:.1} MB, once) from the official PaddleOCR mirror into {}; `markitai doctor --fix` installs them ahead of time.",
            needed.len(),
            bytes as f64 / 1e6,
            root.display()
        );
    }
    let mut installed = Vec::new();
    for (model, state) in needed {
        lock.validate(failure)?;
        state.validate()?;
        fetch(root, client, model, &url(model), state, &lock)?;
        installed.push(path_in(root, model));
    }
    Ok(installed)
}

fn client() -> Result<reqwest::blocking::Client> {
    let redirects = reqwest::redirect::Policy::custom(|attempt| {
        if attempt.previous().len() >= 5 {
            attempt.error("too many redirects")
        } else if attempt.url().scheme() != "https" {
            attempt.error("redirect away from HTTPS")
        } else {
            attempt.follow()
        }
    });
    crate::proxy::http(
        reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(900))
            .redirect(redirects)
            .user_agent(concat!("markitai/", env!("CARGO_PKG_VERSION"))),
    )?
    .build()
    .map_err(|_| failure("cannot initialize the download client"))
}

/// Verify the actual staged leaf, retaining its no-follow handle until
/// publication. Hashing the incoming stream alone cannot verify disk contents.
fn verified_stage(
    staged: &tempfile::NamedTempFile,
    model: &Model,
) -> Result<private_install::HeldFile> {
    let mut held = private_install::HeldFile::open(staged.path(), failure)?;
    let original = platform::file_status(staged.as_file())?;
    if original.id() != held.named.id() || original.metadata().len() != model.bytes {
        return Err(failure("staged model changed or has an incorrect size"));
    }
    if held.digest(model.bytes, failure)? != model.sha256 {
        return Err(failure("staged model does not match its published SHA-256"));
    }
    Ok(held)
}

/// The bare message of a download step.
fn reason(message: &str) -> Error {
    Error::Conversion(message.to_owned())
}

/// Download `model` from `url` into its directory under `root`: a staged
/// file that replaces nothing until its size and digest match.
fn fetch(
    root: &Path,
    client: &reqwest::blocking::Client,
    model: &Model,
    url: &str,
    observed: Observation,
    lock: &private_install::InstallLock,
) -> Result<()> {
    let target = path_in(root, model);
    let directory = target
        .parent()
        .ok_or_else(|| failure("invalid model directory"))?;
    let dirs = private_install::directories(model_home(root), directory, true, failure)?
        .ok_or_else(|| failure("missing managed model directory"))?;
    lock.validate(failure)?;
    observed.validate()?;
    let mut staged = tempfile::Builder::new()
        .prefix(".download-")
        .make_in(directory, |path| {
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            platform::private_file(&mut options);
            platform::open_no_follow(&options, path)
        })
        .map_err(|_| failure("cannot stage a download"))?;
    dirs.validate(failure)?;
    lock.validate(failure)?;
    let digest = private_install::download(client, url, model.bytes, staged.as_file_mut(), reason)
        .map_err(|error| {
            let cause = match error {
                Error::Conversion(message) => message,
                other => other.to_string(),
            };
            Error::Unsupported(format!(
                "Local OCR download of {} failed ({cause}); any existing damaged file was kept: {}",
                model.name,
                remedy(root, model)
            ))
        })?;
    let written = platform::file_status(staged.as_file())?.metadata().len();
    if digest != model.sha256 || written != model.bytes {
        return Err(failure(&format!(
            "the download of {} does not match its published size and SHA-256; nothing was installed",
            model.name
        )));
    }
    staged.as_file_mut().flush()?;
    staged.as_file().sync_all()?;
    let staged_held = verified_stage(&staged, model)?;
    dirs.validate(failure)?;
    lock.validate(failure)?;
    observed.validate()?;
    staged_held.validate(failure)?;
    if platform::file_status(staged.as_file())?.id() != staged_held.named.id() {
        return Err(failure("staged model changed before publication"));
    }
    let installed = match observed.state {
        State::Missing => platform::persist_noclobber(staged, &target).map_err(|error| error.error),
        State::Corrupt => {
            // Keep the observed file pinned during replacement. On Windows,
            // std's handle-based rename can replace this open target, whereas
            // NamedTempFile::persist's MoveFileEx publication cannot.
            let (file, mut path) = staged.into_parts();
            platform::rename(&path, &target).map(|()| {
                path.disable_cleanup(true);
                file
            })
        }
        _ => return Err(failure("invalid model publication state")),
    }
    .map_err(|_| {
        failure(&format!(
            "cannot install {}; any existing file was kept",
            target.display()
        ))
    })?;
    platform::sync_renamed(&installed)?;
    platform::sync_directory(directory)?;
    dirs.validate(failure)?;
    lock.validate(failure)?;
    if observe_in(root, model).state != State::Ready {
        return Err(failure("installed model changed before verification"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn the_manifest_names_official_downloads_with_digests_and_licenses() {
        let models = parse(include_str!("models.json")).unwrap();
        assert_eq!(models.len(), manifest().len());
        for model in &models {
            assert!(
                model.url.starts_with(
                    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/"
                ),
                "{}",
                model.url
            );
            assert!(model.url.ends_with(&format!("/{}", model.file)));
            assert_eq!(model.license, "Apache-2.0");
        }
        // The default set: one detector, the classifier, the multilingual
        // recognizer and the Korean one.
        let names: Vec<&str> = defaults().map(|model| model.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "ppocrv6-det-small",
                "ppocr-cls-mobile-v2",
                "ppocrv6-rec-small",
                "korean-ppocrv5-rec-mobile"
            ]
        );
        let bytes: u64 = defaults().map(|model| model.bytes).sum();
        assert!(bytes < 50_000_000, "{bytes}");
        assert!(named("arabic-ppocrv5-rec-mobile").unwrap().right_to_left);
        assert!(named("../escape").is_err());
    }

    #[test]
    fn malformed_manifests_are_refused() {
        let model = |name: &str, sha: &str, url: &str| {
            format!(
                r#"{{"schema":1,"source":"x","models":[{{"name":"{name}","role":"detect","file":"a.onnx","url":"{url}","sha256":"{sha}","bytes":1,"license":"Apache-2.0"}}]}}"#
            )
        };
        let sha = "0".repeat(64);
        assert!(parse(&model("ok", &sha, "https://example.test/a.onnx")).is_ok());
        assert!(parse(&model("../up", &sha, "https://example.test/a.onnx")).is_err());
        assert!(parse(&model("ok", "ABC", "https://example.test/a.onnx")).is_err());
        assert!(parse(&model("ok", &"A".repeat(64), "https://example.test/a.onnx")).is_err());
        assert!(parse(&model("ok", &sha, "http://example.test/a.onnx")).is_err());
        assert!(parse(r#"{"schema":2,"source":"x","models":[]}"#).is_err());
    }

    /// A model of `bytes` served from `url`.
    fn model(bytes: &[u8], url: &str) -> Model {
        Model {
            name: "test-model".into(),
            role: Role::Recognize,
            file: "model.onnx".into(),
            url: url.into(),
            sha256: crate::hex(<sha2::Sha256 as sha2::Digest>::digest(bytes)),
            bytes: bytes.len() as u64,
            license: "Apache-2.0".into(),
            default: false,
            right_to_left: false,
        }
    }

    /// Serve `bytes` once from a loopback HTTP server.
    fn serve(bytes: &'static [u8]) -> (String, std::thread::JoinHandle<()>) {
        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/model.onnx", listener.local_addr().unwrap());
        let task = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut request = [0u8; 4096];
            let _ = stream.read(&mut request).unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")
                .unwrap();
            stream.write_all(bytes).unwrap();
        });
        (url, task)
    }

    fn client() -> reqwest::blocking::Client {
        reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(3))
            .build()
            .unwrap()
    }

    #[test]
    fn a_download_is_installed_only_with_its_published_size_and_digest() {
        let root = tempfile::tempdir().unwrap();
        let root = root.path().join("models/ocr");
        let (url, task) = serve(b"model bytes");
        let wanted = model(b"model bytes", &url);
        assert!(!present_in(&root, &wanted));
        // A missing model is not downloaded when that is not allowed.
        let error = load_from(&root, &wanted, false).unwrap_err().to_string();
        assert!(
            error.contains("doctor --fix") && error.contains(&url),
            "{error}"
        );
        let installed =
            install_into(&root, &[&wanted], false, true, &client(), |m| m.url.clone()).unwrap();
        task.join().unwrap();
        assert_eq!(installed, [path_in(&root, &wanted)]);
        assert!(installed[0].ends_with(format!("test-model-{}/model.onnx", &wanted.sha256[..8])));
        assert_eq!(load_from(&root, &wanted, false).unwrap(), b"model bytes");
        // Present models are not fetched again (nothing listens now).
        assert!(
            install_into(&root, &[&wanted], false, true, &client(), |m| m.url.clone())
                .unwrap()
                .is_empty()
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&root), 0o700);
            assert_eq!(mode(installed[0].parent().unwrap()), 0o700);
            assert_eq!(mode(&installed[0]), 0o600);
        }
        // A file that no longer matches its digest is refused, not read.
        fs::write(&installed[0], b"model bytez").unwrap();
        let error = load_from(&root, &wanted, false).unwrap_err().to_string();
        assert!(
            error.contains("does not match its published SHA-256"),
            "{error}"
        );
    }

    #[test]
    fn a_wrong_or_unreachable_download_installs_nothing_and_names_the_remedy() {
        let root = tempfile::tempdir().unwrap();
        let root = root.path().join("models/ocr");
        let (url, task) = serve(b"other bytes");
        let wanted = model(b"model bytes", &url);
        let error = install_into(&root, &[&wanted], false, true, &client(), |m| m.url.clone())
            .unwrap_err()
            .to_string();
        task.join().unwrap();
        assert!(
            error.contains("does not match its published size"),
            "{error}"
        );
        assert!(!present_in(&root, &wanted));
        let directory = path_in(&root, &wanted).parent().unwrap().to_path_buf();
        assert_eq!(
            fs::read_dir(&directory).unwrap().count(),
            0,
            "staged file removed"
        );
        // Too long for its published size.
        let (url, task) = serve(b"model bytes and more");
        assert!(install_into(&root, &[&wanted], false, true, &client(), |_| url.clone()).is_err());
        task.join().unwrap();
        // Nothing listening: the error says how to install by hand.
        let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/model.onnx", closed.local_addr().unwrap());
        drop(closed);
        let error =
            install_into(&root, &[&wanted], false, true, &client(), |_| url.clone()).unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
        let error = error.to_string();
        assert!(error.contains("could not be reached"), "{error}");
        assert!(
            error.contains("doctor --fix") && error.contains(&wanted.sha256),
            "{error}"
        );
        assert!(!present_in(&root, &wanted));
    }

    fn managed_file(root: &Path, wanted: &Model, bytes: &[u8]) -> PathBuf {
        let path = path_in(root, wanted);
        private_install::directories(model_home(root), path.parent().unwrap(), true, failure)
            .unwrap();
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        let mut file =
            platform::open_no_follow(platform::private_file(&mut options), &path).unwrap();
        file.write_all(bytes).unwrap();
        path
    }

    #[test]
    fn missing_observations_and_lightweight_preflight_do_not_create_state() {
        let home = tempfile::tempdir().unwrap();
        let root = home.path().join("models/ocr");
        let wanted = model(b"model bytes", "http://127.0.0.1:1/model.onnx");
        assert_eq!(observe_in(&root, &wanted).state, State::Missing);
        paths_preflight_in(&root, &[&wanted]).unwrap();
        assert!(!root.parent().unwrap().exists());
        // Contents deliberately are not an ONNX graph. Path preflight allows
        // a safe damaged file, while the integrity observation detects it.
        let path = managed_file(&root, &wanted, b"model bytez");
        let id = platform::status(&path).unwrap().id();
        paths_preflight_in(&root, &[&wanted]).unwrap();
        assert_eq!(observe_in(&root, &wanted).state, State::Corrupt);
        assert_eq!(platform::status(&path).unwrap().id(), id);
        assert!(!root.join("install.lock").exists());
    }

    #[cfg(unix)]
    #[test]
    fn the_required_set_is_refused_before_missing_paths_are_created_or_downloaded() {
        use std::os::unix::fs::symlink;
        let home = tempfile::tempdir().unwrap();
        let root = home.path().join("models/ocr");
        let missing = model(b"model bytes", "http://127.0.0.1:1/model.onnx");
        let mut unsafe_model = model(b"model bytes", "http://127.0.0.1:1/model.onnx");
        unsafe_model.name = "second-model".into();
        let path = managed_file(&root, &unsafe_model, b"model bytez");
        fs::remove_file(&path).unwrap();
        let outside = home.path().join("outside");
        fs::write(&outside, b"external unchanged").unwrap();
        symlink(&outside, &path).unwrap();
        let all = [&missing, &unsafe_model];
        assert!(paths_preflight_in(&root, &all).is_err());
        assert!(
            install_into(&root, &all, false, true, &client(), |_| panic!(
                "the entire set must be safe before downloading"
            ))
            .is_err()
        );
        assert!(!path_in(&root, &missing).parent().unwrap().exists());
        assert!(!root.join("install.lock").exists());
        assert_eq!(fs::read(&outside).unwrap(), b"external unchanged");
    }

    #[test]
    fn same_length_corruption_is_observed_and_only_explicit_repair_replaces_it() {
        let home = tempfile::tempdir().unwrap();
        let root = home.path().join("models/ocr");
        let wanted = model(b"model bytes", "http://127.0.0.1:1/model.onnx");
        let path = managed_file(&root, &wanted, b"model bytez");
        let before = platform::status(&path).unwrap().id();
        assert_eq!(observe_in(&root, &wanted).state, State::Corrupt);
        assert!(
            load_from(&root, &wanted, true)
                .unwrap_err()
                .to_string()
                .contains("ordinary OCR will not overwrite")
        );
        assert!(
            install_into(&root, &[&wanted], false, false, &client(), |_| panic!(
                "no download for ordinary OCR"
            ))
            .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), b"model bytez");
        let (url, task) = serve(b"model bytes");
        install_into(&root, &[&wanted], false, true, &client(), |_| url.clone()).unwrap();
        task.join().unwrap();
        assert_eq!(observe_in(&root, &wanted).state, State::Ready);
        assert_eq!(fs::read(&path).unwrap(), b"model bytes");
        assert_ne!(platform::status(&path).unwrap().id(), before);
    }

    #[test]
    fn failed_repair_keeps_the_original_and_removes_its_download_stage() {
        let home = tempfile::tempdir().unwrap();
        let root = home.path().join("models/ocr");
        let wanted = model(b"model bytes", "http://127.0.0.1:1/model.onnx");
        let path = managed_file(&root, &wanted, b"model bytez");
        let id = platform::status(&path).unwrap().id();
        let (url, task) = serve(b"other bytes");
        assert!(install_into(&root, &[&wanted], false, true, &client(), |_| url.clone()).is_err());
        task.join().unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"model bytez");
        assert_eq!(platform::status(&path).unwrap().id(), id);
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
    }

    #[test]
    fn same_length_stage_corruption_or_leaf_replacement_cannot_be_published() {
        let home = tempfile::tempdir().unwrap();
        let wanted = model(b"model bytes", "http://127.0.0.1:1/model.onnx");
        let mut stage = tempfile::NamedTempFile::new_in(home.path()).unwrap();
        stage.write_all(b"model bytes").unwrap();
        assert!(verified_stage(&stage, &wanted).is_ok());
        fs::write(stage.path(), b"model bytez").unwrap();
        assert!(
            verified_stage(&stage, &wanted)
                .err()
                .unwrap()
                .to_string()
                .contains("SHA-256")
        );
        fs::rename(stage.path(), home.path().join("held-old-stage")).unwrap();
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        let mut replacement =
            platform::open_no_follow(platform::private_file(&mut options), stage.path()).unwrap();
        replacement.write_all(b"model bytes").unwrap();
        assert!(
            verified_stage(&stage, &wanted)
                .err()
                .unwrap()
                .to_string()
                .contains("staged model changed")
        );
    }

    #[test]
    fn concurrent_installers_share_the_stable_lock_and_download_only_once() {
        let home = tempfile::tempdir().unwrap();
        let root = home.path().join("models/ocr");
        let (url, task) = serve(b"model bytes");
        let wanted = std::sync::Arc::new(model(b"model bytes", &url));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let tasks: Vec<_> = (0..2)
            .map(|_| {
                let root = root.clone();
                let wanted = wanted.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    install_into(&root, &[&wanted], false, true, &client(), |model| {
                        model.url.clone()
                    })
                })
            })
            .collect();
        let count: usize = tasks
            .into_iter()
            .map(|task| task.join().unwrap().unwrap().len())
            .sum();
        task.join().unwrap();
        assert_eq!(count, 1);
        assert_eq!(observe_in(&root, &wanted).state, State::Ready);
    }

    #[test]
    fn replacing_the_install_lock_before_fetch_prevents_the_request() {
        let home = tempfile::tempdir().unwrap();
        let root = home.path().join("models/ocr");
        let wanted = model(b"model bytes", "http://127.0.0.1:1/model.onnx");
        let error = install_into(&root, &[&wanted], false, true, &client(), |_| {
            let path = root.join("install.lock");
            fs::rename(&path, root.join("old-lock")).unwrap();
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            platform::open_no_follow(platform::private_file(&mut options), &path).unwrap();
            "http://127.0.0.1:1/must-not-request".into()
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("lock was replaced"), "{error}");
        assert!(!path_in(&root, &wanted).exists());
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_fifos_hardlinks_and_public_paths_are_unsafe_before_any_download() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let home = tempfile::tempdir().unwrap();
        let root = home.path().join("models/ocr");
        let wanted = model(b"model bytes", "http://127.0.0.1:1/model.onnx");
        let path = managed_file(&root, &wanted, b"model bytez");
        fs::remove_file(&path).unwrap();
        let outside = home.path().join("outside");
        fs::write(&outside, b"model bytes").unwrap();
        symlink(&outside, &path).unwrap();
        assert_eq!(observe_in(&root, &wanted).state, State::Unsafe);
        assert!(
            install_into(&root, &[&wanted], false, true, &client(), |_| panic!(
                "unsafe leaf must not fetch"
            ))
            .is_err()
        );
        assert_eq!(fs::read(&outside).unwrap(), b"model bytes");
        fs::remove_file(&path).unwrap();
        let cpath = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(cpath.as_ptr(), 0o600) }, 0);
        assert_eq!(observe_in(&root, &wanted).state, State::Unsafe);
        assert!(load_from(&root, &wanted, true).is_err());
        fs::remove_file(&path).unwrap();
        let path = managed_file(&root, &wanted, b"model bytez");
        fs::hard_link(&path, home.path().join("alias")).unwrap();
        assert_eq!(observe_in(&root, &wanted).state, State::Unsafe);
        assert!(
            install_into(&root, &[&wanted], false, true, &client(), |_| panic!(
                "hardlink must not fetch"
            ))
            .is_err()
        );
        fs::remove_file(home.path().join("alias")).unwrap();
        fs::set_permissions(path.parent().unwrap(), fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(observe_in(&root, &wanted).state, State::Unsafe);
        assert!(
            install_into(&root, &[&wanted], false, true, &client(), |_| panic!(
                "public managed directory must not fetch"
            ))
            .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn an_init_style_home_container_is_allowed_and_a_managed_parent_link_is_not() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let home = tempfile::tempdir().unwrap();
        fs::set_permissions(home.path(), fs::Permissions::from_mode(0o755)).unwrap();
        let root = home.path().join("models/ocr");
        let wanted = model(b"model bytes", "http://127.0.0.1:1/model.onnx");
        let path = managed_file(&root, &wanted, b"model bytes");
        assert_eq!(observe_in(&root, &wanted).state, State::Ready);
        fs::remove_file(&path).unwrap();
        fs::remove_dir(path.parent().unwrap()).unwrap();
        fs::remove_dir(&root).unwrap();
        fs::remove_dir(root.parent().unwrap()).unwrap();
        let outside = home.path().join("outside");
        platform::private_directory().create(&outside).unwrap();
        symlink(&outside, root.parent().unwrap()).unwrap();
        assert_eq!(observe_in(&root, &wanted).state, State::Unsafe);
        assert!(
            install_into(&root, &[&wanted], false, true, &client(), |_| panic!(
                "unsafe parent must not fetch"
            ))
            .is_err()
        );
        assert_eq!(fs::read_dir(&outside).unwrap().count(), 0);
    }

    #[cfg(windows)]
    #[test]
    fn a_managed_directory_junction_is_refused_before_download_or_external_writes() {
        let home = tempfile::tempdir().unwrap();
        let root = home.path().join("models/ocr");
        let outside = home.path().join("outside");
        std::fs::create_dir(&outside).unwrap();
        let junction = root.parent().unwrap();
        let processor = std::env::var_os("ComSpec").unwrap_or_else(|| "cmd.exe".into());
        let mut command = std::process::Command::new(processor);
        command.env_clear();
        for name in ["SystemRoot", "WINDIR", "ComSpec"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        let result = command
            .args(["/d", "/c", "mklink", "/J"])
            .arg(junction)
            .arg(&outside)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let wanted = model(b"model bytes", "http://127.0.0.1:1/model.onnx");
        assert_eq!(observe_in(&root, &wanted).state, State::Unsafe);
        assert!(
            install_into(&root, &[&wanted], false, true, &client(), |_| panic!(
                "junction must not download"
            ))
            .is_err()
        );
        assert_eq!(std::fs::read_dir(&outside).unwrap().count(), 0);
        std::fs::remove_dir(junction).unwrap();
    }
}
