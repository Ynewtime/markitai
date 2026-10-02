//! Optional Chromium rendering through native CDP with caller-owned session reuse.
mod auth;
#[cfg(test)]
mod auth_tests;
mod cdp;
#[cfg(test)]
mod digest_tests;
mod download;
#[cfg(test)]
mod download_tests;
mod options;
pub(crate) mod pool;
#[cfg(test)]
mod runtime_tests;

use crate::{Asset, Error, Result};
use base64::Engine;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use url::Url;

const MAX_HTML: usize = 100 * 1024 * 1024;
const MAX_SCREENSHOT_BYTES: usize = 100 * 1024 * 1024;
const MAX_SCREENSHOT_PIXELS: u64 = 50_000_000;

pub(crate) enum BrowserResponse {
    Page(BrowserPage),
    Pdf(BrowserPdf),
}

pub(crate) struct BrowserPdf {
    pub bytes: Vec<u8>,
    pub final_url: String,
    pub warnings: Vec<String>,
}

#[cfg(test)]
impl BrowserResponse {
    fn page(self) -> BrowserPage {
        match self {
            Self::Page(page) => page,
            Self::Pdf(_) => panic!("expected rendered HTML"),
        }
    }
}

pub(crate) struct BrowserPage {
    pub html: String,
    pub final_url: String,
    pub title: String,
    pub screenshots: Vec<Asset>,
    pub warnings: Vec<String>,
}

/// Executable by someone on Unix; a program or command script on Windows.
fn executable(path: &Path) -> bool {
    crate::process_groups::launchable(path)
}

/// Why no browser can be used: the configured executable that is not one,
/// or none installed.
fn missing_browser(configured: Option<&Path>) -> String {
    match configured {
        Some(path) => format!(
            "MARKITAI_BROWSER_EXECUTABLE is {}, which is not an executable file; point it at Chrome or Chromium, or unset it to use an installed browser",
            path.display()
        ),
        None => "Chromium is not installed; install Chrome/Chromium or set MARKITAI_BROWSER_EXECUTABLE to its executable".into(),
    }
}

pub(crate) fn discover() -> Option<PathBuf> {
    let pathext = std::env::var_os("PATHEXT");
    if let Some(path) = std::env::var_os("MARKITAI_BROWSER_EXECUTABLE") {
        let path = crate::process_groups::configured_program(Path::new(&path), pathext.as_deref());
        return executable(&path).then_some(path);
    }
    if let Some(path) = crate::browser_install::installed() {
        return Some(path);
    }
    // PATH as a shell searches it (PATHEXT on Windows), skipping empty and
    // relative entries.
    if let Some(path) = crate::process_groups::find_program(
        &[
            "chromium",
            "chromium-browser",
            "google-chrome",
            "google-chrome-stable",
            "chrome",
        ],
        std::env::var_os("PATH").as_deref(),
        pathext.as_deref(),
    ) {
        return Some(path);
    }
    // Only macOS and Windows have well-known installation paths.
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "windows")),
        allow(unused_mut)
    )]
    let mut paths: Vec<PathBuf> = Vec::new();
    #[cfg(target_os = "macos")]
    for path in [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ] {
        paths.push(PathBuf::from(path));
    }
    #[cfg(target_os = "windows")]
    for base in ["PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"] {
        if let Some(base) = std::env::var_os(base) {
            paths.push(PathBuf::from(&base).join("Google/Chrome/Application/chrome.exe"));
            paths.push(PathBuf::from(base).join("Microsoft/Edge/Application/msedge.exe"));
        }
    }
    if let Some(path) = paths.into_iter().find(|path| executable(path)) {
        return Some(path);
    }
    let mut caches = Vec::new();
    if let Some(path) = std::env::var_os("PLAYWRIGHT_BROWSERS_PATH").filter(|path| path != "0") {
        caches.push(PathBuf::from(path));
    } else if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))
    {
        #[cfg(target_os = "macos")]
        caches.push(PathBuf::from(home).join("Library/Caches/ms-playwright"));
        #[cfg(target_os = "windows")]
        caches.push(PathBuf::from(home).join("AppData/Local/ms-playwright"));
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        caches.push(PathBuf::from(home).join(".cache/ms-playwright"));
    }
    for cache in caches {
        let Ok(entries) = std::fs::read_dir(cache) else {
            continue;
        };
        let mut directories = entries
            .filter_map(std::result::Result::ok)
            .filter(|entry| {
                entry.file_name().to_str().is_some_and(|name| {
                    name.starts_with("chromium-") || name.starts_with("chromium_headless_shell-")
                })
            })
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        crate::sort::by(&mut directories, |a, b| b.cmp(a));
        for directory in directories {
            for suffix in [
                "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
                "chrome-mac/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
                "chrome-mac-arm64/Chromium.app/Contents/MacOS/Chromium",
                "chrome-mac/Chromium.app/Contents/MacOS/Chromium",
                "chrome-headless-shell-mac-arm64/chrome-headless-shell",
                "chrome-headless-shell-mac-x64/chrome-headless-shell",
                "chrome-linux64/chrome",
                "chrome-linux/chrome",
                "chrome-win64/chrome.exe",
                "chrome-win/chrome.exe",
                "chrome-headless-shell-linux64/chrome-headless-shell",
                "chrome-headless-shell-win64/chrome-headless-shell.exe",
            ] {
                let path = directory.join(suffix);
                if executable(&path) {
                    return Some(path);
                }
            }
        }
    }
    None
}

/// Check installed executables without launching a browser or opening a profile.
pub fn available() -> bool {
    discover().is_some()
}

/// Launch a private blank page and verify its protocol connection without user settings.
/// Startup is bounded to 15 seconds, followed by a five-second protocol deadline.
pub fn diagnostic() -> Result<Option<PathBuf>> {
    let Some(executable) = discover() else {
        return Ok(None);
    };
    diagnostic_with(&executable)?;
    Ok(Some(executable))
}

pub(crate) fn diagnostic_with(executable: &Path) -> Result<()> {
    let run = || -> Result<()> {
        let mut browser = cdp::Browser::launch(executable, &options::Options::diagnostic())?;
        if browser.evaluate("location.href")?.as_str() != Some("about:blank") {
            return Err(Error::Fetch(
                "Browser diagnostic did not open a blank page".into(),
            ));
        }
        Ok(())
    };
    run().map_err(|_| {
        Error::Fetch("Chromium diagnostic could not initialize a private blank page".into())
    })
}

/// Presence determines routing; full validation runs before browser discovery/launch.
pub(crate) fn http_credentials_configured(cfg: &Value) -> bool {
    cfg.pointer("/fetch/playwright/http_credentials")
        .is_some_and(|value| !value.is_null())
}

/// A browser identity cannot share an anonymous HTTP probe or cached document.
pub(crate) fn identity_configured(cfg: &Value) -> bool {
    http_credentials_configured(cfg)
        || cfg
            .pointer("/fetch/playwright/session_mode")
            .and_then(Value::as_str)
            == Some("domain_persistent")
        || cfg
            .pointer("/fetch/playwright/cookies")
            .and_then(Value::as_array)
            .is_some_and(|cookies| !cookies.is_empty())
        || cfg
            .pointer("/fetch/playwright/extra_http_headers")
            .and_then(Value::as_object)
            .is_some_and(|headers| !headers.is_empty())
}

fn filename(url: &Url) -> String {
    let mut parts = vec![match url.port() {
        Some(port) => format!("{}:{port}", url.host_str().unwrap_or("page")),
        None => url.host_str().unwrap_or("page").to_owned(),
    }];
    parts.extend(
        url.path()
            .trim_matches('/')
            .split('/')
            .filter(|part| !part.is_empty())
            .map(str::to_owned),
    );
    let route = url
        .fragment()
        .filter(|fragment| fragment.starts_with(['/', '!']));
    let variant = format!(
        "{}{}",
        url.query().unwrap_or(""),
        route.map(|route| format!("#{route}")).unwrap_or_default()
    );
    let suffix = if variant.is_empty() {
        String::new()
    } else {
        format!("_q{}", &crate::hex(Sha256::digest(variant.as_bytes()))[..8])
    };
    let mut name = String::new();
    for ch in parts.join("_").chars() {
        let ch = if ch.is_control() || "<>:\"/\\|?*".contains(ch) {
            '_'
        } else {
            ch
        };
        if ch != '_' || !name.ends_with('_') {
            name.push(ch);
        }
    }
    let name = name
        .trim_matches('_')
        .chars()
        .take(200usize.saturating_sub(suffix.len()))
        .collect::<String>();
    format!("{name}{suffix}.full.jpg")
}

fn capture(
    browser: &mut cdp::Browser,
    options: &options::Options,
    url: &Url,
) -> Result<Vec<Asset>> {
    let layout = browser.call("Page.getLayoutMetrics", json!({}))?;
    let size = layout
        .get("cssContentSize")
        .or_else(|| layout.get("contentSize"))
        .ok_or_else(|| Error::Fetch("Chromium returned no page dimensions".into()))?;
    let dimension = |key: &str| {
        size.get(key)
            .and_then(Value::as_f64)
            .filter(|number| number.is_finite() && *number > 0.0)
            .map(|number| number.ceil() as u64)
    };
    let width = dimension("width")
        .ok_or_else(|| Error::Fetch("Invalid browser screenshot width".into()))?;
    let height = dimension("height")
        .ok_or_else(|| Error::Fetch("Invalid browser screenshot height".into()))?;
    if width > 8192
        || height > 100_000
        || width
            .checked_mul(height)
            .is_none_or(|pixels| pixels > MAX_SCREENSHOT_PIXELS)
    {
        return Err(Error::Fetch(
            "Browser screenshot exceeds the 50-million-pixel capture limit".into(),
        ));
    }
    let primary = filename(url);
    let tile_height = if options.tile_height == 0 {
        height
    } else {
        options.tile_height
    };
    let count = height.div_ceil(tile_height);
    if count > 128 {
        return Err(Error::Fetch("Browser screenshot exceeds 128 tiles".into()));
    }
    let mut result = Vec::new();
    let mut total = 0usize;
    for index in 0..count {
        let top = index * tile_height;
        let tile = (height - top).min(tile_height);
        let value = browser.call("Page.captureScreenshot", json!({"format":"jpeg","quality":options.quality,"captureBeyondViewport":true,"fromSurface":true,"clip":{"x":0,"y":top,"width":width,"height":tile,"scale":1}}))?;
        let encoded = value
            .get("data")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Fetch("Chromium returned no screenshot payload".into()))?;
        if encoded.len() > MAX_SCREENSHOT_BYTES * 4 / 3 + 4 {
            return Err(Error::Fetch("Browser screenshot exceeds 100 MiB".into()));
        }
        let mut bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_| Error::Fetch("Invalid Chromium screenshot payload".into()))?;
        if options.tile_height == 0 && height > options.max_height {
            let image = image::ImageReader::with_format(
                crate::images::ImageBytes::new(&bytes),
                image::ImageFormat::Jpeg,
            )
            .decode()
            .map_err(|_| Error::Fetch("Cannot decode browser screenshot".into()))?;
            let image = image.resize(
                width as u32,
                options.max_height as u32,
                image::imageops::FilterType::Lanczos3,
            );
            let mut resized = Vec::new();
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut resized, options.quality as u8)
                .encode_image(&image)
                .map_err(|_| Error::Fetch("Cannot resize browser screenshot".into()))?;
            bytes = resized;
        }
        total = total
            .checked_add(bytes.len())
            .ok_or_else(|| Error::Fetch("Browser screenshot size overflow".into()))?;
        if total > MAX_SCREENSHOT_BYTES {
            return Err(Error::Fetch("Browser screenshots exceed 100 MiB".into()));
        }
        let name = if index == 0 {
            primary.clone()
        } else {
            format!("{}--{index}.jpg", primary.trim_end_matches(".jpg"))
        };
        result.push(Asset { name, bytes });
    }
    Ok(result)
}

fn final_http_url(value: &str) -> Result<Url> {
    let url = Url::parse(value)
        .map_err(|_| Error::Fetch("Chromium returned an invalid final URL".into()))?;
    if !["http", "https"].contains(&url.scheme())
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err(Error::Fetch(
            "Browser final document must be HTTP(S) without embedded credentials".into(),
        ));
    }
    Ok(url)
}

pub(crate) fn fetch(source: &str, cfg: &Value, screenshot: bool) -> Result<BrowserResponse> {
    let runtime = crate::BrowserRuntime::new(1)?;
    fetch_with_runtime(source, cfg, screenshot, Some(&runtime))
}

pub(crate) fn fetch_with_runtime(
    source: &str,
    cfg: &Value,
    screenshot: bool,
    runtime: Option<&crate::BrowserRuntime>,
) -> Result<BrowserResponse> {
    let Some(runtime) = runtime else {
        return fetch(source, cfg, screenshot);
    };
    let url = Url::parse(source).map_err(|_| Error::InvalidInput("Invalid browser URL".into()))?;
    let options = options::Options::from_config(cfg, &url, screenshot)?;
    let executable = discover().ok_or_else(|| {
        let configured = std::env::var_os("MARKITAI_BROWSER_EXECUTABLE").map(PathBuf::from);
        Error::Unsupported(missing_browser(configured.as_deref()))
    })?;
    let mut lease = runtime.pool.acquire(&executable, &url, cfg, &options)?;
    lease.browser().begin_page(&options)?;
    let result = fetch_page(source, cfg, screenshot, &url, &options, lease.browser());
    match result {
        Ok(mut response) => {
            if lease.finish().is_err() {
                let warnings = match &mut response {
                    BrowserResponse::Page(page) => &mut page.warnings,
                    BrowserResponse::Pdf(pdf) => &mut pdf.warnings,
                };
                warnings.push("Browser session cleanup failed; the completed response was retained and the process discarded.".into());
            }
            Ok(response)
        }
        Err(error) => Err(error),
    }
}

/// Why the browser could not navigate: Chromium's `net::ERR_*` name, which
/// says whether the name did not resolve, the connection was refused or reset,
/// the certificate failed or the site turned the browser away. Nothing but
/// that name is kept, so no address, header or credential can reach the
/// message.
/// A page with almost no text that runs a script: a challenge interstitial
/// rather than an error page with content of its own.
fn interstitial(browser: &mut cdp::Browser) -> Result<bool> {
    Ok(browser
        .evaluate("(() => { const t = (document.body && document.body.innerText || '').trim(); return t.length < 300 && document.scripts.length > 0; })()")?
        .as_bool()
        == Some(true))
}

fn navigation_failure(reason: &Value) -> String {
    match reason.as_str().map(str::trim).filter(|reason| {
        reason.len() <= 64
            && reason.strip_prefix("net::ERR_").is_some_and(|name| {
                !name.is_empty()
                    && name
                        .bytes()
                        .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
            })
    }) {
        Some(reason) => format!("Browser navigation failed: {reason}"),
        None => "Browser navigation failed".into(),
    }
}

fn fetch_page(
    source: &str,
    cfg: &Value,
    screenshot: bool,
    url: &Url,
    options: &options::Options,
    browser: &mut cdp::Browser,
) -> Result<BrowserResponse> {
    browser.deadline = Instant::now() + Duration::from_millis(options.timeout);
    let navigation = match browser.navigate(source)? {
        cdp::Navigation::Page(value) => value,
        cdp::Navigation::Pdf(pdf) => return Ok(BrowserResponse::Pdf(pdf)),
    };
    if navigation.get("isDownload").and_then(Value::as_bool) == Some(true) {
        return Err(Error::Fetch(
            "Browser navigation returned a download".into(),
        ));
    }
    if let Some(reason) = navigation.get("errorText") {
        return Err(Error::Fetch(navigation_failure(reason)));
    }
    browser.main_frame = navigation
        .get("frameId")
        .and_then(Value::as_str)
        .map(str::to_owned);
    loop {
        let state = browser.evaluate("document.readyState")?;
        let ready = match options.wait_for.as_str() {
            "load" => state.as_str() == Some("complete"),
            "networkidle" => {
                state.as_str().is_some_and(|state| state != "loading") && browser.idle()
            }
            _ => state.as_str().is_some_and(|state| state != "loading"),
        };
        if browser.navigation_failed {
            return Err(Error::Fetch("Chromium page crashed".into()));
        }
        if ready {
            break;
        }
        browser.pause(Duration::from_millis(25))?;
    }
    // An interstitial answers 403, 429 or 503 with a nearly empty page whose
    // own script sets a cookie and loads the page again, as it expects of a
    // browser (Zhihu, Cloudflare's "Just a moment"). That reload is awaited
    // like any navigation the page makes; nothing is solved or forged.
    if browser
        .status
        .is_some_and(|status| matches!(status, 403 | 429 | 503))
        && interstitial(browser)?
    {
        let deadline = Instant::now() + Duration::from_millis(options.timeout.min(15_000));
        while Instant::now() < deadline {
            browser.pause(Duration::from_millis(100))?;
            if browser.navigation_failed {
                return Err(Error::Fetch("Chromium page crashed".into()));
            }
            if browser.status.is_some_and(|status| status < 400)
                && browser
                    .evaluate("document.readyState")?
                    .as_str()
                    .is_some_and(|state| state != "loading")
            {
                break;
            }
        }
    }
    if let Some(status) = browser.status.filter(|status| *status >= 400) {
        return Err(Error::Fetch(format!(
            "Browser navigation returned HTTP {status}"
        )));
    }
    let mut warnings = Vec::new();
    browser.deadline =
        Instant::now() + Duration::from_millis(options.timeout + options.extra_wait + 5600);
    let mut selector_found = true;
    if let Some(selector) = &options.selector {
        let deadline = Instant::now() + Duration::from_millis(options.timeout.min(10_000));
        let selector = serde_json::to_string(selector)?;
        let expression = format!(
            "(() => {{ const e = document.querySelector({selector}); return !!e && !!(e.getClientRects().length) && getComputedStyle(e).visibility !== 'hidden'; }})()"
        );
        loop {
            if browser.evaluate(&expression)?.as_bool() == Some(true) {
                break;
            }
            if Instant::now() >= deadline {
                selector_found = false;
                warnings.push(
                    "Browser wait_for_selector timed out; extracted the available rendered page."
                        .into(),
                );
                break;
            }
            browser.pause(Duration::from_millis(50))?;
        }
    }
    if selector_found && options.extra_wait > 0 {
        browser.pause(Duration::from_millis(options.extra_wait))?;
    }
    if !options.skip_scroll {
        browser.evaluate("(async () => { let height = document.body?.scrollHeight || 0; for (let i=0;i<8;i++) { window.scrollTo(0,height); await new Promise(r=>setTimeout(r,600)); const next=document.body?.scrollHeight || 0; if(next===height) break; height=next; } window.scrollTo(0,0); await new Promise(r=>setTimeout(r,800)); return true; })()")?;
    }
    let final_url = browser
        .evaluate("location.href")?
        .as_str()
        .ok_or_else(|| Error::Fetch("Chromium returned no final URL".into()))?
        .to_owned();
    final_http_url(&final_url)?;
    let mut screenshots = Vec::new();
    if screenshot {
        browser.deadline = Instant::now() + Duration::from_millis(options.timeout);
        match capture(browser, options, url) {
            Ok(assets) => screenshots = assets,
            Err(error)
                if crate::config::enabled(cfg, "/screenshot/screenshot_only")
                    && !(crate::config::enabled(cfg, "/llm/enabled")
                        && crate::config::enabled(cfg, "/llm/pure")) =>
            {
                return Err(error);
            }
            Err(_) => {
                warnings.push("Browser screenshot failed; rendered text remains available.".into())
            }
        }
    }
    browser.deadline = Instant::now() + Duration::from_millis(options.timeout);
    // Flatten open shadow roots only after capture, so the screenshot retains
    // its original styling. Closed roots and browser-internal content stay opaque.
    let page = browser.evaluate("(() => { const roots=[document], flattened=new WeakSet(); let seen=0; while(roots.length) { const root=roots.pop(); const walker=document.createTreeWalker(root,NodeFilter.SHOW_ELEMENT); const hosts=[]; let node; while((node=walker.nextNode())) { if(++seen>200000) throw new Error('DOM limit'); if(node.shadowRoot && !flattened.has(node)) { flattened.add(node); hosts.push(node); } } for(const host of hosts) { while(host.shadowRoot.firstChild) host.appendChild(host.shadowRoot.firstChild); roots.push(host); } } const html=document.documentElement.outerHTML; if(html.length>52428800) throw new Error('DOM size limit'); return {html,title:document.title}; })()")?;
    let html = page
        .get("html")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Fetch("Browser returned no rendered DOM".into()))?
        .to_owned();
    if html.len() > MAX_HTML {
        return Err(Error::Fetch("Browser HTML exceeds 100 MiB".into()));
    }
    let title = page
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    Ok(BrowserResponse::Page(BrowserPage {
        html,
        final_url,
        title,
        screenshots,
        warnings,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_missing_browser_names_the_configured_executable_that_is_not_one() {
        let configured = missing_browser(Some(Path::new("/opt/none/chrome")));
        assert!(
            configured.contains("MARKITAI_BROWSER_EXECUTABLE is /opt/none/chrome"),
            "{configured}"
        );
        assert!(missing_browser(None).starts_with("Chromium is not installed"));
    }
    #[test]
    fn a_navigation_failure_keeps_chromiums_reason_and_nothing_else() {
        assert_eq!(
            navigation_failure(&json!("net::ERR_NAME_NOT_RESOLVED")),
            "Browser navigation failed: net::ERR_NAME_NOT_RESOLVED"
        );
        for other in [
            json!("net::ERR_FAILED at https://user:secret@example.test/?token=x"),
            json!("see https://example.test/"),
            json!("net::ERR_\u{0}"),
            json!(format!("net::ERR_{}", "A".repeat(80))),
            json!(7),
            json!(null),
        ] {
            assert_eq!(navigation_failure(&other), "Browser navigation failed");
        }
    }

    #[test]
    fn screenshot_names_distinguish_queries_and_hash_routes_not_plain_anchors() {
        let base = filename(&Url::parse("https://example.com/a/b").unwrap());
        assert_eq!(base, "example.com_a_b.full.jpg");
        assert_eq!(
            base,
            filename(&Url::parse("https://example.com/a/b#heading").unwrap())
        );
        assert_ne!(
            base,
            filename(&Url::parse("https://example.com/a/b#/route").unwrap())
        );
        assert_ne!(
            filename(&Url::parse("https://example.com/a?x=1").unwrap()),
            filename(&Url::parse("https://example.com/a?x=2").unwrap())
        );
    }
    #[test]
    fn invalid_session_and_malformed_credentials_fail_before_browser_discovery() {
        let cfg = json!({"fetch":{"playwright":{"session_mode":"invalid"}}});
        assert!(
            fetch("http://127.0.0.1/", &cfg, false)
                .err()
                .unwrap()
                .to_string()
                .contains("session_mode")
        );
        let cfg = json!({"fetch":{"playwright":{"http_credentials":{"username":"private","password":123}}}});
        let error = fetch("http://127.0.0.1/", &cfg, false)
            .err()
            .unwrap()
            .to_string();
        assert!(!error.contains("private") && !error.contains("123"));
        assert!(http_credentials_configured(&cfg));
        assert!(!http_credentials_configured(&json!({})));
        assert!(!http_credentials_configured(
            &json!({"fetch":{"playwright":{"http_credentials":null}}})
        ));
    }

    #[test]
    fn redirected_documents_reject_embedded_credentials_and_non_http_schemes() {
        for target in [
            "https://user:private-value@example.test/",
            "https://user@example.test/",
            "file:///private/data",
        ] {
            let error = final_http_url(target).unwrap_err().to_string();
            assert!(!error.contains("private-value"));
        }
        assert!(final_http_url("https://example.test/page?token=private-value").is_ok());
    }
}
