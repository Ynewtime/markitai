use super::options::Options;
use crate::{Error, Result};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tungstenite::{Message, WebSocket};
use url::Url;

const MAX_MESSAGE: usize = 140 * 1024 * 1024;
const ACCEPT_LANGUAGE: &str = "en-US,en;q=0.9";

fn failure(message: &str) -> Error {
    Error::Fetch(message.into())
}

fn private_profile() -> Result<TempDir> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("markitai-browser-");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(std::fs::Permissions::from_mode(0o700));
    }
    Ok(builder.tempdir()?)
}

struct Process {
    child: Child,
    group: crate::process_groups::Slot,
    _profile: TempDir,
}
impl Drop for Process {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            // The child owns a new process group; Chromium renderers belong to it.
            // Polling does not reap the leader, so its group id stays valid for
            // the final SIGKILL and for a concurrent fatal-signal cleanup.
            unsafe {
                libc::kill(-(self.child.id() as i32), libc::SIGTERM);
            }
            let deadline = Instant::now() + Duration::from_millis(500);
            while Instant::now() < deadline && !self.group.exited(&self.child) {
                std::thread::sleep(Duration::from_millis(20));
            }
            unsafe {
                libc::kill(-(self.child.id() as i32), libc::SIGKILL);
            }
        }
        self.group.retire();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Process {
    fn launch(executable: &Path, options: &Options) -> Result<(Self, u16, String)> {
        let profile = private_profile()?;
        let mut command = Command::new(executable);
        command
            .env_clear()
            .current_dir(profile.path())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        for name in [
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "TMPDIR",
            "TEMP",
            "TMP",
            "LANG",
            "LC_ALL",
            "TZ",
        ] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        command.args([
            "--headless=new",
            "--remote-debugging-port=0",
            "--remote-debugging-address=127.0.0.1",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-default-apps",
            "--disable-extensions",
            "--disable-sync",
            "--disable-client-side-phishing-detection",
            "--metrics-recording-only",
            "--disable-features=MediaRouter,OptimizationHints,AutofillServerCommunication",
            "--password-store=basic",
            "--use-mock-keychain",
        ]);
        command.arg(format!("--user-data-dir={}", profile.path().display()));
        command.arg(format!(
            "--disk-cache-dir={}",
            profile.path().join("cache").display()
        ));
        if let Some(proxy) = &options.proxy {
            command.arg(format!("--proxy-server={proxy}"));
            command.arg(format!("--proxy-bypass-list={}", options.bypass));
        } else {
            command.arg("--no-proxy-server");
        }
        command.arg("about:blank");
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let group = crate::process_groups::Slot::reserve()
            .ok_or_else(|| failure("Too many external runtime process groups are active"))?;
        let child = command.spawn().map_err(|_| failure("Cannot launch Chromium; check MARKITAI_BROWSER_EXECUTABLE or install Chrome/Chromium"))?;
        group.publish(&child);
        let mut process = Self {
            child,
            group,
            _profile: profile,
        };
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if process.group.try_reap(&mut process.child)?.is_some() {
                return Err(failure(
                    "Chromium exited before its debugging endpoint became ready",
                ));
            }
            let path = process._profile.path().join("DevToolsActivePort");
            if let Ok(bytes) = std::fs::read(&path)
                && bytes.len() < 4096
                && let Ok(text) = std::str::from_utf8(&bytes)
            {
                let mut lines = text.lines();
                if let (Some(port), Some(path)) = (lines.next(), lines.next())
                    && let Ok(port) = port.parse::<u16>()
                    && port > 0
                    && path.starts_with("/devtools/browser/")
                    && !path.contains(['\r', '\n', '?', '#'])
                {
                    return Ok((process, port, path.to_owned()));
                }
            }
            if Instant::now() >= deadline {
                return Err(failure("Chromium startup timed out"));
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }
}

pub(super) enum Navigation {
    Page(Value),
    Pdf(super::BrowserPdf),
}

pub(super) struct Browser {
    context: Option<String>,
    target: Option<String>,
    healthy: bool,
    socket: WebSocket<TcpStream>,
    _process: Option<Process>,
    session: String,
    next_id: u64,
    pub deadline: Instant,
    blocked: Vec<regex::Regex>,
    auth: super::auth::State,
    in_flight: HashSet<String>,
    pub last_network_change: Instant,
    pub main_frame: Option<String>,
    pub status: Option<u64>,
    pub navigation_failed: bool,
    navigation_id: Option<u64>,
    navigation_reply: Option<Value>,
    document_response_seen: bool,
    paused_document: Option<super::download::Response>,
}
impl Drop for Browser {
    fn drop(&mut self) {
        let _ = self
            .socket
            .get_mut()
            .set_write_timeout(Some(Duration::from_millis(100)));
        let _ = self.socket.send(Message::Text(
            json!({"id":u64::MAX,"method":"Browser.close"})
                .to_string()
                .into(),
        ));
    }
}
impl Browser {
    pub fn launch(executable: &Path, options: &Options) -> Result<Self> {
        let mut browser = Self::connect(executable, options)?;
        browser.begin_page(options)?;
        Ok(browser)
    }

    pub fn connect(executable: &Path, options: &Options) -> Result<Self> {
        let (process, port, path) = Process::launch(executable, options)?;
        let stream = TcpStream::connect_timeout(
            &SocketAddr::from((Ipv4Addr::LOCALHOST, port)),
            Duration::from_secs(3),
        )
        .map_err(|_| failure("Cannot connect to Chromium debugging endpoint"))?;
        stream.set_read_timeout(Some(Duration::from_secs(3)))?;
        stream.set_write_timeout(Some(Duration::from_secs(3)))?;
        let config = tungstenite::protocol::WebSocketConfig::default()
            .max_message_size(Some(MAX_MESSAGE))
            .max_frame_size(Some(MAX_MESSAGE));
        let (socket, _) = tungstenite::client::client_with_config(
            format!("ws://127.0.0.1:{port}{path}"),
            stream,
            Some(config),
        )
        .map_err(|_| failure("Cannot establish Chromium protocol connection"))?;
        let browser = Self {
            context: None,
            target: None,
            healthy: true,
            socket,
            _process: Some(process),
            session: String::new(),
            next_id: 1,
            deadline: Instant::now() + Duration::from_millis(options.timeout),
            blocked: options.blocked.clone(),
            auth: super::auth::State::new(options.credentials.clone()),
            in_flight: HashSet::new(),
            last_network_change: Instant::now(),
            main_frame: None,
            status: None,
            navigation_failed: false,
            navigation_id: None,
            navigation_reply: None,
            document_response_seen: false,
            paused_document: None,
        };
        Ok(browser)
    }

    pub fn begin_page(&mut self, options: &Options) -> Result<()> {
        if self.target.is_some() || !self.session.is_empty() || !self.healthy {
            return Err(failure("Browser session is not ready for a new page"));
        }
        self.deadline = Instant::now() + Duration::from_millis(options.timeout);
        self.blocked = options.blocked.clone();
        self.auth = super::auth::State::new(options.credentials.clone());
        self.in_flight.clear();
        self.last_network_change = Instant::now();
        self.main_frame = None;
        self.status = None;
        self.navigation_failed = false;
        self.navigation_id = None;
        self.navigation_reply = None;
        self.document_response_seen = false;
        self.paused_document = None;
        if self.context.is_none() {
            let context = self.call(
                "Target.createBrowserContext",
                json!({"disposeOnDetach":true}),
            )?;
            self.context = Some(
                context["browserContextId"]
                    .as_str()
                    .filter(|id| !id.is_empty() && id.len() <= 4096)
                    .ok_or_else(|| failure("Chromium did not create a private context"))?
                    .to_owned(),
            );
        }
        let target = self.call(
            "Target.createTarget",
            json!({"url":"about:blank", "browserContextId":self.context}),
        )?;
        let target_id = target["targetId"]
            .as_str()
            .filter(|id| !id.is_empty() && id.len() <= 4096)
            .ok_or_else(|| failure("Chromium did not create a page"))?
            .to_owned();
        self.target = Some(target_id.clone());
        let attached = self.call(
            "Target.attachToTarget",
            json!({"targetId":target_id,"flatten":true}),
        )?;
        self.session = attached["sessionId"]
            .as_str()
            .filter(|id| !id.is_empty() && id.len() <= 4096)
            .ok_or_else(|| failure("Chromium did not attach a page"))?
            .to_owned();
        for method in ["Page.enable", "Runtime.enable", "Network.enable"] {
            self.call(method, json!({}))?;
        }
        self.call("Network.setCacheDisabled", json!({"cacheDisabled":true}))?;
        self.call("Network.setBypassServiceWorker", json!({"bypass":true}))?;
        self.call(
            "Browser.setDownloadBehavior",
            json!({"behavior":"deny","browserContextId":self.context}),
        )?;
        self.call("Fetch.enable", json!({"patterns":[{"urlPattern":"*","requestStage":"Request"},{"urlPattern":"*","resourceType":"Document","requestStage":"Response"}],"handleAuthRequests":self.auth.enabled()}))?;
        self.call("Emulation.setDeviceMetricsOverride", json!({"width":options.width,"height":options.height,"deviceScaleFactor":1,"mobile":false}))?;
        // A site whose reader parses English dates and counters renders in
        // English whatever the system's language is; a header the caller
        // sends itself decides the language instead.
        let english = options.english
            && !options
                .headers
                .keys()
                .any(|name| name.eq_ignore_ascii_case("accept-language"));
        let mut headers = options.headers.clone();
        if english {
            self.call("Emulation.setLocaleOverride", json!({"locale":"en-US"}))?;
            headers.insert("Accept-Language".into(), json!(ACCEPT_LANGUAGE));
        }
        if !headers.is_empty() {
            self.call("Network.setExtraHTTPHeaders", json!({"headers":headers}))?;
        }
        let agent = match &options.user_agent {
            Some(agent) => Some(agent.clone()),
            None if options.regular_user_agent => {
                // The browser's own string, without the word that sites block.
                let version = self.call("Browser.getVersion", json!({}))?;
                version["userAgent"]
                    .as_str()
                    .filter(|agent| agent.len() <= 512)
                    .map(|agent| agent.replace("HeadlessChrome/", "Chrome/"))
            }
            None => None,
        };
        if let Some(agent) = agent {
            let mut parameters = json!({"userAgent":agent});
            if english {
                // A bare list: Chromium writes the weights itself.
                parameters["acceptLanguage"] = json!("en-US,en");
            }
            self.call("Network.setUserAgentOverride", parameters)?;
        }
        if !options.cookies.is_empty() {
            self.call("Network.setCookies", json!({"cookies":options.cookies}))?;
        }
        Ok(())
    }

    // Only a fully extracted page can be returned to the pool. A failed close
    // leaves ownership with the lease, whose Drop terminates the whole process.
    pub fn finish_page(&mut self, persistent: bool) -> Result<()> {
        if !self.healthy || self.navigation_failed {
            return Err(failure(
                "Browser page cannot be reused after an operation failure",
            ));
        }
        self.deadline = Instant::now() + Duration::from_secs(2);
        self.navigation_id = None;
        self.paused_document = None;
        self.target
            .take()
            .ok_or_else(|| failure("Browser has no active page"))?;
        let context = self
            .context
            .clone()
            .ok_or_else(|| failure("Browser has no active context"))?;
        if persistent {
            // A document may open other pages in its context. Closing just the
            // requested target would leave those scripts alive across leases.
            let targets = self.context_targets(&context)?;
            for (id, page) in targets {
                if page {
                    let response = self.call("Target.closeTarget", json!({"targetId":id}))?;
                    if response["success"].as_bool() != Some(true) {
                        return Err(failure("Browser could not close an owned page"));
                    }
                }
            }
            // Child frames disappear with their pages. Any surviving worker or
            // newly opened page prevents reuse; the lease drops the process.
            while !self.context_targets(&context)?.is_empty() {
                if Instant::now() >= self.deadline {
                    return Err(failure("Browser context retained active targets"));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        } else {
            // Disposing the whole isolated context closes popups as well and
            // does not run beforeunload handlers.
            self.call(
                "Target.disposeBrowserContext",
                json!({"browserContextId":context}),
            )?;
            self.context = None;
        }
        self.session.clear();
        self.auth = super::auth::State::new(None);
        self.blocked.clear();
        self.in_flight.clear();
        self.main_frame = None;
        self.navigation_reply = None;
        Ok(())
    }

    fn context_targets(&mut self, context: &str) -> Result<Vec<(String, bool)>> {
        let response = self.call("Target.getTargets", json!({}))?;
        let targets = response["targetInfos"]
            .as_array()
            .filter(|targets| targets.len() <= 256)
            .ok_or_else(|| failure("Browser context target inventory is invalid or too large"))?;
        let mut owned = Vec::new();
        for target in targets {
            if target["browserContextId"].as_str() != Some(context) {
                continue;
            }
            if owned.len() >= 128 {
                return Err(failure("Browser context target limit exceeded"));
            }
            let id = target["targetId"]
                .as_str()
                .filter(|id| !id.is_empty() && id.len() <= 4096)
                .ok_or_else(|| failure("Browser context target identity is invalid"))?;
            owned.push((id.to_owned(), target["type"].as_str() == Some("page")));
        }
        Ok(owned)
    }

    fn send(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        let mut message = json!({"id":id,"method":method,"params":params});
        if !self.session.is_empty()
            && !method.starts_with("Browser.")
            && !method.starts_with("Target.")
        {
            message["sessionId"] = json!(self.session);
        }
        self.socket
            .send(Message::Text(message.to_string().into()))
            .map_err(|_| failure("Chromium protocol write failed"))?;
        Ok(id)
    }
    fn event(&mut self, message: &Value) -> Result<()> {
        if self.navigation_id.is_some() && message["id"].as_u64() == self.navigation_id {
            self.navigation_reply = Some(message.clone());
            return Ok(());
        }
        if message.get("sessionId").and_then(Value::as_str) != Some(self.session.as_str()) {
            return Ok(());
        }
        let params = &message["params"];
        match message.get("method").and_then(Value::as_str).unwrap_or("") {
            "Fetch.authRequired" => {
                let response = self.auth.respond(params)?;
                self.send("Fetch.continueWithAuth", response)?;
            }
            "Fetch.requestPaused" => {
                let id = params["requestId"]
                    .as_str()
                    .ok_or_else(|| failure("Malformed Chromium request event"))?;
                let url = params
                    .pointer("/request/url")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let safe = Url::parse(url).is_ok_and(|url| {
                    ["http", "https", "data", "blob", "about"].contains(&url.scheme())
                });
                if !safe || self.blocked.iter().any(|pattern| pattern.is_match(url)) {
                    self.send(
                        "Fetch.failRequest",
                        json!({"requestId":id,"errorReason":"BlockedByClient"}),
                    )?;
                } else {
                    let main_document = self.navigation_id.is_some()
                        && params["resourceType"].as_str() == Some("Document")
                        && self.main_frame.as_deref() == params["frameId"].as_str();
                    if main_document && params.get("responseStatusCode").is_some() {
                        let status = params["responseStatusCode"].as_u64().unwrap_or(0);
                        if !matches!(status, 301 | 302 | 303 | 307 | 308) {
                            self.document_response_seen = true;
                            self.status = Some(status);
                            if let Some(response) = super::download::Response::candidate(params)? {
                                if self.paused_document.replace(response).is_some() {
                                    return Err(failure(
                                        "Chromium paused overlapping document downloads",
                                    ));
                                }
                                return Ok(());
                            }
                        }
                    }
                    self.send("Fetch.continueRequest", json!({"requestId":id}))?;
                }
            }
            "Network.requestWillBeSent" => {
                if let Some(id) = params["requestId"].as_str() {
                    self.in_flight.insert(id.to_owned());
                    self.last_network_change = Instant::now();
                }
            }
            "Network.loadingFinished" | "Network.loadingFailed" => {
                if let Some(id) = params["requestId"].as_str() {
                    self.in_flight.remove(id);
                }
                self.last_network_change = Instant::now();
            }
            "Network.responseReceived" if params["type"].as_str() == Some("Document") => {
                if self
                    .main_frame
                    .as_deref()
                    .is_none_or(|frame| params["frameId"].as_str() == Some(frame))
                {
                    self.status = params.pointer("/response/status").and_then(Value::as_u64);
                }
            }
            "Page.javascriptDialogOpening" => {
                self.send("Page.handleJavaScriptDialog", json!({"accept":false}))?;
            }
            "Inspector.targetCrashed" => self.navigation_failed = true,
            _ => {}
        }
        Ok(())
    }
    /// Response interception precedes Page.navigate completion. Pump both here
    /// so taking a body stream cannot deadlock behind the paused navigation.
    pub fn navigate(&mut self, source: &str) -> Result<Navigation> {
        let tree = self.call("Page.getFrameTree", json!({}))?;
        self.main_frame = Some(
            tree.pointer("/frameTree/frame/id")
                .and_then(Value::as_str)
                .ok_or_else(|| failure("Chromium returned no main frame"))?
                .to_owned(),
        );
        self.document_response_seen = false;
        self.navigation_reply = None;
        self.navigation_id = Some(self.send("Page.navigate", json!({"url":source}))?);
        loop {
            if let Some(response) = self.paused_document.take() {
                let body = self.read_document(&response)?;
                response.validate_length(&body)?;
                if response.is_pdf(&body) {
                    self.call(
                        "Fetch.failRequest",
                        json!({"requestId":response.id,"errorReason":"Aborted"}),
                    )?;
                    self.navigation_id = None;
                    self.navigation_reply = None;
                    return Ok(Navigation::Pdf(response.pdf(body)));
                }
                self.call("Fetch.fulfillRequest", response.fulfilled(&body))?;
            }
            if let Some(reply) = self.navigation_reply.as_ref() {
                if reply.get("error").is_some() {
                    return Err(failure("Chromium protocol rejected Page.navigate"));
                }
                let result = reply.get("result").cloned().unwrap_or_else(|| json!({}));
                if self.document_response_seen
                    || result.get("errorText").is_some()
                    || result["isDownload"].as_bool() == Some(true)
                {
                    self.navigation_id = None;
                    self.navigation_reply = None;
                    return Ok(Navigation::Page(result));
                }
            }
            if let Some(message) = self.receive()? {
                self.event(&message)?;
            }
        }
    }

    fn read_document(&mut self, response: &super::download::Response) -> Result<Vec<u8>> {
        let result = self.call(
            "Fetch.takeResponseBodyAsStream",
            json!({"requestId":response.id}),
        )?;
        let handle = result["stream"]
            .as_str()
            .filter(|handle| !handle.is_empty() && handle.len() <= 4096)
            .ok_or_else(|| failure("Chromium returned no download stream"))?;
        let read = |browser: &mut Self| -> Result<Vec<u8>> {
            let mut bytes = Vec::new();
            loop {
                let chunk = browser.call(
                    "IO.read",
                    json!({"handle":handle,"size":super::download::CHUNK}),
                )?;
                if super::download::append_chunk(&mut bytes, &chunk)? {
                    return Ok(bytes);
                }
            }
        };
        let bytes = read(self);
        let closed = self.call("IO.close", json!({"handle":handle}));
        match bytes {
            Ok(bytes) => {
                closed?;
                Ok(bytes)
            }
            Err(error) => Err(error),
        }
    }

    fn receive(&mut self) -> Result<Option<Value>> {
        let remaining = self
            .deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| failure("Browser operation timed out"))?;
        self.socket.get_mut().set_read_timeout(Some(remaining))?;
        match self
            .socket
            .read()
            .map_err(|_| failure("Browser operation timed out or connection closed"))?
        {
            Message::Text(text) => serde_json::from_str(&text)
                .map(Some)
                .map_err(|_| failure("Invalid Chromium protocol response")),
            Message::Ping(_) | Message::Pong(_) => Ok(None),
            Message::Close(_) => Err(failure("Chromium closed its page")),
            _ => Err(failure("Unexpected Chromium protocol frame")),
        }
    }

    pub fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        let result = self.call_inner(method, params);
        if result.is_err() {
            self.healthy = false;
        }
        result
    }

    fn call_inner(&mut self, method: &str, params: Value) -> Result<Value> {
        let remaining = self
            .deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| failure("Browser operation timed out"))?;
        self.socket.get_mut().set_write_timeout(Some(remaining))?;
        let id = self.send(method, params)?;
        loop {
            let remaining = self
                .deadline
                .checked_duration_since(Instant::now())
                .ok_or_else(|| failure("Browser operation timed out"))?;
            self.socket.get_mut().set_read_timeout(Some(remaining))?;
            let message = self
                .socket
                .read()
                .map_err(|_| failure("Browser operation timed out or connection closed"))?;
            match message {
                Message::Text(text) => {
                    let value: Value = serde_json::from_str(&text)
                        .map_err(|_| failure("Invalid Chromium protocol response"))?;
                    if value["id"].as_u64() == Some(id) {
                        if value.get("error").is_some() {
                            return Err(failure(&format!("Chromium protocol rejected {method}")));
                        }
                        return Ok(value.get("result").cloned().unwrap_or_else(|| json!({})));
                    }
                    self.event(&value)?;
                }
                Message::Ping(_) | Message::Pong(_) => {}
                Message::Close(_) => return Err(failure("Chromium closed its page")),
                _ => return Err(failure("Unexpected Chromium protocol frame")),
            }
        }
    }
    pub fn evaluate(&mut self, expression: &str) -> Result<Value> {
        let timeout = self
            .deadline
            .saturating_duration_since(Instant::now())
            .as_millis()
            .max(1);
        let result = self.call("Runtime.evaluate", json!({"expression":expression,"returnByValue":true,"awaitPromise":true,"timeout":timeout.min(u64::MAX as u128) as u64}))?;
        if result.get("exceptionDetails").is_some() {
            self.healthy = false;
            return Err(failure("Browser page evaluation failed"));
        }
        Ok(result
            .pointer("/result/value")
            .cloned()
            .unwrap_or(Value::Null))
    }
    #[cfg(test)]
    pub(super) fn test_context(&self) -> Option<&str> {
        self.context.as_deref()
    }

    #[cfg(test)]
    pub(super) fn test_process(&self) -> Option<(u32, std::path::PathBuf)> {
        self._process
            .as_ref()
            .map(|process| (process.child.id(), process._profile.path().to_owned()))
    }

    pub fn idle(&self) -> bool {
        self.in_flight.is_empty()
            && self.last_network_change.elapsed() >= Duration::from_millis(500)
    }
    pub fn pause(&mut self, duration: Duration) -> Result<()> {
        let millis = duration.as_millis().min(u64::MAX as u128) as u64;
        self.evaluate(&format!(
            "new Promise(resolve => setTimeout(() => resolve(true), {millis}))"
        ))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn mock(
        limit: usize,
        handler: impl FnOnce(WebSocket<TcpStream>) + Send + 'static,
    ) -> (Browser, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            handler(tungstenite::accept(stream).unwrap());
        });
        let stream = TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let config = tungstenite::protocol::WebSocketConfig::default()
            .max_message_size(Some(limit))
            .max_frame_size(Some(limit));
        let (socket, _) = tungstenite::client::client_with_config(
            format!("ws://{address}/mock"),
            stream,
            Some(config),
        )
        .unwrap();
        (
            Browser {
                context: None,
                target: None,
                healthy: true,
                socket,
                _process: None,
                session: "test-session".into(),
                next_id: 1,
                deadline: Instant::now() + Duration::from_secs(3),
                auth: super::super::auth::State::new(None),
                blocked: vec![regex::Regex::new(r"^https://example\.test/ads/").unwrap()],
                in_flight: HashSet::new(),
                last_network_change: Instant::now(),
                main_frame: Some("main".into()),
                status: None,
                navigation_failed: false,
                navigation_id: None,
                navigation_reply: None,
                document_response_seen: false,
                paused_document: None,
            },
            worker,
        )
    }
    fn receive(socket: &mut WebSocket<TcpStream>) -> Value {
        let message = socket.read().unwrap().into_text().unwrap();
        serde_json::from_str(&message).unwrap()
    }
    fn send(socket: &mut WebSocket<TcpStream>, value: Value) {
        socket
            .send(Message::Text(value.to_string().into()))
            .unwrap();
    }

    #[test]
    fn paused_requests_are_filtered_while_waiting_for_the_matching_reply() {
        let (mut browser, worker) = mock(4096, |mut socket| {
            for (url, expected) in [
                ("https://example.test/content", "Fetch.continueRequest"),
                ("https://example.test/ads/banner", "Fetch.failRequest"),
                ("file:///private/secret", "Fetch.failRequest"),
            ] {
                let command = receive(&mut socket);
                send(
                    &mut socket,
                    json!({"sessionId":"test-session","method":"Fetch.requestPaused","params":{"requestId":"request-1","request":{"url":url}}}),
                );
                let action = receive(&mut socket);
                assert_eq!(action["method"], expected);
                assert_eq!(action["params"]["requestId"], "request-1");
                send(&mut socket, json!({"id":action["id"],"result":{}}));
                send(
                    &mut socket,
                    json!({"id":command["id"],"result":{"complete":true}}),
                );
            }
        });
        for _ in 0..3 {
            assert_eq!(
                browser.call("Runtime.evaluate", json!({})).unwrap(),
                json!({"complete":true})
            );
        }
        worker.join().unwrap();
    }

    #[test]
    fn authentication_events_preserve_matching_replies_and_ignore_other_sessions() {
        let (mut browser, worker) = mock(4096, |mut socket| {
            let command = receive(&mut socket);
            let event = |session: &str| json!({"sessionId":session,"method":"Fetch.authRequired","params":{"requestId":"auth-1","request":{"url":"https://example.test/page"},"authChallenge":{"source":"Server","origin":"https://example.test","scheme":"basic","realm":"fixture"}}});
            send(&mut socket, event("other-session"));
            send(&mut socket, event("test-session"));
            let action = receive(&mut socket);
            assert_eq!(action["method"], "Fetch.continueWithAuth");
            assert_eq!(
                action["params"]["authChallengeResponse"],
                json!({"response":"ProvideCredentials","username":"u","password":"p"})
            );
            send(&mut socket, json!({"id":action["id"],"result":{}}));
            send(&mut socket, event("test-session"));
            let cancel = receive(&mut socket);
            assert_eq!(
                cancel["params"]["authChallengeResponse"],
                json!({"response":"CancelAuth"})
            );
            send(&mut socket, json!({"id":cancel["id"],"result":{}}));
            send(
                &mut socket,
                json!({"id":command["id"],"result":{"complete":true}}),
            );
        });
        browser.auth = super::super::auth::State::new(
            super::super::auth::parse(
                Some(&json!({"username":"u","password":"p"})),
                &Url::parse("https://example.test/").unwrap(),
            )
            .unwrap(),
        );
        assert_eq!(
            browser.call("Page.navigate", json!({})).unwrap(),
            json!({"complete":true})
        );
        worker.join().unwrap();
    }

    #[test]
    fn unrelated_sessions_and_iframe_status_do_not_change_main_navigation() {
        let (mut browser, worker) = mock(4096, |mut socket| {
            let command = receive(&mut socket);
            for (session, frame, status) in [
                ("other", "main", 503),
                ("test-session", "iframe", 404),
                ("test-session", "main", 201),
            ] {
                send(
                    &mut socket,
                    json!({"sessionId":session,"method":"Network.responseReceived","params":{"type":"Document","frameId":frame,"response":{"status":status}}}),
                );
            }
            send(
                &mut socket,
                json!({"sessionId":"test-session","method":"Network.requestWillBeSent","params":{"requestId":"pending"}}),
            );
            send(&mut socket, json!({"id":command["id"],"result":{}}));
        });
        browser.call("Page.getFrameTree", json!({})).unwrap();
        assert_eq!(browser.status, Some(201));
        assert!(!browser.idle());
        worker.join().unwrap();
    }

    #[test]
    fn protocol_errors_do_not_echo_page_or_credential_payloads() {
        let (mut browser, worker) = mock(4096, |mut socket| {
            let command = receive(&mut socket);
            send(
                &mut socket,
                json!({"id":command["id"],"error":{"message":"private-token and cookie-secret","code":-1}}),
            );
        });
        let error = browser
            .call("Network.setCookies", json!({}))
            .unwrap_err()
            .to_string();
        assert_eq!(error, "Chromium protocol rejected Network.setCookies");
        worker.join().unwrap();
    }

    #[test]
    fn oversized_protocol_messages_fail_instead_of_returning_partial_content() {
        let (mut browser, worker) = mock(256, |mut socket| {
            let command = receive(&mut socket);
            send(
                &mut socket,
                json!({"id":command["id"],"result":{"html":"x".repeat(1024)}}),
            );
        });
        assert!(browser.call("Runtime.evaluate", json!({})).is_err());
        worker.join().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn browser_profile_is_private_and_removed_when_ownership_ends() {
        use std::os::unix::fs::PermissionsExt;
        let profile = private_profile().unwrap();
        let path = profile.path().to_owned();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o700
        );
        std::fs::write(path.join("cookie-state"), b"isolated test state").unwrap();
        drop(profile);
        assert!(!path.exists());
    }
}
