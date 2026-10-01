//! Stdio transport and lifetime ownership for the four conversion tools.
mod failure;
mod jobs;
mod tools;

use failure::Failure;

use rmcp::{
    ErrorData, RoleServer, ServerHandler, ServiceExt,
    model::{
        CacheScope, CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock,
        Implementation, ListToolsResult, PaginatedRequestParams, ProtocolVersion,
        ServerCapabilities, ServerConfig, Tool,
    },
    service::RequestContext,
};
use serde_json::Value;
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::task::JoinHandle;

struct Work {
    closing: bool,
    handles: Vec<crate::task::Blocking<()>>,
}

struct State {
    browser_runtime: markitai_core::BrowserRuntime,
    config_path: Option<PathBuf>,
    overrides: Option<Value>,
    jobs: Mutex<jobs::Table>,
    background: Mutex<Vec<JoinHandle<()>>>,
    work: Mutex<Work>,
    closing: AtomicBool,
}

impl State {
    fn config(&self) -> Result<Value, String> {
        markitai_core::config::load(self.config_path.as_deref(), self.overrides.clone())
            .map_err(|error| error.to_string())
    }

    async fn convert(
        self: &Arc<Self>,
        source: String,
        directory: PathBuf,
        options: tools::Options,
        runtime: Option<Arc<markitai_core::LlmRuntime>>,
    ) -> Result<Value, Failure> {
        let (send, receive) = tokio::sync::oneshot::channel();
        {
            let mut work = self.work.lock().unwrap();
            if work.closing {
                return Err("MCP server is shutting down".into());
            }
            work.handles.retain(|handle| !handle.is_finished());
            let state = self.clone();
            work.handles.push(crate::task::blocking(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    tools::convert(&state, &source, directory, options, runtime.as_deref())
                }))
                .unwrap_or_else(|_| Err("internal conversion error".into()));
                let _ = send.send(result);
            }));
        }
        receive
            .await
            .map_err(|_| "internal conversion error".to_owned())?
    }

    /// Stop admitting conversions and starting queued batch items; work
    /// already running continues.
    fn close_dispatch(&self) {
        self.closing.store(true, Ordering::SeqCst);
        self.work.lock().unwrap().closing = true;
    }

    async fn shutdown(&self) {
        self.close_dispatch();
        let tasks = std::mem::take(&mut *self.background.lock().unwrap());
        for task in tasks {
            let _ = task.await;
        }
        let tasks = std::mem::take(&mut self.work.lock().unwrap().handles);
        for task in tasks {
            let _ = task.await;
        }
        self.browser_runtime.close();
    }
}

/// The client's input, which calls `on_end` the first time it reaches its
/// end. A client closing its input ends the session; dispatch stops then,
/// not only once the MCP session has finished tearing down, so a batch
/// does not start a queued item (and a paid request) after the client left.
struct ClosingInput<R, F> {
    inner: R,
    on_end: Option<F>,
}

impl<R, F> ClosingInput<R, F> {
    fn new(inner: R, on_end: F) -> Self {
        ClosingInput {
            inner,
            on_end: Some(on_end),
        }
    }
}

impl<R: tokio::io::AsyncRead + Unpin, F: FnOnce() + Unpin> tokio::io::AsyncRead
    for ClosingInput<R, F>
{
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
        buffer: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let (room, before) = (buffer.remaining() > 0, buffer.filled().len());
        let poll = std::pin::Pin::new(&mut self.inner).poll_read(context, buffer);
        if room
            && matches!(poll, std::task::Poll::Ready(Ok(())))
            && buffer.filled().len() == before
            && let Some(on_end) = self.on_end.take()
        {
            on_end();
        }
        poll
    }
}

#[derive(Clone)]
struct Handler(Arc<State>);

impl ServerHandler for Handler {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("markitai", markitai_core::VERSION))
            .with_instructions("Convert local documents and HTTP(S) pages to Markdown. Use batch_convert and job_status for multiple sources. Results are written to disk; read markdown_file when the inline result is truncated. LLM requires MODEL and a provider key or configured llm.model_list.")
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let mut result = ListToolsResult {
            tools: tools::definitions(),
            ..Default::default()
        };
        // 2026-07-28 requires cache directives on tools/list. Like the
        // reference SDK default, the listing is immediately stale and private;
        // earlier protocol versions keep their original result shape.
        if context
            .protocol_version()
            .is_some_and(|version| version.as_str() >= ProtocolVersion::V_2026_07_28.as_str())
        {
            result = result.with_ttl_ms(0).with_cache_scope(CacheScope::Private);
        }
        Ok(result)
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        tools::definitions()
            .into_iter()
            .find(|tool| tool.name == name)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let name = request.name.as_ref();
        let arguments = Value::Object(request.arguments.unwrap_or_default());
        let result = tools::dispatch(&self.0, name, arguments).await;
        let result = match result {
            Ok(value) => {
                let text = serde_json::to_string_pretty(&value).expect("JSON value serializes");
                let mut result = CallToolResult::structured(value);
                result.content = vec![ContentBlock::text(text)];
                result
            }
            Err(failure) => {
                let mut result = CallToolResult::error(vec![ContentBlock::text(format!(
                    "Error executing tool {name}: {}",
                    failure.message
                ))]);
                if let Some(diagnostics) = failure.diagnostics {
                    result.structured_content = Some(serde_json::json!({
                        "error": failure.message, "diagnostics": diagnostics
                    }));
                }
                result
            }
        };
        Ok(result.into())
    }
}

pub fn run(config_path: Option<PathBuf>, overrides: Option<Value>) -> Result<(), String> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|_| "Cannot initialize MCP runtime".to_owned())?;
    let state = Arc::new(State {
        browser_runtime: markitai_core::BrowserRuntime::new(8)
            .map_err(|error| error.to_string())?,
        config_path,
        overrides,
        jobs: Mutex::new(jobs::Table::default()),
        background: Mutex::new(Vec::new()),
        work: Mutex::new(Work {
            closing: false,
            handles: Vec::new(),
        }),
        closing: AtomicBool::new(false),
    });
    let result = runtime.block_on(async {
        let handler = Handler(state.clone());
        let session = async {
            let service = handler
                .serve((
                    ClosingInput::new(tokio::io::stdin(), {
                        let state = state.clone();
                        move || state.close_dispatch()
                    }),
                    tokio::io::stdout(),
                ))
                .await
                .map_err(|_| "Cannot establish MCP stdio session".to_owned())?;
            service
                .waiting()
                .await
                .map_err(|_| "MCP session failed".to_owned())?;
            Ok(())
        };
        let result = tokio::select! {
            result = session => result,
            _ = tokio::signal::ctrl_c() => Ok(()),
        };
        state.shutdown().await;
        result
    });
    // All conversion work has drained. Tokio's stdin reader can remain blocked
    // after SIGINT until the parent closes its pipe; it must not delay exit.
    runtime.shutdown_timeout(Duration::from_millis(100));
    result
}

#[cfg(test)]
mod closing_input_tests {
    use super::ClosingInput;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::AsyncReadExt;

    #[tokio::test]
    async fn the_end_of_input_is_reported_once_and_only_at_the_end() {
        let ends = std::sync::Arc::new(AtomicUsize::new(0));
        let counter = ends.clone();
        let mut input = ClosingInput::new(&b"request\n"[..], move || {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        let mut line = [0u8; 4];
        input.read_exact(&mut line).await.unwrap();
        assert_eq!(ends.load(Ordering::SeqCst), 0);
        let mut rest = Vec::new();
        input.read_to_end(&mut rest).await.unwrap();
        assert_eq!(rest, b"est\n");
        assert_eq!(ends.load(Ordering::SeqCst), 1);
        // Reading past the end again does not report it twice.
        assert_eq!(input.read(&mut line).await.unwrap(), 0);
        assert_eq!(ends.load(Ordering::SeqCst), 1);
    }
}
