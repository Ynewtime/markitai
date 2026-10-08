use super::{State, failure::Failure, jobs};
use crate::diagnostics::{AttemptDiagnostics, Operation};
use markitai_core::{ConversionOutput, ConvertContext, ConvertOptions, LlmRuntime};
use rmcp::model::Tool;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::{
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone, Default, Deserialize)]
pub(super) struct Options {
    pub llm: Option<bool>,
    pub ocr: Option<bool>,
    pub screenshot: Option<bool>,
    pub alt: Option<bool>,
    pub desc: Option<bool>,
    pub profile: Option<String>,
}

#[derive(Deserialize)]
struct Document {
    path: String,
    output_dir: Option<String>,
    #[serde(flatten)]
    options: Options,
}
#[derive(Deserialize)]
struct Url {
    url: String,
    output_dir: Option<String>,
    #[serde(flatten)]
    options: Options,
}
#[derive(Deserialize)]
struct Batch {
    sources: Vec<String>,
    output_dir: Option<String>,
    #[serde(default, deserialize_with = "integral")]
    concurrency: Option<i64>,
    #[serde(flatten)]
    options: Options,
}
#[derive(Deserialize)]
struct Status {
    job_id: String,
}

/// JSON Schema `integer` admits any number with a zero fractional part, such
/// as `2.0`, as the published schema and the reference validator do.
fn integral<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<i64>, D::Error> {
    use serde::de::Error;
    let Some(number) = Option::<serde_json::Number>::deserialize(deserializer)? else {
        return Ok(None);
    };
    if let Some(value) = number.as_i64() {
        return Ok(Some(value));
    }
    match number.as_f64() {
        Some(value) if value.fract() == 0.0 && value.abs() < 9_007_199_254_740_992.0 => {
            Ok(Some(value as i64))
        }
        _ => Err(D::Error::custom(format!(
            "expected an integer, got {number}"
        ))),
    }
}

fn arguments<T: serde::de::DeserializeOwned>(value: Value) -> Result<T, String> {
    serde_json::from_value(value).map_err(|error| format!("Invalid tool arguments: {error}"))
}

pub(super) fn is_url(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

fn absolute(path: &str, field: &str) -> Result<PathBuf, String> {
    let path = markitai_core::config::expand_home(Path::new(path));
    if !path.is_absolute() {
        return Err(format!(
            "{field} must be absolute — the MCP server's working directory is not the client's."
        ));
    }
    Ok(path)
}

fn workdir(path: Option<&str>) -> Result<PathBuf, String> {
    match path.filter(|path| !path.is_empty()) {
        Some(path) => absolute(path, "output_dir"),
        None => {
            let mut builder = tempfile::Builder::new();
            builder.prefix("markitai-mcp-");
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                builder.permissions(std::fs::Permissions::from_mode(0o700));
            }
            builder
                .tempdir()
                .map(tempfile::TempDir::keep)
                .map_err(|error| error.to_string())
        }
    }
}

fn validate_options(options: &Options) -> Result<(), String> {
    if let Some(profile) = &options.profile
        && !matches!(profile.as_str(), "rag" | "obsidian" | "okf")
    {
        return Err("profile must be rag, obsidian, okf or null".into());
    }
    Ok(())
}

pub(super) async fn dispatch(
    state: &Arc<State>,
    name: &str,
    value: Value,
) -> Result<Value, Failure> {
    match name {
        "convert_document" => {
            let args: Document = arguments(value)?;
            validate_options(&args.options)?;
            let source = absolute(&args.path, "path")?.to_string_lossy().into_owned();
            let directory = workdir(args.output_dir.as_deref())?;
            state.convert(source, directory, args.options, None).await
        }
        "convert_url" => {
            let args: Url = arguments(value)?;
            validate_options(&args.options)?;
            if !is_url(&args.url) {
                return Err("url must start with http:// or https:// — for local files use convert_document.".into());
            }
            let directory = workdir(args.output_dir.as_deref())?;
            state.convert(args.url, directory, args.options, None).await
        }
        "batch_convert" => {
            let args: Batch = arguments(value)?;
            validate_options(&args.options)?;
            if args.sources.is_empty() {
                return Err("sources must contain at least one path or URL.".into());
            }
            if args.sources.len() > 10_000 {
                return Err("sources exceeds the 10000-item MCP job limit".into());
            }
            for source in &args.sources {
                if !is_url(source) {
                    absolute(source, "sources")?;
                }
            }
            let directory = workdir(args.output_dir.as_deref())?;
            let concurrency = args.concurrency.unwrap_or(10).max(1) as u64;
            let concurrency = concurrency.min(args.sources.len() as u64) as usize;
            jobs::start(state, args.sources, directory, args.options, concurrency)
                .map_err(Into::into)
        }
        "job_status" => {
            let args: Status = arguments(value)?;
            state
                .jobs
                .lock()
                .unwrap()
                .snapshot(&args.job_id)
                .map_err(Into::into)
        }
        _ => Err(format!("Unknown tool: {name}").into()),
    }
}

pub(super) fn convert(
    state: &State,
    source: &str,
    directory: PathBuf,
    options: Options,
    runtime: Option<&LlmRuntime>,
) -> Result<Value, Failure> {
    let cfg = state.config()?;
    let source = if is_url(source) {
        source.to_owned()
    } else {
        absolute(source, "path")?.to_string_lossy().into_owned()
    };
    let skip_paths = existing_paths(&source, &directory, &cfg);
    let allow_symlinks = markitai_core::config::enabled(&cfg, "/output/allow_symlinks");
    let mut output = markitai_core::convert_with_context_detailed(
        &source,
        ConvertOptions {
            output_dir: Some(directory.clone()),
            config: Some(cfg),
            llm: options.llm,
            ocr: options.ocr,
            screenshot: options.screenshot,
            alt: options.alt,
            desc: options.desc,
            profile: options.profile,
        },
        ConvertContext {
            llm_runtime: runtime,
            browser_runtime: Some(&state.browser_runtime),
            ..Default::default()
        },
    )
    .map_err(Failure::from)?;
    if output.skip_reason.as_deref() == Some("exists")
        && let Some((base, enhanced)) = skip_paths
    {
        if let Some((frontmatter, markdown)) = read_existing(&enhanced, allow_symlinks)
            .map_err(|error| Failure::observed(error, output.usage.clone()))?
        {
            output.llm_output_path = Some(enhanced);
            output.llm_markdown = Some(markdown);
            output.frontmatter = frontmatter;
        }
        if let Some((frontmatter, markdown)) = read_existing(&base, allow_symlinks)
            .map_err(|error| Failure::observed(error, output.usage.clone()))?
        {
            output.output_path = Some(base);
            output.markdown = markdown;
            if output.frontmatter.is_empty() {
                output.frontmatter = frontmatter;
            }
        }
    }
    Ok(project(output, &directory))
}

fn existing_paths(source: &str, directory: &Path, cfg: &Value) -> Option<(PathBuf, PathBuf)> {
    if cfg.pointer("/output/on_conflict").and_then(Value::as_str) != Some("skip") {
        return None;
    }
    let source_name = if is_url(source) {
        markitai_core::output::url_name(source)
    } else {
        Path::new(source)
            .file_name()?
            .to_string_lossy()
            .into_owned()
    };
    let name = cfg
        .pointer("/output/filename")
        .and_then(Value::as_str)
        .map(|name| name.strip_suffix(".md").unwrap_or(name))
        .or_else(|| cfg.pointer("/output/reserved_stem").and_then(Value::as_str))
        .unwrap_or(&source_name);
    Some((
        directory.join(format!("{name}.md")),
        directory.join(format!("{name}.llm.md")),
    ))
}

type ExistingMarkdown = (Map<String, Value>, String);

fn read_existing(path: &Path, allow_symlinks: bool) -> Result<Option<ExistingMarkdown>, String> {
    markitai_core::output::check_path(path, allow_symlinks).map_err(|error| error.to_string())?;
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    const MAX_EXISTING_BYTES: u64 = 500 * 1024 * 1024;
    if !metadata.is_file() {
        return Err("Existing Markdown output must be a regular file".into());
    }
    if metadata.len() > MAX_EXISTING_BYTES {
        return Err("Existing Markdown output exceeds the 500 MiB read limit".into());
    }
    let mut text = String::new();
    std::fs::File::open(path)
        .map_err(|error| error.to_string())?
        .take(MAX_EXISTING_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|error| error.to_string())?;
    if text.len() as u64 > MAX_EXISTING_BYTES {
        return Err("Existing Markdown output exceeds the 500 MiB read limit".into());
    }
    let (frontmatter, markdown) = markitai_core::output::split_frontmatter(&text);
    Ok(Some((frontmatter, markdown.to_owned())))
}

fn project(output: ConversionOutput, directory: &Path) -> Value {
    let text = output.llm_markdown.unwrap_or(output.markdown);
    let truncated = text.chars().take(40_001).count() > 40_000;
    let text = if truncated {
        text.chars().take(2_000).collect()
    } else {
        text
    };
    let mut value = json!({"source":output.source,"markdown":text,"truncated":truncated,
        "markdown_file":output.llm_output_path.or(output.output_path),"output_dir":directory,
        "assets":output.assets,"screenshots":output.screenshots,"cost_usd":output.usage.cost_usd,
        "skip_reason":output.skip_reason,"duration_s":(output.duration * 100.0).round_ties_even() / 100.0,
        "warnings":output.warnings});
    if let Some(pricing) = crate::pricing::Pricing::from_usage(&output.usage) {
        value["pricing"] = json!(pricing);
    }
    if let Some(diagnostics) = AttemptDiagnostics::completed(Operation::Convert, output.usage) {
        value["diagnostics"] = json!(diagnostics);
    }
    value
}

fn object(properties: Value, required: &[&str]) -> Map<String, Value> {
    json!({"type":"object","properties":properties,"required":required})
        .as_object()
        .unwrap()
        .clone()
}

fn diagnostics_schema() -> Value {
    let counters = object(
        json!({
            "cost_usd":{"type":"number","minimum":0},
            "requests":{"type":"integer","minimum":0},
            "input_tokens":{"type":"integer","minimum":0},
            "output_tokens":{"type":"integer","minimum":0}
        }),
        &["cost_usd", "requests", "input_tokens", "output_tokens"],
    );
    let mut model = counters.clone();
    model["properties"].as_object_mut().unwrap().extend(
        json!({
            "priced_requests":{"type":"integer","minimum":0},
            "unpriced_requests":{"type":"integer","minimum":0},
            "cost_status":{"type":"string","enum":["complete","partial","unknown"]},
            "pricing_snapshot":{"type":"string"},
            "pricing_snapshots":{"type":"array","items":{"type":"string"}}
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    let mut usage = counters.clone();
    usage["properties"].as_object_mut().unwrap().insert(
        "by_model".into(),
        json!({"type":"object","additionalProperties":model}),
    );
    usage["required"]
        .as_array_mut()
        .unwrap()
        .push(json!("by_model"));
    json!({"type":"object","required":["last_attempt"],"properties":{
        "last_attempt":{"type":"object","required":["operation","status","error","usage"],"properties":{
            "operation":{"type":"string","enum":["convert"]},
            "status":{"type":"string","enum":["done","error"]},
            "error":{"type":["string","null"]},"usage":usage
        }}
    }})
}

fn with_error_schema(success: Map<String, Value>) -> Map<String, Value> {
    let failure = object(
        json!({
            "error":{"type":"string"},"diagnostics":diagnostics_schema()
        }),
        &["error", "diagnostics"],
    );
    // Tool failures stay application results. Their structured diagnostic value
    // does not have the mandatory fields of a successful conversion or job.
    json!({"type":"object","anyOf":[success,failure]})
        .as_object()
        .unwrap()
        .clone()
}

pub(super) fn definitions() -> Vec<Tool> {
    let conversion_output = object(
        json!({
            "source":{"type":"string"},"markdown":{"type":"string"},"truncated":{"type":"boolean"},
            "markdown_file":{"anyOf":[{"type":"string"},{"type":"null"}]},"output_dir":{"type":"string"},
            "assets":{"type":"array","items":{"type":"string"}},"screenshots":{"type":"array","items":{"type":"string"}},
            "cost_usd":{"type":"number"},"skip_reason":{"anyOf":[{"type":"string"},{"type":"null"}]},
            "duration_s":{"type":"number"},"warnings":{"type":"array","items":{"type":"string"}},
            "diagnostics":diagnostics_schema(),"pricing":crate::pricing::schema()
        }),
        &[
            "source",
            "markdown",
            "truncated",
            "markdown_file",
            "output_dir",
            "assets",
            "screenshots",
            "cost_usd",
            "skip_reason",
            "duration_s",
            "warnings",
        ],
    );
    let batch_output = object(
        json!({"job_id":{"type":"string"},"status":{"type":"string"},"total":{"type":"integer"},"output_dir":{"type":"string"}}),
        &["job_id", "status", "total", "output_dir"],
    );
    let status_output = object(
        json!({"job_id":{"type":"string"},"status":{"type":"string"},"total":{"type":"integer"},"done":{"type":"integer"},"failed":{"type":"integer"},"output_dir":{"type":"string"},"pricing":crate::pricing::schema(),"cost_usd":{"type":"number"},"results":{"type":"array","items":{"type":"object","properties":{"diagnostics":diagnostics_schema(),"pricing":crate::pricing::schema()}}}}),
        &[
            "job_id",
            "status",
            "total",
            "done",
            "failed",
            "output_dir",
            "results",
        ],
    );
    let mut tools = Vec::new();
    for (name, source, description) in [
        (
            "convert_document",
            "path",
            "Convert one local document. path and output_dir must be absolute; ~ is expanded. Omit output_dir for a persistent temporary directory. Large Markdown is returned as a preview: read markdown_file for the full document.",
        ),
        (
            "convert_url",
            "url",
            "Convert an HTTP(S) page. output_dir must be absolute if supplied. Large Markdown is returned as a preview: read markdown_file for the full document.",
        ),
        (
            "batch_convert",
            "sources",
            "Convert a nonempty list of absolute file paths and HTTP(S) URLs in the background. Poll job_status. Items have isolated numbered directories and results retain input order. concurrency defaults to 10; jobs live only in this server process.",
        ),
    ] {
        let mut properties = Map::new();
        properties.insert(
            source.into(),
            if source == "sources" {
                json!({"type":"array","items":{"type":"string"}})
            } else {
                json!({"type":"string"})
            },
        );
        properties.insert(
            "output_dir".into(),
            json!({"anyOf":[{"type":"string"},{"type":"null"}],"default":null}),
        );
        for key in ["llm", "ocr", "screenshot", "alt", "desc"] {
            properties.insert(
                key.into(),
                json!({"anyOf":[{"type":"boolean"},{"type":"null"}],"default":null}),
            );
        }
        properties.insert("profile".into(), json!({"anyOf":[{"type":"string","enum":["rag","obsidian","okf"]},{"type":"null"}],"default":null}));
        if source == "sources" {
            properties.insert(
                "concurrency".into(),
                json!({"anyOf":[{"type":"integer"},{"type":"null"}],"default":null}),
            );
        }
        let description = format!(
            "{description} Omitted/null feature flags follow server configuration. llm=true requires MODEL and a provider API key or configured llm.model_list; false disables enhancement. Unsupported core features fail explicitly."
        );
        tools.push(
            Tool::new(
                name,
                description,
                object(Value::Object(properties), &[source]),
            )
            .with_raw_output_schema(Arc::new(with_error_schema(
                if source == "sources" {
                    batch_output.clone()
                } else {
                    conversion_output.clone()
                },
            ))),
        );
    }
    tools.push(Tool::new("job_status", "Return batch progress and finished results in input order. done counts successes and failures; failed counts errors. Unknown and forgotten jobs have distinct errors. Files remain on disk after job eviction or restart.", object(json!({"job_id":{"type":"string"}}), &["job_id"]))
        .with_raw_output_schema(Arc::new(with_error_schema(status_output))));
    tools
}

#[cfg(test)]
mod pricing_tests {
    use super::*;
    #[test]
    fn conversion_projection_distinguishes_known_zero_unknown_and_no_request() {
        for (model, expected) in [
            (
                json!({"requests":1,"input_tokens":0,"output_tokens":0,"cost_usd":0.0,"priced_requests":1,"unpriced_requests":0,"cost_status":"complete","pricing_snapshot":"catalog-v1"}),
                "complete",
            ),
            (
                json!({"requests":1,"input_tokens":0,"output_tokens":0,"cost_usd":0.0}),
                "unknown",
            ),
        ] {
            let usage = markitai_core::ConversionUsage {
                requests: 1,
                by_model: json!({"m":model}).as_object().unwrap().clone(),
                ..Default::default()
            };
            let value = project(
                {
                    let mut result = ConversionOutput::default();
                    result.usage = usage;
                    result
                },
                Path::new("/isolated/output"),
            );
            assert_eq!(value["cost_usd"], 0.0);
            assert_eq!(value["pricing"]["cost_status"], expected);
            assert_eq!(value["diagnostics"]["last_attempt"]["usage"]["requests"], 1);
        }
        let empty = project(ConversionOutput::default(), Path::new("/isolated/output"));
        assert!(empty.get("pricing").is_none());
        assert!(empty.get("diagnostics").is_none());
    }
}

#[cfg(test)]
mod argument_tests {
    use super::*;

    #[test]
    fn batch_concurrency_follows_json_schema_integer_semantics() {
        let parse = |value: Value| {
            arguments::<Batch>(json!({"sources":["/a.md"],"concurrency":value}))
                .map(|batch| batch.concurrency)
        };
        assert_eq!(parse(json!(2)), Ok(Some(2)));
        assert_eq!(parse(json!(2.0)), Ok(Some(2)));
        assert_eq!(parse(json!(-3.0)), Ok(Some(-3)));
        assert_eq!(parse(Value::Null), Ok(None));
        assert_eq!(
            arguments::<Batch>(json!({"sources":["/a.md"]})).map(|b| b.concurrency),
            Ok(None)
        );
        for rejected in [json!(2.5), json!("2"), json!(true), json!(1e300)] {
            assert!(parse(rejected.clone()).is_err(), "{rejected}");
        }
    }
}
