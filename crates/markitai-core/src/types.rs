use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Input file not found: {0}")]
    NotFound(String),
    #[error("Input is a directory: {0}")]
    IsDirectory(String),
    #[error("No model configured; set MODEL and a provider API key, or llm.model_list")]
    NoModelConfigured,
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Unsupported(String),
    #[error("{0}")]
    Conversion(String),
    #[error("{0}")]
    ImageOnly(String),
    #[error("{0}")]
    Fetch(String),
    #[error("{0}")]
    Config(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl Error {
    pub fn code(&self) -> &'static str {
        match self {
            Self::NotFound(_) => "not_found",
            Self::IsDirectory(_) => "is_directory",
            Self::NoModelConfigured => "no_model_configured",
            Self::InvalidInput(_) => "invalid_input",
            Self::Unsupported(_) => "unsupported",
            Self::Conversion(_) | Self::ImageOnly(_) => "conversion_error",
            Self::Fetch(_) => "fetch_error",
            Self::Config(_) => "config_error",
            Self::Io(_) => "io_error",
            Self::Json(_) => "invalid_json",
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// A failed conversion's existing error category and already recorded model usage.
/// Costs remain subject to the same pricing limitations as successful results.
#[derive(Debug, thiserror::Error)]
#[error("{error}")]
pub struct ConversionFailure {
    #[source]
    pub error: Error,
    pub usage: ConversionUsage,
}

impl From<Error> for ConversionFailure {
    fn from(error: Error) -> Self {
        Self {
            error,
            usage: ConversionUsage::default(),
        }
    }
}

impl ConversionFailure {
    pub fn code(&self) -> &'static str {
        self.error.code()
    }
}

pub type DetailedResult<T> = std::result::Result<T, ConversionFailure>;

#[derive(Clone, Debug, Serialize)]
pub struct LlmCapabilities {
    pub configured: bool,
    pub routable: bool,
    pub effective: bool,
    pub models: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConvertOptions {
    pub output_dir: Option<PathBuf>,
    pub config: Option<Value>,
    pub llm: Option<bool>,
    pub ocr: Option<bool>,
    pub screenshot: Option<bool>,
    pub alt: Option<bool>,
    pub desc: Option<bool>,
    pub profile: Option<String>,
}

/// Native run context that is not part of the host JSON protocol.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ConvertContext<'a> {
    pub explicit_fetch_strategy: Option<&'a str>,
    pub llm_runtime: Option<&'a crate::LlmRuntime>,
    pub browser_runtime: Option<&'a crate::BrowserRuntime>,
    /// Store for the images of a conversion without an output directory (the
    /// CLI's stdout mode): referenced images are saved there under content
    /// hashes and the Markdown links to them with `file://` URIs. Ignored when
    /// an output directory is given; library callers leave it unset.
    pub stdout_assets: Option<&'a std::path::Path>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub source: String,
    #[serde(default)]
    pub options: ConvertOptions,
}

#[derive(Debug, Default)]
pub struct Document {
    pub markdown: String,
    pub metadata: Map<String, Value>,
    pub assets: Vec<Asset>,
    pub warnings: Vec<String>,
}

#[derive(Debug)]
pub struct Asset {
    pub name: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ConversionUsage {
    pub cost_usd: f64,
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub by_model: Map<String, Value>,
}

impl ConversionUsage {
    /// Whether every observed request has a recorded reviewed tariff quote.
    /// Legacy records with requests but no completeness metadata remain unknown.
    pub fn cost_complete(&self) -> bool {
        if self.by_model.values().any(|row| {
            row.get("incomplete_request_observations")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                > 0
        }) {
            return false;
        }
        if self.requests == 0 {
            return true;
        }
        let priced = self.by_model.values().try_fold(0u64, |total, row| {
            let requests = row.get("requests")?.as_u64()?;
            let priced = row.get("priced_requests")?.as_u64()?;
            let unpriced = row.get("unpriced_requests")?.as_u64()?;
            (priced == requests
                && unpriced == 0
                && row.get("cost_status")?.as_str()? == "complete")
                .then_some(())?;
            total.checked_add(priced)
        });
        priced == Some(self.requests)
    }

    /// Whether an official subscription runtime (Claude, ChatGPT, Copilot)
    /// served any observed request; its rows carry no dollar quote.
    pub fn subscription_observed(&self) -> bool {
        self.by_model.keys().any(|key| subscription_row(key))
    }

    /// Whether requests served by priced-API models lack a complete tariff
    /// quote. Subscription rows are covered by the subscription warning.
    pub fn has_unpriced_non_subscription(&self) -> bool {
        self.by_model.iter().any(|(key, row)| {
            !subscription_row(key)
                && (row
                    .get("incomplete_request_observations")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    > 0
                    || row
                        .get("unpriced_requests")
                        .and_then(Value::as_u64)
                        .unwrap_or(0)
                        > 0
                    || row.get("cost_status").and_then(Value::as_str) != Some("complete"))
        })
    }

    /// The rows that keep the cost incomplete, as `(model, requests, reason)`,
    /// for priced-API models only. The map iterates in key order, so the
    /// resulting warning is stable for one usage record.
    pub fn unpriced_breakdown(&self) -> Vec<(String, u64, UnpricedReason)> {
        let mut rows = Vec::new();
        for (model, row) in &self.by_model {
            if subscription_row(model)
                || row.get("cost_status").and_then(Value::as_str) == Some("complete")
            {
                continue;
            }
            let count = |key: &str| row.get(key).and_then(Value::as_u64).unwrap_or(0);
            let unreported = count("incomplete_request_observations");
            let unpriced = count("unpriced_requests");
            if unreported > 0 {
                rows.push((model.clone(), unreported, UnpricedReason::NoCounts));
            }
            if unpriced > 0 {
                rows.push((model.clone(), unpriced, UnpricedReason::NoTariff));
            }
            if unreported == 0 && unpriced == 0 {
                // Observations without counts, without a quote and without a
                // status still name a model the reader has to know about.
                rows.push((
                    model.clone(),
                    count("requests").max(1),
                    UnpricedReason::NoTariff,
                ));
            }
        }
        rows
    }

    /// The warning naming what could not be priced, or `None` when the cost is
    /// complete or only subscription rows lack a quote (those carry their own
    /// notice). Naming the model and the reason is what lets a reader act:
    /// "some requests" never says which price is missing.
    pub fn unpriced_warning(&self) -> Option<String> {
        let rows = self.unpriced_breakdown();
        if rows.is_empty() {
            return None;
        }
        // Three named models are enough to act on; the rest are counted.
        const NAMED: usize = 3;
        let mut parts: Vec<String> = rows
            .iter()
            .take(NAMED)
            .map(|(model, requests, reason)| {
                let count = if *requests == 1 {
                    "1 request".to_owned()
                } else {
                    format!("{requests} requests")
                };
                match reason {
                    UnpricedReason::NoTariff => {
                        let verb = if *requests == 1 { "has" } else { "have" };
                        format!("{count} to {model} {verb} no reviewed price")
                    }
                    UnpricedReason::NoCounts => {
                        format!("{count} to {model} reported no usage counts")
                    }
                }
            })
            .collect();
        if rows.len() > NAMED {
            parts.push(format!("{} more models", rows.len() - NAMED));
        }
        Some(format!(
            "Cost is incomplete: {}. cost_usd is the known priced subtotal; the complete cost is unknown.",
            parts.join("; ")
        ))
    }
}

/// Why an observed request has no complete price.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnpricedReason {
    /// The request was observed with counts, but no reviewed tariff covers it.
    NoTariff,
    /// The provider reported no usage counts for the request.
    NoCounts,
}

/// Usage rows of official subscription runtimes are keyed by these prefixes.
fn subscription_row(key: &str) -> bool {
    ["claude-agent/", "chatgpt/", "copilot/"]
        .iter()
        .any(|prefix| key.starts_with(prefix))
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ConversionOutput {
    pub source: String,
    pub markdown: String,
    pub llm_markdown: Option<String>,
    pub frontmatter: Map<String, Value>,
    pub output_path: Option<PathBuf>,
    pub llm_output_path: Option<PathBuf>,
    pub assets: Vec<PathBuf>,
    pub screenshots: Vec<PathBuf>,
    pub images: Vec<Value>,
    pub usage: ConversionUsage,
    pub skip_reason: Option<String>,
    pub duration: f64,
    pub warnings: Vec<String>,
    /// Original YAML bytes retained for writing pure input without reformatting.
    #[serde(skip)]
    pub(crate) pure_prefix: Option<String>,
    #[serde(skip)]
    pub(crate) pure_llm_prefix: Option<String>,
    #[serde(skip)]
    pub(crate) base_frontmatter: Option<Map<String, Value>>,
    /// Base text before output-only filtering, retained for chained enhancement.
    #[serde(skip)]
    pub(crate) enhancement_source: Option<String>,
    #[serde(skip)]
    pub(crate) llm_cache_hit: bool,
    #[serde(skip)]
    pub(crate) fetch_cache_hit: bool,
    #[serde(skip)]
    pub(crate) fetch_strategy: Option<String>,
}

impl ConversionOutput {
    /// Text for subsequent enhancement, before output-only page/slide-marker removal.
    /// Serialized results and published files always use the selected output form.
    pub fn enhancement_source(&self) -> &str {
        self.enhancement_source.as_deref().unwrap_or(&self.markdown)
    }

    /// Whether this conversion reused an existing document enhancement.
    /// Host JSON remains compatible; the CLI has separate cache-status fields.
    pub fn llm_cache_hit(&self) -> bool {
        self.llm_cache_hit
    }

    /// Whether a fetched page was reused directly or after HTTP validation.
    pub fn fetch_cache_hit(&self) -> bool {
        self.fetch_cache_hit
    }

    /// The strategy actually used to fetch a URL, including pure output mode.
    pub fn fetch_strategy(&self) -> Option<&str> {
        self.fetch_strategy.as_deref().or_else(|| {
            self.frontmatter
                .get("fetch_strategy")
                .and_then(Value::as_str)
        })
    }
}
