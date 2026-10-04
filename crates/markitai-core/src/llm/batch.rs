//! Frozen text plans reuse the live document contract without executing live calls.
use super::*;
use crate::{ConversionFailure, DetailedResult};
use serde::{Deserialize, Serialize};

const PLAN_VERSION: u32 = 1;
const MAX_DOCUMENT_BYTES: usize = 8 * 1024 * 1024;

/// The endpoint is configuration identity, never a provider-returned download URL.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub provider: String,
    pub api_base: String,
    pub model: String,
}

/// Credentials live only in this process and cannot be serialized with a plan.
pub struct Session {
    entry: Deployment,
    identity: Endpoint,
    pool: String,
    timeout: Duration,
    can_submit: bool,
}

impl Session {
    pub fn new(cfg: &Value) -> Result<Self> {
        Self::configured(cfg, &config::environment())
    }

    /// Reading an existing cloud job cannot start a new paid model request.
    pub fn for_collection(cfg: &Value) -> Result<Self> {
        Self::configured_for(cfg, &config::environment(), false)
    }

    fn configured(cfg: &Value, env: &HashMap<String, String>) -> Result<Self> {
        Self::configured_for(cfg, env, true)
    }

    fn configured_for(
        cfg: &Value,
        env: &HashMap<String, String>,
        submitting: bool,
    ) -> Result<Self> {
        if cfg
            .pointer("/llm/router_settings/fallbacks")
            .and_then(Value::as_array)
            .is_some_and(|groups| !groups.is_empty())
        {
            return Err(Error::Unsupported(
                "Provider Batch requires one effective deployment without fallback groups".into(),
            ));
        }
        if submitting
            && cfg
                .pointer("/llm/max_cost_per_document_usd")
                .and_then(Value::as_f64)
                .is_some_and(|limit| limit > 0.0)
        {
            return Err(Error::Unsupported(
                "Provider Batch cannot enforce a continuation dollar budget after cloud submission"
                    .into(),
            ));
        }
        let entries = deployments(cfg, env)?;
        let expected = cfg
            .pointer("/llm/model_list")
            .and_then(Value::as_array)
            .filter(|entries| !entries.is_empty())
            .map(|entries| {
                entries
                    .iter()
                    .filter(|entry| {
                        entry
                            .pointer("/litellm_params/weight")
                            .and_then(Value::as_u64)
                            .unwrap_or(1)
                            > 0
                    })
                    .count()
            });
        if expected.is_some_and(|count| count != entries.len()) {
            return Err(Error::Config(
                "Provider Batch has unresolved enabled model configuration".into(),
            ));
        }
        let first = entries.first().ok_or(Error::NoModelConfigured)?;
        if first.provider != "openai" || first.protocol != Protocol::Chat {
            return Err(Error::Unsupported(
                "This Provider Batch implementation supports OpenAI text requests".into(),
            ));
        }
        if entries.iter().any(|entry| {
            entry.provider != first.provider
                || entry.model != first.model
                || entry.endpoint != first.endpoint
                || entry.key != first.key
                || entry.max_tokens != first.max_tokens
        }) {
            return Err(Error::Unsupported(
                "Provider Batch requires one effective model, endpoint and account".into(),
            ));
        }
        if first.key.as_deref().is_none_or(str::is_empty) {
            return Err(Error::Config(
                "Provider Batch requires an API key for the configured deployment".into(),
            ));
        }
        let mut endpoint = url::Url::parse(&first.endpoint)
            .map_err(|_| Error::Config("Provider Batch endpoint is invalid".into()))?;
        if !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            return Err(Error::Config(
                "Provider Batch endpoint cannot contain user information, query or fragment".into(),
            ));
        }
        let base_path = endpoint
            .path()
            .strip_suffix("/chat/completions")
            .ok_or_else(|| {
                Error::Config("Provider Batch requires a chat completions endpoint".into())
            })?
            .to_owned();
        endpoint.set_path(&base_path);
        let identity = Endpoint {
            provider: first.provider.clone(),
            api_base: endpoint.to_string().trim_end_matches('/').into(),
            model: first.model.clone(),
        };
        let timeout = cfg
            .pointer("/llm/router_settings/timeout")
            .and_then(Value::as_u64)
            .unwrap_or(120);
        let session = Self {
            entry: first.clone(),
            identity,
            pool: llm_cache::model_scope(entries.iter().map(|entry| entry.id.as_str())),
            timeout: Duration::from_secs(timeout),
            can_submit: submitting,
        };
        // Constructing the transport validates the endpoint/key/timeout, without I/O.
        session.client()?;
        Ok(session)
    }

    pub fn identity(&self) -> &Endpoint {
        &self.identity
    }

    pub fn client(&self) -> Result<crate::provider_batch::Client> {
        self.client_with_timeout(self.timeout)
    }

    pub fn client_with_timeout(
        &self,
        remaining: Duration,
    ) -> Result<crate::provider_batch::Client> {
        crate::provider_batch::Client::new(
            &self.identity.api_base,
            self.entry.key.as_deref().unwrap_or(""),
            self.timeout.min(remaining),
            crate::provider_batch::Limits::default(),
        )
        .map_err(|error| Error::Config(error.to_string()))
    }

    /// Validate the whole text capability before upload; no silent live fallback.
    pub fn prepare(
        &self,
        markdown: &str,
        source: &str,
        metadata: &serde_json::Map<String, Value>,
        cfg: &Value,
    ) -> Result<Prepared> {
        if !self.can_submit {
            return Err(Error::Unsupported(
                "A collection session cannot prepare new provider requests".into(),
            ));
        }
        for path in [
            "/llm/pure",
            "/ocr/enabled",
            "/screenshot/enabled",
            "/image/alt_enabled",
            "/image/desc_enabled",
        ] {
            if config::enabled(cfg, path) {
                return Err(Error::Unsupported("Provider Batch currently requires structured text without pure, OCR, screenshots or image analysis".into()));
            }
        }
        if markdown.trim().is_empty() || markdown.len() > MAX_DOCUMENT_BYTES || source.len() > 8192
        {
            return Err(Error::InvalidInput(
                "Provider Batch document is empty or exceeds its planning limit".into(),
            ));
        }
        let protected = chunks::Protected::new(markdown);
        let limit = chunks::limit(cfg, || document::prompt_tokens(source, false, cfg))?;
        if protected.split_within(limit).len() != 1 {
            return Err(Error::Unsupported("Provider Batch text currently requires one document request; use live processing for long documents".into()));
        }
        let prompts = document::document_prompts(&protected.text, source, false, cfg)?;
        let key = llm_cache::document_key(&protected.text, &prompts.cache_scope, &self.pool);
        let flags = structured::capabilities(&self.entry);
        let mode = if flags.0 {
            structured::Mode::Tools
        } else if flags.1 {
            structured::Mode::JsonSchema
        } else {
            structured::Mode::JsonText
        };
        let wire = structured::Wire {
            mode,
            schema: structured::Schema::Document,
        };
        let mut plan = Plan {
            version: PLAN_VERSION,
            endpoint: self.identity.clone(),
            markdown: markdown.into(),
            source: source.into(),
            metadata: metadata.clone(),
            profile: cfg
                .pointer("/output/profile")
                .and_then(Value::as_str)
                .map(str::to_owned),
            wikilinks: config::enabled(cfg, "/output/wikilinks"),
            page_markers: cfg.pointer("/output/page_markers").and_then(Value::as_bool),
            mode,
            request: wire.payload(&self.entry, &prompts),
            cache_key: key,
            pool: self.pool.clone(),
            warnings: Vec::new(),
        };
        plan.validate()?;
        if let Some(cache) = llm_cache::Cache::configured(cfg, source) {
            match cache.get_json(&plan.cache_key) {
                Ok(Some(value)) => match plan.answer(&value, true) {
                    Ok(answer) => return Ok(Prepared::Cached(plan.finish(answer, ConversionUsage::default(), true))),
                    Err(_) => plan.warnings.push("A malformed document cache entry was ignored.".into()),
                },
                Ok(None) => (),
                Err(_) => plan.warnings.push("Persistent LLM cache is unavailable; Batch preparation continued without a cached answer.".into()),
            }
        }
        Ok(Prepared::Request(Box::new(plan)))
    }
}

pub enum Prepared {
    Cached(DecodedDocument),
    Request(Box<Plan>),
}

/// Sensitive document material; callers persist it privately, separately from credentials.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    version: u32,
    endpoint: Endpoint,
    markdown: String,
    source: String,
    metadata: serde_json::Map<String, Value>,
    profile: Option<String>,
    wikilinks: bool,
    // Missing in older saved plans means the historical enabled default.
    #[serde(default, alias = "slide_markers")]
    page_markers: Option<bool>,
    mode: structured::Mode,
    request: Value,
    cache_key: String,
    pool: String,
    warnings: Vec<String>,
}

impl Plan {
    pub fn identity(&self) -> &Endpoint {
        &self.endpoint
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != PLAN_VERSION
            || self.endpoint.provider != "openai"
            || self.markdown.trim().is_empty()
            || self.markdown.len() > MAX_DOCUMENT_BYTES
            || self.source.len() > 8192
            || self.request.get("model").and_then(Value::as_str)
                != Some(self.endpoint.model.as_str())
            || self
                .profile
                .as_deref()
                .is_some_and(|profile| !matches!(profile, "rag" | "obsidian" | "okf"))
            || self
                .request
                .get("messages")
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty)
            || self.cache_key.len() > 256
            || self.pool.len() > 256
            || chunks::Protected::new(&self.markdown).split().len() != 1
        {
            return Err(Error::InvalidInput(
                "Frozen Provider Batch plan is invalid".into(),
            ));
        }
        let endpoint = url::Url::parse(&self.endpoint.api_base)
            .map_err(|_| Error::InvalidInput("Frozen Provider Batch endpoint is invalid".into()))?;
        if !matches!(endpoint.scheme(), "http" | "https")
            || endpoint.host_str().is_none()
            || !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            return Err(Error::InvalidInput(
                "Frozen Provider Batch endpoint is invalid".into(),
            ));
        }
        Ok(())
    }

    pub fn request(&self, custom_id: &str) -> Result<Value> {
        self.validate()?;
        if custom_id.is_empty()
            || custom_id.len() > 64
            || !custom_id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'))
        {
            return Err(Error::InvalidInput(
                "Provider Batch custom ID is invalid".into(),
            ));
        }
        Ok(
            json!({"custom_id":custom_id,"method":"POST","url":"/v1/chat/completions","body":self.request}),
        )
    }

    fn deployment(&self) -> Deployment {
        Deployment {
            id: format!("openai/{}", self.endpoint.model),
            explicit_id: None,
            group: "default".into(),
            model: self.endpoint.model.clone(),
            provider: "openai".into(),
            weight: 1,
            key: None,
            endpoint: format!("{}/chat/completions", self.endpoint.api_base),
            protocol: Protocol::Chat,
            max_tokens: None,
            supports_vision: None,
        }
    }

    /// Usage is observed before provider/structured validation; callers deduplicate
    /// it by immutable job/custom-ID result identity before publishing anything.
    pub fn decode(
        &self,
        item: &crate::provider_batch::ResultItem,
        cfg: &Value,
    ) -> DetailedResult<DecodedDocument> {
        let usage = self.observe(item)?;
        self.decode_recorded(item, cfg, usage)
    }

    pub fn observe(&self, item: &crate::provider_batch::ResultItem) -> Result<ConversionUsage> {
        self.validate()?;
        let mut usage = ConversionUsage::default();
        if let Some(body) = &item.body
            && (body.get("usage").is_some_and(Value::is_object)
                || item
                    .http_status
                    .is_some_and(|code| (200..300).contains(&code)))
        {
            super::record_usage_class(
                &mut usage,
                &self.deployment(),
                body,
                crate::pricing::BillingClass::Batch,
            );
        }
        Ok(usage)
    }

    /// A collector persists the observed usage first and supplies that same quote
    /// on replay, even when a later executable has a newer tariff snapshot.
    pub fn decode_recorded(
        &self,
        item: &crate::provider_batch::ResultItem,
        cfg: &Value,
        usage: ConversionUsage,
    ) -> DetailedResult<DecodedDocument> {
        self.validate()?;
        let result = (|| {
            if item.error.is_some()
                || !item
                    .http_status
                    .is_some_and(|code| (200..300).contains(&code))
            {
                return Err(Error::Conversion(
                    "Provider Batch request did not succeed; observed usage is retained".into(),
                ));
            }
            let body = item.body.as_ref().ok_or_else(|| {
                Error::Conversion("Provider Batch response body is missing".into())
            })?;
            let text = structured::Wire {
                mode: self.mode,
                schema: structured::Schema::Document,
            }
            .decode(Protocol::Chat, body)
            .map_err(|failure| failure.error)?;
            let answer = self.answer(&structured::parse(&text)?, false)?;
            let mut warnings: Vec<String> = answer.salvaged.iter().cloned().collect();
            if answer.salvaged.is_none()
                && let Some(cache) = llm_cache::Cache::configured(cfg, &self.source)
                && cache
                    .set_json(&self.cache_key, &self.pool, &answer.value())
                    .is_err()
            {
                warnings.push(
                    "Persistent LLM cache could not save a Batch document; processing succeeded."
                        .into(),
                );
            }
            let mut decoded = self.finish(answer, copy_usage(&usage), false);
            decoded.warnings.extend(warnings);
            Ok(decoded)
        })();
        result.map_err(|error| ConversionFailure { error, usage })
    }

    fn answer(&self, value: &Value, cached: bool) -> Result<document::Answer> {
        let protected = chunks::Protected::new(&self.markdown);
        document::checked(
            document::parse_value(value, cached)?,
            &protected,
            &protected.text,
            false,
            false,
        )
    }

    fn finish(
        &self,
        answer: document::Answer,
        usage: ConversionUsage,
        cache_hit: bool,
    ) -> DecodedDocument {
        // answer() proved the exact marker sequence before restoration.
        let protected = chunks::Protected::new(&self.markdown);
        let mut markdown = crate::markdown::normalize(
            &protected
                .restore(&answer.markdown)
                .expect("validated protected markers"),
        );
        if self.page_markers == Some(false) {
            markdown = crate::output_profiles::remove_page_markers(&markdown);
        }
        let mut metadata = self.metadata.clone();
        metadata.insert("description".into(), json!(answer.metadata.description));
        metadata.insert("tags".into(), json!(answer.metadata.tags));
        crate::output_profiles::apply(
            &mut markdown,
            &mut metadata,
            &json!({"output":{"profile":self.profile,"wikilinks":self.wikilinks}}),
        );
        let mut warnings = self.warnings.clone();
        if !usage.cost_complete() {
            warnings.push("Some observed LLM requests could not be priced. cost_usd is the known priced subtotal; the complete cost is unknown.".into());
        }
        DecodedDocument {
            markdown,
            metadata,
            usage,
            cache_hit,
            warnings,
        }
    }
}

pub struct DecodedDocument {
    pub markdown: String,
    pub metadata: serde_json::Map<String, Value>,
    pub usage: ConversionUsage,
    pub cache_hit: bool,
    pub warnings: Vec<String>,
}
impl DecodedDocument {
    pub fn content(&self) -> Result<String> {
        crate::output::render(&self.metadata, &self.markdown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Value {
        json!({"llm":{"model_list":[{"litellm_params":{"model":"openai/gpt-4.1","api_key":"fixture-api-secret"}}]},"cache":{"enabled":false}})
    }
    fn plan(cfg: &Value, text: &str) -> Plan {
        let session = Session::configured(cfg, &HashMap::new()).unwrap();
        match session
            .prepare(text, "source.md", &serde_json::Map::new(), cfg)
            .unwrap()
        {
            Prepared::Request(plan) => *plan,
            Prepared::Cached(_) => panic!("cache is disabled"),
        }
    }
    fn response(plan: &Plan, text: &str) -> crate::provider_batch::ResultItem {
        let answer = json!({"cleaned_markdown":text,"frontmatter":{"description":"A fixture document", "tags":["fixture"]}});
        crate::provider_batch::ResultItem {
            custom_id: "doc-0".into(),
            http_status: Some(200),
            body: Some(
                json!({"model":plan.endpoint.model,"usage":{"prompt_tokens":1000,"completion_tokens":100},
                "choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[{"type":"function","function":{"name":"MarkitaiDocument","arguments":answer.to_string()}}]}}]}),
            ),
            error: None,
            request_id: Some("request-fixture".into()),
        }
    }
    #[test]
    fn frozen_plan_roundtrip_excludes_credentials_and_preserves_request_and_markers() {
        let cfg = config();
        let text = "# Document\n\nSee [the source](https://example.test/a).\n\n```rust\nlet value = 7;\n```\n";
        let plan = plan(&cfg, text);
        let saved = serde_json::to_string(&plan).unwrap();
        assert!(!saved.contains("fixture-api-secret"));
        let restored: Plan = serde_json::from_str(&saved).unwrap();
        assert_eq!(
            restored.request("doc-0").unwrap(),
            plan.request("doc-0").unwrap()
        );
        let protected = chunks::Protected::new(text);
        let decoded = restored
            .decode(&response(&restored, &protected.text), &cfg)
            .unwrap();
        assert!(decoded.markdown.contains("let value = 7;"));
        assert!(decoded.markdown.contains("https://example.test/a"));
        assert!(!decoded.markdown.contains("⟦MKTI:"));
        assert_eq!(decoded.usage.requests, 1);
        assert_eq!(decoded.usage.input_tokens, 1000);
        assert!((decoded.usage.cost_usd - 0.0014).abs() < 1e-12);
        assert!(
            decoded
                .content()
                .unwrap()
                .contains("description: A fixture document")
        );
    }
    #[test]
    fn frozen_page_marker_preference_applies_only_after_restoring_literals() {
        let mut cfg = config();
        cfg["output"] = json!({"page_markers":false});
        let text = "<!-- Slide number: 1 -->\n# Title\n\n```html\n<!-- Slide number: 2 -->\n```\n";
        let original = plan(&cfg, text);
        assert_eq!(original.markdown, text);
        let protected = chunks::Protected::new(text);
        let mut saved = serde_json::to_value(&original).unwrap();
        let restored: Plan = serde_json::from_value(saved.clone()).unwrap();
        // Collection-time settings must not change the frozen output preference.
        let mut collect = config();
        collect["output"] = json!({"page_markers":true});
        let decoded = restored
            .decode(&response(&restored, &protected.text), &collect)
            .unwrap();
        assert!(!decoded.markdown.contains("<!-- Slide number: 1 -->"));
        assert!(
            decoded
                .markdown
                .contains("```html\n<!-- Slide number: 2 -->\n```")
        );
        // A plan saved under the earlier development name keeps its choice.
        let mut renamed = saved.clone();
        let choice = renamed
            .as_object_mut()
            .unwrap()
            .remove("page_markers")
            .unwrap();
        renamed["slide_markers"] = choice;
        let renamed: Plan = serde_json::from_value(renamed).unwrap();
        let decoded = renamed
            .decode(&response(&renamed, &protected.text), &collect)
            .unwrap();
        assert!(!decoded.markdown.contains("<!-- Slide number: 1 -->"));
        saved.as_object_mut().unwrap().remove("page_markers");
        let legacy: Plan = serde_json::from_value(saved).unwrap();
        let decoded = legacy
            .decode(&response(&legacy, &protected.text), &cfg)
            .unwrap();
        assert!(decoded.markdown.contains("<!-- Slide number: 1 -->"));
    }

    #[test]
    fn invalid_paid_answers_and_http_failures_retain_batch_usage() {
        let cfg = config();
        let plan = plan(&cfg, "# Source\n\n[link](https://example.test)");
        let invalid = response(&plan, "# Changed without required link");
        let failure = plan.decode(&invalid, &cfg).err().unwrap();
        assert_eq!(failure.usage.requests, 1);
        assert_eq!(failure.usage.input_tokens, 1000);
        let mut rejected = invalid;
        rejected.http_status = Some(429);
        let failure = plan.decode(&rejected, &cfg).err().unwrap();
        assert_eq!(failure.usage.requests, 1);
        assert!((failure.usage.cost_usd - 0.0014).abs() < 1e-12);
        rejected.body = None;
        assert_eq!(
            plan.decode(&rejected, &cfg).err().unwrap().usage.requests,
            0
        );
    }
    #[test]
    fn planning_rejects_ambiguous_accounts_unresolved_models_and_unsupported_work() {
        let mut cfg = config();
        let second =
            json!({"litellm_params":{"model":"openai/gpt-4.1","api_key":"another-account"}});
        cfg["llm"]["model_list"]
            .as_array_mut()
            .unwrap()
            .push(second);
        assert!(Session::configured(&cfg, &HashMap::new()).is_err());
        cfg["llm"]["model_list"][1]["litellm_params"]["api_key"] = json!("env:MISSING_FIXTURE_KEY");
        assert!(Session::configured(&cfg, &HashMap::new()).is_err());
        let mut cfg = config();
        cfg["llm"]["max_cost_per_document_usd"] = json!(0.01);
        assert!(Session::configured(&cfg, &HashMap::new()).is_err());
        let collection = Session::configured_for(&cfg, &HashMap::new(), false).unwrap();
        assert!(
            collection
                .prepare("# Source", "source.md", &Default::default(), &cfg)
                .is_err()
        );
        let cfg = config();
        let session = Session::configured(&cfg, &HashMap::new()).unwrap();
        assert!(
            session
                .prepare(
                    &"meaningful prose ".repeat(4000),
                    "source.md",
                    &Default::default(),
                    &cfg
                )
                .is_err()
        );
        let mut cfg = cfg;
        cfg["image"] = json!({"alt_enabled":true});
        assert!(
            session
                .prepare("# Source", "source.md", &Default::default(), &cfg)
                .is_err()
        );
    }
    #[test]
    fn exact_cached_answer_avoids_a_new_request_and_preserves_validation() {
        let private = tempfile::tempdir().unwrap();
        let mut cfg = config();
        cfg["cache"] = json!({"enabled":true,"global_dir":private.path()});
        let session = Session::configured(&cfg, &HashMap::new()).unwrap();
        let plan = match session
            .prepare("# Source", "source.md", &Default::default(), &cfg)
            .unwrap()
        {
            Prepared::Request(plan) => plan,
            _ => panic!("first call must miss"),
        };
        plan.decode(&response(&plan, "# Source"), &cfg).unwrap();
        match session
            .prepare("# Source", "source.md", &Default::default(), &cfg)
            .unwrap()
        {
            Prepared::Cached(cached) => {
                assert!(cached.cache_hit);
                assert_eq!(cached.usage.requests, 0);
                assert_eq!(cached.usage.cost_usd, 0.0);
            }
            _ => panic!("successful typed result must reuse the same live cache identity"),
        }
    }
}
