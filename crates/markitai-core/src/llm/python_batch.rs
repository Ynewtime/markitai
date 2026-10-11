//! Provider Batch jobs that Python markitai 1.2.x submitted and left pending,
//! read so that a collection after upgrading publishes the answers Python
//! paid for. Python's request rows (its prompts, tool names and protected
//! placeholders) are kept exactly as submitted; this module reproduces only
//! what reading their answers needs: the state file, Python's placeholder
//! protection and the answer schema. Outputs then follow this build's own
//! rules (frontmatter, page markers, output profiles).
use super::batch::{DecodedDocument, Endpoint};
use super::*;
use crate::provider_batch::{Api, ResultItem};
use crate::{ConversionFailure, DetailedResult};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

const PLAN_VERSION: u32 = 1;
const MAX_ITEMS: usize = 50_000;
const MAX_TEXT: usize = 8 * 1024 * 1024;

/// What one Python batch request asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PythonKind {
    /// A document's text enhancement.
    Doc,
    /// A document enhanced with its page screenshots.
    Vision,
    /// The analysis of one image of a document.
    Image,
}

#[derive(Deserialize)]
struct RawItem {
    custom_id: String,
    source: String,
    #[serde(default)]
    input_md: String,
    base_md: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    image: String,
    #[serde(default)]
    answer: Option<Value>,
}
#[derive(Deserialize)]
struct RawState {
    batch_id: String,
    model: String,
    provider: String,
    items: Vec<RawItem>,
    #[serde(default)]
    collected_at: Option<String>,
    #[serde(default)]
    ended_status: Option<String>,
    #[serde(default)]
    alt_enabled: Option<bool>,
    #[serde(default)]
    desc_enabled: Option<bool>,
}

/// One item of a Python state, with its paths checked.
#[derive(Clone, Debug)]
pub struct PythonItem {
    pub custom_id: String,
    pub source: String,
    /// The LLM input Markdown, relative to the state directory (documents).
    pub input_md: String,
    /// The base Markdown, relative to the output directory.
    pub base_md: String,
    pub kind: PythonKind,
    /// The image, relative to the output directory (images).
    pub image: String,
    /// An image answer Python already had (from its cache): no request.
    pub answer: Option<Value>,
}

/// `<output>/.markitai/batch-<id>/state.json` of Python markitai 1.2.x.
#[derive(Clone, Debug)]
pub struct PythonState {
    pub batch_id: String,
    pub model: String,
    pub provider: String,
    pub items: Vec<PythonItem>,
    /// Python finished collecting it.
    pub collected: bool,
    /// The batch ended without results (failed, expired, cancelled).
    pub ended_status: Option<String>,
    pub alt_enabled: bool,
    pub desc_enabled: bool,
}

fn relative(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 4096
        && !path.contains('\0')
        && !path.contains('\\')
        && std::path::Path::new(path)
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)))
}

impl PythonState {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let invalid = |what: &str| Error::InvalidInput(format!("Python batch state {what}"));
        let raw: RawState =
            serde_json::from_slice(bytes).map_err(|_| invalid("is not valid JSON"))?;
        if !crate::provider_batch::batch_id(&raw.batch_id) {
            return Err(invalid("names an invalid batch ID"));
        }
        if Api::for_provider(&raw.provider).is_none() {
            return Err(Error::Unsupported(format!(
                "Python batch state uses provider '{}'; only openai and anthropic batches can be collected",
                raw.provider
            )));
        }
        if raw.model.trim().is_empty() || raw.model.len() > 256 || raw.items.is_empty() {
            return Err(invalid("has no model or no items"));
        }
        if raw.items.len() > MAX_ITEMS {
            return Err(invalid("has too many items"));
        }
        let mut ids = HashSet::new();
        let mut items = Vec::with_capacity(raw.items.len());
        for item in raw.items {
            let kind = match item.kind.as_deref().unwrap_or("doc") {
                "doc" => PythonKind::Doc,
                "vision" => PythonKind::Vision,
                "image" => PythonKind::Image,
                _ => return Err(invalid("has an item of an unknown kind")),
            };
            let custom_id_ok = !item.custom_id.is_empty()
                && item.custom_id.len() <= 64
                && item
                    .custom_id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'));
            if !custom_id_ok
                || !ids.insert(item.custom_id.clone())
                || item.source.is_empty()
                || item.source.len() > 8192
                || item.source.contains('\0')
                || !relative(&item.base_md)
                || !item.base_md.ends_with(".md")
                || item.base_md.ends_with(".llm.md")
                || item.base_md.starts_with(".markitai/")
                || match kind {
                    PythonKind::Image => !relative(&item.image),
                    _ => !relative(&item.input_md) || item.answer.is_some(),
                }
            {
                return Err(invalid("has an invalid item"));
            }
            items.push(PythonItem {
                custom_id: item.custom_id,
                source: item.source,
                input_md: item.input_md,
                base_md: item.base_md,
                kind,
                image: item.image,
                answer: item.answer.filter(|answer| answer.is_object()),
            });
        }
        Ok(Self {
            batch_id: raw.batch_id,
            model: raw.model,
            provider: raw.provider,
            items,
            collected: raw.collected_at.is_some(),
            ended_status: raw.ended_status,
            alt_enabled: raw.alt_enabled.unwrap_or(false),
            desc_enabled: raw.desc_enabled.unwrap_or(false),
        })
    }

    /// Image requests whose document has no request of its own (Python
    /// enhanced a too-long document live and batched only its images).
    pub fn orphan_images(&self) -> Vec<&PythonItem> {
        self.items
            .iter()
            .filter(|item| {
                item.kind == PythonKind::Image
                    && item.answer.is_none()
                    && !self
                        .items
                        .iter()
                        .any(|doc| doc.kind != PythonKind::Image && doc.base_md == item.base_md)
            })
            .collect()
    }
}

static IMAGE_LINK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"!\[[^\]]*\]\([^)]+\)").expect("valid image pattern"));
static PAGE_NUMBER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<!--\s*Page number:\s*(\d+)\s*-->").expect("valid page marker pattern")
});
static SLIDE_NUMBER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<!--\s*Slide number:\s*\d+\s*-->").expect("valid slide marker pattern")
});
static PAGE_COMMENTS: LazyLock<[Regex; 4]> = LazyLock::new(|| {
    [
        r"<!--\s*Page images for reference\s*-->",
        r"<!--\s*!\[Page\s+\d+\]\([^)]*\)\s*-->",
        r"<!--\s*Screenshot for reference\s*-->",
        r"<!--\s*!\[Screenshot\]\([^)]*\)\s*-->",
    ]
    .map(|pattern| Regex::new(pattern).expect("valid page comment pattern"))
});
static PLACEHOLDER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"__MARKITAI_[A-Z]+_\d+__").expect("valid placeholder pattern"));
static INVENTED_MARKER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<!--\s*(?:Slide|Page)\s+number:\s*\d+\s*-->\s*\n?").expect("valid marker pattern")
});

/// Python's placeholder protection (`protect_image_positions` without
/// screenshots for documents, then `protect_content`): each match is
/// replaced at its first remaining occurrence, numbered as Python numbers it.
fn protect(markdown: &str, images: bool) -> (String, Vec<(String, String)>) {
    fn replace(
        result: &mut String,
        mapping: &mut Vec<(String, String)>,
        found: Vec<(usize, String)>,
        kind: &str,
    ) {
        for (index, text) in found {
            let placeholder = format!("__MARKITAI_{kind}_{index}__");
            *result = result.replacen(&text, &placeholder, 1);
            mapping.push((placeholder, text));
        }
    }
    let matches = |pattern: &Regex, text: &str, first: usize| -> Vec<(usize, String)> {
        pattern
            .find_iter(text)
            .enumerate()
            .map(|(index, found)| (first + index, found.as_str().to_owned()))
            .collect()
    };
    let mut result = markdown.to_owned();
    let mut mapping = Vec::new();
    if images {
        // Python numbers every image, including the screenshots it leaves alone.
        let found = matches(&IMAGE_LINK, &result, 0)
            .into_iter()
            .filter(|(_, text)| !text.contains(".markitai/screenshots/"))
            .collect();
        replace(&mut result, &mut mapping, found, "IMG");
    }
    let found = matches(&PAGE_NUMBER, &result, 0);
    replace(&mut result, &mut mapping, found, "PAGENUM");
    let found = matches(&SLIDE_NUMBER, &result, 0);
    replace(&mut result, &mut mapping, found, "SLIDENUM");
    let mut page = 0;
    for pattern in PAGE_COMMENTS.iter() {
        let found = matches(pattern, &result, page);
        page += found.len();
        replace(&mut result, &mut mapping, found, "PAGE");
    }
    (result, mapping)
}

/// Placeholders an answer must keep: page and slide starts, and (documents)
/// images. Python's comments about page images may be dropped.
fn required(placeholder: &str) -> bool {
    ["_IMG_", "_PAGENUM_", "_SLIDENUM_"]
        .iter()
        .any(|kind| placeholder.contains(kind))
}

/// Put Python's protected content back. Every placeholder in the answer must
/// be one of this request's, at most once, and every required one present;
/// raw page or slide markers in the answer were invented, as Python treats them.
fn restore(answer: &str, mapping: &[(String, String)]) -> Result<String> {
    let text = INVENTED_MARKER.replace_all(answer, "").into_owned();
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for found in PLACEHOLDER.find_iter(&text) {
        let known = mapping
            .iter()
            .find(|(placeholder, _)| placeholder == found.as_str())
            .ok_or_else(|| {
                Error::Conversion("LLM answer contains a placeholder that was never sent".into())
            })?;
        *seen.entry(known.0.as_str()).or_default() += 1;
    }
    if seen.values().any(|count| *count > 1) {
        return Err(Error::Conversion(
            "LLM answer repeats a protected placeholder".into(),
        ));
    }
    let lost = mapping
        .iter()
        .filter(|(placeholder, _)| {
            required(placeholder) && !seen.contains_key(placeholder.as_str())
        })
        .count();
    if lost > 0 {
        return Err(Error::Conversion(format!(
            "LLM answer dropped {lost} protected image, page or slide placeholder(s)"
        )));
    }
    let mut result = text;
    for (placeholder, original) in mapping {
        if placeholder.contains("_PAGENUM_") || placeholder.contains("_SLIDENUM_") {
            let pattern = Regex::new(&format!(r"\s*{}\s*", regex::escape(placeholder)))
                .map_err(|_| Error::Conversion("Invalid placeholder".into()))?;
            result = pattern
                .replacen(&result, 1, regex::NoExpand(&format!("\n\n{original}\n\n")))
                .into_owned();
        } else {
            result = result.replace(placeholder.as_str(), original);
        }
    }
    Ok(result.trim_start_matches('\n').to_owned())
}

/// The text of every user message of a Python request body, joined.
fn user_text(body: &Value) -> String {
    let mut text = String::new();
    for message in body
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("user"))
    {
        match message.get("content") {
            Some(Value::String(part)) => text.push_str(part),
            Some(Value::Array(parts)) => {
                for part in parts {
                    if let Some(part) = part.get("text").and_then(Value::as_str) {
                        text.push_str(part);
                        text.push('\n');
                    }
                }
            }
            _ => {}
        }
    }
    text
}

/// The structured mode and tool (or schema) name a Python request used.
fn request_mode(body: &Value) -> Result<(structured::Mode, String)> {
    let named = |pointer: &str| {
        body.pointer(pointer)
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty() && name.len() <= 128)
            .map(str::to_owned)
    };
    if body.get("tools").is_some() {
        let name = named("/tool_choice/function/name")
            .or_else(|| named("/tool_choice/name"))
            .ok_or_else(|| {
                Error::InvalidInput("Python batch request forces no named tool".into())
            })?;
        return Ok((structured::Mode::Tools, name));
    }
    if let Some(name) = named("/response_format/json_schema/name") {
        return Ok((structured::Mode::JsonSchema, name));
    }
    Ok((structured::Mode::JsonText, String::new()))
}

/// One analyzed image of an imported document.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonImage {
    /// The batch request, or `None` for an answer Python already had.
    custom_id: Option<String>,
    /// Relative to the output directory.
    image: String,
    mode: structured::Mode,
    tool: String,
    /// Python's own answer (`alt`, `desc`, `text`), when it had one.
    answer: Option<Value>,
    /// The image belongs to this document; otherwise to a document Python
    /// enhanced live, whose output this job does not own.
    applies: bool,
}

/// A Python document request, frozen for collection by this build.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonPlan {
    version: u32,
    /// Marks the plan format; native plans have no such field.
    python_import: String,
    endpoint: Endpoint,
    kind: PythonKind,
    source: String,
    /// Python's LLM input, before its placeholder protection.
    markdown: String,
    /// The base Markdown's frontmatter as Python wrote it.
    metadata: serde_json::Map<String, Value>,
    mapping: Vec<(String, String)>,
    mode: structured::Mode,
    tool: String,
    images: Vec<PythonImage>,
    alt_enabled: bool,
    desc_enabled: bool,
    profile: Option<String>,
    wikilinks: bool,
    page_markers: Option<bool>,
}

/// What collecting one imported document produces besides its Markdown.
pub struct PythonExtras {
    /// `images.json` entries, keyed by the directory (relative to the output)
    /// that holds the image.
    pub descriptions: Vec<(String, Value)>,
}

impl PythonPlan {
    /// Freeze one Python document request. `request` is its row of Python's
    /// `requests.jsonl`, `input` its saved LLM input, `base` the base Markdown
    /// on disk and `images` its image items with their request rows;
    /// `detached` are image requests of documents Python enhanced live, kept
    /// with this document so their usage is recorded.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        state: &PythonState,
        item: &PythonItem,
        request: &Value,
        input: &str,
        base: &str,
        images: &[(&PythonItem, Option<&Value>)],
        detached: &[(&PythonItem, Option<&Value>)],
        endpoint: &Endpoint,
        cfg: &Value,
    ) -> Result<Self> {
        if item.kind == PythonKind::Image || input.len() > MAX_TEXT || base.len() > MAX_TEXT {
            return Err(Error::InvalidInput(
                "Python batch document is invalid or too large".into(),
            ));
        }
        let body = python_body(request, &item.custom_id)?;
        let (mode, tool) = request_mode(body)?;
        let (protected, mapping) = protect(input, item.kind == PythonKind::Doc);
        // The saved input must be what was sent: its protected form appears in
        // the request exactly as Python built it.
        if input.trim().is_empty() || !user_text(body).contains(protected.trim()) {
            return Err(Error::InvalidInput(format!(
                "Python batch request {} does not match its saved input",
                item.custom_id
            )));
        }
        let (metadata, _) = crate::output::split_frontmatter(base);
        let mut frozen = Vec::with_capacity(images.len() + detached.len());
        let all = images
            .iter()
            .map(|image| (image, true))
            .chain(detached.iter().map(|image| (image, false)));
        for ((image, row), applies) in all {
            if image.answer.is_none() && row.is_none() {
                return Err(Error::InvalidInput(format!(
                    "Python batch image {} has no request",
                    image.custom_id
                )));
            }
            let (mode, tool) = match row {
                Some(row) => request_mode(python_body(row, &image.custom_id)?)?,
                None => (structured::Mode::JsonText, String::new()),
            };
            frozen.push(PythonImage {
                custom_id: image.answer.is_none().then(|| image.custom_id.clone()),
                image: image.image.clone(),
                mode,
                tool,
                answer: image.answer.clone(),
                applies,
            });
        }
        Ok(Self {
            version: PLAN_VERSION,
            python_import: "markitai-1.2".into(),
            endpoint: endpoint.clone(),
            kind: item.kind,
            source: item.source.clone(),
            markdown: input.to_owned(),
            metadata,
            mapping,
            mode,
            tool,
            images: frozen,
            alt_enabled: state.alt_enabled,
            desc_enabled: state.desc_enabled,
            profile: cfg
                .pointer("/output/profile")
                .and_then(Value::as_str)
                .map(str::to_owned),
            wikilinks: config::enabled(cfg, "/output/wikilinks"),
            page_markers: cfg.pointer("/output/page_markers").and_then(Value::as_bool),
        })
    }

    /// The custom IDs of this document's image requests.
    pub fn image_requests(&self) -> Vec<String> {
        self.images
            .iter()
            .filter_map(|image| image.custom_id.clone())
            .collect()
    }

    fn validate(&self) -> Result<()> {
        if self.version != PLAN_VERSION
            || self.python_import != "markitai-1.2"
            || Api::for_provider(&self.endpoint.provider).is_none()
            || self.markdown.trim().is_empty()
            || self.markdown.len() > MAX_TEXT
            || self.kind == PythonKind::Image
        {
            return Err(Error::InvalidInput(
                "Imported Python batch plan is invalid".into(),
            ));
        }
        Ok(())
    }

    fn protocol(&self) -> Protocol {
        match self.endpoint.api() {
            Api::OpenAi => Protocol::Chat,
            Api::Anthropic => Protocol::Anthropic,
        }
    }

    fn deployment(&self) -> Deployment {
        let protocol = self.protocol();
        Deployment {
            id: format!("{}/{}", self.endpoint.provider, self.endpoint.model),
            explicit_id: None,
            group: "default".into(),
            model: self.endpoint.model.clone(),
            provider: self.endpoint.provider.clone(),
            weight: 1,
            key: None,
            endpoint: format!(
                "{}{}",
                self.endpoint.api_base,
                if protocol == Protocol::Anthropic {
                    "/messages"
                } else {
                    "/chat/completions"
                }
            ),
            protocol,
            max_tokens: None,
            reasoning_effort: None,
            supports_vision: None,
            cloud: None,
        }
    }

    /// Observed usage at batch prices, before any validation.
    pub fn observe(&self, item: &ResultItem) -> Result<ConversionUsage> {
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

    fn answer_text(&self, item: &ResultItem, mode: structured::Mode, tool: &str) -> Result<String> {
        if item.error.is_some()
            || !item
                .http_status
                .is_some_and(|code| (200..300).contains(&code))
        {
            return Err(Error::Conversion(
                "Provider Batch request did not succeed; observed usage is retained".into(),
            ));
        }
        let body = item
            .body
            .as_ref()
            .ok_or_else(|| Error::Conversion("Provider Batch response body is missing".into()))?;
        structured::decode_as(mode, tool, self.protocol(), body).map_err(|failure| failure.error)
    }

    /// Decode the document answer with its image answers (`None`: the image
    /// request has no result). `usage` is the recorded quote of the document
    /// and its images together.
    pub fn decode_recorded(
        &self,
        item: &ResultItem,
        images: &[Option<ResultItem>],
        usage: ConversionUsage,
    ) -> DetailedResult<(DecodedDocument, PythonExtras)> {
        let result = (|| {
            self.validate()?;
            let text = self.answer_text(item, self.mode, &self.tool)?;
            let answer = document::parse_value(&structured::parse(&text)?, false)?;
            let mut warnings = Vec::new();
            let (protected, _) = protect(&self.markdown, self.kind == PythonKind::Doc);
            let mut cleaned = answer.markdown;
            if let Some(salvage) = degeneration::salvage(&cleaned, &protected) {
                warnings.push(salvage.warning());
                cleaned = salvage.text;
            }
            let restored = restore(&cleaned, &self.mapping)?;
            document::plausible(&self.markdown, &restored, false)?;
            let mut markdown = crate::markdown::normalize(&restored);
            if self.page_markers == Some(false) {
                markdown = crate::output_profiles::remove_page_markers(&markdown);
            }
            let mut descriptions = Vec::new();
            let mut alts = HashMap::new();
            let mut slots = images.iter();
            for image in &self.images {
                let name = image
                    .image
                    .rsplit('/')
                    .next()
                    .unwrap_or_default()
                    .to_owned();
                let analysis = match (&image.custom_id, &image.answer) {
                    (None, Some(answer)) => Some((
                        answer
                            .get("alt")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned(),
                        answer.get("desc").cloned().unwrap_or(Value::Null),
                        answer.get("text").cloned().unwrap_or(Value::Null),
                    )),
                    _ => slots
                        .next()
                        .and_then(Option::as_ref)
                        .ok_or_else(|| Error::Conversion("no result".into()))
                        .and_then(|result| self.answer_text(result, image.mode, &image.tool))
                        .and_then(|text| structured::parse(&text))
                        .and_then(|value| {
                            let caption = value
                                .get("caption")
                                .and_then(Value::as_str)
                                .map(str::trim)
                                .filter(|caption| !caption.is_empty())
                                .ok_or_else(|| Error::Conversion("no caption".into()))?;
                            Ok((
                                caption.to_owned(),
                                value.get("description").cloned().unwrap_or(Value::Null),
                                value.get("extracted_text").cloned().unwrap_or(Value::Null),
                            ))
                        })
                        .ok(),
                };
                let Some((caption, description, text)) = analysis else {
                    warnings.push(format!(
                        "Image analysis failed for {name} in the Python batch; its original alt text was kept"
                    ));
                    continue;
                };
                if self.alt_enabled && !caption.is_empty() {
                    if image.applies {
                        alts.insert(name.clone(), caption.clone());
                    } else {
                        warnings.push(format!(
                            "The Python batch's alt text for {name} belongs to a document Python enhanced live; it was not applied"
                        ));
                    }
                }
                if self.desc_enabled {
                    let directory = image
                        .image
                        .rsplit_once('/')
                        .map(|(directory, _)| directory.to_owned())
                        .unwrap_or_default();
                    descriptions.push((
                        directory,
                        json!({"asset":image.image,"alt":caption,"desc":description,
                            "text":if text.is_null() { json!("") } else { text },
                            "created":chrono::Local::now().to_rfc3339()}),
                    ));
                }
            }
            if !alts.is_empty() {
                markdown = replace_alt_text(&markdown, &alts);
            }
            let mut metadata = self.metadata.clone();
            metadata.insert("description".into(), json!(answer.metadata.description));
            metadata.insert("tags".into(), json!(answer.metadata.tags));
            crate::output_profiles::apply(
                &mut markdown,
                &mut metadata,
                &json!({"output":{"profile":self.profile,"wikilinks":self.wikilinks}}),
            );
            if !usage.cost_complete()
                && let Some(warning) = usage.unpriced_warning()
            {
                warnings.push(warning);
            }
            Ok((
                DecodedDocument {
                    markdown,
                    metadata,
                    usage: copy_usage(&usage),
                    cache_hit: false,
                    warnings,
                },
                PythonExtras { descriptions },
            ))
        })();
        result.map_err(|error| ConversionFailure { error, usage })
    }
}

fn python_body<'a>(row: &'a Value, custom_id: &str) -> Result<&'a Value> {
    if row.get("custom_id").and_then(Value::as_str) != Some(custom_id) {
        return Err(Error::InvalidInput(
            "Python batch request row does not match its item".into(),
        ));
    }
    row.get("body")
        .or_else(|| row.get("params"))
        .filter(|body| body.is_object())
        .ok_or_else(|| Error::InvalidInput("Python batch request row has no body".into()))
}

static IMAGE_REFERENCE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"!\[(?:[^\]\\]|\\.)*\]\((?P<target>[^)]+)\)").expect("valid image reference")
});

/// Python's `apply_alt_text_updates`: an image whose file name has an
/// analysis gets that caption as its alt text.
fn replace_alt_text(markdown: &str, alts: &HashMap<String, String>) -> String {
    IMAGE_REFERENCE
        .replace_all(markdown, |found: &regex::Captures<'_>| {
            let target = &found["target"];
            let name = target.replace('\\', "/");
            let name = name.rsplit('/').next().unwrap_or_default();
            match alts.get(name) {
                Some(alt) => {
                    let alt: String = alt
                        .chars()
                        .map(|ch| if ch == '\n' || ch == '\r' { ' ' } else { ch })
                        .collect::<String>()
                        .replace('\\', "\\\\")
                        .replace('[', "\\[")
                        .replace(']', "\\]");
                    format!("![{alt}]({target})")
                }
                None => found[0].to_owned(),
            }
        })
        .into_owned()
}

/// A frozen Provider Batch plan: this build's own, or one imported from a
/// Python state.
pub enum FrozenPlan {
    Native(Box<super::batch::Plan>),
    Python(Box<PythonPlan>),
}
impl FrozenPlan {
    pub fn from_value(value: Value) -> Result<Self> {
        let invalid = |_| Error::InvalidInput("Frozen Provider Batch plan is invalid".into());
        if value.get("python_import").is_some() {
            Ok(Self::Python(Box::new(
                serde_json::from_value(value).map_err(invalid)?,
            )))
        } else {
            Ok(Self::Native(Box::new(
                serde_json::from_value(value).map_err(invalid)?,
            )))
        }
    }
    pub fn observe(&self, item: &ResultItem) -> Result<ConversionUsage> {
        match self {
            Self::Native(plan) => plan.observe(item),
            Self::Python(plan) => plan.observe(item),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protection_numbers_like_python_and_skips_screenshots() {
        let input = "<!-- Slide number: 1 -->\n\n# Deck\n\n![a](.markitai/assets/x.jpg)\n\n![s](.markitai/screenshots/p1.png)\n\n![a](.markitai/assets/x.jpg)\n\n<!-- Slide number: 2 -->\n\n<!-- Page number: 3 -->\n<!-- Page images for reference -->\n<!-- ![Page 1](.markitai/screenshots/p1.png) -->\n";
        let (text, mapping) = protect(input, true);
        assert_eq!(
            text,
            "__MARKITAI_SLIDENUM_0__\n\n# Deck\n\n__MARKITAI_IMG_0__\n\n![s](.markitai/screenshots/p1.png)\n\n__MARKITAI_IMG_2__\n\n__MARKITAI_SLIDENUM_1__\n\n__MARKITAI_PAGENUM_0__\n__MARKITAI_PAGE_0__\n__MARKITAI_PAGE_1__\n"
        );
        let names: Vec<_> = mapping.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            [
                "__MARKITAI_IMG_0__",
                "__MARKITAI_IMG_2__",
                "__MARKITAI_PAGENUM_0__",
                "__MARKITAI_SLIDENUM_0__",
                "__MARKITAI_SLIDENUM_1__",
                "__MARKITAI_PAGE_0__",
                "__MARKITAI_PAGE_1__"
            ]
        );
        let (vision, mapping) = protect(input, false);
        assert!(vision.contains("![a](.markitai/assets/x.jpg)"));
        assert!(mapping.iter().all(|(name, _)| !name.contains("_IMG_")));
    }

    #[test]
    fn protection_matches_python_on_its_recorded_outputs() {
        let fixture: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/python-batch/protection.json"
        )))
        .unwrap();
        let pairs = |value: &Value| -> Vec<(String, String)> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|pair| {
                    (
                        pair[0].as_str().unwrap().to_owned(),
                        pair[1].as_str().unwrap().to_owned(),
                    )
                })
                .collect()
        };
        for case in fixture["cases"].as_array().unwrap() {
            let input = case["input"].as_str().unwrap();
            let (text, mapping) = protect(input, true);
            assert_eq!(text, case["document"].as_str().unwrap(), "{input}");
            assert_eq!(mapping, pairs(&case["document_mapping"]), "{input}");
            let (text, mapping) = protect(input, false);
            assert_eq!(text, case["vision"].as_str().unwrap(), "{input}");
            assert_eq!(mapping, pairs(&case["vision_mapping"]), "{input}");
        }
    }

    #[test]
    fn restoration_requires_every_structural_placeholder_once() {
        let mapping = vec![
            ("__MARKITAI_IMG_0__".to_owned(), "![a](x.jpg)".to_owned()),
            (
                "__MARKITAI_SLIDENUM_0__".to_owned(),
                "<!-- Slide number: 1 -->".to_owned(),
            ),
            (
                "__MARKITAI_PAGE_0__".to_owned(),
                "<!-- Page images for reference -->".to_owned(),
            ),
        ];
        let restored = restore(
            "__MARKITAI_SLIDENUM_0__\n# Deck\n<!-- Slide number: 9 -->\n__MARKITAI_IMG_0__",
            &mapping,
        )
        .unwrap();
        assert_eq!(restored, "<!-- Slide number: 1 -->\n\n# Deck\n![a](x.jpg)");
        assert!(restore("# Deck __MARKITAI_IMG_0__", &mapping).is_err());
        assert!(
            restore(
                "__MARKITAI_SLIDENUM_0__ __MARKITAI_IMG_0__ __MARKITAI_IMG_0__",
                &mapping
            )
            .is_err()
        );
        assert!(
            restore(
                "__MARKITAI_SLIDENUM_0__ __MARKITAI_IMG_0__ __MARKITAI_IMG_7__",
                &mapping
            )
            .is_err()
        );
    }

    #[test]
    fn alt_text_follows_the_image_file_name() {
        let alts = HashMap::from([("x.jpg".to_owned(), "A [chart]".to_owned())]);
        assert_eq!(
            replace_alt_text("![old](.markitai/assets/x.jpg) ![keep](y.png)", &alts),
            "![A \\[chart\\]](.markitai/assets/x.jpg) ![keep](y.png)"
        );
    }
}
