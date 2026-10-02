mod asset_store;
mod browser;
mod browser_install;
mod browser_runtime;
pub mod config;
mod fetch;
pub mod fetch_cache;
pub mod formats;
mod image_enrichment;
mod images;
mod llm;
pub mod llm_cache;
mod llm_runtime;
mod markdown;
mod ocr;
mod office_media;
mod office_render;
pub mod output;
mod output_profiles;
mod pdf_media;
mod pdf_raster;
#[doc(hidden)]
pub mod platform;
mod preparation;
mod pricing;
mod private_install;
mod process_groups;
mod proxy;
#[doc(hidden)]
pub mod sort;
#[doc(hidden)]
pub use preparation::{PreparedConversion, prepare_with_publication};
#[doc(hidden)]
pub mod provider_batch;
pub mod provider_management;
pub mod spa_domains;
pub mod subscription;
#[cfg(target_os = "macos")]
mod system_frameworks;
mod types;

pub use browser_runtime::BrowserRuntime;
/// Remote-fallback consent a host installs once per process (the CLI does;
/// without it `auto` never sends a URL to a remote service).
pub use fetch::consent::{ConsentRequest, RemoteFallback, RemoteNotice, set_remote_fallback};
pub use images::is_image_extension;
pub use llm_runtime::LlmRuntime;
/// Lowercase hexadecimal spelling of digest bytes, as `{:x}` rendered them
/// before `sha2` 0.11 (whose output arrays do not implement `LowerHex`).
#[doc(hidden)]
pub fn hex(bytes: impl AsRef<[u8]>) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push(char::from(DIGITS[usize::from(byte >> 4)]));
        text.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    text
}
/// Kill every external runtime process group this library started (Chromium,
/// LibreOffice, official subscription runtimes). Async-signal-safe, for a host
/// that is about to terminate; caller-held browser sessions are untouched.
pub use process_groups::terminate_all as terminate_child_process_groups;
pub use types::*;
/// Available optional local capabilities; probing never launches a backend.
pub fn browser_available() -> bool {
    browser::available()
}

pub fn local_ocr_available() -> bool {
    ocr::available()
}

pub use ocr::{LocalOcrModel, LocalOcrModelState};

/// The local OCR engine this process reads with: `vision` (macOS), `paddle`
/// (the portable PaddleOCR engine) or `unavailable`.
pub fn local_ocr_backend() -> &'static str {
    ocr::backend()
}

/// The model files the portable OCR engine needs for the configuration's
/// `ocr.lang`, each with whether it is installed; `None` when this process
/// reads with another engine. Probing never downloads or creates files.
pub fn local_ocr_models(config: &Value) -> Result<Option<Vec<LocalOcrModel>>> {
    ocr::portable_models(config)
}

/// Explicitly download the missing files [`local_ocr_models`] lists from the
/// official PaddleOCR mirror, each verified by size and SHA-256, into the
/// private Markitai home; returns the files installed.
pub fn install_local_ocr_models(config: &Value) -> Result<Vec<PathBuf>> {
    ocr::install_portable_models(config)
}

/// Whether this process is an x86_64 build translated by Rosetta on Apple
/// silicon, where Vision text recognition fails even though the API is present.
pub fn rosetta_translated() -> bool {
    #[cfg(target_os = "macos")]
    return system_frameworks::translated();
    #[cfg(not(target_os = "macos"))]
    false
}

pub fn pdf_raster_available() -> bool {
    pdf_raster::available()
}

pub fn office_render_available() -> bool {
    office_render::available() && pdf_raster::available()
}

/// Launch and close an isolated browser session; no document or provider request is made.
pub fn browser_diagnostic() -> Result<Option<PathBuf>> {
    browser::diagnostic()
}

/// Explicitly install and validate the official headless browser in Markitai's private home.
pub fn install_browser() -> Result<PathBuf> {
    browser_install::install()
}

/// Check that the optional Office executable starts within a bounded deadline.
pub fn office_diagnostic() -> Result<Option<PathBuf>> {
    office_render::diagnostic()
}

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Report local model configuration without contacting a provider or exposing keys.
pub fn llm_capabilities(config: &Value) -> LlmCapabilities {
    llm::capabilities(config, &config::environment())
}

/// Return the native router's eligible vision model identities without a network probe.
pub fn llm_vision_models(config: &Value) -> Vec<String> {
    llm::vision_models(config, &config::environment())
}

pub fn convert(source: &str, options: ConvertOptions) -> Result<ConversionOutput> {
    convert_with_context(source, options, ConvertContext::default())
}

/// Convert while retaining recorded model usage if a later stage fails.
pub fn convert_detailed(source: &str, options: ConvertOptions) -> DetailedResult<ConversionOutput> {
    convert_with_context_detailed(source, options, ConvertContext::default())
}

/// Shares native run resources without adding fields to binding requests.
#[doc(hidden)]
pub fn convert_with_context(
    source: &str,
    options: ConvertOptions,
    context: ConvertContext<'_>,
) -> Result<ConversionOutput> {
    convert_with_publication(source, options, context, None)
}

/// Detailed conversion with the same caller-owned runtime and routing context.
#[doc(hidden)]
pub fn convert_with_context_detailed(
    source: &str,
    options: ConvertOptions,
    context: ConvertContext<'_>,
) -> DetailedResult<ConversionOutput> {
    convert_with_publication_detailed(source, options, context, None)
}

/// Native callers may supply an already acquired document publication claim.
/// Existing conversion and serialized adapter entrypoints remain independent.
#[doc(hidden)]
pub fn convert_with_publication(
    source: &str,
    options: ConvertOptions,
    context: ConvertContext<'_>,
    publication: Option<&dyn output::Publication>,
) -> Result<ConversionOutput> {
    convert_with_publication_detailed(source, options, context, publication)
        .map_err(|failure| failure.error)
}

/// Retain recorded usage for failures, including final output publication errors.
#[doc(hidden)]
pub fn convert_with_publication_detailed(
    source: &str,
    options: ConvertOptions,
    context: ConvertContext<'_>,
    publication: Option<&dyn output::Publication>,
) -> DetailedResult<ConversionOutput> {
    let mut scope = None;
    let result = convert_inner(source, options, context, publication, &mut scope, None);
    result.map_err(|error| ConversionFailure {
        error,
        usage: scope
            .as_ref()
            .map(llm::DocumentScope::usage)
            .unwrap_or_default(),
    })
}

fn convert_inner(
    source: &str,
    options: ConvertOptions,
    context: ConvertContext<'_>,
    publication: Option<&dyn output::Publication>,
    document_scope: &mut Option<llm::DocumentScope>,
    mut prepared: Option<&mut output::PreparedOutput>,
) -> Result<ConversionOutput> {
    let start = Instant::now();
    if source.trim().is_empty() {
        return Err(Error::InvalidInput("Input cannot be empty".into()));
    }
    let mut cfg = if let Some(value) = options.config {
        if !value.is_object() {
            return Err(Error::Config("config must be an object".into()));
        }
        config::normalize(&value)?
    } else {
        config::load(None, None)?
    };
    for (path, value) in [
        ("/llm/enabled", options.llm),
        ("/ocr/enabled", options.ocr),
        ("/screenshot/enabled", options.screenshot),
        ("/image/alt_enabled", options.alt),
        ("/image/desc_enabled", options.desc),
    ] {
        if let Some(value) = value {
            *cfg.pointer_mut(path)
                .ok_or_else(|| Error::Config(format!("Invalid config at {path}")))? =
                Value::Bool(value);
        }
    }
    if let Some(profile) = options.profile {
        if !matches!(profile.as_str(), "rag" | "obsidian" | "okf") {
            return Err(Error::Config(format!("Invalid output profile: {profile}")));
        }
        cfg["output"]["profile"] = json!(profile);
    }
    let is_url = is_url(source);
    if source.contains("://") && !is_url {
        return Err(Error::InvalidInput(
            "Only HTTP and HTTPS URLs are supported".into(),
        ));
    }
    let input_path = config::expand_home(Path::new(source));
    let image_input = !is_url
        && input_path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(is_image_extension);
    let output_dir = options.output_dir.map(|path| config::expand_home(&path));
    // Only a document that would otherwise publish nothing uses the store.
    let stdout_assets = context.stdout_assets.filter(|_| output_dir.is_none());
    let name_extension = input_path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    // A document saved under another type's extension (a PDF named .docx, a
    // Word file named .pdf) is read as what its content says it is.
    let real_extension = if is_url || image_input {
        None
    } else {
        formats::real_extension(&input_path, &name_extension)
    };
    let document_extension = real_extension.unwrap_or(&name_extension);
    let mut pdf_input = !is_url && document_extension == "pdf";
    let mut pdf_media_requested = pdf_input
        && (config::enabled(&cfg, "/ocr/enabled") || config::enabled(&cfg, "/screenshot/enabled"));
    let office_kind = if is_url {
        None
    } else {
        office_render::kind(document_extension)
    };
    let office_media_wanted = office_kind.is_some()
        && (config::enabled(&cfg, "/ocr/enabled") || config::enabled(&cfg, "/screenshot/enabled"));
    // Page screenshots and page OCR render the document through LibreOffice.
    // Without it the document's own text is converted and the missing page
    // capture is one warning, so a preset such as `rich`, or OCR over a folder,
    // still converts its Office files. Only `screenshot_only`, where the
    // screenshots are the whole requested output, keeps failing.
    // Numbers capture is unsupported with or without LibreOffice: its own error
    // says so, and no installation would change it.
    let office_numbers = input_path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("numbers"));
    let office_ocr_without_renderer =
        office_media_wanted && !office_numbers && !office_render_available();
    let office_media_requested = office_media_wanted && !office_ocr_without_renderer;
    if !is_url {
        let path = &input_path;
        output::check_path(path, config::enabled(&cfg, "/output/allow_symlinks"))?;
        let meta = std::fs::metadata(path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Error::NotFound(source.into())
            } else {
                Error::Io(error)
            }
        })?;
        let numbers_package = meta.is_dir() && formats::is_numbers_package_path(path);
        if meta.is_dir() && !numbers_package {
            return Err(Error::IsDirectory(source.into()));
        }
        if !meta.is_file() && !numbers_package {
            return Err(Error::InvalidInput(format!(
                "Input is not a file: {source}"
            )));
        }
        if meta.is_file() && meta.len() > 500 * 1024 * 1024 {
            return Err(Error::InvalidInput(
                "Input exceeds the 500 MiB limit".into(),
            ));
        }
        if meta.is_file() {
            formats::check_not_empty(path, &name_extension)?;
        }
    }
    if let Some(real) = real_extension
        && office_media_requested
    {
        return Err(Error::InvalidInput(format!(
            "The content is not a .{name_extension} file; rename it to .{real} to capture its pages"
        )));
    }
    if office_ocr_without_renderer
        && config::enabled(&cfg, "/screenshot/enabled")
        && config::enabled(&cfg, "/screenshot/screenshot_only")
    {
        return Err(Error::Unsupported(
            "Office screenshots require an installed LibreOffice (soffice on PATH) and the native PDF page renderer, and --screenshot-only publishes nothing else. Install LibreOffice (macOS: brew install --cask libreoffice), or drop --screenshot-only to convert the document's text".into(),
        ));
    }
    // `-b cloudflare` (fetch.cloudflare.convert_enabled): Workers AI converts
    // the formats it reads into text. OCR and screenshots need the native
    // reader's pages, so such a file keeps the native path, with a warning.
    let cloudflare_extension = if image_input {
        name_extension.as_str()
    } else {
        document_extension
    };
    let cloudflare_wanted =
        !is_url && fetch::cloudflare::converts(&cfg, &input_path, cloudflare_extension);
    let cloudflare_backend = cloudflare_wanted
        && !config::enabled(&cfg, "/ocr/enabled")
        && !config::enabled(&cfg, "/screenshot/enabled");
    let image_input = image_input && !cloudflare_backend;
    if image_input
        && !config::enabled(&cfg, "/llm/enabled")
        && !config::enabled(&cfg, "/ocr/enabled")
    {
        // A file that is not an image at all is an error, not a skip.
        formats::check_image(&input_path, &name_extension)?;
        return Err(Error::ImageOnly(format!(
            "{} is an image file with no text to extract. Enable LLM (llm=True) or OCR (ocr=True) for content extraction.",
            input_path.file_name().unwrap_or_default().to_string_lossy()
        )));
    }
    let name = if is_url {
        output::url_name(source, &Default::default())
    } else {
        input_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    };
    if let Some(dir) = &output_dir
        && match publication {
            Some(publication) => publication.skip_existing(),
            None => output::should_skip(dir, &name, &cfg)?,
        }
    {
        return Ok(ConversionOutput {
            source: source.into(),
            skip_reason: Some("exists".into()),
            duration: start.elapsed().as_secs_f64(),
            ..Default::default()
        });
    }
    *document_scope = config::enabled(&cfg, "/llm/enabled").then(|| llm::DocumentScope::new(&cfg));
    let mut vision = Vec::new();
    let mut screenshots = Vec::new();
    let mut fetch_cache_hit = false;
    let mut pdf_has_reliable_text = true;
    let vlm_disabled = config::environment()
        .get("MARKITAI_NO_VLM_OCR")
        .is_some_and(|value| vlm_ocr_disabled(value));
    let local_ocr = image_input
        && config::enabled(&cfg, "/ocr/enabled")
        && (!config::enabled(&cfg, "/llm/enabled") || vlm_disabled);
    let mut doc = if cloudflare_backend {
        fetch::cloudflare::convert_file(&input_path, cloudflare_extension, &cfg)?
    } else if image_input {
        let (document, images) = images::extract(&input_path, &cfg, local_ocr)?;
        vision = images;
        document
    } else if is_url {
        let fetched = fetch::fetch_with_runtime(
            source,
            &cfg,
            context.explicit_fetch_strategy,
            output_dir.is_some(),
            context.browser_runtime,
        )?;
        fetch_cache_hit = fetched.cache_hit;
        screenshots = fetched.screenshots;
        match fetched.content {
            fetch::FetchContent::Document(document) => document,
            fetch::FetchContent::Pdf(downloaded) => {
                pdf_input = true;
                pdf_media_requested = config::enabled(&cfg, "/ocr/enabled")
                    || config::enabled(&cfg, "/screenshot/enabled");
                let mut document = if pdf_media_requested {
                    let mut media_cfg = cfg.clone();
                    if config::enabled(&cfg, "/llm/pure") {
                        // URL pure mode also wins over screenshot_only during
                        // PDF vision-budget checks and opt-out diagnostics.
                        media_cfg["screenshot"]["screenshot_only"] = false.into();
                    }
                    let (document, captured, reliable) = prepare_pdf_media(
                        &downloaded.bytes,
                        &name,
                        output_dir.as_deref(),
                        &media_cfg,
                        vlm_disabled,
                    )?;
                    screenshots = captured;
                    pdf_has_reliable_text = reliable;
                    document
                } else {
                    // Config-only screenshot_only does not turn on PDF capture.
                    formats::extract_pdf_with_config(&downloaded.bytes, &cfg)?
                };
                document.warnings.extend(downloaded.warnings);
                document.metadata.insert("format".into(), "PDF".into());
                document
                    .metadata
                    .insert("fetch_strategy".into(), downloaded.strategy.into());
                if downloaded.final_url != source {
                    document.metadata.insert(
                        "source_url".into(),
                        output::redact_url(&downloaded.final_url).into(),
                    );
                }
                document
            }
        }
    } else if pdf_media_requested {
        let (mut document, captured, reliable) = prepare_pdf_media(
            &std::fs::read(&input_path)?,
            &name,
            output_dir.as_deref(),
            &cfg,
            vlm_disabled,
        )
        .map_err(formats::explain_damage)?;
        pdf_has_reliable_text = reliable;
        screenshots = captured;
        document.metadata.insert(
            "source".into(),
            input_path.to_string_lossy().as_ref().into(),
        );
        document.metadata.insert("format".into(), "PDF".into());
        document
    } else if document_extension == "pdf" {
        let mut document = formats::extract_pdf_with_config(&std::fs::read(&input_path)?, &cfg)
            .map_err(formats::explain_damage)?;
        document.metadata.insert(
            "source".into(),
            input_path.to_string_lossy().as_ref().into(),
        );
        document.metadata.insert("format".into(), "PDF".into());
        document
    } else {
        formats::extract_as(&input_path, document_extension).map_err(formats::explain_damage)?
    };
    if let Some(real) = real_extension {
        doc.warnings
            .push(formats::retyped_warning(&name_extension, real));
    }
    if cloudflare_wanted && !cloudflare_backend {
        doc.warnings.push("The Cloudflare backend was not used for this file: OCR and screenshots need the native reader's pages, so it was converted natively.".into());
    }
    if office_ocr_without_renderer {
        let install = "Install LibreOffice (macOS: brew install --cask libreoffice)";
        let warning = match (
            config::enabled(&cfg, "/screenshot/enabled"),
            config::enabled(&cfg, "/ocr/enabled"),
        ) {
            (true, true) => format!(
                "Page screenshots and OCR of Office pages need LibreOffice (soffice on PATH) and the native PDF page renderer; the document's own text was converted without them. {install}, or pass --no-screenshot and --no-ocr to skip them."
            ),
            (true, false) => format!(
                "Page screenshots of Office documents need LibreOffice (soffice on PATH) and the native PDF page renderer; the document's own text was converted without them. {install}, or pass --no-screenshot to skip them."
            ),
            _ => format!(
                "OCR of Office pages needs LibreOffice (soffice on PATH) and the native PDF page renderer; the document's own text was converted without page OCR. {install}, or pass --no-ocr to skip it."
            ),
        };
        doc.warnings.push(warning);
    }
    if office_media_requested {
        let (captured, reliable) = office_media::prepare(
            &mut doc,
            &input_path,
            office_kind.expect("Office media requires a known format"),
            screenshot_prefix(&name, &cfg),
            output_dir.as_deref(),
            &cfg,
            vlm_disabled,
        )?;
        screenshots = captured;
        pdf_has_reliable_text = reliable;
    }
    let format = doc
        .metadata
        .get("format")
        .and_then(Value::as_str)
        .unwrap_or("");
    if format == "PDF" {
        pdf_input = true;
        if config::enabled(&cfg, "/ocr/enabled") && !pdf_media_requested {
            return Err(Error::Unsupported(
                "PDF OCR is not implemented in this development build".into(),
            ));
        }
    }
    if format == "PDF" && !pdf_media_requested && config::enabled(&cfg, "/screenshot/enabled") {
        return Err(Error::Unsupported(
            "Document screenshots are not implemented in this development build".into(),
        ));
    }
    let screenshot_only = is_url
        && !pdf_input
        && config::enabled(&cfg, "/screenshot/screenshot_only")
        && !(config::enabled(&cfg, "/llm/enabled") && config::enabled(&cfg, "/llm/pure"));
    if !screenshot_only {
        // Inline data images become owned assets wherever assets are kept.
        image_enrichment::prepare(
            &mut doc,
            source,
            &cfg,
            output_dir.is_some() || stdout_assets.is_some(),
        )?;
    }
    if !image_input {
        images::prepare_assets(&mut doc, &cfg);
    }
    let fetch_strategy = if is_url {
        doc.metadata
            .get("fetch_strategy")
            .and_then(Value::as_str)
            .map(str::to_owned)
    } else {
        None
    };
    if screenshot_only && !config::enabled(&cfg, "/llm/enabled") && output_dir.is_none() {
        return Err(Error::InvalidInput(
            "Screenshot-only conversion without LLM requires output_dir to retain captured images"
                .into(),
        ));
    }
    if screenshot_only && screenshots.is_empty() {
        return Err(Error::Fetch(
            "Screenshot-only conversion captured no screenshots".into(),
        ));
    }
    if screenshot_only {
        doc.markdown.clear();
        doc.assets.clear();
    }
    let mut result = output::prepare(source, &name, &mut doc, &cfg);
    result.fetch_cache_hit = fetch_cache_hit;
    result.fetch_strategy = fetch_strategy;
    let standalone_analysis = image_input && image_enrichment::enabled(source, &cfg);
    if config::enabled(&cfg, "/llm/enabled") {
        result.base_frontmatter = Some(result.frontmatter.clone());
    }
    if config::enabled(&cfg, "/llm/enabled") && !standalone_analysis {
        let source_context = if is_url {
            output::redact_url(source)
        } else {
            name.clone()
        };
        let pure = config::enabled(&cfg, "/llm/pure");
        let pdf_screenshot_only = (pdf_input || office_media_requested)
            && !(is_url && pure)
            && !screenshots.is_empty()
            && config::enabled(&cfg, "/screenshot/screenshot_only");
        let send_pdf_images =
            (pdf_input || office_media_requested) && (!pure || pdf_screenshot_only);
        let input = if pdf_screenshot_only {
            ""
        } else if pure {
            &doc.markdown
        } else {
            &result.markdown
        };
        let without_cache = |(markdown, usage)| llm::Enhancement {
            markdown,
            usage,
            cache_hit: false,
            warnings: Vec::new(),
            metadata: None,
        };
        let enhanced = if (is_url && !pure || send_pdf_images) && !screenshots.is_empty() {
            let image_refs: Vec<_> = screenshots
                .iter()
                .map(|shot| screenshot_mime(&shot.bytes).map(|mime| (mime, shot.bytes.as_slice())))
                .collect::<Result<_>>()?;
            if pure {
                llm::enhance_images_with_source_and_runtime(
                    input,
                    &source_context,
                    &image_refs,
                    &cfg,
                    context.llm_runtime,
                )
                .map(without_cache)
            } else {
                let frames = image_refs
                    .iter()
                    .enumerate()
                    .map(|(index, (mime, bytes))| llm::VisionFrame {
                        number: index + 1,
                        mime,
                        bytes,
                    })
                    .collect::<Vec<_>>();
                let kind = if pdf_input || office_media_requested {
                    llm::VisionKind::PagedDocument
                } else {
                    llm::VisionKind::WebCapture
                };
                let enhanced = llm::process_vision_with_runtime(
                    llm::VisionRequest {
                        markdown: input,
                        source_label: &source_context,
                        cache_context: source,
                        kind,
                        frames: &frames,
                    },
                    &cfg,
                    context.llm_runtime,
                );
                // A rendered web page still has an independently extracted body.
                // Reuse the same document accounting context for a typed text fallback.
                match enhanced {
                    Err(failure)
                        if is_url
                            && !pdf_input
                            && !screenshot_only
                            && !doc.markdown.trim().is_empty()
                            && failure.allow_text_fallback =>
                    {
                        result.warnings.push("Rendered-page enhancement failed; attempting structured text processing with the remaining document budget.".into());
                        llm::process_document_with_runtime(
                            input,
                            &source_context,
                            source,
                            doc.metadata.get("content_profile").and_then(Value::as_str)
                                == Some("social_post"),
                            &cfg,
                            context.llm_runtime,
                        )
                    }
                    outcome => outcome.map_err(|failure| failure.error),
                }
            }
        } else if !vision.is_empty() {
            let images = vision
                .iter()
                .map(|image| (image.mime, image.bytes.as_slice()))
                .collect::<Vec<_>>();
            if pure {
                llm::enhance_images_with_source_and_runtime(
                    input,
                    &source_context,
                    &images,
                    &cfg,
                    context.llm_runtime,
                )
                .map(without_cache)
            } else {
                let frames = images
                    .iter()
                    .enumerate()
                    .map(|(index, (mime, bytes))| llm::VisionFrame {
                        number: index + 1,
                        mime,
                        bytes,
                    })
                    .collect::<Vec<_>>();
                llm::process_vision_with_runtime(
                    llm::VisionRequest {
                        markdown: input,
                        source_label: &source_context,
                        cache_context: source,
                        kind: llm::VisionKind::PagedDocument,
                        frames: &frames,
                    },
                    &cfg,
                    context.llm_runtime,
                )
                .map_err(|failure| failure.error)
            }
        } else if !pure {
            llm::process_document_with_runtime(
                input,
                &source_context,
                source,
                doc.metadata.get("content_profile").and_then(Value::as_str) == Some("social_post"),
                &cfg,
                context.llm_runtime,
            )
        } else {
            llm::enhance_with_source_and_runtime(input, &source_context, &cfg, context.llm_runtime)
                .map(without_cache)
        };
        match enhanced {
            Ok(enhancement) => {
                let markdown = enhancement.markdown;
                result.llm_cache_hit = enhancement.cache_hit;
                result.warnings.extend(enhancement.warnings);
                let (meta, body) = if enhancement.metadata.is_some() {
                    // Typed metadata is separate from the document body. Source
                    // YAML inside cleaned_markdown remains source content.
                    (serde_json::Map::new(), markdown.as_str())
                } else {
                    output::split_frontmatter(&markdown)
                };
                if pure {
                    let prefix_len = markdown.len() - body.len();
                    result.pure_llm_prefix =
                        (prefix_len > 0).then(|| markdown[..prefix_len].to_owned());
                    result.frontmatter = meta;
                } else if let Some(metadata) = enhancement.metadata {
                    result
                        .frontmatter
                        .insert("description".into(), json!(metadata.description));
                    result
                        .frontmatter
                        .insert("tags".into(), json!(metadata.tags));
                } else {
                    result
                        .frontmatter
                        .extend(meta.into_iter().filter(|(key, _)| {
                            !["title", "source", "markitai_processed"].contains(&key.as_str())
                        }));
                }
                result.llm_markdown = Some(if pure {
                    body.to_owned()
                } else {
                    crate::markdown::normalize(body)
                });
                if send_pdf_images && !screenshots.is_empty() {
                    let enhanced = result.llm_markdown.as_mut().expect("enhanced body is set");
                    let references = screenshots
                        .iter()
                        .enumerate()
                        .map(|(index, screenshot)| match office_kind {
                            Some(kind) => {
                                office_media::reference(kind, index + 1, &screenshot.name)
                            }
                            None => formats::pdf_screenshot_reference(index + 1, &screenshot.name),
                        })
                        .filter(|reference| !enhanced.contains(reference.as_str()))
                        .collect::<Vec<_>>();
                    if !references.is_empty() {
                        enhanced.push_str("\n\n<!-- Page images for reference -->\n");
                        enhanced.push_str(&references.join("\n"));
                    }
                }
                result.usage = enhancement.usage;
            }
            Err(error) => {
                if matches!(error, Error::NoModelConfigured | Error::Unsupported(_)) {
                    return Err(error);
                }
                if output_dir.is_none()
                    && (screenshot_only
                        || (pdf_media_requested || office_media_requested)
                            && !pdf_has_reliable_text)
                {
                    // A failed visual-only memory request has no base text or
                    // persistent screenshots that could make fallback useful.
                    return Err(error);
                }
                if cfg["llm"]["on_failure"] == "fail" {
                    output::apply_profiles(&mut result, &cfg);
                    if let Some(dir) = &output_dir {
                        output::write_document_mode(
                            dir,
                            &name,
                            &mut result,
                            &doc.assets,
                            if pdf_input || office_media_requested {
                                output::Screenshots::PublishedPages(&screenshots)
                            } else {
                                output::Screenshots::New(&screenshots)
                            },
                            &cfg,
                            output::WritePolicy {
                                publication,
                                prepared: prepared.as_deref_mut(),
                            },
                        )?;
                    }
                    return Err(error);
                }
                result.warnings.push(if screenshot_only {
                    format!("LLM enhancement failed; captured screenshots retained: {error}")
                } else {
                    format!("LLM enhancement failed; base Markdown retained: {error}")
                });
            }
        }
    }
    if let Err(error) = image_enrichment::analyze(
        &doc,
        &mut result,
        source,
        image_input,
        &cfg,
        context.llm_runtime,
    ) {
        if matches!(error, Error::NoModelConfigured | Error::Unsupported(_)) || output_dir.is_none()
        {
            return Err(error);
        }
        if cfg["llm"]["on_failure"] == "fail" {
            output::apply_profiles(&mut result, &cfg);
            if let Some(dir) = &output_dir {
                output::write_document_mode(
                    dir,
                    &name,
                    &mut result,
                    &doc.assets,
                    if pdf_input || office_media_requested {
                        output::Screenshots::PublishedPages(&screenshots)
                    } else {
                        output::Screenshots::New(&screenshots)
                    },
                    &cfg,
                    output::WritePolicy {
                        publication,
                        prepared: prepared.as_deref_mut(),
                    },
                )?;
            }
            return Err(error);
        }
        result.warnings.push(format!(
            "Image analysis failed; base Markdown and assets retained: {error}"
        ));
    }
    if let Some(scope) = document_scope.as_ref() {
        result.usage = scope.usage();
        for warning in scope.take_warnings() {
            if !result.warnings.contains(&warning) {
                result.warnings.push(warning);
            }
        }
        if result.usage.requests > 0 || !result.usage.by_model.is_empty() {
            result.llm_cache_hit = false;
            if !result.usage.cost_complete() {
                result.warnings.push("Some observed LLM requests could not be priced. cost_usd is the known priced subtotal; the complete cost is unknown.".into());
            }
        }
    }
    if let Some(store) = stdout_assets {
        // Before profiles: persisted images become `file://` links, which
        // the visible-asset and wikilink profiles leave as they are.
        output::stdout_assets::persist(store, &mut result, &doc.assets, &screenshots, &cfg);
    }
    output::apply_profiles(&mut result, &cfg);
    if let Some(dir) = output_dir {
        output::write_document_mode(
            &dir,
            &name,
            &mut result,
            &doc.assets,
            if pdf_input || office_media_requested {
                output::Screenshots::PublishedPages(&screenshots)
            } else {
                output::Screenshots::New(&screenshots)
            },
            &cfg,
            output::WritePolicy {
                publication,
                prepared,
            },
        )?;
    }
    result.duration = start.elapsed().as_secs_f64();
    Ok(result)
}

fn screenshot_prefix<'a>(name: &'a str, cfg: &'a Value) -> &'a str {
    cfg.pointer("/output/filename")
        .and_then(Value::as_str)
        .map(|name| name.strip_suffix(".md").unwrap_or(name))
        .or_else(|| cfg.pointer("/output/reserved_stem").and_then(Value::as_str))
        .unwrap_or(name)
}

fn prepare_pdf_media(
    bytes: &[u8],
    name: &str,
    output_dir: Option<&Path>,
    cfg: &Value,
    vlm_disabled: bool,
) -> Result<(Document, Vec<Asset>, bool)> {
    let prefix = screenshot_prefix(name, cfg);
    let mut prepared = pdf_media::prepare(bytes, prefix, cfg, vlm_disabled)?;
    let reliable = prepared.has_reliable_text;
    // Freeze capture names before page references or model inputs are assembled.
    if let Some(dir) = output_dir {
        output::publish_page_screenshots(dir, &mut prepared.screenshots, cfg)?;
    }
    let (document, screenshots) = prepared.finish()?;
    Ok((document, screenshots, reliable))
}

pub fn is_url(source: &str) -> bool {
    source
        .get(..7)
        .is_some_and(|s| s.eq_ignore_ascii_case("http://"))
        || source
            .get(..8)
            .is_some_and(|s| s.eq_ignore_ascii_case("https://"))
}

/// Shared native adapter protocol. Errors are values; no host stdout is used.
pub fn convert_json(request: &str) -> String {
    let result = serde_json::from_str::<Request>(request)
        .map_err(|error| ConversionFailure::from(Error::from(error)))
        .and_then(|r| convert_detailed(&r.source, r.options));
    match result {
        Ok(result) => json!({"ok":true,"result":result}).to_string(),
        Err(failure) => {
            let mut error = json!({"code":failure.code(),"message":failure.to_string()});
            if failure.usage.requests > 0
                || failure.usage.input_tokens > 0
                || failure.usage.output_tokens > 0
                || !failure.usage.by_model.is_empty()
            {
                error["usage"] = json!(failure.usage);
            }
            json!({"ok":false,"error":error}).to_string()
        }
    }
}

fn vlm_ocr_disabled(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "no"
    )
}

fn screenshot_mime(bytes: &[u8]) -> Result<&'static str> {
    match image::guess_format(bytes) {
        Ok(image::ImageFormat::Jpeg) => Ok("image/jpeg"),
        Ok(image::ImageFormat::Png) => Ok("image/png"),
        Ok(image::ImageFormat::WebP) => Ok("image/webp"),
        _ => Err(Error::Conversion(
            "Captured page has an unsupported image encoding".into(),
        )),
    }
}

#[cfg(test)]
mod routing_tests {
    #[test]
    fn vlm_ocr_optout_honors_whitespace_and_all_nonfalse_values() {
        for value in ["1", " 1 ", "true", "enabled", "yes", "off", "arbitrary"] {
            assert!(super::vlm_ocr_disabled(value), "{value}");
        }
        for value in ["", " ", "0", " false ", "No"] {
            assert!(!super::vlm_ocr_disabled(value), "{value}");
        }
    }
}

#[cfg(test)]
mod hex_tests {
    #[test]
    fn hex_matches_the_previous_lower_hex_digest_spelling() {
        use sha2::{Digest, Sha256};
        assert_eq!(super::hex([0x00, 0x0f, 0xab, 0xff]), "000fabff");
        assert_eq!(super::hex([]), "");
        // FIPS 180-2 test vector for "abc".
        assert_eq!(
            super::hex(Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
