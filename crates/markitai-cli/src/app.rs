// Declared first so its `text!` macro is visible in the modules below.
#[macro_use]
#[path = "app/i18n.rs"]
pub(crate) mod i18n;
// Standard output goes through `stdout_write`, so a reader that stops early
// (`markitai note.txt | head -1`) ends the run quietly instead of panicking in
// std's macros. Declared before the modules so that they print the same way.
macro_rules! println {
    () => { $crate::app::stdout_line(format_args!("")) };
    ($($argument:tt)*) => { $crate::app::stdout_line(format_args!($($argument)*)) };
}
macro_rules! print {
    ($($argument:tt)*) => { $crate::app::stdout_write(format_args!($($argument)*)) };
}
#[path = "app/auth.rs"]
mod auth;
mod build_info;
#[path = "app/compat.rs"]
mod compat;
#[path = "app/doctor.rs"]
mod doctor;
#[path = "app/dry_run.rs"]
mod dry_run;
#[path = "app/guided.rs"]
mod guided;
#[path = "app/help_zh.rs"]
mod help_zh;
#[path = "app/interactive.rs"]
mod interactive;
#[path = "app/json_output.rs"]
mod json_output;
#[path = "app/logging.rs"]
mod logging;
#[path = "app/progress.rs"]
mod progress;
#[path = "app/remote_consent.rs"]
mod remote_consent;
#[path = "app/warnings.rs"]
mod warnings;
macro_rules! eprintln {
    ($($argument:tt)*) => { $crate::app::logging::diagnostic(format_args!($($argument)*)) };
}
// A stderr line in the terminal language. The file log keeps the English
// line, so log entries read the same in every language. Both literals use the
// same captured variables (or the same trailing arguments).
macro_rules! say {
    ($en:literal, $zh:literal $(, $($argument:tt)+)? $(,)?) => {
        match $crate::app::i18n::lang() {
            $crate::app::i18n::Lang::En => eprintln!($en $(, $($argument)+)?),
            $crate::app::i18n::Lang::Zh => $crate::app::logging::diagnostic_as(
                format_args!($en $(, $($argument)+)?),
                format_args!($zh $(, $($argument)+)?),
            ),
        }
    };
}
// Every platform publishes batches with native ownership.
#[path = "batch_run.rs"]
mod batch_run;
#[path = "provider_batch/mod.rs"]
mod provider_batch;
use crate::report::{
    self, ItemKind, ItemStatus, ReportOptions, RunFinished, RunInfo, RunItem, RunMode,
};
use chrono::{Local, SecondsFormat};
use clap::{ArgAction, CommandFactory, FromArgMatches, Parser, Subcommand};
use markitai_core::{ConversionOutput, ConversionUsage, ConvertContext, ConvertOptions, config};
use serde_json::{Value, json};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

// Help sections. Clap prints them in order of first use.
const OUTPUT_HELP: &str = "Output";
const CONFIG_HELP: &str = "Configuration";
const ENHANCE_HELP: &str = "LLM, OCR and screenshots";
const FETCH_HELP: &str = "URL fetching and backends";
const BATCH_HELP: &str = "Batch processing";
const CACHE_HELP: &str = "Cache and images";
const MESSAGES_HELP: &str = "Messages and logging";

const ROOT_AFTER_HELP: &str = "\
Presets (-p):
  minimal   plain conversion, no model processing
  standard  --llm --alt --desc
  rich      --llm --alt --desc --screenshot

Examples:
  markitai report.docx                     Print Markdown on stdout
  markitai report.pdf -o out/              Write out/report.pdf.md
  markitai notes.html -o out/notes.md      Choose the exact output file
  markitai ./docs -o out/ -g '**/*.pdf'    Convert the PDFs in a directory
  markitai links.urls -o out/ --resume     Continue an interrupted URL list
  markitai https://example.com -o out/     Convert a web page
  markitai scan.png --ocr                  Read the text of an image
  markitai report.pdf -p standard -o out/  Enhance with the configured model
  markitai init                            Create a configuration file
  markitai doctor                          Check models and optional backends";

const CONFIG_AFTER_HELP: &str = "\
Keys use dot notation, e.g. llm.enabled or llm.model_list[0].model_name. The file in
use is -c, then MARKITAI_CONFIG, then ./markitai.json, then the user config.json.

Examples:
  markitai config list -f table           Show every effective setting
  markitai config get llm.enabled         Read one value
  markitai config set output.dir ./out    Validate and save one value
  markitai config path                    Show which file is in use";

const AUTH_AFTER_HELP: &str = "\
Examples:
  markitai auth                         Status of every subscription runtime
  markitai auth claude status           One runtime (add --json for scripts)
  markitai auth chatgpt login           Sign in through the official runtime";

const MCP_AFTER_HELP: &str = "\
Examples:
  markitai mcp                          Serve the tools to an MCP client over stdio
  markitai -c cfg.json mcp              Serve with a particular configuration file";

const CACHE_AFTER_HELP: &str = "\
Examples:
  markitai cache stats                    Entry counts and disk use
  markitai cache stats -v --limit 10      Recent LLM entries per model
  markitai cache clear -y                 Clear without a confirmation prompt";

#[derive(Parser, Debug, Clone)]
#[command(name="markitai", version=markitai_core::VERSION,
    about="Convert documents and URLs to Markdown", after_help=ROOT_AFTER_HELP,
    disable_help_subcommand=true, args_override_self=true)]
struct Cli {
    #[arg(value_name = "INPUT")]
    /// Document, URL, .urls list, directory or atomic .numbers package.
    input: Option<String>,
    #[command(subcommand)]
    command: Option<Command>,
    #[arg(short = 'o', long, value_name = "PATH", help_heading = OUTPUT_HELP)]
    /// Output directory, or exact .md path for a single input. Omit for Markdown on stdout.
    output: Option<PathBuf>,
    // Requires -o; checked in execute() so the message can explain why.
    #[arg(long, conflicts_with_all=["dry_run","llm_batch_collect"], help_heading = OUTPUT_HELP)]
    /// Print one JSON result on stdout; requires -o. Usage errors remain on stderr.
    json: bool,
    #[arg(long, help_heading = OUTPUT_HELP)]
    /// Preview discovery without conversion or output publication.
    dry_run: bool,
    #[arg(long, overrides_with = "no_page_markers", help_heading = OUTPUT_HELP)]
    /// Keep PDF page and slide-number comments in final Markdown (off by default).
    page_markers: bool,
    #[arg(long, overrides_with = "page_markers", hide = true)]
    /// Omit PDF page and slide-number comments from final Markdown.
    no_page_markers: bool,
    #[arg(long, overrides_with = "no_record_history", help_heading = OUTPUT_HELP)]
    /// Archive this run for `markitai serve` history (--no-record-history disables); stdout-only conversions are not archived.
    record_history: bool,
    #[arg(long, overrides_with = "record_history", hide = true)]
    /// Disable history recording for this run.
    no_record_history: bool,
    #[arg(short = 'c', long, global = true, value_name = "PATH", help_heading = CONFIG_HELP)]
    /// Configuration path; must exist except for config set/edit, which can create it.
    config: Option<PathBuf>,
    #[arg(long, global = true, value_name = "JSON", help_heading = CONFIG_HELP)]
    /// Deep-merge inline JSON over the config file; explicit conversion flags still win.
    config_json: Option<String>,
    #[arg(short = 'p', long, value_name = "NAME", help_heading = CONFIG_HELP)]
    /// Use minimal, standard, rich or a configured preset (case-insensitive).
    preset: Option<String>,
    #[arg(long, value_name = "NAME", value_parser=["rag","obsidian","okf"], ignore_case=true, help_heading = CONFIG_HELP)]
    /// Shape assets/frontmatter for rag, obsidian or okf; independent of the preset.
    profile: Option<String>,
    #[arg(short = 'I', long, help_heading = CONFIG_HELP)]
    /// Choose a conversion in a terminal; edits apply to this session only.
    interactive: bool,
    #[arg(long, overrides_with = "no_llm", help_heading = ENHANCE_HELP)]
    /// Enable model processing (--no-llm disables). The last of a flag pair wins; without either, the configuration decides.
    llm: bool,
    #[arg(long, overrides_with = "llm", hide = true)]
    /// Disable model processing.
    no_llm: bool,
    #[arg(long, overrides_with = "no_alt", help_heading = ENHANCE_HELP)]
    /// Generate image alt text when LLM processing is enabled (--no-alt disables).
    alt: bool,
    #[arg(long, overrides_with = "alt", hide = true)]
    /// Disable image alt text generation.
    no_alt: bool,
    #[arg(long, overrides_with = "no_desc", help_heading = ENHANCE_HELP)]
    /// Write image descriptions when LLM processing is enabled (--no-desc disables).
    desc: bool,
    #[arg(long, overrides_with = "desc", hide = true)]
    /// Disable image descriptions.
    no_desc: bool,
    #[arg(long, overrides_with = "no_ocr", help_heading = ENHANCE_HELP)]
    /// Read scanned content with local OCR or a vision model (--no-ocr disables).
    ocr: bool,
    #[arg(long, overrides_with = "ocr", hide = true)]
    /// Disable OCR.
    no_ocr: bool,
    #[arg(long, overrides_with = "no_screenshot", help_heading = ENHANCE_HELP)]
    /// Capture supported document pages or browser pages; optional backends may be required (--no-screenshot disables).
    screenshot: bool,
    #[arg(long, overrides_with = "screenshot", hide = true)]
    /// Disable screenshots.
    no_screenshot: bool,
    #[arg(long, overrides_with = "no_screenshot_only", help_heading = ENHANCE_HELP)]
    /// Use screenshots as content (implies --screenshot). With --llm, read pixels; without --llm, ordinary web pages save images without Markdown. PDF media retains Markdown. URL --pure takes precedence.
    screenshot_only: bool,
    #[arg(long, overrides_with = "screenshot_only", hide = true)]
    /// Disable screenshot-only content selection.
    no_screenshot_only: bool,
    #[arg(long, overrides_with = "no_pure", help_heading = ENHANCE_HELP)]
    /// Preserve source text without ordinary generated metadata; URL pure takes precedence over visual LLM input (--no-pure disables).
    pure: bool,
    #[arg(long, overrides_with = "pure", hide = true)]
    /// Disable pure mode.
    no_pure: bool,
    #[arg(long, help_heading = ENHANCE_HELP)]
    /// Keep the base Markdown alongside enhanced output.
    keep_base: bool,
    #[arg(long, value_name = "N", value_parser = at_least_one, help_heading = ENHANCE_HELP)]
    /// Maximum in-flight model requests shared by this conversion run.
    llm_concurrency: Option<u32>,
    #[arg(short='s', long, value_name = "NAME", value_parser=["auto","static","playwright","defuddle","jina","cloudflare"], help_heading = FETCH_HELP)]
    /// URL strategy: auto (static, then the local browser; remote services only after you opt in with fetch.remote_consent), static or playwright run locally; defuddle, jina and cloudflare (your account) send the URL to that remote service.
    strategy: Option<String>,
    #[arg(short='b', long, value_name = "NAME", value_parser=["native","cloudflare"], help_heading = FETCH_HELP)]
    /// File backend: native, or cloudflare to convert supported files with Workers AI in your Cloudflare account (the file is uploaded).
    backend: Option<String>,
    #[arg(long, help_heading = FETCH_HELP)]
    /// Forbid remote extraction services; does not disable explicitly configured model requests.
    no_remote_fetch: bool,
    #[arg(long, help_heading = BATCH_HELP)]
    /// Resume a directory or .urls batch with matching paths/options; completed entries stay completed.
    resume: bool,
    #[arg(short='j', long, value_name = "N", value_parser = at_least_one, help_heading = BATCH_HELP)]
    /// Maximum concurrent file conversions; independent of URL and model request limits.
    batch_concurrency: Option<u32>,
    #[arg(long, value_name = "N", value_parser = at_least_one, help_heading = BATCH_HELP)]
    /// Maximum concurrent URL conversions, separately from file processing.
    url_concurrency: Option<u32>,
    #[arg(short='g', long="glob", value_name = "PATTERN", action=ArgAction::Append, help_heading = BATCH_HELP)]
    /// Include/exclude relative directory paths; repeatable, ! prefix excludes. Quote patterns in the shell.
    globs: Vec<String>,
    #[arg(long, value_name = "N", help_heading = BATCH_HELP)]
    /// Directory scan depth; 0 scans only the input directory.
    max_depth: Option<usize>,
    #[arg(long, help_heading = BATCH_HELP)]
    /// Submit directory text enhancement to the OpenAI Batch API; --resume continues frozen work in -o.
    llm_batch: bool,
    #[arg(long, value_name = "SECONDS", value_parser=clap::value_parser!(u64).range(60..), help_heading = BATCH_HELP)]
    /// Maximum local wait for a provider batch (seconds, at least 60); expiry does not cancel it.
    llm_batch_timeout: Option<u64>,
    #[arg(long, value_name = "BATCH_ID", help_heading = BATCH_HELP)]
    /// Collect a saved provider batch into its original -o directory; no input is required.
    llm_batch_collect: Option<String>,
    #[arg(long, overrides_with = "cache", help_heading = CACHE_HELP)]
    /// Bypass cache reads while still writing successful fresh results (--cache restores reads).
    no_cache: bool,
    #[arg(long, overrides_with = "no_cache", hide = true)]
    /// Allow cache reads without forcing a disabled cache on.
    cache: bool,
    #[arg(long, value_name = "PATTERNS", help_heading = CACHE_HELP)]
    /// Comma-separated glob patterns that bypass cache reads for matching inputs.
    no_cache_for: Option<String>,
    #[arg(long, overrides_with = "compress", help_heading = CACHE_HELP)]
    /// Disable image compression (--compress enables it).
    no_compress: bool,
    #[arg(long, overrides_with = "no_compress", hide = true)]
    /// Enable image compression.
    compress: bool,
    #[arg(short = 'v', long, help_heading = MESSAGES_HELP)]
    /// Show details such as the report path. Single inputs are otherwise quiet by default: only the written path, warnings and errors; stdout Markdown stays clean.
    verbose: bool,
    #[arg(short = 'q', long, help_heading = MESSAGES_HELP)]
    /// Print errors only: no written path, warnings, progress or batch summary.
    quiet: bool,
    #[arg(long, value_name = "LEVEL", value_parser=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], ignore_case=true, help_heading = MESSAGES_HELP)]
    /// Conversion file-log level; needs log.dir. Console output still follows --verbose/--quiet.
    log_level: Option<String>,
}

/// The root command as printed and parsed, in the terminal language. Option
/// help sits on its own line: without terminal wrapping, long descriptions
/// stay readable that way, while the short command list keeps its one-line
/// layout.
fn cli_command() -> clap::Command {
    let command = match i18n::lang() {
        i18n::Lang::En => english_command_at_width(progress::columns()),
        i18n::Lang::Zh => help_zh::localize(english_command),
    };
    // `--help` ends with the identity of this build: which commit it came from
    // and when it was built. `-h` keeps its short form.
    let closing = command.get_after_help().map(|text| text.to_string());
    let command = command.after_long_help(build_info::append(i18n::lang(), closing));
    if progress::initialize() {
        command
    } else {
        command.color(clap::ColorChoice::Never)
    }
}

/// The command tree as the derive declares it, with English help.
fn english_command() -> clap::Command {
    Cli::command().mut_args(|arg| arg.next_line_help(true))
}

/// Keep clap's root usage tokens intact when its heading would overflow a
/// narrow terminal. Only the heading separator changes; ordinary-width help,
/// subcommand templates and parser usage errors retain clap's defaults.
fn english_command_at_width(width: usize) -> clap::Command {
    let command = english_command();
    // The root usage is ASCII and shorter than 60 columns. Avoid building a
    // reference command on the ordinary-width path used by most invocations.
    if width < 60
        && command
            .clone()
            .render_usage()
            .to_string()
            .lines()
            .any(|line| line.len() > width)
    {
        command.help_template(
            "{before-help}{about-with-newline}\n{usage-heading}\n  {usage}\n\n{all-args}{after-help}",
        )
    } else {
        command
    }
}

/// Clap's built-in range message prints the whole u32 range; say the rule instead.
fn at_least_one(value: &str) -> Result<u32, String> {
    match value.trim().parse::<u32>() {
        Ok(0) => Err("must be at least 1".into()),
        Ok(number) => Ok(number),
        Err(_) => Err("expected a whole number of at least 1".into()),
    }
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    /// Inspect or edit configuration.
    #[command(after_help = CONFIG_AFTER_HELP)]
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Create or update a configuration file with detected API models.
    #[command(
        long_about = "Create or update a configuration file with detected API models.\n\n\
            Detects API models from the environment (MODEL and provider API keys) without\n\
            making requests or saving key values. LLM processing stays disabled in the\n\
            generated file; enable it per run with --llm or with\n\
            `markitai config set llm.enabled true`.",
        after_help = "Examples:\n  markitai init                Choose the location and how to treat an existing file\n  markitai init -y             Create or update the user configuration without prompts\n  markitai init --local        Write ./markitai.json for this project\n  markitai init -o cfg.json    Write to a custom path"
    )]
    Init {
        #[arg(short = 'y', long)]
        /// Do not prompt: create the file, or add newly detected models to an existing one.
        yes: bool,
        #[arg(short = 'o', long, value_name = "PATH")]
        /// Write this file, or markitai.json inside this directory.
        output: Option<PathBuf>,
        #[arg(long)]
        /// Write ./markitai.json in the current directory instead of the user configuration.
        local: bool,
    },
    /// Diagnose configured workflows and optional native backends.
    #[command(
        long_about = "Diagnose configured workflows and optional native backends.\n\n\
            Missing optional backends are reported but do not fail. The exit status is 1\n\
            only when the configuration asks for something this machine cannot deliver:\n\
            an active model whose credentials are missing, a configured browser workflow\n\
            that cannot launch, an unavailable subscription runtime, or a failed --fix\n\
            repair. No model request is sent and no remote page is opened.",
        after_help = "Examples:\n  markitai doctor           Human-readable report\n  markitai doctor --json    Machine-readable checks\n  markitai doctor --fix     Install a missing browser and required local OCR models"
    )]
    Doctor {
        #[arg(long)]
        /// Print the checks as one JSON object.
        json: bool,
        #[arg(long)]
        /// Install the official Chrome headless shell when missing, and install or repair local OCR models.
        fix: bool,
        #[arg(long)]
        /// Python package extras; not applicable to this native build and rejected.
        suggest_extras: bool,
    },
    /// Inspect or clear the LLM and URL fetch caches.
    #[command(after_help = CACHE_AFTER_HELP)]
    Cache {
        #[command(subcommand)]
        command: CacheCommand,
    },
    /// Inspect subscription authentication or delegate login to an official runtime.
    #[command(after_help = AUTH_AFTER_HELP)]
    Auth {
        #[command(subcommand)]
        command: Option<auth::Command>,
    },
    /// Run the native REST conversion service and web interface.
    #[command(
        after_help = "Examples:\n  markitai serve                          http://127.0.0.1:3600, opens a browser\n  markitai serve --port 8080 --no-open    Another port, no browser\n  markitai serve --host 0.0.0.0           LAN access with the printed access token"
    )]
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        /// Interface to bind. All API clients need the access token printed at startup (or MARKITAI_SERVE_TOKEN).
        host: String,
        #[arg(long, default_value_t = 3600)]
        /// Port to listen on.
        port: u16,
        #[arg(long)]
        /// Do not open a browser after startup.
        no_open: bool,
        #[arg(long)]
        /// Disable API token authentication. Remote clients can upload files and read or delete history; URL conversion and model settings require a direct local connection.
        no_auth: bool,
        #[arg(long, action = ArgAction::Append, value_name = "HOSTNAME")]
        /// Also allow this hostname in Host/Origin headers (repeatable). Local Origins must still match the request port.
        allowed_host: Vec<String>,
    },
    /// Run the native MCP service over standard input/output.
    #[command(
        long_about = "Run the native MCP service over standard input/output\n\n\
            Tools: convert_document and convert_url convert one document or page,\n\
            batch_convert starts a directory or URL-list job and job_status reports on it.\n\
            An MCP client starts this command; it answers on stdout and logs on stderr.",
        after_help = MCP_AFTER_HELP
    )]
    Mcp,
}

#[derive(Subcommand, Debug, Clone)]
enum CacheCommand {
    /// Show cache entry counts and disk use.
    #[command(
        after_help = "Examples:\n  markitai cache stats                  Entry counts and disk use\n  markitai cache stats --json           Exact byte counts for scripts\n  markitai cache stats -v --limit 5     The five most recent LLM entries per model"
    )]
    Stats {
        #[arg(long)]
        /// Print the statistics as JSON.
        json: bool,
        #[arg(short, long)]
        /// List recent LLM cache entries per model.
        verbose: bool,
        #[arg(long, value_name = "N", default_value_t = 20)]
        /// Maximum entries listed with --verbose.
        limit: usize,
    },
    /// Clear the LLM and URL fetch caches.
    #[command(
        after_help = "Examples:\n  markitai cache clear                           Ask before clearing\n  markitai cache clear -y                        Clear without asking\n  markitai cache clear -y --include-spa-domains  Also forget learned browser-only domains"
    )]
    Clear {
        #[arg(short, long)]
        /// Clear without asking for confirmation.
        yes: bool,
        #[arg(long)]
        /// Also forget learned browser-only domains.
        include_spa_domains: bool,
    },
    /// Inspect or clear learned browser-domain routing.
    #[command(
        after_help = "Examples:\n  markitai cache spa-domains           List the learned domains\n  markitai cache spa-domains --json    As JSON\n  markitai cache spa-domains --clear   Forget every learned domain"
    )]
    SpaDomains {
        #[arg(long)]
        /// Print the domains as JSON.
        json: bool,
        #[arg(long)]
        /// Forget every learned domain.
        clear: bool,
    },
}

#[derive(Subcommand, Debug, Clone)]
enum ConfigCommand {
    /// Show the effective configuration; secrets are redacted.
    #[command(
        after_help = "Examples:\n  markitai config list               The whole configuration as JSON\n  markitai config list -f table      One setting per line\n  markitai config list -f yaml       As YAML"
    )]
    List {
        #[arg(short='f', long="format", default_value="json", value_parser=["json","yaml","table"], ignore_case=true)]
        /// Output format.
        format: String,
        #[arg(long)]
        /// Show secret values instead of redacting them (unsafe for shared logs).
        show_secrets: bool,
    },
    /// Show which configuration file is in use.
    #[command(
        after_help = "Examples:\n  markitai config path               The file in use, or where files are looked for\n  markitai -c other.json config path The file chosen with -c"
    )]
    Path,
    /// Check a configuration file against the schema.
    #[command(
        after_help = "Examples:\n  markitai config validate                 Check the configuration in use\n  markitai config validate ./markitai.json Check a particular file"
    )]
    Validate {
        /// File to check; defaults to the configuration in use.
        config_file: Option<PathBuf>,
    },
    /// Print one value; sections print as JSON.
    #[command(
        after_help = "Examples:\n  markitai config get llm.enabled     One value\n  markitai config get output          A whole section as JSON"
    )]
    Get {
        /// Dot-notation key, e.g. llm.enabled.
        key: String,
        #[arg(long)]
        /// Show secret values instead of redacting them (unsafe for shared logs).
        show_secrets: bool,
    },
    /// Validate and save one value; invalid values are not written.
    #[command(
        after_help = "Examples:\n  markitai config set output.on_conflict skip   Never replace an existing result\n  markitai config set output.dir ~/Documents/md Where batches write by default\n  markitai config set llm.enabled true          Turn on model processing"
    )]
    Set {
        /// Dot-notation key, e.g. output.on_conflict.
        key: String,
        /// New value; parsed according to the key's declared type.
        value: String,
        #[arg(long)]
        /// Echo secret values instead of redacting them (unsafe for shared logs).
        show_secrets: bool,
    },
    /// Edit settings in a terminal; each value is validated and saved at once.
    #[command(
        after_help = "Examples:\n  markitai config edit    Browse, search (/keyword) and change settings; q quits"
    )]
    Edit,
}

/// Whether the executable was started under the MCP launcher name.
fn is_mcp_launcher(arguments: &[std::ffi::OsString]) -> bool {
    arguments.first().is_some_and(|name| {
        let Some(name) = Path::new(name).file_name().and_then(|name| name.to_str()) else {
            return false;
        };
        #[cfg(windows)]
        {
            name.eq_ignore_ascii_case("markitai-mcp")
                || name.eq_ignore_ascii_case("markitai-mcp.exe")
        }
        #[cfg(not(windows))]
        {
            matches!(name, "markitai-mcp" | "markitai-mcp.exe")
        }
    })
}

/// Under the launcher name the `mcp` subcommand is the whole command: usage
/// lines say `markitai-mcp` (clap's own name for the subcommand in the version
/// line already is) and `--version` is answered. The subcommand of `markitai`
/// has neither change, and its help stays as it is.
fn as_mcp_launcher(command: clap::Command) -> clap::Command {
    command.mut_subcommand("mcp", |mcp| {
        let mcp = mcp
            .override_usage("markitai-mcp [OPTIONS]")
            .version(markitai_core::VERSION);
        match i18n::lang() {
            i18n::Lang::En => mcp,
            i18n::Lang::Zh => mcp.disable_version_flag(true).arg(help_zh::version_flag()),
        }
    })
}

#[cfg(windows)]
pub(crate) fn erase_status_line() {
    progress::erase();
}

pub fn run() -> i32 {
    let _ = progress::initialize();
    let mut arguments: Vec<_> = std::env::args_os().collect();
    // A distribution may expose this executable through the existing MCP name.
    // Select the subcommand before the no-argument help path, so stdout remains
    // the protocol stream even when a client starts the alias without flags.
    let launcher = is_mcp_launcher(&arguments);
    if launcher {
        arguments.insert(1, "mcp".into());
    }
    if arguments.len() == 1 {
        print_root_help();
        return settle_stdout(0);
    }
    if let Some(message) = compat::removed_option(&arguments[1..]) {
        let _ = cli_command()
            .error(clap::error::ErrorKind::UnknownArgument, message)
            .print();
        return 2;
    }
    let mut command = cli_command();
    if launcher {
        command = as_mcp_launcher(command);
    }
    let cli = command
        .try_get_matches_from_mut(arguments)
        .and_then(|matches| Cli::from_arg_matches(&matches))
        .unwrap_or_else(|error| error.format(&mut command).exit());
    // Serve and MCP drain active work on the signals their asynchronous
    // runtime owns; the remaining terminating signals kill external runtimes.
    crate::signals::install_fatal_cleanup(match cli.command {
        Some(Command::Serve { .. }) => crate::signals::Owned::InterruptAndTerminate,
        Some(Command::Mcp) => crate::signals::Owned::Interrupt,
        _ => crate::signals::Owned::None,
    });
    let code = match execute(&cli) {
        Ok(code) => code,
        Err((code, message)) => {
            eprintln!("Error: {message}");
            if cli.json && code != 2 {
                emit_json(&[], Some(&message));
            }
            code
        }
    };
    let code = settle_stdout(code);
    if let Err(error) = logging::finish(code) {
        eprintln!("Error: {error}");
        return if code == 0 { 1 } else { code };
    }
    code
}

/// The first failure writing standard output. Later writes are dropped, and
/// [`settle_stdout`] turns the failure into the outcome of the run.
static STDOUT_FAILURE: Mutex<Option<io::Error>> = Mutex::new(None);

/// Write to standard output, remembering a failure instead of panicking as
/// std's `println!` does.
fn with_stdout(write: impl FnOnce(&mut io::StdoutLock<'_>) -> io::Result<()>) {
    let mut failure = STDOUT_FAILURE.lock().unwrap_or_else(|e| e.into_inner());
    write_unless_failed(&mut failure, &mut io::stdout().lock(), write);
}

/// Run `write` on `out` unless an earlier write failed, which would leave the
/// output with a hole; keep the first failure.
fn write_unless_failed<W>(
    failure: &mut Option<io::Error>,
    out: &mut W,
    write: impl FnOnce(&mut W) -> io::Result<()>,
) {
    if failure.is_none()
        && let Err(error) = write(out)
    {
        *failure = Some(error);
    }
}

/// What `print!` writes with.
fn stdout_write(arguments: std::fmt::Arguments<'_>) {
    with_stdout(|stdout| stdout.write_fmt(arguments));
}

/// What `println!` writes with.
fn stdout_line(arguments: std::fmt::Arguments<'_>) {
    with_stdout(|stdout| {
        stdout.write_fmt(arguments)?;
        stdout.write_all(b"\n")
    });
}

/// The root help on standard output, followed by an empty line.
fn print_root_help() {
    with_stdout(|stdout| {
        cli_command().print_help()?;
        stdout.write_all(b"\n")
    });
}

/// The exit status and the message after standard output failed. A reader that
/// stops early (`markitai note.txt | head -1`) has chosen to: like other Unix
/// tools, the run ends quietly with the status it already had. Any other
/// failure, such as a full disk behind `> file`, is named once and fails a run
/// that would have succeeded.
fn stdout_outcome(code: i32, failure: Option<&io::Error>) -> (i32, Option<String>) {
    match failure {
        Some(error) if error.kind() != io::ErrorKind::BrokenPipe => (
            if code == 0 { 1 } else { code },
            Some(format!("Error: Cannot write to standard output: {error}")),
        ),
        _ => (code, None),
    }
}

fn settle_stdout(code: i32) -> i32 {
    let failure = STDOUT_FAILURE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    let (code, message) = stdout_outcome(code, failure.as_ref());
    if let Some(message) = message {
        eprintln!("{message}");
    }
    code
}

type CliResult<T> = Result<T, (i32, String)>;
fn runtime(error: impl std::fmt::Display) -> (i32, String) {
    (1, error.to_string())
}
fn unsupported(feature: &str) -> (i32, String) {
    (
        1,
        format!("{feature} is not implemented in this Rust development build; see docs/cli.md"),
    )
}
fn tri(yes: bool, no: bool) -> Option<bool> {
    if yes {
        Some(true)
    } else if no {
        Some(false)
    } else {
        None
    }
}

fn execute(cli: &Cli) -> CliResult<i32> {
    if cli.json && cli.output.is_none() {
        return Err((
            2,
            "--json requires -o: stdout carries the JSON result, so the Markdown needs an output directory or .md file".into(),
        ));
    }
    if cli.command.is_some()
        && (cli.input.is_some()
            || cli.interactive
            || cli.llm_batch
            || cli.llm_batch_collect.is_some()
            || cli.llm_batch_timeout.is_some())
    {
        return Err((
            2,
            "Cannot mix INPUT, --interactive or Provider Batch options with a subcommand".into(),
        ));
    }
    let permits_missing = matches!(
        cli.command,
        Some(Command::Config {
            command: ConfigCommand::Set { .. } | ConfigCommand::Edit
        })
    );
    if let Some(path) = &cli.config
        && !path.is_file()
        && !permits_missing
    {
        return Err((
            2,
            format!("Configuration file does not exist: {}", path.display()),
        ));
    }
    let overrides = cli
        .config_json
        .as_deref()
        .map(serde_json::from_str::<Value>)
        .transpose()
        .map_err(|e| (2, format!("Invalid --config-json: {e}")))?;
    if overrides.as_ref().is_some_and(|v| !v.is_object()) {
        return Err((2, "--config-json must be a JSON object".into()));
    }
    if let Some(command) = &cli.command {
        return subcommand(cli, command, overrides);
    }
    if cli.interactive && cli.json {
        return Err((2, "--interactive and --json cannot be used together".into()));
    }
    if let Some(id) = &cli.llm_batch_collect {
        if cli.input.is_some() || cli.interactive || cli.llm_batch || cli.resume || cli.dry_run {
            return Err((2, "Batch collection cannot be mixed with input, submission, resume, preview or interactive mode".into()));
        }
        return provider_batch::collect(cli, id, conversion_config(cli, overrides)?);
    }
    if cli.llm_batch_timeout.is_some() && !cli.llm_batch {
        return Err((2, "--llm-batch-timeout requires --llm-batch".into()));
    }
    if cli.llm_batch && cli.resume {
        if cli.interactive || cli.dry_run {
            return Err((
                2,
                "Frozen Batch resume cannot use preview or interactive mode".into(),
            ));
        }
        return provider_batch::resume(cli, conversion_config(cli, overrides)?);
    }
    if cli.input.is_none() && !cli.interactive {
        print_root_help();
        return Ok(0);
    }
    let cfg = conversion_config(cli, overrides)?;
    if cli.interactive {
        let Some(run) = guided::collect(cli, cfg)? else {
            eprintln!("{}", text!("Cancelled.", "已取消。"));
            return Ok(0);
        };
        let mut effective = cli.clone();
        effective.input = Some(run.input);
        effective.output = Some(run.output);
        effective.interactive = false;
        return execute_conversion(
            &effective,
            effective.input.as_deref().unwrap(),
            run.config,
            effective.output.clone(),
        );
    }
    execute_conversion(cli, cli.input.as_deref().unwrap(), cfg, cli.output.clone())
}

/// An output location with a leading `~` or `~/` replaced by the home
/// directory. Nothing else changes (`~name` and `./~` stay literal names), and
/// a path that cannot be expanded is refused instead of becoming a directory
/// literally called `~`.
fn home_relative(path: &Path) -> CliResult<PathBuf> {
    if path.strip_prefix("~").is_err() {
        return Ok(path.to_owned());
    }
    let expanded = config::expand_home(path);
    if expanded.strip_prefix("~").is_ok() {
        return Err((
            1,
            format!(
                "Cannot expand ~ in {}: HOME is not set. Spell the directory out, or use ./~ for a directory literally named ~",
                path.display()
            ),
        ));
    }
    Ok(expanded)
}

fn conversion_config(cli: &Cli, overrides: Option<Value>) -> CliResult<Value> {
    let raw = config::raw(cli.config.as_deref(), overrides).map_err(runtime)?;
    let mut cfg = config::normalize(&raw).map_err(runtime)?;
    // Defaults are not choices: the default `remote_consent: always` is not an
    // opt-in to remote fallback, nor does the default `fallback_patterns`
    // list make X and other social sites browser-first. Values the user wrote
    // (file or --config-json) are.
    remote_consent::install(&raw, cli.quiet);
    if let Some(name) = &cli.preset {
        let name = name.to_lowercase();
        let preset = match name.as_str() {
            "minimal" => {
                json!({"llm":false,"alt":false,"desc":false,"ocr":false,"screenshot":false})
            }
            "standard" => json!({"llm":true,"alt":true,"desc":true,"ocr":false,"screenshot":false}),
            "rich" => json!({"llm":true,"alt":true,"desc":true,"ocr":false,"screenshot":true}),
            _ => cfg["presets"].get(&name).cloned().ok_or_else(|| {
                let mut custom: Vec<_> = cfg["presets"]
                    .as_object()
                    .map(|presets| presets.keys().cloned().collect())
                    .unwrap_or_default();
                crate::sort::by(&mut custom, String::cmp);
                let mut available = vec!["minimal".to_owned(), "rich".into(), "standard".into()];
                available.extend(custom);
                (
                    1,
                    format!(
                        "Unknown preset '{name}'. Available: {}",
                        available.join(", ")
                    ),
                )
            })?,
        };
        for (key, section, field) in [
            ("llm", "llm", "enabled"),
            ("ocr", "ocr", "enabled"),
            ("alt", "image", "alt_enabled"),
            ("desc", "image", "desc_enabled"),
            ("screenshot", "screenshot", "enabled"),
        ] {
            cfg[section][field] = json!(preset[key].as_bool().unwrap_or(false));
        }
    }
    for (section, field, value) in [
        ("llm", "enabled", tri(cli.llm, cli.no_llm)),
        ("ocr", "enabled", tri(cli.ocr, cli.no_ocr)),
        ("image", "alt_enabled", tri(cli.alt, cli.no_alt)),
        ("image", "desc_enabled", tri(cli.desc, cli.no_desc)),
        ("image", "compress", tri(cli.compress, cli.no_compress)),
        (
            "screenshot",
            "enabled",
            tri(cli.screenshot, cli.no_screenshot),
        ),
        (
            "screenshot",
            "screenshot_only",
            tri(cli.screenshot_only, cli.no_screenshot_only),
        ),
        ("llm", "pure", tri(cli.pure, cli.no_pure)),
        (
            "output",
            "page_markers",
            tri(cli.page_markers, cli.no_page_markers),
        ),
    ] {
        if let Some(value) = value {
            cfg[section][field] = json!(value);
        }
    }
    let env = config::environment();
    if !cli.pure
        && !cli.no_pure
        && env
            .get("MARKITAI_PURE")
            .is_some_and(|v| ["1", "true", "yes"].contains(&v.trim()))
    {
        cfg["llm"]["pure"] = json!(true);
    }
    if cli.keep_base {
        cfg["llm"]["keep_base"] = json!(true);
    }
    if cli.screenshot_only {
        cfg["screenshot"]["enabled"] = json!(true);
    }
    if let Some(value) = env
        .get("MARKITAI_RECORD_HISTORY")
        .filter(|v| !v.trim().is_empty())
    {
        cfg["history"]["record"] =
            json!(["1", "true", "yes", "on"].contains(&value.trim().to_lowercase().as_str()));
    }
    if let Some(enabled) = tri(cli.record_history, cli.no_record_history) {
        cfg["history"]["record"] = json!(enabled);
    }
    if let Some(bypass) = tri(cli.no_cache, cli.cache) {
        cfg["cache"]["no_cache"] = json!(bypass);
    }
    if let Some(patterns) = cli
        .no_cache_for
        .as_ref()
        .filter(|patterns| !patterns.is_empty())
    {
        cfg["cache"]["no_cache_patterns"] = json!(
            patterns
                .split(',')
                .map(str::trim)
                .filter(|p| !p.is_empty())
                .collect::<Vec<_>>()
        );
    }
    if let Some(profile) = &cli.profile {
        cfg["output"]["profile"] = json!(profile.to_lowercase());
    }
    if let Some(strategy) = &cli.strategy {
        cfg["fetch"]["strategy"] = json!(strategy);
    }
    if cli.no_remote_fetch {
        cfg["fetch"]["remote_consent"] = json!("never");
        cfg["fetch"]["no_remote"] = json!(true);
    }
    if let Some(backend) = &cli.backend {
        cfg["fetch"]["cloudflare"]["convert_enabled"] = json!(backend == "cloudflare");
    }
    for (field, value) in [
        ("concurrency", cli.batch_concurrency),
        ("url_concurrency", cli.url_concurrency),
    ] {
        if let Some(n) = value {
            cfg["batch"][field] = json!(n);
        }
    }
    if let Some(n) = cli.llm_concurrency {
        cfg["llm"]["concurrency"] = json!(n);
    }
    config::validate(&cfg).map_err(runtime)?;
    Ok(cfg)
}

fn execute_conversion(
    cli: &Cli,
    input: &str,
    mut cfg: Value,
    output: Option<PathBuf>,
) -> CliResult<i32> {
    // A leading `~` is expanded here, once, for `-o` and `output.dir` alike:
    // the shell leaves `--output=~/x` and a configuration value alone, and the
    // claim, state and report layers below all take the path literally.
    let mut output = output.as_deref().map(home_relative).transpose()?;
    // Clap accepts any spelling of a level; the logger knows the upper-case one.
    let level = cli.log_level.as_deref().map(str::to_ascii_uppercase);
    logging::start(&cfg, level.as_deref()).map_err(runtime)?;
    logging::event(
        logging::Level::Debug,
        "Configuration loaded; native CLI conversion starting",
    );
    let input_path = Path::new(input);
    let directory = !is_url(input)
        && input_path.is_dir()
        && !markitai_core::formats::is_numbers_package_path(input_path);
    if cli.llm_batch && !directory {
        return Err((2, "--llm-batch requires a directory input".into()));
    }
    let batch = !is_url(input)
        && (directory
            || input_path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("urls")));
    if cli.resume && !batch {
        return Err((
            1,
            "--resume for a single file or URL is not implemented: only directory and .urls batches save progress. Run it again without --resume.".into(),
        ));
    }
    let mode = if directory {
        RunMode::Directory
    } else if batch {
        RunMode::UrlList
    } else if is_url(input) {
        RunMode::SingleUrl
    } else {
        RunMode::SingleFile
    };
    if !directory && (!cli.globs.is_empty() || cli.max_depth.is_some()) {
        return Err((2, "--glob and --max-depth require a directory input".into()));
    }
    if is_url(input)
        && config::enabled(&cfg, "/screenshot/screenshot_only")
        && !config::enabled(&cfg, "/llm/enabled")
        && output.is_none()
    {
        output = Some(match cfg["output"]["dir"].as_str() {
            Some(dir) => home_relative(Path::new(dir))?,
            None => std::env::current_dir().map_err(runtime)?,
        });
    }
    if batch && output.is_none() {
        output = cfg["output"]["dir"]
            .as_str()
            .map(|dir| home_relative(Path::new(dir)))
            .transpose()?;
    }
    if batch && output.is_none() {
        return Err((
            1,
            "A directory or .urls batch needs an output directory: pass -o DIR or set output.dir in the configuration".into(),
        ));
    }
    if let Some(path) = &output
        && path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
        && !path.is_dir()
        // A trailing separator, as in `-o notes.md/`, names a directory.
        && !path
            .as_os_str()
            .to_string_lossy()
            .ends_with(std::path::is_separator)
    {
        if batch {
            return Err((
                2,
                "A batch requires an output directory, not a .md file".into(),
            ));
        }
        if cfg["output"]["on_conflict"].as_str() == Some("overwrite")
            && !is_url(input)
            && std::fs::canonicalize(input_path).is_ok_and(|source| {
                std::fs::canonicalize(config::expand_home(path))
                    .is_ok_and(|target| target == source)
            })
        {
            return Err((
                2,
                format!(
                    "Output {} is the input file itself, and overwriting it would destroy the source. Choose another output name or directory.",
                    path.display()
                ),
            ));
        }
        cfg["output"]["filename"] = json!(path.file_name().unwrap_or_default().to_string_lossy());
        output = Some(
            path.parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."))
                .to_owned(),
        );
    }
    if let Some(problem) = output.as_deref().and_then(output_location_problem) {
        // A preview publishes nothing, so it reports the problem and still lists targets.
        if !cli.dry_run {
            return Err((1, problem));
        }
        eprintln!("Warning: {problem}");
    }
    if !cli.quiet
        && !config::enabled(&cfg, "/llm/enabled")
        && let Some((flags, names)) = match (cli.alt, cli.desc) {
            (true, true) => Some(("--alt and --desc have", "--alt 和 --desc")),
            (true, false) => Some(("--alt has", "--alt")),
            (false, true) => Some(("--desc has", "--desc")),
            (false, false) => None,
        }
    {
        say!(
            "Warning: {flags} no effect without --llm (or the standard/rich preset)",
            "Warning: 未启用 --llm（或 standard/rich 预设）时，{names} 不起作用"
        );
    }
    let mut tasks = if directory {
        discover(input_path, output.as_deref().unwrap(), cli, &cfg)?
    } else if batch {
        parse_urls(input_path, output.as_deref().unwrap())?
    } else {
        vec![Task {
            source: input.into(),
            display: input.into(),
            report_key: if is_url(input) {
                input.into()
            } else {
                input_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            },
            output: output.clone(),
            filename: None,
            reserved_stem: None,
            source_file: None,
        }]
    };
    if mode == RunMode::UrlList && tasks.is_empty() {
        return Err((
            1,
            format!("No valid URLs found in {}.", input_path.display()),
        ));
    }
    if cli.dry_run {
        if !is_url(input) && !input_path.exists() {
            return Err((1, format!("Input does not exist: {input}")));
        }
        // One line per item: where it would be written, or why it would not.
        let previews = dry_run::plan(&tasks, &cfg);
        for (task, preview) in tasks.iter().zip(&previews) {
            println!("{} -> {}", task.display, preview.line());
        }
        if !cli.quiet {
            say_with(|lang| dry_run_summary(&tasks, &previews, input_path, lang));
        }
        return Ok(0);
    }
    if tasks.is_empty() && !cli.resume {
        if !cli.quiet {
            let input = input_path.display();
            if cli.globs.is_empty() {
                say!(
                    "No supported files or .urls lists found in {input}; nothing to convert.",
                    "在 {input} 中没有找到受支持的文件或 .urls 列表，无需转换。"
                );
            } else {
                say!(
                    "No supported files or .urls lists match --glob in {input}; nothing to convert.",
                    "在 {input} 中没有与 --glob 匹配的受支持文件或 .urls 列表，无需转换。"
                );
            }
        }
        if cli.json {
            emit_json(&[], None);
        }
        return Ok(0);
    }
    if cli.llm_batch {
        return provider_batch::submit(cli, input, cfg, tasks, output.as_deref().unwrap());
    }
    if batch && cli.resume {
        provider_batch::reject_pending(input_path, output.as_deref().unwrap(), &cfg)?;
    }
    let run_clock = Instant::now();
    let started_at = timestamp();
    let history_plan = crate::history::Plan::new(
        &cfg,
        output.as_deref(),
        mode,
        cli.preset.as_deref(),
        &started_at,
        !cli.quiet && !cli.json,
    );
    let report_plan = if let Some(output_dir) = output.clone() {
        report::plan(
            RunInfo {
                mode,
                input: input.into(),
                output_dir,
                started_at,
                log_file: logging::path(),
                options: ReportOptions::from_config(&cfg, cli.max_depth, &cli.globs),
            },
            cfg["output"]["report"].as_bool(),
            cfg["output"]["on_conflict"].as_str().unwrap_or("rename"),
            config::enabled(&cfg, "/output/allow_symlinks"),
        )
        .map_err(runtime)?
    } else {
        None
    };
    let llm_runtime = if config::enabled(&cfg, "/llm/enabled") {
        Some(
            markitai_core::LlmRuntime::new(
                usize::try_from(cfg["llm"]["concurrency"].as_u64().unwrap_or(1))
                    .map_err(runtime)?,
            )
            .map_err(runtime)?,
        )
    } else {
        None
    };
    let browser_runtime = markitai_core::BrowserRuntime::new(8).map_err(runtime)?;
    let context = ConvertContext {
        explicit_fetch_strategy: cli
            .strategy
            .as_deref()
            .filter(|strategy| *strategy != "auto"),
        llm_runtime: llm_runtime.as_ref(),
        browser_runtime: Some(&browser_runtime),
        stdout_assets: None,
    };
    if !batch {
        let mut task = tasks.remove(0);
        let store = task
            .output
            .is_none()
            .then(|| stdout_asset_store(&cfg))
            .flatten();
        let context = ConvertContext {
            stdout_assets: store.as_deref(),
            ..context
        };
        // A missing or unreadable local input fails before any output directory
        // is created for it; the converter reports a missing one itself.
        let local = mode == RunMode::SingleFile;
        let unreadable = local.then(|| unreadable_input(input)).flatten();
        let absent = local && std::fs::symlink_metadata(config::expand_home(input_path)).is_err();
        let claim = if unreadable.is_some() || absent || publishes_nothing(&task, &cfg) {
            None
        } else {
            batch_run::claim(&mut task, &cfg, None, None, &Default::default()).map_err(|error| {
                match output.as_deref() {
                    Some(directory) => runtime(format!(
                        "Cannot write to output directory {}: {error}",
                        directory.display()
                    )),
                    None => runtime(error),
                }
            })?
        };
        // A conversion that takes more than a moment shows a spinner line on a
        // terminal; it is gone before the result is printed.
        let spinner =
            progress::Spinner::start(progress::wanted(cli.quiet, cli.json), &task.report_key);
        let (record, result) = match unreadable {
            Some(error) => {
                let progress = begin_item(&task, &cfg);
                complete_item(&task, 0, &cfg, progress, Err(error.into()))
            }
            None => convert_item(
                &task,
                0,
                &cfg,
                context,
                claim
                    .as_ref()
                    .map(|claim| claim as &dyn markitai_core::output::Publication),
            ),
        };
        drop(spinner);
        let item = outcome(&record);
        let failed = result.is_err();
        let report_error = if record.status == ItemStatus::Completed {
            finish_report(
                report_plan.as_ref(),
                std::slice::from_ref(&record),
                run_clock,
                cli.verbose && !cli.quiet,
            )
            .err()
        } else {
            None
        };
        if let Some(error) = &report_error {
            eprintln!("Error: {error}");
        }
        if cli.json {
            emit_json(&[item], report_error.as_deref());
        } else {
            match result {
                Ok(result) => {
                    if task.output.is_none() {
                        let rendered = print_stdout(&result, &cfg).map_err(runtime)?;
                        // A persisted image is linked by now; the core warns
                        // about any it could not save.
                        if store.is_none()
                            && !cli.quiet
                            && unsaved_assets(&rendered, &cfg, i18n::Lang::En).is_some()
                        {
                            say_with(|lang| {
                                unsaved_assets(&rendered, &cfg, lang).unwrap_or_default()
                            });
                        }
                    }
                    if !cli.quiet {
                        for warning in warnings::present(&result.warnings, cli.verbose) {
                            eprintln!("Warning: {warning}");
                        }
                        if let Some(reason) = result.skip_reason.as_deref() {
                            // The name as the report lists it, not the path as typed.
                            say_with(|lang| skip_notice(&task.report_key, reason, lang));
                        } else if task.output.is_some()
                            && let Some(path) = &record.output
                        {
                            // The name can differ from the input's (rename on conflict).
                            let existing = renamed_because_of(path, &task.report_key);
                            let path = path.display();
                            match existing {
                                Some(existing) => say!(
                                    "Wrote {path} ({existing} already exists)",
                                    "已写入 {path}（{existing} 已存在）"
                                ),
                                None => say!("Wrote {path}", "已写入 {path}"),
                            }
                        }
                    }
                }
                Err(failure) => {
                    eprintln!("Error: {}", warnings::condense(&failure.to_string()));
                    if matches!(failure.error, markitai_core::Error::NotFound(_))
                        && let Some(command) = mistyped_subcommand(input)
                    {
                        say!(
                            "Hint: did you mean 'markitai {command}'?",
                            "Hint: 你是想运行 'markitai {command}' 吗？"
                        );
                    }
                    if matches!(failure.error, markitai_core::Error::NoModelConfigured) {
                        say!(
                            "Hint: set a provider API key such as OPENAI_API_KEY or ANTHROPIC_API_KEY (optionally with MODEL), or configure llm.model_list; `markitai init` saves a detected model. Run without --llm (or an LLM preset) to convert without a model.",
                            "Hint: 请设置供应商的 API key，例如 OPENAI_API_KEY 或 ANTHROPIC_API_KEY（可同时设置 MODEL），或配置 llm.model_list；`markitai init` 可保存检测到的模型。不加 --llm（或 LLM 预设）即可在不使用模型的情况下转换。"
                        );
                    }
                }
            }
        }
        // After the result lines, so "Recorded in history" follows "Wrote".
        if let Some(plan) = &history_plan {
            plan.record(std::slice::from_ref(&record));
        }
        return Ok(if failed || report_error.is_some() {
            1
        } else {
            0
        });
    }
    batch_run::run(
        cli,
        &cfg,
        tasks,
        BatchDestination {
            mode,
            output: output.as_deref().unwrap(),
            history: history_plan.as_ref(),
        },
        report_plan.as_ref(),
        run_clock,
        context,
    )
}

struct BatchDestination<'a> {
    mode: RunMode,
    output: &'a Path,
    history: Option<&'a crate::history::Plan>,
}

#[derive(Clone)]
struct Task {
    source: String,
    display: String,
    report_key: String,
    output: Option<PathBuf>,
    filename: Option<String>,
    reserved_stem: Option<String>,
    source_file: Option<String>,
}
/// Claim document stems before any worker starts. Disk conflicts use the configured
/// policy; another item in this run always gets a separate stem.
fn reserve_batch_names(tasks: &mut [Task], cfg: &Value) -> CliResult<()> {
    use std::collections::{HashMap, HashSet};
    let mut claimed = HashSet::<(PathBuf, String)>::new();
    let mut case_rules = HashMap::<PathBuf, bool>::new();
    let mut next_versions = HashMap::<(PathBuf, String), u64>::new();
    let mode = cfg["output"]["on_conflict"].as_str().unwrap_or("rename");
    // Durable creation of every missing output directory shares one media
    // fence per volume; the per-task preparation below then finds them.
    crate::output_claims::prepare_output_ancestors(
        tasks.iter().filter_map(|task| task.output.as_deref()),
        config::enabled(cfg, "/output/allow_symlinks"),
    )
    .map_err(runtime)?;
    for task in tasks {
        let Some(directory) = task.output.as_deref() else {
            continue;
        };
        // Name probing must not create unsynchronized output ancestors that
        // later claim acquisition would mistake for an established directory.
        let directory = crate::output_claims::prepare_namespace_parent(
            directory,
            config::enabled(cfg, "/output/allow_symlinks"),
        )
        .map_err(runtime)?;
        let folds = *case_rules.entry(directory.clone()).or_insert_with(|| {
            tempfile::Builder::new()
                .prefix(".MarkitaiCaseProbe-")
                .tempfile_in(&directory)
                .map(|probe| {
                    let lower = probe
                        .path()
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .to_lowercase();
                    directory.join(lower).exists()
                })
                .unwrap_or(true)
        });
        let stem = if let Some(name) = &task.filename {
            name.strip_suffix(".md").unwrap_or(name).to_owned()
        } else if is_url(&task.source) {
            markitai_core::output::url_name(&task.source)
        } else {
            Path::new(&task.source)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        };
        let key = |stem: &str| {
            (
                directory.clone(),
                if folds {
                    stem.to_lowercase()
                } else {
                    stem.to_owned()
                },
            )
        };
        let occupied = |stem: &str| {
            [".md", ".llm.md"].iter().any(|suffix| {
                std::fs::symlink_metadata(directory.join(format!("{stem}{suffix}"))).is_ok()
            })
        };
        let members = |stem: &str| [key(&format!("{stem}.md")), key(&format!("{stem}.llm.md"))];
        let duplicate = members(&stem).iter().any(|member| claimed.contains(member));
        let mut resolved = stem.clone();
        if !duplicate && mode == "skip" && occupied(&stem) {
            // No new result is claimed when this item will reuse an existing file.
            task.reserved_stem = Some(stem);
            continue;
        }
        if duplicate || (mode == "rename" && occupied(&stem)) {
            let mut version = *next_versions.entry(key(&stem)).or_insert(2);
            loop {
                resolved = format!("{stem}.v{version}");
                if !members(&resolved)
                    .iter()
                    .any(|member| claimed.contains(member))
                    && !occupied(&resolved)
                {
                    next_versions.insert(key(&stem), version.saturating_add(1));
                    break;
                }
                version = version
                    .checked_add(1)
                    .ok_or_else(|| runtime("Output version counter exhausted"))?;
            }
        }
        claimed.extend(members(&resolved));
        task.reserved_stem = Some(resolved);
    }
    Ok(())
}

fn task_config(task: &Task, cfg: &Value) -> Value {
    let mut cfg = cfg.clone();
    if let Some(name) = &task.reserved_stem {
        cfg["output"]["reserved_stem"] = json!(name);
        if cfg["output"]["filename"].is_string() {
            cfg["output"]["filename"] = json!(format!("{name}.md"));
        }
    } else if let Some(name) = &task.filename {
        cfg["output"]["reserved_stem"] = json!(name.strip_suffix(".md").unwrap_or(name));
    }
    cfg
}

/// A local image with neither OCR nor a model can only end as a skip ("no text
/// to extract") or as an error ("not a valid image") before anything is
/// published. It therefore needs no output claim: no ownership files, no
/// output directory, no recovery receipt for a result that never exists.
fn publishes_nothing(task: &Task, cfg: &Value) -> bool {
    !is_url(&task.source)
        && Path::new(&task.source)
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(markitai_core::is_image_extension)
        && !config::enabled(cfg, "/llm/enabled")
        && !config::enabled(cfg, "/ocr/enabled")
}

fn image_only_result(
    task: &Task,
    result: markitai_core::DetailedResult<ConversionOutput>,
) -> markitai_core::DetailedResult<ConversionOutput> {
    result.or_else(|failure| {
        if matches!(failure.error, markitai_core::Error::ImageOnly(_)) {
            let mut result = ConversionOutput::default();
            result.source = task.source.clone();
            result.skip_reason = Some("image_only".into());
            result.usage = failure.usage;
            Ok(result)
        } else {
            Err(failure)
        }
    })
}

fn convert_task(
    task: &Task,
    cfg: &Value,
    context: ConvertContext<'_>,
    publication: Option<&dyn markitai_core::output::Publication>,
) -> markitai_core::DetailedResult<ConversionOutput> {
    image_only_result(
        task,
        markitai_core::convert_with_publication_detailed(
            &task.source,
            ConvertOptions {
                output_dir: task.output.clone(),
                config: Some(task_config(task, cfg)),
                ..Default::default()
            },
            context,
            publication,
        ),
    )
}
fn timestamp() -> String {
    Local::now().to_rfc3339_opts(SecondsFormat::Micros, false)
}

struct ItemProgress {
    clock: Instant,
    started_at: String,
    history_enabled: bool,
    history_eligible: bool,
}

fn begin_item(task: &Task, cfg: &Value) -> ItemProgress {
    let clock = Instant::now();
    let started_at = timestamp();
    let history_enabled = task.output.is_some() && config::enabled(cfg, "/history/record");
    let history_eligible = history_enabled && crate::history::eligible(&task.source);
    logging::event(logging::Level::Info, format!("Converting {}", task.display));
    ItemProgress {
        clock,
        started_at,
        history_enabled,
        history_eligible,
    }
}

// The batch coordinator owns this provisional value until publication is durable.
// Neither preparation nor transfer emits a completed record or log entry.
struct PreparedItem {
    progress: ItemProgress,
    conversion: markitai_core::PreparedConversion,
}

fn prepare_item(
    task: &Task,
    cfg: &Value,
    context: ConvertContext<'_>,
    publication: &dyn markitai_core::output::Publication,
) -> PreparedItem {
    let progress = begin_item(task, cfg);
    let conversion = markitai_core::prepare_with_publication(
        &task.source,
        ConvertOptions {
            output_dir: task.output.clone(),
            config: Some(task_config(task, cfg)),
            ..Default::default()
        },
        context,
        publication,
    );
    PreparedItem {
        progress,
        conversion,
    }
}

fn convert_item(
    task: &Task,
    index: usize,
    cfg: &Value,
    context: ConvertContext<'_>,
    publication: Option<&dyn markitai_core::output::Publication>,
) -> (RunItem, markitai_core::DetailedResult<ConversionOutput>) {
    let progress = begin_item(task, cfg);
    let result = convert_task(task, cfg, context, publication);
    complete_item(task, index, cfg, progress, result)
}

fn complete_item(
    task: &Task,
    index: usize,
    cfg: &Value,
    progress: ItemProgress,
    result: markitai_core::DetailedResult<ConversionOutput>,
) -> (RunItem, markitai_core::DetailedResult<ConversionOutput>) {
    let mut record = recorded(task, index, progress.clock, progress.started_at, &result);
    record.history_eligible = progress.history_eligible;
    match &result {
        Ok(output) => {
            logging::event(
                logging::Level::Info,
                format!(
                    "{} {}",
                    if output.skip_reason.is_some() {
                        "Skipped"
                    } else {
                        "Completed"
                    },
                    task.display
                ),
            );
            for warning in &output.warnings {
                logging::event(
                    logging::Level::Warning,
                    format!("{}: {warning}", task.display),
                );
            }
        }
        Err(error) => logging::event(logging::Level::Error, format!("{}: {error}", task.display)),
    }
    if progress.history_enabled && record.skip_reason.as_deref() == Some("exists") {
        record.history_output = task.output.as_ref().map(|directory| {
            let fallback = if is_url(&task.source) {
                markitai_core::output::url_name(&task.source)
            } else {
                Path::new(&task.source)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            };
            let stem = cfg["output"]["filename"]
                .as_str()
                .map(|name| name.strip_suffix(".md").unwrap_or(name))
                .or(task.reserved_stem.as_deref())
                .or_else(|| {
                    task.filename
                        .as_deref()
                        .map(|name| name.strip_suffix(".md").unwrap_or(name))
                })
                .unwrap_or(&fallback);
            config::expand_home(directory).join(format!("{stem}.md"))
        });
    }
    (record, result)
}

fn recorded(
    task: &Task,
    index: usize,
    clock: Instant,
    started_at: String,
    result: &Result<ConversionOutput, markitai_core::ConversionFailure>,
) -> RunItem {
    let mut record = RunItem {
        index,
        kind: if is_url(&task.source) {
            ItemKind::Url
        } else {
            ItemKind::File
        },
        display: task.display.clone(),
        report_key: task.report_key.clone(),
        source_file: task.source_file.clone(),
        status: ItemStatus::Failed,
        output: None,
        history_output: None,
        history_eligible: true,
        error: None,
        warnings: Vec::new(),
        skip_reason: None,
        started_at,
        completed_at: timestamp(),
        elapsed_s: clock.elapsed().as_secs_f64(),
        conversion_duration_s: None,
        images: 0,
        screenshots: 0,
        usage: ConversionUsage::default(),
        diagnostics: None,
        llm_cache_hit: false,
        fetch_cache_hit: false,
        fetch_strategy: None,
    };
    match &result {
        Ok(output) => {
            record.status = if output.skip_reason.is_some() {
                ItemStatus::Skipped
            } else {
                ItemStatus::Completed
            };
            record.output = output
                .llm_output_path
                .as_ref()
                .or(output.output_path.as_ref())
                .or_else(|| output.screenshots.first())
                .cloned();
            record.warnings = output.warnings.clone();
            record.skip_reason = output.skip_reason.clone();
            record.images = output.assets.len();
            record.screenshots = output.screenshots.len();
            record.conversion_duration_s = Some(output.duration);
            record.usage = ConversionUsage {
                cost_usd: output.usage.cost_usd,
                requests: output.usage.requests,
                input_tokens: output.usage.input_tokens,
                output_tokens: output.usage.output_tokens,
                by_model: output.usage.by_model.clone(),
            };
            record.diagnostics = crate::diagnostics::AttemptDiagnostics::completed(
                crate::diagnostics::Operation::Convert,
                output.usage.clone(),
            );
            record.llm_cache_hit = output.llm_cache_hit();
            record.fetch_cache_hit = output.fetch_cache_hit();
            if record.kind == ItemKind::Url {
                record.fetch_strategy = output.fetch_strategy().map(str::to_owned);
            }
        }
        Err(failure) => {
            let message = failure.to_string();
            record.error = Some(message.clone());
            record.diagnostics = crate::diagnostics::AttemptDiagnostics::failed(
                crate::diagnostics::Operation::Convert,
                message,
                failure.usage.clone(),
            );
        }
    }
    record
}

fn outcome(item: &RunItem) -> Value {
    let usage = if item.status == ItemStatus::Failed {
        item.diagnostics
            .as_ref()
            .map(|value| &value.last_attempt.usage)
            .unwrap_or(&item.usage)
    } else {
        &item.usage
    };
    let mut result = json!({
        "kind": if item.kind == ItemKind::Url { "url" } else { "file" },
        "source": item.display,
        "status": match item.status {
            ItemStatus::Completed => "completed",
            ItemStatus::Skipped => "skipped",
            ItemStatus::Failed => "failed",
            ItemStatus::Pending => "pending",
        },
        "output": item.output,
        "error": item.error,
        "warnings": item.warnings,
        "skip_reason": item.skip_reason,
        "images": item.images,
        "screenshots": item.screenshots,
        "cost_usd": round(usage.cost_usd, 1_000_000.0),
        "duration_s": item.conversion_duration_s.map(|duration| round(duration, 1000.0)),
        "cache_hit": item.llm_cache_hit,
        "fetch_cache_hit": item.fetch_cache_hit,
        "llm_cache_hit": item.llm_cache_hit,
        "fetch_strategy": item.fetch_strategy,
        "source_file": item.source_file,
        "llm_usage": usage.by_model,
    });
    if let Some(diagnostics) = &item.diagnostics {
        result["diagnostics"] = json!(diagnostics);
    }
    if let Some(pricing) = crate::pricing::Pricing::from_usage(usage) {
        result["pricing"] = json!(pricing);
    }
    result
}

fn finish_report(
    plan: Option<&report::ReportPlan>,
    items: &[RunItem],
    clock: Instant,
    show_path: bool,
) -> Result<(), String> {
    let Some(plan) = plan else {
        return Ok(());
    };
    let finished = RunFinished {
        updated_at: timestamp(),
        duration_s: clock.elapsed().as_secs_f64(),
    };
    let bytes = report::render(plan, items, &finished)?;
    match report::publish(plan, &bytes)? {
        crate::report_store::Publication::Written(path) => {
            if show_path {
                let path = path.display();
                say!("Report: {path}", "报告：{path}");
            }
        }
        crate::report_store::Publication::SkippedExisting(path) => {
            if show_path {
                let path = path.display();
                say!(
                    "Existing report preserved: {path}",
                    "已保留现有报告：{path}"
                );
            }
        }
    }
    Ok(())
}
fn round(value: f64, factor: f64) -> f64 {
    (value * factor).round() / factor
}
fn envelope(items: &[Value], error: Option<&str>) -> Value {
    let count = |status: &str| items.iter().filter(|i| i["status"] == status).count();
    // `Sum` for f64 starts at -0.0, which JSON would print as `-0.0` for no items.
    let total = |field: &str| {
        items
            .iter()
            .filter_map(|i| i[field].as_f64())
            .fold(0.0, |sum, value| sum + value)
    };
    let mut value = json!({"version":"1.0","ok":count("failed")==0 && count("pending")==0 && error.is_none(),"error":error,"batch":null,"items":items,"totals":{"total":items.len(),"completed":count("completed"),"failed":count("failed"),"skipped":count("skipped"),"pending":count("pending"),"cost_usd":round(total("cost_usd"),1_000_000.0),"duration_s":round(total("duration_s"),1000.0)}});
    let pricing = items
        .iter()
        .filter_map(|item| item.get("pricing"))
        .filter_map(|pricing| {
            serde_json::from_value::<crate::pricing::Pricing>(pricing.clone()).ok()
        })
        .collect::<Vec<_>>();
    if let Some(pricing) = crate::pricing::Pricing::aggregate(&pricing) {
        value["totals"]["pricing"] = json!(pricing);
    }
    value
}
fn emit_json(items: &[Value], error: Option<&str>) {
    println!("{}", json_output::render(&envelope(items, error)));
}
/// Write the document to stdout and return what was rendered. A failed write
/// is not returned: it is settled with the rest of the run (`settle_stdout`).
fn print_stdout(result: &ConversionOutput, cfg: &Value) -> io::Result<String> {
    if result.skip_reason.is_some() {
        return Ok(String::new());
    }
    let rendered = markitai_core::output::content(result, cfg, result.llm_markdown.is_some())
        .map_err(io::Error::other)?;
    with_stdout(|stdout| {
        stdout.write_all(rendered.as_bytes())?;
        if !rendered.ends_with('\n') {
            stdout.write_all(b"\n")?;
        }
        Ok(())
    });
    Ok(rendered)
}

/// The image store for a document printed to stdout, unless
/// `image.stdout_persist` is off. The default `~/.markitai/assets` follows
/// `MARKITAI_HOME`; another configured directory keeps its meaning.
fn stdout_asset_store(cfg: &Value) -> Option<PathBuf> {
    config::enabled(cfg, "/image/stdout_persist").then(|| {
        config::state_path(Path::new(
            cfg["image"]["stdout_persist_dir"]
                .as_str()
                .unwrap_or("~/.markitai/assets"),
        ))
    })
}

/// With `image.stdout_persist` off, stdout mode writes no files, yet
/// extracted images and page captures keep their output-directory
/// references. Say so instead of leaving links that silently point nowhere.
fn unsaved_assets(markdown: &str, cfg: &Value, lang: i18n::Lang) -> Option<String> {
    let profile = cfg["output"]["profile"].as_str();
    let assets = if matches!(profile, Some("rag" | "obsidian")) {
        "assets/"
    } else {
        ".markitai/assets/"
    };
    let mut targets = std::collections::BTreeSet::new();
    for prefix in [assets, ".markitai/screenshots/"] {
        for opener in ["](", "[["] {
            for (index, _) in markdown.match_indices(opener) {
                let rest = &markdown[index + opener.len()..];
                if let Some(target) = rest.strip_prefix(prefix) {
                    let end = target.find([')', ']', '|', ' ']).unwrap_or(target.len());
                    targets.insert((prefix, &target[..end]));
                }
            }
        }
    }
    let count = targets.len();
    (count > 0).then(|| {
        let verb = if count == 1 {
            "reference points"
        } else {
            "references point"
        };
        text!(
            lang => "Warning: {count} image {verb} to files that stdout mode does not write because image.stdout_persist is false; use -o DIR to keep images, or set image.stdout_persist to true",
            "Warning: 有 {count} 个图片引用指向 stdout 模式不会写出的文件，因为 image.stdout_persist 为 false；用 -o DIR 保存图片，或把 image.stdout_persist 设为 true"
        )
    })
}
/// The name a result would have had but for `output.on_conflict = rename`:
/// `a.docx.v2.md` for source `a.docx` was written because `a.docx.md` exists.
/// Only a name that really is the source's name plus `.vN` counts.
pub(crate) fn renamed_because_of(written: &Path, source: &str) -> Option<String> {
    let name = written.file_name()?.to_str()?;
    let source = Path::new(source).file_name()?.to_str()?;
    let (stem, suffix) = name
        .strip_suffix(".llm.md")
        .map(|stem| (stem, ".llm.md"))
        .or_else(|| name.strip_suffix(".md").map(|stem| (stem, ".md")))?;
    let version = stem.strip_prefix(source)?.strip_prefix(".v")?;
    (version.parse::<u64>().is_ok() && !version.starts_with('+'))
        .then(|| format!("{source}{suffix}"))
}

/// The subcommand a missing input is most likely a typo of, within two edits.
fn mistyped_subcommand(input: &str) -> Option<&'static str> {
    const COMMANDS: [&str; 7] = ["config", "init", "doctor", "cache", "auth", "serve", "mcp"];
    if input.len() < 3 || !input.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return None;
    }
    let input = input.to_ascii_lowercase();
    COMMANDS
        .into_iter()
        .map(|command| (edit_distance(&input, command), command))
        .filter(|&(distance, command)| distance <= 2 && distance < command.len())
        .min()
        .map(|(_, command)| command)
}

/// Levenshtein distance between two short ASCII words.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (diagonal + usize::from(ca != *cb))
                .min(above + 1)
                .min(row[j] + 1);
            diagonal = above;
        }
    }
    row[b.len()]
}

fn is_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

/// Why a single item produced no document, and what would change that.
fn skip_notice(display: &str, reason: &str, lang: i18n::Lang) -> String {
    match reason {
        "image_only" => text!(
            lang => "Skipped {display}: an image has no text to extract without --ocr or --llm. Use --ocr for local text recognition or --llm for a vision model.",
            "已跳过 {display}：图片没有文字可提取，需要 --ocr 或 --llm。用 --ocr 做本地文字识别，或用 --llm 调用视觉模型。"
        ),
        "exists" => text!(
            lang => "Skipped {display}: its output already exists and output.on_conflict is skip. Set it to rename or overwrite to convert again.",
            "已跳过 {display}：输出已存在，且 output.on_conflict 为 skip。设为 rename 或 overwrite 可重新转换。"
        ),
        other => text!(lang => "Skipped {display} ({other}).", "已跳过 {display}（{other}）。"),
    }
}

/// The preview's closing line; the listing itself stays on stdout.
fn dry_run_summary(
    tasks: &[Task],
    previews: &[dry_run::Preview],
    input: &Path,
    lang: i18n::Lang,
) -> String {
    if tasks.is_empty() {
        let input = input.display();
        return text!(
            lang => "Dry run: no supported files or URLs in {input}; nothing would be converted.",
            "预览：{input} 中没有受支持的文件或 URL，不会转换任何内容。"
        );
    }
    let (mut files, mut urls, mut skipped) = (0, 0, 0);
    for (task, preview) in tasks.iter().zip(previews) {
        if !preview.converts() {
            skipped += 1;
        } else if is_url(&task.source) {
            urls += 1;
        } else {
            files += 1;
        }
    }
    let mut parts = Vec::new();
    if files > 0 {
        let noun = if files == 1 { "file" } else { "files" };
        parts.push(text!(lang => "{files} {noun}", "{files} 个文件"));
    }
    if urls > 0 {
        let noun = if urls == 1 { "URL" } else { "URLs" };
        parts.push(text!(lang => "{urls} {noun}", "{urls} 个 URL"));
    }
    let parts = parts.join(match lang {
        i18n::Lang::En => " and ",
        i18n::Lang::Zh => "和 ",
    });
    match (parts.is_empty(), skipped) {
        (_, 0) => text!(
            lang => "Dry run: {parts} would be converted; nothing was written.",
            "预览：将转换 {parts}；未写出任何内容。"
        ),
        (true, _) => text!(
            lang => "Dry run: nothing would be converted, {skipped} skipped; nothing was written.",
            "预览：不会转换任何内容，将跳过 {skipped} 项；未写出任何内容。"
        ),
        (false, _) => text!(
            lang => "Dry run: {parts} would be converted and {skipped} skipped; nothing was written.",
            "预览：将转换 {parts}，将跳过 {skipped} 项；未写出任何内容。"
        ),
    }
}

/// Prints one stderr line built per language. The file log keeps the English
/// line, so log entries read the same in every language.
fn say_with(build: impl Fn(i18n::Lang) -> String) {
    match i18n::lang() {
        i18n::Lang::En => eprintln!("{}", build(i18n::Lang::En)),
        i18n::Lang::Zh => logging::diagnostic_as(
            format_args!("{}", build(i18n::Lang::En)),
            format_args!("{}", build(i18n::Lang::Zh)),
        ),
    }
}

/// Each item's warnings and error on stderr, in item order. A warning that
/// several items share is written once, where it first occurs, naming them.
/// `verbose` also shows the notes that only explain how a page was read.
fn print_item_diagnostics(records: &[RunItem], quiet: bool, verbose: bool) {
    for line in item_diagnostics(records, quiet, verbose) {
        eprintln!("{line}");
    }
}

fn item_diagnostics(records: &[RunItem], quiet: bool, verbose: bool) -> Vec<String> {
    use std::collections::{HashMap, HashSet};
    // What the terminal shows of each item's warnings; reports and `--json`
    // keep them all.
    let shown: Vec<Vec<String>> = records
        .iter()
        .map(|record| {
            if quiet {
                Vec::new()
            } else {
                warnings::present(&record.warnings, verbose)
            }
        })
        .collect();
    let mut sharing = HashMap::<&str, Vec<&str>>::new();
    for (record, warnings) in records.iter().zip(&shown) {
        let mut seen = HashSet::new();
        for warning in warnings {
            if seen.insert(warning.as_str()) {
                sharing.entry(warning).or_default().push(&record.display);
            }
        }
    }
    let (mut lines, mut written) = (Vec::new(), HashSet::new());
    for (record, warnings) in records.iter().zip(&shown) {
        for warning in warnings {
            match sharing[warning.as_str()].as_slice() {
                [_] => lines.push(format!("Warning: {}: {warning}", record.display)),
                items if written.insert(warning.as_str()) => {
                    lines.push(format!("Warning: {}: {warning}", shared_names(items)))
                }
                _ => {}
            }
        }
        if let Some(error) = &record.error {
            lines.push(format!(
                "Error: {}: {}",
                record.display,
                warnings::condense(error)
            ));
        }
    }
    lines
}

/// Up to three item names, or the first two and how many more.
fn shared_names(items: &[&str]) -> String {
    if items.len() <= 3 {
        items.join(", ")
    } else {
        format!(
            "{} and {} more files",
            items[..2].join(", "),
            items.len() - 2
        )
    }
}

/// A batch's closing lines on stderr, in the terminal language. `unprocessed`
/// names the items an interruption left unstarted and `resumed` counts what an
/// earlier run of a resumed batch had finished.
fn print_batch_summary(
    records: &[RunItem],
    unprocessed: &[&str],
    resumed: report::Resumed,
    verbose: bool,
    elapsed: std::time::Duration,
    output: &Path,
) {
    let summary = |lang| {
        report::batch_summary(
            records,
            unprocessed,
            resumed,
            verbose,
            elapsed,
            output,
            lang,
        )
    };
    let english = summary(i18n::Lang::En);
    match i18n::lang() {
        i18n::Lang::En => {
            for line in &english {
                eprintln!("{line}");
            }
        }
        i18n::Lang::Zh => {
            let chinese = summary(i18n::Lang::Zh);
            for (english, chinese) in english.iter().zip(&chinese) {
                logging::diagnostic_as(format_args!("{english}"), format_args!("{chinese}"));
            }
        }
    }
}

/// A single local input whose bytes cannot be read, named with its path.
fn unreadable_input(input: &str) -> Option<markitai_core::Error> {
    let path = config::expand_home(Path::new(input));
    // Symlinks, directories and special files keep the converter's own checks.
    if !std::fs::symlink_metadata(&path).is_ok_and(|meta| meta.is_file()) {
        return None;
    }
    match std::fs::File::open(&path) {
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => Some(
            markitai_core::Error::InvalidInput(format!("Cannot read {input}: {error}")),
        ),
        _ => None,
    }
}

/// Explain an unusable output location before claim, state or report storage
/// reports it in its own terms. Advisory only: publication still performs
/// its own checks, so a later change of the directory still fails safely.
fn output_location_problem(directory: &Path) -> Option<String> {
    let directory = config::expand_home(directory);
    let mut existing = directory.as_path();
    loop {
        match std::fs::metadata(existing) {
            Ok(meta) if meta.is_dir() => break,
            Ok(_) if existing == directory => {
                return Some(format!(
                    "Output path {} exists and is not a directory; pass a directory with -o (a single input may also name a .md file)",
                    directory.display()
                ));
            }
            Ok(_) => {
                return Some(format!(
                    "Cannot create output directory {}: {} is not a directory",
                    directory.display(),
                    existing.display()
                ));
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
                ) =>
            {
                let parent = match existing.parent() {
                    Some(parent) if parent.as_os_str().is_empty() => Path::new("."),
                    Some(parent) => parent,
                    None => return None,
                };
                if parent == existing {
                    return None;
                }
                existing = parent;
            }
            Err(error) => {
                return Some(format!(
                    "Cannot use output directory {}: {error}",
                    directory.display()
                ));
            }
        }
    }
    if writable(existing) {
        None
    } else if existing == directory {
        Some(format!(
            "Output directory {} is not writable",
            directory.display()
        ))
    } else {
        Some(format!(
            "Cannot create output directory {}: {} is not writable",
            directory.display(),
            existing.display()
        ))
    }
}

#[cfg(unix)]
fn writable(directory: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(path) = std::ffi::CString::new(directory.as_os_str().as_bytes()) else {
        return true;
    };
    // SAFETY: `path` is a NUL-terminated string that outlives the call.
    unsafe { libc::access(path.as_ptr(), libc::W_OK | libc::X_OK) == 0 }
}

#[cfg(not(unix))]
fn writable(directory: &Path) -> bool {
    // ACLs, read-only shares and controlled-folder access cannot be inferred
    // from a Windows readonly attribute. A private empty probe answers the
    // same question as publication and is removed on drop.
    tempfile::Builder::new()
        .prefix(".markitai-write-probe-")
        .tempfile_in(directory)
        .is_ok()
}

fn discover(input: &Path, output: &Path, cli: &Cli, cfg: &Value) -> CliResult<Vec<Task>> {
    use globset::{GlobBuilder, GlobSetBuilder};
    let mut positive = GlobSetBuilder::new();
    let mut negative = GlobSetBuilder::new();
    let mut has_positive = false;
    for pattern in cli
        .globs
        .iter()
        .map(|pattern| pattern.trim())
        .filter(|pattern| !pattern.is_empty())
    {
        let (exclude, pattern) = pattern
            .strip_prefix('!')
            .map(|p| (true, p))
            .unwrap_or((false, pattern));
        let glob = GlobBuilder::new(pattern)
            .literal_separator(true)
            .build()
            .map_err(|e| (2, format!("Invalid glob: {e}")))?;
        if exclude {
            negative.add(glob);
        } else {
            positive.add(glob);
            has_positive = true;
        }
    }
    let positive = positive.build().map_err(runtime)?;
    let negative = negative.build().map_err(runtime)?;
    // Version-control and dependency folders, dot-files and Office lock files
    // (`~$name.docx`) are not documents. A positive glob that spells such a
    // name out (`.github/**`, `**/node_modules/**`) asks for them.
    let spelled: Vec<&str> = cli
        .globs
        .iter()
        .map(|pattern| pattern.trim())
        .filter(|pattern| !pattern.starts_with('!'))
        .flat_map(|pattern| pattern.split('/'))
        .collect();
    let hidden_wanted = spelled
        .iter()
        .any(|part| part.starts_with('.') && !matches!(*part, "." | ".."));
    let dependencies_wanted = spelled.contains(&"node_modules");
    let depth = cli
        .max_depth
        .or_else(|| cfg["batch"]["scan_max_depth"].as_u64().map(|n| n as usize))
        .unwrap_or(5);
    let max_files = cfg["batch"]["scan_max_files"].as_u64().unwrap_or(10000) as usize;
    let absolute_output = absolute(output);
    let absolute_input = absolute(input);
    let follow = config::enabled(cfg, "/output/allow_symlinks");
    let mut tasks = Vec::new();
    let mut entries = walkdir::WalkDir::new(input)
        .follow_links(follow)
        .max_depth(depth.saturating_add(1))
        .into_iter();
    while let Some(entry) = entries.next() {
        let entry = entry.map_err(runtime)?;
        if entry.depth() == 0 {
            continue;
        }
        let path = entry.path();
        let directory = entry.file_type().is_dir();
        let name = entry.file_name().to_string_lossy();
        if ((name.starts_with('.') || name.starts_with("~$")) && !hidden_wanted)
            || (name == "node_modules" && !dependencies_wanted)
        {
            if directory {
                entries.skip_current_dir();
            }
            continue;
        }
        if entry.file_name() == ".markitai"
            || (directory && absolute_output != absolute_input && absolute(path) == absolute_output)
        {
            if directory {
                entries.skip_current_dir();
            }
            continue;
        }
        let package = directory && markitai_core::formats::is_numbers_package_path(path);
        if package {
            // Even an excluded or malformed package is one input. Never discover
            // its internal files as independent documents or URL lists.
            entries.skip_current_dir();
        } else if !entry.file_type().is_file() {
            continue;
        }
        let relative = path.strip_prefix(input).map_err(runtime)?;
        if (has_positive && !positive.is_match(relative)) || negative.is_match(relative) {
            continue;
        }
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let out = output.join(relative.parent().unwrap_or(Path::new("")));
        if ext == "urls" {
            tasks.extend(parse_urls(path, &out)?);
        } else if markitai_core::formats::supports_extension(&ext)
            || markitai_core::is_image_extension(&ext)
        {
            tasks.push(Task {
                source: path.to_string_lossy().into_owned(),
                display: relative.to_string_lossy().replace('\\', "/"),
                report_key: relative.to_string_lossy().replace('\\', "/"),
                output: Some(out),
                filename: None,
                reserved_stem: None,
                source_file: None,
            });
        }
        if tasks.len() > max_files {
            return Err((
                1,
                format!(
                    "Batch exceeds scan_max_files ({max_files}); narrow --glob or increase the configuration limit"
                ),
            ));
        }
    }
    crate::sort::by(&mut tasks, |a, b| a.display.cmp(&b.display));
    let mut url_keys = std::collections::HashSet::new();
    tasks.retain(|task| !is_url(&task.source) || url_keys.insert(task.report_key.clone()));
    Ok(tasks)
}
fn absolute(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| {
        if path.is_absolute() {
            path.to_owned()
        } else {
            std::env::current_dir().unwrap_or_default().join(path)
        }
    })
}
/// Where an entry of a `.urls` list was found; the split itself is the core's.
struct Place(markitai_core::url_list::Place);

impl From<markitai_core::url_list::Place> for Place {
    fn from(place: markitai_core::url_list::Place) -> Self {
        Self(place)
    }
}

impl Place {
    /// The place named in English and in Chinese.
    fn names(self) -> (String, String) {
        match self.0 {
            markitai_core::url_list::Place::Entry(number) => {
                (format!("entry {number}"), format!("第 {number} 项"))
            }
            markitai_core::url_list::Place::Line(number) => {
                (format!("line {number}"), format!("第 {number} 行"))
            }
        }
    }
}

fn parse_urls(path: &Path, output: &Path) -> CliResult<Vec<Task>> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| runtime(format!("Cannot read {}: {error}", path.display())))?;
    // The split lives in the core, which the workbench reads uploads with too.
    let parsed = markitai_core::url_list::parse(&text).map_err(|error| {
        runtime(format!(
            "Cannot parse {} as a JSON URL list: {error}",
            path.display()
        ))
    })?;
    for place in parsed.skipped {
        let (location, place) = Place::from(place).names();
        let path = path.display();
        say!(
            "Warning: skipping {location} in {path}: expected a URL string or an object with \"url\"",
            "Warning: 跳过 {path} 的{place}：应为 URL 字符串，或带有 \"url\" 字段的对象"
        );
    }
    let mut tasks = Vec::new();
    for entry in parsed.entries {
        let (location, place) = Place::from(entry.place).names();
        let shown = path.display();
        let url = entry.url.trim();
        if !is_url(url) {
            say!(
                "Warning: skipping {location} in {shown}: not an HTTP(S) URL",
                "Warning: 跳过 {shown} 的{place}：不是 HTTP(S) URL"
            );
            continue;
        }
        let name = entry
            .output_name
            .map(|name| name.trim().to_owned())
            .filter(|name| !name.is_empty());
        let report_key = name
            .as_ref()
            .map(|name| format!("{url} {name}"))
            .unwrap_or_else(|| url.to_string());
        let filename = match name {
            Some(name) => {
                if name.contains(['/', '\\']) || name == "." || name == ".." {
                    return Err((
                        1,
                        "URL output_name must be a filename without directory components".into(),
                    ));
                }
                Some(if name.ends_with(".md") {
                    name
                } else {
                    format!("{name}.md")
                })
            }
            None => None,
        };
        if tasks.iter().any(|t: &Task| t.report_key == report_key) {
            continue;
        }
        tasks.push(Task {
            source: url.into(),
            display: url.into(),
            report_key,
            output: Some(output.to_owned()),
            filename,
            reserved_stem: None,
            source_file: Some(path.to_string_lossy().into_owned()),
        });
    }
    Ok(tasks)
}

fn selected_config(cli: &Cli) -> Option<PathBuf> {
    config::selected_path(cli.config.as_deref())
}
fn subcommand(cli: &Cli, command: &Command, overrides: Option<Value>) -> CliResult<i32> {
    match command {
        Command::Config { command } => match command {
            ConfigCommand::Path => {
                if let Some(path) = selected_config(cli) {
                    println!("{}", path.display());
                } else {
                    println!(
                        "{}",
                        text!(
                            "No configuration file found; using built-in defaults. Create one with `markitai init`.",
                            "未找到配置文件，正在使用内建默认值。可运行 `markitai init` 创建。"
                        )
                    );
                    // Where a file would be looked for, in the order it is used.
                    let home = config::home().join("config.json");
                    let home = home.display();
                    println!(
                        "{}",
                        text!(
                            "Searched in this order: -c FILE, the MARKITAI_CONFIG variable, ./markitai.json, {home}. `markitai config set KEY VALUE` creates {home}.",
                            "查找顺序：-c FILE、环境变量 MARKITAI_CONFIG、./markitai.json、{home}。`markitai config set KEY VALUE` 会创建 {home}。"
                        )
                    );
                }
            }
            ConfigCommand::Validate { config_file } => {
                if let Some(path) = config_file
                    && !path.exists()
                {
                    return Err((
                        2,
                        format!("Configuration file does not exist: {}", path.display()),
                    ));
                }
                let raw = config::raw(config_file.as_deref().or(cli.config.as_deref()), overrides)
                    .map_err(runtime)?;
                config::validate(&raw).map_err(runtime)?;
                for warning in config::warnings(&raw) {
                    let key = warning.key;
                    let reason = warning.reason;
                    let chinese = match key {
                        "batch.heavy_task_limit" => {
                            "没有运行效果；请用 batch.concurrency 限制文件转换并发"
                        }
                        "office.macos_fallback" => {
                            "没有运行效果；Office 转换不会自动操作 macOS 应用"
                        }
                        "image.stdout_fetch_external" => {
                            "没有运行效果；本构建不支持终端内联图片显示"
                        }
                        _ => reason,
                    };
                    say!("Warning: {key} {reason}", "Warning: {key} {chinese}");
                }
                println!("{}", text!("Configuration is valid", "配置有效"));
            }
            ConfigCommand::List {
                format,
                show_secrets,
            } => {
                let cfg = config::load(cli.config.as_deref(), overrides).map_err(runtime)?;
                let mut cfg = config::display_value(&cfg, None).map_err(runtime)?;
                if !show_secrets {
                    redact(&mut cfg);
                }
                match format.to_lowercase().as_str() {
                    "yaml" => print!("{}", serde_yaml::to_string(&cfg).map_err(runtime)?),
                    "table" => print_table("", &cfg),
                    _ => println!("{}", serde_json::to_string_pretty(&cfg).map_err(runtime)?),
                }
            }
            ConfigCommand::Get { key, show_secrets } => {
                let cfg = config::load(cli.config.as_deref(), overrides).map_err(runtime)?;
                let pointer = key_pointer(key)?;
                let value = cfg.pointer(&pointer).ok_or_else(|| {
                    (
                        1,
                        format!(
                            "Unknown configuration key: {key}. Run `markitai config list -f table` to see every key."
                        ),
                    )
                })?;
                if value.is_null() {
                    println!("null");
                    return Ok(0);
                }
                let value = config::display_value(&cfg, Some(key)).map_err(runtime)?;
                let visible = if *show_secrets {
                    value
                } else {
                    config::redact_for_key(key, &value)
                };
                let value = &visible;
                match value {
                    Value::String(s) => println!("{s}"),
                    _ => println!("{}", serde_json::to_string_pretty(value).map_err(runtime)?),
                }
            }
            ConfigCommand::Set {
                key,
                value,
                show_secrets,
            } => {
                if overrides.is_some() {
                    return Err((2, "--config-json overrides cannot be saved; drop --config-json to write a config file".into()));
                }
                let path =
                    selected_config(cli).unwrap_or_else(|| config::home().join("config.json"));
                let mut raw = match config::read_file(&path).map_err(runtime)? {
                    Some(bytes) => serde_json::from_slice::<Value>(&bytes).map_err(runtime)?,
                    None => json!({}),
                };
                if !raw.is_object() {
                    return Err((1, "Configuration must be a JSON object".into()));
                }
                let value = config::parse_cli_value(&raw, key, value).map_err(runtime)?;
                let value = config::set_value(&mut raw, key, value).map_err(runtime)?;
                write_config(&path, &raw)?;
                // Echo what `config get` shows: unset model fields are omitted
                // rather than displayed as redacted secrets.
                let visible = if value.is_null() {
                    value
                } else {
                    let shown = config::normalize(&raw)
                        .and_then(|cfg| config::display_value(&cfg, Some(key)))
                        .unwrap_or(value);
                    if *show_secrets {
                        shown
                    } else {
                        config::redact_for_key(key, &shown)
                    }
                };
                println!("{key} = {visible}");
                // stdout keeps the value for scripts; which file changed goes to stderr.
                let saved = path.display();
                say!("Saved to {saved}", "已保存到 {saved}");
            }
            ConfigCommand::Edit => {
                if overrides.is_some() {
                    return Err((2, "--config-json overrides cannot be saved; drop --config-json to edit a config file".into()));
                }
                let path =
                    selected_config(cli).unwrap_or_else(|| config::home().join("config.json"));
                interactive::edit(&path)?;
            }
        },
        Command::Init { yes, output, local } => {
            interactive::init(*yes, output.as_deref(), *local)?;
        }
        Command::Doctor {
            json: as_json,
            fix,
            suggest_extras,
        } => {
            if *as_json && *fix {
                return Err((2, "--json and --fix cannot be used together".into()));
            }
            if *suggest_extras {
                return Err(unsupported(
                    "Python extras recommendations (this build is native Rust)",
                ));
            }
            let cfg = config::load(cli.config.as_deref(), overrides).map_err(runtime)?;
            return doctor::run(&cfg, selected_config(cli).as_deref(), *as_json, *fix);
        }
        Command::Cache { command } => {
            let cfg = config::load(cli.config.as_deref(), overrides).map_err(runtime)?;
            return cache_command(command, &cfg);
        }
        Command::Auth { command } => return auth::run(command.as_ref()),
        Command::Serve {
            host,
            port,
            no_open,
            no_auth,
            allowed_host,
        } => {
            let source = crate::server::settings_source(cli.config.as_deref(), overrides.clone())
                .map_err(runtime)?;
            let cfg = crate::server::settings_config(&source).map_err(runtime)?;
            crate::server::run(
                cfg,
                source,
                crate::server::ServeOptions {
                    host: host.clone(),
                    port: *port,
                    no_open: *no_open,
                    no_auth: *no_auth,
                    allowed_host: allowed_host.clone(),
                },
            )
            .map_err(runtime)?;
        }
        Command::Mcp => crate::mcp::run(cli.config.clone(), overrides).map_err(runtime)?,
    }
    Ok(0)
}
fn cache_command(command: &CacheCommand, cfg: &Value) -> CliResult<i32> {
    match command {
        CacheCommand::Stats {
            json: as_json,
            verbose,
            limit,
        } => {
            let stats = markitai_core::llm_cache::stats(cfg, *verbose, *limit).map_err(runtime)?;
            let failed = ["cache", "fetch_cache"]
                .iter()
                .any(|name| stats[name].get("error").is_some());
            if *as_json {
                println!("{}", serde_json::to_string_pretty(&stats).map_err(runtime)?);
            } else {
                let enabled = &stats["enabled"];
                println!(
                    "{}",
                    match (i18n::lang(), enabled.as_bool()) {
                        (i18n::Lang::Zh, Some(true)) => "缓存：已启用".to_owned(),
                        (i18n::Lang::Zh, Some(false)) => "缓存：已禁用".to_owned(),
                        _ => text!("Cache enabled: {enabled}", "缓存已启用：{enabled}"),
                    }
                );
                let llm = cache_state(&stats["cache"]);
                println!("{}", text!("LLM cache: {llm}", "LLM 缓存：{llm}"));
                if *verbose && !stats["cache"].is_null() && stats["cache"].get("error").is_none() {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&stats["cache"]).map_err(runtime)?
                    );
                }
                let fetch = cache_state(&stats["fetch_cache"]);
                println!(
                    "{}",
                    text!("URL fetch cache: {fetch}", "URL 抓取缓存：{fetch}")
                );
            }
            Ok(i32::from(failed))
        }
        CacheCommand::Clear {
            yes,
            include_spa_domains,
        } => {
            let dir = config::state_path(Path::new(
                cfg["cache"]["global_dir"].as_str().unwrap_or("~/.markitai"),
            ));
            if !yes {
                let dir = dir.display();
                let question = if *include_spa_domains {
                    text!(
                        "Clear LLM + URL fetch caches + learned browser domains ({dir})? [y/N]: ",
                        "清理 LLM 与 URL 抓取缓存及已学习的浏览器域名（{dir}）？[y/N]："
                    )
                } else {
                    text!(
                        "Clear LLM + URL fetch caches ({dir})? [y/N]: ",
                        "清理 LLM 与 URL 抓取缓存（{dir}）？[y/N]："
                    )
                };
                print!("{question}");
                io::stdout().flush().map_err(runtime)?;
                let mut answer = String::new();
                let read = io::stdin().read_line(&mut answer).map_err(runtime)?;
                if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
                    println!("{}", text!("Aborted", "已取消"));
                    // No answer at all (a script without --yes) is a failure, as
                    // in the reference; an explicit "no" is not.
                    return Ok(if read == 0 { 1 } else { 0 });
                }
            }
            markitai_core::llm_cache::preflight_clear(cfg).map_err(runtime)?;
            markitai_core::fetch_cache::preflight_clear(cfg).map_err(runtime)?;
            if *include_spa_domains {
                markitai_core::spa_domains::preflight_clear(cfg).map_err(runtime)?;
            }
            let llm_count = markitai_core::llm_cache::clear(cfg).map_err(runtime)?;
            let fetch_count = markitai_core::fetch_cache::clear(cfg).map_err(|error| {
                runtime(text!(
                    "LLM cache cleared ({llm_count} entries); URL fetch cache clear failed: {error}",
                    "已清理 LLM 缓存（{llm_count} 条）；URL 抓取缓存清理失败：{error}"
                ))
            })?;
            let cleared = llm_count.saturating_add(fetch_count);
            let spa_count = if *include_spa_domains {
                Some(markitai_core::spa_domains::clear(cfg).map_err(|error| {
                    runtime(text!(
                        "LLM and URL fetch caches cleared ({cleared} entries); learned browser-domain clear failed: {error}",
                        "已清理 LLM 与 URL 抓取缓存（{cleared} 条）；已学习的浏览器域名清理失败：{error}"
                    ))
                })?)
            } else {
                None
            };
            let noun = if cleared == 1 { "entry" } else { "entries" };
            println!(
                "{}",
                text!("Cleared {cleared} cache {noun}", "已清理 {cleared} 条缓存")
            );
            if let Some(count) = spa_count {
                println!("{}", spa_cleared(count));
            }
            Ok(0)
        }
        CacheCommand::SpaDomains {
            json: as_json,
            clear,
        } => {
            if *clear {
                let count = markitai_core::spa_domains::clear(cfg).map_err(runtime)?;
                if *as_json {
                    println!("{}", json!({"cleared":count}));
                } else {
                    println!("{}", spa_cleared(count));
                }
            } else {
                let entries = markitai_core::spa_domains::list(cfg).map_err(runtime)?;
                if *as_json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&entries).map_err(runtime)?
                    );
                } else if entries.is_empty() {
                    println!(
                        "{}",
                        text!("No learned SPA domains.", "没有已学习的 SPA 域名。")
                    );
                } else {
                    println!(
                        "{}",
                        text!(
                            "Domain\tHits\tLearned at\tLast hit\tExpired",
                            "域名\t命中次数\t学习时间\t最近命中\t已过期"
                        )
                    );
                    for entry in entries {
                        println!(
                            "{}\t{}\t{}\t{}\t{}",
                            entry.domain,
                            entry.hits,
                            entry.learned_at,
                            entry.last_hit,
                            entry.expired
                        );
                    }
                }
            }
            Ok(0)
        }
    }
}
fn spa_cleared(count: u64) -> String {
    let noun = if count == 1 { "domain" } else { "domains" };
    text!(
        "Cleared {count} learned SPA {noun}",
        "已清理 {count} 个已学习的 SPA 域名"
    )
}

/// The size line of one store, or the reason it could not be read.
fn cache_state(stats: &Value) -> String {
    match stats.get("error") {
        Some(error) => error
            .as_str()
            .map_or_else(|| text!("unavailable", "不可用"), str::to_owned),
        None => cache_size(stats),
    }
}

/// "3 entries (1.2 MiB)"; the JSON form keeps exact byte counts.
fn cache_size(stats: &Value) -> String {
    let bytes = stats["size_bytes"].as_u64().unwrap_or(0);
    let size = if bytes < 1024 {
        format!("{bytes} B")
    } else {
        let mut value = bytes as f64 / 1024.0;
        let mut unit = "KiB";
        for next in ["MiB", "GiB", "TiB"] {
            if value < 1024.0 {
                break;
            }
            value /= 1024.0;
            unit = next;
        }
        format!("{value:.1} {unit}")
    };
    let count = stats["count"].as_u64().unwrap_or(0);
    let noun = if count == 1 { "entry" } else { "entries" };
    text!("{count} {noun} ({size})", "{count} 条（{size}）")
}

fn key_pointer(key: &str) -> CliResult<String> {
    config::key_pointer(key).map_err(runtime)
}
fn redact(value: &mut Value) {
    *value = config::redact(value);
}
fn print_table(prefix: &str, value: &Value) {
    if let Value::Object(map) = value {
        for (key, value) in map {
            let path = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{prefix}.{key}")
            };
            print_table(&path, value);
        }
    } else {
        println!("{prefix}\t{value}");
    }
}
fn write_config(path: &Path, value: &Value) -> CliResult<()> {
    // Configuration paths are user-selected; an existing symlink is preserved
    // and its target atomically updated, as the Python configuration manager did.
    let mut bytes = serde_json::to_vec_pretty(value).map_err(runtime)?;
    bytes.push(b'\n');
    config::write_file(path, &bytes).map_err(runtime)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_root_help_fits_40_columns_without_splitting_usage_tokens() {
        for long in [false, true] {
            let render = |mut command: clap::Command| {
                command = command.term_width(40).color(clap::ColorChoice::Never);
                if long {
                    command.render_long_help().to_string()
                } else {
                    command.render_help().to_string()
                }
            };
            let before = render(english_command());
            let after = render(english_command_at_width(40));
            assert!(before.contains("Usage: markitai [OPTIONS] [INPUT] [COMMAND]"));
            assert_eq!(after, before.replacen("Usage: ", "Usage:\n  ", 1));
            assert!(after.contains("Usage:\n  markitai [OPTIONS] [INPUT] [COMMAND]"));
            for line in after.lines() {
                assert!(line.chars().count() <= 40, "long={long}: {line}");
            }
        }
    }

    #[test]
    fn english_help_that_already_fits_keeps_its_exact_rendered_bytes() {
        for width in [43, 80, 120] {
            for long in [false, true] {
                let render = |mut command: clap::Command| {
                    command = command.term_width(width).color(clap::ColorChoice::Never);
                    if long {
                        command.render_long_help().to_string()
                    } else {
                        command.render_help().to_string()
                    }
                };
                assert_eq!(
                    render(english_command_at_width(width)),
                    render(english_command()),
                    "width={width}, long={long}"
                );
            }
        }
    }

    #[test]
    fn narrow_root_template_preserves_subcommand_help_and_usage_errors() {
        let render = |command: clap::Command, arguments: &[&str]| {
            command
                .term_width(40)
                .color(clap::ColorChoice::Never)
                .try_get_matches_from(arguments)
                .unwrap_err()
                .to_string()
        };
        for arguments in [
            vec!["markitai", "serve", "--help"],
            vec!["markitai", "config", "set", "--help"],
            vec!["markitai", "--unknown-option"],
        ] {
            assert_eq!(
                render(english_command_at_width(40), &arguments),
                render(english_command(), &arguments),
                "{arguments:?}"
            );
        }
    }
    #[test]
    fn a_warning_several_items_share_is_written_once_naming_them() {
        let item = |index: usize, warnings: &[&str], error: Option<&str>| RunItem {
            index,
            kind: report::ItemKind::File,
            display: format!("f{index}.docx"),
            report_key: format!("f{index}.docx"),
            source_file: None,
            status: ItemStatus::Completed,
            output: None,
            history_output: None,
            history_eligible: true,
            error: error.map(str::to_owned),
            warnings: warnings.iter().map(|&warning| warning.to_owned()).collect(),
            skip_reason: None,
            started_at: String::new(),
            completed_at: String::new(),
            elapsed_s: 0.0,
            conversion_duration_s: None,
            images: 0,
            screenshots: 0,
            usage: ConversionUsage::default(),
            diagnostics: None,
            llm_cache_hit: false,
            fetch_cache_hit: false,
            fetch_strategy: None,
        };
        let records = [
            item(0, &["own", "priced"], None),
            item(1, &["priced", "priced"], Some("broken")),
            item(2, &["pair"], None),
            item(3, &["priced", "pair"], None),
            item(4, &["priced"], None),
        ];
        assert_eq!(
            item_diagnostics(&records, false, false),
            [
                "Warning: f0.docx: own",
                "Warning: f0.docx, f1.docx and 2 more files: priced",
                "Error: f1.docx: broken",
                "Warning: f2.docx, f3.docx: pair",
            ]
        );
        assert_eq!(
            item_diagnostics(&records, true, false),
            ["Error: f1.docx: broken"]
        );
    }

    #[test]
    fn after_a_failed_write_later_writes_are_dropped_and_the_first_failure_stays() {
        let mut failure = None;
        let mut out = Vec::new();
        write_unless_failed(&mut failure, &mut out, |out| out.write_all(b"a"));
        assert!(failure.is_none());
        write_unless_failed(&mut failure, &mut out, |_| {
            Err(io::ErrorKind::BrokenPipe.into())
        });
        write_unless_failed(&mut failure, &mut out, |out| out.write_all(b"b"));
        write_unless_failed(&mut failure, &mut out, |_| {
            Err(io::Error::from_raw_os_error(28))
        });
        assert_eq!(out, b"a");
        assert_eq!(failure.unwrap().kind(), io::ErrorKind::BrokenPipe);
    }
    #[test]
    fn a_closed_pipe_keeps_the_status_and_any_other_write_failure_is_named() {
        let closed = io::Error::from(io::ErrorKind::BrokenPipe);
        for code in [0, 1, 10, 130] {
            assert_eq!(stdout_outcome(code, Some(&closed)), (code, None));
            assert_eq!(stdout_outcome(code, None), (code, None));
        }
        let full = io::Error::from_raw_os_error(28);
        let (code, message) = stdout_outcome(0, Some(&full));
        assert_eq!(code, 1);
        let message = message.unwrap();
        assert!(
            message.starts_with("Error: Cannot write to standard output: "),
            "{message}"
        );
        assert!(message.contains(&full.to_string()), "{message}");
        // A run that already failed keeps its own status.
        assert_eq!(stdout_outcome(10, Some(&full)).0, 10);
        assert_eq!(stdout_outcome(130, Some(&full)).0, 130);
    }
    #[test]
    fn envelope_keeps_failed_items_and_totals() {
        let items = vec![
            json!({"status":"completed","cost_usd":0.1234567,"duration_s":0.101}),
            json!({"status":"failed","cost_usd":0.0,"duration_s":null}),
        ];
        let result = envelope(&items, None);
        assert_eq!(result["version"], "1.0");
        assert_eq!(result["ok"], false);
        assert_eq!(result["totals"]["failed"], 1);
        assert_eq!(result["totals"]["cost_usd"], 0.123457);
        // No items must not serialize as negative zero.
        let empty = serde_json::to_string(&envelope(&[], Some("run failed"))).unwrap();
        assert!(empty.contains(r#""cost_usd":0.0"#), "{empty}");
        assert!(empty.contains(r#""duration_s":0.0"#), "{empty}");
    }
    #[test]
    fn stdout_documents_warn_about_unwritten_asset_references() {
        let cfg = config::defaults();
        let markdown = "![a](.markitai/assets/one.png) ![b](.markitai/assets/one.png)\n\
            ![c](.markitai/assets/two.jpg)\n`.markitai/assets/literal`\n\
            <!-- ![Page 1](.markitai/screenshots/doc.pdf.page0001.jpg) -->\n";
        let warning = unsaved_assets(markdown, &cfg, i18n::Lang::En).unwrap();
        assert!(
            warning.starts_with("Warning: 3 image references point to files that stdout mode does not write because image.stdout_persist is false"),
            "{warning}"
        );
        assert!(warning.contains("use -o DIR"));
        assert_eq!(
            unsaved_assets("![x](https://example.com/x.png)\n", &cfg, i18n::Lang::En),
            None
        );
        let mut obsidian = cfg.clone();
        obsidian["output"]["profile"] = json!("obsidian");
        let warning =
            unsaved_assets("![[assets/one.png|caption]]\n", &obsidian, i18n::Lang::En).unwrap();
        assert!(warning.starts_with("Warning: 1 image reference points to files"));
        assert_eq!(
            unsaved_assets("![[.markitai/assets/one.png]]\n", &obsidian, i18n::Lang::En),
            None
        );
    }

    #[test]
    fn notices_have_the_same_content_in_chinese_and_english() {
        let cfg = config::defaults();
        let markdown = "![a](.markitai/assets/one.png) ![c](.markitai/assets/two.jpg)\n";
        assert_eq!(
            unsaved_assets(markdown, &cfg, i18n::Lang::Zh).unwrap(),
            "Warning: 有 2 个图片引用指向 stdout 模式不会写出的文件，因为 image.stdout_persist 为 false；用 -o DIR 保存图片，或把 image.stdout_persist 设为 true"
        );
        assert_eq!(
            unsaved_assets("![a](.markitai/assets/one.png)\n", &cfg, i18n::Lang::Zh).unwrap(),
            "Warning: 有 1 个图片引用指向 stdout 模式不会写出的文件，因为 image.stdout_persist 为 false；用 -o DIR 保存图片，或把 image.stdout_persist 设为 true"
        );
        for (reason, english, chinese) in [
            (
                "image_only",
                "Skipped a.png: an image has no text to extract without --ocr or --llm. Use --ocr for local text recognition or --llm for a vision model.",
                "已跳过 a.png：图片没有文字可提取，需要 --ocr 或 --llm。用 --ocr 做本地文字识别，或用 --llm 调用视觉模型。",
            ),
            (
                "exists",
                "Skipped a.png: its output already exists and output.on_conflict is skip. Set it to rename or overwrite to convert again.",
                "已跳过 a.png：输出已存在，且 output.on_conflict 为 skip。设为 rename 或 overwrite 可重新转换。",
            ),
            ("weird", "Skipped a.png (weird).", "已跳过 a.png（weird）。"),
        ] {
            assert_eq!(skip_notice("a.png", reason, i18n::Lang::En), english);
            assert_eq!(skip_notice("a.png", reason, i18n::Lang::Zh), chinese);
        }
        // A rejected `.urls` entry is located in both languages.
        assert_eq!(
            Place::from(markitai_core::url_list::Place::Entry(3)).names(),
            ("entry 3".to_owned(), "第 3 项".to_owned())
        );
        assert_eq!(
            Place::from(markitai_core::url_list::Place::Line(12)).names(),
            ("line 12".to_owned(), "第 12 行".to_owned())
        );
    }

    #[test]
    fn stdout_images_are_stored_in_the_isolated_home_unless_turned_off() {
        let mut cfg = config::defaults();
        let store = stdout_asset_store(&cfg).unwrap();
        assert_eq!(store, config::state_path(Path::new("~/.markitai/assets")));
        cfg["image"]["stdout_persist_dir"] = json!("/srv/markitai-images");
        assert_eq!(
            stdout_asset_store(&cfg).unwrap(),
            Path::new("/srv/markitai-images")
        );
        cfg["image"]["stdout_persist"] = json!(false);
        assert_eq!(stdout_asset_store(&cfg), None);
    }
    #[test]
    fn numeric_limits_and_previews_state_their_rule_plainly() {
        assert_eq!(at_least_one("4"), Ok(4));
        assert_eq!(at_least_one("0").unwrap_err(), "must be at least 1");
        assert!(at_least_one("-1").is_err());
        let task = |source: &str| Task {
            source: source.into(),
            display: source.into(),
            report_key: source.into(),
            output: None,
            filename: None,
            reserved_stem: None,
            source_file: None,
        };
        use dry_run::Preview;
        let both = [task("a.txt"), task("https://example.com/")];
        let converted = [Preview::Stdout, Preview::Stdout];
        assert_eq!(
            dry_run_summary(&both, &converted, Path::new("in"), i18n::Lang::En),
            "Dry run: 1 file and 1 URL would be converted; nothing was written."
        );
        assert!(
            dry_run_summary(&[], &[], Path::new("in"), i18n::Lang::En)
                .contains("nothing would be converted")
        );
        assert_eq!(
            dry_run_summary(&both, &converted, Path::new("in"), i18n::Lang::Zh),
            "预览：将转换 1 个文件和 1 个 URL；未写出任何内容。"
        );
        assert_eq!(
            dry_run_summary(
                &[task("a.txt"), task("b.txt")],
                &converted,
                Path::new("in"),
                i18n::Lang::Zh
            ),
            "预览：将转换 2 个文件；未写出任何内容。"
        );
        assert_eq!(
            dry_run_summary(&[], &[], Path::new("in"), i18n::Lang::Zh),
            "预览：in 中没有受支持的文件或 URL，不会转换任何内容。"
        );
        // Items that would be skipped are counted apart from those converted.
        let mixed = [Preview::Stdout, Preview::Skip("image".into())];
        assert_eq!(
            dry_run_summary(&both, &mixed, Path::new("in"), i18n::Lang::En),
            "Dry run: 1 file would be converted and 1 skipped; nothing was written."
        );
        assert_eq!(
            dry_run_summary(&both, &mixed, Path::new("in"), i18n::Lang::Zh),
            "预览：将转换 1 个文件，将跳过 1 项；未写出任何内容。"
        );
        let skipped = [Preview::Skip("a".into()), Preview::Skip("b".into())];
        assert_eq!(
            dry_run_summary(&both, &skipped, Path::new("in"), i18n::Lang::En),
            "Dry run: nothing would be converted, 2 skipped; nothing was written."
        );
    }
    #[test]
    fn a_renamed_result_names_the_file_that_was_in_the_way() {
        let renamed = |written: &str, source: &str| renamed_because_of(Path::new(written), source);
        assert_eq!(
            renamed("out/a.docx.v2.md", "a.docx").as_deref(),
            Some("a.docx.md")
        );
        assert_eq!(
            renamed("out/a.docx.v13.llm.md", "dir/a.docx").as_deref(),
            Some("a.docx.llm.md")
        );
        // Not renamed: the plain name, a source that is itself called v2, a
        // version that is not a number.
        assert_eq!(renamed("out/a.docx.md", "a.docx"), None);
        assert_eq!(renamed("out/notes.v2.md", "notes.v2"), None);
        assert_eq!(renamed("out/a.docx.vx.md", "a.docx"), None);
        assert_eq!(renamed("out/b.docx.v2.md", "a.docx"), None);
    }

    #[cfg(windows)]
    #[test]
    fn writable_probe_checks_publication_and_leaves_no_probe_file() {
        let root = tempfile::tempdir().unwrap();
        assert!(writable(root.path()));
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
        let file = root.path().join("not-a-directory");
        std::fs::write(&file, b"keep").unwrap();
        assert!(!writable(&file));
        assert!(!writable(&root.path().join("missing")));
        assert_eq!(std::fs::read(&file).unwrap(), b"keep");
    }

    #[test]
    fn a_missing_input_close_to_a_subcommand_name_suggests_the_command() {
        for (typed, command) in [
            ("docter", "doctor"),
            ("conifg", "config"),
            ("cahce", "cache"),
            ("inti", "init"),
            ("Doctor", "doctor"),
            ("serv", "serve"),
        ] {
            assert_eq!(mistyped_subcommand(typed), Some(command), "{typed}");
        }
        // Paths, file names and unrelated words never get a suggestion.
        for typed in [
            "notes.txt",
            "dir/doctor",
            "report",
            "ab",
            "documents",
            "my file",
        ] {
            assert_eq!(mistyped_subcommand(typed), None, "{typed}");
        }
    }

    #[test]
    fn a_leading_tilde_names_the_home_directory_and_nothing_else_changes() {
        // The expansion reads HOME; this check only concerns what it leaves alone.
        for same in ["out", "./~", "~name/out", "a/~/b", "/abs/~"] {
            assert_eq!(home_relative(Path::new(same)).unwrap(), Path::new(same));
        }
        if let Some(home) = std::env::var_os("HOME") {
            let home = PathBuf::from(home);
            assert_eq!(home_relative(Path::new("~")).unwrap(), home);
            assert_eq!(
                home_relative(Path::new("~/out/dir")).unwrap(),
                home.join("out/dir")
            );
        }
    }

    #[test]
    fn credentials_remain_redacted_at_nested_paths() {
        let mut cfg = json!({"llm":{"model_list":[{"litellm_params":{"api_key":"secret","model":"test"}}]},"fetch":{"playwright":{"extra_http_headers":{"Authorization":"secret"}}}});
        redact(&mut cfg);
        assert_eq!(
            cfg["llm"]["model_list"][0]["litellm_params"]["api_key"],
            "[REDACTED]"
        );
        assert_eq!(
            cfg["fetch"]["playwright"]["extra_http_headers"]["Authorization"],
            "[REDACTED]"
        );
        assert_eq!(
            key_pointer("llm.model_list[0].model_name").unwrap(),
            "/llm/model_list/0/model_name"
        );
    }
    #[test]
    fn reservations_rename_same_run_collisions_under_skip_and_overwrite() {
        for mode in ["skip", "overwrite"] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("shared.v2.llm.md"), "existing").unwrap();
            let mut tasks: Vec<_> = ["one", "two"]
                .iter()
                .map(|id| Task {
                    source: format!("https://example.com/page?id={id}"),
                    display: id.to_string(),
                    report_key: format!("https://example.com/page?id={id}"),
                    output: Some(dir.path().to_owned()),
                    filename: Some("shared.md".into()),
                    reserved_stem: None,
                    source_file: None,
                })
                .collect();
            let mut cfg = config::defaults();
            cfg["output"]["on_conflict"] = json!(mode);
            reserve_batch_names(&mut tasks, &cfg).unwrap();
            assert_eq!(tasks[0].reserved_stem.as_deref(), Some("shared"));
            assert_eq!(tasks[1].reserved_stem.as_deref(), Some("shared.v3"));
            assert_eq!(
                std::fs::read_to_string(dir.path().join("shared.v2.llm.md")).unwrap(),
                "existing"
            );
            assert!(!dir.path().join("shared.md").exists());
        }
    }
}
