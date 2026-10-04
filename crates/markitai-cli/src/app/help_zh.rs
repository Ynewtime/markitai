//! Chinese `--help`.
//!
//! Clap's derive declares the command tree once, in English. When the terminal
//! language is Chinese, [`localize`] rewrites that tree before it is printed
//! or parsed; for English nothing here runs, so English help is built exactly
//! as before. Option names, value names, paths and code literals are never
//! translated. Clap's own usage errors stay English.
//!
//! Three things clap does not offer are handled here. Its help and version
//! flags are replaced by equivalent ones carrying Chinese text. Clap prints
//! the headings "Arguments" and "Options" on its own, so every option gets an
//! explicit Chinese heading instead. And it measures every character as one
//! column and breaks lines only at spaces, so Chinese text is wrapped here by
//! display width and printed with each option's help on its own line (as the
//! root command already does).
use clap::{Arg, ArgAction, Command};

/// Upper limit for Chinese help. A narrower console supplies its real width.
const WIDTH: usize = 78;
/// Clap indents the help of an option by two tabs' worth of columns.
const ARGUMENT_INDENT: usize = 10;
/// Subcommand summaries sit after the command name in the parent's list.
const SUMMARY_WIDTH: usize = 60;
const OPTIONS: &str = "选项";

/// Rewrites the English command tree that `english` builds into its Chinese
/// form.
pub(super) fn localize(english: fn() -> Command) -> Command {
    localize_at_width(english, super::progress::columns().min(WIDTH))
}

fn localize_at_width(english: fn() -> Command, width: usize) -> Command {
    let width = width.max(ARGUMENT_INDENT + 2);
    // Before it is built, a command does not yet know which of its options
    // take values, their defaults and possible values, or whether it has a
    // long help. A second, built copy answers those questions the way clap
    // will.
    let mut reference = english();
    reference.build();
    rewrite(english(), &reference, width, SUMMARY_WIDTH.min(width))
}

fn rewrite(
    mut command: Command,
    reference: &Command,
    width: usize,
    summary_width: usize,
) -> Command {
    let name = command.get_name().to_owned();
    let flag = |id: &str| reference.get_arguments().find(|arg| arg.get_id() == id);
    let long_help = flag("help").is_some_and(|arg| arg.get_long_help().is_some());
    let versioned = flag("version").is_some();

    // The flags go in first so that they take part in the ordering below.
    command = command.disable_help_flag(true).arg(help_flag(long_help));
    if versioned {
        command = command.disable_version_flag(true).arg(version_flag());
    }
    // Clap prints headings in the order their first option appears. Put the
    // positionals first, then the options that had no heading of their own
    // (the help and version flags among them), then the grouped ones: the
    // order English help shows. `mut_arg` moves an option to the end, so the
    // processing order is the final order.
    let rank = |arg: &Arg| {
        if arg.is_positional() {
            0
        } else if arg
            .get_help_heading()
            .is_none_or(|heading| heading == OPTIONS)
        {
            1
        } else {
            2
        }
    };
    let order: Vec<String> = (0..3)
        .flat_map(|wanted| {
            command
                .get_arguments()
                .filter(move |arg| rank(arg) == wanted)
                .map(|arg| arg.get_id().as_str().to_owned())
                .collect::<Vec<_>>()
        })
        .collect();
    for id in order {
        command = command.mut_arg(id, |arg| argument(arg, &name, reference, width));
    }
    if let Some(text) = describe(&name) {
        command = command.about(wrap(text.about, summary_width));
        if command.get_long_about().is_some() && !text.long_about.is_empty() {
            command = command.long_about(wrap(text.long_about, width));
        }
        if command.get_after_help().is_some() && !text.after.is_empty() {
            command = command.after_help(wrap(text.after, width));
        }
    }
    let usage = *command.get_styles().get_usage();
    let usage_line = if width < 60 {
        "\n  {usage}"
    } else {
        " {usage}"
    };
    let summary_width = width
        .saturating_sub(
            4 + reference
                .get_subcommands()
                .map(|child| child.get_name().len())
                .max()
                .unwrap_or(0),
        )
        .clamp(1, SUMMARY_WIDTH);
    command = command
        .term_width(width)
        .subcommand_help_heading("命令")
        .help_template(format!(
            "{{before-help}}{{about-with-newline}}\n{usage}用法:{usage:#}{usage_line}\n\n{{all-args}}{{after-help}}"
        ));
    for child in command.get_subcommands_mut() {
        let built = reference
            .find_subcommand(child.get_name())
            .expect("the built copy has every subcommand");
        *child = rewrite(std::mem::take(child), built, width, summary_width);
    }
    command
}

/// `-V, --version`, replacing clap's own flag.
pub(super) fn version_flag() -> Arg {
    Arg::new("version")
        .short('V')
        .long("version")
        .action(ArgAction::Version)
        .help("显示版本")
        .help_heading(OPTIONS)
}

/// `-h, --help`. With a long form, `-h` points at `--help` and the other way
/// round, as clap's own flag does.
fn help_flag(long_help: bool) -> Arg {
    let flag = Arg::new("help")
        .short('h')
        .long("help")
        .action(ArgAction::Help)
        .help_heading(OPTIONS);
    if long_help {
        flag.help("显示帮助（--help 查看详细说明）")
            .long_help("显示帮助（-h 查看摘要）")
    } else {
        flag.help("显示帮助")
    }
}

/// Chinese heading, text and value hints for one option or argument.
fn argument(arg: Arg, command: &str, reference: &Command, width: usize) -> Arg {
    let id = arg.get_id().as_str().to_owned();
    let heading = match arg.get_help_heading() {
        Some(heading) => heading_zh(heading),
        None if arg.is_positional() => "参数",
        None => OPTIONS,
    };
    let mut arg = arg.help_heading(heading);
    if arg.is_hide_set() {
        return arg;
    }
    let builtin = matches!(id.as_str(), "help" | "version");
    let Some(text) = (if builtin {
        // Already Chinese, but its text may need wrapping like the rest.
        arg.get_help().map(ToString::to_string)
    } else {
        argument_text(command, &id).map(str::to_owned)
    }) else {
        return arg;
    };
    // Clap prints defaults and possible values in English brackets after the
    // help; say them in Chinese instead.
    let mut note = Vec::new();
    if let Some(built) = reference
        .get_arguments()
        .find(|built| built.get_id() == id.as_str())
    {
        if built.get_action().takes_values() && !built.is_hide_default_value_set() {
            let defaults: Vec<_> = built
                .get_default_values()
                .iter()
                .map(|value| value.to_string_lossy().into_owned())
                .collect();
            if !defaults.is_empty() {
                note.push(format!("[默认：{}]", defaults.join(" ")));
                arg = arg.hide_default_value(true);
            }
        }
        if !built.is_hide_possible_values_set() {
            let values: Vec<_> = built
                .get_possible_values()
                .iter()
                .filter(|value| !value.is_hide_set())
                .map(|value| value.get_name().to_owned())
                .collect();
            if !values.is_empty() {
                note.push(format!("[可选值：{}]", values.join(", ")));
                arg = arg.hide_possible_values(true);
            }
        }
    }
    let mut help = text;
    if !note.is_empty() {
        help.push(' ');
        help.push_str(&note.join(" "));
    }
    let arg = arg.help(wrap(&help, width - ARGUMENT_INDENT));
    if builtin {
        arg
    } else {
        arg.next_line_help(true)
    }
}

fn heading_zh(heading: &str) -> &'static str {
    match heading {
        "Output" => "输出",
        "Configuration" => "配置",
        "LLM, OCR and screenshots" => "LLM、OCR 与截图",
        "URL fetching and backends" => "URL 抓取与后端",
        "Batch processing" => "批量处理",
        "Cache and images" => "缓存与图片",
        "Messages and logging" => "消息与日志",
        OPTIONS => OPTIONS,
        // Left alone, a new group would show up in English; the tests that
        // walk the tree catch it.
        _ => "其他",
    }
}

const ROOT: &str = "markitai";

struct Description {
    about: &'static str,
    long_about: &'static str,
    after: &'static str,
}

fn describe_with(
    about: &'static str,
    long_about: &'static str,
    after: &'static str,
) -> Description {
    Description {
        about,
        long_about,
        after,
    }
}

/// The summary, long description and closing examples of a command, by name.
fn describe(command: &str) -> Option<Description> {
    Some(match command {
        ROOT => describe_with("把文档和 URL 转换为 Markdown", "", ROOT_AFTER),
        "config" => describe_with("查看或编辑配置", "", CONFIG_AFTER),
        "init" => describe_with(
            "创建或更新配置文件，并写入检测到的 API 模型",
            "创建或更新配置文件，并写入检测到的 API 模型。\n\n从环境变量（MODEL 与各供应商的 API key）检测 API 模型，不发出请求，也不保存 key 的值。生成的配置默认关闭 LLM 处理；需要时可在单次运行中加 --llm 启用，或执行 `markitai config set llm.enabled true`。",
            "示例:\n  markitai init                选择保存位置，以及如何处理已有文件\n  markitai init -y             不提示，创建或更新用户配置\n  markitai init --local        为当前项目写入 ./markitai.json\n  markitai init -o cfg.json    写入自定义路径",
        ),
        "doctor" => describe_with(
            "诊断已配置的工作流和可选的原生后端",
            "诊断已配置的工作流和可选的原生后端。\n\n缺少可选后端只会报告，不算失败。只有当配置要求了本机无法提供的能力时，退出状态才是 1：已启用的模型缺少凭据、配置的浏览器工作流无法启动、订阅运行时不可用，或 --fix 修复失败。不会发送模型请求，也不会打开远程页面。",
            "示例:\n  markitai doctor           人类可读的报告\n  markitai doctor --json    机器可读的检查结果\n  markitai doctor --fix     安装缺少的浏览器，以及安装或修复本地 OCR 模型",
        ),
        "cache" => describe_with("查看或清理 LLM 与 URL 抓取缓存", "", CACHE_AFTER),
        "auth" => describe_with("查看订阅认证状态，或委托官方运行时完成登录", "", AUTH_AFTER),
        "serve" => describe_with(
            "运行原生 REST 转换服务和网页界面",
            "",
            "示例:\n  markitai serve                          http://127.0.0.1:3600，并打开浏览器\n  markitai serve --port 8080 --no-open    换一个端口，不打开浏览器\n  markitai serve --host 0.0.0.0           局域网访问，需要启动时打印的访问令牌",
        ),
        "mcp" => describe_with(
            "通过标准输入/输出运行原生 MCP 服务",
            "通过标准输入/输出运行原生 MCP 服务\n\n工具：convert_document 与 convert_url 转换单个文档或网页，batch_convert 启动目录或 URL 列表任务，job_status 查询任务。由 MCP 客户端启动本命令；它在 stdout 回答，日志写 stderr。",
            MCP_AFTER,
        ),
        "list" => describe_with(
            "显示生效的配置；密钥已脱敏",
            "",
            "示例:\n  markitai config list               以 JSON 显示全部配置\n  markitai config list -f table      每行一项设置\n  markitai config list -f yaml       以 YAML 显示",
        ),
        "path" => describe_with(
            "显示正在使用的配置文件",
            "",
            "示例:\n  markitai config path                 正在使用的文件，或查找位置\n  markitai -c other.json config path   用 -c 指定的文件",
        ),
        "validate" => describe_with(
            "按 schema 校验配置文件",
            "",
            "示例:\n  markitai config validate                  校验正在使用的配置\n  markitai config validate ./markitai.json  校验指定文件",
        ),
        "get" => describe_with(
            "输出单个配置值；配置段以 JSON 输出",
            "",
            "示例:\n  markitai config get llm.enabled     读取单个值\n  markitai config get output          以 JSON 读取整个配置段",
        ),
        "set" => describe_with(
            "校验并保存单个配置值；无效值不会写入",
            "",
            "示例:\n  markitai config set output.on_conflict skip    永不替换已有结果\n  markitai config set output.dir ~/Documents/md  批处理默认写到这里\n  markitai config set llm.enabled true           启用模型处理",
        ),
        "edit" => describe_with(
            "在终端中编辑配置项；每个值校验后立即保存",
            "",
            "示例:\n  markitai config edit    浏览、搜索（/关键字）并修改设置；q 退出",
        ),
        "stats" => describe_with(
            "显示缓存条目数与磁盘占用",
            "",
            "示例:\n  markitai cache stats                  条目数与磁盘占用\n  markitai cache stats --json           供脚本使用的精确字节数\n  markitai cache stats -v --limit 5     按模型列出最近 5 条 LLM 条目",
        ),
        "clear" => describe_with(
            "清理 LLM 与 URL 抓取缓存",
            "",
            "示例:\n  markitai cache clear                           先确认再清理\n  markitai cache clear -y                        不经确认直接清理\n  markitai cache clear -y --include-spa-domains  同时忘记已学习的浏览器域名",
        ),
        "spa-domains" => describe_with(
            "查看或清空已学习的浏览器域名路由",
            "",
            "示例:\n  markitai cache spa-domains           列出已学习的域名\n  markitai cache spa-domains --json    以 JSON 输出\n  markitai cache spa-domains --clear   忘记全部已学习的域名",
        ),
        "copilot" => describe_with(
            "通过已安装的官方 CLI 使用 GitHub Copilot",
            "",
            "示例:\n  markitai auth copilot status           运行时是否已登录？\n  markitai auth copilot status --json    机器可读的状态\n  markitai auth copilot login            通过官方运行时登录",
        ),
        "claude" => describe_with(
            "Claude 订阅适配器状态",
            "",
            "示例:\n  markitai auth claude status           运行时是否已登录？\n  markitai auth claude status --json    机器可读的状态\n  markitai auth claude login            通过官方运行时登录",
        ),
        "chatgpt" => describe_with(
            "ChatGPT 订阅适配器状态",
            "",
            "示例:\n  markitai auth chatgpt status           运行时是否已登录？\n  markitai auth chatgpt status --json    机器可读的状态\n  markitai auth chatgpt login            通过官方运行时登录",
        ),
        "status" => describe_with("查看现有认证状态，不发起登录", "", ""),
        "login" => describe_with("打开官方运行时的交互式登录流程", "", ""),
        _ => return None,
    })
}

const ROOT_AFTER: &str = "\
-p 预设:
  minimal   纯转换，不经过模型处理
  standard  --llm --alt --desc
  rich      --llm --alt --desc --screenshot

示例:
  markitai report.docx                     把 Markdown 输出到 stdout
  markitai report.pdf -o out/              写出 out/report.pdf.md
  markitai notes.html -o out/notes.md      指定输出文件名
  markitai ./docs -o out/ -g '**/*.pdf'    转换目录中的 PDF
  markitai links.urls -o out/ --resume     继续被中断的 URL 列表
  markitai https://example.com -o out/     转换网页
  markitai scan.png --ocr                  识别图片中的文字
  markitai report.pdf -p standard -o out/  用已配置的模型增强
  markitai init                            创建配置文件
  markitai doctor                          检查模型和可选后端";

const CONFIG_AFTER: &str = "\
键使用点号表示法，例如 llm.enabled 或 llm.model_list[0].model_name。使用哪个配置文件，依次取 -c、MARKITAI_CONFIG、./markitai.json，最后是用户目录下的 config.json。

示例:
  markitai config list -f table           显示全部生效配置
  markitai config get llm.enabled         读取单个值
  markitai config set output.dir ./out    校验并保存单个值
  markitai config path                    显示正在使用哪个配置文件";

const CACHE_AFTER: &str = "\
示例:
  markitai cache stats                    条目数与磁盘占用
  markitai cache stats -v --limit 10      按模型列出最近的 LLM 条目
  markitai cache clear -y                 不经确认直接清理";

const AUTH_AFTER: &str = "\
示例:
  markitai auth                         全部订阅运行时的状态
  markitai auth claude status           单个运行时（加 --json 供脚本使用）
  markitai auth chatgpt login           通过官方运行时登录";

const MCP_AFTER: &str = "\
示例:
  markitai mcp                          通过 stdio 向 MCP 客户端提供这些工具
  markitai -c cfg.json mcp              使用指定的配置文件提供服务";

const SECRETS_SHOWN: &str = "显示密钥的值而不是脱敏（不要写入共享日志）";

/// (command, option id, text). Global options are listed under the root.
const ARGUMENTS: &[(&str, &str, &str)] = &[
    (
        ROOT,
        "input",
        "文档、URL、.urls 列表、目录，或作为整体处理的 .numbers 包",
    ),
    (
        ROOT,
        "output",
        "输出目录；单个输入时也可指定确切的 .md 路径。省略则把 Markdown 输出到 stdout",
    ),
    (
        ROOT,
        "json",
        "在 stdout 输出一个 JSON 结果；需要 -o。用法错误仍写入 stderr",
    ),
    (
        ROOT,
        "dry_run",
        "预览要处理的输入，不转换，也不写出任何输出",
    ),
    (
        ROOT,
        "slide_markers",
        "保留幻灯片编号注释（--no-slide-markers 关闭）",
    ),
    (
        ROOT,
        "record_history",
        "把本次运行归档到 `markitai serve` 的历史（--no-record-history 关闭）；只输出到 stdout 的转换不会归档",
    ),
    (
        ROOT,
        "config",
        "配置文件路径；必须已存在，config set/edit 可以新建",
    ),
    (
        ROOT,
        "config_json",
        "把内联 JSON 深度合并到配置文件之上；显式的转换参数仍然优先",
    ),
    (
        ROOT,
        "preset",
        "使用 minimal、standard、rich 或已配置的预设（不区分大小写）",
    ),
    (
        ROOT,
        "profile",
        "为 rag、obsidian 或 okf 调整资源与 frontmatter 的形态；与预设相互独立",
    ),
    (
        ROOT,
        "interactive",
        "在终端中选择转换内容；修改只对本次会话生效",
    ),
    (
        ROOT,
        "llm",
        "启用模型处理（--no-llm 关闭）。成对开关以最后一个为准；都不给时由配置决定",
    ),
    (
        ROOT,
        "alt",
        "启用 LLM 处理时生成图片 alt 文本（--no-alt 关闭）",
    ),
    (
        ROOT,
        "desc",
        "启用 LLM 处理时写入图片描述（--no-desc 关闭）",
    ),
    (
        ROOT,
        "ocr",
        "识别扫描内容：使用本地 OCR 或视觉模型（--no-ocr 关闭）",
    ),
    (
        ROOT,
        "screenshot",
        "截取受支持的文档页面或浏览器页面；可能需要可选后端（--no-screenshot 关闭）",
    ),
    (
        ROOT,
        "screenshot_only",
        "把截图本身作为内容（隐含 --screenshot）。配合 --llm 时读取像素；不用 --llm 时，普通网页只保存图片而不生成 Markdown。PDF 媒体仍保留 Markdown。URL 的 --pure 优先",
    ),
    (
        ROOT,
        "pure",
        "保留源文本，不生成常规元数据；对 URL，pure 优先于视觉 LLM 输入（--no-pure 关闭）",
    ),
    (ROOT, "keep_base", "增强输出之外，同时保留基础 Markdown"),
    (ROOT, "llm_concurrency", "本次转换共享的最大并发模型请求数"),
    (
        ROOT,
        "strategy",
        "URL 抓取策略：auto（先静态抓取，必要时再用本地浏览器；只有通过 fetch.remote_consent 明确同意后才使用远程服务）、static 和 playwright 在本地运行；defuddle、jina 和 cloudflare（你的账户）会把 URL 发送给对应的远程服务",
    ),
    (
        ROOT,
        "backend",
        "文件后端：native，或 cloudflare：在你的 Cloudflare 账户中用 Workers AI 转换受支持的文件（会上传文件）",
    ),
    (
        ROOT,
        "no_remote_fetch",
        "禁止使用远程提取服务；不会关闭已显式配置的模型请求",
    ),
    (
        ROOT,
        "resume",
        "按相同路径与选项继续目录或 .urls 批处理；已完成的条目保持完成",
    ),
    (
        ROOT,
        "batch_concurrency",
        "文件转换的最大并发数；与 URL 并发数和模型请求上限相互独立",
    ),
    (
        ROOT,
        "url_concurrency",
        "URL 转换的最大并发数，与文件处理分别计算",
    ),
    (
        ROOT,
        "globs",
        "包含或排除目录内的相对路径；可重复，以 ! 开头表示排除。请在 shell 中给模式加引号",
    ),
    (ROOT, "max_depth", "目录扫描深度；0 表示只扫描输入目录本身"),
    (
        ROOT,
        "llm_batch",
        "把目录的文本增强提交给 OpenAI Batch API；--resume 会在 -o 目录中继续已冻结的任务",
    ),
    (
        ROOT,
        "llm_batch_timeout",
        "本地等待供应商批处理的最长时间（秒，至少 60）；超时不会取消该批处理",
    ),
    (
        ROOT,
        "llm_batch_collect",
        "把已保存的供应商批处理收回到它原来的 -o 目录；无需再给输入",
    ),
    (
        ROOT,
        "no_cache",
        "跳过缓存读取，但仍写入成功的新结果（--cache 恢复读取）",
    ),
    (
        ROOT,
        "no_cache_for",
        "逗号分隔的 glob 模式；匹配的输入会跳过缓存读取",
    ),
    (ROOT, "no_compress", "关闭图片压缩（--compress 启用）"),
    (
        ROOT,
        "verbose",
        "显示报告路径等详情。单个输入默认保持安静：只输出写入的路径、警告和错误；stdout 上的 Markdown 保持干净",
    ),
    (
        ROOT,
        "quiet",
        "只输出错误：不显示写入路径、警告、进度和批量摘要",
    ),
    (
        ROOT,
        "log_level",
        "转换文件日志的级别；需要设置 log.dir。控制台输出仍由 --verbose/--quiet 决定",
    ),
    (
        "init",
        "yes",
        "不提示：创建文件，或把新检测到的模型追加到已有文件",
    ),
    (
        "init",
        "output",
        "写入这个文件，或写入该目录中的 markitai.json",
    ),
    (
        "init",
        "local",
        "写入当前目录的 ./markitai.json，而不是用户配置",
    ),
    ("doctor", "json", "把检查结果输出为一个 JSON 对象"),
    (
        "doctor",
        "fix",
        "安装缺少的官方 Chrome headless shell，以及安装或修复本地 OCR 模型",
    ),
    (
        "doctor",
        "suggest_extras",
        "Python 包 extras；本原生构建不适用，会被拒绝",
    ),
    ("stats", "json", "以 JSON 输出统计"),
    ("stats", "verbose", "按模型列出最近的 LLM 缓存条目"),
    ("stats", "limit", "配合 --verbose 时最多列出的条目数"),
    ("clear", "yes", "不经确认直接清理"),
    (
        "clear",
        "include_spa_domains",
        "同时清除已学习的仅浏览器渲染域名",
    ),
    ("spa-domains", "json", "以 JSON 输出域名"),
    ("spa-domains", "clear", "清除全部已学习的域名"),
    ("list", "format", "输出格式"),
    ("list", "show_secrets", SECRETS_SHOWN),
    (
        "validate",
        "config_file",
        "要校验的文件；默认为正在使用的配置",
    ),
    ("get", "key", "点号表示法的键，例如 llm.enabled"),
    ("get", "show_secrets", SECRETS_SHOWN),
    ("set", "key", "点号表示法的键，例如 output.on_conflict"),
    ("set", "value", "新值；按该键声明的类型解析"),
    (
        "set",
        "show_secrets",
        "回显密钥的值而不是脱敏（不要写入共享日志）",
    ),
    ("status", "json", "以 JSON 输出状态"),
    (
        "serve",
        "host",
        "监听的网卡地址。所有 API 客户端都需要启动时打印的访问令牌（或 MARKITAI_SERVE_TOKEN）",
    ),
    ("serve", "port", "监听端口"),
    ("serve", "no_open", "启动后不打开浏览器"),
    (
        "serve",
        "no_auth",
        "关闭 API 令牌验证。远程客户端可以上传文件、读取或删除历史；URL 转换和模型设置需要直接本机连接",
    ),
    (
        "serve",
        "allowed_host",
        "同时允许此主机名的 Host/Origin（可重复）；本地主机的 Origin 仍须匹配请求端口",
    ),
];

fn argument_text(command: &str, id: &str) -> Option<&'static str> {
    ARGUMENTS
        .iter()
        .find(|(owner, name, _)| *owner == command && *name == id)
        .map(|(_, _, text)| *text)
}

/// Display columns of a character: Chinese and other fullwidth characters take
/// two, everything else one.
fn columns(c: char) -> usize {
    match c as u32 {
        0x1100..=0x115F
        | 0x2E80..=0xA4CF
        | 0xAC00..=0xD7A3
        | 0xF900..=0xFAFF
        | 0xFE30..=0xFE6F
        | 0xFF00..=0xFF60
        | 0xFFE0..=0xFFE6
        | 0x20000..=0x3FFFD => 2,
        _ => 1,
    }
}

/// Closing marks stay with what precedes them; opening marks with what follows.
fn closing(c: char) -> bool {
    matches!(
        c,
        '，' | '。' | '、' | '；' | '：' | '！' | '？' | '）' | '】' | '」' | '』' | '》'
    )
}

fn opening(c: char) -> bool {
    matches!(c, '（' | '【' | '「' | '『' | '《')
}

/// A line may break between two characters when one of them is wide or a
/// space, so that Chinese text breaks anywhere while option names, paths and
/// other words stay whole, except next to the marks above.
fn may_break(before: char, after: char) -> bool {
    !opening(before)
        && !closing(after)
        && (before == ' ' || after == ' ' || columns(before) == 2 || columns(after) == 2)
}

/// Wraps `text` to `limit` display columns at the places [`may_break`] allows;
/// a space at a break is dropped, and a word wider than the limit is kept
/// whole. Lines that start with a space are hand-aligned examples and stay as
/// written.
fn wrap(text: &str, limit: usize) -> String {
    let mut wrapped = String::with_capacity(text.len() + 16);
    for (index, line) in text.split('\n').enumerate() {
        if index > 0 {
            wrapped.push('\n');
        }
        if line.starts_with(' ') || line.chars().map(columns).sum::<usize>() <= limit {
            wrapped.push_str(line);
            continue;
        }
        let mut used = 0;
        // The latest place to break at: its offset in `wrapped`.
        let mut chance = None;
        let mut previous = ' ';
        for c in line.chars() {
            if used > 0 && may_break(previous, c) {
                chance = Some(wrapped.len());
            }
            wrapped.push(c);
            used += columns(c);
            if used > limit
                && let Some(at) = chance.take()
            {
                let tail = wrapped.split_off(at);
                wrapped.truncate(wrapped.trim_end_matches(' ').len());
                wrapped.push('\n');
                let tail = tail.trim_start_matches(' ');
                wrapped.push_str(tail);
                used = tail.chars().map(columns).sum();
            }
            previous = c;
        }
    }
    wrapped
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn english() -> Command {
        super::super::Cli::command()
    }

    fn chinese() -> Command {
        let mut command = localize(super::super::english_command);
        command.build();
        command
    }

    fn has_chinese(text: &str) -> bool {
        text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
    }

    /// Every command of the tree with its space-joined path.
    fn walk<'a>(command: &'a Command, path: &str, out: &mut Vec<(String, &'a Command)>) {
        out.push((path.to_owned(), command));
        for child in command.get_subcommands() {
            let path = format!("{path} {}", child.get_name()).trim().to_owned();
            walk(child, &path, out);
        }
    }

    fn commands(root: &Command) -> Vec<(String, &Command)> {
        let mut out = Vec::new();
        walk(root, "", &mut out);
        out
    }

    #[test]
    fn every_visible_argument_command_and_note_has_chinese_text() {
        let mut tree = english();
        tree.build();
        let zh = chinese();
        let (before, after) = (commands(&tree), commands(&zh));
        assert_eq!(before.len(), after.len());
        let mut missing = Vec::new();
        for ((path, en), (_, zh)) in before.iter().zip(&after) {
            if !has_chinese(&zh.get_about().map(ToString::to_string).unwrap_or_default()) {
                missing.push(format!("{path}: summary"));
            }
            for (note, english, chinese) in [
                ("long description", en.get_long_about(), zh.get_long_about()),
                ("closing notes", en.get_after_help(), zh.get_after_help()),
            ] {
                if english.is_some()
                    && !has_chinese(&chinese.map(ToString::to_string).unwrap_or_default())
                {
                    missing.push(format!("{path}: {note}"));
                }
            }
            // The Chinese tree orders options by heading, so match them by id.
            for arg in en.get_arguments().filter(|arg| !arg.is_hide_set()) {
                let found = zh
                    .get_arguments()
                    .find(|candidate| candidate.get_id() == arg.get_id())
                    .unwrap();
                let text = found
                    .get_help()
                    .map(ToString::to_string)
                    .unwrap_or_default();
                if !has_chinese(&text) {
                    missing.push(format!("{path}: --{}", arg.get_id()));
                }
                let heading = found.get_help_heading().unwrap_or_default();
                if !has_chinese(heading) || heading == "其他" {
                    missing.push(format!("{path}: heading of {}", arg.get_id()));
                }
            }
        }
        assert!(
            missing.is_empty(),
            "add Chinese text in app/help_zh.rs for: {missing:#?}"
        );
    }

    #[test]
    fn no_table_entry_names_an_argument_or_command_that_does_not_exist() {
        let mut tree = english();
        tree.build();
        let all = commands(&tree);
        for (owner, id, _) in ARGUMENTS {
            let found = all
                .iter()
                .filter(|(_, command)| command.get_name() == *owner)
                .any(|(_, command)| command.get_arguments().any(|a| a.get_id() == *id));
            assert!(found, "stale entry: {owner} {id}");
        }
        for name in all.iter().map(|(_, command)| command.get_name()) {
            assert!(describe(name).is_some(), "no description for {name}");
        }
        // Every entry is used exactly once per name.
        for (index, (owner, id, _)) in ARGUMENTS.iter().enumerate() {
            assert!(
                !ARGUMENTS[..index]
                    .iter()
                    .any(|(other, name, _)| other == owner && name == id),
                "duplicate entry: {owner} {id}"
            );
        }
    }

    #[test]
    fn chinese_help_keeps_the_same_options_and_commands_in_english_order() {
        let mut tree = english();
        tree.build();
        let zh = chinese();
        for ((path, en), (_, zh)) in commands(&tree).iter().zip(&commands(&zh)) {
            let mut before: Vec<_> = en
                .get_arguments()
                .map(|arg| (arg.get_id().to_string(), arg.get_long(), arg.get_short()))
                .collect();
            let mut after: Vec<_> = zh
                .get_arguments()
                .map(|arg| (arg.get_id().to_string(), arg.get_long(), arg.get_short()))
                .collect();
            before.sort();
            after.sort();
            assert_eq!(before, after, "{path}");
            let names = |command: &Command| {
                command
                    .get_subcommands()
                    .map(|child| child.get_name().to_owned())
                    .collect::<Vec<_>>()
            };
            assert_eq!(names(en), names(zh), "{path}");
        }
    }

    #[test]
    fn chinese_help_fits_an_80_column_terminal_and_shows_no_english_frame() {
        for (path, command) in commands(&chinese()) {
            for long in [false, true] {
                let mut command = command.clone().term_width(100);
                let help = if long {
                    command.render_long_help()
                } else {
                    command.render_help()
                }
                .to_string();
                for line in help.lines() {
                    let width: usize = line.chars().map(columns).sum();
                    assert!(width <= 80, "{path} (long: {long}) is {width} wide: {line}");
                }
                for english in [
                    "Usage:",
                    "Options:",
                    "Arguments:",
                    "Commands:",
                    "Print help",
                    "Print version",
                    "possible values",
                    "[default",
                ] {
                    assert!(!help.contains(english), "{path}: {english}\n{help}");
                }
                assert!(help.contains("用法:"), "{path}");
            }
        }
    }

    #[test]
    fn chinese_root_help_fits_narrow_display_columns() {
        for width in [40, 80] {
            let mut command = localize_at_width(super::super::english_command, width);
            let help = command.render_long_help().to_string();
            for line in help
                .lines()
                .filter(|line| line.chars().any(|c| columns(c) == 2))
            {
                assert!(
                    line.chars().map(columns).sum::<usize>() <= width,
                    "{width}: {line}"
                );
            }
            for flag in ["--json", "--no-llm", "--resume", "--batch-concurrency"] {
                assert!(help.contains(flag), "{width}: {flag}");
            }
        }
    }

    #[test]
    fn wrapping_counts_chinese_as_two_columns_and_keeps_options_whole() {
        let line = "启用模型处理（--no-llm 关闭）。成对开关以最后一个为准；都不给时由配置决定";
        let wrapped = wrap(line, 30);
        for part in wrapped.lines() {
            assert!(part.chars().map(columns).sum::<usize>() <= 30, "{part}");
        }
        assert!(wrapped.contains("--no-llm"));
        // Nothing but the dropped spaces and the inserted breaks differ.
        assert_eq!(wrapped.replace(['\n', ' '], ""), line.replace(' ', ""));
        // A closing mark never starts a line and an opening mark never ends one.
        for part in wrapped.lines() {
            assert!(!part.starts_with(['，', '。', '）', '；']), "{part}");
            assert!(!part.ends_with(['（']), "{part}");
        }
        // Hand-aligned lines and short lines stay as written.
        assert_eq!(
            wrap("  markitai x     很长很长很长很长很长很长很长很长很长", 10),
            "  markitai x     很长很长很长很长很长很长很长很长很长"
        );
        assert_eq!(wrap("短行", 10), "短行");
        // The break goes before an opening mark, not after it, and after the
        // text a closing mark follows, not before it.
        assert_eq!(wrap("甲乙丙丁（戊己", 10), "甲乙丙丁\n（戊己");
        assert_eq!(wrap("甲乙丙丁，戊己庚辛", 8), "甲乙丙\n丁，戊己\n庚辛");
        // An English word wider than the limit is not split.
        assert_eq!(
            wrap("很 abcdefghijklmnopqrstuvwxyz 长", 8),
            "很\nabcdefghijklmnopqrstuvwxyz\n长"
        );
    }
}
