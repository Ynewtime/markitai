//! Chinese wording for the diagnostics the CLI prints in English only because
//! they are built where no terminal language is known: run-level errors, and
//! the reasons and warnings that converters return.
//!
//! The English text is the one `--json`, reports and the file log carry, so it
//! is not changed at its source. The terminal translates the lines it knows
//! when it prints them in Chinese, which keeps a run in one language. A
//! message that is not in the table (a provider's reply, an error from the
//! operating system, a parser's detail inside parentheses) is printed as it
//! came; those are technical details, not wording of this program.

/// English templates with `{}` for the variable parts, and their Chinese.
/// The first match wins.
const RULES: &[(&str, &str)] = &[
    ("Input file not found: {}", "找不到输入文件：{}"),
    ("Input is a directory: {}", "输入是目录：{}"),
    ("Input does not exist: {}", "输入不存在：{}"),
    (
        "No model configured; set MODEL and a provider API key, or llm.model_list",
        "未配置 LLM 模型；请设置 MODEL 和供应商的 API key，或配置 llm.model_list",
    ),
    (
        "The installed Codex could not list its models, so chatgpt/{} is sent as named, unvalidated",
        "已安装的 Codex 无法列出它的模型，因此按名称发送 chatgpt/{}，未经验证",
    ),
    (
        "The installed Codex does not offer chatgpt/{}; it offers {}",
        "已安装的 Codex 不提供 chatgpt/{}；它提供 {}",
    ),
    (
        "{} is newer than {}, the release this adapter was verified with; it is used, and every message it sends is still checked",
        "{} 比本适配器验证过的 {} 更新；仍会使用它，并检查它发来的每条消息",
    ),
    (
        "{} is left out of the model pool: {} is not set; set it with: {} (or add {}=your_key to {})",
        "{} 已移出模型池：未设置 {}；设置方法：{}（或把 {}=your_key 写入 {}）",
    ),
    (
        "{} is left out of the model pool: it has no API key (set litellm_params.api_key)",
        "{} 已移出模型池：没有 API key（请设置 litellm_params.api_key）",
    ),
    (
        "{} is not set, and {} needs it; set it with: {} (or add {}=your_key to {})",
        "未设置 {}，{} 需要它；设置方法：{}（或把 {}=your_key 写入 {}）",
    ),
    (
        "{} is not set, and {} need it; set it with: {} (or add {}=your_key to {})",
        "未设置 {}，{} 需要它；设置方法：{}（或把 {}=your_key 写入 {}）",
    ),
    (
        "No LLM model can be used: {} need an API key; set litellm_params.api_key",
        "没有可用的 LLM 模型：{} 需要 API key；请设置 litellm_params.api_key",
    ),
    (
        "No LLM model can be used: {} needs an API key; set litellm_params.api_key",
        "没有可用的 LLM 模型：{} 需要 API key；请设置 litellm_params.api_key",
    ),
    (
        "No LLM model can be used: {}. Set one of them, for example {} (or in {})",
        "没有可用的 LLM 模型：{}。设置其中一个即可，例如 {}（或写入 {}）",
    ),
    (
        "Unsupported file format: {}. Supported extensions: {}.",
        "不支持的文件格式：{}。支持的扩展名：{}。",
    ),
    (
        "The file appears to be damaged or truncated ({})",
        "文件似乎已损坏或被截断（{}）",
    ),
    ("No valid URLs found in {}.", "{} 中没有找到有效的 URL。"),
    (
        "PDF page {}: native text was not recovered ({}); OCR is required to read it. Run again with --ocr.",
        "PDF 第 {} 页：未能提取文字（{}）；需要 OCR 才能读取。请加上 --ocr 重新运行。",
    ),
    (
        "PDF pages {}: native text was not recovered ({}); OCR is required to read them. Run again with --ocr.",
        "PDF 第 {} 页：未能提取文字（{}）；需要 OCR 才能读取。请加上 --ocr 重新运行。",
    ),
    (
        "--llm-batch requires enabled LLM processing",
        "--llm-batch 需要启用 LLM 处理",
    ),
    (
        "--llm-batch requires a directory input",
        "--llm-batch 需要目录作为输入",
    ),
    (
        "--llm-batch-timeout requires --llm-batch",
        "--llm-batch-timeout 需要同时使用 --llm-batch",
    ),
    (
        "--glob and --max-depth require a directory input",
        "--glob 和 --max-depth 需要目录作为输入",
    ),
    (
        "Unknown preset '{}'. Available: {}",
        "未知预设 '{}'。可用：{}",
    ),
    (
        "A directory or .urls batch needs an output directory: pass -o DIR or set output.dir in the configuration",
        "目录或 .urls 批量转换需要输出目录：请用 -o DIR 指定，或在配置中设置 output.dir",
    ),
    (
        "A batch requires an output directory, not a .md file",
        "批量转换需要输出目录，不能是 .md 文件",
    ),
    (
        "--json requires -o: stdout carries the JSON result, so the Markdown needs an output directory or .md file",
        "--json 需要同时使用 -o：标准输出用于输出 JSON 结果，Markdown 需要写入输出目录或 .md 文件",
    ),
    (
        "--interactive and --json cannot be used together",
        "--interactive 和 --json 不能同时使用",
    ),
    (
        "--json and --fix cannot be used together",
        "--json 和 --fix 不能同时使用",
    ),
    (
        "--config-json must be a JSON object",
        "--config-json 必须是 JSON 对象",
    ),
    (
        "--config-json overrides cannot be saved; drop --config-json to write a config file",
        "无法保存 --config-json 的覆盖设置；要写入配置文件，请去掉 --config-json",
    ),
    (
        "--config-json overrides cannot be saved; drop --config-json to edit a config file",
        "无法保存 --config-json 的覆盖设置；要编辑配置文件，请去掉 --config-json",
    ),
    (
        "Configuration must be a JSON object",
        "配置必须是 JSON 对象",
    ),
    (
        "Configuration file does not exist: {}",
        "配置文件不存在：{}",
    ),
    (
        "Cannot mix INPUT, --interactive or Provider Batch options with a subcommand",
        "INPUT、--interactive 或 Provider Batch 选项不能与子命令混用",
    ),
    (
        "Batch collection cannot be mixed with input, submission, resume, preview or interactive mode",
        "批量收集不能与输入、提交、续跑、预览或交互模式混用",
    ),
    (
        "Frozen Batch resume cannot use preview or interactive mode",
        "续跑已冻结的批量任务时不能使用预览或交互模式",
    ),
    (
        "--resume for a single file or URL is not implemented: only directory and .urls batches save progress. Run it again without --resume.",
        "单个文件或 URL 不支持 --resume：只有目录和 .urls 批量转换会保存进度。请去掉 --resume 重新运行。",
    ),
    (
        "Output {} is the input file itself, and overwriting it would destroy the source. Choose another output name or directory.",
        "输出 {} 就是输入文件本身，覆盖会毁掉源文件。请换一个输出名称或目录。",
    ),
    (
        "Output path {} exists and is not a directory; pass a directory with -o (a single input may also name a .md file)",
        "输出路径 {} 已存在且不是目录；请用 -o 指定目录（单个输入也可以指定 .md 文件）",
    ),
    (
        "Cannot create output directory {}: {} is not a directory",
        "无法创建输出目录 {}：{} 不是目录",
    ),
    (
        "Batch exceeds scan_max_files ({}); narrow --glob or increase the configuration limit",
        "批量文件数超过 scan_max_files（{}）；请缩小 --glob 范围或提高配置中的上限",
    ),
    (
        "URL output_name must be a filename without directory components",
        "URL 的 output_name 必须是不含目录的文件名",
    ),
    (
        "Cannot expand ~ in {}: HOME is not set. Spell the directory out, or use ./~ for a directory literally named ~",
        "无法展开 {} 中的 ~：未设置 HOME。请写出完整目录；若目录名就是 ~，请写成 ./~",
    ),
    // Configuration files, the environment and the core's own notes.
    ("Invalid JSON in {}: {}", "{} 不是有效的 JSON：{}"),
    ("Cannot read config {}: {}", "无法读取配置 {}：{}"),
    ("Environment variable not found: {}", "找不到环境变量：{}"),
    (
        "Config file not found: {} (using defaults)",
        "找不到配置文件：{}（使用默认配置）",
    ),
    (
        "Unknown configuration key: {}. Run `markitai config list -f table` to see every key.",
        "未知的配置键：{}。运行 `markitai config list -f table` 查看全部配置键。",
    ),
    (
        "log.rotation '{}' is not a size ('10 MB'), an interval ('1 week'), a time of day ('00:00', 'monday at 12:00') or hourly/daily/weekly/monthly/yearly",
        "log.rotation '{}' 不是大小（'10 MB'）、间隔（'1 week'）、时刻（'00:00'、'monday at 12:00'），也不是 hourly/daily/weekly/monthly/yearly",
    ),
    (
        "log.retention '{}' is not a duration such as '7 days' or '1 week, 3 days'",
        "log.retention '{}' 不是时长（如 '7 days' 或 '1 week, 3 days'）",
    ),
    (
        "Ignored {} from ./.env: a folder's .env may not set programs, paths, endpoints, proxies, home folders or credential files; set them in your environment or in {}",
        "已忽略 ./.env 中的 {}：文件夹里的 .env 不能设置程序、路径、端点、代理、主目录或凭据文件；请在环境变量或 {} 中设置",
    ),
    (
        "./markitai.json in this folder: model {} names its own address, so its key reference {} is not read from the environment; write the key in the entry, or keep this configuration in your own config file or pass it with -c",
        "当前文件夹的 ./markitai.json：模型 {} 指定了自己的地址，因此不会从环境变量读取它的密钥引用 {}；请把密钥写进该条目，或把这份配置放进你自己的配置文件，或用 -c 指定",
    ),
    (
        "./markitai.json in this folder: connection {} names its own address, so its key reference {} is not read from the environment; write the key in the entry, or keep this configuration in your own config file or pass it with -c",
        "当前文件夹的 ./markitai.json：连接 {} 指定了自己的地址，因此不会从环境变量读取它的密钥引用 {}；请把密钥写进该条目，或把这份配置放进你自己的配置文件，或用 -c 指定",
    ),
    (
        "./markitai.json in this folder: {} names its own address, so its key reference {} is not read from the environment; write the key in the entry, or keep this configuration in your own config file or pass it with -c",
        "当前文件夹的 ./markitai.json：{} 指定了自己的地址，因此不会从环境变量读取它的密钥引用 {}；请把密钥写进该条目，或把这份配置放进你自己的配置文件，或用 -c 指定",
    ),
    (
        "{} not sent to {}: Anthropic API {} rejects temperature, top_p and top_k",
        "未向 {1} 发送 {0}：Anthropic API {2} 不接受 temperature、top_p 和 top_k",
    ),
    (
        "Certificates in SSL_CERT_FILE/SSL_CERT_DIR not read: {}",
        "未能读取 SSL_CERT_FILE/SSL_CERT_DIR 中的证书：{}",
    ),
    (
        "Local OCR: downloading {} ({} MB, once) from the official PaddleOCR mirror into {}; `markitai doctor --fix` installs the models ahead of time.",
        "本地 OCR：正在从 PaddleOCR 官方镜像下载 {}（{} MB，只下载一次）到 {}；`markitai doctor --fix` 可以提前安装这些模型。",
    ),
    (
        "Local OCR: the portable engine's models could not be downloaded; reading with macOS Vision instead (the first Vision use of this program takes about 20 seconds). `markitai doctor --fix` installs the models.",
        "本地 OCR：无法下载便携引擎的模型，改用 macOS Vision 识别（本程序第一次使用 Vision 约需 20 秒）。`markitai doctor --fix` 可以安装这些模型。",
    ),
    (
        "Python extras recommendations (this build is native Rust) is not implemented in this Rust development build; see docs/cli.md",
        "这个原生 Rust 版本没有 Python extras 推荐；见 docs/cli.md",
    ),
    (
        "Cannot write to standard output: {}",
        "无法写入标准输出：{}",
    ),
    (
        "Cannot write the JSON document {}: {}",
        "无法写入 JSON 文档 {}：{}",
    ),
    (
        "Cannot write the batch index {}: {}",
        "无法写入批量索引 {}：{}",
    ),
    (
        "File logging failed; conversion output has been preserved",
        "写入日志文件失败；转换结果已保留",
    ),
    // Reading a URL. The HTTP client's own chain stays in the parentheses.
    (
        "error sending request{}",
        "请求失败（error sending request{}）",
    ),
    (
        "error decoding response body{}",
        "无法读取响应内容（error decoding response body{}）",
    ),
    (
        "error following redirect{}",
        "重定向失败（error following redirect{}）",
    ),
    (
        "request or response body error{}",
        "读取响应失败（request or response body error{}）",
    ),
    (
        "Fetch target refused: a private or local address (this fetch may reach public addresses only)",
        "已拒绝抓取：这是私有或本地地址（这次抓取只能访问公网地址）",
    ),
    (
        "HTTP {} for {}: the page may have been removed or is not public",
        "HTTP {}：{} 可能已被删除或不公开",
    ),
    (
        "HTTP {} for {}: the site refused access; it may block automated clients or need a login",
        "HTTP {}：{} 拒绝访问；该网站可能屏蔽自动化客户端，或需要登录",
    ),
    (
        "HTTP {} for {}: rate limited; try again later",
        "HTTP {}：{} 限制了访问频率；请稍后再试",
    ),
    (
        "HTTP {} for {}: the site had a server error",
        "HTTP {}：{} 的服务器出错",
    ),
    (
        "No strategy could read the page: {}",
        "所有抓取策略都没能读取这个网页：{}",
    ),
    (
        "The {} service could not be reached: {}",
        "无法连接 {} 服务：{}",
    ),
    ("HTTP {} from the {} service", "{1} 服务返回 HTTP {0}"),
    (
        "The {} service received HTTP {} from the site: {}",
        "{} 服务从网站收到 HTTP {}：{}",
    ),
    (
        "The {} service received HTTP {} from the site",
        "{} 服务从网站收到 HTTP {}",
    ),
    // History.
    ("Recorded in history: {}", "已记入历史：{}"),
    ("Could not record history: {}", "无法记入历史：{}"),
    (
        "History output unavailable ({}): {}",
        "历史中的输出不可用（{}）：{}",
    ),
    (
        "History omitted {} image metadata records without an asset in this archive.",
        "历史省略了 {} 条在此归档中没有对应图片的图片元数据。",
    ),
    // Resuming a batch and the output it owns.
    (
        "--resume refused: the saved recovery state was made with different options: {}. Run with the saved options to resume, or omit --resume to start a fresh batch that replaces the saved progress.",
        "已拒绝 --resume：保存的恢复状态使用了不同的选项：{}。请用保存时的选项续跑，或去掉 --resume 重新开始一个批量转换（会替换已保存的进度）。",
    ),
    (
        "Recovery state contains an item inside a Numbers directory package; preserve this state and use a fresh output directory without --resume to convert the package as one document",
        "恢复状态里有位于 Numbers 目录包内的项目；请保留这份状态，换一个新的输出目录并去掉 --resume，把整个包作为一个文档转换",
    ),
    ("Invalid recovery state: {}", "恢复状态无效：{}"),
    (
        "Recovery state scope mismatch: {}",
        "恢复状态的范围不匹配：{}",
    ),
    (
        "Another process owns this recovery checkpoint",
        "另一个进程正在使用这个恢复检查点",
    ),
    (
        "Recovery journal sequence error: {}",
        "恢复日志的序号有误：{}",
    ),
    (
        "Recovery journal line limit reached; later events were not replayed",
        "恢复日志达到行数上限；之后的事件没有重放",
    ),
    (
        "Recovery journal contains invalid UTF-8; later events were not replayed",
        "恢复日志含有无效的 UTF-8；之后的事件没有重放",
    ),
    (
        "Malformed recovery journal JSON was skipped",
        "已跳过格式错误的恢复日志 JSON",
    ),
    (
        "Invalid recovery journal event; later events were not replayed",
        "恢复日志事件无效；之后的事件没有重放",
    ),
    (
        "Recovery journal sequence is duplicate, reversed or incomplete; replay stopped",
        "恢复日志的序号重复、倒序或不完整；已停止重放",
    ),
    (
        "Native recovery journal references an uncheckpointed item; replay stopped",
        "恢复日志引用了检查点中没有的项目；已停止重放",
    ),
    (
        "Native journal event cannot be replayed over a legacy base; replay stopped",
        "恢复日志事件不能在旧版检查点上重放；已停止重放",
    ),
    (
        "Invalid recovery journal mutation; later events were not replayed",
        "恢复日志的变更无效；之后的事件没有重放",
    ),
    (
        "Recovery journal byte limit reached; later events were not replayed",
        "恢复日志达到大小上限；之后的事件没有重放",
    ),
    ("Output publication failed: {}", "发布输出失败：{}"),
    ("Invalid output claim: {}", "输出占用记录无效：{}"),
    (
        "Another process owns the requested output member",
        "另一个进程正在写这个输出文件",
    ),
    ("Output ownership mismatch: {}", "输出归属不一致：{}"),
    (
        "Output coordination cleanup incomplete: {}",
        "输出协调文件没有清理完：{}",
    ),
    // Provider Batch.
    (
        "Unfinished provider batch {} retains this input scope ({}); collect or resolve it before starting another run",
        "未完成的供应商批量任务 {} 仍占用这个输入范围（{}）；请先收集或处理它，再开始新的运行",
    ),
    (
        "Provider batch {} is retained. Collect it later with:",
        "供应商批量任务 {} 已保留。之后用下面的命令收集：",
    ),
    (
        "Re-supply the original model configuration when collecting; inline configuration is not written into this command.",
        "收集时请再次提供原来的模型配置；内联配置不会写进这条命令。",
    ),
    (
        "--llm-batch-timeout exceeds this platform's supported clock range",
        "--llm-batch-timeout 超出了这个平台支持的时钟范围",
    ),
    (
        "Python markitai left batch {} pending in this output directory; collect it with:",
        "Python markitai 在这个输出目录里留下了未完成的批量任务 {}；用下面的命令收集：",
    ),
    (
        "An unfinished provider batch owns this output directory; collect or resolve it before preparing another submission",
        "这个输出目录里有未完成的供应商批量任务；请先收集或处理它，再准备新的提交",
    ),
    (
        "Provider status could not be read: {}",
        "无法读取供应商的状态：{}",
    ),
    (
        "Provider Batch enhancement is pending; collect with: {}",
        "供应商批量增强尚未完成；收集命令：{}",
    ),
    (
        "Provider Batch enhancement is pending; collect it before retrying.",
        "供应商批量增强尚未完成；请先收集再重试。",
    ),
    (
        "Provider Batch collection could not finish; saved outputs and observed usage are retained.",
        "供应商批量收集没有完成；已保存的输出和记录的用量都保留着。",
    ),
    (
        "--llm-batch-collect requires the original output directory with -o",
        "--llm-batch-collect 需要用 -o 指定原来的输出目录",
    ),
    (
        "Provider batch {} was already collected by Python markitai.",
        "供应商批量任务 {} 已由 Python markitai 收集。",
    ),
    (
        "Provider batch {} was already collected.",
        "供应商批量任务 {} 已经收集过。",
    ),
    (
        "Collected provider batch {}: {} observed requests; known priced subtotal ${}.",
        "已收集供应商批量任务 {}：记录到 {} 个请求；已知价格的小计 ${}。",
    ),
    (
        "Provider Batch collection stopped; previously recorded observations are retained: {}",
        "供应商批量收集已停止；之前记录的结果都保留着：{}",
    ),
    (
        "Provider Batch collection stopped; recorded usage and result evidence are retained: {}",
        "供应商批量收集已停止；记录的用量和结果都保留着：{}",
    ),
    (
        "Base Markdown changed after Batch submission; outputs were preserved",
        "提交批量任务后基础 Markdown 被改动；输出已保留",
    ),
    (
        "Provider Batch enhancement failed; base Markdown retained without an automatic live request: {}",
        "供应商批量增强失败；保留基础 Markdown，没有自动改用实时请求：{}",
    ),
    (
        "Python batch {} ended with status {}; there are no results to collect. Run the conversion again to submit a new batch",
        "Python 批量任务 {} 以状态 {} 结束，没有可收集的结果。请重新运行转换以提交新的批量任务",
    ),
    (
        "Python batch {} was submitted to {}; configure that model to collect it",
        "Python 批量任务 {} 提交给了 {}；请配置该模型后再收集",
    ),
    (
        "Python batch has no document request to collect (only image analyses of documents Python enhanced live)",
        "Python 批量任务里没有可收集的文档请求（只有 Python 已实时增强的文档的图片分析）",
    ),
    (
        "Imported Python markitai batch {} ({} documents); its state files were left unchanged.",
        "已导入 Python markitai 批量任务 {}（{} 个文档）；它的状态文件保持不变。",
    ),
    (
        "A Python batch base changed while it was imported; nothing was adopted",
        "导入期间 Python 批量任务的基础文件被改动；没有接管任何内容",
    ),
    (
        "Configuration differs from the frozen model and endpoint",
        "配置与冻结时的模型和端点不同",
    ),
    (
        "Base Markdown changed; frozen Batch submission was not continued",
        "基础 Markdown 已改动；没有继续冻结的批量提交",
    ),
    (
        "Submission evidence retained at {}. Use --llm-batch --resume -o to reconcile an uncertain submission.",
        "提交记录保留在 {}。用 --llm-batch --resume -o 核对结果不确定的提交。",
    ),
    (
        "No matching remote batch was observed; this does not prove that creation failed",
        "没有看到匹配的远程批量任务；这不能证明创建失败",
    ),
    (
        "Multiple matching remote batches were observed; choose and verify an exact batch ID",
        "看到多个匹配的远程批量任务；请选择并核对确切的批量任务 ID",
    ),
    (
        "Remote reconciliation was incomplete; the uncertain submission remains retained",
        "远程核对没有完成；结果不确定的提交仍保留着",
    ),
    (
        "{}. No new batch was submitted. Resume later, or use --llm-batch-collect ID -o with this output directory.",
        "{}。没有提交新的批量任务。请稍后续跑，或对这个输出目录使用 --llm-batch-collect ID -o。",
    ),
    (
        "Frozen Batch resume requires -o",
        "续跑冻结的批量任务需要 -o",
    ),
    (
        "No unfinished frozen provider batch was found in this output directory",
        "这个输出目录里没有未完成的冻结供应商批量任务",
    ),
    (
        "Several pending Python markitai batches are in this output directory ({}); collect each with --llm-batch-collect ID -o",
        "这个输出目录里有多个未完成的 Python markitai 批量任务（{}）；请用 --llm-batch-collect ID -o 分别收集",
    ),
    (
        "Frozen Batch submission requires enabled LLM processing",
        "冻结的批量提交需要启用 LLM 处理",
    ),
    (
        "No unfinished frozen submission can be bound to this batch ID",
        "没有可以绑定到这个批量任务 ID 的未完成冻结提交",
    ),
    (
        "Manual batch binding requires a locally uncertain submission",
        "只有本地结果不确定的提交才能手动绑定批量任务",
    ),
    (
        "Another process is using this provider batch store",
        "另一个进程正在使用这个供应商批量任务存储",
    ),
    (
        "An unfinished provider batch already owns part of this output family; collect or resolve it before submitting again",
        "未完成的供应商批量任务已占用这组输出的一部分；请先收集或处理它，再重新提交",
    ),
    (
        "Multiple unfinished provider batches require explicit collection",
        "有多个未完成的供应商批量任务，需要分别指定收集",
    ),
    (
        "A rejected provider submission cannot be resumed; retain its evidence and start a new run",
        "被拒绝的供应商提交不能续跑；请保留它的记录并开始新的运行",
    ),
    // MCP and subscription sign-in.
    (
        "Cannot establish MCP stdio session",
        "无法建立 MCP 标准输入输出会话",
    ),
    ("MCP session failed", "MCP 会话失败"),
    (
        "The official subscription login process could not be started",
        "无法启动官方订阅登录程序",
    ),
    (
        "Subscription interactive login delegation is not supported on this platform",
        "这个平台不支持委托订阅的交互式登录",
    ),
    (
        "Secret entry is unsupported in this terminal; use an env: reference with config set",
        "这个终端不支持输入密钥；请用 config set 设置 env: 引用",
    ),
    // The service's own diagnostics.
    (
        "Serve: ignoring the saved access tokens ({}); no user token works until an admin creates or revokes one.",
        "服务：忽略已保存的访问令牌（{}）；在管理员新建或吊销令牌之前，所有用户令牌都不可用。",
    ),
    (
        "Serve: could not save the access tokens: {}",
        "服务：无法保存访问令牌：{}",
    ),
    (
        "Serve: could not delete the models of the revoked access token \"{}\" ({}); the next start deletes them.",
        "服务：无法删除已吊销访问令牌“{}”的模型（{}）；下次启动时会删除。",
    ),
    (
        "Serve: history retention sweep failed: {}",
        "服务：清理过期历史失败：{}",
    ),
    ("Serve task failed: {}", "服务任务失败：{}"),
    (
        "Serve: skipped unreadable history job {}: {}",
        "服务：跳过无法读取的历史任务 {}：{}",
    ),
    (
        "Serve: ignored invalid stored attempt diagnostics",
        "服务：忽略了保存的无效尝试诊断",
    ),
    (
        "Serve: ignored invalid stored rerun failure",
        "服务：忽略了保存的无效重跑失败记录",
    ),
    (
        "Serve: history persistence failed: {}",
        "服务：保存历史失败：{}",
    ),
    ("Serve: page preview failed: {}", "服务：页面预览失败：{}"),
    (
        "Serve: rejected upload cleanup failed: {}",
        "服务：清理被拒绝的上传失败：{}",
    ),
    (
        "Serve: rejected job cleanup failed: {}",
        "服务：清理被拒绝的任务失败：{}",
    ),
    (
        "Serve: rejected job cleanup task failed: {}",
        "服务：清理被拒绝任务的后台任务失败：{}",
    ),
    (
        "Serve: ignoring a user token's saved models ({}); it has none until it saves one.",
        "服务：忽略一个用户令牌已保存的模型（{}）；在它重新保存之前没有模型。",
    ),
    (
        "Serve: could not save a user token's models: {}",
        "服务：无法保存用户令牌的模型：{}",
    ),
    (
        "Serve: could not delete the saved models of a revoked access token ({}): {}",
        "服务：无法删除已吊销访问令牌保存的模型（{}）：{}",
    ),
    (
        "Serve: could not save a user token's conversion defaults: {}",
        "服务：无法保存用户令牌的转换默认值：{}",
    ),
    (
        "Serve: a user token's models could not be combined ({}); its conversions run without models.",
        "服务：无法合并一个用户令牌的模型（{}）；它的转换不使用模型。",
    ),
    ("Serve: {}", "服务：{}"),
];
/// The pieces of `english` between its `{}` marks matched against `text`;
/// the variable parts in order. The first piece must start the text and the
/// last must end it.
pub(super) fn captures<'a>(template: &str, text: &'a str) -> Option<Vec<&'a str>> {
    let pieces: Vec<&str> = template.split("{}").collect();
    let (first, rest) = pieces.split_first()?;
    let mut remaining = text.strip_prefix(first)?;
    let mut found = Vec::new();
    for (index, piece) in rest.iter().enumerate() {
        let last = index + 1 == rest.len();
        let at = if last {
            remaining.strip_suffix(piece).map(|head| head.len())?
        } else {
            remaining.find(piece)?
        };
        found.push(&remaining[..at]);
        remaining = &remaining[at + piece.len()..];
    }
    remaining.is_empty().then_some(found)
}

pub(super) fn fill(template: &str, parts: &[&str]) -> String {
    // `{}` takes the next part; `{N}` names one, for Chinese that orders
    // them differently.
    let mut out = String::new();
    let mut next = 0;
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let Some(close) = after.find('}') else {
            out.push_str(&rest[open..]);
            return out;
        };
        let index = match &after[..close] {
            "" => {
                next += 1;
                next - 1
            }
            digits => digits.parse().unwrap_or(usize::MAX),
        };
        out.push_str(parts.get(index).copied().unwrap_or_default());
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    out
}

/// The reasons configuration validation gives after a key (`llm.enabled must
/// be a boolean`), and their Chinese. A leading space joins the key and the
/// Chinese; a reason may carry one `{}`.
const CONFIG_REASONS: &[(&str, &str)] = &[
    ("must be one of: {}", " 必须是以下之一：{}"),
    ("must be at least {}", " 不能小于 {}"),
    ("must be at most {}", " 不能大于 {}"),
    ("must be null", " 必须为 null"),
    ("must be a string", " 必须是字符串"),
    ("must be a boolean", " 必须是布尔值（true 或 false）"),
    (
        "must be an integer representable in 64 bits",
        " 必须是 64 位范围内的整数",
    ),
    ("must be a finite number", " 必须是有限的数值"),
    ("must be an array", " 必须是数组"),
    ("must be an object", " 必须是对象"),
    ("must be a JSON object", " 必须是 JSON 对象"),
    (
        "overrides must be a JSON object",
        "的覆盖设置必须是 JSON 对象",
    ),
    ("is required", " 是必填项"),
    ("is not a known configuration key", " 不是已知的配置键"),
    ("is not a configuration model", " 不是配置中的模型"),
    ("does not exist", " 不存在"),
    ("does not match an allowed type", " 的类型不符合要求"),
    ("must not be empty if set", " 设置后不能为空"),
    ("contains an invalid fetch strategy", " 含有无效的抓取策略"),
    ("contains duplicate strategies", " 含有重复的策略"),
    ("has an empty path component", " 含有空的路径段"),
    ("has an invalid array index", " 的数组下标无效"),
    ("requires an array index", " 需要数组下标"),
    ("has an out-of-range array index", " 的数组下标超出范围"),
    ("cannot be traversed", " 不能再向下访问"),
];

/// `key reason` from configuration validation, when the subject is a key
/// path (lowercase first, no spaces: `llm.model_list[0].model_name`) or
/// `Configuration` itself.
fn config_reason(text: &str) -> Option<String> {
    let (key, reason) = text.split_once(' ')?;
    let key_like = key == "Configuration"
        || (key.starts_with(|c: char| c.is_ascii_lowercase())
            && key
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '[' | ']' | '-')));
    if !key_like {
        return None;
    }
    let key = if key == "Configuration" {
        "配置"
    } else {
        key
    };
    CONFIG_REASONS.iter().find_map(|(english, chinese)| {
        let parts = captures(english, reason)?;
        Some(format!("{key}{}", fill(chinese, &parts)))
    })
}

fn known(text: &str) -> Option<String> {
    RULES
        .iter()
        .find_map(|(english, chinese)| {
            let parts = captures(english, text)?;
            Some(fill(chinese, &parts))
        })
        .or_else(|| config_reason(text))
}

/// The Chinese for a diagnostic's text (without its `Error:`/`Warning:`
/// label), or the text itself when it is not a known message. An item's line
/// is `name: reason`; the reason is translated and the name kept.
pub(super) fn localize(text: &str) -> String {
    if let Some(chinese) = known(text) {
        return chinese;
    }
    let mut from = 0;
    while let Some(at) = text[from..].find(": ") {
        let split = from + at;
        if let Some(chinese) = known(&text[split + 2..]) {
            return format!("{}: {chinese}", &text[..split]);
        }
        from = split + 2;
    }
    text.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_messages_keep_their_variable_parts() {
        assert_eq!(
            localize("Input file not found: a b.pdf"),
            "找不到输入文件：a b.pdf"
        );
        assert_eq!(
            localize("Unsupported file format: '.xyz'. Supported extensions: .md .pdf."),
            "不支持的文件格式：'.xyz'。支持的扩展名：.md .pdf。"
        );
        assert_eq!(
            localize("Cannot create output directory a/b: a is not a directory"),
            "无法创建输出目录 a/b：a 不是目录"
        );
    }

    #[test]
    fn an_item_name_is_kept_and_its_reason_translated() {
        assert_eq!(
            localize(
                "broken.docx: The file appears to be damaged or truncated (Native: zip (os error 2))"
            ),
            "broken.docx: 文件似乎已损坏或被截断（Native: zip (os error 2)）"
        );
        assert_eq!(
            localize("a: b: Input file not found: c: d"),
            "a: b: 找不到输入文件：c: d"
        );
    }

    #[test]
    fn the_merged_pdf_page_line_is_translated_for_one_page_and_for_ranges() {
        assert_eq!(
            localize(
                "scan.pdf: PDF pages 1-3, 5: native text was not recovered (no text layer); OCR is required to read them. Run again with --ocr."
            ),
            "scan.pdf: PDF 第 1-3, 5 页：未能提取文字（no text layer）；需要 OCR 才能读取。请加上 --ocr 重新运行。"
        );
        assert!(
            localize(
                "PDF page 2: native text was not recovered (x); OCR is required to read it. Run again with --ocr."
            )
            .starts_with("PDF 第 2 页")
        );
    }

    #[test]
    fn unknown_text_is_not_touched() {
        for text in [
            "connection refused (os error 111)",
            "Input file not found",
            "x: provider said no",
            "",
        ] {
            assert_eq!(localize(text), text);
        }
    }

    #[test]
    fn every_template_has_the_same_number_of_variable_parts_in_both_languages() {
        // `{}` in order, or `{N}` naming each part once.
        let parts = |template: &str| {
            let mut found = Vec::new();
            let mut next = 0;
            for piece in template.split('{').skip(1) {
                let name = &piece[..piece.find('}').unwrap()];
                found.push(if name.is_empty() {
                    next += 1;
                    next - 1
                } else {
                    name.parse::<usize>().unwrap()
                });
            }
            found.sort_unstable();
            found
        };
        for (english, chinese) in RULES.iter().chain(CONFIG_REASONS) {
            assert_eq!(parts(english), parts(chinese), "{english}");
        }
    }

    #[test]
    fn chinese_may_order_the_parts_its_own_way() {
        assert_eq!(
            localize(
                "temperature, top_k not sent to anthropic/claude-haiku-5-5: Anthropic API 2026-10-21 rejects temperature, top_p and top_k"
            ),
            "未向 anthropic/claude-haiku-5-5 发送 temperature, top_k：Anthropic API 2026-10-21 不接受 temperature、top_p 和 top_k"
        );
    }

    #[test]
    fn configuration_reasons_follow_the_key_they_are_about() {
        assert_eq!(
            localize("llm.enabled must be a boolean"),
            "llm.enabled 必须是布尔值（true 或 false）"
        );
        assert_eq!(
            localize("output.on_conflict must be one of: skip, overwrite, rename"),
            "output.on_conflict 必须是以下之一：skip, overwrite, rename"
        );
        assert_eq!(
            localize("llm.model_list[0].litellm_params is required"),
            "llm.model_list[0].litellm_params 是必填项"
        );
        assert_eq!(
            localize("Configuration overrides must be a JSON object"),
            "配置的覆盖设置必须是 JSON 对象"
        );
        // Only a key path is a subject: a sentence that happens to end the
        // same way is left alone.
        for text in [
            "Model is required",
            "the model is required",
            "OCR is required",
        ] {
            assert_eq!(localize(text), text);
        }
    }
}
