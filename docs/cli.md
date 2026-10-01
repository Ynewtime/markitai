# Rust CLI 开发版

`markitai` 与 `mkai` 使用同一个 Rust 核心。当前这是迁移中的开发版本，完整参考契约见 [compatibility.md](compatibility.md)，尚未实现的请求必须返回明确错误。命令帮助通过 `markitai --help` 查看。

```sh
markitai note.txt
markitai note.txt --pure
markitai document.docx -o output/
markitai note.txt -o chosen.md --json
markitai ./documents -o output/ -j 4 --glob '**/*.txt'
markitai urls.urls -o output/ --url-concurrency 3
markitai urls.urls -o output/ --resume
markitai https://example.com/article -s static -o output/
markitai -c isolated.json --config-json '{"output":{"on_conflict":"skip"}}' note.txt -o output/
markitai note.txt -o output/ --config-json '{"output":{"report":true}}'
markitai serve --host 127.0.0.1 --port 3600 --no-open
markitai mcp
markitai note.txt -o output/ --log-level DEBUG --config-json '{"log":{"dir":"./logs","format":"json"}}'
markitai config edit
markitai init --local
markitai -I
markitai -c isolated.json doctor --json
```

## 参数帮助与移除提示

`-h/--help` 说明输出、JSON、配置优先级、三类并发、缓存绕过、OCR/截图、
恢复与日志的实际行为；未实现的取值（`-s cloudflare`、`-b cloudflare`）直接标明限制。
选项按输出、配置、LLM/OCR/截图、URL 抓取、批处理、缓存与图片、消息与日志分组，
末尾列出 preset 含义、常用示例和退出状态。成对开关只列出主开关，帮助文本写明反向
拼写（如 `--llm` 注明 `--no-llm`）；反向开关照常可用，只是不单独占一行。
`config`、`cache`、`init`、`doctor`、`serve` 及其子命令的每个参数都有说明，并附示例。
没有输入时显示同一份帮助。帮助布局不要求与 Python Rich 输出逐字节一致。

旧参数 `--playwright`、`--defuddle`、`--static`、`--jina`、`--cloudflare`
已移除；使用时在 stderr 指向 `-s <策略>`，退出 2。这个提示只说明替代拼写，
不承诺尚未实现的远端策略可用。`--kreuzberg` 没有替代开关，提示 RTF 已原生转换。
这些名字不重新注册为兼容别名，不出现在帮助中；`--名字=value` 同样给出迁移提示。
错误发生在读取配置或转换之前，`--json` 也不会因此输出 envelope。

移除检查尊重 `--` 终止符、取值选项的参数及 attached value，避免把同名字面路径
当作旧开关；例如 `--output=--static` 是路径值。原版简单扫描每个 argv token，
原生这里有意保留正常参数/路径解析；其他无效用法仍由 Clap 拒绝。

## 终端语言

`doctor`、`cache stats/clear/spa-domains` 与 `config path/validate` 的终端文字可显示中文或英文，范围对应参考版本做了本地化的命令。语言选择沿用参考规则：非空的 `MARKITAI_LANG` 单独决定；未设置或为空时先读 `LANG`，再读 `LC_ALL`。取值以 `zh` 开头（不分大小写）为中文，其他取值和未设置均为英文；非空但不是 `zh` 的 `MARKITAI_LANG`（如 `en`、`fr`）不会再回退到 `LANG`。参考先读 `LANG` 后读 `LC_ALL`，与 POSIX 中 `LC_ALL` 优先的约定相反，这里保持参考顺序。变量读取与配置选择相同：进程环境优先，其次是当前目录 `.env` 和 `MARKITAI_HOME/.env`。

```sh
MARKITAI_LANG=zh markitai doctor
LANG=zh_CN.UTF-8 markitai cache stats
```

翻译的是面向人的句子：标题、配置来源、状态词和“配置要求”标注、总结、浏览器修复进度、缓存条目与清理结果、清理确认与取消。doctor 每项的名称、说明和安装提示与 `--json` 字段同源，仍为英文；核心给出的错误原因、路径和大小单位不变。`--json`、`config path` 找到文件时打印的路径、`config list/get/set`、`--help`、参数错误、日志与退出码在两种语言下逐字节相同。参考同样未翻译的 `init`、`-I` 向导、`config edit`、转换进度与批量摘要、`auth`、`serve`/`mcp` 终端输出保持英文；网页工作台另有自己的语言切换。

## 已实现的命令行为

- 单文件和 URL 未给 `-o` 时输出 Markdown 到 stdout；`--pure` 去除 frontmatter；提供 `-o chosen.md` 可选择准确文件名。
- stdout 模式下文档引用的图片与页面截图默认保存到 `MARKITAI_HOME/assets/blobs/`（未设置时为 `~/.markitai/assets/blobs/`），按内容哈希命名、重复运行复用同一文件，Markdown 中的引用改为可直接打开的绝对 `file://` URI；内嵌 `data:image/…` 也一并保存。`image.stdout_persist=false` 时保留指向未写出文件的 `.markitai/...` 相对引用，并在 stderr 警告一次；个别图片保存失败时转换仍成功，这些引用保持相对并警告原因。详见[图片：stdout 中的图片](images.md#images-on-stdout)。
- 单项默认不显示进度，但写入文件后在 stderr 打印一行 `Wrote <路径>`（冲突改名时可见实际文件名）；跳过时说明原因和下一步（图片需要 `--ocr` 或 `--llm`，已存在输出受 `output.on_conflict=skip` 约束）。`-q` 只保留错误；`--json` 的 stdout 只有 envelope。缺少模型的 `--llm` 失败附一行配置提示。
- 转换前检查 `-o`：已存在的非目录路径、最近的已存在祖先不是目录或不可写时，直接以退出 1 报告路径与原因，不再交给输出归属或恢复状态层用内部术语报错；`--dry-run` 只警告并照常列出目标。单个本地输入不存在或无读取权限时不创建输出目录，后者报 `Cannot read <路径>: …`。
- 目录中没有受支持文件（或全部被 `--glob` 排除）时在 stderr 说明后退出 0；目录/URL 列表的 `--dry-run` 在 stderr 汇总将转换的文件和 URL 数。`.urls` 中被跳过的条目按行号（JSON 数组按条目序号）警告，不回显条目文本。`--alt/--desc` 在未启用 LLM 时警告其无效。
- `--json -o` 输出 version 1.0 envelope，字段名与参考接口一致。运行失败仍有机器可读条目；参数错误只输出 stderr 并退出 2（未给 `-o` 时说明 stdout 已被 JSON 占用）。有模型请求的条目额外带 `pricing`（`priced_requests`、`unpriced_requests`、`cost_status`、`pricing_snapshots`，观察不完整时另有 `incomplete_request_observations`），`totals.pricing` 汇总这些条目；没有模型用量时两者省略，定价规则见 [定价](pricing.md)。没有条目时 totals 的 `cost_usd`/`duration_s` 为 0，不输出 `-0.0`。
- 目录递归转换保留相对路径，支持重复 `--glob`、`!` 排除和 `--max-depth`，使用受限线程并发；目录中的 `.urls` 也会发现。同批任务预留独立名称，即使冲突策略为 overwrite/skip，也不会让两个新结果互相覆盖；大小写匹配按输出文件系统探测。URL 列表支持文本和 JSON 两种格式。
- `--llm-concurrency` 限制整个运行中的在途模型请求，目录中的文件与 URL 共用该上限；重试等待和缓存命中不占请求槽。文件与 URL 的转换并发仍分别由 `-j` 和 `--url-concurrency` 控制。
- 目录/URL 列表普通转换项失败退出 10；单项失败退出 1，成功退出 0；状态存储致命错误退出 1，中断退出 130/143。`--quiet` 仍显示错误。批量结束时 stderr 摘要依次给出完成数、耗时与费用、按原因分组的跳过项、失败/未完成项示例（错误在上方逐项列出）、全部因缺少模型而失败时的一行配置提示，以及输出目录。
- 四种输入模式支持持久 JSON 报告。`output.report` 为 null 或省略时，目录/URL 列表默认启用，单文件/URL 默认关闭；true/false 显式覆盖。报告写入输出目录的 `.markitai/reports/`，各模式的字段和计数差异见 [reports.md](reports.md)。
- 报告发布失败保留已完成文件及 stdout JSON 条目，并退出非零；报告不替代 stdout envelope。stdout 转换、dry run、无可恢复状态的空目录和失败/跳过的单项不生成报告；批量部分失败仍可生成报告。报告的 skip 冲突策略保留已有报告。
- Unix 目录/URL 列表每次保存恢复状态；`--resume` 合并新发现任务、保留完成项并重试未完成项。输出归属凭证保护隐式重试，旧状态按普通冲突策略升级，升级前私有保存原始 base/journal 及存在性；备份仅是状态回退材料，不撤销输出或模型请求。首次 Ctrl-C 停止派发、同步状态并等待在途转换，退出 130；再次中断先终止本进程启动的订阅运行时、Chromium 和 LibreOffice 进程组后立即退出。单输入转换收到 SIGINT/SIGTERM/SIGHUP 时同样先终止这些进程组，再按原信号默认方式退出；它们位于独立进程组，终端中断本身不会到达。详见 [恢复状态](state-storage.md) 与 [输出归属](output-ownership.md)。
- `--record-history` 将本次实际处理项保存到隔离 home 下的 `serve/jobs/`，包含独立的最终文档、资产和兼容元数据；归档失败只警告，stdout/dry-run/中断不归档。开关覆盖环境和配置，详见 [历史归档](history.md)。
- 配置优先级由核心解析；根级 `-c` 和 `--config-json` 对子命令同样生效。布尔参数支持显式否定；重复正反开关以最后一个为准，preset 名称按小写查找，应用后显式参数覆盖。没有 INPUT/子命令且未指定 `-I` 时显示帮助并退出 0，包括只给转换选项的情况；参数本身非法仍退出 2。
- `config list/get/path/validate/set/edit` 可用；list 支持 JSON/YAML/table；默认隐藏凭据。set/edit 原子更新配置，保留未知字段，不写入临时 `--config-json` 内容。
- `config edit` 是终端中的导航编辑器：`/关键字` 模糊搜索、数字选择当前页设置、`n/p` 翻页，也可输入完整配置 key。输入值后按同一配置 schema 验证并立即保存；无效值不落盘。空输入保留当前值，`:empty` 写空字符串，`:cancel` 返回，`q` 退出。列表隐藏凭据，外部修改配置后拒绝覆盖，要求重新打开。
- 编辑器遍历标量设置和嵌套配置，跳过数组、字典、prompts、presets、domain_profiles；复杂值仍用 `config set`。界面使用按行导航，不复刻参考版全屏方向键界面。stdin/stderr 非终端时明确拒绝，不等待脚本输入。Unix 敏感值输入暂时关闭终端回显；配置编辑中 Ctrl-C/SIGTERM 恢复回显后退出 130/143，不保存未确认值。其他平台敏感输入明确拒绝，可使用 `config set` 保存 `env:VARIABLE` 引用。
- `init [--local|-o path]` 提供位置选择与已有文件 update/overwrite/keep；默认 keep。新建或覆盖后在 stderr 给出下一步（转换命令、`doctor`、启用 LLM 或补充 API key 的方法），stdout 仍只有一行结果。`init --yes` 无提示创建配置，已有配置只追加新发现的模型、保留已有模型和其他字段，重复运行无变化。`-o` 指向已有目录时写入该目录的 `markitai.json`。坏配置仅允许交互明确覆盖，自动更新报错并保留原字节。
- 初始化只检测当前核心支持的 API 环境配置和 `MODEL`，不发模型请求，不保存凭据明文，生成配置默认关闭 LLM。不检测订阅登录、不安装运行时、不创建参考版 `.env` 模板；可用环境变量或现有隔离 home 下的 `.env` 配置凭据。交互编辑/初始化与其他新功能的验证状态以调度中心为准。
- `-I/--interactive` 引导选择文件、目录或 URL、输出目录、LLM、alt/desc/pure、OCR 与截图，确认后复用普通转换、报告和批量流程。显式配置、preset 和命令行开关成为向导默认值；Enter 保留当前有效值，y/n 显式启用/关闭，摘要显示最终效果。这比参考版仅发送勾选的正向开关更明确，也保留了参考版重新启动 CLI 时会丢失的显式配置。缺少模型时可设置仅用于本次运行的 API 模型/环境引用/隐藏密钥、重试检测或关闭 LLM；不写配置，不调用订阅 CLI。提示与摘要写 stderr，不改变转换 stdout。q、EOF、拒绝确认退出 0；Unix 向导提示期间 Ctrl-C 同样退出 0并恢复密钥输入回显，转换开始后使用普通中断行为。非 Unix 中断状态依终端平台，尚未实机验证。`-I --json`、向导与子命令混用、非终端输入均退出 2。
- `doctor [--json]` 恢复参考的顶层检查字典：`playwright/libreoffice/rapidocr/anydoc/serve/llm-api/vision-model/vlm-ocr`，配置订阅模型时追加对应 SDK/auth 项。每项包含 `name/description/status/message/install_hint`，状态为 `ok/warning/missing/error`，按检查需要附 `path/optional/models`；保留旧 key，但描述实际原生后端，不声称安装了同名 Python 包。Chromium 实际执行私有 profile 的 about:blank/CDP 启停，LibreOffice 有界执行隔离的 `--version`；这不证明网页/文档完整兼容。OCR 检查平台 API 可用性，不预热模型；LLM 只核对本地配置、环境引用和路由资格，不联系 provider。
- doctor 文本输出首行后注明所用配置文件（或内建默认值及 `markitai init` 提示），末行汇总：配置要求的检查是否就绪、可选项有几项未就绪；JSON 结构不变。缺少可选组件退出 0；活跃模型缺少环境引用或不可用、配置明确要求的浏览器不可启动时退出 1，weight=0 的模型不阻断。显式 playwright、截图以及带 HTTP credentials、非空 cookies 或额外 HTTP headers 的 auto 抓取要求浏览器，显式 static 不因该凭据字段而要求浏览器。VLM OCR 项显示 `MARKITAI_NO_VLM_OCR` 的实际选择。`doctor --fix` 在浏览器不可用时，通过原生安装器下载官方 Chrome headless shell，私有启动验证后原子启用；已有浏览器正常则不下载。显式 `MARKITAI_BROWSER_EXECUTABLE` 会阻止自动替换该路径。详见[浏览器安装](browser-installation.md)。`--json --fix` 返回 2，Python 专属 `--suggest-extras` 明确未支持。未配置 `llm.model_list` 时，LLM 项按 `MODEL`/供应商 API key 报告 `--llm` 实际会用的模型（有 key 为 ok，只有 `MODEL` 而缺 key 为 warning），不再报 missing；设置了 `MARKITAI_BROWSER_EXECUTABLE` 而浏览器不可用时，浏览器项直接指出该变量，并说明 `--fix` 不会替换它。
- `serve` 启动原生 REST 服务，支持提交文件/URL、任务快照与 SSE、结果/资产/ZIP 下载及持久历史；根路径提供内嵌的浏览器工作区（转换、预览、历史与模型设置）。沿用 host、port、no-open、no-auth、allowed-host 参数；接口与安全边界见 [REST 服务](serve.md) 和 [浏览器工作区](web-ui.md)。
- `mcp` 通过标准输入/输出提供 `convert_document`、`convert_url`、`batch_convert`、`job_status` 四个工具。配置与文件输出沿用同一核心，批处理任务保存在当前 MCP 进程内；协议和结果边界见 [MCP 服务](mcp.md)。
- `--dry-run` 仅枚举输入和目标，不调用转换器、不创建输出目录。
- `--compress/--no-compress` 映射共享图片处理配置；未启用 LLM/OCR 的独立图片返回 `image_only` 跳过状态，不写空文档。
- 非 pure 本地文档及文本 URL 的 LLM 结果可逐块跨进程复用；`--no-cache` 跳过读取但仍写入成功结果，`--cache` 清除此绕过设置，不强制启用已禁用的缓存。`--no-cache-for` 接受逗号分隔的 glob，JSON 条目的 `cache_hit/llm_cache_hit` 反映实际命中。
- 静态 HTML/文本抓取支持独立页面缓存：无验证头时按 TTL 复用，有 ETag/Last-Modified 时发送条件请求。`fetch_cache_hit` 记录直接复用或 304 命中；`cache_hit/llm_cache_hit` 仍只表示 LLM 缓存。显式 `-s static` 与配置文件中的默认策略使用独立缓存作用域。
- `cache stats [--json] [-v] [--limit N]` 查看 LLM 与抓取缓存；不存在的数据库不会因查看而创建。LLM 详细列表最多返回 1,000 条。`cache clear [-y]` 清理两个缓存，未给 `-y` 时需要在终端确认。清理前检查两个数据库，后续部分失败会明确报告并退出非零。`cache spa-domains [--json] [--clear]` 查看或清空学到的浏览器渲染域名，`cache clear --include-spa-domains` 同时清空它们。文本统计按单复数显示条目数，大小以 B/KiB/MiB 显示；`--json` 保留精确字节数。

- `--ocr` 在 macOS 使用原生 Vision 识别独立图片，读取原始像素；`MARKITAI_NO_VLM_OCR` 可选择本地识别后仅发送文字增强。语言和平台限制见 [本地 OCR](ocr.md)。
- 本地 PDF 和静态/自动下载的 PDF 支持 macOS 逐页 OCR 与截图。下载字节直接复用，支持重定向和无后缀 URL；PDF 仅截图模式仍保留 Markdown，URL `pure` 优先级不变。详见 [PDF 媒体](pdf-ocr.md) 与 [抓取](fetch.md)。
- `-s playwright` 直接控制本机 Chromium；`auto` 可在静态质量失败时回退浏览器。URL 截图支持完整长页分块，`--screenshot-only` 不隐含 LLM，历史归档保留所有分块。未给 `-o` 的无 LLM 仅截图模式使用配置输出目录或当前目录。依赖和边界见 [浏览器](browser.md)。

## 明确的迁移缺口

仍有迁移缺口：`-s cloudflare` 抓取与 `-b cloudflare` 文件后端、非 Unix 断点恢复、Anthropic Batch API 与旧版 Python Batch 状态导入、终端内联图片显示（因此 `image.stdout_fetch_external` 可设置但无作用）。`-s jina`/`-s defuddle` 远程抽取、OpenAI Batch API、经官方运行时的订阅登录（`auth <provider> login`）和 serve 的浏览器工作区均已实现。pure 按参考行为绕过 LLM 缓存；文本、独立图片与分页视觉请求的缓存范围分别见 [LLM 处理](llm.md)，不能将一次命中理解成所有输入已支持缓存。其余未实现的开关/命令请求会失败并说明原因。Office 演示与文字文档可通过可选的独立 LibreOffice 安装获得全页截图和 OCR 补充，详见 [Office 渲染](office-rendering.md)；XLS/XLSX/ODS 支持每张完整工作表一页，包含隐藏和空表；Numbers 完整画布和其他平台本地 OCR/PDF 渲染仍未完成；未实现的选项只在遇到相关格式或图片时拒绝，不应阻断纯文本转换。独立栅格图片、完整多页 TIFF 和 SVG 可经 LLM 视觉模型读取。alt/desc 已接入真实图片引用、结构化分析及 images.json 合并，详见[图片分析](image-enrichment.md)；需要启用 LLM。rich/standard preset 仍不是对所有格式可用的完整模式。

持久报告、可选历史导出和 Unix 批量恢复已实现；单项和非 Unix 恢复仍明确拒绝。普通非 Unix 转换保留既有行为，但尚未完成实机验证。混合目录分别应用文件与 URL 并发上限。URL 列表的自定义文件名只允许一个安全 basename；旧实现的名称清理细节尚待配对验收。帮助使用原生 Clap 布局，不复刻 Rich 框线；非 Unix 进程中断清理及全部非 ASCII 终端行为仍需专门测试。

具体文件格式支持取决于核心当前实现，注册参考扩展名不意味着全部可用。性能与质量对比未完成前，不承诺生产替代、完整旧版兼容或具体加速比。

## 文件日志

转换的文件日志默认关闭（`log.dir=null`）。设置目录后，默认文件级别为 INFO；
`--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL` 覆盖 `log.level`，仅影响文件。
`--quiet`/`--verbose` 沿用终端策略，日志不进入 stdout Markdown 或 `--json` envelope。
配置、缓存、服务等子命令保留自己的输出，不因根级 `--log-level` 启用转换日志。

`MARKITAI_LOG_DIR` 的非空值覆盖目录，合法的 `MARKITAI_LOG_FORMAT=text|json` 覆盖格式。
路径支持 `~`；显式隔离 `MARKITAI_HOME` 时，`~/.markitai/...` 路径跟随隔离 home。
每次运行独立建立 `markitai_<日期>_<时间>_<微秒>_<进程号>.log`，Unix 新文件权限为 0600。
持久报告的 `log_file` 指向本次首个日志文件；轮转文件使用同一前缀和序号。

文本日志逐行写入，换行转义；JSON 日志每行含 `ts`、`lvl`、`src`、`msg`，
其中 `src="cli"`。记录配置加载、实际文档开始/完成/跳过/失败、转换 warnings、
CLI 诊断和退出状态；不记录文档正文、模型请求响应或全配置，也不截获所有核心库和第三方内部日志。
线程共享同一个受锁保护的 sink，整行写入，退出时 flush/sync。
URL 的用户信息、全部 query/fragment、可疑路径段和已知配置/环境凭据在日志及 CLI 诊断前脱敏；
这是诊断脱敏，不改变输出文档、报告或 stdout JSON 的既有数据合同。

`log.rotation` 支持正数 B/KB/MB/GB/KiB/MiB/GiB，默认 `10 MB`；
`log.retention` 支持秒、分钟、小时、天、周，默认 `7 days`。
超阈值在下一整条记录之前切换文件，单条记录不会拆开。
启动时只清理符合 Markitai 日志命名的过期普通文件，跳过符号链接及被其他原生运行锁定的文件。
参考 Loguru 的时间点轮转、组合时长等其他表达式会明确报错，不会静默忽略。
日志初始化失败使转换报错；运行中写入或最终 flush 失败会在 stderr 报错并退出非零，
保留已完成的文档和已发出的唯一 stdout JSON，不追加第二个 envelope。

## Numbers 目录包

扩展名为 `.numbers` 的目录按一个文档处理，扩展名匹配不区分 ASCII 大小写。
单包支持 stdout、指定输出文件、报告和可选历史；默认输出为 `名称.numbers.md`。
`--resume`、`--glob`、`--max-depth` 仍是批处理选项，不因包在磁盘上是目录而改变
单文档规则。转换向导将包归入 file 选择。

扫描普通父目录时，包路径本身参与 glob、深度和文件数限制，内部文件不参与任务
发现。被 glob 排除、缺少 IWA 或格式损坏的包都不会下钻成一组 TXT、图片或 URL
列表任务；无效包只产生一次转换失败。恢复状态和目录报告用包的相对路径作为一个
文件身份，保留原来的输出冲突和完成项跳过规则。

现代目录包复用有界 Numbers 表格读取器。这里不增加旧 XML 容器或完整画布的能力；
截图与 OCR 请求仍明确拒绝。边界与夹具来源见 [Numbers](numbers.md)。

## 最近一次转换的用量诊断

`--json` 中有实际模型用量的条目增加可选的
`diagnostics.last_attempt`，包含 `operation:"convert"`、`status:"done"|"error"`、
原有字符串或 null 的 `error`，以及 `usage` 的 requests/input_tokens/output_tokens/
cost_usd/by_model。失败条目的既有 `llm_usage` 和 `cost_usd` 同时投影这次已观察用量；
原来的错误字符串、状态和退出码不变。一次请求已记录但 token 为零仍保留诊断；
没有可读取用量的响应、转换前校验失败和未派发任务则省略新字段，不能据此认定免费。
当前零美元成本也不代表已获得供应商定价。

批量 `--resume` 仍是 convert 操作。新尝试清除旧诊断，终止后保存自己的观察值；
重试不把旧失败用量再加一次。已完成且未重跑的记录保留其保存诊断，但 stdout
仍只列本次实际处理的条目。报告使用独立的 [terminal_diagnostics](reports.md#latest-attempt-diagnostics)
区域，历史元数据保留同一结构。它们不是累计账本：进程强制退出、不可解析的响应
或状态尚未持久化都会留下观察缺口。

## 开发验证

```sh
cargo test -p markitai-cli
cargo run -p markitai-cli --bin markitai -- --help
cargo build --release -p markitai-cli --bin markitai
```

CLI 测试使用临时工作目录与 `MARKITAI_HOME`，不修改用户 `~/.markitai`。手动测试也应将 `MARKITAI_HOME` 指向项目的忽略目录，并按需只读传入测试凭据。

报告的验证范围与剩余工作见 [报告验证说明](reports.md#validation-scope-and-remaining-work)，最新门禁记录见 [调度中心](CONTROL.md)。
