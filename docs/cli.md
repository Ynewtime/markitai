# Rust CLI

`markitai` 与 `mkai` 使用同一个 Rust 核心，版本见 `markitai --version`。
安装见[快速开始](quickstart.md#1-install)，完整参数以 `markitai --help` 和
`markitai <子命令> --help` 为准；参考版差异见[兼容合同](compatibility.md)。

```sh
markitai report.docx                         # Markdown 到 stdout
markitai report.docx --pure                  # 不生成 frontmatter
markitai report.docx -o out/                 # out/report.docx.md
markitai report.docx -o chosen.md --json      # 产物写文件，stdout 为 JSON
markitai report.docx -f json                 # 文档 JSON 到 stdout（字段稳定、带版本）
markitai ./documents -o out/ -j 4 -g '**/*.pdf'
markitai links.urls -o out/ --url-concurrency 3
markitai ./documents -o out/ --resume
markitai https://example.com/article -s static -o out/
markitai scan.png --ocr --no-llm -o out/
markitai -c isolated.json doctor --json
markitai serve --port 3600 --no-open
markitai mcp
markitai update --check                       # 有新版本时退出 10
```

## 开发包与 Windows 入口

选择与系统和架构匹配的开发包，保持程序、许可和离线指南一起安装。
CLI 不需要 Python、Node.js 或 Go；语言绑定另需对应运行时。
Unix single-binary tar 含一个主程序及相对链接 `mkai`、`markitai-mcp`；
Windows ZIP 含直接可执行的 `markitai.exe`、`mkai.exe`、`markitai-mcp.exe`。

```powershell
.\markitai.exe --version
.\markitai.exe .\note.txt --no-llm -o .\output\
.\markitai-mcp.exe --help
```

MCP 别名直接启动 stdio 服务，支持自己的 `--help` 和 `--version`。
OCR 模型不随归档提供；按[本地 OCR](ocr.md)提前准备。
离线入口是根目录 `llms.txt`、`llms-full.txt` 及精选 `docs/` 指南；
其他专题链接需要完整仓库。

## 参数帮助与移除提示

| 选项 | 行为 |
|---|---|
| `-o/--output PATH` | 指定输出目录或准确的 `.md` 文件名 |
| `--json` | 转换输出 version 1.0 JSON envelope；必须同时指定 `-o`，不能与 `--dry-run` 合用 |
| `-f/--format markdown\|json` | `json` 把每个文档写成带 `schema_version` 的 JSON：无 `-o` 时输出到 stdout，有 `-o` 时写在 Markdown 旁（`-o x.json` 指定确切文件名），批量另写 `markitai.index.json`；见 [JSON 输出](json-output.md) |
| `-c/--config PATH`、`--config-json JSON` | 选择配置文件、叠加本次临时配置 |
| `-p/--preset minimal\|standard\|rich` | minimal 关闭增强；standard 开 LLM/alt/desc；rich 再开截图；均不开 OCR |
| `--llm`、`--ocr`、`--screenshot` | 模型增强、本地识别、页面图像；各有 `--no-*` 反向开关 |
| `--alt`、`--desc`、`--keep-base` | 图片说明、描述及保留基础 Markdown；需 LLM |
| `--profile rag\|obsidian\|okf` | 选择[输出形态](output.md) |
| `--text-only`、`--no-text-only` | 图片和 PDF 只写出识别到的文字：无 frontmatter、标题、图片链接、描述和代码围栏；见[图片](images.md#text-only) |
| `--page-markers`、`--no-page-markers` | 保留或关闭最终 Markdown 中的 PDF 页码与幻灯片编号注释；默认关闭，显式参数覆盖 `output.page_markers` |
| `-s/--strategy` | URL 抓取策略；远程策略/授权见[抓取](fetch.md) |
| `-b/--backend native\|cloudflare` | 文件后端；Cloudflare 使用自己的账户并上传支持的文件 |
| `--no-remote-fetch` | 禁止第三方抽取服务；不阻断源 URL 请求、浏览器访问或已启用的模型请求 |
| `--screenshot-only` | 隐含截图；普通网页无 LLM 时只保存图像，PDF 保留 Markdown；URL pure 优先 |
| `--llm-batch`、`--llm-batch-timeout`、`--llm-batch-collect` | OpenAI 与 Anthropic 目录批处理提交、等待、收集；见[供应商 Batch](provider-batch.md) |
| `-j`、`--url-concurrency`、`--llm-concurrency` | 文件转换、URL 转换、模型在途请求三个独立上限 |
| `-g/--glob`、`--max-depth` | 批处理筛选与扫描深度；glob 必须加引号，`!` 排除 |
| `--resume` | 目录或 URL 列表断点续跑；单项输入不支持 |
| `--dry-run` | 预览目标、冲突改名与跳过，不转换、不创建输出目录；启用 LLM 却没有可用模型时给出警告（实际运行会以退出码 1 停止） |
| `--no-cache`、`--no-cache-for` | 绕过缓存读取，成功结果仍可写缓存 |
| `--record-history` | 保存供工作台查看的[历史](history.md) |
| `-q/--quiet`、`-v/--verbose` | 仅错误、更多诊断；quiet 不会弹出远程授权询问 |
| `--log-level` | 文件日志级别；不启用尚未配置的日志目录 |
| `-I/--interactive` | 终端转换向导；不能与 `--json` 或子命令合用 |

`--help` 末尾打印本次构建的身份：版本、提交（工作区有未提交改动时标为
`(local changes)`，非检出目录为 `unknown`）、目标与构建类型、以及构建时间。
时间按**本机时区**显示并带 UTC 偏移（如 `2026-10-05 21:15:06 +08:00`），复制到
别处也不会产生歧义。`SOURCE_DATE_EPOCH` 可固定该时间，使可复现构建报告被复现
构建的身份。
`-h` 与 `--version` 保持原样（`--version` 仍是一行，版本号在最后一个词）。

布尔正反开关重复时最后一个生效。preset 先应用，显式参数再覆盖。
`-s static`/`playwright` 在本机抓取，但仍访问输入网址；`--no-remote-fetch`
不是断网开关。离线用本地文件、native 后端并关闭 LLM；OCR 模型或 Office
应用需事先准备。`fetch.remote_consent=ask` 在非终端或 `--quiet` 下按拒绝处理。
旧 `--playwright`、`--static`、`--jina`、`--defuddle`、`--cloudflare` 改为
`-s <策略>`；`--kreuzberg` 已移除，RTF 原生读取。旧拼写给出迁移提示并退出 2。
`--` 后可放与开关同名的路径。

## 终端语言

`MARKITAI_LANG=zh` 选中文，其他非空值选英文；未设置时依次读
`LC_ALL`、`LC_MESSAGES`、`LANG`，跳过 `C`/`POSIX`。没有可用值时，Windows
读系统显示语言（`GetUserDefaultUILanguage`）再回退用户区域语言，其他系统默认英文；
测试与脚本要固定语言时设 `MARKITAI_LANG`。变量来自进程环境、
当前目录 `.env`、`MARKITAI_HOME/.env`，前者优先。由 locale 或系统选中的中文
在输出无法显示中文时回退为英文：Unix 上字符集不是 UTF-8 的 locale（如 `zh_CN.GBK`），
或代码页既非 CJK 也非 UTF-8 的旧版 Windows 控制台（Windows 的管道与文件输出是 UTF-8，
不回退）；`MARKITAI_LANG=zh` 照常生效。

```sh
MARKITAI_LANG=zh markitai --help
MARKITAI_LANG=zh markitai doctor
```

帮助、参数错误、诊断、提示与批处理摘要都有中英文，一次运行里的每一行同属一种语言：
`doctor`、`auth`、`cache`、`config`（含交互编辑器）、`init` 与 `-I` 向导、`serve` 的
启动信息与运行提示、`mcp` 的 stderr，以及运行级错误（找不到输入、未知预设、缺少输出目录、
配置校验等）、供应商批量任务与续跑的提示和转换器常见的失败原因（文件损坏、格式不支持、
未配置模型、PDF 页需要 OCR、网页无法读取）在终端上以中文显示。括号内解析器、操作系统、
供应商返回的技术细节、参数名、JSON 字段、路径、文件日志保持原文，`--json` 与报告里的
`error` 也是英文；stdout 上的数据（转换结果、JSON、`config list/get`、`--dry-run` 的逐项
列表）不翻译。少数内部一致性错误（输出占用与恢复记录的存储损坏等）和不常见的转换器
警告仍只有英文。文件日志里的 `Error:`、`Warning:` 等标签固定，方便检索；终端上由标记
代替（见下文）。工作台独立切换语言。

## 终端样式

终端输出与 Python markitai 1.3.0 使用同一套样式。标记说明每行的含义，颜色只是
重复它：`✓` 成功（绿）、`!` 警告（黄）、`✗` 错误（红）、`→` 下一步或正在进行
的操作（青）、`•` 事实或列表项（灰）、`○` 有意跳过（灰）、`?` 等待输入（青）、
`◆` 标题（青）；节标题加粗不着色，要复制的命令单独一行、青色、不折行。条目缩进
两列，结论行（如 `doctor` 与批处理的最后一行）从第 0 列开始，前面空一行。

- 终端上长行在正文下方缩进折行；管道、文件与 CI 输出每条消息一行，没有状态行。
- `NO_COLOR`（任意非空值）关闭颜色，标记保留；`FORCE_COLOR` 在管道中也输出颜色；
  `TERM=dumb` 既无颜色也不折行，并改用 ASCII 标记。
- 无法显示这些标记的输出改用 `+ ! x > - o ? *`：Unix 上字符集不是 UTF-8 的 locale
  （`C`/`POSIX` 或未设置按 UTF-8 处理，与 Python 相同），以及不能处理转义序列的
  Windows 控制台（Windows 10 1511 之前的版本，或勾选了“使用旧版控制台”）。
  Windows 10 及以后的控制台主机、Windows Terminal、VS Code、mintty 与 SSH 终端
  会话都显示 Unicode 标记。
- Windows 上的管道与文件输出仍是 UTF-8，标记跟随所附控制台的输出代码页：Windows
  PowerShell 按这个代码页（`[Console]::OutputEncoding`）解码程序的输出，所以只有
  代码页为 65001（UTF-8）时使用 Unicode 标记，其他代码页（默认的 437、936 等）
  使用 ASCII 标记，不会出现 `Γ£ô` 这样的乱码。没有控制台的进程（服务、分离启动
  的进程）没有代码页可跟随，保持 Unicode 标记。转圈字符与状态行的省略号与标记
  采用同一判断。见[故障排查](troubleshooting.md)。
- 只用 8 种普通颜色加灰色 90，浅色与深色主题都清晰；经典蓝底 Windows PowerShell
  窗口把暗黄显示成近白色，在那里警告、错误、成功与青色改用亮色槽。

### 中文帮助

中文按终端显示宽度折行，上限 78 列；参数名与代码字面量保留完整。
窄终端会把用法标题与参数行分开，普通宽度英文帮助保持既有布局。
参数错误（未知参数或子命令、缺少值、取值无效、参数冲突）同样以中文显示，参数名与取值
保持原样；解析器少见的错误保持英文。

## 已实现的命令行为

- 单文件/URL 默认输出到 stdout；`--pure` 去除生成的 frontmatter。
  stdout 引用的图片默认保存在 `MARKITAI_HOME/assets/blobs/` 并使用本机
  `file://` 链接。要方便传给别人，使用 `-o out/` 保留相对资产路径。
- 已有输出默认改名为 `.v2.md`，不会覆盖。用
  `--config-json '{"output":{"on_conflict":"overwrite"}}'` 或 `skip` 改变策略。
  先用 `--dry-run` 查看实际名称。`-o` 和配置输出路径支持开头的 `~`。
- 没启用 OCR/LLM 的有效图片正常跳过，退出 0，不写空文档。损坏图片仍失败。
  脚本需检查 JSON `status`/`skip_reason`，不能仅把退出 0 当作有产物。
- 目录保留相对路径，默认跳过点文件/目录、`node_modules` 和 `~$` Office 锁文件。
  明写这些名字的正向 glob 可包含它们；直接指定点目录也可处理。
- `.urls` 支持每行 `URL [output-name]` 或 JSON 列表，忽略空行和 `#` 注释；
  自定义名称只能是安全 basename。同批任务预留独立输出名称。
- macOS/Linux/Windows 均支持批量恢复。用原命令加 `--resume`，保留兼容的完成项、
  重试未完成项并发现新增输入。状态及其文件名记录功能开关（`--llm`、`--ocr`、
  `--screenshot`、`--alt`、`--desc`）、输入输出路径及目录扫描深度/glob，以及不是默认值的
  改变输出的设置（`--profile`、`--keep-base`/`--pure`（有无 `--llm` 都算）、`--screenshot-only`、
  `-s`，以及 `-f json`、`--text-only`、`output.page_markers`、`output.wikilinks`），与 Python 版
  markitai 一致（Python 版只在带 `--llm` 时记录 `--keep-base`/`--pure`）；不记录文件并发（`-j`），
  也不是整个配置文件的比较。换一种设置续跑找不到原来的状态，就是另一个批次：退出码 0，
  重新转换各项（已有文件按 `output.on_conflict` 处理）；回到原设置仍找到原来的进度。
  更换模型或 prompt 后要重新转换时，使用新的输出目录并省略 `--resume`。
  Unix 首次中断停止派发并等待在途工作，再次中断终止所启动的外部进程组。
  详见[恢复状态](state-storage.md)。
- 批量默认在输出目录 `.markitai/reports/` 写[报告](reports.md)，单项默认不写。
  `output.report` 可覆盖。报告失败会返回非零，已完成产物仍保留。
- 终端会显示进度，管道、`--json`、`-q` 或 `TERM=dumb` 不显示。
  Windows 使用控制台 VT，无法启用时用空格清除。stdout 下游提前关闭时安静退出；
  其他写入错误仍失败。
- `config list/get/path/validate/set/edit` 查看、验证或编辑[配置](configuration.md)。
  默认隐藏密钥；`set/edit` 原子保存，不保存临时 `--config-json`。
  `init --yes` 检测本地环境（已登录的订阅运行时与 API 密钥）并生成默认关闭 LLM 的配置，不发模型请求或保存密钥明文。
- `doctor [--json]` 检查本地依赖与配置，不发模型请求。
  `doctor --fix` 可下载浏览器与缺少的 Paddle 模型；普通 doctor 不下载。
  缺少可选组件不等于程序不可用；配置要求的组件失败才导致退出 1。
  LLM API 一项会列出模型缺少的密钥变量，以及已设置但没有生效模型用到的供应商密钥（JSON 的 `unused_api_keys`）。
- `cache stats [--json]` 查看缓存，`cache clear -y` 清理模型/页面缓存；
  `cache spa-domains` 管理已学习的浏览器域名。详见[缓存](cache.md)。
- `serve` 提供工作台、REST、SSE、下载和历史；即使本机访问，API 默认也需令牌。
  使用启动时含 fragment token 的浏览器地址。详见[服务](serve.md)。
- `mcp` 提供 `convert_document`、`convert_url`、`batch_convert`、`job_status`。
  使用绝对路径，stdout 专供协议，配置及结果见[MCP](mcp.md)。
- `update` 检查新版本。安装脚本装的副本就地替换：按发行版 `SHA256SUMS` 校验、
  先试启动一次再替换，失败则全部恢复；其他安装方式只给出更新命令。`--check` 只检查，
  `--json` 在 stdout 输出一个对象，`--version X.Y.Z` 安装指定版本（较旧的版本先询问，
  无终端时需 `--yes`）。正在运行的 `serve` 会询问是否重启。安装目录或 `MARKITAI_HOME`
  不属于 root 时，以 root 运行会被拒绝。退出码：0 已是最新或已更新，
  10 有新版本但未在此安装，1 失败，130 中断。
- 终端上的转换结束后，有新版本时每天最多提示一次：`• 有新版本 2.3.0（当前 2.2.0）：markitai update`。
  检查在后台进行，每天最多一次，最多等 2 秒；JSON、stdout 输出、`-q`、非终端、CI、MCP 与源码构建
  不检查。用 `update.check: false` 或 `MARKITAI_NO_UPDATE_CHECK=1` 关闭。

转换 `--json` 顶层为 `version`、`ok`、`error`、`batch`、`items`、`totals`。
逐项读取 `items[].status`（completed/failed/skipped/pending）、`output`、`error`、
`warnings` 与 `skip_reason`；PDF 条目的 `page_sources` 逐页说明文字来源（`text` 文字层、`ocr`、`mixed`，
见 [PDF](pdf.md#where-each-pages-text-came-from)），续跑时此前已完成的条目为空列表。按字段名解析，不依赖显示顺序。
单个文件的 `source` 是文件名；失败条目的 `duration_s` 是它所用的时间，`cost_usd` 与 `llm_usage`
是它已付费的请求。输入不存在或格式不受支持时整次运行被拒绝：`error` 说明原因，`items` 为空。

`--resume` 续跑时，`items` 先列出此前运行已完成的条目（`earlier_run: true`，带状态文件记录的输出、
耗时、警告、费用与用量；旧状态未记录费用时 `cost_usd`、`llm_usage` 为 null，表示未知而非 0），
再列本次处理的条目（`earlier_run: false`）。`totals.cost_usd` 覆盖整个批次，
`totals.this_run_cost_usd` 只算本次。中断时未启动的条目不会出现在 `items`。

退出码：0 成功或正常跳过；1 单项/运行错误；2 用法错误或供应商 Batch 尚未完成；
10 普通批量有失败项；130 中断（Ctrl-C，Windows 也含 Ctrl-Break）；143 终止（SIGTERM，Windows 为关闭控制台）。参数错误只有 stderr，没有 JSON envelope。

## 明确的迁移缺口

已支持的扩展名不保证任意文档都能完整读取。格式与保真边界见[格式](formats.md)。
本地 OCR/PDF 后端按[平台](quickstart.md#platform-support)选择；Office 页面需要
LibreOffice，缺少时通常保留正文并警告，`--screenshot-only` 则失败。
Numbers 支持表格及目录包，完整画布、截图与 OCR 仍未支持。

尚未提供终端内联图片。
`image.stdout_fetch_external` 因此无运行效果；其他兼容配置的状态见
[配置说明](configuration.md#配置键的运行状态)。OpenAI 与 Anthropic Batch、官方订阅运行时与
远程 URL/Cloudflare 后端已有实现，外部账户与网络可用性需另行满足。
性能结论仅适用于对应测量样本，不能当作所有输入的统一加速倍数。

## 文件日志

日志默认关闭。设置 `log.dir` 或 `MARKITAI_LOG_DIR` 后启用；
`--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL` 覆盖文件级别。
`log.format`/`MARKITAI_LOG_FORMAT` 可选 `text` 或逐行 `json`。
日志不进入 stdout，语言固定英文，URL 与已知凭据在诊断中脱敏。
不记录正文、模型完整请求响应或全量配置。

默认 `log.rotation="10 MB"`、`log.retention="7 days"`；按整条记录轮转。
`log.rotation` 可写大小（`10 MB`）、间隔（`1 week`）、本地时间（`00:00`、
`monday at 12:00`、`sunday`）或 `hourly`/`daily`/`weekly`/`monthly`/`yearly`；
`log.retention` 写时长（`7 days`、`1 month`、`1 week, 3 days`），与 Python 版写法相同，
`config validate` 拒绝其他取值。每次运行仍各写自己的文件，轮转只切换该次运行的文件；
保留期只清理本程序命名的过期普通日志。日志写入失败返回非零，但不删除已完成产物。
报告的 `log_file` 指向本次日志；报告和 stdout JSON 保留自己的公开数据合同。

## Numbers 目录包

`.numbers` 目录按一个文档转换，内部成员不作为批量任务；无效包只报一次失败。
支持 stdout、输出文件、报告及历史。其 `--resume`/glob 行为仍按单文件处理。
详见[Numbers](numbers.md)。

## 最近一次转换的用量诊断

有实际模型用量时，JSON 条目附 `diagnostics.last_attempt`，记录 convert 的终态、
错误和本次观察到的请求/token/费用。未观察到用量时省略，不代表免费。
`pricing` 区分已定价、部分定价和未知；`cost_usd` 可能仅是已知小计。
恢复重试不把旧失败用量再次加入；历史完成项可保留原诊断。
这不是跨运行账本，强制退出或不可解析响应可能留下观察缺口。
详见[定价](pricing.md)与[报告诊断](reports.md#latest-attempt-diagnostics)。

## 开发验证

参考[开发指南](development.md)。测试同时隔离
`HOME` 和 `MARKITAI_HOME`，只用项目夹具或自拟样本，不读取真实用户配置或凭据。
