# Rust CLI 开发版

`markitai` 与 `mkai` 使用同一个 Rust 核心。当前版本是 `1.3.0-dev`。
安装见[快速开始](quickstart.md#1-install)，完整参数以 `markitai --help` 和
`markitai <子命令> --help` 为准；参考版差异见[兼容合同](compatibility.md)。

```sh
markitai report.docx                         # Markdown 到 stdout
markitai report.docx --pure                  # 不生成 frontmatter
markitai report.docx -o out/                 # out/report.docx.md
markitai report.docx -o chosen.md --json      # 产物写文件，stdout 为 JSON
markitai ./documents -o out/ -j 4 -g '**/*.pdf'
markitai links.urls -o out/ --url-concurrency 3
markitai ./documents -o out/ --resume
markitai https://example.com/article -s static -o out/
markitai scan.png --ocr --no-llm -o out/
markitai -c isolated.json doctor --json
markitai serve --port 3600 --no-open
markitai mcp
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
| `-c/--config PATH`、`--config-json JSON` | 选择配置文件、叠加本次临时配置 |
| `-p/--preset minimal\|standard\|rich` | minimal 关闭增强；standard 开 LLM/alt/desc；rich 再开截图；均不开 OCR |
| `--llm`、`--ocr`、`--screenshot` | 模型增强、本地识别、页面图像；各有 `--no-*` 反向开关 |
| `--alt`、`--desc`、`--keep-base` | 图片说明、描述及保留基础 Markdown；需 LLM |
| `--profile rag\|obsidian\|okf` | 选择[输出形态](output.md) |
| `--slide-markers`、`--no-slide-markers` | 保留或关闭最终 Markdown 的幻灯片编号注释；默认保留，显式参数覆盖 `output.slide_markers` |
| `-s/--strategy` | URL 抓取策略；远程策略/授权见[抓取](fetch.md) |
| `-b/--backend native\|cloudflare` | 文件后端；Cloudflare 使用自己的账户并上传支持的文件 |
| `--no-remote-fetch` | 禁止第三方抽取服务；不阻断源 URL 请求、浏览器访问或已启用的模型请求 |
| `--screenshot-only` | 隐含截图；普通网页无 LLM 时只保存图像，PDF 保留 Markdown；URL pure 优先 |
| `--llm-batch`、`--llm-batch-timeout`、`--llm-batch-collect` | OpenAI 目录批处理提交、等待、收集；见[供应商 Batch](provider-batch.md) |
| `-j`、`--url-concurrency`、`--llm-concurrency` | 文件转换、URL 转换、模型在途请求三个独立上限 |
| `-g/--glob`、`--max-depth` | 批处理筛选与扫描深度；glob 必须加引号，`!` 排除 |
| `--resume` | 目录或 URL 列表断点续跑；单项输入不支持 |
| `--dry-run` | 预览目标、冲突改名与跳过，不转换、不创建输出目录 |
| `--no-cache`、`--no-cache-for` | 绕过缓存读取，成功结果仍可写缓存 |
| `--record-history` | 保存供工作台查看的[历史](history.md) |
| `-q/--quiet`、`-v/--verbose` | 仅错误、更多诊断；quiet 不会弹出远程授权询问 |
| `--log-level` | 文件日志级别；不启用尚未配置的日志目录 |
| `-I/--interactive` | 终端转换向导；不能与 `--json` 或子命令合用 |

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
读系统显示语言再回退用户区域语言，其他系统默认英文。变量来自进程环境、
当前目录 `.env`、`MARKITAI_HOME/.env`，前者优先。

```sh
MARKITAI_LANG=zh markitai --help
MARKITAI_LANG=zh markitai doctor
```

帮助、主要诊断与批处理摘要有中英文；参数名、JSON 字段、路径、核心原始错误、
文件日志保持原文。`Error:`、`Warning:` 等标签固定，方便检索。
交互编辑器、向导、订阅认证及部分服务输出仍为英文；工作台独立切换语言。

### 中文帮助

中文按终端显示宽度折行，上限 78 列；参数名与代码字面量保留完整。
窄终端会把用法标题与参数行分开，普通宽度英文帮助保持既有布局。
解析器自身的参数错误仍为英文。

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
  重试未完成项并发现新增输入。恢复比对保存的功能开关（LLM/OCR/截图/alt/desc）、
  输入输出路径及目录扫描深度/glob/文件并发，不是整个配置文件的比较。更换模型、
  prompt 或输出 profile 后要重新转换时，使用新的输出目录并省略 `--resume`。
  Unix 首次中断停止派发并等待在途工作，再次中断终止所启动的外部进程组。
  详见[恢复状态](state-storage.md)。
- 批量默认在输出目录 `.markitai/reports/` 写[报告](reports.md)，单项默认不写。
  `output.report` 可覆盖。报告失败会返回非零，已完成产物仍保留。
- 终端会显示进度，管道、`--json`、`-q` 或 `TERM=dumb` 不显示。
  Windows 使用控制台 VT，无法启用时用空格清除。stdout 下游提前关闭时安静退出；
  其他写入错误仍失败。
- `config list/get/path/validate/set/edit` 查看、验证或编辑[配置](configuration.md)。
  默认隐藏密钥；`set/edit` 原子保存，不保存临时 `--config-json`。
  `init --yes` 检测本地环境并生成默认关闭 LLM 的配置，不发模型请求或保存密钥明文。
- `doctor [--json]` 检查本地依赖与配置，不发模型请求。
  `doctor --fix` 可下载浏览器与缺少的 Paddle 模型；普通 doctor 不下载。
  缺少可选组件不等于程序不可用；配置要求的组件失败才导致退出 1。
- `cache stats [--json]` 查看缓存，`cache clear -y` 清理模型/页面缓存；
  `cache spa-domains` 管理已学习的浏览器域名。详见[缓存](cache.md)。
- `serve` 提供工作台、REST、SSE、下载和历史；即使本机访问，API 默认也需令牌。
  使用启动时含 fragment token 的浏览器地址。详见[服务](serve.md)。
- `mcp` 提供 `convert_document`、`convert_url`、`batch_convert`、`job_status`。
  使用绝对路径，stdout 专供协议，配置及结果见[MCP](mcp.md)。

转换 `--json` 顶层为 `version`、`ok`、`error`、`batch`、`items`、`totals`。
逐项读取 `items[].status`（completed/failed/skipped/pending）、`output`、`error`、
`warnings` 与 `skip_reason`；按字段名解析，不依赖显示顺序。普通批量中断/续跑
只列本次已经结束的条目，未启动项不会凭空出现在 `items`，也不应把它当全历史。

退出码：0 成功或正常跳过；1 单项/运行错误；2 用法错误或供应商 Batch 尚未完成；
10 普通批量有失败项；130/143 Unix 中断。参数错误只有 stderr，没有 JSON envelope。

## 明确的迁移缺口

已支持的扩展名不保证任意文档都能完整读取。格式与保真边界见[格式](formats.md)。
本地 OCR/PDF 后端按[平台](quickstart.md#platform-support)选择；Office 页面需要
LibreOffice，缺少时通常保留正文并警告，`--screenshot-only` 则失败。
Numbers 支持表格及目录包，完整画布、截图与 OCR 仍未支持。

尚未提供 Anthropic Batch API、旧 Python Batch 状态导入、终端内联图片。
`image.stdout_fetch_external` 因此无运行效果；其他兼容配置的状态见
[配置说明](configuration.md#配置键的运行状态)。OpenAI Batch、官方订阅运行时与
远程 URL/Cloudflare 后端已有实现，外部账户与网络可用性需另行满足。
性能结论仅适用于对应测量样本，不能当作所有输入的统一加速倍数。

## 文件日志

日志默认关闭。设置 `log.dir` 或 `MARKITAI_LOG_DIR` 后启用；
`--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL` 覆盖文件级别。
`log.format`/`MARKITAI_LOG_FORMAT` 可选 `text` 或逐行 `json`。
日志不进入 stdout，语言固定英文，URL 与已知凭据在诊断中脱敏。
不记录正文、模型完整请求响应或全量配置。

默认 `log.rotation="10 MB"`、`log.retention="7 days"`；按整条记录轮转，
只清理本程序命名的过期普通日志。日志写入失败返回非零，但不删除已完成产物。
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
