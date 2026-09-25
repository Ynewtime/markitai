# 更新日志

本项目的所有重要变更都会记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，
版本号遵循[语义化版本](https://semver.org/spec/v2.0.0.html)规范。

## [1.2.0]

### 新增

- 新增 `markitai.enable_worker_processes()`：带 `if __name__ == "__main__":` 保护的 Python 脚本调用后，可以和 CLI、`markitai serve`、MCP 服务器一样并行提取 PDF。`MARKITAI_PDF_WORKERS` 设置工作进程数，`0` 或 `1` 表示关闭。
- 新增 `llm.on_failure`，决定 LLM 增强失败时条目怎么算：`"fallback"`（默认）把未增强的 `.md` 作为输出并给出警告；`"fail"` 把条目记为失败（单个输入以 `1` 退出，批量以 `10` 退出）。
- `.urls` 列表支持 `--resume`；各种模式下按 Ctrl-C 都会保存批处理状态，续跑从真正完成的地方继续。
- `--json` 能描述交接出去的 `--llm-batch` 运行：顶层 `batch` 对象给出批次 id 和 `--llm-batch-collect` 命令，新增 `pending` 条目状态与 `pending` 合计，以及逐项 `warnings`。条目的 `cache_hit` 也会如实报告。
- 新增 `cache.fetch_ttl_seconds`（默认 24 小时），没有 ETag/Last-Modified 的已抓取页面会过期。`cache stats`/`cache clear` 覆盖 URL 抓取缓存；`--no-cache-for` 的模式匹配到某个 URL 时，该 URL 不走抓取缓存。
- 可操作的提示在默认模式下也会显示（不再只在 `-v` 下）：未开 `--ocr` 的扫描页、OCR 失败、PDF 隐藏文字、无法渲染的 PPTX 幻灯片、没拍到的 URL 截图。批处理会在汇总里列出。
- `markitai doctor` 在装了 PowerPoint、但 macOS 不允许 markitai 写入其容器目录时给出警告，并说明如何授权。
- Python API 导出 `ConversionError`、`FetchError` 和 `NoModelConfiguredError`（`ValueError` 的子类，启用 LLM 但没有可用模型时抛出）。
- 不开 LLM 时，`--screenshot` 会在基础 `.md` 里以 HTML 注释引用每页 PDF 和每张 PPTX 幻灯片的截图。

### 变更

- **PDF 转换大幅提速。** 长 PDF 以及同时转换的多个 PDF 改由工作进程按页提取，每个进程单线程运行版面模型，输出与原来完全一致。在 18 核的 Mac 上，300 页的 PDF 提速 4.1×，40 个 PDF 的批处理提速 4.1×，240 个文件的混合批处理提速 3.3×；每页所用的 CPU 时间约为原来的三分之一，每个工作进程约占 450 MB 内存。隐藏文字检查和扫描页检查在提取页面的同时进行。
- LLM 批处理让转换和增强重叠进行：文件转换完成后，在等待模型期间就把转换名额让给下一个文件，CLI 批处理和网页工作台都是如此。LiteLLM 在转换输入的同时于后台加载；`--alt`/`--desc` 的视觉模型检查不再拖慢运行的开始，没有视觉模型时以提示的形式报告。
- Python API 和 MCP 服务器加载 LiteLLM 时不再联网获取模型价格表。内存不足时，OCR 会减少同时处理的页数。
- **LLM 增强失败不再悄无声息。** key 无效、服务端报错、超时或模型拒答时，以前会把未增强的文本当作结果。现在条目默认仍算成功，但输出是未增强的 `.md`，并通过警告（stderr、`--json` 的 `warnings`、网页工作台）说明失败原因。把 `llm.on_failure` 设为 `"fail"` 可以让这类条目记为失败。
- 超过单次 LLM 请求长度的文档改为分块增强，不再在 32,000 字符处截断。所需请求数超过 `max_requests_per_document` 剩余额度的文档不做增强，一个请求都不发。
- 找到多个服务商的 key 且没有配置模型时，markitai 会说明请求将分散到哪些模型，以及如何固定为一个。Python API 和 MCP 服务器现在按与 CLI 相同的方式确定模型。
- 输出文件跟随 umask（通常为 `0644`），不再一律是 `0600`；markitai 写出的配置文件和 `.env` 仍保持私有。
- OCR 文本按阅读顺序排版：双栏页面逐栏读取，竖排中日文从右到左读取，能识别倒置的扫描页，间隔排列的行输出为 Markdown 表格。页边印章和侧注不再并入正文行，居中标题和地址块不再被误判为分栏，“标签 值”式表单和编号列表按普通文本输出。
- `--ocr` 在有真实文字层的页面上保留标题、列表、表格和图片，并照常执行隐藏文字清理。不支持的 `ocr.lang` 值会直接报错，RapidOCR 的最低版本改为 `>=3.9`。
- 提取出的图片和页面/幻灯片截图按最终输出名命名（如 `report.pdf.v2-0001-01.jpg`），改名重跑不会再覆盖旧版本引用的文件。
- `-c`/`--config-json` 对子命令生效：`config get/set/path`、`cache` 和 `doctor` 读取、`config set` 写入的都是你指定的文件。`-c` 指向的文件还不存在时，`config set`/`config edit` 会创建它；其他命令遇到不存在的 `-c` 路径直接报错（退出码 `2`）。
- `markitai-mcp` 与 `markitai mcp` 一样加载 `./.env` 和 `~/.markitai/.env`。
- `--screenshot-only` 在文本提取失败时改用截图完成，没拍到截图时算失败；目录批处理中的 `.urls` 文件也遵守该选项。
- 警告会出现在所有入口：网页工作台按条目显示，Python API 放在 `ConversionOutput.warnings` 里，MCP 结果带 `warnings` 字段。
- 文档写明 `MARKITAI_NO_VLM_OCR` 只管 OCR；同时使用 `--screenshot --llm` 时，markitai 会提示页面截图仍会发给视觉模型。
- 网页工作台：删除条目时一并删除其上传原件、图片和截图。从 CLI 记录的文件条目标记为不可重试（重试接口返回 `409`），其中的 URL 条目仍可重试或增强。因服务关闭而中止的条目，错误文案由 `cancelled` 改为 `cancelled (server shutdown)`。
- defuddle 对照语料同步至上游 0.19.4，并移植其提取改动：代码块围栏按内容中的反引号串自动加长而不再转义；`<sub>`/`<sup>` 保留为内联 HTML 并紧贴前后文（`2021<sub>5ya</sub>` 不再被压成 `20215ya`）；邮件式 `Date:` 等带标签行中的日期随标签一起保留；arXiv 隐藏的重复脚注标记被移除；`closed`、旧式 `shadowroot` 属性及嵌套的声明式 Shadow DOM 都能被提取。
- 网页提取结果中移除惰性 `<template>` 片段和 SVG SMIL 动画元素。

### 修复

- 批处理不再把自己的输出目录或 `.markitai/` 资源当作输入，也能识别 `.Docx` 这类大小写混合的扩展名。
- 续跑时重做的条目覆盖自己之前的输出，不再生成 `.v2` 副本。输出名在批次内预留，同名 URL、以及 URL 与同名文件之间不再互相覆盖。
- `--profile rag`/`obsidian` 重跑不再在 `assets/` 里保留旧图；`--record-history` 不再把不同子目录下的同名资源合并覆盖。
- 独立图片的视觉分析失败时，不再无输出却报告成功；图片分析失败时保留原作者的 alt 文本。
- 模型拒答不再被当作文档正文接受并写入缓存，`--pure` 和视觉路径也一样。
- 增强失败后，兜底的 `.md` 是本轮的转换结果；覆盖模式下会删除本条目遗留的旧 `.llm.md`。文档调用失败后不再额外付费做一次清理调用；遇到鉴权错误时，该文档剩余的请求会停止。
- 图片分析失败不再覆盖 alt 文本，也不写入 `images.json`，而是作为条目警告报告（stderr 和 `--json`）。
- 在另一个工作目录下 `--resume`，结果仍写回原来的输出目录。列表里同一个 URL 用不同名字出现两次时，各自保留一份状态；完全重复的行会跳过并给出警告。没有记录输出路径的续跑条目遵守 `--on-conflict`。目录批处理会在 stderr 和 `--json` 中报告 URL 的警告（例如截图没有拍到）。
- 从 URL 下载的图片按条目实际得到的输出名命名（`docs.v2.*`；用 `-o custom.md` 时为 `custom.*`）。批处理中 `--llm --pure --screenshot-only` 和单个 URL 一样读取正文。
- 覆盖只读（`0444`）的输出文件不再失败。
- `--llm-batch`：缓存命中的图片答案不再丢失；单次请求放不下的长文档仍会分析其中的图片。`--resume` 不再重新提交批次尚未收取的文档，而是打印收取命令。提交后按 Ctrl-C 会打印收取命令；`--screenshot-only` 的 URL 不参与增强；收取时沿用提交时的 `--alt`/`--desc`。交接时如果还有失败条目，仍以 `2` 退出，并报告失败数。
- `--llm-batch`：带 `--alt`/`--desc` 的批次收取时不再崩溃；Anthropic 图片请求改用 Messages API 格式；批次使用 `llm.model_list` 中的 `api_key`/`api_base`；不支持的模型池在转换前就被拒绝；提交失败时只输出一行提示（默认是警告，`llm.on_failure = "fail"` 时是错误）；中断或提交失败前已转换的文档在 `--resume` 时会被增强；部分文件转换失败不再导致整批跳过增强。
- 网页按 `<meta charset>` 解码（GBK、Shift_JIS、CP1252 等）；重定向后的相对链接按最终 URL 解析；PDF 和 Office 文档的 URL 改用 markitai 自己的转换器。
- 条件请求返回验证页或 JavaScript 空壳时不再覆盖原有的正常缓存；如果随后的完整抓取也失败，就返回缓存副本，metadata 中带 `stale` 和 `fetch_warning`。较短的纯文本响应不再被误判为单页应用，也只有 HTML 响应才会做验证页检查。
- 一个非法字节不再让整页乱码：按声明的编码（即使声明位于前 1 KiB 之后）或 UTF-8 替换解码；未声明编码的西文文本回退到 CP1252，不再被误判成别的代码页。
- Playwright 在清理 DOM 之前截图，保留全部截图分块，不同查询串使用各自的截图文件，页面加载后卡死会超时退出；截图有单独的时间预算，页面繁忙时仍能拿到正文；同一批次中 URL 使用不同代理时不再关掉共享浏览器；NO_PROXY 和本机地址不走代理，环境变量里设置的 `HTTP(S)_PROXY` 也一样。`--no-cache-for` 匹配主机名时不区分大小写；普通的 `#锚点` 不再让 URL 生成单独的截图文件。
- OCR 失败（图片损坏、为空或过大，语言不受支持）会让条目失败，不再把错误写进输出。多帧 TIFF 逐帧识别，并发 OCR 不再调低共享引擎的阈值，本地 OCR 计为重任务。
- PPTX 的 `--ocr --llm` 遵守 `MARKITAI_NO_VLM_OCR`；PDF/PPTX 的页面渲染图不再计入提取的图片数。
- stdout 输出中的图片链接指向不可变的副本，使用 `--profile rag`/`obsidian` 时也是如此。
- 输出路径含空格或括号时（例如 iCloud 的 `Mobile Documents`）PDF 图片能正确写出；文件名含空格或中文的图片也会做 OCR；同前缀的兄弟 PDF 的图片不再被认领或重新压缩。`--ocr --screenshot` 不再把页面渲染图计入图片数。
- CSV/TSV 中不配对的引号只影响它所在的那一行。多帧 TIFF 逐帧解码。
- 带 UTF-8 BOM 的 `.urls` 文件、带引号的多行 TSV 单元格，以及正文只在 HTML 或 ANSI 部分的 `.msg` 邮件现在都能正确转换。
- 输出目录无法创建时只输出一行错误（`--json` 中带 `error`），不再打印堆栈。MCP 工具会返回失败原因，`batch_convert` 拒绝相对路径。
- 网页工作台：有事件流连接时按 Ctrl-C 能及时关闭，任务仍保留在历史中；输出名和上传名按不区分大小写去重；结果中列出 PDF 图片；重试 CLI 记录的 `--llm` URL 不再写出 `.llm.llm.md`；增强失败时恢复之前的结果，重跑途中关闭服务也一样。同时删除同一条目两次返回 `404`，不再是 `500`。名字只差一个后缀的条目（`notes` 和 `notes.llm`）不再共用输出。等待收取的 `--llm-batch` 条目显示为等待中；“重试全部失败”只重试可以重试的条目。

## [1.1.0] - 2026-09-14

- **综合提速约 6 倍**：网页 URL **5.8×**、Office/PDF **5.6×**、批处理 **6.6×**，由三类等权的本地基准测得，不计 LLM、OCR 和公网时延。
- **更快得到结果**：启动和重复解析的开销都少了；短文和中日韩正文只要第一次就抓全，就不再回退到浏览器。
- **输出质量不变**：正文、元数据、公式和图片沿用原有处理，复杂文档照旧走成熟的回退路径。
- **修复公网 URL 误判**：Clash 等 Fake-IP 环境下，Defuddle 等远程抓取现在能正常工作，私网保护照常生效。
- **改善 X/Twitter 抓取**：允许远程服务时，匿名推文不用启动浏览器就能拿到；遇到 HTTP 错误及时回退，富文本、链接、图片和发布日期都保留下来。
- **保留域名调优**：自定义域名配置只覆盖显式设置的字段，其余内置浏览器参数照常生效。
- **减少重复提示**：远程抓取和 VLM OCR 的提示每位用户只看一次，授权询问和禁用选项照旧有效。

## [1.0.1] - 2026-09-10

### 新增

- `--json` 输出逐项状态、产物、usage、费用及运行级错误；需配合 `-o`，与 `--dry-run` / `--llm-batch-collect` 互斥。
- `--no-remote-fetch` 禁用远程 URL 抽取，`--log-level` 控制已配置的文件日志。
- MCP 支持输出 profile 和有界批量并发（默认 10）；省略 `llm` 时跟随服务器配置。
- 双语网站对比页、搜索与社交元数据，以及文档和 UI 的 CI 检查。

### 变更

- Serve 登录地址改用 `#token=`；config list/get/set 默认脱敏，包括嵌套请求头，显式 `--show-secrets` 才显示原值。
- 网页工作台增加可取消的上传进度、行内下载、清晰的选项与错误反馈，改善无障碍访问，设置和预览按需加载。移除 Tailwind 与无用 UI 代码。
- CLI 进度写入 stderr；帮助与报错明确输出路径、支持格式、可选依赖和已移除参数。

### 修复

- 防止 MCP 同名输出互相覆盖，以及 API 混入残留的 LLM 文件。
- 保留重试和重启后的任务选项，修复提前完成、恢复幽灵行及并发 ZIP 下载冲突。
- 未认证远程任务逐跳校验并固定公网目标，覆盖重定向与浏览器子请求，隔离抓取缓存和浏览器会话。
- Batch 只增强本次转换的文件并正确回填最终产物、usage 与失败状态；RAG/Obsidian 图片、wikilink 和 alt 在 Batch 与历史中保持完整。
- 统一处理非法配置、缓存替换容量、空 URL 列表及错误退出码；发布检查适配 fragment 登录地址。

## [1.0.0] - 2026-09-09

### 新增

- **markitai 现在既是 CLI 也是库**：`markitai.convert("report.pdf")` 及其异步孪生 `aconvert` 返回带类型的 `ConversionOutput`，里面有 markdown、frontmatter、资源与截图路径、逐图分析和用量合计，配置分层则复用 CLI 自己的那一套。暂定：签名与结果字段仍可能变化
- **`markitai mcp` 从 CLI 启动内置 MCP 服务**，这也是官方 MCP Registry 条目（`io.github.Ynewtime/markitai`）使用的形式：`uvx --from "markitai[mcp]" markitai mcp`
- **MCP agent 通过 `mcp` extra 接入 markitai**：`markitai.mcp` 随主 wheel 发布，经 stdio 暴露 `convert_document`、`convert_url`、`batch_convert`、`job_status`。`claude mcp add markitai -- uvx --from "markitai[mcp]" markitai-mcp`
- **`--profile rag|obsidian|okf` 按下游消费者塑造输出**：`rag` 把图片移到可见的 `assets/`（LlamaIndex 的 `SimpleDirectoryReader` 等摄取器会跳过隐藏路径）并改写 PDF 页标记；`obsidian` 提供可选 wikilink；`okf` 将 frontmatter 映射到 Open Knowledge Format。与 `--preset` 正交；不加它输出逐字节不变
- **`--llm-batch` 让整个目录的 LLM 阶段以半价运行**：在单模型的 OpenAI 或 Anthropic 池上走提供商的 Batch API，最长等待 `--llm-batch-timeout`（默认 1 小时）后转交 `--llm-batch-collect`
- **`--llm-batch` 覆盖图片分析与页截图**：`--alt`/`--desc` 与 `--screenshot` 同批提交、同享折扣。页数超过单次调用上限的文档改为实时增强，因为批处理每多一轮就多一次等待
- **网页端可以选输出格式**：与 Preset 并列，并且刻意不随 LLM 开关隐藏。profile 塑造的是输出形状而非增强，纯本地转换同样能带
- **三个限制单文档成本的断路器**：`llm.max_requests_per_document`（默认 50）限制重试倍增；`llm.max_vision_pages_per_document` 在发送前检查，超限文档一分钱不花；`llm.max_cost_per_document_usd` 每次拿到回答后计费，约束该文档后续还能花多少。触发后跳过剩余增强、保留未增强产出。两个成本上限默认关闭
- **旧版 `.doc`/`.ppt` 转换改由 `markitai[legacy]` 提供**：内置 Rust 的 anydoc 后端以毫秒级处理 Office 97-2003，任何平台都不需要 Microsoft Office 或 LibreOffice
- **`--ocr --llm` 现在有名字、有度量、有开关**：这个组合发送的是页图给视觉模型而非运行 RapidOCR，此前无处说明。help、`doctor` 和中英 CLI 指南统称它 VLM-OCR，metadata 记录实际走了哪条路径，`MARKITAI_NO_VLM_OCR=1` 可彻底禁用
- **超长页面截图改为切片而非压缩**：高于 `screenshot.tile_height`（2000px）的页面切成多张全宽 tile，不再被压成一条读不了的长图
- **数学公式以 LaTeX 形式回来了**：每个会看到公式的阶段都要求输出 `$...$` / `$$...$$`。pymupdf 会把行间公式当图片交出，把行内数学揉成会吞掉旁边正文的 markdown 噪声
- **CLI 转换可以选择进入网页端历史**：`--record-history`（或 `MARKITAI_RECORD_HISTORY`、或 `history.record`）把完成的运行记录为 `~/.markitai/serve/jobs/` 下的一个任务
- **转换质量现在有三种度量**：快照护栏冻结 PDF/DOCX/PPTX/XLSX 样本集的默认输出，可选的 A/B harness 比较 prompt 与模型改动，webextract 语料对照 defuddle 给 HTML 抽取打分
- **serve 的 API 契约由机器校验**：每个 JSON 路由声明 pydantic 响应模型，`scripts/export_openapi.py` 导出带 SSE 载荷的 schema，并有测试拿 `webapp/src/api/types.ts` 与之比对
- **defuddle 移植有了清单与上游追踪**：`PORT_MANIFEST.md` 记录每个 `webextract` 模块跟踪的上游来源和语料 pin 的提交，测试保证两个 pin 一致
- **网站提供 `/llms.txt`**，README 带一张对照 markitdown、docling、anydoc 的表格，明说各自更擅长什么
- **`NOTICE` 记录 MIT 自身覆盖不了的第三方义务**：AGPL-3.0 的 PyMuPDF 技术栈及其对再分发与网络使用的含义、`webextract/` 所基于的 defuddle 移植（MIT © kepano），以及源自 marker 的基准打分器
- **CI 会因非商用或意料之外的 copyleft 依赖而失败**：`scripts/check_licenses.py` 从无 extras 的安装读取许可证元数据，拒绝任何非商用或专有许可，以及白名单之外的 AGPL/GPL 包
- **Substack 文章提取**能读已渲染正文和 `window._preloads` JSON，自定义域名和署名栏日期也算在内；Notes 提取与通用回退照旧
- **Linux 桌面代理发现**读取 GNOME/Unity 和 KDE 手动设置的 HTTP 代理及绕过列表。显式的环境代理仍然优先；PAC、仅 SOCKS 和需要认证的桌面代理设置不会导入
- **可选的基准 LLM 评分**：`score_with_llm_judge` 返回经过校验的内容、结构和噪声分数，复用离线缓存，限制输入长度，也不会自己重试。缓存未命中时必须指定模型并设置 `allow_network=True`；默认基准仍是离线的启发式评分
- **八种格式不再需要 extra**：`.tsv`、`.xml`、`.rst`、`.org`、`.tex`、`.odt`、`.ods`、`.rtf` 现在只靠标准库就能在基础安装里转换。制表符分隔文件本就是一张 Markdown 表格，OpenDocument 文件本就是一包 XML
- **`.rtf` 有了真正的原生解析器**：分词器处理分组、控制字与 `\'xx` 字节，驱动一份按分组进出栈的排版状态，于是 outlinelevel 与样式表名变成标题，`\trowd`/`\cell`/`\row` 变成 Markdown 表格，Word 97 的 `\pntext` 符号与新式 `\ls`/`\ilvl` 列表保住各自的项目符号与层级，`HYPERLINK` 域变成链接，`\ansicpg`/`\fcharset` 正确解码中日韩与 Windows 代码页。不需要 extra，也不需要抽取引擎
- **每个布尔 CLI 开关都有明确的否定形式**：新增 `--cache`、`--compress`、`--no-pure`、`--no-screenshot-only`，配置里的默认值可以在命令行上朝任一方向覆盖

### 变更

- **三个 LLM 路由层合并为一个 `MarkitaiRouter`**：本地提供商经各自的 handler 表分发，标准模型交回 litellm 自己的分组负载均衡、冷却与 fallback，不再手工重实现
- **结构化 LLM 调用采用模型真正支持的最强模式**：先原生 tool calling，再 `response_format` JSON schema（这也顺带启用了 claude-agent 提供商此前从未走过的原生路径），markdown-JSON 降为最后手段
- **四个入口共用一条 URL 管线**：`serve`、Python API、`markitai <url>` 和 URL 批量各自维护着几乎相同的「抓取 → 本地化 → 增强 → 写出」副本，现统一到 `workflow/url.py` 的一条级联
- **Prompt 缓存真正生效了**：五个系统模板把逐请求变量放在靠前位置，稳定前缀短到任何提供商的缓存都匹配不上。变量段移到末尾，指令文本未变
- **转换一个 `.txt` 不再加载 PDF 与 Office 工具链**：转换器此前全部急切注册，每个进程多花约 430ms，且无论输入是什么都会拉起 markitdown、Magika 和 onnxruntime
- **一个提供商只在一张表里描述**：它的默认模型和 API key 变量曾被手抄到凭据检测、安装向导、`serve` 启动候选和 `init` 另外四处，几份抄本早已各自漂移
- **退役告警与成本估算改为推导**：替代模型取自默认表，退役日期取自 litellm 自己的记录，取代了一个已经过时的字面量和一个盖在所有模型上的硬编码日期
- **自解释错误不再泄漏异常类名**：缺少 OCR 后端、文件超限、格式不支持时只打印可操作的那句话
- **对照语料与算法 pin 到同一个上游**：样本从 2026 年 3 月快照重新同步到 defuddle 0.19.3（83 → 208），并补齐了新语料暴露出的六项上游行为
- **webextract 少解析也少复制**：fragment 与 document 统一到一条 lxml 路径，五份 block 标签表收敛，重试不再深拷贝整棵树，`srcset` 只剩一份实现
- **默认安装体积减少 155 MB**，633 MB 降到 477 MB。`opencv-python` 完全移出核心，RapidOCR 移到新的 `ocr` extra。**扫描件与图片 OCR 现在需要 `uv tool install "markitai[ocr]" --force`**
- **去掉 OpenCV 后图片质量反而更好**：在所有实测样本上纯 Pillow 都胜出，PSNR 与 SSIM 更高、文件小 7.7%，因为 OpenCV 的 `INTER_LANCZOS4` 缩放跳过抗锯齿预滤波
- **所有依赖升级到当前版本**：56 项更新中包括 `litellm` 1.91.1 → 1.97.0、`markitdown` 0.1.6 → 0.1.7，并对升级后的依赖树重跑了许可证审计
- **PDF 引擎下限提升到 `pymupdf4llm>=1.28.2`**：1.28.0 引入的 `pymupdf-layout` 采用 Polyform Noncommercial 许可，直接禁止商用；1.28.2 恢复了 AGPL-3.0 双许可
- **安装器不再询问 LibreOffice**：幻灯片渲染是可选的运行时路径，Windows 与 macOS 上本机的 MS Office 已能覆盖，引导安装彻底不再过问；PPTX `--screenshot`/`--ocr` 转换在本机没有任何渲染器时于转换时告警，并给出对应平台的安装命令
- **`--no-pure` 优先于 `MARKITAI_PURE`**：命令行显式给出的开关现在双向覆盖环境变量，之前是环境变量优先

### 移除

- **三平台 Office 自动化整体退役**：约 1.3k 行 Windows COM、macOS AppleScript 和 LibreOffice CLI 驱动代码，连同围绕它的批量预转换机制。它服务的格式改由 `markitai[legacy]` 覆盖，LibreOffice 的文案收窄到 PPTX 幻灯片渲染
- **`kreuzberg` extra 与 `-b kreuzberg` 后端全部移除**：它覆盖的每一种格式现在都原生转换，这个 extra 已经没有任何东西可以解锁。`--backend` 只剩 `native|cloudflare`，`fetch.kreuzberg_convert_enabled` 配置项删除，传入 `--kreuzberg` 会得到一条说明 `.rtf` 已原生转换的用法错误，而不是指向某个替代写法
- **六个已弃用的抓取与后端别名全部移除**（`--playwright`、`--defuddle`、`--static`、`--jina`、`--cloudflare`、`--kreuzberg`），改用 `-s` 与 `-b`。传入已移除的名字现在是会指出替代写法的用法错误，`--help` 的选项从 36 个降到 30 个
- **LLM 路径里两层多余的防御**：一个手写的 JSON 模式策略压在本就做同一件修复的结构化阶梯之下，以及一份实际流量早已不再走到的手抄 Copilot 价格表
- **`llm.prompts.page_content_system` 与 `llm.prompts.page_content_user` 两个配置项**，连同它们配置的那条代码路径：其 docstring 声称在用它的两种模式早已改道，所以设了没有任何效果
- **FFmpeg 不再被检查、宣传或安装**：markitai 从来不支持音视频，`doctor` 却把它呈现为「音视频文件处理」，安装脚本还会主动询问

### 修复

- **OpenAI 系默认模型刷新到当前代**：`gpt-5.6-luna` 取代 `gpt-5.4-nano`，$0.20/$1.20 对 $0.20/$1.25。陈旧的默认值只有在模型开始拒绝服务时才出错，因此新增守护测试：任一默认距 litellm 记录的退役日期不足 120 天即失败
- **Azure OpenAI 部署终于用上配置里写的 `api_version`**：`litellm_params` 从未声明这个字段而未知键默认被忽略，所以文档给出的 Azure 示例里它在解析时就被丢弃，从未到达 litellm
- **`--ocr` 转出来的 PDF 重新按页划分**：两条 OCR 路径都用空行拼接各页，纯截图模式写的标记又无人匹配，于是 OCR 产出到达 LLM 分批、漂移保护和输出 profile 时是一整段没有页边界的文本
- **LLM 调用失败的那篇文档，布局与其余保持一致**：所有失败路径都在管线的 profile 步骤之前返回，`--profile` 运行中会剩下一个仍是默认资源布局的文件
- **`--llm-batch` 拒绝 `--ocr` 时会说清原因**：页图走在文档请求内部，而 batch 第一阶段关闭 LLM 转换，那正是用本地 OCR 读扫描页、而非渲染页图的分支
- **`--llm` 在没有配置文件时形同虚设**：从 `MODEL` 或已检测到的 key 填充模型池的那一步被配置自身的 `llm.enabled` 拦住，且运行在 `--preset` 与 `--llm` 生效之前
- **`--resume` 无法真正续跑**：CLI 批量路径从未写出加载器所需的基准状态文件，中断的运行只留下无法回放的增量 sidecar，于是每篇文档都重新付费
- **`--llm-batch-collect` 现在遵守 `--config-json`**：它构建配置时丢掉了内联覆盖，导致 collect 的实时兜底用的是配置文件里碰巧写的那个池
- **发往推理模型的批请求不再整批失败**：OpenAI 的 batch 部署在推理开启时拒绝 function tools，于是每篇文档都返回 400 并回落到实时重跑。现在对「推理能力模型 + tools」的请求注入 `reasoning_effort="none"`
- **`router_settings.fallbacks` 真正生效**：此前部署是按 id 直接寻址的，litellm 从未看到可供回落的分组。未指定 `default` 分组的配置现在会在启动时明确报错
- **Claude 模型不再在用量报告里显示为 $0**：litellm 对表里没有的名字会返回零价格，读起来像「免费」，还掩盖了模糊匹配本可找到的条目。相关地，`gpt-5.4-codex` 曾被按普通 `gpt-5.4` 计价
- **被截断的 LLM 响应现在计入成本报告**：达到 token 上限被截断的响应在记录用量之前就被丢弃，于是最贵的那些调用恰好是合计里缺失的
- **改进 prompt 现在真的会改变输出**：持久缓存按调用的类别而非 prompt 本身作键，所以 prompt 改动之前转换过的文档会一直返回旧结果
- **失败的 LLM 调用不会再让文档永久空白**：所有重试都返回空时，那个空字符串被写进了没有过期时间的缓存，此后每次运行该文档都「转换」为空
- **已完成的转换不会再报告失败**：onnxruntime 在解释器关闭时的清理偶尔会 abort 进程，把一次已经写出产物的转换变成非零退出
- **没装 OCR extra 时的 `--ocr` 不再报告成功**：它会回落到图片占位符并以 0 退出，文件「转换」完成却什么都没读出来
- **安装提示给出的是真能用的命令**：extras 名被 rich 当作标记吃掉，而 `pip install` / `uv add` 作用于当前项目而非 markitai 所在的隔离环境。现在由一个 helper 统一渲染命令并读取 `sys.prefix`，pipx 装的会被告知用 pipx，`uv tool` 装的会被告知用 `uv tool`
- **wheel 现在带上了 `LICENSE` 与 `NOTICE`**：`license-files` 指向的路径在包目录内并不存在，于是每个已发布的 wheel 都没有许可证
- **对齐 defuddle 0.19.3 的六个抽取缺口**：可关闭的 `aria-hidden` 浮层内正文、CodeMirror 代码块、文中图片行、Substack Notes、SVG 图形、行内相关文章块。resync 还修复了它新暴露的问题：Hugo admonition、lightbox 去重、LaTeX 图片服务、`<noscript>` 回退与带行号的代码
- **Reddit、Hacker News 与 YouTube 页面重新受质量校验**：它们的校验门注册在一个 extractor 从不产出的 profile 名下，三者都落到了通用文章校验
- **PDF 内嵌图片在符号链接与树外输出目录下不再丢失**：pymupdf4llm 可能写出符号链接解析后的路径或相对工作目录的路径，而此前只改写了其中一种写法
- **CMYK JPEG 不再导致图片压缩失败**：来自印刷源 PDF 的图片走到了只处理 RGBA、P、LA 的分支而被丢弃
- **截图视口设置终于起作用**：`screenshot.viewport_width` 与 `viewport_height` 有声明、有描述、也发布在 schema 里，却没有任何代码读取
- **两个看起来可配置、实际从来不可配置的设置**：`auto_proxy` 和截图的 `full_page` 用带默认值的 `getattr` 读取，而没有任何配置模型声明它们
- **每条抓取路径都遵守 `NO_PROXY`**：统一的解析器此前没有任何调用方，Playwright 启动、两条 Cloudflare 路径和批量渲染器都在不查例外列表的情况下解析代理，macOS 与 Windows 的系统列表也不例外
- **代理自动检测不再凭空发明代理**：它此前靠探测常见本地端口的 TCP 连通性判断，而在 TUN 模式 VPN 下任何端口都会成功
- **`markitai --help` 不再以 Batch API 开场**：`--llm-batch` 系列选项不属于任何面板，而未分组选项会被优先渲染。示例里还提供过一个从不存在的 YouTube 转换
- **安装器只在默认索引确实慢时才询问镜像**，不再对所有没配代理的用户发出警告。此前一位法兰克福用户的第一印象，是一条关于中国镜像的提问
- **安装器现在尊重对 OCR 说的「不」**：建议列表被合并回选择结果，导致拒绝了仍会安装
- **测试套件对开发者自己的配置免疫**：真实的 `~/.markitai/config.json` 或 `.env` 曾让部分测试在真正使用 markitai 的人的机器上失败
- **移动端转换采用紧凑的一体输入卡片**：URL 输入区下面是选项、上传和转换共用的一行，取代三层堆叠控件，触控目标保持 44px，移动端 CLI 预览自然换行；桌面端仍是行内输入布局
- **输入卡片的三个操作统一为轻量样式**：选项、上传文件和转换都是无边框、等距内边距的控件，悬停与聚焦表现一致；转换按钮靠更重的字重和浅色强调底色保持主操作的辨识度，不再用实心色块
- **选项面板改为分组**：预设作为首行主选项，保留「已调整」状态与提示；「增强」汇集 LLM、OCR 与图片分析；「输出」放输出规范；URL／文件抓取选择器、内容源开关和缓存／压缩开关收进「高级」折叠区，其中任一项已非默认时自动展开。所有控件、提示、联动规则与翻译都没变
- **预设与转换选项统一联动**：面板、API 请求和 CLI 预览读的都是服务端的预设定义和解析规则。选择预设会重置它的五项功能；图片分析随 LLM／纯净模式暂停和恢复；截图来源与远程后端在三处解析结果一致。CLI 预览默认按服务端预设生成「预设加差异」的精简命令，并写明它假设的是默认配置；本地配置不同时，勾选「包含配置覆盖参数」就能保留截图来源、纯净模式、缓存和压缩的显式关闭开关

### 安全

- **`markitai serve` 现在签发访问令牌**：启动横幅打印登录 URL，来自其它机器的请求必须出示令牌，此前无鉴权的部分访问改为 401
- **`markitai serve` 不再能被指使去探测它所在的网络**：`--host 0.0.0.0` 时，局域网内任何机器都能提交 URL 让服务端去抓取，包括私网地址和云元数据端点
- **绑定到非回环地址时会说明这意味着什么**：`--host 0.0.0.0` 会警告任何能连到该服务的人都可以转换文件，并读取、下载或删除全部转换历史
- **`fetch.remote_consent=ask` 现在在 X/Twitter 富化前也会询问**：那条路径此前不经提示就走向远端，只遵守更早做出的决定

## [0.23.0] - 2026-07-18

### 新增

- **网页工作台的预览可导出为 PDF**：渲染后的 Markdown 会打印成整洁的 A4 文档，四边都有真实页边距，代码块与图片不会在页面中间被截断，跨页表格会重复表头。「PDF 设置」菜单中提供可选的自定义页眉页脚（默认开启）；PDF 始终以浅色打印，因此在 Chrome、Safari 和 macOS 预览中效果一致

### 变更

- **网页工作台现已适配手机与窄屏**：≤780px 采用统一布局，让输入区、选项与任务列表形成一条连贯的单列。顶栏收拢为标识加历史/设置，语言与主题切换移入设置弹窗，外部链接与通知停靠到页面底部，批量下载移到任务列表下方
- **批量下载在桌面端移到转换选项旁边**，不再与工作台顶部的清空操作争抢位置

### 修复

- **本地服务现在会拒绝跨源与 DNS 重绑定请求**：Host/Origin 白名单（回环地址、IP 字面量主机，以及你传入的任意 `--allowed-host`）可阻止恶意网页访问 API，否则它可能读取已保存的提供商凭据，或驱动服务端去抓取 URL。同源使用与局域网绑定不受影响
- **就地重试与 LLM 增强变得可靠**：在同任务其他条目仍在转换时重试某条，不再让它卡在「排队中」；增强失败会保留可用的基础结果，而非把该行降级为无输出的错误；重跑被中断后重启也不再从历史中抹掉整个任务；重跑会覆盖自身此前的输出，而不是累积带版本号的重复文件或返回过期的增强版本
- **扫描文档的 OCR 在整批任务中保持准确**：稀疏页的分块回退不再全局降低检测阈值，后续页面均以完整置信度识别
- **LLM 替代文本可应用于名称含空格或中文的图片**：此前以百分号编码写入的引用无法匹配，导致这些资源的图注被静默丢弃
- **X 的引用推文保留原文**：引用内部的省略号或「显示更多」文本不再对完整内容伪造截断标记
- **仅修改提供商的 API Key 不再固化其默认 API Base**：当连接使用提供商默认值时，编辑框为空（默认值显示为占位符），保存后会继续跟随该默认值
- **整段历史下载会自我清理**：生成的归档在下载完成后即被删除，不再残留为孤儿文件；构建过程中被删除的文件会被跳过，而不是导致下载失败
- **任务全部成功时完成通知显示「N 完成」**，而非「N 完成 · 0 失败」；本地服务在为大型输出目录计算体积时，不再阻塞其他任务的实时进度

## [0.22.0] - 2026-07-14

### 新增

- **`markitai serve` 新增本地网页工作台**：安装 `serve` 附加组件后，可以上传文件或文件夹、提交 URL、查看实时进度、预览和下载结果，并在中英双语的无障碍界面中回看七天内保存在磁盘上的任务历史
- **LLM 设置可直接在网页工作台中完成**：它会找出你机器上可用的本地与 API 提供商，实时列出模型，让你配置带权重的部署、在不暴露已保存凭据的前提下测试连接，转换任务用的也是这同一套设置
- **网页任务可以重试和对比**：失败的文件或 URL 条目可以单独重试，结果多了可以筛选，任务在后台完成时会有通知，等效的 CLI 命令一键复制，基础 Markdown 与 LLM 增强版本并排对比

### 修复

- **文件名包含空格或括号的 PDF 现在也会分析内嵌图片**：pymupdf4llm 写出图片时会清洗源文件名，但此前收集阶段仍按原名匹配，导致 `--alt`、`--desc` 和 `rich` 静默跳过全部内嵌图片，而页面截图不受影响。现在会解析 Markdown 中实际写入的图片引用，保留原有前缀匹配作为后备，并在引用的资源不存在时记录警告
- **Windows 上的网页结果可正常下载嵌套资源**：API 路径统一使用 URL 分隔符，不再泄漏文件系统反斜杠，因此图片及其他生成资源在所有受支持系统上都能通过相同链接打开

## [0.21.1] - 2026-07-12

### 修复

- **macOS 与 Windows 全新安装不再尝试从源码编译 litellm**：litellm 1.92.0 只发布了 Linux 平台的 wheel，其他平台的 `uv tool install markitai` 会退回从 sdist 构建其新增的 Rust 扩展。没有 Rust 工具链时直接失败，构建环境选中不受支持的 Python（如系统默认的 3.14）时则报出大段 pyo3 错误。现已排除该版本，依赖解析会落在各平台均有 wheel 的版本上

## [0.21.0] - 2026-07-12

### 变更

- **旧版 `.xls` 文件改为纯 Python 转换**（经 MarkItDown 的 xlrd 路径，与 `.xlsx` 同引擎）：任何平台都不再启动或依赖 Office 应用与 LibreOffice。单元格内容与此前 Excel 自动化的输出完全一致（真机验证）；转换更快，也不再受应用级锁串行化影响
- **持久 LLM 缓存按模型池划定作用域**：更换模型配置后不再继续使用旧模型产出的结果，而轮换 API key 不会让缓存失效；同一内容的文档改名后清洗结果仍然命中。升级后现有缓存会一次性全部未命中；旧条目本来就读不到（见"修复"）
- **结构化 LLM 调用接入完整的传输层重试**（指数退避、配额/账单错误短路、空响应重试）。此前这类调用完全绕过了纯文本调用一直具备的重试机制。模型持续输出退化内容时最坏情况的尝试次数会相应增加，这是真正执行重试的代价
- **独立 `.urls` 批量在 `--screenshot` 多源场景下获得视觉增强**，与单条 URL 转换同一页面的行为保持一致
- **内部：fetch、LLM 与 CLI 三个子系统在行为不变的前提下完成结构重组**。机器可执行的 import 分层契约进入 CI，fetch.py 从约 3000 行缩至约 1100 行并按策略拆分模块，LLM 管线由五份手写副本收敛为单一引擎，报告/退出码/输出处理在全部输入路径间单源化。针对 0.15–0.20 全部修复条目的 43 项回归矩阵验证零回归

### 移除

- **Python 3.14 支持**：包与安装脚本的 `requires-python` 收窄为 `>=3.11,<3.14`，CI 与发布矩阵不再测试 3.14；仅为让 3.14 可解析而存在的平台条件 ONNX Runtime 约束一并移除
- Excel COM 自动化（Windows）与 Excel AppleScript 自动化（macOS），以及批量预转换、重任务调度中的 `.xls` 条目，`.xls` 已经不再需要它们

### 修复

- **跨会话 LLM 缓存此前从未真正命中**：持久缓存的读取与写入在缓存键的 model 列上不一致，任何一次运行都读不到之前运行写入的结果，重复处理同一文档每次都在全额消耗 LLM 调用。现已修复读写一致（回归测试锁定），重跑未变更的文档近乎即时且零成本
- **macOS 上经 Microsoft Office 转换 `.doc`/`.ppt` 此前整体不可用**：Word 与 PowerPoint 的 AppleScript `open` 没有返回值，脚本的文档绑定从未建立。首个引用即报错，而错误处理器对同一变量的引用又把真实错误掩盖成难以理解的"variable openedItem is not defined"，文档从未被关闭，残留的僵尸文档让应用状态随重试持续恶化。脚本现在按唯一暂存名绑定文档、在保存前后两个名字下分别关闭、并始终抛出原始错误。Excel 的"Parameter error -50"同根同修（`.xls` 已改纯 Python 转换后此路径不再使用，见"移除"，但该修复同样覆盖 PPTX 转 PDF 导出路径）
- **macOS：Office 更新后应用首次被脚本拉起时 `.doc`/`.ppt` 转换失败**（"document never registered" / PowerPoint 错误 -9074）。该状态下应用会静默丢弃 Word 带参数的 `open` 请求但正常响应其他事件；Word 脚本现在会在轮询停滞时改用无参数 `open` 重试（实测可穿透并治愈该状态）并记录恢复日志。停滞与 -9074 的报错文案改为经过验证的补救措施（手动打开该应用一次让其完成初始化），替换掉误导性的"应用卡死或过载，退出后重试"
- **截断或退化的 LLM 输出不再污染缓存**：截图提取与 URL 增强现在会像其他调用点一样拒绝因 max_tokens 截断的响应；URL 增强也不再缓存被截去退化尾部的输出
- **`markitai -I` 遇到无效配置文件不再抛出裸 traceback**：交互向导现在给出与其他命令一致的可操作错误信息
- **目录批量的报告不再丢失 URL 条目的截图数**
- **defuddle HTTP 客户端在代理或超时设置变更后会重建**（此前整个进程沿用首个客户端，而 jina 一直是正确处理的）
- **禁用的模型（`weight: 0`）不再影响缓存作用域**：禁用某个模型后其旧缓存结果不再被使用，增删禁用条目也不再让整个池的缓存失效
- **Copilot 认证状态在新版 Copilot CLI 下能正确读取**：新版 CLI 会写入 JSONC 风格注释并更换了登录状态键名，导致 `doctor` 与 `auth copilot status` 把解析错误当作"未登录"。现在两种格式均可识别，无法解析时报告"无法确定"而非"未认证"；CLI 自身报告成功的登录也不会再被后续的配置读取否决

## [0.20.0] - 2026-07-10

### Added

- **macOS 未安装 LibreOffice 时可使用本机 Microsoft Office**：旧版 `.doc`、`.ppt`、`.xls` 文件可通过 AppleScript 调用 Word、PowerPoint 或 Excel 完成转换，PowerPoint 也能先将 PPTX 幻灯片导出为 PDF 再渲染。此备选方案由 `office.macos_fallback` 控制，无头环境可将其关闭，`doctor` 和文档也会说明一次性的自动化授权要求

### Changed

- **公网 URL 不再询问，私网 URL 永远不出本机**：公网 URL 可以直接使用远程后备服务，首次使用前会在 stderr 输出一次进程级说明，覆盖 defuddle.md、Jina、Cloudflare、FxTwitter 和 Twitter oEmbed。私网、本机、DNS 解析到非公网地址及带凭据的 URL（包括路径中的敏感令牌）始终留在本机，`MARKITAI_NO_REMOTE_FETCH=1` 会挡住所有远程路径，显式指定的远程 `-s` 策略也一样
- **`doctor` 改为检查能力是否真的可用，而不只是看包有没有装**：普通检查只有 RapidOCR 和已配置的工作流会影响退出状态；Playwright 会实际启动 Chromium 来验证，活跃模型引用的环境变量也会查一遍，你要求 `--fix` 时它只安装并复查 Chromium，不动当前项目的依赖
- **`markitai init` 默认保留已有配置**：直接按 Enter 会选择 Keep，Update 和 Overwrite 仍需明确选择
- **入门流程从便携安装脚本开始**：首页会在首次绘制前判断是 Windows 还是 macOS 或 Linux，并推荐 `setup.ps1` 或 `setup.sh`，`uv tool install markitai` 保留为手动安装方式。页面还加了一个 60 秒无 LLM 示例，中文导航、屏幕阅读器和高对比度支持也都改好了
- **显式配置 `gpt-5.6-luna` 时可识别其 Copilot 价格信息**：OpenAI 和 ChatGPT 的自动入门配置仍用已普遍开放的模型，受限预览模型要自己选

### Fixed

- **`--quiet` 模式在单项和批量任务中保持一致**：你要求输出到 stdout 的 Markdown 照常输出，错误仍写入 stderr，`--quiet --dry-run` 不显示预览，URL 批量任务部分失败时保留成功结果并退出 10，进度和成功提示则一律不显示
- **未启用任何提取方式的图片转换不再误报成功**：单张图片未指定 `--ocr` 或 `--llm` 时会退出 1 并给出处理建议，不再以成功状态结束却没有任何输出
- **重复运行安装脚本会保留用户意图**：shell 和 PowerShell 脚本发现 `~/.markitai/config.json` 已存在时会跳过 `markitai init --yes`，同时保留已有 extras；即使 Markitai 已安装，显式设置的 `MARKITAI_VERSION` 也不会被普通升级路径绕过
- **首页快速开始命令在亮色模式下保持清晰**：深色命令面板现在始终用浅色文字和透明代码背景，命令快挤不下时就切成纵向排版；站点也提供了真实的 `/favicon.ico`，不再返回 404
- **Python 3.14 的依赖解析不再固定到不兼容的 ONNX Runtime**：约束改为按平台设置，Magika 的 Windows 上限在需要的地方保留，其他平台可以用支持 Python 3.14 的 ONNX Runtime 版本

### Security

- **配置输出默认隐藏秘密**：`markitai config list` 会递归遮蔽秘密和自定义请求头，`api_base` 只保留协议、主机名和端口，显式传入 `--show-secrets` 才会显示原始值
- **URL 凭据始终留在本机，也不会出现在诊断信息中**：终端错误、进度标签、`--dry-run` 预览、控制台与文件日志，以及自动生成的输出文件名都会去掉用户信息、敏感路径令牌、查询参数和片段；主机名只要解析到任何非公网地址，就不会越过远程分发边界
- **macOS Office 自动化会隔离不受信任的文档**：备选转换打开的是只读暂存副本，宏和外部链接更新都禁用，只绑定并关闭自己打开的文档，跨进程串行访问 Office 应用，可清理的暂存文件也以私有权限保存
- **无头安装不算同意安装可选软件**：没有可用终端时，便携安装脚本只装 uv、Python 和 Markitai，别的一概不装；显式设置 `MARKITAI_INSTALL_OPTIONAL=1` 才会装可选包、浏览器二进制、系统依赖和第三方 CLI

## [0.19.0] - 2026-07-10

### Changed

- **远程提取默认不再弹出确认**（`fetch.remote_consent` 默认值 `ask` → `always`）：公网 URL 会在本地策略失败后直接按链回退到远程提取服务（defuddle.md、Jina、Cloudflare，逐个尝试，成功即停），不再打断询问；首次使用会有一条 INFO 日志说明。私有/本地 URL 无论此配置如何都不会使用远程服务，netloc 携带凭据的 URL（`user:pass@host`）现在也视同私有。可通过 `fetch.remote_consent=ask`/`never` 或 `MARKITAI_NO_REMOTE_FETCH=1` 恢复询问或禁用远程服务
- **确认提示文案重写**（针对 `remote_consent=ask`）：提示现在会说明弹出原因（本地提取未成功）、逐个尝试的机制（一次一个、成功即停），并只列出实际在链中的服务，所以 Cloudflare（使用你自己的账户凭据）仅在已配置时出现。交互式确认现在也会先暂停实时进度显示，不再撕裂界面

### Fixed

- **`--resume` 此前完全不生效**：CLI 批量入口接受该参数，但每次都会从头重新处理所有文件。现在会加载已保存的状态，跳过已完成的文件，重试失败或中断的文件，纳入本次新发现的文件，并报告 `Resuming batch: N completed, M remaining`
- **输出命名恢复为追加式方案**：`sample.pdf` → `sample.pdf.md`（而非 `sample.md`），撤销了 0.15.0 引入的"替换扩展名"命名方案。那个方案会隐藏源文件格式、破坏多重后缀文件名（如 `archive.tar.gz`），并导致同一文件在单文件模式和批量模式下的输出名不一致
- **Windows 安装一行命令 404**：站点现在会部署 `setup.ps1`（文档指向 https://markitai.dev/setup.ps1，但此前只部署了 `setup.sh`）；中文更新日志的改动现在也会触发站点重新部署
- **Prompt 尾部 REMINDER 泄漏进清理结果**：使用较小模型时（在 `gpt-5.4-mini` 上观察到），视觉清理 prompt 末尾的 `REMINDER: ...` 指令行（连同 `---` 分隔符）可能被逐字回显到 `.llm.md` 输出末尾。现改用 `<document>` 标签定界文档、全部指令置于内容之前，并新增出口防护剥离回显的 prompt 片段（对已缓存的历史结果同样生效）
- **部分 URL 转换会静默跳过图片 alt 分析**：有截图但无多源内容的 URL（如经站点提取器抓取的 X 帖子）会落入纯文本 LLM 分支、不做图片分析；URL 批量模式则完全不分析图片，`--alt`/`--desc` 在这些路径下形同虚设。现在两条路径都会分析已下载的图片（alt 文本 + `images.json`）。stdout 模式的资产链接重写也不再用文件名覆盖 LLM 生成的 alt 描述
- **批量图片分析不再因"裸负载 JSON"而失败**：小模型有时会对单图批次直接返回裸的结果对象而非 `{"images": [...]}` 包裹结构，导致 Instructor 重试耗尽后降级为逐张分析并打出 ERROR 日志。JSON 修复层现在会就地矫正这类形状，批量分析直接成功
- **弱模型可能在 LLM 清理时破坏社媒帖正文**（引用推文的 blockquote 被拍平、CJK 与英文之间被私自加空格，在 `claude-agent/haiku` 上观察到）：标记为 `social_post` 的内容现在正文原样直通，LLM 仅生成元数据；其他文档类型的处理 prompt 也补充了 blockquote 保护与 CJK 空格显式规则
- **ChatGPT 连接错误此前不可重试且信息为空**：httpx 传输层错误（连接重置/拒绝、超时）被映射为不可重试的 `ProviderError` 且消息为空，导致所有重试层被绕过。现在标记为可重试并携带底层错误文本
- **控制台日志不再撕裂实时进度显示**：日志输出改经共享的 rich stderr console 路由，日志行会打印在 StageList 转轮上方而不再留下残影帧；quiet/stdout 模式也应用与普通模式相同的第三方重试噪音过滤（此前 instructor 的原始重试错误会直接漏到控制台）
- **LLM 增强失败现在在输出中可见**：当所有 LLM 路径都失败时，兜底生成的 `.llm.md` frontmatter 会带上 `llm_enhanced: false` 标记并打出 ERROR 级日志。此前降级输出的唯一线索只有一个空的 description

### Removed

- **死代码清理**：移除了不可达的异步 enricher 注册表、未使用的异常层级、无调用方的废弃辅助函数、仅供测试的工具函数，以及约 5MB 无引用的测试固件；`markitdown` 依赖从 `[all]` 收窄为实际用到的 office 附加组件（去掉 azure/audio/pdfminer/youtube 等传递依赖）；`httpx` 和 `lxml` 现在改为直接声明

## [0.18.0] - 2026-07-09

### Changed

- **网页提取效果对齐 Defuddle**：正文主体选择和噪声内容清理算法现已高度对齐 Defuddle 的实现（评分机制、内容模式识别、内容边界检测）。基准测试语料库均分对比 Defuddle 基准：91.04 → 92.72

## [0.17.0] - 2026-07-08

### Removed

- **移除 Gemini CLI 提供商**（`gemini-cli/`）：Google 已下线它依赖的 OAuth 接入流程。请改用直接的 `GEMINI_API_KEY`，或通过 OpenRouter 接入

### Added

- 支持 `COPILOT_GITHUB_TOKEN` 认证，检测优先级高于 `GH_TOKEN`/`GITHUB_TOKEN`
- 耗时较长的转换阶段会显示已用时长提示，避免长时间运行的 LLM 调用看起来像卡死

### Fixed

- **Cloudflare 抓取现在能正确走到站点专用提取器**：由 `/markdown` 端点切换为 `/content` 端点，使经 Cloudflare 抓取的页面获得与其他策略一致的提取质量
- 减少了重复的重试/校验日志噪音

### Changed

- 本地 provider 不再为未使用的扩展思考/推理输出浪费 token
- provider 检测和 `init` 现在会建议更便宜的默认模型

## [0.16.0] - 2026-07-07

### Added

- 新增 **B 站专栏（opus）提取器**，支持 `bilibili.com/opus/<id>` 页面
- **反爬/验证码检测**：能识别 Geetest、Cloudflare、reCAPTCHA、hCaptcha 等挑战页面，不再将其当作正常内容处理

### Changed

- **X/Twitter 提取重新以 DOM 解析为主**，仅当原生提取效果不足时才回退到 FxTwitter/oEmbed 补充方案

### Fixed

- X Article 的 URL 匹配、抓取性能、frontmatter 字数统计相关问题

## [0.15.0] - 2026-07-04

一个维护版本：依赖刷新、支持 Python 3.14，并经过多轮排查，修复了批量处理、抓取/缓存、LLM provider、图片处理和配置系统中 30 多个已验证的 bug。

### Added

- 支持 Python 3.14；包元数据中新增 MIT 许可证声明
- 通过 rich-click 实现分组、更快的 `--help`
- PDF 乱码/扫描文本检测，并给出使用 `--ocr` 的建议
- 跨页面重复出现的页眉/页脚会被自动清理
- VLM 退化保护（截断视觉/OCR 输出中的重复循环内容）
- HTML 提取与脚注处理效果对齐 Defuddle（MathJax/MathML、代码块、多种站点的脚注格式）
- 统一的 `-s/--strategy` 抓取参数（旧的分策略参数保留为已弃用别名）
- **远程抓取同意机制**：未经同意不会将 URL 发送给第三方服务（`fetch.remote_consent`、`MARKITAI_NO_REMOTE_FETCH`）
- PDF 隐藏文本清理（提示注入防护）：`security.pdf_sanitize`
- 混合数字/扫描文档的按页 OCR 路由
- 针对 Defuddle 基准语料库的转换质量基准测试工具
- 通过 release-please 实现发布自动化

### Fixed

多轮 bug 排查中的主要修复：

- 针对 X 2026 年改版重写了 DOM 提取器；FxTwitter 回退路径现在能被默认抓取链路正确调用到
- 原生支持 `.eml` 邮件解析、HEIC/HEIF/AVIF 图片输入（`markitai[heif]`）、CI 质量守护门槛
- `markitai init` 现在会合并进已有配置而非直接覆盖；各 provider 的登录失败提示更清晰
- 抓取/缓存正确性修复：AUTO 策略缓存复用陈旧数据、Playwright 上下文泄漏、代理自动检测误报、空 URL 导致批量任务崩溃
- LLM/provider 正确性修复：失败不再被误报为成功、视觉分析缓存污染、Copilot 并发临时文件竞争、阻塞调用导致的事件循环停滞、重试退避期间占用并发槽位
- 图片/转换正确性修复：EXIF 方向、LA 模式透明通道、未压缩图片命名、EMF/WMF 格式误标、OCR 引擎配置漂移、临时目录泄漏
- 配置/CLI 正确性修复：配置编辑器保存前先校验、修复符号链接安全检查、`llm.concurrency` 下限校验、JSON 日志格式修复、`config set` 类型强制转换与方括号写法支持

### Changed

- 输入路径与子命令混用现在会报错，而不是静默丢弃输入路径
- 单文件/URL 输入下 `-o out.md` 会精确写入该文件
- 诊断信息改为输出到 stderr，保持 stdout 管道输出干净
- 输出命名改为"替换扩展名"方案（`sample.pdf` → `sample.md`），**已在 0.19.0 中撤销**
- `image.stdout_persist` 现在默认开启
- 转换报告（`.markitai/reports/`）默认仅在批量任务中生成

### Security

- 解除 litellm 供应链安全钉版（改为 `>=1.83.0`），上游已完成审计并对发布版本签名

## [0.14.0] - 2026-03-25

- Added: Steam News 提取器；结构化 MathML 转 LaTeX；LibreOffice 功能性检测（不只是检测程序是否存在）
- Fixed: PDF 数学公式提取回退逻辑；Steam BBCode 内容的 XSS 防护；修复了不稳定的集成测试
- Security: litellm 钉版至 `<1.82.7`（供应链安全事件应对）

## [0.13.1] - 2026-03-23

- Added: 配置编辑器重新设计：模糊搜索、可滚动列表、原地刷新 UI
- Added: 为 66 个 Pydantic 配置项新增字段说明
- Fixed: 配置编辑器中 Esc 键支持、布尔值编辑器一致性、Literal 类型取值保留

## [0.12.1] - 2026-03-22

- Added: stdout 模式下的终端内联图片显示（Kitty/iTerm2），基于内容寻址的资产存储实现
- Added: 新增中文用户旅程文档
- Fixed: 静默/stdout 模式下 LLM 错误现在可见；修复 Kitty 协议图片格式问题；`init` 不再生成重复的 provider 条目

## [0.12.0] - 2026-03-20

- Added: 原生 HTML 提取流水线：基于 resolver 的提取、frontmatter 构建器、质量档案，以及针对 GitHub Discussions、X 讨论串、YouTube 页面的结构化提取器
- Added: 新增 `--static` 与 `--kreuzberg` CLI 参数
- Changed: HTML 文件默认改为走原生 webextract 流水线
- Fixed: URL 的 stdout 回退逻辑、共享缓存/信号量的线程安全、配置原子写入

## [0.11.2] - 2026-03-14

- Fixed: Windows 下的内存检测用于任务规模调节；`~/.markitai/` 目录改为延迟创建（避免只读场景下产生副作用）；输出/日志目录默认值改为 `None`（不再硬编码路径）

## [0.11.1] - 2026-03-14

- Added: 交互式向导中新增 pure 模式选项
- Fixed: 修复 `--pure` 错误触发视觉/截图路径的问题；降低了过于激进的"内容过短"判定阈值

## [0.11.0] - 2026-03-13

- Added: **`--pure` 模式**：LLM 透明直通（仅做文本清洗，不生成 frontmatter/不做后处理），与 `--llm` 解耦
- Added: 新增 `--keep-base`，可在生成 `.llm.md` 的同时强制保留基础 `.md`
- Fixed: URL 处理流程现在与文件处理一致地遵循 `--pure`/`--llm`/`--keep-base` 参数

## [0.10.0] - 2026-03-12

- Added: 无配置文件时，自动从环境变量和已认证的 CLI 中检测可用的 LLM provider
- Changed: `-v` 改为 `--verbose`（此前是 `--version`）；`-V` 表示 `--version`
- Changed: 通过延迟导入缩短冷启动耗时（约 5s → 0.3s）
- Fixed: alt 文本语言现在跟随文档语言，不再默认使用英文

## [0.9.2] - 2026-03-11

- Fixed: Copilot/Claude 登录改为始终使用继承的标准输入输出（修复凭据存储失败问题）；错误提示更清晰，不再是难以理解的包装异常

## [0.9.1] - 2026-03-09

- Added: 新增 `markitai doctor --suggest-extras`，作为安装脚本获取推荐 extras 的唯一权威来源
- Fixed: 修复安装脚本中的登录守卫和 extras 解析 bug；provider 名称的 Rich 标记转义问题

## [0.9.0] - 2026-03-09

- Added: 支持配置全局/按域名的抓取**策略优先级**，以及 `local_only_patterns`/`inherit_no_proxy`，用于将敏感域名限制为仅使用本地策略
- Fixed: LLM 输出不再把混合语言页面内容错误地翻译成其他语言

## [0.8.1] - 2026-03-06

- Added: **新增 Defuddle 抓取策略**（免费、无需认证），并作为最高优先级选项；新增 `--defuddle` CLI 参数
- Changed: 默认策略顺序调整为优先使用 Defuddle/Jina

## [0.8.0] - 2026-03-06

- Added: 通过 markitdown/kreuzberg 新增 20 多种文件格式支持（HTML、CSV、EPUB、MSG、IPYNB、Numbers、TSV、XML、ODS、ODT、SVG、RTF、RST、ORG、TEX、EML）；新增 GIF/BMP/TIFF 图片支持
- Fixed: Claude Agent SDK 兼容性升级；i18n 测试隔离问题；kreuzberg 注册时导入阶段的日志噪音

## [0.7.0] - 2026-03-05

- Added: 新增 **ChatGPT provider**（`chatgpt/`），通过 OAuth 设备码流程接入
- Added: 新增 **Gemini CLI provider**（`gemini-cli/`），后于 0.17.0 移除
- Added: 支持 `weight: 0` 显式禁用某个模型的路由
- Fixed: 修复所有模型 weight 都为 0 时路由器除零错误

## [0.6.1] - 2026-03-05

- Fixed: Claude Agent SDK 合规性修复；认证预检查现在能识别更多基于环境变量的凭据

## [0.6.0] - 2026-03-04

- Added: **Cloudflare 集成**：URL 使用 Browser Rendering，文件使用 Workers AI `toMarkdown`
- Added: 新增 Fetch Policy 引擎，支持域名配置和 Playwright 会话持久化
- Added: 可插拔的静态 HTTP 后端（`httpx`/`curl-cffi`）
- Fixed: 修复所有视觉模型被禁用时路由器除零错误；清理了代码库中 21 个死函数

## [0.5.2] - 2026-02-07

- Fixed: SQLite 连接泄漏；Windows 路径处理 bug；OAuth 过期状态误报；清除全部 Pyright 警告

## [0.5.1] - 2026-02-07

- Added: Playwright 自动滚动以触发懒加载内容；提取前清理 DOM 噪音（导航栏/广告/Cookie 提示条）；支持 `python -m markitai` 调用方式
- Changed: `init`/交互模式/doctor 中的默认模型全面更新；缓存指纹改为基于完整内容的哈希（此前仅用较短前缀，容易冲突）

## [0.5.0] - 2026-02-06

- Added: 新增 **`markitai init`** 配置向导与**交互模式**（`-I`）；`doctor --fix` 自动安装缺失组件
- Changed: 通过延迟加载模块，CLI 启动速度提升约 3 倍；批量 UI 简化为更紧凑的进度显示
- Fixed: Windows 下 LibreOffice/FFmpeg 检测问题；修复导致挂起的 Playwright 默认等待条件

## [0.4.2] - 2026-02-03

- Changed: 调整 Playwright 等待默认值以更好支持 SPA 页面
- Fixed: X/Twitter 页面现在会等待 JS 完全渲染后再截取；缓存改为遵循配置目录，不再使用硬编码路径

## [0.4.1] - 2026-02-02

- Added: 新增 **`markitai doctor`** 诊断命令；本地 provider 支持自适应超时；Claude Agent 长系统提示词支持缓存

## [0.4.0] - 2026-01-28

- Added: 新增 **Claude Agent SDK** 与 **GitHub Copilot SDK** 本地 provider；URL 支持 HTTP 条件缓存（ETag/Last-Modified）；新增 `--quiet`/`-q` 参数
- Changed: 模块结构重大调整（`cli/`、`llm/`、`providers/`）

## [0.3.2] - 2026-01-27

- Added: 新增中文 README 和中文安装脚本

## [0.3.1] - 2026-01-27

- Added: **SPA 域名学习机制**：自动检测并记住重度依赖 JS 的站点，避免重复浪费静态抓取尝试
- Added: Windows 性能调优（线程池规模、OCR 引擎单例、更快的图片压缩）
- Fixed: 提示词泄漏防护（拆分 system/user 提示词）；抓取时的自动代理检测

## [0.3.0] - 2026-01-26

- Added: 支持**直接转换 URL**，以及 `.urls` 批量文件
- Added: 多策略抓取（`static`/`agent-browser`/`jina`/`auto`），配合 SQLite 抓取缓存和截图功能
- Added: 新增 `--no-cache-for <pattern>` 精细化跳过缓存；`cache stats -v`
- Added: 官方 VitePress 文档网站上线（中英双语）
- Added: 新增 MIT 许可证；CI/CD 工作流

## [0.2.4] - 2026-01-21

- Fixed: Office/PPTX 兼容性补丁；符号链接安全加固；LLM 空响应重试；frontmatter 字段顺序问题

## [0.2.3] - 2026-01-20

- Added: **持久化 SQLite LLM 缓存**，支持 LRU 淘汰；新增 `cache stats`/`cache clear` 命令
- Added: 视觉感知的模型路由；PDF/图片并行处理

## [0.2.2] - 2026-01-20

- Added: 新增 `constants.py` 模块统一管理硬编码常量；扩充单元测试覆盖率

## [0.2.1] - 2026-01-20

- Added: 按文件统计 LLM 用量/成本；新增类型化的用量/资产模型；跨平台 Office/LibreOffice 检测
- Changed: 文件命名冲突时的重命名方式改为 `.v2.md` 风格的自然排序

## [0.2.0] - 2026-01-19

- Added: **Monorepo 重写**：采用 uv workspace、基于 LiteLLM 的 provider 接入方式、全新的转换器/工作流架构、JSON Schema 校验的配置
- Breaking: 全新配置格式和 CLI 语法；不再支持 Python 3.13 以下版本；移除旧版 `src/markitai/` 架构

## [0.1.6] - 2026-01-14

- Fixed: 模型路由相关 bug；文档准确性修正

## [0.1.5] - 2026-01-13

- Changed: 提示词管理与清洗模块重构

## [0.1.4] - 2026-01-13

- Fixed: LLM JSON 解析边界情况；日志格式问题

## [0.1.3] - 2026-01-12

- Changed: 采用 `src` 目录布局；新增 CI 工作流

## [0.1.2] - 2026-01-12

- Added: 网络健壮性增强（重试/超时处理）；新增面向 AI 助手的文档（`CLAUDE.md`、`AGENTS.md`）

## [0.1.1] - 2026-01-11

- Changed: 架构重大重构，采用服务层模式

## [0.1.0] - 2026-01-10

- Added: 基于能力的模型路由、Provider 惰性初始化、超时并发回退、`--fast` 执行模式、按模型统计的批量处理数据

## [0.0.1] - 2026-01-08

- Added: **首次发布**：CLI（`convert`/`batch`/`config`/`provider`）、Office/PDF/HTML 转换、5 个可回退的 LLM provider、图片处理、支持断点续传的批量处理
