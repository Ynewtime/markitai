# 更新日志

本项目的所有重要变更都会记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，
版本号遵循[语义化版本](https://semver.org/spec/v2.0.0.html)规范。

## [0.20.0] - 2026-07-10

### Added

- **macOS 未安装 LibreOffice 时可使用本机 Microsoft Office**：旧版 `.doc`、`.ppt`、`.xls` 文件可通过 AppleScript 调用 Word、PowerPoint 或 Excel 完成转换，PowerPoint 也能先将 PPTX 幻灯片导出为 PDF 再渲染。此备选方案由 `office.macos_fallback` 控制，无头环境可将其关闭，`doctor` 和文档也会说明一次性的自动化授权要求

### Changed

- **公网 URL 无需确认，同时收紧远程抓取的隐私边界**：公网 URL 可以直接使用远程后备服务，首次使用前会在 stderr 输出一次进程级说明，覆盖 defuddle.md、Jina、Cloudflare、FxTwitter 和 Twitter oEmbed；私网、本机、DNS 解析到非公网地址及带凭据的 URL（包括路径中的敏感令牌）始终留在本机，`MARKITAI_NO_REMOTE_FETCH=1` 也会阻止显式指定的远程 `-s` 策略
- **`doctor` 改为检查能力是否真的可用**：普通检查只有 RapidOCR 和已配置的工作流会影响退出状态；Playwright 会实际启动 Chromium，活跃模型引用的环境变量会被检查，用户请求的 `--fix` 也只会安装并复查 Chromium，不会修改当前项目的依赖
- **`markitai init` 默认保留已有配置**：直接按 Enter 会选择 Keep，Update 和 Overwrite 仍需明确选择
- **入门流程从便携安装脚本开始**：首页会在首次绘制前判断 Windows 与 macOS 或 Linux，并推荐 `setup.ps1` 或 `setup.sh`，同时保留 `uv tool install markitai` 作为手动安装方式；页面还加入了 60 秒无 LLM 示例，并完善中文导航、屏幕阅读器和高对比度支持
- **显式配置 `gpt-5.6-luna` 时可识别其 Copilot 价格信息**：OpenAI 和 ChatGPT 的自动入门配置继续使用已普遍开放的模型，受限预览模型仅供用户主动选择

### Fixed

- **`--quiet` 模式在单项和批量任务中保持一致**：请求输出到 stdout 的 Markdown 会保留，错误仍写入 stderr，`--quiet --dry-run` 不显示预览，URL 批量任务部分失败时保留成功结果并退出 10，进度和成功路径等提示则保持隐藏
- **未启用任何提取方式的图片转换不再误报成功**：单张图片未指定 `--ocr` 或 `--llm` 时会退出 1 并给出处理建议，不再以成功状态结束却没有任何输出
- **重复运行安装脚本会保留用户意图**：shell 和 PowerShell 脚本发现 `~/.markitai/config.json` 已存在时会跳过 `markitai init --yes`，同时保留已有 extras；即使 Markitai 已安装，显式设置的 `MARKITAI_VERSION` 也不会被普通升级路径绕过
- **首页快速开始命令在亮色模式下保持清晰**：深色命令面板现在会稳定使用浅色文字和透明代码背景，并在命令变得拥挤前切换为纵向排版；站点也会提供真实的 `/favicon.ico`，不再返回 404
- **Python 3.14 的依赖解析不再固定到不兼容的 ONNX Runtime**：按平台设置的约束会在必要时保留 Magika 的 Windows 上限，其他环境则可使用支持 Python 3.14 的 ONNX Runtime 版本

### Security

- **配置输出默认隐藏秘密**：`markitai config list` 会递归遮蔽秘密和自定义请求头，`api_base` 只保留协议、主机名和端口，只有显式传入 `--show-secrets` 才会显示原始值
- **URL 凭据始终留在本机且不会出现在诊断信息中**：终端错误、进度标签、`--dry-run` 预览、控制台与文件日志，以及自动生成的输出文件名都会移除用户信息、敏感路径令牌、查询参数和片段；只要主机名解析结果中包含非公网地址，就无法越过远程分发边界
- **macOS Office 自动化会隔离不受信任的文档**：备选转换在打开只读暂存副本时会禁用宏和外部链接更新，只绑定并关闭自己打开的文档，跨进程串行访问 Office 应用，并以私有权限保留可清理的暂存文件
- **无头安装不会被视为同意安装可选软件**：没有可用终端时，便携安装脚本只安装 uv、Python 和 Markitai；只有显式设置 `MARKITAI_INSTALL_OPTIONAL=1`，才会安装可选包、浏览器二进制、系统依赖和第三方 CLI

## [0.19.0] - 2026-07-10

### Changed

- **远程提取默认不再弹出确认**（`fetch.remote_consent` 默认值 `ask` → `always`）：公网 URL 会在本地策略失败后直接按链回退到远程提取服务（defuddle.md、Jina、Cloudflare——逐个尝试，成功即停），不再打断询问；首次使用会通过 INFO 日志披露。私有/本地 URL 无论此配置如何都不会使用远程服务，netloc 携带凭据的 URL（`user:pass@host`）现在也视同私有。可通过 `fetch.remote_consent=ask`/`never` 或 `MARKITAI_NO_REMOTE_FETCH=1` 恢复询问或禁用远程服务
- **确认提示文案重写**（针对 `remote_consent=ask`）：提示现在会说明弹出原因（本地提取未成功）、逐个尝试的机制（一次一个、成功即停），并动态列出实际在链中的服务——Cloudflare（使用你自己的账户凭据）仅在已配置时出现。交互式确认现在也会先暂停实时进度显示，不再撕裂界面

### Fixed

- **`--resume` 此前完全不生效**：CLI 批量入口接受该参数，但每次都会从头重新处理所有文件。现已修正为正确加载已保存的状态——已完成的文件会跳过，失败/中断的文件会重试，本次新发现的文件也会被纳入——并报告 `Resuming batch: N completed, M remaining`
- **输出命名恢复为追加式方案**：`sample.pdf` → `sample.pdf.md`（而非 `sample.md`），撤销了 0.15.0 引入的"替换扩展名"命名方案——该方案会隐藏源文件格式、破坏多重后缀文件名（如 `archive.tar.gz`），并导致同一文件在单文件模式和批量模式下的输出名不一致
- **Windows 安装一行命令 404**：站点现在会部署 `setup.ps1`（文档指向 https://markitai.dev/setup.ps1，但此前只部署了 `setup.sh`）；中文更新日志的改动现在也会触发站点重新部署
- **Prompt 尾部 REMINDER 泄漏进清理结果**：使用较小模型时（在 `gpt-5.4-mini` 上观察到），视觉清理 prompt 末尾的 `REMINDER: ...` 指令行（连同 `---` 分隔符）可能被逐字回显到 `.llm.md` 输出末尾。现改用 `<document>` 标签定界文档、全部指令置于内容之前，并新增出口防护剥离回显的 prompt 片段（对已缓存的历史结果同样生效）
- **部分 URL 转换会静默跳过图片 alt 分析**：有截图但无多源内容的 URL（如经站点提取器抓取的 X 帖子）会落入纯文本 LLM 分支、不做图片分析；URL 批量模式则完全不分析图片——`--alt`/`--desc` 在这些路径下形同虚设。现在两条路径都会分析已下载的图片（alt 文本 + `images.json`）。stdout 模式的资产链接重写也不再用文件名覆盖 LLM 生成的 alt 描述
- **批量图片分析不再因"裸负载 JSON"而失败**：小模型有时会对单图批次直接返回裸的结果对象而非 `{"images": [...]}` 包裹结构，导致 Instructor 重试耗尽后降级为逐张分析并打出 ERROR 日志。JSON 修复层现在会就地矫正这类形状，批量分析直接成功
- **弱模型可能在 LLM 清理时破坏社媒帖正文**（引用推文的 blockquote 被拍平、CJK 与英文之间被私自加空格——在 `claude-agent/haiku` 上观察到）：标记为 `social_post` 的内容现在正文原样直通，LLM 仅生成元数据；其他文档类型的处理 prompt 也补充了 blockquote 保护与 CJK 空格显式规则
- **ChatGPT 连接错误此前不可重试且信息为空**：httpx 传输层错误（连接重置/拒绝、超时）被映射为不可重试的 `ProviderError` 且消息为空，导致所有重试层被绕过。现在标记为可重试并携带底层错误文本
- **控制台日志不再撕裂实时进度显示**：日志输出改经共享的 rich stderr console 路由，日志行会打印在 StageList 转轮上方而不再留下残影帧；quiet/stdout 模式也应用与普通模式相同的第三方重试噪音过滤（此前 instructor 的原始重试错误会直接漏到控制台）
- **LLM 增强失败现在在输出中可见**：当所有 LLM 路径都失败时，兜底生成的 `.llm.md` frontmatter 会带上 `llm_enhanced: false` 标记并打出 ERROR 级日志——此前降级输出的唯一线索只有一个空的 description

### Removed

- **死代码清理**：移除了不可达的异步 enricher 注册表、未使用的异常层级、无调用方的废弃辅助函数、仅供测试的工具函数，以及约 5MB 无引用的测试固件；`markitdown` 依赖从 `[all]` 收窄为实际用到的 office 附加组件（去掉 azure/audio/pdfminer/youtube 等传递依赖）；`httpx` 和 `lxml` 现在改为直接声明

## [0.18.0] - 2026-07-09

### Changed

- **网页提取效果对齐 Defuddle**：正文主体选择和噪声内容清理算法现已高度对齐 Defuddle 的实现（评分机制、内容模式识别、内容边界检测）。基准测试语料库均分对比 Defuddle 基准：91.04 → 92.72

## [0.17.0] - 2026-07-08

### Removed

- **移除 Gemini CLI 提供商**（`gemini-cli/`）—— Google 已下线相关 OAuth 接入流程。请改用直接的 `GEMINI_API_KEY`，或通过 OpenRouter 接入

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

维护性大版本：依赖全面刷新、支持 Python 3.14，并经过多轮排查修复了批量处理、抓取/缓存、LLM provider、图片处理和配置系统中 30 多个已验证的 bug。

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

多轮质量与 bug 排查中的重点修复：

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
- 输出命名改为"替换扩展名"方案（`sample.pdf` → `sample.md`）——**已在 0.19.0 中撤销**
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
- Changed: 通过延迟导入大幅缩短冷启动耗时（约 5s → 0.3s）
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
- Added: 新增 **Gemini CLI provider**（`gemini-cli/`）—— 后于 0.17.0 移除
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
