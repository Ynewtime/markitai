# 更新日志

本项目的所有重要变更都会记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，
版本号遵循[语义化版本](https://semver.org/spec/v2.0.0.html)规范。

## [Unreleased]

### Fixed

- **`--resume` 此前完全不生效**：CLI 批量入口接受该参数，但每次都会从头重新处理所有文件。现已修正为正确加载已保存的状态——已完成的文件会跳过，失败/中断的文件会重试，本次新发现的文件也会被纳入——并报告 `Resuming batch: N completed, M remaining`
- **输出命名恢复为追加式方案**：`sample.pdf` → `sample.pdf.md`（而非 `sample.md`），撤销了 0.15.0 引入的"替换扩展名"命名方案——该方案会隐藏源文件格式、破坏多重后缀文件名（如 `archive.tar.gz`），并导致同一文件在单文件模式和批量模式下的输出名不一致
- **Windows 安装一行命令 404**：站点现在会部署 `setup.ps1`（文档指向 https://markitai.dev/setup.ps1，但此前只部署了 `setup.sh`）；中文更新日志的改动现在也会触发站点重新部署
- **Prompt 尾部 REMINDER 泄漏进清理结果**：使用较小模型时（在 `gpt-5.4-mini` 上观察到），视觉清理 prompt 末尾的 `REMINDER: ...` 指令行（连同 `---` 分隔符）可能被逐字回显到 `.llm.md` 输出末尾。现改用 `<document>` 标签定界文档、全部指令置于内容之前，并新增出口防护剥离回显的 prompt 片段（对已缓存的历史结果同样生效）
- **部分 URL 转换会静默跳过图片 alt 分析**：有截图但无多源内容的 URL（如经站点提取器抓取的 X 帖子）会落入纯文本 LLM 分支、不做图片分析；URL 批量模式则完全不分析图片——`--alt`/`--desc` 在这些路径下形同虚设。现在两条路径都会分析已下载的图片（alt 文本 + `images.json`）。stdout 模式的资产链接重写也不再用文件名覆盖 LLM 生成的 alt 描述
- **批量图片分析不再因"裸负载 JSON"而失败**：小模型有时会对单图批次直接返回裸的结果对象而非 `{"images": [...]}` 包裹结构，导致 Instructor 重试耗尽后降级为逐张分析并打出 ERROR 日志。JSON 修复层现在会就地矫正这类形状，批量分析直接成功
- **弱模型可能在 LLM 清理时破坏社媒帖正文**（引用推文的 blockquote 被拍平、CJK 与英文之间被私自加空格——在 `claude-agent/haiku` 上观察到）：标记为 `social_post` 的内容现在正文原样直通，LLM 仅生成元数据；其他文档类型的处理 prompt 也补充了 blockquote 保护与 CJK 空格显式规则

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
- 输出命名改为"替换扩展名"方案（`sample.pdf` → `sample.md`）——**已在 [Unreleased] 中撤销**
- `image.stdout_persist` 现在默认开启
- 转换报告（`.markitai/reports/`）默认仅在批量任务中生成

### Security

- 解除 litellm 供应链安全钉版（改为 `>=1.83.0`），上游已完成审计并对发布版本签名

## [0.14.0] - 2026-03-25

### Added

- Steam News 提取器；结构化 MathML 转 LaTeX；LibreOffice 功能性检测（不只是检测程序是否存在）

### Fixed

- PDF 数学公式提取回退逻辑；Steam BBCode 内容的 XSS 防护；修复了不稳定的集成测试

### Security

- litellm 钉版至 `<1.82.7`（供应链安全事件应对）

## [0.13.1] - 2026-03-23

### Added

- 配置编辑器重新设计：模糊搜索、可滚动列表、原地刷新 UI
- 为 66 个 Pydantic 配置项新增字段说明

### Fixed

- 配置编辑器中 Esc 键支持、布尔值编辑器一致性、Literal 类型取值保留

## [0.12.1] - 2026-03-22

### Added

- stdout 模式下的终端内联图片显示（Kitty/iTerm2），基于内容寻址的资产存储实现
- 新增中文用户旅程文档

### Fixed

- 静默/stdout 模式下 LLM 错误现在可见；修复 Kitty 协议图片格式问题；`init` 不再生成重复的 provider 条目

## [0.12.0] - 2026-03-20

### Added

- 原生 HTML 提取流水线：基于 resolver 的提取、frontmatter 构建器、质量档案，以及针对 GitHub Discussions、X 讨论串、YouTube 页面的结构化提取器
- 新增 `--static` 与 `--kreuzberg` CLI 参数

### Changed

- HTML 文件默认改为走原生 webextract 流水线

### Fixed

- URL 的 stdout 回退逻辑、共享缓存/信号量的线程安全、配置原子写入

## [0.11.2] - 2026-03-14

### Fixed

- Windows 下的内存检测用于任务规模调节；`~/.markitai/` 目录改为延迟创建（避免只读场景下产生副作用）；输出/日志目录默认值改为 `None`（不再硬编码路径）

## [0.11.1] - 2026-03-14

### Added

- 交互式向导中新增 pure 模式选项

### Fixed

- 修复 `--pure` 错误触发视觉/截图路径的问题；降低了过于激进的"内容过短"判定阈值

## [0.11.0] - 2026-03-13

### Added

- **`--pure` 模式**：LLM 透明直通（仅做文本清洗，不生成 frontmatter/不做后处理），与 `--llm` 解耦
- 新增 `--keep-base`，可在生成 `.llm.md` 的同时强制保留基础 `.md`

### Fixed

- URL 处理流程现在与文件处理一致地遵循 `--pure`/`--llm`/`--keep-base` 参数

## [0.10.0] - 2026-03-12

### Added

- 无配置文件时，自动从环境变量和已认证的 CLI 中检测可用的 LLM provider

### Changed

- `-v` 改为 `--verbose`（此前是 `--version`）；`-V` 表示 `--version`
- 通过延迟导入大幅缩短冷启动耗时（约 5s → 0.3s）

### Fixed

- alt 文本语言现在跟随文档语言，不再默认使用英文

## [0.9.2] - 2026-03-11

### Fixed

- Copilot/Claude 登录改为始终使用继承的标准输入输出（修复凭据存储失败问题）；错误提示更清晰，不再是难以理解的包装异常

## [0.9.1] - 2026-03-09

### Added

- 新增 `markitai doctor --suggest-extras`，作为安装脚本获取推荐 extras 的唯一权威来源

### Fixed

- 修复安装脚本中的登录守卫和 extras 解析 bug；provider 名称的 Rich 标记转义问题

## [0.9.0] - 2026-03-09

### Added

- 支持配置全局/按域名的抓取**策略优先级**，以及 `local_only_patterns`/`inherit_no_proxy`，用于将敏感域名限制为仅使用本地策略

### Fixed

- LLM 输出不再把混合语言页面内容错误地翻译成其他语言

## [0.8.1] - 2026-03-06

### Added

- **新增 Defuddle 抓取策略**（免费、无需认证），并作为最高优先级选项；新增 `--defuddle` CLI 参数

### Changed

- 默认策略顺序调整为优先使用 Defuddle/Jina

## [0.8.0] - 2026-03-06

### Added

- 通过 markitdown/kreuzberg 新增 20 多种文件格式支持（HTML、CSV、EPUB、MSG、IPYNB、Numbers、TSV、XML、ODS、ODT、SVG、RTF、RST、ORG、TEX、EML）；新增 GIF/BMP/TIFF 图片支持

### Fixed

- Claude Agent SDK 兼容性升级；i18n 测试隔离问题；kreuzberg 注册时导入阶段的日志噪音

## [0.7.0] - 2026-03-05

### Added

- 新增 **ChatGPT provider**（`chatgpt/`），通过 OAuth 设备码流程接入
- 新增 **Gemini CLI provider**（`gemini-cli/`）—— 后于 0.17.0 移除
- 支持 `weight: 0` 显式禁用某个模型的路由

### Fixed

- 修复所有模型 weight 都为 0 时路由器除零错误

## [0.6.1] - 2026-03-05

### Fixed

- Claude Agent SDK 合规性修复；认证预检查现在能识别更多基于环境变量的凭据

## [0.6.0] - 2026-03-04

### Added

- **Cloudflare 集成**：URL 使用 Browser Rendering，文件使用 Workers AI `toMarkdown`
- 新增 Fetch Policy 引擎，支持域名配置和 Playwright 会话持久化
- 可插拔的静态 HTTP 后端（`httpx`/`curl-cffi`）

### Fixed

- 修复所有视觉模型被禁用时路由器除零错误；清理了代码库中 21 个死函数

## [0.5.2] - 2026-02-07

### Fixed

- SQLite 连接泄漏；Windows 路径处理 bug；OAuth 过期状态误报；清除全部 Pyright 警告

## [0.5.1] - 2026-02-07

### Added

- Playwright 自动滚动以触发懒加载内容；提取前清理 DOM 噪音（导航栏/广告/Cookie 提示条）；支持 `python -m markitai` 调用方式

### Changed

- `init`/交互模式/doctor 中的默认模型全面更新；缓存指纹改为基于完整内容的哈希（此前仅用较短前缀，容易冲突）

## [0.5.0] - 2026-02-06

### Added

- 新增 **`markitai init`** 配置向导与**交互模式**（`-I`）；`doctor --fix` 自动安装缺失组件

### Changed

- 通过延迟加载模块，CLI 启动速度提升约 3 倍；批量 UI 简化为更紧凑的进度显示

### Fixed

- Windows 下 LibreOffice/FFmpeg 检测问题；修复导致挂起的 Playwright 默认等待条件

## [0.4.2] - 2026-02-03

### Changed

- 调整 Playwright 等待默认值以更好支持 SPA 页面

### Fixed

- X/Twitter 页面现在会等待 JS 完全渲染后再截取；缓存改为遵循配置目录，不再使用硬编码路径

## [0.4.1] - 2026-02-02

### Added

- 新增 **`markitai doctor`** 诊断命令；本地 provider 支持自适应超时；Claude Agent 长系统提示词支持缓存

## [0.4.0] - 2026-01-28

### Added

- 新增 **Claude Agent SDK** 与 **GitHub Copilot SDK** 本地 provider；URL 支持 HTTP 条件缓存（ETag/Last-Modified）；新增 `--quiet`/`-q` 参数

### Changed

- 模块结构重大调整（`cli/`、`llm/`、`providers/`）

## [0.3.2] - 2026-01-27

### Added

- 新增中文 README 和中文安装脚本

## [0.3.1] - 2026-01-27

### Added

- **SPA 域名学习机制**：自动检测并记住重度依赖 JS 的站点，避免重复浪费静态抓取尝试
- Windows 性能调优（线程池规模、OCR 引擎单例、更快的图片压缩）

### Fixed

- 提示词泄漏防护（拆分 system/user 提示词）；抓取时的自动代理检测

## [0.3.0] - 2026-01-26

### Added

- 支持**直接转换 URL**，以及 `.urls` 批量文件
- 多策略抓取（`static`/`agent-browser`/`jina`/`auto`），配合 SQLite 抓取缓存和截图功能
- 新增 `--no-cache-for <pattern>` 精细化跳过缓存；`cache stats -v`
- 官方 VitePress 文档网站上线（中英双语）
- 新增 MIT 许可证；CI/CD 工作流

## [0.2.4] - 2026-01-21

### Fixed

- Office/PPTX 兼容性补丁；符号链接安全加固；LLM 空响应重试；frontmatter 字段顺序问题

## [0.2.3] - 2026-01-20

### Added

- **持久化 SQLite LLM 缓存**，支持 LRU 淘汰；新增 `cache stats`/`cache clear` 命令
- 视觉感知的模型路由；PDF/图片并行处理

## [0.2.2] - 2026-01-20

### Added

- 新增 `constants.py` 模块统一管理硬编码常量；扩充单元测试覆盖率

## [0.2.1] - 2026-01-20

### Added

- 按文件统计 LLM 用量/成本；新增类型化的用量/资产模型；跨平台 Office/LibreOffice 检测

### Changed

- 文件命名冲突时的重命名方式改为 `.v2.md` 风格的自然排序

## [0.2.0] - 2026-01-19

### Added

- **Monorepo 重写**：采用 uv workspace、基于 LiteLLM 的 provider 接入方式、全新的转换器/工作流架构、JSON Schema 校验的配置

### Breaking Changes

- 全新配置格式和 CLI 语法；不再支持 Python 3.13 以下版本；移除旧版 `src/markitai/` 架构

## [0.1.6] - 2026-01-14

### Fixed

- 模型路由相关 bug；文档准确性修正

## [0.1.5] - 2026-01-13

### Changed

- 提示词管理与清洗模块重构

## [0.1.4] - 2026-01-13

### Fixed

- LLM JSON 解析边界情况；日志格式问题

## [0.1.3] - 2026-01-12

### Changed

- 采用 `src` 目录布局；新增 CI 工作流

## [0.1.2] - 2026-01-12

### Added

- 网络健壮性增强（重试/超时处理）；新增面向 AI 助手的文档（`CLAUDE.md`、`AGENTS.md`）

## [0.1.1] - 2026-01-11

### Changed

- 架构重大重构，采用服务层模式

## [0.1.0] - 2026-01-10

### Added

- 基于能力的模型路由、Provider 惰性初始化、超时并发回退、`--fast` 执行模式、按模型统计的批量处理数据

## [0.0.1] - 2026-01-08

### Added

- **首次发布**：CLI（`convert`/`batch`/`config`/`provider`）、Office/PDF/HTML 转换、5 个可回退的 LLM provider、图片处理、支持断点续传的批量处理

[0.12.1]: https://github.com/Ynewtime/markitai/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/Ynewtime/markitai/compare/v0.11.2...v0.12.0
[0.11.2]: https://github.com/Ynewtime/markitai/compare/v0.11.1...v0.11.2
[0.11.1]: https://github.com/Ynewtime/markitai/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/Ynewtime/markitai/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/Ynewtime/markitai/compare/v0.9.2...v0.10.0
[0.9.2]: https://github.com/Ynewtime/markitai/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/Ynewtime/markitai/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/Ynewtime/markitai/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/Ynewtime/markitai/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/Ynewtime/markitai/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Ynewtime/markitai/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/Ynewtime/markitai/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/Ynewtime/markitai/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/Ynewtime/markitai/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/Ynewtime/markitai/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Ynewtime/markitai/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/Ynewtime/markitai/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/Ynewtime/markitai/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Ynewtime/markitai/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/Ynewtime/markitai/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Ynewtime/markitai/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Ynewtime/markitai/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/Ynewtime/markitai/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/Ynewtime/markitai/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/Ynewtime/markitai/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Ynewtime/markitai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Ynewtime/markitai/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/Ynewtime/markitai/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/Ynewtime/markitai/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/Ynewtime/markitai/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/Ynewtime/markitai/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Ynewtime/markitai/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Ynewtime/markitai/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Ynewtime/markitai/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/Ynewtime/markitai/releases/tag/v0.0.1
