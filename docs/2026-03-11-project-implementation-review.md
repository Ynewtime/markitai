# Markitai 项目实现评审报告

评审时间: 2026-03-11

## 1. 评审目标

本次评审聚焦以下方向:

- 不合常理或与注释/CLI 语义明显不一致的实现
- 明显错误与潜在功能故障
- 性能瓶颈、资源浪费、全局状态污染
- CLI 关键路径可用性
- 测试覆盖盲区

## 2. 评审方法

本次评审结合了静态审查、子系统并行审查和本地验证。

已执行/观察到的验证:

- `UV_CACHE_DIR=/tmp/markitai-uv-cache uv run pyright`
  - 结果: `0 errors, 0 warnings, 0 informations`
- `UV_CACHE_DIR=/tmp/markitai-uv-cache uv run ruff check packages/markitai/src/markitai packages/markitai/tests`
  - 结果: `All checks passed!`
- `UV_CACHE_DIR=/tmp/markitai-uv-cache uv run pytest -m 'not slow and not network'`
  - 在评审窗口内持续运行，输出已超过 70% 进度，未观察到失败，但我没有拿到完整收尾结果，因此不能把它记为“全量通过”
- 转换器/WebExtract 定向子集测试
  - `test_converter_pdf.py`
  - `test_image_converter.py`
  - `test_pipeline.py`
  - `test_image.py`
  - 结果: `210 passed`

## 3. 总体结论

项目整体工程化程度不低，类型检查、lint、单测基线都在，核心模块也有明确分层。但当前实现仍存在一批“测试能过、用户路径会出错”的问题，尤其集中在:

- CLI 交互/诊断命令
- OCR + Vision 路径
- 共享单例/全局状态
- 图片与截图处理
- 自称有 fallback/retry，但实际上没有执行 fallback 的逻辑

其中最值得优先处理的是 4 个高优先级问题:

1. `doctor` 会在关键诊断失败时仍输出 “all good”
2. `--interactive` 会吞掉真实失败退出码
3. PDF 的 `--ocr --llm` 默认不会产出 Vision 所需页面图像
4. PPTX 的 `--ocr` / `--ocr --llm` 也会被 `screenshot` 开关错误短路

## 4. 详细问题

### High-1: `doctor` 会对关键失败给出假阳性总结

严重度: High

现象:

- `doctor` 的总结逻辑只看 `required_deps` 和 `optional_missing`
- LLM 相关检查、vision model 检查、认证检查即使报错，也不会阻止最终输出 `doctor.all_good`

证据:

- `packages/markitai/src/markitai/cli/commands/doctor.py:732`
- `packages/markitai/src/markitai/cli/commands/doctor.py:733`
- `packages/markitai/src/markitai/cli/commands/doctor.py:736`

影响:

- `doctor` 本应是用户排障入口，但现在会把“认证失败/LLM 不可用/vision 配置错误”伪装成健康状态
- 这类假阳性会直接误导用户，降低 CLI 可信度

建议:

- 总结逻辑应覆盖:
  - required dependencies
  - LLM runtime checks
  - auth checks
  - vision model checks
- 至少应把 `error` 级结果纳入 `all_good` 判定

### High-2: 交互模式不会传播真实失败退出码

严重度: High

现象:

- `run_interactive_mode()` 通过 `subprocess.run([sys.executable, "-m", "markitai"] + args)` 重新调用 CLI
- 但没有检查子进程退出码
- 随后无条件 `ctx.exit(0)`

证据:

- `packages/markitai/src/markitai/cli/main.py:62`
- `packages/markitai/src/markitai/cli/main.py:81`
- `packages/markitai/src/markitai/cli/main.py:84`

影响:

- 子进程转换失败时，外层 `markitai --interactive` 仍然返回成功
- 这会破坏 shell 脚本、CI、用户对退出码的预期

建议:

- 使用 `subprocess.run(..., check=False)` 后显式读取 `returncode`
- 或直接 `ctx.exit(result.returncode)`

### High-3: PDF 的 `--ocr --llm` 默认不会产出页面图像

严重度: High

现象:

- `convert()` 在 `use_ocr and use_llm` 时进入 `_render_pages_for_llm()`
- 但 `_render_pages_for_llm()` 内部又要求 `self.config.screenshot.enabled` 为真才真正渲染页面图像
- `ScreenshotConfig.enabled` 默认是 `False`

证据:

- `packages/markitai/src/markitai/converter/pdf.py:89`
- `packages/markitai/src/markitai/converter/pdf.py:91`
- `packages/markitai/src/markitai/converter/pdf.py:654`
- `packages/markitai/src/markitai/converter/pdf.py:660`
- `packages/markitai/src/markitai/config.py:279`
- `packages/markitai/src/markitai/config.py:286`

影响:

- `--ocr --llm` 的行为与注释“Render pages as images for LLM Vision analysis”不一致
- 默认情况下这条路径会退化成文本-only，Vision 增强实际失效

建议:

- `OCR+LLM` 与 `--screenshot` 应分离
- 进入 Vision 路径时，应强制产出页面图像，而不是再受 `screenshot.enabled` 门控

### High-4: PPTX 的 `--ocr` / `--ocr --llm` 也被 `screenshot` 开关错误短路

严重度: High

现象:

- `PptxConverter.convert()` 文档字符串声明:
  - `--ocr --llm`: Extract text + render slides for LLM Vision
  - `--ocr only`: Extract text + commented slide images
- 实际实现中:
  - `_convert_with_slide_images()` 只有 `enable_screenshot` 时才渲染 slides
  - `_render_slides_for_llm()` 也只有 `enable_screenshot` 时才渲染 slides

证据:

- `packages/markitai/src/markitai/converter/office.py:104`
- `packages/markitai/src/markitai/converter/office.py:116`
- `packages/markitai/src/markitai/converter/office.py:169`
- `packages/markitai/src/markitai/converter/office.py:175`
- `packages/markitai/src/markitai/converter/office.py:470`
- `packages/markitai/src/markitai/converter/office.py:490`
- `packages/markitai/src/markitai/converter/office.py:497`
- `packages/markitai/src/markitai/config.py:286`

影响:

- 默认配置下，这两条 OCR 路径都会静默缺失 slide images
- 用户按帮助文本理解 CLI 时，拿到的结果与承诺不一致

建议:

- PPTX 的 OCR / OCR+LLM 路径应始终生成 slide images
- `--screenshot` 应只控制“额外截图能力”，不应反向破坏 OCR/Vision 主流程

### Medium-1: 交互模式的 “Auto-detect” 选项实际上会静默禁用 LLM

严重度: Medium

现象:

- `prompt_configure_provider()` 提供 `Auto-detect (Claude CLI / Copilot CLI)`
- 但对 `result == "auto"` 没有任何处理
- 函数直接 `return False`
- `run_interactive()` 把这个 `False` 解释为“关闭 LLM”

证据:

- `packages/markitai/src/markitai/cli/interactive.py:353`
- `packages/markitai/src/markitai/cli/interactive.py:360`
- `packages/markitai/src/markitai/cli/interactive.py:384`
- `packages/markitai/src/markitai/cli/interactive.py:567`
- `packages/markitai/src/markitai/cli/interactive.py:568`

影响:

- 用户选择“自动检测”，结果不是自动检测失败，而是直接把 LLM 关掉
- 这是明显的交互逻辑故障

建议:

- 为 `auto` 分支补上 `detect_llm_provider()` 重试/重新检测逻辑
- 若仍未检测到，应继续留在配置流程，而不是隐式禁用 LLM

### Medium-2: 认证预检顺序不合理，会在本不执行转换的命令上提前触发

严重度: Medium

现象:

- `run_workflow()` 一进入就执行 `preflight_auth_check()`
- 但 URL 缺少 `-o` 的参数校验在后面
- 单文件 `--dry-run` 的早退也在后面

证据:

- `packages/markitai/src/markitai/cli/main.py:542`
- `packages/markitai/src/markitai/cli/main.py:553`
- `packages/markitai/src/markitai/cli/main.py:642`
- `packages/markitai/src/markitai/cli/main.py:645`
- `packages/markitai/src/markitai/cli/processors/file.py:77`

影响:

- `--dry-run` 也可能先要求登录
- 参数不完整的 URL 命令也可能先进入认证交互
- 这会拖慢命令、污染 stdout/stderr，并损害脚本化调用体验

建议:

- 先完成:
  - 输入模式判定
  - 输出目录必填校验
  - dry-run 快速返回
- 再做 provider 认证预检

### Medium-3: `doctor --fix` 的“修复”实现既不完整，也过于宽泛

严重度: Medium

现象:

- `_install_component("playwright")` 只执行 `uv run playwright install chromium`
- 但最常见缺失场景是包本身未安装，需要先 `uv add playwright`
- `doctor --fix` 会把所有 `missing/warning` key 都喂给 `_install_component()`
- 其中包含 `vision-model` 这种根本不是可安装组件的检查项

证据:

- `packages/markitai/src/markitai/cli/commands/doctor.py:72`
- `packages/markitai/src/markitai/cli/commands/doctor.py:87`
- `packages/markitai/src/markitai/cli/commands/doctor.py:94`
- `packages/markitai/src/markitai/cli/commands/doctor.py:97`
- `packages/markitai/src/markitai/cli/commands/doctor.py:754`
- `packages/markitai/src/markitai/cli/commands/doctor.py:759`

影响:

- `--fix` 会承诺“尝试修复”，但对真实缺失场景经常修不动
- 还会对非安装类检查项执行不合语义的“修复”

建议:

- 为每个 fixable component 建立明确映射
- 区分:
  - installable dependency
  - auth issue
  - config issue
  - model availability issue

### Medium-4: PDF 普通转换在 `output_dir is None` 时返回悬空图片路径

严重度: Medium

现象:

- `PdfConverter.convert()` 在没有 `output_dir` 时把图片写到临时目录
- 返回前直接 `shutil.rmtree(temp_dir)`
- 但 `images` 中的 `ExtractedImage.path` 仍指向该目录

证据:

- `packages/markitai/src/markitai/converter/pdf.py:96`
- `packages/markitai/src/markitai/converter/pdf.py:103`
- `packages/markitai/src/markitai/converter/pdf.py:229`
- `packages/markitai/src/markitai/converter/pdf.py:233`

影响:

- API 返回对象包含不可访问路径
- 对 CLI 以外的调用者尤其危险，因为结果对象看起来完整，实际已损坏

建议:

- 无 `output_dir` 时不要返回落盘路径
- 或保留临时目录到调用者完成消费后再清理

### Medium-5: OCR 共享单例忽略配置差异，首次配置会污染后续调用

严重度: Medium

现象:

- `OCRProcessor.get_shared_engine()` 只在 `_global_engine is None` 时初始化
- 会保存 `_global_config = config`
- 但后续完全不比较新旧配置

证据:

- `packages/markitai/src/markitai/ocr.py:52`
- `packages/markitai/src/markitai/ocr.py:65`
- `packages/markitai/src/markitai/ocr.py:69`
- `packages/markitai/src/markitai/ocr.py:70`

影响:

- 首次以 `zh` 初始化后，再以 `en` 创建 OCRProcessor，仍会复用旧引擎
- 多语言 OCR 正确性会被“首调者获胜”的全局状态污染

建议:

- 以配置指纹为 key 维护共享引擎
- 或在配置变化时强制重建引擎

### Medium-6: 截图压缩兜底会写出“扩展名与内容格式不一致”的坏文件

严重度: Medium

现象:

- `save_screenshot()` 正常路径按 `output_format` 保存
- 超过阈值后最后兜底固定写成 `JPEG`
- 但输出路径保持原扩展名不变

证据:

- `packages/markitai/src/markitai/image.py:663`
- `packages/markitai/src/markitai/image.py:681`
- `packages/markitai/src/markitai/image.py:703`
- `packages/markitai/src/markitai/image.py:706`
- `packages/markitai/src/markitai/image.py:708`

影响:

- 用户选择 `png` / `webp` 时，最终可能得到文件名是 `.png` / `.webp`，内容却是 JPEG
- 这会破坏 MIME、后续处理链和调试判断

建议:

- 兜底压缩时同步修正扩展名与 MIME
- 或在原格式内完成降采样/重新编码，不要跨格式偷换

### Medium-7: `webextract` 的 adaptive retry 只有标记，没有真正重试

严重度: Medium

现象:

- `extract_web_content()` 在 markdown 过短时，只设置 `diagnostics["adaptive_retry_used"] = True`
- 没有重新选 root
- 没有删除 partial selectors
- 没有第二次抽取

证据:

- `packages/markitai/src/markitai/webextract/pipeline.py:38`
- `packages/markitai/src/markitai/webextract/pipeline.py:41`
- `packages/markitai/src/markitai/webextract/pipeline.py:50`
- `packages/markitai/src/markitai/webextract/pipeline.py:51`
- `packages/markitai/src/markitai/webextract/pipeline.py:52`

影响:

- 诊断字段会误导上层，以为已经做过“自适应重试”
- 实际上短内容页面没有任何补救动作

建议:

- 要么实现真正的二次提取策略
- 要么移除该诊断字段，避免伪 telemetry

### Medium-8: PDF/OCR 路径在共享线程池里再创建内部线程池，容易造成线程过量和内存峰值

严重度: Medium

现象:

- `workflow/core.py` 已经把重型转换放进共享 executor
- `PdfConverter` 内部在 `_render_pages_parallel()`、OCR 分支又额外创建 `ThreadPoolExecutor`

证据:

- `packages/markitai/src/markitai/workflow/core.py:230`
- `packages/markitai/src/markitai/workflow/core.py:253`
- `packages/markitai/src/markitai/converter/pdf.py:239`
- `packages/markitai/src/markitai/converter/pdf.py:322`
- `packages/markitai/src/markitai/converter/pdf.py:526`
- `packages/markitai/src/markitai/converter/pdf.py:572`

影响:

- batch 场景下，一个“重型任务”内部还会再扩张线程数
- 会放大 CPU oversubscription、内存压力和 PDF 文档重复打开次数

建议:

- 避免嵌套线程池
- 用共享 executor 或明确的 page-level 并发上限

### Low-1: 关键状态文件写入仍有多处非原子实现

严重度: Low

现象:

- SPA domain cache 直接 `open(..., "w")`
- interactive 的 `.env` 直接 `write_text()`
- `init` 生成 `.env` 模板和配置文件时也直接写

证据:

- `packages/markitai/src/markitai/fetch.py:613`
- `packages/markitai/src/markitai/fetch.py:616`
- `packages/markitai/src/markitai/cli/interactive.py:459`
- `packages/markitai/src/markitai/cli/interactive.py:475`
- `packages/markitai/src/markitai/cli/commands/init.py:357`
- `packages/markitai/src/markitai/cli/commands/init.py:367`
- `packages/markitai/src/markitai/cli/commands/init.py:371`
- `packages/markitai/src/markitai/cli/commands/init.py:374`

影响:

- 与仓库自己的“关键写入必须原子化”原则不一致
- 进程中断或写入失败时可能留下半写入文件

建议:

- 统一替换为 `security.py` 中的原子写入接口

### Low-2: PDF OCR+LLM 路径下，embedded image 收集器漏掉 `webp`

严重度: Low

现象:

- `_collect_embedded_images()` 正则仅匹配 `png|jpg|jpeg`
- 但 `_render_pages_for_llm()` 会使用 `self.config.image.format`

证据:

- `packages/markitai/src/markitai/converter/pdf.py:376`
- `packages/markitai/src/markitai/converter/pdf.py:632`
- `packages/markitai/src/markitai/converter/pdf.py:634`

影响:

- 当图片格式设为 `webp` 时，嵌入图片会被静默漏收集

建议:

- 正则应与支持格式列表保持一致

### Low-3: 异步图片处理分支没有关闭 `compressed_img`

严重度: Low

现象:

- `process_and_save_async()` 在 `compress_enabled` 路径只读取 `compressed_img.size`
- 没有显式关闭 `compressed_img`

证据:

- `packages/markitai/src/markitai/image.py:991`
- `packages/markitai/src/markitai/image.py:992`
- `packages/markitai/src/markitai/image.py:998`

影响:

- 长批次处理时会增加不必要的内存占用

建议:

- 与同步路径对齐，显式关闭临时 `PIL.Image`

## 5. 设计与架构风险

这一部分不是“已证明必炸”的缺陷，但值得尽快收敛。

### 5.1 Fetch 相关全局单例对配置不敏感

现状:

- `get_fetch_cache()` 只在首次创建时使用 `cache_dir` / `max_size_bytes`
- `_get_jina_client()` 只在首次创建时使用 `timeout` / `proxy`
- `_get_playwright_renderer()` 也只在首次创建时采纳 `proxy` / session cache 设置

证据:

- `packages/markitai/src/markitai/fetch.py:831`
- `packages/markitai/src/markitai/fetch.py:844`
- `packages/markitai/src/markitai/fetch.py:1294`
- `packages/markitai/src/markitai/fetch.py:1301`
- `packages/markitai/src/markitai/fetch.py:1348`
- `packages/markitai/src/markitai/tests/unit/test_fetch.py:1962`

风险:

- 在单进程内切换配置时，后续调用会静默沿用第一次的参数
- 这类问题在 CLI 单次运行中不一定爆，但在库化/长生命周期进程中会变成隐蔽 bug

### 5.2 FetchCache 的同步锁和异步锁彼此独立

现状:

- 复用同一 SQLite connection
- 同步方法受 `threading.Lock` 保护
- 异步方法受 `asyncio.Lock` 保护
- 两把锁互不互斥

证据:

- `packages/markitai/src/markitai/fetch.py:152`
- `packages/markitai/src/markitai/fetch.py:169`
- `packages/markitai/src/markitai/fetch.py:170`
- `packages/markitai/src/markitai/fetch.py:171`
- `packages/markitai/src/markitai/fetch.py:183`
- `packages/markitai/src/markitai/fetch.py:295`

风险:

- 一旦同步/异步调用并发混用，同一连接可能在两条路径上同时操作
- 这比“每次新建连接”的方案更省开销，但一致性边界更脆弱

## 6. 测试覆盖观察

当前测试基线不弱，但以下缺口值得补上:

- 没有测试 `run_interactive_mode()` 是否传播子进程失败退出码
  - 现有只测 `-I/--interactive` 能不能识别
  - 证据: `packages/markitai/tests/unit/cli/test_main.py:1`
- 没有测试交互模式里 `Auto-detect` 分支的真实行为
  - 现有 `run_interactive` 测试把 `prompt_configure_provider` 整体 mock 掉了
  - 证据: `packages/markitai/tests/unit/cli/test_interactive.py:257`
- 没有测试 `doctor` 总结是否会在 auth/LLM 错误下错误显示 `all good`
  - `doctor --fix` 相关测试也只是 patch 掉 `_doctor_impl`
  - 证据: `packages/markitai/tests/unit/cli/test_doctor.py:146`
- `config list` 的集成测试覆盖了 `json` 和 `table`，没有覆盖 `yaml`
  - 证据: `packages/markitai/tests/integration/test_cli_full.py:543`

## 7. 修复优先级建议

建议按下面顺序处理:

### 第一批: 立即修

- 修复 `doctor` 假阳性总结
- 修复 `--interactive` 吞退出码
- 修复 PDF `--ocr --llm` 图像缺失
- 修复 PPTX `--ocr` / `--ocr --llm` 图像缺失

### 第二批: 紧随其后

- 修复交互模式 `Auto-detect` 逻辑
- 调整 auth preflight 顺序
- 修复 `doctor --fix`
- 修复 PDF 悬空图片路径
- 修复截图压缩格式错配
- 修复 OCR 单例配置污染

### 第三批: 工程质量收口

- 实现真正的 adaptive retry，或删除伪标记
- 消除 PDF 内部嵌套线程池
- 收敛非原子写入
- 补齐 CLI 关键分支回归测试

## 8. 最终判断

这个仓库不是“整体不可用”，但当前有多处“表面功能存在、关键路径语义不成立”的实现，尤其集中在 CLI 交互与 OCR/Vision 组合路径。它们的共同问题不是代码风格，而是:

- 注释和帮助文本承诺了一种行为
- 实际代码在关键条件判断处把该行为短路了
- 现有测试没有覆盖到真实用户路径

如果只看 lint、type check 和大多数单测，很容易高估当前实现质量。优先把这批行为级缺陷补上，再继续做性能和架构收口，收益会最大。
