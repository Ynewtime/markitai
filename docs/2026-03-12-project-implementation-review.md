# Markitai 项目实现评审报告

评审时间: 2026-03-12

## 1. 评审范围与方法

本次评审聚焦以下内容:

- 当前主干实现的业务故障风险
- 会导致吞吐下降、内存升高、重复 I/O 的性能瓶颈
- 缓存、一致性、退出语义、批处理路径中的系统性问题
- 测试与静态检查能否覆盖上述风险

本次评审基于:

- 全量静态审查: `cli/`、`workflow/`、`fetch/`、`webextract/`、`converter/`、`llm/`、`providers/`
- 本地验证:
  - `uv run pytest -q`
    - 结果: `2667 passed, 1 skipped`
  - `UV_CACHE_DIR=/tmp/markitai-uv-cache uv run pyright`
    - 结果: `0 errors, 0 warnings, 0 informations`
  - `UV_CACHE_DIR=/tmp/markitai-uv-cache uv run ruff check packages/markitai/src/markitai packages/markitai/tests`
    - 结果: `All checks passed!`
- 定向复现:
  - `FetchCache` round-trip 复现了 `screenshot_path/static_content/browser_content` 丢失
  - `SQLiteCache._compute_hash()` 复现了长文本中段修改 hash 不变

## 2. 总体结论

项目的工程化基础是稳的:

- 模块边界清晰，转换、抓取、LLM、CLI、工作流已做出分层
- 类型检查、lint、测试基线都在，而且当前主测试集是绿的
- 安全基线相对可靠，原子写入、路径检查、symlink 防护都已落地

但当前实现仍然存在一批“测试通过，但真实用户路径会出错或明显变慢”的问题，主要集中在 4 个主题:

1. URL 抓取缓存与策略语义不一致，导致显式选项和缓存命中互相覆盖
2. stdout / batch / skip 这些边界路径存在静默失效或多做无用功
3. LLM 与 Vision 缓存键设计过粗，存在静默复用旧结果的 correctness 风险
4. 若干热路径仍有明显重复 I/O、重复 HTML 解析、事件循环阻塞和连接复用缺失

从优先级上看，最值得立即处理的是:

1. URL batch 模式忽略 `--screenshot` / `--screenshot-only`
2. Fetch cache 只按 URL 命中，显式抓取策略会被缓存吞掉
3. Fetch cache 命中后丢失 `screenshot/static_content/browser_content`
4. stdout 模式会输出注定失效的图片/截图引用
5. LLM / Vision 缓存键过粗，存在静默返回旧结果的问题

## 3. 与前次评审相比已确认修复的问题

以下问题在当前代码中已不再成立:

- `doctor` 总结不再只看 required deps，已经通过 `_has_errors()` 汇总所有检查类别
- 交互模式会传播子进程退出码，不再无条件 `ctx.exit(0)`
- PDF / PPTX 的 OCR+LLM 路径已经与 `screenshot.enabled` 解耦，不再默认失效

因此本报告不重复列出这些旧问题，只记录当前版本仍然成立的风险。

## 4. 详细问题

### High-1: URL batch 模式静默忽略 `--screenshot` / `--screenshot-only`

严重度: High

现象:

- 单 URL 路径会把 `screenshot`、`screenshot_dir`、`screenshot_config` 传给 `fetch_url()`
- URL batch 路径没有传这些参数
- batch 路径里也没有 `screenshot_only` 的专门分支

证据:

- `packages/markitai/src/markitai/cli/processors/url.py:173-189`
- `packages/markitai/src/markitai/cli/processors/url.py:263-271`
- `packages/markitai/src/markitai/cli/processors/url.py:322-378`
- `packages/markitai/src/markitai/cli/processors/url.py:701-710`
- `packages/markitai/src/markitai/cli/processors/url.py:782-793`

影响:

- `markitai urls.txt -o out --screenshot` 在 batch 模式下不会真正抓截图
- `--screenshot-only` 在 URL batch 中没有语义闭环，和单 URL 行为不一致
- 依赖 screenshot 的 Vision URL 增强在 batch 路径会直接退化

为什么是问题:

- 这是用户显式 CLI 选项被静默忽略，不是质量下降，而是功能语义失真
- 单 URL 和 batch URL 同属一个产品能力，行为不应出现这种分叉

建议:

- 让 `process_url_batch()` 与 `process_url()` 使用同一组 screenshot 参数
- 为 URL batch 补齐 `screenshot_only` 语义分支
- 增加 batch URL 的 screenshot / screenshot-only 集成测试

### High-2: Fetch cache 只按 URL 命中，显式抓取策略会被缓存吞掉

严重度: High

现象:

- `FetchCache._compute_hash()` 只接受 `url`
- `fetch_url()` 在进入显式策略分支前，已经先做了 `cache.aget(url)` / `cache.aget_with_validators(url)`
- 这意味着同一 URL 只要已有缓存，`--playwright`、`--jina`、`--defuddle`、`--cloudflare` 都可能直接返回旧结果

证据:

- `packages/markitai/src/markitai/fetch.py:265-271`
- `packages/markitai/src/markitai/fetch.py:2411-2458`
- `packages/markitai/src/markitai/fetch.py:2461-2527`

影响:

- 显式策略不再可信，用户看到的是“缓存里上一次谁先写入就用谁”
- `--playwright` 想强制浏览器渲染时，可能直接吃到旧的 static/defuddle/jina 结果
- 与策略绑定的抓取质量、截图、JS 渲染能力都可能被静默绕过

为什么是问题:

- 缓存命中条件没有和输出语义绑定
- CLI 明确提供“强制策略”的控制能力，但缓存层把这层控制抹掉了

建议:

- Fetch cache key 至少纳入:
  - URL
  - strategy family
  - screenshot mode
  - 会影响输出的 fetch 配置
- 或者只允许复用 `strategy_used` 兼容的缓存结果
- 补一个“同 URL + 不同策略”不能串缓存的测试

### High-3: Fetch cache 命中后会丢失 screenshot / multi-source 内容

严重度: High

现象:

- `FetchResult` 明明定义了 `screenshot_path`、`static_content`、`browser_content`
- 但 `FetchCache._set_unlocked()` / `_get_unlocked()` 实际只保存和恢复 `content`、`strategy_used`、`title`、`url`、`final_url`、`metadata`
- 我本地复现后确认，cache round-trip 后这 3 个字段都变成 `None`

证据:

- `packages/markitai/src/markitai/fetch.py:121-141`
- `packages/markitai/src/markitai/fetch.py:282-290`
- `packages/markitai/src/markitai/fetch.py:311-360`
- `packages/markitai/src/markitai/fetch.py:1840-1852`

本地复现:

- 写入包含 `screenshot_path/static_content/browser_content` 的 `FetchResult`
- 再 `cache.get(url)`，得到:
  - `screenshot_path: None`
  - `static_content: None`
  - `browser_content: None`

影响:

- URL 缓存命中后，多源内容增强会退化成单源
- screenshot 命中后仍可能再次触发 Playwright 去补抓截图
- 同一个 URL 的“首跑输出”和“缓存命中输出”不再等价

为什么是问题:

- 这是典型的 lossy cache，缓存后的对象不再等价于原对象
- 对用户来说，缓存命中不应改变语义，只应减少成本

建议:

- Fetch cache 序列化时完整保留 `screenshot_path/static_content/browser_content`
- 如果 `screenshot_path` 不能直接缓存，应缓存相对路径或元数据，并在命中后恢复
- 增加“缓存命中前后 URL Vision 行为一致”的测试

### High-4: stdout 模式会输出注定失效的图片/截图引用

严重度: High

现象:

- 单文件 stdout 模式会先创建临时目录做转换
- 然后直接把最终 Markdown 打到 stdout
- 最后把临时目录整个删掉
- 但 Markdown 内部仍可能包含 `.markitai/assets/...` 或 `.markitai/screenshots/...` 的引用

证据:

- `packages/markitai/src/markitai/cli/processors/file.py:122-127`
- `packages/markitai/src/markitai/cli/processors/file.py:257-261`
- `packages/markitai/src/markitai/cli/processors/file.py:294-298`
- `packages/markitai/src/markitai/converter/image.py:63-77`
- `packages/markitai/src/markitai/workflow/core.py:603-614`

影响:

- `markitai sample.jpg`
- 带图片的 PDF / PPTX
- 开启 screenshot / vision 的文档

这些 stdout 输出在落盘后会天然带着坏链接。

为什么是问题:

- CLI 对 stdout 模式的承诺不只是“能打印字符串”，还应该是“打印出来的 Markdown 可用”
- 当前实现对无图纯文本可用，但对带资产的内容天然不可用

建议:

- stdout 模式下禁止生成外部资产引用，改成:
  - data URI
  - 显式报错/告警
  - 或要求必须指定 `-o`
- 至少增加一个测试，验证 stdout 最终输出中的图片引用是否仍然可解析

### High-5: LLM 持久缓存会对长文档中段修改、提示词变更和模型切换复用旧结果

严重度: High

现象:

- `SQLiteCache._compute_hash()` 只使用 `prompt + length + head + tail`
- 代码注释已明确承认: `content > 50k` 时，中段修改可能不会改变 hash
- 上层调用 `PersistentCache` 时，`prompt` 传入的是 `"cleaner"` 这类类别名，而不是实际 prompt 内容
- 模型/provider 也不参与缓存 key

证据:

- `packages/markitai/src/markitai/llm/cache.py:105-115`
- `packages/markitai/src/markitai/llm/document.py:299-308`
- `packages/markitai/src/markitai/llm/document.py:321-324`
- `packages/markitai/src/markitai/llm/document.py:350-352`

本地复现:

- 两个长度相同、前 25k 和后 25k 完全相同、只有中段不同的字符串
- `_compute_hash("prompt", content1)` 与 `_compute_hash("prompt", content2)` 完全相同

影响:

- 长文档中段修改后，LLM 清洗结果可能直接命中旧缓存
- prompt 文件更新后，旧缓存仍可能被复用
- 切换 provider / model 后，也可能继续复用上一次模型的结果

为什么是问题:

- 这是 silent corruption：用户不会看到报错，只会拿到“看起来正常”的陈旧结果
- 对内容处理产品来说，这比显式失败更危险

建议:

- 缓存 key 至少纳入:
  - 完整内容哈希或分块哈希
  - 实际 prompt 模板内容哈希
  - 模型/provider 标识
  - 关键行为开关
- 为长文档中段修改、prompt 变更、模型切换补缓存失效测试

### High-6: Vision 图片缓存忽略 document context，会跨文档复用错误描述

严重度: High

现象:

- Vision 分析 prompt 中会拼入 `document_context`
- 但持久缓存只用 `cache_key="image_analysis"` + `image_fingerprint`
- `context=` 参数只用于 no-cache pattern 判断，并不参与 cache key

证据:

- `packages/markitai/src/markitai/llm/vision.py:108-112`
- `packages/markitai/src/markitai/llm/vision.py:129-133`
- `packages/markitai/src/markitai/llm/vision.py:170-171`
- `packages/markitai/src/markitai/llm/vision.py:324-335`
- `packages/markitai/src/markitai/llm/vision.py:372-376`
- `packages/markitai/src/markitai/llm/vision.py:494-495`
- `packages/markitai/src/markitai/llm/cache.py:471-500`

影响:

- 同一张图在不同文档上下文里，本应产出不同 caption/description
- 现在会被上一份文档的结果静默覆盖
- 多语言文档、不同业务语境下尤其容易出错

为什么是问题:

- 这里的 cache key 和 prompt 输入不一致
- 缓存返回的是语义结果，不是纯视觉特征，因此不能只看图片哈希

建议:

- 将 `document_context`、prompt 版本、模型指纹纳入 Vision cache key
- 为“同图不同上下文不能串缓存”增加单测

### Medium-1: 重任务识别漏掉了 OCR+LLM / PPTX OCR 路径，批处理容易出现内存尖峰

严重度: Medium

现象:

- `workflow.core.convert_document()` 把“是否重任务”主要绑定在 `ctx.config.screenshot.enabled`
- 但 PDF 的 `--ocr --llm` 会无条件渲染 page images
- PPTX 的 `--ocr --llm` 和 `--ocr` 也会无条件渲染 slide images

证据:

- `packages/markitai/src/markitai/workflow/core.py:233-238`
- `packages/markitai/src/markitai/converter/pdf.py:95-100`
- `packages/markitai/src/markitai/converter/pdf.py:672-675`
- `packages/markitai/src/markitai/converter/office.py:113-123`
- `packages/markitai/src/markitai/converter/office.py:469-474`

影响:

- 在 screenshot 关闭但 OCR/Vision 开启的情况下，真正的重任务不会进入 heavy semaphore
- batch 下多个 PDF / PPTX 可能并发渲染，造成 CPU / 内存突刺

为什么是问题:

- 最近转换器的行为已经变了，但调度层的“重任务判定”没有同步演进

建议:

- 将以下条件也纳入 heavy task 判定:
  - PDF `ocr && llm`
  - PPTX `ocr`
  - PPTX `ocr && llm`
- 最好直接由 converter 暴露 workload hint，而不是在 workflow 靠扩展名猜

### Medium-2: 认证预检发生在参数/模式校验之前，错误路径会先触发无关登录交互

严重度: Medium

现象:

- `run_workflow()` 一开始就执行 `preflight_auth_check()`
- URL / URL batch 对 `-o` 的必填校验在后面

证据:

- `packages/markitai/src/markitai/cli/main.py:543-610`
- `packages/markitai/src/markitai/cli/main.py:621-665`

影响:

- 用户如果执行 `markitai --llm https://example.com` 但漏掉 `-o`
- 当前实现可能先去做 provider 认证检查，甚至弹登录提示
- 然后才告诉用户“URL mode requires -o”

为什么是问题:

- 这会放大失败路径的噪音和等待时间
- 对 shell/CI 也不友好，因为简单参数错误会被认证逻辑污染

建议:

- 先做:
  - 输入模式判定
  - 输出目录校验
  - dry-run / skip 快速返回
- 再做 auth preflight

### Medium-3: `on_conflict=skip` 的跳过判定太晚，很多场景会先做完整重活再跳过

严重度: Medium

现象:

- 文件工作流要先完成 conversion，之后才 resolve output
- 单 URL 先 fetch，之后才 resolve output
- URL batch 甚至会先 fetch、再下载图片，之后才 resolve output

证据:

- `packages/markitai/src/markitai/workflow/core.py:866-873`
- `packages/markitai/src/markitai/cli/processors/url.py:168-220`
- `packages/markitai/src/markitai/cli/processors/url.py:742-764`

影响:

- 对已存在输出的 PDF / PPTX / URL，重复运行时仍会消耗转换、网络、图片下载和部分 LLM 前置成本
- 对大批量重复执行尤其浪费

为什么是问题:

- 这些输出名在多数路径里都是可提前推导的，不需要等重活做完再判断

建议:

- 把 output path resolve 前移到真正重活之前
- 对必须在后面 resolve 的少数情况，至少给出明确理由

### Medium-4: screenshot 压缩兜底会偷偷改成 `.jpg`，但上层元数据仍保留旧扩展名

严重度: Medium

现象:

- `save_screenshot()` 在极端压缩分支会把输出后缀改成 `.jpg`
- 但 PDF / PPTX 调用方继续使用旧的 `image_path/image_name/mime_type`

证据:

- `packages/markitai/src/markitai/image.py:703-723`
- `packages/markitai/src/markitai/converter/pdf.py:304-315`
- `packages/markitai/src/markitai/converter/office.py:431-446`

影响:

- `page_images` / `slide_images` 元数据可能指向不存在文件
- 后续 Vision 处理、注释引用、调试排障都会被误导

建议:

- 让 `save_screenshot()` 返回最终路径和最终格式
- 或者完全禁止在底层函数内部偷偷改后缀，由调用方决定 fallback 格式

### Medium-5: URL hot path 存在明显的同步阻塞和连接复用缺失

严重度: Medium

现象:

- `fetch_with_static_conditional()` 在 async 函数里同步写临时文件并调用 `md.convert()`
- `fetch_http.HttpxClient` / `CurlCffiClient` 每次请求都新建并销毁 client/session

证据:

- `packages/markitai/src/markitai/fetch.py:2009-2018`
- `packages/markitai/src/markitai/fetch_http.py:71-85`
- `packages/markitai/src/markitai/fetch_http.py:99-118`

影响:

- URL batch 并发度越高，static 路径越容易被同步转换阶段拖慢
- 缺少连接池会反复做 TCP/TLS 建连，削弱 static 路径本该拥有的低成本优势

为什么是问题:

- 这是典型的热路径工程问题: 最常用、最便宜的策略反而没有被优化

建议:

- 把 `markitdown.convert()` 放到线程池或共享 executor
- 为 static backend 引入共享 client 生命周期

### Medium-6: Playwright / webextract 路径存在重复 HTML→Markdown 开销

严重度: Medium

现象:

- Playwright 路径会先做一轮 `_html_to_markdown(html_content)`
- 之后再尝试 `extract_web_content(html_content, ...)`
- `webextract` 内部又会重新 parse/sanitize，并重新构造 `MarkItDown`

证据:

- `packages/markitai/src/markitai/fetch_playwright.py:476-493`
- `packages/markitai/src/markitai/webextract/pipeline.py:34-66`
- `packages/markitai/src/markitai/webextract/pipeline.py:134-146`

影响:

- 大页面和 URL batch 会重复做 DOM 解析、序列化、Markdown 转换
- CPU 和内存消耗被放大

建议:

- 先走 native webextract，只有质量不达标时才 fallback 到 `_html_to_markdown`
- 共享 `MarkItDown` 实例，避免在热路径中反复构造

### Medium-7: 代理自动探测/绕过语义不闭合，`auto_proxy=False` 和 `NO_PROXY` 不能全链路生效

严重度: Medium

现象:

- Playwright 会尊重 `auto_proxy=False`
- 但 static / Jina / Defuddle / Cloudflare 仍直接调用 `_detect_proxy()`
- `_detect_proxy()` 虽然记录了 `_detected_proxy_bypass`，但后续没有按目标 URL 做 bypass 应用
- 并且“本机常见端口能连通”会被直接当作 HTTP 代理

证据:

- `packages/markitai/src/markitai/fetch.py:1032-1098`
- `packages/markitai/src/markitai/fetch.py:1186-1194`
- `packages/markitai/src/markitai/fetch.py:1350-1358`
- `packages/markitai/src/markitai/fetch.py:1893-1900`
- `packages/markitai/src/markitai/fetch.py:2141-2145`
- `packages/markitai/src/markitai/fetch.py:2390-2391`

影响:

- `auto_proxy=False` 对多数后端无效
- `NO_PROXY` 命中的地址仍可能经过代理
- 本地端口误判会把抓取流量错误导向非代理服务

建议:

- 统一抽象 `get_proxy_for_url(url, respect_auto_proxy=True)`
- 真正实现 `NO_PROXY` / bypass 语义
- 去掉“端口开放即代理”的弱启发式，至少增加代理握手校验

### Medium-8: “已学习 SPA / browser-first” 的承诺和真实策略排序相互矛盾

严重度: Medium

现象:

- 代码注释、变量名、缓存命名都把这条路径描述为 `browser-first`
- 但 policy 对 `known_spa=True` 的真实顺序仍然是 `defuddle -> jina -> playwright -> ...`

证据:

- `packages/markitai/src/markitai/fetch.py:2551-2562`
- `packages/markitai/src/markitai/fetch.py:2753-2803`
- `packages/markitai/src/markitai/fetch_policy.py:159-160`
- `packages/markitai/src/markitai/fetch_policy.py:216-220`

影响:

- 已知 JS-only 站点仍会先把请求送去外部 extractor，再退回本地浏览器
- 延迟更高，也更容易把本来应该“本地渲染”的页面再次发送给第三方

为什么是问题:

- 行为、注释、命名和设计目标已经不一致
- 这会直接干扰后续对 SPA 域名学习缓存是否有效的判断

建议:

- 二选一:
  - 要么真正把 known SPA 调整为 browser-first
  - 要么把注释和命名改成“skip static but still prefer cloud extractors”
- 补一个真正覆盖 `defuddle/jina/playwright` 相对顺序的测试

### Medium-9: `webextract` 候选节点评分在大 DOM 上会明显退化

严重度: Medium

现象:

- 每个候选节点都会重复执行 `get_text()`、`find_all("p")`、`find_all("a")`
- pipeline 里还会在热路径中额外计算 diagnostics

证据:

- `packages/markitai/src/markitai/webextract/scoring.py:19-23`
- `packages/markitai/src/markitai/webextract/scoring.py:28-52`
- `packages/markitai/src/markitai/webextract/pipeline.py:43`
- `packages/markitai/src/markitai/webextract/pipeline.py:130-131`

影响:

- DOM 越深、容器越多，评分阶段越接近反复扫描整棵子树
- static 和 Playwright 两条 URL 链路都会被拖慢

为什么是问题:

- 为了少量启发式分数，当前实现付出了重复整树遍历的成本
- 这类开销在小 fixture 上不显，但在真实新闻页、论坛页、文档页上会迅速放大

建议:

- 预聚合文本长度、段落数、链接数等统计
- 把 diagnostics 从热路径移到可选调试路径
- 增加至少一个大 DOM 基准测试或性能回归测试

## 5. 测试盲区

当前测试数量不少，但以下关键行为没有被有效覆盖:

- URL batch 的 screenshot / screenshot-only 路径
- 同 URL 不同 fetch strategy 的缓存隔离
- Fetch cache round-trip 后多源字段是否完整
- stdout 模式下 Markdown 中的图片/截图引用是否仍然有效
- 长文档中段修改、prompt 变更、模型切换后的缓存失效
- 同图不同 document context 的 Vision cache 隔离
- screenshot 极限压缩后 PDF/PPTX 元数据路径是否仍一致
- 大 DOM / 大 URL batch 的性能回归保护

## 6. 优先修复建议

建议按以下顺序推进:

1. 修 URL 语义问题
   - batch screenshot
   - fetch cache key
   - fetch cache round-trip 完整性
2. 修 correctness 缓存问题
   - LLM 持久缓存键
   - Vision 图片缓存键
3. 修输出边界问题
   - stdout 资产引用
   - screenshot 扩展名漂移
4. 修批处理性能问题
   - heavy task 判定
   - skip 提前
   - static client 连接复用
   - async 热路径阻塞

## 7. 最终判断

Markitai 当前不是“不可用”，但它已经进入一个很典型的阶段:

- 正常主路径基本可用
- 工程化基础也不错
- 但缓存、批处理、stdout、URL 多策略这几类边界路径已经开始出现系统性偏差

这些问题的共同特点是:

- 不一定让测试红
- 不一定直接抛异常
- 但会让用户拿到错误语义、低质量结果或明显变慢的体验

因此下一轮优化，不建议再继续堆功能，而应优先修复“语义正确性 + 热路径成本 + 缓存一致性”这三件事。对这个项目而言，这会比再加新 provider 或新抓取策略更能提升真实可用性。
