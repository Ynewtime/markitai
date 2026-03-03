# Scrapling: a deep technical architecture analysis

**Scrapling is a Python web scraping framework that unifies HTTP fetching, stealth browser automation, adaptive element tracking, and spider-based crawling into a single library — achieving parsing speeds on par with Scrapy/lxml while adding unique self-healing selector capabilities that no competing library offers.** The library, created by Karim Shoair (D4Vinci), has reached ~17.7k GitHub stars and version 0.4.1 as of February 2026. Its core innovation lies in a fingerprint-and-similarity algorithm that automatically relocates page elements after website redesigns, eliminating the #1 maintenance burden in web scraping. Built on lxml for parsing, curl_cffi for TLS-impersonating HTTP, and Playwright/Camoufox for stealth browser automation, Scrapling occupies a unique position in the Python scraping ecosystem as a lightweight yet full-featured alternative to the Scrapy + Selenium/Playwright stack.

---

## Technical architecture: four layers from UI to storage

Scrapling follows a **four-layer architecture**: User Interfaces → Fetching Engines → Parsing Engine → Support Systems. Each layer is independently useful, and the modular install system (`pip install scrapling` for parser-only, `scrapling[fetchers]` for browsers, `scrapling[all]` for everything) reflects this separation.

### The dependency stack

The tech stack choices reveal deliberate optimization at every level:

| Layer | Library | Why chosen |
|-------|---------|------------|
| **HTML Parsing** | `lxml` + `cssselect` | Fastest Python HTML parser. Benchmarks show **2.02ms** for 5,000 nested elements vs 1,584ms for BeautifulSoup — a **784× speed advantage** |
| **CSS→XPath Translation** | Custom `HTMLTranslator` (adapted from Parsel, BSD) | Adds Scrapy-compatible `::text` and `::attr()` pseudo-elements |
| **HTTP Client** | `curl_cffi` | Only Python HTTP library supporting **TLS fingerprint impersonation** and **HTTP/3 (QUIC)** — critical for evading bot detection at the network level |
| **Browser Automation** | `playwright` + `patchright` (stealth fork) | Patchright strips automation indicators that Playwright leaves behind |
| **Stealth Browser** | `camoufox` (modified Firefox) | Native fingerprint spoofing at the browser binary level — far harder to detect than JavaScript-level patches |
| **Fingerprint Generation** | `browserforge` | Creates realistic browser fingerprint datasets for impersonation |
| **JSON Serialization** | `orjson` | **10× faster** than Python's standard `json` module |
| **Config Validation** | `msgspec` | Struct-based type-safe validation for all fetcher arguments — catches configuration errors before browser launch |
| **Domain Extraction** | `tld` | Lightweight TLD extraction (replaced heavier `tldextract` in v0.4) |
| **HTML Entities** | `w3lib` | Handles HTML entity replacement (replaced internal `_html_utils.py` in v0.4) |
| **Async Framework** | `anyio` + optional `uvloop` | Powers the Spider framework; `uvloop` provides C-level event loop performance |
| **Adaptive Storage** | SQLite (built-in) | Stores element fingerprints for the self-healing selector system |
| **IP Geolocation** | `geoip2` | Matches timezone/locale spoofing to proxy IP location in StealthyFetcher |

### Async and sync: dual-mode across all components

Every fetcher offers both sync and async variants. The `Fetcher` class has a dedicated `AsyncFetcher` counterpart. Browser-based fetchers expose `.fetch()` (sync) and `.async_fetch()` (async) on the same class. Session classes split more explicitly: `DynamicSession` for sync, `AsyncDynamicSession` for async; `StealthySession` for sync, `AsyncStealthySession` for async. The `FetcherSession` (HTTP) works in both modes via context managers (`with` and `async with`). The Spider framework is **fully async**, built on `anyio` with optional `uvloop` acceleration. Browser sessions implement a **tab pool** (`max_pages=1–50`) that manages concurrent browser tabs, with `get_pool_stats()` reporting busy/free/error tab status.

### How it compares to alternatives

Scrapling occupies a distinct niche that none of the established tools fully cover:

| Capability | Scrapling | BeautifulSoup | Scrapy | Selenium | Playwright |
|------------|-----------|---------------|--------|----------|------------|
| Parsing speed (5k elements) | **~2ms** | ~1,584ms | ~2ms (Parsel) | N/A | N/A |
| Adaptive/self-healing selectors | ✅ Built-in | ❌ | ❌ | ❌ | ❌ |
| Anti-bot bypass | ✅ Built-in | ❌ | ❌ | Partial | Partial |
| JS rendering | ✅ | ❌ | Via Splash | ✅ | ✅ |
| Spider/crawling framework | ✅ (v0.4+) | ❌ | ✅ Core | ❌ | ❌ |
| TLS fingerprint impersonation | ✅ | ❌ | ❌ | N/A | N/A |
| Built-in proxy rotation | ✅ | ❌ | Via middleware | ❌ | ❌ |
| Memory per browser instance | Low (shared context) | N/A | N/A | **~500MB+** | ~200MB |

**BeautifulSoup** is parsing-only with no fetching, no JS rendering, and 784× slower parsing. **Scrapy** matches Scrapling's parsing speed (both use lxml) and has a far more mature spider/crawling ecosystem, but lacks adaptive selectors, anti-bot evasion, and browser automation. **Selenium** and **Playwright** are browser automation tools without built-in parsing optimization, anti-detection, or crawler frameworks. Scrapling's API deliberately borrows from both Scrapy (`.css()`, `.xpath()`, `.get()`, `.getall()`, `::text`, `::attr()`) and BeautifulSoup (`.find()`, `.find_all()`) to ease migration.

---

## The request-to-data pipeline: how a scraping workflow works

A typical Scrapling workflow follows a clean **Fetcher → Response/Selector → Parse → Extract** pipeline, with the same parsing API regardless of which fetcher is used.

### Entry points and the three-tier fetcher model

Scrapling provides a progressive escalation path. Start with the fastest option and upgrade only when needed:

1. **`Fetcher`** — Pure HTTP via curl_cffi. Fastest option. Supports TLS fingerprint impersonation (`impersonate='chrome'`), stealthy headers, and HTTP/3. Use for static pages and APIs.
2. **`StealthyFetcher`** — Uses Camoufox (modified Firefox) via Patchright. Adds canvas/WebGL fingerprint spoofing, WebRTC blocking, automatic Cloudflare Turnstile solving (`solve_cloudflare=True`), and humanized cursor movement. Use for bot-protected pages.
3. **`DynamicFetcher`** — Full Playwright Chromium with optional stealth patches. Adds JS rendering, `page_action` callbacks for custom automation (clicking, scrolling, form filling), and `wait_selector`/`network_idle` for dynamic content. Use for JS-heavy SPAs.

All three return identical `Selector`-based response objects. Swapping fetchers is a one-line change — the parsing code stays identical. This is a clean **Strategy Pattern** implementation.

### Session management, cookies, and proxies

Each fetcher type has corresponding session classes that maintain state between requests. `FetcherSession` persists cookies and connection pools for HTTP. `StealthySession` and `DynamicSession` keep the browser context alive, preserving cookies, localStorage, and login state. Browser sessions use **persistent context by default** (since v0.3), and `user_data_dir` enables reusing browser session data across separate program runs.

Proxy support is uniform across all fetchers. Pass a string (`'http://user:pass@host:port'`) or dictionary (`{'server': 'host:port', 'username': 'user', 'password': 'pass'}`) to any request. The `ProxyRotator` class provides thread-safe cyclic rotation across a proxy list, with per-request override capability. The Spider framework integrates proxy rotation at the crawl level.

### The Spider framework (v0.4+)

The Spider framework brings Scrapy-like crawling with a modern async architecture:

```python
from scrapling.spiders import Spider, Request, Response

class QuotesSpider(Spider):
    name = "quotes"
    start_urls = ["https://quotes.toscrape.com/"]
    concurrent_requests = 10

    async def parse(self, response: Response):
        for quote in response.css('.quote'):
            yield {"text": quote.css('.text::text').get(),
                   "author": quote.css('.author::text').get()}
        next_page = response.css('.next a')
        if next_page:
            yield response.follow(next_page[0].attrib['href'])

result = QuotesSpider().start()
result.items.to_json("quotes.json")
```

Key Spider capabilities include **multi-session support** (mixing HTTP and stealth sessions in one spider via session IDs), **checkpoint-based pause/resume** (`crawldir` parameter, Ctrl+C graceful shutdown), **streaming mode** (`async for item in spider.stream()` with real-time stats), and configurable concurrency with per-domain throttling.

---

## The adaptive selector engine: Scrapling's core innovation

The feature that most distinguishes Scrapling from every competing library is its **adaptive element tracking system** — a fingerprint-and-similarity algorithm that automatically relocates page elements after website redesigns. No other Python scraping library offers this capability.

### How the algorithm works

The system operates in two phases. During the **Save Phase**, when you select an element with `auto_save=True`, Scrapling generates a lightweight fingerprint capturing the element's **tag name, text content, all attributes (names and values), sibling tag names, DOM path (tag names from root), parent tag name, parent attributes, and parent text**. This fingerprint is persisted in SQLite, keyed by `(domain, identifier)` — where domain is extracted from the page URL and identifier defaults to the CSS/XPath selector string.

During the **Match Phase** on subsequent runs, if the original selector returns nothing (the page structure changed), passing `adaptive=True` triggers relocation. Scrapling loads the stored fingerprint, **compares every element on the page** against it using fuzzy similarity scoring across all dimensions (tag, text, attributes, siblings, path, parent context), and returns the element(s) with the highest match score. The comparison handles attribute reordering, class name shuffling, and structural shifts gracefully. Benchmarks show this matching completes in **~2.39ms** — 5.2× faster than AutoScraper's equivalent operation.

```python
# First run: save element fingerprint
page = Fetcher.get('https://example.com')
element = page.css('#product-price', auto_save=True)

# Later run after site redesign: relocate automatically
page = Fetcher.get('https://example.com')
element = page.css('#product-price', adaptive=True)  # Finds it even if ID changed
```

### `find_similar()` extends the concept further

The `find_similar()` method, available on any located element, uses a **three-step filtering algorithm**: (1) find all page elements at the same DOM tree depth, (2) filter to those sharing the same tag, parent tag, and grandparent tag, (3) apply fuzzy attribute matching with a configurable `similarity_threshold` (default 0.2). This is conceptually similar to AutoScraper but **5.2× faster** and returns full element objects rather than just text.

---

## Anti-bot evasion: a multi-layer stealth stack

Scrapling implements anti-detection at three distinct levels, each progressively more sophisticated.

**Network level (Fetcher)**: curl_cffi impersonates real browser TLS ClientHello fingerprints — Chrome, Firefox, and Safari variants including specific version numbers (e.g., `'firefox135'`). Combined with matching HTTP headers, HTTP/3 support, and Google referer spoofing, this defeats TLS-based bot fingerprinting that catches raw `requests` or `httpx` traffic. The `impersonate` parameter accepts a list for **random rotation per request**.

**Browser level (DynamicFetcher)**: Patchright, a stealth-patched Playwright fork, removes automation indicators that vanilla Playwright exposes. Additional options include `hide_canvas=True` (injects random noise into canvas operations), `disable_webgl=True`, custom `init_script` injection, and `google_search=True` referer simulation. A `bypasses/` directory contains JavaScript files like `webdriver_fully.js` that are injected to override `navigator.webdriver` and related detection vectors.

**Deep stealth (StealthyFetcher)**: Camoufox is a **modified Firefox binary** with native-level fingerprint spoofing — changes happen in the browser's C++ code, making them invisible to JavaScript-based detection. Combined with `geoip2` for timezone/locale matching to proxy IPs, `block_webrtc=True` to prevent IP leaks, and `humanize` cursor movement simulation, this layer defeats advanced behavioral and fingerprint analysis. The `solve_cloudflare=True` parameter handles **all types of Cloudflare Turnstile and interstitial challenges** automatically, though external reviewers note this works for basic-to-moderate protections rather than Cloudflare's most aggressive configurations.

---

## Full feature inventory across all subsystems

### All fetcher and session types

- `Fetcher` / `AsyncFetcher` — HTTP with TLS impersonation
- `FetcherSession` — Persistent HTTP sessions (sync + async)
- `StealthyFetcher` — Camoufox-based stealth browser
- `StealthySession` / `AsyncStealthySession` — Persistent stealth sessions with tab pooling
- `DynamicFetcher` — Playwright Chromium with JS rendering
- `DynamicSession` / `AsyncDynamicSession` — Persistent browser sessions with tab pooling
- `ProxyRotator` — Thread-safe cyclic proxy rotation

### Seven selection methods on every parsed page

Scrapling provides CSS selectors (`.css()`, `.css_first()` with `::text`/`::attr()` pseudo-elements), XPath (`.xpath()`, `.xpath_first()`), BeautifulSoup-style (`.find()`, `.find_all()` with keyword arguments), text search (`.find_by_text()` with partial/case-insensitive matching), regex search (`.find_by_regex()`), similarity-based search (`.find_similar()` with configurable threshold), and adaptive relocation (`.relocate()` for manual fingerprint matching). Results support chaining: `page.css('.quote').css('.text::text').getall()`.

### Developer tools and integrations

The **interactive shell** (`scrapling shell`) provides an IPython environment with built-in shortcuts (`get`, `post`, `fetch`, `stealthy_fetch`), curl-to-Scrapling conversion (`uncurl`/`curl2fetcher`), and browser preview capability. The **CLI** (`scrapling extract`) enables codeless scraping with output in HTML, Markdown, or text formats. The **MCP Server** (`scrapling mcp`) exposes 6 tools for AI-assisted scraping via the Model Context Protocol, enabling Claude, Cursor, and other AI agents to scrape and extract content with reduced token usage. **Docker images** are auto-built with each release and include all browser binaries.

### Configuration parameters across all fetchers

DynamicFetcher exposes `headless`, `google_search`, `hide_canvas`, `disable_webgl`, `real_chrome`, `stealth`, `wait`, `page_action`, `proxy`, `locale`, `extra_headers`, `useragent`, `cdp_url`, `timeout`, `disable_resources`, `wait_selector`, `wait_selector_state`, `init_script`, `cookies`, `network_idle`, and `custom_config`. StealthyFetcher adds `solve_cloudflare`, `block_webrtc`, `allow_webgl`, `humanize`, `addons` (custom Firefox extensions), `os_randomize`, `disable_ads`, `geoip`, and `additional_args`. The Selector/parser accepts `adaptive`, `adaptive_domain`, `encoding`, `keep_comments`, `keep_cdata`, `huge_tree`, `storage`, and `storage_args`.

### Data extraction utilities

`TextHandler` extends Python's `str` with `.clean()` for whitespace normalization and `.re()`/`.re_first()` for regex extraction. `AttributesHandler` provides an optimized dictionary for element attributes. `.get_all_text(strip=True, ignored_tags=[])` extracts all nested text **40% faster** than previous versions. `.generate_css_selector` and `.generate_xpath_selector` auto-create selectors for any element. Navigation includes `.parent`, `.children`, `.next_sibling`, `.previous_sibling`, `.below_elements`, and `.find_ancestor()`.

---

## Benchmarks, limitations, and the road ahead

### Performance numbers in context

Scrapling's self-reported benchmarks (from `benchmarks.py`, averaged over 100+ runs) show parsing at **2.02ms** for 5,000 nested elements — essentially tied with Scrapy/Parsel (2.04ms) since both build on lxml. The real performance story is against higher-level libraries: **784× faster than BeautifulSoup+lxml** and **41× faster than Selectolax**. The adaptive matching system operates at **2.39ms** versus AutoScraper's 12.45ms. No independent third-party benchmarks exist to verify these claims, though multiple external reviewers have confirmed the parsing speed advantage is real since it inherits directly from lxml's C implementation.

### Known limitations worth noting

External reviewers have identified several honest caveats. The ScrapingBee review (February 2026) encountered **"a couple small bugs"** and noted documentation can drift as the library evolves rapidly. ZenRows explicitly states Scrapling **"is not enough to reliably bypass Cloudflare"** for advanced protections and behavioral analysis — the Turnstile solver handles basic-to-moderate challenges, not Cloudflare's most aggressive enterprise configurations. The library has had **breaking API changes** across major versions (v0.3 renamed core classes, v0.4 changed return types), and full installation requires multiple steps including browser binary downloads. Community presence beyond GitHub stars remains limited — no Stack Overflow questions, minimal Reddit discussion, and low Hacker News engagement despite multiple submissions. The library is still young (first released 2024) and the ecosystem of third-party tutorials and extensions is developing.

## Conclusion

Scrapling represents a genuinely novel entry in the Python scraping ecosystem. Its adaptive selector system solves a real, persistent pain point — broken selectors after website changes — that no other library addresses. The three-tier fetcher architecture (HTTP → stealth → full browser) with a unified parsing API is a clean design that lets developers start simple and escalate only when needed. The tech stack choices are sharp: lxml for speed, curl_cffi for TLS stealth, Camoufox for deep browser-level evasion, and msgspec for type-safe configuration. Where Scrapy excels in mature crawling infrastructure and middleware ecosystem, Scrapling excels in **adaptive resilience and anti-detection** — making it particularly suited for scraping bot-protected sites that change their structure regularly. The v0.4 Spider framework begins closing the crawling gap with Scrapy, though it remains far less battle-tested. For teams tired of maintaining brittle selectors and fighting Cloudflare, Scrapling offers a compelling, if still maturing, alternative.

---

# Scrapling：深度技术架构分析

**Scrapling 是一个 Python 网页抓取框架，将 HTTP 请求、隐身浏览器自动化、自适应元素追踪和 Spider 爬虫统一到一个库中 —— 在解析速度上与 Scrapy/lxml 持平，同时提供竞品库所不具备的"自愈式"选择器能力。** 该库由 Karim Shoair (D4Vinci) 创建，截至 2026 年 2 月已获得约 17,700 个 GitHub Star，最新版本为 v0.4.1。其核心创新在于一套基于指纹和相似度的算法，能够在网站重新设计后自动重定位页面元素，消除了网页抓取中头号维护痛点。底层基于 lxml 做解析、curl_cffi 做 TLS 指纹伪装 HTTP 请求、Playwright/Camoufox 做隐身浏览器自动化，Scrapling 在 Python 抓取生态中占据了独特定位 —— 作为 Scrapy + Selenium/Playwright 技术栈的轻量级但功能完备的替代方案。

---

## 技术架构：从接口到存储的四层设计

Scrapling 遵循**四层架构**：用户接口层 → 请求引擎层 → 解析引擎层 → 支撑系统层。每一层都可独立使用，模块化安装机制（`pip install scrapling` 仅安装解析器、`scrapling[fetchers]` 安装浏览器、`scrapling[all]` 全量安装）体现了这种分离设计。

### 依赖技术栈

技术选型在每个层级都经过刻意优化：

| Layer | Library | Reason |
|-------|---------|--------|
| **HTML Parsing** | `lxml` + `cssselect` | Python 最快的 HTML 解析器。5,000 个嵌套元素的解析基准为 **2.02ms**，而 BeautifulSoup 需要 1,584ms —— **784 倍速度优势** |
| **CSS→XPath Translation** | Custom `HTMLTranslator` (from Parsel, BSD) | 增加了 Scrapy 兼容的 `::text` 和 `::attr()` 伪元素 |
| **HTTP Client** | `curl_cffi` | 唯一支持 **TLS 指纹伪装**和 **HTTP/3 (QUIC)** 的 Python HTTP 库 —— 对网络层反爬至关重要 |
| **Browser Automation** | `playwright` + `patchright` (stealth fork) | Patchright 去除了 Playwright 遗留的自动化标识 |
| **Stealth Browser** | `camoufox` (modified Firefox) | 在浏览器二进制层面原生伪装指纹 —— 比 JavaScript 层补丁更难被检测 |
| **Fingerprint Generation** | `browserforge` | 生成逼真的浏览器指纹数据集用于伪装 |
| **JSON Serialization** | `orjson` | 比 Python 标准 `json` 模块**快 10 倍** |
| **Config Validation** | `msgspec` | 基于 Struct 的类型安全校验，在浏览器启动前即捕获配置错误 |
| **Domain Extraction** | `tld` | 轻量级 TLD 提取（v0.4 中替换了更重的 `tldextract`） |
| **HTML Entities** | `w3lib` | 处理 HTML 实体替换（v0.4 中替换了内部的 `_html_utils.py`） |
| **Async Framework** | `anyio` + optional `uvloop` | 驱动 Spider 框架；`uvloop` 提供 C 级别的事件循环性能 |
| **Adaptive Storage** | SQLite (built-in) | 存储自愈式选择器系统的元素指纹 |
| **IP Geolocation** | `geoip2` | 在 StealthyFetcher 中将时区/语言环境伪装与代理 IP 的地理位置匹配 |

### 异步与同步：全组件双模式

每个 Fetcher 都提供同步和异步两种变体。`Fetcher` 类有专门的 `AsyncFetcher` 对应。浏览器类 Fetcher 在同一个类上暴露 `.fetch()`（同步）和 `.async_fetch()`（异步）。Session 类则做了更显式的拆分：`DynamicSession`（同步）与 `AsyncDynamicSession`（异步）、`StealthySession`（同步）与 `AsyncStealthySession`（异步）。`FetcherSession`（HTTP）通过上下文管理器（`with` 和 `async with`）同时支持两种模式。Spider 框架**完全异步**，基于 `anyio` 构建，可选 `uvloop` 加速。浏览器 Session 实现了**标签页池**（`max_pages=1~50`）来管理并发浏览器标签页，通过 `get_pool_stats()` 报告忙碌/空闲/错误状态。

### 与替代方案的对比

Scrapling 占据了现有工具都无法完全覆盖的独特定位：

| Capability | Scrapling | BeautifulSoup | Scrapy | Selenium | Playwright |
|------------|-----------|---------------|--------|----------|------------|
| Parsing speed (5k elements) | **~2ms** | ~1,584ms | ~2ms (Parsel) | N/A | N/A |
| Adaptive selectors | Built-in | No | No | No | No |
| Anti-bot bypass | Built-in | No | No | Partial | Partial |
| JS rendering | Yes | No | Via Splash | Yes | Yes |
| Spider framework | Yes (v0.4+) | No | Core | No | No |
| TLS fingerprint impersonation | Yes | No | No | N/A | N/A |
| Proxy rotation | Built-in | No | Via middleware | No | No |
| Memory / browser instance | Low (shared ctx) | N/A | N/A | **~500MB+** | ~200MB |

**BeautifulSoup** 仅做解析，无请求能力、无 JS 渲染，且解析速度慢 784 倍。**Scrapy** 在解析速度上与 Scrapling 持平（二者都用 lxml），且拥有更成熟的爬虫/中间件生态，但缺少自适应选择器、反爬对抗和浏览器自动化。**Selenium** 和 **Playwright** 是浏览器自动化工具，不具备内置的解析优化、反检测或爬虫框架。Scrapling 的 API 刻意借鉴了 Scrapy（`.css()`、`.xpath()`、`.get()`、`.getall()`、`::text`、`::attr()`）和 BeautifulSoup（`.find()`、`.find_all()`），以降低迁移成本。

---

## 请求到数据的管道：抓取工作流详解

典型的 Scrapling 工作流遵循清晰的 **Fetcher → Response/Selector → 解析 → 提取** 管道，无论使用哪种 Fetcher，解析 API 完全一致。

### 入口点与三级 Fetcher 模型

Scrapling 提供渐进升级路径。从最快的选项开始，按需升级：

1. **`Fetcher`** —— 基于 curl_cffi 的纯 HTTP。速度最快。支持 TLS 指纹伪装（`impersonate='chrome'`）、隐身请求头和 HTTP/3。用于静态页面和 API。
2. **`StealthyFetcher`** —— 使用 Camoufox（修改版 Firefox）+ Patchright。增加了 Canvas/WebGL 指纹伪装、WebRTC 屏蔽、Cloudflare Turnstile 自动解决（`solve_cloudflare=True`）和人性化鼠标移动模拟。用于有反爬保护的页面。
3. **`DynamicFetcher`** —— 完整的 Playwright Chromium，可选隐身补丁。增加了 JS 渲染、`page_action` 回调函数（用于自定义点击、滚动、表单填写等自动化操作）、`wait_selector`/`network_idle` 等待动态内容。用于 JS 重度 SPA。

三者都返回相同的基于 `Selector` 的响应对象。切换 Fetcher 只需改一行代码 —— 解析代码完全不变。这是一个干净的**策略模式**实现。

### Session 管理、Cookie 与代理

每种 Fetcher 类型都有对应的 Session 类来维护跨请求状态。`FetcherSession` 持久化 Cookie 和连接池用于 HTTP。`StealthySession` 和 `DynamicSession` 保持浏览器上下文活跃，保留 Cookie、localStorage 和登录状态。浏览器 Session 从 v0.3 起**默认使用持久化上下文**，`user_data_dir` 参数支持跨程序复用浏览器会话数据。

代理支持在所有 Fetcher 上统一。传入字符串（`'http://user:pass@host:port'`）或字典（`{'server': 'host:port', 'username': 'user', 'password': 'pass'}`）即可。`ProxyRotator` 类提供线程安全的循环轮换代理列表，支持按请求覆盖。Spider 框架在爬取级别集成了代理轮换。

### Spider 框架（v0.4+）

Spider 框架以现代异步架构带来了类 Scrapy 的爬取能力：

```python
from scrapling.spiders import Spider, Request, Response

class QuotesSpider(Spider):
    name = "quotes"
    start_urls = ["https://quotes.toscrape.com/"]
    concurrent_requests = 10

    async def parse(self, response: Response):
        for quote in response.css('.quote'):
            yield {"text": quote.css('.text::text').get(),
                   "author": quote.css('.author::text').get()}
        next_page = response.css('.next a')
        if next_page:
            yield response.follow(next_page[0].attrib['href'])

result = QuotesSpider().start()
result.items.to_json("quotes.json")
```

Spider 的关键能力包括：**多 Session 支持**（通过 Session ID 在同一个 Spider 中混用 HTTP 和隐身 Session）、**基于 Checkpoint 的暂停/恢复**（`crawldir` 参数、Ctrl+C 优雅关闭）、**流式模式**（`async for item in spider.stream()` 带实时统计）、以及可配置的并发控制和域名级别限速。

---

## 自适应选择器引擎：Scrapling 的核心创新

最能将 Scrapling 与所有竞品库区分开来的功能是其**自适应元素追踪系统** —— 一套基于指纹和相似度的算法，能在网站重新设计后自动重定位页面元素。目前没有任何其他 Python 抓取库提供此能力。

### 算法工作原理

系统分两个阶段运行。**保存阶段**：当使用 `auto_save=True` 选取元素时，Scrapling 生成一个轻量级指纹，捕获元素的**标签名、文本内容、所有属性（名称和值）、兄弟节点标签名、DOM 路径（从根到元素的标签名序列）、父节点标签名、父节点属性和父节点文本**。该指纹持久化到 SQLite，以 `(domain, identifier)` 为键 —— 其中 domain 从页面 URL 提取，identifier 默认为 CSS/XPath 选择器字符串。

**匹配阶段**：在后续运行中，如果原始选择器返回空结果（页面结构已变），传入 `adaptive=True` 即触发重定位。Scrapling 加载存储的指纹，**将页面上每个元素**与之进行模糊相似度评分比较（覆盖标签、文本、属性、兄弟节点、路径、父节点上下文等所有维度），返回匹配得分最高的元素。比较过程能优雅处理属性重排、class 名变化和结构性迁移。基准测试显示匹配在 **~2.39ms** 内完成 —— 比 AutoScraper 的等效操作**快 5.2 倍**。

```python
# 首次运行：保存元素指纹
page = Fetcher.get('https://example.com')
element = page.css('#product-price', auto_save=True)

# 网站改版后再次运行：自动重定位
page = Fetcher.get('https://example.com')
element = page.css('#product-price', adaptive=True)  # 即使 ID 变了也能找到
```

### `find_similar()` 进一步扩展概念

`find_similar()` 方法可在任何已定位元素上调用，使用**三步过滤算法**：(1) 找到页面上同一 DOM 树深度的所有元素；(2) 过滤出具有相同标签、父标签和祖父标签的元素；(3) 对属性进行模糊匹配，使用可配置的 `similarity_threshold`（默认 0.2）。概念上类似 AutoScraper，但**快 5.2 倍**且返回完整元素对象而非仅文本。

---

## 反爬对抗：多层隐身技术栈

Scrapling 在三个不同层级实施反检测，每一级比前一级更为精密。

**网络层（Fetcher）**：curl_cffi 伪装真实浏览器的 TLS ClientHello 指纹 —— 包括 Chrome、Firefox 和 Safari 的多个变体及具体版本号（如 `'firefox135'`）。结合匹配的 HTTP 请求头、HTTP/3 支持和 Google Referer 伪装，可以击败基于 TLS 的机器人指纹识别（这种检测能轻松识别原生 `requests` 或 `httpx` 流量）。`impersonate` 参数接受列表以实现**每次请求随机轮换**。

**浏览器层（DynamicFetcher）**：Patchright 是 Playwright 的隐身补丁分支，去除了原版 Playwright 暴露的自动化标识。额外选项包括 `hide_canvas=True`（向 Canvas 操作注入随机噪声）、`disable_webgl=True`、自定义 `init_script` 注入和 `google_search=True` Referer 模拟。`bypasses/` 目录包含如 `webdriver_fully.js` 等 JavaScript 文件，注入后覆盖 `navigator.webdriver` 及相关检测向量。

**深度隐身层（StealthyFetcher）**：Camoufox 是一个**修改版 Firefox 二进制文件**，在浏览器 C++ 代码中实现原生级指纹伪装，使其对 JavaScript 检测完全不可见。结合 `geoip2` 实现时区/语言环境与代理 IP 的地理位置匹配、`block_webrtc=True` 防止 IP 泄露、`humanize` 光标移动模拟，这一层能对抗高级行为分析和指纹分析。`solve_cloudflare=True` 参数自动处理 **Cloudflare Turnstile 和中间页挑战**的所有类型，但外部评测指出这对基础到中等保护有效，而非 Cloudflare 最激进的企业级配置。

---

## 全量特性清单

### 所有 Fetcher 和 Session 类型

- `Fetcher` / `AsyncFetcher` —— 带 TLS 伪装的 HTTP 请求
- `FetcherSession` —— 持久化 HTTP Session（同步 + 异步）
- `StealthyFetcher` —— 基于 Camoufox 的隐身浏览器
- `StealthySession` / `AsyncStealthySession` —— 带标签页池的持久化隐身 Session
- `DynamicFetcher` —— 带 JS 渲染的 Playwright Chromium
- `DynamicSession` / `AsyncDynamicSession` —— 带标签页池的持久化浏览器 Session
- `ProxyRotator` —— 线程安全的循环代理轮换

### 每个解析页面上的七种选择方法

Scrapling 提供 CSS 选择器（`.css()`、`.css_first()` 带 `::text`/`::attr()` 伪元素）、XPath（`.xpath()`、`.xpath_first()`）、BeautifulSoup 风格（`.find()`、`.find_all()` 带关键字参数）、文本搜索（`.find_by_text()` 支持部分匹配/不区分大小写）、正则搜索（`.find_by_regex()`）、相似度搜索（`.find_similar()` 带可配置阈值）和自适应重定位（`.relocate()` 手动指纹匹配）。结果支持链式调用：`page.css('.quote').css('.text::text').getall()`。

### 开发者工具与集成

**交互式 Shell**（`scrapling shell`）提供 IPython 环境，内置快捷方法（`get`、`post`、`fetch`、`stealthy_fetch`）、curl 到 Scrapling 的转换（`uncurl`/`curl2fetcher`）和浏览器预览功能。**CLI**（`scrapling extract`）实现免代码抓取，支持 HTML、Markdown 或纯文本输出。**MCP Server**（`scrapling mcp`）通过模型上下文协议暴露 6 个工具，用于 AI 辅助抓取，支持 Claude、Cursor 等 AI Agent 以更少的 Token 消耗进行抓取和内容提取。**Docker 镜像**随每次发布自动构建，包含所有浏览器二进制文件。

### 所有 Fetcher 的配置参数

DynamicFetcher 暴露 `headless`、`google_search`、`hide_canvas`、`disable_webgl`、`real_chrome`、`stealth`、`wait`、`page_action`、`proxy`、`locale`、`extra_headers`、`useragent`、`cdp_url`、`timeout`、`disable_resources`、`wait_selector`、`wait_selector_state`、`init_script`、`cookies`、`network_idle` 和 `custom_config`。StealthyFetcher 额外增加 `solve_cloudflare`、`block_webrtc`、`allow_webgl`、`humanize`、`addons`（自定义 Firefox 扩展）、`os_randomize`、`disable_ads`、`geoip` 和 `additional_args`。Selector/解析器接受 `adaptive`、`adaptive_domain`、`encoding`、`keep_comments`、`keep_cdata`、`huge_tree`、`storage` 和 `storage_args`。

### 数据提取工具

`TextHandler` 扩展 Python 的 `str`，增加 `.clean()` 空白规范化和 `.re()`/`.re_first()` 正则提取。`AttributesHandler` 提供优化的元素属性字典。`.get_all_text(strip=True, ignored_tags=[])` 提取所有嵌套文本，比旧版**快 40%**。`.generate_css_selector` 和 `.generate_xpath_selector` 为任意元素自动生成选择器。导航包括 `.parent`、`.children`、`.next_sibling`、`.previous_sibling`、`.below_elements` 和 `.find_ancestor()`。

---

## 基准测试、已知局限与未来方向

### 性能数据

Scrapling 的自报基准（来自 `benchmarks.py`，100+ 次运行平均）显示 5,000 个嵌套元素的解析耗时 **2.02ms** —— 与 Scrapy/Parsel (2.04ms) 基本持平，因为二者都基于 lxml。真正的性能故事在于与高层库的对比：比 **BeautifulSoup+lxml 快 784 倍**，比 **Selectolax 快 41 倍**。自适应匹配系统以 **2.39ms** 运行，而 AutoScraper 需要 12.45ms。目前没有独立第三方基准来验证这些数据，但多个外部评测者已确认解析速度优势是真实的，因为它直接继承自 lxml 的 C 实现。

### 值得注意的已知局限

外部评测者指出了几个客观的不足。ScrapingBee 评测（2026 年 2 月）遇到了一些小 Bug，并指出文档在库快速迭代时可能滞后。ZenRows 明确表示 Scrapling 对于高级防护和行为分析**不足以可靠绕过 Cloudflare** —— Turnstile 求解器处理基础到中等挑战，而非 Cloudflare 最激进的企业级配置。该库在主要版本之间存在**破坏性 API 变更**（v0.3 重命名核心类，v0.4 改变返回类型），完整安装需要多个步骤包括浏览器二进制下载。GitHub Star 之外的社区存在感仍然有限 —— Stack Overflow 上没有相关问题、Reddit 讨论很少、Hacker News 参与度低（尽管多次提交）。该库仍然年轻（2024 年首次发布），第三方教程和扩展生态正在发展中。

## 总结

Scrapling 代表了 Python 抓取生态中一个真正新颖的存在。其自适应选择器系统解决了一个真实且持续的痛点 —— 网站变更后选择器失效 —— 而目前没有任何其他库对此给出方案。三级 Fetcher 架构（HTTP → 隐身 → 完整浏览器）配合统一解析 API 是一个干净的设计，让开发者从简单开始、按需升级。技术选型精准：lxml 追求速度、curl_cffi 实现 TLS 隐身、Camoufox 做深度浏览器级伪装、msgspec 做类型安全配置。Scrapy 在成熟的爬虫基础设施和中间件生态方面更优，而 Scrapling 在**自适应韧性和反检测**方面更强 —— 使其特别适合抓取有反爬保护且经常变更结构的网站。v0.4 的 Spider 框架开始缩小与 Scrapy 在爬取方面的差距，但距经受实战检验还有相当距离。对于厌倦了维护脆弱选择器和与 Cloudflare 对抗的团队而言，Scrapling 提供了一个引人注目的、虽然仍在成熟中的替代方案。
