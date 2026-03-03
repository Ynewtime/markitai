# Scrapling 引入探索报告

> 日期：2026-03-02  
> 版本：Scrapling 0.4.1 / markitai 0.5.2  
> 状态：探索评估

## 1. 背景与动机

markitai 当前的 URL→Markdown 管线由三层构成：

| 层次 | 当前实现 | 职责 |
|------|---------|------|
| **获取层** | `markitdown` (HTTP) / Playwright / CF BR / Jina | 拿到原始 HTML 或 Markdown |
| **转换层** | `markitdown`（内置 HTML→MD）/ CF BR `/markdown` | HTML → Markdown |
| **增强层** | LiteLLM + Instructor | 格式清理、元数据生成、图像分析 |

该架构在静态页面上工作良好，但在以下场景存在可改进空间：

1. **SPA / 反爬站点**：依赖 Playwright 冷启动浏览器，无 TLS fingerprint 伪装、无反机器人绕过，抓取 Cloudflare Turnstile 等保护的页面需要额外处理。
2. **获取层多策略串联**：`fetch.py` 的 `_fetch_with_fallback` 链式尝试 `static → playwright → cloudflare → jina`，但每个后端的 HTTP 能力各异、接口不统一，维护成本递增。
3. **浏览器 Session 复用**：当前 `PlaywrightRenderer` 虽有全局实例，但不支持跨请求 cookie/状态保持，批量抓取同域名页面时每次都是全新上下文。

Scrapling 作为一个自适应 Web Scraping 框架，声称在获取层和解析层均有优势。本报告基于实际测试评估其是否适合引入 markitai。

## 2. Scrapling 架构概览

```
                           Scrapling
┌──────────────────────────────────────────────────────┐
│                                                      │
│   Fetcher (HTTP)          ← curl-cffi, TLS 伪装     │
│   StealthyFetcher         ← patchright, fingerprint │
│   DynamicFetcher          ← playwright, 完整自动化  │
│         ↓                                            │
│   Selector / Adaptor      ← lxml, CSS/XPath/文本搜索│
│         ↓                                            │
│   Convertor               ← markdownify → Markdown  │
│                                                      │
│   Spider                  ← 并发爬取框架            │
│   Session                 ← 持久化 cookie/状态      │
│   ProxyRotator            ← 代理轮换                │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 三种 Fetcher

| Fetcher | 底层 | 特性 | 典型耗时 |
|---------|------|------|----------|
| `Fetcher` | curl-cffi | TLS fingerprint 伪装、HTTP/3、stealth headers | 0.1–0.6s |
| `StealthyFetcher` | patchright | 指纹伪装、Canvas 噪音、WebRTC 阻断、自动绕过 Cloudflare Turnstile | 1.5–8s |
| `DynamicFetcher` | playwright | 完整浏览器自动化、network_idle 等待 | 1.5–47s |

### 解析层

- 基于 lxml，性能与 Parsel/Scrapy 齐平（5000 嵌套元素 2ms）
- 支持 CSS / XPath / BeautifulSoup 风格 / `find_by_text` / `find_similar` / `::text` 伪元素
- 自适应元素追踪（网站结构变化后自动重新定位）

### Markdown 转换

- 内部使用 `markdownify` 库
- `Convertor._extract_content(page, "markdown", main_content_only=True)` 管线：`<body>` → 移除 `script/style/noscript/svg` → `markdownify(html)`
- CLI `scrapling extract get URL output.md` 支持 `.md/.html/.txt` 三种输出

## 3. 实测数据

### 3.1 测试集

| URL | 类型 | 难度 |
|-----|------|------|
| `ynewtime.com/jekyll-ynewtime/人是什么单位` | 静态博客 (Jekyll)，中文路径 + 301 重定向 | 低 |
| `stephango.com/concise` | 静态博客，标准 HTML | 低 |
| `x.com/Blankwonder/status/...` | SPA (React)，需 JS 渲染 + 反爬 | 高 |

### 3.2 获取层对比

| URL | Fetcher | OK | 耗时 | 正文字数 | 链接数 |
|-----|---------|:--:|-----:|-------:|------:|
| ynewtime | Fetcher | ✅ | 1.05s | 3166 | 15 |
| ynewtime | StealthyFetcher | ✅ | 4.55s | 3174 | 15 |
| ynewtime | DynamicFetcher | ✅ | 2.49s | 3174 | 15 |
| stephango | Fetcher | ✅ | 0.61s | 4026 | 23 |
| stephango | StealthyFetcher | ✅ | 1.86s | 4026 | 23 |
| stephango | DynamicFetcher | ✅ | 1.38s | 4026 | 23 |
| **x.com** | **Fetcher** | **❌** | **0.49s** | **214240 (噪音)** | **6** |
| **x.com** | **StealthyFetcher** | **✅** | **7.57s** | **215230** | **23** |
| **x.com** | **DynamicFetcher** | **✅** | **47.06s** | **215230** | **23** |

关键发现：

- **静态站点**：三种 Fetcher 内容完全一致，`Fetcher` 速度最快（亚秒级）
- **X.com (SPA)**：
  - `Fetcher`：HTTP 200 但内容全是 JS 空壳（`"JavaScript is not available"`），`<title>` 为空
  - `StealthyFetcher`：完整渲染推文 DOM，`[data-testid="tweetText"]` 找到 1 个，推文全文通过 `::text` 正确提取
  - `DynamicFetcher`：内容同上，但 `network_idle` 等待 X.com 大量异步请求导致 **47s** 才返回

### 3.3 Markdown 输出质量

使用 Scrapling CLI `scrapling extract` 和 Python API `Convertor._extract_content` 两种方式测试。

#### 静态站点

| 方式 | 文件大小 | 质量评价 |
|------|---------|---------|
| CLI `get` 全页 `.md` | 2354 B (stephango) / 6449 B (ynewtime) | 含导航、footer、`<script>` 内 JS 代码残留 |
| CLI `get` + `--css-selector article` | 834 B / 6259 B | **纯净正文**，标题/图片/引用/粗体均正确转换 |
| Python API `main_content_only=True` | 2291 chars / 6422 chars | body-only 去除了 `script/style`，但导航/footer 仍保留 |

stephango `--css-selector article` 输出示例（完整）：

```markdown
If you want to progress faster, write concise explanations. Explain ideas in
simple terms, strongly and clearly, so that they can be rebutted, remixed,
reworked — or built upon.

Concise explanations spread faster because they are easier to read and
understand. The sooner your idea is understood, the sooner others can build
on it.
...
```

ynewtime `--css-selector article` 输出示例（节选）：

```markdown
[人是什么单位？](/人是什么单位)
==================================================================

2018-04-10 | 熊培云 | [转载](/tags#转载 "转载")

---

![](https://yenwtime-1255970624.cos.ap-guangzhou.myqcloud.com/JPG/unit.jpg)

我曾说，一人即一国，每个人都有属于自己的疆土……

> **以生命与时间的名义，每个人作为其所生息的时代中的一员，不应该停留于
> 寻找地理意义上的与生俱来的归属，而应忠诚于自己一生的光阴，不断创造
> 并享有属于自己的幸福时光。**
```

评价：`markdownify` 的转换质量在元素级别是可靠的——标题、图片、引用、加粗、链接均正确保留。**核心瓶颈不在转换器本身，而在正文区域的识别**。

#### X.com (SPA)

| 方式 | 内容 | 质量 |
|------|------|------|
| `StealthyFetcher` + `article` 选择器 | 头像、用户名、推文全文、时间、互动数据 | ⭐ 最佳 |
| `StealthyFetcher` + `[data-testid="tweetText"]` 选择器 | 纯推文文本 | 零噪音，289 bytes |
| `StealthyFetcher` 全页 `main_content_only=True` | 推文 + cookie 弹窗 + 登录提示 + footer | 大量 UI 噪音 |
| `Fetcher` (HTTP) | "JavaScript is not available" | ❌ 完全不可用 |

`StealthyFetcher` + `article` 选择器输出：

```markdown
[![](https://pbs.twimg.com/profile_images/.../vESyD2hO_x96.jpg)](/Blankwonder)

[Yachen Liu](/Blankwonder)

[@Blankwonder](/Blankwonder)

软件工程师作为离 AI 最近的一波人，最先惊呼：工作没了。但我认为，在 AI
全面改造各行各业的过程中，依然有很多地方需要软件工程师介入……

Translate post

[3:55 PM · Jan 16, 2026](/Blankwonder/status/2012071164638535850)

56 29 327 143

Read 56 replies
```

#### 三种格式对比（stephango `article`）

| 格式 | 大小 | 特点 |
|------|------|------|
| `.md` | 834 B | Markdown 纯文本段落 |
| `.html` | 904 B | 原始 `<article>` HTML |
| `.txt` | 829 B | 纯文本，无格式 |

## 4. 与 markitai 现有架构的对照分析

### 4.1 获取层

| 维度 | markitai 当前 | Scrapling | 对比 |
|------|-------------|-----------|------|
| HTTP 请求 | `markitdown` (requests) | `Fetcher` (curl-cffi) | Scrapling 有 TLS fingerprint 伪装，markitdown 原生支持 CF Markdown for Agents 内容协商 |
| 浏览器渲染 | Playwright Python (直接) | `StealthyFetcher` (patchright) / `DynamicFetcher` (playwright) | Scrapling 的 StealthyFetcher 多了 fingerprint 伪装 + Cloudflare Turnstile 自动绕过 |
| 云端获取 | CF Browser Rendering / Jina Reader | 无内置云端 | markitai 的云端能力更丰富 |
| 代理检测 | 自动检测系统代理 + 常用端口探测 | 需手动配置 `ProxyRotator` | markitai 的自动代理检测更友好 |
| Session 管理 | 无持久化 Session | `FetcherSession` / `StealthySession` / `DynamicSession` | Scrapling 优势明显 |
| SPA 学习 | `SPADomainCache`（JSON 持久化） | 无 | markitai 独有 |
| 条件请求 | `ETag` / `If-Modified-Since` 支持 | 无 | markitai 独有 |

### 4.2 转换层

| 维度 | markitai 当前 | Scrapling | 对比 |
|------|-------------|-----------|------|
| HTML→MD | `markitdown` 内置（基于 BeautifulSoup） | `markdownify` 库 | 两者质量相近；markitdown 额外支持 CF Markdown for Agents 协商获取服务端 Markdown |
| 正文提取 | `_html_to_text`（BeautifulSoup，识别 main/article/body） | `Convertor._strip_noise_tags`（仅去 script/style） | markitai 的正文识别逻辑更强 |
| 噪音过滤 | `DOM_NOISE_SELECTORS`（24 个选择器）+ `DOM_NOISE_ATTRIBUTES` | 仅 4 种标签（script/style/noscript/svg） | markitai 远更激进 |
| JS 检测 | `detect_js_required` 多策略检测 | 无 | markitai 独有 |
| 文件格式 | PDF / Office / 图片 / CSV 等 | 仅 HTML | markitai 覆盖广得多 |

### 4.3 解析层

| 维度 | markitai 当前 | Scrapling | 对比 |
|------|-------------|-----------|------|
| DOM 解析 | BeautifulSoup（`_html_to_text`）/ Playwright `page.content()` | lxml + Adaptor（CSS/XPath/文本搜索/`::text`） | **Scrapling 显著更强**：选择器丰富度、性能、链式操作 |
| 元素定位 | 硬编码选择器列表 (`DOM_NOISE_SELECTORS`) | 自适应追踪 + `find_similar` + `find_by_text` | Scrapling 更灵活 |

## 5. 引入评估

### 5.1 Scrapling 能补强的领域

| 序号 | 能力 | 当前痛点 | Scrapling 方案 | 影响面 |
|------|------|---------|---------------|--------|
| 1 | **反爬绕过** | Playwright 原生无 fingerprint 伪装，Cloudflare Turnstile 等无法通过 | `StealthyFetcher` 开箱即用绕过 | `fetch.py` 获取层 |
| 2 | **HTTP 隐秘请求** | `markitdown` 的 requests 无 TLS 伪装 | `Fetcher` 的 curl-cffi + `stealthy_headers` | `fetch_with_static` |
| 3 | **Session 持久化** | 无跨请求 cookie/状态保持 | `StealthySession` / `FetcherSession` 持久化 Session | 批量抓取同域名 |
| 4 | **DOM 解析能力** | `_html_to_text` 基于 BeautifulSoup 粗粒度提取 | Scrapling Selector 精细 CSS/XPath + `::text` | 内容质量提升 |

### 5.2 引入的代价和风险

| 序号 | 风险 | 严重程度 | 说明 |
|------|------|:--------:|------|
| 1 | **依赖膨胀** | 🟡 中 | Scrapling 带入 `curl-cffi`(8MB)、`patchright`(44MB)、`browserforge`、`tld` 等。markitai 当前已有 Playwright 依赖，patchright 是其 fork，二者共存需验证兼容性 |
| 2 | **Playwright vs Patchright 冲突** | 🔴 高 | markitai 固定 `playwright>=1.50.0`，Scrapling 固定 `playwright==1.56.0` + `patchright==1.56.0`；安装时 Scrapling 已将 playwright 降级到 1.56.0，可能影响现有 `fetch_playwright.py` |
| 3 | **Python 版本** | 🟢 低 | Scrapling 要求 ≥3.10，markitai ≥3.11，兼容 |
| 4 | **markitdown 整合** | 🟡 中 | markitai 对 `markitdown` 有深度整合（`_get_markitdown` 单例、CF Markdown for Agents Accept header 注入、条件请求逻辑），替换获取层需保留这些能力 |
| 5 | **Markdown 质量** | 🟡 中 | Scrapling 的 `markdownify` 不如 markitai 已有的 `DOM_NOISE_SELECTORS` + `_html_to_text` 管线，全页 Markdown 噪音明显；但 Scrapling 的精确选择器可以从源头缩小提取范围 |
| 6 | **维护负担** | 🟢 低 | Scrapling 社区活跃（92% 覆盖率，每日使用者数百），但仍是个人项目而非组织维护 |

### 5.3 不需要引入的领域

以下 Scrapling 能力与 markitai 已有实现重叠或不适用：

- **Spider 爬虫框架**：markitai 是转换工具，不做站点级爬取
- **自适应元素追踪 / `auto_save`**：适用于长期监控场景，markitai 是一次性转换
- **MCP Server**：markitai 有独立的 LLM 增强管线
- **代理轮换**：markitai 的自动代理检测已覆盖
- **CLI extract**：markitai 已有完整 CLI

## 6. 引入建议

### 方案 A：不引入（维持现状 + 定向改进）⭐ 推荐

**理由**：

Scrapling 的核心优势（反爬绕过、TLS 伪装、Session 持久化）对 markitai 的主要用例（文档/URL 一次性转换 Markdown）的收益有限。markitai 的 `fetch.py` 已建立了成熟的四级降级链（static → playwright → cloudflare → jina），且有条件请求、SPA 学习等 Scrapling 没有的能力。

可定向吸收 Scrapling 的优秀设计思路，不引入依赖：

1. **TLS fingerprint 伪装**：在 `fetch_with_static` 中从 `markitdown` 的 requests 切换到 `curl-cffi`（独立引入这一个轻量依赖），伪装 Chrome TLS 指纹
2. **DOM 选择器增强**：在 `_html_to_text` 中引入 CSS 选择器精确提取（用已有的 lxml 或引入 `cssselect`），替代当前的 BeautifulSoup 硬编码标签列表
3. **Stealth 能力**：在 Playwright 渲染路径中引入 `playwright-stealth` 插件（轻量，不需要整个 Scrapling）

### 方案 B：选择性引入（仅获取层）

如果后续 markitai 需要大量处理反爬保护的站点：

```python
# fetch.py 新增 scrapling 策略
class FetchStrategy(Enum):
    AUTO = "auto"
    STATIC = "static"
    PLAYWRIGHT = "playwright"
    CLOUDFLARE = "cloudflare"
    JINA = "jina"
    SCRAPLING = "scrapling"       # 新增
    SCRAPLING_STEALTH = "scrapling_stealth"  # 新增
```

整合点：
- `fetch_with_scrapling`：替代 `fetch_with_static`，使用 `Fetcher.get` + `stealthy_headers=True`
- `fetch_with_scrapling_stealth`：替代 `fetch_with_playwright` 中的隐秘场景，使用 `StealthyFetcher.fetch`
- 将 Scrapling 的 `Convertor._extract_content(page, "markdown", css_selector=...)` 用于 HTML→MD 转换
- 保留 markitai 的 `DOM_NOISE_SELECTORS` 作为后处理清洗

依赖声明：

```toml
[project.optional-dependencies]
scrapling = ["scrapling[fetchers]>=0.4.1"]
```

**风险**：需解决 Playwright/Patchright 版本冲突。

### 方案 C：深度整合（替换获取层 + 解析层）

不推荐。Scrapling 的解析层（Selector/Adaptor）与 markitai 的 `markitdown` 生态重叠，深度整合改造量大且引入过多外部依赖，破坏现有的 CF Markdown for Agents、条件请求等已验证的能力。

## 7. 后续行动项

无论是否引入 Scrapling，以下改进基于本次探索可立即推进：

| 优先级 | 行动 | 来源 | 预估工作量 |
|:------:|------|------|:---------:|
| P1 | `fetch_with_static` 引入 `curl-cffi` 替代 `requests`，增加 TLS fingerprint 伪装 | Scrapling `Fetcher` 思路 | 1d |
| P1 | `_html_to_text` 用 CSS 选择器替代硬编码标签列表（尝试 `article` → `main` → `body` 降级） | Scrapling `Convertor._extract_content` 思路 | 0.5d |
| P2 | Playwright 路径加入 `playwright-stealth`（`page.add_init_script`） | Scrapling `StealthyFetcher` 思路 | 0.5d |
| P2 | 评估 `markdownify` 替换/补充当前 `_html_to_text` 中的手写 Markdown 拼接逻辑 | Scrapling Convertor | 1d |
| P3 | 增加 `FetcherSession` 式的 cookie/状态持久化能力，优化批量同域名抓取 | Scrapling Session API | 2d |

## 8. 附录：测试环境

- OS: Linux (Ubuntu 24.04 like)
- Python: 3.13.7
- Scrapling: 0.4.1（via `uv pip install "scrapling[fetchers]"`）
- markitai: 0.5.2
- Playwright: 1.56.0（被 Scrapling 降级，原 1.58.0）
- Patchright: 1.56.0
- 网络：中国大陆，通过本地代理访问 X.com

### 测试脚本（已清理）

获取层 + Markdown 输出的完整验证通过 `scripts/test_scrapling.py` 和 `scripts/test_scrapling_md.py` 执行，覆盖：

- 用例 1: CLI `scrapling extract get` → `.md`（HTTP 模式，3 URL）
- 用例 2: CLI `scrapling extract stealthy-fetch` → `.md`（隐秘浏览器，3 URL）
- 用例 3: CLI `scrapling extract get` + `--css-selector article`（精确提取，2 URL）
- 用例 4: Python API `Convertor._extract_content(main_content_only=True)`（去噪，2 URL）
- 用例 5: Python API `StealthyFetcher` + CSS 选择器（X.com 推文，3 种选择器粒度）
- 用例 6: 同一页面 `.md` / `.html` / `.txt` 三格式对比

全部 **16 项子用例通过**，详见正文第 3 节。
