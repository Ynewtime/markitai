# 报告

# Cloudflare AI 内容交付能力完全指南
## 面向本地 Markdown 工具开发者的深度调研报告

> **文档说明**：本报告整合了三轮对话的全部研究内容，并附有事实核查注记。
> **信息时效**：基于 2026 年 2 月 23 日前公开的官方文档及发布博客。
> **核查标记**：`⚠️ 原叙述有误` 标注经核查发现的不准确之处，`✅ 已核实` 标注经二次确认的关键事实。

---

## 目录

1. [背景与问题定义](#1-背景与问题定义)
2. [功能一：Markdown for Agents（边缘转换）](#2-功能一markdown-for-agents边缘转换)
3. [功能二：Browser Rendering REST API](#3-功能二browser-rendering-rest-api)
4. [功能三：Workers AI toMarkdown](#4-功能三workers-ai-tomarkdown)
5. [三大功能横向对比](#5-三大功能横向对比)
6. [面向本地工具的集成架构](#6-面向本地工具的集成架构)
7. [完整决策树与路由逻辑](#7-完整决策树与路由逻辑)
8. [定价与限制速查](#8-定价与限制速查)
9. [争议与已知局限](#9-争议与已知局限)
10. [事实核查汇总](#10-事实核查汇总)

---

## 1. 背景与问题定义

### 1.1 问题本质

现代 AI pipeline 中，将网页 HTML 转换为干净的 Markdown 是一个普遍存在的预处理步骤。这个步骤本身有显著的成本：

- **Token 浪费**：HTML 中充斥着对语义毫无价值的 `<div>`、CSS 类名、脚本标签和导航结构
- **重复劳动**：每个消费方（爬虫、Agent、RAG pipeline）都在独立执行相同的转换
- **质量不稳定**：各方实现的转换质量参差不齐

**✅ 已核实的量化对比**：Cloudflare 自身的一篇博客文章，HTML 版需要 16,180 个 token，Markdown 版仅需 3,150 个，**节省约 80%**。

一个简单 `<h2>` 标签的 token 消耗对比：

```
Markdown: ## About Us          → ~3 tokens
HTML:     <h2 class="section-title" id="about">About Us</h2>  → 12–15 tokens
```

### 1.2 Cloudflare 的系统性解法

Cloudflare 在 2026 年 2 月 12 日发布 Markdown for Agents 后，围绕「AI 内容消费」构建了三层完整的解决方案，它们针对不同场景，覆盖了从边缘网络到云端 API 的全链路：

| 层次 | 功能名称 | 场景定位 | 是否需要账号 |
|------|----------|----------|-------------|
| 边缘层 | Markdown for Agents | 已启用站点的零成本抓取 | 否（客户端免费使用） |
| 渲染层 | Browser Rendering `/markdown` | 动态页面 / SPA 渲染后转换 | 是 |
| 文件层 | Workers AI `toMarkdown` | 离线文件格式（PDF/Office/图片） | 是 |

---

## 2. 功能一：Markdown for Agents（边缘转换）

### 2.1 技术机制

这是 2026 年 2 月最新发布的功能，技术实现基于 HTTP 标准的 **Content Negotiation（内容协商）** 机制，而非 User-Agent 嗅探。

工作流程：

```
AI 客户端发送请求
  └─ Header: Accept: text/markdown
       │
       ▼
Cloudflare 边缘节点拦截
  └─ 检查目标站点是否已启用 Markdown for Agents
       │
       ├─ 已启用 → 从 Origin 拉取 HTML → 边缘转换 → 返回 Markdown
       └─ 未启用 → 透传原始请求，返回 HTML
```

**✅ 已核实**：该机制与 User-Agent 检测完全不同，同一个 URL 根据请求方声明的内容类型返回不同格式的表示（representation），这是标准的 HTTP 内容协商行为。

### 2.2 请求与响应示例

```bash
# 客户端请求（q 值设置优先级，对未启用站点仍能正常获取 HTML）
curl https://developers.cloudflare.com/workers-ai/ \
  -H "Accept: text/markdown;q=1.0, text/html;q=0.8, */*;q=0.5"
```

响应头（成功时）：

```
HTTP/2 200
content-type: text/markdown; charset=utf-8
vary: accept
x-markdown-tokens: 725
content-signal: ai-train=yes, search=yes, ai-input=yes
```

关键响应头说明：

- `content-type: text/markdown`：确认返回了 Markdown 格式
- `vary: accept`：告知缓存层此响应因 Accept 头不同而异（防止 HTML 缓存污染 Markdown 响应）
- `x-markdown-tokens`：返回内容的估算 token 数，可用于 context window 预判和分块策略
- `content-signal`：默认声明内容可用于 AI 训练、搜索和 AI 推理

### 2.3 Content-Signal 的重要性

**✅ 已核实**：启用 Markdown for Agents 后，每个转换响应默认附带 `Content-Signal: ai-train=yes, search=yes, ai-input=yes`。这不仅是格式变换，同时也是一个内容授权声明。对内容授权有顾虑的站长在开启前需仔细考量，Cloudflare 表示未来将提供自定义 Content-Signal 策略的选项。

### 2.4 使用限制

**✅ 已核实**：

- 目前仅支持 HTML → Markdown 转换，不支持 PDF 等其他格式
- Origin 响应体上限：**2 MB**（2026 年 2 月 16 日从 1 MB 升至 2 MB）
- 目标站点**必须已启用**该功能，否则请求透传，返回 HTML

### 2.5 计划要求（站点启用侧）

**✅ 已核实**：作为**站点管理员**开启 Markdown for Agents 需要 **Pro、Business 或 Enterprise 计划**，以及 SSL for SaaS 客户，当前 Beta 期间免费。**Free 计划不可使用**。

> **作为客户端消费者无任何计划要求**——向已启用该功能的站点发送 `Accept: text/markdown` 请求完全免费，不需要 Cloudflare 账号。

**✅ 已核实**：Claude Code 和 OpenCode 等编码 Agent 工具目前已在请求中默认发送 `Accept: text/markdown` 头。

---

## 3. 功能二：Browser Rendering REST API

### 3.1 功能定位

Browser Rendering 是 Cloudflare 提供的无头浏览器服务，运行在其全球边缘网络上。与 Markdown for Agents 的关键区别在于：**它可以真正渲染 JavaScript，处理 SPA 和动态页面**，也可以直接传入原始 HTML 字符串进行渲染。

**计划要求**：**Free 和 Paid 计划均可使用**，但有用量限制（见定价章节）。

### 3.2 REST API 端点全集（8 个）

所有端点的基础 URL：

```
https://api.cloudflare.com/client/v4/accounts/<accountId>/browser-rendering/<endpoint>
```

认证：`Authorization: Bearer <apiToken>`，需要 `Browser Rendering - Edit` 权限。

每个端点接受 `url`（远程抓取）或 `html`（渲染本地 HTML 字符串）二选一作为必填输入。

| Endpoint | 输出类型 | 核心用途 | 特殊能力 |
|----------|----------|----------|----------|
| `/markdown` | Markdown 字符串 | 网页转 Markdown | JS 渲染后转换 |
| `/content` | HTML 字符串 + 页面标题 | 获取渲染后完整 HTML | 返回 `title` 字段（2025-07 新增）|
| `/screenshot` | PNG/JPEG 二进制 | 网页截图 | 默认视口 1920×1080 |
| `/pdf` | PDF 二进制 | 生成打印级 PDF | 上限 50 MB，支持页眉页脚模板变量 |
| `/snapshot` | JSON（HTML + screenshot） | 同时捕获结构与视觉 | 组合快照 |
| `/scrape` | JSON（按 CSS selector） | 定向元素抽取 | 精确 DOM 提取 |
| `/links` | URL 数组 | 提取页面所有链接 | 支持 `excludeExternalLinks` 参数 |
| `/json` | 结构化 JSON | **AI 驱动**数据抽取 | 需提供 prompt 或 JSON Schema |

### 3.3 `/markdown` 端点参数详解

这是本地工具开发者最核心的端点，完整参数集：

```jsonc
{
  // ── 必填之一 ──
  "url": "https://example.com",
  // "html": "<div>raw html</div>",

  // ── JS 渲染控制（关键） ──
  "gotoOptions": {
    // 可选值: domcontentloaded | load | networkidle0 | networkidle2
    // networkidle0: 等待所有网络请求完成（SPA 推荐）
    // networkidle2: 允许 2 个活跃请求时视为完成（大多数动态页面）
    "waitUntil": "networkidle0",
    "timeout": 30000      // 导航超时，毫秒
  },

  // 等待特定元素出现后再抓取（比 networkidle 更精准高效）
  "waitForSelector": "#main-content",

  // ── 资源过滤（节省浏览器时间和费用） ──
  "rejectRequestPattern": ["/\\.css$/", "/\\.woff2?$/", "/analytics/"],

  // ── 请求控制 ──
  "setExtraHTTPHeaders": {
    "Accept-Language": "zh-CN,zh;q=0.9"
  },
  "cookies": [
    { "name": "session", "value": "xxx", "domain": ".example.com" }
  ],
  "authenticate": { "username": "user", "password": "pass" }, // HTTP Basic Auth

  // ── 注入自定义 JS（执行后再抓取） ──
  "addScriptTag": [{ "content": "document.querySelector('.cookie-banner')?.remove()" }],

  // ── 自定义 UA（注意：不能绕过 bot 检测） ──
  "userAgent": "Mozilla/5.0 ...",

  // ── 缓存控制 ──
  "cacheTTL": 0          // 0=禁用; 默认=5秒; 最大=86400（1天）
}
```

### 3.4 `/json` 端点：AI 驱动的结构化抽取

这个端点能力往往被低估，它内置 Workers AI 模型进行语义理解：

```jsonc
{
  "url": "https://shop.example.com/product/123",

  // 方式 A：自然语言 prompt
  "prompt": "Extract product name, price, and stock availability",

  // 方式 B：JSON Schema 强类型约束（推荐，结果更稳定）
  "schema": {
    "type": "object",
    "properties": {
      "name":    { "type": "string" },
      "price":   { "type": "number" },
      "inStock": { "type": "boolean" },
      "sku":     { "type": "string" }
    },
    "required": ["name", "price"]
  },

  // 可指定自定义模型 + fallback 链（2025-07 新增）
  "custom_ai": [
    "@cf/meta/llama-3.1-8b-instruct-fp8-fast",
    "@cf/mistral/mistral-7b-instruct-v0.1"
  ],

  "gotoOptions": { "waitUntil": "networkidle2" }
}
```

> **计费说明**：`/json` 端点使用 Workers AI 模型，会消耗 Workers AI Neurons 配额，在 Workers AI 费用侧产生额外计费。

### 3.5 自动附加的请求标识头（不可移除）

Browser Rendering 在所有出站请求中自动附加以下三个头，使用密码学签名证明请求源自 Cloudflare：

```
Signature-agent: cloudflare-browser-rendering
Signature: <cryptographic-signature>
Signature-input: <input-parameters>
```

**这些头无法被覆盖或移除**，这是 Cloudflare「透明合规抓取」的设计原则。实际影响：

- **有利方面**：声誉良好的目标站点可能因此放行请求，认为是合规机器人
- **不利方面**：明确封锁 Cloudflare IP 段的站点，BR 无法规避
- **实际上**：用户指定的 `userAgent` 参数不能绕过 bot 检测，请求始终会被标识为机器人

### 3.6 超时体系

Browser Rendering 有三个独立计时器，互不干扰：

```
┌─────────────────────────────────────────────────────┐
│  1. gotoOptions.timeout   → 页面导航超时（自定义）    │
│  2. waitForSelector 超时  → 等待特定元素的超时        │
│  3. Session 不活跃超时    → 默认 60 秒，最长 10 分钟  │
└─────────────────────────────────────────────────────┘
```

响应头 `X-Browser-Ms-Used` 返回本次请求实际消耗的浏览器毫秒数，是估算费用的直接依据。

---

## 4. 功能三：Workers AI toMarkdown

### 4.1 功能定位

Workers AI `toMarkdown` 是一个**文件内容转换引擎**，与 Browser Rendering 的根本区别在于：

```
Workers AI toMarkdown  → 处理离线文档（PDF、Office、图片等）
Browser Rendering      → 渲染在线动态网页
```

它不启动浏览器，不执行 JavaScript，只是将各种文件格式转换为 Markdown。

### 4.2 支持格式完整列表

| 类别 | 支持的扩展名 | MIME Type | 备注 |
|------|-------------|-----------|------|
| PDF | `.pdf` | `application/pdf` | 提取 Metadata + 按页结构化内容 |
| 图片 | `.jpg/.jpeg/.png/.webp/.svg` | `image/*` | 调用**两个** AI 模型：物体检测 + 文字摘要 |
| HTML | `.html` | `text/html` | 静态 HTML，**不执行 JS** |
| XML | `.xml` | `application/xml` | |
| Excel | `.xlsx/.xlsm/.xlsb/.xls/.et` | `vnd.ms-excel.*` | |
| Word | `.docx` | `vnd.openxmlformats...wordprocessingml...` | |
| ODF | `.ods/.odt` | `vnd.oasis.opendocument.*` | |
| CSV | `.csv` | `text/csv` | |
| Apple Numbers | `.numbers` | `vnd.apple.numbers` | |

> 通过 `GET /ai/tomarkdown/supported` 接口可动态获取最新支持格式列表，无需硬编码。

### 4.3 定价说明（已勘误）

**⚠️ 原叙述有误**：此前对话中将 `toMarkdown` 描述为「按 token 计费」，这是不准确的。

**✅ 正确说明**：

- **PDF、Office、HTML、XML、CSV** 等格式转换：**免费**，不消耗任何配额
- **图片转换**：会调用两个 Workers AI 视觉模型（物体检测 + 摘要生成），消耗 Workers AI **Neurons** 配额

Workers AI Neurons 配额：

| 计划 | 免费额度 | 超出费率 |
|------|----------|----------|
| Free + Paid | 10,000 Neurons/天 | $0.011 / 1,000 Neurons |

Neurons 是 Cloudflare 衡量 GPU 算力消耗的单位，与 token 不是同一个概念，但官方提供了换算参考表。对于图片 OCR 场景，实际成本取决于使用的具体视觉模型。

### 4.4 REST API 调用

```bash
# 单文件转换
curl https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/tomarkdown \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  -F "files=@report.pdf;type=application/pdf"

# 批量转换（一次请求多个文件）
curl https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/tomarkdown \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  -F "files=@report.pdf;type=application/pdf" \
  -F "files=@data.xlsx;type=application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# 查询当前支持的格式列表
curl https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/tomarkdown/supported \
  -H "Authorization: Bearer $CF_API_TOKEN"
```

### 4.5 响应数据结构

```jsonc
{
  "result": [
    {
      "name": "report.pdf",          // 与输入文件名一致
      "format": "markdown",          // "markdown" | "error"
      "mimetype": "application/pdf", // 检测到的 MIME 类型
      "tokens": 4231,                // 估算 token 数（仅 format=markdown 时存在）
      "data": "# report.pdf\n## Metadata\n- PDFFormatVersion=1.4\n..."
      // 若 format=error: "error": "失败原因描述"
    }
  ]
}
```

PDF 转换结果包含结构化元数据段：

```markdown
# report.pdf
## Metadata
- PDFFormatVersion=1.4
- IsLinearized=false
- Producer=Adobe Acrobat 2024
- Title=Annual Report 2025

## Contents
### Page 1
正文内容按页面分节组织...
```

---

## 5. 三大功能横向对比

### 5.1 核心维度对比

| 维度 | Markdown for Agents | BR `/markdown` | `toMarkdown` |
|------|---------------------|----------------|--------------|
| **输入类型** | URL（远程站点） | URL 或原始 HTML 字符串 | 本地/远程文件 Blob |
| **JS 渲染** | ❌ 依赖 Origin 返回的 HTML | ✅ 完整 Chrome 渲染 | ❌ |
| **支持格式** | 仅 HTML | 仅 HTML | PDF/Office/图片/XML/CSV 等 |
| **依赖条件** | 目标站点需启用 | CF 账号 | CF 账号 |
| **客户端费用** | 免费 | 按浏览器时长计费 | 大多数格式免费 |
| **Origin 体积上限** | 2 MB | 无明确文档限制 | — |
| **响应 token hint** | ✅ `x-markdown-tokens` | ❌ | ✅ `result[].tokens` |
| **缓存支持** | CF 边缘缓存 | `cacheTTL` 参数（默认 5 秒） | 无缓存层 |

### 5.2 站点类型适配性

| 站点类型 | Accept: text/markdown | BR `/markdown` | `toMarkdown` |
|----------|----------------------|----------------|--------------|
| CF 托管文档站（已启用） | ✅ 最佳，零成本 | ✅ 可用但冗余 | ❌ 不适用 |
| 普通静态博客 | 🔶 大多数未启用，会 fallback | ✅ | ❌ |
| React/Vue/Next.js SPA | ❌ | ✅ + `networkidle0` | ❌ |
| 需要登录的内容 | ❌ | ✅ + `cookies` 注入 | ❌ |
| 强反爬站点 | ❌ | 🔶 CF 签名头可识别，效果不保证 | ❌ |
| 本地 PDF / Office 文档 | ❌ | ❌ | ✅ 最佳 |
| 图片 OCR | ❌ | ❌ | ✅ 有费用 |
| 动态渲染 + 结构化抽取 | ❌ | ✅ + `/json` 端点 | ❌ |

---

## 6. 面向本地工具的集成架构

### 6.1 设计原则

作为本地工具（类似 markitdown），面向最终用户的设计应遵循「**渐进增强**」原则：

```
免费能力  →  无感知默认开启，不需要任何配置
付费能力  →  可选的 --cloud 模式，用户显式配置 API Token 后激活
```

这样对于最常见的 URL 抓取场景（命中 CF 托管站点），工具立刻比纯本地方案更强；对不想注册 CF 账号的用户也不构成障碍。

### 6.2 分层客户端实现

#### 层一：URL 抓取器（零成本）

```python
# converter/fetchers/url_fetcher.py
import httpx
from dataclasses import dataclass
from typing import Optional

@dataclass
class FetchResult:
    content: str
    source: str          # "cf_markdown" | "html_raw"
    token_hint: Optional[int] = None
    is_markdown: bool = False

USER_AGENT = "MyMarkdownTool/1.0 (like MarkItDown; +https://github.com/you/tool)"

async def fetch_url(url: str, timeout: int = 15) -> FetchResult:
    """优先尝试 CF Markdown for Agents，自动降级到 HTML"""
    headers = {
        "Accept": "text/markdown;q=1.0, text/html;q=0.8, */*;q=0.5",
        "User-Agent": USER_AGENT,
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()

        content_type = r.headers.get("content-type", "")
        if "text/markdown" in content_type:
            token_hint = r.headers.get("x-markdown-tokens")
            return FetchResult(
                content=r.text,
                source="cf_markdown",
                token_hint=int(token_hint) if token_hint else None,
                is_markdown=True,
            )
        # 降级：拿到 HTML，交给本地 pipeline 或 BR API 处理
        return FetchResult(content=r.text, source="html_raw")
```

#### 层二：Browser Rendering 客户端（动态页面 + 复杂抓取）

```python
# converter/backends/browser_rendering.py
import httpx
from typing import Optional

class BrowserRenderingClient:
    BASE = "https://api.cloudflare.com/client/v4"

    def __init__(self, account_id: str, api_token: str):
        self.account_id = account_id
        self._headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

    async def fetch_markdown(
        self,
        url: str,
        wait_until: str = "load",
        wait_for_selector: Optional[str] = None,
        reject_patterns: Optional[list] = None,
        cookies: Optional[list] = None,
        cache_ttl: int = 300,
    ) -> str:
        payload = {
            "url": url,
            "gotoOptions": {"waitUntil": wait_until, "timeout": 30000},
            "cacheTTL": cache_ttl,
        }
        if wait_for_selector:
            payload["waitForSelector"] = wait_for_selector
        if reject_patterns:
            payload["rejectRequestPattern"] = reject_patterns
        if cookies:
            payload["cookies"] = cookies

        endpoint = f"{self.BASE}/accounts/{self.account_id}/browser-rendering/markdown"
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(endpoint, headers=self._headers, json=payload)
            r.raise_for_status()
            # X-Browser-Ms-Used 头可用于成本估算
            ms_used = r.headers.get("X-Browser-Ms-Used", "0")
            data = r.json()
            return data["result"]

    async def extract_json(self, url: str, schema: dict) -> dict:
        """结构化数据抽取，使用 Workers AI 模型"""
        payload = {
            "url": url,
            "schema": schema,
            "gotoOptions": {"waitUntil": "networkidle2"},
        }
        endpoint = f"{self.BASE}/accounts/{self.account_id}/browser-rendering/json"
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(endpoint, headers=self._headers, json=payload)
            r.raise_for_status()
            return r.json()["result"]
```

#### 层三：Workers AI toMarkdown 客户端（本地文件）

```python
# converter/backends/workers_ai.py
import httpx
import mimetypes
from pathlib import Path

class WorkersAIClient:
    BASE = "https://api.cloudflare.com/client/v4"

    # 大多数格式免费；图片格式会消耗 Neurons 配额
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".svg"}
    SUPPORTED_EXTENSIONS = {
        ".pdf", ".html", ".xml",
        ".xlsx", ".xlsm", ".xlsb", ".xls", ".et",
        ".docx", ".ods", ".odt", ".csv", ".numbers",
    } | IMAGE_EXTENSIONS

    def __init__(self, account_id: str, api_token: str):
        self.account_id = account_id
        self._headers = {"Authorization": f"Bearer {api_token}"}

    def is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def will_incur_cost(self, path: Path) -> bool:
        """图片转换会消耗 Workers AI Neurons 配额"""
        return path.suffix.lower() in self.IMAGE_EXTENSIONS

    async def convert_file(self, path: Path) -> dict:
        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            raise ValueError(f"Cannot determine MIME type for: {path}")

        url = f"{self.BASE}/accounts/{self.account_id}/ai/tomarkdown"
        async with httpx.AsyncClient(timeout=60) as client:
            with open(path, "rb") as f:
                r = await client.post(
                    url,
                    headers=self._headers,
                    files={"files": (path.name, f, mime)},
                )
            r.raise_for_status()
            result = r.json()["result"][0]

        if result["format"] == "error":
            raise RuntimeError(f"Conversion failed: {result['error']}")

        return {
            "markdown": result["data"],
            "tokens": result.get("tokens"),
            "mimeType": result["mimetype"],
        }

    async def get_supported_formats(self) -> list:
        """动态获取支持格式列表，无需硬编码"""
        url = f"{self.BASE}/accounts/{self.account_id}/ai/tomarkdown/supported"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=self._headers)
            r.raise_for_status()
            return r.json()["result"]
```

#### 层四：统一入口

```python
# converter/core.py
from pathlib import Path
from typing import Optional
from .fetchers.url_fetcher import fetch_url
from .backends.browser_rendering import BrowserRenderingClient
from .backends.workers_ai import WorkersAIClient
from .local.html_parser import local_html_to_markdown  # 自行实现的 fallback

class MarkdownConverter:
    def __init__(
        self,
        br_client: Optional[BrowserRenderingClient] = None,
        ai_client: Optional[WorkersAIClient] = None,
    ):
        self.br = br_client
        self.ai = ai_client

    async def from_url(
        self,
        url: str,
        force_browser: bool = False,
        spa: bool = False,
    ) -> str:
        if force_browser or spa:
            if not self.br:
                raise RuntimeError("Browser Rendering requires --cf-token and --cf-account")
            wait = "networkidle0" if spa else "load"
            return await self.br.fetch_markdown(url, wait_until=wait)

        result = await fetch_url(url)
        if result.is_markdown:
            return result.content  # CF 边缘已经转好了，直接使用

        # HTML fallback：优先本地解析（免费），避免不必要的 API 调用
        return local_html_to_markdown(result.content)

    async def from_file(self, path: Path) -> str:
        path = Path(path)

        if path.suffix.lower() in {".md", ".txt"}:
            return path.read_text(encoding="utf-8")

        if path.suffix.lower() in {".html", ".htm"}:
            return local_html_to_markdown(path.read_text())

        if not self.ai:
            raise RuntimeError(
                f"Converting {path.suffix} requires --cf-token and --cf-account. "
                "Supported: PDF, DOCX, XLSX, images, and more."
            )

        if self.ai.will_incur_cost(path):
            # 图片转换：给用户提示
            print(f"Note: Image conversion uses Workers AI (Neurons quota)")

        if not self.ai.is_supported(path):
            raise ValueError(f"Unsupported format: {path.suffix}")

        result = await self.ai.convert_file(path)
        return result["markdown"]
```

### 6.3 CLI 设计建议

```
# 最简使用（不需要任何账号）
$ mytool https://developers.cloudflare.com/workers-ai/

# 本地 HTML 文件（本地解析）
$ mytool ./page.html

# 动态 SPA 页面（需要 CF 账号）
$ mytool --cf-token $TOKEN --cf-account $ID --spa https://app.example.com

# 需要等待特定元素（精确控制 + 节省费用）
$ mytool --cf-token $TOKEN --cf-account $ID \
         --wait-for "#article-content" \
         https://news.example.com/article/123

# 本地 PDF/Office 文件（需要 CF 账号）
$ mytool --cf-token $TOKEN --cf-account $ID report.pdf

# 强制使用 Browser Rendering（绕过 Accept: text/markdown 尝试）
$ mytool --cf-token $TOKEN --cf-account $ID --browser https://example.com
```

配置文件 `~/.config/mytool/config.toml`：

```toml
[cloudflare]
account_id = "abc123def456"
api_token  = "your_token_here"

[defaults]
# URL 抓取时是否先尝试 Accept: text/markdown（建议 true）
try_cf_markdown = true
# 多少 token 以上触发拆分警告
token_warn_threshold = 100_000
# 默认 BR 缓存时间（秒）
br_cache_ttl = 300
```

### 6.4 token hint 的实际运用

`x-markdown-tokens` 响应头是一个被低估的实用工具：

```python
result = await fetch_url(url)

if result.token_hint:
    if result.token_hint > 128_000:
        print(f"⚠️  ~{result.token_hint:,} tokens — exceeds most LLM context windows")
        print("    Consider using --chunk flag for automatic splitting")
    elif result.token_hint > 32_000:
        print(f"ℹ️  ~{result.token_hint:,} tokens — fits GPT-4/Claude, may exceed smaller models")
    else:
        print(f"✓  ~{result.token_hint:,} tokens")
```

---

## 7. 完整决策树与路由逻辑

```
输入类型判断
    │
    ├─ 本地文件路径
    │    ├─ .md / .txt
    │    │    └─→ 直接读取，输出（免费，无需任何 API）
    │    │
    │    ├─ .html / .htm
    │    │    └─→ 本地 HTML 解析（推荐自行实现）
    │    │         或 toMarkdown REST API（有网络往返但简单）
    │    │
    │    ├─ .pdf / .docx / .xlsx / .ods 等 Office 格式
    │    │    └─→ Workers AI toMarkdown API ✓（绝大多数格式免费）
    │    │
    │    ├─ .jpg / .png / .webp（图片 OCR）
    │    │    └─→ Workers AI toMarkdown API ✓（消耗 Neurons 配额，有费用）
    │    │
    │    └─ 其他格式
    │         └─→ 报错提示，引导用户使用支持的格式
    │
    └─ URL
         │
         ├─ Step 1：发送 Accept: text/markdown 请求（免费优先）
         │    ├─ 返回 text/markdown
         │    │    └─→ 直接使用，读取 x-markdown-tokens ✓
         │    │
         │    └─ 返回 text/html / 其他
         │         │
         │         ├─ 内容已足够（静态页面，无重要 JS 渲染内容）
         │         │    └─→ 本地 HTML 解析（免费 fallback）
         │         │
         │         └─ 内容不完整 / 疑似 SPA
         │              │
         │              └─ Step 2：Browser Rendering API
         │                   ├─ 普通页面        → waitUntil: "load"
         │                   ├─ 新闻/内容站      → waitUntil: "domcontentloaded"
         │                   ├─ SPA（React/Vue） → waitUntil: "networkidle0"
         │                   ├─ 已知页面结构     → waitForSelector: "#content"
         │                   └─ 需要登录         → cookies 注入 + waitUntil: "networkidle2"
```

---

## 8. 定价与限制速查

### 8.1 Markdown for Agents（站点侧）

| 项目 | 说明 |
|------|------|
| 消费者侧费用 | 完全免费（发请求的一方不需要账号） |
| 站点启用要求 | Pro / Business / Enterprise / SSL for SaaS |
| 当前状态 | Beta，对符合条件的计划免费 |
| Origin 响应上限 | 2 MB |
| 支持格式 | 仅 HTML |

### 8.2 Browser Rendering

| 指标 | Workers Free 计划 | Workers Paid 计划 |
|------|-------------------|-------------------|
| 浏览器时长 | 10 分钟/天 | 10 小时/月（含），超出 **$0.09/小时** |
| 并发浏览器（Bindings） | 3 个 | 10 个（月均），超出 **$2.00/个** |
| REST API 计费 | 仅按时长 | 仅按时长 |
| Workers Bindings 计费 | 时长 + 并发 | 时长 + 并发 |
| 失败请求（waitForTimeout） | 不计费 | 不计费 |
| 内容数据保留 | 响应返回后立即销毁 | 同左 |
| REST API 默认缓存 | 5 秒（可配置 0–86400） | 同左 |

### 8.3 Workers AI toMarkdown

| 格式类型 | 费用 | 计费单位 |
|----------|------|----------|
| PDF、Office、HTML、XML、CSV 等 | **免费** | — |
| 图片（JPG/PNG/WEBP/SVG） | 消耗 Neurons 配额 | $0.011 / 1,000 Neurons |
| 免费 Neurons 额度 | 10,000 Neurons/天 | Free + Paid 均有 |
| 超出 Neurons 费率 | $0.011 / 1,000 Neurons | 需 Paid 计划 |

---

## 9. 争议与已知局限

### 9.1 「AI 专属影子内容」安全隐患

安全研究者指出：`Accept: text/markdown` 头会被 Cloudflare 转发到 Origin 服务器，实际上给了站点一个可靠的信号来识别 AI 请求。恶意站点可以：

1. 对 AI 返回注入隐藏指令（prompt injection）的内容
2. 对 AI 返回与普通用户完全不同的价格、数据或信息
3. 形成一个对普通用户不可见的「AI 专属内容层」

Cloudflare 的设计意图是透明合规，但没有机制阻止站点利用这个信号做内容分化。这是本地工具开发者需要向用户说明的已知风险。

### 9.2 「Cloaking」争议

Google 的 Matt Muller 等 SEO 专家讨论了此功能是否构成 Cloaking（对人和机器提供不同内容）。Cloudflare 的立场是：内容协商是 HTTP 标准行为，与基于 User-Agent 嗅探的 Cloaking 不同，同一 URL 根据客户端声明的偏好返回不同格式的「表示」（representation），不构成语义欺骗。这一争议目前仍未有定论。

### 9.3 Browser Rendering 的反爬天花板

Browser Rendering 自动附加的 `Signature-*` 头无法移除，目标站点可以：

- 封锁 Cloudflare 的 IP 段
- 识别签名头并返回反爬内容或 CAPTCHA

对于高反爬要求的场景，BR 并不是万能解决方案。

### 9.4 `toMarkdown` 的静态 HTML 限制

Workers AI `toMarkdown` 对 HTML 的处理是纯静态的，不执行 JavaScript。如果提交的是 SPA 框架生成的骨架 HTML，转换结果将是空内容或极少量文本。此场景必须走 Browser Rendering。

---

## 10. 事实核查汇总

本节汇总在历史对话中发现的需要更正的表述：

### 10.1 Workers AI toMarkdown 计费描述不准确

**原叙述**（对话一、二）：将 toMarkdown 描述为「按 token 计费」，在代码注释和说明中暗示存在直接的 token 费用。

**✅ 正确说法**：toMarkdown 的计费分两类。PDF、Office、HTML、XML、CSV 等文件格式的转换**免费**，不消耗任何配额。图片转换例外，会调用 Workers AI 视觉模型，消耗 Neurons 配额（而非 token），每天有 10,000 Neurons 免费额度，超出按 $0.011/1,000 Neurons 计费。

### 10.2 Browser Rendering 的 Free 计划可用性未充分说明

**原叙述**（对话二）：在三方案对比表中，将 BR 归类为「需要 CF 账号」，容易给人「需要付费计划」的印象。

**✅ 正确说法**：Browser Rendering 在 **Free 计划也可使用**（10 分钟/天，3 个并发），无需 Paid 计划。但 REST API 从 2025 年 4 月才 GA，2025 年 8 月开始计费。

### 10.3 Markdown for Agents 的 Free Plan 限制

**原叙述**：曾提及该功能需要 Pro 及以上，这是正确的——但需要补充说明清楚：这个计划限制仅适用于**站点管理员开启该功能**，**作为消费者（请求方）发送 `Accept: text/markdown` 完全免费且无需账号**。

**✅ 正确说法**：两个角色的限制是完全不同的，不应混为一谈。

### 10.4 其余已核实的关键数据

- ✅ 80% token 节省（16,180 → 3,150）：数据正确，来自 Cloudflare 官方博客，基于其自身一篇博客文章的实测
- ✅ Origin 上限 2 MB（初版 1 MB，2026-02-16 升至 2 MB）：正确
- ✅ `x-markdown-tokens` 响应头：正确
- ✅ `Content-Signal: ai-train=yes, search=yes, ai-input=yes` 默认值：正确
- ✅ BR 定价 $0.09/小时：正确
- ✅ BR Paid 月包含 10 小时：正确
- ✅ Claude Code 和 OpenCode 发送 `Accept: text/markdown`：正确，来自 CF 官方博客
- ✅ BR 8 个 REST API 端点：正确
- ✅ `toMarkdown` 支持批量处理（一次请求多文件）：正确
- ✅ `/json` 端点的 `custom_ai` 参数（2025-07 新增）：正确

---

*报告生成时间：2026-02-23*
*数据来源：Cloudflare 官方文档、变更日志、发布博客（均为第一手资料）*

以下是原始问题 Q&A：

# 深度调研下 cloudflare 最近新出的直接请求网站 markdown 内容的功能，以及我作为 AI 开发者如何利用这一功能获取干净网页内容

这是 Cloudflare 在 2026 年 2 月中旬刚发布的新功能，非常值得深入了解。下面系统整理。

---

## Cloudflare Markdown for Agents 深度解析

### 背景：为什么需要这个功能

随着 AI 爬虫和 Agent 成为 Web 流量的重要来源，HTML 与 AI 的矛盾日益突出：Web 是为人类浏览器设计的，充斥着导航栏、样式标签、`<div>` 嵌套等对语义毫无贡献的噪音。将 HTML 转成 Markdown 是目前几乎所有 AI pipeline 的标配步骤，但这意味着每个消费方都在重复做同一件事，浪费算力、增加成本。

token 效率的差距非常显著：一个 `## About Us` 标题在 Markdown 中约消耗 3 个 token，而其 HTML 等价形式 `<h2 class="section-title" id="about">About Us</h2>` 要消耗 12–15 个，这还没算外层的 `div`、导航栏和脚本标签。Cloudflare 自己的一篇博客文章，HTML 版需要 16,180 个 token，Markdown 版只需 3,150 个，**节省约 80%**。

---

### 核心机制：HTTP 内容协商

这个功能的技术实现非常优雅，完全遵循 HTTP 标准的 [Content Negotiation](https://developer.mozilla.org/en-US/docs/Web/HTTP/Content_negotiation) 机制：

当 AI 系统向已启用该功能的站点发出请求时，只需在请求头中声明 `Accept: text/markdown`，Cloudflare 的边缘网络就会自动从 Origin 拉取 HTML，实时转换为 Markdown 后返回给客户端。

一个最基本的 curl 示例：

```bash
curl https://developers.cloudflare.com/fundamentals/reference/markdown-for-agents/ \
  -H "Accept: text/markdown"
```

响应头中会额外包含 `x-markdown-tokens`，这个 header 包含返回内容的估算 token 数，帮助开发者判断文档是否超出 context window、或者是否需要拆分处理。

---

### 功能约束与限制

| 约束项 | 详情 |
|--------|------|
| 仅转换 HTML | 目前不支持 PDF 等其他格式 |
| Origin 响应上限 | 2 MB（初版为 1 MB，2 月 16 日已升至 2 MB） |
| 需站点开启 | 必须是启用了该功能的 Cloudflare 托管站点 |
| 付费计划 | Pro / Business / Enterprise / SSL for SaaS |

目前 Cloudflare 已在自己的 Developer Documentation 和 Blog 上启用了该功能，邀请所有 AI 爬虫和 Agent 使用 Markdown 而非 HTML 消费内容。

---

### Content-Signal：隐藏的附加影响

启用 Markdown for Agents 后，每个转换响应默认附带 `Content-Signal: ai-train=yes, search=yes, ai-input=yes` header，意味着内容默认被声明为可用于 AI 训练、搜索结果和 AI 推理（包含 Agent 使用）。对内容授权有顾虑的站长在开启前需仔细考量。

---

## 作为 AI 开发者，你有三条路可以用

### 路径一：直接请求已启用站点（零成本）

适合：**爬取 Cloudflare 托管的文档、博客类站点**（如 Cloudflare 自己的 docs、很多技术博客）。

```python
import httpx

async def fetch_markdown(url: str) -> str:
    headers = {
        "Accept": "text/markdown, text/html;q=0.9",
        "User-Agent": "MyAIAgent/1.0"
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=headers)

        # 检查是否真的返回了 markdown
        content_type = r.headers.get("content-type", "")
        if "text/markdown" in content_type:
            token_hint = r.headers.get("x-markdown-tokens")
            print(f"Token estimate: {token_hint}")
            return r.text
        else:
            # fallback：站点未启用，自行处理 HTML
            return html_to_markdown_fallback(r.text)
```

`Accept` header 中用 `q` 值设置优先级，这样对未启用的站点仍能正常获取 HTML 而不报错。

---

### 路径二：Workers AI `toMarkdown()` API（支持多种格式）

适合：**需要处理任意文档（PDF、Office、图片等）**，不依赖目标站点是否启用该功能。

Workers AI 提供 `env.AI.toMarkdown()` 工具方法，支持 PDF、HTML、XML、Excel（`.xlsx`/`.xlsm`/`.xls`）、Word（`.docx`）、ODF、CSV、Apple Numbers 以及图片（`.jpeg`/`.png`/`.webp`/`.svg`，使用模型做 OCR 和摘要）等多种格式。

在 Cloudflare Worker 中使用：

```javascript
export default {
  async fetch(request, env) {
    // 例：从上游拉取 PDF 并转换
    const pdfResponse = await fetch("https://example.com/report.pdf");
    const pdfBlob = await pdfResponse.blob();

    const [result] = await env.AI.toMarkdown([
      {
        name: "report.pdf",
        blob: pdfBlob
      }
    ]);

    if (result.format === "markdown") {
      return new Response(JSON.stringify({
        markdown: result.data,
        tokens: result.tokens,
        mimeType: result.mimetype
      }), {
        headers: { "Content-Type": "application/json" }
      });
    }

    return new Response(`Error: ${result.error}`, { status: 500 });
  }
};
```

REST API 方式（适合非 Worker 环境，比如从你的后端调用）：

```bash
curl https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/tomarkdown \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  -F "files=@document.pdf;type=application/pdf"
```

---

### 路径三：Browser Rendering API + Markdown（处理 JS 渲染页面）

对于需要浏览器渲染才能呈现内容的动态页面（SPA、需要执行 JS 的站点），Cloudflare 提供 Browser Rendering `/markdown` REST API，先完整渲染页面，再转换为 Markdown。

```bash
curl https://api.cloudflare.com/client/v4/accounts/{account_id}/browser-rendering/markdown \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/dynamic-page"}'
```

---

## 三种方案对比

| 方案 | 适用场景 | 成本 | 依赖 |
|------|----------|------|------|
| 直接请求 `Accept: text/markdown` | 已启用的 CF 站点，HTML 页面 | 免费 | 目标站点需启用 |
| Workers AI `toMarkdown()` | 任意文档（PDF/Office/图片等） | 按 token 计费 | 无，只需 CF 账号 |
| Browser Rendering `/markdown` | JS 渲染的动态页面 | 按请求计费 | 无 |

---

## 在 RAG Pipeline 中的最佳实践

```python
import httpx

async def smart_fetch(url: str, cf_api_token: str) -> dict:
    """
    优先尝试 CF Markdown for Agents，
    降级到 Workers AI toMarkdown
    """
    # 1. 先尝试直接请求
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers={"Accept": "text/markdown"})

    if "text/markdown" in r.headers.get("content-type", ""):
        return {
            "markdown": r.text,
            "tokens": int(r.headers.get("x-markdown-tokens", 0)),
            "source": "cf_edge"
        }

    # 2. fallback：将 HTML 送到 Workers AI 转换
    html_blob = r.content
    # 调用 Workers AI REST API...
    result = await call_workers_ai_tomarkdown(html_blob, cf_api_token)
    return {"markdown": result["data"], "tokens": result["tokens"], "source": "workers_ai"}
```

token estimate header 还有一个很实用的场景：在把内容送进 LLM 之前，先读取 `x-markdown-tokens`，如果超出 context window 就提前做分块，避免请求失败。

---

## 值得注意的争议

安全顾问 David McSweeney 指出，`Accept: text/markdown` header 会被转发到 Origin 服务器，这实际上给了站点一个可靠的信号来识别 AI 请求——恶意站主可以对 AI 返回特制内容（注入指令、篡改数据），而对普通用户返回正常 HTML，形成一种"AI 专属影子内容"的风险。 这个问题目前 Cloudflare 尚未给出正式回应，作为 AI 开发者需要对此保持意识。

总体而言，这个功能对 AI 开发者最大的价值在于：**将原本每个消费方都需要重复做的 HTML→Markdown 转换，推到了离内容最近的网络边缘**，既节省了自己的算力和 token 成本，也让 pipeline 更简洁。对于大规模网页抓取或 RAG 场景，收益非常可观。


# 我是一个本地工具（类似微软开源的 markitdown）开发者，请问对于我想要开发的这个本地工具，如何更好地利用上述 cloudflare 的能力？

这是一个很好的切入点。作为本地工具开发者，你面临的核心挑战是：**用户的输入来源多样（URL、本地文件、剪贴板内容等），但你自己没有服务器资源**。Cloudflare 的三个能力可以分层次地补充你的本地处理能力。

---

## 架构全景

```
用户输入
   │
   ├─ URL ──────────────────────────────────────────────────────┐
   │   ├─ 静态页面 → [1] Accept: text/markdown (免费, 无需账号)  │
   │   └─ 动态页面 → [3] CF Browser Rendering API              │
   │                                                            │
   └─ 本地文件 ─────────────────────────────────────────────────┤
       ├─ HTML     → 本地处理（你自己实现）                       │
       ├─ PDF/Office/图片 → [2] CF Workers AI toMarkdown API    │
       └─ 纯文本   → 直接输出                                    │
                                                              输出
                                                           Markdown
```

---

## 层一：URL 抓取 — 零成本的优先路径

这是最值得内置的能力，**完全免费、不需要 CF 账号**，只是一个 HTTP header：

```python
# converter/fetchers/url_fetcher.py

import httpx
from dataclasses import dataclass
from typing import Optional

@dataclass
class FetchResult:
    content: str
    source: str          # "cf_markdown" | "html_raw" | "dynamic"
    token_hint: Optional[int] = None
    is_markdown: bool = False

USER_AGENT = "MyMarkdownTool/1.0 (like MarkItDown; +https://github.com/you/tool)"

async def fetch_url(url: str, timeout: int = 15) -> FetchResult:
    """
    优先尝试 CF Markdown for Agents，自动降级
    """
    headers = {
        # 优先 markdown，退而求其次接受 HTML
        "Accept": "text/markdown;q=1.0, text/html;q=0.8, */*;q=0.5",
        "User-Agent": USER_AGENT,
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()

        content_type = r.headers.get("content-type", "")

        if "text/markdown" in content_type:
            token_hint = r.headers.get("x-markdown-tokens")
            return FetchResult(
                content=r.text,
                source="cf_markdown",
                token_hint=int(token_hint) if token_hint else None,
                is_markdown=True,
            )

        # 降级：拿到 HTML，交给本地 pipeline 继续处理
        return FetchResult(
            content=r.text,
            source="html_raw",
            is_markdown=False,
        )
```

这一层不需要用户配置任何东西，直接内置进工具的默认行为即可。

---

## 层二：CF Workers AI REST API — 处理复杂文件格式

这是你本地工具最难自己实现的部分（PDF 解析、Office 文档、图片 OCR）。Workers AI 的 `toMarkdown` REST API 可以作为**可选的云端增强后端**。

**定价参考**（截至文档发布时）：普通文档转换按 token 计费，图片走视觉模型另计，Free tier 有一定额度。

### 客户端封装

```python
# converter/backends/cloudflare_api.py

import httpx
import mimetypes
from pathlib import Path

class CloudflareMarkdownBackend:
    BASE_URL = "https://api.cloudflare.com/client/v4"

    # Workers AI toMarkdown 支持的格式
    SUPPORTED_EXTENSIONS = {
        ".pdf", ".html", ".xml",
        ".xlsx", ".xlsm", ".xlsb", ".xls", ".et",
        ".docx", ".ods", ".odt", ".csv", ".numbers",
        ".jpg", ".jpeg", ".png", ".webp", ".svg",
    }

    def __init__(self, account_id: str, api_token: str):
        self.account_id = account_id
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    async def convert_file(self, path: Path) -> dict:
        """
        调用 Workers AI toMarkdown REST API
        返回: {"markdown": str, "tokens": int, "mimeType": str}
        """
        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            raise ValueError(f"Cannot determine MIME type for {path}")

        url = f"{self.BASE_URL}/accounts/{self.account_id}/ai/tomarkdown"

        async with httpx.AsyncClient(timeout=60) as client:
            with open(path, "rb") as f:
                r = await client.post(
                    url,
                    headers=self.headers,
                    files={"files": (path.name, f, mime)},
                )
            r.raise_for_status()
            data = r.json()

        # API 返回 list，取第一个结果
        result = data["result"][0]
        if result["format"] == "error":
            raise RuntimeError(f"CF API error: {result['error']}")

        return {
            "markdown": result["data"],
            "tokens": result.get("tokens"),
            "mimeType": result["mimetype"],
        }

    async def convert_url(self, url: str) -> dict:
        """
        对于动态页面，走 Browser Rendering API
        """
        endpoint = f"{self.BASE_URL}/accounts/{self.account_id}/browser-rendering/markdown"
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                endpoint,
                headers={**self.headers, "Content-Type": "application/json"},
                json={"url": url},
            )
            r.raise_for_status()
            return r.json()
```

---

## 层三：把三层组合成统一入口

```python
# converter/core.py

from pathlib import Path
from typing import Optional
from .fetchers.url_fetcher import fetch_url
from .backends.cloudflare_api import CloudflareMarkdownBackend
from .local.html_parser import local_html_to_markdown   # 你自己实现的 fallback

class MarkdownConverter:
    def __init__(self, cf_backend: Optional[CloudflareMarkdownBackend] = None):
        self.cf = cf_backend  # 未配置时降级到本地处理

    async def from_url(self, url: str, force_browser: bool = False) -> str:
        if force_browser and self.cf:
            result = await self.cf.convert_url(url)
            return result["markdown"]

        result = await fetch_url(url)
        if result.is_markdown:
            return result.content  # CF 边缘已经转好了

        # HTML fallback：优先本地解析
        return local_html_to_markdown(result.content)

    async def from_file(self, path: Path) -> str:
        path = Path(path)

        # 纯文本直接返回
        if path.suffix.lower() in {".md", ".txt"}:
            return path.read_text(encoding="utf-8")

        # HTML 本地处理
        if path.suffix.lower() in {".html", ".htm"}:
            return local_html_to_markdown(path.read_text())

        # 复杂格式：优先走 CF API
        if self.cf and self.cf.is_supported(path):
            result = await self.cf.convert_file(path)
            return result["markdown"]

        raise ValueError(
            f"Unsupported format: {path.suffix}. "
            f"Configure a Cloudflare API token to enable cloud conversion."
        )
```

---

## CLI 设计建议

用户体验上，CF 相关配置应该是**可选的渐进增强**，不配置时工具依然可用：

```
# 最简用法（不需要任何账号）
$ mytool https://developers.cloudflare.com/workers-ai/

# 本地 HTML 文件
$ mytool ./page.html

# 需要 CF 账号的高级用法
$ mytool --cf-token $TOKEN --cf-account $ACCOUNT_ID report.pdf
$ mytool --cf-token $TOKEN --cf-account $ACCOUNT_ID --browser https://spa.example.com
```

配置文件 `~/.config/mytool/config.toml`：

```toml
[cloudflare]
account_id = "abc123"
api_token   = "your_token_here"

[defaults]
# URL 抓取时是否总是先尝试 text/markdown
try_cf_markdown = true
# 超过多少 token 触发警告
token_warn_threshold = 100000
```

---

## 一个实用细节：token hint 的利用

`x-markdown-tokens` header 对本地工具有一个非常实用的场景——**在用户将内容送进 LLM 之前提前预警**：

```python
result = await fetch_url(url)

if result.token_hint:
    if result.token_hint > 128_000:
        print(f"⚠️  Content is ~{result.token_hint:,} tokens, "
              f"may exceed most LLM context windows. Consider --chunk flag.")
    else:
        print(f"✓  ~{result.token_hint:,} tokens")
```

---

## 能力边界总结

| 场景 | 推荐方案 | 是否需要 CF 账号 |
|------|----------|-----------------|
| 抓取 CF 托管站点 | `Accept: text/markdown` | 否 |
| 抓取普通静态网页 | 本地 HTML 解析（你自己实现） | 否 |
| 抓取 JS 渲染页面 | CF Browser Rendering API | **是** |
| 转换 PDF / Office | CF Workers AI toMarkdown | **是** |
| 转换本地图片（OCR） | CF Workers AI toMarkdown | **是** |

核心设计原则：**免费的能力无感知地默认开启，付费的 CF API 作为可选的 `--cloud` 模式**。这样你的工具对于最常见的 URL 抓取场景立刻就比纯本地方案强（因为 CF 边缘转换质量和效率都更好），同时对不想注册 CF 账号的用户也不构成障碍。


# 进一步充分调研下 CF Browser Rendering API 和 CF Workers AI toMarkdown 的能力全集和站点适配情况

现在我有了足够充分的第一手文档数据，系统整理如下。

---

## CF Browser Rendering API 能力全集

### REST API 端点全览（8个）

所有端点共享统一的 URL 结构：`https://api.cloudflare.com/client/v4/accounts/<accountId>/browser-rendering/<endpoint>`，认证需要 Bearer Token，并要求 `Browser Rendering - Edit` 权限。

每个端点都接受 `url`（抓取远程页面）或 `html`（渲染本地 HTML 字符串）二选一作为必填输入。

| Endpoint | 输出类型 | 核心用途 |
|----------|----------|----------|
| `/markdown` | Markdown 字符串 | 网页转 Markdown，AI 内容消费 |
| `/content` | HTML 字符串 + 页面标题 | 获取渲染后完整 HTML |
| `/screenshot` | PNG/JPEG 二进制 | 网页截图，默认视口 1920×1080 |
| `/pdf` | PDF 二进制，上限 50 MB | 生成打印级 PDF |
| `/snapshot` | JSON（HTML + screenshot） | 同时捕获结构与视觉状态 |
| `/scrape` | JSON（按 CSS selector 提取） | 定向元素抽取 |
| `/links` | URL 数组 | 提取页面所有链接，支持过滤外链 |
| `/json` | 结构化 JSON | **AI 驱动**的数据抽取，需提供 prompt/schema |

### `/markdown` 端点参数详解

这是作为本地工具开发者最核心的端点，其参数远比表面看起来丰富：

```jsonc
{
  // 必填之一
  "url": "https://example.com",
  // "html": "<div>raw html</div>",

  // JS 渲染控制（最关键！）
  "gotoOptions": {
    // domcontentloaded | load | networkidle0 | networkidle2
    "waitUntil": "networkidle0",
    "timeout": 30000       // 导航超时，毫秒
  },

  // 等待特定元素出现后再抓取（比 networkidle 更精准高效）
  "waitForSelector": "#main-content",

  // 资源过滤（节省浏览器时间）
  "rejectRequestPattern": ["/^.*\\.(css|woff2|png|jpg)/"],

  // 请求头伪装（不能绕过 bot 检测，但可影响内容协商）
  "setExtraHTTPHeaders": {
    "Accept-Language": "zh-CN,zh;q=0.9"
  },

  // 注入自定义 JS（执行后再抓取）
  "addScriptTag": [{ "content": "window.__PRERENDER__=true" }],

  // Cookie 注入（处理需登录的内容）
  "cookies": [{ "name": "session", "value": "xxx", "domain": ".example.com" }],

  // HTTP Basic Auth
  "authenticate": { "username": "user", "password": "pass" },

  // 缓存控制（默认 5 秒，最长 1 天）
  "cacheTTL": 0,

  // 自定义 UA（不能绕过 bot 检测）
  "userAgent": "Mozilla/5.0 ..."
}
```

### `/json` 端点：隐藏的结构化抽取能力

该端点使用 Workers AI 模型驱动，支持指定 `prompt` 或 `schema` 提取结构化数据；还支持 `custom_ai` 参数指定自定义模型，并可以设置多个模型作为 fallback，按顺序尝试直到成功。使用此端点会同时在 Workers AI Dashboard 产生用量计费。

```jsonc
{
  "url": "https://shop.example.com/product/123",
  "prompt": "Extract product name, price, and availability",
  // 或者用 JSON Schema 做强类型约束
  "schema": {
    "type": "object",
    "properties": {
      "name": { "type": "string" },
      "price": { "type": "number" },
      "inStock": { "type": "boolean" }
    }
  },
  // 自定义模型 + fallback 链
  "custom_ai": [
    "@cf/meta/llama-3.1-8b-instruct",
    "@cf/mistral/mistral-7b-instruct-v0.1"
  ],
  "gotoOptions": { "waitUntil": "networkidle2" }
}
```

### 自动附加的请求标识头（不可移除）

Browser Rendering 会在所有出站请求中自动附加 `Signature-agent`、`Signature`、`Signature-input` 三个头，使用 Web Bot Auth 密码学签名验证请求源自 Cloudflare Browser Rendering，这些头无法被覆盖或移除。

这意味着：**目标站点可以感知到请求来自 CF BR，无法伪装成普通用户浏览器**。这是设计上的透明性原则，也是 CF 区别于竞品的关键。对本地工具开发者来说，这有利有弊——有些反爬站点可能因此放行，有些可能因此拦截。

### 超时体系（三个独立计时器）

```
导航超时 (gotoOptions.timeout)   → 页面 load/networkidle 等待时限
Selector 超时 (waitForSelector)  → 等待特定元素出现的时限
全局超时                          → 默认 60 秒不活跃后 session 自动关闭
```

### Browser Rendering 定价与限制

计费模型：REST API 只按浏览器时长计费；Workers Bindings 同时计费时长和并发数。

| 指标 | Free Plan | Paid Plan |
|------|-----------|-----------|
| 浏览器时长 | 10 分钟/天 | 10 小时/月，超出 $0.09/小时 |
| 并发浏览器（Bindings） | 3 个 | 10 个（月均），超出 $2.00/个 |

Cloudflare 以 ephemeral 方式处理内容，提交的 HTML 或生成的输出在响应返回后立即销毁。REST API 生成内容默认缓存 5 秒，可通过 `cacheTTL` 配置最长 1 天，设 0 禁用。waitForTimeout 失败的请求不计费。

响应头 `X-Browser-Ms-Used` 返回本次请求实际消耗的浏览器毫秒数，是估算成本的重要工具。

---

## CF Workers AI `toMarkdown` 能力全集

### 与 Browser Rendering 的根本区别

这两个 API 解决完全不同的问题：

```
toMarkdown        → 文件内容转换引擎（离线文档）
BR /markdown      → 网页渲染引擎（在线动态页面）
```

### 支持格式完整列表

| 类别 | 扩展名 | MIME Type | 备注 |
|------|--------|-----------|------|
| PDF | `.pdf` | `application/pdf` | 含 Metadata 提取 |
| 图片 | `.jpg/.jpeg/.png/.webp/.svg` | `image/*` | 调用视觉模型做 OCR + 摘要 |
| HTML | `.html` | `text/html` | 静态 HTML，不执行 JS |
| XML | `.xml` | `application/xml` | |
| Excel | `.xlsx/.xlsm/.xlsb/.xls/.et` | `vnd.ms-excel...` | |
| Word | `.docx` | `vnd.openxmlformats...wordprocessingml...` | |
| ODF | `.ods/.odt` | `vnd.oasis.opendocument.*` | |
| CSV | `.csv` | `text/csv` | |
| Apple | `.numbers` | `vnd.apple.numbers` | |

大多数格式转换本身是免费的；图片转换例外，会调用 Workers AI 视觉模型做物体检测和摘要，超出免费额度后产生额外费用。

### REST API 调用方式

```bash
# 转换单个文件
curl https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/tomarkdown \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  -F "files=@document.pdf;type=application/pdf"

# 批量转换（一次请求多个文件）
curl ... \
  -F "files=@report.pdf;type=application/pdf" \
  -F "files=@data.xlsx;type=application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# 查询支持格式列表（动态检测，无需硬编码）
curl https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/tomarkdown/supported \
  -H "Authorization: Bearer $CF_API_TOKEN"
```

### ConversionResult 响应结构

```jsonc
{
  "result": [
    {
      "name": "report.pdf",        // 与输入文件名一致
      "format": "markdown",        // "markdown" | "error"
      "mimetype": "application/pdf",
      "tokens": 4231,              // 估算 token 数（仅 format=markdown 时存在）
      "data": "# report.pdf\n## Metadata\n..."  // 转换结果
      // 若 format=error: "error": "失败原因"
    }
  ]
}
```

### PDF 转换结果示例（含 Metadata 段）

PDF 转换结果包含文件元数据段和按页面结构化的内容，例如 PDFFormatVersion、Producer、Title 等字段会被提取到 Metadata 区，内容按页分节组织。

```markdown
# somatosensory.pdf
## Metadata
- PDFFormatVersion=1.4
- IsLinearized=false
- Producer=Prince 20150210
- Title=Anatomy of the Somatosensory System

## Contents
### Page 1
This is a sample document...
```

---

## 对本地工具的完整决策树

综合以上调研，给出一个精确的路由逻辑：

```
输入
 │
 ├── 本地文件路径
 │    ├── .md / .txt                    → 直接读取，无需 API
 │    ├── .html / .htm（静态）           → 本地解析（推荐自行实现，免费）
 │    │                                   或 toMarkdown REST API（简单但有网络往返）
 │    ├── .pdf                           → toMarkdown API ✓
 │    ├── .docx / .xlsx / .ods 等        → toMarkdown API ✓
 │    ├── .jpg/.png/.webp（图片 OCR）    → toMarkdown API ✓（调用视觉模型，有费用）
 │    └── 不支持格式                     → 报错提示
 │
 └── URL
      ├── 已知静态站点 / CDN 托管        → Accept: text/markdown（免费优先）
      │    ├── 返回 text/markdown        → 直接使用 ✓
      │    └── 返回 text/html            → 本地 HTML 解析 fallback
      │
      ├── 疑似 SPA / JS 渲染
      │    └── BR /markdown API
      │         ├── 普通页面             → gotoOptions.waitUntil: "load"
      │         ├── SPA                  → waitUntil: "networkidle0"
      │         └── 特定元素驱动        → waitForSelector: "#content"
      │
      ├── 需要结构化数据（非 Markdown）
      │    └── BR /json API + schema
      │
      └── 需要截图 / PDF
           └── BR /screenshot 或 /pdf API
```

### 关键参数速查表（面向工具实现）

| 参数 | 所属 API | 作用 | 推荐值 |
|------|----------|------|--------|
| `waitUntil: networkidle0` | BR /markdown | 等待所有网络请求完成 | SPA 必选 |
| `waitUntil: networkidle2` | BR /markdown | 允许 2 个活跃请求时视为完成 | 大多数动态页面 |
| `waitForSelector` | BR /markdown | 等待特定 DOM 元素 | 已知页面结构时优先 |
| `rejectRequestPattern` | BR /markdown | 过滤无用资源请求 | `["/\\.css$/", "/\\.woff/"]` |
| `cacheTTL: 0` | BR /markdown | 禁用 CF 端缓存 | 实时内容 |
| `cacheTTL: 86400` | BR /markdown | 缓存 1 天 | 文档类页面，节省费用 |
| `gotoOptions.timeout` | BR /markdown | 导航超时 | `30000`（30 秒） |
| `X-Browser-Ms-Used` | BR 所有端点 | 响应头，实际耗时毫秒 | 用于成本监控 |

### 站点适配性分析

| 站点类型 | Accept: text/markdown | BR /markdown | toMarkdown |
|----------|----------------------|--------------|------------|
| CF 托管文档站（如 CF Docs 自身） | ✅ 最佳 | ✅ | ❌ 不适用 |
| 普通静态博客 | 🔶 可能不支持 | ✅ | ❌ |
| React/Vue SPA | ❌ | ✅ + networkidle | ❌ |
| 需登录的内容 | ❌ | ✅ + cookies | ❌ |
| 有强反爬的站点 | ❌ | 🔶 CF 签名头可能被识别 | ❌ |
| 本地 PDF / Office 文件 | ❌ | ❌ | ✅ 最佳 |
| 图片 OCR | ❌ | ❌ | ✅（有费用）|

值得注意的一个细节：**BR 的 Web Bot Auth 签名头设计初衷是"透明合规"**，Cloudflare 定位自己为"善意爬虫"的基础设施，部分严格反爬的站点看到 CF 签名后反而可能放行（认为是合规机器人）；但也有站点会明确封锁 CF IP 段，这是 BR 无法规避的现实约束。
