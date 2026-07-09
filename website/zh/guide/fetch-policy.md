# Fetch Policy 引擎

Markitai 使用策略驱动的 Fetch Policy 引擎来确定获取 URL 内容的最佳策略。该引擎设计为弹性、高效且用户友好。

## 策略选择逻辑

引擎按以下策略驱动的方式选择抓取策略的顺序：

1. **显式策略**：如果您提供了显式策略（如 `-s playwright`、`-s defuddle` 或 `-s jina`），引擎通常只会使用该策略——但有两个例外：显式的 `-s defuddle`/`-s jina`/`-s cloudflare` 在远程服务拒绝请求时（限流、认证失败等）仍会优雅回退到完整的 `auto` 链路；而域名配置中的内容类设置（`wait_for_selector`、`skip_auto_scroll` 等）即便在显式选择 `playwright` 时也仍然生效。
2. **域名配置**：您可以为特定域名配置专属设置，如自定义等待选择器、额外等待时间，或完整的自定义策略顺序。
3. **自适应回退**：在 `auto` 模式（默认）下，引擎根据域名和历史成功记录智能排序策略。

### 默认顺序（标准域名）

Markitai 采用本地优先策略：对于大多数网站，会先尝试原生本地流水线，再使用任何需要征得同意的远程服务：

```
Static (HTTP) → Playwright (浏览器) → Defuddle → Jina → Cloudflare
```

Static 的原生 webextract 流水线在提取质量基准语料库上已能匹敌远程 Defuddle（在中日韩文本间距处理上甚至更优），因此排在最前——而且与远程策略不同，它不会把 URL 发送到本机之外，也无需抓取同意。

### SPA/重 JS 顺序

对于已知需要 JavaScript 的域名（如 `x.com`、`instagram.com`、`fallback_patterns` 中列出的域名，或此前静态抓取失败并已被学习进 SPA 缓存的域名），Markitai 会直接跳转到浏览器：

```
Playwright (浏览器) → Defuddle → Jina → Cloudflare → Static
```

这里 Static 排在最后，因为对这些域名它已经失败过（或预期会失败），无法产出可用内容。

## 配置

您可以在 `markitai.json` 中调整 fetch policy：

```json
{
  "fetch": {
    "policy": {
      "enabled": true,
      "max_strategy_hops": 5
    },
    "domain_profiles": {
      "x.com": {
        "wait_for_selector": "[data-testid=tweetText]",
        "wait_for": "domcontentloaded",
        "extra_wait_ms": 1200
      }
    },
    "playwright": {
      "session_mode": "domain_persistent",
      "session_ttl_seconds": 600
    }
  }
}
```

### Policy 选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | boolean | `true` | 启用或禁用智能策略排序 |
| `max_strategy_hops` | integer | `5` | 放弃前尝试的最大策略数 |
| `strategy_priority` | list | `null` | 自定义全局策略顺序（覆盖默认优先级） |
| `local_only_patterns` | list | `[]` | 限制为本地策略的域名/IP 模式（NO_PROXY 语法） |
| `inherit_no_proxy` | boolean | `true` | 将 `NO_PROXY` 环境变量合并到 `local_only_patterns` |

### 域名配置

域名配置允许按域名覆盖抓取行为：

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `wait_for_selector` | string | `null` | 提取内容前等待的 CSS 选择器 |
| `wait_for` | string | `null` | 页面加载事件覆盖值（`load`、`domcontentloaded`、`networkidle`）；未设置时继承全局 `fetch.playwright.wait_for`（默认 `domcontentloaded`） |
| `extra_wait_ms` | integer | `null` | 页面加载事件后的额外等待毫秒数覆盖值；未设置时继承全局 `fetch.playwright.extra_wait_ms`（默认 `3000`） |
| `prefer_strategy` | string | `null` | 该域名的首选策略（`static`、`defuddle`、`playwright`、`cloudflare`、`jina`） |
| `strategy_priority` | list | `null` | 该域名的自定义策略顺序（覆盖全局和 `prefer_strategy`） |
| `skip_auto_scroll` | boolean | `false` | 对单内容页面（推文、issue、文档）跳过自动滚动 |
| `reject_resource_patterns` | list | `null` | 阻止 Playwright 导航中匹配这些 URL 模式的资源（如 `["**/analytics/**"]`） |

Markitai 内置了 `x.com`/`twitter.com` 和 `github.com` 的域名配置。如果您为同一域名配置自己的条目，会**整体替换**内置配置，而不是逐字段合并——内置的调优（例如 x.com 配置里的 `skip_auto_scroll`/`reject_resource_patterns`）会随之丢失，除非您自己重新声明。

多域名配置示例：

```json
{
  "fetch": {
    "domain_profiles": {
      "x.com": {
        "wait_for_selector": "[data-testid=tweetText]",
        "extra_wait_ms": 1200
      },
      "instagram.com": {
        "wait_for": "networkidle",
        "extra_wait_ms": 2000
      },
      "docs.example.com": {
        "prefer_strategy": "static"
      }
    }
  }
}
```

### Playwright 会话持久化

控制 Playwright 如何管理浏览器上下文：

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `session_mode` | string | `"isolated"` | `isolated`：每个请求新建上下文；`domain_persistent`：按域名复用上下文 |
| `session_ttl_seconds` | integer | `600` | 持久化会话的保活时间（秒） |

使用 `domain_persistent` 模式可以通过复用 cookies、localStorage 等浏览器状态，显著加速对同一站点的多次请求。

## 静态 HTTP 适配器

静态抓取（Static 策略）默认使用 **httpx**，适用于绝大多数网站。对于有 TLS 指纹检测的反爬站点，可选择启用 **curl-cffi** 适配器。

| 适配器 | 安装方式 | 特点 |
|--------|----------|------|
| **httpx**（默认） | 内置，开箱即用 | 快速可靠，覆盖大多数场景 |
| **curl-cffi**（可选） | `uv pip install markitai[extra-fetch]` | 模拟 Chrome TLS/HTTP 签名，绕过部分反爬保护 |

::: tip 何时需要 curl-cffi？
大多数情况下不需要。如果 Static 策略对某些站点返回 403/空内容，Policy Engine 会自动回退到 Playwright 或 Cloudflare。只有当您需要在**不启动浏览器**的前提下绕过 TLS 指纹检测时，才需要 curl-cffi。
:::

启用 `curl-cffi`：

```bash
# 安装
uv pip install markitai[extra-fetch]

# 设置环境变量激活
export MARKITAI_STATIC_HTTP=curl_cffi
```

即使设置了环境变量但未安装 curl-cffi，Markitai 也会静默降级到 httpx，不会报错。

## 工作原理

```
URL 请求
    │
    ├─ 显式策略 (-s static/playwright/defuddle/jina/cloudflare)?
    │       └─ 是 → 仅使用该策略
    │
    ├─ 域名在 SPA 缓存中或已知需要 JS（fallback_patterns）?
    │       └─ 是 → SPA 顺序（Playwright → Defuddle → Jina → Cloudflare → Static）
    │
    └─ 默认 → 标准顺序（Static → Playwright → Defuddle → Jina → Cloudflare）
            │
            ├─ 尝试策略 #1 → 成功? → 完成
            ├─ 尝试策略 #2 → 成功? → 完成
            ├─ 尝试策略 #3 → 成功? → 完成
            ├─ 尝试策略 #4 → 成功? → 完成
            └─ 尝试策略 #5 → 成功? → 完成 / 放弃
```

域名配置（`strategy_priority` 或 `prefer_strategy`）以及全局 `strategy_priority` 覆盖项，可以在 SPA/默认回退之前按域名或全局重新排序此链路——详见下方[域名配置](#域名配置)。私有/本地/内网域名以及匹配 `local_only_patterns` 的域名，无论上述规则如何，都只会限定使用本地策略（`static`、`playwright`）。

每个策略在接受结果前都会验证内容质量——检查内容是否为空/过短、是否命中登录墙，以及是否是反爬/CAPTCHA 挑战页面（Geetest、Cloudflare、reCAPTCHA、hCaptcha）。校验未通过时会回退到下一个策略。

::: tip
当静态抓取成功但内容显示需要 JavaScript 渲染（或页面为空）时，该域名会被加入 SPA 缓存，有效期 30 天。后续对该域名的请求将直接跳到浏览器渲染，节省时间。其他失败情形（CAPTCHA、登录墙、网络错误）不会触发这一学习机制——只有"需要 JS"这一信号会。
:::
