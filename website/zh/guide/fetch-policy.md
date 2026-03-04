# Fetch Policy 引擎

Markitai 使用策略驱动的 Fetch Policy 引擎来确定获取 URL 内容的最佳策略。该引擎设计为弹性、高效且用户友好。

## 策略选择逻辑

引擎按以下策略驱动的方式选择抓取策略的顺序：

1. **显式策略**：如果您提供了显式策略（如 `--playwright` 或 `--jina`），引擎将仅使用该策略。
2. **域名配置**：您可以为特定域名配置专属设置，如自定义等待选择器或额外等待时间。
3. **自适应回退**：在 `auto` 模式（默认）下，引擎根据域名和历史成功记录智能排序策略。

### 默认顺序（标准域名）

对于大多数网站，Markitai 优先考虑速度：

```
Static (HTTP) → Playwright (浏览器) → Cloudflare → Jina
```

### SPA/重 JS 顺序

对于已知需要 JavaScript 的域名（如 `x.com`、`instagram.com`，或之前静态抓取失败的域名）：

```
Playwright (浏览器) → Cloudflare → Jina → Static
```

## 配置

您可以在 `markitai.json` 中调整 fetch policy：

```json
{
  "fetch": {
    "policy": {
      "enabled": true,
      "max_strategy_hops": 4
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
| `max_strategy_hops` | integer | `4` | 放弃前尝试的最大策略数 |

### 域名配置

域名配置允许按域名覆盖抓取行为：

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `wait_for_selector` | string | `null` | 提取内容前等待的 CSS 选择器 |
| `wait_for` | string | `"domcontentloaded"` | 等待的页面加载事件（`load`、`domcontentloaded`、`networkidle`） |
| `extra_wait_ms` | integer | `3000` | 页面加载事件后的额外等待毫秒数 |
| `prefer_strategy` | string | `null` | 该域名的首选策略（`playwright`、`cloudflare`、`jina`、`static`） |

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
    ├─ 显式策略 (--playwright/--jina/--cloudflare)?
    │       └─ 是 → 仅使用该策略
    │
    ├─ 域名在 SPA 缓存中或已知需要 JS?
    │       └─ 是 → SPA 顺序（Playwright 优先）
    │
    └─ 默认 → 标准顺序（Static 优先）
            │
            ├─ 尝试策略 #1 → 成功? → 完成
            ├─ 尝试策略 #2 → 成功? → 完成
            ├─ 尝试策略 #3 → 成功? → 完成
            └─ 尝试策略 #4 → 成功? → 完成 / 放弃
```

每个策略在接受结果前都会验证内容质量。如果内容为空或过短，将回退到下一个策略。

::: tip
当域名静态抓取失败时，会自动添加到 SPA 域名缓存中。后续对该域名的请求将直接跳到浏览器渲染，节省时间。
:::
