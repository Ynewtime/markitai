# 抓取策略引擎

给 markitai 一个 URL 时，它会按从便宜到昂贵的顺序尝试几种抓取方式，直到拿到可用的内容。这一页讲清楚顺序是什么、哪些不出本机、怎么调。

## 策略选择逻辑

三件事决定顺序：

1. **显式指定**：`-s static`、`-s playwright`、`-s defuddle`、`-s jina` 或 `-s cloudflare` 只用那一种策略。远程服务拒绝时（限流、鉴权失败），会回退到 `auto` 链而不是直接失败。
2. **域名配置**：按域名设置等待的选择器、额外等待时间，或自定义策略顺序。
3. **自适应回退**（默认的 `auto`）：下面两种顺序之一，再根据 markitai 对该域名的经验调整。

### 默认顺序（标准域名）

本机优先。内置的静态抓取器在抽取质量上不输远程阅读服务，而且从不把 URL 发出去。

<StrategyChain />

### SPA/重 JS 顺序

已知需要 JavaScript 的域名直接上浏览器：`instagram.com`、`fallback_patterns` 里列出的域名，以及静态抓取失败后被记入 SPA 缓存的域名。

<StrategyChain mode="spa" />

X/Twitter 默认保持 `playwright → defuddle → jina → cloudflare → static`。Playwright 策略对已允许远程提取的匿名推文/文章先尝试 FxTwitter，再尝试 oEmbed，成功便无需启动 Chromium。配置了 Cookie、自定义请求头、凭据或持久会话，要截图，或者网络提取失败时，markitai 仍会打开浏览器。`playwright(fxtwitter)` 或 `playwright(oembed)` 这个标记说明内容实际从哪来；补充成功后不再请求 defuddle。

浏览器导航遇到 HTTP 错误不一定抛出异常。遇到 HTTP 错误时，markitai 跳过元素等待和稳定等待，立即尝试 X 的补充提取。补充成功标记为 `playwright(fxtwitter)` 或 `playwright(oembed)`，这时浏览器并没有渲染出推文。补充失败或被关掉时，markitai 报告 HTTP 错误，`auto` 换下一个策略。

### 远程后备与仅限本地的 URL {#remote-fallback-and-local-only-urls}

默认（`fetch.remote_consent: always`）下，markitai 可能把公开 URL 发给 Defuddle、Jina 或 Cloudflare。首次使用时它在 stderr 显示一行简短提示，并在 `~/.markitai/notices/remote-fetch` 记下已展示，后续运行不再重复；VLM OCR 的同类提示也按此规则处理。这条记录只说明提示展示过，不代表用户授权：`remote_consent: ask` 仍按进程询问。涉及的服务有 defuddle.md、Jina、Cloudflare、FxTwitter 和 Twitter oEmbed。markitai 逐个尝试，URL 只会发给当前正在尝试的那一个。

X/Twitter 的 Playwright 策略可能调用 FxTwitter 和 Twitter oEmbed。它们和其他远程服务遵循同一个进程级的同意决定：在 `ask` 下可以触发那一次共享的询问或复用之前的回答，无法询问的运行则跳过它们。

下面这些 URL 无论选什么策略都不出本机：

- localhost、私有 IP 和常见的内网主机名
- 带凭据的 URL：userinfo、token、签名、密码、API key 或授权码

在 `auto` 链里，匹配 `fetch.policy.local_only_patterns` 和 `NO_PROXY`（`inherit_no_proxy` 开启时）的域名也只走本地。

| 设置 | 效果 | 显式 `-s` 能否覆盖 |
|------|------|-------------------|
| `remote_consent: always`（默认） | 公开 URL 可远程回退，在 stderr 提示 | — |
| `remote_consent: ask` | 有终端时每个进程问一次；非交互运行跳过所有远程服务 | — |
| `remote_consent: never` | 自动和配置选择的策略只走本地 | 能 |
| `local_only_patterns` / `NO_PROXY` | 匹配的域名在 `auto` 链里只走本地 | 能 |
| 私有、内网或带凭据的 URL | 永远本地 | **不能** |
| `MARKITAI_NO_REMOTE_FETCH=1` | 硬性只走本地 | **不能** |

只在配置文件里设了远程 `fetch.strategy` 不算显式选择：它仍受 `remote_consent` 管，也会打同样的首次提示。

Clash 等代理返回 `198.18.0.0/15` 或 `2001:2::/48` 中的 Fake-IP 地址时，远程抽取会通过 [Cloudflare DNS over HTTPS](https://developers.cloudflare.com/1.1.1.1/encryption/dns-over-https/make-api-requests/dns-json/) 复核域名的公网 A 和 AAAA 记录。复核只在允许远程访问后进行，DNS 只看到域名，看不到 URL 路径和查询参数。复核失败或解析结果含非公网地址时照样拦截。直接写非公网 IP 的 URL，以及本机连接检查（包括未认证的 `serve` 请求），不管复核结果如何都不放行。

## 配置

在 `markitai.json` 里调整策略：

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
| `enabled` | boolean | `true` | 开关策略排序 |
| `max_strategy_hops` | integer | `5` | 放弃前最多尝试几种策略 |
| `strategy_priority` | list | `null` | 自定义全局策略顺序 |
| `local_only_patterns` | list | `[]` | 只走本地策略的域名和 IP（`NO_PROXY` 语法） |
| `inherit_no_proxy` | boolean | `true` | 把 `NO_PROXY` 里的条目也当作只走本地 |

### 域名配置

按域名覆盖：

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `wait_for_selector` | string | `null` | 抽取前等待的 CSS 选择器 |
| `wait_for` | string | `null` | 页面加载事件：`load`、`domcontentloaded`、`networkidle`。不设则继承 `fetch.playwright.wait_for` |
| `extra_wait_ms` | integer | `null` | 加载事件后的额外等待。不设则继承 `fetch.playwright.extra_wait_ms` |
| `prefer_strategy` | string | `null` | 这个域名优先尝试的策略 |
| `strategy_priority` | list | `null` | 这个域名的完整策略顺序（覆盖全局顺序和 `prefer_strategy`） |
| `skip_auto_scroll` | boolean | `false` | 单内容页面（推文、issue、文档）跳过自动滚动 |
| `reject_resource_patterns` | list | `null` | 拦截匹配这些 URL 模式的浏览器请求，如 `["**/analytics/**"]` |

markitai 内置了 `x.com`、`twitter.com` 和 `github.com` 的配置。只有你写了的字段才会覆盖内置浏览器调优；改了策略或额外等待时间，其余设置照旧。设置 `skip_auto_scroll: false` 可恢复滚动，`reject_resource_patterns: []` 可清空资源过滤，将可空浏览器字段设为 `null` 则继承对应的全局设置。

```json
{
  "fetch": {
    "domain_profiles": {
      "x.com": { "wait_for_selector": "[data-testid=tweetText]", "extra_wait_ms": 1200 },
      "instagram.com": { "wait_for": "networkidle", "extra_wait_ms": 2000 },
      "docs.example.com": { "prefer_strategy": "static" }
    }
  }
}
```

### Playwright 会话持久化

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `session_mode` | string | `"isolated"` | `isolated`：每个请求新开浏览器上下文。`domain_persistent`：按域名复用上下文 |
| `session_ttl_seconds` | integer | `600` | 持久会话保留多久 |

`domain_persistent` 会复用 cookie 和本地存储，对同一站点的多次请求快得多。

## 静态 HTTP 适配器

静态策略用 httpx，绝大多数站点都没问题。若站点因为 TLS 指纹检测返回 403 或空内容，`auto` 会自己回退到 Playwright 或 Cloudflare。想不开浏览器就绕过指纹检测，换成 curl-cffi：

```bash
uv pip install markitai[extra-fetch]
export MARKITAI_STATIC_HTTP=curl_cffi
```

没装 curl-cffi 时，markitai 会静默使用 httpx。

## 工作原理

markitai 逐个尝试策略，最多 `max_strategy_hops` 次。第一个通过校验的结果就是最终结果，运行到此结束。

### 结果校验

内容为空或过短、登录墙、反爬或验证码页面（Geetest、Cloudflare、reCAPTCHA、hCaptcha）都会校验失败，转到下一个策略。

### SPA 学习

静态抓取成功但页面说需要 JavaScript 时，markitai 把该域名记入 SPA 缓存 30 天，之后的请求直接上浏览器。只有这个信号会写入缓存，验证码、登录墙和网络错误都不会。用 `markitai cache spa-domains` 查看或清除。
