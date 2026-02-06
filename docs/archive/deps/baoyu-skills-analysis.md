# baoyu-skills 技术分析报告

> 调研日期: 2026-02-06
> 调研范围: baoyu-danger-x-to-markdown | baoyu-url-to-markdown | baoyu-format-markdown
> 来源: `~/.claude/plugins/marketplaces/baoyu-skills/skills/`

## 概述

baoyu-skills 是一套面向 Claude Code 的 skill 插件，实现了三种 Markdown 转换能力。它们以 TypeScript 编写，通过 `npx -y bun` 免安装执行，设计为 LLM agent 调用的工具链而非人类直接使用的 CLI。

| Skill | 定位 | 核心技术 | 代码量 |
|-------|------|----------|--------|
| **x-to-markdown** | X/Twitter → Markdown | 逆向工程 GraphQL API + CDP 认证 | ~1400 行 (13 文件) |
| **url-to-markdown** | 任意 URL → Markdown | Chrome CDP + Network Idle + DOM 清洗 | ~500 行 (5 文件) |
| **format-markdown** | Markdown 排版优化 | Remark AST + CJK 间距 + 全角引号 | ~200 行 (3 文件 + 依赖) |

---

## 一、baoyu-danger-x-to-markdown

### 1.1 架构

```
main.ts              CLI 入口 + 同意管理 + 输出路由
├── graphql.ts       API 发现 + GraphQL 请求构建
│   └── http.ts      缓存的 HTML 获取 + Feature Flag 解析
├── cookies.ts       三层认证: env → file → CDP 浏览器
│   └── cookie-file.ts  Cookie 持久化
├── thread.ts        线程重建: 分页 + 去重 + 排序
├── tweet-to-markdown.ts  编排: thread + article → markdown
│   ├── tweet-article.ts  嵌入式文章提取
│   ├── thread-markdown.ts 推文序列渲染
│   └── markdown.ts       Draft.js content_state → markdown
├── types.ts         全类型定义
├── constants.ts     固化的 queryId + featureSwitches 回退值
└── paths.ts         XDG 规范的跨平台路径
```

### 1.2 核心机制

#### 动态 API 发现 (graphql.ts)

X 的 GraphQL API 使用动态 `queryId`，每次部署会变。发现过程：

```
1. fetchHomeHtml("https://x.com")
2. 正则提取 JS bundle hash:
   "bundle.TwitterArticles.{hash}" → CDN URL
   "main.{hash}" → Tweet API chunk
   "api:{hash}" → TweetDetail API chunk
3. 下载对应 JS bundle
4. 正则提取:
   queryId:"xxx", operationName:"ArticleEntityResultByRestId"
   featureSwitches:[...], fieldToggles:[...]
5. 如果任何步骤失败，使用 FALLBACK_* 常量兜底
```

三套独立的发现逻辑对应三个 GraphQL 操作：
- `ArticleEntityResultByRestId` — 文章内容
- `TweetResultByRestId` — 单条推文
- `TweetDetail` — 推文详情 + 对话线程

#### Feature Flag 动态解析 (http.ts:36-65)

不是硬编码 feature flags，而是从 X 首页 HTML 中动态读取每个 feature 的实际状态：

```typescript
// 匹配两种模式：
// 非转义: "feature_name": {"value": true}
// 转义:   \"feature_name\": {\"value\": true}
function resolveFeatureValue(html, key) {
  const unescaped = /"key"\s*:\s*\{"value"\s*:\s*(true|false)/;
  const escaped = /\\"key\\"\s*:\s*\\{\\"value\\"\s*:\s*(true|false)/;
  return html.match(unescaped) ?? html.match(escaped);
}
```

`buildFeatureMap` 对每个 feature switch 查找真实值，找不到则用 `defaults` 或默认 `true`。

#### 三层 Cookie 认证 (cookies.ts)

优先级从高到低：

| 层级 | 来源 | 说明 |
|------|------|------|
| 1 | 环境变量 `X_AUTH_TOKEN`, `X_CT0` | 优先级最高，始终覆盖 |
| 2 | Cookie 文件缓存 | 上次 CDP 登录写入的 `cookies.json` |
| 3 | Chrome CDP 交互登录 | 最后手段，打开浏览器让用户登录 |

合并逻辑: `{ ...fileMap, ...cdpMap, ...inlineMap }` — env 变量最终胜出。

CDP 登录的完整流程：
1. `getFreePort()` — `net.createServer().listen(0)` 动态分配端口
2. `spawn(chrome, ["--remote-debugging-port=PORT", "--user-data-dir=PROFILE"])` — 启动 Chrome
3. 轮询 `http://127.0.0.1:PORT/json/version` 直到 WebSocket URL 可用
4. `CdpConnection.connect(wsUrl)` — 建立 WebSocket
5. `Target.createTarget → Target.attachToTarget → Network.enable` — 进入 x.com
6. 每秒 `Network.getCookies(["https://x.com/", "https://twitter.com/"])` 轮询
7. 检测到 `auth_token` + `ct0` 后返回，写入 cookie 文件
8. `finally`: `Browser.close` → `SIGTERM` → 2s 后 `SIGKILL`（`.unref()` 避免阻塞退出）

#### 线程重建算法 (thread.ts:140-311)

X 的 TweetDetail API 返回分页的对话数据，需要多次请求才能拿到完整线程：

```
fetchTweetDetail(tweetId) → 初始 entries + cursors
  ↓
topCursor 向上循环: 获取更早的推文（线程开头可能不在第一页）
  ↓
moreCursor 循环: 展开 ShowMore/ShowMoreThreads
  ↓
bottomCursor: 获取最底部的推文
  ↓
以最后一条推文再查一次: 确保线程尾部完整
  ↓
去重: Map<id_str, TweetEntry>
  ↓
追溯根节点: while(rootEntry.in_reply_to_status_id_str) 向上找
  ↓
时间排序 + 从 root 截断
  ↓
isSameThread 过滤
```

`isSameThread` 判定条件（thread.ts:168-179）：
- 同 `user_id_str`（同一作者）
- 同 `conversation_id_str`（同一对话）
- 且满足以下之一：
  - `id_str === rootEntry.id_str`（就是根推文）
  - `in_reply_to_user_id_str === rootEntry.user_id_str`（回复给线程作者）
  - `!in_reply_to_user_id_str`（不是回复任何人的独立推文）

防护措施: `maxRequestCount = 1000` 上限防止无限循环。

#### Draft.js Content State 渲染 (markdown.ts:120-258)

X Article 使用 Draft.js 的 `content_state` 格式存储富文本：

```typescript
content_state: {
  blocks: [
    { type: "header-one", text: "Title", entityRanges: [] },
    { type: "atomic", text: " ", entityRanges: [{ key: 0 }] },
    { type: "unstyled", text: "Paragraph...", entityRanges: [] }
  ],
  entityMap: {
    "0": { value: { type: "MEDIA", data: { mediaItems: [{ mediaId: "123" }] } } }
  }
}
```

渲染器实现：
- 块类型映射: `header-one` → `#`, `header-two` → `##`, ..., `blockquote` → `>`, `code-block` → `` ``` ``
- 列表连续性追踪: `listKind` + `orderedIndex` 状态机
- 代码块合并: `inCodeBlock` 标志，连续 `code-block` 合并为一个 fenced block
- 媒体解析: `entityRanges[].key` → `entityMap[key].value.data.mediaItems` → `mediaById.get(mediaId)` → `![](url)`
- 视频选择: 从 `variants` 中选最高 `bit_rate` 的 mp4
- `usedUrls` Set 全局去重避免重复插入同一媒体

输出格式:
```markdown
---
url: https://x.com/user/status/123
requested_url: <原始输入 URL>
author: "Name (@username)"
tweet_count: 5
---

<推文内容 / 文章内容>
```

### 1.3 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| API 方式 | 逆向 GraphQL（非官方） | 官方 API 限制严格且收费 |
| queryId 获取 | 动态提取 + 硬编码回退 | 适应 X 频繁部署，回退保证可用性 |
| 认证方式 | env > file > CDP 三层 | 自动化场景用 env，交互场景用 CDP |
| 同意机制 | 本地 JSON 文件 + 版本号 | 逆向工程的法律风险需要用户明确知情 |
| 运行时 | Bun via npx | TypeScript 直接执行，零编译步骤 |

---

## 二、baoyu-url-to-markdown

### 2.1 架构

```
main.ts              CLI + 编排: Chrome → 捕获 → 转换 → 保存
├── cdp.ts           Chrome DevTools Protocol 客户端
│   ├── CdpConnection       WebSocket 客户端 + 事件订阅
│   ├── waitForNetworkIdle   pending 请求计数器
│   ├── autoScroll           滚动触发懒加载
│   └── killChrome           SIGTERM → SIGKILL 清理
├── html-to-markdown.ts     DOM 清洗脚本 + HTML→MD 正则转换
├── constants.ts            超时/延迟参数
└── paths.ts                XDG 路径
```

### 2.2 核心机制

#### 事件驱动的 CDP 客户端 (cdp.ts:28-105)

与 x-to-markdown 的 CDP 客户端相比，增加了事件订阅系统：

```typescript
class CdpConnection {
  private pending = new Map<number, { resolve, reject, timer }>();  // request/response
  private eventHandlers = new Map<string, Set<handler>>();          // event push

  // WebSocket message 分流:
  onMessage(data) {
    if (msg.id)         → pending.get(msg.id).resolve(result)   // response
    else if (msg.method) → eventHandlers.get(method).forEach(h)  // event
  }

  on(event, handler)   // 订阅事件
  off(event, handler)  // 取消订阅
}
```

#### Network Idle 检测 (cdp.ts:198-220)

```
状态机:
  pending = 0 (活跃请求计数)
  timer = null (idle 判定定时器)

事件:
  Network.requestWillBeSent → pending++, resetTimer()
  Network.loadingFinished   → pending--, if (pending ≤ 2) resetTimer()
  Network.loadingFailed     → pending--, if (pending ≤ 2) resetTimer()

resolve 条件:
  pending ≤ 2 且持续 1500ms 无新请求
```

**关键设计: `pending ≤ 2` 而非 `=== 0`。** 这比 Playwright 的 `networkidle`（要求 0 pending 500ms）更宽容，能处理有持续后台请求的页面（analytics、WebSocket keepalive、长轮询等）。

#### 自动捕获流水线 (main.ts:104-117)

```
1. Page load event (15s timeout) OR 8s 强制超时  ← Race
2. Network idle (1.5s at ≤2 pending)
3. 800ms 静默延迟
4. Auto scroll (最多 8 步, 每步 600ms, 检测 scrollHeight 变化)
5. 800ms 静默延迟
6. 注入 DOM 清洗脚本 + 提取内容
```

自动滚动算法 (cdp.ts:271-281):
```typescript
for (let i = 0; i < 8; i++) {
  scrollTo(0, document.body.scrollHeight);
  sleep(600ms);
  if (newHeight === lastHeight) break;  // 高度不变则停止
  lastHeight = newHeight;
}
scrollTo(0, 0);  // 滚回顶部
```

#### DOM 清洗 + 元数据提取 (html-to-markdown.ts:15-89)

注入页面执行的 JavaScript 脚本，执行以下步骤：

**步骤 1: 移除噪音元素**（13 类选择器）:
```javascript
const removeSelectors = [
  'script', 'style', 'noscript', 'iframe', 'svg', 'canvas',
  'header nav', 'footer', '.sidebar', '.nav', '.navigation',
  '.advertisement', '.ad', '.ads', '.cookie-banner', '.popup',
  '[role="banner"]', '[role="navigation"]', '[role="complementary"]'
];
```

**步骤 2: 清理属性** — 移除所有 `style`, `onclick`, `onload`, `onerror`

**步骤 3: 相对 URL → 绝对 URL**:
```javascript
a.setAttribute('href', new URL(href, document.baseURI).href);
img.setAttribute('src', new URL(src, document.baseURI).href);
```

**步骤 4: 元数据提取**（优先级链）:
- title: `og:title` > `twitter:title` > `<h1>` > `document.title`
- description: `description` > `og:description` > `twitter:description`
- author: `author` > `article:author` > `twitter:creator`
- published: `<time datetime>` > `article:published_time` > `datePublished`

**步骤 5: 正文选取**:
```javascript
document.querySelector(
  'main, article, [role="main"], .main-content, .post-content, .article-content, .content'
) || document.body;
```

#### HTML→Markdown 正则转换链 (html-to-markdown.ts:112-203)

不使用 DOM parser，纯正则替换：

```
br → \n
hr → ---
h1-h6 → # 到 ######
strong/b → **text**
em/i → *text*
del/s → ~~text~~
mark → ==text==
a[href] → [text](href)
img[src][alt] → ![alt](src)
pre>code → ```\ncode\n```
code → `code`
blockquote → > lines
ul>li → - item
ol>li → 1. item
table → | header | / | --- | / | row |
p → 段落
div → 段落
span → 内联
最后: stripTags → decodeHtmlEntities → normalizeWhitespace
```

优点: 零依赖，速度快。缺点: 嵌套标签处理不佳，如 `<strong><a href>text</a></strong>` 可能丢失格式。

### 2.3 两种捕获模式

| 模式 | 触发 | 适用场景 |
|------|------|----------|
| Auto（默认） | 页面加载 + Network Idle + 滚动 | 公开页面、静态内容 |
| Wait (`--wait`) | 用户按 Enter 信号 | 需登录页面、付费墙、复杂 SPA |

### 2.4 输出

```
url-to-markdown/{domain}/{slug}.md

---
url: https://example.com/page
title: "Page Title"
description: "Meta description"
author: "Author Name"
published: "2026-02-06T..."
captured_at: "2026-02-06T..."
---

# Page Title

Content...
```

slug 生成: title → lowercase → 去特殊字符 → kebab-case → 截断 50 字符。冲突则追加时间戳。

---

## 三、baoyu-format-markdown

### 3.1 架构

```
main.ts              CLI + Remark AST 管道
├── quotes.ts        全角引号替换 (5 行)
└── autocorrect.ts   外部工具桥接 (10 行)

Node 依赖:
├── unified + remark-parse + remark-stringify    AST 核心
├── remark-gfm              GFM 表格/任务列表支持
├── remark-frontmatter       YAML 前置块节点
├── remark-cjk-friendly      CJK 环境强调符号修复
├── unist-util-visit         AST 遍历/修改
└── yaml                     YAML 格式化 (lineWidth: 0)
```

### 3.2 核心机制

#### Remark AST 处理管道 (main.ts:49-85)

```
输入 Markdown
  ↓ remarkParse
Abstract Syntax Tree
  ↓ remarkCjkFriendly (可选)    ← CJK 强调符号修复
  ↓ remarkGfm                   ← GFM 表格/任务列表
  ↓ remarkFrontmatter            ← YAML 节点支持
  ↓ visit(tree, node => ...)     ← 自定义 AST 变换
  ↓ remarkStringify              ← AST → Markdown
输出 Markdown
  ↓ decodeHtmlEntities           ← 还原 CJK 实体
  ↓ applyAutocorrect (可选)      ← CJK/英文间距
最终输出
```

`parse` 和 `stringify` 分开使用（而非 `process()`），允许在中间插入 `visit()` 遍历修改 AST。

#### 有选择性的 AST 变换 (main.ts:64-78)

```typescript
visit(tree, (node) => {
  if (node.type === "text" && options.quotes) {
    // 只在 text 节点上替换引号 — code 节点不受影响
    textNode.value = replaceQuotes(textNode.value);
  }
  if (node.type === "yaml") {
    // YAML frontmatter 格式化
    const doc = YAML.parseDocument(yamlNode.value);
    yamlNode.value = doc.toString({ lineWidth: 0 }).trimEnd();
  }
});
```

这比对整个字符串做正则安全 — 代码块中的引号不会被误替换。

#### 引号替换 (quotes.ts)

```typescript
content
  .replace(/"([^"]+)"/g, "\u201c$1\u201d")     // "text" → "text"
  .replace(/「([^」]+)」/g, "\u201c$1\u201d")   // 「text」→ "text"
```

简洁但有局限：不处理跨行引号、嵌套引号、代码块内的引号（后者由 AST visit 保护）。

#### CJK 间距 (autocorrect.ts)

```typescript
execSync(`npx autocorrect-node --fix "${filePath}"`, { stdio: "inherit" });
```

直接 shell out 到 `autocorrect-node`（Rust 编写的 CJK/英文混排间距工具）。规则覆盖：
- 中英文之间加空格: `你好World` → `你好 World`
- 中文与数字之间加空格: `第3章` → `第 3 章`
- 全角标点修正

### 3.3 处理选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `--quotes` / `-q` | false | ASCII 引号 → 全角引号 |
| `--spacing` / `-s` | true | CJK/英文间距修正 |
| `--emphasis` / `-e` | true | CJK 强调符号修复 |

所有选项可独立开关，可组合使用。

### 3.4 SKILL.md 的 LLM 工作流设计

format-markdown 的 SKILL.md 定义了一个 8 步工作流，其中步骤 1-6 由 Claude 执行（内容分析、标题生成、frontmatter 创建、格式排版），步骤 7 才调用脚本做排版修正。这是一个 **LLM + 工具混合流水线** 的典型设计：

```
步骤 1: Claude 读取源文件
步骤 1.5: Claude 检测内容类型 (纯文本 vs Markdown)
步骤 2: Claude 分析结构
步骤 3: Claude 创建/检查 frontmatter
步骤 4: Claude 处理标题 (生成 3 候选 → AskUserQuestion)
步骤 5: Claude 添加 Markdown 格式 (**bold**, lists, code)
步骤 6: Claude 保存文件
步骤 7: 脚本执行排版修正 (CJK 间距, 引号, emphasis)
步骤 8: Claude 展示结果
```

---

## 四、共性设计模式

### 4.1 SKILL.md 即 LLM 接口契约

三个 skill 都用 SKILL.md 定义 Claude 应如何使用它们。这不是给人看的文档，而是 LLM 的指令协议，包含：
- 精确的命令格式和参数
- 同意/确认流程
- EXTEND.md 配置优先级（项目级 > 用户级）
- 输出格式规范

### 4.2 跨平台路径抽象

所有 skill 共享 XDG 规范的路径解析：

```typescript
function resolveUserDataRoot(): string {
  if (process.platform === "win32")  return APPDATA || ~/AppData/Roaming
  if (process.platform === "darwin") return ~/Library/Application Support
  return XDG_DATA_HOME || ~/.local/share
}
```

每个 skill 还支持环境变量覆盖路径。

### 4.3 Chrome 进程生命周期管理

x-to-markdown 和 url-to-markdown 共享相同模式：

```
spawn(chrome, args, { stdio: "ignore" })
  ↓ 使用
finally:
  Browser.close(CDP)           ← 优雅关闭
  chrome.kill("SIGTERM")       ← 信号终止
  setTimeout(2s):
    chrome.kill("SIGKILL")     ← 强制终止
    .unref()                   ← 不阻塞 Node 退出
```

### 4.4 运行时策略

全部使用 `npx -y bun` 运行 TypeScript — 利用 Bun 的原生 TS 支持，避免编译步骤和 node_modules 膨胀。

---

## 五、对 markitai 的借鉴价值

### 5.1 可直接借鉴

| 技术 | 来源 | 应用场景 | 优先级 |
|------|------|----------|--------|
| 自动滚动加载 | url-to-markdown | Playwright 渲染时触发懒加载图片/内容 | 高 |
| DOM 清洗选择器集 | url-to-markdown | URL 内容提取的预处理层 | 高 |
| `pending ≤ N` Network Idle | url-to-markdown | 替代 `domcontentloaded`/`networkidle` 的中间策略 | 中 |
| CJK/英文自动间距 | format-markdown | 中文文档输出后处理 | 中 |
| 全角引号替换 | format-markdown | 中文排版优化 | 低 |

### 5.2 设计思路参考

| 思路 | 来源 | 说明 |
|------|------|------|
| 动态 API 发现 + 硬编码回退 | x-to-markdown | 适应上游频繁变化的 API，提高韧性 |
| 三层认证优先级 | x-to-markdown | env 变量 > 文件缓存 > 交互式获取 |
| LLM + 工具混合管道 | format-markdown | 让 LLM 做语义分析，工具做确定性变换 |
| AST 级别的选择性变换 | format-markdown | 比全文正则替换安全，不影响代码块 |

### 5.3 不建议借鉴

| 技术 | 理由 |
|------|------|
| X GraphQL 逆向工程 | 维护成本极高，X 频繁改动 API |
| 纯正则 HTML→MD 转换 | markitai 已用 markitdown 的 DOM-based 方案，更可靠 |
| 直接 CDP 替代 Playwright | markitai 已有 Playwright 集成，CDP 是更底层的重复实现 |
| `npx -y bun` 运行时 | markitai 是 Python 项目 |

### 5.4 具体实施建议

**短期 (可立即实施):**

1. **Playwright 自动滚动** — 在 `fetch_playwright.py` 渲染后加入滚动循环：
   ```python
   for _ in range(8):
       old_height = await page.evaluate("document.body.scrollHeight")
       await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
       await page.wait_for_timeout(600)
       new_height = await page.evaluate("document.body.scrollHeight")
       if new_height == old_height:
           break
   await page.evaluate("window.scrollTo(0, 0)")
   ```

2. **DOM 噪音清洗** — 在 Playwright 提取 HTML 前注入清洗脚本，移除 `nav`, `footer`, `.ad`, `.cookie-banner`, `.popup` 等元素。

**中期 (需要评估):**

3. **CJK 间距后处理** — 可在 LLM 增强管道的最后阶段加入 `autocorrect`（有 Python 版本 `autocorrect-py`），不依赖 LLM 也能提升中文文档排版质量。

4. **Network Idle 自定义策略** — 当前 `wait_for` 支持 Playwright 内置的 `domcontentloaded`/`load`/`networkidle`/`commit`。可考虑自定义 `pending ≤ 2` 策略作为第五选项。

---

## 六、WSL 兼容性说明

在 WSL 环境实测中发现：
- Chrome CDP 无法跨越 WSL/Windows 边界 — Windows Chrome 的调试端口在 WSL 的 localhost 上不可达
- `X_CHROME_PATH` 指向 `/mnt/c/.../chrome.exe` 可以启动 Chrome，但 `/json/version` 端点不可访问
- markitai 的 Playwright（安装了 Chromium for Linux）在 WSL 中正常工作
- 如需在 WSL 中使用 x-to-markdown，只能通过 `X_AUTH_TOKEN` + `X_CT0` 环境变量提供认证
