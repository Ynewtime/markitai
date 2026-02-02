# OpenClaw 代码库深度架构分析报告

**调研日期**: 2026年1月31日  
**目标读者**: 架构师、应用开发者  
**用途**: 构建类似系统的技术选型参考

---

## 1. 项目概述与定位

OpenClaw（曾用名 Clawdbot、Moltbot）是一个**自托管的开源 AI 个人助手平台**，其核心差异化在于：

- **本地优先架构**: 数据完全存储在用户设备上
- **消息平台集成**: 统一接入 13+ 即时通讯平台
- **真正的代理能力**: 不仅对话，还能执行 shell 命令、控制浏览器、操作文件系统
- **主动交互**: 支持定时任务、事件触发，可主动联系用户

**技术指标**:
- GitHub Stars: 117,000+
- 提交数: 8,300+
- 贡献者: 150+
- 开源协议: MIT

---

## 2. 整体架构设计

### 2.1 高层架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        消息平台层 (Messaging Layer)                       │
├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────────┤
│WhatsApp │Telegram │ Discord │  Slack  │ Signal  │iMessage │ WebChat/... │
│(Baileys)│ (grammY)│(discord │ (Bolt)  │(signal- │ (imsg)  │             │
│         │         │   .js)  │         │  cli)   │         │             │
└────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴──────┬──────┘
     │         │         │         │         │         │           │
     └─────────┴─────────┴────┬────┴─────────┴─────────┴───────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Gateway (控制平面核心)                                │
│                    ws://127.0.0.1:18789                                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ • WebSocket Server (JSON-RPC 风格协议)                            │  │
│  │ • Session Manager (会话管理 + 消息路由)                            │  │
│  │ • Channel Router (多渠道消息分发)                                  │  │
│  │ • Tool Registry (工具注册与调用)                                   │  │
│  │ • Cron Scheduler (定时任务)                                       │  │
│  │ • Presence Manager (在线状态)                                     │  │
│  │ • Config Validator (TypeBox Schema 校验)                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Pi Agent      │    │   CLI Client    │    │   Node Clients  │
│   (AI Runtime)  │    │ (openclaw ...)  │    │ (macOS/iOS/     │
│                 │    │                 │    │  Android)       │
│ • RPC 模式      │    │ • 命令行交互     │    │ • Canvas 渲染   │
│ • Tool 流式调用 │    │ • 脚本自动化     │    │ • 摄像头/屏幕   │
│ • 多模型支持    │    │ • 健康检查       │    │ • 语音唤醒      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Tool Layer (工具层)                             │
├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────────┤
│  exec   │ browser │  read/  │ canvas  │  cron   │sessions │  nodes.*    │
│ (shell) │  (CDP)  │  write  │ (A2UI)  │(定时)   │(多会话) │ (设备控制)  │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────────┘
```

### 2.2 核心设计原则

| 原则 | 实现方式 |
|------|----------|
| **单一控制平面** | 一个 Gateway 进程管理所有消息渠道和客户端连接 |
| **协议优先** | TypeBox 定义 Schema → JSON Schema → Swift 模型自动生成 |
| **会话隔离** | 主会话 (main) 与群组会话分离，支持沙箱模式 |
| **本地信任** | 本地回环连接可自动批准，远程连接需配对确认 |
| **工具可控** | 工具白名单/黑名单机制，支持按会话粒度控制 |

---

## 3. 技术栈详解

### 3.1 运行时与构建工具

| 类别 | 技术选型 | 说明 |
|------|----------|------|
| **运行时** | Node.js ≥22 | 必须，WhatsApp/Telegram 等依赖 |
| **语言** | TypeScript | 严格类型，tsx 直接运行 |
| **包管理** | pnpm (推荐) | 支持 npm/bun，pnpm 用于开发构建 |
| **构建** | tsc + 自定义脚本 | 产出 dist/ 用于打包发布 |
| **测试** | Vitest | V8 覆盖率，70% 阈值 |
| **Lint/Format** | oxlint + oxfmt + Prettier | 快速的 Rust 工具链 |
| **Schema** | TypeBox | 类型安全的 JSON Schema 生成器 |

### 3.2 Monorepo 结构

```
openclaw/
├── src/                    # 核心源码
│   ├── gateway/           # Gateway WebSocket 服务器
│   ├── agent/             # Pi Agent 运行时
│   ├── telegram/          # Telegram 渠道 (grammY)
│   ├── discord/           # Discord 渠道 (discord.js)
│   ├── slack/             # Slack 渠道 (Bolt)
│   ├── signal/            # Signal 渠道 (signal-cli)
│   ├── imessage/          # iMessage 渠道 (imsg CLI)
│   ├── web/               # WhatsApp Web (Baileys)
│   ├── channels/          # 渠道抽象层
│   ├── routing/           # 消息路由逻辑
│   ├── tools/             # 内置工具实现
│   ├── sessions/          # 会话管理
│   └── config/            # 配置加载与校验
│
├── extensions/            # 扩展渠道 (插件形式)
│   ├── msteams/          # Microsoft Teams
│   ├── matrix/           # Matrix
│   ├── zalo/             # Zalo
│   ├── googlechat/       # Google Chat
│   ├── voice-call/       # 语音通话
│   └── memory-core/      # 记忆系统
│
├── apps/                  # 原生应用
│   └── macos/            # macOS Swift 应用
│       └── Sources/
│           └── MoltbotProtocol/  # 自动生成的协议模型
│
├── ui/                    # Web UI 源码
│   └── control-ui/       # Control UI (React/Vue?)
│
├── skills/                # 内置技能
│   └── */SKILL.md        # 技能定义文件
│
├── vendor/                # 第三方依赖
│   └── a2ui/             # A2UI Canvas 库
│
├── scripts/               # 构建/工具脚本
│   ├── protocol-gen.ts   # 协议 Schema 生成
│   ├── protocol-gen-swift.ts  # Swift 模型生成
│   └── bundle-a2ui.sh    # A2UI 打包
│
├── test/                  # 测试文件
├── docs/                  # 文档
├── patches/               # 依赖补丁
│
├── package.json           # 根配置
├── pnpm-workspace.yaml    # pnpm 工作区
├── tsconfig.json          # TypeScript 配置
├── vitest.config.ts       # 测试配置
├── vitest.e2e.config.ts   # E2E 测试
├── vitest.gateway.config.ts
├── vitest.live.config.ts
│
├── Dockerfile             # 标准 Docker
├── Dockerfile.sandbox     # 沙箱 Docker
├── docker-compose.yml
├── fly.toml               # Fly.io 部署
└── render.yaml            # Render 部署
```

### 3.3 关键依赖分析

**消息平台 SDK**:

| 平台 | 依赖库 | 特点 |
|------|--------|------|
| WhatsApp | `@whiskeysockets/baileys` | Web 协议逆向，需 QR 配对 |
| Telegram | `grammy` | Bot API，轻量高效 |
| Discord | `discord.js` | 功能完整，支持斜杠命令 |
| Slack | `@slack/bolt` | 官方 SDK，Socket Mode |
| Signal | `signal-cli` | CLI 工具封装 |
| iMessage | `imsg` | macOS 专用 CLI |

**核心依赖**:

```json
{
  "dependencies": {
    "@sinclair/typebox": "^0.x",      // Schema 定义
    "ws": "^8.x",                      // WebSocket 服务
    "puppeteer-core": "^x.x",          // 浏览器控制 (CDP)
    "zod": "^3.x"                      // 运行时校验 (部分)
  },
  "devDependencies": {
    "typescript": "^5.x",
    "tsx": "^4.x",                     // TS 直接执行
    "vitest": "^x.x",
    "oxlint": "^x.x"
  }
}
```

---

## 4. Gateway 核心实现

### 4.1 WebSocket 协议设计

Gateway 采用**类 JSON-RPC 的自定义协议**，三种帧类型：

```typescript
// 请求帧
interface RequestFrame {
  type: "req";
  id: string;           // 请求 ID
  method: string;       // 方法名
  params?: object;      // 参数
  idempotencyKey?: string;  // 幂等键 (send/agent 必填)
}

// 响应帧
interface ResponseFrame {
  type: "res";
  id: string;           // 对应请求 ID
  ok: boolean;
  payload?: object;     // 成功时
  error?: object;       // 失败时
}

// 事件帧 (服务端推送)
interface EventFrame {
  type: "event";
  event: string;        // 事件名
  payload: object;
  seq?: number;         // 序列号
  stateVersion?: number;
}
```

**连接生命周期**:

```
Client                         Gateway
   |                              |
   |------ req:connect ---------->|  首帧必须是 connect
   |<----- res (hello-ok) --------|  携带 presence + health 快照
   |                              |
   |<----- event:presence --------|  状态变更推送
   |<----- event:tick ------------|  心跳
   |                              |
   |------ req:agent ------------>|  发起 AI 调用
   |<----- res (accepted) --------|  ACK: {runId, status}
   |<----- event:agent -----------|  流式输出
   |<----- event:agent -----------|  ...
   |<----- res (final) -----------|  完成: {runId, status, summary}
```

**认证机制**:

1. **Token 认证**: `OPENCLAW_GATEWAY_TOKEN` 环境变量或 `--token` 参数
2. **设备配对**: 新设备首次连接需通过 Admin UI 批准
3. **本地信任**: 回环地址连接可配置自动批准
4. **Tailscale 集成**: 支持 Tailscale 身份头认证

### 4.2 会话管理

```typescript
// 会话模型
interface Session {
  id: string;
  mainKey: string;          // "main" 表示主会话
  agentId: string;          // 绑定的 Agent
  channel: string;          // 来源渠道
  peerId: string;           // 对话方 ID
  messages: Message[];      // 历史消息
  thinkingLevel: ThinkingLevel;
  model: string;
  sandbox?: SandboxConfig;  // 沙箱配置
}

// 会话路由规则
// 1. 直接对话 (DM) → 合并到 "main" 会话 (可配置)
// 2. 群组消息 → 每群组独立会话
// 3. 多 Agent → 按绑定路由到不同 Agent
```

**会话持久化**:
- 位置: `~/.openclaw/agents/<agentId>/sessions/sessions.json`
- 策略: 内存 + 定期持久化
- 裁剪: 支持上下文压缩 (compaction)

### 4.3 消息路由引擎

```typescript
// 路由流程
inboundMessage
  │
  ├─ 1. 渠道解析 (WhatsApp/Telegram/...)
  │
  ├─ 2. 发送者鉴权
  │     ├─ allowFrom 白名单检查
  │     └─ dmPolicy: "pairing" | "open"
  │
  ├─ 3. Agent 路由
  │     └─ agents.list[].bindings 匹配
  │
  ├─ 4. 会话定位/创建
  │     ├─ DM → main session
  │     └─ Group → group:<groupId>
  │
  ├─ 5. 消息入队
  │     └─ queue.mode: "fifo" | "latest"
  │
  └─ 6. Agent 调用
        └─ RPC → Pi Agent
```

### 4.4 配置校验与热重载

**严格配置校验**:

OpenClaw 采用**严格拒绝未知键**的策略，配置验证失败时 Gateway 拒绝启动：

```typescript
// 配置校验流程
validateConfig(rawJson5: string) {
  const parsed = JSON5.parse(rawJson5);
  const schema = Type.Strict(OpenClawConfigSchema);  // TypeBox 严格模式
  
  if (!TypeBoxValidate(schema, parsed)) {
    throw new ConfigValidationError(errors);
    // Gateway 拒绝启动，只允许诊断命令
    // openclaw doctor, openclaw logs, openclaw health
  }
}
```

**配置热重载**:

| 操作 | RPC 方法 | 行为 |
|------|----------|------|
| **全量替换** | `config.apply` | 校验 → 写入 → 重启 Gateway |
| **增量更新** | `config.patch` | JSON Merge Patch 语义合并 |
| **UI 编辑** | Control UI | Schema 驱动表单 + Raw JSON 编辑器 |

```typescript
// config.apply 参数
{
  raw: string;           // JSON5 配置内容
  baseHash?: string;     // 乐观锁：配置哈希
  sessionKey?: string;   // 重启后唤醒的会话
  restartDelayMs?: number; // 重启延迟 (默认 2000ms)
}
```

**配置文件包含 ($include)**:

支持拆分大型配置到多个文件：

```json5
// ~/.openclaw/openclaw.json
{
  gateway: { port: 18789 },
  agents: { "$include": "./agents.json5" },  // 单文件包含
  broadcast: { 
    "$include": [                             // 多文件深度合并
      "./clients/mueller.json5",
      "./clients/schmidt.json5"
    ]
  }
}
```

---

## 5. Agent 运行时 (Pi Agent)

### 5.1 Agent Loop 设计

```typescript
async function agentLoop(session: Session, message: Message) {
  // 1. 构建系统提示词
  const systemPrompt = await buildSystemPrompt(session);
  
  // 2. 上下文组装
  const context = await buildContext(session, message);
  
  // 3. 模型调用 (支持流式)
  const stream = await callModel({
    model: session.model,
    messages: context,
    tools: getAvailableTools(session),
    thinking: session.thinkingLevel,
  });
  
  // 4. 流式处理
  for await (const block of stream) {
    if (block.type === 'text') {
      emit('agent', { type: 'text', content: block.content });
    } else if (block.type === 'tool_use') {
      // 5. 工具调用
      const result = await executeTool(block.tool, block.input, session);
      // 6. 工具结果反馈给模型
      context.push({ role: 'tool', content: result });
      // 递归继续
    }
  }
  
  // 7. 响应发送回渠道
  await sendToChannel(session.channel, response);
}
```

### 5.2 系统提示词组成

系统提示词由 OpenClaw 自行组装（不使用 p-coding-agent 默认提示词），包含以下固定段落：

**系统提示词结构**:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Tooling          - 当前工具列表 + 简短描述               │
│  2. Skills           - 可用技能列表 (元数据，指令按需读取)   │
│  3. Self-Update      - 如何运行 config.apply / update.run   │
│  4. Workspace        - 工作目录路径                         │
│  5. Documentation    - 本地文档路径 + 何时参考文档           │
│  6. Sandbox          - 沙箱状态 + 可用的 elevated 选项      │
│  7. Current Date     - 用户时区时间 + 时间格式              │
│  8. Reply Tags       - 平台特定回复标签语法                 │
│  9. Runtime          - 版本 + 仓库根目录                    │
├─────────────────────────────────────────────────────────────┤
│  10. Project Context - 注入的工作区文件 (Bootstrap Files)    │
│      ├── AGENTS.md      行为指南                            │
│      ├── SOUL.md        人格/语气/边界                       │
│      ├── TOOLS.md       工具使用笔记                        │
│      ├── IDENTITY.md    身份名称/emoji                      │
│      ├── USER.md        用户档案                            │
│      ├── HEARTBEAT.md   心跳任务清单                        │
│      └── BOOTSTRAP.md   首次运行仪式 (完成后删除)           │
└─────────────────────────────────────────────────────────────┘
```

**Bootstrap 文件处理**:

```typescript
// 注入策略
interface BootstrapInjection {
  maxCharsPerFile: 20000;           // agents.defaults.bootstrapMaxChars
  truncationMarker: "[TRUNCATED]";  // 超长文件截断标记
  missingMarker: "[FILE MISSING]";  // 缺失文件提示
  loadOrder: ["AGENTS", "SOUL", "TOOLS", "IDENTITY", "USER", "HEARTBEAT", "BOOTSTRAP"];
}

// Prompt 模式
type PromptMode = 
  | "full"     // 完整提示词 (默认)
  | "minimal"  // 子 Agent 用，省略 Skills/Memory/Messaging 等
  | "none";    // 仅返回基础身份行
```

**工作区目录结构**:

```
~/.openclaw/workspace/           # agents.defaults.workspace
├── AGENTS.md                    # Agent 行为指南
├── SOUL.md                      # 人格/语气设定
├── TOOLS.md                     # 工具使用说明
├── IDENTITY.md                  # 身份信息
├── USER.md                      # 用户偏好
├── BOOTSTRAP.md                 # 启动说明 (首次运行)
├── HEARTBEAT.md                 # 心跳任务 (可选)
├── BOOT.md                      # 启动检查清单 (可选)
│
├── memory/                      # 持久记忆
│   ├── MEMORY.md               # 长期记忆精华 (主会话专用)
│   ├── 2026-01-31.md           # 每日日志
│   └── topics/                 # 主题记忆
│
├── skills/                      # 工作区技能 (最高优先级)
│   └── custom-skill/
│       └── SKILL.md
│
└── canvas/                      # A2UI 可视化文件
```

**记忆系统**:

| 文件 | 作用域 | 访问规则 |
|------|--------|----------|
| `MEMORY.md` | 仅主会话 | 安全考虑：不向陌生人泄露 |
| `memory/YYYY-MM-DD.md` | 所有会话 | 每日日志，建议读取今天+昨天 |
| `memory/topics/*.md` | 所有会话 | 主题索引记忆 |

### 5.3 Agent Loop 详细流程

```typescript
// 完整 Agent Loop
async function agentLoop(session: Session, message: Message) {
  // 1. 入口验证
  const { runId, acceptedAt } = await validateAndAccept(session, message);
  emit('lifecycle', { phase: 'accepted', runId });
  
  // 2. 会话锁 + 队列管理
  await acquireSessionLane(session.key);  // 串行化同一会话
  
  // 3. 工作区准备
  const workspace = resolveWorkspace(session);
  if (session.sandbox.mode !== 'off') {
    workspace = createSandboxWorkspace(session);
  }
  
  // 4. 技能快照
  const skills = snapshotEligibleSkills(session.agentId);
  injectSkillsEnv(skills);  // 注入环境变量
  
  // 5. 系统提示词构建
  const systemPrompt = await buildSystemPrompt({
    mode: session.promptMode || 'full',
    workspace,
    skills,
    sandbox: session.sandbox,
    bootstrapFiles: await loadBootstrapFiles(workspace),
  });
  
  // 6. 上下文组装
  const context = await buildContext(session, message, {
    historyLimit: session.historyLimit,
    pruneToolResults: session.contextPruning,
  });
  
  // 7. 模型调用 (流式)
  const stream = await callModel({
    model: session.model,
    systemPrompt,
    messages: context,
    tools: getAvailableTools(session),
    thinking: session.thinkingLevel,
  });
  
  // 8. 流式处理
  for await (const block of stream) {
    switch (block.type) {
      case 'text':
        emit('agent', { type: 'assistant', content: block.content });
        break;
        
      case 'tool_use':
        emit('agent', { type: 'tool_start', tool: block.tool });
        
        // 9. 工具执行
        const result = await executeTool(block.tool, block.input, {
          session,
          sandbox: session.sandbox,
          elevated: session.elevatedMode,
        });
        
        emit('agent', { type: 'tool_end', tool: block.tool, result });
        
        // 10. 检查消息队列 (steer 模式)
        if (session.queueMode === 'steer' && hasQueuedMessage(session)) {
          skipRemainingToolCalls();
          injectQueuedMessage();
        }
        break;
    }
  }
  
  // 11. 回复整形
  const payloads = shapeReply(stream.output, {
    suppressDuplicates: true,    // 去重消息工具发送
    filterNoReply: true,         // 过滤 NO_REPLY 标记
  });
  
  // 12. 自动压缩检测
  if (shouldCompact(session)) {
    await performCompaction(session);
    emit('lifecycle', { phase: 'compaction' });
  }
  
  // 13. 发送响应
  await sendToChannel(session.channel, payloads);
  
  // 14. 持久化
  await persistSession(session);
  emit('lifecycle', { phase: 'complete', runId });
}
```

**队列模式**:

| 模式 | 行为 |
|------|------|
| `collect` | 等待当前 turn 结束，新消息合并到下一 turn |
| `steer` | 中断当前工具调用，注入新消息 |
| `followup` | 等待结束后启动新 turn |
| `interrupt` | 直接中断当前 run |

### 5.4 工具系统

**内置工具分类**:

| 工具名 | 功能 | 安全等级 | 沙箱行为 |
|--------|------|----------|----------|
| `exec` | Shell 命令执行 | 🔴 高危 | 容器内执行 (默认) |
| `process` | 进程管理 | 🔴 高危 | 容器内执行 |
| `write` | 文件写入 | 🔴 高危 | 沙箱工作区 |
| `edit` | 文件编辑 | 🔴 高危 | 沙箱工作区 |
| `apply_patch` | 补丁应用 | 🔴 高危 | 可选工具 |
| `browser` | CDP 浏览器控制 | 🔴 高危 | 可配置沙箱浏览器 |
| `read` | 文件读取 | 🟡 中等 | 沙箱工作区根目录 |
| `web_fetch` | 网页获取 | 🟡 中等 | - |
| `cron` | 定时任务 | 🟡 中等 | - |
| `nodes` | 设备控制 | 🟡 中等 | 节点需单独批准 |
| `canvas` | A2UI 可视化 | 🟢 低危 | - |
| `sessions_*` | 多会话协作 | 🟢 低危 | - |
| `web_search` | 网页搜索 | 🟢 低危 | - |
| `image` | 图像生成 | 🟢 低危 | - |

**工具策略层级**:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: tools.elevated (全局提升白名单)                    │
│           - 允许沙箱内工具在宿主机执行                        │
│           - 需要发送者在 tools.elevated.allowFrom 白名单     │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: agents.list[].tools (Agent 级别策略)               │
│           - allow: 允许的工具列表                            │
│           - deny: 拒绝的工具列表 (deny 优先)                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: tools.sandbox.tools (沙箱内工具策略)               │
│           - 限制沙箱内可用的工具                             │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: exec.approvals (执行批准)                         │
│           - 高危命令需要用户交互批准                         │
└─────────────────────────────────────────────────────────────┘
```

**工具组快捷方式**:

```typescript
// 配置支持 group:* 展开
const toolGroups = {
  "group:filesystem": ["read", "write", "edit", "apply_patch"],
  "group:sessions": ["sessions_list", "sessions_history", "sessions_send", "sessions_spawn", "session_status"],
  "group:messaging": ["whatsapp", "telegram", "slack", "discord"],
};

// 示例配置
{
  tools: {
    allow: ["group:filesystem", "exec"],
    deny: ["browser"]
  }
}
```

### 5.5 技能系统 (Skills)

技能是 **AgentSkills 规范** 兼容的目录，包含 `SKILL.md` 及相关文件。

**技能加载优先级**:

```
1. 工作区技能: <workspace>/skills/          (最高优先级)
2. 本地技能:   ~/.openclaw/skills/          (用户自定义)
3. 内置技能:   <package>/skills/            (捆绑发布)
4. 扩展目录:   skills.load.extraDirs        (配置指定)
```

**SKILL.md 格式**:

```yaml
---
name: nano-banana-pro
description: Generate or edit images via Gemini 3 Pro Image
metadata: {"openclaw":{"requires":{"bins":["uv"],"env":["GEMINI_API_KEY"]},"primaryEnv":"GEMINI_API_KEY"}}
user-invocable: true
disable-model-invocation: false
---

# 技能指令

使用 `{baseDir}` 引用技能目录路径...
```

**技能门控 (Gating)**:

| 字段 | 作用 |
|------|------|
| `requires.bins` | 需要的二进制 (加载时检查) |
| `requires.env` | 需要的环境变量 |
| `requires.config` | 需要的配置键 |
| `os` | 平台限制 (darwin/linux/win32) |

**技能 Token 影响**:

```
基础开销 (≥1 技能时): ~195 字符
每技能开销: ~(4 + name.length + description.length) 字符

示例: 12 个技能 → ~2,184 字符 (~546 token)
```

**技能快照机制**:

技能在会话首次启动时快照，后续 turn 复用。文件监控可选 (skills.load.watch)，检测 SKILL.md 变更后刷新快照。

**工具权限控制**:

```json
{
  "agents": {
    "list": [{
      "id": "public-agent",
      "tools": {
        "allow": ["read", "sessions_list", "sessions_send"],
        "deny": ["exec", "browser", "write", "edit"]
      }
    }]
  }
}
```

---

## 6. 渠道集成架构

### 6.1 渠道抽象层

```typescript
// 渠道接口定义
interface Channel {
  id: string;
  type: ChannelType;
  
  // 生命周期
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  
  // 消息收发
  onMessage(handler: MessageHandler): void;
  send(peerId: string, message: OutboundMessage): Promise<void>;
  
  // 状态
  getPresence(): PresenceInfo;
  setTyping(peerId: string, typing: boolean): void;
}

// 消息标准化
interface InboundMessage {
  id: string;
  channel: string;
  peerId: string;
  groupId?: string;
  content: MessageContent;
  timestamp: number;
  replyTo?: string;
  attachments?: Attachment[];
}
```

### 6.2 WhatsApp 集成 (Baileys)

```typescript
// 关键实现点
class WhatsAppChannel implements Channel {
  private sock: WASocket;
  
  async connect() {
    // 1. 加载/创建认证状态
    const { state, saveCreds } = await useMultiFileAuthState(
      '~/.openclaw/credentials/whatsapp'
    );
    
    // 2. 创建连接
    this.sock = makeWASocket({
      auth: state,
      printQRInTerminal: true,  // 首次需扫码
    });
    
    // 3. 监听事件
    this.sock.ev.on('messages.upsert', this.handleMessage);
    this.sock.ev.on('creds.update', saveCreds);
  }
  
  // ⚠️ 重要：每主机只能有一个 WhatsApp Web 会话
}
```

### 6.3 Telegram 集成 (grammY)

```typescript
class TelegramChannel implements Channel {
  private bot: Bot;
  
  async connect() {
    this.bot = new Bot(process.env.TELEGRAM_BOT_TOKEN);
    
    // 支持 Webhook 或长轮询
    if (this.config.webhookUrl) {
      await this.bot.api.setWebhook(this.config.webhookUrl);
    } else {
      this.bot.start();  // 长轮询
    }
    
    this.bot.on('message', this.handleMessage);
  }
  
  // 特性：支持草稿流式输出 (editMessageText)
}
```

### 6.4 扩展渠道机制

扩展位于 `extensions/` 目录，作为独立 npm 包：

```typescript
// extensions/msteams/index.ts
export default class MSTeamsPlugin implements ChannelPlugin {
  static id = 'msteams';
  static configSchema = MSTeamsConfigSchema;
  
  async activate(gateway: Gateway) {
    // 注册渠道
    gateway.registerChannel(new MSTeamsChannel(this.config));
  }
}
```

**插件加载流程**:
1. 扫描 `extensions/*/package.json`
2. `npm install --omit=dev` 安装依赖
3. 动态 import 并调用 `activate()`

---

## 7. 安全架构

### 7.1 沙箱模式详解

```typescript
// 完整沙箱配置
interface SandboxConfig {
  mode: "off" | "non-main" | "all";
  // off: 禁用沙箱，工具直接在宿主执行
  // non-main: 非主会话 (群组等) 启用沙箱
  // all: 所有会话都启用沙箱
  
  scope: "session" | "agent" | "shared";
  // session: 每会话独立容器
  // agent: 每 Agent 共享一个容器
  // shared: 全局共享容器
  
  workspaceAccess: "none" | "ro" | "rw";
  // none: 沙箱无法访问工作区
  // ro: 只读挂载工作区到 /agent
  // rw: 读写挂载工作区到 /workspace
  
  docker?: {
    image: string;              // 自定义镜像
    network: "none" | "host" | "bridge";
    env: Record<string, string>; // 注入环境变量
    setupCommand?: string;      // 容器创建后执行
    binds?: string[];           // 额外挂载
  };
  
  browser?: {
    autoStart: boolean;         // 自动启动沙箱浏览器
    allowHostControl: boolean;  // 允许控制宿主浏览器
  };
  
  prune?: {
    enabled: boolean;           // 定期清理容器
    maxAgeMs: number;
  };
}
```

**沙箱架构图**:

```
┌────────────────────────────────────────────────────────────────┐
│                    Gateway (宿主机)                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Session Manager                                          │  │
│  │  ├─ main session      → 沙箱: off (直接宿主执行)          │  │
│  │  ├─ group:xxx session → 沙箱: Docker Container A          │  │
│  │  └─ group:yyy session → 沙箱: Docker Container B          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│              ┌───────────────┴───────────────┐                 │
│              ▼                               ▼                  │
│  ┌─────────────────────┐         ┌─────────────────────┐       │
│  │  Sandbox Container  │         │  Sandbox Container  │       │
│  │  ┌───────────────┐  │         │  ┌───────────────┐  │       │
│  │  │ /workspace    │  │         │  │ /workspace    │  │       │
│  │  │ (隔离工作区)  │  │         │  │ (隔离工作区)  │  │       │
│  │  └───────────────┘  │         │  └───────────────┘  │       │
│  │  ┌───────────────┐  │         │  ┌───────────────┐  │       │
│  │  │ /agent (ro)   │  │         │  │ /agent (ro)   │  │       │
│  │  │ (原始工作区)  │  │         │  │ (原始工作区)  │  │       │
│  │  └───────────────┘  │         │  └───────────────┘  │       │
│  └─────────────────────┘         └─────────────────────┘       │
└────────────────────────────────────────────────────────────────┘
```

**沙箱内工具行为**:

| 工具 | 沙箱行为 |
|------|----------|
| `exec` | sh -lc 在容器内执行，env.PATH 已注入 |
| `read` | 根目录为沙箱工作区 |
| `write/edit` | 只在 workspaceAccess=rw 时可用 |
| `browser` | 可配置使用沙箱浏览器或宿主浏览器 |
| `process` | 只能管理容器内进程 |

### 7.2 Elevated 模式

当沙箱启用时，`/elevated` 指令允许**临时提升**到宿主执行：

```typescript
// Elevated 级别
type ElevatedMode = 
  | "off"   // 禁用，保持沙箱
  | "on"    // 允许宿主执行，需批准
  | "ask"   // 每次询问
  | "full"; // 允许宿主执行，跳过批准

// 使用方式
"/elevated on"        // 会话级别设置
"/elevated full"      // 消息级别内联
```

**Elevated 门控检查**:

1. `tools.elevated.enabled` 全局开关
2. `tools.elevated.allowFrom.<channel>` 发送者白名单
3. `agents.list[].tools.elevated` Agent 级别限制
4. 三层检查全部通过才允许提升

### 7.3 安全默认值

| 设置 | 默认值 | 说明 |
|------|--------|------|
| `gateway.bind` | `"loopback"` | 仅监听 127.0.0.1 |
| `dmPolicy` | `"pairing"` | 陌生人需配对码确认 |
| `groupPolicy` | `"allowlist"` | 群组需白名单 |
| `tools.exec.host` | `"sandbox"` | Shell 默认在沙箱执行 |
| `gateway.auth.mode` | `"token"` | 需要 Token 认证 |

### 7.4 多 Agent 路由与隔离

OpenClaw 支持在单个 Gateway 内运行**多个隔离的 Agent**，每个 Agent 有独立的：
- 工作区 (workspace)
- 会话存储 (agentDir)
- 沙箱配置
- 工具权限

**路由绑定**:

```json5
{
  agents: {
    list: [
      { id: "main", default: true, workspace: "~/.openclaw/workspace" },
      { id: "work", workspace: "~/.openclaw/workspace-work" },
      { id: "family", workspace: "~/.openclaw/workspace-family", 
        sandbox: { mode: "all", scope: "agent" },
        tools: { allow: ["read"], deny: ["exec", "browser"] }
      }
    ]
  },
  bindings: [
    // 精确匹配：特定群组 → family agent
    { agentId: "family", match: { channel: "whatsapp", peer: { kind: "group", id: "[email protected]" } } },
    // 账号匹配：工作微信账号 → work agent  
    { agentId: "work", match: { channel: "whatsapp", accountId: "biz" } },
    // 通配匹配：所有 Telegram 消息 → work agent
    { agentId: "work", match: { channel: "telegram", accountId: "*" } },
  ]
}
```

**绑定匹配优先级** (从高到低):

1. `match.peer` (特定 DM/群组)
2. `match.guildId` (Discord 服务器)
3. `match.teamId` (Slack 团队)
4. `match.accountId` (精确账号)
5. `match.accountId: "*"` (渠道通配)
6. 默认 Agent

### 7.5 已知安全风险

⚠️ **提示注入**: Agent 读取不可信内容（网页/邮件）时可能被诱导执行恶意指令

⚠️ **权限扩散**: 浏览器控制可访问已登录会话的所有数据

⚠️ **配置泄露**: `~/.openclaw/credentials/` 存储明文凭据

**缓解建议**:
1. 使用 `Anthropic Opus 4.5`（提示注入防御更强）
2. 浏览器使用独立 Profile
3. 沙箱模式启用
4. 最小权限工具白名单

---

## 8. 扩展机制

### 8.1 Hook 系统

OpenClaw 提供两种 Hook 扩展点：

**内部 Hook (Gateway 层)**:

| Hook | 触发时机 | 用途 |
|------|----------|------|
| `agent:bootstrap` | 系统提示词构建前 | 修改/替换 Bootstrap 文件 |
| `/new`, `/reset`, `/stop` | 命令执行时 | 命令生命周期 |

**插件 Hook (Agent 生命周期)**:

| Hook | 触发时机 | 用途 |
|------|----------|------|
| `before_agent_start` | Agent 运行开始前 | 注入上下文/覆盖系统提示词 |
| `agent_end` | Agent 运行结束后 | 检查最终消息/元数据 |
| `before_compaction` | 压缩开始前 | 观察/标注 |
| `after_compaction` | 压缩完成后 | 观察/标注 |
| `before_tool_call` | 工具调用前 | 拦截/修改参数 |
| `after_tool_call` | 工具调用后 | 拦截/修改结果 |
| `tool_result_persist` | 工具结果持久化前 | 同步转换结果 |
| `message_received` | 收到消息时 | 入站处理 |
| `message_sending` | 发送消息前 | 出站处理 |
| `message_sent` | 发送消息后 | 出站确认 |

### 8.2 Webhook 表面

Gateway 支持 HTTP Webhook 用于外部触发：

```typescript
// Webhook 配置
{
  hooks: {
    enabled: true,
    token: "webhook-secret-token",  // 认证
    endpoints: {
      // POST /hooks/trigger
      trigger: {
        enabled: true,
        allowedEvents: ["custom:*"]
      }
    }
  }
}
```

**用例**:
- Gmail Pub/Sub 触发
- CI/CD 通知
- 日历事件
- IoT 设备事件

### 8.3 Cron 定时任务

```json5
{
  cron: {
    jobs: [
      {
        id: "daily-summary",
        schedule: "0 9 * * *",        // Cron 表达式
        agentId: "main",              // 目标 Agent
        sessionKey: "agent:main:cron:daily",
        message: "Generate daily summary",
        enabled: true
      }
    ]
  }
}
```

### 8.4 Heartbeat 心跳

心跳是特殊的定时 Agent 调用，用于主动任务：

```json5
{
  heartbeat: {
    enabled: true,
    every: "55m",                     // 间隔 (字符串或毫秒)
    agentId: "main",
    prompt: "Read HEARTBEAT.md if it exists. Follow it strictly."
  }
}
```

**心跳 vs Cron**:

| 特性 | Heartbeat | Cron |
|------|-----------|------|
| 灵活度 | 固定间隔 | Cron 表达式 |
| 会话 | 主会话 | 任意会话 |
| 用途 | 保持缓存热/检查任务 | 定时任务 |
| Token 消耗 | 每次完整 Agent turn | 每次完整 Agent turn |

---

## 9. 部署架构

### 9.1 本地部署

```bash
# 最简安装
npm install -g openclaw@latest
openclaw onboard --install-daemon

# 从源码
git clone https://github.com/openclaw/openclaw.git
cd openclaw
pnpm install
pnpm ui:build
pnpm build
pnpm openclaw onboard --install-daemon
```

**进程管理**:
- macOS: `launchd` 用户服务
- Linux: `systemd` 用户服务
- 守护进程自动重启

### 9.2 上下文窗口与压缩

**上下文组成**:

```
┌────────────────────────────────────────────────────────────┐
│  System Prompt (OpenClaw 构建)                              │
│  ├── 工具列表 + 描述                                        │
│  ├── 技能列表 (元数据)                                      │
│  ├── 工作区 + Bootstrap 文件                               │
│  └── 时间/运行时/沙箱状态                                   │
├────────────────────────────────────────────────────────────┤
│  Conversation History                                       │
│  ├── 用户消息                                               │
│  └── Assistant 消息                                         │
├────────────────────────────────────────────────────────────┤
│  Tool Calls/Results                                         │
│  ├── 命令输出                                               │
│  ├── 文件读取内容                                           │
│  └── 图像/附件 (base64)                                     │
└────────────────────────────────────────────────────────────┘
```

**查看上下文使用**:

```bash
/status          # 快速查看窗口使用率
/context list    # 注入内容 + 大致大小
/context detail  # 详细分解 (每文件/每工具/每技能)
/usage tokens    # 每回复追加 Token 使用
```

**自动压缩 (Compaction)**:

当上下文接近窗口限制时，OpenClaw 自动压缩历史：

```typescript
// 压缩配置
{
  agents: {
    defaults: {
      compaction: {
        mode: "safeguard",                    // off | safeguard | aggressive
        reserveTokensFloor: 24000,            // 保留窗口空间
        memoryFlush: {
          enabled: true,
          softThresholdTokens: 6000,          // 触发阈值
          systemPrompt: "Session nearing compaction...",
          prompt: "Write any lasting notes to memory/YYYY-MM-DD.md"
        }
      }
    }
  }
}
```

**压缩流程**:

1. 检测到上下文接近限制
2. 触发 `memoryFlush` 让 Agent 保存重要信息
3. 生成历史摘要
4. 替换详细历史为压缩摘要
5. 发出 `compaction` 事件
6. 重试当前请求

### 9.3 Docker 部署

```yaml
# docker-compose.yml 核心配置
services:
  gateway:
    build: .
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./data:/root/.openclaw
      - ./workspace:/root/clawd
    ports:
      - "18789:18789"
```

### 9.4 Cloudflare Workers 部署 (Moltworker)

```
┌─────────────────────────────────────────────────┐
│           Cloudflare Worker                     │
│  ┌─────────────────────────────────────────┐   │
│  │   WebSocket 代理 + 认证                  │   │
│  └──────────────────┬──────────────────────┘   │
│                     │                           │
│  ┌──────────────────▼──────────────────────┐   │
│  │   Cloudflare Sandbox (容器)              │   │
│  │   └─ OpenClaw Gateway                   │   │
│  │   └─ Node.js Runtime                    │   │
│  │   └─ 浏览器自动化 (Browser Rendering)   │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │   R2 Storage (可选持久化)                │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

**成本**: Workers Paid ($5/月) + API 调用费

### 9.5 远程访问方案

| 方案 | 安全性 | 复杂度 | 适用场景 |
|------|--------|--------|----------|
| **Tailscale Serve** | 高 | 低 | 个人 Tailnet |
| **Tailscale Funnel** | 中 | 低 | 公开访问（需密码） |
| **SSH Tunnel** | 高 | 中 | 临时访问 |
| **Reverse Proxy** | 中 | 高 | 生产部署 |

---

## 10. 技术选型建议

### 10.1 如果你要构建类似系统

**推荐采用的设计**:

| 模块 | OpenClaw 方案 | 建议 |
|------|---------------|------|
| **协议层** | 自定义 WS + TypeBox | ✅ 采用，类型安全 + 跨平台代码生成 |
| **单一控制平面** | Gateway 模式 | ✅ 采用，简化状态管理 |
| **渠道抽象** | 插件化 Channel | ✅ 采用，易扩展 |
| **工具系统** | 白名单 + 沙箱 | ✅ 采用，安全优先 |
| **会话模型** | main + group 分离 | ✅ 采用，符合实际场景 |

**可改进的点**:

| 模块 | 当前问题 | 改进方向 |
|------|----------|----------|
| **持久化** | JSON 文件 | 考虑 SQLite/嵌入式 DB |
| **测试** | 70% 覆盖率 | 提升至 85%+ |
| **文档** | Mintlify 托管 | 本地文档 + 搜索 |
| **监控** | 较少 | 集成 OpenTelemetry |

### 10.2 核心技术选型参考

```typescript
// 推荐技术栈
const recommendedStack = {
  runtime: "Node.js ≥22",      // 稳定 + ESM 原生
  language: "TypeScript 5.x",   // 严格模式
  packageManager: "pnpm",       // 速度 + 磁盘效率
  schema: "TypeBox",            // 类型安全 Schema
  websocket: "ws",              // 轻量级
  testing: "Vitest",            // 快速 + 兼容 Jest
  formatting: "oxlint + oxfmt", // Rust 工具，极快
  
  // 消息平台
  whatsapp: "@whiskeysockets/baileys",  // 无官方 API 时的选择
  telegram: "grammy",                    // 优于 telegraf
  discord: "discord.js",                 // 功能完整
  slack: "@slack/bolt",                  // 官方推荐
  
  // 浏览器控制
  browser: "puppeteer-core",    // CDP 协议
  
  // 部署
  container: "Docker",
  orchestration: "docker-compose / fly.io / Cloudflare Workers",
};
```

---

## 11. 总结

### 11.1 OpenClaw 的技术亮点

1. **Gateway 单控制平面**: 简化了多渠道、多客户端的状态同步问题
2. **TypeBox 协议优先**: 一次定义，生成 JSON Schema + Swift 模型，减少跨平台不一致
3. **沙箱安全模型**: Docker 隔离 + 工具白名单，在实用性和安全性间取得平衡
4. **渠道插件化**: 核心渠道内置，扩展渠道独立加载，架构清晰
5. **Skills 平台**: 类似 Claude Code 的 SKILL.md 约定，生态可扩展

### 11.2 构建类似系统的关键决策点

| 决策 | 选项 | 建议 |
|------|------|------|
| **单体 vs 微服务** | 单体 Gateway | 对于个人助手场景，单体更合适 |
| **消息协议** | 自定义 vs gRPC vs REST | 自定义 WS 协议，灵活度高 |
| **认证方式** | Token vs OAuth vs Device | 三者结合，覆盖不同场景 |
| **AI 模型** | 单一 vs 多模型 | 多模型 + Fallback，避免供应商锁定 |
| **持久化** | 文件 vs SQLite vs 云 | 本地文件足够，规模大了考虑 SQLite |
| **部署模式** | 本地 vs 云 vs 混合 | 推荐本地 Gateway + 可选云隧道 |

### 11.3 风险提示

- **提示注入防御仍不成熟**: 这是行业级难题，非 OpenClaw 特有
- **高权限工具需谨慎**: shell 访问 = root 访问，必须沙箱化
- **消息平台 TOS 风险**: WhatsApp Web 逆向可能违反服务条款
- **成本控制**: Claude API 调用费用可能超预期（$5-$150+/月）

---

## 附录 A: 关键文件清单

| 文件/目录 | 用途 |
|-----------|------|
| `src/gateway/index.ts` | Gateway 入口 |
| `src/agent/loop.ts` | Agent 循环核心 |
| `src/channels/index.ts` | 渠道注册与管理 |
| `src/tools/index.ts` | 工具注册表 |
| `src/config/schema.ts` | TypeBox 配置 Schema |
| `scripts/protocol-gen.ts` | 协议代码生成 |
| `apps/macos/Sources/MoltbotProtocol/` | Swift 自动生成模型 |
| `~/.openclaw/openclaw.json` | 运行时配置 |
| `~/.openclaw/credentials/` | 渠道凭据存储 |
| `~/clawd/` | 默认工作区 |

## 附录 B: 常用命令

```bash
# 开发
pnpm install                    # 安装依赖
pnpm build                      # 构建
pnpm gateway:watch              # 开发模式（热重载）
pnpm test                       # 运行测试
pnpm lint                       # 代码检查
pnpm protocol:gen               # 重新生成协议

# 运维
openclaw doctor                 # 配置检查
openclaw status --all           # 状态报告
openclaw logs -f                # 实时日志
openclaw dashboard              # 打开 Control UI
openclaw gateway --verbose      # 前台运行
```

---

*报告生成时间: 2026-01-31*  
*数据来源: GitHub 仓库、官方文档、技术博客、安全研究报告*
