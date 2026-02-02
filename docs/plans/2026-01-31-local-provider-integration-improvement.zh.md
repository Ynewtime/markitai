# 本地模型提供商集成优化计划 (Local Provider Integration Improvement Plan)

**日期**: 2026-01-31
**状态**: 已批准
**作者**: Claude (Opus 4.5)
**审核人**: Y

---

## 1. 执行摘要

本文档概述了 Markitai 中 Claude Code (`claude-agent`) 和 GitHub Copilot (`copilot`) 提供商集成的全面改进计划。该计划分为三个阶段：可靠性基础、功能增强以及测试与工具。

**核心成果**:
- 具备优雅降级能力的认证状态处理
- 基于请求复杂度的自适应超时机制
- 针对边缘情况的增强重试策略
- 针对 Claude Agent 的 Prompt Caching（提示词缓存）支持以降低成本
- **架构治理**: 统一的 JSON 模式处理与图像处理管道
- 基于 Mock 的全面集成测试
- 用于故障排除的诊断 CLI 命令

**预计时间**: 8-12 天 (增加了架构重构工作量)
**风险等级**: 低到中

---

## 2. 现状分析与 OpenCode 调研启示

### 2.1 架构概览
目前的集成通过 `HybridRouter` 分流：标准模型走 LiteLLM Router，本地 CLI 订阅模型（Claude Agent, Copilot）走自定义的 `LocalProviderWrapper`。

### 2.2 深度代码审计发现的问题
在对代码库的深度探索中，发现了以下架构级优化机会：
- **图像处理分裂**: `copilot.py` 实现了私有的 PIL 逻辑，未复用核心 `image.py` 的高性能 OpenCV/多进程管道。
- **JSON 逻辑分散**: JSON 提取逻辑散落在 Provider、Vision Mixin 和 Document 处理中，缺乏统一的清洗与重试标准。
- **能力定义硬编码**: `LocalProviderWrapper` 硬编码了支持视觉的模型列表，违反开闭原则。

### 2.3 OpenCode 集成方案的深度参考
根据 `docs/archive/deps/opencode-claudecode-integration.md` 的分析，OpenCode 采用了逆向工程手段，而 Markitai 坚持使用官方 SDK。

| 维度 | OpenCode 方案 (逆向工程) | Markitai 方案 (官方 SDK) | 启示 |
|------|-------------------------|--------------------------|------|
| **认证机制** | 逆向 OAuth PKCE 流程，模拟 Client ID | 委托给 `claude-agent-sdk` | 官方 SDK 路径更稳定，不易受 ToS 封号风险 |
| **请求伪装** | 必须模拟 `User-Agent` 和特定的 `beta` Header | SDK 内部自动处理 | 无需手动维护复杂的 Header 列表 |
| **工具混淆** | 使用 `mcp_` 前缀绕过工具名白名单 | 不涉及工具调用 | 我们主要使用其推理能力，避开了最易被封锁的特征 |
| **身份注入** | 在 System Prompt 注入 "You are Claude Code" | 官方 SDK 原生支持 | 无需进行提示词伪装 (Spoofing) |
| **缓存优化** | 手动处理 `cache_control` | **待实现** | OpenCode 证实了在订阅模式下 Prompt Caching 的巨大价值 |

**核心洞察**: OpenCode 的路径是一场“猫鼠游戏”，容易失效。Markitai 使用官方 SDK 路径，虽然目前稳定性更高，但应参考 OpenCode 记录的常见错误模式（如 402 欠费、429 限流、1 小时 Token 过期）来优化错误提示。

---

## 3. 第一阶段：可靠性基础 (P0)

### 3.1 认证管理与状态感知
新建 `providers/auth.py`，实现 `AuthManager`。
- **Copilot**: 利用 SDK 原生 `client.get_auth_status()` 方法精准获知订阅状态。
- **Claude**: 由于 CLI 无状态查询命令，采用“试探性调用”或运行 `claude doctor` 进行健康检查。
- **友好引导**: 当检测到未认证时，给出准确的引导命令：
  - Copilot: "请在终端运行 `copilot` 并按提示登录"
  - Claude: "请运行 `claude setup-token` 配置认证"

### 3.2 自适应超时机制 (Adaptive Timeout)
废弃硬编码的 120s 超时，根据以下因素动态计算：
- **输入长度**: 每 1000 字符增加约 2 秒。
- **多模态**: 包含图片请求时，超时系数乘以 1.5。
- **预期输出**: 设置合理的输出 Token 增长预期。

### 3.3 增强重试策略
在 `llm/processor.py` 中识别“可重试”错误：
- **重试**: 网络抖动、连接重置、流中断。
- **不重试**: 认证错误 (401)、欠费 (402)、权限不足 (403)。

---

## 4. 第二阶段：功能增强 (P1)

### 4.1 Prompt Caching (针对 Claude Agent)
实现 Anthropic 风格的提示词缓存：
- **机制**: 自动为超过 1024 tokens 的系统提示词或长上下文添加 `cache_control: {"type": "ephemeral"}`。
- **TTL**: 由服务端托管（默认 5 分钟，命中自动刷新），无需客户端指定。
- **价值**: 显著降低由于重复读取长文档产生的计算配额消耗。

### 4.2 统一 JSON 模式 (重构)
新建 `providers/json_mode.py`，并升级为通用的 `StructuredOutputHandler`：
- **目标**: 消除 `claude_agent.py`、`copilot.py` 以及 `llm/vision.py` 中重复的 JSON 处理代码。
- **功能范围**:
  - 生成适配不同模型的 System Prompt (JSON Schema 注入)。
  - 统一的响应解析与清洗逻辑 (处理 Markdown 代码块包裹、控制字符等)。
  - 被 Vision 模块复用，替代现有的 `_analyze_with_json_mode` 手动实现。

### 4.4 统一配置与能力侦测 (Smart Capability Detection)
- **移除硬编码**: 删除 `LocalProviderWrapper` 中的 `_IMAGE_CAPABLE_PATTERNS`。
- **智能推断链 (Zero Config)**: 
  - 修改 `LocalProviderWrapper._is_image_capable`，优先调用 `get_local_provider_model_info`。该函数会自动解析 Provider 别名并查询 LiteLLM 的权威模型数据库。
  - **Fallback**: 仅当 LiteLLM 查无数据时，回退到 Provider 定义的默认策略（例如 `claude-agent/*` 默认为 True），确保用户无需手动配置。
- **统一异常体系**: 新建 `providers/errors.py` 定义标准异常基类。在 SDK 调用层增加显式的 `try-except` 块，将 SDK 特有异常转换为 LiteLLM 标准异常。

---

## 5. 第三阶段：测试与工具 (P1)

### 5.1 增强诊断命令 (Upgrade check-deps to doctor)
将现有的 `markitai check-deps` 升级并重命名为 `markitai doctor`：
- **功能合并**: 保留原有的依赖检查（Playwright, LibreOffice, RapidOCR）。
- **SDK 环境检查**: 验证 SDK 包安装及 CLI 工具路径（复用 `deps.py` 现有逻辑并增强）。
- **认证状态检查**: 
  - 调用 `CopilotClient.get_auth_status()` 显示当前登录用户和状态。
  - 运行 `claude doctor` 检查 Claude 环境健康度。
- **别名保留**: 保留 `check-deps` 作为 `doctor` 的别名以向后兼容。

### 5.2 基于 Mock 的集成测试
新建 `tests/integration/test_local_providers.py`。
- **Mock SDK**: 模拟 `claude_agent_sdk` 的响应，无需真实账号即可测试重试、JSON 提取和错误路径。
- **覆盖场景**: 正常补全、图片识别、Token 超限、网络错误。

---

## 6. 文件变更摘要

| 文件路径 | 变更类型 | 核心说明 |
|-----------|---------|----------|
| `providers/auth.py` | **新增** | 提供商认证状态管理 |
| `providers/errors.py` | **新增** | 结构化的异常定义 (AuthenticationError, QuotaError) |
| `providers/logging.py` | **新增** | 统一日志格式 `[Provider:CallID] Message` |
| `providers/json_mode.py` | **新增** | 统一跨提供商的 JSON 提取与验证逻辑 |
| `cli/commands/deps.py` | **重命名** | 重命名为 `doctor.py` 并增强认证检查功能 |
| `providers/claude_agent.py` | 修改 | 集成超时计算、缓存控制和新日志系统 |
| `providers/copilot.py` | 修改 | 集成 JSON 模式处理器和新日志系统 |
| `llm/processor.py` | 修改 | 增强重试逻辑与响应完整性检查 |

---

## 7. 风险缓解策略

- **SDK 变更风险**: 官方 SDK 如果更新破坏性 API，通过 `LocalProviderWrapper` 抽象层进行隔离，并利用 Mock 测试快速定位。
- **认证失效风险**: 通过 `AuthManager` 的主动检测，在用户发起长时间转换前提前预警，避免在转换中途失败。
- **成本/配额风险**: 引入 Prompt Caching 优化。同时在日志中记录每个 Call 的 Token 消耗，以便监控。

---

**文档版本**: 1.4 (中文版)
**最后更新**: 2026-02-01
**更新日志**: 
- 整合了深度代码审计发现的架构优化点 (图像管道统一、JSON 处理中心化)。
- 增加了对应的重构任务到第二阶段。
- 调整了预计时间以反映新增的治理工作量。
