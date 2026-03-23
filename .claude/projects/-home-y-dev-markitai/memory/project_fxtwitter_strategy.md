---
name: fxtwitter_strategy_redesign
description: FxTwitter integration needs strategy CLI redesign - merge explicit flags into -s/--strategy
type: project
---

FxTwitter 当前作为 playwright 的前置拦截实现，但 `--playwright` 显式传参时被跳过（`not explicit_strategy` guard）。

**Why:** 用户希望统一策略传参为 `-s/--strategy auto|defuddle|static|playwright|markitdown|cloudflare`，将 FxTwitter 集成到 auto 模式的策略链中。

**How to apply:** 在完成所有 Phase 后统一重构 CLI 策略传参。FxTwitter 应在 auto 模式下 defuddle 失败后、playwright 之前自动触发。
