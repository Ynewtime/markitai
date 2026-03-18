# P0-B: 安全边界测试补充

Date: 2026-03-18
Parent: [2026-03-18-codebase-review-report.md](2026-03-18-codebase-review-report.md)
Priority: P0

---

## 目标

为 3 个无测试的安全边界模块补充单元测试：
- `cli/processors/validators.py` (260 行) — 输入校验
- `converter/_patches.py` (98 行) — 兼容性补丁
- `utils/paths.py` (95 行) — 路径操作

---

## Task 1: validators.py 测试

4 个公开函数，测试重点：

### check_vision_model_config()
- LLM 未启用时 alt/desc 开启 → 应有警告
- config override (`model_info.supports_vision`) 优先于自动检测
- 本地 provider (claude-agent/) 自动识别为 vision capable
- 空 model_list + vision 启用 → 应有警告
- verbose 模式输出差异

### _check_copilot_unsupported_models()
- weight=0 的 copilot/o1 → 不警告（已禁用）
- weight>0 的 copilot/o1 → 警告
- 非 o1/o3 的 copilot 模型 → 不警告

### check_playwright_for_urls()
- STATIC/JINA 策略 → 跳过检查
- AUTO 策略 + Playwright 未安装 → 警告

### warn_case_sensitivity_mismatches()
- 文件名大小写不匹配 pattern → 警告
- 文件不在 input_dir 内 → 使用 f.name 兜底
- 空列表 → 无输出

---

## Task 2: _patches.py 测试

3 个函数，测试重点：

### apply_all_patches()
- 幂等性：多次调用只 patch 一次
- `_patches_applied` 全局标志

### apply_openpyxl_patches()
- 带 'bg' kwarg → 移除后调用原函数
- 不带 'bg' kwarg → 正常传递
- 已 patch → 跳过（`_markitai_patched` 标志）
- openpyxl 未安装 → 静默返回

### apply_pptx_patches()
- 正常 XML → 使用原始 parser
- 畸形 XML (XMLSyntaxError) → 回退到宽容 parser
- str 输入 → 转为 bytes
- 已 patch → 跳过
- lxml/pptx 未安装 → 静默返回

---

## Task 3: paths.py 测试

5 个函数，都是 mkdir 薄封装，测试重点：

### ensure_dir / ensure_subdir
- 正常创建 + 幂等性
- 返回 Path 可链式调用

### ensure_assets_dir / ensure_screenshots_dir / ensure_reports_dir
- 创建正确的 `.markitai/{子目录}` 结构
- 相互独立（创建一个不影响其他）

---

## 执行顺序

1. Task 2 (_patches.py) — 最关键，回归风险最高
2. Task 1 (validators.py) — 安全边界校验
3. Task 3 (paths.py) — 最简单

## 测试策略

- TDD 在此场景是"补充测试"而非"新功能"，测试即目标
- Mock 边界：只 mock 缺失的可选依赖（openpyxl、lxml），不 mock 内部逻辑
- 使用 `tmp_path` 做文件系统测试，不 mock Path 操作
