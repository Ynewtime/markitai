# Markitai 代码库全面评审报告

> 审查日期: 2026-03-22
> 修订日期: 2026-03-22（根据第三方交叉评审反馈修订）
> 审查范围: 全代码库 (v0.12.0)
> 审查方法: 静态代码分析 + 逻辑推演 + 事实核查 + 第三方交叉评审

---

## 目录

- [一、项目概况](#一项目概况)
- [二、架构亮点](#二架构亮点)
- [三、已确认问题](#三已确认问题)
  - [P0 - 数据完整性](#p0---数据完整性)
  - [P1 - 运行时稳定性](#p1---运行时稳定性)
  - [P2 - 质量与用户体验](#p2---质量与用户体验)
  - [P3 - 低优先级改善](#p3---低优先级改善)
- [四、事实核查记录](#四事实核查记录)
  - [第一轮内部核查: 初始审查中被否决的发现](#第一轮内部核查-初始审查中被否决的发现)
  - [第二轮交叉评审: 被修正或删除的发现](#第二轮交叉评审-被修正或删除的发现)
- [五、测试覆盖评估](#五测试覆盖评估)
- [六、用户体验评估](#六用户体验评估)
- [七、修复计划](#七修复计划)

---

## 一、项目概况

Markitai 是一个支持 LLM 增强的 Markdown 转换器，将 PDF、Office 文档、图片、网页等多种格式转为结构化 Markdown。项目采用 Python 3.11+ 开发，使用 uv workspace 管理，核心包位于 `packages/markitai/`。

**整体评价**: 代码质量良好，架构清晰，模块化合理。安全意识较强（原子写入、符号链接检测、路径遍历防护）。在少数关键路径上存在资源清理不完整的问题。

**代码规模**: 约 120 个 Python 源文件，覆盖 CLI、Web 提取、文件转换、LLM 处理、图片分析等子系统。

---

## 二、架构亮点

以下设计值得肯定，建议保持：

### 2.1 原子文件写入 (security.py)

使用 `tempfile.mkstemp()` + `os.replace()` 的经典模式，确保文件写入的原子性。Windows 平台有指数退避重试逻辑处理文件锁。`fd_closed` 标志确保文件描述符不被双重关闭。POSIX 下 `mkstemp()` 创建文件权限为 `0o600`，`os.replace()` 保留临时文件 inode 的权限位，配置文件写入后默认不可被其他用户读取。

### 2.2 FetchCache 并发文档化约束 (fetch_cache.py:57-65)

sync/async 双锁独立是已知限制，代码注释清晰说明了约束条件和适用场景：CLI 要么全同步要么全异步，不会混用。这是文档化的设计决策，不是 bug。

### 2.3 批处理断点续传 (batch.py)

`BatchState` 追踪 PENDING → IN_PROGRESS → COMPLETED/FAILED 状态转换，支持中断后恢复。状态保存有间隔控制（默认 5 秒）和双重检查锁定，避免 O(N^2) I/O。加载时有 `validate_file_size()` 防止 DoS。

### 2.4 多策略 URL 获取 (fetch.py)

支持 6 种 fetch 策略（auto/static/defuddle/jina/playwright/cloudflare），带 SPA 域名学习缓存。策略选择有智能 fallback 链。

### 2.5 内容保护与幻觉检测 (llm/content.py)

使用 `__MARKITAI_*__` 格式的 placeholder 保护页码/幻灯片标记，避免被 LLM 篡改。幻觉标记移除逻辑正确——placeholder 存在时原始标记已被保护，反之确实是 LLM 幻觉。frontmatter 的 prompt 泄漏检测覆盖中英文常见模式。

### 2.6 双语 i18n

CLI 命令和安装脚本全程中英文双语，setup.sh 从系统 locale 自动检测语言。Clack 风格 UI 美观一致。

### 2.7 Pydantic 配置验证 (config.py)

基于 Pydantic 的嵌套配置验证，`_modified_keys` 追踪变更实现最小 diff 保存，`env:VAR_NAME` 语法支持环境变量解引用。

### 2.8 LLM 路由层超时 (config.py, llm/processor.py)

`RouterSettings.timeout` 默认 120 秒（`constants.py:56`），传入 LiteLLM `Router()` 构造函数（`processor.py:936`），为所有 LLM API 调用提供了基线超时保护。

---

## 三、已确认问题

以下所有问题已通过直接阅读源码核实。

### P0 - 数据完整性

#### P0-1: PDF 临时目录清理不在 finally 块中

**文件**: `packages/markitai/src/markitai/converter/pdf.py`
**行号**: 190-335

**现状**:

```python
# 第 190 行: 创建临时目录
temp_dir = Path(tempfile.mkdtemp())

# 第 202-331 行: 大量转换逻辑，任何一步可能抛异常
page_results = pymupdf4llm.to_markdown(...)  # 第 202 行
markdown = self._fix_image_paths(...)         # 第 236 行
# ... 图片压缩、截图渲染等 ...

# 第 334 行: 清理逻辑不在 finally 中
if temp_dir and temp_dir.exists():
    shutil.rmtree(temp_dir, ignore_errors=True)
```

**问题**: 第 202 行的 `pymupdf4llm.to_markdown()`、第 236 行之后的图片处理、以及第 320 行的 `_render_pages_parallel()` 中任何一步抛异常，`temp_dir` 都不会被清理。批量处理大量 PDF 时，泄漏的临时目录会持续积累。

**影响**: 磁盘空间泄漏，批量模式下影响更大。

**修复方案**:

```python
temp_dir: Path | None = None
try:
    if output_dir:
        image_path = ensure_assets_dir(output_dir)
    else:
        temp_dir = Path(tempfile.mkdtemp())
        image_path = temp_dir

    # ... 现有的全部转换逻辑 ...

    # 清理临时目录（移到 return 之前）
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        images = [img for img in images if img.path and img.path.exists()]
        metadata.pop("reference_images", None)

    return ConvertResult(markdown=markdown, images=images, metadata=metadata)
finally:
    # 异常路径的兜底清理
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
```

---

#### P0-2: HTML 消毒可考虑增强

**文件**: `packages/markitai/src/markitai/webextract/sanitize.py`
**行号**: 5, 63-68

**现状**:

```python
UNSAFE_URL_PREFIXES = ("javascript:", "data:text/html", "data:image/svg+xml")

for attr in ("href", "src"):
    value = tag.get(attr)
    if isinstance(value, str) and value.strip().lower().startswith(
        UNSAFE_URL_PREFIXES
    ):
        del tag.attrs[attr]
```

**观察**:
1. URL 编码绕过：`javascript%3Aalert(1)` 经 `.lower()` 后仍为 `javascript%3aalert(1)`，不匹配 `javascript:` 前缀
2. 属性范围：只检查 `href` 和 `src`，未覆盖 `action`、`formaction` 等可承载 URL 的属性

**风险评估**: 需要先确认 threat model。当前 sanitized HTML 会继续经过 `webextract/pipeline.py` 的标准化处理并转为 Markdown，`javascript:` 链接在 Markdown 输出中的实际可利用性取决于下游消费者（如果是纯文本工具链则风险很低）。URL 编码的 `javascript%3A` 在浏览器中作为 `href` 值不会被自动 decode 后执行，所以这不是一个可以直接利用的 XSS 向量，而是一个防御深度的改善点。

**建议**: 建议先梳理 threat model（输出 Markdown 的下游消费场景），再决定是否增加 URL decode 和属性范围扩展。如果下游包含 HTML 渲染场景，可考虑以下增强：

```python
from urllib.parse import unquote

_URL_ATTRS = ("href", "src", "action", "formaction", "data-src", "poster")

# 在 _sanitize_tag 中:
for attr in _URL_ATTRS:
    value = tag.get(attr)
    if isinstance(value, str):
        decoded = unquote(value).strip().lower()
        if decoded.startswith(UNSAFE_URL_PREFIXES):
            del tag.attrs[attr]
```

---

### P1 - 运行时稳定性

#### P1-1: webextract 渲染器嵌套回复无深度限制

**文件**: `packages/markitai/src/markitai/webextract/render.py`
**行号**: 136-148

**现状**:

```python
def _render_item_tree(
    item: ConversationItem,
    items: list[ConversationItem],
) -> str:
    parts = [_render_item(item)]
    children = _iter_child_items(item.id, items)
    if children:
        parts.append('<blockquote class="reply-thread">')
        for child in children:
            parts.append(_render_item_tree(child, items))  # 无限递归
        parts.append("</blockquote>")
    return "\n".join(parts)
```

**问题**: 深度嵌套的会话线程（如 Reddit 长评论链、HackerNews 深度讨论）可能导致 Python 默认 1000 层的递归限制被触发，抛出 `RecursionError`。

**影响**: 处理深度嵌套页面时崩溃。

**修复方案**:

```python
_MAX_REPLY_DEPTH = 50

def _render_item_tree(
    item: ConversationItem,
    items: list[ConversationItem],
    depth: int = 0,
) -> str:
    parts = [_render_item(item)]
    if depth < _MAX_REPLY_DEPTH:
        children = _iter_child_items(item.id, items)
        if children:
            parts.append('<blockquote class="reply-thread">')
            for child in children:
                parts.append(_render_item_tree(child, items, depth + 1))
            parts.append("</blockquote>")
    return "\n".join(parts)
```

---

#### P1-2: 部分 LLM 异常路径缺少用户可见提示

**文件**: `packages/markitai/src/markitai/workflow/single.py`
**行号**: 221, 385, 527

**现状**: `process_document()` (第 186 行) 和 `enhance_with_vision()` (第 440 行) 已经有 stderr warning 输出，但以下三处仍然是静默降级：
- `process_document_pure()` (第 221 行) — 仅 `logger.error()`
- `analyze_images()` (第 385 行) — 仅 `logger.error()`
- `extract_from_screenshots()` (第 527 行) — 仅 `logger.error()`

```python
# process_document_pure (第 221 行)
except Exception as e:
    logger.error(f"Pure LLM processing failed: {format_error_message(e)}")
    return markdown, 0.0, {}  # 静默返回，用户不知情
```

**问题**: 用户使用 `--pure` 模式或 `--screenshot-only` 模式时，LLM 失败后静默返回原始/空内容，无任何终端提示。

**影响**: 用户体验——以为 LLM 增强成功但实际未生效。

**修复方案**: 参照已有的 `process_document()` 模式，为这三处添加 stderr warning：

```python
except Exception as e:
    logger.error(f"Pure LLM processing failed: {format_error_message(e)}")
    from rich.console import Console
    Console(stderr=True).print(
        f"[yellow]Warning: Pure LLM processing failed: {format_error_message(e)}[/yellow]"
    )
    return markdown, 0.0, {}
```

---

#### P1-3: Windows COM 线程内图片压缩（优化项）

**文件**: `packages/markitai/src/markitai/converter/office.py`
**行号**: 246-346

**现状**: `_render_slides_with_com()` 在 `pythoncom.CoInitialize()` 的 try 块内，对每个 slide 导出后立即进行 PIL 图片压缩（第 283-306 行）。

**观察**:
1. 如果图片压缩抛异常（如 PIL 格式错误），会中断 slide 循环，后续 slides 不被处理
2. COM 线程内做 CPU 密集的图片压缩效率不高——COM 操作应尽快完成

**说明**: 这更接近一个代码组织优化而非明确缺陷。COM 的 `finally` 块能正确清理资源，异常不会导致 COM 泄漏。但将导出与压缩分离可以提高健壮性。

**建议方案**: 将导出和压缩分为两个阶段：

```python
def _render_slides_with_com(self, input_path, screenshots_dir, image_format):
    # Phase 1: COM 导出 (在 COM 上下文内)
    exported_paths = []
    pythoncom.CoInitialize()
    try:
        ppt = win32com.client.Dispatch("PowerPoint.Application")
        presentation = ppt.Presentations.Open(...)
        for i, slide in enumerate(presentation.Slides, 1):
            image_path = screenshots_dir / f"..."
            slide.Export(str(image_path.resolve()), export_format)
            exported_paths.append((i, image_path))
        presentation.Close()
    finally:
        # COM 清理 ...
        pythoncom.CoUninitialize()

    # Phase 2: 图片压缩 (在 COM 上下文外)
    images, slide_images = [], []
    for i, image_path in exported_paths:
        # ... 压缩和收集逻辑 ...

    return images, slide_images
```

---

### P2 - 质量与用户体验

#### P2-1: Token 计数对中文内容严重偏低

**文件**: `packages/markitai/src/markitai/providers/__init__.py`
**行号**: 135-138

**现状**:

```python
# Fallback: character-based estimation
# Rough estimate: 1 token ≈ 4 characters for English
return len(text) // 4
```

**问题**: 中文文本中每个字符约消耗 1-2 个 token（而非英文的 0.25 个 token）。对一段 1000 字的中文文档：
- 当前估算: `1000 // 4 = 250 tokens`
- 实际消耗: 约 1000-1500 tokens
- 误差: 4-6 倍

这导致费用估算显示远低于实际值。

**影响**: 费用报告误导用户（仅影响非 OpenAI 模型的估算显示，不影响实际 API 调用）。

**修复方案**:

```python
def _estimate_tokens_by_chars(text: str) -> int:
    """Estimate token count based on character types."""
    cjk_count = 0
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF    # CJK Unified Ideographs
            or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
            or 0x3000 <= cp <= 0x303F  # CJK Symbols
            or 0xFF00 <= cp <= 0xFFEF  # Fullwidth Forms
            or 0xAC00 <= cp <= 0xD7AF  # Hangul
            or 0x3040 <= cp <= 0x30FF):  # Hiragana/Katakana
            cjk_count += 1
    non_cjk_count = len(text) - cjk_count
    # CJK: ~1.5 tokens/char; Latin: ~0.25 tokens/char
    return int(cjk_count * 1.5) + non_cjk_count // 4
```

---

#### ~~P2-2: 进度指示缺乏策略信息~~ — 已撤回

**撤回原因**: `url.py:237` 已有 `progress.log(f"Fetched via {used_strategy}: {url}")`，fetch 完成后策略名称已经显示。原始审查未注意到这行代码。

---

### P3 - 低优先级改善

#### P3-1: url.py 存在语义可疑的条件分支

**文件**: `packages/markitai/src/markitai/cli/processors/url.py`
**行号**: 312-314

```python
elif cfg.llm.pure and not cfg.llm.enabled:
    base_content = original_markdown
```

**分析**: 此条件可达——用户可以在 config 中设置 `pure: true` 但命令行不传 `--llm`。语义上 "pure 模式不启用 LLM" 是合理的降级。**不需要修复**，但建议添加注释说明此分支的用途。

#### P3-2: OutputManager 窄终端行计数

**文件**: `packages/markitai/src/markitai/cli/output_manager.py`
**行号**: 70

`rendered.count("\n")` 不考虑 Rich 在窄终端上的自动换行。实际影响极小——OutputManager 用于 stderr spinner 和短提示信息，不太可能超过终端宽度。**不修复**。

#### P3-3: batch.py 状态文件 schema 验证

**文件**: `packages/markitai/src/markitai/batch.py`
**行号**: 900-907

已有 `validate_file_size()` + `BatchState.from_dict()` + `except Exception` 兜底。加载失败时返回 `None`（重新开始处理），不会崩溃。**现有防护已足够**，无需额外 schema 验证。

---

## 四、事实核查记录

### 第一轮内部核查: 初始审查中被否决的发现

| 初始发现 | 核查结论 | 原因 |
|---------|---------|------|
| FetchCache sync/async 锁竞争标为 P0 | **降级为 Won't Fix** | fetch_cache.py:57-65 有明确的文档化约束，CLI 实际使用模式（全同步或全异步）确保安全 |
| url.py 第 312 行条件不可达 | **否决** | 条件 `cfg.llm.pure and not cfg.llm.enabled` 可达——pure 在 config 中设置但 --llm 未传 |
| PROMPT_LEAKAGE_KEY_PATTERNS 重复编译 | **否决** | Python `re` 模块有内置 compiled pattern 缓存（最多 512 个），这些简单模式不会重复编译 |
| content.py 幻觉标记移除过于激进 | **否决** | 逻辑正确: placeholder 存在时原始标记已被保护（不在原文中出现），不存在时确实是 LLM 幻觉。注释清晰说明了两种情况 |
| pipeline.py Level 2 retry 的 2x 阈值过于保守 | **否决** | 多级 fallback 设计：Level 2 要求 2x 改进（确保高质量），Level 3 和 4 只要求 > current（宽松降级）。渐进策略合理 |
| sanitize.py checkbox 双重处理 | **否决** | `tag.decompose()` 从 DOM 移除标签后 `find_all(True)` 不会再遇到。checkbox 的 `return` 只是跳过，不会影响后续遍历 |
| OCR 引擎双重检查锁竞争 | **降级为 Won't Fix** | CLI 场景下是单线程初始化，不存在竞争 |

### 第二轮交叉评审: 被修正或删除的发现

第三方交叉评审指出了报告初版中的 5 处事实错误，以下逐一记录验证结论和修正措施：

#### 1. 原 P0-3「配置文件未设置限制性权限」— 已删除

**第三方论点**: `mkstemp()` 默认 `0o600`，POSIX 下 `os.replace()` 保留临时文件 inode 的权限位，不会按 umask 重新创建。

**验证**: 本地测试确认 `mkstemp()` → `os.replace()` 后最终文件权限为 `0o600`。报告原文错误地声称 "os.replace() 后目标文件继承的权限取决于原文件或 umask"，这在 POSIX 语义下不成立——`os.replace()` 是原子 rename，保留源 inode 的全部属性。

**结论**: 删除。配置文件写入后默认已为 `0o600`，无安全问题。此发现已移至架构亮点 2.1 中说明。

#### 2. 原 P2-2「LLM Cache glob 模式 `**/` 处理缺陷」— 已删除

**第三方论点**: `fnmatch.fnmatch('src/deep/nested/test.py', 'src/**/test.py')` 本身就返回 True，后面的 collapsed 分支是为了补零层目录情况。建议的 `PurePosixPath.match()` 替代方案反而不匹配深层路径。

**验证**: 本地 Python 3.13 测试确认：
- `fnmatch.fnmatch('src/deep/nested/test.py', 'src/**/test.py')` → `True` ✓
- `PurePosixPath('src/deep/nested/test.py').match('src/**/test.py')` → `False` ✗

报告原文对 `fnmatch` 的 `**` 行为理解错误。`fnmatch` 中 `**` 等价于 `*`（匹配包含 `/` 的任意字符），所以深层路径本身就能匹配。collapsed 分支处理的是 `src/test.py` 这种 `**/` 对应零层目录的边界情况。原提议的修复方案会引入回归。

**结论**: 删除。现有 glob 匹配逻辑正确。

#### 3. 测试覆盖评估中的事实错误 — 已重写

**第三方论点**: 仓库中存在 `tests/unit/test_auth_cli.py`、`tests/integration/test_cli.py:86`（init integration 测试）、`tests/integration/test_cli_full.py:559`（config_init 测试）、`tests/unit/cli/test_providers_detect.py`（provider 自动检测测试），但报告声称这些测试缺失。

**验证**: 全部确认存在。报告初版的测试覆盖评估未遍历完整测试树，导致多处事实错误。

**结论**: 测试覆盖评估一节已重写，基于实际测试文件重新评估。

#### 4. 原 P1-3「asyncio.gather 会无限等待」— 已降级并改描述

**第三方论点**: `RouterSettings.timeout = 120`（`constants.py:54`）传入 LiteLLM `Router()`（`processor.py:936`），LLM API 层已有基线超时保护。

**验证**: 确认 `RouterSettings` 的 `timeout` 字段默认值为 120 秒，且通过 `router_settings = self.config.router_settings.model_dump()` 传入 `Router()` 构造函数。LiteLLM Router 在 HTTP 层会在 120 秒后超时。

**结论**: 原 P1-3 声称 "没有 per-task 超时 = 无限卡死" 不成立。已从独立条目中删除。如果需要更细粒度的 workflow-level cancellation（如区分单个图片 vs 整个 gather 的超时），可作为后续改进项，但不是当前缺陷。

#### 5. 原 P1-4「LLM 失败时用户无感知」— 已缩小范围

**第三方论点**: `process_document()` 第 190 行和 `enhance_with_vision()` 第 444 行已经有 `Console(stderr=True).print()` 的 warning 输出。

**验证**: 确认这两处已有 stderr warning。真正还静默的是 `process_document_pure()` (第 221 行)、`analyze_images()` (第 385 行)、`extract_from_screenshots()` (第 527 行) 三处。

**结论**: 问题范围从 "6 处都无感知" 缩小为 "3 处仍静默"。已重写为 P1-2 并精确列出受影响的方法。

---

## 五、测试覆盖评估

> 注: 本节在第三方交叉评审后重写，基于实际测试文件树重新评估。

### 覆盖情况概览

| 模块 | 覆盖度 | 说明 |
|------|--------|------|
| CLI 命令 (config/cache/doctor) | 良好 | 基本覆盖所有子命令及其参数组合 |
| CLI Init 命令 | 良好 | `test_cli.py:86`、`test_cli_full.py:559`、`test_cli_main.py:924` 有 integration 测试 |
| CLI Auth 命令 | 良好 | `test_auth_cli.py` 有专门的 CLI 级 auth 测试 |
| Provider 自动检测 | 良好 | `test_providers_detect.py` 有独立测试 |
| webextract 提取器 | 良好 | 有 parity 测试对比 fixture 合约 |
| 转换器 (PDF/Office/Image) | 一般 | 核心路径有覆盖，但异常路径和边界条件不足 |
| 安装脚本 (setup.sh/ps1) | 弱 | Playwright 多级 fallback、镜像配置未在 CI 中测试 |

### 建议补充的测试

1. **PDF 转换器异常路径**: 模拟 `pymupdf4llm.to_markdown()` 抛异常，验证 temp_dir 清理（对应 P0-1）
2. **HTML 消毒边界情况**: 添加 URL 编码变体的测试用例（对应 P0-2 的 threat model 梳理）
3. **渲染器深度嵌套**: 构造深度 > 100 的 conversation thread，验证不崩溃（对应 P1-1）
4. **LLM 异常路径 stderr 输出**: 验证 `process_document_pure` 等在异常时向 stderr 输出 warning（对应 P1-2）

---

## 六、用户体验评估

### 安装体验

**优点**:
- 一键安装脚本（curl/powershell）
- 双语 UI、自动镜像配置
- 渐进式安装：核心 → 可选依赖 → LLM CLI 工具
- 安装过程有 spinner 反馈

**问题**:
- README 过于简短——只有安装命令，没有功能介绍、使用示例、输出预期
- 安装后没有 "接下来做什么" 的引导

### 日常使用

**优点**:
- `markitai doctor` 可快速诊断环境问题
- `markitai init` 交互式初始化
- 批处理支持断点续传
- dry-run 模式预览操作

**问题**:
- 部分 LLM 模式（pure、screenshot-only）失败时静默降级（P1-2），用户不知道是否生效
- `markitai doctor` 中 model list 为空时显示 "missing"，即使 API key 已正确——可能困惑新用户

### 配置体验

**优点**:
- 多级配置搜索：CLI args → env vars → local config → global config → defaults
- `markitai config path` 可显示生效的配置文件路径
- `markitai config validate` 可验证配置

**问题**:
- 本地配置覆盖全局配置时无提示
- `.env.example` 的双语注释很详细，但缺少 "最小配置" 示例

---

## 七、修复计划

### Phase 1: 数据完整性 (P0) — 预计 1 天

| 编号 | 修复项 | 文件 | 风险 |
|------|--------|------|------|
| Fix 1.1 | PDF temp_dir 加 try-finally | converter/pdf.py | 低 |
| Fix 1.2 | HTML 消毒 threat model 梳理 + 按需增强 | webextract/sanitize.py | 低 |

### Phase 2: 运行时稳定性 (P1) — 预计 1-2 天

| 编号 | 修复项 | 文件 | 风险 |
|------|--------|------|------|
| Fix 2.1 | 递归深度限制 | webextract/render.py | 低 |
| Fix 2.2 | 3 处 LLM 异常路径补 stderr warning | workflow/single.py | 低 |
| Fix 2.3 | COM 导出与压缩分离（优化项） | converter/office.py | 中 |

### Phase 3: 质量改善 (P2) — 预计 0.5 天

| 编号 | 修复项 | 文件 | 风险 |
|------|--------|------|------|
| Fix 3.1 | Token 计数支持 CJK | providers/__init__.py | 低 |

### 实施顺序

```
Phase 1 (Fix 1.1 → 1.2)
Phase 2 (Fix 2.1 → 2.2 → 2.3)
Phase 3 (Fix 3.1)
```

Phase 1 为数据完整性问题，建议优先修复。Phase 2 改善运行时稳定性。Phase 3 为质量改善，可与日常开发并行。Fix 2.3（COM 分离）需 Windows 环境验证，可异步进行。

---

*报告结束*
