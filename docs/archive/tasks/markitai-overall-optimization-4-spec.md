# Markitai 优化待办任务清单

> 基于 markitai-overall-optimization-4.md 评审意见的深入分析
> 创建日期: 2026-01-20
> 最后更新: 2026-01-20

---

## 已完成 ✅

| 任务 | 文件 | 完成日期 |
|------|------|----------|
| 锁定 Python 3.13（spec 与实现一致） | `docs/spec.md` | 2026-01-20 |
| 文件冲突版本号 UUID fallback | `cli.py::resolve_output_path` | 2026-01-20 |
| 嵌套 symlink 递归检测 | `security.py::check_symlink_safety` | 2026-01-20 |
| Hash 长度 spec 对齐（6 位 hex） | `docs/spec.md` | 已对齐 |
| Report 命名 spec 对齐 | `docs/spec.md` | 已对齐 |
| OCR 临时文件泄露修复 | `ocr.py` (使用 BytesIO) | 已修复 |
| TODO 标记清理 | `llm.py` | 2026-01-20 |

---

## P1: 高优先级

### 1. 日志级别生产环境配置

**现状**: `constants.py:131` 中 `DEFAULT_LOG_LEVEL = "DEBUG"` 可能泄露敏感工作流数据

**建议方案**: 保持 DEBUG 作为开发默认值，通过配置文件覆盖

```json
// markitai.json (生产环境示例)
{
  "log": { "level": "INFO" },
  "llm": { "concurrency": 5 },
  "batch": { "concurrency": 5 }
}
```

**可选增强**:
- [ ] 添加 `--profile production` 预设
- [ ] 文档化生产环境配置最佳实践
- [ ] 考虑日志敏感信息脱敏（sanitize API keys in logs）

**涉及文件**: `constants.py`, `cli.py`, `docs/spec.md`

---

### 2. except Exception 异常处理优化

**现状**: 代码中有 **68 处** `except Exception`（11 个文件），吞没错误导致调试困难

**涉及文件统计**:
| 文件 | 数量 | 说明 |
|------|------|------|
| `llm.py` | 21 | 部分已有 RETRYABLE_ERRORS 但未充分使用 |
| `cli.py` | 11 | CLI 层异常处理 |
| `image.py` | 10 | 图片处理 |
| `converter/pdf.py` | 7 | PDF 转换 |
| `converter/legacy.py` | 5 | 旧版转换器 |
| `converter/office.py` | 4 | Office 转换 |
| `workflow/single.py` | 4 | 工作流 |
| `batch.py` | 2 | 批处理 |
| `security.py` | 2 | 安全检查 |
| `converter/image.py` | 1 | 图片转换 |
| `utils/office.py` | 1 | Office 工具 |

**优化原则**:

```python
# ❌ 当前模式
try:
    result = await llm_call()
except Exception as e:
    logger.warning(f"LLM failed: {e}")
    return None  # 静默失败

# ✅ 推荐模式
from litellm.exceptions import RateLimitError, Timeout, APIConnectionError

RETRYABLE_ERRORS = (RateLimitError, Timeout, APIConnectionError)

try:
    result = await llm_call()
except RETRYABLE_ERRORS as e:
    logger.warning(f"Retryable error: {e}, will retry")
    raise  # 让上层重试逻辑处理
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    return None  # 优雅降级
except Exception as e:
    logger.exception(f"Unexpected error")  # 记录完整堆栈
    raise  # 致命错误不吞没
```

**具体优化点**:
- [ ] `llm.py` 使用 `logger.exception()` 替代 `logger.error()` 保留堆栈
- [ ] `workflow/single.py` 区分可恢复错误与致命错误
- [ ] 批处理状态保存等关键路径应 fail-fast
- [ ] 定义统一的异常层次结构（MarkitaiError, RetryableError, FatalError）

---

## P2: 中优先级

### 3. CLI.py 代码迁移至 Workflow

**现状**: `cli.py` 共 2335 行，包含处理逻辑与 CLI 参数解析混合

**目标架构**:
```python
# cli.py 只负责参数解析和调度
@app.command()
def convert(input_path: Path, output: Path, ...):
    config = ConfigManager.load(...)
    workflow = SingleFileWorkflow(config)
    result = await workflow.process(input_path, output)
    # CLI 只做结果展示和报告输出
```

**迁移计划**:

| 函数/逻辑 | 当前位置 | 目标位置 | 优先级 |
|-----------|----------|----------|--------|
| `analyze_images_with_llm` | `cli.py:1577` | workflow/single.py | P2-1 |
| 图片保存/压缩逻辑 | cli.py | ImageProcessor | P2-2 |
| Frontmatter 生成 | cli.py | workflow/single.py | P2-3 |
| 截图处理逻辑 | cli.py | workflow/single.py | P2-4 |
| `process_single_file` | `cli.py:1018` | workflow/single.py | P2-5 |

**当前进度**: `workflow/single.py` 已有 334 行，包含 `SingleFileWorkflow` 类框架

**涉及文件**: `cli.py`, `workflow/single.py`, `image.py`

---

### 4. 并发控制文档化

**现状**: 多层并发独立控制，用户难以理解总资源消耗

**当前配置** (`constants.py:34-36`):
```python
DEFAULT_IO_CONCURRENCY = 20         # I/O 操作并发
DEFAULT_LLM_CONCURRENCY = 10        # LLM API 并发
DEFAULT_BATCH_CONCURRENCY = 10      # 批处理文件并发
# _CONVERTER_MAX_WORKERS = min(cpu_count, 8)  # 转换线程池
```

**实际最大并发分析**:
```
总并发 ≈ min(batch_concurrency, llm_concurrency) + converter_workers
       ≈ 10 + 8 = 18 (因为 batch 内部等待 LLM)
```

**待办**:
- [ ] 在 `docs/spec.md` 添加并发模型说明图
- [ ] 添加资源估算公式到 CLI help
- [ ] 考虑添加 `--max-concurrent` 全局限制参数

---

## P3: 低优先级

### 5. 缓存配置优化

**现状** (`constants.py:72-80`):
- 内存缓存: 500MB (`MAX_TOTAL_IMAGES_SIZE`)
- SQLite 持久缓存: 1GB per layer (`DEFAULT_CACHE_SIZE_LIMIT`)

**待评估**:
- [ ] 缓存大小是否需要根据系统内存动态调整
- [ ] 添加缓存统计到 `markitai config` 输出
- [ ] 考虑 LRU 淘汰策略配置化

---

### 6. PDF 转换重复工作

**现状**: PDF 转换可能同时进行 OCR + 截图 + 嵌入图提取

```python
# converter/pdf.py 中的潜在重复
1. pymupdf4llm.to_markdown (提取文本 + 图片)
2. _render_pages_for_llm (再次渲染页面截图)
3. OCR 处理 (对截图再次 OCR)
```

**待优化**:
- [ ] 分析实际场景，确定是否有不必要的重复渲染
- [ ] 考虑共享页面渲染结果
- [ ] 添加性能指标日志便于分析

---

### 7. 测试覆盖率提升

**待添加测试**:
- [ ] `check_symlink_safety` 嵌套 symlink 测试（Linux/Mac）
- [ ] `resolve_output_path` UUID fallback 边界测试
- [ ] LLM 重试逻辑集成测试
- [ ] 批处理断点恢复端到端测试

---

## 架构改进建议（长期）

### A. 统一错误处理框架

```python
# markitai/exceptions.py (新建)
class MarkitaiError(Exception):
    """Base exception for all Markitai errors."""
    pass

class RetryableError(MarkitaiError):
    """Errors that can be retried (network, rate limit)."""
    pass

class ConfigurationError(MarkitaiError):
    """Configuration-related errors."""
    pass

class ConversionError(MarkitaiError):
    """Document conversion errors."""
    pass
```

### B. 插件式转换器架构

当前 `register_converter` 装饰器是良好起点，可扩展为：
- 动态加载外部转换器
- 转换器优先级/fallback 链
- 转换器健康检查

### C. 可观测性增强

- [ ] 添加 OpenTelemetry tracing 支持
- [ ] 结构化日志（当前已用 loguru serialize=True）
- [ ] 指标导出（处理速度、LLM 成本等）

---

## 参考资料

- 原始评审: [markitai-overall-optimization-4.md](markitai-overall-optimization-4.md)
- 技术规格: [../spec.md](../spec.md)
- LiteLLM 异常: https://docs.litellm.ai/docs/exception_mapping
