# 模块拆分实施任务

> 目标：将 `cli/main.py` (4,267 行) 和 `llm/processor.py` (4,367 行) 拆分为更小的模块

---

## 进度概览

| 阶段 | 任务 | 状态 |
|------|------|------|
| Phase 2 | 2.1 创建 llm/content.py | ✅ 完成 |
| Phase 2 | 2.2 创建 llm/vision.py | ✅ 完成 |
| Phase 2 | 2.3 创建 llm/document.py | ✅ 完成 |
| Phase 3 | 1.1 删除 CLI 重复代码 | ✅ 完成 |
| Phase 3 | 1.2 创建 cli/framework.py | ✅ 完成 |
| Phase 3 | 1.3 创建 cli/logging_config.py | ✅ 完成 |
| Phase 3 | 1.4 创建 cli/commands/deps.py | ✅ 完成 |
| Phase 3 | 1.5 创建 cli/processors/ 包 | ✅ 完成 |
| Phase 3 | 1.6 更新 cli/__init__.py | ✅ 完成 |
| Phase 4 | 最终验证 | ✅ 完成 |

---

## Phase 2: LLM 模块拆分（剩余）

### 任务 2.2: 创建 llm/vision.py

**目标**: 提取视觉分析相关方法到独立模块

**源文件**: `llm/processor.py`

**提取方法**:
| 方法名 | 行号范围 | 说明 |
|--------|---------|------|
| `analyze_image` | 2487-2582 | 分析单张图片 |
| `analyze_images_batch` | 2584-2683 | 批量分析图片 |
| `analyze_batch` | 2685-2945 | 批量分析（带缓存） |
| `_analyze_image_with_fallback` | 2957-3002 | 带回退的分析 |
| `_analyze_with_instructor` | 3004-3074 | Instructor 方式 |
| `_analyze_with_json_mode` | 3076-3144 | JSON 模式方式 |
| `_analyze_with_two_calls` | 3146-3232 | 双调用方式 |
| `extract_page_content` | 3234-3284 | 提取页面内容 |

**实现步骤**:
1. 创建 `llm/vision.py`，定义 `VisionMixin` 类
2. 将上述方法移入 `VisionMixin`
3. 在 `LLMProcessor` 类声明中添加 `VisionMixin` 继承
4. 删除 `processor.py` 中的原方法
5. 运行测试验证

**预计行数**: ~600 行

---

### 任务 2.3: 创建 llm/document.py

**目标**: 提取文档处理相关方法到独立模块

**源文件**: `llm/processor.py`

**提取方法**:
| 方法名 | 行号范围 | 说明 |
|--------|---------|------|
| `clean_markdown` | 2347-2485 | 清理 Markdown |
| `generate_frontmatter` | ~3285-3450 | 生成 frontmatter |
| `enhance_url_with_vision` | ~3450-3550 | URL 视觉增强 |
| `enhance_document_with_vision` | ~3550-3650 | 文档视觉增强 |
| `enhance_document_complete` | ~3650-3830 | 完整文档增强 |
| `_enhance_document_batched_simple` | 3913-3963 | 批量增强 |
| `process_document` | 3965-4100 | 处理文档 |
| `_process_document_combined` | ~4100-4200 | 组合处理 |
| `format_llm_output` | ~4200-4300 | 格式化输出 |
| `_validate_no_prompt_leakage` | ~4300-4367 | 验证无泄漏 |

**实现步骤**:
1. 创建 `llm/document.py`，定义 `DocumentMixin` 类
2. 将上述方法移入 `DocumentMixin`
3. 在 `LLMProcessor` 类声明中添加 `DocumentMixin` 继承
4. 删除 `processor.py` 中的原方法
5. 运行测试验证

**预计行数**: ~800 行

---

## Phase 3: CLI 模块拆分

### 任务 1.1: 删除 CLI 重复代码

**目标**: 删除与 `utils/` 模块重复的代码

**删除内容**:
| 内容 | 行号范围 | 替代方案 |
|------|---------|---------|
| `ProgressReporter` 类 | 83-149 | 使用 `utils/progress.py` |
| `is_url()` | ~150-160 | 使用 `utils/cli_helpers.py` |
| `url_to_filename()` | ~160-190 | 使用 `utils/cli_helpers.py` |
| `_sanitize_filename()` | ~190-220 | 使用 `utils/cli_helpers.py` |
| `compute_task_hash()` | ~220-260 | 使用 `utils/cli_helpers.py` |
| `get_report_file_path()` | ~260-300 | 使用 `utils/cli_helpers.py` |

**实现步骤**:
1. 在 `cli/main.py` 顶部添加导入：
   ```python
   from markitai.utils.progress import ProgressReporter
   from markitai.utils.cli_helpers import (
       is_url, url_to_filename, sanitize_filename,
       compute_task_hash, get_report_file_path,
   )
   ```
2. 删除重复的类/函数定义
3. 将 `_sanitize_filename` 调用改为 `sanitize_filename`
4. 运行测试验证

**预计减少**: ~220 行

---

### 任务 1.2: 创建 cli/framework.py

**目标**: 提取 Click 框架扩展类

**源文件**: `cli/main.py`

**提取内容**:
| 内容 | 行号范围 | 说明 |
|------|---------|------|
| `_OPTIONS_WITH_VALUES` | ~300-304 | 需要值的选项列表 |
| `MarkitaiGroup` 类 | 304-428 | 自定义 Click Group |

**实现步骤**:
1. 创建 `cli/framework.py`
2. 移入 `MarkitaiGroup` 类和相关常量
3. 在 `cli/main.py` 中导入使用
4. 运行测试验证

**预计行数**: ~130 行

---

### 任务 1.3: 创建 cli/logging_config.py

**目标**: 提取日志配置相关代码

**源文件**: `cli/main.py`

**提取内容**:
| 内容 | 行号范围 | 说明 |
|------|---------|------|
| `LoggingContext` 类 | 429-478 | 日志上下文管理 |
| `InterceptHandler` 类 | 480-503 | 日志拦截器 |
| `setup_logging()` | 505-576 | 日志设置函数 |
| `print_version()` | 577-720 | 版本信息打印 |

**实现步骤**:
1. 创建 `cli/logging_config.py`
2. 移入上述类和函数
3. 在 `cli/main.py` 中导入使用
4. 运行测试验证

**预计行数**: ~300 行

---

### 任务 1.4: 创建 cli/commands/deps.py

**目标**: 提取依赖检查命令

**源文件**: `cli/main.py`

**提取内容**:
| 内容 | 行号范围 | 说明 |
|------|---------|------|
| `check_deps()` 命令 | 1529-1870 | 检查依赖命令 |

**实现步骤**:
1. 创建 `cli/commands/deps.py`
2. 移入 `check_deps` 命令及其辅助函数
3. 在 `cli/main.py` 中注册命令
4. 运行测试验证

**预计行数**: ~350 行

---

### 任务 1.5: 创建 cli/processors/ 包

**目标**: 提取处理器函数到独立模块

**目标结构**:
```
cli/processors/
├── __init__.py      # 重导出
├── file.py          # 文件处理
├── url.py           # URL 处理
├── llm.py           # LLM 处理
├── validators.py    # 验证器
└── batch.py         # 批量处理
```

#### 子任务 1.5.1: 创建 processors/file.py (~200 行)
| 内容 | 行号范围 |
|------|---------|
| `process_single_file()` | 1871-2066 |

#### 子任务 1.5.2: 创建 processors/url.py (~800 行)
| 内容 | 行号范围 |
|------|---------|
| `process_url()` | 2067-2503 |
| `process_url_batch()` | 2504-2824 |
| `_build_multi_source_content()` | 内嵌 |
| `_process_url_with_vision()` | 内嵌 |

#### 子任务 1.5.3: 创建 processors/llm.py (~350 行)
| 内容 | 行号范围 |
|------|---------|
| `process_with_llm()` | 2917-3012 |
| `analyze_images_with_llm()` | 3039-3205 |
| `enhance_document_with_vision()` | 3206-3267 |
| `_format_standalone_image_markdown()` | 内嵌 |

#### 子任务 1.5.4: 创建 processors/validators.py (~200 行)
| 内容 | 行号范围 |
|------|---------|
| `check_vision_model_config()` | ~2825-2870 |
| `check_agent_browser_for_urls()` | ~2870-2916 |
| `warn_case_sensitivity_mismatches()` | ~3268-3400 |

#### 子任务 1.5.5: 创建 processors/batch.py (~420 行)
| 内容 | 行号范围 |
|------|---------|
| `process_batch()` | 3871-4267 |
| `_create_process_file()` | 内嵌 |
| `_create_url_processor()` | 内嵌 |

**实现步骤**:
1. 按子任务顺序创建各模块
2. 创建 `processors/__init__.py` 统一导出
3. 更新 `cli/main.py` 导入使用
4. 每个子任务后运行测试验证

---

### 任务 1.6: 更新 cli/__init__.py

**目标**: 更新导出以保持向后兼容

**更新内容**:
```python
from markitai.cli.main import app
from markitai.utils.progress import ProgressReporter
from markitai.utils.cli_helpers import (
    is_url,
    url_to_filename,
    sanitize_filename,
    compute_task_hash,
    get_report_file_path,
)
from markitai.cli.processors.validators import warn_case_sensitivity_mismatches

__all__ = [
    "app",
    "ProgressReporter",
    "is_url",
    "url_to_filename",
    "sanitize_filename",
    "warn_case_sensitivity_mismatches",
    "compute_task_hash",
    "get_report_file_path",
]
```

---

## Phase 4: 最终验证

### 验证清单

- [x] 所有测试通过: `uv run pytest -v` (692 passed, 1 skipped)
- [x] 类型检查通过: `uv run pyright packages/markitai/src` (0 errors, 0 warnings)
- [x] Lint 检查通过: `uv run ruff check && ruff format --check`
- [x] 导入验证通过:
  ```bash
  uv run python -c "
  from markitai.cli import app, ProgressReporter, is_url, sanitize_filename
  from markitai.llm import LLMProcessor, ImageAnalysis, protect_content
  print('All imports OK')
  "
  ```

### 最终结果

| 文件 | 拆分前 | 拆分后 | 目标 | 状态 |
|------|--------|--------|------|------|
| cli/main.py | 4,267 行 | 1,013 行 | ~800 行 | ✅ 接近目标 |
| llm/processor.py | 4,875 行 | 2,320 行 | ~1,200 行 | ✅ 大幅减少 |

### 新建模块行数

**CLI 模块**:
| 文件 | 行数 |
|------|------|
| cli/framework.py | 130 |
| cli/logging_config.py | 181 |
| cli/commands/deps.py | 367 |
| cli/processors/file.py | 225 |
| cli/processors/url.py | 904 |
| cli/processors/llm.py | 383 |
| cli/processors/validators.py | 205 |
| cli/processors/batch.py | 867 |

**LLM 模块**:
| 文件 | 行数 |
|------|------|
| llm/content.py | 602 |
| llm/vision.py | 862 |
| llm/document.py | 1,338 |

---

## 命名规范

### 公开 API（无下划线前缀）
- `sanitize_filename`, `warn_case_sensitivity_mismatches`
- `protect_content`, `unprotect_content`, `fix_malformed_image_refs`
- `clean_frontmatter`, `smart_truncate`, `split_text_by_pages`

### 内部方法（保留下划线前缀）
- `_analyze_image_with_fallback`, `_analyze_with_instructor`
- `_enhance_with_frontmatter`, `_process_document_combined`
- `_validate_no_prompt_leakage`
