# MarkIt Agent 指南

本仓库包含 **MarkIt** 的源代码，这是一个使用 Python 构建的智能文档转 Markdown 工具。

## 1. 构建、Lint 和测试

### 环境与虚拟环境
- **Python 版本**: 3.12+
- **依赖管理**: **优先使用 `uv`**（已包含 `uv.lock`）。
- **配置**: `pyproject.toml` 是核心配置文件。

**⚠️ 重要：激活虚拟环境**
在执行任何 Python、`uv` 或 `markit` 相关命令之前，**必须**先根据操作系统激活虚拟环境：

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 常用命令

*确保已激活虚拟环境后执行以下命令：*

| 操作 | 命令 | 说明 |
|--------|---------|-------------|
| **安装依赖** | `uv sync --all-extras` | 使用 uv 同步全量依赖（包含所有可选组件）。 |
| **Lint** | `ruff check .` | 运行代码检查（包含 isort）。 |
| **格式化** | `ruff format .` | 自动格式化代码。 |
| **类型检查** | `mypy .` | 运行静态类型检查（严格模式）。 |
| **测试** | `pytest` | 运行所有测试。 |
| **构建** | `hatch build` | 构建软件包。 |

### 测试详情
- **测试框架**: `pytest`
- **配置**: `pyproject.toml` 中的 `[tool.pytest.ini_options]`
- **位置**: `tests/` 目录（分为 `unit/` 和 `integration/`）。

**运行特定测试:**
```bash
# 运行单个测试文件
pytest tests/unit/test_cli.py

# 运行特定测试用例
pytest tests/unit/test_cli.py::test_version_command

# 运行匹配关键字的测试
pytest -k "conversion"

# 运行并显示详细输出和简短回溯
pytest -v --tb=short
```

**关键 Fixtures (`tests/conftest.py`):**
- `temp_dir`: 提供临时目录路径（`Path` 对象）。
- `sample_text_file`: 创建一个示例 `.txt` 文件。
- `sample_markdown_file`: 创建一个示例 `.md` 文件。
- `cleanup_input_output`: 每次测试后自动清理 `input/output`。

---

## 2. 代码风格与规范

### 格式化
- **行长**: 100 字符（由 `ruff` 强制执行）。
- **引号**: 首选双引号 `"`。
- **缩进**: 4 个空格。

### 导入 (Imports)
- **排序**: `isort`（通过 `ruff`）。
- **顺序**: 标准库 -> 第三方库 -> 本地库 (`markit`)。
- **风格**: 首选绝对导入（例如 `from markit.core import ...`）。

### 类型提示 (Type Hinting)
- **严格程度**: 高 (`disallow_untyped_defs = true`)。
- **语法**: 使用现代 Python 3.10+ 语法（例如使用 `str | None` 代替 `Optional[str]`，`list[str]` 代替 `List[str]`）。
- **库**: Typer CLI 参数使用 `typing.Annotated`。

### 命名规范
- **变量/函数**: `snake_case`（例如 `convert_document`, `file_path`）。
- **类**: `PascalCase`（例如 `ConversionPipeline`, `MarkitError`）。
- **常量**: `UPPER_SNAKE_CASE`（例如 `PROJECT_ROOT`）。
- **私有**:以此为前缀 `_`（例如 `_process_image`）。

### 错误处理
- **基础异常**: `MarkitError`（位于 `markit/exceptions.py`）。
- **继承体系**: 继承 `MarkitError` 以实现特定领域的错误（例如 `ConversionError`, `LLMError`, `ConfigurationError`）。
- **模式**: 捕获特定异常；将低级错误（IOError 等）包装在带有上下文的 `MarkitError` 子类中。

### 路径处理
- **库**: **始终**使用 `pathlib.Path`。**不要**使用 `os.path`。
- **类型**: 类型提示应使用 `Path`（例如 `def process(path: Path) -> None:`）。

### 库与框架
- **CLI**: `typer` 用于命令定义，`rich` 用于输出格式化。
- **配置**: `pydantic-settings` and `pydantic` 模型。
- **异步**: `anyio` 用于异步操作。
- **日志**: `structlog`。

### 文档字符串 (Docstrings)
- **风格**: 首选 Google 风格文档字符串。
- **内容**: 简短的摘要行，后跟详细信息，以及 `Args:`、`Returns:` 和 `Raises:`（如果适用）。

```python
def convert_document(file_path: Path, output_dir: Path) -> Path:
    """Convert a document to Markdown.

    Args:
        file_path: Absolute path to the source file.
        output_dir: Directory where the output should be saved.

    Returns:
        Path to the generated Markdown file.

    Raises:
        ConversionError: If the file format is not supported or conversion fails.
    """
    ...
```

---

## 3. 文档同步规则

**重要**：更新文档时必须同步更新关联文档：

| 主文档 | 关联文档 | 说明 |
|--------|----------|------|
| `README.md` | `README_CN.md` | 中英文内容保持同步 |
| `CLAUDE.md` | `AGENTS.md` | 开发规范保持一致 |
| `docs/ROADMAP.md` | - | 任务进展及时更新 |

**操作要求**：
- 更新 README 后，同步更新 README_CN
- 新增开发规范时，同步更新 CLAUDE.md 和 AGENTS.md
- 完成任务后，更新 ROADMAP.md 中对应任务批次的进展
```
