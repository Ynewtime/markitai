# 贡献指南

感谢你对 Markitai 项目的兴趣！本文档将帮助你了解如何参与项目开发。

## 开发环境设置

### 前置要求

- Python 3.11-3.13
- [uv](https://docs.astral.sh/uv/) 包管理器
- Git

### 克隆仓库

```bash
git clone https://github.com/Ynewtime/markitai.git
cd markitai
```

### 安装依赖

```bash
# 安装所有依赖（包括开发依赖）
uv sync

# 安装可选的 LLM 提供商 SDK
uv sync --all-extras
```

### 安装 Pre-commit 钩子

```bash
uv run pre-commit install
```

### 验证安装

```bash
# 运行测试
uv run pytest

# 运行 lint
uv run ruff check

# 运行类型检查
uv run pyright
```

---

## 代码风格规范

### Python 版本

- 目标版本: Python 3.13
- 兼容版本: Python 3.11+

### 格式化和 Lint

项目使用 [Ruff](https://docs.astral.sh/ruff/) 进行代码格式化和 lint：

```bash
# 检查
uv run ruff check

# 自动修复
uv run ruff check --fix

# 格式化
uv run ruff format
```

### 类型标注

- 所有函数必须有类型标注
- 使用现代语法: `str | None` 而非 `Optional[str]`
- 文件开头添加 `from __future__ import annotations`

```python
from __future__ import annotations

def process(text: str | None = None) -> dict[str, Any]:
    ...
```

### 文档字符串

使用 Google 风格：

```python
def convert(path: Path, options: ConvertOptions) -> ConversionResult:
    """Convert a file to Markdown.

    Args:
        path: Path to the input file.
        options: Conversion options.

    Returns:
        The conversion result containing markdown content and metadata.

    Raises:
        FileNotFoundError: If the input file does not exist.
        UnsupportedFormatError: If the file format is not supported.
    """
```

### 导入顺序

由 Ruff 自动管理，遵循 isort 规则：

```python
# 标准库
import asyncio
from pathlib import Path

# 第三方库
import click
from pydantic import BaseModel

# 本地模块
from markitai.config import get_config
```

---

## 提交规范

### Commit 消息格式

```
<type>: <description>

[optional body]

[optional footer]
```

### 类型

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档更新 |
| `style` | 代码格式（不影响功能） |
| `refactor` | 重构（不新增功能或修复 bug） |
| `perf` | 性能优化 |
| `test` | 测试相关 |
| `chore` | 构建/工具链/CI |

### 示例

```
feat: add support for EPUB format conversion

- Add EpubConverter class
- Register .epub extension
- Add unit tests

Closes #123
```

```
fix: handle empty PDF pages correctly

Previously, empty pages would cause an IndexError.
Now they are skipped with a warning.
```

---

## Pull Request 流程

### 1. 创建分支

```bash
# 功能分支
git checkout -b feat/your-feature

# 修复分支
git checkout -b fix/your-fix
```

### 2. 开发

- 编写代码
- 添加测试
- 更新文档（如需要）

### 3. 本地验证

```bash
# 运行所有检查
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest
```

### 4. 提交

```bash
git add .
git commit -m "feat: your feature description"
```

### 5. 推送并创建 PR

```bash
git push -u origin feat/your-feature
```

然后在 GitHub 上创建 Pull Request。

### PR 检查清单

- [ ] 代码通过所有 CI 检查
- [ ] 新功能有对应的测试
- [ ] 文档已更新（如需要）
- [ ] Commit 消息符合规范
- [ ] PR 描述清晰说明了变更内容

---

## 测试要求

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest tests/unit/test_config.py

# 运行带覆盖率
uv run pytest --cov=markitai

# 并行运行
uv run pytest -n auto
```

### 快速测试（跳过慢速测试）

部分测试因 OCR 处理或网络请求较慢。使用 pytest 标记跳过它们：

```bash
# 跳过慢速测试（如 OCR 处理 ~40s）
uv run pytest -m "not slow"

# 跳过需要网络的测试
uv run pytest -m "not network"

# 同时跳过慢速和网络测试
uv run pytest -m "not slow and not network"

# 仅运行 CLI 测试（快速）
uv run pytest packages/markitai/tests/integration/test_cli_full.py
```

### 测试标记

| 标记 | 说明 |
|------|------|
| `@pytest.mark.slow` | 耗时 >10s 的测试（如 OCR）|
| `@pytest.mark.network` | 需要网络访问的测试 |

### 测试结构

```
packages/markitai/tests/
├── unit/              # 单元测试
│   ├── test_config.py
│   ├── test_llm.py
│   └── ...
├── integration/       # 集成测试
│   ├── test_cli.py
│   └── ...
└── conftest.py        # 共享 fixtures
```

### 编写测试

```python
import pytest
from markitai.config import resolve_env_value

class TestResolveEnvValue:
    def test_plain_value(self):
        """Plain values should be returned as-is."""
        assert resolve_env_value("hello") == "hello"

    def test_env_value(self, monkeypatch):
        """env: prefix should resolve environment variables."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        assert resolve_env_value("env:TEST_VAR") == "test_value"

    def test_missing_env_strict(self):
        """Missing env var should raise in strict mode."""
        with pytest.raises(EnvVarNotFoundError):
            resolve_env_value("env:NONEXISTENT", strict=True)
```

### 异步测试

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

---

## 文档要求

### 代码文档

- 所有公开函数必须有文档字符串
- 复杂逻辑需要内联注释
- 使用类型标注增强可读性

### 用户文档

用户文档位于 `website/` 目录：

```
website/
├── guide/
│   ├── getting-started.md
│   ├── configuration.md
│   └── cli.md
└── zh/                # 中文版本
    └── guide/
```

### 构建文档

```bash
cd website
pnpm install
pnpm docs:dev    # 开发模式
pnpm docs:build  # 构建
```

---

## 项目结构

```
markitai/
├── packages/markitai/     # 主包
│   ├── src/markitai/      # 源代码
│   │   ├── cli/           # CLI 包
│   │   ├── llm/           # LLM 集成包
│   │   ├── providers/     # 自定义 LLM 提供商
│   │   ├── converter/     # 格式转换器
│   │   ├── workflow/      # 处理工作流
│   │   └── utils/         # 工具函数
│   └── tests/             # 测试
├── scripts/               # 安装脚本
├── docs/                  # 内部文档
├── website/               # 用户文档
├── pyproject.toml         # 工作区配置
└── CONTRIBUTING.md        # 本文件
```

---

## 获取帮助

- **Issues**: [GitHub Issues](https://github.com/Ynewtime/markitai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Ynewtime/markitai/discussions)

---

## 许可证

通过提交 Pull Request，你同意你的贡献将按照项目的 MIT 许可证进行授权。
