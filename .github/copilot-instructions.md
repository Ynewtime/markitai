# Markitai 项目指南

## 项目概述

Markitai 是一个带有原生 LLM 增强支持的专业 Markdown 转换器。

- **Python 版本**: 3.11-3.13（开发使用 3.13）
- **包管理器**: uv
- **构建系统**: hatchling
- **项目结构**: Monorepo (packages/markitai/)

---

## 环境检测

- 开始工作前，先确认好你所在的系统（Windows|Linux|MacOS），选择对应的执行命令
- Linux/macOS 使用 `.sh` 脚本，Windows 使用 `.ps1` 脚本

---

## 开发偏好

- Keep plan mode concise, remove unnecessary grammar
- Ask me questions about design/implementation decisions before coding
- Use only English for ASCII arts
- 请尽可能使用最新的依赖，如果对依赖是否是最新版本（结合当下的时间戳）有疑义，请调用 agent-browser 查询最新版本信息

---

## 语言规则

- 始终使用中文回复用户
- 错误信息和解释都用中文输出
- 代码注释使用英文
- 文档字符串使用英文（Google 风格）

---

## 代码规范

### 类型标注
- 所有函数必须有类型标注
- 使用现代语法: `str | None` 而非 `Optional[str]`
- 文件开头添加 `from __future__ import annotations`

### 代码风格
- 行长度: 88 字符
- 引号风格: 双引号
- 缩进: 4 空格
- Linter: Ruff
- 类型检查: Pyright

### 错误处理
- 使用 loguru 进行日志记录
- 错误信息需要有上下文和解决方案
- 使用自定义异常处理领域错误

---

## 测试要求

### 运行测试
```bash
uv run pytest                           # 运行所有测试
uv run pytest --cov=markitai            # 带覆盖率
uv run pytest tests/unit -v             # 仅单元测试
uv run pytest tests/integration -v      # 仅集成测试
uv run pytest -m "not slow"             # 跳过慢速测试（OCR ~40s）
uv run pytest -m "not network"          # 跳过网络测试
```

### 测试规范
- 新代码必须有对应的测试
- 目标覆盖率: 80%+
- 异步测试使用 `@pytest.mark.asyncio`
- Mock 外部依赖（LLM、网络请求等）

---

## Git 工作流

### 提交规范
```
<type>: <description>

[optional body]

[optional footer]
```

**类型**:
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具链

### 分支策略
- `main`: 稳定分支
- `feat/<name>`: 功能分支
- `fix/<name>`: 修复分支

---

## 常用命令

```bash
# 环境管理
uv sync                     # 安装依赖
uv sync --all-extras        # 安装所有可选依赖

# 代码质量
uv run ruff check --fix     # Lint 并修复
uv run ruff format          # 格式化
uv run pyright              # 类型检查

# 测试
uv run pytest               # 运行测试
uv run pytest --cov         # 带覆盖率

# CLI
uv run markitai --help      # 查看帮助
uv run markitai <file>      # 转换文件
uv run markitai --llm       # 启用 LLM 增强
```

---

## 工具

### 浏览器自动化（web/url fetch）

Use `agent-browser` for web automation. Run `agent-browser --help` for all commands.

Core workflow:

1. `agent-browser open <url>` - Navigate to page
2. `agent-browser snapshot -i` - Get interactive elements with refs (@e1, @e2)
3. `agent-browser click @e1` / `fill @e2 "text"` - Interact using refs
4. Re-snapshot after page changes

---

## AI 工具集成

### Claude Code CLI
项目原生支持通过 Claude Agent SDK 使用 Claude Code CLI：

```python
# 配置使用 Claude Code
{
  "llm": {
    "model_list": [{
      "model_name": "default",
      "litellm_params": {
        "model": "claude-agent/sonnet"
      }
    }]
  }
}
```

**支持的模型别名**:
- `claude-agent/sonnet` → 最新 Sonnet
- `claude-agent/opus` → 最新 Opus
- `claude-agent/haiku` → 最新 Haiku

### GitHub Copilot CLI
项目也支持通过 Copilot SDK 使用 GitHub Copilot：

```python
# 配置使用 Copilot
{
  "llm": {
    "model_list": [{
      "model_name": "default",
      "litellm_params": {
        "model": "copilot/claude-sonnet-4.5"
      }
    }]
  }
}
```

### 安装可选依赖
```bash
uv sync --extra claude-agent    # Claude Agent SDK
uv sync --extra copilot         # Copilot SDK
uv sync --all-extras            # 全部
```

---

## 项目结构

```
packages/markitai/src/markitai/
├── cli/             # CLI 包 (Click)
│   ├── main.py      # 主 CLI 入口
│   └── commands/    # 子命令 (cache, config)
├── llm/             # LLM 集成包 (LiteLLM)
│   ├── processor.py # LLM 处理器
│   ├── cache.py     # 响应缓存
│   ├── models.py    # 模型定义
│   └── types.py     # 类型定义
├── providers/       # LLM 提供商 (Claude Agent, Copilot)
├── workflow/        # 处理工作流 (core, single, helpers)
├── converter/       # 格式转换器 (PDF, Office, Image)
├── fetch.py         # URL 抓取
├── batch.py         # 批量处理
├── image.py         # 图像处理
├── config.py        # 配置管理 (Pydantic)
└── utils/           # 工具函数
```

---

## 代码审查标准

### 必须检查项
- [ ] 类型标注完整
- [ ] 有对应的测试
- [ ] 通过 Ruff lint
- [ ] 通过 Pyright 类型检查
- [ ] 文档字符串完整（公开函数）

### 建议检查项
- [ ] 错误处理有上下文
- [ ] 日志记录适当
- [ ] 无硬编码配置
- [ ] 考虑跨平台兼容性
