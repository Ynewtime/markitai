# Markitai 任务规划

## 20260127-需求规划

### 需求概览

1. **URL 缓存策略优化** - 基于 HTTP Header (Last-Modified/ETag) 的条件缓存
2. **LLM 本地客户端支持** - 支持 Claude Code CLI 和 GitHub Copilot SDK

---

## 需求 1：URL HTTP 条件缓存

### 设计方案

**核心思路**：抓取 URL 时记录 HTTP Header (ETag/Last-Modified)，下次请求时发送条件 GET 请求，304 响应则命中缓存。

**优化实现（单次请求）**：
1. 数据库表新增字段：`etag`, `last_modified`
2. 使用 httpx 发送条件 GET 请求（带 If-None-Match/If-Modified-Since）
3. 304 响应直接返回缓存，200 响应保存到临时文件让 markitdown 处理
4. **无额外网络往返**

**策略兼容性**：
| 策略 | 支持 | 说明 |
|------|------|------|
| static | ✓ | 条件 GET + 临时文件 |
| browser | ✗ | Playwright 无法获取 HTTP Header |
| jina | ✗ | 第三方服务 |

### 关键修改

**文件**: `packages/markitai/src/markitai/fetch.py`

1. `FetchCache._init_db()` - 数据库迁移，新增 `etag`, `last_modified` 字段
2. `FetchCache.get_with_validators()` - 返回缓存结果及 HTTP 验证器
3. `FetchCache.set_with_validators()` - 存储时保存 HTTP Header
4. `fetch_with_static_conditional()` - 新函数，替代原 `fetch_with_static`
5. `fetch_url()` - 集成条件缓存逻辑

### 流程

```
fetch_url(url)
  └─ cache.get_with_validators(url)
       ├─ 有缓存 + validators?
       │     └─ httpx.get(url, headers={If-None-Match, If-Modified-Since})
       │           ├─ 304? → 返回缓存（更新 accessed_at）
       │           └─ 200? → 保存临时文件 → markitdown.convert(temp) → 更新缓存
       └─ 无缓存?
             └─ httpx.get(url) → 保存临时文件 → markitdown.convert(temp) → 保存缓存 + Header
```

### 核心代码逻辑

```python
async def fetch_with_static_conditional(
    url: str,
    cached_etag: str | None = None,
    cached_last_modified: str | None = None,
) -> tuple[FetchResult | None, dict]:
    """条件 GET 请求，单次网络往返"""
    headers = {}
    if cached_etag:
        headers["If-None-Match"] = cached_etag
    if cached_last_modified:
        headers["If-Modified-Since"] = cached_last_modified

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, follow_redirects=True)

        # 304 Not Modified - 使用缓存
        if response.status_code == 304:
            return None, {"not_modified": True}

        # 保存响应头
        response_headers = {
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }

        # 保存到临时文件，让 markitdown 处理
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            f.write(response.content)
            temp_path = f.name

    md = MarkItDown()
    result = md.convert(temp_path)
    os.unlink(temp_path)

    return FetchResult(...), response_headers
```

---

## 需求 2：LLM 本地客户端支持

### 设计方案

**核心思路**：通过 LiteLLM CustomLLM 接口实现自定义 provider，完全兼容现有 Router 和配置体系。

**Claude Code 实现方式（优化方案：使用 Agent SDK 替代 CLI subprocess）**：

| 方案 | 跨平台 | 实现复杂度 | 稳定性 |
|------|--------|----------|--------|
| CLI subprocess | 需处理 Windows 编码 | 中 | 低 |
| **Agent SDK** | 原生 Python，无平台问题 | 低 | 高 |

**推荐使用 Agent SDK**：
1. 设置 `allowed_tools=[]` 实现纯 LLM 对话（不使用任何工具）
2. 原生 Python async/await，跨平台兼容
3. 自动使用 Claude Code CLI 认证（订阅额度）
4. 返回完整的 usage 统计

**实现方式**：
1. 新建 `providers/` 模块
2. `ClaudeAgentProvider` - 使用 claude-agent-sdk（推荐）
3. `CopilotProvider` - 调用 GitHub Copilot SDK
4. 配置使用 `model: "claude-agent/xxx"` 或 `model: "copilot/xxx"` 前缀

### 关键修改

**新建文件**：
- `packages/markitai/src/markitai/providers/__init__.py`
- `packages/markitai/src/markitai/providers/claude_agent.py`（使用 Agent SDK）
- `packages/markitai/src/markitai/providers/copilot.py`

**修改文件**：
- `packages/markitai/src/markitai/llm.py` - 添加 provider 自动注册
- `packages/markitai/src/markitai/config.py` - 扩展 `local_client_timeout` 配置
- `pyproject.toml` - 添加 `claude-agent-sdk` 可选依赖

### 配置示例

```json
{
  "llm": {
    "model_list": [
      {
        "model_name": "default",
        "litellm_params": {
          "model": "claude-agent/sonnet",
          "weight": 10
        }
      },
      {
        "model_name": "default",
        "litellm_params": {
          "model": "copilot/gpt-4.1",
          "weight": 5
        }
      }
    ]
  }
}
```

### Claude Agent Provider 实现

```python
"""Claude Agent SDK provider for LiteLLM."""

from typing import Any
import litellm
from litellm import CustomLLM, ModelResponse


class ClaudeAgentProvider(CustomLLM):
    """Custom LiteLLM provider using Claude Agent SDK."""

    def __init__(self, timeout: int = 120) -> None:
        self.timeout = timeout
        self._validate_sdk()

    def _validate_sdk(self) -> None:
        try:
            from claude_agent_sdk import query  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Claude Agent SDK not installed. "
                "Install with: pip install claude-agent-sdk"
            )

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Convert OpenAI-style messages to prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                )
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)

    async def acompletion(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ModelResponse:
        """Async completion using Claude Agent SDK."""
        from claude_agent_sdk import query, ClaudeAgentOptions
        from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

        model_name = model.replace("claude-agent/", "")
        prompt = self._messages_to_prompt(messages)

        result_text = ""
        usage_info = {}

        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                allowed_tools=[],  # 纯 LLM 对话，不使用工具
                max_turns=1,
                model=model_name,
            )
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        result_text += block.text
            elif isinstance(message, ResultMessage):
                usage_info = message.usage or {}

        return litellm.ModelResponse(
            id=f"claude-agent-{id(message)}",
            choices=[{
                "message": {"role": "assistant", "content": result_text},
                "finish_reason": "stop",
                "index": 0,
            }],
            model=model_name,
            usage={
                "prompt_tokens": usage_info.get("input_tokens", 0),
                "completion_tokens": usage_info.get("output_tokens", 0),
                "total_tokens": usage_info.get("input_tokens", 0) + usage_info.get("output_tokens", 0),
            },
        )

    def completion(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ModelResponse:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.acompletion(model, messages, **kwargs))
```

### 认证方式

**自动使用 Claude Code CLI 认证**：
- 用户先运行 `claude` 命令完成认证
- Agent SDK 自动使用此认证
- 订阅用户可使用订阅额度

**或使用 API Key**：
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Copilot Provider 实现

```python
"""GitHub Copilot SDK provider for LiteLLM."""

import asyncio
from typing import Any, AsyncIterator, Iterator

import litellm
from litellm import CustomLLM, ModelResponse
from litellm.types.utils import GenericStreamingChunk


class CopilotProvider(CustomLLM):
    """Custom LiteLLM provider using GitHub Copilot SDK."""

    def __init__(self, timeout: int = 120) -> None:
        self.timeout = timeout
        self._client = None
        self._validate_sdk()

    def _validate_sdk(self) -> None:
        """Check if Copilot SDK is available."""
        try:
            from copilot import CopilotClient  # noqa: F401
        except ImportError:
            raise RuntimeError("GitHub Copilot SDK not installed. Install with: pip install github-copilot-sdk")

    async def _get_client(self):
        """Get or create Copilot client."""
        if self._client is None:
            from copilot import CopilotClient
            self._client = CopilotClient()
            await self._client.start()
        return self._client

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
            parts.append(content)
        return "\n\n".join(parts)

    def completion(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ModelResponse:
        return asyncio.get_event_loop().run_until_complete(self.acompletion(model, messages, **kwargs))

    async def acompletion(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> ModelResponse:
        model_name = model.replace("copilot/", "")
        prompt = self._messages_to_prompt(messages)
        client = await self._get_client()
        session = await client.create_session({"model": model_name})
        response = await asyncio.wait_for(session.send_and_wait({"prompt": prompt}), timeout=self.timeout)
        content = response.data.content if hasattr(response.data, "content") else str(response.data)
        estimated_input, estimated_output = len(prompt) // 4, len(content) // 4
        return litellm.ModelResponse(
            id=f"copilot-{id(response)}",
            choices=[{"message": {"role": "assistant", "content": content}, "finish_reason": "stop", "index": 0}],
            model=model_name,
            usage={"prompt_tokens": estimated_input, "completion_tokens": estimated_output, "total_tokens": estimated_input + estimated_output},
        )

    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        raise NotImplementedError("Copilot streaming not yet implemented")

    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        raise NotImplementedError("Copilot async streaming not yet implemented")

    async def close(self) -> None:
        if self._client:
            await self._client.stop()
            self._client = None
```

### Provider 注册

```python
"""providers/__init__.py"""

from markitai.providers.claude_code import ClaudeCodeProvider
from markitai.providers.copilot import CopilotProvider

__all__ = ["ClaudeCodeProvider", "CopilotProvider", "register_providers"]


def register_providers() -> None:
    """Register custom providers with LiteLLM."""
    import litellm

    claude_code_provider = ClaudeCodeProvider()
    litellm.custom_provider_map.append({"provider": "claude-code", "custom_handler": claude_code_provider})

    copilot_provider = CopilotProvider()
    litellm.custom_provider_map.append({"provider": "copilot", "custom_handler": copilot_provider})
```

### 限制

- 本地 provider 不支持 Vision（图片分析），需在 `_is_vision_model()` 中排除
- 不支持 streaming

---

## 验证方案

### 需求 1 测试
```bash
# 首次抓取（无缓存）
markitai https://example.com --verbose
# 观察日志：记录 ETag/Last-Modified

# 再次抓取（命中缓存）
markitai https://example.com --verbose
# 观察日志：304 Not Modified，使用缓存

# 强制刷新
markitai https://example.com --no-cache
```

### 需求 2 测试
```bash
# 使用 Claude Code 订阅
markitai document.pdf --llm
# 观察日志：claude-code provider 调用

# 查看 token 统计
markitai document.pdf --llm --verbose
```

---

## 实现顺序

1. **需求 1**: URL 条件缓存（static 策略）
2. **需求 2**: Claude Code Provider + Copilot Provider

## 单元测试

### 需求 1 测试用例
- `test_conditional_cache_304_not_modified` - 304 响应命中缓存
- `test_conditional_cache_200_updated` - 200 响应更新缓存
- `test_conditional_cache_no_validators` - 无 ETag/Last-Modified 时正常抓取
- `test_db_migration_add_etag_columns` - 数据库迁移

### 需求 2 测试用例
- `test_claude_code_provider_completion` - Claude Code CLI 调用
- `test_copilot_provider_completion` - Copilot SDK 调用
- `test_provider_auto_registration` - Provider 自动注册
- `test_vision_model_excludes_local_providers` - 本地 Provider 不支持 Vision

---

## 20260127-scripts：scripts 脚本安全审计与加固（方案 2）

### 范围

本次审计覆盖 `scripts/` 下全部脚本（8 个），逻辑上分为 4 组（用户版/开发者版 * sh/ps1，中英文镜像）：

- 用户版：`scripts/setup.sh`、`scripts/setup-zh.sh`、`scripts/setup.ps1`、`scripts/setup-zh.ps1`
- 开发者版：`scripts/setup-dev.sh`、`scripts/setup-dev-zh.sh`、`scripts/setup-dev.ps1`、`scripts/setup-dev-zh.ps1`

### 问题描述

1. 远程脚本直接执行
   - Shell：`curl ... | sh`（UV 自动安装）
   - PowerShell：`Invoke-RestMethod ... | Invoke-Expression`（UV 自动安装）

2. 默认交互选择偏“自动执行”
   - UV 自动安装默认 Yes（用户无感执行远程脚本）
   - Chromium 下载默认 Yes
   - Linux `agent-browser install --with-deps` 默认 Yes（可能触发系统依赖安装）

3. 用户版 Shell 在 `set -e` 下存在“跳过 UV 即退出”的功能性缺陷
   - `main()` 直接调用 `detect_uv`，而 `detect_uv` 在用户选择跳过时返回非 0，触发 `set -e` 直接终止
   - 结果：即使脚本具备 pipx/pip 回退安装逻辑，也无法执行到

4. Shell 探测逻辑在边界环境易被 `set -e` 放大
   - `python` 指向 Python2 时，f-string 探测命令会语法错误；后续对空变量做整数比较会直接退出
   - Node 版本解析异常时，同类整数比较也会导致脚本终止

5. 供应链与可复现性风险
   - `uv tool install markitai[all]` / `pipx install` / `pip install --user` / `npm install -g` 均未固定版本
   - 安装结果随时间漂移，且对上游包/脚本投毒更敏感

### 分析（安全性与可靠性）

**安全性**

- 高风险：远程代码执行链路（`curl|sh` / `irm|iex`）属于典型 RCE 入口；一旦上游变更或链路被劫持，影响直接落到用户机器。
- 中风险：`agent-browser install --with-deps` 可能触发系统包变更（间接 sudo/包管理器）；需要明确告知并默认不执行。
- 中风险：PATH 前置用户可写目录（`~/.local/bin` 等）在“root 执行脚本”场景下容易放大 PATH 劫持风险；至少需要强提示。
- 中风险：全局安装 npm 包（`npm -g`）以及 Python 工具安装不固定版本带来供应链漂移；建议提供可选版本固定。

**可靠性**

- 用户版 Shell 的 `detect_uv` 返回值与 `set -e` 组合会导致脚本不按设计工作；一旦把“默认 Yes”改为“默认 No”，该问题会频繁暴露。
- Python/Node 探测在非预期输出时容易退出，应在整数比较前做“纯数字校验”，异常则跳过候选。

### 结论

在保留“可自动安装”的前提下，采用“默认 N/false + 显式确认”的交互策略，可以显著降低误触发远程执行/系统改动的风险；同时必须修复用户版 Shell 在跳过 UV 时提前退出的问题，否则脚本会变得不可用。

### 改动清单（不改功能，只做默认值调整 + 鲁棒性加固）

#### 1) 所有高影响操作默认 N/false（交互层）

- UV 自动安装默认改为 N/false
  - Shell：`scripts/setup.sh`、`scripts/setup-zh.sh`、`scripts/setup-dev.sh`、`scripts/setup-dev-zh.sh`
  - PowerShell：`scripts/setup.ps1`、`scripts/setup-zh.ps1`、`scripts/setup-dev.ps1`、`scripts/setup-dev-zh.ps1`

- Chromium 下载默认改为 N/false
  - Shell：`scripts/setup.sh`、`scripts/setup-zh.sh`、`scripts/setup-dev.sh`、`scripts/setup-dev-zh.sh`
  - PowerShell：`scripts/setup.ps1`、`scripts/setup-zh.ps1`、`scripts/setup-dev.ps1`、`scripts/setup-dev-zh.ps1`

- Linux 系统依赖安装（`agent-browser install --with-deps`）默认改为 N
  - Shell：`scripts/setup.sh`、`scripts/setup-zh.sh`、`scripts/setup-dev.sh`、`scripts/setup-dev-zh.sh`

#### 2) 远程脚本执行增加“二次确认”（降低误触发）

- 在用户第一次选择“安装 UV”后，追加二次确认提示
  - Shell：明确提示将执行 `curl -LsSf ... | sh`
  - PowerShell：明确提示将执行 `irm ... | iex`（`Invoke-Expression`）

#### 3) 修复用户版 Shell：跳过/失败 UV 不应导致脚本退出

- 用户版 Shell：`detect_uv()` 在以下情况都应返回 0（继续后续 pipx/pip 安装流程）
  - 用户选择跳过 UV
  - UV 自动安装失败
- 涉及文件：`scripts/setup.sh`、`scripts/setup-zh.sh`
- 开发者版保持现状：UV 缺失应直接失败（开发依赖同步强依赖 uv）

#### 4) Shell 探测逻辑加固：避免 `set -e` 因边界输出崩溃

- Python 探测
  - 将版本输出改为兼容写法（避免 Python2 的 f-string 语法错误）
  - 在执行 `-eq/-ge` 前先校验 `major/minor` 是否为纯数字；否则跳过该候选命令
  - 涉及文件：`scripts/setup.sh`、`scripts/setup-zh.sh`、`scripts/setup-dev.sh`、`scripts/setup-dev-zh.sh`

- Node 探测
  - `major` 为空/非数字时不做整数比较，给出提示并返回“未满足要求/未检测到”
  - 涉及文件：同上

#### 5) 执行前置校验与安装后校验（提升可诊断性）

- 执行 `curl` 前校验 `curl` 是否存在；缺失则给出安装指引并走回退逻辑
- `npm install -g agent-browser` 后校验 `agent-browser` 是否可执行；不可执行时提示 npm global bin 加入 PATH 的方法
- 涉及文件：所有包含 agent-browser 安装逻辑的脚本

#### 6) Root/管理员执行风险提示

- Shell：脚本检测到 root 执行时打印强警告（PATH 前置用户可写目录 + root 场景风险更高）
- PowerShell：检测到管理员权限执行时同样打印警告
- 涉及文件：所有 8 个脚本

#### 7) 供应链可复现性增强（版本固定）

- 支持通过环境变量固定版本（默认使用最新稳定版）
- 环境变量：
  - `MARKITAI_VERSION` - markitai 版本，如 `0.3.2`
  - `AGENT_BROWSER_VERSION` - agent-browser 版本，如 `1.0.0`
  - `UV_VERSION` - uv 版本，如 `0.5.0`
- 涉及文件：所有 8 个脚本

### 实现细节

#### Python 版本探测兼容写法

```bash
# 兼容 Python2/3（避免 f-string）
detect_python() {
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            ver=$("$cmd" -c "import sys; v=sys.version_info; print('%d.%d' % (v[0], v[1]))" 2>/dev/null)
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)

            # 校验是否为纯数字再比较
            if ! [[ "$major" =~ ^[0-9]+$ ]] || ! [[ "$minor" =~ ^[0-9]+$ ]]; then
                continue  # 跳过该候选命令
            fi

            if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
                PYTHON_CMD="$cmd"
                return 0
            fi
        fi
    done
    return 1
}
```

#### Node 版本探测加固

```bash
detect_node() {
    if ! command -v node &>/dev/null; then
        return 1
    fi

    local ver
    ver=$(node --version 2>/dev/null | sed 's/^v//')
    local major
    major=$(echo "$ver" | cut -d. -f1)

    # 校验是否为纯数字
    if ! [[ "$major" =~ ^[0-9]+$ ]]; then
        echo "警告：无法解析 Node 版本号: $ver"
        return 1
    fi

    if [ "$major" -ge 18 ]; then
        return 0
    fi

    echo "警告：Node 版本 $ver 低于要求的 18.x"
    return 1
}
```

#### 二次确认提示文本

**Shell**：
```bash
confirm_remote_script() {
    local script_url="$1"
    local script_name="$2"

    echo ""
    echo "=========================================="
    echo "  ⚠️  警告：即将执行远程脚本"
    echo "=========================================="
    echo ""
    echo "  脚本来源: $script_url"
    echo "  用途: 安装 $script_name"
    echo ""
    echo "  此操作将从互联网下载并执行代码。"
    echo "  请确保您信任该来源。"
    echo ""
    read -p "确认执行？[y/N] " confirm
    case "$confirm" in
        [Yy]|[Yy][Ee][Ss]) return 0 ;;
        *) return 1 ;;
    esac
}

# 使用示例
if confirm_remote_script "https://astral.sh/uv/install.sh" "uv"; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
```

**PowerShell**：
```powershell
function Confirm-RemoteScript {
    param(
        [string]$ScriptUrl,
        [string]$ScriptName
    )

    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Yellow
    Write-Host "  ⚠️  警告：即将执行远程脚本" -ForegroundColor Yellow
    Write-Host "==========================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  脚本来源: $ScriptUrl"
    Write-Host "  用途: 安装 $ScriptName"
    Write-Host ""
    Write-Host "  此操作将从互联网下载并执行代码。"
    Write-Host "  请确保您信任该来源。"
    Write-Host ""

    $confirm = Read-Host "确认执行？[y/N]"
    return $confirm -match '^[Yy]'
}

# 使用示例
if (Confirm-RemoteScript -ScriptUrl "https://astral.sh/uv/install.ps1" -ScriptName "uv") {
    irm https://astral.sh/uv/install.ps1 | iex
}
```

#### Root/管理员检测

**Shell**：
```bash
warn_if_root() {
    if [ "$(id -u)" -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "  ⚠️  警告：检测到 root 权限执行"
        echo "=========================================="
        echo ""
        echo "  以 root 身份运行安装脚本存在以下风险："
        echo "  1. PATH 劫持：~/.local/bin 等用户目录可能被低权限用户写入"
        echo "  2. 远程代码执行风险被放大"
        echo ""
        echo "  建议：使用普通用户身份执行此脚本"
        echo ""
        read -p "是否继续？[y/N] " confirm
        case "$confirm" in
            [Yy]|[Yy][Ee][Ss]) return 0 ;;
            *) exit 1 ;;
        esac
    fi
}
```

**PowerShell**：
```powershell
function Test-AdminWarning {
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if ($isAdmin) {
        Write-Host ""
        Write-Host "==========================================" -ForegroundColor Yellow
        Write-Host "  ⚠️  警告：检测到管理员权限执行" -ForegroundColor Yellow
        Write-Host "==========================================" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  以管理员身份运行安装脚本存在以下风险："
        Write-Host "  1. 系统级更改可能影响所有用户"
        Write-Host "  2. 远程代码执行风险被放大"
        Write-Host ""
        Write-Host "  建议：使用普通用户身份执行此脚本"
        Write-Host ""

        $confirm = Read-Host "是否继续？[y/N]"
        if (-not ($confirm -match '^[Yy]')) {
            exit 1
        }
    }
}
```

#### 版本固定实现

**Shell**：
```bash
# 版本变量（可通过环境变量覆盖）
MARKITAI_VERSION="${MARKITAI_VERSION:-}"
AGENT_BROWSER_VERSION="${AGENT_BROWSER_VERSION:-}"
UV_VERSION="${UV_VERSION:-}"

install_markitai() {
    local pkg="markitai[all]"
    if [ -n "$MARKITAI_VERSION" ]; then
        pkg="markitai[all]==$MARKITAI_VERSION"
        echo "安装指定版本: $pkg"
    fi

    if command -v uv &>/dev/null; then
        uv tool install "$pkg"
    elif command -v pipx &>/dev/null; then
        pipx install "$pkg"
    else
        pip install --user "$pkg"
    fi
}

install_agent_browser() {
    local pkg="agent-browser"
    if [ -n "$AGENT_BROWSER_VERSION" ]; then
        pkg="agent-browser@$AGENT_BROWSER_VERSION"
        echo "安装指定版本: $pkg"
    fi
    npm install -g "$pkg"
}
```

**PowerShell**：
```powershell
# 版本变量（可通过环境变量覆盖）
$MarkitaiVersion = $env:MARKITAI_VERSION
$AgentBrowserVersion = $env:AGENT_BROWSER_VERSION
$UvVersion = $env:UV_VERSION

function Install-Markitai {
    $pkg = "markitai[all]"
    if ($MarkitaiVersion) {
        $pkg = "markitai[all]==$MarkitaiVersion"
        Write-Host "安装指定版本: $pkg"
    }

    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv tool install $pkg
    } elseif (Get-Command pipx -ErrorAction SilentlyContinue) {
        pipx install $pkg
    } else {
        pip install --user $pkg
    }
}

function Install-AgentBrowser {
    $pkg = "agent-browser"
    if ($AgentBrowserVersion) {
        $pkg = "agent-browser@$AgentBrowserVersion"
        Write-Host "安装指定版本: $pkg"
    }
    npm install -g $pkg
}
```

### 验证方案（改动完成后）

#### 功能测试

1. **用户版 Shell（setup.sh, setup-zh.sh）**
   - 未安装 uv 环境，默认回车跳过 UV，脚本应继续执行 pipx/pip 安装（不应提前退出）
   - 选择 Y 安装 UV，应出现二次确认；确认后执行远程脚本

2. **用户版 PowerShell（setup.ps1, setup-zh.ps1）**
   - 同上逻辑验证

3. **开发者版 Shell（setup-dev.sh, setup-dev-zh.sh）**
   - 未安装 uv 时默认回车跳过，应明确失败并退出（符合"开发必需"预期）

4. **开发者版 PowerShell（setup-dev.ps1, setup-dev-zh.ps1）**
   - 同上逻辑验证

#### 默认值测试

5. **UV 安装默认 N**
   - 所有脚本：直接回车应跳过 UV 安装

6. **Chromium 下载默认 N**
   - 所有脚本：直接回车应跳过 Chromium 下载

7. **Linux 系统依赖默认 N**
   - Shell 脚本：直接回车应跳过 `--with-deps`

#### 边界条件测试

8. **Python2 环境**
   - 系统只有 Python2 时，探测应跳过并提示未找到合适版本

9. **Node 版本异常**
   - Node 版本输出格式异常时，不应导致脚本崩溃

10. **Root/管理员执行**
    - root/管理员执行时应打印警告并要求确认

#### 版本固定测试

11. **MARKITAI_VERSION 生效**
    ```bash
    MARKITAI_VERSION=0.3.2 ./scripts/setup.sh
    ```

12. **AGENT_BROWSER_VERSION 生效**
    ```bash
    AGENT_BROWSER_VERSION=1.0.0 ./scripts/setup.sh
    ```

#### 安装后校验

13. **agent-browser 可执行性检查**
    - npm 安装后验证 `agent-browser --version` 可执行
    - 不可执行时提示 PATH 配置方法

---

### 任务拆分

#### Task 1: Shell 脚本公共函数库

**文件**: 新建 `scripts/lib.sh`

**内容**:
- `warn_if_root()` - root 执行警告
- `confirm_remote_script()` - 远程脚本二次确认
- `detect_python()` - Python 版本探测（兼容 Python2）
- `detect_node()` - Node 版本探测（加固）
- `install_markitai()` - 版本固定安装
- `install_agent_browser()` - 版本固定安装

**验收标准**: 函数可被其他脚本 source 调用

---

#### Task 2: PowerShell 脚本公共函数库

**文件**: 新建 `scripts/lib.ps1`

**内容**:
- `Test-AdminWarning` - 管理员执行警告
- `Confirm-RemoteScript` - 远程脚本二次确认
- `Test-Python` - Python 版本探测
- `Test-Node` - Node 版本探测
- `Install-Markitai` - 版本固定安装
- `Install-AgentBrowser` - 版本固定安装

**验收标准**: 函数可被其他脚本 dot-source 调用

---

#### Task 3: 用户版 Shell 脚本重构

**文件**: `scripts/setup.sh`, `scripts/setup-zh.sh`

**改动**:
1. source `lib.sh`
2. 脚本入口调用 `warn_if_root()`
3. UV 安装默认改为 N，选择 Y 时调用 `confirm_remote_script()`
4. Chromium 下载默认改为 N
5. Linux `--with-deps` 默认改为 N
6. 修复 `detect_uv()` 返回值问题：跳过/失败均返回 0
7. 替换 Python/Node 探测为公共函数
8. 替换 markitai/agent-browser 安装为公共函数

**验收标准**: 验证方案 1, 5, 6, 7, 8, 9, 10, 11, 12, 13

---

#### Task 4: 用户版 PowerShell 脚本重构

**文件**: `scripts/setup.ps1`, `scripts/setup-zh.ps1`

**改动**:
1. dot-source `lib.ps1`
2. 脚本入口调用 `Test-AdminWarning`
3. UV 安装默认改为 N，选择 Y 时调用 `Confirm-RemoteScript`
4. Chromium 下载默认改为 N
5. 替换 Python/Node 探测为公共函数
6. 替换 markitai/agent-browser 安装为公共函数

**验收标准**: 验证方案 2, 5, 6, 10, 11, 12, 13

---

#### Task 5: 开发者版 Shell 脚本重构

**文件**: `scripts/setup-dev.sh`, `scripts/setup-dev-zh.sh`

**改动**:
1. source `lib.sh`
2. 脚本入口调用 `warn_if_root()`
3. UV 安装默认改为 N，选择 Y 时调用 `confirm_remote_script()`
4. Chromium 下载默认改为 N
5. Linux `--with-deps` 默认改为 N
6. UV 缺失时直接失败退出（保持现有行为）
7. 替换 Python/Node 探测为公共函数

**验收标准**: 验证方案 3, 5, 6, 7, 8, 9, 10

---

#### Task 6: 开发者版 PowerShell 脚本重构

**文件**: `scripts/setup-dev.ps1`, `scripts/setup-dev-zh.ps1`

**改动**:
1. dot-source `lib.ps1`
2. 脚本入口调用 `Test-AdminWarning`
3. UV 安装默认改为 N，选择 Y 时调用 `Confirm-RemoteScript`
4. Chromium 下载默认改为 N
5. UV 缺失时直接失败退出（保持现有行为）
6. 替换 Python/Node 探测为公共函数

**验收标准**: 验证方案 4, 5, 6, 10

---

#### Task 7: 文档更新

**文件**: `README.md`, `website/guide/installation.md`, `website/zh/guide/installation.md`

**改动**:
1. 说明默认行为变更（UV/Chromium/系统依赖默认不自动安装）
2. 说明版本固定环境变量用法
3. 说明 root/管理员执行警告

**验收标准**: 文档与实际行为一致

---

### 实施顺序

```
Task 1 (lib.sh) ──┬──> Task 3 (用户版 Shell)
                  └──> Task 5 (开发者版 Shell)

Task 2 (lib.ps1) ─┬──> Task 4 (用户版 PowerShell)
                  └──> Task 6 (开发者版 PowerShell)

Task 3~6 完成后 ────> Task 7 (文档更新)
```

**预计改动文件数**: 10 个（新建 2 + 修改 8）

### 实施状态

| 任务 | 状态 | 说明 |
|------|------|------|
| Task 1: lib.sh | ✅ 完成 | 创建 Shell 公共函数库 |
| Task 2: lib.ps1 | ✅ 完成 | 创建 PowerShell 公共函数库 |
| Task 3: 用户版 Shell | ✅ 完成 | 重构 setup.sh, setup-zh.sh |
| Task 4: 用户版 PowerShell | ✅ 完成 | 重构 setup.ps1, setup-zh.ps1 |
| Task 5: 开发者版 Shell | ✅ 完成 | 重构 setup-dev.sh, setup-dev-zh.sh |
| Task 6: 开发者版 PowerShell | ✅ 完成 | 重构 setup-dev.ps1, setup-dev-zh.ps1 |
| Task 7: 文档更新 | ✅ 完成 | 更新 website/guide/getting-started.md |
| Task 8: 远程执行支持 | ✅ 完成 | 动态下载 lib.sh/lib.ps1 |

**完成日期**: 2026-01-27

### 远程执行支持

为解决 `curl ... | sh` 和 `irm ... | iex` 远程执行时公共库无法加载的问题，所有脚本增加了动态下载机制：

**Shell 脚本**:
```bash
# 检测本地/远程执行
if [ -f "$0" ] && [ "$(basename "$0")" != "sh" ]; then
    # 本地执行 - 使用相对路径
    . "$SCRIPT_DIR/lib.sh"
else
    # 远程执行 - 从 GitHub 下载
    curl -fsSL "$LIB_BASE_URL/lib.sh" -o "$TEMP_LIB"
    . "$TEMP_LIB"
fi
```

**PowerShell 脚本**:
```powershell
# 检测本地/远程执行
if ($ScriptDir -and (Test-Path "$ScriptDir\lib.ps1")) {
    # 本地执行 - 使用相对路径
    . "$ScriptDir\lib.ps1"
} else {
    # 远程执行 - 从 GitHub 下载
    Invoke-RestMethod "$LIB_BASE_URL/lib.ps1" -OutFile $tempLib
    . $tempLib
}
```
