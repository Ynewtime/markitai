# Claude 与 Copilot 订阅

markitai 可以通过 Claude Code 或 GitHub Copilot 订阅发送请求，不必使用按 token 计费的 API 密钥。两者都通过厂商自己的命令行工作，markitai（包括 `markitai serve`）会在它所在的机器上查找。如果网页工作台提示某个订阅“尚未安装”或“尚未登录”，请在那台机器上运行本页的命令，然后重新加载模型列表。

## Claude

模型名为 `claude-agent/<模型>`，例如 `claude-agent/haiku`。

1. 安装 Python 可选依赖：`uv tool install "markitai[claude-agent]" --force`（pip：`pip install "markitai[claude-agent]"`）。
2. 安装 Claude Code 命令行，版本 2.1.284 或更新：`npm install -g @anthropic-ai/claude-code@2.1.284`（macOS 和 Linux 也可用 `curl -fsSL https://claude.ai/install.sh | bash`；Windows：`irm https://claude.ai/install.ps1 | iex`）。
3. 登录：`markitai auth claude login`。

用 `markitai auth claude status` 和 `markitai doctor` 检查。

设置了 `ANTHROPIC_API_KEY` 时，Claude Code 会按该密钥计费而不是用订阅；已登录订阅时，markitai 会把这个变量对 Claude Code 隐藏。

## Copilot

模型名为 `copilot/<模型>`，例如 `copilot/auto`。

1. 安装 Python 可选依赖：`uv tool install "markitai[copilot]" --force`。
2. 安装 Copilot 命令行，版本 1.0.90 或更新：`npm install -g @github/copilot@1.0.90-2`（或 `curl -fsSL https://gh.io/copilot-install | bash`；Windows：`winget install GitHub.Copilot`）。
3. 登录：`markitai auth copilot login`。带有“Copilot Requests”权限的 `COPILOT_GITHUB_TOKEN`、`GH_TOKEN` 或 `GITHUB_TOKEN` 同样可用。

## ChatGPT

见 [ChatGPT 订阅](/zh/guide/subscription-chatgpt)。
