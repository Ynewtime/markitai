# 快速开始

## 一键安装（推荐）

安装脚本一步装好 Python（如需要）、uv 和 markitai：

::: code-group
```bash [Linux/macOS]
curl -fsSL https://markitai.dev/setup.sh | sh
```

```powershell [Windows]
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```
:::

::: warning 安全提示
- 脚本会检查 root/管理员权限，并在继续前询问
- 在交互终端中，可选组件会逐项询问：Playwright 浏览器、Web UI、OCR 默认为是；LibreOffice 和 Claude/Copilot CLI 默认为否
- 没有可用终端时，只安装 uv、Python 和 markitai。自动化场景设 `MARKITAI_INSTALL_OPTIONAL=1` 启用可选步骤
- 默认使用官方包索引，除非实测它在你的网络下缓慢或不可达。`MARKITAI_USE_MIRROR=1` 总是提供镜像选择；`=0` 从不询问
:::

通过环境变量固定精确版本（两个变量都不设则取各自最新稳定版，即推荐默认）：

::: code-group
```bash [Linux/macOS]
export MARKITAI_VERSION="X.Y.Z"   # https://pypi.org/project/markitai/#history
export UV_VERSION="X.Y.Z"         # https://github.com/astral-sh/uv/releases
curl -fsSL https://markitai.dev/setup.sh | sh
```

```powershell [Windows]
$env:MARKITAI_VERSION = "X.Y.Z"   # https://pypi.org/project/markitai/#history
$env:UV_VERSION = "X.Y.Z"         # https://github.com/astral-sh/uv/releases
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```
:::

## 第一次转换

转换一个真实网页——就是你正在读的这一篇：

```bash
mkai https://markitai.dev/zh/guide/getting-started --pure
```

每次安装都会同时提供 `markitai` 命令和更短的 `mkai` 别名（两者完全相同；若 `PATH` 上已有其他 `mkai`，请用全名）。`--pure` 会把不含 frontmatter 的 Markdown 正文直接打印到 stdout。要写入文件就加 `-o output/`：

```bash
markitai document.docx -o output/          # 文档
markitai https://example.com/article -o output/   # 网页
markitai ./docs -o ./output                # 整个目录
```

如需 AI 增强（`--llm`），再配置 LLM 提供商：

```bash
markitai init                # 引导式配置（或用 markitai -I 交互模式）
markitai doctor              # 查看核心与可选能力状态
```

## 可选能力

需要时才装对应 extra（`uv tool install 'markitai[<extra>]' --force`）：

| Extra / 依赖 | 启用能力 |
|--------------------|---------|
| `markitai[browser]`（Playwright） | `-s playwright` 浏览器渲染（SPA/重 JS 页面） |
| `markitai[ocr]`（RapidOCR） | `--ocr` 扫描件 PDF/图片的本地 OCR |
| `markitai[legacy]`（anydoc） | 旧版 Office `.doc`/`.ppt` 转换 |
| `markitai[heif]` | HEIC/HEIF/AVIF 图片输入 |
| `markitai[svg]` | 高质量 SVG 渲染 |
| `markitai[kreuzberg]` | 经 Kreuzberg 转换 `.xml`、`.tsv`、`.rtf`、`.rst`、`.org`、`.tex`、`.odt`、`.ods` |
| `markitai[serve]` | 本地 Web 工作区与 REST API |
| Jina API key | `-s jina` 远程阅读器（环境变量 `JINA_API_KEY`） |
| Cloudflare | `-s cloudflare` 云端渲染（`CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`） |

装了 browser extra 后，装一次 Chromium：

```bash
markitai doctor --fix
```

（`doctor --fix` 只在已安装 Playwright 包时才安装 Chromium；纯核心安装下它会安全退出并告知该装哪个 extra。）

## 手动安装

已有 Python 3.11–3.13、想最小化安装时：

```bash
uv tool install markitai        # 推荐：隔离的工具环境
uv pip install markitai         # 或装进当前虚拟环境
pipx install markitai           # 或 pipx
```

手动安装不会配置任何可选组件：用 `markitai doctor` 查看可用能力，用 `markitai init` 配置 LLM。浏览器渲染需按安装方式补 browser extra（`uv tool install 'markitai[browser]' --force`、`pipx install 'markitai[browser]' --force` 或 `uv pip install 'markitai[browser]'`），然后 `markitai doctor --fix`。

## 功能说明

**URL**：公开 URL 先走本地方法，之后 `auto` 可能不经询问尝试 Defuddle、Jina 或 Cloudflare（进程内首次远程尝试会在 stderr 披露）。私有、本地、内网及带凭据的 URL 始终只走本地。`MARKITAI_NO_REMOTE_FETCH=1` 强制全部本地。

**LLM 增强**（`--llm`）：清洗格式并生成 frontmatter。配置提供商 API key 或订阅制提供商（`chatgpt/` 走 OAuth；`claude-agent/`、`copilot/` 用各自 CLI 登录）——见[配置](/zh/guide/configuration#supported-providers)。

**预设**打包常用参数：`rich`（LLM + alt + desc + 截图）、`standard`（LLM + alt + desc）、`minimal`（仅基础转换）。预设里的任何特性都可用 `--no-*` 覆盖，如 `--preset rich --no-desc`。

**批量运行**会写出 JSON 报告，中断后支持 `--resume`。`--llm-batch`（Batch API，半价）等见 [CLI 参考](/zh/guide/cli)。

## 输出结构

```
output/
├── document.pdf.md          # 基础 Markdown（--llm 模式下默认跳过，除非 --keep-base）
├── document.pdf.llm.md      # LLM 增强版（使用 --llm 时）
├── .markitai/                 # 元数据命名空间
│   ├── assets/
│   │   ├── document.docx.0001.jpg   # 源文档内嵌的图片
│   │   └── images.json      # 图片描述
│   ├── screenshots/          # 页面/幻灯片截图（仅 PDF/PPTX；URL 为整页；--screenshot）
│   │   └── document.pdf.page0001.jpg
│   ├── reports/               # 转换报告（JSON）——批量/URL 批量默认生成，或 output.report = true 时
│   └── states/                # 批量状态文件（供 --resume）
```

输出文件名是在完整输入文件名后追加 `.md`：`document.docx` → `document.docx.md`（`--llm` 时为 `document.docx.llm.md`），因此不同的输入（`report.pdf`、`report.docx`）不会互相覆盖。

## 支持的格式

| 格式 | 扩展名 |
|--------|------------|
| Office 文档 | `.docx`、`.doc`、`.pptx`、`.ppt`、`.xlsx`、`.xls`、`.odt`、`.ods`、`.numbers` |
| PDF | `.pdf` |
| 文本 / 标记 / 结构化数据 | `.txt`、`.md`、`.markdown`、`.html`、`.htm`、`.xhtml`、`.xml`、`.csv`、`.tsv`、`.rtf`、`.rst`、`.org`、`.tex` |
| 图片 | `.jpg`、`.jpeg`、`.png`、`.webp`、`.svg`、`.gif`、`.bmp`、`.tiff`、`.tif`、`.heic`、`.heif`、`.avif`（后三者需要 `markitai[heif]`） |
| 其他文档 | `.epub`、`.eml`、`.msg`、`.ipynb` |
| URL | `http://`、`https://` |

## 平台特定功能

### Windows

| 功能 | 支持 | 说明 |
|------|------|------|
| 旧版 Office（`.doc`、`.ppt`） | ✅ 完全支持 | 需要 `markitai[legacy]`（anydoc Rust 后端，无需安装 Office；PPT 表格会展开为纯文本行） |
| 旧版 Excel（`.xls`） | ✅ 完全支持 | 内置（纯 Python） |
| PPTX 幻灯片渲染 | ✅ 完全支持 | 优先 MS Office，LibreOffice 备选 |
| EMF/WMF 图片 | ✅ 完全支持 | 原生支持 |
| 浏览器自动化 | ✅ 完全支持 | 隐藏窗口模式 |

### Linux

| 功能 | 支持 | 说明 |
|------|------|------|
| 旧版 Office（`.doc`、`.ppt`） | ✅ 完全支持 | 需要 `markitai[legacy]`（无需 LibreOffice） |
| 旧版 Excel（`.xls`） | ✅ 完全支持 | 内置（纯 Python） |
| PPTX 幻灯片渲染 | ✅ 完全支持 | 需要 LibreOffice（`apt-get install libreoffice` / `dnf install libreoffice`） |
| EMF/WMF 图片 | ❌ 不支持 | Windows 专有格式 |
| 浏览器自动化 | ✅ 完全支持 | 需要系统依赖 |

### macOS

| 功能 | 支持 | 说明 |
|------|------|------|
| 旧版 Office（`.doc`、`.ppt`） | ✅ 完全支持 | 需要 `markitai[legacy]`（无需安装 Office） |
| 旧版 Excel（`.xls`） | ✅ 完全支持 | 内置（纯 Python） |
| PPTX 幻灯片渲染 | ✅ 完全支持 | 优先 LibreOffice（`brew install --cask libreoffice`）；未安装时回退到已装的 MS PowerPoint |
| EMF/WMF 图片 | ❌ 不支持 | Windows 专有格式 |
| 浏览器自动化 | ✅ 完全支持 | - |

macOS 的 PowerPoint 回退通过 AppleScript 驱动 PowerPoint：首次渲染会触发一次性的授权弹窗（“Terminal 想要控制 Microsoft PowerPoint”），会短暂打开应用窗口，且需要图形界面会话。无头环境请在配置里设 `"office": { "macos_fallback": false }` 关闭。

## 下一步

- [配置](/zh/guide/configuration) - LLM 提供商与全部设置
- [CLI 参考](/zh/guide/cli) - 完整命令参考
