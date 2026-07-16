# Markitai Setup Script - Unified installer with i18n support
# PowerShell 5.1+
# Auto-detects: language (en/zh), mode (user/dev)
#
# Usage:
#   powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"   # User install
#   .\scripts\setup.ps1                                                               # Dev setup (in repo)
# Full failure logs use a private temp file; set MARKITAI_SETUP_LOG to override.

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ============================================================
# Internationalization (i18n) System
# ============================================================

# Detect language from environment
# Returns: "zh" for Chinese, "en" for English (default)
function Get-Lang {
    $culture = (Get-Culture).Name
    if ($culture -match "^zh") {
        return "zh"
    }
    return "en"
}

$script:LANG_CODE = Get-Lang

# Internationalization function
# Usage: i18n "key"
# Returns localized string for the given key
function i18n {
    param([string]$Key)

    if ($script:LANG_CODE -eq "zh") {
        switch ($Key) {
            # Intro/Outro
            "welcome"                   { return "欢迎使用 Markitai 安装程序!" }
            "setup_complete"            { return "安装完成!" }
            "setup_complete_with_warnings" { return "安装完成，但有部分组件失败" }
            "dev_setup_complete"        { return "开发环境设置完成!" }
            "dev_setup_complete_with_warnings" { return "开发环境设置完成，但有部分组件失败" }

            # Mode
            "mode_user"                 { return "模式: 用户安装" }
            "mode_dev"                  { return "模式: 开发环境" }

            # Sections
            "section_prerequisites"     { return "前置条件" }
            "section_core"              { return "核心组件" }
            "section_optional"          { return "可选组件" }
            "section_dev_env"           { return "开发环境" }
            "section_llm_cli"           { return "LLM CLI 工具" }
            "section_summary"           { return "安装摘要" }

            # Status
            "installed"                 { return "已安装" }
            "installing"                { return "正在安装" }
            "skipped"                   { return "已跳过" }
            "failed"                    { return "失败" }
            "success"                   { return "成功" }
            "already_installed"         { return "已经安装" }
            "not_found"                 { return "未找到" }

            # Components
            "uv"                        { return "uv 包管理器" }
            "python"                    { return "Python" }
            "markitai"                  { return "markitai" }
            "serve"                     { return "Web UI (markitai serve)" }
            "playwright"                { return "Playwright 浏览器" }
            "libreoffice"               { return "LibreOffice" }
            "ffmpeg"                    { return "FFmpeg" }
            "claude_cli"                { return "Claude Code CLI" }
            "copilot_cli"               { return "Copilot CLI" }
            "precommit"                 { return "pre-commit hooks" }
            "python_deps"               { return "Python 依赖" }

            # Confirmations
            "confirm_serve"             { return "安装 Web UI 依赖? (启用 markitai serve)" }
            "confirm_playwright"        { return "安装 Playwright 浏览器? (用于 JS 渲染页面)" }
            "confirm_libreoffice"       { return "安装 LibreOffice? (用于 Office 文档转换)" }
            "confirm_ffmpeg"            { return "安装 FFmpeg? (用于音视频处理)" }
            "confirm_claude_cli"        { return "安装 Claude Code CLI? (使用 Claude 订阅)" }
            "confirm_copilot_cli"       { return "安装 Copilot CLI? (使用 GitHub Copilot 订阅)" }
            "confirm_uv"                { return "安装 uv 包管理器?" }
            "confirm_continue_as_admin" { return "以管理员身份继续?" }

            # Info messages
            "info_libreoffice_purpose"  { return "LibreOffice 用于转换旧版 Office 文档 (.doc/.ppt) 并渲染幻灯片截图" }
            "info_ffmpeg_purpose"       { return "FFmpeg 用于处理音频和视频文件" }
            "info_playwright_purpose"   { return "Playwright 用于获取 JavaScript 渲染的网页内容" }
            "info_project_dir"          { return "项目目录" }
            "info_docs"                 { return "文档" }
            "info_issues"               { return "问题反馈" }
            "info_syncing_deps"         { return "正在同步依赖..." }
            "info_deps_synced"          { return "依赖同步完成" }
            "info_precommit_installed"  { return "pre-commit hooks 已安装" }
            "info_error_log"            { return "完整错误日志" }

            # Error messages
            "error_uv_required"         { return "需要安装 uv 包管理器" }
            "error_python_required"     { return "需要安装 Python 3.11-3.13" }
            "error_unexpected"          { return "发生意外错误" }
            "error_setup_failed"        { return "安装失败" }

            # Install source (repo detection)
            "repo_detected"             { return "检测到 markitai 源码仓库" }
            "confirm_local_install"     { return "从本地源码安装 markitai? (默认使用 PyPI 发布版)" }
            "source_local"              { return "安装来源: 本地源码" }
            "source_pypi"               { return "安装来源: PyPI" }
            "info_repo_noninteractive"  { return "非交互模式: 已检测到源码仓库, 将使用 PyPI 发布版" }

            # Network / Mirrors
            "section_network"           { return "网络环境" }
            "mirror_no_proxy"           { return "未检测到代理，部分资源可能无法访问" }
            "mirror_confirm"            { return "启用国内镜像加速? (推荐无代理环境使用)" }
            "mirror_select"             { return "选择镜像源" }
            "mirror_tuna"               { return "清华 TUNA (推荐)" }
            "mirror_aliyun"             { return "阿里云" }
            "mirror_tencent"            { return "腾讯云" }
            "mirror_huawei"             { return "华为云" }
            "mirror_enabled"            { return "已启用国内镜像加速" }
            "mirror_skipped"            { return "已跳过镜像配置" }
            "mirror_pypi"               { return "PyPI 镜像" }
            "mirror_playwright"         { return "Playwright 镜像" }
            "mirror_npm"                { return "npm 镜像" }

            # Warnings
            "warn_admin"                { return "警告: 以管理员身份运行" }
            "warn_admin_risk"           { return "以管理员运行安装脚本存在安全风险" }
            "warn_wsl"                  { return "警告: 检测到 WSL 环境" }
            "warn_wsl_tip"              { return "建议使用 shell 脚本: ./scripts/setup.sh" }

            # Getting started
            "getting_started"           { return "开始使用" }
            "quick_start"               { return "快速开始" }
            "activate_venv"             { return "激活虚拟环境" }
            "run_tests"                 { return "运行测试" }
            "run_cli"                   { return "运行 CLI" }
            "interactive_mode"          { return "交互模式" }
            "start_web_ui"              { return "启动 Web UI" }
            "configure_llm"             { return "配置 LLM" }
            "configure_env"             { return "配置环境变量" }
            "convert_file"              { return "转换文件" }
            "show_help"                 { return "显示帮助" }
            "short_alias"               { return "提示：mkai 是 markitai 的简写，两个命令等价" }
            "mkai_conflict"             { return "系统已有 mkai 命令，会遮蔽 markitai 的别名；请使用完整命令 markitai" }

            # Summary
            "summary_installed"         { return "已安装" }
            "summary_skipped"           { return "已跳过" }
            "summary_failed"            { return "安装失败" }

            # Default fallback
            default                     { return $Key }
        }
    } else {
        # English (default)
        switch ($Key) {
            # Intro/Outro
            "welcome"                   { return "Welcome to Markitai Setup!" }
            "setup_complete"            { return "Setup complete!" }
            "setup_complete_with_warnings" { return "Setup complete with warnings" }
            "dev_setup_complete"        { return "Development environment ready!" }
            "dev_setup_complete_with_warnings" { return "Development environment ready with warnings" }

            # Mode
            "mode_user"                 { return "Mode: User Install" }
            "mode_dev"                  { return "Mode: Development" }

            # Sections
            "section_prerequisites"     { return "Prerequisites" }
            "section_core"              { return "Core Components" }
            "section_optional"          { return "Optional Components" }
            "section_dev_env"           { return "Development Environment" }
            "section_llm_cli"           { return "LLM CLI Tools" }
            "section_summary"           { return "Installation Summary" }

            # Status
            "installed"                 { return "installed" }
            "installing"                { return "installing" }
            "skipped"                   { return "skipped" }
            "failed"                    { return "failed" }
            "success"                   { return "success" }
            "already_installed"         { return "already installed" }
            "not_found"                 { return "not found" }

            # Components
            "uv"                        { return "uv package manager" }
            "python"                    { return "Python" }
            "markitai"                  { return "markitai" }
            "serve"                     { return "Web UI (markitai serve)" }
            "playwright"                { return "Playwright browser" }
            "libreoffice"               { return "LibreOffice" }
            "ffmpeg"                    { return "FFmpeg" }
            "claude_cli"                { return "Claude Code CLI" }
            "copilot_cli"               { return "Copilot CLI" }
            "precommit"                 { return "pre-commit hooks" }
            "python_deps"               { return "Python dependencies" }

            # Confirmations
            "confirm_serve"             { return "Install Web UI dependencies? (enables markitai serve)" }
            "confirm_playwright"        { return "Install Playwright browser? (for JS-rendered pages)" }
            "confirm_libreoffice"       { return "Install LibreOffice? (for Office document conversion)" }
            "confirm_ffmpeg"            { return "Install FFmpeg? (for audio/video processing)" }
            "confirm_claude_cli"        { return "Install Claude Code CLI? (use your Claude subscription)" }
            "confirm_copilot_cli"       { return "Install Copilot CLI? (use your GitHub Copilot subscription)" }
            "confirm_uv"                { return "Install uv package manager?" }
            "confirm_continue_as_admin" { return "Continue as administrator?" }

            # Info messages
            "info_libreoffice_purpose"  { return "LibreOffice converts legacy Office files (.doc/.ppt) and renders slide screenshots" }
            "info_ffmpeg_purpose"       { return "FFmpeg processes audio and video files" }
            "info_playwright_purpose"   { return "Playwright fetches JavaScript-rendered web pages" }
            "info_project_dir"          { return "Project directory" }
            "info_docs"                 { return "Documentation" }
            "info_issues"               { return "Issues" }
            "info_syncing_deps"         { return "Syncing dependencies..." }
            "info_deps_synced"          { return "Dependencies synced" }
            "info_precommit_installed"  { return "pre-commit hooks installed" }
            "info_error_log"            { return "Full error log" }

            # Error messages
            "error_uv_required"         { return "uv package manager is required" }
            "error_python_required"     { return "Python 3.11-3.13 is required" }
            "error_unexpected"          { return "Unexpected error" }
            "error_setup_failed"        { return "Setup failed" }

            # Install source (repo detection)
            "repo_detected"             { return "markitai source repo detected" }
            "confirm_local_install"     { return "Install markitai from the local repo? (default: PyPI release)" }
            "source_local"              { return "Install source: local repo" }
            "source_pypi"               { return "Install source: PyPI" }
            "info_repo_noninteractive"  { return "Non-interactive mode: source repo detected, using PyPI release" }

            # Network / Mirrors
            "section_network"           { return "Network Environment" }
            "mirror_no_proxy"           { return "No proxy detected, some resources may be inaccessible" }
            "mirror_confirm"            { return "Enable China mirror acceleration? (recommended without proxy)" }
            "mirror_select"             { return "Select mirror source" }
            "mirror_tuna"               { return "Tsinghua TUNA (Recommended)" }
            "mirror_aliyun"             { return "Alibaba Cloud" }
            "mirror_tencent"            { return "Tencent Cloud" }
            "mirror_huawei"             { return "Huawei Cloud" }
            "mirror_enabled"            { return "China mirror acceleration enabled" }
            "mirror_skipped"            { return "Mirror configuration skipped" }
            "mirror_pypi"               { return "PyPI mirror" }
            "mirror_playwright"         { return "Playwright mirror" }
            "mirror_npm"                { return "npm mirror" }

            # Warnings
            "warn_admin"                { return "Warning: Running as administrator" }
            "warn_admin_risk"           { return "Running setup scripts as administrator carries security risks" }
            "warn_wsl"                  { return "Warning: WSL environment detected" }
            "warn_wsl_tip"              { return "Consider using the shell script: ./scripts/setup.sh" }

            # Getting started
            "getting_started"           { return "Getting Started" }
            "quick_start"               { return "Quick Start" }
            "activate_venv"             { return "Activate virtual environment" }
            "run_tests"                 { return "Run tests" }
            "run_cli"                   { return "Run CLI" }
            "interactive_mode"          { return "Interactive mode" }
            "start_web_ui"              { return "Start Web UI" }
            "configure_llm"             { return "Configure LLM" }
            "configure_env"             { return "Configure environment" }
            "convert_file"              { return "Convert a file" }
            "show_help"                 { return "Show help" }
            "short_alias"               { return "Tip: mkai is a short alias for markitai (both work)" }
            "mkai_conflict"             { return "An existing 'mkai' command shadows markitai's alias; use the full 'markitai' command" }

            # Summary
            "summary_installed"         { return "Installed" }
            "summary_skipped"           { return "Skipped" }
            "summary_failed"            { return "Failed" }

            # Default fallback
            default                     { return $Key }
        }
    }
}

# ============================================================
# Compact tree-style visual components
# ============================================================
# The one-column guide follows the `withGuide` visual grammar popularized by
# @clack/prompts. Keeping this renderer inline preserves PowerShell 5.1 support
# and lets the bootstrap run before Node or Python is installed.

$S_BAR = [char]0x2502           # │
$S_BAR_H = [char]0x2500         # ─
$S_BRANCH = [char]0x251C        # ├
$S_CORNER_TOP = [char]0x256D    # ╭
$S_CORNER_BOT = [char]0x2570    # ╰
$S_CHECK = [char]0x2713         # ✓
$S_CROSS = [char]0x2717         # ✗
$S_ARROW = [char]0x2192         # →
$S_CIRCLE = [char]0x25CB        # ○

# Session state lets the global catch close an interrupted tree exactly once.
$script:CLACK_SESSION_ACTIVE = $false
$script:CLACK_SESSION_CLOSED = $false
$script:SETUP_LOG_FILE = $env:MARKITAI_SETUP_LOG
$script:SETUP_LOG_AUTO = $false
$script:SETUP_LOG_SHOWN = $false

# Print an empty row while preserving the session guide.
function Write-GuideLine {
    Write-Host $S_BAR -ForegroundColor DarkGray
}

# Print a branch connected to the session guide.
function Write-TreeBranch {
    param(
        [string]$Text,
        [ConsoleColor]$Color = [ConsoleColor]::DarkGray
    )
    Write-Host $S_BRANCH -ForegroundColor DarkGray -NoNewline
    Write-Host $S_BAR_H -ForegroundColor $Color -NoNewline
    Write-Host " $Text"
}

# Session intro - start of CLI flow
function Clack-Intro {
    param([string]$Title)
    $script:CLACK_SESSION_ACTIVE = $true
    $script:CLACK_SESSION_CLOSED = $false
    Write-Host ""
    Write-Host $S_CORNER_TOP -ForegroundColor DarkGray -NoNewline
    Write-Host $S_BAR_H -ForegroundColor DarkGray -NoNewline
    Write-Host " $Title"
    Write-GuideLine
}

# Session outro - end of CLI flow
function Clack-Outro {
    param([string]$Message)
    $script:CLACK_SESSION_CLOSED = $true
    if ($script:SETUP_LOG_AUTO -and $script:SETUP_LOG_FILE) {
        Remove-Item $script:SETUP_LOG_FILE -Force -ErrorAction SilentlyContinue
        $script:SETUP_LOG_FILE = $null
    }
    Write-GuideLine
    Write-Host $S_CORNER_BOT -ForegroundColor DarkGray -NoNewline
    Write-Host $S_BAR_H -ForegroundColor DarkGray -NoNewline
    Write-Host " $Message" -ForegroundColor Green
    Write-Host ""
}

# Non-fatal component failures close the tree in warning yellow, not success
# green, while still returning a successful core installation.
function Clack-OutroWarning {
    param([string]$Message)
    Show-SetupErrorLog
    $script:CLACK_SESSION_CLOSED = $true
    Write-GuideLine
    Write-Host $S_CORNER_BOT -ForegroundColor DarkGray -NoNewline
    Write-Host $S_BAR_H -ForegroundColor DarkGray -NoNewline
    Write-Host " $Message" -ForegroundColor Yellow
    Write-Host ""
}

# Section header connected to the session guide
function Clack-Section {
    param([string]$Title)
    Write-GuideLine
    Write-TreeBranch -Text $Title -Color Magenta
}

# Log with guide line - success
function Clack-Success {
    param([string]$Message)
    Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
    Write-Host "  " -NoNewline
    Write-Host $S_CHECK -ForegroundColor Green -NoNewline
    Write-Host " $Message"
}

# Log with guide line - error
function Clack-Error {
    param([string]$Message)
    Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
    Write-Host "  " -NoNewline
    Write-Host $S_CROSS -ForegroundColor Red -NoNewline
    Write-Host " $Message"
}

# Log with guide line - warning
function Clack-Warn {
    param([string]$Message)
    Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
    Write-Host "  " -NoNewline
    Write-Host "!" -ForegroundColor Yellow -NoNewline
    Write-Host " $Message"
}

# Log with guide line - info
function Clack-Info {
    param([string]$Message)
    Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
    Write-Host "  " -NoNewline
    Write-Host $S_ARROW -ForegroundColor Cyan -NoNewline
    Write-Host " $Message"
}

# Log with guide line - skipped
function Clack-Skip {
    param([string]$Message)
    Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
    Write-Host "  " -NoNewline
    Write-Host $S_CIRCLE -ForegroundColor DarkGray -NoNewline
    Write-Host " $Message" -ForegroundColor DarkGray
}

# Log with guide line - plain text
function Clack-Log {
    param([string]$Message)
    if ([string]::IsNullOrEmpty($Message)) {
        Write-GuideLine
        return
    }
    Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
    Write-Host "  $Message"
}

function Initialize-SetupLog {
    if ($script:SETUP_LOG_FILE) {
        try {
            $null = [IO.File]::Open(
                $script:SETUP_LOG_FILE,
                [IO.FileMode]::Append,
                [IO.FileAccess]::Write,
                [IO.FileShare]::Read
            ).Dispose()
            return $true
        } catch {
            $script:SETUP_LOG_FILE = $null
            return $false
        }
    }

    try {
        $script:SETUP_LOG_FILE = Join-Path (
            [IO.Path]::GetTempPath()
        ) "markitai-setup-$([Guid]::NewGuid()).log"
        $null = [IO.File]::Create($script:SETUP_LOG_FILE).Dispose()
        $script:SETUP_LOG_AUTO = $true
        return $true
    } catch {
        $script:SETUP_LOG_FILE = $null
        return $false
    }
}

function Write-SetupDiagnostic {
    param(
        [string]$Text,
        [string]$Context = "diagnostic"
    )

    if ([string]::IsNullOrWhiteSpace($Text)) { return }
    if (-not (Initialize-SetupLog)) { return }
    try {
        $entry = "`r`n== $Context ==`r`n$Text`r`n"
        [IO.File]::AppendAllText($script:SETUP_LOG_FILE, $entry)
    } catch {}
}

function Show-SetupErrorLog {
    if (-not $script:SETUP_LOG_FILE -or $script:SETUP_LOG_SHOWN) { return }
    $script:SETUP_LOG_SHOWN = $true
    Clack-Info "$(i18n 'info_error_log'): $($script:SETUP_LOG_FILE)"
}

# Print only the final few third-party diagnostic lines. ANSI sequences are
# removed so captured native errors cannot turn the whole block red; the full
# sanitized text is retained in a private setup log for troubleshooting.
function Clack-Detail {
    param(
        [AllowNull()]
        [object]$Detail,
        [int]$MaxLines = 6,
        [string]$Context = "diagnostic"
    )

    if ($null -eq $Detail) { return }
    $text = if ($Detail -is [string]) {
        $Detail
    } else {
        (@($Detail) | ForEach-Object { "$_" }) -join [Environment]::NewLine
    }
    if ([string]::IsNullOrWhiteSpace($text)) { return }

    $escape = [Regex]::Escape([string][char]27)
    $clean = [Regex]::Replace($text, "$escape\[[0-9;?]*[A-Za-z]", "")
    Write-SetupDiagnostic -Text $clean -Context $Context
    $lines = @(
        $clean -split "`r?`n" |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            Select-Object -Last $MaxLines
    )
    foreach ($line in $lines) {
        $bounded = if ($line.Length -gt 300) {
            $line.Substring(0, 297) + "..."
        } else {
            $line
        }
        Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
        Write-Host "    $bounded" -ForegroundColor DarkGray
    }
}

# Confirm prompt connected to the session guide
function Clack-Confirm {
    param(
        [string]$Prompt,
        [string]$Default = "n"
    )

    if ($Default -eq "y") {
        $hint = "Y/n"
    } else {
        $hint = "y/N"
    }

    Write-GuideLine
    Write-Host $S_BRANCH -ForegroundColor DarkGray -NoNewline
    Write-Host $S_BAR_H -ForegroundColor Cyan -NoNewline
    if (Test-InteractiveInput) {
        $answer = Read-Host " $Prompt [$hint]"
    } else {
        Write-Host " $Prompt [$hint] $Default"
        $answer = $Default
    }

    if ([string]::IsNullOrWhiteSpace($answer)) {
        $answer = $Default
    }

    return $answer -match "^[Yy]"
}

# PowerShell's redirected-input state is the equivalent of the POSIX setup
# script having no usable terminal for prompts.
function Test-InteractiveInput {
    try {
        return -not [Console]::IsInputRedirected
    } catch {
        return $false
    }
}

function Test-OptionalInstallRequested {
    return $env:MARKITAI_INSTALL_OPTIONAL -match "^(1|true|yes|on)$"
}

# Without interactive input, optional components are disabled unless the
# caller explicitly opts in through MARKITAI_INSTALL_OPTIONAL.
function Confirm-OptionalInstall {
    param(
        [string]$Prompt,
        [string]$Default = "n"
    )

    if (-not (Test-InteractiveInput)) {
        return (Test-OptionalInstallRequested)
    }
    return (Clack-Confirm $Prompt $Default)
}

# Compact note block. Content stays on the single outer guide instead of
# drawing a second, competing box down the left side.
function Clack-Note {
    param(
        [Parameter(Position=0)]
        [string]$Title,
        [Parameter(Position=1, ValueFromRemainingArguments=$true)]
        [string[]]$Lines
    )

    Write-GuideLine
    Write-TreeBranch -Text $Title -Color Green

    foreach ($line in $Lines) {
        Clack-Log $line
    }
}

# Cancel message
function Clack-Cancel {
    param([string]$Message)
    Show-SetupErrorLog
    $script:CLACK_SESSION_CLOSED = $true
    Write-GuideLine
    Write-Host $S_CORNER_BOT -ForegroundColor DarkGray -NoNewline
    Write-Host $S_BAR_H -ForegroundColor DarkGray -NoNewline
    Write-Host " $Message" -ForegroundColor Red
    Write-Host ""
}

# Last-resort guard for exceptions outside the expected per-component paths.
function Complete-UnexpectedFailure {
    param([System.Management.Automation.ErrorRecord]$ErrorRecord)

    if ($script:CLACK_SESSION_ACTIVE -and -not $script:CLACK_SESSION_CLOSED) {
        Clack-Error (i18n "error_unexpected")
        Clack-Detail -Detail $ErrorRecord.Exception.Message -MaxLines 3
        Clack-Cancel (i18n "error_setup_failed")
        return
    }

    Write-Host (i18n "error_setup_failed") -ForegroundColor Red
}

# ============================================================
# Installation Status Tracking
# ============================================================
$script:INSTALLED_COMPONENTS = @()
$script:SKIPPED_COMPONENTS = @()
$script:FAILED_COMPONENTS = @()

function Track-Install {
    param(
        [string]$Component,
        [ValidateSet("installed", "skipped", "failed")]
        [string]$Status
    )

    switch ($Status) {
        "installed" { $script:INSTALLED_COMPONENTS += $Component }
        "skipped" { $script:SKIPPED_COMPONENTS += $Component }
        "failed" { $script:FAILED_COMPONENTS += $Component }
    }
}

# Optional components are failure-isolated: an unexpected exception is shown
# as a bounded tree item, tracked, and does not prevent later choices/summary.
function Invoke-OptionalStep {
    param(
        [string]$Component,
        [scriptblock]$Action
    )

    try {
        & $Action | Out-Null
    } catch {
        Clack-Error "$(i18n $Component) $(i18n 'failed')"
        Clack-Detail -Detail $_.Exception.Message -MaxLines 3
        $alreadyTracked =
            ($script:INSTALLED_COMPONENTS -contains $Component) -or
            ($script:SKIPPED_COMPONENTS -contains $Component) -or
            ($script:FAILED_COMPONENTS -contains $Component)
        if (-not $alreadyTracked) {
            Track-Install -Component $Component -Status "failed"
        }
    }
}

# Best-effort maintenance steps have no standalone summary component, but must
# still not tear down an otherwise successful setup.
function Invoke-BestEffortStep {
    param(
        [string]$FailureMessage,
        [scriptblock]$Action
    )

    try {
        & $Action | Out-Null
    } catch {
        Clack-Warn $FailureMessage
        Clack-Detail -Detail $_.Exception.Message -MaxLines 3
    }
}

# ============================================================
# Version Variables (can be overridden via environment)
# ============================================================
$script:MarkitaiVersion = $env:MARKITAI_VERSION
$script:UvVersion = $env:UV_VERSION
$script:PYTHON_CMD = $null

# ============================================================
# Utility Functions
# ============================================================

# Check if a command exists
function Test-CommandExists {
    param([string]$CommandName)
    $cmd = Get-Command $CommandName -ErrorAction SilentlyContinue
    return ($null -ne $cmd)
}

# Run a native command without allowing PowerShell 5.1 to paint native stderr
# as unbounded red ErrorRecords. Callers decide whether to show the bounded
# Output through Clack-Detail.
function Invoke-NativeQuietly {
    param([scriptblock]$Command)

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $output = @()
    $exitCode = 1
    try {
        $LASTEXITCODE = $null
        $output = @(& $Command 2>&1)
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    } catch {
        $output = @($_.Exception.Message)
        $exitCode = 1
    } finally {
        $ErrorActionPreference = $oldErrorAction
    }

    return [PSCustomObject]@{
        Success = ($exitCode -eq 0)
        ExitCode = $exitCode
        Output = $output
    }
}

# Download official PowerShell installers to a temporary file and execute them
# in a child process. An installer calling `exit` can then never terminate this
# parent script before its tree is closed.
function Invoke-PowerShellInstallerQuietly {
    param([string]$Uri)

    $tempPath = Join-Path ([IO.Path]::GetTempPath()) "markitai-installer-$([Guid]::NewGuid()).ps1"
    try {
        $content = Invoke-RestMethod -Uri $Uri -ErrorAction Stop
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($tempPath, [string]$content, $utf8NoBom)
        $engine = (Get-Process -Id $PID -ErrorAction Stop).Path
        return Invoke-NativeQuietly {
            & $engine -NoLogo -NoProfile -ExecutionPolicy Bypass -File $tempPath
        }
    } catch {
        return [PSCustomObject]@{
            Success = $false
            ExitCode = 1
            Output = @($_.Exception.Message)
        }
    } finally {
        Remove-Item $tempPath -Force -ErrorAction SilentlyContinue
    }
}

# Get project root directory
function Get-ProjectRoot {
    $script:ScriptDir = $PSScriptRoot
    if (-not $script:ScriptDir -and $MyInvocation.MyCommand.Path) {
        $script:ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    }
    if ($script:ScriptDir) {
        return Split-Path -Parent $script:ScriptDir
    }
    return $PWD.Path
}

# Check if running as administrator and warn
function Test-AdminWarning {
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if ($isAdmin) {
        Clack-Warn (i18n "warn_admin")
        Clack-Log (i18n "warn_admin_risk")

        if (-not (Clack-Confirm (i18n "confirm_continue_as_admin") "n")) {
            Clack-Cancel (i18n "error_setup_failed")
            exit 1
        }
    }
}

# Check for WSL environment and warn
function Test-WSLWarning {
    if ($env:WSL_DISTRO_NAME) {
        Clack-Warn (i18n "warn_wsl")
        Clack-Log (i18n "warn_wsl_tip")

        if (-not (Clack-Confirm (i18n "confirm_continue_as_admin") "n")) {
            Clack-Cancel (i18n "error_setup_failed")
            exit 1
        }
    }
}

# Check for proxy environment variables
function Test-Proxy {
    if ($env:HTTPS_PROXY -or $env:HTTP_PROXY -or $env:ALL_PROXY -or
        $env:https_proxy -or $env:http_proxy -or $env:all_proxy) {
        return $true
    }
    return $false
}

# Prompt user to enable China mirrors if no proxy is detected
function Configure-Mirrors {
    if (Test-Proxy) { return }

    Clack-Warn (i18n "mirror_no_proxy")

    if (-not (Clack-Confirm (i18n "mirror_confirm") "n")) {
        Clack-Log (i18n "mirror_skipped")
        return
    }

    # Show mirror source selection on the same continuous tree guide.
    Write-GuideLine
    Write-TreeBranch -Text "$(i18n 'mirror_select') [1]" -Color Cyan
    Clack-Log "1. $(i18n 'mirror_tuna')"
    Clack-Log "2. $(i18n 'mirror_aliyun')"
    Clack-Log "3. $(i18n 'mirror_tencent')"
    Clack-Log "4. $(i18n 'mirror_huawei')"
    Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
    if (Test-InteractiveInput) {
        $choice = Read-Host "  >"
    } else {
        Write-Host "  > 1"
        $choice = "1"
    }

    if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

    switch ($choice) {
        "2" {
            $env:UV_INDEX_URL = "https://mirrors.aliyun.com/pypi/simple/"
            $env:NPM_CONFIG_REGISTRY = "https://registry.npmmirror.com"
        }
        "3" {
            $env:UV_INDEX_URL = "https://mirrors.cloud.tencent.com/pypi/simple"
            $env:NPM_CONFIG_REGISTRY = "https://mirrors.cloud.tencent.com/npm/"
        }
        "4" {
            $env:UV_INDEX_URL = "https://repo.huaweicloud.com/repository/pypi/simple"
            $env:NPM_CONFIG_REGISTRY = "https://mirrors.huaweicloud.com/repository/npm/"
        }
        default {
            $env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
            $env:NPM_CONFIG_REGISTRY = "https://registry.npmmirror.com"
        }
    }

    # Playwright: only npmmirror CDN provides reliable browser binary mirrors
    $env:PLAYWRIGHT_DOWNLOAD_HOST = "https://cdn.npmmirror.com/binaries/playwright"

    Clack-Success (i18n "mirror_enabled")
    Clack-Log "  $(i18n 'mirror_pypi'): $env:UV_INDEX_URL"
    Clack-Log "  $(i18n 'mirror_npm'): $env:NPM_CONFIG_REGISTRY"
    Clack-Log "  $(i18n 'mirror_playwright'): $env:PLAYWRIGHT_DOWNLOAD_HOST"
}

# Check execution policy
function Test-ExecutionPolicy {
    $policy = Get-ExecutionPolicy -Scope CurrentUser
    if ($policy -eq "Restricted" -or $policy -eq "AllSigned") {
        Clack-Warn "Execution policy: $policy"
        Clack-Log "Scripts may be blocked. Run: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned"
        return $false
    }
    return $true
}

# ============================================================
# Mode Detection
# ============================================================

# Check if running in development mode
function Test-DevMode {
    # Check if we're in the markitai project directory
    # Note: .git can be a directory (normal repo) or file (worktree)
    if ((Test-Path ".\pyproject.toml") -and (Test-Path ".\.git") -and (Test-Path ".\scripts")) {
        $content = Get-Content ".\pyproject.toml" -Raw -ErrorAction SilentlyContinue
        if ($content -match "markitai") {
            return $true
        }
    }
    return $false
}

# ============================================================
# Install Source Selection (PyPI vs local repo)
# ============================================================

# Install source: "pypi" (default) or "local" (from source repo)
$script:MARKITAI_SOURCE = "pypi"
$script:MarkitaiLocalPath = $null

# Detect the markitai source repo (for the local install option)
# Checks the script location first, then cwd (irm | iex case)
# Returns: repo root path if detected, $null otherwise
function Get-MarkitaiRepoRoot {
    # 1) Via script location ($PSCommandPath is empty under irm | iex)
    if ($PSCommandPath) {
        $scriptDir = Split-Path -Parent $PSCommandPath
        $repoRoot = Split-Path -Parent $scriptDir
        $pyproject = Join-Path $repoRoot "packages/markitai/pyproject.toml"
        if (Test-Path $pyproject) {
            $content = Get-Content $pyproject -Raw -ErrorAction SilentlyContinue
            if ($content -match 'name\s*=\s*"markitai"') {
                return $repoRoot
            }
        }
    }

    # 2) Via current directory (irm | iex run from inside a checkout)
    $pyproject = Join-Path $PWD.Path "packages/markitai/pyproject.toml"
    if (Test-Path $pyproject) {
        $content = Get-Content $pyproject -Raw -ErrorAction SilentlyContinue
        if ($content -match 'name\s*=\s*"markitai"') {
            return $PWD.Path
        }
    }

    return $null
}

# Ask whether to install markitai from the local repo or PyPI
# Default (enter / non-interactive / not in repo) = PyPI
# Sets: $script:MARKITAI_SOURCE, $script:MarkitaiLocalPath
function Select-InstallSource {
    $repoRoot = Get-MarkitaiRepoRoot
    if (-not $repoRoot) { return }

    # Non-interactive (stdin redirected): never prompt, keep PyPI
    if ([Console]::IsInputRedirected) {
        Clack-Info (i18n "info_repo_noninteractive")
        return
    }

    Clack-Info "$(i18n 'repo_detected'): $repoRoot"
    if (Clack-Confirm (i18n "confirm_local_install") "n") {
        $script:MARKITAI_SOURCE = "local"
        $script:MarkitaiLocalPath = Join-Path $repoRoot "packages/markitai"
        Clack-Info (i18n "source_local")
    } else {
        Clack-Info (i18n "source_pypi")
    }
}

# Build the markitai package spec honoring install source and extras
function Get-MarkitaiPkgSpec {
    param([string]$Extras)

    $base = "markitai"
    if ($Extras) {
        $base = "markitai[$Extras]"
    }

    if ($script:MARKITAI_SOURCE -eq "local") {
        return "$base @ $($script:MarkitaiLocalPath)"
    }
    if ($script:MarkitaiVersion) {
        return "$base==$($script:MarkitaiVersion)"
    }
    return $base
}

# ============================================================
# Installation Functions
# ============================================================

# Check if uv is installed
function Test-UV {
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCmd) {
        return $false
    }
    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $version = & uv --version 2>&1 | Select-Object -First 1
    } finally {
        $ErrorActionPreference = $oldErrorAction
    }
    if ($version -and $version -notmatch "error") {
        return $true
    }
    return $false
}

# Install uv package manager
function Install-UV {
    if (Test-UV) {
        $version = (& uv --version 2>$null).Split(' ')[1]
        Clack-Success "$(i18n 'uv'): $version $(i18n 'already_installed')"
        Track-Install -Component "uv" -Status "installed"
        return $true
    }

    if (-not (Clack-Confirm (i18n "confirm_uv") "y")) {
        Clack-Skip (i18n "uv")
        Track-Install -Component "uv" -Status "skipped"
        return $false
    }

    Clack-Info "$(i18n 'installing') $(i18n 'uv')..."

    # Build install URL (with optional version)
    if ($script:UvVersion) {
        $uvUrl = "https://astral.sh/uv/$($script:UvVersion)/install.ps1"
    } else {
        $uvUrl = "https://astral.sh/uv/install.ps1"
    }

    $installResult = Invoke-PowerShellInstallerQuietly -Uri $uvUrl
    if (-not $installResult.Success) {
        Clack-Error "$(i18n 'uv') $(i18n 'failed')"
        Clack-Detail -Detail $installResult.Output -MaxLines 4
        Track-Install -Component "uv" -Status "failed"
        return $false
    }

    try {
        # Refresh PATH (User paths take precedence)
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + $env:Path

        # Check if uv command exists after PATH refresh
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uvCmd) {
            Clack-Warn "$(i18n 'uv') $(i18n 'installed') - restart PowerShell"
            Track-Install -Component "uv" -Status "installed"
            return $false
        }

        $version = (& uv --version 2>$null).Split(' ')[1]
        Clack-Success "$(i18n 'uv'): $version $(i18n 'installed')"
        Track-Install -Component "uv" -Status "installed"
        return $true
    } catch {
        Clack-Error "$(i18n 'uv') $(i18n 'failed')"
        Clack-Detail -Detail @($installResult.Output + $_.Exception.Message) -MaxLines 4
        Track-Install -Component "uv" -Status "failed"
        return $false
    }
}

# Detect/install Python via uv
function Install-Python {
    if (-not (Test-CommandExists "uv")) {
        Clack-Error (i18n "error_uv_required")
        return $false
    }

    # Try to find any supported Python (3.11-3.13)
    $uvPython = & uv python find ">=3.11,<3.14" 2>$null
    if ($uvPython -and (Test-Path $uvPython)) {
        $version = & $uvPython -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>$null
        if ($version) {
            $script:PYTHON_CMD = $uvPython
            Clack-Success "$(i18n 'python') $version"
            return $true
        }
    }

    # Not found, auto-install (3.13 as default)
    Clack-Info "$(i18n 'installing') $(i18n 'python') 3.13..."
    $installResult = Invoke-NativeQuietly { & uv python install 3.13 }
    if ($installResult.Success) {
        $uvPython = & uv python find 3.13 2>$null
        if ($uvPython -and (Test-Path $uvPython)) {
            $version = & $uvPython -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>$null
            if ($version) {
                $script:PYTHON_CMD = $uvPython
                Clack-Success "$(i18n 'python') $version $(i18n 'installed')"
                return $true
            }
        }
    }

    Clack-Error (i18n "error_python_required")
    Clack-Detail -Detail $installResult.Output -MaxLines 4
    return $false
}

function Get-MarkitaiToolsDir {
    $uvToolsDir = $null
    try { $uvToolsDir = & uv tool dir 2>$null } catch {}
    if ($uvToolsDir) { return $uvToolsDir }
    if ($env:WSL_DISTRO_NAME) { return "$HOME/.local/share/uv/tools" }
    return "$env:APPDATA\uv\tools"
}

# Install markitai (User mode)
function Install-Markitai {
    $uvToolsDir = Get-MarkitaiToolsDir

    # Build package spec with all tracked extras
    $pkg = Get-MarkitaiPkgSpec -Extras $script:MARKITAI_EXTRAS

    # Build Python command for --python argument
    $pythonArg = $script:PYTHON_CMD

    $uvExists = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvExists) {
        Clack-Error "$(i18n 'markitai') $(i18n 'failed')"
        Track-Install -Component "markitai" -Status "failed"
        return $false
    }

    $markitaiToolDir = Join-Path $uvToolsDir "markitai"
    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $exitCode = 1
    $lastOutput = @()

    if (Test-Path $markitaiToolDir) {
        # Only an unpinned PyPI install can use the receipt-based upgrade path.
        # Explicit versions and local sources must apply the exact spec below.
        if (
            $script:MARKITAI_SOURCE -ne "local" -and -not $script:MarkitaiVersion -and
            -not (Test-MarkitaiExtrasNeedUpdate)
        ) {
            try {
                $lastOutput = @(& uv tool upgrade markitai 2>&1)
                $exitCode = $LASTEXITCODE
            } catch {}
        }

        # Upgrade failed (or local source), try force reinstall
        if ($exitCode -ne 0) {
            try {
                $lastOutput = @(& uv tool install $pkg --python $pythonArg --force 2>&1)
                $exitCode = $LASTEXITCODE
            } catch {}
        }
    } else {
        # Fresh install
        Clack-Info "$(i18n 'installing') $(i18n 'markitai')..."
        try {
            $lastOutput = @(& uv tool install $pkg --python $pythonArg 2>&1)
            $exitCode = $LASTEXITCODE
        } catch {}
    }

    $ErrorActionPreference = $oldErrorAction

    if ($exitCode -eq 0) {
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + $env:Path

        $markitaiCmd = Get-Command markitai -ErrorAction SilentlyContinue
        $version = if ($markitaiCmd) { (& markitai --version 2>&1 | Select-Object -First 1).Split(' ')[-1] } else { (i18n "installed") }
        if (-not $version) { $version = (i18n "installed") }
        if (
            $script:MARKITAI_SOURCE -ne "local" -and
            $script:MarkitaiVersion -and
            $version -ne $script:MarkitaiVersion
        ) {
            Clack-Error "$(i18n 'markitai') version mismatch: expected $($script:MarkitaiVersion), got $version"
            Track-Install -Component "markitai" -Status "failed"
            return $false
        }
        Clack-Success "$(i18n 'markitai') $version"
        Track-Install -Component "markitai" -Status "installed"
        return $true
    }

    Clack-Error "$(i18n 'markitai') $(i18n 'failed')"
    Clack-Detail -Detail $lastOutput -MaxLines 6
    Track-Install -Component "markitai" -Status "failed"
    return $false
}

# Global variable tracking all needed extras (comma-separated). A fresh
# non-interactive install starts with core only unless explicitly opted in.
$script:MARKITAI_EXTRAS = ""
$script:MARKITAI_RECEIPT_EXTRAS = @()
$script:MARKITAI_ALL_FALLBACK_EXTRAS = "browser,extra-fetch,kreuzberg,svg,heif,serve"
if ((Test-InteractiveInput) -or (Test-OptionalInstallRequested)) {
    $script:MARKITAI_EXTRAS = "browser"
}

# Return true when an extra is covered by the combined spec. `all` is
# canonical and includes every individual extra, including `serve`.
function Test-MarkitaiExtraEnabled {
    param([string]$ExtraName)

    if ($script:MARKITAI_EXTRAS -eq "all") { return $true }
    return (($script:MARKITAI_EXTRAS -split ",") -contains $ExtraName)
}

# Track a markitai extra for the next combined installation.
function Install-MarkitaiExtra {
    param([string]$ExtraName)

    if ([string]::IsNullOrWhiteSpace($ExtraName)) { return }
    if ($ExtraName -eq "all") {
        $script:MARKITAI_EXTRAS = "all"
        return
    }
    if (Test-MarkitaiExtraEnabled -ExtraName $ExtraName) { return }
    if ([string]::IsNullOrEmpty($script:MARKITAI_EXTRAS)) {
        $script:MARKITAI_EXTRAS = $ExtraName
    } else {
        $script:MARKITAI_EXTRAS = "$($script:MARKITAI_EXTRAS),$ExtraName"
    }
}

# Preserve every extra recorded by uv before asking for new capabilities.
function Import-MarkitaiReceiptExtras {
    $uvToolsDir = Get-MarkitaiToolsDir
    $receiptFile = Join-Path $uvToolsDir "markitai\uv-receipt.toml"
    $script:MARKITAI_RECEIPT_EXTRAS = @()
    if (-not (Test-Path $receiptFile)) { return }

    $receipt = Get-Content $receiptFile -Raw -ErrorAction SilentlyContinue
    if ($receipt -match 'extras\s*=\s*\[([^\]]*)\]') {
        $extrasStr = $Matches[1] -replace '"', '' -replace '\s', ''
        $script:MARKITAI_RECEIPT_EXTRAS = @(
            $extrasStr -split ',' | Where-Object { $_ }
        )
        foreach ($extra in $script:MARKITAI_RECEIPT_EXTRAS) {
            Install-MarkitaiExtra -ExtraName $extra
        }
    }
}

function Test-MarkitaiReceiptHasExtra {
    param([string]$ExtraName)

    return (
        ($script:MARKITAI_RECEIPT_EXTRAS -contains "all") -or
        ($script:MARKITAI_RECEIPT_EXTRAS -contains $ExtraName)
    )
}

# A generic `uv tool upgrade` only reuses the old receipt. Return true when
# the combined target has a newly selected extra and needs an exact reinstall.
function Test-MarkitaiExtrasNeedUpdate {
    foreach ($extra in ($script:MARKITAI_EXTRAS -split ",")) {
        if ($extra -and -not (Test-MarkitaiReceiptHasExtra -ExtraName $extra)) {
            return $true
        }
    }
    return $false
}

# Resolve Web UI support before installing markitai so `serve` joins the same
# package spec as browser/all/existing extras instead of replacing them later.
function Select-MarkitaiServe {
    if (Test-MarkitaiExtraEnabled -ExtraName "serve") { return }
    if (Confirm-OptionalInstall (i18n "confirm_serve") "y") {
        Install-MarkitaiExtra -ExtraName "serve"
    }
}

function Track-MarkitaiServe {
    if (Test-MarkitaiExtraEnabled -ExtraName "serve") {
        Track-Install -Component "serve" -Status "installed"
    } else {
        Track-Install -Component "serve" -Status "skipped"
    }
}

# Finalize markitai extras after all optional components are resolved.
# Merges `markitai doctor --suggest-extras` output with manually tracked
# MARKITAI_EXTRAS (from CLI install functions), so nothing is lost.
function Finalize-MarkitaiExtras {
    if (-not (Test-InteractiveInput) -and -not (Test-OptionalInstallRequested)) {
        return
    }

    # Merge suggested extras INTO manually tracked set (not replace)
    try {
        $suggested = & markitai doctor --suggest-extras 2>$null
        if ($suggested) {
            foreach ($extra in ($suggested.Trim() -split ',')) {
                if ($extra) { Install-MarkitaiExtra -ExtraName $extra }
            }
        }
    } catch {}

    # Refresh the receipt after the initial install, then compare exact extra
    # names (with `all` treated as a superset).
    Import-MarkitaiReceiptExtras
    if (-not (Test-MarkitaiExtrasNeedUpdate)) { return }

    # Reinstall with all extras (progressive fallback on failure)
    $pkg = Get-MarkitaiPkgSpec -Extras $script:MARKITAI_EXTRAS

    $primaryResult = Invoke-NativeQuietly {
        & uv tool install $pkg --python $script:PYTHON_CMD --force
    }
    $installOk = $primaryResult.Success
    $fallbackResult = $null

    if (-not $installOk) {
        # Full install failed — retry without SDK-dependent extras
        $sdkExtras = @("claude-agent", "copilot")
        $safeExtras = @()
        $skipped = @()
        foreach ($e in ($script:MARKITAI_EXTRAS -split ",")) {
            if ($e -eq "all") {
                $safeExtras += ($script:MARKITAI_ALL_FALLBACK_EXTRAS -split ",")
                $skipped += $sdkExtras
            } elseif ($sdkExtras -contains $e) {
                $skipped += $e
            } else {
                $safeExtras += $e
            }
        }
        if ($safeExtras.Count -gt 0) {
            $safeList = $safeExtras -join ","
            $pkg = Get-MarkitaiPkgSpec -Extras $safeList
            $fallbackResult = Invoke-NativeQuietly {
                & uv tool install $pkg --python $script:PYTHON_CMD --force
            }
            if ($fallbackResult.Success) {
                if ($skipped.Count -gt 0) {
                    Clack-Warn "$(i18n 'skipped') extras: $($skipped -join ', ') (SDK $(i18n 'not_found'))"
                }
                $installOk = $true
            }
        }
        if (-not $installOk) {
            Clack-Warn "$(i18n 'markitai') extras update $(i18n 'failed')"
            $details = @($primaryResult.Output)
            if ($fallbackResult) { $details += $fallbackResult.Output }
            Clack-Detail -Detail $details -MaxLines 3 -Context "markitai extras"
        }
    }
}

# Sync project dependencies (Dev mode)
function Sync-Dependencies {
    $projectRoot = Get-ProjectRoot
    Clack-Info "$(i18n 'info_project_dir'): $projectRoot"

    Push-Location $projectRoot

    try {
        Clack-Info (i18n "info_syncing_deps")
        $oldErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $syncResult = & uv sync --all-groups --all-extras --python $script:PYTHON_CMD 2>&1
        $ErrorActionPreference = $oldErrorAction
        if ($LASTEXITCODE -eq 0) {
            Clack-Success (i18n "info_deps_synced")
            Track-Install -Component "python_deps" -Status "installed"
            return $true
        } else {
            Clack-Error "$(i18n 'python_deps') $(i18n 'failed')"
            Clack-Detail -Detail $syncResult -MaxLines 6
            Track-Install -Component "python_deps" -Status "failed"
            return $false
        }
    } finally {
        Pop-Location
    }
}

# Install pre-commit hooks (Dev mode)
function Install-PreCommit {
    $projectRoot = Get-ProjectRoot
    Push-Location $projectRoot

    try {
        if (Test-Path ".pre-commit-config.yaml") {
            $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
            if ($uvCmd) {
                $oldErrorAction = $ErrorActionPreference
                $ErrorActionPreference = "Continue"
                $precommitResult = & uv run pre-commit install 2>&1
                $ErrorActionPreference = $oldErrorAction
                if ($LASTEXITCODE -eq 0) {
                    Clack-Success (i18n "info_precommit_installed")
                    Track-Install -Component "precommit" -Status "installed"
                    return $true
                } else {
                    Clack-Warn "$(i18n 'precommit') $(i18n 'failed')"
                    Clack-Detail -Detail $precommitResult -MaxLines 3
                    return $false
                }
            }
        } else {
            Clack-Skip (i18n "precommit")
            return $false
        }
    } finally {
        Pop-Location
    }
}

# Check if Playwright browser is installed
function Test-PlaywrightBrowser {
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCmd) {
        try {
            $uvToolDir = & uv tool dir 2>$null
            if ($uvToolDir) {
                $markitaiPlaywright = Join-Path $uvToolDir "markitai\Scripts\playwright.exe"
                if (Test-Path $markitaiPlaywright) {
                    $cacheDir = Join-Path $env:LOCALAPPDATA "ms-playwright"
                    if (Test-Path $cacheDir) {
                        $chromiumDirs = Get-ChildItem -Path $cacheDir -Directory -Filter "chromium-*" -ErrorAction SilentlyContinue
                        if ($chromiumDirs) {
                            return $true
                        }
                    }
                }
            }
        } catch {}
    }
    return $false
}

# Install Playwright browser (Optional)
function Install-OptionalPlaywright {
    if (Test-PlaywrightBrowser) {
        Clack-Success "$(i18n 'playwright') $(i18n 'already_installed')"
        Track-Install -Component "playwright" -Status "installed"
        return $true
    }

    Clack-Info (i18n "info_playwright_purpose")

    if (-not (Confirm-OptionalInstall (i18n "confirm_playwright") "y")) {
        Clack-Skip (i18n "playwright")
        Track-Install -Component "playwright" -Status "skipped"
        return $false
    }

    Clack-Info "$(i18n 'installing') $(i18n 'playwright')..."

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $lastOutput = @()

    # Method 1: Use playwright from markitai's uv tool environment
    $markitaiPlaywright = $null
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCmd) {
        try {
            $uvToolDir = & uv tool dir 2>$null
            if ($uvToolDir) {
                $markitaiPlaywright = Join-Path $uvToolDir "markitai\Scripts\playwright.exe"
            }
        } catch {}
    }
    if (-not $markitaiPlaywright -or -not (Test-Path $markitaiPlaywright)) {
        $markitaiPlaywright = Join-Path $env:APPDATA "uv\tools\markitai\Scripts\playwright.exe"
    }

    if (Test-Path $markitaiPlaywright) {
        try {
            $lastOutput = @(& $markitaiPlaywright install chromium 2>&1)
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Clack-Success "$(i18n 'playwright') $(i18n 'installed')"
                Track-Install -Component "playwright" -Status "installed"
                return $true
            }
        } catch {}
    }

    # Method 2: Fallback to Python module
    if ($script:PYTHON_CMD) {
        try {
            $lastOutput = @(& $script:PYTHON_CMD -m playwright install chromium 2>&1)
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Clack-Success "$(i18n 'playwright') $(i18n 'installed')"
                Track-Install -Component "playwright" -Status "installed"
                return $true
            }
        } catch {}
    }

    $ErrorActionPreference = $oldErrorAction
    Clack-Error "$(i18n 'playwright') $(i18n 'failed')"
    Clack-Detail -Detail $lastOutput -MaxLines 4
    Track-Install -Component "playwright" -Status "failed"
    return $false
}

# Install LibreOffice (Optional)
function Install-OptionalLibreOffice {
    # Check if already installed
    $soffice = Get-Command soffice -ErrorAction SilentlyContinue
    if ($soffice) {
        Clack-Success "$(i18n 'libreoffice') $(i18n 'already_installed')"
        Track-Install -Component "libreoffice" -Status "installed"
        return $true
    }

    $commonPaths = @(
        "${env:ProgramFiles}\LibreOffice\program\soffice.exe",
        "${env:ProgramFiles(x86)}\LibreOffice\program\soffice.exe"
    )
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            Clack-Success "$(i18n 'libreoffice') $(i18n 'already_installed')"
            Track-Install -Component "libreoffice" -Status "installed"
            return $true
        }
    }

    Clack-Info (i18n "info_libreoffice_purpose")

    if (-not (Confirm-OptionalInstall (i18n "confirm_libreoffice") "n")) {
        Clack-Skip (i18n "libreoffice")
        Track-Install -Component "libreoffice" -Status "skipped"
        return $false
    }

    Clack-Info "$(i18n 'installing') $(i18n 'libreoffice')..."

    # Priority: winget > scoop > choco
    $lastResult = $null
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        $lastResult = Invoke-NativeQuietly { & winget install TheDocumentFoundation.LibreOffice --accept-package-agreements --accept-source-agreements }
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'libreoffice') $(i18n 'installed')"
            Track-Install -Component "libreoffice" -Status "installed"
            return $true
        }
    }

    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        $null = Invoke-NativeQuietly { & scoop bucket add extras }
        $lastResult = Invoke-NativeQuietly { & scoop install extras/libreoffice }
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'libreoffice') $(i18n 'installed')"
            Track-Install -Component "libreoffice" -Status "installed"
            return $true
        }
    }

    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        $lastResult = Invoke-NativeQuietly { & choco install libreoffice-fresh -y }
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'libreoffice') $(i18n 'installed')"
            Track-Install -Component "libreoffice" -Status "installed"
            return $true
        }
    }

    Clack-Error "$(i18n 'libreoffice') $(i18n 'failed')"
    if ($lastResult) { Clack-Detail -Detail $lastResult.Output -MaxLines 3 }
    Track-Install -Component "libreoffice" -Status "failed"
    return $false
}

# Install FFmpeg (Optional)
function Install-OptionalFFmpeg {
    # Check if already installed
    $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpegCmd) {
        $version = & ffmpeg -version 2>&1 | Select-Object -First 1
        if ($version -match "ffmpeg version ([^\s]+)") {
            Clack-Success "$(i18n 'ffmpeg'): $($Matches[1]) $(i18n 'already_installed')"
        } else {
            Clack-Success "$(i18n 'ffmpeg') $(i18n 'already_installed')"
        }
        Track-Install -Component "ffmpeg" -Status "installed"
        return $true
    }

    Clack-Info (i18n "info_ffmpeg_purpose")

    if (-not (Confirm-OptionalInstall (i18n "confirm_ffmpeg") "n")) {
        Clack-Skip (i18n "ffmpeg")
        Track-Install -Component "ffmpeg" -Status "skipped"
        return $false
    }

    Clack-Info "$(i18n 'installing') $(i18n 'ffmpeg')..."

    # Priority: winget > scoop > choco
    $lastResult = $null
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        $lastResult = Invoke-NativeQuietly { & winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements }
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'ffmpeg') $(i18n 'installed')"
            Track-Install -Component "ffmpeg" -Status "installed"
            return $true
        }
    }

    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        $lastResult = Invoke-NativeQuietly { & scoop install ffmpeg }
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'ffmpeg') $(i18n 'installed')"
            Track-Install -Component "ffmpeg" -Status "installed"
            return $true
        }
    }

    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        $lastResult = Invoke-NativeQuietly { & choco install ffmpeg -y }
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'ffmpeg') $(i18n 'installed')"
            Track-Install -Component "ffmpeg" -Status "installed"
            return $true
        }
    }

    Clack-Error "$(i18n 'ffmpeg') $(i18n 'failed')"
    if ($lastResult) { Clack-Detail -Detail $lastResult.Output -MaxLines 3 }
    Track-Install -Component "ffmpeg" -Status "failed"
    return $false
}

# Install Claude Code CLI (Optional)
function Install-OptionalClaudeCLI {
    # Check if already installed
    $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($claudeCmd) {
        $version = & claude --version 2>&1 | Select-Object -First 1
        Clack-Success "$(i18n 'claude_cli'): $version $(i18n 'already_installed')"
        Install-MarkitaiExtra -ExtraName "claude-agent"
        Track-Install -Component "claude_cli" -Status "installed"
        return $true
    }

    if (-not (Confirm-OptionalInstall (i18n "confirm_claude_cli") "n")) {
        Clack-Skip (i18n "claude_cli")
        Track-Install -Component "claude_cli" -Status "skipped"
        return $false
    }

    Clack-Info "$(i18n 'installing') $(i18n 'claude_cli')..."

    # Prefer official install script (PowerShell)
    $claudeUrl = "https://claude.ai/install.ps1"
    $installResult = Invoke-PowerShellInstallerQuietly -Uri $claudeUrl
    $lastDetail = $installResult.Output
    if ($installResult.Success) {
        # Child-process PATH changes do not flow back; reload persisted paths.
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + $env:Path
        $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
        if ($claudeCmd) {
            Clack-Success "$(i18n 'claude_cli') $(i18n 'installed')"
            # Also install the SDK extra
            Install-MarkitaiExtra -ExtraName "claude-agent" | Out-Null
            Track-Install -Component "claude_cli" -Status "installed"
            return $true
        }
    }

    # Fallback: npm/pnpm
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        $lastResult = Invoke-NativeQuietly { & pnpm add -g @anthropic-ai/claude-code }
        $lastDetail = $lastResult.Output
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'claude_cli') $(i18n 'installed')"
            Install-MarkitaiExtra -ExtraName "claude-agent" | Out-Null
            Track-Install -Component "claude_cli" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        $lastResult = Invoke-NativeQuietly { & npm install -g @anthropic-ai/claude-code }
        $lastDetail = $lastResult.Output
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'claude_cli') $(i18n 'installed')"
            Install-MarkitaiExtra -ExtraName "claude-agent" | Out-Null
            Track-Install -Component "claude_cli" -Status "installed"
            return $true
        }
    }

    Clack-Error "$(i18n 'claude_cli') $(i18n 'failed')"
    Clack-Detail -Detail $lastDetail -MaxLines 3
    Track-Install -Component "claude_cli" -Status "failed"
    return $false
}

# Install Copilot CLI (Optional)
function Install-OptionalCopilotCLI {
    # Check if already installed
    $copilotCmd = Get-Command copilot -ErrorAction SilentlyContinue
    if ($copilotCmd) {
        $version = & copilot --version 2>&1 | Select-Object -First 1
        Clack-Success "$(i18n 'copilot_cli'): $version $(i18n 'already_installed')"
        Install-MarkitaiExtra -ExtraName "copilot"
        Track-Install -Component "copilot_cli" -Status "installed"
        return $true
    }

    if (-not (Confirm-OptionalInstall (i18n "confirm_copilot_cli") "n")) {
        Clack-Skip (i18n "copilot_cli")
        Track-Install -Component "copilot_cli" -Status "skipped"
        return $false
    }

    Clack-Info "$(i18n 'installing') $(i18n 'copilot_cli')..."

    # Prefer WinGet on Windows
    $lastResult = $null
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        $lastResult = Invoke-NativeQuietly { & winget install GitHub.Copilot --accept-package-agreements --accept-source-agreements }
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'copilot_cli') $(i18n 'installed')"
            # Also install the SDK extra
            Install-MarkitaiExtra -ExtraName "copilot" | Out-Null
            Track-Install -Component "copilot_cli" -Status "installed"
            return $true
        }
    }

    # Fallback: npm/pnpm
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        $lastResult = Invoke-NativeQuietly { & pnpm add -g @github/copilot }
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'copilot_cli') $(i18n 'installed')"
            Install-MarkitaiExtra -ExtraName "copilot" | Out-Null
            Track-Install -Component "copilot_cli" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        $lastResult = Invoke-NativeQuietly { & npm install -g @github/copilot }
        if ($lastResult.Success) {
            Clack-Success "$(i18n 'copilot_cli') $(i18n 'installed')"
            Install-MarkitaiExtra -ExtraName "copilot" | Out-Null
            Track-Install -Component "copilot_cli" -Status "installed"
            return $true
        }
    }

    Clack-Error "$(i18n 'copilot_cli') $(i18n 'failed')"
    if ($lastResult) { Clack-Detail -Detail $lastResult.Output -MaxLines 3 }
    Track-Install -Component "copilot_cli" -Status "failed"
    return $false
}

# Print one compact summary group on the shared outer guide.
function Write-SummaryGroup {
    param(
        [string]$Title,
        [string[]]$Components,
        [string]$Symbol,
        [ConsoleColor]$Color
    )

    if (@($Components).Count -eq 0) { return }

    Write-GuideLine
    Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
    Write-Host "  $Title"
    foreach ($comp in $Components) {
        Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
        Write-Host "    " -NoNewline
        Write-Host $Symbol -ForegroundColor $Color -NoNewline
        Write-Host " $(i18n $comp)"
    }
}

# Print installation summary
function Print-Summary {
    Clack-Section (i18n "section_summary")

    Write-SummaryGroup `
        -Title (i18n "summary_installed") `
        -Components $script:INSTALLED_COMPONENTS `
        -Symbol $S_CHECK `
        -Color Green
    Write-SummaryGroup `
        -Title (i18n "summary_skipped") `
        -Components $script:SKIPPED_COMPONENTS `
        -Symbol $S_CIRCLE `
        -Color Yellow
    Write-SummaryGroup `
        -Title (i18n "summary_failed") `
        -Components $script:FAILED_COMPONENTS `
        -Symbol $S_CROSS `
        -Color Red

    # Empty line before docs link
    Clack-Log ""
    Clack-Info "$(i18n 'info_docs'): https://markitai.dev"
    Clack-Info "$(i18n 'info_issues'): https://github.com/Ynewtime/markitai/issues"
}

# Print user mode completion message
# Returns $true when the `mkai` short alias resolves to markitai, $false when
# it is absent or shadowed by a different command (which is warned about).
# The full `markitai` command always works.
function Test-MkaiAlias {
    if (-not (Get-Command mkai -ErrorAction SilentlyContinue)) { return $false }
    $ver = (& mkai --version 2>$null) -join ' '
    if ($ver -match 'markitai') { return $true }
    $path = (Get-Command mkai -ErrorAction SilentlyContinue).Source
    Clack-Warn "$(i18n 'mkai_conflict'): $path"
    return $false
}

function Print-UserCompletion {
    $lines = @(
        "$(i18n 'interactive_mode'):",
        "  markitai -I"
    )
    # Only advertise the serve command when its dependencies were selected.
    if (Test-MarkitaiExtraEnabled -ExtraName "serve") {
        $lines += @(
            "",
            "$(i18n 'start_web_ui'):",
            "  markitai serve"
        )
    }
    $lines += @(
        "",
        "$(i18n 'configure_llm'):",
        "  markitai init",
        "",
        "$(i18n 'convert_file'):",
        "  markitai file.pdf",
        "",
        "$(i18n 'show_help'):",
        "  markitai --help"
    )
    # Only advertise the short alias when it actually resolves to markitai.
    if (Test-MkaiAlias) {
        $lines += @("", (i18n 'short_alias'))
    }
    Clack-Note (i18n "getting_started") @lines
}

# Print dev mode completion message
function Print-DevCompletion {
    $projectRoot = Get-ProjectRoot
    Clack-Note (i18n "quick_start") `
        "$(i18n 'configure_env'):" `
        "  copy .env.example .env" `
        "" `
        "$(i18n 'interactive_mode'):" `
        "  uv run markitai -I" `
        "" `
        "$(i18n 'start_web_ui'):" `
        "  uv run markitai serve" `
        "" `
        "$(i18n 'run_tests'):" `
        "  uv run pytest" `
        "" `
        "$(i18n 'run_cli'):" `
        "  uv run markitai --help"
}

# Initialize markitai config (silent, first install only)
# Skips when a config already exists: `init --yes` would silently append
# newly detected providers to the user's model_list. The detector only records
# a local CLI after login succeeds and its finalized runtime SDK is importable.
function Initialize-Config {
    if (Test-Path "$HOME/.markitai/config.json") {
        return
    }
    $markitaiExists = Get-Command markitai -ErrorAction SilentlyContinue
    if ($markitaiExists) {
        try {
            $null = & markitai init --yes 2>$null
        } catch {}
    }
}

# ============================================================
# Main Entry Point
# ============================================================

# User mode main flow
function Run-UserSetup {
    Clack-Intro (i18n "welcome")
    Clack-Info (i18n "mode_user")
    Test-ExecutionPolicy | Out-Null
    Test-AdminWarning
    Test-WSLWarning
    Configure-Mirrors
    Select-InstallSource

    Clack-Section (i18n "section_core")
    if (-not (Install-UV)) { Print-Summary; Clack-Cancel (i18n "error_setup_failed"); exit 1 }
    if (-not (Install-Python)) { Print-Summary; Clack-Cancel (i18n "error_setup_failed"); exit 1 }
    Import-MarkitaiReceiptExtras
    Select-MarkitaiServe
    if (-not (Install-Markitai)) { Print-Summary; Clack-Cancel (i18n "error_setup_failed"); exit 1 }
    Track-MarkitaiServe

    Clack-Section (i18n "section_optional")
    Invoke-OptionalStep -Component "playwright" -Action { Install-OptionalPlaywright }
    Invoke-OptionalStep -Component "libreoffice" -Action { Install-OptionalLibreOffice }
    Invoke-OptionalStep -Component "ffmpeg" -Action { Install-OptionalFFmpeg }

    Clack-Section (i18n "section_llm_cli")
    Invoke-OptionalStep -Component "claude_cli" -Action { Install-OptionalClaudeCLI }
    Invoke-OptionalStep -Component "copilot_cli" -Action { Install-OptionalCopilotCLI }

    Invoke-BestEffortStep `
        -FailureMessage "$(i18n 'markitai') extras update $(i18n 'failed')" `
        -Action { Finalize-MarkitaiExtras }

    Initialize-Config 2>$null | Out-Null

    Print-Summary
    Print-UserCompletion
    if ($script:FAILED_COMPONENTS.Count -gt 0) {
        Clack-OutroWarning (i18n "setup_complete_with_warnings")
    } else {
        Clack-Outro (i18n "setup_complete")
    }
}

# Dev mode main flow
function Run-DevSetup {
    Clack-Intro (i18n "welcome")
    Clack-Info (i18n "mode_dev")
    Test-ExecutionPolicy | Out-Null
    Test-AdminWarning
    Test-WSLWarning
    Configure-Mirrors

    Clack-Section (i18n "section_prerequisites")
    if (-not (Install-UV)) { Print-Summary; Clack-Cancel (i18n "error_setup_failed"); exit 1 }
    if (-not (Install-Python)) { Print-Summary; Clack-Cancel (i18n "error_setup_failed"); exit 1 }

    Clack-Section (i18n "section_dev_env")
    if (-not (Sync-Dependencies)) { Print-Summary; Clack-Cancel (i18n "error_setup_failed"); exit 1 }
    Invoke-OptionalStep -Component "precommit" -Action { Install-PreCommit }

    Clack-Section (i18n "section_optional")
    Invoke-OptionalStep -Component "playwright" -Action { Install-OptionalPlaywright }
    Invoke-OptionalStep -Component "libreoffice" -Action { Install-OptionalLibreOffice }
    Invoke-OptionalStep -Component "ffmpeg" -Action { Install-OptionalFFmpeg }

    Clack-Section (i18n "section_llm_cli")
    Invoke-OptionalStep -Component "claude_cli" -Action { Install-OptionalClaudeCLI }
    Invoke-OptionalStep -Component "copilot_cli" -Action { Install-OptionalCopilotCLI }

    Print-Summary
    Print-DevCompletion
    if ($script:FAILED_COMPONENTS.Count -gt 0) {
        Clack-OutroWarning (i18n "dev_setup_complete_with_warnings")
    } else {
        Clack-Outro (i18n "dev_setup_complete")
    }
}

# Main entry point
function Main {
    if (Test-DevMode) {
        Run-DevSetup
    } else {
        Run-UserSetup
    }
}

# Run main function behind a final exception boundary so the tree is never
# left open and PowerShell never emits its verbose red stack formatting.
try {
    Main
} catch {
    Complete-UnexpectedFailure -ErrorRecord $_
    exit 1
} finally {
    # Pipeline cancellation (for example Ctrl+C) can bypass catch, but
    # PowerShell still runs finally. Best-effort closure avoids a dangling │.
    if ($script:CLACK_SESSION_ACTIVE -and -not $script:CLACK_SESSION_CLOSED) {
        try {
            Clack-Error (i18n "error_unexpected")
            Clack-Cancel (i18n "error_setup_failed")
        } catch {}
    }
}
