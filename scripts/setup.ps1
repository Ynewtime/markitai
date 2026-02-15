# Markitai Setup Script - Unified installer with i18n support
# PowerShell 5.1+
# Auto-detects: language (en/zh), mode (user/dev)
#
# Usage:
#   irm https://markitai.ynewtime.com/setup.ps1 | iex    # User install
#   .\scripts\setup.ps1                                   # Dev setup (in repo)

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
            "dev_setup_complete"        { return "开发环境设置完成!" }

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

            # Components
            "uv"                        { return "uv 包管理器" }
            "python"                    { return "Python" }
            "markitai"                  { return "markitai" }
            "playwright"                { return "Playwright 浏览器" }
            "libreoffice"               { return "LibreOffice" }
            "ffmpeg"                    { return "FFmpeg" }
            "claude_cli"                { return "Claude Code CLI" }
            "copilot_cli"               { return "Copilot CLI" }
            "precommit"                 { return "pre-commit hooks" }
            "python_deps"               { return "Python 依赖" }

            # Confirmations
            "confirm_playwright"        { return "安装 Playwright 浏览器? (用于 JS 渲染页面)" }
            "confirm_libreoffice"       { return "安装 LibreOffice? (用于 Office 文档转换)" }
            "confirm_ffmpeg"            { return "安装 FFmpeg? (用于音视频处理)" }
            "confirm_claude_cli"        { return "安装 Claude Code CLI? (使用 Claude 订阅)" }
            "confirm_copilot_cli"       { return "安装 Copilot CLI? (使用 GitHub Copilot 订阅)" }
            "confirm_uv"                { return "安装 uv 包管理器?" }
            "confirm_continue_as_admin" { return "以管理员身份继续?" }

            # Info messages
            "info_libreoffice_purpose"  { return "LibreOffice 用于将 Office 文档 (docx/xlsx/pptx) 转换为 Markdown" }
            "info_ffmpeg_purpose"       { return "FFmpeg 用于处理音频和视频文件" }
            "info_playwright_purpose"   { return "Playwright 用于获取 JavaScript 渲染的网页内容" }
            "info_project_dir"          { return "项目目录" }
            "info_docs"                 { return "文档" }
            "info_issues"               { return "问题反馈" }
            "info_syncing_deps"         { return "正在同步依赖..." }
            "info_deps_synced"          { return "依赖同步完成" }
            "info_precommit_installed"  { return "pre-commit hooks 已安装" }

            # Error messages
            "error_uv_required"         { return "需要安装 uv 包管理器" }
            "error_python_required"     { return "需要安装 Python 3.11+" }
            "error_setup_failed"        { return "安装失败" }

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
            "convert_file"              { return "转换文件" }
            "show_help"                 { return "显示帮助" }

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
            "dev_setup_complete"        { return "Development environment ready!" }

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

            # Components
            "uv"                        { return "uv package manager" }
            "python"                    { return "Python" }
            "markitai"                  { return "markitai" }
            "playwright"                { return "Playwright browser" }
            "libreoffice"               { return "LibreOffice" }
            "ffmpeg"                    { return "FFmpeg" }
            "claude_cli"                { return "Claude Code CLI" }
            "copilot_cli"               { return "Copilot CLI" }
            "precommit"                 { return "pre-commit hooks" }
            "python_deps"               { return "Python dependencies" }

            # Confirmations
            "confirm_playwright"        { return "Install Playwright browser? (for JS-rendered pages)" }
            "confirm_libreoffice"       { return "Install LibreOffice? (for Office document conversion)" }
            "confirm_ffmpeg"            { return "Install FFmpeg? (for audio/video processing)" }
            "confirm_claude_cli"        { return "Install Claude Code CLI? (use your Claude subscription)" }
            "confirm_copilot_cli"       { return "Install Copilot CLI? (use your GitHub Copilot subscription)" }
            "confirm_uv"                { return "Install uv package manager?" }
            "confirm_continue_as_admin" { return "Continue as administrator?" }

            # Info messages
            "info_libreoffice_purpose"  { return "LibreOffice converts Office documents (docx/xlsx/pptx) to Markdown" }
            "info_ffmpeg_purpose"       { return "FFmpeg processes audio and video files" }
            "info_playwright_purpose"   { return "Playwright fetches JavaScript-rendered web pages" }
            "info_project_dir"          { return "Project directory" }
            "info_docs"                 { return "Documentation" }
            "info_issues"               { return "Issues" }
            "info_syncing_deps"         { return "Syncing dependencies..." }
            "info_deps_synced"          { return "Dependencies synced" }
            "info_precommit_installed"  { return "pre-commit hooks installed" }

            # Error messages
            "error_uv_required"         { return "uv package manager is required" }
            "error_python_required"     { return "Python 3.11+ is required" }
            "error_setup_failed"        { return "Setup failed" }

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
            "convert_file"              { return "Convert a file" }
            "show_help"                 { return "Show help" }

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
# Clack-style Visual Components
# Inspired by @clack/prompts - beautiful CLI with guide lines
# ============================================================

# Unicode box-drawing characters
$S_BAR = [char]0x2502         # │
$S_BAR_H = [char]0x2500       # ─
$S_CORNER_TOP = [char]0x250C  # ┌
$S_CORNER_BOT = [char]0x2514  # └
$S_STEP_ACTIVE = [char]0x25C6 # ◆
$S_STEP_SUBMIT = [char]0x25C7 # ◇
$S_CHECK = [char]0x2713       # ✓
$S_CROSS = [char]0x2717       # ✗
$S_ARROW = [char]0x2192       # →
$S_CIRCLE = [char]0x25CB      # ○
$S_BOX_TOP = [char]0x256D     # ╭
$S_BOX_BOT = [char]0x2570     # ╰

# Session intro - start of CLI flow
function Clack-Intro {
    param([string]$Title)
    Write-Host ""
    Write-Host $S_CORNER_TOP -ForegroundColor DarkGray -NoNewline
    Write-Host "  $Title"
    Write-Host $S_BAR -ForegroundColor DarkGray
}

# Session outro - end of CLI flow
function Clack-Outro {
    param([string]$Message)
    Write-Host $S_BAR -ForegroundColor DarkGray
    Write-Host $S_CORNER_BOT -ForegroundColor DarkGray -NoNewline
    Write-Host "  $Message" -ForegroundColor Green
    Write-Host ""
}

# Section header with active marker
function Clack-Section {
    param([string]$Title)
    Write-Host $S_BAR -ForegroundColor DarkGray
    Write-Host $S_STEP_ACTIVE -ForegroundColor Magenta -NoNewline
    Write-Host "  $Title"
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
    Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
    Write-Host "  $Message"
}

# Confirm prompt with guide line
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

    Write-Host $S_BAR -ForegroundColor DarkGray
    Write-Host $S_STEP_SUBMIT -ForegroundColor Cyan -NoNewline
    $answer = Read-Host "  $Prompt [$hint]"

    if ([string]::IsNullOrWhiteSpace($answer)) {
        $answer = $Default
    }

    return $answer -match "^[Yy]"
}

# Note/message box with guide line
function Clack-Note {
    param(
        [Parameter(Position=0)]
        [string]$Title,
        [Parameter(Position=1, ValueFromRemainingArguments=$true)]
        [string[]]$Lines
    )

    Write-Host $S_BAR -ForegroundColor DarkGray
    Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
    Write-Host "  " -NoNewline
    Write-Host $S_BOX_TOP -ForegroundColor DarkGray -NoNewline
    Write-Host $S_BAR_H -ForegroundColor DarkGray -NoNewline
    Write-Host " $Title"

    foreach ($line in $Lines) {
        Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
        Write-Host "  " -NoNewline
        Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
        Write-Host "  $line"
    }

    Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
    Write-Host "  " -NoNewline
    Write-Host $S_BOX_BOT -ForegroundColor DarkGray -NoNewline
    Write-Host $S_BAR_H -ForegroundColor DarkGray
}

# Cancel message
function Clack-Cancel {
    param([string]$Message)
    Write-Host $S_BAR -ForegroundColor DarkGray
    Write-Host $S_CORNER_BOT -ForegroundColor DarkGray -NoNewline
    Write-Host "  $Message" -ForegroundColor Red
    Write-Host ""
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

    # Show mirror source selection
    Write-Host $S_BAR -ForegroundColor DarkGray
    Write-Host $S_STEP_SUBMIT -ForegroundColor Cyan -NoNewline
    $choice = Read-Host "  $(i18n 'mirror_select') [1]`n$S_BAR  1. $(i18n 'mirror_tuna')`n$S_BAR  2. $(i18n 'mirror_aliyun')`n$S_BAR  3. $(i18n 'mirror_tencent')`n$S_BAR  4. $(i18n 'mirror_huawei')`n$S_BAR  >"

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

    try {
        $null = Invoke-RestMethod $uvUrl | Invoke-Expression 2>$null

        # Refresh PATH (User paths take precedence)
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")

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

    $uvPython = & uv python find 3.13 2>$null
    if ($uvPython -and (Test-Path $uvPython)) {
        $version = & $uvPython -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>$null
        if ($version) {
            $script:PYTHON_CMD = $uvPython
            Clack-Success "$(i18n 'python') $version"
            return $true
        }
    }

    # Not found, auto-install
    Clack-Info "$(i18n 'installing') $(i18n 'python') 3.13..."
    $null = & uv python install 3.13 2>&1
    if ($LASTEXITCODE -eq 0) {
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
    return $false
}

# Install markitai (User mode)
function Install-Markitai {
    # Detect existing extras from uv receipt to preserve on upgrade
    $uvToolsDir = $null
    try { $uvToolsDir = & uv tool dir 2>$null } catch {}
    if (-not $uvToolsDir) {
        if ($env:WSL_DISTRO_NAME) {
            $uvToolsDir = "$HOME/.local/share/uv/tools"
        } else {
            $uvToolsDir = "$env:APPDATA\uv\tools"
        }
    }
    $receiptFile = Join-Path $uvToolsDir "markitai\uv-receipt.toml"
    if (Test-Path $receiptFile) {
        $receipt = Get-Content $receiptFile -Raw -ErrorAction SilentlyContinue
        if ($receipt -match "claude-agent") { Install-MarkitaiExtra -ExtraName "claude-agent" }
        if ($receipt -match "copilot") { Install-MarkitaiExtra -ExtraName "copilot" }
    }

    # Build package spec with all tracked extras
    if ($script:MarkitaiVersion) {
        $pkg = "markitai[$($script:MARKITAI_EXTRAS)]==$($script:MarkitaiVersion)"
    } else {
        $pkg = "markitai[$($script:MARKITAI_EXTRAS)]"
    }

    # Build Python command for --python argument
    $pythonArg = $script:PYTHON_CMD

    # Check if already installed
    $markitaiCmd = Get-Command markitai -ErrorAction SilentlyContinue
    $isUpgrade = $false
    if ($markitaiCmd) {
        $oldVersion = & markitai --version 2>&1 | Select-Object -First 1
        if ($oldVersion) {
            $isUpgrade = $true
        }
    }

    if (-not $isUpgrade) {
        Clack-Info "$(i18n 'installing') $(i18n 'markitai')..."
    }

    # Always run uv tool install --upgrade to ensure latest version
    $uvExists = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvExists) {
        $oldErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $null = & uv tool install $pkg --python $pythonArg --upgrade 2>&1
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldErrorAction
        }
        if ($exitCode -eq 0) {
            # Refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")

            $markitaiCmd = Get-Command markitai -ErrorAction SilentlyContinue
            $version = if ($markitaiCmd) { (& markitai --version 2>&1 | Select-Object -First 1).Split(' ')[-1] } else { (i18n "installed") }
            if (-not $version) { $version = (i18n "installed") }
            Clack-Success "$(i18n 'markitai') $version"
            Track-Install -Component "markitai" -Status "installed"
            return $true
        }
    }

    Clack-Error "$(i18n 'markitai') $(i18n 'failed')"
    Track-Install -Component "markitai" -Status "failed"
    return $false
}

# Global variable tracking all needed extras (comma-separated)
$script:MARKITAI_EXTRAS = "browser"

# Track a markitai extra for deferred installation
# Extras are accumulated and installed once via Finalize-MarkitaiExtras
function Install-MarkitaiExtra {
    param([string]$ExtraName)

    # Check if already tracked
    $extras = $script:MARKITAI_EXTRAS -split ","
    if ($extras -contains $ExtraName) {
        return
    }
    $script:MARKITAI_EXTRAS = "$($script:MARKITAI_EXTRAS),$ExtraName"
}

# Finalize markitai extras after all optional components are resolved
# Reinstalls markitai with all accumulated extras if needed
function Finalize-MarkitaiExtras {
    # Read current receipt to check if extras changed
    $uvToolsDir = $null
    try { $uvToolsDir = & uv tool dir 2>$null } catch {}
    if (-not $uvToolsDir) {
        if ($env:WSL_DISTRO_NAME) {
            $uvToolsDir = "$HOME/.local/share/uv/tools"
        } else {
            $uvToolsDir = "$env:APPDATA\uv\tools"
        }
    }
    $receiptFile = Join-Path $uvToolsDir "markitai\uv-receipt.toml"
    $current = ""
    if (Test-Path $receiptFile) {
        $current = Get-Content $receiptFile -Raw -ErrorAction SilentlyContinue
    }

    # Check if new extras need to be added
    $needsUpdate = $false
    foreach ($extra in ($script:MARKITAI_EXTRAS -split ",")) {
        if ($current -notmatch [regex]::Escape($extra)) {
            $needsUpdate = $true
            break
        }
    }

    if (-not $needsUpdate) { return }

    # Reinstall with all extras
    if ($script:MarkitaiVersion) {
        $pkg = "markitai[$($script:MARKITAI_EXTRAS)]==$($script:MarkitaiVersion)"
    } else {
        $pkg = "markitai[$($script:MARKITAI_EXTRAS)]"
    }

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $null = & uv tool install $pkg --python $script:PYTHON_CMD --upgrade 2>&1
    } catch {}
    $ErrorActionPreference = $oldErrorAction
}

# Sync project dependencies (Dev mode)
function Sync-Dependencies {
    $projectRoot = Get-ProjectRoot
    Clack-Info "$(i18n 'info_project_dir'): $projectRoot"

    Push-Location $projectRoot

    try {
        Clack-Info (i18n "info_syncing_deps")
        $syncResult = & uv sync --all-extras --python $script:PYTHON_CMD 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success (i18n "info_deps_synced")
            Track-Install -Component "python_deps" -Status "installed"
            return $true
        } else {
            Clack-Error "$(i18n 'python_deps') $(i18n 'failed')"
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
                $precommitResult = & uv run pre-commit install 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Clack-Success (i18n "info_precommit_installed")
                    Track-Install -Component "precommit" -Status "installed"
                    return $true
                } else {
                    Clack-Warn "$(i18n 'precommit') $(i18n 'failed')"
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

    if (-not (Clack-Confirm (i18n "confirm_playwright") "y")) {
        Clack-Skip (i18n "playwright")
        Track-Install -Component "playwright" -Status "skipped"
        return $false
    }

    Clack-Info "$(i18n 'installing') $(i18n 'playwright')..."

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

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
            $null = & $markitaiPlaywright install chromium 2>&1
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
            $null = & $script:PYTHON_CMD -m playwright install chromium 2>&1
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

    if (-not (Clack-Confirm (i18n "confirm_libreoffice") "n")) {
        Clack-Skip (i18n "libreoffice")
        Track-Install -Component "libreoffice" -Status "skipped"
        return $false
    }

    Clack-Info "$(i18n 'installing') $(i18n 'libreoffice')..."

    # Priority: winget > scoop > choco
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        $null = & winget install TheDocumentFoundation.LibreOffice --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "$(i18n 'libreoffice') $(i18n 'installed')"
            Track-Install -Component "libreoffice" -Status "installed"
            return $true
        }
    }

    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        $null = & scoop bucket add extras 2>$null
        $null = & scoop install extras/libreoffice 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "$(i18n 'libreoffice') $(i18n 'installed')"
            Track-Install -Component "libreoffice" -Status "installed"
            return $true
        }
    }

    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        $null = & choco install libreoffice-fresh -y 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "$(i18n 'libreoffice') $(i18n 'installed')"
            Track-Install -Component "libreoffice" -Status "installed"
            return $true
        }
    }

    Clack-Error "$(i18n 'libreoffice') $(i18n 'failed')"
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

    if (-not (Clack-Confirm (i18n "confirm_ffmpeg") "n")) {
        Clack-Skip (i18n "ffmpeg")
        Track-Install -Component "ffmpeg" -Status "skipped"
        return $false
    }

    Clack-Info "$(i18n 'installing') $(i18n 'ffmpeg')..."

    # Priority: winget > scoop > choco
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        $null = & winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "$(i18n 'ffmpeg') $(i18n 'installed')"
            Track-Install -Component "ffmpeg" -Status "installed"
            return $true
        }
    }

    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        $null = & scoop install ffmpeg 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "$(i18n 'ffmpeg') $(i18n 'installed')"
            Track-Install -Component "ffmpeg" -Status "installed"
            return $true
        }
    }

    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        $null = & choco install ffmpeg -y 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "$(i18n 'ffmpeg') $(i18n 'installed')"
            Track-Install -Component "ffmpeg" -Status "installed"
            return $true
        }
    }

    Clack-Error "$(i18n 'ffmpeg') $(i18n 'failed')"
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

    if (-not (Clack-Confirm (i18n "confirm_claude_cli") "n")) {
        Clack-Skip (i18n "claude_cli")
        Track-Install -Component "claude_cli" -Status "skipped"
        return $false
    }

    Clack-Info "$(i18n 'installing') $(i18n 'claude_cli')..."

    # Prefer official install script (PowerShell)
    $claudeUrl = "https://claude.ai/install.ps1"
    try {
        $null = Invoke-Expression (Invoke-RestMethod -Uri $claudeUrl) 2>&1
        $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
        if ($claudeCmd) {
            Clack-Success "$(i18n 'claude_cli') $(i18n 'installed')"
            # Also install the SDK extra
            Install-MarkitaiExtra -ExtraName "claude-agent" | Out-Null
            Track-Install -Component "claude_cli" -Status "installed"
            return $true
        }
    } catch {}

    # Fallback: npm/pnpm
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        $null = & pnpm add -g @anthropic-ai/claude-code 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "$(i18n 'claude_cli') $(i18n 'installed')"
            Install-MarkitaiExtra -ExtraName "claude-agent" | Out-Null
            Track-Install -Component "claude_cli" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        $null = & npm install -g @anthropic-ai/claude-code 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "$(i18n 'claude_cli') $(i18n 'installed')"
            Install-MarkitaiExtra -ExtraName "claude-agent" | Out-Null
            Track-Install -Component "claude_cli" -Status "installed"
            return $true
        }
    }

    Clack-Error "$(i18n 'claude_cli') $(i18n 'failed')"
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

    if (-not (Clack-Confirm (i18n "confirm_copilot_cli") "n")) {
        Clack-Skip (i18n "copilot_cli")
        Track-Install -Component "copilot_cli" -Status "skipped"
        return $false
    }

    Clack-Info "$(i18n 'installing') $(i18n 'copilot_cli')..."

    # Prefer WinGet on Windows
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        $null = & winget install GitHub.Copilot --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
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
        $null = & pnpm add -g @github/copilot 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "$(i18n 'copilot_cli') $(i18n 'installed')"
            Install-MarkitaiExtra -ExtraName "copilot" | Out-Null
            Track-Install -Component "copilot_cli" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        $null = & npm install -g @github/copilot 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "$(i18n 'copilot_cli') $(i18n 'installed')"
            Install-MarkitaiExtra -ExtraName "copilot" | Out-Null
            Track-Install -Component "copilot_cli" -Status "installed"
            return $true
        }
    }

    Clack-Error "$(i18n 'copilot_cli') $(i18n 'failed')"
    Track-Install -Component "copilot_cli" -Status "failed"
    return $false
}

# Print installation summary
function Print-Summary {
    Clack-Section (i18n "section_summary")

    # Print installed components
    if ($script:INSTALLED_COMPONENTS.Count -gt 0) {
        Write-Host $S_BAR -ForegroundColor DarkGray
        Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
        Write-Host "  " -NoNewline
        Write-Host $S_BOX_TOP -ForegroundColor DarkGray -NoNewline
        Write-Host $S_BAR_H -ForegroundColor DarkGray -NoNewline
        Write-Host " $(i18n 'summary_installed')"
        foreach ($comp in $script:INSTALLED_COMPONENTS) {
            Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
            Write-Host "  " -NoNewline
            Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
            Write-Host "  " -NoNewline
            Write-Host $S_CHECK -ForegroundColor Green -NoNewline
            Write-Host " $(i18n $comp)"
        }
        Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
        Write-Host "  " -NoNewline
        Write-Host $S_BOX_BOT -ForegroundColor DarkGray -NoNewline
        Write-Host $S_BAR_H -ForegroundColor DarkGray
    }

    # Print skipped components
    if ($script:SKIPPED_COMPONENTS.Count -gt 0) {
        Write-Host $S_BAR -ForegroundColor DarkGray
        Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
        Write-Host "  " -NoNewline
        Write-Host $S_BOX_TOP -ForegroundColor DarkGray -NoNewline
        Write-Host $S_BAR_H -ForegroundColor DarkGray -NoNewline
        Write-Host " $(i18n 'summary_skipped')"
        foreach ($comp in $script:SKIPPED_COMPONENTS) {
            Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
            Write-Host "  " -NoNewline
            Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
            Write-Host "  " -NoNewline
            Write-Host $S_CIRCLE -ForegroundColor Yellow -NoNewline
            Write-Host " $(i18n $comp)"
        }
        Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
        Write-Host "  " -NoNewline
        Write-Host $S_BOX_BOT -ForegroundColor DarkGray -NoNewline
        Write-Host $S_BAR_H -ForegroundColor DarkGray
    }

    # Print failed components
    if ($script:FAILED_COMPONENTS.Count -gt 0) {
        Write-Host $S_BAR -ForegroundColor DarkGray
        Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
        Write-Host "  " -NoNewline
        Write-Host $S_BOX_TOP -ForegroundColor DarkGray -NoNewline
        Write-Host $S_BAR_H -ForegroundColor DarkGray -NoNewline
        Write-Host " $(i18n 'summary_failed')"
        foreach ($comp in $script:FAILED_COMPONENTS) {
            Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
            Write-Host "  " -NoNewline
            Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
            Write-Host "  " -NoNewline
            Write-Host $S_CROSS -ForegroundColor Red -NoNewline
            Write-Host " $(i18n $comp)"
        }
        Write-Host $S_BAR -ForegroundColor DarkGray -NoNewline
        Write-Host "  " -NoNewline
        Write-Host $S_BOX_BOT -ForegroundColor DarkGray -NoNewline
        Write-Host $S_BAR_H -ForegroundColor DarkGray
    }

    # Empty line before docs link
    Clack-Log ""
    Clack-Info "$(i18n 'info_docs'): https://markitai.ynewtime.com"
    Clack-Info "$(i18n 'info_issues'): https://github.com/Ynewtime/markitai/issues"
}

# Print user mode completion message
function Print-UserCompletion {
    Clack-Note (i18n "getting_started") `
        "$(i18n 'interactive_mode'):" `
        "  markitai -I" `
        "" `
        "$(i18n 'convert_file'):" `
        "  markitai file.pdf" `
        "" `
        "$(i18n 'show_help'):" `
        "  markitai --help"
}

# Print dev mode completion message
function Print-DevCompletion {
    $projectRoot = Get-ProjectRoot
    Clack-Note (i18n "quick_start") `
        "$(i18n 'activate_venv'):" `
        "  $projectRoot\.venv\Scripts\Activate.ps1" `
        "" `
        "$(i18n 'run_tests'):" `
        "  uv run pytest" `
        "" `
        "$(i18n 'run_cli'):" `
        "  uv run markitai --help"
}

# Initialize markitai config (silent)
function Initialize-Config {
    $markitaiExists = Get-Command markitai -ErrorAction SilentlyContinue
    if ($markitaiExists) {
        try {
            $null = & markitai config init --yes 2>$null
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

    Clack-Section (i18n "section_core")
    if (-not (Install-UV)) { Print-Summary; Clack-Cancel (i18n "error_setup_failed"); exit 1 }
    if (-not (Install-Python)) { Print-Summary; Clack-Cancel (i18n "error_setup_failed"); exit 1 }
    if (-not (Install-Markitai)) { Print-Summary; Clack-Cancel (i18n "error_setup_failed"); exit 1 }

    Clack-Section (i18n "section_optional")
    Install-OptionalPlaywright | Out-Null
    Install-OptionalLibreOffice | Out-Null
    Install-OptionalFFmpeg | Out-Null

    Clack-Section (i18n "section_llm_cli")
    Install-OptionalClaudeCLI | Out-Null
    Install-OptionalCopilotCLI | Out-Null

    Finalize-MarkitaiExtras

    Initialize-Config 2>$null | Out-Null

    Print-Summary
    Print-UserCompletion
    Clack-Outro (i18n "setup_complete")
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
    Install-PreCommit | Out-Null

    Clack-Section (i18n "section_optional")
    Install-OptionalPlaywright | Out-Null
    Install-OptionalLibreOffice | Out-Null
    Install-OptionalFFmpeg | Out-Null

    Clack-Section (i18n "section_llm_cli")
    Install-OptionalClaudeCLI | Out-Null
    Install-OptionalCopilotCLI | Out-Null

    Print-Summary
    Print-DevCompletion
    Clack-Outro (i18n "dev_setup_complete")
}

# Main entry point
function Main {
    if (Test-DevMode) {
        Run-DevSetup
    } else {
        Run-UserSetup
    }
}

# Run main function
Main
