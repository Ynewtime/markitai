# Markitai Setup Script (User Edition) - Chinese
# PowerShell 5.1+
# Encoding: UTF-8 (no BOM)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ============================================================
# Library Loading / 库加载（支持本地和远程执行）
# ============================================================

$LIB_BASE_URL = "https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts"

# 在脚本级别获取脚本目录（不在函数内，避免作用域问题）
$script:ScriptDir = $PSScriptRoot
if (-not $script:ScriptDir -and $MyInvocation.MyCommand.Path) {
    $script:ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}

# 检测是否为本地执行（脚本路径存在）
if ($script:ScriptDir -and (Test-Path "$script:ScriptDir\lib.ps1" -ErrorAction SilentlyContinue)) {
    . "$script:ScriptDir\lib.ps1"
} else {
    # 远程执行 (irm | iex) - 下载 lib.ps1
    try {
        $tempLib = [System.IO.Path]::GetTempFileName()
        $tempLib = [System.IO.Path]::ChangeExtension($tempLib, ".ps1")

        Invoke-RestMethod "$LIB_BASE_URL/lib.ps1" -OutFile $tempLib
        . $tempLib
        Remove-Item $tempLib -ErrorAction SilentlyContinue
    } catch {
        Write-Host "错误: 下载 lib.ps1 失败: $_" -ForegroundColor Red
        exit 1
    }
}

# ============================================================
# 可选组件检测辅助函数
# ============================================================

function Test-PlaywrightBrowserZh {
    # 检查是否已安装 Chromium 浏览器
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCmd) {
        try {
            $uvToolDir = & uv tool dir 2>$null
            if ($uvToolDir) {
                $markitaiPlaywright = Join-Path $uvToolDir "markitai\Scripts\playwright.exe"
                if (Test-Path $markitaiPlaywright) {
                    # 通过检查浏览器目录判断是否安装
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

function Test-LibreOfficeZh {
    $soffice = Get-Command soffice -ErrorAction SilentlyContinue
    if ($soffice) { return $true }

    $commonPaths = @(
        "${env:ProgramFiles}\LibreOffice\program\soffice.exe",
        "${env:ProgramFiles(x86)}\LibreOffice\program\soffice.exe"
    )
    foreach ($path in $commonPaths) {
        if (Test-Path $path) { return $true }
    }
    return $false
}

function Test-FFmpegZh {
    $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    return ($null -ne $ffmpegCmd)
}

function Test-ClaudeCLIZh {
    $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
    return ($null -ne $claudeCmd)
}

function Test-CopilotCLIZh {
    $copilotCmd = Get-Command copilot -ErrorAction SilentlyContinue
    return ($null -ne $copilotCmd)
}

# ============================================================
# 中文 Clack 风格函数
# ============================================================

# 中文确认提示 (基于 Clack-Confirm)
# Usage: Clack-ConfirmZh "问题?" "y|n"
# Returns: $true for yes, $false for no
function Clack-ConfirmZh {
    param(
        [string]$Prompt,
        [string]$Default = "n"
    )

    if ($Default -eq "y") {
        $hint = "Y/n"
    } else {
        $hint = "y/N"
    }

    Write-Host ([char]0x2502) -ForegroundColor DarkGray
    Write-Host ([char]0x25C7) -ForegroundColor Cyan -NoNewline
    $answer = Read-Host "  $Prompt [$hint]"

    if ([string]::IsNullOrWhiteSpace($answer)) {
        $answer = $Default
    }

    return $answer -match "^[Yy]"
}

# ============================================================
# 安全检查函数 (Clack 风格)
# ============================================================

# 执行策略检查（中文）
function Test-ExecutionPolicyZh {
    $policy = Get-ExecutionPolicy -Scope CurrentUser
    if ($policy -eq "Restricted" -or $policy -eq "AllSigned") {
        Clack-Warn "执行策略警告"
        Clack-Note "当前策略: $policy" `
            "脚本可能无法运行。" `
            "" `
            "解决方法:" `
            "Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned"
        return $false
    }
    return $true
}

# 管理员警告检查（中文）
function Test-AdminWarningZh {
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if ($isAdmin) {
        Clack-Warn "正在以管理员身份运行"
        Clack-Note "安全提示" `
            "以管理员身份运行安装脚本存在以下风险:" `
            "1. 系统级更改可能影响所有用户" `
            "2. 远程代码执行风险被放大" `
            "" `
            "建议: 使用普通用户身份运行此脚本"

        if (-not (Clack-ConfirmZh "是否继续以管理员身份运行?" "n")) {
            Clack-Cancel "已取消。请使用普通用户身份运行。"
            exit 1
        }
    }
}

# WSL 环境警告检查（中文）
function Test-WSLWarningZh {
    # 检测是否在 WSL (Windows Subsystem for Linux) 中运行
    if ($env:WSL_DISTRO_NAME) {
        Clack-Warn "正在 WSL 环境中运行"
        Clack-Note "环境提示" `
            "检测到您正在 WSL 中运行 PowerShell。" `
            "建议使用原生 shell 脚本以获得最佳效果:" `
            "" `
            "./scripts/setup-zh.sh"

        if (-not (Clack-ConfirmZh "是否继续使用 PowerShell 脚本?" "n")) {
            Clack-Cancel "已取消。请使用 .sh 脚本。"
            exit 1
        }
    }
}

# ============================================================
# 安装函数 (Clack 风格，自动检测)
# ============================================================

# 安装 UV 包管理器
function Install-UVZh {
    # 自动检测
    if (Test-UV) {
        $version = (& uv --version 2>$null).Split(' ')[1]
        Clack-Success "uv $version"
        Track-Install -Component "uv" -Status "installed"
        return $true
    }

    Clack-Info "正在安装 uv..."

    # 构建安装 URL（支持版本指定）
    if ($script:UvVersion) {
        $uvUrl = "https://astral.sh/uv/$($script:UvVersion)/install.ps1"
    } else {
        $uvUrl = "https://astral.sh/uv/install.ps1"
    }

    try {
        $null = Invoke-RestMethod $uvUrl | Invoke-Expression 2>$null

        # 刷新 PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")

        # 检查 uv 命令是否可用
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uvCmd) {
            Clack-Warn "uv 已安装，但需要重新打开 PowerShell"
            Track-Install -Component "uv" -Status "installed"
            return $false
        }

        $version = (& uv --version 2>$null).Split(' ')[1]
        Clack-Success "uv $version 安装成功"
        Track-Install -Component "uv" -Status "installed"
        return $true
    } catch {
        Clack-Error "uv 安装失败: $_"
        Clack-Info "手动安装: irm https://astral.sh/uv/install.ps1 | iex"
        Track-Install -Component "uv" -Status "failed"
        return $false
    }
}

# 检测/安装 Python（通过 uv 管理）
function Install-PythonZh {
    # 优先使用 uv 管理的 Python 3.13
    if (Test-CommandExists "uv") {
        $uvPython = & uv python find 3.13 2>$null
        if ($uvPython -and (Test-Path $uvPython)) {
            $version = & $uvPython -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>$null
            if ($version) {
                $script:PYTHON_CMD = $uvPython
                Clack-Success "Python $version"
                return $true
            }
        }

        # 未找到，自动安装
        Clack-Info "正在安装 Python 3.13..."
        $installResult = & uv python install 3.13 2>&1
        if ($LASTEXITCODE -eq 0) {
            $uvPython = & uv python find 3.13 2>$null
            if ($uvPython -and (Test-Path $uvPython)) {
                $version = & $uvPython -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>$null
                if ($version) {
                    $script:PYTHON_CMD = $uvPython
                    Clack-Success "Python $version 安装成功"
                    return $true
                }
            }
        }
        Clack-Error "Python 3.13 安装失败"
    } else {
        Clack-Error "uv 未安装，无法管理 Python"
    }

    return $false
}

# 安装 markitai
function Install-MarkitaiZh {
    # 检查是否已安装
    $markitaiCmd = Get-Command markitai -ErrorAction SilentlyContinue
    if ($markitaiCmd) {
        $version = & markitai --version 2>&1 | Select-Object -First 1
        if ($version) {
            Clack-Success "markitai $version"
            Track-Install -Component "markitai" -Status "installed"
            return $true
        }
    }

    Clack-Info "正在安装 markitai..."

    # 构建包规格（支持版本指定）
    # 注意: 使用 [browser] 而非 [all] 避免安装不必要的 SDK 包
    if ($script:MarkitaiVersion) {
        $pkg = "markitai[browser]==$($script:MarkitaiVersion)"
    } else {
        $pkg = "markitai[browser]"
    }

    $pythonArg = $script:PYTHON_CMD
    if ($pythonArg -match "^py\s+-(\d+\.\d+)$") {
        $pythonArg = $Matches[1]
    }

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
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")

            $markitaiCmd = Get-Command markitai -ErrorAction SilentlyContinue
            $version = if ($markitaiCmd) { & markitai --version 2>&1 | Select-Object -First 1 } else { "已安装" }
            if (-not $version) { $version = "已安装" }
            Clack-Success "markitai $version"
            Track-Install -Component "markitai" -Status "installed"
            return $true
        }
    }

    # 回退到 pipx
    $pipxExists = Get-Command pipx -ErrorAction SilentlyContinue
    if ($pipxExists) {
        $oldErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $null = & pipx install $pkg --python $pythonArg --force 2>&1
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldErrorAction
        }
        if ($exitCode -eq 0) {
            $markitaiCmd = Get-Command markitai -ErrorAction SilentlyContinue
            $version = if ($markitaiCmd) { & markitai --version 2>&1 | Select-Object -First 1 } else { "已安装" }
            if (-not $version) { $version = "已安装" }
            Clack-Success "markitai $version"
            Track-Install -Component "markitai" -Status "installed"
            return $true
        }
    }

    # 回退到 pip --user
    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $cmdParts = $script:PYTHON_CMD -split " "
        $exe = $cmdParts[0]
        $baseArgs = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] } else { @() }
        $pipArgs = $baseArgs + @("-m", "pip", "install", "--user", "--upgrade", $pkg)
        $null = & $exe @pipArgs 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorAction
    }
    if ($exitCode -eq 0) {
        $markitaiCmd = Get-Command markitai -ErrorAction SilentlyContinue
        $version = if ($markitaiCmd) { & markitai --version 2>&1 | Select-Object -First 1 } else { "已安装" }
        if (-not $version) { $version = "已安装" }
        Clack-Success "markitai $version"
        Clack-Warn "可能需要将 Python Scripts 目录添加到 PATH"
        Track-Install -Component "markitai" -Status "installed"
        return $true
    }

    Clack-Error "markitai 安装失败"
    Clack-Info "请手动安装: uv tool install markitai --python $pythonArg"
    Track-Install -Component "markitai" -Status "failed"
    return $false
}

# 安装 Playwright 浏览器 (Chromium)
function Install-PlaywrightBrowserZh {
    Clack-Info "正在下载 Chromium 浏览器..."

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    # 方法 1: 使用 markitai 的 uv tool 环境中的 playwright
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
            & $markitaiPlaywright install chromium
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Clack-Success "Chromium 浏览器"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        } catch {}
    }

    # 方法 2: 回退到 Python 模块
    if ($script:PYTHON_CMD) {
        $cmdParts = $script:PYTHON_CMD -split " "
        $exe = $cmdParts[0]
        $baseArgs = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] } else { @() }
        $pwArgs = $baseArgs + @("-m", "playwright", "install", "chromium")

        try {
            & $exe @pwArgs
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Clack-Success "Chromium 浏览器"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        } catch {}
    }

    $ErrorActionPreference = $oldErrorAction
    Clack-Error "Chromium 浏览器安装失败"
    Clack-Info "稍后可手动安装: playwright install chromium"
    Track-Install -Component "Playwright Browser" -Status "failed"
    return $false
}

# 安装 LibreOffice
function Install-LibreOfficeZh {
    Clack-Info "正在安装 LibreOffice..."

    # 优先级: winget > scoop > choco
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        & winget install TheDocumentFoundation.LibreOffice --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "LibreOffice"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        & scoop bucket add extras 2>$null
        & scoop install extras/libreoffice
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "LibreOffice"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        & choco install libreoffice-fresh -y
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "LibreOffice"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    Clack-Error "LibreOffice 安装失败"
    Clack-Note "手动安装" `
        "winget: winget install TheDocumentFoundation.LibreOffice" `
        "scoop: scoop install extras/libreoffice" `
        "choco: choco install libreoffice-fresh" `
        "下载: https://www.libreoffice.org/download/"
    Track-Install -Component "LibreOffice" -Status "failed"
    return $false
}

# 安装 FFmpeg
function Install-FFmpegZh {
    Clack-Info "正在安装 FFmpeg..."

    # 优先级: winget > scoop > choco
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        & winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "FFmpeg"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        & scoop install ffmpeg
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "FFmpeg"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        & choco install ffmpeg -y
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "FFmpeg"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    Clack-Error "FFmpeg 安装失败"
    Clack-Note "手动安装" `
        "winget: winget install Gyan.FFmpeg" `
        "scoop: scoop install ffmpeg" `
        "choco: choco install ffmpeg" `
        "下载: https://ffmpeg.org/download.html"
    Track-Install -Component "FFmpeg" -Status "failed"
    return $false
}

# 安装 Claude Code CLI
function Install-ClaudeCLIZh {
    Clack-Info "正在安装 Claude Code CLI..."

    # 尝试官方安装脚本
    $claudeUrl = "https://claude.ai/install.ps1"
    try {
        $null = Invoke-Expression (Invoke-RestMethod -Uri $claudeUrl) 2>&1
        $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
        if ($claudeCmd) {
            Clack-Success "Claude Code CLI"
            Clack-Info "请运行 'claude /login' 进行认证"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    } catch {}

    # 回退: npm/pnpm
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        & pnpm add -g @anthropic-ai/claude-code
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "Claude Code CLI"
            Clack-Info "请运行 'claude /login' 进行认证"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        & npm install -g @anthropic-ai/claude-code
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "Claude Code CLI"
            Clack-Info "请运行 'claude /login' 进行认证"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    }

    Clack-Error "Claude Code CLI 安装失败"
    Clack-Note "手动安装" `
        "pnpm: pnpm add -g @anthropic-ai/claude-code" `
        "npm: npm install -g @anthropic-ai/claude-code" `
        "文档: https://code.claude.com/docs/en/setup"
    Track-Install -Component "Claude Code CLI" -Status "failed"
    return $false
}

# 安装 GitHub Copilot CLI
function Install-CopilotCLIZh {
    Clack-Info "正在安装 Copilot CLI..."

    # 优先使用 WinGet
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        & winget install GitHub.Copilot --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "Copilot CLI"
            Clack-Info "请运行 'copilot /login' 进行认证"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    # 回退: npm/pnpm
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        & pnpm add -g @github/copilot
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "Copilot CLI"
            Clack-Info "请运行 'copilot /login' 进行认证"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        & npm install -g @github/copilot
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "Copilot CLI"
            Clack-Info "请运行 'copilot /login' 进行认证"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    Clack-Error "Copilot CLI 安装失败"
    Clack-Note "手动安装" `
        "winget: winget install GitHub.Copilot" `
        "pnpm: pnpm add -g @github/copilot" `
        "npm: npm install -g @github/copilot"
    Track-Install -Component "Copilot CLI" -Status "failed"
    return $false
}

# 初始化配置
function Initialize-ConfigZh {
    $markitaiExists = Get-Command markitai -ErrorAction SilentlyContinue
    if (-not $markitaiExists) {
        return
    }

    $configPath = Join-Path $env:USERPROFILE ".markitai\config.json"

    # 如果配置已存在，询问是否覆盖
    if (Test-Path $configPath) {
        if (Clack-ConfirmZh "$configPath 已存在，是否覆盖？" "n") {
            try {
                & markitai config init --yes 2>$null
                Clack-Success "配置初始化完成"
            } catch {}
        } else {
            Clack-Skip "保留现有配置"
        }
    } else {
        try {
            & markitai config init 2>$null
            Clack-Success "配置初始化完成"
        } catch {}
    }
}

# ============================================================
# 主逻辑
# ============================================================

function Main {
    # 检查执行策略
    Test-ExecutionPolicyZh | Out-Null

    # 安全检查: 管理员警告
    Test-AdminWarningZh

    # 环境检查: WSL 警告
    Test-WSLWarningZh

    # === 开始 Clack 风格界面 ===
    Clack-Intro "Markitai 环境配置向导"

    # 核心组件部分
    Clack-Section "安装核心组件"

    # 步骤 1: 检测/安装 UV
    if (-not (Install-UVZh)) {
        Clack-Cancel "安装失败"
        exit 1
    }

    # 步骤 2: 检测/安装 Python
    if (-not (Install-PythonZh)) {
        Clack-Cancel "安装失败"
        exit 1
    }

    # 步骤 3: 安装 markitai
    if (-not (Install-MarkitaiZh)) {
        Clack-Cancel "安装失败"
        exit 1
    }

    # 可选组件部分
    Clack-Section "可选组件"

    # Playwright 浏览器 - 自动检测
    if (Test-PlaywrightBrowserZh) {
        Clack-Success "Chromium 浏览器 (已安装)"
        Track-Install -Component "Playwright Browser" -Status "installed"
    } elseif (Clack-ConfirmZh "是否下载 Chromium 浏览器？(用于 JS 渲染页面)" "y") {
        Install-PlaywrightBrowserZh | Out-Null
    } else {
        Clack-Skip "Chromium 浏览器"
        Track-Install -Component "Playwright Browser" -Status "skipped"
    }

    # LibreOffice - 自动检测
    if (Test-LibreOfficeZh) {
        Clack-Success "LibreOffice (已安装)"
        Track-Install -Component "LibreOffice" -Status "installed"
    } elseif (Clack-ConfirmZh "是否安装 LibreOffice？(用于转换 .doc/.ppt/.xls)" "n") {
        Install-LibreOfficeZh | Out-Null
    } else {
        Clack-Skip "LibreOffice"
        Track-Install -Component "LibreOffice" -Status "skipped"
    }

    # FFmpeg - 自动检测
    if (Test-FFmpegZh) {
        Clack-Success "FFmpeg (已安装)"
        Track-Install -Component "FFmpeg" -Status "installed"
    } elseif (Clack-ConfirmZh "是否安装 FFmpeg？(用于处理音视频文件)" "n") {
        Install-FFmpegZh | Out-Null
    } else {
        Clack-Skip "FFmpeg"
        Track-Install -Component "FFmpeg" -Status "skipped"
    }

    # Claude CLI - 自动检测
    if (Test-ClaudeCLIZh) {
        $version = & claude --version 2>&1 | Select-Object -First 1
        Clack-Success "Claude Code CLI $version (已安装)"
        Track-Install -Component "Claude Code CLI" -Status "installed"
    } elseif (Clack-ConfirmZh "是否安装 Claude Code CLI？" "n") {
        if (Install-ClaudeCLIZh) {
            # 安装 Claude Agent SDK
            Install-MarkitaiExtra -ExtraName "claude-agent" | Out-Null
        }
    } else {
        Clack-Skip "Claude Code CLI"
        Track-Install -Component "Claude Code CLI" -Status "skipped"
    }

    # Copilot CLI - 自动检测
    if (Test-CopilotCLIZh) {
        $version = & copilot --version 2>&1 | Select-Object -First 1
        Clack-Success "Copilot CLI $version (已安装)"
        Track-Install -Component "Copilot CLI" -Status "installed"
    } elseif (Clack-ConfirmZh "是否安装 GitHub Copilot CLI？" "n") {
        if (Install-CopilotCLIZh) {
            # 安装 Copilot SDK
            Install-MarkitaiExtra -ExtraName "copilot" | Out-Null
        }
    } else {
        Clack-Skip "Copilot CLI"
        Track-Install -Component "Copilot CLI" -Status "skipped"
    }

    # 初始化配置
    Initialize-ConfigZh

    # 使用提示
    Clack-Note "开始使用" `
        "markitai -I          交互模式" `
        "markitai file.pdf   转换文件" `
        "markitai --help     查看帮助" `
        "" `
        "文档: https://markitai.ynewtime.com"

    # 完成
    Clack-Outro "配置完成!"
}

# 运行主函数
Main
