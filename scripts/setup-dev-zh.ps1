# Markitai Setup Script (Developer Edition) - Chinese
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
    # 远程执行 - 开发者版需要本地克隆
    Write-Host ""
    Write-Host ([char]0x250C) -ForegroundColor DarkGray -NoNewline
    Write-Host "  开发者版需要本地仓库" -ForegroundColor Red
    Write-Host ([char]0x2502) -ForegroundColor DarkGray
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "  请先克隆仓库:"
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "    git clone https://github.com/Ynewtime/markitai.git" -ForegroundColor Yellow
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "    cd markitai" -ForegroundColor Yellow
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "    .\scripts\setup-dev-zh.ps1" -ForegroundColor Yellow
    Write-Host ([char]0x2502) -ForegroundColor DarkGray
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "  或使用用户版快速安装:"
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "    irm https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup-zh.ps1 | iex" -ForegroundColor Yellow
    Write-Host ([char]0x2502) -ForegroundColor DarkGray
    Write-Host ([char]0x2514) -ForegroundColor DarkGray -NoNewline
    Write-Host "  退出" -ForegroundColor Red
    Write-Host ""
    exit 1
}

# ============================================================
# 开发者版专用函数
# ============================================================

function Get-ProjectRoot {
    return Split-Path -Parent $ScriptDir
}

# 执行策略检查（中文）
function Test-ExecutionPolicyZh {
    $policy = Get-ExecutionPolicy -Scope CurrentUser
    if ($policy -eq "Restricted" -or $policy -eq "AllSigned") {
        Clack-Warn "执行策略警告: 当前策略为 $policy"
        Clack-Log "  脚本可能无法运行。"
        Clack-Log ""
        Clack-Log "  解决方法:"
        Clack-Log "    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned"
        return $false
    }
    return $true
}

function Test-AdminWarningZh {
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if ($isAdmin) {
        Clack-Warn "正在以管理员身份运行"
        Clack-Log "  以管理员身份运行安装脚本存在以下风险:"
        Clack-Log "  1. 系统级更改可能影响所有用户"
        Clack-Log "  2. 远程代码执行风险被放大"
        Clack-Log ""
        Clack-Log "  建议: 使用普通用户身份运行此脚本"

        if (-not (Clack-Confirm "是否继续以管理员身份运行?" "n")) {
            Clack-Cancel "退出。请使用普通用户身份运行。"
            exit 1
        }
    }
}

function Test-WSLWarningZh {
    # 检测是否在 WSL (Windows Subsystem for Linux) 中运行
    # Note: Only check WSL_DISTRO_NAME, not WSLENV
    # WSLENV is used to configure env var sharing between Windows and WSL,
    # and may exist on Windows host even when not running inside WSL
    if ($env:WSL_DISTRO_NAME) {
        Clack-Warn "正在 WSL 环境中运行"
        Clack-Log "  检测到您正在 WSL 中运行 PowerShell。"
        Clack-Log "  建议使用原生 shell 脚本以获得最佳效果:"
        Clack-Log ""
        Clack-Log "    ./scripts/setup-dev-zh.sh"

        if (-not (Clack-Confirm "是否继续使用 PowerShell 脚本?" "n")) {
            Clack-Cancel "退出。请使用 .sh 脚本。"
            exit 1
        }
    }
}

function Confirm-RemoteScriptZh {
    param(
        [string]$ScriptUrl,
        [string]$ScriptName
    )

    Clack-Warn "即将执行远程脚本"
    Clack-Log "  来源: $ScriptUrl"
    Clack-Log "  用途: 安装 $ScriptName"
    Clack-Log ""
    Clack-Log "  此操作将从互联网下载并执行代码。"
    Clack-Log "  请确保您信任该来源。"

    return (Clack-Confirm "确认执行?" "n")
}

# 检测/安装 Python（通过 uv 管理）
function Test-PythonZh {
    # 优先使用 uv 管理的 Python 3.13
    if (Test-CommandExists "uv") {
        $uvPython = & uv python find 3.13 2>$null
        if ($uvPython -and (Test-Path $uvPython)) {
            $version = & $uvPython -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>$null
            if ($version) {
                $script:PYTHON_CMD = $uvPython
                Clack-Success "Python $version (uv 管理)"
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
                    Clack-Success "Python $version 安装成功 (uv 管理)"
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

# 安装 UV（开发者版必需）
function Install-UVZh {
    if (Test-UV) {
        $version = (& uv --version 2>$null).Split(' ')[1]
        Clack-Success "uv $version"
        Track-Install -Component "uv" -Status "installed"
        return $true
    }

    Clack-Error "UV 未安装"

    if (-not (Clack-Confirm "是否自动安装 UV?" "n")) {
        Clack-Error "UV 是开发所必需的"
        Track-Install -Component "uv" -Status "failed"
        return $false
    }

    if ($script:UvVersion) {
        $uvUrl = "https://astral.sh/uv/$($script:UvVersion)/install.ps1"
        Clack-Info "安装 UV 版本: $($script:UvVersion)"
    } else {
        $uvUrl = "https://astral.sh/uv/install.ps1"
    }

    if (-not (Confirm-RemoteScriptZh -ScriptUrl $uvUrl -ScriptName "UV")) {
        Clack-Error "UV 是开发所必需的"
        Track-Install -Component "uv" -Status "failed"
        return $false
    }

    Clack-Info "正在安装 UV..."

    try {
        Invoke-RestMethod $uvUrl | Invoke-Expression

        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")

        # Check if uv command exists after PATH refresh
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uvCmd) {
            Clack-Warn "UV 已安装，但需要重新打开 PowerShell"
            Clack-Info "请重新打开 PowerShell 后再次运行此脚本"
            Track-Install -Component "uv" -Status "installed"
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
            Clack-Success "$version 安装成功"
            Track-Install -Component "uv" -Status "installed"
            return $true
        } else {
            Clack-Warn "UV 已安装，但需要重新打开 PowerShell"
            Clack-Info "请重新打开 PowerShell 后再次运行此脚本"
            Track-Install -Component "uv" -Status "installed"
            return $false
        }
    } catch {
        Clack-Error "UV 安装失败: $_"
        Clack-Info "手动安装: irm https://astral.sh/uv/install.ps1 | iex"
        Track-Install -Component "uv" -Status "failed"
        return $false
    }
}

function Sync-DependenciesZh {
    $projectRoot = Get-ProjectRoot
    Clack-Info "项目目录: $projectRoot"

    # 获取 Python 版本号用于 uv --python 参数
    $pythonArg = $script:PYTHON_CMD
    if ($pythonArg -match "^py\s+-(\d+\.\d+)$") {
        # 格式: py -3.13 -> 3.13
        $pythonArg = $Matches[1]
    } else {
        # 从 Python 命令获取实际版本号
        $cmdParts = $pythonArg -split " "
        $exe = $cmdParts[0]
        $baseArgs = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] } else { @() }
        $versionOutput = Invoke-PythonWithTimeout -Exe $exe -Arguments ($baseArgs + @("-c", "import sys; print('%d.%d' % (sys.version_info[0], sys.version_info[1]))")) -TimeoutSeconds 5
        if ($versionOutput) {
            $pythonArg = $versionOutput.Trim()
        }
    }

    Push-Location $projectRoot

    try {
        $syncResult = & uv sync --all-extras --python $pythonArg 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "依赖同步完成"
            return $true
        } else {
            Clack-Error "依赖同步失败"
            Clack-Log ($syncResult | Out-String)
            return $false
        }
    } finally {
        Pop-Location
    }
}

function Install-PreCommitZh {
    $projectRoot = Get-ProjectRoot
    Push-Location $projectRoot

    try {
        if (Test-Path ".pre-commit-config.yaml") {
            $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
            if ($uvCmd) {
                $precommitResult = & uv run pre-commit install 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Clack-Success "pre-commit hooks 安装完成"
                } else {
                    Clack-Warn "pre-commit 安装失败，请手动运行: uv run pre-commit install"
                }
            } else {
                Clack-Warn "未找到 uv 命令，跳过 pre-commit 安装"
            }
        } else {
            Clack-Skip "未找到 .pre-commit-config.yaml"
        }
    } finally {
        Pop-Location
    }
}

# 安装 Claude Code CLI
function Install-ClaudeCLIZh {
    # 检查是否已安装
    $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($claudeCmd) {
        $version = & claude --version 2>&1 | Select-Object -First 1
        Clack-Success "Claude Code CLI 已安装: $version"
        Track-Install -Component "Claude Code CLI" -Status "installed"
        return $true
    }

    Clack-Info "正在安装 Claude Code CLI..."

    # 尝试 npm/pnpm
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        Clack-Info "使用 pnpm 安装..."
        & pnpm add -g @anthropic-ai/claude-code
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "通过 pnpm 安装 Claude Code CLI 成功"
            Clack-Info "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        Clack-Info "使用 npm 安装..."
        & npm install -g @anthropic-ai/claude-code
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "通过 npm 安装 Claude Code CLI 成功"
            Clack-Info "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    }

    # 尝试 WinGet (Windows)
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Clack-Info "使用 WinGet 安装..."
        $null = & winget install Anthropic.ClaudeCode --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "通过 WinGet 安装 Claude Code CLI 成功"
            Clack-Info "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    }

    Clack-Warn "Claude Code CLI 安装失败"
    Clack-Note "手动安装选项" "pnpm: pnpm add -g @anthropic-ai/claude-code" "winget: winget install Anthropic.ClaudeCode" "文档: https://code.claude.com/docs/en/setup"
    Track-Install -Component "Claude Code CLI" -Status "failed"
    return $false
}

# 安装 GitHub Copilot CLI
function Install-CopilotCLIZh {
    # 检查是否已安装
    $copilotCmd = Get-Command copilot -ErrorAction SilentlyContinue
    if ($copilotCmd) {
        $version = & copilot --version 2>&1 | Select-Object -First 1
        Clack-Success "Copilot CLI 已安装: $version"
        Track-Install -Component "Copilot CLI" -Status "installed"
        return $true
    }

    Clack-Info "正在安装 GitHub Copilot CLI..."

    # 尝试 npm/pnpm
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        Clack-Info "使用 pnpm 安装..."
        $null = & pnpm add -g @github/copilot 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "通过 pnpm 安装 Copilot CLI 成功"
            Clack-Info "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        Clack-Info "使用 npm 安装..."
        $null = & npm install -g @github/copilot 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "通过 npm 安装 Copilot CLI 成功"
            Clack-Info "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    # 尝试 WinGet (Windows)
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Clack-Info "使用 WinGet 安装..."
        $null = & winget install GitHub.Copilot --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "通过 WinGet 安装 Copilot CLI 成功"
            Clack-Info "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    Clack-Warn "Copilot CLI 安装失败"
    Clack-Note "手动安装选项" "pnpm: pnpm add -g @github/copilot" "winget: winget install GitHub.Copilot"
    Track-Install -Component "Copilot CLI" -Status "failed"
    return $false
}

# 安装 Playwright 浏览器 (Chromium) - 开发环境
# 优先使用 uv run，回退到 python 模块
# 返回: $true 成功, $false 失败/跳过
function Install-PlaywrightBrowserDevZh {
    # 自动检测是否已安装
    $projectRoot = Get-ProjectRoot
    Push-Location $projectRoot

    try {
        # 检查 Chromium 是否已安装（通过实际调用 playwright 检测）
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if ($uvCmd) {
            $checkResult = & uv run python -c "from playwright.sync_api import sync_playwright; p = sync_playwright().start(); p.chromium.executable_path; p.stop()" 2>&1
            if ($LASTEXITCODE -eq 0) {
                Clack-Success "Playwright 浏览器 (Chromium)"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        }
    } catch {}
    finally {
        Pop-Location
    }

    # 下载前先征询用户同意
    if (-not (Clack-Confirm "是否下载 Chromium 浏览器？(用于 JS 渲染页面)" "y")) {
        Clack-Skip "Playwright 浏览器"
        Track-Install -Component "Playwright Browser" -Status "skipped"
        return $false
    }

    Clack-Info "正在下载 Chromium 浏览器..."

    Push-Location $projectRoot

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        # 优先使用 uv run（开发环境使用 .venv）
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if ($uvCmd) {
            # 显示下载进度（Chromium 约 200MB）
            & uv run playwright install chromium
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Clack-Success "Chromium 浏览器安装成功"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        }

        # 回退到 Python 模块
        if ($script:PYTHON_CMD) {
            $cmdParts = $script:PYTHON_CMD -split " "
            $exe = $cmdParts[0]
            $baseArgs = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] } else { @() }
            $pwArgs = $baseArgs + @("-m", "playwright", "install", "chromium")

            # 显示下载进度
            & $exe @pwArgs
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Clack-Success "Chromium 浏览器安装成功"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        }
    } finally {
        Pop-Location
        $ErrorActionPreference = $oldErrorAction
    }

    Clack-Warn "Playwright 浏览器安装失败"
    Clack-Info "稍后可手动安装: uv run playwright install chromium"
    Track-Install -Component "Playwright Browser" -Status "failed"
    return $false
}

# 检测 LibreOffice 安装（可选，用于旧版 Office 文件）
function Install-LibreOfficeDevZh {
    # 自动检测
    $soffice = Get-Command soffice -ErrorAction SilentlyContinue
    if ($soffice) {
        try {
            $version = & soffice --version 2>&1 | Select-Object -First 1
            Clack-Success "LibreOffice 已安装: $version"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        } catch {}
    }

    $commonPaths = @(
        "${env:ProgramFiles}\LibreOffice\program\soffice.exe",
        "${env:ProgramFiles(x86)}\LibreOffice\program\soffice.exe"
    )

    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            Clack-Success "LibreOffice 已安装"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    # 未安装，询问用户
    if (-not (Clack-Confirm "是否安装 LibreOffice？(用于转换 .doc/.ppt/.xls 文件)" "n")) {
        Clack-Skip "LibreOffice (新版格式 .docx/.pptx/.xlsx 无需)"
        Track-Install -Component "LibreOffice" -Status "skipped"
        return $false
    }

    Clack-Info "正在安装 LibreOffice..."

    # 优先级: winget > scoop > choco
    # 优先使用 WinGet
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Clack-Info "通过 WinGet 安装..."
        $null = & winget install TheDocumentFoundation.LibreOffice --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "LibreOffice 通过 WinGet 安装成功"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    # 备选：Scoop
    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        Clack-Info "通过 Scoop 安装..."
        & scoop bucket add extras 2>$null
        $null = & scoop install extras/libreoffice 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "LibreOffice 通过 Scoop 安装成功"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    # 备选：Chocolatey
    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        Clack-Info "通过 Chocolatey 安装..."
        $null = & choco install libreoffice-fresh -y 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "LibreOffice 通过 Chocolatey 安装成功"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    Clack-Warn "LibreOffice 安装失败"
    Clack-Note "手动安装方式" "winget: winget install TheDocumentFoundation.LibreOffice" "scoop: scoop install extras/libreoffice" "choco: choco install libreoffice-fresh" "下载: https://www.libreoffice.org/download/"
    Track-Install -Component "LibreOffice" -Status "failed"
    return $false
}

# 安装 FFmpeg（可选，用于音视频文件处理）
function Install-FFmpegDevZh {
    # 自动检测
    $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpegCmd) {
        try {
            $version = & ffmpeg -version 2>&1 | Select-Object -First 1
            if ($version -match "ffmpeg version ([^\s]+)") {
                Clack-Success "FFmpeg $($Matches[1])"
            } else {
                Clack-Success "FFmpeg"
            }
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        } catch {}
    }

    # 未安装，询问用户
    if (-not (Clack-Confirm "是否安装 FFmpeg？(用于处理音视频文件)" "n")) {
        Clack-Skip "FFmpeg"
        Track-Install -Component "FFmpeg" -Status "skipped"
        return $false
    }

    Clack-Info "正在安装 FFmpeg..."

    # 优先级: winget > scoop > choco
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Clack-Info "通过 WinGet 安装..."
        $null = & winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "FFmpeg 通过 WinGet 安装成功"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        Clack-Info "通过 Scoop 安装..."
        $null = & scoop install ffmpeg 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "FFmpeg 通过 Scoop 安装成功"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        Clack-Info "通过 Chocolatey 安装..."
        $null = & choco install ffmpeg -y 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "FFmpeg 通过 Chocolatey 安装成功"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    Clack-Warn "FFmpeg 安装失败"
    Clack-Note "手动安装方式" "winget: winget install Gyan.FFmpeg" "scoop: scoop install ffmpeg" "choco: choco install ffmpeg" "下载: https://ffmpeg.org/download.html"
    Track-Install -Component "FFmpeg" -Status "failed"
    return $false
}

# ============================================================
# 主逻辑
# ============================================================

function Main {
    # 欢迎信息
    Clack-Intro "Markitai 开发环境配置"

    # 检查执行策略
    if (-not (Test-ExecutionPolicyZh)) {
        # 继续，但已显示警告
    }

    # 安全检查: 管理员警告
    Test-AdminWarningZh

    # 环境检查: WSL 警告
    Test-WSLWarningZh

    # ========================================
    # 步骤 1: 检测前置依赖
    # ========================================
    Clack-Section "检测前置依赖"

    # 检测/安装 UV（用于管理 Python 和依赖）
    if (-not (Install-UVZh)) {
        Clack-Cancel "UV 安装失败，无法继续"
        exit 1
    }

    # 检测/安装 Python（通过 uv 自动安装）
    if (-not (Test-PythonZh)) {
        Clack-Cancel "Python 安装失败，无法继续"
        exit 1
    }

    # ========================================
    # 步骤 2: 配置开发环境
    # ========================================
    Clack-Section "配置开发环境"

    # 同步依赖（包含所有 extras: browser, claude-agent, copilot）
    if (-not (Sync-DependenciesZh)) {
        Clack-Cancel "依赖同步失败，无法继续"
        exit 1
    }
    Track-Install -Component "Python 依赖" -Status "installed"
    Track-Install -Component "Claude Agent SDK" -Status "installed"
    Track-Install -Component "Copilot SDK" -Status "installed"

    # 安装 pre-commit
    Install-PreCommitZh
    Track-Install -Component "pre-commit hooks" -Status "installed"

    # ========================================
    # 步骤 3: 可选组件
    # ========================================
    Clack-Section "可选组件"

    # 安装 Playwright 浏览器（SPA/JS 渲染页面需要）
    Install-PlaywrightBrowserDevZh | Out-Null

    # 安装 LibreOffice（可选，用于旧版 Office 文件）
    Install-LibreOfficeDevZh | Out-Null

    # 安装 FFmpeg（可选，用于音视频文件）
    Install-FFmpegDevZh | Out-Null

    # LLM CLI 工具
    Clack-Section "LLM CLI 工具"

    # 自动检测 Claude Code CLI
    $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($claudeCmd) {
        $version = & claude --version 2>&1 | Select-Object -First 1
        Clack-Success "Claude Code CLI 已安装: $version"
        Track-Install -Component "Claude Code CLI" -Status "installed"
    } else {
        if (Clack-Confirm "是否安装 Claude Code CLI?" "n") {
            Install-ClaudeCLIZh | Out-Null
        } else {
            Clack-Skip "Claude Code CLI"
            Track-Install -Component "Claude Code CLI" -Status "skipped"
        }
    }

    # 自动检测 Copilot CLI
    $copilotCmd = Get-Command copilot -ErrorAction SilentlyContinue
    if ($copilotCmd) {
        $version = & copilot --version 2>&1 | Select-Object -First 1
        Clack-Success "Copilot CLI 已安装: $version"
        Track-Install -Component "Copilot CLI" -Status "installed"
    } else {
        if (Clack-Confirm "是否安装 GitHub Copilot CLI?" "n") {
            Install-CopilotCLIZh | Out-Null
        } else {
            Clack-Skip "Copilot CLI"
            Track-Install -Component "Copilot CLI" -Status "skipped"
        }
    }

    # ========================================
    # 完成
    # ========================================
    $projectRoot = Get-ProjectRoot

    Clack-Note "快速开始" "激活环境: $projectRoot\.venv\Scripts\Activate.ps1" "运行测试: uv run pytest" "运行 CLI: uv run markitai --help"

    Clack-Outro "开发环境配置完成!"
}

# 运行主函数
Main
