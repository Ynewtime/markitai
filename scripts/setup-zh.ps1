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
# 中文输出函数
# ============================================================

# 中文欢迎信息（用户版）
function Write-WelcomeUserZh {
    Write-Host ""
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host "  欢迎使用 Markitai 安装向导!" -ForegroundColor White
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  本脚本将安装:"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Green -NoNewline; Write-Host "markitai - 支持 LLM 的 Markdown 转换器"
    Write-Host ""
    Write-Host "  可选组件:"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Yellow -NoNewline; Write-Host "Playwright - 浏览器自动化（JS 渲染页面）"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Yellow -NoNewline; Write-Host "Claude Code CLI - 使用 Claude 订阅"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Yellow -NoNewline; Write-Host "Copilot CLI - 使用 GitHub Copilot 订阅"
    Write-Host ""
    Write-Host "  随时按 Ctrl+C 取消" -ForegroundColor White
    Write-Host ""
}

# 中文安装总结
function Write-SummaryZh {
    Write-Host ""
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host "  安装总结" -ForegroundColor White
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host ""

    # 已安装
    if ($script:InstalledComponents.Count -gt 0) {
        Write-Host "  已安装:" -ForegroundColor Green
        foreach ($comp in $script:InstalledComponents) {
            Write-Host "    " -NoNewline
            Write-Host "[OK] " -ForegroundColor Green -NoNewline
            Write-Host $comp
        }
        Write-Host ""
    }

    # 已跳过
    if ($script:SkippedComponents.Count -gt 0) {
        Write-Host "  已跳过:" -ForegroundColor Yellow
        foreach ($comp in $script:SkippedComponents) {
            Write-Host "    " -NoNewline
            Write-Host "[--] " -ForegroundColor Yellow -NoNewline
            Write-Host $comp
        }
        Write-Host ""
    }

    # 安装失败
    if ($script:FailedComponents.Count -gt 0) {
        Write-Host "  安装失败:" -ForegroundColor Red
        foreach ($comp in $script:FailedComponents) {
            Write-Host "    " -NoNewline
            Write-Host "[X] " -ForegroundColor Red -NoNewline
            Write-Host $comp
        }
        Write-Host ""
    }

    Write-Host "  文档: https://markitai.ynewtime.com"
    Write-Host "  问题反馈: https://github.com/Ynewtime/markitai/issues"
    Write-Host ""
}

# 执行策略检查（中文）
function Test-ExecutionPolicyZh {
    $policy = Get-ExecutionPolicy -Scope CurrentUser
    if ($policy -eq "Restricted" -or $policy -eq "AllSigned") {
        Write-Host ""
        Write-Host ("=" * 45) -ForegroundColor Yellow
        Write-Host "  执行策略警告" -ForegroundColor Yellow
        Write-Host ("=" * 45) -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  当前策略: $policy"
        Write-Host "  脚本可能无法运行。"
        Write-Host ""
        Write-Host "  解决方法:" -ForegroundColor White
        Write-Host "    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned" -ForegroundColor Yellow
        Write-Host ""
        return $false
    }
    return $true
}

function Write-HeaderZh {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor White
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host ""
}

function Write-SuccessZh {
    param([string]$Text)
    Write-Host "  " -NoNewline
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host $Text
}

function Write-ErrorZh {
    param([string]$Text)
    Write-Host "  " -NoNewline
    Write-Host "[X] " -ForegroundColor Red -NoNewline
    Write-Host $Text
}

function Write-InfoZh {
    param([string]$Text)
    Write-Host "  " -NoNewline
    Write-Host "-> " -ForegroundColor Yellow -NoNewline
    Write-Host $Text
}

function Write-WarningZh {
    param([string]$Text)
    Write-Host "  " -NoNewline
    Write-Host "[!] " -ForegroundColor Yellow -NoNewline
    Write-Host $Text
}

function Test-AdminWarningZh {
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if ($isAdmin) {
        Write-Host ""
        Write-Host ("=" * 45) -ForegroundColor Yellow
        Write-Host "  警告: 正在以管理员身份运行" -ForegroundColor Yellow
        Write-Host ("=" * 45) -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  以管理员身份运行安装脚本存在以下风险:"
        Write-Host "  1. 系统级更改可能影响所有用户"
        Write-Host "  2. 远程代码执行风险被放大"
        Write-Host ""
        Write-Host "  建议: 使用普通用户身份运行此脚本"
        Write-Host ""

        if (-not (Ask-YesNo "是否继续以管理员身份运行?" $false)) {
            Write-Host ""
            Write-InfoZh "退出。请使用普通用户身份运行。"
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
        Write-Host ""
        Write-Host ("=" * 45) -ForegroundColor Yellow
        Write-Host "  警告: 正在 WSL 环境中运行" -ForegroundColor Yellow
        Write-Host ("=" * 45) -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  检测到您正在 WSL 中运行 PowerShell。"
        Write-Host "  建议使用原生 shell 脚本以获得最佳效果:"
        Write-Host ""
        Write-Host "    ./scripts/setup-zh.sh" -ForegroundColor Yellow
        Write-Host ""

        if (-not (Ask-YesNo "是否继续使用 PowerShell 脚本?" $false)) {
            Write-Host ""
            Write-InfoZh "退出。请使用 .sh 脚本。"
            exit 1
        }
    }
}

function Confirm-RemoteScriptZh {
    param(
        [string]$ScriptUrl,
        [string]$ScriptName
    )

    Write-Host ""
    Write-Host ("=" * 45) -ForegroundColor Yellow
    Write-Host "  警告: 即将执行远程脚本" -ForegroundColor Yellow
    Write-Host ("=" * 45) -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  来源: $ScriptUrl"
    Write-Host "  用途: 安装 $ScriptName"
    Write-Host ""
    Write-Host "  此操作将从互联网下载并执行代码。"
    Write-Host "  请确保您信任该来源。"
    Write-Host ""

    return (Ask-YesNo "确认执行?" $false)
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
                Write-SuccessZh "Python $version (uv 管理)"
                return $true
            }
        }

        # 未找到，自动安装
        Write-InfoZh "正在安装 Python 3.13..."
        $installResult = & uv python install 3.13 2>&1
        if ($LASTEXITCODE -eq 0) {
            $uvPython = & uv python find 3.13 2>$null
            if ($uvPython -and (Test-Path $uvPython)) {
                $version = & $uvPython -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>$null
                if ($version) {
                    $script:PYTHON_CMD = $uvPython
                    Write-SuccessZh "Python $version 安装成功 (uv 管理)"
                    return $true
                }
            }
        }
        Write-ErrorZh "Python 3.13 安装失败"
    } else {
        Write-ErrorZh "uv 未安装，无法管理 Python"
    }

    return $false
}

function Install-UVZh {
    Write-InfoZh "检查 UV 安装..."

    if (Test-UV) {
        Track-Install -Component "uv" -Status "installed"
        return 0
    }

    Write-ErrorZh "UV 未安装"

    if (-not (Ask-YesNo "是否自动安装 UV?" $false)) {
        Write-InfoZh "跳过 UV 安装"
        Write-WarningZh "markitai 推荐使用 UV 进行安装"
        Track-Install -Component "uv" -Status "skipped"
        return 2
    }

    if ($script:UvVersion) {
        $uvUrl = "https://astral.sh/uv/$($script:UvVersion)/install.ps1"
        Write-InfoZh "安装 UV 版本: $($script:UvVersion)"
    } else {
        $uvUrl = "https://astral.sh/uv/install.ps1"
    }

    if (-not (Confirm-RemoteScriptZh -ScriptUrl $uvUrl -ScriptName "UV")) {
        Write-InfoZh "跳过 UV 安装"
        Track-Install -Component "uv" -Status "skipped"
        return 2
    }

    Write-InfoZh "正在安装 UV..."

    try {
        Invoke-RestMethod $uvUrl | Invoke-Expression

        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")

        # Check if uv command exists after PATH refresh
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uvCmd) {
            Write-WarningZh "UV 已安装，但需要重新打开 PowerShell"
            Write-InfoZh "请重新打开 PowerShell 后再次运行此脚本"
            Track-Install -Component "uv" -Status "installed"
            return 1
        }

        $oldErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $version = & uv --version 2>&1 | Select-Object -First 1
        } finally {
            $ErrorActionPreference = $oldErrorAction
        }
        if ($version -and $version -notmatch "error") {
            Write-SuccessZh "$version 安装成功"
            Track-Install -Component "uv" -Status "installed"
            return 0
        } else {
            Write-WarningZh "UV 已安装，但需要重新打开 PowerShell"
            Write-InfoZh "请重新打开 PowerShell 后再次运行此脚本"
            Track-Install -Component "uv" -Status "installed"
            return 1
        }
    } catch {
        Write-ErrorZh "UV 安装失败: $_"
        Write-InfoZh "手动安装: irm https://astral.sh/uv/install.ps1 | iex"
        Track-Install -Component "uv" -Status "failed"
        return 1
    }
}

function Install-MarkitaiZh {
    Write-InfoZh "正在安装 markitai..."

    # 注意: 使用 [browser] 而非 [all] 避免安装不必要的 SDK 包
    # SDK 包 (claude-agent, copilot) 将在用户选择安装 CLI 工具时安装
    if ($script:MarkitaiVersion) {
        $pkg = "markitai[browser]==$($script:MarkitaiVersion)"
        Write-InfoZh "安装版本: $($script:MarkitaiVersion)"
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
            # 使用 --upgrade 确保安装最新版本
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
            Write-SuccessZh "markitai $version 安装成功 (使用 Python $pythonArg)"
            Track-Install -Component "markitai" -Status "installed"
            return $true
        }
    }

    $pipxExists = Get-Command pipx -ErrorAction SilentlyContinue
    if ($pipxExists) {
        $oldErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            # 使用 --force 确保安装最新版本
            $null = & pipx install $pkg --python $pythonArg --force 2>&1
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldErrorAction
        }
        if ($exitCode -eq 0) {
            $markitaiCmd = Get-Command markitai -ErrorAction SilentlyContinue
            $version = if ($markitaiCmd) { & markitai --version 2>&1 | Select-Object -First 1 } else { "已安装" }
            if (-not $version) { $version = "已安装" }
            Write-SuccessZh "markitai $version 安装成功"
            Track-Install -Component "markitai" -Status "installed"
            return $true
        }
    }

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $cmdParts = $script:PYTHON_CMD -split " "
        $exe = $cmdParts[0]
        $baseArgs = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] } else { @() }
        # 使用 --upgrade 确保安装最新版本
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
        Write-SuccessZh "markitai $version 安装成功"
        Write-WarningZh "可能需要将 Python Scripts 目录添加到 PATH"
        Track-Install -Component "markitai" -Status "installed"
        return $true
    }

    Write-ErrorZh "markitai 安装失败"
    Write-InfoZh "请手动安装: uv tool install markitai --python $pythonArg"
    Track-Install -Component "markitai" -Status "failed"
    return $false
}

# 安装 Playwright 浏览器 (Chromium)
# 安全性: 使用 markitai 虚拟环境中的 playwright 确保使用正确版本
# 返回: $true 成功, $false 失败/跳过
function Install-PlaywrightBrowserZh {
    Write-InfoZh "Playwright 浏览器 (Chromium):"
    Write-InfoZh "  用途: 浏览器自动化，用于 JavaScript 渲染页面 (Twitter, SPA)"

    # 下载前先征询用户同意
    if (-not (Ask-YesNo "是否下载 Chromium 浏览器？" $true)) {
        Write-InfoZh "跳过 Playwright 浏览器安装"
        Track-Install -Component "Playwright Browser" -Status "skipped"
        return $false
    }

    Write-InfoZh "正在下载 Chromium 浏览器..."

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    # 方法 1: 使用 markitai 的 uv tool 环境中的 playwright（首选）
    # 确保使用与 markitai 依赖相同的 playwright 版本
    # 先检查 UV_TOOL_DIR（用户覆盖），然后使用默认路径
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
    # 如果 uv tool dir 检测失败，回退到默认路径
    # 注意: uv 在 Windows 上使用 APPDATA (Roaming)，而不是 LOCALAPPDATA (Local)
    if (-not $markitaiPlaywright -or -not (Test-Path $markitaiPlaywright)) {
        $markitaiPlaywright = Join-Path $env:APPDATA "uv\tools\markitai\Scripts\playwright.exe"
    }

    if (Test-Path $markitaiPlaywright) {
        try {
            # 显示下载进度（Chromium 约 200MB）
            & $markitaiPlaywright install chromium
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Write-SuccessZh "Chromium 浏览器安装成功"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        } catch {}
    }

    # 方法 2: 回退到 Python 模块（用于 pip 安装）
    if ($script:PYTHON_CMD) {
        $cmdParts = $script:PYTHON_CMD -split " "
        $exe = $cmdParts[0]
        $baseArgs = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] } else { @() }
        $pwArgs = $baseArgs + @("-m", "playwright", "install", "chromium")

        try {
            # 显示下载进度
            & $exe @pwArgs
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Write-SuccessZh "Chromium 浏览器安装成功"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        } catch {}
    }

    $ErrorActionPreference = $oldErrorAction
    Write-WarningZh "Playwright 浏览器安装失败"
    Write-InfoZh "稍后可手动安装: playwright install chromium"
    Track-Install -Component "Playwright Browser" -Status "failed"
    return $false
}

# 检测 LibreOffice 安装
# LibreOffice 用于转换 .doc, .ppt, .xls 文件
function Install-LibreOfficeZh {
    Write-InfoZh "正在检测 LibreOffice..."
    Write-InfoZh "  用途: 转换旧版 Office 文件 (.doc, .ppt, .xls)"

    # 检测 soffice 命令
    $soffice = Get-Command soffice -ErrorAction SilentlyContinue
    if ($soffice) {
        try {
            $version = & soffice --version 2>&1 | Select-Object -First 1
            Write-SuccessZh "LibreOffice 已安装: $version"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        } catch {}
    }

    # Windows 常见安装路径
    $commonPaths = @(
        "${env:ProgramFiles}\LibreOffice\program\soffice.exe",
        "${env:ProgramFiles(x86)}\LibreOffice\program\soffice.exe"
    )

    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            Write-SuccessZh "LibreOffice 已安装: $path"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    Write-WarningZh "LibreOffice 未安装（可选）"
    Write-InfoZh "  若未安装，无法转换 .doc/.ppt/.xls 文件"
    Write-InfoZh "  新版格式 (.docx/.pptx/.xlsx) 无需 LibreOffice"

    if (-not (Ask-YesNo "是否安装 LibreOffice？" $false)) {
        Write-InfoZh "跳过 LibreOffice 安装"
        Track-Install -Component "LibreOffice" -Status "skipped"
        return $false
    }

    Write-InfoZh "正在安装 LibreOffice..."

    # 优先级: winget > scoop > choco
    # 优先使用 WinGet
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Write-InfoZh "通过 WinGet 安装..."
        & winget install TheDocumentFoundation.LibreOffice --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "LibreOffice 通过 WinGet 安装成功"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    # 备选：Scoop
    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        Write-InfoZh "通过 Scoop 安装..."
        & scoop bucket add extras 2>$null
        & scoop install extras/libreoffice
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "LibreOffice 通过 Scoop 安装成功"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    # 备选：Chocolatey
    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        Write-InfoZh "通过 Chocolatey 安装..."
        & choco install libreoffice-fresh -y
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "LibreOffice 通过 Chocolatey 安装成功"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    Write-WarningZh "LibreOffice 安装失败"
    Write-InfoZh "手动安装方式:"
    Write-InfoZh "  winget: winget install TheDocumentFoundation.LibreOffice"
    Write-InfoZh "  scoop: scoop install extras/libreoffice"
    Write-InfoZh "  choco: choco install libreoffice-fresh"
    Write-InfoZh "  下载: https://www.libreoffice.org/download/"
    Track-Install -Component "LibreOffice" -Status "failed"
    return $false
}

# 检测 FFmpeg 安装
# FFmpeg 用于处理音视频文件
function Install-FFmpegZh {
    Write-InfoZh "正在检测 FFmpeg..."
    Write-InfoZh "  用途: 处理音视频文件 (.mp3, .mp4, .wav 等)"

    $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpegCmd) {
        try {
            $version = & ffmpeg -version 2>&1 | Select-Object -First 1
            Write-SuccessZh "FFmpeg 已安装: $version"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        } catch {}
    }

    Write-WarningZh "FFmpeg 未安装（可选）"
    Write-InfoZh "  若未安装，无法处理音视频文件"

    if (-not (Ask-YesNo "是否安装 FFmpeg？" $false)) {
        Write-InfoZh "跳过 FFmpeg 安装"
        Track-Install -Component "FFmpeg" -Status "skipped"
        return $false
    }

    Write-InfoZh "正在安装 FFmpeg..."

    # 优先级: winget > scoop > choco
    # 优先使用 WinGet
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Write-InfoZh "通过 WinGet 安装..."
        & winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "FFmpeg 通过 WinGet 安装成功"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    # 备选：Scoop
    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        Write-InfoZh "通过 Scoop 安装..."
        & scoop install ffmpeg
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "FFmpeg 通过 Scoop 安装成功"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    # 备选：Chocolatey
    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        Write-InfoZh "通过 Chocolatey 安装..."
        & choco install ffmpeg -y
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "FFmpeg 通过 Chocolatey 安装成功"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    Write-WarningZh "FFmpeg 安装失败"
    Write-InfoZh "手动安装方式:"
    Write-InfoZh "  winget: winget install Gyan.FFmpeg"
    Write-InfoZh "  scoop: scoop install ffmpeg"
    Write-InfoZh "  choco: choco install ffmpeg"
    Write-InfoZh "  下载: https://ffmpeg.org/download.html"
    Track-Install -Component "FFmpeg" -Status "failed"
    return $false
}

function Test-NodeJSZh {
    Write-InfoZh "检测 Node.js..."

    $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    if (-not $nodeCmd) {
        Write-ErrorZh "未找到 Node.js"
        Write-Host ""
        Write-WarningZh "请安装 Node.js 18+:"
        Write-InfoZh "官网下载: https://nodejs.org/"
        Write-InfoZh "winget: winget install OpenJS.NodeJS.LTS"
        Write-InfoZh "scoop: scoop install nodejs-lts"
        Write-InfoZh "choco: choco install nodejs-lts"
        return $false
    }

    try {
        $version = & node --version 2>$null
        if ($version) {
            $versionStr = $version -replace "v", ""
            $parts = $versionStr -split "\."

            if ($parts[0] -notmatch '^\d+$') {
                Write-WarningZh "无法解析 Node 版本: $version"
                return $false
            }

            $major = [int]$parts[0]

            if ($major -ge 18) {
                Write-SuccessZh "Node.js $version 已安装"
                return $true
            } else {
                Write-WarningZh "Node.js $version 版本较低，建议 18+"
                return $true
            }
        }
    } catch {}

    Write-ErrorZh "未找到 Node.js"
    Write-Host ""
    Write-WarningZh "请安装 Node.js 18+:"
    Write-InfoZh "官网下载: https://nodejs.org/"
    Write-InfoZh "winget: winget install OpenJS.NodeJS.LTS"
    Write-InfoZh "scoop: scoop install nodejs-lts"
    Write-InfoZh "choco: choco install nodejs-lts"
    return $false
}

# 安装 Claude Code CLI
function Install-ClaudeCLIZh {
    Write-InfoZh "正在安装 Claude Code CLI..."

    # 检查是否已安装
    $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($claudeCmd) {
        $version = & claude --version 2>&1 | Select-Object -First 1
        Write-SuccessZh "Claude Code CLI 已安装: $version"
        Track-Install -Component "Claude Code CLI" -Status "installed"
        return $true
    }

    # 尝试 npm/pnpm
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        Write-InfoZh "通过 pnpm 安装..."
        & pnpm add -g @anthropic-ai/claude-code
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "Claude Code CLI 安装成功 (pnpm)"
            Write-InfoZh "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        Write-InfoZh "通过 npm 安装..."
        & npm install -g @anthropic-ai/claude-code
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "Claude Code CLI 安装成功 (npm)"
            Write-InfoZh "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    }

    # 尝试 WinGet
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Write-InfoZh "通过 WinGet 安装..."
        & winget install Anthropic.ClaudeCode
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "Claude Code CLI 安装成功 (WinGet)"
            Write-InfoZh "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    }

    Write-WarningZh "Claude Code CLI 安装失败"
    Write-InfoZh "手动安装方式:"
    Write-InfoZh "  pnpm: pnpm add -g @anthropic-ai/claude-code"
    Write-InfoZh "  winget: winget install Anthropic.ClaudeCode"
    Write-InfoZh "  文档: https://code.claude.com/docs/en/setup"
    Track-Install -Component "Claude Code CLI" -Status "failed"
    return $false
}

# 安装 GitHub Copilot CLI
function Install-CopilotCLIZh {
    Write-InfoZh "正在安装 GitHub Copilot CLI..."

    # 检查是否已安装
    $copilotCmd = Get-Command copilot -ErrorAction SilentlyContinue
    if ($copilotCmd) {
        $version = & copilot --version 2>&1 | Select-Object -First 1
        Write-SuccessZh "Copilot CLI 已安装: $version"
        Track-Install -Component "Copilot CLI" -Status "installed"
        return $true
    }

    # 尝试 npm/pnpm
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        Write-InfoZh "通过 pnpm 安装..."
        & pnpm add -g @github/copilot
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "Copilot CLI 安装成功 (pnpm)"
            Write-InfoZh "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        Write-InfoZh "通过 npm 安装..."
        & npm install -g @github/copilot
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "Copilot CLI 安装成功 (npm)"
            Write-InfoZh "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    # 尝试 WinGet
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Write-InfoZh "通过 WinGet 安装..."
        & winget install GitHub.Copilot
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "Copilot CLI 安装成功 (WinGet)"
            Write-InfoZh "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    Write-WarningZh "Copilot CLI 安装失败"
    Write-InfoZh "手动安装方式:"
    Write-InfoZh "  pnpm: pnpm add -g @github/copilot"
    Write-InfoZh "  winget: winget install GitHub.Copilot"
    Track-Install -Component "Copilot CLI" -Status "failed"
    return $false
}

function Initialize-ConfigZh {
    Write-InfoZh "初始化配置..."

    $markitaiExists = Get-Command markitai -ErrorAction SilentlyContinue
    if (-not $markitaiExists) {
        return
    }

    $configPath = Join-Path $env:USERPROFILE ".markitai\config.json"
    $yesFlag = ""

    # Check if config exists and ask user
    if (Test-Path $configPath) {
        if (Ask-YesNo "$configPath 已存在，是否覆盖？" $false) {
            $yesFlag = "--yes"
        } else {
            Write-InfoZh "保留现有配置"
            return
        }
    }

    try {
        if ($yesFlag) {
            & markitai config init $yesFlag 2>$null
        } else {
            & markitai config init 2>$null
        }
        Write-SuccessZh "配置初始化完成"
    } catch {}
}

function Write-CompletionZh {
    Write-Host ""
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host "配置完成!" -ForegroundColor White
    Write-Host ""
    Write-Host "  开始使用:" -ForegroundColor White
    Write-Host "    markitai --help" -ForegroundColor Yellow
    Write-Host ""
}

# ============================================================
# 主逻辑
# ============================================================

function Main {
    # 检查执行策略
    if (-not (Test-ExecutionPolicyZh)) {
        # 继续，但已显示警告
    }

    # 安全检查: 管理员警告
    Test-AdminWarningZh

    # 环境检查: WSL 警告
    Test-WSLWarningZh

    # 欢迎信息
    Write-WelcomeUserZh

    Write-HeaderZh "Markitai 环境配置向导"

    # 步骤 1: 检测/安装 UV（用于管理 Python 和依赖）
    Write-Step 1 5 "检测 UV 包管理器..."
    if (-not (Install-UVZh)) {
        Write-SummaryZh
        exit 1
    }

    # 步骤 2: 检测/安装 Python（通过 uv 自动安装）
    Write-Step 2 5 "检测 Python..."
    if (-not (Test-PythonZh)) {
        exit 1
    }

    # 步骤 3: 安装 markitai
    Write-Step 3 5 "安装 markitai..."
    if (-not (Install-MarkitaiZh)) {
        Write-SummaryZh
        exit 1
    }

    # 安装 Playwright 浏览器 (SPA/JS 渲染页面需要)
    Install-PlaywrightBrowserZh | Out-Null

    # 安装 LibreOffice（可选，用于旧版 Office 文件）
    Install-LibreOfficeZh | Out-Null

    # 检测 FFmpeg（可选，用于音视频文件）
    Install-FFmpegZh | Out-Null

    # 步骤 4: 可选 - LLM CLI 工具
    Write-Step 4 5 "可选: LLM CLI 工具"
    Write-InfoZh "LLM CLI 工具为 AI 提供商提供本地认证"
    if (Ask-YesNo "是否安装 Claude Code CLI?" $false) {
        if (Install-ClaudeCLIZh) {
            # 安装 Claude Agent SDK 以支持编程式访问
            Install-MarkitaiExtra -ExtraName "claude-agent" | Out-Null
        }
    } else {
        Track-Install -Component "Claude Code CLI" -Status "skipped"
    }
    if (Ask-YesNo "是否安装 GitHub Copilot CLI?" $false) {
        if (Install-CopilotCLIZh) {
            # 安装 Copilot SDK 以支持编程式访问
            Install-MarkitaiExtra -ExtraName "copilot" | Out-Null
        }
    } else {
        Track-Install -Component "Copilot CLI" -Status "skipped"
    }

    # 初始化配置
    Initialize-ConfigZh

    # 打印总结
    Write-SummaryZh

    # 完成
    Write-CompletionZh
}

# 运行主函数
Main
