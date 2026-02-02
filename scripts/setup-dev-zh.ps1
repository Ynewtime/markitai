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
    Write-Host ("=" * 48) -ForegroundColor Cyan
    Write-Host "  开发者版需要本地仓库" -ForegroundColor White
    Write-Host ("=" * 48) -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  请先克隆仓库:"
    Write-Host ""
    Write-Host "    git clone https://github.com/Ynewtime/markitai.git" -ForegroundColor Yellow
    Write-Host "    cd markitai" -ForegroundColor Yellow
    Write-Host "    .\scripts\setup-dev-zh.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  或使用用户版快速安装:"
    Write-Host ""
    Write-Host "    irm https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup-zh.ps1 | iex" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# ============================================================
# 开发者版专用函数
# ============================================================

function Get-ProjectRoot {
    return Split-Path -Parent $ScriptDir
}

# 中文欢迎信息（开发者版）
function Write-WelcomeDevZh {
    Write-Host ""
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host "  Markitai 开发环境配置" -ForegroundColor White
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  本脚本将配置:"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Green -NoNewline; Write-Host "Python 虚拟环境及所有依赖"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Green -NoNewline; Write-Host "pre-commit hooks 代码质量检查"
    Write-Host ""
    Write-Host "  可选组件:"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Yellow -NoNewline; Write-Host "Playwright - 浏览器自动化"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Yellow -NoNewline; Write-Host "LLM CLI 工具 - Claude Code / Copilot"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Yellow -NoNewline; Write-Host "LLM Python SDKs - 程序化 LLM 访问"
    Write-Host ""
    Write-Host "  随时按 Ctrl+C 取消" -ForegroundColor White
    Write-Host ""
}

# 中文安装总结
function Write-SummaryDevZh {
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
        Write-Host "    ./scripts/setup-dev-zh.sh" -ForegroundColor Yellow
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

function Test-PythonZh {
    $pyLauncher = Test-RealCommand "py"
    $availablePyVersions = @()

    if ($pyLauncher) {
        $listOutput = Invoke-PythonWithTimeout -Exe "py" -Arguments @("--list") -TimeoutSeconds 3
        if ($listOutput) {
            # 解析 py --list 输出，支持传统格式和新版 pymanager 格式
            $lines = $listOutput -split "`n"
            foreach ($line in $lines) {
                # 传统 py launcher 格式: -V:3.13
                if ($line -match "-V:3\.(1[1-3])") {
                    $minor = $Matches[1]
                    $availablePyVersions += "py -3.$minor"
                }
                # 新版 pymanager 格式: 3.13[-64] 或 3.13-64
                elseif ($line -match "^\s*3\.(1[1-3])[\[-]") {
                    $minor = $Matches[1]
                    $availablePyVersions += "py -3.$minor"
                }
            }
        }
    }

    $pythonCommands = @()
    $pythonCommands += $availablePyVersions | Sort-Object -Descending | Select-Object -Unique

    # 尝试版本特定命令 (python3.13, python3.12, python3.11)
    foreach ($minor in @("13", "12", "11")) {
        foreach ($cmd in @("python3.$minor", "python3$minor")) {
            if (Test-RealCommand $cmd) {
                $pythonCommands += $cmd
            }
        }
    }

    foreach ($cmd in @("python", "python3")) {
        if (Test-RealCommand $cmd) {
            $pythonCommands += $cmd
        }
    }

    # 如果 py launcher 存在但没找到特定版本，添加通用 py 命令
    if ($pyLauncher -and $pythonCommands.Count -eq 0) {
        $pythonCommands += "py"
    }

    # 最后尝试：即使在 WindowsApps 中，pymanager 配置的 python 也可用
    if ($pythonCommands.Count -eq 0) {
        foreach ($cmd in @("python", "python3")) {
            if (Test-CommandExists $cmd) {
                $pythonCommands += $cmd
            }
        }
    }

    if ($pythonCommands.Count -eq 0) {
        Write-ErrorZh "未找到 Python 安装"
        Write-Host ""
        Write-WarningZh "请安装 Python 3.13 (推荐) 或 3.11/3.12:"
        Write-InfoZh "官网下载: https://www.python.org/downloads/"
        Write-InfoZh "winget: winget install Python.Python.3.13"
        Write-InfoZh "scoop: scoop install python@3.13"
        Write-InfoZh "提示: onnxruntime 暂不支持 Python 3.14"
        return $false
    }

    foreach ($cmd in $pythonCommands) {
        $cmdParts = $cmd -split " "
        $exe = $cmdParts[0]
        # Force array to prevent string concatenation issues when only one element
        $baseArgs = @(if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] } else { @() })

        $versionArgs = $baseArgs + @("-c", "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))")
        $version = Invoke-PythonWithTimeout -Exe $exe -Arguments $versionArgs -TimeoutSeconds 5

        if (-not $version) { continue }

        $majorArgs = $baseArgs + @("-c", "import sys; print(sys.version_info[0])")
        $major = Invoke-PythonWithTimeout -Exe $exe -Arguments $majorArgs -TimeoutSeconds 5

        $minorArgs = $baseArgs + @("-c", "import sys; print(sys.version_info[1])")
        $minor = Invoke-PythonWithTimeout -Exe $exe -Arguments $minorArgs -TimeoutSeconds 5

        if (-not $major -or -not $minor) { continue }
        if ($major -notmatch '^\d+$' -or $minor -notmatch '^\d+$') { continue }

        if ($major -eq 3 -and $minor -ge 11 -and $minor -le 13) {
            $script:PYTHON_CMD = $cmd
            Write-SuccessZh "Python $version 已安装 ($cmd)"
            return $true
        } elseif ($major -eq 3 -and $minor -ge 14) {
            Write-WarningZh "Python $version 检测到，但 onnxruntime 不支持 Python 3.14+"
        }
    }

    Write-ErrorZh "未找到 Python 3.11-3.13"
    Write-Host ""
    Write-WarningZh "请安装 Python 3.13 (推荐) 或 3.11/3.12:"
    Write-InfoZh "官网下载: https://www.python.org/downloads/"
    Write-InfoZh "scoop: scoop install python@3.13"
    Write-InfoZh "winget: winget install Python.Python.3.13"
    Write-InfoZh "提示: onnxruntime 暂不支持 Python 3.14"
    return $false
}

# 安装 UV（开发者版必需）
function Install-UVZh {
    Write-InfoZh "检查 UV 安装..."

    if (Test-UV) {
        Track-Install -Component "uv" -Status "installed"
        return $true
    }

    Write-ErrorZh "UV 未安装"

    if (-not (Ask-YesNo "是否自动安装 UV?" $false)) {
        Write-ErrorZh "UV 是开发所必需的"
        Track-Install -Component "uv" -Status "failed"
        return $false
    }

    if ($script:UvVersion) {
        $uvUrl = "https://astral.sh/uv/$($script:UvVersion)/install.ps1"
        Write-InfoZh "安装 UV 版本: $($script:UvVersion)"
    } else {
        $uvUrl = "https://astral.sh/uv/install.ps1"
    }

    if (-not (Confirm-RemoteScriptZh -ScriptUrl $uvUrl -ScriptName "UV")) {
        Write-ErrorZh "UV 是开发所必需的"
        Track-Install -Component "uv" -Status "failed"
        return $false
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
            Write-SuccessZh "$version 安装成功"
            Track-Install -Component "uv" -Status "installed"
            return $true
        } else {
            Write-WarningZh "UV 已安装，但需要重新打开 PowerShell"
            Write-InfoZh "请重新打开 PowerShell 后再次运行此脚本"
            Track-Install -Component "uv" -Status "installed"
            return $false
        }
    } catch {
        Write-ErrorZh "UV 安装失败: $_"
        Write-InfoZh "手动安装: irm https://astral.sh/uv/install.ps1 | iex"
        Track-Install -Component "uv" -Status "failed"
        return $false
    }
}

function Sync-DependenciesZh {
    $projectRoot = Get-ProjectRoot
    Write-InfoZh "项目目录: $projectRoot"

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
        Write-InfoZh "运行 uv sync --all-extras --python $pythonArg..."
        & uv sync --all-extras --python $pythonArg
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "依赖同步完成 (使用 Python $pythonArg)"
            return $true
        } else {
            Write-ErrorZh "依赖同步失败"
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
            Write-InfoZh "安装 pre-commit hooks..."

            $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
            if ($uvCmd) {
                & uv run pre-commit install
                if ($LASTEXITCODE -eq 0) {
                    Write-SuccessZh "pre-commit hooks 安装完成"
                } else {
                    Write-WarningZh "pre-commit 安装失败，请手动运行: uv run pre-commit install"
                }
            } else {
                Write-WarningZh "未找到 uv 命令，跳过 pre-commit 安装"
            }
        } else {
            Write-InfoZh "未找到 .pre-commit-config.yaml，跳过"
        }
    } finally {
        Pop-Location
    }
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
        Write-InfoZh "使用 pnpm 安装..."
        & pnpm add -g @anthropic-ai/claude-code
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "通过 pnpm 安装 Claude Code CLI 成功"
            Write-InfoZh "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        Write-InfoZh "使用 npm 安装..."
        & npm install -g @anthropic-ai/claude-code
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "通过 npm 安装 Claude Code CLI 成功"
            Write-InfoZh "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    }

    # 尝试 WinGet (Windows)
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Write-InfoZh "使用 WinGet 安装..."
        & winget install Anthropic.ClaudeCode
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "通过 WinGet 安装 Claude Code CLI 成功"
            Write-InfoZh "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    }

    Write-WarningZh "Claude Code CLI 安装失败"
    Write-InfoZh "手动安装选项:"
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
        Write-InfoZh "使用 pnpm 安装..."
        & pnpm add -g @github/copilot
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "通过 pnpm 安装 Copilot CLI 成功"
            Write-InfoZh "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        Write-InfoZh "使用 npm 安装..."
        & npm install -g @github/copilot
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "通过 npm 安装 Copilot CLI 成功"
            Write-InfoZh "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    # 尝试 WinGet (Windows)
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Write-InfoZh "使用 WinGet 安装..."
        & winget install GitHub.Copilot
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessZh "通过 WinGet 安装 Copilot CLI 成功"
            Write-InfoZh "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    Write-WarningZh "Copilot CLI 安装失败"
    Write-InfoZh "手动安装选项:"
    Write-InfoZh "  pnpm: pnpm add -g @github/copilot"
    Write-InfoZh "  winget: winget install GitHub.Copilot"
    Track-Install -Component "Copilot CLI" -Status "failed"
    return $false
}

# 安装 LLM CLI 工具
function Install-LLMCLIsZh {
    Write-InfoZh "LLM CLI 工具提供本地认证:"
    Write-InfoZh "  - Claude Code CLI: 使用你的 Claude 订阅"
    Write-InfoZh "  - Copilot CLI: 使用你的 GitHub Copilot 订阅"

    if (Ask-YesNo "是否安装 Claude Code CLI?" $false) {
        Install-ClaudeCLIZh | Out-Null
    } else {
        Track-Install -Component "Claude Code CLI" -Status "skipped"
    }

    if (Ask-YesNo "是否安装 GitHub Copilot CLI?" $false) {
        Install-CopilotCLIZh | Out-Null
    } else {
        Track-Install -Component "Copilot CLI" -Status "skipped"
    }
}

# 安装 Playwright 浏览器 (Chromium) - 开发环境
# 优先使用 uv run，回退到 python 模块
# 返回: $true 成功, $false 失败/跳过
function Install-PlaywrightBrowserDevZh {
    Write-InfoZh "Playwright 浏览器 (Chromium):"
    Write-InfoZh "  用途: 浏览器自动化，用于 JavaScript 渲染页面 (Twitter, SPA)"

    # 下载前先征询用户同意
    if (-not (Ask-YesNo "是否下载 Chromium 浏览器？" $true)) {
        Write-InfoZh "跳过 Playwright 浏览器安装"
        Track-Install -Component "Playwright Browser" -Status "skipped"
        return $false
    }

    Write-InfoZh "正在下载 Chromium 浏览器..."

    $projectRoot = Get-ProjectRoot
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
                Write-SuccessZh "Chromium 浏览器安装成功"
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
                Write-SuccessZh "Chromium 浏览器安装成功"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        }
    } finally {
        Pop-Location
        $ErrorActionPreference = $oldErrorAction
    }

    Write-WarningZh "Playwright 浏览器安装失败"
    Write-InfoZh "稍后可手动安装: uv run playwright install chromium"
    Track-Install -Component "Playwright Browser" -Status "failed"
    return $false
}

# 检测 LibreOffice 安装（可选，用于旧版 Office 文件）
function Install-LibreOfficeDevZh {
    Write-InfoZh "正在检测 LibreOffice..."
    Write-InfoZh "  用途: 转换旧版 Office 文件 (.doc, .ppt, .xls)"

    $soffice = Get-Command soffice -ErrorAction SilentlyContinue
    if ($soffice) {
        try {
            $version = & soffice --version 2>&1 | Select-Object -First 1
            Write-SuccessZh "LibreOffice 已安装: $version"
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

# 安装 FFmpeg（可选，用于音视频文件处理）
function Install-FFmpegDevZh {
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

function Write-CompletionZh {
    $projectRoot = Get-ProjectRoot

    Write-Host ""
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host "开发环境配置完成!" -ForegroundColor White
    Write-Host ""
    Write-Host "  激活虚拟环境:" -ForegroundColor White
    Write-Host "    $projectRoot\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  运行测试:" -ForegroundColor White
    Write-Host "    uv run pytest" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  运行 CLI:" -ForegroundColor White
    Write-Host "    uv run markitai --help" -ForegroundColor Yellow
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
    Write-WelcomeDevZh

    Write-Header "Markitai 开发环境配置向导"

    # 步骤 1: 检测 Python
    Write-Step 1 5 "检测 Python..."
    if (-not (Test-PythonZh)) {
        exit 1
    }

    # 步骤 2: 检测/安装 UV（开发者版必需）
    Write-Step 2 5 "检测 UV 包管理器..."
    if (-not (Install-UVZh)) {
        Write-SummaryDevZh
        exit 1
    }

    # 步骤 3: 同步依赖（包含所有 extras: browser, claude-agent, copilot）
    Write-Step 3 5 "同步开发依赖..."
    if (-not (Sync-DependenciesZh)) {
        Write-SummaryDevZh
        exit 1
    }
    Track-Install -Component "Python 依赖" -Status "installed"
    Track-Install -Component "Claude Agent SDK" -Status "installed"
    Track-Install -Component "Copilot SDK" -Status "installed"

    # 安装 Playwright 浏览器（SPA/JS 渲染页面需要）
    Install-PlaywrightBrowserDevZh | Out-Null

    # 安装 LibreOffice（可选，用于旧版 Office 文件）
    Install-LibreOfficeDevZh | Out-Null

    # 安装 FFmpeg（可选，用于音视频文件）
    Install-FFmpegDevZh | Out-Null

    # 步骤 4: 安装 pre-commit
    Write-Step 4 5 "配置 pre-commit..."
    Install-PreCommitZh
    Track-Install -Component "pre-commit hooks" -Status "installed"

    # 步骤 5: 可选 - LLM CLI 工具
    Write-Step 5 5 "可选: LLM CLI 工具"
    if (Ask-YesNo "是否安装 LLM CLI 工具 (Claude Code / Copilot)?" $false) {
        Install-LLMCLIsZh
    } else {
        Write-InfoZh "跳过 LLM CLI 安装"
        Track-Install -Component "Claude Code CLI" -Status "skipped"
        Track-Install -Component "Copilot CLI" -Status "skipped"
    }

    # 打印总结
    Write-SummaryDevZh

    # 完成
    Write-CompletionZh
}

# 运行主函数
Main
