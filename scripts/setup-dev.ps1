# Markitai 环境配置脚本 (开发者版)
# PowerShell 5.1+

$ErrorActionPreference = "Stop"

# 颜色辅助函数
function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor White
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([int]$Current, [int]$Total, [string]$Text)
    Write-Host "[$Current/$Total] " -ForegroundColor Blue -NoNewline
    Write-Host $Text
}

function Write-Success {
    param([string]$Text)
    Write-Host "  " -NoNewline
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host $Text
}

function Write-Error2 {
    param([string]$Text)
    Write-Host "  " -NoNewline
    Write-Host "[X] " -ForegroundColor Red -NoNewline
    Write-Host $Text
}

function Write-Info {
    param([string]$Text)
    Write-Host "  " -NoNewline
    Write-Host "-> " -ForegroundColor Yellow -NoNewline
    Write-Host $Text
}

function Write-Warning2 {
    param([string]$Text)
    Write-Host "  " -NoNewline
    Write-Host "[!] " -ForegroundColor Yellow -NoNewline
    Write-Host $Text
}

function Ask-YesNo {
    param(
        [string]$Question,
        [bool]$DefaultYes = $false
    )

    if ($DefaultYes) {
        $prompt = "$Question [Y/n] "
    } else {
        $prompt = "$Question [y/N] "
    }

    Write-Host "  " -NoNewline
    Write-Host "[?] " -ForegroundColor Yellow -NoNewline
    $answer = Read-Host $prompt

    if ([string]::IsNullOrWhiteSpace($answer)) {
        return $DefaultYes
    }

    return $answer -match "^[Yy]"
}

# 获取项目根目录
function Get-ProjectRoot {
    $scriptDir = Split-Path -Parent $MyInvocation.ScriptName
    if (-not $scriptDir) {
        $scriptDir = $PSScriptRoot
    }
    if (-not $scriptDir) {
        $scriptDir = (Get-Location).Path
    }
    return Split-Path -Parent $scriptDir
}

# 检测 Python
function Test-Python {
    Write-Step 1 5 "检测 Python..."

    $pythonCommands = @("python", "python3", "py -3.13", "py -3.12", "py -3.11", "py")

    foreach ($cmd in $pythonCommands) {
        try {
            $cmdParts = $cmd -split " "
            $exe = $cmdParts[0]
            $args = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] } else { @() }

            $versionArgs = $args + @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
            $version = & $exe @versionArgs 2>$null

            $majorArgs = $args + @("-c", "import sys; print(sys.version_info.major)")
            $major = & $exe @majorArgs 2>$null

            $minorArgs = $args + @("-c", "import sys; print(sys.version_info.minor)")
            $minor = & $exe @minorArgs 2>$null

            if ($major -ge 3 -and $minor -ge 11) {
                $script:PYTHON_CMD = $cmd
                Write-Success "Python $version 已安装 ($cmd)"
                return $true
            }
        } catch {
            continue
        }
    }

    Write-Error2 "未找到 Python 3.11+"
    Write-Host ""
    Write-Warning2 "请安装 Python 3.11 或更高版本:"
    Write-Info "官网下载: https://www.python.org/downloads/"
    Write-Info "Microsoft Store: 搜索 Python 3.11"
    Write-Info "winget: winget install Python.Python.3.11"
    return $false
}

# 检测/安装 UV
function Test-UV {
    Write-Step 2 5 "检测 UV 包管理器..."

    try {
        $version = & uv --version 2>$null
        if ($version) {
            Write-Success "$version 已安装"
            return $true
        }
    } catch {}

    Write-Error2 "UV 未安装"

    if (Ask-YesNo "是否自动安装 UV?" $true) {
        Write-Info "正在安装 UV..."

        try {
            Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression

            # 刷新 PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

            $version = & uv --version 2>$null
            if ($version) {
                Write-Success "$version 安装成功"
                return $true
            } else {
                Write-Warning2 "UV 已安装，但需要重新打开 PowerShell"
                Write-Info "请重新打开 PowerShell 后再次运行此脚本"
                return $false
            }
        } catch {
            Write-Error2 "UV 安装失败: $_"
            Write-Info "手动安装: irm https://astral.sh/uv/install.ps1 | iex"
            return $false
        }
    } else {
        Write-Error2 "UV 是开发所必需的"
        return $false
    }
}

# 同步开发依赖
function Sync-Dependencies {
    Write-Step 3 5 "同步开发依赖..."

    $projectRoot = Get-ProjectRoot
    Write-Info "项目目录: $projectRoot"

    Push-Location $projectRoot

    try {
        Write-Info "运行 uv sync --all-extras..."
        & uv sync --all-extras
        if ($LASTEXITCODE -eq 0) {
            Write-Success "依赖同步完成"
            return $true
        } else {
            Write-Error2 "依赖同步失败"
            return $false
        }
    } finally {
        Pop-Location
    }
}

# 安装 pre-commit hooks
function Install-PreCommit {
    Write-Step 4 5 "配置 pre-commit..."

    $projectRoot = Get-ProjectRoot
    Push-Location $projectRoot

    try {
        if (Test-Path ".pre-commit-config.yaml") {
            Write-Info "安装 pre-commit hooks..."

            & uv run pre-commit install
            if ($LASTEXITCODE -eq 0) {
                Write-Success "pre-commit hooks 安装完成"
            } else {
                Write-Warning2 "pre-commit 安装失败，请手动运行: uv run pre-commit install"
            }
        } else {
            Write-Info "未找到 .pre-commit-config.yaml，跳过"
        }
    } finally {
        Pop-Location
    }
}

# 检测 Node.js
function Test-NodeJS {
    Write-Info "检测 Node.js..."

    try {
        $version = & node --version 2>$null
        if ($version) {
            $major = [int]($version -replace "v", "" -split "\.")[0]

            if ($major -ge 18) {
                Write-Success "Node.js $version 已安装"
                return $true
            } else {
                Write-Warning2 "Node.js $version 版本较低，建议 18+"
                return $true
            }
        }
    } catch {}

    Write-Error2 "未找到 Node.js"
    Write-Host ""
    Write-Warning2 "请安装 Node.js 18+:"
    Write-Info "官网下载: https://nodejs.org/"
    Write-Info "winget: winget install OpenJS.NodeJS.LTS"
    Write-Info "Chocolatey: choco install nodejs-lts"
    Write-Info "fnm: winget install Schniz.fnm"
    return $false
}

# 安装 agent-browser
function Install-AgentBrowser {
    if (-not (Test-NodeJS)) {
        Write-Warning2 "跳过 agent-browser 安装 (需要 Node.js)"
        return $false
    }

    Write-Info "正在安装 agent-browser..."

    try {
        & npm install -g agent-browser
        if ($LASTEXITCODE -eq 0) {
            Write-Success "agent-browser 安装成功"

            if (Ask-YesNo "是否下载 Chromium 浏览器?" $true) {
                Write-Info "正在下载 Chromium..."
                & agent-browser install
                Write-Success "Chromium 下载完成"
            }

            return $true
        }
    } catch {
        Write-Error2 "agent-browser 安装失败: $_"
    }

    Write-Info "请手动安装: npm install -g agent-browser"
    return $false
}

# 可选组件
function Install-Optional {
    Write-Step 5 5 "可选组件"

    if (Ask-YesNo "是否安装浏览器自动化支持 (agent-browser)?" $false) {
        Install-AgentBrowser | Out-Null
    } else {
        Write-Info "跳过 agent-browser 安装"
    }
}

# 打印完成信息
function Write-Completion {
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

# 主函数
function Main {
    Write-Header "Markitai 开发环境配置向导"

    # 检测 Python
    if (-not (Test-Python)) {
        exit 1
    }

    # 检测/安装 UV
    if (-not (Test-UV)) {
        exit 1
    }

    # 同步依赖
    if (-not (Sync-Dependencies)) {
        exit 1
    }

    # 安装 pre-commit
    Install-PreCommit

    # 可选组件
    Install-Optional

    # 完成
    Write-Completion
}

# 运行主函数
Main
