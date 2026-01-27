# Markitai Setup Script (User Edition)
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

# 检测 Python (需要 3.11-3.13，不支持 3.14+)
function Test-Python {
    Write-Step 1 4 "检测 Python..."

    # 优先使用 3.11-3.13 版本
    $pythonCommands = @("py -3.13", "py -3.12", "py -3.11", "python", "python3", "py")

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

            # 检查版本范围: 3.11 <= version < 3.14
            if ($major -eq 3 -and $minor -ge 11 -and $minor -le 13) {
                $script:PYTHON_CMD = $cmd
                Write-Success "Python $version 已安装 ($cmd)"
                return $true
            } elseif ($major -eq 3 -and $minor -ge 14) {
                Write-Warning2 "Python $version 检测到，但 onnxruntime 不支持 Python 3.14+"
            }
        } catch {
            continue
        }
    }

    Write-Error2 "未找到 Python 3.11-3.13"
    Write-Host ""
    Write-Warning2 "请安装 Python 3.13 (推荐) 或 3.11/3.12:"
    Write-Info "官网下载: https://www.python.org/downloads/"
    Write-Info "scoop: scoop install python@3.13"
    Write-Info "pim: winget install 9NQ7512CXL7T"
    Write-Info "提示: onnxruntime 暂不支持 Python 3.14"
    return $false
}

# 检测/安装 UV
function Test-UV {
    Write-Step 2 4 "检测 UV 包管理器..."

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $version = & uv --version 2>&1 | Select-Object -First 1
    } finally {
        $ErrorActionPreference = $oldErrorAction
    }
    if ($version -and $version -notmatch "error") {
        Write-Success "$version 已安装"
        return $true
    }

    Write-Error2 "UV 未安装"

    if (Ask-YesNo "是否自动安装 UV?" $true) {
        Write-Info "正在安装 UV..."

        try {
            Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression

            # 刷新 PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

            $oldErrorAction = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                $version = & uv --version 2>&1 | Select-Object -First 1
            } finally {
                $ErrorActionPreference = $oldErrorAction
            }
            if ($version -and $version -notmatch "error") {
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
        Write-Info "跳过 UV 安装"
        Write-Warning2 "markitai 推荐使用 UV 进行安装"
        return $false
    }
}

# 安装 markitai
function Install-Markitai {
    Write-Step 3 4 "安装 markitai..."

    Write-Info "正在安装..."

    # 优先使用 uv tool install（推荐方式）
    $uvExists = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvExists) {
        $oldErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $null = & uv tool install "markitai[all]" 2>&1
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldErrorAction
        }
        if ($exitCode -eq 0) {
            # 刷新 PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
            $version = & markitai --version 2>&1 | Select-Object -First 1
            if (-not $version) { $version = "已安装" }
            Write-Success "markitai $version 安装成功"
            return $true
        }
    }

    # 回退到 pipx
    $pipxExists = Get-Command pipx -ErrorAction SilentlyContinue
    if ($pipxExists) {
        $oldErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $null = & pipx install "markitai[all]" 2>&1
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldErrorAction
        }
        if ($exitCode -eq 0) {
            $version = & markitai --version 2>&1 | Select-Object -First 1
            if (-not $version) { $version = "已安装" }
            Write-Success "markitai $version 安装成功"
            return $true
        }
    }

    # 回退到 pip --user
    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $null = & pip install --user "markitai[all]" 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorAction
    }
    if ($exitCode -eq 0) {
        $version = & markitai --version 2>&1 | Select-Object -First 1
        if (-not $version) { $version = "已安装" }
        Write-Success "markitai $version 安装成功"
        Write-Warning2 "可能需要将 Python Scripts 目录添加到 PATH"
        return $true
    }

    Write-Error2 "markitai 安装失败"
    Write-Info "请手动安装: uv tool install markitai"
    return $false
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
    Write-Step 4 4 "可选组件"

    if (Ask-YesNo "是否安装浏览器自动化支持 (agent-browser)?" $false) {
        Install-AgentBrowser | Out-Null
    } else {
        Write-Info "跳过 agent-browser 安装"
    }
}

# 初始化配置
function Initialize-Config {
    Write-Info "初始化配置..."

    try {
        $markitaiExists = Get-Command markitai -ErrorAction SilentlyContinue
        if ($markitaiExists) {
            & markitai config init 2>$null
            Write-Success "配置初始化完成"
        }
    } catch {}
}

# 打印完成信息
function Write-Completion {
    Write-Host ""
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host "配置完成!" -ForegroundColor White
    Write-Host ""
    Write-Host "  开始使用:" -ForegroundColor White
    Write-Host "    markitai --help" -ForegroundColor Yellow
    Write-Host ""
}

# 主函数
function Main {
    Write-Header "Markitai 环境配置向导"

    # 检测 Python
    if (-not (Test-Python)) {
        exit 1
    }

    # 检测/安装 UV
    Test-UV | Out-Null

    # 安装 markitai
    if (-not (Install-Markitai)) {
        exit 1
    }

    # 可选组件
    Install-Optional

    # 初始化配置
    Initialize-Config

    # 完成
    Write-Completion
}

# 运行主函数
Main
