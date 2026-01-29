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
if (-not $script:ScriptDir) {
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
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Yellow -NoNewline; Write-Host "agent-browser - 浏览器自动化（JS 渲染页面）"
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

    Write-Host "  文档: https://markitai.dev"
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
        Write-InfoZh "scoop: scoop install python@3.13"
        Write-InfoZh "winget: winget install Python.Python.3.13"
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

        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

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

    if ($script:MarkitaiVersion) {
        $pkg = "markitai[all]==$($script:MarkitaiVersion)"
        Write-InfoZh "安装版本: $($script:MarkitaiVersion)"
    } else {
        $pkg = "markitai[all]"
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
            $null = & uv tool install $pkg --python $pythonArg 2>&1
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldErrorAction
        }
        if ($exitCode -eq 0) {
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
            $version = & markitai --version 2>&1 | Select-Object -First 1
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
            $null = & pipx install $pkg --python $pythonArg 2>&1
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldErrorAction
        }
        if ($exitCode -eq 0) {
            $version = & markitai --version 2>&1 | Select-Object -First 1
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
        $pipArgs = $baseArgs + @("-m", "pip", "install", "--user", $pkg)
        $null = & $exe @pipArgs 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorAction
    }
    if ($exitCode -eq 0) {
        $version = & markitai --version 2>&1 | Select-Object -First 1
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

function Test-NodeJSZh {
    Write-InfoZh "检测 Node.js..."

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
    Write-InfoZh "Chocolatey: choco install nodejs-lts"
    Write-InfoZh "fnm: winget install Schniz.fnm"
    return $false
}

# Helper function to run agent-browser command (Chinese version)
# Works around npm shim issues on Windows (both .ps1 and .cmd reference /bin/sh)
# See: https://github.com/vercel-labs/agent-browser/issues/262
function Invoke-AgentBrowserZh {
    param([string[]]$Arguments)

    $npmPrefix = & npm config get prefix 2>$null
    if ($npmPrefix) {
        $npmPrefix = $npmPrefix.Trim()

        # Try native Windows binary first (most reliable)
        $nativeBinPath = Join-Path $npmPrefix "node_modules\agent-browser\bin\agent-browser-win32-x64.exe"
        if (Test-Path $nativeBinPath) {
            & $nativeBinPath @Arguments
            return $LASTEXITCODE
        }
    }

    # Fallback: try global node_modules path
    $globalRoot = & npm root -g 2>$null
    if ($globalRoot) {
        $globalRoot = $globalRoot.Trim()

        # Try native binary in global node_modules
        $nativeBinPath = Join-Path $globalRoot "agent-browser\bin\agent-browser-win32-x64.exe"
        if (Test-Path $nativeBinPath) {
            & $nativeBinPath @Arguments
            return $LASTEXITCODE
        }

        # Fallback: run via node
        $jsPath = Join-Path $globalRoot "agent-browser\bin\agent-browser.js"
        if (Test-Path $jsPath) {
            & node $jsPath @Arguments
            return $LASTEXITCODE
        }
    }

    # Last resort: try the command directly (will likely fail due to shim bug)
    & agent-browser @Arguments
    return $LASTEXITCODE
}

function Install-AgentBrowserZh {
    if (-not (Test-NodeJSZh)) {
        Write-WarningZh "跳过 agent-browser 安装 (需要 Node.js)"
        Track-Install -Component "agent-browser" -Status "skipped"
        return $false
    }

    Write-InfoZh "正在安装 agent-browser..."

    if ($script:AgentBrowserVersion) {
        $pkg = "agent-browser@$($script:AgentBrowserVersion)"
        Write-InfoZh "安装版本: $($script:AgentBrowserVersion)"
    } else {
        $pkg = "agent-browser"
    }

    # 优先 npm，备选 pnpm
    $installSuccess = $false
    $npmExists = Get-Command npm -ErrorAction SilentlyContinue
    $pnpmExists = Get-Command pnpm -ErrorAction SilentlyContinue

    if ($npmExists) {
        Write-InfoZh "通过 npm 安装..."
        try {
            & npm install -g $pkg
            if ($LASTEXITCODE -eq 0) {
                $installSuccess = $true
            }
        } catch {
            Write-WarningZh "npm 安装失败，尝试 pnpm..."
        }
    }

    if (-not $installSuccess -and $pnpmExists) {
        Write-InfoZh "通过 pnpm 安装..."
        try {
            & pnpm add -g $pkg
            if ($LASTEXITCODE -eq 0) {
                $installSuccess = $true
            }
        } catch {
            Write-ErrorZh "pnpm 安装失败: $_"
        }
    }

    if ($installSuccess) {
        # Verify installation - check for native binary or node_modules
        $abExists = Get-Command agent-browser -ErrorAction SilentlyContinue
        $nativeBinExists = $false

        $npmPrefix = & npm config get prefix 2>$null
        if ($npmPrefix) {
            $npmPrefix = $npmPrefix.Trim()
            $nativeBinPath = Join-Path $npmPrefix "node_modules\agent-browser\bin\agent-browser-win32-x64.exe"
            $nativeBinExists = Test-Path $nativeBinPath
        }

        if (-not $nativeBinExists) {
            # Also check global node_modules
            $globalRoot = & npm root -g 2>$null
            if ($globalRoot) {
                $globalRoot = $globalRoot.Trim()
                $nativeBinPath = Join-Path $globalRoot "agent-browser\bin\agent-browser-win32-x64.exe"
                $nativeBinExists = Test-Path $nativeBinPath
            }
        }

        if (-not $abExists -and -not $nativeBinExists) {
            Write-WarningZh "agent-browser 已安装但不在 PATH 中"
            Write-InfoZh "可能需要将全局 bin 目录添加到 PATH:"
            Write-InfoZh "  pnpm bin -g  # 或: npm config get prefix"
            Track-Install -Component "agent-browser" -Status "installed"
            return $false
        }

        Write-SuccessZh "agent-browser 安装成功"
        Track-Install -Component "agent-browser" -Status "installed"

        if (Ask-YesNo "是否下载 Chromium 浏览器?" $false) {
            Write-InfoZh "正在下载 Chromium..."
            $null = Invoke-AgentBrowserZh -Arguments @("install")
            Write-SuccessZh "Chromium 下载完成"
            Track-Install -Component "Chromium" -Status "installed"
        } else {
            Write-InfoZh "跳过 Chromium 下载"
            Write-InfoZh "稍后可运行: agent-browser install"
            Track-Install -Component "Chromium" -Status "skipped"
        }

        return $true
    }

    Write-InfoZh "请手动安装: npm install -g agent-browser"
    Track-Install -Component "agent-browser" -Status "failed"
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

    try {
        $markitaiExists = Get-Command markitai -ErrorAction SilentlyContinue
        if ($markitaiExists) {
            & markitai config init 2>$null
            Write-SuccessZh "配置初始化完成"
        }
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

    # 步骤 1: 检测 Python
    Write-Step 1 5 "检测 Python..."
    if (-not (Test-PythonZh)) {
        exit 1
    }

    # 步骤 2: 检测/安装 UV (用户版可选)
    Write-Step 2 5 "检测 UV 包管理器..."
    $uvResult = Install-UVZh
    # 用户版: UV 是可选的，跳过/失败都继续

    # 步骤 3: 安装 markitai
    Write-Step 3 5 "安装 markitai..."
    if (-not (Install-MarkitaiZh)) {
        Write-SummaryZh
        exit 1
    }

    # 步骤 4: 可选 - agent-browser
    Write-Step 4 5 "可选: 浏览器自动化"
    if (Ask-YesNo "是否安装浏览器自动化支持 (agent-browser)?" $false) {
        Install-AgentBrowserZh | Out-Null
    } else {
        Write-InfoZh "跳过 agent-browser 安装"
        Track-Install -Component "agent-browser" -Status "skipped"
    }

    # 步骤 5: 可选 - LLM CLI 工具
    Write-Step 5 5 "可选: LLM CLI 工具"
    Write-InfoZh "LLM CLI 工具为 AI 提供商提供本地认证"
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

    # 初始化配置
    Initialize-ConfigZh

    # 打印总结
    Write-SummaryZh

    # 完成
    Write-CompletionZh
}

# 运行主函数
Main
