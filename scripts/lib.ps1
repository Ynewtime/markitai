# Markitai Setup Library - Common Functions
# PowerShell 5.1+
# Encoding: UTF-8 (no BOM)

# Set UTF-8 encoding for consistent output
if (-not $script:EncodingSet) {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $script:EncodingSet = $true
}

# ============================================================
# Version Variables (can be overridden via environment)
# ============================================================
$script:MarkitaiVersion = $env:MARKITAI_VERSION
$script:UvVersion = $env:UV_VERSION

# ============================================================
# Installation Status Tracking
# ============================================================
$script:InstalledComponents = @()
$script:SkippedComponents = @()
$script:FailedComponents = @()

function Track-Install {
    param(
        [string]$Component,
        [ValidateSet("installed", "skipped", "failed")]
        [string]$Status
    )

    switch ($Status) {
        "installed" { $script:InstalledComponents += $Component }
        "skipped" { $script:SkippedComponents += $Component }
        "failed" { $script:FailedComponents += $Component }
    }
}

# ============================================================
# Output Helpers
# ============================================================

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor White
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host ""
}

# Print welcome message (user edition)
function Write-WelcomeUser {
    Write-Host ""
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host "  Welcome to Markitai Setup!" -ForegroundColor White
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  This script will install:"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Green -NoNewline; Write-Host "markitai - Markdown converter with LLM support"
    Write-Host ""
    Write-Host "  Optional components:"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Yellow -NoNewline; Write-Host "Playwright - Browser automation for JS-rendered pages"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Yellow -NoNewline; Write-Host "Claude Code CLI - Use your Claude subscription"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Yellow -NoNewline; Write-Host "Copilot CLI - Use your GitHub Copilot subscription"
    Write-Host ""
    Write-Host "  Press Ctrl+C to cancel at any time" -ForegroundColor White
    Write-Host ""
}

# Print welcome message (developer edition)
function Write-WelcomeDev {
    Write-Host ""
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host "  Markitai Development Environment Setup" -ForegroundColor White
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  This script will set up:"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Green -NoNewline; Write-Host "Python virtual environment with all dependencies"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Green -NoNewline; Write-Host "pre-commit hooks for code quality"
    Write-Host ""
    Write-Host "  Optional components:"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Yellow -NoNewline; Write-Host "Playwright - Browser automation"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Yellow -NoNewline; Write-Host "LLM CLI tools - Claude Code / Copilot"
    Write-Host "    " -NoNewline; Write-Host "* " -ForegroundColor Yellow -NoNewline; Write-Host "LLM Python SDKs - Programmatic LLM access"
    Write-Host ""
    Write-Host "  Press Ctrl+C to cancel at any time" -ForegroundColor White
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

# ============================================================
# User Interaction
# ============================================================

function Ask-YesNo {
    param(
        [string]$Question,
        [bool]$DefaultYes = $false
    )

    if ($DefaultYes) {
        $hint = "[Y/n, default: Yes]"
    } else {
        $hint = "[y/N, default: No]"
    }

    Write-Host "  " -NoNewline
    Write-Host "[?] " -ForegroundColor Yellow -NoNewline
    $answer = Read-Host "$Question $hint"

    if ([string]::IsNullOrWhiteSpace($answer)) {
        return $DefaultYes
    }

    return $answer -match "^[Yy]"
}

# ============================================================
# Security Functions
# ============================================================

function Test-AdminWarning {
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if ($isAdmin) {
        Write-Host ""
        Write-Host ("=" * 45) -ForegroundColor Yellow
        Write-Host "  WARNING: Running as Administrator" -ForegroundColor Yellow
        Write-Host ("=" * 45) -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Running setup scripts as Administrator carries risks:"
        Write-Host "  1. System-level changes may affect all users"
        Write-Host "  2. Remote code execution risks are amplified"
        Write-Host ""
        Write-Host "  Recommendation: Run as a regular user instead"
        Write-Host ""

        if (-not (Ask-YesNo "Continue as Administrator?" $false)) {
            Write-Host ""
            Write-Info "Exiting. Please run as a regular user."
            exit 1
        }
    }
}

function Test-WSLWarning {
    # Detect if running inside WSL (Windows Subsystem for Linux)
    # Note: Only check WSL_DISTRO_NAME, not WSLENV
    # WSLENV is used to configure env var sharing between Windows and WSL,
    # and may exist on Windows host even when not running inside WSL
    if ($env:WSL_DISTRO_NAME) {
        Write-Host ""
        Write-Host ("=" * 45) -ForegroundColor Yellow
        Write-Host "  WARNING: Running in WSL environment" -ForegroundColor Yellow
        Write-Host ("=" * 45) -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  You appear to be running PowerShell inside WSL."
        Write-Host "  For best results, use the native shell script instead:"
        Write-Host ""
        Write-Host "    ./scripts/setup.sh" -ForegroundColor Yellow
        Write-Host "    # or for Chinese:" -ForegroundColor Gray
        Write-Host "    ./scripts/setup-zh.sh" -ForegroundColor Yellow
        Write-Host ""

        if (-not (Ask-YesNo "Continue with PowerShell script anyway?" $false)) {
            Write-Host ""
            Write-Info "Exiting. Please use the .sh script."
            exit 1
        }
    }
}

function Confirm-RemoteScript {
    param(
        [string]$ScriptUrl,
        [string]$ScriptName
    )

    Write-Host ""
    Write-Host ("=" * 45) -ForegroundColor Yellow
    Write-Host "  WARNING: About to execute remote script" -ForegroundColor Yellow
    Write-Host ("=" * 45) -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Source: $ScriptUrl"
    Write-Host "  Purpose: Install $ScriptName"
    Write-Host ""
    Write-Host "  This will download and execute code from the internet."
    Write-Host "  Make sure you trust this source."
    Write-Host ""

    return (Ask-YesNo "Confirm execution?" $false)
}

# ============================================================
# Helper Functions
# ============================================================

function Invoke-PythonWithTimeout {
    param(
        [string]$Exe,
        [string[]]$Arguments,
        [int]$TimeoutSeconds = 5
    )

    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Exe
        # Properly quote arguments containing spaces or special characters
        $quotedArgs = @()
        foreach ($arg in $Arguments) {
            if ($arg -match '[\s;()%!^"<>&|]') {
                # Escape internal double quotes and wrap in quotes
                $quotedArgs += "`"$($arg -replace '"', '\"')`""
            } else {
                $quotedArgs += $arg
            }
        }
        $psi.Arguments = $quotedArgs -join " "
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true

        $process = [System.Diagnostics.Process]::Start($psi)
        if ($process.WaitForExit($TimeoutSeconds * 1000)) {
            if ($process.ExitCode -eq 0) {
                return $process.StandardOutput.ReadToEnd().Trim()
            }
        } else {
            $process.Kill()
        }
    } catch {
        # Silently ignore errors (timeout, process not found, etc.)
        # This is expected behavior when testing Python availability
    }
    return $null
}

function Test-RealCommand {
    param([string]$CommandName)

    $cmd = Get-Command $CommandName -ErrorAction SilentlyContinue
    if (-not $cmd) { return $false }

    # Special case: py.exe in WindowsApps is the real pymanager launcher
    if ($CommandName -eq "py" -and $cmd.Source -match "WindowsApps.*py\.exe$") {
        return $true
    }

    # Windows Store aliases are typically in WindowsApps folder
    # These are placeholder executables that redirect to Microsoft Store
    if ($cmd.Source -match "WindowsApps") { return $false }

    return $true
}

# Check if a command exists (including WindowsApps)
function Test-CommandExists {
    param([string]$CommandName)

    $cmd = Get-Command $CommandName -ErrorAction SilentlyContinue
    return ($null -ne $cmd)
}

# ============================================================
# Detection Functions
# ============================================================

function Test-Python {
    # First, check if py.exe launcher exists and get available versions
    $pyLauncher = Test-RealCommand "py"
    $availablePyVersions = @()

    if ($pyLauncher) {
        $listOutput = Invoke-PythonWithTimeout -Exe "py" -Arguments @("--list") -TimeoutSeconds 3
        if ($listOutput) {
            # Parse py --list output to find available 3.11-3.13 versions
            # Support both traditional format (-V:3.13) and new pymanager format (3.13[-64])
            $lines = $listOutput -split "`n"
            foreach ($line in $lines) {
                # Traditional py launcher format: -V:3.13
                if ($line -match "-V:3\.(1[1-3])") {
                    $minor = $Matches[1]
                    $availablePyVersions += "py -3.$minor"
                }
                # New pymanager format: 3.13[-64] or 3.13-64
                elseif ($line -match "^\s*3\.(1[1-3])[\[-]") {
                    $minor = $Matches[1]
                    $availablePyVersions += "py -3.$minor"
                }
            }
        }
    }

    # Build command list: py launcher versions first (if available), then direct commands
    $pythonCommands = @()
    $pythonCommands += $availablePyVersions | Sort-Object -Descending | Select-Object -Unique

    # Try version-specific commands (python3.13, python3.12, python3.11)
    foreach ($minor in @("13", "12", "11")) {
        foreach ($cmd in @("python3.$minor", "python3$minor")) {
            if (Test-RealCommand $cmd) {
                $pythonCommands += $cmd
            }
        }
    }

    # Only add direct python commands if they are real executables (not WindowsApps placeholder)
    foreach ($cmd in @("python", "python3")) {
        if (Test-RealCommand $cmd) {
            $pythonCommands += $cmd
        }
    }

    # Add generic py as last resort
    if ($pyLauncher -and $pythonCommands.Count -eq 0) {
        $pythonCommands += "py"
    }

    # Final fallback: try python even if in WindowsApps (pymanager makes it work)
    if ($pythonCommands.Count -eq 0) {
        foreach ($cmd in @("python", "python3")) {
            if (Test-CommandExists $cmd) {
                $pythonCommands += $cmd
            }
        }
    }

    if ($pythonCommands.Count -eq 0) {
        Write-Error2 "No Python installation found"
        Write-Host ""
        Write-Warning2 "Please install Python 3.13 (recommended) or 3.11/3.12:"
        Write-Info "Download: https://www.python.org/downloads/"
        Write-Info "winget: winget install Python.Python.3.13"
        Write-Info "scoop: scoop install python@3.13"
        Write-Info "Note: onnxruntime doesn't support Python 3.14 yet"
        return $false
    }

    foreach ($cmd in $pythonCommands) {
        $cmdParts = $cmd -split " "
        if ($cmdParts.Length -eq 0) { continue }
        $exe = $cmdParts[0]
        # Force array to prevent string concatenation issues when only one element
        $baseArgs = @()
        if ($cmdParts.Length -gt 1) {
            $baseArgs = @($cmdParts[1..($cmdParts.Length-1)])
        }

        # Use Python2-compatible syntax (no f-string)
        $versionArgs = $baseArgs + @("-c", "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))")
        $version = Invoke-PythonWithTimeout -Exe $exe -Arguments $versionArgs -TimeoutSeconds 5

        if (-not $version) { continue }

        $majorArgs = $baseArgs + @("-c", "import sys; print(sys.version_info[0])")
        $major = Invoke-PythonWithTimeout -Exe $exe -Arguments $majorArgs -TimeoutSeconds 5

        $minorArgs = $baseArgs + @("-c", "import sys; print(sys.version_info[1])")
        $minor = Invoke-PythonWithTimeout -Exe $exe -Arguments $minorArgs -TimeoutSeconds 5

        if (-not $major -or -not $minor) { continue }

        # Validate numeric
        if ($major -notmatch '^\d+$' -or $minor -notmatch '^\d+$') { continue }

        # Check version range: 3.11 <= version < 3.14
        if ($major -eq 3 -and $minor -ge 11 -and $minor -le 13) {
            $script:PYTHON_CMD = $cmd
            Write-Success "Python $version installed ($cmd)"
            return $true
        } elseif ($major -eq 3 -and $minor -ge 14) {
            Write-Warning2 "Python $version detected, but onnxruntime doesn't support Python 3.14+"
        }
    }

    Write-Error2 "Python 3.11-3.13 not found"
    Write-Host ""
    Write-Warning2 "Please install Python 3.13 (recommended) or 3.11/3.12:"
    Write-Info "Download: https://www.python.org/downloads/"
    Write-Info "scoop: scoop install python@3.13"
    Write-Info "winget: winget install Python.Python.3.13"
    Write-Info "Note: onnxruntime doesn't support Python 3.14 yet"
    return $false
}

function Test-NodeJS {
    Write-Info "Detecting Node.js..."

    $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    if (-not $nodeCmd) {
        Write-Error2 "Node.js not found"
        Write-Host ""
        Write-Warning2 "Please install Node.js 18+:"
        Write-Info "Download: https://nodejs.org/"
        Write-Info "winget: winget install OpenJS.NodeJS.LTS"
        Write-Info "scoop: scoop install nodejs-lts"
        Write-Info "choco: choco install nodejs-lts"
        return $false
    }

    try {
        $version = & node --version 2>$null
        if ($version) {
            $versionStr = $version -replace "v", ""
            $parts = $versionStr -split "\."

            # Validate numeric
            if ($parts[0] -notmatch '^\d+$') {
                Write-Warning2 "Unable to parse Node version: $version"
                return $false
            }

            $major = [int]$parts[0]

            if ($major -ge 18) {
                Write-Success "Node.js $version installed"
                return $true
            } else {
                Write-Warning2 "Node.js $version is outdated, 18+ recommended"
                return $true
            }
        }
    } catch {}

    Write-Error2 "Node.js not found"
    Write-Host ""
    Write-Warning2 "Please install Node.js 18+:"
    Write-Info "Download: https://nodejs.org/"
    Write-Info "winget: winget install OpenJS.NodeJS.LTS"
    Write-Info "scoop: scoop install nodejs-lts"
    Write-Info "choco: choco install nodejs-lts"
    return $false
}

function Test-UV {
    # Check if uv command exists first
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
        Write-Success "$version installed"
        return $true
    }
    return $false
}

# ============================================================
# Installation Functions
# ============================================================

# Install UV package manager
# Returns: 0 = success, 1 = failure, 2 = skipped
function Install-UV {
    Write-Info "Checking UV installation..."

    if (Test-UV) {
        Track-Install -Component "uv" -Status "installed"
        return 0
    }

    Write-Error2 "UV not installed"

    if (-not (Ask-YesNo "Install UV automatically?" $false)) {
        Write-Info "Skipping UV installation"
        Write-Warning2 "markitai recommends using UV for installation"
        Track-Install -Component "uv" -Status "skipped"
        return 2  # Skipped
    }

    # Build install URL (with optional version)
    if ($script:UvVersion) {
        $uvUrl = "https://astral.sh/uv/$($script:UvVersion)/install.ps1"
        Write-Info "Installing UV version: $($script:UvVersion)"
    } else {
        $uvUrl = "https://astral.sh/uv/install.ps1"
    }

    # Confirm remote script execution
    if (-not (Confirm-RemoteScript -ScriptUrl $uvUrl -ScriptName "UV")) {
        Write-Info "Skipping UV installation"
        Track-Install -Component "uv" -Status "skipped"
        return 2  # Skipped
    }

    Write-Info "Installing UV..."

    try {
        Invoke-RestMethod $uvUrl | Invoke-Expression

        # Refresh PATH (User paths take precedence)
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")

        # Check if uv command exists after PATH refresh
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uvCmd) {
            Write-Warning2 "UV installed, but PowerShell needs to be restarted"
            Write-PathHelp "$env:USERPROFILE\.local\bin"
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
            Write-Success "$version installed successfully"
            Track-Install -Component "uv" -Status "installed"
            return 0
        } else {
            Write-Warning2 "UV installed, but PowerShell needs to be restarted"
            Write-PathHelp "$env:USERPROFILE\.local\bin"
            Track-Install -Component "uv" -Status "installed"
            return 1
        }
    } catch {
        Write-Error2 "UV installation failed: $_"
        if (-not (Test-Network)) {
            Write-NetworkError
        } else {
            Write-Info "Manual install: irm https://astral.sh/uv/install.ps1 | iex"
        }
        Track-Install -Component "uv" -Status "failed"
        return 1
    }
}

function Install-Markitai {
    Write-Info "Installing markitai..."

    # Build package spec with optional version
    # Note: Use [browser] instead of [all] to avoid installing unnecessary SDK packages
    # SDK packages (claude-agent, copilot) will be installed when user selects CLI tools
    if ($script:MarkitaiVersion) {
        $pkg = "markitai[browser]==$($script:MarkitaiVersion)"
        Write-Info "Installing version: $($script:MarkitaiVersion)"
    } else {
        $pkg = "markitai[browser]"
    }

    # Build Python command for --python argument
    $pythonArg = $script:PYTHON_CMD
    if ($pythonArg -match "^py\s+-(\d+\.\d+)$") {
        $pythonArg = $Matches[1]
    }

    # Prefer uv tool install (recommended)
    $uvExists = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvExists) {
        $oldErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            # Use --upgrade to ensure latest version is installed
            $null = & uv tool install $pkg --python $pythonArg --upgrade 2>&1
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldErrorAction
        }
        if ($exitCode -eq 0) {
            # Refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
            
            $markitaiCmd = Get-Command markitai -ErrorAction SilentlyContinue
            $version = if ($markitaiCmd) { & markitai --version 2>&1 | Select-Object -First 1 } else { "installed" }
            if (-not $version) { $version = "installed" }
            Write-Success "markitai $version installed successfully (using Python $pythonArg)"
            Track-Install -Component "markitai" -Status "installed"
            return $true
        }
    }

    # Fallback to pipx
    $pipxExists = Get-Command pipx -ErrorAction SilentlyContinue
    if ($pipxExists) {
        $oldErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            # Use --force to ensure latest version is installed
            $null = & pipx install $pkg --python $pythonArg --force 2>&1
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldErrorAction
        }
    if ($exitCode -eq 0) {
        $markitaiCmd = Get-Command markitai -ErrorAction SilentlyContinue
        $version = if ($markitaiCmd) { & markitai --version 2>&1 | Select-Object -First 1 } else { "installed" }
        if (-not $version) { $version = "installed" }
        Write-Success "markitai $version installed successfully"
        Track-Install -Component "markitai" -Status "installed"
        return $true
    }
    }

    # Fallback to pip --user
    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $cmdParts = $script:PYTHON_CMD -split " "
        $exe = $cmdParts[0]
        $baseArgs = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] } else { @() }
        # Use --upgrade to ensure latest version is installed
        $pipArgs = $baseArgs + @("-m", "pip", "install", "--user", "--upgrade", $pkg)
        $null = & $exe @pipArgs 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorAction
    }
    if ($exitCode -eq 0) {
        $markitaiCmd = Get-Command markitai -ErrorAction SilentlyContinue
        $version = if ($markitaiCmd) { & markitai --version 2>&1 | Select-Object -First 1 } else { "installed" }
        if (-not $version) { $version = "installed" }
        Write-Success "markitai $version installed successfully"
        Write-PathHelp "$env:USERPROFILE\AppData\Roaming\Python\Scripts"
        Track-Install -Component "markitai" -Status "installed"
        return $true
    }

    Write-Error2 "markitai installation failed"
    if (-not (Test-Network)) {
        Write-NetworkError
    } else {
        Write-Info "Manual install: uv tool install markitai --python $pythonArg"
    }
    Track-Install -Component "markitai" -Status "failed"
    return $false
}

# Install Playwright browser (Chromium)
# Security: Use markitai's uv tool environment's playwright to ensure correct version
# Returns: $true on success, $false on failure/skip
function Install-PlaywrightBrowser {
    Write-Info "Playwright browser (Chromium):"
    Write-Info "  Purpose: Browser automation for JavaScript-rendered pages (Twitter, SPAs)"

    # Ask user consent before downloading
    if (-not (Ask-YesNo "Download Chromium browser?" $true)) {
        Write-Info "Skipping Playwright browser installation"
        Track-Install -Component "Playwright Browser" -Status "skipped"
        return $false
    }

    Write-Info "Downloading Chromium browser..."

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    # Method 1: Use playwright from markitai's uv tool environment (preferred)
    # This ensures we use the same playwright version that markitai depends on
    # Check UV_TOOL_DIR first (user override), then use default path
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
    # Fallback to default path if uv tool dir detection failed
    # Note: uv uses APPDATA (Roaming) on Windows, not LOCALAPPDATA (Local)
    if (-not $markitaiPlaywright -or -not (Test-Path $markitaiPlaywright)) {
        $markitaiPlaywright = Join-Path $env:APPDATA "uv\tools\markitai\Scripts\playwright.exe"
    }

    if (Test-Path $markitaiPlaywright) {
        try {
            & $markitaiPlaywright install chromium 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Write-Success "Chromium browser installed successfully"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        } catch {}
    }

    # Method 2: Fallback to Python module (for pip installs)
    if ($script:PYTHON_CMD) {
        $cmdParts = $script:PYTHON_CMD -split " "
        $exe = $cmdParts[0]
        $baseArgs = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] } else { @() }
        $pwArgs = $baseArgs + @("-m", "playwright", "install", "chromium")

        try {
            & $exe @pwArgs 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Write-Success "Chromium browser installed successfully"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        } catch {}
    }

    $ErrorActionPreference = $oldErrorAction
    Write-Warning2 "Playwright browser installation failed"
    Write-Info "You can install later with: playwright install chromium"
    Track-Install -Component "Playwright Browser" -Status "failed"
    return $false
}

# Install LibreOffice (optional)
# LibreOffice is required for converting .doc, .ppt, .xls files
function Install-LibreOffice {
    Write-Info "Checking LibreOffice installation..."
    Write-Info "  Purpose: Convert legacy Office files (.doc, .ppt, .xls)"

    # Check for soffice (LibreOffice command)
    $soffice = Get-Command soffice -ErrorAction SilentlyContinue
    if ($soffice) {
        try {
            $version = & soffice --version 2>&1 | Select-Object -First 1
            Write-Success "LibreOffice installed: $version"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        } catch {}
    }

    # On Windows, check common install paths
    $commonPaths = @(
        "${env:ProgramFiles}\LibreOffice\program\soffice.exe",
        "${env:ProgramFiles(x86)}\LibreOffice\program\soffice.exe"
    )

    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            Write-Success "LibreOffice found at: $path"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    Write-Warning2 "LibreOffice not installed (optional)"
    Write-Info "  Without LibreOffice, .doc/.ppt/.xls files cannot be converted"
    Write-Info "  Modern formats (.docx/.pptx/.xlsx) work without LibreOffice"

    if (-not (Ask-YesNo "Install LibreOffice?" $false)) {
        Write-Info "Skipping LibreOffice installation"
        Track-Install -Component "LibreOffice" -Status "skipped"
        return $false
    }

    Write-Info "Installing LibreOffice..."

    # Priority: winget > scoop > choco
    # Try WinGet first
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Write-Info "Installing via WinGet..."
        & winget install TheDocumentFoundation.LibreOffice --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            Write-Success "LibreOffice installed via WinGet"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    # Try Scoop as second option
    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        Write-Info "Installing via Scoop..."
        & scoop bucket add extras 2>$null
        & scoop install extras/libreoffice
        if ($LASTEXITCODE -eq 0) {
            Write-Success "LibreOffice installed via Scoop"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    # Try Chocolatey as last fallback
    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        Write-Info "Installing via Chocolatey..."
        & choco install libreoffice-fresh -y
        if ($LASTEXITCODE -eq 0) {
            Write-Success "LibreOffice installed via Chocolatey"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    Write-Warning2 "LibreOffice installation failed"
    Write-Info "Manual install options:"
    Write-Info "  winget: winget install TheDocumentFoundation.LibreOffice"
    Write-Info "  scoop: scoop install extras/libreoffice"
    Write-Info "  choco: choco install libreoffice-fresh"
    Write-Info "  Download: https://www.libreoffice.org/download/"
    Track-Install -Component "LibreOffice" -Status "failed"
    return $false
}

# Install FFmpeg (optional)
# FFmpeg is required for audio/video file processing
function Install-FFmpeg {
    Write-Info "Checking FFmpeg installation..."
    Write-Info "  Purpose: Process audio/video files (.mp3, .mp4, .wav, etc.)"

    $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpegCmd) {
        try {
            $version = & ffmpeg -version 2>&1 | Select-Object -First 1
            Write-Success "FFmpeg installed: $version"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        } catch {}
    }

    Write-Warning2 "FFmpeg not installed (optional)"
    Write-Info "  Without FFmpeg, audio/video files cannot be processed"

    if (-not (Ask-YesNo "Install FFmpeg?" $false)) {
        Write-Info "Skipping FFmpeg installation"
        Track-Install -Component "FFmpeg" -Status "skipped"
        return $false
    }

    Write-Info "Installing FFmpeg..."

    # Priority: winget > scoop > choco
    # Try WinGet first
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Write-Info "Installing via WinGet..."
        & winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            Write-Success "FFmpeg installed via WinGet"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    # Try Scoop as second option
    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        Write-Info "Installing via Scoop..."
        & scoop install ffmpeg
        if ($LASTEXITCODE -eq 0) {
            Write-Success "FFmpeg installed via Scoop"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    # Try Chocolatey as last fallback
    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        Write-Info "Installing via Chocolatey..."
        & choco install ffmpeg -y
        if ($LASTEXITCODE -eq 0) {
            Write-Success "FFmpeg installed via Chocolatey"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    Write-Warning2 "FFmpeg installation failed"
    Write-Info "Manual install options:"
    Write-Info "  winget: winget install Gyan.FFmpeg"
    Write-Info "  scoop: scoop install ffmpeg"
    Write-Info "  choco: choco install ffmpeg"
    Write-Info "  Download: https://ffmpeg.org/download.html"
    Track-Install -Component "FFmpeg" -Status "failed"
    return $false
}

# Install markitai extra package
# Usage: Install-MarkitaiExtra -ExtraName "claude-agent"
function Install-MarkitaiExtra {
    param([string]$ExtraName)

    $pkg = "markitai[$ExtraName]"

    # Build Python command for --python argument
    $pythonArg = $script:PYTHON_CMD
    if ($pythonArg -match "^py\s+-(\d+\.\d+)$") {
        $pythonArg = $Matches[1]
    }

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    # Prefer uv tool install
    $uvExists = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvExists) {
        try {
            $null = & uv tool install $pkg --python $pythonArg --upgrade 2>&1
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Write-Success "markitai[$ExtraName] installed"
                return $true
            }
        } catch {}
    }

    # Fallback to pipx
    $pipxExists = Get-Command pipx -ErrorAction SilentlyContinue
    if ($pipxExists) {
        try {
            $null = & pipx install $pkg --python $pythonArg --force 2>&1
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Write-Success "markitai[$ExtraName] installed"
                return $true
            }
        } catch {}
    }

    # Fallback to pip --user
    try {
        $cmdParts = $script:PYTHON_CMD -split " "
        $exe = $cmdParts[0]
        $baseArgs = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] } else { @() }
        $pipArgs = $baseArgs + @("-m", "pip", "install", "--user", "--upgrade", $pkg)
        $null = & $exe @pipArgs 2>&1
        if ($LASTEXITCODE -eq 0) {
            $ErrorActionPreference = $oldErrorAction
            Write-Success "markitai[$ExtraName] installed"
            return $true
        }
    } catch {}

    $ErrorActionPreference = $oldErrorAction
    Write-Warning2 "markitai[$ExtraName] installation failed"
    return $false
}

# Install Claude Code CLI
function Install-ClaudeCLI {
    Write-Info "Installing Claude Code CLI..."
    Write-Info "  Purpose: Use your Claude Pro/Team subscription with markitai"

    # Check if already installed
    $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($claudeCmd) {
        $version = & claude --version 2>&1 | Select-Object -First 1
        Write-Success "Claude Code CLI already installed: $version"
        Track-Install -Component "Claude Code CLI" -Status "installed"
        return $true
    }

    # Prefer official install script (PowerShell)
    $claudeUrl = "https://claude.ai/install.ps1"
    if (Confirm-RemoteScript -ScriptUrl $claudeUrl -ScriptName "Claude Code CLI") {
        Write-Info "Installing via official script..."
        try {
            Invoke-Expression (Invoke-RestMethod -Uri $claudeUrl)
            # Check if installation succeeded
            $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
            if ($claudeCmd) {
                Write-Success "Claude Code CLI installed via official script"
                Write-Info "Run 'claude /login' to authenticate with your Claude subscription or API key"
                Track-Install -Component "Claude Code CLI" -Status "installed"
                return $true
            }
        } catch {
            Write-Info "Official script failed: $_"
        }
    }

    # Fallback: npm/pnpm
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        Write-Info "Installing via pnpm..."
        & pnpm add -g @anthropic-ai/claude-code
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Claude Code CLI installed via pnpm"
            Write-Info "Run 'claude /login' to authenticate with your Claude subscription or API key"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        Write-Info "Installing via npm..."
        & npm install -g @anthropic-ai/claude-code
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Claude Code CLI installed via npm"
            Write-Info "Run 'claude /login' to authenticate with your Claude subscription or API key"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    }

    Write-Warning2 "Claude Code CLI installation failed"
    if (-not (Test-Network)) {
        Write-NetworkError
    } else {
        Write-Info "Manual install options:"
        Write-Info "  PowerShell: irm https://claude.ai/install.ps1 | iex"
        Write-Info "  pnpm: pnpm add -g @anthropic-ai/claude-code"
        Write-Info "  Docs: https://docs.anthropic.com/en/docs/claude-code"
    }
    Track-Install -Component "Claude Code CLI" -Status "failed"
    return $false
}

# Install GitHub Copilot CLI
function Install-CopilotCLI {
    Write-Info "Installing GitHub Copilot CLI..."
    Write-Info "  Purpose: Use your GitHub Copilot subscription with markitai"

    # Check if already installed
    $copilotCmd = Get-Command copilot -ErrorAction SilentlyContinue
    if ($copilotCmd) {
        $version = & copilot --version 2>&1 | Select-Object -First 1
        Write-Success "Copilot CLI already installed: $version"
        Track-Install -Component "Copilot CLI" -Status "installed"
        return $true
    }

    # Prefer WinGet on Windows
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Write-Info "Installing via WinGet..."
        & winget install GitHub.Copilot --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Copilot CLI installed via WinGet"
            Write-Info "Run 'copilot /login' to authenticate with your GitHub Copilot subscription"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    # Fallback: npm/pnpm
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        Write-Info "Installing via pnpm..."
        & pnpm add -g @github/copilot
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Copilot CLI installed via pnpm"
            Write-Info "Run 'copilot /login' to authenticate with your GitHub Copilot subscription"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        Write-Info "Installing via npm..."
        & npm install -g @github/copilot
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Copilot CLI installed via npm"
            Write-Info "Run 'copilot /login' to authenticate with your GitHub Copilot subscription"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    Write-Warning2 "Copilot CLI installation failed"
    if (-not (Test-Network)) {
        Write-NetworkError
    } else {
        Write-Info "Manual install options:"
        Write-Info "  winget: winget install GitHub.Copilot"
        Write-Info "  pnpm: pnpm add -g @github/copilot"
    }
    Track-Install -Component "Copilot CLI" -Status "failed"
    return $false
}

function Initialize-Config {
    Write-Info "Initializing configuration..."

    try {
        $markitaiExists = Get-Command markitai -ErrorAction SilentlyContinue
        if ($markitaiExists) {
            & markitai config init 2>$null
            Write-Success "Configuration initialized"
        }
    } catch {
        # Config initialization is optional, ignore errors
    }
}

function Write-Completion {
    Write-Host ""
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host "Setup complete!" -ForegroundColor White
    Write-Host ""
    Write-Host "  Get started:" -ForegroundColor White
    Write-Host "    markitai --help" -ForegroundColor Yellow
    Write-Host ""
}

# ============================================================
# Network and Diagnostics
# ============================================================

# Check basic network connectivity
function Test-Network {
    try {
        $response = Invoke-WebRequest -Uri "https://pypi.org" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

# Print network error diagnostics
function Write-NetworkError {
    Write-Host ""
    Write-Host "  Network Error" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Possible causes:"
    Write-Host "    1. No internet connection"
    Write-Host "    2. Firewall blocking access"
    Write-Host "    3. Proxy configuration required"
    Write-Host "    4. DNS resolution failure"
    Write-Host ""
    Write-Host "  Solutions:"
    Write-Host "    * Check your network connection"
    Write-Host "    * If behind proxy: `$env:HTTPS_PROXY = 'http://proxy:port'"
    Write-Host "    * Try again later if server is temporarily unavailable"
    Write-Host ""
}

# ============================================================
# Execution Policy Check
# ============================================================

# Check and warn about execution policy
function Test-ExecutionPolicy {
    $policy = Get-ExecutionPolicy -Scope CurrentUser
    if ($policy -eq "Restricted" -or $policy -eq "AllSigned") {
        Write-Host ""
        Write-Host ("=" * 45) -ForegroundColor Yellow
        Write-Host "  Execution Policy Warning" -ForegroundColor Yellow
        Write-Host ("=" * 45) -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Current policy: $policy"
        Write-Host "  Scripts may be blocked from running."
        Write-Host ""
        Write-Host "  To allow scripts, run:" -ForegroundColor White
        Write-Host "    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned" -ForegroundColor Yellow
        Write-Host ""
        return $false
    }
    return $true
}

# ============================================================
# PATH Configuration Helpers
# ============================================================

# Print PATH configuration help
function Write-PathHelp {
    param([string]$TargetDir)

    Write-Host ""
    Write-Host "  Command not found?" -ForegroundColor Yellow
    Write-Host "  Add to PATH:"
    Write-Host ""
    Write-Host "  Temporary (current session):" -ForegroundColor White
    Write-Host "    `$env:Path += `";$TargetDir`"" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Permanent:" -ForegroundColor White
    Write-Host "    [Environment]::SetEnvironmentVariable(" -ForegroundColor Yellow
    Write-Host "      'Path'," -ForegroundColor Yellow
    Write-Host "      [Environment]::GetEnvironmentVariable('Path', 'User') + ';$TargetDir'," -ForegroundColor Yellow
    Write-Host "      'User')" -ForegroundColor Yellow
    Write-Host ""
}

# ============================================================
# Installation Summary
# ============================================================

# Print installation summary
function Write-Summary {
    Write-Host ""
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host "  Installation Summary" -ForegroundColor White
    Write-Host ("=" * 45) -ForegroundColor Cyan
    Write-Host ""

    # Print installed components
    if ($script:InstalledComponents.Count -gt 0) {
        Write-Host "  Installed:" -ForegroundColor Green
        foreach ($comp in $script:InstalledComponents) {
            Write-Host "    " -NoNewline
            Write-Host "[OK] " -ForegroundColor Green -NoNewline
            Write-Host $comp
        }
        Write-Host ""
    }

    # Print skipped components
    if ($script:SkippedComponents.Count -gt 0) {
        Write-Host "  Skipped:" -ForegroundColor Yellow
        foreach ($comp in $script:SkippedComponents) {
            Write-Host "    " -NoNewline
            Write-Host "[--] " -ForegroundColor Yellow -NoNewline
            Write-Host $comp
        }
        Write-Host ""
    }

    # Print failed components
    if ($script:FailedComponents.Count -gt 0) {
        Write-Host "  Failed:" -ForegroundColor Red
        foreach ($comp in $script:FailedComponents) {
            Write-Host "    " -NoNewline
            Write-Host "[X] " -ForegroundColor Red -NoNewline
            Write-Host $comp
        }
        Write-Host ""
    }

    Write-Host "  Documentation: https://markitai.ynewtime.com"
    Write-Host "  Issues: https://github.com/Ynewtime/markitai/issues"
    Write-Host ""
}
