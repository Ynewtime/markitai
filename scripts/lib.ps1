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
    Write-Host "  $Text" -ForegroundColor White
    Write-Host ""
}

# Simplified status output (matches bash script style)
function Write-Status {
    param(
        [string]$Status,
        [string]$Message
    )
    switch ($Status) {
        "ok"   { Write-Host "  [OK] $Message" -ForegroundColor Green }
        "skip" { Write-Host "  [--] $Message" -ForegroundColor Yellow }
        "fail" { Write-Host "  [X] $Message" -ForegroundColor Red }
        "info" { Write-Host "  -> $Message" -ForegroundColor Cyan }
    }
}

# Run command silently, capture and only show output on error
function Invoke-Silent {
    param([scriptblock]$Command)
    try {
        $output = & $Command 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host $output -ForegroundColor Red
        }
        return $LASTEXITCODE -eq 0
    } catch {
        Write-Host $_.Exception.Message -ForegroundColor Red
        return $false
    }
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
# Clack-style Visual Components
# Inspired by @clack/prompts - beautiful CLI with guide lines
# ============================================================

# Session intro - start of CLI flow
# Usage: Clack-Intro "Title"
function Clack-Intro {
    param([string]$Title)
    Write-Host ""
    Write-Host ([char]0x250C) -ForegroundColor DarkGray -NoNewline
    Write-Host "  $Title"
    Write-Host ([char]0x2502) -ForegroundColor DarkGray
}

# Session outro - end of CLI flow
# Usage: Clack-Outro "Message"
function Clack-Outro {
    param([string]$Message)
    Write-Host ([char]0x2502) -ForegroundColor DarkGray
    Write-Host ([char]0x2514) -ForegroundColor DarkGray -NoNewline
    Write-Host "  $Message" -ForegroundColor Green
    Write-Host ""
}

# Section header with active marker
# Usage: Clack-Section "Section title"
function Clack-Section {
    param([string]$Title)
    Write-Host ([char]0x2502) -ForegroundColor DarkGray
    Write-Host ([char]0x25C6) -ForegroundColor Magenta -NoNewline
    Write-Host "  $Title"
}

# Log with guide line - success
# Usage: Clack-Success "Message"
function Clack-Success {
    param([string]$Message)
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "  " -NoNewline
    Write-Host ([char]0x2713) -ForegroundColor Green -NoNewline
    Write-Host " $Message"
}

# Log with guide line - error
# Usage: Clack-Error "Message"
function Clack-Error {
    param([string]$Message)
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "  " -NoNewline
    Write-Host ([char]0x2717) -ForegroundColor Red -NoNewline
    Write-Host " $Message"
}

# Log with guide line - warning
# Usage: Clack-Warn "Message"
function Clack-Warn {
    param([string]$Message)
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "  " -NoNewline
    Write-Host "!" -ForegroundColor Yellow -NoNewline
    Write-Host " $Message"
}

# Log with guide line - info
# Usage: Clack-Info "Message"
function Clack-Info {
    param([string]$Message)
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "  " -NoNewline
    Write-Host ([char]0x2192) -ForegroundColor Cyan -NoNewline
    Write-Host " $Message"
}

# Log with guide line - skipped
# Usage: Clack-Skip "Message"
function Clack-Skip {
    param([string]$Message)
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "  " -NoNewline
    Write-Host ([char]0x25CB) -ForegroundColor DarkGray -NoNewline
    Write-Host " $Message" -ForegroundColor DarkGray
}

# Log with guide line - plain text
# Usage: Clack-Log "Message"
function Clack-Log {
    param([string]$Message)
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "  $Message"
}

# Spinner with guide line
# Usage: Clack-Spinner "Message" -ScriptBlock { ... }
# Shows spinner while command runs, then shows result
function Clack-Spinner {
    param(
        [string]$Message,
        [scriptblock]$ScriptBlock
    )

    # Spinner frames (ASCII compatible)
    $spinnerChars = @('|', '/', '-', '\')
    $spinnerIndex = 0

    # Start the job
    $job = Start-Job -ScriptBlock $ScriptBlock

    # Show spinner while job is running
    while ($job.State -eq 'Running') {
        $char = $spinnerChars[$spinnerIndex]
        Write-Host "`r$([char]0x2502)  " -ForegroundColor DarkGray -NoNewline
        Write-Host $char -ForegroundColor Cyan -NoNewline
        Write-Host " $Message" -NoNewline
        $spinnerIndex = ($spinnerIndex + 1) % 4
        Start-Sleep -Milliseconds 100
    }

    # Clear spinner line
    Write-Host "`r$(' ' * ($Message.Length + 10))" -NoNewline
    Write-Host "`r" -NoNewline

    # Get job result
    $result = Receive-Job -Job $job
    $hadError = $job.State -eq 'Failed'
    Remove-Job -Job $job -Force

    if ($hadError) {
        return $false
    }
    return $true
}

# Confirm prompt with guide line
# Usage: Clack-Confirm "Question?" "y|n"
# Returns: $true for yes, $false for no
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

    Write-Host ([char]0x2502) -ForegroundColor DarkGray
    Write-Host ([char]0x25C7) -ForegroundColor Cyan -NoNewline
    $answer = Read-Host "  $Prompt [$hint]"

    if ([string]::IsNullOrWhiteSpace($answer)) {
        $answer = $Default
    }

    return $answer -match "^[Yy]"
}

# Note/message box with guide line
# Usage: Clack-Note "Title" "Line1" "Line2" ...
# Or:    Clack-Note "Title" @("Line1", "Line2")
function Clack-Note {
    param(
        [Parameter(Position=0)]
        [string]$Title,
        [Parameter(Position=1, ValueFromRemainingArguments=$true)]
        [string[]]$Lines
    )

    Write-Host ([char]0x2502) -ForegroundColor DarkGray
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "  " -NoNewline
    Write-Host ([char]0x256D) -ForegroundColor DarkGray -NoNewline
    Write-Host ([char]0x2500) -ForegroundColor DarkGray -NoNewline
    Write-Host " $Title"

    foreach ($line in $Lines) {
        Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
        Write-Host "  " -NoNewline
        Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
        Write-Host "  $line"
    }

    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "  " -NoNewline
    Write-Host ([char]0x2570) -ForegroundColor DarkGray -NoNewline
    Write-Host ([char]0x2500) -ForegroundColor DarkGray
}

# Cancel message
# Usage: Clack-Cancel "Message"
function Clack-Cancel {
    param([string]$Message)
    Write-Host ([char]0x2502) -ForegroundColor DarkGray
    Write-Host ([char]0x2514) -ForegroundColor DarkGray -NoNewline
    Write-Host "  $Message" -ForegroundColor Red
    Write-Host ""
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
    # Use uv-managed Python 3.13
    if (Test-CommandExists "uv") {
        $uvPython = & uv python find 3.13 2>$null
        if ($uvPython -and (Test-Path $uvPython)) {
            $version = & $uvPython -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>$null
            if ($version) {
                $script:PYTHON_CMD = $uvPython
                Write-Status -Status "ok" -Message "Python $version"
                return $true
            }
        }

        # Not found, auto-install
        Write-Status -Status "info" -Message "Installing Python 3.13..."
        $null = & uv python install 3.13 2>&1
        if ($LASTEXITCODE -eq 0) {
            $uvPython = & uv python find 3.13 2>$null
            if ($uvPython -and (Test-Path $uvPython)) {
                $version = & $uvPython -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>$null
                if ($version) {
                    $script:PYTHON_CMD = $uvPython
                    Write-Status -Status "ok" -Message "Python $version installed"
                    return $true
                }
            }
        }
        Write-Status -Status "fail" -Message "Python 3.13 installation failed"
    } else {
        Write-Status -Status "fail" -Message "uv not installed"
    }

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
        return $true
    }
    return $false
}

# ============================================================
# Installation Functions
# ============================================================

# Install UV package manager
# Returns: $true on success, $false on failure/skip
function Install-UV {
    if (Test-UV) {
        $version = (& uv --version 2>$null).Split(' ')[1]
        Write-Status -Status "ok" -Message "uv $version"
        Track-Install -Component "uv" -Status "installed"
        return $true
    }

    Write-Status -Status "info" -Message "Installing uv..."

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
            Write-Status -Status "fail" -Message "uv installed but needs shell restart"
            Track-Install -Component "uv" -Status "installed"
            return $false
        }

        $version = (& uv --version 2>$null).Split(' ')[1]
        Write-Status -Status "ok" -Message "uv $version installed"
        Track-Install -Component "uv" -Status "installed"
        return $true
    } catch {
        Write-Status -Status "fail" -Message "uv installation failed"
        Track-Install -Component "uv" -Status "failed"
        return $false
    }
}

function Install-Markitai {
    Write-Status -Status "info" -Message "Installing markitai..."

    # Build package spec with optional version
    # Note: Use [browser] instead of [all] to avoid installing unnecessary SDK packages
    # SDK packages (claude-agent, copilot) will be installed when user selects CLI tools
    if ($script:MarkitaiVersion) {
        $pkg = "markitai[browser]==$($script:MarkitaiVersion)"
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
            Write-Status -Status "ok" -Message "markitai $version"
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
            Write-Status -Status "ok" -Message "markitai $version"
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
        Write-Status -Status "ok" -Message "markitai $version"
        Track-Install -Component "markitai" -Status "installed"
        return $true
    }

    Write-Status -Status "fail" -Message "markitai installation failed"
    Track-Install -Component "markitai" -Status "failed"
    return $false
}

# Install Playwright browser (Chromium)
# Security: Use markitai's uv tool environment's playwright to ensure correct version
# Returns: $true on success, $false on failure/skip
function Install-PlaywrightBrowser {
    Write-Status -Status "info" -Message "Installing Playwright browser..."

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
            $null = & $markitaiPlaywright install chromium 2>&1
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Write-Status -Status "ok" -Message "Playwright browser"
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
            $null = & $exe @pwArgs 2>&1
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Write-Status -Status "ok" -Message "Playwright browser"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        } catch {}
    }

    $ErrorActionPreference = $oldErrorAction
    Write-Status -Status "fail" -Message "Playwright browser"
    Track-Install -Component "Playwright Browser" -Status "failed"
    return $false
}

# Install LibreOffice (optional)
# LibreOffice is required for converting .doc, .ppt, .xls files
function Install-LibreOffice {
    # Check for soffice (LibreOffice command)
    $soffice = Get-Command soffice -ErrorAction SilentlyContinue
    if ($soffice) {
        Write-Status -Status "ok" -Message "LibreOffice"
        Track-Install -Component "LibreOffice" -Status "installed"
        return $true
    }

    # On Windows, check common install paths
    $commonPaths = @(
        "${env:ProgramFiles}\LibreOffice\program\soffice.exe",
        "${env:ProgramFiles(x86)}\LibreOffice\program\soffice.exe"
    )

    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            Write-Status -Status "ok" -Message "LibreOffice"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    Write-Status -Status "info" -Message "Installing LibreOffice..."

    # Priority: winget > scoop > choco
    # Try WinGet first
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        $null = & winget install TheDocumentFoundation.LibreOffice --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status -Status "ok" -Message "LibreOffice"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    # Try Scoop as second option
    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        $null = & scoop bucket add extras 2>$null
        $null = & scoop install extras/libreoffice 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status -Status "ok" -Message "LibreOffice"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    # Try Chocolatey as last fallback
    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        $null = & choco install libreoffice-fresh -y 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status -Status "ok" -Message "LibreOffice"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    Write-Status -Status "fail" -Message "LibreOffice"
    Track-Install -Component "LibreOffice" -Status "failed"
    return $false
}

# Install FFmpeg (optional)
# FFmpeg is required for audio/video file processing
function Install-FFmpeg {
    $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpegCmd) {
        Write-Status -Status "ok" -Message "FFmpeg"
        Track-Install -Component "FFmpeg" -Status "installed"
        return $true
    }

    Write-Status -Status "info" -Message "Installing FFmpeg..."

    # Priority: winget > scoop > choco
    # Try WinGet first
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        $null = & winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status -Status "ok" -Message "FFmpeg"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    # Try Scoop as second option
    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        $null = & scoop install ffmpeg 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status -Status "ok" -Message "FFmpeg"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    # Try Chocolatey as last fallback
    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        $null = & choco install ffmpeg -y 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status -Status "ok" -Message "FFmpeg"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    Write-Status -Status "fail" -Message "FFmpeg"
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
                Write-Status -Status "ok" -Message "markitai[$ExtraName]"
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
                Write-Status -Status "ok" -Message "markitai[$ExtraName]"
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
            Write-Status -Status "ok" -Message "markitai[$ExtraName]"
            return $true
        }
    } catch {}

    $ErrorActionPreference = $oldErrorAction
    Write-Status -Status "fail" -Message "markitai[$ExtraName]"
    return $false
}

# Install Claude Code CLI
function Install-ClaudeCLI {
    # Check if already installed
    $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($claudeCmd) {
        $version = & claude --version 2>&1 | Select-Object -First 1
        Write-Status -Status "ok" -Message "Claude CLI $version"
        Track-Install -Component "Claude Code CLI" -Status "installed"
        return $true
    }

    Write-Status -Status "info" -Message "Installing Claude CLI..."

    # Prefer official install script (PowerShell)
    $claudeUrl = "https://claude.ai/install.ps1"
    try {
        $null = Invoke-Expression (Invoke-RestMethod -Uri $claudeUrl) 2>&1
        # Check if installation succeeded
        $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
        if ($claudeCmd) {
            Write-Status -Status "ok" -Message "Claude CLI"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    } catch {}

    # Fallback: npm/pnpm
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        $null = & pnpm add -g @anthropic-ai/claude-code 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status -Status "ok" -Message "Claude CLI"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        $null = & npm install -g @anthropic-ai/claude-code 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status -Status "ok" -Message "Claude CLI"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    }

    Write-Status -Status "fail" -Message "Claude CLI"
    Track-Install -Component "Claude Code CLI" -Status "failed"
    return $false
}

# Install GitHub Copilot CLI
function Install-CopilotCLI {
    # Check if already installed
    $copilotCmd = Get-Command copilot -ErrorAction SilentlyContinue
    if ($copilotCmd) {
        $version = & copilot --version 2>&1 | Select-Object -First 1
        Write-Status -Status "ok" -Message "Copilot CLI $version"
        Track-Install -Component "Copilot CLI" -Status "installed"
        return $true
    }

    Write-Status -Status "info" -Message "Installing Copilot CLI..."

    # Prefer WinGet on Windows
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        $null = & winget install GitHub.Copilot --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status -Status "ok" -Message "Copilot CLI"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    # Fallback: npm/pnpm
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        $null = & pnpm add -g @github/copilot 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status -Status "ok" -Message "Copilot CLI"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        $null = & npm install -g @github/copilot 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status -Status "ok" -Message "Copilot CLI"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    Write-Status -Status "fail" -Message "Copilot CLI"
    Track-Install -Component "Copilot CLI" -Status "failed"
    return $false
}

function Initialize-Config {
    $markitaiExists = Get-Command markitai -ErrorAction SilentlyContinue
    if (-not $markitaiExists) {
        return
    }

    $configPath = Join-Path $env:USERPROFILE ".markitai\config.json"

    # Skip if config exists
    if (Test-Path $configPath) {
        return
    }

    try {
        $null = & markitai config init 2>$null
    } catch {
        # Config initialization is optional, ignore errors
    }
}

function Write-Completion {
    Write-Host ""
    Write-Host "  Get started:" -ForegroundColor White
    Write-Host "    markitai -I          Interactive mode" -ForegroundColor Cyan
    Write-Host "    markitai file.pdf   Convert a file" -ForegroundColor Cyan
    Write-Host "    markitai --help     Show all options" -ForegroundColor Cyan
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
