# Markitai Setup Script (Developer Edition)
# PowerShell 5.1+

$ErrorActionPreference = "Stop"

# Helper functions
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

# Get project root directory
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

# Detect Python (requires 3.11-3.13, 3.14+ not supported)
function Test-Python {
    Write-Step 1 5 "Detecting Python..."

    # Prefer 3.11-3.13 versions
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

            # Check version range: 3.11 <= version < 3.14
            if ($major -eq 3 -and $minor -ge 11 -and $minor -le 13) {
                $script:PYTHON_CMD = $cmd
                Write-Success "Python $version installed ($cmd)"
                return $true
            } elseif ($major -eq 3 -and $minor -ge 14) {
                Write-Warning2 "Python $version detected, but onnxruntime doesn't support Python 3.14+"
            }
        } catch {
            continue
        }
    }

    Write-Error2 "Python 3.11-3.13 not found"
    Write-Host ""
    Write-Warning2 "Please install Python 3.13 (recommended) or 3.11/3.12:"
    Write-Info "Download: https://www.python.org/downloads/"
    Write-Info "scoop: scoop install python@3.13"
    Write-Info "pim: winget install 9NQ7512CXL7T"
    Write-Info "Note: onnxruntime doesn't support Python 3.14 yet"
    return $false
}

# Detect/install UV
function Test-UV {
    Write-Step 2 5 "Detecting UV package manager..."

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

    Write-Error2 "UV not installed"

    if (Ask-YesNo "Install UV automatically?" $true) {
        Write-Info "Installing UV..."

        try {
            Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression

            # Refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

            $oldErrorAction = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                $version = & uv --version 2>&1 | Select-Object -First 1
            } finally {
                $ErrorActionPreference = $oldErrorAction
            }
            if ($version -and $version -notmatch "error") {
                Write-Success "$version installed successfully"
                return $true
            } else {
                Write-Warning2 "UV installed, but PowerShell needs to be restarted"
                Write-Info "Please restart PowerShell and run this script again"
                return $false
            }
        } catch {
            Write-Error2 "UV installation failed: $_"
            Write-Info "Manual install: irm https://astral.sh/uv/install.ps1 | iex"
            return $false
        }
    } else {
        Write-Error2 "UV is required for development"
        return $false
    }
}

# Sync development dependencies
function Sync-Dependencies {
    Write-Step 3 5 "Syncing development dependencies..."

    $projectRoot = Get-ProjectRoot
    Write-Info "Project directory: $projectRoot"

    Push-Location $projectRoot

    try {
        Write-Info "Running uv sync --all-extras..."
        & uv sync --all-extras
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Dependencies synced successfully"
            return $true
        } else {
            Write-Error2 "Dependency sync failed"
            return $false
        }
    } finally {
        Pop-Location
    }
}

# Install pre-commit hooks
function Install-PreCommit {
    Write-Step 4 5 "Configuring pre-commit..."

    $projectRoot = Get-ProjectRoot
    Push-Location $projectRoot

    try {
        if (Test-Path ".pre-commit-config.yaml") {
            Write-Info "Installing pre-commit hooks..."

            & uv run pre-commit install
            if ($LASTEXITCODE -eq 0) {
                Write-Success "pre-commit hooks installed successfully"
            } else {
                Write-Warning2 "pre-commit installation failed, please run manually: uv run pre-commit install"
            }
        } else {
            Write-Info ".pre-commit-config.yaml not found, skipping"
        }
    } finally {
        Pop-Location
    }
}

# Detect Node.js
function Test-NodeJS {
    Write-Info "Detecting Node.js..."

    try {
        $version = & node --version 2>$null
        if ($version) {
            $major = [int]($version -replace "v", "" -split "\.")[0]

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
    Write-Info "Chocolatey: choco install nodejs-lts"
    Write-Info "fnm: winget install Schniz.fnm"
    return $false
}

# Install agent-browser
function Install-AgentBrowser {
    if (-not (Test-NodeJS)) {
        Write-Warning2 "Skipping agent-browser installation (requires Node.js)"
        return $false
    }

    Write-Info "Installing agent-browser..."

    try {
        & npm install -g agent-browser
        if ($LASTEXITCODE -eq 0) {
            Write-Success "agent-browser installed successfully"

            if (Ask-YesNo "Download Chromium browser?" $true) {
                Write-Info "Downloading Chromium..."
                & agent-browser install
                Write-Success "Chromium download complete"
            }

            return $true
        }
    } catch {
        Write-Error2 "agent-browser installation failed: $_"
    }

    Write-Info "Manual install: npm install -g agent-browser"
    return $false
}

# Optional components
function Install-Optional {
    Write-Step 5 5 "Optional components"

    if (Ask-YesNo "Install browser automation support (agent-browser)?" $false) {
        Install-AgentBrowser | Out-Null
    } else {
        Write-Info "Skipping agent-browser installation"
    }
}

# Print completion message
function Write-Completion {
    $projectRoot = Get-ProjectRoot

    Write-Host ""
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host "Development environment setup complete!" -ForegroundColor White
    Write-Host ""
    Write-Host "  Activate virtual environment:" -ForegroundColor White
    Write-Host "    $projectRoot\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Run tests:" -ForegroundColor White
    Write-Host "    uv run pytest" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Run CLI:" -ForegroundColor White
    Write-Host "    uv run markitai --help" -ForegroundColor Yellow
    Write-Host ""
}

# Main function
function Main {
    Write-Header "Markitai Dev Environment Setup"

    # Detect Python
    if (-not (Test-Python)) {
        exit 1
    }

    # Detect/install UV
    if (-not (Test-UV)) {
        exit 1
    }

    # Sync dependencies
    if (-not (Sync-Dependencies)) {
        exit 1
    }

    # Install pre-commit
    Install-PreCommit

    # Optional components
    Install-Optional

    # Complete
    Write-Completion
}

# Run main function
Main
