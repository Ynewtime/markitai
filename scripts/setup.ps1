# Markitai Setup Script (User Edition)
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

# Detect Python (requires 3.11-3.13, 3.14+ not supported)
function Test-Python {
    Write-Step 1 4 "Detecting Python..."

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
    Write-Step 2 4 "Detecting UV package manager..."

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
        Write-Info "Skipping UV installation"
        Write-Warning2 "markitai recommends using UV for installation"
        return $false
    }
}

# Install markitai
function Install-Markitai {
    Write-Step 3 4 "Installing markitai..."

    Write-Info "Installing..."

    # Prefer uv tool install (recommended)
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
            # Refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
            $version = & markitai --version 2>&1 | Select-Object -First 1
            if (-not $version) { $version = "installed" }
            Write-Success "markitai $version installed successfully"
            return $true
        }
    }

    # Fallback to pipx
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
            if (-not $version) { $version = "installed" }
            Write-Success "markitai $version installed successfully"
            return $true
        }
    }

    # Fallback to pip --user
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
        if (-not $version) { $version = "installed" }
        Write-Success "markitai $version installed successfully"
        Write-Warning2 "You may need to add Python Scripts directory to PATH"
        return $true
    }

    Write-Error2 "markitai installation failed"
    Write-Info "Manual install: uv tool install markitai"
    return $false
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
    Write-Step 4 4 "Optional components"

    if (Ask-YesNo "Install browser automation support (agent-browser)?" $false) {
        Install-AgentBrowser | Out-Null
    } else {
        Write-Info "Skipping agent-browser installation"
    }
}

# Initialize config
function Initialize-Config {
    Write-Info "Initializing configuration..."

    try {
        $markitaiExists = Get-Command markitai -ErrorAction SilentlyContinue
        if ($markitaiExists) {
            & markitai config init 2>$null
            Write-Success "Configuration initialized"
        }
    } catch {}
}

# Print completion message
function Write-Completion {
    Write-Host ""
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host "Setup complete!" -ForegroundColor White
    Write-Host ""
    Write-Host "  Get started:" -ForegroundColor White
    Write-Host "    markitai --help" -ForegroundColor Yellow
    Write-Host ""
}

# Main function
function Main {
    Write-Header "Markitai Setup Wizard"

    # Detect Python
    if (-not (Test-Python)) {
        exit 1
    }

    # Detect/install UV
    Test-UV | Out-Null

    # Install markitai
    if (-not (Install-Markitai)) {
        exit 1
    }

    # Optional components
    Install-Optional

    # Initialize config
    Initialize-Config

    # Complete
    Write-Completion
}

# Run main function
Main
