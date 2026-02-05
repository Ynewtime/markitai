# Markitai Setup Script (Developer Edition)
# PowerShell 5.1+
# Encoding: UTF-8 (no BOM)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ============================================================
# Library Loading / Supports both local and remote execution
# ============================================================

$LIB_BASE_URL = "https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts"

# Get script directory at script level (not inside a function to avoid scope issues)
$script:ScriptDir = $PSScriptRoot
if (-not $script:ScriptDir -and $MyInvocation.MyCommand.Path) {
    $script:ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}

# Check if running locally (script path exists)
if ($script:ScriptDir -and (Test-Path "$script:ScriptDir\lib.ps1" -ErrorAction SilentlyContinue)) {
    . "$script:ScriptDir\lib.ps1"
} else {
    # Remote execution - Developer edition requires local clone
    Write-Host ""
    Write-Host ([char]0x250C) -ForegroundColor DarkGray -NoNewline
    Write-Host "  Developer Edition requires local repository" -ForegroundColor Red
    Write-Host ([char]0x2502) -ForegroundColor DarkGray
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "  Please clone the repository first:"
    Write-Host ([char]0x2502) -ForegroundColor DarkGray
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "    git clone https://github.com/Ynewtime/markitai.git" -ForegroundColor Yellow
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "    cd markitai" -ForegroundColor Yellow
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "    .\scripts\setup-dev.ps1" -ForegroundColor Yellow
    Write-Host ([char]0x2502) -ForegroundColor DarkGray
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "  Or use the user edition for quick install:"
    Write-Host ([char]0x2502) -ForegroundColor DarkGray
    Write-Host ([char]0x2502) -ForegroundColor DarkGray -NoNewline
    Write-Host "    irm https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.ps1 | iex" -ForegroundColor Yellow
    Write-Host ([char]0x2502) -ForegroundColor DarkGray
    Write-Host ([char]0x2514) -ForegroundColor DarkGray -NoNewline
    Write-Host "  Exiting." -ForegroundColor Red
    Write-Host ""
    exit 1
}

# ============================================================
# Developer-specific Functions
# ============================================================

function Get-ProjectRoot {
    return Split-Path -Parent $ScriptDir
}

# Print installation summary (clack style)
function Print-SummaryDev {
    # Installed
    if ($script:INSTALLED_COMPONENTS.Count -gt 0) {
        $installed = $script:INSTALLED_COMPONENTS | ForEach-Object { "✓ $_" }
        Clack-Note "Installed" @installed
    }

    # Skipped
    if ($script:SKIPPED_COMPONENTS.Count -gt 0) {
        $skipped = $script:SKIPPED_COMPONENTS | ForEach-Object { "○ $_" }
        Clack-Note "Skipped" @skipped
    }

    # Failed
    if ($script:FAILED_COMPONENTS.Count -gt 0) {
        $failed = $script:FAILED_COMPONENTS | ForEach-Object { "✗ $_" }
        Clack-Note "Failed" @failed
    }

    Clack-Info "Documentation: https://markitai.ynewtime.com"
    Clack-Info "Issues: https://github.com/Ynewtime/markitai/issues"
}

# Install UV (required for developer edition)
# Returns: $true on success, $false on failure
function Install-UVDev {
    if (Test-UV) {
        $version = & uv --version 2>$null | Select-Object -First 1
        Clack-Success "$version installed"
        Track-Install -Component "uv" -Status "installed"
        return $true
    }

    Clack-Warn "uv not installed"

    if (-not (Clack-Confirm "Install uv automatically?" "n")) {
        Clack-Error "uv is required for development"
        Track-Install -Component "uv" -Status "failed"
        return $false
    }

    # Build install URL (with optional version)
    if ($script:UvVersion) {
        $uvUrl = "https://astral.sh/uv/$($script:UvVersion)/install.ps1"
        Clack-Info "Installing uv version: $($script:UvVersion)"
    } else {
        $uvUrl = "https://astral.sh/uv/install.ps1"
    }

    # Confirm remote script execution
    if (-not (Confirm-RemoteScript -ScriptUrl $uvUrl -ScriptName "UV")) {
        Clack-Error "uv is required for development"
        Track-Install -Component "uv" -Status "failed"
        return $false
    }

    Clack-Info "Installing uv..."

    try {
        Invoke-RestMethod $uvUrl | Invoke-Expression

        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")

        # Check if uv command exists after PATH refresh
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uvCmd) {
            Clack-Warn "uv installed, but PowerShell needs to be restarted"
            Clack-Info "Please restart PowerShell and run this script again"
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
            Clack-Success "$version installed"
            Track-Install -Component "uv" -Status "installed"
            return $true
        } else {
            Clack-Warn "uv installed, but PowerShell needs to be restarted"
            Clack-Info "Please restart PowerShell and run this script again"
            Track-Install -Component "uv" -Status "installed"
            return $false
        }
    } catch {
        Clack-Error "uv installation failed: $_"
        Clack-Info "Manual install: irm https://astral.sh/uv/install.ps1 | iex"
        Track-Install -Component "uv" -Status "failed"
        return $false
    }
}

# Test Python with clack-style output
# Returns: $true on success, $false on failure
function Test-PythonDev {
    # Use uv-managed Python 3.13
    if (Test-CommandExists "uv") {
        $uvPython = & uv python find 3.13 2>$null
        if ($uvPython -and (Test-Path $uvPython)) {
            $version = & $uvPython -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>$null
            if ($version) {
                $script:PYTHON_CMD = $uvPython
                Clack-Success "Python $version"
                return $true
            }
        }

        # Not found, auto-install
        Clack-Info "Installing Python 3.13..."
        $null = & uv python install 3.13 2>&1
        if ($LASTEXITCODE -eq 0) {
            $uvPython = & uv python find 3.13 2>$null
            if ($uvPython -and (Test-Path $uvPython)) {
                $version = & $uvPython -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>$null
                if ($version) {
                    $script:PYTHON_CMD = $uvPython
                    Clack-Success "Python $version installed"
                    return $true
                }
            }
        }
        Clack-Error "Python 3.13 installation failed"
    } else {
        Clack-Error "uv not installed"
    }

    return $false
}

function Sync-Dependencies {
    $projectRoot = Get-ProjectRoot

    # Build Python argument for --python (need version number, not command name)
    $pythonArg = $script:PYTHON_CMD
    if ($pythonArg -match "^py\s+-(\d+\.\d+)$") {
        # Format: py -3.13 -> 3.13
        $pythonArg = $Matches[1]
    } else {
        # Get actual version number from Python command
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
            Clack-Success "Dependencies synced"
            return $true
        } else {
            Clack-Error "Dependency sync failed"
            Clack-Log ($syncResult | Out-String)
            return $false
        }
    } finally {
        Pop-Location
    }
}

function Install-PreCommit {
    $projectRoot = Get-ProjectRoot
    Push-Location $projectRoot

    try {
        if (Test-Path ".pre-commit-config.yaml") {
            $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
            if ($uvCmd) {
                $precommitResult = & uv run pre-commit install 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Clack-Success "Pre-commit hooks installed"
                    return $true
                } else {
                    Clack-Warn "Pre-commit installation failed"
                    Clack-Info "Run manually: uv run pre-commit install"
                    return $false
                }
            } else {
                Clack-Warn "uv command not found, skipping pre-commit install"
                return $false
            }
        } else {
            Clack-Skip ".pre-commit-config.yaml not found"
            return $false
        }
    } finally {
        Pop-Location
    }
}

# Install Claude Code CLI
# Returns: $true on success, $false on failure/skip
function Install-ClaudeCodeDev {
    # Check if already installed
    $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($claudeCmd) {
        $version = & claude --version 2>&1 | Select-Object -First 1
        Clack-Success "Claude Code CLI $version"
        Track-Install -Component "Claude Code CLI" -Status "installed"
        return $true
    }

    Clack-Info "Installing Claude Code CLI..."

    # Try npm/pnpm
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        Clack-Info "Installing via pnpm..."
        $null = & pnpm add -g @anthropic-ai/claude-code 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "Claude Code CLI installed"
            Clack-Info "Run 'claude /login' to authenticate"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        Clack-Info "Installing via npm..."
        $null = & npm install -g @anthropic-ai/claude-code 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "Claude Code CLI installed"
            Clack-Info "Run 'claude /login' to authenticate"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    }

    # Try WinGet (Windows)
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Clack-Info "Installing via WinGet..."
        $null = & winget install Anthropic.ClaudeCode --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "Claude Code CLI installed"
            Clack-Info "Run 'claude /login' to authenticate"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    }

    Clack-Error "Claude Code CLI installation failed"
    Clack-Note "Manual install options" `
        "pnpm: pnpm add -g @anthropic-ai/claude-code" `
        "winget: winget install Anthropic.ClaudeCode" `
        "Docs: https://code.claude.com/docs/en/setup"
    Track-Install -Component "Claude Code CLI" -Status "failed"
    return $false
}

# Install GitHub Copilot CLI
# Returns: $true on success, $false on failure/skip
function Install-CopilotCLIDev {
    # Check if already installed
    $copilotCmd = Get-Command copilot -ErrorAction SilentlyContinue
    if ($copilotCmd) {
        $version = & copilot --version 2>&1 | Select-Object -First 1
        Clack-Success "Copilot CLI $version"
        Track-Install -Component "Copilot CLI" -Status "installed"
        return $true
    }

    Clack-Info "Installing Copilot CLI..."

    # Try npm/pnpm
    $pnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue

    if ($pnpmCmd) {
        Clack-Info "Installing via pnpm..."
        $null = & pnpm add -g @github/copilot 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "Copilot CLI installed"
            Clack-Info "Run 'copilot /login' to authenticate"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        Clack-Info "Installing via npm..."
        $null = & npm install -g @github/copilot 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "Copilot CLI installed"
            Clack-Info "Run 'copilot /login' to authenticate"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    # Try WinGet (Windows)
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Clack-Info "Installing via WinGet..."
        $null = & winget install GitHub.Copilot --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "Copilot CLI installed"
            Clack-Info "Run 'copilot /login' to authenticate"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    Clack-Error "Copilot CLI installation failed"
    Clack-Note "Manual install options" `
        "pnpm: pnpm add -g @github/copilot" `
        "winget: winget install GitHub.Copilot"
    Track-Install -Component "Copilot CLI" -Status "failed"
    return $false
}

# Install Playwright browser (Chromium) for development
# Uses uv run (preferred) with fallback to python module
# Returns: $true on success, $false on failure/skip
function Install-PlaywrightBrowserDev {
    # Auto-detect if Playwright is already installed
    $projectRoot = Get-ProjectRoot
    Push-Location $projectRoot

    try {
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if ($uvCmd) {
            # Check if chromium is already installed by testing if playwright can find it
            $checkResult = & uv run python -c "from playwright.sync_api import sync_playwright; p = sync_playwright().start(); p.chromium.executable_path; p.stop()" 2>&1
            if ($LASTEXITCODE -eq 0) {
                Clack-Success "Playwright browser (Chromium)"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        }
    } catch {}
    finally {
        Pop-Location
    }

    # Ask user consent before downloading
    if (-not (Clack-Confirm "Install Playwright browser (Chromium)?" "y")) {
        Clack-Skip "Playwright browser"
        Track-Install -Component "Playwright Browser" -Status "skipped"
        return $false
    }

    Clack-Info "Downloading Chromium browser..."

    Push-Location $projectRoot

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        # Prefer uv run in dev environment
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if ($uvCmd) {
            # Show download progress (Chromium is ~200MB)
            & uv run playwright install chromium
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Clack-Success "Chromium browser installed"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        }

        # Fallback to Python module
        if ($script:PYTHON_CMD) {
            $cmdParts = $script:PYTHON_CMD -split " "
            $exe = $cmdParts[0]
            $baseArgs = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] } else { @() }
            $pwArgs = $baseArgs + @("-m", "playwright", "install", "chromium")

            # Show download progress
            & $exe @pwArgs
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Clack-Success "Chromium browser installed"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        }
    } finally {
        Pop-Location
        $ErrorActionPreference = $oldErrorAction
    }

    Clack-Error "Playwright browser installation failed"
    Clack-Info "You can install later with: uv run playwright install chromium"
    Track-Install -Component "Playwright Browser" -Status "failed"
    return $false
}

# Detect LibreOffice installation (for legacy Office files)
# Returns: $true if installed, $false otherwise
function Install-LibreOfficeDev {
    Clack-Info "LibreOffice (optional):"
    Clack-Info "  Purpose: Convert legacy Office files (.doc, .ppt, .xls)"

    # Auto-detect LibreOffice
    $soffice = Get-Command soffice -ErrorAction SilentlyContinue
    if ($soffice) {
        try {
            $version = & soffice --version 2>&1 | Select-Object -First 1
            Clack-Success "LibreOffice $version"
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
            Clack-Success "LibreOffice found"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    # Not installed - ask user
    if (-not (Clack-Confirm "Install LibreOffice (for .doc/.ppt/.xls files)?" "n")) {
        Clack-Skip "LibreOffice"
        Track-Install -Component "LibreOffice" -Status "skipped"
        return $false
    }

    Clack-Info "Installing LibreOffice..."

    # Priority: winget > scoop > choco
    # Try WinGet first
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Clack-Info "Installing via WinGet..."
        $null = & winget install TheDocumentFoundation.LibreOffice --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "LibreOffice installed"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    # Try Scoop as second option
    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        Clack-Info "Installing via Scoop..."
        & scoop bucket add extras 2>$null
        $null = & scoop install extras/libreoffice 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "LibreOffice installed"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    # Try Chocolatey as last fallback
    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        Clack-Info "Installing via Chocolatey..."
        $null = & choco install libreoffice-fresh -y 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "LibreOffice installed"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    Clack-Error "LibreOffice installation failed"
    Clack-Note "Manual install options" `
        "winget: winget install TheDocumentFoundation.LibreOffice" `
        "scoop: scoop install extras/libreoffice" `
        "choco: choco install libreoffice-fresh" `
        "Download: https://www.libreoffice.org/download/"
    Track-Install -Component "LibreOffice" -Status "failed"
    return $false
}

# Install FFmpeg (optional, for audio/video file processing)
# Returns: $true if installed, $false otherwise
function Install-FFmpegDev {
    Clack-Info "FFmpeg (optional):"
    Clack-Info "  Purpose: Process audio/video files (.mp3, .mp4, .wav, etc.)"

    # Auto-detect FFmpeg
    $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpegCmd) {
        try {
            $version = & ffmpeg -version 2>&1 | Select-Object -First 1
            if ($version -match "ffmpeg version ([^\s]+)") {
                Clack-Success "FFmpeg $($Matches[1])"
            } else {
                Clack-Success "FFmpeg installed"
            }
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        } catch {}
    }

    # Not installed - ask user
    if (-not (Clack-Confirm "Install FFmpeg (for audio/video files)?" "n")) {
        Clack-Skip "FFmpeg"
        Track-Install -Component "FFmpeg" -Status "skipped"
        return $false
    }

    Clack-Info "Installing FFmpeg..."

    # Priority: winget > scoop > choco
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Clack-Info "Installing via WinGet..."
        $null = & winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "FFmpeg installed"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        Clack-Info "Installing via Scoop..."
        $null = & scoop install ffmpeg 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "FFmpeg installed"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        Clack-Info "Installing via Chocolatey..."
        $null = & choco install ffmpeg -y 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "FFmpeg installed"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    Clack-Error "FFmpeg installation failed"
    Clack-Note "Manual install options" `
        "winget: winget install Gyan.FFmpeg" `
        "scoop: scoop install ffmpeg" `
        "choco: choco install ffmpeg" `
        "Download: https://ffmpeg.org/download.html"
    Track-Install -Component "FFmpeg" -Status "failed"
    return $false
}

# ============================================================
# Main Logic
# ============================================================

function Main {
    # Check execution policy
    if (-not (Test-ExecutionPolicy)) {
        # Continue anyway, but warn was shown
    }

    # Security check: warn if running as Administrator
    Test-AdminWarning

    # Environment check: warn if running in WSL
    Test-WSLWarning

    # Welcome message with clack-style intro
    Clack-Intro "Markitai Development Environment Setup"

    # ============================================================
    # Section 1: Prerequisites
    # ============================================================
    Clack-Section "Checking prerequisites"

    # Step 1: Detect/install UV (required, manages Python)
    if (-not (Install-UVDev)) {
        Clack-Cancel "Setup failed: uv is required"
        exit 1
    }

    # Step 2: Detect/install Python (auto-installed via uv)
    if (-not (Test-PythonDev)) {
        Clack-Cancel "Setup failed: Python is required"
        exit 1
    }

    # ============================================================
    # Section 2: Development Environment
    # ============================================================
    Clack-Section "Setting up development environment"

    # Step 3: Sync dependencies (includes all extras: browser, claude-agent, copilot)
    if (-not (Sync-Dependencies)) {
        Clack-Cancel "Setup failed: dependency sync failed"
        exit 1
    }
    Track-Install -Component "Python dependencies" -Status "installed"
    Track-Install -Component "Claude Agent SDK" -Status "installed"
    Track-Install -Component "Copilot SDK" -Status "installed"

    # Step 4: Install pre-commit
    Install-PreCommit | Out-Null
    Track-Install -Component "pre-commit hooks" -Status "installed"

    # ============================================================
    # Section 3: Optional Components (with auto-detection)
    # ============================================================
    Clack-Section "Optional components"

    # Install Playwright browser (required for SPA/JS-rendered pages)
    Install-PlaywrightBrowserDev | Out-Null

    # Install LibreOffice (optional, for legacy Office files)
    Install-LibreOfficeDev | Out-Null

    # Install FFmpeg (optional, for audio/video files)
    Install-FFmpegDev | Out-Null

    # ============================================================
    # Section 4: LLM CLI Tools
    # ============================================================
    Clack-Section "LLM CLI tools"

    # Auto-detect Claude Code CLI
    $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($claudeCmd) {
        $version = & claude --version 2>&1 | Select-Object -First 1
        Clack-Success "Claude Code CLI $version"
        Track-Install -Component "Claude Code CLI" -Status "installed"
    } else {
        if (Clack-Confirm "Install Claude Code CLI?" "n") {
            Install-ClaudeCodeDev | Out-Null
        } else {
            Clack-Skip "Claude Code CLI"
            Track-Install -Component "Claude Code CLI" -Status "skipped"
        }
    }

    # Auto-detect Copilot CLI
    $copilotCmd = Get-Command copilot -ErrorAction SilentlyContinue
    if ($copilotCmd) {
        $version = & copilot --version 2>&1 | Select-Object -First 1
        Clack-Success "Copilot CLI $version"
        Track-Install -Component "Copilot CLI" -Status "installed"
    } else {
        if (Clack-Confirm "Install GitHub Copilot CLI?" "n") {
            Install-CopilotCLIDev | Out-Null
        } else {
            Clack-Skip "Copilot CLI"
            Track-Install -Component "Copilot CLI" -Status "skipped"
        }
    }

    # ============================================================
    # Completion
    # ============================================================
    $projectRoot = Get-ProjectRoot

    # Print installation summary
    Print-SummaryDev

    Clack-Note "Getting started" `
        "Activate venv:" `
        "  $projectRoot\.venv\Scripts\Activate.ps1" `
        "" `
        "Run tests:" `
        "  uv run pytest" `
        "" `
        "Run CLI:" `
        "  uv run markitai --help"

    Clack-Outro "Development environment ready!"
}

# Run main function
Main
