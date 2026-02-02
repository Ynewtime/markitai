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
    Write-Host ("=" * 48) -ForegroundColor Cyan
    Write-Host "  Developer Edition requires local repository" -ForegroundColor White
    Write-Host ("=" * 48) -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Please clone the repository first:"
    Write-Host ""
    Write-Host "    git clone https://github.com/Ynewtime/markitai.git" -ForegroundColor Yellow
    Write-Host "    cd markitai" -ForegroundColor Yellow
    Write-Host "    .\scripts\setup-dev.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Or use the user edition for quick install:"
    Write-Host ""
    Write-Host "    irm https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.ps1 | iex" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# ============================================================
# Developer-specific Functions
# ============================================================

function Get-ProjectRoot {
    return Split-Path -Parent $ScriptDir
}

# Install UV (required for developer edition)
function Install-UVDev {
    Write-Info "Checking UV installation..."

    if (Test-UV) {
        Track-Install -Component "uv" -Status "installed"
        return $true
    }

    Write-Error2 "UV not installed"

    if (-not (Ask-YesNo "Install UV automatically?" $false)) {
        Write-Error2 "UV is required for development"
        Track-Install -Component "uv" -Status "failed"
        return $false
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
        Write-Error2 "UV is required for development"
        Track-Install -Component "uv" -Status "failed"
        return $false
    }

    Write-Info "Installing UV..."

    try {
        Invoke-RestMethod $uvUrl | Invoke-Expression

        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")

        # Check if uv command exists after PATH refresh
        $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        if (-not $uvCmd) {
            Write-Warning2 "UV installed, but PowerShell needs to be restarted"
            Write-Info "Please restart PowerShell and run this script again"
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
            Write-Success "$version installed successfully"
            Track-Install -Component "uv" -Status "installed"
            return $true
        } else {
            Write-Warning2 "UV installed, but PowerShell needs to be restarted"
            Write-Info "Please restart PowerShell and run this script again"
            Track-Install -Component "uv" -Status "installed"
            return $false
        }
    } catch {
        Write-Error2 "UV installation failed: $_"
        Write-Info "Manual install: irm https://astral.sh/uv/install.ps1 | iex"
        Track-Install -Component "uv" -Status "failed"
        return $false
    }
}

function Sync-Dependencies {
    $projectRoot = Get-ProjectRoot
    Write-Info "Project directory: $projectRoot"

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
        Write-Info "Running uv sync --all-extras --python $pythonArg..."
        & uv sync --all-extras --python $pythonArg
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Dependencies synced successfully (using Python $pythonArg)"
            return $true
        } else {
            Write-Error2 "Dependency sync failed"
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
            Write-Info "Installing pre-commit hooks..."

            $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
            if ($uvCmd) {
                & uv run pre-commit install
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "pre-commit hooks installed successfully"
                } else {
                    Write-Warning2 "pre-commit installation failed, please run manually: uv run pre-commit install"
                }
            } else {
                Write-Warning2 "uv command not found, skipping pre-commit install"
            }
        } else {
            Write-Info ".pre-commit-config.yaml not found, skipping"
        }
    } finally {
        Pop-Location
    }
}

# Install Claude Code CLI
function Install-ClaudeCLI {
    Write-Info "Installing Claude Code CLI..."

    # Check if already installed
    $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($claudeCmd) {
        $version = & claude --version 2>&1 | Select-Object -First 1
        Write-Success "Claude Code CLI already installed: $version"
        Track-Install -Component "Claude Code CLI" -Status "installed"
        return $true
    }

    # Try npm/pnpm
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

    # Try WinGet (Windows)
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Write-Info "Installing via WinGet..."
        & winget install Anthropic.ClaudeCode
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Claude Code CLI installed via WinGet"
            Write-Info "Run 'claude /login' to authenticate with your Claude subscription or API key"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    }

    Write-Warning2 "Claude Code CLI installation failed"
    Write-Info "Manual install options:"
    Write-Info "  pnpm: pnpm add -g @anthropic-ai/claude-code"
    Write-Info "  winget: winget install Anthropic.ClaudeCode"
    Write-Info "  Docs: https://code.claude.com/docs/en/setup"
    Track-Install -Component "Claude Code CLI" -Status "failed"
    return $false
}

# Install GitHub Copilot CLI
function Install-CopilotCLI {
    Write-Info "Installing GitHub Copilot CLI..."

    # Check if already installed
    $copilotCmd = Get-Command copilot -ErrorAction SilentlyContinue
    if ($copilotCmd) {
        $version = & copilot --version 2>&1 | Select-Object -First 1
        Write-Success "Copilot CLI already installed: $version"
        Track-Install -Component "Copilot CLI" -Status "installed"
        return $true
    }

    # Try npm/pnpm
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

    # Try WinGet (Windows)
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Write-Info "Installing via WinGet..."
        & winget install GitHub.Copilot
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Copilot CLI installed via WinGet"
            Write-Info "Run 'copilot /login' to authenticate with your GitHub Copilot subscription"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    Write-Warning2 "Copilot CLI installation failed"
    Write-Info "Manual install options:"
    Write-Info "  pnpm: pnpm add -g @github/copilot"
    Write-Info "  winget: winget install GitHub.Copilot"
    Track-Install -Component "Copilot CLI" -Status "failed"
    return $false
}

# Install LLM CLI tools
function Install-LLMCLIs {
    Write-Info "LLM CLI tools provide local authentication for AI providers:"
    Write-Info "  - Claude Code CLI: Use your Claude subscription"
    Write-Info "  - Copilot CLI: Use your GitHub Copilot subscription"

    if (Ask-YesNo "Install Claude Code CLI?" $false) {
        Install-ClaudeCLI | Out-Null
    } else {
        Track-Install -Component "Claude Code CLI" -Status "skipped"
    }

    if (Ask-YesNo "Install GitHub Copilot CLI?" $false) {
        Install-CopilotCLI | Out-Null
    } else {
        Track-Install -Component "Copilot CLI" -Status "skipped"
    }
}

# Install Playwright browser (Chromium) for development
# Uses uv run (preferred) with fallback to python module
# Returns: $true on success, $false on failure/skip
function Install-PlaywrightBrowserDev {
    Write-Info "Playwright browser (Chromium):"
    Write-Info "  Purpose: Browser automation for JavaScript-rendered pages (Twitter, SPAs)"

    # Ask user consent before downloading
    if (-not (Ask-YesNo "Download Chromium browser?" $true)) {
        Write-Info "Skipping Playwright browser installation"
        Track-Install -Component "Playwright Browser" -Status "skipped"
        return $false
    }

    Write-Info "Downloading Chromium browser..."

    $projectRoot = Get-ProjectRoot
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
                Write-Success "Chromium browser installed successfully"
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
                Write-Success "Chromium browser installed successfully"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        }
    } finally {
        Pop-Location
        $ErrorActionPreference = $oldErrorAction
    }

    Write-Warning2 "Playwright browser installation failed"
    Write-Info "You can install later with: uv run playwright install chromium"
    Track-Install -Component "Playwright Browser" -Status "failed"
    return $false
}

# Detect LibreOffice installation (for legacy Office files)
function Install-LibreOfficeDev {
    Write-Info "Checking LibreOffice installation..."
    Write-Info "  Purpose: Convert legacy Office files (.doc, .ppt, .xls)"

    $soffice = Get-Command soffice -ErrorAction SilentlyContinue
    if ($soffice) {
        try {
            $version = & soffice --version 2>&1 | Select-Object -First 1
            Write-Success "LibreOffice installed: $version"
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

# Install FFmpeg (optional, for audio/video file processing)
function Install-FFmpegDev {
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

function Write-CompletionDev {
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

    # Welcome message
    Write-WelcomeDev

    Write-Header "Markitai Dev Environment Setup"

    # Step 1: Detect Python
    Write-Step 1 5 "Detecting Python..."
    if (-not (Test-Python)) {
        exit 1
    }

    # Step 2: Detect/install UV (required for developer edition)
    Write-Step 2 5 "Detecting UV package manager..."
    if (-not (Install-UVDev)) {
        Write-Summary
        exit 1
    }

    # Step 3: Sync dependencies (includes all extras: browser, claude-agent, copilot)
    Write-Step 3 5 "Syncing development dependencies..."
    if (-not (Sync-Dependencies)) {
        Write-Summary
        exit 1
    }
    Track-Install -Component "Python dependencies" -Status "installed"
    Track-Install -Component "Claude Agent SDK" -Status "installed"
    Track-Install -Component "Copilot SDK" -Status "installed"

    # Install Playwright browser (required for SPA/JS-rendered pages)
    Install-PlaywrightBrowserDev | Out-Null

    # Install LibreOffice (optional, for legacy Office files)
    Install-LibreOfficeDev | Out-Null

    # Install FFmpeg (optional, for audio/video files)
    Install-FFmpegDev | Out-Null

    # Step 4: Install pre-commit
    Write-Step 4 5 "Configuring pre-commit..."
    Install-PreCommit
    Track-Install -Component "pre-commit hooks" -Status "installed"

    # Step 5: Optional - LLM CLI tools
    Write-Step 5 5 "Optional: LLM CLI tools"
    if (Ask-YesNo "Install LLM CLI tools (Claude Code / Copilot)?" $false) {
        Install-LLMCLIs
    } else {
        Write-Info "Skipping LLM CLI installation"
        Track-Install -Component "Claude Code CLI" -Status "skipped"
        Track-Install -Component "Copilot CLI" -Status "skipped"
    }

    # Print summary
    Write-Summary

    # Complete
    Write-CompletionDev
}

# Run main function
Main
