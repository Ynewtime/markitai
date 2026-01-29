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
if (-not $script:ScriptDir) {
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

# Helper function to run agent-browser command (Dev version)
# Works around npm shim issues on Windows (both .ps1 and .cmd reference /bin/sh)
# See: https://github.com/vercel-labs/agent-browser/issues/262
function Invoke-AgentBrowserDev {
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

function Install-AgentBrowserDev {
    if (-not (Test-NodeJS)) {
        Write-Warning2 "Skipping agent-browser installation (requires Node.js)"
        Track-Install -Component "agent-browser" -Status "skipped"
        return $false
    }

    Write-Info "Installing agent-browser..."

    # Build package spec with optional version
    if ($script:AgentBrowserVersion) {
        $pkg = "agent-browser@$($script:AgentBrowserVersion)"
        Write-Info "Installing version: $($script:AgentBrowserVersion)"
    } else {
        $pkg = "agent-browser"
    }

    # Try npm first, then pnpm
    $installSuccess = $false
    $npmExists = Get-Command npm -ErrorAction SilentlyContinue
    $pnpmExists = Get-Command pnpm -ErrorAction SilentlyContinue

    if ($npmExists) {
        Write-Info "Installing via npm..."
        try {
            & npm install -g $pkg
            if ($LASTEXITCODE -eq 0) {
                $installSuccess = $true
            }
        } catch {
            Write-Warning2 "npm installation failed, trying pnpm..."
        }
    }

    if (-not $installSuccess -and $pnpmExists) {
        Write-Info "Installing via pnpm..."
        try {
            & pnpm add -g $pkg
            if ($LASTEXITCODE -eq 0) {
                $installSuccess = $true
            }
        } catch {
            Write-Error2 "pnpm installation failed: $_"
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
            Write-Warning2 "agent-browser installed but not in PATH"
            Write-Info "You may need to add global bin to PATH:"
            Write-Info "  pnpm bin -g  # or: npm config get prefix"
            Track-Install -Component "agent-browser" -Status "installed"
            return $false
        }

        Write-Success "agent-browser installed successfully"
        Track-Install -Component "agent-browser" -Status "installed"

        # Chromium download (default: No)
        if (Ask-YesNo "Download Chromium browser?" $false) {
            Write-Info "Downloading Chromium..."
            $null = Invoke-AgentBrowserDev -Arguments @("install")
            Write-Success "Chromium download complete"
            Track-Install -Component "Chromium" -Status "installed"
        } else {
            Write-Info "Skipping Chromium download"
            Write-Info "You can install later: agent-browser install"
            Track-Install -Component "Chromium" -Status "skipped"
        }

        return $true
    }

    Write-Info "Manual install: npm install -g agent-browser"
    Track-Install -Component "agent-browser" -Status "failed"
    return $false
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

# Install LLM provider SDKs
function Install-ProviderSDKs {
    $projectRoot = Get-ProjectRoot
    Push-Location $projectRoot

    try {
        Write-Info "Python SDKs for programmatic LLM access:"
        Write-Info "  - Claude Agent SDK (requires Claude Code CLI)"
        Write-Info "  - GitHub Copilot SDK (requires Copilot CLI)"

        if (Ask-YesNo "Install Claude Agent SDK?" $false) {
            Write-Info "Installing claude-agent-sdk..."
            & uv sync --extra claude-agent
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Claude Agent SDK installed"
                Track-Install -Component "Claude Agent SDK" -Status "installed"
            } else {
                Write-Warning2 "Claude Agent SDK installation failed"
                Track-Install -Component "Claude Agent SDK" -Status "failed"
            }
        } else {
            Track-Install -Component "Claude Agent SDK" -Status "skipped"
        }

        if (Ask-YesNo "Install GitHub Copilot SDK?" $false) {
            Write-Info "Installing github-copilot-sdk..."
            & uv sync --extra copilot
            if ($LASTEXITCODE -eq 0) {
                Write-Success "GitHub Copilot SDK installed"
                Track-Install -Component "Copilot SDK" -Status "installed"
            } else {
                Write-Warning2 "GitHub Copilot SDK installation failed"
                Track-Install -Component "Copilot SDK" -Status "failed"
            }
        } else {
            Track-Install -Component "Copilot SDK" -Status "skipped"
        }
    } finally {
        Pop-Location
    }
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
    Write-Step 1 7 "Detecting Python..."
    if (-not (Test-Python)) {
        exit 1
    }

    # Step 2: Detect/install UV (required for developer edition)
    Write-Step 2 7 "Detecting UV package manager..."
    if (-not (Install-UVDev)) {
        Write-Summary
        exit 1
    }

    # Step 3: Sync dependencies
    Write-Step 3 7 "Syncing development dependencies..."
    if (-not (Sync-Dependencies)) {
        Write-Summary
        exit 1
    }
    Track-Install -Component "Python dependencies" -Status "installed"

    # Step 4: Install pre-commit
    Write-Step 4 7 "Configuring pre-commit..."
    Install-PreCommit
    Track-Install -Component "pre-commit hooks" -Status "installed"

    # Step 5: Optional - agent-browser
    Write-Step 5 7 "Optional: Browser automation"
    if (Ask-YesNo "Install browser automation support (agent-browser)?" $false) {
        Install-AgentBrowserDev | Out-Null
    } else {
        Write-Info "Skipping agent-browser installation"
        Track-Install -Component "agent-browser" -Status "skipped"
    }

    # Step 6: Optional - LLM CLI tools
    Write-Step 6 7 "Optional: LLM CLI tools"
    if (Ask-YesNo "Install LLM CLI tools (Claude Code / Copilot)?" $false) {
        Install-LLMCLIs
    } else {
        Write-Info "Skipping LLM CLI installation"
        Track-Install -Component "Claude Code CLI" -Status "skipped"
        Track-Install -Component "Copilot CLI" -Status "skipped"
    }

    # Step 7: Optional - LLM Python SDKs
    Write-Step 7 7 "Optional: LLM Python SDKs"
    if (Ask-YesNo "Install LLM Python SDKs (claude-agent-sdk / github-copilot-sdk)?" $false) {
        Install-ProviderSDKs
    } else {
        Write-Info "Skipping LLM Python SDK installation"
        Write-Info "Install later: uv sync --all-extras"
    }

    # Print summary
    Write-Summary

    # Complete
    Write-CompletionDev
}

# Run main function
Main
