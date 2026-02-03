# Markitai Setup Script (User Edition)
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
    # Remote execution (irm | iex) - download lib.ps1
    try {
        $tempLib = [System.IO.Path]::GetTempFileName()
        $tempLib = [System.IO.Path]::ChangeExtension($tempLib, ".ps1")

        Invoke-RestMethod "$LIB_BASE_URL/lib.ps1" -OutFile $tempLib
        . $tempLib
        Remove-Item $tempLib -ErrorAction SilentlyContinue
    } catch {
        Write-Host "Error: Failed to download lib.ps1: $_" -ForegroundColor Red
        exit 1
    }
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
    Write-WelcomeUser

    Write-Header "Markitai Setup Wizard"

    # Step 1: Detect/install UV (required, manages Python)
    Write-Step 1 5 "Detecting UV package manager..."
    if (-not (Install-UV)) {
        Write-Summary
        exit 1
    }

    # Step 2: Detect/install Python (auto-installed via uv)
    Write-Step 2 5 "Detecting Python..."
    if (-not (Test-Python)) {
        exit 1
    }

    # Step 3: Install markitai
    Write-Step 3 5 "Installing markitai..."
    if (-not (Install-Markitai)) {
        Write-Summary
        exit 1
    }

    # Install Playwright browser (required for SPA/JS-rendered pages)
    Install-PlaywrightBrowser | Out-Null

    # Install LibreOffice (optional, for legacy Office files)
    Install-LibreOffice | Out-Null

    # Detect FFmpeg (optional, for audio/video files)
    Install-FFmpeg | Out-Null

    # Step 4: Optional - LLM CLI tools
    Write-Step 4 5 "Optional: LLM CLI tools"
    Write-Info "LLM CLI tools provide local authentication for AI providers"
    if (Ask-YesNo "Install Claude Code CLI?" $false) {
        if (Install-ClaudeCLI) {
            # Install Claude Agent SDK for programmatic access
            Install-MarkitaiExtra -ExtraName "claude-agent" | Out-Null
        }
    } else {
        Track-Install -Component "Claude Code CLI" -Status "skipped"
    }
    if (Ask-YesNo "Install GitHub Copilot CLI?" $false) {
        if (Install-CopilotCLI) {
            # Install Copilot SDK for programmatic access
            Install-MarkitaiExtra -ExtraName "copilot" | Out-Null
        }
    } else {
        Track-Install -Component "Copilot CLI" -Status "skipped"
    }

    # Initialize config
    Initialize-Config

    # Print summary
    Write-Summary

    # Complete
    Write-Completion
}

# Run main function
Main
