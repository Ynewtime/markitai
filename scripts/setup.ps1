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
    # Security checks
    if (-not (Test-ExecutionPolicy)) {}
    Test-AdminWarning
    Test-WSLWarning

    # Header
    Write-Header "Markitai Setup"

    # Core installation
    if (-not (Install-UV)) { exit 1 }
    if (-not (Test-Python)) { exit 1 }
    if (-not (Install-Markitai)) { exit 1 }

    # Optional components
    Write-Host ""
    Write-Host "  Optional components:" -ForegroundColor White

    if (Ask-YesNo "Playwright browser (for JS-rendered pages)?" $true) {
        Install-PlaywrightBrowser | Out-Null
    } else {
        Write-Status -Status "skip" -Message "Playwright browser"
        Track-Install -Component "Playwright Browser" -Status "skipped"
    }

    if (Ask-YesNo "LibreOffice (for .doc/.xls/.ppt)?" $false) {
        Install-LibreOffice | Out-Null
    } else {
        Write-Status -Status "skip" -Message "LibreOffice"
        Track-Install -Component "LibreOffice" -Status "skipped"
    }

    if (Ask-YesNo "FFmpeg (for audio/video)?" $false) {
        Install-FFmpeg | Out-Null
    } else {
        Write-Status -Status "skip" -Message "FFmpeg"
        Track-Install -Component "FFmpeg" -Status "skipped"
    }

    if (Ask-YesNo "Claude Code CLI?" $false) {
        if (Install-ClaudeCLI) {
            Install-MarkitaiExtra -ExtraName "claude-agent" | Out-Null
        }
    } else {
        Write-Status -Status "skip" -Message "Claude CLI"
        Track-Install -Component "Claude Code CLI" -Status "skipped"
    }

    if (Ask-YesNo "GitHub Copilot CLI?" $false) {
        if (Install-CopilotCLI) {
            Install-MarkitaiExtra -ExtraName "copilot" | Out-Null
        }
    } else {
        Write-Status -Status "skip" -Message "Copilot CLI"
        Track-Install -Component "Copilot CLI" -Status "skipped"
    }

    # Config
    Initialize-Config 2>$null | Out-Null

    # Summary and completion
    Write-Summary
    Write-Completion
}

# Run main function
Main
