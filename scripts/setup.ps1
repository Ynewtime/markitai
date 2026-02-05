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
# Detection Helpers for Optional Components
# ============================================================

function Test-PlaywrightBrowser {
    # Check if Chromium browser is installed via markitai's playwright
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCmd) {
        try {
            $uvToolDir = & uv tool dir 2>$null
            if ($uvToolDir) {
                $markitaiPlaywright = Join-Path $uvToolDir "markitai\Scripts\playwright.exe"
                if (Test-Path $markitaiPlaywright) {
                    # Check if chromium is installed by looking for browser directories
                    $cacheDir = Join-Path $env:LOCALAPPDATA "ms-playwright"
                    if (Test-Path $cacheDir) {
                        $chromiumDirs = Get-ChildItem -Path $cacheDir -Directory -Filter "chromium-*" -ErrorAction SilentlyContinue
                        if ($chromiumDirs) {
                            return $true
                        }
                    }
                }
            }
        } catch {}
    }
    return $false
}

function Test-LibreOffice {
    $soffice = Get-Command soffice -ErrorAction SilentlyContinue
    if ($soffice) { return $true }

    $commonPaths = @(
        "${env:ProgramFiles}\LibreOffice\program\soffice.exe",
        "${env:ProgramFiles(x86)}\LibreOffice\program\soffice.exe"
    )
    foreach ($path in $commonPaths) {
        if (Test-Path $path) { return $true }
    }
    return $false
}

function Test-FFmpeg {
    $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    return ($null -ne $ffmpegCmd)
}

function Test-ClaudeCLI {
    $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
    return ($null -ne $claudeCmd)
}

function Test-CopilotCLI {
    $copilotCmd = Get-Command copilot -ErrorAction SilentlyContinue
    return ($null -ne $copilotCmd)
}

# ============================================================
# Clack-style Installation Functions
# ============================================================

function Install-UV-Clack {
    if (Test-UV) {
        $version = (& uv --version 2>$null).Split(' ')[1]
        Clack-Success "uv $version"
        Track-Install -Component "uv" -Status "installed"
        return $true
    }

    Clack-Info "Installing uv..."

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
            Clack-Warn "uv installed but needs shell restart"
            Track-Install -Component "uv" -Status "installed"
            return $false
        }

        $version = (& uv --version 2>$null).Split(' ')[1]
        Clack-Success "uv $version"
        Track-Install -Component "uv" -Status "installed"
        return $true
    } catch {
        Clack-Error "uv installation failed"
        Track-Install -Component "uv" -Status "failed"
        return $false
    }
}

function Test-Python-Clack {
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
                    Clack-Success "Python $version"
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

function Install-Markitai-Clack {
    Clack-Info "Installing markitai..."

    # Build package spec with optional version
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
            Clack-Success "markitai $version"
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
            $null = & pipx install $pkg --python $pythonArg --force 2>&1
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldErrorAction
        }
        if ($exitCode -eq 0) {
            $markitaiCmd = Get-Command markitai -ErrorAction SilentlyContinue
            $version = if ($markitaiCmd) { & markitai --version 2>&1 | Select-Object -First 1 } else { "installed" }
            if (-not $version) { $version = "installed" }
            Clack-Success "markitai $version"
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
        Clack-Success "markitai $version"
        Track-Install -Component "markitai" -Status "installed"
        return $true
    }

    Clack-Error "markitai installation failed"
    Track-Install -Component "markitai" -Status "failed"
    return $false
}

function Install-PlaywrightBrowser-Clack {
    Clack-Info "Installing Playwright browser..."

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    # Method 1: Use playwright from markitai's uv tool environment (preferred)
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
    if (-not $markitaiPlaywright -or -not (Test-Path $markitaiPlaywright)) {
        $markitaiPlaywright = Join-Path $env:APPDATA "uv\tools\markitai\Scripts\playwright.exe"
    }

    if (Test-Path $markitaiPlaywright) {
        try {
            $null = & $markitaiPlaywright install chromium 2>&1
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Clack-Success "Playwright browser (Chromium)"
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
                Clack-Success "Playwright browser (Chromium)"
                Track-Install -Component "Playwright Browser" -Status "installed"
                return $true
            }
        } catch {}
    }

    $ErrorActionPreference = $oldErrorAction
    Clack-Error "Playwright browser installation failed"
    Track-Install -Component "Playwright Browser" -Status "failed"
    return $false
}

function Install-LibreOffice-Clack {
    Clack-Info "Installing LibreOffice..."

    # Priority: winget > scoop > choco
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        $null = & winget install TheDocumentFoundation.LibreOffice --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "LibreOffice"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        $null = & scoop bucket add extras 2>$null
        $null = & scoop install extras/libreoffice 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "LibreOffice"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        $null = & choco install libreoffice-fresh -y 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "LibreOffice"
            Track-Install -Component "LibreOffice" -Status "installed"
            return $true
        }
    }

    Clack-Error "LibreOffice installation failed"
    Track-Install -Component "LibreOffice" -Status "failed"
    return $false
}

function Install-FFmpeg-Clack {
    Clack-Info "Installing FFmpeg..."

    # Priority: winget > scoop > choco
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        $null = & winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "FFmpeg"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    $scoopCmd = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoopCmd) {
        $null = & scoop install ffmpeg 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "FFmpeg"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    $chocoCmd = Get-Command choco -ErrorAction SilentlyContinue
    if ($chocoCmd) {
        $null = & choco install ffmpeg -y 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "FFmpeg"
            Track-Install -Component "FFmpeg" -Status "installed"
            return $true
        }
    }

    Clack-Error "FFmpeg installation failed"
    Track-Install -Component "FFmpeg" -Status "failed"
    return $false
}

function Install-ClaudeCLI-Clack {
    Clack-Info "Installing Claude CLI..."

    # Prefer official install script (PowerShell)
    $claudeUrl = "https://claude.ai/install.ps1"
    try {
        $null = Invoke-Expression (Invoke-RestMethod -Uri $claudeUrl) 2>&1
        $claudeCmd = Get-Command claude -ErrorAction SilentlyContinue
        if ($claudeCmd) {
            $version = & claude --version 2>&1 | Select-Object -First 1
            Clack-Success "Claude CLI $version"
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
            Clack-Success "Claude CLI"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        $null = & npm install -g @anthropic-ai/claude-code 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "Claude CLI"
            Track-Install -Component "Claude Code CLI" -Status "installed"
            return $true
        }
    }

    Clack-Error "Claude CLI installation failed"
    Track-Install -Component "Claude Code CLI" -Status "failed"
    return $false
}

function Install-CopilotCLI-Clack {
    Clack-Info "Installing Copilot CLI..."

    # Prefer WinGet on Windows
    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        $null = & winget install GitHub.Copilot --accept-package-agreements --accept-source-agreements 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "Copilot CLI"
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
            Clack-Success "Copilot CLI"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    } elseif ($npmCmd) {
        $null = & npm install -g @github/copilot 2>&1
        if ($LASTEXITCODE -eq 0) {
            Clack-Success "Copilot CLI"
            Track-Install -Component "Copilot CLI" -Status "installed"
            return $true
        }
    }

    Clack-Error "Copilot CLI installation failed"
    Track-Install -Component "Copilot CLI" -Status "failed"
    return $false
}

function Install-MarkitaiExtra-Clack {
    param([string]$ExtraName)

    $pkg = "markitai[$ExtraName]"

    $pythonArg = $script:PYTHON_CMD
    if ($pythonArg -match "^py\s+-(\d+\.\d+)$") {
        $pythonArg = $Matches[1]
    }

    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    $uvExists = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvExists) {
        try {
            $null = & uv tool install $pkg --python $pythonArg --upgrade 2>&1
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Clack-Success "markitai[$ExtraName]"
                return $true
            }
        } catch {}
    }

    $pipxExists = Get-Command pipx -ErrorAction SilentlyContinue
    if ($pipxExists) {
        try {
            $null = & pipx install $pkg --python $pythonArg --force 2>&1
            if ($LASTEXITCODE -eq 0) {
                $ErrorActionPreference = $oldErrorAction
                Clack-Success "markitai[$ExtraName]"
                return $true
            }
        } catch {}
    }

    try {
        $cmdParts = $script:PYTHON_CMD -split " "
        $exe = $cmdParts[0]
        $baseArgs = if ($cmdParts.Length -gt 1) { $cmdParts[1..($cmdParts.Length-1)] } else { @() }
        $pipArgs = $baseArgs + @("-m", "pip", "install", "--user", "--upgrade", $pkg)
        $null = & $exe @pipArgs 2>&1
        if ($LASTEXITCODE -eq 0) {
            $ErrorActionPreference = $oldErrorAction
            Clack-Success "markitai[$ExtraName]"
            return $true
        }
    } catch {}

    $ErrorActionPreference = $oldErrorAction
    Clack-Error "markitai[$ExtraName] installation failed"
    return $false
}

# ============================================================
# Main Logic
# ============================================================

function Main {
    # Security checks
    if (-not (Test-ExecutionPolicy)) {}
    Test-AdminWarning
    Test-WSLWarning

    # Intro
    Clack-Intro "Markitai Setup"

    # Core installation section
    Clack-Section "Installing core components"

    if (-not (Install-UV-Clack)) { Clack-Cancel "Setup failed"; exit 1 }
    if (-not (Test-Python-Clack)) { Clack-Cancel "Setup failed"; exit 1 }
    if (-not (Install-Markitai-Clack)) { Clack-Cancel "Setup failed"; exit 1 }

    # Optional components section
    Clack-Section "Optional components"

    # Playwright browser - auto-detect
    if (Test-PlaywrightBrowser) {
        Clack-Success "Playwright browser (already installed)"
        Track-Install -Component "Playwright Browser" -Status "installed"
    } elseif (Clack-Confirm "Install Playwright browser (for JS-rendered pages)?" "y") {
        Install-PlaywrightBrowser-Clack | Out-Null
    } else {
        Clack-Skip "Playwright browser"
        Track-Install -Component "Playwright Browser" -Status "skipped"
    }

    # LibreOffice - auto-detect
    if (Test-LibreOffice) {
        Clack-Success "LibreOffice (already installed)"
        Track-Install -Component "LibreOffice" -Status "installed"
    } elseif (Clack-Confirm "Install LibreOffice (for .doc/.xls/.ppt)?" "n") {
        Install-LibreOffice-Clack | Out-Null
    } else {
        Clack-Skip "LibreOffice"
        Track-Install -Component "LibreOffice" -Status "skipped"
    }

    # FFmpeg - auto-detect
    if (Test-FFmpeg) {
        Clack-Success "FFmpeg (already installed)"
        Track-Install -Component "FFmpeg" -Status "installed"
    } elseif (Clack-Confirm "Install FFmpeg (for audio/video)?" "n") {
        Install-FFmpeg-Clack | Out-Null
    } else {
        Clack-Skip "FFmpeg"
        Track-Install -Component "FFmpeg" -Status "skipped"
    }

    # Claude CLI - auto-detect
    if (Test-ClaudeCLI) {
        $version = & claude --version 2>&1 | Select-Object -First 1
        Clack-Success "Claude CLI $version (already installed)"
        Track-Install -Component "Claude Code CLI" -Status "installed"
    } elseif (Clack-Confirm "Install Claude Code CLI?" "n") {
        if (Install-ClaudeCLI-Clack) {
            Install-MarkitaiExtra-Clack -ExtraName "claude-agent" | Out-Null
        }
    } else {
        Clack-Skip "Claude CLI"
        Track-Install -Component "Claude Code CLI" -Status "skipped"
    }

    # Copilot CLI - auto-detect
    if (Test-CopilotCLI) {
        $version = & copilot --version 2>&1 | Select-Object -First 1
        Clack-Success "Copilot CLI $version (already installed)"
        Track-Install -Component "Copilot CLI" -Status "installed"
    } elseif (Clack-Confirm "Install GitHub Copilot CLI?" "n") {
        if (Install-CopilotCLI-Clack) {
            Install-MarkitaiExtra-Clack -ExtraName "copilot" | Out-Null
        }
    } else {
        Clack-Skip "Copilot CLI"
        Track-Install -Component "Copilot CLI" -Status "skipped"
    }

    # Config
    Initialize-Config 2>$null | Out-Null

    # Getting started note
    Clack-Note "Getting started" "markitai -I          Interactive mode" "markitai file.pdf   Convert a file" "markitai --help     Show all options"

    # Outro
    Clack-Outro "Setup complete!"
}

# Run main function
Main
