# MarkIt - Windows System Dependencies Installation Script
#
# Permission Requirements:
#   - Scoop:  No admin required (installs to user directory)
#   - Winget: Admin required (UAC prompt will appear)
#   - Cargo:  No admin required (installs to ~/.cargo)
#
# Recommendation: Install scoop first for admin-free experience
#   irm get.scoop.sh | iex

Write-Host "========================================"
Write-Host "  MarkIt - Windows Dependencies Installer"
Write-Host "========================================"
Write-Host ""

function Test-CommandExists {
    param ([string]$Command)
    $null -ne (Get-Command -Name $Command -ErrorAction SilentlyContinue)
}

function Test-IsAdmin {
    $currentUser = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    return $currentUser.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Install-WithWinget {
    param (
        [string]$PackageId,
        [string]$Name
    )
    
    if (Test-CommandExists "winget") {
        Write-Host "[INFO] Installing $Name via winget..."
        winget install --id $PackageId --silent --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] $Name installed successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[WARNING] $Name installation may have failed" -ForegroundColor Yellow
            return $false
        }
    } else {
        Write-Host "[WARNING] winget is not available" -ForegroundColor Yellow
        return $false
    }
}

function Install-WithScoop {
    param (
        [string]$PackageName,
        [string]$Name
    )
    
    if (Test-CommandExists "scoop") {
        Write-Host "[INFO] Installing $Name via scoop..."
        scoop install $PackageName
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] $Name installed successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[WARNING] $Name installation may have failed" -ForegroundColor Yellow
            return $false
        }
    } else {
        Write-Host "[WARNING] scoop is not available" -ForegroundColor Yellow
        return $false
    }
}

function Install-Oxipng {
    # Already installed?
    if (Test-CommandExists "oxipng") {
        Write-Host "[INFO] oxipng is already installed" -ForegroundColor Green
        return $true
    }
    
    # Try scoop first (simpler for CLI tools)
    if (Test-CommandExists "scoop") {
        if (Install-WithScoop "oxipng" "oxipng") {
            return $true
        }
    }
    
    # Try cargo
    if (Test-CommandExists "cargo") {
        Write-Host "[INFO] Installing oxipng via cargo..."
        cargo install oxipng
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] oxipng installed successfully" -ForegroundColor Green
            return $true
        }
    }
    
    # Manual instructions
    Write-Host "[WARNING] Could not install oxipng automatically." -ForegroundColor Yellow
    Write-Host "  Install options:"
    Write-Host "    1. Scoop:    scoop install oxipng"
    Write-Host "    2. Cargo:    cargo install oxipng"
    Write-Host "    3. Download: https://github.com/shssoichiro/oxipng/releases"
    return $false
}

# Detect available package managers
$hasWinget = Test-CommandExists "winget"
$hasScoop = Test-CommandExists "scoop"
$isAdmin = Test-IsAdmin

Write-Host "=== Package Managers ===" -ForegroundColor Cyan
if ($hasScoop) {
    Write-Host "[OK] scoop is available (no admin required)" -ForegroundColor Green
} else {
    Write-Host "[MISSING] scoop is not installed" -ForegroundColor Yellow
}
if ($hasWinget) {
    Write-Host "[OK] winget is available (requires admin)" -ForegroundColor Green
} else {
    Write-Host "[MISSING] winget is not installed" -ForegroundColor Yellow
}

if (-not $hasWinget -and -not $hasScoop) {
    Write-Host ""
    Write-Host "[ERROR] Neither winget nor scoop is installed." -ForegroundColor Red
    Write-Host "Please install at least one package manager:"
    Write-Host "  - scoop (recommended, no admin): irm get.scoop.sh | iex"
    Write-Host "  - winget: https://www.microsoft.com/store/productId/9NBLGGH4NNS1"
    exit 1
}

# Warn if only winget is available and not running as admin
if (-not $hasScoop -and $hasWinget -and -not $isAdmin) {
    Write-Host ""
    Write-Host "[WARNING] Running winget without Admin rights will cause UAC prompts" -ForegroundColor Yellow
    Write-Host "          Consider installing scoop first: irm get.scoop.sh | iex" -ForegroundColor Yellow
}
Write-Host ""

# Install Pandoc
Write-Host "=== Installing Pandoc ===" -ForegroundColor Cyan
if (Test-CommandExists "pandoc") {
    Write-Host "[INFO] Pandoc is already installed: $(pandoc --version | Select-Object -First 1)" -ForegroundColor Green
} else {
    $installed = $false
    if ($hasScoop) {
        $installed = Install-WithScoop "pandoc" "Pandoc"
    }
    if (-not $installed -and $hasWinget) {
        Install-WithWinget "JohnMacFarlane.Pandoc" "Pandoc"
    }
}
Write-Host ""

# Check for MS Office
Write-Host "=== Checking for MS Office ===" -ForegroundColor Cyan
$officeInstalled = $false
$officePaths = @(
    "$env:ProgramFiles\Microsoft Office",
    "${env:ProgramFiles(x86)}\Microsoft Office",
    "$env:ProgramFiles\Microsoft Office 15",
    "$env:ProgramFiles\Microsoft Office 16"
)
foreach ($path in $officePaths) {
    if (Test-Path $path) {
        Write-Host "[INFO] MS Office found at: $path" -ForegroundColor Green
        $officeInstalled = $true
        break
    }
}

if (-not $officeInstalled) {
    Write-Host "[INFO] MS Office not found. Installing LibreOffice as alternative..."
    $installed = $false
    if ($hasScoop) {
        $installed = Install-WithScoop "libreoffice" "LibreOffice"
    }
    if (-not $installed -and $hasWinget) {
        Install-WithWinget "TheDocumentFoundation.LibreOffice" "LibreOffice"
    }
}
Write-Host ""

# Install Inkscape (optional, for vector graphics)
Write-Host "=== Installing Inkscape (optional) ===" -ForegroundColor Cyan
if (Test-CommandExists "inkscape") {
    Write-Host "[INFO] Inkscape is already installed" -ForegroundColor Green
} else {
    $installInkscape = Read-Host "Install Inkscape for EMF/WMF support? (y/n)"
    if ($installInkscape -eq "y") {
        $installed = $false
        if ($hasScoop) {
            $installed = Install-WithScoop "inkscape" "Inkscape"
        }
        if (-not $installed -and $hasWinget) {
            Install-WithWinget "Inkscape.Inkscape" "Inkscape"
        }
    } else {
        Write-Host "[INFO] Skipping Inkscape installation"
    }
}
Write-Host ""

# Image compression tools (optional)
Write-Host "=== Image Compression Tools (optional) ===" -ForegroundColor Cyan
Install-Oxipng
Write-Host ""
Write-Host "[INFO] For JPEG compression, optionally download mozjpeg:"
Write-Host "  - https://mozjpeg.codelove.de/binaries.html"
Write-Host ""

# TODO (v0.2.0): Tesseract OCR for scanned PDF support
# Write-Host "=== Tesseract OCR (optional) ===" -ForegroundColor Cyan
# if ($hasScoop) {
#     Install-WithScoop "tesseract" "Tesseract OCR"
# } elseif ($hasWinget) {
#     Install-WithWinget "UB-Mannheim.TesseractOCR" "Tesseract OCR"
# }
# Write-Host "[INFO] For Chinese language support, download chi_sim.traineddata and chi_tra.traineddata"
# Write-Host "  - https://github.com/tesseract-ocr/tessdata"
# Write-Host ""

# Verify installations
Write-Host "=== Verification ===" -ForegroundColor Cyan
Write-Host ""

if (Test-CommandExists "pandoc") {
    Write-Host "[OK] Pandoc: $(pandoc --version | Select-Object -First 1)" -ForegroundColor Green
} else {
    Write-Host "[MISSING] Pandoc" -ForegroundColor Red
}

if ($officeInstalled) {
    Write-Host "[OK] MS Office: Installed" -ForegroundColor Green
} elseif (Test-CommandExists "soffice") {
    Write-Host "[OK] LibreOffice: Installed" -ForegroundColor Green
} else {
    Write-Host "[MISSING] Office suite (MS Office or LibreOffice)" -ForegroundColor Red
}

if (Test-CommandExists "inkscape") {
    Write-Host "[OK] Inkscape: Installed" -ForegroundColor Green
} else {
    Write-Host "[OPTIONAL] Inkscape: Not installed" -ForegroundColor Yellow
}

if (Test-CommandExists "oxipng") {
    Write-Host "[OK] oxipng: Installed" -ForegroundColor Green
} else {
    Write-Host "[OPTIONAL] oxipng: Not installed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================"
Write-Host "  Installation Complete"
Write-Host "========================================"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Open a new terminal (to refresh PATH)"
Write-Host "  2. Navigate to the markit directory"
Write-Host "  3. Create virtual environment and install: uv sync --all-extras"
Write-Host "  4. Activate: .venv\Scripts\activate"
Write-Host ""
Write-Host "Note: python-magic-bin will be installed automatically."
Write-Host ""
