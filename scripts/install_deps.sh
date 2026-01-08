#!/bin/bash
# MarkIt - System Dependencies Installation Script
# This script detects the operating system and installs required system dependencies
# Note: sudo is called internally when needed, no need to run this script with sudo

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            echo "debian"
        elif [ -f /etc/redhat-release ]; then
            echo "redhat"
        elif [ -f /etc/arch-release ]; then
            echo "arch"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Install dependencies on Debian/Ubuntu
install_debian() {
    log_info "Installing dependencies for Debian/Ubuntu..."
    
    sudo apt-get update
    
    # Pandoc
    log_info "Installing Pandoc..."
    sudo apt-get install -y pandoc
    
    # LibreOffice (headless) - as cross-platform alternative to MS Office
    log_info "Installing LibreOffice..."
    sudo apt-get install -y libreoffice --no-install-recommends
    
    # python-magic dependency
    log_info "Installing libmagic..."
    sudo apt-get install -y libmagic1
    
    # Image compression tools (optional)
    log_info "Installing image compression tools (optional)..."
    # libjpeg-turbo for JPEG compression
    sudo apt-get install -y libjpeg-turbo-progs || log_warn "libjpeg-turbo-progs not available"
    
    # oxipng - not available in apt, use brew or cargo
    install_oxipng
    
    # Optional: EMF/WMF support
    log_info "Installing EMF/WMF support (optional)..."
    sudo apt-get install -y libwmf-bin || log_warn "libwmf-bin not available"
    sudo apt-get install -y inkscape || log_warn "inkscape not available"
    
    log_info "Debian/Ubuntu dependencies installed successfully!"
}

# Install dependencies on Red Hat/CentOS/Fedora
install_redhat() {
    log_info "Installing dependencies for Red Hat/CentOS/Fedora..."
    
    # Detect package manager
    if command -v dnf &> /dev/null; then
        PKG_MGR="dnf"
    else
        PKG_MGR="yum"
    fi
    
    # Pandoc
    log_info "Installing Pandoc..."
    sudo $PKG_MGR install -y pandoc
    
    # LibreOffice
    log_info "Installing LibreOffice..."
    sudo $PKG_MGR install -y libreoffice
    
    # python-magic dependency
    log_info "Installing file-libs..."
    sudo $PKG_MGR install -y file-libs
    
    # Image tools (optional)
    log_info "Installing image tools (optional)..."
    sudo $PKG_MGR install -y libjpeg-turbo-utils || log_warn "libjpeg-turbo-utils not available"
    
    # oxipng - not in dnf/yum, use brew or cargo
    install_oxipng
    
    log_info "Red Hat/CentOS/Fedora dependencies installed successfully!"
}

# Install dependencies on Arch Linux
install_arch() {
    log_info "Installing dependencies for Arch Linux..."
    
    # Pandoc
    log_info "Installing Pandoc..."
    sudo pacman -S --needed --noconfirm pandoc
    
    # LibreOffice
    log_info "Installing LibreOffice..."
    sudo pacman -S --needed --noconfirm libreoffice-fresh
    
    # python-magic dependency
    log_info "Installing file..."
    sudo pacman -S --needed --noconfirm file
    
    # Image tools (optional)
    log_info "Installing image tools (optional)..."
    sudo pacman -S --needed --noconfirm libjpeg-turbo || log_warn "libjpeg-turbo not available"
    # Try pacman first for Arch, fallback to brew/cargo
    sudo pacman -S --needed --noconfirm oxipng 2>/dev/null || install_oxipng
    
    # Optional: EMF/WMF support
    log_info "Installing EMF/WMF support (optional)..."
    sudo pacman -S --needed --noconfirm libwmf || log_warn "libwmf not available"
    sudo pacman -S --needed --noconfirm inkscape || log_warn "inkscape not available"
    
    log_info "Arch Linux dependencies installed successfully!"
}

# Install dependencies on macOS
install_macos() {
    log_info "Installing dependencies for macOS..."
    
    # Check for Homebrew (required for most macOS dependencies)
    if ! command -v brew &> /dev/null; then
        log_error "Homebrew is required but not installed."
        log_info "Install Homebrew from https://brew.sh"
        exit 1
    fi
    
    # Pandoc
    log_info "Installing Pandoc..."
    brew install pandoc
    
    # LibreOffice - as alternative to MS Office
    log_info "Installing LibreOffice..."
    brew install --cask libreoffice
    
    # python-magic dependency
    log_info "Installing libmagic..."
    brew install libmagic
    
    # Image compression tools (optional)
    log_info "Installing image compression tools (optional)..."
    install_oxipng
    brew install mozjpeg || log_warn "mozjpeg not available"
    
    # Optional: EMF/WMF support
    log_info "Installing EMF/WMF support (optional)..."
    brew install libwmf || log_warn "libwmf not available"
    brew install --cask inkscape || log_warn "inkscape not available"
    
    log_info "macOS dependencies installed successfully!"
}

# Show Windows instructions
install_windows() {
    log_info "Windows detected."
    echo ""
    echo "Please use the PowerShell script instead:"
    echo ""
    echo "   .\\scripts\\install_deps_windows.ps1"
    echo ""
}

# Check if a command exists
check_command() {
    if command -v "$1" &> /dev/null; then
        log_info "$1 is installed: $(command -v $1)"
        return 0
    else
        log_warn "$1 is not installed"
        return 1
    fi
}

# Install oxipng (not available in most Linux package managers)
# Note: This function should run as normal user, not root (brew/cargo are user-installed)
install_oxipng() {
    # Already installed?
    if command -v oxipng &> /dev/null; then
        log_info "oxipng is already installed"
        return 0
    fi

    # Get the actual user (even when running with sudo)
    local ACTUAL_USER="${SUDO_USER:-$USER}"
    local ACTUAL_HOME=$(eval echo "~$ACTUAL_USER")
    
    # Check for brew in common locations
    local BREW_PATH=""
    if [ -x "/home/linuxbrew/.linuxbrew/bin/brew" ]; then
        BREW_PATH="/home/linuxbrew/.linuxbrew/bin/brew"
    elif [ -x "$ACTUAL_HOME/.linuxbrew/bin/brew" ]; then
        BREW_PATH="$ACTUAL_HOME/.linuxbrew/bin/brew"
    elif [ -x "/opt/homebrew/bin/brew" ]; then
        BREW_PATH="/opt/homebrew/bin/brew"
    elif [ -x "/usr/local/bin/brew" ]; then
        BREW_PATH="/usr/local/bin/brew"
    fi

    # Try Homebrew first (works on Linux too)
    if [ -n "$BREW_PATH" ]; then
        log_info "Installing oxipng via Homebrew..."
        if [ -n "$SUDO_USER" ]; then
            sudo -u "$SUDO_USER" "$BREW_PATH" install oxipng && return 0
        else
            "$BREW_PATH" install oxipng && return 0
        fi
        log_warn "Homebrew installation failed, trying alternatives..."
    fi

    # Check for cargo in common locations
    local CARGO_PATH=""
    if [ -x "$ACTUAL_HOME/.cargo/bin/cargo" ]; then
        CARGO_PATH="$ACTUAL_HOME/.cargo/bin/cargo"
    elif command -v cargo &> /dev/null; then
        CARGO_PATH="cargo"
    fi

    # Try cargo
    if [ -n "$CARGO_PATH" ]; then
        log_info "Installing oxipng via cargo..."
        if [ -n "$SUDO_USER" ]; then
            sudo -u "$SUDO_USER" "$CARGO_PATH" install oxipng && return 0
        else
            "$CARGO_PATH" install oxipng && return 0
        fi
        log_warn "Cargo installation failed"
    fi

    # Provide manual instructions
    log_warn "Could not install oxipng automatically."
    log_warn "Install options:"
    log_warn "  1. Homebrew: brew install oxipng"
    log_warn "  2. Cargo:    cargo install oxipng"
    log_warn "  3. Download: https://github.com/shssoichiro/oxipng/releases"
    return 1
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    echo ""
    
    check_command pandoc
    check_command soffice || check_command libreoffice
    check_command oxipng || log_warn "oxipng not found (optional)"
    check_command cjpeg || log_warn "mozjpeg/cjpeg not found (optional)"
    check_command inkscape || log_warn "inkscape not found (optional)"
    
    echo ""
    log_info "Verification complete!"
}

# Main
main() {
    echo "========================================"
    echo "  MarkIt - System Dependencies Installer"
    echo "========================================"
    echo ""
    
    OS=$(detect_os)
    log_info "Detected OS: $OS"
    echo ""
    
    case $OS in
        debian)
            install_debian
            ;;
        redhat)
            install_redhat
            ;;
        arch)
            install_arch
            ;;
        macos)
            install_macos
            ;;
        windows)
            install_windows
            ;;
        *)
            log_error "Unsupported operating system: $OS"
            log_info "Please install dependencies manually:"
            echo "  - Pandoc"
            echo "  - LibreOffice or MS Office"
            echo "  - oxipng (optional)"
            echo "  - mozjpeg (optional)"
            echo "  - libmagic"
            exit 1
            ;;
    esac
    
    echo ""
    verify_installation
    
    echo ""
    log_info "Next steps:"
    echo "  1. Create virtual environment and install: uv sync --all-extras"
    echo "  2. Activate: source .venv/bin/activate"
    echo ""
}

# Run main function
main "$@"
