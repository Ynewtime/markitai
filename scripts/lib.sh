#!/bin/sh
# Markitai Setup Library - Common Functions
# Supports bash/zsh/dash and other POSIX-compatible shells

# ============================================================
# Version Variables (can be overridden via environment)
# ============================================================
MARKITAI_VERSION="${MARKITAI_VERSION:-}"
UV_VERSION="${UV_VERSION:-}"

# ============================================================
# Installation Status Tracking
# ============================================================
INSTALLED_COMPONENTS=""
SKIPPED_COMPONENTS=""
FAILED_COMPONENTS=""

# Track component installation status
# Usage: track_install "component_name" "status"
# Status: installed, skipped, failed
track_install() {
    _component="$1"
    _status="$2"
    case "$_status" in
        installed)
            INSTALLED_COMPONENTS="${INSTALLED_COMPONENTS}${_component}|"
            ;;
        skipped)
            SKIPPED_COMPONENTS="${SKIPPED_COMPONENTS}${_component}|"
            ;;
        failed)
            FAILED_COMPONENTS="${FAILED_COMPONENTS}${_component}|"
            ;;
    esac
}

# ============================================================
# Color Definitions
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# ============================================================
# Output Helpers
# ============================================================
print_header() {
    printf "\n"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "  ${BOLD}%s${NC}\n" "$1"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n\n"
}

# Print welcome message (user edition)
# Usage: print_welcome_user
print_welcome_user() {
    printf "\n"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "  ${BOLD}Welcome to Markitai Setup!${NC}\n"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "\n"
    printf "  This script will install:\n"
    printf "    ${GREEN}•${NC} markitai - Markdown converter with LLM support\n"
    printf "\n"
    printf "  Optional components:\n"
    printf "    ${YELLOW}•${NC} Playwright - Browser automation for JS-rendered pages\n"
    printf "    ${YELLOW}•${NC} Claude Code CLI - Use your Claude subscription\n"
    printf "    ${YELLOW}•${NC} Copilot CLI - Use your GitHub Copilot subscription\n"
    printf "\n"
    printf "  ${BOLD}Press Ctrl+C to cancel at any time${NC}\n"
    printf "\n"
}

# Print welcome message (developer edition)
# Usage: print_welcome_dev
print_welcome_dev() {
    printf "\n"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "  ${BOLD}Markitai Development Environment Setup${NC}\n"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "\n"
    printf "  This script will set up:\n"
    printf "    ${GREEN}•${NC} Python virtual environment with all dependencies\n"
    printf "    ${GREEN}•${NC} pre-commit hooks for code quality\n"
    printf "\n"
    printf "  Optional components:\n"
    printf "    ${YELLOW}•${NC} Playwright - Browser automation\n"
    printf "    ${YELLOW}•${NC} LLM CLI tools - Claude Code / Copilot\n"
    printf "    ${YELLOW}•${NC} LLM Python SDKs - Programmatic LLM access\n"
    printf "\n"
    printf "  ${BOLD}Press Ctrl+C to cancel at any time${NC}\n"
    printf "\n"
}

print_step() {
    printf "${BLUE}[%s/%s]${NC} %s\n" "$1" "$2" "$3"
}

print_success() {
    printf "  ${GREEN}✓${NC} %s\n" "$1"
}

print_error() {
    printf "  ${RED}✗${NC} %s\n" "$1"
}

print_info() {
    printf "  ${YELLOW}→${NC} %s\n" "$1"
}

print_warning() {
    printf "  ${YELLOW}!${NC} %s\n" "$1"
}

# ============================================================
# User Interaction
# ============================================================

# Ask yes/no question with default
# Usage: ask_yes_no "Question?" "y|n"
# Note: Uses /dev/tty for input to support 'curl | sh' execution
ask_yes_no() {
    prompt="$1"
    default="$2"

    if [ "$default" = "y" ]; then
        hint="[Y/n, default: Yes]"
    else
        hint="[y/N, default: No]"
    fi

    printf "  ${YELLOW}?${NC} %s %s " "$prompt" "$hint"

    # Read from /dev/tty to support piped execution (curl | sh)
    # When script is piped, stdin is occupied by the pipe, so we read from tty directly
    if [ -t 0 ]; then
        # stdin is a terminal, read normally
        read -r answer
    else
        # stdin is not a terminal (piped), read from /dev/tty
        # Need explicit newline since tty input won't echo to stdout
        read -r answer < /dev/tty
        printf "\n"
    fi

    if [ -z "$answer" ]; then
        answer="$default"
    fi

    case "$answer" in
        [Yy]*) return 0 ;;
        *) return 1 ;;
    esac
}

# ============================================================
# Security Functions
# ============================================================

# Check if running as root and warn
warn_if_root() {
    if [ "$(id -u)" -eq 0 ]; then
        printf "\n"
        printf "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
        printf "  ${YELLOW}WARNING: Running as root${NC}\n"
        printf "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
        printf "\n"
        printf "  Running setup scripts as root carries risks:\n"
        printf "  1. PATH hijacking: ~/.local/bin may be writable by others\n"
        printf "  2. Remote code execution risks are amplified\n"
        printf "\n"
        printf "  Recommendation: Run as a regular user instead\n"
        printf "\n"

        if ! ask_yes_no "Continue as root?" "n"; then
            printf "\n"
            print_info "Exiting. Please run as a regular user."
            exit 1
        fi
    fi
    return 0
}

# Confirm before executing remote script
# Usage: confirm_remote_script "url" "name"
# Returns: 0 if confirmed, 1 if rejected
confirm_remote_script() {
    script_url="$1"
    script_name="$2"

    printf "\n"
    printf "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "  ${YELLOW}WARNING: About to execute remote script${NC}\n"
    printf "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "\n"
    printf "  Source: %s\n" "$script_url"
    printf "  Purpose: Install %s\n" "$script_name"
    printf "\n"
    printf "  This will download and execute code from the internet.\n"
    printf "  Make sure you trust this source.\n"
    printf "\n"

    if ask_yes_no "Confirm execution?" "n"; then
        return 0
    else
        return 1
    fi
}

# ============================================================
# Detection Functions
# ============================================================

# Detect OS and architecture
# Sets: OS_TYPE, ARCH_TYPE, IS_MACOS_ARM
detect_platform() {
    OS_TYPE=$(uname -s)
    ARCH_TYPE=$(uname -m)
    IS_MACOS_ARM=false

    if [ "$OS_TYPE" = "Darwin" ] && [ "$ARCH_TYPE" = "arm64" ]; then
        IS_MACOS_ARM=true
    fi
}

# Initialize platform detection
detect_platform

# Detect/install Python via uv
# Sets: PYTHON_CMD
# Returns: 0 if found, 1 if not
lib_detect_python() {
    # Use uv-managed Python 3.13
    if command -v uv >/dev/null 2>&1; then
        uv_python=$(uv python find 3.13 2>/dev/null)
        if [ -n "$uv_python" ] && [ -x "$uv_python" ]; then
            PYTHON_CMD="$uv_python"
            ver=$("$uv_python" -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>/dev/null)
            print_success "Python $ver (uv managed)"
            return 0
        fi

        # Not found, auto-install
        print_info "Installing Python 3.13..."
        if uv python install 3.13; then
            uv_python=$(uv python find 3.13 2>/dev/null)
            if [ -n "$uv_python" ] && [ -x "$uv_python" ]; then
                PYTHON_CMD="$uv_python"
                ver=$("$uv_python" -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>/dev/null)
                print_success "Python $ver installed (uv managed)"
                return 0
            fi
        fi
        print_error "Python 3.13 installation failed"
    else
        print_error "uv not installed, cannot manage Python"
    fi

    return 1
}

# Detect Node.js (requires 18+)
# Returns: 0 if found and meets requirements, 1 otherwise
lib_detect_node() {
    if ! command -v node >/dev/null 2>&1; then
        print_error "Node.js not found"
        printf "\n"
        print_warning "Please install Node.js 18+:"
        print_info "Download: https://nodejs.org/"
        if [ "$OS_TYPE" = "Darwin" ]; then
            if [ "$IS_MACOS_ARM" = true ]; then
                print_info "macOS (Apple Silicon): brew install node"
                print_info "  Homebrew installs native ARM64 binaries"
            else
                print_info "macOS (Intel): brew install node"
            fi
            print_info "fnm: curl -fsSL https://fnm.vercel.app/install | bash"
        elif [ "$OS_TYPE" = "Linux" ]; then
            print_info "Ubuntu/Debian: sudo apt install nodejs npm"
            print_info "nvm: curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash"
            print_info "fnm: curl -fsSL https://fnm.vercel.app/install | bash"
        else
            print_info "nvm: curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash"
        fi
        return 1
    fi

    # Get version and strip CR (Windows CRLF compatibility)
    version=$(node --version 2>/dev/null | tr -d '\r')

    # Check if version is empty
    if [ -z "$version" ]; then
        print_warning "Unable to get Node version (empty output)"
        return 1
    fi

    # Extract major version number
    major=$(printf '%s' "$version" | sed 's/^v//' | cut -d. -f1)

    # Validate: must be numeric
    case "$major" in
        ''|*[!0-9]*)
            print_warning "Unable to parse Node version: $version"
            return 1
            ;;
    esac

    if [ "$major" -ge 18 ]; then
        print_success "Node.js $version installed"
        return 0
    else
        print_warning "Node.js $version is outdated, 18+ recommended"
        return 0
    fi
}

# Detect UV package manager
# Returns: 0 if installed, 1 if not
lib_detect_uv() {
    if command -v uv >/dev/null 2>&1; then
        # Strip CR for CRLF compatibility (WSL/Git Bash)
        version=$(uv --version 2>/dev/null | head -n1 | tr -d '\r')
        print_success "$version installed"
        return 0
    fi
    return 1
}

# ============================================================
# Installation Functions
# ============================================================

# Install UV package manager
# Returns: 0 on success, 1 on failure, 2 if skipped
lib_install_uv() {
    print_info "Checking UV installation..."

    if lib_detect_uv; then
        track_install "uv" "installed"
        return 0
    fi

    print_info "UV not installed (required for Python and dependency management)"

    if ! ask_yes_no "Install UV automatically?" "y"; then
        print_error "UV is required, cannot continue"
        track_install "uv" "failed"
        return 1
    fi

    # Check curl availability
    if ! command -v curl >/dev/null 2>&1; then
        print_error "curl not found, cannot download UV installer"
        print_info "Please install curl first:"
        print_info "  Ubuntu/Debian: sudo apt install curl"
        print_info "  macOS: brew install curl"
        print_info "  Or install UV manually: https://docs.astral.sh/uv/getting-started/installation/"
        track_install "uv" "failed"
        return 1
    fi

    # Build install URL (with optional version)
    if [ -n "$UV_VERSION" ]; then
        uv_url="https://astral.sh/uv/$UV_VERSION/install.sh"
        print_info "Installing UV version: $UV_VERSION"
    else
        uv_url="https://astral.sh/uv/install.sh"
    fi

    # Confirm remote script execution
    if ! confirm_remote_script "$uv_url" "UV"; then
        print_info "Skipping UV installation"
        track_install "uv" "skipped"
        return 2  # Skipped
    fi

    print_info "Installing UV..."

    if curl -LsSf "$uv_url" | sh; then
        # Refresh PATH
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

        if command -v uv >/dev/null 2>&1; then
            version=$(uv --version 2>/dev/null | head -n1)
            print_success "$version installed successfully"
            track_install "uv" "installed"
            return 0
        else
            print_warning "UV installed, but shell needs to be reloaded"
            print_path_help "$HOME/.local/bin"
            print_info "Then run this script again"
            track_install "uv" "installed"
            return 1
        fi
    else
        print_error "UV installation failed"
        if ! check_network; then
            print_network_error
        else
            print_info "Manual install: curl -LsSf https://astral.sh/uv/install.sh | sh"
        fi
        track_install "uv" "failed"
        return 1
    fi
}

# Install markitai
# Requires: PYTHON_CMD to be set
# Returns: 0 on success, 1 on failure
lib_install_markitai() {
    print_info "Installing markitai..."

    # Build package spec with optional version
    # Note: Use [browser] instead of [all] to avoid installing unnecessary SDK packages
    # SDK packages (claude-agent, copilot) will be installed when user selects CLI tools
    if [ -n "$MARKITAI_VERSION" ]; then
        pkg="markitai[browser]==$MARKITAI_VERSION"
        print_info "Installing version: $MARKITAI_VERSION"
    else
        pkg="markitai[browser]"
    fi

    # Prefer uv tool install (recommended, installs to ~/.local/bin)
    if command -v uv >/dev/null 2>&1; then
        # CRITICAL: Use --python to specify the detected Python version
        # Use --upgrade to ensure latest version is installed
        if uv tool install "$pkg" --python "$PYTHON_CMD" --upgrade 2>/dev/null; then
            export PATH="$HOME/.local/bin:$PATH"
            version=$(markitai --version 2>/dev/null || echo "installed")
            print_success "markitai $version installed successfully"
            print_info "Installed to ~/.local/bin (using $PYTHON_CMD)"
            track_install "markitai" "installed"
            return 0
        fi
    fi

    # Fallback to pipx
    if command -v pipx >/dev/null 2>&1; then
        # Use --force to ensure latest version is installed
        if pipx install "$pkg" --python "$PYTHON_CMD" --force; then
            version=$(markitai --version 2>/dev/null || echo "installed")
            print_success "markitai $version installed successfully"
            track_install "markitai" "installed"
            return 0
        fi
    fi

    # Fallback to pip --user
    # Use --upgrade to ensure latest version is installed
    if "$PYTHON_CMD" -m pip install --user --upgrade "$pkg" 2>/dev/null; then
        export PATH="$HOME/.local/bin:$PATH"
        version=$(markitai --version 2>/dev/null || echo "installed")
        print_success "markitai $version installed successfully"
        print_path_help "$HOME/.local/bin"
        track_install "markitai" "installed"
        return 0
    fi

    print_error "markitai installation failed"
    # Check if network issue
    if ! check_network; then
        print_network_error
    else
        print_info "Manual install: uv tool install markitai --python $PYTHON_CMD"
    fi
    track_install "markitai" "failed"
    return 1
}

# Install Playwright browser (Chromium) and system dependencies
# Requires: markitai/playwright to be installed
# Security: Use uv tool environment's playwright to ensure correct version
# Returns: 0 on success, 1 on failure, 2 if skipped
lib_install_playwright_browser() {
    print_info "Playwright browser (Chromium):"
    print_info "  Purpose: Browser automation for JavaScript-rendered pages (Twitter, SPAs)"

    # Ask user consent before downloading
    if ! ask_yes_no "Download Chromium browser?" "y"; then
        print_info "Skipping Playwright browser installation"
        track_install "Playwright Browser" "skipped"
        return 2
    fi

    print_info "Downloading Chromium browser..."
    browser_installed=false

    # Method 1: Use playwright from markitai's uv tool environment (preferred)
    # This ensures we use the same playwright version that markitai depends on
    # Use 'uv tool dir' to get the correct path (respects UV_TOOL_DIR, XDG_DATA_HOME)
    markitai_playwright=""
    if command -v uv >/dev/null 2>&1; then
        uv_tool_dir=$(uv tool dir 2>/dev/null)
        if [ -n "$uv_tool_dir" ]; then
            markitai_playwright="$uv_tool_dir/markitai/bin/playwright"
        fi
    fi
    # Fallback to default path if uv tool dir detection failed
    if [ -z "$markitai_playwright" ] || [ ! -x "$markitai_playwright" ]; then
        markitai_playwright="$HOME/.local/share/uv/tools/markitai/bin/playwright"
    fi

    if [ -x "$markitai_playwright" ]; then
        if "$markitai_playwright" install chromium 2>/dev/null; then
            print_success "Chromium browser downloaded successfully"
            browser_installed=true
        fi
    fi

    # Method 2: Fallback to Python module (for pip/pipx installs)
    if [ "$browser_installed" = false ] && [ -n "$PYTHON_CMD" ]; then
        if "$PYTHON_CMD" -m playwright install chromium 2>/dev/null; then
            print_success "Chromium browser downloaded successfully"
            browser_installed=true
        fi
    fi

    if [ "$browser_installed" = false ]; then
        print_warning "Playwright browser installation failed"
        print_info "You can install later with: playwright install chromium"
        track_install "Playwright Browser" "failed"
        return 1
    fi

    # On Linux, install system dependencies (requires sudo)
    if [ "$(uname)" = "Linux" ]; then
        print_info "Chromium requires system dependencies on Linux"
        if ask_yes_no "Install system dependencies (requires sudo)?" "y"; then
            print_info "Installing system dependencies..."

            # Arch Linux: use pacman (playwright install-deps doesn't support Arch)
            if [ -f /etc/arch-release ]; then
                print_info "Detected Arch Linux, using pacman..."
                # Playwright Chromium core dependencies
                local arch_deps="nss nspr at-spi2-core cups libdrm mesa alsa-lib libxcomposite libxdamage libxrandr libxkbcommon pango cairo"
                # Optional fonts (better CJK support)
                local arch_fonts="noto-fonts noto-fonts-cjk noto-fonts-emoji ttf-liberation"
                if sudo pacman -S --noconfirm --needed $arch_deps $arch_fonts 2>/dev/null; then
                    print_success "System dependencies installed successfully"
                    track_install "Playwright Browser" "installed"
                    return 0
                else
                    print_warning "Some dependencies failed to install"
                    print_info "Manual install: sudo pacman -S $arch_deps"
                fi
            # Debian/Ubuntu: use playwright install-deps
            elif command -v apt-get >/dev/null 2>&1; then
                # Method 1: Use playwright from markitai's uv tool environment
                if [ -x "$markitai_playwright" ]; then
                    if "$markitai_playwright" install-deps chromium 2>/dev/null; then
                        print_success "System dependencies installed successfully"
                        track_install "Playwright Browser" "installed"
                        return 0
                    fi
                fi
                # Method 2: Fallback to Python module
                if [ -n "$PYTHON_CMD" ]; then
                    if "$PYTHON_CMD" -m playwright install-deps chromium 2>/dev/null; then
                        print_success "System dependencies installed successfully"
                        track_install "Playwright Browser" "installed"
                        return 0
                    fi
                fi
                print_warning "System dependencies installation failed"
                print_info "Manual install: sudo playwright install-deps chromium"
                print_info "Or: sudo apt install libnspr4 libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 libxdamage1 libgbm1 libpango-1.0-0 libcairo2 libasound2"
            # Other distros
            else
                print_warning "Unrecognized Linux distribution"
                print_info "Please install Chromium dependencies manually"
            fi
            track_install "Playwright Browser" "installed"
            return 0
        else
            print_warning "Skipped system dependencies installation"
            print_info "Chromium may not work"
            if [ -f /etc/arch-release ]; then
                print_info "Install later: sudo pacman -S nss nspr at-spi2-core cups libdrm mesa alsa-lib"
            else
                print_info "Install later: sudo playwright install-deps chromium"
            fi
            track_install "Playwright Browser" "installed"
            return 0
        fi
    fi

    track_install "Playwright Browser" "installed"
    return 0
}

# Install LibreOffice (optional)
# LibreOffice is required for converting .doc, .ppt, .xls files
# Returns: 0 on success, 1 on failure, 2 if skipped
lib_install_libreoffice() {
    print_info "Checking LibreOffice installation..."
    print_info "  Purpose: Convert legacy Office files (.doc, .ppt, .xls)"

    # Check for soffice (LibreOffice command)
    if command -v soffice >/dev/null 2>&1; then
        version=$(soffice --version 2>/dev/null | head -n1)
        print_success "LibreOffice installed: $version"
        track_install "LibreOffice" "installed"
        return 0
    fi

    # Check for libreoffice command (alternative)
    if command -v libreoffice >/dev/null 2>&1; then
        version=$(libreoffice --version 2>/dev/null | head -n1)
        print_success "LibreOffice installed: $version"
        track_install "LibreOffice" "installed"
        return 0
    fi

    print_warning "LibreOffice not installed (optional)"
    print_info "  Without LibreOffice, .doc/.ppt/.xls files cannot be converted"
    print_info "  Modern formats (.docx/.pptx/.xlsx) work without LibreOffice"

    if ! ask_yes_no "Install LibreOffice?" "n"; then
        print_info "Skipping LibreOffice installation"
        track_install "LibreOffice" "skipped"
        return 2  # Skipped
    fi

    print_info "Installing LibreOffice..."

    case "$(uname)" in
        Darwin)
            # macOS: use Homebrew
            if command -v brew >/dev/null 2>&1; then
                if brew install --cask libreoffice; then
                    print_success "LibreOffice installed via Homebrew"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            else
                print_error "Homebrew not found"
                print_info "Install Homebrew first: https://brew.sh"
                print_info "Then run: brew install --cask libreoffice"
            fi
            ;;
        Linux)
            # Linux: use package manager
            if [ -f /etc/debian_version ]; then
                # Debian/Ubuntu
                print_info "Installing via apt (requires sudo)..."
                if sudo apt update && sudo apt install -y libreoffice; then
                    print_success "LibreOffice installed via apt"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            elif [ -f /etc/fedora-release ]; then
                # Fedora
                print_info "Installing via dnf (requires sudo)..."
                if sudo dnf install -y libreoffice; then
                    print_success "LibreOffice installed via dnf"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            elif [ -f /etc/arch-release ]; then
                # Arch Linux
                print_info "Installing via pacman (requires sudo)..."
                if sudo pacman -S --noconfirm libreoffice-fresh; then
                    print_success "LibreOffice installed via pacman"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            else
                print_error "Unknown Linux distribution"
                print_info "Please install LibreOffice manually using your package manager"
            fi
            ;;
        *)
            print_error "Automatic installation not supported on this platform"
            print_info "Download from: https://www.libreoffice.org/download/"
            ;;
    esac

    print_warning "LibreOffice installation failed"
    print_info "Manual install options:"
    case "$(uname)" in
        Darwin)
            print_info "  brew install --cask libreoffice"
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                print_info "  sudo apt install libreoffice"
            elif [ -f /etc/fedora-release ]; then
                print_info "  sudo dnf install libreoffice"
            elif [ -f /etc/arch-release ]; then
                print_info "  sudo pacman -S libreoffice-fresh"
            else
                print_info "  Use your package manager to install libreoffice"
            fi
            ;;
        *)
            print_info "  Download from: https://www.libreoffice.org/download/"
            ;;
    esac
    track_install "LibreOffice" "failed"
    return 1
}

# Install FFmpeg (optional)
# FFmpeg is required for audio/video file processing
# Returns: 0 on success, 1 on failure, 2 if skipped
lib_install_ffmpeg() {
    print_info "Checking FFmpeg installation..."
    print_info "  Purpose: Process audio/video files (.mp3, .mp4, .wav, etc.)"

    if command -v ffmpeg >/dev/null 2>&1; then
        version=$(ffmpeg -version 2>/dev/null | head -n1)
        print_success "FFmpeg installed: $version"
        track_install "FFmpeg" "installed"
        return 0
    fi

    print_warning "FFmpeg not installed (optional)"
    print_info "  Without FFmpeg, audio/video files cannot be processed"

    if ! ask_yes_no "Install FFmpeg?" "n"; then
        print_info "Skipping FFmpeg installation"
        track_install "FFmpeg" "skipped"
        return 2
    fi

    print_info "Installing FFmpeg..."

    case "$(uname)" in
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                if brew install ffmpeg; then
                    print_success "FFmpeg installed via Homebrew"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            else
                print_error "Homebrew not found"
                print_info "Install Homebrew first: https://brew.sh"
            fi
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                print_info "Installing via apt (requires sudo)..."
                if sudo apt update && sudo apt install -y ffmpeg; then
                    print_success "FFmpeg installed via apt"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            elif [ -f /etc/fedora-release ]; then
                print_info "Installing via dnf (requires sudo)..."
                if sudo dnf install -y ffmpeg; then
                    print_success "FFmpeg installed via dnf"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            elif [ -f /etc/arch-release ]; then
                print_info "Installing via pacman (requires sudo)..."
                if sudo pacman -S --noconfirm ffmpeg; then
                    print_success "FFmpeg installed via pacman"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            else
                print_error "Unknown Linux distribution"
                print_info "Please install FFmpeg manually using your package manager"
            fi
            ;;
        *)
            print_error "Automatic installation not supported on this platform"
            print_info "Download from: https://ffmpeg.org/download.html"
            ;;
    esac

    print_warning "FFmpeg installation failed"
    print_info "Manual install options:"
    case "$(uname)" in
        Darwin)
            print_info "  brew install ffmpeg"
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                print_info "  sudo apt install ffmpeg"
            elif [ -f /etc/fedora-release ]; then
                print_info "  sudo dnf install ffmpeg"
            elif [ -f /etc/arch-release ]; then
                print_info "  sudo pacman -S ffmpeg"
            else
                print_info "  Use your package manager to install ffmpeg"
            fi
            ;;
        *)
            print_info "  Download from: https://ffmpeg.org/download.html"
            ;;
    esac
    track_install "FFmpeg" "failed"
    return 1
}

# Install markitai extra package
# Usage: lib_install_markitai_extra "claude-agent"
# Returns: 0 on success, 1 on failure
lib_install_markitai_extra() {
    extra_name="$1"
    pkg="markitai[$extra_name]"

    # Prefer uv tool install
    if command -v uv >/dev/null 2>&1; then
        if uv tool install "$pkg" --python "$PYTHON_CMD" --upgrade 2>/dev/null; then
            print_success "markitai[$extra_name] installed"
            return 0
        fi
    fi

    # Fallback to pipx
    if command -v pipx >/dev/null 2>&1; then
        if pipx install "$pkg" --python "$PYTHON_CMD" --force 2>/dev/null; then
            print_success "markitai[$extra_name] installed"
            return 0
        fi
    fi

    # Fallback to pip --user
    if "$PYTHON_CMD" -m pip install --user --upgrade "$pkg" 2>/dev/null; then
        print_success "markitai[$extra_name] installed"
        return 0
    fi

    print_warning "markitai[$extra_name] installation failed"
    return 1
}

# Install Claude Code CLI
# Returns: 0 on success, 1 on failure
lib_install_claude_cli() {
    print_info "Installing Claude Code CLI..."
    print_info "  Purpose: Use your Claude Pro/Team subscription with markitai"

    # Check if already installed
    if command -v claude >/dev/null 2>&1; then
        version=$(claude --version 2>/dev/null | head -n1)
        print_success "Claude Code CLI already installed: $version"
        track_install "Claude Code CLI" "installed"
        return 0
    fi

    # Prefer official install script (macOS/Linux/WSL)
    claude_url="https://claude.ai/install.sh"
    if confirm_remote_script "$claude_url" "Claude Code CLI"; then
        print_info "Installing via official script..."
        if curl -fsSL "$claude_url" | bash; then
            print_success "Claude Code CLI installed via official script"
            print_info "Run 'claude /login' to authenticate with your Claude subscription or API key"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    fi

    # Fallback: npm/pnpm if Node.js available
    if command -v pnpm >/dev/null 2>&1; then
        print_info "Installing via pnpm..."
        if pnpm add -g @anthropic-ai/claude-code; then
            print_success "Claude Code CLI installed via pnpm"
            print_info "Run 'claude /login' to authenticate with your Claude subscription or API key"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    elif command -v npm >/dev/null 2>&1; then
        print_info "Installing via npm..."
        if npm install -g @anthropic-ai/claude-code; then
            print_success "Claude Code CLI installed via npm"
            print_info "Run 'claude /login' to authenticate with your Claude subscription or API key"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    fi

    print_warning "Claude Code CLI installation failed"
    if ! check_network; then
        print_network_error
    else
        print_info "Manual install options:"
        print_info "  curl: curl -fsSL https://claude.ai/install.sh | bash"
        print_info "  pnpm: pnpm add -g @anthropic-ai/claude-code"
        print_info "  Docs: https://docs.anthropic.com/en/docs/claude-code"
    fi
    track_install "Claude Code CLI" "failed"
    return 1
}

# Install GitHub Copilot CLI
# Returns: 0 on success, 1 on failure
lib_install_copilot_cli() {
    print_info "Installing GitHub Copilot CLI..."
    print_info "  Purpose: Use your GitHub Copilot subscription with markitai"

    # Check if already installed
    if command -v copilot >/dev/null 2>&1; then
        version=$(copilot --version 2>/dev/null | head -n1)
        print_success "Copilot CLI already installed: $version"
        track_install "Copilot CLI" "installed"
        return 0
    fi

    # Prefer official install script (macOS/Linux/WSL)
    copilot_url="https://gh.io/copilot-install"
    if confirm_remote_script "$copilot_url" "GitHub Copilot CLI"; then
        print_info "Installing via official script..."
        if curl -fsSL "$copilot_url" | bash; then
            print_success "Copilot CLI installed via official script"
            print_info "Run 'copilot /login' to authenticate with your GitHub Copilot subscription"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    # Fallback: npm/pnpm if Node.js available
    if command -v pnpm >/dev/null 2>&1; then
        print_info "Installing via pnpm..."
        if pnpm add -g @github/copilot; then
            print_success "Copilot CLI installed via pnpm"
            print_info "Run 'copilot /login' to authenticate with your GitHub Copilot subscription"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    elif command -v npm >/dev/null 2>&1; then
        print_info "Installing via npm..."
        if npm install -g @github/copilot; then
            print_success "Copilot CLI installed via npm"
            print_info "Run 'copilot /login' to authenticate with your GitHub Copilot subscription"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    print_warning "Copilot CLI installation failed"
    if ! check_network; then
        print_network_error
    else
        print_info "Manual install options:"
        print_info "  curl: curl -fsSL https://gh.io/copilot-install | bash"
        print_info "  pnpm: pnpm add -g @github/copilot"
    fi
    track_install "Copilot CLI" "failed"
    return 1
}

# Initialize markitai config
lib_init_config() {
    print_info "Initializing configuration..."

    if ! command -v markitai >/dev/null 2>&1; then
        return 0
    fi

    local config_path="$HOME/.markitai/config.json"
    local yes_flag=""

    # Check if config exists and ask user (using ask_yes_no for piped execution)
    if [ -f "$config_path" ]; then
        if ask_yes_no "$config_path already exists. Overwrite?" "n"; then
            yes_flag="--yes"
        else
            print_info "Keeping existing configuration"
            return 0
        fi
    fi

    if markitai config init $yes_flag 2>/dev/null; then
        print_success "Configuration initialized"
    fi
    return 0
}

# Print completion message
lib_print_completion() {
    printf "\n"
    printf "${GREEN}✓${NC} ${BOLD}Setup complete!${NC}\n"
    printf "\n"
    printf "  ${BOLD}Get started:${NC}\n"
    printf "    ${YELLOW}markitai --help${NC}\n"
    printf "\n"
}

# ============================================================
# Network and Diagnostics
# ============================================================

# Check basic network connectivity
# Returns: 0 if network available, 1 if not
check_network() {
    # Try to reach common endpoints
    if command -v curl >/dev/null 2>&1; then
        if curl -fsS --connect-timeout 5 "https://pypi.org" >/dev/null 2>&1; then
            return 0
        fi
    elif command -v wget >/dev/null 2>&1; then
        if wget -q --timeout=5 --spider "https://pypi.org" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# Print network error diagnostics
print_network_error() {
    printf "\n"
    printf "  ${RED}Network Error${NC}\n"
    printf "\n"
    printf "  Possible causes:\n"
    printf "    1. No internet connection\n"
    printf "    2. Firewall blocking access\n"
    printf "    3. Proxy configuration required\n"
    printf "    4. DNS resolution failure\n"
    printf "\n"
    printf "  Solutions:\n"
    printf "    • Check your network connection\n"
    printf "    • If behind proxy: export https_proxy=http://proxy:port\n"
    printf "    • Try again later if server is temporarily unavailable\n"
    printf "\n"
}

# ============================================================
# PATH Configuration Helpers
# ============================================================

# Print PATH configuration help
print_path_help() {
    target_dir="$1"
    shell_name=$(basename "$SHELL" 2>/dev/null || echo "bash")

    printf "\n"
    printf "  ${YELLOW}Command not found?${NC} Add to PATH:\n"
    printf "\n"
    printf "  ${BOLD}Temporary (current session):${NC}\n"
    printf "    export PATH=\"%s:\$PATH\"\n" "$target_dir"
    printf "\n"
    printf "  ${BOLD}Permanent:${NC}\n"

    case "$shell_name" in
        zsh)
            printf "    echo 'export PATH=\"%s:\$PATH\"' >> ~/.zshrc\n" "$target_dir"
            printf "    source ~/.zshrc\n"
            ;;
        fish)
            printf "    fish_add_path %s\n" "$target_dir"
            ;;
        ksh|ksh93|mksh|pdksh)
            # ksh uses ~/.kshrc or ~/.profile
            if [ -f "$HOME/.kshrc" ]; then
                printf "    echo 'export PATH=\"%s:\$PATH\"' >> ~/.kshrc\n" "$target_dir"
                printf "    . ~/.kshrc\n"
            else
                printf "    echo 'export PATH=\"%s:\$PATH\"' >> ~/.profile\n" "$target_dir"
                printf "    . ~/.profile\n"
            fi
            ;;
        *)
            printf "    echo 'export PATH=\"%s:\$PATH\"' >> ~/.bashrc\n" "$target_dir"
            printf "    source ~/.bashrc\n"
            ;;
    esac
    printf "\n"
}

# ============================================================
# Installation Summary
# ============================================================

# Print installation summary
# Usage: print_summary
print_summary() {
    printf "\n"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "  ${BOLD}Installation Summary${NC}\n"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "\n"

    # Print installed components
    if [ -n "$INSTALLED_COMPONENTS" ]; then
        printf "  ${GREEN}Installed:${NC}\n"
        echo "$INSTALLED_COMPONENTS" | tr '|' '\n' | while read -r comp; do
            [ -n "$comp" ] && printf "    ${GREEN}✓${NC} %s\n" "$comp"
        done
        printf "\n"
    fi

    # Print skipped components
    if [ -n "$SKIPPED_COMPONENTS" ]; then
        printf "  ${YELLOW}Skipped:${NC}\n"
        echo "$SKIPPED_COMPONENTS" | tr '|' '\n' | while read -r comp; do
            [ -n "$comp" ] && printf "    ${YELLOW}○${NC} %s\n" "$comp"
        done
        printf "\n"
    fi

    # Print failed components
    if [ -n "$FAILED_COMPONENTS" ]; then
        printf "  ${RED}Failed:${NC}\n"
        echo "$FAILED_COMPONENTS" | tr '|' '\n' | while read -r comp; do
            [ -n "$comp" ] && printf "    ${RED}✗${NC} %s\n" "$comp"
        done
        printf "\n"
    fi

    printf "  ${BOLD}Documentation:${NC} https://markitai.ynewtime.com\n"
    printf "  ${BOLD}Issues:${NC} https://github.com/Ynewtime/markitai/issues\n"
    printf "\n"
    return 0
}
