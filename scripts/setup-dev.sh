#!/bin/sh
# Markitai Setup Script (Developer Edition)
# Supports bash/zsh/dash and other POSIX-compatible shells

set -e

# ============================================================
# Library Loading (supports both local and remote execution)
# ============================================================

LIB_BASE_URL="https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts"

load_library() {
    # Check if running locally (script file exists and is not sh/bash)
    if [ -f "$0" ] && [ "$(basename "$0")" != "sh" ] && [ "$(basename "$0")" != "bash" ] && [ "$(basename "$0")" != "dash" ]; then
        # Local execution
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        if [ -f "$SCRIPT_DIR/lib.sh" ]; then
            . "$SCRIPT_DIR/lib.sh"
            return 0
        fi
    fi

    # Remote execution - Developer edition requires local clone
    echo ""
    echo "================================================"
    echo "  Developer Edition requires local repository"
    echo "================================================"
    echo ""
    echo "  Please clone the repository first:"
    echo ""
    echo "    git clone https://github.com/Ynewtime/markitai.git"
    echo "    cd markitai"
    echo "    ./scripts/setup-dev.sh"
    echo ""
    echo "  Or use the user edition for quick install:"
    echo ""
    echo "    curl -fsSL https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.sh | sh"
    echo ""
    exit 1
}

load_library

# ============================================================
# Developer-specific Functions
# ============================================================

# Get project root directory (parent of script directory)
get_project_root() {
    dirname "$SCRIPT_DIR"
}

# Install UV (required for developer edition)
# Returns: 0 on success, 1 on failure (will exit script)
dev_install_uv() {
    if command -v uv >/dev/null 2>&1; then
        version=$(uv --version 2>/dev/null | head -n1)
        clack_success "$version"
        track_install "uv" "installed"
        return 0
    fi

    clack_error "UV not installed"

    if ! clack_confirm "Install UV automatically?" "n"; then
        clack_error "UV is required for development"
        track_install "uv" "failed"
        return 1
    fi

    # Check curl availability
    if ! command -v curl >/dev/null 2>&1; then
        clack_error "curl not found, cannot download UV installer"
        clack_info "Please install curl first:"
        clack_info "  Ubuntu/Debian: sudo apt install curl"
        clack_info "  macOS: brew install curl"
        track_install "uv" "failed"
        return 1
    fi

    # Build install URL (with optional version)
    if [ -n "$UV_VERSION" ]; then
        uv_url="https://astral.sh/uv/$UV_VERSION/install.sh"
        clack_info "Installing UV version: $UV_VERSION"
    else
        uv_url="https://astral.sh/uv/install.sh"
    fi

    # Confirm remote script execution
    if ! confirm_remote_script "$uv_url" "UV"; then
        clack_error "UV is required for development"
        track_install "uv" "failed"
        return 1
    fi

    clack_info "Installing UV..."

    if curl -LsSf "$uv_url" | sh; then
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

        if command -v uv >/dev/null 2>&1; then
            version=$(uv --version 2>/dev/null | head -n1)
            clack_success "$version installed successfully"
            track_install "uv" "installed"
            return 0
        else
            clack_warn "UV installed, but shell needs to be reloaded"
            clack_info "Run: source ~/.bashrc or restart terminal"
            clack_info "Then run this script again"
            track_install "uv" "installed"
            return 1
        fi
    else
        clack_error "UV installation failed"
        clack_info "Manual install: curl -LsSf https://astral.sh/uv/install.sh | sh"
        track_install "uv" "failed"
        return 1
    fi
}

# Detect Python via uv (clack-style output)
dev_detect_python() {
    # Use uv-managed Python 3.13
    if command -v uv >/dev/null 2>&1; then
        uv_python=$(uv python find 3.13 2>/dev/null)
        if [ -n "$uv_python" ] && [ -x "$uv_python" ]; then
            PYTHON_CMD="$uv_python"
            ver=$("$uv_python" -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>/dev/null)
            clack_success "Python $ver"
            return 0
        fi

        # Not found, auto-install
        clack_info "Installing Python 3.13..."
        if uv python install 3.13 >/dev/null 2>&1; then
            uv_python=$(uv python find 3.13 2>/dev/null)
            if [ -n "$uv_python" ] && [ -x "$uv_python" ]; then
                PYTHON_CMD="$uv_python"
                ver=$("$uv_python" -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>/dev/null)
                clack_success "Python $ver installed"
                return 0
            fi
        fi
        clack_error "Python 3.13 installation failed"
    else
        clack_error "uv not installed"
    fi

    return 1
}

# Sync development dependencies
sync_dependencies() {
    project_root=$(get_project_root)
    clack_info "Project directory: $project_root"

    cd "$project_root"

    # CRITICAL: Use --python to specify the detected Python version
    if clack_spinner "Syncing dependencies..." uv sync --all-extras --python "$PYTHON_CMD"; then
        clack_success "Dependencies synced"
        return 0
    else
        clack_error "Dependency sync failed"
        return 1
    fi
}

# Install pre-commit hooks
install_precommit() {
    project_root=$(get_project_root)
    cd "$project_root"

    if [ -f ".pre-commit-config.yaml" ]; then
        if clack_spinner "Installing pre-commit hooks..." uv run pre-commit install; then
            clack_success "pre-commit hooks installed"
            return 0
        else
            clack_warn "pre-commit installation failed"
            clack_info "Run manually: uv run pre-commit install"
            return 0
        fi
    else
        clack_skip ".pre-commit-config.yaml not found"
    fi

    return 0
}

# Install Claude Code CLI
dev_install_claude_cli() {
    clack_info "Installing Claude Code CLI..."

    # Check if already installed
    if command -v claude >/dev/null 2>&1; then
        version=$(claude --version 2>/dev/null | head -n1)
        clack_success "Claude Code CLI already installed: $version"
        track_install "Claude Code CLI" "installed"
        return 0
    fi

    # Prefer npm/pnpm if Node.js available
    if command -v pnpm >/dev/null 2>&1; then
        clack_info "Installing via pnpm..."
        if pnpm add -g @anthropic-ai/claude-code; then
            clack_success "Claude Code CLI installed via pnpm"
            clack_info "Run 'claude /login' to authenticate with your Claude subscription or API key"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    elif command -v npm >/dev/null 2>&1; then
        clack_info "Installing via npm..."
        if npm install -g @anthropic-ai/claude-code >/dev/null 2>&1; then
            clack_success "Claude Code CLI installed via npm"
            clack_info "Run 'claude /login' to authenticate with your Claude subscription or API key"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    fi

    # Fallback: Homebrew (macOS/Linux)
    if command -v brew >/dev/null 2>&1; then
        clack_info "Installing via Homebrew..."
        if brew install claude-code >/dev/null 2>&1; then
            clack_success "Claude Code CLI installed via Homebrew"
            clack_info "Run 'claude /login' to authenticate with your Claude subscription or API key"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    fi

    clack_warn "Claude Code CLI installation failed"
    clack_info "Manual install options:"
    clack_info "  pnpm: pnpm add -g @anthropic-ai/claude-code"
    clack_info "  brew: brew install claude-code"
    clack_info "  Docs: https://code.claude.com/docs/en/setup"
    track_install "Claude Code CLI" "failed"
    return 1
}

# Install GitHub Copilot CLI
dev_install_copilot_cli() {
    clack_info "Installing GitHub Copilot CLI..."

    # Check if already installed
    if command -v copilot >/dev/null 2>&1; then
        version=$(copilot --version 2>/dev/null | head -n1)
        clack_success "Copilot CLI already installed: $version"
        track_install "Copilot CLI" "installed"
        return 0
    fi

    # Prefer npm/pnpm if Node.js available
    if command -v pnpm >/dev/null 2>&1; then
        clack_info "Installing via pnpm..."
        if pnpm add -g @github/copilot; then
            clack_success "Copilot CLI installed via pnpm"
            clack_info "Run 'copilot /login' to authenticate with your GitHub Copilot subscription"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    elif command -v npm >/dev/null 2>&1; then
        clack_info "Installing via npm..."
        if npm install -g @github/copilot >/dev/null 2>&1; then
            clack_success "Copilot CLI installed via npm"
            clack_info "Run 'copilot /login' to authenticate with your GitHub Copilot subscription"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    # Fallback: Homebrew (macOS/Linux)
    if command -v brew >/dev/null 2>&1; then
        clack_info "Installing via Homebrew..."
        if brew install copilot-cli >/dev/null 2>&1; then
            clack_success "Copilot CLI installed via Homebrew"
            clack_info "Run 'copilot /login' to authenticate with your GitHub Copilot subscription"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    # Fallback: Install script (requires confirmation)
    copilot_url="https://gh.io/copilot-install"
    if confirm_remote_script "$copilot_url" "GitHub Copilot CLI"; then
        clack_info "Trying install script..."
        if curl -fsSL "$copilot_url" | bash; then
            clack_success "Copilot CLI installed via install script"
            clack_info "Run 'copilot /login' to authenticate with your GitHub Copilot subscription"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    clack_warn "Copilot CLI installation failed"
    clack_info "Manual install options:"
    clack_info "  pnpm: pnpm add -g @github/copilot"
    clack_info "  brew: brew install copilot-cli"
    clack_info "  curl: curl -fsSL https://gh.io/copilot-install | bash"
    track_install "Copilot CLI" "failed"
    return 1
}

# Install LLM CLI tools (Claude Code, Copilot)
dev_install_llm_clis() {
    clack_info "LLM CLI tools provide local authentication for AI providers:"
    clack_log "  - Claude Code CLI: Use your Claude subscription"
    clack_log "  - Copilot CLI: Use your GitHub Copilot subscription"

    if clack_confirm "Install Claude Code CLI?" "n"; then
        dev_install_claude_cli
    else
        track_install "Claude Code CLI" "skipped"
    fi

    if clack_confirm "Install GitHub Copilot CLI?" "n"; then
        dev_install_copilot_cli
    else
        track_install "Copilot CLI" "skipped"
    fi
    return 0
}

# Install Playwright browser (Chromium) and system dependencies for development
# Uses uv run (preferred) with fallback to python module
# Returns: 0 on success, 1 on failure, 2 if skipped
dev_install_playwright_browser() {
    project_root=$(get_project_root)
    cd "$project_root"

    # Check if already installed
    if lib_detect_playwright_browser; then
        clack_success "Playwright browser (Chromium)"
        track_install "Playwright Browser" "installed"
        return 0
    fi

    clack_info "Playwright browser (Chromium):"
    clack_log "  Purpose: Browser automation for JavaScript-rendered pages (Twitter, SPAs)"

    # Ask user consent before downloading
    if ! clack_confirm "Download Chromium browser?" "y"; then
        clack_skip "Playwright browser installation"
        track_install "Playwright Browser" "skipped"
        return 2
    fi

    clack_info "Downloading Chromium browser..."
    browser_installed=false

    # Prefer uv run in dev environment (uses .venv)
    if command -v uv >/dev/null 2>&1; then
        if uv run playwright install chromium 2>/dev/null; then
            clack_success "Chromium browser downloaded"
            browser_installed=true
        fi
    fi

    # Fallback to Python module
    if [ "$browser_installed" = false ] && [ -n "$PYTHON_CMD" ]; then
        if "$PYTHON_CMD" -m playwright install chromium 2>/dev/null; then
            clack_success "Chromium browser downloaded"
            browser_installed=true
        fi
    fi

    if [ "$browser_installed" = false ]; then
        clack_warn "Playwright browser installation failed"
        clack_info "You can install later with: uv run playwright install chromium"
        track_install "Playwright Browser" "failed"
        return 1
    fi

    # On Linux, install system dependencies (requires sudo)
    if [ "$(uname)" = "Linux" ]; then
        clack_info "Chromium requires system dependencies on Linux"
        if clack_confirm "Install system dependencies (requires sudo)?" "y"; then
            clack_info "Installing system dependencies..."

            # Arch Linux: use pacman (playwright install-deps doesn't support Arch)
            if [ -f /etc/arch-release ]; then
                clack_info "Detected Arch Linux, using pacman..."
                # Playwright Chromium core dependencies
                arch_deps="nss nspr at-spi2-core cups libdrm mesa alsa-lib libxcomposite libxdamage libxrandr libxkbcommon pango cairo"
                # Optional fonts (better CJK support)
                arch_fonts="noto-fonts noto-fonts-cjk noto-fonts-emoji ttf-liberation"
                if sudo pacman -S --noconfirm --needed $arch_deps $arch_fonts 2>/dev/null; then
                    clack_success "System dependencies installed"
                    track_install "Playwright Browser" "installed"
                    return 0
                else
                    clack_warn "Some dependencies failed to install"
                    clack_info "Manual install: sudo pacman -S $arch_deps"
                fi
            # Debian/Ubuntu: use playwright install-deps
            elif command -v apt-get >/dev/null 2>&1; then
                if command -v uv >/dev/null 2>&1; then
                    if uv run playwright install-deps chromium 2>/dev/null; then
                        clack_success "System dependencies installed"
                        track_install "Playwright Browser" "installed"
                        return 0
                    fi
                fi
                # Fallback to Python module
                if [ -n "$PYTHON_CMD" ]; then
                    if "$PYTHON_CMD" -m playwright install-deps chromium 2>/dev/null; then
                        clack_success "System dependencies installed"
                        track_install "Playwright Browser" "installed"
                        return 0
                    fi
                fi
                clack_warn "System dependencies installation failed"
                clack_info "Manual install: sudo playwright install-deps chromium"
            # Other distros
            else
                clack_warn "Unrecognized Linux distribution"
                clack_info "Please install Chromium dependencies manually"
            fi
            track_install "Playwright Browser" "installed"
            return 0
        else
            clack_warn "Skipped system dependencies installation"
            clack_info "Chromium may not work"
            if [ -f /etc/arch-release ]; then
                clack_info "Install later: sudo pacman -S nss nspr at-spi2-core cups libdrm mesa alsa-lib"
            else
                clack_info "Install later: sudo playwright install-deps chromium"
            fi
            track_install "Playwright Browser" "installed"
            return 0
        fi
    fi

    track_install "Playwright Browser" "installed"
    return 0
}

# Install LibreOffice (optional, for legacy Office files)
dev_install_libreoffice() {
    clack_info "LibreOffice:"
    clack_log "  Purpose: Convert legacy Office files (.doc, .ppt, .xls)"

    if command -v soffice >/dev/null 2>&1; then
        version=$(soffice --version 2>/dev/null | head -n1)
        clack_success "LibreOffice installed: $version"
        track_install "LibreOffice" "installed"
        return 0
    fi

    if command -v libreoffice >/dev/null 2>&1; then
        version=$(libreoffice --version 2>/dev/null | head -n1)
        clack_success "LibreOffice installed: $version"
        track_install "LibreOffice" "installed"
        return 0
    fi

    clack_warn "LibreOffice not installed (optional)"
    clack_log "  Without LibreOffice, .doc/.ppt/.xls files cannot be converted"
    clack_log "  Modern formats (.docx/.pptx/.xlsx) work without LibreOffice"

    if ! clack_confirm "Install LibreOffice?" "n"; then
        clack_skip "LibreOffice installation"
        track_install "LibreOffice" "skipped"
        return 2
    fi

    clack_info "Installing LibreOffice..."

    case "$(uname)" in
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                if brew install --cask libreoffice; then
                    clack_success "LibreOffice installed via Homebrew"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            else
                clack_error "Homebrew not found"
                clack_info "Install Homebrew first: https://brew.sh"
            fi
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                clack_info "Installing via apt (requires sudo)..."
                if sudo apt update && sudo apt install -y libreoffice; then
                    clack_success "LibreOffice installed via apt"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            elif [ -f /etc/fedora-release ]; then
                clack_info "Installing via dnf (requires sudo)..."
                if sudo dnf install -y libreoffice; then
                    clack_success "LibreOffice installed via dnf"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            elif [ -f /etc/arch-release ]; then
                clack_info "Installing via pacman (requires sudo)..."
                if sudo pacman -S --noconfirm libreoffice-fresh; then
                    clack_success "LibreOffice installed via pacman"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            else
                clack_error "Unknown Linux distribution"
                clack_info "Please install LibreOffice manually"
            fi
            ;;
    esac

    clack_warn "LibreOffice installation failed"
    clack_info "Manual install options:"
    case "$(uname)" in
        Darwin)
            clack_info "  brew install --cask libreoffice"
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                clack_info "  sudo apt install libreoffice"
            else
                clack_info "  Use your package manager to install libreoffice"
            fi
            ;;
    esac
    track_install "LibreOffice" "failed"
    return 1
}

# Install FFmpeg (optional, for audio/video file processing)
dev_install_ffmpeg() {
    clack_info "FFmpeg:"
    clack_log "  Purpose: Process audio/video files (.mp3, .mp4, .wav, etc.)"

    if command -v ffmpeg >/dev/null 2>&1; then
        version=$(ffmpeg -version 2>/dev/null | head -n1 | sed 's/ffmpeg version \([^ ]*\).*/\1/')
        clack_success "FFmpeg $version"
        track_install "FFmpeg" "installed"
        return 0
    fi

    clack_warn "FFmpeg not installed (optional)"
    clack_log "  Without FFmpeg, audio/video files cannot be processed"

    if ! clack_confirm "Install FFmpeg?" "n"; then
        clack_skip "FFmpeg installation"
        track_install "FFmpeg" "skipped"
        return 2
    fi

    clack_info "Installing FFmpeg..."

    case "$(uname)" in
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                if brew install ffmpeg; then
                    clack_success "FFmpeg installed via Homebrew"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            else
                clack_error "Homebrew not found"
                clack_info "Install Homebrew first: https://brew.sh"
            fi
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                clack_info "Installing via apt (requires sudo)..."
                if sudo apt update && sudo apt install -y ffmpeg; then
                    clack_success "FFmpeg installed via apt"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            elif [ -f /etc/fedora-release ]; then
                clack_info "Installing via dnf (requires sudo)..."
                if sudo dnf install -y ffmpeg; then
                    clack_success "FFmpeg installed via dnf"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            elif [ -f /etc/arch-release ]; then
                clack_info "Installing via pacman (requires sudo)..."
                if sudo pacman -S --noconfirm ffmpeg; then
                    clack_success "FFmpeg installed via pacman"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            else
                clack_error "Unknown Linux distribution"
                clack_info "Please install FFmpeg manually"
            fi
            ;;
    esac

    clack_warn "FFmpeg installation failed"
    clack_info "Manual install options:"
    case "$(uname)" in
        Darwin)
            clack_info "  brew install ffmpeg"
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                clack_info "  sudo apt install ffmpeg"
            elif [ -f /etc/fedora-release ]; then
                clack_info "  sudo dnf install ffmpeg"
            elif [ -f /etc/arch-release ]; then
                clack_info "  sudo pacman -S ffmpeg"
            else
                clack_info "  Use your package manager to install ffmpeg"
            fi
            ;;
    esac
    track_install "FFmpeg" "failed"
    return 1
}

# ============================================================
# Summary Function (clack style)
# ============================================================

dev_print_summary() {
    # Installed
    if [ -n "$INSTALLED_COMPONENTS" ]; then
        clack_note "Installed" <<EOF
$(echo "$INSTALLED_COMPONENTS" | tr '|' '\n' | while read -r comp; do
    [ -n "$comp" ] && printf "✓ %s\n" "$comp"
done)
EOF
    fi

    # Skipped
    if [ -n "$SKIPPED_COMPONENTS" ]; then
        clack_note "Skipped" <<EOF
$(echo "$SKIPPED_COMPONENTS" | tr '|' '\n' | while read -r comp; do
    [ -n "$comp" ] && printf "○ %s\n" "$comp"
done)
EOF
    fi

    # Failed
    if [ -n "$FAILED_COMPONENTS" ]; then
        clack_note "Failed" <<EOF
$(echo "$FAILED_COMPONENTS" | tr '|' '\n' | while read -r comp; do
    [ -n "$comp" ] && printf "✗ %s\n" "$comp"
done)
EOF
    fi

    clack_info "Documentation: https://markitai.ynewtime.com"
    clack_info "Issues: https://github.com/Ynewtime/markitai/issues"
    return 0
}

# ============================================================
# Main Logic
# ============================================================

main() {
    # Security check: warn if running as root
    warn_if_root

    # Welcome message with clack intro
    clack_intro "Markitai Development Environment Setup"

    # Section: Checking prerequisites
    clack_section "Checking prerequisites"

    # Detect/install UV (required, also manages Python)
    if ! dev_install_uv; then
        dev_print_summary
        clack_cancel "Setup failed: UV is required"
        exit 1
    fi

    # Detect/install Python (auto-installed via uv)
    if ! dev_detect_python; then
        dev_print_summary
        clack_cancel "Setup failed: Python 3.13 is required"
        exit 1
    fi

    # Section: Setting up development environment
    clack_section "Setting up development environment"

    # Sync dependencies (includes all extras: browser, claude-agent, copilot)
    if ! sync_dependencies; then
        dev_print_summary
        clack_cancel "Setup failed: Dependency sync failed"
        exit 1
    fi
    track_install "Python dependencies" "installed"
    track_install "Claude Agent SDK" "installed"
    track_install "Copilot SDK" "installed"

    # Install pre-commit
    if install_precommit; then
        track_install "pre-commit hooks" "installed"
    fi

    # Section: Optional components
    clack_section "Optional components"

    # Install Playwright browser (required for SPA/JS-rendered pages)
    dev_install_playwright_browser

    # Install LibreOffice (optional, for legacy Office files)
    dev_install_libreoffice

    # Install FFmpeg (optional, for audio/video files)
    dev_install_ffmpeg

    # Section: LLM CLI tools
    clack_section "LLM CLI tools"

    # Auto-detect Claude Code CLI
    if command -v claude >/dev/null 2>&1; then
        version=$(claude --version 2>/dev/null | head -n1)
        clack_success "Claude Code CLI: $version"
        track_install "Claude Code CLI" "installed"
    else
        if clack_confirm "Install Claude Code CLI?" "n"; then
            dev_install_claude_cli
        else
            clack_skip "Claude Code CLI"
            track_install "Claude Code CLI" "skipped"
        fi
    fi

    # Auto-detect Copilot CLI
    if command -v copilot >/dev/null 2>&1; then
        version=$(copilot --version 2>/dev/null | head -n1)
        clack_success "Copilot CLI: $version"
        track_install "Copilot CLI" "installed"
    else
        if clack_confirm "Install GitHub Copilot CLI?" "n"; then
            dev_install_copilot_cli
        else
            clack_skip "Copilot CLI"
            track_install "Copilot CLI" "skipped"
        fi
    fi

    # Print summary
    dev_print_summary

    # Completion message with clack note and outro
    project_root=$(get_project_root)
    clack_note "Next steps" \
        "Activate virtual environment:" \
        "  source $project_root/.venv/bin/activate" \
        "" \
        "Run tests:" \
        "  uv run pytest" \
        "" \
        "Run CLI:" \
        "  uv run markitai --help"

    clack_outro "Development environment ready!"
}

# Run main function
main
