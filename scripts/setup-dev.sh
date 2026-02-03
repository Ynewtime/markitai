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
    print_info "Checking UV installation..."

    if command -v uv >/dev/null 2>&1; then
        version=$(uv --version 2>/dev/null | head -n1)
        print_success "$version installed"
        track_install "uv" "installed"
        return 0
    fi

    print_error "UV not installed"

    if ! ask_yes_no "Install UV automatically?" "n"; then
        print_error "UV is required for development"
        track_install "uv" "failed"
        return 1
    fi

    # Check curl availability
    if ! command -v curl >/dev/null 2>&1; then
        print_error "curl not found, cannot download UV installer"
        print_info "Please install curl first:"
        print_info "  Ubuntu/Debian: sudo apt install curl"
        print_info "  macOS: brew install curl"
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
        print_error "UV is required for development"
        track_install "uv" "failed"
        return 1
    fi

    print_info "Installing UV..."

    if curl -LsSf "$uv_url" | sh; then
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

        if command -v uv >/dev/null 2>&1; then
            version=$(uv --version 2>/dev/null | head -n1)
            print_success "$version installed successfully"
            track_install "uv" "installed"
            return 0
        else
            print_warning "UV installed, but shell needs to be reloaded"
            print_info "Run: source ~/.bashrc or restart terminal"
            print_info "Then run this script again"
            track_install "uv" "installed"
            return 1
        fi
    else
        print_error "UV installation failed"
        print_info "Manual install: curl -LsSf https://astral.sh/uv/install.sh | sh"
        track_install "uv" "failed"
        return 1
    fi
}

# Sync development dependencies
sync_dependencies() {
    project_root=$(get_project_root)
    print_info "Project directory: $project_root"

    cd "$project_root"

    # CRITICAL: Use --python to specify the detected Python version
    print_info "Running uv sync --all-extras --python $PYTHON_CMD..."
    if uv sync --all-extras --python "$PYTHON_CMD"; then
        print_success "Dependencies synced successfully (using $PYTHON_CMD)"
        return 0
    else
        print_error "Dependency sync failed"
        return 1
    fi
}

# Install pre-commit hooks
install_precommit() {
    project_root=$(get_project_root)
    cd "$project_root"

    if [ -f ".pre-commit-config.yaml" ]; then
        print_info "Installing pre-commit hooks..."

        if uv run pre-commit install; then
            print_success "pre-commit hooks installed successfully"
            return 0
        else
            print_warning "pre-commit installation failed, please run manually: uv run pre-commit install"
            return 0
        fi
    else
        print_info ".pre-commit-config.yaml not found, skipping"
    fi

    return 0
}

# Install Claude Code CLI
dev_install_claude_cli() {
    print_info "Installing Claude Code CLI..."

    # Check if already installed
    if command -v claude >/dev/null 2>&1; then
        version=$(claude --version 2>/dev/null | head -n1)
        print_success "Claude Code CLI already installed: $version"
        track_install "Claude Code CLI" "installed"
        return 0
    fi

    # Prefer npm/pnpm if Node.js available
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

    # Fallback: Homebrew (macOS/Linux)
    if command -v brew >/dev/null 2>&1; then
        print_info "Installing via Homebrew..."
        if brew install claude-code; then
            print_success "Claude Code CLI installed via Homebrew"
            print_info "Run 'claude /login' to authenticate with your Claude subscription or API key"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    fi

    print_warning "Claude Code CLI installation failed"
    print_info "Manual install options:"
    print_info "  pnpm: pnpm add -g @anthropic-ai/claude-code"
    print_info "  brew: brew install claude-code"
    print_info "  Docs: https://code.claude.com/docs/en/setup"
    track_install "Claude Code CLI" "failed"
    return 1
}

# Install GitHub Copilot CLI
dev_install_copilot_cli() {
    print_info "Installing GitHub Copilot CLI..."

    # Check if already installed
    if command -v copilot >/dev/null 2>&1; then
        version=$(copilot --version 2>/dev/null | head -n1)
        print_success "Copilot CLI already installed: $version"
        track_install "Copilot CLI" "installed"
        return 0
    fi

    # Prefer npm/pnpm if Node.js available
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

    # Fallback: Homebrew (macOS/Linux)
    if command -v brew >/dev/null 2>&1; then
        print_info "Installing via Homebrew..."
        if brew install copilot-cli; then
            print_success "Copilot CLI installed via Homebrew"
            print_info "Run 'copilot /login' to authenticate with your GitHub Copilot subscription"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    # Fallback: Install script (requires confirmation)
    copilot_url="https://gh.io/copilot-install"
    if confirm_remote_script "$copilot_url" "GitHub Copilot CLI"; then
        print_info "Trying install script..."
        if curl -fsSL "$copilot_url" | bash; then
            print_success "Copilot CLI installed via install script"
            print_info "Run 'copilot /login' to authenticate with your GitHub Copilot subscription"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    print_warning "Copilot CLI installation failed"
    print_info "Manual install options:"
    print_info "  pnpm: pnpm add -g @github/copilot"
    print_info "  brew: brew install copilot-cli"
    print_info "  curl: curl -fsSL https://gh.io/copilot-install | bash"
    track_install "Copilot CLI" "failed"
    return 1
}

# Install LLM CLI tools (Claude Code, Copilot)
dev_install_llm_clis() {
    print_info "LLM CLI tools provide local authentication for AI providers:"
    print_info "  - Claude Code CLI: Use your Claude subscription"
    print_info "  - Copilot CLI: Use your GitHub Copilot subscription"

    if ask_yes_no "Install Claude Code CLI?" "n"; then
        dev_install_claude_cli
    else
        track_install "Claude Code CLI" "skipped"
    fi

    if ask_yes_no "Install GitHub Copilot CLI?" "n"; then
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
    print_info "Playwright browser (Chromium):"
    print_info "  Purpose: Browser automation for JavaScript-rendered pages (Twitter, SPAs)"

    project_root=$(get_project_root)
    cd "$project_root"

    # Ask user consent before downloading
    if ! ask_yes_no "Download Chromium browser?" "y"; then
        print_info "Skipping Playwright browser installation"
        track_install "Playwright Browser" "skipped"
        return 2
    fi

    print_info "Downloading Chromium browser..."
    browser_installed=false

    # Prefer uv run in dev environment (uses .venv)
    if command -v uv >/dev/null 2>&1; then
        if uv run playwright install chromium 2>/dev/null; then
            print_success "Chromium browser downloaded successfully"
            browser_installed=true
        fi
    fi

    # Fallback to Python module
    if [ "$browser_installed" = false ] && [ -n "$PYTHON_CMD" ]; then
        if "$PYTHON_CMD" -m playwright install chromium 2>/dev/null; then
            print_success "Chromium browser downloaded successfully"
            browser_installed=true
        fi
    fi

    if [ "$browser_installed" = false ]; then
        print_warning "Playwright browser installation failed"
        print_info "You can install later with: uv run playwright install chromium"
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
                if command -v uv >/dev/null 2>&1; then
                    if uv run playwright install-deps chromium 2>/dev/null; then
                        print_success "System dependencies installed successfully"
                        track_install "Playwright Browser" "installed"
                        return 0
                    fi
                fi
                # Fallback to Python module
                if [ -n "$PYTHON_CMD" ]; then
                    if "$PYTHON_CMD" -m playwright install-deps chromium 2>/dev/null; then
                        print_success "System dependencies installed successfully"
                        track_install "Playwright Browser" "installed"
                        return 0
                    fi
                fi
                print_warning "System dependencies installation failed"
                print_info "Manual install: sudo playwright install-deps chromium"
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

# Install LibreOffice (optional, for legacy Office files)
dev_install_libreoffice() {
    print_info "Checking LibreOffice installation..."
    print_info "  Purpose: Convert legacy Office files (.doc, .ppt, .xls)"

    if command -v soffice >/dev/null 2>&1; then
        version=$(soffice --version 2>/dev/null | head -n1)
        print_success "LibreOffice installed: $version"
        track_install "LibreOffice" "installed"
        return 0
    fi

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
        return 2
    fi

    print_info "Installing LibreOffice..."

    case "$(uname)" in
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                if brew install --cask libreoffice; then
                    print_success "LibreOffice installed via Homebrew"
                    track_install "LibreOffice" "installed"
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
                if sudo apt update && sudo apt install -y libreoffice; then
                    print_success "LibreOffice installed via apt"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            elif [ -f /etc/fedora-release ]; then
                print_info "Installing via dnf (requires sudo)..."
                if sudo dnf install -y libreoffice; then
                    print_success "LibreOffice installed via dnf"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            elif [ -f /etc/arch-release ]; then
                print_info "Installing via pacman (requires sudo)..."
                if sudo pacman -S --noconfirm libreoffice-fresh; then
                    print_success "LibreOffice installed via pacman"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            else
                print_error "Unknown Linux distribution"
                print_info "Please install LibreOffice manually"
            fi
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
            else
                print_info "  Use your package manager to install libreoffice"
            fi
            ;;
    esac
    track_install "LibreOffice" "failed"
    return 1
}

# Install FFmpeg (optional, for audio/video file processing)
dev_install_ffmpeg() {
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
                print_info "Please install FFmpeg manually"
            fi
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
    esac
    track_install "FFmpeg" "failed"
    return 1
}

# Print completion message
dev_print_completion() {
    project_root=$(get_project_root)

    printf "\n"
    printf "${GREEN}✓${NC} ${BOLD}Development environment setup complete!${NC}\n"
    printf "\n"
    printf "  ${BOLD}Activate virtual environment:${NC}\n"
    printf "    ${YELLOW}source %s/.venv/bin/activate${NC}\n" "$project_root"
    printf "\n"
    printf "  ${BOLD}Run tests:${NC}\n"
    printf "    ${YELLOW}uv run pytest${NC}\n"
    printf "\n"
    printf "  ${BOLD}Run CLI:${NC}\n"
    printf "    ${YELLOW}uv run markitai --help${NC}\n"
    printf "\n"
    return 0
}

# ============================================================
# Main Logic
# ============================================================

main() {
    # Security check: warn if running as root
    warn_if_root

    # Welcome message
    print_welcome_dev

    print_header "Markitai Dev Environment Setup"

    # Step 1: Detect/install UV (required, also manages Python)
    print_step 1 5 "Detecting UV package manager..."
    if ! dev_install_uv; then
        print_summary
        exit 1
    fi

    # Step 2: Detect/install Python (auto-installed via uv)
    print_step 2 5 "Detecting Python..."
    if ! lib_detect_python; then
        exit 1
    fi

    # Step 3: Sync dependencies (includes all extras: browser, claude-agent, copilot)
    print_step 3 5 "Syncing development dependencies..."
    if ! sync_dependencies; then
        print_summary
        exit 1
    fi
    track_install "Python dependencies" "installed"
    track_install "Claude Agent SDK" "installed"
    track_install "Copilot SDK" "installed"

    # Install Playwright browser (required for SPA/JS-rendered pages)
    dev_install_playwright_browser

    # Install LibreOffice (optional, for legacy Office files)
    dev_install_libreoffice

    # Install FFmpeg (optional, for audio/video files)
    dev_install_ffmpeg

    # Step 4: Install pre-commit
    print_step 4 5 "Configuring pre-commit..."
    if install_precommit; then
        track_install "pre-commit hooks" "installed"
    fi

    # Step 5: Optional components - LLM CLI tools
    print_step 5 5 "Optional: LLM CLI tools"
    if ask_yes_no "Install LLM CLI tools (Claude Code / Copilot)?" "n"; then
        dev_install_llm_clis
    else
        print_info "Skipping LLM CLI installation"
        track_install "Claude Code CLI" "skipped"
        track_install "Copilot CLI" "skipped"
    fi

    # Print summary
    print_summary

    # Complete
    dev_print_completion
}

# Run main function
main
