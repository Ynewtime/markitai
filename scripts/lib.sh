#!/bin/sh
# Markitai Setup Library - Common Functions
# Supports bash/zsh/dash and other POSIX-compatible shells

# ============================================================
# Version Variables (can be overridden via environment)
# ============================================================
MARKITAI_VERSION="${MARKITAI_VERSION:-}"
# Lock agent-browser to 0.7.6 due to daemon startup bug in 0.8.x on Windows
AGENT_BROWSER_VERSION="${AGENT_BROWSER_VERSION:-0.7.6}"
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
    component="$1"
    status="$2"
    case "$status" in
        installed)
            INSTALLED_COMPONENTS="${INSTALLED_COMPONENTS}${component}|"
            ;;
        skipped)
            SKIPPED_COMPONENTS="${SKIPPED_COMPONENTS}${component}|"
            ;;
        failed)
            FAILED_COMPONENTS="${FAILED_COMPONENTS}${component}|"
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
    printf "    ${YELLOW}•${NC} agent-browser - Browser automation for JS-rendered pages\n"
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
    printf "    ${YELLOW}•${NC} agent-browser - Browser automation\n"
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
ask_yes_no() {
    prompt="$1"
    default="$2"

    if [ "$default" = "y" ]; then
        hint="[Y/n, default: Yes]"
    else
        hint="[y/N, default: No]"
    fi

    printf "  ${YELLOW}?${NC} %s %s " "$prompt" "$hint"
    read -r answer

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

# Detect Python (requires 3.11-3.13, 3.14+ not supported)
# Sets: PYTHON_CMD
# Returns: 0 if found, 1 if not
lib_detect_python() {
    # Try different Python commands (prefer 3.11-3.13)
    for cmd in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            # Use Python2-compatible syntax (no f-string)
            ver=$("$cmd" -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>/dev/null)
            major=$("$cmd" -c "import sys; print(sys.version_info[0])" 2>/dev/null)
            minor=$("$cmd" -c "import sys; print(sys.version_info[1])" 2>/dev/null)

            # Validate: must be numeric
            case "$major" in
                ''|*[!0-9]*) continue ;;
            esac
            case "$minor" in
                ''|*[!0-9]*) continue ;;
            esac

            # Check version range: 3.11 <= version < 3.14
            if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && [ "$minor" -le 13 ]; then
                PYTHON_CMD="$cmd"
                print_success "Python $ver installed ($cmd)"
                return 0
            elif [ "$major" -eq 3 ] && [ "$minor" -ge 14 ]; then
                print_warning "Python $ver detected, but onnxruntime doesn't support Python 3.14+"
            fi
        fi
    done

    print_error "Python 3.11-3.13 not found"
    printf "\n"
    print_warning "Please install Python 3.13 (recommended) or 3.11/3.12:"
    print_info "Download: https://www.python.org/downloads/"
    if [ "$OS_TYPE" = "Darwin" ]; then
        if [ "$IS_MACOS_ARM" = true ]; then
            print_info "macOS (Apple Silicon): brew install python@3.13"
            print_info "  Homebrew installs native ARM64 binaries"
        else
            print_info "macOS (Intel): brew install python@3.13"
        fi
    elif [ "$OS_TYPE" = "Linux" ]; then
        print_info "Ubuntu/Debian: sudo apt install python3.13"
        print_info "Fedora: sudo dnf install python3.13"
    fi
    print_info "pyenv: pyenv install 3.13"
    print_info "Note: onnxruntime doesn't support Python 3.14 yet"
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

    print_error "UV not installed"

    if ! ask_yes_no "Install UV automatically?" "n"; then
        print_info "Skipping UV installation"
        print_warning "markitai recommends using UV for installation"
        track_install "uv" "skipped"
        return 2  # Skipped
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
    if [ -n "$MARKITAI_VERSION" ]; then
        pkg="markitai[all]==$MARKITAI_VERSION"
        print_info "Installing version: $MARKITAI_VERSION"
    else
        pkg="markitai[all]"
    fi

    # Prefer uv tool install (recommended, installs to ~/.local/bin)
    if command -v uv >/dev/null 2>&1; then
        # CRITICAL: Use --python to specify the detected Python version
        if uv tool install "$pkg" --python "$PYTHON_CMD" 2>/dev/null; then
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
        if pipx install "$pkg" --python "$PYTHON_CMD"; then
            version=$(markitai --version 2>/dev/null || echo "installed")
            print_success "markitai $version installed successfully"
            track_install "markitai" "installed"
            return 0
        fi
    fi

    # Fallback to pip --user
    if "$PYTHON_CMD" -m pip install --user "$pkg" 2>/dev/null; then
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

# Install agent-browser
# Returns: 0 on success, 1 on failure
lib_install_agent_browser() {
    print_info "Detecting Node.js..."

    if ! lib_detect_node; then
        print_warning "Skipping agent-browser installation (requires Node.js)"
        track_install "agent-browser" "skipped"
        return 1
    fi

    print_info "Installing agent-browser..."
    print_info "  Purpose: Browser automation for JavaScript-rendered pages"
    print_info "  Size: ~150MB (includes Chromium)"

    # Build package spec with optional version
    if [ -n "$AGENT_BROWSER_VERSION" ]; then
        pkg="agent-browser@$AGENT_BROWSER_VERSION"
        print_info "Installing version: $AGENT_BROWSER_VERSION"
    else
        pkg="agent-browser"
    fi

    # Try npm first, then pnpm
    install_success=false
    if command -v npm >/dev/null 2>&1; then
        print_info "Installing via npm..."
        if npm install -g "$pkg"; then
            install_success=true
        fi
    fi

    if [ "$install_success" = false ] && command -v pnpm >/dev/null 2>&1; then
        print_info "Installing via pnpm..."
        if pnpm add -g "$pkg"; then
            install_success=true
        fi
    fi

    if [ "$install_success" = true ]; then
        # Verify installation
        if ! command -v agent-browser >/dev/null 2>&1; then
            print_warning "agent-browser installed but not in PATH"
            # Get global bin directory for PATH help
            global_bin=""
            if command -v pnpm >/dev/null 2>&1; then
                global_bin=$(pnpm config get global-bin-dir 2>/dev/null)
                if [ -z "$global_bin" ]; then
                    # pnpm bin -g returns the actual bin directory
                    global_bin=$(pnpm bin -g 2>/dev/null)
                fi
            fi
            if [ -z "$global_bin" ] && command -v npm >/dev/null 2>&1; then
                npm_prefix=$(npm config get prefix 2>/dev/null)
                if [ -n "$npm_prefix" ]; then
                    global_bin="$npm_prefix/bin"
                fi
            fi
            if [ -n "$global_bin" ]; then
                print_path_help "$global_bin"
            fi
            track_install "agent-browser" "installed"
            return 1
        fi

        print_success "agent-browser installed successfully"
        track_install "agent-browser" "installed"

        # Chromium download (default: No)
        if ask_yes_no "Download Chromium browser?" "n"; then
            print_info "Downloading Chromium..."

            # Detect OS
            os_type=$(uname -s)

            if [ "$os_type" = "Linux" ]; then
                # Linux: system dependencies (default: No)
                if ask_yes_no "Also install system dependencies (requires sudo)?" "n"; then
                    agent-browser install --with-deps
                else
                    agent-browser install
                fi
            else
                agent-browser install
            fi

            print_success "Chromium download complete"
            track_install "Chromium" "installed"
        else
            print_info "Skipping Chromium download"
            print_info "You can install later: agent-browser install"
            track_install "Chromium" "skipped"
        fi

        return 0
    else
        print_error "agent-browser installation failed"
        if ! check_network; then
            print_network_error
        else
            print_info "Manual install: npm install -g agent-browser"
        fi
        track_install "agent-browser" "failed"
        return 1
    fi
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
    if ! check_network; then
        print_network_error
    else
        print_info "Manual install options:"
        print_info "  pnpm: pnpm add -g @anthropic-ai/claude-code"
        print_info "  brew: brew install claude-code"
        print_info "  Docs: https://code.claude.com/docs/en/setup"
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
    if ! check_network; then
        print_network_error
    else
        print_info "Manual install options:"
        print_info "  pnpm: pnpm add -g @github/copilot"
        print_info "  brew: brew install copilot-cli"
        print_info "  curl: curl -fsSL https://gh.io/copilot-install | bash"
    fi
    track_install "Copilot CLI" "failed"
    return 1
}

# Initialize markitai config
lib_init_config() {
    print_info "Initializing configuration..."

    if command -v markitai >/dev/null 2>&1; then
        if markitai config init 2>/dev/null; then
            print_success "Configuration initialized"
        fi
    fi
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

    printf "  ${BOLD}Documentation:${NC} https://markitai.dev\n"
    printf "  ${BOLD}Issues:${NC} https://github.com/Ynewtime/markitai/issues\n"
    printf "\n"
}
