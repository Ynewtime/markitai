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
MAGENTA='\033[0;35m'
GRAY='\033[0;90m'
NC='\033[0m'
BOLD='\033[1m'
DIM='\033[2m'

# ============================================================
# Clack-style Visual Components
# Inspired by @clack/prompts - beautiful CLI with guide lines
# ============================================================

# Guide line characters
S_BAR="│"
S_BAR_H="─"
S_CORNER_TOP_RIGHT="┐"
S_CORNER_BOTTOM_RIGHT="┘"
S_CONNECT_LEFT="├"
S_STEP_ACTIVE="◆"
S_STEP_SUBMIT="◇"
S_RADIO_ACTIVE="●"
S_RADIO_INACTIVE="○"
S_CHECKBOX_ACTIVE="◼"
S_CHECKBOX_INACTIVE="◻"

# Session intro - start of CLI flow
# Usage: clack_intro "Title"
clack_intro() {
    printf "\n"
    printf "${GRAY}┌${NC}  ${BOLD}%s${NC}\n" "$1"
    printf "${GRAY}│${NC}\n"
}

# Session outro - end of CLI flow
# Usage: clack_outro "Message"
clack_outro() {
    printf "${GRAY}│${NC}\n"
    printf "${GRAY}└${NC}  ${GREEN}%s${NC}\n" "$1"
    printf "\n"
}

# Section header with active marker
# Usage: clack_section "Section title"
clack_section() {
    printf "${GRAY}│${NC}\n"
    printf "${MAGENTA}◆${NC}  ${BOLD}%s${NC}\n" "$1"
}

# Log with guide line - success
# Usage: clack_success "Message"
clack_success() {
    printf "${GRAY}│${NC}  ${GREEN}✓${NC} %s\n" "$1"
}

# Log with guide line - error
# Usage: clack_error "Message"
clack_error() {
    printf "${GRAY}│${NC}  ${RED}✗${NC} %s\n" "$1"
}

# Log with guide line - warning
# Usage: clack_warn "Message"
clack_warn() {
    printf "${GRAY}│${NC}  ${YELLOW}!${NC} %s\n" "$1"
}

# Log with guide line - info
# Usage: clack_info "Message"
clack_info() {
    printf "${GRAY}│${NC}  ${CYAN}→${NC} %s\n" "$1"
}

# Log with guide line - skipped
# Usage: clack_skip "Message"
clack_skip() {
    printf "${GRAY}│${NC}  ${GRAY}○${NC} ${GRAY}%s${NC}\n" "$1"
}

# Log with guide line - plain text
# Usage: clack_log "Message"
clack_log() {
    printf "${GRAY}│${NC}  %s\n" "$1"
}

# Spinner with guide line
# Usage: clack_spinner "message" command args...
# Shows spinner while command runs, then shows result
clack_spinner() {
    _cs_message="$1"
    shift

    # Spinner frames (ASCII compatible)
    _cs_pid=""

    # Start spinner in background
    (
        while true; do
            for _cs_frame in '|' '/' '-' '\'; do
                printf "\r${GRAY}│${NC}  ${CYAN}%s${NC} %s" "$_cs_frame" "$_cs_message"
                sleep 0.1 2>/dev/null || sleep 1
            done
        done
    ) &
    _cs_pid=$!

    # Run the actual command
    "$@" >/dev/null 2>&1
    _cs_status=$?

    # Stop spinner
    kill $_cs_pid 2>/dev/null
    wait $_cs_pid 2>/dev/null

    # Clear spinner line
    printf "\r\033[K"

    return $_cs_status
}

# Confirm prompt with guide line
# Usage: clack_confirm "Question?" "y|n"
# Returns: 0 for yes, 1 for no
clack_confirm() {
    _cc_prompt="$1"
    _cc_default="$2"

    if [ "$_cc_default" = "y" ]; then
        _cc_hint="${BOLD}Y${NC}${GRAY}/n${NC}"
    else
        _cc_hint="${GRAY}y/${NC}${BOLD}N${NC}"
    fi

    printf "${GRAY}│${NC}\n"
    printf "${CYAN}◇${NC}  %s ${GRAY}[%b]${NC} " "$_cc_prompt" "$_cc_hint"

    # Read from /dev/tty for piped execution support
    if [ -t 0 ]; then
        read -r _cc_answer
    else
        read -r _cc_answer < /dev/tty
        printf "\n"
    fi

    if [ -z "$_cc_answer" ]; then
        _cc_answer="$_cc_default"
    fi

    case "$_cc_answer" in
        [Yy]*) return 0 ;;
        *) return 1 ;;
    esac
}

# Note/message box with guide line
# Usage: clack_note "title" "line1" "line2" ...
# Or:    clack_note "title" <<EOF
#        line1
#        line2
#        EOF
clack_note() {
    _cn_title="$1"
    shift

    printf "${GRAY}│${NC}\n"
    printf "${GRAY}│${NC}  ${GRAY}╭─${NC} ${BOLD}%s${NC}\n" "$_cn_title"

    # If arguments provided, use them as lines
    if [ $# -gt 0 ]; then
        for _cn_line in "$@"; do
            printf "${GRAY}│${NC}  ${GRAY}│${NC}  %s\n" "$_cn_line"
        done
    else
        # Read from stdin (heredoc support)
        while IFS= read -r _cn_line; do
            printf "${GRAY}│${NC}  ${GRAY}│${NC}  %s\n" "$_cn_line"
        done
    fi

    printf "${GRAY}│${NC}  ${GRAY}╰─${NC}\n"
}

# Cancel message
# Usage: clack_cancel "Message"
clack_cancel() {
    printf "${GRAY}│${NC}\n"
    printf "${GRAY}└${NC}  ${RED}%s${NC}\n" "$1"
    printf "\n"
}

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

# Run command silently, only show output on error
run_silent() {
    _rs_output=$("$@" 2>&1)
    _rs_status=$?
    if [ $_rs_status -ne 0 ]; then
        printf "%s\n" "$_rs_output" >&2
    fi
    return $_rs_status
}

# Simplified status output (single line)
print_status() {
    _ps_status="$1"
    _ps_message="$2"
    case "$_ps_status" in
        ok)      printf "  ${GREEN}✓${NC} %s\n" "$_ps_message" ;;
        skip)    printf "  ${YELLOW}○${NC} %s\n" "$_ps_message" ;;
        fail)    printf "  ${RED}✗${NC} %s\n" "$_ps_message" ;;
        info)    printf "  ${CYAN}→${NC} %s\n" "$_ps_message" ;;
    esac
}

# ============================================================
# User Interaction
# ============================================================

# Run command with spinner animation
# Usage: run_with_spinner "message" command args...
# Shows spinner while command runs, replaces with result when done
run_with_spinner() {
    _rws_message="$1"
    shift
    _rws_pid=""

    # Start spinner in background (ASCII compatible: | / - \)
    (
        while true; do
            for _rws_char in '|' '/' '-' '\'; do
                printf "\r  ${CYAN}%s${NC} %s" "$_rws_char" "$_rws_message"
                sleep 0.1 2>/dev/null || sleep 1
            done
        done
    ) &
    _rws_pid=$!

    # Run the actual command
    "$@" >/dev/null 2>&1
    _rws_status=$?

    # Stop spinner
    kill $_rws_pid 2>/dev/null
    wait $_rws_pid 2>/dev/null

    # Clear spinner line
    printf "\r\033[K"

    return $_rws_status
}

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
            print_status ok "Python $ver"
            return 0
        fi

        # Not found, auto-install
        print_status info "Installing Python 3.13..."
        if uv python install 3.13 >/dev/null 2>&1; then
            uv_python=$(uv python find 3.13 2>/dev/null)
            if [ -n "$uv_python" ] && [ -x "$uv_python" ]; then
                PYTHON_CMD="$uv_python"
                ver=$("$uv_python" -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>/dev/null)
                print_status ok "Python $ver installed"
                return 0
            fi
        fi
        print_status fail "Python 3.13 installation failed"
    else
        print_status fail "uv not installed"
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
# Returns: 0 on success, 1 on failure
lib_install_uv() {
    if command -v uv >/dev/null 2>&1; then
        version=$(uv --version 2>/dev/null | head -n1 | cut -d' ' -f2)
        print_status ok "uv $version"
        track_install "uv" "installed"
        return 0
    fi

    print_status info "Installing uv..."

    # Check curl availability
    if ! command -v curl >/dev/null 2>&1; then
        print_status fail "curl not found"
        track_install "uv" "failed"
        return 1
    fi

    # Build install URL (with optional version)
    if [ -n "$UV_VERSION" ]; then
        uv_url="https://astral.sh/uv/$UV_VERSION/install.sh"
    else
        uv_url="https://astral.sh/uv/install.sh"
    fi

    if curl -LsSf "$uv_url" 2>/dev/null | sh >/dev/null 2>&1; then
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        if command -v uv >/dev/null 2>&1; then
            version=$(uv --version 2>/dev/null | head -n1 | cut -d' ' -f2)
            print_status ok "uv $version installed"
            track_install "uv" "installed"
            return 0
        fi
    fi

    print_status fail "uv installation failed"
    track_install "uv" "failed"
    return 1
}

# Install markitai
# Requires: PYTHON_CMD to be set
# Returns: 0 on success, 1 on failure
lib_install_markitai() {
    print_status info "Installing markitai..."

    # Build package spec with optional version
    if [ -n "$MARKITAI_VERSION" ]; then
        pkg="markitai[browser]==$MARKITAI_VERSION"
    else
        pkg="markitai[browser]"
    fi

    # Prefer uv tool install (recommended, installs to ~/.local/bin)
    if command -v uv >/dev/null 2>&1; then
        if uv tool install "$pkg" --python "$PYTHON_CMD" --upgrade >/dev/null 2>&1; then
            export PATH="$HOME/.local/bin:$PATH"
            version=$(markitai --version 2>/dev/null || echo "installed")
            print_status ok "markitai $version"
            track_install "markitai" "installed"
            return 0
        fi
    fi

    # Fallback to pipx
    if command -v pipx >/dev/null 2>&1; then
        if pipx install "$pkg" --python "$PYTHON_CMD" --force >/dev/null 2>&1; then
            version=$(markitai --version 2>/dev/null || echo "installed")
            print_status ok "markitai $version"
            track_install "markitai" "installed"
            return 0
        fi
    fi

    # Fallback to pip --user
    if "$PYTHON_CMD" -m pip install --user --upgrade "$pkg" >/dev/null 2>&1; then
        export PATH="$HOME/.local/bin:$PATH"
        version=$(markitai --version 2>/dev/null || echo "installed")
        print_status ok "markitai $version"
        track_install "markitai" "installed"
        return 0
    fi

    print_status fail "markitai installation failed"
    track_install "markitai" "failed"
    return 1
}

# Check if Playwright Chromium browser is installed
# Returns: 0 if installed, 1 if not
lib_detect_playwright_browser() {
    # Check common Playwright browser locations
    # macOS: ~/Library/Caches/ms-playwright
    # Linux: ~/.cache/ms-playwright
    # Windows: %LOCALAPPDATA%\ms-playwright (not applicable in shell)

    playwright_cache=""
    case "$(uname)" in
        Darwin)
            playwright_cache="$HOME/Library/Caches/ms-playwright"
            ;;
        Linux)
            playwright_cache="$HOME/.cache/ms-playwright"
            ;;
    esac

    if [ -n "$playwright_cache" ] && [ -d "$playwright_cache" ]; then
        # Check for chromium directory
        if ls "$playwright_cache"/chromium-* >/dev/null 2>&1; then
            return 0
        fi
    fi

    return 1
}

# Install Playwright browser (Chromium) and system dependencies
# Requires: markitai/playwright to be installed
# Returns: 0 on success, 1 on failure, 2 if skipped
lib_install_playwright_browser() {
    browser_installed=false

    # Get playwright path from markitai's uv tool environment
    markitai_playwright=""
    if command -v uv >/dev/null 2>&1; then
        uv_tool_dir=$(uv tool dir 2>/dev/null)
        if [ -n "$uv_tool_dir" ]; then
            markitai_playwright="$uv_tool_dir/markitai/bin/playwright"
        fi
    fi
    # Fallback to default path
    if [ -z "$markitai_playwright" ] || [ ! -x "$markitai_playwright" ]; then
        markitai_playwright="$HOME/.local/share/uv/tools/markitai/bin/playwright"
    fi

    # Install browser with spinner (network operation can be slow)
    if [ -x "$markitai_playwright" ]; then
        if run_with_spinner "Downloading Chromium..." "$markitai_playwright" install chromium; then
            browser_installed=true
        fi
    fi

    # Fallback to Python module
    if [ "$browser_installed" = false ] && [ -n "$PYTHON_CMD" ]; then
        if run_with_spinner "Downloading Chromium..." "$PYTHON_CMD" -m playwright install chromium; then
            browser_installed=true
        fi
    fi

    if [ "$browser_installed" = false ]; then
        print_status fail "Playwright browser"
        track_install "Playwright Browser" "failed"
        return 1
    fi

    # On Linux, install system dependencies silently
    if [ "$(uname)" = "Linux" ]; then
        if [ -f /etc/arch-release ]; then
            arch_deps="nss nspr at-spi2-core cups libdrm mesa alsa-lib libxcomposite libxdamage libxrandr libxkbcommon pango cairo noto-fonts noto-fonts-cjk noto-fonts-emoji ttf-liberation"
            sudo pacman -S --noconfirm --needed $arch_deps >/dev/null 2>&1 || true
        elif command -v apt-get >/dev/null 2>&1; then
            if [ -x "$markitai_playwright" ]; then
                "$markitai_playwright" install-deps chromium >/dev/null 2>&1 || true
            elif [ -n "$PYTHON_CMD" ]; then
                "$PYTHON_CMD" -m playwright install-deps chromium >/dev/null 2>&1 || true
            fi
        fi
    fi

    print_status ok "Playwright browser"
    track_install "Playwright Browser" "installed"
    return 0
}

# Install LibreOffice (optional)
# Returns: 0 on success, 1 on failure
lib_install_libreoffice() {
    # Check if already installed
    if command -v soffice >/dev/null 2>&1 || command -v libreoffice >/dev/null 2>&1; then
        print_status ok "LibreOffice"
        track_install "LibreOffice" "installed"
        return 0
    fi

    # Install silently based on OS
    case "$(uname)" in
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                if brew install --cask libreoffice >/dev/null 2>&1; then
                    print_status ok "LibreOffice"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            fi
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                if sudo apt update >/dev/null 2>&1 && sudo apt install -y libreoffice >/dev/null 2>&1; then
                    print_status ok "LibreOffice"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            elif [ -f /etc/fedora-release ]; then
                if sudo dnf install -y libreoffice >/dev/null 2>&1; then
                    print_status ok "LibreOffice"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            elif [ -f /etc/arch-release ]; then
                if sudo pacman -S --noconfirm libreoffice-fresh >/dev/null 2>&1; then
                    print_status ok "LibreOffice"
                    track_install "LibreOffice" "installed"
                    return 0
                fi
            fi
            ;;
    esac

    print_status fail "LibreOffice"
    track_install "LibreOffice" "failed"
    return 1
}

# Install FFmpeg (optional)
# Returns: 0 on success, 1 on failure
lib_install_ffmpeg() {
    # Check if already installed
    if command -v ffmpeg >/dev/null 2>&1; then
        print_status ok "FFmpeg"
        track_install "FFmpeg" "installed"
        return 0
    fi

    # Install silently based on OS
    case "$(uname)" in
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                if brew install ffmpeg >/dev/null 2>&1; then
                    print_status ok "FFmpeg"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            fi
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                if sudo apt update >/dev/null 2>&1 && sudo apt install -y ffmpeg >/dev/null 2>&1; then
                    print_status ok "FFmpeg"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            elif [ -f /etc/fedora-release ]; then
                if sudo dnf install -y ffmpeg >/dev/null 2>&1; then
                    print_status ok "FFmpeg"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            elif [ -f /etc/arch-release ]; then
                if sudo pacman -S --noconfirm ffmpeg >/dev/null 2>&1; then
                    print_status ok "FFmpeg"
                    track_install "FFmpeg" "installed"
                    return 0
                fi
            fi
            ;;
    esac

    print_status fail "FFmpeg"
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
        if uv tool install "$pkg" --python "$PYTHON_CMD" --upgrade >/dev/null 2>&1; then
            return 0
        fi
    fi

    # Fallback to pipx
    if command -v pipx >/dev/null 2>&1; then
        if pipx install "$pkg" --python "$PYTHON_CMD" --force >/dev/null 2>&1; then
            return 0
        fi
    fi

    # Fallback to pip --user
    if "$PYTHON_CMD" -m pip install --user --upgrade "$pkg" >/dev/null 2>&1; then
        return 0
    fi

    return 1
}

# Install Claude Code CLI
# Returns: 0 on success, 1 on failure
lib_install_claude_cli() {
    # Check if already installed
    if command -v claude >/dev/null 2>&1; then
        print_status ok "Claude Code CLI"
        track_install "Claude Code CLI" "installed"
        return 0
    fi

    # Try official install script
    if curl -fsSL "https://claude.ai/install.sh" 2>/dev/null | bash >/dev/null 2>&1; then
        if command -v claude >/dev/null 2>&1; then
            print_status ok "Claude Code CLI"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    fi

    # Fallback: npm/pnpm
    if command -v pnpm >/dev/null 2>&1; then
        if pnpm add -g @anthropic-ai/claude-code >/dev/null 2>&1; then
            print_status ok "Claude Code CLI"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    elif command -v npm >/dev/null 2>&1; then
        if npm install -g @anthropic-ai/claude-code >/dev/null 2>&1; then
            print_status ok "Claude Code CLI"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    fi

    print_status fail "Claude Code CLI"
    track_install "Claude Code CLI" "failed"
    return 1
}

# Install GitHub Copilot CLI
# Returns: 0 on success, 1 on failure
lib_install_copilot_cli() {
    # Check if already installed
    if command -v copilot >/dev/null 2>&1; then
        print_status ok "Copilot CLI"
        track_install "Copilot CLI" "installed"
        return 0
    fi

    # Try official install script
    if curl -fsSL "https://gh.io/copilot-install" 2>/dev/null | bash >/dev/null 2>&1; then
        if command -v copilot >/dev/null 2>&1; then
            print_status ok "Copilot CLI"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    # Fallback: npm/pnpm
    if command -v pnpm >/dev/null 2>&1; then
        if pnpm add -g @github/copilot >/dev/null 2>&1; then
            print_status ok "Copilot CLI"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    elif command -v npm >/dev/null 2>&1; then
        if npm install -g @github/copilot >/dev/null 2>&1; then
            print_status ok "Copilot CLI"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    print_status fail "Copilot CLI"
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
    printf "  ${BOLD}Get started:${NC}\n"
    printf "    ${CYAN}markitai -I${NC}          Interactive mode\n"
    printf "    ${CYAN}markitai file.pdf${NC}   Convert a file\n"
    printf "    ${CYAN}markitai --help${NC}     Show all options\n"
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
