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

# Install agent-browser for development
dev_install_agent_browser() {
    print_info "Detecting Node.js..."

    if ! lib_detect_node; then
        print_warning "Skipping agent-browser installation (requires Node.js)"
        track_install "agent-browser" "skipped"
        return 1
    fi

    print_info "Installing agent-browser..."

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
            print_info "You may need to add global bin to PATH:"
            print_info "  pnpm bin -g  # or: npm config get prefix"
            track_install "agent-browser" "installed"
            return 1
        fi

        print_success "agent-browser installed successfully"
        track_install "agent-browser" "installed"

        # Chromium download (default: No)
        if ask_yes_no "Download Chromium browser?" "n"; then
            print_info "Downloading Chromium..."

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
        print_info "Manual install: npm install -g agent-browser"
        track_install "agent-browser" "failed"
        return 1
    fi
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
}

# Install LLM provider SDKs (optional extras)
dev_install_provider_sdks() {
    project_root=$(get_project_root)
    cd "$project_root"

    print_info "Python SDKs for programmatic LLM access:"
    print_info "  - Claude Agent SDK (requires Claude Code CLI)"
    print_info "  - GitHub Copilot SDK (requires Copilot CLI)"

    if ask_yes_no "Install Claude Agent SDK?" "n"; then
        print_info "Installing claude-agent-sdk..."
        if uv sync --extra claude-agent; then
            print_success "Claude Agent SDK installed"
            track_install "Claude Agent SDK" "installed"
        else
            print_warning "Claude Agent SDK installation failed"
            track_install "Claude Agent SDK" "failed"
        fi
    else
        track_install "Claude Agent SDK" "skipped"
    fi

    if ask_yes_no "Install GitHub Copilot SDK?" "n"; then
        print_info "Installing github-copilot-sdk..."
        if uv sync --extra copilot; then
            print_success "GitHub Copilot SDK installed"
            track_install "Copilot SDK" "installed"
        else
            print_warning "GitHub Copilot SDK installation failed"
            track_install "Copilot SDK" "failed"
        fi
    else
        track_install "Copilot SDK" "skipped"
    fi
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

    # Step 1: Detect Python
    print_step 1 7 "Detecting Python..."
    if ! lib_detect_python; then
        exit 1
    fi

    # Step 2: Detect/install UV (required for developer edition)
    print_step 2 7 "Detecting UV package manager..."
    if ! dev_install_uv; then
        print_summary
        exit 1
    fi

    # Step 3: Sync dependencies
    print_step 3 7 "Syncing development dependencies..."
    if ! sync_dependencies; then
        print_summary
        exit 1
    fi
    track_install "Python dependencies" "installed"

    # Step 4: Install pre-commit
    print_step 4 7 "Configuring pre-commit..."
    if install_precommit; then
        track_install "pre-commit hooks" "installed"
    fi

    # Step 5: Optional components - agent-browser
    print_step 5 7 "Optional: Browser automation"
    if ask_yes_no "Install browser automation support (agent-browser)?" "n"; then
        dev_install_agent_browser
    else
        print_info "Skipping agent-browser installation"
        track_install "agent-browser" "skipped"
    fi

    # Step 6: Optional components - LLM CLI tools
    print_step 6 7 "Optional: LLM CLI tools"
    if ask_yes_no "Install LLM CLI tools (Claude Code / Copilot)?" "n"; then
        dev_install_llm_clis
    else
        print_info "Skipping LLM CLI installation"
        track_install "Claude Code CLI" "skipped"
        track_install "Copilot CLI" "skipped"
    fi

    # Step 7: Optional components - LLM provider SDKs
    print_step 7 7 "Optional: LLM Python SDKs"
    if ask_yes_no "Install LLM Python SDKs (claude-agent-sdk / github-copilot-sdk)?" "n"; then
        dev_install_provider_sdks
    else
        print_info "Skipping LLM Python SDK installation"
        print_info "Install later: uv sync --all-extras"
    fi

    # Print summary
    print_summary

    # Complete
    dev_print_completion
}

# Run main function
main
