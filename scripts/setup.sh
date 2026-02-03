#!/bin/sh
# Markitai Setup Script (User Edition)
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

    # Remote execution (curl | sh) - download lib.sh
    if ! command -v curl >/dev/null 2>&1; then
        echo "Error: curl is required for remote execution"
        exit 1
    fi

    TEMP_LIB=$(mktemp)
    trap 'rm -f "$TEMP_LIB" 2>/dev/null' EXIT INT TERM

    if curl -fsSL "$LIB_BASE_URL/lib.sh" -o "$TEMP_LIB"; then
        . "$TEMP_LIB"
        return 0
    else
        echo "Error: Failed to download lib.sh"
        exit 1
    fi
}

load_library

# ============================================================
# Main Logic
# ============================================================

main() {
    # Security check: warn if running as root
    warn_if_root

    # Welcome message
    print_welcome_user

    print_header "Markitai Setup Wizard"

    # Step 1: Detect/install UV (required, manages Python)
    print_step 1 5 "Detecting UV package manager..."
    if ! lib_install_uv; then
        print_summary
        exit 1
    fi

    # Step 2: Detect/install Python (auto-installed via uv)
    print_step 2 5 "Detecting Python..."
    if ! lib_detect_python; then
        exit 1
    fi

    # Step 3: Install markitai
    print_step 3 5 "Installing markitai..."
    if ! lib_install_markitai; then
        print_summary
        exit 1
    fi

    # Install Playwright browser (required for SPA/JS-rendered pages)
    lib_install_playwright_browser

    # Install LibreOffice (optional, for legacy Office files)
    lib_install_libreoffice

    # Install FFmpeg (optional, for audio/video files)
    lib_install_ffmpeg

    # Step 4: Optional - LLM CLI tools
    print_step 4 5 "Optional: LLM CLI tools"
    print_info "LLM CLI tools provide local authentication for AI providers"
    if ask_yes_no "Install Claude Code CLI?" "n"; then
        if lib_install_claude_cli; then
            # Install Claude Agent SDK for programmatic access
            lib_install_markitai_extra "claude-agent"
        fi
    else
        track_install "Claude Code CLI" "skipped"
    fi
    if ask_yes_no "Install GitHub Copilot CLI?" "n"; then
        if lib_install_copilot_cli; then
            # Install Copilot SDK for programmatic access
            lib_install_markitai_extra "copilot"
        fi
    else
        track_install "Copilot CLI" "skipped"
    fi

    # Initialize config
    lib_init_config

    # Print summary
    print_summary

    # Complete
    lib_print_completion
}

# Run main function
main
