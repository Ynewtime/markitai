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

    # Header
    print_header "Markitai Setup"

    # Core installation
    lib_install_uv || exit 1
    lib_detect_python || exit 1
    lib_install_markitai || exit 1

    # Optional components
    printf "\n"
    printf "  ${BOLD}Optional components:${NC}\n"

    # Playwright browser - auto-detect first
    if lib_detect_playwright_browser; then
        print_status ok "Playwright browser (already installed)"
        track_install "Playwright Browser" "installed"
    elif ask_yes_no "Playwright browser (for JS-rendered pages)?" "y"; then
        lib_install_playwright_browser
    else
        print_status skip "Playwright browser"
        track_install "Playwright Browser" "skipped"
    fi

    # LibreOffice - auto-detect first
    if command -v soffice >/dev/null 2>&1 || command -v libreoffice >/dev/null 2>&1; then
        print_status ok "LibreOffice (already installed)"
        track_install "LibreOffice" "installed"
    elif ask_yes_no "LibreOffice (for .doc/.xls/.ppt)?" "n"; then
        lib_install_libreoffice
    else
        print_status skip "LibreOffice"
        track_install "LibreOffice" "skipped"
    fi

    # FFmpeg - auto-detect first
    if command -v ffmpeg >/dev/null 2>&1; then
        print_status ok "FFmpeg (already installed)"
        track_install "FFmpeg" "installed"
    elif ask_yes_no "FFmpeg (for audio/video)?" "n"; then
        lib_install_ffmpeg
    else
        print_status skip "FFmpeg"
        track_install "FFmpeg" "skipped"
    fi

    # Claude Code CLI - auto-detect first
    if command -v claude >/dev/null 2>&1; then
        version=$(claude --version 2>/dev/null | head -n1)
        print_status ok "Claude Code CLI: $version"
        track_install "Claude Code CLI" "installed"
    elif ask_yes_no "Claude Code CLI?" "n"; then
        lib_install_claude_cli && lib_install_markitai_extra "claude-agent"
    else
        print_status skip "Claude Code CLI"
        track_install "Claude Code CLI" "skipped"
    fi

    # Copilot CLI - auto-detect first
    if command -v copilot >/dev/null 2>&1; then
        version=$(copilot --version 2>/dev/null | head -n1)
        print_status ok "Copilot CLI: $version"
        track_install "Copilot CLI" "installed"
    elif ask_yes_no "GitHub Copilot CLI?" "n"; then
        lib_install_copilot_cli && lib_install_markitai_extra "copilot"
    else
        print_status skip "Copilot CLI"
        track_install "Copilot CLI" "skipped"
    fi

    # Initialize config silently
    lib_init_config >/dev/null 2>&1

    # Summary
    print_summary

    # Get started
    lib_print_completion
}

# Run main function
main
