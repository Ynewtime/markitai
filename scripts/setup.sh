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

    if ask_yes_no "Playwright browser (for JS-rendered pages)?" "y"; then
        lib_install_playwright_browser
    else
        print_status skip "Playwright browser"
        track_install "Playwright Browser" "skipped"
    fi

    if ask_yes_no "LibreOffice (for .doc/.xls/.ppt)?" "n"; then
        lib_install_libreoffice
    else
        print_status skip "LibreOffice"
        track_install "LibreOffice" "skipped"
    fi

    if ask_yes_no "FFmpeg (for audio/video)?" "n"; then
        lib_install_ffmpeg
    else
        print_status skip "FFmpeg"
        track_install "FFmpeg" "skipped"
    fi

    if ask_yes_no "Claude Code CLI?" "n"; then
        lib_install_claude_cli && lib_install_markitai_extra "claude-agent"
    else
        print_status skip "Claude Code CLI"
        track_install "Claude Code CLI" "skipped"
    fi

    if ask_yes_no "GitHub Copilot CLI?" "n"; then
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
