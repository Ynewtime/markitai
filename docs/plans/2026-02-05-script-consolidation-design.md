# Script Consolidation Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate 10 setup scripts into 2 files with automatic i18n and mode detection.

**Architecture:** Single entry point per platform (setup.sh, setup.ps1) that auto-detects language (en/zh) and mode (user/dev) based on environment.

**Tech Stack:** POSIX shell, PowerShell 5.1+

---

## Current State

```
scripts/
├── lib.sh              # Shared library
├── lib.ps1
├── setup.sh            # User edition (English)
├── setup.ps1
├── setup-zh.sh         # User edition (Chinese)
├── setup-zh.ps1
├── setup-dev.sh        # Developer edition (English)
├── setup-dev.ps1
├── setup-dev-zh.sh     # Developer edition (Chinese)
├── setup-dev-zh.ps1
└── test_scripts.sh
```

**Problems:**
- 10 files, ~8300 lines total
- Changes must be synced across 8 files manually
- Easy to miss updates, causing inconsistencies
- i18n text scattered throughout files

## Target State

```
scripts/
├── setup.sh            # All logic + i18n (Shell)
└── setup.ps1           # All logic + i18n (PowerShell)
```

**Files to delete (8):**
- `lib.sh`, `lib.ps1`
- `setup-zh.sh`, `setup-zh.ps1`
- `setup-dev.sh`, `setup-dev.ps1`
- `setup-dev-zh.sh`, `setup-dev-zh.ps1`

## Design Details

### 1. Language Detection

```bash
detect_lang() {
    case "${LANG:-}" in
        zh_CN*|zh_TW*|zh_HK*) echo "zh" ;;
        *) echo "en" ;;
    esac
}
```

PowerShell:
```powershell
function Get-Lang {
    $culture = (Get-Culture).Name
    if ($culture -match "^zh") { return "zh" }
    return "en"
}
```

### 2. i18n Text Function

```bash
LANG_CODE=$(detect_lang)

i18n() {
    case "$1" in
        # Intro/Outro
        welcome)
            [ "$LANG_CODE" = "zh" ] && echo "欢迎使用 Markitai 安装向导!" || echo "Welcome to Markitai Setup!" ;;
        setup_complete)
            [ "$LANG_CODE" = "zh" ] && echo "配置完成!" || echo "Setup complete!" ;;
        dev_setup_complete)
            [ "$LANG_CODE" = "zh" ] && echo "开发环境配置完成!" || echo "Development environment ready!" ;;

        # Sections
        section_prerequisites)
            [ "$LANG_CODE" = "zh" ] && echo "检测前置依赖" || echo "Checking prerequisites" ;;
        section_core)
            [ "$LANG_CODE" = "zh" ] && echo "安装核心组件" || echo "Installing core components" ;;
        section_optional)
            [ "$LANG_CODE" = "zh" ] && echo "可选组件" || echo "Optional components" ;;
        section_dev_env)
            [ "$LANG_CODE" = "zh" ] && echo "配置开发环境" || echo "Setting up development environment" ;;

        # Status messages
        installed)
            [ "$LANG_CODE" = "zh" ] && echo "已安装" || echo "installed" ;;
        skipped)
            [ "$LANG_CODE" = "zh" ] && echo "已跳过" || echo "skipped" ;;
        failed)
            [ "$LANG_CODE" = "zh" ] && echo "失败" || echo "failed" ;;

        # Confirmations
        confirm_playwright)
            [ "$LANG_CODE" = "zh" ] && echo "是否下载 Chromium 浏览器？(用于 JS 渲染页面)" || echo "Download Chromium browser (for JS-rendered pages)?" ;;
        confirm_libreoffice)
            [ "$LANG_CODE" = "zh" ] && echo "是否安装 LibreOffice？(用于 .doc/.ppt/.xls)" || echo "Install LibreOffice (for .doc/.xls/.ppt)?" ;;
        confirm_ffmpeg)
            [ "$LANG_CODE" = "zh" ] && echo "是否安装 FFmpeg？(用于音视频文件)" || echo "Install FFmpeg (for audio/video)?" ;;
        confirm_claude_cli)
            [ "$LANG_CODE" = "zh" ] && echo "是否安装 Claude Code CLI？" || echo "Install Claude Code CLI?" ;;
        confirm_copilot_cli)
            [ "$LANG_CODE" = "zh" ] && echo "是否安装 GitHub Copilot CLI？" || echo "Install GitHub Copilot CLI?" ;;

        # ... more text keys
        *)
            echo "$1" ;;  # Fallback: return key as-is
    esac
}
```

### 3. Mode Detection

```bash
is_dev_mode() {
    # All conditions must be true for dev mode
    [ -f "pyproject.toml" ] || return 1
    grep -q 'name = "markitai"' pyproject.toml 2>/dev/null || return 1
    [ -d ".git" ] || return 1
    [ -d "scripts" ] || return 1
    return 0
}
```

**Detection matrix:**

| Scenario | pyproject.toml | Contains markitai | .git | scripts/ | Result |
|----------|:-:|:-:|:-:|:-:|--------|
| `curl \| sh` remote | ❌ | - | - | - | User mode |
| ZIP download (no git) | ✅ | ✅ | ❌ | ✅ | User mode |
| Other Python project | ✅ | ❌ | ✅ | ❌ | User mode |
| Cloned repository | ✅ | ✅ | ✅ | ✅ | **Dev mode** |

### 4. File Structure

```bash
#!/bin/sh
# setup.sh - Markitai Setup Script
# Supports: bash, zsh, dash, and other POSIX shells

set -e

# ============================================================
# i18n
# ============================================================
detect_lang() { ... }
LANG_CODE=$(detect_lang)
i18n() { ... }

# ============================================================
# Clack UI Components
# ============================================================
# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
# ...

clack_intro() { ... }
clack_outro() { ... }
clack_section() { ... }
clack_success() { ... }
clack_error() { ... }
clack_warn() { ... }
clack_info() { ... }
clack_skip() { ... }
clack_log() { ... }
clack_note() { ... }
clack_confirm() { ... }
clack_spinner() { ... }
clack_cancel() { ... }

# ============================================================
# Utility Functions
# ============================================================
track_install() { ... }
get_project_root() { ... }
warn_if_root() { ... }

# ============================================================
# Mode Detection
# ============================================================
is_dev_mode() { ... }

# ============================================================
# Installation Functions (Shared)
# ============================================================
install_uv() { ... }
detect_python() { ... }
install_playwright_browser() { ... }
install_libreoffice() { ... }
install_ffmpeg() { ... }
install_claude_cli() { ... }
install_copilot_cli() { ... }

# ============================================================
# User Mode Functions
# ============================================================
install_markitai() { ... }  # uv tool install
install_markitai_extra() { ... }
init_config() { ... }

# ============================================================
# Dev Mode Functions
# ============================================================
sync_dependencies() { ... }  # uv sync --all-extras
install_precommit() { ... }

# ============================================================
# Summary & Completion
# ============================================================
print_summary() { ... }
print_user_completion() { ... }
print_dev_completion() { ... }

# ============================================================
# Main Logic
# ============================================================
run_user_setup() {
    clack_intro "$(i18n welcome)"

    clack_section "$(i18n section_core)"
    install_uv || exit 1
    detect_python || exit 1
    install_markitai || exit 1

    clack_section "$(i18n section_optional)"
    # Playwright, LibreOffice, FFmpeg, CLI tools...

    print_summary
    print_user_completion
    clack_outro "$(i18n setup_complete)"
}

run_dev_setup() {
    clack_intro "$(i18n welcome)"

    clack_section "$(i18n section_prerequisites)"
    install_uv || exit 1
    detect_python || exit 1

    clack_section "$(i18n section_dev_env)"
    sync_dependencies || exit 1
    install_precommit

    clack_section "$(i18n section_optional)"
    # Playwright, LibreOffice, FFmpeg, CLI tools...

    print_summary
    print_dev_completion
    clack_outro "$(i18n dev_setup_complete)"
}

main() {
    if is_dev_mode; then
        run_dev_setup
    else
        run_user_setup
    fi
}

main "$@"
```

### 5. Usage

```bash
# Remote install (user mode - auto-detected)
curl -fsSL https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.sh | sh

# Local dev setup (dev mode - auto-detected)
git clone https://github.com/Ynewtime/markitai.git
cd markitai
./scripts/setup.sh

# Force language (optional)
LANG=zh_CN.UTF-8 ./scripts/setup.sh
```

PowerShell:
```powershell
# Remote install
irm https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts/setup.ps1 | iex

# Local dev setup
git clone https://github.com/Ynewtime/markitai.git
cd markitai
.\scripts\setup.ps1
```

## Benefits

1. **Maintainability**: 2 files instead of 10
2. **Consistency**: Single source of truth for each platform
3. **i18n**: Centralized text, easy to add languages
4. **UX**: Zero configuration - auto-detects everything
5. **Simplicity**: No dependencies between files

## Migration Notes

- Preserve all existing functionality
- Ensure backward compatibility for remote execution URLs
- Test both user and dev modes on all platforms
- Test language detection on various locales
