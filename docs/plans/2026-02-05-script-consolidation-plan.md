# Script Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate 10 setup scripts into 2 files (setup.sh + setup.ps1) with auto i18n and mode detection.

**Architecture:** Single file per platform containing all logic. Auto-detect language from system locale, auto-detect user/dev mode from environment (pyproject.toml + .git presence).

**Tech Stack:** POSIX shell, PowerShell 5.1+

---

## Task 1: Create setup.sh with i18n and Clack UI

**Files:**
- Create: `scripts/setup.sh.new` (will replace `scripts/setup.sh` after testing)

**Step 1: Create file header and i18n system**

```bash
#!/bin/sh
# Markitai Setup Script
# Supports: bash, zsh, dash, and other POSIX shells
# Auto-detects: language (en/zh), mode (user/dev)

set -e

# ============================================================
# i18n System
# ============================================================

detect_lang() {
    case "${LANG:-}${LC_ALL:-}${LC_MESSAGES:-}" in
        *zh_CN*|*zh_TW*|*zh_HK*|*zh_SG*) echo "zh" ;;
        *) echo "en" ;;
    esac
}

LANG_CODE=$(detect_lang)

i18n() {
    case "$1" in
        # === Intro/Outro ===
        welcome) [ "$LANG_CODE" = "zh" ] && echo "欢迎使用 Markitai 安装向导!" || echo "Welcome to Markitai Setup!" ;;
        setup_complete) [ "$LANG_CODE" = "zh" ] && echo "配置完成!" || echo "Setup complete!" ;;
        dev_setup_complete) [ "$LANG_CODE" = "zh" ] && echo "开发环境配置完成!" || echo "Development environment ready!" ;;

        # === Sections ===
        section_prerequisites) [ "$LANG_CODE" = "zh" ] && echo "检测前置依赖" || echo "Checking prerequisites" ;;
        section_core) [ "$LANG_CODE" = "zh" ] && echo "安装核心组件" || echo "Installing core components" ;;
        section_optional) [ "$LANG_CODE" = "zh" ] && echo "可选组件" || echo "Optional components" ;;
        section_dev_env) [ "$LANG_CODE" = "zh" ] && echo "配置开发环境" || echo "Setting up development environment" ;;
        section_llm_cli) [ "$LANG_CODE" = "zh" ] && echo "LLM CLI 工具" || echo "LLM CLI tools" ;;
        section_summary) [ "$LANG_CODE" = "zh" ] && echo "安装总结" || echo "Summary" ;;

        # === Status ===
        installed) [ "$LANG_CODE" = "zh" ] && echo "已安装" || echo "installed" ;;
        installing) [ "$LANG_CODE" = "zh" ] && echo "正在安装" || echo "Installing" ;;
        skipped) [ "$LANG_CODE" = "zh" ] && echo "已跳过" || echo "skipped" ;;
        failed) [ "$LANG_CODE" = "zh" ] && echo "失败" || echo "failed" ;;
        success) [ "$LANG_CODE" = "zh" ] && echo "成功" || echo "success" ;;
        already_installed) [ "$LANG_CODE" = "zh" ] && echo "已安装" || echo "already installed" ;;

        # === Components ===
        uv) echo "uv" ;;
        python) echo "Python" ;;
        markitai) echo "markitai" ;;
        playwright) [ "$LANG_CODE" = "zh" ] && echo "Playwright 浏览器 (Chromium)" || echo "Playwright browser (Chromium)" ;;
        libreoffice) echo "LibreOffice" ;;
        ffmpeg) echo "FFmpeg" ;;
        claude_cli) echo "Claude Code CLI" ;;
        copilot_cli) echo "Copilot CLI" ;;
        precommit) echo "pre-commit hooks" ;;
        python_deps) [ "$LANG_CODE" = "zh" ] && echo "Python 依赖" || echo "Python dependencies" ;;

        # === Confirmations ===
        confirm_playwright) [ "$LANG_CODE" = "zh" ] && echo "是否下载 Chromium 浏览器？(用于 JS 渲染页面)" || echo "Download Chromium browser (for JS-rendered pages)?" ;;
        confirm_libreoffice) [ "$LANG_CODE" = "zh" ] && echo "是否安装 LibreOffice？(用于 .doc/.ppt/.xls)" || echo "Install LibreOffice (for .doc/.xls/.ppt)?" ;;
        confirm_ffmpeg) [ "$LANG_CODE" = "zh" ] && echo "是否安装 FFmpeg？(用于音视频文件)" || echo "Install FFmpeg (for audio/video)?" ;;
        confirm_claude_cli) [ "$LANG_CODE" = "zh" ] && echo "是否安装 Claude Code CLI？" || echo "Install Claude Code CLI?" ;;
        confirm_copilot_cli) [ "$LANG_CODE" = "zh" ] && echo "是否安装 GitHub Copilot CLI？" || echo "Install GitHub Copilot CLI?" ;;
        confirm_uv) [ "$LANG_CODE" = "zh" ] && echo "是否自动安装 UV?" || echo "Install uv automatically?" ;;
        confirm_continue_as_root) [ "$LANG_CODE" = "zh" ] && echo "是否继续以 root 身份运行?" || echo "Continue running as root?" ;;

        # === Info messages ===
        info_libreoffice_purpose) [ "$LANG_CODE" = "zh" ] && echo "用途: 转换旧版 Office 文件 (.doc, .ppt, .xls)" || echo "Purpose: Convert legacy Office files (.doc, .ppt, .xls)" ;;
        info_ffmpeg_purpose) [ "$LANG_CODE" = "zh" ] && echo "用途: 处理音视频文件 (.mp3, .mp4, .wav 等)" || echo "Purpose: Process audio/video files (.mp3, .mp4, .wav, etc.)" ;;
        info_playwright_purpose) [ "$LANG_CODE" = "zh" ] && echo "用途: 浏览器自动化，用于 JavaScript 渲染页面" || echo "Purpose: Browser automation for JS-rendered pages" ;;
        info_project_dir) [ "$LANG_CODE" = "zh" ] && echo "项目目录" || echo "Project directory" ;;
        info_docs) [ "$LANG_CODE" = "zh" ] && echo "文档" || echo "Documentation" ;;
        info_issues) [ "$LANG_CODE" = "zh" ] && echo "问题反馈" || echo "Issues" ;;
        info_syncing_deps) [ "$LANG_CODE" = "zh" ] && echo "同步依赖..." || echo "Syncing dependencies..." ;;
        info_deps_synced) [ "$LANG_CODE" = "zh" ] && echo "依赖同步完成" || echo "Dependencies synced" ;;
        info_precommit_installed) [ "$LANG_CODE" = "zh" ] && echo "pre-commit hooks 安装完成" || echo "pre-commit hooks installed" ;;

        # === Errors ===
        error_uv_required) [ "$LANG_CODE" = "zh" ] && echo "UV 是必需的" || echo "uv is required" ;;
        error_python_required) [ "$LANG_CODE" = "zh" ] && echo "Python 3.11+ 是必需的" || echo "Python 3.11+ is required" ;;
        error_setup_failed) [ "$LANG_CODE" = "zh" ] && echo "配置失败" || echo "Setup failed" ;;

        # === Warnings ===
        warn_root) [ "$LANG_CODE" = "zh" ] && echo "警告: 正在以 root 身份运行" || echo "Warning: Running as root" ;;
        warn_root_risk) [ "$LANG_CODE" = "zh" ] && echo "以 root 身份运行存在安全风险" || echo "Running as root has security risks" ;;

        # === Getting started ===
        getting_started) [ "$LANG_CODE" = "zh" ] && echo "开始使用" || echo "Getting started" ;;
        quick_start) [ "$LANG_CODE" = "zh" ] && echo "快速开始" || echo "Quick start" ;;
        activate_venv) [ "$LANG_CODE" = "zh" ] && echo "激活虚拟环境" || echo "Activate virtual environment" ;;
        run_tests) [ "$LANG_CODE" = "zh" ] && echo "运行测试" || echo "Run tests" ;;
        run_cli) [ "$LANG_CODE" = "zh" ] && echo "运行 CLI" || echo "Run CLI" ;;
        interactive_mode) [ "$LANG_CODE" = "zh" ] && echo "交互模式" || echo "Interactive mode" ;;
        convert_file) [ "$LANG_CODE" = "zh" ] && echo "转换文件" || echo "Convert a file" ;;
        show_help) [ "$LANG_CODE" = "zh" ] && echo "查看帮助" || echo "Show all options" ;;

        # === Summary labels ===
        summary_installed) [ "$LANG_CODE" = "zh" ] && echo "已安装" || echo "Installed" ;;
        summary_skipped) [ "$LANG_CODE" = "zh" ] && echo "已跳过" || echo "Skipped" ;;
        summary_failed) [ "$LANG_CODE" = "zh" ] && echo "安装失败" || echo "Failed" ;;

        # === Default ===
        *) echo "$1" ;;
    esac
}
```

**Step 2: Add Clack UI components**

Copy Clack UI functions from existing lib.sh:
- Color definitions (RED, GREEN, YELLOW, etc.)
- clack_intro, clack_outro
- clack_section, clack_log
- clack_success, clack_error, clack_warn, clack_info, clack_skip
- clack_note, clack_confirm, clack_spinner, clack_cancel

**Step 3: Add utility functions**

```bash
# ============================================================
# Utility Functions
# ============================================================

# Track installation status
INSTALLED_COMPONENTS=""
SKIPPED_COMPONENTS=""
FAILED_COMPONENTS=""

track_install() {
    component="$1"
    status="$2"
    case "$status" in
        installed) INSTALLED_COMPONENTS="${INSTALLED_COMPONENTS}${INSTALLED_COMPONENTS:+|}$component" ;;
        skipped) SKIPPED_COMPONENTS="${SKIPPED_COMPONENTS}${SKIPPED_COMPONENTS:+|}$component" ;;
        failed) FAILED_COMPONENTS="${FAILED_COMPONENTS}${FAILED_COMPONENTS:+|}$component" ;;
    esac
}

get_project_root() {
    if [ -n "$SCRIPT_DIR" ]; then
        dirname "$SCRIPT_DIR"
    else
        pwd
    fi
}

warn_if_root() {
    if [ "$(id -u)" -eq 0 ]; then
        clack_warn "$(i18n warn_root)"
        clack_log "  $(i18n warn_root_risk)"
        if ! clack_confirm "$(i18n confirm_continue_as_root)" "n"; then
            exit 1
        fi
    fi
}
```

**Step 4: Add mode detection**

```bash
# ============================================================
# Mode Detection
# ============================================================

is_dev_mode() {
    [ -f "pyproject.toml" ] || return 1
    grep -q 'name = "markitai"' pyproject.toml 2>/dev/null || return 1
    [ -d ".git" ] || return 1
    [ -d "scripts" ] || return 1
    return 0
}
```

**Step 5: Commit skeleton**

```bash
git add scripts/setup.sh.new
git commit -m "feat(scripts): add setup.sh skeleton with i18n and clack UI"
```

---

## Task 2: Add installation functions to setup.sh

**Files:**
- Modify: `scripts/setup.sh.new`

**Step 1: Add UV installation function**

Extract from existing scripts, using i18n() for all text.

**Step 2: Add Python detection function**

**Step 3: Add markitai installation function (user mode)**

**Step 4: Add dependency sync function (dev mode)**

**Step 5: Add pre-commit installation function (dev mode)**

**Step 6: Add optional component functions**

- install_playwright_browser
- install_libreoffice
- install_ffmpeg
- install_claude_cli
- install_copilot_cli

**Step 7: Add summary and completion functions**

**Step 8: Commit**

```bash
git add scripts/setup.sh.new
git commit -m "feat(scripts): add installation functions to setup.sh"
```

---

## Task 3: Add main logic to setup.sh

**Files:**
- Modify: `scripts/setup.sh.new`

**Step 1: Implement run_user_setup()**

```bash
run_user_setup() {
    clack_intro "$(i18n welcome)"
    warn_if_root

    clack_section "$(i18n section_core)"
    install_uv || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }
    detect_python || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }
    install_markitai || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }

    clack_section "$(i18n section_optional)"
    install_optional_playwright
    install_optional_libreoffice
    install_optional_ffmpeg

    clack_section "$(i18n section_llm_cli)"
    install_optional_claude_cli
    install_optional_copilot_cli

    init_config >/dev/null 2>&1

    print_summary
    print_user_completion
    clack_outro "$(i18n setup_complete)"
}
```

**Step 2: Implement run_dev_setup()**

```bash
run_dev_setup() {
    clack_intro "$(i18n welcome)"
    warn_if_root

    clack_section "$(i18n section_prerequisites)"
    install_uv || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }
    detect_python || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }

    clack_section "$(i18n section_dev_env)"
    clack_info "$(i18n info_project_dir): $(get_project_root)"
    sync_dependencies || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }
    install_precommit

    clack_section "$(i18n section_optional)"
    install_optional_playwright
    install_optional_libreoffice
    install_optional_ffmpeg

    clack_section "$(i18n section_llm_cli)"
    install_optional_claude_cli
    install_optional_copilot_cli

    print_summary
    print_dev_completion
    clack_outro "$(i18n dev_setup_complete)"
}
```

**Step 3: Implement main()**

```bash
main() {
    if is_dev_mode; then
        run_dev_setup
    else
        run_user_setup
    fi
}

main "$@"
```

**Step 4: Commit**

```bash
git add scripts/setup.sh.new
git commit -m "feat(scripts): add main logic to setup.sh"
```

---

## Task 4: Test setup.sh

**Step 1: Test language detection**

```bash
# Test English
LANG=en_US.UTF-8 bash -c 'source scripts/setup.sh.new; echo "Lang: $LANG_CODE"'
# Expected: Lang: en

# Test Chinese
LANG=zh_CN.UTF-8 bash -c 'source scripts/setup.sh.new; echo "Lang: $LANG_CODE"'
# Expected: Lang: zh
```

**Step 2: Test mode detection**

```bash
# In repo root (should be dev mode)
cd /home/y/dev/markitai/.worktrees/feat-ux-simplification
bash -c 'source scripts/setup.sh.new; is_dev_mode && echo "DEV" || echo "USER"'
# Expected: DEV

# In temp dir (should be user mode)
cd /tmp
bash -c 'source /home/y/dev/markitai/.worktrees/feat-ux-simplification/scripts/setup.sh.new; is_dev_mode && echo "DEV" || echo "USER"'
# Expected: USER
```

**Step 3: Run full script in dev mode**

```bash
cd /home/y/dev/markitai/.worktrees/feat-ux-simplification
./scripts/setup.sh.new
```

**Step 4: Commit tested version**

```bash
git add scripts/setup.sh.new
git commit -m "test(scripts): verify setup.sh works correctly"
```

---

## Task 5: Create setup.ps1

**Files:**
- Create: `scripts/setup.ps1.new`

**Step 1: Create file with i18n system**

PowerShell version of i18n using hashtable:

```powershell
# Markitai Setup Script
# PowerShell 5.1+
# Auto-detects: language (en/zh), mode (user/dev)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ============================================================
# i18n System
# ============================================================

function Get-Lang {
    $culture = (Get-Culture).Name
    if ($culture -match "^zh") { return "zh" }
    return "en"
}

$script:LANG_CODE = Get-Lang

function i18n($key) {
    $texts = @{
        # Intro/Outro
        "welcome" = @{ zh = "欢迎使用 Markitai 安装向导!"; en = "Welcome to Markitai Setup!" }
        "setup_complete" = @{ zh = "配置完成!"; en = "Setup complete!" }
        "dev_setup_complete" = @{ zh = "开发环境配置完成!"; en = "Development environment ready!" }
        # ... (all other keys)
    }

    if ($texts.ContainsKey($key)) {
        return $texts[$key][$script:LANG_CODE]
    }
    return $key
}
```

**Step 2: Add Clack UI components**

Port from lib.ps1.

**Step 3: Add utility and mode detection**

**Step 4: Add installation functions**

**Step 5: Add main logic**

**Step 6: Commit**

```bash
git add scripts/setup.ps1.new
git commit -m "feat(scripts): create setup.ps1 with i18n and auto mode detection"
```

---

## Task 6: Replace old scripts

**Step 1: Backup and replace**

```bash
# Backup
mkdir -p scripts/archive
mv scripts/lib.sh scripts/archive/
mv scripts/lib.ps1 scripts/archive/
mv scripts/setup-zh.sh scripts/archive/
mv scripts/setup-zh.ps1 scripts/archive/
mv scripts/setup-dev.sh scripts/archive/
mv scripts/setup-dev.ps1 scripts/archive/
mv scripts/setup-dev-zh.sh scripts/archive/
mv scripts/setup-dev-zh.ps1 scripts/archive/
mv scripts/setup.sh scripts/archive/
mv scripts/setup.ps1 scripts/archive/

# Replace
mv scripts/setup.sh.new scripts/setup.sh
mv scripts/setup.ps1.new scripts/setup.ps1
chmod +x scripts/setup.sh
```

**Step 2: Update any documentation references**

Check README.md and docs/ for script references.

**Step 3: Final testing**

```bash
# Test remote-style execution (user mode)
cd /tmp
curl -fsSL file:///home/y/dev/markitai/.worktrees/feat-ux-simplification/scripts/setup.sh | sh

# Test dev mode
cd /home/y/dev/markitai/.worktrees/feat-ux-simplification
./scripts/setup.sh
```

**Step 4: Commit final version**

```bash
git add -A
git commit -m "feat(scripts): consolidate to setup.sh + setup.ps1

BREAKING CHANGE: Removed separate *-zh and *-dev script variants.
- Auto-detect language from system locale
- Auto-detect user/dev mode from environment
- Reduced from 10 files to 2 files
- Centralized i18n text management"
```

---

## Task 7: Cleanup

**Step 1: Remove archive after verification**

```bash
rm -rf scripts/archive
git add -A
git commit -m "chore(scripts): remove archived old scripts"
```

**Step 2: Update CLAUDE.md if needed**

Remove references to old script files.

---

## Summary

| Task | Description | Estimated Steps |
|------|-------------|-----------------|
| 1 | Create setup.sh skeleton | 5 |
| 2 | Add installation functions | 8 |
| 3 | Add main logic | 4 |
| 4 | Test setup.sh | 4 |
| 5 | Create setup.ps1 | 6 |
| 6 | Replace old scripts | 4 |
| 7 | Cleanup | 2 |

**Total: 7 tasks, ~33 steps**
