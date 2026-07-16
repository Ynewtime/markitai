#!/bin/sh
# Markitai Setup Script - Unified installer with i18n support
# Supports bash/zsh/dash and other POSIX-compatible shells
# Auto-detects: language (en/zh), mode (user/dev)
#
# Usage:
#   curl -fsSL https://markitai.dev/setup.sh | sh    # User install
#   ./scripts/setup.sh                               # Dev setup (in repo)
# Full failure logs use a private temp file; set MARKITAI_SETUP_LOG to override.

# -e: exit on error; -u: error on unset variables (pipefail is not POSIX sh)
set -eu

# ============================================================
# Internationalization (i18n) System
# ============================================================

# Detect language from environment
# Returns: "zh" for Chinese, "en" for English (default)
detect_lang() {
    _lang_env="${LANG:-}${LC_ALL:-}${LC_MESSAGES:-}"
    case "$_lang_env" in
        zh_CN*|zh_TW*|zh_HK*|zh_SG*|zh.*)
            echo "zh"
            ;;
        *)
            echo "en"
            ;;
    esac
}

LANG_CODE=$(detect_lang)

# Internationalization function
# Usage: i18n "key"
# Returns localized string for the given key
i18n() {
    _key="$1"

    if [ "$LANG_CODE" = "zh" ]; then
        case "$_key" in
            # Intro/Outro
            welcome)                    echo "欢迎使用 Markitai 安装程序!" ;;
            setup_complete)             echo "安装完成!" ;;
            setup_complete_with_warnings) echo "安装完成，但有部分组件失败" ;;
            dev_setup_complete)         echo "开发环境设置完成!" ;;
            dev_setup_complete_with_warnings) echo "开发环境设置完成，但有部分组件失败" ;;

            # Mode
            mode_user)                  echo "模式: 用户安装" ;;
            mode_dev)                   echo "模式: 开发环境" ;;

            # Sections
            section_prerequisites)      echo "前置条件" ;;
            section_core)               echo "核心组件" ;;
            section_optional)           echo "可选组件" ;;
            section_dev_env)            echo "开发环境" ;;
            section_llm_cli)            echo "LLM CLI 工具" ;;
            section_summary)            echo "安装摘要" ;;

            # Status
            installed)                  echo "已安装" ;;
            installing)                 echo "正在安装" ;;
            skipped)                    echo "已跳过" ;;
            failed)                     echo "失败" ;;
            success)                    echo "成功" ;;
            already_installed)          echo "已经安装" ;;
            not_found)                  echo "未找到" ;;

            # Components
            uv)                         echo "uv 包管理器" ;;
            python)                     echo "Python" ;;
            markitai)                   echo "markitai" ;;
            serve)                      echo "Web UI (markitai serve)" ;;
            playwright)                 echo "Playwright 浏览器" ;;
            libreoffice)                echo "LibreOffice" ;;
            ffmpeg)                     echo "FFmpeg" ;;
            claude_cli)                 echo "Claude Code CLI" ;;
            copilot_cli)                echo "Copilot CLI" ;;
            precommit)                  echo "pre-commit hooks" ;;
            python_deps)                echo "Python 依赖" ;;

            # Confirmations
            confirm_serve)              echo "安装 Web UI 依赖? (启用 markitai serve)" ;;
            confirm_playwright)         echo "安装 Playwright 浏览器? (用于 JS 渲染页面)" ;;
            confirm_libreoffice)        echo "安装 LibreOffice? (用于 Office 文档转换)" ;;
            confirm_ffmpeg)             echo "安装 FFmpeg? (用于音视频处理)" ;;
            confirm_claude_cli)         echo "安装 Claude Code CLI? (使用 Claude 订阅)" ;;
            confirm_copilot_cli)        echo "安装 Copilot CLI? (使用 GitHub Copilot 订阅)" ;;
            confirm_uv)                 echo "安装 uv 包管理器?" ;;
            confirm_continue_as_root)   echo "以 root 身份继续?" ;;

            # Info messages
            info_libreoffice_purpose)   echo "LibreOffice 用于转换旧版 Office 文档 (.doc/.ppt) 并渲染幻灯片截图" ;;
            info_ffmpeg_purpose)        echo "FFmpeg 用于处理音频和视频文件" ;;
            info_playwright_purpose)    echo "Playwright 用于获取 JavaScript 渲染的网页内容" ;;
            info_project_dir)           echo "项目目录" ;;
            info_docs)                  echo "文档" ;;
            info_issues)                echo "问题反馈" ;;
            info_syncing_deps)          echo "正在同步依赖..." ;;
            info_deps_synced)           echo "依赖同步完成" ;;
            info_precommit_installed)   echo "pre-commit hooks 已安装" ;;
            info_error_log)             echo "完整错误日志" ;;

            # Error messages
            error_uv_required)          echo "需要安装 uv 包管理器" ;;
            error_python_required)      echo "需要安装 Python 3.11-3.13" ;;
            error_unexpected)           echo "发生意外错误" ;;
            error_setup_failed)         echo "安装失败" ;;

            # Install source (repo detection)
            repo_detected)              echo "检测到 markitai 源码仓库" ;;
            confirm_local_install)      echo "从本地源码安装 markitai? (默认使用 PyPI 发布版)" ;;
            source_local)               echo "安装来源: 本地源码" ;;
            source_pypi)                echo "安装来源: PyPI" ;;
            info_repo_noninteractive)   echo "非交互模式: 已检测到源码仓库, 将使用 PyPI 发布版" ;;

            # Network / Mirrors
            section_network)            echo "网络环境" ;;
            mirror_no_proxy)            echo "未检测到代理，部分资源可能无法访问" ;;
            mirror_confirm)             echo "启用国内镜像加速? (推荐无代理环境使用)" ;;
            mirror_select)              echo "选择镜像源" ;;
            mirror_tuna)                echo "清华 TUNA (推荐)" ;;
            mirror_aliyun)              echo "阿里云" ;;
            mirror_tencent)             echo "腾讯云" ;;
            mirror_huawei)              echo "华为云" ;;
            mirror_enabled)             echo "已启用国内镜像加速" ;;
            mirror_skipped)             echo "已跳过镜像配置" ;;
            mirror_pypi)                echo "PyPI 镜像" ;;
            mirror_playwright)          echo "Playwright 镜像" ;;
            mirror_npm)                 echo "npm 镜像" ;;

            # Warnings
            warn_root)                  echo "警告: 以 root 身份运行" ;;
            warn_root_risk)             echo "以 root 运行安装脚本存在安全风险" ;;

            # Getting started
            getting_started)            echo "开始使用" ;;
            quick_start)                echo "快速开始" ;;
            activate_venv)              echo "激活虚拟环境" ;;
            run_tests)                  echo "运行测试" ;;
            run_cli)                    echo "运行 CLI" ;;
            interactive_mode)           echo "交互模式" ;;
            start_web_ui)               echo "启动 Web UI" ;;
            configure_llm)              echo "配置 LLM" ;;
            configure_env)              echo "配置环境变量" ;;
            convert_file)               echo "转换文件" ;;
            show_help)                  echo "显示帮助" ;;
            short_alias)                echo "提示：mkai 是 markitai 的简写，两个命令等价" ;;
            mkai_conflict)              echo "系统已有 mkai 命令，会遮蔽 markitai 的别名；请使用完整命令 markitai" ;;

            # Summary
            summary_installed)          echo "已安装" ;;
            summary_skipped)            echo "已跳过" ;;
            summary_failed)             echo "安装失败" ;;

            # Default fallback
            *)                          echo "$_key" ;;
        esac
    else
        # English (default)
        case "$_key" in
            # Intro/Outro
            welcome)                    echo "Welcome to Markitai Setup!" ;;
            setup_complete)             echo "Setup complete!" ;;
            setup_complete_with_warnings) echo "Setup complete with warnings" ;;
            dev_setup_complete)         echo "Development environment ready!" ;;
            dev_setup_complete_with_warnings) echo "Development environment ready with warnings" ;;

            # Mode
            mode_user)                  echo "Mode: User Install" ;;
            mode_dev)                   echo "Mode: Development" ;;

            # Sections
            section_prerequisites)      echo "Prerequisites" ;;
            section_core)               echo "Core Components" ;;
            section_optional)           echo "Optional Components" ;;
            section_dev_env)            echo "Development Environment" ;;
            section_llm_cli)            echo "LLM CLI Tools" ;;
            section_summary)            echo "Installation Summary" ;;

            # Status
            installed)                  echo "installed" ;;
            installing)                 echo "installing" ;;
            skipped)                    echo "skipped" ;;
            failed)                     echo "failed" ;;
            success)                    echo "success" ;;
            already_installed)          echo "already installed" ;;
            not_found)                  echo "not found" ;;

            # Components
            uv)                         echo "uv package manager" ;;
            python)                     echo "Python" ;;
            markitai)                   echo "markitai" ;;
            serve)                      echo "Web UI (markitai serve)" ;;
            playwright)                 echo "Playwright browser" ;;
            libreoffice)                echo "LibreOffice" ;;
            ffmpeg)                     echo "FFmpeg" ;;
            claude_cli)                 echo "Claude Code CLI" ;;
            copilot_cli)                echo "Copilot CLI" ;;
            precommit)                  echo "pre-commit hooks" ;;
            python_deps)                echo "Python dependencies" ;;

            # Confirmations
            confirm_serve)              echo "Install Web UI dependencies? (enables markitai serve)" ;;
            confirm_playwright)         echo "Install Playwright browser? (for JS-rendered pages)" ;;
            confirm_libreoffice)        echo "Install LibreOffice? (for Office document conversion)" ;;
            confirm_ffmpeg)             echo "Install FFmpeg? (for audio/video processing)" ;;
            confirm_claude_cli)         echo "Install Claude Code CLI? (use your Claude subscription)" ;;
            confirm_copilot_cli)        echo "Install Copilot CLI? (use your GitHub Copilot subscription)" ;;
            confirm_uv)                 echo "Install uv package manager?" ;;
            confirm_continue_as_root)   echo "Continue as root?" ;;

            # Info messages
            info_libreoffice_purpose)   echo "LibreOffice converts legacy Office files (.doc/.ppt) and renders slide screenshots" ;;
            info_ffmpeg_purpose)        echo "FFmpeg processes audio and video files" ;;
            info_playwright_purpose)    echo "Playwright fetches JavaScript-rendered web pages" ;;
            info_project_dir)           echo "Project directory" ;;
            info_docs)                  echo "Documentation" ;;
            info_issues)                echo "Issues" ;;
            info_syncing_deps)          echo "Syncing dependencies..." ;;
            info_deps_synced)           echo "Dependencies synced" ;;
            info_precommit_installed)   echo "pre-commit hooks installed" ;;
            info_error_log)             echo "Full error log" ;;

            # Error messages
            error_uv_required)          echo "uv package manager is required" ;;
            error_python_required)      echo "Python 3.11-3.13 is required" ;;
            error_unexpected)           echo "Unexpected error" ;;
            error_setup_failed)         echo "Setup failed" ;;

            # Install source (repo detection)
            repo_detected)              echo "markitai source repo detected" ;;
            confirm_local_install)      echo "Install markitai from the local repo? (default: PyPI release)" ;;
            source_local)               echo "Install source: local repo" ;;
            source_pypi)                echo "Install source: PyPI" ;;
            info_repo_noninteractive)   echo "Non-interactive mode: source repo detected, using PyPI release" ;;

            # Network / Mirrors
            section_network)            echo "Network Environment" ;;
            mirror_no_proxy)            echo "No proxy detected, some resources may be inaccessible" ;;
            mirror_confirm)             echo "Enable China mirror acceleration? (recommended without proxy)" ;;
            mirror_select)              echo "Select mirror source" ;;
            mirror_tuna)                echo "Tsinghua TUNA (Recommended)" ;;
            mirror_aliyun)              echo "Alibaba Cloud" ;;
            mirror_tencent)             echo "Tencent Cloud" ;;
            mirror_huawei)              echo "Huawei Cloud" ;;
            mirror_enabled)             echo "China mirror acceleration enabled" ;;
            mirror_skipped)             echo "Mirror configuration skipped" ;;
            mirror_pypi)                echo "PyPI mirror" ;;
            mirror_playwright)          echo "Playwright mirror" ;;
            mirror_npm)                 echo "npm mirror" ;;

            # Warnings
            warn_root)                  echo "Warning: Running as root" ;;
            warn_root_risk)             echo "Running setup scripts as root carries security risks" ;;

            # Getting started
            getting_started)            echo "Getting Started" ;;
            quick_start)                echo "Quick Start" ;;
            activate_venv)              echo "Activate virtual environment" ;;
            run_tests)                  echo "Run tests" ;;
            run_cli)                    echo "Run CLI" ;;
            interactive_mode)           echo "Interactive mode" ;;
            start_web_ui)               echo "Start Web UI" ;;
            configure_llm)              echo "Configure LLM" ;;
            configure_env)              echo "Configure environment" ;;
            convert_file)               echo "Convert a file" ;;
            show_help)                  echo "Show help" ;;
            short_alias)                echo "Tip: mkai is a short alias for markitai (both work)" ;;
            mkai_conflict)              echo "An existing 'mkai' command shadows markitai's alias; use the full 'markitai' command" ;;

            # Summary
            summary_installed)          echo "Installed" ;;
            summary_skipped)            echo "Skipped" ;;
            summary_failed)             echo "Failed" ;;

            # Default fallback
            *)                          echo "$_key" ;;
        esac
    fi
}

# ============================================================
# Color Definitions
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
GRAY='\033[0;90m'
NC='\033[0m'
BOLD='\033[1m'
DIM='\033[2m'

# ============================================================
# Compact tree-style visual components
# ============================================================
# The one-column guide follows the `withGuide` visual grammar popularized by
# @clack/prompts. Shell installers cannot rely on Node being present (or redraw
# completed prompts like Clack), so the tiny renderer stays inline and uses
# tree branches to keep the guide visibly connected in static terminal output.

S_BAR="│"
S_BAR_H="─"
S_CORNER_TOP="╭"
S_CORNER_BOTTOM="╰"
S_BRANCH="├"
S_CHECK="✓"
S_CROSS="✗"
S_ARROW="→"
S_CIRCLE="○"

# Session state lets the global exit guard close an interrupted tree exactly
# once. Spinner state is also global so signals cannot leave it running.
CLACK_SESSION_ACTIVE=false
CLACK_SESSION_CLOSED=false
CLACK_SPINNER_PID=""
CLACK_SPINNER_ERRFILE=""
CLACK_PENDING_DETAIL=""
SETUP_LOG_FILE="${MARKITAI_SETUP_LOG:-}"
SETUP_LOG_AUTO=false
SETUP_LOG_SHOWN=false

# Print an empty row while preserving the session guide.
clack_guide() {
    printf "${GRAY}%s${NC}\n" "$S_BAR"
}

# Session intro - start of CLI flow
# Usage: clack_intro "Title"
clack_intro() {
    CLACK_SESSION_ACTIVE=true
    CLACK_SESSION_CLOSED=false
    printf "\n"
    printf "${GRAY}%s%s${NC} ${BOLD}%s${NC}\n" "$S_CORNER_TOP" "$S_BAR_H" "$1"
    clack_guide
}

# Session outro - end of CLI flow
# Usage: clack_outro "Message"
clack_outro() {
    CLACK_SESSION_CLOSED=true
    if [ "$SETUP_LOG_AUTO" = true ] && [ -n "$SETUP_LOG_FILE" ]; then
        rm -f "$SETUP_LOG_FILE" 2>/dev/null || true
        SETUP_LOG_FILE=""
    fi
    clack_guide
    printf "${GRAY}%s%s${NC} ${GREEN}%s${NC}\n\n" "$S_CORNER_BOTTOM" "$S_BAR_H" "$1"
}

# Non-fatal component failures close the tree in warning yellow, not success
# green, while still returning a successful core installation.
clack_outro_warn() {
    clack_show_error_log
    CLACK_SESSION_CLOSED=true
    clack_guide
    printf "${GRAY}%s%s${NC} ${YELLOW}%s${NC}\n\n" "$S_CORNER_BOTTOM" "$S_BAR_H" "$1"
}

# Section header connected to the session guide
# Usage: clack_section "Section title"
clack_section() {
    clack_guide
    printf "${GRAY}%s${MAGENTA}%s${NC} ${BOLD}%s${NC}\n" "$S_BRANCH" "$S_BAR_H" "$1"
}

# Log with guide line - success
# Usage: clack_success "Message"
clack_success() {
    printf "${GRAY}%s${NC}  ${GREEN}%s${NC} %s\n" "$S_BAR" "$S_CHECK" "$1"
}

# Log with guide line - error
# Usage: clack_error "Message"
clack_error() {
    printf "${GRAY}%s${NC}  ${RED}%s${NC} %s\n" "$S_BAR" "$S_CROSS" "$1"
    clack_flush_detail
}

# Log with guide line - warning
# Usage: clack_warn "Message"
clack_warn() {
    printf "${GRAY}%s${NC}  ${YELLOW}!${NC} %s\n" "$S_BAR" "$1"
    clack_flush_detail
}

# Log with guide line - info
# Usage: clack_info "Message"
clack_info() {
    printf "${GRAY}%s${NC}  ${CYAN}%s${NC} %s\n" "$S_BAR" "$S_ARROW" "$1"
}

# Log with guide line - skipped
# Usage: clack_skip "Message"
clack_skip() {
    printf "${GRAY}%s${NC}  ${GRAY}%s %s${NC}\n" "$S_BAR" "$S_CIRCLE" "$1"
}

# Log with guide line - plain text
# Usage: clack_log "Message"
clack_log() {
    if [ -n "$1" ]; then
        printf "${GRAY}%s${NC}  %b\n" "$S_BAR" "$1"
    else
        clack_guide
    fi
}

# Remove terminal color/control sequences before third-party diagnostics are
# placed inside our own tree. This prevents captured ANSI red from leaking.
clack_strip_ansi() {
    _csa_esc=$(printf '\033')
    sed "s/${_csa_esc}\\[[0-9;?]*[[:alpha:]]//g" | tr -d '\r'
}

setup_log_init() {
    if [ -n "$SETUP_LOG_FILE" ]; then
        if : >> "$SETUP_LOG_FILE" 2>/dev/null; then
            chmod 600 "$SETUP_LOG_FILE" 2>/dev/null || true
            return 0
        fi
        SETUP_LOG_FILE=""
        return 1
    fi

    _sli_dir="${TMPDIR:-/tmp}"
    SETUP_LOG_FILE=$(mktemp "$_sli_dir/markitai-setup.XXXXXX.log" 2>/dev/null) || {
        SETUP_LOG_FILE=""
        return 1
    }
    SETUP_LOG_AUTO=true
    chmod 600 "$SETUP_LOG_FILE" 2>/dev/null || true
}

clack_record_detail() {
    _crd_context="$1"
    _crd_text="$2"
    [ -n "$_crd_text" ] || return 0
    setup_log_init || return 0
    {
        printf '\n== %s ==\n' "$_crd_context"
        printf '%s\n' "$_crd_text" | clack_strip_ansi
    } >> "$SETUP_LOG_FILE" 2>/dev/null || true
}

clack_record_file() {
    _crf_context="$1"
    _crf_file="$2"
    [ -s "$_crf_file" ] || return 0
    setup_log_init || return 0
    {
        printf '\n== %s ==\n' "$_crf_context"
        clack_strip_ansi < "$_crf_file"
    } >> "$SETUP_LOG_FILE" 2>/dev/null || true
}

clack_print_detail() {
    _cpd_text="$1"
    _cpd_limit="${2:-6}"
    [ -n "$_cpd_text" ] || return 0

    printf '%s\n' "$_cpd_text" \
        | clack_strip_ansi \
        | awk 'NF { print }' \
        | tail -n "$_cpd_limit" \
        | cut -c 1-300 \
        | while IFS= read -r _cpd_line; do
            printf "${GRAY}%s${NC}    ${DIM}%s${NC}\n" "$S_BAR" "$_cpd_line"
        done
}

# Preserve the complete diagnostic in a private log, but print only its tail.
# Usage: clack_detail "multiline text" [max lines] [context]
clack_detail() {
    _cd_text="$1"
    _cd_limit="${2:-6}"
    _cd_context="${3:-diagnostic}"
    [ -n "$_cd_text" ] || return 0
    clack_record_detail "$_cd_context" "$_cd_text"
    clack_print_detail "$_cd_text" "$_cd_limit"
}

clack_show_error_log() {
    [ -n "$SETUP_LOG_FILE" ] || return 0
    [ "$SETUP_LOG_SHOWN" = true ] && return 0
    SETUP_LOG_SHOWN=true
    clack_info "$(i18n info_error_log): $SETUP_LOG_FILE"
}

# Spinner errors are held until the caller prints its concise error/warning,
# keeping diagnostics visually nested beneath the status that explains them.
clack_flush_detail() {
    [ -n "${CLACK_PENDING_DETAIL:-}" ] || return 0
    _cfd_detail="$CLACK_PENDING_DETAIL"
    CLACK_PENDING_DETAIL=""
    clack_print_detail "$_cfd_detail" 6
}

# Spinner with guide line
# Usage: clack_spinner "message" command args...
# Shows spinner while command runs, then shows result
# On failure, displays last lines of stderr for debugging
clack_spinner() {
    _cs_message="$1"
    shift

    _cs_pid=""
    _cs_dynamic=false
    CLACK_PENDING_DETAIL=""
    # No predictable /tmp fallback (avoids symlink attacks); discard stderr instead
    _cs_errfile=$(mktemp 2>/dev/null) || _cs_errfile="/dev/null"
    CLACK_SPINNER_ERRFILE="$_cs_errfile"

    # Cursor animation is only safe on a real terminal. CI/log output gets one
    # stable progress row instead of hundreds of carriage-return frames.
    if [ -t 1 ] && [ "${TERM:-}" != "dumb" ] && [ "${CI:-}" != "true" ]; then
        _cs_dynamic=true
        (
            while true; do
                for _cs_frame in '◒' '◐' '◓' '◑'; do
                    printf "\r${GRAY}%s${NC}  ${CYAN}%s${NC} %s" "$S_BAR" "$_cs_frame" "$_cs_message"
                    sleep 0.1 2>/dev/null || sleep 1
                done
            done
        ) &
        _cs_pid=$!
        CLACK_SPINNER_PID="$_cs_pid"
    else
        clack_info "$_cs_message"
    fi

    # Guard the command explicitly so `set -e` can never skip spinner cleanup.
    if "$@" >/dev/null 2>"$_cs_errfile"; then
        _cs_status=0
    else
        _cs_status=$?
    fi

    if [ "$_cs_dynamic" = true ]; then
        kill "$_cs_pid" 2>/dev/null || true
        wait "$_cs_pid" 2>/dev/null || true
        printf "\r\033[K"
    fi
    CLACK_SPINNER_PID=""

    # Preserve the complete stderr, then hold only its tail for terminal output.
    if [ "$_cs_status" -ne 0 ] && [ -s "$_cs_errfile" ]; then
        clack_record_file "$_cs_message" "$_cs_errfile"
        CLACK_PENDING_DETAIL=$(tail -n 10 "$_cs_errfile" 2>/dev/null || true)
    fi

    if [ "$_cs_errfile" != "/dev/null" ]; then
        rm -f "$_cs_errfile" 2>/dev/null || true
    fi
    CLACK_SPINNER_ERRFILE=""
    return "$_cs_status"
}

# Run a command silently while preserving complete stdout/stderr on failure.
# Unlike the spinner, callers are expected to have already printed a status row.
clack_run_quiet() {
    _crq_context="$1"
    shift
    CLACK_PENDING_DETAIL=""
    _crq_file=$(mktemp 2>/dev/null) || _crq_file="/dev/null"

    if "$@" >"$_crq_file" 2>&1; then
        _crq_status=0
    else
        _crq_status=$?
        if [ "$_crq_file" != "/dev/null" ] && [ -s "$_crq_file" ]; then
            clack_record_file "$_crq_context" "$_crq_file"
            CLACK_PENDING_DETAIL=$(tail -n 10 "$_crq_file" 2>/dev/null || true)
        fi
    fi

    if [ "$_crq_file" != "/dev/null" ]; then
        rm -f "$_crq_file" 2>/dev/null || true
    fi
    return "$_crq_status"
}

run_remote_shell_installer() {
    _rsi_context="$1"
    _rsi_url="$2"
    _rsi_file=$(mktemp 2>/dev/null) || return 1
    # shellcheck disable=SC2016 # $1/$2 expand in the child sh, not here
    if clack_run_quiet "$_rsi_context" \
        sh -c 'curl -fsSL -o "$1" "$2" && bash "$1"' sh "$_rsi_file" "$_rsi_url"; then
        _rsi_status=0
    else
        _rsi_status=$?
    fi
    rm -f "$_rsi_file" 2>/dev/null || true
    return "$_rsi_status"
}

# Confirm prompt connected to the session guide
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

    clack_guide
    printf "${GRAY}%s${CYAN}%s${NC} %s ${GRAY}[%b]${NC} " "$S_BRANCH" "$S_BAR_H" "$_cc_prompt" "$_cc_hint"

    # Read from /dev/tty for piped execution support. A terminal echoes its own
    # newline, so printing another one creates the stray blank rows seen with
    # `curl | sh`; only redirected/headless output needs an explicit value.
    _cc_answer=""
    if [ -t 0 ]; then
        if read -r _cc_answer; then
            [ -t 1 ] || printf "%s\n" "${_cc_answer:-$_cc_default}"
        else
            _cc_answer="$_cc_default"
            printf "%s\n" "$_cc_answer"
        fi
    elif { read -r _cc_answer < /dev/tty; } 2>/dev/null; then
        [ -t 1 ] || printf "%s\n" "${_cc_answer:-$_cc_default}"
    else
        _cc_answer="$_cc_default"
        printf "%s\n" "$_cc_answer"
    fi

    if [ -z "$_cc_answer" ]; then
        _cc_answer="$_cc_default"
    fi

    case "$_cc_answer" in
        [Yy]*) return 0 ;;
        *) return 1 ;;
    esac
}

# Return success when an interactive terminal is available. Piped installers
# may still have a controlling terminal at /dev/tty, so stdin alone is not
# enough to decide whether prompting is safe.
has_interactive_tty() {
    if [ -t 0 ]; then
        return 0
    fi
    ( : </dev/tty ) 2>/dev/null
}

# Headless installs must opt in before any optional package, browser binary,
# system dependency, or third-party CLI is installed.
optional_install_requested() {
    case "${MARKITAI_INSTALL_OPTIONAL:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

# Interactive runs keep each component's normal default. Without a terminal,
# every optional component defaults to off unless MARKITAI_INSTALL_OPTIONAL is
# explicitly truthy.
clack_confirm_optional() {
    if ! has_interactive_tty; then
        if optional_install_requested; then
            return 0
        fi
        return 1
    fi
    clack_confirm "$1" "$2"
}

# Compact note block. Content stays on the single outer guide instead of
# drawing a second, competing box down the left side.
# Usage: clack_note "title" "line1" "line2" ...
# Or:    clack_note "title" <<EOF
#        line1
#        line2
#        EOF
# Note: clack_log uses %b to interpret color escapes in lines.
clack_note() {
    _cn_title="$1"
    shift

    clack_guide
    printf "${GRAY}%s${GREEN}%s${NC} ${BOLD}%s${NC}\n" "$S_BRANCH" "$S_BAR_H" "$_cn_title"

    if [ $# -gt 0 ]; then
        for _cn_line in "$@"; do
            clack_log "$_cn_line"
        done
    else
        while IFS= read -r _cn_line; do
            clack_log "$_cn_line"
        done
    fi
}

# Cancel message
# Usage: clack_cancel "Message"
clack_cancel() {
    clack_show_error_log
    CLACK_SESSION_CLOSED=true
    clack_guide
    printf "${GRAY}%s%s${NC} ${RED}%s${NC}\n\n" "$S_CORNER_BOTTOM" "$S_BAR_H" "$1"
}

# Last-resort guard for an unanticipated `set -e`, signal, or shell failure.
# Expected failure paths call clack_cancel first and therefore are not repeated.
setup_on_exit() {
    _soe_status=$?
    trap - 0 HUP INT TERM

    if [ -n "${CLACK_SPINNER_PID:-}" ]; then
        kill "$CLACK_SPINNER_PID" 2>/dev/null || true
        wait "$CLACK_SPINNER_PID" 2>/dev/null || true
        if [ -t 1 ]; then
            printf '\r\033[K'
        fi
    fi
    if [ -n "${CLACK_SPINNER_ERRFILE:-}" ] && [ "$CLACK_SPINNER_ERRFILE" != "/dev/null" ]; then
        if [ -s "$CLACK_SPINNER_ERRFILE" ]; then
            clack_record_file "$(i18n error_unexpected)" "$CLACK_SPINNER_ERRFILE"
            CLACK_PENDING_DETAIL=$(tail -n 10 "$CLACK_SPINNER_ERRFILE" 2>/dev/null || true)
        fi
        rm -f "$CLACK_SPINNER_ERRFILE" 2>/dev/null || true
    fi

    if [ "$_soe_status" -ne 0 ] && [ "${CLACK_SESSION_ACTIVE:-false}" = true ] \
        && [ "${CLACK_SESSION_CLOSED:-false}" != true ]; then
        clack_error "$(i18n error_unexpected)"
        clack_cancel "$(i18n error_setup_failed)"
    fi

    exit "$_soe_status"
}

setup_on_signal() {
    exit "$1"
}

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
# Utility Functions
# ============================================================

# Get project root directory
# Returns: absolute path to project root, or empty if not in project
get_project_root() {
    # Start from current directory
    _dir="$PWD"

    while [ "$_dir" != "/" ]; do
        # Check for markitai project indicators
        if [ -f "$_dir/pyproject.toml" ] && [ -d "$_dir/.git" ]; then
            # Verify it's actually markitai project
            if grep -q "markitai" "$_dir/pyproject.toml" 2>/dev/null; then
                echo "$_dir"
                return 0
            fi
        fi
        _dir=$(dirname "$_dir")
    done

    return 1
}

# Check if running as root and warn
# Returns: 0 to continue, exits if user declines
warn_if_root() {
    if [ "$(id -u)" -eq 0 ]; then
        clack_warn "$(i18n warn_root)"
        clack_log "$(i18n warn_root_risk)"

        if ! clack_confirm "$(i18n confirm_continue_as_root)" "n"; then
            clack_cancel "$(i18n error_setup_failed)"
            exit 1
        fi
    fi
    return 0
}

# Check for proxy environment variables
# Returns: 0 if proxy detected, 1 if not
detect_proxy() {
    if [ -n "${HTTPS_PROXY:-}" ] || [ -n "${HTTP_PROXY:-}" ] || [ -n "${ALL_PROXY:-}" ] || \
       [ -n "${https_proxy:-}" ] || [ -n "${http_proxy:-}" ] || [ -n "${all_proxy:-}" ]; then
        return 0
    fi
    return 1
}

# Prompt user to enable China mirrors if no proxy is detected
# Sets: UV_INDEX_URL, PLAYWRIGHT_DOWNLOAD_HOST, NPM_CONFIG_REGISTRY
configure_mirrors() {
    if detect_proxy; then
        return 0
    fi

    clack_warn "$(i18n mirror_no_proxy)"

    if ! clack_confirm "$(i18n mirror_confirm)" "n"; then
        clack_log "$(i18n mirror_skipped)"
        return 0
    fi

    # Show mirror source selection on the same continuous tree guide.
    clack_guide
    printf "${GRAY}%s${CYAN}%s${NC} %s ${GRAY}[1]${NC}\n" "$S_BRANCH" "$S_BAR_H" "$(i18n mirror_select)"
    printf "${GRAY}%s${NC}  ${CYAN}1.${NC} %s\n" "$S_BAR" "$(i18n mirror_tuna)"
    printf "${GRAY}%s${NC}  ${GRAY}2.${NC} %s\n" "$S_BAR" "$(i18n mirror_aliyun)"
    printf "${GRAY}%s${NC}  ${GRAY}3.${NC} %s\n" "$S_BAR" "$(i18n mirror_tencent)"
    printf "${GRAY}%s${NC}  ${GRAY}4.${NC} %s\n" "$S_BAR" "$(i18n mirror_huawei)"
    printf "${GRAY}%s${NC}  > " "$S_BAR"

    # Fall back to the default choice when no TTY is available (CI, cron).
    # As with confirmations, do not add a second newline on terminal output.
    _mirror_choice=""
    if [ -t 0 ]; then
        if read -r _mirror_choice; then
            [ -t 1 ] || printf "%s\n" "${_mirror_choice:-1}"
        else
            _mirror_choice="1"
            printf "%s\n" "$_mirror_choice"
        fi
    elif { read -r _mirror_choice < /dev/tty; } 2>/dev/null; then
        [ -t 1 ] || printf "%s\n" "${_mirror_choice:-1}"
    else
        _mirror_choice="1"
        printf "%s\n" "$_mirror_choice"
    fi

    [ -z "$_mirror_choice" ] && _mirror_choice="1"

    case "$_mirror_choice" in
        2)
            export UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
            export NPM_CONFIG_REGISTRY="https://registry.npmmirror.com"
            ;;
        3)
            export UV_INDEX_URL="https://mirrors.cloud.tencent.com/pypi/simple"
            export NPM_CONFIG_REGISTRY="https://mirrors.cloud.tencent.com/npm/"
            ;;
        4)
            export UV_INDEX_URL="https://repo.huaweicloud.com/repository/pypi/simple"
            export NPM_CONFIG_REGISTRY="https://mirrors.huaweicloud.com/repository/npm/"
            ;;
        *)
            export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
            export NPM_CONFIG_REGISTRY="https://registry.npmmirror.com"
            ;;
    esac

    # Playwright: only npmmirror CDN provides reliable browser binary mirrors
    export PLAYWRIGHT_DOWNLOAD_HOST="https://cdn.npmmirror.com/binaries/playwright"

    clack_success "$(i18n mirror_enabled)"
    clack_log "  $(i18n mirror_pypi): $UV_INDEX_URL"
    clack_log "  $(i18n mirror_npm): $NPM_CONFIG_REGISTRY"
    clack_log "  $(i18n mirror_playwright): $PLAYWRIGHT_DOWNLOAD_HOST"
}

# ============================================================
# Mode Detection
# ============================================================

# Check if running in development mode
# Returns: 0 if dev mode, 1 if user mode
is_dev_mode() {
    # Check if we're in the markitai project directory
    # Note: .git can be a directory (normal repo) or file (worktree)
    if [ -f "./pyproject.toml" ] && [ -e "./.git" ] && [ -d "./scripts" ]; then
        if grep -q "markitai" "./pyproject.toml" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# ============================================================
# Install Source Selection (PyPI vs local repo)
# ============================================================

# Install source: "pypi" (default) or "local" (from source repo)
MARKITAI_SOURCE="pypi"
MARKITAI_LOCAL_PATH=""
MARKITAI_REPO_ROOT=""

# Detect the markitai source repo (for the local install option)
# Checks the script location first, then cwd (curl | sh case)
# Sets: MARKITAI_REPO_ROOT
# Returns: 0 if repo detected, 1 if not
detect_markitai_repo() {
    MARKITAI_REPO_ROOT=""

    # 1) Via script location ($0 is not a file under curl | sh)
    if [ -f "$0" ]; then
        _dmr_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" 2>/dev/null && pwd) || _dmr_dir=""
        if [ -n "$_dmr_dir" ]; then
            _dmr_root=$(dirname -- "$_dmr_dir")
            if [ -f "$_dmr_root/packages/markitai/pyproject.toml" ] && \
               grep -q '^name = "markitai"' "$_dmr_root/packages/markitai/pyproject.toml" 2>/dev/null; then
                MARKITAI_REPO_ROOT="$_dmr_root"
                return 0
            fi
        fi
    fi

    # 2) Via current directory (curl | sh run from inside a checkout)
    if [ -f "$PWD/packages/markitai/pyproject.toml" ] && \
       grep -q '^name = "markitai"' "$PWD/packages/markitai/pyproject.toml" 2>/dev/null; then
        MARKITAI_REPO_ROOT="$PWD"
        return 0
    fi

    return 1
}

# Ask whether to install markitai from the local repo or PyPI
# Default (enter / non-interactive / not in repo) = PyPI
# Sets: MARKITAI_SOURCE, MARKITAI_LOCAL_PATH
choose_install_source() {
    if ! detect_markitai_repo; then
        return 0
    fi

    # Non-interactive (stdin not a TTY, e.g. curl | sh): never prompt
    if [ ! -t 0 ]; then
        clack_info "$(i18n info_repo_noninteractive)"
        return 0
    fi

    clack_info "$(i18n repo_detected): $MARKITAI_REPO_ROOT"
    if clack_confirm "$(i18n confirm_local_install)" "n"; then
        MARKITAI_SOURCE="local"
        MARKITAI_LOCAL_PATH="$MARKITAI_REPO_ROOT/packages/markitai"
        clack_info "$(i18n source_local)"
    else
        clack_info "$(i18n source_pypi)"
    fi
    return 0
}

# Build the markitai package spec honoring install source and extras
# Usage: markitai_pkg_spec "extra1,extra2"
markitai_pkg_spec() {
    if [ -n "$1" ]; then
        _mi_base="markitai[$1]"
    else
        _mi_base="markitai"
    fi

    if [ "$MARKITAI_SOURCE" = "local" ]; then
        echo "$_mi_base @ $MARKITAI_LOCAL_PATH"
    elif [ -n "$MARKITAI_VERSION" ]; then
        echo "$_mi_base==$MARKITAI_VERSION"
    else
        echo "$_mi_base"
    fi
}

# ============================================================
# Version Variables (can be overridden via environment)
# ============================================================
MARKITAI_VERSION="${MARKITAI_VERSION:-}"
UV_VERSION="${UV_VERSION:-}"
# Set to 1/true/yes/on to install optional extras and components when no TTY
# is available. Interactive runs continue to prompt component by component.
MARKITAI_INSTALL_OPTIONAL="${MARKITAI_INSTALL_OPTIONAL:-}"

# Global variable for Python command path
PYTHON_CMD=""

# ============================================================
# Platform Detection
# ============================================================

# Detect OS type
# Sets: OS_TYPE (Darwin, Linux, etc.)
detect_os() {
    OS_TYPE=$(uname -s)
}

detect_os

# ============================================================
# Installation Functions
# ============================================================

# Install uv package manager
# Returns: 0 on success, 1 on failure
install_uv() {
    # Check if already installed
    if command -v uv >/dev/null 2>&1; then
        _uv_version=$(uv --version 2>/dev/null | head -n1 | tr -d '\r')
        clack_success "$(i18n uv): $_uv_version $(i18n already_installed)"
        track_install "uv" "installed"
        return 0
    fi

    # Prompt to install
    if ! clack_confirm "$(i18n confirm_uv)" "y"; then
        clack_skip "$(i18n uv)"
        track_install "uv" "skipped"
        return 1
    fi

    # Check curl availability
    if ! command -v curl >/dev/null 2>&1; then
        clack_error "curl not found"
        track_install "uv" "failed"
        return 1
    fi

    # Build install URL (with optional version)
    if [ -n "$UV_VERSION" ]; then
        _uv_url="https://astral.sh/uv/$UV_VERSION/install.sh"
    else
        _uv_url="https://astral.sh/uv/install.sh"
    fi

    # Install uv: download the installer to a temp file first, so a partial
    # download is never executed (piping curl into sh would also conflict
    # with clack_spinner, which redirects the command's stdout)
    _uv_installer=$(mktemp) || {
        clack_error "$(i18n uv) $(i18n failed)"
        track_install "uv" "failed"
        return 1
    }
    _uv_ok=false
    # shellcheck disable=SC2016 # $1/$2 expand in the child sh, not here
    if clack_spinner "$(i18n installing) $(i18n uv)..." \
        sh -c 'curl -LsSf -o "$1" "$2" && sh "$1"' sh "$_uv_installer" "$_uv_url"; then
        _uv_ok=true
    fi
    rm -f "$_uv_installer"

    if [ "$_uv_ok" = true ]; then
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        if command -v uv >/dev/null 2>&1; then
            _uv_version=$(uv --version 2>/dev/null | head -n1 | tr -d '\r')
            clack_success "$(i18n uv): $_uv_version $(i18n installed)"
            track_install "uv" "installed"
            return 0
        fi
    fi

    clack_error "$(i18n uv) $(i18n failed)"
    track_install "uv" "failed"
    return 1
}

# Detect/install Python via uv
# Sets: PYTHON_CMD
# Returns: 0 on success, 1 on failure
detect_python() {
    if ! command -v uv >/dev/null 2>&1; then
        clack_error "$(i18n error_uv_required)"
        return 1
    fi

    # Try to find any supported Python (3.11-3.13)
    _py_path=$(uv python find '>=3.11,<3.14' 2>/dev/null) || _py_path=""
    if [ -n "$_py_path" ] && [ -x "$_py_path" ]; then
        PYTHON_CMD="$_py_path"
        _py_ver=$("$_py_path" -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>/dev/null)
        clack_success "$(i18n python) $_py_ver"
        return 0
    fi

    # Not found, auto-install (3.13 as default)
    clack_info "$(i18n installing) $(i18n python) 3.13..."
    if clack_run_quiet "$(i18n installing) $(i18n python) 3.13" uv python install 3.13; then
        _py_path=$(uv python find 3.13 2>/dev/null) || _py_path=""
        if [ -n "$_py_path" ] && [ -x "$_py_path" ]; then
            PYTHON_CMD="$_py_path"
            _py_ver=$("$_py_path" -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>/dev/null)
            clack_success "$(i18n python) $_py_ver $(i18n installed)"
            return 0
        fi
    fi

    clack_error "$(i18n error_python_required)"
    return 1
}

# Install markitai (User mode)
# Requires: PYTHON_CMD to be set
# Returns: 0 on success, 1 on failure
install_markitai() {
    _uv_tools_dir=$(markitai_tools_dir)

    # Build package spec with all tracked extras
    _mi_pkg=$(markitai_pkg_spec "$MARKITAI_EXTRAS")

    if ! command -v uv >/dev/null 2>&1; then
        clack_error "$(i18n markitai) $(i18n failed)"
        track_install "markitai" "failed"
        return 1
    fi

    # Upgrade existing installation, or fresh install
    if [ -d "$_uv_tools_dir/markitai" ]; then
        # An unpinned PyPI install can use the receipt-based upgrade path.
        # Explicit versions and local sources must replace the receipt with
        # the exact package spec assembled above.
        if [ "$MARKITAI_SOURCE" != "local" ] && [ -z "$MARKITAI_VERSION" ] \
            && ! markitai_extras_need_update; then
            if clack_spinner "$(i18n installing) $(i18n markitai)..." uv tool upgrade markitai; then
                export PATH="$HOME/.local/bin:$PATH"
                _mi_version=$(markitai --version 2>/dev/null | awk '{print $NF}' || echo "")
                clack_success "$(i18n markitai) $_mi_version"
                track_install "markitai" "installed"
                return 0
            fi
        fi
        # Upgrade failed (or local source), try force reinstall
        if clack_spinner "$(i18n installing) $(i18n markitai)..." uv tool install "$_mi_pkg" --python "$PYTHON_CMD" --force; then
            export PATH="$HOME/.local/bin:$PATH"
            _mi_version=$(markitai --version 2>/dev/null | awk '{print $NF}' || echo "")
            if [ "$MARKITAI_SOURCE" != "local" ] && [ -n "$MARKITAI_VERSION" ] \
                && [ "$_mi_version" != "$MARKITAI_VERSION" ]; then
                clack_error "$(i18n markitai) version mismatch: expected $MARKITAI_VERSION, got ${_mi_version:-unknown}"
                track_install "markitai" "failed"
                return 1
            fi
            clack_success "$(i18n markitai) $_mi_version"
            track_install "markitai" "installed"
            return 0
        fi
    else
        # Fresh install
        if clack_spinner "$(i18n installing) $(i18n markitai)..." uv tool install "$_mi_pkg" --python "$PYTHON_CMD"; then
            export PATH="$HOME/.local/bin:$PATH"
            _mi_version=$(markitai --version 2>/dev/null | awk '{print $NF}' || echo "")
            if [ "$MARKITAI_SOURCE" != "local" ] && [ -n "$MARKITAI_VERSION" ] \
                && [ "$_mi_version" != "$MARKITAI_VERSION" ]; then
                clack_error "$(i18n markitai) version mismatch: expected $MARKITAI_VERSION, got ${_mi_version:-unknown}"
                track_install "markitai" "failed"
                return 1
            fi
            clack_success "$(i18n markitai) $_mi_version"
            track_install "markitai" "installed"
            return 0
        fi
    fi

    clack_error "$(i18n markitai) $(i18n failed)"
    track_install "markitai" "failed"
    return 1
}

# Global variable tracking all needed extras (comma-separated). Fresh
# headless installs start with the core package; the explicit opt-in flag or
# an interactive session keeps the guided browser-extra default.
MARKITAI_EXTRAS=""
MARKITAI_RECEIPT_EXTRAS=""
MARKITAI_ALL_FALLBACK_EXTRAS="browser,extra-fetch,kreuzberg,svg,heif,serve"
if has_interactive_tty || optional_install_requested; then
    MARKITAI_EXTRAS="browser"
fi

markitai_tools_dir() {
    uv tool dir 2>/dev/null || echo "$HOME/.local/share/uv/tools"
}

# Return success when an extra is already covered by the combined spec.
# `all` is canonical and is a superset of every individual extra, including
# `serve`; never generate redundant requirements such as `markitai[all,serve]`.
markitai_extra_enabled() {
    _extra_name="$1"
    [ "$MARKITAI_EXTRAS" = "all" ] && return 0
    case ",$MARKITAI_EXTRAS," in
        *",$_extra_name,"*) return 0 ;;
        *) return 1 ;;
    esac
}

# Track a markitai extra for the next combined installation.
# Usage: install_markitai_extra "claude-agent"
install_markitai_extra() {
    _extra_name="$1"
    [ -z "$_extra_name" ] && return 0
    if [ "$_extra_name" = "all" ]; then
        MARKITAI_EXTRAS="all"
        return 0
    fi
    markitai_extra_enabled "$_extra_name" && return 0
    if [ -z "$MARKITAI_EXTRAS" ]; then
        MARKITAI_EXTRAS="$_extra_name"
    else
        MARKITAI_EXTRAS="${MARKITAI_EXTRAS},$_extra_name"
    fi
}

# Preserve every extra recorded by uv before asking for new capabilities.
load_existing_markitai_extras() {
    _uv_tools_dir=$(markitai_tools_dir)
    _receipt_file="$_uv_tools_dir/markitai/uv-receipt.toml"
    MARKITAI_RECEIPT_EXTRAS=""
    if [ ! -f "$_receipt_file" ]; then
        return 0
    fi

    # Extract extras array: extras = ["browser", "copilot", ...]
    MARKITAI_RECEIPT_EXTRAS=$(grep -o 'extras = \[[^]]*\]' "$_receipt_file" 2>/dev/null \
        | head -n 1 | sed 's/extras = \[//;s/\]//;s/"//g;s/ //g')
    _old_ifs="$IFS"; IFS=','
    for _e in $MARKITAI_RECEIPT_EXTRAS; do
        [ -n "$_e" ] && install_markitai_extra "$_e"
    done
    IFS="$_old_ifs"
}

markitai_receipt_has_extra() {
    _extra_name="$1"
    [ "$MARKITAI_RECEIPT_EXTRAS" = "all" ] && return 0
    case ",$MARKITAI_RECEIPT_EXTRAS," in
        *",$_extra_name,"*) return 0 ;;
        *) return 1 ;;
    esac
}

# Return success when the combined target contains an extra absent from the
# current uv receipt. In that case `uv tool upgrade` is insufficient because
# it only upgrades the old receipt and ignores newly selected extras.
markitai_extras_need_update() {
    _old_ifs="$IFS"; IFS=','
    for _extra in $MARKITAI_EXTRAS; do
        if [ -n "$_extra" ] && ! markitai_receipt_has_extra "$_extra"; then
            IFS="$_old_ifs"
            return 0
        fi
    done
    IFS="$_old_ifs"
    return 1
}

# Resolve Web UI support before installing markitai so `serve` joins the same
# package spec as browser/all/existing extras instead of replacing them later.
select_markitai_serve() {
    markitai_extra_enabled "serve" && return 0
    if clack_confirm_optional "$(i18n confirm_serve)" "y"; then
        install_markitai_extra "serve"
    fi
}

track_markitai_serve() {
    if markitai_extra_enabled "serve"; then
        track_install "serve" "installed"
    else
        track_install "serve" "skipped"
    fi
}

# Finalize markitai extras after all optional components are resolved.
# Merges `markitai doctor --suggest-extras` output with manually tracked
# MARKITAI_EXTRAS (from CLI install functions), so nothing is lost.
finalize_markitai_extras() {
    # Do not silently add optional Python extras during a headless core install.
    if ! has_interactive_tty && ! optional_install_requested; then
        return 0
    fi

    # Merge suggested extras INTO manually tracked set (not replace)
    _suggested=$(markitai doctor --suggest-extras 2>/dev/null || true)
    if [ -n "$_suggested" ]; then
        _old_ifs="$IFS"; IFS=','
        for _e in $_suggested; do
            install_markitai_extra "$_e"
        done
        IFS="$_old_ifs"
    fi

    # Refresh the receipt after the initial install, then compare exact extra
    # names (with `all` treated as a superset).
    load_existing_markitai_extras
    if ! markitai_extras_need_update; then
        return 0
    fi

    # Reinstall with all extras (progressive fallback on failure)
    _mi_pkg=$(markitai_pkg_spec "$MARKITAI_EXTRAS")
    _uv_err=""
    if ! _uv_err=$(uv tool install "$_mi_pkg" --python "$PYTHON_CMD" --force 2>&1); then
        clack_record_detail "$(i18n markitai) extras" "$_uv_err"
        # Full install failed — retry without SDK-dependent extras
        _safe_extras=""
        _skipped=""
        _old_ifs="$IFS"; IFS=','
        for _e in $MARKITAI_EXTRAS; do
            case "$_e" in
                all)
                    _safe_extras="$MARKITAI_ALL_FALLBACK_EXTRAS"
                    _skipped="claude-agent, copilot"
                    ;;
                claude-agent|copilot) _skipped="${_skipped:+$_skipped, }$_e" ;;
                *) _safe_extras="${_safe_extras:+$_safe_extras,}$_e" ;;
            esac
        done
        IFS="$_old_ifs"

        if [ -n "$_safe_extras" ]; then
            _mi_pkg=$(markitai_pkg_spec "$_safe_extras")
            if clack_run_quiet "$(i18n markitai) extras fallback" \
                uv tool install "$_mi_pkg" --python "$PYTHON_CMD" --force; then
                [ -n "$_skipped" ] && clack_warn "$(i18n skipped) extras: $_skipped (SDK $(i18n not_found))"
            else
                clack_warn "$(i18n markitai) extras update $(i18n failed)"
            fi
        else
            clack_warn "$(i18n markitai) extras update $(i18n failed)"
            clack_print_detail "$_uv_err" 3
        fi
    fi
}

# Sync project dependencies (Dev mode)
# Returns: 0 on success, 1 on failure
sync_dependencies() {
    _project_root=$(get_project_root)
    if [ -z "$_project_root" ]; then
        clack_error "$(i18n error_setup_failed)"
        return 1
    fi

    clack_info "$(i18n info_project_dir): $_project_root"

    cd "$_project_root" || return 1

    if clack_spinner "$(i18n info_syncing_deps)" uv sync --all-groups --all-extras --python "$PYTHON_CMD"; then
        clack_success "$(i18n info_deps_synced)"
        track_install "python_deps" "installed"
        return 0
    else
        clack_error "$(i18n python_deps) $(i18n failed)"
        track_install "python_deps" "failed"
        return 1
    fi
}

# Install pre-commit hooks (Dev mode)
# Returns: 0 on success, 1 on failure
install_precommit() {
    _project_root=$(get_project_root)
    if [ -z "$_project_root" ]; then
        return 1
    fi

    cd "$_project_root" || return 1

    if [ ! -f ".pre-commit-config.yaml" ]; then
        clack_skip "$(i18n precommit)"
        return 0
    fi

    if clack_spinner "$(i18n installing) $(i18n precommit)..." uv run pre-commit install; then
        clack_success "$(i18n info_precommit_installed)"
        track_install "precommit" "installed"
        return 0
    else
        clack_warn "$(i18n precommit) $(i18n failed)"
        return 1
    fi
}

# Check if Playwright browser is installed
# Returns: 0 if installed, 1 if not
detect_playwright_browser() {
    _pw_cache=""
    case "$OS_TYPE" in
        Darwin)
            _pw_cache="$HOME/Library/Caches/ms-playwright"
            ;;
        Linux)
            _pw_cache="$HOME/.cache/ms-playwright"
            ;;
    esac

    if [ -n "$_pw_cache" ] && [ -d "$_pw_cache" ]; then
        if ls "$_pw_cache"/chromium-* >/dev/null 2>&1; then
            return 0
        fi
    fi

    return 1
}

# Install Playwright browser (Optional)
# Returns: 0 on success, 1 on failure, 2 if skipped
install_optional_playwright() {
    # Check if already installed
    if detect_playwright_browser; then
        clack_success "$(i18n playwright) $(i18n already_installed)"
        track_install "playwright" "installed"
        return 0
    fi

    clack_info "$(i18n info_playwright_purpose)"

    if ! clack_confirm_optional "$(i18n confirm_playwright)" "y"; then
        clack_skip "$(i18n playwright)"
        track_install "playwright" "skipped"
        return 2
    fi

    # Get playwright path from markitai's uv tool environment
    _pw_cmd=""
    if command -v uv >/dev/null 2>&1; then
        _uv_tool_dir=$(uv tool dir 2>/dev/null)
        if [ -n "$_uv_tool_dir" ] && [ -x "$_uv_tool_dir/markitai/bin/playwright" ]; then
            _pw_cmd="$_uv_tool_dir/markitai/bin/playwright"
        fi
    fi

    # Fallback to default path
    if [ -z "$_pw_cmd" ] || [ ! -x "$_pw_cmd" ]; then
        _pw_cmd="$HOME/.local/share/uv/tools/markitai/bin/playwright"
    fi

    # Install browser
    _browser_installed=false
    if [ -x "$_pw_cmd" ]; then
        if clack_spinner "$(i18n installing) $(i18n playwright)..." "$_pw_cmd" install chromium; then
            _browser_installed=true
        fi
    fi

    # Fallback to Python module
    if [ "$_browser_installed" = false ] && [ -n "$PYTHON_CMD" ]; then
        if clack_spinner "$(i18n installing) $(i18n playwright)..." "$PYTHON_CMD" -m playwright install chromium; then
            _browser_installed=true
        fi
    fi

    if [ "$_browser_installed" = false ]; then
        clack_error "$(i18n playwright) $(i18n failed)"
        track_install "playwright" "failed"
        return 1
    fi

    # On Linux, install system dependencies silently
    if [ "$OS_TYPE" = "Linux" ]; then
        if [ -f /etc/arch-release ]; then
            _arch_deps="nss nspr at-spi2-core cups libdrm mesa alsa-lib libxcomposite libxdamage libxrandr libxkbcommon pango cairo noto-fonts noto-fonts-cjk noto-fonts-emoji ttf-liberation"
            # shellcheck disable=SC2086 # Word splitting is intentional for package list
            sudo pacman -S --noconfirm --needed $_arch_deps >/dev/null 2>&1 || true
        elif command -v apt-get >/dev/null 2>&1; then
            if [ -x "$_pw_cmd" ]; then
                "$_pw_cmd" install-deps chromium >/dev/null 2>&1 || true
            elif [ -n "$PYTHON_CMD" ]; then
                "$PYTHON_CMD" -m playwright install-deps chromium >/dev/null 2>&1 || true
            fi
        fi
    fi

    clack_success "$(i18n playwright) $(i18n installed)"
    track_install "playwright" "installed"
    return 0
}

# Install LibreOffice (Optional)
# Returns: 0 on success, 1 on failure, 2 if skipped
install_optional_libreoffice() {
    # Check if already installed
    if command -v soffice >/dev/null 2>&1; then
        _lo_version=$(soffice --version 2>/dev/null | head -n1)
        clack_success "$(i18n libreoffice): $_lo_version"
        track_install "libreoffice" "installed"
        return 0
    fi

    if command -v libreoffice >/dev/null 2>&1; then
        _lo_version=$(libreoffice --version 2>/dev/null | head -n1)
        clack_success "$(i18n libreoffice): $_lo_version"
        track_install "libreoffice" "installed"
        return 0
    fi

    clack_info "$(i18n info_libreoffice_purpose)"

    if ! clack_confirm_optional "$(i18n confirm_libreoffice)" "n"; then
        clack_skip "$(i18n libreoffice)"
        track_install "libreoffice" "skipped"
        return 2
    fi

    clack_info "$(i18n installing) $(i18n libreoffice)..."

    case "$OS_TYPE" in
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                if clack_run_quiet "$(i18n installing) $(i18n libreoffice)" brew install --cask libreoffice; then
                    clack_success "$(i18n libreoffice) $(i18n installed)"
                    track_install "libreoffice" "installed"
                    return 0
                fi
            fi
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                if clack_run_quiet "apt update" sudo apt update && \
                   clack_run_quiet "$(i18n installing) $(i18n libreoffice)" sudo apt install -y libreoffice; then
                    clack_success "$(i18n libreoffice) $(i18n installed)"
                    track_install "libreoffice" "installed"
                    return 0
                fi
            elif [ -f /etc/fedora-release ]; then
                if clack_run_quiet "$(i18n installing) $(i18n libreoffice)" sudo dnf install -y libreoffice; then
                    clack_success "$(i18n libreoffice) $(i18n installed)"
                    track_install "libreoffice" "installed"
                    return 0
                fi
            elif [ -f /etc/arch-release ]; then
                if clack_run_quiet "$(i18n installing) $(i18n libreoffice)" sudo pacman -S --noconfirm libreoffice-fresh; then
                    clack_success "$(i18n libreoffice) $(i18n installed)"
                    track_install "libreoffice" "installed"
                    return 0
                fi
            fi
            ;;
    esac

    clack_error "$(i18n libreoffice) $(i18n failed)"
    track_install "libreoffice" "failed"
    return 1
}

# Install FFmpeg (Optional)
# Returns: 0 on success, 1 on failure, 2 if skipped
install_optional_ffmpeg() {
    # Check if already installed
    if command -v ffmpeg >/dev/null 2>&1; then
        _ff_version=$(ffmpeg -version 2>/dev/null | head -n1 | sed 's/ffmpeg version \([^ ]*\).*/\1/')
        clack_success "$(i18n ffmpeg): $_ff_version"
        track_install "ffmpeg" "installed"
        return 0
    fi

    clack_info "$(i18n info_ffmpeg_purpose)"

    if ! clack_confirm_optional "$(i18n confirm_ffmpeg)" "n"; then
        clack_skip "$(i18n ffmpeg)"
        track_install "ffmpeg" "skipped"
        return 2
    fi

    clack_info "$(i18n installing) $(i18n ffmpeg)..."

    case "$OS_TYPE" in
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                if clack_run_quiet "$(i18n installing) $(i18n ffmpeg)" brew install ffmpeg; then
                    clack_success "$(i18n ffmpeg) $(i18n installed)"
                    track_install "ffmpeg" "installed"
                    return 0
                fi
            fi
            ;;
        Linux)
            if [ -f /etc/debian_version ]; then
                if clack_run_quiet "apt update" sudo apt update && \
                   clack_run_quiet "$(i18n installing) $(i18n ffmpeg)" sudo apt install -y ffmpeg; then
                    clack_success "$(i18n ffmpeg) $(i18n installed)"
                    track_install "ffmpeg" "installed"
                    return 0
                fi
            elif [ -f /etc/fedora-release ]; then
                if clack_run_quiet "$(i18n installing) $(i18n ffmpeg)" sudo dnf install -y ffmpeg; then
                    clack_success "$(i18n ffmpeg) $(i18n installed)"
                    track_install "ffmpeg" "installed"
                    return 0
                fi
            elif [ -f /etc/arch-release ]; then
                if clack_run_quiet "$(i18n installing) $(i18n ffmpeg)" sudo pacman -S --noconfirm ffmpeg; then
                    clack_success "$(i18n ffmpeg) $(i18n installed)"
                    track_install "ffmpeg" "installed"
                    return 0
                fi
            fi
            ;;
    esac

    clack_error "$(i18n ffmpeg) $(i18n failed)"
    track_install "ffmpeg" "failed"
    return 1
}

# Install Claude Code CLI (Optional)
# Returns: 0 on success, 1 on failure, 2 if skipped
install_optional_claude_cli() {
    # Check if already installed
    if command -v claude >/dev/null 2>&1; then
        _cl_version=$(claude --version 2>/dev/null | head -n1)
        clack_success "$(i18n claude_cli): $_cl_version"
        install_markitai_extra "claude-agent"
        track_install "claude_cli" "installed"
        return 0
    fi

    if ! clack_confirm_optional "$(i18n confirm_claude_cli)" "n"; then
        clack_skip "$(i18n claude_cli)"
        track_install "claude_cli" "skipped"
        return 2
    fi

    clack_info "$(i18n installing) $(i18n claude_cli)..."

    # Try official install script in a child shell so its exit is contained.
    if run_remote_shell_installer "$(i18n installing) $(i18n claude_cli)" "https://claude.ai/install.sh"; then
        if command -v claude >/dev/null 2>&1; then
            clack_success "$(i18n claude_cli) $(i18n installed)"
            # Also install the SDK extra
            install_markitai_extra "claude-agent" || true
            track_install "claude_cli" "installed"
            return 0
        fi
    fi

    # Fallback: npm/pnpm
    if command -v pnpm >/dev/null 2>&1; then
        if clack_run_quiet "$(i18n installing) $(i18n claude_cli)" pnpm add -g @anthropic-ai/claude-code; then
            clack_success "$(i18n claude_cli) $(i18n installed)"
            install_markitai_extra "claude-agent" || true
            track_install "claude_cli" "installed"
            return 0
        fi
    elif command -v npm >/dev/null 2>&1; then
        if clack_run_quiet "$(i18n installing) $(i18n claude_cli)" npm install -g @anthropic-ai/claude-code; then
            clack_success "$(i18n claude_cli) $(i18n installed)"
            install_markitai_extra "claude-agent" || true
            track_install "claude_cli" "installed"
            return 0
        fi
    fi

    clack_error "$(i18n claude_cli) $(i18n failed)"
    track_install "claude_cli" "failed"
    return 1
}

# Install Copilot CLI (Optional)
# Returns: 0 on success, 1 on failure, 2 if skipped
install_optional_copilot_cli() {
    # Check if already installed
    if command -v copilot >/dev/null 2>&1; then
        _cp_version=$(copilot --version 2>/dev/null | head -n1)
        clack_success "$(i18n copilot_cli): $_cp_version"
        install_markitai_extra "copilot"
        track_install "copilot_cli" "installed"
        return 0
    fi

    if ! clack_confirm_optional "$(i18n confirm_copilot_cli)" "n"; then
        clack_skip "$(i18n copilot_cli)"
        track_install "copilot_cli" "skipped"
        return 2
    fi

    clack_info "$(i18n installing) $(i18n copilot_cli)..."

    # Try official install script in a child shell so its exit is contained.
    if run_remote_shell_installer "$(i18n installing) $(i18n copilot_cli)" "https://gh.io/copilot-install"; then
        if command -v copilot >/dev/null 2>&1; then
            clack_success "$(i18n copilot_cli) $(i18n installed)"
            # Also install the SDK extra
            install_markitai_extra "copilot" || true
            track_install "copilot_cli" "installed"
            return 0
        fi
    fi

    # Fallback: npm/pnpm
    if command -v pnpm >/dev/null 2>&1; then
        if clack_run_quiet "$(i18n installing) $(i18n copilot_cli)" pnpm add -g @github/copilot; then
            clack_success "$(i18n copilot_cli) $(i18n installed)"
            install_markitai_extra "copilot" || true
            track_install "copilot_cli" "installed"
            return 0
        fi
    elif command -v npm >/dev/null 2>&1; then
        if clack_run_quiet "$(i18n installing) $(i18n copilot_cli)" npm install -g @github/copilot; then
            clack_success "$(i18n copilot_cli) $(i18n installed)"
            install_markitai_extra "copilot" || true
            track_install "copilot_cli" "installed"
            return 0
        fi
    fi

    clack_error "$(i18n copilot_cli) $(i18n failed)"
    track_install "copilot_cli" "failed"
    return 1
}

# Print installation summary
# Usage: print_summary
print_summary() {
    clack_section "$(i18n section_summary)"

    # Print installed components using heredoc
    if [ -n "$INSTALLED_COMPONENTS" ]; then
        _ifs_old="$IFS"
        IFS='|'
        # shellcheck disable=SC2086
        set -- $INSTALLED_COMPONENTS
        IFS="$_ifs_old"
        if [ $# -gt 0 ]; then
            clack_guide
            printf "${GRAY}%s${NC}  ${BOLD}%s${NC}\n" "$S_BAR" "$(i18n summary_installed)"
            for _comp in "$@"; do
                [ -n "$_comp" ] && printf "${GRAY}%s${NC}    ${GREEN}%s${NC} %s\n" "$S_BAR" "$S_CHECK" "$(i18n "$_comp")"
            done
        fi
    fi

    # Print skipped components
    if [ -n "$SKIPPED_COMPONENTS" ]; then
        _ifs_old="$IFS"
        IFS='|'
        # shellcheck disable=SC2086
        set -- $SKIPPED_COMPONENTS
        IFS="$_ifs_old"
        if [ $# -gt 0 ]; then
            clack_guide
            printf "${GRAY}%s${NC}  ${BOLD}%s${NC}\n" "$S_BAR" "$(i18n summary_skipped)"
            for _comp in "$@"; do
                [ -n "$_comp" ] && printf "${GRAY}%s${NC}    ${YELLOW}%s${NC} %s\n" "$S_BAR" "$S_CIRCLE" "$(i18n "$_comp")"
            done
        fi
    fi

    # Print failed components
    if [ -n "$FAILED_COMPONENTS" ]; then
        _ifs_old="$IFS"
        IFS='|'
        # shellcheck disable=SC2086
        set -- $FAILED_COMPONENTS
        IFS="$_ifs_old"
        if [ $# -gt 0 ]; then
            clack_guide
            printf "${GRAY}%s${NC}  ${BOLD}%s${NC}\n" "$S_BAR" "$(i18n summary_failed)"
            for _comp in "$@"; do
                [ -n "$_comp" ] && printf "${GRAY}%s${NC}    ${RED}%s${NC} %s\n" "$S_BAR" "$S_CROSS" "$(i18n "$_comp")"
            done
        fi
    fi

    # Empty line before docs link
    clack_log ""
    clack_info "$(i18n info_docs): https://markitai.dev"
    clack_info "$(i18n info_issues): https://github.com/Ynewtime/markitai/issues"
}

# Print user mode completion message
# Usage: print_user_completion
# Determine whether the `mkai` short alias is usable (resolves to markitai)
# or shadowed by a pre-existing command of the same name. Sets MKAI_USABLE
# and warns on a shadowing conflict. `markitai` (the full name) always works.
check_mkai_alias() {
    MKAI_USABLE=false
    command -v mkai >/dev/null 2>&1 || return 0   # not on PATH yet — don't advertise
    if mkai --version 2>/dev/null | grep -qi "markitai"; then
        MKAI_USABLE=true
    else
        # A different `mkai` (user's own tool/alias) shadows markitai's.
        clack_warn "$(i18n mkai_conflict): $(command -v mkai 2>/dev/null)"
    fi
}

print_user_completion() {
    check_mkai_alias

    set -- \
        "$(i18n interactive_mode):" \
        "  ${CYAN}markitai -I${NC}"

    # Only advertise the serve command when its dependencies were selected.
    if markitai_extra_enabled "serve"; then
        set -- "$@" \
            "" \
            "$(i18n start_web_ui):" \
            "  ${CYAN}markitai serve${NC}"
    fi

    set -- "$@" \
        "" \
        "$(i18n configure_llm):" \
        "  ${CYAN}markitai init${NC}" \
        "" \
        "$(i18n convert_file):" \
        "  ${CYAN}markitai file.pdf${NC}" \
        "" \
        "$(i18n show_help):" \
        "  ${CYAN}markitai --help${NC}"

    # Only advertise the short alias when it actually resolves to markitai.
    if [ "${MKAI_USABLE:-false}" = "true" ]; then
        set -- "$@" "" "$(i18n short_alias)"
    fi

    clack_note "$(i18n getting_started)" "$@"
}

# Print dev mode completion message
# Usage: print_dev_completion
print_dev_completion() {
    _project_root=$(get_project_root)
    clack_note "$(i18n quick_start)" \
        "$(i18n configure_env):" \
        "  ${CYAN}cp .env.example .env${NC}" \
        "" \
        "$(i18n interactive_mode):" \
        "  ${CYAN}uv run markitai -I${NC}" \
        "" \
        "$(i18n start_web_ui):" \
        "  ${CYAN}uv run markitai serve${NC}" \
        "" \
        "$(i18n run_tests):" \
        "  ${CYAN}uv run pytest${NC}" \
        "" \
        "$(i18n run_cli):" \
        "  ${CYAN}uv run markitai --help${NC}"
}

# Initialize markitai config (silent, first install only)
# Skips when a config already exists: `init --yes` would silently append
# newly detected providers to the user's model_list. The detector only records
# a local CLI after login succeeds and its finalized runtime SDK is importable.
# Returns: 0 always
init_config() {
    if [ -f "$HOME/.markitai/config.json" ]; then
        return 0
    fi
    if command -v markitai >/dev/null 2>&1; then
        markitai init --yes >/dev/null 2>&1 || true
    fi
    return 0
}

# ============================================================
# Main Entry Point
# ============================================================

# User mode main flow
run_user_setup() {
    clack_intro "$(i18n welcome)"
    clack_info "$(i18n mode_user)"
    warn_if_root
    configure_mirrors
    choose_install_source

    clack_section "$(i18n section_core)"
    install_uv || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }
    detect_python || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }
    load_existing_markitai_extras
    select_markitai_serve
    install_markitai || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }
    track_markitai_serve

    clack_section "$(i18n section_optional)"
    install_optional_playwright || true
    install_optional_libreoffice || true
    install_optional_ffmpeg || true

    clack_section "$(i18n section_llm_cli)"
    install_optional_claude_cli || true
    install_optional_copilot_cli || true

    finalize_markitai_extras || true

    init_config >/dev/null 2>&1

    print_summary
    print_user_completion
    if [ -n "$FAILED_COMPONENTS" ]; then
        clack_outro_warn "$(i18n setup_complete_with_warnings)"
    else
        clack_outro "$(i18n setup_complete)"
    fi
}

# Dev mode main flow
run_dev_setup() {
    clack_intro "$(i18n welcome)"
    clack_info "$(i18n mode_dev)"
    warn_if_root
    configure_mirrors

    clack_section "$(i18n section_prerequisites)"
    install_uv || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }
    detect_python || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }

    clack_section "$(i18n section_dev_env)"
    sync_dependencies || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }
    install_precommit || true

    clack_section "$(i18n section_optional)"
    install_optional_playwright || true
    install_optional_libreoffice || true
    install_optional_ffmpeg || true

    clack_section "$(i18n section_llm_cli)"
    install_optional_claude_cli || true
    install_optional_copilot_cli || true

    print_summary
    print_dev_completion
    if [ -n "$FAILED_COMPONENTS" ]; then
        clack_outro_warn "$(i18n dev_setup_complete_with_warnings)"
    else
        clack_outro "$(i18n dev_setup_complete)"
    fi
}

# Main entry point
main() {
    if is_dev_mode; then
        run_dev_setup
    else
        run_user_setup
    fi
}

# Always close an active guide on an unexpected exit or signal.
trap 'setup_on_exit' 0
trap 'setup_on_signal 129' HUP
trap 'setup_on_signal 130' INT
trap 'setup_on_signal 143' TERM

# ${1+"$@"} instead of "$@": with `set -u` and no arguments, old shells
# (e.g. macOS bash 3.2) would abort on "$@"
main ${1+"$@"}
