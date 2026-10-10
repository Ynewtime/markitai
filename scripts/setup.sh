#!/bin/sh
# Markitai Setup Script - Unified installer with i18n support
# Supports bash/zsh/dash and other POSIX-compatible shells
# Auto-detects the language (en/zh). Run from the root of a markitai checkout,
# it asks whether to install markitai or set up a development environment.
#
# Usage:
#   curl -fsSL https://markitai.dev/setup.sh | sh    # Install markitai
#   ./scripts/setup.sh                               # In a checkout: install or dev setup
#
# Environment:
#   MARKITAI_SETUP_MODE=install|dev  choose without the question (default: install)
#   MARKITAI_SETUP_SOURCE=local|pypi in a checkout, install from it (default) or
#                                    from the PyPI release, without the question
#   MARKITAI_INSTALL_OPTIONAL=1      install optional components without a terminal
#   MARKITAI_VERSION=X.Y.Z           pin the markitai release
#   MARKITAI_USE_MIRROR=1|0          use (or never offer) a PyPI mirror
#   MARKITAI_SETUP_LOG=PATH          full failure log (default: a private temp file)
#   NO_COLOR=1                       no colours; the marks stay

# -e: exit on error; -u: error on unset variables (pipefail is not POSIX sh)
set -eu

# ============================================================
# Internationalization (i18n) System
# ============================================================

# Detect language from the locale that governs messages
# (LC_ALL, then LC_MESSAGES, then LANG). Chinese needs a UTF-8 locale: on a
# GBK or C terminal the UTF-8 text would print as mojibake.
# Returns: "zh" for Chinese, "en" for English (default)
detect_lang() {
    _lang_env="${LC_ALL:-${LC_MESSAGES:-${LANG:-}}}"
    case "$_lang_env" in
        zh*.[Uu][Tt][Ff]-8*|zh*.[Uu][Tt][Ff]8*)
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
# Returns localized string for the given key. setup.ps1 carries the same
# Chinese strings as \uXXXX escapes (it must stay ASCII for Windows
# PowerShell 5.1); a test checks that the shared keys match.
i18n() {
    _key="$1"

    if [ "$LANG_CODE" = "zh" ]; then
        case "$_key" in
            # Intro/Outro
            welcome)                    echo "Markitai 安装" ;;
            setup_complete)             echo "markitai 已就绪" ;;
            setup_complete_with_warnings) echo "markitai 已就绪，但有部分组件失败" ;;
            dev_setup_complete)         echo "开发环境已就绪" ;;
            dev_setup_complete_with_warnings) echo "开发环境已就绪，但有部分组件失败" ;;
            setup_cancelled)            echo "已取消安装" ;;

            # Mode
            mode_user)                  echo "模式：安装 markitai" ;;
            mode_dev)                   echo "模式：开发环境" ;;
            repo_root_detected)         echo "当前目录是 markitai 源码仓库" ;;
            confirm_dev_mode)           echo "在这里设置开发环境，而不是安装 markitai?" ;;
            info_mode_default)          echo "将安装 markitai；设置 MARKITAI_SETUP_MODE=dev 可改为设置开发环境" ;;
            warn_unknown_mode)          echo "忽略 MARKITAI_SETUP_MODE (应为 install 或 dev)" ;;
            error_dev_needs_repo)       echo "开发环境需要 markitai 源码仓库；请在仓库根目录运行安装脚本" ;;

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
            already_installed)          echo "已经安装" ;;
            not_found)                  echo "未找到" ;;

            # Components
            uv)                         echo "uv 包管理器" ;;
            python)                     echo "Python" ;;
            markitai)                   echo "markitai" ;;
            serve)                      echo "Web UI (markitai serve)" ;;
            ocr)                        echo "OCR (扫描件文字识别)" ;;
            playwright)                 echo "Playwright 浏览器" ;;
            claude_cli)                 echo "Claude Code CLI" ;;
            copilot_cli)                echo "Copilot CLI" ;;
            precommit)                  echo "pre-commit hooks" ;;
            python_deps)                echo "Python 依赖" ;;

            # Confirmations
            confirm_serve)              echo "安装 Web UI 依赖? (启用 markitai serve)" ;;
            confirm_ocr)                echo "安装 OCR 支持? (识别扫描件和图片中的文字, 约 150MB)" ;;
            confirm_playwright)         echo "安装 Playwright 浏览器? (用于 JS 渲染页面)" ;;
            confirm_claude_cli)         echo "安装 Claude Code CLI? (使用 Claude 订阅)" ;;
            confirm_copilot_cli)        echo "安装 Copilot CLI? (使用 GitHub Copilot 订阅)" ;;
            confirm_uv)                 echo "安装 uv 包管理器?" ;;
            confirm_continue_as_root)   echo "仍以 root 身份继续?" ;;

            # Info messages
            info_chatgpt_route)         echo "ChatGPT 订阅无需 CLI，也无需 extra（首次使用时登录）" ;;
            optional_headless_hint)     echo "没有终端，可选组件已跳过；要一并安装，设置此变量后重新运行安装脚本" ;;
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
            confirm_local_install)      echo "从这个源码仓库安装 markitai? (n: 改用 PyPI 发布版)" ;;
            source_local)               echo "安装来源: 本地源码" ;;
            source_pypi)                echo "安装来源: PyPI" ;;
            info_repo_noninteractive)   echo "无终端可询问: 从这个源码仓库安装; 设置 MARKITAI_SETUP_SOURCE=pypi 可改用 PyPI 发布版" ;;
            info_source_pinned)         echo "已设置 MARKITAI_VERSION: 安装该 PyPI 发布版" ;;
            warn_unknown_source)        echo "忽略 MARKITAI_SETUP_SOURCE (应为 local 或 pypi)" ;;
            error_local_needs_repo)     echo "MARKITAI_SETUP_SOURCE=local 需要在 markitai 源码仓库中运行本脚本" ;;

            # Network / Mirrors
            mirror_slow_index)          echo "PyPI 默认源响应缓慢或不可达" ;;
            mirror_confirm)             echo "使用镜像源加速安装?" ;;
            mirror_select)              echo "选择镜像源" ;;
            mirror_tuna)                echo "清华 TUNA (推荐)" ;;
            mirror_aliyun)              echo "阿里云" ;;
            mirror_tencent)             echo "腾讯云" ;;
            mirror_huawei)              echo "华为云" ;;
            mirror_enabled)             echo "已启用镜像加速" ;;
            mirror_skipped)             echo "继续使用默认源" ;;
            mirror_pypi)                echo "PyPI 镜像" ;;
            mirror_playwright)          echo "Playwright 镜像" ;;
            mirror_npm)                 echo "npm 镜像" ;;

            # Warnings
            warn_root)                  echo "正在以 root 身份运行" ;;
            warn_root_risk)             echo "markitai 按用户安装，从不需要 root 权限。以 root 运行时，它可能装进 root 的主目录，或留下你自己的账户无法更新的文件；下载的安装程序也都拥有这个系统的完全权限。" ;;
            info_root_noninteractive)   echo "没有可供确认的终端，继续以 root 身份运行" ;;
            warn_bin_off_path)          echo "当前 shell 的 PATH 不包含 markitai 的命令目录" ;;
            info_bin_off_path_fix)      echo "请打开新的终端；如果仍找不到 markitai，请运行:" ;;

            # Getting started
            getting_started)            echo "开始使用" ;;
            quick_start)                echo "快速开始" ;;
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

            # Default fallback
            *)                          echo "$_key" ;;
        esac
    else
        # English (default)
        case "$_key" in
            # Intro/Outro
            welcome)                    echo "Markitai Setup" ;;
            setup_complete)             echo "Markitai is ready" ;;
            setup_complete_with_warnings) echo "Markitai is ready, but some components failed" ;;
            dev_setup_complete)         echo "Development environment ready" ;;
            dev_setup_complete_with_warnings) echo "Development environment ready, but some components failed" ;;
            setup_cancelled)            echo "Setup cancelled" ;;

            # Mode
            mode_user)                  echo "Mode: install markitai" ;;
            mode_dev)                   echo "Mode: development environment" ;;
            repo_root_detected)         echo "This folder is a markitai source checkout" ;;
            confirm_dev_mode)           echo "Set up a development environment here instead of installing markitai?" ;;
            info_mode_default)          echo "Installing markitai; set MARKITAI_SETUP_MODE=dev for a development environment instead" ;;
            warn_unknown_mode)          echo "Ignoring MARKITAI_SETUP_MODE (expected install or dev)" ;;
            error_dev_needs_repo)       echo "A development environment needs a markitai checkout; run setup from its root folder" ;;

            # Sections
            section_prerequisites)      echo "Prerequisites" ;;
            section_core)               echo "Core components" ;;
            section_optional)           echo "Optional components" ;;
            section_dev_env)            echo "Development environment" ;;
            section_llm_cli)            echo "LLM CLI tools" ;;
            section_summary)            echo "Summary" ;;

            # Status
            installed)                  echo "installed" ;;
            installing)                 echo "Installing" ;;
            skipped)                    echo "skipped" ;;
            failed)                     echo "failed" ;;
            already_installed)          echo "already installed" ;;
            not_found)                  echo "not found" ;;

            # Components
            uv)                         echo "uv package manager" ;;
            python)                     echo "Python" ;;
            markitai)                   echo "markitai" ;;
            serve)                      echo "Web UI (markitai serve)" ;;
            ocr)                        echo "OCR (scanned-document text recognition)" ;;
            playwright)                 echo "Playwright browser" ;;
            claude_cli)                 echo "Claude Code CLI" ;;
            copilot_cli)                echo "Copilot CLI" ;;
            precommit)                  echo "pre-commit hooks" ;;
            python_deps)                echo "Python dependencies" ;;

            # Confirmations
            confirm_serve)              echo "Install Web UI dependencies? (enables markitai serve)" ;;
            confirm_ocr)                echo "Install OCR support? (text recognition in scanned PDFs and images, ~150MB)" ;;
            confirm_playwright)         echo "Install Playwright browser? (for JS-rendered pages)" ;;
            confirm_claude_cli)         echo "Install Claude Code CLI? (use your Claude subscription)" ;;
            confirm_copilot_cli)        echo "Install Copilot CLI? (use your GitHub Copilot subscription)" ;;
            confirm_uv)                 echo "Install uv package manager?" ;;
            confirm_continue_as_root)   echo "Continue as root anyway?" ;;

            # Info messages
            info_chatgpt_route)         echo "ChatGPT subscription needs no CLI and no extra (it signs in on first use)" ;;
            optional_headless_hint)     echo "No terminal, so optional components were skipped; to install them too, set this and run setup again" ;;
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
            confirm_local_install)      echo "Install markitai from this checkout? (n: the PyPI release instead)" ;;
            source_local)               echo "Install source: local repo" ;;
            source_pypi)                echo "Install source: PyPI" ;;
            info_repo_noninteractive)   echo "No terminal to ask: installing from this checkout; set MARKITAI_SETUP_SOURCE=pypi for the PyPI release" ;;
            info_source_pinned)         echo "MARKITAI_VERSION is set: installing that PyPI release" ;;
            warn_unknown_source)        echo "Ignoring MARKITAI_SETUP_SOURCE (expected local or pypi)" ;;
            error_local_needs_repo)     echo "MARKITAI_SETUP_SOURCE=local needs this script to run from a markitai checkout" ;;

            # Network / Mirrors
            mirror_slow_index)          echo "The default PyPI index is slow or unreachable from here" ;;
            mirror_confirm)             echo "Use a mirror to speed up the install?" ;;
            mirror_select)              echo "Select mirror source" ;;
            mirror_tuna)                echo "Tsinghua TUNA (Recommended)" ;;
            mirror_aliyun)              echo "Alibaba Cloud" ;;
            mirror_tencent)             echo "Tencent Cloud" ;;
            mirror_huawei)              echo "Huawei Cloud" ;;
            mirror_enabled)             echo "Mirror acceleration enabled" ;;
            mirror_skipped)             echo "Keeping the default index" ;;
            mirror_pypi)                echo "PyPI mirror" ;;
            mirror_playwright)          echo "Playwright mirror" ;;
            mirror_npm)                 echo "npm mirror" ;;

            # Warnings
            warn_root)                  echo "Running as root" ;;
            warn_root_risk)             echo "markitai installs per user and never needs root. As root, it can end up in root's home or leave files your own account cannot update, and every downloaded installer runs with full access to this system." ;;
            info_root_noninteractive)   echo "No terminal to ask; continuing as root" ;;
            warn_bin_off_path)          echo "This shell's PATH does not include markitai's command folder" ;;
            info_bin_off_path_fix)      echo "Open a new terminal; if markitai is still not found, run:" ;;

            # Getting started
            getting_started)            echo "Getting started" ;;
            quick_start)                echo "Quick start" ;;
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

            # Default fallback
            *)                          echo "$_key" ;;
        esac
    fi
}

# ============================================================
# Color Definitions
# ============================================================
# One terminal style for everything markitai prints, the CLI included:
# structure is bold, the guide gray, cyan marks prompts, actions and commands,
# green is only success, yellow only warnings, red only errors. Only the
# normal eight colours plus gray 90 are used, because terminal themes tune
# those for light and dark backgrounds alike (bright yellow vanishes on white).
# Colour is off for NO_COLOR, TERM=dumb and output that is not a terminal
# (unless FORCE_COLOR is set); the marks carry the meaning either way.

setup_use_color() {
    [ -n "${NO_COLOR:-}" ] && return 1
    [ "${TERM:-}" = "dumb" ] && return 1
    case "${FORCE_COLOR:-}" in
        ''|0|false) ;;
        *) return 0 ;;
    esac
    [ -t 1 ]
}

if setup_use_color; then
    RED='\033[31m'
    GREEN='\033[32m'
    YELLOW='\033[33m'
    CYAN='\033[36m'
    GRAY='\033[90m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' CYAN='' GRAY='' BOLD='' NC=''
fi

# ============================================================
# Compact tree-style visual components
# ============================================================
# The one-column guide follows the `withGuide` visual grammar popularized by
# @clack/prompts. Shell installers cannot rely on Node being present (or redraw
# completed prompts like Clack), so the tiny renderer stays inline and uses
# tree branches to keep the guide visibly connected in static terminal output.
#
# Layout: a guide-only row before each section header and none inside a
# section; items are "│  <mark> <text>", details and commands sit two columns
# further in. Marks: ✓ success, ! warning, ✗ error, → action, • info,
# ○ skipped, ? prompt.

# Unicode marks need a UTF-8 locale; LANG=C or TERM=dumb get the ASCII set so
# a terminal never shows replacement boxes.
setup_use_unicode() {
    [ "${TERM:-}" = "dumb" ] && return 1
    case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in
        *[Uu][Tt][Ff]-8*|*[Uu][Tt][Ff]8*) return 0 ;;
    esac
    return 1
}

if setup_use_unicode; then
    S_BAR="│"
    S_TOP="╭─"
    S_BRANCH="├─"
    S_BOTTOM="╰─"
    S_CHECK="✓"
    S_CROSS="✗"
    S_ARROW="→"
    S_DOT="•"
    S_CIRCLE="○"
    S_SPIN_FRAMES="◒ ◐ ◓ ◑"
else
    S_BAR="|"
    S_TOP="+-"
    S_BRANCH="+-"
    S_BOTTOM="+-"
    S_CHECK="+"
    S_CROSS="x"
    S_ARROW=">"
    S_DOT="-"
    S_CIRCLE="o"
    # shellcheck disable=SC1003 # a literal backslash frame, not an escape
    S_SPIN_FRAMES='| / - \'
fi
S_WARN="!"
S_ASK="?"
NEWLINE='
'

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

# Terminal width used to wrap long rows. Output that is not a terminal is
# never wrapped: piped and logged output keeps one line per message.
setup_terminal_columns() {
    case "${COLUMNS:-}" in
        ''|*[!0-9]*) ;;
        *) echo "$COLUMNS"; return 0 ;;
    esac
    _stc=$( (stty size </dev/tty) 2>/dev/null | awk '{print $2}') || _stc=""
    case "$_stc" in
        ''|0|*[!0-9]*) _stc=$(tput cols 2>/dev/null) || _stc="" ;;
    esac
    case "$_stc" in
        ''|*[!0-9]*) echo 0 ;;
        *) echo "$_stc" ;;
    esac
}

SETUP_WRAP_WIDTH=0
if [ -t 1 ] && [ "${TERM:-}" != "dumb" ]; then
    SETUP_WRAP_WIDTH=$(setup_terminal_columns)
fi

# Word-wrap stdin to $1 display columns. CJK characters count two columns and
# a run of them longer than a line may break between characters; other long
# words (paths, URLs) are never split. Bytes are decoded by hand so the result
# does not depend on the awk implementation's locale support.
clack_wrap() {
    LC_ALL=C awk -v width="$1" '
        BEGIN { for (i = 128; i < 256; i++) ord[sprintf("%c", i)] = i }
        function byteat(s, i,    c) { c = substr(s, i, 1); return (c in ord) ? ord[c] : 0 }
        function clen(b) { return b < 192 ? 1 : (b < 224 ? 2 : (b < 240 ? 3 : 4)) }
        function cwidth(s, i, b,    b2) {
            if (b >= 227 && b <= 237) return 2
            if (b == 239) { b2 = byteat(s, i + 1); return (b2 == 188 || b2 == 189) ? 2 : 1 }
            if (b >= 240) return 2
            return 1
        }
        function swidth(s,    i, n, b, w) {
            w = 0; n = length(s); i = 1
            while (i <= n) { b = byteat(s, i); w += cwidth(s, i, b); i += clen(b) }
            return w
        }
        {
            n = split($0, words, " "); line = ""; lw = 0
            for (k = 1; k <= n; k++) {
                word = words[k]; ww = swidth(word)
                if (line != "" && lw + 1 + ww <= width) { line = line " " word; lw += 1 + ww; continue }
                if (line != "") { print line; line = ""; lw = 0 }
                if (ww <= width || ww == length(word)) { line = word; lw = ww; continue }
                m = length(word); i = 1
                while (i <= m) {
                    b = byteat(word, i); l = clen(b); w = cwidth(word, i, b)
                    if (line != "" && lw + w > width) { print line; line = ""; lw = 0 }
                    line = line substr(word, i, l); lw += w; i += l
                }
            }
            print line
        }'
}

# Print text on the guide. On a terminal, long text wraps at word boundaries
# with a hanging indent under its first character.
# Usage: clack_emit "<first lead>" "<continuation lead>" "<text>" \
#                   "<text colour>" "<suffix>" "<line end>" [reserved columns]
# The first lead and the suffix may hold colour codes; the continuation lead
# is plain spaces as wide as the first lead.
clack_emit() {
    _ce_head="$1"
    _ce_pad="$2"
    _ce_text="$3"
    _ce_tc="$4"
    _ce_suffix="$5"
    _ce_end="$6"
    _ce_width=0
    if [ "$SETUP_WRAP_WIDTH" -gt 0 ]; then
        _ce_width=$((SETUP_WRAP_WIDTH - 3 - ${#_ce_pad} - ${7:-0}))
        [ "$_ce_width" -ge 16 ] || _ce_width=0
    fi
    if [ "$_ce_width" -gt 0 ]; then
        _ce_rest=$(printf '%s\n' "$_ce_text" | clack_wrap "$_ce_width")
    else
        _ce_rest="$_ce_text"
    fi
    _ce_lead="$_ce_head"
    while :; do
        case "$_ce_rest" in
            *"$NEWLINE"*)
                _ce_line="${_ce_rest%%"$NEWLINE"*}"
                _ce_rest="${_ce_rest#*"$NEWLINE"}"
                printf "${GRAY}%s${NC}  %b${_ce_tc}%s${NC}\n" "$S_BAR" "$_ce_lead" "$_ce_line"
                _ce_lead="$_ce_pad"
                ;;
            *)
                printf "${GRAY}%s${NC}  %b${_ce_tc}%s${NC}%b${_ce_end}" \
                    "$S_BAR" "$_ce_lead" "$_ce_rest" "$_ce_suffix"
                return 0
                ;;
        esac
    done
}

# One item row: "│  <mark> <text>".
# Usage: clack_item "<mark colour>" "<mark>" "<text>" ["<text colour>"]
clack_item() {
    clack_emit "${1}${2}${NC} " "  " "$3" "${4:-}" "" "\n"
}

# Print an empty row while preserving the session guide.
clack_guide() {
    printf "${GRAY}%s${NC}\n" "$S_BAR"
}

# Session intro - start of CLI flow. Items may follow directly; each section
# adds its own guide row above its header.
# Usage: clack_intro "Title"
clack_intro() {
    CLACK_SESSION_ACTIVE=true
    CLACK_SESSION_CLOSED=false
    printf "\n${GRAY}%s${NC} ${BOLD}%s${NC}\n" "$S_TOP" "$1"
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
    printf "${GRAY}%s${NC} ${GREEN}%s${NC}\n\n" "$S_BOTTOM" "$1"
}

# Non-fatal component failures close the tree in warning yellow, not success
# green, while still returning a successful core installation.
clack_outro_warn() {
    clack_show_error_log
    CLACK_SESSION_CLOSED=true
    clack_guide
    printf "${GRAY}%s${NC} ${YELLOW}%s${NC}\n\n" "$S_BOTTOM" "$1"
}

# Section header connected to the session guide. Structure is never coloured:
# a coloured header reads as a status.
# Usage: clack_section "Section title"
clack_section() {
    clack_guide
    printf "${GRAY}%s${NC} ${BOLD}%s${NC}\n" "$S_BRANCH" "$1"
}

# Item rows. Usage: clack_success "Message" (and the same for the others)
clack_success() {
    clack_item "$GREEN" "$S_CHECK" "$1"
}

clack_error() {
    clack_item "$RED" "$S_CROSS" "$1"
    clack_flush_detail
}

clack_warn() {
    clack_item "$YELLOW" "$S_WARN" "$1"
    clack_flush_detail
}

# A neutral fact (paths, choices made, what a component is for).
clack_info() {
    clack_item "$GRAY" "$S_DOT" "$1"
}

# An action in progress (non-terminal output; a terminal gets the spinner).
clack_step() {
    clack_item "$CYAN" "$S_ARROW" "$1"
}

clack_skip() {
    clack_item "$GRAY" "$S_CIRCLE" "$1" "$GRAY"
}

# Gray detail under the item above it, two columns further in.
clack_detail_row() {
    clack_emit "  " "  " "$1" "$GRAY" "" "\n"
}

# A command to copy: accent colour, its own row, never wrapped, so a terminal
# narrower than the command still copies it back as one line.
clack_cmd() {
    printf "${GRAY}%s${NC}    ${CYAN}%s${NC}\n" "$S_BAR" "$1"
}

# A label and the command it describes.
# Usage: clack_hint "Label" "command"
clack_hint() {
    clack_info "$1:"
    clack_cmd "$2"
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
            clack_detail_row "$_cpd_line"
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
                for _cs_frame in $S_SPIN_FRAMES; do
                    printf "\r${GRAY}%s${NC}  ${CYAN}%s${NC} %s" "$S_BAR" "$_cs_frame" "$_cs_message"
                    sleep 0.1 2>/dev/null || sleep 1
                done
            done
        ) &
        _cs_pid=$!
        CLACK_SPINNER_PID="$_cs_pid"
    else
        clack_step "$_cs_message"
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

# Read one answer from the terminal into CLACK_ANSWER, falling back to $1.
# Reads /dev/tty for piped execution (curl | sh). A terminal echoes its own
# newline, so printing another one creates stray blank rows; only redirected
# or headless output needs the answer written out.
clack_read_answer() {
    CLACK_ANSWER=""
    if [ -t 0 ]; then
        if read -r CLACK_ANSWER; then
            [ -t 1 ] || printf "%s\n" "${CLACK_ANSWER:-$1}"
        else
            CLACK_ANSWER="$1"
            printf "%s\n" "$CLACK_ANSWER"
        fi
    elif { read -r CLACK_ANSWER < /dev/tty; } 2>/dev/null; then
        [ -t 1 ] || printf "%s\n" "${CLACK_ANSWER:-$1}"
    else
        CLACK_ANSWER="$1"
        printf "%s\n" "$CLACK_ANSWER"
    fi
    [ -n "$CLACK_ANSWER" ] || CLACK_ANSWER="$1"
}

# Yes/no question on the guide: "│  ? Question [y/N] ".
# Usage: clack_confirm "Question?" "y|n"
# Returns: 0 for yes, 1 for no
clack_confirm() {
    _cc_prompt="$1"
    _cc_default="$2"

    if [ "$_cc_default" = "y" ]; then
        _cc_hint="[Y/n]"
    else
        _cc_hint="[y/N]"
    fi

    clack_emit "${CYAN}${S_ASK}${NC} " "  " "$_cc_prompt" "" " ${GRAY}${_cc_hint}${NC} " "" 7
    clack_read_answer "$_cc_default"

    case "$CLACK_ANSWER" in
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

# Cancel message
# Usage: clack_cancel "Message"
clack_cancel() {
    clack_show_error_log
    CLACK_SESSION_CLOSED=true
    clack_guide
    printf "${GRAY}%s${NC} ${RED}%s${NC}\n\n" "$S_BOTTOM" "$1"
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

# Check if running as root and warn, saying why it matters. A terminal user
# is asked (default no); without a terminal (a container build) there is
# nobody to ask, so setup continues after the warning.
# Returns: 0 to continue, exits if user declines
warn_if_root() {
    if [ "$(id -u)" -eq 0 ]; then
        clack_warn "$(i18n warn_root)"
        clack_detail_row "$(i18n warn_root_risk)"

        if ! has_interactive_tty; then
            clack_info "$(i18n info_root_noninteractive)"
            return 0
        fi
        if ! clack_confirm "$(i18n confirm_continue_as_root)" "n"; then
            clack_cancel "$(i18n setup_cancelled)"
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

# Time-boxed reachability probe of the default package index.
#
# This must stay an APPLICATION-layer check. A bare TCP/port probe (nc -z,
# /dev/tcp, connect_ex) is meaningless on a machine running a TUN-mode proxy:
# every port answers, so every probe "succeeds". And an HTTP status alone is
# not enough either, because an intermittently empty body still returns 200.
# So: real GET, verify status *and* that bytes actually arrived.
#
# The timeout doubles as the "too slow to install through" threshold — a few
# KB of index metadata that cannot arrive within it predicts a miserable
# multi-hundred-MB download.
#
# Env: MARKITAI_INDEX_PROBE_URL / MARKITAI_INDEX_PROBE_TIMEOUT (test seams).
# Returns: 0 when the default index answered in time, 1 otherwise.
probe_default_index() {
    _probe_url="${MARKITAI_INDEX_PROBE_URL:-https://pypi.org/simple/markitai/}"
    _probe_timeout="${MARKITAI_INDEX_PROBE_TIMEOUT:-3}"

    # No HTTP client to measure with: stay silent rather than show an
    # unrelated mirror question. install_uv reports the missing curl itself.
    command -v curl >/dev/null 2>&1 || return 0

    _probe_out=$(curl -fsS -o /dev/null \
        --connect-timeout "$_probe_timeout" --max-time "$_probe_timeout" \
        -w '%{http_code} %{size_download}' "$_probe_url" 2>/dev/null) || return 1

    _probe_code="${_probe_out%% *}"
    _probe_bytes="${_probe_out##* }"
    [ "$_probe_code" = "200" ] || return 1
    case "$_probe_bytes" in
        ''|0|*[!0-9]*) return 1 ;;
    esac
    return 0
}

# Offer China mirrors only when this machine has an actual problem reaching
# the default index.
#
# The previous rule was "no proxy variable set" — which is true for most of
# the planet, so the very first thing a user in Berlin or São Paulo saw was a
# warning about Chinese mirrors that had nothing to do with them. Locale and
# timezone are no better: they misfire on travellers, expats and CI runners.
# Measured reachability is the only evidence that actually predicts the
# problem mirrors solve.
#
# Sets: UV_INDEX_URL, PLAYWRIGHT_DOWNLOAD_HOST, NPM_CONFIG_REGISTRY
configure_mirrors() {
    # Explicit override in either direction wins: no probe, no question.
    case "${MARKITAI_USE_MIRROR:-}" in
        1|true|TRUE|yes|YES|on|ON)
            select_mirror_source
            return 0
            ;;
        0|false|FALSE|no|NO|off|OFF)
            return 0
            ;;
    esac

    # A configured proxy is the user's own routing decision; do not second-guess it.
    if detect_proxy; then
        return 0
    fi

    # Never ask without a terminal, and never spend the probe budget there.
    if ! has_interactive_tty; then
        return 0
    fi

    # Fast, healthy index: say nothing at all and use the default.
    if probe_default_index; then
        return 0
    fi

    clack_warn "$(i18n mirror_slow_index)"

    if ! clack_confirm "$(i18n mirror_confirm)" "y"; then
        clack_info "$(i18n mirror_skipped)"
        return 0
    fi

    select_mirror_source
}

# Pick a mirror and export it. Reached from an accepted prompt or from an
# explicit MARKITAI_USE_MIRROR=1.
select_mirror_source() {
    # The choices first, then the question, so the cursor waits after it.
    clack_detail_row "1. $(i18n mirror_tuna)"
    clack_detail_row "2. $(i18n mirror_aliyun)"
    clack_detail_row "3. $(i18n mirror_tencent)"
    clack_detail_row "4. $(i18n mirror_huawei)"
    clack_emit "${CYAN}${S_ASK}${NC} " "  " "$(i18n mirror_select)" "" " ${GRAY}[1]${NC} " "" 5

    # Fall back to the default choice when no TTY is available (CI, cron).
    clack_read_answer "1"
    _mirror_choice="$CLACK_ANSWER"

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
    clack_detail_row "$(i18n mirror_pypi): $UV_INDEX_URL"
    clack_detail_row "$(i18n mirror_npm): $NPM_CONFIG_REGISTRY"
    clack_detail_row "$(i18n mirror_playwright): $PLAYWRIGHT_DOWNLOAD_HOST"
}

# ============================================================
# Mode Detection
# ============================================================

# Check whether the current directory is the root of a markitai checkout,
# where a development environment can be set up.
# Returns: 0 in a checkout root, 1 elsewhere
is_dev_mode() {
    # Note: .git can be a directory (normal repo) or file (worktree)
    if [ -f "./pyproject.toml" ] && [ -e "./.git" ] && [ -d "./scripts" ]; then
        if grep -q "markitai" "./pyproject.toml" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# What setup does: "install" (markitai for this user) or "dev" (sync the
# checkout, pre-commit hooks). Someone who cloned the repo and runs the
# installer from its root most likely wants markitai installed, so a checkout
# root no longer switches to dev mode silently: it asks, defaulting to install.
SETUP_MODE="install"
CHECKOUT_ANNOUNCED=false

# Sets: SETUP_MODE (from MARKITAI_SETUP_MODE or the question)
choose_setup_mode() {
    _csm_env=$(printf '%s' "${MARKITAI_SETUP_MODE:-}" | tr '[:upper:]' '[:lower:]')
    case "$_csm_env" in
        install|user)
            SETUP_MODE="install"
            ;;
        dev|development)
            SETUP_MODE="dev"
            ;;
        *)
            if [ -n "$_csm_env" ]; then
                clack_warn "$(i18n warn_unknown_mode): ${MARKITAI_SETUP_MODE}"
            fi
            if is_dev_mode; then
                CHECKOUT_ANNOUNCED=true
                clack_info "$(i18n repo_root_detected)"
                # Ask only when stdin is a terminal, like the install-source
                # question: `curl | sh` from a checkout is meant to install.
                if [ -t 0 ]; then
                    if clack_confirm "$(i18n confirm_dev_mode)" "n"; then
                        SETUP_MODE="dev"
                    fi
                else
                    clack_info "$(i18n info_mode_default)"
                    return 0
                fi
            fi
            ;;
    esac

    if [ "$SETUP_MODE" = "dev" ] && ! get_project_root >/dev/null; then
        clack_error "$(i18n error_dev_needs_repo)"
        clack_cancel "$(i18n error_setup_failed)"
        exit 1
    fi

    if [ "$SETUP_MODE" = "dev" ]; then
        clack_info "$(i18n mode_dev)"
    else
        clack_info "$(i18n mode_user)"
    fi
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

# In a markitai checkout, install markitai from it (default) or from PyPI.
# Someone running the installer from a checkout expects that code installed;
# MARKITAI_SETUP_SOURCE chooses without the question, and a pinned
# MARKITAI_VERSION names a PyPI release.
# Sets: MARKITAI_SOURCE, MARKITAI_LOCAL_PATH
choose_install_source() {
    _cis_env=$(printf '%s' "${MARKITAI_SETUP_SOURCE:-}" | tr '[:upper:]' '[:lower:]')
    case "$_cis_env" in
        local|checkout|source) _cis_want="local" ;;
        pypi|release) _cis_want="pypi" ;;
        "") _cis_want="" ;;
        *)
            clack_warn "$(i18n warn_unknown_source): ${MARKITAI_SETUP_SOURCE}"
            _cis_want=""
            ;;
    esac

    if ! detect_markitai_repo; then
        if [ "$_cis_want" = "local" ]; then
            clack_error "$(i18n error_local_needs_repo)"
            clack_cancel "$(i18n error_setup_failed)"
            exit 1
        fi
        return 0
    fi

    if [ "$CHECKOUT_ANNOUNCED" != true ]; then
        clack_info "$(i18n repo_detected): $MARKITAI_REPO_ROOT"
    fi
    if [ -z "$_cis_want" ]; then
        if [ -n "${MARKITAI_VERSION:-}" ]; then
            _cis_want="pypi"
            clack_info "$(i18n info_source_pinned)"
        elif [ ! -t 0 ]; then
            # No terminal (e.g. curl | sh): the default without asking
            _cis_want="local"
            clack_info "$(i18n info_repo_noninteractive)"
        elif clack_confirm "$(i18n confirm_local_install)" "y"; then
            _cis_want="local"
        else
            _cis_want="pypi"
        fi
    fi

    if [ "$_cis_want" = "local" ]; then
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

# PATH as setup found it, to tell whether markitai's commands will be found
# in the user's next shell; and the folder uv puts tool commands in.
SETUP_ORIGINAL_PATH="$PATH"
UV_TOOL_BIN=""

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
        _uv_version=$(uv --version 2>/dev/null | head -n1 | awk '{print $1" "$2}')
        clack_success "$_uv_version ($(i18n already_installed))"
        track_install "uv" "installed"
        return 0
    fi

    # Prompt to install
    if ! clack_confirm "$(i18n confirm_uv)" "y"; then
        clack_skip "$(i18n uv) $(i18n skipped)"
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
            _uv_version=$(uv --version 2>/dev/null | head -n1 | awk '{print $1" "$2}')
            clack_success "$_uv_version ($(i18n installed))"
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
    clack_step "$(i18n installing) $(i18n python) 3.13..."
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

# Put the folder uv installs tool commands into on PATH for the rest of the
# run. UV_TOOL_BIN_DIR or XDG_BIN_HOME can move it away from ~/.local/bin, and
# the version check, `doctor --suggest-extras` and `init` all run markitai.
use_uv_tool_bin_dir() {
    UV_TOOL_BIN=$(uv tool dir --bin 2>/dev/null) || UV_TOOL_BIN=""
    [ -n "$UV_TOOL_BIN" ] || UV_TOOL_BIN="$HOME/.local/bin"
    case ":$PATH:" in
        *":$UV_TOOL_BIN:"*) ;;
        *) PATH="$UV_TOOL_BIN:$PATH"; export PATH ;;
    esac
}

# Say so when markitai's command folder is not on the PATH setup started with:
# otherwise "Getting started" recommends commands the shell cannot find.
warn_if_tool_bin_off_path() {
    [ -n "$UV_TOOL_BIN" ] || return 0
    case ":$SETUP_ORIGINAL_PATH:" in
        *":$UV_TOOL_BIN:"*) return 0 ;;
    esac
    clack_warn "$(i18n warn_bin_off_path): $UV_TOOL_BIN"
    clack_detail_row "$(i18n info_bin_off_path_fix)"
    clack_cmd "uv tool update-shell"
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
                use_uv_tool_bin_dir
                _mi_version=$(markitai --version 2>/dev/null | awk '{print $NF}' || echo "")
                clack_success "$(i18n markitai) $_mi_version"
                track_install "markitai" "installed"
                return 0
            fi
        fi
        # Upgrade failed (or local source), try force reinstall
        if clack_spinner "$(i18n installing) $(i18n markitai)..." uv tool install "$_mi_pkg" --python "$PYTHON_CMD" --force; then
            use_uv_tool_bin_dir
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
            use_uv_tool_bin_dir
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
MARKITAI_ALL_FALLBACK_EXTRAS="browser,extra-fetch,svg,heif,legacy,mcp,ocr,serve"
# Only drop explicitly retired capabilities; preserve other receipt extras,
# including extensions supplied by a local/custom package.
MARKITAI_RETIRED_EXTRAS="kreuzberg"
# Extras the user explicitly turned down. `markitai doctor --suggest-extras`
# recommends `ocr` unconditionally, so without this list the finalize pass
# would reinstall exactly what was just declined and make the prompt a lie.
MARKITAI_DECLINED_EXTRAS=""
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

markitai_extra_retired() {
    # These extras were retired in 1.0.0. An explicit pre-1.0 PyPI pin still
    # supports them; a local source takes precedence over the version pin.
    if [ "${MARKITAI_SOURCE:-pypi}" != "local" ]; then
        case "${MARKITAI_VERSION:-}" in
            0.*) return 1 ;;
        esac
    fi
    case ",$MARKITAI_RETIRED_EXTRAS," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

# Track a markitai extra for the next combined installation.
# Usage: install_markitai_extra "claude-agent"
install_markitai_extra() {
    _extra_name="$1"
    [ -z "$_extra_name" ] && return 0
    if markitai_extra_retired "$_extra_name"; then
        return 0
    fi
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

# Return success when the user answered "no" to an extra in this run.
markitai_extra_declined() {
    case ",$MARKITAI_DECLINED_EXTRAS," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

# Remember a declined extra so no later suggestion pass re-adds it.
decline_markitai_extra() {
    markitai_extra_declined "$1" && return 0
    MARKITAI_DECLINED_EXTRAS="${MARKITAI_DECLINED_EXTRAS:+$MARKITAI_DECLINED_EXTRAS,}$1"
}

# Preserve non-retired extras before asking for new capabilities. Keep the
# raw receipt too: retired entries require a reinstall even with no additions.
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

# Return success when the receipt needs retired extras removed or new ones
# added. `uv tool upgrade` reuses the old requirements, so either change needs
# `uv tool install` with the cleaned combined spec (including core-only).
markitai_extras_need_update() {
    _old_ifs="$IFS"; IFS=','
    for _extra in $MARKITAI_RECEIPT_EXTRAS; do
        if markitai_extra_retired "$_extra"; then
            IFS="$_old_ifs"
            return 0
        fi
    done
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
    else
        decline_markitai_extra "serve"
    fi
}

track_markitai_serve() {
    if markitai_extra_enabled "serve"; then
        track_install "serve" "installed"
    else
        track_install "serve" "skipped"
    fi
}

# Resolve OCR before installing markitai, for the same reason as `serve`.
# rapidocr moved out of the core dependencies: it is ~150MB of models that
# nobody converting born-digital documents ever executes, so the guided
# installer asks instead of deciding. A repeat run skips the question when the
# uv receipt already records the extra.
select_markitai_ocr() {
    markitai_extra_enabled "ocr" && return 0
    if clack_confirm_optional "$(i18n confirm_ocr)" "y"; then
        install_markitai_extra "ocr"
    else
        decline_markitai_extra "ocr"
    fi
}

track_markitai_ocr() {
    if markitai_extra_enabled "ocr"; then
        track_install "ocr" "installed"
    else
        track_install "ocr" "skipped"
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
            markitai_extra_declined "$_e" && continue
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
    if clack_spinner "$(i18n installing) $(i18n markitai) extras..." \
        uv tool install "$_mi_pkg" --python "$PYTHON_CMD" --force; then
        clack_success "$(i18n markitai) extras: $(printf '%s' "$MARKITAI_EXTRAS" | sed 's/,/, /g')"
        return 0
    fi
    # The spinner kept the full stderr in the setup log; reuse its tail.
    _uv_err="$CLACK_PENDING_DETAIL"
    CLACK_PENDING_DETAIL=""

    # Full install failed: retry without SDK-dependent extras
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
            clack_success "$(i18n markitai) extras: $(printf '%s' "$_safe_extras" | sed 's/,/, /g')"
            [ -n "$_skipped" ] && clack_warn "$(i18n skipped) extras: $_skipped (SDK $(i18n not_found))"
        else
            clack_warn "$(i18n markitai) extras update $(i18n failed)"
        fi
    else
        clack_warn "$(i18n markitai) extras update $(i18n failed)"
        clack_print_detail "$_uv_err" 3
    fi
    return 0
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
        clack_skip "$(i18n precommit) $(i18n skipped)"
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
        clack_skip "$(i18n playwright) $(i18n skipped)"
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
        clack_skip "$(i18n claude_cli) $(i18n skipped)"
        track_install "claude_cli" "skipped"
        return 2
    fi

    clack_step "$(i18n installing) $(i18n claude_cli)..."

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

    # Fallback: bun/pnpm/npm
    if command -v bun >/dev/null 2>&1; then
        if clack_run_quiet "$(i18n installing) $(i18n claude_cli)" bun add -g @anthropic-ai/claude-code; then
            clack_success "$(i18n claude_cli) $(i18n installed)"
            install_markitai_extra "claude-agent" || true
            track_install "claude_cli" "installed"
            return 0
        fi
    elif command -v pnpm >/dev/null 2>&1; then
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

# This section reports what is installed. ChatGPT is markitai's third
# subscription route and installs nothing at all, so without a line saying
# so it looks absent rather than already satisfied. Signing in belongs with
# the other configuration guidance at the end.
note_chatgpt_route() {
    clack_info "$(i18n info_chatgpt_route)"
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
        clack_skip "$(i18n copilot_cli) $(i18n skipped)"
        track_install "copilot_cli" "skipped"
        return 2
    fi

    clack_step "$(i18n installing) $(i18n copilot_cli)..."

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

    # Fallback: bun/pnpm/npm
    if command -v bun >/dev/null 2>&1; then
        if clack_run_quiet "$(i18n installing) $(i18n copilot_cli)" bun add -g @github/copilot; then
            clack_success "$(i18n copilot_cli) $(i18n installed)"
            install_markitai_extra "copilot" || true
            track_install "copilot_cli" "installed"
            return 0
        fi
    elif command -v pnpm >/dev/null 2>&1; then
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

# Print one row per tracked component.
# Usage: print_summary_rows "comp1|comp2|" installed|skipped|failed
print_summary_rows() {
    _psr_rest="$1"
    while [ -n "$_psr_rest" ]; do
        case "$_psr_rest" in
            *"|"*) ;;
            *) _psr_rest="$_psr_rest|" ;;
        esac
        _psr_comp="${_psr_rest%%|*}"
        _psr_rest="${_psr_rest#*|}"
        [ -n "$_psr_comp" ] || continue
        case "$2" in
            installed) clack_success "$(i18n "$_psr_comp")" ;;
            skipped) clack_skip "$(i18n "$_psr_comp") $(i18n skipped)" ;;
            failed) clack_item "$RED" "$S_CROSS" "$(i18n "$_psr_comp") $(i18n failed)" ;;
        esac
    done
}

# Print installation summary: one row per component, its mark saying what
# happened (installed, skipped on purpose, failed).
# Usage: print_summary [--no-header]
print_summary() {
    [ "${1:-}" = "--no-header" ] || clack_section "$(i18n section_summary)"
    print_summary_rows "$INSTALLED_COMPONENTS" installed
    print_summary_rows "$SKIPPED_COMPONENTS" skipped
    print_summary_rows "$FAILED_COMPONENTS" failed
    # A headless run skips every optional component by default; say how to
    # get them without a terminal (doctor would list one fix per extra).
    if [ -n "$SKIPPED_COMPONENTS" ] && ! has_interactive_tty \
        && ! optional_install_requested; then
        clack_hint "$(i18n optional_headless_hint)" "export MARKITAI_INSTALL_OPTIONAL=1"
    fi
}

# Documentation and issue links close both completion notes.
print_help_links() {
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

    clack_section "$(i18n getting_started)"
    warn_if_tool_bin_off_path
    clack_hint "$(i18n convert_file)" "markitai file.pdf"
    clack_hint "$(i18n interactive_mode)" "markitai -I"
    # Only advertise the serve command when its dependencies were selected.
    if markitai_extra_enabled "serve"; then
        clack_hint "$(i18n start_web_ui)" "markitai serve"
    fi
    clack_hint "$(i18n configure_llm)" "markitai init"
    clack_hint "$(i18n show_help)" "markitai --help"
    # Only advertise the short alias when it actually resolves to markitai.
    if [ "${MKAI_USABLE:-false}" = "true" ]; then
        clack_info "$(i18n short_alias)"
    fi
    print_help_links
}

# Print dev mode completion message
# Usage: print_dev_completion
print_dev_completion() {
    clack_section "$(i18n quick_start)"
    clack_hint "$(i18n configure_env)" "cp .env.example .env"
    clack_hint "$(i18n interactive_mode)" "uv run markitai -I"
    clack_hint "$(i18n start_web_ui)" "uv run markitai serve"
    clack_hint "$(i18n run_tests)" "uv run pytest"
    clack_hint "$(i18n run_cli)" "uv run markitai --help"
    print_help_links
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
    configure_mirrors
    choose_install_source

    clack_section "$(i18n section_core)"
    install_uv || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }
    detect_python || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }
    load_existing_markitai_extras
    select_markitai_serve
    select_markitai_ocr
    install_markitai || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }
    track_markitai_serve
    track_markitai_ocr

    clack_section "$(i18n section_optional)"
    install_optional_playwright || true

    clack_section "$(i18n section_llm_cli)"
    install_optional_claude_cli || true
    install_optional_copilot_cli || true
    note_chatgpt_route

    # The extras step reports in the summary, not under the last section.
    clack_section "$(i18n section_summary)"
    finalize_markitai_extras || true

    init_config >/dev/null 2>&1

    print_summary --no-header
    print_user_completion
    if [ -n "$FAILED_COMPONENTS" ]; then
        clack_outro_warn "$(i18n setup_complete_with_warnings)"
    else
        clack_outro "$(i18n setup_complete)"
    fi
}

# Dev mode main flow
run_dev_setup() {
    configure_mirrors

    clack_section "$(i18n section_prerequisites)"
    install_uv || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }
    detect_python || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }

    clack_section "$(i18n section_dev_env)"
    sync_dependencies || { print_summary; clack_cancel "$(i18n error_setup_failed)"; exit 1; }
    install_precommit || true

    clack_section "$(i18n section_optional)"
    install_optional_playwright || true

    clack_section "$(i18n section_llm_cli)"
    install_optional_claude_cli || true
    install_optional_copilot_cli || true
    note_chatgpt_route

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
    clack_intro "$(i18n welcome)"
    warn_if_root
    choose_setup_mode
    if [ "$SETUP_MODE" = "dev" ]; then
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
