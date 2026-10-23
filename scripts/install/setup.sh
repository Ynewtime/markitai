#!/bin/sh
# Markitai setup: installs the markitai executable (2.x) from a GitHub Release.
# POSIX sh (bash, zsh, dash); English or Chinese after the locale.
#
# Usage:
#   curl -fsSL https://markitai.dev/setup.sh | sh                    # install or update
#   curl -fsSL https://markitai.dev/setup.sh | sh -s -- --uninstall  # remove it
#
# It downloads markitai-v<version>-<target>.tar.gz and the release's SHA256SUMS,
# checks the archive against it, and installs `markitai` with its `mkai` and
# `markitai-mcp` links into ~/.local/bin; the license notices and offline guides
# the archive carries go to ~/.local/share/markitai/current. No root needed.
#
# Environment:
#   MARKITAI_VERSION=X.Y.Z       install this release (default: the latest)
#   MARKITAI_INSTALL_DIR=DIR     command folder (default: ~/.local/bin)
#   MARKITAI_INSTALL_OPTIONAL=1  also prepare the optional components
#                                (markitai doctor --fix: local OCR models and,
#                                when no browser is found, a headless Chromium)
#                                without a terminal to ask; 0 never prepares them
#   MARKITAI_DOWNLOAD_URL=URL    releases base, for a mirror
#                                (default: https://github.com/Ynewtime/markitai/releases)
#   MARKITAI_UNINSTALL=1         the same as --uninstall
#   NO_COLOR=1                   no colours; the marks stay

set -eu

REPOSITORY_RELEASES="https://github.com/Ynewtime/markitai/releases"
SCRIPT_URL="https://markitai.dev/setup.sh"

detect_lang() {
    case "${LC_ALL:-${LC_MESSAGES:-${LANG:-}}}" in
        zh*.[Uu][Tt][Ff]-8*|zh*.[Uu][Tt][Ff]8*) echo zh ;;
        *) echo en ;;
    esac
}
LANG_CODE=$(detect_lang)

# setup.ps1 carries the same messages.
msg() {
    if [ "$LANG_CODE" = zh ]; then
        case "$1" in
            welcome) echo "Markitai 安装" ;;
            welcome_uninstall) echo "Markitai 卸载" ;;
            ready) echo "markitai 已就绪" ;;
            done_warnings) echo "markitai 已就绪，但有部分组件失败" ;;
            done_uninstall) echo "markitai 已卸载" ;;
            cancelled) echo "已取消安装" ;;
            failed) echo "安装失败" ;;
            section_install) echo "安装" ;;
            section_optional) echo "可选组件" ;;
            section_start) echo "开始使用" ;;
            platform) echo "平台" ;;
            release) echo "版本" ;;
            downloading) echo "正在下载" ;;
            verified) echo "已校验 SHA-256" ;;
            installed) echo "已安装" ;;
            notices) echo "许可声明与离线文档" ;;
            replaced_python) echo "已移除 uv 安装的 Python 版 markitai" ;;
            confirm_optional) echo "现在准备可选组件? (本地 OCR 模型约 45 MB；找不到浏览器时下载无头 Chromium，约 280 MB)" ;;
            optional_ready) echo "可选组件已就绪" ;;
            optional_failed) echo "可选组件未能全部准备好；稍后可再运行" ;;
            optional_headless) echo "没有终端，可选组件已跳过；要一并准备，设置此变量后重新运行安装脚本" ;;
            optional_later) echo "稍后准备可选组件" ;;
            warn_root) echo "正在以 root 身份运行" ;;
            warn_root_risk) echo "markitai 按用户安装，从不需要 root；以 root 运行会装进 root 的主目录" ;;
            confirm_root) echo "仍以 root 身份继续?" ;;
            root_noninteractive) echo "无终端可询问；以 root 继续" ;;
            off_path) echo "当前 PATH 不含 markitai 的命令目录" ;;
            off_path_fix) echo "把它加入 PATH，然后打开新终端" ;;
            shadowed) echo "PATH 中更靠前的另一个 markitai 会被优先使用" ;;
            alias_skipped) echo "已存在同名命令，未覆盖" ;;
            convert_file) echo "转换文件" ;;
            interactive) echo "交互模式" ;;
            web_ui) echo "启动 Web UI" ;;
            configure_llm) echo "配置 LLM" ;;
            show_help) echo "查看帮助" ;;
            update) echo "更新" ;;
            uninstall) echo "卸载" ;;
            removed) echo "已删除" ;;
            nothing_installed) echo "未找到安装脚本安装的 markitai" ;;
            state_kept) echo "配置、缓存和模型仍在此目录；不再需要时可删除" ;;
            docs) echo "文档" ;;
            error_os) echo "不支持此操作系统；Windows 请使用 setup.ps1" ;;
            error_arch) echo "不支持此处理器架构" ;;
            error_glibc) echo "需要 glibc 2.28 或更高版本的 Linux (不支持 musl/Alpine)" ;;
            error_download_tool) echo "需要 curl 或 wget" ;;
            error_hash_tool) echo "需要 sha256sum、shasum 或 openssl 来校验下载" ;;
            error_download) echo "下载失败" ;;
            error_no_asset) echo "该版本没有此平台的归档" ;;
            error_checksum) echo "SHA-256 不匹配，下载的文件已删除" ;;
            error_run) echo "下载的 markitai 无法在此系统运行" ;;
            error_python_version) echo "markitai 1.x 是 Python 包；请用 uv tool install 或 pip 安装该版本" ;;
            error_version) echo "MARKITAI_VERSION 应为 X.Y.Z 形式" ;;
            warn_http) echo "下载地址不是 HTTPS" ;;
            *) echo "$1" ;;
        esac
        return
    fi
    case "$1" in
        welcome) echo "Markitai setup" ;;
        welcome_uninstall) echo "Markitai uninstall" ;;
        ready) echo "markitai is ready" ;;
        done_warnings) echo "markitai is ready, but some components failed" ;;
        done_uninstall) echo "markitai is uninstalled" ;;
        cancelled) echo "Setup cancelled" ;;
        failed) echo "Setup failed" ;;
        section_install) echo "Install" ;;
        section_optional) echo "Optional components" ;;
        section_start) echo "Getting started" ;;
        platform) echo "Platform" ;;
        release) echo "Release" ;;
        downloading) echo "Downloading" ;;
        verified) echo "SHA-256 verified" ;;
        installed) echo "Installed" ;;
        notices) echo "License notices and offline guides" ;;
        replaced_python) echo "Removed the Python markitai that uv installed" ;;
        confirm_optional) echo "Prepare the optional components now? (local OCR models, about 45 MB; a headless Chromium, about 280 MB, when no browser is found)" ;;
        optional_ready) echo "Optional components are ready" ;;
        optional_failed) echo "Not every optional component could be prepared; run this again later" ;;
        optional_headless) echo "No terminal to ask, so optional components were skipped; to prepare them too, set this and run setup again" ;;
        optional_later) echo "Prepare optional components later" ;;
        warn_root) echo "Running as root" ;;
        warn_root_risk) echo "markitai installs per user and never needs root; as root it goes into root's home" ;;
        confirm_root) echo "Continue as root anyway?" ;;
        root_noninteractive) echo "No terminal to ask; continuing as root" ;;
        off_path) echo "This shell's PATH does not include markitai's command folder" ;;
        off_path_fix) echo "Add it to PATH, then open a new terminal" ;;
        shadowed) echo "Another markitai comes first on PATH and runs instead" ;;
        alias_skipped) echo "A command of that name exists; left alone" ;;
        convert_file) echo "Convert a file" ;;
        interactive) echo "Interactive mode" ;;
        web_ui) echo "Start Web UI" ;;
        configure_llm) echo "Configure LLM" ;;
        show_help) echo "Show help" ;;
        update) echo "Update" ;;
        uninstall) echo "Uninstall" ;;
        removed) echo "Removed" ;;
        nothing_installed) echo "No markitai installed by this script was found" ;;
        state_kept) echo "Configuration, cache and models stay in this folder; delete it when no longer needed" ;;
        docs) echo "Docs" ;;
        error_os) echo "This operating system is not supported; on Windows use setup.ps1" ;;
        error_arch) echo "This processor architecture is not supported" ;;
        error_glibc) echo "Needs Linux with glibc 2.28 or later (musl/Alpine is not supported)" ;;
        error_download_tool) echo "Needs curl or wget" ;;
        error_hash_tool) echo "Needs sha256sum, shasum or openssl to verify the download" ;;
        error_download) echo "Download failed" ;;
        error_no_asset) echo "This release has no archive for this platform" ;;
        error_checksum) echo "SHA-256 mismatch; the download was deleted" ;;
        error_run) echo "The downloaded markitai does not run on this system" ;;
        error_python_version) echo "markitai 1.x is the Python package; install that release with uv tool install or pip" ;;
        error_version) echo "MARKITAI_VERSION must look like X.Y.Z" ;;
        warn_http) echo "The download address is not HTTPS" ;;
        *) echo "$1" ;;
    esac
}

# ---- output: the tree-style rows the 1.x setup scripts print ----

use_color() {
    [ -z "${NO_COLOR:-}" ] || return 1
    [ "${TERM:-}" != dumb ] || return 1
    case "${FORCE_COLOR:-}" in ''|0|false) ;; *) return 0 ;; esac
    [ -t 1 ]
}
if use_color; then
    RED='\033[31m' GREEN='\033[32m' YELLOW='\033[33m' CYAN='\033[36m' GRAY='\033[90m' BOLD='\033[1m' NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' CYAN='' GRAY='' BOLD='' NC=''
fi
case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in
    *[Uu][Tt][Ff]-8*|*[Uu][Tt][Ff]8*)
        S_BAR="│" S_TOP="╭─" S_BRANCH="├─" S_BOTTOM="╰─" S_CHECK="✓" S_CROSS="✗" S_ARROW="→" S_DOT="•" S_CIRCLE="○" ;;
    *)
        S_BAR="|" S_TOP="+-" S_BRANCH="+-" S_BOTTOM="+-" S_CHECK="+" S_CROSS="x" S_ARROW=">" S_DOT="-" S_CIRCLE="o" ;;
esac
[ "${TERM:-}" != dumb ] || { S_BAR="|" S_TOP="+-" S_BRANCH="+-" S_BOTTOM="+-" S_CHECK="+" S_CROSS="x" S_ARROW=">" S_DOT="-" S_CIRCLE="o"; }

item() { printf "${GRAY}%s${NC}  %b%s${NC} %s\n" "$S_BAR" "$1" "$2" "$3"; }
intro() { printf "\n${GRAY}%s${NC} ${BOLD}%s${NC}\n" "$S_TOP" "$1"; }
section() { printf "${GRAY}%s${NC}\n${GRAY}%s${NC} ${BOLD}%s${NC}\n" "$S_BAR" "$S_BRANCH" "$1"; }
ok() { item "$GREEN" "$S_CHECK" "$1"; }
warn() { item "$YELLOW" "!" "$1"; }
info() { item "$GRAY" "$S_DOT" "$1"; }
step() { item "$CYAN" "$S_ARROW" "$1"; }
skip() { item "$GRAY" "$S_CIRCLE" "$1"; }
cmd() { printf "${GRAY}%s${NC}    ${CYAN}%s${NC}\n" "$S_BAR" "$1"; }
hint() { info "$1:"; cmd "$2"; }
outro() { printf "${GRAY}%s${NC}\n${GRAY}%s${NC} %b%s${NC}\n\n" "$S_BAR" "$S_BOTTOM" "$1" "$2"; }
fail() {
    item "$RED" "$S_CROSS" "$1"
    [ -z "${2:-}" ] || printf "${GRAY}%s${NC}    ${GRAY}%s${NC}\n" "$S_BAR" "$2"
    outro "$RED" "$(msg failed)"
    exit 1
}

has_tty() { [ -t 0 ] || ( : </dev/tty ) 2>/dev/null; }

# "? Question [Y/n] "; the answer comes from the terminal even under curl | sh.
confirm() {
    if [ "$2" = y ]; then _hint="[Y/n]"; else _hint="[y/N]"; fi
    printf "${GRAY}%s${NC}  ${CYAN}?${NC} %s ${GRAY}%s${NC} " "$S_BAR" "$1" "$_hint"
    _answer=""
    if [ -t 0 ]; then
        read -r _answer || _answer=""
    else
        { read -r _answer </dev/tty; } 2>/dev/null || _answer=""
    fi
    [ -n "$_answer" ] || _answer=$2
    case "$_answer" in [Yy]*) return 0 ;; *) return 1 ;; esac
}

truthy() { case "${1:-}" in 1|true|TRUE|yes|YES|on|ON) return 0 ;; *) return 1 ;; esac; }
falsy() { case "${1:-}" in 0|false|FALSE|no|NO|off|OFF) return 0 ;; *) return 1 ;; esac; }

# Home-relative paths read better, and copy back unchanged.
pretty() {
    case "$1" in
        "$HOME"/*) printf '~%s\n' "${1#"$HOME"}" ;;
        *) printf '%s\n' "$1" ;;
    esac
}

# ---- platform ----

detect_target() {
    case "$(uname -s)" in
        Darwin) _os=apple-darwin ;;
        Linux) _os=unknown-linux-gnu ;;
        *) fail "$(msg error_os)" "$(uname -s)" ;;
    esac
    case "$(uname -m)" in
        x86_64|amd64) _arch=x86_64 ;;
        arm64|aarch64) _arch=aarch64 ;;
        *) fail "$(msg error_arch)" "$(uname -m)" ;;
    esac
    # A shell under Rosetta 2 reports x86_64; the native build is faster.
    if [ "$_os" = apple-darwin ] && [ "$_arch" = x86_64 ] && \
        [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
        _arch=aarch64
    fi
    if [ "$_os" = unknown-linux-gnu ]; then
        _glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null || true)
        _major=$(echo "$_glibc" | sed -n 's/^glibc \([0-9]*\)\..*/\1/p')
        _minor=$(echo "$_glibc" | sed -n 's/^glibc [0-9]*\.\([0-9]*\).*/\1/p')
        if [ -z "$_major" ] || [ -z "$_minor" ] || ! glibc_at_least "$_major" "$_minor"; then
            fail "$(msg error_glibc)" "$(getconf GNU_LIBC_VERSION 2>/dev/null || echo "no glibc")"
        fi
    fi
    TARGET="$_arch-$_os"
}

glibc_at_least() { [ "$1" -gt 2 ] || { [ "$1" -eq 2 ] && [ "$2" -ge 28 ]; }; }

# ---- downloads ----

download() {
    case "$1" in
        https://*) _secure=true ;;
        *) _secure=false ;;
    esac
    if command -v curl >/dev/null 2>&1; then
        if [ "$_secure" = true ]; then
            curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL --retry 3 -o "$2" "$1"
        else
            curl -fsSL --retry 3 -o "$2" "$1"
        fi
    elif command -v wget >/dev/null 2>&1; then
        if [ "$_secure" = true ]; then
            wget --https-only -q -O "$2" "$1"
        else
            wget -q -O "$2" "$1"
        fi
    else
        fail "$(msg error_download_tool)"
    fi
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d ' ' -f 1
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$1" | sed 's/^.*= *//'
    else
        fail "$(msg error_hash_tool)"
    fi
}

# Sets VERSION and ARCHIVE from the release's SHA256SUMS (the latest
# release's, unless MARKITAI_VERSION names one).
resolve_release() {
    if [ -n "${MARKITAI_VERSION:-}" ]; then
        _wanted=${MARKITAI_VERSION#v}
        case "$_wanted" in
            0.*|1.*) fail "$(msg error_python_version)" "uv tool install markitai==$_wanted" ;;
            [0-9]*.[0-9]*.[0-9]*) ;;
            *) fail "$(msg error_version)" "$MARKITAI_VERSION" ;;
        esac
        _sums_url="$BASE/download/v$_wanted/SHA256SUMS"
    else
        _sums_url="$BASE/latest/download/SHA256SUMS"
    fi
    download "$_sums_url" "$WORK/SHA256SUMS" 2>"$WORK/error" || fail "$(msg error_download)" "$_sums_url"
    _line=$(grep -E "^[0-9a-f]{64}  markitai-v[^ ]+-$TARGET\.tar\.gz\$" "$WORK/SHA256SUMS" | head -n 1) || true
    [ -n "$_line" ] || fail "$(msg error_no_asset)" "$TARGET ($_sums_url)"
    EXPECTED_SHA256=${_line%% *}
    ARCHIVE=${_line#*  }
    VERSION=${ARCHIVE#markitai-v}
    VERSION=${VERSION%-"$TARGET".tar.gz}
    if [ -n "${MARKITAI_VERSION:-}" ] && [ "$VERSION" != "$_wanted" ]; then
        fail "$(msg error_no_asset)" "$ARCHIVE"
    fi
}

# ---- install ----

# Python markitai 1.x installed with uv would keep its own markitai, mkai and
# markitai-mcp in the same folder; the 2.x executable replaces it.
remove_python_markitai() {
    command -v uv >/dev/null 2>&1 || return 0
    uv tool list 2>/dev/null | grep -q '^markitai ' || return 0
    if uv tool uninstall markitai >/dev/null 2>&1; then
        ok "$(msg replaced_python)"
    else
        warn "uv tool uninstall markitai"
    fi
}

# A link is replaced, and so is a Python markitai entry-point script (pip
# --user writes those); any other file of that name is another program's.
link_alias() {
    _alias="$BIN_DIR/$1"
    if [ -e "$_alias" ] && [ ! -L "$_alias" ]; then
        if [ "$(head -c 2 "$_alias" 2>/dev/null)" != "#!" ] || ! grep -q "markitai" "$_alias" 2>/dev/null; then
            warn "$(msg alias_skipped): $(pretty "$_alias")"
            return 0
        fi
    fi
    rm -f "$_alias"
    ln -s markitai "$_alias"
}

install_release() {
    _unpacked="$WORK/unpacked"
    mkdir "$_unpacked"
    tar -xzf "$WORK/$ARCHIVE" -C "$_unpacked"
    [ -f "$_unpacked/markitai" ] || fail "$(msg error_run)" "markitai is missing from $ARCHIVE"
    if ! _reported=$("$_unpacked/markitai" --version 2>&1); then
        fail "$(msg error_run)" "$_reported"
    fi
    remove_python_markitai
    mkdir -p "$BIN_DIR"
    # Renaming over the old executable is atomic, and a running markitai keeps its file.
    cp "$_unpacked/markitai" "$BIN_DIR/.markitai.new.$$"
    chmod 755 "$BIN_DIR/.markitai.new.$$"
    mv -f "$BIN_DIR/.markitai.new.$$" "$BIN_DIR/markitai"
    link_alias mkai
    link_alias markitai-mcp
    ok "$(msg installed): $(pretty "$BIN_DIR/markitai") ($_reported)"

    mkdir -p "$SHARE_ROOT"
    rm -rf "$SHARE_ROOT/current.new" "$SHARE_ROOT/current.old"
    rm -f "$_unpacked/markitai" "$_unpacked/mkai" "$_unpacked/markitai-mcp"
    mv "$_unpacked" "$SHARE_ROOT/current.new"
    printf 'version=%s\ntarget=%s\nbin_dir=%s\n' "$VERSION" "$TARGET" "$BIN_DIR" >"$SHARE_ROOT/current.new/receipt"
    [ ! -d "$SHARE_ROOT/current" ] || mv "$SHARE_ROOT/current" "$SHARE_ROOT/current.old"
    mv "$SHARE_ROOT/current.new" "$SHARE_ROOT/current"
    rm -rf "$SHARE_ROOT/current.old"
    info "$(msg notices): $(pretty "$SHARE_ROOT/current")"
}

prepare_optional() {
    section "$(msg section_optional)"
    if falsy "${MARKITAI_INSTALL_OPTIONAL:-}"; then
        hint "$(msg optional_later)" "markitai doctor --fix"
        return 0
    fi
    if ! truthy "${MARKITAI_INSTALL_OPTIONAL:-}"; then
        if ! has_tty; then
            skip "$(msg optional_headless)"
            cmd "MARKITAI_INSTALL_OPTIONAL=1"
            return 0
        fi
        if ! confirm "$(msg confirm_optional)" y; then
            hint "$(msg optional_later)" "markitai doctor --fix"
            return 0
        fi
    fi
    step "markitai doctor --fix"
    if "$BIN_DIR/markitai" doctor --fix >"$WORK/doctor.log" 2>&1; then
        ok "$(msg optional_ready)"
    else
        warn "$(msg optional_failed)"
        cmd "markitai doctor --fix"
        WARNINGS=true
    fi
}

# First install only: `init --yes` writes a configuration with the providers
# this machine already has; an existing one is never touched.
init_config() {
    _home=${MARKITAI_HOME:-$HOME/.markitai}
    [ ! -f "$_home/config.json" ] || return 0
    "$BIN_DIR/markitai" init --yes >/dev/null 2>&1 || true
}

path_advice() {
    case ":$ORIGINAL_PATH:" in
        *":$BIN_DIR:"*) ;;
        *)
            warn "$(msg off_path): $(pretty "$BIN_DIR")"
            info "$(msg off_path_fix):"
            # Written as $HOME/... when it is under home, so the line can be copied anywhere.
            case "$BIN_DIR" in
                "$HOME"/*) _entry="\$HOME${BIN_DIR#"$HOME"}" ;;
                *) _entry=$BIN_DIR ;;
            esac
            case "${SHELL:-}" in
                */fish) cmd "fish_add_path $(pretty "$BIN_DIR")" ;;
                */zsh) cmd "echo 'export PATH=\"$_entry:\$PATH\"' >> ~/.zshrc" ;;
                */bash)
                    if [ "$(uname -s)" = Darwin ]; then
                        cmd "echo 'export PATH=\"$_entry:\$PATH\"' >> ~/.bash_profile"
                    else
                        cmd "echo 'export PATH=\"$_entry:\$PATH\"' >> ~/.bashrc"
                    fi ;;
                *) cmd "export PATH=\"$_entry:\$PATH\"" ;;
            esac
            return 0 ;;
    esac
    _first=$(PATH=$ORIGINAL_PATH command -v markitai 2>/dev/null || true)
    if [ -n "$_first" ] && [ "$_first" != "$BIN_DIR/markitai" ]; then
        warn "$(msg shadowed): $(pretty "$_first")"
    fi
}

getting_started() {
    section "$(msg section_start)"
    path_advice
    hint "$(msg convert_file)" "markitai file.pdf"
    hint "$(msg interactive)" "markitai -I"
    hint "$(msg web_ui)" "markitai serve"
    hint "$(msg configure_llm)" "markitai init"
    hint "$(msg show_help)" "markitai --help"
    hint "$(msg update)" "curl -fsSL $SCRIPT_URL | sh"
    hint "$(msg uninstall)" "curl -fsSL $SCRIPT_URL | sh -s -- --uninstall"
    info "$(msg docs): https://markitai.dev"
}

uninstall() {
    intro "$(msg welcome_uninstall)"
    _bin=$BIN_DIR
    if [ -f "$SHARE_ROOT/current/receipt" ]; then
        _recorded=$(sed -n 's/^bin_dir=//p' "$SHARE_ROOT/current/receipt")
        [ -z "$_recorded" ] || _bin=$_recorded
    fi
    _removed=false
    for _name in mkai markitai-mcp; do
        if [ -L "$_bin/$_name" ] && [ "$(readlink "$_bin/$_name")" = markitai ]; then
            rm -f "$_bin/$_name"
            ok "$(msg removed): $(pretty "$_bin/$_name")"
        fi
    done
    if [ -f "$_bin/markitai" ] && [ ! -L "$_bin/markitai" ]; then
        rm -f "$_bin/markitai"
        ok "$(msg removed): $(pretty "$_bin/markitai")"
        _removed=true
    fi
    if [ -d "$SHARE_ROOT/current" ]; then
        rm -rf "$SHARE_ROOT/current"
        rmdir "$SHARE_ROOT" 2>/dev/null || true
        ok "$(msg removed): $(pretty "$SHARE_ROOT/current")"
        _removed=true
    fi
    [ "$_removed" = true ] || info "$(msg nothing_installed)"
    _home=${MARKITAI_HOME:-$HOME/.markitai}
    if [ -d "$_home" ]; then
        info "$(msg state_kept): $(pretty "$_home")"
    fi
    outro "$GREEN" "$(msg done_uninstall)"
}

usage() {
    cat <<EOF
Install or update markitai: curl -fsSL $SCRIPT_URL | sh
Remove it:                  curl -fsSL $SCRIPT_URL | sh -s -- --uninstall

Environment: MARKITAI_VERSION=X.Y.Z, MARKITAI_INSTALL_DIR=DIR (default ~/.local/bin),
MARKITAI_INSTALL_OPTIONAL=1|0 (markitai doctor --fix without asking, or never),
MARKITAI_DOWNLOAD_URL=URL (releases base), MARKITAI_UNINSTALL=1, NO_COLOR=1
EOF
}

main() {
    MODE=install
    for _arg in "$@"; do
        case "$_arg" in
            --uninstall) MODE=uninstall ;;
            -h|--help) usage; exit 0 ;;
            *) echo "setup.sh: unknown option $_arg (try --help)" >&2; exit 2 ;;
        esac
    done
    ! truthy "${MARKITAI_UNINSTALL:-}" || MODE=uninstall
    : "${HOME:?HOME is not set}"
    BIN_DIR=${MARKITAI_INSTALL_DIR:-$HOME/.local/bin}
    SHARE_ROOT=${XDG_DATA_HOME:-$HOME/.local/share}/markitai
    ORIGINAL_PATH=$PATH
    WARNINGS=false
    if [ "$MODE" = uninstall ]; then
        uninstall
        return 0
    fi

    intro "$(msg welcome)"
    if [ "$(id -u)" -eq 0 ]; then
        warn "$(msg warn_root)"
        info "$(msg warn_root_risk)"
        if has_tty; then
            confirm "$(msg confirm_root)" n || { outro "$RED" "$(msg cancelled)"; exit 1; }
        else
            info "$(msg root_noninteractive)"
        fi
    fi
    BASE=${MARKITAI_DOWNLOAD_URL:-$REPOSITORY_RELEASES}
    BASE=${BASE%/}
    case "$BASE" in https://*) ;; *) warn "$(msg warn_http): $BASE" ;; esac

    section "$(msg section_install)"
    detect_target
    info "$(msg platform): $TARGET"
    WORK=$(mktemp -d "${TMPDIR:-/tmp}/markitai-setup.XXXXXX")
    trap 'rm -rf "$WORK"' EXIT
    trap 'rm -rf "$WORK"; exit 130' INT TERM
    resolve_release
    info "$(msg release): $VERSION"
    step "$(msg downloading) $ARCHIVE"
    _url="$BASE/download/v$VERSION/$ARCHIVE"
    download "$_url" "$WORK/$ARCHIVE" || fail "$(msg error_download)" "$_url"
    _actual=$(sha256_of "$WORK/$ARCHIVE")
    if [ "$_actual" != "$EXPECTED_SHA256" ]; then
        rm -f "$WORK/$ARCHIVE"
        fail "$(msg error_checksum)" "$ARCHIVE: $_actual != $EXPECTED_SHA256"
    fi
    ok "$(msg verified)"
    install_release
    init_config
    prepare_optional
    getting_started
    if [ "$WARNINGS" = true ]; then
        outro "$YELLOW" "$(msg done_warnings)"
    else
        outro "$GREEN" "$(msg ready)"
    fi
}

main "$@"
