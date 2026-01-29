#!/bin/sh
# Markitai 环境配置脚本 (用户版)
# 支持 bash/zsh/dash 等 POSIX 兼容 shell

set -e

# ============================================================
# 库加载（支持本地和远程执行）
# ============================================================

LIB_BASE_URL="https://raw.githubusercontent.com/Ynewtime/markitai/main/scripts"

load_library() {
    # 检测是否为本地执行（脚本文件存在且不是 sh/bash）
    if [ -f "$0" ] && [ "$(basename "$0")" != "sh" ] && [ "$(basename "$0")" != "bash" ] && [ "$(basename "$0")" != "dash" ]; then
        # 本地执行
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        if [ -f "$SCRIPT_DIR/lib.sh" ]; then
            . "$SCRIPT_DIR/lib.sh"
            return 0
        fi
    fi

    # 远程执行 (curl | sh) - 下载 lib.sh
    if ! command -v curl >/dev/null 2>&1; then
        echo "错误: 远程执行需要 curl"
        exit 1
    fi

    TEMP_LIB=$(mktemp)
    trap 'rm -f "$TEMP_LIB" 2>/dev/null' EXIT INT TERM

    if curl -fsSL "$LIB_BASE_URL/lib.sh" -o "$TEMP_LIB"; then
        . "$TEMP_LIB"
        return 0
    else
        echo "错误: 下载 lib.sh 失败"
        exit 1
    fi
}

load_library

# ============================================================
# 中文输出覆盖
# ============================================================

# 中文欢迎信息（用户版）
zh_print_welcome_user() {
    printf "\n"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "  ${BOLD}欢迎使用 Markitai 安装向导!${NC}\n"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "\n"
    printf "  本脚本将安装:\n"
    printf "    ${GREEN}•${NC} markitai - 支持 LLM 的 Markdown 转换器\n"
    printf "\n"
    printf "  可选组件:\n"
    printf "    ${YELLOW}•${NC} agent-browser - 浏览器自动化（JS 渲染页面）\n"
    printf "    ${YELLOW}•${NC} Claude Code CLI - 使用 Claude 订阅\n"
    printf "    ${YELLOW}•${NC} Copilot CLI - 使用 GitHub Copilot 订阅\n"
    printf "\n"
    printf "  ${BOLD}随时按 Ctrl+C 取消${NC}\n"
    printf "\n"
}

# 中文安装总结
zh_print_summary() {
    printf "\n"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "  ${BOLD}安装总结${NC}\n"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "\n"

    # 已安装
    if [ -n "$INSTALLED_COMPONENTS" ]; then
        printf "  ${GREEN}已安装:${NC}\n"
        echo "$INSTALLED_COMPONENTS" | tr '|' '\n' | while read -r comp; do
            [ -n "$comp" ] && printf "    ${GREEN}✓${NC} %s\n" "$comp"
        done
        printf "\n"
    fi

    # 已跳过
    if [ -n "$SKIPPED_COMPONENTS" ]; then
        printf "  ${YELLOW}已跳过:${NC}\n"
        echo "$SKIPPED_COMPONENTS" | tr '|' '\n' | while read -r comp; do
            [ -n "$comp" ] && printf "    ${YELLOW}○${NC} %s\n" "$comp"
        done
        printf "\n"
    fi

    # 安装失败
    if [ -n "$FAILED_COMPONENTS" ]; then
        printf "  ${RED}安装失败:${NC}\n"
        echo "$FAILED_COMPONENTS" | tr '|' '\n' | while read -r comp; do
            [ -n "$comp" ] && printf "    ${RED}✗${NC} %s\n" "$comp"
        done
        printf "\n"
    fi

    printf "  ${BOLD}文档:${NC} https://markitai.ynewtime.com\n"
    printf "  ${BOLD}问题反馈:${NC} https://github.com/Ynewtime/markitai/issues\n"
    printf "\n"
    return 0
}

# 覆盖库函数以使用中文输出
zh_print_header() {
    printf "\n"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "  ${BOLD}%s${NC}\n" "$1"
    printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n\n"
}

zh_warn_if_root() {
    if [ "$(id -u)" -eq 0 ]; then
        printf "\n"
        printf "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
        printf "  ${YELLOW}警告: 正在以 root 身份运行${NC}\n"
        printf "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
        printf "\n"
        printf "  以 root 身份运行安装脚本存在以下风险:\n"
        printf "  1. PATH 劫持: ~/.local/bin 可能被其他用户写入\n"
        printf "  2. 远程代码执行风险被放大\n"
        printf "\n"
        printf "  建议: 使用普通用户身份运行此脚本\n"
        printf "\n"

        if ! ask_yes_no "是否继续以 root 身份运行?" "n"; then
            printf "\n"
            print_info "退出。请使用普通用户身份运行。"
            exit 1
        fi
    fi
    return 0
}

zh_confirm_remote_script() {
    script_url="$1"
    script_name="$2"

    printf "\n"
    printf "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "  ${YELLOW}警告: 即将执行远程脚本${NC}\n"
    printf "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "\n"
    printf "  来源: %s\n" "$script_url"
    printf "  用途: 安装 %s\n" "$script_name"
    printf "\n"
    printf "  此操作将从互联网下载并执行代码。\n"
    printf "  请确保您信任该来源。\n"
    printf "\n"

    if ask_yes_no "确认执行?" "n"; then
        return 0
    else
        return 1
    fi
}

zh_detect_python() {
    for cmd in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            ver=$("$cmd" -c "import sys; v=sys.version_info; print('%d.%d.%d' % (v[0], v[1], v[2]))" 2>/dev/null)
            major=$("$cmd" -c "import sys; print(sys.version_info[0])" 2>/dev/null)
            minor=$("$cmd" -c "import sys; print(sys.version_info[1])" 2>/dev/null)

            case "$major" in
                ''|*[!0-9]*) continue ;;
            esac
            case "$minor" in
                ''|*[!0-9]*) continue ;;
            esac

            if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ] && [ "$minor" -le 13 ]; then
                PYTHON_CMD="$cmd"
                print_success "Python $ver 已安装 ($cmd)"
                return 0
            elif [ "$major" -eq 3 ] && [ "$minor" -ge 14 ]; then
                print_warning "Python $ver 检测到，但 onnxruntime 不支持 Python 3.14+"
            fi
        fi
    done

    print_error "未找到 Python 3.11-3.13"
    printf "\n"
    print_warning "请安装 Python 3.13 (推荐) 或 3.11/3.12:"
    print_info "官网下载: https://www.python.org/downloads/"
    print_info "Ubuntu/Debian: sudo apt install python3.13"
    print_info "macOS: brew install python@3.13"
    print_info "pyenv: pyenv install 3.13"
    print_info "提示: onnxruntime 暂不支持 Python 3.14"
    return 1
}

zh_install_uv() {
    print_info "检查 UV 安装..."

    if command -v uv >/dev/null 2>&1; then
        version=$(uv --version 2>/dev/null | head -n1)
        print_success "$version 已安装"
        track_install "uv" "installed"
        return 0
    fi

    print_error "UV 未安装"

    if ! ask_yes_no "是否自动安装 UV?" "n"; then
        print_info "跳过 UV 安装"
        print_warning "markitai 推荐使用 UV 进行安装"
        track_install "uv" "skipped"
        return 2
    fi

    if ! command -v curl >/dev/null 2>&1; then
        print_error "未找到 curl，无法下载 UV 安装脚本"
        print_info "请先安装 curl:"
        print_info "  Ubuntu/Debian: sudo apt install curl"
        print_info "  macOS: brew install curl"
        print_info "  或手动安装 UV: https://docs.astral.sh/uv/getting-started/installation/"
        return 1
    fi

    if [ -n "$UV_VERSION" ]; then
        uv_url="https://astral.sh/uv/$UV_VERSION/install.sh"
        print_info "安装 UV 版本: $UV_VERSION"
    else
        uv_url="https://astral.sh/uv/install.sh"
    fi

    if ! zh_confirm_remote_script "$uv_url" "UV"; then
        print_info "跳过 UV 安装"
        track_install "uv" "skipped"
        return 2
    fi

    print_info "正在安装 UV..."

    if curl -LsSf "$uv_url" | sh; then
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

        if command -v uv >/dev/null 2>&1; then
            version=$(uv --version 2>/dev/null | head -n1)
            print_success "$version 安装成功"
            track_install "uv" "installed"
            return 0
        else
            print_warning "UV 已安装，但需要重新加载 shell"
            print_info "请运行: source ~/.bashrc 或重新打开终端"
            print_info "然后重新运行此脚本"
            track_install "uv" "installed"
            return 1
        fi
    else
        print_error "UV 安装失败"
        print_info "手动安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
        track_install "uv" "failed"
        return 1
    fi
}

zh_install_markitai() {
    print_info "正在安装 markitai..."

    if [ -n "$MARKITAI_VERSION" ]; then
        pkg="markitai[all]==$MARKITAI_VERSION"
        print_info "安装版本: $MARKITAI_VERSION"
    else
        pkg="markitai[all]"
    fi

    if command -v uv >/dev/null 2>&1; then
        # 使用 --upgrade 确保安装最新版本
        if uv tool install "$pkg" --python "$PYTHON_CMD" --upgrade 2>/dev/null; then
            export PATH="$HOME/.local/bin:$PATH"
            version=$(markitai --version 2>/dev/null || echo "已安装")
            print_success "markitai $version 安装成功"
            print_info "已安装到 ~/.local/bin (使用 $PYTHON_CMD)"
            track_install "markitai" "installed"
            return 0
        fi
    fi

    if command -v pipx >/dev/null 2>&1; then
        # 使用 --force 确保安装最新版本
        if pipx install "$pkg" --python "$PYTHON_CMD" --force; then
            version=$(markitai --version 2>/dev/null || echo "已安装")
            print_success "markitai $version 安装成功"
            track_install "markitai" "installed"
            return 0
        fi
    fi

    # 使用 --upgrade 确保安装最新版本
    if "$PYTHON_CMD" -m pip install --user --upgrade "$pkg" 2>/dev/null; then
        export PATH="$HOME/.local/bin:$PATH"
        version=$(markitai --version 2>/dev/null || echo "已安装")
        print_success "markitai $version 安装成功"
        print_warning "可能需要将 ~/.local/bin 添加到 PATH"
        track_install "markitai" "installed"
        return 0
    fi

    print_error "markitai 安装失败"
    print_info "请手动安装: uv tool install markitai --python $PYTHON_CMD"
    track_install "markitai" "failed"
    return 1
}

zh_detect_node() {
    if ! command -v node >/dev/null 2>&1; then
        print_error "未找到 Node.js"
        printf "\n"
        print_warning "请安装 Node.js 18+:"
        print_info "官网下载: https://nodejs.org/"
        print_info "使用 nvm: curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash"
        print_info "使用 fnm: curl -fsSL https://fnm.vercel.app/install | bash"
        print_info "Ubuntu/Debian: sudo apt install nodejs npm"
        print_info "macOS: brew install node"
        return 1
    fi

    # Get version and strip CR (Windows CRLF compatibility)
    version=$(node --version 2>/dev/null | tr -d '\r')

    # Check if version is empty
    if [ -z "$version" ]; then
        print_warning "无法获取 Node 版本 (输出为空)"
        return 1
    fi

    # Extract major version number
    major=$(printf '%s' "$version" | sed 's/^v//' | cut -d. -f1)

    case "$major" in
        ''|*[!0-9]*)
            print_warning "无法解析 Node 版本: $version"
            return 1
            ;;
    esac

    if [ "$major" -ge 18 ]; then
        print_success "Node.js $version 已安装"
        return 0
    else
        print_warning "Node.js $version 版本较低，建议 18+"
        return 0
    fi
}

zh_install_agent_browser() {
    print_info "检测 Node.js..."

    if ! zh_detect_node; then
        print_warning "跳过 agent-browser 安装 (需要 Node.js)"
        track_install "agent-browser" "skipped"
        return 1
    fi

    print_info "正在安装 agent-browser..."

    if [ -n "$AGENT_BROWSER_VERSION" ]; then
        pkg="agent-browser@$AGENT_BROWSER_VERSION"
        print_info "安装版本: $AGENT_BROWSER_VERSION"
    else
        pkg="agent-browser"
    fi

    # 优先 npm，备选 pnpm
    install_success=false
    if command -v npm >/dev/null 2>&1; then
        print_info "通过 npm 安装..."
        if npm install -g "$pkg"; then
            install_success=true
        fi
    fi

    if [ "$install_success" = false ] && command -v pnpm >/dev/null 2>&1; then
        print_info "通过 pnpm 安装..."
        if pnpm add -g "$pkg"; then
            install_success=true
        fi
    fi

    if [ "$install_success" = true ]; then
        if ! command -v agent-browser >/dev/null 2>&1; then
            print_warning "agent-browser 已安装但不在 PATH 中"
            print_info "可能需要将全局 bin 目录添加到 PATH:"
            print_info "  pnpm bin -g  # 或: npm config get prefix"
            track_install "agent-browser" "installed"
            return 1
        fi

        print_success "agent-browser 安装成功"
        track_install "agent-browser" "installed"

        if ask_yes_no "是否下载 Chromium 浏览器?" "n"; then
            print_info "正在下载 Chromium..."

            os_type=$(uname -s)

            if [ "$os_type" = "Linux" ]; then
                if ask_yes_no "是否同时安装系统依赖 (需要 sudo)?" "n"; then
                    agent-browser install --with-deps
                else
                    agent-browser install
                fi
            else
                agent-browser install
            fi

            print_success "Chromium 下载完成"
            track_install "Chromium" "installed"
        else
            print_info "跳过 Chromium 下载"
            print_info "稍后可运行: agent-browser install"
            track_install "Chromium" "skipped"
        fi

        return 0
    else
        print_error "agent-browser 安装失败"
        print_info "请手动安装: npm install -g agent-browser"
        track_install "agent-browser" "failed"
        return 1
    fi
}

# 安装 Claude Code CLI
zh_install_claude_cli() {
    print_info "正在安装 Claude Code CLI..."

    # 检查是否已安装
    if command -v claude >/dev/null 2>&1; then
        version=$(claude --version 2>/dev/null | head -n1)
        print_success "Claude Code CLI 已安装: $version"
        track_install "Claude Code CLI" "installed"
        return 0
    fi

    # 优先使用 npm/pnpm
    if command -v pnpm >/dev/null 2>&1; then
        print_info "通过 pnpm 安装..."
        if pnpm add -g @anthropic-ai/claude-code; then
            print_success "Claude Code CLI 安装成功 (pnpm)"
            print_info "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    elif command -v npm >/dev/null 2>&1; then
        print_info "通过 npm 安装..."
        if npm install -g @anthropic-ai/claude-code; then
            print_success "Claude Code CLI 安装成功 (npm)"
            print_info "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    fi

    # 备选: Homebrew
    if command -v brew >/dev/null 2>&1; then
        print_info "通过 Homebrew 安装..."
        if brew install claude-code; then
            print_success "Claude Code CLI 安装成功 (Homebrew)"
            print_info "请运行 'claude /login' 使用 Claude 订阅或 API 密钥进行认证"
            track_install "Claude Code CLI" "installed"
            return 0
        fi
    fi

    print_warning "Claude Code CLI 安装失败"
    print_info "手动安装方式:"
    print_info "  pnpm: pnpm add -g @anthropic-ai/claude-code"
    print_info "  brew: brew install claude-code"
    print_info "  文档: https://code.claude.com/docs/en/setup"
    track_install "Claude Code CLI" "failed"
    return 1
}

# 安装 GitHub Copilot CLI
zh_install_copilot_cli() {
    print_info "正在安装 GitHub Copilot CLI..."

    # 检查是否已安装
    if command -v copilot >/dev/null 2>&1; then
        version=$(copilot --version 2>/dev/null | head -n1)
        print_success "Copilot CLI 已安装: $version"
        track_install "Copilot CLI" "installed"
        return 0
    fi

    # 优先使用 npm/pnpm
    if command -v pnpm >/dev/null 2>&1; then
        print_info "通过 pnpm 安装..."
        if pnpm add -g @github/copilot; then
            print_success "Copilot CLI 安装成功 (pnpm)"
            print_info "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    elif command -v npm >/dev/null 2>&1; then
        print_info "通过 npm 安装..."
        if npm install -g @github/copilot; then
            print_success "Copilot CLI 安装成功 (npm)"
            print_info "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    # 备选: Homebrew
    if command -v brew >/dev/null 2>&1; then
        print_info "通过 Homebrew 安装..."
        if brew install copilot-cli; then
            print_success "Copilot CLI 安装成功 (Homebrew)"
            print_info "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    # 备选: 安装脚本 (需要确认)
    copilot_url="https://gh.io/copilot-install"
    if zh_confirm_remote_script "$copilot_url" "GitHub Copilot CLI"; then
        print_info "尝试安装脚本..."
        if curl -fsSL "$copilot_url" | bash; then
            print_success "Copilot CLI 安装成功"
            print_info "请运行 'copilot /login' 使用 GitHub Copilot 订阅进行认证"
            track_install "Copilot CLI" "installed"
            return 0
        fi
    fi

    print_warning "Copilot CLI 安装失败"
    print_info "手动安装方式:"
    print_info "  pnpm: pnpm add -g @github/copilot"
    print_info "  brew: brew install copilot-cli"
    print_info "  curl: curl -fsSL https://gh.io/copilot-install | bash"
    track_install "Copilot CLI" "failed"
    return 1
}

zh_init_config() {
    print_info "初始化配置..."

    if command -v markitai >/dev/null 2>&1; then
        if markitai config init 2>/dev/null; then
            print_success "配置初始化完成"
        fi
    fi
    return 0
}

zh_print_completion() {
    printf "\n"
    printf "${GREEN}✓${NC} ${BOLD}配置完成!${NC}\n"
    printf "\n"
    printf "  ${BOLD}开始使用:${NC}\n"
    printf "    ${YELLOW}markitai --help${NC}\n"
    printf "\n"
}

# ============================================================
# 主逻辑
# ============================================================

main() {
    # 安全检查: root 警告
    zh_warn_if_root

    # 欢迎信息
    zh_print_welcome_user

    zh_print_header "Markitai 环境配置向导"

    # 步骤 1: 检测 Python
    print_step 1 5 "检测 Python..."
    if ! zh_detect_python; then
        exit 1
    fi

    # 步骤 2: 检测/安装 UV (用户版可选)
    print_step 2 5 "检测 UV 包管理器..."
    zh_install_uv || true
    # 用户版: UV 是可选的，跳过/失败都继续

    # 步骤 3: 安装 markitai
    print_step 3 5 "安装 markitai..."
    if ! zh_install_markitai; then
        zh_print_summary
        exit 1
    fi

    # 步骤 4: 可选 - agent-browser
    print_step 4 5 "可选: 浏览器自动化"
    if ask_yes_no "是否安装浏览器自动化支持 (agent-browser)?" "n"; then
        zh_install_agent_browser
    else
        print_info "跳过 agent-browser 安装"
        track_install "agent-browser" "skipped"
    fi

    # 步骤 5: 可选 - LLM CLI 工具
    print_step 5 5 "可选: LLM CLI 工具"
    print_info "LLM CLI 工具为 AI 提供商提供本地认证"
    if ask_yes_no "是否安装 Claude Code CLI?" "n"; then
        zh_install_claude_cli
    else
        track_install "Claude Code CLI" "skipped"
    fi
    if ask_yes_no "是否安装 GitHub Copilot CLI?" "n"; then
        zh_install_copilot_cli
    else
        track_install "Copilot CLI" "skipped"
    fi

    # 初始化配置
    zh_init_config

    # 打印总结
    zh_print_summary

    # 完成
    zh_print_completion
}

# 运行主函数
main
